"""
Unit tests for the Stage 4 window-planning algorithm.

Synthetic spectra with known lines exercise the plan's invariants and the four
reference cases from the plan's test matrix: an isolated strong line, two
strong lines with overlapping reach (one primary joint window), a weak line on
a strong line's far skirt (a fixed contributor + a dependency edge), and
leakage-artifact pruning of the free set.
"""

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    MergeRequest,
    Peak,
    PeakClassification,
    WindowDifficulty,
)
from ftmwpipeline.preprocessing.window_planning import build_window_plan, replan


def _h_t(df_hz, t_s):
    """Finite-T (undamped) leakage kernel; |h(0)| = T."""
    s = 1j * 2.0 * np.pi * np.asarray(df_hz, dtype=float)
    safe = np.where(s == 0, 1.0 + 0j, s)
    h = (1.0 - np.exp(-safe * t_s)) / safe
    return np.where(s == 0, t_s + 0j, h)


def _synthetic(lines, n=4000, f_lo=30000.0, step=0.02, t_us=15.0, sigma=0.01, seed=0):
    """Build a synthetic complex spectrum + peak list.

    ``lines`` is a list of ``(frequency_mhz, intensity, classification)``.
    Every line is added to the complex spectrum as a finite-T leakage kernel
    scaled so its apex magnitude equals ``intensity``; a matching promoted
    :class:`Peak` is returned for each.
    """
    freqs = f_lo + np.arange(n) * step
    t_s = t_us * 1e-6
    spec = np.zeros(n, dtype=complex)
    for f0, intensity, _cls in lines:
        df_hz = (freqs - f0) * 1e6
        spec += intensity * _h_t(df_hz, t_s) / t_s
    rng = np.random.default_rng(seed)
    spec += rng.normal(0, sigma / np.sqrt(2), n) + 1j * rng.normal(
        0, sigma / np.sqrt(2), n
    )
    rms = np.full(n, sigma)

    peaks = []
    for f0, intensity, cls in lines:
        gi = int(round((f0 - f_lo) / step))
        peaks.append(
            Peak(
                frequency=f0,
                intensity=intensity,
                index=gi,
                snr=intensity / sigma,
                noise_std_local=sigma,
                classification=cls,
                promoted=True,
            )
        )
    return freqs, spec, rms, peaks


def _assert_invariants(plan):
    """Disjoint windows, unique free peaks, acyclic DAG, valid topo order."""
    ws = sorted(plan.windows, key=lambda w: w.freq_range[0])
    for a, b in zip(ws, ws[1:]):
        assert a.freq_range[1] <= b.freq_range[0], "windows overlap"
    free = [li for w in plan.windows for li in w.free_peak_indices]
    assert len(free) == len(set(free)), "a peak is free in more than one window"
    ids = sorted(w.window_id for w in plan.windows)
    assert sorted(plan.topological_order) == ids, "topo order misses windows"
    pos = {w: i for i, w in enumerate(plan.topological_order)}
    for w, dep in plan.dependency_edges:
        assert pos[dep] < pos[w], "dependency edge violates topological order"


class TestIsolatedStrongLine:
    def test_single_window_hard(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 1
        w = plan.windows[0]
        assert w.free_peak_indices == [0]
        assert w.difficulty == WindowDifficulty.HARD
        assert not w.fixed_contributors
        assert not plan.dependency_edges
        assert w.freq_range[0] <= 30040.0 <= w.freq_range[1]
        _assert_invariants(plan)

    def test_deterministic(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        p1 = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        p2 = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert p1.n_windows == p2.n_windows
        assert [w.freq_range for w in p1.windows] == [w.freq_range for w in p2.windows]


class TestStrongCluster:
    def test_overlapping_strong_lines_form_one_joint_window(self):
        """Two strong lines whose skirts overlap merge into one window."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30038.0, 3.0, PeakClassification.STRONG),
                (30042.0, 3.0, PeakClassification.STRONG),
            ]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 1
        w = plan.windows[0]
        assert sorted(w.free_peak_indices) == [0, 1]
        assert w.difficulty == WindowDifficulty.HARD
        _assert_invariants(plan)


class TestWeakLineOnSkirt:
    def test_weak_window_gets_strong_fixed_contributor(self):
        """A weak line on a strong line's far skirt becomes a separate window
        with the strong line attached as a fixed contributor."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 6.0, PeakClassification.STRONG),
                (30056.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=8000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 2
        _assert_invariants(plan)

        strong_w = next(w for w in plan.windows if 0 in w.free_peak_indices)
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        assert strong_w.window_id != weak_w.window_id

        # The weak window carries the strong line as a fixed contributor.
        assert len(weak_w.fixed_contributors) == 1
        fc = weak_w.fixed_contributors[0]
        assert fc.peak_index == 0
        assert fc.primary_window_id == strong_w.window_id
        assert fc.freeze_eligible is True  # SNR 600 >= default min_freeze_snr

        # ... and a dependency edge + batch ordering follows.
        assert (weak_w.window_id, strong_w.window_id) in plan.dependency_edges
        assert weak_w.batch > strong_w.batch
        assert weak_w.difficulty == WindowDifficulty.HARD

    def test_distant_weak_line_is_independent(self):
        """A weak line beyond the analytic skirt-magnitude threshold of any
        strong line is an easy, independent window with no fixed
        contributor. With Tier-1 magnitude-based attachment, "independent"
        means the strong line's predicted mean |skirt| on this window's
        grid sits below ``threshold * sigma_c`` -- not that the window
        sits outside any rolling-coherence-touched region."""
        # Strong intensity 0.3 at 140 MHz separation predicts mean skirt
        # ~4.5e-5 = 0.006 * sigma_c, well below the 0.1 sigma_c threshold.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30010.0, 0.3, PeakClassification.STRONG),
                (30150.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=12000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        assert not weak_w.fixed_contributors
        assert weak_w.difficulty == WindowDifficulty.EASY
        assert weak_w.batch == 0
        _assert_invariants(plan)


class TestMagnitudeAttachment:
    """Tier-1 contributor-attachment rule (O5-10 fix).

    A strong promoted peak is attached as a :class:`FixedContributor` of a
    candidate window when its predicted mean |skirt| on that window's grid
    exceeds ``magnitude_attachment_threshold * sigma_c(w)``. The previous
    rule (a strong line's rolling-coherence-touched run had to reach the
    window) missed the cumulative tail of many far-line skirts -- the bias
    mechanism diagnosed in scratch/stage5-validation/.
    """

    def test_strong_far_skirt_above_threshold_is_attached(self):
        """A strong line 140 MHz from a weak window predicts a mean |skirt|
        of ~9e-4 (= 13 sigma_c at sigma=0.01), well above the 0.1 sigma_c
        threshold. The magnitude rule attaches it; the touched-region rule
        previously missed it."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30010.0, 6.0, PeakClassification.STRONG),
                (30150.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=12000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        strong_w = next(w for w in plan.windows if 0 in w.free_peak_indices)
        assert len(weak_w.fixed_contributors) == 1
        fc = weak_w.fixed_contributors[0]
        assert fc.peak_index == 0
        assert fc.primary_window_id == strong_w.window_id
        assert (weak_w.window_id, strong_w.window_id) in plan.dependency_edges
        _assert_invariants(plan)

    def test_threshold_respected(self):
        """Raising ``magnitude_attachment_threshold`` drops borderline
        contributors; lowering it adds them. Same physical layout."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30010.0, 6.0, PeakClassification.STRONG),
                (30150.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=12000,
        )
        plan_low = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            magnitude_attachment_threshold=0.01,  # very permissive
        )
        plan_high = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            magnitude_attachment_threshold=10.0,  # very strict
        )
        weak_low = next(w for w in plan_low.windows if 1 in w.free_peak_indices)
        weak_high = next(w for w in plan_high.windows if 1 in w.free_peak_indices)
        assert len(weak_low.fixed_contributors) == 1
        assert len(weak_high.fixed_contributors) == 0
        _assert_invariants(plan_low)
        _assert_invariants(plan_high)

    def test_threshold_persisted_in_plan_parameters(self):
        """The configured threshold lands in ``plan.parameters`` so a
        downstream ``replan`` reproduces the same attachment behaviour."""
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            magnitude_attachment_threshold=0.42,
        )
        assert plan.parameters["magnitude_attachment_threshold"] == pytest.approx(0.42)

    def test_mutual_attachment_does_not_break_dag(self):
        """Two strong lines whose skirts mutually reach each other form a
        2-cycle in the attachment graph. The cycle-breaker drops both
        fit-ordering edges from ``dependency_edges`` to keep the DAG acyclic,
        but the orphaned contributors are *not* discarded: each is converted
        to an EDGE-FREE contributor (read self-contained at fit time), so the
        leakage subtraction survives without a dependency edge. The
        execution-time invariant ("primary fit must exist before an
        edge-bearing dependent fits") still holds because an edge-free
        contributor carries no such requirement."""
        # Two strong lines at 30050 and 30090 -- 40 MHz apart, each strong
        # enough that the other's skirt clears the 0.1 sigma_c threshold.
        # They don't fall in the same touched region (clean band between
        # them) so they end up in separate windows under the rule.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 8.0, PeakClassification.STRONG),
                (30090.0, 8.0, PeakClassification.STRONG),
            ],
            n=8000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        _assert_invariants(plan)
        # Edge consistency now depends on the contributor kind: an edge-bearing
        # contributor's (window, primary) edge must be present; an edge-free
        # contributor's must be absent (it is deliberately out of the DAG).
        for w in plan.windows:
            for fc in w.fixed_contributors:
                edge = (w.window_id, fc.primary_window_id)
                if fc.edge_free:
                    assert edge not in plan.dependency_edges, (
                        f"edge-free contributor on window {w.window_id} must "
                        f"not appear in dependency_edges, found {edge}"
                    )
                else:
                    assert edge in plan.dependency_edges, (
                        f"edge-bearing contributor on window {w.window_id} "
                        f"points at primary {fc.primary_window_id} but that "
                        f"edge was dropped from dependency_edges"
                    )
        # The 2-cycle's orphaned contributors are recovered as edge-free, not
        # discarded -- this is the issue-#3 leakage-subtraction fix.
        edge_free = [
            fc for w in plan.windows for fc in w.fixed_contributors if fc.edge_free
        ]
        assert edge_free, "mutual-attachment 2-cycle should yield edge-free contributors"
        assert plan.diagnostics.get("n_edge_free_contributors", 0) == len(edge_free)


class TestLeakageArtifactPruning:
    def test_sidelobe_pruned_genuine_line_kept(self):
        """A detection below a strong line's analytic skirt is pruned from the
        free set; a genuine line poking above the skirt is retained."""
        # Strong line + a sub-skirt artifact + a genuine line above the skirt.
        lines = [
            (30040.0, 2.0, PeakClassification.STRONG),
            (30040.8, 0.03, PeakClassification.WEAK),  # below envelope
            (30041.0, 0.20, PeakClassification.MEDIUM),  # above envelope
        ]
        freqs, spec, rms, peaks = _synthetic(lines)
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)

        free = {li for w in plan.windows for li in w.free_peak_indices}
        assert 0 in free, "strong line must remain free"
        assert 1 not in free, "sub-skirt artifact must be pruned"
        assert 2 in free, "genuine line above the skirt must be retained"
        assert plan.diagnostics.get("n_pruned_leakage_artifacts") == 1
        _assert_invariants(plan)


class TestDifficultyAndWidthCap:
    def test_width_cap_flags_too_wide_window(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_mhz=2.0,
        )
        w = plan.windows[0]
        assert w.difficulty == WindowDifficulty.HARD
        assert w.diagnostics["width_cap_hit"] is True
        # Too wide -> either a split proposal or a joint-treatment marker.
        assert (w.split_proposal is not None) or w.needs_joint_treatment


class TestBoundedMergeAndCapSplit:
    """The strong-cluster merge is bounded at the width cap and merged spans are
    split at their sparsest gaps until each window is <= max_peaks_per_window and
    <= max_window_width_mhz, so a dense, mutually-coupled strong-line forest does
    not collapse into one unfittable mega-window."""

    @staticmethod
    def _dense_cluster():
        # 20 strong lines 3 MHz apart over ~57 MHz: the per-peak proto-spans
        # (+/- 2 MHz) overlap and the skirts keep S_coh lit between them, so the
        # legacy build would chain them into one ~61 MHz / 20-peak window.
        lines = [(30040.0 + 3.0 * i, 3.0, PeakClassification.STRONG) for i in range(20)]
        return _synthetic(lines, n=8000)

    def test_dense_strong_forest_is_split_to_caps(self):
        freqs, spec, rms, peaks = self._dense_cluster()
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        _assert_invariants(plan)
        # Must NOT be one mega-window.
        assert plan.n_windows >= 3
        for w in plan.windows:
            assert w.width_mhz <= 40.0 + 1e-6, "window exceeds the width cap"
            assert len(w.free_peak_indices) <= 8, "window exceeds the peak cap"
        # Every promoted line is still covered exactly once (no dropped peaks).
        covered = sorted(li for w in plan.windows for li in w.free_peak_indices)
        assert covered == list(range(len(peaks)))

    def test_tighter_peak_cap_makes_more_windows(self):
        freqs, spec, rms, peaks = self._dense_cluster()
        loose = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        tight = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_peaks_per_window=4
        )
        assert tight.n_windows > loose.n_windows
        for w in tight.windows:
            assert len(w.free_peak_indices) <= 4
        _assert_invariants(tight)

    def test_coupled_pair_within_cap_stays_merged(self):
        # Two strong lines a few MHz apart (< the width cap, < the peak cap) must
        # still merge into one joint window -- the cap split must not break a
        # genuinely-coupled close pair (the 2638 doublet back-compat case).
        freqs, spec, rms, peaks = _synthetic(
            [
                (30038.0, 3.0, PeakClassification.STRONG),
                (30042.0, 3.0, PeakClassification.STRONG),
            ]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 1
        assert sorted(plan.windows[0].free_peak_indices) == [0, 1]

    def test_max_peaks_per_window_recorded_in_parameters(self):
        freqs, spec, rms, peaks = self._dense_cluster()
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_peaks_per_window=6
        )
        assert plan.parameters["max_peaks_per_window"] == 6


class TestEmptyAndEdgeCases:
    def test_no_promoted_peaks_gives_empty_plan(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        for p in peaks:
            p.properties["promoted"] = False
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 0
        assert plan.dependency_edges == []

    def test_empty_peak_list(self):
        freqs, spec, rms, _ = _synthetic([(30040.0, 2.0, PeakClassification.STRONG)])
        plan = build_window_plan([], freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 0

    def test_bad_inputs_raise(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        with pytest.raises(ValueError):
            build_window_plan(peaks, freqs, spec, rms[:-1], acquisition_us=15.0)
        with pytest.raises(ValueError):
            build_window_plan(peaks, freqs, spec, rms, acquisition_us=0.0)


class TestDeRamp:
    """The full-record-rfft turn-on ramp and the de-ramp that undoes it."""

    def test_deramp_recovers_plan_from_ramped_spectrum(self):
        lines = [
            (30038.0, 3.0, PeakClassification.STRONG),
            (30042.0, 3.0, PeakClassification.STRONG),
        ]
        freqs, spec, rms, peaks = _synthetic(lines)
        probe_mhz = 30000.0
        t0_us = 3.0
        ramp = np.exp(-2j * np.pi * (freqs - probe_mhz) * 1e6 * (t0_us * 1e-6))
        ramped = spec * ramp

        ref = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        fixed = build_window_plan(
            peaks,
            freqs,
            ramped,
            rms,
            acquisition_us=15.0,
            probe_freq_mhz=probe_mhz,
            start_us=t0_us,
        )
        broken = build_window_plan(peaks, freqs, ramped, rms, acquisition_us=15.0)

        # Matching start_us de-ramps back to the reference spectrum exactly.
        assert fixed.n_windows == ref.n_windows
        assert [w.freq_range for w in fixed.windows] == [
            w.freq_range for w in ref.windows
        ]
        ref_stat = max(w.diagnostics["edge_coherence_statistic"] for w in ref.windows)
        fixed_stat = max(
            w.diagnostics["edge_coherence_statistic"] for w in fixed.windows
        )
        broken_stat = max(
            w.diagnostics["edge_coherence_statistic"] for w in broken.windows
        )
        assert fixed_stat == pytest.approx(ref_stat, rel=1e-6)
        # Leaving the ramp in place collapses the coherent edge statistic.
        assert broken_stat < 0.5 * fixed_stat
        _assert_invariants(fixed)

    def test_start_us_recorded_in_parameters(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            probe_freq_mhz=30000.0,
            start_us=2.35,
        )
        assert plan.parameters["start_us"] == 2.35
        assert plan.parameters["probe_freq_mhz"] == 30000.0


# ---------------------------------------------------------------------------
# Structural renegotiation: replan() with MergeRequests
# ---------------------------------------------------------------------------
class TestReplanMerge:
    """``replan`` applies :class:`MergeRequest`s to an existing
    :class:`WindowPlan`, re-runs Stage 4's bookkeeping tail (contributor /
    dependency / difficulty / batch recomputation, artifact pruning) against
    the merged window list, and bumps :attr:`WindowPlan.plan_revision`. The
    disjoint-coverage invariant is preserved.
    """

    def _two_window_plan(self):
        """Two well-separated EASY weak windows -- a clean merge fixture."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30030.0, 0.05, PeakClassification.WEAK),
                (30060.0, 0.05, PeakClassification.WEAK),
            ]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        return plan, peaks, freqs, spec, rms

    def test_empty_request_list_bumps_revision_only(self):
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        original_n = plan.n_windows
        revised = replan(plan, [], peaks, freqs, spec, rms, acquisition_us=15.0)
        assert revised.plan_revision == plan.plan_revision + 1
        assert revised.n_windows == original_n
        _assert_invariants(revised)

    def test_merges_two_adjacent_windows(self):
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        assert plan.n_windows == 2
        a, b = plan.windows[0], plan.windows[1]
        survivor_id = min(a.window_id, b.window_id)

        revised = replan(
            plan,
            [MergeRequest(a.window_id, b.window_id, reason="test merge")],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        assert revised.n_windows == 1
        merged = revised.windows[0]
        assert merged.window_id == survivor_id
        assert merged.freq_range[0] == pytest.approx(a.freq_range[0])
        assert merged.freq_range[1] == pytest.approx(b.freq_range[1])
        # Union of free peaks.
        assert sorted(merged.free_peak_indices) == sorted(
            list(a.free_peak_indices) + list(b.free_peak_indices)
        )
        # Audit fields recorded.
        assert merged.diagnostics["merged_from"] == sorted([a.window_id, b.window_id])
        assert merged.diagnostics["merge_reason"] == "test merge"
        _assert_invariants(revised)

    def test_non_adjacent_merge_raises(self):
        """A merge of windows with another window between them is rejected."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30020.0, 0.05, PeakClassification.WEAK),
                (30050.0, 0.05, PeakClassification.WEAK),
                (30080.0, 0.05, PeakClassification.WEAK),
            ]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 3
        w0, _w1, w2 = sorted(plan.windows, key=lambda w: w.freq_range[0])
        with pytest.raises(ValueError, match="not adjacent"):
            replan(
                plan,
                [MergeRequest(w0.window_id, w2.window_id)],
                peaks,
                freqs,
                spec,
                rms,
                acquisition_us=15.0,
            )

    def test_unknown_window_id_raises(self):
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        with pytest.raises(ValueError, match="unknown window"):
            replan(
                plan,
                [MergeRequest(99, plan.windows[0].window_id)],
                peaks,
                freqs,
                spec,
                rms,
                acquisition_us=15.0,
            )

    def test_self_merge_raises(self):
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        wid = plan.windows[0].window_id
        with pytest.raises(ValueError, match="distinct windows"):
            replan(
                plan,
                [MergeRequest(wid, wid)],
                peaks,
                freqs,
                spec,
                rms,
                acquisition_us=15.0,
            )

    def test_merge_drops_now_internal_contributor(self):
        """A merge that swallows a primary makes its contributor internal.

        Setup (same fixture shape as ``TestWeakLineOnSkirt``): a strong line
        and a weak line on its far skirt. Stage 4 builds a 2-window plan
        with the strong line attached to the weak window as a fixed
        contributor + a dependency edge. Merge the two windows: the merged
        window now contains the strong line as a free peak, so its earlier
        role as a fixed contributor to the (now-merged) weak window is
        no longer needed and the contributor list comes back empty.
        """
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 6.0, PeakClassification.STRONG),
                (30056.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=8000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        # Sanity: there should be two windows with a dep edge.
        assert plan.n_windows == 2
        assert plan.dependency_edges, "expected a dependency edge in setup"
        a, b = sorted(plan.windows, key=lambda w: w.freq_range[0])

        revised = replan(
            plan,
            [MergeRequest(a.window_id, b.window_id)],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        assert revised.n_windows == 1
        merged = revised.windows[0]
        # No contributors: the strong line is in-band of the merged window.
        assert merged.fixed_contributors == []
        # And no dependency edges remain.
        assert revised.dependency_edges == []
        _assert_invariants(revised)

    def test_topological_order_recomputed(self):
        """A 3-window plan with a chain dep: merge the outer two. The merged
        window subsumes the chain endpoint, so the surviving edge re-ranks the
        topological order against the new id set.
        """
        # Three lines: the leftmost strong, the other two weak on its skirt.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30030.0, 2.5, PeakClassification.STRONG),
                (30055.0, 0.06, PeakClassification.WEAK),
                (30075.0, 0.06, PeakClassification.WEAK),
            ]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        if plan.n_windows < 3:
            pytest.skip("setup did not produce 3 windows for this fixture")
        ws = sorted(plan.windows, key=lambda w: w.freq_range[0])
        # Merge the two adjacent weak windows.
        revised = replan(
            plan,
            [MergeRequest(ws[1].window_id, ws[2].window_id)],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        ids = {w.window_id for w in revised.windows}
        assert sorted(revised.topological_order) == sorted(ids)
        # And the topological order is consistent with the surviving deps.
        pos = {wid: i for i, wid in enumerate(revised.topological_order)}
        for child, parent in revised.dependency_edges:
            assert pos[parent] < pos[child]
        _assert_invariants(revised)

    def test_revision_counter_chains_across_replans(self):
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        once = replan(plan, [], peaks, freqs, spec, rms, acquisition_us=15.0)
        twice = replan(once, [], peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.plan_revision == 0
        assert once.plan_revision == 1
        assert twice.plan_revision == 2

    def test_original_plan_unmutated(self):
        """replan must not mutate the input plan in place."""
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        snapshot_n = plan.n_windows
        snapshot_revision = plan.plan_revision
        snapshot_window_ids = sorted(w.window_id for w in plan.windows)

        revised = replan(
            plan,
            [MergeRequest(plan.windows[0].window_id, plan.windows[1].window_id)],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        assert plan.n_windows == snapshot_n
        assert plan.plan_revision == snapshot_revision
        assert sorted(w.window_id for w in plan.windows) == snapshot_window_ids
        assert revised.n_windows < plan.n_windows  # merge succeeded
