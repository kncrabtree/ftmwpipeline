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
    FitWindow,
    MergeRequest,
    Peak,
    PeakClassification,
    WindowPlan,
)
from ftmwpipeline.preprocessing.window_planning import (
    DEFAULT_MAX_PEAKS_PER_WINDOW,
    DEFAULT_MAX_WINDOW_WIDTH_MHZ,
    DEFAULT_MAX_WINDOW_WIDTH_POINTS,
    build_window_plan,
    merge_geometry,
    merged_window_ids,
    next_window_id,
    replan,
    retired_window_ids,
)


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


def _content_mhz(window, peaks):
    """Span between the first and last promoted peak owned by ``window`` -- the
    quantity the width cap bounds (not the padded ``freq_range``, which carries
    the per-peak proto-margins)."""
    fs = [peaks[li].frequency for li in window.free_peak_indices]
    return (max(fs) - min(fs)) if len(fs) >= 2 else 0.0


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
    def test_single_window(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert plan.n_windows == 1
        w = plan.windows[0]
        assert w.free_peak_indices == [0]
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
        """Two strong lines whose skirts overlap merge into one window.

        The lines are 1 MHz apart -- within 2 * the window margin and close
        enough that their coherent skirts keep S_coh above threshold between
        them (one shared leakage-touched region), so both the proto-overlap and
        the strong-cluster merge bind them into one joint window.
        """
        freqs, spec, rms, peaks = _synthetic(
            [
                (30038.0, 3.0, PeakClassification.STRONG),
                (30039.0, 3.0, PeakClassification.STRONG),
            ]
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_window_width_points=0
        )
        assert plan.n_windows == 1
        w = plan.windows[0]
        assert sorted(w.free_peak_indices) == [0, 1]
        _assert_invariants(plan)


class TestWeakLineOnSkirt:
    def test_weak_window_gets_strong_fixed_contributor(self):
        """A weak line on a strong line's skirt becomes a separate window with
        the strong line attached as an edge-bearing fixed contributor when the
        skirt is material. The ``skirt_level_keep`` bar is set low so the
        attached skirt clears the Step-7 materiality gate (the gate itself is
        exercised in ``test_skirt_level_gate``)."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 6.0, PeakClassification.STRONG),
                (30056.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=8000,
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1.0
        )
        assert plan.n_windows == 2
        _assert_invariants(plan)

        strong_w = next(w for w in plan.windows if 0 in w.free_peak_indices)
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        assert strong_w.window_id != weak_w.window_id

        # The weak window carries the strong line as an edge-bearing contributor.
        assert len(weak_w.fixed_contributors) == 1
        fc = weak_w.fixed_contributors[0]
        assert fc.peak_index == 0
        assert fc.primary_window_id == strong_w.window_id
        assert fc.freeze_eligible is True  # SNR 600 >= default min_freeze_snr
        assert fc.edge_free is False  # the new path never demotes to edge-free

        # ... and a dependency edge + batch ordering follows (oriented strong->weak).
        assert (weak_w.window_id, strong_w.window_id) in plan.dependency_edges
        assert weak_w.batch > strong_w.batch

    def test_distant_weak_line_is_independent(self):
        """A weak line beyond the analytic skirt-magnitude threshold of any
        strong line is an independent window with no fixed contributor. With
        Tier-1 magnitude-based attachment, "independent" means the strong
        line's predicted mean |skirt| on this window's grid sits below
        ``threshold * sigma_c`` -- not that the window sits outside any
        rolling-coherence-touched region."""
        # Strong intensity 0.3 at 140 MHz separation predicts mean skirt
        # ~4.5e-5 = 0.006 * sigma_c, well below the 0.1 sigma_c threshold.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30010.0, 0.3, PeakClassification.STRONG),
                (30150.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=12000,
        )
        # Points cap pinned off (the default 96-point cap would split this
        # fine synthetic grid).
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_window_width_points=0
        )
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        assert not weak_w.fixed_contributors
        assert weak_w.batch == 0
        _assert_invariants(plan)


class TestMagnitudeAttachment:
    """Tier-1 contributor-attachment rule (O5-10 fix).

    A strong promoted peak is attached as a :class:`FixedContributor` of a
    candidate window when its predicted mean |skirt| on that window's grid
    exceeds ``magnitude_attachment_threshold * sigma_c(w)``. The previous
    rule (a strong line's rolling-coherence-touched run had to reach the
    window) missed the cumulative tail of many far-line skirts -- the bias
    mechanism this attachment rule fixes.
    """

    def test_strong_far_skirt_attached_and_gate_kept(self):
        """A strong line 140 MHz from a weak window predicts a mean |skirt|
        of ~9e-4 (= 13 sigma_c at sigma=0.01), well above the 0.1 sigma_c
        attachment threshold, so Step 4 attaches it. With ``skirt_level_keep``
        low the Step-7 materiality gate keeps it edge-bearing."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30010.0, 6.0, PeakClassification.STRONG),
                (30150.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=12000,
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1.0
        )
        weak_w = next(w for w in plan.windows if 1 in w.free_peak_indices)
        strong_w = next(w for w in plan.windows if 0 in w.free_peak_indices)
        assert len(weak_w.fixed_contributors) == 1
        fc = weak_w.fixed_contributors[0]
        assert fc.peak_index == 0
        assert fc.primary_window_id == strong_w.window_id
        assert fc.edge_free is False
        assert (weak_w.window_id, strong_w.window_id) in plan.dependency_edges
        _assert_invariants(plan)

    def test_threshold_respected(self):
        """Raising ``magnitude_attachment_threshold`` drops borderline
        contributors at Step 4; lowering it adds them. ``skirt_level_keep`` is
        held low so the Step-7 gate does not also remove the attached skirt,
        isolating the attachment threshold."""
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
            skirt_level_keep=1.0,
        )
        plan_high = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            magnitude_attachment_threshold=10.0,  # very strict
            skirt_level_keep=1.0,
        )
        weak_low = next(w for w in plan_low.windows if 1 in w.free_peak_indices)
        weak_high = next(w for w in plan_high.windows if 1 in w.free_peak_indices)
        assert len(weak_low.fixed_contributors) == 1
        assert len(weak_high.fixed_contributors) == 0
        _assert_invariants(plan_low)
        _assert_invariants(plan_high)

    def test_skirt_level_gate(self):
        """The Step-7 ``skirt_level_keep`` gate decides whether an *attached*
        downward skirt survives edge-bearing or is dropped to the dependent's
        baseline. Same layout, two bars: a low bar keeps the contributor (with a
        dependency edge); an unreachably high bar drops it (no contributor, no
        edge), the skirt left to the baseline polynomial."""
        layout = [
            (30050.0, 6.0, PeakClassification.STRONG),
            (30056.0, 0.12, PeakClassification.MEDIUM),
        ]
        freqs, spec, rms, peaks = _synthetic(layout, n=8000)
        kept = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1.0
        )
        freqs, spec, rms, peaks = _synthetic(layout, n=8000)
        dropped = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1e12
        )
        weak_kept = next(w for w in kept.windows if 1 in w.free_peak_indices)
        weak_dropped = next(w for w in dropped.windows if 1 in w.free_peak_indices)
        assert len(weak_kept.fixed_contributors) == 1
        assert kept.dependency_edges
        assert weak_dropped.fixed_contributors == []
        assert dropped.dependency_edges == []
        _assert_invariants(kept)
        _assert_invariants(dropped)

    def test_thresholds_persisted_in_plan_parameters(self):
        """The Step-7 gate thresholds land in ``plan.parameters`` so a
        downstream ``replan`` reproduces the same gating."""
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            skirt_level_keep=123.0,
            curvature_keep_sigma=7.0,
        )
        assert plan.parameters["skirt_level_keep"] == pytest.approx(123.0)
        assert plan.parameters["curvature_keep_sigma"] == pytest.approx(7.0)

    def test_threshold_persisted_in_plan_parameters(self):
        """The configured threshold lands in ``plan.parameters`` so a
        downstream ``replan`` reproduces the same attachment behavior."""
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

    def test_mutual_attachment_oriented_acyclic(self):
        """Two strong lines whose skirts mutually reach each other attach to one
        another at Step 4. Step 7 orients the pair by strength (ties broken by
        window id) into a single strong->weak edge rather than a 2-cycle: the
        weaker window keeps the stronger as an edge-bearing contributor, the
        stronger drops its reverse arc, no edge-free contributor is produced, and
        the graph is acyclic by construction. ``skirt_level_keep`` is low so the
        surviving arc clears the materiality gate."""
        # Two equal-intensity strong lines 40 MHz apart in separate windows.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 8.0, PeakClassification.STRONG),
                (30090.0, 8.0, PeakClassification.STRONG),
            ],
            n=8000,
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1.0
        )
        _assert_invariants(plan)

        # No contributor is ever demoted to edge-free in the oriented path.
        assert not [
            fc for w in plan.windows for fc in w.fixed_contributors if fc.edge_free
        ]
        # Every surviving contributor is edge-bearing and carries its edge.
        for w in plan.windows:
            for fc in w.fixed_contributors:
                assert (w.window_id, fc.primary_window_id) in plan.dependency_edges
        # The mutual pair collapses to a single oriented edge (not a 2-cycle):
        # the lower-strength window depends on the higher, never both ways.
        wa, wb = sorted(plan.windows, key=lambda w: w.window_id)
        assert (wa.window_id, wb.window_id) in plan.dependency_edges
        assert (wb.window_id, wa.window_id) not in plan.dependency_edges
        # The stronger (higher-id, tie-broken) window has no contributor.
        assert wb.fixed_contributors == []


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


class TestWidthCap:
    def test_over_cap_cluster_is_split(self):
        # The width cap is enforced structurally by the cap split (on peak
        # content), so an over-cap cluster is broken into within-cap windows
        # rather than left whole (the F2 fix: never re-bisect a content-fitting
        # window, but always split a genuinely over-content cluster).
        lines = [
            (30040.0 + 0.5 * i, 3.0, PeakClassification.STRONG) for i in range(8)
        ]  # 8 strong lines 0.5 MHz apart -> ~3.5 MHz of coupled content
        freqs, spec, rms, peaks = _synthetic(lines, n=8000)
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_mhz=2.0,
            max_window_width_points=0,
        )
        assert plan.n_windows >= 2, "over-cap cluster must be split"
        for w in plan.windows:
            assert _content_mhz(w, peaks) <= 2.0 + 1e-6
        _assert_invariants(plan)


class TestBoundedMergeAndCapSplit:
    """The strong-cluster merge is bounded at the width cap and merged spans are
    split at their sparsest gaps until each window's *peak content* is <=
    max_window_width_mhz (and, when ``max_peaks_per_window`` is positive, <= that
    peak cap), so a dense, mutually-coupled strong-line forest does not collapse
    into one mega-window. The default ``max_peaks_per_window=0`` bounds windows by
    width alone. The bound is on the peak content, not the padded ``freq_range``:
    a cluster whose lines fit the cap stays in one window even when its empty
    proto-margins push the padded span over the cap (review finding F2)."""

    @staticmethod
    def _dense_cluster():
        # 20 strong lines 0.8 MHz apart over ~15 MHz: the spacing is below
        # 2 * the 32-point (0.64 MHz on this 0.02 MHz grid) window margin, so the
        # per-peak proto-spans overlap and chain them into one ~15 MHz / 20-peak
        # span -- a genuinely coupled forest the width/peak caps must break up.
        lines = [(30040.0 + 0.8 * i, 3.0, PeakClassification.STRONG) for i in range(20)]
        return _synthetic(lines, n=8000)

    def test_dense_strong_forest_is_split_to_width_cap(self):
        # max_peaks_per_window=0 and the points cap pinned off: bounded by the
        # MHz width cap alone. The cap is set below the ~15 MHz forest content so
        # it must split (the default 40 MHz cap would hold the whole forest).
        freqs, spec, rms, peaks = self._dense_cluster()
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_mhz=5.0,
            max_window_width_points=0,
        )
        _assert_invariants(plan)
        # The forest must NOT collapse into one mega-window: the width cap splits
        # it even with no peak cap.
        assert plan.n_windows >= 2
        for w in plan.windows:
            assert (
                _content_mhz(w, peaks) <= 5.0 + 1e-6
            ), "window peak content exceeds the width cap"
        # Every promoted line is still covered exactly once (no dropped peaks).
        covered = sorted(li for w in plan.windows for li in w.free_peak_indices)
        assert covered == list(range(len(peaks)))

    def test_default_points_cap_bounds_windows(self):
        # The hard default is a 96-point cap (the small-window operating
        # point of the Stage 5 window-invariant gates); with no explicit
        # width knobs every window respects it.
        freqs, spec, rms, peaks = self._dense_cluster()
        step = float(np.mean(np.diff(np.sort(freqs))))
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        _assert_invariants(plan)
        assert plan.parameters["max_window_width_points"] == 96
        # A window's padded width is its peak content (bounded by the 96-point
        # cap) plus the 32-point margin on each side, so the bound is
        # (cap + 2 * margin) grid points.
        bound = (96 + 2 * 32) * step + 1e-6
        for w in plan.windows:
            assert w.width_mhz <= bound
        covered = sorted(li for w in plan.windows for li in w.free_peak_indices)
        assert covered == list(range(len(peaks)))

    def test_tighter_peak_cap_makes_more_windows(self):
        # Two explicit positive caps: the tighter one must split into more
        # windows. Points cap pinned off so the peak cap (not the width cap
        # at this fine grid step) drives the partition.
        freqs, spec, rms, peaks = self._dense_cluster()
        loose = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_peaks_per_window=8,
            max_window_width_points=0,
        )
        tight = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_peaks_per_window=4,
            max_window_width_points=0,
        )
        assert tight.n_windows > loose.n_windows
        for w in tight.windows:
            assert len(w.free_peak_indices) <= 4
        for w in loose.windows:
            assert len(w.free_peak_indices) <= 8
        _assert_invariants(tight)

    def test_coupled_pair_within_cap_stays_merged(self):
        # Two strong lines 1 MHz apart (< the MHz width cap, < the peak cap, and
        # within the coherence coupling range) must merge into one joint window
        # -- the cap split must not break a genuinely-coupled close pair (the
        # 2638 doublet back-compat case). Points cap pinned off.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30038.0, 3.0, PeakClassification.STRONG),
                (30039.0, 3.0, PeakClassification.STRONG),
            ]
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_window_width_points=0
        )
        assert plan.n_windows == 1
        assert sorted(plan.windows[0].free_peak_indices) == [0, 1]

    def test_content_fitting_cluster_not_split_at_subminimum_gap(self):
        # Review finding F2: a cluster whose peak content fits the cap must NOT
        # be split at a sub-minimum interior gap just because its proto-margins
        # push the padded span over the cap. Four lines spanning 0.94 MHz with a
        # 0.39 MHz largest interior gap (the 2638 33723.5-33724.6 case): with a
        # +/- 2 MHz proto-margin the merged span is ~4.9 MHz, over a 4 MHz cap,
        # but the 0.94 MHz of content fits, so it stays one centered window.
        freqs, spec, rms, peaks = _synthetic(
            [
                (30040.00, 1.0, PeakClassification.WEAK),
                (30040.24, 1.0, PeakClassification.WEAK),
                (30040.63, 1.0, PeakClassification.WEAK),
                (30040.94, 1.0, PeakClassification.WEAK),
            ]
        )
        step = float(np.mean(np.diff(np.sort(freqs))))
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_points=int(round(4.0 / step)),
        )
        assert plan.n_windows == 1, "content-fitting cluster was split"
        assert sorted(plan.windows[0].free_peak_indices) == [0, 1, 2, 3]
        # The lone window is centered on its content, not edge-piled.
        w = plan.windows[0]
        content_lo, content_hi = 30040.00, 30040.94
        margin_lo = content_lo - w.freq_range[0]
        margin_hi = w.freq_range[1] - content_hi
        assert margin_lo > 0 and margin_hi > 0
        assert abs(margin_lo - margin_hi) < 0.5, "cluster is off-center"
        _assert_invariants(plan)

    def test_max_peaks_per_window_recorded_in_parameters(self):
        freqs, spec, rms, peaks = self._dense_cluster()
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, max_peaks_per_window=6
        )
        assert plan.parameters["max_peaks_per_window"] == 6

    def test_points_cap_matches_equivalent_mhz_cap(self):
        # The points cap is the portable form of the width cap: a positive
        # value supersedes the MHz cap, and ``points = mhz / grid step``
        # must reproduce the identical partition.
        freqs, spec, rms, peaks = self._dense_cluster()
        step = float(np.mean(np.diff(np.sort(freqs))))
        cap_mhz = 12.0
        cap_points = int(round(cap_mhz / step))
        by_mhz = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_mhz=cap_mhz,
            max_window_width_points=0,
        )
        by_points = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_points=cap_points,
        )
        assert by_points.n_windows == by_mhz.n_windows
        assert [w.freq_range for w in by_points.windows] == [
            w.freq_range for w in by_mhz.windows
        ]
        assert by_points.parameters["max_window_width_points"] == cap_points
        _assert_invariants(by_points)

    def test_points_cap_supersedes_mhz_cap(self):
        # With both set, the points cap governs: a tight points cap splits
        # the forest even when the MHz cap alone would not.
        freqs, spec, rms, peaks = self._dense_cluster()
        step = float(np.mean(np.diff(np.sort(freqs))))
        plan = build_window_plan(
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_mhz=1000.0,
            max_window_width_points=int(round(10.0 / step)),
        )
        assert plan.n_windows >= 4
        for w in plan.windows:
            assert _content_mhz(w, peaks) <= 10.0 + 2 * step


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


class TestActiveFrame:
    """The plan scores the active FT directly: it is in the ``[0, T]`` frame
    (sliced active region), so there is no turn-on ramp to de-ramp. A ramped
    spectrum -- which a caller should never pass -- is scored as-is, confirming
    the de-ramp is not silently reintroduced (the full-record turn-on physics is
    tested on ``deramp_to_active_start`` itself)."""

    def test_active_frame_is_scored_directly(self):
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
        ramped_plan = build_window_plan(peaks, freqs, ramped, rms, acquisition_us=15.0)

        ref_stat = max(w.diagnostics["edge_coherence_statistic"] for w in ref.windows)
        ramped_stat = max(
            w.diagnostics["edge_coherence_statistic"] for w in ramped_plan.windows
        )
        # The [0, T] spectrum carries coherent leakage; a spurious ramp collapses
        # the coherent edge statistic -- the plan does not de-ramp it back.
        assert ref_stat > 2.0 * ramped_stat
        _assert_invariants(ref)

    def test_no_deramp_parameters_recorded(self):
        freqs, spec, rms, peaks = _synthetic(
            [(30040.0, 2.0, PeakClassification.STRONG)]
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        assert "start_us" not in plan.parameters
        assert "probe_freq_mhz" not in plan.parameters


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
        """Two well-separated weak windows -- a clean merge fixture."""
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
        and a weak line on its skirt. With ``skirt_level_keep`` low, Stage 4
        builds a 2-window plan with the strong line attached to the weak window
        as an edge-bearing contributor + a dependency edge. Merge the two
        windows: the merged window now contains the strong line as a free peak,
        so its earlier role as a fixed contributor to the (now-merged) weak
        window is no longer needed and the contributor list comes back empty.
        """
        freqs, spec, rms, peaks = _synthetic(
            [
                (30050.0, 6.0, PeakClassification.STRONG),
                (30056.0, 0.12, PeakClassification.MEDIUM),
            ],
            n=8000,
        )
        plan = build_window_plan(
            peaks, freqs, spec, rms, acquisition_us=15.0, skirt_level_keep=1.0
        )
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
            skirt_level_keep=1.0,
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

    def test_the_revised_plan_records_the_caps_merge_geometry_reads(self):
        """A revised plan carries both caps, so a later round's
        :func:`merge_geometry` judges a merge by the settings Stage 4 used."""
        plan, peaks, freqs, spec, rms = self._two_window_plan()
        revised = replan(
            plan,
            [],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
            max_window_width_points=80,
            max_peaks_per_window=5,
        )
        assert revised.parameters["max_window_width_points"] == 80
        assert revised.parameters["max_peaks_per_window"] == 5
        a, b = revised.windows
        geom = merge_geometry(revised, a.window_id, b.window_id, peaks, freqs)
        assert (geom.cap_bins, geom.max_peaks) == (80.0, 5)

    def test_a_chain_of_merges_names_every_absorbed_id(self):
        """Round 1 folds the top window into the middle one; round 2 folds the
        middle survivor into the bottom one. The final survivor names both
        absorbed ids, whichever side carried them, and neither is reusable."""
        freqs, spec, rms, peaks = _synthetic(
            [
                (30030.0, 0.05, PeakClassification.WEAK),
                (30060.0, 0.05, PeakClassification.WEAK),
                (30090.0, 0.05, PeakClassification.WEAK),
            ],
            n=6000,
        )
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        if plan.n_windows != 3:
            pytest.skip("setup did not produce 3 windows for this fixture")
        lo, mid, hi = sorted(plan.windows, key=lambda w: w.freq_range[0])
        once = replan(
            plan,
            [MergeRequest(mid.window_id, hi.window_id)],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        twice = replan(
            once,
            [MergeRequest(lo.window_id, min(mid.window_id, hi.window_id))],
            peaks,
            freqs,
            spec,
            rms,
            acquisition_us=15.0,
        )
        (survivor,) = twice.windows
        ids = sorted(w.window_id for w in (lo, mid, hi))
        assert survivor.window_id == ids[0]
        assert survivor.diagnostics["merged_from"] == ids
        assert merged_window_ids(twice) == {ids[0]: ids[1:]}
        assert retired_window_ids(twice) == set(ids[1:])
        assert next_window_id(twice) == ids[-1] + 1

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


# ---------------------------------------------------------------------------
# merge_geometry: what a proposed merge would span (Stage 5's structural
# renegotiation reads it to admit only touching pairs within the width cap)
# ---------------------------------------------------------------------------
_GEOM_F0 = 30000.0
_GEOM_STEP = 0.02


def _geom_grid(n=2000):
    return _GEOM_F0 + np.arange(n) * _GEOM_STEP


def _geom_plan(windows, parameters=None):
    """A hand-built plan on the ``_geom_grid``. ``windows`` is a list of
    ``(window_id, lo_bin, hi_bin, [peak bins])``: each window spans grid bins
    ``lo_bin..hi_bin`` and owns one free peak per listed bin. Returns the plan
    and the peak list its ``free_peak_indices`` index into."""
    freqs = _geom_grid()
    peaks, fit_windows = [], []
    for wid, lo, hi, pbins in windows:
        idx = []
        for b in pbins:
            idx.append(len(peaks))
            peaks.append(
                Peak(
                    frequency=float(freqs[b]),
                    intensity=1.0,
                    index=b,
                    snr=100.0,
                    noise_std_local=0.01,
                    classification=PeakClassification.STRONG,
                    promoted=True,
                )
            )
        fit_windows.append(
            FitWindow(
                window_id=wid,
                freq_range=(float(freqs[lo]), float(freqs[hi])),
                free_peak_indices=idx,
            )
        )
    plan = WindowPlan(
        windows=fit_windows,
        topological_order=[w.window_id for w in fit_windows],
        parameters={} if parameters is None else dict(parameters),
    )
    return plan, peaks, freqs


class TestMergeGeometry:
    @pytest.mark.parametrize("uncovered", [0, 1, 2, 7])
    def test_uncovered_bins_between_the_windows(self, uncovered):
        """Windows ending at bin 100 and starting at bin 101 + k leave k grid
        points that neither fits."""
        plan, peaks, freqs = _geom_plan(
            [
                (0, 50, 100, [70]),
                (1, 101 + uncovered, 150 + uncovered, [130 + uncovered]),
            ]
        )
        geom = merge_geometry(plan, 0, 1, peaks, freqs)
        assert geom.uncovered_bins == uncovered
        assert (geom.lower_window_id, geom.upper_window_id) == (0, 1)

    def test_windows_sharing_an_endpoint_leave_nothing_uncovered(self):
        # The hand-built plans of the Stage 5 tests abut at one frequency.
        plan, peaks, freqs = _geom_plan([(0, 50, 100, [70]), (1, 100, 150, [130])])
        assert merge_geometry(plan, 0, 1, peaks, freqs).uncovered_bins == 0

    def test_argument_order_and_window_order_do_not_matter(self):
        # The higher id sits at the lower frequency, and is named first.
        plan, peaks, freqs = _geom_plan([(5, 50, 100, [70]), (2, 103, 150, [130])])
        a = merge_geometry(plan, 5, 2, peaks, freqs)
        b = merge_geometry(plan, 2, 5, peaks, freqs)
        assert a == b
        assert (a.lower_window_id, a.upper_window_id) == (5, 2)
        assert a.uncovered_bins == 2

    def test_a_reversed_frequency_axis_gives_the_same_geometry(self):
        plan, peaks, freqs = _geom_plan([(0, 50, 100, [70]), (1, 102, 150, [130])])
        fwd = merge_geometry(plan, 0, 1, peaks, freqs)
        assert merge_geometry(plan, 0, 1, peaks, freqs[::-1]) == fwd

    def test_content_is_the_span_between_the_outermost_free_peaks(self):
        plan, peaks, freqs = _geom_plan(
            [
                (0, 50, 100, [60, 90]),
                (1, 101, 150, [110, 140]),
            ],
            {"max_window_width_points": 96},
        )
        geom = merge_geometry(plan, 0, 1, peaks, freqs)
        assert geom.content_bins == 140 - 60
        assert geom.cap_bins == 96.0
        assert geom.within_cap

    def test_points_cap_bounds_the_content(self):
        params = {"max_window_width_points": 96}
        at_cap, peaks, freqs = _geom_plan(
            [(0, 0, 100, [10]), (1, 101, 200, [10 + 96])], params
        )
        over, peaks2, _ = _geom_plan(
            [(0, 0, 100, [10]), (1, 101, 200, [10 + 97])], params
        )
        assert merge_geometry(at_cap, 0, 1, peaks, freqs).within_cap
        assert not merge_geometry(over, 0, 1, peaks2, freqs).within_cap

    def test_mhz_cap_applies_when_the_points_cap_is_off(self):
        """``max_window_width_points == 0`` falls back to the MHz cap over the
        grid step: 1.0 MHz at 0.02 MHz per bin is 50 bins (cases kept a few bins off the
        boundary, which the float grid step does not resolve)."""
        params = {"max_window_width_points": 0, "max_window_width_mhz": 1.0}
        ok, peaks, freqs = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [58])], params)
        bad, peaks2, _ = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [62])], params)
        g = merge_geometry(ok, 0, 1, peaks, freqs)
        assert g.cap_bins == pytest.approx(50.0)
        assert g.within_cap
        assert not merge_geometry(bad, 0, 1, peaks2, freqs).within_cap

    def test_a_positive_points_cap_supersedes_the_mhz_cap(self):
        plan, peaks, freqs = _geom_plan(
            [(0, 0, 100, [10]), (1, 101, 200, [60])],
            {"max_window_width_points": 20, "max_window_width_mhz": 40.0},
        )
        geom = merge_geometry(plan, 0, 1, peaks, freqs)
        assert geom.cap_bins == 20.0
        assert not geom.within_cap

    @pytest.mark.parametrize(
        "max_peaks, within", [(0, True), (4, True), (3, True), (2, False)]
    )
    def test_the_peak_cap_bounds_the_merged_line_count(self, max_peaks, within):
        """``max_peaks_per_window`` (when positive) bounds the promoted peaks
        the merged window holds, as the planner's cap split does; ``0`` is no
        cap."""
        plan, peaks, freqs = _geom_plan(
            [(0, 0, 100, [60, 90]), (1, 101, 200, [110])],
            {"max_peaks_per_window": max_peaks},
        )
        geom = merge_geometry(plan, 0, 1, peaks, freqs)
        assert (geom.n_peaks, geom.max_peaks) == (3, max_peaks)
        assert geom.within_width_cap
        assert geom.within_peak_cap is within
        assert geom.within_cap is within

    def test_a_plan_without_width_parameters_uses_the_stage4_defaults(self):
        plan, peaks, freqs = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [60])])
        geom = merge_geometry(plan, 0, 1, peaks, freqs)
        expected = (
            float(DEFAULT_MAX_WINDOW_WIDTH_POINTS)
            if DEFAULT_MAX_WINDOW_WIDTH_POINTS > 0
            else DEFAULT_MAX_WINDOW_WIDTH_MHZ / _GEOM_STEP
        )
        assert geom.cap_bins == pytest.approx(expected)
        assert geom.max_peaks == DEFAULT_MAX_PEAKS_PER_WINDOW

    def test_windows_with_fewer_than_two_peaks_have_no_content_span(self):
        plan, peaks, freqs = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [])])
        assert merge_geometry(plan, 0, 1, peaks, freqs).content_bins == 0

    def test_unknown_window_raises(self):
        plan, peaks, freqs = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [60])])
        with pytest.raises(ValueError, match="unknown window 9"):
            merge_geometry(plan, 0, 9, peaks, freqs)

    def test_identical_ids_raise(self):
        plan, peaks, freqs = _geom_plan([(0, 0, 100, [10]), (1, 101, 200, [60])])
        with pytest.raises(ValueError, match="distinct windows"):
            merge_geometry(plan, 1, 1, peaks, freqs)

    def test_a_real_planner_cap_split_leaves_its_halves_touching(self):
        """A run of lines longer than the cap is split by the planner into
        windows that abut -- the neighbours a structural merge is meant for --
        while two lines far apart leave a wide gap."""
        lines = [(30040.0 + 0.06 * k, 0.05, PeakClassification.WEAK) for k in range(60)]
        freqs, spec, rms, peaks = _synthetic(lines)
        plan = build_window_plan(peaks, freqs, spec, rms, acquisition_us=15.0)
        ws = sorted(plan.windows, key=lambda w: w.freq_range[0])
        assert len(ws) >= 2
        for a, b in zip(ws, ws[1:]):
            geom = merge_geometry(plan, a.window_id, b.window_id, peaks, freqs)
            assert geom.uncovered_bins <= 1
            assert geom.cap_bins == float(DEFAULT_MAX_WINDOW_WIDTH_POINTS)

        far = TestReplanMerge()._two_window_plan()
        fplan, fpeaks, ffreqs, _spec, _rms = far
        a, b = fplan.windows
        assert (
            merge_geometry(
                fplan, a.window_id, b.window_id, fpeaks, ffreqs
            ).uncovered_bins
            > 100
        )


# ---------------------------------------------------------------------------
# Window ids after a structural merge: an absorbed id is never minted again.
# ---------------------------------------------------------------------------


def _plan_of(*windows):
    return WindowPlan(windows=list(windows))


def test_an_unmerged_plan_retires_nothing():
    plan = _plan_of(FitWindow(0, (1.0, 2.0)), FitWindow(3, (4.0, 5.0)))
    assert merged_window_ids(plan) == {}
    assert retired_window_ids(plan) == set()
    assert next_window_id(plan) == 4


def test_a_new_id_goes_above_an_absorbed_top_id():
    """Window 7 (the highest id) was folded into 5: the next id is 8, not 7."""
    plan = _plan_of(
        FitWindow(2, (1.0, 2.0)),
        FitWindow(5, (3.0, 6.0), diagnostics={"merged_from": [5, 7]}),
    )
    assert merged_window_ids(plan) == {5: [7]}
    assert retired_window_ids(plan) == {7}
    assert next_window_id(plan) == 8
    assert next_window_id(plan, reserved=[11]) == 12


def test_a_merged_from_naming_only_itself_is_not_a_merge():
    plan = _plan_of(FitWindow(4, (1.0, 2.0), diagnostics={"merged_from": [4]}))
    assert merged_window_ids(plan) == {}
    assert next_window_id(plan) == 5
