"""
Stage 6 window creation (issue #39).

``review create`` installs a fit window for a line the automatic detection
missed. The whole point is that it is **additive**: reaching such a line by
lowering the detection threshold re-runs Stage 3, which drops Stages 5 and 6 and
destroys the entire curated edit set, so creating the window has to leave every
existing window, fit, and decision standing.

The contract these tests pin down, in the order the issue states it:

1. **Explicit, not implicit.** Creating a window is its own decision kind
   (``"create_window"``); putting a line in it is a separate ``"add"``. Two log
   entries, never one.
2. **Fresh ids, never renumber.** An existing window keeps its id and its peaks'
   ``window_id`` after a create -- the property a consumer relies on when it
   partitions peaks on ``window_id`` to decide what an edit touched.
3. **Deterministic bounds.** The extent is a function of the anchor and the base
   plan, not of the curated state, so replaying the edit set reproduces the same
   window.
4. **No cascade.** No existing window is re-fit or thawed by a create.

Plus the two resolved open questions: an anchor already inside a window is
refused (that case is ``review edit --add``), and a gap too narrow to hold a
fittable window widens the adjacent window instead.

The ``stage5_small_file`` fixture is a 3-window 2638 subset; Stage 5 keeps only
the windows whose peaks survive their gates, so the helpers below work from the
*fitted* windows rather than the plan.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    CreateWindowResult,
    apply_curation_impl,
    create_window_impl,
    effective_window_plan,
    refit_window_impl,
    review_log_impl,
    review_run_impl,
    review_undo_impl,
)
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _live_ranges(path: Path) -> List[Tuple[int, float, float]]:
    """``(window_id, lo, hi)`` for every window that carries a Stage 5 fit."""
    sf = _load_spectrum_fit(path)
    plan = load_windows_impl(str(path))["plan"]
    by_id = {int(w.window_id): w for w in plan.windows}
    out: List[Tuple[int, float, float]] = []
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        w = by_id.get(int(wf.window_id))
        if w is None:
            continue
        lo, hi = w.freq_range
        out.append((int(wf.window_id), min(lo, hi), max(lo, hi)))
    return sorted(out, key=lambda t: t[1])


def _free_anchor(path: Path) -> float:
    """A frequency well clear of every fitted window (and inside the trim)."""
    live = _live_ranges(path)
    assert live, "fixture has no fitted windows"
    return live[-1][2] + 5.0


@pytest.fixture
def working_file(stage5_small_source, tmp_path) -> Path:
    dst = tmp_path / "working.ftmw"
    shutil.copy(stage5_small_source, dst)
    review_run_impl(str(dst))
    return dst


# ---------------------------------------------------------------------------
# 1. The window gets created, and it is structure only
# ---------------------------------------------------------------------------


class TestCreateInstallsAWindow:
    def test_returns_a_fresh_window_covering_the_anchor(self, working_file):
        anchor = _free_anchor(working_file)
        before = {
            int(w.window_id)
            for w in load_windows_impl(str(working_file))["plan"].windows
        }

        result = create_window_impl(str(working_file), anchor)

        assert isinstance(result, CreateWindowResult)
        assert result.mode == "created"
        assert result.window_id not in before
        lo, hi = result.freq_range
        assert lo <= anchor <= hi
        assert result.n_points > 0

    def test_new_window_is_visible_in_the_effective_plan(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        eff = effective_window_plan(str(working_file))
        ids = [int(w.window_id) for w in eff.windows]
        assert result.window_id in ids

        # ... and it is an overlay, not a mutation of the Stage 4 product.
        base_ids = [
            int(w.window_id)
            for w in load_windows_impl(str(working_file))["plan"].windows
        ]
        assert result.window_id not in base_ids

    def test_create_adds_no_peaks(self, working_file):
        """Creating a window installs structure; the line is a separate decision."""
        anchor = _free_anchor(working_file)
        before = len(_load_spectrum_fit(working_file).fitted_peaks)

        result = create_window_impl(str(working_file), anchor)

        assert result.n_peaks == 0
        assert len(_load_spectrum_fit(working_file).fitted_peaks) == before

    def test_the_new_window_is_editable_like_any_other(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        refit = refit_window_impl(str(working_file), result.window_id, add=[anchor])

        assert refit.n_peaks_before == 0
        assert refit.n_peaks_after == 1
        assert refit.fitted_peaks[0].origin == "user"

    def test_window_survives_the_hdf5_round_trip(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        overlay = load_stage6_review_from_file(str(working_file)).created_windows
        assert [int(w.window_id) for w in overlay] == [result.window_id]
        lo, hi = overlay[0].freq_range
        assert (min(lo, hi), max(lo, hi)) == result.freq_range


# ---------------------------------------------------------------------------
# 2. Explicit, not implicit: two decisions, not one
# ---------------------------------------------------------------------------


class TestDecisionLog:
    def test_create_records_its_own_decision_kind(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        log = review_log_impl(str(working_file))
        assert len(log) == 1
        entry = log[0]
        assert entry.kind == "create_window"
        assert entry.window_id == result.window_id
        assert entry.frequency_mhz == pytest.approx(anchor)
        assert entry.provenance == "user"

    def test_create_then_add_are_two_decisions(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), result.window_id, add=[anchor])

        assert [e.kind for e in review_log_impl(str(working_file))] == [
            "create_window",
            "add",
        ]

    def test_evidence_records_the_installed_geometry(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        ev = review_log_impl(str(working_file))[0].evidence
        assert ev["mode"] == "created"
        assert ev["freq_min_mhz"] == pytest.approx(result.freq_range[0])
        assert ev["freq_max_mhz"] == pytest.approx(result.freq_range[1])
        assert ev["n_points"] == result.n_points


# ---------------------------------------------------------------------------
# 3. Fresh ids, never renumber; no cascade
# ---------------------------------------------------------------------------


class TestPurelyAdditive:
    def test_existing_window_ids_are_untouched(self, working_file):
        before = {
            int(w.window_id)
            for w in load_windows_impl(str(working_file))["plan"].windows
        }
        create_window_impl(str(working_file), _free_anchor(working_file))
        after = {
            int(w.window_id)
            for w in load_windows_impl(str(working_file))["plan"].windows
        }
        assert before == after

    def test_existing_peaks_keep_their_window_id_and_frequency(self, working_file):
        before = sorted(
            (p.window_id, round(float(p.frequency_mhz), 9))
            for p in _load_spectrum_fit(working_file).fitted_peaks
        )
        create_window_impl(str(working_file), _free_anchor(working_file))
        after = sorted(
            (p.window_id, round(float(p.frequency_mhz), 9))
            for p in _load_spectrum_fit(working_file).fitted_peaks
        )
        assert before == after, "a create must not re-fit or re-identify any peak"

    def test_no_existing_window_is_refit(self, working_file):
        """No cascade: every existing window's chi-squared is bit-identical."""
        before = {
            int(wf.window_id): float(wf.reduced_chi2)
            for wf in _load_spectrum_fit(working_file).window_fits
            if wf.window_id is not None
        }
        result = create_window_impl(str(working_file), _free_anchor(working_file))
        after = {
            int(wf.window_id): float(wf.reduced_chi2)
            for wf in _load_spectrum_fit(working_file).window_fits
            if wf.window_id is not None
        }
        assert before == {k: v for k, v in after.items() if k != result.window_id}

    def test_the_new_window_is_a_leaf(self, working_file):
        """It reads leakage inward; nothing depends on it."""
        result = create_window_impl(str(working_file), _free_anchor(working_file))
        eff = effective_window_plan(str(working_file))
        outward = [e for e in eff.dependency_edges if e[1] == result.window_id]
        assert outward == []


# ---------------------------------------------------------------------------
# 4. Deterministic bounds
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_anchor_on_two_copies_gives_the_same_window(
        self, stage5_small_source, tmp_path
    ):
        a = tmp_path / "a.ftmw"
        b = tmp_path / "b.ftmw"
        shutil.copy(stage5_small_source, a)
        shutil.copy(stage5_small_source, b)
        anchor = _free_anchor(a)

        ra = create_window_impl(str(a), anchor)
        rb = create_window_impl(str(b), anchor)

        assert (ra.window_id, ra.mode, ra.n_points) == (
            rb.window_id,
            rb.mode,
            rb.n_points,
        )
        assert ra.freq_range == rb.freq_range

    def test_extent_does_not_depend_on_earlier_edits(
        self, stage5_small_source, tmp_path
    ):
        """Bounds are a function of the anchor and the BASE plan, so an
        unrelated edit made first must not move them."""
        plain = tmp_path / "plain.ftmw"
        edited = tmp_path / "edited.ftmw"
        shutil.copy(stage5_small_source, plain)
        shutil.copy(stage5_small_source, edited)
        anchor = _free_anchor(plain)

        live = _live_ranges(edited)
        wid = live[0][0]
        sf = _load_spectrum_fit(edited)
        wf = next(w for w in sf.window_fits if int(w.window_id) == wid)
        if wf.fitted_peaks:
            refit_window_impl(
                str(edited), wid, remove=[float(wf.fitted_peaks[0].frequency_mhz)]
            )

        r_plain = create_window_impl(str(plain), anchor)
        r_edited = create_window_impl(str(edited), anchor)
        assert r_plain.freq_range == r_edited.freq_range
        assert r_plain.n_points == r_edited.n_points


# ---------------------------------------------------------------------------
# 5. Overlap: refuse an anchor a window already covers
# ---------------------------------------------------------------------------


class TestAnchorInsideAnExistingWindow:
    def test_refused(self, working_file):
        wid, lo, hi = _live_ranges(working_file)[0]
        with pytest.raises(ValueError, match="already falls inside window"):
            create_window_impl(str(working_file), 0.5 * (lo + hi))

    def test_error_points_at_review_edit(self, working_file):
        wid, lo, hi = _live_ranges(working_file)[0]
        with pytest.raises(ValueError) as exc:
            create_window_impl(str(working_file), 0.5 * (lo + hi))
        assert f"--window {wid}" in str(exc.value)

    def test_nothing_is_written(self, working_file):
        wid, lo, hi = _live_ranges(working_file)[0]
        with pytest.raises(ValueError):
            create_window_impl(str(working_file), 0.5 * (lo + hi))
        assert review_log_impl(str(working_file)) == []
        assert load_stage6_review_from_file(str(working_file)).created_windows == []

    def test_anchor_outside_the_analysis_band_is_refused(self, working_file):
        with pytest.raises(ValueError, match="outside the analysis band"):
            create_window_impl(str(working_file), 1000.0)


# ---------------------------------------------------------------------------
# 6. Narrow gap: widen the adjacent window instead of starving a new one
# ---------------------------------------------------------------------------


class TestNarrowGapWidens:
    def _bin_width_mhz(self, path: Path) -> float:
        plan = load_windows_impl(str(path))["plan"]
        w = next(
            w
            for w in plan.windows
            if "grid_span" in w.diagnostics
            and w.diagnostics["grid_span"][1] > w.diagnostics["grid_span"][0]
        )
        lo, hi = w.freq_range
        span = w.diagnostics["grid_span"]
        return abs(hi - lo) / (int(span[1]) - int(span[0]))

    def test_anchor_hugging_an_edge_widens_that_window(self, working_file):
        step = self._bin_width_mhz(working_file)
        wid, lo, hi = _live_ranges(working_file)[-1]

        result = create_window_impl(str(working_file), hi + 2 * step)

        assert result.mode == "widened"
        assert result.window_id == wid
        assert result.freq_range[1] > hi
        # The window still covers everything it did before.
        assert result.freq_range[0] <= lo

    def test_widening_keeps_the_windows_peaks(self, working_file):
        step = self._bin_width_mhz(working_file)
        wid, lo, hi = _live_ranges(working_file)[-1]
        sf = _load_spectrum_fit(working_file)
        n_before = len(
            next(w for w in sf.window_fits if int(w.window_id) == wid).fitted_peaks
        )

        result = create_window_impl(str(working_file), hi + 2 * step)

        assert result.n_peaks == n_before

    def test_widening_keeps_the_windows_peak_uids(self, working_file):
        """A stronger form of ``test_widening_keeps_the_windows_peaks``: not
        just the peak COUNT survives a widening, but each peak's own
        ``peak_uid`` -- stamped once at birth and never re-derived
        (``peak_uid`` is the identity a downstream consumer tracks a peak by
        across an edit, not its frequency or index)."""
        step = self._bin_width_mhz(working_file)
        wid, lo, hi = _live_ranges(working_file)[-1]
        sf = _load_spectrum_fit(working_file)
        uids_before = {
            p.peak_uid
            for p in next(
                w for w in sf.window_fits if int(w.window_id) == wid
            ).fitted_peaks
        }

        create_window_impl(str(working_file), hi + 2 * step)

        sf_after = _load_spectrum_fit(working_file)
        uids_after = {
            p.peak_uid
            for p in next(
                w for w in sf_after.window_fits if int(w.window_id) == wid
            ).fitted_peaks
        }
        assert uids_after == uids_before

    def test_widening_records_its_mode_in_the_log(self, working_file):
        step = self._bin_width_mhz(working_file)
        wid, lo, hi = _live_ranges(working_file)[-1]
        create_window_impl(str(working_file), hi + 2 * step)

        entry = review_log_impl(str(working_file))[0]
        assert entry.kind == "create_window"
        assert entry.window_id == wid
        assert entry.evidence["mode"] == "widened"

    def test_widened_window_accepts_the_anchor_as_an_add(self, working_file):
        step = self._bin_width_mhz(working_file)
        _wid, _lo, hi = _live_ranges(working_file)[-1]
        anchor = hi + 2 * step
        result = create_window_impl(str(working_file), anchor)

        refit = refit_window_impl(str(working_file), result.window_id, add=[anchor])
        assert refit.n_peaks_after == refit.n_peaks_before + 1


# ---------------------------------------------------------------------------
# 6a. A widened window cascades; a created one still doesn't (W1)
#
# 2638 (the fixture this whole file builds from) has zero ``frozen_peak_``
# entries, so no window is ever a real freeze source for another and
# ``_cascade_succs`` returns an empty graph -- a cascade edge cannot occur
# naturally here. Both tests below fake one, exactly as
# ``test_curation.py::test_cascade_downstream_of_two_edits_refit_once`` does:
# wrap ``_cascade_succs`` to inject a dependent, and spy on
# ``refit_window_core`` to see which windows actually got refit.
# ---------------------------------------------------------------------------


class TestWidenedWindowCascades:
    def _bin_width_mhz(self, path: Path) -> float:
        plan = load_windows_impl(str(path))["plan"]
        w = next(
            w
            for w in plan.windows
            if "grid_span" in w.diagnostics
            and w.diagnostics["grid_span"][1] > w.diagnostics["grid_span"][0]
        )
        lo, hi = w.freq_range
        span = w.diagnostics["grid_span"]
        return abs(hi - lo) / (int(span[1]) - int(span[0]))

    def test_widened_windows_dependent_is_refit(self, working_file, monkeypatch):
        """The new behavior: a widened window's fit just moved on the wider
        grid, so a (faked) dependent that froze on its old leakage skirt gets
        refit in the same combined cascade.

        ``working_file`` carries only one live (Stage-5-fitted) window, so a
        second one is minted first (a plain create, well clear of the
        widening anchor below) to serve as the fake dependent -- the widen
        itself still targets the original window's edge, exactly as
        ``test_anchor_hugging_an_edge_widens_that_window`` does.
        """
        step = self._bin_width_mhz(working_file)
        wid, lo, hi = _live_ranges(working_file)[-1]
        dep_anchor = _free_anchor(working_file)
        dep_wid = create_window_impl(str(working_file), dep_anchor).window_id

        orig_succs = s6._cascade_succs

        def fake_succs(window_fits, fit_window_map):
            d = orig_succs(window_fits, fit_window_map)
            d.setdefault(wid, set()).add(dep_wid)
            return d

        monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

        orig_core = s6.refit_window_core
        calls: List[int] = []

        def spy_core(fit_ctx, fit_win, wf, **kwargs):
            calls.append(int(fit_win.window_id))
            return orig_core(fit_ctx, fit_win, wf, **kwargs)

        monkeypatch.setattr(s6, "refit_window_core", spy_core)

        result = create_window_impl(str(working_file), hi + 2 * step)

        assert result.mode == "widened"
        assert dep_wid in calls

    def test_created_windows_dependent_is_not_refit(self, working_file, monkeypatch):
        """The guard this fix must not swallow: a freshly created window is a
        leaf with no outbound dependency edge, so it stays mutated-but-not-
        dirty and no cascade runs for it -- even when the (faked) graph says
        one of its neighbors would otherwise be reachable. This is what pins
        the created/widened distinction against a later "simplification" into
        a blanket ``dirty_wids.add`` for both modes: were that regression
        introduced, the newly created window's id would land in
        ``dirty_wids``, the fake edge below would be found, and this
        assertion would fail."""
        anchor = _free_anchor(working_file)
        live = _live_ranges(working_file)
        dep_wid = live[0][0]

        orig_succs = s6._cascade_succs

        def fake_succs(window_fits, fit_window_map):
            d = orig_succs(window_fits, fit_window_map)
            # The new window's id isn't known ahead of time, so point EVERY
            # live window (including whatever the create mints) at dep_wid.
            for wf in window_fits:
                if wf.window_id is not None:
                    d.setdefault(int(wf.window_id), set()).add(dep_wid)
            return d

        monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

        orig_core = s6.refit_window_core
        calls: List[int] = []

        def spy_core(fit_ctx, fit_win, wf, **kwargs):
            calls.append(int(fit_win.window_id))
            return orig_core(fit_ctx, fit_win, wf, **kwargs)

        monkeypatch.setattr(s6, "refit_window_core", spy_core)

        result = create_window_impl(str(working_file), anchor)

        assert result.mode == "created"
        assert dep_wid not in calls


# ---------------------------------------------------------------------------
# 6b. Several created windows: ids append, and stay put when one is undone
# ---------------------------------------------------------------------------


class TestMultipleCreatedWindows:
    def _two_anchors(self, path: Path) -> Tuple[float, float]:
        hi = _live_ranges(path)[-1][2]
        return hi + 5.0, hi + 15.0

    def test_ids_keep_appending(self, working_file):
        a1, a2 = self._two_anchors(working_file)
        r1 = create_window_impl(str(working_file), a1)
        r2 = create_window_impl(str(working_file), a2)

        assert r2.window_id == r1.window_id + 1
        ids = [
            int(w.window_id) for w in effective_window_plan(str(working_file)).windows
        ]
        assert r1.window_id in ids and r2.window_id in ids

    def test_second_window_is_unaffected_by_the_first(self, working_file, tmp_path):
        """Two well-separated creates do not interact."""
        a1, a2 = self._two_anchors(working_file)
        solo = tmp_path / "solo.ftmw"
        shutil.copy(working_file, solo)

        r_solo = create_window_impl(str(solo), a2)
        create_window_impl(str(working_file), a1)
        r_pair = create_window_impl(str(working_file), a2)

        assert r_pair.freq_range == r_solo.freq_range

    def test_undoing_an_earlier_create_does_not_renumber_a_later_one(
        self, working_file
    ):
        """The identity guarantee that makes window_id usable as a key: a
        surviving created window keeps its id even when an earlier create is
        dropped, leaving a harmless gap in the numbering rather than sliding
        every later window down one."""
        a1, a2 = self._two_anchors(working_file)
        r1 = create_window_impl(str(working_file), a1)
        r2 = create_window_impl(str(working_file), a2)
        refit_window_impl(str(working_file), r2.window_id, add=[a2])

        review_undo_impl(str(working_file), [0])

        ids = [
            int(w.window_id) for w in effective_window_plan(str(working_file)).windows
        ]
        assert r2.window_id in ids
        assert r1.window_id not in ids

        # The surviving decisions still name that window ...
        log = review_log_impl(str(working_file))
        assert [(e.kind, e.window_id) for e in log] == [
            ("create_window", r2.window_id),
            ("add", r2.window_id),
        ]
        # ... and so does the peak the add created.
        peaks = [
            p
            for p in _load_spectrum_fit(working_file).fitted_peaks
            if p.window_id == r2.window_id
        ]
        assert len(peaks) == 1
        assert peaks[0].origin == "user"

    def test_a_curation_file_can_pin_a_window_id(self, working_file, tmp_path):
        a1, _a2 = self._two_anchors(working_file)
        base = max(
            int(w.window_id)
            for w in load_windows_impl(str(working_file))["plan"].windows
        )
        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{base + 7},{a1:.6f},\n")

        apply_curation_impl(str(working_file), cf)

        assert review_log_impl(str(working_file))[0].window_id == base + 7

    def test_pinning_an_id_that_is_taken_is_refused(self, working_file, tmp_path):
        a1, _a2 = self._two_anchors(working_file)
        taken = _live_ranges(working_file)[0][0]
        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{taken},{a1:.6f},\n")

        with pytest.raises(ValueError, match="already in use"):
            apply_curation_impl(str(working_file), cf)


# ---------------------------------------------------------------------------
# 7. Replay: curation files and undo
# ---------------------------------------------------------------------------


class TestReplay:
    def test_curation_file_creates_then_adds(self, working_file, tmp_path):
        anchor = _free_anchor(working_file)
        expected = create_window_impl(str(working_file), anchor).window_id
        # Start over on a clean copy and drive the same edit from a file.
        fresh = tmp_path / "fresh.ftmw"
        shutil.copy(working_file, fresh)
        review_undo_impl(str(fresh), [0])

        cf = tmp_path / "cur.csv"
        cf.write_text(
            "action,window,freqs,params\n"
            f"create,new,{anchor:.6f},\n"
            f"add,{expected},{anchor:.6f},\n"
        )
        result = apply_curation_impl(str(fresh), cf)

        assert result.applied == 2
        assert [e.kind for e in review_log_impl(str(fresh))] == [
            "create_window",
            "add",
        ]

    def test_curation_dry_run_describes_the_create(self, working_file, tmp_path):
        anchor = _free_anchor(working_file)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,new,{anchor:.6f},\n")
        result = apply_curation_impl(str(working_file), cf, dry_run=True)
        assert result.applied == 0
        assert len(result.plan) == 1
        assert result.plan[0].kind == "create"

    def test_pinned_id_is_refused_when_the_anchor_now_widens_instead(
        self, working_file, tmp_path
    ):
        """A pinned id replays the *label*, not the geometry. If the anchor now
        resolves to widening an existing window rather than creating a new one,
        the label is meaningless and the replay must say so before writing."""
        plan = load_windows_impl(str(working_file))["plan"]
        step = TestNarrowGapWidens()._bin_width_mhz(working_file)
        _wid, _lo, hi = _live_ranges(working_file)[-1]
        fresh_id = max(int(w.window_id) for w in plan.windows) + 1

        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{fresh_id},{hi + 2 * step:.6f},\n")

        with pytest.raises(ValueError, match="widens window"):
            apply_curation_impl(str(working_file), cf)

    def test_a_failed_pinned_replay_writes_nothing(self, working_file, tmp_path):
        step = TestNarrowGapWidens()._bin_width_mhz(working_file)
        _wid, _lo, hi = _live_ranges(working_file)[-1]
        plan = load_windows_impl(str(working_file))["plan"]
        fresh_id = max(int(w.window_id) for w in plan.windows) + 1
        before = [
            (p.window_id, p.frequency_mhz)
            for p in _load_spectrum_fit(working_file).fitted_peaks
        ]

        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{fresh_id},{hi + 2 * step:.6f},\n")
        with pytest.raises(ValueError):
            apply_curation_impl(str(working_file), cf)

        after = [
            (p.window_id, p.frequency_mhz)
            for p in _load_spectrum_fit(working_file).fitted_peaks
        ]
        assert before == after
        assert review_log_impl(str(working_file)) == []
        assert load_stage6_review_from_file(str(working_file)).created_windows == []

    def test_create_needs_exactly_one_frequency(self, working_file, tmp_path):
        cf = tmp_path / "cur.csv"
        cf.write_text("create,new,,\n")
        with pytest.raises(ValueError, match="create needs exactly one frequency"):
            apply_curation_impl(str(working_file), cf)

    def test_undo_removes_the_window(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)

        review_undo_impl(str(working_file), [0])

        assert review_log_impl(str(working_file)) == []
        assert load_stage6_review_from_file(str(working_file)).created_windows == []
        ids = [
            int(w.window_id) for w in effective_window_plan(str(working_file)).windows
        ]
        assert result.window_id not in ids

    def test_undo_of_a_later_edit_replays_the_create_to_the_same_id(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), result.window_id, add=[anchor])

        review_undo_impl(str(working_file), [1])

        log = review_log_impl(str(working_file))
        assert [e.kind for e in log] == ["create_window"]
        assert log[0].window_id == result.window_id
        ids = [
            int(w.window_id) for w in effective_window_plan(str(working_file)).windows
        ]
        assert result.window_id in ids

    def test_undoing_only_the_create_is_refused_when_edits_depend_on_it(
        self, working_file
    ):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), result.window_id, add=[anchor])

        with pytest.raises(ValueError, match="Undo them together"):
            review_undo_impl(str(working_file), [0])


# ---------------------------------------------------------------------------
# 8. Cross-interface consistency (the repo's dual-interface invariant)
# ---------------------------------------------------------------------------


class TestCrossInterface:
    def _result_tuple(self, r: CreateWindowResult):
        return (r.window_id, r.mode, r.freq_range, r.n_points, r.n_contributors)

    def test_api_pipeline_and_impl_agree(self, stage5_small_source, tmp_path):
        paths = []
        for name in ("impl.ftmw", "pipe.ftmw", "api.ftmw"):
            p = tmp_path / name
            shutil.copy(stage5_small_source, p)
            paths.append(p)
        anchor = _free_anchor(paths[0])

        r_impl = create_window_impl(str(paths[0]), anchor)
        r_pipe = Pipeline.open(paths[1]).review_create(anchor)
        r_api = ftmw.review_create(str(paths[2]), anchor)

        assert self._result_tuple(r_impl) == self._result_tuple(r_pipe)
        assert self._result_tuple(r_impl) == self._result_tuple(r_api)

    def test_cli_creates_the_same_window(self, stage5_small_source, tmp_path):
        import argparse

        from ftmwpipeline.cli.review_commands import cmd_review_create

        ref = tmp_path / "ref.ftmw"
        cli = tmp_path / "cli.ftmw"
        shutil.copy(stage5_small_source, ref)
        shutil.copy(stage5_small_source, cli)
        anchor = _free_anchor(ref)

        expected = create_window_impl(str(ref), anchor)
        rc = cmd_review_create(
            argparse.Namespace(file_path=str(cli), anchor=anchor, verbose=False)
        )
        assert rc == 0

        overlay = load_stage6_review_from_file(str(cli)).created_windows
        assert [int(w.window_id) for w in overlay] == [expected.window_id]
        lo, hi = overlay[0].freq_range
        assert (min(lo, hi), max(lo, hi)) == expected.freq_range

    def test_cli_reports_a_bad_anchor_as_a_user_error(
        self, stage5_small_source, tmp_path, capsys
    ):
        import argparse

        from ftmwpipeline.cli.review_commands import cmd_review_create

        p = tmp_path / "bad.ftmw"
        shutil.copy(stage5_small_source, p)
        rc = cmd_review_create(
            argparse.Namespace(file_path=str(p), anchor=1000.0, verbose=False)
        )
        assert rc == 1
        assert "Error:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 9. The created window flows into the downstream Stage 6 products
# ---------------------------------------------------------------------------


class TestDownstreamProducts:
    def test_review_run_gives_the_new_window_a_status(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), result.window_id, add=[anchor])

        review_run_impl(str(working_file))
        status = ftmw.get_review_status(str(working_file))
        assert result.window_id in status.window_statuses

    def test_added_peak_reaches_final_products(self, working_file):
        anchor = _free_anchor(working_file)
        result = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), result.window_id, add=[anchor])

        products = ftmw.get_final_products(str(working_file))
        assert products is not None
        assert any(p.window_id == result.window_id for p in products.peaks)
