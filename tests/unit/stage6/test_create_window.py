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

import json
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

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


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
# 2638 (the fixture this whole file builds from) has no plan contributor
# edges, so no window is ever a real cascade source for another and
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

        def fake_succs(sources):
            d = orig_succs(sources)
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

        def fake_succs(sources):
            d = orig_succs(sources)
            # The new window's id isn't known ahead of time, so point EVERY
            # live window (including whatever the create mints) at dep_wid.
            for w in sources:
                d.setdefault(int(w), set()).add(dep_wid)
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
        # The undone create's id is never minted again: the file's create
        # takes the next one (the window-id high-water mark).
        minted = expected + 1

        cf = tmp_path / "cur.csv"
        cf.write_text(
            "action,window,freqs,params\n"
            f"create,new,{anchor:.6f},\n"
            f"add,{minted},{anchor:.6f},\n"
        )
        result = apply_curation_impl(str(fresh), cf)

        assert result.applied == 2
        log = review_log_impl(str(fresh))
        assert [(e.kind, e.window_id) for e in log] == [
            ("create_window", minted),
            ("add", minted),
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
# 7a. Window ids are never reused, and created ids increase along the log
# ---------------------------------------------------------------------------


class TestWindowIdHighWater:
    def test_an_undone_creates_id_is_never_minted_again(self, working_file):
        """Undoing the newest create leaves no window above the plan, yet a
        fresh create does not take its id: the window-id high-water mark the
        review records survives the undo."""
        anchor = _free_anchor(working_file)
        first = create_window_impl(str(working_file), anchor)
        assert (
            load_stage6_review_from_file(str(working_file)).window_id_high_water
            == first.window_id
        )

        review_undo_impl(str(working_file), [0])
        review = load_stage6_review_from_file(str(working_file))
        assert review.created_windows == []
        assert review.window_id_high_water == first.window_id

        again = create_window_impl(str(working_file), anchor)
        assert again.window_id == first.window_id + 1
        assert again.freq_range == first.freq_range

    def test_review_run_keeps_the_high_water_mark(self, working_file):
        created = create_window_impl(str(working_file), _free_anchor(working_file))
        review_run_impl(str(working_file))
        assert (
            load_stage6_review_from_file(str(working_file)).window_id_high_water
            == created.window_id
        )

    def test_replaying_the_newest_create_keeps_its_id(self, working_file):
        """The trap the mark must not fall into: replaying the newest create
        pins the id the mark itself holds, and that is no conflict."""
        anchor = _free_anchor(working_file)
        created = create_window_impl(str(working_file), anchor)
        s6.review_accept_impl(str(working_file), _live_ranges(working_file)[0][0])

        review_undo_impl(str(working_file), [1])

        log = review_log_impl(str(working_file))
        assert [(e.kind, e.window_id) for e in log] == [
            ("create_window", created.window_id)
        ]

    @pytest.mark.parametrize("log_prefix", [None, 0])
    def test_a_file_pin_may_redo_an_undone_create_under_its_id(
        self, working_file, tmp_path, log_prefix
    ):
        """The mark bounds only the ids a create mints: once the create is
        gone from the log (undone, or cut by a log prefix), a curation file
        may pin its id again."""
        a1 = _free_anchor(working_file)
        created = create_window_impl(str(working_file), a1)
        if log_prefix is None:
            review_undo_impl(str(working_file), [0])

        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{created.window_id},{a1:.6f},\n")
        apply_curation_impl(str(working_file), cf, log_prefix=log_prefix)
        review = load_stage6_review_from_file(str(working_file))
        assert [int(w.window_id) for w in review.created_windows] == [created.window_id]
        assert review.window_id_high_water == created.window_id

    def test_a_file_pin_below_a_live_create_is_refused(self, working_file, tmp_path):
        """A curation file may pin a create's id, but only above every created
        window the log still holds: created ids increase along the log. A
        free id below a live create is refused, though nothing holds it."""
        hi = _live_ranges(working_file)[-1][2]
        first = create_window_impl(str(working_file), hi + 5.0)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"create,{first.window_id + 3},{hi + 15.0:.6f},\n")
        apply_curation_impl(str(working_file), cf)
        before = working_file.read_bytes()

        cf.write_text(f"create,{first.window_id + 1},{hi + 25.0:.6f},\n")
        with pytest.raises(s6.CurationConflictError, match="only increase") as exc:
            apply_curation_impl(str(working_file), cf)
        assert exc.value.reason == "replay_conflict"
        assert exc.value.ids == [first.window_id + 1]
        assert working_file.read_bytes() == before

        cf.write_text(f"create,{first.window_id + 4},{hi + 25.0:.6f},\n")
        apply_curation_impl(str(working_file), cf)
        review = load_stage6_review_from_file(str(working_file))
        assert review.window_id_high_water == first.window_id + 4

    def test_a_log_whose_created_ids_decrease_is_corrupt(self, working_file):
        """Only the engine writes the log, and it mints increasing ids, so a
        log whose created ids decrease is refused as corrupt by a replay."""
        from ftmwpipeline._internal.replay_reference import replay_full
        from ftmwpipeline.file_manager import PipelineCorruptionError

        hi = _live_ranges(working_file)[-1][2]
        r1 = create_window_impl(str(working_file), hi + 5.0)
        r2 = create_window_impl(str(working_file), hi + 15.0)
        s6.review_accept_impl(str(working_file), _live_ranges(working_file)[0][0])
        with h5py.File(str(working_file), "a") as h5f:
            grp = h5f["stage6_review/decision_log"]
            rows = json.loads(grp.attrs["data"])
            assert [r["window_id"] for r in rows[:2]] == [r1.window_id, r2.window_id]
            rows[0]["window_id"], rows[1]["window_id"] = r2.window_id, r1.window_id
            grp.attrs["data"] = json.dumps(rows)
        log = review_log_impl(str(working_file))

        with pytest.raises(PipelineCorruptionError, match="must increase"):
            review_undo_impl(str(working_file), [log[2].serial])
        with pytest.raises(PipelineCorruptionError, match="must increase"):
            replay_full(working_file, log)


# ---------------------------------------------------------------------------
# 7b. An undo reports the windows whose geometry it changes
# ---------------------------------------------------------------------------


class TestUndoGeometryReport:
    def test_an_undo_that_changes_no_structure_reports_nothing(self, working_file):
        anchor = _free_anchor(working_file)
        created = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), created.window_id, add=[anchor])

        assert (
            review_undo_impl(
                str(working_file), [1], dry_run=True
            ).geometry_changed_window_ids
            == []
        )
        assert (
            review_undo_impl(str(working_file), [1]).geometry_changed_window_ids == []
        )

    def test_replanned_creates_that_did_not_move_report_nothing(
        self, working_file, monkeypatch
    ):
        """Undoing a create that a surviving create was planned after replans
        the survivor; a survivor whose geometry comes back unchanged is not
        reported (the persisted overlay round-trips the replan exactly)."""
        hi = _live_ranges(working_file)[-1][2]
        create_window_impl(str(working_file), hi + 5.0)
        second = create_window_impl(str(working_file), hi + 25.0)
        stored = {
            int(w.window_id): s6._window_geometry(w)
            for w in load_stage6_review_from_file(str(working_file)).created_windows
        }
        replans: List[int] = []
        real = s6._walk_log_rows

        def spy(*args, **kwargs):
            replans.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(s6, "_walk_log_rows", spy)
        dry = review_undo_impl(str(working_file), [0], dry_run=True)
        assert replans
        assert dry.geometry_changed_window_ids == []
        assert (
            review_undo_impl(str(working_file), [0]).geometry_changed_window_ids == []
        )

        survivors = load_stage6_review_from_file(str(working_file)).created_windows
        assert [int(w.window_id) for w in survivors] == [second.window_id]
        assert s6._window_geometry(survivors[0]) == stored[second.window_id]

    def test_undoing_a_widening_reports_the_widened_window(self, working_file):
        """A widened window returns to its fitted-plan extent when its
        widening is undone: it still exists, and its geometry changed."""
        step = TestNarrowGapWidens()._bin_width_mhz(working_file)
        wid, _lo, hi = _live_ranges(working_file)[-1]
        widened = create_window_impl(str(working_file), hi + 2 * step)
        assert widened.mode == "widened"

        dry = review_undo_impl(str(working_file), [0], dry_run=True)
        assert dry.geometry_changed_window_ids == [wid]
        result = review_undo_impl(str(working_file), [0])
        assert result.geometry_changed_window_ids == [wid]
        assert load_stage6_review_from_file(str(working_file)).created_windows == []


class TestUndoGeometryReportInterfaces:
    """The geometry report is one implementation: the functional API, the
    Pipeline class and the CLI all carry it."""

    def _widened_copy(self, source: Path, tmp_path: Path, name: str):
        fp = tmp_path / name
        shutil.copy(source, fp)
        review_run_impl(str(fp))
        step = TestNarrowGapWidens()._bin_width_mhz(fp)
        wid, _lo, hi = _live_ranges(fp)[-1]
        assert create_window_impl(str(fp), hi + 2 * step).mode == "widened"
        return fp, wid

    def test_api_and_pipeline_report_the_same_ids(self, stage5_small_source, tmp_path):
        fp_api, wid = self._widened_copy(stage5_small_source, tmp_path, "api.ftmw")
        fp_pipe, _ = self._widened_copy(stage5_small_source, tmp_path, "pipe.ftmw")
        serial = review_log_impl(str(fp_api))[0].serial

        assert ftmw.review_undo(fp_api, [serial]).geometry_changed_window_ids == [wid]
        undone = Pipeline.open(fp_pipe).review_undo([serial])
        assert undone.geometry_changed_window_ids == [wid]

    def test_cli_prints_the_ids_and_counts_them(
        self, stage5_small_source, tmp_path, capsys
    ):
        from ftmwpipeline.cli.main import main as cli_main

        fp, wid = self._widened_copy(stage5_small_source, tmp_path, "cli.ftmw")
        serial = str(review_log_impl(str(fp))[0].serial)

        capsys.readouterr()
        assert cli_main(["review", "undo", str(fp), "--id", serial, "--dry-run"]) == 0
        assert f"geometry changed: window(s) {wid}" in capsys.readouterr().out

        assert cli_main(["review", "undo", str(fp), "--id", serial, "--json"]) == 0
        assert '"n_geometry_changed": 1' in capsys.readouterr().out

    def test_cli_prints_nothing_when_no_geometry_changed(self, working_file, capsys):
        from ftmwpipeline.cli.main import main as cli_main

        anchor = _free_anchor(working_file)
        created = create_window_impl(str(working_file), anchor)
        refit_window_impl(str(working_file), created.window_id, add=[anchor])
        serial = str(review_log_impl(str(working_file))[1].serial)

        capsys.readouterr()
        assert cli_main(["review", "undo", str(working_file), "--id", serial]) == 0
        assert "geometry changed" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7c. The structural layer: windows from the ordered creates, no fit
# ---------------------------------------------------------------------------


class TestStructuralLayer:
    def test_the_overlay_of_a_log_is_planned_without_a_fit(
        self, working_file, monkeypatch
    ):
        """A log's creates replan to exactly the windows the file stores, and
        the planning makes no fit."""
        hi = _live_ranges(working_file)[-1][2]
        create_window_impl(str(working_file), hi + 5.0)
        create_window_impl(str(working_file), hi + 25.0)
        s6.review_accept_impl(str(working_file), _live_ranges(working_file)[0][0])
        log = review_log_impl(str(working_file))
        stored = load_stage6_review_from_file(str(working_file)).created_windows
        shared = s6._build_shared_fit_ctx(str(working_file))

        def no_fit(*args, **kwargs):
            raise AssertionError("the structural layer made a fit")

        monkeypatch.setattr(s6, "refit_window_core", no_fit)
        planned = s6._walk_log_rows(
            str(working_file), shared, log, min_new_window_id=0
        ).overlay

        assert [int(w.window_id) for w in planned] == [int(w.window_id) for w in stored]
        assert [s6._window_geometry(w) for w in planned] == [
            s6._window_geometry(w) for w in stored
        ]

    def test_a_pinned_id_that_is_taken_is_refused_before_any_fit(
        self, working_file, monkeypatch
    ):
        import dataclasses

        create_window_impl(str(working_file), _free_anchor(working_file))
        taken = _live_ranges(working_file)[0][0]
        row = review_log_impl(str(working_file))[0]
        shared = s6._build_shared_fit_ctx(str(working_file))

        def no_fit(*args, **kwargs):
            raise AssertionError("a fit ran before the refusal")

        monkeypatch.setattr(s6, "refit_window_core", no_fit)
        with pytest.raises(s6.CurationConflictError) as exc:
            s6._walk_log_rows(
                str(working_file),
                shared,
                [dataclasses.replace(row, window_id=taken)],
                min_new_window_id=0,
            )
        assert exc.value.reason == "replay_conflict"
        assert exc.value.ids == [taken]

    def test_a_widening_row_is_not_held_to_the_created_id_order(self, working_file):
        """Widen rows carry base ids, below every created id: a log whose
        widening follows a create is not corrupt."""
        step = TestNarrowGapWidens()._bin_width_mhz(working_file)
        wid, _lo, hi = _live_ranges(working_file)[-1]
        created = create_window_impl(str(working_file), hi + 5.0 + 20 * step)
        widened = create_window_impl(str(working_file), hi + 2 * step)
        assert (created.mode, widened.mode) == ("created", "widened")
        assert widened.window_id == wid < created.window_id

        log = review_log_impl(str(working_file))
        s6._check_created_ids_monotone(str(working_file), log)
        assert review_undo_impl(str(working_file), [log[0].serial]).applied == 1


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
        from ftmwpipeline.cli.main import main as cli_main

        p = tmp_path / "bad.ftmw"
        shutil.copy(stage5_small_source, p)
        # bad_setting propagates from the verb; main maps it to exit 1 and
        # writes the error to stderr.
        rc = cli_main(["review", "create", str(p), "--at", "1000.0"])
        assert rc == 1
        assert "Error:" in capsys.readouterr().err


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
