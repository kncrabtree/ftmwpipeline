"""
Tests for ``review_preview()`` -- ``review_preview_impl`` / ``Pipeline.review_
preview`` / ``ftmwpipeline.api.review_preview`` / ``review preview`` (Task C,
``scratch/preview-session-plan.md``, items C1-C6).

The contract (``scratch/bq-correspondence/reply-preview-execute.md`` section
2): run a curation plan to completion in memory, on the same appliers and the
same one combined cascade a live ``review apply`` uses, and report the fitted
outcome WITHOUT persisting -- keyed by window id, read *after* the cascade
(not per-action, since the cascade can supersede an action's own returned
numbers), with final-product numbers (calibrated frequency, the three-term
sigma budget) rather than the raw / stat-only ``RefitWindowResult`` values.

An ``rb_locked``/``uncalibrated`` fixture has ``epsilon == 0``, where raw and
calibrated final-product numbers coincide and a frame regression is
invisible -- every test here that checks the persisted-value contract uses a
``self_calibrated`` fixture, forced by the recipe from
``scratch/preview-session-plan.md`` ("Verified facts for later units").
Duplicated locally rather than imported across test files, matching this
suite's own precedent (``test_frame_parameter.py``'s docstring, ``tests/
AGENTS.md``).

The shared 3-window build in ``conftest.py`` is trimmed to the first three
DEPENDENCY-FREE windows in topological order (see its own docstring), so it
can never exercise the cascade -- the whole point of this module's most
important test. The wider 12-window slice (``test_curation.py``'s
``_build_stage5_multi``, duplicated here for the same reason) is ALSO built
free of real dependency edges on this data (measured in the plan doc:
``plan.dependency_edges == []``). ``test_curation.py`` itself already
established the fix for this: fake a dependency edge by monkeypatching
``_cascade_succs`` (``test_cascade_downstream_of_two_edits_refit_once``) --
the graph is faked, but the refit machinery downstream of it (closure, topo
order, ``_cascade_refit_dependents``) is entirely real. The contract test
below uses that same technique to guarantee a non-trivial cascade actually
fires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    get_final_products_impl,
    review_preview_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_preview
from ftmwpipeline.core.data_structures import FinalPeak
from ftmwpipeline.core.environment import ANALYSIS_EPOCH, capture_environment
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]

EPS = 2.2e-6
SIGMA_EPS = 0.1e-6


# ---------------------------------------------------------------------------
# A wider multi-window, self_calibrated fixture: several live fitted windows
# (needed for the batch engine's cross-window guarantees, and to fake a
# dependency edge between two of them) stamped self_calibrated (needed so a
# frame or final-products regression is actually visible).
# ---------------------------------------------------------------------------


def _build_stage5_multi(dest: Path, data_path: str) -> None:
    """Import 2638, keep up to 12 windows, fit Stage 5. Mirrors
    ``test_curation.py``'s ``_build_stage5_multi`` exactly."""
    ftmw.import_data(dest, source=data_path)
    ftmw.compute_ft(dest, trim=(26500, 40000))
    ftmw.estimate_noise(dest)
    ftmw.detect_peaks(dest)
    ftmw.assign_windows(dest)

    plan = load_windows_impl(str(dest))["plan"]
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= 12:
            break
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(dest), plan)

    ftmw.fit_peaks(str(dest))


def _declare_unlocked_digitizer(path: Path) -> None:
    """Load-modify-save: preserve every other persisted Stage 5 fit setting."""
    persisted = load_stage_fit_settings_from_h5(str(path))
    assert persisted is not None, "Stage 5 must have persisted fit settings"
    new_settings = replace(
        persisted,
        spur=replace(
            persisted.spur,
            clocks=(
                ClockSource(5120.0, locked=True),
                ClockSource(6250.0, locked=False),
            ),
        ),
    )
    save_stage_fit_settings_to_h5(str(path), new_settings)


def _stamp_timebase(path: Path, *, epsilon: float, sigma_epsilon: float) -> None:
    result = TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=sigma_epsilon,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=320.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        grp = h5f.create_group(GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)


@pytest.fixture(scope="session")
def _stage5_multi_sc_built(exp_2638_data_path, tmp_path_factory) -> Path:
    """The shared wider (up to 12-window) post-fit build, stamped
    self_calibrated -- read-only, built once."""
    fp = tmp_path_factory.mktemp("stage6_preview_multi_sc") / "stage5_multi_sc.ftmw"
    _build_stage5_multi(fp, exp_2638_data_path)
    _declare_unlocked_digitizer(fp)
    _stamp_timebase(fp, epsilon=EPS, sigma_epsilon=SIGMA_EPS)
    return fp


@pytest.fixture
def sc_multi_file(_stage5_multi_sc_built: Path, tmp_path: Path) -> Path:
    """A fresh writable copy of the shared self_calibrated multi-window build."""
    fp = tmp_path / "multi_sc.ftmw"
    shutil.copy(_stage5_multi_sc_built, fp)
    return fp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_window_ids(path: Path) -> List[int]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return sorted(
        int(wf.window_id) for wf in sf.window_fits if wf.window_id is not None
    )


def _window_center(path: Path, wid: int) -> float:
    plan = load_windows_impl(str(path))["plan"]
    for w in plan.windows:
        if int(w.window_id) == wid:
            lo, hi = w.freq_range
            return 0.5 * (min(lo, hi) + max(lo, hi))
    raise KeyError(wid)


def _clear_add_freq(path: Path, wid: int) -> float:
    """An in-window frequency that is a legitimate ``add`` target.

    NOT the window center: Stage 4 builds a window around the detection that
    seeded it, so the center is the *birth position* of the peak already fitted
    there. An ``add`` at the center therefore asks for a second peak at an
    existing peak's exact identity -- both are stamped with the same
    ``peak_uid`` and the refit is refused as a duplicate identifier. (Resolving
    a blend is what ``split`` is for.) Pick the in-window position furthest
    from every fitted peak instead, which is what "add a line the detector
    missed" actually means.
    """
    import h5py as _h5py
    import numpy as _np

    from ftmwpipeline._internal.stage4_impl import load_windows_impl as _lw
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5 as _load_sf,
    )

    plan = _lw(str(path))["plan"]
    w = next(x for x in plan.windows if int(x.window_id) == wid)
    lo, hi = w.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    with _h5py.File(str(path), "r") as h5f:
        sf = _load_sf(h5f["stage5_fitting"])
    wf = next((x for x in sf.window_fits if int(x.window_id) == wid), None)
    peaks = [float(p.frequency_mhz) for p in (wf.fitted_peaks if wf else [])]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = _np.linspace(lo, hi, 512)[1:-1]
    dist = _np.min(_np.abs(grid[:, None] - _np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(_np.argmax(dist))])


def _window_stats(path: Path) -> Dict[int, tuple]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): (len(wf.fitted_peaks), float(wf.reduced_chi2))
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _anchor_in_a_gap(path: Path) -> float:
    """A molecular frequency inside the analysis band but outside every planned
    window -- the input ``create`` needs, and the only way to reach a preview
    entry for a window that has no "before" fit."""
    plan = load_windows_impl(str(path))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        if lo2 - hi1 > 4.0:
            return 0.5 * (hi1 + lo2)
    pytest.skip("no gap between planned windows wide enough to create into")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _force_fit_epoch(path: Path, epoch: int) -> None:
    """Rewrite the recorded Stage 5 epoch, simulating a fit from another
    epoch -- same technique as ``test_engine_invariants.py``."""
    with h5py.File(path, "a") as f:
        g = f.require_group("pipeline_stages")
        blob = json.loads(str(g.attrs.get("stage_environments", "{}")))
        rec = capture_environment().to_dict()
        rec["analysis_epoch"] = epoch
        blob["stage5_fitting"] = rec
        g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)


def _assert_final_peaks_equal(a: FinalPeak, b: FinalPeak) -> None:
    """Field-by-field comparison (floats via ``approx``) rather than dataclass
    ``==``, so a same-computation-different-object comparison is not made
    brittle by exact float representation."""
    assert a.window_id == b.window_id
    assert a.origin == b.origin
    assert a.derivation == b.derivation
    assert a.clock_lattice == b.clock_lattice
    for name in (
        "frequency_mhz",
        "frequency_raw_mhz",
        "f_baseband_mhz",
        "sigma_f_khz",
        "sigma_stat_khz",
        "sigma_eps_khz",
        "sigma_floor_khz",
        "amplitude",
    ):
        assert getattr(a, name) == pytest.approx(
            getattr(b, name), rel=1e-9, abs=1e-12
        ), name
    for name in ("phase", "snr", "amplitude_error", "phase_error", "snr_error"):
        va, vb = getattr(a, name), getattr(b, name)
        if va is None or vb is None:
            assert va is vb, name
        else:
            assert va == pytest.approx(vb, rel=1e-9, abs=1e-12), name


# ---------------------------------------------------------------------------
# Fixture sanity: the fixture is actually self_calibrated (an uncalibrated
# fixture would let every test below pass by accident).
# ---------------------------------------------------------------------------


def test_fixture_is_actually_self_calibrated(sc_multi_file: Path) -> None:
    stamp = s6._current_calibration_stamp(str(sc_multi_file))
    assert stamp is not None
    assert stamp[0] == "self_calibrated"
    assert stamp[1] == pytest.approx(EPS)


def test_fixture_has_at_least_two_live_windows(sc_multi_file: Path) -> None:
    assert len(_fitted_window_ids(sc_multi_file)) >= 2


# ---------------------------------------------------------------------------
# C3: epoch gate (same as apply), no undo baseline, no writes.
# ---------------------------------------------------------------------------


class TestEpochGateAndNoWrites:
    def test_preview_writes_no_bytes_on_success(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        before = _digest(sc_multi_file)
        result = review_preview_impl(sc_multi_file, cur, frame="raw")
        after = _digest(sc_multi_file)

        assert after == before, "a successful preview must not change any bytes"
        assert wid in result.windows

    def test_preview_refuses_across_epoch_mismatch_and_writes_nothing(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        _force_fit_epoch(sc_multi_file, ANALYSIS_EPOCH + 1)
        before = _digest(sc_multi_file)

        with pytest.raises(ValueError, match="analysis epoch"):
            review_preview_impl(sc_multi_file, cur, frame="raw")

        assert _digest(sc_multi_file) == before
        with h5py.File(sc_multi_file, "r") as f:
            assert (
                "stage5_fitting_baseline" not in f
            ), "a refused preview must not take the undo baseline either"

    def test_bare_accept_only_plan_is_not_gated(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """A bare-accept-only plan touches no fit, so -- exactly like the live
        apply's own bare-accept path -- it is not epoch-gated at all."""
        wid = _fitted_window_ids(sc_multi_file)[0]
        cur = tmp_path / "cur.csv"
        cur.write_text(f"accept,{wid},,\n")

        _force_fit_epoch(sc_multi_file, ANALYSIS_EPOCH + 1)
        before = _digest(sc_multi_file)

        result = review_preview_impl(sc_multi_file, cur, frame="raw")

        assert result.windows == {}
        assert _digest(sc_multi_file) == before


# ---------------------------------------------------------------------------
# Structural guard: a context built with snapshot=False (a preview) must be
# refused by _finish_batch, not merely documented as preview-only. Without
# this, a future caller could pair snapshot=False with _finish_batch and
# persist edits with no undo baseline -- 'review undo' would then silently
# break for that file (a later edit would snapshot an already-edited fit as
# if it were the automatic one). See _BatchCtx.baseline_taken.
# ---------------------------------------------------------------------------


class TestUnbaselinedContextCannotBePersisted:
    def test_finish_batch_refuses_a_snapshot_false_context(
        self, sc_multi_file: Path
    ) -> None:
        ctx = s6._open_batch(
            str(sc_multi_file),
            snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
            snapshot=False,
        )
        assert ctx.baseline_taken is False

        before = _digest(sc_multi_file)
        with pytest.raises(ValueError, match="undo baseline was.*never taken"):
            s6._finish_batch(
                ctx,
                str(sc_multi_file),
                snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
            )
        assert _digest(sc_multi_file) == before, (
            "a refused _finish_batch must not have written anything before " "raising"
        )
        with h5py.File(sc_multi_file, "r") as f:
            assert "stage5_fitting_baseline" not in f

    def test_finish_batch_accepts_a_snapshot_true_context(
        self, sc_multi_file: Path
    ) -> None:
        """Sanity check: the guard is specific to snapshot=False, not a
        blanket refusal -- the normal (snapshot=True) path still persists."""
        ctx = s6._open_batch(
            str(sc_multi_file),
            snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
            snapshot=True,
        )
        assert ctx.baseline_taken is True
        s6._finish_batch(
            ctx,
            str(sc_multi_file),
            snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
        )
        with h5py.File(sc_multi_file, "r") as f:
            assert "stage5_fitting_baseline" in f


# ---------------------------------------------------------------------------
# C5: an entirely-bare-accept plan does no fits, writes nothing, returns an
# empty result -- short-circuited before the live apply's write-bearing
# per-action path (_execute_planned_action) is ever reached.
# ---------------------------------------------------------------------------


class TestBareAcceptOnlyShortCircuit:
    def test_bare_accept_only_plan_returns_empty_and_writes_nothing(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wids = _fitted_window_ids(sc_multi_file)
        cur = tmp_path / "cur.csv"
        cur.write_text("\n".join(f"accept,{w},," for w in wids[:2]) + "\n")

        before = _digest(sc_multi_file)
        result = review_preview_impl(sc_multi_file, cur, frame="raw")

        assert result.windows == {}
        assert _digest(sc_multi_file) == before
        # Confirm this is not a false negative from a bad curation file: the
        # live apply of the SAME file really does apply 2 (bare) actions.
        applied = apply_curation_impl(sc_multi_file, cur, frame="raw")
        assert applied.applied == 2

    def test_empty_plan_returns_empty_and_writes_nothing(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur = tmp_path / "empty.csv"
        cur.write_text("")

        before = _digest(sc_multi_file)
        result = review_preview_impl(sc_multi_file, cur, frame="raw")

        assert result.windows == {}
        assert result.plan == []
        assert _digest(sc_multi_file) == before


# ---------------------------------------------------------------------------
# C4: per-action failure attribution, matching apply's own message exactly.
# ---------------------------------------------------------------------------


class TestPerActionFailureAttribution:
    def test_failure_is_tagged_with_action_index_and_matches_apply(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        # Nowhere near any fitted peak in this window -> guaranteed failure.
        cur = tmp_path / "cur.csv"
        cur.write_text(f"remove,{wid},99999.0,\n")

        preview_copy = tmp_path / "preview.ftmw"
        apply_copy = tmp_path / "apply.ftmw"
        shutil.copy(sc_multi_file, preview_copy)
        shutil.copy(sc_multi_file, apply_copy)

        with pytest.raises(ValueError) as exc_preview:
            review_preview_impl(preview_copy, cur, frame="raw")
        with pytest.raises(ValueError) as exc_apply:
            apply_curation_impl(apply_copy, cur, frame="raw")

        assert "curation action 1" in str(exc_preview.value)
        assert str(exc_preview.value) == str(exc_apply.value)
        # Failure leaves the preview copy exactly as it started.
        assert _digest(preview_copy) == _digest(sc_multi_file)


# ---------------------------------------------------------------------------
# Cross-interface consistency (impl / Pipeline / api / CLI).
# ---------------------------------------------------------------------------


class TestCrossInterfaceConsistency:
    def test_impl_pipeline_api_cli_agree_and_write_nothing(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)

        paths: Dict[str, Path] = {}
        for name in ("impl", "pipe", "api", "cli"):
            p = tmp_path / f"{name}.ftmw"
            shutil.copy(sc_multi_file, p)
            paths[name] = p

        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        before = {k: _digest(p) for k, p in paths.items()}

        r_impl = review_preview_impl(paths["impl"], cur, frame="raw")
        r_pipe = Pipeline.open(paths["pipe"]).review_preview(cur, frame="raw")
        r_api = ftmw.review_preview(str(paths["api"]), cur, frame="raw")
        rc = cmd_review_preview(
            argparse.Namespace(
                file_path=str(paths["cli"]),
                curation_file=str(cur),
                frame="raw",
                verbose=False,
            )
        )
        assert rc == 0

        for k, p in paths.items():
            assert _digest(p) == before[k], f"{k}: preview wrote bytes"

        assert wid in r_impl.windows
        for r in (r_pipe, r_api):
            assert set(r.windows) == set(r_impl.windows)
            for w in r_impl.windows:
                a, b = r.windows[w], r_impl.windows[w]
                assert a.origin == b.origin
                assert a.action_indices == b.action_indices
                assert a.n_peaks_before == b.n_peaks_before
                assert a.n_peaks_after == b.n_peaks_after
                assert a.chi2r_before == pytest.approx(b.chi2r_before)
                assert a.chi2r_after == pytest.approx(b.chi2r_after)
                assert len(a.peaks) == len(b.peaks)
                for pa, pb in zip(a.peaks, b.peaks):
                    _assert_final_peaks_equal(pa, pb)

        # The CLI path returned nothing, but it wrote nothing either, and a
        # direct call against the (untouched) CLI-run file must agree too.
        r_cli = review_preview_impl(paths["cli"], cur, frame="raw")
        assert set(r_cli.windows) == set(r_impl.windows)

    def test_omitted_frame_raises_on_self_calibrated(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")
        with pytest.raises(ValueError, match="frame is required"):
            review_preview_impl(sc_multi_file, cur)


# ---------------------------------------------------------------------------
# The plan field: a client can join action_indices back through it, and it
# agrees with what a dry-run apply of the identical curation file resolves.
# ---------------------------------------------------------------------------


class TestPlanField:
    def test_plan_matches_apply_dry_run_plan(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        preview = review_preview_impl(sc_multi_file, cur, frame="raw")
        dry = apply_curation_impl(sc_multi_file, cur, dry_run=True, frame="raw")

        assert len(preview.plan) == len(dry.plan)
        for pa, pb in zip(preview.plan, dry.plan):
            assert pa == pb


# ---------------------------------------------------------------------------
# C6, the contract test: preview output equals the subsequent apply's
# persisted products -- on a self_calibrated fixture, with a GENUINE cascade.
#
# Neither the 3-window nor the 12-window 2638 slice has real dependency
# edges (measured in the plan doc: plan.dependency_edges == []), so the
# cascade is faked by monkeypatching _cascade_succs -- the same technique
# test_curation.py's own test_cascade_downstream_of_two_edits_refit_once
# uses for the identical reason. Everything downstream of the graph (closure,
# topological order, the actual re-fit) is real.
# ---------------------------------------------------------------------------


class TestPreviewEqualsApplyWithGenuineCascade:
    def _fake_edge(
        self, monkeypatch: pytest.MonkeyPatch, primary: int, dep: int
    ) -> None:
        orig_succs = s6._cascade_succs

        def fake_succs(window_fits, fit_window_map):
            d = orig_succs(window_fits, fit_window_map)
            d.setdefault(primary, set()).add(dep)
            return d

        monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

    def test_pure_cascaded_dependent_matches_persisted_apply(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wids = _fitted_window_ids(sc_multi_file)
        w0, w2 = wids[0], wids[1]
        self._fake_edge(monkeypatch, w0, w2)

        preview_path = tmp_path / "preview.ftmw"
        apply_path = tmp_path / "apply.ftmw"
        shutil.copy(sc_multi_file, preview_path)
        shutil.copy(sc_multi_file, apply_path)

        freq0 = _window_center(preview_path, w0)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{w0},{freq0},\n")

        before = _digest(preview_path)
        preview = review_preview_impl(preview_path, cur, frame="raw")
        assert _digest(preview_path) == before, "preview must write nothing"

        # The cascade actually fired -- the whole point of this test.
        assert w2 in preview.windows, "the faked dependent never got refitted"
        assert preview.windows[w2].origin == "cascaded"
        assert preview.windows[w2].action_indices == []
        assert w0 in preview.windows
        assert preview.windows[w0].origin == "direct"
        assert preview.windows[w0].action_indices == [0]

        applied = apply_curation_impl(apply_path, cur, frame="raw")
        assert applied.applied == 1

        fp = get_final_products_impl(apply_path)
        assert fp is not None
        persisted_by_window: Dict[int, List[FinalPeak]] = {}
        for peak in fp.peaks:
            if peak.window_id is not None:
                persisted_by_window.setdefault(int(peak.window_id), []).append(peak)
        persisted_stats = _window_stats(apply_path)

        assert set(preview.windows) <= set(persisted_stats)
        for wid, pw in preview.windows.items():
            n_after, chi2r_after = persisted_stats[wid]
            assert pw.n_peaks_after == n_after
            assert pw.chi2r_after == pytest.approx(chi2r_after, rel=1e-9)

            got = sorted(pw.peaks, key=lambda p: p.frequency_mhz)
            want = sorted(
                persisted_by_window.get(wid, []), key=lambda p: p.frequency_mhz
            )
            assert len(got) == len(want)
            for a, b in zip(got, want):
                _assert_final_peaks_equal(a, b)

    def test_direct_edit_downstream_of_sibling_edit_uses_post_cascade_numbers(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the plan calls out specifically: a window that is BOTH
        directly edited by its own action AND downstream of another
        directly-edited window in the same batch. ``_cascade_closure`` adds
        it back into the cascade pass so its background is refreshed from
        its sibling -- the preview must report the numbers from that SECOND
        (cascade) refit, not the applier's own first-pass return value.

        On this fixture the faked edge carries no real frozen-parameter
        linkage, so a second (identity) refit of w2 can converge to numbers
        indistinguishable from the first -- comparing against apply's
        persisted output alone would not catch a preview that silently kept
        the pre-cascade result, because apply's OWN cascade would drift the
        same way if it were broken too (a mutation probe confirmed this: a
        neutered ``_cascade_batch`` left the origin/apply-agreement
        assertions passing). So this test tags w2's SECOND ``refit_window_
        core`` call (the cascade pass) with a detectable marker and asserts
        the preview reports exactly that pass's number -- a check that fails
        outright if the cascade never revisits w2 a second time.
        """
        wids = _fitted_window_ids(sc_multi_file)
        w0, w2 = wids[0], wids[1]
        self._fake_edge(monkeypatch, w0, w2)

        preview_path = tmp_path / "preview.ftmw"
        apply_path = tmp_path / "apply.ftmw"
        shutil.copy(sc_multi_file, preview_path)
        shutil.copy(sc_multi_file, apply_path)

        freq0 = _clear_add_freq(preview_path, w0)
        freq2 = _clear_add_freq(preview_path, w2)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{w0},{freq0},\nadd,{w2},{freq2},\n")

        orig_core = s6.refit_window_core
        w2_calls = {"n": 0}
        MARKER = 123456.0

        def spy_core(fit_ctx, fit_win, wf, **kwargs):
            result = orig_core(fit_ctx, fit_win, wf, **kwargs)
            if int(fit_win.window_id) == w2:
                w2_calls["n"] += 1
                if w2_calls["n"] >= 2:
                    # The SECOND call for w2 is the cascade's re-fit (the
                    # first is w2's own direct "add"). Stamp a value the
                    # first call could never have produced, so the assertion
                    # below can tell which pass's number the preview
                    # actually reports.
                    result.reduced_chi2 = MARKER
            return result

        monkeypatch.setattr(s6, "refit_window_core", spy_core)

        preview = review_preview_impl(preview_path, cur, frame="raw")

        assert w2_calls["n"] >= 2, (
            "w2 was not refit a second time -- the cascade never revisited "
            "a directly-edited window downstream of its sibling's edit, so "
            "this test cannot exercise the guarantee it names"
        )
        assert w2 in preview.windows
        assert preview.windows[w2].origin == "direct"
        assert preview.windows[w2].action_indices == [1]
        # The load-bearing assertion: the reported chi2r is the SECOND
        # (post-cascade) pass's marked value, not the first applier call's
        # own (unmarked) return.
        assert preview.windows[w2].chi2r_after == pytest.approx(MARKER)

        # Apply, under the identical patch, persists the same marked value --
        # preview and apply agree on which pass's number is the real one.
        apply_curation_impl(apply_path, cur, frame="raw")
        persisted_stats = _window_stats(apply_path)
        assert persisted_stats[w2][1] == pytest.approx(MARKER)


class TestAbsentFitReportsNoneNotZero:
    """A window with no fit on one side reports ``None``, never a fabricated
    ``0.0`` (BlackQuill ask of 2026-08-19).

    chi2r = 0.0 is a value a genuine fit essentially never produces, so a
    fabricated one is indistinguishable from an extraordinary one: a consumer
    rendering "before -> after" shows ``0.00 -> 1.4`` and reads it as a perfect
    fit that got worse.
    """

    def test_dataclass_defaults_are_absent_not_zero(self) -> None:
        """The default itself is the contract -- a 0.0 default would reintroduce
        the fabrication wherever a field is not explicitly set."""
        entry = s6.PreviewWindowResult(window_id=1, origin="direct")
        assert entry.chi2r_before is None
        assert entry.chi2r_after is None

    def test_created_window_has_no_before_chi2r(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """A window the batch itself creates never had a "before" fit."""
        anchor = _anchor_in_a_gap(sc_multi_file)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"create,new,{anchor:.6f},\n")

        result = review_preview_impl(sc_multi_file, cur, frame="raw")

        created = [w for w in result.windows.values() if w.origin == "direct"]
        assert created, "the create action should report a direct window"
        for w in created:
            assert w.chi2r_before is None, (
                f"window {w.window_id} reports chi2r_before="
                f"{w.chi2r_before!r} for a fit that never existed"
            )

    def test_an_ordinary_edit_still_reports_both_sides(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """The guard is specific to a missing fit, not a blanket None: a window
        that existed before and after still carries two real numbers."""
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        result = review_preview_impl(sc_multi_file, cur, frame="raw")

        w = result.windows[wid]
        assert w.chi2r_before is not None and w.chi2r_before > 0.0
        assert w.chi2r_after is not None and w.chi2r_after > 0.0
