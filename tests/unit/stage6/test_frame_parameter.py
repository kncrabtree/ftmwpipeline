"""
Tests for the A2/A6 ``frame`` parameter (raw <-> calibrated), added in
``af5df9f`` with no tests of its own (see ``scratch/preview-session-plan.md``,
"Task A-core" and its HANDOFF block).

An ``rb_locked``/``uncalibrated`` fixture has ``epsilon == 0``, where the raw
and calibrated frames coincide and every frame bug is invisible -- so every
test that matters here uses a ``self_calibrated`` fixture, forced by the
recipe from ``scratch/preview-session-plan.md`` ("Verified facts for later
units"): declare an unlocked digitizer clock, then stamp a passing
``TimebaseCalibrationResult`` directly. (Duplicated locally rather than
imported from ``test_final_products_staleness.py`` -- each test file states
its own fixture contract, matching that file's own precedent.)

This module checks five claims from the handoff, each treated as a hypothesis
that might be false rather than as an established fact:

1. Conversion to raw happens before snapping -- a calibrated-frame input
   lands on the same peak a correctly-converted raw input would
   (``TestConversionBeforeSnapping``).
2. Omitting ``frame`` on a frequency-bearing call errors on a
   ``self_calibrated`` file; an explicit ``frame="raw"`` is always accepted;
   omitted and explicit-raw are genuinely distinguishable
   (``TestOmittedFrameRequired``).
3. Nothing calibrated reaches persisted state -- decision log, created
   windows -- and a later eps change does not perturb an already-persisted
   decision or its replay (``TestNothingCalibratedIsPersisted``).
4. CLI, ``Pipeline`` and the functional API agree on all of the above
   (``TestCrossInterfaceFrameConsistency``).
5. Every returned frequency is labeled and carries both frames plus
   ``calibration_state`` and the applied ``(epsilon, sigma_epsilon)``, and the
   two frames actually differ on a ``self_calibrated`` file
   (``TestReturnedFrequenciesLabeled``).

Also ``TestFrameConversionArithmetic``: the conversion primitives themselves,
checked against an independently re-derived reference formula rather than
each other, so a sign or formula error in the implementation cannot hide
behind a circular check.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    FittingResult,
    _current_calibration_stamp,
    _frame_to_calibrated,
    _frame_to_raw,
    apply_curation_impl,
    create_window_impl,
    merge_peaks_impl,
    refit_window_impl,
    review_accept_impl,
    review_undo_impl,
    split_peak_impl,
)
from ftmwpipeline.cli.review_commands import (
    cmd_review_create,
    cmd_review_edit,
)
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
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

EPS_1 = 2.2e-6
EPS_2 = 8.8e-6
SIGMA_EPS = 0.1e-6


# ---------------------------------------------------------------------------
# self_calibrated fixture recipe (scratch/preview-session-plan.md, proven in
# test_final_products_staleness.py).
# ---------------------------------------------------------------------------


def _declare_unlocked_digitizer(path: Path) -> None:
    """Declare an unlocked digitizer clock, preserving every other persisted
    Stage 5 fit setting -- load-modify-save, never a fresh ``resolve()``."""
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
    """Overwrite ``/timebase_calibration`` with a passing result -- exactly
    what a real timebase re-run persists, and nothing else."""
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


def _make_self_calibrated(
    path: Path, *, epsilon: float, sigma_epsilon: float = SIGMA_EPS
) -> None:
    _declare_unlocked_digitizer(path)
    _stamp_timebase(path, epsilon=epsilon, sigma_epsilon=sigma_epsilon)


@pytest.fixture(scope="session")
def _stage5_sc_small_built(_stage5_small_built: Path, tmp_path_factory) -> Path:
    """The shared post-fit small build (from ``conftest.py``), stamped
    ``self_calibrated`` at ``EPS_1`` -- read-only, built once."""
    fp = tmp_path_factory.mktemp("stage6_frame_sc") / "stage5_sc_small.ftmw"
    shutil.copy(_stage5_small_built, fp)
    _make_self_calibrated(fp, epsilon=EPS_1)
    return fp


@pytest.fixture
def sc_file(_stage5_sc_small_built: Path, tmp_path: Path) -> Path:
    """A fresh writable copy of the self_calibrated small build."""
    fp = tmp_path / "sc.ftmw"
    shutil.copy(_stage5_sc_small_built, fp)
    return fp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _first_fitted_window(path: Path) -> FittingResult:
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.fitted_peaks:
            return wf
    raise AssertionError("fixture has no fitted window")


def _ref_calibrated(f_raw: float, *, probe: float, eps: float) -> float:
    """Independent re-derivation of ``f_corr = probe + (f_raw-probe)/(1+eps)``
    -- used to compute expected test values without calling the
    implementation's own conversion helpers, so a formula bug in the
    implementation cannot hide behind a circular check."""
    return probe + (f_raw - probe) / (1.0 + eps)


def _live_ranges(path: Path) -> List[Tuple[int, float, float]]:
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
    """A frequency well clear of every fitted window (and inside the trim) --
    same recipe as ``test_create_window.py``."""
    live = _live_ranges(path)
    assert live, "fixture has no fitted windows"
    return live[-1][2] + 5.0


# ---------------------------------------------------------------------------
# The conversion primitives, checked against an independent reference.
# ---------------------------------------------------------------------------


class TestFrameConversionArithmetic:
    PROBE = 40960.0
    F_RAW = 26613.58186731606

    def test_frame_to_calibrated_matches_reference_formula(self) -> None:
        for eps in (0.0, 2.2e-6, 8.8e-6, -3.5e-6):
            expected = _ref_calibrated(self.F_RAW, probe=self.PROBE, eps=eps)
            got = _frame_to_calibrated(
                self.F_RAW, probe_freq_mhz=self.PROBE, epsilon=eps
            )
            assert got == pytest.approx(expected, rel=1e-12)

    def test_frame_to_raw_inverts_frame_to_calibrated(self) -> None:
        eps = 2.2e-6
        stamp = ("self_calibrated", eps, 0.1e-6, 5.0, self.PROBE, "lower")
        f_cal = _ref_calibrated(self.F_RAW, probe=self.PROBE, eps=eps)
        got = _frame_to_raw(f_cal, frame="calibrated", stamp=stamp)
        assert got == pytest.approx(self.F_RAW, abs=1e-9)

    def test_frame_to_raw_identity_when_frame_raw(self) -> None:
        stamp = ("self_calibrated", 2.2e-6, 0.1e-6, 5.0, self.PROBE, "lower")
        assert _frame_to_raw(self.F_RAW, frame="raw", stamp=stamp) == self.F_RAW

    def test_frame_to_raw_identity_when_epsilon_zero(self) -> None:
        stamp = ("rb_locked", 0.0, 0.0, 5.0, self.PROBE, "lower")
        assert _frame_to_raw(12345.0, frame="calibrated", stamp=stamp) == 12345.0

    def test_frame_to_raw_identity_when_stamp_none(self) -> None:
        assert _frame_to_raw(12345.0, frame="calibrated", stamp=None) == 12345.0

    def test_round_trip_raw_to_calibrated_to_raw(self) -> None:
        eps = 3.7e-6
        f_cal = _frame_to_calibrated(self.F_RAW, probe_freq_mhz=self.PROBE, epsilon=eps)
        stamp = ("self_calibrated", eps, 0.1e-6, 5.0, self.PROBE, "lower")
        back = _frame_to_raw(f_cal, frame="calibrated", stamp=stamp)
        assert back == pytest.approx(self.F_RAW, abs=1e-9)

    def test_disagreement_is_exact_and_under_snap_tolerance(self) -> None:
        """The silent-bug magnitude: a calibrated frequency submitted as raw
        (i.e. omitted conversion) lands off by ``(probe-f_raw)*eps/(1+eps)``
        -- exact, not first-order -- and that offset is deliberately under
        the 50 kHz snap tolerance for this fixture's numbers."""
        eps = 2.2e-6
        f_cal = _frame_to_calibrated(self.F_RAW, probe_freq_mhz=self.PROBE, epsilon=eps)
        expected_delta = (self.PROBE - self.F_RAW) * eps / (1.0 + eps)
        assert (f_cal - self.F_RAW) == pytest.approx(expected_delta, rel=1e-9)
        assert 0.010 < abs(f_cal - self.F_RAW) < 0.050


# ---------------------------------------------------------------------------
# Claim 1: conversion happens before snapping.
# ---------------------------------------------------------------------------


class TestConversionBeforeSnapping:
    def test_calibrated_input_lands_on_same_peak_as_correctly_converted_raw(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        assert len(wf.fitted_peaks) >= 2
        target = min(wf.fitted_peaks, key=lambda p: float(p.frequency_mhz))
        f_raw = float(target.frequency_mhz)

        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None and stamp[0] == "self_calibrated"
        eps, probe = stamp[1], stamp[4]
        f_cal = _ref_calibrated(f_raw, probe=probe, eps=eps)
        # Sanity: the disagreement must be real but under the snap tolerance,
        # or this test would not exercise the silent-bug regime at all.
        assert 0.010 < abs(f_cal - f_raw) < 0.050

        path_raw = tmp_path / "correct_raw.ftmw"
        path_cal = tmp_path / "correct_cal.ftmw"
        path_wrong = tmp_path / "wrong_frame.ftmw"
        shutil.copy(sc_file, path_raw)
        shutil.copy(sc_file, path_cal)
        shutil.copy(sc_file, path_wrong)

        n_before = len(wf.fitted_peaks)

        # Correct: the true raw frequency, frame="raw".
        result_raw = refit_window_impl(
            str(path_raw), wf.window_id, remove=[f_raw], frame="raw"
        )
        # Correct: the calibrated equivalent, frame="calibrated" -- must
        # convert to the same raw frequency before the applier ever sees it.
        result_cal = refit_window_impl(
            str(path_cal), wf.window_id, remove=[f_cal], frame="calibrated"
        )
        # Wrong: the calibrated equivalent declared raw -- simulates omitting
        # the conversion. Still resolves to the SAME peak (that is exactly
        # the silent-failure mode the plan describes: the 50 kHz snap
        # tolerance forgives the ~31.6 kHz disagreement), but its own record
        # of what it removed is wrong.
        result_wrong = refit_window_impl(
            str(path_wrong), wf.window_id, remove=[f_cal], frame="raw"
        )

        assert result_raw.n_peaks_after == n_before - 1
        assert result_cal.n_peaks_after == n_before - 1
        assert result_wrong.n_peaks_after == n_before - 1

        # The correctly-converted calibrated input reproduces the raw input's
        # result: same underlying raw frequency reaches the applier, so the
        # deterministic NLS refit lands in the same place.
        freqs_raw = sorted(float(p.frequency_mhz) for p in result_raw.fitted_peaks)
        freqs_cal = sorted(float(p.frequency_mhz) for p in result_cal.fitted_peaks)
        assert len(freqs_raw) == len(freqs_cal)
        for a, b in zip(freqs_raw, freqs_cal):
            assert a == pytest.approx(b, abs=1e-6)

        # The decision log is where the difference becomes visible: it
        # records the caller's (converted, pre-snap) frequency verbatim.
        log_raw = load_stage6_review_from_file(str(path_raw)).decision_log
        log_cal = load_stage6_review_from_file(str(path_cal)).decision_log
        log_wrong = load_stage6_review_from_file(str(path_wrong)).decision_log

        rec_raw = next(e for e in log_raw if e.kind == "remove")
        rec_cal = next(e for e in log_cal if e.kind == "remove")
        rec_wrong = next(e for e in log_wrong if e.kind == "remove")

        assert rec_raw.frequency_mhz == pytest.approx(f_raw, abs=1e-6)
        assert rec_cal.frequency_mhz == pytest.approx(f_raw, abs=1e-6)

        expected_delta = (probe - f_raw) * eps / (1.0 + eps)
        assert (rec_wrong.frequency_mhz - f_raw) == pytest.approx(
            expected_delta, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Claim 2: omitting frame errors on self_calibrated; explicit "raw" never
# does; omitted and explicit-raw are genuinely distinguishable.
# ---------------------------------------------------------------------------


class TestOmittedFrameRequired:
    def test_edit_add_omitted_frame_raises_on_self_calibrated(
        self, sc_file: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        add_freq = float(wf.fitted_peaks[0].frequency_mhz) + 0.3
        with pytest.raises(ValueError, match="frame is required"):
            refit_window_impl(str(sc_file), wf.window_id, add=[add_freq])

    def test_edit_add_explicit_raw_never_refused(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        add_freq = float(wf.fitted_peaks[0].frequency_mhz) + 0.3
        result = refit_window_impl(
            str(sc_file), wf.window_id, add=[add_freq], frame="raw"
        )
        assert result is not None

    def test_edit_no_frequencies_omitted_frame_is_inert(self, sc_file: Path) -> None:
        """No add/remove carries no frequency at all, so frame is moot even
        on a self_calibrated file -- an identity refit must not raise."""
        wf = _first_fitted_window(sc_file)
        result = refit_window_impl(str(sc_file), wf.window_id)
        assert result is not None

    def test_omitted_frame_is_inert_on_rb_locked_file(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        fp = tmp_path / "rb.ftmw"
        shutil.copy(stage5_small_source, fp)
        wf = _first_fitted_window(fp)
        add_freq = float(wf.fitted_peaks[0].frequency_mhz) + 0.3
        # Must NOT raise: epsilon == 0 on an rb_locked/uncalibrated file, so
        # the choice of frame has no consequence and omission is inert --
        # this is exactly why an rb_locked fixture cannot catch a frame bug.
        result = refit_window_impl(str(fp), wf.window_id, add=[add_freq])
        assert result is not None

    def test_create_omitted_frame_raises_on_self_calibrated(
        self, sc_file: Path
    ) -> None:
        anchor = _free_anchor(sc_file)
        with pytest.raises(ValueError, match="frame is required"):
            create_window_impl(str(sc_file), anchor)

    def test_create_explicit_raw_never_refused(self, sc_file: Path) -> None:
        anchor = _free_anchor(sc_file)
        result = create_window_impl(str(sc_file), anchor, frame="raw")
        assert result is not None

    def test_merge_omitted_frame_raises_on_self_calibrated(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks[:2]]
        assert len(peaks) == 2
        with pytest.raises(ValueError, match="frame is required"):
            merge_peaks_impl(str(sc_file), wf.window_id, peaks)

    def test_merge_explicit_raw_never_refused(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks[:2]]
        result = merge_peaks_impl(str(sc_file), wf.window_id, peaks, frame="raw")
        assert result is not None

    def test_split_omitted_frame_raises_on_self_calibrated(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        peak = float(wf.fitted_peaks[0].frequency_mhz)
        with pytest.raises(ValueError, match="frame is required"):
            split_peak_impl(str(sc_file), wf.window_id, peak, into=2)

    def test_split_explicit_raw_never_refused(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        peak = float(wf.fitted_peaks[0].frequency_mhz)
        result = split_peak_impl(str(sc_file), wf.window_id, peak, into=2, frame="raw")
        assert result is not None

    def test_accept_with_candidate_omitted_frame_raises(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        candidate = float(wf.fitted_peaks[0].frequency_mhz) + 0.2
        with pytest.raises(ValueError, match="frame is required"):
            review_accept_impl(str(sc_file), wf.window_id, candidate_freq=candidate)

    def test_accept_with_candidate_explicit_raw_never_refused(
        self, sc_file: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        candidate = float(wf.fitted_peaks[0].frequency_mhz) + 0.2
        result = review_accept_impl(
            str(sc_file), wf.window_id, candidate_freq=candidate, frame="raw"
        )
        assert result is not None

    def test_bare_accept_never_requires_frame(self, sc_file: Path) -> None:
        wf = _first_fitted_window(sc_file)
        result = review_accept_impl(str(sc_file), wf.window_id)
        assert result is None

    def test_apply_curation_omitted_frame_raises_when_plan_has_freq(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        peak = float(wf.fitted_peaks[0].frequency_mhz)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"remove,{wf.window_id},{peak},\n")
        with pytest.raises(ValueError, match="frame is required"):
            apply_curation_impl(str(sc_file), cur)

    def test_apply_curation_explicit_raw_never_refused(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        peak = float(wf.fitted_peaks[0].frequency_mhz)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"remove,{wf.window_id},{peak},\n")
        result = apply_curation_impl(str(sc_file), cur, frame="raw")
        assert result.applied == 1

    def test_apply_curation_bare_accept_only_never_requires_frame(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wf = _first_fitted_window(sc_file)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"accept,{wf.window_id},,\n")
        result = apply_curation_impl(str(sc_file), cur)
        assert result.applied == 1


# ---------------------------------------------------------------------------
# Claim 3: nothing calibrated reaches persisted state.
# ---------------------------------------------------------------------------


class TestNothingCalibratedIsPersisted:
    def test_created_window_and_decision_log_are_raw_and_immune_to_later_eps(
        self, sc_file: Path
    ) -> None:
        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None
        eps, probe = stamp[1], stamp[4]
        anchor_raw = _free_anchor(sc_file)
        anchor_cal = _ref_calibrated(anchor_raw, probe=probe, eps=eps)
        assert anchor_cal != pytest.approx(anchor_raw, rel=1e-9)

        result = create_window_impl(str(sc_file), anchor_cal, frame="calibrated")
        assert result.anchor_mhz == pytest.approx(anchor_raw, abs=1e-6)
        # It must NOT have stored the calibrated value verbatim.
        assert result.anchor_mhz != pytest.approx(anchor_cal, rel=1e-9)

        review = load_stage6_review_from_file(str(sc_file))
        dec = next(e for e in review.decision_log if e.kind == "create_window")
        assert dec.frequency_mhz == pytest.approx(anchor_raw, abs=1e-6)

        created = review.created_windows
        assert len(created) == 1
        original_range = created[0].freq_range

        # A later timebase re-run must not perturb the persisted decision or
        # the persisted window -- storage stays raw regardless of what the
        # current epsilon is.
        _stamp_timebase(sc_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

        review_after = load_stage6_review_from_file(str(sc_file))
        dec_after = next(
            e for e in review_after.decision_log if e.kind == "create_window"
        )
        assert dec_after.frequency_mhz == dec.frequency_mhz
        assert review_after.created_windows[0].freq_range == original_range

    def test_undo_replay_uses_raw_frame_immune_to_intervening_eps_change(
        self, sc_file: Path
    ) -> None:
        """The good test the task spec suggests: perform a curation action
        with frame="calibrated", change eps, and confirm the persisted
        decision still replays to the same raw frequency. Undo forces a
        replay of every surviving decision (``_execute_planned_action``
        always replays with ``frame="raw"`` straight off the raw-by-
        construction decision log), so this exercises the actual replay path
        rather than just re-reading unchanged bytes."""
        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None
        eps, probe = stamp[1], stamp[4]
        anchor_raw = _free_anchor(sc_file)
        anchor_cal = _ref_calibrated(anchor_raw, probe=probe, eps=eps)

        # Decision 0: create a window via a CALIBRATED-frame anchor.
        created = create_window_impl(str(sc_file), anchor_cal, frame="calibrated")
        wid = created.window_id
        original_range = created.freq_range

        # Decision 1: an edit on a DIFFERENT, already-fitted window, so
        # undoing it does not orphan the create.
        other_wf = _first_fitted_window(sc_file)
        assert other_wf.window_id != wid
        refit_window_impl(
            str(sc_file),
            other_wf.window_id,
            add=[float(other_wf.fitted_peaks[0].frequency_mhz) + 0.3],
            frame="raw",
        )

        log_before = load_stage6_review_from_file(str(sc_file)).decision_log
        assert [e.kind for e in log_before] == ["create_window", "add"]

        # Bump eps between the original decisions and the undo/replay.
        _stamp_timebase(sc_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

        # Undo the edit (decision 1), forcing decision 0 to replay from the
        # automatic-fit baseline.
        review_undo_impl(str(sc_file), [1])

        review_after = load_stage6_review_from_file(str(sc_file))
        assert [e.kind for e in review_after.decision_log] == ["create_window"]
        replayed = review_after.created_windows
        assert len(replayed) == 1
        assert int(replayed[0].window_id) == wid
        lo_a, hi_a = replayed[0].freq_range
        lo_b, hi_b = original_range
        assert lo_a == pytest.approx(lo_b, abs=1e-6)
        assert hi_a == pytest.approx(hi_b, abs=1e-6)


# ---------------------------------------------------------------------------
# Claim 5 (A6): returned frequencies are labeled and carry both frames.
# ---------------------------------------------------------------------------


class TestReturnedFrequenciesLabeled:
    def test_refit_result_carries_both_frames_and_calibration_state(
        self, sc_file: Path
    ) -> None:
        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None
        cal_state, eps, sigma_eps, _, probe, _ = stamp
        wf = _first_fitted_window(sc_file)
        result = refit_window_impl(str(sc_file), wf.window_id, frame="raw")

        assert result.calibration_state == "self_calibrated" == cal_state
        assert result.epsilon == pytest.approx(eps)
        assert result.sigma_epsilon == pytest.approx(sigma_eps)
        assert len(result.fitted_peaks_calibrated_mhz) == len(result.fitted_peaks)

        for p, f_cal in zip(result.fitted_peaks, result.fitted_peaks_calibrated_mhz):
            f_raw = float(p.frequency_mhz)
            expected = _ref_calibrated(f_raw, probe=probe, eps=eps)
            assert f_cal == pytest.approx(expected, rel=1e-9)
            # The two frames must actually DIFFER on a self_calibrated file --
            # an rb_locked fixture could pass this assertion by accident.
            assert abs(f_cal - f_raw) > 1.0e-6

    def test_create_result_carries_both_frames_and_calibration_state(
        self, sc_file: Path
    ) -> None:
        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None
        _, eps, sigma_eps, _, probe, _ = stamp
        anchor_raw = _free_anchor(sc_file)
        result = create_window_impl(str(sc_file), anchor_raw, frame="raw")

        assert result.calibration_state == "self_calibrated"
        assert result.epsilon == pytest.approx(eps)
        assert result.sigma_epsilon == pytest.approx(sigma_eps)

        expected_anchor_cal = _ref_calibrated(anchor_raw, probe=probe, eps=eps)
        assert result.anchor_calibrated_mhz == pytest.approx(
            expected_anchor_cal, rel=1e-9
        )
        assert abs(result.anchor_calibrated_mhz - result.anchor_mhz) > 1.0e-6

        lo_raw, hi_raw = result.freq_range
        lo_cal, hi_cal = result.freq_range_calibrated
        assert lo_cal == pytest.approx(
            _ref_calibrated(lo_raw, probe=probe, eps=eps), rel=1e-9
        )
        assert hi_cal == pytest.approx(
            _ref_calibrated(hi_raw, probe=probe, eps=eps), rel=1e-9
        )

    def test_rb_locked_fixture_frames_coincide(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        """Sanity check, not a claim test: on an rb_locked file the two
        frames must be identical -- this is exactly why an uncalibrated
        fixture cannot catch a frame bug, and the reason every test above
        uses ``sc_file`` instead."""
        fp = tmp_path / "rb.ftmw"
        shutil.copy(stage5_small_source, fp)
        wf = _first_fitted_window(fp)
        result = refit_window_impl(str(fp), wf.window_id)
        assert result.calibration_state in ("rb_locked", "uncalibrated")
        assert result.epsilon == 0.0
        for p, f_cal in zip(result.fitted_peaks, result.fitted_peaks_calibrated_mhz):
            assert f_cal == pytest.approx(float(p.frequency_mhz), abs=1e-9)


# ---------------------------------------------------------------------------
# Claim 4: CLI, Pipeline and api agree.
# ---------------------------------------------------------------------------


class TestCrossInterfaceFrameConsistency:
    def test_omitted_frame_error_agrees_across_interfaces(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        paths: Dict[str, Path] = {}
        for name in ("impl", "pipe", "api", "cli"):
            p = tmp_path / f"{name}.ftmw"
            shutil.copy(sc_file, p)
            paths[name] = p
        wf = _first_fitted_window(paths["impl"])
        add_freq = float(wf.fitted_peaks[0].frequency_mhz) + 0.3

        with pytest.raises(ValueError, match="frame is required"):
            refit_window_impl(str(paths["impl"]), wf.window_id, add=[add_freq])
        with pytest.raises(ValueError, match="frame is required"):
            Pipeline.open(paths["pipe"]).review_edit(wf.window_id, add=[add_freq])
        with pytest.raises(ValueError, match="frame is required"):
            ftmw.review_edit(str(paths["api"]), wf.window_id, add=[add_freq])

        rc = cmd_review_edit(
            argparse.Namespace(
                file_path=str(paths["cli"]),
                window=wf.window_id,
                add=[add_freq],
                remove=[],
                verbose=False,
            )
        )
        assert rc == 1
        # The refused call must leave the file untouched, on every interface.
        assert not load_stage6_review_from_file(str(paths["cli"])).decision_log
        assert not load_stage6_review_from_file(str(paths["impl"])).decision_log
        assert not load_stage6_review_from_file(str(paths["pipe"])).decision_log
        assert not load_stage6_review_from_file(str(paths["api"])).decision_log

    def test_explicit_raw_agrees_and_succeeds_across_interfaces(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        paths: Dict[str, Path] = {}
        for name in ("impl", "pipe", "api", "cli"):
            p = tmp_path / f"{name}.ftmw"
            shutil.copy(sc_file, p)
            paths[name] = p
        wf = _first_fitted_window(paths["impl"])
        add_freq = float(wf.fitted_peaks[0].frequency_mhz) + 0.3

        refit_window_impl(str(paths["impl"]), wf.window_id, add=[add_freq], frame="raw")
        Pipeline.open(paths["pipe"]).review_edit(
            wf.window_id, add=[add_freq], frame="raw"
        )
        ftmw.review_edit(str(paths["api"]), wf.window_id, add=[add_freq], frame="raw")
        rc = cmd_review_edit(
            argparse.Namespace(
                file_path=str(paths["cli"]),
                window=wf.window_id,
                add=[add_freq],
                remove=[],
                frame="raw",
                verbose=False,
            )
        )
        assert rc == 0

        def _peak_freqs(p: Path) -> List[float]:
            wf2 = _first_fitted_window(p)
            return sorted(float(pk.frequency_mhz) for pk in wf2.fitted_peaks)

        ref = _peak_freqs(paths["impl"])
        for name in ("pipe", "api", "cli"):
            got = _peak_freqs(paths[name])
            assert len(got) == len(ref)
            for a, b in zip(got, ref):
                assert a == pytest.approx(b, abs=1e-6)

    def test_calibrated_frame_conversion_agrees_across_interfaces(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None
        eps, probe = stamp[1], stamp[4]
        anchor_raw = _free_anchor(sc_file)
        anchor_cal = _ref_calibrated(anchor_raw, probe=probe, eps=eps)

        paths: Dict[str, Path] = {}
        for name in ("impl", "pipe", "api", "cli"):
            p = tmp_path / f"{name}.ftmw"
            shutil.copy(sc_file, p)
            paths[name] = p

        r_impl = create_window_impl(str(paths["impl"]), anchor_cal, frame="calibrated")
        r_pipe = Pipeline.open(paths["pipe"]).review_create(
            anchor_cal, frame="calibrated"
        )
        r_api = ftmw.review_create(str(paths["api"]), anchor_cal, frame="calibrated")
        rc = cmd_review_create(
            argparse.Namespace(
                file_path=str(paths["cli"]),
                anchor=anchor_cal,
                frame="calibrated",
                verbose=False,
            )
        )
        assert rc == 0

        for r in (r_impl, r_pipe, r_api):
            assert r.anchor_mhz == pytest.approx(anchor_raw, abs=1e-6)
            assert r.anchor_mhz != pytest.approx(anchor_cal, rel=1e-9)

        cli_created = load_stage6_review_from_file(str(paths["cli"])).created_windows
        assert len(cli_created) == 1
        lo, hi = cli_created[0].freq_range
        cli_range = (min(lo, hi), max(lo, hi))
        assert cli_range[0] == pytest.approx(r_impl.freq_range[0], abs=1e-6)
        assert cli_range[1] == pytest.approx(r_impl.freq_range[1], abs=1e-6)
