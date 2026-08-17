"""
Tests for the A7 final-products staleness fix.

``FinalProducts`` is derived from the raw Stage 5 fit plus the file's
calibration (epsilon, sigma_epsilon, calibration_state, sigma_floor_khz,
probe_freq_mhz, sideband) and already stamps all six as provenance. The bug:
nothing ever compared that stamp against the file's *current* calibration, so
a timebase re-run after ``review run`` left every read path -- including a
bare ``accept``, whose own doc comment reasoned (wrongly) that "accepting
as-is does not change the fit, so the table stays valid" -- silently serving
or persisting a table built from the old epsilon.

An ``rb_locked`` fixture has epsilon = 0 always and cannot exercise any of
this (rebuilding a table that is already all-zeros looks identical to not
rebuilding it), so every test here uses a ``self_calibrated`` fixture, forced
by declaring an unlocked digitizer clock and stamping a passing
``TimebaseCalibrationResult`` directly -- the recipe from
``scratch/preview-session-plan.md`` ("Verified facts for later units").
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_impl import report_table_impl
from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import (
    _current_final_products,
    _fid_header_for_stamp,
    _final_products_is_stale,
    _rebuild_final_products,
    apply_curation_impl,
    get_final_products_impl,
    review_accept_impl,
    review_run_impl,
)
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
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

# ---------------------------------------------------------------------------
# self_calibrated fixture recipe (scratch/preview-session-plan.md)
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
    what a real timebase re-run persists, and nothing else. In particular this
    never touches Stage 6, which is the point: it is the "no Stage 6 action at
    all" half of the bug."""
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
    path: Path, *, epsilon: float, sigma_epsilon: float = 0.1e-6
) -> None:
    _declare_unlocked_digitizer(path)
    _stamp_timebase(path, epsilon=epsilon, sigma_epsilon=sigma_epsilon)


EPS_1 = 2.2e-6
EPS_2 = 8.8e-6
SIGMA_EPS = 0.1e-6


# ---------------------------------------------------------------------------
# A wider multi-window build: the shared 3-window fixture typically keeps only
# one window with an actual fit, but "every line moves, including windows no
# action touched" needs several fitted windows in play at once.
# ---------------------------------------------------------------------------


def _build_stage5_multi(dest: Path, data_path: str) -> None:
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


@pytest.fixture(scope="session")
def _stage5_sc_multi_built(exp_2638_data_path, tmp_path_factory) -> Path:
    """A wider post-fit build, stamped self_calibrated at EPS_1 -- read-only,
    built once."""
    fp = tmp_path_factory.mktemp("stage6_sc_multi") / "stage5_sc_multi.ftmw"
    _build_stage5_multi(fp, exp_2638_data_path)
    _make_self_calibrated(fp, epsilon=EPS_1, sigma_epsilon=SIGMA_EPS)
    return fp


@pytest.fixture
def sc_multi_file(_stage5_sc_multi_built, tmp_path) -> Path:
    """A fresh writable copy of the self-calibrated multi-window build."""
    fp = tmp_path / "stage5_sc_multi.ftmw"
    shutil.copy(_stage5_sc_multi_built, fp)
    return fp


def _fitted_window_ids(fp: Path) -> List[int]:
    with h5py.File(str(fp), "r") as h5f:
        from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return sorted(
        {
            int(wf.window_id)
            for wf in sf.window_fits
            if wf.window_id is not None and wf.fitted_peaks
        }
    )


# ---------------------------------------------------------------------------
# The fixture recipe itself: sanity-check it actually reaches self_calibrated.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fixture_recipe_reaches_self_calibrated(sc_multi_file):
    review_run_impl(str(sc_multi_file))
    products = get_final_products_impl(str(sc_multi_file))
    assert products is not None
    assert products.calibration_state == "self_calibrated"
    assert products.epsilon == pytest.approx(EPS_1)
    assert products.sigma_epsilon == pytest.approx(SIGMA_EPS)
    assert len(products.peaks) > 0
    # An rb_locked/uncalibrated fixture would have epsilon == 0 for every
    # peak's contribution; this one must not.
    assert any(p.frequency_mhz != p.frequency_raw_mhz for p in products.peaks)


@pytest.mark.integration
def test_fid_header_for_stamp_agrees_with_full_fid_load(sc_multi_file):
    """The cheap header-only read the staleness check uses on every
    ``get_final_products_impl`` call must report exactly what
    ``load_fid_from_pipeline_impl`` (which materializes the whole FID) does
    -- a mismatch here would make the staleness check compare against the
    wrong probe frequency / sideband and force a spurious rebuild on every
    single read, forever."""
    fp = sc_multi_file
    header = _fid_header_for_stamp(str(fp))
    assert header is not None
    header_probe, header_sideband = header

    fid = load_fid_from_pipeline_impl(str(fp))
    assert header_probe == fid.probe_freq_mhz
    assert header_sideband == fid.sideband.value


# ---------------------------------------------------------------------------
# 1. Timebase re-run with NO Stage 6 action at all.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_timebase_rerun_no_stage6_action_rebuilds_on_read(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))
    before = get_final_products_impl(str(fp))
    assert before is not None and before.epsilon == pytest.approx(EPS_1)

    # A timebase re-run and nothing else -- no accept, no edit, no review run.
    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

    after = get_final_products_impl(str(fp))
    assert after is not None
    assert after.calibration_state == "self_calibrated"
    assert after.epsilon == pytest.approx(EPS_2)
    assert after.epsilon != pytest.approx(before.epsilon)

    _assert_arithmetic(before, after, delta_eps=EPS_2 - EPS_1)


# ---------------------------------------------------------------------------
# 2. The bare-accept path specifically (review_accept_impl, no candidate).
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bare_accept_does_not_launder_stale_table(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))
    before = get_final_products_impl(str(fp))
    assert before is not None and before.epsilon == pytest.approx(EPS_1)

    wids = _fitted_window_ids(fp)
    assert len(wids) >= 2, "need at least 2 fitted windows for this check"
    accepted_wid, untouched_wid = wids[0], wids[1]

    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

    # A single bare accept on one window -- no candidate, no fit mutation.
    result = review_accept_impl(str(fp), accepted_wid)
    assert result is None  # bare accept returns None

    # The write itself must not have re-persisted the stale table.
    persisted = load_stage6_review_from_file(str(fp)).final_products
    assert persisted is not None
    assert persisted.epsilon == pytest.approx(EPS_2)
    assert persisted.calibration_state == "self_calibrated"

    # The read path agrees, and untouched windows moved too.
    after = get_final_products_impl(str(fp))
    assert after is not None
    _assert_arithmetic(before, after, delta_eps=EPS_2 - EPS_1)
    assert any(p.window_id == untouched_wid for p in after.peaks)

    # Bare accept still recorded its decision.
    log = load_stage6_review_from_file(str(fp)).decision_log
    assert any(e.window_id == accepted_wid and e.kind == "accept" for e in log)


# ---------------------------------------------------------------------------
# 3. A curation pass consisting only of bare accepts.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_curation_pass_of_only_bare_accepts_rebuilds(sc_multi_file, tmp_path):
    fp = sc_multi_file
    review_run_impl(str(fp))
    before = get_final_products_impl(str(fp))
    assert before is not None and before.epsilon == pytest.approx(EPS_1)

    wids = _fitted_window_ids(fp)
    assert len(wids) >= 3, "need at least 3 fitted windows for this check"
    accepted = wids[:2]
    untouched = wids[2]

    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

    cur = tmp_path / "bare_only.csv"
    cur.write_text("".join(f"accept,{wid},,\n" for wid in accepted))
    apply_curation_impl(str(fp), cur)

    persisted = load_stage6_review_from_file(str(fp)).final_products
    assert persisted is not None
    assert persisted.epsilon == pytest.approx(EPS_2)

    after = get_final_products_impl(str(fp))
    assert after is not None
    _assert_arithmetic(before, after, delta_eps=EPS_2 - EPS_1)
    # The window that received no curation row at all still shows up
    # corrected -- proof the whole table was rebuilt, not just the touched
    # windows re-derived.
    assert any(p.window_id == untouched for p in after.peaks)

    log = load_stage6_review_from_file(str(fp)).decision_log
    assert [e.window_id for e in log if e.kind == "accept"] == accepted


# ---------------------------------------------------------------------------
# 4. Decision log preserved across a rebuild.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_decision_log_preserved_across_rebuild(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))

    wids = _fitted_window_ids(fp)
    assert len(wids) >= 2

    # First decision at the original epsilon.
    review_accept_impl(str(fp), wids[0])
    log_before = load_stage6_review_from_file(str(fp)).decision_log
    assert len(log_before) == 1
    assert log_before[0].order_index == 0

    # Re-run the timebase, then a second bare accept -- this one's the write
    # that has to rebuild the (now-stale) table.
    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
    review_accept_impl(str(fp), wids[1])

    review = load_stage6_review_from_file(str(fp))
    assert review.final_products is not None
    assert review.final_products.epsilon == pytest.approx(EPS_2)

    log_after = review.decision_log
    assert len(log_after) == 2
    assert [e.window_id for e in log_after] == [wids[0], wids[1]]
    assert [e.order_index for e in log_after] == [0, 1]
    assert [e.kind for e in log_after] == ["accept", "accept"]


# ---------------------------------------------------------------------------
# 5. The arithmetic, standalone: delta_eps * f_baseband, every line.
# ---------------------------------------------------------------------------


def _assert_arithmetic(before, after, *, delta_eps: float) -> None:
    """delta_eps moves every corrected frequency by delta_eps * f_baseband,
    to first order -- including peaks in windows neither accept nor edit ever
    named. ``f_corr = probe + (f_raw - probe) / (1 + eps)``, so exactly:
    ``shift = (f_raw - probe) * (1/(1+eps2) - 1/(1+eps1))
            = -(f_raw - probe) * delta_eps / ((1+eps1)(1+eps2))``.
    At eps ~ 1e-6 to 1e-5 the ``(1+eps1)(1+eps2)`` denominator is a ~1e-5
    relative correction to the first-order ``delta_eps * f_baseband``
    estimate -- real, but three orders below the kHz scale this whole budget
    lives at, so a generous relative tolerance still distinguishes "the
    correction is applied" from "it isn't" or "the sign / baseband is wrong".
    """
    by_key_before = {
        (p.window_id, round(p.frequency_raw_mhz, 6)): p for p in before.peaks
    }
    by_key_after = {
        (p.window_id, round(p.frequency_raw_mhz, 6)): p for p in after.peaks
    }
    assert by_key_before.keys() == by_key_after.keys()
    assert len(by_key_before) > 0
    for key, pb in by_key_before.items():
        pa = by_key_after[key]
        shift = pa.frequency_mhz - pb.frequency_mhz
        assert abs(shift) == pytest.approx(
            abs(delta_eps) * pb.f_baseband_mhz, rel=1e-3, abs=1e-9
        )


@pytest.mark.integration
def test_arithmetic_delta_eps_times_baseband(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))
    before = get_final_products_impl(str(fp))
    assert before is not None

    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
    after = get_final_products_impl(str(fp))
    assert after is not None

    _assert_arithmetic(before, after, delta_eps=EPS_2 - EPS_1)


# ---------------------------------------------------------------------------
# Report path: report_table_impl (report_impl.py) must not serve stale rows.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_report_table_reflects_current_calibration(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))
    before_csv = report_table_impl(str(fp), fmt="csv")

    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
    after_csv = report_table_impl(str(fp), fmt="csv")

    assert after_csv != before_csv
    products = get_final_products_impl(str(fp))
    assert products is not None
    assert products.epsilon == pytest.approx(EPS_2)
    # The rendered frequency of the first peak appears in the CSV text.
    first_freq = f"{products.peaks[0].frequency_mhz:.6f}"
    assert any(first_freq[:10] in line for line in after_csv.splitlines())


# ---------------------------------------------------------------------------
# Cross-interface: api / Pipeline must see the same rebuilt table.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cross_interface_sees_rebuilt_table(sc_multi_file, tmp_path):
    fp_api = tmp_path / "api.ftmw"
    fp_pipe = tmp_path / "pipe.ftmw"
    shutil.copy(sc_multi_file, fp_api)
    shutil.copy(sc_multi_file, fp_pipe)

    ftmw.review_run(str(fp_api))
    Pipeline.open(fp_pipe).review_run()

    _stamp_timebase(fp_api, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
    _stamp_timebase(fp_pipe, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

    prod_api = ftmw.get_final_products(str(fp_api))
    prod_pipe = Pipeline.open(fp_pipe).final_products()

    assert prod_api is not None and prod_pipe is not None
    assert prod_api.epsilon == pytest.approx(EPS_2)
    assert prod_pipe.epsilon == pytest.approx(EPS_2)
    for a, b in zip(prod_api.peaks, prod_pipe.peaks):
        assert a.frequency_mhz == pytest.approx(b.frequency_mhz)


# ---------------------------------------------------------------------------
# Direct unit coverage of the staleness helpers (no fixture needed for shape).
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_final_products_is_stale_true_after_rerun(sc_multi_file):
    fp = sc_multi_file
    review_run_impl(str(fp))
    review = load_stage6_review_from_file(str(fp))
    assert review.final_products is not None
    assert _final_products_is_stale(review.final_products, str(fp)) is False

    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
    assert _final_products_is_stale(review.final_products, str(fp)) is True


def test_final_products_is_stale_none_is_never_stale(sc_multi_file):
    assert _final_products_is_stale(None, str(sc_multi_file)) is False


@pytest.mark.integration
def test_current_final_products_read_only(sc_multi_file):
    """The rebuild must not write to the file -- it has to be safe to call on
    a file the caller only opened for reading."""
    fp = sc_multi_file
    review_run_impl(str(fp))
    review = load_stage6_review_from_file(str(fp))
    _stamp_timebase(fp, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

    before_bytes = fp.read_bytes()
    rebuilt = _current_final_products(review.final_products, str(fp))
    after_bytes = fp.read_bytes()

    assert rebuilt is not None
    assert rebuilt.epsilon == pytest.approx(EPS_2)
    assert before_bytes == after_bytes

    # The on-disk stage6_review group is unchanged -- the rebuild is purely
    # in-memory.
    still_persisted = load_stage6_review_from_file(str(fp)).final_products
    assert still_persisted is not None
    assert still_persisted.epsilon == pytest.approx(EPS_1)


def test_rebuild_final_products_none_without_stage5(tmp_path):
    fp = tmp_path / "empty.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    assert _rebuild_final_products(str(fp)) is None
