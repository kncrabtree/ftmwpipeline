"""
Unit tests for the Stage 6 final-products consolidation (reports Step 1).

Covers the calibration math and budget in isolation (synthetic SpectrumFit),
the file-level frequency-calibration provenance + final-products serialization
round-trips, and an end-to-end review_run consolidation with cross-interface
consistency on the small 2638 fixture.

See ``dev-docs/planning/stage6-reports.md``.
"""

from __future__ import annotations

import math
import shutil

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    _build_final_products,
    get_final_products_impl,
    review_run_impl,
    set_sigma_floor_impl,
)
from ftmwpipeline.core.data_structures import (
    FinalProducts,
    FittedPeak,
    FrequencyCalibration,
    Sideband,
    SpectrumFit,
    Stage6Review,
)
from ftmwpipeline.io.frequency_calibration_serialization import (
    load_frequency_calibration_from_hdf5,
    save_frequency_calibration_to_hdf5,
)
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Pure-math unit tests on a synthetic SpectrumFit
# ---------------------------------------------------------------------------


def _synthetic_fit() -> SpectrumFit:
    """Two peaks at known molecular frequencies with known fit errors."""
    return SpectrumFit(
        fitted_peaks=[
            FittedPeak(
                peak_id=0,
                frequency_mhz=30000.0,
                amplitude=1.0,
                phase=0.5,
                frequency_error=0.0002,  # 0.2 kHz
                snr=100.0,
                window_id=3,
                origin="auto",
            ),
            FittedPeak(
                peak_id=1,
                frequency_mhz=39000.0,
                amplitude=2.0,
                phase=-0.3,
                frequency_error=0.0001,  # 0.1 kHz
                snr=500.0,
                window_id=7,
                origin="user",
            ),
        ]
    )


PROBE = 40960.0  # MHz, lower sideband (2638-like)


def test_baseband_is_distance_from_probe():
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="rb_locked",
        epsilon=0.0,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
    )
    assert fp.peaks[0].f_baseband_mhz == pytest.approx(40960.0 - 30000.0)
    assert fp.peaks[1].f_baseband_mhz == pytest.approx(40960.0 - 39000.0)


def test_epsilon_correction_pushes_baseband():
    eps = 2.0e-6
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="self_calibrated",
        epsilon=eps,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
    )
    p0 = fp.peaks[0]
    expected = PROBE + (30000.0 - PROBE) / (1.0 + eps)
    assert p0.frequency_mhz == pytest.approx(expected, abs=1e-9)
    assert p0.frequency_raw_mhz == pytest.approx(30000.0)
    # The correction moves the line by ~ eps * baseband ~ 0.0219 MHz.
    assert p0.frequency_mhz - p0.frequency_raw_mhz == pytest.approx(
        eps * (PROBE - 30000.0), rel=1e-3
    )


def test_no_correction_when_epsilon_zero():
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="rb_locked",
        epsilon=0.0,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
    )
    for p in fp.peaks:
        assert p.frequency_mhz == p.frequency_raw_mhz


def test_three_term_budget():
    sigma_eps_ppm = 0.1e-6
    floor = 3.0  # kHz
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="self_calibrated",
        epsilon=2.0e-6,
        sigma_epsilon=sigma_eps_ppm,
        sigma_floor_khz=floor,
    )
    p0 = fp.peaks[0]
    expected_stat = 0.0002 * 1e3  # 0.2 kHz
    expected_eps = sigma_eps_ppm * (PROBE - 30000.0) * 1e3
    assert p0.sigma_stat_khz == pytest.approx(expected_stat)
    assert p0.sigma_eps_khz == pytest.approx(expected_eps)
    assert p0.sigma_floor_khz == pytest.approx(floor)
    assert p0.sigma_f_khz == pytest.approx(
        math.sqrt(expected_stat**2 + expected_eps**2 + floor**2)
    )


def test_budget_eps_term_scales_with_baseband():
    """sigma_eps grows with baseband; the line nearer the probe is tighter."""
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="self_calibrated",
        epsilon=2.0e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=0.0,
    )
    # peak 1 (39000) is closer to the probe than peak 0 (30000).
    assert fp.peaks[1].sigma_eps_khz < fp.peaks[0].sigma_eps_khz


def test_metadata_and_origin_carried():
    fp = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="self_calibrated",
        epsilon=2.0e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=1.0,
    )
    assert fp.calibration_state == "self_calibrated"
    assert fp.sideband == "lower"
    assert fp.probe_freq_mhz == PROBE
    assert [p.origin for p in fp.peaks] == ["auto", "user"]
    assert [p.window_id for p in fp.peaks] == [3, 7]


def test_clock_lattice_carried_and_roundtrips(tmp_path):
    fit = _synthetic_fit()
    # One line sits on the declared clock lattice; the other is off-lattice.
    fit.fitted_peaks[0].clock_lattice = "320x6 (bb)"
    fit.fitted_peaks[1].clock_lattice = None
    fp = _build_final_products(
        fit,
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="rb_locked",
        epsilon=0.0,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
    )
    assert [p.clock_lattice for p in fp.peaks] == ["320x6 (bb)", None]

    review = Stage6Review(final_products=fp)
    out = tmp_path / "review.h5"
    with h5py.File(out, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)
    with h5py.File(out, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.final_products is not None
    assert [p.clock_lattice for p in loaded.final_products.peaks] == [
        "320x6 (bb)",
        None,
    ]


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


def test_frequency_calibration_roundtrip(tmp_path):
    fp = tmp_path / "fc.h5"
    with h5py.File(fp, "w") as h5f:
        save_frequency_calibration_to_hdf5(FrequencyCalibration(7.5), h5f)
    with h5py.File(fp, "r") as h5f:
        loaded = load_frequency_calibration_from_hdf5(h5f)
    assert loaded.sigma_floor_khz == pytest.approx(7.5)


def test_frequency_calibration_absent_returns_default(tmp_path):
    fp = tmp_path / "empty.h5"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with h5py.File(fp, "r") as h5f:
        loaded = load_frequency_calibration_from_hdf5(h5f)
    assert loaded.sigma_floor_khz == 0.0


def test_final_products_roundtrip(tmp_path):
    products = _build_final_products(
        _synthetic_fit(),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="self_calibrated",
        epsilon=2.0e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=2.0,
    )
    review = Stage6Review(final_products=products)
    fp = tmp_path / "review.h5"
    with h5py.File(fp, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)
    with h5py.File(fp, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])

    assert loaded.final_products is not None
    lp = loaded.final_products
    assert lp.calibration_state == "self_calibrated"
    assert lp.epsilon == pytest.approx(2.0e-6)
    assert lp.sideband == "lower"
    assert len(lp.peaks) == 2
    for got, exp in zip(lp.peaks, products.peaks):
        assert got.frequency_mhz == pytest.approx(exp.frequency_mhz)
        assert got.sigma_f_khz == pytest.approx(exp.sigma_f_khz)
        assert got.origin == exp.origin
        assert got.window_id == exp.window_id


def test_review_without_final_products_roundtrips_none(tmp_path):
    review = Stage6Review()
    fp = tmp_path / "review.h5"
    with h5py.File(fp, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)
    with h5py.File(fp, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.final_products is None


# ---------------------------------------------------------------------------
# set_sigma_floor validation
# ---------------------------------------------------------------------------


def test_set_sigma_floor_rejects_negative(tmp_path):
    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with pytest.raises(ValueError):
        set_sigma_floor_impl(str(fp), -1.0)
    with pytest.raises(ValueError):
        set_sigma_floor_impl(str(fp), float("inf"))


def test_set_sigma_floor_persists(tmp_path):
    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    set_sigma_floor_impl(str(fp), 4.0)
    with h5py.File(fp, "r") as h5f:
        assert load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz == 4.0


# ---------------------------------------------------------------------------
# End-to-end on the small 2638 fixture + cross-interface
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_review_run_builds_final_products(stage5_small_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    review_run_impl(str(fp))
    products = get_final_products_impl(str(fp))

    assert isinstance(products, FinalProducts)
    # 2638 autopopulates an unlocked-digitizer clock declaration; with no
    # timebase calibration run, the axis is free-running but not self-calibrated.
    assert products.calibration_state == "uncalibrated"
    assert products.epsilon == 0.0
    assert products.sigma_floor_khz == 0.0
    assert len(products.peaks) > 0

    # Uncalibrated + no floor: no correction applied, budget is precision-only.
    for p in products.peaks:
        assert p.frequency_mhz == p.frequency_raw_mhz
        assert p.sigma_eps_khz == 0.0
        assert p.sigma_f_khz == pytest.approx(p.sigma_stat_khz)


# ---------------------------------------------------------------------------
# Calibration-state derivation (the three states), crafted directly
# ---------------------------------------------------------------------------


def _write_clocks(path, clocks) -> None:
    from dataclasses import replace

    from ftmwpipeline.core.stage_fit_settings import resolve
    from ftmwpipeline.io.stage_fit_settings_serialization import (
        save_stage_fit_settings_to_h5,
    )

    settings = resolve()
    settings = replace(settings, spur=replace(settings.spur, clocks=clocks))
    save_stage_fit_settings_to_h5(str(path), settings)


def _write_timebase(path, *, epsilon, sigma_epsilon, preconditions_passed) -> None:
    import h5py as _h5

    from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
    from ftmwpipeline.io.timebase_serialization import (
        GROUP_PATH,
        save_timebase_calibration_to_hdf5,
    )

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
        preconditions_passed=preconditions_passed,
    )
    with _h5.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        grp = h5f.create_group(GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)


def test_derive_state_rb_locked_when_all_locked(tmp_path):
    from ftmwpipeline._internal.stage6_impl import _derive_frequency_calibration
    from ftmwpipeline.core.stage_fit_settings import ClockSource

    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    _write_clocks(fp, (ClockSource(5120.0, locked=True),))

    state, eps, sig = _derive_frequency_calibration(str(fp))
    assert state == "rb_locked"
    assert eps == 0.0 and sig == 0.0


def test_derive_state_rb_locked_when_no_declaration(tmp_path):
    from ftmwpipeline._internal.stage6_impl import _derive_frequency_calibration

    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    state, eps, sig = _derive_frequency_calibration(str(fp))
    assert state == "rb_locked"


def test_derive_state_uncalibrated_when_unlocked_no_timebase(tmp_path):
    from ftmwpipeline._internal.stage6_impl import _derive_frequency_calibration
    from ftmwpipeline.core.stage_fit_settings import ClockSource

    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    _write_clocks(
        fp,
        (ClockSource(5120.0, locked=True), ClockSource(6250.0, locked=False)),
    )
    state, eps, sig = _derive_frequency_calibration(str(fp))
    assert state == "uncalibrated"
    assert eps == 0.0


def test_derive_state_self_calibrated_with_timebase(tmp_path):
    from ftmwpipeline._internal.stage6_impl import _derive_frequency_calibration
    from ftmwpipeline.core.stage_fit_settings import ClockSource

    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    _write_clocks(
        fp,
        (ClockSource(5120.0, locked=True), ClockSource(6250.0, locked=False)),
    )
    _write_timebase(fp, epsilon=2.2e-6, sigma_epsilon=0.1e-6, preconditions_passed=True)
    state, eps, sig = _derive_frequency_calibration(str(fp))
    assert state == "self_calibrated"
    assert eps == pytest.approx(2.2e-6)
    assert sig == pytest.approx(0.1e-6)


def test_derive_state_uncalibrated_when_timebase_failed(tmp_path):
    """A timebase whose preconditions failed must not be applied."""
    from ftmwpipeline._internal.stage6_impl import _derive_frequency_calibration
    from ftmwpipeline.core.stage_fit_settings import ClockSource

    fp = tmp_path / "x.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    _write_clocks(
        fp,
        (ClockSource(5120.0, locked=True), ClockSource(6250.0, locked=False)),
    )
    _write_timebase(
        fp, epsilon=0.0, sigma_epsilon=float("inf"), preconditions_passed=False
    )
    state, eps, sig = _derive_frequency_calibration(str(fp))
    assert state == "uncalibrated"
    assert eps == 0.0


@pytest.mark.integration
def test_review_run_sigma_floor_folds_into_budget(stage5_small_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    review_run_impl(str(fp), sigma_floor_khz=5.0)
    products = get_final_products_impl(str(fp))
    assert products is not None
    assert products.sigma_floor_khz == pytest.approx(5.0)
    for p in products.peaks:
        assert p.sigma_floor_khz == pytest.approx(5.0)
        assert p.sigma_f_khz == pytest.approx(math.sqrt(p.sigma_stat_khz**2 + 5.0**2))

    # The floor persists; a later run without the flag keeps it.
    review_run_impl(str(fp))
    products2 = get_final_products_impl(str(fp))
    assert products2 is not None
    assert products2.sigma_floor_khz == pytest.approx(5.0)


@pytest.mark.integration
def test_final_products_cross_interface(stage5_small_file, tmp_path):
    fp_api = tmp_path / "api.ftmw"
    fp_pipe = tmp_path / "pipe.ftmw"
    shutil.copy(stage5_small_file, fp_api)
    shutil.copy(stage5_small_file, fp_pipe)

    ftmw.review_run(str(fp_api), sigma_floor_khz=2.5)
    Pipeline.open(fp_pipe).review_run(sigma_floor_khz=2.5)

    prod_api = ftmw.get_final_products(str(fp_api))
    prod_pipe = Pipeline.open(fp_pipe).final_products()

    assert prod_api is not None and prod_pipe is not None
    assert prod_api.calibration_state == prod_pipe.calibration_state
    assert len(prod_api.peaks) == len(prod_pipe.peaks)
    for a, b in zip(prod_api.peaks, prod_pipe.peaks):
        assert a.frequency_mhz == pytest.approx(b.frequency_mhz)
        assert a.sigma_f_khz == pytest.approx(b.sigma_f_khz)
