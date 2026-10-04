"""Frozen contributors follow the window's shared fitted tau (ROADMAP D18).

``stage5_fitting.rst`` writes a dependent window's frozen contributor as
``h_T(u - delta_c; tau)`` at the window's shared ``tau``. With ``tau`` free the
skirt must follow the trial ``tau`` inside the optimisation; with ``tau`` held
the fit must be bit-identical to subtracting the skirt once.
"""

import numpy as np
import pytest

from ftmwpipeline.core.environment import ANALYSIS_EPOCH
from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum
from ftmwpipeline.fitting.plan_execution import (
    FrozenPeak,
    fit_window_with_fixed_contributors,
    subtract_frozen_background,
)
from ftmwpipeline.fitting.window_fit import (
    FrozenSkirt,
    fit_window,
    frozen_skirt_delta,
)

pytestmark = [pytest.mark.unit]

T = 12.73
TAU_TRUE = 8.5
TAU0 = 6.4
SHAPE = "gaussian"
U = np.arange(-30, 31) / T
SIGMA = np.ones(U.size)
FROZEN = ModelPeak(20000.0, 3.3, -0.7)
FREE = ModelPeak(200.0, 0.11, 0.4)


def _data(seed=20261003):
    rng = np.random.default_rng(seed)
    noise = (rng.normal(size=U.size) + 1j * rng.normal(size=U.size)) / np.sqrt(2.0)
    return (
        model_spectrum(U, [FREE], TAU_TRUE, T, shape=SHAPE)
        + model_spectrum(U, [FROZEN], TAU_TRUE, T, shape=SHAPE)
        + noise
    )


def _chi2(residual):
    return float(np.sum(np.abs(residual) ** 2 / (SIGMA**2 / 2)))


def test_free_tau_skirt_follows_the_fitted_tau():
    z = _data()
    fp = FrozenPeak(0, 1, FROZEN, float("nan"))
    res, bg, full, resid = fit_window_with_fixed_contributors(
        U, z, SIGMA, [fp], [0.1], TAU0, T, fit_tau=True, shape=SHAPE
    )
    f = res.fit
    # One line, at the true decay time: the skirt no longer leaves a stale
    # pedestal for a spurious line to absorb.
    assert f.n_peaks == 1
    assert f.tau_us == pytest.approx(TAU_TRUE, abs=0.02)
    assert f.peaks[0].amplitude == pytest.approx(200.0, rel=0.01)
    # The background is the skirt at the fitted tau, and the reported chi2 is
    # the chi2 of the D18 model at that tau.
    np.testing.assert_array_equal(
        bg, model_spectrum(U, [FROZEN], f.tau_us, T, shape=SHAPE)
    )
    d18 = model_spectrum(U, f.peaks, f.tau_us, T, shape=SHAPE) + bg
    assert _chi2(z - d18) == pytest.approx(f.chi_squared, rel=1e-9)
    assert _chi2(resid) == pytest.approx(f.chi_squared, rel=1e-9)


def test_held_tau_is_bit_identical_to_one_subtraction():
    z = _data()
    fp = FrozenPeak(0, 1, FROZEN, float("nan"))
    bg, data_minus_bg = subtract_frozen_background(U, z, [fp], TAU0, T, shape=SHAPE)
    seeds = [ModelPeak(150.0, 0.1, 0.0)]
    kw = dict(fit_tau=False, shape=SHAPE)
    plain = fit_window(U, data_minus_bg, SIGMA, seeds, TAU0, T, **kw)
    skirt = fit_window(
        U,
        data_minus_bg,
        SIGMA,
        seeds,
        TAU0,
        T,
        frozen_skirts=(FrozenSkirt((FROZEN,), TAU0),),
        **kw,
    )
    assert plain.tau_us == skirt.tau_us == TAU0
    np.testing.assert_array_equal(plain.residual, skirt.residual)
    np.testing.assert_array_equal(plain.fitted_spectrum, skirt.fitted_spectrum)
    assert plain.chi_squared == skirt.chi_squared
    for a, b in zip(plain.peaks, skirt.peaks):
        assert (a.amplitude, a.offset_mhz, a.phase) == (
            b.amplitude,
            b.offset_mhz,
            b.phase,
        )
    _, _, _, resid_held = fit_window_with_fixed_contributors(
        U, z, SIGMA, [fp], [0.1], TAU0, T, fit_tau=False, shape=SHAPE
    )
    assert np.all(np.isfinite(resid_held))


def test_delta_is_absent_at_the_reference_tau():
    sk = FrozenSkirt((FROZEN,), TAU0)
    assert frozen_skirt_delta((sk,), U, TAU0, T, SHAPE) is None
    d = frozen_skirt_delta((sk,), U, TAU_TRUE, T, SHAPE)
    expect = model_spectrum(U, [FROZEN], TAU_TRUE, T, shape=SHAPE) - model_spectrum(
        U, [FROZEN], TAU0, T, shape=SHAPE
    )
    np.testing.assert_array_equal(d, expect)


def test_bins_restrict_the_skirt_and_follow_a_reordering():
    bins = np.zeros(U.size, dtype=bool)
    bins[: U.size // 2] = True
    sk = FrozenSkirt((FROZEN,), TAU0, bins=bins)
    d = frozen_skirt_delta((sk,), U, TAU_TRUE, T, SHAPE)
    assert not d[~bins].any() and d[bins].any()
    order = np.argsort(-U)
    d_sorted = frozen_skirt_delta((sk.reordered(order),), U[order], TAU_TRUE, T, SHAPE)
    np.testing.assert_array_equal(d_sorted, d[order])


def test_analysis_epoch_records_the_d18_fit_change():
    # The skirt following tau moves free-tau fits with frozen contributors, so
    # files fitted before it must be re-fit or acknowledged.
    assert ANALYSIS_EPOCH == 4
