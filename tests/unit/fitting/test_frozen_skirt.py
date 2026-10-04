"""Frozen contributors follow the window's shared fitted tau (ROADMAP D18).

``stage5_fitting.rst`` writes a dependent window's frozen contributor as
``h_T(u - delta_c; tau)`` at the window's shared ``tau``. With ``tau`` free the
skirt must follow the trial ``tau`` inside the optimisation; with ``tau`` held
the fit must be bit-identical to subtracting the skirt once.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.core.environment import ANALYSIS_EPOCH
from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from ftmwpipeline.fitting.plan_execution import (
    FrozenPeak,
    _apply_baseline_to_outcome,
    _apply_rescue_to_outcome,
    fit_seeds_window_outcome,
    fit_window_with_fixed_contributors,
    frozen_skirts_for,
    local_thaw_cofit,
    subtract_frozen_background,
)
from ftmwpipeline.fitting.residual_rescue import rescue_and_consolidate
from ftmwpipeline.fitting.validation import DEFAULT_N_EFF_KIND
from ftmwpipeline.fitting.window_fit import (
    FrozenSkirt,
    conservative_fit,
    evaluate_baseline,
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
    # Catches: a skirt delta that is nonzero at the held tau (wrong tau_ref in
    # frozen_skirts_for, a skirt re-drawn at a different tau, the delta applied
    # when fit_tau=False) and background_at_fitted_tau re-evaluating the skirt
    # instead of returning the reference background when tau never moved.
    # Exact equality, not isfinite: held tau must reproduce the pre-D18 fit.
    z = _data()
    fp = FrozenPeak(0, 1, FROZEN, float("nan"))
    bg0, data_minus_bg = subtract_frozen_background(U, z, [fp], TAU0, T, shape=SHAPE)
    kw = dict(fit_tau=False, shape=SHAPE)
    ref = conservative_fit(
        U, data_minus_bg, SIGMA, [0.1], TAU0, T, gate_background=bg0, **kw
    )
    fit, bg, full, resid = fit_window_with_fixed_contributors(
        U, z, SIGMA, [fp], [0.1], TAU0, T, **kw
    )
    r, f = ref.fit, fit.fit
    assert f.tau_us == r.tau_us == TAU0
    assert f.n_peaks == r.n_peaks >= 1
    assert np.array_equal(f.residual, r.residual)
    assert np.array_equal(f.fitted_spectrum, r.fitted_spectrum)
    assert f.chi_squared == r.chi_squared
    assert f.cost == r.cost
    for a, b in zip(r.peaks, f.peaks):
        assert (a.amplitude, a.offset_mhz, a.phase) == (
            b.amplitude,
            b.offset_mhz,
            b.phase,
        )
    # The returned background is the reference array itself, the full model is
    # free + that background, and the full residual is the data minus it.
    assert np.array_equal(bg, bg0)
    assert np.array_equal(full, r.fitted_spectrum + bg0)
    assert np.array_equal(resid, z - full)
    # The reported chi2 is the one-subtraction chi2 of the full residual.
    assert _chi2(resid) == pytest.approx(f.chi_squared, rel=1e-9)


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


# ---------------------------------------------------------------------------
# Free tau through the later phases (rescue, late baseline, Stage 6 refit)
# ---------------------------------------------------------------------------
# A dependent window with one frozen line and one free line. The rescue test
# adds a second weak free line the starting fit misses so the rescue accepts a
# round (without an accepted round the refit chain never runs).
MISSED = ModelPeak(120.0, -1.4, 1.0)
FW_KWARGS = {"fit_tau": True, "shape": SHAPE}


def _window_data(extra=(), seed=20261003):
    rng = np.random.default_rng(seed)
    noise = (rng.normal(size=U.size) + 1j * rng.normal(size=U.size)) / np.sqrt(2.0)
    return model_spectrum(U, [FREE, FROZEN, *extra], TAU_TRUE, T, shape=SHAPE) + noise


def _held_outcome(z, *, fw_kwargs=None):
    """The window fitted once with tau held at ``TAU0`` (its starting state)."""
    fp = FrozenPeak(0, 1, FROZEN, float("nan"))
    bg, dmb = subtract_frozen_background(U, z, [fp], TAU0, T, shape=SHAPE)
    out = fit_seeds_window_outcome(
        U,
        z,
        SIGMA,
        0.0,
        bg,
        dmb,
        [fp],
        [ModelPeak(150.0, 0.1, 0.0)],
        TAU0,
        T,
        fw_kwargs or {"fit_tau": False, "shape": SHAPE},
        None,
        DEFAULT_N_EFF_KIND,
        16,
        1,
    )
    return out, fp, bg, dmb


def _d18_chi2(z, inner):
    """chi2 of data minus (free lines + baseline + skirt), all at the fitted tau."""
    free = model_spectrum(U, inner.peaks, inner.tau_us, T, shape=SHAPE)
    free = free + evaluate_baseline(inner, U)
    skirt = model_spectrum(U, [FROZEN], inner.tau_us, T, shape=SHAPE)
    return _chi2(z - free - skirt)


def _skirt_at(tau):
    return model_spectrum(U, [FROZEN], tau, T, shape=SHAPE)


def _assert_outcome_is_d18(out, z):
    inner = out.fit.fit
    assert inner.tau_us != TAU0  # tau actually moved off the starting value
    assert _d18_chi2(z, inner) == pytest.approx(inner.chi_squared, rel=1e-9)
    # The outcome's background is the skirt at the FINAL tau, and the stored
    # full residual is the residual of the model that was minimised.
    np.testing.assert_array_equal(out.background, _skirt_at(inner.tau_us))
    assert _chi2(out.full_residual) == pytest.approx(inner.chi_squared, rel=1e-9)


def test_seed_refit_follows_tau_with_the_frozen_skirt():
    # Stage 6 refit entry (fit_seeds_window_outcome) with tau free.
    # Catches: frozen_skirts_for dropped from fit_seeds_window_outcome (chi2 is
    # then against the tau0 skirt), the knockout refits losing the skirt, and
    # background_at_fitted_tau returning the reference background
    # unconditionally (outcome.background stays the tau0 skirt).
    z = _window_data()
    out, _, _, _ = _held_outcome(z, fw_kwargs=FW_KWARGS)
    assert out.fit.fit.tau_us == pytest.approx(TAU_TRUE, abs=0.1)
    _assert_outcome_is_d18(out, z)


def test_rescue_and_consolidate_follows_tau_with_the_frozen_skirt():
    # Catches: frozen_skirts not forwarded to the rescue's inner fit, the joint
    # refit, or the merge / knockout / cleanup refits inside
    # rescue_and_consolidate (the consolidated chi2 is then the chi2 of a model
    # whose skirt stayed at tau0 while tau moved).
    z = _window_data([MISSED])
    out, fp, _, dmb = _held_outcome(z)
    assert out.fit.fit.tau_us == TAU0
    cons = rescue_and_consolidate(
        U,
        dmb,
        SIGMA,
        out.fit,
        TAU0,
        T,
        max_rescue_rounds=2,
        conservative_kwargs={"shape": SHAPE},
        shape=SHAPE,
        frozen_skirts=frozen_skirts_for([fp], TAU0),
    )
    assert any(r.accepted for r in cons.rounds)
    inner = cons.fit.fit
    assert inner.tau_us == pytest.approx(TAU_TRUE, abs=0.3)
    assert _d18_chi2(z, inner) == pytest.approx(inner.chi_squared, rel=1e-9)


def test_apply_rescue_to_outcome_redraws_the_background_at_the_final_tau():
    # Catches: frozen_skirts=_outcome_skirts(outcome) dropped from the
    # rescue_and_consolidate call in _apply_rescue_to_outcome, and
    # _install_free_model / background_at_fitted_tau leaving the background at
    # the tau the rescue started from.
    z = _window_data([MISSED])
    out, _, _, _ = _held_outcome(z)
    _apply_rescue_to_outcome(
        SimpleNamespace(window_id=1),
        out,
        acquisition_us=T,
        tau0_us=TAU0,
        residual_edge_m=16,
        conservative_kwargs={"shape": SHAPE},
        max_residual_rescue_rounds=2,
        rescue_kwargs={},
    )
    assert any(ev.accepted for ev in out.rescue_events)
    _assert_outcome_is_d18(out, z)


def test_late_baseline_refit_follows_tau_with_the_frozen_skirt():
    # Catches: frozen_skirts dropped from the refit in
    # _apply_baseline_to_outcome (chi2 / residual then use the tau0 skirt) and
    # the re-drawn background not following the re-fit tau.
    z = _window_data()
    out, _, _, _ = _held_outcome(z)
    fired = _apply_baseline_to_outcome(
        out,
        acquisition_us=T,
        residual_edge_m=16,
        baseline_order=0,
        baseline_edge_threshold=0.0,
        baseline_smooth_threshold=0.0,
        tau0_us=TAU0,
        conservative_kwargs={},
    )
    assert fired is True
    assert out.baseline_applied and out.fit.fit.baseline_order == 0
    _assert_outcome_is_d18(out, z)


# ---------------------------------------------------------------------------
# Thaw co-fit: the dependent's other contributors live in the dependent frame
# ---------------------------------------------------------------------------
def _thaw_pair(*, fit_tau, tau_data):
    """A primary / dependent pair with centers 4 MHz apart.

    The primary carries a strong free line (the thaw candidate, in the primary
    frame at ``u_p``). The dependent sees that line as a frozen contributor and
    a second strong frozen contributor from another primary window, at offset
    ``other_dep`` in the DEPENDENT frame.
    """
    s = sideband_sign(Sideband.LOWER)
    fc_p, fc_d = 1000.0, 1004.0
    shift = s * (fc_d - fc_p)
    u_p = -0.3
    thaw_peak = ModelPeak(5000.0, u_p, 0.2)
    free_dep = ModelPeak(300.0, 0.4, 0.9)
    other_dep = ModelPeak(8000.0, -2.0, -0.4)
    f_thaw = fc_p + s * u_p
    thaw_dep = ModelPeak(thaw_peak.amplitude, s * (f_thaw - fc_d), thaw_peak.phase)
    assert thaw_dep.offset_mhz == pytest.approx(u_p - shift)
    u_prim = np.arange(-40, 41) / T
    u_dep = np.arange(-40, 41) / T
    sig = np.ones(u_prim.size)
    # Every physical line leaks into both windows (finite-T ringing), so the
    # primary's data carry the dependent's lines too, in the primary frame.
    free_in_prim = replace(free_dep, offset_mhz=free_dep.offset_mhz + shift)
    other_in_prim = replace(other_dep, offset_mhz=other_dep.offset_mhz + shift)
    z_prim = model_spectrum(
        u_prim, [thaw_peak, free_in_prim, other_in_prim], tau_data, T, shape=SHAPE
    )
    z_dep = model_spectrum(
        u_dep, [free_dep, thaw_dep, other_dep], tau_data, T, shape=SHAPE
    )
    thawed = FrozenPeak(0, 0, thaw_dep, f_thaw)
    f_other = fc_d + s * other_dep.offset_mhz
    other = FrozenPeak(1, 2, other_dep, f_other)
    other_prim = FrozenPeak(1, 2, other_in_prim, f_other)

    def outcome(wid, center, u, z, peaks, fixed):
        bg, dmb = subtract_frozen_background(u, z, fixed, tau_data, T, shape=SHAPE)
        out = fit_seeds_window_outcome(
            u,
            z,
            sig,
            center,
            bg,
            dmb,
            fixed,
            peaks,
            tau_data,
            T,
            {"fit_tau": False, "shape": SHAPE},
            None,
            DEFAULT_N_EFF_KIND,
            16,
            wid,
        )
        return out

    primary = outcome(0, fc_p, u_prim, z_prim, [thaw_peak], [other_prim])
    dependent = outcome(1, fc_d, u_dep, z_dep, [free_dep], [thawed, other])
    return primary, dependent, thawed, other, shift


def test_thaw_cofit_remaps_the_dependents_other_contributors():
    # Catches: dep_other built without the ``+ shift`` remap (or with the wrong
    # sign / the primary-frame offset). The dependent's other contributor is
    # then subtracted at its dependent-frame offset on the primary-frame joint
    # grid, 4 MHz away from where the data carry it, and the joint chi2 is the
    # chi2 of an 8000-amplitude unsubtracted line, not noise-free data.
    primary, dependent, thawed, other, shift = _thaw_pair(
        fit_tau=False, tau_data=TAU_TRUE
    )
    assert abs(shift) == pytest.approx(4.0)
    # The dependent is exact (noise-free) given its two frozen lines.
    assert dependent.fit.fit.chi_squared < 1e-6
    joint, idx = local_thaw_cofit(
        dependent,
        primary,
        thawed=thawed,
        sideband=Sideband.LOWER,
        tau0_us=TAU_TRUE,
        acquisition_us=T,
        fit_tau=False,
        shape=SHAPE,
    )
    # Right frame: the other contributor's skirt sits on its data, chi2 ~ 0.
    assert joint.chi_squared < 1e-6
    # The thawed peak is still at its primary-frame position and amplitude.
    pk = joint.peaks[int(idx[0])]
    assert pk.offset_mhz == pytest.approx(-0.3, abs=1e-4)
    assert pk.amplitude == pytest.approx(5000.0, rel=1e-4)
    # The dependent's free line lands at its offset in the primary frame.
    dep_free = max(joint.peaks[1:], key=lambda p: p.amplitude)
    assert dep_free.offset_mhz == pytest.approx(0.4 + shift, abs=1e-3)


def test_thaw_cofit_other_skirt_follows_the_joint_tau_in_the_dependent_frame():
    # Catches: the same missing remap on the FrozenSkirt side (dep_other feeds
    # both the subtraction and joint_skirts): with tau free the skirt follows
    # the trial tau at the unremapped offset and the fit cannot reach chi2 ~ 0
    # at the data's true tau even though the tau0 subtraction is right.
    primary, dependent, thawed, other, shift = _thaw_pair(
        fit_tau=True, tau_data=TAU_TRUE
    )
    joint, _ = local_thaw_cofit(
        dependent,
        primary,
        thawed=thawed,
        sideband=Sideband.LOWER,
        tau0_us=TAU0,
        acquisition_us=T,
        fit_tau=True,
        shape=SHAPE,
    )
    assert joint.tau_us == pytest.approx(TAU_TRUE, abs=1e-4)
    # Noise-free data: only the tau penalty / bound policy can leave residue.
    assert joint.chi_squared < 1e-6
