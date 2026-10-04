"""Every F-test site stores ``nan`` for a degenerate test, ``(0, 1)`` only for a
measured non-improvement.

The add-one-peak loop and both knockout calls go through
:func:`~ftmwpipeline.fitting.validation.calculate_chi_squared_improvement`; a
window with no residual degrees of freedom has no F-statistic, so its audit
step and its knockout ``p_value`` are ``nan`` (they used to read ``F = 0``,
``p = 1``, indistinguishable from a real non-improvement). The decisions are the
AICc gates', unchanged.
"""

from __future__ import annotations

import numpy as np

from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum
from ftmwpipeline.fitting.window_fit import (
    conservative_fit,
    fit_window,
    knockout_test,
)

T_US = 12.65
TAU_US = 5.0


def _tiny_window(n_bins, peaks, *, span=1.0, seed=1):
    """A window of *n_bins* complex bins (``2 * n_bins`` real data)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(-span, span, n_bins)
    z = model_spectrum(u, peaks, TAU_US, T_US)
    z = z + 0.05 * (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins))
    return u, z


class TestAddLoop:
    def test_a_step_with_no_residual_dof_is_undefined(self):
        # Two lines on three bins: 6 real data for 6 parameters once the second
        # line is added, so that step's F-test has zero residual dof.
        u, z = _tiny_window(3, [ModelPeak(5.0, -0.8, 0.3), ModelPeak(4.0, 0.7, 1.0)])
        res = conservative_fit(
            u,
            z,
            1.0,
            [-0.8, 0.7],
            TAU_US,
            T_US,
            fit_tau=False,
            seeder_max_k=1,
            candidate_passes=["primary", "gap"],
        )
        degenerate = [s for s in res.audit_trail if np.isnan(s.f_statistic)]
        assert degenerate, [(s.decision, s.f_statistic) for s in res.audit_trail]
        for step in degenerate:
            assert np.isnan(step.p_value)
            assert np.isfinite(step.chi2_before) and np.isfinite(step.chi2_after)
            assert step.chi2_after > 0.0  # undefined because of the dof alone

    def test_f_and_p_are_undefined_together(self):
        u, z = _tiny_window(3, [ModelPeak(5.0, -0.8, 0.3), ModelPeak(4.0, 0.7, 1.0)])
        res = conservative_fit(
            u,
            z,
            1.0,
            [-0.8, 0.7],
            TAU_US,
            T_US,
            fit_tau=False,
            seeder_max_k=1,
            candidate_passes=["primary", "gap"],
        )
        for step in res.audit_trail:
            if step.decision == "knockout-null":
                continue  # its p comes from the reversed knockout test
            assert np.isnan(step.f_statistic) == np.isnan(step.p_value)

    def test_a_well_posed_step_has_a_value(self):
        u, z = _tiny_window(12, [ModelPeak(5.0, -0.8, 0.3), ModelPeak(4.0, 0.7, 1.0)])
        res = conservative_fit(
            u,
            z,
            1.0,
            [-0.8, 0.7],
            TAU_US,
            T_US,
            fit_tau=False,
            seeder_max_k=1,
            candidate_passes=["primary", "gap"],
        )
        seed = res.audit_trail[0]
        assert seed.decision == "seed"
        assert np.isfinite(seed.f_statistic) and np.isfinite(seed.p_value)


class TestKnockout:
    def test_no_residual_dof_is_an_undefined_p_value(self):
        peaks = [ModelPeak(5.0, 0.0, 0.3), ModelPeak(3.0, 0.02, 1.0)]
        u, z = _tiny_window(2, peaks, span=0.05)
        fit = fit_window(u, z, 1.0, peaks, TAU_US, T_US, fit_tau=False)
        assert fit.n_data <= fit.n_params  # 4 real data, 6 parameters
        results = knockout_test(u, z, 1.0, fit, T_US)
        assert len(results) == 2
        for ko in results:
            assert np.isnan(ko.p_value)
            assert np.isfinite(ko.delta_chi2)
            # The decision is the AICc gate's; the F-test never drove it.
            assert isinstance(ko.supported, bool)

    def test_single_line_null_branch_is_undefined_too(self):
        """K = 1 -> 0 compares against the null with the same F-test."""
        peaks = [ModelPeak(5.0, 0.0, 0.3)]
        u = np.array([0.0])  # 2 real data, 3 parameters
        z = model_spectrum(u, peaks, TAU_US, T_US)
        fit = fit_window(u, z, 1.0, peaks, TAU_US, T_US, fit_tau=False)
        assert fit.n_data <= fit.n_params
        (ko,) = knockout_test(u, z, 1.0, fit, T_US)
        assert np.isnan(ko.p_value)

    def test_a_well_posed_test_has_a_p_value(self):
        peaks = [ModelPeak(5.0, 0.0, 0.3)]
        u, z = _tiny_window(40, peaks, span=0.05)
        fit = fit_window(u, z, 1.0, peaks, TAU_US, T_US, fit_tau=False)
        (ko,) = knockout_test(u, z, 1.0, fit, T_US)
        assert np.isfinite(ko.p_value) and 0.0 <= ko.p_value <= 1.0
