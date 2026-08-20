"""
Unit tests for P3 birth-site stamping on the Stage 5 in-walk auto-merge
passes (:func:`~ftmwpipeline._internal.stage5_impl._collapse_outcome` and
:func:`~ftmwpipeline._internal.stage5_impl._degenerate_merge_trial_outcome`).

Both passes build a merged-line ``ModelPeak`` seed from two collapsing
fitted lines and refit it via :func:`~ftmwpipeline.fitting.plan_execution.
refit_outcome`. Per the peak-identity design
(``scratch/peak-identity-plan.md``), a merge product is a new entity and
must be stamped from its own seed position -- never left to inherit either
parent's identifier, and never re-derived from the refit's converged
position.

The scenario: two sub-resolution lines close enough to trigger the VIF
collapse gate, built with :mod:`ftmwpipeline.fitting.peak_model`'s ``h_T``
directly (mirrors ``test_plan_execution.py``'s ``_synth_spectrum``), noise-
free so the collapse decision is deterministic.
"""

from __future__ import annotations

import numpy as np

from ftmwpipeline._internal.stage5_impl import _collapse_outcome
from ftmwpipeline.core.data_structures import FitWindow, Sideband, WindowPlan
from ftmwpipeline.fitting.active_ft import ActiveFTResult, peak_uid_from_offset
from ftmwpipeline.fitting.peak_model import effective_tau, h_T
from ftmwpipeline.fitting.plan_execution import execute_plan

T_US = 12.65
TAU_US = 5.0
DF_MHZ = 0.0122
PROBE_MHZ = 40960.0
SIDEBAND = Sideband.LOWER
N_ACTIVE = 1000
SAMPLE_DT_US = 0.05


def _synth_spectrum(
    freq_array, peaks_mhz_amp_phase, tau_us=TAU_US, acquisition_us=T_US
):
    s = -1.0  # lower sideband
    z = np.zeros(freq_array.shape, dtype=np.complex128)
    for f_j, amp, phase in peaks_mhz_amp_phase:
        du = s * (freq_array - f_j)
        z += 0.5 * amp * np.exp(1j * phase) * h_T(du, tau_us, acquisition_us)
    return z


def _amp_for_snr(snr, sigma=1.0):
    return 2.0 * snr * sigma / effective_tau(TAU_US, T_US)


def _make_active_ft(freq_array, complex_spectrum, alpha=1.0):
    spec = np.asarray(complex_spectrum, dtype=np.complex128)
    n_active = spec.size
    n_raw = max(int(round(n_active / alpha)), n_active)
    return ActiveFTResult(
        freq_mhz=np.asarray(freq_array, dtype=float),
        complex_spectrum=spec,
        alpha=float(alpha),
        n_active=n_active,
        n_raw=n_raw,
    )


def _degenerate_pair_outcome():
    """Fit one window on two sub-resolution lines whose covariance is
    degenerate enough to clear the VIF collapse gate at a permissive
    threshold. Returns the converged (pre-collapse) WindowOutcome."""
    f0, f1 = 36100.0, 36100.02
    center = 0.5 * (f0 + f1)
    freq_array = np.arange(center - 5.0, center + 5.0, DF_MHZ)
    a0 = _amp_for_snr(200.0)
    a1 = _amp_for_snr(15.0)
    spectrum = _synth_spectrum(freq_array, [(f0, a0, 0.1), (f1, a1, 0.2)])
    rms_noise = np.full(freq_array.size, 1.0)
    win = FitWindow(
        window_id=0,
        freq_range=(center - 4.0, center + 4.0),
        free_peak_indices=[0, 1],
        fixed_contributors=[],
        batch=0,
    )
    plan = WindowPlan(windows=[win], dependency_edges=[], topological_order=[0])
    out = execute_plan(
        plan,
        _make_active_ft(freq_array, spectrum),
        rms_noise,
        [f0, f1],
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
    )
    return out.window_outcomes[0], center


class TestCollapseOutcomePeakUid:
    """The VIF-collapse merge product (stage5_impl._collapse_outcome)."""

    def _run_collapse(self, outcome, **overrides):
        records: list = []
        kwargs = dict(
            vif_threshold=1.0,
            frac_threshold=0.0,
            frac_max_sep_res=100.0,
            max_sep_res=100.0,
            res_element_mhz=1.0 / T_US,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            snap_tol_mhz=0.5,
            max_iterations=5,
            records=records,
        )
        kwargs.update(overrides)
        result = _collapse_outcome(outcome, **kwargs)
        return result, records

    def test_no_frame_context_leaves_merge_product_unstamped(self):
        """Omitting probe_freq_mhz/n_active/sample_dt_us must not raise and
        must not invent a fallback identifier -- absent stays absent."""
        outcome, _center = _degenerate_pair_outcome()
        result, records = self._run_collapse(outcome)
        assert (
            len(records) == 1
        ), "the pair must actually collapse for this test to mean anything"
        assert result.fit.n_peaks == 1
        assert result.fit.peaks[0].peak_uid is None

    def test_merge_product_is_stamped_from_its_seed_not_the_refit(self):
        outcome, center = _degenerate_pair_outcome()
        result, records = self._run_collapse(
            outcome,
            probe_freq_mhz=PROBE_MHZ,
            n_active=N_ACTIVE,
            sample_dt_us=SAMPLE_DT_US,
        )
        assert (
            len(records) == 1
        ), "the pair must actually collapse for this test to mean anything"
        assert result.fit.n_peaks == 1
        merged = result.fit.peaks[0]
        assert merged.peak_uid is not None

        # The seed offset the merge was stamped from (recorded on the
        # collapse provenance record, in the molecular frame).
        seed_freq_mhz = records[0]["merged_frequency_mhz"]
        s = -1.0
        seed_offset_mhz = s * (seed_freq_mhz - center)
        expected_uid = peak_uid_from_offset(
            seed_offset_mhz, center, SIDEBAND, PROBE_MHZ, N_ACTIVE, SAMPLE_DT_US
        )
        assert merged.peak_uid == expected_uid

        # The negative case that matters most: the refit that followed the
        # merge moved the line off its seed position (a genuine re-
        # convergence, not a no-op), so the identifier recomputed from the
        # FITTED offset must differ from the stamped one -- proving the
        # stamp was never re-derived from the fit.
        recomputed_from_fit = peak_uid_from_offset(
            merged.offset_mhz, center, SIDEBAND, PROBE_MHZ, N_ACTIVE, SAMPLE_DT_US
        )
        assert merged.offset_mhz != seed_offset_mhz, (
            "the post-collapse refit did not move the merged line at all, so "
            "this case cannot distinguish carry from re-derivation"
        )
        assert merged.peak_uid != recomputed_from_fit
