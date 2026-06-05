"""
Unit tests for the Stage 5 active-portion FT (task 6, D9).

Covers :mod:`ftmwpipeline.fitting.active_ft`:

* :func:`compute_active_ft` on a synthetic damped-cosine FID -- recover
  ``(A, f, phi, tau)`` to ~3 kHz on both sidebands, confirm
  ``alpha = N_active/N_padded`` is reported correctly, and confirm there is
  no phase ramp (the active-FT is in the ``[0, T]`` form natively, so the
  spectrum's phase at the line bin equals the input phase ``phi`` -- no
  ``exp(-i*2*pi*f_bb*t0)`` factor).
* The Stage 2 scatter noise estimator
  (:func:`~ftmwpipeline.preprocessing.noise_estimation.estimate_noise_scatter`)
  applied directly to the active-FT magnitude spectrum -- the per-bin RMS it
  reports matches the theoretical prediction for the noise actually present
  in the active-FT (no scale-conversion factor between persisted and active).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.active_ft import (
    ActiveFTResult,
    compute_active_ft,
)
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    h_T,
    model_spectrum,
    sideband_sign,
)
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter

# 2638-style acquisition, scaled down for fast tests. dt = 0.05 us, full
# record 60 us -> N_total = 1200 samples, active 50 us -> N_active = 1000.
DT_US = 0.05
N_TOTAL = 1200
START_US = 5.0
END_US = 55.0  # gives ~50 us active
T_ACTIVE_US = END_US - START_US
TAU_APOD_US = 5.0
PROBE_MHZ = 40960.0
SEED = 20260524


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _damped_cosine_fid(
    n: int,
    dt_us: float,
    amplitude: float,
    f_bb_mhz: float,
    phase: float,
    tau_intrinsic_us: float,
    *,
    start_us: float = 0.0,
) -> np.ndarray:
    """Build a damped-cosine FID: ``A * cos(2pi f t + phi) * exp(-(t-t0)/tau)``.

    The cosine starts at ``t = start_us`` (so the signal is zero before that --
    matching the way the active region is windowed in Stage 1). Returns a real
    1-D array of length ``n``.
    """
    t_us = np.arange(n) * dt_us
    out = np.zeros(n, dtype=float)
    active = t_us >= start_us
    rel = t_us[active] - start_us
    out[active] = (
        amplitude
        * np.cos(2.0 * np.pi * f_bb_mhz * rel + phase)
        * np.exp(-rel / tau_intrinsic_us)
    )
    return out


def _fit_one_line(
    freq_mhz: np.ndarray,
    z: np.ndarray,
    f0_guess_mhz: float,
    tau_guess_us: float,
    acquisition_us: float,
    sideband: Sideband,
) -> tuple[float, float, float, float, float]:
    """Tiny single-line LSQ on an active-FT slice. Returns (A, f_mhz, phi, tau, rms_rel)."""
    s = sideband_sign(sideband)

    def model(params: np.ndarray, u: np.ndarray) -> np.ndarray:
        amp, off, ph, tau = params
        return 0.5 * amp * np.exp(1j * ph) * h_T(u - off, tau, acquisition_us)

    f_c = f0_guess_mhz
    u_grid = s * (freq_mhz - f_c)

    def residual(params: np.ndarray) -> np.ndarray:
        m = model(params, u_grid)
        r = z - m
        return np.concatenate([r.real, r.imag])

    amp_guess = (
        2.0
        * float(np.max(np.abs(z)))
        / max(effective_tau(tau_guess_us, acquisition_us), 1e-9)
    )
    sol = least_squares(
        residual,
        np.asarray([amp_guess, 0.0, 0.0, tau_guess_us], dtype=float),
        method="trf",
        bounds=(
            [0.0, -1.0, -np.pi, tau_guess_us / 5.0],
            [np.inf, 1.0, np.pi, tau_guess_us * 5.0],
        ),
        max_nfev=400,
    )
    amp, off, ph, tau = sol.x
    f_recovered = f_c + s * float(off)
    rms_rel = float(np.linalg.norm(sol.fun)) / float(
        np.linalg.norm(np.concatenate([z.real, z.imag]))
    )
    return float(amp), float(f_recovered), float(ph), float(tau), rms_rel


# ---------------------------------------------------------------------------
# compute_active_ft -- structural checks
# ---------------------------------------------------------------------------
class TestComputeActiveFTStructure:
    def test_alpha_equals_active_over_padded(self):
        """Reported ``alpha`` is exactly ``N_active / N_padded``."""
        fid = np.zeros(N_TOTAL)
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_padded=2048,
        )
        assert result.n_active == 1000
        assert result.n_padded == 2048
        assert result.alpha == pytest.approx(1000.0 / 2048.0)

    def test_freq_grid_bin_spacing_is_one_over_t_active(self):
        """Active-FT bin spacing in MHz is ``1 / T_active``."""
        fid = np.zeros(N_TOTAL)
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.UPPER,
            n_padded=N_TOTAL,
        )
        bin_widths = np.abs(np.diff(result.freq_mhz))
        # Upper sideband: monotone ascending.
        expected_df = 1.0 / T_ACTIVE_US
        assert np.allclose(bin_widths, expected_df, rtol=1e-9)

    def test_lower_sideband_descending(self):
        """Lower sideband: molecular axis descends (baseband ascends from DC)."""
        fid = np.zeros(N_TOTAL)
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_padded=N_TOTAL,
        )
        assert result.freq_mhz[0] > result.freq_mhz[-1]
        # DC bin == probe frequency.
        assert result.freq_mhz[0] == pytest.approx(PROBE_MHZ)

    def test_upper_sideband_ascending(self):
        fid = np.zeros(N_TOTAL)
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.UPPER,
            n_padded=N_TOTAL,
        )
        assert result.freq_mhz[0] < result.freq_mhz[-1]
        assert result.freq_mhz[0] == pytest.approx(PROBE_MHZ)


class TestComputeActiveFTInputValidation:
    def test_empty_active_region_raises(self):
        fid = np.zeros(N_TOTAL)
        with pytest.raises(ValueError, match="end_us must be greater"):
            compute_active_ft(
                fid,
                sample_dt_us=DT_US,
                start_us=10.0,
                end_us=10.0,
                expf_us=TAU_APOD_US,
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_padded=N_TOTAL,
            )

    def test_non_positive_dt_raises(self):
        with pytest.raises(ValueError, match="sample_dt_us must be positive"):
            compute_active_ft(
                np.zeros(10),
                sample_dt_us=0.0,
                start_us=0.0,
                end_us=1.0,
                expf_us=None,
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_padded=10,
            )

    def test_non_positive_expf_disables_apodization(self):
        """expf_us <= 0 is normalised to None (no apodization) — must
        produce the same spectrum as an explicit ``expf_us=None`` call."""
        rng = np.random.default_rng(0)
        fid = rng.standard_normal(N_TOTAL)
        kwargs = dict(
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_padded=N_TOTAL,
        )
        result_none = compute_active_ft(fid, expf_us=None, **kwargs)
        result_zero = compute_active_ft(fid, expf_us=0.0, **kwargs)
        result_neg = compute_active_ft(fid, expf_us=-1.0, **kwargs)
        np.testing.assert_allclose(
            result_zero.complex_spectrum, result_none.complex_spectrum
        )
        np.testing.assert_allclose(
            result_neg.complex_spectrum, result_none.complex_spectrum
        )

    def test_n_padded_smaller_than_active_raises(self):
        with pytest.raises(ValueError, match="n_padded"):
            compute_active_ft(
                np.zeros(N_TOTAL),
                sample_dt_us=DT_US,
                start_us=START_US,
                end_us=END_US,
                expf_us=TAU_APOD_US,
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_padded=100,
            )


# ---------------------------------------------------------------------------
# compute_active_ft -- parameter recovery
# ---------------------------------------------------------------------------
class TestRecoverDampedCosine:
    """A synthetic ``A cos(2pi f t + phi) e^{-(t-t0)/tau}`` FID, computed via
    :func:`compute_active_ft` and fit on its active-FT, must recover the input
    ``(A, f, phi, tau_combined)`` to within a kHz / a few percent. Tests both
    sidebands -- the sideband sign is load-bearing for the demod (a wrong sign
    is a silent hundreds-of-kHz frequency bias).
    """

    @pytest.mark.parametrize("sideband", [Sideband.LOWER, Sideband.UPPER])
    def test_recovers_isolated_line(self, sideband):
        s = sideband_sign(sideband)
        amplitude = 2.5
        # Park the line a few MHz away from DC -- f_bb in MHz.
        f_bb_mhz = 3.7
        f_molecular = PROBE_MHZ + s * f_bb_mhz
        phase = 0.85
        # Intrinsic natural decay long compared to active duration so the
        # combined decay tau_combined ~ tau_apod.
        tau_intrinsic_us = 1000.0
        # tau_combined: 1/tau_c = 1/tau_intrinsic + 1/tau_apod -> ~tau_apod.
        tau_combined = 1.0 / (1.0 / tau_intrinsic_us + 1.0 / TAU_APOD_US)

        fid = _damped_cosine_fid(
            N_TOTAL,
            DT_US,
            amplitude,
            f_bb_mhz,
            phase,
            tau_intrinsic_us,
            start_us=START_US,
        )

        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=sideband,
            n_padded=N_TOTAL,
        )

        # Slice a +-1 MHz window around the line.
        mask = np.abs(result.freq_mhz - f_molecular) < 1.0
        amp_fit, f_fit, ph_fit, tau_fit, rms_rel = _fit_one_line(
            result.freq_mhz[mask],
            result.complex_spectrum[mask],
            f0_guess_mhz=f_molecular,
            tau_guess_us=TAU_APOD_US,
            acquisition_us=T_ACTIVE_US,
            sideband=sideband,
        )
        # The single-line fit absorbs almost all the data; the few-percent
        # residual is the conjugate "mirror" skirt of the cos(2*pi*f*t)
        # source (h_T evaluated at 2*f_bb away from the line), which the
        # plan defers to the near-DC mirror term (D-4).
        assert rms_rel < 0.05
        assert abs(f_fit - f_molecular) < 0.003  # 3 kHz
        assert abs(amp_fit - amplitude) < 0.05 * amplitude
        # Wrap phase difference into (-pi, pi].
        dphi = ((ph_fit - phase + np.pi) % (2.0 * np.pi)) - np.pi
        assert abs(dphi) < 0.05
        assert abs(tau_fit - tau_combined) < 0.1 * tau_combined


# ---------------------------------------------------------------------------
# No phase ramp -- the [0, T] form claim
# ---------------------------------------------------------------------------
class TestNoPhaseRamp:
    """The active-FT is in the ``[0, T]`` form, so a damped cosine of phase
    ``phi`` lands in a bin whose complex value is ``0.5 * A * h_T(Δf) *
    exp(i*phi)`` -- *no* ``exp(-i*2*pi*f_bb*t0)`` ramp. We confirm this by
    comparing the active-FT bin phase at the line center against the analytic
    ``h_T``-form prediction at the bin's offset.
    """

    @pytest.mark.parametrize("sideband", [Sideband.LOWER, Sideband.UPPER])
    def test_phase_matches_h_T_at_line(self, sideband):
        s = sideband_sign(sideband)
        amplitude = 1.7
        f_bb_mhz = 2.3
        f_molecular = PROBE_MHZ + s * f_bb_mhz
        phase = -1.4
        tau_intrinsic_us = 1000.0
        tau_combined = 1.0 / (1.0 / tau_intrinsic_us + 1.0 / TAU_APOD_US)

        fid = _damped_cosine_fid(
            N_TOTAL,
            DT_US,
            amplitude,
            f_bb_mhz,
            phase,
            tau_intrinsic_us,
            start_us=START_US,
        )
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=sideband,
            n_padded=N_TOTAL,
        )

        # Bin closest to the line.
        bin_idx = int(np.argmin(np.abs(result.freq_mhz - f_molecular)))
        bin_freq = float(result.freq_mhz[bin_idx])
        delta_f = s * (bin_freq - f_molecular)  # signed baseband offset
        expected = (
            0.5
            * amplitude
            * np.exp(1j * phase)
            * h_T(np.asarray([delta_f]), tau_combined, T_ACTIVE_US)[0]
        )
        got = result.complex_spectrum[bin_idx]

        # Phase agrees within a few percent (DC removal + finite-grid artefacts
        # contribute a small bias).
        phi_expected = np.angle(expected)
        phi_got = np.angle(got)
        dphi = ((phi_got - phi_expected + np.pi) % (2.0 * np.pi)) - np.pi
        assert abs(dphi) < 0.05

        # If a phase ramp were present, the offset from "natural" h_T form
        # would be exp(-2j*pi*f_bb*t0). For f_bb~2.3 MHz and t0=5 us this
        # accumulates ~14*pi radians (mod 2pi -- huge offset). The fact that
        # phi_got agrees with phi_expected confirms no such ramp is present.


# ---------------------------------------------------------------------------
# Per-bin noise: Stage 2 estimator run directly on the active-FT
# ---------------------------------------------------------------------------
class TestStage2NoiseOnActiveFT:
    """The active-FT's per-bin noise is measured by running the existing
    Stage 2 scatter estimator
    (:func:`~ftmwpipeline.preprocessing.noise_estimation.estimate_noise_scatter`)
    on the active-FT magnitude spectrum -- the *same* algorithm Stage 2 uses
    on the persisted spectrum, just applied to the active-FT instead. No
    scale conversion: σ comes from the same spectrum the fit sees, so any
    normalization choices cancel by construction.
    """

    def test_noise_rms_matches_time_domain_prediction(self):
        """Run Stage 2 on a noise-only active-FT and verify the per-bin RMS
        matches the theoretical prediction from the time-domain noise.

        For ``active_ft = dt_us * rfft(active_samples)`` with no apodization
        and time-domain noise variance ``sigma_t**2``, each non-DC / non-
        Nyquist bin has complex variance ``dt_us**2 * N_active * sigma_t**2``,
        i.e. complex RMS ``dt_us * sqrt(N_active) * sigma_t``.
        """
        rng = np.random.default_rng(SEED)
        n_total = 20000
        n_padded = 32768
        sigma_t = 0.5
        fid = rng.normal(0.0, sigma_t, n_total)

        start_us = 5.0
        end_us = 5.0 + (n_total - 100) * DT_US

        active = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=start_us,
            end_us=end_us,
            expf_us=None,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_padded=n_padded,
            rdc=False,
        )

        # Stage 2 wants ascending frequencies; LSB descends, so sort first.
        order = np.argsort(active.freq_mhz)
        freq_sorted = active.freq_mhz[order]
        mag_sorted = np.abs(active.complex_spectrum[order])
        noise_result = estimate_noise_scatter(
            frequencies=freq_sorted,
            magnitudes=mag_sorted,
        )

        # All-noise spectrum: Stage 2 should report ~constant RMS across the
        # band. Theoretical complex RMS = dt_us * sqrt(N_active) * sigma_t.
        # The scatter estimator high-passes |X| (subtracting a median-filter
        # pedestal), which on a *pure-noise* synthetic with no leakage pedestal
        # reads a few % low relative to the ideal -- it is calibrated to ~1.0x
        # against frame-difference truth on real leakage-bearing spectra. The
        # convention check is the dt*sqrt(N) scaling, so ~15% is the bar here.
        theory_complex_rms = DT_US * sigma_t * float(np.sqrt(active.n_active))
        median_rms = float(np.median(noise_result.rms_noise))
        assert median_rms == pytest.approx(theory_complex_rms, rel=0.15)

    def test_noise_rms_matches_under_apodization(self):
        """With the canonical Stage 1 apodization, Stage 2 still recovers the
        per-bin complex RMS to within a few percent. The apodization scales
        the bin variance by ``sum(apod**2)`` instead of ``N_active``; Stage 2
        measures whatever is in the spectrum, so the result tracks the
        apodized variance automatically.
        """
        rng = np.random.default_rng(SEED + 7)
        n_total = 20000
        n_padded = 32768
        sigma_t = 0.3
        fid = rng.normal(0.0, sigma_t, n_total)

        start_us = 5.0
        end_us = 5.0 + (n_total - 100) * DT_US

        active = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=start_us,
            end_us=end_us,
            expf_us=TAU_APOD_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_padded=n_padded,
            rdc=True,
        )

        order = np.argsort(active.freq_mhz)
        freq_sorted = active.freq_mhz[order]
        mag_sorted = np.abs(active.complex_spectrum[order])
        noise_result = estimate_noise_scatter(
            frequencies=freq_sorted,
            magnitudes=mag_sorted,
        )

        # Theoretical per-bin complex RMS with apodization:
        # dt_us * sigma_t * sqrt(sum(apod**2)).
        rel_t = np.arange(active.n_active) * DT_US
        apod = np.exp(-rel_t / TAU_APOD_US)
        theory_complex_rms = DT_US * sigma_t * float(np.sqrt(np.sum(apod**2)))
        median_rms = float(np.median(noise_result.rms_noise))
        assert median_rms == pytest.approx(theory_complex_rms, rel=0.15)


# ---------------------------------------------------------------------------
# ActiveFTResult dataclass sanity
# ---------------------------------------------------------------------------
class TestActiveFTResult:
    def test_can_be_constructed_directly(self):
        """Synthetic tests build ActiveFTResult directly without computing it."""
        result = ActiveFTResult(
            freq_mhz=np.array([100.0, 101.0]),
            complex_spectrum=np.array([1 + 0j, 0 + 1j]),
            alpha=0.5,
            n_active=10,
            n_padded=20,
        )
        assert result.alpha == 0.5
        assert result.n_active == 10
        assert result.n_padded == 20
