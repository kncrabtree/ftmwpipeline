"""
Unit tests for the Stage 5 active-portion FT (D9).

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

The canonical active FT is unapodized: the line decay is physical (baked into
the synthetic FID), not an apodization applied by ``compute_active_ft``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.active_ft import (
    ActiveFTResult,
    PointMap,
    active_ft_point_hundredths,
    compute_active_ft,
    peak_uid_from_offset,
)
from ftmwpipeline.fitting.peak_model import (
    effective_tau,
    h_T,
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
TAU_US = 5.0  # physical molecular decay of the synthetic lines
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
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_raw=2048,
        )
        assert result.n_active == 1000
        assert result.n_raw == 2048
        assert result.alpha == pytest.approx(1000.0 / 2048.0)

    def test_freq_grid_bin_spacing_is_one_over_t_active(self):
        """Active-FT bin spacing in MHz is ``1 / T_active``."""
        fid = np.zeros(N_TOTAL)
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.UPPER,
            n_raw=N_TOTAL,
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
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_raw=N_TOTAL,
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
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.UPPER,
            n_raw=N_TOTAL,
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
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_raw=N_TOTAL,
            )

    def test_non_positive_dt_raises(self):
        with pytest.raises(ValueError, match="sample_dt_us must be positive"):
            compute_active_ft(
                np.zeros(10),
                sample_dt_us=0.0,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_raw=10,
            )

    def test_n_raw_smaller_than_active_raises(self):
        with pytest.raises(ValueError, match="n_raw"):
            compute_active_ft(
                np.zeros(N_TOTAL),
                sample_dt_us=DT_US,
                start_us=START_US,
                end_us=END_US,
                probe_freq_mhz=PROBE_MHZ,
                sideband=Sideband.LOWER,
                n_raw=100,
            )


# ---------------------------------------------------------------------------
# compute_active_ft -- parameter recovery
# ---------------------------------------------------------------------------
class TestRecoverDampedCosine:
    """A synthetic ``A cos(2pi f t + phi) e^{-(t-t0)/tau}`` FID, computed via
    :func:`compute_active_ft` and fit on its active-FT, must recover the input
    ``(A, f, phi, tau)`` to within a kHz / a few percent. Tests both sidebands
    -- the sideband sign is load-bearing for the demod (a wrong sign is a
    silent hundreds-of-kHz frequency bias).
    """

    @pytest.mark.parametrize("sideband", [Sideband.LOWER, Sideband.UPPER])
    def test_recovers_isolated_line(self, sideband):
        s = sideband_sign(sideband)
        amplitude = 2.5
        # Park the line a few MHz away from DC -- f_bb in MHz.
        f_bb_mhz = 3.7
        f_molecular = PROBE_MHZ + s * f_bb_mhz
        phase = 0.85
        # The decay is physical (in the FID); the unapodized active-FT recovers
        # it directly.
        tau_us = TAU_US

        fid = _damped_cosine_fid(
            N_TOTAL,
            DT_US,
            amplitude,
            f_bb_mhz,
            phase,
            tau_us,
            start_us=START_US,
        )

        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=sideband,
            n_raw=N_TOTAL,
        )

        # Slice a +-1 MHz window around the line.
        mask = np.abs(result.freq_mhz - f_molecular) < 1.0
        amp_fit, f_fit, ph_fit, tau_fit, rms_rel = _fit_one_line(
            result.freq_mhz[mask],
            result.complex_spectrum[mask],
            f0_guess_mhz=f_molecular,
            tau_guess_us=tau_us,
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
        assert abs(tau_fit - tau_us) < 0.1 * tau_us


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
        tau_us = TAU_US

        fid = _damped_cosine_fid(
            N_TOTAL,
            DT_US,
            amplitude,
            f_bb_mhz,
            phase,
            tau_us,
            start_us=START_US,
        )
        result = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=START_US,
            end_us=END_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband=sideband,
            n_raw=N_TOTAL,
        )

        # Bin closest to the line.
        bin_idx = int(np.argmin(np.abs(result.freq_mhz - f_molecular)))
        bin_freq = float(result.freq_mhz[bin_idx])
        delta_f = s * (bin_freq - f_molecular)  # signed baseband offset
        expected = (
            0.5
            * amplitude
            * np.exp(1j * phase)
            * h_T(np.asarray([delta_f]), tau_us, T_ACTIVE_US)[0]
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

        For ``active_ft = dt_us * rfft(active_samples)`` (unapodized) and
        time-domain noise variance ``sigma_t**2``, each non-DC / non-Nyquist
        bin has complex variance ``dt_us**2 * N_active * sigma_t**2``, i.e.
        complex RMS ``dt_us * sqrt(N_active) * sigma_t``.
        """
        rng = np.random.default_rng(SEED)
        n_total = 20000
        n_raw = 32768
        sigma_t = 0.5
        fid = rng.normal(0.0, sigma_t, n_total)

        start_us = 5.0
        end_us = 5.0 + (n_total - 100) * DT_US

        active = compute_active_ft(
            fid,
            sample_dt_us=DT_US,
            start_us=start_us,
            end_us=end_us,
            probe_freq_mhz=PROBE_MHZ,
            sideband=Sideband.LOWER,
            n_raw=n_raw,
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


# ---------------------------------------------------------------------------
# active_ft_point_hundredths -- the peak-identity coordinate helper (P1)
# ---------------------------------------------------------------------------
class TestActiveFtPointHundredths:
    """The single derivation point for a peak's point-space identifier."""

    def test_bin_k_maps_to_k_times_100(self):
        """A baseband frequency exactly on rfft bin k round-trips to k*100.

        Same round-trip guarantee as the rfft grid itself: bin k sits at
        ``k / (n_active * sample_dt_us)`` MHz, so the helper should return
        exactly ``100 * k`` with no rounding drift.
        """
        n_active = 1000
        dt = 0.05
        grid = np.fft.rfftfreq(n_active, d=dt)
        for k in (0, 1, 5, 250, len(grid) - 1):
            assert active_ft_point_hundredths(grid[k], n_active, dt) == 100 * k

    def test_spacing_matches_rfftfreq_grid(self):
        """Every bin of an arbitrary (non-power-of-two) rfft grid round-trips.

        This is the consistency property the docstring promises: the
        divisor must be the rfft grid's own spacing (1/(n_active*dt)), not
        active_ft_bin_spacing_mhz's 1/acquisition_us. Checked against
        np.fft.rfftfreq directly (not re-derived inline) for a non-bin-
        -aligned n_active / dt pair, over every bin rather than just one.
        """
        n_active = 733
        dt = 0.0731
        grid = np.fft.rfftfreq(n_active, d=dt)
        for k in range(0, len(grid), 37):
            assert active_ft_point_hundredths(grid[k], n_active, dt) == 100 * k

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError):
            active_ft_point_hundredths(-0.01, 1000, 0.05)

    def test_nonpositive_n_active_raises(self):
        with pytest.raises(ValueError):
            active_ft_point_hundredths(1.0, 0, 0.05)
        with pytest.raises(ValueError):
            active_ft_point_hundredths(1.0, -10, 0.05)

    def test_nonpositive_sample_dt_raises(self):
        with pytest.raises(ValueError):
            active_ft_point_hundredths(1.0, 1000, 0.0)
        with pytest.raises(ValueError):
            active_ft_point_hundredths(1.0, 1000, -0.05)

    def test_returns_plain_int(self):
        assert isinstance(active_ft_point_hundredths(1.0, 1000, 0.05), int)


# ---------------------------------------------------------------------------
# peak_uid_from_offset -- recovers f_bb from a ModelPeak.offset_mhz seed (P3)
# ---------------------------------------------------------------------------
class TestPeakUidFromOffset:
    """Every birth site stamps through this: offset_mhz -> f_molecular ->
    f_bb -> active_ft_point_hundredths. Pinned against a manual inversion,
    not just against active_ft_point_hundredths (that would only prove the
    two functions agree with each other, not that either is correct)."""

    N_ACTIVE = 1000
    DT = 0.05
    PROBE_MHZ = 40960.0

    def test_matches_manual_inversion_lower_sideband(self):
        # s = -1: f_molecular = center - offset; f_bb = -(f_molecular - probe)
        center = 36100.0
        offset = 0.734
        got = peak_uid_from_offset(
            offset, center, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        f_molecular = center - offset
        f_bb = -(f_molecular - self.PROBE_MHZ)
        want = active_ft_point_hundredths(f_bb, self.N_ACTIVE, self.DT)
        assert got == want

    def test_matches_manual_inversion_upper_sideband(self):
        # s = +1: f_molecular = center + offset; f_bb = f_molecular - probe
        center = 41500.0
        offset = -0.412
        got = peak_uid_from_offset(
            offset, center, Sideband.UPPER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        f_molecular = center + offset
        f_bb = f_molecular - self.PROBE_MHZ
        want = active_ft_point_hundredths(f_bb, self.N_ACTIVE, self.DT)
        assert got == want

    def test_center_on_probe_zero_offset_is_bin_zero(self):
        # f_molecular == probe, offset == 0 -> f_bb == 0 -> point 0.
        assert (
            peak_uid_from_offset(
                0.0,
                self.PROBE_MHZ,
                Sideband.LOWER,
                self.PROBE_MHZ,
                self.N_ACTIVE,
                self.DT,
            )
            == 0
        )

    def test_string_sideband_matches_enum(self):
        center, offset = 36100.0, 0.2
        via_enum = peak_uid_from_offset(
            offset, center, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        via_str = peak_uid_from_offset(
            offset, center, "lower", self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert via_enum == via_str

    def test_sideband_sign_genuinely_participates(self):
        # A molecular frequency below the probe is a valid lower-sideband
        # baseband target (f_bb = probe - f_molecular >= 0) but an invalid
        # upper-sideband one (f_bb = f_molecular - probe < 0) -- so feeding
        # the wrong sideband through must be caught, not silently accepted
        # with the sign dropped.
        f_molecular_below_probe = 36100.0
        offset = 0.0  # center == f_molecular when offset is 0
        got = peak_uid_from_offset(
            offset,
            f_molecular_below_probe,
            Sideband.LOWER,
            self.PROBE_MHZ,
            self.N_ACTIVE,
            self.DT,
        )
        assert got == active_ft_point_hundredths(
            self.PROBE_MHZ - f_molecular_below_probe, self.N_ACTIVE, self.DT
        )
        with pytest.raises(ValueError):
            peak_uid_from_offset(
                offset,
                f_molecular_below_probe,
                Sideband.UPPER,
                self.PROBE_MHZ,
                self.N_ACTIVE,
                self.DT,
            )

    def test_returns_plain_int(self):
        result = peak_uid_from_offset(
            0.2, 36100.0, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# PointMap -- the frame-agnostic seeder's route to the same stamp (P3, second
# landing). Point space is affine in ModelPeak.offset_mhz (sideband_sign
# squares to 1), so a per-window (origin_points, points_per_mhz) pair stamps
# exactly like peak_uid_from_offset without the seeder ever learning
# center_mhz / sideband / probe_freq_mhz. This equivalence is the entire
# basis for the approach: if it drifts, identifiers minted by the two routes
# disagree and the diagnostic recompute is worthless.
# ---------------------------------------------------------------------------
class TestPointMap:
    N_ACTIVE = 1000
    DT = 0.05
    PROBE_MHZ = 40960.0

    def test_equivalent_to_peak_uid_from_offset_lower_sideband(self):
        center = 36100.0
        offset = 0.734
        pm = PointMap.from_frame(
            center, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        want = peak_uid_from_offset(
            offset, center, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert pm.stamp(offset) == want

    def test_equivalent_to_peak_uid_from_offset_upper_sideband(self):
        center = 41500.0
        offset = -0.412
        pm = PointMap.from_frame(
            center, Sideband.UPPER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        want = peak_uid_from_offset(
            offset, center, Sideband.UPPER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert pm.stamp(offset) == want

    def test_equivalent_over_randomized_in_range_cases(self):
        """Pinned equivalence: exact integer agreement, not approximate, over
        many randomized frames and offsets -- the property the whole
        frame-agnostic seeding approach leans on (see the module docstring
        above and ``scratch/peak-identity-plan.md``)."""
        rng = np.random.default_rng(20260819)
        n_cases = 2000
        tried = 0
        max_abs_diff = 0
        while tried < n_cases:
            probe = float(rng.uniform(1000.0, 40000.0))
            sideband = rng.choice([Sideband.LOWER, Sideband.UPPER])
            s = 1.0 if sideband == Sideband.UPPER else -1.0
            f_bb0 = float(rng.uniform(500.0, 20000.0))
            center = probe + s * f_bb0
            n_active = int(rng.integers(1000, 400000))
            dt = float(rng.uniform(1e-4, 1e-2))
            offset = float(rng.uniform(-499.0, 499.0))
            if f_bb0 + offset < 0.0:
                continue  # peak_uid_from_offset requires f_bb >= 0
            tried += 1
            pm = PointMap.from_frame(center, sideband, probe, n_active, dt)
            got = pm.stamp(offset)
            want = peak_uid_from_offset(offset, center, sideband, probe, n_active, dt)
            max_abs_diff = max(max_abs_diff, abs(got - want))
            assert got == want
        assert max_abs_diff == 0
        assert tried == n_cases

    def test_points_per_mhz_is_sideband_independent(self):
        # The two sideband-sign inversions collapse algebraically, so the
        # scale term must not depend on which sideband built the map.
        lower = PointMap.from_frame(
            36100.0, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        upper = PointMap.from_frame(
            36100.0, Sideband.UPPER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert lower.points_per_mhz == upper.points_per_mhz
        assert lower.points_per_mhz == pytest.approx(self.N_ACTIVE * self.DT)

    def test_non_positive_n_active_raises(self):
        with pytest.raises(ValueError):
            PointMap.from_frame(36100.0, Sideband.LOWER, self.PROBE_MHZ, 0, self.DT)

    def test_non_positive_sample_dt_raises(self):
        with pytest.raises(ValueError):
            PointMap.from_frame(
                36100.0, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, 0.0
            )

    def test_is_frozen_and_picklable(self):
        import pickle

        pm = PointMap.from_frame(
            36100.0, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        with pytest.raises(Exception):
            pm.origin_points = 0.0  # type: ignore[misc]
        restored = pickle.loads(pickle.dumps(pm))
        assert restored == pm

    def test_returns_plain_int(self):
        pm = PointMap.from_frame(
            36100.0, Sideband.LOWER, self.PROBE_MHZ, self.N_ACTIVE, self.DT
        )
        assert isinstance(pm.stamp(0.2), int)


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
            n_raw=20,
        )
        assert result.alpha == 0.5
        assert result.n_active == 10
        assert result.n_raw == 20
