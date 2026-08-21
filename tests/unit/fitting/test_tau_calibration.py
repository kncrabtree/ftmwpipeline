"""Unit tests for the STFT tau-calibration module.

Validates the algorithmic kernels (sliding STFT, per-bin classification,
SNR-weighted majority, GMM bimodality, spur clustering) on controlled
synthetic FIDs. The method and its synthetic acceptance gate are documented
in ``docs/source/stage2b_tau.rst``; these tests are the fast in-tree
smoke checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ftmwpipeline.fitting import tau_calibration as tau_calibration_module
from ftmwpipeline.fitting.tau_calibration import (
    DEFAULT_N_SEG,
    DEFAULT_SIGMA_TAU_FLOOR_US,
    DEFAULT_TAU_G_BOUND_HI,
    BandMajority,
    ShapeRecommendation,
    _aggregate_shape_verdict,
    _nls_polish_step,
    band_majority_for_frequency,
    compute_band_majorities,
    compute_shape_recommendation,
    estimate_sigma_time_from_tail,
    extract_tau_G_majority,
    extract_tau_majority,
    gmm_bimodality,
    group_spur_bins,
    majority_tau,
    sliding_stft,
    stft_calibration,
)

# 2638-shaped cell: T_full = 12.65 us, sample_dt = 20 ps (50 GS/s).
SAMPLE_DT_US = 0.020
T_FULL_US = 12.65
PROBE_MHZ = 40000.0
TRIM_LO_MHZ = 26500.0
TRIM_HI_MHZ = 40000.0


def _synth_fid(
    *,
    rng: np.random.Generator,
    n_samples: int,
    line_bins: list[int],
    line_taus_us: list[float],
    line_snrs: list[float],
    spur_bins: list[int] | None = None,
    spur_snrs: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Build a noisy single/multi-line FID with optional CW spurs.

    Returns ``(fid_noisy, sigma_time)``. Calibrates sigma_x against the
    strongest planted line so each line achieves the requested on-line SNR.
    """
    spur_bins = list(spur_bins or [])
    spur_snrs = list(spur_snrs or [])
    T_full_us = n_samples * SAMPLE_DT_US
    t_us = np.arange(n_samples) * SAMPLE_DT_US
    fid = np.zeros(n_samples, dtype=float)
    f_bb_lines = np.asarray(line_bins, dtype=float) / T_full_us
    tau_arr = np.asarray(line_taus_us, dtype=float)
    snr_arr = np.asarray(line_snrs, dtype=float)
    tau_eff = tau_arr * (1.0 - np.exp(-T_full_us / tau_arr))
    amps = snr_arr / tau_eff
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(line_bins))
    for i in range(len(line_bins)):
        fid += (
            amps[i]
            * np.cos(2.0 * np.pi * f_bb_lines[i] * t_us + phases[i])
            * np.exp(-t_us / tau_arr[i])
        )

    spec_clean = SAMPLE_DT_US * np.fft.rfft(fid)
    i_strongest = int(np.argmax(snr_arr))
    on_line_mag = np.abs(spec_clean[line_bins[i_strongest]])
    sigma_x = on_line_mag / float(snr_arr[i_strongest])
    sigma_time = sigma_x / (SAMPLE_DT_US * np.sqrt(n_samples / 2.0))

    if spur_bins:
        spur_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spur_bins))
        for j, b in enumerate(spur_bins):
            A_s = 2.0 * spur_snrs[j] * sigma_x / (n_samples * SAMPLE_DT_US)
            f_bb = b / T_full_us
            fid += A_s * np.cos(2.0 * np.pi * f_bb * t_us + spur_phases[j])

    noise = rng.normal(scale=sigma_time, size=n_samples)
    return fid + noise, sigma_time


# ---------------------------------------------------------------------------
# Core kernels
# ---------------------------------------------------------------------------
class TestSlidingSTFT:
    def test_shape_and_a_centers(self):
        N = 600
        n_seg = 10
        fid = np.zeros(N)
        fid[N // 4] = 1.0
        mag, a_centers_us, freq_bb_mhz = sliding_stft(fid, SAMPLE_DT_US, n_seg)
        assert mag.shape == (n_seg, N // 2 + 1)
        assert a_centers_us.shape == (n_seg,)
        # Frame midpoints monotonically increasing, spanning the FID.
        diffs = np.diff(a_centers_us)
        assert np.all(diffs > 0.0)
        assert a_centers_us[0] >= 0.0
        # Sub-window length = N / n_seg samples, midpoint at (Nw - 1) * dt / 2.
        Nw = N // n_seg
        assert a_centers_us[0] == pytest.approx(
            (Nw - 1) * 0.5 * SAMPLE_DT_US,
            rel=0,
            abs=1e-12,
        )

    def test_rejects_oversize_n_seg(self):
        with pytest.raises(ValueError, match="n_seg"):
            sliding_stft(np.zeros(20), SAMPLE_DT_US, n_seg=10)


class TestStftCalibrationSingleLine:
    """A single isolated line at 2638-shape recovers tau within a few percent."""

    def test_single_line_recovery(self):
        rng = np.random.default_rng(20260525)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bin = N // 4
        fid, sigma_t = _synth_fid(
            rng=rng,
            n_samples=N,
            line_bins=[line_bin],
            line_taus_us=[7.5],
            line_snrs=[100.0],
        )
        cal = stft_calibration(fid, SAMPLE_DT_US, sigma_t, n_seg=DEFAULT_N_SEG)
        # On-line bin classified as contributor.
        assert cal.classification[line_bin] == 3
        # Recovered tau within +- 10 % of truth (Phase-1 case 1 cell typically
        # +3-5 %; loose bound is robust against the EM/noise jitter).
        recovered = cal.tau_per_bin[line_bin]
        assert recovered == pytest.approx(7.5, rel=0.10)


class TestStftCalibrationSpur:
    """A lone CW spur saturates tau and is classified as spur."""

    def test_isolated_spur_classification(self):
        rng = np.random.default_rng(20260525 + 1)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        # A real line anchors the sigma calibration; the spur is the test.
        spur_bin = N // 3
        line_bin = N // 5
        fid, sigma_t = _synth_fid(
            rng=rng,
            n_samples=N,
            line_bins=[line_bin],
            line_taus_us=[7.5],
            line_snrs=[50.0],
            spur_bins=[spur_bin],
            spur_snrs=[100.0],
        )
        cal = stft_calibration(fid, SAMPLE_DT_US, sigma_t, n_seg=DEFAULT_N_SEG)
        assert cal.classification[spur_bin] == 1, "spur bin should classify as spur"
        # tau saturates near the upper bound.
        assert cal.tau_per_bin[spur_bin] >= 0.95 * cal.tau_max_us


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
class TestMajorityTau:
    def test_snr_weighted_collapses_to_strong_lines(self):
        # Two strong lines at 7.5 and one weak (noisy) bin biased high.
        taus = np.array([7.4, 7.5, 7.6, 15.0])
        snrs = np.array([100.0, 100.0, 100.0, 5.0])
        tau_maj, sigma_tau = majority_tau(taus, snrs, weighted=True)
        assert tau_maj == pytest.approx(7.5, abs=0.5)
        # Unweighted median is pulled toward 7.5 too here, so use a stronger
        # bias case to differentiate.
        taus2 = np.array([7.5, 7.5, 7.5, 15.0, 15.0, 15.0])
        snrs2 = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        tau_w, _ = majority_tau(taus2, snrs2, weighted=True)
        tau_u, _ = majority_tau(taus2, snrs2, weighted=False)
        assert abs(tau_w - 7.5) < abs(tau_u - 7.5)

    def test_empty(self):
        tau, sigma = majority_tau(np.array([]), np.array([]))
        assert np.isnan(tau) and np.isnan(sigma)


class TestGMMBimodality:
    def test_bimodal_detection(self):
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(5.0, 0.4, size=60)
        cluster_b = rng.normal(10.0, 0.5, size=60)
        x = np.concatenate([cluster_a, cluster_b])
        bm = gmm_bimodality(x)
        assert bm.two_component_preferred
        assert bm.mu_a == pytest.approx(5.0, abs=0.5)
        assert bm.mu_b == pytest.approx(10.0, abs=0.5)
        assert bm.delta_aic > 10.0

    def test_unimodal_not_preferred(self):
        rng = np.random.default_rng(43)
        x = rng.normal(7.0, 1.0, size=200)
        bm = gmm_bimodality(x)
        # Either delta_aic < 2 (1-component preferred) or marginally above; the
        # important contract is that we report a sensible mu1.
        assert bm.mu1 == pytest.approx(7.0, abs=0.3)
        assert not bm.two_component_preferred or bm.delta_aic < 5.0

    def test_returns_degenerate_for_small_n(self):
        bm = gmm_bimodality(np.arange(10, dtype=float))
        assert not bm.two_component_preferred
        assert np.isnan(bm.delta_aic)


class TestGroupSpurBins:
    def test_groups_adjacent_within_n_seg(self):
        # Three CW sources: a 10-bin cluster around bin 100, another at 500,
        # one isolated bin at 800.
        spur_bins = np.concatenate([np.arange(95, 106), np.arange(498, 503), [800]])
        mean_mag = np.zeros(1000)
        mean_mag[100] = 5.0
        mean_mag[500] = 3.0
        mean_mag[800] = 1.0
        freqs = np.linspace(0.0, 1000.0, 1000)
        clusters = group_spur_bins(spur_bins, mean_mag, freqs, n_seg=10)
        assert len(clusters) == 3
        assert clusters[0].peak_bin_index == 100
        assert clusters[1].peak_bin_index == 500
        assert clusters[2].peak_bin_index == 800
        assert clusters[0].n_bins == 11
        assert clusters[1].n_bins == 5
        assert clusters[2].n_bins == 1

    def test_empty(self):
        clusters = group_spur_bins(
            np.array([], dtype=np.int64),
            np.zeros(10),
            np.arange(10.0),
            n_seg=10,
        )
        assert clusters == ()


# ---------------------------------------------------------------------------
# End-to-end extractor + tail sigma_t fallback
# ---------------------------------------------------------------------------
class TestExtractTauMajority:
    def test_recovers_tau_for_multi_line_lower_sideband(self):
        rng = np.random.default_rng(20260525 + 99)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        # Plant 8 lines at tau = 6.0 us spread across the active band.
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        line_taus = [6.0] * len(line_bins)
        line_snrs = [200.0] * len(line_bins)
        fid, sigma_t = _synth_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_us=line_taus,
            line_snrs=line_snrs,
        )
        # Embed in the trim band: probe = 40000 MHz, lower sideband,
        # f_bb maps to molecular f = probe - f_bb. line_bins live at
        # f_bb in [N/8 / T_full, 5N/8 / T_full] ~ [1.0, 6.2] GHz so molecular
        # frequencies sit at probe - f_bb in [33.8, 39] GHz — comfortably inside
        # 26500-40000.
        result = extract_tau_majority(
            fid,
            SAMPLE_DT_US,
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            min_contributors=20,  # synthetic, just a few lines
        )
        assert result.tau_maj_us == pytest.approx(6.0, rel=0.10)
        assert result.n_contributors >= 8
        assert result.sideband == "lower"

    def test_preconditions_fail_on_too_few_contributors(self):
        """A real ``preconditions_passed=False`` outcome (not just round-trip).

        Plant the same clean multi-line FID but demand far more contributors
        than the band can supply, so the count pre-condition fails and the
        computed flag flips to ``False`` with the matching note.
        """
        rng = np.random.default_rng(20260525 + 99)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        fid, sigma_t = _synth_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_us=[6.0] * len(line_bins),
            line_snrs=[200.0] * len(line_bins),
        )
        result = extract_tau_majority(
            fid,
            SAMPLE_DT_US,
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            min_contributors=100_000,
        )
        assert result.preconditions_passed is False
        # The count note carries the shape-specific noun and the threshold.
        assert any(
            "contributors (< 100000)" in note for note in result.preconditions_notes
        ), result.preconditions_notes
        # τ is still computed and returned for downstream consumers to use.
        assert result.tau_maj_us == pytest.approx(6.0, rel=0.10)

    def test_tail_sigma_fallback(self):
        # Quick: estimator returns a positive number on a real noisy FID.
        rng = np.random.default_rng(20260525 + 100)
        noise = rng.normal(0.0, 0.5, size=10000)
        # Add a strong decaying line to verify the tail is dominated by noise.
        t = np.arange(noise.size) * SAMPLE_DT_US
        line = 100.0 * np.cos(2.0 * np.pi * 200.0 * t) * np.exp(-t / 1.0)
        sigma_t = estimate_sigma_time_from_tail(noise + line)
        assert 0.3 < sigma_t < 0.8

    def test_rejects_empty_active_region(self):
        with pytest.raises(ValueError, match="too few samples"):
            extract_tau_majority(
                np.zeros(50),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=0.1,
                probe_freq_mhz=PROBE_MHZ,
                sideband="lower",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
            )

    def test_polish_closes_log_linear_bias(self):
        """Polish drops the +3-5 % log-linear-weighting bias to sub-1 % on a single line.

        Phase 1 § Case 1 (2638-shaped cell, T_full = 12.65 us, tau = 7.5 us,
        SNR = 100) shows the polish-OFF SNR-weighted majority lands at
        +3 % over truth; the polish-ON majority lands within +- 1 % over
        the same trials. This test verifies the bias direction and the
        relative magnitude of the improvement, not the absolute number
        (jitter across N_seg-frame realizations is ~ +- 1 %).
        """
        rng_seed = 20260525 + 200
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bin = N // 4
        n_trials = 12

        off_errs = []
        on_errs = []
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_seed + trial)
            fid, sigma_t = _synth_fid(
                rng=rng,
                n_samples=N,
                line_bins=[line_bin],
                line_taus_us=[7.5],
                line_snrs=[100.0],
            )
            common_kwargs = dict(
                start_us=0.0,
                end_us=N * SAMPLE_DT_US,
                probe_freq_mhz=PROBE_MHZ,
                sideband="lower",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=sigma_t,
                min_contributors=5,
            )
            off = extract_tau_majority(
                fid,
                SAMPLE_DT_US,
                polish=False,
                **common_kwargs,
            )
            on = extract_tau_majority(
                fid,
                SAMPLE_DT_US,
                polish=True,
                **common_kwargs,
            )
            off_errs.append(100.0 * (off.tau_maj_us - 7.5) / 7.5)
            on_errs.append(100.0 * (on.tau_maj_us - 7.5) / 7.5)

        med_off = float(np.median(off_errs))
        med_on = float(np.median(on_errs))
        # The biased path lands in the documented +1.5 to +5 % band; polish
        # brings it materially closer to zero.
        assert (
            0.5 < med_off < 8.0
        ), f"polish=OFF median error {med_off:.2f}% outside expected +1-+8% band"
        assert (
            abs(med_on) < 1.5
        ), f"polish=ON median error {med_on:.2f}% should be sub-1.5%"
        # And the polished path strictly improves the bias magnitude.
        assert abs(med_on) < abs(med_off)

    def test_polish_step_well_conditioned(self):
        """One Gauss-Newton step lands within ~1 % from a 10 %-biased seed.

        Gauss-Newton has quadratic convergence near the optimum; one step
        on a noise-free single-line magnitude drops a +10 % seed bias to
        well under +1 %. The real log-linear seed bias is ~3-5 %, where
        one step would land at sub-0.5 % -- well inside the +-1 % target.
        """
        a = np.linspace(0.5, 12.0, 10)
        tau_true = 7.5
        C_true = 4.0
        mag = (C_true * np.exp(-a / tau_true)).reshape(-1, 1)
        tau_seed = np.array([tau_true * 1.10])
        C_seed = np.array([C_true * 1.05])
        tau_polished, C_polished = _nls_polish_step(mag, a, tau_seed, C_seed)
        assert tau_polished[0] == pytest.approx(tau_true, rel=2e-2)
        assert C_polished[0] == pytest.approx(C_true, rel=2e-2)
        # And strictly improves over the seed.
        assert abs(tau_polished[0] - tau_true) < abs(tau_seed[0] - tau_true)

    def test_polish_step_mask_respected(self):
        """Bins outside the mask are left untouched."""
        a = np.linspace(0.5, 12.0, 10)
        mag = np.zeros((10, 3))
        mag[:, 0] = 4.0 * np.exp(-a / 7.5)
        mag[:, 1] = 2.0 * np.exp(-a / 5.0)
        mag[:, 2] = 1.0  # constant -- pretend this is a spur
        tau_seed = np.array([7.5 * 1.10, 5.0 * 1.10, 100.0])
        C_seed = np.array([4.0 * 1.05, 2.0 * 1.05, 1.0])
        mask = np.array([True, True, False])
        tau_out, C_out = _nls_polish_step(mag, a, tau_seed, C_seed, mask=mask)
        # Masked bin (index 2) untouched.
        assert tau_out[2] == 100.0
        assert C_out[2] == 1.0
        # Polished bins land within 2 % of the truths (one Gauss-Newton step
        # from a 10 % seed bias; quadratic convergence).
        assert tau_out[0] == pytest.approx(7.5, rel=2e-2)
        assert tau_out[1] == pytest.approx(5.0, rel=2e-2)

    def test_rejects_bad_sideband(self):
        with pytest.raises(ValueError, match="sideband"):
            extract_tau_majority(
                np.zeros(1000),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband="middle",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
            )

    def test_sigma_tau_floor_us_reaches_compute_band_majorities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``sigma_tau_floor_us`` must reach ``compute_band_majorities``.

        Regression guard for a previously silent no-op: ``extract_tau_
        majority`` had no ``sigma_tau_floor_us`` parameter at all, so
        ``_finalize_tau_result`` could never forward a caller's choice --
        ``compute_band_majorities`` always floored at a hardcoded 0.5.
        Spy on ``compute_band_majorities`` and check the exact kwarg it
        receives, both when the caller passes an explicit floor and when
        it relies on the default (which must equal
        ``DEFAULT_SIGMA_TAU_FLOOR_US``, not just happen to match it).
        """
        captured: dict[str, list[Any]] = {"sigma_floor_us": []}
        real = tau_calibration_module.compute_band_majorities

        def _spy(*args: Any, **kwargs: Any) -> Any:
            captured["sigma_floor_us"].append(kwargs.get("sigma_floor_us"))
            return real(*args, **kwargs)

        monkeypatch.setattr(tau_calibration_module, "compute_band_majorities", _spy)

        rng = np.random.default_rng(20260525 + 400)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        fid, sigma_t = _synth_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_us=[6.0] * len(line_bins),
            line_snrs=[200.0] * len(line_bins),
        )
        common: dict[str, Any] = dict(
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            min_contributors=20,
            compute_band_majorities_flag=True,
        )

        # Default: not passed by the caller -> forwards the named constant.
        extract_tau_majority(fid, SAMPLE_DT_US, **common)
        # Explicit: a distinct caller-chosen floor must reach the kernel.
        extract_tau_majority(fid, SAMPLE_DT_US, sigma_tau_floor_us=2.75, **common)

        assert captured["sigma_floor_us"] == [DEFAULT_SIGMA_TAU_FLOOR_US, 2.75]


# ---------------------------------------------------------------------------
# Gaussian-twin extractor: extract_tau_G_majority
# ---------------------------------------------------------------------------
def _synth_gaussian_fid(
    *,
    rng: np.random.Generator,
    n_samples: int,
    line_bins: list[int],
    line_taus_G_us: list[float],
    line_snrs: list[float],
) -> tuple[np.ndarray, float]:
    """FID with Gaussian-envelope lines: ``s_i(t) = A_i cos(2π f_i t) exp(-(t/τ_G_i)²)``.

    SNR is calibrated against the strongest planted line's on-line magnitude
    in the full-record rfft (same convention as :func:`_synth_fid`); the
    Gaussian effective area is ``∫_0^T exp(-(t/τ_G)²) dt ≈ τ_G √π / 2``
    for ``τ_G ≪ T`` so we use that as the effective tau.
    """
    T_full_us = n_samples * SAMPLE_DT_US
    t_us = np.arange(n_samples) * SAMPLE_DT_US
    fid = np.zeros(n_samples, dtype=float)
    f_bb_lines = np.asarray(line_bins, dtype=float) / T_full_us
    tau_arr = np.asarray(line_taus_G_us, dtype=float)
    snr_arr = np.asarray(line_snrs, dtype=float)
    # ∫_0^T exp(-(t/τ_G)²) dt = (τ_G √π / 2) erf(T/τ_G). At T/τ_G ≳ 2 the
    # erf saturates at 1, so τ_eff ≈ τ_G √π / 2.
    from scipy.special import erf

    tau_eff = 0.5 * tau_arr * np.sqrt(np.pi) * erf(T_full_us / tau_arr)
    amps = snr_arr / tau_eff
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(line_bins))
    for i in range(len(line_bins)):
        fid += (
            amps[i]
            * np.cos(2.0 * np.pi * f_bb_lines[i] * t_us + phases[i])
            * np.exp(-((t_us / tau_arr[i]) ** 2))
        )
    spec_clean = SAMPLE_DT_US * np.fft.rfft(fid)
    i_strongest = int(np.argmax(snr_arr))
    on_line_mag = np.abs(spec_clean[line_bins[i_strongest]])
    sigma_x = on_line_mag / float(snr_arr[i_strongest])
    sigma_time = sigma_x / (SAMPLE_DT_US * np.sqrt(n_samples / 2.0))
    noise = rng.normal(scale=sigma_time, size=n_samples)
    return fid + noise, sigma_time


class TestExtractTauGMajority:
    def test_recovers_tau_G_for_multi_line(self):
        """SNR-weighted majority τ_G lands within ±10 % on Gaussian lines.

        The shape-aware STFT classifier (``shape='gaussian'``) gates the
        bad-fit pool on the pure-Gauss residual, so the strong on-line
        bins planted by ``_synth_gaussian_fid`` enter the cls=3
        contributor set directly. The SNR-weighted majority τ_G then
        sits within ±10 % of the planted truth -- a real test of the
        recovery, not just a "did the machinery run end-to-end" smoke
        check. Stage 5 χ² improvement against the free-τ window-fit
        distribution on the 2638 fixture is the production acceptance
        test (validated separately).
        """
        rng = np.random.default_rng(20260525 + 311)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        tau_G_truth = 6.0
        line_taus = [tau_G_truth] * len(line_bins)
        line_snrs = [200.0] * len(line_bins)
        fid, sigma_t = _synth_gaussian_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_G_us=line_taus,
            line_snrs=line_snrs,
        )
        result = extract_tau_G_majority(
            fid,
            SAMPLE_DT_US,
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            snr_min=10.0,
            min_contributors=2,
            min_contributors_per_band=2,
        )
        assert (
            result.n_contributors >= 2
        ), f"only {result.n_contributors} eligible bins (need ≥ 2)"
        assert result.tau_maj_us == pytest.approx(
            tau_G_truth, rel=0.10
        ), f"τ_G recovered as {result.tau_maj_us:.2f} us, expected ~{tau_G_truth} ± 10%"
        assert result.sideband == "lower"
        # The eligible subset must not be saturated against the upper bound.
        assert np.all(result.contributor_taus_us < 0.7 * DEFAULT_TAU_G_BOUND_HI)
        # The result struct's tau_max_us mirrors the τ_G upper bound (not
        # the underlying STFT classifier's tau_max), because under the
        # twin's semantics the persisted ``tau_max_us`` is the Gaussian
        # parameter ceiling that gated eligibility.
        assert result.tau_max_us == DEFAULT_TAU_G_BOUND_HI

    def test_rejects_bad_sideband(self):
        with pytest.raises(ValueError, match="sideband"):
            extract_tau_G_majority(
                np.zeros(1000),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband="middle",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
            )

    def test_rejects_inverted_tau_g_bounds(self):
        with pytest.raises(ValueError, match="tau_G_bound_hi"):
            extract_tau_G_majority(
                np.zeros(1000),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband="lower",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
                tau_G_bound_lo=10.0,
                tau_G_bound_hi=5.0,
            )

    def test_sigma_tau_floor_us_reaches_compute_band_majorities_gaussian(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Twin of the same guard on :func:`extract_tau_majority`.

        ``extract_tau_G_majority`` shares ``_finalize_tau_result`` with the
        pure-exp extractor; this confirms the Gaussian entry point also
        forwards its ``sigma_tau_floor_us`` all the way to
        ``compute_band_majorities`` (default ``compute_band_majorities_
        flag=True`` here, so no need to set it explicitly).
        """
        captured: dict[str, list[Any]] = {"sigma_floor_us": []}
        real = tau_calibration_module.compute_band_majorities

        def _spy(*args: Any, **kwargs: Any) -> Any:
            captured["sigma_floor_us"].append(kwargs.get("sigma_floor_us"))
            return real(*args, **kwargs)

        monkeypatch.setattr(tau_calibration_module, "compute_band_majorities", _spy)

        rng = np.random.default_rng(20260525 + 411)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        fid, sigma_t = _synth_gaussian_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_G_us=[6.0] * len(line_bins),
            line_snrs=[200.0] * len(line_bins),
        )
        common: dict[str, Any] = dict(
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            snr_min=10.0,
            min_contributors=2,
            min_contributors_per_band=2,
        )

        extract_tau_G_majority(fid, SAMPLE_DT_US, **common)
        extract_tau_G_majority(fid, SAMPLE_DT_US, sigma_tau_floor_us=1.4, **common)

        assert captured["sigma_floor_us"] == [DEFAULT_SIGMA_TAU_FLOOR_US, 1.4]


# ---------------------------------------------------------------------------
# 3-way L / G / V shape-recommendation hook
# ---------------------------------------------------------------------------
def _row(verdict: str, snr: float = 100.0, **kw) -> dict:
    """Build a minimal per-bin AICc row for the aggregator tests."""
    base = dict(
        idx=0,
        freq=30000.0,
        snr=snr,
        aicc_exp=0.0,
        aicc_gauss=0.0,
        aicc_voigt=0.0,
        d_aicc_gauss_exp=0.0,
        d_aicc_voigt_exp=0.0,
        d_aicc_voigt_gauss=0.0,
        verdict=verdict,
    )
    base.update(kw)
    return base


class TestAggregateShapeVerdict:
    def test_empty_rows_returns_none_with_diagnostic(self):
        rec = _aggregate_shape_verdict([])
        assert isinstance(rec, ShapeRecommendation)
        assert rec.recommended_shape is None
        assert rec.n_contributors == 0
        assert all(r == 0.0 for r in rec.vote_rates.values())
        assert any("no contributor" in n for n in rec.notes)

    def test_pure_exp_majority_recommends_lorentzian(self):
        rows = (
            [_row("exp", snr=100.0)] * 70
            + [_row("gauss", snr=100.0)] * 10
            + [_row("voigt", snr=100.0)] * 20
        )
        rec = _aggregate_shape_verdict(rows)
        assert rec.recommended_shape == "lorentzian"
        assert rec.vote_rates["exp"] == pytest.approx(0.70)
        assert rec.vote_rates["gauss"] == pytest.approx(0.10)
        assert rec.vote_rates["voigt"] == pytest.approx(0.20)
        assert rec.n_contributors == 100

    def test_pure_gauss_majority_recommends_gaussian(self):
        rows = (
            [_row("gauss", snr=100.0)] * 70
            + [_row("exp", snr=100.0)] * 10
            + [_row("voigt", snr=100.0)] * 20
        )
        rec = _aggregate_shape_verdict(rows)
        assert rec.recommended_shape == "gaussian"
        assert rec.vote_rates["gauss"] == pytest.approx(0.70)
        assert rec.vote_rates["exp"] == pytest.approx(0.10)

    def test_voigt_majority_with_gauss_pure_lead_picks_gaussian(self):
        # Voigt has the most votes but gauss beats exp on the pure-shape
        # margin -- the aggregator picks the pure shape, not voigt.
        rows = (
            [_row("voigt", snr=100.0)] * 50
            + [_row("gauss", snr=100.0)] * 35
            + [_row("exp", snr=100.0)] * 15
        )
        rec = _aggregate_shape_verdict(rows)
        assert rec.recommended_shape == "gaussian"

    def test_tied_pure_shapes_no_recommendation(self):
        # exp ≈ gauss within the margin threshold; no winner.
        rows = (
            [_row("exp", snr=100.0)] * 32
            + [_row("gauss", snr=100.0)] * 30
            + [_row("voigt", snr=100.0)] * 38
        )
        rec = _aggregate_shape_verdict(rows, pure_margin_threshold=0.10)
        assert rec.recommended_shape is None
        assert any("no clear winner" in n for n in rec.notes)

    def test_snr_weighting_collapses_low_snr_noise(self):
        # Many low-SNR ambiguous bins shouldn't override a strong on-line
        # cluster: 10 strong gauss votes outweigh 90 weak voigt votes.
        rows = [_row("gauss", snr=200.0)] * 10 + [_row("voigt", snr=5.0)] * 90
        rec = _aggregate_shape_verdict(rows)
        # Gauss carries 10 * 200 = 2000 SNR vs voigt's 90 * 5 = 450;
        # gauss rate = 2000 / 2450 ≈ 0.816, voigt rate ≈ 0.184, exp = 0.
        assert rec.vote_rates["gauss"] == pytest.approx(2000 / 2450, abs=1e-6)
        assert rec.recommended_shape == "gaussian"


class TestComputeShapeRecommendation:
    def test_returns_shape_recommendation_struct(self):
        """End-to-end on a multi-line synthetic FID returns a well-formed verdict.

        The synthetic plants 8 Gaussian-envelope lines and the
        shape-aware classifier (``shape='best_of_three'``) lets the
        strong on-line bins into the cls=3 contributor pool, so the
        per-bin AICc vote reflects the planted shape. The test asserts
        well-formedness (vote rates sum to 1, recommended_shape in the
        allowed set); the integration suite carries the real-data
        verdict on 2638.
        """
        rng = np.random.default_rng(20260526 + 1)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        fid, sigma_t = _synth_gaussian_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_G_us=[6.0] * len(line_bins),
            line_snrs=[200.0] * len(line_bins),
        )
        rec = compute_shape_recommendation(
            fid,
            SAMPLE_DT_US,
            start_us=0.0,
            end_us=N * SAMPLE_DT_US,
            probe_freq_mhz=PROBE_MHZ,
            sideband="lower",
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            sigma_time=sigma_t,
            snr_min=10.0,
        )
        assert isinstance(rec, ShapeRecommendation)
        assert rec.n_contributors >= 2
        assert set(rec.vote_rates.keys()) == {"exp", "gauss", "voigt"}
        assert sum(rec.vote_rates.values()) == pytest.approx(1.0, abs=1e-6)
        assert set(rec.median_d_aicc.keys()) == {
            "gauss_vs_exp",
            "voigt_vs_exp",
            "voigt_vs_gauss",
        }
        assert rec.recommended_shape in (None, "lorentzian", "gaussian")

    def test_rejects_bad_sideband(self):
        with pytest.raises(ValueError, match="sideband"):
            compute_shape_recommendation(
                np.zeros(1000),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband="middle",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
            )

    def test_rejects_inverted_tau_bounds(self):
        with pytest.raises(ValueError, match="tau_bound_hi"):
            compute_shape_recommendation(
                np.zeros(1000),
                SAMPLE_DT_US,
                start_us=0.0,
                end_us=1.0,
                probe_freq_mhz=PROBE_MHZ,
                sideband="lower",
                trim_lo_mhz=TRIM_LO_MHZ,
                trim_hi_mhz=TRIM_HI_MHZ,
                sigma_time=1.0,
                tau_bound_lo=10.0,
                tau_bound_hi=5.0,
            )


class TestVectorizedShapeSolver:
    """The default vectorized shape solver agrees with the scipy oracle.

    The per-bin 3-way fit (``_run_shape_fits``) defaults to a batched
    closed-form-log-seed + clipped-Gauss-Newton solver; ``solver='scipy'``
    keeps the per-bin ``least_squares`` multistart loop as the reference. The
    two must agree on the contributor pool and the per-bin Gaussian τ (the
    quantity Stage 5 consumes for Gaussian-shape windows) on a clean
    multi-line synthetic; the production 7-fixture cross-check lives in
    ``dev-docs`` / ``scratch``.
    """

    @staticmethod
    def _cal(solver: str):
        rng = np.random.default_rng(20260531 + 7)
        N = int(round(T_FULL_US / SAMPLE_DT_US))
        N = (N // DEFAULT_N_SEG) * DEFAULT_N_SEG
        line_bins = list(range(N // 8, 5 * N // 8, N // 16))[:8]
        fid, sigma_t = _synth_gaussian_fid(
            rng=rng,
            n_samples=N,
            line_bins=line_bins,
            line_taus_G_us=[6.0] * len(line_bins),
            line_snrs=[200.0] * len(line_bins),
        )
        return stft_calibration(
            fid,
            SAMPLE_DT_US,
            sigma_t,
            n_seg=DEFAULT_N_SEG,
            shape="best_of_three",
            shape_solver=solver,
        )

    @pytest.fixture(scope="class")
    def vec(self):
        return self._cal("vectorized")

    @pytest.fixture(scope="class")
    def sci(self):
        return self._cal("scipy")

    def test_pool_and_gaussian_tau_match_scipy(self, vec, sci):
        # Identical above-threshold non-spur classification (the cls=3 pool is
        # set before the shape fits, but the bad-fit gate uses their RSS, so
        # this also exercises that the vectorized RSS gates the same bins).
        assert int((vec.classification == 3).sum()) == int(
            (sci.classification == 3).sum()
        )
        # Per-bin Gaussian τ agrees where both converge to an in-bound fit.
        fv, fs = vec.shape_fits, sci.shape_fits
        assert fv is not None and fs is not None
        cap = 0.7 * DEFAULT_TAU_G_BOUND_HI
        both = (
            fv.converged_gauss
            & fs.converged_gauss
            & (fv.tau_G_gauss < cap)
            & (fs.tau_G_gauss < cap)
            & np.isfinite(fv.tau_G_gauss)
            & np.isfinite(fs.tau_G_gauss)
        )
        assert both.sum() >= 5, f"too few comparable gauss bins ({int(both.sum())})"
        rel = np.abs(fv.tau_G_gauss[both] - fs.tau_G_gauss[both]) / fs.tau_G_gauss[both]
        assert (
            np.median(rel) < 0.02
        ), f"vectorized gauss τ median rel error {np.median(rel)*100:.1f}% vs scipy"

    def test_recommendation_matches_scipy(self, vec, sci):
        from ftmwpipeline.fitting.tau_calibration import (
            _shape_recommendation_bin_clean,
            _three_way_rows_from_shape_fits,
        )

        cap = 0.7 * DEFAULT_TAU_G_BOUND_HI

        def _rec(cal):
            fmol = PROBE_MHZ - cal.freq_bb_mhz  # lower sideband
            pool = (cal.classification == 3) & (cal.snr_per_bin > 20.0)
            idx = np.where(pool)[0]
            rows = _three_way_rows_from_shape_fits(cal, idx, fmol, n_seg=DEFAULT_N_SEG)
            rows = [r for r in rows if _shape_recommendation_bin_clean(r, cap)]
            return _aggregate_shape_verdict(rows).recommended_shape

        assert _rec(vec) == _rec(sci)

    def test_rejects_bad_solver(self):
        with pytest.raises(ValueError, match="solver"):
            stft_calibration(
                np.zeros(1000),
                SAMPLE_DT_US,
                1.0,
                n_seg=DEFAULT_N_SEG,
                shape="best_of_three",
                shape_solver="banana",
            )


# ---------------------------------------------------------------------------
# Per-band majorities (compute_band_majorities / band_majority_for_frequency)
# ---------------------------------------------------------------------------
class TestBandMajorities:
    def test_per_band_majority_separates_bands(self):
        """Each well-populated band reports its own local majority tau."""
        # Two bands with distinct true tau; enough contributors in each.
        lo_freqs = np.linspace(27000.0, 30000.0, 60)
        hi_freqs = np.linspace(36000.0, 39000.0, 60)
        freqs = np.concatenate([lo_freqs, hi_freqs])
        taus = np.concatenate([np.full(60, 8.0), np.full(60, 5.0)])
        snrs = np.full(120, 50.0)
        bands = compute_band_majorities(
            freqs,
            taus,
            snrs,
            trim_lo_mhz=26500.0,
            trim_hi_mhz=40000.0,
            band_edges_mhz=(33000.0,),
            band_labels=("low", "high"),
            min_contributors_per_band=10,
        )
        assert [b.label for b in bands] == ["low", "high"]
        low, high = bands
        assert low.tau_maj_us == pytest.approx(8.0, abs=1e-6)
        assert high.tau_maj_us == pytest.approx(5.0, abs=1e-6)
        assert low.n == 60 and high.n == 60

    def test_short_band_falls_back_to_band_wide(self):
        """A band below ``min_contributors_per_band`` uses the band-wide majority."""
        # 50 contributors in the low band, only 3 in the high band.
        freqs = np.concatenate(
            [np.linspace(27000.0, 30000.0, 50), [37000.0, 38000.0, 39000.0]]
        )
        taus = np.concatenate([np.full(50, 8.0), np.full(3, 5.0)])
        snrs = np.full(53, 50.0)
        band_wide_tau, _ = majority_tau(taus, snrs, weighted=True)
        bands = compute_band_majorities(
            freqs,
            taus,
            snrs,
            trim_lo_mhz=26500.0,
            trim_hi_mhz=40000.0,
            band_edges_mhz=(33000.0,),
            band_labels=("low", "high"),
            min_contributors_per_band=10,
            sigma_floor_us=0.5,
        )
        low, high = bands
        assert low.tau_maj_us == pytest.approx(8.0, abs=1e-6)
        # Short high band collapses to the band-wide majority, not its own 5.0.
        assert high.n == 3
        assert high.tau_maj_us == pytest.approx(band_wide_tau, abs=1e-6)
        assert high.sigma_tau_us >= 0.5  # floored

    def test_band_majority_for_frequency_routing(self):
        bands = (
            BandMajority("low", 26500.0, 33000.0, 10, 8.0, 0.5),
            BandMajority("high", 33000.0, 40000.0, 10, 5.0, 0.5),
        )
        assert band_majority_for_frequency(bands, 28000.0).label == "low"
        assert band_majority_for_frequency(bands, 37000.0).label == "high"
        # Closed-closed at the very top edge so a contributor at trim_hi routes.
        assert band_majority_for_frequency(bands, 40000.0).label == "high"
        # Outside every band -> None (caller falls back to band-wide).
        assert band_majority_for_frequency(bands, 10000.0) is None
        assert band_majority_for_frequency((), 28000.0) is None


# ---------------------------------------------------------------------------
# aggregation.sigma_tau_floor_us knob and plumbing tests
# ---------------------------------------------------------------------------
# Before this knob was wired up, ``compute_band_majorities`` floored every
# band-local ``sigma_tau_us`` at a hardcoded literal ``0.5`` -- callers of
# :func:`extract_tau_majority` / :func:`extract_tau_G_majority` had no way to
# change it despite ``aggregation.sigma_tau_floor_us`` being fully declared,
# persisted, and even exported as ``DEFAULT_SIGMA_TAU_FLOOR_US``. These tests
# prove the knob now actually reaches the floor and changes it.
class TestComputeBandMajoritiesSigmaFloor:
    def test_default_sigma_floor_matches_constant(self) -> None:
        """The unset ``sigma_floor_us`` default is the exported constant.

        Guards against the default silently drifting back to a bare
        literal: it must track ``DEFAULT_SIGMA_TAU_FLOOR_US`` (currently
        0.5) so a default call is behavior-identical to before this knob
        was wired up.
        """
        assert DEFAULT_SIGMA_TAU_FLOOR_US == 0.5

        freqs = np.linspace(27000.0, 30000.0, 20)
        taus = np.full(20, 8.0)  # identical taus -> near-zero raw sigma
        snrs = np.full(20, 50.0)
        (band,) = compute_band_majorities(
            freqs,
            taus,
            snrs,
            trim_lo_mhz=26500.0,
            trim_hi_mhz=30000.0,
            band_labels=("only",),
            min_contributors_per_band=5,
        )
        assert band.sigma_tau_us == pytest.approx(DEFAULT_SIGMA_TAU_FLOOR_US, abs=1e-9)

    def test_sigma_floor_us_is_caller_configurable(self) -> None:
        """A non-default ``sigma_floor_us`` actually changes the applied floor.

        Before this knob was wired up, ``compute_band_majorities`` floored
        at a bare literal 0.5 no matter what a caller passed. Same
        near-zero-scatter setup as the default test above, but with an
        explicit floor well above the default -- the reported sigma must
        track the caller's value, not 0.5.
        """
        freqs = np.linspace(27000.0, 30000.0, 20)
        taus = np.full(20, 8.0)
        snrs = np.full(20, 50.0)
        (band_default,) = compute_band_majorities(
            freqs,
            taus,
            snrs,
            trim_lo_mhz=26500.0,
            trim_hi_mhz=30000.0,
            band_labels=("only",),
            min_contributors_per_band=5,
        )
        (band_custom,) = compute_band_majorities(
            freqs,
            taus,
            snrs,
            trim_lo_mhz=26500.0,
            trim_hi_mhz=30000.0,
            band_labels=("only",),
            min_contributors_per_band=5,
            sigma_floor_us=3.0,
        )
        assert band_default.sigma_tau_us == pytest.approx(0.5, abs=1e-9)
        assert band_custom.sigma_tau_us == pytest.approx(3.0, abs=1e-9)
        assert band_custom.sigma_tau_us > band_default.sigma_tau_us
