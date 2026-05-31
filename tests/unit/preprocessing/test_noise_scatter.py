"""Acceptance tests for the scatter (high-pass) Stage 2 noise estimator.

The decisive invariants from ``dev-docs/research/noise-snr-scaling/report.md``
(§9 acceptance plan):

* **1/√N slope ≈ −0.5.** True thermal noise averages down as 1/√N while the
  leakage pedestal is constant in shot count. A level-based estimator plateaus
  on a high-SNR spectrum; the scatter estimator must keep falling as 1/√N. This
  is the regression guard the old estimator fails (§5.2).
* **Pedestal independence.** Scaling the line amplitude (hence the pedestal) by
  10–100× must not move the estimated σ — the estimator measures the noise, not
  the pedestal.

Both are sample-independent and need no fixtures; they are built from synthetic
white-noise + strong-line spectra.
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.noise_estimation import (
    NoiseResult,
    estimate_noise_scatter,
    estimate_noise_adaptive,
    SCATTER_ALGORITHM,
    _gaussian_smooth_1d,
)

# Synthetic-spectrum geometry. A 30k-bin grid over the 2638 active band so the
# physical-MHz knobs (window/pedestal) translate to a realistic number of bins.
_N_BINS = 30000
_F_LO, _F_HI = 26500.0, 40000.0
_LINE_POSITIONS = (_N_BINS // 5, _N_BINS // 2, 4 * _N_BINS // 5)
_LINE_GAMMA = 3.0  # HWHM in bins


def _frequencies() -> np.ndarray:
    return np.linspace(_F_LO, _F_HI, _N_BINS)


def _magnitude_spectrum(sigma_c: float, line_amp: float, seed: int) -> np.ndarray:
    """|X| of complex white noise (per-quadrature ``sigma_c``) + strong lines.

    The lines are narrow Lorentzians; in the raw (boxcar) FT their far-wings sum
    into the smooth leakage pedestal whose amplitude scales with ``line_amp`` and
    is independent of the noise level.
    """
    rng = np.random.default_rng(seed)
    spec = (rng.standard_normal(_N_BINS) + 1j * rng.standard_normal(_N_BINS)) * sigma_c
    idx = np.arange(_N_BINS)
    for pos in _LINE_POSITIONS:
        spec += line_amp * _LINE_GAMMA / ((idx - pos) + 1j * _LINE_GAMMA)
    return np.abs(spec)


@pytest.mark.parametrize("sigma_bins", [3.0, 50.0, 8389.0])
def test_gaussian_smooth_matches_scipy(sigma_bins):
    """The FFT-based broad σ-smoother reproduces
    ``scipy.ndimage.gaussian_filter1d(order=0, mode="nearest")`` to round-off.

    The estimator's step-removing pass uses a Gaussian whose width is a fixed
    frequency span, which on a fine detection grid is thousands of bins -- the
    regime where scipy's direct spatial correlation is O(N·kernel) and dominates
    the whole estimator. ``_gaussian_smooth_1d`` swaps that for an O(N log N) FFT
    convolution; this test pins it to the reference output it replaces.
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(7)
    x = np.abs(rng.standard_normal(_N_BINS)) + 1e-3
    ref = gaussian_filter1d(x, sigma=sigma_bins, mode="nearest")
    got = _gaussian_smooth_1d(x, sigma_bins)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_gaussian_smooth_zero_sigma_is_identity():
    x = np.linspace(1.0, 2.0, 100)
    np.testing.assert_array_equal(_gaussian_smooth_1d(x, 0.0), x)


def test_returns_noise_result_with_scatter_metadata():
    f = _frequencies()
    mag = _magnitude_spectrum(sigma_c=0.5, line_amp=50.0, seed=1)
    result = estimate_noise_scatter(f, mag)

    assert isinstance(result, NoiseResult)
    assert result.rms_noise.shape == f.shape
    assert result.noise_mask.shape == f.shape
    assert result.noise_mask.dtype == bool
    assert np.all(np.isfinite(result.rms_noise))
    assert np.all(result.rms_noise > 0)
    assert result.bin_info["algorithm"] == SCATTER_ALGORITHM
    # All instrument-tunable knobs are recorded for provenance.
    for knob in ("window_mhz", "pedestal_mhz", "line_k", "n_iter",
                 "smoothing_mhz", "smoothing_percentile", "convolve_mhz"):
        assert knob in result.bin_info


def test_outputs_complex_rms_convention():
    """σ output is the complex-RMS σ_x = σ_c·√2 (the canonical Stage 2
    convention, drop-in for the adaptive estimator), not the per-quadrature σ_c.

    The Rayleigh-end Monte-Carlo table edge makes the absolute value read a bit
    low (~0.85× σ_x on pure noise); the band here only has to exclude the σ_c
    convention (which would read ~0.7× lower still)."""
    f = _frequencies()
    sigma_c = 0.5
    sigma_x = sigma_c * np.sqrt(2.0)
    mag = _magnitude_spectrum(sigma_c=sigma_c, line_amp=40.0, seed=3)
    est = float(np.median(estimate_noise_scatter(f, mag).rms_noise))
    ratio = est / sigma_x
    assert 0.75 < ratio < 1.20, f"σ_x ratio {ratio:.3f} outside the expected band"


def test_pedestal_independence():
    """Scaling the line amplitude (the pedestal) by 100× must not move σ̂."""
    f = _frequencies()
    estimates = []
    for line_amp in (5.0, 50.0, 500.0):
        mag = _magnitude_spectrum(sigma_c=0.5, line_amp=line_amp, seed=7)
        estimates.append(float(np.median(estimate_noise_scatter(f, mag).rms_noise)))
    spread = max(estimates) / min(estimates)
    assert spread < 1.25, (
        f"σ̂ moved by {spread:.3f}× across a 100× pedestal sweep "
        f"(estimates={estimates}); the estimator is tracking the pedestal"
    )


def test_sqrt_n_slope_is_minus_half():
    """The estimator falls as 1/√N (slope ≈ −0.5 in log–log).

    Noise averages down as 1/√N; the pedestal is held constant in N (fixed line
    amplitude). The scatter estimator must follow the noise, not plateau."""
    f = _frequencies()
    shot_counts = np.array([1.0e3, 4.0e3, 1.6e4, 6.4e4, 2.56e5])
    base_sigma_c = 0.5
    sigmas = []
    for n_shots in shot_counts:
        sigma_c = base_sigma_c / np.sqrt(n_shots / shot_counts[0])
        mag = _magnitude_spectrum(
            sigma_c=sigma_c, line_amp=200.0, seed=int(n_shots) % 97 + 11
        )
        sigmas.append(float(np.median(estimate_noise_scatter(f, mag).rms_noise)))
    slope = float(np.polyfit(np.log(shot_counts), np.log(sigmas), 1)[0])
    assert -0.60 < slope < -0.40, f"1/√N slope {slope:.3f} not ≈ −0.5"


def test_old_estimator_plateaus_where_scatter_tracks():
    """Contrast guard: the level-based adaptive estimator flattens on the same
    high-pedestal 1/√N series (it measures the constant pedestal), confirming the
    two estimators are genuinely different and the scatter slope is meaningful."""
    f = _frequencies()
    shot_counts = np.array([1.0e3, 4.0e3, 1.6e4, 6.4e4, 2.56e5])
    base_sigma_c = 0.5
    old, new = [], []
    for n_shots in shot_counts:
        sigma_c = base_sigma_c / np.sqrt(n_shots / shot_counts[0])
        mag = _magnitude_spectrum(
            sigma_c=sigma_c, line_amp=200.0, seed=int(n_shots) % 97 + 11
        )
        old.append(float(np.median(estimate_noise_adaptive(f, mag).rms_noise)))
        new.append(float(np.median(estimate_noise_scatter(f, mag).rms_noise)))
    old_slope = float(np.polyfit(np.log(shot_counts), np.log(old), 1)[0])
    new_slope = float(np.polyfit(np.log(shot_counts), np.log(new), 1)[0])
    # The pedestal flattens the old estimator well short of −0.5; the scatter
    # estimator stays near −0.5. A clear separation is the point.
    assert new_slope < old_slope - 0.2, (
        f"expected scatter slope ({new_slope:.3f}) clearly steeper than old "
        f"({old_slope:.3f})"
    )


def test_smoothing_reduces_under_line_inflation_without_clean_bias():
    """The broad-median smoothing flattens the σ over line-dense regions (lower
    p95/p50) while leaving the clean-region level essentially unbiased."""
    f = _frequencies()
    # Many strong lines clustered in the upper third -> a line-dense band whose
    # skirts inflate the raw per-window scatter there.
    rng = np.random.default_rng(2)
    sigma_c = 0.5
    spec = (rng.standard_normal(_N_BINS) + 1j * rng.standard_normal(_N_BINS)) * sigma_c
    idx = np.arange(_N_BINS)
    for pos in range(int(0.7 * _N_BINS), int(0.95 * _N_BINS), 300):
        spec += 80.0 * 3.0 / ((idx - pos) + 1j * 3.0)
    mag = np.abs(spec)

    raw = estimate_noise_scatter(f, mag, smoothing_mhz=0.0).rms_noise
    smoothed = estimate_noise_scatter(f, mag).rms_noise  # default 800 MHz median

    raw_flatness = float(np.percentile(raw, 95) / np.median(raw))
    sm_flatness = float(np.percentile(smoothed, 95) / np.median(smoothed))
    assert sm_flatness < raw_flatness, "smoothing should reduce σ p95/p50"

    # Median (the bulk clean level) is preserved within a few percent.
    level_ratio = float(np.median(smoothed) / np.median(raw))
    assert 0.92 < level_ratio < 1.08, f"clean-level shifted by {level_ratio:.3f}"


def test_gaussian_pass_removes_median_staircase():
    """The Gaussian second pass (convolve_mhz>0) makes σ smoother than the
    median alone, without materially shifting its level (de-inflation preserved)."""
    f = _frequencies()
    mag = _magnitude_spectrum(sigma_c=0.5, line_amp=60.0, seed=8)

    median_only = estimate_noise_scatter(f, mag, convolve_mhz=0.0).rms_noise
    two_stage = estimate_noise_scatter(f, mag).rms_noise  # default 200 MHz Gaussian

    def roughness(y):
        return float(np.mean(np.abs(np.diff(y, 2))) / np.median(y))

    assert roughness(two_stage) < roughness(median_only), (
        "Gaussian pass should reduce staircase roughness"
    )
    level_ratio = float(np.median(two_stage) / np.median(median_only))
    assert 0.97 < level_ratio < 1.03, f"convolution shifted the level ({level_ratio:.3f})"


def test_smoothing_disabled_passes_through():
    f = _frequencies()
    mag = _magnitude_spectrum(sigma_c=0.5, line_amp=50.0, seed=4)
    nr = estimate_noise_scatter(f, mag, smoothing_mhz=0.0)
    assert nr.bin_info["smoothing_mhz"] == 0.0
    assert np.all(nr.rms_noise > 0)


def test_region_aware_off_uses_fixed_factor():
    """region_aware=False is a valid mode and still returns a sane positive σ."""
    f = _frequencies()
    mag = _magnitude_spectrum(sigma_c=0.5, line_amp=50.0, seed=5)
    result = estimate_noise_scatter(f, mag, region_aware=False)
    assert np.all(result.rms_noise > 0)
    assert result.bin_info["region_aware"] is False


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        estimate_noise_scatter(np.arange(10.0), np.arange(9.0))
