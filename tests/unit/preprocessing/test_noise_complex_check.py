"""Tests for the complex-domain σ cross-check guardrail.

The real and imaginary parts of pure complex-Gaussian noise are each
``N(0, σ_c)``: symmetric, zero-mean, no Rayleigh skew/pedestal and no Rician
regime. So :func:`estimate_noise_complex_scatter` recovers ``σ_x = σ_c·√2``
nearly unbiased, where the magnitude estimator carries an intrinsic few-percent
low bias. :func:`estimate_active_ft_noise` folds the comparison into
``bin_info`` as a guardrail.
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.noise_estimation import (
    SCATTER_COMPLEX_DIVERGENCE_WARN,
    estimate_active_ft_noise,
    estimate_noise_complex_scatter,
)

# Realistic active-FT geometry: df = 1/T_active ~ 0.067 MHz/bin (a ~13 us record),
# so the physical-MHz knobs translate to a production-like number of bins. (A
# much coarser grid lets the running-median pedestal over-fit the noise and
# inflates the magnitude estimator's low bias.)
_N = 60000
_DF = 1.0 / 15.0
_F_LO = 26500.0
_F_HI = _F_LO + _DF * (_N - 1)
_SIGMA_C = 1.0
_TRUE_RMS = _SIGMA_C * np.sqrt(2.0)


def _freqs(ascending: bool = True) -> np.ndarray:
    f = np.linspace(_F_LO, _F_HI, _N)
    return f if ascending else f[::-1]


def _complex_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, _SIGMA_C, _N) + 1j * rng.normal(0.0, _SIGMA_C, _N)


def _with_lines(cs: np.ndarray, amp: float) -> np.ndarray:
    """Add a few strong complex Lorentzian lines to a noise spectrum."""
    out = cs.copy()
    idx = np.arange(_N)
    for pos in (_N // 5, _N // 2, 4 * _N // 5):
        lor = amp / (1.0 + ((idx - pos) / 3.0) ** 2)
        out += lor * np.exp(1j * 0.7)  # arbitrary fixed phase
    return out


def test_complex_scatter_unbiased_on_pure_noise():
    cs = _complex_noise(0)
    sigma = estimate_noise_complex_scatter(
        _freqs(), cs, np.ones(_N, dtype=bool)
    )
    # Within a few percent of the true complex RMS -- the magnitude estimator
    # is ~4% low here, the complex one is not.
    assert np.median(sigma) == pytest.approx(_TRUE_RMS, rel=0.03)


def test_complex_scatter_orientation_free():
    """A descending grid needs no pre-sort: the estimate is the same data."""
    cs = _complex_noise(1)
    asc = estimate_noise_complex_scatter(_freqs(True), cs, np.ones(_N, dtype=bool))
    desc = estimate_noise_complex_scatter(
        _freqs(False), cs[::-1], np.ones(_N, dtype=bool)
    )
    assert np.median(asc) == pytest.approx(np.median(desc), rel=1e-3)


def test_complex_scatter_ignores_masked_lines():
    """Strong lines, when masked out, do not inflate the complex estimate."""
    cs = _with_lines(_complex_noise(2), amp=200.0)
    # Mask the line cores (and a small skirt) as the magnitude pass would.
    mask = np.ones(_N, dtype=bool)
    for pos in (_N // 5, _N // 2, 4 * _N // 5):
        mask[pos - 30 : pos + 30] = False
    sigma = estimate_noise_complex_scatter(_freqs(), cs, mask)
    assert np.median(sigma) == pytest.approx(_TRUE_RMS, rel=0.05)


def test_active_ft_noise_carries_complex_diagnostic():
    cs = _complex_noise(3)
    result = estimate_active_ft_noise(_freqs(), cs)
    info = result.bin_info
    for key in (
        "complex_sigma_median",
        "complex_magnitude_ratio",
        "complex_divergence",
        "complex_divergence_warn",
    ):
        assert key in info
    # On clean noise the magnitude estimator sits slightly below the complex one
    # but well inside the warn threshold.
    assert 0.85 < info["complex_magnitude_ratio"] <= 1.05
    assert info["complex_divergence"] < SCATTER_COMPLEX_DIVERGENCE_WARN
    assert info["complex_divergence_warn"] is False
