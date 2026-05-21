"""
Unit tests for the Stage 4 complex-edge coherence statistic.

Regression tests against the research-report calibration
(``dev-docs/research/complex-edge-coherence/report.md``): on clean complex-
Gaussian noise ``S_coh`` has mean ~0.886 and is M-independent; a coherent band
fires above the T_edge = 3 threshold.
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.edge_coherence import (
    DEFAULT_EDGE_THRESHOLD,
    NULL_MEAN,
    above_threshold_intervals,
    coherence_statistic,
    max_cumsum_statistic,
    rolling_coherence,
)


def _complex_noise(n, sigma, seed):
    """Complex Gaussian noise with E[|n|^2] = sigma^2."""
    rng = np.random.default_rng(seed)
    s = sigma / np.sqrt(2.0)
    return rng.normal(0, s, n) + 1j * rng.normal(0, s, n)


class TestNullCalibration:
    def test_null_mean_matches_closed_form(self):
        """Mean S_coh on clean noise matches sqrt(pi/4) ~= 0.886."""
        sigma = 0.02
        vals = []
        for m in (8, 16, 32, 64, 128):
            for trial in range(200):
                z = _complex_noise(m, sigma, seed=1000 * m + trial)
                vals.append(coherence_statistic(z, sigma))
        assert np.mean(vals) == pytest.approx(NULL_MEAN, abs=0.03)

    def test_null_is_m_independent(self):
        """The null mean does not depend on the band width M."""
        sigma = 0.02
        means = {}
        for m in (16, 64, 256):
            vals = [
                coherence_statistic(_complex_noise(m, sigma, 7 * m + t), sigma)
                for t in range(300)
            ]
            means[m] = np.mean(vals)
        assert max(means.values()) - min(means.values()) < 0.1

    def test_clean_band_rarely_exceeds_threshold(self):
        """Per-band false-positive rate at T_edge = 3 is well below 1%."""
        sigma = 0.015
        fired = sum(
            coherence_statistic(_complex_noise(64, sigma, t), sigma)
            > DEFAULT_EDGE_THRESHOLD
            for t in range(2000)
        )
        assert fired / 2000 < 0.01


class TestCoherentBand:
    def test_coherent_band_fires(self):
        """A phase-aligned (coherent) band drives S_coh far above threshold."""
        sigma = 0.02
        m = 64
        # A coherent contamination of amplitude ~sigma on every bin.
        coherent = np.full(m, 0.5 * sigma + 0.0j)
        z = coherent + _complex_noise(m, sigma, seed=3)
        s = coherence_statistic(z, sigma)
        # S_coh ~ (L/sigma)*sqrt(M) = 0.5*8 = 4
        assert s > DEFAULT_EDGE_THRESHOLD

    def test_degenerate_inputs(self):
        assert coherence_statistic(np.array([]), 1.0) == 0.0
        assert coherence_statistic(np.array([1 + 1j]), 0.0) == 0.0


class TestMaxCumsum:
    def test_hotspot_localised(self):
        """Max-cumsum localises a coherent sub-stretch at its far edge."""
        sigma = 0.02
        z = _complex_noise(64, sigma, seed=5)
        z[:20] += 3.0 * sigma  # coherent hot spot in the first 20 bins
        stat, hot = max_cumsum_statistic(z, sigma)
        assert stat > DEFAULT_EDGE_THRESHOLD
        assert 10 <= hot <= 30  # near the edge of the hot stretch


class TestRollingCoherence:
    def test_shape_and_edges(self):
        n, m = 500, 64
        z = _complex_noise(n, 0.02, seed=9)
        roll = rolling_coherence(z, np.full(n, 0.02), band_m=m)
        assert roll.shape == (n,)
        # The first/last ~M/2 points have no centred band -> NaN.
        assert np.isnan(roll[0]) and np.isnan(roll[-1])
        assert np.isfinite(roll[n // 2])

    def test_fires_on_an_injected_line(self):
        """Rolling S_coh rises above threshold around a strong coherent line."""
        n = 2000
        sigma = 0.02
        z = _complex_noise(n, sigma, seed=11)
        # A coherent skirt centred at index 1000 (slow 1/df decay).
        idx = np.arange(n)
        with np.errstate(divide="ignore"):
            skirt = 1.0 / np.maximum(np.abs(idx - 1000), 1.0)
        z = z + 30.0 * sigma * skirt
        roll = rolling_coherence(z, np.full(n, sigma), band_m=64)
        assert np.nanmax(roll) > DEFAULT_EDGE_THRESHOLD
        # The peak of the statistic sits near the line.
        assert abs(int(np.nanargmax(roll)) - 1000) < 100

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rolling_coherence(np.zeros(10, complex), np.zeros(9), band_m=4)


class TestAboveThresholdIntervals:
    def test_finds_contiguous_runs(self):
        roll = np.array([np.nan, 1.0, 5.0, 6.0, 1.0, 1.0, 4.0, 4.0, 4.0, 1.0, np.nan])
        intervals = above_threshold_intervals(roll, threshold=3.0)
        assert intervals == [(2, 3), (6, 8)]

    def test_empty_when_all_below(self):
        roll = np.full(50, 0.5)
        assert above_threshold_intervals(roll, threshold=3.0) == []
