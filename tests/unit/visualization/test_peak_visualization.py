"""Unit tests for the peak-detection plot panel logic.

The 1-panel (default) vs 2-panel (``snr_histogram=True``) behavior and the
promotion-cutoff marker live entirely in ``plot_peak_detection``; they need
no real spectrum. These replace two ex-integration tests that drove the full
2638 pipeline (~60 s) to assert the same axis counts.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import Peak, PeakClassification
from ftmwpipeline.visualization.peak_visualization import plot_peak_detection


@pytest.fixture
def synthetic_spectrum():
    """A tiny spectrum + a few classified peaks (above and below a cutoff)."""
    freq = np.linspace(26500.0, 40000.0, 200)
    rng = np.random.default_rng(0)
    mag = np.abs(rng.normal(0.0, 1e-3, freq.size)) + 1e-4
    rms = np.full(freq.size, 1e-3)
    peaks = [
        Peak(
            frequency=float(freq[i]),
            intensity=float(mag[i] + amp),
            index=int(i),
            snr=snr,
            noise_std_local=1e-3,
            classification=cls,
            detection_pass=dp,
            promoted=snr >= 3.0,
        )
        for i, amp, snr, cls, dp in (
            (40, 5e-2, 50.0, PeakClassification.STRONG, "primary"),
            (90, 8e-3, 8.0, PeakClassification.MEDIUM, "primary"),
            (140, 2e-3, 2.2, PeakClassification.WEAK, "gap"),
        )
    ]
    return freq, mag, rms, peaks


def test_default_is_single_panel(synthetic_spectrum):
    freq, mag, rms, peaks = synthetic_spectrum
    fig = plot_peak_detection(freq, mag, rms, peaks)
    try:
        assert len(fig.get_axes()) == 1
    finally:
        plt.close(fig)


def test_snr_histogram_adds_second_panel(synthetic_spectrum):
    freq, mag, rms, peaks = synthetic_spectrum
    fig = plot_peak_detection(
        freq, mag, rms, peaks, snr_histogram=True, promotion_min_snr=3.0
    )
    try:
        axes = fig.get_axes()
        assert len(axes) == 2
        # The promotion cutoff is drawn as a vertical line on the hist panel.
        hist_ax = axes[1]
        assert any(
            np.allclose(line.get_xdata(), 3.0) for line in hist_ax.get_lines()
        ), "promotion cutoff line not drawn at SNR=3.0"
    finally:
        plt.close(fig)


def test_histogram_without_cutoff_still_two_panels(synthetic_spectrum):
    freq, mag, rms, peaks = synthetic_spectrum
    fig = plot_peak_detection(freq, mag, rms, peaks, snr_histogram=True)
    try:
        assert len(fig.get_axes()) == 2
    finally:
        plt.close(fig)
