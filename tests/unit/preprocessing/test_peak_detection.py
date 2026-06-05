"""
Unit tests for the ported Stage 3 peak locator (``locate_peaks``).

Covers the core second-derivative detector, the per-point threshold path,
the strong-peak split/merge heuristic, the ``PeakResult`` shape, and input
validation. Synthetic spectra only -- real-data 2638 checks live in the
integration suite.
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.peak_detection import PeakResult, locate_peaks


def _gaussian(x, center, amp, width):
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


@pytest.fixture
def clean_spectrum():
    """Three well-separated Gaussian peaks on a flat baseline, no noise."""
    x = np.linspace(8000.0, 12000.0, 4000)  # 1 MHz/pt
    centers = [8500.0, 10000.0, 11500.0]
    amps = [3.0, 1.5, 5.0]
    width = 2.0  # MHz
    y = np.zeros_like(x)
    for c, a in zip(centers, amps):
        y += _gaussian(x, c, a, width)
    return x, y, centers


class TestPeakResultShape:
    def test_returns_named_tuple_with_three_arrays(self, clean_spectrum):
        x, y, _ = clean_spectrum
        result = locate_peaks(x, y)
        assert isinstance(result, PeakResult)
        assert result.freqs.shape == result.intensities.shape == result.indices.shape
        # indices must address the input arrays
        np.testing.assert_array_equal(result.freqs, x[result.indices])
        np.testing.assert_array_equal(result.intensities, y[result.indices])


class TestBasicDetection:
    def test_finds_all_known_peaks_without_threshold(self, clean_spectrum):
        x, y, centers = clean_spectrum
        result = locate_peaks(x, y, window=11, order=3)
        # Every known center should have a detection within ~1 linewidth.
        for c in centers:
            assert np.min(np.abs(result.freqs - c)) < 2.0

    def test_threshold_rejects_subthreshold_peaks(self):
        x = np.linspace(8000.0, 12000.0, 4000)
        # One strong peak (amp 5) and one weak peak (amp 0.4) on flat baseline.
        y = _gaussian(x, 9000.0, 5.0, 2.0) + _gaussian(x, 11000.0, 0.4, 2.0)
        thresh = np.full_like(y, 1.0)  # between the two peak heights
        result = locate_peaks(x, y, window=11, order=3, thresh=thresh)
        assert len(result.freqs) == 1
        assert abs(result.freqs[0] - 9000.0) < 2.0

    def test_all_below_threshold_returns_empty(self):
        x = np.linspace(8000.0, 12000.0, 2000)
        y = _gaussian(x, 10000.0, 0.5, 2.0)
        thresh = np.full_like(y, 10.0)
        result = locate_peaks(x, y, window=11, order=3, thresh=thresh)
        assert len(result.freqs) == 0
        assert len(result.indices) == 0


class TestStrongPeakSplitMerge:
    def test_single_strong_peak_not_double_counted(self):
        """A tall, sharp line must yield exactly one detection at its center.

        With a Savitzky-Golay window wider than the line, the concave-down
        second-derivative test can flag a minimum on each flank of a strong
        peak. The split/merge heuristic must collapse that pair back to the
        line center rather than report two peaks.
        """
        x = np.linspace(9990.0, 10010.0, 4000)  # 5 kHz/pt, fine grid
        # Very tall, very narrow line relative to a wide SG window.
        y = _gaussian(x, 10000.0, 100.0, 0.05)
        thresh = np.full_like(y, 5.0)
        result = locate_peaks(x, y, window=21, order=3, thresh=thresh)
        assert (
            len(result.freqs) == 1
        ), f"expected merge to a single peak, got {result.freqs}"
        assert abs(result.freqs[0] - 10000.0) < 0.05

    def test_two_genuine_close_peaks_not_merged(self):
        """A real doublet (a true valley between the lines) stays two peaks."""
        x = np.linspace(9980.0, 10020.0, 8000)
        y = _gaussian(x, 9995.0, 4.0, 0.6) + _gaussian(x, 10005.0, 4.0, 0.6)
        thresh = np.full_like(y, 1.0)
        result = locate_peaks(x, y, window=9, order=3, thresh=thresh)
        assert len(result.freqs) == 2
        assert abs(result.freqs[0] - 9995.0) < 1.0
        assert abs(result.freqs[1] - 10005.0) < 1.0


class TestInputValidation:
    @pytest.fixture
    def xy(self):
        x = np.linspace(0.0, 100.0, 500)
        return x, np.sin(x)

    def test_even_window_rejected(self, xy):
        x, y = xy
        with pytest.raises(ValueError, match="odd"):
            locate_peaks(x, y, window=8, order=3)

    def test_window_not_greater_than_order_rejected(self, xy):
        x, y = xy
        with pytest.raises(ValueError, match="greater than order"):
            locate_peaks(x, y, window=5, order=5)

    def test_nonpositive_order_rejected(self, xy):
        x, y = xy
        with pytest.raises(ValueError, match="order"):
            locate_peaks(x, y, window=7, order=0)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            locate_peaks(np.arange(10.0), np.arange(9.0))

    def test_non_1d_rejected(self):
        a = np.ones((4, 4))
        with pytest.raises(ValueError, match="1-dimensional"):
            locate_peaks(a, a)

    def test_thresh_shape_mismatch_rejected(self, xy):
        x, y = xy
        with pytest.raises(ValueError, match="thresh"):
            locate_peaks(x, y, window=7, order=3, thresh=np.ones(3))
