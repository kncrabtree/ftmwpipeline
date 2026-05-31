"""
Unit tests for the Stage 3 two-pass driver and SNR classification.

Synthetic spectra only: classification bin edges, gap-pass recovery of a weak
line the (simulated) apodized primary pass misses, suppression of a strong
line's sidelobe via the continuous gap leakage-aware floor, the gap-pass
switch, provenance, validation. Real-data 2638 behaviour is the integration
suite.
"""

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import PeakClassification
from ftmwpipeline.preprocessing.peak_detection import (
    classify_by_snr,
    detect_peaks,
)


def _gauss(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


class TestClassifyBySnr:
    def test_bin_edges(self):
        # weak < t1 <= medium < t2 <= strong
        assert classify_by_snr(9.999, 10.0, 50.0) is PeakClassification.WEAK
        assert classify_by_snr(10.0, 10.0, 50.0) is PeakClassification.MEDIUM
        assert classify_by_snr(49.999, 10.0, 50.0) is PeakClassification.MEDIUM
        assert classify_by_snr(50.0, 10.0, 50.0) is PeakClassification.STRONG

    def test_equal_thresholds_collapse_medium(self):
        assert classify_by_snr(5.0, 10.0, 10.0) is PeakClassification.WEAK
        assert classify_by_snr(10.0, 10.0, 10.0) is PeakClassification.STRONG

    def test_invalid_thresholds_rejected(self):
        with pytest.raises(ValueError, match="weak_medium_snr"):
            classify_by_snr(5.0, 50.0, 10.0)
        with pytest.raises(ValueError, match="weak_medium_snr"):
            classify_by_snr(5.0, 0.0, 10.0)


def _gap_leakage_amp(freq, sd, amp=25.0):
    """Continuous gap leakage-floor amplitude (same length as ``sd``): a high
    additive floor over the ~10037.5-10062.5 MHz sidelobe stretch around the
    @10050 bump and zero elsewhere. The gap threshold ``min_snr*sigma + amp``
    therefore buries the (amplitude-20) sidelobe while leaving the @10500 weak
    line, in a leakage-free gap, at the plain ``min_snr*sigma`` floor. Excludes
    the strong line @10000 and the weak line @10500."""
    out = np.zeros_like(sd)
    out[np.abs(freq - 10050.0) <= 12.5] = amp
    return out


@pytest.fixture
def two_pass_spectra():
    """Strong line @10000; sidelobe bump @10050 (under the gap leakage floor,
    see ``_gap_leakage_amp``); weak line @10500 (in a leakage-free gap).
    Primary (apodized) sees only the strong line; gap (unapodized) sees all
    three."""
    freq = np.linspace(9000.0, 11000.0, 8000)  # 0.25 MHz/pt
    sd = np.ones_like(freq)

    strong = _gauss(freq, 10000.0, 300.0, 2.0)
    sidelobe = _gauss(freq, 10050.0, 20.0, 2.0)
    weak = _gauss(freq, 10500.0, 6.0, 2.0)

    primary_mag = strong.copy()  # apodization suppressed sidelobe + weak line
    gap_mag = strong + sidelobe + weak
    return freq, primary_mag, gap_mag, sd


class TestTwoPassDriver:
    def test_gap_pass_recovers_weak_line_primary_misses(self, two_pass_spectra):
        freq, primary_mag, gap_mag, sd = two_pass_spectra
        peaks = detect_peaks(
            freq,
            primary_mag,
            sd,
            freq,
            gap_mag,
            sd,
            min_snr=3.0,
            weak_medium_snr=10.0,
            medium_strong_snr=50.0,
            gap_leakage_amp=_gap_leakage_amp(freq, sd),
        )
        freqs = np.array([p.frequency for p in peaks])

        # Strong line found by primary, classified STRONG.
        strong = [p for p in peaks if abs(p.frequency - 10000.0) < 3.0]
        assert len(strong) == 1
        assert strong[0].classification is PeakClassification.STRONG
        assert strong[0].properties["detection_pass"] == "primary"

        # Weak line in the gap recovered by the gap pass, classified WEAK.
        weak = [p for p in peaks if abs(p.frequency - 10500.0) < 3.0]
        assert len(weak) == 1
        assert weak[0].classification is PeakClassification.WEAK
        assert weak[0].properties["detection_pass"] == "gap"

        # Sidelobe @10050 sits under the continuous gap leakage floor -> buried.
        assert not np.any(np.abs(freqs - 10050.0) < 5.0)

        # Output sorted by frequency.
        assert list(freqs) == sorted(freqs)

    def test_gap_pass_disabled_does_not_recover_weak(self, two_pass_spectra):
        """Gap-pass switch: with the gap pass off, only the primary list
        remains."""
        freq, primary_mag, gap_mag, sd = two_pass_spectra
        peaks = detect_peaks(
            freq,
            primary_mag,
            sd,
            freq,
            gap_mag,
            sd,
            min_snr=3.0,
            run_gap_pass=False,
        )
        freqs = np.array([p.frequency for p in peaks])
        assert np.any(np.abs(freqs - 10000.0) < 3.0)  # strong still found
        assert not np.any(np.abs(freqs - 10500.0) < 3.0)  # weak NOT recovered
        assert all(p.properties["detection_pass"] == "primary" for p in peaks)

    def test_without_leakage_floor_sidelobe_leaks_through(self, two_pass_spectra):
        """Control for the gap leakage floor: with ``gap_leakage_amp`` omitted
        (plain ``min_snr*sigma`` floor) the amplitude-20 sidelobe @10050 clears
        the threshold and is detected -- confirming it is the continuous floor,
        not luck, that buries it in
        ``test_gap_pass_recovers_weak_line_primary_misses``."""
        freq, primary_mag, gap_mag, sd = two_pass_spectra
        peaks = detect_peaks(
            freq,
            primary_mag,
            sd,
            freq,
            gap_mag,
            sd,
            min_snr=3.0,
            min_exclusion_mhz=0.0,
        )
        freqs = np.array([p.frequency for p in peaks])
        assert np.any(np.abs(freqs - 10050.0) < 5.0)

    def test_primary_only_when_no_gap_arrays(self, two_pass_spectra):
        freq, primary_mag, gap_mag, sd = two_pass_spectra
        peaks = detect_peaks(freq, primary_mag, sd, min_snr=3.0)
        assert len(peaks) == 1
        assert peaks[0].properties["detection_pass"] == "primary"


class TestValidation:
    def test_nonpositive_min_snr_rejected(self):
        x = np.linspace(0.0, 10.0, 100)
        with pytest.raises(ValueError, match="min_snr"):
            detect_peaks(x, x, np.ones_like(x), min_snr=0.0)

    def test_primary_shape_mismatch_rejected(self):
        x = np.linspace(0.0, 10.0, 100)
        with pytest.raises(ValueError, match="primary"):
            detect_peaks(x, x, np.ones(99))

    def test_gap_shape_mismatch_rejected(self):
        x = np.linspace(0.0, 10.0, 100)
        with pytest.raises(ValueError, match="gap"):
            detect_peaks(x, x, np.ones_like(x), x, x, np.ones(99))
