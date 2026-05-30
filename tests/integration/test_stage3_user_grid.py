"""
Stage 3 user-grid snap-back integration tests.

Behaviour under test (Behaviour B):
  - detect_peaks_impl runs detection internally at zpf=1 (native resolution)
    for apex localization.
  - Every returned/persisted Peak is snapped back onto the Stage 1 persisted
    user spectrum: peak.frequency == user_ft.freq_array[peak.index] (exact),
    peak.intensity == user_ft.magnitude_spectrum[peak.index] (exact), and
    peak.snr is computed as intensity / user_rms[index].
  - Internal-grid values kept under peak.properties: internal_frequency,
    internal_snr (for diagnostics).  internal_index and internal_intensity are
    intentionally NOT stored (they are on a different grid and not needed
    downstream; see Stage 3 contract).
  - The internal (zpf=1) grid genuinely differs in size from the user (zpf=2)
    grid, so the snap-back is non-trivial.
  - For experiment 2638 (lower sideband / descending frequency axis), indices
    are NOT monotone with frequency; the snap-back correctly handles that.

All tests share one run of detect_peaks_impl to avoid re-running the expensive
dual-pass detection + noise estimation.
"""

import numpy as np
import pytest

from ftmwpipeline._internal.stage0_impl import import_data_impl
from ftmwpipeline._internal.stage3_impl import detect_peaks_impl
import ftmwpipeline.api as ftmw

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRIM = (26500.0, 40000.0)


# ---------------------------------------------------------------------------
# Session-scoped fixture: one detection run on 2638 shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detection_result(exp_2638_data_path, tmp_path_factory):
    """
    Import 2638, compute FT at zpf=2/expf_us=5.0/trim, estimate noise, detect
    peaks — once per test-module to keep the slow dual-pass from running for
    every test.

    Returns the full dict from detect_peaks_impl (includes 'peaks', 'user_ft',
    'user_rms', 'primary_ft', 'gap_ft', 'primary_noise', 'gap_noise').
    """
    tmp = tmp_path_factory.mktemp("stage3_user_grid")
    fp = str(tmp / "exp.ftmw")
    import_data_impl(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, zpf=2, expf_us=5.0, trim=TRIM)
    ftmw.estimate_noise(fp, method="adaptive")  # frozen Stage 3 reference
    result = detect_peaks_impl(fp)
    return result


# ---------------------------------------------------------------------------
# User-grid fidelity: frequency and intensity are exact
# ---------------------------------------------------------------------------

class TestUserGridSnapBack:
    def test_all_peaks_lie_exactly_on_user_freq_array(self, detection_result):
        """peak.frequency must equal user_ft.freq_array[peak.index] exactly."""
        user_ft = detection_result["user_ft"]
        peaks = detection_result["peaks"]

        assert peaks, "No peaks returned — cannot validate user-grid properties"

        for p in peaks:
            expected_freq = float(user_ft.freq_array[p.index])
            assert p.frequency == expected_freq, (
                f"Peak frequency {p.frequency} != user_ft.freq_array[{p.index}]"
                f" = {expected_freq}"
            )

    def test_all_peaks_intensity_exact_on_user_magnitude(self, detection_result):
        """peak.intensity must equal user_ft.magnitude_spectrum[peak.index] exactly."""
        user_ft = detection_result["user_ft"]
        peaks = detection_result["peaks"]

        for p in peaks:
            expected_int = float(user_ft.magnitude_spectrum[p.index])
            assert p.intensity == expected_int, (
                f"Peak intensity {p.intensity} != user_ft.magnitude_spectrum"
                f"[{p.index}] = {expected_int}"
            )

    def test_snr_matches_intensity_over_user_rms(self, detection_result):
        """peak.snr ~ peak.intensity / user_rms[peak.index] (sample check)."""
        user_rms = detection_result["user_rms"]
        peaks = detection_result["peaks"]

        # Check a sample of peaks (every 10th, plus first and last).
        sample_indices = list(range(0, len(peaks), max(1, len(peaks) // 20)))
        sample_indices = sorted(set(sample_indices + [0, len(peaks) - 1]))

        for i in sample_indices:
            p = peaks[i]
            sd = float(user_rms[p.index])
            if sd > 0:
                expected_snr = p.intensity / sd
                assert np.isclose(p.snr, expected_snr, rtol=1e-6), (
                    f"peaks[{i}].snr {p.snr} != intensity/rms {expected_snr}"
                )


# ---------------------------------------------------------------------------
# Internal-grid properties are present and non-trivially different
# ---------------------------------------------------------------------------

class TestInternalGridProperties:
    def test_internal_properties_present(self, detection_result):
        """Every peak must have internal_frequency and internal_snr under
        peak.properties.  internal_index is intentionally NOT stored (it lives
        on a different grid and is not consumed by Stage 4)."""
        peaks = detection_result["peaks"]

        # internal_index / internal_intensity are intentionally absent per the
        # Stage 3 store-all/promotion contract (see peak_serialization.py docs).
        required_keys = {"internal_frequency", "internal_snr"}
        for p in peaks:
            missing = required_keys - set(p.properties.keys())
            assert not missing, (
                f"Peak at {p.frequency:.3f} MHz missing properties: {missing}"
            )

    def test_internal_grid_genuinely_differs_from_user_grid(self, detection_result):
        """The internal zpf=1 gap spectrum must have a different number of
        points than the user zpf=2 spectrum, confirming non-trivial snap-back."""
        user_ft = detection_result["user_ft"]
        gap_ft = detection_result["gap_ft"]

        assert len(gap_ft.freq_array) != user_ft.n_points, (
            f"gap_ft and user_ft have identical point counts ({user_ft.n_points}); "
            "zpf=1 vs zpf=2 should produce different grid sizes"
        )

    def test_internal_frequency_recorded_for_all_peaks(self, detection_result):
        """All peaks must carry internal_frequency in their properties.

        The snap-back translates from the internal zpf=1 detection grid to the
        user zpf=2 grid.  internal_frequency records the detection position before
        snapping (preserved for curation/diagnosis).  On a zpf=2 user grid the
        zpf=1 frequency points are a strict subset of the user grid, so
        internal_frequency may equal the user frequency -- but the property must
        always be present and must be a finite number.

        Note: internal_index is intentionally NOT stored (see Stage 3 contract).
        """
        peaks = detection_result["peaks"]
        missing = [p for p in peaks if "internal_frequency" not in p.properties]
        assert not missing, (
            f"{len(missing)} peaks missing 'internal_frequency' in properties"
        )
        non_finite = [
            p for p in peaks
            if not np.isfinite(p.properties.get("internal_frequency", float("nan")))
        ]
        assert not non_finite, (
            f"{len(non_finite)} peaks have non-finite internal_frequency"
        )


# ---------------------------------------------------------------------------
# Sanity: reasonable peak set within persisted trim
# ---------------------------------------------------------------------------

class TestDetectionSanity:
    def test_reasonable_peak_count(self, detection_result):
        """Experiment 2638 should yield well over 100 peaks, some strong."""
        peaks = detection_result["peaks"]
        assert len(peaks) > 100, (
            f"Only {len(peaks)} peaks detected — implausibly few for exp 2638"
        )

    def test_some_strong_peaks(self, detection_result):
        """At least some peaks must be classified strong."""
        peaks = detection_result["peaks"]
        strong = [p for p in peaks if p.is_strong]
        assert strong, "No peaks classified strong"

    def test_all_frequencies_within_trim(self, detection_result):
        """All reported peak frequencies must lie within the persisted trim range."""
        peaks = detection_result["peaks"]
        freqs = np.array([p.frequency for p in peaks])
        # Allow 1 MHz tolerance for grid-snap rounding.
        assert freqs.min() >= TRIM[0] - 1.0, (
            f"Peak at {freqs.min():.2f} MHz below trim min {TRIM[0]}"
        )
        assert freqs.max() <= TRIM[1] + 1.0, (
            f"Peak at {freqs.max():.2f} MHz above trim max {TRIM[1]}"
        )

    def test_peaks_sorted_ascending_by_frequency(self, detection_result):
        """The returned list must be sorted ascending by frequency."""
        peaks = detection_result["peaks"]
        freqs = [p.frequency for p in peaks]
        assert freqs == sorted(freqs), (
            "Peaks not sorted ascending by frequency"
        )


# ---------------------------------------------------------------------------
# Descending-axis handling (2638 is lower-sideband / descending freq axis)
# ---------------------------------------------------------------------------

class TestDescendingAxisHandling:
    def test_indices_not_monotone_with_frequency(self, detection_result):
        """Experiment 2638 has a descending frequency axis (lower sideband):
        higher index -> lower frequency.  Peak indices must therefore NOT be
        monotone increasing with frequency, confirming the snap-back handled
        the descending axis correctly."""
        peaks = detection_result["peaks"]
        # peaks are sorted ascending by frequency; on a descending axis the
        # HDF5 index must decrease as frequency increases.
        indices = [p.index for p in peaks if p.index is not None]
        assert len(indices) >= 2, "Too few peaks to check index monotonicity"

        # A monotone-increasing index sequence would mean the snap-back ignored
        # the descending axis.  At least some adjacent pairs must be decreasing.
        decreasing_pairs = sum(
            1 for a, b in zip(indices, indices[1:]) if b < a
        )
        assert decreasing_pairs > 0, (
            "Peak indices are monotone increasing with frequency on a descending "
            "axis — snap-back may not have handled the lower-sideband correctly"
        )

    def test_user_freq_array_is_descending(self, detection_result):
        """Verify experiment 2638 really does have a descending frequency axis
        (index 0 = highest frequency) so the above test is meaningful."""
        user_ft = detection_result["user_ft"]
        freqs = user_ft.freq_array
        assert freqs[0] > freqs[-1], (
            f"user_ft.freq_array is ascending ({freqs[0]:.1f} < {freqs[-1]:.1f}); "
            "expected descending for 2638 lower sideband"
        )
