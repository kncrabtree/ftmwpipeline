"""
Tests for NoiseResult serialization and deserialization.

The HDF5 serialization stores the boolean noise_mask compactly as signal
indices (its False positions) and the per-bin sigma array verbatim, so the
round-trip is exact for any estimator.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from ftmwpipeline.io.noise_result_serialization import (
    _extract_signal_indices,
    _reconstruct_noise_mask,
    load_noise_result_from_hdf5,
    save_noise_result_to_hdf5,
)
from ftmwpipeline.preprocessing.noise_estimation import (
    NoiseResult,
    estimate_noise_scatter,
)


@pytest.fixture
def test_output_dir():
    """Create and cleanup test output directory."""
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    for file in output_dir.glob("test_noise_result_*.h5"):
        file.unlink()


@pytest.fixture
def sample_spectrum_data():
    """Create sample frequency and magnitude data for testing."""
    n_points = 100000
    frequencies = np.linspace(26500, 40000, n_points)  # MHz

    np.random.seed(42)
    magnitudes = np.random.exponential(1.0, n_points)  # noise baseline

    # Add some spectral peaks (signal)
    for peak_freq in (28000, 32000, 36000):
        peak_idx = int(np.argmin(np.abs(frequencies - peak_freq)))
        peak_width = 50
        peak_indices = np.arange(peak_idx - peak_width, peak_idx + peak_width + 1)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < n_points)]
        gaussian = np.exp(
            -((peak_indices - peak_idx) ** 2) / (2 * (peak_width / 3) ** 2)
        )
        magnitudes[peak_indices] += 10 * gaussian

    return frequencies, magnitudes


@pytest.fixture
def sample_noise_result(sample_spectrum_data):
    """Create a realistic NoiseResult via the scatter estimator."""
    frequencies, magnitudes = sample_spectrum_data
    return estimate_noise_scatter(frequencies, magnitudes)


class TestSignalIndicesConversion:
    """Signal-indices extraction and mask reconstruction."""

    def test_extract_signal_indices(self):
        """Boolean mask -> signal indices (the False positions)."""
        noise_mask = np.array([True, True, False, True, False, False, True])
        signal_indices = _extract_signal_indices(noise_mask)
        np.testing.assert_array_equal(
            signal_indices, np.array([2, 4, 5], dtype=np.int32)
        )

    def test_reconstruct_noise_mask(self):
        """Signal indices -> boolean mask."""
        signal_indices = np.array([2, 4, 5], dtype=np.int32)
        noise_mask = _reconstruct_noise_mask(signal_indices, 7)
        expected = np.array([True, True, False, True, False, False, True])
        np.testing.assert_array_equal(noise_mask, expected)

    def test_roundtrip_signal_indices(self, sample_noise_result):
        """mask -> indices -> mask is identity."""
        original_mask = sample_noise_result.noise_mask
        signal_indices = _extract_signal_indices(original_mask)
        reconstructed = _reconstruct_noise_mask(signal_indices, len(original_mask))
        np.testing.assert_array_equal(reconstructed, original_mask)

    def test_storage_reduction(self, sample_noise_result):
        """Signal indices are smaller than the full boolean mask for the
        typical mostly-noise mask."""
        original_mask = sample_noise_result.noise_mask
        signal_indices = _extract_signal_indices(original_mask)
        assert signal_indices.nbytes < original_mask.nbytes


class TestSaveLoadRoundtrip:
    """Verbatim save/load round-trip."""

    TEST_PREFIX = "test_noise_result_"

    def _get_test_file(self, output_dir, suffix):
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"

    def test_save_load_roundtrip_exact(
        self, sample_spectrum_data, sample_noise_result, test_output_dir
    ):
        """Round-trip restores rms_noise and noise_mask bit-for-bit."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "roundtrip")

        with h5py.File(test_file, "w") as f:
            grp = f.create_group("noise_result")
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, grp)

        with h5py.File(test_file, "r") as f:
            loaded = load_noise_result_from_hdf5(
                f["noise_result"], frequencies, magnitudes
            )

        np.testing.assert_array_equal(loaded.noise_mask, sample_noise_result.noise_mask)
        np.testing.assert_array_equal(loaded.rms_noise, sample_noise_result.rms_noise)
        assert loaded.bin_info.keys() == sample_noise_result.bin_info.keys()
        for key in ("noise_fraction", "algorithm"):
            if key in sample_noise_result.bin_info:
                assert loaded.bin_info[key] == sample_noise_result.bin_info[key]

    def test_hdf5_structure_validation(
        self, sample_spectrum_data, sample_noise_result, test_output_dir
    ):
        """The persisted group carries the verbatim-sigma structure."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "structure")

        with h5py.File(test_file, "w") as f:
            grp = f.create_group("noise_result")
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, grp)

        with h5py.File(test_file, "r") as f:
            grp = f["noise_result"]
            assert "signal_indices" in grp
            assert "rms_noise_full" in grp
            assert "bin_info" in grp
            assert "algorithm_info" in grp
            algo = grp["algorithm_info"]
            assert algo.attrs["method"] == "verbatim_sigma"
            assert "version" in algo.attrs
            assert "storage_optimization" in algo.attrs

    def test_error_handling(self, test_output_dir):
        """Mismatched array lengths raise on save."""
        test_file = self._get_test_file(test_output_dir, "errors")
        frequencies = np.linspace(0, 100, 100)
        magnitudes = np.random.random(50)  # different length
        noise_result = NoiseResult(
            rms_noise=np.random.random(100),
            noise_mask=np.random.random(100) > 0.5,
            bin_info={},
        )
        with pytest.raises(RuntimeError, match="Failed to save NoiseResult to HDF5"):
            with h5py.File(test_file, "w") as f:
                grp = f.create_group("noise_result")
                save_noise_result_to_hdf5(noise_result, frequencies, magnitudes, grp)


class TestScatterEstimatorRoundTrip:
    """The scatter estimator's sigma is stored and restored verbatim."""

    def _scatter_spectrum(self):
        n = 20000
        frequencies = np.linspace(26500, 40000, n)
        rng = np.random.default_rng(0)
        spec = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.5
        idx = np.arange(n)
        for pos in (n // 4, n // 2, 3 * n // 4):
            spec += 60.0 * 3.0 / ((idx - pos) + 1j * 3.0)
        return frequencies, np.abs(spec)

    def test_scatter_rms_is_stored_verbatim(self, tmp_path):
        frequencies, magnitudes = self._scatter_spectrum()
        original = estimate_noise_scatter(frequencies, magnitudes)

        path = tmp_path / "scatter_roundtrip.h5"
        with h5py.File(path, "w") as h5f:
            grp = h5f.create_group("noise_result")
            save_noise_result_to_hdf5(original, frequencies, magnitudes, grp)
            assert "rms_noise_full" in grp

        with h5py.File(path, "r") as h5f:
            restored = load_noise_result_from_hdf5(
                h5f["noise_result"], frequencies, magnitudes
            )

        np.testing.assert_array_equal(restored.rms_noise, original.rms_noise)
        np.testing.assert_array_equal(restored.noise_mask, original.noise_mask)
        assert restored.bin_info["algorithm"] == original.bin_info["algorithm"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
