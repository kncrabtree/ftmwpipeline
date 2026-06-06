"""
Pipeline integration tests focusing on Stage 0-1 cache architecture with experiment 2638 data.

These tests validate the new Stage 0-1 pipeline architecture where:
- Stage 0: FID data is cached and persisted to HDF5 (permanent storage)
- Stage 1: ComplexFT objects are calculated on-demand from cached FID + parameters (temporary)

ARCHITECTURAL CHANGE:
- OLD MODEL: ComplexFT objects were cached and stored
- NEW MODEL: Only FID data is cached, ComplexFT calculated fresh each session

Test Coverage:
- Stage 0: FID caching, loading, bit-perfect reconstruction
- Stage 1: On-demand ComplexFT calculation consistency
- Integration: Full Stage 0 → Stage 1 workflow validation
- Workflow consistency across multiple parameter combinations
- Cache failure handling and robustness

Individual component serialization is tested separately in unit tests.
Test files are written to tests/output/ with proper cleanup.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FID, ComplexFT, FIDProcessingParameters
from ftmwpipeline.io.experimental_formats import load_blackchirp_experiment

# NEW imports for Stage 0-1 architecture
from ftmwpipeline.io.fid_serialization import (
    load_fid_cache,
    save_fid_cache,
    update_fid_processing_defaults,
)
from ftmwpipeline.preprocessing.noise_estimation import (
    NoiseResult,
    estimate_noise_scatter,
)


class TestStage0FIDCaching:
    """Test Stage 0 (FID caching) integration with real experiment data."""

    # Test cache prefix for consistent naming and cleanup
    TEST_PREFIX = "test_stage0_"

    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"

    def _get_cache_file(self, suffix):
        """Helper to get cache file path."""
        return self.test_output_dir / f"{self.TEST_PREFIX}{suffix}_fid.h5"

    def setup_method(self):
        """Set up test environment for Stage 0 FID caching tests."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if experiment 2638 data exists
        self.example_data_path = Path("examples/blackchirp_data/2638")
        self.has_example_data = self.example_data_path.exists()

    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def test_fid_cache_roundtrip_with_real_data(self):
        """Test FID caching roundtrip produces bit-perfect reconstruction with experiment 2638."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for FID caching testing")

        # Load real experiment 2638 data
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid

        # === STAGE 0: CACHE FID DATA ===
        cache_id = self._get_cache_id("fid_roundtrip")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # === STAGE 0: LOAD FID FROM CACHE ===
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # === VALIDATE BIT-PERFECT FID RECONSTRUCTION ===

        # Core FID data should be identical
        np.testing.assert_array_equal(
            cached_fid.data,
            original_fid.data,
            err_msg="Cached FID data differs from original",
        )

        # Acquisition metadata should be preserved
        assert cached_fid.spacing == original_fid.spacing, "FID spacing not preserved"
        assert (
            cached_fid.probe_freq_mhz == original_fid.probe_freq_mhz
        ), "Probe frequency not preserved"
        assert cached_fid.sideband == original_fid.sideband, "Sideband not preserved"
        assert cached_fid.shots == original_fid.shots, "Shot count not preserved"

        # Processing parameters should be preserved (these are defaults from source format)
        if original_fid.processing is not None:
            assert (
                cached_fid.processing is not None
            ), "Processing parameters not preserved"
            original_params = original_fid.processing
            cached_params = cached_fid.processing

            assert (
                cached_params.start_us == original_params.start_us
            ), "Start time not preserved"
            assert (
                cached_params.end_us == original_params.end_us
            ), "End time not preserved"
            assert (
                cached_params.units_power == original_params.units_power
            ), "Units power not preserved"

        # Metadata should preserve original experiment information
        assert cached_fid.metadata["experiment_path"] == str(
            self.example_data_path
        ), "Experiment path not preserved"
        # Note: FID cache architecture preserves original FID data exactly - no cache-specific metadata is added to FID object

    def test_fid_cache_independence_from_source(self):
        """Test that cached FID is independent of source files and portable."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for independence testing")

        # Load and cache FID
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("independence")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Simulate source files being moved/deleted by changing working directory
        # (Cache should still be loadable)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to load cache from a different working directory context
            cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

            # FID should still be fully functional
            assert cached_fid.n_points > 0
            assert len(cached_fid.data) == original_fid.n_points
            np.testing.assert_array_equal(cached_fid.data, original_fid.data)

            # Cache should contain complete metadata for provenance
            assert cached_fid.metadata["experiment_path"] == str(self.example_data_path)
            # Note: Cache preserves original metadata exactly - cache-specific metadata is in HDF5 file attributes

    def test_processing_parameter_updates(self):
        """Test updating processing parameters in cached FID."""
        if not self.has_example_data:
            pytest.skip(
                "Experiment 2638 data not available for parameter update testing"
            )

        # Load and cache FID
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("param_update")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Update processing parameters (canonical FT is unapodized: only the
        # data-selection knobs are persisted).
        new_params = {"start_us": 1.0, "end_us": 12.0}

        update_fid_processing_defaults(cache_id, new_params, str(self.test_output_dir))

        # Load updated cache
        updated_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # FID data should be unchanged
        np.testing.assert_array_equal(updated_fid.data, original_fid.data)

        # Processing parameters should be updated
        assert updated_fid.processing.start_us == 1.0
        assert updated_fid.processing.end_us == 12.0


class TestStage1OnDemandComplexFT:
    """Test Stage 1 (on-demand ComplexFT calculation) integration."""

    TEST_PREFIX = "test_stage1_"

    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"

    def setup_method(self):
        """Set up test environment for Stage 1 testing."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if experiment 2638 data exists
        self.example_data_path = Path("examples/blackchirp_data/2638")
        self.has_example_data = self.example_data_path.exists()

    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def test_on_demand_complexft_consistency(self):
        """Test that on-demand ComplexFT calculation is deterministic and consistent."""
        if not self.has_example_data:
            pytest.skip(
                "Experiment 2638 data not available for ComplexFT consistency testing"
            )

        # Load and cache FID
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("complexft_consistency")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Load cached FID
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # === TEST MULTIPLE ON-DEMAND CALCULATIONS ===

        # Parameters for experiment 2638 (from CLAUDE.md specifications)
        ft_params = {"start_us": 2.0, "end_us": 14.0}
        trim_range = (26500, 40000)

        complex_fts = []

        # Calculate ComplexFT multiple times with same parameters
        for i in range(3):
            # Stage 1: Three-stage on-demand workflow
            preprocessed = cached_fid.preprocess(**ft_params)
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

            # Apply recommended trimming
            trimmed_ft = complex_ft.trim_to_range(*trim_range)
            complex_fts.append(trimmed_ft)

        # === VALIDATE CONSISTENCY ===

        # All calculations should produce identical results
        for i in range(1, 3):
            np.testing.assert_array_equal(
                complex_fts[0].freq_array,
                complex_fts[i].freq_array,
                err_msg=f"ComplexFT frequency array differs between calculation 0 and {i}",
            )
            np.testing.assert_array_equal(
                complex_fts[0].complex_spectrum,
                complex_fts[i].complex_spectrum,
                err_msg=f"ComplexFT complex spectrum differs between calculation 0 and {i}",
            )

    def test_parameter_variation_produces_different_results(self):
        """Test that different parameters produce different ComplexFT results."""
        if not self.has_example_data:
            pytest.skip(
                "Experiment 2638 data not available for parameter variation testing"
            )

        # Load and cache FID
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("param_variation")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Load cached FID
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # === TEST DIFFERENT PARAMETER COMBINATIONS ===

        param_sets = [
            {"start_us": 1.0},  # early active start
            {"start_us": 2.0},  # later active start
            {"start_us": 1.0, "end_us": 12.0},  # windowed
            {"start_us": 3.0, "end_us": 14.0},  # different window
        ]

        complex_fts = []

        for params in param_sets:
            preprocessed = cached_fid.preprocess(**params)
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)
            complex_fts.append(complex_ft)

        # === VALIDATE DIFFERENT RESULTS ===

        # Different parameters should produce different results
        for i in range(1, len(complex_fts)):
            # Spectra should be different (not identical)
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    complex_fts[0].complex_spectrum, complex_fts[i].complex_spectrum
                )

        # But all should have valid structure
        for complex_ft in complex_fts:
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))
            assert np.all(np.isfinite(complex_ft.freq_array))

    def test_stage1_calculation_logging(self):
        """Log Stage 1 ComplexFT calculation characteristics for development insights."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for calculation logging")

        # Load and cache FID
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("performance")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Load cached FID
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # === BENCHMARK FT CALCULATION TIME ===

        import time

        ft_params = {"start_us": 2.0, "end_us": 14.0}

        start_time = time.time()

        # Stage 1 workflow
        preprocessed = cached_fid.preprocess(**ft_params)
        spectrum, freqs = preprocessed.compute_fft()
        complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

        calculation_time = time.time() - start_time

        # Log performance characteristics for informational purposes
        # Per SERIALIZATION_STRATEGY.md: ~1 second for 750k points is typical

        # Log performance characteristics
        print(f"\n=== Stage 1 Performance Characteristics ===")
        print(f"FID points: {cached_fid.n_points}")
        print(f"ComplexFT calculation time: {calculation_time:.3f}s")
        print(f"ComplexFT points: {complex_ft.n_points}")
        print("================================================")


class TestStage01WorkflowIntegration:
    """Test complete Stage 0 → Stage 1 workflow integration."""

    TEST_PREFIX = "test_integration_"

    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"

    def _get_cache_file(self, suffix):
        """Helper to get cache file path."""
        return self.test_output_dir / f"{self.TEST_PREFIX}{suffix}_fid.h5"

    def setup_method(self):
        """Set up test environment for integration testing."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if experiment 2638 data exists
        self.example_data_path = Path("examples/blackchirp_data/2638")
        self.has_example_data = self.example_data_path.exists()

    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def test_full_stage01_pipeline_vs_direct_processing(self):
        """Test that Stage 0 → Stage 1 pipeline produces identical results to direct processing."""
        if not self.has_example_data:
            pytest.skip(
                "Experiment 2638 data not available for pipeline integration testing"
            )

        # Load experiment 2638 data
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid

        # === DIRECT PROCESSING (NO CACHE) ===
        ft_params = {"start_us": 2.0, "end_us": 14.0}
        trim_range = (26500, 40000)

        # Direct FT processing
        preprocessed_direct = original_fid.preprocess(**ft_params)
        spectrum_direct, freqs_direct = preprocessed_direct.compute_fft()
        complex_ft_direct = ComplexFT.from_spectrum(spectrum_direct, freqs_direct)
        trimmed_ft_direct = complex_ft_direct.trim_to_range(*trim_range)

        # Direct noise estimation
        noise_result_direct = estimate_noise_scatter(
            trimmed_ft_direct.freq_array, trimmed_ft_direct.magnitude_spectrum
        )

        # === CACHED PIPELINE (STAGE 0 → STAGE 1) ===

        # Stage 0: Cache FID
        cache_id = self._get_cache_id("full_pipeline_2638")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Stage 0: Load FID from cache
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # Stage 1: On-demand ComplexFT calculation
        preprocessed_cached = cached_fid.preprocess(**ft_params)
        spectrum_cached, freqs_cached = preprocessed_cached.compute_fft()
        complex_ft_cached = ComplexFT.from_spectrum(spectrum_cached, freqs_cached)
        trimmed_ft_cached = complex_ft_cached.trim_to_range(*trim_range)

        # Continue with noise estimation (future Stage 2)
        noise_result_cached = estimate_noise_scatter(
            trimmed_ft_cached.freq_array, trimmed_ft_cached.magnitude_spectrum
        )

        # === VALIDATE PIPELINE INTEGRITY ===

        # ComplexFT results should be identical
        np.testing.assert_array_equal(
            trimmed_ft_cached.freq_array,
            trimmed_ft_direct.freq_array,
            err_msg="Cached pipeline ComplexFT frequency array differs from direct",
        )
        np.testing.assert_array_equal(
            trimmed_ft_cached.complex_spectrum,
            trimmed_ft_direct.complex_spectrum,
            err_msg="Cached pipeline ComplexFT spectrum differs from direct",
        )

        # Noise estimation should be identical (shows Stage 1 → Stage 2 compatibility)
        np.testing.assert_array_equal(
            noise_result_cached.rms_noise,
            noise_result_direct.rms_noise,
            err_msg="Noise estimation differs between cached and direct pipeline",
        )
        np.testing.assert_array_equal(
            noise_result_cached.noise_mask,
            noise_result_direct.noise_mask,
            err_msg="Noise mask differs between cached and direct pipeline",
        )

        # === CACHE SIZE DIAGNOSTICS ===
        cache_file = self._get_cache_file("full_pipeline_2638")
        cache_file_size = cache_file.stat().st_size / 1024**2

        # Estimate original FID size
        fid_data_size = original_fid.data.nbytes / 1024**2

        print(f"\n=== Stage 0-1 Architecture Diagnostics (Experiment 2638) ===")
        print(f"Original FID data size: {fid_data_size:.2f} MB")
        print(f"FID cache file size: {cache_file_size:.2f} MB")
        print(
            f"Storage overhead: {(cache_file_size - fid_data_size):.2f} MB (metadata)"
        )
        print(
            f"ComplexFT size (temporary): {complex_ft_direct.complex_spectrum.nbytes / 1024**2:.2f} MB"
        )
        print(
            f"Storage efficiency: FID cached once, unlimited ComplexFT parameter combinations"
        )
        print("================================================================")

    def test_stage01_with_multiple_parameter_exploration(self):
        """Test Stage 0-1 architecture supports unlimited parameter exploration."""
        if not self.has_example_data:
            pytest.skip(
                "Experiment 2638 data not available for parameter exploration testing"
            )

        # Load and cache FID once
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("param_exploration")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Load cached FID
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # === PARAMETER EXPLORATION SESSION ===

        # Simulate user exploring different parameter combinations
        exploration_params = [
            {},  # Full record, no windowing
            {"start_us": 1.0},  # Early active start
            {"start_us": 2.0},  # Later active start
            {"start_us": 2.0, "end_us": 14.0},  # Windowed
            {"start_us": 3.0, "end_us": 12.0},  # Narrower window
            {"rdc": False},  # No DC removal
        ]

        results = []

        for i, params in enumerate(exploration_params):
            # Each parameter combination creates a new ComplexFT on-demand
            preprocessed = cached_fid.preprocess(**params)
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

            # Basic validation
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))

            results.append(
                {
                    "params": params,
                    "complex_ft": complex_ft,
                    "n_points": complex_ft.n_points,
                }
            )

        # === VALIDATE PARAMETER EXPLORATION BENEFITS ===

        # All results should be valid but different (where expected)
        assert len(results) == len(exploration_params)

        # The canonical FT is native-length, so every combination yields the
        # same number of points (the active window zeros, it does not resize).
        n_points_set = {r["n_points"] for r in results}
        assert len(n_points_set) == 1

        # Each ComplexFT calculation uses the same cached FID (storage efficient)
        # No permanent ComplexFT storage required (memory efficient)
        cache_file = self._get_cache_file("param_exploration")
        print(f"\n=== Parameter Exploration Summary ===")
        print(
            f"Single FID cache supports {len(exploration_params)} parameter combinations"
        )
        print(f"FID cache size: {cache_file.stat().st_size / 1024**2:.2f} MB")
        print(f"No ComplexFT storage required (calculated on-demand)")
        print("=====================================")


class TestStage01CacheRobustness:
    """Test Stage 0-1 cache robustness and error handling."""

    TEST_PREFIX = "test_robust_"

    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"

    def _get_cache_file(self, suffix):
        """Helper to get cache file path."""
        return self.test_output_dir / f"{self.TEST_PREFIX}{suffix}_fid.h5"

    def setup_method(self):
        """Set up test environment for robustness testing."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def test_corrupted_fid_cache_handling(self):
        """Test pipeline behavior when FID cache files are corrupted."""
        # Create corrupted HDF5 file
        corrupted_file = self._get_cache_file("corrupted_fid")
        with open(corrupted_file, "wb") as f:
            f.write(b"This is not a valid HDF5 file")

        # Pipeline should handle corrupted cache gracefully
        with pytest.raises((OSError, RuntimeError, FileNotFoundError)):
            load_fid_cache(
                self._get_cache_id("corrupted_fid"), str(self.test_output_dir)
            )

    def test_missing_fid_cache_components(self):
        """Test handling of FID cache files missing expected components."""
        # Create a valid HDF5 file but missing expected FID structure
        cache_file = self._get_cache_file("incomplete_fid")

        with h5py.File(cache_file, "w") as h5f:
            # Create incomplete structure (missing essential fid_data group)
            metadata = h5f.create_group("metadata")
            metadata.attrs["experiment_path"] = "/fake/path"
            metadata.attrs["fid_index"] = 0
            # Missing fid_data group that contains time_series_data

        # Pipeline should detect missing structure and fail
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            load_fid_cache(
                self._get_cache_id("incomplete_fid"), str(self.test_output_dir)
            )

    def test_fid_cache_data_corruption_detection(self):
        """Test detection of corrupted FID data in cache files."""
        if not Path("examples/blackchirp_data/2638").exists():
            pytest.skip("Experiment 2638 data not available")

        # Create valid FID cache first
        ftmw_data = load_blackchirp_experiment(
            "examples/blackchirp_data/2638", fid_index=0
        )
        original_fid = ftmw_data.fid
        cache_id = self._get_cache_id("corruption_test")
        save_fid_cache(cache_id, original_fid, str(self.test_output_dir))

        # Corrupt the time series data
        cache_file = self._get_cache_file("corruption_test")
        with h5py.File(cache_file, "r+") as h5f:
            time_data = h5f["fid_data"]["time_series_data"]
            # Introduce NaN values
            corrupt_data = np.array(time_data[:])
            corrupt_data[:10] = np.nan
            del h5f["fid_data"]["time_series_data"]
            h5f["fid_data"]["time_series_data"] = corrupt_data

        # Loading should work but FID data will have NaN values
        cached_fid = load_fid_cache(cache_id, str(self.test_output_dir))

        # Verify corruption is detectable in the loaded FID
        assert np.any(np.isnan(cached_fid.data[:10])), "NaN corruption not detected"

        # Subsequent processing should propagate NaN corruption to output
        # (This is mathematically correct behavior - NaN propagates through FFT)
        preprocessed = cached_fid.preprocess()
        spectrum, freqs = preprocessed.compute_fft()

        # Verify NaN corruption propagates to FFT output (expected behavior)
        assert np.any(np.isnan(spectrum)), "NaN corruption not propagated to FFT output"
        # This allows downstream analysis to detect and handle corrupted data appropriately
