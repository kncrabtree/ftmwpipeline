"""
Pipeline integration tests focusing on cache system with experiment 2638 data.

These tests validate pipeline-level cache integration including:
- Data integrity between cached vs non-cached pipeline phases
- Real experiment 2638 data processing pipeline
- Cache consistency across multiple pipeline runs
- Cache failure handling (corrupted cache, missing components, etc.)

Individual component serialization is tested separately in unit tests.
Test files are written to tests/output/ with proper cleanup.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch

from ftmwpipeline.io.result_serialization import (
    save_pipeline_cache,
    load_pipeline_cache,
    save_stage_result,
    load_stage_result,
    get_cache_info,
    clear_cache
)
from ftmwpipeline.io.experimental_formats import load_blackchirp_experiment
from ftmwpipeline.core.data_structures import ComplexFT
from ftmwpipeline.preprocessing.noise_estimation import NoiseResult, estimate_noise_adaptive


class TestPipelineCacheIntegration:
    """Test pipeline-level cache integration with real experiment data."""
    
    # Test cache prefix for consistent naming and cleanup  
    TEST_PREFIX = "test_pipeline_"
    
    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"
    
    def _get_cache_file(self, suffix):
        """Helper to get cache file path."""
        return self.test_output_dir / f"{self.TEST_PREFIX}{suffix}_cache.h5"
    
    def setup_method(self):
        """Set up test environment for pipeline integration tests."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if experiment 2638 data exists
        self.example_data_path = Path("examples/blackchirp_data/2638")
        self.has_example_data = self.example_data_path.exists()
    
    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*_cache.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
    
    def test_full_pipeline_cached_vs_noncached_consistency(self):
        """Test full pipeline with caching produces identical results to non-cached pipeline."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for pipeline integration testing")
        
        # Load real experiment 2638 data following CLAUDE.md specifications
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        
        # === NON-CACHED PIPELINE ===
        complex_ft_direct = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
        trimmed_ft_direct = complex_ft_direct.trim_to_range(26500, 40000)
        noise_result_direct = estimate_noise_adaptive(
            trimmed_ft_direct.freq_array,
            trimmed_ft_direct.magnitude_spectrum
        )
        
        # === CACHED PIPELINE ===
        # Stage 1: Save ComplexFT to cache
        save_pipeline_cache(
            self._get_cache_id("integration_2638"),
            complex_ft=trimmed_ft_direct,
            noise_result=None,
            cache_dir=str(self.test_output_dir)
        )
        
        # Stage 2: Load from cache and continue pipeline
        cache_data = load_pipeline_cache(self._get_cache_id("integration_2638"), cache_dir=str(self.test_output_dir))
        cached_complex_ft = cache_data["complex_ft"]
        
        # Compute NoiseResult from cached data
        noise_result_cached = estimate_noise_adaptive(
            cached_complex_ft.freq_array,
            cached_complex_ft.magnitude_spectrum
        )
        
        # === PIPELINE INTEGRITY VALIDATION ===
        # ComplexFT should be identical
        np.testing.assert_array_equal(
            cached_complex_ft.freq_array,
            trimmed_ft_direct.freq_array,
            err_msg="Cached ComplexFT frequency array differs from direct"
        )
        np.testing.assert_array_equal(
            cached_complex_ft.complex_spectrum,
            trimmed_ft_direct.complex_spectrum,
            err_msg="Cached ComplexFT spectrum differs from direct"
        )
        
        # NoiseResult should be identical (bit-perfect)
        np.testing.assert_array_equal(
            noise_result_cached.rms_noise,
            noise_result_direct.rms_noise,
            err_msg="Noise estimation differs between cached and direct pipeline"
        )
        np.testing.assert_array_equal(
            noise_result_cached.noise_mask,
            noise_result_direct.noise_mask,
            err_msg="Noise mask differs between cached and direct pipeline"
        )
        
        # === CACHE SIZE DIAGNOSTICS ===
        cache_info = get_cache_info(self._get_cache_id("integration_2638"), cache_dir=str(self.test_output_dir))
        cache_file_size = self._get_cache_file("integration_2638").stat().st_size / 1024**2
        
        # Estimate original data size
        complex_spectrum_size = trimmed_ft_direct.complex_spectrum.nbytes / 1024**2
        freq_array_size = trimmed_ft_direct.freq_array.nbytes / 1024**2
        original_size = complex_spectrum_size + freq_array_size
        
        print(f"\n=== Cache Size Diagnostics (Experiment 2638) ===")
        print(f"Original data size: {original_size:.2f} MB")
        print(f"- ComplexFT spectrum: {complex_spectrum_size:.2f} MB")
        print(f"- Frequency array: {freq_array_size:.2f} MB")
        print(f"Cache file size: {cache_file_size:.2f} MB")
        print(f"Compression ratio: {original_size/cache_file_size:.2f}x")
        print("================================================")
    
    def test_partial_cache_pipeline_integration(self):
        """Test pipeline behavior with partial cache hits using real experiment data."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for pipeline integration testing")
        
        # Load real experiment data
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
        trimmed_ft = complex_ft.trim_to_range(26500, 40000)
        
        # Save only ComplexFT (no NoiseResult) to test partial cache
        save_pipeline_cache(
            self._get_cache_id("partial_test_2638"),
            complex_ft=trimmed_ft,
            noise_result=None,
            cache_dir=str(self.test_output_dir)
        )
        
        # Load cache and compute NoiseResult fresh
        cache_data = load_pipeline_cache(self._get_cache_id("partial_test_2638"), cache_dir=str(self.test_output_dir))
        cached_complex_ft = cache_data["complex_ft"]
        
        # Compute NoiseResult from cached ComplexFT
        noise_result_from_cache = estimate_noise_adaptive(
            cached_complex_ft.freq_array,
            cached_complex_ft.magnitude_spectrum
        )
        
        # Compute NoiseResult directly (no cache)
        noise_result_direct = estimate_noise_adaptive(
            trimmed_ft.freq_array,
            trimmed_ft.magnitude_spectrum  
        )
        
        # Pipeline integrity: cached vs direct should be identical
        np.testing.assert_array_equal(
            noise_result_from_cache.rms_noise, 
            noise_result_direct.rms_noise
        )
        np.testing.assert_array_equal(
            noise_result_from_cache.noise_mask,
            noise_result_direct.noise_mask
        )
    
    def test_cache_consistency_across_multiple_runs(self):
        """Test cache consistency when running pipeline multiple times."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for consistency testing")
        
        # Load experiment data
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        
        results = []
        for run in range(3):  # Multiple pipeline runs
            complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
            trimmed_ft = complex_ft.trim_to_range(26500, 40000)
            noise_result = estimate_noise_adaptive(
                trimmed_ft.freq_array,
                trimmed_ft.magnitude_spectrum
            )
            
            # Save complete pipeline cache
            save_pipeline_cache(
                self._get_cache_id(f"consistency_run_{run}"),
                complex_ft=trimmed_ft,
                noise_result=noise_result,
                cache_dir=str(self.test_output_dir)
            )
            
            # Load and verify
            loaded = load_pipeline_cache(self._get_cache_id(f"consistency_run_{run}"), cache_dir=str(self.test_output_dir))
            results.append(loaded)
        
        # All runs should produce identical results (deterministic processing)
        for i in range(1, 3):
            np.testing.assert_array_equal(
                results[0]["complex_ft"].complex_spectrum,
                results[i]["complex_ft"].complex_spectrum,
                err_msg=f"ComplexFT differs between run 0 and run {i}"
            )
            np.testing.assert_array_equal(
                results[0]["noise_result"].rms_noise,
                results[i]["noise_result"].rms_noise,
                err_msg=f"NoiseResult differs between run 0 and run {i}"
            )
    
    def test_cache_management_and_info(self):
        """Test cache management functions in pipeline context."""
        if not self.has_example_data:
            pytest.skip("Experiment 2638 data not available for cache management testing")
        
        # Create pipeline cache
        ftmw_data = load_blackchirp_experiment(str(self.example_data_path), fid_index=0)
        complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0).trim_to_range(26500, 40000)
        noise_result = estimate_noise_adaptive(complex_ft.freq_array, complex_ft.magnitude_spectrum)
        
        # Save main pipeline cache
        save_pipeline_cache(
            self._get_cache_id("management_test"),
            complex_ft=complex_ft,
            noise_result=noise_result,
            cache_dir=str(self.test_output_dir)
        )
        
        # Test cache info
        info = get_cache_info(self._get_cache_id("management_test"), cache_dir=str(self.test_output_dir))
        # Verify cache info contains expected structure
        assert "has_complex_ft" in info
        assert "has_noise_result" in info
        assert info["has_complex_ft"] is True
        assert info["has_noise_result"] is True
        
        # Test cache clearing
        cache_file = self._get_cache_file("management_test")
        assert cache_file.exists()
        
        clear_cache(self._get_cache_id("management_test"), cache_dir=str(self.test_output_dir))
        assert not cache_file.exists()


class TestPipelineCacheEdgeCases:
    """Test pipeline cache edge cases and robustness."""
    
    # Test cache prefix for consistent naming and cleanup
    TEST_PREFIX = "test_edge_"
    
    def _get_cache_id(self, suffix):
        """Helper to generate consistent cache IDs."""
        return f"{self.TEST_PREFIX}{suffix}"
    
    def _get_cache_file(self, suffix):
        """Helper to get cache file path."""
        return self.test_output_dir / f"{self.TEST_PREFIX}{suffix}_cache.h5"
    
    def setup_method(self):
        """Set up test environment for edge case testing."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
    def teardown_method(self):
        """Clean up test files."""
        if self.test_output_dir.exists():
            for cache_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*_cache.h5"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
    
    def test_corrupted_cache_handling(self):
        """Test pipeline behavior when cache files are corrupted."""
        # Create corrupted HDF5 file
        corrupted_file = self._get_cache_file("corrupted")
        with open(corrupted_file, 'wb') as f:
            f.write(b"This is not a valid HDF5 file")
        
        # Pipeline should handle corrupted cache gracefully
        with pytest.raises(RuntimeError, match="Failed to load pipeline cache"):
            load_pipeline_cache(self._get_cache_id("corrupted"), cache_dir=str(self.test_output_dir))
    
    def test_missing_cache_components(self):
        """Test handling of cache files missing expected pipeline components."""
        # Create a valid HDF5 file but missing expected pipeline structure
        cache_file = self._get_cache_file("incomplete")
        
        with h5py.File(cache_file, 'w') as h5f:
            # Create incomplete structure (missing complex_ft group)
            h5f.create_group('noise_result')
            pipeline_info = h5f.create_group('pipeline_info')
            pipeline_info.attrs['experiment_id'] = self._get_cache_id("incomplete")
            pipeline_info.attrs['has_complex_ft'] = True  # Lie - it's missing
            pipeline_info.attrs['has_noise_result'] = False
        
        # Pipeline should detect inconsistency and fail
        with pytest.raises(ValueError, match="missing 'complex_ft' group"):
            load_pipeline_cache(self._get_cache_id("incomplete"), cache_dir=str(self.test_output_dir))
    
    def test_cache_invalidation_scenarios(self):
        """Test cache invalidation when data corruption is detected."""
        if not Path("examples/blackchirp_data/2638").exists():
            pytest.skip("Experiment 2638 data not available")
        
        # Create valid cache first
        ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
        complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0).trim_to_range(26500, 40000)
        
        save_pipeline_cache(
            self._get_cache_id("invalidation"),
            complex_ft=complex_ft,
            noise_result=None,
            cache_dir=str(self.test_output_dir)
        )
        
        # Corrupt the complex spectrum data
        cache_file = self._get_cache_file("invalidation")
        with h5py.File(cache_file, 'r+') as h5f:
            spectrum_data = h5f['complex_ft']['complex_spectrum']
            # Introduce NaN values
            corrupt_spectrum = np.array(spectrum_data[:])
            corrupt_spectrum[:10] = np.nan
            del h5f['complex_ft']['complex_spectrum']
            h5f['complex_ft']['complex_spectrum'] = corrupt_spectrum
        
        # Loading should detect corruption (warning issued, but may not raise)
        # The corruption detection issues a warning rather than raising
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_pipeline_cache(self._get_cache_id("invalidation"), cache_dir=str(self.test_output_dir))
            # Verify corruption warning was issued
            assert len(w) > 0
            assert "checksum mismatch" in str(w[0].message).lower()
    
    def test_concurrent_cache_access_simulation(self):
        """Simulate concurrent cache access scenarios."""
        if not Path("examples/blackchirp_data/2638").exists():
            pytest.skip("Experiment 2638 data not available")
        
        # Test that cache loading is robust to concurrent access attempts
        ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
        complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0).trim_to_range(26500, 40000)
        
        save_pipeline_cache(
            self._get_cache_id("concurrent"),
            complex_ft=complex_ft,
            noise_result=None,
            cache_dir=str(self.test_output_dir)
        )
        
        # Simulate multiple concurrent loads (simplified test)
        results = []
        for _ in range(5):
            loaded = load_pipeline_cache(self._get_cache_id("concurrent"), cache_dir=str(self.test_output_dir))
            results.append(loaded["complex_ft"])
        
        # All concurrent loads should return identical data
        for i in range(1, 5):
            np.testing.assert_array_equal(
                results[0].complex_spectrum,
                results[i].complex_spectrum,
                err_msg=f"Concurrent load {i} differs from load 0"
            )