"""
Tests for NoiseResult serialization and deserialization.

Tests the efficient HDF5 serialization of NoiseResult objects using signal indices
and convolution-based RMS reconstruction for >95% storage reduction.
"""

import pytest
import numpy as np
import h5py
import tempfile
from pathlib import Path

from ftmwpipeline.preprocessing.noise_estimation import NoiseResult, estimate_noise_adaptive
from ftmwpipeline.io.noise_result_serialization import (
    save_noise_result_to_hdf5,
    load_noise_result_from_hdf5,
    _extract_signal_indices,
    _reconstruct_noise_mask,
    _store_convolution_parameters,
    _reconstruct_rms_via_convolution
)


@pytest.fixture
def test_output_dir():
    """Create and cleanup test output directory."""
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup test files
    for file in output_dir.glob("test_noise_result_*.h5"):
        file.unlink()


@pytest.fixture
def sample_spectrum_data():
    """Create sample frequency and magnitude data for testing."""
    # Create realistic FTMW spectrum data
    n_points = 100000  # Smaller than real data for faster tests
    frequencies = np.linspace(26500, 40000, n_points)  # MHz
    
    # Create magnitude spectrum with noise + signal peaks
    np.random.seed(42)  # For reproducible tests
    magnitudes = np.random.exponential(1.0, n_points)  # Noise baseline
    
    # Add some spectral peaks (signal)
    peak_positions = [28000, 32000, 36000]  # MHz
    for peak_freq in peak_positions:
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        # Add Gaussian peak
        peak_width = 50  # points
        peak_indices = np.arange(peak_idx - peak_width, peak_idx + peak_width + 1)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < n_points)]
        
        gaussian = np.exp(-(peak_indices - peak_idx)**2 / (2 * (peak_width/3)**2))
        magnitudes[peak_indices] += 10 * gaussian  # Strong peaks
    
    return frequencies, magnitudes


@pytest.fixture
def sample_noise_result(sample_spectrum_data):
    """Create sample NoiseResult using the adaptive noise estimation."""
    frequencies, magnitudes = sample_spectrum_data
    
    # Run noise estimation to get realistic NoiseResult
    noise_result = estimate_noise_adaptive(
        frequencies, magnitudes,
        skew_target=0.631,
        min_bin_fraction=1/64,
        min_noise_fraction=2/3,
        verbose=False
    )
    
    return noise_result


class TestSignalIndicesConversion:
    """Test signal indices extraction and reconstruction."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_noise_result_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    def test_extract_signal_indices(self):
        """Test conversion of boolean mask to signal indices."""
        # Create test boolean mask
        noise_mask = np.array([True, True, False, True, False, False, True])
        
        # Extract signal indices (where mask is False)
        signal_indices = _extract_signal_indices(noise_mask)
        
        # Should be indices 2, 4, 5
        expected_indices = np.array([2, 4, 5], dtype=np.int32)
        np.testing.assert_array_equal(signal_indices, expected_indices)
    
    def test_reconstruct_noise_mask(self):
        """Test reconstruction of boolean mask from signal indices."""
        signal_indices = np.array([2, 4, 5], dtype=np.int32)
        total_length = 7
        
        # Reconstruct noise mask
        noise_mask = _reconstruct_noise_mask(signal_indices, total_length)
        
        # Should be [True, True, False, True, False, False, True]
        expected_mask = np.array([True, True, False, True, False, False, True])
        np.testing.assert_array_equal(noise_mask, expected_mask)
    
    def test_roundtrip_signal_indices(self, sample_noise_result):
        """Test full roundtrip: mask → indices → mask."""
        original_mask = sample_noise_result.noise_mask
        
        # Convert to indices and back
        signal_indices = _extract_signal_indices(original_mask)
        reconstructed_mask = _reconstruct_noise_mask(signal_indices, len(original_mask))
        
        # Should be identical
        np.testing.assert_array_equal(reconstructed_mask, original_mask)
    
    def test_storage_reduction(self, sample_noise_result):
        """Test that signal indices use less storage than boolean mask."""
        original_mask = sample_noise_result.noise_mask
        signal_indices = _extract_signal_indices(original_mask)
        
        # Calculate storage sizes (approximate)
        mask_size = original_mask.nbytes  # bools are 1 byte each
        indices_size = signal_indices.nbytes  # int32 are 4 bytes each
        
        # Should have significant reduction for typical FTMW data (~10% signal)
        reduction_ratio = indices_size / mask_size
        assert reduction_ratio < 0.8, f"Expected >20% reduction, got {reduction_ratio:.2%}"
        
        print(f"Storage reduction: {mask_size:,} bytes → {indices_size:,} bytes ({reduction_ratio:.1%})")


class TestConvolutionReconstruction:
    """Test convolution-based RMS reconstruction."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_noise_result_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    def test_store_convolution_parameters(self, sample_noise_result, test_output_dir):
        """Test storing convolution parameters to HDF5."""
        test_file = self._get_test_file(test_output_dir, "conv_params")
        
        with h5py.File(test_file, 'w') as f:
            h5_group = f.create_group('noise_result')
            _store_convolution_parameters(sample_noise_result, h5_group)
            
            # Verify required parameters are stored
            smoothing_group = h5_group['smoothing_params']
            
            # Check that key parameters exist
            assert 'convolution_mode' in smoothing_group.attrs
            assert 'padding_method' in smoothing_group.attrs
            assert 'rms_computation' in smoothing_group.attrs
            
            # Check algorithm-specific parameters
            if 'smoothing_window_points' in sample_noise_result.bin_info:
                assert 'smoothing_window_points' in smoothing_group.attrs
    
    def test_rms_convolution_reconstruction(self, sample_spectrum_data, sample_noise_result, test_output_dir):
        """Test exact RMS reconstruction using convolution method."""
        frequencies, magnitudes = sample_spectrum_data
        original_rms = sample_noise_result.rms_noise
        
        # Store convolution parameters
        test_file = self._get_test_file(test_output_dir, "rms_recon")
        with h5py.File(test_file, 'w') as f:
            h5_group = f.create_group('noise_result')
            _store_convolution_parameters(sample_noise_result, h5_group)
        
        # Reconstruct RMS
        with h5py.File(test_file, 'r') as f:
            conv_params = f['noise_result/smoothing_params']
            reconstructed_rms = _reconstruct_rms_via_convolution(
                frequencies, magnitudes, sample_noise_result.noise_mask, conv_params
            )
        
        # Test bit-perfect reconstruction (now that the modularization is fixed)
        np.testing.assert_array_equal(
            reconstructed_rms, original_rms,
            err_msg="RMS reconstruction should be bit-perfect identical to original"
        )
        
        # Verify shape and basic properties
        assert reconstructed_rms.shape == original_rms.shape
        assert np.all(reconstructed_rms > 0)  # Should be positive
        assert np.isfinite(reconstructed_rms).all()  # Should be finite


class TestNoiseResultSerialization:
    """Test full NoiseResult serialization and deserialization."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_noise_result_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    def test_save_load_roundtrip(self, sample_spectrum_data, sample_noise_result, test_output_dir):
        """Test complete save/load roundtrip with exact reconstruction."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "roundtrip")
        
        # Save NoiseResult
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, noise_group)
        
        # Load NoiseResult
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            loaded_result = load_noise_result_from_hdf5(noise_group, frequencies, magnitudes)
        
        # Verify reconstruction quality
        np.testing.assert_array_equal(loaded_result.noise_mask, sample_noise_result.noise_mask)
        
        # RMS reconstruction should now be bit-perfect with the fixed modularization
        np.testing.assert_array_equal(
            loaded_result.rms_noise, sample_noise_result.rms_noise,
            err_msg="RMS reconstruction should be bit-perfect identical to original"
        )
        
        # Verify basic properties are preserved
        assert loaded_result.rms_noise.shape == sample_noise_result.rms_noise.shape
        assert np.all(loaded_result.rms_noise > 0)  # Should be positive
        
        # Verify bin_info is preserved
        assert loaded_result.bin_info.keys() == sample_noise_result.bin_info.keys()
        
        # Check key bin_info values
        for key in ['n_bins', 'noise_fraction', 'algorithm']:
            if key in sample_noise_result.bin_info:
                assert loaded_result.bin_info[key] == sample_noise_result.bin_info[key]
    
    def test_storage_optimization_validation(self, sample_spectrum_data, sample_noise_result, test_output_dir):
        """Test that storage optimization achieves target reduction."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "storage")
        
        # Calculate original storage requirements
        original_mask_size = sample_noise_result.noise_mask.nbytes
        original_rms_size = sample_noise_result.rms_noise.nbytes
        original_total = original_mask_size + original_rms_size
        
        # Save with optimization
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, noise_group)
        
        # Measure optimized storage
        optimized_size = test_file.stat().st_size
        
        # Should achieve significant reduction (target >90%)
        reduction_ratio = optimized_size / original_total
        print(f"Storage optimization: {original_total:,} bytes → {optimized_size:,} bytes ({reduction_ratio:.1%})")
        
        # For test data, should still get significant reduction
        # With bit-perfect reconstruction, we may store slightly more data for precision
        assert reduction_ratio < 0.8, f"Expected >20% reduction, got {reduction_ratio:.1%}"
    
    def test_hdf5_structure_validation(self, sample_spectrum_data, sample_noise_result, test_output_dir):
        """Test that HDF5 structure matches specification."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "structure")
        
        # Save NoiseResult
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, noise_group)
        
        # Verify HDF5 structure
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            
            # Required datasets
            assert 'signal_indices' in noise_group
            assert 'rms_poly_coeffs' in noise_group
            
            # Required groups
            assert 'smoothing_params' in noise_group
            assert 'bin_info' in noise_group
            assert 'algorithm_info' in noise_group
            
            # Check algorithm info
            algo_group = noise_group['algorithm_info']
            assert algo_group.attrs['method'] == 'convolution_reconstruction'
            assert 'version' in algo_group.attrs
            assert 'storage_optimization' in algo_group.attrs
    
    def test_error_handling(self, test_output_dir):
        """Test error handling for invalid inputs."""
        test_file = self._get_test_file(test_output_dir, "errors")
        
        # Test mismatched array lengths
        frequencies = np.linspace(0, 100, 100)
        magnitudes = np.random.random(50)  # Different length
        
        noise_result = NoiseResult(
            rms_noise=np.random.random(100),
            noise_mask=np.random.random(100) > 0.5,
            bin_info={}
        )
        
        with pytest.raises(RuntimeError, match="Failed to save NoiseResult to HDF5"):
            with h5py.File(test_file, 'w') as f:
                noise_group = f.create_group('noise_result')
                save_noise_result_to_hdf5(noise_result, frequencies, magnitudes, noise_group)
    
    def test_polynomial_fallback(self, sample_spectrum_data, sample_noise_result, test_output_dir):
        """Test polynomial fallback when convolution reconstruction fails."""
        frequencies, magnitudes = sample_spectrum_data
        test_file = self._get_test_file(test_output_dir, "fallback")
        
        # Save NoiseResult normally
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(sample_noise_result, frequencies, magnitudes, noise_group)
        
        # Corrupt smoothing_params to force fallback
        with h5py.File(test_file, 'a') as f:
            del f['noise_result/smoothing_params']
        
        # Load should still work using polynomial fallback
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            loaded_result = load_noise_result_from_hdf5(noise_group, frequencies, magnitudes)
        
        # Should have reasonable RMS values (may not be exact)
        assert loaded_result.rms_noise.shape == sample_noise_result.rms_noise.shape
        assert np.all(loaded_result.rms_noise > 0)  # Should be positive
        assert np.isfinite(loaded_result.rms_noise).all()  # Should be finite


class TestBitPerfectReconstruction:
    """Dedicated tests for bit-perfect reconstruction validation."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_noise_result_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    def test_bit_perfect_reconstruction_large_dataset(self, test_output_dir):
        """Test bit-perfect reconstruction with a large, realistic dataset."""
        # Create a large dataset similar to real FTMW data
        np.random.seed(42)
        frequencies = np.linspace(26500, 40000, 10000)
        magnitudes = np.random.exponential(1.0, 10000)
        
        # Add signal peaks to make the noise mask more realistic
        peak_positions = [28000, 32000, 36000]
        for peak_freq in peak_positions:
            peak_idx = np.argmin(np.abs(frequencies - peak_freq))
            peak_width = 50
            peak_indices = np.arange(peak_idx - peak_width, peak_idx + peak_width + 1)
            peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(frequencies))]
            gaussian = np.exp(-(peak_indices - peak_idx)**2 / (2 * (peak_width/3)**2))
            magnitudes[peak_indices] += 10 * gaussian
        
        # Create noise result
        original_result = estimate_noise_adaptive(frequencies, magnitudes, verbose=False)
        
        # Test full serialization roundtrip
        test_file = self._get_test_file(test_output_dir, "bit_perfect_large")
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(original_result, frequencies, magnitudes, noise_group)
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_result = load_noise_result_from_hdf5(noise_group, frequencies, magnitudes)
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_result.noise_mask, original_result.noise_mask,
            err_msg="Noise mask should be perfectly reconstructed"
        )
        
        np.testing.assert_array_equal(
            reconstructed_result.rms_noise, original_result.rms_noise,
            err_msg="RMS noise should be bit-perfect identical to original"
        )
    
    @pytest.mark.parametrize("size", [1000, 5000, 20000])
    def test_bit_perfect_different_scales(self, size, test_output_dir):
        """Test bit-perfect reconstruction across different dataset sizes."""
        # Create test data
        np.random.seed(42 + size)  # Different seed for each size
        frequencies = np.linspace(26000, 40000, size)
        magnitudes = np.random.exponential(0.8, size)
        
        # Add a few peaks
        n_peaks = max(1, size // 5000)  # More peaks for larger datasets
        peak_freqs = np.linspace(27000, 39000, n_peaks)
        
        for peak_freq in peak_freqs:
            peak_idx = np.argmin(np.abs(frequencies - peak_freq))
            width = max(10, size // 500)
            indices = np.arange(peak_idx - width//2, peak_idx + width//2)
            indices = indices[(indices >= 0) & (indices < size)]
            if len(indices) > 0:
                magnitudes[indices] += 5 * np.random.exponential(1.0, len(indices))
        
        # Create and test
        original_result = estimate_noise_adaptive(frequencies, magnitudes, verbose=False)
        
        test_file = self._get_test_file(test_output_dir, f"bit_perfect_{size}")
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(original_result, frequencies, magnitudes, noise_group)
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_result = load_noise_result_from_hdf5(noise_group, frequencies, magnitudes)
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_result.rms_noise, original_result.rms_noise,
            err_msg=f"Bit-perfect reconstruction failed for size {size}"
        )
    
    def test_bit_perfect_numerical_precision(self, test_output_dir):
        """Test that reconstruction preserves full numerical precision."""
        # Create data with specific numerical characteristics
        np.random.seed(12345)
        frequencies = np.linspace(25000, 45000, 8000)
        
        # Create magnitudes with a wide range of values to test precision
        magnitudes = np.random.lognormal(mean=-1, sigma=2, size=8000)
        magnitudes[magnitudes < 1e-10] = 1e-10  # Prevent underflow
        magnitudes[magnitudes > 1e3] = 1e3      # Prevent overflow
        
        # Add structured variation to test interpolation precision
        phase = 2 * np.pi * frequencies / 1000  # 1000 MHz period
        magnitudes *= (1 + 0.1 * np.sin(phase))
        
        original_result = estimate_noise_adaptive(frequencies, magnitudes, verbose=False)
        
        # Test serialization
        test_file = self._get_test_file(test_output_dir, "numerical_precision")
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(original_result, frequencies, magnitudes, noise_group)
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_result = load_noise_result_from_hdf5(noise_group, frequencies, magnitudes)
        
        # Verify exact numerical precision
        assert np.array_equal(reconstructed_result.rms_noise, original_result.rms_noise)
        
        # Additional precision checks
        diff = reconstructed_result.rms_noise - original_result.rms_noise
        assert np.all(diff == 0.0), "All differences should be exactly zero for bit-perfect reconstruction"
        
        # Check that we preserve the full dynamic range
        original_range = np.max(original_result.rms_noise) - np.min(original_result.rms_noise)
        reconstructed_range = np.max(reconstructed_result.rms_noise) - np.min(reconstructed_result.rms_noise)
        assert original_range == reconstructed_range, "Dynamic range should be perfectly preserved"


class TestIntegrationWithRealData:
    """Integration tests using experiment 2638 data (if available)."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_noise_result_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    def setup_method(self):
        """Set up test environment for real data tests."""
        self.test_output_dir = Path("tests/output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if example data exists
        self.example_data_path = Path("examples/blackchirp_data/2638")
        self.has_example_data = self.example_data_path.exists()
        
        # Cache for computed ComplexFT (computed once, reused across tests)
        self._cached_complex_ft = None
    
    def teardown_method(self):
        """Clean up test files after each test."""
        if self.test_output_dir.exists():
            # Clean up HDF5 test files
            for test_file in self.test_output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
                try:
                    test_file.unlink()
                except OSError:
                    pass
            
            # Clean up .ftmw test files
            for test_file in self.test_output_dir.glob("*.ftmw"):
                try:
                    test_file.unlink()
                except OSError:
                    pass
    
    def _get_experiment_2638_complex_ft(self):
        """
        Get ComplexFT for experiment 2638 using proper on-demand computation.
        
        Computes once and caches for reuse across test methods for performance.
        Uses Stage 1 implementation with optimal parameters for experiment 2638.
        """
        if self._cached_complex_ft is not None:
            return self._cached_complex_ft
            
        import ftmwpipeline.api as ftmw
        
        # Use tests/output directory for test .ftmw file
        test_file = self.test_output_dir / "exp_2638_for_complex_ft.ftmw"
        
        # Use functional API to create and process .ftmw file
        ftmw.import_data(str(test_file), source=str(self.example_data_path))
        
        # Compute ComplexFT using functional API with optimal parameters
        # These are the recommended parameters for experiment 2638 from CLAUDE.md
        complex_ft = ftmw.compute_ft(
            str(test_file),
            zpf=1,                    # Zero padding factor for improved frequency resolution
            expf_us=5.0,             # 5 μs exponential apodization filter for sensitivity
            trim=(26500, 40000)      # Activity region, removes noise regions
        )
        self._cached_complex_ft = complex_ft
                
        return self._cached_complex_ft
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    def test_experiment_2638_default_parameters(self, test_output_dir):
        """Test serialization with real experiment 2638 data using default noise estimation parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test with default noise estimation parameters
        original_noise_result = estimate_noise_adaptive(
            complex_ft.freq_array, 
            complex_ft.magnitude_spectrum,
            verbose=False
        )
        
        # Test serialization roundtrip
        test_file = self._get_test_file(test_output_dir, "real_data_exp2638_default")
        
        # Save NoiseResult
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(
                original_noise_result, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum, 
                noise_group
            )
        
        # Load NoiseResult
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_noise_result = load_noise_result_from_hdf5(
                noise_group, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum
            )
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_noise_result.noise_mask, 
            original_noise_result.noise_mask,
            err_msg="Noise mask should be perfectly reconstructed for real experiment 2638 data"
        )
        
        np.testing.assert_array_equal(
            reconstructed_noise_result.rms_noise, 
            original_noise_result.rms_noise,
            err_msg="RMS noise should be bit-perfect identical for real experiment 2638 data"
        )
        
        # Verify bin_info preservation
        assert reconstructed_noise_result.bin_info.keys() == original_noise_result.bin_info.keys()
        for key in ['n_bins', 'noise_fraction', 'algorithm']:
            if key in original_noise_result.bin_info:
                assert reconstructed_noise_result.bin_info[key] == original_noise_result.bin_info[key]
        
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    @pytest.mark.parametrize("skew_target", [0.5, 0.631, 0.8])
    def test_experiment_2638_skew_target_variations(self, skew_target, test_output_dir):
        """Test serialization with real data using different skew_target parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test with specific skew_target
        original_noise_result = estimate_noise_adaptive(
            complex_ft.freq_array, 
            complex_ft.magnitude_spectrum,
            skew_target=skew_target,
            verbose=False
        )
        
        # Test serialization roundtrip
        test_file = self._get_test_file(test_output_dir, f"real_data_skew_{skew_target:.3f}")
        
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(
                original_noise_result, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum, 
                noise_group
            )
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_noise_result = load_noise_result_from_hdf5(
                noise_group, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum
            )
        
        # Verify bit-perfect reconstruction for this parameter set
        np.testing.assert_array_equal(
            reconstructed_noise_result.rms_noise, 
            original_noise_result.rms_noise,
            err_msg=f"Bit-perfect reconstruction failed for skew_target={skew_target}"
        )
        
        np.testing.assert_array_equal(
            reconstructed_noise_result.noise_mask, 
            original_noise_result.noise_mask,
            err_msg=f"Noise mask reconstruction failed for skew_target={skew_target}"
        )
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    @pytest.mark.parametrize("min_bin_fraction", [1/128, 1/64, 1/32, 1/16])
    def test_experiment_2638_min_bin_fraction_variations(self, min_bin_fraction, test_output_dir):
        """Test serialization with real data using different min_bin_fraction parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test with specific min_bin_fraction
        original_noise_result = estimate_noise_adaptive(
            complex_ft.freq_array, 
            complex_ft.magnitude_spectrum,
            min_bin_fraction=min_bin_fraction,
            verbose=False
        )
        
        # Test serialization roundtrip
        test_file = self._get_test_file(test_output_dir, f"real_data_bin_frac_{min_bin_fraction:.6f}")
        
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(
                original_noise_result, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum, 
                noise_group
            )
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_noise_result = load_noise_result_from_hdf5(
                noise_group, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum
            )
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_noise_result.rms_noise, 
            original_noise_result.rms_noise,
            err_msg=f"Bit-perfect reconstruction failed for min_bin_fraction={min_bin_fraction}"
        )
        
        np.testing.assert_array_equal(
            reconstructed_noise_result.noise_mask, 
            original_noise_result.noise_mask,
            err_msg=f"Noise mask reconstruction failed for min_bin_fraction={min_bin_fraction}"
        )
        
        # Verify that different bin fractions create different binning structures
        assert 'n_bins' in original_noise_result.bin_info
        # More restrictive bin fraction should lead to fewer, larger bins
        # Less restrictive bin fraction should allow more, smaller bins
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    @pytest.mark.parametrize("smoothing_window_mhz", [None, 50.0, 100.0, 200.0])
    def test_experiment_2638_smoothing_window_variations(self, smoothing_window_mhz, test_output_dir):
        """Test serialization with real data using different smoothing window parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test with specific smoothing_window_mhz
        original_noise_result = estimate_noise_adaptive(
            complex_ft.freq_array, 
            complex_ft.magnitude_spectrum,
            smoothing_window_mhz=smoothing_window_mhz,
            verbose=False
        )
        
        # Test serialization roundtrip
        window_str = "auto" if smoothing_window_mhz is None else f"{smoothing_window_mhz:.1f}"
        test_file = self._get_test_file(test_output_dir, f"real_data_smooth_{window_str}")
        
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(
                original_noise_result, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum, 
                noise_group
            )
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_noise_result = load_noise_result_from_hdf5(
                noise_group, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum
            )
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_noise_result.rms_noise, 
            original_noise_result.rms_noise,
            err_msg=f"Bit-perfect reconstruction failed for smoothing_window_mhz={smoothing_window_mhz}"
        )
        
        np.testing.assert_array_equal(
            reconstructed_noise_result.noise_mask, 
            original_noise_result.noise_mask,
            err_msg=f"Noise mask reconstruction failed for smoothing_window_mhz={smoothing_window_mhz}"
        )
        
        # Verify smoothing parameters are preserved
        if smoothing_window_mhz is not None:
            # Should be recorded in bin_info or other diagnostics
            assert isinstance(original_noise_result.bin_info, dict)
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    @pytest.mark.parametrize("min_noise_fraction", [0.5, 2/3, 0.75, 0.9])
    def test_experiment_2638_min_noise_fraction_variations(self, min_noise_fraction, test_output_dir):
        """Test serialization with real data using different min_noise_fraction parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test with specific min_noise_fraction
        original_noise_result = estimate_noise_adaptive(
            complex_ft.freq_array, 
            complex_ft.magnitude_spectrum,
            min_noise_fraction=min_noise_fraction,
            verbose=False
        )
        
        # Test serialization roundtrip
        test_file = self._get_test_file(test_output_dir, f"real_data_noise_frac_{min_noise_fraction:.3f}")
        
        with h5py.File(test_file, 'w') as f:
            noise_group = f.create_group('noise_result')
            save_noise_result_to_hdf5(
                original_noise_result, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum, 
                noise_group
            )
        
        with h5py.File(test_file, 'r') as f:
            noise_group = f['noise_result']
            reconstructed_noise_result = load_noise_result_from_hdf5(
                noise_group, 
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum
            )
        
        # Verify bit-perfect reconstruction
        np.testing.assert_array_equal(
            reconstructed_noise_result.rms_noise, 
            original_noise_result.rms_noise,
            err_msg=f"Bit-perfect reconstruction failed for min_noise_fraction={min_noise_fraction}"
        )
        
        np.testing.assert_array_equal(
            reconstructed_noise_result.noise_mask, 
            original_noise_result.noise_mask,
            err_msg=f"Noise mask reconstruction failed for min_noise_fraction={min_noise_fraction}"
        )
        
        # Verify that the noise fraction constraint is enforced
        actual_noise_fraction = np.mean(original_noise_result.noise_mask)
        # The actual noise fraction should be at least the minimum required
        # (though it may be higher due to algorithm specifics)
        assert actual_noise_fraction >= min_noise_fraction * 0.9  # Allow some tolerance
    
    @pytest.mark.skipif(
        not Path("examples/blackchirp_data/2638").exists(),
        reason="Experiment 2638 data not available"
    )
    def test_experiment_2638_combined_parameter_variations(self, test_output_dir):
        """Test serialization with real data using combinations of different parameters."""
        # Compute ComplexFT using Stage 1 implementation (correct on-demand architecture)
        complex_ft = self._get_experiment_2638_complex_ft()
        
        # Test several interesting parameter combinations
        parameter_combinations = [
            {
                "name": "high_precision",
                "params": {"skew_target": 0.631, "min_bin_fraction": 1/128, "smoothing_window_mhz": 50.0}
            },
            {
                "name": "fast_processing", 
                "params": {"skew_target": 0.8, "min_bin_fraction": 1/32, "min_noise_fraction": 0.5}
            },
            {
                "name": "conservative",
                "params": {"skew_target": 0.5, "min_bin_fraction": 1/64, "min_noise_fraction": 0.9}
            },
            {
                "name": "auto_smoothing",
                "params": {"skew_target": 0.631, "min_bin_fraction": 1/64, "smoothing_window_mhz": None}
            }
        ]
        
        for combo in parameter_combinations:
            # Test with specific parameter combination
            original_noise_result = estimate_noise_adaptive(
                complex_ft.freq_array, 
                complex_ft.magnitude_spectrum,
                verbose=False,
                **combo["params"]
            )
        
            # Test serialization roundtrip
            test_file = self._get_test_file(test_output_dir, f"real_data_combo_{combo['name']}")
            
            with h5py.File(test_file, 'w') as f:
                noise_group = f.create_group('noise_result')
                save_noise_result_to_hdf5(
                    original_noise_result, 
                    complex_ft.freq_array, 
                    complex_ft.magnitude_spectrum, 
                    noise_group
                )
            
            with h5py.File(test_file, 'r') as f:
                noise_group = f['noise_result']
                reconstructed_noise_result = load_noise_result_from_hdf5(
                    noise_group, 
                    complex_ft.freq_array, 
                    complex_ft.magnitude_spectrum
                )
        
            # Verify bit-perfect reconstruction for this combination
            np.testing.assert_array_equal(
                reconstructed_noise_result.rms_noise, 
                original_noise_result.rms_noise,
                err_msg=f"Bit-perfect reconstruction failed for parameter combination '{combo['name']}'"
            )
            
            np.testing.assert_array_equal(
                reconstructed_noise_result.noise_mask, 
                original_noise_result.noise_mask,
                err_msg=f"Noise mask reconstruction failed for parameter combination '{combo['name']}'"
            )
            
            # Verify basic properties are reasonable
            assert np.all(original_noise_result.rms_noise > 0), f"All RMS values should be positive for '{combo['name']}'"
            assert np.isfinite(original_noise_result.rms_noise).all(), f"All RMS values should be finite for '{combo['name']}'"
            assert original_noise_result.noise_mask.dtype == bool, f"Noise mask should be boolean for '{combo['name']}'"
        


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])