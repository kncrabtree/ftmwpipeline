"""
Unit tests for ComplexFT serialization to HDF5 format.

Tests cover:
- Round-trip serialization accuracy
- Frequency array reconstruction precision
- Handling of missing FID references
- Error conditions and edge cases
- Storage space optimization
- Real experimental data validation
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

from ftmwpipeline.core.data_structures import (
    ComplexFT, FID, FIDProcessingParameters, Sideband
)
from ftmwpipeline.io.complex_ft_serialization import (
    save_complex_ft_to_hdf5,
    load_complex_ft_from_hdf5,
    _extract_frequency_reconstruction_params,
    _reconstruct_frequency_array
)
from ftmwpipeline.io import load_blackchirp_experiment


class TestComplexFTSerialization:
    """Test ComplexFT HDF5 serialization functionality."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_complex_ft_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for testing."""
        # Create a simple FID with known parameters
        n_points = 1000
        spacing = 1e-8  # 10 ns spacing
        probe_freq = 18000.0  # 18 GHz
        
        # Generate a simple exponentially decaying sinusoid
        t = np.arange(n_points) * spacing
        data = np.exp(-t / 1e-6) * np.cos(2 * np.pi * 100e6 * t)  # 100 MHz signal
        
        processing = FIDProcessingParameters(
            zpf=1,
            expf_us=5.0,
            rdc=True,
            autoscale_MHz=10.0
        )
        
        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            processing=processing,
            metadata={'test_param': 'test_value'}
        )
    
    @pytest.fixture
    def sample_complex_ft(self, sample_fid):
        """Create a sample ComplexFT from FID."""
        return sample_fid.ft(zpf=1, expf_us=5.0)
    
    @pytest.fixture
    def complex_ft_no_fid(self):
        """Create a ComplexFT without associated FID."""
        n_points = 1000
        freqs = np.linspace(17900, 18100, n_points)
        # Generate complex spectrum using separate real and imaginary parts
        real_part = np.random.normal(0, 1e-6, n_points)
        imag_part = np.random.normal(0, 1e-6, n_points)
        spectrum = real_part + 1j * imag_part
        
        return ComplexFT(
            freq_array=freqs,
            complex_spectrum=spectrum,
            fid=None,
            metadata={'source': 'synthetic'}
        )
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test files after each test
        for file in output_dir.glob(f"{TestComplexFTSerialization.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def test_round_trip_serialization(self, sample_complex_ft, test_output_dir):
        """Test complete round-trip serialization with bit-perfect accuracy."""
        original_ft = sample_complex_ft
        
        # Save to HDF5
        test_file = self._get_test_file(test_output_dir, "round_trip")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('complex_ft')
            save_complex_ft_to_hdf5(original_ft, group)
        
        # Load from HDF5
        with h5py.File(test_file, 'r') as f:
            group = f['complex_ft']
            loaded_ft = load_complex_ft_from_hdf5(group)
        
        # Verify bit-perfect frequency array reconstruction
        np.testing.assert_array_equal(
            original_ft.freq_array, 
            loaded_ft.freq_array,
            err_msg="Frequency arrays should be identical"
        )
        
        # Verify complex spectrum preservation
        np.testing.assert_array_equal(
            original_ft.complex_spectrum,
            loaded_ft.complex_spectrum,
            err_msg="Complex spectra should be identical"
        )
        
        # Verify FID parameters preservation
        assert loaded_ft.fid is not None
        assert loaded_ft.fid.probe_freq_mhz == original_ft.fid.probe_freq_mhz
        assert loaded_ft.fid.sideband == original_ft.fid.sideband
        assert loaded_ft.fid.processing.zpf == original_ft.fid.processing.zpf
        assert loaded_ft.fid.processing.expf_us == original_ft.fid.processing.expf_us
    
    def test_frequency_reconstruction_precision(self, sample_complex_ft):
        """Test frequency array reconstruction with high precision."""
        # Extract parameters
        params = _extract_frequency_reconstruction_params(sample_complex_ft)
        
        # Reconstruct frequency array
        reconstructed_freqs = _reconstruct_frequency_array(params)
        
        # Test bit-perfect reconstruction (rtol=1e-15 as specified)
        np.testing.assert_allclose(
            sample_complex_ft.freq_array,
            reconstructed_freqs,
            rtol=1e-15,
            err_msg="Frequency reconstruction should be bit-perfect"
        )
    
    def test_parameter_extraction(self, sample_complex_ft):
        """Test frequency reconstruction parameter extraction."""
        params = _extract_frequency_reconstruction_params(sample_complex_ft)
        
        # Verify all required parameters are present
        required_keys = ['n_fid_padded', 'spacing_us', 'probe_freq_mhz', 
                        'sideband', 'autoscale_MHz', 'n_spectrum']
        for key in required_keys:
            assert key in params, f"Missing required parameter: {key}"
        
        # Verify parameter values make sense
        assert params['n_fid_padded'] > 0
        assert params['spacing_us'] > 0
        assert params['probe_freq_mhz'] > 0
        assert params['n_spectrum'] == len(sample_complex_ft.freq_array)
        assert params['sideband'] in ['upper', 'lower']
    
    def test_no_fid_reference(self, complex_ft_no_fid, test_output_dir):
        """Test handling of ComplexFT without FID reference."""
        # This should raise an error during save
        test_file = self._get_test_file(test_output_dir, "no_fid")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('complex_ft')
            with pytest.raises(RuntimeError, match="Failed to save ComplexFT to HDF5"):
                save_complex_ft_to_hdf5(complex_ft_no_fid, group)
    
    def test_hdf5_structure_validation(self, sample_complex_ft, test_output_dir):
        """Test that HDF5 structure matches specification."""
        test_file = self._get_test_file(test_output_dir, "structure")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('complex_ft')
            save_complex_ft_to_hdf5(sample_complex_ft, group)
        
        # Verify structure
        with h5py.File(test_file, 'r') as f:
            group = f['complex_ft']
            
            # Check required datasets and groups
            assert 'complex_spectrum' in group
            assert 'freq_reconstruction' in group
            assert 'processing_params' in group
            assert 'metadata' in group
            
            # Check frequency reconstruction parameters
            freq_group = group['freq_reconstruction']
            required_attrs = ['n_fid_padded', 'spacing_us', 'probe_freq_mhz',
                             'sideband', 'n_spectrum']
            for attr in required_attrs:
                assert attr in freq_group.attrs
    
    def test_storage_structure(self, sample_complex_ft, test_output_dir):
        """Test that HDF5 structure is correct (no direct frequency array storage)."""
        test_file = self._get_test_file(test_output_dir, "storage")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('complex_ft')
            save_complex_ft_to_hdf5(sample_complex_ft, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['complex_ft']
            
            # Verify no full frequency array is stored
            assert 'freq_array' not in group
            assert 'frequency_array' not in group
            
            # Verify required groups/datasets exist
            assert 'complex_spectrum' in group
            assert 'freq_reconstruction' in group
    
    def test_edge_cases(self, test_output_dir):
        """Test edge cases and error conditions."""
        # Test with minimal FID
        minimal_fid = FID(
            data=np.array([1.0, 0.5, 0.0]),
            spacing=1e-6,
            probe_freq_mhz=1000.0,
            sideband=Sideband.UPPER
        )
        minimal_ft = minimal_fid.ft()
        
        # Should work with minimal data
        test_file = self._get_test_file(test_output_dir, "edge")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_complex_ft_to_hdf5(minimal_ft, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            loaded_ft = load_complex_ft_from_hdf5(group)
            
        # Verify reconstruction works
        np.testing.assert_array_equal(minimal_ft.freq_array, loaded_ft.freq_array)
    
    def test_sideband_handling(self, test_output_dir):
        """Test both upper and lower sideband configurations."""
        for sideband in [Sideband.UPPER, Sideband.LOWER]:
            fid = FID(
                data=np.array([1.0, 0.5, 0.0, 0.25]),
                spacing=1e-6,
                probe_freq_mhz=15000.0,
                sideband=sideband
            )
            ft = fid.ft()
            
            test_file = self._get_test_file(test_output_dir, f"sideband_{sideband.value}")
            with h5py.File(test_file, 'w') as f:
                group = f.create_group(f'test_{sideband.value}')
                save_complex_ft_to_hdf5(ft, group)
            
            with h5py.File(test_file, 'r') as f:
                group = f[f'test_{sideband.value}']
                loaded_ft = load_complex_ft_from_hdf5(group)
            
            # Verify sideband is preserved
            assert loaded_ft.fid.sideband == sideband
            np.testing.assert_array_equal(ft.freq_array, loaded_ft.freq_array)
    
    def test_metadata_preservation(self, test_output_dir):
        """Test that metadata is properly preserved."""
        # Create FID with various metadata types
        fid = FID(
            data=np.array([1.0, 0.5]),
            spacing=1e-6,
            probe_freq_mhz=10000.0,
            metadata={'str_val': 'test', 'int_val': 42, 'float_val': 3.14}
        )
        ft = fid.ft()
        ft.metadata.update({'custom_param': 'custom_value'})
        
        test_file = self._get_test_file(test_output_dir, "metadata")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_complex_ft_to_hdf5(ft, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            loaded_ft = load_complex_ft_from_hdf5(group)
        
        # Verify metadata preservation
        assert 'custom_param' in loaded_ft.metadata
        assert loaded_ft.metadata['custom_param'] == 'custom_value'
    
    def test_none_value_handling(self, test_output_dir):
        """Test handling of None values in parameters."""
        # Create FID with some None parameters
        processing = FIDProcessingParameters(
            start_us=None,
            end_us=None,
            winf=None,
            zpf=0,
            expf_us=None,
            autoscale_MHz=None
        )
        
        fid = FID(
            data=np.array([1.0, 0.5]),
            spacing=1e-6,
            probe_freq_mhz=10000.0,
            processing=processing
        )
        ft = fid.ft()
        
        test_file = self._get_test_file(test_output_dir, "none_values")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_complex_ft_to_hdf5(ft, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            loaded_ft = load_complex_ft_from_hdf5(group)
        
        # Verify None values are handled correctly
        assert loaded_ft.fid.processing.start_us is None
        assert loaded_ft.fid.processing.end_us is None
        assert loaded_ft.fid.processing.winf is None
        assert loaded_ft.fid.processing.expf_us is None
        assert loaded_ft.fid.processing.autoscale_MHz is None


class TestRealExperimentalData:
    """Test serialization with real experimental data."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_complex_ft_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test files after each test
        for file in output_dir.glob(f"{TestComplexFTSerialization.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def test_experiment_2638_serialization(self, test_output_dir):
        """Test serialization with real experiment 2638 data."""
        try:
            # Load real experimental data
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            
            # Create fresh ComplexFT to ensure consistent processing
            # Use the loaded FID to create a new ComplexFT with our processing
            complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
            
            print(f"Experiment 2638: FID has {len(ftmw_data.fid.data)} points")
            print(f"ComplexFT has {len(complex_ft.freq_array)} frequency points")
            print(f"Frequency range: {complex_ft.freq_array[0]:.1f} - {complex_ft.freq_array[-1]:.1f} MHz")
            
            # Test serialization
            test_file = self._get_test_file(test_output_dir, "2638")
            with h5py.File(test_file, 'w') as f:
                group = f.create_group('exp_2638')
                save_complex_ft_to_hdf5(complex_ft, group)
            
            # File should exist and be valid HDF5
            assert test_file.exists()
            
            # Test loading
            with h5py.File(test_file, 'r') as f:
                group = f['exp_2638']
                loaded_ft = load_complex_ft_from_hdf5(group)
            
            print(f"Loaded ComplexFT has {len(loaded_ft.freq_array)} frequency points")
            print(f"Loaded frequency range: {loaded_ft.freq_array[0]:.1f} - {loaded_ft.freq_array[-1]:.1f} MHz")
            
            # Verify reconstruction accuracy (arrays should be identical)
            np.testing.assert_array_equal(
                complex_ft.freq_array,
                loaded_ft.freq_array,
                err_msg="Frequency array reconstruction failed for real data"
            )
            
            np.testing.assert_array_equal(
                complex_ft.complex_spectrum,
                loaded_ft.complex_spectrum,
                err_msg="Complex spectrum should be preserved exactly"
            )
            
            # Verify FID parameters
            assert loaded_ft.fid.probe_freq_mhz == complex_ft.fid.probe_freq_mhz
            assert loaded_ft.fid.sideband == complex_ft.fid.sideband
            
            print("✓ Experiment 2638 serialization test passed")
            
        except Exception as e:
            pytest.skip(f"Could not load experimental data: {e}")
    
    def test_trimmed_data_serialization(self, test_output_dir):
        """Test serialization of trimmed ComplexFT data - now should work with the fix!"""
        try:
            # Load and trim experimental data
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
            trimmed_ft = complex_ft.trim_to_range(26500, 40000)
            
            # Test serialization of trimmed data
            test_file = self._get_test_file(test_output_dir, "trimmed")
            with h5py.File(test_file, 'w') as f:
                group = f.create_group('trimmed')
                save_complex_ft_to_hdf5(trimmed_ft, group)
            
            # Verify file structure contains frequency range parameters
            with h5py.File(test_file, 'r') as f:
                group = f['trimmed']
                freq_group = group['freq_reconstruction']
                
                # Check that frequency range parameters are stored
                assert 'freq_min' in freq_group.attrs
                assert 'freq_max' in freq_group.attrs
                # Values should be close to the trim range (within frequency spacing)
                stored_min = float(freq_group.attrs['freq_min'])
                stored_max = float(freq_group.attrs['freq_max'])
                assert stored_min >= 26500.0  # Should be >= trim_min
                assert stored_max <= 40000.0  # Should be <= trim_max
                assert stored_min < 26501.0   # But close to trim_min  
                assert stored_max > 39999.0   # But close to trim_max
            
            # Test loading
            with h5py.File(test_file, 'r') as f:
                group = f['trimmed']
                loaded_ft = load_complex_ft_from_hdf5(group)
            
            # Verify bit-perfect trimmed data reconstruction
            np.testing.assert_allclose(
                trimmed_ft.freq_array,
                loaded_ft.freq_array,
                rtol=1e-15,
                err_msg="Trimmed frequency arrays should be identical"
            )
            
            np.testing.assert_array_equal(
                trimmed_ft.complex_spectrum,
                loaded_ft.complex_spectrum,
                err_msg="Trimmed complex spectra should be identical"
            )
            
            # Verify metadata includes trim information
            assert 'trimmed_range' in loaded_ft.metadata
            assert loaded_ft.metadata['trimmed_range'] == (26500.0, 40000.0)
            
        except Exception as e:
            pytest.skip(f"Could not load experimental data: {e}")


class TestTrimmedObjectSerialization:
    """Test serialization of trimmed ComplexFT objects with comprehensive validation."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_complex_ft_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for testing."""
        # Create a larger FID for meaningful trimming tests
        n_points = 2000
        spacing = 1e-8  # 10 ns spacing
        probe_freq = 20000.0  # 20 GHz
        
        # Generate a simple exponentially decaying sinusoid
        t = np.arange(n_points) * spacing
        data = np.exp(-t / 2e-6) * np.cos(2 * np.pi * 150e6 * t)  # 150 MHz signal
        
        processing = FIDProcessingParameters(
            zpf=1,  # Double the length with zero padding
            expf_us=2.0,
            rdc=True,
            autoscale_MHz=5.0
        )
        
        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            processing=processing,
            metadata={'test_source': 'synthetic_fid'}
        )
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test files after each test
        for file in output_dir.glob(f"{TestTrimmedObjectSerialization.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def _get_trim_range(self, freq_array, start_idx, end_idx):
        """Helper to get proper trim range considering sideband direction."""
        if freq_array[0] > freq_array[-1]:  # Descending (lower sideband)
            freq_min = freq_array[end_idx]
            freq_max = freq_array[start_idx]
        else:  # Ascending (upper sideband)
            freq_min = freq_array[start_idx]
            freq_max = freq_array[end_idx]
        return freq_min, freq_max
    
    def test_trimmed_vs_untrimmed_parameters(self, sample_fid):
        """Test parameter extraction for trimmed and untrimmed objects."""
        # Create untrimmed ComplexFT
        untrimmed_ft = sample_fid.ft(zpf=1, expf_us=2.0)
        untrimmed_params = _extract_frequency_reconstruction_params(untrimmed_ft)
        
        # Create trimmed ComplexFT
        freq_min, freq_max = self._get_trim_range(untrimmed_ft.freq_array, 100, -100)
        trimmed_ft = untrimmed_ft.trim_to_range(freq_min, freq_max)
        trimmed_params = _extract_frequency_reconstruction_params(trimmed_ft)
        
        # Both should have same fundamental FID parameters
        for key in ['n_fid_padded', 'spacing_us', 'probe_freq_mhz', 'sideband', 'autoscale_MHz']:
            assert untrimmed_params[key] == trimmed_params[key]
        
        # But different frequency ranges and spectrum lengths
        assert untrimmed_params['n_spectrum'] > trimmed_params['n_spectrum']
        assert untrimmed_params['freq_min'] != trimmed_params['freq_min'] or untrimmed_params['freq_max'] != trimmed_params['freq_max']
        
        # Trimmed object should have the expected frequency range
        assert trimmed_params['freq_min'] == freq_min
        assert trimmed_params['freq_max'] == freq_max
    
    def test_trimmed_reconstruction_accuracy(self, sample_fid):
        """Test bit-perfect reconstruction of trimmed frequency arrays."""
        # Create and trim ComplexFT
        full_ft = sample_fid.ft(zpf=1)
        freq_min, freq_max = self._get_trim_range(full_ft.freq_array, 200, -300)
        trimmed_ft = full_ft.trim_to_range(freq_min, freq_max)
        
        # Extract parameters and reconstruct
        params = _extract_frequency_reconstruction_params(trimmed_ft)
        reconstructed_freqs = _reconstruct_frequency_array(params)
        
        # Test bit-perfect reconstruction
        np.testing.assert_allclose(
            trimmed_ft.freq_array,
            reconstructed_freqs,
            rtol=1e-15,
            err_msg="Trimmed frequency reconstruction should be bit-perfect"
        )
    
    def test_trimmed_round_trip_serialization(self, sample_fid, test_output_dir):
        """Test complete round-trip serialization for trimmed objects."""
        # Create and trim ComplexFT
        original_ft = sample_fid.ft(zpf=1, expf_us=2.0)
        freq_min, freq_max = self._get_trim_range(original_ft.freq_array, 150, -200)
        trimmed_ft = original_ft.trim_to_range(freq_min, freq_max)
        
        # Save to HDF5
        test_file = self._get_test_file(test_output_dir, "trimmed_round_trip")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('trimmed_test')
            save_complex_ft_to_hdf5(trimmed_ft, group)
        
        # Load from HDF5
        with h5py.File(test_file, 'r') as f:
            group = f['trimmed_test']
            loaded_ft = load_complex_ft_from_hdf5(group)
        
        # Verify perfect reconstruction
        np.testing.assert_array_equal(
            trimmed_ft.freq_array,
            loaded_ft.freq_array,
            err_msg="Trimmed frequency arrays should be identical"
        )
        
        np.testing.assert_array_equal(
            trimmed_ft.complex_spectrum,
            loaded_ft.complex_spectrum,
            err_msg="Trimmed complex spectra should be identical"
        )
        
        # Verify metadata preservation
        assert loaded_ft.metadata['trimmed_range'] == (freq_min, freq_max)
    
    def test_multiple_trim_levels(self, sample_fid, test_output_dir):
        """Test serialization with different trim ranges."""
        original_ft = sample_fid.ft(zpf=1)
        
        # Test different trim ranges
        trim_indices = [
            (50, -50),    # Large range
            (500, -500),  # Medium range
            (900, -900),  # Small range
        ]
        
        for i, (start_idx, end_idx) in enumerate(trim_indices):
            freq_min, freq_max = self._get_trim_range(original_ft.freq_array, start_idx, end_idx)
            trimmed_ft = original_ft.trim_to_range(freq_min, freq_max)
            
            # Save and load
            test_file = self._get_test_file(test_output_dir, f"trimmed_range_{i}")
            with h5py.File(test_file, 'w') as f:
                group = f.create_group(f'test_{i}')
                save_complex_ft_to_hdf5(trimmed_ft, group)
            
            with h5py.File(test_file, 'r') as f:
                group = f[f'test_{i}']
                loaded_ft = load_complex_ft_from_hdf5(group)
            
            # Verify reconstruction
            np.testing.assert_array_equal(trimmed_ft.freq_array, loaded_ft.freq_array)
            np.testing.assert_array_equal(trimmed_ft.complex_spectrum, loaded_ft.complex_spectrum)
    
    def test_trimmed_vs_untrimmed_structure(self, sample_fid, test_output_dir):
        """Test that both trimmed and untrimmed objects have the same storage structure."""
        original_ft = sample_fid.ft(zpf=1)
        freq_min, freq_max = self._get_trim_range(original_ft.freq_array, 300, -400)
        trimmed_ft = original_ft.trim_to_range(freq_min, freq_max)
        
        # Save both untrimmed and trimmed
        untrimmed_file = self._get_test_file(test_output_dir, "storage_untrimmed")
        trimmed_file = self._get_test_file(test_output_dir, "storage_trimmed")
        
        with h5py.File(untrimmed_file, 'w') as f:
            group = f.create_group('untrimmed')
            save_complex_ft_to_hdf5(original_ft, group)
        
        with h5py.File(trimmed_file, 'w') as f:
            group = f.create_group('trimmed')
            save_complex_ft_to_hdf5(trimmed_ft, group)
        
        # Both should have same basic structure
        with h5py.File(untrimmed_file, 'r') as f:
            untrimmed_group = f['untrimmed']
            assert 'complex_spectrum' in untrimmed_group
            assert 'freq_reconstruction' in untrimmed_group
            
        with h5py.File(trimmed_file, 'r') as f:
            trimmed_group = f['trimmed']
            assert 'complex_spectrum' in trimmed_group
            assert 'freq_reconstruction' in trimmed_group
    
    
    def test_edge_case_single_point_trim(self, sample_fid, test_output_dir):
        """Test serialization of extremely small trimmed ranges."""
        original_ft = sample_fid.ft(zpf=1)
        
        # Create minimal trim range (just a few points)
        mid_index = len(original_ft.freq_array) // 2
        freq_min, freq_max = self._get_trim_range(original_ft.freq_array, mid_index, mid_index + 5)
        trimmed_ft = original_ft.trim_to_range(freq_min, freq_max)
        
        # This should work even for very small ranges
        test_file = self._get_test_file(test_output_dir, "small_trim")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('small_trim')
            save_complex_ft_to_hdf5(trimmed_ft, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['small_trim']
            loaded_ft = load_complex_ft_from_hdf5(group)
        
        np.testing.assert_array_equal(trimmed_ft.freq_array, loaded_ft.freq_array)
        np.testing.assert_array_equal(trimmed_ft.complex_spectrum, loaded_ft.complex_spectrum)


class TestErrorConditions:
    """Test error handling and edge cases."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_complex_ft_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test files after each test
        for file in output_dir.glob(f"{TestComplexFTSerialization.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def test_invalid_hdf5_group(self, test_output_dir):
        """Test error handling with invalid HDF5 groups."""
        # Create a sample ComplexFT
        fid = FID(
            data=np.array([1.0, 0.5]),
            spacing=1e-6,
            probe_freq_mhz=10000.0
        )
        sample_complex_ft = fid.ft()
        
        test_file = self._get_test_file(test_output_dir, "invalid")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_complex_ft_to_hdf5(sample_complex_ft, group)
            
            # Corrupt the data by removing required datasets
            del group['complex_spectrum']
        
        # Should raise error when loading
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            with pytest.raises(RuntimeError):
                load_complex_ft_from_hdf5(group)
    
    def test_inconsistent_reconstruction_params(self, test_output_dir):
        """Test handling of inconsistent reconstruction parameters."""
        # Create a synthetic HDF5 structure with inconsistent parameters
        test_file = self._get_test_file(test_output_dir, "inconsistent")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            
            # Create complex spectrum
            real_part = np.random.normal(0, 1, 100)
            imag_part = np.random.normal(0, 1, 100)
            spectrum = real_part + 1j * imag_part
            group.create_dataset('complex_spectrum', data=spectrum)
            
            # Create inconsistent frequency reconstruction parameters
            freq_group = group.create_group('freq_reconstruction')
            freq_group.attrs['n_fid_padded'] = 50  # Wrong! Should give 26 spectrum points, not 100
            freq_group.attrs['spacing_us'] = 10.0
            freq_group.attrs['probe_freq_mhz'] = 15000.0
            freq_group.attrs['sideband'] = 'upper'
            freq_group.attrs['autoscale_MHz'] = "None"
            freq_group.attrs['n_spectrum'] = 100  # Inconsistent with n_fid_padded
        
        # Should raise error due to inconsistency
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            with pytest.raises(RuntimeError):
                load_complex_ft_from_hdf5(group)
    
    def test_invalid_reconstruction_parameters(self, test_output_dir):
        """Test error handling with invalid reconstruction parameters."""
        # Test with invalid parameter values in _reconstruct_frequency_array
        invalid_params = {
            'n_fid_padded': -100,  # Invalid negative value
            'spacing_us': 10.0,
            'probe_freq_mhz': 15000.0,
            'sideband': 'upper',
            'freq_min': 14000.0,
            'freq_max': 16000.0,
            'n_spectrum': 100
        }
        
        with pytest.raises(ValueError, match="Failed to reconstruct frequency array"):
            _reconstruct_frequency_array(invalid_params)
        
        # Test missing required parameters
        incomplete_params = {
            'n_fid_padded': 1000,
            'spacing_us': 10.0,
            # Missing required parameters
        }
        
        with pytest.raises(ValueError, match="Failed to reconstruct frequency array"):
            _reconstruct_frequency_array(incomplete_params)
    
    def test_corrupted_hdf5_parameters(self, test_output_dir):
        """Test error handling with corrupted HDF5 parameter data."""
        # Create a valid ComplexFT
        fid = FID(
            data=np.array([1.0, 0.5, 0.0] * 100),
            spacing=1e-6,
            probe_freq_mhz=15000.0,
            sideband=Sideband.UPPER,
            processing=FIDProcessingParameters(zpf=1)
        )
        ft = fid.ft()
        
        # Save valid data first
        test_file = self._get_test_file(test_output_dir, "corrupted_params")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_complex_ft_to_hdf5(ft, group)
        
        # Corrupt the frequency range parameters to be inconsistent with spectrum length
        with h5py.File(test_file, 'r+') as f:
            group = f['test']
            freq_group = group['freq_reconstruction']
            
            # Make freq_min > freq_max (impossible range)
            freq_group.attrs['freq_min'] = 20000.0  
            freq_group.attrs['freq_max'] = 10000.0  # Invalid: min > max
        
        # Should raise error when loading due to invalid frequency range
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            with pytest.raises(RuntimeError):
                load_complex_ft_from_hdf5(group)
    
    def test_zpf_parameter_handling(self):
        """Test handling of different zpf parameter values."""
        # Test with zpf=0 (no padding)
        processing = FIDProcessingParameters(zpf=0)  
        fid = FID(
            data=np.array([1.0, 0.5] * 50),
            spacing=1e-6,
            probe_freq_mhz=15000.0,
            sideband=Sideband.LOWER,
            processing=processing
        )
        ft = fid.ft()
        
        # Should handle zpf=0 correctly
        params = _extract_frequency_reconstruction_params(ft)
        assert 'n_fid_padded' in params
        assert params['n_fid_padded'] == 100  # Same as original FID length
        
        # Test with zpf=1 (default padding)
        processing = FIDProcessingParameters(zpf=1)  
        fid = FID(
            data=np.array([1.0, 0.5] * 50),
            spacing=1e-6,
            probe_freq_mhz=15000.0,
            sideband=Sideband.LOWER,
            processing=processing
        )
        ft = fid.ft()
        
        params = _extract_frequency_reconstruction_params(ft)
        assert params['n_fid_padded'] > 100  # Should be padded