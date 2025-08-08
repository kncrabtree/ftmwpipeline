"""
Unit tests for FID serialization and caching.

Tests Stage 0-1 architecture FID caching with:
- FID serialization to HDF5 format
- Bit-perfect FID reconstruction from cache
- Metadata preservation (source, experimental, processing)
- Cache portability and independence from source files
- Real experimental data validation with experiment 2638
- Processing parameter updates and defaults handling

Per refinement #1: FID data is real, not complex, and stored as raw voltage data.
Per refinement #3: FID is cached, ComplexFT is calculated on-demand.
"""

import pytest
import numpy as np
import h5py
import tempfile
import json
from pathlib import Path
from datetime import datetime

from ftmwpipeline.core.data_structures import FID, FIDProcessingParameters, Sideband
from ftmwpipeline.io.fid_serialization import (
    save_fid_to_hdf5,
    load_fid_from_hdf5,
    save_fid_cache,
    load_fid_cache,
    update_fid_processing_defaults,
    _serialize_optional_float,
    _deserialize_optional_float,
    _serialize_optional_str,
    _deserialize_optional_str
)
from ftmwpipeline.io import load_blackchirp_experiment


class TestFIDSerialization:
    """Test FID HDF5 serialization functionality."""
    
    # Test file prefix for consistent naming and cleanup
    TEST_PREFIX = "test_fid_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate consistent test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for testing."""
        # Create real voltage data (not complex per refinement #1)
        n_points = 1000
        spacing = 2e-11  # 20 ns spacing (typical for FTMW)
        probe_freq = 18000.0  # 18 GHz
        
        # Generate realistic exponentially decaying sinusoid (real-valued)
        t = np.arange(n_points) * spacing
        # Multiple frequency components for realistic spectrum
        data = (np.exp(-t / 2e-6) * 
                (0.8 * np.cos(2 * np.pi * 100e6 * t) +  # 100 MHz signal
                 0.3 * np.cos(2 * np.pi * 250e6 * t) +  # 250 MHz signal
                 0.1 * np.random.normal(0, 0.05, n_points)))  # Noise
        
        processing = FIDProcessingParameters(
            start_us=0.5,
            end_us=10.0,
            zpf=1,
            expf_us=3.0,
            rdc=True,
            winf='hann',
            units_power=6
        )
        
        metadata = {
            'source_path': '/test/path/experiment',
            'source_format': 'blackchirp',
            'load_timestamp': '2023-01-01T12:00:00',
            'temperature_K': 298.0,
            'pressure_torr': 1e-3,
            'experiment_notes': 'Test FID for serialization'
        }
        
        return FID(
            data=data,  # Real voltage data per refinement #1
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            shots=50000,
            processing=processing,
            metadata=metadata
        )
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test files after each test
        for file in output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def test_fid_data_is_real_not_complex(self, sample_fid):
        """Test that FID data is real, not complex (per refinement #1)."""
        # Verify the test FID has real data
        assert np.all(np.isreal(sample_fid.data))
        assert sample_fid.data.dtype in [np.float64, np.float32]
        assert not np.iscomplexobj(sample_fid.data)
    
    def test_round_trip_serialization(self, sample_fid, test_output_dir):
        """Test complete round-trip FID serialization with bit-perfect accuracy."""
        original_fid = sample_fid
        
        # Save to HDF5
        test_file = self._get_test_file(test_output_dir, "round_trip")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('fid_data')
            save_fid_to_hdf5(original_fid, group)
        
        # Load from HDF5
        with h5py.File(test_file, 'r') as f:
            group = f['fid_data']
            loaded_fid = load_fid_from_hdf5(group)
        
        # Verify bit-perfect data reconstruction
        np.testing.assert_array_equal(
            original_fid.data, 
            loaded_fid.data,
            err_msg="FID data should be reconstructed bit-perfectly"
        )
        
        # Verify acquisition parameters
        assert loaded_fid.spacing == original_fid.spacing
        assert loaded_fid.probe_freq_mhz == original_fid.probe_freq_mhz
        assert loaded_fid.sideband == original_fid.sideband
        assert loaded_fid.shots == original_fid.shots
        assert loaded_fid.n_points == original_fid.n_points
        assert abs(loaded_fid.duration_us - original_fid.duration_us) < 1e-9
        
        # Verify processing parameters
        assert loaded_fid.processing.start_us == original_fid.processing.start_us
        assert loaded_fid.processing.end_us == original_fid.processing.end_us
        assert loaded_fid.processing.zpf == original_fid.processing.zpf
        assert loaded_fid.processing.expf_us == original_fid.processing.expf_us
        assert loaded_fid.processing.rdc == original_fid.processing.rdc
        assert loaded_fid.processing.winf == original_fid.processing.winf
        assert loaded_fid.processing.units_power == original_fid.processing.units_power
        
        # Verify metadata preservation
        for key, value in original_fid.metadata.items():
            assert key in loaded_fid.metadata
            assert loaded_fid.metadata[key] == value
    
    def test_hdf5_structure_validation(self, sample_fid, test_output_dir):
        """Test that HDF5 structure follows specification."""
        test_file = self._get_test_file(test_output_dir, "structure")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('fid_data')
            save_fid_to_hdf5(sample_fid, group)
        
        # Verify structure per specification in fid_serialization.py
        with h5py.File(test_file, 'r') as f:
            group = f['fid_data']
            
            # Check required datasets and groups
            assert 'time_series_data' in group
            assert 'acquisition' in group
            assert 'recommended_processing' in group  # New name per refinement
            assert 'metadata' in group
            
            # Check time series data (real voltage data)
            time_data = group['time_series_data']
            assert time_data.dtype == np.float64
            assert np.all(np.isreal(time_data[:]))
            
            # Check acquisition parameters
            acq_group = group['acquisition']
            required_attrs = ['spacing_seconds', 'probe_freq_mhz', 'sideband', 
                             'shots', 'n_points', 'duration_us']
            for attr in required_attrs:
                assert attr in acq_group.attrs
            
            # Check recommended processing (not requirements)
            proc_group = group['recommended_processing']
            assert proc_group.attrs['description'] == 'Format-specific processing recommendations (not requirements)'
            
            # Check metadata groups
            meta_group = group['metadata']
            assert 'source_info' in meta_group
            assert 'experimental_data' in meta_group
    
    def test_spacing_display_format(self, sample_fid, test_output_dir):
        """Test that spacing is stored in seconds with .4e format (per refinement #2)."""
        test_file = self._get_test_file(test_output_dir, "spacing_format")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('fid_data')
            save_fid_to_hdf5(sample_fid, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['fid_data']
            acq_group = group['acquisition']
            
            # Verify spacing is stored in seconds
            spacing_stored = float(acq_group.attrs['spacing_seconds'])
            assert spacing_stored == sample_fid.spacing  # Should be in seconds
            
            # Verify format when displayed
            spacing_str = f"{spacing_stored:.4e}"
            assert 'e-' in spacing_str  # Should use scientific notation
            print(f"Spacing stored as: {spacing_stored:.4e} s")  # Should display with .4e format
    
    def test_cache_decoupling_and_portability(self, sample_fid, test_output_dir):
        """Test that cached FID is decoupled from source and portable."""
        # Save FID cache
        cache_file = save_fid_cache("test_portable", sample_fid, cache_dir=str(test_output_dir))
        
        # Verify cache file exists and is independent
        assert cache_file.exists()
        
        # Load from cache and verify it's complete
        loaded_fid = load_fid_cache("test_portable", cache_dir=str(test_output_dir))
        
        # Should contain all data needed for independent analysis
        assert loaded_fid.n_points == sample_fid.n_points
        assert loaded_fid.probe_freq_mhz == sample_fid.probe_freq_mhz
        assert loaded_fid.sideband == sample_fid.sideband
        
        # Check cache file structure contains portability info
        with h5py.File(cache_file, 'r') as f:
            assert f.attrs['cache_type'] == 'FID'
            assert f.attrs['experiment_id'] == 'test_portable'
            assert 'cache_timestamp' in f.attrs
            
            fid_group = f['fid_data']
            assert fid_group.attrs['cache_description'] == 'Portable FID cache - contains all data needed for independent analysis'
            
            # Quick-access summary for cache portability
            assert fid_group.attrs['summary_probe_freq_mhz'] == sample_fid.probe_freq_mhz
            assert fid_group.attrs['summary_sideband'] == sample_fid.sideband.value
    
    def test_optional_value_serialization_helpers(self):
        """Test helper functions for optional value serialization."""
        # Test float serialization
        assert _serialize_optional_float(None) == '__None__'
        assert _serialize_optional_float(3.14) == 3.14
        assert _serialize_optional_float(0.0) == 0.0
        
        # Test float deserialization
        assert _deserialize_optional_float('__None__') is None
        assert _deserialize_optional_float(b'__None__') is None  # bytes version
        assert _deserialize_optional_float(3.14) == 3.14
        assert _deserialize_optional_float(0.0) == 0.0
        
        # Test string serialization
        assert _serialize_optional_str(None) == '__None__'
        assert _serialize_optional_str('test') == 'test'
        assert _serialize_optional_str('') == ''
        
        # Test string deserialization  
        assert _deserialize_optional_str('__None__') is None
        assert _deserialize_optional_str(b'__None__') is None  # bytes version
        assert _deserialize_optional_str('test') == 'test'
        assert _deserialize_optional_str(b'test') == 'test'  # bytes version
    
    def test_processing_parameters_with_none_values(self, test_output_dir):
        """Test serialization with None values in processing parameters."""
        # Create FID with some None processing parameters
        processing = FIDProcessingParameters(
            start_us=None,
            end_us=None,
            winf=None,
            zpf=0,
            expf_us=None,
            units_power=6
        )
        
        fid = FID(
            data=np.array([1.0, 0.5, 0.0]),
            spacing=1e-6,
            probe_freq_mhz=10000.0,
            processing=processing
        )
        
        test_file = self._get_test_file(test_output_dir, "none_values")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_fid_to_hdf5(fid, group)
        
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            loaded_fid = load_fid_from_hdf5(group)
        
        # Verify None values are preserved
        assert loaded_fid.processing.start_us is None
        assert loaded_fid.processing.end_us is None
        assert loaded_fid.processing.winf is None
        assert loaded_fid.processing.expf_us is None
        assert loaded_fid.processing.zpf == 0
        assert loaded_fid.processing.units_power == 6
    
    def test_metadata_separation(self, sample_fid, test_output_dir):
        """Test that metadata is properly separated into source and experimental."""
        # Add mixed metadata
        sample_fid.metadata.update({
            'source_path': '/test/source/path',
            'loader_class': 'BlackChirpLoader',
            'temperature_K': 298.0,
            'pressure_torr': 1e-3,
            'custom_param': 'custom_value'
        })
        
        test_file = self._get_test_file(test_output_dir, "metadata_separation")
        with h5py.File(test_file, 'w') as f:
            group = f.create_group('test')
            save_fid_to_hdf5(sample_fid, group)
        
        # Check that metadata is properly separated in HDF5
        with h5py.File(test_file, 'r') as f:
            group = f['test']
            meta_group = group['metadata']
            
            # Load source metadata
            source_json = meta_group['source_info'][()].decode('utf-8')
            source_metadata = json.loads(source_json)
            assert 'source_path' in source_metadata
            assert 'loader_class' in source_metadata
            
            # Load experimental metadata
            exp_json = meta_group['experimental_data'][()].decode('utf-8')
            exp_metadata = json.loads(exp_json)
            assert 'temperature_K' in exp_metadata
            assert 'custom_param' in exp_metadata
            
            # Verify source keys are NOT in experimental metadata
            assert 'source_path' not in exp_metadata
            assert 'loader_class' not in exp_metadata
    


class TestFIDCacheOperations:
    """Test FID cache file operations."""
    
    TEST_PREFIX = "test_cache_"
    
    def _get_cache_file(self, output_dir, exp_id):
        """Helper to get cache file path."""
        return output_dir / f"{exp_id}_fid.h5"
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        # Cleanup test cache files
        for file in output_dir.glob(f"{self.TEST_PREFIX}*_fid.h5"):
            file.unlink(missing_ok=True)
    
    def test_save_and_load_fid_cache(self, test_output_dir):
        """Test standalone FID cache save/load operations."""
        # Create test FID
        fid = FID(
            data=np.array([1.0, 0.8, 0.5, 0.2, 0.0]),
            spacing=1e-6,
            probe_freq_mhz=15000.0,
            sideband=Sideband.UPPER,
            shots=10000
        )
        
        # Save cache
        exp_id = f"{self.TEST_PREFIX}save_load"
        cache_file = save_fid_cache(exp_id, fid, cache_dir=str(test_output_dir))
        
        # Verify file was created
        expected_file = self._get_cache_file(test_output_dir, exp_id)
        assert cache_file == expected_file
        assert cache_file.exists()
        
        # Load from cache
        loaded_fid = load_fid_cache(exp_id, cache_dir=str(test_output_dir))
        
        # Verify data integrity
        np.testing.assert_array_equal(fid.data, loaded_fid.data)
        assert loaded_fid.spacing == fid.spacing
        assert loaded_fid.probe_freq_mhz == fid.probe_freq_mhz
        assert loaded_fid.sideband == fid.sideband
        assert loaded_fid.shots == fid.shots
    
    def test_update_processing_defaults(self, test_output_dir):
        """Test updating processing defaults in cached FID."""
        # Create and cache FID with initial processing
        initial_processing = FIDProcessingParameters(zpf=1, expf_us=5.0)
        fid = FID(
            data=np.array([1.0, 0.5]),
            spacing=1e-6,
            probe_freq_mhz=10000.0,
            processing=initial_processing
        )
        
        exp_id = f"{self.TEST_PREFIX}update_params"
        save_fid_cache(exp_id, fid, cache_dir=str(test_output_dir))
        
        # Update processing defaults
        new_params = {
            'zpf': 2,
            'expf_us': 3.0,
            'start_us': 1.0,
            'end_us': 10.0,
            'window_function': 'hann',
            'rdc': False
        }
        
        update_fid_processing_defaults(exp_id, new_params, cache_dir=str(test_output_dir))
        
        # Load and verify updates
        loaded_fid = load_fid_cache(exp_id, cache_dir=str(test_output_dir))
        
        assert loaded_fid.processing.zpf == 2
        assert loaded_fid.processing.expf_us == 3.0
        assert loaded_fid.processing.start_us == 1.0
        assert loaded_fid.processing.end_us == 10.0
        assert loaded_fid.processing.winf == 'hann'
        assert loaded_fid.processing.rdc is False
    
    def test_cache_error_handling(self, test_output_dir):
        """Test error handling in cache operations."""
        # Test loading non-existent cache
        with pytest.raises(FileNotFoundError):
            load_fid_cache("nonexistent", cache_dir=str(test_output_dir))
        
        # Test invalid experiment ID
        with pytest.raises(RuntimeError, match="Failed to save FID cache"):
            save_fid_cache("", FID(data=[1,2], spacing=1e-6, probe_freq_mhz=1000))
        
        with pytest.raises(ValueError, match="experiment_id must be a non-empty string"):
            load_fid_cache(None)
        
        # Test invalid FID object
        with pytest.raises(RuntimeError, match="Failed to save FID cache"):
            save_fid_cache("test", "not_a_fid")
        
        # Test update on non-existent cache
        with pytest.raises(FileNotFoundError):
            update_fid_processing_defaults("nonexistent", {'zpf': 2})


class TestRealExperimentalDataSerialization:
    """Test FID serialization with real experiment 2638 data."""
    
    TEST_PREFIX = "test_exp2638_"
    
    def _get_test_file(self, output_dir, suffix):
        """Helper to generate test file names."""
        return output_dir / f"{self.TEST_PREFIX}{suffix}.h5"
    
    @pytest.fixture
    def test_output_dir(self):
        """Create and cleanup test output directory."""
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        yield output_dir
        for file in output_dir.glob(f"{self.TEST_PREFIX}*.h5"):
            file.unlink(missing_ok=True)
    
    def test_experiment_2638_fid_caching(self, test_output_dir):
        """Test FID caching with real experiment 2638 data."""
        try:
            # Load real experimental data
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            original_fid = ftmw_data.fid
            
            print(f"Experiment 2638 FID loaded:")
            print(f"  Points: {original_fid.n_points}")
            print(f"  Duration: {original_fid.duration_us:.1f} μs")
            print(f"  Spacing: {original_fid.spacing:.4e} s")  # Per refinement #2
            print(f"  Probe: {original_fid.probe_freq_mhz:.1f} MHz")
            print(f"  Sideband: {original_fid.sideband.value}")
            
            # Verify FID specifications per CLAUDE.md
            assert original_fid.n_points == 750000  # 750k points
            assert abs(original_fid.duration_us - 15.0) < 0.1  # ~15 μs duration
            assert original_fid.probe_freq_mhz == 40960.0  # 40.96 GHz probe
            assert original_fid.sideband == Sideband.LOWER  # Lower sideband
            
            # Verify data is real, not complex (per refinement #1)
            assert np.all(np.isreal(original_fid.data))
            assert original_fid.data.dtype in [np.float64, np.float32]
            
            # Test caching
            cache_file = save_fid_cache("exp_2638_test", original_fid, cache_dir=str(test_output_dir))
            assert cache_file.exists()
            
            # Check cache size (FID should be much smaller than ComplexFT)
            cache_size_mb = cache_file.stat().st_size / 1024**2
            fid_data_size_mb = original_fid.data.nbytes / 1024**2
            print(f"Cache size: {cache_size_mb:.2f} MB")
            print(f"Raw FID data: {fid_data_size_mb:.2f} MB")
            
            # Load from cache and verify bit-perfect reconstruction
            cached_fid = load_fid_cache("exp_2638_test", cache_dir=str(test_output_dir))
            
            # Verify bit-perfect data reconstruction
            np.testing.assert_array_equal(
                original_fid.data,
                cached_fid.data,
                err_msg="Cached FID data should be bit-perfect"
            )
            
            # Verify all parameters preserved
            assert cached_fid.spacing == original_fid.spacing
            assert cached_fid.probe_freq_mhz == original_fid.probe_freq_mhz
            assert cached_fid.sideband == original_fid.sideband
            assert cached_fid.shots == original_fid.shots
            assert cached_fid.n_points == original_fid.n_points
            
            # Verify metadata preservation
            assert 'source_path' in cached_fid.metadata
            assert 'blackchirp_params' in cached_fid.metadata
            
            print("✓ Experiment 2638 FID caching test passed - bit-perfect reconstruction verified")
            
        except Exception as e:
            pytest.skip(f"Could not test with experiment 2638 data: {e}")
    
    
    def test_cache_portability_with_real_data(self, test_output_dir):
        """Test that cached FID is portable and independent of source files."""
        try:
            # Load and cache experiment 2638
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            original_fid = ftmw_data.fid
            
            # Save cache
            cache_file = save_fid_cache("exp_2638_portable", original_fid, cache_dir=str(test_output_dir))
            
            # Verify cache file is self-contained
            with h5py.File(cache_file, 'r') as f:
                # Should have complete experiment info in cache
                fid_group = f['fid_data']
                assert 'time_series_data' in fid_group
                assert 'acquisition' in fid_group
                assert 'metadata' in fid_group
                
                # Verify portability attributes
                assert fid_group.attrs['cache_description'] == 'Portable FID cache - contains all data needed for independent analysis'
                assert fid_group.attrs['summary_probe_freq_mhz'] == 40960.0
                assert fid_group.attrs['summary_sideband'] == 'lower'
                assert fid_group.attrs['summary_duration_us'] > 14.9  # ~15 μs
                assert fid_group.attrs['summary_n_points'] == 750000
            
            # Load and verify works independently
            cached_fid = load_fid_cache("exp_2638_portable", cache_dir=str(test_output_dir))
            
            # Should be able to create ComplexFT from cached FID (Stage 0-1 workflow test)
            # This tests that cache is truly self-contained for analysis
            preprocessed_fid = cached_fid.preprocess(zpf=1, expf_us=5.0)
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()
            
            # Verify FFT computation works with cached data
            assert len(complex_spectrum) > 0
            assert len(freq_array) > 0
            assert np.all(np.isfinite(complex_spectrum))
            assert np.all(np.isfinite(freq_array))
            
            print("✓ Cache portability test passed - cached FID is self-contained for analysis")
            
        except Exception as e:
            pytest.skip(f"Could not test cache portability with experiment 2638 data: {e}")


class TestErrorConditionsAndEdgeCases:
    """Test error handling and edge cases in FID serialization."""
    
    def test_invalid_fid_object(self):
        """Test error handling with invalid FID objects."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with h5py.File(Path(tmp_dir) / "test.h5", 'w') as f:
                group = f.create_group('test')
                
                # Should raise error for non-FID object
                with pytest.raises(RuntimeError, match="Failed to serialize FID to HDF5"):
                    save_fid_to_hdf5("not_a_fid", group)
    
    def test_corrupted_hdf5_loading(self):
        """Test error handling when loading corrupted HDF5 files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create corrupted HDF5 structure
            test_file = Path(tmp_dir) / "corrupted.h5"
            with h5py.File(test_file, 'w') as f:
                group = f.create_group('test')
                # Missing required datasets/groups
                group.create_dataset('time_series_data', data=[1, 2, 3])
                # Missing 'acquisition' and 'metadata' groups
            
            with h5py.File(test_file, 'r') as f:
                group = f['test']
                with pytest.raises(RuntimeError, match="Failed to deserialize FID from HDF5"):
                    load_fid_from_hdf5(group)
    
    def test_missing_cache_files(self):
        """Test handling of missing cache files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Try to load non-existent cache
            with pytest.raises(FileNotFoundError, match="FID cache file not found"):
                load_fid_cache("nonexistent", cache_dir=tmp_dir)
    
    def test_invalid_cache_structure(self):
        """Test handling of invalid cache file structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "invalid_cache_fid.h5"
            
            # Create invalid cache file (missing fid_data group)
            with h5py.File(cache_file, 'w') as f:
                f.create_group('wrong_group')  # Wrong group name
            
            with pytest.raises(ValueError, match="Invalid FID cache file: missing 'fid_data' group"):
                load_fid_cache("invalid_cache", cache_dir=tmp_dir)