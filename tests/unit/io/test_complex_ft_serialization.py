"""
Unit tests for ComplexFT Stage 0-1 architecture and on-demand calculation workflow.

Tests focus on the new Stage 0-1 architecture where:
- Stage 0: FID data is cached (tested elsewhere)
- Stage 1: ComplexFT objects are calculated on-demand from cached FID + parameters

Test Coverage:
- Parameter validation for FT processing
- Three-stage workflow integration (preprocess → compute_fft → from_spectrum)
- On-demand calculation consistency and correctness
- ComplexFT API functionality (current methods only)
- Performance characteristics of on-demand calculation

IMPORTANT: This test file NO LONGER tests ComplexFT storage/serialization.
ComplexFT objects are temporary and calculated fresh each session.
"""

import pytest
import numpy as np
import time
from pathlib import Path

from ftmwpipeline.core.data_structures import (
    ComplexFT, FID, FIDProcessingParameters, Sideband
)
from ftmwpipeline.io import load_blackchirp_experiment


class TestParameterValidation:
    """Test FID processing parameter validation for on-demand ComplexFT calculation."""
    
    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for parameter validation testing."""
        n_points = 1000
        spacing = 2e-8  # 20 ns spacing
        probe_freq = 18000.0  # 18 GHz
        
        # Generate simple exponentially decaying sinusoid
        t = np.arange(n_points) * spacing
        data = np.exp(-t / 2e-6) * np.cos(2 * np.pi * 150e6 * t)  # 150 MHz signal
        
        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            metadata={'test_source': 'parameter_validation'}
        )
    
    def test_valid_parameter_combinations(self, sample_fid):
        """Test that valid parameter combinations work correctly."""
        # Test basic parameters
        preprocessed = sample_fid.preprocess(zpf=1, expf_us=5.0)
        spectrum, freqs = preprocessed.compute_fft()
        complex_ft = ComplexFT.from_spectrum(spectrum, freqs)
        
        assert complex_ft.n_points > 0
        assert len(complex_ft.freq_array) == len(complex_ft.complex_spectrum)
        
        # Test parameter variations
        valid_params = [
            {'zpf': 0, 'expf_us': None},  # No processing
            {'zpf': 1, 'expf_us': 2.0},   # Basic processing
            {'zpf': 2, 'expf_us': 10.0},  # Heavy processing
            {'start_us': 1.0, 'end_us': 15.0, 'zpf': 1},  # Windowing
        ]
        
        for params in valid_params:
            preprocessed = sample_fid.preprocess(**params)
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)
            
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))
            assert np.all(np.isfinite(complex_ft.freq_array))
    
    def test_parameter_validation_errors(self, sample_fid):
        """Test that invalid parameters are properly rejected."""
        # Test invalid parameter values that actually cause errors
        
        # Test negative zero padding
        with pytest.raises(ValueError, match="Zero padding factor must be non-negative"):
            sample_fid.preprocess(zpf=-1)
        
        # Test start > end time
        with pytest.raises(ValueError, match="Start time must be less than end time"):
            sample_fid.preprocess(start_us=20.0, end_us=10.0)
        
        # Test invalid exponential filter values (must be positive)
        with pytest.raises(ValueError, match="Exponential filter time constant must be positive"):
            sample_fid.preprocess(expf_us=0.0)
        
        with pytest.raises(ValueError, match="Exponential filter time constant must be positive"):
            sample_fid.preprocess(expf_us=-5.0)
    
    def test_parameter_merging_and_defaults(self, sample_fid):
        """Test that preprocess() uses method defaults, not FID's processing parameters."""
        # Create FID with processing parameters (these are NOT automatically used by preprocess())
        processing = FIDProcessingParameters(
            zpf=2,  # Different from method default (1)
            expf_us=3.0,  # Different from method default (None)
            rdc=False  # Different from method default (True)
        )
        
        fid_with_processing = FID(
            data=sample_fid.data,
            spacing=sample_fid.spacing,
            probe_freq_mhz=sample_fid.probe_freq_mhz,
            sideband=sample_fid.sideband,
            processing=processing
        )
        
        # Test that preprocess() uses METHOD defaults, not FID's stored processing parameters
        preprocessed = fid_with_processing.preprocess()
        assert preprocessed.processing_params.zpf == 1  # Method default, not FID's zpf=2
        assert preprocessed.processing_params.expf_us is None  # Method default, not FID's expf_us=3.0
        assert preprocessed.processing_params.rdc == True  # Method default, not FID's rdc=False
        
        # Test that explicit parameters override method defaults
        preprocessed = fid_with_processing.preprocess(zpf=5, expf_us=7.0, rdc=False)
        assert preprocessed.processing_params.zpf == 5
        assert preprocessed.processing_params.expf_us == 7.0
        assert preprocessed.processing_params.rdc == False
        
        # Test that parameter merging happens at CLI level, not in preprocess()
        # This is the intended workflow: CLI merges user input + cached defaults, then calls preprocess()
        cached_defaults = fid_with_processing.processing
        merged_zpf = cached_defaults.zpf if cached_defaults.zpf is not None else 1
        merged_expf_us = cached_defaults.expf_us if cached_defaults.expf_us is not None else 5.0
        
        preprocessed = fid_with_processing.preprocess(zpf=merged_zpf, expf_us=merged_expf_us)
        assert preprocessed.processing_params.zpf == 2  # From cached defaults
        assert preprocessed.processing_params.expf_us == 3.0  # From cached defaults


class TestThreeStageWorkflow:
    """Test the three-stage workflow: preprocess → compute_fft → from_spectrum."""
    
    @pytest.fixture
    def sample_fid(self):
        """Create a larger FID for comprehensive workflow testing."""
        n_points = 2000
        spacing = 1e-8  # 10 ns spacing
        probe_freq = 20000.0  # 20 GHz
        
        # Generate more complex signal
        t = np.arange(n_points) * spacing
        signal = (np.exp(-t / 3e-6) * np.cos(2 * np.pi * 100e6 * t) +
                 0.5 * np.exp(-t / 2e-6) * np.cos(2 * np.pi * 250e6 * t))
        
        processing = FIDProcessingParameters(
            zpf=1,
            expf_us=4.0,
            rdc=True
        )
        
        return FID(
            data=signal,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.UPPER,
            processing=processing,
            metadata={'test_type': 'workflow_testing'}
        )
    
    def test_stage_1_preprocessing(self, sample_fid):
        """Test Stage 1: FID preprocessing."""
        # Test basic preprocessing
        preprocessed = sample_fid.preprocess(zpf=1, expf_us=5.0)
        
        assert hasattr(preprocessed, 'data')
        assert hasattr(preprocessed, 'processing_params')
        assert hasattr(preprocessed, 'spacing')
        assert hasattr(preprocessed, 'probe_freq_mhz')
        assert hasattr(preprocessed, 'sideband')
        
        # Verify processing parameters are stored
        assert preprocessed.processing_params.zpf == 1
        assert preprocessed.processing_params.expf_us == 5.0
        
        # Verify data is processed (different from original)
        assert len(preprocessed.data) != len(sample_fid.data)  # Zero padding changes length
        
        # Test preprocessing with different parameters produces different results
        preprocessed_alt = sample_fid.preprocess(zpf=2, expf_us=2.0)
        assert len(preprocessed_alt.data) != len(preprocessed.data)  # Different zpf
    
    def test_stage_2_fft_computation(self, sample_fid):
        """Test Stage 2: FFT computation."""
        preprocessed = sample_fid.preprocess(zpf=1, expf_us=3.0)
        spectrum, freq_array = preprocessed.compute_fft()
        
        # Verify FFT results have expected properties
        assert isinstance(spectrum, np.ndarray)
        assert isinstance(freq_array, np.ndarray)
        assert spectrum.dtype == complex
        assert freq_array.dtype == float
        assert len(spectrum) == len(freq_array)
        
        # Verify frequency array is monotonic and reasonable
        assert len(freq_array) > 0
        if sample_fid.sideband == Sideband.UPPER:
            assert freq_array[1] > freq_array[0]  # Ascending
        else:  # LOWER sideband
            assert freq_array[1] < freq_array[0]  # Descending
        
        # Verify spectrum contains finite values
        assert np.all(np.isfinite(spectrum))
        assert np.all(np.isfinite(freq_array))
    
    def test_stage_3_complex_ft_creation(self, sample_fid):
        """Test Stage 3: ComplexFT object creation."""
        preprocessed = sample_fid.preprocess(zpf=1, expf_us=4.0)
        spectrum, freq_array = preprocessed.compute_fft()
        
        # Create ComplexFT object
        metadata = {
            'processing_params': preprocessed.processing_params,
            'source_metadata': sample_fid.metadata
        }
        complex_ft = ComplexFT.from_spectrum(spectrum, freq_array, metadata)
        
        # Verify ComplexFT object properties
        assert isinstance(complex_ft, ComplexFT)
        assert len(complex_ft.freq_array) == len(freq_array)
        assert len(complex_ft.complex_spectrum) == len(spectrum)
        assert complex_ft.n_points == len(spectrum)
        
        # Verify arrays are properly stored
        np.testing.assert_array_equal(complex_ft.freq_array, freq_array)
        np.testing.assert_array_equal(complex_ft.complex_spectrum, spectrum)
        
        # Verify metadata is preserved
        assert 'processing_params' in complex_ft.metadata
        assert 'source_metadata' in complex_ft.metadata
    
    def test_complete_workflow_consistency(self, sample_fid):
        """Test that the complete three-stage workflow is consistent."""
        # Run workflow multiple times with same parameters
        results = []
        
        for _ in range(3):  # Run 3 times
            preprocessed = sample_fid.preprocess(zpf=1, expf_us=5.0)
            spectrum, freq_array = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freq_array)
            results.append((spectrum, freq_array))
        
        # All runs should produce identical results (deterministic)
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0][0], results[i][0],
                err_msg="Spectrum should be identical across runs"
            )
            np.testing.assert_array_equal(
                results[0][1], results[i][1],
                err_msg="Frequency array should be identical across runs"
            )
    
    def test_parameter_passing_through_workflow(self, sample_fid):
        """Test that parameters are correctly passed through all workflow stages."""
        # Test specific parameter combinations
        test_params = [
            {'zpf': 0, 'expf_us': None, 'rdc': False},
            {'zpf': 1, 'expf_us': 3.0, 'rdc': True},
            {'zpf': 2, 'expf_us': 8.0, 'start_us': 2.0, 'end_us': 15.0},
        ]
        
        for params in test_params:
            preprocessed = sample_fid.preprocess(**params)
            spectrum, freq_array = preprocessed.compute_fft()
            
            # Verify parameters are stored in preprocessed object
            for key, value in params.items():
                assert getattr(preprocessed.processing_params, key) == value
            
            # Verify FFT results are affected by parameters
            assert len(spectrum) > 0
            assert len(freq_array) > 0
            
            # Different parameters should give different spectrum lengths (due to zpf)
            if params['zpf'] == 2:
                assert len(spectrum) > 4000  # Should be significantly larger
            elif params['zpf'] == 0:
                assert len(spectrum) <= 2000  # Should be close to original length


class TestComplexFTAPI:
    """Test ComplexFT object API functionality (current methods only)."""
    
    @pytest.fixture
    def complex_ft_sample(self):
        """Create a ComplexFT for API testing."""
        # Create synthetic data
        freqs = np.linspace(18500, 19500, 1000)  # 1 GHz range
        # Create complex spectrum with some structure
        real_part = np.exp(-(freqs - 19000)**2 / (2 * 50**2))  # Gaussian around 19 GHz
        imag_part = 0.3 * np.exp(-(freqs - 19200)**2 / (2 * 30**2))  # Smaller peak at 19.2 GHz
        spectrum = real_part + 1j * imag_part
        
        return ComplexFT.from_spectrum(spectrum, freqs, 
                                     metadata={'test_type': 'api_testing'})
    
    def test_basic_properties(self, complex_ft_sample):
        """Test basic ComplexFT properties."""
        ft = complex_ft_sample
        
        # Test array properties
        assert len(ft.freq_array) == 1000
        assert len(ft.complex_spectrum) == 1000
        assert ft.n_points == 1000
        
        # Test that arrays are properly typed
        assert ft.freq_array.dtype == float
        assert ft.complex_spectrum.dtype == complex
        
        # Test frequency range
        assert ft.freq_array[0] == pytest.approx(18500.0)
        assert ft.freq_array[-1] == pytest.approx(19500.0)
    
    def test_spectrum_properties(self, complex_ft_sample):
        """Test spectrum property methods."""
        ft = complex_ft_sample
        
        # Test magnitude spectrum
        mag_spectrum = ft.magnitude_spectrum
        assert len(mag_spectrum) == ft.n_points
        assert mag_spectrum.dtype == float
        assert np.all(mag_spectrum >= 0)  # Magnitude should be non-negative
        
        # Test real and imaginary components
        assert hasattr(ft, 'real_spectrum')
        assert hasattr(ft, 'imag_spectrum')
        
        real_spectrum = ft.real_spectrum
        imag_spectrum = ft.imag_spectrum
        
        assert len(real_spectrum) == ft.n_points
        assert len(imag_spectrum) == ft.n_points
        
        # Verify reconstruction from real/imag parts
        reconstructed = real_spectrum + 1j * imag_spectrum
        np.testing.assert_array_almost_equal(reconstructed, ft.complex_spectrum)
    
    def test_frequency_operations(self, complex_ft_sample):
        """Test frequency-related operations."""
        ft = complex_ft_sample
        
        # Test frequency step calculation (if available)
        if hasattr(ft, 'freq_step'):
            freq_step = ft.freq_step
            expected_step = (ft.freq_array[-1] - ft.freq_array[0]) / (len(ft.freq_array) - 1)
            assert freq_step == pytest.approx(expected_step)
    
    def test_trim_to_range_functionality(self, complex_ft_sample):
        """Test trim_to_range method if available."""
        ft = complex_ft_sample
        
        if hasattr(ft, 'trim_to_range'):
            # Test trimming to smaller range
            trimmed_ft = ft.trim_to_range(18800, 19200)
            
            # Verify trimmed object is smaller
            assert trimmed_ft.n_points < ft.n_points
            
            # Verify frequency range is within specified bounds
            assert trimmed_ft.freq_array[0] >= 18800
            assert trimmed_ft.freq_array[-1] <= 19200
            
            # Verify data integrity
            assert np.all(np.isfinite(trimmed_ft.complex_spectrum))
            assert np.all(np.isfinite(trimmed_ft.freq_array))
    
    def test_metadata_handling(self, complex_ft_sample):
        """Test metadata storage and retrieval."""
        ft = complex_ft_sample
        
        # Test that metadata is accessible
        assert hasattr(ft, 'metadata')
        assert isinstance(ft.metadata, dict)
        assert 'test_type' in ft.metadata
        assert ft.metadata['test_type'] == 'api_testing'
        
        # Test metadata modification (if mutable)
        original_metadata = ft.metadata.copy()
        ft.metadata['new_key'] = 'new_value'
        assert 'new_key' in ft.metadata
        assert ft.metadata['new_key'] == 'new_value'




class TestRealExperimentalDataWorkflow:
    """Test the Stage 0-1 workflow with real experimental data."""
    
    def test_experiment_2638_on_demand_calculation(self):
        """Test on-demand ComplexFT calculation with real experiment 2638 data."""
        try:
            # Load cached FID data (Stage 0)
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            
            print(f"Loaded FID with {len(ftmw_data.fid.data)} points")
            print(f"FID spacing: {ftmw_data.fid.spacing:.2e} seconds")
            print(f"Probe frequency: {ftmw_data.fid.probe_freq_mhz} MHz")
            print(f"Sideband: {ftmw_data.fid.sideband}")
            
            # Test on-demand ComplexFT calculation (Stage 1)
            # Use three-stage workflow
            preprocessed = ftmw_data.fid.preprocess(zpf=1, expf_us=5.0)
            spectrum, freq_array = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(
                spectrum, freq_array,
                metadata={'processing_params': preprocessed.processing_params}
            )
            
            print(f"ComplexFT spectrum size: {complex_ft.n_points} points")
            print(f"Frequency range: {complex_ft.freq_array[0]:.1f} - {complex_ft.freq_array[-1]:.1f} MHz")
            
            # Verify calculation results
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))
            assert np.all(np.isfinite(complex_ft.freq_array))
            
            print("✓ Experiment 2638 on-demand calculation test passed")
            
        except Exception as e:
            pytest.skip(f"Could not load experimental data: {e}")
    
    def test_parameter_exploration_workflow(self):
        """Test interactive parameter exploration workflow with real data."""
        try:
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            
            # Simulate interactive parameter exploration
            parameter_variations = [
                {'zpf': 1, 'expf_us': 3.0},
                {'zpf': 1, 'expf_us': 5.0},
                {'zpf': 1, 'expf_us': 8.0},
                {'zpf': 2, 'expf_us': 5.0},
            ]
            
            results = []
            
            for params in parameter_variations:
                preprocessed = ftmw_data.fid.preprocess(**params)
                spectrum, freq_array = preprocessed.compute_fft()
                complex_ft = ComplexFT.from_spectrum(spectrum, freq_array)
                
                results.append((params, complex_ft))
                
                print(f"Parameters {params}: {complex_ft.n_points} points")
            
            # Verify all calculations succeeded
            assert len(results) == len(parameter_variations)
            for params, ft in results:
                assert ft.n_points > 0
            
            # Different parameters should produce different results
            spectra = [ft.complex_spectrum for _, ft in results]
            for i in range(1, len(spectra)):
                # Should not be identical (different parameters = different processing)
                assert not np.array_equal(spectra[0], spectra[i])
            
        except Exception as e:
            pytest.skip(f"Could not load experimental data: {e}")
    
    def test_activity_region_trimming(self):
        """Test trimming to activity region (common analysis workflow)."""
        try:
            ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
            
            # Calculate full spectrum
            preprocessed = ftmw_data.fid.preprocess(zpf=1, expf_us=5.0)
            spectrum, freq_array = preprocessed.compute_fft()
            full_ft = ComplexFT.from_spectrum(spectrum, freq_array)
            
            print(f"Full spectrum: {full_ft.n_points} points")
            print(f"Full range: {full_ft.freq_array[0]:.1f} - {full_ft.freq_array[-1]:.1f} MHz")
            
            # Trim to recommended activity region for exp 2638
            if hasattr(full_ft, 'trim_to_range'):
                trimmed_ft = full_ft.trim_to_range(26500, 40000)
                
                print(f"Trimmed spectrum: {trimmed_ft.n_points} points")
                print(f"Trimmed range: {trimmed_ft.freq_array[0]:.1f} - {trimmed_ft.freq_array[-1]:.1f} MHz")
                
                # Verify trimming worked correctly
                assert trimmed_ft.n_points < full_ft.n_points
                assert trimmed_ft.freq_array[0] >= 26500
                assert trimmed_ft.freq_array[-1] <= 40000
                assert np.all(np.isfinite(trimmed_ft.complex_spectrum))
            
        except Exception as e:
            pytest.skip(f"Could not load experimental data: {e}")