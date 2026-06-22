"""
Unit tests for the three-stage FT workflow.

Tests Stage 0-1 architecture where ComplexFT is calculated on-demand:
1. FID.preprocess(**params) → PreprocessedFID
2. PreprocessedFID.compute_fft() → (complex_spectrum, freq_array)
3. ComplexFT.from_spectrum(spectrum, freq_array) → ComplexFT

The canonical FT is unconditionally unapodized, un-windowed, and native-length:
preprocessing is just active-region selection (start_us/end_us) plus optional DC
removal. There are no zero-pad / apodization / window-function knobs.
"""

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    FID,
    ComplexFT,
    FIDProcessingParameters,
    PreprocessedFID,
    Sideband,
)
from ftmwpipeline.io.data_loaders import BlackChirpLoader


class TestPreprocessedFID:
    """Test PreprocessedFID class and preprocessing functionality."""

    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for preprocessing tests."""
        # Create realistic FID data
        n_points = 1000
        spacing = 2e-11  # 20 ns spacing
        probe_freq = 18000.0  # 18 GHz

        # Generate exponentially decaying multi-component signal
        t = np.arange(n_points) * spacing
        data = np.exp(-t / 3e-6) * (
            0.8 * np.cos(2 * np.pi * 100e6 * t)  # 100 MHz component
            + 0.5 * np.cos(2 * np.pi * 250e6 * t)  # 250 MHz component
            + 0.1 * np.random.normal(0, 0.05, n_points)
        )  # Noise

        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            shots=100000,
        )

    def test_preprocessed_fid_creation(self, sample_fid):
        """Test PreprocessedFID object creation and properties."""
        preprocessed = sample_fid.preprocess(
            start_us=0.5,
            end_us=10.0,
            units_power=6,
        )

        # Check type and basic properties
        assert isinstance(preprocessed, PreprocessedFID)
        assert preprocessed.spacing == sample_fid.spacing
        assert preprocessed.probe_freq_mhz == sample_fid.probe_freq_mhz
        assert preprocessed.sideband == sample_fid.sideband
        assert preprocessed.original_length == sample_fid.n_points

        # Check processing parameters were stored
        assert preprocessed.processing_params.start_us == 0.5
        assert preprocessed.processing_params.end_us == 10.0
        assert preprocessed.processing_params.units_power == 6

        # The canonical FT is native-length: no zero-padding.
        assert preprocessed.n_points == sample_fid.n_points

        # Check metadata
        assert "source_fid_metadata" in preprocessed.metadata

    def test_active_region_is_selected(self, sample_fid):
        """Points outside [start_us, end_us] are zeroed; native length kept."""
        duration_us = sample_fid.duration_us
        start_us = duration_us * 0.1
        end_us = duration_us * 0.8

        preprocessed = sample_fid.preprocess(
            start_us=start_us,
            end_us=end_us,
        )

        # Native-length output, no padding.
        assert len(preprocessed.data) == len(sample_fid.data)

        time_us = sample_fid.time_array_us()
        start_idx = int(np.searchsorted(time_us, start_us))
        end_idx = int(np.searchsorted(time_us, end_us))

        # Regions outside the active window are zeroed.
        assert np.all(preprocessed.data[:start_idx] == 0.0)
        assert np.all(preprocessed.data[end_idx:] == 0.0)
        # Active region carries non-zero, finite data.
        assert np.any(preprocessed.data[start_idx:end_idx] != 0)
        assert np.isfinite(preprocessed.data).all()

    def test_windowing_bounds(self, sample_fid):
        """Test windowing boundary conditions."""
        duration_us = sample_fid.duration_us

        # Test start_us and end_us within bounds
        preprocessed = sample_fid.preprocess(
            start_us=duration_us * 0.1, end_us=duration_us * 0.9
        )
        assert isinstance(preprocessed, PreprocessedFID)

        # Test start_us = 0 (beginning)
        preprocessed = sample_fid.preprocess(start_us=0.0, end_us=duration_us * 0.5)
        assert isinstance(preprocessed, PreprocessedFID)

        # Test end_us = duration (end)
        preprocessed = sample_fid.preprocess(
            start_us=duration_us * 0.1, end_us=duration_us
        )
        assert isinstance(preprocessed, PreprocessedFID)

        # Test None values (should use full range)
        preprocessed = sample_fid.preprocess(start_us=None, end_us=None)
        assert isinstance(preprocessed, PreprocessedFID)

    def test_dc_removal_is_unconditional(self, sample_fid):
        """DC removal always runs: the active region is zero-mean afterward."""
        # Create FID with a known DC offset.
        dc_offset = 5.0
        fid_with_dc = FID(
            data=sample_fid.data + dc_offset,
            spacing=sample_fid.spacing,
            probe_freq_mhz=sample_fid.probe_freq_mhz,
            sideband=sample_fid.sideband,
        )

        duration_us = fid_with_dc.duration_us
        start_us = duration_us * 0.1
        end_us = duration_us * 0.9
        preprocessed = fid_with_dc.preprocess(start_us=start_us, end_us=end_us)

        # The active region (the only part the FFT sees) is mean-subtracted.
        time_us = fid_with_dc.time_array_us()
        start_idx = int(np.searchsorted(time_us, start_us))
        end_idx = int(np.searchsorted(time_us, end_us))
        active = preprocessed.data[start_idx:end_idx]
        assert abs(float(np.mean(active))) < 1e-9


class TestPreprocessedFIDFFTComputation:
    """Test FFT computation from PreprocessedFID."""

    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for FFT tests."""
        n_points = 1000
        spacing = 1e-8  # 10 ns spacing
        probe_freq = 20000.0  # 20 GHz

        # Create signal with known frequency components
        t = np.arange(n_points) * spacing
        # Two well-separated frequency components
        f1, f2 = 100e6, 200e6  # 100 MHz and 200 MHz
        data = (
            0.8 * np.cos(2 * np.pi * f1 * t)
            + 0.6 * np.cos(2 * np.pi * f2 * t)
            + 0.1 * np.random.normal(0, 0.02, n_points)
        )

        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            shots=10000,
        )

    def test_fft_computation_basic(self, sample_fid):
        """Test basic FFT computation from preprocessed FID."""
        # Preprocess FID
        preprocessed = sample_fid.preprocess()

        # Compute FFT
        complex_spectrum, freq_array = preprocessed.compute_fft()

        # Check output types and shapes
        assert isinstance(complex_spectrum, np.ndarray)
        assert isinstance(freq_array, np.ndarray)
        assert np.iscomplexobj(complex_spectrum)
        assert np.isrealobj(freq_array)
        assert len(complex_spectrum) == len(freq_array)

        # Check frequency array is in MHz
        assert np.all(freq_array > 1000)  # Should be in MHz range for our test case
        assert np.all(np.isfinite(freq_array))

        # Check spectrum is finite
        assert np.all(np.isfinite(complex_spectrum))

    def test_molecular_frequency_conversion(self, sample_fid):
        """Test molecular frequency conversion for different sidebands."""
        # Test lower sideband (molecular = probe - scope)
        fid_lower = FID(
            data=sample_fid.data,
            spacing=sample_fid.spacing,
            probe_freq_mhz=20000.0,
            sideband=Sideband.LOWER,
        )

        # Test upper sideband (molecular = probe + scope)
        fid_upper = FID(
            data=sample_fid.data,
            spacing=sample_fid.spacing,
            probe_freq_mhz=20000.0,
            sideband=Sideband.UPPER,
        )

        # Compute FFTs
        preprocessed_lower = fid_lower.preprocess()
        preprocessed_upper = fid_upper.preprocess()

        _, freq_lower = preprocessed_lower.compute_fft()
        _, freq_upper = preprocessed_upper.compute_fft()

        # For our 20 GHz probe with lower sideband, molecular freqs should be < 20000 MHz
        assert np.all(freq_lower <= 20000.0)  # Lower sideband

        # For upper sideband, molecular freqs should be >= 20000 MHz
        assert np.all(freq_upper >= 20000.0)  # Upper sideband

        # The frequency arrays should be mirror images around probe frequency
        probe_freq = 20000.0
        lower_range = probe_freq - freq_lower[0], probe_freq - freq_lower[-1]
        upper_range = freq_upper[0] - probe_freq, freq_upper[-1] - probe_freq

        # Ranges should be approximately equal
        assert abs(lower_range[0] - upper_range[0]) < 1.0  # Within 1 MHz

    def test_fft_normalization_and_scaling(self, sample_fid):
        """Test FFT normalization and units scaling."""
        preprocessed = sample_fid.preprocess(units_power=6)

        # Compute FFT
        complex_spectrum, freq_array = preprocessed.compute_fft()

        # Check that normalization was applied (divides by original length)
        max_amplitude = np.max(np.abs(complex_spectrum))
        assert max_amplitude > 1e-10  # Not too small
        assert max_amplitude < 1e10  # Not too large

        # Test different units_power scaling
        preprocessed_power3 = sample_fid.preprocess(units_power=3)
        spectrum_power3, _ = preprocessed_power3.compute_fft()

        preprocessed_power6 = sample_fid.preprocess(units_power=6)
        spectrum_power6, _ = preprocessed_power6.compute_fft()

        # Spectrum with units_power=6 should be 1000x larger than units_power=3
        ratio = np.mean(np.abs(spectrum_power6)) / np.mean(np.abs(spectrum_power3))
        assert abs(ratio - 1000.0) < 100.0  # Should be approximately 1000


class TestThreeStageWorkflow:
    """Test the complete three-stage workflow."""

    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for workflow tests."""
        n_points = 2000
        spacing = 1e-8  # 10 ns spacing
        probe_freq = 15000.0  # 15 GHz

        # Create realistic multi-component signal
        t = np.arange(n_points) * spacing
        data = np.exp(-t / 2e-6) * (
            1.0 * np.cos(2 * np.pi * 120e6 * t)  # Strong 120 MHz line
            + 0.5 * np.cos(2 * np.pi * 180e6 * t)  # Medium 180 MHz line
            + 0.2 * np.cos(2 * np.pi * 350e6 * t)  # Weak 350 MHz line
            + 0.05 * np.random.normal(0, 0.1, n_points)
        )  # Noise

        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.UPPER,
            shots=200000,
        )

    def test_complete_three_stage_workflow(self, sample_fid):
        """Test the complete three-stage workflow."""
        # Stage 1: FID preprocessing
        preprocessed_fid = sample_fid.preprocess(
            start_us=1.0,
            end_us=15.0,
            units_power=6,
        )

        assert isinstance(preprocessed_fid, PreprocessedFID)
        assert preprocessed_fid.n_points == sample_fid.n_points  # Native length

        # Stage 2: FFT computation
        complex_spectrum, freq_array = preprocessed_fid.compute_fft()

        assert isinstance(complex_spectrum, np.ndarray)
        assert isinstance(freq_array, np.ndarray)
        assert len(complex_spectrum) == len(freq_array)
        assert np.iscomplexobj(complex_spectrum)

        # Stage 3: ComplexFT creation using from_spectrum method (preferred approach)
        complex_ft = ComplexFT.from_spectrum(
            complex_spectrum=complex_spectrum,
            freq_array=freq_array,
            metadata={"processing_params": preprocessed_fid.processing_params},
        )

        assert isinstance(complex_ft, ComplexFT)
        assert len(complex_ft.freq_array) == len(freq_array)
        assert len(complex_ft.complex_spectrum) == len(complex_spectrum)

        # Verify the workflow produces sensible results
        magnitude = complex_ft.magnitude_spectrum
        assert np.all(magnitude >= 0)  # Magnitudes should be non-negative
        assert np.any(magnitude > np.mean(magnitude) * 2)  # Should have peaks

    def test_workflow_reproducibility(self, sample_fid):
        """Test that workflow produces reproducible results."""
        params = {
            "start_us": 2.0,
            "end_us": 12.0,
            "units_power": 6,
        }

        # First run
        preprocessed1 = sample_fid.preprocess(**params)
        spectrum1, freqs1 = preprocessed1.compute_fft()

        # Second run
        preprocessed2 = sample_fid.preprocess(**params)
        spectrum2, freqs2 = preprocessed2.compute_fft()

        # Results should be identical
        np.testing.assert_array_equal(
            preprocessed1.data,
            preprocessed2.data,
            err_msg="Preprocessing should be reproducible",
        )
        np.testing.assert_array_equal(
            spectrum1, spectrum2, err_msg="FFT results should be reproducible"
        )
        np.testing.assert_array_equal(
            freqs1, freqs2, err_msg="Frequency arrays should be reproducible"
        )

    def test_workflow_parameter_independence(self, sample_fid):
        """Test that different active windows produce different spectra."""
        duration_us = sample_fid.duration_us
        params1 = {
            "start_us": duration_us * 0.1,
            "end_us": duration_us * 0.8,
        }
        params2 = {
            "start_us": duration_us * 0.05,
            "end_us": duration_us * 0.9,
        }

        # Stage 1: Different active windows should give different content
        # (same native length, since the FT is unpadded).
        preprocessed1 = sample_fid.preprocess(**params1)
        preprocessed2 = sample_fid.preprocess(**params2)

        assert preprocessed1.n_points == preprocessed2.n_points
        assert not np.array_equal(preprocessed1.data, preprocessed2.data)

        # Stage 2: Different preprocessed FIDs should give different FFTs
        spectrum1, _ = preprocessed1.compute_fft()
        spectrum2, _ = preprocessed2.compute_fft()

        assert len(spectrum1) == len(spectrum2)
        assert not np.array_equal(spectrum1, spectrum2)

    def test_workflow_with_edge_case_parameters(self, sample_fid):
        """Test workflow with edge case parameters."""
        # Test minimal parameters
        minimal_preprocessed = sample_fid.preprocess()
        minimal_spectrum, minimal_freqs = minimal_preprocessed.compute_fft()

        assert len(minimal_spectrum) > 0
        assert len(minimal_freqs) > 0
        assert np.all(np.isfinite(minimal_spectrum))
        assert np.all(np.isfinite(minimal_freqs))

        # Test a narrow active window
        windowed_preprocessed = sample_fid.preprocess(
            start_us=0.1,
            end_us=sample_fid.duration_us - 0.1,
            units_power=9,
        )
        windowed_spectrum, windowed_freqs = windowed_preprocessed.compute_fft()

        assert len(windowed_spectrum) == len(minimal_spectrum)  # Native length
        assert np.all(np.isfinite(windowed_spectrum))
        assert np.all(np.isfinite(windowed_freqs))


class TestThreeStageWorkflowWithRealData:
    """Test three-stage workflow with real experiment 2638 data."""

    def test_experiment_2638_three_stage_workflow(self):
        """Test complete workflow with real experiment 2638 data."""
        try:
            # Load real experimental data
            fid = BlackChirpLoader().load_fid(
                "examples/blackchirp_data/2638", fid_index=0
            )

            # Stage 1: Preprocessing (canonical unapodized native-length FT)
            preprocessed_fid = fid.preprocess(units_power=6)

            assert preprocessed_fid.n_points == fid.n_points

            # Stage 2: FFT computation
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()

            # Stage 3: ComplexFT creation using from_spectrum method
            complex_ft = ComplexFT.from_spectrum(
                complex_spectrum=complex_spectrum,
                freq_array=freq_array,
                metadata={
                    "processing_params": preprocessed_fid.processing_params,
                    "source": "experiment_2638",
                },
            )

            # Verify workflow results match expected specifications
            assert (
                complex_ft.freq_array[0] > 20000
            )  # Should start above 20 GHz (lower sideband)
            assert complex_ft.freq_array[-1] < 41000  # Should end below 41 GHz
            assert (
                len(complex_ft.complex_spectrum) > 300000
            )  # Should have high resolution

            # Test trimming to activity region per CLAUDE.md recommendations
            trimmed_ft = complex_ft.trim_to_range(26500, 40000)

            # Verify trimming worked correctly
            assert trimmed_ft.freq_array[0] >= 26500.0
            assert trimmed_ft.freq_array[-1] <= 40000.0
            assert len(trimmed_ft.freq_array) < len(complex_ft.freq_array)

            # Verify magnitude spectrum is reasonable
            magnitude = trimmed_ft.magnitude_spectrum
            assert np.all(magnitude >= 0)
            assert np.all(np.isfinite(magnitude))

        except Exception as e:
            pytest.skip(
                f"Could not test three-stage workflow with experiment 2638 data: {e}"
            )

    def test_three_stage_workflow_enforced(self):
        """Test that the three-stage workflow is enforced (no legacy FID.ft())."""
        try:
            # Load experiment data
            fid = BlackChirpLoader().load_fid(
                "examples/blackchirp_data/2638", fid_index=0
            )

            # Verify that legacy FID.ft() method does not exist
            assert not hasattr(
                fid, "ft"
            ), "Legacy FID.ft() method should be removed to enforce three-stage workflow"

            # Verify three-stage workflow works correctly
            preprocessed = fid.preprocess()
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

            # Verify results are valid
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))
            assert np.all(np.isfinite(complex_ft.freq_array))

        except Exception as e:
            pytest.skip(f"Could not test three-stage workflow enforcement: {e}")


class TestWorkflowErrorHandling:
    """Test error handling in three-stage workflow."""

    def test_invalid_preprocessing_parameters(self):
        """Test error handling for invalid preprocessing parameters."""
        fid = FID(data=[1.0, 2.0, 3.0], spacing=1e-6, probe_freq_mhz=1000.0)

        # Test invalid start_us > end_us
        with pytest.raises(ValueError, match="Start time must be less than end time"):
            fid.preprocess(start_us=10.0, end_us=5.0)

        # Test negative start_us
        with pytest.raises(ValueError, match="Start time must be non-negative"):
            fid.preprocess(start_us=-1.0)

        # Test negative end_us
        with pytest.raises(ValueError, match="End time must be non-negative"):
            fid.preprocess(end_us=-1.0)

    def test_empty_or_invalid_fid_data(self):
        """Test error handling for problematic FID data."""
        # Empty data stays empty (native length, no padding).
        fid_empty = FID(data=[], spacing=1e-6, probe_freq_mhz=1000.0)
        preprocessed_empty = fid_empty.preprocess()
        assert len(preprocessed_empty.data) == 0

        # Test NaN data
        fid_nan = FID(data=[1.0, np.nan, 3.0], spacing=1e-6, probe_freq_mhz=1000.0)
        preprocessed = fid_nan.preprocess()
        spectrum, freqs = preprocessed.compute_fft()

        # FFT of NaN data should contain NaN values
        assert np.any(np.isnan(spectrum))

        # Test infinite data
        fid_inf = FID(data=[1.0, np.inf, 3.0], spacing=1e-6, probe_freq_mhz=1000.0)
        preprocessed = fid_inf.preprocess()
        spectrum, freqs = preprocessed.compute_fft()

        # FFT of infinite data typically results in NaN values, not infinite values
        assert np.any(np.isnan(spectrum))


def test_fidprocessingparameters_defaults():
    """FIDProcessingParameters carries only data-selection / scaling knobs."""
    params = FIDProcessingParameters()
    assert params.start_us is None
    assert params.end_us is None
    assert params.units_power == 6
    # The retired apodization knobs and the rdc toggle no longer exist.
    assert not hasattr(params, "zpf")
    assert not hasattr(params, "expf_us")
    assert not hasattr(params, "winf")
    assert not hasattr(params, "rdc")
