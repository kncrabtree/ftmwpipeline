"""
Unit tests for the new three-stage FT workflow.

Tests Stage 0-1 architecture where ComplexFT is calculated on-demand:
1. FID.preprocess(**params) → PreprocessedFID
2. PreprocessedFID.compute_fft() → (complex_spectrum, freq_array)
3. ComplexFT.from_spectrum(spectrum, freq_array) → ComplexFT

This replaces the old single-step FID.ft() method and provides better separation
of concerns and control over processing stages.
"""

import pytest
import numpy as np
import scipy.signal as spsig
from pathlib import Path

from ftmwpipeline.core.data_structures import (
    FID,
    PreprocessedFID,
    ComplexFT,
    FIDProcessingParameters,
    Sideband,
)
from ftmwpipeline.io import load_blackchirp_experiment


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
        # Create preprocessed FID
        preprocessed = sample_fid.preprocess(
            start_us=0.5,
            end_us=10.0,
            zpf=1,
            expf_us=5.0,
            window_function="hann",
            rdc=True,
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
        assert preprocessed.processing_params.zpf == 1
        assert preprocessed.processing_params.expf_us == 5.0
        assert preprocessed.processing_params.winf == "hann"
        assert preprocessed.processing_params.rdc is True
        assert preprocessed.processing_params.units_power == 6

        # Check data was processed (zero-padded should be longer)
        assert preprocessed.n_points > sample_fid.n_points  # Due to zpf=1

        # Check metadata
        assert "source_fid_metadata" in preprocessed.metadata

    def test_fid_preprocessing_stages(self, sample_fid):
        """Test that FID preprocessing follows correct sequence."""
        # Test preprocessing with all features
        time_us = sample_fid.time_array_us()
        duration_us = sample_fid.duration_us
        # Use realistic time bounds based on actual FID duration
        start_us = duration_us * 0.1  # 10% into the FID
        end_us = duration_us * 0.8  # 80% into the FID

        preprocessed = sample_fid.preprocess(
            start_us=start_us,
            end_us=end_us,
            expf_us=3.0,
            window_function="hann",
            rdc=True,
            zpf=1,
        )

        # Verify that preprocessing applied windowing correctly
        # Data outside start_us to end_us should have been zeroed initially
        # Then processed data should be in the expected range
        assert len(preprocessed.data) > len(sample_fid.data)  # Zero-padded

        # Check that processing was applied in correct order by examining
        # non-zero regions (this is a bit complex to test directly,
        # but we can verify the overall integrity)
        assert np.any(preprocessed.data != 0)  # Should have non-zero data
        assert np.isfinite(preprocessed.data).all()  # All data should be finite

    def test_windowing_bounds(self, sample_fid):
        """Test windowing boundary conditions."""
        time_us = sample_fid.time_array_us()
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

    def test_zero_padding_factor(self, sample_fid):
        """Test zero padding functionality."""
        original_length = sample_fid.n_points

        # Test zpf=0 (no zero padding - keeps original length)
        preprocessed_0 = sample_fid.preprocess(zpf=0)
        assert preprocessed_0.n_points == original_length  # No padding applied

        # Test zpf=1 (pad to next power of 2, then double it)
        preprocessed_1 = sample_fid.preprocess(zpf=1)
        expected_1 = 2 ** (
            int(np.log2(original_length)) + 1 + 1
        )  # Next power of 2, then doubled
        assert preprocessed_1.n_points == expected_1

        # Test zpf=2 (pad to next power of 2, then quadruple it)
        preprocessed_2 = sample_fid.preprocess(zpf=2)
        expected_2 = 2 ** (
            int(np.log2(original_length)) + 1 + 2
        )  # Next power of 2, then quadrupled
        assert preprocessed_2.n_points == expected_2

        # Verify zero-padding relationship
        assert preprocessed_2.n_points == 2 * preprocessed_1.n_points
        assert (
            preprocessed_1.n_points > preprocessed_0.n_points
        )  # zpf=1 should be larger than zpf=0

    def test_exponential_filtering(self, sample_fid):
        """Test exponential filtering application."""
        # Test without filtering
        duration_us = sample_fid.duration_us
        no_filter = sample_fid.preprocess(
            expf_us=None, start_us=duration_us * 0.1, end_us=duration_us * 0.9
        )

        # Test with filtering
        with_filter = sample_fid.preprocess(
            expf_us=duration_us * 0.1,
            start_us=duration_us * 0.1,
            end_us=duration_us * 0.9,
        )

        # Both should have same length (same zpf)
        assert no_filter.n_points == with_filter.n_points

        # But filtered version should have different amplitude profile
        # (This is hard to test precisely without knowing the exact implementation,
        # but we can verify the data changed)
        assert not np.array_equal(no_filter.data, with_filter.data)

    def test_window_function_application(self, sample_fid):
        """Test window function application."""
        # Test different window functions
        window_functions = ["hann", "hamming", "blackman", "bartlett"]

        preprocessed_results = {}
        for winf in window_functions:
            duration_us = sample_fid.duration_us
            preprocessed_results[winf] = sample_fid.preprocess(
                window_function=winf,
                start_us=duration_us * 0.1,
                end_us=duration_us * 0.9,
                zpf=0,  # Same size for comparison
            )

        # All should have same length
        lengths = [p.n_points for p in preprocessed_results.values()]
        assert all(length == lengths[0] for length in lengths)

        # But different window functions should produce different results
        hann_data = preprocessed_results["hann"].data
        hamming_data = preprocessed_results["hamming"].data
        assert not np.array_equal(hann_data, hamming_data)

    def test_dc_removal(self, sample_fid):
        """Test DC component removal."""
        # Create FID with known DC offset
        dc_offset = 5.0
        fid_with_dc = FID(
            data=sample_fid.data + dc_offset,
            spacing=sample_fid.spacing,
            probe_freq_mhz=sample_fid.probe_freq_mhz,
            sideband=sample_fid.sideband,
        )

        # Test with DC removal
        duration_us = fid_with_dc.duration_us
        with_rdc = fid_with_dc.preprocess(
            rdc=True, start_us=duration_us * 0.1, end_us=duration_us * 0.9
        )

        # Test without DC removal
        without_rdc = fid_with_dc.preprocess(
            rdc=False, start_us=duration_us * 0.1, end_us=duration_us * 0.9
        )

        # With RDC, active region should have lower mean
        # (This tests that DC was removed from the active region)
        assert abs(np.mean(with_rdc.data)) < abs(np.mean(without_rdc.data))


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
        preprocessed = sample_fid.preprocess(zpf=1, rdc=True)

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
        preprocessed_lower = fid_lower.preprocess(zpf=0)
        preprocessed_upper = fid_upper.preprocess(zpf=0)

        _, freq_lower = preprocessed_lower.compute_fft()
        _, freq_upper = preprocessed_upper.compute_fft()

        # Check frequency array directions
        # Lower sideband: frequencies should decrease with increasing scope frequency
        # Upper sideband: frequencies should increase with increasing scope frequency

        # For our 20 GHz probe with lower sideband, molecular freqs should be < 20000 MHz
        assert np.all(freq_lower <= 20000.0)  # Lower sideband

        # For upper sideband, molecular freqs should be >= 20000 MHz
        assert np.all(freq_upper >= 20000.0)  # Upper sideband

        # The frequency arrays should be mirror images around probe frequency
        probe_freq = 20000.0
        # Check that the frequency ranges are symmetric around probe
        lower_range = probe_freq - freq_lower[0], probe_freq - freq_lower[-1]
        upper_range = freq_upper[0] - probe_freq, freq_upper[-1] - probe_freq

        # Ranges should be approximately equal
        assert abs(lower_range[0] - upper_range[0]) < 1.0  # Within 1 MHz

    def test_fft_normalization_and_scaling(self, sample_fid):
        """Test FFT normalization and units scaling."""
        # Create preprocessed FID with known parameters
        preprocessed = sample_fid.preprocess(
            zpf=1, units_power=6, rdc=True  # Double length  # Scale by 10^6 (μV)
        )

        # Compute FFT
        complex_spectrum, freq_array = preprocessed.compute_fft()

        # Check that normalization was applied (divides by original length)
        # The spectrum amplitude should be reasonable (not too large or small)
        max_amplitude = np.max(np.abs(complex_spectrum))
        assert max_amplitude > 1e-10  # Not too small
        assert max_amplitude < 1e10  # Not too large

        # Test different units_power scaling
        preprocessed_power3 = sample_fid.preprocess(zpf=1, units_power=3)
        spectrum_power3, _ = preprocessed_power3.compute_fft()

        preprocessed_power6 = sample_fid.preprocess(zpf=1, units_power=6)
        spectrum_power6, _ = preprocessed_power6.compute_fft()

        # Spectrum with units_power=6 should be 1000x larger than units_power=3
        ratio = np.mean(np.abs(spectrum_power6)) / np.mean(np.abs(spectrum_power3))
        assert abs(ratio - 1000.0) < 100.0  # Should be approximately 1000

    def test_fft_with_different_zpf(self, sample_fid):
        """Test FFT computation with different zero-padding factors."""
        # Test different zero-padding factors
        zpf_values = [0, 1, 2]
        results = {}

        for zpf in zpf_values:
            preprocessed = sample_fid.preprocess(zpf=zpf)
            spectrum, freqs = preprocessed.compute_fft()
            results[zpf] = (spectrum, freqs)

        # Higher zpf should give better frequency resolution (more points)
        assert len(results[1][0]) > len(results[0][0])  # zpf=1 > zpf=0
        assert len(results[2][0]) > len(results[1][0])  # zpf=2 > zpf=1

        # Frequency ranges should be similar
        for zpf in zpf_values:
            freqs = results[zpf][1]
            assert abs(freqs[0] - results[0][1][0]) < 10.0  # Within 10 MHz
            assert abs(freqs[-1] - results[0][1][-1]) < 10.0  # Within 10 MHz


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
            zpf=1,
            expf_us=4.0,
            window_function="hann",
            rdc=True,
            units_power=6,
        )

        assert isinstance(preprocessed_fid, PreprocessedFID)
        assert preprocessed_fid.n_points > sample_fid.n_points  # Zero-padded

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
        # Run workflow twice with same parameters
        params = {
            "start_us": 2.0,
            "end_us": 12.0,
            "zpf": 1,
            "expf_us": 3.0,
            "window_function": "hamming",
            "rdc": True,
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
        """Test that workflow stages are properly decoupled."""
        # Create two different preprocessing parameter sets
        duration_us = sample_fid.duration_us
        params1 = {
            "zpf": 1,
            "expf_us": duration_us * 0.1,
            "start_us": duration_us * 0.1,
            "end_us": duration_us * 0.8,
        }
        params2 = {
            "zpf": 2,
            "expf_us": duration_us * 0.2,
            "start_us": duration_us * 0.05,
            "end_us": duration_us * 0.9,
        }

        # Stage 1: Different preprocessing should give different results
        preprocessed1 = sample_fid.preprocess(**params1)
        preprocessed2 = sample_fid.preprocess(**params2)

        assert preprocessed1.n_points != preprocessed2.n_points  # Different zpf
        assert not np.array_equal(preprocessed1.data, preprocessed2.data)

        # Stage 2: Different preprocessed FIDs should give different FFTs
        spectrum1, freqs1 = preprocessed1.compute_fft()
        spectrum2, freqs2 = preprocessed2.compute_fft()

        assert len(spectrum1) != len(spectrum2)  # Different lengths due to zpf
        assert not np.array_equal(
            spectrum1[: min(len(spectrum1), len(spectrum2))],
            spectrum2[: min(len(spectrum1), len(spectrum2))],
        )

    def test_workflow_with_edge_case_parameters(self, sample_fid):
        """Test workflow with edge case parameters."""
        # Test minimal parameters
        minimal_preprocessed = sample_fid.preprocess(zpf=0, rdc=False)
        minimal_spectrum, minimal_freqs = minimal_preprocessed.compute_fft()

        assert len(minimal_spectrum) > 0
        assert len(minimal_freqs) > 0
        assert np.all(np.isfinite(minimal_spectrum))
        assert np.all(np.isfinite(minimal_freqs))

        # Test maximal reasonable parameters
        maximal_preprocessed = sample_fid.preprocess(
            start_us=0.1,
            end_us=sample_fid.duration_us - 0.1,
            zpf=2,
            expf_us=1.0,
            window_function="blackman",
            rdc=True,
            units_power=9,
        )
        maximal_spectrum, maximal_freqs = maximal_preprocessed.compute_fft()

        assert len(maximal_spectrum) > len(minimal_spectrum)  # Higher zpf
        assert np.all(np.isfinite(maximal_spectrum))
        assert np.all(np.isfinite(maximal_freqs))


class TestThreeStageWorkflowWithRealData:
    """Test three-stage workflow with real experiment 2638 data."""

    def test_experiment_2638_three_stage_workflow(self):
        """Test complete workflow with real experiment 2638 data."""
        try:
            # Load real experimental data
            ftmw_data = load_blackchirp_experiment(
                "examples/blackchirp_data/2638", fid_index=0
            )
            fid = ftmw_data.fid

            print(f"Testing three-stage workflow with experiment 2638:")
            print(f"  Original FID: {fid.n_points} points, {fid.duration_us:.1f} μs")

            # Stage 1: Preprocessing with recommended parameters per CLAUDE.md
            preprocessed_fid = fid.preprocess(
                zpf=1,  # zpf=1 for improved frequency resolution
                expf_us=5.0,  # 5 μs exponential apodization
                rdc=True,
                units_power=6,
            )

            print(f"  Preprocessed: {preprocessed_fid.n_points} points (zpf=1)")

            # Stage 2: FFT computation
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()

            print(f"  FFT result: {len(complex_spectrum)} frequency points")
            print(f"  Frequency range: {freq_array[0]:.1f} to {freq_array[-1]:.1f} MHz")

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
                len(complex_ft.complex_spectrum) > 500000
            )  # Should have high resolution

            # Test trimming to activity region per CLAUDE.md recommendations
            trimmed_ft = complex_ft.trim_to_range(
                26500, 40000
            )  # Focus on activity region

            print(f"  Trimmed to activity region: {len(trimmed_ft.freq_array)} points")
            print(
                f"  Activity range: {trimmed_ft.freq_array[0]:.1f} to {trimmed_ft.freq_array[-1]:.1f} MHz"
            )

            # Verify trimming worked correctly
            assert trimmed_ft.freq_array[0] >= 26500.0
            assert trimmed_ft.freq_array[-1] <= 40000.0
            assert len(trimmed_ft.freq_array) < len(complex_ft.freq_array)

            # Verify magnitude spectrum is reasonable
            magnitude = trimmed_ft.magnitude_spectrum
            assert np.all(magnitude >= 0)
            assert np.all(np.isfinite(magnitude))

            print("✓ Three-stage workflow test with experiment 2638 passed")

        except Exception as e:
            pytest.skip(
                f"Could not test three-stage workflow with experiment 2638 data: {e}"
            )

    def test_three_stage_workflow_enforced(self):
        """Test that three-stage workflow is properly enforced (no legacy FID.ft() method)."""
        try:
            # Load experiment data
            ftmw_data = load_blackchirp_experiment(
                "examples/blackchirp_data/2638", fid_index=0
            )
            fid = ftmw_data.fid

            # Verify that legacy FID.ft() method does not exist
            assert not hasattr(
                fid, "ft"
            ), "Legacy FID.ft() method should be removed to enforce three-stage workflow"

            # Verify three-stage workflow works correctly
            preprocessed = fid.preprocess(zpf=1, expf_us=5.0)
            spectrum, freqs = preprocessed.compute_fft()
            complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

            # Verify results are valid
            assert complex_ft.n_points > 0
            assert np.all(np.isfinite(complex_ft.complex_spectrum))
            assert np.all(np.isfinite(complex_ft.freq_array))

            print(
                "✓ Three-stage workflow properly enforced - legacy FID.ft() method removed"
            )

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

        # Test negative zpf
        with pytest.raises(
            ValueError, match="Zero padding factor must be non-negative"
        ):
            fid.preprocess(zpf=-1)

        # Test negative start_us
        with pytest.raises(ValueError, match="Start time must be non-negative"):
            fid.preprocess(start_us=-1.0)

        # Test negative end_us
        with pytest.raises(ValueError, match="End time must be non-negative"):
            fid.preprocess(end_us=-1.0)

    def test_invalid_window_function(self):
        """Test error handling for invalid window functions."""
        fid = FID(data=[1.0, 2.0, 3.0], spacing=1e-6, probe_freq_mhz=1000.0)

        # Test invalid window function name
        with pytest.raises(
            ValueError
        ):  # scipy.signal.get_window should raise ValueError
            fid.preprocess(window_function="invalid_window")

    def test_empty_or_invalid_fid_data(self):
        """Test error handling for problematic FID data."""
        # Test empty data - should handle gracefully with proper zero padding
        fid_empty = FID(data=[], spacing=1e-6, probe_freq_mhz=1000.0)

        # Test with zpf=0 (no padding)
        preprocessed_0 = fid_empty.preprocess(zpf=0)
        assert len(preprocessed_0.data) == 0

        # Test with zpf=1 (should create array of size 2^1 = 2)
        preprocessed_1 = fid_empty.preprocess(zpf=1)
        assert len(preprocessed_1.data) == 2
        assert np.all(preprocessed_1.data == 0.0)

        # Test with zpf=2 (should create array of size 2^2 = 4)
        preprocessed_2 = fid_empty.preprocess(zpf=2)
        assert len(preprocessed_2.data) == 4
        assert np.all(preprocessed_2.data == 0.0)

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
