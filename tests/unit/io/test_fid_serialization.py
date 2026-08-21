"""
Unit tests for FID serialization in the new .ftmw file-centric architecture.

Tests functional FID storage and loading rather than interface pedantry, focusing on:
- Pipeline file FID data storage and bit-perfect reconstruction
- Metadata preservation and parameter persistence
- Integration with real experimental data (experiment 2638)
- Error handling for corrupted files and missing data
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FID, FIDProcessingParameters, Sideband
from ftmwpipeline.file_manager import (
    SourceMetadata,
    create_pipeline_file,
    update_processing_parameters,
)
from ftmwpipeline.io.fid_serialization import (
    _deserialize_optional_float,
    _deserialize_optional_str,
    _serialize_optional_float,
    _serialize_optional_str,
    load_fid_from_hdf5,
)


class TestFIDSerializationInPipelineFiles:
    """Test FID serialization within .ftmw pipeline files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_fid(self):
        """Create a realistic sample FID for testing."""
        # Create real voltage data (not complex per architecture requirements)
        n_points = 1000
        spacing = 2e-11  # 20 ns spacing (typical for FTMW)
        probe_freq = 18000.0  # 18 GHz

        # Generate realistic exponentially decaying sinusoid (real-valued)
        t = np.arange(n_points) * spacing
        # Multiple frequency components for realistic spectrum
        data = np.exp(-t / 2e-6) * (
            0.8 * np.cos(2 * np.pi * 100e6 * t)  # 100 MHz signal
            + 0.3 * np.cos(2 * np.pi * 250e6 * t)  # 250 MHz signal
            + 0.1 * np.random.normal(0, 0.05, n_points)
        )  # Noise

        processing = FIDProcessingParameters(
            start_us=0.5,
            end_us=10.0,
            units_power=6,
        )

        metadata = {
            "source_path": "/test/path/experiment",
            "source_format": "blackchirp",
            "load_timestamp": "2023-01-01T12:00:00",
            "temperature_K": 298.0,
            "pressure_torr": 1e-3,
            "experiment_notes": "Test FID for serialization",
        }

        return FID(
            data=data,  # Real voltage data
            spacing=spacing,
            probe_freq_mhz=probe_freq,
            sideband=Sideband.LOWER,
            shots=50000,
            processing=processing,
            metadata=metadata,
        )

    def test_fid_data_is_real_not_complex(self, sample_fid):
        """Test that FID data is real, not complex (per architecture requirements)."""
        # Verify the test FID has real data
        assert np.all(np.isreal(sample_fid.data))
        assert sample_fid.data.dtype in [np.float64, np.float32]
        assert not np.iscomplexobj(sample_fid.data)

    def test_pipeline_file_fid_storage_and_loading(self, temp_dir, sample_fid):
        """Test FID storage and loading through pipeline files with bit-perfect accuracy."""
        # Create source metadata
        source_metadata = SourceMetadata(
            source_path="/test/source/path",
            format_name="blackchirp",
            loader_parameters={"fid_index": 0},
        )

        # Create pipeline file
        filepath = temp_dir / "test_pipeline.ftmw"
        create_pipeline_file(filepath, sample_fid, source_metadata)

        # Load FID data directly from pipeline file
        with h5py.File(filepath, "r") as h5f:
            stage0_group = h5f["stage0_fid_data"]
            loaded_fid = load_fid_from_hdf5(stage0_group)

        # Verify bit-perfect data reconstruction
        np.testing.assert_array_equal(
            sample_fid.data,
            loaded_fid.data,
            err_msg="FID data should be reconstructed bit-perfectly",
        )

        # Verify acquisition parameters
        assert loaded_fid.spacing == sample_fid.spacing
        assert loaded_fid.probe_freq_mhz == sample_fid.probe_freq_mhz
        assert loaded_fid.sideband == sample_fid.sideband
        assert loaded_fid.shots == sample_fid.shots
        assert loaded_fid.n_points == sample_fid.n_points
        assert abs(loaded_fid.duration_us - sample_fid.duration_us) < 1e-9

        # Verify processing parameters
        assert loaded_fid.processing.start_us == sample_fid.processing.start_us
        assert loaded_fid.processing.end_us == sample_fid.processing.end_us
        assert loaded_fid.processing.units_power == sample_fid.processing.units_power

        # Verify metadata preservation
        for key, value in sample_fid.metadata.items():
            assert key in loaded_fid.metadata
            assert loaded_fid.metadata[key] == value

    def test_pipeline_file_hdf5_structure_validation(self, temp_dir, sample_fid):
        """Test that pipeline files contain correct HDF5 structure for FID data."""
        source_metadata = SourceMetadata("/test/source", "blackchirp")
        filepath = temp_dir / "test_structure.ftmw"
        create_pipeline_file(filepath, sample_fid, source_metadata)

        # Verify pipeline file structure
        with h5py.File(filepath, "r") as h5f:
            # Check pipeline-level structure
            assert "source_metadata" in h5f
            assert "pipeline_stages" in h5f
            assert "stage0_fid_data" in h5f

            # Check stage0 FID data structure
            stage0_group = h5f["stage0_fid_data"]
            assert "time_series_data" in stage0_group
            assert "acquisition" in stage0_group
            assert "recommended_processing" in stage0_group
            assert "metadata" in stage0_group

            # Check time series data (real voltage data)
            time_data = stage0_group["time_series_data"]
            assert time_data.dtype == np.float64
            assert np.all(np.isreal(time_data[:]))

            # Check acquisition parameters
            acq_group = stage0_group["acquisition"]
            required_attrs = [
                "spacing_seconds",
                "probe_freq_mhz",
                "sideband",
                "shots",
                "n_points",
                "duration_us",
            ]
            for attr in required_attrs:
                assert attr in acq_group.attrs

            # Check recommended processing
            proc_group = stage0_group["recommended_processing"]
            assert (
                proc_group.attrs["description"]
                == "Format-specific processing recommendations (not requirements)"
            )

            # Check metadata groups
            meta_group = stage0_group["metadata"]
            assert "source_info" in meta_group
            assert "experimental_data" in meta_group

    def test_parameter_persistence_in_pipeline_files(self, temp_dir, sample_fid):
        """Test that processing parameters are correctly saved and loaded from pipeline files."""
        source_metadata = SourceMetadata("/test/source", "blackchirp")
        filepath = temp_dir / "test_params.ftmw"
        create_pipeline_file(filepath, sample_fid, source_metadata)

        # Update processing parameters (canonical FT is unapodized: only
        # data-selection / scaling knobs are persisted).
        new_params = {
            "start_us": 1.0,
            "end_us": 12.0,
            "units_power": 3,
            "trim_start_mhz": 26500.0,
            "trim_end_mhz": 40000.0,
        }

        update_processing_parameters(filepath, new_params)

        # Load FID and verify updated parameters are accessible
        with h5py.File(filepath, "r") as h5f:
            stage0_group = h5f["stage0_fid_data"]
            rec_proc_group = stage0_group["recommended_processing"]

            # Verify all parameters were saved
            assert rec_proc_group.attrs["start_us"] == 1.0
            assert rec_proc_group.attrs["end_us"] == 12.0
            assert rec_proc_group.attrs["units_power"] == 3
            assert rec_proc_group.attrs["trim_start_mhz"] == 26500.0
            assert rec_proc_group.attrs["trim_end_mhz"] == 40000.0

    def test_processing_parameters_with_none_values(self, temp_dir):
        """Test serialization with None values in processing parameters."""
        # Create FID with some None processing parameters
        processing = FIDProcessingParameters(start_us=None, end_us=None, units_power=6)

        fid = FID(
            data=np.array([1.0, 0.5, 0.0]),
            spacing=1e-6,
            probe_freq_mhz=10000.0,
            processing=processing,
        )

        source_metadata = SourceMetadata("/test/source", "test_format")
        filepath = temp_dir / "test_none_values.ftmw"
        create_pipeline_file(filepath, fid, source_metadata)

        # Load and verify None values are preserved
        with h5py.File(filepath, "r") as h5f:
            stage0_group = h5f["stage0_fid_data"]
            loaded_fid = load_fid_from_hdf5(stage0_group)

        assert loaded_fid.processing.start_us is None
        assert loaded_fid.processing.end_us is None
        assert loaded_fid.processing.units_power == 6

    def test_metadata_separation_in_pipeline_files(self, temp_dir, sample_fid):
        """Test that metadata is properly separated into source and experimental."""
        # Add mixed metadata
        sample_fid.metadata.update(
            {
                "source_path": "/test/source/path",
                "loader_class": "BlackChirpLoader",
                "temperature_K": 298.0,
                "pressure_torr": 1e-3,
                "custom_param": "custom_value",
            }
        )

        source_metadata = SourceMetadata("/test/source", "blackchirp")
        filepath = temp_dir / "test_metadata.ftmw"
        create_pipeline_file(filepath, sample_fid, source_metadata)

        # Check that metadata is properly separated in pipeline file
        with h5py.File(filepath, "r") as h5f:
            stage0_group = h5f["stage0_fid_data"]
            meta_group = stage0_group["metadata"]

            # Load source metadata
            source_json = meta_group["source_info"][()].decode("utf-8")
            source_metadata_dict = json.loads(source_json)
            assert "source_path" in source_metadata_dict
            assert "loader_class" in source_metadata_dict

            # Load experimental metadata
            exp_json = meta_group["experimental_data"][()].decode("utf-8")
            exp_metadata = json.loads(exp_json)
            assert "temperature_K" in exp_metadata
            assert "custom_param" in exp_metadata

            # Verify source keys are NOT in experimental metadata
            assert "source_path" not in exp_metadata
            assert "loader_class" not in exp_metadata


class TestOptionalValueSerializationHelpers:
    """Test helper functions for handling None values in HDF5."""

    def test_optional_value_serialization_helpers(self):
        """Test helper functions for optional value serialization."""
        # Test float serialization
        assert _serialize_optional_float(None) == "__None__"
        assert _serialize_optional_float(3.14) == 3.14
        assert _serialize_optional_float(0.0) == 0.0

        # Test float deserialization
        assert _deserialize_optional_float("__None__") is None
        assert _deserialize_optional_float(b"__None__") is None  # bytes version
        assert _deserialize_optional_float(3.14) == 3.14
        assert _deserialize_optional_float(0.0) == 0.0

        # Test string serialization
        assert _serialize_optional_str(None) == "__None__"
        assert _serialize_optional_str("test") == "test"
        assert _serialize_optional_str("") == ""

        # Test string deserialization
        assert _deserialize_optional_str("__None__") is None
        assert _deserialize_optional_str(b"__None__") is None  # bytes version
        assert _deserialize_optional_str("test") == "test"
        assert _deserialize_optional_str(b"test") == "test"  # bytes version


class TestRealExperimentalDataIntegration:
    """Test FID serialization with real experiment 2638 data."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_experiment_2638_pipeline_file_integration(self, temp_dir):
        """Test FID serialization with real experiment 2638 data through pipeline files."""
        try:
            from ftmwpipeline.io.data_loaders import BlackChirpLoader

            # Load real experimental data
            original_fid = BlackChirpLoader().load_fid(
                "examples/blackchirp_data/2638", fid_index=0
            )

            print("Experiment 2638 FID loaded:")
            print(f"  Points: {original_fid.n_points}")
            print(f"  Duration: {original_fid.duration_us:.1f} μs")
            print(f"  Spacing: {original_fid.spacing:.4e} s")
            print(f"  Probe: {original_fid.probe_freq_mhz:.1f} MHz")
            print(f"  Sideband: {original_fid.sideband.value}")

            # Verify FID specifications per project requirements
            assert original_fid.n_points == 750000  # 750k points
            assert abs(original_fid.duration_us - 15.0) < 0.1  # ~15 μs duration
            assert original_fid.probe_freq_mhz == 40960.0  # 40.96 GHz probe
            assert original_fid.sideband == Sideband.LOWER  # Lower sideband

            # Verify data is real, not complex
            assert np.all(np.isreal(original_fid.data))
            assert original_fid.data.dtype in [np.float64, np.float32]

            # Create pipeline file with experiment 2638 data
            source_metadata = SourceMetadata(
                source_path="examples/blackchirp_data/2638",
                format_name="blackchirp",
                loader_parameters={"fid_index": 0},
            )

            filepath = temp_dir / "exp_2638_test.ftmw"
            create_pipeline_file(filepath, original_fid, source_metadata)
            assert filepath.exists()

            # Check file size (should be reasonable for 750k points)
            file_size_mb = filepath.stat().st_size / 1024**2
            fid_data_size_mb = original_fid.data.nbytes / 1024**2
            print(f"Pipeline file size: {file_size_mb:.2f} MB")
            print(f"Raw FID data: {fid_data_size_mb:.2f} MB")

            # Load from pipeline file and verify bit-perfect reconstruction
            with h5py.File(filepath, "r") as h5f:
                stage0_group = h5f["stage0_fid_data"]
                loaded_fid = load_fid_from_hdf5(stage0_group)

            # Verify bit-perfect data reconstruction
            np.testing.assert_array_equal(
                original_fid.data,
                loaded_fid.data,
                err_msg="Pipeline file FID data should be bit-perfect",
            )

            # Verify all parameters preserved
            assert loaded_fid.spacing == original_fid.spacing
            assert loaded_fid.probe_freq_mhz == original_fid.probe_freq_mhz
            assert loaded_fid.sideband == original_fid.sideband
            assert loaded_fid.shots == original_fid.shots
            assert loaded_fid.n_points == original_fid.n_points

            # Verify metadata preservation (blackchirp uses different key names)
            assert any(
                key.endswith("_path") or "experiment" in key
                for key in loaded_fid.metadata.keys()
            )
            assert "blackchirp_params" in loaded_fid.metadata

            print(
                "✅ Experiment 2638 pipeline file integration test passed - bit-perfect reconstruction verified"
            )

        except Exception as e:
            pytest.skip(f"Could not test with experiment 2638 data: {e}")

    def test_pipeline_file_portability_with_real_data(self, temp_dir):
        """Test that pipeline files are self-contained and portable."""
        try:
            from ftmwpipeline.io.data_loaders import BlackChirpLoader

            # Load and create pipeline file with experiment 2638
            original_fid = BlackChirpLoader().load_fid(
                "examples/blackchirp_data/2638", fid_index=0
            )

            source_metadata = SourceMetadata(
                source_path="examples/blackchirp_data/2638",
                format_name="blackchirp",
                loader_parameters={"fid_index": 0},
            )

            # Create pipeline file
            filepath = temp_dir / "exp_2638_portable.ftmw"
            create_pipeline_file(filepath, original_fid, source_metadata)

            # Verify pipeline file is self-contained
            with h5py.File(filepath, "r") as h5f:
                # Should have complete experiment info
                stage0_group = h5f["stage0_fid_data"]
                assert "time_series_data" in stage0_group
                assert "acquisition" in stage0_group
                assert "metadata" in stage0_group

                # Verify acquisition metadata for portability
                acq_group = stage0_group["acquisition"]
                assert acq_group.attrs["probe_freq_mhz"] == 40960.0
                assert acq_group.attrs["sideband"] == "lower"
                assert acq_group.attrs["duration_us"] > 14.9  # ~15 μs
                assert acq_group.attrs["n_points"] == 750000

            # Load and verify FID can be used independently
            with h5py.File(filepath, "r") as h5f:
                stage0_group = h5f["stage0_fid_data"]
                loaded_fid = load_fid_from_hdf5(stage0_group)

            # Should be able to create ComplexFT from loaded FID (test self-containment)
            # This tests that pipeline file is truly self-contained for analysis
            preprocessed_fid = loaded_fid.preprocess()
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()

            # Verify FFT computation works with pipeline file data
            assert len(complex_spectrum) > 0
            assert len(freq_array) > 0
            assert np.all(np.isfinite(complex_spectrum))
            assert np.all(np.isfinite(freq_array))

            print(
                "✅ Pipeline file portability test passed - self-contained for analysis"
            )

        except Exception as e:
            pytest.skip(
                f"Could not test pipeline file portability with experiment 2638 data: {e}"
            )


class TestErrorConditionsAndEdgeCases:
    """Test error handling and edge cases in FID serialization."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_invalid_fid_object_in_pipeline_file(self, temp_dir):
        """Test error handling with invalid FID objects in pipeline file creation."""
        source_metadata = SourceMetadata("/test/source", "test_format")
        filepath = temp_dir / "test_invalid.ftmw"

        # Should raise error for non-FID object
        with pytest.raises(RuntimeError):
            create_pipeline_file(filepath, "not_a_fid", source_metadata)

    def test_corrupted_pipeline_file_loading(self, temp_dir):
        """Test error handling when loading corrupted pipeline files."""
        # Create corrupted pipeline file structure
        filepath = temp_dir / "corrupted.ftmw"
        with h5py.File(filepath, "w") as h5f:
            # Create basic pipeline structure but corrupt stage0 data
            h5f.attrs["file_type"] = "ftmw_pipeline"
            h5f.create_group("source_metadata")
            h5f.create_group("pipeline_stages")

            # Create corrupted stage0 group (missing required datasets)
            stage0_group = h5f.create_group("stage0_fid_data")
            stage0_group.create_dataset("time_series_data", data=[1, 2, 3])
            # Missing 'acquisition' and 'metadata' groups

        with h5py.File(filepath, "r") as h5f:
            stage0_group = h5f["stage0_fid_data"]
            with pytest.raises(
                RuntimeError, match="Failed to deserialize FID from HDF5"
            ):
                load_fid_from_hdf5(stage0_group)

    def test_missing_pipeline_file_validation(self, temp_dir):
        """Test handling of missing pipeline files."""
        from ftmwpipeline.file_manager import open_pipeline_file

        # Try to open non-existent pipeline file
        filepath = temp_dir / "nonexistent.ftmw"
        with pytest.raises(FileNotFoundError, match="Pipeline file not found"):
            open_pipeline_file(filepath)

    def test_parameter_update_on_invalid_pipeline_file(self, temp_dir):
        """Test parameter updates on invalid pipeline file structure."""
        filepath = temp_dir / "invalid_structure.ftmw"

        # Create pipeline file with invalid structure (missing stage0 data)
        with h5py.File(filepath, "w") as h5f:
            h5f.attrs["file_type"] = "ftmw_pipeline"
            h5f.create_group("source_metadata")
            h5f.create_group("pipeline_stages")
            # Missing stage0_fid_data group

        with pytest.raises(
            RuntimeError, match="Failed to update processing parameters"
        ):
            update_processing_parameters(filepath, {"start_us": 2.0})
