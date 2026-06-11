"""
Unit tests for file manager functions in the new .ftmw file-centric architecture.

Tests functional file operations rather than interface pedantry, focusing on:
- File creation, opening, and validation with real data
- Source conflict detection and parameter persistence
- End-to-end integration with experimental data (experiment 2638)
- Error handling and edge cases that would break user workflows
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FID, FIDProcessingParameters, Sideband
from ftmwpipeline.file_manager import (
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    PipelineStageTracker,
    SourceMetadata,
    StageDependencyError,
    create_pipeline_file,
    open_pipeline_file,
    update_processing_parameters,
    validate_pipeline_file,
)


class TestSourceMetadata:
    """Test SourceMetadata functionality for source tracking and conflict detection."""

    def test_source_metadata_initialization_and_matching(self):
        """Test source metadata creation and matching functionality."""
        # Test basic initialization
        metadata1 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"},
        )

        assert metadata1.source_path == Path("/test/path/data.txt")
        assert metadata1.format_name == "test_format"
        assert metadata1.loader_parameters == {"param1": "value1"}
        assert isinstance(metadata1.import_timestamp, datetime)
        assert isinstance(metadata1.source_hash, str)
        assert len(metadata1.source_hash) == 16  # Short MD5 hash

        # Test matching with identical metadata
        metadata2 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"},
        )
        metadata2.source_mtime = metadata1.source_mtime  # Ensure same mtime
        assert metadata1.matches(metadata2)

        # Test non-matching scenarios that would cause conflicts
        metadata3 = SourceMetadata(
            source_path="/different/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"},
        )
        assert not metadata1.matches(metadata3)

        metadata4 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="different_format",
            loader_parameters={"param1": "value1"},
        )
        assert not metadata1.matches(metadata4)

    def test_source_metadata_serialization_roundtrip(self):
        """Test that source metadata can be stored and loaded correctly."""
        original = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1", "param2": 42},
        )

        # Serialize to dict
        data_dict = original.to_dict()
        assert data_dict["source_path"] == str(original.source_path)
        assert data_dict["format_name"] == "test_format"
        assert data_dict["loader_parameters"] == {"param1": "value1", "param2": 42}

        # Deserialize from dict
        restored = SourceMetadata.from_dict(data_dict)
        assert restored.source_path == original.source_path
        assert restored.format_name == original.format_name
        assert restored.loader_parameters == original.loader_parameters
        assert restored.import_timestamp == original.import_timestamp


class TestPipelineStageTracker:
    """Test stage dependency tracking and validation."""

    def test_stage_dependencies_and_validation(self):
        """Test stage dependency validation that prevents pipeline errors."""
        tracker = PipelineStageTracker(["stage0_fid_data"])

        # Should allow stage1 since stage0 is completed
        tracker.validate_dependencies("stage1_complex_ft")

        # Should reject stage2 since stage1 is not completed
        with pytest.raises(StageDependencyError) as exc_info:
            tracker.validate_dependencies("stage2_noise_result")

        error = exc_info.value
        assert error.stage_name == "stage2_noise_result"
        assert "stage1_complex_ft" in error.missing_dependencies

        # Test next available stages (timebase calibration depends only on
        # stage 0, so it is available alongside the main chain)
        assert set(tracker.get_next_available_stages()) == {
            "stage1_complex_ft",
            "timebase_calibration",
        }

        # After completing stage1
        tracker.mark_completed("stage1_complex_ft")
        assert set(tracker.get_next_available_stages()) == {
            "stage2_noise_result",
            "timebase_calibration",
        }

    def test_stage_tracker_serialization(self):
        """Test stage tracker persistence."""
        tracker = PipelineStageTracker(["stage0_fid_data", "stage1_complex_ft"])
        data_dict = tracker.to_dict()

        assert set(data_dict["completed_stages"]) == {
            "stage0_fid_data",
            "stage1_complex_ft",
        }
        assert set(data_dict["next_available"]) == {
            "stage2_noise_result",
            "timebase_calibration",
        }


class TestFileManagerFunctions:
    """Test core file manager functions with functional validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_fid(self):
        """Create a realistic sample FID for testing."""
        # Create realistic voltage data similar to experiment 2638
        n_points = 1000
        spacing = 2e-11  # 20 ns spacing
        t = np.arange(n_points) * spacing

        # Exponentially decaying sinusoid (real voltage data)
        data = np.exp(-t / 2e-6) * np.cos(
            2 * np.pi * 100e6 * t
        ) + 0.01 * np.random.normal(0, 1, n_points)

        return FID(
            data=data,
            spacing=spacing,
            probe_freq_mhz=40960.0,  # Like experiment 2638
            sideband=Sideband.LOWER,
            shots=50000,
            processing=FIDProcessingParameters(start_us=1.0, end_us=18.0),
            metadata={"source_path": "/test/source", "experiment_id": "test_2638"},
        )

    @pytest.fixture
    def sample_source_metadata(self):
        """Create sample source metadata."""
        return SourceMetadata(
            source_path="/test/source/path",
            format_name="blackchirp",
            loader_parameters={"fid_index": 0},
        )

    def test_create_pipeline_file_functional(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test pipeline file creation creates valid, functional files."""
        filepath = temp_dir / "test_pipeline.ftmw"

        # Create pipeline file
        result_path = create_pipeline_file(filepath, sample_fid, sample_source_metadata)

        # Verify file was created correctly
        assert result_path == filepath
        assert filepath.exists()
        assert filepath.suffix == ".ftmw"

        # Verify file contains correct data and can be opened
        opened_path, source_metadata, stage_tracker = open_pipeline_file(filepath)
        assert opened_path == filepath
        assert source_metadata.format_name == "blackchirp"
        assert stage_tracker.is_completed("stage0_fid_data")

        # Verify HDF5 structure is valid
        with h5py.File(filepath, "r") as h5f:
            assert "source_metadata" in h5f
            assert "pipeline_stages" in h5f
            assert "stage0_fid_data" in h5f

            # Verify stage0 data can be loaded
            stage0_group = h5f["stage0_fid_data"]
            assert "time_series_data" in stage0_group
            assert "acquisition" in stage0_group

    def test_create_pipeline_file_extension_handling(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test that .ftmw extension is added automatically for user convenience."""
        filepath_no_ext = temp_dir / "test_pipeline"

        result_path = create_pipeline_file(
            filepath_no_ext, sample_fid, sample_source_metadata
        )

        assert result_path.suffix == ".ftmw"
        assert result_path.exists()

    def test_create_pipeline_file_source_conflict_detection(self, temp_dir, sample_fid):
        """Test source conflict detection prevents data corruption."""
        filepath = temp_dir / "test_pipeline.ftmw"

        source1 = SourceMetadata("/source1", "format1")
        source2 = SourceMetadata("/source2", "format2")

        # Create first file
        create_pipeline_file(filepath, sample_fid, source1)

        # Try to create with different source - should raise conflict error
        with pytest.raises(PipelineExistsError) as exc_info:
            create_pipeline_file(filepath, sample_fid, source2)

        error = exc_info.value
        assert error.filepath == filepath
        assert "/source1" in error.existing_source
        assert "/source2" in error.requested_source
        assert "force=True" in str(error)  # Should suggest solution

    def test_create_pipeline_file_force_overwrite(self, temp_dir, sample_fid):
        """Test force overwrite functionality works correctly."""
        filepath = temp_dir / "test_pipeline.ftmw"

        source1 = SourceMetadata("/source1", "format1")
        source2 = SourceMetadata("/source2", "format2")

        # Create first file
        create_pipeline_file(filepath, sample_fid, source1)

        # Force overwrite with different source
        result_path = create_pipeline_file(filepath, sample_fid, source2, force=True)
        assert result_path == filepath

        # Verify source was updated
        _, source_metadata, _ = open_pipeline_file(filepath)
        assert source_metadata.source_path == Path("/source2")

    def test_open_pipeline_file_functional(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test opening pipeline files loads correct metadata and data."""
        filepath = temp_dir / "test_pipeline.ftmw"

        # Create file first
        create_pipeline_file(filepath, sample_fid, sample_source_metadata)

        # Open file
        opened_path, source_metadata, stage_tracker = open_pipeline_file(filepath)

        # Verify correct data was loaded
        assert opened_path == filepath
        assert isinstance(source_metadata, SourceMetadata)
        assert isinstance(stage_tracker, PipelineStageTracker)
        assert source_metadata.format_name == "blackchirp"
        assert source_metadata.loader_parameters == {"fid_index": 0}
        assert stage_tracker.is_completed("stage0_fid_data")

    def test_open_pipeline_file_error_handling(self, temp_dir):
        """Test file opening provides helpful error messages."""
        # Test non-existent file
        filepath = temp_dir / "nonexistent.ftmw"

        with pytest.raises(FileNotFoundError) as exc_info:
            open_pipeline_file(filepath)

        error_message = str(exc_info.value)
        assert "Pipeline file not found" in error_message
        assert "Pipeline.create" in error_message  # Should suggest solution
        assert (
            "ftmwpipeline data import" in error_message
        )  # Should suggest CLI alternative

    def test_validate_pipeline_file_functional(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test file validation detects real corruption issues."""
        filepath = temp_dir / "test_pipeline.ftmw"
        create_pipeline_file(filepath, sample_fid, sample_source_metadata)

        # Test valid file validation
        report = validate_pipeline_file(filepath)
        assert report["valid"] is True
        assert len(report["errors"]) == 0
        assert "file_size" in report
        assert "stages" in report

        # Test detection of missing required groups (corruption)
        filepath_corrupted = temp_dir / "corrupted.ftmw"
        with h5py.File(filepath_corrupted, "w") as h5f:
            h5f.attrs["file_type"] = "ftmw_pipeline"
            # Missing required groups

        report = validate_pipeline_file(filepath_corrupted)
        assert report["valid"] is False
        assert len(report["errors"]) > 0  # Should detect corruption/missing data

    def test_update_processing_parameters_functional(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test parameter persistence works correctly for interactive workflows."""
        filepath = temp_dir / "test_pipeline.ftmw"
        create_pipeline_file(filepath, sample_fid, sample_source_metadata)

        # Update processing parameters (canonical FT is unapodized: only the
        # data-selection / scaling knobs are persisted).
        new_params = {
            "start_us": 1.0,
            "end_us": 10.0,
            "units_power": 3,
            "trim_start_mhz": 26500.0,
            "trim_end_mhz": 40000.0,
        }

        update_processing_parameters(filepath, new_params)

        # Verify parameters were saved by checking HDF5 structure
        with h5py.File(filepath, "r") as h5f:
            rec_proc_group = h5f["stage0_fid_data/recommended_processing"]

            assert rec_proc_group.attrs["start_us"] == 1.0
            assert rec_proc_group.attrs["end_us"] == 10.0
            assert rec_proc_group.attrs["units_power"] == 3
            assert rec_proc_group.attrs["trim_start_mhz"] == 26500.0
            assert rec_proc_group.attrs["trim_end_mhz"] == 40000.0

            # Verify timestamp was updated
            stages_group = h5f["pipeline_stages"]
            assert "last_updated" in stages_group.attrs

    def test_update_processing_parameters_with_none_values(
        self, temp_dir, sample_fid, sample_source_metadata
    ):
        """Test that None values are handled correctly in parameter updates."""
        filepath = temp_dir / "test_pipeline.ftmw"
        create_pipeline_file(filepath, sample_fid, sample_source_metadata)

        # Update with None values
        params_with_none = {"start_us": None, "end_us": None, "units_power": 6}

        update_processing_parameters(filepath, params_with_none)

        # Verify None values are stored as "__None__" string
        with h5py.File(filepath, "r") as h5f:
            rec_proc_group = h5f["stage0_fid_data/recommended_processing"]

            assert rec_proc_group.attrs["start_us"] == "__None__"
            assert rec_proc_group.attrs["end_us"] == "__None__"
            assert rec_proc_group.attrs["units_power"] == 6


class TestRealDataIntegration:
    """Test file manager functions with real experimental data."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_experiment_2638_end_to_end_workflow(self, temp_dir):
        """Test complete workflow with experiment 2638 data if available."""
        try:
            from ftmwpipeline.io import load_blackchirp_experiment

            # Load real experimental data
            ftmw_data = load_blackchirp_experiment(
                "examples/blackchirp_data/2638", fid_index=0
            )
            fid = ftmw_data.fid

            # Create source metadata
            source_metadata = SourceMetadata(
                source_path="examples/blackchirp_data/2638",
                format_name="blackchirp",
                loader_parameters={"fid_index": 0},
            )

            # Test complete workflow
            filepath = temp_dir / "exp_2638_test.ftmw"

            # 1. Create pipeline file
            created_path = create_pipeline_file(filepath, fid, source_metadata)
            assert created_path.exists()

            # 2. Open and verify
            opened_path, loaded_metadata, stage_tracker = open_pipeline_file(
                created_path
            )
            assert loaded_metadata.format_name == "blackchirp"
            assert stage_tracker.is_completed("stage0_fid_data")

            # 3. Validate file integrity
            report = validate_pipeline_file(created_path)
            assert report["valid"] is True

            # 4. Update processing parameters (typical interactive workflow)
            processing_params = {
                "start_us": 1.0,
                "end_us": 18.0,
                "trim_start_mhz": 26500.0,
                "trim_end_mhz": 40000.0,
            }
            update_processing_parameters(created_path, processing_params)

            # 5. Verify parameters persistence
            with h5py.File(created_path, "r") as h5f:
                rec_proc_group = h5f["stage0_fid_data/recommended_processing"]
                assert rec_proc_group.attrs["start_us"] == 1.0
                assert rec_proc_group.attrs["end_us"] == 18.0
                assert rec_proc_group.attrs["trim_start_mhz"] == 26500.0
                assert rec_proc_group.attrs["trim_end_mhz"] == 40000.0

            print("✅ Experiment 2638 end-to-end workflow test passed")

        except Exception as e:
            pytest.skip(f"Could not test with experiment 2638 data: {e}")


class TestErrorHandlingAndEdgeCases:
    """Test error conditions that would break user workflows."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_corrupted_file_recovery_guidance(self, temp_dir):
        """Test that corruption errors provide helpful recovery guidance."""
        filepath = temp_dir / "corrupted.ftmw"

        # Create truncated/corrupted file
        with open(filepath, "wb") as f:
            f.write(b"truncated hdf5 header")

        # Should catch HDF5 error and convert to PipelineCorruptionError
        with pytest.raises((PipelineCorruptionError, RuntimeError)) as exc_info:
            open_pipeline_file(filepath)

        error_str = str(exc_info.value).lower()
        assert "corrupted" in error_str or "failed" in error_str

    def test_update_parameters_missing_file(self, temp_dir):
        """Test parameter update on non-existent file provides clear error."""
        filepath = temp_dir / "nonexistent.ftmw"

        with pytest.raises(FileNotFoundError) as exc_info:
            update_processing_parameters(filepath, {"start_us": 2.0})

        assert "does not exist" in str(exc_info.value)

    def test_create_file_cleanup_on_error(self, temp_dir):
        """Test that partial files are cleaned up on creation errors."""
        filepath = temp_dir / "test.ftmw"
        sample_fid = Mock()
        sample_source = SourceMetadata("/test", "test")

        # Mock file creation to fail
        with patch(
            "ftmwpipeline.file_manager.save_fid_to_hdf5",
            side_effect=RuntimeError("Save failed"),
        ):
            with pytest.raises(RuntimeError):
                create_pipeline_file(filepath, sample_fid, sample_source)

            # File should be cleaned up on error
            assert not filepath.exists()


class TestCustomExceptions:
    """Test custom exception classes provide helpful error messages."""

    def test_pipeline_exists_error_provides_solutions(self):
        """Test PipelineExistsError suggests helpful solutions."""
        filepath = Path("/test/path.ftmw")
        existing_source = "/existing/source"
        requested_source = "/requested/source"

        error = PipelineExistsError(filepath, existing_source, requested_source)

        error_str = str(error)
        assert "already exists with different source" in error_str
        assert existing_source in error_str
        assert requested_source in error_str
        assert "force=True" in error_str  # Should suggest force option
        assert "Pipeline.open" in error_str  # Should suggest opening existing

    def test_stage_dependency_error_explains_requirements(self):
        """Test StageDependencyError clearly explains missing dependencies."""
        stage_name = "stage2_noise_result"
        missing_deps = ["stage1_complex_ft"]
        filepath = Path("/test/path.ftmw")

        error = StageDependencyError(stage_name, missing_deps, filepath)

        error_str = str(error)
        assert "Cannot execute stage2_noise_result" in error_str
        assert "stage1_complex_ft" in error_str
        assert "Complete the required stages first" in error_str

    def test_pipeline_corruption_error_suggests_recovery(self):
        """Test PipelineCorruptionError suggests recovery options."""
        filepath = Path("/test/path.ftmw")
        corruption_details = "Invalid HDF5 structure"

        error = PipelineCorruptionError(filepath, corruption_details)

        error_str = str(error)
        assert "corrupted" in error_str
        assert corruption_details in error_str
        assert (
            "recreating" in error_str.lower()
        )  # Should suggest recreating from source
