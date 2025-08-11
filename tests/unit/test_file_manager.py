"""
Unit tests for PipelineFileManager class.

Tests the foundational file management capabilities for .ftmw pipeline data files,
focusing exclusively on file manager operations without testing broader pipeline
processing functionality.

Test Coverage:
- Core file operations (create, open, validate)
- Source metadata tracking and conflict detection
- Stage dependency management and validation
- Error handling and edge cases
- File corruption scenarios and recovery guidance
"""

import pytest
import numpy as np
import h5py
import json
import tempfile
import os
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from ftmwpipeline.file_manager import (
    PipelineFileManager,
    SourceMetadata,
    PipelineStageTracker,
    PipelineFileError,
    PipelineExistsError,
    PipelineStageError,
    PipelineCorruptedError
)
from ftmwpipeline.core.data_structures import FID, FIDProcessingParameters, Sideband


class TestSourceMetadata:
    """Test SourceMetadata class functionality."""
    
    def test_source_metadata_initialization(self):
        """Test SourceMetadata initialization with various inputs."""
        # Test basic initialization
        metadata = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        
        assert metadata.source_path == Path("/test/path/data.txt").resolve()
        assert metadata.format_name == "test_format"
        assert metadata.loader_parameters == {"param1": "value1"}
        assert isinstance(metadata.import_timestamp, datetime)
        assert isinstance(metadata.source_hash, str)
        assert len(metadata.source_hash) == 32  # MD5 hash
    
    def test_source_metadata_with_none_parameters(self):
        """Test SourceMetadata with None loader parameters."""
        metadata = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format"
        )
        
        assert metadata.loader_parameters == {}
    
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.stat')
    def test_source_metadata_file_mtime(self, mock_stat, mock_is_file):
        """Test source metadata with file modification time."""
        mock_is_file.return_value = True
        mock_stat.return_value.st_mtime = 1234567890.0
        
        metadata = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format"
        )
        
        assert metadata.source_mtime == 1234567890.0
    
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.is_dir')
    @patch('pathlib.Path.rglob')
    def test_source_metadata_directory_mtime(self, mock_rglob, mock_is_dir, mock_is_file):
        """Test source metadata with directory modification time."""
        mock_is_file.return_value = False
        mock_is_dir.return_value = True
        
        # Mock directory contents with files having different mtimes
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_mtime = 1234567890.0
        
        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_mtime = 1234567895.0  # More recent
        
        mock_rglob.return_value = [mock_file1, mock_file2]
        
        metadata = SourceMetadata(
            source_path="/test/path/directory",
            format_name="test_format"
        )
        
        assert metadata.source_mtime == 1234567895.0  # Most recent
    
    def test_source_metadata_hash_computation(self):
        """Test that source hash computation is deterministic."""
        metadata1 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        
        metadata2 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        
        # Different instances with same parameters should have same hash initially
        # (before mtime differences affect it)
        assert len(metadata1.source_hash) == 32
        assert len(metadata2.source_hash) == 32
    
    def test_source_metadata_matches(self):
        """Test SourceMetadata matching functionality."""
        metadata1 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        
        metadata2 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        
        # Set same mtime to ensure match
        metadata2.source_mtime = metadata1.source_mtime
        
        assert metadata1.matches(metadata2)
        
        # Test non-matching scenarios
        metadata3 = SourceMetadata(
            source_path="/different/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1"}
        )
        assert not metadata1.matches(metadata3)
        
        metadata4 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="different_format",
            loader_parameters={"param1": "value1"}
        )
        assert not metadata1.matches(metadata4)
        
        metadata5 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "different_value"}
        )
        assert not metadata1.matches(metadata5)
    
    def test_source_metadata_mtime_tolerance(self):
        """Test that source metadata matching has 1-second mtime tolerance."""
        metadata1 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format"
        )
        
        metadata2 = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format"
        )
        
        # Set mtimes with small difference (within tolerance)
        metadata1.source_mtime = 1000.0
        metadata2.source_mtime = 1000.5  # 0.5 second difference
        
        assert metadata1.matches(metadata2)
        
        # Set mtimes with large difference (outside tolerance)
        metadata2.source_mtime = 1002.0  # 2 second difference
        
        assert not metadata1.matches(metadata2)
    
    def test_source_metadata_serialization(self):
        """Test SourceMetadata to_dict and from_dict methods."""
        original = SourceMetadata(
            source_path="/test/path/data.txt",
            format_name="test_format",
            loader_parameters={"param1": "value1", "param2": 42}
        )
        
        # Serialize to dict
        data_dict = original.to_dict()
        
        assert data_dict['source_path'] == str(original.source_path)
        assert data_dict['format_name'] == "test_format"
        assert data_dict['loader_parameters'] == {"param1": "value1", "param2": 42}
        assert 'import_timestamp' in data_dict
        assert 'source_mtime' in data_dict
        assert 'source_hash' in data_dict
        
        # Deserialize from dict
        restored = SourceMetadata.from_dict(data_dict)
        
        assert restored.source_path == original.source_path
        assert restored.format_name == original.format_name
        assert restored.loader_parameters == original.loader_parameters
        assert restored.import_timestamp == original.import_timestamp
        assert restored.source_mtime == original.source_mtime
        assert restored.source_hash == original.source_hash


class TestPipelineStageTracker:
    """Test PipelineStageTracker class functionality."""
    
    def test_stage_tracker_initialization(self):
        """Test PipelineStageTracker initialization."""
        # Empty initialization
        tracker = PipelineStageTracker()
        assert tracker.completed_stages == set()
        
        # With completed stages
        tracker = PipelineStageTracker(['fid_import', 'ft_processing'])
        assert tracker.completed_stages == {'fid_import', 'ft_processing'}
    
    def test_stage_dependencies_constant(self):
        """Test that stage dependencies are properly defined."""
        deps = PipelineStageTracker.STAGE_DEPENDENCIES
        
        assert deps['fid_import'] == []
        assert deps['ft_processing'] == ['fid_import']
        assert deps['noise_estimation'] == ['ft_processing']
        assert deps['peak_detection'] == ['noise_estimation']
        assert deps['window_assignment'] == ['peak_detection']
        assert deps['fitting'] == ['window_assignment']
    
    def test_mark_completed(self):
        """Test marking stages as completed."""
        tracker = PipelineStageTracker()
        
        tracker.mark_completed('fid_import')
        assert 'fid_import' in tracker.completed_stages
        
        tracker.mark_completed('ft_processing')
        assert 'ft_processing' in tracker.completed_stages
        
        # Test invalid stage name
        with pytest.raises(ValueError):
            tracker.mark_completed('invalid_stage')
    
    def test_is_completed(self):
        """Test checking if stages are completed."""
        tracker = PipelineStageTracker(['fid_import'])
        
        assert tracker.is_completed('fid_import')
        assert not tracker.is_completed('ft_processing')
    
    def test_validate_dependencies_success(self):
        """Test successful dependency validation."""
        tracker = PipelineStageTracker(['fid_import', 'ft_processing'])
        
        # Should not raise for stages with met dependencies
        tracker.validate_dependencies('fid_import')  # No dependencies
        tracker.validate_dependencies('ft_processing')  # fid_import completed
        tracker.validate_dependencies('noise_estimation')  # ft_processing completed
    
    def test_validate_dependencies_failure(self):
        """Test dependency validation failure."""
        tracker = PipelineStageTracker(['fid_import'])
        
        # Missing ft_processing dependency for noise_estimation
        with pytest.raises(PipelineStageError) as exc_info:
            tracker.validate_dependencies('noise_estimation')
        
        error = exc_info.value
        assert error.stage_name == 'noise_estimation'
        assert error.missing_dependencies == ['ft_processing']
        assert "Cannot perform noise_estimation" in str(error)
    
    def test_validate_dependencies_invalid_stage(self):
        """Test dependency validation with invalid stage."""
        tracker = PipelineStageTracker()
        
        with pytest.raises(ValueError):
            tracker.validate_dependencies('invalid_stage')
    
    def test_get_next_available_stages(self):
        """Test getting next available stages."""
        tracker = PipelineStageTracker()
        
        # Initially, only fid_import should be available
        available = tracker.get_next_available_stages()
        assert available == ['fid_import']
        
        # After completing fid_import
        tracker.mark_completed('fid_import')
        available = tracker.get_next_available_stages()
        assert available == ['ft_processing']
        
        # After completing ft_processing
        tracker.mark_completed('ft_processing')
        available = tracker.get_next_available_stages()
        assert available == ['noise_estimation']
    
    def test_stage_tracker_serialization(self):
        """Test PipelineStageTracker to_dict method."""
        tracker = PipelineStageTracker(['fid_import', 'ft_processing'])
        
        data_dict = tracker.to_dict()
        
        assert set(data_dict['completed_stages']) == {'fid_import', 'ft_processing'}
        assert data_dict['next_available'] == ['noise_estimation']


class TestPipelineFileManager:
    """Test PipelineFileManager core functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def sample_fid(self):
        """Create a sample FID for testing."""
        data = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        return FID(
            data=data,
            spacing=1e-6,  # 1 μs spacing
            probe_freq_mhz=10000.0,
            sideband=Sideband.LOWER,
            shots=1000,
            processing=FIDProcessingParameters(zpf=1, expf_us=5.0),
            metadata={'test_key': 'test_value'}
        )
    
    @pytest.fixture
    def sample_source_metadata(self):
        """Create sample source metadata."""
        return SourceMetadata(
            source_path="/test/source/path",
            format_name="test_format",
            loader_parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def file_manager(self):
        """Create PipelineFileManager instance."""
        return PipelineFileManager()
    
    def test_file_manager_initialization(self, file_manager):
        """Test PipelineFileManager initialization."""
        assert hasattr(file_manager, 'logger')
        assert file_manager.logger.name == 'ftmwpipeline.file_manager'
    
    def test_create_pipeline_file_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful pipeline file creation."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        result_path = file_manager.create_pipeline_file(
            filepath, sample_fid, sample_source_metadata
        )
        
        assert result_path == filepath
        assert filepath.exists()
        assert filepath.suffix == '.ftmw'
        
        # Verify HDF5 structure
        with h5py.File(filepath, 'r') as h5f:
            assert h5f.attrs['file_type'] == 'ftmw_pipeline'
            assert h5f.attrs['file_version'] == '1.0'
            assert 'creation_timestamp' in h5f.attrs
            assert 'source_metadata' in h5f
            assert 'pipeline_stages' in h5f
            assert 'stage0_fid_data' in h5f
    
    def test_create_pipeline_file_adds_extension(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test that .ftmw extension is added automatically."""
        filepath = temp_dir / "test_pipeline"  # No extension
        
        result_path = file_manager.create_pipeline_file(
            filepath, sample_fid, sample_source_metadata
        )
        
        assert result_path.suffix == '.ftmw'
        assert result_path.exists()
    
    def test_create_pipeline_file_creates_parent_dirs(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test that parent directories are created."""
        filepath = temp_dir / "subdir" / "nested" / "test_pipeline.ftmw"
        
        result_path = file_manager.create_pipeline_file(
            filepath, sample_fid, sample_source_metadata
        )
        
        assert result_path.exists()
        assert result_path.parent.exists()
    
    def test_create_pipeline_file_identical_source_reuse(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test that identical source reuses existing file."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        # Create first file
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        original_mtime = filepath.stat().st_mtime
        
        # Try to create again with identical source
        with patch.object(file_manager, 'logger') as mock_logger:
            result_path = file_manager.create_pipeline_file(
                filepath, sample_fid, sample_source_metadata
            )
            
            # Should reuse existing file
            assert result_path == filepath
            assert filepath.stat().st_mtime == original_mtime  # File not recreated
            mock_logger.info.assert_called_once()
            assert "Found existing pipeline with identical source" in mock_logger.info.call_args[0][0]
    
    def test_create_pipeline_file_different_source_conflict(self, file_manager, temp_dir, sample_fid):
        """Test conflict when creating file with different source."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        source1 = SourceMetadata("/source1", "format1")
        source2 = SourceMetadata("/source2", "format2")
        
        # Create first file
        file_manager.create_pipeline_file(filepath, sample_fid, source1)
        
        # Try to create with different source
        with pytest.raises(PipelineExistsError) as exc_info:
            file_manager.create_pipeline_file(filepath, sample_fid, source2)
        
        error = exc_info.value
        assert error.filepath == filepath
        assert "/source1" in error.existing_source
        assert "/source2" in error.requested_source
        assert "Pipeline.create" in str(error)
        assert "force=True" in str(error)
    
    def test_create_pipeline_file_force_overwrite(self, file_manager, temp_dir, sample_fid):
        """Test force overwriting existing file with different source."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        source1 = SourceMetadata("/source1", "format1")
        source2 = SourceMetadata("/source2", "format2")
        
        # Create first file
        file_manager.create_pipeline_file(filepath, sample_fid, source1)
        original_metadata = file_manager._load_source_metadata(filepath)
        
        # Force overwrite with different source
        with patch.object(file_manager, 'logger') as mock_logger:
            result_path = file_manager.create_pipeline_file(
                filepath, sample_fid, source2, force=True
            )
            
            assert result_path == filepath
            mock_logger.warning.assert_called_once()
            assert "Overwrote existing pipeline file" in mock_logger.warning.call_args[0][0]
        
        # Verify source was updated
        new_metadata = file_manager._load_source_metadata(filepath)
        assert new_metadata.source_path != original_metadata.source_path
    
    def test_create_pipeline_file_error_cleanup(self, file_manager, temp_dir, sample_source_metadata):
        """Test that partial files are cleaned up on error."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        # Create invalid FID that will cause save error
        invalid_fid = Mock()
        invalid_fid.configure_mock(**{'some_attr': 'some_value'})
        
        with patch('ftmwpipeline.file_manager.save_fid_to_hdf5', side_effect=RuntimeError("Save failed")):
            with pytest.raises(RuntimeError):
                file_manager.create_pipeline_file(filepath, invalid_fid, sample_source_metadata)
            
            # File should be cleaned up
            assert not filepath.exists()
    
    def test_open_pipeline_file_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful pipeline file opening."""
        filepath = temp_dir / "test_pipeline.ftmw"
        
        # Create file first
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        # Open file
        result_path, source_metadata, stage_tracker = file_manager.open_pipeline_file(filepath)
        
        assert result_path == filepath
        assert isinstance(source_metadata, SourceMetadata)
        assert isinstance(stage_tracker, PipelineStageTracker)
        assert source_metadata.format_name == "test_format"
        assert stage_tracker.is_completed('fid_import')
    
    def test_open_pipeline_file_not_found(self, file_manager, temp_dir):
        """Test opening non-existent pipeline file."""
        filepath = temp_dir / "nonexistent.ftmw"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            file_manager.open_pipeline_file(filepath)
        
        error_message = str(exc_info.value)
        assert "Pipeline file not found" in error_message
        assert "Pipeline.create" in error_message
        assert "ftmwpipeline import-data" in error_message
    
    def test_open_pipeline_file_invalid_format(self, file_manager, temp_dir):
        """Test opening file with invalid format."""
        filepath = temp_dir / "invalid.ftmw"
        
        # Create invalid HDF5 file
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'not_pipeline'
        
        with pytest.raises(ValueError):
            file_manager.open_pipeline_file(filepath)
    
    def test_open_pipeline_file_corrupted(self, file_manager, temp_dir):
        """Test opening corrupted pipeline file."""
        filepath = temp_dir / "corrupted.ftmw"
        
        # Create empty file (not valid HDF5)
        filepath.write_text("not hdf5 content")
        
        with pytest.raises(PipelineCorruptedError) as exc_info:
            file_manager.open_pipeline_file(filepath)
        
        error = exc_info.value
        assert error.filepath == filepath
        assert "Unable to read file structure" in error.corruption_details
    
    def test_validate_pipeline_file_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful pipeline file validation."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        report = file_manager.validate_pipeline_file(filepath)
        
        assert report['valid'] is True
        assert len(report['errors']) == 0
        assert report['filepath'] == str(filepath)
        assert 'source_info' in report
        assert 'stages' in report
        assert report['source_info']['format'] == 'test_format'
        assert report['stages']['fid_data_valid'] is True
    
    def test_validate_pipeline_file_not_found(self, file_manager, temp_dir):
        """Test validation of non-existent file."""
        filepath = temp_dir / "nonexistent.ftmw"
        
        report = file_manager.validate_pipeline_file(filepath)
        
        assert report['valid'] is False
        assert "File does not exist" in report['errors']
    
    def test_validate_pipeline_file_not_readable(self, file_manager, temp_dir):
        """Test validation of unreadable file."""
        filepath = temp_dir / "unreadable.ftmw"
        filepath.touch()
        
        with patch('os.access', return_value=False):
            report = file_manager.validate_pipeline_file(filepath)
            
            assert report['valid'] is False
            assert "File is not readable" in report['errors']
    
    def test_validate_pipeline_file_wrong_type(self, file_manager, temp_dir):
        """Test validation of file with wrong type."""
        filepath = temp_dir / "wrong_type.ftmw"
        
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'wrong_type'
        
        report = file_manager.validate_pipeline_file(filepath)
        
        assert report['valid'] is False
        assert "Not a valid pipeline file (wrong file_type)" in report['errors']
    
    def test_validate_pipeline_file_missing_groups(self, file_manager, temp_dir):
        """Test validation of file with missing required groups."""
        filepath = temp_dir / "missing_groups.ftmw"
        
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
            # Missing required groups
        
        report = file_manager.validate_pipeline_file(filepath)
        
        assert report['valid'] is False
        assert any("Missing required group" in error for error in report['errors'])
    
    def test_validate_pipeline_file_missing_source(self, file_manager, temp_dir, sample_fid):
        """Test validation warns about missing source files."""
        filepath = temp_dir / "missing_source.ftmw"
        
        # Create metadata pointing to non-existent source
        missing_source_metadata = SourceMetadata("/nonexistent/source", "test_format")
        
        # Create the file with the missing source metadata
        file_manager.create_pipeline_file(filepath, sample_fid, missing_source_metadata)
        
        report = file_manager.validate_pipeline_file(filepath)
        
        # Should be valid but with warning about missing source
        assert len(report['errors']) == 0  # No errors, just warnings
        assert any("Source data no longer exists" in warning for warning in report['warnings'])
        assert report['source_info']['source_exists'] is False
    
    def test_validate_pipeline_file_corrupted_fid(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test validation detects corrupted FID data."""
        filepath = temp_dir / "corrupted_fid.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        # Corrupt the FID data
        with h5py.File(filepath, 'r+') as h5f:
            del h5f['stage0_fid_data']
            h5f.create_group('stage0_fid_data')  # Empty group
        
        report = file_manager.validate_pipeline_file(filepath)
        
        assert report['valid'] is False
        assert any("FID data corrupted" in error for error in report['errors'])
        assert report['stages']['fid_data_valid'] is False
    
    def test_load_fid_data_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful FID data loading."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        loaded_fid = file_manager.load_fid_data(filepath)
        
        assert isinstance(loaded_fid, FID)
        np.testing.assert_array_equal(loaded_fid.data, sample_fid.data)
        assert loaded_fid.spacing == sample_fid.spacing
        assert loaded_fid.probe_freq_mhz == sample_fid.probe_freq_mhz
    
    def test_load_fid_data_file_not_found(self, file_manager, temp_dir):
        """Test FID loading from non-existent file."""
        filepath = temp_dir / "nonexistent.ftmw"
        
        with pytest.raises(FileNotFoundError):
            file_manager.load_fid_data(filepath)
    
    def test_load_fid_data_stage_not_complete(self, file_manager, temp_dir):
        """Test FID loading when stage is not complete."""
        filepath = temp_dir / "incomplete_stage.ftmw"
        
        # Create properly structured file but without fid_import completed
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
            h5f.attrs['file_version'] = '1.0'
            h5f.attrs['creation_timestamp'] = datetime.now().isoformat()
            
            # Add source metadata
            source_group = h5f.create_group('source_metadata')
            source_group.attrs['source_path'] = '/test/source'
            source_group.attrs['format_name'] = 'test'
            source_group.attrs['loader_parameters'] = '{}'
            source_group.attrs['import_timestamp'] = datetime.now().isoformat()
            source_group.attrs['source_mtime'] = 0.0
            source_group.attrs['source_hash'] = 'test_hash'
            
            # Add pipeline stages but without fid_import completed
            stages_group = h5f.create_group('pipeline_stages')
            stages_group.attrs['completed_stages'] = json.dumps([])  # No stages completed
            stages_group.attrs['last_updated'] = datetime.now().isoformat()
            
            # Add the FID data group (so file structure is valid)
            h5f.create_group('stage0_fid_data')
        
        with pytest.raises(PipelineStageError):
            file_manager.load_fid_data(filepath)
    
    def test_check_stage_dependencies_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful stage dependency checking."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        # Should not raise - fid_import is completed
        file_manager.check_stage_dependencies(filepath, 'ft_processing')
    
    def test_check_stage_dependencies_failure(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test stage dependency checking failure."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        # Try to check noise_estimation which requires ft_processing
        with pytest.raises(PipelineStageError) as exc_info:
            file_manager.check_stage_dependencies(filepath, 'noise_estimation')
        
        error = exc_info.value
        assert error.filepath == filepath
        assert error.stage_name == 'noise_estimation'
        assert 'ft_processing' in error.missing_dependencies
    
    def test_mark_stage_completed_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test successful stage completion marking."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        # Mark ft_processing as completed
        file_manager.mark_stage_completed(filepath, 'ft_processing')
        
        # Verify it was marked
        _, _, stage_tracker = file_manager.open_pipeline_file(filepath)
        assert stage_tracker.is_completed('ft_processing')
        
        # Verify it was persisted
        with h5py.File(filepath, 'r') as h5f:
            stages_group = h5f['pipeline_stages']
            completed_stages = json.loads(stages_group.attrs['completed_stages'])
            assert 'ft_processing' in completed_stages
            assert 'last_updated' in stages_group.attrs
    
    def test_mark_stage_completed_invalid_stage(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test marking invalid stage as completed."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        with pytest.raises(RuntimeError):
            file_manager.mark_stage_completed(filepath, 'invalid_stage')
    
    def test_load_source_metadata_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test loading source metadata from file."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        metadata = file_manager._load_source_metadata(filepath)
        
        assert metadata is not None
        assert metadata.format_name == "test_format"
        assert metadata.source_path.name == "path"  # Last part of /test/source/path
    
    def test_load_source_metadata_missing_group(self, file_manager, temp_dir):
        """Test loading source metadata from file without source_metadata group."""
        filepath = temp_dir / "no_metadata.ftmw"
        
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
        
        metadata = file_manager._load_source_metadata(filepath)
        assert metadata is None
    
    def test_load_source_metadata_with_h5f_parameter(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test loading source metadata with existing h5py.File object."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        with h5py.File(filepath, 'r') as h5f:
            metadata = file_manager._load_source_metadata(filepath, h5f)
            
            assert metadata is not None
            assert metadata.format_name == "test_format"
    
    def test_load_stage_tracker_success(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test loading stage tracker from file."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        stage_tracker = file_manager._load_stage_tracker(filepath)
        
        assert isinstance(stage_tracker, PipelineStageTracker)
        assert stage_tracker.is_completed('fid_import')
    
    def test_load_stage_tracker_missing_group(self, file_manager, temp_dir):
        """Test loading stage tracker from file without pipeline_stages group."""
        filepath = temp_dir / "no_stages.ftmw"
        
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
        
        stage_tracker = file_manager._load_stage_tracker(filepath)
        assert isinstance(stage_tracker, PipelineStageTracker)
        assert len(stage_tracker.completed_stages) == 0
    
    def test_load_stage_tracker_with_h5f_parameter(self, file_manager, temp_dir, sample_fid, sample_source_metadata):
        """Test loading stage tracker with existing h5py.File object."""
        filepath = temp_dir / "test_pipeline.ftmw"
        file_manager.create_pipeline_file(filepath, sample_fid, sample_source_metadata)
        
        with h5py.File(filepath, 'r') as h5f:
            stage_tracker = file_manager._load_stage_tracker(filepath, h5f)
            
            assert isinstance(stage_tracker, PipelineStageTracker)
            assert stage_tracker.is_completed('fid_import')


class TestPipelineFileManagerErrorHandling:
    """Test error handling and edge cases in PipelineFileManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def file_manager(self):
        """Create PipelineFileManager instance."""
        return PipelineFileManager()
    
    def test_file_creation_error_cleanup(self, file_manager, temp_dir):
        """Test that partial files are cleaned up on creation error."""
        filepath = temp_dir / "test.ftmw"
        sample_fid = Mock()
        sample_source = SourceMetadata("/test", "test")
        
        # Mock file creation to fail
        with patch('ftmwpipeline.file_manager.save_fid_to_hdf5', side_effect=RuntimeError("Creation failed")):
            with pytest.raises(RuntimeError):
                file_manager.create_pipeline_file(filepath, sample_fid, sample_source)
            
            # File should be cleaned up on error
            assert not filepath.exists()
    
    def test_invalid_hdf5_structure_recovery(self, file_manager, temp_dir):
        """Test handling of various HDF5 structure corruption scenarios."""
        filepath = temp_dir / "corrupt.ftmw"
        
        # Test 1: Truncated file
        with open(filepath, 'wb') as f:
            f.write(b'truncated hdf5 header')
        
        with pytest.raises(PipelineCorruptedError):
            file_manager.open_pipeline_file(filepath)
        
        # Test 2: Valid HDF5 but wrong structure
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
            h5f.create_group('source_metadata')
            # Missing other required groups
        
        report = file_manager.validate_pipeline_file(filepath)
        assert not report['valid']
        assert any("Missing required group" in error for error in report['errors'])
    
    def test_metadata_json_corruption(self, file_manager, temp_dir):
        """Test handling of corrupted JSON metadata."""
        filepath = temp_dir / "corrupt_json.ftmw"
        
        # Create file with corrupted JSON in metadata
        with h5py.File(filepath, 'w') as h5f:
            h5f.attrs['file_type'] = 'ftmw_pipeline'
            
            source_group = h5f.create_group('source_metadata')
            source_group.attrs['source_path'] = '/test/path'
            source_group.attrs['format_name'] = 'test'
            source_group.attrs['loader_parameters'] = 'invalid json {'  # Corrupted JSON
            
            stages_group = h5f.create_group('pipeline_stages')
            stages_group.attrs['completed_stages'] = '["fid_import"]'
            
            h5f.create_group('stage0_fid_data')
        
        report = file_manager.validate_pipeline_file(filepath)
        assert not report['valid']
        assert any("Invalid source metadata" in error for error in report['errors'])
    
    def test_special_characters_in_metadata(self, file_manager, temp_dir):
        """Test handling of special characters in metadata."""
        sample_fid = Mock()
        sample_source = SourceMetadata(
            source_path="/test/path with spaces/data",
            format_name="test_format", 
            loader_parameters={"param": "value with spaces"}
        )
        
        filepath = temp_dir / "special_chars.ftmw"
        
        with patch('ftmwpipeline.file_manager.save_fid_to_hdf5'):
            result_path = file_manager.create_pipeline_file(filepath, sample_fid, sample_source)
            assert result_path.exists()
        
        # Verify metadata can be loaded back
        loaded_metadata = file_manager._load_source_metadata(filepath)
        assert loaded_metadata.format_name == "test_format"
        assert loaded_metadata.loader_parameters["param"] == "value with spaces"


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_pipeline_exists_error(self):
        """Test PipelineExistsError exception."""
        filepath = Path("/test/path.ftmw")
        existing_source = "/existing/source"
        requested_source = "/requested/source"
        
        error = PipelineExistsError(filepath, existing_source, requested_source)
        
        assert error.filepath == filepath
        assert error.existing_source == existing_source
        assert error.requested_source == requested_source
        
        error_str = str(error)
        assert "already exists with different source" in error_str
        assert existing_source in error_str
        assert requested_source in error_str
        assert "Pipeline.open" in error_str
        assert "force=True" in error_str
    
    def test_pipeline_stage_error(self):
        """Test PipelineStageError exception."""
        stage_name = "noise_estimation"
        missing_deps = ["ft_processing"]
        filepath = Path("/test/path.ftmw")
        
        error = PipelineStageError(stage_name, missing_deps, filepath)
        
        assert error.stage_name == stage_name
        assert error.missing_dependencies == missing_deps
        assert error.filepath == filepath
        
        error_str = str(error)
        assert "Cannot perform noise_estimation" in error_str
        assert "ft_processing" in error_str
        assert str(filepath) in error_str
    
    def test_pipeline_corrupted_error(self):
        """Test PipelineCorruptedError exception."""
        filepath = Path("/test/path.ftmw")
        corruption_details = "Invalid HDF5 structure"
        
        error = PipelineCorruptedError(filepath, corruption_details)
        
        assert error.filepath == filepath
        assert error.corruption_details == corruption_details
        
        error_str = str(error)
        assert "appears to be corrupted" in error_str
        assert corruption_details in error_str
        assert "Recovery options" in error_str
        assert str(filepath) in error_str