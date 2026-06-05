"""
Unit tests for data loaders (Stage 0 - Data Loading).

Tests the new Stage 0-1 architecture where FID data is loaded and cached separately
from ComplexFT processing. Covers:
- BlackChirp data loader functionality
- Format detection and validation
- Error handling for invalid/missing files
- Metadata preservation during loading
- Loader registry functionality

Uses experiment 2638 test data as specified in CLAUDE.md.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from ftmwpipeline.io.data_loaders.blackchirp import BlackChirpLoader
from ftmwpipeline.io.data_loaders.registry import (
    FormatRegistry,
    register_loader,
    detect_format,
)
from ftmwpipeline.io.data_loaders.base import BaseLoader, LoaderError
from ftmwpipeline.core.data_structures import FID, Sideband, FIDProcessingParameters


class TestBlackChirpLoader:
    """Test BlackChirp data format loader."""

    @pytest.fixture
    def sample_blackchirp_dir(self):
        """Create a temporary BlackChirp directory structure for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        exp_dir = temp_dir / "test_exp"
        fid_dir = exp_dir / "fid"
        fid_dir.mkdir(parents=True)

        # Create fidparams.csv (faithful Blackchirp layout: leading ``index``
        # column, canonical ``LowerSideband`` enum name, ``size`` column).
        fidparams_data = {
            "index": [0],
            "spacing": [2.44140625e-11],
            "probefreq": [40960.0],
            "vmult": [0.001],
            "shots": [100000],
            "sideband": ["LowerSideband"],
            "size": [9],
        }
        fidparams_df = pd.DataFrame(fidparams_data)
        fidparams_df.to_csv(fid_dir / "fidparams.csv", sep=";", index=False)

        # Create FID data file (0.csv) with base-36 encoded integers
        fid_data = [
            "0",
            "1k",
            "2z",
            "a0",
            "z1",
            "10",
            "0",
            "-1k",
            "-2z",
        ]  # Sample base-36 data
        fid_df = pd.DataFrame({"data": fid_data})
        fid_df.to_csv(fid_dir / "0.csv", index=False)

        # Create optional processing.csv
        processing_data = {
            "ObjKey": [
                "FidStartUs",
                "FidEndUs",
                "FidZeroPadFactor",
                "FidExpfUs",
                "FidRemoveDC",
            ],
            "Value": [0.0, 10.0, 1, 5.0, "true"],
        }
        processing_df = pd.DataFrame(processing_data)
        processing_df.to_csv(fid_dir / "processing.csv", sep=";", index=False)

        # Create optional metadata files
        header_data = {
            "experiment": ["Test Experiment"],
            "date": ["2023-01-01"],
            "operator": ["Test User"],
        }
        header_df = pd.DataFrame(header_data)
        header_df.to_csv(exp_dir / "header.csv", sep=";", index=False)

        # version.csv: first line is the CSV delimiter, then key;value rows
        # (the format BCExperiment/BCFTMW expect).
        with open(exp_dir / "version.csv", "w") as f:
            f.write(";\nkey;value\nBCMajorVersion;1\nBCMinorVersion;0\n")

        yield exp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def loader(self):
        """Get BlackChirp loader instance."""
        return BlackChirpLoader()

    def test_loader_properties(self, loader):
        """Test loader basic properties."""
        assert loader.format_name == "blackchirp"
        assert loader.file_extensions == []  # Uses directories
        assert "fid" in loader.directory_indicators
        assert "fidparams.csv" in loader.directory_indicators

    def test_can_load_valid_directory(self, loader, sample_blackchirp_dir):
        """Test can_load() with valid BlackChirp directory."""
        assert loader.can_load(sample_blackchirp_dir) is True

    def test_can_load_invalid_paths(self, loader):
        """Test can_load() with various invalid paths."""
        # Non-existent path
        assert loader.can_load("/nonexistent/path") is False

        # File instead of directory
        with tempfile.NamedTemporaryFile() as tmp_file:
            assert loader.can_load(tmp_file.name) is False

        # Directory without fid subdirectory
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert loader.can_load(tmp_dir) is False

        # Directory with fid but no fidparams.csv
        with tempfile.TemporaryDirectory() as tmp_dir:
            fid_dir = Path(tmp_dir) / "fid"
            fid_dir.mkdir()
            assert loader.can_load(tmp_dir) is False

    def test_validate_source_valid(self, loader, sample_blackchirp_dir):
        """Test validate_source() with valid BlackChirp directory."""
        result = loader.validate_source(sample_blackchirp_dir)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

        # Check metadata
        assert result["metadata"]["n_fids"] == 1
        assert result["metadata"]["probe_freq_mhz"] == 40960.0
        assert result["metadata"]["sideband"] == "LowerSideband"
        assert result["metadata"]["shots"] == 100000
        assert "spacing_us" in result["metadata"]

        # Check options
        assert result["options"]["available_fid_indices"] == [0]

        # Check processing parameters were loaded
        assert "processing_parameters" in result["metadata"]
        assert result["metadata"]["processing_parameters"]["FidStartUs"] == "0.0"

    def test_validate_source_invalid(self, loader):
        """Test validate_source() with invalid directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = loader.validate_source(tmp_dir)

            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "Not a valid Blackchirp experiment directory" in result["errors"][0]

    def test_load_fid_success(self, loader, sample_blackchirp_dir):
        """Test successful FID loading."""
        fid = loader.load_fid(sample_blackchirp_dir, fid_index=0)

        # Check FID properties
        assert isinstance(fid, FID)
        assert fid.probe_freq_mhz == 40960.0
        assert fid.sideband == Sideband.LOWER
        assert fid.shots == 100000
        assert fid.spacing == 2.44140625e-11  # seconds

        # Check data conversion (base-36 to voltage)
        assert len(fid.data) == 9  # Number of data points in test data
        assert isinstance(fid.data[0], float)

        # Check processing parameters
        assert isinstance(fid.processing, FIDProcessingParameters)
        assert fid.processing.start_us == 0.0
        assert fid.processing.end_us == 10.0
        assert fid.processing.zpf == 1
        assert fid.processing.expf_us == 5.0
        assert fid.processing.rdc is True

        # Check metadata preservation
        assert "source_path" in fid.metadata
        assert "source_format" in fid.metadata
        # BlackChirp params are nested under loader_parameters
        assert "loader_parameters" in fid.metadata
        assert "blackchirp_params" in fid.metadata["loader_parameters"]

    def test_load_fid_invalid_index(self, loader, sample_blackchirp_dir):
        """Test FID loading with invalid FID index."""
        with pytest.raises(LoaderError, match="FID index 5 not found"):
            loader.load_fid(sample_blackchirp_dir, fid_index=5)

    def test_load_fid_missing_data_file(self, loader, sample_blackchirp_dir):
        """Test FID loading with missing data file."""
        # Remove the data file
        data_file = sample_blackchirp_dir / "fid" / "0.csv"
        data_file.unlink()

        with pytest.raises(LoaderError, match="Invalid Blackchirp source"):
            loader.load_fid(sample_blackchirp_dir, fid_index=0)

    def test_load_fid_corrupted_data(self, loader, sample_blackchirp_dir):
        """Test FID loading with corrupted base-36 data."""
        # Create corrupted data file with actually invalid base-36 data
        # Note: 'xyz123' is valid base-36, so use truly invalid characters
        data_file = sample_blackchirp_dir / "fid" / "0.csv"
        corrupted_data = pd.DataFrame({"data": ["invalid@#$", "not_base36!"]})
        corrupted_data.to_csv(data_file, index=False)

        with pytest.raises(LoaderError, match="Failed to load Blackchirp FID"):
            loader.load_fid(sample_blackchirp_dir, fid_index=0)

    def test_sideband_conversion(self, loader, sample_blackchirp_dir):
        """Test sideband string to enum conversion."""
        # Test upper sideband
        fidparams_file = sample_blackchirp_dir / "fid" / "fidparams.csv"
        fidparams_df = pd.read_csv(fidparams_file, sep=";")
        fidparams_df["sideband"] = ["UpperSideband"]
        fidparams_df.to_csv(fidparams_file, sep=";", index=False)

        fid = loader.load_fid(sample_blackchirp_dir, fid_index=0)
        assert fid.sideband == Sideband.UPPER

    def test_required_optional_parameters(self, loader):
        """Test parameter specification methods."""
        required = loader.get_required_parameters()
        optional = loader.get_optional_parameters()

        assert isinstance(required, list)
        assert len(required) == 0  # BlackChirp has no required params

        assert isinstance(optional, dict)
        assert "fid_index" in optional
        assert optional["fid_index"] == 0


class TestExperiment2638Integration:
    """Test BlackChirp loader with real experiment 2638 data."""

    def test_load_real_experiment_2638(self):
        """Test loading real experiment 2638 data."""
        exp_path = Path("examples/blackchirp_data/2638")
        if not exp_path.exists():
            pytest.skip("Experiment 2638 data not available")

        loader = BlackChirpLoader()

        # Test validation
        validation = loader.validate_source(exp_path)
        assert validation["valid"] is True

        # Check expected parameters for experiment 2638
        assert validation["metadata"]["probe_freq_mhz"] == 40960.0  # 40.96 GHz
        assert (
            validation["metadata"]["sideband"] == "LowerSideband"
        )  # Actual value in experiment 2638
        assert validation["metadata"]["n_fids"] >= 1

        # Test FID loading
        fid = loader.load_fid(exp_path, fid_index=0)

        # Check FID specs match CLAUDE.md specifications
        assert fid.probe_freq_mhz == 40960.0  # 40.96 GHz probe
        assert fid.sideband == Sideband.LOWER  # Lower Sideband
        assert fid.n_points == 750000  # 750k points
        assert abs(fid.duration_us - 15.0) < 0.1  # ~15 μs duration

        # Check time spacing is in seconds with correct format
        assert fid.spacing > 0
        spacing_str = f"{fid.spacing:.4e}"  # Should use .4e format per refinement #2
        assert "e-" in spacing_str  # Verify scientific notation

        # Check data is real, not complex (per refinement #1)
        assert np.all(np.isreal(fid.data))
        assert fid.data.dtype in [np.float64, np.float32]

        print(f"Experiment 2638 loaded successfully:")
        print(f"  Points: {fid.n_points}")
        print(f"  Duration: {fid.duration_us:.1f} μs")
        print(f"  Spacing: {fid.spacing:.4e} s")  # Per refinement #2
        print(f"  Probe: {fid.probe_freq_mhz:.1f} MHz")
        print(f"  Sideband: {fid.sideband.value}")


class TestFormatRegistry:
    """Test loader registry functionality."""

    def test_registry_initialization(self):
        """Test that loader registry initializes properly."""
        registry = FormatRegistry()

        # New registry should start empty
        assert len(registry._loaders) == 0
        assert len(registry._loader_instances) == 0

        # But global registry should have BlackChirp registered (via module import)
        from ftmwpipeline.io.data_loaders.registry import _global_registry

        assert "blackchirp" in _global_registry._loaders

    def test_register_new_loader(self):
        """Test registering a new loader."""
        registry = FormatRegistry()

        # Create mock loader
        class MockLoader(BaseLoader):
            format_name = "mock_format"
            file_extensions = [".mock"]

            def can_load(self, source_path):
                return True

            def validate_source(self, source_path, **kwargs):
                return {"valid": True}

            def load_fid(self, source_path, **kwargs):
                pass

            def get_required_parameters(self):
                return []

            def get_optional_parameters(self):
                return {}

        # Register the loader class, not the instance
        registry.register_loader("mock_format", MockLoader)

        assert "mock_format" in registry._loaders
        assert registry._loaders["mock_format"] is MockLoader
        # Check that instance was created
        assert "mock_format" in registry._loader_instances
        assert isinstance(registry._loader_instances["mock_format"], MockLoader)

    def test_detect_format_directory(self):
        """Test format detection for directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create BlackChirp-like structure
            exp_dir = Path(tmp_dir) / "test_exp"
            fid_dir = exp_dir / "fid"
            fid_dir.mkdir(parents=True)
            (fid_dir / "fidparams.csv").touch()
            (fid_dir / "0.csv").touch()

            detected_format = detect_format(exp_dir)
            assert detected_format == "blackchirp"

    def test_detect_format_unknown(self):
        """Test format detection for unknown formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Empty directory - should not be detected
            detected_format = detect_format(tmp_dir)
            assert detected_format is None

    def test_global_register_function(self):
        """Test global register_loader function."""

        class TestGlobalLoader(BaseLoader):
            format_name = "global_test"
            file_extensions = [".test"]

            def can_load(self, source_path):
                return False

            def validate_source(self, source_path, **kwargs):
                return {"valid": False}

            def load_fid(self, source_path, **kwargs):
                pass

            def get_required_parameters(self):
                return []

            def get_optional_parameters(self):
                return {}

        # Register with global function
        register_loader("global_test", TestGlobalLoader)

        # Should be registered in global registry
        from ftmwpipeline.io.data_loaders.registry import _global_registry

        assert "global_test" in _global_registry._loaders


class TestBaseLoaderErrorHandling:
    """Test error handling in base loader functionality."""

    def test_loader_error_creation(self):
        """Test LoaderError exception creation."""
        error = LoaderError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_base_loader_abstract_methods(self):
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_invalid_loader_registration(self):
        """Test error handling for invalid loader registration."""
        registry = FormatRegistry()

        # Try to register non-loader class
        with pytest.raises(
            ValueError, match="loader_class must inherit from BaseLoader"
        ):
            registry.register_loader(
                "invalid_format", str
            )  # str is not a BaseLoader subclass

    def test_metadata_creation_error_handling(self):
        """Test error handling in metadata creation."""
        loader = BlackChirpLoader()

        # Test with non-existent path
        with pytest.raises(LoaderError):
            loader.load_fid("/nonexistent/path")

        # Test with invalid source
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(LoaderError, match="Invalid Blackchirp source"):
                loader.load_fid(tmp_dir)


class TestDataLoadingEdgeCases:
    """Test edge cases and boundary conditions in data loading."""

    @pytest.fixture
    def minimal_blackchirp_dir(self):
        """Create minimal valid BlackChirp directory."""
        temp_dir = Path(tempfile.mkdtemp())

        exp_dir = temp_dir / "minimal_exp"
        fid_dir = exp_dir / "fid"
        fid_dir.mkdir(parents=True)

        # Minimal fidparams.csv
        fidparams_data = {
            "index": [0],
            "spacing": [1e-6],
            "probefreq": [1000.0],
            "vmult": [1.0],
            "shots": [1],
            "sideband": ["UpperSideband"],
            "size": [3],
        }
        pd.DataFrame(fidparams_data).to_csv(
            fid_dir / "fidparams.csv", sep=";", index=False
        )

        # Minimal FID data
        fid_data = pd.DataFrame({"data": ["0", "1", "0"]})
        fid_data.to_csv(fid_dir / "0.csv", index=False)

        yield exp_dir

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_minimal_valid_experiment(self, minimal_blackchirp_dir):
        """Test loading minimal valid BlackChirp experiment."""
        loader = BlackChirpLoader()

        # Should be valid
        assert loader.can_load(minimal_blackchirp_dir) is True

        # Should load successfully
        fid = loader.load_fid(minimal_blackchirp_dir)
        assert isinstance(fid, FID)
        assert fid.probe_freq_mhz == 1000.0
        assert fid.n_points == 3

    def test_empty_fid_data(self):
        """Test handling of empty FID data files."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            exp_dir = temp_dir / "empty_exp"
            fid_dir = exp_dir / "fid"
            fid_dir.mkdir(parents=True)

            # Valid fidparams
            fidparams_data = {
                "probefreq": [1000.0],
                "spacing": [1e-6],
                "sideband": ["Upper"],
                "shots": [1],
                "vmult": [1.0],
            }
            pd.DataFrame(fidparams_data).to_csv(
                fid_dir / "fidparams.csv", sep=";", index=False
            )

            # Empty FID data file
            pd.DataFrame().to_csv(fid_dir / "0.csv", index=False)

            loader = BlackChirpLoader()

            # Should detect as valid directory structure
            assert loader.can_load(exp_dir) is True

            # But loading should handle empty data gracefully
            with pytest.raises(LoaderError):
                loader.load_fid(exp_dir)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_large_fid_index(self, minimal_blackchirp_dir):
        """Test requesting FID index larger than available."""
        loader = BlackChirpLoader()

        with pytest.raises(LoaderError, match="FID index 999 not found"):
            loader.load_fid(minimal_blackchirp_dir, fid_index=999)

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation with edge case values."""
        loader = BlackChirpLoader()

        # Test with negative FID index (should be handled gracefully)
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Even though it will fail on directory validation,
            # the parameter should be validated appropriately
            with pytest.raises(LoaderError):  # Will fail on source validation
                loader.load_fid(tmp_dir, fid_index=-1)
