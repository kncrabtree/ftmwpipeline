"""
Test basic package imports and structure.

These tests verify that the package can be imported correctly and that
the basic API is accessible.
"""


class TestPackageImports:
    """Test basic package import functionality."""

    def test_main_package_import(self):
        """Test that main package can be imported."""
        import ftmwpipeline

        assert hasattr(ftmwpipeline, "__version__")
        assert hasattr(ftmwpipeline, "__author__")

    def test_core_module_import(self):
        """Test that core module can be imported."""
        from ftmwpipeline import core

        # Basic structure test - actual classes will be implemented in Phase 2
        assert hasattr(core, "__all__")

    def test_pipeline_class_import(self):
        """Test that the file-bound Pipeline class exposes its real API."""
        from ftmwpipeline import Pipeline

        # Pipeline is file-bound: constructed via create()/open() factories,
        # not a bare constructor.
        assert hasattr(Pipeline, "create")
        assert hasattr(Pipeline, "open")

        # Implemented stage methods (Stages 0-2).
        for method in (
            "load_data",
            "compute_ft",
            "visualize_ft",
            "estimate_noise",
            "visualize_noise",
            "info",
            "validate",
        ):
            assert hasattr(Pipeline, method), f"Pipeline missing {method}()"

    def test_validation_function(self):
        """Test the installation validation function."""
        from ftmwpipeline.workflows import validate_installation

        results = validate_installation()
        assert isinstance(results, dict)
        assert "core_imports" in results
        assert "dependencies" in results
        assert "pipeline_creation" in results

        # At minimum, core imports and pipeline creation should work
        assert results["core_imports"] is True
        assert results["pipeline_creation"] is True


class TestSubmoduleImports:
    """Test that all submodules can be imported without errors."""

    def test_preprocessing_import(self):
        """Test preprocessing module import."""
        from ftmwpipeline import preprocessing

        assert hasattr(preprocessing, "__all__")

    def test_fitting_import(self):
        """Test fitting module import."""
        from ftmwpipeline import fitting

        assert hasattr(fitting, "__all__")

    def test_visualization_import(self):
        """Test visualization module import."""
        from ftmwpipeline import visualization

        assert hasattr(visualization, "__all__")

    def test_io_import(self):
        """Test IO module import."""
        from ftmwpipeline import io

        assert hasattr(io, "__all__")

    def test_utils_import(self):
        """Test utils module import."""
        from ftmwpipeline import utils

        assert hasattr(utils, "__all__")


class TestPackageMetadata:
    """Test package metadata and version information."""

    def test_version_format(self):
        """Test that version is a valid PEP 440 release string."""
        import re

        import ftmwpipeline

        version = ftmwpipeline.__version__
        assert isinstance(version, str)

        # PEP 440: a numeric release (at least major.minor) optionally followed
        # by pre-release (aN/bN/rcN), post (.postN), or dev (.devN) segments.
        # Accepts e.g. "0.1.0", "0.1.0b1", "1.2.3rc1", "1.0.dev0".
        match = re.fullmatch(
            r"(\d+)\.(\d+)(?:\.\d+)*"  # release: at least major.minor
            r"(?:(?:a|b|rc)\d+)?"  # optional pre-release
            r"(?:\.post\d+)?"  # optional post-release
            r"(?:\.dev\d+)?",  # optional dev-release
            version,
        )
        assert match is not None, f"version {version!r} is not a valid PEP 440 release"

    def test_package_info(self):
        """Test package info dictionary."""
        import ftmwpipeline

        info = ftmwpipeline.PACKAGE_INFO
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "description" in info
        assert info["name"] == "ftmwpipeline"
