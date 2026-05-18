"""
Shared fixtures and utilities for integration tests.

This module provides common test infrastructure for integration tests that
validate cross-interface consistency and real workflow functionality.
"""

import pytest
import subprocess
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time

# Import all interfaces for testing
from ftmwpipeline import Pipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import ComplexFT, FID


@pytest.fixture(scope="session")
def exp_2638_data_path():
    """Path to experiment 2638 data (used across all integration tests)."""
    data_path = Path("examples/blackchirp_data/2638")
    if not data_path.exists():
        pytest.skip("Experiment 2638 data not available for integration testing")
    return str(data_path)


@pytest.fixture
def temp_ftmw_dir():
    """Create temporary directory for .ftmw files with automatic cleanup."""
    temp_dir = tempfile.mkdtemp(prefix="ftmw_integration_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_ftmw_file(temp_ftmw_dir):
    """Generate temporary .ftmw file path."""
    return temp_ftmw_dir / "test_pipeline.ftmw"


@pytest.fixture
def standard_ft_params():
    """Standard FT parameters for consistent testing."""
    return {
        'zpf': 2,
        'expf_us': 5.0,
        'trim': (26500, 40000)
    }


@pytest.fixture  
def cli_helper():
    """Helper class for CLI command execution and validation."""
    
    class CLIHelper:
        """Helper for running CLI commands and capturing output."""
        
        @staticmethod
        def run_command(args: List[str], check_return_code: bool = True) -> Tuple[int, str, str]:
            """
            Run ftmwpipeline CLI command and capture output.
            
            Args:
                args: Command arguments (without 'ftmwpipeline' prefix)
                check_return_code: Whether to assert return code is 0
                
            Returns:
                Tuple of (return_code, stdout, stderr)
            """
            result = subprocess.run(
                ["ftmwpipeline"] + args,
                capture_output=True,
                text=True,
                timeout=30  # Prevent hanging tests
            )
            
            if check_return_code and result.returncode != 0:
                pytest.fail(
                    f"CLI command failed:\n"
                    f"Command: ftmwpipeline {' '.join(args)}\n"
                    f"Return code: {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            
            return result.returncode, result.stdout, result.stderr
        
        @staticmethod
        def import_data(ftmw_file: Path, source: str) -> Tuple[int, str, str]:
            """Run import-data command."""
            return CLIHelper.run_command([
                "import-data", str(ftmw_file), "--source", source
            ])
        
        @staticmethod
        def compute_ft(ftmw_file: Path, **params) -> Tuple[int, str, str]:
            """Run compute-ft command with parameters."""
            args = ["compute-ft", str(ftmw_file)]
            
            if 'zpf' in params:
                args.extend(["--zpf", str(params['zpf'])])
            if 'expf_us' in params:
                args.extend(["--expf_us", str(params['expf_us'])])
            if 'trim' in params and params['trim'] is not None:
                trim_start, trim_end = params['trim']
                args.extend(["--trim", f"{trim_start}:{trim_end}"])
            
            return CLIHelper.run_command(args)
        
        @staticmethod
        def visualize_ft(ftmw_file: Path, save_params: bool = False, **params) -> Tuple[int, str, str]:
            """Run visualize-ft command with parameters."""
            args = ["visualize-ft", str(ftmw_file)]
            
            if save_params:
                args.append("--save-params")
            
            if 'zpf' in params:
                args.extend(["--zpf", str(params['zpf'])])
            if 'expf_us' in params:
                args.extend(["--expf_us", str(params['expf_us'])])
            if 'trim' in params and params['trim'] is not None:
                trim_start, trim_end = params['trim']
                args.extend(["--trim", f"{trim_start}:{trim_end}"])
            
            return CLIHelper.run_command(args)
        
        @staticmethod
        def get_pipeline_info(ftmw_file: Path) -> Dict[str, Any]:
            """Get pipeline info and parse JSON output."""
            _, stdout, _ = CLIHelper.run_command([
                "info", str(ftmw_file), "--format", "json"
            ])
            return json.loads(stdout)
    
    return CLIHelper()


@pytest.fixture
def interface_results_validator():
    """Validator for comparing results across different interfaces."""
    
    class InterfaceResultsValidator:
        """Validate consistency between different interface results."""
        
        @staticmethod
        def compare_complex_ft(ft1: ComplexFT, ft2: ComplexFT, rtol: float = 1e-10) -> None:
            """
            Compare two ComplexFT objects for numerical consistency.
            
            Args:
                ft1, ft2: ComplexFT objects to compare
                rtol: Relative tolerance for floating point comparison
            """
            # Frequency arrays should be identical
            np.testing.assert_array_equal(
                ft1.freq_array, ft2.freq_array,
                err_msg="ComplexFT frequency arrays differ"
            )
            
            # Complex spectra should be numerically equivalent
            np.testing.assert_allclose(
                ft1.complex_spectrum, ft2.complex_spectrum,
                rtol=rtol, atol=1e-15,
                err_msg="ComplexFT complex spectra differ beyond tolerance"
            )
            
            # Magnitude spectra should be consistent
            np.testing.assert_allclose(
                ft1.magnitude_spectrum, ft2.magnitude_spectrum,
                rtol=rtol, atol=1e-15,
                err_msg="ComplexFT magnitude spectra differ beyond tolerance"
            )
            
            # Phase spectra should be consistent
            np.testing.assert_allclose(
                ft1.phase_spectrum, ft2.phase_spectrum,
                rtol=rtol, atol=1e-15,
                err_msg="ComplexFT phase spectra differ beyond tolerance"
            )
        
        @staticmethod
        def compare_pipeline_states(info1: Dict[str, Any], info2: Dict[str, Any]) -> None:
            """
            Compare pipeline state information from different interfaces.
            
            Args:
                info1, info2: Pipeline info dictionaries to compare
            """
            # Source metadata should be identical
            assert info1['source_metadata'] == info2['source_metadata'], \
                "Source metadata differs between interfaces"
            
            # Stage completion status should match
            assert info1['stage_tracker'] == info2['stage_tracker'], \
                "Stage completion differs between interfaces"
            
            # Processing parameters should be consistent
            if 'processing_parameters' in info1 and 'processing_parameters' in info2:
                assert info1['processing_parameters'] == info2['processing_parameters'], \
                    "Processing parameters differ between interfaces"
        
        @staticmethod
        def validate_file_equivalence(file1: Path, file2: Path) -> None:
            """
            Validate that two .ftmw files are functionally equivalent.
            
            Args:
                file1, file2: Paths to .ftmw files to compare
            """
            # Both files should exist
            assert file1.exists(), f"File {file1} does not exist"
            assert file2.exists(), f"File {file2} does not exist"
            
            # Compare pipeline states
            info1 = ftmw.get_pipeline_info(str(file1))
            info2 = ftmw.get_pipeline_info(str(file2))
            InterfaceResultsValidator.compare_pipeline_states(info1, info2)
            
            # Compare FID data if both have it
            try:
                fid1 = ftmw.load_fid(str(file1))
                fid2 = ftmw.load_fid(str(file2))
                
                np.testing.assert_array_equal(
                    fid1.data, fid2.data,
                    err_msg="FID data differs between files"
                )
                
                assert fid1.spacing == fid2.spacing, "FID spacing differs"
                assert fid1.probe_freq_mhz == fid2.probe_freq_mhz, "Probe frequency differs"
                
            except Exception:
                # If FID loading fails, files might not have completed Stage 0
                pass
    
    return InterfaceResultsValidator()


@pytest.fixture
def cli_command_runner():
    """Utility for running CLI commands in integration tests."""
    
    class CLICommandRunner:
        """Helper for running CLI commands and validating output."""
        
        @staticmethod
        def run_command(args: List[str], check_success: bool = True, timeout: int = 30) -> Tuple[bool, str, str]:
            """
            Run ftmwpipeline CLI command and capture output.
            
            Args:
                args: Command arguments (without 'ftmwpipeline' prefix)
                check_success: Whether to fail test if command fails
                timeout: Command timeout in seconds
                
            Returns:
                Tuple of (success, stdout, stderr)
            """
            result = subprocess.run(
                ["ftmwpipeline"] + args,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
            
            success = result.returncode == 0
            
            if check_success and not success:
                pytest.fail(
                    f"CLI command failed:\n"
                    f"Command: ftmwpipeline {' '.join(args)}\n"
                    f"Return code: {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            
            return success, result.stdout, result.stderr
        
        @staticmethod
        def import_data(ftmw_file: Path, source: str, **kwargs) -> Tuple[bool, str, str]:
            """Run import-data command."""
            args = ["import-data", str(ftmw_file), "--source", source]
            
            if 'format' in kwargs:
                args.extend(["--format", kwargs['format']])
            if 'fid_index' in kwargs:
                args.extend(["--fid-index", str(kwargs['fid_index'])])
            
            return CLICommandRunner.run_command(args)
        
        @staticmethod
        def compute_ft(ftmw_file: Path, **params) -> Tuple[bool, str, str]:
            """Run compute-ft command with parameters."""
            args = ["compute-ft", str(ftmw_file)]
            
            if 'zpf' in params:
                args.extend(["--zpf", str(params['zpf'])])
            if 'expf_us' in params:
                args.extend(["--expf_us", str(params['expf_us'])])
            if 'trim' in params and params['trim'] is not None:
                trim_start, trim_end = params['trim']
                args.extend(["--trim", f"{trim_start}:{trim_end}"])
            
            return CLICommandRunner.run_command(args)
        
        @staticmethod
        def visualize_ft(ftmw_file: Path, **params) -> Tuple[bool, str, str]:
            """Run visualize-ft command with parameters."""
            args = ["visualize-ft", str(ftmw_file), "--no-interactive"]
            
            if 'zpf' in params:
                args.extend(["--zpf", str(params['zpf'])])
            if 'expf_us' in params:
                args.extend(["--expf_us", str(params['expf_us'])])
            if 'trim' in params and params['trim'] is not None:
                trim_start, trim_end = params['trim']
                args.extend(["--trim", f"{trim_start}:{trim_end}"])
            
            return CLICommandRunner.run_command(args)
    
    return CLICommandRunner()


# Pytest markers for integration test categories
pytest_integration_markers = [
    "single_interface: Tests focusing on individual interface workflows",
    "cross_interface: Tests validating consistency across interfaces", 
    "file_portability: Tests validating file compatibility between interfaces",
    "parameter_persistence: Tests validating parameter saving/loading",
    "error_consistency: Tests validating consistent error handling"
]

def pytest_configure(config):
    """Configure integration test markers."""
    for marker in pytest_integration_markers:
        config.addinivalue_line("markers", marker)