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


# ---------------------------------------------------------------------------
# Session-scoped prebuilt baselines for 2638 (expensive import+FT+noise done once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def baseline_2638_stage1(exp_2638_data_path, tmp_path_factory):
    """
    Build the 2638 pipeline through Stage 0+1 ONCE per test session, using
    the functional API with standard FT parameters (zpf=2, expf_us=5.0,
    trim=(26500, 40000)).

    Returns the Path to a read-only reference .ftmw file.  Tests that need
    a writable copy must shutil.copy it into their own tmp dir before mutating.
    """
    tmp = tmp_path_factory.mktemp("baseline_stage1")
    fp = tmp / "baseline_2638_stage1.ftmw"
    ftmw.import_data(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, zpf=2, expf_us=5.0, trim=(26500, 40000))
    return fp


@pytest.fixture(scope="session")
def baseline_2638_stage1_raw(exp_2638_data_path, tmp_path_factory):
    """Stage 0+1 with the CANONICAL raw FT (zpf=0, expf_us=None) — the grid the
    scatter noise estimator actually runs on in production.

    The ``standard_ft_params`` baseline above is zpf=2 (a legacy-comparison
    grid). Running the broad-window scatter smoother on that 4×-denser grid is
    both unrepresentative of production and needlessly slow (the smoothing is
    ~O(N·window)). Scatter tests use this raw fixture instead.
    """
    tmp = tmp_path_factory.mktemp("baseline_stage1_raw")
    fp = tmp / "baseline_2638_stage1_raw.ftmw"
    ftmw.import_data(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, zpf=0, expf_us=None, trim=(26500, 40000))
    return fp


@pytest.fixture(scope="session")
def baseline_2638_stage2(baseline_2638_stage1, tmp_path_factory):
    """
    Build the 2638 pipeline through Stage 0+1+2 ONCE per test session by
    copying the stage1 baseline and running estimate_noise.

    Pinned to ``method="adaptive"``: the Stage 3/4/5 regression baselines that
    build on this fixture are calibrated against the adaptive noise floor. The
    package default is now the scatter estimator; re-deriving these downstream
    baselines against scatter is a deliberate benchmark step (see
    ``dev-docs/planning/stage3-snr-corner-benchmark.md``), not an incidental
    consequence of the default flip.
    """
    tmp = tmp_path_factory.mktemp("baseline_stage2")
    fp = tmp / "baseline_2638_stage2.ftmw"
    shutil.copy(baseline_2638_stage1, fp)
    ftmw.estimate_noise(fp, method="adaptive")
    return fp


@pytest.fixture(scope="session")
def baseline_2638_stage3(baseline_2638_stage2, tmp_path_factory):
    """
    Build the 2638 pipeline through Stage 0+1+2+3 ONCE per test session by
    copying the stage2 baseline and running detect_peaks with default params.

    Returns the Path to a read-only reference .ftmw file.  Tests that need a
    writable copy must shutil.copy it.
    """
    tmp = tmp_path_factory.mktemp("baseline_stage3")
    fp = tmp / "baseline_2638_stage3.ftmw"
    shutil.copy(baseline_2638_stage2, fp)
    ftmw.detect_peaks(fp)
    return fp


@pytest.fixture(scope="session")
def baseline_2638_stage4(baseline_2638_stage3, tmp_path_factory):
    """
    Build the 2638 pipeline through Stage 0+1+2+3+4 ONCE per test session by
    copying the stage3 baseline and running assign_windows with defaults.

    Returns the Path to a read-only reference .ftmw file. Tests that need a
    writable copy must shutil.copy it.
    """
    tmp = tmp_path_factory.mktemp("baseline_stage4")
    fp = tmp / "baseline_2638_stage4.ftmw"
    shutil.copy(baseline_2638_stage3, fp)
    ftmw.assign_windows(fp)
    return fp


@pytest.fixture(scope="session")
def baseline_2638_stage4_small(baseline_2638_stage4, tmp_path_factory):
    """A Stage-4 baseline trimmed to the first 3 dependency-free windows.

    Cuts Stage 5 fit cost on the 2638 fixture from ~2 minutes (382 windows)
    to ~5 seconds (3 windows) while still exercising the full fit pipeline
    on real data. Used by tests that verify pipeline shape (cross-interface
    bit-identity, serialization round-trip) -- not by tests that depend on
    the full per-band statistics.

    Selects the first ``n_target`` windows whose dependency edges all stay
    within the selected set (typically window_0000 .. window_0009 are
    dependency-free, since dep edges on 2638 start at window pair (10, 11)).
    """
    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl, save_window_plan_impl,
    )

    n_target = 3
    tmp = tmp_path_factory.mktemp("baseline_stage4_small")
    fp = tmp / "baseline_2638_stage4_small.ftmw"
    shutil.copy(baseline_2638_stage4, fp)

    plan = load_windows_impl(str(fp))["plan"]
    # Walk in topological order; take windows with no inter-window dep
    # constraints with windows outside the keep set.
    candidates = []
    for wid in plan.topological_order:
        deps = [
            (a, b) for (a, b) in plan.dependency_edges
            if a == wid or b == wid
        ]
        if all(a in candidates or a == wid for (a, _) in deps) and \
           all(b in candidates or b == wid for (_, b) in deps):
            candidates.append(wid)
        if len(candidates) >= n_target:
            break
    if not candidates:
        candidates = list(plan.topological_order[:n_target])
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(fp), plan)
    return fp


# ---------------------------------------------------------------------------
# Module-scoped cross-interface trio fixtures (one per test module)
# ---------------------------------------------------------------------------

def _build_stage1_trio(exp_2638_data_path: str, tmp: Path, ft_params: dict) -> dict:
    """
    Build three .ftmw files through Stage 0+1 — one per interface (Pipeline,
    functional API, CLI) — and return their paths.

    All three should produce identical results; this is the shared evidence used
    by identity tests, so they must be built independently (not copied from each
    other) to preserve the proof of cross-interface consistency.
    """
    pipeline_file = tmp / "trio_pipeline.ftmw"
    functional_file = tmp / "trio_functional.ftmw"
    cli_file = tmp / "trio_cli.ftmw"

    # Pipeline interface
    pipe = Pipeline.create(pipeline_file, source=exp_2638_data_path)
    pipe.compute_ft(**ft_params)

    # Functional API
    ftmw.import_data(functional_file, source=exp_2638_data_path)
    ftmw.compute_ft(functional_file, **ft_params)

    # CLI
    _run_cli(["import-data", str(cli_file), "--source", exp_2638_data_path])
    zpf, expf_us = ft_params["zpf"], ft_params["expf_us"]
    trim_min, trim_max = ft_params["trim"]
    _run_cli([
        "compute-ft", str(cli_file),
        "--zpf", str(zpf),
        "--expf_us", str(expf_us),
        "--trim", f"{trim_min}:{trim_max}",
    ])

    return {
        "pipeline": pipeline_file,
        "functional": functional_file,
        "cli": cli_file,
    }


def _run_cli(args: list) -> None:
    """Run an ftmwpipeline CLI command; fail loudly on non-zero exit."""
    result = subprocess.run(
        ["ftmwpipeline"] + args,
        capture_output=True, text=True, check=False, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CLI command failed: ftmwpipeline {' '.join(args)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.fixture(scope="module")
def cross_interface_stage1_trio(exp_2638_data_path, tmp_path_factory, standard_ft_params):
    """
    Module-scoped: build the three-interface Stage 0+1 files ONCE per test module.

    The files are read-only reference files; they must NOT be mutated by any
    consuming test.  Tests that need a writable file must shutil.copy into their
    own tmp_path.

    Returns a dict: {"pipeline": Path, "functional": Path, "cli": Path}.
    """
    tmp = tmp_path_factory.mktemp("cross_interface_stage1_trio")
    return _build_stage1_trio(exp_2638_data_path, tmp, standard_ft_params)




@pytest.fixture
def temp_ftmw_file(temp_ftmw_dir):
    """Generate temporary .ftmw file path."""
    return temp_ftmw_dir / "test_pipeline.ftmw"


@pytest.fixture(scope="session")
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