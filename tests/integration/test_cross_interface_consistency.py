"""
Test cross-interface consistency for the FTMW Pipeline.

This module tests that all three interfaces (Pipeline class, functional API, CLI)
produce identical results when given the same parameters and .ftmw files.
Focus is purely on functional consistency - NO performance testing.
"""

import pytest
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from ftmwpipeline import Pipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import ComplexFT, FID


@pytest.mark.cross_interface
class TestIdenticalResults:
    """Test that interfaces produce identical numerical results."""
    
    def test_identical_complex_ft_results(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Verify all interfaces produce identical ComplexFT for same parameters."""
        # Create separate files for each interface to avoid cross-contamination
        pipeline_file = temp_ftmw_dir / "pipeline_test.ftmw"
        functional_file = temp_ftmw_dir / "functional_test.ftmw"
        cli_file = temp_ftmw_dir / "cli_test.ftmw"
        
        # Create .ftmw files using each interface
        # Pipeline class
        pipe = Pipeline.create(pipeline_file, source=exp_2638_data_path)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)
        
        # Functional API
        ftmw.import_data(functional_file, source=exp_2638_data_path)
        complex_ft_functional = ftmw.compute_ft(functional_file, **standard_ft_params)
        
        # CLI interface
        self._run_cli_command([
            "data-load", str(cli_file),
            "--source", exp_2638_data_path
        ])
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "ft-process", str(cli_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        # Load result using functional API (CLI doesn't return ComplexFT directly)
        complex_ft_cli = ftmw.compute_ft(cli_file, **standard_ft_params)
        
        # Compare results using strict tolerance
        self._compare_complex_ft_objects(complex_ft_pipeline, complex_ft_functional, "Pipeline vs Functional")
        self._compare_complex_ft_objects(complex_ft_pipeline, complex_ft_cli, "Pipeline vs CLI")
        self._compare_complex_ft_objects(complex_ft_functional, complex_ft_cli, "Functional vs CLI")
    
    def test_identical_fid_loading(self, exp_2638_data_path, temp_ftmw_dir):
        """Verify all interfaces load identical FID data."""
        # Create separate files for each interface
        pipeline_file = temp_ftmw_dir / "pipeline_fid.ftmw"
        functional_file = temp_ftmw_dir / "functional_fid.ftmw"
        cli_file = temp_ftmw_dir / "cli_fid.ftmw"
        
        # Create .ftmw files using each interface
        pipe = Pipeline.create(pipeline_file, source=exp_2638_data_path)
        fid_pipeline = pipe.load_data()
        
        ftmw.import_data(functional_file, source=exp_2638_data_path)
        fid_functional = ftmw.load_fid(functional_file)
        
        self._run_cli_command([
            "data-load", str(cli_file),
            "--source", exp_2638_data_path
        ])
        fid_cli = ftmw.load_fid(cli_file)  # Use functional API to load
        
        # Compare FID objects
        self._compare_fid_objects(fid_pipeline, fid_functional, "Pipeline vs Functional")
        self._compare_fid_objects(fid_pipeline, fid_cli, "Pipeline vs CLI")
        self._compare_fid_objects(fid_functional, fid_cli, "Functional vs CLI")
    
    def test_identical_trim_behavior(self, exp_2638_data_path, temp_ftmw_dir):
        """Verify trim parameters work identically across interfaces."""
        # Test multiple trim ranges
        trim_ranges = [
            (26500, 40000),  # Standard range
            (30000, 35000),  # Narrower range
            (25000, 45000)   # Wider range
        ]
        
        for i, trim_range in enumerate(trim_ranges):
            pipeline_file = temp_ftmw_dir / f"pipeline_trim_{i}.ftmw"
            functional_file = temp_ftmw_dir / f"functional_trim_{i}.ftmw"
            
            # Create files with different interfaces
            pipe = Pipeline.create(pipeline_file, source=exp_2638_data_path)
            ftmw.import_data(functional_file, source=exp_2638_data_path)
            
            # Compute FT with trim
            params = {'zpf': 2, 'expf_us': 5.0, 'trim': trim_range}
            complex_ft_pipeline = pipe.compute_ft(**params)
            complex_ft_functional = ftmw.compute_ft(functional_file, **params)
            
            # Verify trim was applied correctly
            min_freq, max_freq = trim_range
            assert complex_ft_pipeline.freq_array.min() >= min_freq, f"Pipeline trim min failed for range {trim_range}"
            assert complex_ft_pipeline.freq_array.max() <= max_freq, f"Pipeline trim max failed for range {trim_range}"
            assert complex_ft_functional.freq_array.min() >= min_freq, f"Functional trim min failed for range {trim_range}"
            assert complex_ft_functional.freq_array.max() <= max_freq, f"Functional trim max failed for range {trim_range}"
            
            # Compare results
            self._compare_complex_ft_objects(
                complex_ft_pipeline, complex_ft_functional, 
                f"Trim range {trim_range}"
            )
    
    def _compare_complex_ft_objects(self, ft1: ComplexFT, ft2: ComplexFT, context: str):
        """Compare two ComplexFT objects for numerical consistency."""
        # Frequency arrays should be identical
        np.testing.assert_array_equal(
            ft1.freq_array, ft2.freq_array,
            err_msg=f"{context}: Frequency arrays differ"
        )
        
        # Complex spectra should be numerically equivalent
        np.testing.assert_allclose(
            ft1.complex_spectrum, ft2.complex_spectrum,
            rtol=1e-10, atol=1e-15,
            err_msg=f"{context}: Complex spectra differ beyond tolerance"
        )
        
        # Magnitude spectra should be consistent
        np.testing.assert_allclose(
            ft1.magnitude_spectrum, ft2.magnitude_spectrum,
            rtol=1e-10, atol=1e-15,
            err_msg=f"{context}: Magnitude spectra differ beyond tolerance"
        )
        
        # Real and imaginary parts should be consistent
        np.testing.assert_allclose(
            ft1.real_spectrum, ft2.real_spectrum,
            rtol=1e-10, atol=1e-15,
            err_msg=f"{context}: Real spectra differ beyond tolerance"
        )
        
        np.testing.assert_allclose(
            ft1.imag_spectrum, ft2.imag_spectrum,
            rtol=1e-10, atol=1e-15,
            err_msg=f"{context}: Imaginary spectra differ beyond tolerance"
        )
    
    def _compare_fid_objects(self, fid1: FID, fid2: FID, context: str):
        """Compare two FID objects for consistency."""
        # Data arrays should be identical
        np.testing.assert_array_equal(
            fid1.data, fid2.data,
            err_msg=f"{context}: FID data arrays differ"
        )
        
        # Metadata should be identical
        assert fid1.spacing == fid2.spacing, f"{context}: FID spacing differs"
        assert fid1.probe_freq_mhz == fid2.probe_freq_mhz, f"{context}: Probe frequency differs"
        assert fid1.sideband == fid2.sideband, f"{context}: Sideband differs"
        assert fid1.shots == fid2.shots, f"{context}: Shots differ"
        assert fid1.duration_us == fid2.duration_us, f"{context}: Duration differs"
        assert fid1.n_points == fid2.n_points, f"{context}: Number of points differs"
    
    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds."""
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"CLI command failed:\n"
                f"Command: ftmwpipeline {' '.join(args)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
        
        return result.stdout, result.stderr


@pytest.mark.parameter_persistence
class TestParameterPersistence:
    """Test parameter persistence across all interfaces."""
    
    def test_parameter_persistence_across_interfaces(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Verify saved parameters work across all interfaces."""
        test_file = temp_ftmw_dir / "param_persistence.ftmw"
        
        # Save parameters with Pipeline class
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        pipe.visualize_ft(save_params=True, interactive=False, **standard_ft_params)
        
        # Load with functional API using saved params
        complex_ft_functional_saved = ftmw.compute_ft(test_file, from_saved_params=True)
        
        # Load with Pipeline class using saved params
        complex_ft_pipeline_saved = pipe.compute_ft(from_saved_params=True)
        
        # Load with explicit parameters for comparison
        complex_ft_explicit = pipe.compute_ft(**standard_ft_params)
        
        # All should produce identical results
        self._compare_complex_ft_results(
            complex_ft_pipeline_saved, complex_ft_explicit,
            "Pipeline saved vs explicit"
        )
        self._compare_complex_ft_results(
            complex_ft_functional_saved, complex_ft_explicit,
            "Functional saved vs explicit"
        )
        self._compare_complex_ft_results(
            complex_ft_pipeline_saved, complex_ft_functional_saved,
            "Pipeline saved vs Functional saved"
        )
    
    def test_functional_api_parameter_saving(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Test functional API parameter saving works with all interfaces."""
        test_file = temp_ftmw_dir / "functional_param_save.ftmw"
        
        # Create file with Pipeline class
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        
        # Save parameters using functional API
        params_to_save = {
            'zpf': standard_ft_params['zpf'],
            'expf_us': standard_ft_params['expf_us'],
            'trim_min_mhz': standard_ft_params['trim'][0],
            'trim_max_mhz': standard_ft_params['trim'][1]
        }
        ftmw.save_ft_parameters(test_file, params_to_save)
        
        # Load with both interfaces using saved params
        complex_ft_pipeline = pipe.compute_ft(from_saved_params=True)
        complex_ft_functional = ftmw.compute_ft(test_file, from_saved_params=True)
        
        # Compare results
        self._compare_complex_ft_results(
            complex_ft_pipeline, complex_ft_functional,
            "Pipeline vs Functional using saved params"
        )
    
    def test_pipeline_to_functional_parameter_transfer(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Test parameter transfer from Pipeline class to functional API."""
        test_file = temp_ftmw_dir / "pipeline_to_functional_params.ftmw"
        
        # Create file with Pipeline class and save parameters
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        pipe.visualize_ft(save_params=True, interactive=False, **standard_ft_params)
        
        # Load with functional API using saved params
        complex_ft_functional_saved = ftmw.compute_ft(test_file, from_saved_params=True)
        complex_ft_pipeline_saved = pipe.compute_ft(from_saved_params=True)
        
        # Compare results
        self._compare_complex_ft_results(
            complex_ft_pipeline_saved, complex_ft_functional_saved,
            "Pipeline vs Functional using Pipeline-saved params"
        )
    
    def _compare_complex_ft_results(self, ft1: ComplexFT, ft2: ComplexFT, context: str):
        """Compare ComplexFT results for parameter persistence tests."""
        np.testing.assert_allclose(
            ft1.complex_spectrum, ft2.complex_spectrum,
            rtol=1e-12, atol=1e-15,
            err_msg=f"{context}: Complex spectra differ"
        )
        
        np.testing.assert_array_equal(
            ft1.freq_array, ft2.freq_array,
            err_msg=f"{context}: Frequency arrays differ"
        )
    
    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds."""
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"CLI command failed:\n"
                f"Command: ftmwpipeline {' '.join(args)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )


@pytest.mark.file_portability
class TestFilePortability:
    """Test .ftmw file portability between interfaces."""
    
    def test_file_portability_pipeline_to_functional(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Verify .ftmw files work across interfaces: Pipeline → Functional API."""
        test_file = temp_ftmw_dir / "pipeline_to_functional.ftmw"
        
        # Create with Pipeline class
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)
        
        # Process with functional API
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # Results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_functional)
    
    def test_file_portability_functional_to_cli(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Verify .ftmw files work across interfaces: Functional API → CLI."""
        test_file = temp_ftmw_dir / "functional_to_cli.ftmw"
        
        # Create with functional API
        ftmw.import_data(test_file, source=exp_2638_data_path)
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # Process with CLI
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "ft-process", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        
        # Load result with functional API for comparison
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # Results should be identical
        self._verify_file_portability(complex_ft_functional, complex_ft_cli)
    
    def test_file_portability_cli_to_pipeline(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Verify .ftmw files work across interfaces: CLI → Pipeline class."""
        test_file = temp_ftmw_dir / "cli_to_pipeline.ftmw"
        
        # Create with CLI
        self._run_cli_command([
            "data-load", str(test_file),
            "--source", exp_2638_data_path
        ])
        
        # Process with Pipeline class
        pipe = Pipeline.open(test_file)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)
        
        # Process with CLI for comparison
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "ft-process", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # Results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_cli)
    
    def test_round_trip_portability(self, exp_2638_data_path, temp_ftmw_dir, standard_ft_params):
        """Test complete round-trip portability: CLI → Pipeline → Functional → CLI."""
        test_file = temp_ftmw_dir / "round_trip.ftmw"
        
        # Start with CLI
        self._run_cli_command([
            "data-load", str(test_file),
            "--source", exp_2638_data_path
        ])
        
        # Process with Pipeline class
        pipe = Pipeline.open(test_file)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)
        
        # Process with functional API
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # Process with CLI again
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "ft-process", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)
        
        # All results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_functional)
        self._verify_file_portability(complex_ft_functional, complex_ft_cli)
        self._verify_file_portability(complex_ft_pipeline, complex_ft_cli)
    
    def _verify_file_portability(self, ft1: ComplexFT, ft2: ComplexFT):
        """Verify two ComplexFT objects are identical for portability testing."""
        # Strict comparison for portability
        np.testing.assert_array_equal(
            ft1.freq_array, ft2.freq_array,
            err_msg="Frequency arrays should be identical across interfaces"
        )
        
        np.testing.assert_allclose(
            ft1.complex_spectrum, ft2.complex_spectrum,
            rtol=1e-12, atol=1e-15,
            err_msg="Complex spectra should be identical across interfaces"
        )
    
    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds."""
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"CLI command failed:\n"
                f"Command: ftmwpipeline {' '.join(args)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )


@pytest.mark.error_consistency
class TestErrorConsistency:
    """Test consistent error handling across interfaces."""
    
    def test_missing_file_errors(self, temp_ftmw_dir):
        """Test all interfaces handle missing files consistently."""
        nonexistent_file = temp_ftmw_dir / "nonexistent.ftmw"
        
        # Pipeline class should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            Pipeline.open(nonexistent_file)
        
        # Functional API should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ftmw.load_fid(nonexistent_file)
        
        with pytest.raises(FileNotFoundError):
            ftmw.compute_ft(nonexistent_file)
        
        # CLI should fail with non-zero return code
        result = subprocess.run(
            ["ftmwpipeline", "ft-process", str(nonexistent_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        assert result.returncode != 0, "CLI should fail with missing file"
    
    def test_invalid_source_errors(self, temp_ftmw_dir):
        """Test all interfaces handle invalid source data consistently."""
        test_file = temp_ftmw_dir / "invalid_source.ftmw"
        nonexistent_source = "nonexistent_source_directory"
        
        # Pipeline class should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            Pipeline.create(test_file, source=nonexistent_source)
        
        # Functional API should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ftmw.import_data(test_file, source=nonexistent_source)
        
        # CLI should fail with non-zero return code
        result = subprocess.run(
            ["ftmwpipeline", "data-load", str(test_file), "--source", nonexistent_source],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        assert result.returncode != 0, "CLI should fail with invalid source"
    
    def test_invalid_parameter_errors(self, exp_2638_data_path, temp_ftmw_dir):
        """Test all interfaces handle invalid parameters consistently."""
        test_file = temp_ftmw_dir / "invalid_params.ftmw"
        
        # Create valid file first
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        ftmw.import_data(test_file, source=exp_2638_data_path)
        
        # Test invalid zpf parameter (negative value)
        with pytest.raises((ValueError, RuntimeError)):
            pipe.compute_ft(zpf=-1)
        
        with pytest.raises((ValueError, RuntimeError)):
            ftmw.compute_ft(test_file, zpf=-1)
        
        # CLI should also fail
        result = subprocess.run(
            ["ftmwpipeline", "ft-process", str(test_file), "--zpf", "-1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        assert result.returncode != 0, "CLI should fail with invalid zpf parameter"
    
    def test_corruption_detection_consistency(self, exp_2638_data_path, temp_ftmw_dir):
        """Test all interfaces detect file corruption consistently."""
        test_file = temp_ftmw_dir / "corruption_test.ftmw"
        
        # Create valid file
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        
        # Corrupt the file by truncating it
        with open(test_file, 'r+b') as f:
            f.truncate(100)  # Truncate to 100 bytes
        
        # All interfaces should detect corruption
        validation_pipeline = pipe.validate()
        assert not validation_pipeline['valid'], "Pipeline should detect corruption"
        
        validation_functional = ftmw.validate_pipeline(test_file)
        assert not validation_functional['valid'], "Functional API should detect corruption"
        
        # CLI should also detect corruption (though exact command may vary)
        result = subprocess.run(
            ["ftmwpipeline", "data-visualize", str(test_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        assert result.returncode != 0, "CLI should fail with corrupted file"