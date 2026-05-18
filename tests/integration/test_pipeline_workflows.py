"""
Test individual interface workflows for functional correctness.

This module tests each interface (Pipeline class, functional API, CLI) independently
to ensure they work correctly with .ftmw files and produce valid results.
Focus is purely on functional correctness - NO performance testing.
"""

import pytest
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ftmwpipeline import Pipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import ComplexFT, FID


@pytest.mark.single_interface
class TestPipelineClassWorkflows:
    """Test Pipeline class workflows for functional correctness."""
    
    def test_pipeline_class_basic_workflow(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test Pipeline class creates valid .ftmw files and processes data correctly."""
        # Pipeline.create() → load_data() → compute_ft() → visualize_ft()
        
        # Stage 0: Create pipeline from data
        pipe = Pipeline.create(temp_ftmw_file, source=exp_2638_data_path)
        
        # Verify file was created
        assert temp_ftmw_file.exists(), "Pipeline file not created"
        
        # Verify pipeline info is valid
        info = pipe.info()
        assert info['valid'], f"Pipeline file invalid: {info.get('errors', 'Unknown error')}"
        assert info['source_path'] == exp_2638_data_path
        assert info['format'] == 'blackchirp'
        assert 'stage0_fid_data' in info['completed_stages']
        
        # Stage 0: Load FID data 
        fid = pipe.load_data()
        assert isinstance(fid, FID), "load_data() should return FID object"
        assert fid.n_points > 0, "FID should have data points"
        assert fid.duration_us > 0, "FID should have positive duration"
        assert fid.probe_freq_mhz > 0, "FID should have valid probe frequency"
        
        # Stage 1: Compute FT with standard parameters
        complex_ft = pipe.compute_ft(**standard_ft_params)
        assert isinstance(complex_ft, ComplexFT), "compute_ft() should return ComplexFT object"
        assert complex_ft.n_points > 0, "ComplexFT should have frequency points"
        assert len(complex_ft.complex_spectrum) == complex_ft.n_points, "Complex spectrum length should match n_points"
        
        # Verify trim was applied
        if standard_ft_params['trim']:
            min_freq, max_freq = standard_ft_params['trim']
            assert complex_ft.freq_array.min() >= min_freq, "Trim minimum not applied"
            assert complex_ft.freq_array.max() <= max_freq, "Trim maximum not applied"
        
        # Stage 1: Visualize FT (should not raise errors)
        fig = pipe.visualize_ft(interactive=False, **standard_ft_params)
        assert fig is not None, "visualize_ft() should return figure object"
        
        # Verify pipeline state updated
        info_after = pipe.info()
        assert 'stage1_complex_ft' in info_after['completed_stages'], "Stage 1 should be marked complete"
    
    def test_pipeline_class_parameter_persistence(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test Pipeline class parameter saving and loading functionality."""
        # Create pipeline and process with save_params=True
        pipe = Pipeline.create(temp_ftmw_file, source=exp_2638_data_path)
        
        # Save parameters during visualization
        pipe.visualize_ft(save_params=True, interactive=False, **standard_ft_params)
        
        # Compute FT using saved parameters
        complex_ft_saved = pipe.compute_ft(from_saved_params=True)
        
        # Compute FT with explicit parameters
        complex_ft_explicit = pipe.compute_ft(**standard_ft_params)
        
        # Results should be identical
        np.testing.assert_allclose(
            complex_ft_saved.complex_spectrum,
            complex_ft_explicit.complex_spectrum,
            rtol=1e-12,
            err_msg="Saved parameters should produce identical results"
        )
        
        # Frequency arrays should be identical
        np.testing.assert_array_equal(
            complex_ft_saved.freq_array,
            complex_ft_explicit.freq_array,
            err_msg="Frequency arrays should be identical"
        )
    
    def test_pipeline_class_error_handling(self, temp_ftmw_file):
        """Test Pipeline class error handling for missing files and invalid operations."""
        # Test opening non-existent file
        with pytest.raises(FileNotFoundError):
            Pipeline.open("nonexistent_file.ftmw")
        
        # Test creating from non-existent source
        with pytest.raises(FileNotFoundError):
            Pipeline.create(temp_ftmw_file, source="nonexistent_source")
    
    def test_pipeline_class_file_validation(self, exp_2638_data_path, temp_ftmw_file):
        """Test Pipeline class file validation functionality."""
        # Create valid pipeline
        pipe = Pipeline.create(temp_ftmw_file, source=exp_2638_data_path)
        
        # Validate file
        validation_report = pipe.validate()
        assert validation_report['valid'], f"Validation failed: {validation_report.get('errors', 'Unknown')}"
        assert len(validation_report.get('errors', [])) == 0, "Should have no validation errors"


@pytest.mark.single_interface
class TestFunctionalAPIWorkflows:
    """Test functional API workflows for functional correctness."""
    
    def test_functional_api_basic_workflow(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test functional API creates valid .ftmw files and processes data correctly."""
        # ftmw.import_data() → ftmw.load_fid() → ftmw.compute_ft() → ftmw.visualize_ft()
        
        # Stage 0: Import data
        result = ftmw.import_data(temp_ftmw_file, source=exp_2638_data_path)
        
        # Verify import result
        assert result['status'] == 'success', f"Import failed: {result}"
        assert result['pipeline_file'] == str(temp_ftmw_file), "Wrong pipeline file path"
        assert result['format_name'] == 'blackchirp', "Wrong format detected"
        assert 'fid_metadata' in result, "FID metadata missing from import result"
        
        # Verify file was created
        assert temp_ftmw_file.exists(), "Pipeline file not created"
        
        # Stage 0: Load FID data
        fid = ftmw.load_fid(temp_ftmw_file)
        assert isinstance(fid, FID), "load_fid() should return FID object"
        assert fid.n_points > 0, "FID should have data points"
        assert fid.duration_us > 0, "FID should have positive duration"
        
        # Stage 1: Compute FT with standard parameters
        complex_ft = ftmw.compute_ft(temp_ftmw_file, **standard_ft_params)
        assert isinstance(complex_ft, ComplexFT), "compute_ft() should return ComplexFT object"
        assert complex_ft.n_points > 0, "ComplexFT should have frequency points"
        
        # Verify trim was applied
        if standard_ft_params['trim']:
            min_freq, max_freq = standard_ft_params['trim']
            assert complex_ft.freq_array.min() >= min_freq, "Trim minimum not applied"
            assert complex_ft.freq_array.max() <= max_freq, "Trim maximum not applied"
        
        # Stage 1: Visualize FT (should not raise errors)
        fig = ftmw.visualize_ft(temp_ftmw_file, interactive=False, **standard_ft_params)
        assert fig is not None, "visualize_ft() should return figure object"
        
        # Get pipeline info and verify file is valid
        info = ftmw.get_pipeline_info(temp_ftmw_file)
        assert info['valid'], f"Pipeline invalid: {info.get('errors', 'Unknown')}"
        # Note: Functional API doesn't maintain stage state like Pipeline class
    
    def test_functional_api_parameter_persistence(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test functional API parameter saving and loading functionality."""
        # Import and process with save_params=True
        ftmw.import_data(temp_ftmw_file, source=exp_2638_data_path)
        
        # Save parameters during visualization
        ftmw.visualize_ft(temp_ftmw_file, save_params=True, interactive=False, **standard_ft_params)
        
        # Compute FT using saved parameters
        complex_ft_saved = ftmw.compute_ft(temp_ftmw_file, from_saved_params=True)
        
        # Compute FT with explicit parameters
        complex_ft_explicit = ftmw.compute_ft(temp_ftmw_file, **standard_ft_params)
        
        # Results should be identical
        np.testing.assert_allclose(
            complex_ft_saved.complex_spectrum,
            complex_ft_explicit.complex_spectrum,
            rtol=1e-12,
            err_msg="Saved parameters should produce identical results"
        )
    
    def test_functional_api_parameter_saving(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test functional API direct parameter saving functionality."""
        # Import data
        ftmw.import_data(temp_ftmw_file, source=exp_2638_data_path)
        
        # Save parameters directly
        params_to_save = {
            'zpf': standard_ft_params['zpf'],
            'expf_us': standard_ft_params['expf_us'],
            'trim_min_mhz': standard_ft_params['trim'][0],
            'trim_max_mhz': standard_ft_params['trim'][1]
        }
        ftmw.save_ft_parameters(temp_ftmw_file, params_to_save)
        
        # Load using saved parameters
        complex_ft_saved = ftmw.compute_ft(temp_ftmw_file, from_saved_params=True)
        
        # Load with explicit parameters
        complex_ft_explicit = ftmw.compute_ft(temp_ftmw_file, **standard_ft_params)
        
        # Results should be identical
        np.testing.assert_allclose(
            complex_ft_saved.complex_spectrum,
            complex_ft_explicit.complex_spectrum,
            rtol=1e-12,
            err_msg="Directly saved parameters should produce identical results"
        )
    
    def test_functional_api_error_handling(self, temp_ftmw_file):
        """Test functional API error handling for missing files and invalid operations."""
        # Test loading from non-existent file
        with pytest.raises(FileNotFoundError):
            ftmw.load_fid("nonexistent_file.ftmw")
        
        # Test importing from non-existent source
        with pytest.raises(FileNotFoundError):
            ftmw.import_data(temp_ftmw_file, source="nonexistent_source")
        
        # Test computing FT on non-existent file
        with pytest.raises(FileNotFoundError):
            ftmw.compute_ft("nonexistent_file.ftmw")
    
    def test_functional_api_validation(self, exp_2638_data_path, temp_ftmw_file):
        """Test functional API validation functionality."""
        # Import data
        ftmw.import_data(temp_ftmw_file, source=exp_2638_data_path)
        
        # Validate pipeline
        validation_report = ftmw.validate_pipeline(temp_ftmw_file)
        assert validation_report['valid'], f"Validation failed: {validation_report.get('errors', 'Unknown')}"
        
        # Get pipeline info
        info = ftmw.get_pipeline_info(temp_ftmw_file)
        assert info['valid'], f"Pipeline info invalid: {info.get('error', 'Unknown')}"
        assert info['source_path'] == exp_2638_data_path
        
        # Get available stages
        stages = ftmw.list_available_stages(temp_ftmw_file)
        assert isinstance(stages, list), "list_available_stages should return list"
        assert 'stage1_complex_ft' in stages, "Stage 1 should be available"


@pytest.mark.single_interface
class TestCLIWorkflows:
    """Test CLI workflows for functional correctness."""
    
    def run_cli_command(self, args, check_return_code=True):
        """Run CLI command and return success/failure and output."""
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        if check_return_code and result.returncode != 0:
            pytest.fail(
                f"CLI command failed:\n"
                f"Command: ftmwpipeline {' '.join(args)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
        
        return result.returncode == 0, result.stdout, result.stderr
    
    def test_cli_basic_workflow(self, exp_2638_data_path, temp_ftmw_file, standard_ft_params):
        """Test CLI creates valid .ftmw files and processes data correctly."""
        # data-load → ft-process → ft-visualize (via subprocess)
        
        # Stage 0: Import data
        success, stdout, stderr = self.run_cli_command([
            "data-load", str(temp_ftmw_file), 
            "--source", exp_2638_data_path
        ])
        assert success, f"data-load failed: {stderr}"
        assert temp_ftmw_file.exists(), "Pipeline file not created by CLI"
        assert "Data import completed successfully" in stdout, "Import success message missing"
        
        # Stage 1: Process FT
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        
        success, stdout, stderr = self.run_cli_command([
            "ft-process", str(temp_ftmw_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        assert success, f"ft-process failed: {stderr}"
        assert "FT processing validation and parameter storage completed successfully" in stdout, "FT validation success message missing"
        
        # Stage 1: Visualize FT (non-interactive)
        success, stdout, stderr = self.run_cli_command([
            "ft-visualize", str(temp_ftmw_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}",
            "--no-interactive"
        ])
        assert success, f"ft-visualize failed: {stderr}"
        assert "Enhanced plot saved" in stdout, "Visualization success message missing"
        
        # Verify pipeline file is valid using functional API
        info = ftmw.get_pipeline_info(temp_ftmw_file)
        assert info['valid'], f"CLI-created pipeline invalid: {info.get('errors', 'Unknown')}"
    
    def test_cli_file_creation_only(self, exp_2638_data_path, temp_ftmw_file):
        """Test CLI file creation produces valid files usable by other interfaces."""
        # Import data with CLI
        success, stdout, stderr = self.run_cli_command([
            "data-load", str(temp_ftmw_file),
            "--source", exp_2638_data_path
        ])
        assert success, f"data-load failed: {stderr}"
        
        # Verify file can be used by functional API
        fid = ftmw.load_fid(temp_ftmw_file)
        assert isinstance(fid, FID), "CLI-created file should be readable by functional API"
        
        # Verify file can be used by Pipeline class
        pipe = Pipeline.open(temp_ftmw_file)
        info = pipe.info()
        assert info['valid'], f"CLI-created file should be valid for Pipeline class: {info.get('errors', 'Unknown')}"
    
    def test_cli_error_handling(self, temp_ftmw_file):
        """Test CLI error handling for missing files and invalid parameters."""
        # Test data-load with non-existent source
        success, stdout, stderr = self.run_cli_command([
            "data-load", str(temp_ftmw_file),
            "--source", "nonexistent_source"
        ], check_return_code=False)
        assert not success, "data-load should fail with non-existent source"
        
        # Test ft-process with non-existent file
        success, stdout, stderr = self.run_cli_command([
            "ft-process", "nonexistent_file.ftmw"
        ], check_return_code=False)
        assert not success, "ft-process should fail with non-existent file"
        
        # Test invalid trim range
        success, stdout, stderr = self.run_cli_command([
            "ft-process", str(temp_ftmw_file),
            "--trim", "invalid_range"
        ], check_return_code=False)
        assert not success, "ft-process should fail with invalid trim range"
    
    def test_cli_validation_commands(self):
        """Test CLI validation and info commands."""
        # Test validate command (allow failure since it tests internal components)
        success, stdout, stderr = self.run_cli_command(["validate"], check_return_code=False)
        # Command should run and produce output even if some components fail
        assert "ftmwpipeline installation validation" in stdout, "Validation output missing"
        
        # Test version command
        success, stdout, stderr = self.run_cli_command(["version"])
        assert success, f"version command failed: {stderr}"
        assert "ftmwpipeline" in stdout, "Version output missing"
        
        # Test data-info command
        success, stdout, stderr = self.run_cli_command(["data-info"])
        assert success, f"data-info command failed: {stderr}"
        assert "Available Data Formats" in stdout, "Data formats listing missing"