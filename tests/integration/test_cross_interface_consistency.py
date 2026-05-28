"""
Test cross-interface consistency for the FTMW Pipeline.

This module tests that all three interfaces (Pipeline class, functional API, CLI)
produce identical results when given the same parameters and .ftmw files.
Focus is purely on functional consistency - NO performance testing.

Performance strategy:
  - Identity tests (TestIdenticalResults) consume module-scoped trio fixtures
    built once per module run.  The files are read-only; no test may mutate them.
  - Parameter-persistence and portability tests each need a writable private copy
    of the baseline pipeline.  They shutil.copy from the session-scoped
    baseline_2638_stage1 / baseline_2638_stage2 fixtures (copying an HDF5 file
    is ~milliseconds; rebuilding from the raw FID is seconds).
  - test_identical_trim_behavior is intentionally left with per-test builds
    because it exercises three distinct trim ranges; those results cannot be
    shared with the standard-params fixtures.
  - TestErrorConsistency tests are lightweight and do not require sharing.
"""

import pytest
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from ftmwpipeline import Pipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import ComplexFT, FID
from ftmwpipeline.preprocessing.noise_estimation import NoiseResult


@pytest.mark.cross_interface
class TestIdenticalResults:
    """Test that interfaces produce identical numerical results."""

    def test_identical_complex_ft_results(
        self, cross_interface_stage1_trio, standard_ft_params
    ):
        """Verify all interfaces produce identical ComplexFT for same parameters.

        The trio files were built independently via Pipeline / api / CLI; here we
        just load the already-computed FT from each and compare.  The files are
        NOT mutated.
        """
        paths = cross_interface_stage1_trio

        # Load FT results from the independently-built files
        complex_ft_pipeline = ftmw.compute_ft(paths["pipeline"], **standard_ft_params)
        complex_ft_functional = ftmw.compute_ft(paths["functional"], **standard_ft_params)
        complex_ft_cli = ftmw.compute_ft(paths["cli"], **standard_ft_params)

        # Compare results using strict tolerance
        self._compare_complex_ft_objects(complex_ft_pipeline, complex_ft_functional, "Pipeline vs Functional")
        self._compare_complex_ft_objects(complex_ft_pipeline, complex_ft_cli, "Pipeline vs CLI")
        self._compare_complex_ft_objects(complex_ft_functional, complex_ft_cli, "Functional vs CLI")

    def test_identical_fid_loading(self, cross_interface_stage1_trio):
        """Verify all interfaces load identical FID data.

        The trio files each imported 2638 via a different interface; we load
        the FID from each to prove all three produce the same raw data.
        """
        paths = cross_interface_stage1_trio

        fid_pipeline = ftmw.load_fid(paths["pipeline"])
        fid_functional = ftmw.load_fid(paths["functional"])
        fid_cli = ftmw.load_fid(paths["cli"])

        # Compare FID objects
        self._compare_fid_objects(fid_pipeline, fid_functional, "Pipeline vs Functional")
        self._compare_fid_objects(fid_pipeline, fid_cli, "Pipeline vs CLI")
        self._compare_fid_objects(fid_functional, fid_cli, "Functional vs CLI")

    def test_identical_trim_behavior(self, exp_2638_data_path, temp_ftmw_dir):
        """Verify trim parameters work identically across interfaces.

        This test exercises three different trim ranges that differ from the
        standard_ft_params, so it cannot reuse the shared trio fixtures.
        Each sub-case builds its own pair of files.
        """
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

    def test_identical_noise_estimation_results(
        self, cross_interface_stage1_trio, tmp_path
    ):
        """Verify all interfaces produce identical NoiseResult for same parameters.

        Cross-interface identity only requires one parameter set (default).
        Parameter-variation behaviour is already covered by unit noise tests
        in tests/unit/io/test_noise_result_serialization.py.

        We copy the module-scoped stage1 trio files into this test's own tmp
        dir (fast: millisecond copy vs. seconds to rebuild Stage 0+1), then
        run estimate_noise independently on each copy.  The shared trio files
        are NOT mutated.
        """
        paths = cross_interface_stage1_trio

        # Copy each trio file into a private per-test location before mutating
        p_copy = tmp_path / "pipeline_noise.ftmw"
        f_copy = tmp_path / "functional_noise.ftmw"
        c_copy = tmp_path / "cli_noise.ftmw"
        shutil.copy(paths["pipeline"], p_copy)
        shutil.copy(paths["functional"], f_copy)
        shutil.copy(paths["cli"], c_copy)

        # Run estimate_noise (default params) independently on each copy
        noise_result_pipeline = ftmw.estimate_noise(p_copy)
        noise_result_functional = ftmw.estimate_noise(f_copy)
        noise_result_cli = ftmw.estimate_noise(c_copy)

        # Compare results using bit-perfect consistency
        self._compare_noise_results(
            noise_result_pipeline, noise_result_functional,
            "default: Pipeline vs Functional"
        )
        self._compare_noise_results(
            noise_result_pipeline, noise_result_cli,
            "default: Pipeline vs CLI"
        )
        self._compare_noise_results(
            noise_result_functional, noise_result_cli,
            "default: Functional vs CLI"
        )

    def test_identical_tau_calibration_results(
        self, cross_interface_stage1_trio, tmp_path
    ):
        """Verify all interfaces produce identical TauCalibrationResult.

        Stage 2b runs the STFT tau calibration on the FID. Each interface
        should land on bit-identical results (same FID + same Stage 1 + same
        default knobs).
        """
        paths = cross_interface_stage1_trio

        p_copy = tmp_path / "pipeline_tau.ftmw"
        f_copy = tmp_path / "functional_tau.ftmw"
        c_copy = tmp_path / "cli_tau.ftmw"
        shutil.copy(paths["pipeline"], p_copy)
        shutil.copy(paths["functional"], f_copy)
        shutil.copy(paths["cli"], c_copy)

        # Each interface needs Stage 2 first.
        for f in (p_copy, f_copy, c_copy):
            ftmw.estimate_noise(f)

        # Run calibration via the three interfaces. Each uses defaults so
        # the calibration knobs are identical across interfaces.
        pipe = Pipeline.open(p_copy)
        tc_pipeline = pipe.calibrate_tau()
        tc_functional = ftmw.calibrate_tau(f_copy)
        self._run_cli_command(["calibrate-tau", str(c_copy)])
        tc_cli = ftmw.load_tau_calibration(c_copy)

        # Bit-identical scalars; per-bin arrays bit-identical too because the
        # algorithm is deterministic (no RNG, integer-grid FFT).
        for a, b, ctx in (
            (tc_pipeline, tc_functional, "Pipeline vs Functional"),
            (tc_pipeline, tc_cli, "Pipeline vs CLI"),
        ):
            assert a.tau_maj_us == b.tau_maj_us, f"{ctx}: tau_maj differs"
            assert a.sigma_tau_us == b.sigma_tau_us, f"{ctx}: sigma_tau differs"
            assert a.n_contributors == b.n_contributors, f"{ctx}: n_contributors differs"
            assert a.n_spur_bins == b.n_spur_bins, f"{ctx}: n_spur_bins differs"
            np.testing.assert_array_equal(
                a.contributor_taus_us, b.contributor_taus_us,
                err_msg=f"{ctx}: contributor_taus_us differ",
            )
            np.testing.assert_array_equal(
                a.contributor_snrs, b.contributor_snrs,
                err_msg=f"{ctx}: contributor_snrs differ",
            )
            assert a.bimodality.delta_aic == b.bimodality.delta_aic, (
                f"{ctx}: GMM delta_aic differs"
            )
            assert len(a.spur_clusters) == len(b.spur_clusters), (
                f"{ctx}: spur cluster count differs"
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

    def _compare_noise_results(self, result1, result2, context: str):
        """Compare NoiseResult objects for bit-perfect consistency."""
        assert isinstance(result1, NoiseResult), f"{context}: First result should be NoiseResult"
        assert isinstance(result2, NoiseResult), f"{context}: Second result should be NoiseResult"

        # RMS noise must be bit-perfect identical
        np.testing.assert_array_equal(
            result1.rms_noise, result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be bit-perfect identical"
        )

        # Noise masks must be identical
        np.testing.assert_array_equal(
            result1.noise_mask, result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical"
        )

        # Verify basic properties
        assert result1.rms_noise.shape == result2.rms_noise.shape, f"{context}: RMS shape mismatch"
        assert result1.noise_mask.dtype == result2.noise_mask.dtype == bool, f"{context}: Mask should be boolean"
        assert np.all(result1.rms_noise > 0), f"{context}: RMS values should be positive"
        assert np.all(result2.rms_noise > 0), f"{context}: RMS values should be positive"

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
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestParameterPersistence:
    """Test parameter persistence across all interfaces.

    Each test here mutates its own private copy of the pipeline file.
    Rather than rebuilding from the raw FID (slow), each test copies the
    session-scoped baseline_2638_stage1 file into a per-test tmp location.
    Copying an HDF5 file is ~milliseconds vs. seconds for import+FT.

    The legacy per-knob kwarg + ``from_saved_params=True`` paths are part
    of the back-compat contract these tests cover; the matching
    deprecation warnings are expected and suppressed.
    """

    def test_parameter_persistence_across_interfaces(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Verify saved parameters work across all interfaces."""
        test_file = tmp_path / "param_persistence.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)
        pipe = Pipeline.open(test_file)

        # Save parameters with Pipeline class (this mutates the file)
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

    def test_functional_api_parameter_saving(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Test functional API parameter saving works with all interfaces."""
        test_file = tmp_path / "functional_param_save.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)
        pipe = Pipeline.open(test_file)

        # Save parameters using functional API (this mutates the file)
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

    def test_pipeline_to_functional_parameter_transfer(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Test parameter transfer from Pipeline class to functional API."""
        test_file = tmp_path / "pipeline_to_functional_params.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)
        pipe = Pipeline.open(test_file)

        # Save parameters with Pipeline class (mutates file)
        pipe.visualize_ft(save_params=True, interactive=False, **standard_ft_params)

        # Load with functional API using saved params
        complex_ft_functional_saved = ftmw.compute_ft(test_file, from_saved_params=True)
        complex_ft_pipeline_saved = pipe.compute_ft(from_saved_params=True)

        # Compare results
        self._compare_complex_ft_results(
            complex_ft_pipeline_saved, complex_ft_functional_saved,
            "Pipeline vs Functional using Pipeline-saved params"
        )

    def test_noise_parameter_persistence_across_interfaces(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Verify noise parameter save/load works across interfaces."""
        test_file = tmp_path / "noise_param_persistence.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)
        # baseline_2638_stage1 already has Stage 0+1; open it and proceed to Stage 2
        pipe = Pipeline.open(test_file)

        # Estimate noise with the parameters we want to test (mutates file)
        noise_params = {
            'skew_target': 0.7,
            'min_bin_fraction': 1/32,
            'smoothing_window_mhz': 100.0
        }
        pipe.estimate_noise(**noise_params)

        # Save parameters using Pipeline class visualize_noise (mutates file)
        pipe.visualize_noise(save_params=True, interactive=False)

        # Load with functional API using saved params
        noise_result_functional_saved = ftmw.estimate_noise(test_file, from_saved_params=True)

        # Load with Pipeline class using saved params
        noise_result_pipeline_saved = pipe.estimate_noise(from_saved_params=True)

        # Load with explicit parameters for comparison
        noise_result_explicit = pipe.estimate_noise(**noise_params)

        # All should produce identical results
        self._compare_noise_results(noise_result_pipeline_saved, noise_result_explicit, "Pipeline saved vs explicit")
        self._compare_noise_results(noise_result_functional_saved, noise_result_explicit, "Functional saved vs explicit")
        self._compare_noise_results(noise_result_pipeline_saved, noise_result_functional_saved, "Pipeline saved vs Functional saved")

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

    def _compare_noise_results(self, result1, result2, context: str):
        """Compare NoiseResult objects for bit-perfect consistency."""
        assert isinstance(result1, NoiseResult), f"{context}: First result should be NoiseResult"
        assert isinstance(result2, NoiseResult), f"{context}: Second result should be NoiseResult"

        # RMS noise must be bit-perfect identical
        np.testing.assert_array_equal(
            result1.rms_noise, result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be bit-perfect identical"
        )

        # Noise masks must be identical
        np.testing.assert_array_equal(
            result1.noise_mask, result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical"
        )

        # Verify basic properties
        assert result1.rms_noise.shape == result2.rms_noise.shape, f"{context}: RMS shape mismatch"
        assert result1.noise_mask.dtype == result2.noise_mask.dtype == bool, f"{context}: Mask should be boolean"
        assert np.all(result1.rms_noise > 0), f"{context}: RMS values should be positive"
        assert np.all(result2.rms_noise > 0), f"{context}: RMS values should be positive"

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
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFilePortability:
    """Test .ftmw file portability between interfaces.

    Each test needs a writable file that gets modified (FT re-computed or noise
    added by a different interface).  Rather than rebuilding from scratch, each
    test copies the session-scoped baseline into its own tmp location.

    Round-trips exercise the legacy per-knob kwarg path for
    ``estimate_noise``; the matching deprecation warnings are expected
    and suppressed.
    """

    def test_file_portability_pipeline_to_functional(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Verify .ftmw files work across interfaces: Pipeline → Functional API."""
        test_file = tmp_path / "pipeline_to_functional.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)

        # Open with Pipeline (file already has Stage 0+1 from baseline)
        pipe = Pipeline.open(test_file)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)

        # Process with functional API on the same file
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)

        # Results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_functional)

    def test_file_portability_functional_to_cli(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Verify .ftmw files work across interfaces: Functional API → CLI."""
        test_file = tmp_path / "functional_to_cli.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)

        # Load existing FT from the file (Stage 0+1 already done)
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)

        # Process with CLI (re-run compute-ft on the same file to test CLI interop)
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "compute-ft", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])

        # Load result with functional API for comparison
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)

        # Results should be identical
        self._verify_file_portability(complex_ft_functional, complex_ft_cli)

    def test_file_portability_cli_to_pipeline(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Verify .ftmw files work across interfaces: CLI → Pipeline class."""
        test_file = tmp_path / "cli_to_pipeline.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)

        # Process with Pipeline class (file already has Stage 0+1)
        pipe = Pipeline.open(test_file)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)

        # Process with CLI for comparison
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "compute-ft", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)

        # Results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_cli)

    def test_round_trip_portability(
        self, baseline_2638_stage1, tmp_path, standard_ft_params
    ):
        """Test complete round-trip portability: baseline → Pipeline → Functional → CLI."""
        test_file = tmp_path / "round_trip.ftmw"
        shutil.copy(baseline_2638_stage1, test_file)

        # Process with Pipeline class
        pipe = Pipeline.open(test_file)
        complex_ft_pipeline = pipe.compute_ft(**standard_ft_params)

        # Process with functional API
        complex_ft_functional = ftmw.compute_ft(test_file, **standard_ft_params)

        # Process with CLI again
        zpf, expf_us = standard_ft_params['zpf'], standard_ft_params['expf_us']
        trim_min, trim_max = standard_ft_params['trim']
        self._run_cli_command([
            "compute-ft", str(test_file),
            "--zpf", str(zpf),
            "--expf_us", str(expf_us),
            "--trim", f"{trim_min}:{trim_max}"
        ])
        complex_ft_cli = ftmw.compute_ft(test_file, **standard_ft_params)

        # All results should be identical
        self._verify_file_portability(complex_ft_pipeline, complex_ft_functional)
        self._verify_file_portability(complex_ft_functional, complex_ft_cli)
        self._verify_file_portability(complex_ft_pipeline, complex_ft_cli)

    def test_noise_file_portability_round_trip(
        self, baseline_2638_stage2, tmp_path, standard_ft_params
    ):
        """Test .ftmw files with NoiseResult work across interfaces.

        Copies the session-scoped stage2 baseline (Stage 0+1+2 already done)
        to avoid rebuilding the expensive noise estimation from scratch.
        """
        test_file = tmp_path / "noise_portability.ftmw"
        shutil.copy(baseline_2638_stage2, test_file)
        pipe = Pipeline.open(test_file)

        # Load noise result from the existing Stage 2 (default params in baseline)
        noise_result_pipeline = pipe.estimate_noise(skew_target=0.631, min_bin_fraction=1/64)

        # Process with functional API
        noise_result_functional = ftmw.estimate_noise(test_file, skew_target=0.631, min_bin_fraction=1/64)

        # Process with CLI (just verify CLI can process the file)
        self._run_cli_command([
            "estimate-noise", str(test_file),
            "--skew-target", "0.631", "--min-bin-fraction", str(1/64)
        ])
        noise_result_cli = ftmw.estimate_noise(test_file, skew_target=0.631, min_bin_fraction=1/64)

        # Results should be identical across interfaces
        self._verify_noise_portability(noise_result_pipeline, noise_result_functional, "Pipeline to Functional portability")
        self._verify_noise_portability(noise_result_functional, noise_result_cli, "Functional to CLI portability")
        self._verify_noise_portability(noise_result_pipeline, noise_result_cli, "Pipeline to CLI portability")

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

    def _verify_noise_portability(self, result1, result2, context: str):
        """Verify two NoiseResult objects are identical for portability testing."""
        assert isinstance(result1, NoiseResult), f"{context}: First result should be NoiseResult"
        assert isinstance(result2, NoiseResult), f"{context}: Second result should be NoiseResult"

        # Strict comparison for portability
        np.testing.assert_array_equal(
            result1.rms_noise, result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be identical"
        )

        np.testing.assert_array_equal(
            result1.noise_mask, result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical"
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
            ["ftmwpipeline", "compute-ft", str(nonexistent_file)],
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
            ["ftmwpipeline", "import-data", str(test_file), "--source", nonexistent_source],
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
            ["ftmwpipeline", "compute-ft", str(test_file), "--zpf", "-1"],
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
            ["ftmwpipeline", "visualize-data", str(test_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        assert result.returncode != 0, "CLI should fail with corrupted file"

    def test_noise_stage_dependency_errors(self, exp_2638_data_path, temp_ftmw_dir):
        """Test consistent error handling for missing Stage 1 dependency."""
        test_file = temp_ftmw_dir / "incomplete_pipeline.ftmw"

        # Create file with only Stage 0 (no Stage 1)
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        # Don't call compute_ft() - missing Stage 1

        # All interfaces should raise appropriate errors for missing Stage 1
        with pytest.raises((ValueError, RuntimeError), match="Stage 1.*must be completed"):
            pipe.estimate_noise()

        with pytest.raises((ValueError, RuntimeError), match="Stage 1.*must be completed"):
            ftmw.estimate_noise(test_file)

        # CLI should fail with non-zero return code
        result = subprocess.run(
            ["ftmwpipeline", "estimate-noise", str(test_file)],
            capture_output=True, text=True, check=False, timeout=10
        )
        assert result.returncode != 0, "CLI should fail when Stage 1 missing"
        assert "Stage 1" in result.stderr or "Stage 1" in result.stdout, "Error should mention Stage 1"
