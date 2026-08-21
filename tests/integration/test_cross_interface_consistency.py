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

import shutil
import subprocess

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.data_structures import FID, ComplexFT
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
        complex_ft_functional = ftmw.compute_ft(
            paths["functional"], **standard_ft_params
        )
        complex_ft_cli = ftmw.compute_ft(paths["cli"], **standard_ft_params)

        # Compare results using strict tolerance
        self._compare_complex_ft_objects(
            complex_ft_pipeline, complex_ft_functional, "Pipeline vs Functional"
        )
        self._compare_complex_ft_objects(
            complex_ft_pipeline, complex_ft_cli, "Pipeline vs CLI"
        )
        self._compare_complex_ft_objects(
            complex_ft_functional, complex_ft_cli, "Functional vs CLI"
        )

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
        self._compare_fid_objects(
            fid_pipeline, fid_functional, "Pipeline vs Functional"
        )
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
            (25000, 45000),  # Wider range
        ]

        for i, trim_range in enumerate(trim_ranges):
            pipeline_file = temp_ftmw_dir / f"pipeline_trim_{i}.ftmw"
            functional_file = temp_ftmw_dir / f"functional_trim_{i}.ftmw"

            # Create files with different interfaces
            pipe = Pipeline.create(pipeline_file, source=exp_2638_data_path)
            ftmw.import_data(functional_file, source=exp_2638_data_path)

            # Compute FT with trim
            params = {"trim": trim_range}
            complex_ft_pipeline = pipe.compute_ft(**params)
            complex_ft_functional = ftmw.compute_ft(functional_file, **params)

            # Verify trim was applied correctly
            min_freq, max_freq = trim_range
            assert (
                complex_ft_pipeline.freq_array.min() >= min_freq
            ), f"Pipeline trim min failed for range {trim_range}"
            assert (
                complex_ft_pipeline.freq_array.max() <= max_freq
            ), f"Pipeline trim max failed for range {trim_range}"
            assert (
                complex_ft_functional.freq_array.min() >= min_freq
            ), f"Functional trim min failed for range {trim_range}"
            assert (
                complex_ft_functional.freq_array.max() <= max_freq
            ), f"Functional trim max failed for range {trim_range}"

            # Compare results
            self._compare_complex_ft_objects(
                complex_ft_pipeline, complex_ft_functional, f"Trim range {trim_range}"
            )

    def test_identical_noise_estimation_results(
        self, cross_interface_stage1_trio, tmp_path
    ):
        """Verify all interfaces produce identical NoiseResult for same parameters.

        Cross-interface identity only requires one parameter set (default).
        Parameter-variation behavior is already covered by unit noise tests
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

        # Run the estimator independently on each copy; this proves the
        # Stage 0/1 cross-build identity carries into Stage 2 on the trio.
        noise_result_pipeline = ftmw.estimate_noise(p_copy)
        noise_result_functional = ftmw.estimate_noise(f_copy)
        noise_result_cli = ftmw.estimate_noise(c_copy)

        # Compare results using bit-perfect consistency
        self._compare_noise_results(
            noise_result_pipeline,
            noise_result_functional,
            "default: Pipeline vs Functional",
        )
        self._compare_noise_results(
            noise_result_pipeline, noise_result_cli, "default: Pipeline vs CLI"
        )
        self._compare_noise_results(
            noise_result_functional, noise_result_cli, "default: Functional vs CLI"
        )

    def test_identical_noise_estimation_scatter(
        self, baseline_2638_stage1_raw, tmp_path
    ):
        """The scatter (high-pass) estimator must be bit-identical across the
        three interfaces.

        Run on the canonical unapodized FT — the grid scatter uses in production
        (an apodized/zero-padded grid would be denser and needlessly slow under scatter's
        broad-window smoothing). Stage 0/1 cross-build identity is already proven
        by ``test_identical_noise_estimation_results``; here we copy one raw
        baseline three ways and check the estimator + interface plumbing agree.
        Pipeline / functional return the NoiseResult in-process; the CLI persists
        it, so we reload the CLI file to compare.
        """
        from ftmwpipeline._internal.stage2_impl import load_noise_result_impl

        p_copy = tmp_path / "pipeline_scatter.ftmw"
        f_copy = tmp_path / "functional_scatter.ftmw"
        c_copy = tmp_path / "cli_scatter.ftmw"
        shutil.copy(baseline_2638_stage1_raw, p_copy)
        shutil.copy(baseline_2638_stage1_raw, f_copy)
        shutil.copy(baseline_2638_stage1_raw, c_copy)

        nr_pipeline = Pipeline.open(p_copy).estimate_noise()
        nr_functional = ftmw.estimate_noise(f_copy)
        self._run_cli_command(["noise", "run", str(c_copy)])
        nr_cli = load_noise_result_impl(c_copy)["noise_result"]

        assert nr_pipeline.bin_info["algorithm"] == "scatter_highpass_region_aware"
        self._compare_noise_results(
            nr_pipeline, nr_functional, "scatter: Pipeline vs Functional"
        )
        self._compare_noise_results(nr_pipeline, nr_cli, "scatter: Pipeline vs CLI")

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
        #
        # The Stage 2b auto-recommend pass (~50s on 2638) is unrelated to
        # this test's τ-identity assertions; opt out via a settings instance
        # for the in-process interfaces and a one-line preset YAML for CLI.
        from tests.integration._stage2b_helpers import (
            skip_auto_recommend_preset_yaml,
            skip_auto_recommend_settings,
        )

        skip = skip_auto_recommend_settings()
        skip_yaml = skip_auto_recommend_preset_yaml(tmp_path)
        pipe = Pipeline.open(p_copy)
        tc_pipeline = pipe.calibrate_tau(settings=skip)
        tc_functional = ftmw.calibrate_tau(f_copy, settings=skip)
        self._run_cli_command(
            [
                "tau",
                "run",
                str(c_copy),
                "--preset",
                str(skip_yaml),
            ]
        )
        tc_cli = ftmw.load_tau_calibration(c_copy)

        # Bit-identical scalars; per-bin arrays bit-identical too because the
        # algorithm is deterministic (no RNG, integer-grid FFT).
        for a, b, ctx in (
            (tc_pipeline, tc_functional, "Pipeline vs Functional"),
            (tc_pipeline, tc_cli, "Pipeline vs CLI"),
        ):
            assert a.tau_maj_us == b.tau_maj_us, f"{ctx}: tau_maj differs"
            assert a.sigma_tau_us == b.sigma_tau_us, f"{ctx}: sigma_tau differs"
            assert (
                a.n_contributors == b.n_contributors
            ), f"{ctx}: n_contributors differs"
            assert a.n_spur_bins == b.n_spur_bins, f"{ctx}: n_spur_bins differs"
            np.testing.assert_array_equal(
                a.contributor_taus_us,
                b.contributor_taus_us,
                err_msg=f"{ctx}: contributor_taus_us differ",
            )
            np.testing.assert_array_equal(
                a.contributor_snrs,
                b.contributor_snrs,
                err_msg=f"{ctx}: contributor_snrs differ",
            )
            assert (
                a.bimodality.delta_aic == b.bimodality.delta_aic
            ), f"{ctx}: GMM delta_aic differs"
            assert len(a.spur_clusters) == len(
                b.spur_clusters
            ), f"{ctx}: spur cluster count differs"

    @pytest.mark.integration
    def test_identical_timebase_calibration_results(
        self, baseline_2638_stage1, tmp_path
    ):
        """All interfaces produce identical TimebaseCalibrationResult.

        The scope-timebase self-calibration needs only Stage 0 (the FID) plus
        the canonical Stage 1 active-region bounds and an instrument clock
        declaration. The in-process interfaces pass the locked clocks via the
        ``clocks=`` argument; the CLI reads the same declaration persisted as
        ``spur.clocks`` so all three land on bit-identical results.
        """
        from ftmwpipeline.core.stage_fit_settings import (
            ClockSource,
            SpurSubSettings,
            StageFitSettings,
        )
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            save_stage_fit_settings_to_h5,
        )

        clocks = [
            ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
            ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
            ClockSource(freq_mhz=16000.0, locked=True, label="awg"),
        ]

        p_copy = tmp_path / "pipeline_tb.ftmw"
        f_copy = tmp_path / "functional_tb.ftmw"
        c_copy = tmp_path / "cli_tb.ftmw"
        shutil.copy(baseline_2638_stage1, p_copy)
        shutil.copy(baseline_2638_stage1, f_copy)
        shutil.copy(baseline_2638_stage1, c_copy)

        # Persist the same declaration the CLI path will read.
        save_stage_fit_settings_to_h5(
            str(c_copy),
            StageFitSettings(spur=SpurSubSettings(clocks=tuple(clocks))),
        )

        tc_pipeline = Pipeline.open(p_copy).calibrate_timebase(clocks=clocks)
        tc_functional = ftmw.calibrate_timebase(f_copy, clocks=clocks)
        self._run_cli_command(["timebase", "run", str(c_copy)])
        tc_cli = ftmw.load_timebase_calibration(c_copy)

        for a, b, ctx in (
            (tc_pipeline, tc_functional, "Pipeline vs Functional"),
            (tc_pipeline, tc_cli, "Pipeline vs CLI"),
        ):
            assert a.epsilon == b.epsilon, f"{ctx}: epsilon differs"
            assert a.sigma_epsilon == b.sigma_epsilon, f"{ctx}: sigma_epsilon differs"
            assert a.n_used == b.n_used, f"{ctx}: n_used differs"
            assert a.n_detected == b.n_detected, f"{ctx}: n_detected differs"
            assert a.lattice_g_mhz == b.lattice_g_mhz, f"{ctx}: lattice_g differs"
            assert len(a.tone_reads) == len(b.tone_reads), f"{ctx}: tone count differs"

    def _compare_complex_ft_objects(self, ft1: ComplexFT, ft2: ComplexFT, context: str):
        """Compare two ComplexFT objects for numerical consistency."""
        # Frequency arrays should be identical
        np.testing.assert_array_equal(
            ft1.freq_array,
            ft2.freq_array,
            err_msg=f"{context}: Frequency arrays differ",
        )

        # Complex spectra should be numerically equivalent
        np.testing.assert_allclose(
            ft1.complex_spectrum,
            ft2.complex_spectrum,
            rtol=1e-10,
            atol=1e-15,
            err_msg=f"{context}: Complex spectra differ beyond tolerance",
        )

        # Magnitude spectra should be consistent
        np.testing.assert_allclose(
            ft1.magnitude_spectrum,
            ft2.magnitude_spectrum,
            rtol=1e-10,
            atol=1e-15,
            err_msg=f"{context}: Magnitude spectra differ beyond tolerance",
        )

        # Real and imaginary parts should be consistent
        np.testing.assert_allclose(
            ft1.real_spectrum,
            ft2.real_spectrum,
            rtol=1e-10,
            atol=1e-15,
            err_msg=f"{context}: Real spectra differ beyond tolerance",
        )

        np.testing.assert_allclose(
            ft1.imag_spectrum,
            ft2.imag_spectrum,
            rtol=1e-10,
            atol=1e-15,
            err_msg=f"{context}: Imaginary spectra differ beyond tolerance",
        )

    def _compare_fid_objects(self, fid1: FID, fid2: FID, context: str):
        """Compare two FID objects for consistency."""
        # Data arrays should be identical
        np.testing.assert_array_equal(
            fid1.data, fid2.data, err_msg=f"{context}: FID data arrays differ"
        )

        # Metadata should be identical
        assert fid1.spacing == fid2.spacing, f"{context}: FID spacing differs"
        assert (
            fid1.probe_freq_mhz == fid2.probe_freq_mhz
        ), f"{context}: Probe frequency differs"
        assert fid1.sideband == fid2.sideband, f"{context}: Sideband differs"
        assert fid1.shots == fid2.shots, f"{context}: Shots differ"
        assert fid1.duration_us == fid2.duration_us, f"{context}: Duration differs"
        assert fid1.n_points == fid2.n_points, f"{context}: Number of points differs"

    def _compare_noise_results(self, result1, result2, context: str):
        """Compare NoiseResult objects for bit-perfect consistency."""
        assert isinstance(
            result1, NoiseResult
        ), f"{context}: First result should be NoiseResult"
        assert isinstance(
            result2, NoiseResult
        ), f"{context}: Second result should be NoiseResult"

        # RMS noise must be bit-perfect identical
        np.testing.assert_array_equal(
            result1.rms_noise,
            result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be bit-perfect identical",
        )

        # Noise masks must be identical
        np.testing.assert_array_equal(
            result1.noise_mask,
            result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical",
        )

        # Verify basic properties
        assert (
            result1.rms_noise.shape == result2.rms_noise.shape
        ), f"{context}: RMS shape mismatch"
        assert (
            result1.noise_mask.dtype == result2.noise_mask.dtype == bool
        ), f"{context}: Mask should be boolean"
        assert np.all(
            result1.rms_noise > 0
        ), f"{context}: RMS values should be positive"
        assert np.all(
            result2.rms_noise > 0
        ), f"{context}: RMS values should be positive"

    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds.

        Timeout is 120s rather than 30s because ``calibrate-tau`` now
        auto-runs the 3-way shape recommendation (Stage 2b
        ``auto_recommend=True`` by default), which adds ~50s on the
        2638 fixture beyond the τ calibration itself.
        """
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
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
            complex_ft_pipeline_saved, complex_ft_explicit, "Pipeline saved vs explicit"
        )
        self._compare_complex_ft_results(
            complex_ft_functional_saved,
            complex_ft_explicit,
            "Functional saved vs explicit",
        )
        self._compare_complex_ft_results(
            complex_ft_pipeline_saved,
            complex_ft_functional_saved,
            "Pipeline saved vs Functional saved",
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
            "trim_min_mhz": standard_ft_params["trim"][0],
            "trim_max_mhz": standard_ft_params["trim"][1],
        }
        ftmw.save_ft_parameters(test_file, params_to_save)

        # Load with both interfaces using saved params
        complex_ft_pipeline = pipe.compute_ft(from_saved_params=True)
        complex_ft_functional = ftmw.compute_ft(test_file, from_saved_params=True)

        # Compare results
        self._compare_complex_ft_results(
            complex_ft_pipeline,
            complex_ft_functional,
            "Pipeline vs Functional using saved params",
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
            complex_ft_pipeline_saved,
            complex_ft_functional_saved,
            "Pipeline vs Functional using Pipeline-saved params",
        )

    def _compare_complex_ft_results(self, ft1: ComplexFT, ft2: ComplexFT, context: str):
        """Compare ComplexFT results for parameter persistence tests."""
        np.testing.assert_allclose(
            ft1.complex_spectrum,
            ft2.complex_spectrum,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"{context}: Complex spectra differ",
        )

        np.testing.assert_array_equal(
            ft1.freq_array,
            ft2.freq_array,
            err_msg=f"{context}: Frequency arrays differ",
        )

    def _compare_noise_results(self, result1, result2, context: str):
        """Compare NoiseResult objects for bit-perfect consistency."""
        assert isinstance(
            result1, NoiseResult
        ), f"{context}: First result should be NoiseResult"
        assert isinstance(
            result2, NoiseResult
        ), f"{context}: Second result should be NoiseResult"

        # RMS noise must be bit-perfect identical
        np.testing.assert_array_equal(
            result1.rms_noise,
            result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be bit-perfect identical",
        )

        # Noise masks must be identical
        np.testing.assert_array_equal(
            result1.noise_mask,
            result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical",
        )

        # Verify basic properties
        assert (
            result1.rms_noise.shape == result2.rms_noise.shape
        ), f"{context}: RMS shape mismatch"
        assert (
            result1.noise_mask.dtype == result2.noise_mask.dtype == bool
        ), f"{context}: Mask should be boolean"
        assert np.all(
            result1.rms_noise > 0
        ), f"{context}: RMS values should be positive"
        assert np.all(
            result2.rms_noise > 0
        ), f"{context}: RMS values should be positive"

    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds.

        Timeout is 120s rather than 30s because ``calibrate-tau`` now
        auto-runs the 3-way shape recommendation (Stage 2b
        ``auto_recommend=True`` by default), which adds ~50s on the
        2638 fixture beyond the τ calibration itself.
        """
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
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
        trim_min, trim_max = standard_ft_params["trim"]
        self._run_cli_command(
            [
                "ft",
                "run",
                str(test_file),
                "--trim",
                f"{trim_min}:{trim_max}",
            ]
        )

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
        trim_min, trim_max = standard_ft_params["trim"]
        self._run_cli_command(
            [
                "ft",
                "run",
                str(test_file),
                "--trim",
                f"{trim_min}:{trim_max}",
            ]
        )
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
        trim_min, trim_max = standard_ft_params["trim"]
        self._run_cli_command(
            [
                "ft",
                "run",
                str(test_file),
                "--trim",
                f"{trim_min}:{trim_max}",
            ]
        )
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

        # Re-estimate with the scatter estimator (the canonical default) across
        # all three interfaces and confirm the portable file round-trips.
        noise_result_pipeline = pipe.estimate_noise()

        # Process with functional API
        noise_result_functional = ftmw.estimate_noise(test_file)

        # Process with CLI (just verify CLI can process the file)
        self._run_cli_command(["noise", "run", str(test_file)])
        noise_result_cli = ftmw.estimate_noise(test_file)

        # Results should be identical across interfaces
        self._verify_noise_portability(
            noise_result_pipeline,
            noise_result_functional,
            "Pipeline to Functional portability",
        )
        self._verify_noise_portability(
            noise_result_functional, noise_result_cli, "Functional to CLI portability"
        )
        self._verify_noise_portability(
            noise_result_pipeline, noise_result_cli, "Pipeline to CLI portability"
        )

    def _verify_file_portability(self, ft1: ComplexFT, ft2: ComplexFT):
        """Verify two ComplexFT objects are identical for portability testing."""
        # Strict comparison for portability
        np.testing.assert_array_equal(
            ft1.freq_array,
            ft2.freq_array,
            err_msg="Frequency arrays should be identical across interfaces",
        )

        np.testing.assert_allclose(
            ft1.complex_spectrum,
            ft2.complex_spectrum,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Complex spectra should be identical across interfaces",
        )

    def _verify_noise_portability(self, result1, result2, context: str):
        """Verify two NoiseResult objects are identical for portability testing."""
        assert isinstance(
            result1, NoiseResult
        ), f"{context}: First result should be NoiseResult"
        assert isinstance(
            result2, NoiseResult
        ), f"{context}: Second result should be NoiseResult"

        # Strict comparison for portability
        np.testing.assert_array_equal(
            result1.rms_noise,
            result2.rms_noise,
            err_msg=f"{context}: RMS noise arrays should be identical",
        )

        np.testing.assert_array_equal(
            result1.noise_mask,
            result2.noise_mask,
            err_msg=f"{context}: Noise masks should be identical",
        )

    def _run_cli_command(self, args):
        """Run CLI command and ensure it succeeds.

        Timeout is 120s rather than 30s because ``calibrate-tau`` now
        auto-runs the 3-way shape recommendation (Stage 2b
        ``auto_recommend=True`` by default), which adds ~50s on the
        2638 fixture beyond the τ calibration itself.
        """
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
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
            ["ftmwpipeline", "ft", "run", str(nonexistent_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
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
            [
                "ftmwpipeline",
                "data",
                "import",
                str(test_file),
                nonexistent_source,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode != 0, "CLI should fail with invalid source"

    def test_invalid_parameter_errors(self, exp_2638_data_path, temp_ftmw_dir):
        """Test all interfaces handle invalid parameters consistently."""
        test_file = temp_ftmw_dir / "invalid_params.ftmw"

        # Create valid file first
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        ftmw.import_data(test_file, source=exp_2638_data_path)

        # Test invalid start_us parameter (negative value)
        with pytest.raises((ValueError, RuntimeError)):
            pipe.compute_ft(start_us=-1.0)

        with pytest.raises((ValueError, RuntimeError)):
            ftmw.compute_ft(test_file, start_us=-1.0)

        # CLI should also fail (inverted trim range is rejected by the parser).
        result = subprocess.run(
            ["ftmwpipeline", "ft", "run", str(test_file), "--trim", "40000:26500"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode != 0, "CLI should fail with invalid trim range"

    def test_corruption_detection_consistency(self, exp_2638_data_path, temp_ftmw_dir):
        """Test all interfaces detect file corruption consistently."""
        test_file = temp_ftmw_dir / "corruption_test.ftmw"

        # Create valid file
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)

        # Corrupt the file by truncating it
        with open(test_file, "r+b") as f:
            f.truncate(100)  # Truncate to 100 bytes

        # All interfaces should detect corruption
        validation_pipeline = pipe.validate()
        assert not validation_pipeline["valid"], "Pipeline should detect corruption"

        validation_functional = ftmw.validate_pipeline(test_file)
        assert not validation_functional[
            "valid"
        ], "Functional API should detect corruption"

        # CLI should also detect corruption (though exact command may vary)
        result = subprocess.run(
            ["ftmwpipeline", "data", "show", str(test_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode != 0, "CLI should fail with corrupted file"

    def test_noise_stage_dependency_errors(self, exp_2638_data_path, temp_ftmw_dir):
        """Test consistent error handling for missing Stage 1 dependency."""
        test_file = temp_ftmw_dir / "incomplete_pipeline.ftmw"

        # Create file with only Stage 0 (no Stage 1)
        pipe = Pipeline.create(test_file, source=exp_2638_data_path)
        # Don't call compute_ft() - missing Stage 1

        # All interfaces should raise appropriate errors for missing Stage 1
        with pytest.raises(
            (ValueError, RuntimeError), match="Stage 1.*must be completed"
        ):
            pipe.estimate_noise()

        with pytest.raises(
            (ValueError, RuntimeError), match="Stage 1.*must be completed"
        ):
            ftmw.estimate_noise(test_file)

        # CLI should fail with non-zero return code
        result = subprocess.run(
            ["ftmwpipeline", "noise", "run", str(test_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode != 0, "CLI should fail when Stage 1 missing"
        assert (
            "Stage 1" in result.stderr or "Stage 1" in result.stdout
        ), "Error should mention Stage 1"


@pytest.mark.cross_interface
class TestSettingsShowConsistency:
    """The resolved-settings view (issue #28) must be identical across the
    Pipeline / functional-API surfaces, and the CLI table must reflect the same
    rows. The headline ``start_us`` row must read correctly end to end."""

    def test_pipeline_api_rows_identical(self, baseline_2638_stage2):
        api_rows = ftmw.settings_show(baseline_2638_stage2, include_advanced=True)
        pipe_rows = Pipeline.open(baseline_2638_stage2).settings_show(
            include_advanced=True
        )
        assert api_rows == pipe_rows
        assert len(api_rows) > 0

    def test_pipeline_api_rows_identical_with_preset(
        self, baseline_2638_stage2, tmp_path
    ):
        preset = tmp_path / "inst.yaml"
        preset.write_text("name: inst\nstage2:\n  window_mhz: 137.0\n")
        api_rows = ftmw.settings_show(
            baseline_2638_stage2, include_advanced=True, preset=preset
        )
        pipe_rows = Pipeline.open(baseline_2638_stage2).settings_show(
            include_advanced=True, preset=preset
        )
        assert api_rows == pipe_rows

    def test_headline_start_us_resolves_to_persisted(self, baseline_2638_stage2):
        # A file taken through compute_ft has start_us persisted in ft_processing.
        rows = {r.path: r for r in ftmw.settings_show(baseline_2638_stage2)}
        start = rows["stage1.start_us"]
        assert start.source == ".ftmw"
        assert start.value is not None

    def test_cli_table_matches_rows(self, baseline_2638_stage2):
        from ftmwpipeline.cli.settings_commands import _fmt_value

        by_path = {
            r.path: r
            for r in ftmw.settings_show(baseline_2638_stage2, include_advanced=True)
        }
        stdout, _ = TestSettingsShowConsistency._run_cli(
            ["settings", "show", str(baseline_2638_stage2), "--all"]
        )
        # stage2.window_mhz is the first stage2 row, so its full path prints.
        win = by_path["stage2.window_mhz"]
        line = next(ln for ln in stdout.splitlines() if "stage2.window_mhz" in ln)
        assert win.source in line
        assert _fmt_value(win.value) in line
        # The headline start_us line carries its source too.
        start_line = next(ln for ln in stdout.splitlines() if "stage1.start_us" in ln)
        assert ".ftmw" in start_line

    def test_cli_selector_scopes_output(self, baseline_2638_stage2):
        stdout, _ = TestSettingsShowConsistency._run_cli(
            ["settings", "show", str(baseline_2638_stage2), "stage2b.gaussian", "--all"]
        )
        assert "stage2b.gaussian.snr_min" in stdout
        assert "stage5" not in stdout

    @staticmethod
    def _run_cli(args):
        result = subprocess.run(
            ["ftmwpipeline"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            pytest.fail(
                f"CLI command failed:\nCommand: ftmwpipeline {' '.join(args)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        return result.stdout, result.stderr


@pytest.mark.cross_interface
class TestSettingsMutationConsistency:
    """`settings set` / `settings export` (issue #28) must behave identically
    across the Pipeline / functional-API surfaces, and the CLI must drive the
    same core."""

    def test_set_parity(self, baseline_2638_stage2, tmp_path):
        a = tmp_path / "a.ftmw"
        b = tmp_path / "b.ftmw"
        shutil.copyfile(baseline_2638_stage2, a)
        shutil.copyfile(baseline_2638_stage2, b)

        ra = ftmw.settings_set(a, "stage2.window_mhz", "111")
        rb = Pipeline.open(b).settings_set("stage2.window_mhz", "111")
        assert ra.value == rb.value == 111.0
        assert ra.invalidated == rb.invalidated
        assert "stage2_noise_result" in ra.invalidated

        rows_a = {r.path: r for r in ftmw.settings_show(a, include_advanced=True)}
        rows_b = {r.path: r for r in ftmw.settings_show(b, include_advanced=True)}
        assert rows_a["stage2.window_mhz"].value == rows_b["stage2.window_mhz"].value
        assert rows_a["stage2.window_mhz"].source == ".ftmw"

    def test_export_parity(self, baseline_2638_stage2, tmp_path):
        out_a = tmp_path / "a.yml"
        out_b = tmp_path / "b.yml"
        ra = ftmw.settings_export(baseline_2638_stage2, out_a, name="inst")
        rb = Pipeline.open(baseline_2638_stage2).settings_export(out_b, name="inst")
        assert ra.paths == rb.paths
        assert out_a.read_text() == out_b.read_text()
        # The exported preset is loadable and reproduces a chosen value.
        from ftmwpipeline.core import noise_settings as ns

        loaded = ns.load_preset(out_a)
        rows = {r.path: r for r in ftmw.settings_show(baseline_2638_stage2)}
        assert loaded.window_mhz == rows["stage2.window_mhz"].value

    def test_cli_set_then_show_reflects(self, baseline_2638_stage2, tmp_path):
        work = tmp_path / "w.ftmw"
        shutil.copyfile(baseline_2638_stage2, work)
        TestSettingsShowConsistency._run_cli(
            ["settings", "set", str(work), "stage2.smoothing_mhz", "650"]
        )
        row = {r.path: r for r in ftmw.settings_show(work)}["stage2.smoothing_mhz"]
        assert row.source == ".ftmw"
        assert row.value == 650.0


# ---------------------------------------------------------------------------
# Stage 6 review / candidate-ledger cross-interface consistency
# ---------------------------------------------------------------------------


@pytest.mark.cross_interface
class TestCandidateLedgerConsistency:
    """Verify that Pipeline.candidate_ledger == api.get_candidate_ledger == CLI review show.

    Uses the small 3-window Stage 5 baseline so the test runs quickly.
    Cross-interface correctness: all three surfaces must return the same list.
    """

    def test_pipeline_vs_api_identical(self, baseline_2638_stage5_small):
        """Pipeline.candidate_ledger and api.get_candidate_ledger return the same list."""
        fp = baseline_2638_stage5_small

        # Pipeline interface
        pipe = Pipeline.open(fp)
        cands_pipe = pipe.candidate_ledger()

        # Functional API
        cands_api = ftmw.get_candidate_ledger(fp)

        assert len(cands_pipe) == len(
            cands_api
        ), f"Candidate count mismatch: Pipeline={len(cands_pipe)}, API={len(cands_api)}"
        for i, (cp, ca) in enumerate(zip(cands_pipe, cands_api)):
            assert cp.frequency_mhz == pytest.approx(
                ca.frequency_mhz, abs=1e-9
            ), f"Candidate {i}: freq differs (Pipeline={cp.frequency_mhz}, API={ca.frequency_mhz})"
            assert cp.window_id == ca.window_id, f"Candidate {i}: window_id differs"
            assert (
                cp.evidence_kind == ca.evidence_kind
            ), f"Candidate {i}: evidence_kind differs"
            assert cp.best_evidence == pytest.approx(
                ca.best_evidence, rel=1e-6
            ), f"Candidate {i}: best_evidence differs"

    def test_pipeline_vs_api_window_filter(self, baseline_2638_stage5_small):
        """window_id filter returns a consistent subset."""
        fp = baseline_2638_stage5_small

        # Get all windows to find one
        pipe = Pipeline.open(fp)
        all_fits = pipe.load_fit().window_fits
        if not all_fits:
            pytest.skip("No windows in Stage 5 fit")

        wid = all_fits[0].window_id

        cands_pipe = pipe.candidate_ledger(window_id=wid)
        cands_api = ftmw.get_candidate_ledger(fp, window_id=wid)

        assert len(cands_pipe) == len(cands_api)
        assert all(c.window_id == wid for c in cands_pipe)
        assert all(c.window_id == wid for c in cands_api)

    def test_cli_review_show_succeeds(self, baseline_2638_stage5_small):
        """CLI 'review show <file>' exits 0 and produces output."""
        fp = str(baseline_2638_stage5_small)
        result = subprocess.run(
            ["ftmwpipeline", "review", "show", fp],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert (
            result.returncode == 0
        ), f"CLI review show failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        # Should contain a header line
        assert "win" in result.stdout.lower() or "window" in result.stdout.lower()

    def test_cli_review_show_candidates_succeeds(self, baseline_2638_stage5_small):
        """CLI 'review show --candidates <file>' exits 0."""
        fp = str(baseline_2638_stage5_small)
        result = subprocess.run(
            ["ftmwpipeline", "review", "show", fp, "--candidates"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert (
            result.returncode == 0
        ), f"CLI review show --candidates failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_cli_candidate_count_matches_api(self, baseline_2638_stage5_small):
        """CLI candidate count (from stdout) matches API total."""
        fp = str(baseline_2638_stage5_small)
        # API total
        cands_api = ftmw.get_candidate_ledger(fp)

        result = subprocess.run(
            ["ftmwpipeline", "review", "show", fp, "--candidates"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0

        # The header line prints e.g. "Candidate ledger (all windows, bar=3.0): N candidate(s)"
        header_line = result.stdout.splitlines()[0] if result.stdout else ""
        # Extract the number from the first line
        import re

        m = re.search(r"(\d+) candidate", header_line)
        if m:
            cli_count = int(m.group(1))
            assert cli_count == len(
                cands_api
            ), f"CLI reports {cli_count} candidates but API returns {len(cands_api)}"
