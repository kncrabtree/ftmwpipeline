"""Object-oriented Pipeline class for FTMW spectroscopy data processing.

This module provides the high-level object-oriented interface for the FTMW
pipeline, implementing the dual-interface architecture alongside functional
and CLI interfaces. Each Pipeline instance is bound to a specific .ftmw file.
"""

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from ._internal.tuning import (
        BatchItem,
        ExportResult,
        KnobSpec,
        SettingRow,
        SetResult,
        SweepResult,
    )

from ._internal.report_html_impl import report_run_impl
from ._internal.report_impl import report_table_impl
from ._internal.run_impl import run_pipeline_impl
from ._internal.shape_recommendation_impl import recommend_shape_impl
from ._internal.stage0_impl import import_data_impl, load_fid_from_pipeline_impl
from ._internal.stage1_impl import (
    compute_ft_impl,
    save_ft_parameters_impl,
    visualize_ft_impl,
)
from ._internal.stage2_impl import compute_noise_estimation_impl, visualize_noise_impl
from ._internal.stage2b_g_impl import (
    calibrate_tau_G_impl,
    load_tau_G_calibration_impl,
)
from ._internal.stage2b_impl import (
    calibrate_tau_impl,
    load_tau_calibration_impl,
)
from ._internal.stage3_impl import (
    detect_peaks_impl,
    load_peaks_impl,
    visualize_peaks_impl,
)
from ._internal.stage4_impl import (
    assign_windows_impl,
    load_windows_impl,
    visualize_windows_impl,
)
from ._internal.stage5_impl import (
    fit_peaks_impl,
    fit_show_impl,
    load_fit_impl,
    visualize_fit_impl,
)
from ._internal.stage5_validation_impl import validate_stage5_shape_error_impl
from ._internal.stage6_impl import (
    DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    DEFAULT_DISPLAY_BAR,
    RankedWindow,
    RefitWindowResult,
    ReviewRunResult,
    get_candidate_ledger_impl,
    get_final_products_impl,
    get_review_status_impl,
    merge_peaks_impl,
    rank_windows_impl,
    refit_window_impl,
    review_accept_impl,
    review_run_impl,
    set_sigma_floor_impl,
    split_peak_impl,
)
from ._internal.start_detection_impl import detect_start_time_impl
from ._internal.timebase_impl import (
    calibrate_timebase_impl,
    load_timebase_calibration_impl,
)
from .core.data_structures import (
    FID,
    ComplexFT,
    FinalProducts,
    LedgerCandidate,
    Peak,
    SpectrumFit,
    Stage6Review,
    WindowPlan,
)
from .core.noise_settings import NoiseSettings
from .core.peak_detection_settings import PeakDetectionSettings
from .core.settings import FTSettings
from .core.stage_fit_settings import StageFitSettings, coerce_clock_sources
from .core.start_detection_settings import StartDetectionSettings
from .core.tau_calibration_settings import TauCalibrationSettings
from .core.window_planning_settings import WindowPlanningSettings
from .file_manager import (
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
from .fitting.tau_calibration import ShapeRecommendation, TauCalibrationResult
from .fitting.timebase_calibration import TimebaseCalibrationResult
from .io.data_loaders import detect_format, load_fid, validate_source
from .io.stage_fit_settings_serialization import write_recommended_clock_sources
from .preprocessing.noise_estimation import NoiseResult
from .preprocessing.start_detection import StartDetectionResult


class Pipeline:
    """
    Object-oriented pipeline interface for FTMW spectroscopy data processing.

    Each Pipeline instance is bound to a specific .ftmw pipeline file and provides
    high-level methods for data processing and analysis. This implements the
    object-oriented interface of the dual-interface architecture.

    Key Features:
    - File-bound design: each instance manages one .ftmw pipeline file
    - Safe re-execution: methods can be called multiple times
    - Consistent with CLI: methods behave identically to CLI commands
    - Error handling: clear error messages using custom exceptions

    Example Usage:
    ```python
    # Create new pipeline from raw data
    pipe = Pipeline.create("exp_2638.ftmw", source='examples/blackchirp_data/2638/')

    # Open existing pipeline for analysis
    pipe = Pipeline.open("exp_2638.ftmw")

    # Stage 1: FT Processing (canonical FT is unapodized, native-length)
    pipe.compute_ft(trim=(26500, 40000))
    pipe.visualize_ft(save_params=True)

    # File info and validation
    pipe.info()       # Show pipeline status and metadata
    pipe.validate()   # Check file integrity
    ```
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        source_metadata: Optional[SourceMetadata] = None,
        stage_tracker: Optional[PipelineStageTracker] = None,
    ):
        """
        Bind a Pipeline instance to a specific .ftmw file.

        Smart constructor: ``Pipeline(path)`` opens an existing pipeline file,
        raising ``FileNotFoundError`` with guidance if it does not exist. Use
        ``Pipeline.create()`` to make a new analysis from raw data.

        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file.
        source_metadata : SourceMetadata, optional
            Source provenance. When omitted (the smart-constructor path) it is
            loaded from the file.
        stage_tracker : PipelineStageTracker, optional
            Stage completion tracker. When omitted it is loaded from the file.

        Notes
        -----
        ``create()`` and ``open()`` supply both metadata arguments directly.
        Passing only a path performs the same load as ``open()``.
        """
        self.logger = logging.getLogger(__name__)

        if source_metadata is None or stage_tracker is None:
            # Smart-constructor path: load state from the file (raises
            # FileNotFoundError with guidance if absent).
            filepath, source_metadata, stage_tracker = open_pipeline_file(filepath)

        # File binding
        self.filepath = Path(filepath)
        self.source_metadata = source_metadata
        self.stage_tracker = stage_tracker

    @classmethod
    def create(
        cls,
        filepath: Union[str, Path],
        source: Union[str, Path],
        format_name: Optional[str] = None,
        fid_index: Optional[int] = None,
        force: bool = False,
        **loader_params: Any,
    ) -> "Pipeline":
        """
        Create new pipeline from raw experimental data.

        Parameters
        ----------
        filepath : str or Path
            Path for new pipeline file (should have .ftmw extension)
        source : str or Path
            Path to source data (file or directory)
        format_name : str, optional
            Data format name. If None, auto-detect format.
        fid_index : int, optional
            FID index for multi-FID formats (e.g., BlackChirp)
        force : bool, default False
            If True, overwrite existing file even with different source
        **loader_params
            Additional parameters for data loader

        Returns
        -------
        Pipeline
            New Pipeline instance bound to the created file

        Raises
        ------
        PipelineExistsError
            If file exists with different source and force=False
        FileNotFoundError
            If source data does not exist
        ValueError
            If format detection or validation fails
        RuntimeError
            If data loading or file creation fails
        """
        source_path = Path(source)

        # Validate source exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        # Format detection if not specified
        if format_name is None:
            format_name = detect_format(source_path)
            if format_name is None:
                raise ValueError(f"Could not detect format for: {source_path}")

        # Validate source with format
        validation = validate_source(source_path, format_name)
        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            raise ValueError(f"Source validation failed: {errors}")

        # Prepare loader parameters
        if format_name == "blackchirp" and fid_index is not None:
            loader_params["fid_index"] = fid_index

        # Load FID data
        try:
            fid = load_fid(source_path, format_name, **loader_params)
        except Exception as e:
            raise RuntimeError(f"Failed to load FID data: {e}") from e

        # Create source metadata
        source_metadata = SourceMetadata(
            source_path=source_path,
            format_name=format_name,
            loader_parameters=loader_params,
        )

        # Create pipeline file
        created_filepath = create_pipeline_file(
            filepath, fid, source_metadata, force=force
        )

        # Persist any instrument clock declaration extracted by the loader so
        # the Stage 5 resolver can surface it at the recommended layer.
        raw_clocks = fid.metadata.get("clock_sources")
        if raw_clocks is not None:
            try:
                clock_tuple = coerce_clock_sources(raw_clocks)
                write_recommended_clock_sources(str(created_filepath), clock_tuple)
            except Exception:  # pragma: no cover
                pass  # non-fatal: clock metadata is advisory

        # Chirp-window persistence + import-time start recommendation live
        # once in the stage 0 impl.
        from ._internal.stage0_impl import persist_chirp_window_metadata

        persist_chirp_window_metadata(str(created_filepath), fid)

        # Load file info for Pipeline instance
        filepath, source_metadata, stage_tracker = open_pipeline_file(created_filepath)

        return cls(filepath, source_metadata, stage_tracker)

    @classmethod
    def open(cls, filepath: Union[str, Path]) -> "Pipeline":
        """
        Open existing pipeline file.

        Parameters
        ----------
        filepath : str or Path
            Path to existing pipeline file

        Returns
        -------
        Pipeline
            Pipeline instance bound to the opened file

        Raises
        ------
        FileNotFoundError
            If pipeline file doesn't exist
        PipelineCorruptedError
            If file appears to be corrupted
        ValueError
            If file format is invalid
        """
        filepath, source_metadata, stage_tracker = open_pipeline_file(filepath)

        return cls(filepath, source_metadata, stage_tracker)

    @classmethod
    def build(
        cls,
        source: Union[str, Path],
        *,
        trim: Optional[Tuple[float, float]] = None,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Drive *source* through every stage end-to-end and return the result.

        Convenience classmethod over :func:`run_pipeline_impl`: imports the raw
        *source*, then runs FT -> noise -> tau -> peaks -> windows -> fit ->
        timebase -> review (and, with ``report=True``, the report), with live
        per-stage progress. *trim* (the active-band FT range, MHz) is required.
        ``output`` is the destination ``.ftmw`` (derived from *source* if
        omitted). Remaining keyword arguments are forwarded to
        :func:`run_pipeline_impl` (per-stage ``*_params`` override dicts,
        ``detect_start`` / ``calibrate`` / ``clocks``, ``report`` /
        ``report_output_dir``, ``sigma_floor_khz``, ``force``, ``progress``, …).

        Returns the structured run result (``pipeline_file``, ``status``,
        ``completed_stages``, ``failed_stage``, ``error``, ``timebase``,
        ``report``, ``elapsed_s``). Open the finished file with
        :meth:`Pipeline.open` (``result["pipeline_file"]``).
        """
        return run_pipeline_impl(source, output, trim=trim, **kwargs)

    def load_data(self) -> FID:
        """
        Load FID data from the pipeline file.

        This method implements Stage 0 data loading, providing access to the
        raw FID data stored in the pipeline file.

        Returns
        -------
        FID
            The loaded FID object with all metadata

        Raises
        ------
        FileNotFoundError
            If pipeline file does not exist
        PipelineCorruptionError
            If file is corrupted or invalid
        RuntimeError
            If FID loading fails
        """
        try:
            fid = load_fid_from_pipeline_impl(str(self.filepath))
            self.logger.info(
                f"Loaded FID data: {fid.n_points:,} points, {fid.duration_us:.1f} μs"
            )
            return fid
        except Exception as e:
            raise RuntimeError(f"Failed to load FID data: {e}") from e

    def compute_ft(
        self,
        trim: Optional[Tuple[float, float]] = None,
        start_us: Optional[float] = None,
        end_us: Optional[float] = None,
        units_power: Optional[int] = None,
        from_saved_params: bool = False,
    ) -> ComplexFT:
        """Compute Fourier Transform (Stage 1, user-driven).

        Resolves settings through ``explicit > persisted > recommended`` and
        persists the resolved canonical settings to the ``.ftmw`` file.  Can be
        called multiple times safely. The canonical FT is unconditionally
        unapodized, un-windowed, and native-length.

        Parameters
        ----------
        trim : tuple of float, optional
            ``(min_mhz, max_mhz)`` frequency analysis range to keep.
        start_us : float, optional
            FID window start time in microseconds.
        end_us : float, optional
            FID window end time in microseconds.
        units_power : int, optional
            Spectrum scaling as power of 10.
        from_saved_params : bool, default False
            If ``True``, ignore the explicit kwargs above and use only the
            persisted / recommended settings (no explicit overrides).

        Returns
        -------
        ComplexFT
            Computed frequency-domain data.

        Raises
        ------
        StageDependencyError
            If Stage 0 (FID data) is not available.
        RuntimeError
            If FT computation fails.
        """
        try:
            settings: Optional[FTSettings] = None
            if not from_saved_params:
                settings = FTSettings(
                    start_us=start_us,
                    end_us=end_us,
                    units_power=units_power,
                    trim=trim,
                )
            result = compute_ft_impl(
                file_path=str(self.filepath),
                settings=settings,
                validate_only=False,
                persist=True,
            )
            complex_ft: ComplexFT = result["complex_ft"]
            self.logger.info(f"FT computed: {complex_ft.n_points:,} frequency points")
            if trim:
                self.logger.info(f"Trimmed to {trim[0]:.1f}-{trim[1]:.1f} MHz")
            return complex_ft
        except Exception as e:
            raise RuntimeError(f"Failed to compute FT: {e}") from e

    def visualize_ft(
        self,
        trim: Optional[Tuple[float, float]] = None,
        start_us: Optional[float] = None,
        end_us: Optional[float] = None,
        units_power: Optional[int] = None,
        save_params: bool = False,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_fid_panels: bool = True,
    ) -> Any:
        """Create enhanced FT visualization with processing workflow display.

        Equivalent to the CLI ``ft show`` command.  Never persists
        settings; exploration only.  Pass ``save_params=True`` to write the
        explicitly provided kwargs to the canonical ``ft_processing`` record.
        The canonical FT is unconditionally unapodized, un-windowed, and
        native-length.

        Parameters
        ----------
        trim : tuple of float, optional
            ``(min_mhz, max_mhz)`` frequency analysis range.
        start_us : float, optional
            FID window start time in microseconds.
        end_us : float, optional
            FID window end time in microseconds.
        units_power : int, optional
            Spectrum scaling as power of 10.
        save_params : bool, default False
            If ``True``, persist the explicitly provided settings.
        backend : str, default ``'matplotlib'``
            Plotting backend (``'matplotlib'`` or ``'plotly'``).
        interactive : bool, default True
            Whether to show an interactive plot.
        output_file : str or Path, optional
            Path to save the plot image (non-interactive mode).
        show_fid_panels : bool, default True
            Whether to include FID processing panels.

        Returns
        -------
        figure
            Matplotlib or Plotly figure object.

        Raises
        ------
        StageDependencyError
            If required dependencies are not available.
        RuntimeError
            If visualization fails.
        """
        try:
            settings = FTSettings(
                start_us=start_us,
                end_us=end_us,
                units_power=units_power,
                trim=trim,
            )
            fig = visualize_ft_impl(
                file_path=str(self.filepath),
                settings=settings,
                title=None,
                show_fid_panels=show_fid_panels,
                backend=backend,
                interactive=interactive,
            )

            # Handle output
            if not interactive and output_file:
                fig.savefig(output_file, dpi=150, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")
            elif not interactive:
                default_name = f"{self.filepath.stem}_enhanced_spectrum.png"
                fig.savefig(default_name, dpi=150, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {default_name}")
            elif interactive and backend == "matplotlib":
                import matplotlib.pyplot as plt

                plt.show()

            # Save parameters if requested
            if save_params:
                params: Dict[str, Any] = {}
                for key, value in (
                    ("start_us", start_us),
                    ("end_us", end_us),
                    ("units_power", units_power),
                ):
                    if value is not None:
                        params[key] = value
                if trim is not None:
                    params["trim_min_mhz"] = trim[0]
                    params["trim_max_mhz"] = trim[1]
                if params:
                    save_ft_parameters_impl(str(self.filepath), params)
                    self.logger.info(f"Saved {len(params)} processing parameters")
                else:
                    self.logger.info("No custom parameters to save")

            self.logger.info("FT visualization completed")
            return fig

        except Exception as e:
            raise RuntimeError(f"Failed to create FT visualization: {e}") from e

    def estimate_noise(
        self,
        *,
        window_mhz: Optional[float] = None,
        pedestal_mhz: Optional[float] = None,
        line_k: Optional[float] = None,
        n_iter: Optional[int] = None,
        region_aware: Optional[bool] = None,
        smoothing_mhz: Optional[float] = None,
        smoothing_percentile: Optional[float] = None,
        convolve_mhz: Optional[float] = None,
        settings: Optional[NoiseSettings] = None,
        preset: Optional[str] = None,
    ) -> NoiseResult:
        """
        Estimate frequency-dependent noise with the scatter estimator.

        This method implements Stage 2 noise estimation, equivalent to the CLI
        ``noise run`` command. Requires Stage 1 (FT computation) to be completed.

        The scatter estimator is high-pass and region-aware: it is immune to the
        leakage pedestal on high-SNR, line-dense spectra.

        Parameters
        ----------
        window_mhz, pedestal_mhz, line_k, n_iter, region_aware, smoothing_mhz,
        smoothing_percentile, convolve_mhz
            Scatter-estimator knobs; each defaults to the kernel's hard default
            when left unset. ``smoothing_mhz`` / ``smoothing_percentile`` control
            the broad lower-envelope median σ smoothing that rides the noise
            floor through line-dense bands (``smoothing_mhz=0`` disables it);
            ``convolve_mhz`` is the Gaussian σ of the second pass that removes
            the median's staircase.
        settings, preset
            Alternative ways to populate the preset layer of the settings chain.

        Returns
        -------
        NoiseResult
            Container with RMS noise estimate, noise mask, and diagnostics

        Raises
        ------
        StageDependencyError
            If Stage 1 (FT computation) has not been completed
        ValueError
            If noise estimation parameters are invalid
        RuntimeError
            If noise estimation computation fails
        """
        try:
            # Compute noise estimation using shared implementation (handles dependency checking and storage)
            result = compute_noise_estimation_impl(
                file_path=str(self.filepath),
                window_mhz=window_mhz,
                pedestal_mhz=pedestal_mhz,
                line_k=line_k,
                n_iter=n_iter,
                region_aware=region_aware,
                smoothing_mhz=smoothing_mhz,
                smoothing_percentile=smoothing_percentile,
                convolve_mhz=convolve_mhz,
                settings=settings,
                preset=preset,
            )

            # Storage and stage tracking handled by shared implementation
            self.logger.info("Stage 2: Noise estimation completed successfully")
            return cast(NoiseResult, result["noise_result"])

        except StageDependencyError:
            # Re-raise dependency errors with clear message
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to estimate noise: {e}") from e

    def visualize_noise(
        self,
        y_max_factor: Optional[float] = None,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        show_bin_boundaries: Optional[bool] = None,
        show_noise_points: Optional[bool] = None,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        **plot_kwargs: Any,
    ) -> Any:
        """
        Create noise estimation diagnostic visualization.

        This method creates diagnostic plots showing spectrum, noise points,
        bin boundaries, and RMS noise estimates. Equivalent to the CLI
        ``noise show`` command.

        Parameters
        ----------
        y_max_factor : float, optional
            Y-axis maximum as multiple of median RMS noise (default: 20.0)
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (16, 6))
        title : str, optional
            Custom title for the plot
        show_bin_boundaries : bool, optional
            Whether to show bin boundaries (default: True)
        show_noise_points : bool, optional
            Whether to highlight noise points (default: True)
        backend : str, default 'matplotlib'
            Plotting backend ('matplotlib' or 'plotly')
        interactive : bool, default True
            Whether to create interactive plots
        output_file : str or Path, optional
            If provided, save plot to this file
        **plot_kwargs
            Additional plotting parameters

        Returns
        -------
        matplotlib.Figure or plotly.Figure
            The created figure object

        Raises
        ------
        StageDependencyError
            If Stage 2 (noise estimation) has not been completed
        RuntimeError
            If visualization fails
        """
        try:
            # Create visualization using shared implementation (handles dependency checking)
            fig = visualize_noise_impl(
                file_path=str(self.filepath),
                y_max_factor=y_max_factor,
                figsize=figsize,
                title=title,
                show_bin_boundaries=show_bin_boundaries,
                show_noise_points=show_noise_points,
                backend=backend,
                interactive=interactive,
                **plot_kwargs,
            )

            # Save output file if requested
            if output_file:
                try:
                    if backend == "plotly":
                        fig.write_html(str(output_file))
                    else:
                        fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
                    self.logger.info(f"Visualization saved to: {output_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save visualization: {e}")

            self.logger.info("Noise visualization completed")
            return fig

        except StageDependencyError:
            # Re-raise dependency errors with clear message
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to create noise visualization: {e}") from e

    def calibrate_tau(
        self,
        n_seg: Optional[int] = None,
        t_sigma: Optional[float] = None,
        tau_max_us: Optional[float] = None,
        rss_gate_factor: Optional[float] = None,
        sigma_time: Optional[float] = None,
        min_contributors: Optional[int] = None,
        sigma_tau_fraction_max: Optional[float] = None,
        bimodality_dominant_fraction: Optional[float] = None,
        compute_band_majorities: Optional[bool] = None,
        min_contributors_per_band: Optional[int] = None,
        settings: Optional[TauCalibrationSettings] = None,
        preset: Optional[str] = None,
    ) -> TauCalibrationResult:
        """Run the Stage 2b data-driven tau calibration.

        Requires Stages 0-2 completed. Extracts a global majority-vote
        molecular decay constant ``tau_maj`` (with robust spread
        ``sigma_tau``) from the raw FID by sliding a
        ``T_w = T_full / n_seg``-long active sub-window across the
        zero-padded record and fitting a per-bin exponential to the
        magnitude vs frame-start time. Persists the result to
        ``/stage2b_tau_calibration`` and invalidates downstream stages.

        Parameters left as ``None`` fall through the resolution chain
        (``explicit > persisted > preset > recommended > hard default``);
        pass ``settings=`` to drive the calibration from a Python
        :class:`TauCalibrationSettings`, or ``preset=NAME_OR_PATH`` to
        load from packaged YAML. They are mutually exclusive. The
        resolved settings are stamped to
        ``processing_parameters/stage2b_tau`` so a follow-up no-kwargs
        call on the same file inherits them.
        """
        try:
            result = calibrate_tau_impl(
                file_path=str(self.filepath),
                n_seg=n_seg,
                t_sigma=t_sigma,
                tau_max_us=tau_max_us,
                rss_gate_factor=rss_gate_factor,
                sigma_time=sigma_time,
                min_contributors=min_contributors,
                sigma_tau_fraction_max=sigma_tau_fraction_max,
                bimodality_dominant_fraction=bimodality_dominant_fraction,
                compute_band_majorities=compute_band_majorities,
                min_contributors_per_band=min_contributors_per_band,
                settings=settings,
                preset=preset,
            )
            tc = result["tau_calibration"]
            self.logger.info(
                "Stage 2b: tau_maj=%.3f us, sigma_tau=%.3f us, "
                "n_contributors=%d, preconditions=%s",
                tc.tau_maj_us,
                tc.sigma_tau_us,
                tc.n_contributors,
                "pass" if tc.preconditions_passed else "fail",
            )
            return cast(TauCalibrationResult, tc)
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to calibrate tau: {e}") from e

    def load_tau_calibration(self) -> TauCalibrationResult:
        """Load the persisted Stage 2b :class:`TauCalibrationResult`."""
        return cast(
            TauCalibrationResult,
            load_tau_calibration_impl(str(self.filepath))["tau_calibration"],
        )

    def calibrate_timebase(
        self,
        *,
        clocks: Optional[Any] = None,
        kappa_sys: Optional[float] = None,
        snr_min: Optional[float] = None,
    ) -> TimebaseCalibrationResult:
        """Measure the scope-timebase scale error ``eps`` from Rb-locked tones.

        Requires Stage 0 (the raw FID); the canonical Stage 1 active-region
        bounds are read opportunistically. Resolves the instrument clock
        declaration from the explicit ``clocks`` argument, else the persisted
        Stage 5 ``spur.clocks``; raises ``ValueError`` if neither yields a
        non-empty declaration with at least one locked source. Persists the
        measured ``eps`` to ``/timebase_calibration``. Measures ``eps`` only;
        applying it to the frequency axis is out of scope.
        """
        try:
            result = calibrate_timebase_impl(
                file_path=str(self.filepath),
                clocks=clocks,
                kappa_sys=kappa_sys,
                snr_min=snr_min,
            )
            tc = result["timebase_calibration"]
            self.logger.info(
                "Timebase: eps=%+.3f ppm, sigma=%.3f ppm, used %d/%d tones, "
                "g=%.1f MHz, preconditions=%s",
                tc.epsilon * 1e6,
                tc.sigma_epsilon * 1e6,
                tc.n_used,
                tc.n_detected,
                tc.lattice_g_mhz,
                "pass" if tc.preconditions_passed else "fail",
            )
            return cast(TimebaseCalibrationResult, tc)
        except StageDependencyError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to calibrate timebase: {e}") from e

    def load_timebase_calibration(self) -> TimebaseCalibrationResult:
        """Load the persisted :class:`TimebaseCalibrationResult`."""
        return cast(
            TimebaseCalibrationResult,
            load_timebase_calibration_impl(str(self.filepath))["timebase_calibration"],
        )

    def calibrate_tau_G(
        self,
        n_seg: Optional[int] = None,
        t_sigma: Optional[float] = None,
        tau_max_us: Optional[float] = None,
        rss_gate_factor: Optional[float] = None,
        sigma_time: Optional[float] = None,
        snr_min: Optional[float] = None,
        tau_G_bound_lo: Optional[float] = None,
        tau_G_bound_hi: Optional[float] = None,
        tau_G_seeds: Optional[List[float]] = None,
        delta_chi2r_min: Optional[float] = None,
        tau_G_upper_fraction: Optional[float] = None,
        min_contributors: Optional[int] = None,
        sigma_tau_fraction_max: Optional[float] = None,
        bimodality_dominant_fraction: Optional[float] = None,
        compute_band_majorities: Optional[bool] = None,
        min_contributors_per_band: Optional[int] = None,
        settings: Optional[TauCalibrationSettings] = None,
        preset: Optional[str] = None,
    ) -> TauCalibrationResult:
        """Run the Stage 2b Gaussian-shape τ_G calibration.

        Twin of :meth:`calibrate_tau`: per-bin Voigt fits on the STFT
        contributor pool yield a per-band τ_G majority that the Stage 5
        Gaussian path consumes. Persists to ``/stage2b_tau_G_calibration``
        and invalidates downstream stages.

        Parameters left as ``None`` fall through the four-layer
        resolution chain. ``settings=`` and ``preset=`` populate the
        preset layer (mutually exclusive). The resolved settings share
        the ``processing_parameters/stage2b_tau`` block with the pure-exp
        twin -- both twins are alternative outputs of the same algorithm.
        """
        try:
            result = calibrate_tau_G_impl(
                file_path=str(self.filepath),
                n_seg=n_seg,
                t_sigma=t_sigma,
                tau_max_us=tau_max_us,
                rss_gate_factor=rss_gate_factor,
                sigma_time=sigma_time,
                snr_min=snr_min,
                tau_G_bound_lo=tau_G_bound_lo,
                tau_G_bound_hi=tau_G_bound_hi,
                tau_G_seeds=tau_G_seeds,
                delta_chi2r_min=delta_chi2r_min,
                tau_G_upper_fraction=tau_G_upper_fraction,
                min_contributors=min_contributors,
                sigma_tau_fraction_max=sigma_tau_fraction_max,
                bimodality_dominant_fraction=bimodality_dominant_fraction,
                compute_band_majorities=compute_band_majorities,
                min_contributors_per_band=min_contributors_per_band,
                settings=settings,
                preset=preset,
            )
            tc = result["tau_G_calibration"]
            self.logger.info(
                "Stage 2b τ_G: tau_G_maj=%.3f us, sigma_tau_G=%.3f us, "
                "n_eligible=%d, preconditions=%s",
                tc.tau_maj_us,
                tc.sigma_tau_us,
                tc.n_contributors,
                "pass" if tc.preconditions_passed else "fail",
            )
            return cast(TauCalibrationResult, tc)
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to calibrate τ_G: {e}") from e

    def load_tau_G_calibration(self) -> TauCalibrationResult:
        """Load the persisted Gaussian Stage 2b :class:`TauCalibrationResult`."""
        return cast(
            TauCalibrationResult,
            load_tau_G_calibration_impl(str(self.filepath))["tau_G_calibration"],
        )

    def recommend_shape(
        self,
        n_seg: Optional[int] = None,
        t_sigma: Optional[float] = None,
        tau_max_us: Optional[float] = None,
        rss_gate_factor: Optional[float] = None,
        sigma_time: Optional[float] = None,
        snr_min: Optional[float] = None,
        tau_bound_lo: Optional[float] = None,
        tau_bound_hi: Optional[float] = None,
        tau_G_seeds: Optional[List[float]] = None,
        pure_margin_threshold: Optional[float] = None,
        settings: Optional[TauCalibrationSettings] = None,
        preset: Optional[str] = None,
    ) -> ShapeRecommendation:
        """Run the 3-way L/G/V per-bin AICc shape-recommendation hook.

        Mirrors the τ calibrations' STFT classifier and per-bin fit
        machinery, then fits exp / gauss / voigt on every contributor
        bin, computes AICc per bin, and aggregates an SNR-weighted
        majority vote. The verdict's ``recommended_shape`` (``"lorentzian"``
        / ``"gaussian"`` / ``None``) is stamped onto every Stage 2b
        group present on the file so the Stage 5 resolver's *recommended*
        layer picks it up automatically. The Voigt vote mass is
        reported as a diagnostic but does not enter the recommendation
        (the production Stage 5 line-shape selector supports L and G
        only).

        Requires Stage 1 (active region + frequency trim) to have
        completed. The Stage 2b calibrations are *not* required for the
        verdict itself, but the persisted contract only fires when at
        least one of them has run; without a Stage 2b group the
        verdict is returned but no attr is stamped.

        Parameters left as ``None`` fall through the four-layer
        resolution chain. ``settings=`` and ``preset=`` populate the
        preset layer (mutually exclusive). The resolved settings are
        stamped to ``processing_parameters/stage2b_tau`` so a follow-up
        no-kwargs call inherits the same recipe.
        """
        try:
            result = recommend_shape_impl(
                file_path=str(self.filepath),
                n_seg=n_seg,
                t_sigma=t_sigma,
                tau_max_us=tau_max_us,
                rss_gate_factor=rss_gate_factor,
                sigma_time=sigma_time,
                snr_min=snr_min,
                tau_bound_lo=tau_bound_lo,
                tau_bound_hi=tau_bound_hi,
                tau_G_seeds=tau_G_seeds,
                pure_margin_threshold=pure_margin_threshold,
                settings=settings,
                preset=preset,
            )
            rec = result["shape_recommendation"]
            groups = result["groups_written"]
            self.logger.info(
                "Shape recommendation: %s (n_contributors=%d, "
                "vote rates exp=%.1f%%/gauss=%.1f%%/voigt=%.1f%%); "
                "written to %s",
                rec.recommended_shape,
                rec.n_contributors,
                rec.vote_rates["exp"] * 100,
                rec.vote_rates["gauss"] * 100,
                rec.vote_rates["voigt"] * 100,
                groups if groups else "no Stage 2b group present",
            )
            return cast(ShapeRecommendation, rec)
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to recommend shape: {e}") from e

    def visualize_tau_heatmap(
        self,
        output_file: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        figsize: Optional[tuple] = None,
    ) -> Any:
        """Render the 2D STFT-magnitude heatmap (frame x molecular frequency)."""
        from .visualization.tau_calibration_visualization import (
            plot_tau_heatmap_from_file,
        )

        fig = plot_tau_heatmap_from_file(str(self.filepath), figsize=figsize)
        if output_file:
            fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
            self.logger.info(f"Plot saved to: {output_file}")
        elif interactive:
            import matplotlib.pyplot as plt

            plt.show()
        return fig

    def visualize_tau_distribution(
        self,
        output_file: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        figsize: Optional[tuple] = None,
    ) -> Any:
        """Render the tau-distribution analysis panel."""
        from .visualization.tau_calibration_visualization import (
            plot_tau_distribution_from_file,
        )

        fig = plot_tau_distribution_from_file(str(self.filepath), figsize=figsize)
        if output_file:
            fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
            self.logger.info(f"Plot saved to: {output_file}")
        elif interactive:
            import matplotlib.pyplot as plt

            plt.show()
        return fig

    def detect_start_time(
        self,
        sweep_max_us: Optional[float] = None,
        step_us: Optional[float] = None,
        guard_margin_us: Optional[float] = None,
        floor_factor: Optional[float] = None,
        band: Optional[Tuple[float, float]] = None,
        stamp: bool = True,
        *,
        settings: Optional[StartDetectionSettings] = None,
    ) -> StartDetectionResult:
        """Infer a good FID ``start_us`` from the data and stamp it.

        Sweeps the FID window start time, integrates the FT magnitude over the
        active band, and locates the chirp-end collapse; the recommended start
        is ``chirp_end + guard_margin_us`` (the switch-bounce settling time).
        When ``stamp=True`` (default) the recommended ``start_us`` is written to
        the Stage 0 ``recommended_processing`` layer, so a later ``compute_ft``
        with no explicit ``start_us`` inherits it. Requires Stage 0 (FID) only;
        the integration band is resolved from the canonical Stage 1 trim when
        present, else the full positive spectrum. Equivalent to the CLI
        ``start run`` command and ``ftmwpipeline.api.detect_start_time``.

        Parameters
        ----------
        sweep_max_us, step_us, guard_margin_us, floor_factor :
            Individual overrides of the matching
            :class:`~ftmwpipeline.core.start_detection_settings.StartDetectionSettings`
            fields. ``guard_margin_us`` is the instrument-specific ringdown
            margin added past the chirp end.
        band : tuple of float, optional
            Explicit ``(min_mhz, max_mhz)`` integration band override.
        stamp : bool, default True
            Whether to persist the recommended ``start_us`` to the recommended
            layer.
        settings : StartDetectionSettings, optional
            A full settings bundle; the explicit kwargs above win per-field.

        Returns
        -------
        StartDetectionResult
            The recommendation plus diagnostics (chirp-end, sweep arrays).
        """
        resolved = self._resolve_start_detection_settings(
            settings,
            sweep_max_us=sweep_max_us,
            step_us=step_us,
            guard_margin_us=guard_margin_us,
            floor_factor=floor_factor,
            band=band,
        )
        result = detect_start_time_impl(
            str(self.filepath), settings=resolved, stamp=stamp
        )
        return cast(StartDetectionResult, result["start_detection"])

    @staticmethod
    def _resolve_start_detection_settings(
        settings: Optional[StartDetectionSettings],
        *,
        sweep_max_us: Optional[float],
        step_us: Optional[float],
        guard_margin_us: Optional[float],
        floor_factor: Optional[float],
        band: Optional[Tuple[float, float]],
    ) -> StartDetectionSettings:
        """Overlay explicit per-knob kwargs onto a base settings bundle."""
        from dataclasses import replace

        base = settings or StartDetectionSettings()
        overrides: Dict[str, Any] = {}
        if sweep_max_us is not None:
            overrides["sweep_max_us"] = float(sweep_max_us)
        if step_us is not None:
            overrides["step_us"] = float(step_us)
        if guard_margin_us is not None:
            overrides["guard_margin_us"] = float(guard_margin_us)
        if floor_factor is not None:
            overrides["floor_factor"] = float(floor_factor)
        if band is not None:
            overrides["band_min_mhz"] = float(band[0])
            overrides["band_max_mhz"] = float(band[1])
        return replace(base, **overrides) if overrides else base

    def visualize_start_detection(
        self,
        output_file: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        figsize: Optional[tuple] = None,
        *,
        settings: Optional[StartDetectionSettings] = None,
    ) -> Any:
        """Render the start-detection sweep diagnostic (no stamping)."""
        from .visualization.start_detection_visualization import (
            plot_start_detection_from_file,
        )

        fig = plot_start_detection_from_file(
            str(self.filepath), settings=settings, figsize=figsize
        )
        if output_file:
            fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
            self.logger.info(f"Plot saved to: {output_file}")
        elif interactive:
            import matplotlib.pyplot as plt

            plt.show()
        return fig

    def detect_peaks(
        self,
        min_snr: Optional[float] = None,
        weak_medium_snr: Optional[float] = None,
        medium_strong_snr: Optional[float] = None,
        sg_window: Optional[int] = None,
        sg_order: Optional[int] = None,
        primary_window: Optional[str] = None,
        min_exclusion_mhz: Optional[float] = None,
        run_gap_pass: Optional[bool] = None,
        *,
        settings: Optional[PeakDetectionSettings] = None,
        preset: Optional[str] = None,
    ) -> List[Peak]:
        """Detect and classify peaks (Stage 3, two-pass).

        Requires Stage 1 (FT) and Stage 2 (noise) completed. Runs an apodized
        primary pass plus a leakage-masked unapodized gap pass, scores every
        peak on the unapodized spectrum, classifies by SNR, and persists ALL
        detected peaks to the .ftmw file. Equivalent to the CLI
        ``peaks run`` command and ``ftmwpipeline.api.detect_peaks``.

        Detection operates on the Stage 1 persisted canonical spectrum,
        including its frequency trim range.  Peaks are reported on that user
        grid with amplitude and SNR measured against the canonical Stage 2
        noise.  There is no per-Stage-3 trim or zpf.

        Parameters
        ----------
        min_snr : float, optional
            Promotion SNR cutoff (peaks at/above this threshold on the user
            grid are marked ``promoted=True`` and move to Stage 4); detection
            runs aggressively below this internally. ALL detected peaks are
            stored; ``promoted`` marks the Stage-4 gate. Default 3.0.
        weak_medium_snr : float, optional
            Weak/medium SNR boundary for classification (default 10.0).
        medium_strong_snr : float, optional
            Medium/strong SNR boundary for classification (default 50.0).
        sg_window : int, optional
            Savitzky-Golay smoothing window in points (default 11).
        sg_order : int, optional
            Savitzky-Golay polynomial order (default 3).
        primary_window : str, optional
            Apodization window for the primary (position-finding) pass; any
            scipy.signal window name (e.g. ``"blackmanharris"``,
            ``"blackman"``, ``"hann"``). Default ``"blackmanharris"`` -- a
            strong window that suppresses truncation sidelobes so the primary
            strong-line list is clean. Affects only which positions are found,
            never reported amplitude/SNR.
        min_exclusion_mhz : float, optional
            Minimum gap-pass exclusion half-width per primary peak in MHz.
        run_gap_pass : bool, optional
            If False, disable the unapodized gap pass (primary pass only).
        settings : PeakDetectionSettings, optional
            Bundle of Stage 3 knobs (preset-layer of the four-layer
            resolution chain); fields left ``None`` fall through. Mutually
            exclusive with ``preset``.
        preset : str, optional
            Bare preset name or path to a YAML file carrying a ``stage3:``
            block. Mutually exclusive with ``settings``.

        Returns
        -------
        list of Peak
            ALL detected peaks (promoted and non-promoted), sorted by
            frequency. Each peak's ``properties`` dict includes ``promoted``
            (bool), ``internal_snr``, ``internal_frequency``, and
            ``detection_pass``.

        Raises
        ------
        StageDependencyError
            If Stage 1 or Stage 2 has not been completed.
        RuntimeError
            If detection fails.
        """
        try:
            result = detect_peaks_impl(
                file_path=str(self.filepath),
                min_snr=min_snr,
                weak_medium_snr=weak_medium_snr,
                medium_strong_snr=medium_strong_snr,
                sg_window=sg_window,
                sg_order=sg_order,
                primary_window=primary_window,
                min_exclusion_mhz=min_exclusion_mhz,
                run_gap_pass=run_gap_pass,
                settings=settings,
                preset=preset,
            )
            self.logger.info(
                "Stage 3: %d detected (%d promoted, SNR >= %.1f); "
                "%d primary, %d gap",
                result["n_peaks"],
                result["n_promoted"],
                result["promotion_min_snr"],
                result["n_primary"],
                result["n_gap"],
            )
            return cast(List[Peak], result["peaks"])
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to detect peaks: {e}") from e

    def load_peaks(self) -> List[Peak]:
        """Load the persisted Stage 3 peak list (validates structure loudly)."""
        return cast(List[Peak], load_peaks_impl(str(self.filepath))["peaks"])

    def visualize_peaks(
        self,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        y_max_factor: Optional[float] = None,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_snr_histogram: bool = False,
    ) -> Any:
        """Overlay classified detected peaks on the unapodized spectrum (Stage 3).

        Equivalent to the CLI ``peaks show`` command. Interactive
        matplotlib (log-y) by default; pass ``interactive=False`` with
        ``output_file`` to save instead.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.
        title : str, optional
            Custom plot title.
        y_max_factor : float, optional
            Y-axis max as multiple of median RMS noise (default 25.0).
        backend : str, default ``'matplotlib'``
            Plotting backend (``'matplotlib'`` or ``'plotly'``).
        interactive : bool, default True
            Whether to open an interactive window.
        output_file : str or Path, optional
            Save plot to this path (non-interactive mode).
        show_snr_histogram : bool, default False
            If True, add a second panel showing the user-grid SNR distribution
            with the promotion cutoff marked (curation view).

        Returns
        -------
        figure
            Matplotlib figure (single-panel or two-panel when
            ``show_snr_histogram=True``).

        Raises
        ------
        RuntimeError
            If Stage 3 has not been completed or visualization fails.
        """
        try:
            fig = visualize_peaks_impl(
                file_path=str(self.filepath),
                figsize=figsize,
                title=title,
                y_max_factor=y_max_factor,
                backend=backend,
                interactive=interactive,
                show_snr_histogram=show_snr_histogram,
            )
            if not interactive and output_file:
                fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")
            elif interactive and backend == "matplotlib":
                import matplotlib.pyplot as plt

                plt.show()
            return fig
        except Exception as e:
            raise RuntimeError(f"Failed to create peak visualization: {e}") from e

    def assign_windows(
        self,
        edge_m: Optional[int] = None,
        trim_m: Optional[int] = None,
        edge_threshold: Optional[float] = None,
        max_window_width_mhz: Optional[float] = None,
        min_freeze_snr: Optional[float] = None,
        min_window_half_width_mhz: Optional[float] = None,
        magnitude_attachment_threshold: Optional[float] = None,
        tau_us: Optional[float] = None,
        max_peaks_per_window: Optional[int] = None,
        max_window_width_points: Optional[int] = None,
        min_window_half_width_points: Optional[int] = None,
        *,
        settings: Optional[WindowPlanningSettings] = None,
        preset: Optional[str] = None,
    ) -> WindowPlan:
        """Assign analysis windows (Stage 4), turning promoted peaks into a fit plan.

        Requires Stage 3 (peak detection) completed. Builds a set of disjoint
        fit windows over the persisted user spectrum, each annotated with the
        peaks to fit freely, the strong out-of-band lines whose leakage is
        carried frozen, a fit dependency order, and a difficulty class. Stage 4
        is purely structural -- it makes no fits. Equivalent to the CLI
        ``windows run`` command and ``ftmwpipeline.api.assign_windows``.

        Consumes only the peaks flagged ``promoted`` by Stage 3, on the Stage 1
        canonical spectrum with the canonical Stage 2 noise. The result is
        persisted to ``/stage4_windows`` and the stage marked complete.

        Parameters
        ----------
        edge_m : int, optional
            Rolling-scan complex-edge coherence band width (default 64).
        trim_m : int, optional
            Trim-refinement band width (default 32).
        edge_threshold : float, optional
            ``S_coh`` threshold separating leakage-touched from line-free
            regions (default 8.0).
        max_window_width_mhz : float, optional
            Width cap; a wider window is HARD and gets a split proposal
            (default 40.0).
        min_freeze_snr : float, optional
            Freeze-eligibility SNR cutoff for fixed contributors (default 50.0).
        min_window_half_width_mhz : float, optional
            Minimum half-width of a window around an isolated weak line
            (default 2.0).
        magnitude_attachment_threshold : float, optional
            Tier-1 contributor-attachment threshold in units of σ_c
            (default 0.1). See
            :data:`ftmwpipeline.preprocessing.window_planning.DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD`.
        tau_us : float, optional
            Assumed decay constant for the analytic leakage reach
            (default: undamped/boxcar limit).
        max_window_width_points : int, optional
            Width cap in active-FT grid points -- the portable form of the
            cap (bin width varies across instruments). ``0`` defers to
            ``max_window_width_mhz``; a positive value (default 96)
            supersedes it.
        min_window_half_width_points : int, optional
            Window margin in active-FT grid points -- the noise budget each
            side of a window's outermost peak (proto half-width and trim
            budget). Supersedes ``min_window_half_width_mhz`` when positive
            (default 32).
        settings : WindowPlanningSettings, optional
            Bundle of Stage 4 knobs (preset-layer of the four-layer
            resolution chain); fields left ``None`` fall through. Mutually
            exclusive with ``preset``.
        preset : str, optional
            Bare preset name or path to a YAML file carrying a ``stage4:``
            block. Mutually exclusive with ``settings``.

        Returns
        -------
        WindowPlan
            The fit plan: disjoint windows, dependency DAG, topological order,
            parallel batches, parameters and diagnostics.

        Raises
        ------
        StageDependencyError
            If Stage 3 has not been completed.
        RuntimeError
            If window assignment fails.
        """
        try:
            result = assign_windows_impl(
                file_path=str(self.filepath),
                edge_m=edge_m,
                trim_m=trim_m,
                edge_threshold=edge_threshold,
                max_window_width_mhz=max_window_width_mhz,
                min_freeze_snr=min_freeze_snr,
                min_window_half_width_mhz=min_window_half_width_mhz,
                magnitude_attachment_threshold=magnitude_attachment_threshold,
                tau_us=tau_us,
                max_peaks_per_window=max_peaks_per_window,
                max_window_width_points=max_window_width_points,
                min_window_half_width_points=min_window_half_width_points,
                settings=settings,
                preset=preset,
            )
            self.logger.info(
                "Stage 4: %d windows (%d hard), %d batches, %d free peaks, "
                "%d fixed contributors",
                result["n_windows"],
                result["n_hard"],
                result["n_batches"],
                result["n_free_peaks"],
                result["n_fixed_contributors"],
            )
            return cast(WindowPlan, result["plan"])
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to assign windows: {e}") from e

    def load_windows(self) -> WindowPlan:
        """Load the persisted Stage 4 window plan (validates structure loudly)."""
        return cast(WindowPlan, load_windows_impl(str(self.filepath))["plan"])

    def visualize_windows(
        self,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        y_max_factor: Optional[float] = None,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Overlay the Stage 4 window plan on the spectrum.

        Equivalent to the CLI ``windows show`` command. Shows each fit
        window's span (shaded by difficulty), free peaks, fixed contributors,
        and the rolling complex-edge coherence statistic. Requires Stage 4
        completed.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.
        title : str, optional
            Custom plot title.
        y_max_factor : float, optional
            Spectrum-panel y-axis headroom (default 25.0).
        backend : str, default ``'matplotlib'``
            Plotting backend (only ``'matplotlib'`` supported).
        interactive : bool, default True
            Whether to open an interactive window.
        output_file : str or Path, optional
            Save plot to this path (non-interactive mode).

        Returns
        -------
        figure
            Matplotlib figure.

        Raises
        ------
        RuntimeError
            If Stage 4 has not been completed or visualization fails.
        """
        try:
            fig = visualize_windows_impl(
                file_path=str(self.filepath),
                figsize=figsize,
                title=title,
                y_max_factor=y_max_factor,
                backend=backend,
                interactive=interactive,
            )
            if not interactive and output_file:
                fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")
            elif interactive and backend == "matplotlib":
                import matplotlib.pyplot as plt

                plt.show()
            return fig
        except Exception as e:
            raise RuntimeError(f"Failed to create window visualization: {e}") from e

    def fit_peaks(
        self,
        tau0_us: Optional[float] = None,
        fit_tau: Optional[bool] = None,
        max_decay_factor: Optional[float] = None,
        residual_edge_threshold: Optional[float] = None,
        residual_edge_m: Optional[int] = None,
        max_thaw_rounds: Optional[int] = None,
        max_replan_rounds: Optional[int] = None,
        max_residual_rescue_rounds: Optional[int] = None,
        rescue_snr_threshold: Optional[float] = None,
        rescue_prominence_threshold: Optional[float] = None,
        tau_maj_override_us: Optional[float] = None,
        sigma_tau_override_us: Optional[float] = None,
        per_band_tau: Optional[bool] = None,
        shape: Optional[str] = None,
        settings: Optional["StageFitSettings"] = None,
        preset: Optional[str] = None,
    ) -> SpectrumFit:
        """Fit each Stage 4 window's lines (Stage 5).

        Requires Stage 4 (window assignment) completed. Drives the conservative
        add-one-peak loop over each window with the shared per-window decay
        ``tau`` and the frozen-contributor model, then the residual
        edge-coherence handshake (local thaw + structural replan). Equivalent
        to the CLI ``fit run`` command and ``ftmwpipeline.api.fit_peaks``.

        The fit runs on the active-portion FT (computed on demand from the
        FID + canonical Stage 1 settings), so per-bin statistics are
        independent and reduced chi-squared / F-test / AIC are calibrated as
        written. The persistent :class:`SpectrumFit` -- per-window
        :class:`FittingResult` s, the merged global fitted-peak list, the
        thaw / replan histories, and the parameters used -- is written to
        ``/stage5_fitting``.

        Parameters
        ----------
        tau0_us : float, optional
            Starting / default shared decay constant per window
            (microseconds). Defaults to the Stage 1 ``expf_us`` when set,
            otherwise to ``T_active / 3``.
        fit_tau : bool, optional
            Free vs fixed per-window tau (default True).
        max_decay_factor : float, optional
            Tau bound factor ``k``: tau in ``[tau0/k, tau0*k]`` (default 5).
        residual_edge_threshold : float, optional
            ``S_coh`` threshold above which a residual edge triggers a thaw
            attempt.
        residual_edge_m : int, optional
            Band width (in active-FT bins) of the residual-edge coherence
            test.
        max_thaw_rounds : int, optional
            Maximum local-thaw rounds per window per call.
        max_replan_rounds : int, optional
            Maximum structural-replan rounds per call. Pass 0 to disable
            structural renegotiation.
        max_residual_rescue_rounds : int, optional
            Cap on per-window residual-rescue + joint-refit cycles.
            ``None`` (the default) resolves to the calibrated default
            cap (currently 5); pass ``0`` to disable the rescue pass
            entirely (escape hatch for diagnostic re-fits). The rescue
            is a structural part of the fit and runs on every window's
            post-thaw fit by default.
        rescue_snr_threshold, rescue_prominence_threshold : optional
            Detector tuning knobs for the rescue -- see
            :func:`fit_peaks_impl` for defaults. Ignored when
            ``max_residual_rescue_rounds`` is 0.
        tau_maj_override_us, sigma_tau_override_us : float, optional
            Atomic-pair manual override for the Stage 2b tau calibration.
            When both are supplied (positive), they replace any persisted
            Stage 2b result for this fit -- useful for A/B-ing a hand-tuned
            tau anchor against the persisted one, or for forcing a
            calibrated tau when Stage 2b has not been run. Supplying only
            one of the pair raises ``ValueError``.
        per_band_tau : bool, default True
            Route each window to its band-local ``(tau_maj, sigma_tau)``
            from the persisted Stage 2b ``band_majorities``. When
            ``True`` (the default) and ``calibrate_tau(...,
            compute_band_majorities=True)`` has been run, the per-window
            prior reflects the band's own tau majority -- crucial for
            wide bands with monotonic horn-coupling τ ∝ 1/f. Falls back
            to the band-wide ``(tau_maj_us, sigma_tau_us)`` (and then to
            no prior) when band_majorities aren't persisted. Pass
            ``False`` to skip per-band routing even when band majorities
            are available. Silently skipped when ``tau_maj_override_us``
            / ``sigma_tau_override_us`` is set (the explicit override
            wins).
        shape : {"lorentzian", "gaussian"}, default "lorentzian"
            Time-domain envelope of the per-line model.
            ``"lorentzian"`` uses ``exp(-t/τ)``; ``"gaussian"`` uses
            ``exp(-(t/τ_G)²)``. When ``"gaussian"`` is selected the
            Stage 2b τ_G calibration (``calibrate_tau_G(...)``) is read
            in place of the pure-exp ``calibrate_tau`` for the
            bidirectional τ anchoring penalty; missing τ_G calibration
            still fits, but without a prior.
        settings : StageFitSettings, optional
            Bundle of Stage 5 knobs that enters the resolution chain at
            the *preset* layer. Per-kwarg explicit overrides above
            (``tau0_us``, ``max_decay_factor``, ...) win over the
            corresponding field on ``settings``; ``settings`` wins over
            persisted and recommended values, which win over the hard
            defaults. Build with
            :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
            or load from a preset YAML
            (:func:`~ftmwpipeline.core.stage_fit_settings.from_yaml`).
        preset : str, optional
            Name of a packaged preset (e.g. ``"instrument_bc_2638"``) or
            a path to a YAML file. Enters the resolution chain at the
            same *preset* layer as ``settings`` -- the two are
            alternatives; passing both raises ``ValueError``. The preset
            name is captured in the persisted Stage 5 fit's audit attrs
            for reproducibility.

        Returns
        -------
        SpectrumFit
            The persistent fit aggregate.

        Raises
        ------
        StageDependencyError
            If Stage 4 has not been completed.
        RuntimeError
            If fitting fails.
        """
        try:
            result = fit_peaks_impl(
                file_path=str(self.filepath),
                tau0_us=tau0_us,
                fit_tau=fit_tau,
                max_decay_factor=max_decay_factor,
                residual_edge_threshold=residual_edge_threshold,
                residual_edge_m=residual_edge_m,
                max_thaw_rounds=max_thaw_rounds,
                max_replan_rounds=max_replan_rounds,
                max_residual_rescue_rounds=max_residual_rescue_rounds,
                rescue_snr_threshold=rescue_snr_threshold,
                rescue_prominence_threshold=rescue_prominence_threshold,
                tau_maj_override_us=tau_maj_override_us,
                sigma_tau_override_us=sigma_tau_override_us,
                per_band_tau=per_band_tau,
                shape=shape,
                settings=settings,
                preset=preset,
            )
            self.logger.info(
                "Stage 5: %d windows, %d fitted peaks; thaw %d/%d, "
                "rescue %d/%d (added %d, %d rescue-origin pruned), "
                "%d structural replans accepted (revision %d)",
                result["n_windows"],
                result["n_fitted_peaks"],
                result["n_thaw_accepted"],
                result["n_thaw_events"],
                result["n_rescue_accepted"],
                result["n_rescue_events"],
                result["n_rescue_added"],
                result["n_rescue_origin_pruned"],
                result["n_replan_accepted"],
                result["final_plan_revision"],
            )
            return cast(SpectrumFit, result["fit"])
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to fit peaks: {e}") from e

    def load_fit(self) -> SpectrumFit:
        """Load the persisted Stage 5 fit (validates structure loudly)."""
        return cast(SpectrumFit, load_fit_impl(str(self.filepath))["fit"])

    def candidate_ledger(
        self,
        window_id: Optional[int] = None,
        *,
        bar: float = DEFAULT_DISPLAY_BAR,
    ) -> List[LedgerCandidate]:
        """Derive the Stage 6 candidate ledger from the persisted Stage 5 fit.

        Returns revivable candidates from the conservative add-loop and rescue
        records -- candidates that were considered but not accepted by the
        automatic fit.  The ledger is derived on demand from the already-persisted
        audit trail; it does not re-fit anything.

        Parameters
        ----------
        window_id :
            When given, return candidates for that window only.
        bar :
            Display SNR / evidence bar; candidates below it are dropped.

        Returns
        -------
        list of LedgerCandidate
            Sorted by molecular frequency.

        Requires Stage 5 completed.
        """
        return get_candidate_ledger_impl(self.filepath, window_id=window_id, bar=bar)

    def review_edit(
        self,
        window_id: int,
        *,
        add: Sequence[float] = (),
        remove: Sequence[float] = (),
    ) -> RefitWindowResult:
        """User-directed single-window refit (Stage 6 ``review edit``).

        Re-fits ``window_id`` from the persisted Stage 5 fit using the
        production NLS primitive, applying ``add``/``remove`` edits.  User-
        added peaks carry ``origin="user"`` and survive the HDF5 round-trip;
        removed peaks are excluded from the refit and will not be re-added by
        the rescue pass.

        Parameters
        ----------
        window_id :
            The window to refit.
        add :
            Molecular frequencies (MHz) of peaks to add.  Snapped to the
            nearest ledger candidate within 50 kHz or seeded fresh at F.
        remove :
            Molecular frequencies (MHz) of fitted peaks to remove.  Snapped
            to the nearest fitted peak within 50 kHz.

        Returns
        -------
        RefitWindowResult
            Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks.

        Requires Stage 5 completed.
        """
        return refit_window_impl(
            self.filepath,
            window_id,
            add=add,
            remove=remove,
        )

    def review_merge(
        self,
        window_id: int,
        peaks: Sequence[float],
        *,
        snap_tol_mhz: float = 0.05,
    ) -> RefitWindowResult:
        """Collapse ≥2 fitted peaks in a window into one (Stage 6 ``review merge``).

        Removes the named peaks and adds one replacement seeded at their
        SNR-weighted centroid (or amplitude-weighted centroid when SNR is
        unavailable).  When the pair matches a persisted doublet-alternative
        record with a successful merged refit, the recorded merged seed is
        used instead of the centroid.  All products carry ``origin="user"``.

        Parameters
        ----------
        window_id :
            The window containing the peaks to merge.
        peaks :
            Molecular frequencies (MHz) of the peaks to collapse (≥2).
        snap_tol_mhz :
            Maximum distance (MHz) for frequency snapping to fitted peaks.

        Returns
        -------
        RefitWindowResult
            Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks.

        Requires Stage 5 completed.
        """
        return merge_peaks_impl(
            self.filepath,
            window_id,
            peaks,
            snap_tol_mhz=snap_tol_mhz,
        )

    def review_split(
        self,
        window_id: int,
        peak: float,
        *,
        into: int = 2,
        snap_tol_mhz: float = 0.05,
    ) -> RefitWindowResult:
        """Replace one fitted peak with ``into`` peaks (Stage 6 ``review split``).

        Removes the named peak and adds ``into`` replacements spread
        symmetrically about it by ±½ of one Fourier resolution element
        (``1 / acquisition_us`` MHz).  All products carry ``origin="user"``.

        Parameters
        ----------
        window_id :
            The window containing the peak to split.
        peak :
            Molecular frequency (MHz) of the peak to split.
        into :
            Number of replacement peaks (≥2, default 2).
        snap_tol_mhz :
            Maximum distance (MHz) for frequency snapping to fitted peaks.

        Returns
        -------
        RefitWindowResult
            Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks.

        Requires Stage 5 completed.
        """
        return split_peak_impl(
            self.filepath,
            window_id,
            peak,
            into=into,
            snap_tol_mhz=snap_tol_mhz,
        )

    def review_run(
        self,
        *,
        bar: float = DEFAULT_DISPLAY_BAR,
        attention_candidate_evidence: float = DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
        sigma_floor_khz: Optional[float] = None,
    ) -> ReviewRunResult:
        """Build or refresh the Stage 6 curation layer and final-products table.

        Computes advisory attention reasons for each fitted window, consolidates
        the calibrated final-products table (timebase-corrected frequencies + the
        three-term ``sigma_f`` budget), persists the
        :class:`~ftmwpipeline.core.data_structures.Stage6Review` to the
        ``stage6_review`` HDF5 group, and marks the stage complete.

        Existing per-window provenance (``"reviewed"``/``"user-edited"``) and
        the decision log are preserved; only attention reasons are refreshed.

        Parameters
        ----------
        bar :
            Display bar for the candidate-bearing attention reason.
        attention_candidate_evidence :
            Evidence threshold above which a candidate-bearing window flags
            (stiffer than ``bar``; keeps the attention surface actionable).
        sigma_floor_khz :
            When given, persist this user-declared accuracy floor (kHz) into the
            file-level ``/frequency_calibration`` record and fold it into the
            budget.  ``None`` (default) keeps the persisted floor unchanged.

        Returns
        -------
        ReviewRunResult
            Total window count, attention count, and per-kind breakdown.

        Requires Stage 5 completed.
        """
        return review_run_impl(
            self.filepath,
            bar=bar,
            attention_candidate_evidence=attention_candidate_evidence,
            sigma_floor_khz=sigma_floor_khz,
        )

    def set_sigma_floor(self, sigma_floor_khz: float) -> None:
        """Declare the systematic frequency-accuracy floor (kHz), persisted in-file.

        Stores the floor as file-level provenance (``/frequency_calibration``)
        so any reported ``sigma_f`` is reproducible from the record alone.  Call
        :meth:`review_run` afterwards to fold the new floor into the budget.
        """
        set_sigma_floor_impl(self.filepath, sigma_floor_khz)

    def final_products(self) -> Optional[FinalProducts]:
        """Return the persisted Stage 6 calibrated final-products table.

        Read-only; ``None`` until :meth:`review_run` has built it.
        """
        return get_final_products_impl(self.filepath)

    def report_table(
        self,
        *,
        fmt: str = "csv",
        output: Optional[Union[str, Path]] = None,
        catalog: Optional[Union[str, Path]] = None,
        catalog_n_sigma: float = 3.0,
    ) -> str:
        """Render the calibrated final-products table (report Level 1).

        Serializes the persisted :class:`FinalProducts` table to ``csv``,
        ``json``, or a LaTeX ``booktabs`` table. Renders the persisted record;
        does not recompute. Requires :meth:`review_run` to have built the table.

        Parameters
        ----------
        fmt :
            ``"csv"`` (default), ``"json"``, or ``"latex"``.
        output :
            When given, also write the rendered text to this path.
        catalog :
            Optional frequency-catalog path; when given, each line is
            proximity-flagged against the nearest catalog entry (label echo
            only, never an assignment) and the match is added to the output.
        catalog_n_sigma :
            Catalog match tolerance in combined sigmas (default ``3``).

        Returns
        -------
        str
            The rendered table.
        """
        return report_table_impl(
            self.filepath,
            fmt=fmt,
            output=output,
            catalog=catalog,
            catalog_n_sigma=catalog_n_sigma,
        )

    def report_run(
        self,
        *,
        output_dir: Union[str, Path],
        windows: str = "all",
        emit_table: bool = True,
        emit_html: bool = True,
        table_format: str = "csv",
        scope: str = "full",
        catalog: Optional[Union[str, Path]] = None,
        catalog_n_sigma: float = 3.0,
    ) -> Dict[str, Optional[str]]:
        """Write the default Stage 6 deliverables: the L1 table + the L3 report.

        Writes the Level-1 final-products table (``<stem>_lines.csv``) and the
        self-contained Level-3 HTML report with every window folded in
        (``<stem>_report.html``) into *output_dir*, reusing the existing
        renderers. Renders the persisted record; does not recompute. Requires
        :meth:`review_run` to have built the final-products table.

        Parameters
        ----------
        output_dir :
            Directory to write the artifacts into (created if absent).
        windows :
            ``"all"`` (a detail page per window) or ``"attention"`` (pages only
            for review-flagged windows). The index always lists every window.
        emit_table :
            Write the Level-1 table artifact (default ``True``).
        emit_html :
            Write the Level-3 HTML report (default ``True``).
        table_format :
            Format for the table artifact: ``"csv"`` (default), ``"json"``, or
            ``"latex"``.
        scope :
            ``"full"`` (default) writes one self-contained ``<stem>_report.html``
            with every per-window detail folded in; ``"summary"`` writes
            ``<stem>_report_summary.html`` (index + methods only). Either way the
            output is a single self-contained file.
        catalog :
            Optional frequency-catalog path; adds proximity-match cross-references
            to the table and HTML (label echo only, never an assignment).
        catalog_n_sigma :
            Catalog match tolerance in combined sigmas (default ``3``).

        Returns
        -------
        dict
            ``{"table": <path|None>, "html": <path|None>}`` -- the path of each
            artifact written (``None`` when suppressed).
        """
        return report_run_impl(
            self.filepath,
            output_dir=output_dir,
            windows=windows,
            emit_table=emit_table,
            emit_html=emit_html,
            table_format=table_format,
            scope=scope,
            catalog=catalog,
            catalog_n_sigma=catalog_n_sigma,
        )

    def review_accept(
        self,
        window_id: int,
        *,
        candidate_freq: Optional[float] = None,
    ) -> Optional[RefitWindowResult]:
        """Accept a window as-is or accept a specific revived candidate.

        With no ``candidate_freq``: records a ``"accept"`` log entry, sets
        the window's provenance to ``"reviewed"``, and returns ``None``.  The
        fit is left unchanged; the acceptance is a "looked, no change needed"
        decision.

        With ``candidate_freq``: adds the candidate peak via a single-window
        refit, records an ``"add"`` log entry, sets provenance to
        ``"user-edited"``, and returns the :class:`RefitWindowResult`.

        Parameters
        ----------
        window_id :
            The window to accept.
        candidate_freq :
            When given, accept by adding this molecular frequency (MHz) as a
            new peak.  Snapped to the nearest ledger candidate within 50 kHz.

        Returns
        -------
        RefitWindowResult or None
        """
        return review_accept_impl(
            self.filepath, window_id, candidate_freq=candidate_freq
        )

    def review_status(self) -> Stage6Review:
        """Load the persisted Stage 6 review state, or return an empty one.

        Read-only: safe to call before ``review run``.

        Returns
        -------
        Stage6Review
            The persisted per-window statuses and decision log.
        """
        return get_review_status_impl(self.filepath)

    def rank_windows(
        self,
        by: str,
        *,
        top: Optional[int] = None,
    ) -> List["RankedWindow"]:
        """Rank fit windows by a persisted per-window statistic (read-only).

        On-demand exploration decoupled from the attention flags: ranks all
        windows worst-first by ``by`` (see
        :data:`~ftmwpipeline._internal.stage6_impl.RANK_METRICS` for the choices
        -- ``min-snr``, ``max-vif``, ``chi2r``, ``candidate-evidence``,
        ``edge-distance``, ``spur-proximity``, ``merged-chi2r``).

        Parameters
        ----------
        by :
            Metric name (``_`` and ``-`` interchangeable).
        top :
            Return at most this many windows; ``None`` returns all.

        Requires Stage 5 completed.
        """
        return rank_windows_impl(self.filepath, by=by, top=top)

    def validate_stage5_shape_error(
        self,
        kappa: Optional[float] = None,
        noise_floor: Optional[float] = None,
        ground_truth: Optional[Union[str, Path]] = None,
        match_tol_fwhm: float = 0.5,
    ) -> Dict[str, Any]:
        """Assess the persisted Stage 5 fit against the SNR-aware framework.

        Read-only. Returns a Tier 1 (SNR-aware per-window acceptance
        ``chi2r <= F + (kappa*SNR_max)**2`` with the fractional deficit ``eps``
        binned by SNR) / Tier 2 (rescue/merge/thaw gate firing) / Tier 3
        (known-line ground truth, when ``ground_truth`` is given) report.
        Equivalent to the CLI ``fit check`` command. Requires
        Stage 5 completed.
        """
        try:
            report = validate_stage5_shape_error_impl(
                file_path=str(self.filepath),
                kappa=kappa,
                noise_floor=noise_floor,
                ground_truth=str(ground_truth) if ground_truth is not None else None,
                match_tol_fwhm=match_tol_fwhm,
            )
            t1 = report["tier1"]
            self.logger.info(
                "Stage 5 shape-error validation: %d windows, SNR-aware pass "
                "rate %.3f (kappa=%.3g)",
                t1.get("n_windows", 0),
                t1.get("pass_rate", 0.0),
                report["parameters"]["kappa"],
            )
            return report
        except StageDependencyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to validate Stage 5 shape error: {e}") from e

    def visualize_fit(
        self,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        window_id: Optional[int] = None,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Overlay the Stage 5 fit on the spectrum.

        Equivalent to the CLI ``fit show`` command. With ``window_id``
        set, draws a per-window detail figure (re/im, magnitude+residual,
        time envelope, audit-trail); otherwise an overview overlay of the
        fitted model on the persisted spectrum. Requires Stage 5 completed.
        """
        try:
            fig = visualize_fit_impl(
                file_path=str(self.filepath),
                figsize=figsize,
                title=title,
                window_id=window_id,
                backend=backend,
                interactive=interactive,
            )
            if not interactive and output_file:
                fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
                self.logger.info(f"Plot saved to: {output_file}")
            elif interactive and backend == "matplotlib":
                import matplotlib.pyplot as plt

                plt.show()
            return fig
        except Exception as e:
            raise RuntimeError(f"Failed to create fit visualization: {e}") from e

    def show_fit(
        self,
        *,
        window_ids: Optional[list] = None,
        freqs: Optional[list] = None,
        random_n: Optional[int] = None,
        random_seed: Optional[int] = None,
        top_snr: Optional[int] = None,
        all_windows: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        show_audit: bool = False,
        apodize: Optional[str] = None,
        apodize_us: Optional[float] = None,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Show the Stage 5 fit -- overview, or a consolidated per-window detail
        figure for each selected window.

        Equivalent to the CLI ``fit show`` command. With no selector, returns
        the spectrum-wide overview. The selectors (``window_ids`` / ``freqs`` /
        ``random_n`` / ``top_snr`` / ``all_windows``) compose as a union; with
        ``output_dir`` each detail figure is written as
        ``<stem>_window_<id>.png``. Returns
        ``{"mode", "window_ids", "figures", "paths", "log"}``. Requires Stage 5.
        """
        try:
            result = fit_show_impl(
                file_path=str(self.filepath),
                window_ids=window_ids,
                freqs=freqs,
                random_n=random_n,
                random_seed=random_seed,
                top_snr=top_snr,
                all_windows=all_windows,
                output_dir=str(output_dir) if output_dir is not None else None,
                show_audit=show_audit,
                apodize=apodize,
                apodize_us=apodize_us,
                figsize=figsize,
                title=title,
            )
            if interactive and not result["paths"]:
                import matplotlib.pyplot as plt

                plt.show()
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to show fit: {e}") from e

    def info(self) -> Dict[str, Any]:
        """
        Get pipeline file information and status.

        Returns
        -------
        dict
            Pipeline status and metadata information
        """
        try:
            validation_report = validate_pipeline_file(self.filepath)

            # Refresh from disk: stages completed by compute_ft()/estimate_noise()
            # (or by another interface) are written to the file, so the
            # in-memory tracker captured at open()/create() time is stale.
            _, self.source_metadata, self.stage_tracker = open_pipeline_file(
                self.filepath
            )

            info_dict = {
                "filepath": str(self.filepath),
                "valid": validation_report["valid"],
                "source_path": str(self.source_metadata.source_path),
                "format": self.source_metadata.format_name,
                "import_time": self.source_metadata.import_timestamp.isoformat(),
                "completed_stages": list(self.stage_tracker.completed_stages),
                "next_available_stages": self.stage_tracker.get_next_available_stages(),
            }

            if not validation_report["valid"]:
                info_dict["errors"] = validation_report["errors"]

            if validation_report.get("warnings"):
                info_dict["warnings"] = validation_report["warnings"]

            return info_dict

        except Exception as e:
            return {
                "filepath": str(self.filepath),
                "valid": False,
                "error": f"Failed to get info: {e}",
            }

    def validate(self) -> Dict[str, Any]:
        """
        Validate pipeline file integrity.

        Returns
        -------
        dict
            Detailed validation report

        Raises
        ------
        PipelineCorruptionError
            If file is corrupted and cannot be validated
        """
        return validate_pipeline_file(self.filepath)

    # =========================================================================
    # Parameter-scan surface
    # =========================================================================

    @staticmethod
    def scan_list(
        selector: Optional[str] = None,
        *,
        include_advanced: bool = False,
    ) -> Tuple["KnobSpec", ...]:
        """List the registered tunable knobs.

        Equivalent to :func:`ftmwpipeline.api.scan_list`. The returned
        :class:`KnobSpec` tuple is independent of any file, so this is a
        staticmethod; it is exposed on the class for dual-interface parity.
        ``selector`` filters by dotted-path prefix (e.g. ``"stage2b"`` /
        ``"stage2b.gaussian"``); ``include_advanced`` reveals advanced-tier
        knobs hidden from the default listing.
        """
        from ._internal.tuning import list_knobs

        return list_knobs(selector, include_advanced=include_advanced)

    def scan_run(
        self,
        knob: str,
        grid: Optional[Sequence[Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        reuse: bool = False,
        make_plot: bool = True,
        quiet: bool = False,
        zoom_regions: Optional[Sequence[Tuple[float, float]]] = None,
        n_zoom: Optional[int] = None,
        zoom_width_mhz: Optional[float] = None,
        fit_top_snr: int = 3,
        fit_sample: int = 20,
        fit_freqs: Optional[Sequence[float]] = None,
        fit_sample_seed: int = 0,
        fit_all: bool = False,
    ) -> "SweepResult":
        """Sweep a single knob across a grid on a copy of this file.

        Equivalent to :func:`ftmwpipeline.api.scan_run`. Re-runs the knob's
        stage for each grid value on a working copy (this file is never
        mutated), returning a :class:`SweepResult` with the table, CSV path,
        optional plot, recommendation, and how-to-apply instructions. A progress
        indicator is printed to stderr unless ``quiet=True``.

        ``zoom_regions`` pins explicit ``(lo_mhz, hi_mhz)`` windows for the
        region-based plot adapters (Stage 3 / Stage 4), overriding their
        divergence auto-selection; ``n_zoom`` / ``zoom_width_mhz`` instead tune
        how many regions to auto-select and how wide each is. The ``fit_*``
        controls bound a Stage 5 fit sweep to a window subset (the ``fit_top_snr``
        brightest + a seeded ``fit_sample`` sample + the windows nearest
        ``fit_freqs``); ``fit_all`` re-fits every window.
        """
        from ._internal.tuning import get_knob, run_scan

        return run_scan(
            get_knob(knob),
            self.filepath,
            grid=grid,
            output_dir=Path(output_dir) if output_dir is not None else None,
            reuse=reuse,
            make_plot=make_plot,
            quiet=quiet,
            zoom_regions=zoom_regions,
            n_zoom=n_zoom,
            zoom_width_mhz=zoom_width_mhz,
            fit_top_snr=fit_top_snr,
            fit_sample=fit_sample,
            fit_freqs=fit_freqs,
            fit_sample_seed=fit_sample_seed,
            fit_all=fit_all,
        )

    def scan_all(
        self,
        selector: Optional[str] = None,
        *,
        include_advanced: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        reuse: bool = False,
        make_plot: bool = True,
        quiet: bool = False,
        zoom_regions: Optional[Sequence[Tuple[float, float]]] = None,
        n_zoom: Optional[int] = None,
        zoom_width_mhz: Optional[float] = None,
        fit_top_snr: int = 3,
        fit_sample: int = 20,
        fit_freqs: Optional[Sequence[float]] = None,
        fit_sample_seed: int = 0,
        fit_all: bool = False,
    ) -> "List[BatchItem]":
        """Sweep every knob matched by ``selector`` on its default grid.

        Equivalent to :func:`ftmwpipeline.api.scan_all`. A convenience
        over :meth:`scan_run` for reviewing a whole stage / sub-block at once:
        ``selector`` filters by dotted-path prefix (e.g. ``"stage2b"`` /
        ``"stage2b.gaussian"``) exactly as :meth:`scan_list`, and
        ``include_advanced`` adds the advanced-tier knobs. Each knob runs on its
        own working copy (this file is never mutated); a knob whose scan fails
        (e.g. its required stage is absent) is recorded as a failed
        :class:`BatchItem` and the batch continues. The ``zoom_*`` controls apply
        the same explicit-regions / count-width steering to every knob's plot.
        """
        from ._internal.tuning import list_knobs, run_scan_batch

        specs = list_knobs(selector, include_advanced=include_advanced)
        return run_scan_batch(
            specs,
            self.filepath,
            output_dir=Path(output_dir) if output_dir is not None else None,
            reuse=reuse,
            make_plot=make_plot,
            quiet=quiet,
            zoom_regions=zoom_regions,
            n_zoom=n_zoom,
            zoom_width_mhz=zoom_width_mhz,
            fit_top_snr=fit_top_snr,
            fit_sample=fit_sample,
            fit_freqs=fit_freqs,
            fit_sample_seed=fit_sample_seed,
            fit_all=fit_all,
        )

    def settings_show(
        self,
        selector: Optional[str] = None,
        *,
        include_advanced: bool = False,
        preset: Optional[Union[str, Path]] = None,
    ) -> "Tuple[SettingRow, ...]":
        """Resolved value + provenance for each covered setting of this file.

        Equivalent to :func:`ftmwpipeline.api.settings_show`. Returns one
        :class:`SettingRow` per setting -- its ``path``, the resolved ``value``,
        the ``source`` layer that supplied it (``.ftmw`` / ``.yml:<name>`` /
        ``recommended`` / ``default``), the ``hard_default`` for reference, and
        the registry ``tier`` / ``help`` enrichment. ``selector`` filters by
        dotted-path prefix (e.g. ``"stage2b"`` / ``"stage2b.gaussian"``);
        ``include_advanced`` reveals advanced-tier rows; ``preset`` (a bare name
        or YAML path) populates the ``.yml`` provenance layer. Because a
        persisted ``.ftmw`` value outranks a preset, a named preset changes the
        resolved view only for fields the file has not persisted.
        """
        from ._internal.tuning import resolve_settings_view

        return resolve_settings_view(
            self.filepath,
            selector,
            include_advanced=include_advanced,
            preset=preset,
        )

    def settings_set(self, knob: str, value: str) -> "SetResult":
        """Persist a chosen value for ``knob`` into this file's persisted layer.

        Equivalent to :func:`ftmwpipeline.api.settings_set`. ``knob`` is a dotted
        settings path (``stage2.window_mhz`` / ``stage5.tau.max_decay_factor``);
        ``value`` is the string form, coerced to the field's type. Because the
        stored stage results were computed against the old value, the affected
        stage and all downstream stages are invalidated (their results dropped
        and completion cleared) so the file never carries results inconsistent
        with its settings. Stage 1 FT-shaping knobs (``zpf`` / ``expf_us`` /
        ``window_function``) are rejected -- set them via :meth:`compute_ft`.
        Returns a :class:`SetResult` with the invalidated stage names.
        """
        from ._internal.tuning import set_setting

        return set_setting(self.filepath, knob, value)

    def settings_export(
        self,
        out_path: Union[str, Path],
        selector: Optional[str] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ExportResult":
        """Write this file's chosen Stage 2--5 values to a ``.yml`` preset block.

        Equivalent to :func:`ftmwpipeline.api.settings_export`. Each stage's
        persisted settings are serialized into the matching ``stageN:`` block,
        filtered by the dotted ``selector``; the result loads back through the
        stages' ``--preset`` path. Stage 1 is excluded (presets do not carry FT
        settings). Returns an :class:`ExportResult` with the written path and the
        exported setting paths.
        """
        from ._internal.tuning import export_settings

        return export_settings(
            self.filepath,
            out_path,
            selector,
            name=name,
            description=description,
        )

    def __repr__(self) -> str:
        """String representation of Pipeline instance."""
        return (
            f"Pipeline(file={self.filepath.name}, "
            f"source={self.source_metadata.source_path.name}, "
            f"stages={len(self.stage_tracker.completed_stages)})"
        )
