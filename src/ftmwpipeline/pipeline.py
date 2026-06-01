"""Object-oriented Pipeline class for FTMW spectroscopy data processing.

This module provides the high-level object-oriented interface for the FTMW
pipeline, implementing the dual-interface architecture alongside functional
and CLI interfaces. Each Pipeline instance is bound to a specific .ftmw file.
"""

from typing import Dict, List, Optional, Sequence, Union, Any, Tuple, cast, TYPE_CHECKING
import logging
from pathlib import Path

if TYPE_CHECKING:
    from ._internal.tuning import KnobSpec, SweepResult

from .core.data_structures import FID, ComplexFT
from .core.settings import FTSettings
from .core.noise_settings import NoiseSettings
from .core.peak_detection_settings import PeakDetectionSettings
from .core.stage_fit_settings import StageFitSettings
from .core.tau_calibration_settings import TauCalibrationSettings
from .core.window_planning_settings import WindowPlanningSettings
from .core.start_detection_settings import StartDetectionSettings
from .preprocessing.noise_estimation import NoiseResult
from .preprocessing.start_detection import StartDetectionResult
from .file_manager import (
    SourceMetadata, PipelineStageTracker,
    PipelineFileError, PipelineExistsError, StageDependencyError, PipelineCorruptionError,
    create_pipeline_file, open_pipeline_file, validate_pipeline_file, update_processing_parameters
)
from .io.data_loaders import load_fid, detect_format, validate_source
from ._internal.stage0_impl import import_data_impl, load_fid_from_pipeline_impl
from ._internal.stage1_impl import (
    compute_ft_impl, visualize_ft_impl, save_ft_parameters_impl
)
from ._internal.stage2_impl import (
    compute_noise_estimation_impl, visualize_noise_impl
)
from ._internal.stage2b_impl import (
    calibrate_tau_impl, load_tau_calibration_impl,
)
from ._internal.stage2b_g_impl import (
    calibrate_tau_G_impl, load_tau_G_calibration_impl,
)
from ._internal.shape_recommendation_impl import recommend_shape_impl
from ._internal.start_detection_impl import detect_start_time_impl
from ._internal.stage3_impl import (
    detect_peaks_impl, visualize_peaks_impl, load_peaks_impl
)
from ._internal.stage4_impl import (
    assign_windows_impl, visualize_windows_impl, load_windows_impl
)
from ._internal.stage5_impl import (
    fit_peaks_impl, visualize_fit_impl, load_fit_impl
)
from ._internal.stage5_validation_impl import validate_stage5_shape_error_impl
from .core.data_structures import Peak, SpectrumFit, WindowPlan
from .fitting.tau_calibration import ShapeRecommendation, TauCalibrationResult


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
    
    # Stage 1: FT Processing
    pipe.compute_ft(zpf=2, expf_us=5.0, trim=(26500, 40000))
    pipe.visualize_ft(zpf=1, expf_us=3.0, save_params=True)
    
    # File info and validation
    pipe.info()       # Show pipeline status and metadata
    pipe.validate()   # Check file integrity
    ```
    """
    
    def __init__(self, filepath: Union[str, Path],
                 source_metadata: Optional[SourceMetadata] = None,
                 stage_tracker: Optional[PipelineStageTracker] = None):
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
    def create(cls, filepath: Union[str, Path], source: Union[str, Path], 
               format_name: Optional[str] = None, fid_index: Optional[int] = None,
               force: bool = False, **loader_params) -> 'Pipeline':
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
        if not validation['valid']:
            errors = "; ".join(validation['errors'])
            raise ValueError(f"Source validation failed: {errors}")
        
        # Prepare loader parameters
        if format_name == 'blackchirp' and fid_index is not None:
            loader_params['fid_index'] = fid_index
        
        # Load FID data
        try:
            fid = load_fid(source_path, format_name, **loader_params)
        except Exception as e:
            raise RuntimeError(f"Failed to load FID data: {e}") from e
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_path=source_path,
            format_name=format_name,
            loader_parameters=loader_params
        )
        
        # Create pipeline file
        created_filepath = create_pipeline_file(
            filepath, fid, source_metadata, force=force
        )
        
        # Load file info for Pipeline instance
        filepath, source_metadata, stage_tracker = open_pipeline_file(created_filepath)
        
        return cls(filepath, source_metadata, stage_tracker)
    
    @classmethod
    def open(cls, filepath: Union[str, Path]) -> 'Pipeline':
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
            self.logger.info(f"Loaded FID data: {fid.n_points:,} points, {fid.duration_us:.1f} μs")
            return fid
        except Exception as e:
            raise RuntimeError(f"Failed to load FID data: {e}") from e
    
    def compute_ft(
        self,
        zpf: Optional[int] = None,
        expf_us: Optional[float] = None,
        trim: Optional[Tuple[float, float]] = None,
        start_us: Optional[float] = None,
        end_us: Optional[float] = None,
        window_function: Optional[str] = None,
        units_power: Optional[int] = None,
        from_saved_params: bool = False,
    ) -> ComplexFT:
        """Compute Fourier Transform (Stage 1, user-driven).

        Resolves settings through ``explicit > persisted > recommended`` and
        persists the resolved canonical settings to the ``.ftmw`` file.  Can be
        called multiple times safely.

        Parameters
        ----------
        zpf : int, optional
            Zero-padding factor.
        expf_us : float, optional
            Exponential apodization time constant in microseconds.
            ``None`` (or any non-positive value) disables apodization;
            there is no implicit fallback default.
        trim : tuple of float, optional
            ``(min_mhz, max_mhz)`` frequency analysis range to keep.
        start_us : float, optional
            FID window start time in microseconds.
        end_us : float, optional
            FID window end time in microseconds.
        window_function : str, optional
            Window function name (hann, blackman, …).
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
                    zpf=zpf,
                    expf_us=expf_us,
                    window_function=window_function,
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
            self.logger.info(
                f"FT computed: {complex_ft.n_points:,} frequency points"
            )
            if trim:
                self.logger.info(
                    f"Trimmed to {trim[0]:.1f}-{trim[1]:.1f} MHz"
                )
            return complex_ft
        except Exception as e:
            raise RuntimeError(f"Failed to compute FT: {e}") from e
    
    def visualize_ft(
        self,
        zpf: Optional[int] = None,
        expf_us: Optional[float] = None,
        trim: Optional[Tuple[float, float]] = None,
        start_us: Optional[float] = None,
        end_us: Optional[float] = None,
        window_function: Optional[str] = None,
        units_power: Optional[int] = None,
        save_params: bool = False,
        backend: str = "matplotlib",
        interactive: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_fid_panels: bool = True,
    ) -> Any:
        """Create enhanced FT visualization with processing workflow display.

        Equivalent to the CLI ``visualize-ft`` command.  Never persists
        settings; exploration only.  Pass ``save_params=True`` to write the
        explicitly provided kwargs to the canonical ``ft_processing`` record.

        Parameters
        ----------
        zpf : int, optional
            Zero-padding factor.
        expf_us : float, optional
            Exponential apodization time constant in microseconds.
            ``None`` (or any non-positive value) disables apodization;
            there is no implicit fallback default.
        trim : tuple of float, optional
            ``(min_mhz, max_mhz)`` frequency analysis range.
        start_us : float, optional
            FID window start time in microseconds.
        end_us : float, optional
            FID window end time in microseconds.
        window_function : str, optional
            Window function name.
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
                zpf=zpf,
                expf_us=expf_us,
                window_function=window_function,
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
                    ("zpf", zpf),
                    ("expf_us", expf_us),
                    ("window_function", window_function),
                    ("units_power", units_power),
                ):
                    if value is not None:
                        params[key] = value
                if trim is not None:
                    params["trim_min_mhz"] = trim[0]
                    params["trim_max_mhz"] = trim[1]
                if params:
                    save_ft_parameters_impl(str(self.filepath), params)
                    self.logger.info(
                        f"Saved {len(params)} processing parameters"
                    )
                else:
                    self.logger.info("No custom parameters to save")

            self.logger.info("FT visualization completed")
            return fig

        except Exception as e:
            raise RuntimeError(
                f"Failed to create FT visualization: {e}"
            ) from e
    
    def estimate_noise(self, skew_target: Optional[float] = None,
                       min_bin_fraction: Optional[float] = None,
                       smoothing_window_mhz: Optional[float] = None,
                       min_noise_fraction: Optional[float] = None,
                       from_saved_params: bool = False,
                       *,
                       method: str = "scatter",
                       window_mhz: Optional[float] = None,
                       pedestal_mhz: Optional[float] = None,
                       line_k: Optional[float] = None,
                       n_iter: Optional[int] = None,
                       region_aware: bool = True,
                       smoothing_mhz: Optional[float] = None,
                       smoothing_percentile: Optional[float] = None,
                       convolve_mhz: Optional[float] = None,
                       settings: Optional[NoiseSettings] = None,
                       preset: Optional[str] = None) -> NoiseResult:
        """
        Estimate frequency-dependent noise using adaptive binning.
        
        This method implements Stage 2 noise estimation, equivalent to the CLI
        estimate-noise command. Requires Stage 1 (FT computation) to be completed.
        
        Parameters
        ----------
        skew_target : float, optional
            Target skewness for noise identification (default: 0.631 for Rayleigh)
        min_bin_fraction : float, optional
            Minimum bin size as fraction of total data (default: 1/64)
        smoothing_window_mhz : float, optional
            RMS smoothing window size in MHz (default: auto-calculated)
        min_noise_fraction : float, optional
            Minimum fraction of points that must be noise per bin (default: 2/3)
        from_saved_params : bool, default False
            If True, ignore provided parameters and use saved parameters only
        method : str, default "adaptive"
            Noise estimator to run. ``"adaptive"`` is the level-based binning
            estimator (skew_target / min_bin_fraction / smoothing_window_mhz /
            min_noise_fraction / settings / preset apply). ``"scatter"`` is the
            high-pass, region-aware estimator that is immune to the leakage
            pedestal on high-SNR, line-dense spectra; it uses its own knobs
            (window_mhz, pedestal_mhz, line_k, n_iter, region_aware).
        window_mhz, pedestal_mhz, line_k, n_iter, region_aware, smoothing_mhz,
        smoothing_percentile, convolve_mhz
            Scatter-estimator knobs (``method="scatter"`` only); each defaults to
            the module-level constant when left unset. ``smoothing_mhz`` /
            ``smoothing_percentile`` control the broad lower-envelope median σ
            smoothing that rides the noise floor through line-dense bands
            (``smoothing_mhz=0`` disables it); ``convolve_mhz`` is the Gaussian σ
            of the second pass that removes the median's staircase.

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
                skew_target=skew_target,
                min_bin_fraction=min_bin_fraction,
                smoothing_window_mhz=smoothing_window_mhz,
                min_noise_fraction=min_noise_fraction,
                from_saved_params=from_saved_params,
                method=method,
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
            return result['noise_result']
            
        except StageDependencyError:
            # Re-raise dependency errors with clear message
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to estimate noise: {e}") from e
    
    def visualize_noise(self, y_max_factor: Optional[float] = None,
                        figsize: Optional[tuple] = None, title: Optional[str] = None,
                        show_bin_boundaries: Optional[bool] = None,
                        show_noise_points: Optional[bool] = None,
                        save_params: bool = False, backend: str = 'matplotlib',
                        interactive: bool = True, output_file: Optional[Union[str, Path]] = None,
                        **plot_kwargs):
        """
        Create noise estimation diagnostic visualization.
        
        This method creates diagnostic plots showing spectrum, noise points,
        adaptive bin boundaries, and RMS noise estimates. Equivalent to the CLI
        visualize-noise command.
        
        Parameters
        ----------
        y_max_factor : float, optional
            Y-axis maximum as multiple of median RMS noise (default: 20.0)
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (16, 6))
        title : str, optional
            Custom title for the plot
        show_bin_boundaries : bool, optional
            Whether to show adaptive bin boundaries (default: True)
        show_noise_points : bool, optional
            Whether to highlight noise points (default: True)
        save_params : bool, default False
            Whether to save custom parameters for future use
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
            # Create visualization using shared implementation (handles dependency checking and parameter saving)
            fig = visualize_noise_impl(
                file_path=str(self.filepath),
                y_max_factor=y_max_factor,
                figsize=figsize,
                title=title,
                show_bin_boundaries=show_bin_boundaries,
                show_noise_points=show_noise_points,
                backend=backend,
                interactive=interactive,
                save_params=save_params,
                **plot_kwargs
            )
            
            # Save output file if requested
            if output_file:
                try:
                    if backend == 'plotly':
                        fig.write_html(str(output_file))
                    else:
                        fig.savefig(str(output_file), dpi=300, bbox_inches='tight')
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
        (``explicit > preset > persisted > recommended > hard default``);
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
                tc.tau_maj_us, tc.sigma_tau_us, tc.n_contributors,
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
                tc.tau_maj_us, tc.sigma_tau_us, tc.n_contributors,
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
        knee_strength_min: Optional[float] = None,
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
        ``detect-start`` command and ``ftmwpipeline.api.detect_start_time``.

        Parameters
        ----------
        sweep_max_us, step_us, guard_margin_us, floor_factor, knee_strength_min :
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
            The recommendation plus diagnostics (chirp-end, knee, sweep arrays).
        """
        resolved = self._resolve_start_detection_settings(
            settings,
            sweep_max_us=sweep_max_us,
            step_us=step_us,
            guard_margin_us=guard_margin_us,
            floor_factor=floor_factor,
            knee_strength_min=knee_strength_min,
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
        knee_strength_min: Optional[float],
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
        if knee_strength_min is not None:
            overrides["knee_strength_min"] = float(knee_strength_min)
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
        ``detect-peaks`` command and ``ftmwpipeline.api.detect_peaks``.

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

        Equivalent to the CLI ``visualize-peaks`` command. Interactive
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
            raise RuntimeError(
                f"Failed to create peak visualization: {e}"
            ) from e

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
        ``assign-windows`` command and ``ftmwpipeline.api.assign_windows``.

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

        Equivalent to the CLI ``visualize-windows`` command. Shows each fit
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
            raise RuntimeError(
                f"Failed to create window visualization: {e}"
            ) from e

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
        to the CLI ``fit-peaks`` command and ``ftmwpipeline.api.fit_peaks``.

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
        Equivalent to the CLI ``validate-stage5-shape-error`` command. Requires
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

        Equivalent to the CLI ``visualize-fit`` command. With ``window_id``
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
            raise RuntimeError(
                f"Failed to create fit visualization: {e}"
            ) from e

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
            _, self.source_metadata, self.stage_tracker = open_pipeline_file(self.filepath)

            info_dict = {
                'filepath': str(self.filepath),
                'valid': validation_report['valid'],
                'source_path': str(self.source_metadata.source_path),
                'format': self.source_metadata.format_name,
                'import_time': self.source_metadata.import_timestamp.isoformat(),
                'completed_stages': list(self.stage_tracker.completed_stages),
                'next_available_stages': self.stage_tracker.get_next_available_stages()
            }
            
            if not validation_report['valid']:
                info_dict['errors'] = validation_report['errors']
                
            if validation_report.get('warnings'):
                info_dict['warnings'] = validation_report['warnings']
            
            return info_dict
            
        except Exception as e:
            return {
                'filepath': str(self.filepath),
                'valid': False,
                'error': f"Failed to get info: {e}"
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
    # Companion parameter tuning
    # =========================================================================

    @staticmethod
    def tune_list(stage: Optional[str] = None) -> Tuple["KnobSpec", ...]:
        """List the registered tunable knobs (optionally filtered to a stage).

        Equivalent to :func:`ftmwpipeline.api.tune_list`. The returned
        :class:`KnobSpec` tuple is independent of any file, so this is a
        staticmethod; it is exposed on the class for dual-interface parity.
        """
        from ._internal.tuning import list_knobs

        return list_knobs(stage)

    def tune_scan(
        self,
        knob: str,
        grid: Optional[Sequence[Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        reuse: bool = False,
        make_plot: bool = True,
        quiet: bool = False,
    ) -> "SweepResult":
        """Sweep a single knob across a grid on a copy of this file.

        Equivalent to :func:`ftmwpipeline.api.tune_scan`. Re-runs the knob's
        stage for each grid value on a working copy (this file is never
        mutated), returning a :class:`SweepResult` with the table, CSV path,
        optional plot, recommendation, and how-to-apply instructions. A progress
        indicator is printed to stderr unless ``quiet=True``.
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
        )

    def __repr__(self) -> str:
        """String representation of Pipeline instance."""
        return (f"Pipeline(file={self.filepath.name}, "
                f"source={self.source_metadata.source_path.name}, "
                f"stages={len(self.stage_tracker.completed_stages)})")