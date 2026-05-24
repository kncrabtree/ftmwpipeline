"""Object-oriented Pipeline class for FTMW spectroscopy data processing.

This module provides the high-level object-oriented interface for the FTMW
pipeline, implementing the dual-interface architecture alongside functional
and CLI interfaces. Each Pipeline instance is bound to a specific .ftmw file.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, cast
import logging
from pathlib import Path

from .core.data_structures import FID, ComplexFT
from .core.settings import FTSettings
from .preprocessing.noise_estimation import NoiseResult
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
from ._internal.stage3_impl import (
    detect_peaks_impl, visualize_peaks_impl, load_peaks_impl
)
from ._internal.stage4_impl import (
    assign_windows_impl, visualize_windows_impl, load_windows_impl
)
from ._internal.stage5_impl import (
    fit_peaks_impl, visualize_fit_impl, load_fit_impl
)
from .core.data_structures import Peak, SpectrumFit, WindowPlan


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
            Exponential filter time constant in microseconds.
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
            Exponential filter time constant in microseconds.
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
                       from_saved_params: bool = False) -> NoiseResult:
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
                from_saved_params=from_saved_params
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
        rescue_coherence_cluster_fwhm: Optional[float] = None,
        rescue_coherence_isolated_fwhm: Optional[float] = None,
        rescue_coherence_close_threshold: Optional[float] = None,
        rescue_coherence_isolated_threshold: Optional[float] = None,
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
        rescue_snr_threshold, rescue_prominence_threshold,
        rescue_coherence_cluster_fwhm, rescue_coherence_isolated_fwhm,
        rescue_coherence_close_threshold,
        rescue_coherence_isolated_threshold : optional
            Detector and phase-coherence tuning knobs for the rescue --
            see :func:`fit_peaks_impl` for defaults. The coherence knobs
            parameterise the sliding-threshold scheme (close-to-neighbour
            ratio ramping up to isolated ratio across the cluster→isolated
            FWHM band). All ignored when
            ``max_residual_rescue_rounds`` is 0.

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
                rescue_coherence_cluster_fwhm=rescue_coherence_cluster_fwhm,
                rescue_coherence_isolated_fwhm=rescue_coherence_isolated_fwhm,
                rescue_coherence_close_threshold=rescue_coherence_close_threshold,
                rescue_coherence_isolated_threshold=rescue_coherence_isolated_threshold,
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
    
    def __repr__(self) -> str:
        """String representation of Pipeline instance."""
        return (f"Pipeline(file={self.filepath.name}, "
                f"source={self.source_metadata.source_path.name}, "
                f"stages={len(self.stage_tracker.completed_stages)})")