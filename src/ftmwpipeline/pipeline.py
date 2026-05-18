"""Object-oriented Pipeline class for FTMW spectroscopy data processing.

This module provides the high-level object-oriented interface for the FTMW
pipeline, implementing the dual-interface architecture alongside functional
and CLI interfaces. Each Pipeline instance is bound to a specific .ftmw file.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path

from .core.data_structures import FID, ComplexFT
from .preprocessing.noise_estimation import NoiseResult
from .file_manager import (
    SourceMetadata, PipelineStageTracker,
    PipelineFileError, PipelineExistsError, StageDependencyError, PipelineCorruptionError,
    create_pipeline_file, open_pipeline_file, validate_pipeline_file, update_processing_parameters
)
from .io.data_loaders import load_fid, detect_format, validate_source
from ._internal.stage0_impl import import_data_impl, load_fid_from_pipeline_impl
from ._internal.stage1_impl import compute_ft_impl, visualize_ft_impl, save_ft_parameters_impl
from ._internal.stage2_impl import (
    compute_noise_estimation_impl, visualize_noise_impl
)


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
    
    def compute_ft(self, zpf: Optional[int] = None, expf_us: Optional[float] = None, 
                   trim: Optional[Tuple[float, float]] = None, start_us: Optional[float] = None,
                   end_us: Optional[float] = None, window_function: Optional[str] = None,
                   units_power: Optional[int] = None, from_saved_params: bool = False) -> ComplexFT:
        """
        Compute Fourier Transform with specified processing parameters.
        
        This method implements Stage 1 FT processing, equivalent to the CLI
        compute-ft command. Can be called multiple times safely.
        
        Parameters
        ----------
        zpf : int, optional
            Zero padding factor. If None, uses cached default or 1.
        expf_us : float, optional
            Exponential filter in microseconds. If None, uses cached default or 5.0.
        trim : tuple of float, optional
            (min_freq, max_freq) in MHz to trim spectrum
        start_us : float, optional
            FID window start time in microseconds
        end_us : float, optional
            FID window end time in microseconds  
        window_function : str, optional
            Windowing function name
        units_power : int, optional
            Scaling factor as power of 10. If None, uses cached default or 6.
        from_saved_params : bool, default False
            If True, use previously saved parameters and ignore other arguments
            
        Returns
        -------
        ComplexFT
            Computed frequency domain data
            
        Raises
        ------
        StageDependencyError
            If required dependencies (FID data) are not available
        RuntimeError
            If FT computation fails
        """
        try:
            # Use shared implementation for FT computation (handles dependency checking)
            result = compute_ft_impl(
                file_path=str(self.filepath),
                start_us=start_us if not from_saved_params else None,
                end_us=end_us if not from_saved_params else None,
                zpf=zpf if not from_saved_params else None,
                expf_us=expf_us if not from_saved_params else None,
                window_function=window_function if not from_saved_params else None,
                units_power=units_power if not from_saved_params else None,
                trim_range=trim,
                validate_only=False
            )
            
            complex_ft = result['complex_ft']
            
            # Storage and stage completion handled by shared implementation
            self.logger.info(f"FT computed: {complex_ft.n_points:,} frequency points")
            if trim:
                self.logger.info(f"Trimmed to {trim[0]:.1f}-{trim[1]:.1f} MHz")
                
            return complex_ft
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute FT: {e}") from e
    
    def visualize_ft(self, zpf: Optional[int] = None, expf_us: Optional[float] = None,
                     trim: Optional[Tuple[float, float]] = None, start_us: Optional[float] = None,
                     end_us: Optional[float] = None, window_function: Optional[str] = None,
                     units_power: Optional[int] = None, save_params: bool = False,
                     backend: str = 'matplotlib', interactive: bool = True, 
                     output_file: Optional[Union[str, Path]] = None,
                     show_fid_panels: bool = True):
        """
        Create enhanced FT visualization with processing workflow display.
        
        This method implements enhanced FT visualization equivalent to the CLI 
        visualize-ft command, showing complete FID-to-spectrum processing workflow.
        
        Parameters
        ----------
        zpf : int, optional
            Zero padding factor. If None, uses cached default or 1.
        expf_us : float, optional
            Exponential filter in microseconds. If None, uses cached default or 5.0.
        trim : tuple of float, optional
            (min_freq, max_freq) in MHz to trim spectrum
        start_us : float, optional
            FID window start time in microseconds
        end_us : float, optional
            FID window end time in microseconds
        window_function : str, optional
            Windowing function name
        units_power : int, optional
            Scaling factor as power of 10. If None, uses cached default or 6.
        save_params : bool, default False
            Whether to save parameters as defaults for this experiment
        backend : str, default 'matplotlib'
            Plotting backend ('matplotlib' or 'plotly')
        interactive : bool, default True
            Whether to show interactive plot
        output_file : str or Path, optional
            Path to save plot image (for non-interactive mode)
        show_fid_panels : bool, default True
            Whether to show FID processing panels
            
        Returns
        -------
        figure
            Matplotlib or Plotly figure object
            
        Raises
        ------
        StageDependencyError
            If required dependencies are not available
        RuntimeError
            If visualization fails
        """
        try:
            # Use shared implementation for FT visualization
            fig = visualize_ft_impl(
                file_path=str(self.filepath),
                start_us=start_us,
                end_us=end_us,
                zpf=zpf,
                expf_us=expf_us,
                window_function=window_function,
                units_power=units_power,
                trim_range=trim,
                title=None,  # Let implementation generate title
                show_fid_panels=show_fid_panels,
                backend=backend,
                interactive=interactive
            )
            
            # Handle output
            if not interactive and output_file:
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                self.logger.info(f"Plot saved to: {output_file}")
            elif not interactive:
                # Save with default name
                default_name = f"{self.filepath.stem}_enhanced_spectrum.png"
                fig.savefig(default_name, dpi=150, bbox_inches='tight')
                self.logger.info(f"Plot saved to: {default_name}")
            elif interactive and backend == 'matplotlib':
                import matplotlib.pyplot as plt
                plt.show()
            
            # Save parameters if requested
            if save_params:
                # Collect parameters for saving
                params = {
                    'start_us': start_us,
                    'end_us': end_us,
                    'zpf': zpf,
                    'expf_us': expf_us,
                    'window_function': window_function,
                    'units_power': units_power,
                    'trim_min_mhz': trim[0] if trim else None,
                    'trim_max_mhz': trim[1] if trim else None
                }
                # Filter out None values
                params = {k: v for k, v in params.items() if v is not None}
                
                if params:
                    save_ft_parameters_impl(str(self.filepath), params)
                    self.logger.info(f"Saved {len(params)} processing parameters")
                else:
                    self.logger.info("No custom parameters to save")
            
            self.logger.info("FT visualization completed")
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to create FT visualization: {e}") from e
    
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