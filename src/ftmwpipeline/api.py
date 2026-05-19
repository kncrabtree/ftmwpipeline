"""
Functional API for FTMW Pipeline - Stateless file-based operations.

This module provides a functional, stateless API for FTMW spectroscopy data 
processing as an alternative to the object-oriented Pipeline class. All functions
operate on .ftmw file paths and delegate to the Pipeline class internally to
ensure identical behavior and avoid code duplication.

Key Features:
- File-centric design: All functions take .ftmw file paths as first argument
- Stateless: Each function call is independent, no shared state
- Consistent: Same behavior as Pipeline class methods
- Efficient: Leverages existing tested implementations

Example Usage:
```python
import ftmwpipeline.api as ftmw

# Create pipeline from data
ftmw.import_data("experiment.ftmw", source="examples/blackchirp_data/2638/")

# Load and process data  
fid = ftmw.load_fid("experiment.ftmw")
complex_ft = ftmw.compute_ft("experiment.ftmw", zpf=2, expf_us=5.0, trim=(26500, 40000))

# Visualization and parameter management
ftmw.visualize_ft("experiment.ftmw", zpf=2, expf_us=5.0, save_params=True)
ftmw.save_ft_parameters("experiment.ftmw", {'zpf': 2, 'expf_us': 5.0})

# File management
info = ftmw.get_pipeline_info("experiment.ftmw")
stages = ftmw.list_available_stages("experiment.ftmw")
```
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging

from .pipeline import Pipeline
from .core.data_structures import FID, ComplexFT, Peak
from .preprocessing.noise_estimation import NoiseResult

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# File Management Functions
# =============================================================================

def import_data(file_path: Union[str, Path], source: Union[str, Path], 
                format_name: Optional[str] = None, fid_index: Optional[int] = None,
                force: bool = False, **loader_params) -> Dict[str, Any]:
    """
    Create new pipeline from raw experimental data.
    
    This function creates a new .ftmw pipeline file from experimental data,
    equivalent to Pipeline.create(). It handles format detection, data loading,
    and source metadata tracking.
    
    Parameters
    ----------
    file_path : str or Path
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
    dict
        Import result with pipeline file path, source info, and FID metadata
        
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
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> result = ftmw.import_data("exp_2638.ftmw", 
    ...                           source="examples/blackchirp_data/2638/")
    >>> print(f"Created: {result['pipeline_file']}")
    """
    try:
        # Create pipeline using Pipeline class
        pipeline = Pipeline.create(
            filepath=file_path,
            source=source,
            format_name=format_name,
            fid_index=fid_index,
            force=force,
            **loader_params
        )
        
        # Get pipeline info to return
        info = pipeline.info()
        
        # Return result consistent with _internal implementation
        result = {
            'pipeline_file': str(pipeline.filepath),
            'source_path': info['source_path'],
            'format_name': info['format'],
            'status': 'success'
        }
        
        # Add FID metadata if available
        try:
            fid = pipeline.load_data()
            result['fid_metadata'] = {
                'n_points': fid.n_points,
                'duration_us': fid.duration_us,
                'probe_freq_mhz': fid.probe_freq_mhz,
                'sideband': fid.sideband.value,
                'shots': fid.shots,
                'spacing': fid.spacing
            }
        except Exception as e:
            logger.warning(f"Could not load FID metadata: {e}")
        
        logger.info(f"Pipeline created successfully: {pipeline.filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to import data: {e}")
        raise


def load_fid(file_path: Union[str, Path]) -> FID:
    """
    Load FID data from pipeline file.
    
    This function loads the raw FID data stored in a .ftmw pipeline file,
    equivalent to Pipeline.load_data().
    
    Parameters
    ----------
    file_path : str or Path
        Path to existing .ftmw pipeline file
        
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
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> fid = ftmw.load_fid("experiment.ftmw")
    >>> print(f"FID: {fid.n_points:,} points, {fid.duration_us:.1f} μs")
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.load_data()
    except Exception as e:
        logger.error(f"Failed to load FID from {file_path}: {e}")
        raise


def validate_pipeline(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate pipeline file integrity.
    
    This function performs comprehensive validation of a .ftmw pipeline file,
    equivalent to Pipeline.validate().
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file to validate
        
    Returns
    -------
    dict
        Validation report with status and any issues found
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> report = ftmw.validate_pipeline("experiment.ftmw")
    >>> if report['valid']:
    ...     print("Pipeline file is valid")
    >>> else:
    ...     print(f"Issues found: {report['errors']}")
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.validate()
    except Exception as e:
        logger.error(f"Failed to validate {file_path}: {e}")
        return {
            'valid': False,
            'errors': [f"Failed to validate pipeline file: {e}"],
            'warnings': []
        }


# =============================================================================
# Stage 1 FT Processing Functions  
# =============================================================================

def compute_ft(file_path: Union[str, Path], zpf: Optional[int] = None, 
               expf_us: Optional[float] = None, trim: Optional[Tuple[float, float]] = None,
               start_us: Optional[float] = None, end_us: Optional[float] = None,
               window_function: Optional[str] = None, units_power: Optional[int] = None,
               from_saved_params: bool = False) -> ComplexFT:
    """
    Compute Fourier Transform with specified processing parameters.
    
    This function performs FT computation on FID data stored in a .ftmw pipeline
    file, equivalent to Pipeline.compute_ft(). Can be called multiple times safely.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing FID data
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
        If ``True``, ignore the explicit kwargs and use only the persisted /
        recommended settings (no explicit overrides).

    Returns
    -------
    ComplexFT
        Computed frequency domain data.

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist.
    StageDependencyError
        If required dependencies (FID data) are not available.
    RuntimeError
        If FT computation fails.

    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> # Compute with specific parameters (persisted as canonical)
    >>> complex_ft = ftmw.compute_ft("experiment.ftmw", zpf=2, expf_us=5.0,
    ...                              trim=(26500, 40000))
    >>>
    >>> # Use saved/recommended settings only
    >>> complex_ft = ftmw.compute_ft("experiment.ftmw", from_saved_params=True)
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.compute_ft(
            zpf=zpf,
            expf_us=expf_us,
            trim=trim,
            start_us=start_us,
            end_us=end_us,
            window_function=window_function,
            units_power=units_power,
            from_saved_params=from_saved_params
        )
    except Exception as e:
        logger.error(f"Failed to compute FT for {file_path}: {e}")
        raise


def visualize_ft(file_path: Union[str, Path], zpf: Optional[int] = None,
                 expf_us: Optional[float] = None, trim: Optional[Tuple[float, float]] = None,
                 start_us: Optional[float] = None, end_us: Optional[float] = None,
                 window_function: Optional[str] = None, units_power: Optional[int] = None,
                 save_params: bool = False, backend: str = 'matplotlib',
                 interactive: bool = True, output_file: Optional[Union[str, Path]] = None,
                 show_fid_panels: bool = True):
    """
    Create enhanced FT visualization with processing workflow display.
    
    This function creates comprehensive FT visualization showing the complete
    FID-to-spectrum processing workflow, equivalent to Pipeline.visualize_ft().
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
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
    FileNotFoundError
        If pipeline file does not exist
    StageDependencyError
        If required dependencies are not available
    RuntimeError
        If visualization fails
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> # Create interactive visualization with custom parameters
    >>> fig = ftmw.visualize_ft("experiment.ftmw", zpf=2, expf_us=5.0,
    ...                         save_params=True)
    >>> 
    >>> # Save plot to file
    >>> fig = ftmw.visualize_ft("experiment.ftmw", interactive=False,
    ...                         output_file="spectrum.png")
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_ft(
            zpf=zpf,
            expf_us=expf_us,
            trim=trim,
            start_us=start_us,
            end_us=end_us,
            window_function=window_function,
            units_power=units_power,
            save_params=save_params,
            backend=backend,
            interactive=interactive,
            output_file=output_file,
            show_fid_panels=show_fid_panels
        )
    except Exception as e:
        logger.error(f"Failed to visualize FT for {file_path}: {e}")
        raise


def save_ft_parameters(file_path: Union[str, Path], 
                       parameters: Dict[str, Any]) -> None:
    """
    Save FT processing parameters as defaults for pipeline file.
    
    This function saves processing parameters to the .ftmw pipeline file
    for use in subsequent computations with from_saved_params=True.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
    parameters : dict
        Processing parameters to save. Valid keys include:
        - 'zpf': Zero padding factor
        - 'expf_us': Exponential filter time constant
        - 'start_us', 'end_us': FID time window
        - 'window_function': Windowing function name
        - 'units_power': Scaling factor
        - 'trim_min_mhz', 'trim_max_mhz': Frequency trimming range
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    RuntimeError
        If parameter saving fails
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> params = {
    ...     'zpf': 2,
    ...     'expf_us': 5.0,
    ...     'trim_min_mhz': 26500,
    ...     'trim_max_mhz': 40000
    ... }
    >>> ftmw.save_ft_parameters("experiment.ftmw", params)
    """
    try:
        # Use internal implementation for parameter saving
        from ._internal.stage1_impl import save_ft_parameters_impl
        save_ft_parameters_impl(str(file_path), parameters)
        logger.info(f"Saved {len(parameters)} FT parameters to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save FT parameters to {file_path}: {e}")
        raise


# =============================================================================
# Stage 2: Noise Estimation Functions
# =============================================================================

def estimate_noise(file_path: Union[str, Path], skew_target: Optional[float] = None,
                   min_bin_fraction: Optional[float] = None,
                   smoothing_window_mhz: Optional[float] = None,
                   min_noise_fraction: Optional[float] = None,
                   from_saved_params: bool = False) -> NoiseResult:
    """
    Estimate frequency-dependent noise using adaptive binning.
    
    This function performs noise estimation on ComplexFT data stored in a .ftmw
    pipeline file, equivalent to Pipeline.estimate_noise(). Requires Stage 1 
    (FT computation) to be completed first.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing ComplexFT data
    skew_target : float, optional
        Target skewness for noise identification (default: 0.631 for Rayleigh)
    min_bin_fraction : float, optional
        Minimum bin size as fraction of total data (default: 1/64)
    smoothing_window_mhz : float, optional
        RMS smoothing window size in MHz (default: auto-calculated)
    min_noise_fraction : float, optional
        Minimum fraction of points that must be noise per bin (default: 2/3)
    from_saved_params : bool, default False
        If True, use saved parameters and ignore provided parameters
        
    Returns
    -------
    NoiseResult
        Container with RMS noise estimate, noise mask, and diagnostics
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If Stage 1 dependencies are not met or parameters are invalid
    RuntimeError
        If noise estimation fails
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> # First compute FT if not already done
    >>> ftmw.compute_ft("experiment.ftmw", zpf=2, trim=(26500, 40000))
    >>> # Estimate noise with default parameters
    >>> noise_result = ftmw.estimate_noise("experiment.ftmw")
    >>> # Use custom parameters
    >>> noise_result = ftmw.estimate_noise("experiment.ftmw", 
    ...                                     skew_target=0.7, 
    ...                                     min_bin_fraction=1/32)
    """
    try:
        # Delegate to Pipeline class for consistent behavior
        pipeline = Pipeline.open(file_path)
        return pipeline.estimate_noise(
            skew_target=skew_target,
            min_bin_fraction=min_bin_fraction,
            smoothing_window_mhz=smoothing_window_mhz,
            min_noise_fraction=min_noise_fraction,
            from_saved_params=from_saved_params
        )
        
    except Exception as e:
        logger.error(f"Failed to estimate noise for {file_path}: {e}")
        raise


def visualize_noise(file_path: Union[str, Path], y_max_factor: Optional[float] = None,
                    figsize: Optional[tuple] = None, title: Optional[str] = None,
                    show_bin_boundaries: Optional[bool] = None,
                    show_noise_points: Optional[bool] = None,
                    save_params: bool = False, backend: str = 'matplotlib',
                    interactive: bool = True, output_file: Optional[Union[str, Path]] = None,
                    **plot_kwargs):
    """
    Create noise estimation diagnostic visualization.
    
    This function creates diagnostic plots showing spectrum, noise points,
    adaptive bin boundaries, and RMS noise estimates, equivalent to
    Pipeline.visualize_noise(). Requires Stage 2 (noise estimation) completion.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing noise estimation results
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
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If Stage 2 dependencies are not met
    RuntimeError
        If visualization fails
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> # Create basic noise visualization
    >>> fig = ftmw.visualize_noise("experiment.ftmw")
    >>> # Customize visualization and save parameters
    >>> fig = ftmw.visualize_noise("experiment.ftmw", 
    ...                           y_max_factor=15.0, 
    ...                           show_bin_boundaries=True,
    ...                           save_params=True)
    >>> # Save to file
    >>> fig = ftmw.visualize_noise("experiment.ftmw", 
    ...                           output_file="noise_diagnostics.png")
    """
    try:
        # Delegate to Pipeline class for consistent behavior
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_noise(
            y_max_factor=y_max_factor,
            figsize=figsize,
            title=title,
            show_bin_boundaries=show_bin_boundaries,
            show_noise_points=show_noise_points,
            save_params=save_params,
            backend=backend,
            interactive=interactive,
            output_file=output_file,
            **plot_kwargs
        )
        
    except Exception as e:
        logger.error(f"Failed to create noise visualization for {file_path}: {e}")
        raise


def save_noise_parameters(file_path: Union[str, Path], 
                          parameters: Dict[str, Any]) -> None:
    """
    Save noise estimation parameters as defaults for pipeline file.
    
    This function saves noise estimation parameters to the .ftmw pipeline file
    for use in subsequent computations with from_saved_params=True.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
    parameters : dict
        Noise estimation parameters to save. Valid keys include:
        - 'skew_target': Target skewness for noise identification
        - 'min_bin_fraction': Minimum bin size fraction
        - 'smoothing_window_mhz': RMS smoothing window size
        - 'min_noise_fraction': Minimum noise fraction per bin
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    RuntimeError
        If parameter saving fails
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> params = {
    ...     'skew_target': 0.7,
    ...     'min_bin_fraction': 1/32,
    ...     'smoothing_window_mhz': 100.0
    ... }
    >>> ftmw.save_noise_parameters("experiment.ftmw", params)
    """
    try:
        # Use internal implementation for parameter saving
        from ._internal.stage2_impl import save_noise_parameters_impl
        save_noise_parameters_impl(str(file_path), parameters)
        logger.info(f"Saved {len(parameters)} noise parameters to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save noise parameters to {file_path}: {e}")
        raise


# =============================================================================
# Stage 3: Peak Detection Functions
# =============================================================================

def detect_peaks(file_path: Union[str, Path], min_snr: Optional[float] = None,
                  weak_medium_snr: Optional[float] = None,
                  medium_strong_snr: Optional[float] = None,
                  sg_window: Optional[int] = None,
                  sg_order: Optional[int] = None,
                  apodization_us: Optional[float] = None,
                  tau_us: Optional[float] = None,
                  min_exclusion_mhz: Optional[float] = None,
                  run_gap_pass: Optional[bool] = None) -> List[Peak]:
    """
    Detect and classify peaks (Stage 3), equivalent to Pipeline.detect_peaks().

    Requires Stage 1 (FT) and Stage 2 (noise). Two-pass detection operates on
    the Stage 1 persisted canonical spectrum (including its frequency trim
    range); peaks are reported on that user grid with SNR measured against the
    canonical Stage 2 noise.  There is no per-Stage-3 trim or zpf.

    Returns
    -------
    list of Peak
        Classified peaks, sorted by frequency.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.detect_peaks(
            min_snr=min_snr,
            weak_medium_snr=weak_medium_snr,
            medium_strong_snr=medium_strong_snr,
            sg_window=sg_window,
            sg_order=sg_order,
            apodization_us=apodization_us,
            tau_us=tau_us,
            min_exclusion_mhz=min_exclusion_mhz,
            run_gap_pass=run_gap_pass,
        )
    except Exception as e:
        logger.error(f"Failed to detect peaks for {file_path}: {e}")
        raise


def load_peaks(file_path: Union[str, Path]) -> List[Peak]:
    """Load the persisted Stage 3 peak list, equivalent to
    Pipeline.load_peaks(). Validates the on-disk structure loudly."""
    try:
        return Pipeline.open(file_path).load_peaks()
    except Exception as e:
        logger.error(f"Failed to load peaks from {file_path}: {e}")
        raise


def visualize_peaks(file_path: Union[str, Path],
                    figsize: Optional[tuple] = None,
                    title: Optional[str] = None,
                    y_max_factor: Optional[float] = None,
                    backend: str = 'matplotlib', interactive: bool = True,
                    output_file: Optional[Union[str, Path]] = None) -> Any:
    """
    Overlay classified detected peaks on the spectrum (Stage 3), equivalent
    to Pipeline.visualize_peaks(). Requires Stage 3 completion.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_peaks(
            figsize=figsize,
            title=title,
            y_max_factor=y_max_factor,
            backend=backend,
            interactive=interactive,
            output_file=output_file,
        )
    except Exception as e:
        logger.error(
            f"Failed to create peak visualization for {file_path}: {e}"
        )
        raise


def save_peak_parameters(file_path: Union[str, Path],
                         parameters: Dict[str, Any]) -> None:
    """Save Stage 3 detection parameters for reuse."""
    try:
        from ._internal.stage3_impl import save_peak_parameters_impl
        save_peak_parameters_impl(str(file_path), parameters)
        logger.info(f"Saved {len(parameters)} peak parameters to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save peak parameters to {file_path}: {e}")
        raise


# =============================================================================
# Utility Functions
# =============================================================================

def get_pipeline_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get pipeline file information and status.
    
    This function retrieves comprehensive information about a .ftmw pipeline
    file including source metadata, completed stages, and validation status,
    equivalent to Pipeline.info().
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
        
    Returns
    -------
    dict
        Pipeline information including:
        - 'filepath': Full path to pipeline file
        - 'valid': Whether file is valid
        - 'source_path': Original data source
        - 'format': Data format name
        - 'import_time': When data was imported
        - 'completed_stages': List of completed processing stages
        - 'next_available_stages': Stages ready to run
        - 'errors': List of issues if invalid
        - 'warnings': List of warnings if any
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> info = ftmw.get_pipeline_info("experiment.ftmw")
    >>> print(f"Source: {info['source_path']}")
    >>> print(f"Completed stages: {info['completed_stages']}")
    >>> print(f"Next available: {info['next_available_stages']}")
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.info()
    except Exception as e:
        logger.error(f"Failed to get info for {file_path}: {e}")
        return {
            'filepath': str(file_path),
            'valid': False,
            'error': f"Failed to get pipeline info: {e}"
        }


def list_available_stages(file_path: Union[str, Path]) -> List[str]:
    """
    Get list of processing stages ready to run.
    
    This function returns the names of processing stages that can be executed
    based on the current completion status of the pipeline file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
        
    Returns
    -------
    list of str
        Names of stages that can be executed next
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    RuntimeError
        If stage information cannot be retrieved
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> stages = ftmw.list_available_stages("experiment.ftmw")
    >>> print(f"Available stages: {stages}")
    >>> if 'stage1_complex_ft' in stages:
    ...     print("Ready for FT computation")
    """
    try:
        info = get_pipeline_info(file_path)
        return info.get('next_available_stages', [])
    except Exception as e:
        logger.error(f"Failed to get available stages for {file_path}: {e}")
        raise RuntimeError(f"Could not determine available stages: {e}")


# =============================================================================
# Module-level convenience functions
# =============================================================================

def workflow_summary(file_path: Union[str, Path]) -> str:
    """
    Generate a human-readable summary of pipeline status and workflow.
    
    This convenience function provides a formatted summary of the pipeline
    file status, completed stages, and suggested next steps.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
        
    Returns
    -------
    str
        Formatted summary string
        
    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> print(ftmw.workflow_summary("experiment.ftmw"))
    Pipeline: experiment.ftmw
    Source: examples/blackchirp_data/2638/ (blackchirp format)
    Status: Valid
    Completed: ['stage0_data_import']
    Next available: ['stage1_complex_ft']
    
    Suggested workflow:
    1. ftmw.compute_ft("experiment.ftmw", zpf=2, expf_us=5.0)
    2. ftmw.visualize_ft("experiment.ftmw", save_params=True)
    """
    try:
        info = get_pipeline_info(file_path)
        
        lines = [
            f"Pipeline: {Path(file_path).name}",
            f"Source: {Path(info['source_path']).name} ({info['format']} format)",
            f"Status: {'Valid' if info['valid'] else 'Invalid'}",
            f"Completed: {info['completed_stages']}",
            f"Next available: {info['next_available_stages']}"
        ]
        
        # Add suggested workflow for common stages
        if 'stage1_complex_ft' in info['next_available_stages']:
            lines.extend([
                "",
                "Suggested workflow:",
                f"1. ftmw.compute_ft(\"{Path(file_path).name}\", zpf=2, expf_us=5.0)",
                f"2. ftmw.visualize_ft(\"{Path(file_path).name}\", save_params=True)"
            ])
        elif 'stage2_noise_estimation' in info['next_available_stages']:
            lines.extend([
                "",
                "Suggested workflow:",
                f"1. ftmw.estimate_noise(\"{Path(file_path).name}\")",
                f"2. ftmw.visualize_noise(\"{Path(file_path).name}\")"
            ])
        
        # Add error information if invalid
        if not info['valid'] and 'errors' in info:
            lines.extend([
                "",
                "Issues found:",
                *[f"  - {error}" for error in info['errors']]
            ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error getting workflow summary for {file_path}: {e}"