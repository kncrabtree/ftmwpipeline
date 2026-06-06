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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from .core.data_structures import FID, ComplexFT, Peak, SpectrumFit, WindowPlan
from .core.noise_settings import NoiseSettings
from .core.peak_detection_settings import PeakDetectionSettings
from .core.stage_fit_settings import StageFitSettings
from .core.start_detection_settings import StartDetectionSettings
from .core.tau_calibration_settings import TauCalibrationSettings
from .core.window_planning_settings import WindowPlanningSettings
from .fitting.tau_calibration import ShapeRecommendation, TauCalibrationResult
from .pipeline import Pipeline
from .preprocessing.noise_estimation import NoiseResult
from .preprocessing.start_detection import StartDetectionResult

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# File Management Functions
# =============================================================================


def import_data(
    file_path: Union[str, Path],
    source: Union[str, Path],
    format_name: Optional[str] = None,
    fid_index: Optional[int] = None,
    force: bool = False,
    **loader_params: Any,
) -> Dict[str, Any]:
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
            **loader_params,
        )

        # Get pipeline info to return
        info = pipeline.info()

        # Return result consistent with _internal implementation
        result = {
            "pipeline_file": str(pipeline.filepath),
            "source_path": info["source_path"],
            "format_name": info["format"],
            "status": "success",
        }

        # Add FID metadata if available
        try:
            fid = pipeline.load_data()
            result["fid_metadata"] = {
                "n_points": fid.n_points,
                "duration_us": fid.duration_us,
                "probe_freq_mhz": fid.probe_freq_mhz,
                "sideband": fid.sideband.value,
                "shots": fid.shots,
                "spacing": fid.spacing,
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
            "valid": False,
            "errors": [f"Failed to validate pipeline file: {e}"],
            "warnings": [],
        }


# =============================================================================
# Start-time Detection Functions (pre-Stage 1)
# =============================================================================


def detect_start_time(
    file_path: Union[str, Path],
    sweep_max_us: Optional[float] = None,
    step_us: Optional[float] = None,
    guard_margin_us: Optional[float] = None,
    floor_factor: Optional[float] = None,
    band: Optional[Tuple[float, float]] = None,
    stamp: bool = True,
    *,
    settings: Optional[StartDetectionSettings] = None,
) -> StartDetectionResult:
    """Infer a good FID ``start_us`` from the data, equivalent to
    :meth:`Pipeline.detect_start_time`.

    Sweeps the FID window start time and integrates the FT magnitude over the
    active band; the chirp-end collapse plus an instrument-specific guard margin
    gives the recommended ``start_us``. When ``stamp=True`` (default) the value
    is written to the Stage 0 ``recommended_processing`` layer so a later
    :func:`compute_ft` with no explicit ``start_us`` inherits it. Requires only
    Stage 0 (FID); the band is resolved from the canonical Stage 1 trim when
    present, else the full positive spectrum.

    Parameters
    ----------
    file_path : str or Path
        Path to a ``.ftmw`` file with the FID imported.
    sweep_max_us, step_us, guard_margin_us, floor_factor :
        Individual overrides of the matching
        :class:`~ftmwpipeline.core.start_detection_settings.StartDetectionSettings`
        fields.
    band : tuple of float, optional
        Explicit ``(min_mhz, max_mhz)`` integration band override.
    stamp : bool, default True
        Whether to persist the recommended ``start_us``.
    settings : StartDetectionSettings, optional
        A full settings bundle; the explicit kwargs above win per-field.

    Returns
    -------
    StartDetectionResult
        The recommendation plus diagnostics.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.detect_start_time(
            sweep_max_us=sweep_max_us,
            step_us=step_us,
            guard_margin_us=guard_margin_us,
            floor_factor=floor_factor,
            band=band,
            stamp=stamp,
            settings=settings,
        )
    except Exception as e:
        logger.error(f"Failed to detect start time for {file_path}: {e}")
        raise


def visualize_start_detection(
    file_path: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    figsize: Optional[tuple] = None,
    *,
    settings: Optional[StartDetectionSettings] = None,
) -> Any:
    """Render the start-detection sweep diagnostic, equivalent to
    :meth:`Pipeline.visualize_start_detection` (runs detection without
    stamping)."""
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_start_detection(
            output_file=output_file,
            interactive=interactive,
            figsize=figsize,
            settings=settings,
        )
    except Exception as e:
        logger.error(f"Failed to visualize start detection for {file_path}: {e}")
        raise


# =============================================================================
# Stage 1 FT Processing Functions
# =============================================================================


def compute_ft(
    file_path: Union[str, Path],
    zpf: Optional[int] = None,
    expf_us: Optional[float] = None,
    trim: Optional[Tuple[float, float]] = None,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    window_function: Optional[str] = None,
    units_power: Optional[int] = None,
    from_saved_params: bool = False,
) -> ComplexFT:
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
        Exponential filter time constant in microseconds. ``None`` (or any
        non-positive value) disables apodization. There is no implicit
        fallback default — request apodization explicitly when you want it.
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
            from_saved_params=from_saved_params,
        )
    except Exception as e:
        logger.error(f"Failed to compute FT for {file_path}: {e}")
        raise


def visualize_ft(
    file_path: Union[str, Path],
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
        Exponential filter time constant in microseconds. ``None`` (or any
        non-positive value) disables apodization. There is no implicit
        fallback default — request apodization explicitly when you want it.
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
            show_fid_panels=show_fid_panels,
        )
    except Exception as e:
        logger.error(f"Failed to visualize FT for {file_path}: {e}")
        raise


def save_ft_parameters(file_path: Union[str, Path], parameters: Dict[str, Any]) -> None:
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


def estimate_noise(
    file_path: Union[str, Path],
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
    Estimate frequency-dependent noise with the scatter (high-pass) estimator.

    This function performs noise estimation on ComplexFT data stored in a .ftmw
    pipeline file, equivalent to Pipeline.estimate_noise(). Requires Stage 1
    (FT computation) to be completed first.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing ComplexFT data
    window_mhz, pedestal_mhz, line_k, n_iter, region_aware, smoothing_mhz,
    smoothing_percentile, convolve_mhz
        Scatter-estimator knobs; each defaults to the kernel's hard default
        when left unset. ``smoothing_mhz`` / ``smoothing_percentile`` set the
        broad lower-envelope median σ smoothing (``smoothing_mhz=0`` disables
        it); ``convolve_mhz`` is the Gaussian σ of the second step-removing
        pass.
    settings, preset :
        Alternative ways to populate the preset layer of the settings chain.

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
    >>> ftmw.compute_ft("experiment.ftmw", trim=(26500, 40000))
    >>> # Estimate noise with default parameters
    >>> noise_result = ftmw.estimate_noise("experiment.ftmw")
    >>> # Override a scatter knob
    >>> noise_result = ftmw.estimate_noise("experiment.ftmw", window_mhz=120.0)
    """
    try:
        # Delegate to Pipeline class for consistent behavior
        pipeline = Pipeline.open(file_path)
        return pipeline.estimate_noise(
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

    except Exception as e:
        logger.error(f"Failed to estimate noise for {file_path}: {e}")
        raise


def visualize_noise(
    file_path: Union[str, Path],
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

    This function creates diagnostic plots showing spectrum, noise points,
    bin boundaries, and RMS noise estimates, equivalent to
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
            backend=backend,
            interactive=interactive,
            output_file=output_file,
            **plot_kwargs,
        )

    except Exception as e:
        logger.error(f"Failed to create noise visualization for {file_path}: {e}")
        raise


# =============================================================================
# Stage 2b: Tau Calibration Functions
# =============================================================================


def calibrate_tau(
    file_path: Union[str, Path],
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
    """Run the Stage 2b data-driven tau calibration, equivalent to
    :meth:`Pipeline.calibrate_tau`.

    Requires Stages 0-2 completed. Persists the calibration to
    ``/stage2b_tau_calibration``. Parameters left as ``None`` fall
    through the four-layer resolution chain (``explicit > preset >
    persisted > recommended > hard default``); ``settings=`` and
    ``preset=`` populate the preset layer and are mutually exclusive.
    The resolved settings are stamped to
    ``processing_parameters/stage2b_tau`` so a follow-up no-kwargs
    call inherits the same recipe.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.calibrate_tau(
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
    except Exception as e:
        logger.error(f"Failed to calibrate tau for {file_path}: {e}")
        raise


def load_tau_calibration(file_path: Union[str, Path]) -> TauCalibrationResult:
    """Load the persisted Stage 2b :class:`TauCalibrationResult`."""
    try:
        return Pipeline.open(file_path).load_tau_calibration()
    except Exception as e:
        logger.error(f"Failed to load tau calibration from {file_path}: {e}")
        raise


def calibrate_tau_G(
    file_path: Union[str, Path],
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
    """Run the Stage 2b Gaussian-shape τ_G calibration, equivalent to
    :meth:`Pipeline.calibrate_tau_G`.

    Per-bin Voigt fits on the STFT contributor pool yield a per-band τ_G
    majority that the Stage 5 Gaussian path consumes. Persists to
    ``/stage2b_tau_G_calibration``. Independent of the pure-exp
    :func:`calibrate_tau`; both can coexist on one ``.ftmw`` file.

    Parameters left as ``None`` fall through the four-layer resolution
    chain; ``settings=`` and ``preset=`` populate the preset layer and
    are mutually exclusive. The resolved settings share the
    ``processing_parameters/stage2b_tau`` block with the pure-exp twin.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.calibrate_tau_G(
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
    except Exception as e:
        logger.error(f"Failed to calibrate τ_G for {file_path}: {e}")
        raise


def load_tau_G_calibration(file_path: Union[str, Path]) -> TauCalibrationResult:
    """Load the persisted Gaussian Stage 2b :class:`TauCalibrationResult`."""
    try:
        return Pipeline.open(file_path).load_tau_G_calibration()
    except Exception as e:
        logger.error(f"Failed to load τ_G calibration from {file_path}: {e}")
        raise


def recommend_shape(
    file_path: Union[str, Path],
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
    """Run the 3-way L/G/V shape-recommendation hook, equivalent to
    :meth:`Pipeline.recommend_shape`.

    Per-bin AICc vote (exp / gauss / voigt) over the same STFT
    contributor pool the τ calibrations use; SNR-weighted majority
    decides between the two pure shapes (Voigt is reported as a
    diagnostic but does not enter the recommendation). The verdict's
    ``recommended_shape`` is stamped onto every Stage 2b group present
    on the file so the Stage 5 resolver's *recommended* layer picks it
    up automatically. Requires Stage 1 (active region + frequency trim)
    to have completed; the Stage 2b calibrations are optional but the
    persisted contract only fires when at least one of them has run.

    Parameters left as ``None`` fall through the four-layer resolution
    chain; ``settings=`` and ``preset=`` populate the preset layer and
    are mutually exclusive.
    """
    try:
        return Pipeline.open(file_path).recommend_shape(
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
    except Exception as e:
        logger.error(f"Failed to recommend shape for {file_path}: {e}")
        raise


def visualize_tau_heatmap(
    file_path: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    figsize: Optional[tuple] = None,
) -> Any:
    """2D STFT magnitude heatmap, equivalent to
    :meth:`Pipeline.visualize_tau_heatmap`."""
    try:
        return Pipeline.open(file_path).visualize_tau_heatmap(
            output_file=output_file,
            interactive=interactive,
            figsize=figsize,
        )
    except Exception as e:
        logger.error(f"Failed to visualize tau heatmap for {file_path}: {e}")
        raise


def visualize_tau_distribution(
    file_path: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    figsize: Optional[tuple] = None,
) -> Any:
    """tau-distribution analysis panel, equivalent to
    :meth:`Pipeline.visualize_tau_distribution`."""
    try:
        return Pipeline.open(file_path).visualize_tau_distribution(
            output_file=output_file,
            interactive=interactive,
            figsize=figsize,
        )
    except Exception as e:
        logger.error(f"Failed to visualize tau distribution for {file_path}: {e}")
        raise


# =============================================================================
# Stage 3: Peak Detection Functions
# =============================================================================


def detect_peaks(
    file_path: Union[str, Path],
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
    """Detect and classify peaks (Stage 3), equivalent to Pipeline.detect_peaks().

    Requires Stage 1 (FT) and Stage 2 (noise). Two-pass detection operates on
    the Stage 1 persisted canonical spectrum (including its frequency trim
    range); peaks are reported on that user grid with SNR measured against the
    canonical Stage 2 noise.  There is no per-Stage-3 trim or zpf.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    min_snr : float, optional
        Promotion SNR cutoff (peaks at/above this threshold on the user grid
        are marked ``promoted=True`` and move to Stage 4); detection runs
        aggressively below this internally. ALL detected peaks are stored;
        ``promoted`` marks the Stage-4 gate. Default 3.0.
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
        scipy.signal window name (e.g. ``"blackmanharris"``, ``"blackman"``,
        ``"hann"``). Default ``"blackmanharris"`` -- a strong window that
        suppresses truncation sidelobes so the primary strong-line list is
        clean. Affects only which positions are found, never amplitude/SNR.
    min_exclusion_mhz : float, optional
        Minimum gap-pass exclusion half-width per primary peak in MHz.
    run_gap_pass : bool, optional
        If False, disable the unapodized gap pass (primary pass only).
    settings : PeakDetectionSettings, optional
        Bundle of Stage 3 knobs (preset-layer of the four-layer
        resolution chain); fields left ``None`` fall through. Mutually
        exclusive with ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage3:`` block.
        Mutually exclusive with ``settings``.

    Returns
    -------
    list of Peak
        ALL detected peaks (promoted and non-promoted), sorted by frequency.
        Each peak's ``properties`` dict includes ``promoted`` (bool),
        ``internal_snr``, ``internal_frequency``, and ``detection_pass``.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.detect_peaks(
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


def visualize_peaks(
    file_path: Union[str, Path],
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    y_max_factor: Optional[float] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
    output_file: Optional[Union[str, Path]] = None,
    show_snr_histogram: bool = False,
) -> Any:
    """Overlay classified detected peaks on the spectrum (Stage 3), equivalent
    to Pipeline.visualize_peaks(). Requires Stage 3 completion.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file with Stage 3 results.
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
            show_snr_histogram=show_snr_histogram,
        )
    except Exception as e:
        logger.error(f"Failed to create peak visualization for {file_path}: {e}")
        raise


def save_peak_parameters(
    file_path: Union[str, Path], parameters: Dict[str, Any]
) -> None:
    """Save Stage 3 detection parameters for reuse."""
    try:
        from ._internal.stage3_impl import save_peak_parameters_impl

        save_peak_parameters_impl(str(file_path), parameters)
        logger.info(f"Saved {len(parameters)} peak parameters to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save peak parameters to {file_path}: {e}")
        raise


# =============================================================================
# Stage 4: Window Assignment Functions
# =============================================================================


def assign_windows(
    file_path: Union[str, Path],
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
    """Assign analysis windows (Stage 4), equivalent to Pipeline.assign_windows().

    Requires Stage 3 (peak detection). Turns the promoted Stage 3 peaks into a
    fit plan -- a set of disjoint analysis windows, each annotated with the
    peaks to fit freely, the strong out-of-band lines whose leakage is carried
    frozen, a fit dependency order, and a difficulty class. Stage 4 is purely
    structural; the plan is persisted to the .ftmw file.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    edge_m : int, optional
        Rolling-scan complex-edge coherence band width (default 64).
    trim_m : int, optional
        Trim-refinement band width (default 32).
    edge_threshold : float, optional
        ``S_coh`` threshold ``T_edge`` (default 8.0).
    max_window_width_mhz : float, optional
        Width cap; a wider window is HARD and gets a split proposal
        (default 40.0).
    min_freeze_snr : float, optional
        Freeze-eligibility SNR cutoff for fixed contributors (default 50.0).
    min_window_half_width_mhz : float, optional
        Minimum half-width of a window around an isolated weak line
        (default 2.0).
    magnitude_attachment_threshold : float, optional
        Tier-1 contributor-attachment threshold in units of σ_c.
        A strong promoted peak is attached to a window's
        ``fixed_contributors`` when its predicted mean |skirt| on
        that window's grid is at least ``threshold * sigma_c(w)``
        (default 0.1).
    tau_us : float, optional
        Assumed decay constant for the analytic leakage reach
        (default: undamped/boxcar limit).
    settings : WindowPlanningSettings, optional
        Bundle of Stage 4 knobs (preset-layer of the four-layer resolution
        chain); fields left ``None`` fall through. Mutually exclusive with
        ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage4:`` block.
        Mutually exclusive with ``settings``.

    Returns
    -------
    WindowPlan
        The fit plan: disjoint windows, dependency DAG, topological order,
        parallel batches, parameters and diagnostics.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.assign_windows(
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
    except Exception as e:
        logger.error(f"Failed to assign windows for {file_path}: {e}")
        raise


def load_windows(file_path: Union[str, Path]) -> WindowPlan:
    """Load the persisted Stage 4 window plan, equivalent to
    Pipeline.load_windows(). Validates the on-disk structure loudly."""
    try:
        return Pipeline.open(file_path).load_windows()
    except Exception as e:
        logger.error(f"Failed to load windows from {file_path}: {e}")
        raise


def visualize_windows(
    file_path: Union[str, Path],
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    y_max_factor: Optional[float] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
    output_file: Optional[Union[str, Path]] = None,
) -> Any:
    """Overlay the Stage 4 window plan on the spectrum, equivalent to
    Pipeline.visualize_windows(). Requires Stage 4 completion.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file with Stage 4 results.
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
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_windows(
            figsize=figsize,
            title=title,
            y_max_factor=y_max_factor,
            backend=backend,
            interactive=interactive,
            output_file=output_file,
        )
    except Exception as e:
        logger.error(f"Failed to create window visualization for {file_path}: {e}")
        raise


def save_window_parameters(
    file_path: Union[str, Path], parameters: Dict[str, Any]
) -> None:
    """Save Stage 4 window-assignment parameters for reuse."""
    try:
        from ._internal.stage4_impl import save_window_parameters_impl

        save_window_parameters_impl(str(file_path), parameters)
        logger.info(f"Saved {len(parameters)} window parameters to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save window parameters to {file_path}: {e}")
        raise


def fit_peaks(
    file_path: Union[str, Path],
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
    settings: Optional[StageFitSettings] = None,
    preset: Optional[str] = None,
) -> SpectrumFit:
    """Fit each Stage 4 window's lines (Stage 5), equivalent to Pipeline.fit_peaks().

    Requires Stage 4 (window assignment). The fit runs on the active-portion FT
    computed on demand from the FID plus the canonical Stage 1 settings; per-bin
    noise is measured on the active-FT directly. Persists the resulting
    :class:`SpectrumFit` to ``/stage5_fitting``.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    tau0_us : float, optional
        Starting / default shared decay constant per window (microseconds).
        Defaults to ``expf_us`` when the canonical Stage 1 setting is set,
        otherwise to ``T_active / 3``.
    fit_tau : bool, optional
        Free vs fixed per-window tau (default True).
    max_decay_factor : float, optional
        Tau bound factor (default 5).
    residual_edge_threshold : float, optional
        ``S_coh`` threshold above which a residual edge triggers a thaw attempt.
    residual_edge_m : int, optional
        Band width (in active-FT bins) of the residual-edge coherence test.
    max_thaw_rounds : int, optional
        Maximum local-thaw rounds per window per call.
    max_replan_rounds : int, optional
        Maximum structural-replan rounds per call (0 disables).
    max_residual_rescue_rounds : int, optional
        Cap on per-window residual-rescue + joint-refit cycles. ``None``
        (the default) resolves to the calibrated default cap; pass ``0``
        to disable the rescue pass entirely (escape hatch for diagnostic
        re-fits). The rescue runs on every window's post-thaw fit by
        default.
    rescue_snr_threshold, rescue_prominence_threshold : optional
        Rescue tuning knobs -- see :func:`Pipeline.fit_peaks` for the
        defaults. Ignored when ``max_residual_rescue_rounds`` is 0.
    tau_maj_override_us, sigma_tau_override_us : float, optional
        Atomic-pair manual override for the Stage 2b tau calibration. When
        both are supplied (positive), they replace any persisted Stage 2b
        result for this fit; useful for A/B-ing a hand-tuned tau anchor
        against the persisted one. Supplying only one of the pair raises
        ``ValueError``.
    settings : StageFitSettings, optional
        Bundle of Stage 5 knobs that enters the resolution chain at the
        preset layer (see
        :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
        for the layered precedence). The explicit kwargs above still win
        per-field over ``settings``.
    preset : str, optional
        Name of a packaged preset or a path to a YAML file -- an
        alternative to ``settings``. Passing both raises ``ValueError``.

    Returns
    -------
    SpectrumFit
        The persistent fit aggregate.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.fit_peaks(
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
    except Exception as e:
        logger.error(f"Failed to fit peaks for {file_path}: {e}")
        raise


def load_fit(file_path: Union[str, Path]) -> SpectrumFit:
    """Load the persisted Stage 5 fit, equivalent to Pipeline.load_fit().
    Validates the on-disk structure loudly."""
    try:
        return Pipeline.open(file_path).load_fit()
    except Exception as e:
        logger.error(f"Failed to load fit from {file_path}: {e}")
        raise


def validate_stage5_shape_error(
    file_path: Union[str, Path],
    kappa: Optional[float] = None,
    noise_floor: Optional[float] = None,
    ground_truth: Optional[Union[str, Path]] = None,
    match_tol_fwhm: float = 0.5,
) -> Dict[str, Any]:
    """Assess a persisted Stage 5 fit against the SNR-aware acceptance framework.

    Read-only. Returns the Tier 1 (SNR-aware health) / Tier 2 (gate firing) /
    Tier 3 (known-line ground truth, when ``ground_truth`` is given) report,
    equivalent to :meth:`Pipeline.validate_stage5_shape_error`.
    """
    try:
        return Pipeline.open(file_path).validate_stage5_shape_error(
            kappa=kappa,
            noise_floor=noise_floor,
            ground_truth=ground_truth,
            match_tol_fwhm=match_tol_fwhm,
        )
    except Exception as e:
        logger.error(f"Failed to validate Stage 5 shape error for {file_path}: {e}")
        raise


def visualize_fit(
    file_path: Union[str, Path],
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    window_id: Optional[int] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
    output_file: Optional[Union[str, Path]] = None,
) -> Any:
    """Overlay the Stage 5 fit on the spectrum, equivalent to
    Pipeline.visualize_fit(). Requires Stage 5 completion.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file with Stage 5 results.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    title : str, optional
        Custom plot title.
    window_id : int, optional
        When set, draw a per-window detail figure (re/im, magnitude+residual,
        time envelope, audit-trail); otherwise an overview overlay of the
        fitted model on the persisted spectrum.
    backend : str, default ``'matplotlib'``
        Plotting backend (only ``'matplotlib'`` supported).
    interactive : bool, default True
        Whether to open an interactive window.
    output_file : str or Path, optional
        Save plot to this path (non-interactive mode).
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_fit(
            figsize=figsize,
            title=title,
            window_id=window_id,
            backend=backend,
            interactive=interactive,
            output_file=output_file,
        )
    except Exception as e:
        logger.error(f"Failed to create fit visualization for {file_path}: {e}")
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
            "filepath": str(file_path),
            "valid": False,
            "error": f"Failed to get pipeline info: {e}",
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
        return cast(List[str], info.get("next_available_stages", []))
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
            f"Next available: {info['next_available_stages']}",
        ]

        # Add suggested workflow for common stages
        if "stage1_complex_ft" in info["next_available_stages"]:
            lines.extend(
                [
                    "",
                    "Suggested workflow:",
                    f'1. ftmw.compute_ft("{Path(file_path).name}", zpf=2, expf_us=5.0)',
                    f'2. ftmw.visualize_ft("{Path(file_path).name}", save_params=True)',
                ]
            )
        elif "stage2_noise_estimation" in info["next_available_stages"]:
            lines.extend(
                [
                    "",
                    "Suggested workflow:",
                    f'1. ftmw.estimate_noise("{Path(file_path).name}")',
                    f'2. ftmw.visualize_noise("{Path(file_path).name}")',
                ]
            )

        # Add error information if invalid
        if not info["valid"] and "errors" in info:
            lines.extend(
                ["", "Issues found:", *[f"  - {error}" for error in info["errors"]]]
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting workflow summary for {file_path}: {e}"


# =============================================================================
# Companion parameter tuning
# =============================================================================


def tune_list(
    selector: Optional[str] = None,
    *,
    include_advanced: bool = False,
) -> Tuple[Any, ...]:
    """List the registered tunable knobs, equivalent to
    :meth:`Pipeline.tune_list`.

    Parameters
    ----------
    selector : str, optional
        Filter by dotted-path prefix (e.g. ``"stage2b"`` /
        ``"stage2b.gaussian"``); the legacy stage-label match
        (``"stage2_noise"`` / ``"start_detection"``) is kept as a fallback.
    include_advanced : bool, default False
        Reveal advanced-tier knobs hidden from the default listing.

    Returns
    -------
    tuple of KnobSpec
        Path-sorted knob specifications.
    """
    return Pipeline.tune_list(selector, include_advanced=include_advanced)


def tune_scan(
    file_path: Union[str, Path],
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
) -> Any:
    """Sweep a single pipeline knob across a grid, equivalent to
    :meth:`Pipeline.tune_scan`.

    Re-runs the knob's stage for each grid value on a working copy of
    ``file_path`` (the input is never mutated) and returns a ``SweepResult``
    with the table rows, a CSV path, an optional plot, a best-effort
    recommendation, and instructions for applying the chosen value.

    Parameters
    ----------
    file_path : str or Path
        A ``.ftmw`` already built through the knob's upstream stage.
    knob : str
        Dotted knob path (see :func:`tune_list`), e.g.
        ``"stage2.window_mhz"``.
    grid : sequence, optional
        Values to sweep; defaults to the knob's registered grid.
    output_dir : str or Path, optional
        Where the CSV/plot/working-copy land (default: current directory).
    reuse : bool, default False
        Reuse an existing working copy instead of re-copying the input.
    make_plot : bool, default True
        Render the knob's plot adapter if it has one.
    quiet : bool, default False
        Suppress the progress indicator (printed to stderr by default).
    zoom_regions : sequence of (lo_mhz, hi_mhz), optional
        Explicit zoom windows for the region-based plot adapters (Stage 3 /
        Stage 4), overriding their divergence auto-selection.
    n_zoom, zoom_width_mhz : optional
        When ``zoom_regions`` is not given, how many regions to auto-select and
        how wide each is; ``None`` keeps the adapter defaults.
    fit_top_snr, fit_sample, fit_freqs, fit_sample_seed, fit_all : optional
        Window selection for Stage 5 fit knobs: re-fit only the ``fit_top_snr``
        brightest windows + a seeded ``fit_sample`` random sample + the windows
        nearest each ``fit_freqs`` value, rather than the whole plan.
        ``fit_all=True`` re-fits every window. Ignored by non-fit knobs.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.tune_scan(
            knob,
            grid=grid,
            output_dir=output_dir,
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
    except Exception as e:
        logger.error(f"Failed to scan knob {knob!r} for {file_path}: {e}")
        raise


def tune_scan_batch(
    file_path: Union[str, Path],
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
) -> Any:
    """Sweep every knob matched by ``selector`` on its default grid, equivalent
    to :meth:`Pipeline.tune_scan_batch`.

    A convenience over :func:`tune_scan` for reviewing a whole stage / sub-block
    at once instead of driving knobs one-by-one. ``selector`` filters by
    dotted-path prefix (e.g. ``"stage2b"`` / ``"stage2b.gaussian"``) just like
    :func:`tune_list`; ``include_advanced`` adds the advanced-tier knobs. Each
    knob runs on its own working copy of ``file_path`` (never mutated); a knob
    whose scan fails (e.g. its required stage is absent) is recorded as a failed
    ``BatchItem`` and the batch continues.

    Returns
    -------
    list of BatchItem
        One per matched knob, in registry order; ``item.ok`` / ``item.result`` /
        ``item.error`` report each knob's outcome.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.tune_scan_batch(
            selector,
            include_advanced=include_advanced,
            output_dir=output_dir,
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
    except Exception as e:
        logger.error(f"Failed to batch-scan {selector!r} for {file_path}: {e}")
        raise


def settings_show(
    file_path: Union[str, Path],
    selector: Optional[str] = None,
    *,
    include_advanced: bool = False,
    preset: Optional[Union[str, Path]] = None,
) -> Tuple[Any, ...]:
    """Resolved value + provenance per setting, equivalent to
    :meth:`Pipeline.settings_show`.

    Reports, for each covered setting of ``file_path``, the value actually in
    effect and the layer that supplied it -- the resolved-view counterpart to
    :func:`tune_list`'s tunable-knob listing.

    Parameters
    ----------
    file_path : str or Path
        The ``.ftmw`` experiment to inspect.
    selector : str, optional
        Filter by dotted-path prefix (e.g. ``"stage2b"`` /
        ``"stage2b.gaussian"``); ``None`` returns every setting.
    include_advanced : bool, default False
        Reveal advanced-tier settings hidden from the default view.
    preset : str or Path, optional
        Populate the ``.yml`` provenance layer from this preset (bare name or
        path). Because a persisted ``.ftmw`` value outranks a preset, a named
        preset changes the resolved view only for fields the file has not fixed.

    Returns
    -------
    tuple of SettingRow
        One row per setting, carrying ``path`` / ``value`` / ``source`` /
        ``hard_default`` / ``tier`` / ``help``.
    """
    return Pipeline.open(file_path).settings_show(
        selector,
        include_advanced=include_advanced,
        preset=preset,
    )


def settings_set(file_path: Union[str, Path], knob: str, value: str) -> Any:
    """Persist a chosen value into the ``.ftmw``, equivalent to
    :meth:`Pipeline.settings_set`.

    Coerces ``value`` to ``knob``'s field type, writes it to the persisted
    layer, and invalidates the affected stage plus every downstream stage so the
    file never carries results inconsistent with its settings. Stage 1 FT-shaping
    knobs (``zpf`` / ``expf_us`` / ``window_function``) are rejected -- set them
    via :func:`compute_ft`.

    Parameters
    ----------
    file_path : str or Path
        The ``.ftmw`` to modify.
    knob : str
        Dotted settings path (``stage2.window_mhz`` /
        ``stage5.tau.max_decay_factor`` / ``stage5.shape``).
    value : str
        The value in string form; coerced to the field's declared type.

    Returns
    -------
    SetResult
        Carries the knob path, the coerced value, and the invalidated stages.
    """
    return Pipeline.open(file_path).settings_set(knob, value)


def settings_export(
    file_path: Union[str, Path],
    out_path: Union[str, Path],
    selector: Optional[str] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Write the file's chosen Stage 2--5 values to a ``.yml`` preset, equivalent
    to :meth:`Pipeline.settings_export`.

    Each stage's persisted settings serialize into the matching ``stageN:``
    block, filtered by the dotted ``selector``; the result loads back through the
    stages' ``--preset`` path. Stage 1 is excluded (presets do not carry FT
    settings).

    Parameters
    ----------
    file_path : str or Path
        The ``.ftmw`` to read chosen values from.
    out_path : str or Path
        Destination ``.yml`` preset file.
    selector : str, optional
        Restrict the export to a dotted prefix (e.g. ``"stage5"`` /
        ``"stage5.rescue"``); ``None`` exports every persisted Stage 2--5 value.
    name, description : str, optional
        Preset metadata written alongside the stage blocks (``name`` defaults to
        the output file stem).

    Returns
    -------
    ExportResult
        Carries the written path and the exported setting paths.
    """
    return Pipeline.open(file_path).settings_export(
        out_path,
        selector,
        name=name,
        description=description,
    )
