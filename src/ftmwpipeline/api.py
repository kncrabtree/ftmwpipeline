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

Example Usage::

    import ftmwpipeline.api as ftmw

    # Create pipeline from data
    ftmw.import_data("experiment.ftmw", source="examples/blackchirp_data/2638/")

    # Load and process data
    fid = ftmw.load_fid("experiment.ftmw")
    complex_ft = ftmw.compute_ft("experiment.ftmw", trim=(26500, 40000))

    # Visualization and parameter management
    ftmw.visualize_ft("experiment.ftmw", save_params=True)
    ftmw.save_ft_parameters("experiment.ftmw", {'trim': (26500, 40000)})

    # File management
    info = ftmw.get_pipeline_info("experiment.ftmw")
    stages = ftmw.list_available_stages("experiment.ftmw")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ._internal.read_impl import (
    read_metadata_impl,
    read_table_impl,
    read_tables_impl,
)
from ._internal.stage5_impl import _DETAIL_PAD_FACTOR
from ._internal.stage6_impl import (
    DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    DEFAULT_DISPLAY_BAR,
    CreateWindowResult,
    CurationApplyResult,
    DecisionLogEntry,
    EnvironmentAckResult,
    RankedWindow,
    RefitWindowResult,
    ReviewPreviewResult,
    ReviewRunResult,
    UndoResult,
)
from .core.calibration import CalibrationStamp
from .core.curation import REFIT_SNAP_TOL_MHZ, Frame
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
from .core.stage_fit_settings import ClockSource, StageFitSettings
from .core.start_detection_settings import StartDetectionSettings
from .core.tau_calibration_settings import TauCalibrationSettings
from .core.window_planning_settings import WindowPlanningSettings
from .fitting.tau_calibration import ShapeRecommendation, TauCalibrationResult
from .fitting.timebase_calibration import TimebaseCalibrationResult
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
        FID index for multi-FID formats (e.g., Blackchirp)
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


def get_clock_sources(
    file_path: Union[str, Path],
) -> Optional[Tuple[ClockSource, ...]]:
    """Return the declared (recommended) instrument clock sources, or ``None``.

    These clock fundamentals seed the Stage 5 spur-gate lattice. The
    declaration is the *recommended* resolver layer; an explicit ``clocks=`` at
    fit time and persisted Stage 5 settings outrank it.

    Parameters
    ----------
    file_path : str or Path
        Path to an existing .ftmw pipeline file.

    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> ftmw.get_clock_sources("exp_2638.ftmw")
    """
    return Pipeline(file_path).get_clock_sources()


def set_clock_sources(
    file_path: Union[str, Path],
    clocks: Any,
    *,
    replace: bool = True,
) -> Tuple[ClockSource, ...]:
    """Declare instrument clock sources on an existing experiment.

    ``clocks`` is a sequence of :class:`ClockSource` or
    ``{freq_mhz, locked, label}`` mappings. Declare chain *fundamentals*
    (e.g. 5760, not the 11520 product). With ``replace=False`` the sources are
    appended to the current declaration. Returns the resulting declaration.

    The declaration is written to the recommended layer, so it never overrides a
    persisted Stage 5 setting (the D11 reproducibility guarantee).

    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> ftmw.set_clock_sources(
    ...     "exp.ftmw",
    ...     [{"freq_mhz": 5760.0, "locked": True, "label": "synth"}],
    ... )
    """
    return Pipeline(file_path).set_clock_sources(clocks, replace=replace)


def remove_clock_sources(
    file_path: Union[str, Path],
    freqs_mhz: Sequence[float],
) -> Tuple[ClockSource, ...]:
    """Remove declared clock sources matching the given frequencies (MHz)."""
    return Pipeline(file_path).remove_clock_sources(freqs_mhz)


def clear_clock_sources(file_path: Union[str, Path]) -> None:
    """Clear the clock-source declaration on an existing experiment."""
    Pipeline(file_path).clear_clock_sources()


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
    *,
    band: Optional[Tuple[float, float]] = None,
    stamp: bool = True,
    settings: Optional[StartDetectionSettings] = None,
) -> StartDetectionResult:
    """Infer a good FID ``start_us`` from the data, equivalent to
    :meth:`Pipeline.detect_start_time`.

    Sweeps the FID window start time and integrates the FT magnitude over the
    active band; the chirp-end collapse plus an instrument-specific guard margin
    gives the recommended ``start_us``. When ``stamp=True`` (default) the value
    is written to the Stage 0 ``recommended_processing`` layer so a later
    :func:`compute_ft` with no explicit ``start_us`` inherits it. Requires only
    Stage 0 (FID); the band is resolved from the persisted Stage 1 trim when
    present, else the full positive spectrum.

    Parameters
    ----------
    file_path : str or Path
        Path to a ``.ftmw`` file with the FID imported.
    band : tuple of float, optional
        Explicit ``(min_mhz, max_mhz)`` integration band override (a convenience
        for the ``settings`` band fields).
    stamp : bool, default True
        Whether to persist the recommended ``start_us``.
    settings : StartDetectionSettings, optional
        The detection knobs. Stage 0 is a flat bundle with concrete defaults:
        construct a
        :class:`~ftmwpipeline.core.start_detection_settings.StartDetectionSettings`
        with the fields to override (``sweep_max_us`` / ``step_us`` /
        ``guard_margin_us`` / ``floor_factor`` / …).

    Returns
    -------
    StartDetectionResult
        The recommendation plus diagnostics.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.detect_start_time(
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
    trim: Optional[Tuple[float, float]] = None,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    units_power: Optional[int] = None,
    from_saved_params: bool = False,
) -> ComplexFT:
    """
    Compute Fourier Transform with specified processing parameters.

    This function performs FT computation on FID data stored in a .ftmw pipeline
    file, equivalent to Pipeline.compute_ft(). Can be called multiple times
    safely. The FT is unconditionally unapodized, un-windowed, and
    native-length.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing FID data
    trim : tuple of float, optional
        (min_freq, max_freq) in MHz to trim spectrum
    start_us : float, optional
        FID window start time in microseconds
    end_us : float, optional
        FID window end time in microseconds
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
    >>> # Compute with specific parameters (persisted to the file)
    >>> complex_ft = ftmw.compute_ft("experiment.ftmw", trim=(26500, 40000))
    >>>
    >>> # Use saved/recommended settings only
    >>> complex_ft = ftmw.compute_ft("experiment.ftmw", from_saved_params=True)
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.compute_ft(
            trim=trim,
            start_us=start_us,
            end_us=end_us,
            units_power=units_power,
            from_saved_params=from_saved_params,
        )
    except Exception as e:
        logger.error(f"Failed to compute FT for {file_path}: {e}")
        raise


def compute_display_ft(
    file_path: Union[str, Path],
    pad_factor: int = _DETAIL_PAD_FACTOR,
) -> ComplexFT:
    """
    Compute the zero-padded, analysis-band DISPLAY FT, equivalent to
    :meth:`Pipeline.compute_display_ft`.

    This is the same 2x-zero-padded magnitude spectrum the Stage 5 report and
    'fit show' detail panels render -- contrast with :func:`compute_ft`, the
    standard FT, which is unpadded and native-length and is what everything
    downstream actually fits and scores on. This FT zero-fills the
    active-region FID slice by ``pad_factor`` (display default ``2``, the
    information limit for a magnitude spectrum) purely to interpolate the
    magnitude curve between the native bins, then trims to
    :func:`compute_ft`'s own frequency band (``from_saved_params=True``) --
    it is display-only, differs from the standard FT only in bin density,
    never extent, and never feeds fitting, noise, or chi-squared. Display
    magnitude is ``abs(spectrum) * amplitude_scale``.

    Depends on Stage 1 (the FID plus persisted FT settings, including any
    trim) only -- does not require a persisted Stage 5 fit.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file containing FID data and persisted FT
        settings (Stage 0 + 1 complete).
    pad_factor : int, default 2
        Zero-fill factor applied to the active-region FID slice.

    Returns
    -------
    ComplexFT
        ``freq_array`` sorted ascending in molecular frequency, trimmed to
        :func:`compute_ft`'s band at ``pad_factor``x its density.
        ``complex_spectrum`` aligned to it. ``metadata`` carries
        ``amplitude_scale`` (float), ``units_label`` (str), and
        ``pad_factor`` (int).

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist.
    StageDependencyError
        If required dependencies (FID data / persisted FT settings) are not
        available.
    RuntimeError
        If FT computation fails.

    Examples
    --------
    >>> import numpy as np
    >>> import ftmwpipeline.api as ftmw
    >>> display_ft = ftmw.compute_display_ft("experiment.ftmw")
    >>> magnitude = np.abs(display_ft.complex_spectrum) * (
    ...     display_ft.metadata["amplitude_scale"]
    ... )
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.compute_display_ft(pad_factor=pad_factor)
    except Exception as e:
        logger.error(f"Failed to compute display FT for {file_path}: {e}")
        raise


def visualize_ft(
    file_path: Union[str, Path],
    trim: Optional[Tuple[float, float]] = None,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    units_power: Optional[int] = None,
    save_params: bool = False,
    interactive: bool = True,
    output_file: Optional[Union[str, Path]] = None,
    show_fid_panels: bool = True,
) -> Any:
    """
    Create enhanced FT visualization with processing workflow display.

    This function creates comprehensive FT visualization showing the complete
    FID-to-spectrum processing workflow, equivalent to Pipeline.visualize_ft().
    The FT is unconditionally unapodized, un-windowed, and
    native-length.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file
    trim : tuple of float, optional
        (min_freq, max_freq) in MHz to trim spectrum
    start_us : float, optional
        FID window start time in microseconds
    end_us : float, optional
        FID window end time in microseconds
    units_power : int, optional
        Scaling factor as power of 10. If None, uses cached default or 6.
    save_params : bool, default False
        Whether to save parameters as defaults for this experiment
    interactive : bool, default True
        Whether to show interactive plot
    output_file : str or Path, optional
        Path to save plot image (for non-interactive mode)
    show_fid_panels : bool, default True
        Whether to show FID processing panels

    Returns
    -------
    figure
        Matplotlib figure object

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
    >>> # Create interactive visualization
    >>> fig = ftmw.visualize_ft("experiment.ftmw", save_params=True)
    >>>
    >>> # Save plot to file
    >>> fig = ftmw.visualize_ft("experiment.ftmw", interactive=False,
    ...                         output_file="spectrum.png")
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.visualize_ft(
            trim=trim,
            start_us=start_us,
            end_us=end_us,
            units_power=units_power,
            save_params=save_params,
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
        - 'start_us', 'end_us': FID time window
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
    settings, preset :
        Composable ways to drive the settings chain (a ``NoiseSettings`` bundle
        as the explicit override layer, a YAML preset's ``stage2:`` block as the
        preset layer beneath the persisted record). Individual scatter knobs
        (``window_mhz`` / ``smoothing_mhz`` / ``smoothing_percentile`` /
        ``convolve_mhz`` / …) are set on the ``NoiseSettings`` instance; the
        explicit layer outranks the persisted record, which outranks the preset
        (per D11), so a no-arg call reproduces the persisted recipe.

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
    >>> from ftmwpipeline.core.noise_settings import NoiseSettings
    >>> # First compute FT if not already done
    >>> ftmw.compute_ft("experiment.ftmw", trim=(26500, 40000))
    >>> # Estimate noise with default parameters
    >>> noise_result = ftmw.estimate_noise("experiment.ftmw")
    >>> # Override a scatter knob
    >>> noise_result = ftmw.estimate_noise(
    ...     "experiment.ftmw", settings=NoiseSettings(window_mhz=120.0)
    ... )
    """
    try:
        # Delegate to Pipeline class for consistent behavior
        pipeline = Pipeline.open(file_path)
        return pipeline.estimate_noise(
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
    show_noise_points: Optional[bool] = None,
    interactive: bool = True,
    output_file: Optional[Union[str, Path]] = None,
    **plot_kwargs: Any,
) -> Any:
    """
    Create noise estimation diagnostic visualization.

    This function creates diagnostic plots showing the spectrum, the noise
    points, and the per-bin σ estimate (with 3×/5×σ reference levels),
    equivalent to Pipeline.visualize_noise(). Requires Stage 2 (noise
    estimation) completion.

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
    show_noise_points : bool, optional
        Whether to highlight noise points (default: True)
    interactive : bool, default True
        Whether to create interactive plots
    output_file : str or Path, optional
        If provided, save plot to this file
    **plot_kwargs
        Additional plotting parameters

    Returns
    -------
    matplotlib.figure.Figure
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
            show_noise_points=show_noise_points,
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
    *,
    shape: str = "lorentzian",
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
) -> TauCalibrationResult:
    """Run the Stage 2b data-driven tau calibration, equivalent to
    :meth:`Pipeline.calibrate_tau`.

    Requires Stages 0-2 completed. ``shape`` selects the decay model and its
    persistence group: ``"lorentzian"`` (pure-exponential,
    ``/stage2b_tau_calibration``) or ``"gaussian"`` (pure-Gaussian envelope τ_G,
    ``/stage2b_tau_G_calibration``); the two are independent and can coexist on
    one ``.ftmw`` file, consumed by the matching-shape Stage 5 fit. Settings
    resolve through the chain (``settings`` / ``preset`` > persisted > hard
    default); ``settings=`` and ``preset=`` may be combined -- the ``settings``
    bundle is the explicit override that outranks the persisted record, while the
    ``preset`` seeds only the fields neither the explicit layer nor the persisted
    record has fixed (the persisted record outranks the preset, per D11).
    Individual knobs are set on a :class:`TauCalibrationSettings` instance or a
    YAML preset's ``stage2b:`` block. The resolved settings are stamped to the
    shared ``processing_parameters/stage2b_tau`` block so a no-arg follow-up call
    reproduces the same recipe.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.calibrate_tau(
            shape=shape,
            settings=settings,
            preset=preset,
        )
    except Exception as e:
        logger.error(f"Failed to calibrate tau for {file_path}: {e}")
        raise


def load_tau_calibration(
    file_path: Union[str, Path], *, shape: str = "lorentzian"
) -> TauCalibrationResult:
    """Load the persisted Stage 2b :class:`TauCalibrationResult` for ``shape``."""
    try:
        return Pipeline.open(file_path).load_tau_calibration(shape=shape)
    except Exception as e:
        logger.error(f"Failed to load tau calibration from {file_path}: {e}")
        raise


def calibrate_timebase(
    file_path: Union[str, Path],
    *,
    clocks: Optional[Any] = None,
    kappa_sys: Optional[float] = None,
    snr_min: Optional[float] = None,
) -> TimebaseCalibrationResult:
    """Measure the scope-timebase scale error ``eps``, equivalent to
    :meth:`Pipeline.calibrate_timebase`.

    Requires Stage 0 (the raw FID). Demodulates the active FID at the
    Rb-locked clock spur lattice and fits the shared fractional scale error
    ``eps``. The clock declaration comes from the explicit ``clocks`` argument,
    else the persisted Stage 5 ``spur.clocks``; a non-empty declaration with at
    least one locked source is required. Persists ``eps`` to
    ``/timebase_calibration``; measuring ``eps`` is the whole job -- applying it
    is out of scope.

    ``eps`` scales the **baseband** frequency (it is a digitizer-clock error),
    so the molecular-frame correction is about the probe rather than a rescale
    of the absolute frequency::

        f_corr = probe + (f_raw - probe) / (1 + eps)

    Applying ``f_raw / (1 + eps)`` instead is wrong by exactly
    ``probe_freq * eps / (1 + eps)``, a constant sideband-independent offset.
    See :class:`~ftmwpipeline.fitting.timebase_calibration.TimebaseCalibrationResult`.
    """
    try:
        return Pipeline.open(file_path).calibrate_timebase(
            clocks=clocks,
            kappa_sys=kappa_sys,
            snr_min=snr_min,
        )
    except Exception as e:
        logger.error(f"Failed to calibrate timebase for {file_path}: {e}")
        raise


def load_timebase_calibration(
    file_path: Union[str, Path],
) -> TimebaseCalibrationResult:
    """Load the persisted :class:`TimebaseCalibrationResult`."""
    try:
        return Pipeline.open(file_path).load_timebase_calibration()
    except Exception as e:
        logger.error(f"Failed to load timebase calibration from {file_path}: {e}")
        raise


def frequency_calibration(file_path: Union[str, Path]) -> CalibrationStamp:
    """Return the frequency calibration ``file_path`` is under right now.

    Equivalent to :meth:`Pipeline.frequency_calibration`.  Read-only,
    non-mutating, and answerable at any stage -- including on a file that has
    only been imported.  The returned
    :class:`~ftmwpipeline.core.calibration.CalibrationStamp` carries the
    derived ``state``, the ``epsilon`` / ``sigma_epsilon`` that will be
    applied, the declared ``sigma_floor_khz``, and the ``probe_freq_mhz`` /
    ``sideband`` the calibrated frame is defined against -- everything needed
    to label an axis honestly or to move a frequency between the ``"raw"`` and
    ``"calibrated"`` frames (:data:`~ftmwpipeline.core.curation.Frame`).

    This is the state *derived* from the clock declaration and the live
    ``timebase_calibration``, not a persisted copy: use it rather than
    :func:`load_timebase_calibration` (which requires a measurement to exist
    and cannot answer the state question) or ``FinalProducts``' stamp (which
    describes a past Stage 6 build).
    """
    return Pipeline.open(file_path).frequency_calibration()


def recommend_shape(
    file_path: Union[str, Path],
    *,
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

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); ``settings=`` and ``preset=`` may be combined -- the
    ``settings`` bundle is the explicit override that outranks the persisted
    record, while the ``preset`` seeds only the fields neither the explicit
    layer nor the persisted record has fixed (the persisted record outranks the
    preset, per D11).
    """
    try:
        return Pipeline.open(file_path).recommend_shape(
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
    shape: str = "lorentzian",
) -> Any:
    """2D STFT magnitude heatmap, equivalent to
    :meth:`Pipeline.visualize_tau_heatmap`. ``shape`` selects the pure-exp or
    Gaussian calibration group."""
    try:
        return Pipeline.open(file_path).visualize_tau_heatmap(
            output_file=output_file,
            interactive=interactive,
            figsize=figsize,
            shape=shape,
        )
    except Exception as e:
        logger.error(f"Failed to visualize tau heatmap for {file_path}: {e}")
        raise


def visualize_tau_distribution(
    file_path: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    figsize: Optional[tuple] = None,
    shape: str = "lorentzian",
) -> Any:
    """tau-distribution analysis panel, equivalent to
    :meth:`Pipeline.visualize_tau_distribution`. ``shape`` selects the pure-exp
    or Gaussian calibration group."""
    try:
        return Pipeline.open(file_path).visualize_tau_distribution(
            output_file=output_file,
            interactive=interactive,
            figsize=figsize,
            shape=shape,
        )
    except Exception as e:
        logger.error(f"Failed to visualize tau distribution for {file_path}: {e}")
        raise


# =============================================================================
# Stage 3: Peak Detection Functions
# =============================================================================


def detect_peaks(
    file_path: Union[str, Path],
    *,
    settings: Optional[PeakDetectionSettings] = None,
    preset: Optional[str] = None,
) -> List[Peak]:
    """Detect and classify peaks (Stage 3), equivalent to Pipeline.detect_peaks().

    Requires Stage 1 (FT) and Stage 2 (noise). Two-pass detection scores on
    the active FT (built from the persisted Stage 1 settings, including its
    frequency trim range); peaks are reported on that active grid with SNR
    measured against the persisted Stage 2 noise.  There is no per-Stage-3
    trim or zpf.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); pass ``settings=`` to drive detection from a
    :class:`PeakDetectionSettings` dataclass, or ``preset=NAME_OR_PATH`` to load
    from packaged YAML. They may be combined: a ``settings`` bundle is the
    explicit override that outranks the persisted record, while a ``preset``
    seeds only the fields neither the explicit layer nor the persisted record
    has fixed (the persisted record outranks the preset, per D11). Set
    individual knobs via
    ``settings=PeakDetectionSettings(...)`` or a YAML preset's ``stage3:`` block.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    settings : PeakDetectionSettings, optional
        Bundle of Stage 3 knobs; fields left ``None`` fall through the
        resolution chain. May be combined with ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage3:`` block.
        Seeds the preset layer beneath the persisted record; may be combined
        with ``settings``.

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
    *,
    settings: Optional[WindowPlanningSettings] = None,
    preset: Optional[str] = None,
) -> WindowPlan:
    """Assign analysis windows (Stage 4), equivalent to Pipeline.assign_windows().

    Requires Stage 3 (peak detection). Turns the promoted Stage 3 peaks into a
    fit plan -- a set of disjoint analysis windows, each annotated with the
    peaks to fit freely, the strong out-of-band lines whose leakage is carried
    frozen, and a fit dependency order. Stage 4 is purely structural; the plan
    is persisted to the .ftmw file.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); pass ``settings=`` to drive window planning from a
    :class:`WindowPlanningSettings` dataclass, or ``preset=NAME_OR_PATH`` to
    load from packaged YAML. They may be combined: a ``settings`` bundle is the
    explicit override that outranks the persisted record, while a ``preset``
    seeds only the fields neither the explicit layer nor the persisted record
    has fixed (the persisted record outranks the preset, per D11). Set
    individual knobs via
    ``settings=WindowPlanningSettings(...)`` or a YAML preset's ``stage4:`` block.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    settings : WindowPlanningSettings, optional
        Bundle of Stage 4 knobs; fields left ``None`` fall through the
        resolution chain. May be combined with ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage4:`` block.
        Seeds the preset layer beneath the persisted record; may be combined
        with ``settings``.

    Returns
    -------
    WindowPlan
        The fit plan: disjoint windows, dependency DAG, topological order,
        parallel batches, parameters and diagnostics.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.assign_windows(
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
    *,
    shape: Optional[str] = None,
    tau_maj_override_us: Optional[float] = None,
    sigma_tau_override_us: Optional[float] = None,
    settings: Optional[StageFitSettings] = None,
    preset: Optional[str] = None,
    jobs: Optional[int] = None,
) -> SpectrumFit:
    """Fit each Stage 4 window's lines (Stage 5), equivalent to Pipeline.fit_peaks().

    Requires Stage 4 (window assignment). Drives the conservative add-one-peak
    loop over each window with the shared per-window decay ``tau`` and the
    frozen-contributor model, then the residual edge-coherence handshake (local
    thaw + structural replan). The fit runs on the active-portion FT (computed
    on demand from the FID + persisted Stage 1 settings), so per-bin statistics
    are independent and reduced chi-squared / F-test / AIC are calibrated as
    written. The persistent :class:`SpectrumFit` -- per-window
    :class:`FittingResult` s, the merged global fitted-peak list, the thaw /
    replan histories, and the parameters used -- is written to
    ``/stage5_fitting``.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    recommended > hard default); pass ``settings=`` to drive the fit from a
    :class:`StageFitSettings` dataclass, or ``preset=NAME_OR_PATH`` to load from
    packaged YAML. They may be combined: a ``settings`` bundle outranks a value
    persisted in the ``.ftmw`` while a ``preset`` .yml seeds only the fields
    neither the explicit layer nor the persisted record has fixed (the persisted
    record outranks the preset, per D11).

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    shape : {"lorentzian", "gaussian"}, optional
        Time-domain envelope of the per-line model, kept as a first-class
        convenience argument. ``"lorentzian"`` uses ``exp(-t/τ)``;
        ``"gaussian"`` uses ``exp(-(t/τ_G)²)`` and consumes the Stage 2b τ_G
        calibration (``calibrate_tau(shape="gaussian")``) in place of the
        pure-exp variant for the bidirectional τ anchoring penalty. ``None``
        falls through to the resolved settings (preset / persisted / default).
    tau_maj_override_us, sigma_tau_override_us : float, optional
        Atomic-pair manual override for the Stage 2b tau calibration, kept as
        explicit arguments (an A/B escape hatch crossing a stage boundary, not
        a fit knob). When both are supplied (positive), they replace any
        persisted Stage 2b result for this fit. Supplying only one of the pair
        raises ``ValueError``.
    settings : StageFitSettings, optional
        Bundle of Stage 5 knobs; fields left ``None`` fall through the
        resolution chain. Resolves at the explicit override layer (outranks the
        persisted record). May be combined with ``preset``.
    preset : str, optional
        Bare preset name (e.g. ``"defaults"``) or a path to a YAML
        file carrying a ``stage5:`` block. Seeds the preset layer beneath the
        persisted record; may be combined with ``settings``.
    jobs : int, optional
        Worker-pool size for the cross-window parallel fit. ``None`` (the
        default) resolves the pool from the ``FTMW_MAX_WORKERS`` environment
        variable, falling back to ``cpu_count() - 2``; ``1`` forces a sequential
        fit. The fit result is byte-identical regardless of the worker count.

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
        pipeline = Pipeline.open(file_path)
        return pipeline.fit_peaks(
            shape=shape,
            tau_maj_override_us=tau_maj_override_us,
            sigma_tau_override_us=sigma_tau_override_us,
            settings=settings,
            preset=preset,
            jobs=jobs,
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


def get_candidate_ledger(
    file_path: Union[str, Path],
    window_id: Optional[int] = None,
    *,
    bar: float = DEFAULT_DISPLAY_BAR,
) -> List[LedgerCandidate]:
    """Derive the Stage 6 candidate ledger from the persisted Stage 5 fit.

    Returns revivable candidates from the conservative add-loop and rescue
    records.  Equivalent to :meth:`Pipeline.candidate_ledger`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
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
    return Pipeline.open(file_path).candidate_ledger(window_id=window_id, bar=bar)


def review_edit(
    file_path: Union[str, Path],
    window_id: int,
    *,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
) -> RefitWindowResult:
    """User-directed single-window refit (Stage 6 ``review edit``).

    Re-fits ``window_id`` from the persisted Stage 5 fit using the production
    NLS primitive, applying ``add``/``remove`` edits.  User-added peaks carry
    ``origin="user"`` and survive the HDF5 round-trip; removed peaks are
    excluded from the refit and will not be re-added by the rescue pass.
    Equivalent to :meth:`Pipeline.review_edit`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The window to refit.
    add :
        Molecular frequencies (MHz) of peaks to add.
    remove :
        Molecular frequencies (MHz) of fitted peaks to remove.
    snap_tol_mhz :
        Snap tolerance for ``add``/``remove`` (MHz; default
        :data:`~ftmwpipeline.core.curation.REFIT_SNAP_TOL_MHZ`, 50 kHz).
    frame :
        The frame ``add``/``remove`` are expressed in: ``"raw"`` or
        ``"calibrated"``; converted to raw before any snapping. Omitting it
        defaults to ``"raw"`` and is an error on a ``self_calibrated`` file
        when ``add`` or ``remove`` is non-empty (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks
        (both frames).

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_edit(
        window_id, add=add, remove=remove, snap_tol_mhz=snap_tol_mhz, frame=frame
    )


def review_acknowledge_environment(
    file_path: Union[str, Path], *, reason: str = ""
) -> EnvironmentAckResult:
    """Accept an analysis-epoch mismatch so Stage 6 editing can proceed.

    Equivalent to :meth:`Pipeline.review_acknowledge_environment`.  The
    acknowledgement is persisted in the file, so a curated result carries the
    fact that its curation crossed an epoch boundary.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    reason :
        Optional free-text note stored alongside the acknowledgement.

    Returns
    -------
    dict
        The acknowledged environment, the fit's environment, and whether a
        mismatch actually existed.
    """
    return Pipeline.open(file_path).review_acknowledge_environment(reason=reason)


def review_create(
    file_path: Union[str, Path],
    anchor_mhz: float,
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
) -> CreateWindowResult:
    """Install a fit window covering ``anchor_mhz`` (Stage 6 ``review create``).

    For a line the automatic pass left with no window at all.  Purely
    structural and purely additive: no existing window is renumbered, re-fit,
    or thawed, and adding the line to the new window is a separate
    :func:`review_edit` decision.  Equivalent to
    :meth:`Pipeline.review_create`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    anchor_mhz :
        Molecular frequency (MHz) the window must cover.
    snap_tol_mhz :
        Snap tolerance forwarded to the fit core (MHz; default
        :data:`~ftmwpipeline.core.curation.REFIT_SNAP_TOL_MHZ`).
    frame :
        The frame ``anchor_mhz`` is expressed in; converted to raw before
        installing the window. Omitting it is an error on a
        ``self_calibrated`` file (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    CreateWindowResult
        The installed window's id, mode (``"created"`` / ``"widened"``),
        extent, and contributor count (both frames).

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_create(
        anchor_mhz, snap_tol_mhz=snap_tol_mhz, frame=frame
    )


def review_merge(
    file_path: Union[str, Path],
    window_id: int,
    peaks: Sequence[float],
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
) -> RefitWindowResult:
    """Collapse ≥2 fitted peaks in a window into one (Stage 6 ``review merge``).

    Equivalent to :meth:`Pipeline.review_merge`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The window containing the peaks to merge.
    peaks :
        Molecular frequencies (MHz) of the peaks to collapse (≥2).
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping.
    frame :
        The frame ``peaks`` is expressed in. Omitting it is an error on a
        ``self_calibrated`` file (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks
        (both frames).

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_merge(
        window_id, peaks, snap_tol_mhz=snap_tol_mhz, frame=frame
    )


def review_split(
    file_path: Union[str, Path],
    window_id: int,
    peak: float,
    *,
    into: int = 2,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
) -> RefitWindowResult:
    """Replace one fitted peak with ``into`` peaks (Stage 6 ``review split``).

    Equivalent to :meth:`Pipeline.review_split`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The window containing the peak to split.
    peak :
        Molecular frequency (MHz) of the peak to split.
    into :
        Number of replacement peaks (≥2, default 2).
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping.
    frame :
        The frame ``peak`` is expressed in. Omitting it is an error on a
        ``self_calibrated`` file (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks
        (both frames).

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_split(
        window_id, peak, into=into, snap_tol_mhz=snap_tol_mhz, frame=frame
    )


def review_run(
    file_path: Union[str, Path],
    *,
    bar: float = DEFAULT_DISPLAY_BAR,
    attention_candidate_evidence: float = DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    sigma_floor_khz: Optional[float] = None,
) -> "ReviewRunResult":
    """Build or refresh the Stage 6 curation layer and final-products table.

    Equivalent to :meth:`Pipeline.review_run`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    bar :
        Display bar for the candidate-bearing attention reason.
    attention_candidate_evidence :
        Evidence threshold above which a candidate-bearing window flags
        (stiffer than ``bar``; keeps the attention surface actionable).
    sigma_floor_khz :
        When given, persist this user-declared accuracy floor (kHz) and fold it
        into the budget; ``None`` keeps the persisted floor unchanged.

    Returns
    -------
    ReviewRunResult
        Total window count, attention count, and per-kind breakdown.

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_run(
        bar=bar,
        attention_candidate_evidence=attention_candidate_evidence,
        sigma_floor_khz=sigma_floor_khz,
    )


def set_sigma_floor(file_path: Union[str, Path], sigma_floor_khz: float) -> None:
    """Declare the systematic frequency-accuracy floor (kHz), persisted in-file.

    Equivalent to :meth:`Pipeline.set_sigma_floor`.  Stores the floor as
    file-level provenance so any reported ``sigma_f`` is reproducible from the
    record alone.  Run :func:`review_run` afterwards to fold it into the budget.
    """
    Pipeline.open(file_path).set_sigma_floor(sigma_floor_khz)


def get_final_products(file_path: Union[str, Path]) -> Optional[FinalProducts]:
    """Return the persisted Stage 6 calibrated final-products table, or ``None``.

    Equivalent to :meth:`Pipeline.final_products`.  Read-only.
    """
    return Pipeline.open(file_path).final_products()


def report_table(
    file_path: Union[str, Path],
    *,
    fmt: str = "csv",
    output: Optional[Union[str, Path]] = None,
    catalog: Optional[Union[str, Path]] = None,
    catalog_n_sigma: float = 3.0,
) -> str:
    """Render the calibrated final-products table (report Level 1).

    Equivalent to :meth:`Pipeline.report_table`.  Serializes the persisted
    final-products table to ``csv`` / ``json`` / ``latex``; renders the
    persisted record (does not recompute).  Requires ``review_run`` to have
    built the table.  Pass ``catalog`` to proximity-flag each line against a
    frequency catalog (label echo only, never an assignment).
    """
    return Pipeline.open(file_path).report_table(
        fmt=fmt, output=output, catalog=catalog, catalog_n_sigma=catalog_n_sigma
    )


def report_run(
    file_path: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    windows: str = "all",
    emit_table: bool = True,
    emit_html: bool = True,
    table_format: str = "csv",
    scope: str = "full",
    catalog: Optional[Union[str, Path]] = None,
    catalog_n_sigma: float = 3.0,
    jobs: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    """Write the default Stage 6 deliverables: the L1 table + the L3 report.

    Equivalent to :meth:`Pipeline.report_run`.  Writes the Level-1
    final-products table (``<stem>_lines.csv``) and the self-contained Level-3
    HTML report with every window folded in (``<stem>_report.html``) into
    *output_dir* (the current working directory when omitted).  Either
    artifact can be suppressed (``emit_table`` /
    ``emit_html``); the HTML content follows *scope* (``"full"`` folds in every
    window, ``"summary"`` keeps the index + methods only).  Renders the persisted
    record (does not recompute).  Requires ``review_run`` to have built the
    final-products table.  Pass ``catalog`` to add proximity-match
    cross-references (label echo only, never an assignment).  ``jobs`` sets the
    worker-pool size for the per-window figure rendering (``None`` resolves it
    from the ``FTMW_MAX_WORKERS`` environment variable, falling back to
    ``cpu_count() - 2``; ``1`` renders sequentially).
    Returns ``{"table": <path|None>, "html": <path|None>}``.
    """
    return Pipeline.open(file_path).report_run(
        output_dir=output_dir,
        windows=windows,
        emit_table=emit_table,
        emit_html=emit_html,
        table_format=table_format,
        scope=scope,
        catalog=catalog,
        catalog_n_sigma=catalog_n_sigma,
        jobs=jobs,
    )


def report_diff(
    file_path: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    dpi: int = 110,
) -> str:
    """Render the post-curation before/after diff report; return its path.

    Equivalent to :meth:`Pipeline.report_diff`.  Compares the automatic-fit
    baseline snapshot with the current curated fit and writes a self-contained
    ``<stem>_diff.html`` with a side-by-side panel for every window that differs
    materially -- both the windows edited directly and the dependents the
    contributor-edit cascade changed -- so the changes can be evaluated before
    they are committed.  When no curation edit has been made, a report stating
    that is written instead.  Read-only; never recomputes the fit.
    """
    return Pipeline.open(file_path).report_diff(output_dir=output_dir, dpi=dpi)


def run_pipeline(
    source: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    *,
    trim: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Drive a raw *source* through every pipeline stage end-to-end.

    Equivalent to :meth:`Pipeline.build`.  Imports *source*, then runs FT ->
    noise -> tau -> peaks -> windows -> fit -> timebase -> review (and, with
    ``report=True``, the report) in order, showing live per-stage progress.
    *trim* (the active-band FT range, MHz) is required; ``output`` is the
    destination ``.ftmw`` (derived from *source* if omitted).

    Per-stage behavior is tuned with override dicts forwarded to each stage
    (``ft_params``, ``noise_params``, ``tau_params``, ``peak_params``,
    ``window_params``, ``fit_params``, ``timebase_params``, ``review_params``,
    ``report_params``) plus an optional ``preset`` name; ``detect_start`` /
    ``calibrate`` / ``clocks`` gate the optional stages (timebase is non-fatal --
    it warns and skips when no clock declaration is resolvable), ``report`` /
    ``report_output_dir`` emit the report, and ``progress=False`` silences the
    display.  Returns the structured run result (``pipeline_file``, ``status``,
    ``completed_stages``, ``failed_stage``, ``error``, ``timebase``, ``report``,
    ``elapsed_s``); stops at the first failing stage.
    """
    return Pipeline.build(source, trim=trim, output=output, **kwargs)


def review_accept(
    file_path: Union[str, Path],
    window_id: int,
    *,
    candidate_freq: Optional[float] = None,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
) -> Optional[RefitWindowResult]:
    """Accept a window as-is or accept a specific revived candidate.

    Equivalent to :meth:`Pipeline.review_accept`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    window_id :
        The window to accept.
    candidate_freq :
        When given, accept by adding this molecular frequency (MHz) as a
        new peak.
    snap_tol_mhz :
        Snap tolerance for ``candidate_freq`` (MHz; default
        :data:`~ftmwpipeline.core.curation.REFIT_SNAP_TOL_MHZ`, 50 kHz).
        Ignored when accepting a window as-is.
    frame :
        The frame ``candidate_freq`` is expressed in. Irrelevant when
        ``candidate_freq`` is ``None``. Omitting it while ``candidate_freq``
        is given is an error on a ``self_calibrated`` file (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    RefitWindowResult or None
        ``None`` when accepting as-is; the refit result when
        ``candidate_freq`` is given (both frames).

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_accept(
        window_id,
        candidate_freq=candidate_freq,
        snap_tol_mhz=snap_tol_mhz,
        frame=frame,
    )


def review_apply(
    file_path: Union[str, Path],
    curation_path: Union[str, Path],
    *,
    dry_run: bool = False,
    frame: Optional[Frame] = None,
) -> CurationApplyResult:
    """Apply a curation file of batched review edits.

    Equivalent to :meth:`Pipeline.review_apply`.  Coalesces add/remove rows on
    one window into a single refit (merge/split/accept stand alone), then
    applies the resolved plan as one batch: one shared fit context, one combined
    cascade, one persist.  Cross-window execution order is canonical, so the
    final state does not depend on the file's row order.  With ``dry_run`` the
    resolved plan and frequency-resolution warnings are returned without
    modifying the file.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    curation_path :
        Path to the curation CSV to apply.
    dry_run :
        Preview the resolved plan without writing (default ``False``).
    frame :
        The frame every frequency in the curation file is expressed in --
        applies uniformly (no per-row frame column). The file's own optional
        ``# frame: ...`` / ``# epsilon: ...`` header takes precedence over
        (or must agree with) this argument; omitting both is an error on a
        ``self_calibrated`` file when the file carries any frequency (see
        :data:`~ftmwpipeline.core.curation.Frame`).

    Returns
    -------
    CurationApplyResult

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_apply(
        curation_path, dry_run=dry_run, frame=frame
    )


def review_preview(
    file_path: Union[str, Path],
    curation_path: Union[str, Path],
    *,
    frame: Optional[Frame] = None,
) -> ReviewPreviewResult:
    """Run a curation file's resolved plan to completion in memory and report
    the fitted outcome, without writing anything.

    Equivalent to :meth:`Pipeline.review_preview`. Shares every hook
    :func:`review_apply` uses -- the same parse/resolve/frame-convert
    prologue, the same per-action appliers, the same one combined cascade --
    except the batch is never persisted: no undo baseline is taken and
    nothing is written to *file_path*. Epoch-gated exactly like
    :func:`review_apply`, except a plan of entirely bare ``accept`` rows,
    which touches no fit and so is not gated (matching the live apply).

    The result is keyed by window id, read after the cascade -- not
    per-action -- and reports final-product numbers (calibrated frequency,
    the three-term sigma budget), identical to what a subsequent
    :func:`review_apply` of the same plan would persist.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    curation_path :
        Path to the curation CSV to preview.
    frame :
        The frame every frequency in the curation file is expressed in --
        same rules as :func:`review_apply`.

    Returns
    -------
    ReviewPreviewResult

    Requires Stage 5 completed.
    """
    return Pipeline.open(file_path).review_preview(curation_path, frame=frame)


def review_log(file_path: Union[str, Path]) -> List[DecisionLogEntry]:
    """Return the persisted Stage 6 decision log (read-only, execution order).

    Equivalent to :meth:`Pipeline.review_log`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.

    Returns
    -------
    list of DecisionLogEntry
    """
    return Pipeline.open(file_path).review_log()


def review_undo(
    file_path: Union[str, Path],
    ids: Sequence[int],
    *,
    dry_run: bool = False,
) -> UndoResult:
    """Undo recorded decisions by id, replaying the rest from baseline.

    Equivalent to :meth:`Pipeline.review_undo`.  Restores the automatic Stage 5
    fit and re-applies every surviving decision (ids are renumbered afterward);
    ``dry_run`` previews without writing.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    ids :
        Decision ids (from :func:`review_log`) to undo.
    dry_run :
        Preview without mutating (default ``False``).

    Returns
    -------
    UndoResult
    """
    return Pipeline.open(file_path).review_undo(ids, dry_run=dry_run)


def get_review_status(file_path: Union[str, Path]) -> Stage6Review:
    """Load the Stage 6 review state from *file_path*, or return an empty one.

    Equivalent to :meth:`Pipeline.review_status`.  Read-only; safe to call
    before ``review run``.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.

    Returns
    -------
    Stage6Review
        The persisted per-window statuses and decision log.
    """
    return Pipeline.open(file_path).review_status()


def rank_windows(
    file_path: Union[str, Path],
    by: str,
    *,
    top: Optional[int] = None,
) -> List["RankedWindow"]:
    """Rank fit windows by a persisted per-window statistic (read-only).

    Equivalent to :meth:`Pipeline.rank_windows`.  On-demand exploration
    decoupled from the attention flags: ranks all windows worst-first by ``by``
    (``min-snr``, ``max-vif``, ``chi2r``, ``candidate-evidence``,
    ``edge-distance``, ``spur-proximity``, ``merged-chi2r``).

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    by :
        Metric name (``_`` and ``-`` interchangeable).
    top :
        Return at most this many windows; ``None`` returns all.
    """
    return Pipeline.open(file_path).rank_windows(by, top=top)


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
            interactive=interactive,
            output_file=output_file,
        )
    except Exception as e:
        logger.error(f"Failed to create fit visualization for {file_path}: {e}")
        raise


def show_fit(
    file_path: Union[str, Path],
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
    rescue: bool = False,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    interactive: bool = False,
) -> Dict[str, Any]:
    """Show the Stage 5 fit, equivalent to ``Pipeline.show_fit()`` and the CLI
    ``fit show`` command.

    With no selector, returns the spectrum-wide overview. Selectors
    (``window_ids`` / ``freqs`` / ``random_n`` / ``top_snr`` / ``all_windows``)
    compose as a union and produce one consolidated per-window detail figure
    each. With ``output_dir`` each detail figure is written there. With
    ``rescue`` an extra residual-rescue summary figure is produced per window
    (no re-fit): data/model, the final residual with each round's nominations,
    the chi-squared trajectory, and the peak budget. Returns
    ``{"mode", "window_ids", "figures", "paths", "log"}``. Requires Stage 5.
    """
    try:
        pipeline = Pipeline.open(file_path)
        return pipeline.show_fit(
            window_ids=window_ids,
            freqs=freqs,
            random_n=random_n,
            random_seed=random_seed,
            top_snr=top_snr,
            all_windows=all_windows,
            output_dir=output_dir,
            show_audit=show_audit,
            apodize=apodize,
            apodize_us=apodize_us,
            rescue=rescue,
            figsize=figsize,
            title=title,
            interactive=interactive,
        )
    except Exception as e:
        logger.error(f"Failed to show fit for {file_path}: {e}")
        raise


# =============================================================================
# Utility Functions
# =============================================================================


# =============================================================================
# Read-only access to persisted data
# =============================================================================
#
# These three delegate straight to the shared ``_internal.read_impl`` core
# rather than through ``Pipeline.open``, unlike the rest of this module. That
# is deliberate: ``Pipeline.open`` also validates the provenance record, which a
# read-only column tap has no business demanding, and routing through it would
# leave the CLI (which calls the core directly, as every CLI module does)
# accepting files this surface rejected. One gate, three interfaces.


def read_table(
    file_path: Union[str, Path],
    table: str,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read one persisted table as bulk columns, equivalent to
    :meth:`Pipeline.read_table`.

    The cheap counterpart to :func:`load_peaks` / :func:`load_windows` /
    :func:`load_fit`. Those rebuild the complete persisted record -- audit
    trails, thaw and rescue histories, fixed-contributor records, covariance
    blocks -- because a curator editing it needs all of it. This reads only the
    columns asked for, as whole datasets, and reconstructs no object graph.
    Nothing is recomputed; the file is opened read-only.

    Parameters
    ----------
    file_path : str or Path
        Path to .ftmw pipeline file.
    table : str
        One of :func:`read_tables`' keys; hyphens and underscores are
        equivalent. Stage 2b offers ``tau_bands`` / ``tau_thirds`` /
        ``tau_contributors`` / ``tau_spurs``, mirrored under a ``tau_g_`` prefix
        for the Gaussian twin; Stage 3 ``peaks``; Stage 4 ``windows`` plus
        ``window_free_peaks`` / ``window_contributors`` in long form; Stage 5
        ``fit_peaks`` / ``fit_windows`` plus the decision record
        (``fit_audit``, ``fit_doublets``, ``fit_thaw``, ``fit_replans``,
        ``fit_rescues``).
    columns : sequence of str, optional
        Columns to read, in the order wanted; ``None`` reads all of them.
        Reading only what you need is what makes this cheap.

    Notes
    -----
    The event-log tables (``fit_audit``, ``fit_doublets``, ``fit_thaw``,
    ``fit_replans``, ``fit_rescues``) read JSON-encoded records, because that is
    how the fit persists its narrative; reaching them means parsing, and there
    is no narrower path. They still cost far less than a full :func:`load_fit`,
    but they are not the whole-dataset reads the rest of this surface is.

    Returns
    -------
    dict
        ``{column_name: numpy array}``, all of equal length. Values keep the
        persisted sentinels (NaN for an absent float, ``-1`` for an absent id,
        tri-state small ints for the knockout/tau flags) rather than the
        ``None`` the full loaders substitute.

    Raises
    ------
    ValueError
        If the table or a column name is unknown, or the stage that produces
        the table has not been run.

    Examples
    --------
    >>> import ftmwpipeline.api as ftmw
    >>> cols = ftmw.read_table(
    ...     "experiment.ftmw",
    ...     "fit_peaks",
    ...     columns=["frequency_mhz", "decay_rate", "shape"],
    ... )
    >>> cols["frequency_mhz"][:3]
    """
    try:
        return read_table_impl(file_path, table, columns)
    except Exception as e:
        logger.error(f"Failed to read table {table!r} from {file_path}: {e}")
        raise


def read_tables(file_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """List the readable tables and their columns, equivalent to
    :meth:`Pipeline.read_tables`.

    Returns one entry per table name accepted by :func:`read_table`, each with
    ``available`` (whether the producing stage has been run), ``n_rows``,
    ``columns``, and the backing HDF5 ``group``. Reads group attributes only.
    """
    try:
        return read_tables_impl(file_path)
    except Exception as e:
        logger.error(f"Failed to list readable tables for {file_path}: {e}")
        raise


def read_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read the file's cheap top-level scalars, equivalent to
    :meth:`Pipeline.read_metadata`.

    Group attributes only -- nothing is deserialized and no window is
    traversed. Keys are dotted (``file.*``, ``source.*``, ``fid.*``,
    ``stage3.*``, ``stage4.*``, ``stage5.*``, ``timebase.*``) and a section is
    simply absent when its stage has not been run, so read with ``.get()``.

    ``stage5.acquisition_us`` is here: the active-FT acquisition length the
    Fourier resolution element ``1 / acquisition_us`` follows from, reachable
    without deserializing the fit that carries it.
    """
    try:
        return read_metadata_impl(file_path)
    except Exception as e:
        logger.error(f"Failed to read metadata from {file_path}: {e}")
        raise


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
        - 'stage_environments', 'last_written_with', 'environment_drift':
          the per-stage analysis-environment record and whether the file's
          own stages disagree
        - 'current_environment', 'runtime_environment_drift': the running
          interpreter and how the file's stamps differ from it

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
    1. ftmw.compute_ft("experiment.ftmw", trim=(26500, 40000))
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
                    f'1. ftmw.compute_ft("{Path(file_path).name}", trim=(26500, 40000))',
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
# Parameter-scan surface
# =============================================================================


def scan_list(
    selector: Optional[str] = None,
    *,
    include_advanced: bool = False,
) -> Tuple[Any, ...]:
    """List the registered tunable knobs, equivalent to
    :meth:`Pipeline.scan_list`.

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
    return Pipeline.scan_list(selector, include_advanced=include_advanced)


def scan_run(
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
    :meth:`Pipeline.scan_run`.

    Re-runs the knob's stage for each grid value on a working copy of
    ``file_path`` (the input is never mutated) and returns a ``SweepResult``
    with the table rows, a CSV path, an optional plot, a best-effort
    recommendation, and instructions for applying the chosen value.

    Parameters
    ----------
    file_path : str or Path
        A ``.ftmw`` already built through the knob's upstream stage.
    knob : str
        Dotted knob path (see :func:`scan_list`), e.g.
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
        return pipeline.scan_run(
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


def scan_all(
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
    to :meth:`Pipeline.scan_all`.

    A convenience over :func:`scan_run` for reviewing a whole stage / sub-block
    at once instead of driving knobs one-by-one. ``selector`` filters by
    dotted-path prefix (e.g. ``"stage2b"`` / ``"stage2b.gaussian"``) just like
    :func:`scan_list`; ``include_advanced`` adds the advanced-tier knobs. Each
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
        return pipeline.scan_all(
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
    :func:`scan_list`'s tunable-knob listing.

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
    file never carries results inconsistent with its settings. The retired
    apodization knobs (``zpf`` / ``expf_us`` / ``window_function``) no longer
    exist -- the FT is unconditionally unapodized and native-length.

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
