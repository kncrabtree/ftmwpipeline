"""
Shared implementation for Stage 2: Noise Estimation.

This module contains the core implementation functions for noise estimation
and visualization that are shared between CLI, Pipeline class, and functional 
API interfaces.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import h5py

from ..preprocessing.noise_estimation import (
    estimate_noise_adaptive,
    estimate_noise_scatter,
    NoiseResult,
)
from ..core.noise_settings import (
    NoiseSettings,
    ScatterSubSettings,
    load_preset as load_noise_preset,
    resolve as resolve_noise_settings,
)
from ..io.noise_result_serialization import save_noise_result_to_hdf5, load_noise_result_from_hdf5
from ..io.noise_settings_serialization import (
    load_noise_settings_from_h5,
    save_noise_settings_to_h5,
)
from ..file_manager import open_pipeline_file
from .deprecation import warn_legacy_flag, warn_legacy_kwargs


logger = logging.getLogger(__name__)


def _required(value: Any, name: str) -> Any:
    """Coerce a post-resolve field that must be filled (hard default present)."""
    if value is None:
        raise AssertionError(
            f"resolved NoiseSettings.{name} is None; missing hard default"
        )
    return value


def _build_explicit_from_kwargs(
    *,
    skew_target: Optional[float],
    min_bin_fraction: Optional[float],
    smoothing_window_mhz: Optional[float],
    min_noise_fraction: Optional[float],
) -> NoiseSettings:
    """Bundle the four legacy per-knob kwargs into an explicit-layer settings instance.

    The remaining knobs that the resolver covers
    (``subdivision_threshold``, ``abs_min_bin_size``, ``inc``,
    ``strong_peak_snr``, ``skirt_exclusion_k``, ``max_skirt_exclusion_mhz``)
    are not yet on the public Stage 2 signatures; they flow through
    ``settings=`` / ``preset=`` only.
    """
    explicit = NoiseSettings()
    explicit.binning.min_bin_fraction = min_bin_fraction
    explicit.binning.min_noise_fraction = min_noise_fraction
    explicit.skewness.skew_target = skew_target
    explicit.smoothing.smoothing_window_mhz = smoothing_window_mhz
    return explicit


def _build_scatter_explicit_from_kwargs(
    *,
    window_mhz: Optional[float],
    pedestal_mhz: Optional[float],
    line_k: Optional[float],
    n_iter: Optional[int],
    region_aware: Optional[bool],
    smoothing_mhz: Optional[float],
    smoothing_percentile: Optional[float],
    convolve_mhz: Optional[float],
) -> NoiseSettings:
    """Bundle the scatter per-knob kwargs into an explicit-layer settings instance.

    Sibling of :func:`_build_explicit_from_kwargs` for the scatter (high-pass)
    estimator; populates only the ``scatter`` sub-block so the resolver merges it
    with any preset / persisted layer.
    """
    explicit = NoiseSettings()
    explicit.scatter = ScatterSubSettings(
        window_mhz=window_mhz,
        pedestal_mhz=pedestal_mhz,
        line_k=line_k,
        n_iter=n_iter,
        region_aware=region_aware,
        smoothing_mhz=smoothing_mhz,
        smoothing_percentile=smoothing_percentile,
        convolve_mhz=convolve_mhz,
    )
    return explicit


def compute_noise_estimation_impl(
    file_path: str,
    skew_target: Optional[float] = None,
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
    region_aware: Optional[bool] = None,
    smoothing_mhz: Optional[float] = None,
    smoothing_percentile: Optional[float] = None,
    convolve_mhz: Optional[float] = None,
    settings: Optional[NoiseSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shared implementation for noise estimation from .ftmw pipeline files.
    
    This function implements the Stage 2 workflow:
    1. Load ComplexFT from pipeline file (requires Stage 1)
    2. Merge user parameters with any saved noise parameters
    3. Call estimate_noise_adaptive() with merged parameters
    4. Return structured result with NoiseResult and metadata
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    skew_target : float, optional
        Target skewness for noise identification (default: 0.631)
    min_bin_fraction : float, optional
        Minimum bin size as fraction of total data (default: 1/64)
    smoothing_window_mhz : float, optional
        RMS smoothing window size in MHz
    min_noise_fraction : float, optional
        Minimum fraction of points that must be noise per bin (default: 2/3)
    from_saved_params : bool, default False
        If True, use only saved parameters and ignore user parameters
        
    Returns
    -------
    dict
        Processing results with keys:
        - 'noise_result': NoiseResult object
        - 'complex_ft': ComplexFT object used for estimation
        - 'parameters_used': Dict of actual parameters used
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If Stage 1 dependencies are not met or parameters are invalid
    """
    warn_legacy_kwargs(
        func_name="estimate_noise",
        legacy_kwargs={
            "skew_target": skew_target,
            "min_bin_fraction": min_bin_fraction,
            "smoothing_window_mhz": smoothing_window_mhz,
            "min_noise_fraction": min_noise_fraction,
            "window_mhz": window_mhz,
            "pedestal_mhz": pedestal_mhz,
            "line_k": line_k,
            "n_iter": n_iter,
            "region_aware": region_aware,
            "smoothing_mhz": smoothing_mhz,
            "smoothing_percentile": smoothing_percentile,
            "convolve_mhz": convolve_mhz,
        },
        migration_hint=(
            "use settings=NoiseSettings(...) or preset='name' to drive "
            "Stage 2 from the settings resolver"
        ),
    )
    if from_saved_params:
        warn_legacy_flag(
            func_name="estimate_noise",
            flag_name="from_saved_params",
            migration_hint=(
                "drop from_saved_params=True; a no-kwargs follow-up call "
                "now inherits the persisted stage2_noise settings block "
                "automatically via the resolver"
            ),
        )

    # Compute ComplexFT on-demand using Stage 1 implementation (correct architecture)
    try:
        # Check that Stage 1 parameters are available (Stage 1 dependency)
        with h5py.File(file_path, 'r') as h5f:
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError(
                    "Stage 1 (FT computation) must be completed before noise estimation. "
                    "Run compute_ft() or compute-ft command first."
                )
        
        # Import Stage 1 implementation for on-demand ComplexFT computation
        from .stage1_impl import compute_ft_impl
        
        # Compute ComplexFT using saved Stage 1 parameters (lightweight on-demand computation)
        stage1_result = compute_ft_impl(file_path=file_path)
        
        complex_ft = stage1_result['complex_ft']
        logger.info(f"Computed ComplexFT on-demand with {len(complex_ft.freq_array):,} frequency points")
            
    except Exception as e:
        raise RuntimeError(f"Failed to compute ComplexFT from pipeline file {file_path}: {e}")

    if method not in ("adaptive", "scatter"):
        raise ValueError(
            f"unknown noise estimator method {method!r}; expected "
            "'adaptive' or 'scatter'"
        )

    # The scatter (high-pass) estimator resolves its ``scatter`` sub-block
    # through the same four-layer chain as the adaptive path: explicit per-knob
    # kwargs > preset > persisted > hard default. ``from_saved_params`` is the
    # adaptive-only legacy contract and is not honoured here.
    if method == "scatter":
        if preset is not None and settings is not None:
            raise ValueError(
                "'preset' and 'settings' are alternative ways to populate "
                "the preset layer of the noise-settings chain; pass exactly "
                "one (or override individual fields via explicit kwargs)"
            )
        scatter_explicit = _build_scatter_explicit_from_kwargs(
            window_mhz=window_mhz,
            pedestal_mhz=pedestal_mhz,
            line_k=line_k,
            n_iter=n_iter,
            region_aware=region_aware,
            smoothing_mhz=smoothing_mhz,
            smoothing_percentile=smoothing_percentile,
            convolve_mhz=convolve_mhz,
        )
        scatter_preset_layer = settings
        scatter_preset_name: Optional[str] = None
        if preset is not None:
            scatter_preset_layer = load_noise_preset(preset)
            scatter_preset_name = str(preset)
        scatter_resolved = resolve_noise_settings(
            explicit=scatter_explicit,
            preset=scatter_preset_layer,
            persisted=load_noise_settings_from_h5(file_path),
            recommended=None,
        )
        return _compute_noise_scatter(
            file_path=file_path,
            complex_ft=complex_ft,
            settings=scatter_resolved,
            preset_name=scatter_preset_name,
        )

    # When ``from_saved_params=True`` the legacy contract reads the
    # four user-tunable knobs from ``processing_parameters/noise_estimation``
    # and treats them as the only override layer; explicit per-knob
    # kwargs are ignored. Kept for back-compat with code paths that
    # round-trip those four parameters via ``visualize_noise(save_params=True)``.
    if from_saved_params:
        if settings is not None or preset is not None:
            raise ValueError(
                "'from_saved_params=True' is the legacy persistence "
                "contract; pass 'settings=' / 'preset=' instead, or drop "
                "'from_saved_params' to let the resolver pick up the "
                "persisted layer automatically."
            )
        saved_params: Dict[str, Any] = {}
        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'processing_parameters' in h5f and 'noise_estimation' in h5f['processing_parameters']:
                    noise_group = h5f['processing_parameters/noise_estimation']
                    for param_name in ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']:
                        if param_name in noise_group.attrs:
                            value = noise_group.attrs[param_name]
                            if isinstance(value, str) and value == "__None__":
                                value = None
                            saved_params[param_name] = value
                    logger.info(f"Loaded saved noise parameters: {saved_params}")
        except Exception as e:
            logger.warning(f"Could not load saved noise parameters: {e}")
        explicit = _build_explicit_from_kwargs(
            skew_target=saved_params.get('skew_target'),
            min_bin_fraction=saved_params.get('min_bin_fraction'),
            smoothing_window_mhz=saved_params.get('smoothing_window_mhz'),
            min_noise_fraction=saved_params.get('min_noise_fraction'),
        )
        preset_layer: Optional[NoiseSettings] = None
        preset_name: Optional[str] = None
        # The legacy block does not capture the persisted ``stage2_noise``
        # block; honour the legacy semantic by skipping the resolver's
        # persisted layer too.
        persisted_layer: Optional[NoiseSettings] = None
    else:
        if preset is not None and settings is not None:
            raise ValueError(
                "'preset' and 'settings' are alternative ways to populate "
                "the preset layer of the noise-settings chain; pass exactly "
                "one (or override individual fields via explicit kwargs)"
            )
        explicit = _build_explicit_from_kwargs(
            skew_target=skew_target,
            min_bin_fraction=min_bin_fraction,
            smoothing_window_mhz=smoothing_window_mhz,
            min_noise_fraction=min_noise_fraction,
        )
        preset_layer = settings
        preset_name = None
        if preset is not None:
            preset_layer = load_noise_preset(preset)
            preset_name = str(preset)
        persisted_layer = load_noise_settings_from_h5(file_path)

    resolved = resolve_noise_settings(
        explicit=explicit,
        preset=preset_layer,
        persisted=persisted_layer,
        recommended=None,
    )

    # Lift the resolved fields into the kernel call. Every field with a
    # hard default in NoiseSettings._HARD_DEFAULTS is guaranteed non-None.
    binning = resolved.binning
    skew = resolved.skewness
    smoothing = resolved.smoothing
    skirt = resolved.skirt_exclusion
    skew_target_v = _required(skew.skew_target, "skewness.skew_target")
    inc_v = _required(skew.inc, "skewness.inc")
    min_bin_fraction_v = _required(
        binning.min_bin_fraction, "binning.min_bin_fraction"
    )
    min_noise_fraction_v = _required(
        binning.min_noise_fraction, "binning.min_noise_fraction"
    )
    smoothing_window_mhz_v = smoothing.smoothing_window_mhz  # None is OK -> kernel default
    subdivision_threshold_v = _required(
        binning.subdivision_threshold, "binning.subdivision_threshold"
    )
    abs_min_bin_size_v = _required(
        binning.abs_min_bin_size, "binning.abs_min_bin_size"
    )
    strong_peak_snr_v = _required(
        skirt.strong_peak_snr, "skirt_exclusion.strong_peak_snr"
    )
    skirt_exclusion_k_v = _required(
        skirt.skirt_exclusion_k, "skirt_exclusion.skirt_exclusion_k"
    )
    max_skirt_exclusion_mhz_v = _required(
        skirt.max_skirt_exclusion_mhz,
        "skirt_exclusion.max_skirt_exclusion_mhz",
    )

    # ``processing_params`` retains the legacy schema (the four user-tunable
    # knobs) because downstream serialisation (``parameters_used`` attr on
    # ``/stage2_noise_result``) and the legacy save/load_noise_parameters
    # round-trip both read it.
    processing_params: Dict[str, Any] = {
        'skew_target': float(skew_target_v),
        'min_bin_fraction': float(min_bin_fraction_v),
        'smoothing_window_mhz': (
            float(smoothing_window_mhz_v) if smoothing_window_mhz_v is not None else None
        ),
        'min_noise_fraction': float(min_noise_fraction_v),
    }

    logger.info("Noise estimation parameters (resolved):")
    for param, value in processing_params.items():
        logger.info(f"  {param}: {value}")

    # Perform noise estimation
    try:
        noise_result = estimate_noise_adaptive(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            skew_target=float(skew_target_v),
            inc=float(inc_v),
            min_bin_fraction=float(min_bin_fraction_v),
            smoothing_window_mhz=(
                float(smoothing_window_mhz_v)
                if smoothing_window_mhz_v is not None
                else None
            ),
            min_noise_fraction=float(min_noise_fraction_v),
            verbose=True,
            subdivision_threshold=float(subdivision_threshold_v),
            abs_min_bin_size=int(abs_min_bin_size_v),
            strong_peak_snr=float(strong_peak_snr_v),
            skirt_exclusion_k=float(skirt_exclusion_k_v),
            max_skirt_exclusion_mhz=float(max_skirt_exclusion_mhz_v),
        )
        logger.info("Noise estimation completed successfully")
        logger.info(f"  Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}")
        logger.info(f"  Number of bins: {noise_result.bin_info.get('n_bins', 'unknown')}")
        logger.info(f"  RMS noise range: {noise_result.rms_noise.min():.2e} - {noise_result.rms_noise.max():.2e}")
    except Exception as e:
        raise ValueError(f"Noise estimation failed: {e}")
    
    # Store NoiseResult and mark stage as completed
    try:
        # Save NoiseResult to pipeline file
        save_noise_result_impl(
            file_path=file_path,
            noise_result=noise_result,
            complex_ft=complex_ft,
            parameters_used=processing_params
        )

        # Persist the resolved NoiseSettings to ``processing_parameters/stage2_noise``.
        # The four legacy user-tunable kwargs continue to land in
        # ``processing_parameters/noise_estimation`` via ``save_noise_result_impl``
        # for ``from_saved_params=True`` back-compat; the new canonical
        # record below is what the resolver's persisted layer reads.
        save_noise_settings_to_h5(
            file_path, resolved, preset_name=preset_name,
        )

        # Update stage tracker to mark stage2_noise_result complete
        _update_stage_completion(file_path, 'stage2_noise_result')

        logger.info("Stage 2: Noise estimation results saved and marked complete")

    except Exception as e:
        logger.error(f"Failed to save noise estimation results: {e}")
        raise RuntimeError(f"Noise estimation succeeded but storage failed: {e}")
    
    # Return comprehensive results
    result = {
        'status': 'success',
        'noise_result': noise_result,
        'complex_ft': complex_ft,
        'parameters_used': processing_params,
        'frequency_points': len(complex_ft.freq_array),
        'frequency_range': (complex_ft.freq_array[0], complex_ft.freq_array[-1]),
        'noise_points': int(noise_result.noise_mask.sum()),
        'total_points': len(complex_ft.freq_array)
    }

    return result


def _compute_noise_scatter(
    *,
    file_path: str,
    complex_ft: Any,
    settings: NoiseSettings,
    preset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the scatter (high-pass) Stage 2 estimator and persist its result.

    Sibling of the adaptive path in :func:`compute_noise_estimation_impl`: it
    runs :func:`estimate_noise_scatter` on the supplied ComplexFT, stores the
    NoiseResult in ``stage2_noise_result``, persists the resolved
    :class:`NoiseSettings` to ``stage2_noise``, and marks the stage complete.
    The ``settings`` argument is a *resolved* bundle whose ``scatter`` sub-block
    has every field filled from the hard defaults if no layer supplied one.
    """
    scatter = settings.scatter
    window_mhz_v = float(_required(scatter.window_mhz, "scatter.window_mhz"))
    pedestal_mhz_v = float(_required(scatter.pedestal_mhz, "scatter.pedestal_mhz"))
    line_k_v = float(_required(scatter.line_k, "scatter.line_k"))
    n_iter_v = int(_required(scatter.n_iter, "scatter.n_iter"))
    region_aware_v = bool(_required(scatter.region_aware, "scatter.region_aware"))
    smoothing_mhz_v = float(_required(scatter.smoothing_mhz, "scatter.smoothing_mhz"))
    smoothing_percentile_v = float(
        _required(scatter.smoothing_percentile, "scatter.smoothing_percentile")
    )
    convolve_mhz_v = float(_required(scatter.convolve_mhz, "scatter.convolve_mhz"))

    processing_params: Dict[str, Any] = {
        'method': 'scatter',
        'window_mhz': window_mhz_v,
        'pedestal_mhz': pedestal_mhz_v,
        'line_k': line_k_v,
        'n_iter': n_iter_v,
        'region_aware': region_aware_v,
        'smoothing_mhz': smoothing_mhz_v,
        'smoothing_percentile': smoothing_percentile_v,
        'convolve_mhz': convolve_mhz_v,
    }

    logger.info("Noise estimation parameters (scatter estimator, resolved):")
    for param, value in processing_params.items():
        logger.info(f"  {param}: {value}")

    try:
        noise_result = estimate_noise_scatter(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            window_mhz=window_mhz_v,
            pedestal_mhz=pedestal_mhz_v,
            line_k=line_k_v,
            n_iter=n_iter_v,
            region_aware=region_aware_v,
            smoothing_mhz=smoothing_mhz_v,
            smoothing_percentile=smoothing_percentile_v,
            convolve_mhz=convolve_mhz_v,
        )
        logger.info("Noise estimation completed successfully")
        logger.info(f"  Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}")
        logger.info(f"  RMS noise range: {noise_result.rms_noise.min():.2e} - {noise_result.rms_noise.max():.2e}")
    except Exception as e:
        raise ValueError(f"Noise estimation failed: {e}")

    try:
        save_noise_result_impl(
            file_path=file_path,
            noise_result=noise_result,
            complex_ft=complex_ft,
            parameters_used=processing_params,
        )
        # Persist the resolved NoiseSettings (scatter sub-block) to
        # ``processing_parameters/stage2_noise`` so a no-kwargs re-run inherits
        # it via the resolver's persisted layer — mirrors the adaptive path.
        save_noise_settings_to_h5(file_path, settings, preset_name=preset_name)
        _update_stage_completion(file_path, 'stage2_noise_result')
        logger.info("Stage 2: Noise estimation results saved and marked complete")
    except Exception as e:
        logger.error(f"Failed to save noise estimation results: {e}")
        raise RuntimeError(f"Noise estimation succeeded but storage failed: {e}")

    return {
        'status': 'success',
        'noise_result': noise_result,
        'complex_ft': complex_ft,
        'parameters_used': processing_params,
        'frequency_points': len(complex_ft.freq_array),
        'frequency_range': (complex_ft.freq_array[0], complex_ft.freq_array[-1]),
        'noise_points': int(noise_result.noise_mask.sum()),
        'total_points': len(complex_ft.freq_array),
    }


def visualize_noise_impl(
    file_path: str,
    y_max_factor: Optional[float] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    show_bin_boundaries: Optional[bool] = None,
    show_noise_points: Optional[bool] = None,
    backend: str = 'matplotlib',
    interactive: bool = True,
    save_params: bool = False,
    **plot_kwargs
) -> Any:
    """
    Shared implementation for noise visualization from .ftmw pipeline files.
    
    This function creates noise estimation diagnostic plots showing spectrum,
    noise points, bin boundaries, and RMS estimates.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
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
    backend : str, default 'matplotlib'
        Plotting backend ('matplotlib' or 'plotly')
    interactive : bool, default True
        Whether to create interactive plots
    save_params : bool, default False
        Whether to save custom parameters for future use
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
    """
    # Load NoiseResult and ComplexFT from pipeline file
    try:
        with h5py.File(file_path, 'r') as h5f:
            # Check dependencies
            if 'stage2_noise_result' not in h5f:
                raise ValueError(
                    "Stage 2 (noise estimation) must be completed before visualization. "
                    "Run estimate_noise() or estimate-noise command first."
                )
            
            # Check that Stage 1 parameters exist (needed for on-demand ComplexFT computation)
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError(
                    "Stage 1 (FT computation) required for noise visualization. "
                    "Run compute_ft() or compute-ft command first."
                )
        
        # Compute ComplexFT on-demand for visualization (consistent with Stage 2 architecture)
        from .stage1_impl import compute_ft_impl
        stage1_result = compute_ft_impl(file_path=file_path)
        complex_ft = stage1_result['complex_ft']
        
        # Load NoiseResult data
        with h5py.File(file_path, 'r') as h5f:
            
            noise_result = load_noise_result_from_hdf5(
                h5f['stage2_noise_result'],
                complex_ft.freq_array,
                complex_ft.magnitude_spectrum
            )
            logger.info("Loaded NoiseResult and ComplexFT from pipeline file")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load data from pipeline file {file_path}: {e}")
    
    # Import visualization function
    try:
        from ..visualization.noise_visualization import plot_noise_estimation
    except ImportError:
        raise ImportError("Noise visualization not available - visualization module missing")
    
    # Set parameter defaults
    plot_params = {
        'y_max_factor': y_max_factor if y_max_factor is not None else 20.0,
        'figsize': figsize if figsize is not None else (16, 6),
        'show_bin_boundaries': show_bin_boundaries if show_bin_boundaries is not None else True,
        'show_noise_points': show_noise_points if show_noise_points is not None else True,
        'backend': backend,
    }
    
    # Generate title if not provided
    if title is None:
        pipeline_name = Path(file_path).stem
        title = f"Pipeline {pipeline_name} - Noise Estimation"
        freq_range = (complex_ft.freq_array[0], complex_ft.freq_array[-1])
        title += f" ({freq_range[0]:.0f}-{freq_range[1]:.0f} MHz)"
    
    plot_params['title'] = title
    plot_params.update(plot_kwargs)
    
    # Create diagnostic plot
    try:
        fig = plot_noise_estimation(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            noise_result=noise_result,
            **plot_params
        )
        
        # Save parameters if requested
        if save_params:
            try:
                # Read the current noise estimation parameters from the pipeline file
                # These are the parameters that were used to create the current NoiseResult
                with h5py.File(file_path, 'r') as h5f:
                    if 'processing_parameters' in h5f and 'noise_estimation' in h5f['processing_parameters']:
                        noise_group = h5f['processing_parameters/noise_estimation']
                        current_noise_params = {}
                        for param_name in ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']:
                            if param_name in noise_group.attrs:
                                value = noise_group.attrs[param_name]
                                # Handle None values stored as strings
                                if isinstance(value, str) and value == "__None__":
                                    value = None
                                current_noise_params[param_name] = value
                        
                        # Save the current noise estimation parameters (this enables from_saved_params=True)
                        save_noise_parameters_impl(file_path, current_noise_params)
                        logger.info(f"Saved noise estimation parameters for future use: {current_noise_params}")
                    else:
                        logger.warning("No noise estimation parameters found in pipeline file")
                        
                # Also save any custom visualization parameters that differ from defaults
                custom_vis_params = {}
                if y_max_factor is not None and y_max_factor != 20.0:
                    custom_vis_params['y_max_factor'] = y_max_factor
                if figsize is not None and figsize != (16, 6):
                    custom_vis_params['figsize'] = figsize
                if show_bin_boundaries is not None and show_bin_boundaries != True:
                    custom_vis_params['show_bin_boundaries'] = show_bin_boundaries
                if show_noise_points is not None and show_noise_points != True:
                    custom_vis_params['show_noise_points'] = show_noise_points
                
                if custom_vis_params:
                    save_noise_parameters_impl(file_path, {'visualization': custom_vis_params})
                    logger.info(f"Saved {len(custom_vis_params)} visualization parameters")
                    
            except Exception as e:
                logger.warning(f"Failed to save parameters: {e}")
        
        logger.info("Noise estimation visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create noise visualization: {e}")


def save_noise_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """
    Save noise estimation parameters to .ftmw file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    parameters : dict
        Noise estimation parameters to save
    """
    try:
        # Create parameters dict for noise estimation (filter to known parameter names)
        noise_params = {}
        param_names = ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']
        
        for param_name in param_names:
            if param_name in parameters:
                value = parameters[param_name]
                # Convert None to string marker for HDF5 storage
                if value is None:
                    value = "__None__"
                noise_params[param_name] = value
        
        # Save noise parameters to processing_parameters/noise_estimation (same pattern as FT)
        if noise_params:
            with h5py.File(file_path, 'a') as h5f:
                # Ensure processing_parameters group exists
                if 'processing_parameters' not in h5f:
                    h5f.create_group('processing_parameters')
                
                processing_group = h5f['processing_parameters']
                
                # Remove existing noise parameters if present (allow parameter updates)
                if 'noise_estimation' in processing_group:
                    del processing_group['noise_estimation']
                
                # Create noise parameters group and save parameters
                noise_params_group = processing_group.create_group('noise_estimation')
                
                # Store parameters individually as attributes (for easy loading with from_saved_params)
                for param_name, value in noise_params.items():
                    noise_params_group.attrs[param_name] = value
                
                # Also store as JSON for completeness
                noise_params_group.attrs['parameters'] = json.dumps(noise_params, default=str)
                noise_params_group.attrs['last_updated'] = datetime.now().isoformat()
                
            logger.info(f"Saved {len(noise_params)} noise estimation parameters to processing_parameters/noise_estimation")
        else:
            logger.warning("No valid noise parameters to save")
            
    except Exception as e:
        raise RuntimeError(f"Failed to save noise parameters: {e}")


def save_noise_result_impl(
    file_path: str, 
    noise_result: NoiseResult, 
    complex_ft,
    parameters_used: Dict[str, Any]
) -> None:
    """
    Save NoiseResult to .ftmw pipeline file in stage2_noise_result group.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    noise_result : NoiseResult
        NoiseResult object to save
    complex_ft : ComplexFT
        ComplexFT object used for noise estimation (for freq/mag arrays)
    parameters_used : dict
        Parameters used for noise estimation
    """
    try:
        with h5py.File(file_path, 'a') as h5f:
            # Remove existing noise result if present
            if 'stage2_noise_result' in h5f:
                del h5f['stage2_noise_result']
            
            # Create stage2_noise_result group
            stage2_group = h5f.create_group('stage2_noise_result')
            
            # Save NoiseResult using existing serialization
            save_noise_result_to_hdf5(
                noise_result=noise_result,
                frequencies=complex_ft.freq_array,
                magnitudes=complex_ft.magnitude_spectrum,
                h5_group=stage2_group
            )
            
            # Add metadata and timestamp
            stage2_group.attrs['creation_time'] = datetime.now().isoformat()
            stage2_group.attrs['stage_name'] = 'stage2_noise_estimation'
            stage2_group.attrs['parameters_used'] = json.dumps(parameters_used, default=str)
            
        logger.info("NoiseResult saved to pipeline file successfully")
        
        # Also save parameters for future use
        save_noise_parameters_impl(file_path, parameters_used)
        
    except Exception as e:
        raise RuntimeError(f"Failed to save NoiseResult to pipeline file: {e}")


def load_noise_result_impl(file_path: str) -> Dict[str, Any]:
    """
    Load NoiseResult from .ftmw pipeline file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
        
    Returns
    -------
    dict
        Dict containing 'noise_result', 'complex_ft', and metadata
    """
    try:
        with h5py.File(file_path, 'r') as h5f:
            # Check dependencies
            if 'stage2_noise_result' not in h5f:
                raise ValueError("No noise estimation results found in pipeline file")
                
            # Check that Stage 1 parameters exist (needed for ComplexFT computation)
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError("Stage 1 parameters missing - cannot compute ComplexFT for NoiseResult loading")
        
        # Compute ComplexFT on-demand (consistent with new architecture)
        from .stage1_impl import compute_ft_impl
        stage1_result = compute_ft_impl(file_path=file_path)
        complex_ft = stage1_result['complex_ft']
        
        # Load NoiseResult using computed ComplexFT
        with h5py.File(file_path, 'r') as h5f:
            
            # Load NoiseResult
            noise_result = load_noise_result_from_hdf5(
                h5f['stage2_noise_result'],
                complex_ft.freq_array,
                complex_ft.magnitude_spectrum
            )
            
            # Load metadata
            stage2_group = h5f['stage2_noise_result']
            creation_time = stage2_group.attrs.get('creation_time', 'unknown')
            parameters_used = {}
            if 'parameters_used' in stage2_group.attrs:
                try:
                    parameters_used = json.loads(stage2_group.attrs['parameters_used'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse saved parameters")
            
        return {
            'noise_result': noise_result,
            'complex_ft': complex_ft,
            'creation_time': creation_time,
            'parameters_used': parameters_used
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to load NoiseResult from pipeline file: {e}")


# Add missing import
from datetime import datetime
import json


def _update_stage_completion(file_path: str, stage_name: str) -> None:
    """
    Update stage completion in the pipeline file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    stage_name : str
        Name of the stage to mark as completed
    """
    try:
        with h5py.File(file_path, 'a') as h5f:
            # Load current stage tracker
            from ..file_manager import _load_stage_tracker
            stage_tracker = _load_stage_tracker(file_path, h5f)
            
            # Mark stage as completed
            stage_tracker.mark_completed(stage_name)
            
            # Update pipeline_stages group
            if 'pipeline_stages' not in h5f:
                stages_group = h5f.create_group('pipeline_stages')
            else:
                stages_group = h5f['pipeline_stages']
            
            # Save updated completion status
            stages_group.attrs['completed_stages'] = json.dumps(list(stage_tracker.completed_stages))
            stages_group.attrs['last_updated'] = datetime.now().isoformat()
            
    except Exception as e:
        raise RuntimeError(f"Failed to update stage completion: {e}")