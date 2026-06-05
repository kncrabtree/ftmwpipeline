"""
Shared implementation for Stage 2: Noise Estimation.

This module contains the core implementation functions for noise estimation
and visualization that are shared between CLI, Pipeline class, and functional
API interfaces.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np

from ..core.noise_settings import (
    NoiseSettings,
)
from ..core.noise_settings import load_preset as load_noise_preset
from ..core.noise_settings import resolve as resolve_noise_settings
from ..file_manager import open_pipeline_file
from ..io.noise_result_serialization import (
    load_noise_result_from_hdf5,
    save_noise_result_to_hdf5,
)
from ..io.noise_settings_serialization import (
    load_noise_settings_from_h5,
    save_noise_settings_to_h5,
)
from ..preprocessing.noise_estimation import (
    NoiseResult,
    estimate_active_ft_noise,
)
from .active_ft_support import build_trimmed_active_ft
from .deprecation import warn_legacy_kwargs

logger = logging.getLogger(__name__)


def _required(value: Any, name: str) -> Any:
    """Coerce a post-resolve field that must be filled (hard default present)."""
    if value is None:
        raise AssertionError(
            f"resolved NoiseSettings.{name} is None; missing hard default"
        )
    return value


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

    Builds an explicit-layer ``NoiseSettings`` so the resolver merges it with
    any preset / persisted layer.
    """
    return NoiseSettings(
        window_mhz=window_mhz,
        pedestal_mhz=pedestal_mhz,
        line_k=line_k,
        n_iter=n_iter,
        region_aware=region_aware,
        smoothing_mhz=smoothing_mhz,
        smoothing_percentile=smoothing_percentile,
        convolve_mhz=convolve_mhz,
    )


def compute_noise_estimation_impl(
    file_path: str,
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
) -> Dict[str, Any]:
    """
    Shared implementation for Stage 2 noise estimation from .ftmw files.

    Runs the scatter (high-pass, region-aware) estimator on the ComplexFT
    computed on-demand from the persisted Stage 1 parameters. The scatter
    ``scatter`` sub-block resolves through the four-layer chain (explicit
    per-knob kwargs > preset/settings > persisted > hard default), the
    NoiseResult is stored in ``stage2_noise_result``, and the resolved
    settings are persisted to ``processing_parameters/stage2_noise``.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file.
    window_mhz, pedestal_mhz, line_k, n_iter, region_aware, smoothing_mhz,
    smoothing_percentile, convolve_mhz :
        Scatter-estimator per-knob overrides (the explicit layer).
    settings, preset :
        Alternative ways to populate the preset layer of the chain.

    Returns
    -------
    dict
        Processing results: ``noise_result``, ``complex_ft``,
        ``parameters_used``, plus frequency/point diagnostics.

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist.
    ValueError
        If Stage 1 dependencies are not met or parameters are invalid.
    """
    warn_legacy_kwargs(
        func_name="estimate_noise",
        legacy_kwargs={
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

    # Compute ComplexFT on-demand using Stage 1 implementation (correct architecture)
    try:
        # Check that Stage 1 parameters are available (Stage 1 dependency)
        with h5py.File(file_path, "r") as h5f:
            if (
                "processing_parameters" not in h5f
                or "ft_processing" not in h5f["processing_parameters"]
            ):
                raise ValueError(
                    "Stage 1 (FT computation) must be completed before noise estimation. "
                    "Run compute_ft() or compute-ft command first."
                )

        # Import Stage 1 implementation for on-demand ComplexFT computation
        from .stage1_impl import compute_ft_impl

        # Compute ComplexFT using saved Stage 1 parameters (lightweight on-demand computation)
        stage1_result = compute_ft_impl(file_path=file_path)

        complex_ft = stage1_result["complex_ft"]
        logger.info(
            f"Computed ComplexFT on-demand with {len(complex_ft.freq_array):,} frequency points"
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to compute ComplexFT from pipeline file {file_path}: {e}"
        )

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
        trim_range=stage1_result.get("trim_range"),
        settings=scatter_resolved,
        preset_name=scatter_preset_name,
    )


def _compute_noise_scatter(
    *,
    file_path: str,
    complex_ft: Any,
    trim_range: Optional[tuple] = None,
    settings: NoiseSettings,
    preset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the scatter (high-pass) Stage 2 estimator and persist its result.

    Measures noise on the canonical **unapodized active FT** (trimmed to the
    analysis band) -- the single grid every later stage scores, plans, and
    fits on -- and stores the NoiseResult in ``stage2_noise_result`` on that
    grid. Persists the resolved :class:`NoiseSettings` to ``stage2_noise`` and
    marks the stage complete. The full-record ``complex_ft`` is carried to the
    result dict for the display/range summary only. The ``settings`` argument
    is a *resolved* bundle with every field filled from the hard defaults if
    no layer supplied one.
    """
    window_mhz_v = float(_required(settings.window_mhz, "window_mhz"))
    pedestal_mhz_v = float(_required(settings.pedestal_mhz, "pedestal_mhz"))
    line_k_v = float(_required(settings.line_k, "line_k"))
    n_iter_v = int(_required(settings.n_iter, "n_iter"))
    region_aware_v = bool(_required(settings.region_aware, "region_aware"))
    smoothing_mhz_v = float(_required(settings.smoothing_mhz, "smoothing_mhz"))
    smoothing_percentile_v = float(
        _required(settings.smoothing_percentile, "smoothing_percentile")
    )
    convolve_mhz_v = float(_required(settings.convolve_mhz, "convolve_mhz"))

    processing_params: Dict[str, Any] = {
        "method": "scatter",
        "window_mhz": window_mhz_v,
        "pedestal_mhz": pedestal_mhz_v,
        "line_k": line_k_v,
        "n_iter": n_iter_v,
        "region_aware": region_aware_v,
        "smoothing_mhz": smoothing_mhz_v,
        "smoothing_percentile": smoothing_percentile_v,
        "convolve_mhz": convolve_mhz_v,
    }

    logger.info("Noise estimation parameters (scatter estimator, resolved):")
    for param, value in processing_params.items():
        logger.info(f"  {param}: {value}")

    # Measure on the canonical unapodized active FT (trimmed to the analysis
    # band) -- the single grid every later stage consumes.
    active_ft = build_trimmed_active_ft(file_path, trim_range)

    try:
        noise_result = estimate_active_ft_noise(
            active_ft.freq_array,
            active_ft.complex_spectrum,
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
        logger.info(
            f"  Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}"
        )
        logger.info(
            f"  RMS noise range: {noise_result.rms_noise.min():.2e} - {noise_result.rms_noise.max():.2e}"
        )
    except Exception as e:
        raise ValueError(f"Noise estimation failed: {e}")

    try:
        save_noise_result_impl(
            file_path=file_path,
            noise_result=noise_result,
            frequencies=active_ft.freq_array,
            magnitudes=active_ft.magnitude_spectrum,
            parameters_used=processing_params,
        )
        # Persist the resolved NoiseSettings to
        # ``processing_parameters/stage2_noise`` so a no-kwargs re-run inherits
        # it via the resolver's persisted layer.
        save_noise_settings_to_h5(file_path, settings, preset_name=preset_name)
        _update_stage_completion(file_path, "stage2_noise_result")
        logger.info("Stage 2: Noise estimation results saved and marked complete")
    except Exception as e:
        logger.error(f"Failed to save noise estimation results: {e}")
        raise RuntimeError(f"Noise estimation succeeded but storage failed: {e}")

    active_freq = active_ft.freq_array
    return {
        "status": "success",
        "noise_result": noise_result,
        "complex_ft": complex_ft,
        "active_ft": active_ft,
        "parameters_used": processing_params,
        "frequency_points": len(active_freq),
        "frequency_range": (float(active_freq.min()), float(active_freq.max())),
        "noise_points": int(noise_result.noise_mask.sum()),
        "total_points": len(active_freq),
    }


def visualize_noise_impl(
    file_path: str,
    y_max_factor: Optional[float] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    show_bin_boundaries: Optional[bool] = None,
    show_noise_points: Optional[bool] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
    **plot_kwargs: Any,
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
        with h5py.File(file_path, "r") as h5f:
            # Check dependencies
            if "stage2_noise_result" not in h5f:
                raise ValueError(
                    "Stage 2 (noise estimation) must be completed before visualization. "
                    "Run estimate_noise() or estimate-noise command first."
                )

            # Check that Stage 1 parameters exist (needed for on-demand ComplexFT computation)
            if (
                "processing_parameters" not in h5f
                or "ft_processing" not in h5f["processing_parameters"]
            ):
                raise ValueError(
                    "Stage 1 (FT computation) required for noise visualization. "
                    "Run compute_ft() or compute-ft command first."
                )

        # Rebuild the canonical trimmed active FT -- the grid the noise was
        # measured on -- and overlay sigma there (the noise diagnostic shows
        # the same active spectrum every later stage scores/fits on).
        from .stage1_impl import compute_ft_impl

        stage1_result = compute_ft_impl(file_path=file_path)
        complex_ft = build_trimmed_active_ft(file_path, stage1_result.get("trim_range"))

        # Load NoiseResult data
        with h5py.File(file_path, "r") as h5f:

            noise_result = load_noise_result_from_hdf5(
                h5f["stage2_noise_result"],
                complex_ft.freq_array,
                complex_ft.magnitude_spectrum,
            )
            logger.info("Loaded NoiseResult and active FT from pipeline file")

    except Exception as e:
        raise RuntimeError(f"Failed to load data from pipeline file {file_path}: {e}")

    # Import visualization function
    try:
        from ..visualization.noise_visualization import plot_noise_estimation
    except ImportError:
        raise ImportError(
            "Noise visualization not available - visualization module missing"
        )

    # Set parameter defaults
    plot_params: Dict[str, Any] = {
        "y_max_factor": y_max_factor if y_max_factor is not None else 20.0,
        "figsize": figsize if figsize is not None else (16, 6),
        "show_bin_boundaries": (
            show_bin_boundaries if show_bin_boundaries is not None else True
        ),
        "show_noise_points": (
            show_noise_points if show_noise_points is not None else True
        ),
        "backend": backend,
    }

    # Generate title if not provided
    if title is None:
        pipeline_name = Path(file_path).stem
        title = f"Pipeline {pipeline_name} - Noise Estimation"
        freq_range = (complex_ft.freq_array[0], complex_ft.freq_array[-1])
        title += f" ({freq_range[0]:.0f}-{freq_range[1]:.0f} MHz)"

    plot_params["title"] = title
    plot_params.update(plot_kwargs)

    # Create diagnostic plot
    try:
        fig = plot_noise_estimation(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            noise_result=noise_result,
            **plot_params,
        )

        logger.info("Noise estimation visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create noise visualization: {e}")


def save_noise_result_impl(
    file_path: str,
    noise_result: NoiseResult,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    parameters_used: Dict[str, Any],
) -> None:
    """
    Save NoiseResult to .ftmw pipeline file in stage2_noise_result group.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    noise_result : NoiseResult
        NoiseResult object to save
    frequencies, magnitudes : np.ndarray
        The active-FT grid the noise was measured on (bin order). The σ array
        is stored verbatim against this grid; the loader rebuilds the identical
        canonical active FT to reconstruct on the same grid.
    parameters_used : dict
        Parameters used for noise estimation
    """
    try:
        with h5py.File(file_path, "a") as h5f:
            # Remove existing noise result if present
            if "stage2_noise_result" in h5f:
                del h5f["stage2_noise_result"]

            # Create stage2_noise_result group
            stage2_group = h5f.create_group("stage2_noise_result")

            # Save NoiseResult using existing serialization
            save_noise_result_to_hdf5(
                noise_result=noise_result,
                frequencies=np.asarray(frequencies, dtype=float),
                magnitudes=np.asarray(magnitudes, dtype=float),
                h5_group=stage2_group,
            )

            # Add metadata and timestamp
            stage2_group.attrs["creation_time"] = datetime.now().isoformat()
            stage2_group.attrs["stage_name"] = "stage2_noise_estimation"
            stage2_group.attrs["parameters_used"] = json.dumps(
                parameters_used, default=str
            )

        logger.info("NoiseResult saved to pipeline file successfully")

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
        with h5py.File(file_path, "r") as h5f:
            # Check dependencies
            if "stage2_noise_result" not in h5f:
                raise ValueError("No noise estimation results found in pipeline file")

            # Check that Stage 1 parameters exist (needed for ComplexFT computation)
            if (
                "processing_parameters" not in h5f
                or "ft_processing" not in h5f["processing_parameters"]
            ):
                raise ValueError(
                    "Stage 1 parameters missing - cannot compute ComplexFT for NoiseResult loading"
                )

        # Rebuild the canonical trimmed active FT on-demand: the sigma was
        # measured and stored on this grid, so reconstruction reads it back
        # element-for-element. (The full-record FT is display-only.)
        from .stage1_impl import compute_ft_impl

        stage1_result = compute_ft_impl(file_path=file_path)
        active_ft = build_trimmed_active_ft(file_path, stage1_result.get("trim_range"))
        complex_ft = active_ft

        # Load NoiseResult on the active-FT grid
        with h5py.File(file_path, "r") as h5f:

            # Load NoiseResult
            noise_result = load_noise_result_from_hdf5(
                h5f["stage2_noise_result"],
                active_ft.freq_array,
                active_ft.magnitude_spectrum,
            )

            # Load metadata
            stage2_group = h5f["stage2_noise_result"]
            creation_time = stage2_group.attrs.get("creation_time", "unknown")
            parameters_used = {}
            if "parameters_used" in stage2_group.attrs:
                try:
                    parameters_used = json.loads(stage2_group.attrs["parameters_used"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse saved parameters")

        return {
            "noise_result": noise_result,
            "complex_ft": complex_ft,
            "creation_time": creation_time,
            "parameters_used": parameters_used,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to load NoiseResult from pipeline file: {e}")


import json

# Add missing import
from datetime import datetime


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
        with h5py.File(file_path, "a") as h5f:
            # Load current stage tracker
            from ..file_manager import _load_stage_tracker

            stage_tracker = _load_stage_tracker(Path(file_path), h5f)

            # Mark stage as completed
            stage_tracker.mark_completed(stage_name)

            # Update pipeline_stages group
            if "pipeline_stages" not in h5f:
                stages_group = h5f.create_group("pipeline_stages")
            else:
                stages_group = h5f["pipeline_stages"]

            # Save updated completion status
            stages_group.attrs["completed_stages"] = json.dumps(
                list(stage_tracker.completed_stages)
            )
            stages_group.attrs["last_updated"] = datetime.now().isoformat()

    except Exception as e:
        raise RuntimeError(f"Failed to update stage completion: {e}")
