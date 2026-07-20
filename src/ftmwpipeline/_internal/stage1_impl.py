"""
Shared implementation for Stage 1: FT Processing and Spectrum Operations.

Stage 1 owns the *canonical* FT processing settings. The settings actually
used are resolved through ``explicit override > persisted user settings >
import-time recommended`` (see :mod:`ftmwpipeline.core.settings`) and, when
this is a user-driven Stage 1 invocation (``persist=True``), the resolved
:class:`FTSettings` -- including the frequency ``trim`` -- are written to
``processing_parameters/ft_processing`` as the experiment's binding settings.
Internal recomputes by later stages (``persist=False``, no overrides) rebuild
exactly that spectrum.

This module is a thin orchestration layer; it is wrapped identically by the
CLI, the ``Pipeline`` class, and the functional API.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py

from ..core.data_structures import ComplexFT
from ..core.settings import (
    FT_PROCESSING_PATH,
    RECOMMENDED_PATH,
    FTSettings,
    resolve,
)
from .stage0_impl import load_fid_from_pipeline_impl

logger = logging.getLogger(__name__)


def _read_settings_layer(file_path: str, group_path: str) -> Optional[FTSettings]:
    """Read one resolution layer (persisted or recommended) as ``FTSettings``.

    Tolerant of older ``ft_processing`` records that stored only a JSON
    ``parameters`` blob: its keys are folded in so those files still resolve
    sensibly.
    """
    with h5py.File(file_path, "r") as h5f:
        if group_path not in h5f:
            return None
        attrs: Dict[str, Any] = dict(h5f[group_path].attrs)
    if "parameters" in attrs and "zpf" not in attrs:
        try:
            blob = json.loads(attrs["parameters"])
            if isinstance(blob, dict):
                for key, value in blob.items():
                    attrs.setdefault(key, value)
        except (json.JSONDecodeError, TypeError):  # pragma: no cover
            pass
    return FTSettings.from_attrs(attrs)


def _resolve_settings(file_path: str, explicit: Optional[FTSettings]) -> FTSettings:
    """Resolve effective FT settings for ``file_path`` (the D7 chain)."""
    persisted = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    recommended = _read_settings_layer(file_path, RECOMMENDED_PATH)
    return resolve(explicit, persisted, recommended)


def compute_ft_impl(
    file_path: str,
    settings: Optional[FTSettings] = None,
    validate_only: bool = False,
    persist: bool = False,
) -> Dict[str, Any]:
    """Compute the FT for a ``.ftmw`` file using the resolved canonical settings.

    Parameters
    ----------
    file_path:
        Path to the ``.ftmw`` pipeline file.
    settings:
        Explicit overrides (sparse :class:`FTSettings`; ``None`` fields fall
        through to persisted, then recommended). ``None`` means "no overrides"
        -- the normal path used by downstream stages.
    validate_only:
        If True, return validation info without constructing the ``ComplexFT``.
    persist:
        If True (a user-driven Stage 1 invocation), write the *resolved*
        settings to ``processing_parameters/ft_processing`` as canonical and
        mark Stage 1 complete. Internal recomputes pass ``False``.

    Returns
    -------
    dict
        ``complex_ft`` plus metadata. Return keys are kept stable for the
        untouched Stage 2/3 callers (``complex_ft``, ``trim_range``,
        ``processing_params``, ...).
    """
    try:
        fid = load_fid_from_pipeline_impl(file_path)
        logger.info(f"Loaded FID with {len(fid.data):,} points from pipeline file")
    except Exception as e:
        raise RuntimeError(f"Failed to load FID from pipeline file {file_path}: {e}")

    resolved = _resolve_settings(file_path, settings)
    trim_range = resolved.trim

    logger.info("Resolved FT processing settings:")
    for name, value in resolved.to_preprocess_kwargs().items():
        logger.info(f"  {name}: {value}")
    logger.info(f"  trim: {trim_range}")

    try:
        preprocessed_fid = fid.preprocess(**resolved.to_preprocess_kwargs())
        logger.info(f"Preprocessing complete: {len(preprocessed_fid.data):,} points")
    except Exception as e:
        raise ValueError(f"FID preprocessing failed: {e}")

    try:
        complex_spectrum, freq_array = preprocessed_fid.compute_fft()
        logger.info(
            f"FFT complete: {len(complex_spectrum):,} frequency points "
            f"({freq_array[0]:.1f} - {freq_array[-1]:.1f} MHz)"
        )
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}")

    processing_params_dict = resolved.to_preprocess_kwargs()

    if validate_only:
        result: Dict[str, Any] = {
            "status": "validated",
            "fid_points": len(fid.data),
            "preprocessed_points": len(preprocessed_fid.data),
            "frequency_points": len(complex_spectrum),
            "frequency_range": (freq_array[0], freq_array[-1]),
            "processing_params": processing_params_dict,
        }
        if trim_range is not None:
            import numpy as np

            mask = (freq_array >= trim_range[0]) & (freq_array <= trim_range[1])
            if not np.any(mask):
                raise ValueError(
                    f"No data points in trim range "
                    f"[{trim_range[0]:.1f}, {trim_range[1]:.1f}] MHz"
                )
            result["trimmed_points"] = int(np.sum(mask))
            result["trim_range"] = trim_range
        if persist:
            _persist_canonical_settings(file_path, resolved)
        return result

    try:
        fid_context = {
            "probe_freq_mhz": fid.probe_freq_mhz,
            "spacing_us": fid.spacing * 1e6,
            "sideband": fid.sideband.value,
            "original_fid_length": len(fid.data),
        }
        complex_ft = ComplexFT.from_spectrum(
            complex_spectrum=complex_spectrum,
            freq_array=freq_array,
            metadata={
                "processing_params": preprocessed_fid.processing_params,
                "fid_context": fid_context,
            },
        )
        logger.info(f"ComplexFT created: {len(complex_ft.complex_spectrum):,} points")
    except Exception as e:
        raise RuntimeError(f"ComplexFT creation failed: {e}")

    if trim_range is not None:
        try:
            complex_ft = complex_ft.trim_to_range(trim_range[0], trim_range[1])
            logger.info(
                f"Trimmed spectrum: {len(complex_ft.complex_spectrum):,} points"
            )
        except Exception as e:
            raise ValueError(f"Frequency trimming failed: {e}")

    result = {
        "status": "success",
        "complex_ft": complex_ft,
        "preprocessed_fid": preprocessed_fid,
        "original_fid": fid,
        "processing_params": processing_params_dict,
        "resolved_settings": resolved,
        "fid_points": len(fid.data),
        "preprocessed_points": len(preprocessed_fid.data),
        "frequency_points": len(complex_ft.complex_spectrum),
        "frequency_range": (complex_ft.freq_array[0], complex_ft.freq_array[-1]),
    }
    if trim_range is not None:
        result["trim_range"] = trim_range

    if persist:
        try:
            _persist_canonical_settings(file_path, resolved)
            logger.info("Canonical FT settings + Stage 1 completion persisted")
        except Exception as e:
            logger.warning(f"Failed to persist canonical FT settings: {e}")

    return result


def visualize_ft_impl(
    file_path: str,
    settings: Optional[FTSettings] = None,
    title: Optional[str] = None,
    show_fid_panels: bool = True,
    interactive: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Render the FID-to-spectrum workflow for the resolved settings.

    Visualization never persists; it is an exploration tool.
    """
    ft_result = compute_ft_impl(
        file_path=file_path, settings=settings, validate_only=False
    )
    complex_ft = ft_result["complex_ft"]
    preprocessed_fid = ft_result["preprocessed_fid"]
    original_fid = ft_result["original_fid"]
    trim_range = ft_result.get("trim_range")

    try:
        from ..visualization.spectrum_visualization import plot_complex_ft
    except ImportError:
        raise ImportError(
            "Spectrum visualization not available - visualization module missing"
        )

    if title is None:
        title = f"Pipeline {Path(file_path).stem} - FT Visualization"
        if trim_range is not None:
            title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"

    try:
        fig = plot_complex_ft(
            complex_ft=complex_ft,
            title=title,
            interactive=interactive,
            fid=original_fid if show_fid_panels else None,
            preprocessed_fid=preprocessed_fid if show_fid_panels else None,
            show_fid_panels=show_fid_panels,
            **plot_kwargs,
        )
        logger.info("FT visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create FT visualization: {e}")


def _dependents_of(stage: str, deps: Dict[str, Any]) -> set:
    """Transitive set of stages that depend (directly/indirectly) on ``stage``.

    Excludes ``stage`` itself. Used to invalidate everything built on the FT
    when the canonical FT settings change.
    """
    result: set = set()
    changed = True
    while changed:
        changed = False
        for st, required in deps.items():
            if st in result:
                continue
            if any(r == stage or r in result for r in required):
                result.add(st)
                changed = True
    return result


def _persist_canonical_settings(file_path: str, resolved: FTSettings) -> None:
    """Write resolved settings to ``ft_processing`` and mark Stage 1 complete.

    No ``ComplexFT`` is stored -- the lightweight ``.ftmw`` model recomputes it
    on demand from the FID + these settings. If the resolved settings *differ*
    from a previously persisted canonical record, every stage built on the FT
    (Stage 2 noise, Stage 3 peaks, ...) is invalidated: its stored result is
    removed, it is dropped from the completed set, and a loud warning is
    logged. An identical re-persist (idempotent Jupyter re-run) changes
    nothing.
    """
    from ..file_manager import PipelineStageTracker

    new_attrs = resolved.to_attrs()
    with h5py.File(file_path, "a") as h5f:
        proc = h5f.require_group("processing_parameters")
        old_attrs = None
        if "ft_processing" in proc:
            old_attrs = FTSettings.from_attrs(
                dict(proc["ft_processing"].attrs)
            ).to_attrs()
            del proc["ft_processing"]
        ft_group = proc.create_group("ft_processing")
        for name, value in new_attrs.items():
            ft_group.attrs[name] = value
        # Human/debug mirror of the canonical record.
        ft_group.attrs["parameters"] = json.dumps(new_attrs, default=str)
        ft_group.attrs["last_updated"] = datetime.now().isoformat()

        stages = h5f.require_group("pipeline_stages")
        completed = json.loads(stages.attrs.get("completed_stages", "[]"))

        if old_attrs is not None and old_attrs != new_attrs:
            deps = PipelineStageTracker.STAGE_DEPENDENCIES
            paths = PipelineStageTracker.STAGE_DATA_PATHS
            invalidated = []
            for st in _dependents_of("stage1_complex_ft", deps):
                data_path = paths.get(st, st)
                if data_path in h5f:
                    del h5f[data_path]
                if st in completed:
                    completed.remove(st)
                    invalidated.append(st)
            if invalidated:
                logger.warning(
                    "Canonical FT settings changed (%s -> %s); invalidated "
                    "downstream stage(s) %s -- re-run them on the new "
                    "spectrum.",
                    old_attrs,
                    new_attrs,
                    sorted(invalidated),
                )

        if "stage1_complex_ft" not in completed:
            completed.append("stage1_complex_ft")
        stages.attrs["completed_stages"] = json.dumps(completed)
        stages.attrs["last_updated"] = datetime.now().isoformat()
    logger.info("FT parameters and stage tracking saved to pipeline file")


def save_ft_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """Persist an explicit settings dict as canonical (used by --save flows)."""
    settings = FTSettings.from_attrs(parameters)
    resolved = _resolve_settings(file_path, settings)
    _persist_canonical_settings(file_path, resolved)
    logger.info("Processing parameters saved successfully")


def compare_ft_parameters_impl(
    file_path: str, current_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Report which of ``current_params`` differ from the resolved defaults."""
    defaults = _resolve_settings(file_path, None).to_attrs()
    custom = {
        k: v
        for k, v in current_params.items()
        if v is not None and defaults.get(k) != v
    }
    return {
        "default_params": defaults,
        "current_params": current_params,
        "custom_params": custom,
        "has_custom_params": len(custom) > 0,
    }
