"""
Shared implementation for Stage 4: Window assignment.

Orchestration only -- the planning algorithm lives in
``ftmwpipeline.preprocessing.window_planning`` and the coherence statistic in
``ftmwpipeline.preprocessing.edge_coherence``. Stage 4 turns the *promoted*
Stage 3 peak list into a :class:`~ftmwpipeline.core.data_structures.WindowPlan`:
a set of disjoint fit windows, each annotated with the peaks to fit freely, the
strong out-of-band lines whose leakage is carried frozen, a fit dependency DAG,
and a difficulty classification. It is purely structural -- it makes no fits
and changes no spectrum.

Stage 4 owns no FT settings: it operates on the Stage 1 persisted canonical
spectrum (incl. trim) and the canonical Stage 2 noise, exactly the surface the
Stage 3 peaks were scored on. Wrapped identically by the CLI, Pipeline class,
and functional API. See ``dev-docs/planning/stage4-window-assignment.md``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py

from ..core.data_structures import ComplexFT, WindowDifficulty, WindowPlan
from ..core.window_planning_settings import (
    WindowPlanningSettings,
)
from ..core.window_planning_settings import load_preset as load_window_planning_preset
from ..core.window_planning_settings import resolve as resolve_window_planning_settings
from ..file_manager import invalidate_downstream_stages
from ..io.window_planning_settings_serialization import (
    load_window_planning_settings_from_h5,
    save_window_planning_settings_to_h5,
)
from ..io.window_serialization import (
    load_window_plan_from_hdf5,
    save_window_plan_to_hdf5,
)
from ..preprocessing.window_planning import (
    build_window_plan,
)
from .active_ft_support import build_active_grid_with_noise
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage3_impl import (
    _active_acquisition_us,
    load_peaks_impl,
)

logger = logging.getLogger(__name__)


def _required(value: Any, name: str) -> Any:
    """Coerce a post-resolve field that must be filled (hard default present)."""
    if value is None:
        raise AssertionError(
            f"resolved WindowPlanningSettings.{name} is None; missing hard default"
        )
    return value


def assign_windows_impl(
    file_path: str,
    *,
    settings: Optional[WindowPlanningSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Stage 4 window plan from the promoted Stage 3 peaks and persist it.

    Requires Stage 3 (peak detection) completed. Operates on the Stage 1
    canonical spectrum and the canonical Stage 2 noise; consumes only the
    peaks flagged ``properties['promoted']``.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); pass ``settings=`` to drive window planning from a
    :class:`WindowPlanningSettings` dataclass, or ``preset=NAME_OR_PATH`` to
    load from packaged YAML. They are mutually exclusive, and a value persisted
    in the ``.ftmw`` outranks either (D11). Returns the plan plus diagnostics;
    also writes ``/stage4_windows`` and marks the stage done.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file.
    settings : WindowPlanningSettings, optional
        Bundle of Stage 4 knobs; fields left ``None`` fall through the
        resolution chain. Resolves at the explicit override layer (outranks the
        persisted record). Mutually exclusive with ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage4:`` block.
        Mutually exclusive with ``settings``.

    Raises
    ------
    ValueError
        If Stage 3 has not been completed, or if ``settings=`` and ``preset=``
        are both supplied.
    """
    if preset is not None and settings is not None:
        raise ValueError(
            "'preset' and 'settings' are mutually exclusive; pass exactly one. "
            "A 'settings' bundle is the explicit override (outranks the "
            "persisted record); a 'preset' .yml seeds only unfixed fields "
            "(the persisted record outranks it, per D11)."
        )

    with h5py.File(file_path, "r") as h5f:
        if "stage3_peaks" not in h5f:
            raise ValueError(
                "Stage 3 (peak detection) must be completed before window "
                "assignment. Run detect_peaks()/'peaks run' first."
            )

    preset_layer: Optional[WindowPlanningSettings] = None
    preset_name: Optional[str] = None
    if preset is not None:
        preset_layer = load_window_planning_preset(preset)
        preset_name = str(preset)
    persisted_layer = load_window_planning_settings_from_h5(file_path)

    resolved = resolve_window_planning_settings(
        explicit=settings,
        preset=preset_layer,
        persisted=persisted_layer,
        recommended=None,
    )

    coh = resolved.coherence
    clus = resolved.clustering
    contrib = resolved.contributor
    leak = resolved.leakage

    edge_m_v: int = int(_required(coh.edge_m, "coherence.edge_m"))
    trim_m_v: int = int(_required(coh.trim_m, "coherence.trim_m"))
    edge_threshold_v: float = float(
        _required(coh.edge_threshold, "coherence.edge_threshold")
    )
    max_width_v: float = float(
        _required(clus.max_window_width_mhz, "clustering.max_window_width_mhz")
    )
    min_half_v: float = float(
        _required(
            clus.min_window_half_width_mhz,
            "clustering.min_window_half_width_mhz",
        )
    )
    max_peaks_per_window_v: int = int(
        _required(clus.max_peaks_per_window, "clustering.max_peaks_per_window")
    )
    max_width_points_v: int = int(
        _required(clus.max_window_width_points, "clustering.max_window_width_points")
    )
    min_half_points_v: int = int(
        _required(
            clus.min_window_half_width_points,
            "clustering.min_window_half_width_points",
        )
    )
    min_freeze_v: float = float(
        _required(contrib.min_freeze_snr, "contributor.min_freeze_snr")
    )
    mag_thresh_v: float = float(
        _required(
            contrib.magnitude_attachment_threshold,
            "contributor.magnitude_attachment_threshold",
        )
    )
    # ``leakage.tau_us`` is legitimately allowed to remain ``None`` after
    # resolution -- ``None`` selects the undamped/boxcar limit downstream.
    tau_us_v: Optional[float] = float(leak.tau_us) if leak.tau_us is not None else None

    loaded = load_peaks_impl(file_path)
    peaks = loaded["peaks"]

    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]
    trim_range = stage1.get("trim_range")
    # Window planning reasons over the active FT + its authority noise -- the
    # same grid/amplitude convention Stage 3's peaks were scored on, so the
    # magnitude-attachment / skirt comparisons stay consistent.
    active_ft, active_rms = build_active_grid_with_noise(file_path, trim_range)

    fid = load_fid_from_pipeline_impl(file_path)
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    plan = build_window_plan(
        peaks,
        active_ft.freq_array,
        active_ft.complex_spectrum,
        active_rms,
        acquisition_us=acquisition_us,
        tau_us=tau_us_v,
        probe_freq_mhz=fid.probe_freq_mhz,
        start_us=base_pp.start_us or 0.0,
        edge_m=edge_m_v,
        trim_m=trim_m_v,
        edge_threshold=edge_threshold_v,
        max_window_width_mhz=max_width_v,
        min_freeze_snr=min_freeze_v,
        min_window_half_width_mhz=min_half_v,
        min_window_half_width_points=min_half_points_v,
        magnitude_attachment_threshold=mag_thresh_v,
        max_peaks_per_window=max_peaks_per_window_v,
        max_window_width_points=max_width_points_v,
    )

    save_window_plan_impl(file_path, plan)
    save_window_parameters_impl(file_path, plan.parameters)
    # Persist the resolved WindowPlanningSettings to
    # ``processing_parameters/stage4_windows``. The legacy JSON-encoded
    # ``processing_parameters/window_assignment`` block is kept by
    # ``save_window_parameters_impl`` above as a back-compat shim; the new
    # canonical record below is what the resolver's persisted layer reads.
    save_window_planning_settings_to_h5(
        file_path,
        resolved,
        preset_name=preset_name,
    )
    _update_stage_completion(file_path, "stage4_windows")
    # Re-assignment supersedes any Stage 5 fit built on the old plan.
    invalidate_downstream_stages(file_path, "stage4_windows")

    n_hard = sum(1 for w in plan.windows if w.difficulty == WindowDifficulty.HARD)
    n_free = sum(w.n_free_peaks for w in plan.windows)
    n_fixed = sum(len(w.fixed_contributors) for w in plan.windows)
    logger.info(
        "Stage 4: %d windows (%d hard), %d batches, %d free peaks, "
        "%d fixed contributors, %d dependencies",
        plan.n_windows,
        n_hard,
        plan.n_batches,
        n_free,
        n_fixed,
        len(plan.dependency_edges),
    )
    return {
        "status": "success",
        "plan": plan,
        "n_windows": plan.n_windows,
        "n_hard": n_hard,
        "n_easy": plan.n_windows - n_hard,
        "n_batches": plan.n_batches,
        "n_free_peaks": n_free,
        "n_fixed_contributors": n_fixed,
        "n_dependencies": len(plan.dependency_edges),
        "n_promoted": loaded["n_promoted"],
        "parameters_used": plan.parameters,
        "active_ft": active_ft,
        "active_rms": active_rms,
    }


def save_window_plan_impl(file_path: str, plan: WindowPlan) -> None:
    """Persist a window plan to ``/stage4_windows`` (overwriting any existing)."""
    with h5py.File(file_path, "a") as h5f:
        if "stage4_windows" in h5f:
            del h5f["stage4_windows"]
        grp = h5f.create_group("stage4_windows")
        save_window_plan_to_hdf5(plan, grp)
    logger.info("Saved %d-window plan to %s", plan.n_windows, file_path)


def load_windows_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted Stage 4 window plan (validates structure loudly)."""
    with h5py.File(file_path, "r") as h5f:
        if "stage4_windows" not in h5f:
            raise ValueError(
                "No Stage 4 window plan found. Run assign_windows()/"
                "'windows run' first."
            )
        grp = h5f["stage4_windows"]
        plan = load_window_plan_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
    n_hard = sum(1 for w in plan.windows if w.difficulty == WindowDifficulty.HARD)
    return {
        "plan": plan,
        "n_windows": plan.n_windows,
        "n_hard": n_hard,
        "n_batches": plan.n_batches,
        "creation_time": creation_time,
        "parameters_used": plan.parameters,
    }


def visualize_windows_impl(
    file_path: str,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    y_max_factor: Optional[float] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
) -> Any:
    """Overlay the persisted Stage 4 window plan on the user's spectrum.

    Shows each fit window's span, its free peaks, fixed contributors, and the
    rolling complex-edge coherence statistic that drove the partition. Requires
    Stage 4 completed.
    """
    loaded = load_windows_impl(file_path)
    plan: WindowPlan = loaded["plan"]

    peaks = load_peaks_impl(file_path)["peaks"]
    stage1 = compute_ft_impl(file_path=file_path)
    trim_range = stage1.get("trim_range")
    active_ft, active_rms = build_active_grid_with_noise(file_path, trim_range)

    from ..visualization.window_visualization import plot_window_plan

    if title is None:
        name = Path(file_path).stem
        title = (
            f"Pipeline {name} - Stage 4 Window Assignment "
            f"({plan.n_windows} windows)"
        )

    return plot_window_plan(
        frequencies=active_ft.freq_array,
        complex_spectrum=active_ft.complex_spectrum,
        rms_noise=active_rms,
        peaks=peaks,
        plan=plan,
        figsize=figsize if figsize is not None else (16, 8),
        title=title,
        y_max_factor=y_max_factor if y_max_factor is not None else 25.0,
        backend=backend,
    )


def save_window_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """Save Stage 4 parameters for reuse (JSON under processing_parameters)."""
    with h5py.File(file_path, "a") as h5f:
        grp = h5f.require_group("processing_parameters")
        if "window_assignment" in grp:
            del grp["window_assignment"]
        wa = grp.create_group("window_assignment")
        wa.attrs["parameters"] = json.dumps(parameters, default=str)
        wa.attrs["last_updated"] = datetime.now().isoformat()
    logger.info("Saved Stage 4 parameters to %s", file_path)
