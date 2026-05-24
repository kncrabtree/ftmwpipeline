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

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py

from ..core.data_structures import ComplexFT, WindowDifficulty, WindowPlan
from ..io.window_serialization import (
    load_window_plan_from_hdf5,
    save_window_plan_to_hdf5,
)
from ..preprocessing.edge_coherence import (
    DEFAULT_EDGE_M,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_TRIM_M,
)
from ..file_manager import invalidate_downstream_stages
from ..preprocessing.window_planning import (
    DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    DEFAULT_MAX_WINDOW_WIDTH_MHZ,
    DEFAULT_MIN_FREEZE_SNR,
    DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ,
    build_window_plan,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage3_impl import (
    _active_acquisition_us,
    _load_canonical_noise,
    load_peaks_impl,
)

logger = logging.getLogger(__name__)


def assign_windows_impl(
    file_path: str,
    edge_m: Optional[int] = None,
    trim_m: Optional[int] = None,
    edge_threshold: Optional[float] = None,
    max_window_width_mhz: Optional[float] = None,
    min_freeze_snr: Optional[float] = None,
    min_window_half_width_mhz: Optional[float] = None,
    magnitude_attachment_threshold: Optional[float] = None,
    tau_us: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the Stage 4 window plan from the promoted Stage 3 peaks and persist it.

    Requires Stage 3 (peak detection) completed. Operates on the Stage 1
    canonical spectrum and the canonical Stage 2 noise; consumes only the
    peaks flagged ``properties['promoted']``.

    Parameters left as ``None`` fall back to the documented defaults from
    :mod:`ftmwpipeline.preprocessing.edge_coherence` /
    :mod:`ftmwpipeline.preprocessing.window_planning`. Returns the plan plus
    diagnostics; also writes ``/stage4_windows`` and marks the stage done.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file.
    edge_m : int, optional
        Rolling-scan coherence band width (default 64).
    trim_m : int, optional
        Trim-refinement band width (default 32).
    edge_threshold : float, optional
        ``S_coh`` threshold ``T_edge`` (default 3.0).
    max_window_width_mhz : float, optional
        Width cap; wider windows are HARD and get a split proposal (default 40).
    min_freeze_snr : float, optional
        Freeze-eligibility SNR cutoff for fixed contributors (default 50).
    min_window_half_width_mhz : float, optional
        Minimum half-width of a window around an isolated weak line (default 2).
    tau_us : float, optional
        Assumed decay constant for leakage reach (default: undamped/boxcar).

    Raises
    ------
    ValueError
        If Stage 3 has not been completed.
    """
    edge_m_v = DEFAULT_EDGE_M if edge_m is None else int(edge_m)
    trim_m_v = DEFAULT_TRIM_M if trim_m is None else int(trim_m)
    edge_threshold_v = (
        DEFAULT_EDGE_THRESHOLD if edge_threshold is None else float(edge_threshold)
    )
    max_width_v = (
        DEFAULT_MAX_WINDOW_WIDTH_MHZ
        if max_window_width_mhz is None
        else float(max_window_width_mhz)
    )
    min_freeze_v = (
        DEFAULT_MIN_FREEZE_SNR if min_freeze_snr is None else float(min_freeze_snr)
    )
    min_half_v = (
        DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ
        if min_window_half_width_mhz is None
        else float(min_window_half_width_mhz)
    )
    mag_thresh_v = (
        DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD
        if magnitude_attachment_threshold is None
        else float(magnitude_attachment_threshold)
    )

    with h5py.File(file_path, "r") as h5f:
        if "stage3_peaks" not in h5f:
            raise ValueError(
                "Stage 3 (peak detection) must be completed before window "
                "assignment. Run detect_peaks()/detect-peaks first."
            )

    loaded = load_peaks_impl(file_path)
    peaks = loaded["peaks"]

    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]
    user_rms = _load_canonical_noise(file_path, user_ft)

    fid = load_fid_from_pipeline_impl(file_path)
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    plan = build_window_plan(
        peaks,
        user_ft.freq_array,
        user_ft.complex_spectrum,
        user_rms,
        acquisition_us=acquisition_us,
        tau_us=tau_us,
        probe_freq_mhz=fid.probe_freq_mhz,
        start_us=base_pp.start_us or 0.0,
        edge_m=edge_m_v,
        trim_m=trim_m_v,
        edge_threshold=edge_threshold_v,
        max_window_width_mhz=max_width_v,
        min_freeze_snr=min_freeze_v,
        min_window_half_width_mhz=min_half_v,
        magnitude_attachment_threshold=mag_thresh_v,
    )

    save_window_plan_impl(file_path, plan)
    save_window_parameters_impl(file_path, plan.parameters)
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
        "user_ft": user_ft,
        "user_rms": user_rms,
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
                "assign-windows first."
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
    user_ft: ComplexFT = stage1["complex_ft"]
    user_rms = _load_canonical_noise(file_path, user_ft)

    from ..visualization.window_visualization import plot_window_plan

    if title is None:
        name = Path(file_path).stem
        title = (
            f"Pipeline {name} - Stage 4 Window Assignment "
            f"({plan.n_windows} windows)"
        )

    return plot_window_plan(
        frequencies=user_ft.freq_array,
        complex_spectrum=user_ft.complex_spectrum,
        rms_noise=user_rms,
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
