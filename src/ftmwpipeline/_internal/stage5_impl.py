"""
Shared implementation for Stage 5: Per-window fitting.

Orchestration only -- the per-window least-squares core, the conservative
add-one-peak loop, the active-portion FT, the fixed-contributor / DAG walk,
and the local thaw + structural-replan dispatchers live in
:mod:`ftmwpipeline.fitting`. Stage 5 turns the Stage 4
:class:`~ftmwpipeline.core.data_structures.WindowPlan` into a fitted line
list (the persistent :class:`~ftmwpipeline.core.data_structures.SpectrumFit`).

Stage 5 owns no FT settings: the active-portion FT it fits on is computed
on demand from the persisted FID plus the canonical Stage 1 settings (the
same ``start_us``, ``end_us``, ``expf_us``, ``rdc`` the user picked for the
persisted spectrum). Per-bin noise on the active-FT is measured fresh by
running the Stage 2 adaptive estimator on the active-FT magnitude spectrum
(see ``dev-docs/planning/stage5-fitting.md`` § "Spectral domain for the fit"
for why we measure rather than rescale).

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from ..core.data_structures import (
    ComplexFT,
    Sideband,
    SpectrumFit,
    WindowPlan,
)
from ..file_manager import invalidate_downstream_stages
from ..fitting.active_ft import compute_active_ft
from ..fitting.plan_execution import (
    DEFAULT_MAX_REPLAN_ROUNDS,
    DEFAULT_MAX_THAW_ROUNDS,
    DEFAULT_RESIDUAL_EDGE_M,
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    ReplanContext,
    execute_plan,
)
from ..fitting.residual_rescue import (
    DEFAULT_RESCUE_MAX_ROUNDS,
    DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    DEFAULT_RESCUE_SNR_THRESHOLD,
)
from ..fitting.residual_screening import (
    DEFAULT_COHERENCE_CLOSE_THRESHOLD,
    DEFAULT_COHERENCE_CLUSTER_FWHM,
    DEFAULT_COHERENCE_ISOLATED_FWHM,
    DEFAULT_COHERENCE_ISOLATED_THRESHOLD,
)
from ..fitting.result_conversion import plan_fit_outcome_to_spectrum_fit
from ..io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    save_spectrum_fit_to_hdf5,
)
from ..preprocessing.noise_estimation import estimate_noise_adaptive
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage3_impl import (
    _active_acquisition_us,
    _load_canonical_noise,
    load_peaks_impl,
)
from .stage4_impl import load_windows_impl

logger = logging.getLogger(__name__)

# Default per-window tau bound factor k (tau in [tau0/k, tau0*k]) and the
# strongest-line SNR below which a window holds tau fixed (O5-4). These are
# starting values; the planning doc calls for empirical calibration on 2638.
DEFAULT_MAX_DECAY_FACTOR = 5.0
DEFAULT_FIT_TAU_MIN_SNR = 50.0


def _resolve_sideband(value: Sideband | str) -> Sideband:
    """Coerce a string sideband to the :class:`Sideband` enum."""
    if isinstance(value, Sideband):
        return value
    key = str(value).strip().lower()
    if key in ("lower", "lsb"):
        return Sideband.LOWER
    if key in ("upper", "usb"):
        return Sideband.UPPER
    raise ValueError(f"unknown sideband: {value!r}")


def _build_active_ft_inputs(
    file_path: str,
) -> Tuple[
    np.ndarray,  # fid samples
    float,  # sample dt (us)
    float,  # start_us
    float,  # end_us (active end)
    Optional[float],  # expf_us
    float,  # probe_freq_mhz
    Sideband,
    int,  # n_padded
    float,  # acquisition_us (= end - start)
    ComplexFT,  # the user (persisted) ComplexFT
    np.ndarray,  # canonical noise on user grid (for replan context)
]:
    """Gather the Stage 0/1 inputs the active-FT and the replan context need.

    Reads the persisted FID, recomputes the user ComplexFT via the shared
    Stage 1 path (so canonical settings drive what the user sees and what
    Stage 5 fits on), and pulls the canonical Stage 2 noise on the user grid
    for the structural-replan context.
    """
    fid = load_fid_from_pipeline_impl(file_path)
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]
    sample_dt_us = fid.spacing * 1e6
    start_us = float(base_pp.start_us) if base_pp.start_us is not None else 0.0
    end_us = (
        float(base_pp.end_us) if base_pp.end_us is not None else float(fid.duration_us)
    )
    expf_us = float(base_pp.expf_us) if base_pp.expf_us is not None else None
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )
    if acquisition_us <= 0:
        raise ValueError(
            f"Stage 1 canonical settings produce a non-positive active "
            f"acquisition length ({acquisition_us} us)"
        )
    sideband = _resolve_sideband(fid.sideband)

    # n_padded: the zero-padded FFT length the persisted FT uses.
    # base_pp.zpf parameterizes this; the active-FT records alpha for
    # diagnostic only -- the fit itself is independent of n_padded.
    n_active_estimate = int(round(acquisition_us / sample_dt_us))
    n_padded = max(
        2 ** (int(np.log2(max(n_active_estimate, 1))) + 1 + int(base_pp.zpf)),
        n_active_estimate,
    )

    user_rms = _load_canonical_noise(file_path, user_ft)

    return (
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us,
        end_us,
        expf_us,
        float(fid.probe_freq_mhz),
        sideband,
        n_padded,
        acquisition_us,
        user_ft,
        user_rms,
    )


def fit_peaks_impl(
    file_path: str,
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
) -> Dict[str, Any]:
    """Run Stage 5 per-window fitting and persist the result.

    Requires Stage 4 (window assignment) completed (which transitively
    requires Stages 1-3). The fit operates on the active-portion FT
    computed on demand from the persisted FID and the canonical Stage 1
    settings; per-bin noise is measured on the active-FT directly.

    Parameters left as ``None`` fall back to the documented defaults from
    :mod:`ftmwpipeline.fitting`. Returns the persistent
    :class:`SpectrumFit` plus diagnostics; also writes ``/stage5_fitting``
    and marks the stage done.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file.
    tau0_us : float, optional
        Starting / default shared decay constant per window (microseconds).
        Defaults to the Stage 1 ``expf_us`` when set (the apodization
        dominates the line shape), otherwise to ``T/3``.
    fit_tau : bool, optional
        Free vs fixed per-window tau. ``None`` (the default) lets each
        window keep tau free (the strong-anchor common case); a future
        per-window heuristic (O5-4) will override this on weak-only
        windows.
    max_decay_factor : float, optional
        ``tau`` is bounded to ``[tau0_us / k, tau0_us * k]`` (default 5).
    residual_edge_threshold : float, optional
        ``S_coh`` threshold above which a residual edge triggers a thaw
        attempt (default :data:`DEFAULT_RESIDUAL_EDGE_THRESHOLD`).
    residual_edge_m : int, optional
        Band width (in active-FT bins) of the residual-edge coherence
        test (default :data:`DEFAULT_RESIDUAL_EDGE_M`).
    max_thaw_rounds : int, optional
        Maximum local-thaw rounds per window per call (default
        :data:`DEFAULT_MAX_THAW_ROUNDS`).
    max_replan_rounds : int, optional
        Maximum structural-replan rounds per call (default
        :data:`DEFAULT_MAX_REPLAN_ROUNDS`). Pass 0 to disable structural
        renegotiation entirely.
    max_residual_rescue_rounds : int, optional
        Cap on per-window residual-rescue + joint-refit cycles. ``None``
        (the default) resolves to :data:`DEFAULT_RESCUE_MAX_ROUNDS`;
        explicit ``0`` disables the rescue pass entirely (escape hatch
        for diagnostic re-fits). The rescue is a structural part of the
        fit -- it eliminates the conservative loop's systematic
        under-counting of real lines -- and runs on every window's
        post-thaw fit by default. See
        ``dev-docs/planning/stage5-residual-rescue.md``.
    rescue_snr_threshold, rescue_prominence_threshold : float, optional
        Detector thresholds the rescue uses to nominate candidates on
        ``|residual|`` (defaults :data:`DEFAULT_RESCUE_SNR_THRESHOLD` and
        :data:`DEFAULT_RESCUE_PROMINENCE_THRESHOLD`). Ignored when
        ``max_residual_rescue_rounds == 0``.
    rescue_coherence_cluster_fwhm, rescue_coherence_isolated_fwhm,
    rescue_coherence_close_threshold,
    rescue_coherence_isolated_threshold : float, optional
        Phase-coherence filter knobs governing the sliding ratio
        threshold (close→isolated linear ramp by distance to nearest
        existing peak or candidate). Defaults
        :data:`DEFAULT_COHERENCE_CLUSTER_FWHM` /
        :data:`DEFAULT_COHERENCE_ISOLATED_FWHM` /
        :data:`DEFAULT_COHERENCE_CLOSE_THRESHOLD` /
        :data:`DEFAULT_COHERENCE_ISOLATED_THRESHOLD`. All ignored when
        ``max_residual_rescue_rounds == 0``.

    Raises
    ------
    ValueError
        If Stage 4 has not been completed.
    """
    # --- Resolve parameters with their defaults -----------------------------
    max_decay_v = (
        DEFAULT_MAX_DECAY_FACTOR
        if max_decay_factor is None
        else float(max_decay_factor)
    )
    edge_threshold_v = (
        DEFAULT_RESIDUAL_EDGE_THRESHOLD
        if residual_edge_threshold is None
        else float(residual_edge_threshold)
    )
    edge_m_v = (
        DEFAULT_RESIDUAL_EDGE_M if residual_edge_m is None else int(residual_edge_m)
    )
    max_thaw_v = (
        DEFAULT_MAX_THAW_ROUNDS if max_thaw_rounds is None else int(max_thaw_rounds)
    )
    max_replan_v = (
        DEFAULT_MAX_REPLAN_ROUNDS
        if max_replan_rounds is None
        else int(max_replan_rounds)
    )
    # ``None`` resolves to the calibrated default cap; explicit ``0``
    # disables the rescue (kept as an escape hatch). Any positive value
    # runs the B-loop with that round cap.
    rescue_max_v = (
        DEFAULT_RESCUE_MAX_ROUNDS
        if max_residual_rescue_rounds is None
        else max(0, int(max_residual_rescue_rounds))
    )
    rescue_snr_v = (
        DEFAULT_RESCUE_SNR_THRESHOLD
        if rescue_snr_threshold is None
        else float(rescue_snr_threshold)
    )
    rescue_prom_v = (
        DEFAULT_RESCUE_PROMINENCE_THRESHOLD
        if rescue_prominence_threshold is None
        else float(rescue_prominence_threshold)
    )
    rescue_cluster_v = (
        DEFAULT_COHERENCE_CLUSTER_FWHM
        if rescue_coherence_cluster_fwhm is None
        else float(rescue_coherence_cluster_fwhm)
    )
    rescue_isolated_fwhm_v = (
        DEFAULT_COHERENCE_ISOLATED_FWHM
        if rescue_coherence_isolated_fwhm is None
        else float(rescue_coherence_isolated_fwhm)
    )
    rescue_close_thresh_v = (
        DEFAULT_COHERENCE_CLOSE_THRESHOLD
        if rescue_coherence_close_threshold is None
        else float(rescue_coherence_close_threshold)
    )
    rescue_isolated_thresh_v = (
        DEFAULT_COHERENCE_ISOLATED_THRESHOLD
        if rescue_coherence_isolated_threshold is None
        else float(rescue_coherence_isolated_threshold)
    )

    # --- Validate Stage 4 prerequisite up front ----------------------------
    with h5py.File(file_path, "r") as h5f:
        if "stage4_windows" not in h5f:
            raise ValueError(
                "Stage 4 (window assignment) must be completed before "
                "fitting. Run assign_windows()/assign-windows first."
            )

    plan: WindowPlan = load_windows_impl(file_path)["plan"]
    peaks_loaded = load_peaks_impl(file_path)
    peaks = peaks_loaded["peaks"]
    peak_frequencies_mhz = [float(p.frequency) for p in peaks]

    # --- Build the active-FT and measure noise directly on it --------------
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        expf_us,
        probe_freq_mhz,
        sideband,
        n_padded,
        acquisition_us,
        user_ft,
        user_rms,
    ) = _build_active_ft_inputs(file_path)

    active_ft = compute_active_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
    )

    # Stage 2 adaptive estimator on the active-FT magnitude spectrum -- the
    # noise is measured on the same spectrum the fit sees so any FFT
    # normalization choices cancel by construction (D9).
    sort_idx = np.argsort(active_ft.freq_mhz)
    sorted_freq = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    sorted_mag = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    active_noise = estimate_noise_adaptive(sorted_freq, sorted_mag)
    # Un-sort the per-bin noise back onto the active-FT's bin order so it
    # lines up with active_ft.complex_spectrum element-for-element.
    unsort = np.argsort(sort_idx)
    active_rms = np.asarray(active_noise.rms_noise, dtype=float)[unsort]

    # --- tau0 default --------------------------------------------------------
    if tau0_us is None:
        tau0_us_v = float(expf_us) if expf_us is not None else acquisition_us / 3.0
    else:
        tau0_us_v = float(tau0_us)
    if tau0_us_v <= 0:
        raise ValueError(
            f"tau0_us must be positive (got {tau0_us_v}); the active "
            f"acquisition is {acquisition_us} us"
        )
    fit_tau_v = True if fit_tau is None else bool(fit_tau)

    # --- Structural-replan context (skipped when caller asks for 0 rounds) -
    replan_ctx: Optional[ReplanContext]
    if max_replan_v <= 0:
        replan_ctx = None
    else:
        replan_ctx = ReplanContext(
            peaks=peaks,
            persisted_freq_mhz=user_ft.freq_array,
            persisted_complex_spectrum=user_ft.complex_spectrum,
            persisted_rms_noise=user_rms,
            max_replan_rounds=max_replan_v,
        )

    # --- Drive the executor -------------------------------------------------
    rescue_kwargs: Optional[Dict[str, Any]]
    if rescue_max_v > 0:
        rescue_kwargs = {
            "snr_threshold": rescue_snr_v,
            "prominence_threshold": rescue_prom_v,
            "coherence_cluster_fwhm": rescue_cluster_v,
            "coherence_isolated_fwhm": rescue_isolated_fwhm_v,
            "coherence_close_threshold": rescue_close_thresh_v,
            "coherence_isolated_threshold": rescue_isolated_thresh_v,
        }
    else:
        rescue_kwargs = None

    plan_outcome = execute_plan(
        plan,
        active_ft,
        active_rms,
        peak_frequencies_mhz,
        sideband=sideband,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us_v,
        fit_tau=fit_tau_v,
        residual_edge_threshold=edge_threshold_v,
        residual_edge_m=edge_m_v,
        max_thaw_rounds=max_thaw_v,
        conservative_kwargs={
            "max_decay_factor": max_decay_v,
            # Apodization is the physics-informed hard ceiling on tau and the
            # reference point for the stiff lower-side penalty. None disables
            # both (no apodization means no upper bound to enforce).
            "tau_apodization_us": expf_us,
        },
        replan_context=replan_ctx,
        max_residual_rescue_rounds=rescue_max_v,
        rescue_kwargs=rescue_kwargs,
    )

    parameters = {
        "tau0_us": tau0_us_v,
        "fit_tau": fit_tau_v,
        "max_decay_factor": max_decay_v,
        "residual_edge_threshold": edge_threshold_v,
        "residual_edge_m": edge_m_v,
        "max_thaw_rounds": max_thaw_v,
        "max_replan_rounds": max_replan_v,
        "max_residual_rescue_rounds": rescue_max_v,
        "acquisition_us": acquisition_us,
        "active_ft_alpha": float(active_ft.alpha),
        "n_active": int(active_ft.n_active),
        "n_padded": int(active_ft.n_padded),
        "sideband": sideband.value,
    }
    if rescue_max_v > 0:
        parameters.update(
            {
                "rescue_snr_threshold": rescue_snr_v,
                "rescue_prominence_threshold": rescue_prom_v,
                "rescue_coherence_cluster_fwhm": rescue_cluster_v,
                "rescue_coherence_isolated_fwhm": rescue_isolated_fwhm_v,
                "rescue_coherence_close_threshold": rescue_close_thresh_v,
                "rescue_coherence_isolated_threshold": rescue_isolated_thresh_v,
            }
        )
    spectrum_fit: SpectrumFit = plan_fit_outcome_to_spectrum_fit(
        plan_outcome,
        plan,
        sideband=sideband,
        peak_frequencies_mhz=peak_frequencies_mhz,
        acquisition_us=acquisition_us,
        parameters=parameters,
    )

    save_spectrum_fit_impl(file_path, spectrum_fit)
    _update_stage_completion(file_path, "stage5_fitting")
    # A Stage 5 re-fit invalidates nothing today (Stage 5 is the terminal
    # stage); this call is a no-op now and a guard for future stages.
    invalidate_downstream_stages(file_path, "stage5_fitting")

    n_thaw_accepted = sum(1 for e in spectrum_fit.thaw_history if e.accepted)
    n_replan_accepted = sum(1 for e in spectrum_fit.replan_history if e.accepted)
    # Rescue history is not (yet) persisted on SpectrumFit -- pull it off the
    # live PlanFitOutcome for the in-memory return value and log line.
    rescue_events_live = list(plan_outcome.rescue_history)
    n_rescue_events = len(rescue_events_live)
    n_rescue_accepted = sum(1 for e in rescue_events_live if e.accepted)
    n_rescue_added_total = sum(e.n_rescue_added for e in rescue_events_live)
    n_rescue_origin_pruned_total = sum(
        e.n_pruned_rescue_origin for e in rescue_events_live
    )
    logger.info(
        "Stage 5: %d windows, %d fitted peaks; thaw %d/%d accepted, "
        "rescue %d/%d rounds accepted (added %d peaks, %d rescue-origin pruned), "
        "%d structural replans accepted (revision %d)",
        spectrum_fit.n_windows,
        spectrum_fit.n_fitted_peaks,
        n_thaw_accepted,
        len(spectrum_fit.thaw_history),
        n_rescue_accepted,
        n_rescue_events,
        n_rescue_added_total,
        n_rescue_origin_pruned_total,
        n_replan_accepted,
        spectrum_fit.final_plan_revision,
    )
    return {
        "status": "success",
        "fit": spectrum_fit,
        "n_windows": spectrum_fit.n_windows,
        "n_fitted_peaks": spectrum_fit.n_fitted_peaks,
        "n_thaw_events": len(spectrum_fit.thaw_history),
        "n_thaw_accepted": n_thaw_accepted,
        "n_rescue_events": n_rescue_events,
        "n_rescue_accepted": n_rescue_accepted,
        "n_rescue_added": n_rescue_added_total,
        "n_rescue_origin_pruned": n_rescue_origin_pruned_total,
        "n_replan_events": len(spectrum_fit.replan_history),
        "n_replan_accepted": n_replan_accepted,
        "final_plan_revision": spectrum_fit.final_plan_revision,
        "parameters_used": parameters,
        "active_ft": active_ft,
        "rescue_events": rescue_events_live,
    }


def save_spectrum_fit_impl(file_path: str, fit: SpectrumFit) -> None:
    """Persist a :class:`SpectrumFit` to ``/stage5_fitting`` (overwriting)."""
    with h5py.File(file_path, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(fit, grp)
    logger.info(
        "Saved Stage 5 fit (%d windows, %d peaks) to %s",
        fit.n_windows,
        fit.n_fitted_peaks,
        file_path,
    )


def load_fit_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted Stage 5 fit (validates structure loudly)."""
    with h5py.File(file_path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found. Run fit_peaks()/fit-peaks first.")
        grp = h5f["stage5_fitting"]
        fit = load_spectrum_fit_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
    return {
        "fit": fit,
        "n_windows": fit.n_windows,
        "n_fitted_peaks": fit.n_fitted_peaks,
        "creation_time": creation_time,
        "parameters_used": fit.parameters,
        "final_plan_revision": fit.final_plan_revision,
    }


def visualize_fit_impl(
    file_path: str,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    window_id: Optional[int] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
) -> Any:
    """Overlay the persisted Stage 5 fit on the user spectrum.

    Re-evaluates the fitted model on the persisted (high-res) frequency
    grid so the overlay reads at full resolution -- the model is
    grid-agnostic, the active-FT was only used for the fit itself.
    Requires Stage 5 completed.
    """
    loaded = load_fit_impl(file_path)
    fit: SpectrumFit = loaded["fit"]
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    user_rms = _load_canonical_noise(file_path, user_ft)
    fid = load_fid_from_pipeline_impl(file_path)
    sideband = _resolve_sideband(fid.sideband)
    base_pp = user_ft.metadata["processing_params"]
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # The fit lives in active-FT amplitude units (``dt_us * rfft(active)``);
    # the persisted FT uses ``rfft(padded) / original_length * 10**units_power``.
    # Convert at plot time so the model overlay reads at the persisted scale.
    sample_dt_us = float(fid.spacing * 1e6)
    scale_factor = float(10 ** int(base_pp.units_power))
    model_amplitude_scale = scale_factor / (float(fid.n_points) * sample_dt_us)

    from ..visualization.fit_visualization import plot_spectrum_fit

    if title is None:
        name = Path(file_path).stem
        scope = (
            f"window {window_id}"
            if window_id is not None
            else f"{fit.n_windows} windows"
        )
        title = f"Pipeline {name} - Stage 5 Fit ({scope})"

    start_us = float(base_pp.start_us) if base_pp.start_us is not None else 0.0

    return plot_spectrum_fit(
        frequencies=user_ft.freq_array,
        complex_spectrum=user_ft.complex_spectrum,
        rms_noise=user_rms,
        fit=fit,
        sideband=sideband,
        acquisition_us=acquisition_us,
        figsize=figsize if figsize is not None else (16, 10),
        title=title,
        window_id=window_id,
        backend=backend,
        model_amplitude_scale=model_amplitude_scale,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        start_us=start_us,
    )
