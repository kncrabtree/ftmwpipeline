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
same ``start_us``, ``end_us``, ``rdc`` the user picked for the persisted
spectrum; the canonical FT is unapodized and native-length). Per-bin noise on
the active-FT is measured fresh by
running the Stage 2 adaptive estimator on the active-FT magnitude spectrum
(see ``dev-docs/planning/stage5-fitting.md`` § "Spectral domain for the fit"
for why we measure rather than rescale).

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import h5py
import numpy as np

from ..core.data_structures import (
    ComplexFT,
    FittedPeak,
    FittingResult,
    Sideband,
    SpectrumFit,
    WindowPlan,
)
from ..core.stage_fit_settings import (
    ShapeSpec,
    SpurSubSettings,
    StageFitSettings,
    load_preset,
)
from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
from ..file_manager import invalidate_downstream_stages
from ..fitting.active_ft import compute_active_ft
from ..fitting.clock_lattice import ClockLattice, build_clock_lattice
from ..fitting.peak_model import PeakShape
from ..fitting.plan_execution import (
    ReplanContext,
    execute_plan,
)
from ..fitting.result_conversion import (
    plan_fit_outcome_to_spectrum_fit,
)
from ..fitting.spur_detection import (
    SpurSet,
    build_spur_set,
    make_band_power_probe,
    make_chirp_response_probe,
    make_decay_probe,
)
from ..io.fid_serialization import load_acquisition_segments_from_hdf5
from ..fitting.tau_calibration import (
    TauCalibrationResult,
    band_majority_for_frequency,
)
from ..io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    save_spectrum_fit_to_hdf5,
)
from ..io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    read_recommended_clock_sources,
    read_stage2b_recommended_shape,
    save_stage_fit_settings_to_h5,
)
from ..preprocessing.noise_estimation import estimate_active_ft_noise
from .active_ft_support import _persisted_scatter_knobs, build_active_grid_with_noise
from .deprecation import warn_legacy_kwargs
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage2b_g_impl import (
    load_tau_G_calibration_impl,
    tau_G_calibration_present,
)
from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
from .stage3_impl import (
    _active_acquisition_us,
    load_peaks_impl,
)
from .stage4_impl import load_windows_impl

logger = logging.getLogger(__name__)

# Default per-window tau bound factor k (tau in [tau0/k, tau0*k]). The free-τ
# SNR floor lives with the gate it drives (fitting.window_fit, composed against
# the weak-window floor); it is not duplicated here.
DEFAULT_MAX_DECAY_FACTOR = 5.0


def annotate_lattice_matches(
    spectrum_fit: SpectrumFit, lattice: Optional[ClockLattice]
) -> None:
    """Stamp ``clock_lattice`` on every fitted peak whose frequency matches ``lattice``.

    Called after the fit is assembled but before persistence. Sets
    :attr:`~ftmwpipeline.core.data_structures.FittedPeak.clock_lattice` to the
    matched :attr:`~ftmwpipeline.fitting.clock_lattice.LatticePoint.identity`
    string on both the per-window and the merged global peak lists.  Annotation
    is purely informational; it has no effect on the fit.

    When ``lattice`` is ``None`` (no clock declaration), every peak keeps its
    default ``clock_lattice = None`` and the function is a no-op.
    """
    if lattice is None:
        return
    # Annotate both the global list and the per-window copies (they share the
    # same FittedPeak objects in practice, but iterate both to be safe).
    for peak in spectrum_fit.fitted_peaks:
        point = lattice.match(peak.frequency_mhz)
        if point is not None:
            peak.clock_lattice = point.identity
    for wf in spectrum_fit.window_fits:
        for peak in wf.fitted_peaks:
            point = lattice.match(peak.frequency_mhz)
            if point is not None:
                peak.clock_lattice = point.identity


def _resolve_tau_calibration_for_fit(
    persisted: Optional[TauCalibrationResult],
    tau_maj_override_us: Optional[float],
    sigma_tau_override_us: Optional[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """Resolve which (tau_maj_us, sigma_tau_us) pair drives the Stage 5 fit.

    Precedence: explicit ``(tau_maj_override_us, sigma_tau_override_us)``
    beats the persisted Stage 2b calibration, which beats no calibration at
    all. The two override knobs are an atomic pair -- supplying only one is
    ambiguous (the bidirectional Gaussian-prior penalty needs both ``tau_maj``
    and ``sigma_tau`` to be meaningful) and raises ``ValueError``. Both
    must be strictly positive when set.

    Returns
    -------
    tau_maj_us, sigma_tau_us : float or None
        Resolved values forwarded into ``conservative_kwargs``; both are
        ``None`` when no calibration is in play.
    source : str
        ``"override"``, ``"persisted"``, or ``"none"`` -- diagnostic label
        recorded in ``parameters_used`` so downstream consumers (and the
        fit log) can tell which path was taken.
    """
    has_tau = tau_maj_override_us is not None
    has_sigma = sigma_tau_override_us is not None
    if has_tau ^ has_sigma:
        raise ValueError(
            "tau_maj_override_us and sigma_tau_override_us must be supplied "
            "together; supplying only one is ambiguous"
        )
    if has_tau:
        tau_v = float(tau_maj_override_us)  # type: ignore[arg-type]
        sigma_v = float(sigma_tau_override_us)  # type: ignore[arg-type]
        if tau_v <= 0.0 or sigma_v <= 0.0:
            raise ValueError(
                f"tau_maj_override_us and sigma_tau_override_us must be "
                f"positive (got tau_maj={tau_v}, sigma_tau={sigma_v})"
            )
        return tau_v, sigma_v, "override"
    if persisted is not None:
        return (
            float(persisted.tau_maj_us),
            float(persisted.sigma_tau_us),
            "persisted",
        )
    return None, None, "none"


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
    float,  # probe_freq_mhz
    Sideband,
    int,  # n_padded
    float,  # acquisition_us (= end - start)
    ComplexFT,  # the user (persisted) ComplexFT
    Optional[Tuple[float, float]],  # trim_range (analysis band)
]:
    """Gather the Stage 0/1 inputs the active-FT and the replan context need.

    Reads the persisted FID, recomputes the user ComplexFT via the shared
    Stage 1 path (so canonical settings drive what Stage 5 fits on), and
    returns the canonical trim range so the active-grid replan/visualization
    can be rebuilt on the analysis band.
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
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )
    if acquisition_us <= 0:
        raise ValueError(
            f"Stage 1 canonical settings produce a non-positive active "
            f"acquisition length ({acquisition_us} us)"
        )
    sideband = _resolve_sideband(fid.sideband)

    # n_padded: the canonical full-record FT input length (the native FID
    # length -- the persisted FT is unpadded). The active-FT records alpha for
    # diagnostic only; the fit itself is independent of n_padded.
    n_padded = int(np.asarray(fid.data).size)

    trim_range = stage1.get("trim_range")

    return (
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us,
        end_us,
        float(fid.probe_freq_mhz),
        sideband,
        n_padded,
        acquisition_us,
        user_ft,
        trim_range,
    )


def _required_float(value: Optional[float], name: str) -> float:
    """Coerce a post-resolve field that must be filled into ``float``."""
    if value is None:
        raise AssertionError(
            f"resolved StageFitSettings.{name} is None; missing hard default"
        )
    return float(value)


def _required_int(value: Optional[int], name: str) -> int:
    """Coerce a post-resolve field that must be filled into ``int``."""
    if value is None:
        raise AssertionError(
            f"resolved StageFitSettings.{name} is None; missing hard default"
        )
    return int(value)


def _required_bool(value: Optional[bool], name: str) -> bool:
    """Coerce a post-resolve field that must be filled into ``bool``."""
    if value is None:
        raise AssertionError(
            f"resolved StageFitSettings.{name} is None; missing hard default"
        )
    return bool(value)


def _required_str(value: Optional[str], name: str) -> str:
    """Coerce a post-resolve field that must be filled into ``str``."""
    if value is None:
        raise AssertionError(
            f"resolved StageFitSettings.{name} is None; missing hard default"
        )
    return str(value)


def _is_survival_dust(peak: FittedPeak, floor: float) -> bool:
    """Return True when ``peak`` is auto-origin dust below the SNR floor.

    User-origin peaks are immune; ``None`` / NaN SNR is kept unconditionally
    (conservative).
    """
    import math

    snr = peak.snr
    if peak.origin == "user" or snr is None:
        return False
    if isinstance(snr, float) and math.isnan(snr):
        return False
    return snr < floor


def _survival_prune_window(
    wf: FittingResult,
    floor: float,
    refit_window: Callable[[FittingResult, List[float]], FittingResult],
    pruned_records: List[Dict[str, Any]],
) -> Tuple[Optional[FittingResult], int]:
    """Prune one window's sub-floor dust to a fixpoint; return (result, n_refits).

    Removes the single lowest-SNR dust peak, refits the survivors, and
    **re-classifies** -- because the refit re-estimates the survivors free of
    the removed line, a peak that was sub-floor only because a neighbour stole
    its amplitude can recover and is then kept, while a survivor that the refit
    pushes below the floor is caught on the next pass (the single-pass classify
    left such a survivor sub-floor). Removing one peak per pass (not the whole
    initial dust set at once) is the over-pruning-cascade guard: each borderline
    line gets a fresh refit before its own removal is considered.

    Returns ``(result, n_refits)``; ``result`` is ``None`` when the window
    cascades to empty (its last remaining line is itself dust) and should be
    dropped. ``n_refits`` is 0 when the window had no dust (``result is wf``).
    """
    current = wf
    n_refits = 0
    while True:
        peaks = current.fitted_peaks
        dust = [p for p in peaks if _is_survival_dust(p, floor)]
        if not dust:
            return current, n_refits
        worst = min(dust, key=lambda p: cast(float, p.snr))
        pruned_records.append(
            {
                "window_id": (
                    int(current.window_id) if current.window_id is not None else -1
                ),
                "frequency_mhz": float(worst.frequency_mhz),
                "snr": float(cast(float, worst.snr)),  # finite by _is_survival_dust
            }
        )
        if len(peaks) == 1:
            # The lone remaining line is itself dust -> drop the window. (A
            # pre-existing K=0 window has no dust and never reaches here, so the
            # window-construction signal is preserved.)
            return None, n_refits
        current = refit_window(current, [float(worst.frequency_mhz)])
        n_refits += 1


def apply_snr_survival_prune(
    fit: SpectrumFit,
    floor: float,
    *,
    refit_window: Callable[[FittingResult, List[float]], FittingResult],
) -> None:
    """Prune fitted peaks below the SNR survival floor (in-place).

    For each window, classifies its automatic-origin peaks with finite
    ``snr < floor`` as dust (``origin == "user"`` peaks are immune; ``None`` /
    NaN SNR is kept) and prunes to a fixpoint (see
    :func:`_survival_prune_window`):

    - **no dust** -> the window is untouched;
    - **partial dust** -> the lowest-SNR dust line is removed and the survivors
      are **refitted** via ``refit_window`` so their point estimates,
      covariance, and χ²ᵣ are re-estimated free of the removed line (not a
      marginal slice of the stale joint fit); the result is re-classified and
      the process repeats until no survivor is sub-floor;
    - **cascade to empty** -> a window whose every line is sub-floor (after the
      refits) is dropped from ``window_fits``.

    Re-classifying after each refit closes the single-pass gap where a refit
    pushed a *surviving* peak below the floor and nothing re-checked it.

    Finally rebuilds ``fit.fitted_peaks`` sorted ascending by ``frequency_mhz``
    and records the prune in ``fit.diagnostics["peak_survival"]``.

    Parameters
    ----------
    fit :
        The :class:`SpectrumFit` to prune in place.
    floor :
        Absolute SNR survival floor; auto peaks below it are dust.
    refit_window :
        Callable ``(wf, remove_freqs) -> FittingResult`` that re-fits ``wf``
        with the given molecular frequencies removed and returns the new
        per-window result. Invoked once per removed line. In production this
        routes through the bare in-memory window-refit core
        (:func:`~ftmwpipeline._internal.stage6_impl.refit_window_core`), which
        reconstructs the frozen background, replays the baseline, and
        re-converges the survivors honestly.
    """
    pruned_records: List[Dict[str, Any]] = []
    dropped_window_ids: List[int] = []

    surviving_windows: List[FittingResult] = []
    for wf in fit.window_fits:
        result, _ = _survival_prune_window(wf, floor, refit_window, pruned_records)
        if result is None:
            dropped_window_ids.append(
                int(wf.window_id) if wf.window_id is not None else -1
            )
        else:
            surviving_windows.append(result)

    fit.window_fits = surviving_windows

    # Rebuild the global sorted line list from surviving per-window peaks.
    all_peaks: List[FittedPeak] = [p for wf in fit.window_fits for p in wf.fitted_peaks]
    all_peaks.sort(key=lambda p: p.frequency_mhz)
    fit.fitted_peaks = all_peaks

    fit.diagnostics["peak_survival"] = {
        "snr_floor": float(floor),
        "n_pruned": len(pruned_records),
        "pruned": pruned_records,
        "dropped_window_ids": dropped_window_ids,
    }


def amplitude_vif(peak: FittedPeak) -> Optional[float]:
    """Diagonal amplitude variance-inflation factor ``(amp_err / amp) * snr``.

    The overfit discriminant: ~1 for an identifiable line, >> 1 when a line is
    degenerate with a sub-resolution neighbour (the pair *sum* is constrained,
    neither amplitude individually). A pure function of already-persisted
    per-peak fields -- no covariance matrix needed. Returns ``None`` when any
    input is missing / non-finite or the amplitude is zero.
    """
    import math

    amp = float(peak.amplitude)
    amp_err = peak.amplitude_error
    snr = peak.snr
    if amp_err is None or snr is None:
        return None
    if not (math.isfinite(amp) and math.isfinite(amp_err) and math.isfinite(snr)):
        return None
    if abs(amp) <= 0.0:
        return None
    return abs(amp_err / amp) * float(snr)


def _merged_seed_for_pair(
    wf: FittingResult,
    pa: FittedPeak,
    pb: FittedPeak,
    snap_tol_mhz: float,
) -> Tuple[float, float, float]:
    """Seed (frequency, amplitude, phase) for the line that collapses a pair.

    Defaults to the amplitude-weighted centroid frequency and the summed
    amplitude. When the window recorded a :class:`DoubletAlternativeInfo` for
    exactly this pair (order-independent, within ``snap_tol_mhz``) whose merged
    refit succeeded, reuses its converged ``merged_{frequency,amplitude,phase}``
    instead -- the doublet-adjudication pass already fit the one-line model.
    """
    import math

    fa = float(pa.frequency_mhz)
    fb = float(pb.frequency_mhz)
    aw = abs(float(pa.amplitude))
    bw = abs(float(pb.amplitude))
    total = aw + bw if (aw + bw) > 0 else 1.0
    centroid_freq = (fa * aw + fb * bw) / total
    centroid_amp = float(pa.amplitude) + float(pb.amplitude)

    merge_freq, merge_amp, merge_phase = centroid_freq, centroid_amp, 0.0
    for da in getattr(wf, "doublet_alternatives", []):
        a, b = float(da.frequency_a_mhz), float(da.frequency_b_mhz)
        pair_match = (abs(a - fa) <= snap_tol_mhz and abs(b - fb) <= snap_tol_mhz) or (
            abs(a - fb) <= snap_tol_mhz and abs(b - fa) <= snap_tol_mhz
        )
        if (
            pair_match
            and da.merged_success
            and math.isfinite(float(da.merged_frequency_mhz))
        ):
            merge_freq = float(da.merged_frequency_mhz)
            if math.isfinite(float(da.merged_amplitude)):
                merge_amp = float(da.merged_amplitude)
            if math.isfinite(float(da.merged_phase)):
                merge_phase = float(da.merged_phase)
            break
    return merge_freq, merge_amp, merge_phase


def apply_vif_collapse(
    fit: SpectrumFit,
    *,
    vif_threshold: float,
    max_sep_res: float,
    res_element_mhz: float,
    sideband: Sideband,
    refit_collapse: Callable[
        [FittingResult, List[float], List[float], List[Any]], FittingResult
    ],
    snap_tol_mhz: float = 0.05,
    max_iterations: int = 5,
    max_post_chi2r: float = float("inf"),
) -> None:
    """Collapse degenerate sub-resolution overfit pairs (in-place).

    For each window, a close pair is collapsed to one line when a member's
    amplitude VIF exceeds ``vif_threshold`` *and* the pair sits within
    ``max_sep_res`` resolution elements (``res_element_mhz = 1 / T_active``).
    The collapse fires on VIF + separation **alone** and overrides chi^2 / AICc
    unconditionally: at high SNR chi^2 is a lineshape-fidelity floor, so a
    near-degenerate second line absorbs lineshape mismodeling and chi^2 / AICc
    *reward* the spurious split. The window chi^2_r *rising* on collapse is
    expected and must not veto it.

    The separation bound (``max_sep_res``, the half-resolution-element physical
    limit) is the real guard: two lines closer than that are unresolvable, so a
    split there is spurious regardless of VIF, and every genuine doublet in the
    calibration truth set sits well outside it (>= 0.82 res). ``vif_threshold``
    is therefore only a modest sanity floor that keeps a window whose close
    peaks are both well determined (not degenerate) from collapsing.

    Pairs are matched greedily from the highest-VIF peak down: each high-VIF
    peak pairs with its nearest unused neighbour within the separation bound
    (every VIF>>1 line comes in a degenerate pair, so the partner is
    structural). Each window with at least one pair is refitted with all paired
    members removed and one merged seed added per pair (origin ``"auto"`` -- an
    automatic decision, not a human edit). The merged seed reuses the recorded
    doublet-alternative when present (see :func:`_merged_seed_for_pair`).

    **Iterated to convergence.** A collapse refit is NLS over the surviving
    seeds; on a dense, high-SNR window two seeds can re-converge into a *new*
    degenerate split that a single pass would leave behind (only to be flagged
    for attention later). The sweep therefore repeats until a full pass finds no
    collapsible pair, capped at ``max_iterations`` (the cap is a backstop
    against a window that oscillates collapse<->re-split; any residual pair is
    still surfaced by the Stage 6 ``overfit_vif`` attention reason).

    Parameters
    ----------
    fit :
        The :class:`SpectrumFit` to collapse in place (post SNR-floor prune).
    vif_threshold, max_sep_res :
        Collapse gate: amplitude VIF bound and maximum pair separation in
        resolution elements.
    res_element_mhz :
        One resolution element ``1 / T_active`` (MHz).
    sideband :
        Pipeline sideband (for the molecular -> baseband-offset seed mapping).
    refit_collapse :
        Callable ``(wf, remove_freqs, add_freqs, add_seeds) -> FittingResult``
        that re-fits ``wf`` with the paired peaks removed and the merged seeds
        added (origin ``"auto"``). In production this routes through
        :func:`~ftmwpipeline._internal.stage6_impl.refit_window_core`.
    snap_tol_mhz :
        Tolerance for matching a pair to a recorded doublet alternative.
    max_iterations :
        Maximum collapse sweeps over the window set (convergence backstop).
    max_post_chi2r :
        Catastrophic-merge veto: if a window's post-merge reduced chi^2 exceeds
        this, the merge is reverted (the split is kept and will flag
        ``overfit_vif``) -- the data overwhelmingly demands two components.
        ``inf`` disables the veto. Raw chi2r, not SNR-normalized eps (the D10
        lineshape floor saturates eps at high SNR).
    """
    from ..fitting.peak_model import ModelPeak, sideband_sign

    max_sep_mhz = max_sep_res * res_element_mhz
    s = sideband_sign(sideband)

    def _pairs_for_window(wf: FittingResult) -> List[Tuple[int, int]]:
        """Greedy high-VIF nearest-neighbour pairs within the separation bound."""
        peaks = wf.fitted_peaks
        if len(peaks) < 2:
            return []
        vifs = [amplitude_vif(p) for p in peaks]
        # High-VIF peaks first, so the most degenerate pairs form before a
        # shared neighbour is consumed by a weaker pairing.
        high_vif_order = sorted(
            (i for i, v in enumerate(vifs) if v is not None and v > vif_threshold),
            key=lambda i: -(vifs[i] or 0.0),
        )
        used: set[int] = set()
        pairs: List[Tuple[int, int]] = []
        for i in high_vif_order:
            if i in used:
                continue
            best_j: Optional[int] = None
            best_d = float("inf")
            for j in range(len(peaks)):
                if j == i or j in used:
                    continue
                d = abs(float(peaks[i].frequency_mhz) - float(peaks[j].frequency_mhz))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= max_sep_mhz:
                pairs.append((i, best_j))
                used.add(i)
                used.add(best_j)
        return pairs

    collapse_records: List[Dict[str, Any]] = []
    veto_records: List[Dict[str, Any]] = []
    vetoed_ids: set[int] = set()
    n_iterations = 0

    # Iterate until a full sweep collapses nothing: an NLS collapse refit can
    # re-converge two seeds into a fresh degenerate split, so one pass is not
    # enough. ``max_iterations`` caps a window that oscillates collapse<->split.
    for _ in range(max(1, max_iterations)):
        any_collapsed = False
        new_window_fits = []
        for wf in fit.window_fits:
            wid = int(wf.window_id) if wf.window_id is not None else -1
            peaks = wf.fitted_peaks
            center = None
            if wf.window is not None and wf.window.freq_range is not None:
                lo, hi = wf.window.freq_range
                center = 0.5 * (lo + hi)
            if center is None or wid in vetoed_ids:
                new_window_fits.append(wf)
                continue

            pairs = _pairs_for_window(wf)
            if not pairs:
                new_window_fits.append(wf)
                continue

            vifs = [amplitude_vif(p) for p in peaks]
            remove_freqs = []
            add_freqs = []
            add_seeds = []
            pending: List[Dict[str, Any]] = []
            for i, j in pairs:
                pa, pb = peaks[i], peaks[j]
                merge_freq, merge_amp, merge_phase = _merged_seed_for_pair(
                    wf, pa, pb, snap_tol_mhz
                )
                offset = float(s * (merge_freq - center))
                add_seeds.append(
                    ModelPeak(
                        amplitude=max(abs(merge_amp), 1e-30),
                        offset_mhz=offset,
                        phase=merge_phase,
                    )
                )
                remove_freqs.extend([float(pa.frequency_mhz), float(pb.frequency_mhz)])
                add_freqs.append(merge_freq)
                pending.append(
                    {
                        "window_id": wid,
                        "frequency_a_mhz": float(pa.frequency_mhz),
                        "frequency_b_mhz": float(pb.frequency_mhz),
                        "vif_a": vifs[i],
                        "vif_b": vifs[j],
                        "separation_res": (
                            abs(float(pa.frequency_mhz) - float(pb.frequency_mhz))
                            / res_element_mhz
                            if res_element_mhz > 0
                            else None
                        ),
                        "merged_frequency_mhz": float(merge_freq),
                    }
                )

            new_wf = refit_collapse(wf, remove_freqs, add_freqs, add_seeds)

            # Catastrophic-merge veto: if collapsing the pair leaves a window the
            # 1-line model fits this badly, the data overwhelmingly demands two
            # components, so keep the split (it flags ``overfit_vif`` for human
            # review) rather than ship a broken fit. Raw post-merge chi2r (not the
            # SNR-normalized eps, which the D10 lineshape floor saturates at high
            # SNR). The window is not retried in later iterations.
            post_chi2r_raw = getattr(new_wf, "reduced_chi2", None)
            post_chi2r = (
                float(post_chi2r_raw) if post_chi2r_raw is not None else float("inf")
            )
            if post_chi2r > max_post_chi2r:
                for rec in pending:
                    rec["post_merge_chi2r"] = post_chi2r
                veto_records.extend(pending)
                vetoed_ids.add(wid)
                new_window_fits.append(wf)
                continue

            collapse_records.extend(pending)
            new_window_fits.append(new_wf)
            any_collapsed = True

        fit.window_fits = new_window_fits
        if not any_collapsed:
            break
        n_iterations += 1

    all_peaks: List[FittedPeak] = [p for wf in fit.window_fits for p in wf.fitted_peaks]
    all_peaks.sort(key=lambda p: p.frequency_mhz)
    fit.fitted_peaks = all_peaks

    fit.diagnostics["vif_collapse"] = {
        "vif_threshold": float(vif_threshold),
        "max_separation_res": float(max_sep_res),
        "res_element_mhz": float(res_element_mhz),
        "max_post_chi2r": float(max_post_chi2r),
        "n_collapsed_pairs": len(collapse_records),
        "n_vetoed_pairs": len(veto_records),
        "n_iterations": n_iterations,
        "collapses": collapse_records,
        "vetoed": veto_records,
    }


# Decay-probe source tags that confidently mark an instrumental (non-molecular)
# spur: ``flat`` (non-decaying across the FID -- a real line decays) and
# ``saturated`` (ADC clipping). ``narrow`` alone is ambiguous (erratic real
# lines read as narrow) and ``drift`` is left to clock declaration, so neither
# triggers a single-line-window drop on its own.
_INSTRUMENTAL_SPUR_TAGS = ("flat", "saturated")


def apply_window_cleanup(
    fit: SpectrumFit,
    *,
    spur_set: Optional[Any],
    res_element_mhz: float,
    drop_empty: bool,
    drop_spur_only: bool,
) -> None:
    """Drop non-product windows at the end of Stage 5 (in-place).

    Two conservative cuts, both recorded in ``fit.diagnostics["window_cleanup"]``:

    1. **Empty windows.** A window with no fitted peak (``K == 0``) carries no
       product -- it is dropped when ``drop_empty`` is set.
    2. **Spur-only windows.** A window whose *single* fitted peak sits within the
       spur's residual-mask half-width (or one resolution element, whichever is
       larger) of a confidently-instrumental gated spur (a
       :data:`_INSTRUMENTAL_SPUR_TAGS` decay-probe verdict -- an ADC image or a
       declared clock tone leaking past its mask) is dropped when
       ``drop_spur_only`` is set. A ``user``-origin peak is never dropped, and a
       multi-line window is never touched (the second line makes it ambiguous).

    ``fitted_peaks`` is rebuilt (sorted) from the surviving windows.
    """
    instrumental = (
        [
            sp
            for sp in spur_set.spurs
            if any(tag in (sp.source or "") for tag in _INSTRUMENTAL_SPUR_TAGS)
        ]
        if (drop_spur_only and spur_set)
        else []
    )

    kept: List[FittingResult] = []
    dropped_empty: List[int] = []
    dropped_spur: List[Dict[str, Any]] = []

    for wf in fit.window_fits:
        wid = int(wf.window_id) if wf.window_id is not None else -1
        peaks = wf.fitted_peaks

        if drop_empty and len(peaks) == 0:
            dropped_empty.append(wid)
            continue

        if drop_spur_only and len(peaks) == 1 and instrumental:
            assert spur_set is not None  # instrumental non-empty => spur_set set
            p = peaks[0]
            if getattr(p, "origin", "auto") != "user":
                pf = float(p.frequency_mhz)
                hit = None
                for sp in instrumental:
                    tol = max(spur_set._spur_half_width_mhz(sp), res_element_mhz)
                    if abs(float(sp.center_mhz) - pf) <= tol:
                        hit = sp
                        break
                if hit is not None:
                    dropped_spur.append(
                        {
                            "window_id": wid,
                            "frequency_mhz": pf,
                            "spur_center_mhz": float(hit.center_mhz),
                            "spur_source": hit.source,
                        }
                    )
                    continue

        kept.append(wf)

    fit.window_fits = kept
    all_peaks: List[FittedPeak] = [p for wf in kept for p in wf.fitted_peaks]
    all_peaks.sort(key=lambda p: p.frequency_mhz)
    fit.fitted_peaks = all_peaks

    fit.diagnostics["window_cleanup"] = {
        "n_empty_dropped": len(dropped_empty),
        "empty_window_ids": dropped_empty,
        "n_spur_only_dropped": len(dropped_spur),
        "spur_only_dropped": dropped_spur,
    }


def _build_explicit_from_kwargs(
    *,
    tau0_us: Optional[float],
    fit_tau: Optional[bool],
    max_decay_factor: Optional[float],
    residual_edge_threshold: Optional[float],
    residual_edge_m: Optional[int],
    max_thaw_rounds: Optional[int],
    max_replan_rounds: Optional[int],
    max_residual_rescue_rounds: Optional[int],
    rescue_snr_threshold: Optional[float],
    rescue_prominence_threshold: Optional[float],
    tau_maj_override_us: Optional[float],
    sigma_tau_override_us: Optional[float],
    per_band_tau: Optional[bool],
    shape: "PeakShape | str | None",
) -> StageFitSettings:
    """Bundle the legacy ``fit_peaks`` kwargs into an explicit-layer
    :class:`StageFitSettings`. Any kwarg that is ``None`` (the unset
    sentinel) leaves its sub-dataclass field at ``None``, so the resolver
    can fall through to the preset / hard-default layers.
    """
    explicit = StageFitSettings()
    if shape is not None:
        explicit.shape = ShapeSpec.coerce(shape)
    explicit.tau.tau0_us = tau0_us
    explicit.tau.fit_tau = fit_tau
    explicit.tau.max_decay_factor = max_decay_factor
    explicit.tau.tau_maj_override_us = tau_maj_override_us
    explicit.tau.sigma_tau_override_us = sigma_tau_override_us
    explicit.tau.per_band_tau = per_band_tau
    explicit.thaw.max_thaw_rounds = max_thaw_rounds
    explicit.thaw.max_replan_rounds = max_replan_rounds
    explicit.thaw.residual_edge_threshold = residual_edge_threshold
    explicit.thaw.residual_edge_m = residual_edge_m
    explicit.rescue.max_rounds = max_residual_rescue_rounds
    explicit.rescue.snr_threshold = rescue_snr_threshold
    explicit.rescue.prominence_threshold = rescue_prominence_threshold
    return explicit


@dataclass
class Stage5FitContext:
    """Shared active-FT context assembled once and consumed by fit and refit.

    Produced by :func:`build_stage5_fit_context` and consumed verbatim by
    :func:`fit_peaks_impl` (the full plan executor) and
    :func:`~ftmwpipeline._internal.stage6_impl.refit_window_impl` (the
    single-window user-directed refit).  All arrays are in the native
    active-FT bin order (same as ``active_ft.complex_spectrum``).
    """

    # Active-FT result (frequencies, complex spectrum, alpha, n_active, etc.)
    active_ft: Any  # ActiveFTResult -- avoids a cyclic import at module level

    # Per-bin complex RMS on the full active-FT grid (spur sweep authority).
    active_rms: "np.ndarray"

    # Per-bin complex RMS restricted to the trim (analysis) band where
    # available; equals ``active_rms`` when no trim is present.
    rms_for_fit: "np.ndarray"

    # Ascending-frequency sort indices for the full active-FT (the spur sweep
    # needs ascending order; 2638 has a descending grid).
    sort_idx: "np.ndarray"

    # Sideband enum resolved from the persisted FID.
    sideband: Sideband

    # Active acquisition duration T (µs).
    acquisition_us: float

    # Probe frequency (MHz) from the persisted FID.
    probe_freq_mhz: float

    # Gated spur set (``None`` when spur masking is disabled or no spurs found).
    spur_set: Optional[Any]  # SpurSet | None

    # Declared clock-lattice (``None`` when no clock declaration was present).
    clock_lattice: Optional[Any]  # ClockLattice | None

    # FID samples and sampling parameters needed by the spur probe and the
    # active-FT recompute inside refit_window.
    fid_samples: "np.ndarray"
    sample_dt_us: float
    start_us: float
    end_us: float
    n_padded: int

    # Trim range (lo, hi) in MHz, or None when the full active-FT is the band.
    trim_range: Optional[Tuple[float, float]]

    # The persisted ComplexFT (the user's canonical Stage 1 FT, used to
    # derive the analysis-band extent for the spur sweep).
    user_ft: Any  # ComplexFT


def build_stage5_fit_context(
    file_path: str,
    resolved: Any,  # StageFitSettings
    persisted_cal: Optional[Any],  # TauCalibrationResult | None
    shape_enum: Any,  # PeakShape
    replay_spur_catalogue: Optional[Mapping[str, Any]] = None,
) -> Stage5FitContext:
    """Assemble the active-FT, noise, and spur-set shared context.

    Extracted from :func:`fit_peaks_impl` so the single-window
    :func:`~ftmwpipeline._internal.stage6_impl.refit_window_impl` can
    rebuild the same context VERBATIM (same active-FT, same noise, same
    spur set) without duplicating the assembly logic.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    resolved :
        Fully-resolved :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
        from the settings chain (explicit > persisted > recommended > defaults).
    persisted_cal :
        Persisted Stage 2b calibration result (``None`` when absent).
    shape_enum :
        Resolved :class:`~ftmwpipeline.fitting.peak_model.PeakShape` for the fit.
    replay_spur_catalogue :
        When given (a persisted ``SpectrumFit.parameters`` mapping), the gated
        spur catalogue is **replayed** from it -- the spur detector (decay /
        chirp probes, frequency-domain sweep, clock lattice) does NOT run.
        The detection is a Stage 5 product; a later stage (the Stage 6
        single-window refit) must reproduce the exact mask the fit used rather
        than re-deriving a possibly-drifted catalogue.  ``None`` (the
        production first-fit path) runs the full detector.

    Returns
    -------
    Stage5FitContext
        Self-contained shared context ready for :func:`execute_plan` or a
        single-window refit.
    """
    from ..fitting.active_ft import compute_active_ft
    from ..fitting.clock_lattice import build_clock_lattice
    from ..fitting.spur_detection import (
        SpurSet,
        build_spur_set,
        make_band_power_probe,
        make_chirp_response_probe,
        make_decay_probe,
        spur_set_from_catalogue,
    )
    from ..io.fid_serialization import load_acquisition_segments_from_hdf5
    from ..preprocessing.noise_estimation import estimate_active_ft_noise
    from .active_ft_support import _persisted_scatter_knobs

    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        n_padded,
        acquisition_us,
        user_ft,
        trim_range,
    ) = _build_active_ft_inputs(file_path)

    active_ft = compute_active_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
    )

    scatter_knobs = _persisted_scatter_knobs(file_path)
    active_rms = np.asarray(
        estimate_active_ft_noise(
            active_ft.freq_mhz,
            active_ft.complex_spectrum,
            **scatter_knobs,
        ).rms_noise,
        dtype=float,
    )
    rms_for_fit = active_rms
    if trim_range is not None:
        freq_arr = np.asarray(active_ft.freq_mhz, dtype=float)
        in_band = (freq_arr >= float(min(trim_range))) & (
            freq_arr <= float(max(trim_range))
        )
        if bool(in_band.any()) and not bool(in_band.all()):
            rms_for_fit = active_rms.copy()
            rms_for_fit[in_band] = np.asarray(
                estimate_active_ft_noise(
                    freq_arr[in_band],
                    np.asarray(active_ft.complex_spectrum)[in_band],
                    **scatter_knobs,
                ).rms_noise,
                dtype=float,
            )

    sort_idx = np.argsort(active_ft.freq_mhz)
    sorted_freq = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])

    # --- Spur gating (optional) ------------------------------------------
    spur_set: Optional[Any] = None
    clock_lattice: Optional[Any] = None
    spur_cfg = resolved.spur
    spur_enabled = True if spur_cfg.enabled is None else bool(spur_cfg.enabled)
    if spur_enabled and replay_spur_catalogue is not None:
        # Replay the persisted Stage 5 gated catalogue verbatim (no detection):
        # the catalogue is a Stage 5 product, so a later-stage refit reproduces
        # the exact residual mask the fit used instead of re-deriving it.
        cat = replay_spur_catalogue
        bin_spacing = (
            float(np.median(np.abs(np.diff(sorted_freq))))
            if sorted_freq.size > 1
            else 0.0
        )
        spur_set = spur_set_from_catalogue(
            centers_mhz=list(cat.get("spur_centers_mhz", []) or []),
            sources=list(cat.get("spur_sources", []) or []),
            lattice=list(cat.get("spur_lattice", []) or []),
            drift=list(cat.get("spur_drift", []) or []),
            per_spur_mask_half_width_bins=list(
                cat.get("spur_mask_half_width_bins_per_spur", []) or []
            ),
            bin_spacing_mhz=bin_spacing,
            default_mask_half_width_bins=int(
                cat.get("spur_mask_half_width_bins", 0) or 0
            ),
        )
        logger.info(
            "Stage 5 spur masking: replayed %d persisted gated spur(s) "
            "(no re-detection)",
            len(spur_set.spurs),
        )
    elif spur_enabled:
        use_catalogue = (
            True
            if spur_cfg.use_stft_catalogue is None
            else bool(spur_cfg.use_stft_catalogue)
        )
        saturated_clusters = (
            persisted_cal.spur_clusters
            if (persisted_cal is not None and use_catalogue)
            else ()
        )
        sorted_sig_c = active_rms[sort_idx] / np.sqrt(2.0)
        spur_band = (
            float(np.min(user_ft.freq_array)),
            float(np.max(user_ft.freq_array)),
        )
        decay_probe = make_decay_probe(
            fid_samples,
            sample_dt_us,
            start_us=start_us,
            end_us=end_us,
            probe_freq_mhz=probe_freq_mhz,
            sideband=sideband,
        )
        chirp_response_probe = None
        with h5py.File(file_path, "r") as h5f:
            if "stage0_fid_data" in h5f:
                acq_segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])
                if acq_segs is not None:
                    excluded_comb: Optional[List[float]] = None
                    if acq_segs.interleave_patterns is not None:
                        excluded_comb = [
                            1.0 / (m * acq_segs.sample_dt) / 1e6
                            for m in acq_segs.interleave_patterns.keys()
                        ]
                        logger.info(
                            "Stage 5 spur masking: excluding %d comb spacing(s) "
                            "from chirp-response probe (interleave cleanup): %s MHz",
                            len(excluded_comb),
                            ", ".join(f"{sp:.3f}" for sp in excluded_comb),
                        )
                    chirp_response_probe = make_chirp_response_probe(
                        acq_segs.pre_record,
                        fid_samples,
                        sample_dt_us,
                        start_us=start_us,
                        end_us=end_us,
                        probe_freq_mhz=probe_freq_mhz,
                        sideband=sideband,
                        excluded_comb_mhz=excluded_comb,
                    )
                    logger.info(
                        "Stage 5 spur masking: chirp-response probe built "
                        "from pre-record (%d samples, %.2f µs)",
                        acq_segs.pre_record.size,
                        acq_segs.pre_record_us,
                    )
        integer_tol_v = _required_float(
            spur_cfg.integer_tol_mhz, "spur.integer_tol_mhz"
        )
        band_power_probe = None
        lattice_kwargs: Dict[str, Any] = {}
        if spur_cfg.clocks:
            drift_window_v = _required_float(
                spur_cfg.drift_window_mhz, "spur.drift_window_mhz"
            )
            clock_lattice = build_clock_lattice(
                spur_cfg.clocks,
                probe_freq_mhz=probe_freq_mhz,
                sideband=sideband,
                band=spur_band,
                tol_mhz=integer_tol_v,
                drift_window_mhz=drift_window_v,
            )
            band_power_probe = make_band_power_probe(
                fid_samples,
                sample_dt_us,
                start_us=start_us,
                end_us=end_us,
                probe_freq_mhz=probe_freq_mhz,
                sideband=sideband,
                half_mhz=drift_window_v,
            )
            lattice_kwargs = {
                "lattice": clock_lattice,
                "band_power_probe": band_power_probe,
                "lattice_decay_ratio": _required_float(
                    spur_cfg.lattice_decay_ratio, "spur.lattice_decay_ratio"
                ),
                "drift_band_ratio": _required_float(
                    spur_cfg.drift_band_ratio, "spur.drift_band_ratio"
                ),
                "drift_min_snr": _required_float(
                    spur_cfg.drift_min_snr, "spur.drift_min_snr"
                ),
            }
        spur_set = build_spur_set(
            sorted_freq,
            np.ascontiguousarray(active_ft.complex_spectrum[sort_idx]),
            sorted_sig_c,
            band=spur_band,
            saturated_clusters=saturated_clusters,
            integer_tol_mhz=integer_tol_v,
            narrowness_ratio=_required_float(
                spur_cfg.narrowness_ratio, "spur.narrowness_ratio"
            ),
            snr_threshold=_required_float(spur_cfg.snr_threshold, "spur.snr_threshold"),
            mask_half_width_bins=_required_int(
                spur_cfg.mask_half_width_bins, "spur.mask_half_width_bins"
            ),
            use_stft_catalogue=use_catalogue,
            decay_probe=decay_probe,
            chirp_response_probe=chirp_response_probe,
            chirp_response_gate_ratio=_required_float(
                spur_cfg.chirp_response_gate_ratio,
                "spur.chirp_response_gate_ratio",
            ),
            chirp_response_protect_ratio=_required_float(
                spur_cfg.chirp_response_protect_ratio,
                "spur.chirp_response_protect_ratio",
            ),
            mask_target_residual_snr=_required_float(
                spur_cfg.mask_target_residual_snr,
                "spur.mask_target_residual_snr",
            ),
            mask_max_half_width_bins=_required_int(
                spur_cfg.mask_max_half_width_bins,
                "spur.mask_max_half_width_bins",
            ),
            **lattice_kwargs,
        )
        if spur_set:
            logger.info(
                "Stage 5 spur masking: %d gated spur(s) "
                "(sources: %s); mask half-width %d bins, catalogue=%s",
                len(spur_set.spurs),
                ", ".join(sorted({s.source for s in spur_set.spurs})),
                spur_set.mask_half_width_bins,
                "on" if (use_catalogue and saturated_clusters) else "off",
            )
        else:
            logger.info("Stage 5 spur masking: enabled, no spurs gated")

    return Stage5FitContext(
        active_ft=active_ft,
        active_rms=active_rms,
        rms_for_fit=rms_for_fit,
        sort_idx=sort_idx,
        sideband=sideband,
        acquisition_us=acquisition_us,
        probe_freq_mhz=probe_freq_mhz,
        spur_set=spur_set,
        clock_lattice=clock_lattice,
        fid_samples=fid_samples,
        sample_dt_us=sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        n_padded=n_padded,
        trim_range=trim_range,
        user_ft=user_ft,
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
    tau_maj_override_us: Optional[float] = None,
    sigma_tau_override_us: Optional[float] = None,
    per_band_tau: Optional[bool] = None,
    shape: "PeakShape | str | None" = None,
    settings: Optional[StageFitSettings] = None,
    preset: Optional[str] = None,
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
        Defaults to the Stage 2b ``tau_maj`` when a calibration is present
        (per-band ``tau_maj`` for band-routed windows), otherwise to ``T/3``.
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
    tau_maj_override_us, sigma_tau_override_us : float, optional
        Atomic-pair manual override for the Stage 2b tau calibration. When
        both are supplied (positive), they replace any persisted Stage 2b
        result for this fit -- useful for A/B-ing a hand-tuned tau anchor
        against the persisted one, or for forcing a calibrated tau on
        fixtures where Stage 2b has not been run. Supplying only one of
        the pair raises ``ValueError``.
    per_band_tau : bool, default False
        Route each window to its band-local ``(tau_maj_us, sigma_tau_us)``
        from the persisted Stage 2b ``band_majorities``. Requires the
        Stage 2b calibration to have been run with
        ``compute_band_majorities=True`` (otherwise raises ``ValueError``).
        Windows whose centre frequency falls outside every band, and
        windows whose band has fewer than ``min_contributors_per_band``
        contributors (so the band's tau collapsed to the band-wide
        fallback), inherit the band-wide ``(tau_maj_us, sigma_tau_us)``
        unchanged. Incompatible with ``tau_maj_override_us`` /
        ``sigma_tau_override_us`` (the explicit-override pair beats any
        persisted calibration; per-band routing only makes sense relative
        to the persisted band majorities).

    Raises
    ------
    ValueError
        If Stage 4 has not been completed, or if exactly one of the
        ``tau_maj_override_us`` / ``sigma_tau_override_us`` pair is set.
    """
    warn_legacy_kwargs(
        func_name="fit_peaks",
        legacy_kwargs={
            "tau0_us": tau0_us,
            "fit_tau": fit_tau,
            "max_decay_factor": max_decay_factor,
            "residual_edge_threshold": residual_edge_threshold,
            "residual_edge_m": residual_edge_m,
            "max_thaw_rounds": max_thaw_rounds,
            "max_replan_rounds": max_replan_rounds,
            "max_residual_rescue_rounds": max_residual_rescue_rounds,
            "rescue_snr_threshold": rescue_snr_threshold,
            "rescue_prominence_threshold": rescue_prominence_threshold,
            "tau_maj_override_us": tau_maj_override_us,
            "sigma_tau_override_us": sigma_tau_override_us,
            "per_band_tau": per_band_tau,
            "shape": shape,
        },
        migration_hint=(
            "use settings=StageFitSettings(...) or preset='name' to drive "
            "Stage 5 from the settings resolver"
        ),
    )

    # --- Resolve parameters via the StageFitSettings chain ------------------
    # Legacy per-knob kwargs are bundled into an explicit StageFitSettings;
    # any caller-supplied ``settings`` instance enters as the preset layer.
    # ``resolve()`` walks explicit > persisted > preset > recommended >
    # hard default; the resolved instance is the single source of truth for
    # every downstream call site below. ``_HARD_DEFAULTS`` mirrors each
    # ``DEFAULT_*`` constant in :mod:`ftmwpipeline.fitting`, so an empty
    # call (no kwargs, no settings, no persisted layer) reproduces the
    # documented per-knob defaults exactly.
    explicit_kwargs = _build_explicit_from_kwargs(
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
    )
    if preset is not None and settings is not None:
        raise ValueError(
            "'preset' and 'settings' are alternative ways to populate "
            "the preset layer of the fit-settings chain; pass exactly "
            "one (or override individual fields via explicit kwargs)"
        )
    preset_layer = settings
    preset_name: Optional[str] = None
    if preset is not None:
        preset_layer = load_preset(preset)
        preset_name = str(preset)
    persisted_settings = load_stage_fit_settings_from_h5(file_path)
    recommended_shape_str = read_stage2b_recommended_shape(file_path)
    recommended_clocks = read_recommended_clock_sources(file_path)
    recommended_settings: Optional[StageFitSettings] = None
    if recommended_shape_str is not None or recommended_clocks is not None:
        recommended_settings = StageFitSettings(
            shape=(
                ShapeSpec.coerce(recommended_shape_str)
                if recommended_shape_str is not None
                else None
            ),
            spur=SpurSubSettings(clocks=recommended_clocks),
        )
    resolved = resolve_stage_fit_settings(
        explicit=explicit_kwargs,
        preset=preset_layer,
        persisted=persisted_settings,
        recommended=recommended_settings,
    )
    # All fields backed by ``_HARD_DEFAULTS`` are guaranteed non-None after
    # resolve(); cast through ``_required_*`` helpers so mypy sees concrete
    # types at the call sites below.
    assert resolved.shape is not None
    shape_enum = resolved.shape.kind
    max_decay_v = _required_float(resolved.tau.max_decay_factor, "tau.max_decay_factor")
    edge_threshold_v = _required_float(
        resolved.thaw.residual_edge_threshold, "thaw.residual_edge_threshold"
    )
    edge_m_v = _required_int(resolved.thaw.residual_edge_m, "thaw.residual_edge_m")
    max_thaw_v = _required_int(resolved.thaw.max_thaw_rounds, "thaw.max_thaw_rounds")
    max_replan_v = _required_int(
        resolved.thaw.max_replan_rounds, "thaw.max_replan_rounds"
    )
    # ``rescue.max_rounds`` resolves to the calibrated default cap; explicit
    # ``0`` disables the rescue (kept as an escape hatch). Any positive
    # value runs the B-loop with that round cap. Clamp to non-negative for
    # parity with the prior ``max(0, int(...))`` behaviour.
    rescue_max_v = max(
        0, _required_int(resolved.rescue.max_rounds, "rescue.max_rounds")
    )
    rescue_snr_v = _required_float(
        resolved.rescue.snr_threshold, "rescue.snr_threshold"
    )
    rescue_prom_v = _required_float(
        resolved.rescue.prominence_threshold, "rescue.prominence_threshold"
    )
    # The override-pair and per_band_tau flag also flow through the
    # resolved instance so a preset can carry them.
    tau_maj_override_v = resolved.tau.tau_maj_override_us
    sigma_tau_override_v = resolved.tau.sigma_tau_override_us
    per_band_tau_v = _required_bool(resolved.tau.per_band_tau, "tau.per_band_tau")
    # Leakage-wing baseline knobs (driven by the settings block, like spur).
    baseline_enabled_v = _required_bool(resolved.baseline.enabled, "baseline.enabled")
    baseline_order_v = _required_int(resolved.baseline.order, "baseline.order")
    baseline_edge_threshold_v = _required_float(
        resolved.baseline.edge_threshold, "baseline.edge_threshold"
    )
    baseline_smooth_threshold_v = _required_float(
        resolved.baseline.smooth_threshold, "baseline.smooth_threshold"
    )
    # Doublet-alternative observation pass: observation-only; attaches records
    # but never modifies fitted peaks.
    doublet_cfg = resolved.doublet_alternative
    doublet_enabled_v = _required_bool(
        doublet_cfg.enabled, "doublet_alternative.enabled"
    )
    doublet_kwargs: Optional[Dict[str, Any]]
    if doublet_enabled_v:
        doublet_kwargs = {
            "k_res": _required_float(doublet_cfg.k_res, "doublet_alternative.k_res"),
            "r_min": _required_float(doublet_cfg.r_min, "doublet_alternative.r_min"),
        }
        logger.info(
            "Stage 5 doublet-alternative pass enabled "
            "(k_res=%.2f, r_min=%.3f; observation-only)",
            doublet_kwargs["k_res"],
            doublet_kwargs["r_min"],
        )
    else:
        doublet_kwargs = None

    # Peak-survival pass: SNR-floor dust removal + degenerate-overfit collapse.
    peak_survival_enabled_v = _required_bool(
        resolved.peak_survival.enabled, "peak_survival.enabled"
    )
    peak_survival_floor_v = _required_float(
        resolved.peak_survival.snr_survival_floor, "peak_survival.snr_survival_floor"
    )
    vif_collapse_threshold_v = _required_float(
        resolved.peak_survival.vif_collapse_threshold,
        "peak_survival.vif_collapse_threshold",
    )
    collapse_max_sep_res_v = _required_float(
        resolved.peak_survival.collapse_max_separation_res,
        "peak_survival.collapse_max_separation_res",
    )
    merge_chi2_veto_v = _required_float(
        resolved.peak_survival.merge_chi2_veto,
        "peak_survival.merge_chi2_veto",
    )

    # --- Validate Stage 4 prerequisite up front ----------------------------
    with h5py.File(file_path, "r") as h5f:
        if "stage4_windows" not in h5f:
            raise ValueError(
                "Stage 4 (window assignment) must be completed before "
                "fitting. Run assign_windows()/'windows run' first."
            )

    plan: WindowPlan = load_windows_impl(file_path)["plan"]
    peaks_loaded = load_peaks_impl(file_path)
    peaks = peaks_loaded["peaks"]
    peak_frequencies_mhz = [float(p.frequency) for p in peaks]

    # --- Stage 2b calibration (optional) ------------------------------------
    # When present, ``tau_maj`` and ``sigma_tau`` drive the per-window tau
    # bounds and the bidirectional Gaussian-prior anchoring penalty. Stage 5
    # tolerates its absence (falls back to the legacy apodization-anchored
    # path) so the rollout is non-breaking. Explicit
    # ``(tau_maj_override_us, sigma_tau_override_us)`` beats the persisted
    # calibration for this fit (atomic pair; supplying only one raises).
    #
    # The Gaussian path consumes the τ_G twin
    # (``/stage2b_tau_G_calibration``); the Lorentzian path stays on the
    # pure-exp Stage 2b (``/stage2b_tau_calibration``). The two
    # calibrations are independent and can coexist on one file; we route
    # to the shape-appropriate one based on the caller's ``shape`` arg.
    persisted_cal: Optional[TauCalibrationResult] = None
    if shape_enum is PeakShape.GAUSSIAN:
        if tau_G_calibration_present(file_path):
            persisted_cal = load_tau_G_calibration_impl(file_path)["tau_G_calibration"]
            if not persisted_cal.preconditions_passed:
                logger.warning(
                    "Stage 2b τ_G calibration pre-conditions did not pass "
                    "on %s; Stage 5 (gaussian) will still consume "
                    "tau_G_maj=%.3f (sigma_tau_G=%.3f). Notes: %s",
                    file_path,
                    float(persisted_cal.tau_maj_us),
                    float(persisted_cal.sigma_tau_us),
                    "; ".join(persisted_cal.preconditions_notes),
                )
        else:
            logger.warning(
                "Stage 5 shape='gaussian' but no τ_G calibration is "
                "present on %s; fitting without a τ_G prior. Run "
                "calibrate_tau_G(...) for an anchored fit.",
                file_path,
            )
    else:
        if tau_calibration_present(file_path):
            persisted_cal = load_tau_calibration_impl(file_path)["tau_calibration"]
            if not persisted_cal.preconditions_passed:
                logger.warning(
                    "Stage 2b calibration pre-conditions did not pass on %s; "
                    "Stage 5 will still consume tau_maj=%.3f (sigma_tau=%.3f). "
                    "Notes: %s",
                    file_path,
                    float(persisted_cal.tau_maj_us),
                    float(persisted_cal.sigma_tau_us),
                    "; ".join(persisted_cal.preconditions_notes),
                )
    tau_maj_us, sigma_tau_us, tau_source = _resolve_tau_calibration_for_fit(
        persisted_cal,
        tau_maj_override_v,
        sigma_tau_override_v,
    )
    if tau_source == "override":
        logger.info(
            "Stage 5 using tau override: tau_maj=%.3f us, sigma_tau=%.3f us "
            "(beats persisted=%s)",
            tau_maj_us,
            sigma_tau_us,
            "yes" if persisted_cal is not None else "no",
        )
    elif tau_source == "persisted":
        logger.info(
            "Stage 5 consuming Stage 2b calibration: tau_maj=%.3f us, "
            "sigma_tau=%.3f us",
            tau_maj_us,
            sigma_tau_us,
        )

    # --- Build the active-FT, noise, and spur context ----------------------
    # Extracted into a reusable helper so the single-window Stage 6 refit can
    # rebuild the same context verbatim (same active-FT, same noise, same spur
    # set) without duplicating any assembly logic.  ``persisted_cal`` and
    # ``resolved`` are resolved above before this call and forwarded verbatim.
    fit_ctx = build_stage5_fit_context(file_path, resolved, persisted_cal, shape_enum)
    active_ft = fit_ctx.active_ft
    active_rms = fit_ctx.active_rms
    rms_for_fit = fit_ctx.rms_for_fit
    sideband = fit_ctx.sideband
    acquisition_us = fit_ctx.acquisition_us
    spur_set = fit_ctx.spur_set
    clock_lattice = fit_ctx.clock_lattice
    trim_range = fit_ctx.trim_range
    # Derive the spur-enabled flag from the resolved settings so the parameters
    # dict below can record it without the helper needing to return it.
    spur_enabled: bool = (
        True if resolved.spur.enabled is None else bool(resolved.spur.enabled)
    )

    # --- tau0 default --------------------------------------------------------
    # The global seed is the band-wide Stage 2b ``tau_maj`` when a calibration
    # is present, else ``T_active / 3``. Per-band routing (below) overrides the
    # seed per window with that window's band-local ``tau_maj``.
    if resolved.tau.tau0_us is None:
        if tau_maj_us is not None and tau_maj_us > 0.0:
            tau0_us_v = float(tau_maj_us)
        else:
            tau0_us_v = acquisition_us / 3.0
    else:
        tau0_us_v = float(resolved.tau.tau0_us)
    if tau0_us_v <= 0:
        raise ValueError(
            f"tau0_us must be positive (got {tau0_us_v}); the active "
            f"acquisition is {acquisition_us} us"
        )
    fit_tau_v = True if resolved.tau.fit_tau is None else bool(resolved.tau.fit_tau)

    # --- Structural-replan context (skipped when caller asks for 0 rounds) -
    replan_ctx: Optional[ReplanContext]
    if max_replan_v <= 0:
        replan_ctx = None
    else:
        # Replan re-runs Stage 4's window planner, which operates on the
        # active FT + authority noise; hand it the same trimmed active grid so
        # the re-plan is consistent with the original plan.
        replan_ft, replan_rms = build_active_grid_with_noise(file_path, trim_range)
        replan_ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=replan_ft.freq_array,
            active_complex_spectrum=replan_ft.complex_spectrum,
            active_rms_noise=replan_rms,
            max_replan_rounds=max_replan_v,
        )

    # --- Drive the executor -------------------------------------------------
    rescue_kwargs: Optional[Dict[str, Any]]
    if rescue_max_v > 0:
        rescue_kwargs = {
            "snr_threshold": rescue_snr_v,
            "prominence_threshold": rescue_prom_v,
        }
    else:
        rescue_kwargs = None

    # --- Per-band tau routing (Item 4) --------------------------------------
    # When per_band_tau=True AND the persisted Stage 2b carries
    # ``band_majorities``, build window_tau_overrides[window_id] =
    # (tau_maj_band, sigma_tau_band) by mapping each window's centre
    # frequency to the band whose [freq_lo, freq_hi) contains it. The
    # band-wide ``(tau_maj_us, sigma_tau_us)`` in conservative_kwargs
    # remains the fallback for windows that don't match any band (e.g.
    # window centre outside the calibration trim range).
    window_tau_overrides: Dict[int, tuple[float, float]] = {}
    per_band_used = False
    if per_band_tau_v:
        # Explicit override pair is more specific than per-band routing -- if
        # the caller supplied (tau_maj_override_us, sigma_tau_override_us)
        # they want exactly that anchor across every window. Silently skip
        # per-band routing in that case (the explicit override path drives
        # the fit instead). When the user explicitly sets ``per_band_tau``
        # AND the override pair, the override wins.
        if tau_source == "override":
            logger.info(
                "Stage 5 per-band tau routing requested but explicit "
                "tau_maj_override / sigma_tau_override is set; the explicit "
                "override drives every window and per-band routing is "
                "skipped."
            )
        elif persisted_cal is None or not persisted_cal.band_majorities:
            # Production default is per_band_tau=True so degrade gracefully
            # when band_majorities aren't available: fall through to the
            # band-wide prior (or no prior at all if Stage 2b also missing).
            # An explicit per_band_tau=True caller still gets the soft
            # fallback -- the original strict-raise behaviour penalised
            # workflows that don't run Stage 2b without giving the caller
            # anything actionable.
            logger.info(
                "Stage 5 per-band tau routing requested but no Stage 2b "
                "band_majorities are persisted; falling back to band-wide "
                "tau_maj=%s, sigma_tau=%s (re-run calibrate_tau(..., "
                "compute_band_majorities=True) to enable per-band routing).",
                tau_maj_us if tau_maj_us is not None else "None",
                sigma_tau_us if sigma_tau_us is not None else "None",
            )
        else:
            for win in plan.windows:
                centre_mhz = 0.5 * (win.freq_range[0] + win.freq_range[1])
                band = band_majority_for_frequency(
                    persisted_cal.band_majorities,
                    centre_mhz,
                )
                if band is None:
                    continue
                window_tau_overrides[int(win.window_id)] = (
                    float(band.tau_maj_us),
                    float(band.sigma_tau_us),
                )
            per_band_used = True
            logger.info(
                "Stage 5 per-band tau routing on: %d / %d windows mapped "
                "to a band (others use band-wide tau_maj=%.3f, sigma=%.3f)",
                len(window_tau_overrides),
                len(plan.windows),
                tau_maj_us if tau_maj_us is not None else float("nan"),
                sigma_tau_us if sigma_tau_us is not None else float("nan"),
            )

    n_eff_kind_v = _required_str(
        resolved.conservative.n_eff_kind, "conservative.n_eff_kind"
    )
    conservative_kwargs: Dict[str, Any] = {
        "max_decay_factor": max_decay_v,
        # tau anchoring: when Stage 2b is present, ``tau_maj_us`` and
        # ``sigma_tau_us`` drive the bidirectional Gaussian-prior penalty and
        # the calibrated bounds (``tau_maj +- N*sigma_tau`` intersected with
        # the factor-k cap). Absent Stage 2b, tau is bounded by the factor-k
        # cap alone (there is no apodization anchor -- the canonical FT is
        # unapodized).
        "tau_apodization_us": None,
        "tau_maj_us": tau_maj_us,
        "sigma_tau_us": sigma_tau_us,
        # τ-prior knobs (Stage 2b consumer).
        "tau_penalty_lambda": _required_float(
            resolved.tau.tau_penalty_lambda, "tau.tau_penalty_lambda"
        ),
        "tau_penalty_n_sigma": _required_float(
            resolved.tau.tau_penalty_n_sigma, "tau.tau_penalty_n_sigma"
        ),
        # Add-one-peak loop gates (F-test diagnostic + AICc gate inputs).
        "significance": _required_float(
            resolved.conservative.significance, "conservative.significance"
        ),
        "max_peaks": _required_int(
            resolved.conservative.max_peaks, "conservative.max_peaks"
        ),
        "patience": _required_int(
            resolved.conservative.patience, "conservative.patience"
        ),
        "min_separation_factor": _required_float(
            resolved.conservative.min_separation_factor,
            "conservative.min_separation_factor",
        ),
        "min_pair_separation_factor": _required_float(
            resolved.conservative.min_pair_separation_factor,
            "conservative.min_pair_separation_factor",
        ),
        "min_pair_separation_resolution_factor": _required_float(
            resolved.conservative.min_pair_separation_resolution_factor,
            "conservative.min_pair_separation_resolution_factor",
        ),
        "weak_window_snr_threshold": _required_float(
            resolved.conservative.weak_window_snr_threshold,
            "conservative.weak_window_snr_threshold",
        ),
        "fit_tau_min_snr": _required_float(
            resolved.tau.fit_tau_min_snr, "tau.fit_tau_min_snr"
        ),
        "n_eff_kind": n_eff_kind_v,
        # Blend-aware seeder thresholds.
        "seeder_rchi2_threshold": _required_float(
            resolved.seeder.seeder_rchi2, "seeder.seeder_rchi2"
        ),
        "seeder_straddle_factor": _required_float(
            resolved.seeder.seeder_straddle_factor, "seeder.seeder_straddle_factor"
        ),
        "seeder_max_k": _required_int(
            resolved.seeder.seeder_max_k, "seeder.seeder_max_k"
        ),
        # Phase / amplitude soft penalties.
        "phase_penalty_lambda": _required_float(
            resolved.penalties.phase_penalty_lambda,
            "penalties.phase_penalty_lambda",
        ),
        "phase_penalty_cutoff_fwhm": _required_float(
            resolved.penalties.phase_penalty_cutoff_fwhm,
            "penalties.phase_penalty_cutoff_fwhm",
        ),
        "amp_penalty_lambda": _required_float(
            resolved.penalties.amp_penalty_lambda, "penalties.amp_penalty_lambda"
        ),
        "amp_max_headroom": _required_float(
            resolved.penalties.amp_max_headroom, "penalties.amp_max_headroom"
        ),
    }
    if rescue_kwargs is not None:
        cleanup_sig = _required_float(
            resolved.rescue.cleanup_significance, "rescue.cleanup_significance"
        )
        # The rescue consolidator and its inner knockout-test share the same
        # F-test gate today; expose one dataclass field that drives both.
        rescue_kwargs.update(
            {
                "rescue_significance": cleanup_sig,
                "knockout_significance": cleanup_sig,
                "merge_separation_factor": _required_float(
                    resolved.rescue.merge_separation_factor,
                    "rescue.merge_separation_factor",
                ),
                "structural_merge_factor": _required_float(
                    resolved.rescue.structural_merge_factor,
                    "rescue.structural_merge_factor",
                ),
                "overfit_amp_ratio_band": _required_float(
                    resolved.rescue.overfit_amp_ratio_band,
                    "rescue.overfit_amp_ratio_band",
                ),
                "overfit_amp_ratio_threshold": _required_float(
                    resolved.rescue.overfit_amp_ratio_threshold,
                    "rescue.overfit_amp_ratio_threshold",
                ),
                "n_eff_kind": n_eff_kind_v,
            }
        )

    plan_outcome = execute_plan(
        plan,
        active_ft,
        rms_for_fit,
        peak_frequencies_mhz,
        sideband=sideband,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us_v,
        fit_tau=fit_tau_v,
        shape=shape_enum,
        residual_edge_threshold=edge_threshold_v,
        residual_edge_m=edge_m_v,
        max_thaw_rounds=max_thaw_v,
        conservative_kwargs=conservative_kwargs,
        replan_context=replan_ctx,
        max_residual_rescue_rounds=rescue_max_v,
        rescue_kwargs=rescue_kwargs,
        window_tau_overrides=window_tau_overrides if per_band_used else None,
        spur_set=spur_set,
        baseline_enabled=baseline_enabled_v,
        baseline_order=baseline_order_v,
        baseline_edge_threshold=baseline_edge_threshold_v,
        baseline_smooth_threshold=baseline_smooth_threshold_v,
        doublet_kwargs=doublet_kwargs,
    )

    parameters = {
        "shape": shape_enum.value,
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
        "tau_maj_us": tau_maj_us,
        "sigma_tau_us": sigma_tau_us,
        "tau_calibration_source": tau_source,
        "per_band_tau": per_band_used,
        "n_windows_band_routed": len(window_tau_overrides) if per_band_used else 0,
        # Spur-masking audit: the gated spur catalogue this fit consumed.
        "spur_masking_enabled": spur_enabled,
        "n_spurs_gated": len(spur_set.spurs) if spur_set else 0,
        "spur_centers_mhz": (
            [round(s.center_mhz, 4) for s in spur_set.spurs] if spur_set else []
        ),
        "spur_sources": ([s.source for s in spur_set.spurs] if spur_set else []),
        # Clock-lattice provenance parallel to ``spur_centers_mhz``: the
        # matched lattice identity (or null) and the drifting-family flag.
        "spur_lattice": ([s.lattice for s in spur_set.spurs] if spur_set else []),
        "spur_drift": ([bool(s.drift) for s in spur_set.spurs] if spur_set else []),
        "spur_mask_half_width_bins": (
            int(spur_set.mask_half_width_bins) if spur_set else 0
        ),
        # Per-spur SNR-scaled mask half-width override (parallel to
        # ``spur_centers_mhz``; null where the uniform default applies). A
        # later stage replays the gated catalogue rather than re-deriving it
        # (the detection is a Stage 5 product), so the per-spur widths must
        # round-trip for the residual mask to reconstruct exactly.
        "spur_mask_half_width_bins_per_spur": (
            [
                (
                    int(s.mask_half_width_bins)
                    if s.mask_half_width_bins is not None
                    else None
                )
                for s in spur_set.spurs
            ]
            if spur_set
            else []
        ),
        # Leakage-wing baseline audit: the settings this fit consumed plus
        # how many windows the evidence trigger actually fired on.
        "baseline_enabled": baseline_enabled_v,
        "baseline_order": baseline_order_v,
        "baseline_edge_threshold": baseline_edge_threshold_v,
        "baseline_smooth_threshold": baseline_smooth_threshold_v,
        "n_baseline_windows": sum(
            1
            for o in plan_outcome.window_outcomes.values()
            if getattr(o, "baseline_applied", False)
        ),
    }
    if rescue_max_v > 0:
        parameters.update(
            {
                "rescue_snr_threshold": rescue_snr_v,
                "rescue_prominence_threshold": rescue_prom_v,
            }
        )
    # Persist the gated spur catalogue with the fit: the masked bins are
    # invisible in the per-window residuals, so visualizations need the
    # spur list to label what the fit deliberately did not model.
    diagnostics: Dict[str, Any] = {}
    if spur_set is not None and spur_set:
        diagnostics["gated_spurs"] = [
            {
                "center_mhz": float(s.center_mhz),
                "source": s.source,
                "lattice": s.lattice,
                "drift": bool(s.drift),
                "mask_half_width_bins": (
                    int(s.mask_half_width_bins)
                    if s.mask_half_width_bins is not None
                    else None
                ),
            }
            for s in spur_set.spurs
        ]
    spectrum_fit: SpectrumFit = plan_fit_outcome_to_spectrum_fit(
        plan_outcome,
        plan,
        sideband=sideband,
        peak_frequencies_mhz=peak_frequencies_mhz,
        acquisition_us=acquisition_us,
        parameters=parameters,
        diagnostics=diagnostics,
    )
    # Annotation pass: stamp ``clock_lattice`` on every fitted peak whose
    # molecular frequency lands on the declared instrument lattice.  Pure
    # read -- no effect on the fit statistics.  No-op when no declaration
    # was present (``clock_lattice`` is ``None``).
    annotate_lattice_matches(spectrum_fit, clock_lattice)

    # Peak-survival prune: drop dust below the SNR floor, re-fitting any
    # partially-pruned window so the surviving peaks' parameters / covariance /
    # χ²ᵣ are re-estimated free of the removed dust (not a stale slice). The
    # refit routes through the bare in-memory window-refit core, reusing the
    # live fit context / plan / resolved settings -- no file reload, no spur
    # re-derivation.
    if peak_survival_enabled_v:
        from .stage6_impl import refit_window_core

        survival_window_map = {w.window_id: w for w in plan.windows}

        def _survival_refit(
            wf: FittingResult, dust_freqs: List[float]
        ) -> FittingResult:
            return refit_window_core(
                fit_ctx,
                survival_window_map[cast(int, wf.window_id)],
                wf,
                resolved=resolved,
                shape_enum=shape_enum,
                tau_maj_us=tau_maj_us,
                sigma_tau_us=sigma_tau_us,
                peak_frequencies_mhz=peak_frequencies_mhz,
                remove=tuple(dust_freqs),
            )

        n_before = spectrum_fit.n_fitted_peaks
        apply_snr_survival_prune(
            spectrum_fit, peak_survival_floor_v, refit_window=_survival_refit
        )
        n_pruned = n_before - spectrum_fit.n_fitted_peaks
        if n_pruned:
            logger.info(
                "Stage 5 peak-survival prune: removed %d sub-floor peaks "
                "(snr < %.2f), %d windows emptied",
                n_pruned,
                peak_survival_floor_v,
                len(
                    spectrum_fit.diagnostics.get("peak_survival", {}).get(
                        "dropped_window_ids", []
                    )
                ),
            )

        # Degenerate-overfit collapse: merge near-degenerate sub-resolution
        # pairs (high amplitude VIF within < collapse_max_separation_res
        # resolution elements). Runs after the prune (dust gone), refits the
        # touched window through the same core with the merged line stamped
        # origin="auto". The resolution element is 1 / T_active (MHz).
        def _collapse_refit(
            wf: FittingResult,
            remove_freqs: List[float],
            add_freqs: List[float],
            add_seeds: List[Any],
        ) -> FittingResult:
            return refit_window_core(
                fit_ctx,
                survival_window_map[cast(int, wf.window_id)],
                wf,
                resolved=resolved,
                shape_enum=shape_enum,
                tau_maj_us=tau_maj_us,
                sigma_tau_us=sigma_tau_us,
                peak_frequencies_mhz=peak_frequencies_mhz,
                remove=tuple(remove_freqs),
                add=tuple(add_freqs),
                add_seeds=add_seeds,
                add_origin="auto",
            )

        res_element_mhz = 1.0 / acquisition_us if acquisition_us > 0 else 0.0
        apply_vif_collapse(
            spectrum_fit,
            vif_threshold=vif_collapse_threshold_v,
            max_sep_res=collapse_max_sep_res_v,
            res_element_mhz=res_element_mhz,
            sideband=sideband,
            refit_collapse=_collapse_refit,
            max_post_chi2r=merge_chi2_veto_v,
        )
        n_collapsed = len(
            spectrum_fit.diagnostics.get("vif_collapse", {}).get("collapses", [])
        )
        if n_collapsed:
            logger.info(
                "Stage 5 VIF collapse: merged %d degenerate sub-resolution "
                "pair(s) (VIF > %.0f within %.2f resolution elements)",
                n_collapsed,
                vif_collapse_threshold_v,
                collapse_max_sep_res_v,
            )

        # Final window cleanup: drop empty (K=0) windows and single-line windows
        # whose sole peak sits on a confidently-instrumental gated spur.
        drop_empty_v = _required_bool(
            resolved.peak_survival.drop_empty_windows,
            "peak_survival.drop_empty_windows",
        )
        drop_spur_only_v = _required_bool(
            resolved.peak_survival.drop_spur_only_windows,
            "peak_survival.drop_spur_only_windows",
        )
        if drop_empty_v or drop_spur_only_v:
            apply_window_cleanup(
                spectrum_fit,
                spur_set=fit_ctx.spur_set,
                res_element_mhz=res_element_mhz,
                drop_empty=drop_empty_v,
                drop_spur_only=drop_spur_only_v,
            )
            wc = spectrum_fit.diagnostics.get("window_cleanup", {})
            if wc.get("n_empty_dropped") or wc.get("n_spur_only_dropped"):
                logger.info(
                    "Stage 5 window cleanup: dropped %d empty + %d spur-only "
                    "window(s)",
                    wc.get("n_empty_dropped", 0),
                    wc.get("n_spur_only_dropped", 0),
                )

    save_spectrum_fit_impl(file_path, spectrum_fit)
    # The automatic fit is the curation baseline for 'review undo'; a fresh fit
    # supersedes any snapshot a prior edit session took, so drop it -- the next
    # user edit re-snapshots this fit.
    from .stage6_impl import clear_stage5_baseline

    clear_stage5_baseline(file_path)
    # Stamp the resolved settings as the canonical record for this fit so
    # a follow-up call with no explicit args inherits exactly the same
    # knobs (the persisted layer of the resolution chain).
    save_stage_fit_settings_to_h5(file_path, resolved, preset_name=preset_name)
    _update_stage_completion(file_path, "stage5_fitting")
    # A Stage 5 re-fit invalidates nothing (Stage 5 is the terminal
    # stage); this call is a no-op and a guard for future stages.
    invalidate_downstream_stages(file_path, "stage5_fitting")

    n_thaw_accepted = sum(1 for e in spectrum_fit.thaw_history if e.accepted)
    n_replan_accepted = sum(1 for e in spectrum_fit.replan_history if e.accepted)
    rescue_events_live = list(spectrum_fit.rescue_history)
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
    # The line-shape choice (lorentzian / gaussian) lives in
    # ``fit.parameters['shape']`` from the fit driver; mirror it onto the
    # group attrs so consumers can branch on shape without having to load
    # the full SpectrumFit struct first.
    shape_attr = str(fit.parameters.get("shape", PeakShape.LORENTZIAN.value))
    with h5py.File(file_path, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(fit, grp)
        grp.attrs["shape"] = shape_attr
    logger.info(
        "Saved Stage 5 fit (%d windows, %d peaks, shape=%s) to %s",
        fit.n_windows,
        fit.n_fitted_peaks,
        shape_attr,
        file_path,
    )


def load_fit_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted Stage 5 fit (validates structure loudly)."""
    with h5py.File(file_path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found. Run fit_peaks()/'fit run' first.")
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
    """Overlay the persisted Stage 5 fit on the active FT it was fit on.

    Re-evaluates the fitted model on the canonical active FT (the grid the
    fit lives on), so the overlay and the model share one amplitude
    convention -- no rescale. The full-record persisted spectrum is not a
    display domain here. Requires Stage 5 completed.
    """
    loaded = load_fit_impl(file_path)
    fit: SpectrumFit = loaded["fit"]
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    trim_range = stage1.get("trim_range")
    active_ft, active_rms = build_active_grid_with_noise(file_path, trim_range)
    fid = load_fid_from_pipeline_impl(file_path)
    sideband = _resolve_sideband(fid.sideband)
    base_pp = user_ft.metadata["processing_params"]
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # The fit and the active FT share the ``dt_us * rfft(active)`` amplitude
    # convention, so the model overlay needs no rescale.
    model_amplitude_scale = 1.0

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
        frequencies=active_ft.freq_array,
        complex_spectrum=active_ft.complex_spectrum,
        rms_noise=active_rms,
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


# ===========================================================================
# Consolidated per-window detail ('fit show') -- selection, render, report
# ===========================================================================

UNITS_LABEL_BY_POWER = {0: "V", 3: "mV", 6: "µV", 9: "nV", 12: "pV"}

# Exactly-2x zero-fill for the magnitude display panels (the information limit
# for a magnitude spectrum; see visualization.fit_detail).
_DETAIL_PAD_FACTOR = 2


@dataclass
class _DetailBundle:
    """Per-file inputs the detail renderer needs, resolved once and reused.

    Resolving the active grid + noise + display FT is the expensive part; a
    batch of windows from one file shares a single bundle.
    """

    fit: SpectrumFit
    frequencies: np.ndarray  # native active grid, ascending molecular freq
    complex_spectrum: np.ndarray
    rms_noise: np.ndarray  # per-bin sigma_x aligned to ``frequencies``
    freq_padded: Optional[np.ndarray]  # 2x display grid, ascending
    spec_padded: Optional[np.ndarray]
    sideband: Sideband
    acquisition_us: float
    amplitude_scale: float
    units_label: str
    trim_mhz: Optional[Tuple[float, float]]
    file_stem: str


def _load_display_style(
    file_path: str,
) -> Tuple[float, str, Optional[Tuple[float, float]]]:
    """Display transforms (amplitude scale, units label, overview trim) from
    the persisted canonical FTSettings. Falls back to (1.0, "", None)."""
    from .stage1_impl import _read_settings_layer

    settings = _read_settings_layer(file_path, "/processing_parameters/ft_processing")
    if settings is None:
        return 1.0, "", None
    units_power = settings.units_power
    if units_power is None:
        scale, label = 1.0, ""
    else:
        scale = 10.0 ** int(units_power)
        label = UNITS_LABEL_BY_POWER.get(int(units_power), f"·10^{units_power} V")
    return scale, label, settings.trim


def _padded_active_display_ft(
    fid_samples: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Sideband,
    pad_factor: int = _DETAIL_PAD_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Display-only active FT zero-filled by ``pad_factor`` for the magnitude
    panels. Mirrors the canonical (unapodized) active-region extraction and mean
    removal so the padded curve passes through the native spectrum at the
    measured bins; the extra bins are the single-zero-fill magnitude
    interpolation. Returns ``(freq_mhz, complex_spectrum)`` sorted by ascending
    molecular frequency. Never feeds fitting / noise / chi-squared."""
    from ..fitting.peak_model import sideband_sign

    fid = np.asarray(fid_samples, dtype=float)
    start_idx = max(int(np.floor(start_us / sample_dt_us)), 0)
    end_idx = min(int(np.ceil(end_us / sample_dt_us)), fid.size)
    active = fid[start_idx:end_idx].astype(float, copy=True)
    n_active = active.size
    active -= active.mean()  # match canonical rdc=True
    n_pad = int(pad_factor) * n_active
    padded = np.zeros(n_pad, dtype=float)
    padded[:n_active] = active
    spectrum = sample_dt_us * np.fft.rfft(padded)
    f_bb = np.fft.rfftfreq(n_pad, d=sample_dt_us)
    freq = probe_freq_mhz + sideband_sign(sideband) * f_bb
    order = np.argsort(freq)
    return (
        np.ascontiguousarray(freq[order]),
        np.ascontiguousarray(spectrum[order]),
    )


def _resolve_detail_bundle(file_path: str) -> _DetailBundle:
    """Resolve the shared per-file detail inputs (see :class:`_DetailBundle`)."""
    fit = load_fit_impl(file_path)["fit"]
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        _n_padded,
        acquisition_us,
        _user_ft,
        trim_range,
    ) = _build_active_ft_inputs(file_path)

    active_ft, active_rms = build_active_grid_with_noise(file_path, trim_range)
    order = np.argsort(active_ft.freq_array)
    freqs_sorted = np.ascontiguousarray(np.asarray(active_ft.freq_array)[order])
    spec_sorted = np.ascontiguousarray(np.asarray(active_ft.complex_spectrum)[order])
    rms_sorted = np.ascontiguousarray(np.asarray(active_rms, dtype=float)[order])

    # The canonical active grid (build_active_grid_with_noise) is unapodized,
    # so the display FT is too.
    freq_padded, spec_padded = _padded_active_display_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
    )

    amp_scale, units_label, trim_mhz = _load_display_style(file_path)
    return _DetailBundle(
        fit=fit,
        frequencies=freqs_sorted,
        complex_spectrum=spec_sorted,
        rms_noise=rms_sorted,
        freq_padded=freq_padded,
        spec_padded=spec_padded,
        sideband=sideband,
        acquisition_us=acquisition_us,
        amplitude_scale=amp_scale,
        units_label=units_label,
        trim_mhz=trim_mhz,
        file_stem=Path(file_path).stem,
    )


def _window_for_freq(fit: SpectrumFit, freq_mhz: float) -> Optional[int]:
    """Window id whose fit range contains ``freq_mhz`` (windows are disjoint)."""
    for wf in fit.window_fits:
        if wf.window is None:
            continue
        lo, hi = wf.window.freq_range
        if min(lo, hi) <= freq_mhz <= max(lo, hi):
            return int(cast(int, wf.window_id))
    return None


def _windows_by_peak_snr(fit: SpectrumFit) -> list[int]:
    """Window ids sorted by descending brightest-in-window peak SNR."""

    def _max_snr(wf: Any) -> float:
        snrs = [p.snr for p in wf.fitted_peaks if p.snr is not None]
        return max(snrs) if snrs else float("-inf")

    ranked = sorted(
        (wf for wf in fit.window_fits if wf.window is not None),
        key=_max_snr,
        reverse=True,
    )
    return [int(cast(int, wf.window_id)) for wf in ranked]


def select_window_ids(
    fit: SpectrumFit,
    *,
    window_ids: Optional[list[int]] = None,
    freqs: Optional[list[float]] = None,
    random_n: Optional[int] = None,
    random_seed: Optional[int] = None,
    top_snr: Optional[int] = None,
    all_windows: bool = False,
) -> list[int]:
    """Resolve the selectors to a sorted, de-duplicated list of window ids.

    The selectors compose as a union: explicit ids, frequency lookups, the
    top-SNR windows, and a random sample are all added to one set. ``all_windows``
    short-circuits to every window with attached context. ``random_seed`` only
    matters when ``random_n`` is set. Raises ``ValueError`` for an unknown id or
    a frequency in no window.
    """
    available = [
        int(cast(int, wf.window_id)) for wf in fit.window_fits if wf.window is not None
    ]
    if not available:
        raise ValueError("the fit has no windows with attached context to show")
    if all_windows:
        return sorted(available)
    available_set = set(available)
    selected: set[int] = set()
    for wid in window_ids or []:
        if int(wid) not in available_set:
            raise ValueError(
                f"window id {wid} not in fit "
                f"(available {min(available)}..{max(available)})"
            )
        selected.add(int(wid))
    for fq in freqs or []:
        fwid = _window_for_freq(fit, float(fq))
        if fwid is None:
            raise ValueError(f"frequency {fq} MHz falls in no fit window")
        selected.add(fwid)
    if top_snr:
        for wid in _windows_by_peak_snr(fit)[: int(top_snr)]:
            selected.add(wid)
    if random_n:
        rng = np.random.default_rng(random_seed)
        pool = sorted(available)
        k = min(int(random_n), len(pool))
        for wid in rng.choice(pool, size=k, replace=False):
            selected.add(int(wid))
    return sorted(selected)


def _detail_title(bundle: _DetailBundle, window_id: int) -> str:
    wf = bundle.fit.window_fit(window_id)
    lo, hi = wf.window.freq_range  # type: ignore[union-attr]
    tau = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    return (
        f"{bundle.file_stem}  window {window_id}  "
        f"[{min(lo, hi):.2f}, {max(lo, hi):.2f}] MHz  "
        f"K={len(wf.fitted_peaks)}  chi2_r={float(wf.reduced_chi2):.2f}  "
        f"tau={tau:.3g} us  shape={getattr(wf, 'shape', 'lorentzian')}"
    )


def render_fit_detail_impl(
    file_path: str,
    window_id: int,
    *,
    bundle: Optional[_DetailBundle] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Any:
    """Render the consolidated per-window detail figure for one window."""
    from ..visualization.fit_detail import DEFAULT_FIGSIZE, plot_consolidated_detail

    bundle = bundle if bundle is not None else _resolve_detail_bundle(file_path)
    wf = bundle.fit.window_fit(window_id)
    return plot_consolidated_detail(
        wf,
        frequencies=bundle.frequencies,
        complex_spectrum=bundle.complex_spectrum,
        rms_noise=bundle.rms_noise,
        sideband=bundle.sideband,
        acquisition_us=bundle.acquisition_us,
        title=title if title is not None else _detail_title(bundle, window_id),
        amplitude_scale=bundle.amplitude_scale,
        units_label=bundle.units_label,
        trim_mhz=bundle.trim_mhz,
        freq_padded=bundle.freq_padded,
        spec_padded=bundle.spec_padded,
        figsize=figsize if figsize is not None else DEFAULT_FIGSIZE,
        spurs=(bundle.fit.diagnostics or {}).get("gated_spurs"),
    )


def render_fit_panels_impl(
    file_path: str,
    window_id: int,
    *,
    bundle: Optional[_DetailBundle] = None,
) -> Dict[str, Any]:
    """Render one window's detail as separate, standalone panel figures.

    Returns a dict keyed ``"overview"`` / ``"re"`` / ``"im"`` / ``"mag"`` /
    ``"hist"`` (one :class:`~matplotlib.figure.Figure` per panel) for the HTML
    report's flexbox -- the modular counterpart of
    :func:`render_fit_detail_impl`, sharing its painters. The caller owns
    closing the figures.
    """
    from ..visualization.fit_detail import plot_window_panels

    bundle = bundle if bundle is not None else _resolve_detail_bundle(file_path)
    wf = bundle.fit.window_fit(window_id)
    return plot_window_panels(
        wf,
        frequencies=bundle.frequencies,
        complex_spectrum=bundle.complex_spectrum,
        rms_noise=bundle.rms_noise,
        sideband=bundle.sideband,
        acquisition_us=bundle.acquisition_us,
        amplitude_scale=bundle.amplitude_scale,
        units_label=bundle.units_label,
        trim_mhz=bundle.trim_mhz,
        freq_padded=bundle.freq_padded,
        spec_padded=bundle.spec_padded,
        spurs=(bundle.fit.diagnostics or {}).get("gated_spurs"),
    )


def fit_window_report_text(
    file_path: str,
    window_id: int,
    *,
    bundle: Optional[_DetailBundle] = None,
    show_audit: bool = False,
) -> str:
    """Plain-text fit log for one window: header, fitted-peak table, and (with
    ``show_audit``) the add-one-peak audit trail. Read-only."""
    from ..visualization.fit_detail import (
        _format_spectroscopic,
        frequency_sorted_labels,
    )

    bundle = bundle if bundle is not None else _resolve_detail_bundle(file_path)
    wf = bundle.fit.window_fit(window_id)
    lo, hi = wf.window.freq_range  # type: ignore[union-attr]
    tau = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    units = bundle.units_label
    amp = bundle.amplitude_scale
    amp_hdr = f"amplitude ({units})" if units else "amplitude"

    # Only emit the lattice column when at least one peak in this window is
    # annotated -- the column is omitted entirely on files with no declaration.
    show_lattice = any(
        getattr(pk, "clock_lattice", None) is not None for pk in wf.fitted_peaks
    )
    lattice_col_w = 14
    if show_lattice:
        header = (
            f"  {'pk':>2}  {'frequency (MHz)':>18}  {amp_hdr:>16}  "
            f"{'phase (rad)':>12}  {'SNR':>6}  conf  {'lattice':>{lattice_col_w}}"
        )
    else:
        header = (
            f"  {'pk':>2}  {'frequency (MHz)':>18}  {amp_hdr:>16}  "
            f"{'phase (rad)':>12}  {'SNR':>6}  conf"
        )
    lines = [
        f"Window {window_id}  [{min(lo, hi):.4f}, {max(lo, hi):.4f}] MHz",
        f"  peaks={len(wf.fitted_peaks)}  chi2_r={float(wf.reduced_chi2):.3f}  "
        f"tau={tau:.4g} us  shape={getattr(wf, 'shape', 'lorentzian')}",
        "",
        header,
    ]
    labels = frequency_sorted_labels(
        [float(pk.frequency_mhz) for pk in wf.fitted_peaks]
    )
    for pk, lbl in zip(wf.fitted_peaks, labels):
        freq_s = _format_spectroscopic(float(pk.frequency_mhz), pk.frequency_error)
        amp_val = float(pk.amplitude) * amp
        amp_err = pk.amplitude_error * amp if pk.amplitude_error is not None else None
        amp_s = (
            f"{amp_val:.4g}+/-{amp_err:.2g}"
            if amp_err is not None
            else f"{amp_val:.4g}"
        )
        phase_s = (
            _format_spectroscopic(pk.phase, pk.phase_error)
            if pk.phase is not None
            else "-"
        )
        snr_s = f"{pk.snr:.2f}" if pk.snr is not None else "-"
        cl = getattr(pk, "clock_lattice", None)
        if show_lattice:
            lattice_s = (cl or "")[:lattice_col_w]
            lines.append(
                f"  {lbl:>2}  {freq_s:>18}  {amp_s:>16}  {phase_s:>12}  "
                f"{snr_s:>6}  -  {lattice_s:>{lattice_col_w}}"
            )
        else:
            lines.append(
                f"  {lbl:>2}  {freq_s:>18}  {amp_s:>16}  {phase_s:>12}  "
                f"{snr_s:>6}  -"
            )
    if show_audit:
        lines.append("")
        lines.append("  audit trail (add-one-peak):")
        audit = wf.audit_trail or []
        if not audit:
            lines.append("    (none)")
        else:
            for i, step in enumerate(audit):
                lines.append(
                    f"    step {i}: {step.decision} off={step.candidate_offset_mhz:+.4f} "
                    f"MHz  p={step.p_value:.2e}  "
                    f"chi2 {step.chi2_before:.3g}->{step.chi2_after:.3g}"
                )
    return "\n".join(lines)


def _window_peaks_baseband(
    window_fit: Any, sideband: Sideband, probe_freq_mhz: float
) -> list[Tuple[float, float, float]]:
    """Window's fitted peaks + frozen contributors as (A, f_bb, phase) tuples.

    ``f_bb = s*(f_molecular - f_probe)`` is the absolute baseband frequency the
    synthesized FID modulates at -- the raw active-region frame the real data
    lives in (not the window-centre offset frame ``model_spectrum`` uses).
    """
    from ..fitting.peak_model import sideband_sign

    s = sideband_sign(sideband)
    out: list[Tuple[float, float, float]] = []
    for p in window_fit.fitted_peaks:
        out.append(
            (
                float(p.amplitude),
                float(s * (p.frequency_mhz - probe_freq_mhz)),
                float(p.phase if p.phase is not None else 0.0),
            )
        )
    for key, fp in window_fit.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        out.append(
            (
                float(fp["amplitude"]),
                float(s * (float(fp["frequency_mhz"]) - probe_freq_mhz)),
                float(fp.get("phase", 0.0) or 0.0),
            )
        )
    return out


def render_windowed_view_impl(
    file_path: str,
    window_id: int,
    *,
    apodize: str = "exp",
    apodize_us: Optional[float] = None,
    bundle: Optional[_DetailBundle] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Any:
    """Render the windowed (apodized) data-vs-model comparison for one window.

    Applies the same time-domain window to the real active FID and to the
    persisted fit's re-synthesised model FID, transforms both, and overlays them
    over the window's frequency range. Strictly diagnostic: no re-fit, no fit
    statistics (windowing changes the noise correlation), and the leakage-wing
    baseline is omitted because apodization suppresses the skirt it compensates.
    """
    from ..fitting.peak_model import sideband_sign, synthesize_fid
    from ..utils.signal_processing import make_apodization
    from ..visualization.fit_detail import (
        MODEL_OVERSAMPLE,
        WindowedView,
        plot_windowed_comparison,
    )

    bundle = bundle if bundle is not None else _resolve_detail_bundle(file_path)
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        _n_padded,
        _acquisition_us,
        _user_ft,
        _trim_range,
    ) = _build_active_ft_inputs(file_path)

    # Active region exactly as compute_active_ft extracts it (searchsorted
    # bounds, rdc mean removal), but unapodized -- we apply our own window.
    fid = np.asarray(fid_samples, dtype=float)
    time_us = np.arange(fid.size) * sample_dt_us
    start_idx = int(np.searchsorted(time_us, start_us))
    end_idx = min(int(np.searchsorted(time_us, end_us)), fid.size)
    active = fid[start_idx:end_idx].astype(float, copy=True)
    active -= active.mean()
    n_active = active.size
    t_us = np.arange(n_active) * sample_dt_us
    s = sideband_sign(sideband)

    wf = bundle.fit.window_fit(window_id)
    lo, hi = wf.window.freq_range  # type: ignore[union-attr]
    lo_f, hi_f = float(min(lo, hi)), float(max(lo, hi))
    tau_us = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    shape = str(getattr(wf, "shape", "lorentzian"))
    peaks_bb = _window_peaks_baseband(wf, sideband, probe_freq_mhz)

    # Synthesize on the raw active region with the fitted tau. The canonical FT
    # is unapodized, so the fitted tau is the intrinsic decay and the boxcar
    # model sits on the data with no correction.
    model_fid = synthesize_fid(t_us, peaks_bb, tau_us, shape=shape)
    model_fid -= model_fid.mean()

    window = make_apodization(
        apodize, t_us, width_us=apodize_us, default_width_us=tau_us
    )
    data_w = active * window
    model_w = model_fid * window

    def _spec(signal: np.ndarray, pad_factor: int) -> Tuple[np.ndarray, np.ndarray]:
        n_pad = pad_factor * n_active
        padded = np.zeros(n_pad, dtype=float)
        padded[:n_active] = signal
        spec = sample_dt_us * np.fft.rfft(padded)
        freq = probe_freq_mhz + s * np.fft.rfftfreq(n_pad, d=sample_dt_us)
        order = np.argsort(freq)
        return np.ascontiguousarray(freq[order]), np.ascontiguousarray(spec[order])

    def _slice(freq: np.ndarray, spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = (freq >= lo_f) & (freq <= hi_f)
        return freq[m], spec[m]

    f1, d1 = _spec(data_w, 1)
    fm1, m1 = _spec(model_w, 1)
    f2, d2 = _spec(data_w, 2)
    ff, mf = _spec(model_w, MODEL_OVERSAMPLE)
    freq_native, data_native = _slice(f1, d1)
    _, model_native = _slice(fm1, m1)
    freq_data_2x, data_2x = _slice(f2, d2)
    freq_model_fine, model_fine = _slice(ff, mf)

    view = WindowedView(
        freq_native=freq_native,
        data_native=data_native,
        model_native=model_native,
        freq_data_2x=freq_data_2x,
        data_2x=data_2x,
        freq_model_fine=freq_model_fine,
        model_fine=model_fine,
    )
    if apodize.lower() in ("exp", "exponential"):
        w = apodize_us if apodize_us is not None else tau_us
        suffix = "" if apodize_us is not None else " = tau"
        apo_label = f"exp (W={w:.3g} us{suffix})"
    else:
        apo_label = apodize
    if title is None:
        title = (
            f"{bundle.file_stem}  window {window_id}  "
            f"[{lo_f:.2f}, {hi_f:.2f}] MHz  windowed view"
        )
    return plot_windowed_comparison(
        view,
        title=title,
        apodize_label=apo_label,
        amplitude_scale=bundle.amplitude_scale,
        units_label=bundle.units_label,
        figsize=figsize if figsize is not None else (11, 4.2),
    )


def _has_selection(
    window_ids: Optional[list[int]],
    freqs: Optional[list[float]],
    random_n: Optional[int],
    top_snr: Optional[int],
    all_windows: bool,
) -> bool:
    return bool(window_ids or freqs or random_n or top_snr or all_windows)


def fit_show_impl(
    file_path: str,
    *,
    window_ids: Optional[list[int]] = None,
    freqs: Optional[list[float]] = None,
    random_n: Optional[int] = None,
    random_seed: Optional[int] = None,
    top_snr: Optional[int] = None,
    all_windows: bool = False,
    output_dir: Optional[str] = None,
    show_audit: bool = False,
    apodize: Optional[str] = None,
    apodize_us: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Drive ``fit show``: overview (no selector) or one consolidated detail
    figure per selected window.

    Returns ``{"mode", "window_ids", "figures", "paths", "log"}``. With
    ``output_dir`` each detail figure is written as
    ``<stem>_window_<id>.png`` and its path collected; the figures are also
    returned so an interactive caller can display them. The text fit log for the
    selected windows is in ``"log"``. With ``apodize`` set, an extra windowed
    (apodized) data-vs-model comparison figure is produced per window
    (``<stem>_window_<id>_apodized.png``) -- a diagnostic view, not a re-fit.
    """
    if not _has_selection(window_ids, freqs, random_n, top_snr, all_windows):
        fig = visualize_fit_impl(
            file_path=file_path, figsize=figsize, title=title, window_id=None
        )
        return {
            "mode": "overview",
            "window_ids": [],
            "figures": [fig],
            "paths": [],
            "log": "",
        }

    bundle = _resolve_detail_bundle(file_path)
    ids = select_window_ids(
        bundle.fit,
        window_ids=window_ids,
        freqs=freqs,
        random_n=random_n,
        random_seed=random_seed,
        top_snr=top_snr,
        all_windows=all_windows,
    )

    out_dir: Optional[Path] = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Any] = []
    paths: list[str] = []
    reports: list[str] = []
    for wid in ids:
        fig = render_fit_detail_impl(
            file_path, wid, bundle=bundle, figsize=figsize, title=title
        )
        figures.append(fig)
        reports.append(
            fit_window_report_text(file_path, wid, bundle=bundle, show_audit=show_audit)
        )
        if out_dir is not None:
            dest = out_dir / f"{bundle.file_stem}_window_{wid:03d}.png"
            fig.savefig(str(dest), dpi=130)
            paths.append(str(dest))
        if apodize:
            wfig = render_windowed_view_impl(
                file_path,
                wid,
                apodize=apodize,
                apodize_us=apodize_us,
                bundle=bundle,
            )
            figures.append(wfig)
            if out_dir is not None:
                wdest = out_dir / f"{bundle.file_stem}_window_{wid:03d}_apodized.png"
                wfig.savefig(str(wdest), dpi=130)
                paths.append(str(wdest))
    return {
        "mode": "detail",
        "window_ids": ids,
        "figures": figures,
        "paths": paths,
        "log": "\n\n".join(reports),
    }
