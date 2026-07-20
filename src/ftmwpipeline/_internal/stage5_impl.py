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
same ``start_us`` / ``end_us`` the user picked for the persisted spectrum; the
canonical FT is unapodized, native-length, and unconditionally DC-removed).
Per-bin noise on the active-FT is measured fresh by
running the Stage 2 adaptive estimator on the active-FT magnitude spectrum,
rather than rescaled from the persisted Stage 1/2 noise.

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

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
    FinalizeNode,
    ReplanContext,
    WindowOutcome,
    execute_plan,
    parallel_window_refit_map,
    refit_outcome,
)
from ..fitting.result_conversion import (
    plan_fit_outcome_to_spectrum_fit,
    sort_fitting_result_by_frequency,
)
from ..fitting.spur_detection import (
    SpurSet,
    build_spur_set,
    make_band_power_probe,
    make_chirp_response_probe,
    make_decay_probe,
)
from ..fitting.tau_calibration import (
    BandMajority,
    TauCalibrationResult,
    band_majority_for_frequency,
)
from ..fitting.validation import amplitude_vif
from ..io.fid_serialization import load_acquisition_segments_from_hdf5
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
from ..preprocessing.peak_detection import DEFAULT_MIN_SNR as DEFAULT_PROMOTION_MIN_SNR
from .active_ft_support import (
    _persisted_scatter_knobs,
    build_active_grid_with_noise,
    default_tau0_us,
)
from .shared_utils import require_resolved
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
from .stage3_impl import (
    _active_acquisition_us,
    load_peaks_impl,
)
from .stage4_impl import load_windows_impl

if TYPE_CHECKING:
    from ..fitting.result_conversion import FittedLineView

logger = logging.getLogger(__name__)


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


def annotate_flat_decay_matches(
    spectrum_fit: SpectrumFit, spur_set: Optional[Any]
) -> None:
    """Stamp ``flat_decay`` on every fitted peak the spur gate kept-but-flagged.

    Called after the fit is assembled but before persistence. A Stage-2b
    flat-cluster nominee whose coherent decay was ambiguous (the ``flat_decay``
    band) is fit rather than masked, but surfaced for review: this stamps the
    review hint on the matching fitted peak. Purely informational -- no effect
    on the fit. No-op when the spur set carries no flags.
    """
    if spur_set is None or not getattr(spur_set, "flat_decay_flags", ()):
        return
    for peak in spectrum_fit.fitted_peaks:
        if spur_set.flat_decay_match(peak.frequency_mhz):
            peak.flat_decay = True
    for wf in spectrum_fit.window_fits:
        for peak in wf.fitted_peaks:
            if spur_set.flat_decay_match(peak.frequency_mhz):
                peak.flat_decay = True


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


def resolve_window_tau_anchor(
    center_mhz: float,
    band_majorities: Optional[Tuple["BandMajority", ...]],
    fallback_tau_maj_us: Optional[float],
    fallback_sigma_tau_us: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """The single source of truth for a window's tau penalty anchor.

    Stage 2b persists *per-band* tau majorities; a window's tau penalty (and its
    starting tau) must anchor at the band its center frequency falls in, **not**
    the global band-wide ``tau_maj``. Returns the per-band
    ``(tau_maj_us, sigma_tau_us)`` when ``center_mhz`` maps to a band, else the
    band-wide fallback.

    Every fit and refit path -- the main fit's ``window_tau_overrides``, the
    in-fit survival-prune / VIF-collapse refits, and the Stage 6 edit / cascade
    refit -- must resolve its anchor through this one helper so they reproduce
    the tau the originating fit used. Reaching for the persisted band-wide
    ``tau_maj`` directly is the recurring bug this function exists to prevent.
    See the ROADMAP cleanup note on retiring ``tau_maj`` (it is exactly a
    single-band tau, so per-band majorities should be the only representation).
    """
    if band_majorities:
        band = band_majority_for_frequency(band_majorities, center_mhz)
        if band is not None:
            return float(band.tau_maj_us), float(band.sigma_tau_us)
    return fallback_tau_maj_us, fallback_sigma_tau_us


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
    sideband = Sideband.coerce(fid.sideband)

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
    return float(require_resolved(value, name, cast=float, owner="StageFitSettings"))


def _required_int(value: Optional[int], name: str) -> int:
    """Coerce a post-resolve field that must be filled into ``int``."""
    return int(require_resolved(value, name, cast=int, owner="StageFitSettings"))


def _required_bool(value: Optional[bool], name: str) -> bool:
    """Coerce a post-resolve field that must be filled into ``bool``."""
    return bool(require_resolved(value, name, cast=bool, owner="StageFitSettings"))


def _required_str(value: Optional[str], name: str) -> str:
    """Coerce a post-resolve field that must be filled into ``str``."""
    return str(require_resolved(value, name, cast=str, owner="StageFitSettings"))


# A collapsed line represents one unresolvable feature, so the span of the
# original component frequencies it absorbs must stay within this many resolution
# elements. It bounds a *chained* fold (one sub-pair merges, then the merged line
# would merge again): above this the fold would cross a genuinely-resolvable gap
# into a real neighboring line. Set above the calibration-truth doublet floor
# (~0.82 res) and below the smallest resolved doublet observed in the fixtures
# (~1.5 res), so an over-split cluster (span ~1 res) fully collapses while a real
# resolved doublet over-split into a triplet keeps its two centroids.
_COLLAPSE_FOOTPRINT_MAX_RES = 1.3

# Degenerate-pair merge TRIAL band (Type A over-splits). The unconditional VIF /
# fractional-uncertainty collapse stops at ``collapse_frac_unc_max_separation_res``
# because past it a high fractional uncertainty is ambiguous: a genuine over-split
# (two halves of one feature, the merge HOLDS chi2r) and a real but
# poorly-conditioned doublet (the merge BLOWS chi2r up) are indistinguishable by
# any static quantity. In the band (frac_unc_max_sep, collapse_max_separation_res]
# a pair where BOTH members are clearly degenerate (fractional amplitude
# uncertainty >= ``degenerate_trial_frac`` -- which a bright-line sidelobe never
# satisfies, its bright parent is well-determined) is *trial-merged* and the merge
# KEPT only if the post-merge chi2r does not rise by more than
# ``degenerate_trial_chi2r_rel_tol`` (relative). See
# :class:`~ftmwpipeline.core.stage_fit_settings.PeakSurvivalSubSettings`.


def _inflate_merged_frequency_errors(
    wf: FittingResult, pending: List[Dict[str, Any]]
) -> None:
    """Inflate each merged line's ``frequency_error`` by its component spread.

    The collapsed multiplet's effective frequency uncertainty is
    ``sqrt(formal_frequency_error**2 + unresolved_spread_mhz**2)`` -- the merged
    line's position is honestly known only to within the (unresolved-hyperfine)
    spread of the components it absorbed. Each ``pending`` record is matched to
    its merged peak (nearest ``frequency_mhz`` to ``merged_frequency_mhz``) and
    the peak is rebuilt with the inflated error; the spread is also recorded in
    ``extra_errors['unresolved_spread_mhz']`` for the report/provenance. The
    merged peaks were stamped ``origin="auto"`` by the collapse refit.
    """
    peaks = list(wf.fitted_peaks)
    used: set[int] = set()
    for rec in pending:
        spread = float(rec.get("unresolved_spread_mhz") or 0.0)
        target = float(rec["merged_frequency_mhz"])
        best_k = -1
        best_d = float("inf")
        for k, pk in enumerate(peaks):
            if k in used:
                continue
            d = abs(float(pk.frequency_mhz) - target)
            if d < best_d:
                best_d, best_k = d, k
        if best_k < 0:
            continue
        used.add(best_k)
        pk = peaks[best_k]
        formal = float(pk.frequency_error) if pk.frequency_error is not None else 0.0
        eff = float(np.hypot(formal, spread))
        extra_errors = dict(pk.extra_errors)
        extra_errors["unresolved_spread_mhz"] = spread
        peaks[best_k] = replace(pk, frequency_error=eff, extra_errors=extra_errors)
    wf.fitted_peaks = peaks


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


# ---------------------------------------------------------------------------
# Per-node cleanup, in outcome space (folded into the fit walk's per-node tail)
# ---------------------------------------------------------------------------
# The SNR-survival prune and VIF-collapse, re-expressed on a live
# ``WindowOutcome`` and driven by ``refit_outcome`` instead of the global
# post-pass over persisted ``FittingResult`` s. Folding them into the walk's
# per-node tail means every dependent window is fit against an *already cleaned*
# source (the ``#3b`` stale-frozen-background class), and each refit inherits the
# node's exact fit kwargs (per-band tau anchor, baseline order, spur mask) by
# construction -- the global-anchor bug class cannot recur.


def _is_survival_dust_view(view: "FittedLineView", floor: float) -> bool:
    """True when ``view`` is auto-origin dust below the SNR floor (outcome twin
    of :func:`_is_survival_dust`)."""
    import math

    snr = view.snr
    if view.origin == "user" or snr is None:
        return False
    if isinstance(snr, float) and math.isnan(snr):
        return False
    return snr < floor


def _prune_outcome(
    outcome: "WindowOutcome",
    floor: float,
    *,
    sideband: Sideband,
    acquisition_us: float,
    records: List[Dict[str, Any]],
) -> Tuple[Optional["WindowOutcome"], int]:
    """Prune one window's sub-floor dust to a fixpoint, in outcome space.

    Outcome twin of :func:`_survival_prune_window`: remove the single lowest-SNR
    dust line, refit the survivors via :func:`refit_outcome`, re-classify, repeat.
    Returns ``(outcome, n_refits)``; ``outcome`` is ``None`` when the window
    cascades to empty (its last line is itself dust)."""
    from ..fitting.result_conversion import outcome_line_views

    current = outcome
    n_refits = 0
    while True:
        views = outcome_line_views(
            current, sideband=sideband, acquisition_us=acquisition_us
        )
        dust = [v for v in views if _is_survival_dust_view(v, floor)]
        if not dust:
            return current, n_refits
        worst = min(dust, key=lambda v: cast(float, v.snr))
        records.append(
            {
                "window_id": int(current.window_id),
                "frequency_mhz": float(worst.frequency_mhz),
                "snr": float(cast(float, worst.snr)),
            }
        )
        if len(views) == 1:
            return None, n_refits
        current = refit_outcome(current, remove_offsets=[worst.offset_mhz])
        n_refits += 1


def _collapse_rank(
    view: "FittedLineView",
    vif_threshold: float,
    frac_threshold: float = float("inf"),
) -> Optional[float]:
    """Collapse-eligibility rank for one line, or ``None`` if not eligible.

    The amplitude VIF when the line is not individually constrained -- either its
    VIF clears ``vif_threshold`` (the brightness-invariant degeneracy) or its
    fractional amplitude uncertainty ``amp_err / amp`` (= VIF / snr) reaches
    ``frac_threshold`` (the sub-resolution over-split band at VIF 4-25 the VIF
    gate alone leaves behind); ``+inf`` for the strongest degeneracy (a finite,
    non-zero amplitude whose joint covariance is singular, so the VIF is
    undefined and the gate alone would miss it); ``None`` otherwise. A zero /
    non-finite amplitude is a dead peak, not a degeneracy, so it never ranks.
    Pure decision function over a :class:`FittedLineView` so both the in-walk
    collapse and its unit tests read the same logic."""
    vif = view.amplitude_vif()
    if vif is not None:
        if vif > vif_threshold:
            return vif
        # ``amplitude_vif`` is non-None only when amp / amp_err / snr are all
        # finite and amp != 0, so the fractional uncertainty is well-defined.
        frac = abs(float(view.amplitude_error) / float(view.amplitude))  # type: ignore[arg-type]
        return vif if frac >= frac_threshold else None
    amp = float(view.amplitude)
    snr = view.snr
    amp_err = view.amplitude_error
    amplitude_singular = amp_err is None or not np.isfinite(amp_err)
    if (
        snr is not None
        and np.isfinite(snr)
        and np.isfinite(amp)
        and abs(amp) > 0.0
        and amplitude_singular
    ):
        return float("inf")
    return None


def _collapse_outcome(
    outcome: "WindowOutcome",
    *,
    vif_threshold: float,
    frac_threshold: float,
    frac_max_sep_res: float,
    max_sep_res: float,
    res_element_mhz: float,
    sideband: Sideband,
    acquisition_us: float,
    snap_tol_mhz: float,
    max_iterations: int,
    records: List[Dict[str, Any]],
) -> "WindowOutcome":
    """Collapse degenerate sub-resolution overfit pairs to a fixpoint, in outcome
    space (outcome twin of :func:`apply_vif_collapse`'s ``_collapse_one``).

    Same gate (amplitude VIF + separation + footprint guard), same sequence
    (frozen single-pair merge sweep to a fixpoint -> one all-free relax -> repeat
    over the relaxed result, capped at ``max_iterations``). Each merge appends a
    collapse provenance record; the merged line's frequency-error inflation by
    the unresolved spread is applied at end-of-walk from those records (reusing
    :func:`_inflate_merged_frequency_errors`)."""
    from ..fitting.peak_model import ModelPeak, sideband_sign
    from ..fitting.result_conversion import FittedLineView, outcome_line_views

    max_sep_mhz = max_sep_res * res_element_mhz
    frac_max_sep_mhz = frac_max_sep_res * res_element_mhz
    footprint_cap_mhz = _COLLAPSE_FOOTPRINT_MAX_RES * res_element_mhz
    s = sideband_sign(sideband)
    center_mhz = float(getattr(outcome, "_center_mhz"))

    Footprint = Tuple[float, float, float]

    def _line_max_sep_mhz(view: FittedLineView) -> float:
        """Per-line collapse reach. A line that clears the VIF / singular gate is
        an unambiguous degeneracy and may merge to the full guard; a line
        eligible only via its fractional amplitude uncertainty merges within the
        tighter deep-sub-resolution cap, where a high ``amp_err/amp`` reflects
        non-identifiability rather than modest SNR on a resolvable doublet."""
        vif = view.amplitude_vif()
        if vif is None or vif > vif_threshold:
            return max_sep_mhz
        return frac_max_sep_mhz

    def _select_pair(
        views: List[FittedLineView], foots: List[Footprint]
    ) -> Optional[Tuple[int, int]]:
        if len(views) < 2:
            return None
        ranks = [_collapse_rank(v, vif_threshold, frac_threshold) for v in views]
        high_vif_order = sorted(
            (i for i, r in enumerate(ranks) if r is not None),
            key=lambda i: -(ranks[i] or 0.0),
        )
        for i in high_vif_order:
            fi = float(views[i].frequency_mhz)
            this_max_sep_mhz = _line_max_sep_mhz(views[i])
            best_j: Optional[int] = None
            best_d = float("inf")
            for j in range(len(views)):
                if j == i:
                    continue
                fj = float(views[j].frequency_mhz)
                d = abs(fi - fj)
                if d > this_max_sep_mhz or d >= best_d:
                    continue
                lo = min(foots[i][1], foots[j][1], fi, fj)
                hi = max(foots[i][2], foots[j][2], fi, fj)
                if hi - lo > footprint_cap_mhz:
                    continue
                best_d, best_j = d, j
            if best_j is not None:
                return (i, best_j)
        return None

    def _merged_seed(
        va: FittedLineView, vb: FittedLineView
    ) -> Tuple[float, float, float]:
        """Merged-line seed offset / amplitude / phase. Defaults to the
        amplitude-weighted centroid + summed amplitude; reuses a recorded
        doublet adjudication's converged merged model when present (offset-frame
        twin of :func:`_merged_seed_for_pair`)."""
        import math as _math

        oa, ob = float(va.offset_mhz), float(vb.offset_mhz)
        aw, bw = abs(float(va.amplitude)), abs(float(vb.amplitude))
        total = aw + bw if (aw + bw) > 0 else 1.0
        centroid_off = (oa * aw + ob * bw) / total
        centroid_amp = float(va.amplitude) + float(vb.amplitude)
        m_off, m_amp, m_phase = centroid_off, centroid_amp, 0.0
        for adj in getattr(outcome, "doublet_adjudications", []):
            a, b = float(adj.offset_a_mhz), float(adj.offset_b_mhz)
            pair_match = (
                abs(a - oa) <= snap_tol_mhz and abs(b - ob) <= snap_tol_mhz
            ) or (abs(a - ob) <= snap_tol_mhz and abs(b - oa) <= snap_tol_mhz)
            if (
                pair_match
                and adj.merged_success
                and _math.isfinite(float(adj.merged_offset_mhz))
            ):
                m_off = float(adj.merged_offset_mhz)
                if _math.isfinite(float(adj.merged_amplitude)):
                    m_amp = float(adj.merged_amplitude)
                if _math.isfinite(float(adj.merged_phase)):
                    m_phase = float(adj.merged_phase)
                break
        return m_off, m_amp, m_phase

    def _rekey_footprints(
        views: List[FittedLineView], prev: List[Footprint]
    ) -> List[Footprint]:
        used: set[int] = set()
        out: List[Footprint] = []
        for v in views:
            f = float(v.frequency_mhz)
            best_k = -1
            best_d = float("inf")
            for k, (c, _lo, _hi) in enumerate(prev):
                if k in used:
                    continue
                d = abs(f - c)
                if d < best_d:
                    best_d, best_k = d, k
            if best_k >= 0:
                used.add(best_k)
                _c, lo, hi = prev[best_k]
                out.append((f, lo, hi))
            else:
                out.append((f, f, f))
        return out

    cur = outcome
    views = outcome_line_views(cur, sideband=sideband, acquisition_us=acquisition_us)
    foots: List[Footprint] = [(float(v.frequency_mhz),) * 3 for v in views]
    pending: List[Dict[str, Any]] = []
    wid = int(cur.window_id)

    for _ in range(max(1, max_iterations)):
        merged_this_pass = 0
        while True:
            views = outcome_line_views(
                cur, sideband=sideband, acquisition_us=acquisition_us
            )
            if len(views) < 2:
                break
            pair = _select_pair(views, foots)
            if pair is None:
                break
            i, j = pair
            va, vb = views[i], views[j]
            vif_a, vif_b = va.amplitude_vif(), vb.amplitude_vif()
            m_off, m_amp, m_phase = _merged_seed(va, vb)
            merge_freq = float(center_mhz + s * m_off)
            seed = ModelPeak(
                amplitude=max(abs(m_amp), 1e-30),
                offset_mhz=float(m_off),
                phase=float(m_phase),
            )
            fa, fb = float(va.frequency_mhz), float(vb.frequency_mhz)
            amp_a, amp_b = abs(float(va.amplitude)), abs(float(vb.amplitude))
            w_sum = amp_a + amp_b
            if w_sum > 0.0:
                f_centroid = (amp_a * fa + amp_b * fb) / w_sum
                spread = float(
                    np.sqrt(
                        (
                            amp_a * (fa - f_centroid) ** 2
                            + amp_b * (fb - f_centroid) ** 2
                        )
                        / w_sum
                    )
                )
            else:
                spread = 0.5 * abs(fa - fb)
            pending.append(
                {
                    "window_id": wid,
                    "frequency_a_mhz": fa,
                    "frequency_b_mhz": fb,
                    "vif_a": vif_a,
                    "vif_b": vif_b,
                    "separation_res": (
                        abs(fa - fb) / res_element_mhz if res_element_mhz > 0 else None
                    ),
                    "merged_frequency_mhz": merge_freq,
                    "unresolved_spread_mhz": spread,
                }
            )
            merged_foot: Footprint = (
                merge_freq,
                min(foots[i][1], foots[j][1], fa, fb),
                max(foots[i][2], foots[j][2], fa, fb),
            )
            next_foots = [foots[k] for k in range(len(foots)) if k not in (i, j)] + [
                merged_foot
            ]
            cur = refit_outcome(
                cur,
                remove_offsets=[va.offset_mhz, vb.offset_mhz],
                add_seeds=[seed],
                freeze_inherited=True,
            )
            views = outcome_line_views(
                cur, sideband=sideband, acquisition_us=acquisition_us
            )
            foots = _rekey_footprints(views, next_foots)
            merged_this_pass += 1

        if merged_this_pass == 0:
            break

        cur = refit_outcome(cur, freeze_inherited=False)
        views = outcome_line_views(
            cur, sideband=sideband, acquisition_us=acquisition_us
        )
        foots = [(float(v.frequency_mhz),) * 3 for v in views]

    records.extend(pending)
    return cur


def _degenerate_merge_trial_outcome(
    outcome: "WindowOutcome",
    *,
    min_sep_res: float,
    max_sep_res: float,
    degenerate_frac: float,
    chi2r_rel_tol: float,
    res_element_mhz: float,
    sideband: Sideband,
    acquisition_us: float,
    records: List[Dict[str, Any]],
    max_iterations: int = 6,
) -> "WindowOutcome":
    """Trial-merge comparable-brightness degenerate over-splits in the band the
    unconditional collapse leaves behind (Type A), in outcome space.

    For a pair in ``(min_sep_res, max_sep_res]`` resolution elements where BOTH
    members are clearly degenerate (fractional amplitude uncertainty >=
    ``_DEGEN_TRIAL_FRAC`` -- a bright-line sidelobe never qualifies, its parent is
    well determined, so this does not overlap the sidelobe prune), the pair is
    merged to its amplitude-weighted centroid and refit; the merge is kept only
    when ``chi2r`` does not rise by more than ``_DEGEN_TRIAL_CHI2R_REL`` (relative).
    The merge holding distinguishes a genuine over-split (kept) from a real
    poorly-conditioned doublet (rejected, chi2r blows up). Accepted merges append
    the same collapse provenance the VIF collapse writes, so the merged line's
    frequency-error inflation and the ``auto_merged_review`` flag both apply."""
    if res_element_mhz <= 0.0 or max_sep_res <= min_sep_res or degenerate_frac <= 0.0:
        return outcome
    from ..fitting.peak_model import ModelPeak, sideband_sign
    from ..fitting.result_conversion import FittedLineView, outcome_line_views

    min_sep_mhz = min_sep_res * res_element_mhz
    max_sep_mhz = max_sep_res * res_element_mhz
    s = sideband_sign(sideband)

    def _frac(v: FittedLineView) -> Optional[float]:
        amp, ae = v.amplitude, v.amplitude_error
        if amp is None or ae is None or float(amp) == 0.0 or not np.isfinite(ae):
            return None
        return abs(float(ae) / float(amp))

    rejected: set = set()
    cur = outcome
    for _ in range(max_iterations):
        views = outcome_line_views(
            cur, sideband=sideband, acquisition_us=acquisition_us
        )
        if len(views) < 2:
            break
        center_mhz = float(getattr(cur, "_center_mhz"))
        best: Optional[Tuple[float, int, int, Tuple[float, float]]] = None
        for i in range(len(views)):
            fri = _frac(views[i])
            if fri is None or fri < degenerate_frac:
                continue
            fi = float(views[i].frequency_mhz)
            for j in range(i + 1, len(views)):
                frj = _frac(views[j])
                if frj is None or frj < degenerate_frac:
                    continue
                fj = float(views[j].frequency_mhz)
                d = abs(fi - fj)
                if not (min_sep_mhz < d <= max_sep_mhz):
                    continue
                key = (round(min(fi, fj), 3), round(max(fi, fj), 3))
                if key in rejected:
                    continue
                if best is None or d < best[0]:
                    best = (d, i, j, key)
        if best is None:
            break
        d, i, j, key = best
        va, vb = views[i], views[j]
        oa, ob = float(va.offset_mhz), float(vb.offset_mhz)
        aw, bw = abs(float(va.amplitude)), abs(float(vb.amplitude))
        total = aw + bw if (aw + bw) > 0 else 1.0
        m_off = (oa * aw + ob * bw) / total
        m_amp = float(va.amplitude) + float(vb.amplitude)
        seed = ModelPeak(
            amplitude=max(abs(m_amp), 1e-30), offset_mhz=float(m_off), phase=0.0
        )
        chi2_before = float(cur.fit.fit.reduced_chi2)
        trial = refit_outcome(
            cur,
            remove_offsets=[va.offset_mhz, vb.offset_mhz],
            add_seeds=[seed],
            freeze_inherited=False,
        )
        chi2_after = float(trial.fit.fit.reduced_chi2)
        if np.isfinite(chi2_after) and chi2_after <= chi2_before * (
            1.0 + chi2r_rel_tol
        ):
            fa, fb = float(va.frequency_mhz), float(vb.frequency_mhz)
            f_centroid = (aw * fa + bw * fb) / total
            spread = float(
                np.sqrt(
                    (aw * (fa - f_centroid) ** 2 + bw * (fb - f_centroid) ** 2) / total
                )
            )
            records.append(
                {
                    "window_id": int(cur.window_id),
                    "frequency_a_mhz": fa,
                    "frequency_b_mhz": fb,
                    "vif_a": va.amplitude_vif(),
                    "vif_b": vb.amplitude_vif(),
                    "separation_res": d / res_element_mhz,
                    "merged_frequency_mhz": float(center_mhz + s * m_off),
                    "unresolved_spread_mhz": spread,
                    "trial_chi2r_before": chi2_before,
                    "trial_chi2r_after": chi2_after,
                }
            )
            cur = trial
        else:
            rejected.add(key)
    return cur


def _is_brightness_sidelobe(
    victim: "FittedLineView",
    neighbor: "FittedLineView",
    *,
    max_sep_res: float,
    res_element_mhz: float,
) -> bool:
    """True when *victim* lies inside *neighbor*'s brightness-scaled lineshape
    skirt (Type B), the bright-line artifact predicate.

    *neighbor* must be brighter, and the separation within ``min(max_sep_res,
    SHAPE_ERROR_REACH_KAPPA * snr_neighbor / snr_victim)`` resolution elements --
    the finite-T sinc skirt, brightness-scaled (the same reach the Stage 6
    candidate ledger and the F-2/F-3 install filter use), capped to the near-field
    where the apodization pass cannot vouch for realness. Comparable-brightness
    pairs (``snr_neighbor ~ snr_victim``) get a sub-``kappa`` reach and so never
    qualify -- they are the VIF collapse's business. Pure decision function over
    two :class:`FittedLineView`, mirroring :func:`_collapse_rank`."""
    from ..fitting.validation import SHAPE_ERROR_REACH_KAPPA

    if max_sep_res <= 0.0 or res_element_mhz <= 0.0:
        return False
    sv, sn = victim.snr, neighbor.snr
    if sv is None or sn is None or not np.isfinite(sv) or not np.isfinite(sn):
        return False
    if sv <= 0.0 or sn <= sv:
        return False
    sep_res = (
        abs(float(victim.frequency_mhz) - float(neighbor.frequency_mhz))
        / res_element_mhz
    )
    reach = min(max_sep_res, SHAPE_ERROR_REACH_KAPPA * float(sn) / float(sv))
    return sep_res <= reach


def _sidelobe_prune_outcome(
    outcome: "WindowOutcome",
    *,
    max_sep_res: float,
    res_element_mhz: float,
    sideband: Sideband,
    acquisition_us: float,
    records: List[Dict[str, Any]],
    max_iterations: int = 10,
) -> "WindowOutcome":
    """Remove bright-neighbor lineshape sidelobes to a fixpoint, in outcome space.

    A fitted peak that sits inside a *brighter* neighbor's lineshape-error shadow
    -- ``sep_res <= SHAPE_ERROR_REACH_KAPPA * snr_bright / snr_self`` (the finite-T
    sinc skirt, brightness-scaled; same predicate the Stage 6 candidate ledger and
    the F-2/F-3 install filter use) **and** within ``max_sep_res`` resolution
    elements -- is not a molecular line but the bright line's lineshape artifact.
    Removing it raises ``chi2r`` (the artifact was absorbing real lineshape error),
    which is the honest signal: the goal is reliable molecular lines, not a low
    ``chi2r``, so the removal overrides ``chi2r`` exactly as the VIF collapse does.

    The reach cap matters because the ``kappa * snr_bright / snr_self`` reach runs
    out to tens of resolution elements for a very bright line and a faint
    candidate, where a feature is fully resolved and may be a real faint line; the
    Blackman-Harris apodization pass is the realness arbiter beyond a couple of
    resolution elements, so the cap keeps the prune to the near-field skirt.
    user-origin peaks are immune. Comparable-brightness degenerate pairs
    (``snr_bright ~ snr_self``) have a sub-``kappa`` reach and so are left to the
    VIF collapse, not removed here -- the two passes do not overlap.

    The faintest victim is removed and the survivors refit before re-checking, so
    a window with several sidelobes converges over a few iterations."""
    if max_sep_res <= 0.0 or res_element_mhz <= 0.0:
        return outcome
    from ..fitting.result_conversion import outcome_line_views

    def _finite_snr(v: "FittedLineView") -> Optional[float]:
        snr = v.snr
        if snr is None or not np.isfinite(snr) or snr <= 0.0:
            return None
        return float(snr)

    current = outcome
    for _ in range(max_iterations):
        views = outcome_line_views(
            current, sideband=sideband, acquisition_us=acquisition_us
        )
        if len(views) < 2:
            break
        victim: Optional[Tuple["FittedLineView", float, float]] = None
        for vi in views:
            if vi.origin == "user":
                continue
            si = _finite_snr(vi)
            if si is None:
                continue
            for vj in views:
                if vj is vi:
                    continue
                sj = _finite_snr(vj)
                if sj is None:
                    continue
                if _is_brightness_sidelobe(
                    vi, vj, max_sep_res=max_sep_res, res_element_mhz=res_element_mhz
                ):
                    # The faintest qualifying peak is the clearest artifact.
                    if victim is None or si < victim[1]:
                        victim = (vi, si, sj)
                    break
        if victim is None:
            break
        v, si, sj = victim
        records.append(
            {
                "window_id": int(current.window_id),
                "frequency_mhz": float(v.frequency_mhz),
                "snr": float(si),
                "neighbor_snr": float(sj),
            }
        )
        current = refit_outcome(current, remove_offsets=[v.offset_mhz])
    return current


def build_finalize_node(
    *,
    floor: float,
    enabled: bool,
    vif_threshold: float,
    frac_threshold: float,
    frac_max_sep_res: float,
    max_sep_res: float,
    sidelobe_max_sep_res: float,
    degenerate_trial_frac: float,
    degenerate_trial_chi2r_rel_tol: float,
    res_element_mhz: float,
    sideband: Sideband,
    acquisition_us: float,
    snap_tol_mhz: float = 0.05,
    max_iterations: int = 5,
) -> "FinalizeNode":
    """Build the per-node cleanup callback injected into the fit walk.

    The returned ``finalize_node(outcome) -> NodeCleanup`` prunes one window's
    sub-floor dust to a fixpoint, then collapses its degenerate sub-resolution
    pairs to a fixpoint -- in outcome space, driving :func:`refit_outcome`. The
    per-window prune / collapse provenance rides on the :class:`NodeCleanup`
    (carried out even on a drop); the end-of-walk aggregation derives the
    ``peak_survival`` / ``vif_collapse`` diagnostics from it. A disabled cleanup
    returns the outcome untouched."""
    from ..fitting.plan_execution import NodeCleanup

    def finalize_node(outcome: "WindowOutcome") -> "NodeCleanup":
        if not enabled:
            return NodeCleanup(outcome=outcome)
        pruned_records: List[Dict[str, Any]] = []
        result, _ = _prune_outcome(
            outcome,
            floor,
            sideband=sideband,
            acquisition_us=acquisition_us,
            records=pruned_records,
        )
        if result is None:
            return NodeCleanup(outcome=None, pruned=pruned_records)
        collapse_records: List[Dict[str, Any]] = []
        result = _collapse_outcome(
            result,
            vif_threshold=vif_threshold,
            frac_threshold=frac_threshold,
            frac_max_sep_res=frac_max_sep_res,
            max_sep_res=max_sep_res,
            res_element_mhz=res_element_mhz,
            sideband=sideband,
            acquisition_us=acquisition_us,
            snap_tol_mhz=snap_tol_mhz,
            max_iterations=max_iterations,
            records=collapse_records,
        )
        result = _degenerate_merge_trial_outcome(
            result,
            min_sep_res=frac_max_sep_res,
            max_sep_res=max_sep_res,
            degenerate_frac=degenerate_trial_frac,
            chi2r_rel_tol=degenerate_trial_chi2r_rel_tol,
            res_element_mhz=res_element_mhz,
            sideband=sideband,
            acquisition_us=acquisition_us,
            records=collapse_records,
        )
        sidelobe_records: List[Dict[str, Any]] = []
        result = _sidelobe_prune_outcome(
            result,
            max_sep_res=sidelobe_max_sep_res,
            res_element_mhz=res_element_mhz,
            sideband=sideband,
            acquisition_us=acquisition_us,
            records=sidelobe_records,
        )
        return NodeCleanup(
            outcome=result,
            pruned=pruned_records,
            collapses=collapse_records,
            sidelobes=sidelobe_records,
        )

    return finalize_node


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
    replay_spur_catalog: Optional[Mapping[str, Any]] = None,
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
    replay_spur_catalog :
        When given (a persisted ``SpectrumFit.parameters`` mapping), the gated
        spur catalog is **replayed** from it -- the spur detector (decay /
        chirp probes, frequency-domain sweep, clock lattice) does NOT run.
        The detection is a Stage 5 product; a later stage (the Stage 6
        single-window refit) must reproduce the exact mask the fit used rather
        than re-deriving a possibly-drifted catalog.  ``None`` (the
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
        spur_set_from_catalog,
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
    if spur_enabled and replay_spur_catalog is not None:
        # Replay the persisted Stage 5 gated catalog verbatim (no detection):
        # the catalog is a Stage 5 product, so a later-stage refit reproduces
        # the exact residual mask the fit used instead of re-deriving it.
        cat = replay_spur_catalog
        bin_spacing = (
            float(np.median(np.abs(np.diff(sorted_freq))))
            if sorted_freq.size > 1
            else 0.0
        )
        spur_set = spur_set_from_catalog(
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
        use_catalog = (
            True
            if spur_cfg.use_stft_catalog is None
            else bool(spur_cfg.use_stft_catalog)
        )
        saturated_clusters = (
            persisted_cal.spur_clusters
            if (persisted_cal is not None and use_catalog)
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
            use_stft_catalog=use_catalog,
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
                "(sources: %s); mask half-width %d bins, catalog=%s",
                len(spur_set.spurs),
                ", ".join(sorted({s.source for s in spur_set.spurs})),
                spur_set.mask_half_width_bins,
                "on" if (use_catalog and saturated_clusters) else "off",
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
    *,
    shape: "PeakShape | str | None" = None,
    tau_maj_override_us: Optional[float] = None,
    sigma_tau_override_us: Optional[float] = None,
    settings: Optional[StageFitSettings] = None,
    preset: Optional[str] = None,
    jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Stage 5 with BLAS pinned to one thread per process, then persist.

    Every per-window solve in Stage 5 is single-threaded -- the cross-window
    fork pool gets its parallelism from separate worker *processes*, not from
    BLAS threads -- so multithreaded BLAS buys nothing and, left unpinned, has a
    single process spread one solve across every core. The fork-pool walk pins
    its workers, but the main-process work outside that walk does not: the
    in-process (width-1) levels of the walk, and especially the post-fit
    survival prune, VIF collapse, and doublet-adjudication refits, all run NLS
    in this process. On a dense fixture those passes refit hundreds of windows
    and thrash the machine. Pinning the whole call to one BLAS thread covers
    every path -- the forked children inherit the limit -- and is the
    runtime equivalent of an ``OPENBLAS_NUM_THREADS=1`` environment variable
    (which cannot be set here, the backend having initialized at import).
    :func:`threadpoolctl.threadpool_limits` reconfigures the loaded backend at
    call time. See :func:`_fit_peaks_impl` for the parameters and return value.
    """
    with threadpool_limits(limits=1):
        return _fit_peaks_impl(
            file_path,
            shape=shape,
            tau_maj_override_us=tau_maj_override_us,
            sigma_tau_override_us=sigma_tau_override_us,
            settings=settings,
            preset=preset,
            jobs=jobs,
        )


def _fit_peaks_impl(
    file_path: str,
    *,
    shape: "PeakShape | str | None" = None,
    tau_maj_override_us: Optional[float] = None,
    sigma_tau_override_us: Optional[float] = None,
    settings: Optional[StageFitSettings] = None,
    preset: Optional[str] = None,
    jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Stage 5 per-window fitting and persist the result.

    Requires Stage 4 (window assignment) completed (which transitively
    requires Stages 1-3). The fit operates on the active-portion FT
    computed on demand from the persisted FID and the canonical Stage 1
    settings; per-bin noise is measured on the active-FT directly.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    recommended > hard default); pass ``settings=`` to drive the fit from a
    :class:`StageFitSettings` dataclass, or ``preset=NAME_OR_PATH`` to load from
    packaged YAML. They may be combined: a ``settings`` bundle is the
    explicit override (it outranks the persisted record), while a ``preset``
    .yml seeds only the fields neither the explicit layer nor the persisted
    record has fixed (the persisted record outranks the preset, per D11).
    Returns the persistent :class:`SpectrumFit` plus diagnostics; also writes
    ``/stage5_fitting`` and marks the stage done.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file.
    shape : {"lorentzian", "gaussian"} or PeakShape, optional
        Per-line envelope shape. Kept as a first-class convenience argument
        (it selects the lineshape and drives Stage 2b τ-twin selection, not an
        instrument knob): ``"gaussian"`` consumes the Stage 2b τ_G calibration
        in place of the pure-exp twin. Overlays ``settings.shape`` at the
        explicit layer. ``None`` falls through to the resolved
        :class:`StageFitSettings` (preset / persisted / recommended / default).
    tau_maj_override_us, sigma_tau_override_us : float, optional
        Atomic-pair manual override for the Stage 2b tau calibration, kept as
        explicit arguments (they cross a stage boundary -- an A/B escape hatch,
        not a fit knob). When both are supplied (positive), they replace any
        persisted Stage 2b result for this fit -- useful for A/B-ing a
        hand-tuned tau anchor, or for forcing a calibrated tau on fixtures
        where Stage 2b has not been run. Supplying only one of the pair raises
        ``ValueError``. Each overlays the matching ``settings.tau`` field at
        the explicit layer.
    settings : StageFitSettings, optional
        Bundle of Stage 5 knobs; fields left ``None`` fall through the
        resolution chain. Resolves at the explicit override layer (outranks the
        persisted record). May be combined with ``preset``.
    preset : str, optional
        Bare preset name or path to a YAML file carrying a ``stage5:`` block.
        Seeds the preset layer beneath the persisted record; may be combined
        with ``settings``.

    Raises
    ------
    ValueError
        If Stage 4 has not been completed, or if exactly one of the
        ``tau_maj_override_us`` / ``sigma_tau_override_us`` pair is set.
    """
    # --- Resolve parameters via the StageFitSettings chain ------------------
    # A caller-supplied ``settings`` bundle is the explicit override layer;
    # the three kept convenience args (``shape`` + the τ-override pair) overlay
    # onto a copy of it without mutating the caller's object. ``resolve()``
    # walks explicit > persisted > preset > recommended > hard default; the
    # resolved instance is the single source of truth for every downstream
    # call site below. ``_HARD_DEFAULTS`` mirrors each ``DEFAULT_*`` constant
    # in :mod:`ftmwpipeline.fitting`, so an empty call (no overrides, no
    # settings, no persisted layer) reproduces the documented defaults exactly.
    explicit: Optional[StageFitSettings] = (
        settings if settings is not None else StageFitSettings()
    )
    if (
        shape is not None
        or tau_maj_override_us is not None
        or sigma_tau_override_us is not None
    ):
        explicit = copy.deepcopy(explicit)
        assert explicit is not None
        if shape is not None:
            explicit.shape = ShapeSpec.coerce(shape)
        if tau_maj_override_us is not None:
            explicit.tau.tau_maj_override_us = tau_maj_override_us
        if sigma_tau_override_us is not None:
            explicit.tau.sigma_tau_override_us = sigma_tau_override_us

    preset_layer: Optional[StageFitSettings] = None
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
        explicit=explicit,
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
    # parity with the prior ``max(0, int(...))`` behavior.
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
    # The survival floor tracks the Stage 3 promotion cutoff: by default it is
    # that cutoff times ``snr_survival_factor`` (computed below, once the cutoff
    # is loaded), so a stricter detection threshold raises the survival bar in
    # step. An explicit ``snr_survival_floor`` overrides the factor.
    peak_survival_factor_v = _required_float(
        resolved.peak_survival.snr_survival_factor, "peak_survival.snr_survival_factor"
    )
    peak_survival_floor_override = resolved.peak_survival.snr_survival_floor
    vif_collapse_threshold_v = _required_float(
        resolved.peak_survival.vif_collapse_threshold,
        "peak_survival.vif_collapse_threshold",
    )
    collapse_frac_unc_threshold_v = _required_float(
        resolved.peak_survival.collapse_frac_unc_threshold,
        "peak_survival.collapse_frac_unc_threshold",
    )
    collapse_frac_unc_max_sep_res_v = _required_float(
        resolved.peak_survival.collapse_frac_unc_max_separation_res,
        "peak_survival.collapse_frac_unc_max_separation_res",
    )
    collapse_max_sep_res_v = _required_float(
        resolved.peak_survival.collapse_max_separation_res,
        "peak_survival.collapse_max_separation_res",
    )
    sidelobe_max_sep_res_v = _required_float(
        resolved.peak_survival.sidelobe_prune_max_separation_res,
        "peak_survival.sidelobe_prune_max_separation_res",
    )
    degenerate_trial_frac_v = _required_float(
        resolved.peak_survival.degenerate_trial_frac,
        "peak_survival.degenerate_trial_frac",
    )
    degenerate_trial_chi2r_rel_tol_v = _required_float(
        resolved.peak_survival.degenerate_trial_chi2r_rel_tol,
        "peak_survival.degenerate_trial_chi2r_rel_tol",
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
    peak_detection_passes = [
        str((p.properties or {}).get("detection_pass") or "primary") for p in peaks
    ]

    # Resolve the effective survival floor now the Stage 3 promotion cutoff is
    # available: an explicit absolute floor wins; otherwise scale the cutoff by
    # ``snr_survival_factor``. Stage 3 is a hard dependency, so the cutoff is
    # present; fall back to the Stage 3 promotion default only for a legacy file
    # that predates persisting it.
    promotion_cutoff = peaks_loaded.get("promotion_min_snr")
    if promotion_cutoff is None:
        promotion_cutoff = DEFAULT_PROMOTION_MIN_SNR
    if peak_survival_floor_override is not None:
        peak_survival_floor_v = float(peak_survival_floor_override)
    else:
        peak_survival_floor_v = float(promotion_cutoff) * peak_survival_factor_v

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
        if tau_calibration_present(file_path, shape="gaussian"):
            persisted_cal = load_tau_calibration_impl(file_path, shape="gaussian")[
                "tau_calibration"
            ]
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
                "calibrate_tau(shape='gaussian') for an anchored fit.",
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
            tau0_us_v = default_tau0_us(acquisition_us)
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
    # (tau_maj_band, sigma_tau_band) by mapping each window's center
    # frequency to the band whose [freq_lo, freq_hi) contains it. The
    # band-wide ``(tau_maj_us, sigma_tau_us)`` in conservative_kwargs
    # remains the fallback for windows that don't match any band (e.g.
    # window center outside the calibration trim range).
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
            # fallback -- the original strict-raise behavior penalized
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
                center_mhz = 0.5 * (win.freq_range[0] + win.freq_range[1])
                tm, st = resolve_window_tau_anchor(
                    center_mhz, persisted_cal.band_majorities, None, None
                )
                if tm is None:
                    continue
                window_tau_overrides[int(win.window_id)] = (tm, cast(float, st))
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

    # Per-node cleanup, folded into the fit walk's per-node tail: prune sub-floor
    # dust + collapse degenerate sub-resolution pairs the moment a window
    # converges, before the DAG releases its dependents -- so every dependent is
    # fit against an already-cleaned source (no stale frozen-background snapshot)
    # and each cleanup refit inherits the node's exact fit kwargs (per-band tau /
    # baseline / spur) by construction.
    finalize_node = build_finalize_node(
        floor=peak_survival_floor_v,
        enabled=peak_survival_enabled_v,
        vif_threshold=vif_collapse_threshold_v,
        frac_threshold=collapse_frac_unc_threshold_v,
        frac_max_sep_res=collapse_frac_unc_max_sep_res_v,
        max_sep_res=collapse_max_sep_res_v,
        sidelobe_max_sep_res=sidelobe_max_sep_res_v,
        degenerate_trial_frac=degenerate_trial_frac_v,
        degenerate_trial_chi2r_rel_tol=degenerate_trial_chi2r_rel_tol_v,
        res_element_mhz=(1.0 / acquisition_us if acquisition_us > 0 else 0.0),
        sideband=sideband,
        acquisition_us=acquisition_us,
    )

    final_add_v = resolved.rescue.final_add_snr_threshold
    final_add_snr_v = (
        float(final_add_v) if (final_add_v is not None and final_add_v > 0) else None
    )

    plan_outcome = execute_plan(
        plan,
        active_ft,
        rms_for_fit,
        peak_frequencies_mhz,
        peak_detection_passes=peak_detection_passes,
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
        jobs=jobs,
        finalize_node=finalize_node,
        final_add_snr_threshold=final_add_snr_v,
    )
    # A structural replan (merge) rebuilds the plan inside ``execute_plan`` --
    # the survivor's ``freq_range`` becomes the union of the merged windows.
    # Convert and refit against that revised plan, not the pre-replan one, so a
    # merge survivor is not persisted with its original (narrow) range while its
    # peaks span the merged span (which renders peaks outside the window).
    final_plan = plan_outcome.final_plan or plan

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
        # Spur-masking audit: the gated spur catalog this fit consumed.
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
        # later stage replays the gated catalog rather than re-deriving it
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
                "final_add_snr_threshold": final_add_snr_v,
            }
        )
    # Persist the gated spur catalog with the fit: the masked bins are
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
        final_plan,
        sideband=sideband,
        peak_frequencies_mhz=peak_frequencies_mhz,
        peak_detection_passes=peak_detection_passes,
        acquisition_us=acquisition_us,
        parameters=parameters,
        diagnostics=diagnostics,
    )
    # Annotation pass: stamp ``clock_lattice`` on every fitted peak whose
    # molecular frequency lands on the declared instrument lattice.  Pure
    # read -- no effect on the fit statistics.  No-op when no declaration
    # was present (``clock_lattice`` is ``None``).
    annotate_lattice_matches(spectrum_fit, clock_lattice)
    # Stamp ``flat_decay`` on peaks the spur gate kept-but-flagged (ambiguous
    # cluster decay). Pure read -- a review hint, no effect on the fit.
    annotate_flat_decay_matches(spectrum_fit, spur_set)

    # Peak-survival cleanup now runs *in the walk's per-node tail*
    # (``build_finalize_node`` injected into ``execute_plan``): each window's
    # sub-floor dust is pruned and its degenerate sub-resolution pairs collapsed
    # the moment it converges, before the DAG releases its dependents -- so a
    # dependent reads the cleaned source the first time. Here we only (1) roll the
    # per-window cleanup provenance the walk collected into the same
    # ``peak_survival`` / ``vif_collapse`` diagnostics the global post-pass used
    # to write, and (2) inflate each merged line's frequency error by its
    # unresolved-component spread.
    res_element_mhz = 1.0 / acquisition_us if acquisition_us > 0 else 0.0
    if peak_survival_enabled_v:
        pruned_records: List[Dict[str, Any]] = []
        dropped_window_ids: List[int] = []
        collapse_records: List[Dict[str, Any]] = []
        sidelobe_records: List[Dict[str, Any]] = []
        n_iterations = 0
        collapses_by_window: Dict[int, List[Dict[str, Any]]] = {}
        for rec in plan_outcome.cleanup_history:
            pruned_records.extend(rec.get("pruned", []))
            if rec.get("dropped"):
                dropped_window_ids.append(int(rec["window_id"]))
            wcoll = rec.get("collapses", [])
            if wcoll:
                collapse_records.extend(wcoll)
                collapses_by_window.setdefault(int(rec["window_id"]), []).extend(wcoll)
                n_iterations = max(n_iterations, len(wcoll))
            sidelobe_records.extend(rec.get("sidelobes", []))

        # Inflate each merged line's frequency error to sqrt(formal^2 + spread^2)
        # -- the collapsed multiplet's position is honestly known only to within
        # its unresolved-component spread. Per window, from the recorded merges.
        wf_by_id = {int(cast(int, wf.window_id)): wf for wf in spectrum_fit.window_fits}
        for wid, wcoll in collapses_by_window.items():
            wf_target = wf_by_id.get(wid)
            if wf_target is not None:
                _inflate_merged_frequency_errors(wf_target, wcoll)

        # Rebuild the global sorted line list after the inflation (positions are
        # unchanged, but keep the contract that fitted_peaks mirrors the windows).
        all_peaks: List[FittedPeak] = [
            p for wf in spectrum_fit.window_fits for p in wf.fitted_peaks
        ]
        all_peaks.sort(key=lambda p: p.frequency_mhz)
        spectrum_fit.fitted_peaks = all_peaks

        spectrum_fit.diagnostics["peak_survival"] = {
            "snr_floor": float(peak_survival_floor_v),
            "n_pruned": len(pruned_records),
            "pruned": pruned_records,
            "dropped_window_ids": dropped_window_ids,
        }
        spectrum_fit.diagnostics["vif_collapse"] = {
            "vif_threshold": float(vif_collapse_threshold_v),
            "frac_unc_threshold": float(collapse_frac_unc_threshold_v),
            "frac_unc_max_separation_res": float(collapse_frac_unc_max_sep_res_v),
            "max_separation_res": float(collapse_max_sep_res_v),
            "footprint_max_res": float(_COLLAPSE_FOOTPRINT_MAX_RES),
            "res_element_mhz": float(res_element_mhz),
            "n_collapsed_pairs": len(collapse_records),
            "n_iterations": n_iterations,
            "collapses": collapse_records,
        }
        spectrum_fit.diagnostics["sidelobe_prune"] = {
            "max_separation_res": float(sidelobe_max_sep_res_v),
            "res_element_mhz": float(res_element_mhz),
            "n_removed": len(sidelobe_records),
            "removed": sidelobe_records,
        }
        if pruned_records:
            logger.info(
                "Stage 5 peak-survival prune: removed %d sub-floor peaks "
                "(snr < %.2f), %d windows emptied",
                len(pruned_records),
                peak_survival_floor_v,
                len(dropped_window_ids),
            )
        if collapse_records:
            logger.info(
                "Stage 5 VIF collapse: merged %d degenerate sub-resolution "
                "pair(s) (VIF > %.0f within %.2f resolution elements)",
                len(collapse_records),
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

    # Order each window's fitted peaks (and its parameter covariance) by
    # ascending frequency before persisting, so the line list, the report and
    # ``fit show`` tables, and the covariance matrix / correlation heatmap all
    # read in one ascending order. Runs after every refit pass (peak origins are
    # already stamped), so it only reorders -- a window's covariance block stays
    # with its peak.
    for wf in spectrum_fit.window_fits:
        sort_fitting_result_by_frequency(wf)

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
    sideband = Sideband.coerce(fid.sideband)
    base_pp = user_ft.metadata["processing_params"]
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # The fit and the active FT share the ``dt_us * rfft(active)`` amplitude
    # convention, so the model overlay needs no rescale. They also share the
    # ``[0, T]`` active-region phase frame, so the model needs no phase re-roll
    # either (``start_us = 0`` below); the re-roll is only for a persisted-grid
    # overlay, which this is not.
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
        model_amplitude_scale=model_amplitude_scale,
        start_us=0.0,
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
    from ..fitting.active_ft import active_region_bounds
    from ..fitting.peak_model import sideband_sign

    fid = np.asarray(fid_samples, dtype=float)
    # Extract through the SAME helper compute_active_ft uses (searchsorted), so
    # the 2x-padded grid coincides with the native bins (padded[2k] == native[k]).
    # An independent floor/ceil slice differed by one sample, offsetting the grid
    # and -- on 655 -- erasing lines that landed on a mid-bin null.
    start_idx, end_idx = active_region_bounds(fid.size, sample_dt_us, start_us, end_us)
    active = fid[start_idx:end_idx].astype(float, copy=True)
    n_active = active.size
    active -= active.mean()  # match canonical (unconditional) DC removal
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


def compute_display_ft_impl(
    file_path: str, pad_factor: int = _DETAIL_PAD_FACTOR
) -> ComplexFT:
    """Compute the zero-padded, canonical-band DISPLAY FT (Stage 5 report /
    'fit show' magnitude panels), without requiring a persisted Stage 5 fit.
    Depends on Stage 1 (the FID plus canonical FT settings, including any
    trim) only, via :func:`_build_active_ft_inputs`.

    Contrast with the canonical :func:`ftmwpipeline.api.compute_ft`: that FT
    is unpadded and native-length -- the one everything downstream fits and
    scores on. This FT zero-fills the active-region FID slice by
    ``pad_factor`` (display default ``2``, the information limit for a
    magnitude spectrum) purely to interpolate the magnitude curve between the
    native bins; it is display-only and never feeds fitting, noise, or
    chi-squared. The padded grid is then trimmed to ``compute_ft(file_path,
    from_saved_params=True)``'s own frequency band, so this FT differs from
    the canonical FT only in bin density (``pad_factor``x), never extent --
    a consumer deriving a frequency window from this accessor (e.g. a
    catalog/prediction range) sees exactly the band Stage 1 kept, not the
    full FID active region. (The internal, unpadded-and-untrimmed variant
    ``_resolve_detail_bundle`` builds via :func:`_padded_active_display_ft`
    directly is unaffected -- its report/'fit show' consumers mask per-window
    at render time regardless of the shared bundle's extent.) Display
    magnitude is ``abs(spectrum) * amplitude_scale``; ``units_label`` names
    the persisted display units (e.g. ``"µV"``).

    Returns
    -------
    ComplexFT
        ``freq_array`` sorted ascending in molecular frequency, trimmed to
        ``compute_ft(file_path, from_saved_params=True)``'s band at
        ``pad_factor``x its density. ``complex_spectrum`` aligned to it.
        ``metadata`` carries ``amplitude_scale`` (float), ``units_label``
        (str), and ``pad_factor`` (int).
    """
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        _n_padded,
        _acquisition_us,
        user_ft,
        _trim_range,
    ) = _build_active_ft_inputs(file_path)

    freq, spectrum = _padded_active_display_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        pad_factor=pad_factor,
    )

    # Trim to compute_ft's own band (user_ft is already the canonical,
    # trim_range-applied ComplexFT from _build_active_ft_inputs). A tolerance
    # of a quarter padded-bin guards the boundary bins against the two
    # independent FFT paths (FID.preprocess()+compute_fft() here vs.
    # _padded_active_display_ft's inline rfft) landing a ULP apart, without
    # ever admitting a whole extra bin.
    fmin = float(np.min(user_ft.freq_array))
    fmax = float(np.max(user_ft.freq_array))
    tol = float(freq[1] - freq[0]) / 4.0 if freq.size > 1 else 0.0
    band = (freq >= fmin - tol) & (freq <= fmax + tol)
    freq = np.ascontiguousarray(freq[band])
    spectrum = np.ascontiguousarray(spectrum[band])

    amplitude_scale, units_label, _trim_mhz = _load_display_style(file_path)

    return ComplexFT.from_spectrum(
        complex_spectrum=spectrum,
        freq_array=freq,
        metadata={
            "amplitude_scale": amplitude_scale,
            "units_label": units_label,
            "pad_factor": int(pad_factor),
        },
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
        survival_floor=float(
            (bundle.fit.diagnostics or {})
            .get("peak_survival", {})
            .get("snr_floor", DEFAULT_PROMOTION_MIN_SNR * 1.1)
        ),
        vif_collapse_threshold=float(
            (bundle.fit.diagnostics or {})
            .get("vif_collapse", {})
            .get("vif_threshold", 4.0)
        ),
    )


def render_rescue_summary_impl(
    file_path: str,
    window_id: int,
    *,
    bundle: Optional[_DetailBundle] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Any:
    """Render the residual-rescue summary figure for one window.

    Reconstructs the window's residual spectrum from the persisted fit (the same
    ``prepare_window_panels`` prep the ``fit show`` detail figure uses -- exact,
    no re-fit) and combines, in one figure: the window data/model, the final
    residual with each rescue round's nominations overlaid (filled = retained as
    a fitted line, open = rejected), the rescue chi-squared trajectory, and the
    per-round peak budget. Reads the rounds from ``rescue_events``.
    """
    from ..visualization.fit_detail import prepare_window_panels
    from ..visualization.fit_visualization import plot_rescue_summary

    bundle = bundle if bundle is not None else _resolve_detail_bundle(file_path)
    wf = bundle.fit.window_fit(window_id)
    diag = bundle.fit.diagnostics or {}
    panel_data = prepare_window_panels(
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
        spurs=diag.get("gated_spurs"),
        survival_floor=float(
            diag.get("peak_survival", {}).get(
                "snr_floor", DEFAULT_PROMOTION_MIN_SNR * 1.1
            )
        ),
        vif_collapse_threshold=float(
            diag.get("vif_collapse", {}).get("vif_threshold", 4.0)
        ),
    )
    kwargs: Dict[str, Any] = {
        "window_id": window_id,
        "acquisition_us": bundle.acquisition_us,
        "file_stem": bundle.file_stem,
    }
    if figsize is not None:
        kwargs["figsize"] = figsize
    if title is not None:
        kwargs["title"] = title
    return plot_rescue_summary(
        panel_data, list(wf.rescue_events), bundle.sideband, **kwargs
    )


def render_fit_panels_impl(
    file_path: str,
    window_id: int,
    *,
    bundle: Optional[_DetailBundle] = None,
    with_overview: bool = True,
) -> Dict[str, Any]:
    """Render one window's detail as separate, standalone panel figures.

    Returns a dict keyed ``"overview"`` / ``"re"`` / ``"im"`` / ``"mag"`` /
    ``"hist"`` (one :class:`~matplotlib.figure.Figure` per panel) for the HTML
    report's flexbox -- the modular counterpart of
    :func:`render_fit_detail_impl`, sharing its painters. The caller owns
    closing the figures.

    ``with_overview=False`` skips the full-spectrum ``"overview"`` panel; the
    HTML report discards it (it embeds one shared interactive overview), so
    building one per window is wasted work.
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
        include_overview=with_overview,
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
    # List ascending in frequency so the log starts with peak A.
    for pk, lbl in sorted(
        zip(wf.fitted_peaks, labels), key=lambda t: float(t[0].frequency_mhz)
    ):
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
    lives in (not the window-center offset frame ``model_spectrum`` uses).
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
    persisted fit's re-synthesized model FID, transforms both, and overlays them
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
    # bounds, DC mean removal), but unapodized -- we apply our own window.
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
    rescue: bool = False,
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
    With ``rescue`` set, an extra residual-rescue summary figure is produced per
    window (``<stem>_window_<id>_rescue.png``) from the persisted rescue rounds
    (no re-fit): window data/model, the final residual with each round's
    nominations overlaid, the chi-squared trajectory, and the peak budget.
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
        if rescue:
            rfig = render_rescue_summary_impl(
                file_path, wid, bundle=bundle, figsize=figsize
            )
            figures.append(rfig)
            if out_dir is not None:
                rdest = out_dir / f"{bundle.file_stem}_window_{wid:03d}_rescue.png"
                rfig.savefig(str(rdest), dpi=130)
                paths.append(str(rdest))
    return {
        "mode": "detail",
        "window_ids": ids,
        "figures": figures,
        "paths": paths,
        "log": "\n\n".join(reports),
    }
