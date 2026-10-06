"""
Shared implementation for Stage 6: the human review and finalization layer
over the automatic Stage 5 fit.

It owns the candidate ledger (``review show``/``--candidates``), the
user-directed single-window refit verbs (``review edit``/``merge``/``split``/
``accept``), the advisory attention routing and per-window review status, the
anchored decision log, and the consolidated final-products table with its
frequency-calibration budget (``review run``).

The candidate ledger is a pure function of the already-persisted Stage 5 audit
trail (``FittingResult.audit_trail``) and rescue events
(``FittingResult.rescue_events``).  It is derived on demand; no re-fitting and
no writes to the Stage 5 group.

The refit engine (``refit_window_impl``) re-fits a single window using the
production NLS primitives, starting from the persisted peaks as seeds.  It
supports add/remove edits with protected/forbidden immunity so cleanup and
rescue cannot undo human decisions.

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from ..contract import CancelToken, EventCallback, Stage
from ..core.absent import Absent
from ..core.calibration import CalibrationStamp, CalibrationState
from ..core.curation import (
    REFIT_SNAP_TOL_BINS,
    CurationAction,
    Frame,
    PeakUidToken,
    parse_peak_token,
)
from ..core.data_structures import (
    ENGINE_VERSION,
    AttentionReason,
    AuditStep,
    DecisionLogEntry,
    FinalPeak,
    FinalProducts,
    FittedPeak,
    FittingResult,
    FrequencyCalibration,
    LedgerCandidate,
    RescueRoundInfo,
    ReviewParams,
    Sideband,
    SpectrumFit,
    Stage6Review,
    WindowReviewStatus,
    widen_for_unresolved_spread,
)
from ..file_manager import (
    BadSettingError,
    CurationConflictError,
    NotFoundError,
    NotFoundValueError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineFileError,
    StageDependencyError,
    requires_pipeline_file,
)
from ..fitting.active_ft import active_ft_bin_spacing_mhz, peak_uid_from_offset
from ..fitting.peak_model import ModelPeak
from ..fitting.peak_model import molecular_frequency as _molecular_frequency
from ..fitting.peak_model import sideband_sign
from ..fitting.validation import (
    DEFAULT_CHI2R_NOISE_FLOOR,
    DEFAULT_SHAPE_ERROR_KAPPA,
    feature_fwhm,
)
from ..io.fitting_serialization import (
    FitWindowCoverage,
    fit_has_peak_identity,
    load_spectrum_fit_from_hdf5,
    read_fit_diagnostics,
    read_fit_frozen_primaries_by_window,
    read_fit_parameters,
    read_fit_peak_freqs_and_uids_by_window,
    read_fit_peak_frequencies_by_window,
    read_fit_peak_uids_by_window,
    read_fit_window_coverage,
)
from ..io.frequency_calibration_serialization import (
    frequency_calibration_provenance,
    load_frequency_calibration_from_hdf5,
    save_frequency_calibration_to_hdf5,
)
from ..io.provenance import stamp_stage_epoch_in_file
from ..io.stage6_engine_serialization import (
    STAGE6_ENGINE_GROUP,
    EngineWindowKey,
    Stage6EngineState,
    load_stage6_engine_state,
    save_stage6_engine_state,
)
from ..io.stage6_review_serialization import (
    _fit_window_from_dict,
    _fit_window_to_dict,
    final_products_predate_fit_fields,
    fit_declares_clocks,
    load_stage6_review_from_file,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from .absence_rules import (
    clock_lattice_or_absent,
    float_or_absent,
    int_or_absent,
    knockout_absence,
)
from .atomic import atomic_write
from .atomic import exists as pipeline_exists
from .atomic import h5open, resolve
from .empty_window_attention import (
    EMPTY_WINDOW_KINDS,
    empty_window_reasons,
    flagged_lineless_ids,
    superseded_window_ids,
    takeover_points,
)
from .events import StageScope, detached_scope, operation_events
from .stage0_impl import load_fid_from_pipeline_impl
from .stage2_impl import _update_stage_completion

if TYPE_CHECKING:  # annotation-only imports (PEP 563 lazy)
    from ..core.data_structures import FitWindow, Peak, WindowPlan
    from ..core.environment import EnvironmentRecord
    from ..core.stage_fit_settings import ClockSource, StageFitSettings
    from ..fitting.peak_model import PeakShape
    from ..preprocessing.window_planning import Stage6WindowProposal
    from .stage5_impl import Stage5FitContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events and cancellation (dev-docs/CONTRACT_STRATEGY.md §Events and
# cancellation)
# ---------------------------------------------------------------------------

#: The ``review`` stage scope of the Stage 6 long operation running now, or
#: ``None``. Set by :func:`_review_operation` for the duration of one public
#: call, so the engine (:func:`_open_batch`, the action loops, the cascade, the
#: epoch gate, the frame advisory) reports into it without every internal
#: signature carrying it. A call made with no scope set (a ``ReviewSession``
#: step, a replay helper) reports nothing beyond its log lines and is never
#: cancelled.
_REVIEW_SCOPE: ContextVar[Optional[StageScope]] = ContextVar(
    "ftmw_review_scope", default=None
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _review_operation(
    verb: str,
    summary: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
    *,
    wrote: Callable[[Any, Mapping[str, Any]], bool] = lambda result, args: True,
) -> Callable[[_F], _F]:
    """Make a Stage 6 entry point a long operation (``verb``).

    The decorated function takes ``events`` / ``cancel`` keyword arguments
    (declared in its own signature, unused in its body). The wrapper opens the
    ``review`` stage (cancel check, ``StageStarted``, the operation's
    ``environment_drift`` check), runs the body with the scope current
    (:data:`_REVIEW_SCOPE`), and emits ``StageFinished`` with
    ``summary(result, call_args)`` -- the same builder as the verb's
    ``ftmw/run_result@1`` summary. ``wrote(result, call_args)`` says whether
    the call wrote anything (``False`` for a dry run or a preview, which then
    records no completed stage).

    Called from inside another Stage 6 operation without its own events or
    token (a bare-accept apply dispatching to ``review_accept_impl``), it
    reports into the enclosing operation instead of opening a stage.
    """

    def decorate(fn: _F) -> _F:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            call = bound.arguments
            events = call.get("events")
            cancel = call.get("cancel")
            if _REVIEW_SCOPE.get() is not None and events is None and cancel is None:
                with _caller_frame_ids():
                    return fn(*args, **kwargs)
            ops = operation_events(verb, events, cancel)
            path = str(call["file_path"])
            with ops.stage(Stage.REVIEW, verb=verb, file_path=path) as scope:
                token = _REVIEW_SCOPE.set(scope)
                try:
                    # The whole call -- resolving, curating and persisting a
                    # write, an undo's included -- is one atomic write;
                    # StageFinished follows the replace.
                    with atomic_write(path), _caller_frame_ids():
                        result = fn(*args, **kwargs)
                finally:
                    _REVIEW_SCOPE.reset(token)
                scope.finish(summary(result, call), wrote=wrote(result, call))
            return result

        return wrapper  # type: ignore[return-value]

    return decorate


def _review_scope() -> StageScope:
    """The current Stage 6 scope, or a detached one (log lines only)."""
    scope = _REVIEW_SCOPE.get()
    return scope if scope is not None else detached_scope(Stage.REVIEW)


def _check_cancel() -> None:
    """A Stage 6 cancel check point (between windows); no-op with no scope."""
    scope = _REVIEW_SCOPE.get()
    if scope is not None:
        scope.check_cancel()


def _report_window(
    ctx: "_BatchCtx",
    window_id: int,
    *,
    index: int,
    total: int,
    elapsed_s: float,
) -> None:
    """Emit the ``WindowProgress`` (``stage: review``) of one refit window of
    *ctx*'s in-memory fit. No-op when the batch has no scope."""
    scope = ctx.events
    if scope is None:
        return
    wf = next(
        (
            w
            for w in ctx.changeset.spectrum_fit.window_fits
            if w.window_id is not None and int(w.window_id) == int(window_id)
        ),
        None,
    )
    fit_win = ctx.changeset.fit_window_map.get(int(window_id))
    freq_range: Tuple[float, float] = (
        (float(fit_win.freq_range[0]), float(fit_win.freq_range[1]))
        if fit_win is not None and fit_win.freq_range is not None
        else (math.nan, math.nan)
    )
    scope.window_progress(
        phase="initial",
        round=0,
        index=index,
        total=total,
        window_id=int(window_id),
        n_peaks=Absent.NOT_RUN if wf is None else len(wf.fitted_peaks),
        chi2r=Absent.NOT_RUN if wf is None else float(wf.reduced_chi2),
        elapsed_s=elapsed_s,
        dropped=False,
        freq_range=freq_range,
    )


# The ``ftmw/run_result@1`` summaries of the Stage 6 verbs -- each also the
# ``StageFinished.summary`` of its operation (one builder for both).


def review_edit_summary(
    result: "RefitWindowResult",
    add: Sequence[Union[float, str]],
    remove: Sequence[Union[float, str]],
) -> Dict[str, Any]:
    """``review edit``'s summary (``add`` / ``remove`` as the call gave them)."""
    return {
        "window_id": result.window_id,
        "n_peaks_before": result.n_peaks_before,
        "n_peaks_after": result.n_peaks_after,
        "chi2r_before": result.chi2r_before,
        "chi2r_after": result.chi2r_after,
        "converged": result.converged,
        "n_added": len(add),
        "n_removed": len(remove),
        "created_window_mode": result.created_window_mode,
    }


def review_create_summary(result: "CreateWindowResult") -> Dict[str, Any]:
    """``review create``'s summary."""
    lo, hi = result.freq_range
    return {
        "window_id": result.window_id,
        "mode": result.mode,
        "anchor_mhz": result.anchor_mhz,
        "freq_lo_mhz": lo,
        "freq_hi_mhz": hi,
        "n_points": result.n_points,
        "n_contributors": result.n_contributors,
        "n_peaks": result.n_peaks,
    }


def review_accept_summary(
    result: Optional["RefitWindowResult"], window_id: int
) -> Dict[str, Any]:
    """``review accept``'s summary: a bare accept (``result`` ``None``) or an
    accepted candidate."""
    if result is None:
        return {
            "window_id": window_id,
            "provenance": "reviewed",
            "candidate_accepted": False,
        }
    return {
        "window_id": result.window_id,
        "candidate_accepted": True,
        "n_peaks_before": result.n_peaks_before,
        "n_peaks_after": result.n_peaks_after,
        "chi2r_before": result.chi2r_before,
        "chi2r_after": result.chi2r_after,
        "converged": result.converged,
    }


def review_apply_summary(
    result: "CurationApplyResult", dry_run: bool
) -> Dict[str, Any]:
    """``review apply``'s summary."""
    return {
        "dry_run": bool(dry_run),
        "n_actions": len(result.plan),
        "applied": result.applied,
        "n_warnings": len(result.warnings),
        "n_created_windows": len(result.created_windows),
        "n_windows_refit": len(result.windows),
        "base_changed": bool(result.base_changed),
    }


def review_undo_summary(result: "UndoResult", dry_run: bool) -> Dict[str, Any]:
    """``review undo``'s summary."""
    return {
        "dry_run": bool(dry_run),
        "n_removed": len(result.removed),
        "n_replayed": len(result.plan),
        "applied": result.applied,
        "n_geometry_changed": len(result.geometry_changed_window_ids),
    }


def review_preview_summary(result: "ReviewPreviewResult") -> Dict[str, Any]:
    """``review preview``'s ``StageFinished.summary``. The verb has no
    ``run_result`` (it writes nothing and prints its payload); these are the
    payload's counts."""
    return {
        "n_actions": len(result.plan),
        "n_warnings": len(result.warnings),
        "n_created_windows": len(result.created_windows),
        "n_windows": len(result.windows),
    }


def review_run_summary(
    result: "ReviewRunResult", final_products: Optional[FinalProducts]
) -> Dict[str, Any]:
    """``review run``'s summary; ``final_products`` is the file's current
    table (:func:`get_final_products_impl`), or ``None``."""
    summary: Dict[str, Any] = {
        "n_windows": result.n_windows,
        "n_attention": result.n_attention,
        "reason_counts": dict(result.reason_counts),
    }
    if final_products is not None:
        summary.update(
            n_final_peaks=len(final_products.peaks),
            calibration_state=final_products.calibration_state,
            epsilon=final_products.epsilon,
            sigma_floor_khz=final_products.sigma_floor_khz,
        )
    return summary


# ---------------------------------------------------------------------------
# Display-bar defaults
# ---------------------------------------------------------------------------

# Candidates below this residual / rescue SNR bar are hidden by default. The
# bar is a *display* threshold only -- it has no effect on the fit. The ledger
# is a per-window drill-down (what the fit considered and dropped in a window
# under review), so the default is set to keep the per-window count modest on
# the densest fixtures while preserving genuinely-marginal lines well above it.
DEFAULT_DISPLAY_BAR: float = 4.0

# Attention routing flags a window as candidate-bearing only when its strongest
# revivable candidate clears this (higher) evidence threshold. The display bar
# governs which candidates ``review show --candidates`` lists; the attention
# threshold governs which windows the routing surfaces for review -- a
# separate, stiffer cut so the attention list stays actionable (a quiet window
# with only marginal near-misses does not flag). Tunable, orthogonal to the
# accept gates.
DEFAULT_ATTENTION_CANDIDATE_EVIDENCE: float = 10.0

# The attention-routing parameters a status computation uses until a
# ``review run`` records others (``Stage6Review.review_params``).
DEFAULT_REVIEW_PARAMS = ReviewParams(
    bar=DEFAULT_DISPLAY_BAR,
    attention_candidate_evidence=DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    kappa=DEFAULT_SHAPE_ERROR_KAPPA,
    noise_floor=DEFAULT_CHI2R_NOISE_FLOOR,
)

# If a candidate's evidence is within this factor of the accept gate it
# passes the bar even when its raw SNR is below DEFAULT_DISPLAY_BAR.
_NEAR_GATE_FACTOR: float = 10.0

# Deduplicate candidates whose molecular frequencies are within this window,
# expressed as a fraction of the active-FT bin spacing (Requirement 8,
# dev-docs/SCIENCE_STRATEGY.md) rather than a frozen MHz width -- the ledger's
# candidates are Stage 5 audit-trail / rescue-round frequencies, always on the
# active FT. 0.25 bins, not "roughly half a bin": at the reference 13 us
# acquisition the old 0.02 MHz value against the true 79.052 kHz active
# spacing is 0.253 bins, a quarter bin, not a half -- the previous comment's
# "roughly half" was simply wrong, not merely imprecise. Resolved to MHz in
# :func:`derive_candidate_ledger` via ``res_element_mhz`` (the caller's
# resolved active-FT bin spacing); falls back to 0.0 (exact-frequency dedup
# only) when the caller has no resolved spacing, which should not occur on a
# valid persisted Stage 5 fit (acquisition_us is always recorded there).
_DEDUP_TOL_BINS: float = 0.25

# Brightness-scaled shape-error reach is shared with the Stage 5 final
# add-from-convergence pass (one calibration, two consumers); see
# :data:`ftmwpipeline.fitting.validation.SHAPE_ERROR_REACH_KAPPA`. Re-exported
# here so the ledger filter and its tests keep their module-local name.
from ..fitting.validation import SHAPE_ERROR_REACH_KAPPA  # noqa: E402

# spur_adjacent tolerance. A *surviving* fitted line within this many resolution
# elements of a gated clock-harmonic spur center is suspiciously coincident with
# an instrumental node: the spur was masked during the fit, so a line landing on
# top of it is either a real molecule contaminated by the spur or a spur residual
# that escaped the gate -- either way a human should confirm it is molecular. The
# tolerance is line-on-node (not window-overlaps-spur): only the rare coincident
# line flags, keeping the advisory high-precision per the F1 principle. Sized to
# catch a line within ~1 resolution element of the node while a small margin
# absorbs the gated center's drift excursion from the ideal node.
SPUR_ADJACENT_MAX_SEP_RES: float = 1.5


# ---------------------------------------------------------------------------
# Revivable decision labels from the conservative add-loop
# ---------------------------------------------------------------------------

_REVIVABLE_DECISIONS = frozenset({"reject", "tentative"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_molecular(offset_mhz: float, center_mhz: float, sideband: Sideband) -> float:
    """Convert a single baseband offset to molecular MHz."""
    arr = np.array([offset_mhz])
    result = _molecular_frequency(arr, center_mhz, sideband)
    return float(result[0])


def _window_center(fit: FittingResult) -> Optional[float]:
    """Return the molecular center of ``fit``'s window, or ``None``."""
    if fit.window is not None and fit.window.freq_range is not None:
        lo, hi = fit.window.freq_range
        return (lo + hi) / 2.0
    return None


def _auto_merged_window_ids(spectrum_fit: SpectrumFit) -> set:
    """Window ids the end-of-Stage-5 VIF merge touched (from the diagnostic)."""
    return set(_auto_merged_window_freqs(spectrum_fit))


def _auto_merged_window_freqs(spectrum_fit: SpectrumFit) -> Dict[int, List[float]]:
    """Map each auto-merged window id to its merged-peak molecular frequencies
    (from the ``vif_collapse`` provenance), for the ``auto_merged_review`` marker."""
    vc = spectrum_fit.diagnostics.get("vif_collapse", {}) if spectrum_fit else {}
    out: Dict[int, List[float]] = {}
    for c in vc.get("collapses", []):
        wid = c.get("window_id")
        if wid is None or int(wid) < 0:
            continue
        out.setdefault(int(wid), [])
        mf = c.get("merged_frequency_mhz")
        if mf is not None and math.isfinite(float(mf)):
            out[int(wid)].append(float(mf))
    return out


# ---------------------------------------------------------------------------
# Candidate-extraction helpers (one per source type)
# ---------------------------------------------------------------------------


def _audit_step_candidates(
    audit_trail: List[AuditStep],
    center_mhz: float,
    sideband: Sideband,
) -> List[Dict]:
    """Extract revivable candidates from the add-loop audit trail.

    Each revivable step (decision in ``{"reject", "tentative"}``) becomes one
    raw candidate dict with keys: offset_mhz, freq_mhz, evidence, kind,
    reason, site, amplitude.
    """
    out: List[Dict] = []
    for step in audit_trail:
        if step.decision not in _REVIVABLE_DECISIONS:
            continue

        offset = step.candidate_offset_mhz
        freq = _to_molecular(offset, center_mhz, sideband)

        # Evidence: prefer AICc-delta (gate stat), fall back to F-test p.
        if not math.isnan(step.aicc_delta):
            # The add-loop rejected this candidate because aicc_delta >= 0
            # (the K+1 model was not strictly preferred over K).  A *marginal*
            # reject has aicc_delta close to 0 -- the gate was a near-miss and
            # the candidate is potentially revivable.  A *decisive* reject has a
            # large positive aicc_delta -- re-fitting with a user hint is
            # unlikely to change the verdict.  Expose this as ``kind="aicc_delta"``
            # with ``evidence = aicc_delta`` so ``_passes_bar`` can apply the
            # near-gate criterion (small = interesting, large = filter out).
            evidence = step.aicc_delta
            kind = "aicc_delta"
        elif not math.isnan(step.p_value):
            # Rejected by separation / blend-split / nan-AICc path; use F-test p.
            evidence = step.p_value
            kind = "f_p"
        elif not math.isnan(step.chi2_before - step.chi2_after):
            # A degenerate F-test (no residual degrees of freedom, a
            # non-positive chi-squared): its p is undefined (nan), but with
            # both chi-squared values in hand it carries no evidence, and it
            # ranks as p = 1 -- what it was stored as before the statistic
            # became undefined, so the ledger is unchanged.
            evidence = 1.0
            kind = "f_p"
        else:
            evidence = abs(step.chi2_before - step.chi2_after)
            kind = "delta_chi2"

        out.append(
            {
                "offset_mhz": offset,
                "freq_mhz": freq,
                "evidence": evidence,
                "kind": kind,
                "reason": step.reason or step.decision,
                "site": f"add-loop:{step.decision}",
                "amplitude": None,  # not available from audit trail
            }
        )
    return out


def _rescue_round_candidates(
    rescue_events: List[RescueRoundInfo],
    center_mhz: float,
    sideband: Sideband,
) -> List[Dict]:
    """Extract candidates from all rescue rounds for a window."""
    out: List[Dict] = []
    for rnd in rescue_events:
        for cand in rnd.candidates:
            offset = cand.frequency_mhz  # already baseband offset
            freq = _to_molecular(offset, center_mhz, sideband)
            out.append(
                {
                    "offset_mhz": offset,
                    "freq_mhz": freq,
                    "evidence": cand.snr,
                    "kind": "residual_snr",
                    "reason": "rescue-candidate",
                    "site": f"rescue-round:{rnd.round_idx}",
                    "amplitude": cand.magnitude,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Deduplication and bar filter
# ---------------------------------------------------------------------------


def _dedup_and_merge(raw: List[Dict], tol_mhz: float) -> List[Dict]:
    """Merge raw candidates within ``tol_mhz`` of each other.

    Within a tolerance window, keep the candidate with the highest evidence
    and accumulate reasons / sites from all members.
    """
    if not raw:
        return []

    # Sort by molecular frequency for sequential scan.
    sorted_raw = sorted(raw, key=lambda c: c["freq_mhz"])

    groups: List[List[Dict]] = []
    current: List[Dict] = [sorted_raw[0]]

    for item in sorted_raw[1:]:
        if abs(item["freq_mhz"] - current[-1]["freq_mhz"]) <= tol_mhz:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    merged: List[Dict] = []
    for group in groups:
        # Best evidence: prefer residual_snr kind (most interpretable).  For
        # aicc_delta kind, smaller is better (more marginal = more revivable);
        # for all others, larger is better.  Sort priority: residual_snr first,
        # then aicc_delta ascending, then delta_chi2/f_p descending.
        def _evidence_key(c: Dict) -> tuple:
            k = str(c["kind"])
            ev = float(c["evidence"])
            if k == "residual_snr":
                return (0, -ev)  # highest SNR first
            if k == "aicc_delta":
                return (1, ev)  # smallest delta first (most marginal)
            return (2, -ev)  # largest chi2/fp first

        best = min(group, key=_evidence_key)

        reasons = list(dict.fromkeys(c["reason"] for c in group if c["reason"]))
        sites = list(dict.fromkeys(c["site"] for c in group))
        amplitude = next(
            (c["amplitude"] for c in group if c["amplitude"] is not None), None
        )

        merged.append(
            {
                "offset_mhz": best["offset_mhz"],
                "freq_mhz": best["freq_mhz"],
                "evidence": best["evidence"],
                "kind": best["kind"],
                "reason": reasons,
                "sites": sites,
                "amplitude": amplitude,
            }
        )

    return merged


def _passes_bar(candidate: Dict, bar: float) -> bool:
    """Return True if the candidate clears the display bar."""
    ev: float = float(candidate["evidence"])
    kind: str = str(candidate["kind"])

    if kind == "residual_snr":
        # SNR >= bar passes directly.
        return ev >= bar

    if kind == "aicc_delta":
        # ``aicc_delta`` is the AICc *cost* of adding the candidate peak (>= 0
        # for a rejected K+1 model), NOT support for the line: a *large* delta
        # means the add was decisively rejected, a *small* delta a near-gate
        # miss. So a candidate is revivable only when its delta is within
        # ``_NEAR_GATE_FACTOR`` of the gate value (0). This applies to BOTH
        # ``reject`` and ``tentative`` decisions -- a ``tentative`` ("held
        # pending a jointly-significant batch") that never became significant
        # (large delta, high p-value) is not revivable, so it must NOT pass
        # unconditionally (the prior "patience" pass surfaced decisively-
        # rejected tentatives -- e.g. aicc_delta 59 at p=0.98 -- as if they
        # were strong evidence).
        return ev <= _NEAR_GATE_FACTOR

    # For delta_chi2 (chi2-difference fallback): pass when evidence is large.
    if kind == "delta_chi2":
        return ev >= bar

    # f_p: smaller p-value is stronger; pass when p <= 1/bar (heuristic).
    if kind == "f_p":
        return ev <= (1.0 / bar) if bar > 0 else True

    return True


# ---------------------------------------------------------------------------
# Public derivation API
# ---------------------------------------------------------------------------


def derive_candidate_ledger(
    fitting_result: FittingResult,
    *,
    center_mhz: float,
    sideband: Sideband,
    bar: float = DEFAULT_DISPLAY_BAR,
    res_element_mhz: Optional[float] = None,
) -> List[LedgerCandidate]:
    """Derive the candidate ledger for one fit window.

    Walks ``fitting_result.audit_trail`` and ``fitting_result.rescue_events``,
    converts baseband offsets to molecular MHz, deduplicates within
    ``_DEDUP_TOL_BINS`` active-FT bins, applies the display ``bar``, and
    returns a list of :class:`~ftmwpipeline.core.data_structures.LedgerCandidate`
    sorted by molecular frequency.

    Parameters
    ----------
    fitting_result :
        The per-window :class:`~ftmwpipeline.core.data_structures.FittingResult`.
    center_mhz :
        Molecular center of the fit window (midpoint of its ``freq_range``).
    sideband :
        Pipeline sideband (``Sideband.UPPER`` or ``Sideband.LOWER``).
    bar :
        Display SNR / evidence bar.  Candidates below it are dropped.
    res_element_mhz :
        Fourier resolution element (``1 / T_active`` MHz, from
        :func:`~ftmwpipeline.fitting.active_ft.active_ft_bin_spacing_mhz`).
        Drives two things: (1) the dedup tolerance
        (``_DEDUP_TOL_BINS * res_element_mhz``; ``None`` or non-positive falls
        back to 0.0 -- exact-frequency dedup only), and (2), when given, the
        brightness-scaled shape-error filter: a candidate is a lineshape
        sidelobe of a brighter fitted line -- and is excluded -- when
        ``sep_res <= SHAPE_ERROR_REACH_KAPPA * snr / evidence`` for some fitted
        peak (the ``~1/sep_res`` lineshape-error shadow; see
        :data:`SHAPE_ERROR_REACH_KAPPA`).  ``None`` disables the shape filter
        (legacy behavior).

    Returns
    -------
    list of LedgerCandidate
        Sorted by ``frequency_mhz``.
    """
    window_id: int = (
        fitting_result.window_id if fitting_result.window_id is not None else -1
    )

    dedup_tol_mhz = (
        _DEDUP_TOL_BINS * res_element_mhz
        if res_element_mhz is not None and res_element_mhz > 0.0
        else 0.0
    )

    raw: List[Dict] = []
    raw.extend(_audit_step_candidates(fitting_result.audit_trail, center_mhz, sideband))
    raw.extend(
        _rescue_round_candidates(fitting_result.rescue_events, center_mhz, sideband)
    )

    merged = _dedup_and_merge(raw, dedup_tol_mhz)

    # Drop candidates that coincide with an installed fitted peak.  The rescue
    # round records every *detected* candidate, including those the conservative
    # sub-fit then accepted -- those are now real peaks in the line list and are
    # not revivable.  A candidate within a dedup tolerance of any fitted peak is
    # the same sub-resolution feature, so subtract it.
    fitted_freqs = np.array(
        [p.frequency_mhz for p in fitting_result.fitted_peaks], dtype=float
    )
    if fitted_freqs.size:
        not_installed = [
            c
            for c in merged
            if np.min(np.abs(fitted_freqs - c["freq_mhz"])) > dedup_tol_mhz
        ]
    else:
        not_installed = merged

    # Brightness-scaled shape-error filter: drop a candidate that falls inside a
    # brighter fitted line's ~1/sep_res lineshape-error shadow (residual-SNR
    # evidence is dominated by lineshape mismodeling in bright/dense windows).
    # See :data:`SHAPE_ERROR_REACH_KAPPA`.
    if res_element_mhz is not None and res_element_mhz > 0.0 and fitted_freqs.size:
        fitted_snr = np.array(
            [
                (
                    float(p.snr)
                    if p.snr is not None and math.isfinite(float(p.snr))
                    else 0.0
                )
                for p in fitting_result.fitted_peaks
            ],
            dtype=float,
        )

        def _is_shape_error(c: Dict) -> bool:
            evidence = float(c["evidence"])
            if evidence <= 0.0:
                return False
            sep_res = np.abs(fitted_freqs - c["freq_mhz"]) / res_element_mhz
            # A fitted peak's sidelobe reaches sep_res <= kappa * snr / evidence;
            # a candidate inside any peak's reach is that peak's shape error.
            reach_res = SHAPE_ERROR_REACH_KAPPA * fitted_snr / evidence
            return bool((sep_res <= reach_res).any())

        not_installed = [c for c in not_installed if not _is_shape_error(c)]

    filtered = [c for c in not_installed if _passes_bar(c, bar)]

    candidates: List[LedgerCandidate] = []
    for c in sorted(filtered, key=lambda x: x["freq_mhz"]):
        candidates.append(
            LedgerCandidate(
                frequency_mhz=c["freq_mhz"],
                seed_offset_mhz=c["offset_mhz"],
                seed_amplitude=c["amplitude"],
                best_evidence=c["evidence"],
                evidence_kind=c["kind"],
                reasons=c["reason"],
                decision_sites=c["sites"],
                window_id=window_id,
            )
        )

    return candidates


@requires_pipeline_file()
def get_candidate_ledger_impl(
    file_path: Union[Path, str],
    window_id: Optional[int] = None,
    *,
    bar: float = DEFAULT_DISPLAY_BAR,
    spectrum_fit: Optional[SpectrumFit] = None,
    sideband: Optional[Sideband] = None,
) -> List[LedgerCandidate]:
    """Load Stage 5 fit from ``file_path`` and derive the candidate ledger.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    window_id :
        When given, return candidates for that window only.  ``None`` returns
        candidates across all windows.
    bar :
        Display bar passed to :func:`derive_candidate_ledger`.
    spectrum_fit, sideband :
        Pre-resolved fit and sideband (e.g. from a :class:`_DetailBundle`).
        When *both* are supplied, the two HDF5 reloads (the full Stage 5 fit and
        the 750k-point raw FID) are skipped and the ledger is derived directly —
        the report renderer's per-window hot path. When either is ``None`` the
        standalone behavior (self-load from ``file_path``) is unchanged, so the
        CLI / Pipeline / api ledger verbs see no difference.

    Returns
    -------
    list of LedgerCandidate
        Combined across all (or the selected) window(s), sorted by
        ``frequency_mhz``.

    Raises
    ------
    ValueError
        When Stage 5 has not been run yet (no ``stage5_fitting`` group).
    KeyError
        When ``window_id`` is given but not found in the fit.
    """
    path = str(file_path)

    if spectrum_fit is None or sideband is None:
        with h5open(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                raise StageDependencyError(
                    "review",
                    ["stage5_fitting"],
                    Path(str(path)),
                    command="fit run",
                    message="No Stage 5 fit found in this file. Run 'fit run' first.",
                )
            spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        fid = load_fid_from_pipeline_impl(path)
        sideband = Sideband.coerce(fid.sideband)
    else:
        sideband = Sideband.coerce(sideband)

    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    res_element_mhz = (
        active_ft_bin_spacing_mhz(acquisition_us) if acquisition_us > 0.0 else None
    )

    window_fits = spectrum_fit.window_fits
    if window_id is not None:
        window_fits = [wf for wf in window_fits if wf.window_id == window_id]
        if not window_fits:
            raise NotFoundError(
                "window",
                [window_id],
                message=f"window_id={window_id} not found in the Stage 5 fit",
            )

    all_candidates: List[LedgerCandidate] = []
    for wf in window_fits:
        center = _window_center(wf)
        if center is None:
            logger.warning(
                "window_id=%s has no freq_range; skipping ledger derivation",
                wf.window_id,
            )
            continue
        all_candidates.extend(
            derive_candidate_ledger(
                wf,
                center_mhz=center,
                sideband=sideband,
                bar=bar,
                res_element_mhz=res_element_mhz,
            )
        )

    return sorted(all_candidates, key=lambda c: c.frequency_mhz)


# ---------------------------------------------------------------------------
# review rank: on-demand window ranking by any persisted per-window statistic
# ---------------------------------------------------------------------------


@dataclass
class RankedWindow:
    """One window in a :func:`rank_windows_impl` result.

    Attributes
    ----------
    window_id : int
        The fit window id.
    freq_lo, freq_hi : float
        Window frequency range (molecular MHz).
    metric : str
        The ranking metric name.
    value : float
        The metric's value for this window.
    n_peaks : int
        Number of fitted peaks in the window.
    reduced_chi2 : float
        Window reduced chi-squared (context column).
    """

    window_id: int
    freq_lo: float
    freq_hi: float
    metric: str
    value: float
    n_peaks: int
    reduced_chi2: float


# Ranking metric registry: name -> (one-line description, lower_is_worse).
# ``lower_is_worse`` True means the worst windows have the smallest value
# (sorted ascending so the most-actionable lands first); False = larger is worse.
# Every metric is a pure function of the persisted Stage 5 fit. The surface is
# the "surface on demand" half of the attention principle: high-precision flags
# stay small while a user can rank ALL windows by any of these on request.
RANK_METRICS: Dict[str, Tuple[str, bool]] = {
    "min-snr": ("minimum fitted-peak SNR (weakest line in the window)", True),
    "max-vif": ("maximum amplitude VIF (degeneracy / overfit pressure)", False),
    "chi2r": ("window reduced chi-squared (fit quality)", False),
    "candidate-evidence": (
        "strongest revivable candidate residual SNR (possible missed line)",
        False,
    ),
    "edge-distance": (
        "closest fitted-peak-to-window-edge distance, resolution elements",
        True,
    ),
    "spur-proximity": (
        "closest fitted-peak-to-gated-spur distance, resolution elements",
        True,
    ),
    "merged-chi2r": (
        "post-merge chi2r of auto-merged windows (re-split candidates)",
        False,
    ),
}


def _normalize_metric(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _rank_metric_value(
    metric: str,
    wf: FittingResult,
    *,
    res_element_mhz: Optional[float],
    sideband: Sideband,
    spur_centers_mhz: List[float],
    auto_merged_ids: set,
) -> Optional[float]:
    """Compute one ranking metric for one window, or ``None`` to exclude it."""
    from ..fitting.validation import amplitude_vif

    peaks = wf.fitted_peaks
    chi2r = float(getattr(wf, "reduced_chi2", float("nan")))
    res = res_element_mhz if (res_element_mhz and res_element_mhz > 0.0) else None

    if metric == "min-snr":
        snrs = [
            float(p.snr)
            for p in peaks
            if p.snr is not None and math.isfinite(float(p.snr))
        ]
        return min(snrs) if snrs else None

    if metric == "max-vif":
        vifs = [amplitude_vif(p) for p in peaks]
        finite = [v for v in vifs if v is not None and math.isfinite(v)]
        return max(finite) if finite else None

    if metric == "chi2r":
        return chi2r if math.isfinite(chi2r) else None

    if metric == "candidate-evidence":
        center = _window_center(wf)
        if center is None:
            return None
        cands = derive_candidate_ledger(
            wf,
            center_mhz=center,
            sideband=sideband,
            bar=DEFAULT_DISPLAY_BAR,
            res_element_mhz=res_element_mhz,
        )
        ev = [c.best_evidence for c in cands if c.evidence_kind == "residual_snr"]
        return max(ev) if ev else None

    if metric == "edge-distance":
        if not peaks or wf.window is None or wf.window.freq_range is None:
            return None
        lo, hi = wf.window.freq_range
        d_mhz = min(
            min(abs(float(p.frequency_mhz) - lo), abs(hi - float(p.frequency_mhz)))
            for p in peaks
        )
        return d_mhz / res if res else d_mhz

    if metric == "spur-proximity":
        if not peaks or not spur_centers_mhz:
            return None
        centers = np.asarray(spur_centers_mhz, dtype=float)
        d_mhz = min(
            float(np.min(np.abs(centers - float(p.frequency_mhz)))) for p in peaks
        )
        return d_mhz / res if res else d_mhz

    if metric == "merged-chi2r":
        wid = int(wf.window_id) if wf.window_id is not None else -1
        if wid not in auto_merged_ids:
            return None
        return chi2r if math.isfinite(chi2r) else None

    raise ValueError(f"unknown rank metric: {metric!r}")


@requires_pipeline_file()
def rank_windows_impl(
    file_path: Union[Path, str],
    *,
    by: str,
    top: Optional[int] = None,
) -> List[RankedWindow]:
    """Rank fit windows by a persisted per-window statistic (read-only).

    On-demand exploration decoupled from the attention flags: ranks **all**
    windows (not just flagged ones) by ``by`` (one of :data:`RANK_METRICS`),
    worst-first. Windows for which the metric is undefined (e.g. ``min-snr`` on
    an empty window, ``merged-chi2r`` on a window the merge did not touch) are
    omitted.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    by :
        Metric name (``_`` and ``-`` are interchangeable).
    top :
        Return at most this many windows; ``None`` returns all.

    Returns
    -------
    list of RankedWindow
        Worst-first by the metric.

    Raises
    ------
    ValueError
        When Stage 5 has not been run, or ``by`` is not a known metric.
    """
    metric = _normalize_metric(by)
    if metric not in RANK_METRICS:
        valid = ", ".join(sorted(RANK_METRICS))
        raise BadSettingError(
            "by",
            f"one of: {valid}",
            by,
            message=f"unknown rank metric {by!r}; choose one of: {valid}",
        )
    lower_is_worse = RANK_METRICS[metric][1]

    path = str(file_path)
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise StageDependencyError(
                "review",
                ["stage5_fitting"],
                Path(str(path)),
                command="fit run",
                message="No Stage 5 fit found in this file. Run 'fit run' first.",
            )
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    fid = load_fid_from_pipeline_impl(path)
    sideband = Sideband.coerce(fid.sideband)
    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    res_element_mhz = (
        active_ft_bin_spacing_mhz(acquisition_us) if acquisition_us > 0.0 else None
    )
    spur_centers_mhz = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
    ]
    auto_merged_ids = _auto_merged_window_ids(spectrum_fit)

    ranked: List[RankedWindow] = []
    for wf in spectrum_fit.window_fits:
        if wf.window is None or wf.window.freq_range is None:
            continue
        value = _rank_metric_value(
            metric,
            wf,
            res_element_mhz=res_element_mhz,
            sideband=sideband,
            spur_centers_mhz=spur_centers_mhz,
            auto_merged_ids=auto_merged_ids,
        )
        if value is None or not math.isfinite(value):
            continue
        lo, hi = wf.window.freq_range
        chi2r = float(getattr(wf, "reduced_chi2", float("nan")))
        ranked.append(
            RankedWindow(
                window_id=int(wf.window_id) if wf.window_id is not None else -1,
                freq_lo=float(lo),
                freq_hi=float(hi),
                metric=metric,
                value=float(value),
                n_peaks=len(wf.fitted_peaks),
                reduced_chi2=chi2r,
            )
        )

    ranked.sort(key=lambda r: r.value, reverse=not lower_is_worse)
    if top is not None and top > 0:
        ranked = ranked[:top]
    return ranked


# ---------------------------------------------------------------------------
# Frame conversion: raw (stored / fit-frame) <-> calibrated (report-frame)
#
# Every caller-supplied frequency across all three interfaces -- add/remove,
# candidate_freq, merge peak sets, split peak, create anchor, and a curation
# file's frequencies -- carries a ``Frame`` (see ``core.curation.Frame`` for
# the full rationale). ``_resolve_frame`` is called once per verb invocation,
# before the batch opens (like the arity checks below), so a self_calibrated
# file's omitted ``frame`` is refused before the undo baseline is taken --
# same placement, same reason: a refused call must leave the file untouched.
# The converted (raw) frequency is what every applier below ever sees; no
# applier or the batch engine itself knows what frame the caller used.
# ---------------------------------------------------------------------------

_CalibrationStamp = Tuple[str, float, float, float, float, str]
"""``(calibration_state, epsilon, sigma_epsilon, sigma_floor_khz,
probe_freq_mhz, sideband)`` -- :func:`_current_calibration_stamp`'s return
type, named here for readability at the frame-conversion call sites."""


def _resolve_frame(
    path: str, frame: Optional[Frame]
) -> Tuple[Frame, Optional[_CalibrationStamp]]:
    """Resolve an omitted/explicit ``frame`` against the file's calibration.

    Returns ``(resolved_frame, stamp)``; ``stamp`` is
    :func:`_current_calibration_stamp`'s six-tuple (``None`` when the file has
    no FID header to derive one from, in which case the frame is inert).

    Omitting ``frame`` (``None``) resolves to ``\"raw\"`` -- matching today's
    undocumented behavior -- everywhere except a ``self_calibrated`` file,
    where it is refused: that is the one regime where the choice has
    consequences (a calibrated candidate submitted as raw still resolves, and
    to the right peak, but lands ``|f - probe_freq| * eps/(1+eps)`` off --
    under the snap tolerance, over the statistical sigma, invisible in the
    result).
    Passing ``frame=\"raw\"`` explicitly is never refused, on any file.
    """
    stamp = _current_calibration_stamp(path)
    return _resolve_frame_against(stamp, frame, path), stamp


def _frame_mixup_max_offset_khz(path: str, stamp: _CalibrationStamp) -> Optional[float]:
    """Largest raw-vs-calibrated frequency difference over the file's fit.

    A line at molecular frequency ``f`` reads ``|f - probe| * eps/(1+eps)``
    apart in the two frames (the inverse of
    ``f_corr = probe + (f_raw - probe) / (1 + eps)``), so the difference grows
    with the line's distance from the probe; the largest value over the fitted
    windows' span bounds what a frame mix-up costs on this file. ``None`` when
    there is no Stage 5 fit to take the span from.
    """
    epsilon, probe = stamp[1], stamp[4]
    try:
        with h5open(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                return None
            coverage = read_fit_window_coverage(h5f["stage5_fitting"])
    except OSError:
        return None
    bounds = [b for c in coverage if c.freq_range is not None for b in c.freq_range]
    if not bounds:
        return None
    reach = max(abs(b - probe) for b in bounds)
    return reach * abs(epsilon) / (1.0 + epsilon) * 1e3


def _resolve_frame_against(
    stamp: Optional[_CalibrationStamp],
    frame: Optional[Frame],
    path: Optional[str] = None,
) -> Frame:
    """:func:`_resolve_frame`'s rule against an already-read calibration
    stamp -- the one place the rule lives, so a batch resolving many
    actions' frames reads the stamp once. ``path``, when given, lets the
    refusal quote the size of a frame mix-up on this file."""
    cal_state = stamp[0] if stamp is not None else "rb_locked"
    if frame is None:
        if cal_state == "self_calibrated":
            magnitude = ""
            if path is not None and stamp is not None:
                offset_khz = _frame_mixup_max_offset_khz(path, stamp)
                if offset_khz is not None:
                    magnitude = (
                        f" (up to {offset_khz:.1f} kHz over this file's fitted "
                        "range)"
                    )
            raise BadSettingError(
                "frame",
                'one of: "raw", "calibrated" (required on a self_calibrated file)',
                None,
                message="frame is required on a self_calibrated file: pass "
                'frame="raw" or frame="calibrated" explicitly rather than '
                "relying on the default. A calibrated frequency submitted as "
                "raw still resolves to the right peak, but is wrong by "
                f"|f - probe_freq| * eps/(1+eps){magnitude} -- under the snap "
                "tolerance and over the statistical uncertainty, so the "
                "mistake would be silent.",
            )
        return "raw"
    return frame


def _resolve_curation_frame(
    path: str,
    header: "CurationFileHeader",
    frame: Optional[Frame],
) -> Tuple[Frame, Optional[_CalibrationStamp]]:
    """Resolve a curation file's effective frame, combining its optional
    file-level header (A3) with the per-call ``frame`` argument, and refuse a
    header whose stamped epsilon no longer matches the file's current one.

    Precedence: the header alone wins when only the header declares a frame;
    the ``frame`` argument alone wins when only it is given; when both are
    given and DISAGREE, refuse; when neither is given, fall back to
    :func:`_resolve_frame`'s normal rule (default raw, refuse on a
    self_calibrated file when ``frame`` is omitted).

    When the header stamps an epsilon (only reachable with
    ``header.frame == \"calibrated\"`` -- :func:`parse_curation_file` refuses
    any other combination at parse time), it is compared against the file's
    CURRENT epsilon (:func:`_current_calibration_stamp`) -- never against the
    ``frame`` argument, which carries no epsilon of its own. Any disagreement
    is refused rather than silently resolved with either value: a batch
    staged calibrated against one epsilon and applied after the file's
    calibration has moved (e.g. a timebase re-run) would otherwise resolve
    every candidate against the wrong raw frequency, silently -- exactly the
    failure mode the stamp exists to catch.
    """
    if header.frame is not None and frame is not None and header.frame != frame:
        raise BadSettingError(
            "frame",
            f'"{header.frame}" (the frame the curation file header declares)',
            frame,
            message=f"curation file frame disagreement: the file's header declares "
            f'frame="{header.frame}", but frame="{frame}" was passed '
            f"explicitly. Pass a matching frame (or omit it to use the "
            f"file's header), or edit the file's header to match.",
        )

    if header.frame is not None:
        resolved_frame: Frame = header.frame
        stamp = _current_calibration_stamp(path)
    elif frame is not None:
        resolved_frame = frame
        stamp = _current_calibration_stamp(path)
    else:
        resolved_frame, stamp = _resolve_frame(path, None)

    # stamp is None means the target file has no FID header to derive a
    # current calibration from at all (e.g. a hand-built minimal fixture) --
    # no grounds to declare drift, so trust the header rather than refuse
    # against a fabricated "current epsilon" (same stance as
    # ``_final_products_is_stale`` for the analogous A7 staleness check).
    if header.epsilon is not None and stamp is not None:
        current_eps = stamp[1]
        if not math.isclose(header.epsilon, current_eps, rel_tol=1e-6, abs_tol=1e-12):
            raise BadSettingError(
                (
                    "epsilon"
                    if header.epsilon_line is None
                    else _curation_cell(header.epsilon_line, "epsilon")
                ),
                f"the file's current epsilon ({current_eps:.6e})",
                header.epsilon,
                message=f"curation file frame drift: this file was staged "
                f"frame=calibrated at epsilon={header.epsilon:.6e}, but the "
                f"target file's current epsilon is {current_eps:.6e}. The "
                f"calibration has changed since this file was written (e.g. "
                f"a timebase re-run) -- re-stage the curation file against "
                f"the current calibration rather than applying it as-is.",
            )

    return resolved_frame, stamp


_WRITTEN_FREQS: ContextVar[Optional[Dict[float, float]]] = ContextVar(
    "_WRITTEN_FREQS", default=None
)
"""The running curation call's raw -> as-written frequency map (see
:func:`_caller_frame_ids`); ``None`` outside one."""


@contextmanager
def _caller_frame_ids() -> Iterator[None]:
    """Report a ``not_found`` frequency in the frame the caller wrote it in.

    Every caller frequency is converted to raw once, by :func:`_frame_to_raw`,
    which records the pair here when the conversion moved it; the curation
    machinery underneath works and refuses in raw. A ``not_found`` (kind
    ``peak`` or ``window``) escaping the call has its frequency ``ids`` mapped
    back to what the caller wrote, so ``ids`` can be matched against the
    request. A uid (an ``int``) is frame-independent and stays as it is.
    Re-entrant: a nested call reports into the outermost one.
    """
    if _WRITTEN_FREQS.get() is not None:
        yield
        return
    written: Dict[float, float] = {}
    token = _WRITTEN_FREQS.set(written)
    try:
        yield
    except NotFoundError as exc:
        if written and exc.kind in ("peak", "window"):
            exc.ids = [
                written.get(i, i) if isinstance(i, float) else i for i in exc.ids
            ]
        raise
    finally:
        _WRITTEN_FREQS.reset(token)


def _frame_to_raw(
    freq_mhz: float, *, frame: Frame, stamp: Optional[_CalibrationStamp]
) -> float:
    """Convert one caller-supplied frequency to the raw (stored / fit) frame.

    Inverts the baseband-only correction
    (``f_corr = probe + (f_raw - probe) / (1 + eps)``):
    ``f_raw = probe + (f_corr - probe) * (1 + eps)``. Identity when
    ``frame == \"raw\"``, when ``epsilon == 0`` (rb_locked/uncalibrated), or
    when the file carries no calibration to convert against. A conversion
    that moves the value is recorded for :func:`_caller_frame_ids`.
    """
    if frame == "raw" or stamp is None:
        return float(freq_mhz)
    _, epsilon, _, _, probe_freq_mhz, _ = stamp
    if epsilon == 0.0:
        return float(freq_mhz)
    raw = float(probe_freq_mhz + (freq_mhz - probe_freq_mhz) * (1.0 + epsilon))
    written = _WRITTEN_FREQS.get()
    if written is not None:
        written[raw] = float(freq_mhz)
    return raw


def _frame_to_calibrated(
    freq_mhz: float, *, probe_freq_mhz: float, epsilon: float
) -> float:
    """Convert one raw (fit-frame) frequency to the calibrated frame, for
    labeling a returned result (A6). Mirrors :func:`_build_final_products`'s
    correction exactly; identity when ``epsilon == 0``."""
    if epsilon == 0.0:
        return float(freq_mhz)
    return float(probe_freq_mhz + (freq_mhz - probe_freq_mhz) / (1.0 + epsilon))


# ---------------------------------------------------------------------------
# Single-window refit result
# ---------------------------------------------------------------------------


@dataclass
class RefitWindowResult:
    """Outcome of a user-directed single-window refit.

    Attributes
    ----------
    window_id : int
        The window that was refitted.
    n_peaks_before : int
        Number of fitted peaks in the window before the refit.
    n_peaks_after : int
        Number of fitted peaks after the refit.
    chi2r_before : float or Absent
        Reduced chi-squared before the refit; ``Absent.UNDEFINED`` when it is
        not finite (no degrees of freedom, or a fit that did not converge).
    chi2r_after : float or Absent
        Reduced chi-squared after the refit; ``Absent.UNDEFINED`` on the same
        terms.
    fitted_peaks : list of FittedPeak
        The new per-window fitted peaks (already persisted).  Raw / fit-frame
        frequencies -- the frame the fit and the decision log are stored in.
    fitted_peaks_calibrated_mhz : list of float
        The calibrated molecular frequency (MHz) of each entry in
        ``fitted_peaks``, same order and length: ``fitted_peaks[i]`` in the
        raw frame, ``fitted_peaks_calibrated_mhz[i]`` in the calibrated one.
        Equal to the raw value when ``epsilon == 0``.
    calibration_state : str
        ``\"rb_locked\"`` / ``\"self_calibrated\"`` / ``\"uncalibrated\"`` --
        the file's calibration state at the moment of this refit.
    epsilon : float
        The fractional timebase scale error actually applied to build
        ``fitted_peaks_calibrated_mhz`` (``0.0`` unless
        ``calibration_state == \"self_calibrated\"``).
    sigma_epsilon : float
        1-sigma uncertainty on ``epsilon`` (``0.0`` when inapplicable).
    created_window_mode : str or Absent
        W4. ``"created"`` or ``"widened"`` when this refit was the edit half
        of an implied create (:func:`_apply_refit_steps`) -- i.e.
        ``window_id`` did not exist, or was too narrow to hold a fresh
        window, before this call. ``Absent.NOT_RUN`` for an ordinary edit
        into an already-live window, which installed no structure. Present
        whenever any of the ``created_window_*`` fields below is, since a UI
        needs to know *which* structural change happened, not just that one
        did.
    created_window_freq_range : tuple of float or Absent
        The installed (or widened) window's ``(min_mhz, max_mhz)`` extent,
        raw frame -- :attr:`CreateWindowResult.freq_range` from the create
        this edit's window came from. ``Absent.NOT_RUN`` iff
        ``created_window_mode`` is.
    created_window_n_points : int or Absent
        Grid points the window covers. ``Absent.NOT_RUN`` iff
        ``created_window_mode`` is.
    created_window_n_contributors : int or Absent
        Frozen leakage contributors attached to the window.
        ``Absent.NOT_RUN`` iff ``created_window_mode`` is.
    created_window_depends_on : list of int or Absent
        Window ids the window reads frozen leakage from. ``Absent.NOT_RUN``
        iff ``created_window_mode`` is (``[]`` is a legitimate value -- a
        created window with no dependencies -- and distinct from that).
    converged : bool or Absent
        Whether the joint NLS behind this refit converged
        (:attr:`FittingResult.success`); ``Absent.UNDEFINED`` when the window
        has no fitted peaks, since no solver ran. ``False`` means the solver bailed and
        the window kept its seeds verbatim with an infinite chi-squared, so
        every number reported here describes a fit that did not happen --
        ``chi2r_after`` will be wild and the peak positions are the seeds, not
        measurements. The refit still *ran*: this reports the fit's outcome,
        not the operation's, which is why a non-converged refit returns a
        result rather than raising.
    """

    window_id: int
    n_peaks_before: int
    n_peaks_after: int
    chi2r_before: Union[float, Absent]
    chi2r_after: Union[float, Absent]
    fitted_peaks: List[FittedPeak] = field(default_factory=list)
    fitted_peaks_calibrated_mhz: List[float] = field(default_factory=list)
    calibration_state: str = "rb_locked"
    epsilon: float = 0.0
    sigma_epsilon: float = 0.0
    created_window_mode: Union[str, Absent] = Absent.NOT_RUN
    created_window_freq_range: Union[Tuple[float, float], Absent] = Absent.NOT_RUN
    created_window_n_points: Union[int, Absent] = Absent.NOT_RUN
    created_window_n_contributors: Union[int, Absent] = Absent.NOT_RUN
    created_window_depends_on: Union[List[int], Absent] = Absent.NOT_RUN
    converged: Union[bool, Absent] = True
    #: The stages the call invalidated (canonical names, ``rerun_order``):
    #: always ``()``, since Stage 6 invalidates no stage.
    invalidated: Tuple[str, ...] = field(default=(), compare=False)


def _converged_or_absent(wf: Any) -> Union[bool, Absent]:
    """A window fit's ``converged`` as a contract value.

    A window with no fitted peaks ran no solver (the null model reports
    ``success=False`` although its chi-squared is real), so it has no
    convergence outcome: ``Absent.UNDEFINED``. Otherwise ``bool(wf.success)``.
    ``Absent`` is truthy, so readers must test ``is False``.
    """
    if not wf.fitted_peaks:
        return Absent.UNDEFINED
    return bool(wf.success)


def _chi2r_or_absent(value: Union[float, Absent]) -> Union[float, Absent]:
    """A curation result's reduced chi-squared as a contract value: a
    non-finite one (zero degrees of freedom, a fit that did not converge) is
    ``Absent.UNDEFINED``; an ``Absent`` passes through."""
    return value if isinstance(value, Absent) else float_or_absent(value)


def _chi2r_evidence(value: Union[float, Absent]) -> float:
    """The decision-log evidence form of a result's reduced chi-squared.

    Evidence is a free-form float snapshot (the review log's wire form maps a
    non-finite value to ``null`` plus ``_absent``), so an undefined value is
    stored as the ``inf`` the fit reports for it."""
    return float("inf") if isinstance(value, Absent) else float(value)


def _make_refit_result(
    ctx: "_BatchCtx",
    *,
    window_id: int,
    n_peaks_before: int,
    n_peaks_after: int,
    chi2r_before: Union[float, Absent],
    chi2r_after: Union[float, Absent],
    fitted_peaks: List[FittedPeak],
    converged: Union[bool, Absent],
) -> RefitWindowResult:
    """Build one :class:`RefitWindowResult`, labeled with both frames (A6) and
    stamped with the calibration actually applied -- shared by every applier
    that returns one (edit / merge / split / accept-with-candidate), so the
    stamping logic exists in exactly one place."""
    shared = ctx.shared
    calibrated = [
        _frame_to_calibrated(
            float(p.frequency_mhz),
            probe_freq_mhz=shared.fit_ctx.probe_freq_mhz,
            epsilon=shared.epsilon,
        )
        for p in fitted_peaks
    ]
    return RefitWindowResult(
        window_id=window_id,
        n_peaks_before=n_peaks_before,
        n_peaks_after=n_peaks_after,
        chi2r_before=_chi2r_or_absent(chi2r_before),
        chi2r_after=_chi2r_or_absent(chi2r_after),
        fitted_peaks=fitted_peaks,
        fitted_peaks_calibrated_mhz=calibrated,
        calibration_state=shared.calibration_state,
        epsilon=shared.epsilon,
        sigma_epsilon=shared.sigma_epsilon,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Single-window refit engine
#
# The add/remove/anchor snap tolerance every verb below defaults to is
# ``REFIT_SNAP_TOL_BINS`` active-FT bins, imported from ``core.curation`` --
# public, because an integrator that resolved "the peak at f" at a different
# tolerance would disagree with the file about which peak that is. Being a bin
# count it has no MHz value until a file is named, so every curation entry
# point resolves it exactly once, at the public boundary, through
# :func:`refit_snap_tol_mhz_impl`; everything further in takes the resolved
# ``float`` as a required argument. No call takes a tolerance of its own: the
# same edits must read the same way on every replay of the decision log, which
# records no tolerance. That is the same
# discipline the spur gate's ``gate_spurs`` adopted for ``bin_spacing_mhz``:
# a default further in would be an absolute constant coming back.
# ---------------------------------------------------------------------------


def _active_acquisition_us_for_snap(path: str) -> float:
    """The active-region length (us) this file's snap tolerance resolves against.

    Persisted Stage 5 first, the declared Stage 1 window second -- the same
    precedence, and for the same reason, that ``timebase_impl._resolve_clocks``
    uses for the clock declaration. The fitted peaks a curation verb snaps
    *to* live on the grid the fit actually ran on, so once a fit exists its
    own recorded ``acquisition_us`` is the authority even if the declared
    window has since been edited; before a fit exists, the declared window is
    the grid a fit would run on and there is nothing else to prefer.

    Returns ``0.0`` when the file carries neither -- a file with no Stage 0
    FID acquisition at all, which is a hand-built file rather than anything
    the importer produces.
    """
    from ..io.stage_fit_settings_serialization import declared_active_acquisition_us

    fitted = _persisted_acquisition_us(path)
    if fitted > 0.0:
        return fitted
    with h5open(path, "r") as h5f:
        declared = declared_active_acquisition_us(h5f)
    return float(declared) if declared is not None and declared > 0.0 else 0.0


@requires_pipeline_file()
def refit_snap_tol_mhz_impl(file_path: Union[Path, str]) -> float:
    """The Stage 6 curation snap tolerance (MHz) resolved for ``file_path``.

    The public read behind ``api.refit_snap_tol_mhz`` /
    ``Pipeline.refit_snap_tol_mhz`` / ``review snap-tolerance``, and the single
    tolerance every curation verb, batch and replay snaps with (no call takes
    one of its own).

    ``REFIT_SNAP_TOL_BINS / T_active``.  The tolerance is *defined* in
    active-FT bins (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8), so its MHz
    value is a property of one file and an integrator must read it rather than
    resolve the bin count itself -- that is the whole point of publishing this.

    Read-only, and derived at call time from the same active region the verbs
    consult, so it cannot disagree with what a ``review apply`` on this file
    will snap with.  Answerable on a file that has been through nothing but the
    FID import: an unset Stage 1 window means "the whole record", which is a
    perfectly good active region.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    StageDependencyError
        If the file carries no resolvable active region at all -- no persisted
        Stage 5 ``acquisition_us`` *and* no Stage 0 FID duration to fall back
        on.  **This is the documented degrade: a refusal, not a legacy
        absolute value.**  A bin-defined tolerance on a file with no spectrum
        has no honest MHz answer, and inventing one would hand a caller a
        number the pipeline itself would never pair at.  Unreachable for any
        file this package's importer wrote.
    """
    from ..file_manager import StageDependencyError

    path = str(file_path)
    if not pipeline_exists(path):
        raise FileNotFoundError(
            f"Pipeline file not found: {path}\n\n"
            f"To create a new pipeline:\n"
            f"  ftmwpipeline data import {path} path/to/data/"
        )
    acquisition_us = _active_acquisition_us_for_snap(path)
    if acquisition_us <= 0.0:
        raise StageDependencyError(
            "review snap-tolerance", ["stage0_fid_data"], Path(path)
        )
    return REFIT_SNAP_TOL_BINS * active_ft_bin_spacing_mhz(acquisition_us)


def _parse_complex_amplitude(value: object) -> complex:
    """Parse a complex amplitude stored as ``str(complex)`` in JSON.

    ``result_conversion.py`` stores ``frozen.model_peak.amplitude`` (a real
    float) via ``json.dumps(..., default=str)``, which calls ``repr(v)`` on
    non-serializable values.  For a real float the repr is just the float
    string; for an accidentally-complex value it would be ``"(a+bj)"``.
    Both cases are handled here to cover legacy files.
    """
    if isinstance(value, (int, float)):
        return complex(float(value))
    if isinstance(value, complex):
        return value
    # Try eval on the string repr (safe: only complex/float literals enter here).
    try:
        return complex(float(str(value)))
    except (ValueError, TypeError):
        try:
            return complex(str(value))
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot parse complex amplitude from persisted value {value!r}"
            )


def _reconstruct_frozen_peaks(
    fixed_parameters: Dict[str, Dict],
    center_mhz: float,
    sideband: Sideband,
) -> List:  # list of FrozenPeak-like namedtuples from plan_execution
    """Rebuild :class:`~ftmwpipeline.fitting.plan_execution.FrozenPeak` objects
    from the persisted ``fixed_parameters`` dict on a :class:`FittingResult`.

    The persisted format for each entry (key ``frozen_peak_<N>``) is::

        {
            "peak_index":        int,
            "primary_window_id": int,
            "frequency_mhz":     float,   # molecular frequency
            "amplitude":         float,   # model_peak.amplitude (real)
            "phase":             float,   # model_peak.phase (radians)
            "freeze_eligible":   bool,
            "peak_uid":          int,     # point-space identity, or None
                                          # (absent key -> None; never
                                          # re-derived from frequency_mhz)
        }

    The offset in the dependent window's baseband frame is derived from the
    molecular frequency and the window center (same convention as
    :func:`~ftmwpipeline.fitting.plan_execution.evaluate_ancestor_leakage`).
    """
    from ..fitting.plan_execution import FrozenPeak

    frozen: List[FrozenPeak] = []
    s = sideband_sign(sideband)
    for key, entry in fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        freq_mhz = float(entry["frequency_mhz"])
        amplitude = float(_parse_complex_amplitude(entry["amplitude"]).real)
        phase = float(entry.get("phase", 0.0))
        offset_mhz = float(s * (freq_mhz - center_mhz))
        peak_uid_raw = entry.get("peak_uid")
        model_peak = ModelPeak(
            amplitude=amplitude,
            offset_mhz=offset_mhz,
            phase=phase,
            peak_uid=None if peak_uid_raw is None else int(peak_uid_raw),
        )
        frozen.append(
            FrozenPeak(
                peak_index=int(entry["peak_index"]),
                primary_window_id=int(entry["primary_window_id"]),
                model_peak=model_peak,
                frequency_mhz=freq_mhz,
                freeze_eligible=bool(entry.get("freeze_eligible", True)),
                edge_free=False,
            )
        )
    return frozen


def require_splice_compatible_environment(path: str) -> None:
    """Refuse to splice a newly-computed fit into an artifact from another epoch.

    A Stage 6 write that refits is the one place the pipeline writes a
    *partial* result into a finished one: it re-fits the windows its decision
    log edits, and the dependents the cascade reaches, and writes them back
    into a :class:`SpectrumFit` whose other windows were fit earlier. If the
    fitting code changed in between, the result is two different models inside
    one product -- a state no per-file version stamp can express and no report
    can caveat honestly, because the mixture is *within* the artifact.

    That is why this is the one operation class the environment policy blocks,
    and it guards splicing only: the write path (:func:`_curate`) calls it
    only when the write refits something. A write that refits nothing -- a
    bare accept, ``review run``, an undo that drops only bare accepts, or one
    whose only fit changes restore windows to the automatic fit by copy (every
    fit-changing decision undone, or a window's last one) -- keeps the fits it
    finds and is not gated; neither is a preview of one. The first write that
    refits is gated, and once acknowledged it recomputes every window under
    the running code (the engine's context key changed, :func:`_engine_plan`). Extending a file forward is fine (a new
    stage is self-consistently produced by the current environment, and only
    warns); reading is never gated.

    The gate is on :data:`~ftmwpipeline.core.environment.ANALYSIS_EPOCH` alone.
    An unknown epoch on either side -- a Stage 5 fit written before environment
    recording existed -- is treated as compatible: refusing to curate a legacy
    file would punish the user for an upgrade they did not choose.

    The override is a persisted acknowledgement
    (:func:`acknowledge_environment_impl`), not a per-call flag, so a file
    curated across an epoch boundary carries that fact in its own record and
    its reports say so.

    Raises
    ------
    AnalysisEpochMismatchError
        When the persisted Stage 5 fit was produced under a different
        ``ANALYSIS_EPOCH`` and no acknowledgement is recorded.  It subclasses
        both :class:`ValueError` (so pre-existing ``except ValueError`` callers
        are unaffected) and
        :class:`~ftmwpipeline.file_manager.PipelineFileError`, and carries the
        two environments as attributes so a caller can report the mismatch
        without parsing the message.
    """
    from ..core.environment import (
        EnvironmentRecord,
        capture_environment,
        gating_fields_differ,
    )
    from ..file_manager import AnalysisEpochMismatchError
    from ..io.environment_serialization import (
        load_environment_ack,
        load_stage_environments,
    )

    try:
        with h5open(path, "r") as h5f:
            envs = load_stage_environments(h5f)
            ack = load_environment_ack(h5f)
    except OSError:  # pragma: no cover - the caller's own open reports this
        return

    fit_env = envs.get("stage5_fitting")
    if fit_env is None:
        return
    current = capture_environment()
    if not gating_fields_differ(current, fit_env):
        return

    if ack is not None:
        acked = EnvironmentRecord.from_dict(ack.get("acknowledged_environment", {}))
        if not gating_fields_differ(current, acked):
            # The epoch_acknowledged warning: its log line is rendered from the
            # event (events.WARNING_LINES), through the running operation's
            # scope when there is one.
            _review_scope().warn(
                "epoch_acknowledged",
                file_epoch=fit_env.analysis_epoch,
                current_epoch=current.analysis_epoch,
            )
            return

    raise AnalysisEpochMismatchError(path, fit_env, current)


@dataclass
class EnvironmentAckResult:
    """Outcome of :func:`acknowledge_environment_impl`.

    Attributes
    ----------
    acknowledged_environment : EnvironmentRecord
        The environment the acknowledgement was given under. A later epoch
        change re-raises the gate rather than inheriting this acceptance.
    fit_environment : EnvironmentRecord or None
        The environment that produced the persisted Stage 5 fit, or ``None``
        when the file predates environment recording (or has no fit).
    mismatch : bool
        Whether an epoch mismatch actually existed. ``False`` means the
        acknowledgement was unnecessary -- worth saying rather than implying
        a block was lifted that was never in place.
    reason : str
        The free-text note stored with the acknowledgement.
    """

    acknowledged_environment: "EnvironmentRecord"
    fit_environment: Optional["EnvironmentRecord"]
    mismatch: bool
    reason: str = ""


@requires_pipeline_file()
def acknowledge_environment_impl(
    file_path: Union[Path, str], *, reason: str = ""
) -> EnvironmentAckResult:
    """Record acceptance of an analysis-epoch mismatch for Stage 6 editing.

    Unblocks the Stage 6 edit verbs on a file whose Stage 5 fit came from a
    different :data:`~ftmwpipeline.core.environment.ANALYSIS_EPOCH`. The
    acknowledgement names the environment it was given under, so it does not
    silently carry over to a *third* epoch: upgrading again re-raises the gate.
    """
    from ..core.environment import capture_environment, gating_fields_differ
    from ..io.environment_serialization import (
        load_stage_environments,
        save_environment_ack,
    )

    path = str(file_path)
    current = capture_environment()
    with atomic_write(path):
        with h5open(path, "r") as h5f:
            envs = load_stage_environments(h5f)
        fit_env = envs.get("stage5_fitting")
        mismatch = gating_fields_differ(current, fit_env)
        with h5open(path, "a") as h5f:
            save_environment_ack(h5f, current, reason=reason)

    logger.info(
        "Recorded an analysis-environment acknowledgement for %s (epoch %s)",
        path,
        current.analysis_epoch,
    )
    return EnvironmentAckResult(
        acknowledged_environment=current,
        fit_environment=fit_env,
        mismatch=bool(mismatch),
        reason=reason,
    )


def _overlay_created_windows(
    plan: "WindowPlan", created_windows: Sequence["FitWindow"]
) -> "WindowPlan":
    """Return ``plan`` overlaid with ``created_windows`` (neither argument mutated).

    An overlay entry whose ``window_id`` matches a base window **replaces** it
    (the narrow-gap widening case); a fresh id is appended. The result is sorted
    by ascending frequency, matching the plan's own ordering, and carries the
    overlay's inbound dependency edges.

    Pulled out of :func:`effective_window_plan` so the curation engine can
    reapply the same overlay purely in memory as ``create`` actions
    accumulate within one request or one replay -- geometry for the *next*
    create has to see the previous one without a round trip through the file
    between them.
    """
    if not created_windows:
        return plan

    overlay = {int(w.window_id): w for w in created_windows}
    windows = [overlay.get(int(w.window_id), w) for w in plan.windows]
    known = {int(w.window_id) for w in windows}
    windows.extend(w for wid, w in sorted(overlay.items()) if wid not in known)
    windows.sort(key=lambda w: min(w.freq_range))

    # A created window reads the frozen leakage of windows that existed before
    # it: base windows, and earlier created windows. No base window reads a
    # created one, and created ids only grow, so every edge it adds points
    # back at an older window and cannot introduce a cycle. Splice them in and
    # put the new ids last in the fit order.
    edges = list(plan.dependency_edges)
    topo = list(plan.topological_order)
    for wid, w in sorted(overlay.items()):
        for fc in w.fixed_contributors:
            edge = (wid, int(fc.primary_window_id))
            if edge not in edges:
                edges.append(edge)
        if wid not in topo:
            topo.append(wid)

    return replace(
        plan,
        windows=windows,
        dependency_edges=sorted(set(edges)),
        topological_order=topo,
    )


def effective_window_plan(file_path: Union[Path, str]) -> "WindowPlan":
    """The fitted plan overlaid with any Stage-6-created / widened windows.

    The fitted plan is the plan the Stage 5 fit was made on: the Stage 4 plan,
    or, after a structural merge, the revised plan the fit stored
    (:func:`~._internal.fitted_plan.load_fitted_plan`).

    Stage 6 can install a window for a line the automatic detection missed (see
    :func:`create_window_impl`). Those windows live in the Stage 6 review state,
    not in ``/stage4_windows``, so Stage 4's persisted product stays a function
    of Stage 4's own inputs and re-running Stage 4 (which invalidates Stage 6
    anyway) never has to reconcile them. Every Stage 6 code path that resolves a
    ``window_id`` to its geometry goes through here so the base plan and the
    overlay are never read apart.
    """
    from .fitted_plan import load_fitted_plan

    plan: "WindowPlan" = load_fitted_plan(str(file_path)).plan
    review = load_stage6_review_from_file(str(file_path))
    return _overlay_created_windows(plan, review.created_windows)


def _restore_unresolved_spread(fp: FittedPeak, spread: Optional[float]) -> None:
    """Re-apply an auto-merged line's frequency-error widening after a refit.

    *fp* is an output peak of the joint NLS, so its ``frequency_error`` is the
    formal, covariance-only value this refit just computed -- exactly what
    :func:`widen_for_unresolved_spread` expects, and why this is called here
    rather than anywhere a stored error might already be widened. A no-op when
    the seed carried no spread, which is every line that is not a collapsed
    multiplet.
    """
    if spread is None:
        return
    fp.unresolved_spread_mhz = float(spread)
    fp.frequency_error = widen_for_unresolved_spread(fp.frequency_error, spread)


def refit_window_core(
    fit_ctx: "Stage5FitContext",
    fit_win: "FitWindow",
    wf: FittingResult,
    *,
    resolved: "StageFitSettings",
    shape_enum: "PeakShape",
    tau_maj_us: Optional[float],
    sigma_tau_us: Optional[float],
    peak_frequencies_mhz: List[float],
    peak_detection_passes: Optional[Sequence[str]] = None,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    add_seeds: Optional[List[ModelPeak]] = None,
    add_origin: str = "user",
    add_derivations: Optional[Sequence[Optional[int]]] = None,
    snap_tol_mhz: float,
    freeze_inherited: bool = False,
    remove_uids: Sequence[int] = (),
    add_uids: Optional[Sequence[int]] = None,
) -> FittingResult:
    """In-memory single-window refit core (no file I/O, no spur replay, no
    decision recording).

    ``remove_uids`` removes peaks by identity: each entry is the
    ``peak_uid`` of one of the window's peaks (a thawed held line included),
    matched exactly. ``add_uids``, index-aligned with ``add``, makes every
    ``add`` a recorded birth: the frequency is the seed position itself (no
    ledger snap), and the seed carries the recorded uid instead of one minted
    here. Its starting amplitude is the ledger candidate's when the seed
    coincides with one, else read off the data, as for a fresh add. A Stage 6
    decision reaches the core this way, its targets and seeds resolved and
    checked before any fit (:func:`_resolve_edit_steps`), so the
    ``NotFoundValueError``, ``target_outside_window`` and
    ``line_already_fitted`` raised here are invariant guards for it.

    Given a window's already-loaded shared context (``fit_ctx``), its
    :class:`~ftmwpipeline.core.data_structures.FitWindow`, and its persisted
    :class:`~ftmwpipeline.core.data_structures.FittingResult` ``wf`` (the
    source of the frozen background, thawed lines, replayed baseline, and
    starting tau), this applies the caller's ``add`` / ``remove`` edits to the
    persisted peak set and re-converges the result with a single joint NLS:
    materialize the window, reconstruct the frozen background, derive the
    ``fit_window`` kwargs, run :func:`fit_seeds_window_outcome`, and convert to
    a :class:`FittingResult`.  It is NLS-only -- no conservative discovery, no
    rescue, no thaw, no cascade.

    The edit / thaw / origin semantics are identical to and documented on
    :func:`refit_window_impl`, which is now a thin file-bound shell over this
    core (it loads the fit, builds ``fit_ctx`` with the persisted spur
    catalog replayed, calls this core, then persists and records decisions).
    The Stage 5 peak-survival pass routes through it too, holding ``fit_ctx`` /
    the plan / ``resolved`` live from ``fit_peaks_impl``. ``add_origin`` stamps
    the origin of added peaks: ``"user"`` for a user edit (the default, immune
    to later auto-prune/cleanup), ``"auto"`` for an automatic add such as the
    VIF-collapse merged line (a normal fitted peak, not a human decision).
    ``add_derivations`` (index-aligned with ``add``) stamps
    :attr:`FittedPeak.derivation` -- the ``serial`` of the decision that
    created each added peak -- so a consumer reads which peaks the edit created
    rather than pairing peak sets across it. Inherited peaks keep whatever
    derivation they already carried; a peak that merely re-converged is
    identity-preserved and its tag is untouched.

    Every ``add`` frequency must fall inside ``fit_win.freq_range`` *after*
    snapping. A window's fit sees only its own band, so seeding outside it would
    fit against data the window does not cover -- the optimizer would simply pin
    the peak at the nearest edge. That is rejected rather than silently
    accepted (see :func:`create_window_impl` for the frequency-has-no-window
    case).

    The co-fit leakage-wing baseline is warm-started from the persisted
    converged coefficients (see ``initial_baseline_coeffs`` below): the
    baseline stays a free parameter (the model is unchanged), but the joint
    NLS starts at the originating fit's converged baseline rather than
    cold-starting at zero, which would re-open the near-degenerate
    baseline/position valley on wide, low-SNR windows and slide untouched
    peaks. That keeps an identity refit close to the persisted fit, but it is
    not a fixed point: on the 655 baseline a refresh plus identity refit moves
    peaks by up to 4.3e-4 MHz, and on 1019 window 52 a refresh from unchanged
    sources moves a skirt by 0.53 kHz. Whether a window is refit is therefore
    decided structurally (the cascade's reachability), never by comparing a
    refit's output with its input.

    Returns the new per-window :class:`FittingResult`; the caller splices it
    back into the :class:`SpectrumFit` and persists.
    """
    from ..fitting.plan_execution import (
        FrozenPeak,
        fit_seeds_window_outcome,
        materialize_window,
        subtract_frozen_background,
    )
    from ..fitting.result_conversion import window_outcome_to_fitting_result
    from ..fitting.window_fit import derive_window_fit_constraints
    from .stage5_impl import _required_float, _required_int, _required_str

    window_id = int(fit_win.window_id)
    active_ft = fit_ctx.active_ft
    rms_for_fit = fit_ctx.rms_for_fit
    sideband: Sideband = fit_ctx.sideband
    acquisition_us = fit_ctx.acquisition_us
    spur_set = fit_ctx.spur_set

    # --- Materialize the window (grid / data / noise / center) -------------
    _, offset_grid, z_slice, sig_slice, center_mhz = materialize_window(
        fit_win,
        active_ft,
        rms_for_fit,
        sideband=sideband,
    )

    # An accepted Stage 5 thaw leaves nothing to hold out here: the thawed line
    # is its primary's free peak and this window's frozen contributor, an entry
    # of ``fixed_parameters`` like any other, never one of this window's fitted
    # peaks (epoch 6).

    # --- Reconstruct frozen background from persisted fixed_parameters -----
    frozen_peaks = _reconstruct_frozen_peaks(wf.fixed_parameters, center_mhz, sideband)
    # The frozen background uses the window's persisted tau as the dependent
    # tau (exactly the convention in evaluate_ancestor_leakage).
    from .active_ft_support import default_tau0_us

    tau_persisted = float(
        wf.shared_parameters.get("tau_us", {}).get(
            "value", default_tau0_us(acquisition_us)
        )
    )
    # Compute background and data-minus-background.  If thawed peaks are present
    # they will be added to frozen_peaks during the partition step below, after
    # which background and data_minus_bg are recomputed with the full frozen set.
    # We do a preliminary computation here so that win_constraints (which needs
    # data_minus_bg to derive amplitude bounds) has data to work with; it is
    # immediately replaced after the partition step.
    background, data_minus_bg = subtract_frozen_background(
        offset_grid,
        z_slice,
        frozen_peaks,
        tau_persisted,
        acquisition_us,
        shape=shape_enum,
    )

    # --- Per-window spur mask (same derivation as _fit_one_window) ---------
    spur_mask = None
    if spur_set is not None and spur_set:
        lo, hi = fit_win.freq_range
        spur_mask = spur_set.window_mask_spec(lo, hi, center_mhz, sideband)

    # --- Build conservative_kwargs from resolved settings ------------------
    # Mirror the subset of conservative_kwargs that fit_window needs.
    max_decay_v = _required_float(resolved.tau.max_decay_factor, "tau.max_decay_factor")
    n_eff_kind_v = _required_str(
        resolved.conservative.n_eff_kind, "conservative.n_eff_kind"
    )

    # tau0 for this window: use the persisted tau as the starting point so
    # the optimizer begins at the known-good value.  For windows where tau
    # was fixed (fitted=False), the persisted tau IS the tau; for thawed
    # windows it is the converged free tau from the original fit -- in both
    # cases it is the best seed available.
    tau0_for_window = tau_persisted
    # ``tau.fit_tau`` True means what it means to the fit: tau is free where
    # the fit's SNR gate freed it, so the refit follows the per-window decision
    # the original fit made. Only False holds every window's tau fixed. (Unset,
    # on a record written before the hard default, reads the same as True.)
    tau_was_fit = bool(wf.shared_parameters.get("tau_us", {}).get("fitted", True))
    fit_tau_for_window: bool = tau_was_fit and (
        resolved.tau.fit_tau is None or bool(resolved.tau.fit_tau)
    )

    # Build the constraint kwargs forwarded to fit_window.  Only the knobs
    # fit_window actually accepts (not the conservative-loop add-gate ones).
    # When tau_maj_us is None (no Stage 2b calibration) the tau penalty
    # reference is absent; zero the lambda so fit_window's validation passes,
    # mirroring derive_window_fit_constraints's effective_tau_penalty_lambda.
    raw_tau_penalty_lambda = _required_float(
        resolved.tau.tau_penalty_lambda, "tau.tau_penalty_lambda"
    )
    effective_tau_penalty_lambda: float = (
        raw_tau_penalty_lambda if tau_maj_us is not None and tau_maj_us > 0.0 else 0.0
    )
    phase_penalty_lambda_v: float = _required_float(
        resolved.penalties.phase_penalty_lambda, "penalties.phase_penalty_lambda"
    )
    phase_penalty_cutoff_fwhm_v: float = _required_float(
        resolved.penalties.phase_penalty_cutoff_fwhm,
        "penalties.phase_penalty_cutoff_fwhm",
    )
    amp_penalty_lambda_v: float = _required_float(
        resolved.penalties.amp_penalty_lambda, "penalties.amp_penalty_lambda"
    )

    # fw_kwargs is built in two passes:
    #   1. Penalty/tau knobs that do not depend on the data are set here.
    #   2. Amplitude / FWHM bounds (from derive_window_fit_constraints) are
    #      filled after the thawed-peak partition step, where frozen_peaks is
    #      extended and data_minus_bg is recomputed against the final frozen set.
    fw_kwargs: Dict[str, object] = {
        "fit_tau": fit_tau_for_window,
        "max_decay_factor": max_decay_v,
        "tau_penalty_lambda": effective_tau_penalty_lambda,
        "tau_penalty_reference": tau_maj_us,
        "tau_penalty_sigma_us": sigma_tau_us,
        "phase_penalty_lambda": phase_penalty_lambda_v,
        "phase_penalty_cutoff_fwhm": phase_penalty_cutoff_fwhm_v,
        "amp_penalty_lambda": amp_penalty_lambda_v,
        "shape": shape_enum,
    }
    if spur_mask is not None:
        fw_kwargs["spur_mask"] = spur_mask

    # Reproduce the persisted leakage-wing baseline. In production the baseline
    # is evidence-triggered; for a refit we replay exactly what the persisted
    # fit recorded -- co-fit the same order (and offset scale) on the same
    # frozen-bg-subtracted data so the joint NLS lands on matching baseline
    # coefficients. Without this the leakage pedestal stays in the residual and
    # the peaks shift to absorb it.
    _qm = wf.quality_metrics or {}
    if _qm.get("baseline_applied", 0.0) and "baseline_order" in _qm:
        _border = int(_qm["baseline_order"])
        fw_kwargs["baseline_order"] = _border
        _bscale = _qm.get("baseline_offset_scale")
        if _bscale:
            fw_kwargs["baseline_offset_scale"] = float(_bscale)
        # Warm-start the co-fit baseline from the persisted converged
        # coefficients so a no-op refit is a fixed point (otherwise the
        # baseline cold-starts at zero and untouched peaks slide on the
        # near-degenerate baseline/position valley of wide, low-SNR windows).
        _ibc = np.array(
            [
                _qm.get(f"baseline_coeff{k}_re", 0.0)
                + 1j * _qm.get(f"baseline_coeff{k}_im", 0.0)
                for k in range(_border + 1)
            ],
            dtype=np.complex128,
        )
        fw_kwargs["initial_baseline_coeffs"] = _ibc

    # --- Build seed ModelPeak list from persisted fitted_peaks + edits -----
    s = sideband_sign(sideband)

    # Every fitted peak is a free seed. Held peaks -- frozen into the background
    # and re-appended verbatim after the NLS -- come only from the freeze-
    # inherited mode below.
    held_peaks: List[FittedPeak] = []  # verbatim re-append after NLS

    # (seed, origin, derivation, unresolved_spread) tuples, kept together so
    # the "remove" pop and the by-position stamping below cannot fall out of
    # step. The spread rides along for the same reason the derivation does: it
    # is a property of the LINE, and the refit has to hand it back to whichever
    # output peak came from this seed.
    seed_peaks_with_origin: List[
        Tuple[ModelPeak, str, Optional[int], Optional[float]]
    ] = []
    for fp in wf.fitted_peaks:
        offset = float(s * (float(fp.frequency_mhz) - center_mhz))
        # Inherited-seed path: this window's own peaks warm-start the refit
        # from their previously fitted position. This is a propagation, not a
        # birth -- copy the identifier so it survives the refit even though the
        # fitted frequency moves.
        mp = ModelPeak(
            amplitude=float(fp.amplitude),
            offset_mhz=offset,
            phase=float(fp.phase) if fp.phase is not None else 0.0,
            peak_uid=fp.peak_uid,
        )
        seed_peaks_with_origin.append(
            (mp, fp.origin, fp.derivation, fp.unresolved_spread_mhz)
        )

    # Apply "remove" edits: drop seeds closest to remove frequencies.
    forbidden_offsets: List[float] = []
    # Every remove that matches no fitted peak, reported together (not_found
    # names every unknown id of a request at once).
    unmatched_removes: List[float] = []
    unmatched_details: List[str] = []
    for rm_freq in remove:
        rm_offset = float(s * (float(rm_freq) - center_mhz))
        if not seed_peaks_with_origin:
            unmatched_removes.append(float(rm_freq))
            unmatched_details.append(
                f"remove={rm_freq:.4f} MHz: no fitted peaks in window "
                f"{window_id} to remove"
            )
            continue
        closest_idx = min(
            range(len(seed_peaks_with_origin)),
            key=lambda i: abs(seed_peaks_with_origin[i][0].offset_mhz - rm_offset),
        )
        closest_dist_val = abs(
            seed_peaks_with_origin[closest_idx][0].offset_mhz - rm_offset
        )
        if closest_dist_val > snap_tol_mhz:
            closest_mol_freq = (
                center_mhz + s * seed_peaks_with_origin[closest_idx][0].offset_mhz
            )
            unmatched_removes.append(float(rm_freq))
            unmatched_details.append(
                f"remove={rm_freq:.4f} MHz: no fitted peak within "
                f"{snap_tol_mhz:.3f} MHz (closest is at "
                f"{closest_mol_freq:.4f} MHz, "
                f"distance={closest_dist_val:.4f} MHz)"
            )
            continue
        # Record the exact fitted offset as forbidden (rescue must not re-add it).
        removed_mp = seed_peaks_with_origin.pop(closest_idx)[0]
        forbidden_offsets.append(removed_mp.offset_mhz)
    if unmatched_removes:
        raise NotFoundValueError(
            "peak", unmatched_removes, message="; ".join(unmatched_details)
        )

    # Removes by identity: the window's own peak with that uid. Exact, so no
    # tolerance enters.
    unmatched_uids: List[int] = []
    for rm_uid in remove_uids:
        own = next(
            (
                i
                for i, (mp, _, _, _) in enumerate(seed_peaks_with_origin)
                if mp.peak_uid == int(rm_uid)
            ),
            None,
        )
        if own is not None:
            forbidden_offsets.append(seed_peaks_with_origin.pop(own)[0].offset_mhz)
            continue
        unmatched_uids.append(int(rm_uid))
    if unmatched_uids:
        listed = ", ".join(f"peak_uid={u}" for u in unmatched_uids)
        raise NotFoundValueError(
            "peak",
            unmatched_uids,
            message=f"window {window_id} has no fitted peak with {listed} to remove",
        )

    # Recompute background and data_minus_bg with the final frozen_peaks set
    # (which now includes any thawed peaks that were not removed).  This
    # replaces the preliminary computation made before the partition step.
    # Even for windows with no thawed lines the recompute is a no-op (frozen_peaks
    # is unchanged), so we always do it to keep the code simple.
    background, data_minus_bg = subtract_frozen_background(
        offset_grid,
        z_slice,
        frozen_peaks,
        tau_persisted,
        acquisition_us,
        shape=shape_enum,
    )

    # Derive amp_floor, fwhm_mhz, and amp_max from the (now-final) data_minus_bg
    # so fit_window's amplitude and phase penalties are properly bounded against
    # the data the NLS will actually see (with thawed lines already subtracted).
    win_constraints = derive_window_fit_constraints(
        data_minus_bg,
        sig_slice,
        tau0_for_window,
        acquisition_us,
        fit_tau=fit_tau_for_window,
        max_decay_factor=max_decay_v,
        phase_penalty_lambda=phase_penalty_lambda_v,
        amp_penalty_lambda=amp_penalty_lambda_v,
        tau_penalty_lambda=effective_tau_penalty_lambda,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        shape=shape_enum,
    )
    fw_kwargs["amp_floor"] = win_constraints.amp_floor
    fw_kwargs["fwhm_mhz"] = win_constraints.fwhm
    fw_kwargs["amp_max"] = win_constraints.amp_max

    # Apply "add" edits: append new ModelPeak seeds.
    protected_offsets: List[float] = []
    if add_seeds is not None:
        explicit_seeds = list(add_seeds)
    else:
        explicit_seeds = []

    # Derive per-window ledger candidates for snap-to-candidate logic.
    center_for_ledger = center_mhz
    wf_sideband = Sideband.coerce(sideband)

    win_lo, win_hi = fit_win.freq_range
    if win_lo > win_hi:
        win_lo, win_hi = win_hi, win_lo
    # Bin-width slack on the range test: ``freq_range`` names the first and last
    # *grid points* the window covers, so a frequency a hair outside it still
    # lands on an in-window bin. Reject only what is genuinely off the window's
    # data, not what rounds onto its edge bin.
    grid_slack = (
        0.5 * float(np.min(np.abs(np.diff(offset_grid))))
        if offset_grid.size > 1
        else 0.0
    )

    for i, add_freq in enumerate(add):
        add_offset = float(s * (float(add_freq) - center_mhz))
        if explicit_seeds:
            mp = explicit_seeds[i]
            if add_uids is not None:
                mp = replace(mp, peak_uid=int(add_uids[i]))
        elif add_uids is not None:
            # A recorded birth: seeded at its recorded position under its
            # recorded uid. The starting amplitude is the ledger candidate's
            # when the seed IS that candidate (an add the ledger snapped),
            # else the data's at the seed bin.
            ledger = derive_candidate_ledger(
                wf,
                center_mhz=center_for_ledger,
                sideband=wf_sideband,
                bar=0.0,
                res_element_mhz=(
                    active_ft_bin_spacing_mhz(acquisition_us)
                    if acquisition_us > 0.0
                    else None
                ),
            )
            cand = next(
                (c for c in ledger if float(c.frequency_mhz) == float(add_freq)),
                None,
            )
            if cand is not None:
                mp = ModelPeak(
                    amplitude=(
                        float(cand.seed_amplitude)
                        if cand.seed_amplitude is not None
                        else float(np.max(np.abs(data_minus_bg)))
                    ),
                    offset_mhz=add_offset,
                    phase=0.0,
                    peak_uid=int(add_uids[i]),
                )
            else:
                nearest_bin = int(np.argmin(np.abs(offset_grid - add_offset)))
                amp_seed = float(
                    2.0
                    * np.abs(data_minus_bg[nearest_bin])
                    / max(tau0_for_window, 1e-6)
                )
                mp = ModelPeak(
                    amplitude=max(amp_seed, 1e-30),
                    offset_mhz=add_offset,
                    phase=float(np.angle(data_minus_bg[nearest_bin])),
                    peak_uid=int(add_uids[i]),
                )
        else:
            # Snap to the nearest ledger candidate if within tolerance.
            ledger = derive_candidate_ledger(
                wf,
                center_mhz=center_for_ledger,
                sideband=wf_sideband,
                bar=0.0,  # all candidates; the user has decided to add this peak
                res_element_mhz=(
                    active_ft_bin_spacing_mhz(acquisition_us)
                    if acquisition_us > 0.0
                    else None
                ),
            )
            best_cand = None
            best_dist = float("inf")
            for cand in ledger:
                dist = abs(float(cand.frequency_mhz) - float(add_freq))
                if dist < best_dist:
                    best_dist = dist
                    best_cand = cand
            if best_cand is not None and best_dist <= snap_tol_mhz:
                # Reuse the ledger candidate's recorded seed amplitude and offset.
                cand_offset = float(s * (float(best_cand.frequency_mhz) - center_mhz))
                cand_amp = (
                    float(best_cand.seed_amplitude)
                    if best_cand.seed_amplitude is not None
                    else float(np.max(np.abs(data_minus_bg)))
                )
                mp = ModelPeak(
                    amplitude=cand_amp,
                    offset_mhz=cand_offset,
                    phase=0.0,
                    peak_uid=peak_uid_from_offset(
                        cand_offset,
                        center_mhz,
                        sideband,
                        fit_ctx.probe_freq_mhz,
                        fit_ctx.active_ft.n_active,
                        fit_ctx.sample_dt_us,
                    ),
                )
            else:
                # Fresh seed: amplitude from data at nearest bin.
                nearest_bin = int(np.argmin(np.abs(offset_grid - add_offset)))
                amp_seed = float(
                    2.0
                    * np.abs(data_minus_bg[nearest_bin])
                    / max(tau0_for_window, 1e-6)
                )
                mp = ModelPeak(
                    amplitude=max(amp_seed, 1e-30),
                    offset_mhz=add_offset,
                    phase=float(np.angle(data_minus_bg[nearest_bin])),
                    peak_uid=peak_uid_from_offset(
                        add_offset,
                        center_mhz,
                        sideband,
                        fit_ctx.probe_freq_mhz,
                        fit_ctx.active_ft.n_active,
                        fit_ctx.sample_dt_us,
                    ),
                )
        # Range check on the POST-snap seed: the snap (or an explicit
        # ``add_seeds`` entry) is what the NLS actually starts from, so it is
        # what has to lie on this window's data. Rejecting here is consistent
        # with how an unsnappable ``remove`` is already handled, and turns the
        # "resolved the click to the wrong window" case into an error instead of
        # a fit against data the window does not cover.
        seed_freq_mhz = float(center_mhz + s * mp.offset_mhz)
        if not (win_lo - grid_slack <= seed_freq_mhz <= win_hi + grid_slack):
            raise CurationConflictError(
                "target_outside_window",
                [int(window_id)],
                message=f"add={float(add_freq):.4f} MHz resolves to "
                f"{seed_freq_mhz:.4f} MHz, outside window {window_id}'s range "
                f"[{win_lo:.4f}, {win_hi:.4f}] MHz. Name the window that covers "
                f"the frequency, or name no window: an edit whose only target "
                f"is this add then goes to the window that covers it, or "
                f"creates one there if none does ('review create' also makes "
                f"one).",
            )
        # A curated add that lands on an existing seed's identity is a
        # user-input error with a meaningful answer, so it is refused here
        # rather than left to the conversion path, which nudges an *automatic*
        # collision to the nearest free identifier and says nothing. Two lines
        # cannot be born at the same position: asking for one is asking for a
        # second component of a line that is already there, which is what
        # curation-intent inference reads a nearby add as (a split), but two
        # simultaneous adds at one identical, not-yet-fitted frequency have
        # nothing to snap onto yet -- there is no existing fitted peak for
        # either to read as a split of. Note this compares the POST-snap seed
        # -- the ledger snap above can move a seed by up to the snap
        # tolerance, so a frequency clear of every peak on the plot can still
        # resolve onto one.
        if mp.peak_uid is not None:
            clash = next(
                (
                    prior
                    for prior, _, _, _ in seed_peaks_with_origin
                    if prior.peak_uid == mp.peak_uid
                ),
                None,
            )
            if clash is not None:
                clash_freq = float(center_mhz + s * clash.offset_mhz)
                raise CurationConflictError(
                    "line_already_fitted",
                    [int(mp.peak_uid)],
                    message=f"add={float(add_freq):.4f} MHz seeds at "
                    f"{seed_freq_mhz:.4f} MHz, which is the birth position of "
                    f"the line already fitted at {clash_freq:.4f} MHz (both "
                    f"carry peak_uid={mp.peak_uid}). Two lines cannot be born "
                    f"at the same position; apply one add on its own first, "
                    f"then add the second frequency near the resulting fitted "
                    f"line in a later edit -- an add within snap tolerance of "
                    f"a fitted peak that is not itself being removed is read "
                    f"as a split of it.",
                )
        add_derivation = (
            add_derivations[i]
            if add_derivations is not None and i < len(add_derivations)
            else None
        )
        # An added seed carries no spread: it is a new line (or a split
        # product), not the collapsed multiplet the spread describes.
        seed_peaks_with_origin.append((mp, add_origin, add_derivation, None))
        protected_offsets.append(mp.offset_mhz)

    # Extract final seed list in offset order.
    final_seeds = [mp for mp, _, _, _ in seed_peaks_with_origin]
    origin_flags = [orig for _, orig, _, _ in seed_peaks_with_origin]
    derivation_flags = [deriv for _, _, deriv, _ in seed_peaks_with_origin]
    spread_flags = [spread for _, _, _, spread in seed_peaks_with_origin]

    # "Freeze inherited" mode for the VIF-collapse sequential merge: fit ONLY
    # the added (merged) seeds, holding every inherited peak frozen at its
    # persisted value -- added to the frozen background AND re-appended verbatim
    # after the NLS. The sequential
    # collapse loop calls this once per single pair, so a dominant line stays
    # pinned while each new merged line converges. The all-free relaxation that
    # lets a 1e5 giant drag a weak merged line away (655 w124) is deferred to one
    # final relaxed refit at the end of the per-window merge sequence. No-op
    # unless a seed was added.
    n_added = len(add)
    if freeze_inherited and n_added > 0 and len(final_seeds) > n_added:
        inherited_seeds = final_seeds[:-n_added]
        added_seeds = final_seeds[-n_added:]
        added_origins = origin_flags[-n_added:]
        added_derivations = derivation_flags[-n_added:]
        added_spreads = spread_flags[-n_added:]
        used_src: set[int] = set()
        for mp in inherited_seeds:
            freq = center_mhz + s * mp.offset_mhz
            best_idx = -1
            best_d = float("inf")
            for idx, fp in enumerate(wf.fitted_peaks):
                if idx in used_src:
                    continue
                d = abs(float(fp.frequency_mhz) - freq)
                if d < best_d:
                    best_d, best_idx = d, idx
            if best_idx < 0:
                continue
            used_src.add(best_idx)
            src_fp = wf.fitted_peaks[best_idx]
            frozen_peaks.append(
                FrozenPeak(
                    peak_index=(
                        int(src_fp.detection_index)
                        if src_fp.detection_index is not None
                        else -1
                    ),
                    primary_window_id=-1,
                    model_peak=mp,
                    frequency_mhz=freq,
                    freeze_eligible=False,
                    edge_free=False,
                )
            )
            held_peaks.append(src_fp)
        final_seeds = list(added_seeds)
        origin_flags = list(added_origins)
        derivation_flags = list(added_derivations)
        spread_flags = list(added_spreads)
        background, data_minus_bg = subtract_frozen_background(
            offset_grid,
            z_slice,
            frozen_peaks,
            tau_persisted,
            acquisition_us,
            shape=shape_enum,
        )

    # --- Single joint NLS over the seeded set → WindowOutcome ---------------
    # A user refit is NLS-only: it holds the persisted peak set (plus/minus the
    # user's edit) and re-converges it. It deliberately does NOT run residual
    # rescue / discovery -- that pass re-litigates the whole window (it would
    # add brand-new peaks the user did not ask for and break the
    # "changes its own peaks only" contract). The window's discovery already
    # ran during the automatic fit; the persisted peaks are its result.
    # ``protected_offsets`` / ``forbidden_offsets`` are computed above for the
    # edit bookkeeping but no automatic add/prune pass runs here to consult
    # them.
    outcome = fit_seeds_window_outcome(
        offset_grid,
        z_slice,
        sig_slice,
        center_mhz,
        background,
        data_minus_bg,
        frozen_peaks,
        final_seeds,
        tau0_for_window,
        acquisition_us,
        dict(fw_kwargs),
        spur_mask,
        n_eff_kind_v,
        _required_int(resolved.thaw.residual_edge_m, "thaw.residual_edge_m"),
        window_id,
    )

    # --- Convert to FittingResult ------------------------------------------
    new_wf: FittingResult = window_outcome_to_fitting_result(
        outcome,
        fit_win,
        sideband=sideband,
        peak_frequencies_mhz=peak_frequencies_mhz,
        acquisition_us=acquisition_us,
    )

    # Stamp user-origin (and the Stage 6 derivation tag) on peaks the caller
    # added, and on merge / split products, which seed through the same path.
    # A refit is NLS-only -- it neither adds nor drops peaks -- and
    # ``window_outcome_to_fitting_result`` preserves the fit's peak order, so
    # the i-th output peak is the i-th seed: assign by POSITION.  Matching by
    # frequency is unsafe here because a merge / split product can converge well
    # beyond ``snap_tol_mhz`` from its seed.  Fall back to nearest-frequency
    # matching only if the counts ever diverge (they should not on the NLS-only
    # path).
    #
    # An inherited seed carries its own prior ``derivation`` forward: the refit
    # re-converged it but did not change its identity, so the tag still names
    # the decision that last did.
    #
    # An inherited seed's ``unresolved_spread_mhz`` rides the same path, and
    # for a reason worth spelling out: ``window_outcome_to_fitting_result``
    # has just written a FORMAL, covariance-only ``frequency_error`` for every
    # output peak. For a collapsed multiplet that error is a claim the data do
    # not support -- the line's position is known only to within the spread of
    # the components it absorbed -- so the widening Stage 5 applied has to be
    # re-applied here, against this refit's own formal error. Without it the
    # widening would silently disappear the first time the window was refit,
    # including a cascade refit the user never asked for.
    if len(new_wf.fitted_peaks) == len(origin_flags):
        for fp, orig, deriv, spread in zip(
            new_wf.fitted_peaks, origin_flags, derivation_flags, spread_flags
        ):
            if orig == "user":
                fp.origin = "user"
            if deriv is not None:
                fp.derivation = int(deriv)
            _restore_unresolved_spread(fp, spread)
    else:
        user_seeds = [
            (float(center_mhz + s * mp.offset_mhz), orig, deriv, spread)
            for mp, orig, deriv, spread in zip(
                final_seeds, origin_flags, derivation_flags, spread_flags
            )
            if orig == "user" or deriv is not None or spread is not None
        ]
        for fp in new_wf.fitted_peaks:
            for uf, orig, deriv, spread in user_seeds:
                if abs(float(fp.frequency_mhz) - uf) <= snap_tol_mhz:
                    if orig == "user":
                        fp.origin = "user"
                    if deriv is not None:
                        fp.derivation = int(deriv)
                    _restore_unresolved_spread(fp, spread)
                    break

    # --- Re-insert held peaks verbatim --------------------------------------
    # Held peaks were kept out of the NLS and frozen into the background so
    # the window's free peaks converged correctly against them.  Now re-attach
    # them to the output UNCHANGED (same frequency / amplitude / phase / errors
    # / origin as the persisted FittedPeak).  Their model contribution is
    # already accounted for in ``full_fitted`` / ``full_residual`` (they were
    # part of ``background``), so χ²ᵣ in the result is consistent.
    if held_peaks:
        new_wf.fitted_peaks = sorted(
            new_wf.fitted_peaks + held_peaks,
            key=lambda fp2: float(fp2.frequency_mhz),
        )
        # The NLS ran with only the non-held peaks free; the covariance only
        # covers those K_free params.  After re-inserting the held peaks the
        # fitted_peaks list grows, so the covariance labels no longer match the
        # full peak count.  Clear it rather than persist a partial / mislabeled
        # matrix — the per-peak amplitude_error / frequency_error / phase_error
        # fields already carry the per-parameter uncertainties.
        new_wf.covariance = None
        new_wf.covariance_param_labels = None
        logger.debug(
            "Stage 6 refit window %d: re-inserted %d held peak(s) verbatim",
            window_id,
            len(held_peaks),
        )

    # Carry the construction provenance forward. A refit replays/edits the
    # window rather than rebuilding it, so the original conservative add-one
    # ``audit_trail`` and ``rescue_events`` stay the truthful record of how the
    # peak set arose; the joint-refit core would otherwise leave them empty
    # (e.g. an auto-merged VIF-collapse window, which is why such windows showed
    # no add-one history in the report).
    new_wf.audit_trail = list(wf.audit_trail or [])
    new_wf.rescue_events = list(getattr(wf, "rescue_events", []) or [])

    # ``freeze_inherited`` parked the window's OWN inherited peaks in the frozen
    # background to hold them during the merged-line fit (and re-appended them to
    # ``fitted_peaks`` above). They must NOT persist as fixed contributors -- a
    # later refit would reconstruct them as background AND fit them as peaks
    # (double-count). Restore the original contributor set; the inherited peaks
    # live only in ``fitted_peaks``.
    if freeze_inherited and len(add) > 0:
        new_wf.fixed_parameters = dict(wf.fixed_parameters or {})

    # Annotate the refit's lines against the fit's clock lattice, as Stage 5
    # does, so a refit line's ``clock_lattice`` is a tested result (a match or
    # off-lattice) rather than an untested blank. Informational only.
    lattice = fit_ctx.clock_lattice
    if lattice is not None:
        for pk in new_wf.fitted_peaks:
            point = lattice.match(pk.frequency_mhz)
            pk.clock_lattice = None if point is None else point.identity

    return new_wf


# ---------------------------------------------------------------------------
# Contributor-edit cascade: propagate a Stage-6 edit into dependent windows.
#
# A strong line fit in its own window W contributes its frozen leakage skirt to
# every dependent window D as a FixedContributor. When W is edited during Stage 6
# curation, D keeps the skirt it was given at fit time -- a stale model of W. The
# cascade re-evaluates each dependent's frozen background from its sources' CURRENT
# fits (window-level resolution: "all source peaks >= min_freeze_snr", the same
# rule the in-walk path uses via ``evaluate_ancestor_leakage``) and re-fits it.
# Which windows are a dependent's sources is read from the plan, never from the
# fits being curated (``_base_cascade_sources``, ``_cascade_sources``).
#
# It is internal to the edit verbs -- not a user verb. The persisted truth stays
# (automatic baseline, decision_log); the curated fit is derived by replaying the
# log, and each replayed edit fires this cascade, so reversibility and "undo all ->
# the automatic fit" hold by construction (see ``review_undo_impl``). The cascade
# refits are NOT logged as separate decisions: they are a deterministic function of
# the edit.
#
# Scope note: the gate experiment found propagation is <<sigma_f for every realistic
# edit class (split/merge/satellite are far-field-invariant in the source's total
# power and centroid); the cascade is a correctness/honesty fix with rare practical
# bite. The DAG is wide-shallow, so a serial closure re-walk is adequate.
# ---------------------------------------------------------------------------


def _non_edge_free_primaries(fit_win: Optional["FitWindow"]) -> Optional[set]:
    """Source window ids a dependent reads via a **cascade-bearing** edge.

    An ``edge_free`` contributor reads its frozen ``(amplitude, phase)`` from the
    active FT (the data), not from its primary's fit, so it is cascade-immune and
    must be preserved across an edit (design §1). Returns the set of primaries that
    are *not* edge-free for this window, or ``None`` when the plan window is
    unavailable (caller then treats every primary as a dependency).
    """
    if fit_win is None:
        return None
    return {
        int(c.primary_window_id) for c in fit_win.fixed_contributors if not c.edge_free
    }


def _base_cascade_sources(
    base_plan: "WindowPlan",
    baseline_frozen_primaries: Mapping[int, Sequence[int]],
    unavailable_window_ids: Collection[int] = (),
) -> Dict[int, Tuple[int, ...]]:
    """``E_base``: every baseline-live window's cascade sources, in refresh order.

    *baseline_frozen_primaries* maps each window the automatic fit (the undo
    baseline) holds to its frozen contributors' primaries, in order of first
    appearance (``read_fit_frozen_primaries_by_window``); its keys are the
    baseline-live windows. A window's sources are the non-edge-free
    ``fixed_contributors`` primaries of its FitWindow in the fitted plan,
    restricted to baseline-live windows: plan-derived, so an edit that leaves a
    source with no line above ``min_freeze_snr`` does not drop the edge, and a
    later edit that restores one is propagated again. The order is the baseline
    fit's first appearance, then any further plan sources in
    ``fixed_contributors`` order: the frozen background is a floating-point
    sum, and this keeps a one-batch replay's sum unchanged.

    A window whose fitted geometry the file does not hold (*unavailable*), or
    that has no plan entry, keeps the fit-derived rule: every frozen primary
    of its baseline fit is a source, edge-free or not, since no plan window
    says which are. Reaching one is refused (:func:`_refuse_unavailable_fit_plan`).

    A function of the fitted plan and the baseline only, never of a curated
    fit; no Stage 6 write changes either within a lineage.
    """
    live = {int(w) for w in baseline_frozen_primaries}
    blocked = {int(w) for w in unavailable_window_ids}
    plan_map = {int(w.window_id): w for w in base_plan.windows}
    out: Dict[int, Tuple[int, ...]] = {}
    for wid, frozen in baseline_frozen_primaries.items():
        w = int(wid)
        fit_win = None if w in blocked else plan_map.get(w)
        if fit_win is None:
            out[w] = tuple(
                dict.fromkeys(int(p) for p in frozen if int(p) in live and int(p) != w)
            )
            continue
        plan_sources = [
            int(c.primary_window_id)
            for c in fit_win.fixed_contributors
            if not c.edge_free
            and int(c.primary_window_id) in live
            and int(c.primary_window_id) != w
        ]
        allowed = set(plan_sources)
        ordered = dict.fromkeys(int(p) for p in frozen if int(p) in allowed)
        ordered.update(dict.fromkeys(plan_sources))
        out[w] = tuple(ordered)
    return out


def _cascade_sources(
    base_sources: Mapping[int, Tuple[int, ...]],
    created_windows: Sequence["FitWindow"],
) -> Dict[int, Tuple[int, ...]]:
    """The cascade graph ``G = E_base ∪ E_create`` as each window's ordered
    source list.

    *base_sources* is :func:`_base_cascade_sources` (its keys are the
    baseline-live windows); *created_windows* is the review's overlay. A
    created window's sources are its FitWindow's non-edge-free
    ``fixed_contributors`` primaries, deduplicated in that order and
    restricted to windows live when it was made: baseline-live windows and
    earlier (lower-id) created ones. An overlay entry with a base id (a
    widening) carries its window's base contributors verbatim and adds no
    edge. No edge enters a base window from a created one, and created ids
    only grow, so the graph stays acyclic.
    """
    sources: Dict[int, Tuple[int, ...]] = dict(base_sources)
    created = sorted(
        (int(w.window_id), w)
        for w in created_windows
        if int(w.window_id) not in base_sources
    )
    created_ids = {wid for wid, _ in created}
    for wid, fit_win in created:
        sources[wid] = _created_window_sources(fit_win, base_sources, created_ids)
    return sources


def _created_window_sources(
    fit_win: "FitWindow",
    base_live: Collection[int],
    created_ids: Collection[int],
) -> Tuple[int, ...]:
    """A created window's cascade sources, in refresh order: its FitWindow's
    non-edge-free ``fixed_contributors`` primaries, deduplicated in that
    order, among the baseline-live windows *base_live* and the created
    windows *created_ids* older (lower-id) than it. A function of the window's
    plan entry alone, so the window's starting skirt
    (:func:`_created_window_seed`) and every later cascade refresh read the
    same list."""
    wid = int(fit_win.window_id)
    return tuple(
        dict.fromkeys(
            int(c.primary_window_id)
            for c in fit_win.fixed_contributors
            if not c.edge_free
            and (
                int(c.primary_window_id) in base_live
                or (
                    int(c.primary_window_id) in created_ids
                    and int(c.primary_window_id) < wid
                )
            )
        )
    )


def _cascade_succs(sources: Mapping[int, Sequence[int]]) -> Dict[int, set]:
    """Reverse dependency map ``source -> {dependent}`` of a source-list graph."""
    succs: Dict[int, set] = {int(w): set() for w in sources}
    for d, ps in sources.items():
        for p in ps:
            succs.setdefault(int(p), set()).add(int(d))
    return succs


def _cascade_ancestors(wid: int, sources: Mapping[int, Sequence[int]]) -> Set[int]:
    """Every strict ancestor of *wid* in the source-list graph *sources*."""
    out: Set[int] = set()
    stack = [int(p) for p in sources.get(int(wid), ())]
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(int(q) for q in sources.get(p, ()))
    return out


def _cascade_closure(edited_wids: Sequence[int], succs: Dict[int, set]) -> set:
    """Transitive descendants of ``edited_wids`` over ``succs`` (dependents only)."""
    from collections import deque

    closure: set = set()
    dq: "deque[int]" = deque()
    for w in edited_wids:
        dq.extend(succs.get(int(w), ()))
    while dq:
        x = dq.popleft()
        if x in closure:
            continue
        closure.add(x)
        dq.extend(succs.get(x, ()))
    closure -= {int(w) for w in edited_wids}
    return closure


def _cascade_closure_set(
    edited_wids: Sequence[int],
    sources: Mapping[int, Sequence[int]],
    reached: Collection[int] = (),
) -> Set[int]:
    """The windows a cascade from *edited_wids* refits: their dependents in
    the source-list graph *sources*, plus *reached*. Structural, so a request
    can be checked against it before anything is fit.

    :func:`_cascade_closure` strips the whole ``edited_wids`` set from its
    result, so when one cascade covers several DIRECTLY edited windows (a
    batch's single combined cascade), a window that is both directly edited
    AND downstream of ANOTHER directly-edited window would otherwise never get
    its frozen background refreshed from its sibling's new state. Any edited
    window reachable from the *rest* of the edited set is added back. A lone
    edit's ``edited_wids`` has nothing left after removing itself, so this is
    a no-op for one interactive edit."""
    succs = _cascade_succs(sources)
    closure = _cascade_closure(edited_wids, succs)
    edited_set = {int(w) for w in edited_wids}
    for w in edited_set:
        if w in _cascade_closure(sorted(edited_set - {w}), succs):
            closure.add(w)
    closure |= {int(w) for w in reached}
    return closure


def _cascade_topo(nodes: set, preds: Dict[int, set]) -> List[int]:
    """Kahn topological order of ``nodes`` (predecessors within the set gate).

    The cascade graph is acyclic by construction: its base edges are the
    Stage 4/5 plan's strength-oriented contributor edges (a total order), and a
    created window's edges come only from windows that existed before it. A
    cycle is therefore an internal invariant violation, raised rather than
    walked in some arbitrary order.
    """
    nodes = set(nodes)
    placed: set = set()
    out: List[int] = []
    remaining = sorted(nodes)
    while remaining:
        ready = [w for w in remaining if (preds.get(w, set()) & nodes) <= placed]
        if not ready:
            raise RuntimeError(
                f"internal: the Stage 6 cascade graph has a cycle among windows "
                f"{remaining}; its edges are acyclic by construction"
            )
        out.extend(ready)
        placed.update(ready)
        rs = set(ready)
        remaining = [w for w in remaining if w not in rs]
    return out


def _resolve_refit_window_tau(
    fit_win: "FitWindow",
    resolved: "StageFitSettings",
    persisted_cal: object,
    tau_maj_us: Optional[float],
    sigma_tau_us: Optional[float],
    tau_source: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Per-band tau anchor for one window (the refit replay of the production
    per-band penalty anchor; the global ``tau_maj`` would pull tau off the fit's
    optimum -- the recurring tau_maj-vs-per-band bug). A no-op under an explicit
    override or without a calibration."""
    if (
        bool(resolved.tau.per_band_tau)
        and tau_source != "override"
        and persisted_cal is not None
    ):
        from .stage5_impl import resolve_window_range_tau_anchor

        return resolve_window_range_tau_anchor(
            fit_win.freq_range,
            persisted_cal.band_majorities,  # type: ignore[attr-defined]
            tau_maj_us,
            sigma_tau_us,
        )
    return tau_maj_us, sigma_tau_us


def _refresh_frozen_from_sources(
    wf: FittingResult,
    sources: Sequence[int],
    fit_window_map: Dict[int, "FitWindow"],
    fit_map: Mapping[int, FittingResult],
    min_freeze_snr: float,
) -> None:
    """Rebuild ``wf``'s non-edge-free frozen contributors from its *sources*'
    **current** fitted peaks clearing ``min_freeze_snr`` (window-level
    resolution).

    This is the one piece the cascade adds: a plain refit reconstructs the frozen
    background from ``wf``'s own persisted snapshot (a fixed point -- a no-op), so a
    dependent only tracks its source's edit once its background is re-read from the
    source's live fit. *sources* is the window's source list in the cascade graph
    (:func:`_cascade_sources`), in its canonical order, so a source that an
    earlier edit left with no line above threshold, and which ``wf`` therefore
    holds no entry from, contributes again once its fit has one. Edge-free
    contributors (read from data, cascade-immune) and any non-``frozen_peak_*``
    entries are preserved verbatim; every other frozen entry is dropped, and
    rebuilt only if its primary is a source. Add / remove / split / delete are
    handled uniformly: the source simply has more or fewer peaks above
    threshold. A rebuilt entry has Stage 5's schema
    (:func:`~ftmwpipeline.fitting.result_conversion.window_outcome_to_fitting_result`),
    the source line's ``peak_uid`` included.

    An accepted Stage 5 thaw leaves the thawed line a frozen contributor of the
    dependent, addressed through its primary (``ANALYSIS_EPOCH`` 6), so it is
    rebuilt here from the primary's current fit like any other frozen line.
    """
    wid = wf.window_id
    assert wid is not None
    d = int(wid)
    nonef = _non_edge_free_primaries(fit_window_map.get(d))

    def _is_dep(primary: int) -> bool:
        return nonef is None or primary in nonef

    non_frozen: Dict[str, Dict] = {}
    preserved_edge_free: List[Dict] = []
    for key, entry in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            non_frozen[key] = entry
            continue
        primary = int(entry["primary_window_id"])
        if not _is_dep(primary):
            preserved_edge_free.append(entry)

    rebuilt: List[Dict] = []
    for primary in sources:
        pwf = fit_map.get(int(primary))
        if pwf is None:
            continue  # source dropped/merged away -> contributes no skirt
        for pk in sorted(pwf.fitted_peaks, key=lambda q: float(q.frequency_mhz)):
            if float(pk.snr or 0.0) < min_freeze_snr:
                continue
            rebuilt.append(
                {
                    "peak_index": -1,
                    "primary_window_id": int(primary),
                    "frequency_mhz": float(pk.frequency_mhz),
                    "amplitude": float(pk.amplitude),
                    "phase": float(pk.phase) if pk.phase is not None else 0.0,
                    "freeze_eligible": True,
                    "peak_uid": None if pk.peak_uid is None else int(pk.peak_uid),
                }
            )

    frozen = preserved_edge_free + rebuilt
    rekeyed = {f"frozen_peak_{i}": e for i, e in enumerate(frozen)}
    wf.fixed_parameters = {**non_frozen, **rekeyed}


def _refuse_unavailable_fit_plan(
    unavailable: Collection[int], window_ids: Iterable[int], what: str
) -> None:
    """Refuse to refit a window whose fitted geometry the file does not hold.

    *unavailable* is :attr:`_SharedFitCtx.unavailable_window_ids`: in a fit
    that merged windows before the fitted plan was stored, the windows the
    merges touched. Refitting one on its Stage 4 geometry would move its lines
    to a window the fit was not made on, so the request is refused
    (``curation_conflict``, ``fit_plan_unavailable``, ``ids`` those windows).
    """
    blocked = sorted({int(w) for w in window_ids} & {int(w) for w in unavailable})
    if not blocked:
        return
    from .fitted_plan import FIT_PLAN_UNAVAILABLE

    listed = ", ".join(str(w) for w in blocked)
    raise CurationConflictError(
        FIT_PLAN_UNAVAILABLE,
        blocked,
        message=f"{what} would refit window(s) {listed}, which a structural "
        f"merge changed in a fit that predates the stored fitted plan: the "
        f"windows the fit was made on are not in this file. Re-run 'fit run' "
        f"to curate them.",
    )


def _cascade_refit_dependents(
    *,
    spectrum_fit: SpectrumFit,
    edited_wids: Sequence[int],
    sources: Mapping[int, Sequence[int]],
    fit_window_map: Dict[int, "FitWindow"],
    fit_ctx: "Stage5FitContext",
    resolved: "StageFitSettings",
    shape_enum: "PeakShape",
    persisted_cal: object,
    tau_maj_us: Optional[float],
    sigma_tau_us: Optional[float],
    tau_source: str,
    peak_frequencies_mhz: List[float],
    min_freeze_snr: float,
    snap_tol_mhz: float,
    events: Optional[StageScope] = None,
    unavailable_window_ids: Collection[int] = (),
    reached: Collection[int] = (),
    only: Optional[Collection[int]] = None,
) -> List[int]:
    """Refresh + identity-refit every dependent in the transitive closure of
    ``edited_wids`` (window-level), in dependency order; splice the results back
    into ``spectrum_fit``. ``tau_maj_us`` / ``sigma_tau_us`` are the **global**
    anchors -- each dependent is re-anchored per band. Returns the cascaded ids.

    ``reached`` adds windows the caller knows are downstream of an edit outside
    ``edited_wids`` (a window created this batch whose source an earlier write
    edited, :func:`_cascade_batch`); they are refreshed and refit in the same
    dependency order.

    ``sources`` is the cascade graph (:func:`_cascade_sources`): each window's
    ordered source list. Both the closure and each dependent's refresh read it,
    never the frozen entries of the fits being cascaded, so an edge an earlier
    edit emptied still carries the next one.

    ``only``, when given, replaces the closure: exactly those windows are
    refreshed and refit, in dependency order (the incremental engine's
    reached windows whose keys changed, :func:`_engine_run`).

    The refit is identity (no add/remove): a directly-edited dependent already
    carries its own edit in its peak set, so the identity refit honors both the edit
    and the refreshed skirt in one fit (design §3). Mutates ``spectrum_fit``.

    ``events`` (the Stage 6 operation's scope) gets a ``WindowProgress`` per
    re-fit dependent (phase ``"cascade"``, round 0, ``index`` over the
    cascade) and is checked for a cancel before each one.

    A dependent in ``unavailable_window_ids`` refuses the whole cascade before
    any refit (:func:`_refuse_unavailable_fit_plan`); its sources are every
    frozen contributor of its baseline fit, edge-free or not
    (:func:`_base_cascade_sources`), since the plan window that would say which
    are edge-free is not the one the fit was made on.
    """
    from ..fitting.result_conversion import sort_fitting_result_by_frequency

    window_fits = spectrum_fit.window_fits
    fit_map: Dict[int, FittingResult] = {
        int(wf.window_id): wf for wf in window_fits if wf.window_id is not None
    }
    succs = _cascade_succs(sources)
    closure = (
        _cascade_closure_set(edited_wids, sources, reached)
        if only is None
        else {int(w) for w in only}
    )
    if not closure:
        return []
    preds: Dict[int, set] = {w: set() for w in succs}
    for primary, deps in succs.items():
        for dep in deps:
            preds.setdefault(dep, set()).add(primary)
    ordered = _cascade_topo(closure, preds)
    _refuse_unavailable_fit_plan(
        unavailable_window_ids,
        (d for d in ordered if fit_map.get(d) is not None and fit_window_map.get(d)),
        "the dependency cascade",
    )

    cascaded: List[int] = []
    n_cascade = sum(
        1 for d in ordered if fit_map.get(d) is not None and fit_window_map.get(d)
    )
    for d in ordered:
        wf = fit_map.get(d)
        fit_win = fit_window_map.get(d)
        if wf is None or fit_win is None:
            continue
        if events is not None:
            events.check_cancel()
        t_window = time.monotonic()
        _refresh_frozen_from_sources(
            wf, sources.get(d, ()), fit_window_map, fit_map, min_freeze_snr
        )
        tm, st = _resolve_refit_window_tau(
            fit_win, resolved, persisted_cal, tau_maj_us, sigma_tau_us, tau_source
        )
        new_wf = refit_window_core(
            fit_ctx,
            fit_win,
            wf,
            resolved=resolved,
            shape_enum=shape_enum,
            tau_maj_us=tm,
            sigma_tau_us=st,
            peak_frequencies_mhz=peak_frequencies_mhz,
            snap_tol_mhz=snap_tol_mhz,
        )
        sort_fitting_result_by_frequency(new_wf)
        fit_map[d] = new_wf
        cascaded.append(d)
        if events is not None:
            events.window_progress(
                phase="cascade",
                round=0,
                index=len(cascaded),
                total=n_cascade,
                window_id=int(d),
                n_peaks=len(new_wf.fitted_peaks),
                chi2r=float(new_wf.reduced_chi2),
                elapsed_s=time.monotonic() - t_window,
                dropped=False,
                freq_range=(float(fit_win.freq_range[0]), float(fit_win.freq_range[1])),
            )

    if cascaded:
        cset = set(cascaded)
        spectrum_fit.window_fits = [
            (
                fit_map[int(wf.window_id)]
                if wf.window_id is not None and int(wf.window_id) in cset
                else wf
            )
            for wf in window_fits
        ]
        kept = [p for p in spectrum_fit.fitted_peaks if p.window_id not in cset]
        for d in cascaded:
            kept.extend(fit_map[d].fitted_peaks)
        kept.sort(key=lambda p: float(p.frequency_mhz))
        spectrum_fit.fitted_peaks = kept
    return cascaded


# ---------------------------------------------------------------------------
# Window derivation (W2): the window is optional where it is a COORDINATE
# (an add/remove target), never where it is the SUBJECT (accept, create, a
# bare identity edit). Windows are disjoint (FitWindow's docstring), so a
# frequency covers at most one live window and the derivation is total.
#
# THE RULE THAT MUST NOT BE VIOLATED: deriving the window never widens a
# search. Resolution is always (1) find which live window covers the
# frequency, THEN (2) run the existing, unchanged, snap-tolerance-bounded
# match inside that one window. There is no global nearest-peak search
# anywhere, for any verb, at any point. A frequency no live window covers is
# an error -- for ``remove`` this is permanent (a remove never implies a
# create).
# ---------------------------------------------------------------------------


def _load_curation_window_index(path: str) -> List[FitWindowCoverage]:
    """Load the cheap per-window coverage data used to resolve a derived
    window id, fresh from disk.

    Used only to resolve a derived window id *before* any batch opens (see
    :func:`_window_for_curation_token`) -- the batch engine reloads its own
    fresh full fit afterward (:func:`_build_batch_changeset`), so this is not
    a staleness risk, only an extra cheap read. Reads each window's
    ``window_id``, ``freq_min``/``freq_max`` attrs and its ``peak_uid``
    column -- not the ~30 attrs/datasets per window a full fit load pulls
    (see :func:`~ftmwpipeline.io.fitting_serialization.read_fit_window_coverage`).
    """
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise StageDependencyError(
                "review",
                ["stage5_fitting"],
                Path(str(path)),
                command="fit run",
                message="No Stage 5 fit found in this file. Run 'fit run' first.",
            )
        return read_fit_window_coverage(h5f["stage5_fitting"])


def _window_for_curation_token(
    coverage: Sequence[FitWindowCoverage], token: Union[float, PeakUidToken]
) -> Optional[int]:
    """Resolve one add/remove target to the live window that covers it.

    Live windows only, no snap (THE RULE above): a plain frequency resolves
    through :func:`~.stage5_impl.window_covering_freq` -- the window whose
    ``freq_range`` contains it, shared with the full-fit caller rather than
    reimplemented, so a curation-file row and a ``review edit`` call cannot
    disagree about which window a frequency covers. ``coverage`` is the cheap
    column read that stands in for a full fit here
    (:func:`~ftmwpipeline.io.fitting_serialization.read_fit_window_coverage`),
    already ``window_id``-ascending, which is the order the coverage rule
    matches in. A ``"uid:N"`` token resolves to the window whose fitted peaks
    include that ``peak_uid`` -- an EXACT match, no snap, since a
    ``peak_uid`` is unique across the whole fit.

    Returns ``None`` when nothing resolves; callers phrase their own error,
    since ``review edit`` and a curation file's rows need different wording.
    """
    from .stage5_impl import window_covering_freq

    if isinstance(token, PeakUidToken):
        for w in coverage:
            if token.uid in w.peak_uids:
                return w.window_id
        return None
    return window_covering_freq(
        ((w.window_id, w.freq_range) for w in coverage if w.freq_range is not None),
        float(token),
    )


def _refuse_bare_edit(add: Sequence[object], remove: Sequence[object]) -> None:
    """A ``review edit`` must name at least one ``add`` or ``remove`` target.

    An edit with neither would be an identity refit: it persists a
    re-converged fit and records no decision, so the log could no longer
    reproduce the fit. It is refused on every interface, before anything is
    resolved, fitted or written, whether or not a window is named."""
    if not add and not remove:
        raise BadSettingError(
            "add",
            "at least one add or remove target",
            [],
            message="review edit needs at least one add or remove target: an "
            "edit with neither would refit the window without recording a "
            "decision",
        )


def _derive_review_edit_window_id(
    path: str,
    add: Sequence[float],
    remove: Sequence[Union[float, PeakUidToken]],
) -> Tuple[Optional[int], Optional[float]]:
    """Derive the single live window every ``add``/``remove`` target in one
    ``review edit`` call resolves to -- or, when the call is exactly one
    ``add`` and nothing else and that one frequency is uncovered, the anchor
    for W3's implied create.

    Returns ``(window_id, implied_create_anchor)``, exactly one not ``None``:
    a concrete id when every target resolves (the W2 case, unchanged), or an
    anchor when the sole target is one uncovered ``add`` (W3). ``remove``
    NEVER implies a create -- an uncovered ``remove`` is a permanent error,
    named below -- and a call mixing an uncovered ``add`` with any other
    target (another ``add``, or any ``remove``) is refused with today's
    per-target error rather than guessing which of several possible creates
    the caller meant: ``RefitWindowResult`` is a per-window result, so only
    the single-add shape has an unambiguous single window to bind to.

    *add* and *remove* are never both empty: a bare edit is refused before
    this runs (:func:`_refuse_bare_edit`).

    Every target must resolve to the SAME window: a single edit call is
    scoped to one window's refit (``RefitWindowResult`` is a per-window
    result, and the intent inference reads one window's peak set), so two
    targets resolving to two different windows is an error naming both
    rather than a silent pick -- issue separate calls instead.
    """
    # Only a call shaped as exactly one add and nothing else has an
    # unambiguous single window to bind an implied create to -- see the
    # docstring above.
    eligible_for_implied_create = len(add) == 1 and not remove
    coverage = _load_curation_window_index(path)
    resolutions: List[Tuple[str, int]] = []
    unknown_uids = [
        t.uid
        for t in remove
        if isinstance(t, PeakUidToken)
        and _window_for_curation_token(coverage, t) is None
    ]
    if unknown_uids:
        raise _unknown_peak_uids_error(unknown_uids)
    # Every target no live window covers, reported together (not_found kind
    # "window", ids the uncovered frequencies).
    uncovered: List[float] = []
    uncovered_details: List[str] = []
    for f in add:
        wid = _window_for_curation_token(coverage, f)
        if wid is None:
            if eligible_for_implied_create:
                return None, float(f)
            uncovered.append(float(f))
            uncovered_details.append(
                f"add={float(f):.4f} MHz is not covered by any live window "
                f"(windows are disjoint); create a window at this "
                f"frequency first with 'review create', or add it in an edit "
                f"of its own (a lone add with no window named creates one)"
            )
            continue
        resolutions.append((f"add={float(f):.4f}", wid))
    for t in remove:
        wid = _window_for_curation_token(coverage, t)
        if wid is None:
            assert not isinstance(t, PeakUidToken)  # unknown uids raised above
            uncovered.append(float(t))
            uncovered_details.append(
                f"remove={float(t):.4f} MHz is not covered by any live "
                f"window (windows are disjoint); nothing to remove there"
            )
            continue
        label = (
            f"remove=uid:{t.uid}"
            if isinstance(t, PeakUidToken)
            else f"remove={float(t):.4f}"
        )
        resolutions.append((label, wid))
    if uncovered:
        raise NotFoundValueError(
            "window", uncovered, message="; ".join(uncovered_details)
        )
    wids = {wid for _, wid in resolutions}
    if len(wids) > 1:
        detail = "; ".join(f"{label} -> window {wid}" for label, wid in resolutions)
        raise CurationConflictError(
            "targets_span_windows",
            sorted(wids),
            message=f"add/remove targets in this edit resolve to different "
            f"windows ({detail}); issue separate 'review edit' calls, one per "
            f"window, or pass a window id explicitly",
        )
    return next(iter(wids)), None


@requires_pipeline_file()
@_review_operation(
    "review edit", lambda r, a: review_edit_summary(r, a["add"], a["remove"])
)
def refit_window_impl(
    file_path: Union[Path, str],
    window_id: Optional[int] = None,
    *,
    add: Sequence[Union[float, str]] = (),
    remove: Sequence[Union[float, str]] = (),
    frame: Optional[Frame] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> RefitWindowResult:
    """User-directed single-window refit for Stage 6 review decisions.

    Re-fits one window from the persisted Stage 5 fit using the production
    NLS primitive (``fit_window``).  Starts from the persisted ``fitted_peaks``
    as seed ``ModelPeak`` objects, reconstructs the frozen background from the
    window's own persisted ``fixed_parameters`` and replays the persisted
    leakage-wing baseline, applies the caller's ``add``/``remove`` edits, and
    runs a single joint NLS over the full seeded set.  It is **NLS-only**: it
    deliberately does NOT run the conservative add-one-peak discovery loop or
    the residual-rescue pass -- a refit holds the window's peak *set* (plus or
    minus the edit) and re-converges it, rather than re-discovering peaks (the
    automatic discovery already ran; the persisted peaks are its result).  No
    thaw and no replan: the only other windows this can touch are the dependents
    whose frozen leakage skirt the edit moved, which the cascade re-fits.

    Runs through the one write path (:func:`_curate_request`), exactly as a
    curation file's ``edit`` row does: it takes the undo baseline, resolves
    the edit into decision rows against the displayed fit before any fit
    (:func:`_resolve_edit_steps`: the removes to the uids of the displayed
    peaks they name, each birth to its seed and the uid stamped from it),
    then replays the decision log with the rows appended from the automatic
    fit, cascades, and persists the result (:func:`_curate`; epoch-gated,
    since it refits). The rows apply to the window's state as the replay
    reaches it -- its automatic fit and its own earlier rows -- and the
    cascade then refreshes it: what the user saw is used only to resolve the
    request into rows.

    At least one ``add`` or ``remove`` target is required: an edit with
    neither (an identity refit, which would persist a re-converged fit and
    record no decision) is refused with ``bad_setting`` (``path`` ``"add"``)
    before anything is resolved, fitted or written, whether or not
    ``window_id`` is given.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The ``FitWindow.window_id`` / ``FittingResult.window_id`` to refit.
        Optional (``None``, the default) when ``add`` or ``remove`` is
        non-empty: the window is then derived from the target frequencies
        (or ``"uid:N"`` identifiers) by live-window coverage -- see
        :func:`_derive_review_edit_window_id`. When the call is exactly one
        ``add`` and nothing else and that frequency is covered by no live
        window, the window it needs is minted (or an adjacent one widened)
        and the add applied into it, as ONE decision -- W3, see
        :func:`_resolve_action`; a ``remove`` never implies a
        create, and a call mixing more than one uncovered target is refused
        rather than guessing. A *named* window is still checked: naming the
        wrong one is still an error, exactly as before -- the implied create
        fires only when the window was OMITTED.
    add :
        Molecular frequencies (MHz) of peaks to add, as ``float`` or a numeric
        ``str`` (the CLI passes strings).  Each is snapped to the nearest
        ledger candidate within the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`) (to reuse the recorded seed
        offset/amplitude) or seeded fresh at the given frequency.  User-added
        peaks carry ``origin="user"`` and are stamped with the
        ``serial`` of the decision that added them
        (:attr:`~ftmwpipeline.core.data_structures.FittedPeak.derivation`).
        The post-snap frequency must lie inside the (named) window's own
        ``freq_range``; with ``window_id`` omitted and no live window
        covering it, see W3 above instead of :func:`create_window_impl`.
        ``add`` is frequency-only: a ``"uid:N"`` token is refused (see
        :func:`~ftmwpipeline.core.curation.parse_peak_token`) -- a uid names
        a peak that already exists, and ``add`` has none.
    remove :
        Frequencies (MHz, ``float`` or numeric ``str``) or ``"uid:N"``
        identifier tokens of fitted peaks to remove; the two may be mixed in
        one call.  A frequency is matched to the nearest fitted peak within
        the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`).  A ``"uid:N"`` token is resolved to the fitted peak
        in this window whose
        :attr:`~ftmwpipeline.core.data_structures.FittedPeak.peak_uid`
        equals ``N`` -- frame-independent, and always an exact match rather
        than a snap (see :func:`~ftmwpipeline.core.curation.parse_peak_token`
        for the grammar). Either way the matched peak is dropped from the
        seed set before the NLS.
    frame :
        The frame ``add`` and ``remove`` are expressed in: ``"raw"`` (the
        Stage 5 fit / ledger frame) or ``"calibrated"``. Converted to raw
        before any snapping; storage stays raw regardless. Omitting it
        defaults to ``"raw"`` and is an error on a ``self_calibrated`` file
        when ``add`` or ``remove`` is non-empty (see
        :func:`~ftmwpipeline.core.curation.Frame`).
    _shared :
        Internal. An already-built :class:`_SharedFitCtx` to reuse (see
        ``ReviewSession``, D3) instead of rebuilding it from the file.

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks
        (both frames -- see :class:`RefitWindowResult`). The "before" side
        is the window as displayed before the call; the "after" side is its
        curated fit after the write, post-cascade.

    Raises
    ------
    ValueError
        When Stage 5 has not been run, the ``window_id`` is not found,
        any ``remove`` frequency does not
        match a fitted peak within the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`), any ``remove`` uid
        matches no fitted peak in the window, any ``add`` token is malformed
        or names a ``"uid:N"`` identifier, any ``add`` frequency falls
        outside the named window's ``freq_range`` after snapping,
        ``frame`` is omitted on a ``self_calibrated`` file with a non-empty
        ``add``/``remove``, ``add`` and ``remove`` are both empty, an
        omitted ``window_id``'s ``remove`` target
        (frequency or ``"uid:N"``) is not covered by any live window
        (permanent -- a remove never implies a create), an omitted
        ``window_id``'s uncovered ``add`` target is not the call's sole
        target (mixed with another ``add`` or any ``remove``), several
        omitted-window targets in one call resolve to different windows, or
        the sole uncovered ``add``'s implied create anchor falls outside the
        analysis band.
    """
    _refuse_bare_edit(add, remove)
    path = str(file_path)
    _require_engine_file(path)
    snap_tol = refit_snap_tol_mhz_impl(path)

    add_raw: List[float] = []
    for tok in add:
        parsed = parse_peak_token(tok, path="add")
        if isinstance(parsed, PeakUidToken):
            raise BadSettingError(
                "add",
                "a frequency in MHz, not a peak identifier",
                f"uid:{parsed.uid}",
                message=f"add takes a frequency (MHz), not a peak identifier "
                f"('uid:{parsed.uid}'): a uid names a peak that already "
                f"exists, but add creates a new one",
            )
        add_raw.append(parsed)
    remove_raw: List[Union[float, PeakUidToken]] = [
        parse_peak_token(tok, path="remove") for tok in remove
    ]

    if add_raw or remove_raw:
        resolved_frame, stamp = _resolve_frame(path, frame)
        add_raw = [_frame_to_raw(f, frame=resolved_frame, stamp=stamp) for f in add_raw]
        remove_raw = [
            (
                t
                if isinstance(t, PeakUidToken)
                else _frame_to_raw(t, frame=resolved_frame, stamp=stamp)
            )
            for t in remove_raw
        ]

    resolved_window_id: Optional[int]
    implied_anchor: Optional[float]
    if window_id is not None:
        resolved_window_id, implied_anchor = window_id, None
    else:
        resolved_window_id, implied_anchor = _derive_review_edit_window_id(
            path, add_raw, remove_raw
        )

    if implied_anchor is not None:
        # W3: the sole add target is uncovered by any live window -- mint
        # (or widen) the window it needs, then apply the edit into it, as
        # ONE decision-log entry (the add), not two. See
        # scratch/intent-driven-windowing-plan.md, W3 'Shape'.
        actions = [
            PlannedAction(
                kind="create",
                window_id=_FIRST_IMPLIED_WINDOW_ID,
                anchor=implied_anchor,
                implied_create=True,
            ),
            PlannedAction(
                kind="edit",
                window_id=_FIRST_IMPLIED_WINDOW_ID,
                add=list(add_raw),
                implied_create=True,
            ),
        ]
    else:
        assert resolved_window_id is not None
        actions = [
            PlannedAction(
                kind="edit",
                window_id=resolved_window_id,
                add=list(add_raw),
                remove=list(remove_raw),
            )
        ]

    try:
        req = _curate_request(
            path,
            actions,
            snap_tol_mhz=snap_tol,
            shared=_shared,
            persist=True,
            one_action=True,
        )
    except BadSettingError as exc:
        if implied_anchor is None or exc.path != "anchor_mhz":
            raise
        # The anchor is the caller's add, not review_create's argument.
        raise BadSettingError("add", exc.expected, exc.value, message=str(exc)) from exc
    return _refit_result(req, req.resolved[-1])


def _single_action_refit(
    path: str,
    action: PlannedAction,
    *,
    snap_tol_mhz: float,
    shared: Optional["_SharedFitCtx"],
) -> RefitWindowResult:
    """Write a single-window verb's one fit-changing action (a merge, a
    split, an accept with a candidate) and return its window's result."""
    req = _curate_request(
        path,
        [action],
        snap_tol_mhz=snap_tol_mhz,
        shared=shared,
        persist=True,
        one_action=True,
    )
    return _refit_result(req, req.resolved[-1])


# ---------------------------------------------------------------------------
# merge_peaks_impl: collapse ≥2 fitted peaks into one
# ---------------------------------------------------------------------------


def merge_peaks_impl(
    file_path: Union[Path, str],
    window_id: int,
    peaks: Sequence[float],
    *,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> RefitWindowResult:
    """Collapse ≥2 fitted peaks in a window into a single peak.

    A thin composition over :func:`refit_window_impl`: removes the named peaks
    and adds one replacement seeded at their SNR-weighted centroid (or
    amplitude-weighted centroid when SNR is unavailable).  All products carry
    ``origin="user"`` and the ``serial`` of the single ``"merge"``
    decision this records (:attr:`FittedPeak.derivation`) -- a merge replaces
    identities rather than remeasuring them, which is exactly what the tag
    tells a downstream consumer.

    **Doublet-alternative snap.** When the removed set matches a persisted
    ``DoubletAlternativeInfo`` pair (i.e. exactly two frequencies that
    together map to a recorded doublet pair within the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`)), the
    replacement seed is taken from the recorded ``merged_frequency_mhz`` and
    ``merged_amplitude`` rather than the centroid.  This reuses the
    already-converged merged-alternative optimum from the Stage 5 doublet
    adjudication pass.

    # NOTE(opus-review): doublet-alt snap is wired for the two-peak case only
    # because DoubletAlternativeInfo records exactly one pair at a time.
    # Multi-peak merge (K>2) falls through to the weighted-centroid seed.
    # The snap requires a successful merged refit (``merged_success=True`` and
    # non-NaN ``merged_frequency_mhz``); if the record is absent or the refit
    # failed, the centroid seed is used instead — no silent error.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The window containing the peaks to merge.
    peaks :
        Molecular frequencies (MHz) of the peaks to collapse.  At least 2
        must be provided.  Each is snapped to the nearest fitted peak within
        the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`).
    frame :
        The frame ``peaks`` is expressed in (see :func:`refit_window_impl`).
        Omitting it is an error on a ``self_calibrated`` file.
    _shared :
        Internal. See :func:`refit_window_impl`.

    Returns
    -------
    RefitWindowResult
        Old vs new peak count (reduced by ``len(peaks) - 1``), χ²ᵣ
        before/after, and the new fitted peaks (both frames).

    Raises
    ------
    ValueError
        When fewer than 2 frequencies are supplied, any frequency does not
        match a fitted peak within tolerance, Stage 5 has not been run, or
        ``frame`` is omitted on a ``self_calibrated`` file.
    """
    # Checked before the batch opens so a malformed call cannot even take the
    # undo baseline: a refused edit must leave the file untouched. The applier
    # re-checks, since a curation row reaches it without passing through here.
    _check_merge_arity(peaks)
    path = str(file_path)
    # One atomic write, opened before the inputs are read (joins an enclosing
    # transaction, e.g. a curation replay).
    with atomic_write(path):
        _require_engine_file(path)
        snap_tol = refit_snap_tol_mhz_impl(path)
        resolved_frame, stamp = _resolve_frame(path, frame)
        peaks_raw = [_frame_to_raw(f, frame=resolved_frame, stamp=stamp) for f in peaks]
        action = PlannedAction(kind="merge", window_id=window_id, peaks=peaks_raw)
        return _single_action_refit(path, action, snap_tol_mhz=snap_tol, shared=_shared)


# ---------------------------------------------------------------------------
# split_peak_impl: replace one fitted peak with K peaks
# ---------------------------------------------------------------------------


def split_peak_impl(
    file_path: Union[Path, str],
    window_id: int,
    peak: float,
    *,
    into: int = 2,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> RefitWindowResult:
    """Replace one fitted peak with ``into`` peaks (default 2).

    A thin composition over :func:`refit_window_impl`: removes the named peak
    and adds ``into`` replacements spread symmetrically about it by ±½ of one
    Fourier resolution element (``1 / acquisition_us`` MHz).  All products
    carry ``origin="user"`` and the ``serial`` of the single ``"split"``
    decision this records (:attr:`FittedPeak.derivation`), so a consumer reads
    the products as replacements rather than pairing them to the original.
    Seeds are clamped into the window's own range, so splitting a peak that
    sits within half a resolution element of an edge still works.

    The resolution element is taken from the fit context's ``acquisition_us``
    (the active-FT window length used during the original Stage 5 fit),
    avoiding any recomputation.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The window containing the peak to split.
    peak :
        Molecular frequency (MHz) of the peak to split.  Snapped to the
        nearest fitted peak within the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`).
    into :
        Number of replacement peaks (≥2).  Default is 2.
    frame :
        The frame ``peak`` is expressed in (see :func:`refit_window_impl`).
        Omitting it is an error on a ``self_calibrated`` file.
    _shared :
        Internal. See :func:`refit_window_impl`.

    Returns
    -------
    RefitWindowResult
        Old vs new peak count (increased by ``into - 1``), χ²ᵣ
        before/after, and the new fitted peaks (both frames).

    Raises
    ------
    ValueError
        When ``into < 2``, the frequency does not match a fitted peak within
        tolerance, Stage 5 has not been run, or ``frame`` is omitted on a
        ``self_calibrated`` file.
    """
    # Checked before the batch opens: see :func:`merge_peaks_impl`.
    _check_split_arity(into)
    path = str(file_path)
    # One atomic write, opened before the inputs are read (joins an enclosing
    # transaction, e.g. a curation replay).
    with atomic_write(path):
        _require_engine_file(path)
        snap_tol = refit_snap_tol_mhz_impl(path)
        resolved_frame, stamp = _resolve_frame(path, frame)
        peak_raw = _frame_to_raw(peak, frame=resolved_frame, stamp=stamp)
        action = PlannedAction(
            kind="split", window_id=window_id, peak=peak_raw, into=into
        )
        return _single_action_refit(path, action, snap_tol_mhz=snap_tol, shared=_shared)


# ---------------------------------------------------------------------------
# Decision recording helpers and review-accept/status impls
# ---------------------------------------------------------------------------

#: Evidence key tying the decision-log rows of ONE user action together: the
#: ``serial`` of the first row that action recorded (a one-row action carries
#: its own serial). A ``review edit`` with several ``add``/``remove``
#: frequencies, or a run of add/remove rows on one window in a curation file
#: or action batch, is applied as one joint refit but logs one row per
#: frequency; this key is what lets a replay of the log re-apply those rows
#: as the one action they were (:func:`_decision_action_groups`).
ACTION_INDEX_EVIDENCE_KEY = "action_index"


def _serial_id(entry: DecisionLogEntry) -> int:
    """*entry*'s serial, the id ``review undo`` and the ``curation_conflict``
    reasons name a decision by (every row of an admitted file carries one)."""
    if isinstance(entry.serial, Absent):
        raise ValueError(
            f"internal: decision at log position {entry.order_index} carries no "
            "serial"
        )
    return int(entry.serial)


@requires_pipeline_file()
@_review_operation(
    "review accept", lambda r, a: review_accept_summary(r, a["window_id"])
)
def review_accept_impl(
    file_path: Union[Path, str],
    window_id: int,
    *,
    candidate_freq: Optional[float] = None,
    frame: Optional[Frame] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> Optional[RefitWindowResult]:
    """Accept a window as-is or accept a specific revived candidate.

    With no ``candidate_freq``: records a ``"accept"`` decision log entry and
    sets the window's provenance to ``"reviewed"`` without modifying the fit.
    Its attention reasons stay (they remain advisory after the user has
    looked at the window).  Returns ``None``. It refits nothing, so it needs
    no fit context and is not epoch-gated (:func:`_curate`); like every
    write, it takes the undo baseline if none was taken yet.

    With ``candidate_freq``: delegates to :func:`refit_window_impl` with
    ``add=[candidate_freq]``, which records a ``"add"`` decision log entry and
    sets provenance to ``"user-edited"``.  Returns the
    :class:`RefitWindowResult`.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The :class:`~ftmwpipeline.core.data_structures.FittingResult` window
        to accept.
    candidate_freq :
        When given, add this molecular frequency (MHz) as a new peak and
        accept the resulting fit.  The frequency is snapped to the nearest
        ledger candidate within the file's snap tolerance (:func:`refit_snap_tol_mhz_impl`).
    frame :
        The frame ``candidate_freq`` is expressed in (see
        :func:`refit_window_impl`). Irrelevant, and never validated, when
        ``candidate_freq`` is ``None`` -- a bare accept carries no frequency.
        Omitting it while ``candidate_freq`` is given is an error on a
        ``self_calibrated`` file.
    _shared :
        Internal. See :func:`refit_window_impl`. Unused when
        ``candidate_freq`` is ``None`` -- a bare accept needs no fit context.

    Returns
    -------
    RefitWindowResult or None
        ``None`` when accepting as-is; the refit result when ``candidate_freq``
        was given (both frames -- see :class:`RefitWindowResult`).
    """
    path = str(file_path)
    _require_engine_file(path)
    snap_tol = refit_snap_tol_mhz_impl(path)
    if candidate_freq is None:
        # A bare accept marks the window reviewed and changes no fitted
        # number: it needs no fit context (the expensive part) and refits
        # nothing, so the epoch gate does not apply to it. No frequency is
        # carried, so `frame` is moot and is never resolved/validated here.
        _require_known_window(path, window_id)
        _curate_request(
            path,
            [PlannedAction(kind="accept", window_id=window_id)],
            snap_tol_mhz=snap_tol,
            shared=_shared,
            persist=True,
            one_action=True,
        )
        return None

    resolved_frame, stamp = _resolve_frame(path, frame)
    candidate_raw = _frame_to_raw(candidate_freq, frame=resolved_frame, stamp=stamp)
    action = PlannedAction(kind="accept", window_id=window_id, candidate=candidate_raw)
    return _single_action_refit(path, action, snap_tol_mhz=snap_tol, shared=_shared)


def _require_complete_fit(path: str, verb: str) -> None:
    """Refuse a curation call on a file with no complete Stage 5 fit.

    Never fit, or holding only a partial fit (a cancelled fit's kept windows,
    CONTRACT_STRATEGY §Stage 5 partial fits): ``stage_not_run``, naming
    ``fit run``.
    """
    with h5open(path, "r") as h5f:
        present = "stage5_fitting" in h5f
    if not present:
        raise StageDependencyError(
            verb,
            ["stage5_fitting"],
            Path(path),
            command="fit run",
            message="No Stage 5 fit found in this file. Run 'fit run' first.",
        )


def _known_window_ids(path: str) -> Tuple[Set[int], str]:
    """The window ids a bare accept may name (the Stage 5 fit's, which include
    every created window, plus the windows the review flagged
    ``empty_window_residual``, which the fit holds no line in), and where they
    were read from.

    A bare accept changes no fitted number, but it is a curation decision on
    the fit: with no complete fit (never run, or only a partial fit) it is
    refused with ``stage_not_run`` like every other curation call.
    """
    _require_complete_fit(path, "review accept")
    with h5open(path, "r") as h5f:
        known = {c.window_id for c in read_fit_window_coverage(h5f["stage5_fitting"])}
        review = (
            load_stage6_review_from_hdf5(h5f["stage6_review"])
            if "stage6_review" in h5f
            else Stage6Review()
        )
    # A window the review flagged although the fit holds no line in it
    # (``empty_window_residual``) can be marked reviewed too.
    known |= flagged_lineless_ids(review, known)
    return known, "Stage 5 fit"


def _require_known_window(path: str, window_id: int) -> None:
    """Refuse a window id this file does not have (or a file with no complete
    fit, :func:`_known_window_ids`).

    Without this a typo'd id recorded a "reviewed" status and a decision for a
    window that was never there.
    """
    known, where = _known_window_ids(path)
    if int(window_id) not in known:
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the {where}",
        )


def _plan_window_center(path: str, window_id: int, default: float) -> float:
    """The centre of ``window_id``'s fitted-plan range, or ``default`` when the
    fitted plan has no such window (a lineless flagged window's accept anchor)."""
    from .fitted_plan import load_fitted_plan

    for w in load_fitted_plan(path).plan.windows:
        if int(w.window_id) == int(window_id):
            return 0.5 * (float(w.freq_range[0]) + float(w.freq_range[1]))
    return default


# ---------------------------------------------------------------------------
# create_window_impl: install a fit window for a line the detector missed
# ---------------------------------------------------------------------------


@dataclass
class CreateWindowResult:
    """Outcome of :func:`create_window_impl`.

    Attributes
    ----------
    window_id : int
        The window now covering the anchor -- a fresh id for ``mode="created"``,
        an existing one for ``mode="widened"``.
    mode : str
        ``"created"`` (a new window was built in a gap) or ``"widened"`` (the gap
        was too narrow, so the adjacent window absorbed the anchor).
    anchor_mhz : float
        The requested molecular frequency (MHz), in the raw / fit frame (the
        frame the anchor was converted to before installing the window).
    anchor_calibrated_mhz : float
        The same anchor in the calibrated frame. Equal to ``anchor_mhz`` when
        ``epsilon == 0``.
    freq_range : tuple of float
        The installed window's ``(min_mhz, max_mhz)`` extent, raw frame.
    freq_range_calibrated : tuple of float
        ``freq_range`` in the calibrated frame.
    n_points : int
        Grid points the window covers.
    n_contributors : int
        Frozen leakage contributors attached to it.
    depends_on : list of int
        Window ids it reads frozen leakage from.
    n_peaks : int
        Fitted peaks in the window after the create (``0`` for a fresh window --
        creating a window installs *structure*; adding the line is a separate
        ``review edit --add`` decision).
    calibration_state : str
        ``\"rb_locked\"`` / ``\"self_calibrated\"`` / ``\"uncalibrated\"`` --
        the file's calibration state at the moment of this create.
    epsilon : float
        The fractional timebase scale error actually applied (``0.0`` unless
        ``calibration_state == \"self_calibrated\"``).
    sigma_epsilon : float
        1-sigma uncertainty on ``epsilon`` (``0.0`` when inapplicable).
    """

    window_id: int
    mode: str
    anchor_mhz: float
    freq_range: Tuple[float, float]
    n_points: int
    n_contributors: int
    depends_on: List[int]
    n_peaks: int
    anchor_calibrated_mhz: float = 0.0
    freq_range_calibrated: Tuple[float, float] = (0.0, 0.0)
    calibration_state: str = "rb_locked"
    epsilon: float = 0.0
    sigma_epsilon: float = 0.0
    #: The stages the call invalidated (canonical names, ``rerun_order``):
    #: always ``()``, since Stage 6 invalidates no stage.
    invalidated: Tuple[str, ...] = field(default=(), compare=False)


def _created_window_seed(
    shared: "_SharedFitCtx",
    fit_win: "FitWindow",
    created_windows: Sequence["FitWindow"],
) -> Dict[str, Dict]:
    """``fixed_parameters`` a ``mode="created"`` window starts from: the
    cascade refresh (:func:`_refresh_frozen_from_sources`) of its sources
    (:func:`_created_window_sources`, against the created windows
    *created_windows* installed before it), read from the **automatic fit**
    (:attr:`_SharedFitCtx.baseline_fits`). A created source has no automatic
    fit and contributes nothing.

    The automatic fit is constant within a lineage, so the starting skirt does
    not depend on where the create sits in the log relative to its sources'
    edits or to the adds into an earlier created source. It is also what the
    cascade would rebuild from a source no decision has edited, and the
    cascade refreshes the window from its sources' final fits whenever one of
    them has been edited (:func:`_cascade_batch`), so every source's skirt in
    the persisted fit is read from that source's final fit.
    """
    wid = int(fit_win.window_id)
    base_live = shared.base_cascade_sources
    created_ids = {
        int(w.window_id) for w in created_windows if int(w.window_id) not in base_live
    }
    seed = FittingResult(window_id=wid, shape=shared.shape_enum.value)
    _refresh_frozen_from_sources(
        seed,
        _created_window_sources(fit_win, base_live, created_ids),
        {wid: fit_win},
        shared.baseline_fits,
        shared.min_freeze_snr,
    )
    return seed.fixed_parameters


@requires_pipeline_file()
@_review_operation("review create", lambda r, a: review_create_summary(r))
def create_window_impl(
    file_path: Union[Path, str],
    anchor_mhz: float,
    *,
    frame: Optional[Frame] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> CreateWindowResult:
    """Install a Stage 6 fit window covering ``anchor_mhz`` (``review create``).

    "Add the weak line over there that the detector missed" is a routine
    request, but until a window covers that frequency there is nothing to edit:
    Stage 4 builds windows around *promoted* Stage 3 detections, and re-running
    detection at a lower threshold invalidates Stages 5 and 6, destroying the
    entire curated edit set. This verb creates the missing structure instead,
    leaving every existing decision standing.

    It is a **structural** decision and nothing more: the window is installed
    and fit with an empty peak set (a null fit that reports the window's data
    χ²). Putting a line in it is a separate ``review edit --add`` decision. The
    two are kept apart deliberately -- creating structure and changing a
    window's peak set are different operations, and a log that spelled both
    ``add`` could not be replayed or diffed without re-deriving window
    membership against the base plan. A client is free to offer both as one
    gesture; the log still records two decisions.

    Geometry, contributors, and the narrow-gap widening fallback are decided by
    :func:`~ftmwpipeline.preprocessing.window_planning.plan_stage6_window`,
    which is a pure function of the anchor, the *base* plan and the windows
    earlier creates installed -- never of any fit -- so replaying the creates
    in log order reproduces the same windows. Ids are only ever appended: an
    existing window is never renumbered, so a consumer partitioning peaks on
    ``window_id`` sees exactly the windows the edit touched, and a fresh id is
    minted above every id a create has ever taken in the lineage
    (``Stage6Review.window_id_high_water``), so an undone create's id never
    names a different window later.

    **Creating a window refits no other window.** The new window reads the
    frozen leakage skirts of the windows it is attached to inward, starting
    from their automatic fits (never a curated fit, so the result does not
    depend on where the create sits in the log); when one of those windows
    has been edited, the create refreshes the skirt from its current fit and
    refits the new window again, as the cascade would. No neighbor is re-fit
    or thawed by the create. It is not a leaf, though: no
    base window ever reads a created one, but a later create attaches to an
    earlier created window like to any other source, so a line later added to
    this window cascades to the created windows that read it. That argument
    is specific to ``mode="created"``: when the requested anchor instead falls
    in a gap too narrow to hold a new window and an existing neighbor is
    widened in its place (``mode="widened"``), that neighbor is an established
    window whose fit just moved on the wider grid, so the create itself
    cascades to its dependents.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    anchor_mhz :
        Molecular frequency (MHz) the new window must cover.
    frame :
        The frame ``anchor_mhz`` is expressed in (see
        :func:`refit_window_impl`). Omitting it is an error on a
        ``self_calibrated`` file.
    _shared :
        Internal. See :func:`refit_window_impl`.

    Returns
    -------
    CreateWindowResult
        Both frames on the anchor and the installed range -- see
        :class:`CreateWindowResult`.

    Raises
    ------
    ValueError
        When Stage 5 has not been run, the anchor is outside the analysis band,
        the anchor already falls inside an existing window (that case is an
        ordinary ``review edit --add`` on that window), or ``frame`` is
        omitted on a ``self_calibrated`` file.
    """
    path = str(file_path)
    _require_engine_file(path)
    snap_tol = refit_snap_tol_mhz_impl(path)
    resolved_frame, stamp = _resolve_frame(path, frame)
    anchor_raw = _frame_to_raw(anchor_mhz, frame=resolved_frame, stamp=stamp)
    action = PlannedAction(
        kind="create", window_id=_NEW_WINDOW_SENTINEL, anchor=anchor_raw
    )
    req = _curate_request(
        path,
        [action],
        snap_tol_mhz=snap_tol,
        shared=_shared,
        persist=True,
        one_action=True,
    )
    planned = req.resolved[0].create
    assert planned is not None and req.display is not None
    window = _fit_by_window(req.curated.spectrum_fit)[
        int(planned.proposal.window.window_id)
    ]
    return _create_window_result(
        req.display.shared, planned, n_peaks=len(window.fitted_peaks)
    )


# ---------------------------------------------------------------------------
# Curation files: a human-editable batch language for the review edits.
#
# A curation file (CSV) records an ordered sequence of Stage 6 edits, one per
# row, that ``apply_curation_impl`` replays through the same edit impls the
# interactive verbs call. The CSV is the canonical interchange: diffable,
# hand-editable, and independent of the report that may have authored it.
# ---------------------------------------------------------------------------

_CURATION_ACTIONS = ("add", "remove", "accept", "create")
"""Row actions a curation *file* may name. ``merge``/``split`` are not among
them -- see :func:`parse_curation_file`'s refusal for those tokens -- even
though :class:`CurationOp` and :class:`PlannedAction` still carry ``"merge"``/
``"split"`` kinds internally: the internal verbs (:func:`merge_peaks_impl`,
:func:`split_peak_impl`) plan one, and the inference path
(:func:`_infer_curation_intent`) still resolves an add/remove combination to
one. Only a hand-authored file's own row vocabulary shrank."""
_CURATION_HEADER = ("action", "window", "freqs", "params")

# ``create`` is the one action whose window id is normally an *output*, not an
# input: it installs the window the anchor needs. A hand-authored file writes one
# of these tokens in the window column to say "whichever id this turns out to
# be". A named id instead *pins* the id the create takes -- which is what a file
# generated from the decision log writes, so the window keeps its identity
# across a replay even if an earlier create was dropped from the edit set.
#
# W2 reuses the SAME token set on an ``add``/``remove`` row's window column,
# meaning something different there: the window is a COORDINATE derived from
# the row's own frequency (or "uid:N") by live-window coverage, not an output
# to mint -- see ``_DERIVE_WINDOW_SENTINEL`` and ``_resolve_curation_window_ids``.
# Deliberately the same spellings rather than inventing new ones: "the window
# column is omitted" reads the same way regardless of which action it is on.
_CURATION_NEW_WINDOW_TOKENS = ("new", "auto", "-", "")
_NEW_WINDOW_SENTINEL = -1
_DERIVE_WINDOW_SENTINEL = -2
"""Parsed in place of an add/remove row's omitted window token (W2); resolved
to a concrete live-window id by :func:`_resolve_curation_window_ids` before
:func:`_resolve_curation_plan` ever sees it -- unlike ``_NEW_WINDOW_SENTINEL``,
this value never reaches a :class:`PlannedAction` or the decision log."""

_FIRST_IMPLIED_WINDOW_ID = -3
"""W3: the first of a stream of unique negative *correlation* ids
(``_FIRST_IMPLIED_WINDOW_ID``, ``_FIRST_IMPLIED_WINDOW_ID - 1``, ...)
:func:`_resolve_curation_window_ids` mints when an ``add`` row's frequency is
covered by no live window. That row is expanded into a ``create`` op and an
``add`` op sharing one fresh correlation id as their ``window_id`` -- late
binding for the id the create will actually mint (or widen), which is not
known until execution. Distinct from ``_NEW_WINDOW_SENTINEL`` (an EXPLICIT,
unpinned ``create`` row -- unpinned because the caller does not care which id
it gets, not because something else needs to find it again) and
``_DERIVE_WINDOW_SENTINEL`` (resolved away before this point): a correlation
id is a private handshake between exactly one ``create``/``add`` pair, unique
per pair so several implied creates in one plan cannot cross-wire, and it
never reaches the decision log -- see :func:`_resolve_curation_window_ids`
and :func:`_resolve_action`, which resolves it to the create's real minted id
before anything is recorded. A REPLAYED implied create
(:func:`_replay_action_groups`) does not need this:
the decision log already carries the concrete, pinned id, so ``window_id``
there is real from the start -- ``PlannedAction.implied_create`` (not the id)
is what signals "one merged decision" on that path.
"""


def _is_implied_window_id(window_id: int) -> bool:
    """Whether *window_id* is one of the fresh, not-yet-real correlation
    placeholders :data:`_FIRST_IMPLIED_WINDOW_ID` starts -- true only before
    the paired ``create`` action has run and the real id is known. Never true
    of a REPLAYED implied create, whose ``window_id`` is already the real,
    pinned id (see :data:`_FIRST_IMPLIED_WINDOW_ID`'s docstring)."""
    return window_id <= _FIRST_IMPLIED_WINDOW_ID


@dataclass
class CurationOp:
    """One parsed row of a curation file (before coalescing). A replay of the
    decision log never goes through ops: it applies the recorded rows by peak
    identity (:func:`_resolve_replay_rows`).

    Attributes
    ----------
    action : str
        One of ``add`` / ``remove`` / ``accept`` / ``create`` for anything
        :func:`parse_curation_file` produces. ``merge`` / ``split`` never come
        from a file row (see that function's refusal) but can still appear
        in an op list built directly.
    window_id : int
        The target ``FitWindow.window_id``.
    freqs : list of float or PeakUidToken
        Molecular MHz frequencies (or, on a ``remove`` row only, ``"uid:N"``
        peak-identifier tokens -- see
        :func:`~ftmwpipeline.core.curation.parse_peak_token`) the row
        carries; empty for a bare accept. Every action other than ``remove``
        is frequency-only -- :func:`parse_curation_file` refuses a
        ``PeakUidToken`` anywhere else, so this list holds plain ``float``
        for every action but ``remove``.
    params : dict
        ``key=value`` modifiers (``into`` for split, ``candidate`` for accept).
    line_no : int
        1-based source line, for diagnostics.
    implied_create : bool
        W3. ``True`` on the synthesized ``create``/``add`` pair
        :func:`_resolve_curation_window_ids` builds from one uncovered
        ``add`` row -- signals that the
        pair must record ONE decision (the add), not the create's own
        separate ``create_window`` entry plus the add's. ``False`` (the
        default) for everything else, including an ordinary EXPLICIT
        ``create`` row followed by its own, separately-decided ``add`` row on
        the same window id -- that pair is two decisions, unaffected.
    freq_cell : str or None
        The ``bad_setting`` path of the row's frequency as the caller wrote
        it (``curation[line <n>].freqs``, or ``actions[<i>].freq_mhz`` for a
        batch of actions), which a refused ``create`` anchor is reported at;
        ``None`` for an op replayed from the decision log.
    """

    action: str
    window_id: int
    freqs: List[Union[float, PeakUidToken]]
    params: Dict[str, str]
    line_no: int
    implied_create: bool = False
    freq_cell: Optional[str] = field(default=None, compare=False, repr=False)


@dataclass
class CurationFileHeader:
    """A curation file's optional file-level frame declaration (A3).

    ``# frame: raw`` or ``# frame: calibrated`` as a whole-line comment
    anywhere in the file declares the frame every frequency in the file is
    expressed in -- the curation file is the one place a calibrated
    frequency becomes a DURABLE artifact (everywhere else, ``frame`` is a
    per-call argument that leaves no trace). When ``frame`` is
    ``\"calibrated\"``, the file must also carry ``# epsilon: <value>``,
    stamping the epsilon it was written under -- this is what lets a later
    apply/preview detect that the calibration has drifted (e.g. a timebase
    re-run) since the file was staged, and refuse rather than silently
    resolving against the wrong peaks (see :func:`_resolve_curation_frame`).
    ``epsilon`` without ``frame: calibrated`` is rejected at parse time: an
    epsilon stamp is meaningless without a calibrated-frame declaration to
    attach it to.

    Both directives are optional; ``frame is None`` and ``epsilon is None``
    is an ordinary file with no header, which falls back to the normal
    per-call ``frame`` resolution unchanged.
    """

    frame: Optional[Frame] = None
    epsilon: Optional[float] = None
    #: 1-based line of the ``# frame:`` / ``# epsilon:`` directive, naming the
    #: cell a refusal points at (``curation[line <n>].frame`` / ``.epsilon``).
    frame_line: Optional[int] = field(default=None, compare=False)
    epsilon_line: Optional[int] = field(default=None, compare=False)


def _curation_cell(line_no: int, column: str) -> str:
    """The ``bad_setting`` path of one curation-file cell:
    ``curation[line <n>].<column>`` (a ``# frame:`` / ``# epsilon:``
    directive is the cell ``frame`` / ``epsilon`` of its line)."""
    return f"curation[line {line_no}].{column}"


def _bad_curation_cell(
    line_no: int, column: str, expected: str, value: Any, detail: str
) -> BadSettingError:
    """A ``bad_setting`` refusal of one curation-file cell, with the
    historical ``curation line <n>: ...`` message."""
    return BadSettingError(
        _curation_cell(line_no, column),
        expected,
        value,
        message=f"curation line {line_no}: {detail}",
    )


class ParsedCurationFile(List[CurationOp]):
    """The result of :func:`parse_curation_file`: a list of the file's parsed
    :class:`CurationOp` rows (every existing ``ops = parse_curation_file(...)``
    / ``ops[i]`` / ``len(ops)`` / iteration caller keeps working exactly as
    before -- this is a list) plus the file's optional :class:`CurationFileHeader`
    (A3), attached as an attribute rather than changing the return shape."""

    def __init__(self, ops: Sequence[CurationOp], header: CurationFileHeader) -> None:
        super().__init__(ops)
        self.header = header


_CURATION_FRAME_HEADER_RE = re.compile(r"^#\s*frame\s*:\s*(.+?)\s*$", re.IGNORECASE)
_CURATION_EPSILON_HEADER_RE = re.compile(r"^#\s*epsilon\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class PlannedAction:
    """One resolved curation action (post-coalescing) ready to delegate.

    ``kind`` selects the target impl: ``edit`` -> :func:`refit_window_impl`
    (with the coalesced ``add`` / ``remove`` sets), ``merge`` ->
    :func:`merge_peaks_impl`, ``split`` -> :func:`split_peak_impl`, ``accept``
    -> :func:`review_accept_impl`.

    ``remove`` may hold ``PeakUidToken`` entries (from a ``"uid:N"`` row
    token); every other frequency-bearing field is plain ``float`` -- see
    :class:`CurationOp`.
    """

    kind: str
    window_id: int
    add: List[float] = field(default_factory=list)
    remove: List[Union[float, PeakUidToken]] = field(default_factory=list)
    peaks: List[float] = field(default_factory=list)
    peak: Optional[float] = None
    into: int = 2
    candidate: Optional[float] = None
    anchor: Optional[float] = None
    """``create`` only: the molecular frequency (MHz) the new window must cover.
    Its ``window_id`` is :data:`_NEW_WINDOW_SENTINEL` when the source did not
    name one, and otherwise the id the action is expected to produce."""
    implied_create: bool = False
    """W3. ``True`` on both halves of an implied create/edit pair -- see
    :attr:`CurationOp.implied_create`, which this is copied from by
    :func:`_resolve_curation_plan`. On the ``create`` half, its ``window_id``
    is either a fresh correlation id (a pair :func:`_resolve_curation_window_
    ids` just synthesized -- see :data:`_FIRST_IMPLIED_WINDOW_ID`) or the
    real, pinned id a decision-log replay recorded; either way the request
    (:func:`_request_rows`) and the replay (:func:`_fit_planned_create`)
    record no decision for the create and ONE ``"add"`` entry from the edit
    half instead, carrying the structural consequence on its evidence."""
    anchor_cell: Optional[str] = field(default=None, compare=False, repr=False)
    """``create`` only: the ``bad_setting`` path of the anchor as the caller
    wrote it (:attr:`CurationOp.freq_cell`), where a refused anchor is
    reported inside a batch (:func:`_raise_curation_failure`)."""


@dataclass
class PlannedWindowResult:
    """One window a curation plan installs or grows, as
    :class:`CurationApplyResult` reports it.

    The same five structural facts :class:`CreateWindowResult`,
    ``PreviewWindowResult.created_window_*`` and the ``created_window``
    decision evidence all carry (W4) -- mode, extent, grid points, frozen
    contributors, dependencies -- plus the anchor they were resolved for, and
    nothing that requires the window to have been fit. That is what lets a
    dry run fill this in from :func:`_plan_batch_create` alone while a live
    apply fills it from the create that actually ran, with both agreeing
    field for field.

    Attributes
    ----------
    window_id : int
        The window installed (``mode="created"``) or grown
        (``mode="widened"``). On a dry run this is the id the apply *would*
        mint, resolved the same way the apply resolves it.
    anchor_mhz : float
        The frequency the window was resolved to cover, raw frame -- an
        explicit ``create`` row's anchor, or the uncovered ``add`` frequency
        that implied it (W3).
    mode : str
        ``"created"`` (a new window in a gap) or ``"widened"`` (the gap was
        too narrow, so a neighbor absorbed the anchor).
    freq_range : tuple of float
        The installed or grown window's ``(min_mhz, max_mhz)`` extent, raw
        frame.
    n_points : int
        Active-FT grid points the window covers.
    n_contributors : int
        Frozen leakage contributors attached to it.
    depends_on : list of int
        Window ids it reads frozen leakage from (``[]`` for a widening, whose
        edges are unchanged).
    """

    window_id: int
    anchor_mhz: float
    mode: str
    freq_range: Tuple[float, float]
    n_points: int
    n_contributors: int
    depends_on: List[int]


@dataclass
class AppliedWindowResult:
    """One window a live curation apply touched, as
    :class:`CurationApplyResult` reports it.

    The per-window block ``ReviewPreviewResult`` already carries, narrowed to
    what an apply can report without re-deriving anything: the counts, the
    chi2r pair, and convergence. It deliberately does NOT carry the preview's
    ``peaks`` (an apply persists the final-products table, so the file is the
    place to read it) or its ``created_window_*`` fields (an apply reports
    installed structure on :attr:`CurationApplyResult.created_windows`, which
    a dry run fills too).

    The "before" side is the window as displayed when the request was
    resolved (the persisted fit); the "after" side is read off the curated
    fit the apply just persisted, so a caller can check a preview and its
    apply agreed -- field for field, on the fields both shapes carry --
    without a second read of the file.

    Attributes
    ----------
    window_id : int
        The window this entry reports on.
    origin : str
        ``"direct"`` -- some action in the plan targeted this window;
        ``"cascaded"`` -- no action named it, but the write changed its fit:
        the cascade refit it as a downstream dependent of an edited window. A
        window that is both is ``"direct"``, exactly as in
        :class:`PreviewWindowResult`.
    action_indices : list of int
        0-based indices into :attr:`CurationApplyResult.plan` of every action
        that directly targeted this window. Empty for a purely-cascaded one.
    n_peaks_before, n_peaks_after : int
        Peak count in this window before the batch / after the cascade.
    chi2r_before, chi2r_after : float or Absent
        Reduced chi-squared before the batch / after the cascade.
        ``Absent.NOT_RUN`` when that side carries no fit -- most obviously
        ``chi2r_before`` on a window the batch itself created;
        ``Absent.UNDEFINED`` when the fit's value is not finite. Absent rather
        than ``0.0`` for the reason :class:`PreviewWindowResult` documents.
    converged : bool or Absent
        Whether this window's post-cascade fit converged. ``False`` means the
        solver bailed and the window kept its seeds, so the numbers here
        describe a fit that did not happen. ``Absent.NOT_RUN`` on a window
        with no fit on the after side, exactly where ``chi2r_after`` is;
        ``Absent.UNDEFINED`` for a window left with no peaks (no solver ran).
    """

    window_id: int
    origin: str
    action_indices: List[int] = field(default_factory=list)
    n_peaks_before: int = 0
    n_peaks_after: int = 0
    chi2r_before: Union[float, Absent] = Absent.NOT_RUN
    chi2r_after: Union[float, Absent] = Absent.NOT_RUN
    converged: Union[bool, Absent] = Absent.NOT_RUN


def _applied_windows_block(
    *,
    before: Optional[SpectrumFit],
    after: SpectrumFit,
    action_indices: Dict[int, List[int]],
    direct_wids: Set[int],
    cascaded_wids: Iterable[int],
) -> Dict[int, AppliedWindowResult]:
    """Build the per-window block of a write: *before* is the displayed fit
    the request resolved against, *after* the curated fit it persisted.

    The apply-side twin of the block :func:`_run_review_preview` assembles,
    reading the same two sources in the same way, so the two rungs cannot
    come to disagree about a window they both report. Costs no file read and
    no extra fit: every input is already in hand when a write closes.
    """

    def stats(fit: Optional[SpectrumFit]) -> Dict[int, FittingResult]:
        if fit is None:
            return {}
        return {
            int(wf.window_id): wf for wf in fit.window_fits if wf.window_id is not None
        }

    before_by_wid, after_by_wid = stats(before), stats(after)
    windows: Dict[int, AppliedWindowResult] = {}
    for wid in sorted(direct_wids | set(cascaded_wids)):
        b = before_by_wid.get(wid)
        a = after_by_wid.get(wid)
        windows[wid] = AppliedWindowResult(
            window_id=wid,
            origin="direct" if wid in direct_wids else "cascaded",
            action_indices=sorted(action_indices.get(wid, [])),
            n_peaks_before=len(b.fitted_peaks) if b is not None else 0,
            n_peaks_after=len(a.fitted_peaks) if a is not None else 0,
            chi2r_before=(
                Absent.NOT_RUN if b is None else float_or_absent(float(b.reduced_chi2))
            ),
            chi2r_after=(
                Absent.NOT_RUN if a is None else float_or_absent(float(a.reduced_chi2))
            ),
            converged=Absent.NOT_RUN if a is None else _converged_or_absent(a),
        )
    return windows


def _applied_window_from_preview(pw: "PreviewWindowResult") -> AppliedWindowResult:
    """Narrow one preview entry to the apply-side shape.

    Used by the staged-preview reuse path (D4): that apply persists the
    preview's own already-cascaded fit rather than recomputing it, so its
    per-window block has to be the preview's own numbers too -- deriving them
    a second time is exactly the disagreement staging exists to rule out.
    """
    return AppliedWindowResult(
        window_id=pw.window_id,
        origin=pw.origin,
        action_indices=list(pw.action_indices),
        n_peaks_before=pw.n_peaks_before,
        n_peaks_after=pw.n_peaks_after,
        chi2r_before=pw.chi2r_before,
        chi2r_after=pw.chi2r_after,
        converged=pw.converged,
    )


@dataclass
class CurationApplyResult:
    """Outcome of :func:`apply_curation_impl`.

    Attributes
    ----------
    plan : list of PlannedAction
        The resolved, coalesced action sequence (the same in dry-run and live).
    warnings : list of str
        Advisories that do not block the apply: frequency-resolution
        advisories (ambiguous or unmatched targets) plus the A5
        frame-mismatch diagnostic (:func:`_frame_mismatch_warnings`) when the
        batch's own signature suggests it.
    applied : int
        Number of actions executed (``0`` for a dry run).
    dry_run : bool
        Whether the file was previewed without mutating.
    created_windows : list of PlannedWindowResult
        Every window this plan installs or grows, ascending by window id --
        empty for the common plan that creates none. Filled on a dry run
        *and* on a live apply, from the same five structural facts either
        way (see :class:`PlannedWindowResult`): a dry run resolves them with
        :func:`_plan_batch_create`, without fitting anything; a live apply
        reads them off the create it ran. This is BlackQuill's typo guard for
        implicit window creation (W3/W4) on the cheap rung of the
        dry-run -> preview -> apply ladder -- a caller sees "this add would
        create a window at A-B MHz" without paying for a preview.
    windows : dict of int to AppliedWindowResult
        Per-window outcome of the batch, keyed by window id -- counts, the
        chi2r pair, and convergence, for every window the plan touched
        directly or reached through the cascade. Empty on a dry run (nothing
        was fit) and on a bare-``accept`` plan (no fit touched). The
        preview's block narrowed to what an apply can report for free (see
        :class:`AppliedWindowResult`), so a caller can confirm a preview and
        its apply agreed without re-reading the file.
    base_changed : bool
        D4. ``True`` only when a :class:`ReviewSession` had a staged preview
        for this exact plan that it had to drop and recompute because the
        file's base state moved out from under it since the preview ran
        (either a foreign write, or another mutating verb issued on the same
        session in between). Always ``False`` for every sessionless caller
        (the default) -- there is nothing to have staged.
    """

    plan: List["PlannedAction"]
    warnings: List[str]
    applied: int
    dry_run: bool
    created_windows: List[PlannedWindowResult] = field(default_factory=list)
    windows: Dict[int, AppliedWindowResult] = field(default_factory=dict)
    base_changed: bool = False
    #: The stages the call invalidated (canonical names, ``rerun_order``):
    #: always ``()``, since Stage 6 invalidates no stage.
    invalidated: Tuple[str, ...] = field(default=(), compare=False)


def _parse_curation_params(raw: str, line_no: int) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise _bad_curation_cell(
                line_no,
                "params",
                "';'-separated key=value parameters",
                raw,
                f"malformed parameter {token!r} (expected key=value)",
            )
        key, _, value = token.partition("=")
        params[key.strip().lower()] = value.strip()
    return params


def parse_curation_file(curation_path: Union[Path, str]) -> ParsedCurationFile:
    """Parse a curation CSV into ordered :class:`CurationOp` rows, plus its
    optional file-level :class:`CurationFileHeader` (A3).

    Columns are ``action,window,freqs,params``, where ``action`` is one of
    ``add`` / ``remove`` / ``accept`` / ``create``. Blank lines and ``#``
    comments are ignored; an optional header row (first cell ``action``) is
    skipped. ``freqs`` is a ``;``-separated list of molecular MHz; ``params``
    is a ``;``-separated list of ``key=value`` modifiers. ``add`` and
    ``remove`` each need exactly one token per row -- a run of ``add``/
    ``remove`` rows on one window coalesces into a single refit (see
    :func:`_resolve_curation_plan`), so naming several peaks is a matter of
    writing several rows, not a longer ``freqs`` column. ``remove``'s one
    token may instead be a ``"uid:N"`` peak-identifier token (see
    :func:`~ftmwpipeline.core.curation.parse_peak_token`) -- ``remove,12,
    uid:15425022,`` removes a peak by identifier. ``add``/``create``/
    ``accept``'s ``candidate=`` are frequency-only; a ``"uid:N"`` token there
    is refused (a uid names a peak that already exists, and none of those
    targets one).

    The window column is REQUIRED on ``accept`` and ``create`` (the window is
    the operand there, not a coordinate). On ``add`` / ``remove`` it is
    OPTIONAL (W2): the same tokens ``create`` reserves for an unpinned window
    (``"new"`` / ``"auto"`` / ``"-"`` / an empty cell) mean "derive it from
    this row's own frequency (or ``uid:N``)" -- the live window whose
    ``freq_range`` covers it (windows are disjoint, so this is total and
    unambiguous), resolved by :func:`_resolve_curation_window_ids` before
    :func:`_resolve_curation_plan` coalesces by window id, so a run of
    omitted-window rows for the same window still coalesces into one edit. A
    frequency no live window covers is an error naming it, both in a live
    apply and a ``--dry-run`` preview. A *named* window is still checked: it
    is an assertion, and naming the wrong one is still an error exactly as
    before -- only an omitted window is derived.

    ``split`` and ``merge`` are not row actions: both are read from what an
    add/remove combination *does* to a window's peak set, not typed. A row
    naming either is refused, with the add/remove spelling to write instead
    (see :func:`_infer_curation_intent`, :func:`_resolve_split_step`,
    :func:`_resolve_merge_step`).

    Two ``#``-comment directives are recognized anywhere in the file and
    collected onto the returned :class:`ParsedCurationFile`'s ``.header``
    rather than being treated as ordinary comments: ``# frame: raw`` /
    ``# frame: calibrated`` declares the frame every frequency in the file is
    expressed in, and ``# epsilon: <value>`` (only valid alongside
    ``frame: calibrated``) stamps the epsilon the file was written under.
    Every other ``#``-prefixed line is an ordinary, ignored comment. See
    :func:`_resolve_curation_frame` for how the header interacts with the
    per-call ``frame`` argument.

    Raises ``BadSettingError`` (``bad_setting``, a ``ValueError``) on any
    malformed row or header directive: ``path`` is the offending cell,
    ``curation[line <n>].<column>`` (``action`` / ``window`` / ``freqs`` /
    ``params``, or ``frame`` / ``epsilon`` for a directive), and the message
    names the 1-based source line.
    """
    text = Path(curation_path).read_text()
    ops: List[CurationOp] = []
    header = CurationFileHeader()
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _CURATION_FRAME_HEADER_RE.match(line)
            if m is not None:
                raw_value = m.group(1).strip()
                value = raw_value.lower()
                if value not in ("raw", "calibrated"):
                    raise _bad_curation_cell(
                        line_no,
                        "frame",
                        "one of: raw, calibrated",
                        raw_value,
                        f"'frame' header must be 'raw' or 'calibrated', got "
                        f"{raw_value!r}",
                    )
                if header.frame is not None and header.frame != value:
                    raise _bad_curation_cell(
                        line_no,
                        "frame",
                        f"{header.frame} (the frame declared earlier in this file)",
                        raw_value,
                        f"conflicting 'frame' header (already declared "
                        f"{header.frame!r} earlier in this file)",
                    )
                if header.frame is None:
                    header.frame_line = line_no
                header.frame = value  # type: ignore[assignment]
                continue
            m = _CURATION_EPSILON_HEADER_RE.match(line)
            if m is not None:
                raw_eps = m.group(1).strip()
                try:
                    eps_value = float(raw_eps)
                except ValueError:
                    raise _bad_curation_cell(
                        line_no,
                        "epsilon",
                        "a number",
                        raw_eps,
                        f"'epsilon' header {raw_eps!r} is not a number",
                    ) from None
                if header.epsilon is not None and header.epsilon != eps_value:
                    raise _bad_curation_cell(
                        line_no,
                        "epsilon",
                        f"{header.epsilon!r} (the epsilon declared earlier in "
                        f"this file)",
                        eps_value,
                        f"conflicting 'epsilon' header (already declared "
                        f"{header.epsilon!r} earlier in this file)",
                    )
                if header.epsilon is None:
                    header.epsilon_line = line_no
                header.epsilon = eps_value
                continue
            continue  # an ordinary comment
        fields = [f.strip() for f in line.split(",")]
        action = fields[0].lower()
        if action == "action":  # header row
            continue
        expected_action = "one of: " + ", ".join(_CURATION_ACTIONS)
        if action == "split":
            raise _bad_curation_cell(
                line_no,
                "action",
                expected_action,
                fields[0],
                "'split' is not a curation-file "
                "action; write an 'add' row instead, at the frequency of the "
                "new component -- an add within snap tolerance of a fitted "
                "peak that is not itself being removed is read as a split of "
                "that peak",
            )
        if action == "merge":
            raise _bad_curation_cell(
                line_no,
                "action",
                expected_action,
                fields[0],
                "'merge' is not a curation-file "
                "action; write 'remove' rows for the mutually-close peaks to "
                "collapse, plus one 'add' row at a frequency in their span, "
                "instead -- that combination is read as a merge of them",
            )
        if action not in _CURATION_ACTIONS:
            raise _bad_curation_cell(
                line_no,
                "action",
                expected_action,
                fields[0],
                f"unknown action {fields[0]!r}; choose one of {_CURATION_ACTIONS}",
            )
        raw_window = fields[1] if len(fields) > 1 else ""
        window_token = raw_window.strip().lower()
        if action == "create" and window_token in _CURATION_NEW_WINDOW_TOKENS:
            window_id = _NEW_WINDOW_SENTINEL
        elif (
            action in ("add", "remove") and window_token in _CURATION_NEW_WINDOW_TOKENS
        ):
            # W2: the window is a coordinate on add/remove, derived from the
            # row's own frequency (or "uid:N") by live-window coverage --
            # see _resolve_curation_window_ids, run between this parse and
            # _resolve_curation_plan's coalescing.
            window_id = _DERIVE_WINDOW_SENTINEL
        elif len(fields) < 2 or not fields[1]:
            raise _bad_curation_cell(
                line_no,
                "window",
                f"a window id (required on {action})",
                fields[1] if len(fields) > 1 else None,
                "missing window id",
            )
        else:
            try:
                window_id = int(fields[1])
            except ValueError:
                omit = ", or one of: new, auto, -" if action != "accept" else ""
                raise _bad_curation_cell(
                    line_no,
                    "window",
                    f"an integer window id{omit}",
                    fields[1],
                    f"window id {fields[1]!r} is not an integer",
                ) from None
        freqs_raw = fields[2] if len(fields) > 2 else ""
        params_raw = fields[3] if len(fields) > 3 else ""
        freq_tokens = [x for x in freqs_raw.split(";") if x.strip()]
        freqs: List[Union[float, PeakUidToken]]
        freqs_cell = _curation_cell(line_no, "freqs")
        if action == "remove":
            # The one action whose single token may be a "uid:N"
            # peak-identifier instead of a frequency -- see parse_peak_token.
            try:
                freqs = [parse_peak_token(x, path=freqs_cell) for x in freq_tokens]
            except BadSettingError as exc:
                raise BadSettingError(
                    exc.path,
                    exc.expected,
                    exc.value,
                    message=f"curation line {line_no}: {exc}",
                ) from None
        else:
            try:
                freqs = [float(x) for x in freq_tokens]
            except ValueError:
                raise _bad_curation_cell(
                    line_no,
                    "freqs",
                    "a frequency in MHz",
                    freqs_raw,
                    f"non-numeric frequency in {freqs_raw!r}",
                ) from None
        params = _parse_curation_params(params_raw, line_no)

        # Per-action arity / parameter validation.
        if action in ("add", "remove"):
            if len(freqs) != 1:
                one = (
                    'exactly one frequency in MHz or peak identifier "uid:N"'
                    if action == "remove"
                    else "exactly one frequency in MHz"
                )
                raise _bad_curation_cell(
                    line_no,
                    "freqs",
                    one,
                    freqs_raw,
                    f"{action} needs exactly one frequency",
                )
            if params:
                raise _bad_curation_cell(
                    line_no,
                    "params",
                    "no parameters (an empty cell)",
                    params_raw,
                    f"{action} takes no parameters",
                )
        elif action == "create":
            if len(freqs) != 1:
                raise _bad_curation_cell(
                    line_no,
                    "freqs",
                    "exactly one frequency in MHz (the anchor)",
                    freqs_raw,
                    "create needs exactly one frequency "
                    "(the anchor the new window must cover)",
                )
            if params:
                raise _bad_curation_cell(
                    line_no,
                    "params",
                    "no parameters (an empty cell)",
                    params_raw,
                    "create takes no parameters",
                )
        elif action == "accept":
            if freqs:
                raise _bad_curation_cell(
                    line_no,
                    "freqs",
                    "an empty cell (revive a candidate with params candidate=F)",
                    freqs_raw,
                    "accept takes no frequency column; "
                    "use params candidate=F to revive a candidate",
                )
            if "candidate" in params:
                try:
                    float(params["candidate"])
                except ValueError:
                    raise _bad_curation_cell(
                        line_no,
                        "params",
                        "candidate=<a frequency in MHz>",
                        params_raw,
                        f"candidate={params['candidate']!r} is not a number",
                    ) from None

        ops.append(
            CurationOp(
                action=action,
                window_id=window_id,
                freqs=freqs,
                params=params,
                line_no=line_no,
                freq_cell=_curation_cell(line_no, "freqs"),
            )
        )

    if header.epsilon is not None and header.frame != "calibrated":
        assert header.epsilon_line is not None
        raise BadSettingError(
            _curation_cell(header.epsilon_line, "epsilon"),
            "no epsilon header unless the file declares '# frame: calibrated'",
            header.epsilon,
            message="curation file: an 'epsilon' header requires a 'frame: "
            "calibrated' header alongside it -- an epsilon stamp is "
            "meaningless without a calibrated-frame declaration to attach "
            "it to",
        )
    if header.frame == "calibrated" and header.epsilon is None:
        assert header.frame_line is not None
        raise BadSettingError(
            _curation_cell(header.frame_line, "frame"),
            "raw, or calibrated with an '# epsilon:' header",
            header.frame,
            message="curation file: 'frame: calibrated' requires an 'epsilon' "
            "header stamping the epsilon the file was written under (e.g. "
            "'# epsilon: 2.2e-6') -- otherwise a later apply/preview cannot "
            "detect that the calibration has drifted since this file was "
            "staged",
        )

    return ParsedCurationFile(ops, header)


def _curation_ops_have_freq(ops: Sequence[CurationOp]) -> bool:
    """Whether any parsed row carries a frequency needing frame resolution.

    Mirrors :func:`_planned_action_has_freq`'s predicate, but at the
    pre-coalesce ``CurationOp`` level: window derivation (W2) needs the raw
    frame (window ranges are stored raw) and must run *before*
    :func:`_resolve_curation_plan` coalesces, so the frame has to be resolved
    this early too.
    """
    return any(
        op.freqs or (op.action == "accept" and "candidate" in op.params) for op in ops
    )


# ---------------------------------------------------------------------------
# Curation as data: CurationAction <-> CurationOp.
#
# ``review_apply`` / ``review_preview`` take either a curation-file path or a
# sequence of :class:`~ftmwpipeline.core.curation.CurationAction` -- one
# grammar, two spellings. Actions become exactly the ``CurationOp`` rows the
# parser would produce for the same file, except that each action's
# frequencies are converted to raw up front with its OWN resolved frame (a
# batch may mix frames), and the batch then enters the file path's
# resolution as a raw-frame batch. From there on nothing knows which
# spelling the caller used.
# ---------------------------------------------------------------------------

CurationSource = Union[Path, str, Tuple[CurationAction, ...]]
"""What a curation call resolves: a curation-file path, or a tuple of
actions (normalized by :func:`curation_source`)."""


def curation_source(
    curation_path: Optional[Union[Path, str]],
    actions: Optional[Iterable[Union[CurationAction, Mapping[str, Any]]]],
) -> CurationSource:
    """Validate a curation call's ``curation_path`` / ``actions`` pair and
    return the one that was given.

    Exactly one must be given, else ``bad_setting`` (``path`` ``"actions"``).
    ``actions`` may hold :class:`CurationAction` instances or their wire
    dicts (:meth:`CurationAction.from_dict`); it is returned as a tuple of
    actions. An empty sequence is a valid, empty batch -- as an empty
    curation file is.
    """
    if (curation_path is None) == (actions is None):
        given = "both" if actions is not None else "neither"
        raise BadSettingError(
            "actions",
            "exactly one of curation_path or actions",
            None if actions is None else "<actions>",
            message=f"pass exactly one of curation_path (a curation file) or "
            f"actions (a sequence of CurationAction); got {given}",
        )
    if actions is None:
        assert curation_path is not None
        if not isinstance(curation_path, (str, Path)):
            # A natural slip now that curation_path is optional: a list of
            # actions passed positionally.
            raise BadSettingError(
                "curation_path",
                "a path to a curation file (pass in-memory actions as actions=)",
                curation_path,
                message=f"curation_path must be a file path, got "
                f"{type(curation_path).__name__}; pass a sequence of "
                f"CurationAction as actions=...",
            )
        return curation_path
    if isinstance(actions, (str, bytes, Mapping)):
        raise BadSettingError(
            "actions",
            "a sequence of CurationAction (or their dicts)",
            actions,
            message="actions must be a sequence of CurationAction (or their "
            "dicts), not a single string or mapping",
        )
    out: List[CurationAction] = []
    for item in actions:
        if isinstance(item, CurationAction):
            out.append(item)
        elif isinstance(item, Mapping):
            try:
                out.append(CurationAction.from_dict(item))
            except BadSettingError as exc:
                raise _action_field_error(len(out), exc) from None
        else:
            raise BadSettingError(
                f"actions[{len(out)}]",
                "a CurationAction or a curation action dict",
                item,
                message=f"actions[{len(out)}] is not a CurationAction or a "
                f"curation action dict: {item!r}",
            )
    return tuple(out)


def _action_field_error(
    index: int, exc: BadSettingError, field_name: Optional[str] = None
) -> BadSettingError:
    """Re-issue a refusal of one action's field with the batch path
    ``actions[<index>].<field>`` (``field_name`` defaults to the refusal's
    own path, which names the field) and the action named in the message."""
    name = exc.path if field_name is None else field_name
    return BadSettingError(
        f"actions[{index}].{name}",
        exc.expected,
        exc.value,
        message=f"actions[{index}]: {exc}",
    )


def _action_has_freq(action: CurationAction) -> bool:
    """Whether *action* carries a frequency needing a frame (``peak_uid`` and
    a bare accept do not)."""
    return action.freq_mhz is not None or action.candidate_mhz is not None


def _action_to_op(
    action: CurationAction,
    index: int,
    *,
    frame: Frame,
    stamp: Optional[_CalibrationStamp],
) -> CurationOp:
    """One action as the :class:`CurationOp` row :func:`parse_curation_file`
    produces for the same row, with its frequencies converted from *frame*
    to raw by the same :func:`_frame_to_raw` the file path uses.

    ``line_no`` is the action's 1-based position in the batch (it only ever
    labels a diagnostic).
    """
    if action.window_id is not None:
        window_id = action.window_id
    elif action.action == "create":
        window_id = _NEW_WINDOW_SENTINEL
    else:
        # accept requires a window (CurationAction validates it), so only
        # add/remove reach here: the file's "auto".
        window_id = _DERIVE_WINDOW_SENTINEL
    freqs: List[Union[float, PeakUidToken]] = []
    if action.peak_uid is not None:
        freqs.append(PeakUidToken(action.peak_uid))
    elif action.freq_mhz is not None:
        freqs.append(_frame_to_raw(action.freq_mhz, frame=frame, stamp=stamp))
    params: Dict[str, str] = {}
    if action.candidate_mhz is not None:
        # repr round-trips a float exactly; _resolve_curation_plan reads it
        # back with float(), as it does a file's candidate= text.
        raw_candidate = _frame_to_raw(action.candidate_mhz, frame=frame, stamp=stamp)
        params["candidate"] = repr(raw_candidate)
    return CurationOp(
        action=action.action,
        window_id=window_id,
        freqs=freqs,
        params=params,
        line_no=index + 1,
        freq_cell=f"actions[{index}].freq_mhz",
    )


def _actions_to_ops(
    path: str, actions: Sequence[CurationAction], frame: Optional[Frame]
) -> Tuple[ParsedCurationFile, Optional[_CalibrationStamp], FrozenSet[float]]:
    """Convert a batch of actions to raw-frame :class:`CurationOp` rows.

    Each action's frame is resolved on its own: its ``frame``, else the
    call's ``frame``, else :func:`_resolve_frame`'s rule (raw on an
    ``epsilon == 0`` file; ``bad_setting`` ``path`` ``"frame"`` on a
    ``self_calibrated`` file) -- only for an action that carries a
    frequency. Returns ``(ops, stamp, raw_targets)``: the ops are raw and
    carry no header, so they enter :func:`_resolve_curation_ops` with
    ``frame="raw"``. ``raw_targets`` holds the frequencies of the actions
    that resolved raw -- the ones the A5 frame-mismatch advisory judges
    (:func:`_frame_mismatch_warnings`), action by action, so a raw action is
    still diagnosed when others in the batch are calibrated.
    """
    stamp: Optional[_CalibrationStamp] = None
    stamp_read = False
    raw_targets: Set[float] = set()
    ops: List[CurationOp] = []
    for index, action in enumerate(actions):
        resolved: Frame = "raw"
        if _action_has_freq(action):
            if not stamp_read:
                stamp = _current_calibration_stamp(path)
                stamp_read = True
            if action.frame is not None and frame is not None and action.frame != frame:
                # The same refusal a file whose header disagrees with frame=
                # gets (_resolve_curation_frame), pointed at the action field.
                raise BadSettingError(
                    f"actions[{index}].frame",
                    f'"{frame}" (the frame passed as frame=), or null',
                    action.frame,
                    message=f"actions[{index}] ({action.action}) declares "
                    f'frame="{action.frame}", but frame="{frame}" was passed '
                    f"explicitly. Pass a matching frame, or omit one of them.",
                )
            requested = action.frame if action.frame is not None else frame
            try:
                resolved = _resolve_frame_against(stamp, requested, path)
            except BadSettingError as exc:
                # No frame at all (neither the action's nor the call's): the
                # call's frame= is what is missing, as for a file without a
                # header, so the path stays "frame"; the message names the
                # action.
                raise BadSettingError(
                    exc.path,
                    exc.expected,
                    exc.value,
                    message=f"actions[{index}] ({action.action}): {exc}",
                ) from None
            if resolved == "raw":
                raw_targets.update(
                    float(f)
                    for f in (action.freq_mhz, action.candidate_mhz)
                    if f is not None
                )
            # A stamped epsilon is the file header's drift check, per action.
            if action.epsilon is not None and stamp is not None:
                current_eps = stamp[1]
                if not math.isclose(
                    action.epsilon, current_eps, rel_tol=1e-6, abs_tol=1e-12
                ):
                    raise BadSettingError(
                        f"actions[{index}].epsilon",
                        f"the file's current epsilon ({current_eps:.6e})",
                        action.epsilon,
                        message=f"actions[{index}] ({action.action}) frame "
                        f"drift: it was staged frame=calibrated at epsilon="
                        f"{action.epsilon:.6e}, but the target file's current "
                        f"epsilon is {current_eps:.6e}. The calibration has "
                        f"changed since it was staged (e.g. a timebase re-run) "
                        f"-- re-stage it against the current calibration.",
                    )
        ops.append(_action_to_op(action, index, frame=resolved, stamp=stamp))
    return ParsedCurationFile(ops, CurationFileHeader()), stamp, frozenset(raw_targets)


def action_from_op(
    op: CurationOp,
    frame: Optional[Frame] = None,
    epsilon: Optional[float] = None,
) -> CurationAction:
    """The :class:`CurationAction` a parsed curation-file row spells.

    The inverse of :func:`_action_to_op` at ``frame="raw"``: the derive / new
    window sentinels become ``window_id=None``, a ``uid:N`` token becomes
    ``peak_uid``, and ``candidate=F`` becomes ``candidate_mhz``. *frame* is
    the frame the row's frequencies are in (a file's ``# frame:`` header);
    it is attached only to an action that carries a frequency. Parameters an
    action has no field for (an ``accept`` row's keys other than
    ``candidate``, which the file path ignores too) are dropped.
    """
    window_id: Optional[int] = op.window_id
    if op.window_id in (_NEW_WINDOW_SENTINEL, _DERIVE_WINDOW_SENTINEL):
        window_id = None
    freq_mhz: Optional[float] = None
    peak_uid: Optional[int] = None
    if op.freqs:
        token = op.freqs[0]
        if isinstance(token, PeakUidToken):
            peak_uid = token.uid
        else:
            freq_mhz = float(token)
    cand = op.params.get("candidate") if op.action == "accept" else None
    candidate_mhz = None if cand is None else float(cand)
    has_freq = freq_mhz is not None or candidate_mhz is not None
    return CurationAction(
        action=op.action,  # type: ignore[arg-type]
        window_id=window_id,
        freq_mhz=freq_mhz,
        peak_uid=peak_uid,
        candidate_mhz=candidate_mhz,
        frame=frame if has_freq else None,
        epsilon=epsilon if has_freq and frame == "calibrated" else None,
    )


def actions_from_curation_file(
    curation_path: Union[Path, str],
) -> List[CurationAction]:
    """Parse a curation file into :class:`CurationAction`\\ s -- the same
    actions ``CurationAction.from_dict`` gives of their dicts.

    Each frequency-bearing action carries the file's ``# frame:`` header (or
    ``None`` without one) and, when calibrated, its ``# epsilon:`` stamp, so
    the actions are checked for drift exactly as the file is.
    """
    ops = parse_curation_file(curation_path)
    return [action_from_op(op, ops.header.frame, ops.header.epsilon) for op in ops]


def _resolve_curation_window_ids(
    ops: Sequence[CurationOp],
    path: str,
    *,
    frame: Frame,
    stamp: Optional[_CalibrationStamp],
    coverage: Optional[Sequence[FitWindowCoverage]] = None,
) -> List[CurationOp]:
    """Resolve every add/remove row's omitted window token (parsed to
    :data:`_DERIVE_WINDOW_SENTINEL`) to the live window its frequency (or
    ``"uid:N"`` identifier) resolves to.

    Runs BEFORE :func:`_resolve_curation_plan`, which coalesces a run of
    add/remove rows *by window id* -- resolving after coalescing would leave
    several same-window, omitted-window rows ungrouped (they would not share
    a window id to coalesce on), costing one refit per row instead of one for
    the run. See :func:`_window_for_curation_token` for the shared,
    live-windows-only resolver (THE RULE: never a global nearest-peak
    search).

    A ``remove`` (frequency or ``"uid:N"``) that no live window covers is an
    error naming it and the line -- PERMANENT: a remove never implies a
    create.

    W3: an ``add`` whose frequency (converted to the raw frame first, since
    window ranges are stored raw) is covered by no live window instead
    IMPLIES a create -- this ONE row is expanded into TWO ops sharing a fresh
    negative correlation id (:data:`_FIRST_IMPLIED_WINDOW_ID`) as their
    ``window_id`` and ``implied_create=True``: a ``create`` op anchored at
    the row's own frequency, immediately followed by the original ``add`` op
    (window id swapped from the sentinel to the correlation id). Because the
    correlation id is unique to this one pair, :func:`_resolve_curation_plan`
    -- UNCHANGED -- coalesces the ``add`` half into its own dedicated
    ``PlannedAction`` exactly as it would any other window's row, and
    :func:`_canonicalize_batch_plan` -- also unchanged -- still hoists the
    ``create`` half first; only the resolution (:func:`_resolve_action`)
    knows the id is a placeholder and resolves it to whatever the create
    actually mints (or widens) before resolving the edit. A row that already
    names a window -- including a ``"new"``/``"auto"`` token on a ``create``
    row, which is the unrelated :data:`_NEW_WINDOW_SENTINEL` -- is returned
    unchanged.

    ``coverage`` is the live-window index to resolve against; omitted, it is
    read from *path*. A log-prefix apply passes the index of the fit its kept
    prefix describes, computed in memory (:func:`_resolve_ops_against`),
    since the file holds the dropped decisions too.
    """
    if not any(op.window_id == _DERIVE_WINDOW_SENTINEL for op in ops):
        return list(ops)
    if coverage is None:
        coverage = _load_curation_window_index(path)
    resolved: List[CurationOp] = []
    next_implied_id = _FIRST_IMPLIED_WINDOW_ID
    unknown_uids: List[int] = []
    uncovered: List[float] = []
    uncovered_details: List[str] = []
    for op in ops:
        if op.window_id != _DERIVE_WINDOW_SENTINEL:
            resolved.append(op)
            continue
        token = op.freqs[0]
        if isinstance(token, PeakUidToken):
            wid = _window_for_curation_token(coverage, token)
            if wid is None:
                if token.uid not in unknown_uids:
                    unknown_uids.append(token.uid)
                continue
            resolved.append(replace(op, window_id=wid))
            continue
        freq_raw = _frame_to_raw(float(token), frame=frame, stamp=stamp)
        wid = _window_for_curation_token(coverage, freq_raw)
        if wid is None:
            if op.action != "add":
                # Raw, like every refusal under the call; the call reports it
                # in the frame the caller wrote (_caller_frame_ids).
                if freq_raw not in uncovered:
                    uncovered.append(freq_raw)
                uncovered_details.append(
                    f"curation line {op.line_no}: {op.action} "
                    f"{float(token):.4f} MHz is not covered by any live "
                    f"window (windows are disjoint); nothing to remove "
                    f"there"
                )
                continue
            # W3: implied create -- see the docstring above.
            correlation_id = next_implied_id
            next_implied_id -= 1
            resolved.append(
                CurationOp(
                    action="create",
                    window_id=correlation_id,
                    freqs=[token],
                    params={},
                    line_no=op.line_no,
                    implied_create=True,
                    freq_cell=op.freq_cell,
                )
            )
            resolved.append(replace(op, window_id=correlation_id, implied_create=True))
            continue
        resolved.append(replace(op, window_id=wid))
    if unknown_uids:
        raise _unknown_peak_uids_error(unknown_uids)
    if uncovered:
        raise NotFoundValueError(
            "window", uncovered, message="; ".join(uncovered_details)
        )
    return resolved


def _unknown_peak_uids_error(uids: Sequence[int]) -> NotFoundValueError:
    """The ``not_found`` refusal for peak identifiers no fitted peak carries,
    naming every one of them (a batch reports all its unknown ids at once)."""
    listed = ", ".join(f"peak_uid={u}" for u in uids)
    return NotFoundValueError(
        "peak",
        list(uids),
        message=f"no fitted peak with {listed} in any window (already removed, "
        f"or the identifier is wrong)",
    )


def _unknown_plan_window_ids(
    known: Collection[int],
    plan: Sequence[PlannedAction],
    plan_window_ids: Optional[Collection[int]] = None,
) -> List[int]:
    """Every window id *plan* names that is not in *known* and that no
    ``create`` of the plan can install, in plan order.

    ``create`` rows name no existing window; a negative id is a placeholder
    (:data:`_NEW_WINDOW_SENTINEL`, an implied-create correlation id). A
    pinned ``create`` installs its own id. A ``create`` without a pinned id
    mints an id it cannot know yet -- one above every window of the plan and
    the window-id high-water mark, so above every *known* id and every pinned
    create -- and a later row may
    legitimately name it: such an id is left to the per-action lookup once
    the creates have run. Given *plan_window_ids* (the window plan the
    creates mint against, with the high-water mark), the creates are replayed in plan order to find
    exactly the ids they can mint (an unfitted plan window above every fitted
    one is neither known nor mintable). Every other unknown id cannot exist
    and is reported here, all at once, even when the plan holds an unpinned
    create.
    """
    live = set(known) | {
        int(a.window_id) for a in plan if a.kind == "create" and a.window_id >= 0
    }
    n_mints = sum(
        1 for a in plan if a.kind == "create" and a.window_id == _NEW_WINDOW_SENTINEL
    )
    # Given the window plan the creates mint against, replay the creates in
    # plan order: a minting create (unpinned or fresh implied) takes one above
    # every window present when it runs -- every known and planned window (an
    # unfitted plan window above the fitted ones is neither known nor
    # mintable) and every create before it; a pinned create installs its own
    # id. That gives exactly the ids the batch can mint.
    mintable: Optional[Set[int]] = None
    if plan_window_ids is not None:
        mintable = set()
        top = max(set(known) | set(plan_window_ids), default=-1)
        for a in plan:
            if a.kind != "create":
                continue
            if a.window_id >= 0:
                top = max(top, int(a.window_id))
            elif a.window_id == _NEW_WINDOW_SENTINEL or _is_implied_window_id(
                a.window_id
            ):
                top += 1
                mintable.add(top)
    mint_floor = max(live, default=-1)
    unknown: List[int] = []
    for a in plan:
        wid = int(a.window_id)
        if a.kind == "create" or wid < 0 or wid in live or wid in unknown:
            continue
        if mintable is not None:
            if wid in mintable:
                continue
        elif n_mints and wid > mint_floor:
            # Without the window plan, any id above the known ones may be minted.
            continue
        unknown.append(wid)
    return unknown


def _require_known_plan_windows(
    known: Collection[int],
    plan: Sequence[PlannedAction],
    where: str,
    plan_window_ids: Optional[Collection[int]] = None,
) -> None:
    """Refuse a plan naming windows the fit does not have, all of them at once
    (see :func:`_unknown_plan_window_ids`)."""
    unknown = _unknown_plan_window_ids(known, plan, plan_window_ids)
    if unknown:
        listed = ", ".join(str(w) for w in unknown)
        raise NotFoundValueError(
            "window",
            unknown,
            message=f"window_id={listed} not found in the {where}",
        )


def _batch_known_window_ids(
    ctx: "_BatchCtx", plan: Sequence[PlannedAction], fit_ids: Set[int]
) -> Set[int]:
    """The ids a batch's plan may name: the fit's windows, plus every flagged
    lineless window (``ctx.changeset.lineless_reviewable``) a *bare* accept of
    the plan names -- the one action such a window takes, since the fit holds
    nothing there to edit."""
    lineless = ctx.changeset.lineless_reviewable
    return set(fit_ids) | {
        int(a.window_id)
        for a in plan
        if a.kind == "accept" and a.candidate is None and int(a.window_id) in lineless
    }


def _raise_curation_failure(
    index: int, action: PlannedAction, exc: Exception
) -> NoReturn:
    """Re-raise a failure of plan action *index*, tagged with the action.

    A typed :class:`PipelineFileError` keeps its type (a program routes on
    it); a ``not_found`` or ``curation_conflict`` is re-issued with the tag in
    its message and the same attributes, and a create's refused anchor
    (``bad_setting`` ``anchor_mhz``) with the tag and the path of the cell or
    action field the anchor came from. Anything else becomes the historical
    tagged :class:`ValueError`.
    """
    tag = f"curation action {index + 1} ({describe_planned_action(action)}) failed"
    if (
        isinstance(exc, BadSettingError)
        and exc.path == "anchor_mhz"
        and action.kind == "create"
        and action.anchor_cell is not None
    ):
        # A create's anchor is the cell (or action field) the caller wrote,
        # not review_create's argument.
        raise BadSettingError(
            action.anchor_cell, exc.expected, exc.value, message=f"{tag}: {exc}"
        ) from exc
    if isinstance(exc, NotFoundError):
        # The batch refused with ValueError before it was typed.
        raise NotFoundValueError(exc.kind, exc.ids, message=f"{tag}: {exc}") from exc
    if isinstance(exc, CurationConflictError):
        raise CurationConflictError(
            exc.reason, exc.ids, message=f"{tag}: {exc}"
        ) from exc
    if isinstance(exc, PipelineFileError):
        raise exc
    raise ValueError(f"{tag}: {exc}") from exc


def _assert_plain_freq(token: Union[float, PeakUidToken]) -> float:
    """Narrow a :class:`CurationOp` token to a plain frequency.

    Every action but ``remove`` is frequency-only by construction --
    :func:`parse_curation_file` only ever produces a ``PeakUidToken`` on a
    ``remove`` row. This both proves that invariant to mypy at each non-``remove``
    call site below and guards it at runtime, matching this module's
    preference for guards enforced by structure over a remembered
    convention.
    """
    assert isinstance(
        token, float
    ), f"expected a plain frequency, got a peak identifier: {token!r}"
    return token


def _resolve_curation_plan(ops: Sequence[CurationOp]) -> List[PlannedAction]:
    """Coalesce parsed ops into the delegated action plan.

    A maximal run of ``add`` / ``remove`` rows on one window collapses into a
    single ``edit`` (one refit instead of one per row); a ``merge`` / ``split``
    / ``accept`` on that window is a barrier that flushes the window's pending
    edit first (it changes the peak set with its own seeding). Windows are
    independent, so an edit on another window does not flush a pending group.

    ``create`` never coalesces: it installs structure the rows after it name, so
    it stands alone and in order.

    W3: ``op.implied_create`` is copied onto the resulting :class:`PlannedAction`
    unchanged -- this function does not otherwise know or care what it means,
    only that it rides along (see :attr:`PlannedAction.implied_create`). An
    implied pair's unique correlation id (or, on replay, its real pinned id)
    means no OTHER op can ever land in the same ``pending`` group, so the flag
    set when a group is first opened is always the group's only value.
    """
    plan: List[PlannedAction] = []
    pending: Dict[int, PlannedAction] = {}
    pending_order: List[int] = []

    def flush(wid: int) -> None:
        pa = pending.pop(wid, None)
        if wid in pending_order:
            pending_order.remove(wid)
        if pa is not None and (pa.add or pa.remove):
            plan.append(pa)

    for op in ops:
        wid = op.window_id
        if op.action == "create":
            plan.append(
                PlannedAction(
                    kind="create",
                    window_id=wid,
                    anchor=_assert_plain_freq(op.freqs[0]),
                    implied_create=op.implied_create,
                    anchor_cell=op.freq_cell,
                )
            )
            continue
        if op.action in ("add", "remove"):
            pa = pending.get(wid)
            if pa is None:
                pa = PlannedAction(
                    kind="edit", window_id=wid, implied_create=op.implied_create
                )
                pending[wid] = pa
                pending_order.append(wid)
            if op.action == "add":
                pa.add.append(_assert_plain_freq(op.freqs[0]))
            else:
                # remove's one token per row may be a plain frequency or a
                # PeakUidToken (see CurationOp.freqs); either becomes this
                # row's removal target.
                pa.remove.append(op.freqs[0])
            continue
        # Barrier for this window.
        flush(wid)
        if op.action == "merge":
            plan.append(
                PlannedAction(
                    kind="merge",
                    window_id=wid,
                    peaks=[_assert_plain_freq(f) for f in op.freqs],
                )
            )
        elif op.action == "split":
            plan.append(
                PlannedAction(
                    kind="split",
                    window_id=wid,
                    peak=_assert_plain_freq(op.freqs[0]),
                    into=int(op.params.get("into", 2)),
                )
            )
        elif op.action == "accept":
            cand = op.params.get("candidate")
            plan.append(
                PlannedAction(
                    kind="accept",
                    window_id=wid,
                    candidate=float(cand) if cand is not None else None,
                )
            )
    for wid in list(pending_order):
        flush(wid)
    return plan


def _fmt_remove_token(token: Union[float, PeakUidToken]) -> str:
    """Render one ``remove`` target for a human-readable summary."""
    return f"uid:{token.uid}" if isinstance(token, PeakUidToken) else f"{token:.4f}"


def describe_planned_action(action: PlannedAction) -> str:
    """Render one :class:`PlannedAction` as a one-line human-readable summary."""
    wid = action.window_id
    if action.kind == "create":
        # A fresh W3 correlation id (_is_implied_window_id) is never a real
        # window id, just like _NEW_WINDOW_SENTINEL -- both read as "a new
        # window" here. A REPLAYED implied create's window_id is already the
        # real, pinned id, so it prints exactly like any other pinned create.
        target = (
            "a new window"
            if wid == _NEW_WINDOW_SENTINEL or _is_implied_window_id(wid)
            else f"window {wid}"
        )
        return f"create {target}: anchor {float(action.anchor or 0.0):.4f}"
    if action.kind == "edit":
        wid_label = (
            "the window this add creates"
            if _is_implied_window_id(wid)
            else f"window {wid}"
        )
        parts = []
        if action.add:
            parts.append("add " + ", ".join(f"{f:.4f}" for f in action.add))
        if action.remove:
            parts.append(
                "remove " + ", ".join(_fmt_remove_token(t) for t in action.remove)
            )
        return f"edit {wid_label}: " + "; ".join(parts)
    if action.kind == "merge":
        return f"merge window {wid}: peaks " + ", ".join(
            f"{f:.4f}" for f in action.peaks
        )
    if action.kind == "split":
        return f"split window {wid}: peak {action.peak:.4f} into {action.into}"
    if action.kind == "accept":
        if action.candidate is not None:
            return f"accept window {wid}: candidate {action.candidate:.4f}"
        return f"accept window {wid}"
    return f"{action.kind} window {wid}"


def _fitted_freqs_by_window(path: str) -> Dict[int, List[float]]:
    """Molecular MHz of each window's persisted fitted peaks (empty if no fit).

    A cheap per-window column read (see
    :func:`~ftmwpipeline.io.fitting_serialization.read_fit_peak_frequencies_by_window`)
    rather than a full fit load -- window_id is always set on a persisted
    ``FittingResult`` (it is a required on-disk attribute), so the full
    loader's now-unreachable-in-practice ``wf.window_id is None`` skip has no
    cheap-path equivalent to reproduce.
    """
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return {}
        return read_fit_peak_frequencies_by_window(h5f["stage5_fitting"])


def _fitted_uids_by_window(path: str) -> Dict[int, Set[int]]:
    """``peak_uid`` of each window's persisted fitted peaks (empty if no fit).

    Parallels :func:`_fitted_freqs_by_window` -- a separate accessor rather
    than folding this into it, since most callers of that one want only
    frequencies. A peak from a fit predating ``peak_uid`` (``peak_uid is
    None``) contributes nothing to its window's set, which is what makes a
    ``"uid:N"`` dry-run check against such a window correctly report the uid
    as unmatched rather than crashing on ``None``. Reads only the
    ``peak_uid`` column per window (see
    :func:`~ftmwpipeline.io.fitting_serialization.read_fit_peak_uids_by_window`).
    """
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return {}
        return read_fit_peak_uids_by_window(h5f["stage5_fitting"])


def _fitted_peak_index(
    path: str, plan: Sequence[PlannedAction]
) -> Tuple[Dict[int, List[float]], Dict[int, Set[int]]]:
    """The per-window fitted frequencies, and the uids only if *plan* needs them.

    One place decides how much of the peak table a batch's advisories have to
    read, so the two consumers of that decision -- the ambiguity pass and the
    frame-mismatch diagnostic -- cannot each pay for their own walk of the
    window groups.

    A ``"uid:N"`` remove target is the only thing that needs the ``peak_uid``
    column; most plans carry none, and those get the frequency column alone
    and an empty uid map. A plan that does carry one gets both columns from a
    single walk (:func:`~ftmwpipeline.io.fitting_serialization.read_fit_peak_freqs_and_uids_by_window`)
    rather than one walk each -- the values are identical either way.
    """
    has_uid_target = any(
        action.kind == "edit"
        and any(isinstance(t, PeakUidToken) for t in action.remove)
        for action in plan
    )
    if not has_uid_target:
        return _fitted_freqs_by_window(path), {}
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return {}, {}
        return read_fit_peak_freqs_and_uids_by_window(h5f["stage5_fitting"])


def _planned_window_ranges(path: str) -> Dict[int, Tuple[float, float]]:
    """``freq_range`` of every window in the effective plan, low bound first.

    The fitted plan's bounds (:func:`~._internal.fitted_plan.fitted_window_bounds`:
    the Stage 4 plan's, or the plan a structurally merged fit stored) plus the
    Stage 6 created-window overlay -- a three-column read,
    rather than :func:`effective_window_plan` -- a full plan load also
    materializes every window's free-peak index list, its fixed-contributor
    objects and its per-window JSON diagnostics, which is 120 ms of the 124
    this used to cost on 2638 and none of which a bounds map reads.

    Applying the overlay by dict update is what makes this equal to the full
    path: :func:`_overlay_created_windows` *replaces* a base window carrying
    the same id and *appends* a fresh one, and keyed by ``window_id`` those
    two cases are the same assignment. The rest of what the overlay computes
    -- plan ordering, the spliced dependency edges, the topological order --
    does not survive into a bounds map, so it is not built here.

    Like the cheap fit readers this validates only what it reads, where the
    full loader validated every window's datasets and column lengths; a plan
    group the full loader would refuse can yield ranges here. That divergence
    is the same one those readers took knowingly.

    Best-effort: a file without a Stage 4 plan yields an empty map rather than
    raising, so the advisory pass degrades to the checks it can still make.
    """
    from .fitted_plan import fitted_window_bounds

    try:
        with h5open(path, "r") as h5f:
            fitted = fitted_window_bounds(h5f)
        if fitted is None:
            return {}
        ranges = {
            wid: (min(lo, hi), max(lo, hi)) for wid, (lo, hi) in fitted[0].items()
        }
        for created in load_stage6_review_from_file(path).created_windows:
            ranges[int(created.window_id)] = (
                min(created.freq_range),
                max(created.freq_range),
            )
        return ranges
    except Exception:  # pragma: no cover - advisory only, never fatal
        return {}


def _curation_ambiguity_warnings(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    snap_tol_mhz: float,
    index: Optional[Tuple[Dict[int, List[float]], Dict[int, Set[int]]]] = None,
    planned_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[str]:
    """Advisories where a curation action will not resolve against the file.

    ``remove`` / ``split`` / ``merge`` match an *existing* fitted peak by nearest
    frequency within ``snap_tol_mhz``; if two peaks sit within tolerance the
    matcher's pick is ambiguous, and if none do the edit will fail. A
    ``remove``'s ``"uid:N"`` token instead needs an *exact* ``peak_uid`` match
    in the named window -- no snap, no ambiguity, just present or absent --
    and is checked accordingly.

    ``add`` creates a peak, so it has no target to match -- but it does have a
    target *window*, and that is exactly what goes stale: window ids are
    reassigned by a Stage 4 re-plan, and a plan window whose peaks all failed
    their Stage 5 gate carries no fit to edit at all (a large fraction of the
    plan on a line-dense file). So an ``add`` is checked for the two conditions
    :func:`refit_window_impl` will later enforce: the window is live, and the
    frequency lies on its data. The range test allows ``snap_tol_mhz`` of slack
    on each side because the seed may snap that far onto a ledger candidate --
    it flags only what cannot land in the window however it snaps, so it never
    cries wolf on an edge case that would in fact succeed. Accepted candidates
    are not checked: the window's own ledger supplies the frequency.

    All of this is advisory. It exists so that ``--dry-run`` previews the
    failures a live apply would hit instead of only some of them.

    ``index`` is the ``(frequencies, uids)`` pair from
    :func:`_fitted_peak_index`, which a caller that also runs the
    frame-mismatch diagnostic passes in so the two passes share one read.
    Omitted, it is built here -- the uid half only when the plan carries a
    ``"uid:N"`` target, which most do not.
    """
    by_window, by_uid = _fitted_peak_index(path, plan) if index is None else index
    if planned_ranges is None:
        planned_ranges = _planned_window_ranges(path)
    warnings: List[str] = []

    # A ``create`` in this same plan installs the window that a later ``add``
    # names, and its geometry is not derivable without running the planner, so
    # adds into it are left to the live apply. An unpinned create's id is not
    # even known here, so any otherwise-unresolvable window could be it.
    created_ids = {
        a.window_id
        for a in plan
        if a.kind == "create" and a.window_id != _NEW_WINDOW_SENTINEL
    }
    has_unpinned_create = any(
        a.kind == "create" and a.window_id == _NEW_WINDOW_SENTINEL for a in plan
    )

    def check_add(wid: int, freq: float, what: str) -> None:
        if wid in created_ids:
            return
        if wid not in by_window:
            if has_unpinned_create:
                return
            if wid in planned_ranges:
                warnings.append(
                    f"{what}: window {wid} is in the Stage 4 plan but carries no "
                    f"Stage 5 fit (every peak in it failed its gate), so there "
                    f"is nothing to add to (the edit will fail); create a window "
                    f"at this frequency instead"
                )
            else:
                warnings.append(f"{what}: no window {wid} exists (the edit will fail)")
            return
        window_range = planned_ranges.get(wid)
        if window_range is None:
            return
        lo, hi = window_range
        if not (lo - snap_tol_mhz <= freq <= hi + snap_tol_mhz):
            warnings.append(
                f"{what}: {freq:.4f} MHz is outside window {wid}'s range "
                f"[{lo:.4f}, {hi:.4f}] MHz (the edit will fail); name the window "
                f"that covers it, or create one if none does"
            )

    def check(wid: int, freq: float, what: str) -> None:
        fitted = by_window.get(wid)
        if fitted is None:
            warnings.append(f"{what}: window {wid} has no fitted peaks")
            return
        near = sorted(f for f in fitted if abs(f - freq) <= snap_tol_mhz)
        if not near:
            warnings.append(
                f"{what}: no fitted peak within {snap_tol_mhz * 1e3:.0f} kHz of "
                f"{freq:.4f} MHz in window {wid} (the edit will fail)"
            )
        elif len(near) > 1:
            near_str = ", ".join(f"{f:.4f}" for f in near)
            warnings.append(
                f"{what}: {freq:.4f} MHz in window {wid} is within "
                f"{snap_tol_mhz * 1e3:.0f} kHz of {len(near)} fitted peaks "
                f"({near_str}); the nearest is taken"
            )

    def check_uid(wid: int, uid: int, what: str) -> None:
        # The frequency map separates "no peaks" from "no such peak". (A fit
        # whose peaks carry no peak_uid is refused before any of this runs:
        # _require_engine_file.)
        if not by_window.get(wid):
            warnings.append(
                f"{what}: window {wid} has no fitted peaks (the edit will fail)"
            )
            return
        if uid not in (by_uid.get(wid) or set()):
            warnings.append(
                f"{what}: no fitted peak with peak_uid={uid} in window {wid} "
                f"(the edit will fail)"
            )

    for action in plan:
        wid = action.window_id
        if action.kind == "edit":
            for f in action.remove:
                if isinstance(f, PeakUidToken):
                    # Exact peak_uid match, not a nearest-frequency snap --
                    # checked by identity rather than by check()'s tolerance
                    # logic.
                    check_uid(wid, f.uid, f"remove uid:{f.uid}")
                    continue
                check(wid, f, f"remove {f:.4f}")
            for f in action.add:
                check_add(wid, f, f"add {f:.4f}")
        elif action.kind == "merge":
            for f in action.peaks:
                check(wid, f, f"merge {f:.4f}")
        elif action.kind == "split" and action.peak is not None:
            check(wid, action.peak, f"split {action.peak:.4f}")
    return warnings


# ---------------------------------------------------------------------------
# A5: frame-mismatch diagnostic -- advisory only, never a refusal.
#
# When a curation file was actually staged in the calibrated frame but
# declared (or defaulted to) raw, every remove/merge/split/accept-candidate
# frequency still resolves -- to the right peak, via the ordinary snap
# tolerance -- but lands off by the omitted conversion:
# ``(probe - f_raw) * eps / (1 + eps)``, per candidate (see
# ``TestFrameConversionArithmetic`` / ``TestConversionBeforeSnapping`` in
# ``test_frame_parameter.py`` for the same arithmetic on a single call). A
# whole BATCH doing this in lockstep -- several candidates, all displaced the
# same direction, each by very nearly what THIS file's own current epsilon
# predicts for its own matched frequency -- is a signature an honestly-raw
# batch practically never produces by chance. That specificity (not just "a
# common offset", but the one *this file's calibration* predicts) is what
# keeps the false-positive rate low; see :func:`_frame_mismatch_warnings`.
# ---------------------------------------------------------------------------

_FRAME_MISMATCH_MIN_CANDIDATES = 3
"""Below this many matched candidates, a coincidental near-hit is too easy;
require the diagnostic to explain several independent candidates at once."""

_FRAME_MISMATCH_REL_TOL = 0.25
"""Each residual must land within +/-25% of what this file's OWN epsilon
predicts for its own matched frequency -- a specific, precomputed value, not
merely "some common offset". A genuine hand-typed batch practically never
lands every candidate this close to a value it has no way to know."""

_FRAME_MISMATCH_FLOOR_BINS = 0.05
"""Minimum |predicted offset|, as a fraction of the active-FT bin spacing, to
even consider a candidate. Guards the vanishingly-small-epsilon regime, where
the predicted offset is smaller than ordinary NLS refit jitter (itself a
sub-bin quantity) and indistinguishable from a correctly raw-declared batch --
firing there would be pure noise, not signal. Added 2026-08-18; not part of
the family of constants recovered from a pre-existing nominal-80-kHz design
(see ``scratch/bin-relative-constants-plan.md``) -- 0.05 is a fresh,
reasonable round bin fraction, not a recovered value. Resolved to MHz in
:func:`_frame_mismatch_warnings` via :func:`_persisted_acquisition_us`;
falls back to 0.0 when no Stage 5 fit is persisted, which is moot in
practice since ``by_window`` is then empty and the function returns early."""


def _persisted_acquisition_us(path: str) -> float:
    """Persisted Stage 5 active-region acquisition length (us), or 0.0 absent
    a fit. The active-FT bin spacing every spectral-distance tolerance in
    this module resolves against is ``1 / acquisition_us``
    (:func:`~ftmwpipeline.fitting.active_ft.active_ft_bin_spacing_mhz`).

    Reads only the ``parameters`` JSON attr on ``/stage5_fitting`` (via
    :func:`~ftmwpipeline.io.fitting_serialization.read_fit_parameters`)
    rather than the whole persisted fit -- this is the one value out of the
    ~30 attrs/datasets per window a full load would pull that this function
    ever looks at.
    """
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return 0.0
        parameters = read_fit_parameters(h5f["stage5_fitting"])
    return float(parameters.get("acquisition_us", 0.0))


def _frame_mismatch_warnings(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    raw_targets: Optional[FrozenSet[float]],
    stamp: Optional[_CalibrationStamp],
    fitted_freqs: Optional[Dict[int, List[float]]] = None,
) -> List[str]:
    """Advisory-only diagnostic (A5): flag a batch whose candidates all
    resolve with a residual consistent with a calibrated-frame curation file
    that was declared (or defaulted to) raw.

    Never raises and never blocks anything -- this is a heuristic, and a
    heuristic that refused would be worse than none. Returns ``[]`` (inert)
    unless ALL of the following hold:

    - the candidate was submitted in the raw frame -- if the caller
      correctly declared ``calibrated``, the conversion already happened and
      no systematic residual should remain to diagnose. This is judged per
      action: ``raw_targets`` holds the (raw) frequencies of the actions
      that resolved raw, and only those candidates are judged, so a batch
      mixing frames is still diagnosed on its raw actions. ``None`` means
      every candidate (a curation file whose one frame is raw); an empty set
      means none (a calibrated file);
    - the file is ``self_calibrated`` with a nonzero epsilon -- inert on
      every ``rb_locked``/``uncalibrated`` file, where epsilon is always
      ``0.0`` and the two frames coincide;
    - at least :data:`_FRAME_MISMATCH_MIN_CANDIDATES` of the batch's
      ``remove`` / ``merge`` peaks / ``split`` peak / ``accept`` candidate
      frequencies (the ones that resolve against an *existing* fitted peak --
      ``add`` and ``create`` have no such target and are excluded) land
      within :data:`_FRAME_MISMATCH_REL_TOL` of the offset THIS file's
      current epsilon predicts for that exact candidate
      (``(probe - f_matched) * eps / (1 + eps)``), with the predicted
      magnitude clearing the :data:`_FRAME_MISMATCH_FLOOR_BINS` floor;
    - every one of those residuals shares the same sign -- a real omitted
      conversion pushes every candidate the same direction; independently
      mistyped or mis-snapped frequencies would not.

    ``fitted_freqs`` is :func:`_fitted_freqs_by_window`'s map, threaded in by
    a caller that already built it (the apply path builds it for the
    ambiguity pass moments earlier). Omitted -- the preview path, which runs
    no ambiguity pass -- it is read here, and only after the cheap
    disqualifying checks above have failed to return, so an inert diagnostic
    still reads nothing.
    """
    if stamp is None or (raw_targets is not None and not raw_targets):
        return []
    cal_state, epsilon, _sigma_eps, _floor_khz, probe_freq_mhz, _sideband = stamp
    if cal_state != "self_calibrated" or epsilon == 0.0:
        return []

    acquisition_us = _persisted_acquisition_us(path)
    bin_spacing_mhz = (
        active_ft_bin_spacing_mhz(acquisition_us) if acquisition_us > 0.0 else 0.0
    )
    frame_mismatch_floor_mhz = _FRAME_MISMATCH_FLOOR_BINS * bin_spacing_mhz
    # The same snap the verbs will use, resolved from the same spacing -- this
    # heuristic decides which candidates *would* have paired, so a tolerance of
    # its own would diagnose a batch nobody is going to run.
    snap_tol_mhz = REFIT_SNAP_TOL_BINS * bin_spacing_mhz

    by_window = _fitted_freqs_by_window(path) if fitted_freqs is None else fitted_freqs

    def nearest(wid: int, freq: float) -> Optional[float]:
        fitted = by_window.get(wid)
        if not fitted:
            return None
        best = min(fitted, key=lambda f: abs(f - freq))
        if abs(best - freq) > snap_tol_mhz:
            return None
        return best

    residuals: List[float] = []
    predicted: List[float] = []
    flagged: List[int] = []
    for action_index, action in enumerate(plan):
        wid = action.window_id
        targets: List[float] = []
        if action.kind == "edit":
            # A PeakUidToken carries no frequency of its own -- it always
            # matches its named peak exactly regardless of frame, so it has
            # no residual to contribute to this heuristic and is excluded.
            targets.extend(f for f in action.remove if isinstance(f, float))
        elif action.kind == "merge":
            targets.extend(action.peaks)
        elif action.kind == "split" and action.peak is not None:
            targets.append(action.peak)
        elif action.kind == "accept" and action.candidate is not None:
            targets.append(action.candidate)
        for f in targets:
            if raw_targets is not None and f not in raw_targets:
                continue  # submitted calibrated: already converted
            match = nearest(wid, f)
            if match is None:
                continue
            residuals.append(f - match)
            predicted.append((probe_freq_mhz - match) * epsilon / (1.0 + epsilon))
            if action_index not in flagged:
                flagged.append(action_index)

    if len(residuals) < _FRAME_MISMATCH_MIN_CANDIDATES:
        return []

    for r, p in zip(residuals, predicted):
        if abs(p) < frame_mismatch_floor_mhz:
            return []
        lo = abs(p) * (1.0 - _FRAME_MISMATCH_REL_TOL)
        hi = abs(p) * (1.0 + _FRAME_MISMATCH_REL_TOL)
        if not (lo <= abs(r) <= hi):
            return []
        if (r > 0) != (p > 0):
            return []

    mean_residual_khz = (sum(residuals) / len(residuals)) * 1e3
    advisory = (
        f"{len(residuals)} candidate(s) in this batch resolved with a "
        f"residual clustered near {mean_residual_khz:+.1f} kHz -- matching "
        f"what this file's epsilon ({epsilon * 1e6:+.3f} ppm) predicts for a "
        f"calibrated frequency submitted as raw. The curation file may have "
        f"been staged in the calibrated frame but declared (or defaulted to) "
        f"raw; double check its frame before trusting this batch."
    )
    # The same advisory as a frame_mismatch event of the running operation
    # (``actions``: the 0-based plan positions whose candidates matched). It
    # stays in the result's warnings too; it has no log line.
    scope = _REVIEW_SCOPE.get()
    if scope is not None:
        scope.warn("frame_mismatch", advisory, actions=sorted(flagged))
    return [advisory]


# --- the automatic-fit baseline (for undo replay) --------------------------
#
# A Stage 6 write rewrites ``/stage5_fitting``, so the automatic fit it
# replaced is otherwise unrecoverable. Before the first write (``_open_batch``)
# we snapshot the automatic fit into ``/stage5_fitting_baseline``, and every
# write replays its decision log from it (``_curate``). A fresh Stage 5 fit
# drops the snapshot (``clear_stage5_baseline``) so the next write
# re-snapshots.

STAGE5_BASELINE_GROUP = "stage5_fitting_baseline"
# Decision kinds that mutate ``/stage5_fitting`` and therefore need the
# automatic-fit baseline to be undoable. ``create_window`` belongs here: it
# splices a window fit into the persisted SpectrumFit (and, on the widening
# path, re-fits an existing window over a grown extent).
_FIT_EDIT_KINDS = ("add", "remove", "merge", "split", "create_window")

#: The decision kinds whose rows address peaks: each carries the uids it
#: removes (``targets``) and the seeds and uids of the peaks it births
#: (``seeds_mhz``, ``born_uids``).
_PEAK_ROW_KINDS = ("add", "remove", "merge", "split")


def _log_dirty_window_ids(entries: Sequence[DecisionLogEntry]) -> Set[int]:
    """The windows *entries* edit: every window with an add, remove, merge or
    split row (an accept with a candidate is recorded as an add), and every
    window a create widened. A plain create and a bare accept change no fit
    a dependent reads, so neither makes its window dirty."""
    out: Set[int] = set()
    for e in entries:
        if e.kind in ("add", "remove", "merge", "split") or (
            e.kind == "create_window" and e.evidence.get("mode") == "widened"
        ):
            out.add(int(e.window_id))
    return out


def _installs_window(entry: DecisionLogEntry) -> bool:
    """Whether *entry* is one of the log's create rows: a ``create_window``
    row, or an ``add`` carrying ``created_window`` evidence (an implied
    create's one-row shape, which replays as a create pinned to its id
    followed by the add, :func:`_replay_action_groups`)."""
    return entry.kind == "create_window" or (
        entry.kind == "add" and entry.evidence.get("created_window") is not None
    )


def _create_row_mode(entry: DecisionLogEntry) -> Optional[str]:
    """The ``mode`` a create row recorded (``"created"`` or ``"widened"``);
    ``None`` for a row that is not a create row or recorded no mode."""
    if entry.kind == "create_window":
        mode = entry.evidence.get("mode")
    elif _installs_window(entry):
        mode = entry.evidence["created_window"].get("mode")
    else:
        return None
    return None if mode is None else str(mode)


def _check_created_ids_monotone(path: str, entries: Sequence[DecisionLogEntry]) -> None:
    """Refuse a log whose ``mode="created"`` rows do not mint strictly
    increasing window ids along the log.

    A fresh create always mints above every id the lineage has minted
    (:attr:`~ftmwpipeline.core.data_structures.Stage6Review.window_id_high_water`),
    and only the engine writes the log, so a violation means the file is
    corrupt: a replay would pin a created window under an id an older
    window's skirt could not have read. Widening rows carry base ids and are
    not checked.
    """
    last: Optional[int] = None
    for e in entries:
        if _create_row_mode(e) != "created":
            continue
        wid = int(e.window_id)
        if last is not None and wid <= last:
            raise PipelineCorruptionError(
                Path(path),
                f"the decision log creates window {wid} after window {last}: "
                "created window ids must increase along the log",
            )
        last = wid


#: Attr of the baseline group naming the curation lineage it starts. Stamped
#: when the engine takes the snapshot; a baseline without one was taken by a
#: pre-engine build.
LINEAGE_ID_ATTR = "lineage_id"


def _snapshot_stage5_baseline(path: str, lineage_id: Optional[str] = None) -> None:
    """Copy ``/stage5_fitting`` to the baseline group if not already
    snapshotted, stamping *lineage_id* on it (a fresh one when ``None``)."""
    with h5open(path, "a") as h5f:
        if "stage5_fitting" in h5f and STAGE5_BASELINE_GROUP not in h5f:
            h5f.copy("stage5_fitting", STAGE5_BASELINE_GROUP)
            h5f[STAGE5_BASELINE_GROUP].attrs[LINEAGE_ID_ATTR] = (
                lineage_id if lineage_id is not None else uuid.uuid4().hex
            )


_REFIT_INSTRUCTION = (
    "Re-run 'fit run' to start a new curation lineage: it writes a fit Stage 6 "
    "can curate and discards this file's curation, which then has to be redone."
)

_UPGRADE_INSTRUCTION = (
    "A newer ftmwpipeline curated this file: upgrade ftmwpipeline to change its "
    "curation ('fit run' would discard it)."
)


def refit_required_instruction(reason: str) -> str:
    """What a user does about a file Stage 6 refuses to write, given the
    :func:`_refit_required_reason` slug: upgrade for ``"file_incompatible"``,
    else re-run ``fit run``."""
    if reason == "file_incompatible":
        return _UPGRADE_INSTRUCTION
    return _REFIT_INSTRUCTION


def _refit_required_reason(path: str) -> Optional[str]:
    """The reason a Stage 6 write of *path* is refused with, or ``None`` when
    writes are accepted. Read-only: attrs and one column.

    ``"file_incompatible"``: a *newer* engine wrote the review
    (:func:`_require_engine_file` raises ``file_incompatible``); upgrading is
    the remedy, so it wins over everything else. The rest are
    ``curation_conflict`` reasons whose remedy is ``fit run``:
    ``"predates_peak_identity"``: a fitted peak carries no ``peak_uid`` (a fit
    written before peak identity was persisted), so no decision can address
    it. ``"predates_replay_engine"``: the review was written by a build without
    this replay engine (no ``engine_version``, or an older one), or the undo
    baseline was taken by one (no lineage id); the file's curation was recorded
    under other replay semantics. The first wins when both hold.
    """
    with h5open(path, "r") as h5f:
        review = h5f.get("stage6_review")
        version = None if review is None else review.attrs.get("engine_version")
        if version is not None and int(version) > ENGINE_VERSION:
            return "file_incompatible"
        if "stage5_fitting" in h5f and not fit_has_peak_identity(h5f["stage5_fitting"]):
            return "predates_peak_identity"
        if review is not None:
            if version is None or int(version) < ENGINE_VERSION:
                return "predates_replay_engine"
        baseline = h5f.get(STAGE5_BASELINE_GROUP)
        if baseline is not None and LINEAGE_ID_ATTR not in baseline.attrs:
            return "predates_replay_engine"
    return None


def refit_required_impl(file_path: Union[Path, str]) -> Optional[str]:
    """:func:`_refit_required_reason` for a read path: ``None`` too for a file
    that is not HDF5 (a read of it returns the empty review)."""
    try:
        return _refit_required_reason(str(file_path))
    except OSError:
        return None


def _require_engine_file(path: str) -> None:
    """Refuse a Stage 6 write of a file this engine cannot curate, before
    anything is resolved, fitted or written (design: refuse and flag).

    ``file_incompatible`` when a newer engine wrote the review; otherwise
    ``curation_conflict`` with the :func:`_refit_required_reason` reason.
    """
    reason = _refit_required_reason(path)
    if reason is None:
        return
    if reason == "file_incompatible":
        with h5open(path, "r") as h5f:
            version = int(h5f["stage6_review"].attrs["engine_version"])
        raise PipelineCompatibilityError(
            Path(path),
            f"Stage 6 replay engine {version}",
            f"Stage 6 replay engine {ENGINE_VERSION}",
        )
    what = (
        "its Stage 5 fit predates peak identity (a fitted peak has no peak_uid), "
        "so no decision can address its peaks"
        if reason == "predates_peak_identity"
        else "it was curated by a build that predates the Stage 6 replay engine"
    )
    raise CurationConflictError(
        reason,
        [],
        message=f"cannot write Stage 6 review state to this file: {what}. "
        f"{_REFIT_INSTRUCTION}",
    )


def clear_stage5_baseline(path: Union[Path, str]) -> None:
    """Drop the automatic-fit baseline snapshot (a fresh fit supersedes it),
    and with it the engine keys computed from it (``/stage6_engine``)."""
    with h5open(str(path), "a") as h5f:
        for group in (STAGE5_BASELINE_GROUP, STAGE6_ENGINE_GROUP):
            if group in h5f:
                del h5f[group]


def _has_stage5_baseline(path: str) -> bool:
    with h5open(path, "r") as h5f:
        return STAGE5_BASELINE_GROUP in h5f


# ---------------------------------------------------------------------------
# Batch curation engine: build the shared fit context once, hold a curated
# state in memory, cascade once, persist once.
#
# Running each action through its own read-modify-write of the file would
# reload the FID, rebuild the active-FT context and rewrite
# ``/stage5_fitting`` per action -- N redundant imports/FTs/persists for a
# batch of N. The engine builds that shared state ONCE (``_SharedFitCtx``,
# reusable across a review session's requests), resolves a request against a
# display batch, replays the decision log in a replay batch, runs a single
# combined cascade, and persists ``/stage5_fitting`` and ``/stage6_review``
# once each (the one write path, :func:`_curate`).
#
# Ordering: the final persisted state must not depend on the order actions
# were listed in the curation file. Cross-window order is canonicalized when a
# request is resolved -- creates first (in their own relative order, since
# they install structure later rows name), then every other action grouped by
# ascending window id -- while the intra-window sequence
# ``_resolve_curation_plan`` already coalesced is preserved exactly (a stable
# sort by window id cannot reorder two actions that share one). The rows are
# appended to the log in this canonical order, so the log itself is
# order-of-specification-independent; the log is then replayed as recorded,
# in log order (:func:`_resolve_replay_rows`).
# ---------------------------------------------------------------------------


def _fit_group_fingerprint(path: str) -> Optional[Tuple[str, int, int]]:
    """A cheap stamp identifying the exact contents of ``/stage5_fitting``.

    ``(creation_time, n_windows, n_fitted_peaks)``, three attrs on the group
    itself -- no window walk, no dataset read.
    :func:`~ftmwpipeline.io._hdf5_helpers.stamp_stage_header` re-stamps
    ``creation_time`` on *every* write of the fit, whether the full writer or
    the incremental one, so any rewrite changes this; the two counts
    corroborate it.

    Deliberately scoped to the fit group rather than to the file. The
    file-level fingerprint :class:`ReviewSession` uses cannot serve here: the
    undo-baseline snapshot opens the file in append mode before a batch reads
    the fit, which moves the file's mtime without changing the fit at all,
    and a whole-file stamp would therefore never match twice in a row.

    ``None`` when the file carries no fit -- never equal to itself, so a
    cache stamped against it can never be reused.
    """
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return None
        attrs = h5f["stage5_fitting"].attrs
        raw_time = attrs.get("creation_time", "")
        if isinstance(raw_time, bytes):
            raw_time = raw_time.decode("utf-8")
        return (
            str(raw_time),
            int(attrs.get("n_windows", -1)),
            int(attrs.get("n_fitted_peaks", -1)),
        )


@dataclass
class _FitCache:
    """A working :class:`SpectrumFit` carried across a session's verbs (S5).

    Every batch used to reload the whole fit from disk (415 ms on a
    251-window build) even when the previous verb in the same session had
    just written that exact object. This holds it instead. It rides on
    :class:`_SharedFitCtx` because their lifetimes are identical: a
    sessionless caller builds a fresh shared context per call and therefore
    always finds this empty, so nothing outside a session changes behavior
    or timing.

    **The cache validates itself; nobody has to remember to invalidate it.**
    :meth:`take` re-reads :func:`_fit_group_fingerprint` and refuses the
    cached fit unless the fit on disk is still the one that was cached. That
    is not belt-and-braces: a write outside this session (or a foreign
    writer) can rewrite ``/stage5_fitting`` between two of its verbs, and the
    write stamp catches that with no special case, as it would catch the
    next such writer too.

    Ownership transfers on :meth:`take`: the caller mutates the object it
    receives, so the cache drops its own reference at the same moment. A
    batch that then fails, or one that is never persisted (a preview),
    simply leaves the cache empty and the next verb reloads -- the stale
    object cannot come back.
    """

    _fit: Optional[SpectrumFit] = None
    _fingerprint: Optional[Tuple[str, int, int]] = None

    def take(self, path: str) -> Optional[SpectrumFit]:
        """The cached fit if ``/stage5_fitting`` is still the one it was
        cached from, else ``None``. Always clears the cache either way.
        """
        fit, fingerprint = self._fit, self._fingerprint
        self._fit = None
        self._fingerprint = None
        if fit is None or fingerprint is None:
            return None
        if _fit_group_fingerprint(path) != fingerprint:
            return None
        return fit

    def install(self, fit: SpectrumFit, path: str) -> None:
        """Cache *fit* as the contents of ``/stage5_fitting`` as of now.

        Called immediately after the fit has been written, while the stamp
        on disk is still the one that write left.
        """
        self._fit = fit
        self._fingerprint = _fit_group_fingerprint(path)


@dataclass
class _SharedFitCtx:
    """Batch-invariant state derived from the file: resolved settings,
    calibration, and above all ``fit_ctx`` -- the active-FT reconstruction
    (:func:`~.stage5_impl.build_stage5_fit_context`) this whole engine exists
    to amortize. Nothing here depends on which windows a batch's actions
    touch or on any edit a batch makes, so it is safe to build once and reuse
    across many batches (a later unit does exactly that for preview); nothing
    on this object is ever mutated after :func:`_build_shared_fit_ctx`
    returns it.
    """

    resolved: "StageFitSettings"
    shape_enum: "PeakShape"
    persisted_cal: object
    tau_maj_global: Optional[float]
    sigma_tau_global: Optional[float]
    tau_source: str
    fit_ctx: "Stage5FitContext"
    peaks_loaded: List["Peak"]
    peak_frequencies_mhz: List[float]
    min_freeze_snr: float
    base_plan: "WindowPlan"
    calibration_state: str
    """``\"rb_locked\"`` / ``\"self_calibrated\"`` / ``\"uncalibrated\"``, as
    :func:`_derive_frequency_calibration` reads it at the moment this context
    was built (batch-invariant like everything else here). Stamped on every
    :class:`RefitWindowResult` / :class:`CreateWindowResult` this batch
    returns (A6) -- read via :func:`_current_calibration_stamp`, never
    re-derived per applier call."""
    epsilon: float
    """The fractional timebase scale error actually applied (``0.0`` unless
    ``calibration_state == \"self_calibrated\"``)."""
    sigma_epsilon: float
    """1-sigma uncertainty on :attr:`epsilon` (``0.0`` when inapplicable)."""
    fit_cache: _FitCache = field(default_factory=_FitCache)
    """The one mutable slot on this object (S5), and deliberately so: it
    holds no *derived* state, only a copy of what is already on disk, and
    every read of it is gated on a fingerprint read fresh from the file.
    Everything else here is still built once and never touched again."""
    retired_window_ids: FrozenSet[int] = frozenset()
    """Ids a Stage 5 structural merge absorbed: a create never mints one, and
    a replayed create cannot take one (:class:`~.fitted_plan.FittedPlan`)."""
    unavailable_window_ids: FrozenSet[int] = frozenset()
    """Windows of a merged fit that predates the stored fitted plan, whose
    fitted geometry is not in the file: refitting one is refused
    (``fit_plan_unavailable``). Empty for every other fit."""
    unavailable_spans_mhz: Tuple[Tuple[float, float], ...] = ()
    """The merged ranges of such a fit; a create overlapping one is refused."""
    base_cascade_sources: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    """``E_base``, the cascade graph over the baseline-live windows, as each
    one's ordered source list (:func:`_base_cascade_sources`): read from the
    fitted plan and the undo baseline, so it is fixed for the lineage. Its
    keys are the windows the automatic fit holds."""
    baseline_fits: Mapping[int, FittingResult] = field(default_factory=dict)
    """The automatic fit's window fits (the undo baseline's), by window id: a
    created window's starting skirt is read from them
    (:func:`_created_window_seed`), never from a curated fit, so it does not
    depend on where the create sits in the log. Never mutated."""


@dataclass
class _BatchChangeset:
    """The mutable state of one batch: the working ``spectrum_fit``, the
    created-window overlay and the effective plan's window map, the windows
    the batch edited and created, and the decisions it recorded.

    A batch is one of two things. A *display* batch holds the persisted
    curated state a request resolves against (:func:`_build_batch_ctx`):
    nothing is fit in it. A *replay* batch computes the full-replay reference
    from the automatic fit (:func:`_replay_log`): ``replayed`` is the log it
    replays, and every refit, create and cascade of a write runs there.

    ``created_windows`` and ``fit_window_map`` live here rather than on
    :class:`_SharedFitCtx`: a create the batch resolves or replays appends to
    ``created_windows`` and recomputes ``fit_window_map`` in place
    (``_install_planned_create``).
    """

    spectrum_fit: SpectrumFit
    created_windows: List["FitWindow"]
    fit_window_map: Dict[int, "FitWindow"] = field(default_factory=dict)
    dirty_wids: set = field(default_factory=set)
    """Windows a replayed edit (add/remove/merge/split, or a widening)
    changed -- the seed set for the one combined cascade."""
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    """What the replay recorded for each row of :attr:`replayed`, in order
    (``window_id`` / ``kind`` / ``evidence``): the evidence a row new to the
    log is recorded with (:func:`_replayed_rows`)."""
    replayed: List["DecisionLogEntry"] = field(default_factory=list)
    """The log a replay batch replays, in log order. The peaks a row births
    are stamped with its serial."""
    base_serial: int = 0
    """The file's serial high-water mark when a display batch was built: the
    serial the first decision the request records takes."""
    lineless_reviewable: FrozenSet[int] = frozenset()
    """Windows the persisted review flagged ``empty_window_residual`` that the
    fit has no result for: a bare accept may name one (it records a decision
    and changes no fit)."""
    created_wids: set = field(default_factory=set)
    """Windows a ``mode="created"`` create of the replay installed."""
    window_id_high_water: int = -1
    """The highest window id a create has minted in the lineage: the file's
    :attr:`~ftmwpipeline.core.data_structures.Stage6Review.window_id_high_water`
    when the batch was built, raised by each create the request resolves. A
    fresh create mints above it (:func:`_plan_batch_create`)."""

    def pending_serial(self, ahead: int = 0) -> int:
        """The serial of the replayed row *ahead* rows past the ones recorded
        so far -- what a birth's ``derivation`` is stamped with."""
        entry = self.replayed[len(self.decisions) + ahead]
        if isinstance(entry.serial, Absent):
            raise ValueError(
                f"internal: replayed decision at log position "
                f"{entry.order_index} carries no serial"
            )
        return int(entry.serial)


@dataclass
class _BatchCtx:
    """One batch's full working state: the (possibly reused) shared context
    plus this batch's own changeset. Every resolver, applier and cascade
    helper below addresses fields through ``ctx.shared.*`` /
    ``ctx.changeset.*`` -- the split is deliberately visible at every call
    site, since one ``_SharedFitCtx`` serves many ``_BatchChangeset``s (a
    display batch and a replay batch per write, and every write of a
    session).
    """

    shared: _SharedFitCtx
    changeset: _BatchChangeset
    events: Optional[StageScope] = None
    """The Stage 6 operation's scope (:data:`_REVIEW_SCOPE` when a replay
    batch was built), through which the replay reports its windows and checks
    for a cancel; ``None`` reports nothing and never cancels."""


def _build_shared_fit_ctx(
    path: str, *, fit_group: str = "stage5_fitting"
) -> _SharedFitCtx:
    """Load and resolve everything every batch's fit-mutating actions share:
    settings, calibration, and the active-FT context
    (:func:`~.stage5_impl.build_stage5_fit_context`, the expensive
    FID-load-and-FT step this whole engine exists to amortize). Safe to build
    once and reuse across many batches -- nothing it returns depends on any
    batch's edits.

    ``fit_group`` names the fit group the spur catalog and the fitted plan are
    read from: the full-replay reference (:mod:`.replay_reference`) reads them
    from the undo baseline, since it reads no curated state; the write path
    reads them from ``/stage5_fitting``, which holds the same ones (a write
    rewrites the window fits only).
    """
    from ..core.stage_fit_settings import ShapeSpec, StageFitSettings
    from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
    from ..fitting.peak_model import PeakShape
    from ..io.stage_fit_settings_serialization import (
        load_stage_fit_settings_from_h5,
        read_recommended_clock_sources,
        read_stage2b_recommended_shape,
    )
    from ..preprocessing.window_planning import DEFAULT_MIN_FREEZE_SNR
    from .fitted_plan import load_fitted_plan
    from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
    from .stage3_impl import load_peaks_impl
    from .stage5_impl import (
        Stage5FitContext,
        _resolve_tau_calibration_for_fit,
        build_stage5_fit_context,
        gated_spur_catalog,
    )

    with h5open(path, "r") as h5f:
        if fit_group not in h5f:
            raise StageDependencyError(
                "review",
                ["stage5_fitting"],
                Path(str(path)),
                command="fit run",
                message="No Stage 5 fit found in this file. Run 'fit run' first.",
            )
        # The ``parameters`` and ``diagnostics`` attrs alone, read only to
        # seed the spur-catalog replay below -- never the whole fit, which is
        # per-batch state that each batch reloads for itself (see
        # ``_BatchChangeset``) and which this function has always deliberately
        # declined to return. The spur catalog is a Stage 5 product Stage 6
        # never rewrites, so reading it here, once, is not the staleness risk
        # that retaining the fit would be. ``diagnostics`` holds the catalog
        # at full precision (``parameters`` rounds the centers for display).
        spur_catalog: Dict[str, Any] = gated_spur_catalog(
            read_fit_parameters(h5f[fit_group]),
            read_fit_diagnostics(h5f[fit_group]),
        )

    # The plan the fit was made on: a structural merge's survivor is refit on
    # its merged range, and an absorbed id is not a window.
    fitted = load_fitted_plan(path, fit_group_name=fit_group)
    base_plan: "WindowPlan" = fitted.plan

    peaks_loaded = load_peaks_impl(path)["peaks"]
    peak_frequencies_mhz = [float(p.frequency) for p in peaks_loaded]

    persisted_settings = load_stage_fit_settings_from_h5(path)
    recommended_shape_str = read_stage2b_recommended_shape(path)
    recommended_clocks = read_recommended_clock_sources(path)
    recommended_settings: Optional[StageFitSettings] = None
    if recommended_shape_str is not None or recommended_clocks is not None:
        from ..core.stage_fit_settings import SpurSubSettings

        recommended_settings = StageFitSettings(
            shape=(
                ShapeSpec.coerce(recommended_shape_str)
                if recommended_shape_str is not None
                else None
            ),
            spur=SpurSubSettings(clocks=recommended_clocks),
        )
    resolved = resolve_stage_fit_settings(
        explicit=StageFitSettings(),
        preset=None,
        persisted=persisted_settings,
        recommended=recommended_settings,
    )
    assert resolved.shape is not None
    shape_enum = resolved.shape.kind

    persisted_cal = None
    if shape_enum is PeakShape.GAUSSIAN:
        if tau_calibration_present(path, shape="gaussian"):
            persisted_cal = load_tau_calibration_impl(path, shape="gaussian")[
                "tau_calibration"
            ]
    else:
        if tau_calibration_present(path):
            persisted_cal = load_tau_calibration_impl(path)["tau_calibration"]
    tau_maj_global, sigma_tau_global, tau_source = _resolve_tau_calibration_for_fit(
        persisted_cal,
        resolved.tau.tau_maj_override_us,
        resolved.tau.sigma_tau_override_us,
    )

    # Replay the persisted Stage 5 gated spur catalog, exactly as the
    # single-window verbs do -- see their docstrings for why (the refit has to
    # see the same masking the original fit did).
    fit_ctx: Stage5FitContext = build_stage5_fit_context(
        path,
        resolved,
        persisted_cal,
        shape_enum,
        replay_spur_catalog=spur_catalog,
    )

    min_freeze_snr = float(
        base_plan.parameters.get("min_freeze_snr", DEFAULT_MIN_FREEZE_SNR)
    )

    # The cascade graph's base edges come from the fitted plan, and their
    # refresh order and the baseline-live set from the automatic fit: the undo
    # baseline once a curation has taken it, else the fit group, which no
    # curation has touched yet.
    with h5open(path, "r") as h5f:
        auto_group = (
            STAGE5_BASELINE_GROUP if STAGE5_BASELINE_GROUP in h5f else fit_group
        )
        baseline_frozen = read_fit_frozen_primaries_by_window(h5f[auto_group])
        baseline_fit = load_spectrum_fit_from_hdf5(h5f[auto_group])
    base_cascade_sources = _base_cascade_sources(
        base_plan, baseline_frozen, fitted.unavailable_window_ids
    )
    baseline_fits = {
        int(wf.window_id): wf
        for wf in baseline_fit.window_fits
        if wf.window_id is not None
    }

    # The calibration actually in force, read once via the same cheap
    # attrs-only stamp the final-products staleness check uses (A7) -- never a
    # full FID load. Stamped on every RefitWindowResult / CreateWindowResult
    # this batch returns (A6); batch-invariant, like everything else here.
    stamp = _current_calibration_stamp(path)
    calibration_state = stamp[0] if stamp is not None else "rb_locked"
    epsilon = stamp[1] if stamp is not None else 0.0
    sigma_epsilon = stamp[2] if stamp is not None else 0.0

    return _SharedFitCtx(
        resolved=resolved,
        shape_enum=shape_enum,
        persisted_cal=persisted_cal,
        tau_maj_global=tau_maj_global,
        sigma_tau_global=sigma_tau_global,
        tau_source=tau_source,
        fit_ctx=fit_ctx,
        peaks_loaded=peaks_loaded,
        peak_frequencies_mhz=peak_frequencies_mhz,
        min_freeze_snr=min_freeze_snr,
        base_plan=base_plan,
        calibration_state=calibration_state,
        epsilon=epsilon,
        sigma_epsilon=sigma_epsilon,
        retired_window_ids=fitted.retired_window_ids,
        unavailable_window_ids=fitted.unavailable_window_ids,
        unavailable_spans_mhz=fitted.unavailable_spans_mhz,
        base_cascade_sources=base_cascade_sources,
        baseline_fits=baseline_fits,
    )


def _seed_unresolved_spreads_from_diagnostics(
    spectrum_fit: SpectrumFit, *, snap_tol_mhz: float
) -> int:
    """Recover auto-merge spreads onto peaks loaded from a file written before
    the per-peak column existed. Returns how many peaks were seeded.

    The spread was always recorded at the *window* level, in
    ``diagnostics['vif_collapse']['collapses']``, and that survives every
    curation write. So a legacy file still knows which line absorbed what --
    it just never stamped it on the line. Without this, every file fitted
    before the column silently loses its widening on its first refit, which
    is the defect itself, merely restricted to existing files.

    Matched per window by nearest frequency to the recorded merge centroid,
    within the file's own snap tolerance -- the same "is this that line?"
    question every curation verb asks, answered the same way. Two guards keep
    a record from landing on the wrong peak: a peak that already carries a
    spread is left alone (a fresh fit stamped it directly), and so is one
    carrying a Stage 6 ``derivation``, which marks a line an edit created or
    altered -- a split product, say, whose components are by assertion
    resolved and must not inherit the parent multiplet's spread.

    Seeds the field only; it never rewrites ``frequency_error``. On a legacy
    file that has not been curated yet the stored error already carries the
    widening, so widening it again would count the spread twice; on one that
    has already been refit the widening is gone and cannot be told apart from
    the first case by looking at the number. The next refit computes a fresh
    formal error and re-applies the widening correctly either way, which is
    the case that matters.
    """
    recs = spectrum_fit.diagnostics.get("vif_collapse", {}).get("collapses", [])
    if not recs:
        return 0

    peaks_by_window: Dict[int, List[FittedPeak]] = {}
    for wf in spectrum_fit.window_fits:
        if wf.window_id is not None:
            peaks_by_window[int(wf.window_id)] = list(wf.fitted_peaks)

    seeded = 0
    used: Dict[int, Set[int]] = {}
    for rec in recs:
        try:
            wid = int(rec["window_id"])
            spread = float(rec.get("unresolved_spread_mhz") or 0.0)
            target = float(rec["merged_frequency_mhz"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (spread > 0.0):
            continue
        best_k, best_d = -1, float("inf")
        for k, pk in enumerate(peaks_by_window.get(wid, [])):
            if k in used.get(wid, set()):
                continue
            if pk.unresolved_spread_mhz is not None or pk.derivation is not None:
                continue
            d = abs(float(pk.frequency_mhz) - target)
            if d < best_d:
                best_d, best_k = d, k
        if best_k < 0 or best_d > snap_tol_mhz:
            continue
        used.setdefault(wid, set()).add(best_k)
        peaks_by_window[wid][best_k].unresolved_spread_mhz = spread
        seeded += 1

    if seeded:
        logger.debug(
            "Stage 6: recovered %d auto-merge spread(s) from vif_collapse "
            "diagnostics (fit predates the per-peak column)",
            seeded,
        )
    return seeded


def _build_batch_changeset(
    path: str, shared: _SharedFitCtx, *, snap_tol_mhz: float
) -> _BatchChangeset:
    """The display batch of *path*: the persisted curated state -- the
    current ``spectrum_fit``, the review's ``created_windows`` overlay and the
    ``fit_window_map`` derived from it, and the serial and window-id
    high-water marks -- that a request resolves against. Built once per
    request, regardless of whether ``shared`` was just built or is being
    reused from an earlier request.

    The fit comes from ``shared.fit_cache`` when a previous verb in this same
    session persisted it and nothing has touched the file since (S5);
    otherwise it is read from disk. The cache decides that for itself against
    a freshly-read fingerprint -- see :class:`_FitCache` -- and hands over
    ownership, so what this batch holds is never something the cache still
    holds. A sessionless caller's ``shared`` is built per call and its cache
    is always empty, so it always reads.
    """
    spectrum_fit: Optional[SpectrumFit] = shared.fit_cache.take(path)
    if spectrum_fit is None:
        with h5open(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                raise StageDependencyError(
                    "review",
                    ["stage5_fitting"],
                    Path(str(path)),
                    command="fit run",
                    message="No Stage 5 fit found in this file. Run 'fit run' first.",
                )
            spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return _display_changeset(
        shared,
        spectrum_fit,
        load_stage6_review_from_file(path),
        snap_tol_mhz=snap_tol_mhz,
    )


def _display_changeset(
    shared: _SharedFitCtx,
    spectrum_fit: SpectrumFit,
    review: Stage6Review,
    *,
    snap_tol_mhz: float,
    high_water_from: Optional[Stage6Review] = None,
) -> _BatchChangeset:
    """A display batch over a curated state held in memory: *spectrum_fit*
    and *review*'s overlay, statuses and high-water marks
    (*high_water_from*'s instead when given: a log-prefix apply displays the
    prefix's state but mints above every serial and window id the file's
    lineage has used)."""
    # A fit written before the per-peak column carries its auto-merge spreads
    # only in the window-level diagnostics; put them back on the lines before
    # anything reads them. Idempotent, and a no-op on a fit with no merges.
    _seed_unresolved_spreads_from_diagnostics(spectrum_fit, snap_tol_mhz=snap_tol_mhz)
    created_windows = list(review.created_windows)
    effective_plan = _overlay_created_windows(shared.base_plan, created_windows)
    fit_ids = {
        int(wf.window_id) for wf in spectrum_fit.window_fits if wf.window_id is not None
    }
    marks = review if high_water_from is None else high_water_from
    return _BatchChangeset(
        spectrum_fit=spectrum_fit,
        created_windows=created_windows,
        fit_window_map={w.window_id: w for w in effective_plan.windows},
        base_serial=int(marks.next_serial),
        lineless_reviewable=frozenset(flagged_lineless_ids(review, fit_ids)),
        window_id_high_water=int(marks.window_id_high_water),
    )


def _build_batch_ctx(
    path: str, *, snap_tol_mhz: float, shared: Optional[_SharedFitCtx] = None
) -> _BatchCtx:
    """Build the display batch of *path* (:func:`_build_batch_changeset`).

    Load-or-accept the shared, batch-invariant half (build it fresh unless a
    caller already has one -- a review session reuses one ``_SharedFitCtx``
    across many requests), then always build a brand new changeset: none of
    the changeset is safe to reuse across requests, even when the shared
    context is (see ``_BatchChangeset``).
    """
    if shared is None:
        shared = _build_shared_fit_ctx(path)
    changeset = _build_batch_changeset(path, shared, snap_tol_mhz=snap_tol_mhz)
    return _BatchCtx(shared=shared, changeset=changeset)


def _batch_plan_window_ids(
    ctx: _BatchCtx, plan: Sequence["PlannedAction"]
) -> Optional[Set[int]]:
    """The window ids of the plan this batch's creates mint against, for
    bounding the ids an unpinned create can mint
    (:func:`_unknown_plan_window_ids`); ``None`` (no bound) when the plan
    has no unpinned create, so the overlay is only built when it is needed."""
    if not any(
        a.kind == "create" and a.window_id == _NEW_WINDOW_SENTINEL for a in plan
    ):
        return None
    # The ids a structural merge absorbed count as taken: a create mints above
    # them, so naming one is never a forward reference to a minted window. So
    # does every id a create has minted in the lineage: a fresh create mints
    # above the window-id high-water mark, so an undone create's id is never
    # minted again.
    taken = {int(w.window_id) for w in _batch_effective_plan(ctx).windows}
    taken |= set(ctx.shared.retired_window_ids)
    if ctx.changeset.window_id_high_water >= 0:
        taken.add(ctx.changeset.window_id_high_water)
    return taken


def _batch_effective_plan(ctx: _BatchCtx) -> "WindowPlan":
    """The window plan as of *this point* in the batch (base + this batch's own
    creates so far), recomputed in memory -- no file round trip."""
    return _overlay_created_windows(ctx.shared.base_plan, ctx.changeset.created_windows)


def _splice_edit_result(
    spectrum_fit: SpectrumFit, window_id: int, new_wf: FittingResult
) -> None:
    """Replace ``window_id``'s entry in ``spectrum_fit`` with ``new_wf`` (an
    existing window whose peak count may have changed, but not its identity)."""
    new_global_peaks = [
        p for p in spectrum_fit.fitted_peaks if p.window_id != window_id
    ] + list(new_wf.fitted_peaks)
    new_global_peaks.sort(key=lambda p: float(p.frequency_mhz))
    spectrum_fit.fitted_peaks = new_global_peaks
    spectrum_fit.window_fits = [
        new_wf if wf.window_id == window_id else wf for wf in spectrum_fit.window_fits
    ]


def _splice_new_window_fit(
    spectrum_fit: SpectrumFit, new_wid: int, new_wf: FittingResult
) -> None:
    """Insert a freshly created window's ``FittingResult`` into ``spectrum_fit``,
    re-sorting ``window_fits`` by ascending window id (matching
    :func:`create_window_impl`'s persisted ordering)."""
    other_fits = [wf for wf in spectrum_fit.window_fits if wf.window_id != new_wid]
    spectrum_fit.window_fits = sorted(
        other_fits + [new_wf],
        key=lambda wf: int(wf.window_id) if wf.window_id is not None else -1,
    )
    new_global_peaks = [
        p for p in spectrum_fit.fitted_peaks if p.window_id != new_wid
    ] + list(new_wf.fitted_peaks)
    new_global_peaks.sort(key=lambda p: float(p.frequency_mhz))
    spectrum_fit.fitted_peaks = new_global_peaks


def _batch_lookup_wf(ctx: _BatchCtx, window_id: int) -> FittingResult:
    wf_list = [
        wf for wf in ctx.changeset.spectrum_fit.window_fits if wf.window_id == window_id
    ]
    if not wf_list:
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the Stage 5 fit",
        )
    return wf_list[0]


def _batch_apply_edit_core(
    ctx: _BatchCtx,
    window_id: int,
    *,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    remove_uids: Sequence[int] = (),
    add_uids: Optional[Sequence[int]] = None,
    add_seeds: Optional[List[ModelPeak]] = None,
    add_derivations: Optional[Sequence[Optional[int]]] = None,
    fit_win: Optional["FitWindow"] = None,
    snap_tol_mhz: float,
) -> FittingResult:
    """In-memory equivalent of the fit-mutating middle of
    :func:`refit_window_impl` (materialize -> NLS -> splice), reusing the
    batch's shared context instead of rebuilding it. Marks ``window_id`` dirty.

    *fit_win* is the window's geometry in force for this refit (a decision
    recorded before a widening of its window refits on the narrow geometry);
    the batch's current plan entry when omitted."""
    from ..fitting.result_conversion import sort_fitting_result_by_frequency

    wf = _batch_lookup_wf(ctx, window_id)
    if fit_win is None:
        fit_win = ctx.changeset.fit_window_map.get(window_id)
    if fit_win is None:
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the Stage 4 WindowPlan. "
            "Stage 4 may have been re-run and changed the window geometry.",
        )
    _refuse_unavailable_fit_plan(
        ctx.shared.unavailable_window_ids, [window_id], f"editing window {window_id}"
    )
    tau_maj_us, sigma_tau_us = _resolve_refit_window_tau(
        fit_win,
        ctx.shared.resolved,
        ctx.shared.persisted_cal,
        ctx.shared.tau_maj_global,
        ctx.shared.sigma_tau_global,
        ctx.shared.tau_source,
    )
    new_wf = refit_window_core(
        ctx.shared.fit_ctx,
        fit_win,
        wf,
        resolved=ctx.shared.resolved,
        shape_enum=ctx.shared.shape_enum,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        peak_frequencies_mhz=ctx.shared.peak_frequencies_mhz,
        add=add,
        remove=remove,
        add_seeds=add_seeds,
        add_derivations=add_derivations,
        snap_tol_mhz=snap_tol_mhz,
        remove_uids=remove_uids,
        add_uids=add_uids,
    )
    sort_fitting_result_by_frequency(new_wf)
    _splice_edit_result(ctx.changeset.spectrum_fit, window_id, new_wf)
    # Every edit a decision makes changes the peak set (a bare edit, with no
    # add or remove, is refused at every interface: _refuse_bare_edit), so it
    # moves the leakage skirt its dependents froze and joins the cascade.
    if add or remove or remove_uids or add_seeds:
        ctx.changeset.dirty_wids.add(window_id)
    return new_wf


# ---------------------------------------------------------------------------
# uid-addressed decisions.
#
# A request (a ``review edit``, an accept with a candidate, a curation file's
# actions) is RESOLVED into decision-log rows before anything is fit, against
# the state the user sees: the persisted fit of each window it names. A row
# records the ``peak_uid`` of every peak it removes (``targets``) and the seed
# position and uid of every peak it births (``seeds_mhz`` / ``born_uids``);
# the uid is stamped from the seed position when the row is recorded (the
# birth rule), and a birth whose uid the window already holds is refused,
# never nudged. The rows are then APPLIED by identity, which needs no
# frequency match and no tolerance, so a replay of the log removes exactly the
# peaks the user removed and births every peak under the uid it was born with,
# however far the cascade has moved the window's lines in between.
#
# A write then appends the rows to the log and curates the log from the
# automatic fit (:func:`_curate`). Every refusal is raised by the resolution
# or by the symbolic pass over the new log (:func:`_walk_log_rows`), before
# the first fit -- except that an apply at a ``log_prefix`` resolves the
# caller's own actions against the state the prefix describes, which is
# computed in memory first, so their refusals follow that computation (the
# file is untouched either way).
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _ViewPeak:
    """One peak of a window as a request resolves against it: a peak of the
    window's displayed (persisted) fit, or one an earlier row of the same
    request births, at its seed (design D9). Compared by identity."""

    peak_uid: Optional[int]
    frequency_mhz: float
    amplitude: float = 0.0
    snr: Optional[float] = None


@dataclass
class _PendingRow:
    """One decision-log row a request resolved to (or a recorded row a replay
    re-applies), before its refit. ``evidence`` holds what resolution knows
    (``merged_from``, ``inferred``, ``requested_freq_mhz``, ``split_into``);
    the refit's before/after snapshot is added when the write's replay fits
    it (:func:`_replayed_rows`)."""

    kind: str
    window_id: int
    frequency_mhz: float
    targets: Tuple[int, ...] = ()
    seeds_mhz: Tuple[float, ...] = ()
    born_uids: Tuple[int, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    serial: Optional[int] = None
    """The recorded row's serial, on a row a replay re-applies (a request's
    rows take theirs when they are recorded)."""


@dataclass
class _RefitStep:
    """One refit of one window: a merge row, a split row, or a joint group of
    add/remove rows, applied on *fit_win* (the window's geometry in force when
    the rows were resolved or recorded)."""

    window_id: int
    fit_win: "FitWindow"
    kind: str
    rows: List[_PendingRow]


@dataclass
class _PlannedCreate:
    """A create whose structure is planned and installed, awaiting its fit.
    ``record`` is False for an implied create, whose add records the one row
    of the pair."""

    proposal: "Stage6WindowProposal"
    anchor: float
    record: bool


@dataclass
class _ResolvedAction:
    """One action resolved before any fit: what to fit and record for it.

    ``target_wid`` is the window the action is attributed to (``None`` for a
    bare accept, which fits nothing). ``implied`` is set on the edit half of
    an implied create: the create its one add goes into. ``accept_anchor`` is
    the frequency a bare accept's row is logged at (display only)."""

    original_index: int
    action: PlannedAction
    target_wid: Optional[int] = None
    create: Optional[_PlannedCreate] = None
    steps: List[_RefitStep] = field(default_factory=list)
    accept: bool = False
    accept_anchor: float = 0.0
    implied: Optional[_PlannedCreate] = None


@dataclass
class _ResolveState:
    """What resolving one request accumulates: each touched window's view
    (its displayed peaks, overlaid with the rows resolved so far), the W3
    implied creates awaiting their edit, the W3.1 coalesced ones, and the
    seed-range bounds of each geometry, by ``id`` of the window."""

    views: Dict[int, List[_ViewPeak]] = field(default_factory=dict)
    implied_creates: Dict[int, _PlannedCreate] = field(default_factory=dict)
    coalesced: Dict[int, int] = field(default_factory=dict)
    bounds: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)


def _displayed_wf(ctx: _BatchCtx, window_id: int) -> Optional[FittingResult]:
    """*window_id*'s fit as the request sees it, or ``None`` when it has none
    yet (a window this request creates)."""
    return next(
        (
            wf
            for wf in ctx.changeset.spectrum_fit.window_fits
            if wf.window_id == window_id
        ),
        None,
    )


def _window_view(
    ctx: _BatchCtx, state: _ResolveState, window_id: int
) -> List[_ViewPeak]:
    """*window_id*'s view, built from its displayed fit on first use."""
    view = state.views.get(window_id)
    if view is None:
        wf = _displayed_wf(ctx, window_id)
        view = (
            []
            if wf is None
            else [
                _ViewPeak(
                    peak_uid=None if p.peak_uid is None else int(p.peak_uid),
                    frequency_mhz=float(p.frequency_mhz),
                    amplitude=float(p.amplitude),
                    snr=None if p.snr is None else float(p.snr),
                )
                for p in wf.fitted_peaks
            ]
        )
        state.views[window_id] = view
    return view


def _window_center_of(fit_win: "FitWindow") -> float:
    """The molecular reference frequency a refit on *fit_win* uses (the
    midpoint of its ``freq_range``, :func:`materialize_window`)."""
    lo, hi = fit_win.freq_range
    return 0.5 * (min(lo, hi) + max(lo, hi))


def _mint_peak_uid(ctx: _BatchCtx, fit_win: "FitWindow", seed_mhz: float) -> int:
    """The ``peak_uid`` a peak seeded at *seed_mhz* on *fit_win* is born with
    (the birth rule: stamped from the seed position, once, here)."""
    center = _window_center_of(fit_win)
    fit_ctx = ctx.shared.fit_ctx
    s = sideband_sign(fit_ctx.sideband)
    return peak_uid_from_offset(
        float(s * (float(seed_mhz) - center)),
        center,
        fit_ctx.sideband,
        fit_ctx.probe_freq_mhz,
        fit_ctx.active_ft.n_active,
        fit_ctx.sample_dt_us,
    )


def _seed_bounds(
    fit_ctx: "Stage5FitContext",
    cache: Dict[int, Tuple[float, float, float]],
    fit_win: "FitWindow",
) -> Tuple[float, float, float]:
    """``(lo, hi, slack)`` a seed on *fit_win* must lie within: the window's
    ``freq_range`` plus half a grid bin either side, exactly the test
    :func:`refit_window_core` applies (``freq_range`` names the first and last
    grid points, so a seed a hair outside still lands on an edge bin)."""
    key = id(fit_win)
    hit = cache.get(key)
    if hit is None:
        from ..fitting.plan_execution import materialize_window

        _, offset_grid, _, _, _ = materialize_window(
            fit_win, fit_ctx.active_ft, fit_ctx.rms_for_fit, sideband=fit_ctx.sideband
        )
        lo, hi = fit_win.freq_range
        slack = (
            0.5 * float(np.min(np.abs(np.diff(offset_grid))))
            if offset_grid.size > 1
            else 0.0
        )
        hit = (min(lo, hi), max(lo, hi), slack)
        cache[key] = hit
    return hit


def _refuse_seed_outside_window(
    fit_ctx: "Stage5FitContext",
    cache: Dict[int, Tuple[float, float, float]],
    window_id: int,
    fit_win: "FitWindow",
    seed_mhz: float,
    requested_mhz: float,
) -> None:
    """``target_outside_window`` when *seed_mhz* lies off *fit_win*'s data:
    the window's fit sees only its own band, so a seed outside it would be
    pinned at the nearest edge."""
    lo, hi, slack = _seed_bounds(fit_ctx, cache, fit_win)
    if lo - slack <= float(seed_mhz) <= hi + slack:
        return
    raise CurationConflictError(
        "target_outside_window",
        [int(window_id)],
        message=f"add={float(requested_mhz):.4f} MHz resolves to "
        f"{float(seed_mhz):.4f} MHz, outside window {window_id}'s range "
        f"[{lo:.4f}, {hi:.4f}] MHz. Name the window that covers "
        f"the frequency, or name no window: an edit whose only target "
        f"is this add then goes to the window that covers it, or "
        f"creates one there if none does ('review create' also makes "
        f"one).",
    )


def _refuse_born_uid_clash(
    view: Sequence[_ViewPeak],
    uid: int,
    seed_mhz: float,
    requested_mhz: float,
) -> None:
    """``line_already_fitted`` when a birth's uid is one *view* (the window's
    peaks once the row's own targets are gone, plus the births before it)
    already holds. A curated birth is never nudged to a free uid: that would
    stamp a value that is not its seed's."""
    clash = next((p for p in view if p.peak_uid == int(uid)), None)
    if clash is None:
        return
    raise CurationConflictError(
        "line_already_fitted",
        [int(uid)],
        message=f"add={float(requested_mhz):.4f} MHz seeds at "
        f"{float(seed_mhz):.4f} MHz, which is the birth position of "
        f"the line already fitted at {clash.frequency_mhz:.4f} MHz (both "
        f"carry peak_uid={uid}). Two lines cannot be born "
        f"at the same position; apply one add on its own first, "
        f"then add the second frequency near the resulting fitted "
        f"line in a later edit -- an add within snap tolerance of "
        f"a fitted peak that is not itself being removed is read "
        f"as a split of it.",
    )


def _target_uid(view: Sequence[_ViewPeak], window_id: int, peak: _ViewPeak) -> int:
    """The uid a row targets *peak* by. ``ambiguous_peak`` when the window
    holds that uid more than once (possible only for a thawed copy, which
    carries its primary's uid): the row could not say which one it means."""
    if peak.peak_uid is None:
        raise ValueError(
            f"internal: window {window_id} holds a fitted peak with no peak_uid "
            f"at {peak.frequency_mhz:.4f} MHz"
        )
    uid = int(peak.peak_uid)
    if sum(1 for p in view if p.peak_uid == uid) > 1:
        raise CurationConflictError(
            "ambiguous_peak",
            [uid],
            message=f"window {window_id} holds more than one peak with "
            f"peak_uid={uid} (the one at {peak.frequency_mhz:.4f} MHz among "
            f"them), so a decision cannot name one of them by it. Name a "
            f"frequency the window holds once.",
        )
    return uid


def _require_editable_window(
    ctx: _BatchCtx, window_id: int
) -> Tuple["FitWindow", Optional[FittingResult]]:
    """The geometry in force and the displayed fit of a window a request
    edits, refusing a window the batch cannot edit: one with no fit (and not
    created by this request), one with no plan entry, or one whose fitted
    geometry the file does not hold (``fit_plan_unavailable``)."""
    wf = _displayed_wf(ctx, window_id)
    if wf is None and window_id not in {
        int(w.window_id) for w in ctx.changeset.created_windows
    }:
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the Stage 5 fit",
        )
    fit_win = ctx.changeset.fit_window_map.get(window_id)
    if fit_win is None:
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the Stage 4 WindowPlan. "
            "Stage 4 may have been re-run and changed the window geometry.",
        )
    _refuse_unavailable_fit_plan(
        ctx.shared.unavailable_window_ids, [window_id], f"editing window {window_id}"
    )
    return fit_win, wf


def _infer_curation_intent(
    ctx: _BatchCtx,
    fit_win: "FitWindow",
    view: Sequence[_ViewPeak],
    add: List[float],
    removed: List[_ViewPeak],
    snap_tol_mhz: float,
) -> Tuple[
    Optional[Tuple[List[_ViewPeak], float]],
    List[Tuple[_ViewPeak, float, Optional[List[float]]]],
    List[float],
    List[_ViewPeak],
]:
    """Decompose one coalesced ``edit`` action, against its window's *view*,
    into an inferred merge, zero or more inferred splits, and whatever is left
    over for a plain residual edit. *removed* are the view peaks the action's
    ``remove`` targets resolved to.

    A curation action is read by the change it makes to the peak set, not the
    verb the user typed: an add beside an existing peak is a split of that
    peak into two, and removing the components of one blend while adding
    their replacement is a merge (see :func:`_resolve_edit_steps`, whose only
    caller this is). The outcome is recorded as explicit merge / split / add /
    remove rows and is never re-inferred on a replay.

    Returns
    -------
    tuple
        ``(merge, splits, residual_add, residual_removed)``:

        * ``merge`` -- ``(parents, add_freq)``: the removed peaks forming the
          inferred merge and the one ``add`` frequency it consumed, or
          ``None``.
        * ``splits`` -- one ``(parent, requested_freq, positions)`` per
          inferred split, in the order their ``add`` frequencies appeared.
          ``positions`` is ``[parent_freq, requested_freq]`` to seed at the
          user's requested position, or ``None`` when that position is within
          1 uid unit of the parent's own birth position (the two products
          would be born identical), falling back to the symmetric straddle.
        * ``residual_add`` / ``residual_removed`` -- whatever ``add`` entries
          and removed peaks no merge or split consumed, for a plain edit.
    """

    def nearest(freq: float) -> Tuple[Optional[_ViewPeak], float]:
        best: Optional[_ViewPeak] = None
        best_dist = float("inf")
        for p in view:
            d = abs(p.frequency_mhz - freq)
            if d < best_dist:
                best_dist = d
                best = p
        return best, best_dist

    # --- merge: the WHOLE remove list, when it forms one tight cluster
    # (every pair mutually within snap_tol_mhz -- on a line, equivalent to
    # max - min <= snap_tol_mhz) with exactly one add in its widened span. ---
    merge: Optional[Tuple[List[_ViewPeak], float]] = None
    working_add = list(add)
    working_removed = list(removed)
    if len(removed) >= 2:
        freqs = [p.frequency_mhz for p in removed]
        if max(freqs) - min(freqs) <= snap_tol_mhz:
            lo = min(freqs) - snap_tol_mhz
            hi = max(freqs) + snap_tol_mhz
            qualifying = [a for a in add if lo <= a <= hi]
            if len(qualifying) == 1:
                merge = (list(removed), qualifying[0])
                working_removed = []
                working_add = [a for a in add if a != qualifying[0]]

    # --- splits: each remaining add within snap_tol of a view peak that is
    # neither being removed (decision 1's exception) nor already claimed by an
    # earlier split in this same action (so two adds cannot both split the
    # same parent). ---
    splits: List[Tuple[_ViewPeak, float, Optional[List[float]]]] = []
    claimed: List[_ViewPeak] = []
    residual_add: List[float] = []
    for a in working_add:
        p, d = nearest(a)
        is_removed = p is not None and any(p is r for r in removed)
        is_claimed = p is not None and any(p is c for c in claimed)
        if p is not None and d <= snap_tol_mhz and not is_removed and not is_claimed:
            requested_uid = _mint_peak_uid(ctx, fit_win, float(a))
            positions: Optional[List[float]]
            if p.peak_uid is not None and abs(requested_uid - int(p.peak_uid)) <= 1:
                # The requested position is (near enough) the parent's own
                # birth position that the two products would be born
                # identical -- fall back to the symmetric straddle.
                positions = None
            else:
                positions = [p.frequency_mhz, float(a)]
            splits.append((p, float(a), positions))
            claimed.append(p)
        else:
            residual_add.append(a)

    return merge, splits, residual_add, working_removed


def _ledger_snap(
    ctx: _BatchCtx,
    wf: Optional[FittingResult],
    fit_win: "FitWindow",
    freq_mhz: float,
    snap_tol_mhz: float,
) -> float:
    """Where an add at *freq_mhz* seeds: the nearest candidate of the
    displayed fit's ledger within snap tolerance (its recorded position), else
    the frequency itself. A window with no fit yet has no ledger."""
    if wf is None:
        return float(freq_mhz)
    acquisition_us = float(ctx.shared.fit_ctx.acquisition_us)
    ledger = derive_candidate_ledger(
        wf,
        center_mhz=_window_center_of(fit_win),
        sideband=Sideband.coerce(ctx.shared.fit_ctx.sideband),
        bar=0.0,  # all candidates; the user has decided to add this peak
        res_element_mhz=(
            active_ft_bin_spacing_mhz(acquisition_us) if acquisition_us > 0.0 else None
        ),
    )
    best_cand = None
    best_dist = float("inf")
    for cand in ledger:
        dist = abs(float(cand.frequency_mhz) - float(freq_mhz))
        if dist < best_dist:
            best_dist = dist
            best_cand = cand
    if best_cand is not None and best_dist <= snap_tol_mhz:
        return float(best_cand.frequency_mhz)
    return float(freq_mhz)


def _apply_rows_to_view(view: List[_ViewPeak], rows: Sequence[_PendingRow]) -> None:
    """Overlay *rows* on *view* in place: their targets leave, their births
    join at their seeds, so a later action of the same request resolves
    against the newborns' seed positions (design D9)."""
    gone = {t for r in rows for t in r.targets}
    view[:] = [p for p in view if p.peak_uid not in gone]
    for r in rows:
        for seed, uid in zip(r.seeds_mhz, r.born_uids):
            view.append(_ViewPeak(peak_uid=int(uid), frequency_mhz=float(seed)))


def _merge_seed(
    wf: Optional[FittingResult],
    parents: Sequence[_ViewPeak],
    requested_mhz: Optional[float],
    snap_tol_mhz: float,
) -> float:
    """A merge's seed position: the recorded doublet alternative of the two
    parents when one converged, else the position the user asked for, else
    the parents' SNR-weighted centroid (amplitude-weighted when SNR is
    unknown)."""
    if len(parents) == 2 and wf is not None:
        fa = parents[0].frequency_mhz
        fb = parents[1].frequency_mhz
        for da in getattr(wf, "doublet_alternatives", []):
            pair_match = (
                abs(float(da.frequency_a_mhz) - fa) <= snap_tol_mhz
                and abs(float(da.frequency_b_mhz) - fb) <= snap_tol_mhz
            ) or (
                abs(float(da.frequency_a_mhz) - fb) <= snap_tol_mhz
                and abs(float(da.frequency_b_mhz) - fa) <= snap_tol_mhz
            )
            if (
                pair_match
                and da.merged_success
                and not math.isnan(float(da.merged_frequency_mhz))
            ):
                return float(da.merged_frequency_mhz)
    if requested_mhz is not None:
        # Inferred merge, no recorded doublet alternative: the user said
        # where they want the single line, so seed there rather than at the
        # SNR-weighted centroid (the verb path, with no requested position,
        # keeps the centroid).
        return float(requested_mhz)
    weights = [
        max(float(p.snr) if p.snr is not None else float(p.amplitude), 1e-30)
        for p in parents
    ]
    return sum(p.frequency_mhz * w for p, w in zip(parents, weights)) / sum(weights)


def _resolve_merge_step(
    ctx: _BatchCtx,
    state: _ResolveState,
    window_id: int,
    fit_win: "FitWindow",
    wf: Optional[FittingResult],
    parents: List[_ViewPeak],
    requested_mhz: Optional[float],
    snap_tol_mhz: float,
) -> _RefitStep:
    """The merge row collapsing *parents* into one line, and its refit."""
    view = _window_view(ctx, state, window_id)
    targets = tuple(_target_uid(view, window_id, p) for p in parents)
    seed = _merge_seed(wf, parents, requested_mhz, snap_tol_mhz)
    asked = seed if requested_mhz is None else float(requested_mhz)
    _refuse_seed_outside_window(
        ctx.shared.fit_ctx, state.bounds, window_id, fit_win, seed, asked
    )
    born = _mint_peak_uid(ctx, fit_win, seed)
    _refuse_born_uid_clash(
        [p for p in view if p.peak_uid not in targets], born, seed, asked
    )
    evidence: Dict[str, Any] = {
        # The peaks the request resolved to, at their displayed positions.
        "merged_from": [p.frequency_mhz for p in parents],
    }
    if requested_mhz is not None:
        evidence["inferred"] = True
        evidence["requested_freq_mhz"] = float(requested_mhz)
    row = _PendingRow(
        kind="merge",
        window_id=window_id,
        frequency_mhz=seed,
        targets=targets,
        seeds_mhz=(seed,),
        born_uids=(born,),
        evidence=evidence,
    )
    _apply_rows_to_view(view, [row])
    return _RefitStep(window_id=window_id, fit_win=fit_win, kind="merge", rows=[row])


def _resolve_split_step(
    ctx: _BatchCtx,
    state: _ResolveState,
    window_id: int,
    fit_win: "FitWindow",
    parent: _ViewPeak,
    into: int,
    positions: Optional[Sequence[float]],
    requested_mhz: Optional[float],
) -> _RefitStep:
    """The split row replacing *parent* with *into* lines, and its refit.

    ``positions`` seeds the products at explicit frequencies (the inference
    path: the parent's own position and the user's requested one); ``None``
    straddles the parent by +/-0.5 resolution element. Seeds are clamped into
    the window, so splitting a line near an edge still works."""
    view = _window_view(ctx, state, window_id)
    target = _target_uid(view, window_id, parent)
    if positions is not None:
        if len(positions) != into:
            raise BadSettingError(
                "positions",
                f"exactly into={into} frequencies",
                list(positions),
                message=f"split: len(positions)={len(positions)} must equal into={into}",
            )
        seeds = [float(f) for f in positions]
    else:
        acquisition_us = float(ctx.shared.fit_ctx.acquisition_us)
        resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0 else 0.1
        if into == 2:
            offsets = [-0.5 * resolution_mhz, 0.5 * resolution_mhz]
        else:
            half_span = 0.5 * resolution_mhz
            offsets = [
                -half_span + i * resolution_mhz / (into - 1) for i in range(into)
            ]
        seeds = [parent.frequency_mhz + off for off in offsets]
    lo, hi = sorted(float(v) for v in fit_win.freq_range)
    seeds = [min(max(f, lo), hi) for f in seeds]
    remaining = [p for p in view if p.peak_uid != target]
    born: List[int] = []
    for seed in seeds:
        asked = seed if requested_mhz is None else float(requested_mhz)
        _refuse_seed_outside_window(
            ctx.shared.fit_ctx, state.bounds, window_id, fit_win, seed, asked
        )
        uid = _mint_peak_uid(ctx, fit_win, seed)
        _refuse_born_uid_clash(remaining, uid, seed, asked)
        remaining.append(_ViewPeak(peak_uid=uid, frequency_mhz=seed))
        born.append(uid)
    evidence: Dict[str, Any] = {"split_into": into}
    if requested_mhz is not None:
        evidence["inferred"] = True
        evidence["requested_freq_mhz"] = float(requested_mhz)
    row = _PendingRow(
        kind="split",
        window_id=window_id,
        frequency_mhz=parent.frequency_mhz,
        targets=(target,),
        seeds_mhz=tuple(seeds),
        born_uids=tuple(born),
        evidence=evidence,
    )
    _apply_rows_to_view(view, [row])
    return _RefitStep(window_id=window_id, fit_win=fit_win, kind="split", rows=[row])


def _resolve_remove_targets(
    view: Sequence[_ViewPeak],
    window_id: int,
    remove: Sequence[Union[float, PeakUidToken]],
    snap_tol_mhz: float,
) -> List[_ViewPeak]:
    """The view peaks a ``remove`` list names, in request order, each a
    distinct peak. A ``"uid:N"`` token names its peak exactly; a frequency
    names the nearest peak not already named, within snap tolerance. Unknown
    uids, then unmatched frequencies, are each reported together
    (``not_found``, kind ``"peak"``)."""
    picked: Dict[int, _ViewPeak] = {}
    missing: List[int] = []
    for i, t in enumerate(remove):
        if not isinstance(t, PeakUidToken):
            continue
        match = next(
            (
                p
                for p in view
                if p.peak_uid == t.uid and all(p is not q for q in picked.values())
            ),
            None,
        )
        if match is None:
            missing.append(t.uid)
        else:
            picked[i] = match
    if missing:
        listed = ", ".join(f"peak_uid={u}" for u in missing)
        raise NotFoundValueError(
            "peak",
            missing,
            message=f"window {window_id} has no fitted peak with {listed} to "
            f"remove (already removed, or the identifier is wrong)",
        )
    unmatched: List[float] = []
    details: List[str] = []
    for i, t in enumerate(remove):
        if isinstance(t, PeakUidToken):
            continue
        freq = float(t)
        free = [p for p in view if all(p is not q for q in picked.values())]
        if not free:
            unmatched.append(freq)
            details.append(
                f"remove={freq:.4f} MHz: no fitted peaks in window "
                f"{window_id} to remove"
            )
            continue
        best = min(free, key=lambda p: abs(p.frequency_mhz - freq))
        dist = abs(best.frequency_mhz - freq)
        if dist > snap_tol_mhz:
            unmatched.append(freq)
            details.append(
                f"remove={freq:.4f} MHz: no fitted peak within "
                f"{snap_tol_mhz:.3f} MHz (closest is at "
                f"{best.frequency_mhz:.4f} MHz, distance={dist:.4f} MHz)"
            )
            continue
        picked[i] = best
    if unmatched:
        raise NotFoundValueError("peak", unmatched, message="; ".join(details))
    return [picked[i] for i in sorted(picked)]


def _resolve_edit_steps(
    ctx: _BatchCtx,
    state: _ResolveState,
    window_id: int,
    add: Sequence[float],
    remove: Sequence[Union[float, PeakUidToken]],
    *,
    infer: bool = True,
    snap_tol_mhz: float,
) -> List[_RefitStep]:
    """Resolve one add/remove edit into its rows and refits, without fitting.

    The single implementation behind ``review edit``, an ``edit`` action of a
    curation file and an accept carrying a candidate (``infer=False``). It
    reads the change the edit makes to the peak set, not the verb typed: the
    removes are resolved to the window's displayed peaks
    (:func:`_resolve_remove_targets`), then -- unless ``infer`` is off -- at
    most one inferred merge, zero or more inferred splits
    (:func:`_infer_curation_intent`), and one residual joint add/remove refit
    with whatever is left. Each add seeds at the displayed fit's ledger
    candidate within snap tolerance, else at its own frequency, and every
    birth's uid is minted from its seed. Raises every refusal the edit has,
    before anything is fit: an unknown or unmatched target
    (``not_found``), a seed off the window (``target_outside_window``), a
    birth whose uid the window holds (``line_already_fitted``), a target the
    window holds twice (``ambiguous_peak``) and an unavailable window
    (``fit_plan_unavailable``).

    The window's view is updated with the rows, so a later action of the same
    request resolves against them (design D9).
    """
    fit_win, wf = _require_editable_window(ctx, window_id)
    view = _window_view(ctx, state, window_id)
    removed = _resolve_remove_targets(view, window_id, remove, snap_tol_mhz)

    steps: List[_RefitStep] = []
    residual_add = [float(a) for a in add]
    residual_removed = list(removed)
    if infer and (add or remove):
        merge, splits, residual_add, residual_removed = _infer_curation_intent(
            ctx, fit_win, view, residual_add, removed, snap_tol_mhz
        )
        if merge is not None:
            parents, requested = merge
            steps.append(
                _resolve_merge_step(
                    ctx, state, window_id, fit_win, wf, parents, requested, snap_tol_mhz
                )
            )
        for parent, requested, positions in splits:
            steps.append(
                _resolve_split_step(
                    ctx, state, window_id, fit_win, parent, 2, positions, requested
                )
            )
    if not (residual_add or residual_removed):
        return steps

    targets = [_target_uid(view, window_id, p) for p in residual_removed]
    live = [p for p in view if p.peak_uid not in set(targets)]
    rows: List[_PendingRow] = []
    for a in residual_add:
        seed = _ledger_snap(ctx, wf, fit_win, a, snap_tol_mhz)
        _refuse_seed_outside_window(
            ctx.shared.fit_ctx, state.bounds, window_id, fit_win, seed, a
        )
        uid = _mint_peak_uid(ctx, fit_win, seed)
        _refuse_born_uid_clash(live, uid, seed, a)
        live.append(_ViewPeak(peak_uid=uid, frequency_mhz=seed))
        rows.append(
            _PendingRow(
                kind="add",
                window_id=window_id,
                frequency_mhz=float(a),
                seeds_mhz=(seed,),
                born_uids=(uid,),
            )
        )
    for p, uid in zip(residual_removed, targets):
        # Logged at the peak's displayed position (display only: the replay
        # reads the target uid).
        rows.append(
            _PendingRow(
                kind="remove",
                window_id=window_id,
                frequency_mhz=p.frequency_mhz,
                targets=(uid,),
            )
        )
    _apply_rows_to_view(view, rows)
    steps.append(
        _RefitStep(window_id=window_id, fit_win=fit_win, kind="edit", rows=rows)
    )
    return steps


def _nearest_view_peak(
    view: Sequence[_ViewPeak], freq: float, snap_tol_mhz: float, what: str
) -> _ViewPeak:
    """The view peak nearest *freq* within snap tolerance (``not_found``
    otherwise), for the merge and split verbs."""
    best = min(view, key=lambda p: abs(p.frequency_mhz - freq), default=None)
    dist = float("inf") if best is None else abs(best.frequency_mhz - freq)
    if best is None or dist > snap_tol_mhz:
        raise NotFoundValueError(
            "peak",
            [float(freq)],
            message=f"{what}: no fitted peak within {snap_tol_mhz:.3f} MHz of "
            f"{float(freq):.4f} MHz (closest distance: {dist:.4f} MHz)",
        )
    return best


def _resolve_merge_action(
    ctx: _BatchCtx,
    state: _ResolveState,
    window_id: int,
    peaks: Sequence[float],
    snap_tol_mhz: float,
) -> _RefitStep:
    """The verb-path merge (:func:`merge_peaks_impl`): the named peaks,
    collapsed into one line seeded at the doublet alternative or their
    centroid."""
    _check_merge_arity(peaks)
    fit_win, wf = _require_editable_window(ctx, window_id)
    view = _window_view(ctx, state, window_id)
    parents: List[_ViewPeak] = []
    unmatched: List[float] = []
    details: List[str] = []
    for f in peaks:
        try:
            p = _nearest_view_peak(view, float(f), snap_tol_mhz, "merge")
        except NotFoundValueError as exc:
            unmatched.append(float(f))
            details.append(str(exc))
            continue
        if any(p is q for q in parents):
            raise ValueError(
                f"merge: frequency {float(f):.4f} MHz matched the same "
                f"fitted peak twice"
            )
        parents.append(p)
    if unmatched:
        raise NotFoundValueError("peak", unmatched, message="; ".join(details))
    return _resolve_merge_step(
        ctx, state, window_id, fit_win, wf, parents, None, snap_tol_mhz
    )


def _resolve_split_action(
    ctx: _BatchCtx,
    state: _ResolveState,
    window_id: int,
    peak: float,
    into: int,
    snap_tol_mhz: float,
) -> _RefitStep:
    """The verb-path split (:func:`split_peak_impl`): the named peak, split
    into *into* lines straddling it."""
    _check_split_arity(into)
    fit_win, _ = _require_editable_window(ctx, window_id)
    view = _window_view(ctx, state, window_id)
    parent = _nearest_view_peak(view, float(peak), snap_tol_mhz, "split")
    return _resolve_split_step(ctx, state, window_id, fit_win, parent, into, None, None)


def _apply_refit_step(
    ctx: _BatchCtx, step: _RefitStep, *, snap_tol_mhz: float
) -> FittingResult:
    """Refit one step's window by identity and record its rows.

    Removes every target by uid and births every seed at its recorded
    position under its recorded uid, stamped with the serial of the row that
    births it. A merge's seed starts at the parents' summed amplitude (the
    converged doublet alternative's when the seed is that alternative), a
    split's products at the parent's amplitude over their count, an add's as
    :func:`refit_window_core` reads it: all from the window's state now, the
    parents read by uid. Each row is recorded with the refit's before/after
    snapshot, which a row new to the log keeps as its evidence
    (:func:`_replayed_rows`)."""
    wid = step.window_id
    wf = _batch_lookup_wf(ctx, wid)
    chi2r_before = float(wf.reduced_chi2)
    n_before = len(wf.fitted_peaks)
    serials = [ctx.changeset.pending_serial(k) for k in range(len(step.rows))]
    remove_uids = [t for r in step.rows for t in r.targets]
    seeds = [f for r in step.rows for f in r.seeds_mhz]
    born = [u for r in step.rows for u in r.born_uids]
    derivations: List[Optional[int]] = [
        serials[k] for k, r in enumerate(step.rows) for _ in r.seeds_mhz
    ]
    add_seeds: Optional[List[ModelPeak]] = None
    if step.kind in ("merge", "split"):
        by_uid = {p.peak_uid: p for p in wf.fitted_peaks}
        parents = [by_uid[t] for t in remove_uids if t in by_uid]
        center = _window_center_of(step.fit_win)
        s = sideband_sign(ctx.shared.fit_ctx.sideband)
        if step.kind == "merge":
            amp = sum(float(p.amplitude) for p in parents)
            for da in getattr(wf, "doublet_alternatives", []):
                if (
                    da.merged_success
                    and float(da.merged_frequency_mhz) == seeds[0]
                    and not math.isnan(float(da.merged_amplitude))
                ):
                    amp = float(da.merged_amplitude)
                    break
            amps = [amp]
        else:
            parent_amp = float(parents[0].amplitude) if parents else 0.0
            amps = [parent_amp / len(seeds)] * len(seeds)
        add_seeds = [
            ModelPeak(
                amplitude=max(a, 1e-30),
                offset_mhz=float(s * (f - center)),
                phase=0.0,
                peak_uid=int(u),
            )
            for f, u, a in zip(seeds, born, amps)
        ]
    new_wf = _batch_apply_edit_core(
        ctx,
        wid,
        add=seeds,
        remove_uids=remove_uids,
        add_uids=born,
        add_seeds=add_seeds,
        add_derivations=derivations,
        fit_win=step.fit_win,
        snap_tol_mhz=snap_tol_mhz,
    )
    snapshot: Dict[str, Any] = {
        "chi2r_before": chi2r_before,
        "chi2r_after": float(new_wf.reduced_chi2),
        "n_peaks_before": n_before,
        "n_peaks_after": len(new_wf.fitted_peaks),
    }
    for r in step.rows:
        ctx.changeset.decisions.append(
            {
                "window_id": wid,
                "kind": r.kind,
                "evidence": {**snapshot, **r.evidence},
            }
        )
    return new_wf


def _apply_refit_steps(
    ctx: _BatchCtx,
    steps: Sequence[_RefitStep],
    *,
    implied: Optional[_PlannedCreate] = None,
    snap_tol_mhz: float,
) -> None:
    """Run one action's refits in order (merge, splits, residual edit).

    *implied* is the create an implied create's add went into (W3): its one
    add row then carries the structural consequence on its evidence
    (``created_window``) -- the replay reissues the create from it -- and its
    snapshot runs from the window before the create's fit to after the add's
    (a created window had no fit before, so no ``chi2r_before``)."""
    first_row = len(ctx.changeset.decisions)
    new_wf: Optional[FittingResult] = None
    for step in steps:
        new_wf = _apply_refit_step(ctx, step, snap_tol_mhz=snap_tol_mhz)
    if implied is None or new_wf is None:
        return
    evidence = ctx.changeset.decisions[first_row]["evidence"]
    if implied.proposal.mode == "created":
        # A created window has no "before" fit: the key is omitted rather
        # than stored as a number that was never a fit.
        evidence.pop("chi2r_before", None)
    evidence["chi2r_after"] = _chi2r_evidence(
        _chi2r_or_absent(float(new_wf.reduced_chi2))
    )
    evidence["inferred"] = True
    evidence["created_window"] = _created_window_evidence(implied.proposal)


def _created_window_evidence(proposal: "Stage6WindowProposal") -> Dict[str, Any]:
    """The structural facts a create row (or an implied create's add row, as
    its ``created_window``) records: mode, extent, grid points, frozen
    contributors and dependencies. Read off the planned window alone, so a
    request records them before anything is fit."""
    (lo, hi), n_points = _created_window_extent(proposal.window)
    return {
        "mode": proposal.mode,
        "freq_min_mhz": lo,
        "freq_max_mhz": hi,
        "n_points": n_points,
        "n_contributors": len(proposal.window.fixed_contributors),
        "depends_on": [int(d) for d in proposal.depends_on],
    }


def _batch_apply_accept(ctx: _BatchCtx, window_id: int) -> None:
    """Replay a bare accept: an ``accept`` row changes no fit (the window's
    status then reads "reviewed", :func:`_log_provenance`); the recorded row
    is kept as it is. An accept with a candidate is an add."""
    ctx.changeset.decisions.append(
        {"window_id": window_id, "kind": "accept", "evidence": {}}
    )


def _accept_anchor(
    wf: Optional[FittingResult], plan_win: Optional["FitWindow"]
) -> float:
    """The frequency a bare accept of a window is logged at (display only:
    the row is a marker): the centre of the window's fitted range, else its
    strongest line, else the centre of its planned range (a flagged window
    the fit holds no line in), else 0."""
    if wf is not None:
        c = _window_center(wf)
        if c is not None:
            return c
        if wf.fitted_peaks:
            return float(
                max(
                    wf.fitted_peaks,
                    key=lambda p: (float(p.snr) if p.snr is not None else 0.0),
                ).frequency_mhz
            )
    if plan_win is not None:
        return 0.5 * (float(plan_win.freq_range[0]) + float(plan_win.freq_range[1]))
    return 0.0


def _created_window_extent(
    fit_win: "FitWindow",
) -> Tuple[Tuple[float, float], int]:
    """A created or widened window's ``(min_mhz, max_mhz)`` extent and its
    grid-point count, read off the window itself.

    One reader for both, so the number a dry run predicts and the number an
    apply records cannot drift apart. The span is read from the window's own
    ``diagnostics`` rather than recounted against the grid -- that is where
    the planner put it.
    """
    lo, hi = fit_win.freq_range
    grid_span = fit_win.diagnostics.get("grid_span", [0, -1])
    n_points = int(grid_span[1]) - int(grid_span[0]) + 1
    return (min(lo, hi), max(lo, hi)), n_points


def _batch_live_fit_map(ctx: _BatchCtx) -> Dict[int, FittingResult]:
    """This batch's window fits as they stand *right now*, including any
    window the batch itself has already created. A widening refits the
    widened window from here (it is that window's own action, in its row
    order); a create reads no fit to decide its geometry
    (:func:`_structural_live_window_ids`) or its starting skirt
    (:func:`_created_window_seed`).
    """
    return {
        int(wf.window_id): wf
        for wf in ctx.changeset.spectrum_fit.window_fits
        if wf.window_id is not None
    }


def _structural_live_window_ids(
    shared: _SharedFitCtx, created_windows: Sequence["FitWindow"]
) -> List[int]:
    """The windows that carry a fit at a point of the log, from structure
    alone: every window the automatic fit holds (the baseline-live windows,
    :attr:`_SharedFitCtx.base_cascade_sources`' keys) plus every window the
    creates so far (*created_windows*) installed. No Stage 6 path removes a
    window's fit, so this is the set of fitted windows, read without a fit:
    a Stage-5-dropped window is not in it, and a create planned against it
    neither widens one nor is blocked by one."""
    live = {int(w) for w in shared.base_cascade_sources}
    live.update(int(w.window_id) for w in created_windows)
    return sorted(live)


def _plan_create(
    shared: _SharedFitCtx,
    created_windows: Sequence["FitWindow"],
    anchor_mhz: float,
    *,
    replay_window_id: Optional[int],
    min_new_window_id: int = 0,
) -> "Stage6WindowProposal":
    """Decide WHICH window an anchor gets, from structure alone.

    The structural half of a create: the analysis-band refusal, the
    :func:`~ftmwpipeline.preprocessing.window_planning.plan_stage6_window`
    proposal against the effective plan (the fitted plan overlaid with
    *created_windows*, the overlay the creates before this one installed) and
    its live windows (:func:`_structural_live_window_ids`), and the replay-id
    resolution. A pure function of the fitted plan, the retired and
    baseline-live ids, the static analysis inputs and the ordered creates
    before this one -- never of a fit -- so replaying a log's create rows in
    order reproduces its windows (:func:`_walk_log_rows`).

    *min_new_window_id* is the floor a fresh id is minted at
    (:attr:`~ftmwpipeline.core.data_structures.Stage6Review.window_id_high_water`
    plus one). It feeds only the minting; a pinned replay id is checked
    against the effective plan and the retired ids, as before, so replaying
    the newest create (whose id is the high-water mark itself) is no
    conflict.

    Mutates nothing, so a caller that wants only the geometry leaves no state
    to unwind. Every refusal here is one the apply itself would raise.
    """
    from ..preprocessing.window_planning import plan_stage6_window

    anchor = float(anchor_mhz)
    plan = _overlay_created_windows(shared.base_plan, created_windows)

    if shared.fit_ctx.trim_range is not None:
        t_lo, t_hi = (
            min(shared.fit_ctx.trim_range),
            max(shared.fit_ctx.trim_range),
        )
        if not (t_lo <= anchor <= t_hi):
            raise BadSettingError(
                "anchor_mhz",
                f"a frequency inside the analysis band [{t_lo:.4f}, {t_hi:.4f}] MHz",
                anchor,
                message=f"anchor {anchor:.4f} MHz is outside the analysis band "
                f"[{t_lo:.4f}, {t_hi:.4f}] MHz. Re-run 'ft run' with a trim "
                f"that covers it (which rebuilds the fit) if the line is real.",
            )

    params = plan.parameters
    proposal = plan_stage6_window(
        plan,
        shared.peaks_loaded,
        shared.fit_ctx.active_ft.freq_mhz,
        shared.fit_ctx.active_ft.complex_spectrum,
        shared.fit_ctx.rms_for_fit,
        anchor,
        acquisition_us=float(shared.fit_ctx.acquisition_us),
        tau_us=params.get("tau_us"),
        min_window_half_width_mhz=float(params.get("min_window_half_width_mhz", 2.0)),
        min_window_half_width_points=int(
            params.get("min_window_half_width_points", 32)
        ),
        min_freeze_snr=float(params.get("min_freeze_snr", shared.min_freeze_snr)),
        magnitude_attachment_threshold=float(
            params.get("magnitude_attachment_threshold", 0.1)
        ),
        live_window_ids=_structural_live_window_ids(shared, created_windows),
        reserved_window_ids=shared.retired_window_ids,
        min_new_window_id=int(min_new_window_id),
    )
    new_wid = int(proposal.window.window_id)
    _refuse_create_on_unavailable_fit_plan(shared, proposal, anchor)

    if replay_window_id is not None and int(replay_window_id) != new_wid:
        want = int(replay_window_id)
        if proposal.mode == "widened":
            raise CurationConflictError(
                "replay_conflict",
                [want, new_wid],
                message=f"replaying the window created at {anchor:.4f} MHz now "
                f"widens window {new_wid} instead of creating window {want}; "
                f"the base plan or the surviving edit set has changed",
            )
        # The ids the plan holds and the retired ids -- never the high-water
        # floor, which would make the newest create's own id look taken.
        taken = {int(w.window_id) for w in plan.windows}
        taken |= set(shared.retired_window_ids)
        if want in taken:
            raise CurationConflictError(
                "replay_conflict",
                [want],
                message=f"replaying the window created at {anchor:.4f} MHz wants "
                f"id {want}, which is already in use; the base plan or the "
                f"surviving edit set has changed",
            )
        proposal.window.window_id = want

    return proposal


def _plan_batch_create(
    ctx: _BatchCtx,
    anchor_mhz: float,
    *,
    replay_window_id: Optional[int],
) -> "Stage6WindowProposal":
    """:func:`_plan_create` at this point of the batch: against the overlay
    the batch's creates so far left, minting above the file's window-id
    high-water mark.

    The planning half of a create (:func:`_resolve_action`); everything that
    does after this is the *fit* of the window returned here
    (:func:`_fit_planned_create`). Split out for
    ``review apply --dry-run`` (:func:`_resolve_created_window_structure`),
    which reports the structure a plan would install and nothing else.

    A create a request records that pins its id -- a curation file may name
    one -- must pin above every created window still in the overlay (the
    kept log's creates and the batch's so far): created ids increase along
    the log, which is what keeps every cascade edge between created windows
    pointing from an older window to a newer one
    (:func:`_created_window_sources`). Refused (``replay_conflict``)
    otherwise. The high-water mark is only the floor a fresh id is minted
    at, so a pin may name the id an undone create had -- redoing that create
    under its own id -- as long as no live create holds a higher one. (A
    replayed row keeps the id it recorded: :func:`_walk_log_rows` plans it.)
    """
    hw = ctx.changeset.window_id_high_water
    proposal = _plan_create(
        ctx.shared,
        ctx.changeset.created_windows,
        anchor_mhz,
        replay_window_id=replay_window_id,
        min_new_window_id=hw + 1,
    )
    want = int(proposal.window.window_id)
    if replay_window_id is not None and proposal.mode == "created":
        # Widened base windows sit in the overlay under their base ids.
        base_ids = {int(w.window_id) for w in ctx.shared.base_plan.windows}
        newest = max(
            (
                int(w.window_id)
                for w in ctx.changeset.created_windows
                if int(w.window_id) not in base_ids
            ),
            default=-1,
        )
        if want <= newest:
            raise CurationConflictError(
                "replay_conflict",
                [want],
                message=f"the window created at {float(anchor_mhz):.4f} MHz pins "
                f"id {want}, but created window {newest} is older in the log: "
                f"created window ids only increase. Pin an id above {newest}, "
                f"or let the create mint one.",
            )
    return proposal


def _window_geometry(fit_win: "FitWindow") -> Tuple[Any, ...]:
    """What a window's geometry is, for telling whether it changed: its
    extent, its contributors (with ``edge_free``), its free peaks, its batch
    and its grid span. The anchor and the planner's other diagnostics are not
    geometry."""
    return (
        tuple(float(v) for v in fit_win.freq_range),
        tuple(
            (
                int(c.primary_window_id),
                int(c.peak_index),
                float(c.frequency_mhz),
                bool(c.edge_free),
            )
            for c in fit_win.fixed_contributors
        ),
        tuple(int(i) for i in fit_win.free_peak_indices),
        int(fit_win.batch),
        tuple(int(v) for v in fit_win.diagnostics.get("grid_span", ())),
    )


def _geometry_changed_window_ids(
    base_plan: "WindowPlan",
    before: Sequence["FitWindow"],
    after: Sequence["FitWindow"],
) -> List[int]:
    """The windows that exist under the overlay *after* and whose geometry
    differs from what it was under the overlay *before*, ascending. A window
    only *before* holds (an undone create) no longer exists and is not
    listed."""
    old = {
        int(w.window_id): _window_geometry(w)
        for w in _overlay_created_windows(base_plan, before).windows
    }
    return sorted(
        int(w.window_id)
        for w in _overlay_created_windows(base_plan, after).windows
        if int(w.window_id) in old and old[int(w.window_id)] != _window_geometry(w)
    )


def _refuse_create_on_unavailable_fit_plan(
    shared: _SharedFitCtx, proposal: "Stage6WindowProposal", anchor: float
) -> None:
    """Refuse a create that a fit's unrecorded merge makes unplannable.

    Only for a fit that merged windows before the fitted plan was stored
    (:attr:`_SharedFitCtx.unavailable_window_ids`): the planner sees the
    Stage 4 windows there, so a window it proposes inside a merged range would
    overlap the window the fit has, and one that widens or reads a merged
    window would use geometry the fit was not made on.
    """
    unavailable = shared.unavailable_window_ids
    if not unavailable:
        return
    window = proposal.window
    lo, hi = min(window.freq_range), max(window.freq_range)
    if proposal.mode == "widened":
        touched = [int(window.window_id)]
    else:
        touched = [int(d) for d in proposal.depends_on]
    _refuse_unavailable_fit_plan(
        unavailable, touched, f"creating a window at {anchor:.4f} MHz"
    )
    for s_lo, s_hi in shared.unavailable_spans_mhz:
        if lo <= s_hi and hi >= s_lo:
            from .fitted_plan import FIT_PLAN_UNAVAILABLE

            blocked = sorted(int(w) for w in unavailable)
            raise CurationConflictError(
                FIT_PLAN_UNAVAILABLE,
                blocked,
                message=f"creating a window at {anchor:.4f} MHz "
                f"([{lo:.4f}, {hi:.4f}] MHz) overlaps a range a structural merge "
                f"joined in a fit that predates the stored fitted plan: the "
                f"windows the fit was made on are not in this file. Re-run "
                f"'fit run' to curate there.",
            )


def _batch_implied_create_target(ctx: _BatchCtx, anchor_mhz: float) -> Optional[int]:
    """W3.1: the live window, in this batch's CURRENT state, that already
    covers *anchor_mhz* -- the coalescing check a FRESH implied create runs
    before minting a second window into a gap a PRIOR action in the same
    batch already filled.

    Two omitted-window ``add`` rows in one gap each resolve against live
    windows only at *parse* time (:func:`_resolve_curation_window_ids`), so
    both read as uncovered and both mint their own correlation id. By the
    time the SECOND one's ``create`` action actually runs, the first one's
    create has already installed a window that may well cover the second's
    anchor too -- exactly what :func:`_plan_batch_create` /
    :func:`plan_stage6_window` would otherwise refuse as "anchor already
    falls inside window N", unfollowable advice from inside a batch since
    that window does not exist until the first action applies.

    Builds the identical inputs :func:`_plan_batch_create` hands the planner --
    :func:`_batch_effective_plan` (base plan + this batch's own creates so
    far) and the live window ids at this point
    (:func:`_structural_live_window_ids`) -- so the answer this returns is the
    planner's own answer, asked one step earlier:
    :func:`~ftmwpipeline.preprocessing.window_planning.live_window_covering_anchor`
    is the exact predicate :func:`plan_stage6_window` uses for its refusal,
    extracted so there is exactly one copy of the rule. This is NOT a search:
    no tolerance, no nearest-peak, no radius -- only "is this anchor, on the
    planner's own grid, already inside a live window's span."

    Returns ``None`` when nothing covers the anchor yet, in which case the
    caller mints (or widens) a window for it exactly as before.
    """
    from ..preprocessing.window_planning import live_window_covering_anchor

    return live_window_covering_anchor(
        _batch_effective_plan(ctx),
        ctx.shared.fit_ctx.active_ft.freq_mhz,
        float(anchor_mhz),
        live_window_ids=_structural_live_window_ids(
            ctx.shared, ctx.changeset.created_windows
        ),
    )


def _install_planned_create(ctx: _BatchCtx, proposal: "Stage6WindowProposal") -> None:
    """Install a planned create's structure in the batch: the overlay entry,
    the effective plan's window map and the window-id high-water mark. Reads
    and fits nothing, so a request installs every create it makes while it
    resolves, before the first fit, and a later create or edit in it sees the
    window."""
    fit_win = proposal.window
    new_wid = int(fit_win.window_id)
    ctx.changeset.created_windows = [
        w for w in ctx.changeset.created_windows if int(w.window_id) != new_wid
    ] + [fit_win]
    ctx.changeset.fit_window_map = {
        w.window_id: w for w in _batch_effective_plan(ctx).windows
    }
    if proposal.mode == "created":
        ctx.changeset.window_id_high_water = max(
            ctx.changeset.window_id_high_water, new_wid
        )


def _fit_planned_create(
    ctx: _BatchCtx, planned: _PlannedCreate, *, snap_tol_mhz: float
) -> None:
    """Fit a create :func:`_install_planned_create` installed, in a replay.

    ``mode="created"`` mints a window whose fit holds no line yet, so it
    never joins ``dirty_wids`` -- no cascade from it (a later create may read
    it, but only an add into it changes what it leaks). Its starting skirt is
    read from the automatic fit (:func:`_created_window_seed`), and it is
    recorded in ``created_wids`` so the cascade refreshes it when one of its
    sources has been edited (:func:`_cascade_batch`).
    ``mode="widened"`` instead grows an existing window and refits its
    existing peak set on the wider grid, so it is dirty: its dependents may
    have frozen on a leakage skirt that widening just removed, and the
    cascade must reach them.

    ``planned.record`` False (W3) records no ``"create_window"`` row: an
    IMPLIED create's add records ONE ``"add"`` row for the whole create+edit
    pair instead, carrying the structural consequence on its own evidence
    (:func:`_apply_refit_steps`).
    """
    from ..fitting.result_conversion import sort_fitting_result_by_frequency
    from .active_ft_support import default_tau0_us

    proposal = planned.proposal
    fit_win = proposal.window
    new_wid = int(fit_win.window_id)
    tau_maj_us, sigma_tau_us = _resolve_refit_window_tau(
        fit_win,
        ctx.shared.resolved,
        ctx.shared.persisted_cal,
        ctx.shared.tau_maj_global,
        ctx.shared.sigma_tau_global,
        ctx.shared.tau_source,
    )
    if proposal.mode == "created":
        tau0 = (
            float(tau_maj_us)
            if tau_maj_us is not None and tau_maj_us > 0.0
            else default_tau0_us(float(ctx.shared.fit_ctx.acquisition_us))
        )
        seed_wf = FittingResult(window_id=new_wid, shape=ctx.shared.shape_enum.value)
        seed_wf.fixed_parameters = _created_window_seed(
            ctx.shared, fit_win, ctx.changeset.created_windows
        )
        seed_wf.shared_parameters = {"tau_us": {"value": tau0, "fitted": False}}
    else:
        existing_wf = _batch_live_fit_map(ctx).get(new_wid)
        if existing_wf is None:
            # An invariant guard, not a route: the planner widens only a live
            # window, so a correct caller cannot reach this.
            raise ValueError(
                f"window {new_wid} has no Stage 5 fit to widen; "
                "re-run 'fit run' before creating windows"
            )
        seed_wf = existing_wf

    new_wf: FittingResult = refit_window_core(
        ctx.shared.fit_ctx,
        fit_win,
        seed_wf,
        resolved=ctx.shared.resolved,
        shape_enum=ctx.shared.shape_enum,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        peak_frequencies_mhz=ctx.shared.peak_frequencies_mhz,
        snap_tol_mhz=snap_tol_mhz,
    )
    sort_fitting_result_by_frequency(new_wf)
    _splice_new_window_fit(ctx.changeset.spectrum_fit, new_wid, new_wf)

    if proposal.mode == "created":
        ctx.changeset.created_wids.add(new_wid)
    else:
        # An existing window whose fit just moved: its dependents may have
        # frozen on the leakage skirt it no longer has, so it must join the
        # cascade. A freshly *created* window holds no line yet, so nothing
        # that reads it has anything to refresh: do NOT collapse this into a
        # blanket ``dirty_wids.add(new_wid)`` for both modes.
        ctx.changeset.dirty_wids.add(new_wid)

    if planned.record:
        ctx.changeset.decisions.append(
            {
                "window_id": new_wid,
                "kind": "create_window",
                "evidence": _created_window_evidence(proposal),
            }
        )


def _create_window_result(
    shared: _SharedFitCtx, planned: _PlannedCreate, n_peaks: int
) -> "CreateWindowResult":
    """The result of a create a request resolved: the structure it planned,
    both frames on its anchor and extent, and *n_peaks*, the lines the
    curated fit holds in the window."""
    proposal = planned.proposal
    fit_win = proposal.window
    (lo, hi), n_points = _created_window_extent(fit_win)
    anchor = float(planned.anchor)
    probe_freq_mhz = shared.fit_ctx.probe_freq_mhz
    epsilon = shared.epsilon
    return CreateWindowResult(
        window_id=int(fit_win.window_id),
        mode=proposal.mode,
        anchor_mhz=anchor,
        freq_range=(lo, hi),
        n_points=n_points,
        n_contributors=len(fit_win.fixed_contributors),
        depends_on=[int(d) for d in proposal.depends_on],
        n_peaks=int(n_peaks),
        anchor_calibrated_mhz=_frame_to_calibrated(
            anchor, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon
        ),
        freq_range_calibrated=(
            _frame_to_calibrated(lo, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon),
            _frame_to_calibrated(hi, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon),
        ),
        calibration_state=shared.calibration_state,
        epsilon=epsilon,
        sigma_epsilon=shared.sigma_epsilon,
    )


def _planned_window_result(planned: _PlannedCreate) -> "PlannedWindowResult":
    """A create's structure as :class:`CurationApplyResult` and
    :class:`ReviewPreviewResult` report it (:class:`PlannedWindowResult`)."""
    fit_win = planned.proposal.window
    (lo, hi), n_points = _created_window_extent(fit_win)
    return PlannedWindowResult(
        window_id=int(fit_win.window_id),
        anchor_mhz=float(planned.anchor),
        mode=planned.proposal.mode,
        freq_range=(lo, hi),
        n_points=n_points,
        n_contributors=len(fit_win.fixed_contributors),
        depends_on=[int(d) for d in planned.proposal.depends_on],
    )


def _resolve_bare_accept(ctx: _BatchCtx, window_id: int) -> float:
    """The anchor a bare accept of *window_id* is logged at, refusing a
    window the request does not see: one with a fit, one an earlier action of
    the request creates, or a flagged window the fit holds no line in
    (``changeset.lineless_reviewable``)."""
    wf = _displayed_wf(ctx, window_id)
    created = {int(w.window_id) for w in ctx.changeset.created_windows}
    if (
        wf is None
        and window_id not in created
        and window_id not in ctx.changeset.lineless_reviewable
    ):
        raise NotFoundError(
            "window",
            [window_id],
            message=f"window_id={window_id} not found in the Stage 5 fit",
        )
    return _accept_anchor(wf, ctx.changeset.fit_window_map.get(window_id))


def _resolve_action(
    ctx: _BatchCtx,
    state: _ResolveState,
    original_index: int,
    action: PlannedAction,
    *,
    snap_tol_mhz: float,
) -> _ResolvedAction:
    """Resolve one planned action against the request's state so far, fitting
    nothing: a create is planned and its structure installed, an edit (or an
    accept carrying a candidate, or a verb-path merge / split) is resolved
    into its rows and refits (:func:`_resolve_edit_steps`), a bare accept is
    checked against the windows the request sees and given its anchor. Raises
    every refusal the action has.

    W3: the create half of an implied create is planned here like any other
    (unless a window an earlier action installed already covers its anchor:
    W3.1 coalescing, :func:`_batch_implied_create_target`, after which its
    edit is an ordinary edit of that window), and its edit half must resolve
    to the one add it is: an add the window it lands in reads as a merge or a
    split (only a widened, non-empty window could) is refused, since that row
    could not carry the ``created_window`` evidence a replay reissues the
    create from. Reaching it requires the anchor to land within snap tolerance
    of a peak in the very window a too-narrow gap just widened; measured on
    the real 2638 fit, zero configurations reach it, so naming the window
    explicitly costs nothing.
    """
    out = _ResolvedAction(original_index=original_index, action=action)
    if action.kind == "create":
        if action.anchor is None:
            raise ValueError("create action requires an anchor frequency")
        # W3.1: only a FRESH implied create (a not-yet-real correlation id)
        # can coalesce. An EXPLICIT create keeps the refusal (it asserts a
        # window is needed).
        fresh_implied = action.implied_create and _is_implied_window_id(
            action.window_id
        )
        coalesce_target = (
            _batch_implied_create_target(ctx, action.anchor) if fresh_implied else None
        )
        if coalesce_target is not None:
            state.coalesced[action.window_id] = coalesce_target
            out.target_wid = coalesce_target
            return out
        proposal = _plan_batch_create(
            ctx,
            action.anchor,
            replay_window_id=(
                None
                if action.window_id == _NEW_WINDOW_SENTINEL
                or _is_implied_window_id(action.window_id)
                else action.window_id
            ),
        )
        _install_planned_create(ctx, proposal)
        planned = _PlannedCreate(
            proposal=proposal,
            anchor=float(action.anchor),
            record=not action.implied_create,
        )
        out.create = planned
        out.target_wid = int(proposal.window.window_id)
        if action.implied_create:
            state.implied_creates[action.window_id] = planned
        return out
    if action.kind == "edit":
        wid = action.window_id
        implied: Optional[_PlannedCreate] = None
        if action.implied_create:
            coalesced = state.coalesced.pop(action.window_id, None)
            if coalesced is not None:
                wid = coalesced
            else:
                implied = state.implied_creates.pop(action.window_id, None)
                if implied is None:
                    raise ValueError(
                        f"internal: no matching implied create for window "
                        f"{action.window_id} (canonicalization should always "
                        f"run creates first)"
                    )
                wid = int(implied.proposal.window.window_id)
        out.steps = _resolve_edit_steps(
            ctx, state, wid, action.add, action.remove, snap_tol_mhz=snap_tol_mhz
        )
        kinds = [r.kind for st in out.steps for r in st.rows]
        if implied is not None and kinds != ["add"]:
            raise ValueError(
                f"add={float(action.add[0]):.4f} MHz implies creating a window, "
                f"but inside window {wid} (mode='{implied.proposal.mode}') it "
                f"was reinterpreted as an edit of an existing peak, which cannot "
                f"record the implied create for replay. Name the window "
                f"explicitly with a separate 'review create' plus 'review edit' "
                f"if that reinterpretation is what you want."
            )
        out.implied = implied
        out.target_wid = wid
        return out
    if action.kind == "merge":
        out.steps = [
            _resolve_merge_action(
                ctx, state, action.window_id, action.peaks, snap_tol_mhz
            )
        ]
        out.target_wid = action.window_id
        return out
    if action.kind == "split":
        if action.peak is None:
            raise ValueError("split action requires a peak frequency")
        out.steps = [
            _resolve_split_action(
                ctx, state, action.window_id, action.peak, action.into, snap_tol_mhz
            )
        ]
        out.target_wid = action.window_id
        return out
    if action.kind == "accept":
        if action.candidate is None:
            out.accept = True
            out.accept_anchor = _resolve_bare_accept(ctx, action.window_id)
            return out
        out.steps = _resolve_edit_steps(
            ctx,
            state,
            action.window_id,
            [float(action.candidate)],
            [],
            infer=False,
            snap_tol_mhz=snap_tol_mhz,
        )
        out.target_wid = action.window_id
        return out
    raise ValueError(f"unknown curation action kind {action.kind!r}")


def _run_resolved_action(
    ctx: _BatchCtx, ra: _ResolvedAction, *, snap_tol_mhz: float
) -> None:
    """Fit and record one resolved action of a replay (a create, a bare
    accept, or an action's refits)."""
    if ra.create is not None:
        _fit_planned_create(ctx, ra.create, snap_tol_mhz=snap_tol_mhz)
    elif ra.accept:
        _batch_apply_accept(ctx, ra.action.window_id)
    elif ra.steps:
        _apply_refit_steps(ctx, ra.steps, implied=ra.implied, snap_tol_mhz=snap_tol_mhz)


def _canonicalize_batch_plan(
    plan: Sequence[PlannedAction],
) -> List[Tuple[int, PlannedAction]]:
    """Pair each action with its original plan position, then order for
    execution: creates first (their own relative order -- they install
    structure later rows name), then every other action grouped by ascending
    window id. ``sorted`` is stable, so two actions sharing a window id keep
    the relative order ``_resolve_curation_plan`` already gave them.

    W3, deliberately UNCHANGED: an implied edit's ``window_id`` is a fresh
    negative correlation id (:data:`_FIRST_IMPLIED_WINDOW_ID`), which is not
    "not yet known" from this function's point of view, so a correlation id
    sorts ahead of every real, non-negative window id, clustering same-batch
    implied edits together ahead of ordinary ones. That is safe because
    "creates first" (above) already guarantees every ``create`` -- implied or
    not -- has run, and so every correlation id is resolvable to its real
    window, before ANY ``rest`` action executes; and because windows are
    independent, so the exact relative order among DIFFERENT-window edits in
    ``rest`` affects the decision log's presentation order, never a fit's
    numerical outcome: a create reads its starting skirt from the automatic
    fit and the cascade reads its edges from the plan, so not even a created
    window's numbers depend on where its create sits relative to the edits
    (:func:`test_apply_row_order_independent` pins it for ordinary windows).
    What genuinely could not be known here is the real window an implied
    create MINTS -- that late binding is resolved when the plan is resolved
    (:func:`_resolve_action`), not by this function, which only ever
    schedules.

    Within the negative (correlation-id) group specifically, the sort key is
    ``-window_id`` rather than ``window_id`` itself: correlation ids are
    minted in a DECREASING sequence (:data:`_FIRST_IMPLIED_WINDOW_ID`, then
    one lower per further implied create in the same plan), so a plain
    ascending sort on ``window_id`` would replay them in the REVERSE of the
    order their rows were written. That reversal is harmless while every
    implied create mints its own distinct window -- "presentation order only"
    as above -- but W3.1 coalescing (:func:`_batch_implied_create_target`) can
    now land two implied edits on the SAME real window within one plan, and
    at that point relative order stops being cosmetic: it decides which of
    the two edits the decision log shows first, which must match row order to
    agree with the identical edits applied sequentially (the first row's
    implied create is the one that actually minted the window and so is the
    one that carries ``created_window`` evidence). Real, non-negative window
    ids are untouched -- their ascending-by-id order is the existing, tested
    behavior.
    """

    def _rest_key(item: Tuple[int, PlannedAction]) -> Tuple[int, int]:
        wid = item[1].window_id
        return (0, -wid) if wid < 0 else (1, wid)

    indexed = list(enumerate(plan))
    creates = [t for t in indexed if t[1].kind == "create"]
    rest = sorted((t for t in indexed if t[1].kind != "create"), key=_rest_key)
    return creates + rest


def _replayed_rows(ctx: _BatchCtx, recorded: int) -> List[DecisionLogEntry]:
    """The log a replay batch replayed, as the write persists it.

    Rows are immutable: the first *recorded* rows (the ones the file's log
    already held) are kept verbatim, only their ``order_index`` positions
    recomputed. A row new to the log is the row its request resolved to, its
    evidence completed with what the replay recorded for it -- the refit's
    before/after snapshot, taken as this write fits it -- and fixed from then
    on. A replay that did not record one decision per row, in the rows'
    windows and kinds, is an internal error (``replay_diverged``).
    """
    replayed = ctx.changeset.replayed
    decisions = ctx.changeset.decisions
    out: List[DecisionLogEntry] = []
    for k, src in enumerate(replayed):
        dec = decisions[k] if k < len(decisions) else None
        if dec is None or (int(src.window_id), str(src.kind)) != (
            int(dec["window_id"]),
            str(dec["kind"]),
        ):
            raise CurationConflictError(
                "replay_diverged",
                [_serial_id(src)],
                message=f"cannot replay decision {_serial_id(src)} ({src.kind} "
                f"on window {src.window_id}): replayed from the automatic fit, "
                "the decisions before it no longer record it. Undo it together "
                "with the decisions after it.",
            )
        if k < recorded:
            out.append(replace(src, order_index=k))
        else:
            if dec["evidence"] is None:
                # A new row's window is always refit: its rows changed.
                raise RuntimeError(
                    f"internal: the replay skipped decision {_serial_id(src)}, "
                    "which is new to the log and must be fit to be recorded"
                )
            out.append(
                replace(
                    src, order_index=k, evidence={**dec["evidence"], **src.evidence}
                )
            )
    return out


def _settle_empty_window_reasons(
    statuses: Dict[int, WindowReviewStatus],
    spectrum_fit: SpectrumFit,
    created_windows: Sequence["FitWindow"],
    *,
    base_plan_windows: Sequence["FitWindow"],
    edited_window_ids: Set[int],
) -> None:
    """Drop the ``empty_window_residual`` reason the batch resolved, in place.

    The reason stops describing a window once the fit holds a line in it, a
    fit-changing decision was recorded on it, or a created window took it over
    (:func:`~ftmwpipeline._internal.empty_window_attention.superseded_window_ids`)
    -- the same conditions :func:`review_run_impl` leaves it out under. A
    created window takes the window over only when it covers what was flagged
    there (:func:`~ftmwpipeline._internal.empty_window_attention.takeover_points`).
    A window the fit has no result for and whose status then carries nothing
    has no status left to keep, reviewed or not: its review was of the item.
    """
    flagged = [
        wid
        for wid, st in statuses.items()
        if any(r.kind in EMPTY_WINDOW_KINDS for r in st.attention_reasons)
    ]
    if not flagged:
        return
    fit_ids = {
        int(wf.window_id) for wf in spectrum_fit.window_fits if wf.window_id is not None
    }
    live = {
        int(wf.window_id)
        for wf in spectrum_fit.window_fits
        if wf.window_id is not None and wf.fitted_peaks
    }
    points: Dict[int, List[float]] = {}
    plan_by_id = {int(w.window_id): w for w in base_plan_windows}
    for wid in flagged:
        win = plan_by_id.get(wid)
        if win is None:
            continue
        for r in statuses[wid].attention_reasons:
            if r.kind in EMPTY_WINDOW_KINDS:
                sides = [str(e.get("side")) for e in r.evidence.get("edges", [])]
                points[wid] = takeover_points(win, r.locations, sides)
    gone = (
        live
        | edited_window_ids
        | superseded_window_ids(base_plan_windows, created_windows, points)
    )
    for wid in flagged:
        if wid not in gone:
            continue
        st = statuses[wid]
        kept = [r for r in st.attention_reasons if r.kind not in EMPTY_WINDOW_KINDS]
        if not kept and wid not in fit_ids:
            del statuses[wid]
        else:
            statuses[wid] = replace(st, attention_reasons=kept)


def _write_stage6_review_only(review: Stage6Review, path: str) -> None:
    """Write *review* to ``/stage6_review`` verbatim -- no derivation.

    The review half of :func:`_finish_batch`, which persists a review
    :func:`_curate` already derived (D4's staged-preview reuse persists
    exactly what an immediately-preceding preview computed).

    Every writer of ``/stage6_review`` goes through here, so a stored
    final-products table always carries the clock declaration its calibration
    state was derived from (:func:`_resolve_calibration_clocks`, read now: the
    table being written is consistent with the file's current calibration).
    """
    clocks = (
        _resolve_calibration_clocks(path) if review.final_products is not None else None
    )
    with h5open(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp, calibration_clocks=clocks)


def _check_merge_arity(peaks: Sequence[float]) -> None:
    """A merge collapses a set into one line, so it needs a set to collapse."""
    if len(peaks) < 2:
        raise BadSettingError(
            "peaks",
            "at least 2 peak frequencies",
            list(peaks),
            message=f"merge requires at least 2 peak frequencies; got {len(peaks)}",
        )


def _check_split_arity(into: int) -> None:
    """A split replaces one line with several, so ``into`` must be at least 2."""
    if into < 2:
        raise BadSettingError(
            "into",
            "an integer >= 2",
            into,
            message=f"split requires into >= 2; got {into}",
        )


# ---------------------------------------------------------------------------
# The one write path: every Stage 6 write builds the new log L' and review
# parameters P', then curates -- computes the state the reference replay of
# L' under P' describes, from the automatic fit -- and persists it whole.
#
# A request (an interactive verb, a curation file, an action batch) is first
# resolved into decision rows against the displayed state (the persisted
# curated fit, :func:`_build_batch_ctx`), fitting nothing; an undo drops rows
# by serial; ``review run`` changes only P. :func:`_curate` then validates
# L' structurally (:func:`_walk_log_rows`, before any fit), keys every window
# by what its fit is computed from, gates on the analysis epoch only when it
# refits, and recomputes the windows whose keys changed as a replay of L' from
# the automatic fit in one batch computes them (the incremental engine,
# :func:`_engine_plan` / :func:`_engine_run`), keeping every other window's
# persisted fit. A write whose fit-changing rows are the persisted log's (a
# bare accept, ``review run``, an undo of bare accepts) refits nothing and
# keeps the persisted fits; one that leaves no fit-changing row restores the
# automatic fit. Every refusal is raised before the first fit; a failure
# after it (a cancel, a numerical error) discards the whole call with its
# transaction (``atomic_write``).
#
# The reference oracle (:func:`~.replay_reference.replay_full`) is the
# one-batch replay itself (:func:`_reference`), with nothing persisted read;
# after every write the persisted state is, bit for bit, the reference replay
# of the persisted log under the persisted parameters (design G1/G2, which the
# engine's property tests check).
# ---------------------------------------------------------------------------


def _open_batch(
    path: str, *, snapshot: bool = True, lineage_id: Optional[str] = None
) -> bool:
    """Admit a Stage 6 write and take the undo baseline (design §6.1 step 1).
    Returns whether the baseline is in place for the write to persist.

    The caller has refused a file this engine cannot curate
    (:func:`_require_engine_file`). With no automatic-fit baseline yet and an
    empty decision log, ``/stage5_fitting`` is the automatic fit: it is
    snapshotted and a fresh lineage id stamped on it. No baseline under a
    non-empty log cannot happen on a file this engine wrote (it snapshots
    before it records the first row): the file is corrupt.

    ``snapshot=False`` is for the read-only openers (a preview): they take no
    baseline, since that is a write, and their result must never be
    persisted (:func:`_finish_batch` refuses it). The call stays textually
    inside this function either way, so the baseline is taken in exactly one
    place. *lineage_id* is the lineage a staged preview computed its engine
    keys under (:attr:`_Curated.pending_lineage`), stamped when this write
    takes the baseline so the persisted keys stay valid.
    """
    if not snapshot:
        return False
    with h5open(path, "r") as h5f:
        has_baseline = STAGE5_BASELINE_GROUP in h5f
    if not has_baseline:
        if load_stage6_review_from_file(path).decision_log:
            raise _missing_baseline_error(path)
        # A write abandoned before its persist (a refusal, a cancel, a failing
        # events callback) takes the snapshot back with everything else: the
        # whole call is one transaction (atomic_write).
        _snapshot_stage5_baseline(path, lineage_id)
    return True


def _missing_baseline_error(path: str) -> PipelineCorruptionError:
    """A file that records decisions but holds no automatic-fit baseline: no
    engine write leaves one (it snapshots before recording the first row)."""
    return PipelineCorruptionError(
        Path(path),
        "the file records Stage 6 decisions but holds no automatic-fit "
        "baseline to replay them from (the snapshot was removed)",
    )


def _automatic_fit_group(path: str) -> str:
    """The group holding the automatic fit: the undo baseline once a write
    has taken it, else ``/stage5_fitting``, which no write has curated."""
    with h5open(path, "r") as h5f:
        return (
            STAGE5_BASELINE_GROUP if STAGE5_BASELINE_GROUP in h5f else "stage5_fitting"
        )


def _cascade_batch(ctx: _BatchCtx, *, snap_tol_mhz: float) -> List[int]:
    """Run a replay's one combined cascade, mutating
    ``ctx.changeset.spectrum_fit`` in place. Touches no file. Returns the
    cascaded window ids.

    Every window the cascade graph reaches from a dirty window is refreshed
    from its sources' final fits and identity-refit, in dependency order: the
    dependents of the replay's edited windows, and every created window with
    an edited ancestor (its starting skirt was read from the automatic fit,
    :func:`_created_window_seed`). Cascading once, after every row, is what
    makes a window's final fit independent of where its sources' rows sit in
    the log.
    """
    sources = _cascade_sources(
        ctx.shared.base_cascade_sources, ctx.changeset.created_windows
    )
    edited = set(ctx.changeset.dirty_wids)
    reached = sorted(
        w for w in ctx.changeset.created_wids if _cascade_ancestors(w, sources) & edited
    )
    cascaded: List[int] = []
    if edited or reached:
        cascaded = _cascade_refit_dependents(
            spectrum_fit=ctx.changeset.spectrum_fit,
            edited_wids=sorted(edited),
            sources=sources,
            reached=reached,
            fit_window_map=ctx.changeset.fit_window_map,
            fit_ctx=ctx.shared.fit_ctx,
            resolved=ctx.shared.resolved,
            shape_enum=ctx.shared.shape_enum,
            persisted_cal=ctx.shared.persisted_cal,
            tau_maj_us=ctx.shared.tau_maj_global,
            sigma_tau_us=ctx.shared.sigma_tau_global,
            tau_source=ctx.shared.tau_source,
            peak_frequencies_mhz=ctx.shared.peak_frequencies_mhz,
            min_freeze_snr=ctx.shared.min_freeze_snr,
            snap_tol_mhz=snap_tol_mhz,
            events=ctx.events,
            unavailable_window_ids=ctx.shared.unavailable_window_ids,
        )
        if cascaded:
            logger.info(
                "Stage 6 cascade: re-fit %d dependent window(s) %s",
                len(cascaded),
                sorted(cascaded),
            )
    return cascaded


def _skip_resolved_action(ctx: _BatchCtx, ra: _ResolvedAction) -> None:
    """Record the rows of a replay's action without fitting it: one
    placeholder per row it would have recorded (``evidence`` ``None``), so
    every later row's serial and record stay aligned with the log. The
    engine skips the actions of a window whose fit it keeps."""
    rows: List[Tuple[int, str]] = []
    if ra.create is not None:
        if ra.create.record:
            rows.append((int(ra.create.proposal.window.window_id), "create_window"))
    else:
        rows.extend((st.window_id, r.kind) for st in ra.steps for r in st.rows)
    ctx.changeset.decisions.extend(
        {"window_id": wid, "kind": kind, "evidence": None} for wid, kind in rows
    )


def _resolved_action_window(ra: _ResolvedAction) -> Optional[int]:
    """The window a replay's action fits (``None`` for a bare accept)."""
    if ra.create is not None:
        return int(ra.create.proposal.window.window_id)
    if ra.steps:
        return int(ra.steps[0].window_id)
    return None


def _run_resolved_actions(
    ctx: _BatchCtx,
    resolved: Sequence[_ResolvedAction],
    *,
    snap_tol_mhz: float,
    only: Optional[Collection[int]] = None,
) -> None:
    """Fit and record a replay's *resolved* actions in order, reporting a
    ``WindowProgress`` per action that targets a window and honouring a
    cancel before each.

    *only*, when given, names the windows whose actions are fit (the
    engine's: the windows whose direct phase it recomputes); every other
    window's actions are recorded as placeholders and not fit
    (:func:`_skip_resolved_action`). A replay's actions are each one
    window's, and a window's fit before the cascade depends on its own
    actions alone, so the windows it fits come out as a whole replay would
    leave them."""
    if only is not None:
        keep = {int(w) for w in only}
        run = [
            ra
            for ra in resolved
            if _resolved_action_window(ra) is None
            or _resolved_action_window(ra) in keep
        ]
    else:
        run = list(resolved)
    running = {id(ra) for ra in run}
    fit_total = sum(1 for ra in run if ra.target_wid is not None)
    fit_index = 0
    for ra in resolved:
        if id(ra) not in running:
            _skip_resolved_action(ctx, ra)
            continue
        if ctx.events is not None:
            ctx.events.check_cancel()
        t_action = time.monotonic()
        _run_resolved_action(ctx, ra, snap_tol_mhz=snap_tol_mhz)
        if ra.target_wid is not None:
            fit_index += 1
            _report_window(
                ctx,
                ra.target_wid,
                index=fit_index,
                total=fit_total,
                elapsed_s=time.monotonic() - t_action,
            )


def _replay_log(
    path: str,
    shared: _SharedFitCtx,
    log: Sequence[DecisionLogEntry],
    *,
    fit_group: str,
    snap_tol_mhz: float,
    walk: Optional["_LogWalk"] = None,
) -> _BatchCtx:
    """Replay *log* from the automatic fit in *fit_group*, in memory: the
    load-time spread recovery (:func:`_seed_unresolved_spreads_from_diagnostics`),
    every row applied as recorded, by peak identity and in log order
    (:func:`_resolve_replay_rows`, one action per recorded user action, every
    refusal raised before any fit; *walk* is the symbolic pass when the
    caller has already run it), then one combined cascade
    (:func:`_cascade_batch`). Returns the replay batch, whose fit and overlay
    are the curated state's. Reads the automatic fit and the static inputs
    only."""
    with h5open(path, "r") as h5f:
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f[fit_group])
    _seed_unresolved_spreads_from_diagnostics(spectrum_fit, snap_tol_mhz=snap_tol_mhz)
    changeset = _BatchChangeset(
        spectrum_fit=spectrum_fit,
        created_windows=[],
        fit_window_map={
            w.window_id: w
            for w in _overlay_created_windows(shared.base_plan, []).windows
        },
        replayed=list(log),
    )
    ctx = _BatchCtx(shared=shared, changeset=changeset, events=_REVIEW_SCOPE.get())
    _run_resolved_actions(
        ctx, _resolve_replay_rows(ctx, path, log, walk=walk), snap_tol_mhz=snap_tol_mhz
    )
    _cascade_batch(ctx, snap_tol_mhz=snap_tol_mhz)
    return ctx


@dataclass
class _Curated:
    """A write's curated state, computed and not yet persisted: the window
    fits and the review that the log and parameters describe."""

    spectrum_fit: SpectrumFit
    review: Stage6Review
    fit_changed: bool
    """Whether the fits differ from the persisted ``/stage5_fitting`` (a
    refit, a restore of the automatic fit, a spread the load-time recovery
    seeded), so :func:`_finish_batch` rewrites it."""
    refit: bool
    """Whether computing the state refit anything (then the write was
    epoch-gated)."""
    baseline_taken: bool = False
    """Whether :func:`_open_batch` admitted the write with its baseline in
    place; ``False`` for a preview, which :func:`_finish_batch` refuses."""
    shared: Optional[_SharedFitCtx] = None
    """The shared fit context the state was computed with, if one was
    needed (its fit cache is handed the persisted fit)."""
    engine: Optional[Stage6EngineState] = None
    """The engine keys the state was computed under, which
    :func:`_finish_batch` writes to ``/stage6_engine`` when
    :attr:`write_engine` is set (``None`` then empties the store)."""
    write_engine: bool = False
    """Whether the write replaces ``/stage6_engine``; a write that keeps the
    persisted fits leaves the keys that describe them as they are."""
    pending_lineage: Optional[str] = None
    """The lineage id :attr:`engine` was keyed under when no baseline had
    been taken yet (a preview on a file no write has curated): the baseline
    a later persist of this state takes is stamped with it."""


def _fit_row_serials(log: Sequence[DecisionLogEntry]) -> List[int]:
    """The serials of *log*'s fit-changing rows, in log order. The curated
    fit depends on the log through these alone: a bare accept changes no
    fit, and rows are immutable, so two logs with the same fit-changing rows
    describe the same fits."""
    return [_serial_id(e) for e in log if e.kind in _FIT_EDIT_KINDS]


def _check_log_integrity(path: str, log: Sequence[DecisionLogEntry]) -> None:
    """Refuse a log no engine write could have produced (``file_corrupt``):
    a row without a serial, a serial used twice, or created window ids that
    do not increase along the log (:func:`_check_created_ids_monotone`).
    Only the engine writes the log, so each means a corrupt file."""
    seen: Set[int] = set()
    for e in log:
        if isinstance(e.serial, Absent) or int(e.serial) in seen:
            raise PipelineCorruptionError(
                Path(path),
                f"the decision log's row at position {e.order_index} "
                + (
                    "carries no serial"
                    if isinstance(e.serial, Absent)
                    else f"reuses serial {int(e.serial)}"
                ),
            )
        seen.add(int(e.serial))
    _check_created_ids_monotone(path, log)


def _curated_review(
    path: str,
    spectrum_fit: SpectrumFit,
    log: Sequence[DecisionLogEntry],
    created_windows: Sequence["FitWindow"],
    params: ReviewParams,
    *,
    fit_group: str,
    prior: Optional[Stage6Review] = None,
    unchanged: Collection[int] = (),
) -> Stage6Review:
    """The review a curated state carries: *log* (positions recomputed by
    the caller), the overlay, *params*, every window's status computed from
    the fits, the log and *params* (:func:`_curated_statuses`), and the
    final-products table built from the fits. The serial and window-id
    high-water marks never fall below *prior*'s (the file's review), so a
    serial or an undone create's id is never reused.

    *unchanged* names the windows whose fit is the one *prior* was computed
    from (the engine kept it): their per-line fit fields are read from
    *prior*'s final products when the calibration they depend on is
    unchanged -- the same values, without recomputing them. Every other part
    of the review, every status included, is computed afresh."""
    fid = load_fid_from_pipeline_impl(path)
    serials = [int(e.serial) for e in log if not isinstance(e.serial, Absent)]
    created = [int(e.window_id) for e in log if _create_row_mode(e) == "created"]
    review = Stage6Review(
        decision_log=list(log),
        final_products=_final_products_for_fit(
            path,
            spectrum_fit,
            fid,
            prior=None if prior is None else prior.final_products,
            unchanged=unchanged,
        ),
        created_windows=list(created_windows),
        next_serial=max(
            [0 if prior is None else int(prior.next_serial)] + [s + 1 for s in serials]
        ),
        window_id_high_water=max(
            [-1 if prior is None else int(prior.window_id_high_water)] + created
        ),
        review_params=params,
    )
    review.window_statuses = _curated_statuses(
        path,
        spectrum_fit,
        review,
        params,
        Sideband.coerce(fid.sideband),
        fit_group=fit_group,
    )
    return review


def _reference(
    path: str,
    log: Sequence[DecisionLogEntry],
    params: ReviewParams,
    *,
    recorded: int,
    shared: Optional[_SharedFitCtx],
    snap_tol_mhz: float,
    prior: Optional[Stage6Review] = None,
    walk: Optional["_LogWalk"] = None,
) -> _Curated:
    """The reference curated state of *log* under *params*: replayed in one
    batch from the automatic fit (:func:`_replay_log`), or, for a log with
    no fit-changing row, the automatic fit itself, with the review
    :func:`_curated_review` derives from it. Reads the automatic fit and the
    static inputs only -- no curated fit, no stored status.

    The first *recorded* rows are the file's own and are kept verbatim; the
    rest are a request's new rows, whose evidence the replay completes
    (:func:`_replayed_rows`).
    """
    log = list(log)
    _check_log_integrity(path, log)
    group = _automatic_fit_group(path)
    if any(e.kind in _FIT_EDIT_KINDS for e in log):
        if shared is None:
            shared = _build_shared_fit_ctx(path, fit_group=group)
        ctx = _replay_log(
            path, shared, log, fit_group=group, snap_tol_mhz=snap_tol_mhz, walk=walk
        )
        spectrum_fit = ctx.changeset.spectrum_fit
        _canonical_peak_order(spectrum_fit)
        overlay: List["FitWindow"] = list(ctx.changeset.created_windows)
        rows = _replayed_rows(ctx, recorded)
        refit = True
    else:
        with h5open(path, "r") as h5f:
            spectrum_fit = load_spectrum_fit_from_hdf5(h5f[group])
        _seed_unresolved_spreads_from_diagnostics(
            spectrum_fit, snap_tol_mhz=snap_tol_mhz
        )
        overlay = []
        rows = [replace(e, order_index=k) for k, e in enumerate(log)]
        refit = False
    return _Curated(
        spectrum_fit=spectrum_fit,
        review=_curated_review(
            path, spectrum_fit, rows, overlay, params, fit_group=group, prior=prior
        ),
        fit_changed=True,
        refit=refit,
        shared=shared,
    )


def _curate(
    path: str,
    prior: Stage6Review,
    log: Sequence[DecisionLogEntry],
    params: ReviewParams,
    *,
    recorded: int,
    shared: Optional[_SharedFitCtx],
    baseline_taken: bool,
    snap_tol_mhz: float,
) -> _Curated:
    """The curated state a write leaves: *log* (L') under *params* (P'),
    given the file's review *prior* (design §6.1 steps 2-5). Persists
    nothing.

    A log whose fit-changing rows are the persisted log's
    (:func:`_fit_row_serials`) refits nothing: the persisted fits are kept as
    they stand -- a write that refits nothing never rebuilds a fit under
    another analysis environment -- and only the review is recomputed from
    them. A log left with no fit-changing row restores the automatic fit, by
    copy. Any other log goes through the incremental engine: the symbolic
    pass over it and every window's keys (:func:`_engine_plan`, every refusal
    before any fit), the epoch gate
    (:func:`require_splice_compatible_environment`) when the write refits
    anything, then the windows whose keys changed recomputed as the reference
    (:func:`_reference`) computes them, the rest kept (:func:`_engine_run`).

    The first *recorded* rows of *log* are the file's (kept verbatim); the
    rest are new rows the replay records the evidence of.
    """
    log = list(log)
    _check_log_integrity(path, log)
    group = _automatic_fit_group(path)
    if group != STAGE5_BASELINE_GROUP and prior.decision_log:
        raise _missing_baseline_error(path)
    fit_rows = _fit_row_serials(log)
    prior_rows = _fit_row_serials(prior.decision_log)
    if fit_rows and fit_rows != prior_rows:
        if shared is None:
            shared = _build_shared_fit_ctx(path)
        # A preview with no baseline yet keys under the lineage id the
        # baseline its persist takes will carry (:func:`_open_batch`).
        pending = None if group == STAGE5_BASELINE_GROUP else uuid.uuid4().hex
        plan = _engine_plan(
            path, shared, log, snap_tol_mhz=snap_tol_mhz, pending_lineage=pending
        )
        if plan.refit:
            require_splice_compatible_environment(path)
        curated = _engine_run(
            path, plan, prior, params, recorded=recorded, snap_tol_mhz=snap_tol_mhz
        )
        return replace(curated, baseline_taken=baseline_taken, pending_lineage=pending)

    # Nothing to refit: the persisted fits stand (the same fit-changing rows),
    # or the automatic fit is restored (none left). Only the automatic fit
    # takes the load-time spread recovery (the reference applies it there,
    # before any row); a curated fit already carries what it recovered.
    restore = not fit_rows and bool(prior_rows)
    with h5open(path, "r") as h5f:
        spectrum_fit = load_spectrum_fit_from_hdf5(
            h5f[group if restore else "stage5_fitting"]
        )
    seeded = (
        _seed_unresolved_spreads_from_diagnostics(
            spectrum_fit, snap_tol_mhz=snap_tol_mhz
        )
        if not fit_rows
        else 0
    )
    overlay = [] if restore else list(prior.created_windows)
    rows = [replace(e, order_index=k) for k, e in enumerate(log)]
    return _Curated(
        spectrum_fit=spectrum_fit,
        review=_curated_review(
            path,
            spectrum_fit,
            rows,
            overlay,
            params,
            fit_group=group,
            prior=prior,
            unchanged=() if restore or seeded else _fit_by_window(spectrum_fit),
        ),
        fit_changed=restore or bool(seeded),
        refit=False,
        baseline_taken=baseline_taken,
        shared=shared,
        # The automatic fit restored: no window holds a key worth keeping (the
        # next refitting write recomputes or restores every window).
        write_engine=restore,
    )


# ---------------------------------------------------------------------------
# The incremental engine (design §5.3-5.5).
#
# A refitting write computes the same state as the reference
# (:func:`_reference`) but refits only the windows whose inputs changed. Each
# window is keyed by what its final fit is a function of: ``K_d`` covers its
# direct phase (the shared analysis context ``E_key``, its geometry sequence,
# its seed and its own rows), ``K_f`` adds whether a dirty ancestor reaches it
# and, when one does, its sources' ``K_f`` (:func:`_engine_window_keys`). The
# keys are persisted in ``/stage6_engine`` beside the fits they describe, so a
# window whose ``K_f`` a write leaves unchanged keeps its persisted fit
# bit for bit: by induction over the cascade graph, the same deterministic
# functions run on the same inputs. A window that changed and has rows (or is
# reached) is recomputed from its seed as the reference computes it -- its
# own actions replayed (:func:`_run_resolved_actions` with ``only``), then
# the cascade's refresh and identity refit (:func:`_cascade_refit_dependents`
# with ``only``) -- a changed window with neither is the automatic fit,
# restored by copy, and a window the log no longer installs is dropped.
# ---------------------------------------------------------------------------

#: The groups whose content stands in for the shared analysis context when the
#: file cannot account for every input its stages used (``E_key``'s degraded
#: form): the soft Stage 5 inputs that can change without invalidating Stage 5
#: (``file_manager.py``: Stage 2b and the timebase are recommended, not
#: required), and the shape recommendation Stage 2b leaves beside them.
_ENGINE_SOFT_CONTEXT = (
    "stage2b_tau_calibration",
    "stage2b_tau_G_calibration",
    "timebase_calibration",
    "processing_parameters/stage2b_shape_recommendation",
)


def _engine_digest(*parts: Any) -> str:
    """SHA-256 over the canonical JSON of *parts* (floats exact)."""
    from .fingerprint_impl import canonical_json

    return hashlib.sha256(canonical_json(list(parts))).hexdigest()


def _hdf5_value_bytes(value: Any) -> bytes:
    """One stored attribute or dataset value, exactly."""
    arr = np.asarray(value)
    if arr.dtype.kind in "OSU":
        return repr(arr.tolist()).encode("utf-8") + b"\0"
    head = f"{arr.dtype.str}{arr.shape!r}".encode()
    raw: bytes = arr.tobytes()
    return head + raw + b"\0"


def _hdf5_content_digest(path: str, names: Sequence[str]) -> str:
    """SHA-256 over every attribute and dataset under the groups *names* of
    *path* (an absent group hashes as absent)."""
    import h5py

    out = hashlib.sha256()

    def feed(name: str, obj: Any) -> None:
        out.update(name.encode("utf-8") + b"\0")
        for key in sorted(obj.attrs):
            out.update(str(key).encode("utf-8") + b"=")
            out.update(_hdf5_value_bytes(obj.attrs[key]))
        if isinstance(obj, h5py.Dataset):
            out.update(_hdf5_value_bytes(obj[()]))

    with h5open(path, "r") as h5f:
        for name in names:
            obj = h5f.get(name)
            if obj is None:
                out.update(f"{name}: absent\0".encode("utf-8"))
                continue
            feed(name, obj)
            if isinstance(obj, h5py.Group):
                members: List[str] = []
                obj.visit(members.append)
                for member in sorted(members):
                    feed(f"{name}/{member}", obj[member])
    return out.hexdigest()


def _engine_context_key(
    path: str,
    shared: _SharedFitCtx,
    snap_tol_mhz: float,
    pending_lineage: Optional[str] = None,
) -> str:
    """``E_key``: the digest of the shared analysis context every window's fit
    is computed in (design §5.5).

    The running code's ``ANALYSIS_EPOCH``; the lineage id of the automatic
    fit (which covers everything the baseline holds: its fits, the fitted
    plan, the spur catalog it replays and ``min_freeze_snr``); the per-file
    snap tolerance; the retired and unavailable window ids; and the inputs
    the file's completed stages record (:func:`~.fingerprint_impl.canonical_fingerprint_inputs`:
    settings, consumed blocks and stamped epochs, which carry the tau
    calibration, the band anchors, the timebase and the active-FT identity),
    without the review stage, whose sigma floor and clock declaration feed
    only the final products. With no baseline taken yet, the lineage is
    *pending_lineage*, the one the baseline will be stamped with. A file whose stages cannot account for every
    input they used degrades to a content digest of the soft Stage 5 inputs
    (:data:`_ENGINE_SOFT_CONTEXT`); its hard inputs need none, since re-running
    one deletes the fit and the review.
    """
    from ..core.environment import capture_environment
    from ..file_manager import IncompleteProvenanceError
    from .fingerprint_impl import canonical_fingerprint_inputs

    with h5open(path, "r") as h5f:
        raw = h5f[_automatic_fit_group(path)].attrs.get(LINEAGE_ID_ATTR)
    lineage = pending_lineage if raw is None else str(raw)
    context: Dict[str, Any]
    try:
        inputs = canonical_fingerprint_inputs(path)
        for key in ("review", "review_absent"):
            inputs.pop(key, None)
        context = {"fingerprint": inputs}
    except IncompleteProvenanceError:
        context = {"soft_inputs": _hdf5_content_digest(path, _ENGINE_SOFT_CONTEXT)}
    return _engine_digest(
        "E",
        capture_environment().analysis_epoch,
        lineage,
        float(snap_tol_mhz),
        sorted(int(w) for w in shared.retired_window_ids),
        sorted(int(w) for w in shared.unavailable_window_ids),
        context,
    )


def _window_json(fit_win: "FitWindow") -> str:
    """A window's geometry as one exact string: the overlay's serialized form
    (floats exact), keys sorted."""
    return json.dumps(_fit_window_to_dict(fit_win), sort_keys=True, default=str)


class _CreateChain:
    """The create-prefix cache (design §4.4): the digest chain
    ``c_k = H(c_{k-1}, anchor_k, id_k)`` over a log's create rows, rooted at
    ``E_key``, and the proposal each create installed. A create whose chain
    digest matches the persisted chain's at the same position reuses the
    persisted proposal instead of replanning it: planning reads the static
    inputs and the creates before it, nothing else, so a shared prefix plans
    to the same windows. The first mismatch replans that create and every
    later one."""

    def __init__(self, e_key: str, cached: Sequence[Mapping[str, Any]]) -> None:
        self.digest = _engine_digest("creates", e_key)
        self.cached = list(cached)
        self.entries: List[Dict[str, Any]] = []
        self.reused = 0

    def plan(
        self,
        shared: _SharedFitCtx,
        created_windows: Sequence["FitWindow"],
        anchor_mhz: float,
        *,
        replay_window_id: Optional[int],
        min_new_window_id: int = 0,
    ) -> "Stage6WindowProposal":
        """:func:`_plan_create`, or the cached proposal of this create."""
        from ..preprocessing.window_planning import Stage6WindowProposal

        k = len(self.entries)
        self.digest = _engine_digest(self.digest, float(anchor_mhz), replay_window_id)
        hit = (
            self.reused == k
            and k < len(self.cached)
            and self.cached[k].get("digest") == self.digest
        )
        if hit:
            entry = self.cached[k]
            window = _fit_window_from_dict(entry["window"])
            proposal = Stage6WindowProposal(
                window=window,
                mode=str(entry["mode"]),
                depends_on=[int(d) for d in entry.get("depends_on", [])],
                diagnostics=window.diagnostics,
            )
            self.reused += 1
        else:
            proposal = _plan_create(
                shared,
                created_windows,
                anchor_mhz,
                replay_window_id=replay_window_id,
                min_new_window_id=min_new_window_id,
            )
        self.entries.append(
            json.loads(
                json.dumps(
                    {
                        "digest": self.digest,
                        "window": _fit_window_to_dict(proposal.window),
                        "mode": proposal.mode,
                        "depends_on": [int(d) for d in proposal.depends_on],
                    },
                    default=str,
                )
            )
        )
        return proposal


@dataclass
class _EnginePlan:
    """What a refitting write will do, decided before any fit
    (:func:`_engine_plan`)."""

    ctx: _BatchCtx
    """The replay batch, the log's structure installed and no fit loaded."""
    resolved: List[_ResolvedAction]
    walk: _LogWalk
    sources: Dict[int, Tuple[int, ...]]
    """The cascade graph: every window of the curated state, each with its
    ordered sources."""
    reached: Set[int]
    state: Stage6EngineState
    """The keys the write leaves, and the ``E_key`` they are computed under."""
    direct: Set[int]
    """Windows whose direct phase is recomputed: changed, with rows."""
    cascade: Set[int]
    """Windows the cascade refits: changed and reached."""
    restore: Set[int]
    """Windows restored from the automatic fit by copy: changed, with no
    rows, not reached."""
    persisted: Optional[SpectrumFit]
    """The persisted fit, when a window keeps its persisted fit."""

    @property
    def refit(self) -> Set[int]:
        """The refit set (design §6.1 step 3): what the epoch gate guards."""
        return self.direct | self.cascade


def _engine_window_rows(
    resolved: Sequence[_ResolvedAction],
    base_live: Collection[int],
    created_ids: Collection[int],
) -> Tuple[Dict[int, List[Any]], Dict[int, Any]]:
    """Each window's fit-relevant rows and its seed, as ``K_d`` hashes them.

    The rows are the window's replayed actions in log order, as the replay
    runs them: a create (its serial, anchor, mode, whether it records a row,
    and the geometry it installs) or a refit step (its kind and each row's
    serial, kind, targets, seeds and born uids; the step partition is the
    replay's own, so the action grouping is in it). A step's geometry is the
    last install before it, so the sequence carries every geometry the window
    is fit on. A bare accept fits nothing and is left out. The seed is
    ``("baseline", w)`` for a base window and ``("create", sources)`` for a
    created one (the sources its starting skirt is read from,
    :func:`_created_window_seed`)."""
    rows: Dict[int, List[Any]] = {}
    seeds: Dict[int, Any] = {}
    for ra in resolved:
        if ra.create is not None:
            proposal = ra.create.proposal
            w = int(proposal.window.window_id)
            rows.setdefault(w, []).append(
                [
                    "create",
                    float(ra.create.anchor),
                    proposal.mode,
                    bool(ra.create.record),
                    _window_json(proposal.window),
                ]
            )
            if proposal.mode == "created":
                seeds[w] = [
                    "create",
                    list(
                        _created_window_sources(proposal.window, base_live, created_ids)
                    ),
                ]
            continue
        for step in ra.steps:
            rows.setdefault(int(step.window_id), []).append(
                [
                    "step",
                    step.kind,
                    [
                        [
                            r.serial,
                            r.kind,
                            list(r.targets),
                            [float(f) for f in r.seeds_mhz],
                            list(r.born_uids),
                        ]
                        for r in step.rows
                    ],
                ]
            )
    return rows, seeds


def _engine_window_keys(
    e_key: str,
    sources: Mapping[int, Sequence[int]],
    rows: Mapping[int, List[Any]],
    seeds: Mapping[int, Any],
    dirty: Collection[int],
    reached: Collection[int],
) -> Dict[int, EngineWindowKey]:
    """Every window's ``K_d`` and ``K_f`` under *e_key* (design §5.3)::

        K_d(w) = H("d", E_key, seed(w), rows(w))
        K_f(w) = H("f", K_d(w), reached(w),
                   [(p, K_f(p)) for p in sources(w)] if reached(w) else [])

    All of a reached window's sources enter its ``K_f``, untouched ones
    included: its refresh reads every one of them."""
    order = _cascade_topo(set(sources), {w: set(ps) for w, ps in sources.items()})
    keys: Dict[int, EngineWindowKey] = {}
    for w in order:
        seed = seeds.get(w, ["baseline", w])
        kd = _engine_digest("d", e_key, seed, rows.get(w, []))
        hit = w in reached
        preds = [[int(p), keys[int(p)].kf] for p in sources[w]] if hit else []
        keys[w] = EngineWindowKey(
            kd=kd,
            kf=_engine_digest("f", kd, hit, preds),
            reached=hit,
            dirty=w in dirty,
        )
    return keys


def _engine_plan(
    path: str,
    shared: _SharedFitCtx,
    log: Sequence[DecisionLogEntry],
    *,
    snap_tol_mhz: float,
    pending_lineage: Optional[str] = None,
) -> _EnginePlan:
    """Decide, before any fit, what a refitting write of *log* recomputes
    (design §6.1 steps 2-3).

    The symbolic pass over the log (:func:`_walk_log_rows`, every refusal;
    its creates through the create-prefix cache, :class:`_CreateChain`), the
    replay's actions (:func:`_resolve_replay_rows`), the cascade graph, the
    dirty and reached windows, and every window's keys. The keys are computed
    under the stored ``E_key``, so the windows whose ``K_f`` changed reflect
    the log and the structure alone; when that refits anything and the
    analysis context changed since the keys were stored, every window is
    recomputed instead, keyed under the new context. With no stored keys,
    every window is recomputed or restored. A window whose ``K_f`` is
    unchanged keeps its persisted fit, unless the persisted fit has no entry
    for it. *pending_lineage* is :func:`_engine_context_key`'s."""
    with h5open(path, "r") as h5f:
        stored = load_stage6_engine_state(h5f)
    e_now = _engine_context_key(path, shared, snap_tol_mhz, pending_lineage)
    chain = _CreateChain(e_now, [] if stored is None else stored.create_chain)
    walk = _walk_log_rows(path, shared, log, min_new_window_id=0, creates=chain)
    ctx = _BatchCtx(
        shared=shared,
        changeset=_BatchChangeset(
            spectrum_fit=SpectrumFit(),
            created_windows=[],
            fit_window_map={
                w.window_id: w
                for w in _overlay_created_windows(shared.base_plan, []).windows
            },
            replayed=list(log),
        ),
        events=_REVIEW_SCOPE.get(),
    )
    resolved = _resolve_replay_rows(ctx, path, log, walk=walk)

    base_live = set(shared.base_cascade_sources)
    sources = _cascade_sources(shared.base_cascade_sources, walk.overlay)
    created_ids = set(sources) - base_live
    dirty = {int(e.window_id) for e in log if e.kind in _PEAK_ROW_KINDS} | {
        int(p.window.window_id) for p in walk.proposals.values() if p.mode == "widened"
    }
    reached = _cascade_closure_set(sorted(dirty), sources)
    rows, seeds = _engine_window_rows(resolved, base_live, created_ids)

    persisted: Optional[SpectrumFit] = None
    held: Set[int] = set()
    if stored is not None:
        with h5open(path, "r") as h5f:
            persisted = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        held = set(_fit_by_window(persisted))

    def decide(
        e_key: str, old: Mapping[int, EngineWindowKey]
    ) -> Tuple[Dict[int, EngineWindowKey], Set[int], Set[int]]:
        keys = _engine_window_keys(e_key, sources, rows, seeds, dirty, reached)
        changed = {
            w
            for w in sources
            if w not in held or w not in old or old[w].kf != keys[w].kf
        }
        recompute = {w for w in changed if w in rows or w in reached}
        return keys, changed, recompute

    basis = e_now if stored is None else stored.e_key
    keys, changed, recompute = decide(basis, {} if stored is None else stored.keys)
    if recompute and basis != e_now:
        basis = e_now
        keys, changed, recompute = decide(basis, {})
    if set(sources) <= changed:
        persisted = None  # no window keeps its persisted fit
    return _EnginePlan(
        ctx=ctx,
        resolved=resolved,
        walk=walk,
        sources=sources,
        reached=reached,
        state=Stage6EngineState(e_key=basis, create_chain=chain.entries, keys=keys),
        direct={w for w in recompute if w in rows},
        cascade={w for w in recompute if w in reached},
        restore=changed - recompute,
        persisted=persisted,
    )


def _canonical_peak_order(spectrum_fit: SpectrumFit) -> None:
    """Order *spectrum_fit* as a load from the file orders it: window fits by
    ascending id, and the global line list each window's lines in that order,
    stably sorted by frequency. The final products follow the line list, so
    lines at the same frequency must not be ordered by how the state was
    computed."""
    spectrum_fit.window_fits.sort(
        key=lambda wf: (wf.window_id if wf.window_id is not None else -1)
    )
    peaks: List[FittedPeak] = []
    for wf in spectrum_fit.window_fits:
        peaks.extend(wf.fitted_peaks)
    peaks.sort(key=lambda p: p.frequency_mhz)
    spectrum_fit.fitted_peaks = peaks


def _engine_run(
    path: str,
    plan: _EnginePlan,
    prior: Stage6Review,
    params: ReviewParams,
    *,
    recorded: int,
    snap_tol_mhz: float,
) -> _Curated:
    """Compute the state *plan* decided (design §6.1 steps 4-5): the direct
    phase of every window whose rows it replays, from the automatic fit (the
    load-time spread recovery applied, as the reference applies it); the
    cascade over the changed reached windows, in dependency order, each
    refreshed from its sources' final fits (recomputed, or persisted); every
    other changed window restored from the automatic fit; every unchanged
    one kept as persisted. The review is computed from the result as the
    reference computes it. Touches no file."""
    ctx = plan.ctx
    group = _automatic_fit_group(path)
    fits: Dict[int, FittingResult] = {}
    recomputed = plan.direct | plan.cascade | plan.restore
    container: SpectrumFit
    if plan.persisted is None:
        with h5open(path, "r") as h5f:
            container = load_spectrum_fit_from_hdf5(h5f[group])
        start = container
    else:
        # The windows to recompute start from copies of the automatic fit the
        # shared context holds (it is never mutated), in a fit carrying the
        # header the persisted one shares with it: the spread recovery reads
        # its diagnostics, window by window.
        container = plan.persisted
        base = ctx.shared.baseline_fits
        start = SpectrumFit(
            window_fits=[
                copy.deepcopy(base[w]) for w in sorted(recomputed) if w in base
            ],
            diagnostics=container.diagnostics,
        )
        _canonical_peak_order(start)
    _seed_unresolved_spreads_from_diagnostics(start, snap_tol_mhz=snap_tol_mhz)
    ctx.changeset.spectrum_fit = start
    _run_resolved_actions(
        ctx, plan.resolved, snap_tol_mhz=snap_tol_mhz, only=plan.direct
    )
    computed = _fit_by_window(ctx.changeset.spectrum_fit)
    kept = {} if plan.persisted is None else _fit_by_window(plan.persisted)
    for w in sorted(plan.sources):
        fits[w] = computed[w] if w in recomputed else kept[w]
    container.window_fits = [fits[w] for w in sorted(fits)]
    _canonical_peak_order(container)
    if plan.cascade:
        shared = ctx.shared
        _cascade_refit_dependents(
            spectrum_fit=container,
            edited_wids=[],
            sources=plan.sources,
            fit_window_map=ctx.changeset.fit_window_map,
            fit_ctx=shared.fit_ctx,
            resolved=shared.resolved,
            shape_enum=shared.shape_enum,
            persisted_cal=shared.persisted_cal,
            tau_maj_us=shared.tau_maj_global,
            sigma_tau_us=shared.sigma_tau_global,
            tau_source=shared.tau_source,
            peak_frequencies_mhz=shared.peak_frequencies_mhz,
            min_freeze_snr=shared.min_freeze_snr,
            snap_tol_mhz=snap_tol_mhz,
            events=ctx.events,
            unavailable_window_ids=shared.unavailable_window_ids,
            only=plan.cascade,
        )
        _canonical_peak_order(container)
    held = set(kept) if plan.persisted is not None else None
    fit_changed = bool(recomputed) or held is None or held != set(fits)
    return _Curated(
        spectrum_fit=container,
        review=_curated_review(
            path,
            container,
            _replayed_rows(ctx, recorded),
            list(ctx.changeset.created_windows),
            params,
            fit_group=group,
            prior=prior,
            unchanged=set(fits) - recomputed if held is not None else (),
        ),
        fit_changed=fit_changed,
        refit=bool(plan.refit),
        shared=ctx.shared,
        engine=plan.state,
        write_engine=True,
    )


def _finish_batch(curated: _Curated, path: str) -> None:
    """Persist a write's curated state (design §6.1 step 6): the window fits,
    rewritten whole in their canonical order when they changed, then the
    review, then the engine keys the fits were computed under
    (``/stage6_engine``, when the write replaces them; resized in place, as
    the fit tables are). The engine's one and only writer of
    ``/stage5_fitting`` -- see
    ``test_only_finish_batch_persists_the_fit``. Runs inside the call's
    ``atomic_write`` transaction.

    A cancel is honoured before the write begins, never once it has.

    Refuses a state whose write did not take the undo baseline
    (:func:`_open_batch` with ``snapshot=False``, a preview): persisting
    curated fits without the automatic fit behind them would make every later
    write replay from a curated fit as if it were the automatic one.
    """
    from ..io.fitting_serialization import (
        save_spectrum_fit_to_hdf5,
        update_spectrum_fit_windows_in_hdf5,
    )

    if not curated.baseline_taken:
        raise ValueError(
            "refusing to persist a curated state whose undo baseline was never "
            "taken (_open_batch(..., snapshot=False)); this is a preview-only "
            "state and must never reach _finish_batch"
        )
    # The last cancel check point: past it, the write completes.
    _check_cancel()
    fit = curated.spectrum_fit
    if curated.fit_changed:
        shape_attr = str(fit.parameters.get("shape", "lorentzian"))
        # The two flat tables are rewritten whole, resized in place (S4):
        # deleting an HDF5 group does not return its space. A file with no
        # fit yet takes the full writer.
        with h5open(path, "a") as h5f:
            if "stage5_fitting" not in h5f:
                grp = h5f.create_group("stage5_fitting")
                save_spectrum_fit_to_hdf5(fit, grp)
            else:
                grp = h5f["stage5_fitting"]
                update_spectrum_fit_windows_in_hdf5(
                    fit,
                    grp,
                    [
                        int(wf.window_id)
                        for wf in fit.window_fits
                        if wf.window_id is not None
                    ],
                )
            grp.attrs["shape"] = shape_attr
    _write_stage6_review_only(curated.review, path)
    if curated.write_engine:
        with h5open(path, "a") as h5f:
            save_stage6_engine_state(h5f, curated.engine)
    # S5: publish the just-persisted fit for the next write in this same
    # session, stamped with the fit group's write stamp so the next write can
    # tell whether anything has rewritten it since.
    if curated.shared is not None:
        curated.shared.fit_cache.install(fit, path)


# ---------------------------------------------------------------------------
# Requests: resolve against the displayed state, then curate.
# ---------------------------------------------------------------------------


def _plan_needs_fit(plan: Sequence[PlannedAction]) -> bool:
    """Whether any action of *plan* changes a fit (every action but a bare
    ``accept``)."""
    return any(a.kind != "accept" or a.candidate is not None for a in plan)


def _resolve_request(
    ctx: _BatchCtx,
    plan: Sequence[PlannedAction],
    *,
    attribute: bool,
    snap_tol_mhz: float,
) -> List[_ResolvedAction]:
    """Resolve *plan* against the display batch *ctx*, fitting nothing
    (:func:`_resolve_action`), each later action against the earlier ones'
    rows (design D9); every create's structure is installed as it is
    planned. A curation plan (*attribute*) is resolved in canonical order
    (:func:`_canonicalize_batch_plan`, so a file's row order cannot change its
    outcome) and a refusal is tagged with the action it came from
    (:func:`_raise_curation_failure`); an interactive verb's actions run in
    the given order and refuse untagged (there is one request)."""
    state = _ResolveState()
    order = _canonicalize_batch_plan(plan) if attribute else list(enumerate(plan))
    resolved: List[_ResolvedAction] = []
    for original_index, action in order:
        try:
            resolved.append(
                _resolve_action(
                    ctx, state, original_index, action, snap_tol_mhz=snap_tol_mhz
                )
            )
        except (ValueError, KeyError) as exc:
            if not attribute:
                raise
            _raise_curation_failure(original_index, action, exc)
    return resolved


def _request_rows(
    ctx: _BatchCtx, resolved: Sequence[_ResolvedAction], *, one_action: bool
) -> List[DecisionLogEntry]:
    """The decision rows *resolved* records, in order, with fresh serials
    from the file's high-water mark (``changeset.base_serial``).

    An explicit create records a ``create_window`` row (its structure as
    evidence); an implied create records nothing of its own, its one add
    carrying the structure (``created_window``); a bare accept records an
    ``accept`` row at its anchor; an action's refits record their rows.
    Every row of one user action carries the serial of the action's first
    row as :data:`ACTION_INDEX_EVIDENCE_KEY`: each resolved action is one
    user action, or the whole request is (*one_action*, an interactive verb
    call). Positions (``order_index``) are assigned by the write."""
    rows: List[DecisionLogEntry] = []
    serial = int(ctx.changeset.base_serial)
    first: Optional[int] = None

    def record(
        window_id: int,
        frequency_mhz: float,
        kind: str,
        evidence: Mapping[str, Any],
        targets: Sequence[int] = (),
        seeds: Sequence[float] = (),
        born: Sequence[int] = (),
    ) -> None:
        nonlocal serial, first
        if first is None:
            first = serial
        rows.append(
            DecisionLogEntry(
                order_index=0,
                window_id=int(window_id),
                frequency_mhz=float(frequency_mhz),
                kind=kind,
                provenance="user",
                evidence={**evidence, ACTION_INDEX_EVIDENCE_KEY: first},
                serial=serial,
                targets=tuple(int(u) for u in targets),
                seeds_mhz=tuple(float(f) for f in seeds),
                born_uids=tuple(int(u) for u in born),
            )
        )
        serial += 1

    for ra in resolved:
        if not one_action:
            first = None
        if ra.create is not None:
            if ra.create.record:
                record(
                    int(ra.create.proposal.window.window_id),
                    ra.create.anchor,
                    "create_window",
                    _created_window_evidence(ra.create.proposal),
                )
            continue
        if ra.accept:
            record(ra.action.window_id, ra.accept_anchor, "accept", {})
            continue
        for step in ra.steps:
            for r in step.rows:
                evidence = dict(r.evidence)
                if ra.implied is not None:
                    evidence["inferred"] = True
                    evidence["created_window"] = _created_window_evidence(
                        ra.implied.proposal
                    )
                record(
                    r.window_id,
                    r.frequency_mhz,
                    r.kind,
                    evidence,
                    r.targets,
                    r.seeds_mhz,
                    r.born_uids,
                )
    return rows


def _bare_accept_rows(
    path: str, prior: Stage6Review, window_ids: Sequence[int]
) -> List[DecisionLogEntry]:
    """The rows of bare accepts of *window_ids* (already checked against the
    file's windows), each its own action, with fresh serials from *prior*'s
    high-water mark. Reads the persisted fit for the anchors and needs no
    fit context: a bare accept changes no fit."""
    with h5open(path, "r") as h5f:
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    fits_by_wid = {
        int(wf.window_id): wf
        for wf in spectrum_fit.window_fits
        if wf.window_id is not None
    }
    rows: List[DecisionLogEntry] = []
    serial = int(prior.next_serial)
    for window_id in window_ids:
        wf = fits_by_wid.get(int(window_id))
        anchor = (
            _accept_anchor(wf, None)
            if wf is not None
            else _plan_window_center(path, window_id, 0.0)
        )
        rows.append(
            DecisionLogEntry(
                order_index=0,
                window_id=int(window_id),
                frequency_mhz=anchor,
                kind="accept",
                provenance="user",
                # A one-row action (see ACTION_INDEX_EVIDENCE_KEY).
                evidence={ACTION_INDEX_EVIDENCE_KEY: serial},
                serial=serial,
                targets=(),
                seeds_mhz=(),
                born_uids=(),
            )
        )
        serial += 1
    return rows


@dataclass
class _CuratedRequest:
    """A request resolved against the displayed state and the state it
    curates to: what a verb's result is read from."""

    display: Optional[_BatchCtx]
    """The display batch the request resolved against (``None`` for a
    request of bare accepts, which needs no fit context)."""
    resolved: List[_ResolvedAction]
    curated: _Curated


def _curate_request(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    snap_tol_mhz: float,
    shared: Optional[_SharedFitCtx],
    persist: bool,
    one_action: bool,
) -> _CuratedRequest:
    """One request, start to finish: admit the write and take the baseline
    (:func:`_open_batch`; a preview, *persist* False, takes none), resolve
    *plan* against the displayed state into rows (:func:`_resolve_request`,
    :func:`_request_rows`; a plan of bare accepts needs no fit context,
    :func:`_bare_accept_rows`), curate the log with the rows appended under
    the recorded review parameters (:func:`_curate`), and persist it
    (:func:`_finish_batch`) unless it is a preview."""
    baseline_taken = _open_batch(path, snapshot=persist)
    _check_cancel()
    prior = load_stage6_review_from_file(path)
    display: Optional[_BatchCtx] = None
    resolved: List[_ResolvedAction] = []
    if _plan_needs_fit(plan):
        display = _build_batch_ctx(path, snap_tol_mhz=snap_tol_mhz, shared=shared)
        shared = display.shared
        if not one_action:
            _require_known_plan_windows(
                _batch_known_window_ids(
                    display,
                    plan,
                    {
                        int(wf.window_id)
                        for wf in display.changeset.spectrum_fit.window_fits
                        if wf.window_id is not None
                    },
                ),
                plan,
                "Stage 5 fit",
                _batch_plan_window_ids(display, plan),
            )
        resolved = _resolve_request(
            display, plan, attribute=not one_action, snap_tol_mhz=snap_tol_mhz
        )
        rows = _request_rows(display, resolved, one_action=one_action)
    else:
        known, where = _known_window_ids(path)
        _require_known_plan_windows(known, plan, where)
        rows = _bare_accept_rows(path, prior, [a.window_id for a in plan])
    curated = _curate(
        path,
        prior,
        list(prior.decision_log) + rows,
        _recorded_review_params(prior),
        recorded=len(prior.decision_log),
        shared=shared,
        baseline_taken=baseline_taken,
        snap_tol_mhz=snap_tol_mhz,
    )
    if persist:
        _finish_batch(curated, path)
    return _CuratedRequest(display=display, resolved=resolved, curated=curated)


def _fit_by_window(spectrum_fit: SpectrumFit) -> Dict[int, FittingResult]:
    return {
        int(wf.window_id): wf
        for wf in spectrum_fit.window_fits
        if wf.window_id is not None
    }


def _fit_signature(wf: FittingResult) -> Tuple[Any, ...]:
    """What tells two fits of a window apart, bitwise: its reduced
    chi-squared and each line's identity, position and amplitude."""
    return (
        float(wf.reduced_chi2).hex(),
        tuple(
            (
                p.peak_uid,
                float(p.frequency_mhz).hex(),
                float(p.amplitude).hex(),
            )
            for p in wf.fitted_peaks
        ),
    )


def _refit_result(req: _CuratedRequest, ra: _ResolvedAction) -> RefitWindowResult:
    """The result of an interactive verb's one action *ra*: its window before
    the write (as displayed) and after it (the curated fit, post-cascade),
    with the structure an implied create installed."""
    assert req.display is not None and ra.target_wid is not None
    wid = int(ra.target_wid)
    before = _displayed_wf(req.display, wid)
    after = _fit_by_window(req.curated.spectrum_fit)[wid]
    result = _make_refit_result(
        req.display,
        window_id=wid,
        n_peaks_before=0 if before is None else len(before.fitted_peaks),
        n_peaks_after=len(after.fitted_peaks),
        chi2r_before=Absent.NOT_RUN if before is None else float(before.reduced_chi2),
        chi2r_after=float(after.reduced_chi2),
        fitted_peaks=list(after.fitted_peaks),
        converged=_converged_or_absent(after),
    )
    if ra.implied is not None:
        # W4: the structural consequence, on the result the caller already
        # holds -- a caller of review_edit learns a window was built (or
        # widened), and where.
        planned = _planned_window_result(ra.implied)
        result.created_window_mode = planned.mode
        result.created_window_freq_range = planned.freq_range
        result.created_window_n_points = planned.n_points
        result.created_window_n_contributors = planned.n_contributors
        result.created_window_depends_on = list(planned.depends_on)
    return result


def _request_outcome(
    req: _CuratedRequest,
) -> Tuple[Dict[int, List[int]], Set[int], Set[int], Dict[int, _PlannedCreate]]:
    """How a curation request touched the windows: ``(action_indices,
    direct, cascaded, creates)``. ``action_indices`` maps each window an
    action targeted to the plan positions of those actions (an implied
    create's two halves both name the window built); ``direct`` is those
    windows; ``cascaded`` is every other window whose fit the write changed
    (a dependent the cascade refit, bit for bit different from what was
    displayed); ``creates`` is every create the request ran, by window id
    (a widening named twice once, as it ended up)."""
    action_indices: Dict[int, List[int]] = {}
    creates: Dict[int, _PlannedCreate] = {}
    if req.display is None:
        return action_indices, set(), set(), creates  # bare accepts fit nothing
    for ra in req.resolved:
        if ra.target_wid is not None:
            action_indices.setdefault(int(ra.target_wid), []).append(ra.original_index)
        if ra.create is not None:
            creates[int(ra.create.proposal.window.window_id)] = ra.create
    direct = set(action_indices)
    before = _fit_by_window(req.display.changeset.spectrum_fit)
    changed = {
        wid
        for wid, wf in _fit_by_window(req.curated.spectrum_fit).items()
        if wid not in before or _fit_signature(before[wid]) != _fit_signature(wf)
    }
    return action_indices, direct, changed - direct, creates


def _coverage_from_fit(fit: SpectrumFit) -> List[FitWindowCoverage]:
    """The live-window index :func:`_load_curation_window_index` reads from
    disk, built from an in-memory fit instead -- same fields, same
    ``window_id``-ascending order."""
    return sorted(
        (
            FitWindowCoverage(
                int(wf.window_id),
                (
                    None
                    if wf.window is None
                    else (
                        float(wf.window.freq_range[0]),
                        float(wf.window.freq_range[1]),
                    )
                ),
                {p.peak_uid for p in wf.fitted_peaks if p.peak_uid is not None},
            )
            for wf in fit.window_fits
            if wf.window_id is not None
        ),
        key=lambda c: c.window_id,
    )


def _peak_index_from_fit(
    fit: SpectrumFit, plan: Sequence[PlannedAction]
) -> Tuple[Dict[int, List[float]], Dict[int, Set[int]]]:
    """:func:`_fitted_peak_index` over an in-memory fit: the uid half only
    when *plan* carries a ``"uid:N"`` target, as the on-disk reader does."""
    freqs = {
        int(wf.window_id): [float(p.frequency_mhz) for p in wf.fitted_peaks]
        for wf in fit.window_fits
        if wf.window_id is not None
    }
    has_uid_target = any(
        action.kind == "edit"
        and any(isinstance(t, PeakUidToken) for t in action.remove)
        for action in plan
    )
    if not has_uid_target:
        return freqs, {}
    uids = {
        int(wf.window_id): {
            p.peak_uid for p in wf.fitted_peaks if p.peak_uid is not None
        }
        for wf in fit.window_fits
        if wf.window_id is not None
    }
    return freqs, uids


def _planned_ranges_from_plan(plan: "WindowPlan") -> Dict[int, Tuple[float, float]]:
    """:func:`_planned_window_ranges` over an in-memory effective plan."""
    return {
        int(w.window_id): (
            min(float(w.freq_range[0]), float(w.freq_range[1])),
            max(float(w.freq_range[0]), float(w.freq_range[1])),
        )
        for w in plan.windows
    }


def _resolve_ops_against(
    path: str,
    ops: ParsedCurationFile,
    *,
    frame: Frame,
    stamp: Optional[_CalibrationStamp],
    raw_targets: Optional[FrozenSet[float]],
    ctx: _BatchCtx,
    snap_tol_mhz: float,
) -> Tuple[List[PlannedAction], List[str]]:
    """A parsed, frame-resolved curation file's plan and advisories, resolved
    against the curated state *ctx* displays -- its in-memory fit and
    effective window plan -- rather than against the file: a log-prefix
    apply's rows were written against the state the prefix describes.

    The same three passes :func:`apply_curation_impl` runs against the file
    on an ordinary apply (window derivation + coalescing, the ambiguity
    advisories, the frame-mismatch advisory), fed from memory.
    """
    fit = ctx.changeset.spectrum_fit
    plan = _resolve_curation_ops(
        ops, path, frame=frame, stamp=stamp, coverage=_coverage_from_fit(fit)
    )
    index = _peak_index_from_fit(fit, plan)
    warnings = _curation_ambiguity_warnings(
        path,
        plan,
        snap_tol_mhz=snap_tol_mhz,
        index=index,
        planned_ranges=_planned_ranges_from_plan(
            _overlay_created_windows(
                ctx.shared.base_plan, ctx.changeset.created_windows
            )
        ),
    )
    warnings += _frame_mismatch_warnings(
        path, plan, raw_targets=raw_targets, stamp=stamp, fitted_freqs=index[0]
    )
    return plan, warnings


def _applied_curation(
    plan: List[PlannedAction], warnings: List[str], req: _CuratedRequest
) -> "CurationApplyResult":
    """A live apply's result, read off the request it curated: the windows
    its actions targeted and the ones the write otherwise changed
    (:func:`_request_outcome`), before as displayed and after as persisted,
    and the structure its creates installed."""
    action_indices, direct, cascaded, creates = _request_outcome(req)
    return CurationApplyResult(
        plan=plan,
        warnings=warnings,
        applied=len(plan),
        dry_run=False,
        created_windows=[_planned_window_result(c) for _, c in sorted(creates.items())],
        windows=_applied_windows_block(
            before=(
                None if req.display is None else req.display.changeset.spectrum_fit
            ),
            after=req.curated.spectrum_fit,
            action_indices=action_indices,
            direct_wids=direct,
            cascaded_wids=cascaded,
        ),
    )


def _planned_action_has_freq(action: PlannedAction) -> bool:
    """Whether *action* carries any caller-supplied frequency at all -- a bare
    ``accept`` (``candidate is None``) does not, so it needs no frame."""
    return bool(
        action.add
        or action.remove
        or action.peaks
        or action.peak is not None
        or action.candidate is not None
        or action.anchor is not None
    )


def _planned_action_to_raw(
    action: PlannedAction, *, frame: Frame, stamp: Optional[_CalibrationStamp]
) -> PlannedAction:
    """Convert every frequency on *action* to the raw frame (the batch door's
    counterpart to the per-verb conversions above)."""

    def conv(f: float) -> float:
        return _frame_to_raw(f, frame=frame, stamp=stamp)

    def conv_remove(t: Union[float, PeakUidToken]) -> Union[float, PeakUidToken]:
        # A PeakUidToken is frame-independent (core.curation.PeakUidToken):
        # it carries no frequency of its own to convert, and is resolved to
        # one downstream, against the raw-frame fitted peaks directly.
        return t if isinstance(t, PeakUidToken) else conv(t)

    return replace(
        action,
        add=[conv(f) for f in action.add],
        remove=[conv_remove(t) for t in action.remove],
        peaks=[conv(f) for f in action.peaks],
        peak=None if action.peak is None else conv(action.peak),
        candidate=None if action.candidate is None else conv(action.candidate),
        anchor=None if action.anchor is None else conv(action.anchor),
    )


def _resolve_created_window_structure(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    snap_tol_mhz: float,
    shared: Optional[_SharedFitCtx] = None,
) -> List[PlannedWindowResult]:
    """Resolve every window a plan would install or grow -- WITHOUT fitting
    any of them. ``review apply --dry-run``'s structural report.

    A window's extent comes from the planner, not from plan resolution, so a
    dry run cannot read it off the resolved plan: it only knows a create is
    implied, not where it lands. Before this, the CLI answered that by running
    the whole in-memory preview a second time, which made a dry run *with* a
    create cost what a preview costs and blurred the cheap first rung of the
    dry-run -> preview -> apply ladder. This pays for the proposal alone:
    :func:`_plan_batch_create` per create, and no ``refit_window_core`` call
    at all.

    Returns ascending by window id, and ``[]`` for a plan with no create at
    all -- the common case, which never opens the engine and so pays nothing.

    Creates are walked in :func:`_canonicalize_batch_plan`'s order (creates
    first, in plan order), and each proposal is folded into the overlay the
    next one plans against, exactly as :func:`_install_planned_create` folds
    its own: a second create in the same gap must see the first. Planning reads
    no fit (:func:`_plan_create`), so nothing here is fit. That state is local
    to this call and is discarded on return; no file is touched.

    Refusals are the apply's own, raised here with the apply's own per-action
    attribution: if this returns, every create in the plan resolves. A plan
    with a create refits (the create's window is fit), so this is
    epoch-gated like the apply and the preview it stands in for: an
    epoch-mismatched file refuses here rather than showing structure whose
    apply is guaranteed to refuse. The undo baseline is not taken.
    """
    creates = [(i, a) for i, a in _canonicalize_batch_plan(plan) if a.kind == "create"]
    if not creates:
        return []

    require_splice_compatible_environment(path)
    ctx = _build_batch_ctx(path, snap_tol_mhz=snap_tol_mhz, shared=shared)
    structures: List[PlannedWindowResult] = []
    for original_index, action in creates:
        proposal: Optional["Stage6WindowProposal"] = None
        try:
            if action.anchor is None:
                raise ValueError("create action requires an anchor frequency")
            # W3.1: only a FRESH implied create can coalesce -- see the
            # matching comment in _resolve_action. A coalesced create
            # installs nothing, so it contributes no PlannedWindowResult: the
            # plan installs one window, not two, and this structural report
            # must say so.
            fresh_implied = action.implied_create and _is_implied_window_id(
                action.window_id
            )
            coalesce_target = (
                _batch_implied_create_target(ctx, action.anchor)
                if fresh_implied
                else None
            )
            if coalesce_target is None:
                proposal = _plan_batch_create(
                    ctx,
                    action.anchor,
                    replay_window_id=(
                        None
                        if action.window_id == _NEW_WINDOW_SENTINEL
                        or _is_implied_window_id(action.window_id)
                        else action.window_id
                    ),
                )
        except (ValueError, KeyError) as exc:
            _raise_curation_failure(original_index, action, exc)

        if proposal is None:
            continue

        fit_win = proposal.window
        new_wid = int(fit_win.window_id)
        (lo, hi), n_points = _created_window_extent(fit_win)
        structures.append(
            PlannedWindowResult(
                window_id=new_wid,
                anchor_mhz=float(action.anchor),
                mode=proposal.mode,
                freq_range=(lo, hi),
                n_points=n_points,
                n_contributors=len(fit_win.fixed_contributors),
                depends_on=[int(d) for d in proposal.depends_on],
            )
        )

        ctx.changeset.created_windows = [
            w for w in ctx.changeset.created_windows if int(w.window_id) != new_wid
        ] + [fit_win]
        if proposal.mode == "created":
            ctx.changeset.window_id_high_water = max(
                ctx.changeset.window_id_high_water, new_wid
            )

    structures.sort(key=lambda pw: pw.window_id)
    return structures


@requires_pipeline_file()
@_review_operation(
    "review apply",
    lambda r, a: review_apply_summary(r, a["dry_run"]),
    wrote=lambda r, a: not a["dry_run"],
)
def apply_curation_impl(
    file_path: Union[Path, str],
    curation_path: Optional[Union[Path, str]] = None,
    *,
    actions: Optional[Sequence[Union[CurationAction, Mapping[str, Any]]]] = None,
    dry_run: bool = False,
    frame: Optional[Frame] = None,
    log_prefix: Optional[int] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> CurationApplyResult:
    """Apply a curation file to *file_path*, delegating to the edit impls.

    Parses the curation CSV, derives any omitted add/remove window id by
    live-window coverage (W2 -- see :func:`_resolve_curation_window_ids`;
    ``accept``/``create`` still require the window named), coalesces it into
    a delegated action plan (one refit per window for runs of add/remove;
    merge/split/accept stand alone), converts every frequency on the resolved
    plan to raw (before ambiguity resolution, before any snapping), and --
    unless ``dry_run`` -- applies the whole plan as one request
    (:func:`_curate_request`): every action is resolved into decision rows
    against the displayed fit in a canonical cross-window order (creates
    first, then ascending window id, independent of the file's row order;
    :func:`_canonicalize_batch_plan`), before anything is fit; the rows are
    appended to the log, and the log is replayed from the automatic fit,
    cascaded once and persisted once (:func:`_curate`).
    ``dry_run`` returns the resolved (raw-converted) plan and
    frequency-resolution warnings without mutating the file. It also resolves
    the structure the plan would install -- see
    :func:`_resolve_created_window_structure`, which costs the window
    proposal alone rather than a full in-memory preview, and which refuses
    what the apply would refuse of a CREATE (an anchor outside the analysis
    band, a window that cannot be placed): if a dry run returns, every create
    in the plan resolves. It still does not run the edits, so an edit-side
    failure is a warning at most here and an error at apply --
    :func:`review_preview_impl` is the rung that runs them.

    ``frame`` applies uniformly to every frequency the curation file carries
    -- there is no per-row frame column. The file's own optional header (A3,
    ``# frame: ...`` / ``# epsilon: ...``, see :func:`parse_curation_file`
    and :func:`_resolve_curation_frame`) takes precedence when it disagrees
    with neither, or wins outright when ``frame`` is omitted; when both are
    given and disagree, the call is refused. Omitting both is an error on a
    ``self_calibrated`` file when the plan carries any frequency at all. A
    calibrated header whose stamped epsilon no longer matches the file's
    current one is refused -- never silently resolved with either value.

    Raises ``ValueError`` on a malformed curation file, an omitted-window
    add/remove target no live window covers or whose ``"uid:N"`` matches no
    fitted peak (raised unconditionally -- including on ``dry_run``, since
    there is no window id to put in the plan at all), a ``create`` whose
    window cannot be placed (likewise raised on ``dry_run``), or when an
    action fails to resolve (e.g. a ``remove`` frequency matches no fitted
    peak), tagged with the offending action; ``curation_conflict`` when the
    file cannot be curated (``predates_peak_identity`` /
    ``predates_replay_engine``, :func:`_require_engine_file`) or when a row
    cannot be resolved or replayed (``line_already_fitted``,
    ``target_outside_window``, ``ambiguous_peak``, ``replay_conflict``,
    ``fit_plan_unavailable``, ...); ``file_incompatible`` when a newer engine
    curated the file. A failure leaves the file untouched (nothing is
    persisted until the whole request has succeeded).

    ``log_prefix`` applies the file as if the decision log ended after its
    first ``log_prefix`` decisions (a count: ``0`` keeps none, the log's
    length keeps all and is the ordinary apply). The later decisions are
    dropped and this file's rows appended to the kept ones, and that log is
    curated -- one replay, one cascade, one persist -- so the outcome is
    that of :func:`review_undo_impl` on the dropped decisions followed by an
    ordinary apply. See :func:`_apply_curation_at_prefix` for what resolves
    against what. The kept decisions' refusals come before any fit; this
    file's own resolve against the state the kept prefix describes, which is
    computed in memory first, so they come after that computation, and the
    file is untouched either way. Refused with
    ``dry_run`` when it would shorten the log: the
    advisories and structure a dry run reports are read against the file,
    which at no point holds the state a shortened prefix describes.

    ``actions`` is the same batch as data: a sequence of
    :class:`~ftmwpipeline.core.curation.CurationAction` (or their dicts),
    given instead of ``curation_path`` -- exactly one of the two, else
    ``bad_setting`` (``path`` ``"actions"``). Each action resolves its own
    frame (its ``frame``, else this call's ``frame``, else the default rule)
    and is converted to raw before anything resolves; from there the batch
    is the file's (see :func:`_actions_to_ops`), so the result, decision log
    and file are those of the equivalent curation file.

    ``_shared`` is internal -- see :func:`refit_window_impl`.
    """
    path = str(file_path)
    _require_engine_file(path)
    source = curation_source(curation_path, actions)
    if log_prefix is not None:
        log = load_stage6_review_from_file(path).decision_log
        if log_prefix < 0 or log_prefix > len(log):
            raise BadSettingError(
                "log_prefix",
                f"an integer between 0 and the decision log's length ({len(log)})",
                log_prefix,
                message=f"log_prefix must be between 0 and the decision log's length "
                f"({len(log)}), got {log_prefix}",
            )
        if log_prefix < len(log):
            if dry_run:
                raise BadSettingError(
                    "dry_run",
                    "False when log_prefix is shorter than the decision log",
                    dry_run,
                    message="dry_run cannot be combined with a log_prefix shorter "
                    "than the decision log: the file never holds the state the "
                    "prefix describes, so the preview would resolve against "
                    "the wrong fit. Undo the later decisions first, then dry-run.",
                )
            return _apply_curation_at_prefix(
                path,
                source,
                frame=frame,
                log=log,
                keep=log_prefix,
                shared=_shared,
            )
    plan, raw_targets, stamp = _resolve_curation_call(path, source, frame)

    snap_tol = refit_snap_tol_mhz_impl(path)
    # One read of the fitted peak columns for both advisory passes: the
    # ambiguity pass needs it unconditionally, and the frame diagnostic used
    # to rebuild the identical map moments later in the same call.
    peak_index = _fitted_peak_index(path, plan)
    warnings = _curation_ambiguity_warnings(
        path, plan, snap_tol_mhz=snap_tol, index=peak_index
    )
    warnings += _frame_mismatch_warnings(
        path,
        plan,
        raw_targets=raw_targets,
        stamp=stamp,
        fitted_freqs=peak_index[0],
    )

    # No epoch pre-check here: the write gates itself (_curate), and only
    # when it refits -- a plan of bare accepts refits nothing.
    if dry_run:
        # The apply refuses an unknown window id (not_found); a dry run that
        # accepted it would promise an apply that cannot happen.
        known, where = _known_window_ids(path)
        _require_known_plan_windows(known, plan, where)
        return CurationApplyResult(
            plan=plan,
            warnings=warnings,
            applied=0,
            dry_run=True,
            created_windows=_resolve_created_window_structure(
                path, plan, snap_tol_mhz=snap_tol, shared=_shared
            ),
        )
    if not plan:
        return CurationApplyResult(
            plan=plan, warnings=warnings, applied=0, dry_run=False
        )

    req = _curate_request(
        path,
        plan,
        snap_tol_mhz=snap_tol,
        shared=_shared,
        persist=True,
        one_action=False,
    )
    return _applied_curation(plan, warnings, req)


def _apply_curation_at_prefix(
    path: str,
    source: CurationSource,
    *,
    frame: Optional[Frame],
    log: Sequence[DecisionLogEntry],
    keep: int,
    shared: Optional["_SharedFitCtx"],
) -> CurationApplyResult:
    """The shortened-log path of :func:`apply_curation_impl`: drop the
    decisions after the first ``keep``, append the curation file's rows to
    the kept ones, and curate that log.

    The dropped set is a suffix, so unlike an arbitrary undo it can never
    orphan a kept decision (nothing earlier in the log acts on a window or a
    peak a later decision installed).

    The curation file is parsed and its frame resolved first (both
    state-independent, so a malformed file refuses with the file untouched).
    Its window derivation, coalescing, advisories and resolution then read
    the state the kept prefix describes: the prefix is curated in memory
    (:func:`_curate`, not persisted), and the file's rows resolve against
    that fit and overlay (:func:`_resolve_ops_against`) -- the file on disk
    holds the dropped decisions too, which is not the state the caller's rows
    were written against. The kept rows are kept verbatim, serials included;
    the new rows take serials above every one the lineage has recorded, and
    a fresh create mints above every window id it has minted. The kept
    rows' refusals come before any fit; the file's own come after the prefix
    is computed, and the file is untouched either way. A log with no
    automatic-fit baseline is corrupt (:func:`_open_batch`), as it is for
    every write.
    """
    kept = list(log[:keep])
    ops, resolved_frame, stamp, raw_targets = _parse_curation_call(path, source, frame)
    snap_tol = refit_snap_tol_mhz_impl(path)
    _check_cancel()
    baseline_taken = _open_batch(path)
    prior = load_stage6_review_from_file(path)
    params = _recorded_review_params(prior)
    if shared is None:
        shared = _build_shared_fit_ctx(path)
    prefix = _curate(
        path,
        prior,
        kept,
        params,
        recorded=len(kept),
        shared=shared,
        baseline_taken=False,
        snap_tol_mhz=snap_tol,
    )
    display = _BatchCtx(
        shared=shared,
        changeset=_display_changeset(
            shared,
            prefix.spectrum_fit,
            prefix.review,
            snap_tol_mhz=snap_tol,
            high_water_from=prior,
        ),
    )
    plan, warnings = _resolve_ops_against(
        path,
        ops,
        frame=resolved_frame,
        stamp=stamp,
        raw_targets=raw_targets,
        ctx=display,
        snap_tol_mhz=snap_tol,
    )
    _require_known_plan_windows(
        _batch_known_window_ids(
            display, plan, set(_fit_by_window(prefix.spectrum_fit))
        ),
        plan,
        "Stage 5 fit",
        _batch_plan_window_ids(display, plan),
    )
    resolved = _resolve_request(display, plan, attribute=True, snap_tol_mhz=snap_tol)
    curated = _curate(
        path,
        prior,
        kept + _request_rows(display, resolved, one_action=False),
        params,
        recorded=len(kept),
        shared=shared,
        baseline_taken=baseline_taken,
        snap_tol_mhz=snap_tol,
    )
    _finish_batch(curated, path)
    return _applied_curation(
        plan,
        warnings,
        _CuratedRequest(display=display, resolved=resolved, curated=curated),
    )


# ---------------------------------------------------------------------------
# review_preview_impl: run a curation plan to completion in memory and
# report the fitted outcome, without persisting anything (C1-C6).
#
# The write path itself (_curate_request) with persist=False: the request is
# resolved and the log with its rows curated exactly as an apply would, the
# undo baseline is not taken, and _finish_batch -- the engine's only writer
# of /stage5_fitting -- is never called.
# ---------------------------------------------------------------------------


@dataclass
class PreviewWindowResult:
    """One window's outcome from :func:`review_preview_impl`, read off the
    in-memory fit *after* the batch's one combined cascade -- never off an
    applier's own (potentially superseded) ``RefitWindowResult``. See
    ``scratch/bq-correspondence/reply-preview-execute.md`` section 2: the
    appliers return a result before the cascade runs, and
    ``_cascade_closure`` can supersede it, so per-action alignment was
    deliberately rejected in favor of this shape.

    Attributes
    ----------
    window_id : int
        The window this entry reports on.
    origin : str
        ``"direct"`` -- some action in the plan targeted this window (edit /
        merge / split / accept-with-candidate / create); ``"cascaded"`` --
        no action of the plan names it, but the plan changes its fit: the
        cascade refits it as a downstream dependent of an edited window. A
        window that is both directly edited *and* downstream of an edit is
        ``"direct"``: it has an originating action, even though the cascade
        re-fits it again to pick up the refreshed background.
    action_indices : list of int
        0-based indices into ``ReviewPreviewResult.plan`` of every action
        that directly targeted this window. Empty for a purely-cascaded
        window.
    n_peaks_before, n_peaks_after : int
        Peak count in this window as displayed before the batch (the
        persisted fit) / in the curated fit after it, post-cascade.
    chi2r_before, chi2r_after : float or Absent
        Reduced chi-squared before the batch / after it, on the same terms.
        ``Absent.NOT_RUN`` when that side carries no fit to compute one
        against -- most obviously ``chi2r_before`` on a window the batch
        itself *creates*, which has no "before" at all. ``Absent.UNDEFINED``
        when the fit exists but its value is not finite (no degrees of
        freedom, or a fit that did not converge).

        Absent rather than ``0.0`` because a reduced chi-squared of exactly
        zero is a value a genuine fit essentially never produces, so a
        fabricated one is indistinguishable from an extraordinary one: a
        consumer rendering "before -> after" would show ``0.00 -> 1.4`` and read
        it as a perfect fit that got worse. The same absence-vs-plausible-number
        distinction :class:`~ftmwpipeline.core.calibration.CalibrationStamp`
        makes for ``probe_freq_mhz``. ``n_peaks_before == 0`` is a *tell* for
        this case but not a contract -- it is a default riding alongside, and
        a window can legitimately be emptied to zero peaks by an edit.
    peaks : list of FinalPeak
        This window's rows from the would-be final-products table
        (:func:`_curated_review`) -- calibrated frequencies and the
        three-term sigma budget, identical in shape and value to what a
        subsequent ``apply`` of the same plan would persist. Not the
        raw / stat-only ``RefitWindowResult.fitted_peaks``.
    created_window_mode : str or Absent
        W4, BlackQuill's acceptance condition for implicit window creation
        (``scratch/intent-driven-windowing-plan.md``, W4): ``"created"`` or
        ``"widened"`` when some action in this batch -- an implied create
        (an uncovered ``add``) or an explicit ``create`` -- installed or grew
        this window; ``Absent.NOT_RUN`` for a window this batch only edited,
        merged, split, accepted, or cascaded into, which is the common case.
        A UI renders "this add creates a window at A-B MHz" straight off this
        entry, without re-deriving anything -- see
        ``created_window_freq_range`` below. Absent here, not a fabricated
        ``"created"``/``"widened"``, for any window the batch did not create
        or widen -- the same absence-vs-plausible-value discipline
        ``chi2r_before`` documents.
    created_window_freq_range : tuple of float or Absent
        The installed (or widened) window's ``(min_mhz, max_mhz)`` extent,
        raw frame -- :attr:`CreateWindowResult.freq_range`, unchanged.
        ``Absent.NOT_RUN`` iff ``created_window_mode`` is.
    created_window_n_points : int or Absent
        Grid points the window covers. ``Absent.NOT_RUN`` iff
        ``created_window_mode`` is.
    created_window_n_contributors : int or Absent
        Frozen leakage contributors attached to the window.
        ``Absent.NOT_RUN`` iff ``created_window_mode`` is.
    created_window_depends_on : list of int or Absent
        Window ids the window reads frozen leakage from. ``Absent.NOT_RUN``
        iff ``created_window_mode`` is (``[]`` is a legitimate value -- a
        created window with no dependencies -- and distinct from that).
    converged : bool or Absent
        Whether this window's fit after the batch's one combined cascade
        converged (:attr:`FittingResult.success`), read off the same
        post-cascade in-memory fit ``chi2r_after`` and ``peaks`` are -- never
        off an applier's own ``RefitWindowResult``, which the cascade can
        supersede. ``False`` means the solver bailed: the window kept its
        seeds verbatim with an infinite chi-squared, so ``chi2r_after`` is
        wild and this window's ``peaks`` are seeds rather than measurements.
        ``Absent.NOT_RUN`` when the window carries no fit on the after side
        at all -- the same absence-vs-plausible-value discipline
        ``chi2r_after`` documents, and ``NOT_RUN`` for exactly the same
        windows. ``Absent.UNDEFINED`` for a window left with no peaks: no
        solver ran, so there is no convergence outcome.
    """

    window_id: int
    origin: str
    action_indices: List[int] = field(default_factory=list)
    n_peaks_before: int = 0
    n_peaks_after: int = 0
    chi2r_before: Union[float, Absent] = Absent.NOT_RUN
    chi2r_after: Union[float, Absent] = Absent.NOT_RUN
    peaks: List["FinalPeak"] = field(default_factory=list)
    created_window_mode: Union[str, Absent] = Absent.NOT_RUN
    created_window_freq_range: Union[Tuple[float, float], Absent] = Absent.NOT_RUN
    created_window_n_points: Union[int, Absent] = Absent.NOT_RUN
    created_window_n_contributors: Union[int, Absent] = Absent.NOT_RUN
    created_window_depends_on: Union[List[int], Absent] = Absent.NOT_RUN
    converged: Union[bool, Absent] = Absent.NOT_RUN


@dataclass
class ReviewPreviewResult:
    """Outcome of :func:`review_preview_impl`: a curation plan run to
    completion in memory, never persisted.

    Attributes
    ----------
    windows : dict of int to PreviewWindowResult
        Keyed by window id, read *after* the batch's one combined cascade --
        not per-action (see :class:`PreviewWindowResult`). Empty for a plan
        that touches no fit (e.g. entirely bare ``accept`` rows).
    plan : list of PlannedAction
        The resolved, frame-converted, coalesced action sequence -- the same
        shape ``apply_curation_impl`` would execute. ``action_indices`` on
        each :class:`PreviewWindowResult` index into this list.
    warnings : list of str
        Advisories that do not block the preview -- currently just the A5
        frame-mismatch diagnostic (:func:`_frame_mismatch_warnings`). Empty
        for a plan that touches no fit, since the diagnostic needs matched
        candidates to compare.
    created_windows : list of PlannedWindowResult
        Every window this preview's batch installed or grew, ascending by
        window id -- the same list, in the same shape, that
        :class:`CurationApplyResult` publishes for a dry run and for a live
        apply, so the three rungs of the ladder report one structure rather
        than three.

        Not redundant with :attr:`PreviewWindowResult.created_window_mode`
        and its siblings, which carry the same structure keyed by the window
        it landed in: those cannot carry the **anchor**, and with W3.1
        coalescing one created window can hold two ``add`` rows, so extent
        containment marks both while the anchor marks the one that implied
        the create -- the same call the decision log makes by putting the
        ``created_window`` evidence on the first add's entry.

        Empty for a plan that installs nothing.
    """

    windows: Dict[int, PreviewWindowResult] = field(default_factory=dict)
    plan: List["PlannedAction"] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    created_windows: List[PlannedWindowResult] = field(default_factory=list)


@dataclass
class _PreviewRun:
    """Internal: everything one preview computed, including the piece
    :func:`review_preview_impl` throws away but a :class:`ReviewSession`
    needs to stage for a possible immediately-following apply (D4): the
    curated state, computed and never persisted. ``curated`` is ``None`` only
    for the bare-accept-only short circuit, which fits nothing and so has
    nothing to stage.
    """

    result: ReviewPreviewResult
    curated: Optional[_Curated] = None


def _run_review_preview(
    file_path: Union[Path, str],
    source: CurationSource,
    *,
    frame: Optional[Frame] = None,
    shared: Optional[_SharedFitCtx] = None,
) -> _PreviewRun:
    """Run a curation file's resolved plan to completion in memory and report
    the fitted outcome -- final-product numbers, post-cascade -- without
    writing anything to *file_path*. The body of :func:`review_preview_impl`,
    plus the internal state (D4) a :class:`ReviewSession` needs to persist a
    following apply without recomputing.

    Shares :func:`apply_curation_impl`'s parse / derive-window / coalesce /
    frame-convert prologue exactly (:func:`_resolve_curation_call`),
    including the curation file's optional frame header (A3), W2's
    omitted-window derivation, and the A5 frame-mismatch advisory, then runs
    the write path itself (:func:`_curate_request`) without persisting: the
    request resolves into the same rows, and the log with them appended is
    curated exactly as the apply would curate it, so the preview returns the
    state persisting it would write. The undo baseline snapshot is never
    taken and ``_finish_batch`` is never called: a preview writes nothing,
    byte for byte.

    Refused, first, exactly like a real apply on a file this engine cannot
    curate (:func:`_require_engine_file`: ``predates_peak_identity`` /
    ``predates_replay_engine``, or ``file_incompatible``), and epoch-gated
    exactly like one (the same :func:`_curate`, which gates a write that
    refits): either way it would show numbers whose apply is guaranteed to
    refuse.

    A plan consisting entirely of bare ``accept`` rows (no frequency, no fit
    touched) short-circuits: it does no fits, is not epoch-gated (a bare
    accept in a live apply is not gated either -- see
    :func:`review_accept_impl`), and returns an empty ``windows`` dict.

    Raises the same per-action attributed ``ValueError`` a live apply raises
    (tagged with the 1-based action index and its description), on the same
    failures, since it shares the same resolution.

    ``shared`` lets a caller with an already-built :class:`_SharedFitCtx`
    reuse it (``ReviewSession``, D3); a fresh one is built when omitted.
    """
    path = str(file_path)
    _require_engine_file(path)
    snap_tol = refit_snap_tol_mhz_impl(path)
    plan, raw_targets, stamp = _resolve_curation_call(path, source, frame)

    warnings = _frame_mismatch_warnings(
        path, plan, raw_targets=raw_targets, stamp=stamp
    )

    if not _plan_needs_fit(plan):
        # C5: bare-accept-only (or empty) plan -- no fits, no gate, nothing to
        # report -- and refused, as an apply is, on a file with no complete
        # fit.
        _require_complete_fit(path, "review preview")
        known, where = _known_window_ids(path)
        _require_known_plan_windows(known, plan, where)
        return _PreviewRun(
            result=ReviewPreviewResult(windows={}, plan=plan, warnings=warnings)
        )

    req = _curate_request(
        path,
        plan,
        snap_tol_mhz=snap_tol,
        shared=shared,
        persist=False,
        one_action=False,
    )
    assert req.display is not None
    action_indices, direct_wids, cascaded_wids, creates = _request_outcome(req)
    review = req.curated.review
    peaks_by_window: Dict[int, List["FinalPeak"]] = {}
    if review.final_products is not None:
        for peak in review.final_products.peaks:
            if not isinstance(peak.window_id, Absent):
                peaks_by_window.setdefault(int(peak.window_id), []).append(peak)

    before_by_wid = _fit_by_window(req.display.changeset.spectrum_fit)
    after_by_wid = _fit_by_window(req.curated.spectrum_fit)
    windows: Dict[int, PreviewWindowResult] = {}
    for wid in sorted(direct_wids | cascaded_wids):
        # A window absent from one side has no fit on that side -- a window
        # this plan creates has no "before" at all. Report the absence rather
        # than a plausible-looking 0.0 (see PreviewWindowResult).
        before = before_by_wid.get(wid)
        after = after_by_wid.get(wid)
        # W4: the structural consequence, if this plan created or widened
        # `wid` -- absent (NOT_RUN) for a window it only edited/merged/split/
        # accepted/cascaded into.
        created = creates.get(wid)
        created_fact = None if created is None else _planned_window_result(created)
        windows[wid] = PreviewWindowResult(
            window_id=wid,
            origin="direct" if wid in direct_wids else "cascaded",
            action_indices=sorted(action_indices.get(wid, [])),
            n_peaks_before=0 if before is None else len(before.fitted_peaks),
            n_peaks_after=0 if after is None else len(after.fitted_peaks),
            chi2r_before=(
                Absent.NOT_RUN
                if before is None
                else float_or_absent(float(before.reduced_chi2))
            ),
            chi2r_after=(
                Absent.NOT_RUN
                if after is None
                else float_or_absent(float(after.reduced_chi2))
            ),
            converged=Absent.NOT_RUN if after is None else _converged_or_absent(after),
            peaks=peaks_by_window.get(wid, []),
            created_window_mode=(
                Absent.NOT_RUN if created_fact is None else created_fact.mode
            ),
            created_window_freq_range=(
                Absent.NOT_RUN if created_fact is None else created_fact.freq_range
            ),
            created_window_n_points=(
                Absent.NOT_RUN if created_fact is None else created_fact.n_points
            ),
            created_window_n_contributors=(
                Absent.NOT_RUN if created_fact is None else created_fact.n_contributors
            ),
            created_window_depends_on=(
                Absent.NOT_RUN
                if created_fact is None
                else list(created_fact.depends_on)
            ),
        )

    result = ReviewPreviewResult(
        windows=windows,
        plan=plan,
        warnings=warnings,
        # Published on the result rather than kept internal: a session
        # staging this preview as an apply, and a caller reading it directly,
        # must see the SAME list -- deriving it twice is how two rungs of the
        # ladder come to disagree about what a plan installs.
        created_windows=[_planned_window_result(c) for _, c in sorted(creates.items())],
    )
    return _PreviewRun(result=result, curated=req.curated)


@requires_pipeline_file()
@_review_operation(
    "review preview",
    lambda r, a: review_preview_summary(r),
    wrote=lambda r, a: False,
)
def review_preview_impl(
    file_path: Union[Path, str],
    curation_path: Optional[Union[Path, str]] = None,
    *,
    actions: Optional[Sequence[Union[CurationAction, Mapping[str, Any]]]] = None,
    frame: Optional[Frame] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> ReviewPreviewResult:
    """Run a curation file's resolved plan to completion in memory and report
    the fitted outcome -- final-product numbers, post-cascade -- without
    writing anything to *file_path*. See :func:`_run_review_preview` for the
    full contract; this is the public entry point, which discards the
    internal staging state a :class:`ReviewSession` needs and a sessionless
    caller does not.

    ``actions`` is :func:`apply_curation_impl`'s: the batch as data, given
    instead of ``curation_path`` (exactly one of the two).
    """
    source = curation_source(curation_path, actions)
    return _run_review_preview(file_path, source, frame=frame).result


@requires_pipeline_file()
def review_log_impl(file_path: Union[Path, str]) -> List[DecisionLogEntry]:
    """Return the persisted Stage 6 decision log (read-only, execution order).

    The rows are returned as stored. On a file Stage 6 refuses to write
    (:func:`_refit_required_reason`) the rows of a pre-engine build carry no
    ``serial``, and a warning names the re-run that makes the file curatable.
    """
    path = str(file_path)
    review = load_stage6_review_from_file(path)
    reason = refit_required_impl(path)
    if reason is not None:
        logger.warning(
            "This file's Stage 6 curation cannot be changed (%s). %s",
            reason,
            refit_required_instruction(reason),
        )
    return list(review.decision_log)


@dataclass
class UndoResult:
    """Outcome of :func:`review_undo_impl`.

    Attributes
    ----------
    removed : list of DecisionLogEntry
        The decisions that were (or, in dry-run, would be) undone.
    surviving : list of DecisionLogEntry
        The decisions retained and replayed from the automatic baseline.
    plan : list of PlannedAction
        The resolved replay of the surviving decisions.
    applied : int
        Number of replay actions executed (``0`` for a dry run).
    dry_run : bool
        Whether the undo was previewed without mutating.
    geometry_changed_window_ids : list of int
        The windows whose geometry the undo changes, ascending: a surviving
        created window keeps its id, but its geometry is re-derived from the
        creates that survive, so undoing an earlier create can move it (the
        gap it was planned in, the contributors attached to it); a widened
        window can likewise change, or return to its base extent. A window
        the undo removes (an undone create) is not listed. Reported, not
        refused; a dry run reports the same.
    """

    removed: List[DecisionLogEntry]
    surviving: List[DecisionLogEntry]
    plan: List["PlannedAction"]
    applied: int
    dry_run: bool
    geometry_changed_window_ids: List[int] = field(default_factory=list)
    #: The stages the call invalidated (canonical names, ``rerun_order``):
    #: always ``()``, since Stage 6 invalidates no stage.
    invalidated: Tuple[str, ...] = field(default=(), compare=False)


def _same_decision_action(prev: DecisionLogEntry, entry: DecisionLogEntry) -> bool:
    """Whether *entry* was recorded by the same user action as *prev*, the
    entry immediately before it in a (possibly filtered) decision log: both
    carry the same :data:`ACTION_INDEX_EVIDENCE_KEY` (the serial of the
    action's first row). Every row this engine records carries it."""
    prev_ai = prev.evidence.get(ACTION_INDEX_EVIDENCE_KEY)
    entry_ai = entry.evidence.get(ACTION_INDEX_EVIDENCE_KEY)
    return (
        prev_ai is not None and entry_ai is not None and int(prev_ai) == int(entry_ai)
    )


def _decision_action_groups(
    entries: Sequence[DecisionLogEntry],
) -> List[List[DecisionLogEntry]]:
    """Partition *entries* (in log order) into the user actions that recorded
    them -- see :func:`_same_decision_action` for what one action is.

    One action's rows are always contiguous in the log (an action records
    all of them before the next one runs), so grouping compares each entry
    only with the one before it. *entries* may be a filtered log -- an
    undo's survivors, a log prefix -- and the rows of a partially dropped
    action that remain are still one group.
    """
    groups: List[List[DecisionLogEntry]] = []
    for entry in entries:
        if groups and _same_decision_action(groups[-1][-1], entry):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def _refit_units(rows: Sequence[DecisionLogEntry]) -> List[List[int]]:
    """The positions of *rows*'s add/remove/merge/split rows, partitioned into
    the refits a replay runs (:func:`_resolve_replay_rows`): a run of
    consecutive add/remove rows of one action on one window is one joint
    refit, every other peak row its own. A joint refit removes all of its
    targets before it births any of its seeds, so the symbolic passes take
    each unit's targets out before they place its births, as the recording
    did (:func:`_apply_rows_to_view`)."""
    units: List[List[int]] = []
    prev: Optional[DecisionLogEntry] = None
    for k, e in enumerate(rows):
        if e.kind not in _PEAK_ROW_KINDS:
            prev = None
            continue
        if (
            prev is not None
            and prev.kind in ("add", "remove")
            and e.kind in ("add", "remove")
            and int(prev.window_id) == int(e.window_id)
            and not _installs_window(prev)
            and not _installs_window(e)
            and _same_decision_action(prev, e)
        ):
            units[-1].append(k)
        else:
            units.append([k])
        prev = e
    return units


def _row_tuple(value: Union[Tuple[Any, ...], Absent]) -> Tuple[Any, ...]:
    """A row's peak-identity tuple, empty on a row that carries none."""
    return () if isinstance(value, Absent) else tuple(value)


def _row_peak_fields(
    path: str, entry: DecisionLogEntry
) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[int, ...]]:
    """*entry*'s ``(targets, seeds_mhz, born_uids)``. Every add, remove,
    merge or split row of an admitted file carries them (only the engine
    writes the log), so a row without them, or with seeds and born uids out
    of step, is a corrupt file."""
    if (
        isinstance(entry.targets, Absent)
        or isinstance(entry.seeds_mhz, Absent)
        or isinstance(entry.born_uids, Absent)
        or len(entry.seeds_mhz) != len(entry.born_uids)
    ):
        raise PipelineCorruptionError(
            Path(path),
            f"decision {_serial_id(entry)} ({entry.kind} on window "
            f"{entry.window_id}) carries no consistent peak identity "
            "(targets, seeds_mhz, born_uids)",
        )
    return (
        tuple(int(u) for u in entry.targets),
        tuple(float(f) for f in entry.seeds_mhz),
        tuple(int(u) for u in entry.born_uids),
    )


@dataclass
class _LogWalk:
    """What the symbolic pass over a decision log (:func:`_walk_log_rows`)
    learns without fitting: the created-window overlay the log's creates
    install, each row's window geometry in force at the row (after the row's
    own create, for a create row), and each create row's proposal, by row
    position."""

    overlay: List["FitWindow"]
    geometry: List[Optional["FitWindow"]]
    proposals: Dict[int, "Stage6WindowProposal"]


def _walk_log_rows(
    path: str,
    shared: _SharedFitCtx,
    rows: Sequence[DecisionLogEntry],
    *,
    min_new_window_id: int,
    creates: Optional["_CreateChain"] = None,
) -> _LogWalk:
    """The symbolic pass over a decision log replayed from the automatic
    fit: every refusal the replay has, raised before any fit.

    Replans the log's creates in log order, each pinned to its recorded id
    (:func:`_plan_create`: the analysis-band refusal, ``replay_conflict``,
    ``fit_plan_unavailable``; through *creates*, the engine's create-prefix
    cache, when given), and follows each window's uid set from the
    automatic fit through the rows: a window's uids at any log position are
    its automatic fit's, minus the targets and plus the born uids of the rows
    before it, since no refit, cascade or not, adds or drops a peak. A row
    whose target the window does not hold there, or whose born uid it already
    holds, is ``replay_diverged`` (the message names the row that removed the
    uid, when one did); a target the window holds twice is ``ambiguous_peak``;
    a seed off the geometry in force at the row is ``target_outside_window``;
    a row on, or a cascade reaching, a window whose fitted geometry the file
    does not hold is ``fit_plan_unavailable``. A row without peak identity is
    a corrupt file.
    """
    geom: Dict[int, "FitWindow"] = {
        int(w.window_id): w for w in shared.base_plan.windows
    }
    live: Dict[int, List[int]] = {
        int(wid): [int(p.peak_uid) for p in wf.fitted_peaks if p.peak_uid is not None]
        for wid, wf in shared.baseline_fits.items()
    }
    removed_by: Dict[Tuple[int, int], int] = {}
    born_by: Dict[Tuple[int, int], int] = {}
    overlay: List["FitWindow"] = []
    geometry: List[Optional["FitWindow"]] = []
    proposals: Dict[int, "Stage6WindowProposal"] = {}
    bounds: Dict[int, Tuple[float, float, float]] = {}
    # A joint refit's rows are checked together, at its first row: all of
    # its targets leave before any of its births is placed.
    unit_at = {u[0]: u for u in _refit_units(rows)}
    for k, e in enumerate(rows):
        w = int(e.window_id)
        if _installs_window(e):
            plan = _plan_create if creates is None else creates.plan
            proposal = plan(
                shared,
                overlay,
                float(e.frequency_mhz),
                replay_window_id=w,
                min_new_window_id=min_new_window_id,
            )
            overlay = [x for x in overlay if int(x.window_id) != w] + [proposal.window]
            geom[w] = proposal.window
            if proposal.mode == "created":
                live[w] = []
            proposals[k] = proposal
        geometry.append(geom.get(w))
        unit = unit_at.get(k)
        if unit is None:
            continue
        serial = _serial_id(e)
        fit_win = geom.get(w)
        uids = live.get(w)
        if fit_win is None or uids is None:
            raise CurationConflictError(
                "replay_diverged",
                [serial],
                message=f"cannot replay decision {serial} ({e.kind} on window "
                f"{w}): no fitted window {w} exists at its place in the log. "
                "Undo it together with the decisions after it.",
            )
        _refuse_unavailable_fit_plan(
            shared.unavailable_window_ids, [w], f"replaying decision {serial}"
        )
        fields = [(rows[j], _row_peak_fields(path, rows[j])) for j in unit]
        for r, (targets, _, _) in fields:
            serial = _serial_id(r)
            for t in targets:
                n = uids.count(t)
                if n == 1:
                    uids.remove(t)
                    removed_by[(w, t)] = serial
                    continue
                if n > 1:
                    raise CurationConflictError(
                        "ambiguous_peak",
                        [t],
                        message=f"cannot replay decision {serial} ({r.kind} on "
                        f"window {w}): the window holds more than one peak with "
                        f"peak_uid={t} there.",
                    )
                gone = removed_by.get((w, t))
                why = (
                    f"decision {gone} already removed it"
                    if gone is not None
                    else "no decision before it births it and the automatic fit "
                    "does not hold it"
                )
                raise CurationConflictError(
                    "replay_diverged",
                    [serial],
                    message=f"cannot replay decision {serial} ({r.kind} on window "
                    f"{w}): it removes peak_uid={t}, which the window does not "
                    f"hold there ({why}). Undo it together with the decisions "
                    "after it.",
                )
        for r, (_, seeds, born) in fields:
            serial = _serial_id(r)
            for f in seeds:
                _refuse_seed_outside_window(shared.fit_ctx, bounds, w, fit_win, f, f)
            for u in born:
                if u in uids:
                    holder = born_by.get((w, u))
                    held = (
                        f"decision {holder} births it"
                        if holder is not None
                        else "the automatic fit holds it"
                    )
                    raise CurationConflictError(
                        "replay_diverged",
                        [serial],
                        message=f"cannot replay decision {serial} ({r.kind} on "
                        f"window {w}): it births peak_uid={u}, which the window "
                        f"already holds there ({held}). Undo it together with "
                        "the decisions after it.",
                    )
                uids.append(u)
                born_by[(w, u)] = serial
                removed_by.pop((w, u), None)

    unavailable = shared.unavailable_window_ids
    if unavailable:
        dirty = _log_dirty_window_ids(rows)
        sources = _cascade_sources(shared.base_cascade_sources, overlay)
        created = [
            int(x.window_id)
            for x in overlay
            if int(x.window_id) not in shared.base_cascade_sources
        ]
        reached = [c for c in created if _cascade_ancestors(c, sources) & dirty]
        _refuse_unavailable_fit_plan(
            unavailable,
            (
                d
                for d in _cascade_closure_set(sorted(dirty), sources, reached)
                if d in live and d in geom
            ),
            "the dependency cascade",
        )
    return _LogWalk(overlay=overlay, geometry=geometry, proposals=proposals)


def _orphaned_peak_rows(
    log: Sequence[DecisionLogEntry], undo: Collection[int]
) -> List[int]:
    """The serials of the decisions an undo of *undo* would orphan by peak:
    every kept decision that targets a peak an undone decision births, and,
    transitively, every kept decision that targets a peak one of those
    births. Bound in log order: a target names the latest birth of its uid in
    its window before it -- before its refit, for a row of a joint refit
    (:func:`_refit_units`), whose own births come after all its removals."""
    births: Dict[Tuple[int, int], int] = {}
    deps: Dict[int, Set[int]] = {}
    for unit in _refit_units(log):
        rows = [log[k] for k in unit if not isinstance(log[k].targets, Absent)]
        for e in rows:
            w = int(e.window_id)
            keys = [(w, int(t)) for t in _row_tuple(e.targets)]
            deps[_serial_id(e)] = {births[key] for key in keys if key in births}
        for e in rows:
            for t in _row_tuple(e.targets):
                births.pop((int(e.window_id), int(t)), None)
        for e in rows:
            for u in _row_tuple(e.born_uids):
                births[(int(e.window_id), int(u))] = _serial_id(e)
    gone = {int(i) for i in undo}
    orphaned: List[int] = []
    for e in log:
        if e.kind not in _PEAK_ROW_KINDS:
            continue
        serial = _serial_id(e)
        if serial not in gone and deps.get(serial, set()) & gone:
            gone.add(serial)
            orphaned.append(serial)
    return sorted(orphaned)


def _replay_action_groups(
    entries: Sequence[DecisionLogEntry],
) -> List[Tuple[PlannedAction, List[DecisionLogEntry]]]:
    """*entries* (in log order) as the actions a replay runs, each with the
    rows it re-applies: one action group (:func:`_decision_action_groups`)
    at a time, a create row as a create (an implied create's add as the
    create pinned to its id followed by the add into it), a bare accept as an
    accept, and a group's add/remove/merge/split rows as one edit of their
    window. The action is a description of the replay (an edit's adds are its
    seeds, its removes the ``uid:N`` of its targets); the rows are what is
    applied."""
    out: List[Tuple[PlannedAction, List[DecisionLogEntry]]] = []
    for group in _decision_action_groups(entries):
        pending: List[DecisionLogEntry] = []

        def flush() -> None:
            if not pending:
                return
            out.append(
                (
                    PlannedAction(
                        kind="edit",
                        window_id=int(pending[0].window_id),
                        add=[
                            float(f) for e in pending for f in _row_tuple(e.seeds_mhz)
                        ],
                        remove=[
                            PeakUidToken(int(t))
                            for e in pending
                            for t in _row_tuple(e.targets)
                        ],
                    ),
                    list(pending),
                )
            )
            pending.clear()

        for e in group:
            w = int(e.window_id)
            if pending and int(pending[0].window_id) != w:
                flush()
            if _installs_window(e):
                flush()
                implied = e.kind == "add"
                out.append(
                    (
                        PlannedAction(
                            kind="create",
                            window_id=w,
                            anchor=float(e.frequency_mhz),
                            implied_create=implied,
                        ),
                        [e],
                    )
                )
                if implied:
                    out.append(
                        (
                            PlannedAction(
                                kind="edit",
                                window_id=w,
                                add=[float(f) for f in _row_tuple(e.seeds_mhz)],
                                implied_create=True,
                            ),
                            [e],
                        )
                    )
                continue
            if e.kind == "accept":
                flush()
                out.append((PlannedAction(kind="accept", window_id=w), [e]))
                continue
            if e.kind not in _PEAK_ROW_KINDS:
                raise ValueError(f"cannot replay decision of unknown kind {e.kind!r}")
            pending.append(e)
        flush()
    return out


def _replay_plan(entries: Sequence[DecisionLogEntry]) -> List[PlannedAction]:
    """The replay of *entries* as actions (:func:`_replay_action_groups`):
    what ``review undo`` reports it replays."""
    return [action for action, _ in _replay_action_groups(entries)]


def _resolve_replay_rows(
    ctx: _BatchCtx,
    path: str,
    rows: Sequence[DecisionLogEntry],
    *,
    walk: Optional[_LogWalk] = None,
) -> List[_ResolvedAction]:
    """The recorded rows *rows* as resolved actions, checked and planned
    before any fit (:func:`_walk_log_rows`; *walk* when the caller has
    already run it over *rows*): each create row's structure is installed in
    the batch, and each add/remove/merge/split row becomes a refit by
    identity on the geometry in force at it -- a merge or split row its own
    refit, consecutive add/remove rows of one action one joint refit, as they
    were applied when recorded. Nothing is re-resolved or re-inferred: the
    rows' targets, seeds and born uids are applied as recorded."""
    if walk is None:
        walk = _walk_log_rows(
            path,
            ctx.shared,
            rows,
            min_new_window_id=ctx.changeset.window_id_high_water + 1,
        )
    position = {id(e): k for k, e in enumerate(rows)}
    resolved: List[_ResolvedAction] = []
    implied: Optional[_PlannedCreate] = None
    for index, (action, group) in enumerate(_replay_action_groups(rows)):
        ra = _ResolvedAction(
            original_index=index, action=action, target_wid=action.window_id
        )
        if action.kind == "create":
            e = group[0]
            proposal = walk.proposals[position[id(e)]]
            _install_planned_create(ctx, proposal)
            ra.create = _PlannedCreate(
                proposal=proposal,
                anchor=float(e.frequency_mhz),
                record=e.kind == "create_window",
            )
            implied = ra.create if action.implied_create else None
        elif action.kind == "accept":
            ra.accept = True
            ra.target_wid = None
        else:
            for e in group:
                fit_win = walk.geometry[position[id(e)]]
                assert fit_win is not None  # _walk_log_rows refused otherwise
                targets, seeds, born = _row_peak_fields(path, e)
                row = _PendingRow(
                    kind=e.kind,
                    window_id=int(e.window_id),
                    frequency_mhz=float(e.frequency_mhz),
                    targets=targets,
                    seeds_mhz=seeds,
                    born_uids=born,
                    serial=_serial_id(e),
                )
                last = ra.steps[-1] if ra.steps else None
                if (
                    e.kind in ("add", "remove")
                    and last is not None
                    and last.kind == "edit"
                    and last.fit_win is fit_win
                ):
                    last.rows.append(row)
                else:
                    ra.steps.append(
                        _RefitStep(
                            window_id=int(e.window_id),
                            fit_win=fit_win,
                            kind=e.kind if e.kind in ("merge", "split") else "edit",
                            rows=[row],
                        )
                    )
            if action.implied_create:
                ra.implied = implied
        resolved.append(ra)
    return resolved


@requires_pipeline_file()
@_review_operation(
    "review undo",
    lambda r, a: review_undo_summary(r, a["dry_run"]),
    wrote=lambda r, a: not a["dry_run"],
)
def review_undo_impl(
    file_path: Union[Path, str],
    ids: Sequence[int],
    *,
    dry_run: bool = False,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> UndoResult:
    """Undo one or more recorded decisions by id, replaying the rest.

    A decision's id is its ``serial``, which never changes. The undo drops
    the named rows from the log and curates what survives through the one
    write path (:func:`_curate`): the surviving decisions are replayed from
    the automatic Stage 5 fit (snapshotted before the first decision) as one
    batch, and every status is recomputed under the recorded review
    parameters. An undo that leaves the fit-changing rows as they were (it
    drops bare accepts only) refits nothing; one that leaves none restores
    the automatic fit. Otherwise only the windows the undo can reach are
    refit (the incremental engine, :func:`_curate`): a window left with no
    decision and no edited window upstream gets its automatic fit back by
    copy, and every other window keeps its fit. The surviving rows are kept
    verbatim -- their serials,
    evidence and every peak ``derivation`` naming them unchanged -- and only
    their ``order_index`` positions are recomputed.

    ``dry_run`` returns the removed/surviving split and the resolved replay plan
    without mutating.

    A surviving created window keeps its id, but its geometry is re-derived
    from the creates that survive, in log order, so undoing an earlier create
    can change it; a widened window can change too. The undo is not refused
    for that: ``geometry_changed_window_ids`` lists every window whose
    geometry changes (a dry run lists the same), and the surviving creates are
    replanned before anything is fit, so one that can no longer be replayed
    under its id refuses with the file untouched.

    ``_shared`` is internal (see :func:`refit_window_impl`).

    Raises ``not_found`` (kind ``"decision"``, every unknown id; a
    :class:`ValueError`) if an id is unknown -- on a file with no recorded
    decisions, every id is -- ``bad_setting`` (``path`` ``"ids"``) when no id
    is given, and ``curation_conflict`` when the file cannot be curated
    (``predates_peak_identity`` / ``predates_replay_engine``, dry run
    included: :func:`_require_engine_file`), when undoing would orphan a
    window an undone decision created (``orphans_created_window``, ``ids``
    the serials of the decisions to undo with it), when it would orphan a
    peak an undone decision birthed that a kept decision removes, merges or
    splits (``orphans_peak``, ``ids`` likewise, transitively), when a kept
    decision can no longer be replayed (``replay_diverged``: a peak it
    removes is gone, or one it births is already there -- after undoing a
    remove of a peak a later decision re-added at the same position, say),
    when a surviving create can no longer take its recorded id
    (``replay_conflict``), or when a surviving seed falls off its window's
    re-derived geometry (``target_outside_window``). Every one is raised
    before anything is fit (dry run included), and the file is untouched. A
    log whose created window ids do not increase along it, or that has no
    automatic-fit baseline to replay from (the snapshot was removed from the
    file), is corrupt (``file_corrupt``), as it is for every write;
    ``file_incompatible`` when a newer engine curated the file. An undo that
    refits is epoch-gated (:func:`require_splice_compatible_environment`);
    one that refits nothing is not.

    The surviving decisions are replayed as recorded: each removes the peaks
    it names by ``peak_uid`` (``targets``) and births its peaks at its
    recorded seed positions under its recorded uids (``seeds_mhz``,
    ``born_uids``), one ACTION at a time in log order. Nothing is
    re-resolved by frequency or re-inferred, so a kept decision acts on the
    same peaks however far the undo moves its window's lines, and a peak a
    kept decision births keeps the uid it was born with. An action is the
    group of rows one user action recorded (:func:`_decision_action_groups`,
    the ``action_index`` evidence key): a ``review edit`` with several
    ``--add``/``--remove``, or a run of add/remove rows on one window in a
    curation file or action batch, was ONE joint refit that logs one row per
    frequency, and its surviving rows replay together as one joint refit
    again; an inferred merge or split is its own refit, as it was. The fitted
    *positions* are the replay's, so they can differ from before the undo.
    A peak of the automatic fit keeps its identifier, and undoing every
    decision restores the automatic fit's identifiers exactly.
    """
    path = str(file_path)
    _require_engine_file(path)
    _require_complete_fit(path, "review undo")
    review = load_stage6_review_from_file(path)
    log = list(review.decision_log)
    if log and not _has_stage5_baseline(path):
        raise _missing_baseline_error(path)
    valid_ids = {_serial_id(e) for e in log}
    # Every requested id the log does not hold, in request order -- on an
    # empty log, every requested id.
    unknown: List[int] = []
    for i in ids:
        if int(i) not in valid_ids and int(i) not in unknown:
            unknown.append(int(i))
    if unknown:
        detail = (
            f"no recorded decisions to undo (decision id(s) {unknown} do not exist)"
            if not log
            else f"unknown decision id(s) {unknown}; run 'review log' for valid ids"
        )
        raise NotFoundValueError("decision", unknown, message=detail)
    undo_set = {int(i) for i in ids}
    if not undo_set:
        raise BadSettingError(
            "ids",
            "at least one decision id",
            [int(i) for i in ids],
            message="no decision ids given to undo",
        )

    removed = [e for e in log if _serial_id(e) in undo_set]
    surviving = [e for e in log if _serial_id(e) not in undo_set]

    # Undoing a window creation orphans every decision made against that window:
    # replaying them would fail partway through, leaving the file half-rolled-back.
    # Refuse up front and name the ids the caller has to undo along with it.
    #
    # W3: an "add" entry carrying created_window evidence ALSO installs a
    # window (an implied create, W3's one-entry shape -- see
    # _replay_action_groups and _apply_refit_steps), so it must count here exactly like an
    # explicit "create_window" entry. Otherwise: add at X (implies window W),
    # a later add at Y resolves into W by W2's live-window coverage, then
    # undoing the first drops W out from under the second and the replay
    # fails partway -- precisely the failure this guard exists to refuse.
    dropped_windows = {int(e.window_id) for e in removed if _installs_window(e)} - {
        int(e.window_id) for e in surviving if _installs_window(e)
    }
    orphaned = sorted(
        _serial_id(e) for e in surviving if int(e.window_id) in dropped_windows
    )
    if orphaned:
        raise CurationConflictError(
            "orphans_created_window",
            orphaned,
            message=f"cannot undo: decision(s) {orphaned} act on window(s) "
            f"{sorted(dropped_windows)}, which the undone decision(s) "
            f"installed (a 'create_window' decision, or an implied create on "
            f"an 'add'). Undo them together.",
        )

    # A kept decision that removes, merges or splits a peak an undone one
    # birthed would find it gone: refuse up front, naming every decision to
    # undo with it (transitively: one that acts on a peak an orphan births).
    orphaned_peaks = _orphaned_peak_rows(log, undo_set)
    if orphaned_peaks:
        raise CurationConflictError(
            "orphans_peak",
            orphaned_peaks,
            message=f"cannot undo: decision(s) {orphaned_peaks} remove, merge or "
            f"split peaks that the undone decision(s) birthed (an add, merge or "
            f"split). Undo them together.",
        )

    _check_created_ids_monotone(path, surviving)
    plan = _replay_plan(surviving)

    # The structure the survivors install, and every refusal their replay
    # has, from the symbolic pass over them (_walk_log_rows), before anything
    # is fit: its refusals leave the file as it was, and a dry run reports
    # the same. The created-window overlay is a function of the ordered
    # create rows alone, so the undo changes an existing window's geometry
    # only when it drops a widening, or drops a create that a surviving
    # create was planned after.
    geometry_changed: List[int] = []
    shared = _shared
    undone_creates = [e for e in removed if _installs_window(e)]
    if undone_creates or any(e.kind in _FIT_EDIT_KINDS for e in surviving):
        if shared is None:
            shared = _build_shared_fit_ctx(path)
        walk = _walk_log_rows(
            path,
            shared,
            surviving,
            min_new_window_id=review.window_id_high_water + 1,
        )
        if undone_creates:
            geometry_changed = _geometry_changed_window_ids(
                shared.base_plan, review.created_windows, walk.overlay
            )

    if dry_run:
        return UndoResult(
            removed=removed,
            surviving=surviving,
            plan=plan,
            applied=0,
            dry_run=True,
            geometry_changed_window_ids=geometry_changed,
        )

    # The surviving log, curated as one write: one replay from the automatic
    # fit, one cascade, one persist. A cancel before the persist discards the
    # whole call with its transaction.
    _check_cancel()
    curated = _curate(
        path,
        review,
        surviving,
        _recorded_review_params(review),
        recorded=len(surviving),
        shared=shared,
        baseline_taken=_open_batch(path),
        snap_tol_mhz=refit_snap_tol_mhz_impl(path),
    )
    _finish_batch(curated, path)

    return UndoResult(
        removed=removed,
        surviving=surviving,
        plan=plan,
        applied=len(plan),
        dry_run=False,
        geometry_changed_window_ids=geometry_changed,
    )


@requires_pipeline_file()
def get_review_status_impl(file_path: Union[Path, str]) -> Stage6Review:
    """Load the :class:`Stage6Review` from *file_path*, or return an empty one.

    Read-only: does not write anything.  Safe to call before ``review run``.
    ``refit_required`` names the reason a Stage 6 write of the file would be
    refused (``"predates_peak_identity"`` / ``"predates_replay_engine"``, or
    ``"file_incompatible"`` for a review a newer engine wrote), or is ``None``
    when writes are accepted.
    """
    path = str(file_path)
    return replace(
        load_stage6_review_from_file(path), refit_required=_refit_required_reason(path)
    )


# ---------------------------------------------------------------------------
# review_run_impl: build/refresh the per-window attention routing layer
# ---------------------------------------------------------------------------


@dataclass
class ReviewRunResult:
    """Summary returned by :func:`review_run_impl`.

    Attributes
    ----------
    n_windows : int
        Number of windows the review holds a status for: every window the
        fit has a result for, plus each window flagged
        ``empty_window_residual`` (the fit holds no line in it).
    n_attention : int
        Number of windows with at least one attention reason.
    reason_counts : dict
        Maps attention-reason ``kind`` to the number of windows flagged for
        that reason (a window may contribute to multiple kinds).
    """

    n_windows: int
    n_attention: int
    reason_counts: Dict[str, int]
    #: The stages the call invalidated (canonical names, ``rerun_order``):
    #: always ``()``, since Stage 6 invalidates no stage.
    invalidated: Tuple[str, ...] = field(default=(), compare=False)


def _compute_attention_reasons(
    wf: FittingResult,
    *,
    spur_centers_mhz: List[float],
    acquisition_us: float,
    ledger_bar: float,
    attention_candidate_evidence: float,
    sideband: Sideband,
    kappa: float,
    noise_floor: float,
    auto_merged: bool = False,
    merged_freqs: Sequence[float] = (),
) -> List[AttentionReason]:
    """Derive the set of advisory attention reasons for one window.

    Parameters
    ----------
    wf :
        Per-window :class:`~ftmwpipeline.core.data_structures.FittingResult`.
    spur_centers_mhz :
        Gated spur center frequencies (molecular MHz) from the Stage 5
        ``SpectrumFit.parameters["spur_centers_mhz"]``.
    acquisition_us :
        Active acquisition length (µs); used to compute the Fourier
        resolution element ``1 / acquisition_us`` MHz for edge-boundary
        detection.
    ledger_bar :
        Display bar passed to :func:`derive_candidate_ledger`.
    sideband :
        Pipeline sideband (for ledger derivation).
    kappa :
        Shape-error kappa for the SNR-aware gate.
    noise_floor :
        Noise-regime chi-squared allowance.

    Returns
    -------
    list of AttentionReason
        Advisory flags, possibly empty.
    """
    from ..fitting.validation import shape_error_fraction, snr_aware_chi2_pass

    reasons: List[AttentionReason] = []

    # --- auto_merged_review: the end-of-Stage-5 pass merged a degenerate close
    # pair in this window. Prior-free, multiplicity is a high-bar claim, so the
    # default is to merge; this advisory (low severity) lets a user with catalog
    # support find the merge and split it back out (an add near the merged
    # line, read as a split of it by curation-intent inference). Not urgent --
    # the merge is the more-likely-correct call (~92% of the band is over-fits).
    if auto_merged:
        reasons.append(
            AttentionReason(
                kind="auto_merged_review",
                detail=(
                    "a degenerate sub-resolution pair was auto-merged; add a "
                    "second frequency near it (read as a split) if "
                    "catalog/model supports two lines"
                ),
                severity=0.1,
                locations=[float(f) for f in merged_freqs],
            )
        )

    chi2r = float(getattr(wf, "reduced_chi2", float("inf")))
    # Brightest finite in-window peak SNR -- the same definition as the Stage 5
    # validation gate (``stage5_validation_impl._window_snr_max``), so a
    # ``worst_eps`` flag means exactly "fails the SNR-aware acceptance gate".
    snr_max_val = float(
        max(
            (
                float(p.snr)
                for p in wf.fitted_peaks
                if p.snr is not None and math.isfinite(float(p.snr))
            ),
            default=0.0,
        )
    )

    # --- worst_eps: flag when the window FAILS the SNR-aware gate ----------
    # Guard against empty / SNR-less windows: a window with no finite-SNR peak
    # (snr_max == 0) has only a baseline "fit", so its chi2r is not a line-fit
    # quality signal -- flagging it as a gate failure is spurious (such windows
    # are dropped by the end-of-Stage-5 cleanup, but guard defensively).
    passes = snr_max_val <= 0.0 or snr_aware_chi2_pass(
        chi2r, snr_max_val, kappa, noise_floor
    )
    if not passes:
        eps = shape_error_fraction(chi2r, snr_max_val, noise_floor)
        reasons.append(
            AttentionReason(
                kind="worst_eps",
                detail=(
                    f"chi2r={chi2r:.3g} fails SNR-aware gate "
                    f"(snr_max={snr_max_val:.1f}, eps={eps:.4f})"
                ),
                severity=float(eps * max(snr_max_val, 1.0)),
            )
        )

    # NOTE: ``overfit_vif`` is retired as a standalone flag. A high amplitude VIF
    # has two populations and both are now handled without a user flag: the
    # sub-resolution over-splits are merged at end-of-Stage-5 (the VIF/singular
    # criterion plus the new ``collapse_frac_unc_threshold`` band), surfacing as
    # the low-severity ``auto_merged_review`` advisory; the genuine misfits fail
    # the SNR-aware gate and surface through ``worst_eps``. A well-resolved
    # doublet with a moderate VIF that passes the gate is simply fine and is no
    # longer flagged (it was pure over-production). Degeneracy remains discoverable
    # on demand via ``review rank --by max-vif``.

    # NOTE: a low-SNR fitted peak is deliberately NOT an attention reason. A
    # weak peak just above the survival floor is rarely actionable (an isolated
    # weak false positive does little harm), so flagging the whole band floods
    # the queue with low-value items. Weak windows are surfaced on demand via
    # ``review rank --by min-snr`` instead (exploration decoupled from flags).

    # --- candidate_bearing: flag when the window has candidates above bar ----
    center_mhz = _window_center(wf)
    if center_mhz is not None:
        res_element_mhz = (
            active_ft_bin_spacing_mhz(acquisition_us) if acquisition_us > 0.0 else None
        )
        cands = derive_candidate_ledger(
            wf,
            center_mhz=center_mhz,
            sideband=sideband,
            bar=ledger_bar,
            res_element_mhz=res_element_mhz,
        )
        # The attention flag fires ONLY on a strong ``residual_snr`` candidate
        # (a genuine missed line leaves residual SNR) clearing the stiff
        # attention threshold. The currencies are NOT comparable: an
        # ``aicc_delta`` candidate's value is a rejection *cost* (higher = more
        # rejected), so it must never be max()'d against residual SNR as if it
        # were support -- doing so flagged decisively-rejected near-misses as
        # the strongest "evidence". Audit/near-gate candidates still list under
        # ``review show --candidates`` (and the on-demand ranking), but they do
        # not raise an attention flag on their own.
        strong = [
            c
            for c in cands
            if c.evidence_kind == "residual_snr"
            and c.best_evidence >= attention_candidate_evidence
        ]
        if strong:
            best_ev = max(c.best_evidence for c in strong)
            reasons.append(
                AttentionReason(
                    kind="candidate_bearing",
                    detail=(
                        f"{len(strong)} strong residual candidate(s) "
                        f"(best residual SNR={best_ev:.2f})"
                    ),
                    severity=float(len(strong) + best_ev * 0.1),
                    locations=[float(c.frequency_mhz) for c in strong],
                )
            )

    # --- spur_adjacent: flag a surviving fitted line that sits on a gated spur
    # node. Line-on-node, not window-overlaps-spur: a window merely overlapping a
    # masked spur whose lines are all clear of it is benign and does not flag.
    if wf.fitted_peaks and spur_centers_mhz:
        resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else 0.1
        tol_mhz = SPUR_ADJACENT_MAX_SEP_RES * resolution_mhz
        centers = np.asarray(spur_centers_mhz, dtype=float)
        nearest: Optional[Tuple[float, float, float]] = None  # (sep, peak_f, spur_f)
        for p in wf.fitted_peaks:
            pf = float(p.frequency_mhz)
            j = int(np.argmin(np.abs(centers - pf)))
            sep = abs(pf - float(centers[j]))
            if sep <= tol_mhz and (nearest is None or sep < nearest[0]):
                nearest = (sep, pf, float(centers[j]))
        if nearest is not None:
            sep, pf, spur_f = nearest
            sep_res = sep / resolution_mhz if resolution_mhz > 0.0 else sep
            reasons.append(
                AttentionReason(
                    kind="spur_adjacent",
                    detail=(
                        f"fitted line at {pf:.4f} MHz is {sep_res:.2f} resolution "
                        f"element(s) ({sep * 1e3:.1f} kHz) from gated spur node "
                        f"at {spur_f:.4f} MHz -- confirm it is molecular"
                    ),
                    # closer to the node = higher attention
                    severity=float(2.0 - min(sep_res, SPUR_ADJACENT_MAX_SEP_RES)),
                    locations=[pf],
                )
            )

    # --- flat_decay: a Stage-2b flat-cluster line whose coherent decay was
    # ambiguous (a real line and a CW tone are indistinguishable there) was kept
    # rather than masked. Advisory -- surface it for review without forcing the
    # window into the active queue (most such picks are clock spurs).
    flat_decay_freqs = [
        float(p.frequency_mhz)
        for p in wf.fitted_peaks
        if getattr(p, "flat_decay", False)
    ]
    if flat_decay_freqs:
        reasons.append(
            AttentionReason(
                kind="flat_decay",
                detail=(
                    f"{len(flat_decay_freqs)} line(s) sat in the ambiguous "
                    "spur-decay band (real line vs CW tone indistinguishable); "
                    "kept for review -- confirm molecular or drop (review)"
                ),
                severity=0.2,
                locations=flat_decay_freqs,
            )
        )

    # --- edge_boundary: flag when a fitted peak sits within 1 resolution element of edge ---
    if wf.fitted_peaks and wf.window is not None and wf.window.freq_range is not None:
        flo, fhi = wf.window.freq_range
        resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else 0.1
        edge_peaks = [
            p
            for p in wf.fitted_peaks
            if (
                abs(float(p.frequency_mhz) - flo) <= resolution_mhz
                or abs(float(p.frequency_mhz) - fhi) <= resolution_mhz
            )
        ]
        if edge_peaks:
            freqs_str = ", ".join(f"{p.frequency_mhz:.4f}" for p in edge_peaks[:3])
            reasons.append(
                AttentionReason(
                    kind="edge_boundary",
                    detail=(
                        f"{len(edge_peaks)} peak(s) within 1 resolution element "
                        f"({resolution_mhz:.4f} MHz) of window edge: {freqs_str}"
                    ),
                    severity=float(len(edge_peaks)),
                    locations=[float(p.frequency_mhz) for p in edge_peaks],
                )
            )

    return reasons


def _empty_window_attention(
    path: str,
    spectrum_fit: SpectrumFit,
    review: Stage6Review,
    *,
    acquisition_us: float,
    fit_group: str = "stage5_fitting",
) -> Dict[int, AttentionReason]:
    """The ``empty_window_residual`` reasons of ``path``'s fitted-plan windows.

    Reads the fitted plan (from ``fit_group``) and the Stage 3 peak list only
    when Stage 5 recorded an edge handshake at all. Windows Stage 6 created,
    and windows a fit-changing decision was recorded on, are left out: the
    Stage 5 records do not describe their current fit. See
    :func:`~ftmwpipeline._internal.empty_window_attention.empty_window_reasons`.
    """
    from ..io.peak_serialization import load_peaks_from_hdf5
    from .fitted_plan import load_fitted_plan

    if not spectrum_fit.thaw_history and not spectrum_fit.replan_history:
        return {}
    with h5open(path, "r") as h5f:
        if "stage3_peaks" not in h5f or "stage4_windows" not in h5f:
            return {}
        stage3_peaks = load_peaks_from_hdf5(h5f["stage3_peaks"])
    fitted_plan = load_fitted_plan(path, fit_group_name=fit_group).plan
    excluded = {
        int(e.window_id) for e in review.decision_log if e.kind in _FIT_EDIT_KINDS
    }
    resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else 0.1
    return empty_window_reasons(
        spectrum_fit,
        fitted_plan.windows,
        stage3_peaks,
        excluded_window_ids=excluded,
        spur_tol_mhz=SPUR_ADJACENT_MAX_SEP_RES * resolution_mhz,
        created_windows=review.created_windows,
        dependency_edges=fitted_plan.dependency_edges,
    )


# ---------------------------------------------------------------------------
# Final-products consolidation (frequency calibration + sigma_f budget)
# ---------------------------------------------------------------------------


def set_sigma_floor_impl(file_path: Union[Path, str], sigma_floor_khz: float) -> None:
    """Persist the user's systematic accuracy floor into ``/frequency_calibration``
    and carry it into a stored final-products table.

    The floor is file-level provenance (a sibling of the source metadata), so
    any reported ``sigma_f`` is reproducible from the record alone and never
    depends on a transient flag. A stored table is rebuilt under the new floor
    (curation state untouched; see
    :func:`refresh_persisted_final_products_impl`), so the file never carries a
    ``sigma_f`` budget its own floor contradicts.
    """
    with atomic_write(file_path):
        _store_sigma_floor(file_path, sigma_floor_khz)
        refresh_persisted_final_products_impl(file_path)


def _store_sigma_floor(file_path: Union[Path, str], sigma_floor_khz: float) -> None:
    """Validate and write the floor only. :func:`review_run_impl` uses this
    directly, since it rebuilds the whole table itself right after."""
    floor = float(sigma_floor_khz)
    if floor < 0.0 or not math.isfinite(floor):
        raise BadSettingError(
            "sigma_floor_khz",
            "a finite float >= 0",
            sigma_floor_khz,
            message=f"sigma_floor_khz must be finite and non-negative, got "
            f"{sigma_floor_khz!r}",
        )
    with h5open(str(file_path), "a") as h5f:
        save_frequency_calibration_to_hdf5(FrequencyCalibration(floor), h5f)


def _fid_header_for_stamp(path: str) -> Optional[Tuple[float, str]]:
    """Return ``(probe_freq_mhz, sideband)`` for the staleness stamp, read
    straight from ``/stage0_fid_data/acquisition`` -- the exact attrs
    :func:`~ftmwpipeline.io.fid_serialization.load_fid_from_hdf5` uses to
    build ``FID.probe_freq_mhz`` / ``FID.sideband`` -- without touching the
    ``time_series_data`` dataset (hundreds of thousands of points) or paying
    :func:`load_fid_from_pipeline_impl`'s full pipeline-file validation. This
    runs on every ``get_final_products_impl`` call, so it has to be cheap.

    Deliberately reads the ``acquisition`` subgroup, not the sibling
    ``summary_probe_freq_mhz`` / ``summary_sideband`` attrs on
    ``stage0_fid_data`` -- those are a denormalized quick-access copy for
    cache tooling, not the field the loader treats as authoritative.

    Returns ``None`` when there is no FID header to read (e.g. a file that
    carries only a hand-built ``stage6_review`` group, as some report-table
    tests do) -- nothing to compare a stamp against, not evidence of staleness.
    """
    with h5open(path, "r") as h5f:
        fid_grp = h5f.get("stage0_fid_data")
        if fid_grp is None:
            return None
        acq = fid_grp.get("acquisition")
        if (
            acq is None
            or "probe_freq_mhz" not in acq.attrs
            or "sideband" not in acq.attrs
        ):
            return None
        probe_freq_mhz = float(acq.attrs["probe_freq_mhz"])
        sideband_raw = acq.attrs["sideband"]
    if isinstance(sideband_raw, bytes):
        sideband_raw = sideband_raw.decode("utf-8")
    return probe_freq_mhz, str(sideband_raw)


@requires_pipeline_file()
def frequency_calibration_impl(file_path: Union[Path, str]) -> CalibrationStamp:
    """The frequency calibration ``file_path`` is under right now.

    The public read behind ``api.frequency_calibration`` /
    ``Pipeline.frequency_calibration`` / ``timebase state``, and the single
    definition the staleness stamp is built from (see
    :func:`_current_calibration_stamp`).

    Read-only and *total*: it derives the state from the clock declaration and
    the live ``timebase_calibration`` rather than reading a persisted copy, and
    every input it cannot find degrades to the documented default rather than
    raising -- so it answers on a file that has been through no stage beyond
    the FID import, and it can never disagree with what a ``frame="calibrated"``
    call will actually apply. The one hard error is a file that is not there.

    Lives here, beside the derivation, rather than under Stage 6: it describes
    the *file*, not the Stage 6 products, and predates them.
    """
    path = str(file_path)
    if not pipeline_exists(path):
        raise FileNotFoundError(
            f"Pipeline file not found: {path}\n\n"
            f"To create a new pipeline:\n"
            f"  ftmwpipeline data import {path} path/to/data/"
        )
    cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
    header = _fid_header_for_stamp(path)
    with h5open(path, "r") as h5f:
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
    return CalibrationStamp(
        state=cal_state,
        epsilon=float(epsilon),
        sigma_epsilon=float(sigma_eps),
        sigma_floor_khz=float(floor_khz),
        probe_freq_mhz=Absent.NOT_RUN if header is None else float(header[0]),
        sideband=Absent.NOT_RUN if header is None else header[1],
    )


def _current_calibration_stamp(
    path: str,
) -> Optional[Tuple[str, float, float, float, float, str]]:
    """The six-tuple a fresh :class:`FinalProducts` would be stamped with
    *right now*: ``(calibration_state, epsilon, sigma_epsilon,
    sigma_floor_khz, probe_freq_mhz, sideband)``.

    :func:`frequency_calibration_impl`'s reading of the file, flattened for
    comparison against a persisted ``FinalProducts``' own stamp -- that
    comparison is the whole staleness check. ``None`` when the file has no FID
    header to derive a probe frequency / sideband from (see
    :func:`_fid_header_for_stamp`): nothing to compare against.
    """
    stamp = frequency_calibration_impl(path)
    if isinstance(stamp.probe_freq_mhz, Absent) or isinstance(stamp.sideband, Absent):
        return None
    return (
        stamp.state,
        stamp.epsilon,
        stamp.sigma_epsilon,
        stamp.sigma_floor_khz,
        stamp.probe_freq_mhz,
        stamp.sideband,
    )


def _final_products_is_stale(fp: Optional[FinalProducts], path: str) -> bool:
    """Whether ``fp`` -- the table persisted in ``path`` -- must be rebuilt
    before it is used: its calibration stamp no longer matches the file's
    currently-derived calibration (e.g. after a timebase re-run that never
    touched Stage 6), or the stored table predates the per-line fit fields
    (:func:`_persisted_table_predates_fit_fields`). ``None`` (no table built
    yet) is never "stale" -- there is nothing to have gone stale. Likewise
    when the file has nothing to derive a current stamp from
    (:func:`_current_calibration_stamp` returns ``None``): with no grounds to
    declare staleness, trust the persisted table rather than force a rebuild
    that cannot succeed anyway."""
    if fp is None:
        return False
    # A table that predates the fit fields is rebuilt whatever the calibration
    # stamp says; otherwise it would be served (and re-stored) with every fit
    # field UNDEFINED.
    if _persisted_table_predates_fit_fields(path):
        return True
    current = _current_calibration_stamp(path)
    if current is None:
        return False
    stamped = (
        fp.calibration_state,
        float(fp.epsilon),
        float(fp.sigma_epsilon),
        float(fp.sigma_floor_khz),
        float(fp.probe_freq_mhz),
        fp.sideband,
    )
    return stamped != current


def _persisted_table_predates_fit_fields(path: str) -> bool:
    """Whether the final-products table stored in ``path`` was written before
    ``FinalPeak`` carried the Stage 5 fit fields (``decay_time_us`` ...
    ``fit_window_mhz``). Such a table reads those fields as ``Absent``, so it
    is rebuilt from the raw fit -- in memory on a read, persisted by the next
    write that stores the table. Read-only."""
    try:
        with h5open(path, "r") as h5f:
            group = h5f.get("stage6_review")
            return final_products_predate_fit_fields(group)
    except OSError:
        return False


def _final_products_for_fit(
    path: str,
    spectrum_fit: SpectrumFit,
    fid: Any,
    *,
    prior: Optional[FinalProducts] = None,
    unchanged: Collection[int] = (),
) -> FinalProducts:
    """The final-products table of *spectrum_fit* under the file's current
    calibration: the declared sigma floor, the clock declaration and the
    derived calibration state, all read from ``path``. Read-only.

    *prior* is the table the file holds and *unchanged* the windows whose
    fit it was built from; their per-line fit fields are read off it rather
    than recomputed (:func:`_known_window_fit_fields`)."""
    with h5open(path, "r") as h5f:
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
        clocks_declared = fit_declares_clocks(h5f)
    cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
    probe = float(fid.probe_freq_mhz)
    known: Dict[int, Dict[str, Any]] = {}
    if (
        prior is not None
        and unchanged
        and not _persisted_table_predates_fit_fields(path)
    ):
        known = _known_window_fit_fields(prior, unchanged, probe, epsilon)
    return _build_final_products(
        spectrum_fit,
        probe_freq_mhz=probe,
        sideband=Sideband.coerce(fid.sideband),
        calibration_state=cal_state,
        epsilon=epsilon,
        sigma_epsilon=sigma_eps,
        sigma_floor_khz=floor_khz,
        clocks_declared=clocks_declared,
        known_window_fields=known,
    )


def _known_window_fit_fields(
    prior: FinalProducts,
    windows: Collection[int],
    probe_freq_mhz: float,
    epsilon: float,
) -> Dict[int, Dict[str, Any]]:
    """The per-line fit fields (:func:`_window_fit_fields`) of each window of
    *windows* that has a line in *prior*, read off its first line there. A
    window's fields are a function of its fit, the record length, the probe
    and ``epsilon`` alone, so they are reused only when *prior* was built
    under the same probe and ``epsilon`` (the record length is the fit's
    own). Empty otherwise."""
    if float(prior.probe_freq_mhz) != float(probe_freq_mhz) or float(
        prior.epsilon
    ) != float(epsilon):
        return {}
    wanted = {int(w) for w in windows}
    out: Dict[int, Dict[str, Any]] = {}
    for pk in prior.peaks:
        if isinstance(pk.window_id, Absent):
            continue
        w = int(pk.window_id)
        if w in wanted and w not in out:
            out[w] = {
                "decay_time_us": pk.decay_time_us,
                "decay_time_error_us": pk.decay_time_error_us,
                "shape": pk.shape,
                "fwhm_mhz": pk.fwhm_mhz,
                "fit_window_mhz": pk.fit_window_mhz,
            }
    return out


def _rebuild_final_products(path: str) -> Optional[FinalProducts]:
    """Rebuild the final-products table from the raw Stage 5 fit and the
    file's current calibration. Pure derivation, read-only -- touches no file
    and does not require a Stage 6 action. Returns ``None`` when there is no
    Stage 5 fit to derive from."""
    with h5open(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return None
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return _final_products_for_fit(
        path, spectrum_fit, load_fid_from_pipeline_impl(path)
    )


def _current_final_products(
    existing: Optional[FinalProducts], path: str
) -> Optional[FinalProducts]:
    """Return final products consistent with the file's current calibration.

    ``existing`` is whatever a persisted ``Stage6Review`` carries (``None``
    until ``review run`` first builds a table). When it is present but its
    stamp no longer matches the file's current calibration -- most commonly a
    timebase re-run with no Stage 6 action at all -- or the stored table
    predates the per-line fit fields, rebuild from the raw Stage 5 fit rather
    than returning the stale table.

    Read-only: never writes to ``path``, so it is safe to call on a file the
    caller has open only for reading (or not open at all). Callers that want
    the rebuilt table to stick persist it themselves.
    """
    if existing is None or not _final_products_is_stale(existing, path):
        return existing
    return _rebuild_final_products(path)


def refresh_persisted_final_products_impl(file_path: Union[Path, str]) -> bool:
    """Rewrite a stale persisted final-products table under the file's current
    calibration; return whether anything was written.

    A calibration change (a timebase re-run) moves every calibrated frequency
    and the ``sigma_epsilon`` term of every ``sigma_f`` without re-fitting
    anything: Stage 5 stores raw-frame frequencies, and the timebase result
    only re-derives the calibrated table from them. Reads already rebuild a
    stale table on the fly (:func:`get_final_products_impl`); this makes the
    stored copy agree too, so the file never carries a table its own
    calibration contradicts. Only the table is replaced -- per-window
    provenance, attention reasons, the decision log and created windows are
    written back untouched. No-op when no table has been built yet or the
    stored one is current. A rewrite restamps ``stage6_review``'s analysis
    epoch.
    """
    path = str(file_path)
    review = load_stage6_review_from_file(path)
    existing = review.final_products
    if existing is None or not _final_products_is_stale(existing, path):
        return False
    rebuilt = _rebuild_final_products(path)
    if rebuilt is None:
        return False
    _write_stage6_review_only(replace(review, final_products=rebuilt), path)
    # The rewritten products are Stage 6 output produced now, so Stage 6
    # records the epoch that produced them; a stamp that cannot be written
    # raises (see :mod:`ftmwpipeline.io.provenance`).
    stamp_stage_epoch_in_file(path, "stage6_review")
    logger.info(
        "Refreshed the Stage 6 final-products table under the current "
        "calibration (epsilon=%.3e, sigma_epsilon=%.3e)",
        rebuilt.epsilon,
        rebuilt.sigma_epsilon,
    )
    return True


@requires_pipeline_file()
def get_final_products_impl(file_path: Union[Path, str]) -> Optional[FinalProducts]:
    """Return the current Stage 6 final-products table, or ``None``.

    Rebuilds from the raw Stage 5 fit -- in memory, without persisting --
    when the persisted table's calibration stamp no longer matches the
    file's current calibration (e.g. a timebase re-run since the table was
    last built), or when the stored table predates the per-line fit fields.
    See :func:`_current_final_products`.
    """
    path = str(file_path)
    existing = load_stage6_review_from_file(path).final_products
    return _current_final_products(existing, path)


def _resolve_calibration_clocks(path: str) -> Tuple[ClockSource, ...]:
    """The clock declaration the calibration state is derived from.

    The same rule the timebase calibration resolves its declaration with
    (:func:`~ftmwpipeline._internal.timebase_impl._resolve_clock_sources`): a
    non-empty persisted Stage 5 ``spur.clocks`` first, then the recommended
    declaration (``clocks set`` / a loader-injected one). An *empty* persisted
    declaration falls through, so a ``clocks set`` after the fit followed by
    ``timebase run`` applies the measured epsilon -- the state agrees with the
    declaration the timebase result records it ran with.

    Degrades to "no declaration" on an unreadable record, as the state's
    derivation always has.
    """
    from ..io.stage_fit_settings_serialization import (
        load_stage_fit_settings_from_h5,
        read_recommended_clock_sources,
    )

    try:
        persisted = load_stage_fit_settings_from_h5(path)
    except Exception:
        persisted = None
    if persisted is not None and persisted.spur.clocks:
        return tuple(persisted.spur.clocks)
    try:
        return tuple(read_recommended_clock_sources(path) or ())
    except Exception:
        return ()


def _derive_frequency_calibration(
    path: str,
) -> Tuple[CalibrationState, float, float]:
    """Derive the calibration state and the applied (epsilon, sigma_epsilon).

    The state is *derived*, not stored: it follows from the clock declaration
    (``spur.clocks``) and whether a usable ``timebase_calibration`` is present.

    - No unlocked clock declared (or no declaration) -> ``"rb_locked"`` (the
      "assume Rb-locked when nothing says otherwise" default); epsilon is a
      null op.
    - An unlocked digitizer declared **and** a timebase calibration whose
      preconditions passed -> ``"self_calibrated"``; the measured epsilon and
      its uncertainty are applied.
    - An unlocked digitizer declared but no usable timebase calibration ->
      ``"uncalibrated"``; frequencies are reported as-is (caveated).

    The declaration is :func:`_resolve_calibration_clocks`.
    """
    clocks = _resolve_calibration_clocks(path)

    has_unlocked = any(not c.locked for c in clocks)
    if not has_unlocked:
        return "rb_locked", 0.0, 0.0

    from .timebase_impl import (
        load_timebase_calibration_impl,
        timebase_calibration_present,
    )

    if not timebase_calibration_present(path):
        return "uncalibrated", 0.0, 0.0

    try:
        tc = load_timebase_calibration_impl(path)["timebase_calibration"]
    except Exception:
        return "uncalibrated", 0.0, 0.0

    if not tc.preconditions_passed or not math.isfinite(tc.sigma_epsilon):
        return "uncalibrated", 0.0, 0.0

    return "self_calibrated", float(tc.epsilon), float(tc.sigma_epsilon)


#: ``FinalPeak``'s per-line fit fields for a line with no Stage 5 fit record
#: behind it: every one is undefined.
_NO_FIT_RECORD_FIELDS: Dict[str, Any] = {
    "decay_time_us": Absent.UNDEFINED,
    "decay_time_error_us": Absent.UNDEFINED,
    "shape": Absent.UNDEFINED,
    "fwhm_mhz": Absent.UNDEFINED,
    "detection_index": Absent.UNDEFINED,
    "fit_window_mhz": Absent.UNDEFINED,
}


def _recorded_acquisition_us(spectrum_fit: SpectrumFit) -> Union[float, Absent]:
    """The record length ``T`` (us) the Stage 5 fit recorded.

    The same value :func:`~ftmwpipeline._internal.read_impl.read_metadata_impl`
    reports as ``stage5.acquisition_us`` (both read the fit's persisted
    ``parameters["acquisition_us"]``), so a width computed from it is the one a
    client computes from that key. ``Absent.NOT_RUN`` when the fit recorded
    none; ``Absent.UNDEFINED`` when the recorded value is not a positive
    finite number.
    """
    raw = spectrum_fit.parameters.get("acquisition_us")
    if raw is None:
        return Absent.NOT_RUN
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Absent.UNDEFINED
    if not math.isfinite(value) or value <= 0.0:
        return Absent.UNDEFINED
    return value


def _window_fit_fields(
    wf: FittingResult,
    *,
    acquisition_us: Union[float, Absent],
    probe_freq_mhz: float,
    epsilon: float,
) -> Dict[str, Any]:
    """The per-line fit fields every line of window ``wf`` shares.

    ``decay_time_us`` / ``decay_time_error_us`` are the window's shared
    ``tau`` and its error (the error is undefined when ``tau`` was held fixed,
    or the fit left it without one); ``shape`` is the window's line shape;
    ``fwhm_mhz`` is exactly ``feature_fwhm(tau, acquisition_us, shape=shape)``
    (absent with the same reason as ``acquisition_us`` when that is absent,
    undefined when ``tau`` or ``shape`` is); ``fit_window_mhz`` is the
    window's ``(low, high)`` bounds moved to the calibrated frame with the same
    correction as ``frequency_mhz`` (:func:`_frame_to_calibrated`).
    """
    tau_entry = wf.shared_parameters.get("tau_us") or {}

    decay_time: Union[float, Absent] = Absent.UNDEFINED
    tau_raw = tau_entry.get("value")
    if tau_raw is not None:
        tau = float(tau_raw)
        if math.isfinite(tau) and tau > 0.0:
            decay_time = tau

    decay_error: Union[float, Absent] = Absent.UNDEFINED
    err_raw = tau_entry.get("error")
    if (
        not isinstance(decay_time, Absent)
        and tau_entry.get("fitted") is not False
        and err_raw is not None
    ):
        err = float(err_raw)
        if math.isfinite(err) and err >= 0.0:
            decay_error = err

    shape: Union[str, Absent] = Absent.UNDEFINED
    shape_raw = getattr(wf, "shape", None)
    if shape_raw is not None and str(getattr(shape_raw, "value", shape_raw)):
        shape = str(getattr(shape_raw, "value", shape_raw))

    fwhm: Union[float, Absent]
    if isinstance(acquisition_us, Absent):
        fwhm = acquisition_us
    elif isinstance(decay_time, Absent) or isinstance(shape, Absent):
        fwhm = Absent.UNDEFINED
    else:
        try:
            fwhm = float(feature_fwhm(decay_time, acquisition_us, shape=shape))
        except ValueError:
            fwhm = Absent.UNDEFINED
        if not isinstance(fwhm, Absent) and not math.isfinite(fwhm):
            fwhm = Absent.UNDEFINED

    fit_window: Union[Tuple[float, float], Absent] = Absent.UNDEFINED
    window = wf.window
    if window is not None:
        lo_raw, hi_raw = (float(v) for v in window.freq_range)
        if math.isfinite(lo_raw) and math.isfinite(hi_raw):
            lo, hi = sorted(
                _frame_to_calibrated(v, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon)
                for v in (lo_raw, hi_raw)
            )
            fit_window = (lo, hi)

    return {
        "decay_time_us": decay_time,
        "decay_time_error_us": decay_error,
        "shape": shape,
        "fwhm_mhz": fwhm,
        "fit_window_mhz": fit_window,
    }


def _line_fit_fields(
    pk: FittedPeak, window_fields: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """``FinalPeak``'s per-line fit fields for ``pk``: its window's shared
    fields (:func:`_window_fit_fields`) plus its ``detection_index``
    (undefined when no Stage 3 detection seeded the line). Every
    field is ``Absent.UNDEFINED`` when the line has no fit record behind it
    (no window id, or no fit result for that window)."""
    shared = None if pk.window_id is None else window_fields.get(int(pk.window_id))
    if shared is None:
        return dict(_NO_FIT_RECORD_FIELDS)
    # The fit writes -1 when no Stage 3 detection seeded the line (a line added
    # to a window with no free Stage 3 peaks, e.g. one created during review);
    # the contract carries that as Absent, never as the -1 sentinel.
    detection_index: Union[int, Absent] = Absent.UNDEFINED
    try:
        index = int(pk.detection_index)
    except (TypeError, ValueError):
        index = -1
    if index >= 0:
        detection_index = index
    return {**shared, "detection_index": detection_index}


def _build_final_products(
    spectrum_fit: SpectrumFit,
    *,
    probe_freq_mhz: float,
    sideband: Sideband,
    calibration_state: str,
    epsilon: float,
    sigma_epsilon: float,
    sigma_floor_khz: float,
    clocks_declared: bool,
    known_window_fields: Optional[Mapping[int, Dict[str, Any]]] = None,
) -> FinalProducts:
    """Consolidate the Stage 5 line list into the calibrated final-products table.

    ``known_window_fields`` holds windows' per-line fit fields a caller
    already has under this calibration (:func:`_known_window_fit_fields`);
    they are used as computed.

    ``clocks_declared`` says whether the fit recorded a clock declaration
    (:func:`~ftmwpipeline.io.stage6_review_serialization.fit_declares_clocks`):
    it tells a line the lattice test never ran on from an off-lattice one.

    Applies the timebase scale correction in the baseband frame
    (``f_corr = probe + (f_raw - probe)/(1+epsilon)``, sideband-independent) and
    builds the three-term ``sigma_f`` budget per accepted peak:
    ``sqrt(sigma_stat^2 + (sigma_epsilon * f_baseband)^2 + sigma_floor^2)``.
    Each line also carries the per-line fit fields joined from the fit record
    of its window (:func:`_window_fit_fields`, :func:`_line_fit_fields`), with
    the window bounds moved to the calibrated frame by the same correction.
    """
    floor_khz = float(sigma_floor_khz)
    acquisition_us = _recorded_acquisition_us(spectrum_fit)
    window_fields: Dict[int, Dict[str, Any]] = dict(known_window_fields or {})
    for wf in spectrum_fit.window_fits:
        if wf.window_id is not None and int(wf.window_id) not in window_fields:
            window_fields[int(wf.window_id)] = _window_fit_fields(
                wf,
                acquisition_us=acquisition_us,
                probe_freq_mhz=probe_freq_mhz,
                epsilon=epsilon,
            )
    final_peaks: List[FinalPeak] = []
    for pk in spectrum_fit.fitted_peaks:
        f_raw = float(pk.frequency_mhz)
        f_baseband_mhz = abs(f_raw - probe_freq_mhz)

        if epsilon != 0.0:
            f_corr = probe_freq_mhz + (f_raw - probe_freq_mhz) / (1.0 + epsilon)
        else:
            f_corr = f_raw

        # A line with no statistical frequency error has no honest total
        # either: both are undefined, never computed with the term dropped.
        freq_err = float_or_absent(pk.frequency_error)
        sigma_eps_khz = float(sigma_epsilon) * f_baseband_mhz * 1.0e3
        sigma_stat_khz: Union[float, Absent] = Absent.UNDEFINED
        sigma_f_khz: Union[float, Absent] = Absent.UNDEFINED
        if not isinstance(freq_err, Absent):
            stat = freq_err * 1.0e3
            sigma_stat_khz = stat
            sigma_f_khz = math.sqrt(stat**2 + sigma_eps_khz**2 + floor_khz**2)

        amp = float(pk.amplitude)
        amp_err = float_or_absent(pk.amplitude_error)
        snr_val = float_or_absent(pk.snr)
        # Propagate the amplitude error into an SNR error (SNR scales with
        # amplitude at fixed noise): sigma_snr = snr * sigma_amp / amp.
        snr_err: Union[float, Absent] = Absent.UNDEFINED
        if (
            not isinstance(snr_val, Absent)
            and not isinstance(amp_err, Absent)
            and amp != 0.0
        ):
            snr_err = abs(snr_val) * abs(amp_err / amp)

        # Knockout significance. Never tested -> NOT_RUN for all three;
        # tested but non-finite (a refit that did not converge) -> UNDEFINED.
        ko = pk.knockout
        ko_not_run = knockout_absence(
            None if ko is None else ko.supported,
            None if ko is None else ko.delta_chi2,
        )
        ko_p: Union[float, Absent] = Absent.NOT_RUN
        ko_supported: Union[bool, Absent] = Absent.NOT_RUN
        ko_aicc: Union[float, Absent] = Absent.NOT_RUN
        if ko is not None and ko_not_run is None:
            ko_p = float_or_absent(ko.p_value)
            ko_supported = bool(ko.supported)
            ko_aicc = float_or_absent(ko.aicc_delta)

        final_peaks.append(
            FinalPeak(
                frequency_mhz=f_corr,
                frequency_raw_mhz=f_raw,
                f_baseband_mhz=f_baseband_mhz,
                sigma_f_khz=sigma_f_khz,
                sigma_stat_khz=sigma_stat_khz,
                sigma_eps_khz=sigma_eps_khz,
                sigma_floor_khz=floor_khz,
                amplitude=amp,
                phase=float_or_absent(pk.phase),
                snr=snr_val,
                origin=str(pk.origin),
                window_id=int_or_absent(pk.window_id, sentinel=None),
                amplitude_error=amp_err,
                phase_error=float_or_absent(pk.phase_error),
                snr_error=snr_err,
                clock_lattice=clock_lattice_or_absent(
                    pk.clock_lattice, declared=clocks_declared
                ),
                derivation=int_or_absent(pk.derivation, sentinel=None),
                peak_uid=int_or_absent(pk.peak_uid, sentinel=None),
                knockout_p_value=ko_p,
                knockout_supported=ko_supported,
                knockout_aicc_delta=ko_aicc,
                **_line_fit_fields(pk, window_fields),
            )
        )

    return FinalProducts(
        peaks=final_peaks,
        calibration_state=calibration_state,
        epsilon=float(epsilon),
        sigma_epsilon=float(sigma_epsilon),
        sigma_floor_khz=floor_khz,
        probe_freq_mhz=float(probe_freq_mhz),
        sideband=sideband.value,
    )


@requires_pipeline_file()
def review_run_impl(
    file_path: Union[Path, str],
    *,
    bar: Optional[float] = None,
    attention_candidate_evidence: Optional[float] = None,
    kappa: Optional[float] = None,
    noise_floor: Optional[float] = None,
    sigma_floor_khz: Optional[float] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> ReviewRunResult:
    """Build or refresh the Stage 6 attention-routing layer and final products.

    A long operation (``review run``), reported as the ``review`` stage:
    ``StageStarted`` (after the fit is read, so its start line can name the
    window count), then ``StageFinished`` with :func:`review_run_summary`
    once the review is written. ``cancel`` is checked before the stage
    starts; it has no window loop.

    Loads the Stage 5 fit, computes advisory attention reasons for every
    window, consolidates the calibrated final-products table (frequencies
    corrected for the digitizer timebase scale error and the three-term
    ``sigma_f`` budget), and persists a
    :class:`~ftmwpipeline.core.data_structures.Stage6Review` to the
    ``stage6_review`` HDF5 group.  Marks the ``stage6_review`` tracker stage
    complete.

    Idempotent: the decision log and the created-window overlay are kept,
    and every status is recomputed from the fit (:func:`_curated_statuses`),
    with each window's ``provenance`` taken from its rows in the log. It is
    a Stage 6 write like any other (:func:`_curate`, the log unchanged and
    the review parameters replaced): it refits nothing, so it keeps the
    persisted fits and is not epoch-gated, and it takes the undo baseline if
    none was taken yet.

    The four attention-routing parameters are recorded in the review
    (``Stage6Review.review_params``) and every later Stage 6 write -- an
    edit, an apply, an undo -- computes statuses under them. A parameter left
    ``None`` keeps its recorded value (:data:`DEFAULT_REVIEW_PARAMS` on a
    review that records none).

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    bar :
        Display bar forwarded to :func:`get_candidate_ledger_impl` for the
        candidate-bearing attention reason (default: the recorded value, else
        :data:`DEFAULT_DISPLAY_BAR`).
    attention_candidate_evidence :
        A window flags ``candidate_bearing`` only when its strongest revivable
        candidate's evidence clears this threshold (default: the recorded
        value, else :data:`DEFAULT_ATTENTION_CANDIDATE_EVIDENCE`) -- stiffer
        than ``bar`` so the attention surface stays actionable while the
        ledger still lists every candidate above ``bar``.
    kappa :
        Shape-error kappa for the SNR-aware chi-squared gate (default: the
        recorded value, else :data:`DEFAULT_SHAPE_ERROR_KAPPA`).
    noise_floor :
        Noise-regime chi-squared allowance (default: the recorded value, else
        :data:`DEFAULT_CHI2R_NOISE_FLOOR`).
    sigma_floor_khz :
        When given, persist this user-declared systematic accuracy floor (kHz)
        into the file-level ``/frequency_calibration`` record before
        consolidating, then fold it into every peak's ``sigma_f`` budget.  When
        ``None`` (default) the persisted floor is used unchanged (default
        ``0.0`` if never declared).

    Returns
    -------
    ReviewRunResult
        Total window count, attention window count, and per-kind counts.

    Raises
    ------
    ValueError
        When Stage 5 has not been run yet.
    BadSettingError
        When ``bar``, ``attention_candidate_evidence``, ``kappa`` or
        ``noise_floor`` is given and is not finite and non-negative; nothing
        is read or written.
    CurationConflictError
        ``predates_peak_identity`` / ``predates_replay_engine`` when the file
        cannot be curated (:func:`_require_engine_file`); nothing is written.
    PipelineCompatibilityError
        ``file_incompatible`` when a newer engine curated the file.
    """
    path = str(file_path)
    # Checked before anything is read: every later write routes attention
    # under what this call records, so a bad value must never reach the file.
    passed = _validated_review_params(
        bar=bar,
        attention_candidate_evidence=attention_candidate_evidence,
        kappa=kappa,
        noise_floor=noise_floor,
    )
    ops = operation_events("review run", events, cancel)
    # One atomic write that covers the reads it is built from: the transaction
    # (and its write_conflict stat) begins before the fit is read, and commits
    # before StageFinished. The stage scope opens after the read (its start
    # line names the window count), so the two are closed by hand in order:
    # the transaction first (the replace), then StageFinished.
    with ExitStack() as transaction:
        transaction.enter_context(atomic_write(path))
        _require_engine_file(path)
        with h5open(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                raise StageDependencyError(
                    "review",
                    ["stage5_fitting"],
                    Path(str(path)),
                    command="fit run",
                    message="No Stage 5 fit found in this file. Run 'fit run' first.",
                )
            n_windows = int(h5f["stage5_fitting"].attrs.get("n_windows", 0))
            # The existing review's log, overlay and recorded parameters.
            existing_review: Stage6Review
            if "stage6_review" in h5f:
                existing_review = load_stage6_review_from_hdf5(h5f["stage6_review"])
            else:
                existing_review = Stage6Review()

        params = replace(_recorded_review_params(existing_review), **passed)

        # The "Stage 6 review: routing attention for N windows" start line and
        # the "Saved Stage 6 review to ..." end line are rendered from the
        # stage's StageStarted / StageFinished events.
        with ops.stage(
            Stage.REVIEW,
            verb="review run",
            detail={"n_windows": n_windows, "path": path},
            file_path=path,
        ) as scope:
            result, final_products = _review_run(
                path,
                existing_review,
                params=params,
                sigma_floor_khz=sigma_floor_khz,
            )
            transaction.close()  # the replace
            scope.finish(
                review_run_summary(result, final_products), detail={"path": path}
            )
    return result


def _review_run(
    path: str,
    existing_review: Stage6Review,
    *,
    params: ReviewParams,
    sigma_floor_khz: Optional[float],
) -> Tuple[ReviewRunResult, Optional[FinalProducts]]:
    """The writing half of :func:`review_run_impl` (inside its stage scope):
    the file's log curated under *params* and persisted. Returns the result
    and the final-products table it wrote."""
    # A newly-declared accuracy floor is persisted as file-level provenance
    # first, so the final-products budget reflects exactly what the record
    # carries (never a transient flag).
    if sigma_floor_khz is not None:
        _store_sigma_floor(path, sigma_floor_khz)
    with h5open(path, "r") as h5f:
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
        floor_record = frequency_calibration_provenance(h5f)
    if floor_record is None or floor_record.is_pre_provenance:
        # Never declared (or declared before the record carried a version):
        # write the floor this run applies, so the record says it explicitly
        # rather than leaving "never declared" to read as the default.
        _store_sigma_floor(path, floor_khz)

    log = list(existing_review.decision_log)
    curated = _curate(
        path,
        existing_review,
        log,
        params,
        recorded=len(log),
        shared=None,
        baseline_taken=_open_batch(path),
        snap_tol_mhz=refit_snap_tol_mhz_impl(path),
    )
    _finish_batch(curated, path)
    _update_stage_completion(path, "stage6_review")

    # Compute summary.
    new_statuses = curated.review.window_statuses
    reason_counts: Dict[str, int] = {}
    n_attention = 0
    for status in new_statuses.values():
        if status.needs_attention:
            n_attention += 1
        for reason in status.attention_reasons:
            reason_counts[reason.kind] = reason_counts.get(reason.kind, 0) + 1

    return (
        ReviewRunResult(
            n_windows=len(new_statuses),
            n_attention=n_attention,
            reason_counts=reason_counts,
        ),
        curated.review.final_products,
    )


def _validated_review_params(**values: Optional[float]) -> Dict[str, float]:
    """The attention-routing parameters a ``review run`` was passed, each
    checked finite and non-negative (``bad_setting`` naming the argument);
    one left ``None`` keeps its recorded value and is not in the result."""
    out: Dict[str, float] = {}
    for name, value in values.items():
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = math.nan
        if v < 0.0 or not math.isfinite(v):
            raise BadSettingError(
                name,
                "a finite float >= 0",
                value,
                message=f"{name} must be finite and non-negative, got {value!r}",
            )
        out[name] = v
    return out


def _recorded_review_params(review: Stage6Review) -> ReviewParams:
    """The attention-routing parameters *review* records, or
    :data:`DEFAULT_REVIEW_PARAMS` when none were recorded."""
    return review.review_params or DEFAULT_REVIEW_PARAMS


def _log_provenance(log: Sequence[DecisionLogEntry]) -> Dict[int, str]:
    """Each window's provenance as its rows give it: from its last row, a bare
    accept gives ``"reviewed"`` and anything else ``"user-edited"`` (an accept
    with a candidate is recorded as an ``add``). A window with no row is
    ``"auto"`` and is not in the map."""
    out: Dict[int, str] = {}
    for entry in log:
        out[int(entry.window_id)] = (
            "reviewed" if entry.kind == "accept" else "user-edited"
        )
    return out


def _curated_statuses(
    path: str,
    spectrum_fit: SpectrumFit,
    review: Stage6Review,
    params: ReviewParams,
    sideband: Sideband,
    *,
    fit_group: str = "stage5_fitting",
) -> Dict[int, WindowReviewStatus]:
    """Every window's status, computed from scratch: a pure function of the
    window fits, *params*, and the decision log and created-window overlay
    *review* carries. Read-only.

    Fresh attention reasons for each fitted window under *params*, plus the
    empty-window reasons (:func:`_empty_window_attention`, which reads the log
    and the overlay), then :func:`_settle_empty_window_reasons` once over every
    window, since a created window anywhere can take another window's reason
    over. Provenance comes from the log (:func:`_log_provenance`); the
    statuses *review* stores are not read. ``review run``, every curation
    write and the full-replay reference all compute statuses here, so a write
    leaves the statuses a fresh ``review run`` with the same parameters would.
    ``fit_group`` names the fit group the fitted plan is read from (the
    full-replay reference reads the undo baseline's)."""
    spur_centers_mhz = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
    ]
    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    merged_window_freqs = _auto_merged_window_freqs(spectrum_fit)
    provenance = _log_provenance(review.decision_log)

    statuses: Dict[int, WindowReviewStatus] = {}
    for wf in spectrum_fit.window_fits:
        wid = int(wf.window_id) if wf.window_id is not None else -1
        reasons = _compute_attention_reasons(
            wf,
            spur_centers_mhz=spur_centers_mhz,
            acquisition_us=acquisition_us,
            ledger_bar=params.bar,
            attention_candidate_evidence=params.attention_candidate_evidence,
            sideband=sideband,
            kappa=params.kappa,
            noise_floor=params.noise_floor,
            auto_merged=wid in merged_window_freqs,
            merged_freqs=merged_window_freqs.get(wid, ()),
        )
        statuses[wid] = WindowReviewStatus(
            window_id=wid,
            provenance=provenance.get(wid, "auto"),
            attention_reasons=reasons,
            invalidated=False,
        )

    # A window of the fitted plan the fit holds no line in, whose edge Stage 5
    # still flagged: it has no window result above, so it gets its status here.
    for wid, empty_reason in _empty_window_attention(
        path,
        spectrum_fit,
        review,
        acquisition_us=acquisition_us,
        fit_group=fit_group,
    ).items():
        if wid in statuses:
            statuses[wid].attention_reasons.append(empty_reason)
            continue
        statuses[wid] = WindowReviewStatus(
            window_id=wid,
            provenance=provenance.get(wid, "auto"),
            attention_reasons=[empty_reason],
            invalidated=False,
        )

    if any(
        r.kind in EMPTY_WINDOW_KINDS
        for st in statuses.values()
        for r in st.attention_reasons
    ):
        from .fitted_plan import load_fitted_plan

        _settle_empty_window_reasons(
            statuses,
            spectrum_fit,
            review.created_windows,
            base_plan_windows=load_fitted_plan(
                path, fit_group_name=fit_group
            ).plan.windows,
            edited_window_ids={
                int(e.window_id)
                for e in review.decision_log
                if e.kind in _FIT_EDIT_KINDS
            },
        )
    return statuses


# ---------------------------------------------------------------------------
# D1: the fingerprint a ReviewSession checks before trusting its cached
# _SharedFitCtx. See ``scratch/preview-session-plan.md`` ("Task D") and
# ``scratch/bq-correspondence/reply-preview-execute.md`` section 4.
# ---------------------------------------------------------------------------

_FitCtxFingerprint = Tuple[int, int, Tuple[Any, ...]]
"""``(st_mtime_ns, st_size, stage_provenance)`` -- see
:func:`_compute_fit_ctx_fingerprint`."""


def _fit_ctx_stage_provenance(path: str) -> Tuple[Any, ...]:
    """Cheap, attrs-only proxy for whether :func:`_build_shared_fit_ctx`
    would now return something different than the last time this was read --
    the "stage provenance" half of D1's fingerprint, behind the mtime+size
    fast path.

    ``/pipeline_stages``' own ``completed_stages``/``last_updated`` attrs
    catch a Stage 5 (or earlier) re-run, but NOT a timebase re-run:
    ``timebase_calibration`` is in no stage's dependency list
    (``file_manager.py:283``) and ``timebase_impl`` never calls
    ``invalidate_downstream_stages`` -- the exact gap A7 hit for
    final-products staleness. A timebase re-run changes
    ``_SharedFitCtx.epsilon`` / ``calibration_state`` without touching the
    stage tracker at all, so :func:`_current_calibration_stamp` (the same
    stamp A7 already computes) is read directly here too. The Stage 5
    ``shape`` attr and the Stage 6 decision-log length round out the set:
    together they cover every input :func:`_build_shared_fit_ctx` derives
    from that could plausibly change without moving the file's mtime or
    size -- defense in depth behind the fast path, not a replacement for it
    (see :func:`_compute_fit_ctx_fingerprint`).
    """
    with h5open(path, "r") as h5f:
        stages_attrs = h5f["pipeline_stages"].attrs if "pipeline_stages" in h5f else {}
        completed = str(stages_attrs.get("completed_stages", "[]"))
        last_updated = str(stages_attrs.get("last_updated", ""))

        shape_attr: Optional[str] = None
        if "stage5_fitting" in h5f:
            raw_shape = h5f["stage5_fitting"].attrs.get("shape")
            if isinstance(raw_shape, bytes):
                raw_shape = raw_shape.decode("utf-8")
            shape_attr = None if raw_shape is None else str(raw_shape)

        decision_log_len = 0
        if "stage6_review" in h5f and "decision_log" in h5f["stage6_review"]:
            raw_log = h5f["stage6_review/decision_log"].attrs.get("data", "[]")
            try:
                decision_log_len = len(json.loads(raw_log))
            except (TypeError, ValueError):
                decision_log_len = -1

    cal_stamp = _current_calibration_stamp(path)
    return (completed, last_updated, shape_attr, decision_log_len, cal_stamp)


def _compute_fit_ctx_fingerprint(path: str) -> _FitCtxFingerprint:
    """The validity fingerprint a :class:`ReviewSession` checks before
    trusting its cached :class:`_SharedFitCtx` (D1): ``st_mtime_ns`` and
    ``st_size`` (a few microseconds; the near-free fast path -- any write to
    the file changes at least one) plus :func:`_fit_ctx_stage_provenance` (a
    handful of attrs-only HDF5 reads, no dataset loads; ~0.8 ms measured,
    against ~420 ms for the shared context it guards) as a second-tier check
    for whatever the fast path alone might miss -- a foreign write landing
    inside the filesystem's mtime granularity.

    ALWAYS re-read live from disk -- never predicted from what a caller
    believes it just wrote. In particular, :class:`ReviewSession` re-reads
    this after every one of its own writes rather than computing what the
    new value "should" be, so a second writer landing in the very same
    instant is still caught the next time the session is used (settled
    decision 7 in ``scratch/preview-session-plan.md``: no on-disk generation
    counter -- this re-read is the substitute).
    """
    st = os.stat(resolve(path))
    return (st.st_mtime_ns, st.st_size, _fit_ctx_stage_provenance(path))


def _resolve_curation_call(
    path: str, source: CurationSource, frame: Optional[Frame]
) -> Tuple[
    List["PlannedAction"], Optional[FrozenSet[float]], Optional[_CalibrationStamp]
]:
    """Parse + derive omitted add/remove window ids (W2, live-window
    coverage -- see :func:`_resolve_curation_window_ids`) + coalesce + frame-
    convert a curation file (or a batch of actions -- :func:`curation_source`)
    into the ready-to-run plan.

    Returns ``(plan, raw_targets, stamp)``: ``raw_targets`` is what the A5
    frame-mismatch advisory judges (see :func:`_parse_curation_call`).

    The single prologue :func:`apply_curation_impl`, :func:`_run_review_preview`,
    and :class:`ReviewSession`'s staged-plan comparison all call, so a change
    to window derivation or frame resolution reaches every curation-file
    entry point at once -- exactly the "two resolvers that can disagree"
    failure mode W2 has to avoid. (Originally factored out only for D3/D4, to
    avoid a third inlined copy; W2's window derivation is the reason the two
    pre-existing inlined copies were folded into calling this too.)

    Frame resolution happens before window derivation (derivation needs the
    raw frame, since window ranges are stored raw) and is reused for the
    final plan-level conversion, so it runs exactly once per call.
    """
    ops, resolved_frame, stamp, raw_targets = _parse_curation_call(path, source, frame)
    plan = _resolve_curation_ops(ops, path, frame=resolved_frame, stamp=stamp)
    return plan, raw_targets, stamp


def _parse_curation_call(
    path: str, source: CurationSource, frame: Optional[Frame]
) -> Tuple[
    ParsedCurationFile,
    Frame,
    Optional[_CalibrationStamp],
    Optional[FrozenSet[float]],
]:
    """The state-independent half of :func:`_resolve_curation_call`: parse
    the file and resolve its frame. Neither reads anything a curation
    decision changes (the frame comes from the calibration stamp), so a
    log-prefix apply runs this before it moves the file and defers only
    :func:`_resolve_curation_ops` to the replayed state.

    Returns ``(ops, frame, stamp, raw_targets)``, where ``frame`` is the
    frame *ops* are still in (converted by :func:`_resolve_curation_ops`)
    and ``raw_targets`` is what the A5 frame-mismatch advisory judges
    (:func:`_frame_mismatch_warnings`). A batch of actions (*source* a tuple)
    is converted per action here (:func:`_actions_to_ops`), so its ops come
    back raw and ``raw_targets`` holds its raw-frame actions' frequencies; a
    file's ops keep the file's one frame, so every candidate is judged
    (``None``) when it is raw and none (an empty set) when it is calibrated."""
    if isinstance(source, tuple):
        ops, stamp, raw_targets = _actions_to_ops(path, source, frame)
        return ops, "raw", stamp, raw_targets
    ops = parse_curation_file(source)
    resolved_frame: Frame = "raw"
    file_stamp: Optional[_CalibrationStamp] = None
    if _curation_ops_have_freq(ops):
        resolved_frame, file_stamp = _resolve_curation_frame(path, ops.header, frame)
    file_targets: Optional[FrozenSet[float]] = (
        None if resolved_frame == "raw" else frozenset()
    )
    return ops, resolved_frame, file_stamp, file_targets


def _resolve_curation_ops(
    ops: Sequence[CurationOp],
    path: str,
    *,
    frame: Frame,
    stamp: Optional[_CalibrationStamp],
    coverage: Optional[Sequence[FitWindowCoverage]] = None,
) -> List["PlannedAction"]:
    """The state-dependent half of :func:`_resolve_curation_call`: derive
    omitted window ids against ``coverage`` (the file's live windows unless
    given), coalesce, and convert every frequency to raw."""
    resolved_ops = _resolve_curation_window_ids(
        ops, path, frame=frame, stamp=stamp, coverage=coverage
    )
    plan = _resolve_curation_plan(resolved_ops)
    if any(_planned_action_has_freq(a) for a in plan):
        plan = [_planned_action_to_raw(a, frame=frame, stamp=stamp) for a in plan]
    return plan


# ---------------------------------------------------------------------------
# D3/D4: the amortized review session.
# ---------------------------------------------------------------------------


@dataclass
class _StagedPreview:
    """A curated, never-persisted state retained by a :class:`ReviewSession`
    immediately after ``review_preview`` (D4) -- so an immediately-following
    ``review_apply`` of the identical plan against an unchanged base can
    persist it directly instead of resolving and curating it a second time.
    Dropped the moment anything about the base -- or the requested plan --
    no longer matches (see ``ReviewSession.review_apply``).
    """

    fingerprint: _FitCtxFingerprint
    source_key: Union[str, Tuple[CurationAction, ...]]
    """The previewed request's source: the curation path as a string, or
    the tuple of actions (:func:`_session_source_key`)."""
    frame: Optional[Frame]
    resolved_plan: List["PlannedAction"]
    warnings: List[str]
    curated: _Curated
    created_windows: List[PlannedWindowResult] = field(default_factory=list)
    """The structure this preview's batch installed, carried so that
    persisting it as an apply reports exactly what the preview reported."""
    windows: Dict[int, AppliedWindowResult] = field(default_factory=dict)
    """This preview's per-window block, narrowed to the apply-side shape
    (:func:`_applied_window_from_preview`) and carried for the same reason
    ``created_windows`` is: the apply that persists this preview reports the
    preview's own numbers rather than a second derivation of them."""


def _session_source_key(
    source: CurationSource,
) -> Union[str, Tuple[CurationAction, ...]]:
    """What a :class:`ReviewSession` compares to decide that an apply asks
    for the plan it just previewed: the path as a string, or the actions
    themselves (frozen, so equal actions compare equal)."""
    return source if isinstance(source, tuple) else str(source)


class ReviewSession:
    """An amortized Stage 6 review session bound to one ``.ftmw`` file.

    Obtained from :meth:`Pipeline.review_session
    <ftmwpipeline.pipeline.Pipeline.review_session>` and used as a context
    manager; it is never constructed directly. The class is exported at the
    package top level (``from ftmwpipeline import ReviewSession``) so a
    caller can name the type of the object the ``with`` block yields.

    The session holds one shared active-FT fit context -- the ~420 ms
    reconstruction every fit-mutating Stage 6 verb otherwise rebuilds from
    scratch -- and reuses it across every verb issued through it:
    :meth:`review_edit`, :meth:`review_accept`, :meth:`review_create`,
    :meth:`review_undo`, :meth:`review_preview` and :meth:`review_apply`.
    Hosting the whole verb set rather than only the batch door is
    deliberate: an interactive single-window edit costs ~516 ms cold,
    essentially all of it that same setup, so a session that sped up only
    the batch would leave every interactive click paying full price.

    Warm-up -- building the shared context -- is synchronous and happens on
    entry, blocking. There is no thread inside the library, so a caller that
    wants the warm-up off its own critical path must arrange that itself.

    **Correctness never depends on the reuse.** Every verb re-validates a
    cheap on-disk fingerprint (~0.8 ms) before doing any work; on a mismatch
    -- a foreign writer touched the file, or this is the session's first use
    -- it rebuilds the shared context from scratch, identically to what the
    sessionless :class:`~ftmwpipeline.pipeline.Pipeline` methods do on every
    call. After each of the session's own writes the fingerprint is re-read
    from disk, never predicted, so a foreign writer landing in the very same
    instant is still caught on the session's next use. Results are identical
    with or without a session; only latency differs.

    The session retains roughly 26 MB of active-FT arrays for its lifetime,
    and that lifetime is entirely caller-controlled -- the ``with`` block, or
    an explicit :meth:`close`. There is no module-level cache, so a session
    that is never opened, or one that is closed, costs nothing beyond the
    object itself.

    Not thread-safe, holds no lock, and does not protect the file from a
    second writer: single-writer discipline per file is the caller's, exactly
    as it is for every sessionless verb.

    Usage::

        with Pipeline.open("exp_2638.ftmw").review_session() as session:
            session.review_edit(12, remove=["uid:41"], frame="raw")
            preview = session.review_preview("edits.csv")
            session.review_apply("edits.csv")  # persists the preview's result
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        self._shared: Optional[_SharedFitCtx] = None
        self._fingerprint: Optional[_FitCtxFingerprint] = None
        self._staged: Optional[_StagedPreview] = None
        self._pending_base_changed = False
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "ReviewSession":
        self._shared = _build_shared_fit_ctx(self._path)
        self._fingerprint = _compute_fit_ctx_fingerprint(self._path)
        self._closed = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the retained shared context. Idempotent.

        Called automatically when the ``with`` block exits. Once closed, any
        further verb on this session raises ``ValueError``; open a new
        session to continue.
        """
        self._shared = None
        self._fingerprint = None
        self._staged = None
        self._pending_base_changed = False
        self._closed = True

    # -- internal freshness / staging bookkeeping ----------------------------

    def _require_open(self) -> _SharedFitCtx:
        if self._closed or self._shared is None:
            raise ValueError(
                "review session is closed; use "
                "'with pipeline.review_session() as session:' and call verbs "
                "only inside the block"
            )
        return self._shared

    def _sync(self) -> _SharedFitCtx:
        """Validate the cached shared context against a freshly-read
        fingerprint; rebuild (and drop any staged preview) on a mismatch.
        Called before every verb -- the sole gate that keeps correctness
        independent of whatever ``self._shared`` currently holds. A forced
        rebuild here is byte-for-byte the same rebuild a sessionless caller
        gets automatically on every call.
        """
        self._require_open()
        live = _compute_fit_ctx_fingerprint(self._path)
        if live != self._fingerprint:
            self._shared = _build_shared_fit_ctx(self._path)
            self._fingerprint = live
            self._drop_staged(base_changed=True)
        assert self._shared is not None
        return self._shared

    def _drop_staged(self, *, base_changed: bool) -> None:
        if self._staged is not None:
            self._staged = None
            if base_changed:
                self._pending_base_changed = True

    def _resync_after_write(self) -> None:
        """Re-read (never predict) the fingerprint after one of this
        session's own writes. A session-issued edit never touches anything
        :class:`_SharedFitCtx` derives from (settings, tau calibration, the
        base window plan, the calibration stamp), so this refreshes the
        stored baseline without forcing a rebuild -- but it is read from
        disk, not computed from what was just written, so a foreign writer
        that landed in the very same instant is still caught on the NEXT
        verb call's :meth:`_sync`.

        The session's working-fit cache (S5) needs nothing from here: it
        stamps itself against ``/stage5_fitting`` at the moment of the write
        and re-checks that stamp on its own -- see :class:`_FitCache`.
        """
        self._fingerprint = _compute_fit_ctx_fingerprint(self._path)

    # -- single-window verbs --------------------------------------------------

    def review_edit(
        self,
        window_id: Optional[int] = None,
        *,
        add: Sequence[Union[float, str]] = (),
        remove: Sequence[Union[float, str]] = (),
        frame: Optional[Frame] = None,
    ) -> RefitWindowResult:
        """Re-fit one window with ``add`` / ``remove`` edits.

        Identical in arguments, return value and persisted effect to
        :meth:`Pipeline.review_edit
        <ftmwpipeline.pipeline.Pipeline.review_edit>`, which documents the
        full contract; the only difference is that this session's shared fit
        context is reused -- validated fresh first -- rather than rebuilt.
        """
        _refuse_bare_edit(add, remove)
        # The verb's transaction opens before the freshness check, so its
        # write_conflict stat predates every input the edit is built from.
        with atomic_write(self._path):
            shared = self._sync()
            self._drop_staged(base_changed=True)
            result = refit_window_impl(
                self._path,
                window_id,
                add=add,
                remove=remove,
                frame=frame,
                _shared=shared,
            )
        self._resync_after_write()
        return result

    def review_accept(
        self,
        window_id: int,
        *,
        candidate_freq: Optional[float] = None,
        frame: Optional[Frame] = None,
    ) -> Optional[RefitWindowResult]:
        """Accept a window as reviewed, or revive a named ledger candidate.

        Identical in arguments, return value and persisted effect to
        :meth:`Pipeline.review_accept
        <ftmwpipeline.pipeline.Pipeline.review_accept>`, reusing this
        session's shared fit context.
        """
        with atomic_write(self._path):  # before the check: see review_edit
            shared = self._sync()
            self._drop_staged(base_changed=True)
            result = review_accept_impl(
                self._path,
                window_id,
                candidate_freq=candidate_freq,
                frame=frame,
                _shared=shared,
            )
        self._resync_after_write()
        return result

    def review_create(
        self,
        anchor_mhz: float,
        *,
        frame: Optional[Frame] = None,
    ) -> CreateWindowResult:
        """Install a fit window for a line no window covers.

        Identical in arguments, return value and persisted effect to
        :meth:`Pipeline.review_create
        <ftmwpipeline.pipeline.Pipeline.review_create>`, reusing this
        session's shared fit context.
        """
        with atomic_write(self._path):  # before the check: see review_edit
            shared = self._sync()
            self._drop_staged(base_changed=True)
            result = create_window_impl(
                self._path,
                anchor_mhz,
                frame=frame,
                _shared=shared,
            )
        self._resync_after_write()
        return result

    def review_undo(
        self,
        ids: Sequence[int],
        *,
        dry_run: bool = False,
    ) -> UndoResult:
        """Roll recorded decisions back by id, replaying the survivors.

        Identical in arguments, return value and persisted effect to
        :meth:`Pipeline.review_undo
        <ftmwpipeline.pipeline.Pipeline.review_undo>`, reusing this
        session's shared fit context.
        """
        with atomic_write(self._path):  # before the check: see review_edit
            shared = self._sync()
            if not dry_run:
                self._drop_staged(base_changed=True)
            result = review_undo_impl(self._path, ids, dry_run=dry_run, _shared=shared)
        if not dry_run:
            self._resync_after_write()
        return result

    # -- batch door: preview / apply, with D4's staged reuse -----------------

    def review_preview(
        self,
        curation_path: Optional[Union[str, Path]] = None,
        *,
        actions: Optional[Sequence[Union[CurationAction, Mapping[str, Any]]]] = None,
        frame: Optional[Frame] = None,
    ) -> ReviewPreviewResult:
        """Run a curation file's plan to completion in memory and report the
        fitted outcome, writing nothing.

        Identical in arguments and return value to
        :meth:`Pipeline.review_preview
        <ftmwpipeline.pipeline.Pipeline.review_preview>`, with one addition:
        the finished, cascaded, never-persisted outcome is *staged* on this
        session, so an immediately-following :meth:`review_apply` of the
        identical plan against an unchanged base can persist it directly
        rather than computing it a second time. See that method.
        """
        source = curation_source(curation_path, actions)
        # Refused before the session builds anything, as its apply is.
        _require_engine_file(self._path)
        shared = self._sync()
        # A fresh preview supersedes any earlier drift note.
        self._pending_base_changed = False
        with _caller_frame_ids():
            run = _run_review_preview(self._path, source, frame=frame, shared=shared)
        if run.curated is not None:
            assert self._fingerprint is not None
            self._staged = _StagedPreview(
                fingerprint=self._fingerprint,
                source_key=_session_source_key(source),
                frame=frame,
                resolved_plan=list(run.result.plan),
                warnings=list(run.result.warnings),
                curated=run.curated,
                created_windows=list(run.result.created_windows),
                windows={
                    wid: _applied_window_from_preview(pw)
                    for wid, pw in run.result.windows.items()
                },
            )
        else:
            self._staged = None
        return run.result

    def _persist_staged(self, staged: _StagedPreview) -> None:
        """The write half of D4's staged-reuse apply: admit the write and
        take the undo baseline exactly as a live apply would (via
        :func:`_open_batch` -- the only function structurally permitted to
        take that baseline), then persist the ALREADY-curated state from the
        staged preview -- never recomputing it, so the persisted bytes are
        guaranteed to be exactly what the preview showed rather than a second
        computation trusted to agree with the first. The preview was computed
        by the same :func:`_curate` the apply runs, epoch gate included, so
        a refitting preview was gated when it was made, against the same
        base.
        """
        # Called inside review_apply's transaction.
        _require_engine_file(self._path)
        with atomic_write(self._path):
            _finish_batch(
                replace(
                    staged.curated,
                    baseline_taken=_open_batch(
                        self._path, lineage_id=staged.curated.pending_lineage
                    ),
                ),
                self._path,
            )

    def review_apply(
        self,
        curation_path: Optional[Union[str, Path]] = None,
        *,
        actions: Optional[Sequence[Union[CurationAction, Mapping[str, Any]]]] = None,
        frame: Optional[Frame] = None,
        log_prefix: Optional[int] = None,
    ) -> CurationApplyResult:
        """Apply a curation file as one batch.

        Identical in arguments, return value and persisted effect to
        :meth:`Pipeline.review_apply
        <ftmwpipeline.pipeline.Pipeline.review_apply>`, with one addition:
        when an immediately-preceding :meth:`review_preview` staged the
        identical plan -- same curation file, same ``frame``, same resolved
        actions -- against a base that has not moved since, this persists
        that already-computed result directly instead of re-running the
        appliers, the cascade and the review derivation. The bytes persisted
        are then guaranteed to be exactly the ones the preview showed rather
        than a second computation trusted to agree with the first. Any
        mismatch -- a different plan, or a base that moved -- falls back to a
        full, ordinary apply, identical to the sessionless one.

        ``base_changed`` on the result is ``True`` only when a staged preview
        existed but had to be dropped because the base moved out from under
        it (a foreign write, or another mutating verb issued on this session
        in between) -- never merely because no preview preceded this call.

        ``log_prefix`` is :meth:`Pipeline.review_apply
        <ftmwpipeline.pipeline.Pipeline.review_apply>`'s: a staged preview
        was computed against the log as it stood, so none is reused when a
        prefix is given.
        """
        source = curation_source(curation_path, actions)
        # The transaction opens before the freshness check, so its
        # write_conflict stat predates every input this apply is built from
        # (the fingerprint, the staged preview, the shared context).
        with _caller_frame_ids(), atomic_write(self._path):
            result = self._apply(source, frame=frame, log_prefix=log_prefix)
        self._resync_after_write()
        return result

    def _apply(
        self,
        source: CurationSource,
        *,
        frame: Optional[Frame],
        log_prefix: Optional[int],
    ) -> CurationApplyResult:
        """The body of :meth:`review_apply`, inside its transaction."""
        # Before the staged-reuse path resolves the request: a pre-engine file
        # is refused before anything is resolved, fitted or written.
        _require_engine_file(self._path)
        shared = self._sync()
        base_changed = self._pending_base_changed
        self._pending_base_changed = False

        staged = self._staged
        if staged is not None:
            same_request = (
                log_prefix is None
                and staged.source_key == _session_source_key(source)
                and staged.frame == frame
                and staged.fingerprint == self._fingerprint
            )
            if same_request:
                plan, _, _ = _resolve_curation_call(self._path, source, frame)
                if plan == staged.resolved_plan:
                    self._persist_staged(staged)
                    self._staged = None
                    return CurationApplyResult(
                        plan=plan,
                        warnings=staged.warnings,
                        applied=len(plan),
                        dry_run=False,
                        created_windows=list(staged.created_windows),
                        windows=dict(staged.windows),
                        base_changed=base_changed,
                    )
            # Staged, but it does not match this call -- irrelevant now.
            self._staged = None

        result = apply_curation_impl(
            self._path,
            None if isinstance(source, tuple) else source,
            actions=source if isinstance(source, tuple) else None,
            frame=frame,
            log_prefix=log_prefix,
            _shared=shared,
        )
        if base_changed:
            result = replace(result, base_changed=True)
        return result
