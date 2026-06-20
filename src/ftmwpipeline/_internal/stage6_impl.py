"""
Shared implementation for Stage 6: review, candidate ledger, and user-directed
single-window refit.

Pass 1 — candidate ledger derivation (``review show``/``--candidates``) and
the single-window refit engine (``review edit``).

The ledger is a pure function of the already-persisted Stage 5 audit trail
(``FittingResult.audit_trail``) and rescue events
(``FittingResult.rescue_events``).  It is derived on demand; no re-fitting
and no writes to the Stage 5 group.

The refit engine (``refit_window_impl``) re-fits a single window using the
production NLS primitives, starting from the persisted peaks as seeds.  It
supports add/remove edits with protected/forbidden immunity so cleanup and
rescue cannot undo human decisions.

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..core.data_structures import (
    AttentionReason,
    AuditStep,
    DecisionLogEntry,
    FinalPeak,
    FinalProducts,
    FittedPeak,
    FittingResult,
    FrequencyCalibration,
    LedgerCandidate,
    RescueCandidateInfo,
    RescueRoundInfo,
    Sideband,
    SpectrumFit,
    Stage6Review,
    WindowReviewStatus,
)
from ..fitting.peak_model import ModelPeak
from ..fitting.peak_model import molecular_frequency as _molecular_frequency
from ..fitting.peak_model import sideband_sign
from ..fitting.validation import DEFAULT_CHI2R_NOISE_FLOOR, DEFAULT_SHAPE_ERROR_KAPPA
from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
from ..io.frequency_calibration_serialization import (
    load_frequency_calibration_from_hdf5,
    save_frequency_calibration_to_hdf5,
)
from ..io.stage6_review_serialization import (
    load_stage6_review_from_file,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage2_impl import _update_stage_completion

if TYPE_CHECKING:  # annotation-only imports (PEP 563 lazy)
    from ..core.data_structures import FitWindow
    from ..core.stage_fit_settings import StageFitSettings
    from ..fitting.peak_model import PeakShape
    from .stage5_impl import Stage5FitContext

logger = logging.getLogger(__name__)


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

# If a candidate's evidence is within this factor of the accept gate it
# passes the bar even when its raw SNR is below DEFAULT_DISPLAY_BAR.
_NEAR_GATE_FACTOR: float = 10.0

# Fallback amplitude-VIF attention threshold used when the file carries no
# resolved ``peak_survival.vif_attention_threshold`` (mirrors the hard default
# in :class:`~ftmwpipeline.core.stage_fit_settings.PeakSurvivalSubSettings`).
# A fitted amplitude whose variance-inflation factor ``(amp_err / amp) * snr``
# clears this is non-identifiable with a neighbour (degenerate split) even
# though it survived the end-of-Stage-5 collapse (which fires only on the much
# higher ``vif_collapse_threshold`` at sub-half-resolution separation).
DEFAULT_VIF_ATTENTION_THRESHOLD: float = 4.0

# Deduplicate candidates whose molecular frequencies are within this window.
_DEDUP_TOL_MHZ: float = 0.02  # 20 kHz; roughly half an active-FT bin at 13 µs

# Shape-error candidate filter. A revival "candidate" that sits within
# SHAPE_ERROR_MAX_SEP_RES resolution elements of a fitted peak whose SNR is
# large enough that the candidate's evidence is only a small fraction
# (< SHAPE_ERROR_EVIDENCE_FRACTION) of it is a lineshape sidelobe of that
# brighter line, not a missed line -- the residual-SNR currency is dominated by
# lineshape mismodeling in bright/dense windows. Measured on the 2638 truth set:
# this cleanly tags 22/29 rescue-round candidates as shape error and 0/18
# conservative-loop "tentative" candidates. Same normalize-by-the-local-strong-
# line idea as the amplitude VIF. Both are module constants (the attention layer
# is advisory and recomputed each ``review run``, so no persisted knob).
SHAPE_ERROR_MAX_SEP_RES: float = 2.0
SHAPE_ERROR_EVIDENCE_FRACTION: float = 0.25


# ---------------------------------------------------------------------------
# Revivable decision labels from the conservative add-loop
# ---------------------------------------------------------------------------

_REVIVABLE_DECISIONS = frozenset({"reject", "tentative"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sideband_from_value(value: Union[str, Sideband]) -> Sideband:
    if isinstance(value, Sideband):
        return value
    key = str(value).strip().lower()
    if key in ("lower", "lsb"):
        return Sideband.LOWER
    if key in ("upper", "usb"):
        return Sideband.UPPER
    raise ValueError(f"unknown sideband: {value!r}")


def _to_molecular(offset_mhz: float, center_mhz: float, sideband: Sideband) -> float:
    """Convert a single baseband offset to molecular MHz."""
    arr = np.array([offset_mhz])
    result = _molecular_frequency(arr, center_mhz, sideband)
    return float(result[0])


def _window_center(fit: FittingResult) -> Optional[float]:
    """Return the molecular centre of ``fit``'s window, or ``None``."""
    if fit.window is not None and fit.window.freq_range is not None:
        lo, hi = fit.window.freq_range
        return (lo + hi) / 2.0
    return None


def _auto_merged_window_ids(spectrum_fit: SpectrumFit) -> set:
    """Window ids the end-of-Stage-5 VIF merge touched (from the diagnostic)."""
    vc = spectrum_fit.diagnostics.get("vif_collapse", {}) if spectrum_fit else {}
    return {
        int(c["window_id"])
        for c in vc.get("collapses", [])
        if c.get("window_id") is not None and int(c["window_id"]) >= 0
    }


def _resolve_vif_attention_threshold(path: str) -> float:
    """Resolve ``peak_survival.vif_attention_threshold`` for *path*.

    Reads the knob from the file's persisted settings (persisted layer over the
    hard defaults), so the attention surface picks up exactly where the
    end-of-Stage-5 collapse left off. Falls back to the module default when the
    settings cannot be read.
    """
    from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
    from ..io.stage_fit_settings_serialization import load_stage_fit_settings_from_h5

    vif = DEFAULT_VIF_ATTENTION_THRESHOLD
    try:
        resolved = resolve_stage_fit_settings(
            persisted=load_stage_fit_settings_from_h5(path)
        )
        if resolved.peak_survival.vif_attention_threshold is not None:
            vif = float(resolved.peak_survival.vif_attention_threshold)
    except Exception:
        pass
    return vif


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
    ``_DEDUP_TOL_MHZ``, applies the display ``bar``, and returns a list of
    :class:`~ftmwpipeline.core.data_structures.LedgerCandidate` sorted by
    molecular frequency.

    Parameters
    ----------
    fitting_result :
        The per-window :class:`~ftmwpipeline.core.data_structures.FittingResult`.
    center_mhz :
        Molecular centre of the fit window (midpoint of its ``freq_range``).
    sideband :
        Pipeline sideband (``Sideband.UPPER`` or ``Sideband.LOWER``).
    bar :
        Display SNR / evidence bar.  Candidates below it are dropped.
    res_element_mhz :
        Fourier resolution element (``1 / T_active`` MHz). When given, the
        shape-error filter runs: a candidate within
        :data:`SHAPE_ERROR_MAX_SEP_RES` resolution elements of a fitted peak
        whose ``snr * SHAPE_ERROR_EVIDENCE_FRACTION`` is at least the
        candidate's evidence is a lineshape sidelobe of that brighter line and
        is excluded.  ``None`` disables the filter (legacy behaviour).

    Returns
    -------
    list of LedgerCandidate
        Sorted by ``frequency_mhz``.
    """
    window_id: int = (
        fitting_result.window_id if fitting_result.window_id is not None else -1
    )

    raw: List[Dict] = []
    raw.extend(_audit_step_candidates(fitting_result.audit_trail, center_mhz, sideband))
    raw.extend(
        _rescue_round_candidates(fitting_result.rescue_events, center_mhz, sideband)
    )

    merged = _dedup_and_merge(raw, _DEDUP_TOL_MHZ)

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
            if np.min(np.abs(fitted_freqs - c["freq_mhz"])) > _DEDUP_TOL_MHZ
        ]
    else:
        not_installed = merged

    # Shape-error filter: drop a candidate that is a lineshape sidelobe of a
    # brighter nearby fitted line (residual-SNR evidence is dominated by
    # lineshape mismodeling in bright/dense windows). See the module constants.
    if res_element_mhz is not None and res_element_mhz > 0.0 and fitted_freqs.size:
        max_sep_mhz = SHAPE_ERROR_MAX_SEP_RES * res_element_mhz
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
            near = np.abs(fitted_freqs - c["freq_mhz"]) <= max_sep_mhz
            if not near.any():
                return False
            return bool(
                (
                    fitted_snr[near] * SHAPE_ERROR_EVIDENCE_FRACTION
                    >= float(c["evidence"])
                ).any()
            )

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
        standalone behaviour (self-load from ``file_path``) is unchanged, so the
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
        with h5py.File(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                raise ValueError(
                    "No Stage 5 fit found in this file. Run 'fit run' first."
                )
            spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        fid = load_fid_from_pipeline_impl(path)
        sideband = _sideband_from_value(fid.sideband)
    else:
        sideband = _sideband_from_value(sideband)

    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    res_element_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else None

    window_fits = spectrum_fit.window_fits
    if window_id is not None:
        window_fits = [wf for wf in window_fits if wf.window_id == window_id]
        if not window_fits:
            raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")

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
    from .stage5_impl import amplitude_vif

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
        raise ValueError(f"unknown rank metric {by!r}; choose one of: {valid}")
    lower_is_worse = RANK_METRICS[metric][1]

    path = str(file_path)
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)
    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    res_element_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else None
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
    chi2r_before : float
        Reduced chi-squared before the refit.
    chi2r_after : float
        Reduced chi-squared after the refit.
    fitted_peaks : list of FittedPeak
        The new per-window fitted peaks (already persisted).
    """

    window_id: int
    n_peaks_before: int
    n_peaks_after: int
    chi2r_before: float
    chi2r_after: float
    fitted_peaks: List[FittedPeak] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-window refit engine
# ---------------------------------------------------------------------------

# Tolerance for matching add/remove frequency requests to fitted peaks or
# ledger candidates (in MHz).  Half the dedup tolerance (10 kHz) is tight
# enough to snap unambiguously to one peak while forgiving coarse user input.
_REFIT_SNAP_TOL_MHZ: float = 0.05


def _parse_complex_amplitude(value: object) -> complex:
    """Parse a complex amplitude stored as ``str(complex)`` in JSON.

    ``result_conversion.py`` stores ``frozen.model_peak.amplitude`` (a real
    float) via ``json.dumps(..., default=str)``, which calls ``repr(v)`` on
    non-serialisable values.  For a real float the repr is just the float
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
        }

    The offset in the dependent window's baseband frame is derived from the
    molecular frequency and the window centre (same convention as
    :func:`~ftmwpipeline.fitting.plan_execution.evaluate_fixed_contributor`).
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
        model_peak = ModelPeak(
            amplitude=amplitude,
            offset_mhz=offset_mhz,
            phase=phase,
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
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    add_seeds: Optional[List[ModelPeak]] = None,
    add_origin: str = "user",
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> FittingResult:
    """In-memory single-window refit core (no file I/O, no spur replay, no
    decision recording).

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
    catalogue replayed, calls this core, then persists and records decisions).
    The Stage 5 peak-survival pass routes through it too, holding ``fit_ctx`` /
    the plan / ``resolved`` live from ``fit_peaks_impl``. ``add_origin`` stamps
    the origin of added peaks: ``"user"`` for a user edit (the default, immune
    to later auto-prune/cleanup), ``"auto"`` for an automatic add such as the
    VIF-collapse merged line (a normal fitted peak, not a human decision).

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

    # --- Identify thawed lines in this window ---------------------------------
    # A thaw is accepted when a primary-window frozen contributor is co-fit
    # jointly with the dependent window and the residual edge coherence improves.
    # The thawed line then appears in this window's fitted_peaks (as a free peak)
    # but is owned by its primary window.  Its persisted value is the joint-fit
    # optimum; re-fitting it with only ONE window's data would relax it off that
    # joint optimum (worst observed: 28 kHz / 28% drift).  Decision: freeze it.
    #
    # Collect the molecular frequencies of all *accepted* thaw contributors for
    # this (dependent) window.  Ignore NaN frequencies (no-op records).
    thawed_freqs: List[float] = []
    for te in wf.thaw_events:
        if te.accepted and not math.isnan(te.contributor_frequency_mhz):
            thawed_freqs.append(float(te.contributor_frequency_mhz))

    def _is_thawed(freq_mhz: float) -> bool:
        """True when ``freq_mhz`` matches a thawed contributor within snap_tol."""
        return any(abs(freq_mhz - tf) <= snap_tol_mhz for tf in thawed_freqs)

    # --- Reconstruct frozen background from persisted fixed_parameters -----
    frozen_peaks = _reconstruct_frozen_peaks(wf.fixed_parameters, center_mhz, sideband)
    # The frozen background uses the window's persisted tau as the dependent
    # tau (exactly the convention in evaluate_fixed_contributor).
    tau_persisted = float(
        wf.shared_parameters.get("tau_us", {}).get("value", acquisition_us / 3.0)
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
    # fit_tau legitimately stays None in _HARD_DEFAULTS (the None means "use
    # the per-window decision from the original fit").  Mirror the same None
    # fallback that fit_peaks_impl uses.
    tau_was_fit = bool(wf.shared_parameters.get("tau_us", {}).get("fitted", True))
    fit_tau_for_window: bool = (
        tau_was_fit if resolved.tau.fit_tau is None else bool(resolved.tau.fit_tau)
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
        fw_kwargs["baseline_order"] = int(_qm["baseline_order"])
        _bscale = _qm.get("baseline_offset_scale")
        if _bscale:
            fw_kwargs["baseline_offset_scale"] = float(_bscale)

    # --- Build seed ModelPeak list from persisted fitted_peaks + edits -----
    s = sideband_sign(sideband)

    # Partition fitted_peaks into free seeds (this window's own peaks) and
    # thawed-line hold-outs.  Thawed lines are owned by their primary window;
    # re-fitting them with one window's data relaxes them off the joint optimum.
    # Strategy: add each thawed peak to frozen_peaks (so it stays in the
    # frozen background the free-peak NLS fits against) and remember the
    # FittedPeak verbatim for re-insertion after the NLS.
    #
    # Note: ``frozen_peaks`` was reconstructed from ``fixed_parameters`` above
    # (the original Stage 5 frozen contributors).  A thawed line was *removed*
    # from ``fixed_parameters`` when it was accepted — it is not there.  We
    # must add it back explicitly.
    thawed_held_peaks: List[FittedPeak] = []  # verbatim re-append after NLS

    seed_peaks_with_origin: List[Tuple[ModelPeak, str]] = []
    for fp in wf.fitted_peaks:
        freq_mhz = float(fp.frequency_mhz)
        offset = float(s * (freq_mhz - center_mhz))
        if _is_thawed(freq_mhz):
            # Hold this peak out of the free NLS.  Add it to the frozen
            # background so the window's own peaks fit against it correctly.
            # Use a sentinel primary_window_id of -1 (we don't need it here;
            # the frozen model only needs amplitude/offset/phase and tau).
            mp = ModelPeak(
                amplitude=float(fp.amplitude),
                offset_mhz=offset,
                phase=float(fp.phase) if fp.phase is not None else 0.0,
            )
            frozen_peaks.append(
                FrozenPeak(
                    peak_index=int(fp.peak_id) if fp.peak_id is not None else -1,
                    primary_window_id=-1,
                    model_peak=mp,
                    frequency_mhz=freq_mhz,
                    freeze_eligible=False,
                    edge_free=False,
                )
            )
            thawed_held_peaks.append(fp)
        else:
            mp = ModelPeak(
                amplitude=float(fp.amplitude),
                offset_mhz=offset,
                phase=float(fp.phase) if fp.phase is not None else 0.0,
            )
            seed_peaks_with_origin.append((mp, fp.origin))

    # Apply "remove" edits: drop seeds closest to remove frequencies.
    # A "remove" on a thawed line drops it entirely (not frozen): the user
    # decided to remove it, so it is neither re-fit nor re-appended.
    forbidden_offsets: List[float] = []
    for rm_freq in remove:
        rm_offset = float(s * (float(rm_freq) - center_mhz))

        # Check if this removal targets a thawed held-out peak.
        thawed_match_idx: Optional[int] = None
        for ti, theld in enumerate(thawed_held_peaks):
            if abs(float(theld.frequency_mhz) - float(rm_freq)) <= snap_tol_mhz:
                thawed_match_idx = ti
                break
        if thawed_match_idx is not None:
            # Drop the thawed peak from the frozen background and the hold-out
            # list.  Find the matching FrozenPeak by frequency and remove it.
            removed_fp = thawed_held_peaks.pop(thawed_match_idx)
            frozen_peaks = [
                fp
                for fp in frozen_peaks
                if abs(fp.frequency_mhz - float(removed_fp.frequency_mhz))
                > snap_tol_mhz
            ]
            forbidden_offsets.append(rm_offset)
            continue

        if not seed_peaks_with_origin:
            raise ValueError(
                f"remove={rm_freq:.4f} MHz: no fitted peaks in window "
                f"{window_id} to remove"
            )
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
            raise ValueError(
                f"remove={rm_freq:.4f} MHz: no fitted peak within "
                f"{snap_tol_mhz:.3f} MHz (closest is at "
                f"{closest_mol_freq:.4f} MHz, "
                f"distance={closest_dist_val:.4f} MHz)"
            )
        # Record the exact fitted offset as forbidden (rescue must not re-add it).
        removed_offset = seed_peaks_with_origin.pop(closest_idx)[0].offset_mhz
        forbidden_offsets.append(removed_offset)

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
    wf_sideband = _sideband_from_value(sideband)

    for i, add_freq in enumerate(add):
        add_offset = float(s * (float(add_freq) - center_mhz))
        if explicit_seeds:
            mp = explicit_seeds[i]
        else:
            # Snap to the nearest ledger candidate if within tolerance.
            ledger = derive_candidate_ledger(
                wf,
                center_mhz=center_for_ledger,
                sideband=wf_sideband,
                bar=0.0,  # all candidates; the user has decided to add this peak
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
                )
        seed_peaks_with_origin.append((mp, add_origin))
        protected_offsets.append(mp.offset_mhz)

    # Extract final seed list in offset order.
    final_seeds = [mp for mp, _ in seed_peaks_with_origin]
    origin_flags = [orig for _, orig in seed_peaks_with_origin]

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

    # Stamp user-origin on peaks the caller added (and on merge / split
    # products, which seed through the same path).  A refit is NLS-only -- it
    # neither adds nor drops peaks -- and ``window_outcome_to_fitting_result``
    # preserves the fit's peak order, so the i-th output peak is the i-th seed:
    # assign origin by POSITION.  Matching by frequency is unsafe here because a
    # merge / split product can converge well beyond ``snap_tol_mhz`` from its
    # seed.  Fall back to nearest-frequency matching only if the counts ever
    # diverge (they should not on the NLS-only path).
    if len(new_wf.fitted_peaks) == len(origin_flags):
        for fp, orig in zip(new_wf.fitted_peaks, origin_flags):
            if orig == "user":
                fp.origin = "user"
    else:
        user_freq_mhz = [
            float(center_mhz + s * mp.offset_mhz)
            for mp, orig in zip(final_seeds, origin_flags)
            if orig == "user"
        ]
        for fp in new_wf.fitted_peaks:
            for uf in user_freq_mhz:
                if abs(float(fp.frequency_mhz) - uf) <= snap_tol_mhz:
                    fp.origin = "user"
                    break

    # --- Re-insert thawed lines verbatim ------------------------------------
    # Thawed lines were held out of the NLS and frozen into the background so
    # the window's own peaks converged correctly against them.  Now re-attach
    # them to the output UNCHANGED (same frequency / amplitude / phase / errors
    # / origin as the persisted FittedPeak).  Their model contribution is
    # already accounted for in ``full_fitted`` / ``full_residual`` (they were
    # part of ``background``), so χ²ᵣ in the result is consistent.
    if thawed_held_peaks:
        new_wf.fitted_peaks = sorted(
            new_wf.fitted_peaks + thawed_held_peaks,
            key=lambda fp2: float(fp2.frequency_mhz),
        )
        # The NLS ran with only the non-thawed peaks free; the covariance only
        # covers those K_free params.  After re-inserting the thawed peaks the
        # fitted_peaks list grows, so the covariance labels no longer match the
        # full peak count.  Clear it rather than persist a partial / mislabelled
        # matrix — the per-peak amplitude_error / frequency_error / phase_error
        # fields already carry the per-parameter uncertainties.
        new_wf.covariance = None
        new_wf.covariance_param_labels = None
        logger.debug(
            "Stage 6 refit window %d: re-inserted %d thawed peak(s) verbatim",
            window_id,
            len(thawed_held_peaks),
        )

    # Carry the construction provenance forward. A refit replays/edits the
    # window rather than rebuilding it, so the original conservative add-one
    # ``audit_trail`` and ``rescue_events`` stay the truthful record of how the
    # peak set arose; the joint-refit core would otherwise leave them empty
    # (e.g. an auto-merged VIF-collapse window, which is why such windows showed
    # no add-one history in the report).
    new_wf.audit_trail = list(wf.audit_trail or [])
    new_wf.rescue_events = list(getattr(wf, "rescue_events", []) or [])

    return new_wf


def refit_window_impl(
    file_path: Union[Path, str],
    window_id: int,
    *,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    add_seeds: Optional[List[ModelPeak]] = None,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
    _skip_decision_recording: bool = False,
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
    automatic discovery already ran; the persisted peaks are its result).  The
    result replaces ONLY that window's entry in the persisted ``SpectrumFit``;
    all other windows are untouched (no cascade, no thaw, no replan).

    Identity refit (``add=()`` and ``remove=()``) reproduces the persisted fit
    to ~1e-5 MHz on a window whose peaks are a single-window optimum.  A window
    that was subject to a **thaw co-fit** holds peaks at a *two-window* joint
    optimum; a single-window NLS relaxes those toward the one-window optimum, so
    such windows reproduce only to ~kHz (the neighbour's data is intentionally
    not re-included).  This is an inherent property of thaw, not a defect.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The ``FitWindow.window_id`` / ``FittingResult.window_id`` to refit.
    add :
        Molecular frequencies (MHz) of peaks to add.  Each is snapped to the
        nearest ledger candidate within ``snap_tol_mhz`` (to reuse the
        recorded seed offset/amplitude) or seeded fresh at the given
        frequency.  User-added peaks carry ``origin="user"``.
    remove :
        Molecular frequencies (MHz) of fitted peaks to remove.  Each is
        matched to the nearest fitted peak within ``snap_tol_mhz`` and dropped
        from the seed set before the NLS.
    add_seeds :
        Optional explicit :class:`~ftmwpipeline.fitting.peak_model.ModelPeak`
        seeds (in the window's baseband-offset frame) for the added peaks.
        When given, ``len(add_seeds)`` must equal ``len(add)`` and each seed
        provides the starting amplitude / offset / phase for the corresponding
        ``add`` frequency (overrides the ledger-candidate or default seed).
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping to an existing peak or
        ledger candidate.  Defaults to :data:`_REFIT_SNAP_TOL_MHZ` (50 kHz).

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks.

    Raises
    ------
    ValueError
        When Stage 5 has not been run, the ``window_id`` is not found,
        ``len(add_seeds) != len(add)``, or any ``remove`` frequency does not
        match a fitted peak within ``snap_tol_mhz``.
    """
    from ..core.stage_fit_settings import ShapeSpec, StageFitSettings
    from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
    from ..fitting.peak_model import PeakShape
    from ..io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )
    from ..io.stage_fit_settings_serialization import (
        load_stage_fit_settings_from_h5,
        read_recommended_clock_sources,
        read_stage2b_recommended_shape,
    )
    from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
    from .stage3_impl import load_peaks_impl
    from .stage4_impl import load_windows_impl
    from .stage5_impl import (
        Stage5FitContext,
        _resolve_tau_calibration_for_fit,
        build_stage5_fit_context,
    )

    path = str(file_path)

    if add_seeds is not None and len(add_seeds) != len(add):
        raise ValueError(
            f"len(add_seeds)={len(add_seeds)} must equal len(add)={len(add)}"
        )

    # Snapshot the automatic fit before the first edit mutates it in place, so
    # 'review undo' can restore it and replay the surviving decisions.
    _snapshot_stage5_baseline(path)

    # --- Load persisted Stage 5 fit ----------------------------------------
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    # Locate the target window's FittingResult.
    wf_list = [wf for wf in spectrum_fit.window_fits if wf.window_id == window_id]
    if not wf_list:
        raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")
    wf: FittingResult = wf_list[0]
    chi2r_before = float(wf.reduced_chi2)
    n_peaks_before = len(wf.fitted_peaks)

    # --- Load Stage 4 WindowPlan and locate the FitWindow ------------------
    plan_result = load_windows_impl(path)
    plan = plan_result["plan"]
    fit_window_map = {w.window_id: w for w in plan.windows}
    if window_id not in fit_window_map:
        raise KeyError(
            f"window_id={window_id} not found in the Stage 4 WindowPlan. "
            "Stage 4 may have been re-run and changed the window geometry."
        )
    fit_win = fit_window_map[window_id]

    # --- Load Stage 3 peaks (for peak_id matching in result conversion) ----
    peaks_loaded = load_peaks_impl(path)
    peak_frequencies_mhz = [float(p.frequency) for p in peaks_loaded["peaks"]]

    # --- Resolve StageFitSettings from persisted layer + recommended -------
    # Use the PERSISTED settings as the authoritative source so the refit
    # operates under the same settings the original fit used.  No explicit
    # overrides: the user's edit is at the peak level (add/remove), not the
    # settings level.
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
        explicit=StageFitSettings(),  # no explicit overrides
        preset=None,
        persisted=persisted_settings,
        recommended=recommended_settings,
    )
    assert resolved.shape is not None
    shape_enum = resolved.shape.kind

    # --- Stage 2b calibration (shape-routed, same logic as fit_peaks_impl) -
    persisted_cal = None
    if shape_enum is PeakShape.GAUSSIAN:
        if tau_calibration_present(path, shape="gaussian"):
            persisted_cal = load_tau_calibration_impl(path, shape="gaussian")[
                "tau_calibration"
            ]
    else:
        if tau_calibration_present(path):
            persisted_cal = load_tau_calibration_impl(path)["tau_calibration"]
    tau_maj_override_v = resolved.tau.tau_maj_override_us
    sigma_tau_override_v = resolved.tau.sigma_tau_override_us
    tau_maj_us, sigma_tau_us, _tau_source = _resolve_tau_calibration_for_fit(
        persisted_cal, tau_maj_override_v, sigma_tau_override_v
    )

    # --- Build shared active-FT context (VERBATIM helper) ------------------
    # Replay the persisted Stage 5 gated spur catalogue rather than re-running
    # detection: the catalogue is a Stage 5 product, so the refit must mask the
    # window exactly as the fit did (a detector-code change between the fit and
    # the refit would otherwise silently re-mask the window it is editing).
    fit_ctx: Stage5FitContext = build_stage5_fit_context(
        path,
        resolved,
        persisted_cal,
        shape_enum,
        replay_spur_catalogue=spectrum_fit.parameters,
    )
    # --- In-memory refit core (materialize -> reconstruct frozen -> NLS) ---
    # Everything from the window materialization through the thawed-line
    # re-insertion lives in the file-I/O-free core so the survival pass can
    # reuse it verbatim. The shell keeps the file load, spur-catalogue replay,
    # persistence, and decision recording around this call.
    new_wf: FittingResult = refit_window_core(
        fit_ctx,
        fit_win,
        wf,
        resolved=resolved,
        shape_enum=shape_enum,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        peak_frequencies_mhz=peak_frequencies_mhz,
        add=add,
        remove=remove,
        add_seeds=add_seeds,
        snap_tol_mhz=snap_tol_mhz,
    )

    # --- Replace this window's entry in SpectrumFit + persist --------------
    # Update the global fitted_peaks list: remove old peaks for this window,
    # insert new ones, keep all other windows' peaks unchanged, re-sort.
    new_global_peaks = [
        p for p in spectrum_fit.fitted_peaks if p.window_id != window_id
    ] + list(new_wf.fitted_peaks)
    new_global_peaks.sort(key=lambda p: float(p.frequency_mhz))
    spectrum_fit.fitted_peaks = new_global_peaks

    # Replace the per-window FittingResult entry.
    spectrum_fit.window_fits = [
        new_wf if wf_entry.window_id == window_id else wf_entry
        for wf_entry in spectrum_fit.window_fits
    ]

    # Persist the updated SpectrumFit.
    shape_attr = str(spectrum_fit.parameters.get("shape", "lorentzian"))
    with h5py.File(path, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(spectrum_fit, grp)
        grp.attrs["shape"] = shape_attr

    chi2r_after = float(new_wf.reduced_chi2)
    n_peaks_after = len(new_wf.fitted_peaks)
    logger.info(
        "Stage 6 refit window %d: %d → %d peaks, χ²ᵣ %.3g → %.3g",
        window_id,
        n_peaks_before,
        n_peaks_after,
        chi2r_before,
        chi2r_after,
    )

    # Record per-frequency decisions in the Stage 6 review state.
    # Identity refits (no add, no remove) produce no log entries.
    # Callers that compose over this function (merge, split) pass
    # _skip_decision_recording=True and record their own coarser entries.
    if not _skip_decision_recording:
        _edit_evidence: Dict[str, object] = {
            "chi2r_before": chi2r_before,
            "chi2r_after": chi2r_after,
            "n_peaks_before": n_peaks_before,
            "n_peaks_after": n_peaks_after,
        }
        for _f in add:
            _record_decision(
                path,
                window_id=window_id,
                frequency_mhz=float(_f),
                kind="add",
                evidence=_edit_evidence,
            )
        for _f in remove:
            _record_decision(
                path,
                window_id=window_id,
                frequency_mhz=float(_f),
                kind="remove",
                evidence=_edit_evidence,
            )

    return RefitWindowResult(
        window_id=window_id,
        n_peaks_before=n_peaks_before,
        n_peaks_after=n_peaks_after,
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=list(new_wf.fitted_peaks),
    )


# ---------------------------------------------------------------------------
# merge_peaks_impl: collapse ≥2 fitted peaks into one
# ---------------------------------------------------------------------------


def merge_peaks_impl(
    file_path: Union[Path, str],
    window_id: int,
    peaks: Sequence[float],
    *,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> RefitWindowResult:
    """Collapse ≥2 fitted peaks in a window into a single peak.

    A thin composition over :func:`refit_window_impl`: removes the named peaks
    and adds one replacement seeded at their SNR-weighted centroid (or
    amplitude-weighted centroid when SNR is unavailable).  All products carry
    ``origin="user"``.

    **Doublet-alternative snap.** When the removed set matches a persisted
    ``DoubletAlternativeInfo`` pair (i.e. exactly two frequencies that
    together map to a recorded doublet pair within ``snap_tol_mhz``), the
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
        ``snap_tol_mhz``.
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping.

    Returns
    -------
    RefitWindowResult
        Old vs new peak count (reduced by ``len(peaks) - 1``), χ²ᵣ
        before/after, and the new fitted peaks.

    Raises
    ------
    ValueError
        When fewer than 2 frequencies are supplied, any frequency does not
        match a fitted peak within tolerance, or Stage 5 has not been run.
    """
    if len(peaks) < 2:
        raise ValueError(
            f"merge requires at least 2 peak frequencies; got {len(peaks)}"
        )

    # Load current fit to find the matched peaks and their weights.
    path = str(file_path)
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    wf_list = [wf for wf in spectrum_fit.window_fits if wf.window_id == window_id]
    if not wf_list:
        raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")
    wf: FittingResult = wf_list[0]

    # Match each requested frequency to a fitted peak within snap_tol_mhz.
    matched: List[FittedPeak] = []
    for req_freq in peaks:
        req_freq_f = float(req_freq)
        best: Optional[FittedPeak] = None
        best_dist = float("inf")
        for fp in wf.fitted_peaks:
            d = abs(float(fp.frequency_mhz) - req_freq_f)
            if d < best_dist:
                best_dist = d
                best = fp
        if best is None or best_dist > snap_tol_mhz:
            raise ValueError(
                f"merge: no fitted peak within {snap_tol_mhz:.3f} MHz of "
                f"{req_freq_f:.4f} MHz (closest distance: "
                f"{best_dist:.4f} MHz)"
            )
        if any(m.frequency_mhz == best.frequency_mhz for m in matched):
            raise ValueError(
                f"merge: frequency {req_freq_f:.4f} MHz matched the same "
                f"fitted peak twice"
            )
        matched.append(best)

    # Compute the replacement seed frequency and amplitude via weights.
    # Weight by SNR when available; fall back to amplitude.
    weights: List[float] = []
    for fp in matched:
        w = float(fp.snr) if fp.snr is not None else float(fp.amplitude)
        weights.append(max(w, 1e-30))
    total_w = sum(weights)
    centroid_freq = (
        sum(float(fp.frequency_mhz) * w for fp, w in zip(matched, weights)) / total_w
    )
    centroid_amp = sum(float(fp.amplitude) for fp in matched)

    # Doublet-alternative snap (two-peak case only).
    # Check the window's doublet_alternatives for a record that covers
    # exactly the two matched frequencies (order-independent).
    merge_freq = centroid_freq
    merge_amp = centroid_amp
    if len(matched) == 2:
        fa = float(matched[0].frequency_mhz)
        fb = float(matched[1].frequency_mhz)
        for da in getattr(wf, "doublet_alternatives", []):
            # Match either ordering.
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
                merge_freq = float(da.merged_frequency_mhz)
                merge_amp = (
                    float(da.merged_amplitude)
                    if not math.isnan(float(da.merged_amplitude))
                    else centroid_amp
                )
                logger.debug(
                    "merge_peaks window %d: snapped to doublet-alt seed "
                    "%.4f MHz (centroid was %.4f MHz)",
                    window_id,
                    merge_freq,
                    centroid_freq,
                )
                break

    remove_freqs = [float(fp.frequency_mhz) for fp in matched]
    add_freqs = [merge_freq]

    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)
    s = sideband_sign(sideband)
    center_mhz: Optional[float] = None
    if wf.window is not None and wf.window.freq_range is not None:
        lo, hi = wf.window.freq_range
        center_mhz = (lo + hi) / 2.0

    add_seeds: Optional[List] = None
    if center_mhz is not None:
        from ..fitting.peak_model import ModelPeak as _ModelPeak

        merge_offset = float(s * (merge_freq - center_mhz))
        add_seeds = [
            _ModelPeak(
                amplitude=max(merge_amp, 1e-30),
                offset_mhz=merge_offset,
                phase=0.0,
            )
        ]

    result = refit_window_impl(
        file_path,
        window_id,
        add=add_freqs,
        remove=remove_freqs,
        add_seeds=add_seeds,
        snap_tol_mhz=snap_tol_mhz,
        _skip_decision_recording=True,
    )

    # Record one "merge" decision entry anchored at the replacement frequency.
    _merge_evidence: Dict[str, object] = {
        "chi2r_before": result.chi2r_before,
        "chi2r_after": result.chi2r_after,
        "n_peaks_before": result.n_peaks_before,
        "n_peaks_after": result.n_peaks_after,
        "merged_from": [float(f) for f in peaks],
    }
    _record_decision(
        path,
        window_id=window_id,
        frequency_mhz=merge_freq,
        kind="merge",
        evidence=_merge_evidence,
    )

    return result


# ---------------------------------------------------------------------------
# split_peak_impl: replace one fitted peak with K peaks
# ---------------------------------------------------------------------------


def split_peak_impl(
    file_path: Union[Path, str],
    window_id: int,
    peak: float,
    *,
    into: int = 2,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> RefitWindowResult:
    """Replace one fitted peak with ``into`` peaks (default 2).

    A thin composition over :func:`refit_window_impl`: removes the named peak
    and adds ``into`` replacements spread symmetrically about it by ±½ of one
    Fourier resolution element (``1 / acquisition_us`` MHz).  All products
    carry ``origin="user"``.

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
        nearest fitted peak within ``snap_tol_mhz``.
    into :
        Number of replacement peaks (≥2).  Default is 2.
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping.

    Returns
    -------
    RefitWindowResult
        Old vs new peak count (increased by ``into - 1``), χ²ᵣ
        before/after, and the new fitted peaks.

    Raises
    ------
    ValueError
        When ``into < 2``, the frequency does not match a fitted peak within
        tolerance, or Stage 5 has not been run.
    """
    if into < 2:
        raise ValueError(f"split requires into >= 2; got {into}")

    # Load current fit to find the matched peak and the acquisition length.
    path = str(file_path)
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    wf_list = [wf for wf in spectrum_fit.window_fits if wf.window_id == window_id]
    if not wf_list:
        raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")
    wf: FittingResult = wf_list[0]

    # Match the requested frequency to the nearest fitted peak.
    peak_f = float(peak)
    best: Optional[FittedPeak] = None
    best_dist = float("inf")
    for fp in wf.fitted_peaks:
        d = abs(float(fp.frequency_mhz) - peak_f)
        if d < best_dist:
            best_dist = d
            best = fp
    if best is None or best_dist > snap_tol_mhz:
        raise ValueError(
            f"split: no fitted peak within {snap_tol_mhz:.3f} MHz of "
            f"{peak_f:.4f} MHz (closest distance: {best_dist:.4f} MHz)"
        )
    matched_freq = float(best.frequency_mhz)
    matched_amp = float(best.amplitude)

    # The split seed spacing needs the active acquisition length T and the
    # sideband, both already on the persisted fit -- no need to rebuild the
    # active-FT context for two scalars. T is persisted as ``acquisition_us``;
    # the seed frame matches ``materialize_window`` (window-midpoint centre).
    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)
    acquisition_us = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    if acquisition_us <= 0.0:
        # Legacy fits without the persisted scalar: the FID active duration is
        # a sufficient seed-spacing approximation (the joint NLS refines it).
        acquisition_us = float(fid.duration_us)

    # Resolution element: 1 / acquisition_us MHz.
    resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0 else 0.1

    # Place the K seeds symmetrically about ``matched_freq``.  For K=2 the
    # spacing is ±½ resolution element; for K>2 the spacing is evenly
    # distributed over 1 resolution element centred on ``matched_freq``.
    if into == 2:
        offsets = [-0.5 * resolution_mhz, 0.5 * resolution_mhz]
    else:
        half_span = 0.5 * resolution_mhz
        offsets = [-half_span + i * resolution_mhz / (into - 1) for i in range(into)]

    add_freqs = [matched_freq + off for off in offsets]
    per_peak_amp = matched_amp / into

    s = sideband_sign(sideband)
    center_mhz: Optional[float] = None
    if wf.window is not None and wf.window.freq_range is not None:
        lo, hi = wf.window.freq_range
        center_mhz = (lo + hi) / 2.0

    add_seeds: Optional[List] = None
    if center_mhz is not None:
        from ..fitting.peak_model import ModelPeak as _ModelPeak

        add_seeds = [
            _ModelPeak(
                amplitude=max(per_peak_amp, 1e-30),
                offset_mhz=float(s * (af - center_mhz)),
                phase=0.0,
            )
            for af in add_freqs
        ]

    result = refit_window_impl(
        file_path,
        window_id,
        add=add_freqs,
        remove=[matched_freq],
        add_seeds=add_seeds,
        snap_tol_mhz=snap_tol_mhz,
        _skip_decision_recording=True,
    )

    # Record one "split" decision entry anchored at the original peak frequency.
    _split_evidence: Dict[str, object] = {
        "chi2r_before": result.chi2r_before,
        "chi2r_after": result.chi2r_after,
        "n_peaks_before": result.n_peaks_before,
        "n_peaks_after": result.n_peaks_after,
        "split_into": into,
    }
    _record_decision(
        path,
        window_id=window_id,
        frequency_mhz=matched_freq,
        kind="split",
        evidence=_split_evidence,
    )

    return result


# ---------------------------------------------------------------------------
# Decision recording helpers and review-accept/status impls
# ---------------------------------------------------------------------------


def _record_decision(
    path: str,
    *,
    window_id: int,
    frequency_mhz: float,
    kind: str,
    evidence: Dict[str, object],
    kappa: float = DEFAULT_SHAPE_ERROR_KAPPA,
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR,
) -> None:
    """Append one entry to the Stage 6 decision log and update window provenance.

    Loads the persisted :class:`Stage6Review` (creating an empty one if absent),
    appends a :class:`DecisionLogEntry` for the given window and kind, sets
    the window's provenance to ``"user-edited"``, and re-persists.

    Attention reasons are refreshed from the current Stage 5 fit when available
    (the caller's edit may have resolved a misfit); when Stage 5 is absent the
    existing reasons are preserved.
    """
    with h5py.File(path, "r") as h5f:
        existing_review: Stage6Review = (
            load_stage6_review_from_hdf5(h5f["stage6_review"])
            if "stage6_review" in h5f
            else Stage6Review()
        )
        spectrum_fit: Optional[SpectrumFit] = (
            load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
            if "stage5_fitting" in h5f
            else None
        )

    new_entry = DecisionLogEntry(
        order_index=len(existing_review.decision_log),
        window_id=window_id,
        frequency_mhz=frequency_mhz,
        kind=kind,
        provenance="user",
        evidence=dict(evidence),
    )
    new_log = list(existing_review.decision_log) + [new_entry]

    existing_status = existing_review.window_statuses.get(window_id)

    # A decision may have changed the fit; rebuild the calibrated final-products
    # table from the current fit so the persisted contract never goes stale.
    # When Stage 5 is absent, carry the existing table forward unchanged.
    final_products = existing_review.final_products

    if spectrum_fit is not None:
        spur_centers_mhz: List[float] = [
            float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
        ]
        acquisition_us: float = float(
            spectrum_fit.parameters.get("acquisition_us", 0.0)
        )
        fid = load_fid_from_pipeline_impl(path)
        sideband = _sideband_from_value(fid.sideband)

        vif_attention_threshold = _resolve_vif_attention_threshold(path)
        merged_window_ids = _auto_merged_window_ids(spectrum_fit)
        wf_list = [wf for wf in spectrum_fit.window_fits if wf.window_id == window_id]
        if wf_list:
            new_reasons = _compute_attention_reasons(
                wf_list[0],
                spur_centers_mhz=spur_centers_mhz,
                acquisition_us=acquisition_us,
                ledger_bar=DEFAULT_DISPLAY_BAR,
                attention_candidate_evidence=DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
                sideband=sideband,
                kappa=kappa,
                noise_floor=noise_floor,
                vif_attention_threshold=vif_attention_threshold,
                auto_merged=window_id in merged_window_ids,
            )
        else:
            new_reasons = (
                list(existing_status.attention_reasons)
                if existing_status is not None
                else []
            )

        with h5py.File(path, "r") as h5f:
            floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
        cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
        final_products = _build_final_products(
            spectrum_fit,
            probe_freq_mhz=float(fid.probe_freq_mhz),
            sideband=sideband,
            calibration_state=cal_state,
            epsilon=epsilon,
            sigma_epsilon=sigma_eps,
            sigma_floor_khz=floor_khz,
        )
    else:
        new_reasons = (
            list(existing_status.attention_reasons)
            if existing_status is not None
            else []
        )

    new_statuses = dict(existing_review.window_statuses)
    new_statuses[window_id] = WindowReviewStatus(
        window_id=window_id,
        provenance="user-edited",
        attention_reasons=new_reasons,
        invalidated=False,
    )
    new_review = Stage6Review(
        window_statuses=new_statuses,
        decision_log=new_log,
        final_products=final_products,
    )

    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(new_review, grp)


def review_accept_impl(
    file_path: Union[Path, str],
    window_id: int,
    *,
    candidate_freq: Optional[float] = None,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> Optional[RefitWindowResult]:
    """Accept a window as-is or accept a specific revived candidate.

    With no ``candidate_freq``: records a ``"accept"`` decision log entry and
    sets the window's provenance to ``"reviewed"`` without modifying the fit.
    Attention reasons are preserved (they remain advisory after the user has
    looked at the window).  Returns ``None``.

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
        ledger candidate within ``snap_tol_mhz``.
    snap_tol_mhz :
        Maximum distance (MHz) for snapping to an existing ledger candidate.

    Returns
    -------
    RefitWindowResult or None
        ``None`` when accepting as-is; the refit result when ``candidate_freq``
        was given.
    """
    if candidate_freq is not None:
        return refit_window_impl(
            file_path,
            window_id,
            add=[candidate_freq],
            snap_tol_mhz=snap_tol_mhz,
        )

    path = str(file_path)

    # Determine a representative anchor frequency for the log entry.
    anchor_freq = 0.0
    try:
        with h5py.File(path, "r") as h5f:
            if "stage5_fitting" in h5f:
                sf: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
                wf_list = [wf for wf in sf.window_fits if wf.window_id == window_id]
                if wf_list:
                    c = _window_center(wf_list[0])
                    if c is not None:
                        anchor_freq = c
                    elif wf_list[0].fitted_peaks:
                        anchor_freq = float(
                            max(
                                wf_list[0].fitted_peaks,
                                key=lambda p: (
                                    float(p.snr) if p.snr is not None else 0.0
                                ),
                            ).frequency_mhz
                        )
    except Exception:
        pass

    existing_review: Stage6Review
    with h5py.File(path, "r") as h5f:
        existing_review = (
            load_stage6_review_from_hdf5(h5f["stage6_review"])
            if "stage6_review" in h5f
            else Stage6Review()
        )

    existing_status = existing_review.window_statuses.get(window_id)
    kept_reasons: List[AttentionReason] = (
        list(existing_status.attention_reasons) if existing_status is not None else []
    )

    new_entry = DecisionLogEntry(
        order_index=len(existing_review.decision_log),
        window_id=window_id,
        frequency_mhz=anchor_freq,
        kind="accept",
        provenance="user",
        evidence={},
    )
    new_log = list(existing_review.decision_log) + [new_entry]

    new_statuses = dict(existing_review.window_statuses)
    new_statuses[window_id] = WindowReviewStatus(
        window_id=window_id,
        provenance="reviewed",
        attention_reasons=kept_reasons,
        invalidated=False,
    )
    # Accepting as-is does not change the fit, so the calibrated final-products
    # table stays valid -- carry it forward unchanged.
    new_review = Stage6Review(
        window_statuses=new_statuses,
        decision_log=new_log,
        final_products=existing_review.final_products,
    )

    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(new_review, grp)

    return None


# ---------------------------------------------------------------------------
# Curation files: a human-editable batch language for the review edits.
#
# A curation file (CSV) records an ordered sequence of Stage 6 edits, one per
# row, that ``apply_curation_impl`` replays through the same edit impls the
# interactive verbs call. The CSV is the canonical interchange: diffable,
# hand-editable, and independent of the report that may have authored it.
# ---------------------------------------------------------------------------

_CURATION_ACTIONS = ("add", "remove", "merge", "split", "accept")
_CURATION_HEADER = ("action", "window", "freqs", "params")


@dataclass
class CurationOp:
    """One parsed row of a curation file (before coalescing).

    Attributes
    ----------
    action : str
        One of ``add`` / ``remove`` / ``merge`` / ``split`` / ``accept``.
    window_id : int
        The target ``FitWindow.window_id``.
    freqs : list of float
        Molecular MHz frequencies the row carries (empty for a bare accept).
    params : dict
        ``key=value`` modifiers (``into`` for split, ``candidate`` for accept).
    line_no : int
        1-based source line, for diagnostics.
    """

    action: str
    window_id: int
    freqs: List[float]
    params: Dict[str, str]
    line_no: int


@dataclass
class PlannedAction:
    """One resolved curation action (post-coalescing) ready to delegate.

    ``kind`` selects the target impl: ``edit`` -> :func:`refit_window_impl`
    (with the coalesced ``add`` / ``remove`` sets), ``merge`` ->
    :func:`merge_peaks_impl`, ``split`` -> :func:`split_peak_impl`, ``accept``
    -> :func:`review_accept_impl`.
    """

    kind: str
    window_id: int
    add: List[float] = field(default_factory=list)
    remove: List[float] = field(default_factory=list)
    peaks: List[float] = field(default_factory=list)
    peak: Optional[float] = None
    into: int = 2
    candidate: Optional[float] = None


@dataclass
class CurationApplyResult:
    """Outcome of :func:`apply_curation_impl`.

    Attributes
    ----------
    plan : list of PlannedAction
        The resolved, coalesced action sequence (the same in dry-run and live).
    warnings : list of str
        Frequency-resolution advisories (ambiguous or unmatched targets).
    applied : int
        Number of actions executed (``0`` for a dry run).
    dry_run : bool
        Whether the file was previewed without mutating.
    """

    plan: List["PlannedAction"]
    warnings: List[str]
    applied: int
    dry_run: bool


def _parse_curation_params(raw: str, line_no: int) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"curation line {line_no}: malformed parameter {token!r} "
                f"(expected key=value)"
            )
        key, _, value = token.partition("=")
        params[key.strip().lower()] = value.strip()
    return params


def parse_curation_file(curation_path: Union[Path, str]) -> List[CurationOp]:
    """Parse a curation CSV into ordered :class:`CurationOp` rows.

    Columns are ``action,window,freqs,params``. Blank lines and ``#`` comments
    are ignored; an optional header row (first cell ``action``) is skipped.
    ``freqs`` is a ``;``-separated list of molecular MHz; ``params`` is a
    ``;``-separated list of ``key=value`` modifiers.

    Raises ``ValueError`` (with the 1-based source line) on any malformed row.
    """
    text = Path(curation_path).read_text()
    ops: List[CurationOp] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(",")]
        action = fields[0].lower()
        if action == "action":  # header row
            continue
        if action not in _CURATION_ACTIONS:
            raise ValueError(
                f"curation line {line_no}: unknown action {fields[0]!r}; "
                f"choose one of {_CURATION_ACTIONS}"
            )
        if len(fields) < 2 or not fields[1]:
            raise ValueError(f"curation line {line_no}: missing window id")
        try:
            window_id = int(fields[1])
        except ValueError:
            raise ValueError(
                f"curation line {line_no}: window id {fields[1]!r} is not an integer"
            ) from None
        freqs_raw = fields[2] if len(fields) > 2 else ""
        params_raw = fields[3] if len(fields) > 3 else ""
        try:
            freqs = [float(x) for x in freqs_raw.split(";") if x.strip()]
        except ValueError:
            raise ValueError(
                f"curation line {line_no}: non-numeric frequency in {freqs_raw!r}"
            ) from None
        params = _parse_curation_params(params_raw, line_no)

        # Per-action arity / parameter validation.
        if action in ("add", "remove"):
            if len(freqs) != 1:
                raise ValueError(
                    f"curation line {line_no}: {action} needs exactly one frequency"
                )
            if params:
                raise ValueError(
                    f"curation line {line_no}: {action} takes no parameters"
                )
        elif action == "merge":
            if len(freqs) < 2:
                raise ValueError(
                    f"curation line {line_no}: merge needs at least two frequencies"
                )
        elif action == "split":
            if len(freqs) != 1:
                raise ValueError(
                    f"curation line {line_no}: split needs exactly one frequency"
                )
            if "into" in params:
                try:
                    into = int(params["into"])
                except ValueError:
                    raise ValueError(
                        f"curation line {line_no}: into={params['into']!r} "
                        f"is not an integer"
                    ) from None
                if into < 2:
                    raise ValueError(f"curation line {line_no}: into must be >= 2")
        elif action == "accept":
            if freqs:
                raise ValueError(
                    f"curation line {line_no}: accept takes no frequency column; "
                    f"use params candidate=F to revive a candidate"
                )
            if "candidate" in params:
                try:
                    float(params["candidate"])
                except ValueError:
                    raise ValueError(
                        f"curation line {line_no}: candidate="
                        f"{params['candidate']!r} is not a number"
                    ) from None

        ops.append(
            CurationOp(
                action=action,
                window_id=window_id,
                freqs=freqs,
                params=params,
                line_no=line_no,
            )
        )
    return ops


def _resolve_curation_plan(ops: Sequence[CurationOp]) -> List[PlannedAction]:
    """Coalesce parsed ops into the delegated action plan.

    A maximal run of ``add`` / ``remove`` rows on one window collapses into a
    single ``edit`` (one refit instead of one per row); a ``merge`` / ``split``
    / ``accept`` on that window is a barrier that flushes the window's pending
    edit first (it changes the peak set with its own seeding). Windows are
    independent, so an edit on another window does not flush a pending group.
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
        if op.action in ("add", "remove"):
            pa = pending.get(wid)
            if pa is None:
                pa = PlannedAction(kind="edit", window_id=wid)
                pending[wid] = pa
                pending_order.append(wid)
            if op.action == "add":
                pa.add.append(op.freqs[0])
            else:
                pa.remove.append(op.freqs[0])
            continue
        # Barrier for this window.
        flush(wid)
        if op.action == "merge":
            plan.append(
                PlannedAction(kind="merge", window_id=wid, peaks=list(op.freqs))
            )
        elif op.action == "split":
            plan.append(
                PlannedAction(
                    kind="split",
                    window_id=wid,
                    peak=op.freqs[0],
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


def describe_planned_action(action: PlannedAction) -> str:
    """Render one :class:`PlannedAction` as a one-line human-readable summary."""
    wid = action.window_id
    if action.kind == "edit":
        parts = []
        if action.add:
            parts.append("add " + ", ".join(f"{f:.4f}" for f in action.add))
        if action.remove:
            parts.append("remove " + ", ".join(f"{f:.4f}" for f in action.remove))
        return f"edit window {wid}: " + "; ".join(parts)
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
    """Molecular MHz of each window's persisted fitted peaks (empty if no fit)."""
    out: Dict[int, List[float]] = {}
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return out
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    for wf in spectrum_fit.window_fits:
        if wf.window_id is None:
            continue
        out[int(wf.window_id)] = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return out


def _curation_ambiguity_warnings(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> List[str]:
    """Advisories where a target frequency resolves to zero or >1 fitted peaks.

    ``remove`` / ``split`` / ``merge`` match an *existing* fitted peak by nearest
    frequency within ``snap_tol_mhz``; if two peaks sit within tolerance the
    matcher's pick is ambiguous, and if none do the edit will fail. Both are
    surfaced up front (especially useful with ``--dry-run``). ``add`` / accepted
    candidates create peaks, so they are not checked here.
    """
    by_window = _fitted_freqs_by_window(path)
    warnings: List[str] = []

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

    for action in plan:
        wid = action.window_id
        if action.kind == "edit":
            for f in action.remove:
                check(wid, f, f"remove {f:.4f}")
        elif action.kind == "merge":
            for f in action.peaks:
                check(wid, f, f"merge {f:.4f}")
        elif action.kind == "split" and action.peak is not None:
            check(wid, action.peak, f"split {action.peak:.4f}")
    return warnings


# --- the automatic-fit baseline (for undo replay) --------------------------
#
# A user edit mutates ``/stage5_fitting`` in place, so the automatic fit it
# replaced is otherwise unrecoverable. Before the first edit we snapshot the
# automatic fit into ``/stage5_fitting_baseline``; ``review_undo_impl`` restores
# it and replays the surviving decisions onto it. A fresh Stage 5 fit drops the
# snapshot (``clear_stage5_baseline``) so the next edit re-snapshots.

STAGE5_BASELINE_GROUP = "stage5_fitting_baseline"
_FIT_EDIT_KINDS = ("add", "remove", "merge", "split")


def _snapshot_stage5_baseline(path: str) -> None:
    """Copy ``/stage5_fitting`` to the baseline group if not already snapshotted."""
    with h5py.File(path, "a") as h5f:
        if "stage5_fitting" in h5f and STAGE5_BASELINE_GROUP not in h5f:
            h5f.copy("stage5_fitting", STAGE5_BASELINE_GROUP)


def clear_stage5_baseline(path: Union[Path, str]) -> None:
    """Drop the automatic-fit baseline snapshot (a fresh fit supersedes it)."""
    with h5py.File(str(path), "a") as h5f:
        if STAGE5_BASELINE_GROUP in h5f:
            del h5f[STAGE5_BASELINE_GROUP]


def _has_stage5_baseline(path: str) -> bool:
    with h5py.File(path, "r") as h5f:
        return STAGE5_BASELINE_GROUP in h5f


def _restore_stage5_baseline(path: str) -> None:
    """Replace ``/stage5_fitting`` with the baseline snapshot (kept for reuse)."""
    with h5py.File(path, "a") as h5f:
        if STAGE5_BASELINE_GROUP not in h5f:
            raise ValueError("no automatic-fit baseline to restore")
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        h5f.copy(STAGE5_BASELINE_GROUP, "stage5_fitting")


def _execute_planned_action(path: str, action: PlannedAction) -> None:
    """Dispatch one resolved action to its edit impl (shared by apply + undo)."""
    if action.kind == "edit":
        refit_window_impl(path, action.window_id, add=action.add, remove=action.remove)
    elif action.kind == "merge":
        merge_peaks_impl(path, action.window_id, action.peaks)
    elif action.kind == "split":
        if action.peak is None:
            raise ValueError("split action requires a peak frequency")
        split_peak_impl(path, action.window_id, action.peak, into=action.into)
    elif action.kind == "accept":
        review_accept_impl(path, action.window_id, candidate_freq=action.candidate)


def apply_curation_impl(
    file_path: Union[Path, str],
    curation_path: Union[Path, str],
    *,
    dry_run: bool = False,
) -> CurationApplyResult:
    """Apply a curation file to *file_path*, delegating to the edit impls.

    Parses the curation CSV, coalesces it into a delegated action plan (one
    refit per window for runs of add/remove; merge/split/accept stand alone),
    and -- unless ``dry_run`` -- executes each action through the same impls the
    interactive verbs call, so the result is identical to running the resolved
    plan by hand. ``dry_run`` returns the resolved plan and frequency-resolution
    warnings without mutating the file.

    Raises ``ValueError`` on a malformed curation file or when an action fails
    to resolve (e.g. a ``remove`` frequency matches no fitted peak), tagged with
    the offending action.
    """
    path = str(file_path)
    ops = parse_curation_file(curation_path)
    plan = _resolve_curation_plan(ops)
    warnings = _curation_ambiguity_warnings(path, plan)

    if dry_run:
        return CurationApplyResult(
            plan=plan, warnings=warnings, applied=0, dry_run=True
        )

    applied = 0
    for i, action in enumerate(plan):
        try:
            _execute_planned_action(path, action)
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"curation action {i + 1} ({describe_planned_action(action)}) "
                f"failed: {exc}"
            ) from exc
        applied += 1

    return CurationApplyResult(
        plan=plan, warnings=warnings, applied=applied, dry_run=False
    )


def review_log_impl(file_path: Union[Path, str]) -> List[DecisionLogEntry]:
    """Return the persisted Stage 6 decision log (read-only, execution order)."""
    review = load_stage6_review_from_file(str(file_path))
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
    """

    removed: List[DecisionLogEntry]
    surviving: List[DecisionLogEntry]
    plan: List["PlannedAction"]
    applied: int
    dry_run: bool


def _decision_to_op(entry: DecisionLogEntry) -> CurationOp:
    """Convert a decision-log entry back into a replayable curation op.

    Add/remove/accept replay from the entry alone; merge replays from the
    recorded ``merged_from`` peak set and split from ``split_into`` (both stamped
    by their impls at record time), so the decision log is loss-free for replay.
    """
    wid = entry.window_id
    kind = entry.kind
    if kind in ("add", "remove"):
        return CurationOp(kind, wid, [float(entry.frequency_mhz)], {}, 0)
    if kind == "merge":
        merged_from = entry.evidence.get("merged_from")
        if not merged_from or len(merged_from) < 2:
            raise ValueError(
                f"cannot replay merge on window {wid}: the decision log is "
                f"missing its 'merged_from' peak set"
            )
        return CurationOp("merge", wid, [float(f) for f in merged_from], {}, 0)
    if kind == "split":
        into = int(entry.evidence.get("split_into", 2))
        return CurationOp(
            "split", wid, [float(entry.frequency_mhz)], {"into": str(into)}, 0
        )
    if kind == "accept":
        return CurationOp("accept", wid, [], {}, 0)
    raise ValueError(f"cannot replay decision of unknown kind {kind!r}")


def review_undo_impl(
    file_path: Union[Path, str],
    ids: Sequence[int],
    *,
    dry_run: bool = False,
) -> UndoResult:
    """Undo one or more recorded decisions by id, replaying the rest.

    Rollback is replay-from-baseline: the automatic Stage 5 fit (snapshotted
    before the first edit) is restored, the review state rebuilt from it, and
    every *surviving* decision re-applied through the same engine ``review
    apply`` uses. The undone ids are dropped; the result is the canonical replay
    of the remaining decisions, so decision ids are renumbered afterward.

    ``dry_run`` returns the removed/surviving split and the resolved replay plan
    without mutating.

    Raises ``ValueError`` if an id is unknown, there are no decisions, or the
    automatic-fit baseline is unavailable while fit-mutating decisions exist
    (e.g. Stage 5 was re-run after editing -- rebuild and re-edit instead).
    """
    path = str(file_path)
    review = load_stage6_review_from_file(path)
    log = list(review.decision_log)
    if not log:
        raise ValueError("no recorded decisions to undo")

    valid_ids = {e.order_index for e in log}
    unknown = sorted({int(i) for i in ids} - valid_ids)
    if unknown:
        raise ValueError(
            f"unknown decision id(s) {unknown}; run 'review log' for valid ids"
        )
    undo_set = {int(i) for i in ids}
    if not undo_set:
        raise ValueError("no decision ids given to undo")

    removed = [e for e in log if e.order_index in undo_set]
    surviving = [e for e in log if e.order_index not in undo_set]

    has_fit_edits = any(e.kind in _FIT_EDIT_KINDS for e in log)
    baseline = _has_stage5_baseline(path)
    if has_fit_edits and not baseline:
        raise ValueError(
            "cannot undo: the automatic-fit baseline is unavailable (the fit may "
            "have been re-run after editing). Rebuild from the source and re-edit."
        )

    ops = [_decision_to_op(e) for e in surviving]
    plan = _resolve_curation_plan(ops)

    if dry_run:
        return UndoResult(
            removed=removed, surviving=surviving, plan=plan, applied=0, dry_run=True
        )

    # Restore the automatic fit (when fit-mutating edits existed), rebuild the
    # review afresh from it, then replay the surviving decisions onto it.
    if baseline:
        _restore_stage5_baseline(path)
    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
    review_run_impl(path)
    for action in plan:
        _execute_planned_action(path, action)

    return UndoResult(
        removed=removed,
        surviving=surviving,
        plan=plan,
        applied=len(plan),
        dry_run=False,
    )


def get_review_status_impl(file_path: Union[Path, str]) -> Stage6Review:
    """Load the :class:`Stage6Review` from *file_path*, or return an empty one.

    Read-only: does not write anything.  Safe to call before ``review run``.
    """
    return load_stage6_review_from_file(str(file_path))


# ---------------------------------------------------------------------------
# review_run_impl: build/refresh the per-window attention routing layer
# ---------------------------------------------------------------------------


@dataclass
class ReviewRunResult:
    """Summary returned by :func:`review_run_impl`.

    Attributes
    ----------
    n_windows : int
        Total number of fit windows processed.
    n_attention : int
        Number of windows with at least one attention reason.
    reason_counts : dict
        Maps attention-reason ``kind`` to the number of windows flagged for
        that reason (a window may contribute to multiple kinds).
    """

    n_windows: int
    n_attention: int
    reason_counts: Dict[str, int]


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
    vif_attention_threshold: float = DEFAULT_VIF_ATTENTION_THRESHOLD,
    auto_merged: bool = False,
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
    vif_attention_threshold :
        A window flags ``overfit_vif`` when any fitted amplitude's
        variance-inflation factor ``(amp_err / amp) * snr`` reaches this value
        -- the high-precision non-identifiability signal (degenerate split)
        that survived the end-of-Stage-5 collapse.

    Returns
    -------
    list of AttentionReason
        Advisory flags, possibly empty.
    """
    from ..fitting.validation import shape_error_fraction, snr_aware_chi2_pass
    from .stage5_impl import amplitude_vif

    reasons: List[AttentionReason] = []

    # --- auto_merged_review: the end-of-Stage-5 pass merged a degenerate close
    # pair in this window. Prior-free, multiplicity is a high-bar claim, so the
    # default is to merge; this advisory (low severity) lets a user with catalog
    # support find the merge and re-split it (``review split``). Not urgent --
    # the merge is the more-likely-correct call (~92% of the band is over-fits).
    if auto_merged:
        reasons.append(
            AttentionReason(
                kind="auto_merged_review",
                detail=(
                    "a degenerate sub-resolution pair was auto-merged; "
                    "re-split (review split) if catalog/model supports two lines"
                ),
                severity=0.1,
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

    # --- overfit_vif: flag a non-identifiable (degenerate-split) amplitude ---
    # The diagonal amplitude variance-inflation factor ``(amp_err / amp) * snr``
    # is ~1 for an identifiable line and >> 1 when a line is degenerate with a
    # sub-resolution neighbour (the pair *sum* is constrained, neither amplitude
    # individually). The end-of-Stage-5 collapse already removed the extreme
    # (VIF > vif_collapse_threshold, sub-half-resolution) pairs, so what reaches
    # here is the attention band: high-VIF lines that survived because they sit
    # above the collapse separation guard. SNR-normalized, so a genuine low-SNR
    # multiplet (large *raw* errors, VIF still ~2) is not flagged. The
    # AICc-preferred-doublet observation is *not* an attention trigger -- that is
    # the pipeline confirming a real doublet, not an actionable overfit (the
    # doublet detail still surfaces in ``review show --window N``).
    worst_vif: Optional[float] = None
    worst_vif_freq: float = 0.0
    for p in wf.fitted_peaks:
        vif = amplitude_vif(p)
        if vif is None or not math.isfinite(vif):
            continue
        if vif >= vif_attention_threshold and (worst_vif is None or vif > worst_vif):
            worst_vif = vif
            worst_vif_freq = float(p.frequency_mhz)
    if worst_vif is not None:
        reasons.append(
            AttentionReason(
                kind="overfit_vif",
                detail=(
                    f"amplitude VIF={worst_vif:.1f} at {worst_vif_freq:.4f} MHz "
                    f">= {vif_attention_threshold:.1f} "
                    f"(non-identifiable; degenerate with a neighbour)"
                ),
                severity=float(worst_vif),
            )
        )

    # NOTE: a low-SNR fitted peak is deliberately NOT an attention reason. A
    # weak peak just above the survival floor is rarely actionable (an isolated
    # weak false positive does little harm), so flagging the whole band floods
    # the queue with low-value items. Weak windows are surfaced on demand via
    # ``review rank --by min-snr`` instead (exploration decoupled from flags).

    # --- candidate_bearing: flag when the window has candidates above bar ----
    center_mhz = _window_center(wf)
    if center_mhz is not None:
        res_element_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else None
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
                )
            )

    # --- spur_adjacent: flag when a gated spur center falls in or near the window ---
    if wf.window is not None and wf.window.freq_range is not None:
        flo, fhi = wf.window.freq_range
        resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0.0 else 0.1
        for spur_f in spur_centers_mhz:
            if (flo - resolution_mhz) <= spur_f <= (fhi + resolution_mhz):
                reasons.append(
                    AttentionReason(
                        kind="spur_adjacent",
                        detail=(
                            f"gated spur at {spur_f:.4f} MHz near or within window "
                            f"[{flo:.4f}, {fhi:.4f}] MHz"
                        ),
                        severity=1.0,
                    )
                )
                break  # one spur per window is sufficient for routing

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
                )
            )

    return reasons


# ---------------------------------------------------------------------------
# Final-products consolidation (frequency calibration + sigma_f budget)
# ---------------------------------------------------------------------------


def set_sigma_floor_impl(file_path: Union[Path, str], sigma_floor_khz: float) -> None:
    """Persist the user's systematic accuracy floor into ``/frequency_calibration``.

    The floor is file-level provenance (a sibling of the source metadata), so
    any reported ``sigma_f`` is reproducible from the record alone and never
    depends on a transient flag. Does not rebuild the final-products table;
    call ``review run`` to fold the new floor into the budget.
    """
    floor = float(sigma_floor_khz)
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError(
            f"sigma_floor_khz must be finite and non-negative, got {sigma_floor_khz!r}"
        )
    with h5py.File(str(file_path), "a") as h5f:
        save_frequency_calibration_to_hdf5(FrequencyCalibration(floor), h5f)


def get_final_products_impl(file_path: Union[Path, str]) -> Optional[FinalProducts]:
    """Return the persisted Stage 6 final-products table, or ``None``."""
    return load_stage6_review_from_file(str(file_path)).final_products


def _derive_frequency_calibration(
    path: str,
) -> Tuple[str, float, float]:
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
    """
    from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
    from ..io.stage_fit_settings_serialization import load_stage_fit_settings_from_h5

    clocks: Tuple = ()
    try:
        resolved = resolve_stage_fit_settings(
            persisted=load_stage_fit_settings_from_h5(path)
        )
        clocks = tuple(resolved.spur.clocks or ())
    except Exception:
        clocks = ()

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


def _build_final_products(
    spectrum_fit: SpectrumFit,
    *,
    probe_freq_mhz: float,
    sideband: Sideband,
    calibration_state: str,
    epsilon: float,
    sigma_epsilon: float,
    sigma_floor_khz: float,
) -> FinalProducts:
    """Consolidate the Stage 5 line list into the calibrated final-products table.

    Applies the timebase scale correction in the baseband frame
    (``f_corr = probe + (f_raw - probe)/(1+epsilon)``, sideband-independent) and
    builds the three-term ``sigma_f`` budget per accepted peak:
    ``sqrt(sigma_stat^2 + (sigma_epsilon * f_baseband)^2 + sigma_floor^2)``.
    """
    floor_khz = float(sigma_floor_khz)
    final_peaks: List[FinalPeak] = []
    for pk in spectrum_fit.fitted_peaks:
        f_raw = float(pk.frequency_mhz)
        f_baseband_mhz = abs(f_raw - probe_freq_mhz)

        if epsilon != 0.0:
            f_corr = probe_freq_mhz + (f_raw - probe_freq_mhz) / (1.0 + epsilon)
        else:
            f_corr = f_raw

        sigma_stat_khz = (
            float(pk.frequency_error) * 1.0e3 if pk.frequency_error is not None else 0.0
        )
        sigma_eps_khz = float(sigma_epsilon) * f_baseband_mhz * 1.0e3
        sigma_f_khz = math.sqrt(sigma_stat_khz**2 + sigma_eps_khz**2 + floor_khz**2)

        amp = float(pk.amplitude)
        amp_err = None if pk.amplitude_error is None else float(pk.amplitude_error)
        snr_val = None if pk.snr is None else float(pk.snr)
        # Propagate the amplitude error into an SNR error (SNR scales with
        # amplitude at fixed noise): sigma_snr = snr * sigma_amp / amp.
        snr_err: Optional[float] = None
        if snr_val is not None and amp_err is not None and amp != 0.0:
            snr_err = abs(snr_val) * abs(amp_err / amp)

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
                phase=None if pk.phase is None else float(pk.phase),
                snr=snr_val,
                origin=str(pk.origin),
                window_id=None if pk.window_id is None else int(pk.window_id),
                amplitude_error=amp_err,
                phase_error=None if pk.phase_error is None else float(pk.phase_error),
                snr_error=snr_err,
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


def review_run_impl(
    file_path: Union[Path, str],
    *,
    bar: float = DEFAULT_DISPLAY_BAR,
    attention_candidate_evidence: float = DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    kappa: float = DEFAULT_SHAPE_ERROR_KAPPA,
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR,
    sigma_floor_khz: Optional[float] = None,
) -> ReviewRunResult:
    """Build or refresh the Stage 6 attention-routing layer and final products.

    Loads the Stage 5 fit, computes advisory attention reasons for every
    window, consolidates the calibrated final-products table (frequencies
    corrected for the digitizer timebase scale error and the three-term
    ``sigma_f`` budget), and persists a
    :class:`~ftmwpipeline.core.data_structures.Stage6Review` to the
    ``stage6_review`` HDF5 group.  Marks the ``stage6_review`` tracker stage
    complete.

    Idempotent: if a ``stage6_review`` group already exists, the existing
    per-window ``provenance`` (``"reviewed"``/``"user-edited"``) and the
    ``decision_log`` are preserved; only ``attention_reasons`` are
    recomputed.  The ``"auto"`` provenance is never upgraded by this call.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    bar :
        Display bar forwarded to :func:`get_candidate_ledger_impl` for the
        candidate-bearing attention reason (default :data:`DEFAULT_DISPLAY_BAR`).
    attention_candidate_evidence :
        A window flags ``candidate_bearing`` only when its strongest revivable
        candidate's evidence clears this threshold (default
        :data:`DEFAULT_ATTENTION_CANDIDATE_EVIDENCE`) -- stiffer than ``bar`` so
        the attention surface stays actionable while the ledger still lists
        every candidate above ``bar``.
    kappa :
        Shape-error kappa for the SNR-aware chi-squared gate (default
        :data:`DEFAULT_SHAPE_ERROR_KAPPA`).
    noise_floor :
        Noise-regime chi-squared allowance (default
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
    """
    path = str(file_path)

    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        # Load existing review to preserve provenance/decision_log.
        existing_review: Stage6Review
        if "stage6_review" in h5f:
            existing_review = load_stage6_review_from_hdf5(h5f["stage6_review"])
        else:
            existing_review = Stage6Review()

    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)

    spur_centers_mhz: List[float] = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
    ]
    acquisition_us: float = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    vif_attention_threshold = _resolve_vif_attention_threshold(path)
    merged_window_ids = _auto_merged_window_ids(spectrum_fit)

    new_statuses: Dict[int, WindowReviewStatus] = {}
    for wf in spectrum_fit.window_fits:
        wid = int(wf.window_id) if wf.window_id is not None else -1

        reasons = _compute_attention_reasons(
            wf,
            spur_centers_mhz=spur_centers_mhz,
            acquisition_us=acquisition_us,
            ledger_bar=bar,
            attention_candidate_evidence=attention_candidate_evidence,
            sideband=sideband,
            kappa=kappa,
            noise_floor=noise_floor,
            vif_attention_threshold=vif_attention_threshold,
            auto_merged=wid in merged_window_ids,
        )

        # Preserve existing provenance (never downgrade reviewed/user-edited to auto).
        existing_status = existing_review.window_statuses.get(wid)
        if existing_status is not None and existing_status.provenance != "auto":
            provenance = existing_status.provenance
            invalidated = existing_status.invalidated
        else:
            provenance = "auto"
            invalidated = False

        new_statuses[wid] = WindowReviewStatus(
            window_id=wid,
            provenance=provenance,
            attention_reasons=reasons,
            invalidated=invalidated,
        )

    # Consolidate the calibrated final-products table. A newly-declared
    # accuracy floor is persisted as file-level provenance first, so the budget
    # reflects exactly what the record carries (never a transient flag).
    if sigma_floor_khz is not None:
        set_sigma_floor_impl(path, sigma_floor_khz)
    with h5py.File(path, "r") as h5f:
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
    cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
    final_products = _build_final_products(
        spectrum_fit,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        calibration_state=cal_state,
        epsilon=epsilon,
        sigma_epsilon=sigma_eps,
        sigma_floor_khz=floor_khz,
    )

    new_review = Stage6Review(
        window_statuses=new_statuses,
        decision_log=list(existing_review.decision_log),
        final_products=final_products,
    )

    # Persist: write stage6_review group and mark tracker stage complete.
    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(new_review, grp)

    _update_stage_completion(path, "stage6_review")

    # Compute summary.
    reason_counts: Dict[str, int] = {}
    n_attention = 0
    for status in new_statuses.values():
        if status.needs_attention:
            n_attention += 1
        for reason in status.attention_reasons:
            reason_counts[reason.kind] = reason_counts.get(reason.kind, 0) + 1

    return ReviewRunResult(
        n_windows=len(new_statuses),
        n_attention=n_attention,
        reason_counts=reason_counts,
    )
