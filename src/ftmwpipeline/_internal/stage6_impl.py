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

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np

from ..core.curation import REFIT_SNAP_TOL_MHZ, Frame
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
    RescueRoundInfo,
    Sideband,
    SpectrumFit,
    Stage6Review,
    WindowReviewStatus,
)
from ..fitting.active_ft import active_ft_bin_spacing_mhz
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
    from ..core.data_structures import FitWindow, Peak, WindowPlan
    from ..core.environment import EnvironmentRecord
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
from ..fitting.validation import SHAPE_ERROR_REACH_KAPPA

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
        with h5py.File(path, "r") as h5f:
            if "stage5_fitting" not in h5f:
                raise ValueError(
                    "No Stage 5 fit found in this file. Run 'fit run' first."
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
    to the right peak, but lands ``probe_freq * eps/(1+eps)`` off -- under the
    snap tolerance, over the statistical sigma, invisible in the result).
    Passing ``frame=\"raw\"`` explicitly is never refused, on any file.
    """
    stamp = _current_calibration_stamp(path)
    cal_state = stamp[0] if stamp is not None else "rb_locked"
    if frame is None:
        if cal_state == "self_calibrated":
            raise ValueError(
                "frame is required on a self_calibrated file: pass "
                'frame="raw" or frame="calibrated" explicitly rather than '
                "relying on the default. A calibrated frequency submitted as "
                "raw still resolves to the right peak, but is wrong by "
                "probe_freq * eps/(1+eps) -- under the snap tolerance and "
                "over the statistical uncertainty, so the mistake would be "
                "silent."
            )
        return "raw", stamp
    return frame, stamp


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
        raise ValueError(
            f"curation file frame disagreement: the file's header declares "
            f'frame="{header.frame}", but frame="{frame}" was passed '
            f"explicitly. Pass a matching frame (or omit it to use the "
            f"file's header), or edit the file's header to match."
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
            raise ValueError(
                f"curation file frame drift: this file was staged "
                f"frame=calibrated at epsilon={header.epsilon:.6e}, but the "
                f"target file's current epsilon is {current_eps:.6e}. The "
                f"calibration has changed since this file was written (e.g. "
                f"a timebase re-run) -- re-stage the curation file against "
                f"the current calibration rather than applying it as-is."
            )

    return resolved_frame, stamp


def _frame_to_raw(
    freq_mhz: float, *, frame: Frame, stamp: Optional[_CalibrationStamp]
) -> float:
    """Convert one caller-supplied frequency to the raw (stored / fit) frame.

    Inverts the baseband-only correction
    (``f_corr = probe + (f_raw - probe) / (1 + eps)``):
    ``f_raw = probe + (f_corr - probe) * (1 + eps)``. Identity when
    ``frame == \"raw\"``, when ``epsilon == 0`` (rb_locked/uncalibrated), or
    when the file carries no calibration to convert against.
    """
    if frame == "raw" or stamp is None:
        return float(freq_mhz)
    _, epsilon, _, _, probe_freq_mhz, _ = stamp
    if epsilon == 0.0:
        return float(freq_mhz)
    return float(probe_freq_mhz + (freq_mhz - probe_freq_mhz) * (1.0 + epsilon))


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
    chi2r_before : float
        Reduced chi-squared before the refit.
    chi2r_after : float
        Reduced chi-squared after the refit.
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
    """

    window_id: int
    n_peaks_before: int
    n_peaks_after: int
    chi2r_before: float
    chi2r_after: float
    fitted_peaks: List[FittedPeak] = field(default_factory=list)
    fitted_peaks_calibrated_mhz: List[float] = field(default_factory=list)
    calibration_state: str = "rb_locked"
    epsilon: float = 0.0
    sigma_epsilon: float = 0.0


def _make_refit_result(
    ctx: "_BatchCtx",
    *,
    window_id: int,
    n_peaks_before: int,
    n_peaks_after: int,
    chi2r_before: float,
    chi2r_after: float,
    fitted_peaks: List[FittedPeak],
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
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=fitted_peaks,
        fitted_peaks_calibrated_mhz=calibrated,
        calibration_state=shared.calibration_state,
        epsilon=shared.epsilon,
        sigma_epsilon=shared.sigma_epsilon,
    )


# ---------------------------------------------------------------------------
# Single-window refit engine
#
# The add/remove/anchor snap tolerance every verb below defaults to is
# ``REFIT_SNAP_TOL_MHZ``, imported from ``core.curation`` -- public, because an
# integrator that resolved "the peak at f" at a different tolerance would
# disagree with the file about which peak that is.  It is deliberately looser
# than the resolved ``_DEDUP_TOL_BINS`` tolerance (the input is a frequency a
# person typed, not a fitted value); that constant's docstring has the rest
# of the rationale.
# ---------------------------------------------------------------------------


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


def require_splice_compatible_environment(path: str) -> None:
    """Refuse to splice a newly-computed fit into an artifact from another epoch.

    Stage 6's edit verbs are the one place the pipeline writes a *partial*
    result into a finished one: ``refit_window_impl`` re-fits a single window
    and writes it back into a :class:`SpectrumFit` whose other windows were fit
    earlier, and the cascade then re-fits the dependents. If the fitting code
    changed in between, the result is two different models inside one product --
    a state no per-file version stamp can express and no report can caveat
    honestly, because the mixture is *within* the artifact.

    That is why this is the one operation class the environment policy blocks.
    Extending a file forward is fine (a new stage is self-consistently produced
    by the current environment, and only warns); reading is never gated.

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
        with h5py.File(path, "r") as h5f:
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
            logger.warning(
                "Editing a Stage 5 fit produced under analysis epoch %s with "
                "epoch %s; proceeding on the acknowledgement recorded in the "
                "file. The curated fit mixes two analysis environments.",
                fit_env.analysis_epoch,
                current.analysis_epoch,
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
    with h5py.File(path, "r") as h5f:
        envs = load_stage_environments(h5f)
    fit_env = envs.get("stage5_fitting")
    mismatch = gating_fields_differ(current, fit_env)

    with h5py.File(path, "a") as h5f:
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

    Pulled out of :func:`effective_window_plan` so the batch curation engine
    (:func:`_execute_curation_batch`) can reapply the same overlay purely in
    memory as ``create`` actions accumulate within one batch -- geometry for the
    *next* create in the same batch has to see the previous one without a round
    trip through the file between them.
    """
    if not created_windows:
        return plan

    overlay = {int(w.window_id): w for w in created_windows}
    windows = [overlay.get(int(w.window_id), w) for w in plan.windows]
    known = {int(w.window_id) for w in windows}
    windows.extend(w for wid, w in sorted(overlay.items()) if wid not in known)
    windows.sort(key=lambda w: min(w.freq_range))

    # A created window is a leaf: it reads its neighbours' frozen leakage and
    # nothing reads it, so its edges are purely additive and cannot introduce a
    # cycle. Splice them in and put the new ids last in the fit order.
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
    """The Stage 4 plan overlaid with any Stage-6-created / widened windows.

    Stage 6 can install a window for a line the automatic detection missed (see
    :func:`create_window_impl`). Those windows live in the Stage 6 review state,
    not in ``/stage4_windows``, so Stage 4's persisted product stays a function
    of Stage 4's own inputs and re-running Stage 4 (which invalidates Stage 6
    anyway) never has to reconcile them. Every Stage 6 code path that resolves a
    ``window_id`` to its geometry goes through here so the base plan and the
    overlay are never read apart.
    """
    from .stage4_impl import load_windows_impl

    plan: "WindowPlan" = load_windows_impl(str(file_path))["plan"]
    review = load_stage6_review_from_file(str(file_path))
    return _overlay_created_windows(plan, review.created_windows)


def _next_decision_index(path: str) -> int:
    """``order_index`` the next recorded decision will take.

    ``_record_decision`` appends at ``len(decision_log)``, so an edit can know
    the id of the decision it is about to record *before* running the fit -- the
    hook the per-peak derivation tag hangs on.
    """
    with h5py.File(path, "r") as h5f:
        if "stage6_review" not in h5f:
            return 0
        review = load_stage6_review_from_hdf5(h5f["stage6_review"])
    return len(review.decision_log)


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
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    freeze_inherited: bool = False,
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
    catalog replayed, calls this core, then persists and records decisions).
    The Stage 5 peak-survival pass routes through it too, holding ``fit_ctx`` /
    the plan / ``resolved`` live from ``fit_peaks_impl``. ``add_origin`` stamps
    the origin of added peaks: ``"user"`` for a user edit (the default, immune
    to later auto-prune/cleanup), ``"auto"`` for an automatic add such as the
    VIF-collapse merged line (a normal fitted peak, not a human decision).
    ``add_derivations`` (index-aligned with ``add``) stamps
    :attr:`FittedPeak.derivation` -- the decision-log ``order_index`` that
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
    converged coefficients (see ``initial_baseline_coeffs`` below) so an
    identity refit is a fixed point: the baseline stays a free parameter (the
    model is unchanged), but the joint NLS starts at the originating fit's
    converged baseline rather than cold-starting at zero, which would re-open
    the near-degenerate baseline/position valley on wide, low-SNR windows and
    slide untouched peaks.

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

    # (seed, origin, derivation) triples, kept together so the "remove" pop and
    # the by-position origin/derivation stamping below cannot fall out of step.
    seed_peaks_with_origin: List[Tuple[ModelPeak, str, Optional[int]]] = []
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
            seed_peaks_with_origin.append((mp, fp.origin, fp.derivation))

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
        # Range check on the POST-snap seed: the snap (or an explicit
        # ``add_seeds`` entry) is what the NLS actually starts from, so it is
        # what has to lie on this window's data. Rejecting here is consistent
        # with how an unsnappable ``remove`` is already handled, and turns the
        # "resolved the click to the wrong window" case into an error instead of
        # a fit against data the window does not cover.
        seed_freq_mhz = float(center_mhz + s * mp.offset_mhz)
        if not (win_lo - grid_slack <= seed_freq_mhz <= win_hi + grid_slack):
            raise ValueError(
                f"add={float(add_freq):.4f} MHz resolves to {seed_freq_mhz:.4f} "
                f"MHz, outside window {window_id}'s range "
                f"[{win_lo:.4f}, {win_hi:.4f}] MHz. Name the window that covers "
                f"the frequency, or -- if no window does -- create one with "
                f"'review create' first."
            )
        add_derivation = (
            add_derivations[i]
            if add_derivations is not None and i < len(add_derivations)
            else None
        )
        seed_peaks_with_origin.append((mp, add_origin, add_derivation))
        protected_offsets.append(mp.offset_mhz)

    # Extract final seed list in offset order.
    final_seeds = [mp for mp, _, _ in seed_peaks_with_origin]
    origin_flags = [orig for _, orig, _ in seed_peaks_with_origin]
    derivation_flags = [deriv for _, _, deriv in seed_peaks_with_origin]

    # "Freeze inherited" mode for the VIF-collapse sequential merge: fit ONLY
    # the added (merged) seeds, holding every inherited peak frozen at its
    # persisted value -- added to the frozen background AND re-appended verbatim
    # after the NLS (the same hold-out path thawed lines use). The sequential
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
                        int(src_fp.peak_id) if src_fp.peak_id is not None else -1
                    ),
                    primary_window_id=-1,
                    model_peak=mp,
                    frequency_mhz=freq,
                    freeze_eligible=False,
                    edge_free=False,
                )
            )
            thawed_held_peaks.append(src_fp)
        final_seeds = list(added_seeds)
        origin_flags = list(added_origins)
        derivation_flags = list(added_derivations)
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
    if len(new_wf.fitted_peaks) == len(origin_flags):
        for fp, orig, deriv in zip(new_wf.fitted_peaks, origin_flags, derivation_flags):
            if orig == "user":
                fp.origin = "user"
            if deriv is not None:
                fp.derivation = int(deriv)
    else:
        user_seeds = [
            (float(center_mhz + s * mp.offset_mhz), orig, deriv)
            for mp, orig, deriv in zip(final_seeds, origin_flags, derivation_flags)
            if orig == "user" or deriv is not None
        ]
        for fp in new_wf.fitted_peaks:
            for uf, orig, deriv in user_seeds:
                if abs(float(fp.frequency_mhz) - uf) <= snap_tol_mhz:
                    if orig == "user":
                        fp.origin = "user"
                    if deriv is not None:
                        fp.derivation = int(deriv)
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
        # full peak count.  Clear it rather than persist a partial / mislabeled
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

    # ``freeze_inherited`` parked the window's OWN inherited peaks in the frozen
    # background to hold them during the merged-line fit (and re-appended them to
    # ``fitted_peaks`` above). They must NOT persist as fixed contributors -- a
    # later refit would reconstruct them as background AND fit them as peaks
    # (double-count). Restore the original contributor set; the inherited peaks
    # live only in ``fitted_peaks``.
    if freeze_inherited and len(add) > 0:
        new_wf.fixed_parameters = dict(wf.fixed_parameters or {})

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


def _cascade_succs(
    window_fits: Sequence[FittingResult],
    fit_window_map: Dict[int, "FitWindow"],
) -> Dict[int, set]:
    """Reverse dependency map ``primary -> {dependent}`` over the fitted windows,
    from each window's NON-edge-free frozen contributors (the cascade edges)."""
    fitted = {int(wf.window_id) for wf in window_fits if wf.window_id is not None}
    succs: Dict[int, set] = {w: set() for w in fitted}
    for wf in window_fits:
        if wf.window_id is None:
            continue
        d = int(wf.window_id)
        nonef = _non_edge_free_primaries(fit_window_map.get(d))
        for key, entry in wf.fixed_parameters.items():
            if not key.startswith("frozen_peak_"):
                continue
            p = int(entry["primary_window_id"])
            if p not in fitted or p == d:
                continue
            if nonef is not None and p not in nonef:
                continue  # edge-free contributor: no cascade edge
            succs[p].add(d)
    return succs


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


def _cascade_topo(nodes: set, preds: Dict[int, set]) -> List[int]:
    """Kahn topological order of ``nodes`` (predecessors within the set gate)."""
    nodes = set(nodes)
    placed: set = set()
    out: List[int] = []
    remaining = sorted(nodes)
    while remaining:
        ready = [w for w in remaining if (preds.get(w, set()) & nodes) <= placed]
        if not ready:
            ready = remaining  # residual cycle: bail in input order
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
        from .stage5_impl import resolve_window_tau_anchor

        center_mhz = 0.5 * (fit_win.freq_range[0] + fit_win.freq_range[1])
        return resolve_window_tau_anchor(
            center_mhz,
            persisted_cal.band_majorities,  # type: ignore[attr-defined]
            tau_maj_us,
            sigma_tau_us,
        )
    return tau_maj_us, sigma_tau_us


def _refresh_frozen_window_level(
    wf: FittingResult,
    fit_window_map: Dict[int, "FitWindow"],
    fit_map: Dict[int, FittingResult],
    min_freeze_snr: float,
) -> None:
    """Rebuild ``wf``'s non-edge-free frozen contributors from its source windows'
    **current** fitted peaks clearing ``min_freeze_snr`` (window-level resolution).

    This is the one piece the cascade adds: a plain refit reconstructs the frozen
    background from ``wf``'s own persisted snapshot (a fixed point -- a no-op), so a
    dependent only tracks its source's edit once its background is re-read from the
    source's live fit. Edge-free contributors (read from data, cascade-immune) and
    any non-``frozen_peak_*`` entries are preserved verbatim. Add / remove / split /
    delete are handled uniformly: the source simply has more or fewer peaks above
    threshold.

    Thawed peaks (co-fit lines owned by a primary, held in ``fitted_peaks``, design
    §5) are not refreshed here -- no current fixture exercises thaw; they remain at
    their persisted values, which ``refit_window_core`` holds frozen. Tracked as the
    one §5 gap.
    """
    wid = wf.window_id
    assert wid is not None
    d = int(wid)
    nonef = _non_edge_free_primaries(fit_window_map.get(d))

    def _is_dep(primary: int) -> bool:
        return nonef is None or primary in nonef

    non_frozen: Dict[str, Dict] = {}
    preserved_edge_free: List[Dict] = []
    src_wids: List[int] = []
    for key, entry in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            non_frozen[key] = entry
            continue
        primary = int(entry["primary_window_id"])
        if _is_dep(primary):
            if primary not in src_wids:
                src_wids.append(primary)
        else:
            preserved_edge_free.append(entry)

    rebuilt: List[Dict] = []
    for primary in src_wids:
        pwf = fit_map.get(primary)
        if pwf is None:
            continue  # source dropped/merged away -> contributes no skirt
        for pk in sorted(pwf.fitted_peaks, key=lambda q: float(q.frequency_mhz)):
            if float(pk.snr or 0.0) < min_freeze_snr:
                continue
            rebuilt.append(
                {
                    "peak_index": -1,
                    "primary_window_id": primary,
                    "frequency_mhz": float(pk.frequency_mhz),
                    "amplitude": float(pk.amplitude),
                    "phase": float(pk.phase) if pk.phase is not None else 0.0,
                    "freeze_eligible": True,
                }
            )

    frozen = preserved_edge_free + rebuilt
    rekeyed = {f"frozen_peak_{i}": e for i, e in enumerate(frozen)}
    wf.fixed_parameters = {**non_frozen, **rekeyed}


def _cascade_refit_dependents(
    *,
    spectrum_fit: SpectrumFit,
    edited_wids: Sequence[int],
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
) -> List[int]:
    """Refresh + identity-refit every dependent in the transitive closure of
    ``edited_wids`` (window-level), in dependency order; splice the results back
    into ``spectrum_fit``. ``tau_maj_us`` / ``sigma_tau_us`` are the **global**
    anchors -- each dependent is re-anchored per band. Returns the cascaded ids.

    The refit is identity (no add/remove): a directly-edited dependent already
    carries its own edit in its peak set, so the identity refit honors both the edit
    and the refreshed skirt in one fit (design §3). Mutates ``spectrum_fit``.
    """
    from ..fitting.result_conversion import sort_fitting_result_by_frequency

    window_fits = spectrum_fit.window_fits
    fit_map: Dict[int, FittingResult] = {
        int(wf.window_id): wf for wf in window_fits if wf.window_id is not None
    }
    succs = _cascade_succs(window_fits, fit_window_map)
    closure = _cascade_closure(edited_wids, succs)
    # `_cascade_closure` strips the whole `edited_wids` set from its result, so
    # when this call batches several DIRECTLY edited windows together (the
    # curation batch engine's single combined cascade), a window that is both
    # directly edited AND downstream of ANOTHER directly-edited window in the
    # same call would otherwise never get its frozen background refreshed from
    # its sibling's new state. That refresh happened for free in the old
    # one-edit-at-a-time sequential flow (each edit's own cascade pass reached
    # it, or its own later direct edit reloaded the sibling's already-cascaded
    # background from the file) -- reproduce it here by adding back any edited
    # window reachable from the *rest* of the edited set. A lone edit's
    # `edited_wids` has nothing left after removing itself, so this is a no-op
    # for the single-window call the interactive verbs make.
    edited_set = {int(w) for w in edited_wids}
    for w in edited_set:
        if w in _cascade_closure(sorted(edited_set - {w}), succs):
            closure.add(w)
    if not closure:
        return []
    preds: Dict[int, set] = {w: set() for w in fit_map}
    for primary, deps in succs.items():
        for dep in deps:
            preds[dep].add(primary)
    ordered = _cascade_topo(closure, preds)

    cascaded: List[int] = []
    for d in ordered:
        wf = fit_map.get(d)
        fit_win = fit_window_map.get(d)
        if wf is None or fit_win is None:
            continue
        _refresh_frozen_window_level(wf, fit_window_map, fit_map, min_freeze_snr)
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


def refit_window_impl(
    file_path: Union[Path, str],
    window_id: int,
    *,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    add_seeds: Optional[List[ModelPeak]] = None,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
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

    Runs as a batch of one through the shared engine (:func:`_run_single_action`
    -> :func:`_batch_apply_edit_action`), so it enforces the epoch gate, takes
    the undo baseline, cascades and persists by exactly the same code a curation
    file's ``edit`` row does.

    Identity refit (``add=()`` and ``remove=()``) reproduces the persisted fit
    to ~1e-5 MHz on a window whose peaks are a single-window optimum.  A window
    that was subject to a **thaw co-fit** holds peaks at a *two-window* joint
    optimum; a single-window NLS relaxes those toward the one-window optimum, so
    such windows reproduce only to ~kHz (the neighbor's data is intentionally
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
        frequency.  User-added peaks carry ``origin="user"`` and are stamped
        with the ``order_index`` of the decision that added them
        (:attr:`~ftmwpipeline.core.data_structures.FittedPeak.derivation`).
        The post-snap frequency must lie inside the window's own
        ``freq_range``; a frequency no window covers needs
        :func:`create_window_impl` first.
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
        ledger candidate.  Defaults to :data:`REFIT_SNAP_TOL_MHZ` (50 kHz).
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
        (both frames -- see :class:`RefitWindowResult`).

    Raises
    ------
    ValueError
        When Stage 5 has not been run, the ``window_id`` is not found,
        ``len(add_seeds) != len(add)``, any ``remove`` frequency does not
        match a fitted peak within ``snap_tol_mhz``, any ``add`` frequency
        falls outside the named window's ``freq_range`` after snapping, or
        ``frame`` is omitted on a ``self_calibrated`` file with a non-empty
        ``add``/``remove``.
    """
    # Checked before the batch opens so a malformed call cannot even take the
    # undo baseline: a refused edit must leave the file untouched. The applier
    # re-checks, since a curation row reaches it without passing through here.
    _check_add_seeds_arity(add, add_seeds)
    path = str(file_path)
    add_raw, remove_raw = list(add), list(remove)
    if add_raw or remove_raw:
        resolved_frame, stamp = _resolve_frame(path, frame)
        add_raw = [_frame_to_raw(f, frame=resolved_frame, stamp=stamp) for f in add_raw]
        remove_raw = [
            _frame_to_raw(f, frame=resolved_frame, stamp=stamp) for f in remove_raw
        ]
    return _run_single_action(
        path,
        lambda ctx: _batch_apply_edit_action(
            ctx,
            window_id,
            add_raw,
            remove_raw,
            add_seeds=add_seeds,
            snap_tol_mhz=snap_tol_mhz,
        ),
        snap_tol_mhz=snap_tol_mhz,
        shared=_shared,
    )


# ---------------------------------------------------------------------------
# merge_peaks_impl: collapse ≥2 fitted peaks into one
# ---------------------------------------------------------------------------


def merge_peaks_impl(
    file_path: Union[Path, str],
    window_id: int,
    peaks: Sequence[float],
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> RefitWindowResult:
    """Collapse ≥2 fitted peaks in a window into a single peak.

    A thin composition over :func:`refit_window_impl`: removes the named peaks
    and adds one replacement seeded at their SNR-weighted centroid (or
    amplitude-weighted centroid when SNR is unavailable).  All products carry
    ``origin="user"`` and the ``order_index`` of the single ``"merge"``
    decision this records (:attr:`FittedPeak.derivation`) -- a merge replaces
    identities rather than remeasuring them, which is exactly what the tag
    tells a downstream consumer.

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
    resolved_frame, stamp = _resolve_frame(path, frame)
    peaks_raw = [_frame_to_raw(f, frame=resolved_frame, stamp=stamp) for f in peaks]
    return _run_single_action(
        path,
        lambda ctx: _batch_apply_merge(
            ctx, window_id, peaks_raw, snap_tol_mhz=snap_tol_mhz
        ),
        snap_tol_mhz=snap_tol_mhz,
        shared=_shared,
    )


# ---------------------------------------------------------------------------
# split_peak_impl: replace one fitted peak with K peaks
# ---------------------------------------------------------------------------


def split_peak_impl(
    file_path: Union[Path, str],
    window_id: int,
    peak: float,
    *,
    into: int = 2,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> RefitWindowResult:
    """Replace one fitted peak with ``into`` peaks (default 2).

    A thin composition over :func:`refit_window_impl`: removes the named peak
    and adds ``into`` replacements spread symmetrically about it by ±½ of one
    Fourier resolution element (``1 / acquisition_us`` MHz).  All products
    carry ``origin="user"`` and the ``order_index`` of the single ``"split"``
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
        nearest fitted peak within ``snap_tol_mhz``.
    into :
        Number of replacement peaks (≥2).  Default is 2.
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping.
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
    resolved_frame, stamp = _resolve_frame(path, frame)
    peak_raw = _frame_to_raw(peak, frame=resolved_frame, stamp=stamp)
    return _run_single_action(
        path,
        lambda ctx: _batch_apply_split(
            ctx, window_id, peak_raw, into, snap_tol_mhz=snap_tol_mhz
        ),
        snap_tol_mhz=snap_tol_mhz,
        shared=_shared,
    )


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
        sideband = Sideband.coerce(fid.sideband)

        merged_window_freqs = _auto_merged_window_freqs(spectrum_fit)
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
                auto_merged=window_id in merged_window_freqs,
                merged_freqs=merged_window_freqs.get(window_id, ()),
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
        created_windows=list(existing_review.created_windows),
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
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
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
    frame :
        The frame ``candidate_freq`` is expressed in (see
        :func:`refit_window_impl`). Irrelevant, and never validated, when
        ``candidate_freq`` is ``None`` -- a bare accept carries no frequency.
        Omitting it while ``candidate_freq`` is given is an error on a
        ``self_calibrated`` file.
    _shared :
        Internal. See :func:`refit_window_impl`. Irrelevant when
        ``candidate_freq`` is ``None`` -- a bare accept never opens the batch
        engine at all.

    Returns
    -------
    RefitWindowResult or None
        ``None`` when accepting as-is; the refit result when ``candidate_freq``
        was given (both frames -- see :class:`RefitWindowResult`).
    """
    if candidate_freq is None:
        # A bare accept marks the window reviewed and changes no fitted number,
        # so it deliberately does NOT open a batch: it needs no fit context (the
        # expensive part), takes no undo baseline, and is not a splice, so the
        # epoch gate does not apply to it. No frequency is carried, so `frame`
        # is moot and is never resolved/validated here.
        _record_bare_accept(str(file_path), window_id)
        return None

    path = str(file_path)
    resolved_frame, stamp = _resolve_frame(path, frame)
    candidate_raw = _frame_to_raw(candidate_freq, frame=resolved_frame, stamp=stamp)
    return _run_single_action(
        path,
        lambda ctx: _batch_apply_accept(
            ctx, window_id, candidate_raw, snap_tol_mhz=snap_tol_mhz
        ),
        snap_tol_mhz=snap_tol_mhz,
        shared=_shared,
    )


def _record_bare_accept(path: str, window_id: int) -> None:
    """Mark one window reviewed: a decision entry and a provenance flip, nothing
    else. Shared by :func:`review_accept_impl` and a bare ``accept`` row in a
    curation file, which is why it is not folded into the batch engine."""
    # A representative anchor frequency for the log entry (best effort: the
    # entry is a marker, and a window with no fit still accepts).
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

    with h5py.File(path, "r") as h5f:
        existing_review: Stage6Review = (
            load_stage6_review_from_hdf5(h5f["stage6_review"])
            if "stage6_review" in h5f
            else Stage6Review()
        )

    existing_status = existing_review.window_statuses.get(window_id)
    kept_reasons: List[AttentionReason] = (
        list(existing_status.attention_reasons) if existing_status is not None else []
    )

    new_statuses = dict(existing_review.window_statuses)
    new_statuses[window_id] = WindowReviewStatus(
        window_id=window_id,
        provenance="reviewed",
        attention_reasons=kept_reasons,
        invalidated=False,
    )
    # Accepting as-is does not change the fit, but it can still be the first
    # write since a timebase re-run: the persisted table's calibration stamp
    # may no longer match the file's current calibration, and carrying it
    # forward unchanged would re-persist a stale table (this is exactly the
    # bug -- a bare accept must not launder it back to disk).
    final_products = _current_final_products(existing_review.final_products, path)
    new_review = Stage6Review(
        window_statuses=new_statuses,
        decision_log=list(existing_review.decision_log)
        + [
            DecisionLogEntry(
                order_index=len(existing_review.decision_log),
                window_id=window_id,
                frequency_mhz=anchor_freq,
                kind="accept",
                provenance="user",
                evidence={},
            )
        ],
        final_products=final_products,
        created_windows=list(existing_review.created_windows),
    )

    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(new_review, grp)


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


def _frozen_parameters_from_sources(
    depends_on: Sequence[int],
    fit_map: Dict[int, FittingResult],
    min_freeze_snr: float,
) -> Dict[str, Dict]:
    """``fixed_parameters`` for a new window, read off its sources' current fits.

    Same window-level resolution the cascade uses (``_refresh_frozen_window_level``):
    every source-window line clearing ``min_freeze_snr``, at its *fitted*
    parameters. Reading the fit rather than the Stage 3 detections is what keeps
    a source whose fit collapsed several detections into one line from being
    frozen more than once.
    """
    frozen: List[Dict] = []
    for primary in sorted(set(int(p) for p in depends_on)):
        pwf = fit_map.get(primary)
        if pwf is None:
            continue
        for pk in sorted(pwf.fitted_peaks, key=lambda q: float(q.frequency_mhz)):
            if float(pk.snr or 0.0) < min_freeze_snr:
                continue
            frozen.append(
                {
                    "peak_index": -1,
                    "primary_window_id": primary,
                    "frequency_mhz": float(pk.frequency_mhz),
                    "amplitude": float(pk.amplitude),
                    "phase": float(pk.phase) if pk.phase is not None else 0.0,
                    "freeze_eligible": True,
                }
            )
    return {f"frozen_peak_{i}": e for i, e in enumerate(frozen)}


def create_window_impl(
    file_path: Union[Path, str],
    anchor_mhz: float,
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    frame: Optional[Frame] = None,
    _replay_window_id: Optional[int] = None,
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
    which is a pure function of the anchor and the *base* plan -- never of the
    current curated state -- so replaying an edit set in ``order_index`` order
    reproduces the same window. Ids are only ever appended: an existing window
    is never renumbered, so a consumer partitioning peaks on ``window_id`` sees
    exactly the windows the edit touched.

    **No cascade, in either direction.** The new window reads its neighbours'
    frozen leakage skirts inward; it contributes no outward dependency edge and
    no neighbour is re-fit or thawed. A window created for a line the automatic
    pass missed holds, by construction, a line below the freeze bar, whose own
    leakage into its neighbours is negligible -- which is what makes the whole
    operation purely additive in the stage DAG.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    anchor_mhz :
        Molecular frequency (MHz) the new window must cover.
    snap_tol_mhz :
        Frequency-snapping tolerance forwarded to the fit core (unused by the
        empty-peak-set fit; kept for signature parity with the other verbs).
    frame :
        The frame ``anchor_mhz`` is expressed in (see
        :func:`refit_window_impl`). Omitting it is an error on a
        ``self_calibrated`` file.
    _replay_window_id :
        Internal. The id this create produced the first time round, supplied
        when the decision log is replayed so the window keeps its identity even
        if an earlier create was dropped from the edit set. The *geometry* is
        still re-derived; only the label is replayed, and a structural change
        that makes the label wrong raises before anything is written.
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
    resolved_frame, stamp = _resolve_frame(path, frame)
    anchor_raw = _frame_to_raw(anchor_mhz, frame=resolved_frame, stamp=stamp)
    return _run_single_action(
        path,
        lambda ctx: _batch_apply_create(
            ctx,
            anchor_raw,
            replay_window_id=_replay_window_id,
            snap_tol_mhz=snap_tol_mhz,
        ),
        snap_tol_mhz=snap_tol_mhz,
        shared=_shared,
    )


# ---------------------------------------------------------------------------
# Curation files: a human-editable batch language for the review edits.
#
# A curation file (CSV) records an ordered sequence of Stage 6 edits, one per
# row, that ``apply_curation_impl`` replays through the same edit impls the
# interactive verbs call. The CSV is the canonical interchange: diffable,
# hand-editable, and independent of the report that may have authored it.
# ---------------------------------------------------------------------------

_CURATION_ACTIONS = ("add", "remove", "merge", "split", "accept", "create")
_CURATION_HEADER = ("action", "window", "freqs", "params")

# ``create`` is the one action whose window id is normally an *output*, not an
# input: it installs the window the anchor needs. A hand-authored file writes one
# of these tokens in the window column to say "whichever id this turns out to
# be". A named id instead *pins* the id the create takes -- which is what a file
# generated from the decision log writes, so the window keeps its identity
# across a replay even if an earlier create was dropped from the edit set.
_CURATION_NEW_WINDOW_TOKENS = ("new", "auto", "-", "")
_NEW_WINDOW_SENTINEL = -1


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
    """

    kind: str
    window_id: int
    add: List[float] = field(default_factory=list)
    remove: List[float] = field(default_factory=list)
    peaks: List[float] = field(default_factory=list)
    peak: Optional[float] = None
    into: int = 2
    candidate: Optional[float] = None
    anchor: Optional[float] = None
    """``create`` only: the molecular frequency (MHz) the new window must cover.
    Its ``window_id`` is :data:`_NEW_WINDOW_SENTINEL` when the source did not
    name one, and otherwise the id the action is expected to produce."""


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
    base_changed: bool = False


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


def parse_curation_file(curation_path: Union[Path, str]) -> ParsedCurationFile:
    """Parse a curation CSV into ordered :class:`CurationOp` rows, plus its
    optional file-level :class:`CurationFileHeader` (A3).

    Columns are ``action,window,freqs,params``. Blank lines and ``#`` comments
    are ignored; an optional header row (first cell ``action``) is skipped.
    ``freqs`` is a ``;``-separated list of molecular MHz; ``params`` is a
    ``;``-separated list of ``key=value`` modifiers.

    Two ``#``-comment directives are recognized anywhere in the file and
    collected onto the returned :class:`ParsedCurationFile`'s ``.header``
    rather than being treated as ordinary comments: ``# frame: raw`` /
    ``# frame: calibrated`` declares the frame every frequency in the file is
    expressed in, and ``# epsilon: <value>`` (only valid alongside
    ``frame: calibrated``) stamps the epsilon the file was written under.
    Every other ``#``-prefixed line is an ordinary, ignored comment. See
    :func:`_resolve_curation_frame` for how the header interacts with the
    per-call ``frame`` argument.

    Raises ``ValueError`` (with the 1-based source line) on any malformed row
    or header directive.
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
                    raise ValueError(
                        f"curation line {line_no}: 'frame' header must be "
                        f"'raw' or 'calibrated', got {raw_value!r}"
                    )
                if header.frame is not None and header.frame != value:
                    raise ValueError(
                        f"curation line {line_no}: conflicting 'frame' "
                        f"header (already declared {header.frame!r} earlier "
                        f"in this file)"
                    )
                header.frame = value  # type: ignore[assignment]
                continue
            m = _CURATION_EPSILON_HEADER_RE.match(line)
            if m is not None:
                raw_eps = m.group(1).strip()
                try:
                    eps_value = float(raw_eps)
                except ValueError:
                    raise ValueError(
                        f"curation line {line_no}: 'epsilon' header "
                        f"{raw_eps!r} is not a number"
                    ) from None
                if header.epsilon is not None and header.epsilon != eps_value:
                    raise ValueError(
                        f"curation line {line_no}: conflicting 'epsilon' "
                        f"header (already declared {header.epsilon!r} "
                        f"earlier in this file)"
                    )
                header.epsilon = eps_value
                continue
            continue  # an ordinary comment
        fields = [f.strip() for f in line.split(",")]
        action = fields[0].lower()
        if action == "action":  # header row
            continue
        if action not in _CURATION_ACTIONS:
            raise ValueError(
                f"curation line {line_no}: unknown action {fields[0]!r}; "
                f"choose one of {_CURATION_ACTIONS}"
            )
        raw_window = fields[1] if len(fields) > 1 else ""
        if action == "create" and raw_window.strip().lower() in (
            _CURATION_NEW_WINDOW_TOKENS
        ):
            window_id = _NEW_WINDOW_SENTINEL
        elif len(fields) < 2 or not fields[1]:
            raise ValueError(f"curation line {line_no}: missing window id")
        else:
            try:
                window_id = int(fields[1])
            except ValueError:
                raise ValueError(
                    f"curation line {line_no}: window id {fields[1]!r} is not an "
                    f"integer"
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
        elif action == "create":
            if len(freqs) != 1:
                raise ValueError(
                    f"curation line {line_no}: create needs exactly one frequency "
                    f"(the anchor the new window must cover)"
                )
            if params:
                raise ValueError(f"curation line {line_no}: create takes no parameters")
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

    if header.epsilon is not None and header.frame != "calibrated":
        raise ValueError(
            "curation file: an 'epsilon' header requires a 'frame: "
            "calibrated' header alongside it -- an epsilon stamp is "
            "meaningless without a calibrated-frame declaration to attach "
            "it to"
        )
    if header.frame == "calibrated" and header.epsilon is None:
        raise ValueError(
            "curation file: 'frame: calibrated' requires an 'epsilon' "
            "header stamping the epsilon the file was written under (e.g. "
            "'# epsilon: 2.2e-6') -- otherwise a later apply/preview cannot "
            "detect that the calibration has drifted since this file was "
            "staged"
        )

    return ParsedCurationFile(ops, header)


def _resolve_curation_plan(ops: Sequence[CurationOp]) -> List[PlannedAction]:
    """Coalesce parsed ops into the delegated action plan.

    A maximal run of ``add`` / ``remove`` rows on one window collapses into a
    single ``edit`` (one refit instead of one per row); a ``merge`` / ``split``
    / ``accept`` on that window is a barrier that flushes the window's pending
    edit first (it changes the peak set with its own seeding). Windows are
    independent, so an edit on another window does not flush a pending group.

    ``create`` never coalesces: it installs structure the rows after it name, so
    it stands alone and in order.
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
            plan.append(PlannedAction(kind="create", window_id=wid, anchor=op.freqs[0]))
            continue
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
    if action.kind == "create":
        target = "a new window" if wid == _NEW_WINDOW_SENTINEL else f"window {wid}"
        return f"create {target}: anchor {float(action.anchor or 0.0):.4f}"
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


def _planned_window_ranges(path: str) -> Dict[int, Tuple[float, float]]:
    """``freq_range`` of every window in the effective plan, low bound first.

    Best-effort: a file without a Stage 4 plan yields an empty map rather than
    raising, so the advisory pass degrades to the checks it can still make.
    """
    try:
        plan = effective_window_plan(path)
    except Exception:  # pragma: no cover - advisory only, never fatal
        return {}
    return {
        int(w.window_id): (min(w.freq_range), max(w.freq_range)) for w in plan.windows
    }


def _curation_ambiguity_warnings(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
) -> List[str]:
    """Advisories where a curation action will not resolve against the file.

    ``remove`` / ``split`` / ``merge`` match an *existing* fitted peak by nearest
    frequency within ``snap_tol_mhz``; if two peaks sit within tolerance the
    matcher's pick is ambiguous, and if none do the edit will fail.

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
    """
    by_window = _fitted_freqs_by_window(path)
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

    for action in plan:
        wid = action.window_id
        if action.kind == "edit":
            for f in action.remove:
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
    (:func:`~ftmwpipeline.fitting.active_ft.active_ft_bin_spacing_mhz`)."""
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return 0.0
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return float(spectrum_fit.parameters.get("acquisition_us", 0.0))


def _frame_mismatch_warnings(
    path: str,
    plan: Sequence[PlannedAction],
    *,
    resolved_frame: Frame,
    stamp: Optional[_CalibrationStamp],
) -> List[str]:
    """Advisory-only diagnostic (A5): flag a batch whose candidates all
    resolve with a residual consistent with a calibrated-frame curation file
    that was declared (or defaulted to) raw.

    Never raises and never blocks anything -- this is a heuristic, and a
    heuristic that refused would be worse than none. Returns ``[]`` (inert)
    unless ALL of the following hold:

    - ``resolved_frame == \"raw\"`` -- if the caller correctly declared
      ``calibrated``, the conversion already happened and no systematic
      residual should remain to diagnose;
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
    """
    if resolved_frame != "raw" or stamp is None:
        return []
    cal_state, epsilon, _sigma_eps, _floor_khz, probe_freq_mhz, _sideband = stamp
    if cal_state != "self_calibrated" or epsilon == 0.0:
        return []

    acquisition_us = _persisted_acquisition_us(path)
    frame_mismatch_floor_mhz = (
        _FRAME_MISMATCH_FLOOR_BINS * active_ft_bin_spacing_mhz(acquisition_us)
        if acquisition_us > 0.0
        else 0.0
    )

    by_window = _fitted_freqs_by_window(path)

    def nearest(wid: int, freq: float) -> Optional[float]:
        fitted = by_window.get(wid)
        if not fitted:
            return None
        best = min(fitted, key=lambda f: abs(f - freq))
        if abs(best - freq) > REFIT_SNAP_TOL_MHZ:
            return None
        return best

    residuals: List[float] = []
    predicted: List[float] = []
    for action in plan:
        wid = action.window_id
        targets: List[float] = []
        if action.kind == "edit":
            targets.extend(action.remove)
        elif action.kind == "merge":
            targets.extend(action.peaks)
        elif action.kind == "split" and action.peak is not None:
            targets.append(action.peak)
        elif action.kind == "accept" and action.candidate is not None:
            targets.append(action.candidate)
        for f in targets:
            match = nearest(wid, f)
            if match is None:
                continue
            residuals.append(f - match)
            predicted.append((probe_freq_mhz - match) * epsilon / (1.0 + epsilon))

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
    return [
        f"{len(residuals)} candidate(s) in this batch resolved with a "
        f"residual clustered near {mean_residual_khz:+.1f} kHz -- matching "
        f"what this file's epsilon ({epsilon * 1e6:+.3f} ppm) predicts for a "
        f"calibrated frequency submitted as raw. The curation file may have "
        f"been staged in the calibrated frame but declared (or defaulted to) "
        f"raw; double check its frame before trusting this batch."
    ]


# --- the automatic-fit baseline (for undo replay) --------------------------
#
# A user edit mutates ``/stage5_fitting`` in place, so the automatic fit it
# replaced is otherwise unrecoverable. Before the first edit we snapshot the
# automatic fit into ``/stage5_fitting_baseline``; ``review_undo_impl`` restores
# it and replays the surviving decisions onto it. A fresh Stage 5 fit drops the
# snapshot (``clear_stage5_baseline``) so the next edit re-snapshots.

STAGE5_BASELINE_GROUP = "stage5_fitting_baseline"
# Decision kinds that mutate ``/stage5_fitting`` and therefore need the
# automatic-fit baseline to be undoable. ``create_window`` belongs here: it
# splices a window fit into the persisted SpectrumFit (and, on the widening
# path, re-fits an existing window over a grown extent).
_FIT_EDIT_KINDS = ("add", "remove", "merge", "split", "create_window")


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
    """Dispatch one resolved action to its edit impl (shared by apply + undo).

    Every frequency on ``action`` is already raw: a curation file's
    frequencies are converted to raw by :func:`apply_curation_impl` before a
    :class:`PlannedAction` is built, and a decision-log replay
    (:func:`_decision_to_op`) reads frequencies straight from the log, which
    is raw by construction (see ``core.curation.Frame``). ``frame="raw"`` is
    passed explicitly rather than left to default, so a self_calibrated
    file's omitted-frame error can never fire on a replay.
    """
    if action.kind == "create":
        if action.anchor is None:
            raise ValueError("create action requires an anchor frequency")
        # A named window id is the id the create takes, not merely a value to
        # check afterwards: every following row -- and every peak's
        # ``window_id`` -- is keyed on it, so it has to survive the replay
        # intact even when an earlier create was dropped from the edit set.
        create_window_impl(
            path,
            action.anchor,
            frame="raw",
            _replay_window_id=(
                None if action.window_id == _NEW_WINDOW_SENTINEL else action.window_id
            ),
        )
        return
    if action.kind == "edit":
        refit_window_impl(
            path,
            action.window_id,
            add=action.add,
            remove=action.remove,
            frame="raw",
        )
    elif action.kind == "merge":
        merge_peaks_impl(path, action.window_id, action.peaks, frame="raw")
    elif action.kind == "split":
        if action.peak is None:
            raise ValueError("split action requires a peak frequency")
        split_peak_impl(
            path, action.window_id, action.peak, into=action.into, frame="raw"
        )
    elif action.kind == "accept":
        review_accept_impl(
            path, action.window_id, candidate_freq=action.candidate, frame="raw"
        )


# ---------------------------------------------------------------------------
# Batch curation engine: import once, apply the whole changeset in memory,
# cascade once, persist once.
#
# ``_execute_planned_action`` above (and the single-window verbs it dispatches
# to) each independently reload the FID, rebuild the active-FT context, and
# read-modify-write ``/stage5_fitting`` -- fine for one interactive edit, but
# for a curation batch of N actions that is N redundant imports/FTs/persists
# for work that is batch-invariant (see ``_build_batch_ctx``). The engine below
# builds that shared state ONCE, applies every action against an in-memory
# ``SpectrumFit``, runs a single combined cascade over every directly-edited
# window, and persists ``/stage5_fitting`` and ``/stage6_review`` once each.
#
# Ordering: the final persisted state must not depend on the order actions
# were listed in the curation file (or the decision log, for undo's replay).
# Cross-window order is canonicalized -- creates first (in their own relative
# order, since they install structure later rows name), then every other
# action grouped by ascending window id -- while the intra-window sequence
# ``_resolve_curation_plan`` already coalesced is preserved exactly (a stable
# sort by window id cannot reorder two actions that share one). This is an
# equivalence of OUTCOME, not of execution: we do not assert that edits on
# different windows compose independently peak-by-peak, only that "edit 1 then
# edit 2" and "edit 2 then edit 1" reach the same final state, because both are
# actually applied in this one deterministic order. Decisions are appended to
# the log in this same canonical order, so the log itself is order-of-
# specification-independent too.
# ---------------------------------------------------------------------------


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


@dataclass
class _BatchChangeset:
    """The in-progress mutable state of one curation batch: the working
    ``spectrum_fit`` (reloaded fresh every batch -- it changes on every
    apply, so it is never amortized across batches), the dirty/mutated window
    sets, and the pending decision-log entries.

    ``created_windows`` and ``fit_window_map`` live here rather than on
    :class:`_SharedFitCtx` even though a fresh build derives their starting
    value from the file (``base_plan`` overlaid with the persisted review's
    ``created_windows``): a ``create`` action within the batch appends to
    ``created_windows`` and recomputes ``fit_window_map`` in place
    (``_batch_apply_create``), and the next batch must see whatever the
    previous one persisted, so both have to be reloaded per batch exactly
    like ``spectrum_fit``.
    """

    spectrum_fit: SpectrumFit
    created_windows: List["FitWindow"]
    fit_window_map: Dict[int, "FitWindow"] = field(default_factory=dict)
    dirty_wids: set = field(default_factory=set)
    """Windows a *direct* edit (add/remove/merge/split/accept-candidate)
    touched this batch -- the seed set for the one combined cascade."""
    mutated_wids: set = field(default_factory=set)
    """Every window whose ``FittingResult`` changed this batch (dirty windows,
    newly-created windows, and cascaded dependents) -- drives the
    final-products / attention-reason refresh."""
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    """Pending decision-log entries, in the order they will be recorded
    (``window_id`` / ``frequency_mhz`` / ``kind`` / ``evidence`` / ``bare``)."""
    next_decision_index: int = 0


@dataclass
class _BatchCtx:
    """One batch's full working state: the (possibly reused) shared context
    plus this batch's own changeset. Every appliers/cascade/persist helper
    below still addresses fields through ``ctx.shared.*`` /
    ``ctx.changeset.*`` -- the split is deliberately visible at every call
    site, since a later unit needs to build one ``_SharedFitCtx`` and reuse it
    across many ``_BatchChangeset``s.
    """

    shared: _SharedFitCtx
    changeset: _BatchChangeset
    baseline_taken: bool = True
    """Whether :func:`_open_batch` took the undo baseline snapshot for this
    context. ``False`` only for a context built via
    ``_open_batch(..., snapshot=False)`` -- a preview. :func:`_finish_batch`
    refuses to persist such a context: writing fit-mutating edits without
    ever having taken the baseline would silently break ``review undo`` for
    the file (a later edit would then snapshot an already-edited fit as if
    it were the automatic one). Structural, not a remembered convention --
    see ``test_engine_invariants.py``'s docstring on why this module prefers
    guards enforced by structure.
    """


def _build_shared_fit_ctx(path: str) -> _SharedFitCtx:
    """Load and resolve everything every batch's fit-mutating actions share:
    settings, calibration, and the active-FT context
    (:func:`~.stage5_impl.build_stage5_fit_context`, the expensive
    FID-load-and-FT step this whole engine exists to amortize). Safe to build
    once and reuse across many batches -- nothing it returns depends on any
    batch's edits.
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
    from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
    from .stage3_impl import load_peaks_impl
    from .stage4_impl import load_windows_impl
    from .stage5_impl import (
        Stage5FitContext,
        _resolve_tau_calibration_for_fit,
        build_stage5_fit_context,
    )

    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        # Loaded only to seed the spur-catalog replay below -- deliberately
        # not returned. ``spectrum_fit`` is per-batch state (see
        # ``_BatchChangeset``); the spur catalog in its ``parameters`` is a
        # Stage 5 product that Stage 6 never rewrites, so reading it here,
        # once, is not a staleness risk the way retaining the fit would be.
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    base_plan: "WindowPlan" = load_windows_impl(path)["plan"]

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
        replay_spur_catalog=spectrum_fit.parameters,
    )

    min_freeze_snr = float(
        base_plan.parameters.get("min_freeze_snr", DEFAULT_MIN_FREEZE_SNR)
    )

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
    )


def _build_batch_changeset(path: str, shared: _SharedFitCtx) -> _BatchChangeset:
    """Load-or-reset everything one batch's actions accumulate: the current
    ``spectrum_fit`` (reloaded fresh -- never reused across batches), the
    review's ``created_windows`` overlay and the ``fit_window_map`` derived
    from it, and the next decision-log index. Called once per batch,
    regardless of whether ``shared`` was just built or is being reused from an
    earlier batch.
    """
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    review = load_stage6_review_from_file(path)
    created_windows = list(review.created_windows)
    effective_plan = _overlay_created_windows(shared.base_plan, created_windows)
    fit_window_map = {w.window_id: w for w in effective_plan.windows}

    return _BatchChangeset(
        spectrum_fit=spectrum_fit,
        created_windows=created_windows,
        fit_window_map=fit_window_map,
        next_decision_index=len(review.decision_log),
    )


def _build_batch_ctx(
    path: str, *, snap_tol_mhz: float, shared: Optional[_SharedFitCtx] = None
) -> _BatchCtx:
    """Build one batch's full working context.

    Load-or-accept the shared, batch-invariant half (build it fresh unless a
    caller already has one -- a later unit reuses one ``_SharedFitCtx`` across
    many batches in a review session), then always build a brand new
    changeset: none of the changeset is safe to reuse across batches, even
    when the shared context is (see ``_BatchChangeset``).
    """
    if shared is None:
        shared = _build_shared_fit_ctx(path)
    changeset = _build_batch_changeset(path, shared)
    return _BatchCtx(shared=shared, changeset=changeset)


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
        raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")
    return wf_list[0]


def _batch_apply_edit_core(
    ctx: _BatchCtx,
    window_id: int,
    *,
    add: Sequence[float],
    remove: Sequence[float],
    add_seeds: Optional[List[ModelPeak]] = None,
    add_derivations: Optional[Sequence[Optional[int]]] = None,
    snap_tol_mhz: float,
) -> FittingResult:
    """In-memory equivalent of the fit-mutating middle of
    :func:`refit_window_impl` (materialize -> NLS -> splice), reusing the
    batch's shared context instead of rebuilding it. Marks ``window_id`` dirty."""
    from ..fitting.result_conversion import sort_fitting_result_by_frequency

    wf = _batch_lookup_wf(ctx, window_id)
    fit_win = ctx.changeset.fit_window_map.get(window_id)
    if fit_win is None:
        raise KeyError(
            f"window_id={window_id} not found in the Stage 4 WindowPlan. "
            "Stage 4 may have been re-run and changed the window geometry."
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
    )
    sort_fitting_result_by_frequency(new_wf)
    _splice_edit_result(ctx.changeset.spectrum_fit, window_id, new_wf)
    # An identity refit (no add/remove/seeds) re-converges this window but does
    # not change its peak set, so it does not move the leakage skirt its
    # dependents froze: it is mutated (its own numbers shifted, so the
    # final-products table has to be rebuilt) but not dirty (nothing to cascade).
    ctx.changeset.mutated_wids.add(window_id)
    if add or remove or add_seeds:
        ctx.changeset.dirty_wids.add(window_id)
    return new_wf


def _batch_apply_edit_action(
    ctx: _BatchCtx,
    window_id: int,
    add: Sequence[float],
    remove: Sequence[float],
    *,
    add_seeds: Optional[List[ModelPeak]] = None,
    record_decisions: bool = True,
    snap_tol_mhz: float,
) -> RefitWindowResult:
    """The add/remove edit, as one action: refit, then one decision per
    frequency (adds, then removes), all sharing one evidence dict.

    The single implementation behind both ``review edit`` and an ``edit`` row in
    a curation file. ``record_decisions=False`` suppresses the per-frequency
    entries for a composing caller (merge / split) that records its own coarser
    one; the peaks are still stamped with the composing decision's id through
    ``ctx.changeset.next_decision_index``.
    """
    _check_add_seeds_arity(add, add_seeds)

    wf = _batch_lookup_wf(ctx, window_id)
    chi2r_before = float(wf.reduced_chi2)
    n_before = len(wf.fitted_peaks)

    add_derivations: Optional[List[Optional[int]]] = None
    if add:
        base_idx = ctx.changeset.next_decision_index
        add_derivations = [base_idx + i for i in range(len(add))]

    new_wf = _batch_apply_edit_core(
        ctx,
        window_id,
        add=add,
        remove=remove,
        add_seeds=add_seeds,
        add_derivations=add_derivations,
        snap_tol_mhz=snap_tol_mhz,
    )

    chi2r_after = float(new_wf.reduced_chi2)
    n_after = len(new_wf.fitted_peaks)
    evidence: Dict[str, object] = {
        "chi2r_before": chi2r_before,
        "chi2r_after": chi2r_after,
        "n_peaks_before": n_before,
        "n_peaks_after": n_after,
    }
    if record_decisions:
        for f in add:
            ctx.changeset.decisions.append(
                {
                    "window_id": window_id,
                    "frequency_mhz": float(f),
                    "kind": "add",
                    "evidence": evidence,
                }
            )
            ctx.changeset.next_decision_index += 1
        for f in remove:
            ctx.changeset.decisions.append(
                {
                    "window_id": window_id,
                    "frequency_mhz": float(f),
                    "kind": "remove",
                    "evidence": evidence,
                }
            )
            ctx.changeset.next_decision_index += 1

    return _make_refit_result(
        ctx,
        window_id=window_id,
        n_peaks_before=n_before,
        n_peaks_after=n_after,
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=list(new_wf.fitted_peaks),
    )


def _batch_apply_merge(
    ctx: _BatchCtx,
    window_id: int,
    peaks: Sequence[float],
    *,
    snap_tol_mhz: float,
) -> RefitWindowResult:
    """Collapse >= 2 fitted peaks into one, as one action.

    The single implementation behind both ``review merge`` and a ``merge`` row
    in a curation file. Sideband and window center come from the shared
    ``fit_ctx`` and the window's own persisted geometry, so this never needs a
    FID load of its own.
    """
    _check_merge_arity(peaks)
    wf = _batch_lookup_wf(ctx, window_id)

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
                f"{req_freq_f:.4f} MHz (closest distance: {best_dist:.4f} MHz)"
            )
        if any(m.frequency_mhz == best.frequency_mhz for m in matched):
            raise ValueError(
                f"merge: frequency {req_freq_f:.4f} MHz matched the same "
                f"fitted peak twice"
            )
        matched.append(best)

    weights: List[float] = []
    for fp in matched:
        w = float(fp.snr) if fp.snr is not None else float(fp.amplitude)
        weights.append(max(w, 1e-30))
    total_w = sum(weights)
    centroid_freq = (
        sum(float(fp.frequency_mhz) * w for fp, w in zip(matched, weights)) / total_w
    )
    centroid_amp = sum(float(fp.amplitude) for fp in matched)

    merge_freq = centroid_freq
    merge_amp = centroid_amp
    if len(matched) == 2:
        fa = float(matched[0].frequency_mhz)
        fb = float(matched[1].frequency_mhz)
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
                merge_freq = float(da.merged_frequency_mhz)
                merge_amp = (
                    float(da.merged_amplitude)
                    if not math.isnan(float(da.merged_amplitude))
                    else centroid_amp
                )
                break

    remove_freqs = [float(fp.frequency_mhz) for fp in matched]

    sideband = ctx.shared.fit_ctx.sideband
    s = sideband_sign(sideband)
    center_mhz: Optional[float] = None
    if wf.window is not None and wf.window.freq_range is not None:
        lo, hi = wf.window.freq_range
        center_mhz = (lo + hi) / 2.0

    add_seeds: Optional[List[ModelPeak]] = None
    if center_mhz is not None:
        merge_offset = float(s * (merge_freq - center_mhz))
        add_seeds = [
            ModelPeak(
                amplitude=max(merge_amp, 1e-30), offset_mhz=merge_offset, phase=0.0
            )
        ]

    chi2r_before = float(wf.reduced_chi2)
    n_before = len(wf.fitted_peaks)
    idx = ctx.changeset.next_decision_index
    new_wf = _batch_apply_edit_core(
        ctx,
        window_id,
        add=[merge_freq],
        remove=remove_freqs,
        add_seeds=add_seeds,
        add_derivations=[idx],
        snap_tol_mhz=snap_tol_mhz,
    )
    chi2r_after = float(new_wf.reduced_chi2)
    n_after = len(new_wf.fitted_peaks)
    evidence: Dict[str, object] = {
        "chi2r_before": chi2r_before,
        "chi2r_after": chi2r_after,
        "n_peaks_before": n_before,
        "n_peaks_after": n_after,
        "merged_from": [float(f) for f in peaks],
    }
    ctx.changeset.decisions.append(
        {
            "window_id": window_id,
            "frequency_mhz": merge_freq,
            "kind": "merge",
            "evidence": evidence,
        }
    )
    ctx.changeset.next_decision_index += 1
    return _make_refit_result(
        ctx,
        window_id=window_id,
        n_peaks_before=n_before,
        n_peaks_after=n_after,
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=list(new_wf.fitted_peaks),
    )


def _batch_apply_split(
    ctx: _BatchCtx,
    window_id: int,
    peak: float,
    into: int,
    *,
    snap_tol_mhz: float,
) -> RefitWindowResult:
    """Replace one fitted peak with ``into`` peaks, as one action.

    The single implementation behind both ``review split`` and a ``split`` row
    in a curation file. The resolution element uses the shared
    ``fit_ctx.acquisition_us`` rather than a fresh FID load.
    """
    _check_split_arity(into)
    wf = _batch_lookup_wf(ctx, window_id)

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

    acquisition_us = float(ctx.shared.fit_ctx.acquisition_us)
    resolution_mhz = 1.0 / acquisition_us if acquisition_us > 0 else 0.1

    if into == 2:
        offsets = [-0.5 * resolution_mhz, 0.5 * resolution_mhz]
    else:
        half_span = 0.5 * resolution_mhz
        offsets = [-half_span + i * resolution_mhz / (into - 1) for i in range(into)]

    add_freqs = [matched_freq + off for off in offsets]
    per_peak_amp = matched_amp / into

    sideband = ctx.shared.fit_ctx.sideband
    s = sideband_sign(sideband)
    center_mhz: Optional[float] = None
    if wf.window is not None and wf.window.freq_range is not None:
        lo, hi = wf.window.freq_range
        if lo > hi:
            lo, hi = hi, lo
        center_mhz = (lo + hi) / 2.0
        add_freqs = [min(max(af, lo), hi) for af in add_freqs]

    add_seeds: Optional[List[ModelPeak]] = None
    if center_mhz is not None:
        add_seeds = [
            ModelPeak(
                amplitude=max(per_peak_amp, 1e-30),
                offset_mhz=float(s * (af - center_mhz)),
                phase=0.0,
            )
            for af in add_freqs
        ]

    chi2r_before = float(wf.reduced_chi2)
    n_before = len(wf.fitted_peaks)
    idx = ctx.changeset.next_decision_index
    new_wf = _batch_apply_edit_core(
        ctx,
        window_id,
        add=add_freqs,
        remove=[matched_freq],
        add_seeds=add_seeds,
        add_derivations=[idx] * len(add_freqs),
        snap_tol_mhz=snap_tol_mhz,
    )
    chi2r_after = float(new_wf.reduced_chi2)
    n_after = len(new_wf.fitted_peaks)
    evidence: Dict[str, object] = {
        "chi2r_before": chi2r_before,
        "chi2r_after": chi2r_after,
        "n_peaks_before": n_before,
        "n_peaks_after": n_after,
        "split_into": into,
    }
    ctx.changeset.decisions.append(
        {
            "window_id": window_id,
            "frequency_mhz": matched_freq,
            "kind": "split",
            "evidence": evidence,
        }
    )
    ctx.changeset.next_decision_index += 1
    return _make_refit_result(
        ctx,
        window_id=window_id,
        n_peaks_before=n_before,
        n_peaks_after=n_after,
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=list(new_wf.fitted_peaks),
    )


def _batch_apply_accept(
    ctx: _BatchCtx,
    window_id: int,
    candidate: Optional[float],
    *,
    snap_tol_mhz: float,
) -> Optional[RefitWindowResult]:
    """Batch equivalent of :func:`review_accept_impl`. A bare accept records a
    "reviewed" decision with no fit change (and keeps the window's existing
    attention reasons, matching the single-window impl exactly); an accept with
    a candidate is an add."""
    if candidate is not None:
        wf = _batch_lookup_wf(ctx, window_id)
        chi2r_before = float(wf.reduced_chi2)
        n_before = len(wf.fitted_peaks)
        idx = ctx.changeset.next_decision_index
        new_wf = _batch_apply_edit_core(
            ctx,
            window_id,
            add=[float(candidate)],
            remove=[],
            add_derivations=[idx],
            snap_tol_mhz=snap_tol_mhz,
        )
        chi2r_after = float(new_wf.reduced_chi2)
        n_after = len(new_wf.fitted_peaks)
        evidence: Dict[str, object] = {
            "chi2r_before": chi2r_before,
            "chi2r_after": chi2r_after,
            "n_peaks_before": n_before,
            "n_peaks_after": n_after,
        }
        ctx.changeset.decisions.append(
            {
                "window_id": window_id,
                "frequency_mhz": float(candidate),
                "kind": "add",
                "evidence": evidence,
            }
        )
        ctx.changeset.next_decision_index += 1
        return _make_refit_result(
            ctx,
            window_id=window_id,
            n_peaks_before=n_before,
            n_peaks_after=n_after,
            chi2r_before=chi2r_before,
            chi2r_after=chi2r_after,
            fitted_peaks=list(new_wf.fitted_peaks),
        )

    anchor_freq = 0.0
    wf_list = [
        wf for wf in ctx.changeset.spectrum_fit.window_fits if wf.window_id == window_id
    ]
    if wf_list:
        c = _window_center(wf_list[0])
        if c is not None:
            anchor_freq = c
        elif wf_list[0].fitted_peaks:
            anchor_freq = float(
                max(
                    wf_list[0].fitted_peaks,
                    key=lambda p: (float(p.snr) if p.snr is not None else 0.0),
                ).frequency_mhz
            )
    ctx.changeset.decisions.append(
        {
            "window_id": window_id,
            "frequency_mhz": anchor_freq,
            "kind": "accept",
            "evidence": {},
            "bare": True,
        }
    )
    ctx.changeset.next_decision_index += 1
    return None


def _batch_apply_create(
    ctx: _BatchCtx,
    anchor_mhz: float,
    *,
    replay_window_id: Optional[int],
    snap_tol_mhz: float,
) -> CreateWindowResult:
    """Batch equivalent of :func:`create_window_impl`. Recomputes the effective
    plan (:func:`_batch_effective_plan`) so a second create in the same batch
    sees the first one, but never rebuilds ``fit_ctx``."""
    from ..fitting.result_conversion import sort_fitting_result_by_frequency
    from .active_ft_support import default_tau0_us

    anchor = float(anchor_mhz)
    plan = _batch_effective_plan(ctx)
    fit_map: Dict[int, FittingResult] = {
        int(wf.window_id): wf
        for wf in ctx.changeset.spectrum_fit.window_fits
        if wf.window_id is not None
    }

    if ctx.shared.fit_ctx.trim_range is not None:
        t_lo, t_hi = (
            min(ctx.shared.fit_ctx.trim_range),
            max(ctx.shared.fit_ctx.trim_range),
        )
        if not (t_lo <= anchor <= t_hi):
            raise ValueError(
                f"anchor {anchor:.4f} MHz is outside the analysis band "
                f"[{t_lo:.4f}, {t_hi:.4f}] MHz. Re-run 'ft run' with a trim "
                f"that covers it (which rebuilds the fit) if the line is real."
            )

    from ..preprocessing.window_planning import plan_stage6_window

    params = plan.parameters
    proposal = plan_stage6_window(
        plan,
        ctx.shared.peaks_loaded,
        ctx.shared.fit_ctx.active_ft.freq_mhz,
        ctx.shared.fit_ctx.active_ft.complex_spectrum,
        ctx.shared.fit_ctx.rms_for_fit,
        anchor,
        acquisition_us=float(ctx.shared.fit_ctx.acquisition_us),
        tau_us=params.get("tau_us"),
        min_window_half_width_mhz=float(params.get("min_window_half_width_mhz", 2.0)),
        min_window_half_width_points=int(
            params.get("min_window_half_width_points", 32)
        ),
        min_freeze_snr=float(params.get("min_freeze_snr", ctx.shared.min_freeze_snr)),
        magnitude_attachment_threshold=float(
            params.get("magnitude_attachment_threshold", 0.1)
        ),
        live_window_ids=sorted(fit_map),
    )
    fit_win = proposal.window
    new_wid = int(fit_win.window_id)

    if replay_window_id is not None and int(replay_window_id) != new_wid:
        want = int(replay_window_id)
        if proposal.mode == "widened":
            raise ValueError(
                f"replaying the window created at {anchor:.4f} MHz now widens "
                f"window {new_wid} instead of creating window {want}; the base "
                f"plan or the surviving edit set has changed"
            )
        taken = {int(w.window_id) for w in plan.windows}
        if want in taken:
            raise ValueError(
                f"replaying the window created at {anchor:.4f} MHz wants id "
                f"{want}, which is already in use; the base plan or the "
                f"surviving edit set has changed"
            )
        fit_win.window_id = want
        new_wid = want

    if proposal.mode == "created":
        tau_maj_us, sigma_tau_us = _resolve_refit_window_tau(
            fit_win,
            ctx.shared.resolved,
            ctx.shared.persisted_cal,
            ctx.shared.tau_maj_global,
            ctx.shared.sigma_tau_global,
            ctx.shared.tau_source,
        )
        tau0 = (
            float(tau_maj_us)
            if tau_maj_us is not None and tau_maj_us > 0.0
            else default_tau0_us(float(ctx.shared.fit_ctx.acquisition_us))
        )
        seed_wf = FittingResult(window_id=new_wid, shape=ctx.shared.shape_enum.value)
        seed_wf.fixed_parameters = _frozen_parameters_from_sources(
            proposal.depends_on, fit_map, ctx.shared.min_freeze_snr
        )
        seed_wf.shared_parameters = {"tau_us": {"value": tau0, "fitted": False}}
    else:
        existing_wf = fit_map.get(new_wid)
        if existing_wf is None:
            raise ValueError(
                f"window {new_wid} has no Stage 5 fit to widen; "
                "re-run 'fit run' before creating windows"
            )
        seed_wf = existing_wf
        tau_maj_us, sigma_tau_us = _resolve_refit_window_tau(
            fit_win,
            ctx.shared.resolved,
            ctx.shared.persisted_cal,
            ctx.shared.tau_maj_global,
            ctx.shared.sigma_tau_global,
            ctx.shared.tau_source,
        )

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

    ctx.changeset.created_windows = [
        w for w in ctx.changeset.created_windows if int(w.window_id) != new_wid
    ] + [fit_win]
    ctx.changeset.fit_window_map = {
        w.window_id: w for w in _batch_effective_plan(ctx).windows
    }
    ctx.changeset.mutated_wids.add(new_wid)

    lo, hi = fit_win.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    grid_span = fit_win.diagnostics.get("grid_span", [0, -1])
    n_points = int(grid_span[1]) - int(grid_span[0]) + 1

    ctx.changeset.decisions.append(
        {
            "window_id": new_wid,
            "frequency_mhz": anchor,
            "kind": "create_window",
            "evidence": {
                "mode": proposal.mode,
                "freq_min_mhz": lo,
                "freq_max_mhz": hi,
                "n_points": n_points,
                "n_contributors": len(fit_win.fixed_contributors),
                "depends_on": [int(d) for d in proposal.depends_on],
            },
        }
    )
    ctx.changeset.next_decision_index += 1

    probe_freq_mhz = ctx.shared.fit_ctx.probe_freq_mhz
    epsilon = ctx.shared.epsilon
    return CreateWindowResult(
        window_id=new_wid,
        mode=proposal.mode,
        anchor_mhz=anchor,
        freq_range=(lo, hi),
        n_points=n_points,
        n_contributors=len(fit_win.fixed_contributors),
        depends_on=[int(d) for d in proposal.depends_on],
        n_peaks=len(new_wf.fitted_peaks),
        anchor_calibrated_mhz=_frame_to_calibrated(
            anchor, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon
        ),
        freq_range_calibrated=(
            _frame_to_calibrated(lo, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon),
            _frame_to_calibrated(hi, probe_freq_mhz=probe_freq_mhz, epsilon=epsilon),
        ),
        calibration_state=ctx.shared.calibration_state,
        epsilon=epsilon,
        sigma_epsilon=ctx.shared.sigma_epsilon,
    )


def _canonicalize_batch_plan(
    plan: Sequence[PlannedAction],
) -> List[Tuple[int, PlannedAction]]:
    """Pair each action with its original plan position, then order for
    execution: creates first (their own relative order -- they install
    structure later rows name), then every other action grouped by ascending
    window id. ``sorted`` is stable, so two actions sharing a window id keep
    the relative order ``_resolve_curation_plan`` already gave them."""
    indexed = list(enumerate(plan))
    creates = [t for t in indexed if t[1].kind == "create"]
    rest = sorted(
        (t for t in indexed if t[1].kind != "create"), key=lambda t: t[1].window_id
    )
    return creates + rest


def _derive_batch_review(ctx: _BatchCtx, path: str) -> Stage6Review:
    """Derive the new ``/stage6_review`` for the whole batch, without writing
    it: append every pending decision (assigning sequential ``order_index``
    values after whatever the file already held), refresh each touched
    window's provenance / attention reasons, rebuild the final-products table
    once if any fit changed or its persisted stamp has gone stale against the
    file's current calibration (a bare-accept-only batch mutates no fit but
    must still not launder a stale table back to disk), and carry the (now
    possibly batch-updated) ``created_windows`` overlay.

    Reloads ``/stage6_review``, the FID and the frequency calibration from
    ``path`` -- current on-disk state, not anything cached on ``ctx`` -- since
    those are what the new review has to be consistent with. Split out from
    the write so a preview (a later unit) can derive the would-be review
    in memory and never call :func:`_persist_batch_review` at all.
    """
    with h5py.File(path, "r") as h5f:
        existing_review: Stage6Review = (
            load_stage6_review_from_hdf5(h5f["stage6_review"])
            if "stage6_review" in h5f
            else Stage6Review()
        )

    base_index = len(existing_review.decision_log)
    statuses = dict(existing_review.window_statuses)
    final_products = existing_review.final_products

    sideband: Optional[Sideband] = None
    merged_window_freqs: Dict[int, List[float]] = {}
    reason_cache: Dict[int, List[AttentionReason]] = {}
    spur_centers_mhz: List[float] = [
        float(v)
        for v in ctx.changeset.spectrum_fit.parameters.get("spur_centers_mhz", [])
    ]
    acquisition_us: float = float(
        ctx.changeset.spectrum_fit.parameters.get("acquisition_us", 0.0)
    )

    # Rebuild whenever the fit changed *or* the persisted stamp no longer
    # matches the file's current calibration (a timebase re-run since the
    # table was last built, with no fit-mutating Stage 6 action this batch --
    # a bare accept is exactly this case, since it adds no mutated_wids).
    if ctx.changeset.mutated_wids or _final_products_is_stale(final_products, path):
        fid = load_fid_from_pipeline_impl(path)
        sideband = Sideband.coerce(fid.sideband)
        merged_window_freqs = _auto_merged_window_freqs(ctx.changeset.spectrum_fit)

        with h5py.File(path, "r") as h5f:
            floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
        cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
        final_products = _build_final_products(
            ctx.changeset.spectrum_fit,
            probe_freq_mhz=float(fid.probe_freq_mhz),
            sideband=sideband,
            calibration_state=cal_state,
            epsilon=epsilon,
            sigma_epsilon=sigma_eps,
            sigma_floor_khz=floor_khz,
        )

    def _attention_reasons_for(window_id: int) -> List[AttentionReason]:
        if window_id in reason_cache:
            return reason_cache[window_id]
        wf_list = [
            wf
            for wf in ctx.changeset.spectrum_fit.window_fits
            if wf.window_id == window_id
        ]
        if wf_list and sideband is not None:
            reasons = _compute_attention_reasons(
                wf_list[0],
                spur_centers_mhz=spur_centers_mhz,
                acquisition_us=acquisition_us,
                ledger_bar=DEFAULT_DISPLAY_BAR,
                attention_candidate_evidence=DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
                sideband=sideband,
                kappa=DEFAULT_SHAPE_ERROR_KAPPA,
                noise_floor=DEFAULT_CHI2R_NOISE_FLOOR,
                auto_merged=window_id in merged_window_freqs,
                merged_freqs=merged_window_freqs.get(window_id, ()),
            )
        else:
            existing_status = statuses.get(window_id)
            reasons = (
                list(existing_status.attention_reasons)
                if existing_status is not None
                else []
            )
        reason_cache[window_id] = reasons
        return reasons

    new_entries: List[DecisionLogEntry] = []
    for offset, dec in enumerate(ctx.changeset.decisions):
        wid = int(dec["window_id"])
        kind = str(dec["kind"])
        new_entries.append(
            DecisionLogEntry(
                order_index=base_index + offset,
                window_id=wid,
                frequency_mhz=float(dec["frequency_mhz"]),
                kind=kind,
                provenance="user",
                evidence=dict(dec["evidence"]),
            )
        )
        if dec.get("bare"):
            existing_status = statuses.get(wid)
            kept_reasons = (
                list(existing_status.attention_reasons)
                if existing_status is not None
                else []
            )
            statuses[wid] = WindowReviewStatus(
                window_id=wid,
                provenance="reviewed",
                attention_reasons=kept_reasons,
                invalidated=False,
            )
        else:
            statuses[wid] = WindowReviewStatus(
                window_id=wid,
                provenance="user-edited",
                attention_reasons=_attention_reasons_for(wid),
                invalidated=False,
            )

    return Stage6Review(
        window_statuses=statuses,
        decision_log=list(existing_review.decision_log) + new_entries,
        final_products=final_products,
        created_windows=list(ctx.changeset.created_windows),
    )


def _write_stage6_review_only(review: Stage6Review, path: str) -> None:
    """Write *review* to ``/stage6_review`` verbatim -- no derivation.

    The write half of :func:`_persist_batch_review`, split out so
    :func:`_finish_batch` can persist an ALREADY-derived review (D4's
    staged-preview reuse: a ``ReviewSession`` accept that persists exactly
    what an immediately-preceding preview already computed, rather than
    re-deriving and trusting the two to agree) without going through
    :func:`_derive_batch_review` a second time.
    """
    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)


def _persist_batch_review(ctx: _BatchCtx, path: str) -> None:
    """Derive the batch's new ``/stage6_review`` and write it.

    The write half of :func:`_derive_batch_review`; kept separate so a
    preview can call the derive half alone. This function is the engine's
    only writer of ``/stage6_review`` for the ordinary (non-staged) path --
    see :func:`_write_stage6_review_only` for the staged-reuse path.
    """
    new_review = _derive_batch_review(ctx, path)
    _write_stage6_review_only(new_review, path)


def _check_add_seeds_arity(
    add: Sequence[float], add_seeds: Optional[Sequence[ModelPeak]]
) -> None:
    """Explicit seeds are positional: one per added frequency, or none at all."""
    if add_seeds is not None and len(add_seeds) != len(add):
        raise ValueError(
            f"len(add_seeds)={len(add_seeds)} must equal len(add)={len(add)}"
        )


def _check_merge_arity(peaks: Sequence[float]) -> None:
    """A merge collapses a set into one line, so it needs a set to collapse."""
    if len(peaks) < 2:
        raise ValueError(
            f"merge requires at least 2 peak frequencies; got {len(peaks)}"
        )


def _check_split_arity(into: int) -> None:
    """A split replaces one line with several, so ``into`` must be at least 2."""
    if into < 2:
        raise ValueError(f"split requires into >= 2; got {into}")


def _open_batch(
    path: str,
    *,
    snap_tol_mhz: float,
    snapshot: bool = True,
    shared: Optional[_SharedFitCtx] = None,
) -> _BatchCtx:
    """Gate, (optionally) snapshot, and build the shared context for a
    fit-mutating batch.

    The one entry into the Stage 6 fit-editing engine. Every fit-mutating
    operation -- the interactive single-window verbs and a curation file alike
    -- opens its work here, so the epoch gate and the undo baseline are enforced
    structurally rather than remembered at each call site.

    ``snapshot=False`` is for :func:`review_preview_impl` alone: a preview
    must be epoch-gated exactly like a real batch, but it may never take the
    undo baseline, because that is a write and a preview writes nothing. The
    call stays textually inside this function either way, so
    ``test_only_open_batch_takes_the_undo_baseline`` still holds: the
    baseline is taken in exactly one place, just not on every call.

    ``shared`` lets a caller with an already-built :class:`_SharedFitCtx`
    reuse it (a later unit does this for an amortized review session); a
    fresh one is built when omitted.
    """
    # Splicing a freshly-computed window into an existing fit is the one
    # operation that can mix two analysis models inside one artifact. Checked
    # before the snapshot so a refused batch leaves the file untouched.
    require_splice_compatible_environment(path)
    if snapshot:
        # Snapshot the automatic fit before the first edit mutates it in place
        # (a no-op after the first time), so 'review undo' can restore it and
        # replay.
        _snapshot_stage5_baseline(path)
    ctx = _build_batch_ctx(path, snap_tol_mhz=snap_tol_mhz, shared=shared)
    ctx.baseline_taken = snapshot
    return ctx


def _cascade_batch(ctx: _BatchCtx, *, snap_tol_mhz: float) -> List[int]:
    """Run the batch's one combined cascade over the union of the
    directly-edited windows, mutating ``ctx.changeset.spectrum_fit`` and
    ``ctx.changeset.mutated_wids`` in place. Touches no file.

    Cascading once -- rather than once per action -- is what makes the result
    independent of the order the actions were applied in. Split out from
    :func:`_finish_batch` so a preview (a later unit) can run this same
    cascade in memory and stop there, never reaching the persists below.
    Returns the cascaded window ids.
    """
    cascaded: List[int] = []
    if ctx.changeset.dirty_wids:
        cascaded = _cascade_refit_dependents(
            spectrum_fit=ctx.changeset.spectrum_fit,
            edited_wids=sorted(ctx.changeset.dirty_wids),
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
        )
        if cascaded:
            ctx.changeset.mutated_wids.update(cascaded)
            logger.info(
                "Stage 6 cascade: re-fit %d dependent window(s) %s",
                len(cascaded),
                sorted(cascaded),
            )
    return cascaded


def _finish_batch(
    ctx: _BatchCtx,
    path: str,
    *,
    snap_tol_mhz: float,
    cascaded: Optional[List[int]] = None,
    precomputed_review: Optional[Stage6Review] = None,
) -> List[int]:
    """Close a batch: one combined cascade, one fit persist, one review persist.

    The counterpart to :func:`_open_batch`. Kept as cascade-then-persist in one
    function (rather than letting a caller cascade and persist separately) so
    that this remains the engine's one and only writer of ``/stage5_fitting``
    -- see ``test_only_finish_batch_persists_the_fit``. Returns the cascaded
    window ids.

    ``cascaded``/``precomputed_review``, when given, are an ALREADY-cascaded
    window-id list and an ALREADY-derived :class:`Stage6Review` from an
    earlier :func:`_cascade_batch` / :func:`_derive_batch_review` call against
    this EXACT ``ctx`` -- the staged-preview reuse path (D4,
    ``ReviewSession``). Skips re-cascading and/or re-deriving and persists
    exactly those, so the bytes on disk are guaranteed to be what a preceding
    preview showed rather than a second computation trusted to agree with the
    first. ``None`` (every sessionless caller) computes them here, unchanged
    from before.
    """
    from ..io.fitting_serialization import save_spectrum_fit_to_hdf5

    if not ctx.baseline_taken:
        raise ValueError(
            "refusing to persist a batch context whose undo baseline was "
            "never taken (built via _open_batch(..., snapshot=False)); this "
            "is a preview-only context and must never reach _finish_batch"
        )

    if cascaded is None:
        cascaded = _cascade_batch(ctx, snap_tol_mhz=snap_tol_mhz)

    shape_attr = str(ctx.changeset.spectrum_fit.parameters.get("shape", "lorentzian"))
    with h5py.File(path, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(ctx.changeset.spectrum_fit, grp)
        grp.attrs["shape"] = shape_attr

    if precomputed_review is not None:
        _write_stage6_review_only(precomputed_review, path)
    else:
        _persist_batch_review(ctx, path)
    return cascaded


_T = TypeVar("_T")


def _run_single_action(
    path: str,
    apply: "Callable[[_BatchCtx], _T]",
    *,
    snap_tol_mhz: float,
    shared: Optional[_SharedFitCtx] = None,
) -> _T:
    """Run one fit-mutating action as a batch of one, returning its result.

    The interactive single-window verbs are this plus their own argument
    checking: they and ``review apply`` share the same context build, the same
    appliers, the same cascade and the same persist, so a change to any of those
    reaches every caller at once and a new verb cannot be written that skips one.

    ``shared`` lets a caller with an already-built :class:`_SharedFitCtx`
    reuse it (``ReviewSession``, D3); a fresh one is built when omitted,
    exactly as before.
    """
    ctx = _open_batch(path, snap_tol_mhz=snap_tol_mhz, shared=shared)
    result = apply(ctx)
    _finish_batch(ctx, path, snap_tol_mhz=snap_tol_mhz)
    return result


def _execute_curation_batch(
    path: str,
    plan: List[PlannedAction],
    *,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    shared: Optional[_SharedFitCtx] = None,
) -> int:
    """Execute a resolved curation plan as one batch: shared context, one
    combined cascade, one persist of ``/stage5_fitting`` and one of
    ``/stage6_review``. Falls back to the cheap per-action path
    (:func:`_execute_planned_action`) when the whole plan is bare ``accept``
    (no Stage 5 fit touched, so there is nothing to batch and -- unlike every
    other action -- a bare accept does not even require a Stage 5 fit to
    exist).

    ``shared`` lets a caller with an already-built :class:`_SharedFitCtx`
    reuse it (``ReviewSession``, D3); a fresh one is built when omitted,
    exactly as before. Unused on the bare-accept-only fallback path, which
    never opens the engine at all.

    Returns the number of actions applied (``len(plan)`` on success; a failure
    raises before this returns and leaves the file untouched, since nothing is
    persisted until every action in the batch has succeeded).
    """
    if not plan:
        return 0

    needs_fit = any(a.kind != "accept" or a.candidate is not None for a in plan)
    if not needs_fit:
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
        return applied

    # Gate + snapshot + context, exactly as a single-window verb does: the
    # engine enforces them itself rather than trusting its callers. An
    # all-``accept`` plan whose accepts carry candidates is fit-mutating but
    # reads as non-mutating to the callers' "any non-accept action" pre-check,
    # so a caller-side gate alone would let that shape through.
    ctx = _open_batch(path, snap_tol_mhz=snap_tol_mhz, shared=shared)

    applied = 0
    for original_index, action in _canonicalize_batch_plan(plan):
        try:
            if action.kind == "create":
                if action.anchor is None:
                    raise ValueError("create action requires an anchor frequency")
                _batch_apply_create(
                    ctx,
                    action.anchor,
                    replay_window_id=(
                        None
                        if action.window_id == _NEW_WINDOW_SENTINEL
                        else action.window_id
                    ),
                    snap_tol_mhz=snap_tol_mhz,
                )
            elif action.kind == "edit":
                _batch_apply_edit_action(
                    ctx,
                    action.window_id,
                    action.add,
                    action.remove,
                    snap_tol_mhz=snap_tol_mhz,
                )
            elif action.kind == "merge":
                _batch_apply_merge(
                    ctx, action.window_id, action.peaks, snap_tol_mhz=snap_tol_mhz
                )
            elif action.kind == "split":
                if action.peak is None:
                    raise ValueError("split action requires a peak frequency")
                _batch_apply_split(
                    ctx,
                    action.window_id,
                    action.peak,
                    action.into,
                    snap_tol_mhz=snap_tol_mhz,
                )
            elif action.kind == "accept":
                _batch_apply_accept(
                    ctx,
                    action.window_id,
                    action.candidate,
                    snap_tol_mhz=snap_tol_mhz,
                )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"curation action {original_index + 1} "
                f"({describe_planned_action(action)}) failed: {exc}"
            ) from exc
        applied += 1

    _finish_batch(ctx, path, snap_tol_mhz=snap_tol_mhz)

    return applied


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

    return replace(
        action,
        add=[conv(f) for f in action.add],
        remove=[conv(f) for f in action.remove],
        peaks=[conv(f) for f in action.peaks],
        peak=None if action.peak is None else conv(action.peak),
        candidate=None if action.candidate is None else conv(action.candidate),
        anchor=None if action.anchor is None else conv(action.anchor),
    )


def apply_curation_impl(
    file_path: Union[Path, str],
    curation_path: Union[Path, str],
    *,
    dry_run: bool = False,
    frame: Optional[Frame] = None,
    _shared: Optional["_SharedFitCtx"] = None,
) -> CurationApplyResult:
    """Apply a curation file to *file_path*, delegating to the edit impls.

    Parses the curation CSV, coalesces it into a delegated action plan (one
    refit per window for runs of add/remove; merge/split/accept stand alone),
    converts every frequency on the resolved plan to raw (before ambiguity
    resolution, before any snapping), and -- unless ``dry_run`` -- applies the
    whole plan as one batch (:func:`_execute_curation_batch`): the Stage 5 fit
    context is built once, every action is applied to an in-memory
    ``SpectrumFit`` in a canonical cross-window order (ascending window id,
    independent of the file's row order), the dependents of every
    directly-edited window are cascaded once, and the result is persisted
    once. This is an equivalence of *outcome*, not of per-row execution -- see
    :func:`_execute_curation_batch` for the exact ordering contract.
    ``dry_run`` returns the resolved (raw-converted) plan and
    frequency-resolution warnings without mutating the file.

    ``frame`` applies uniformly to every frequency the curation file carries
    -- there is no per-row frame column. The file's own optional header (A3,
    ``# frame: ...`` / ``# epsilon: ...``, see :func:`parse_curation_file`
    and :func:`_resolve_curation_frame`) takes precedence when it disagrees
    with neither, or wins outright when ``frame`` is omitted; when both are
    given and disagree, the call is refused. Omitting both is an error on a
    ``self_calibrated`` file when the plan carries any frequency at all. A
    calibrated header whose stamped epsilon no longer matches the file's
    current one is refused -- never silently resolved with either value.

    Raises ``ValueError`` on a malformed curation file or when an action fails
    to resolve (e.g. a ``remove`` frequency matches no fitted peak), tagged with
    the offending action; a failure leaves the file untouched (nothing is
    persisted until every action in the plan has succeeded).

    ``_shared`` is internal -- see :func:`refit_window_impl`.
    """
    path = str(file_path)
    ops = parse_curation_file(curation_path)
    plan = _resolve_curation_plan(ops)

    resolved_frame: Frame = "raw"
    stamp: Optional[_CalibrationStamp] = None
    if any(_planned_action_has_freq(a) for a in plan):
        resolved_frame, stamp = _resolve_curation_frame(path, ops.header, frame)
        plan = [
            _planned_action_to_raw(a, frame=resolved_frame, stamp=stamp) for a in plan
        ]

    warnings = _curation_ambiguity_warnings(path, plan)
    warnings += _frame_mismatch_warnings(
        path, plan, resolved_frame=resolved_frame, stamp=stamp
    )

    # No epoch pre-check here: _execute_curation_batch gates itself before it
    # snapshots or writes anything, and it knows which plans are fit-mutating
    # (an accept carrying a candidate is; a bare one is not), which this layer
    # would have to duplicate to get right.
    if dry_run:
        return CurationApplyResult(
            plan=plan, warnings=warnings, applied=0, dry_run=True
        )

    applied = _execute_curation_batch(path, plan, shared=_shared)

    return CurationApplyResult(
        plan=plan, warnings=warnings, applied=applied, dry_run=False
    )


# ---------------------------------------------------------------------------
# review_preview_impl: run a curation plan to completion in memory and
# report the fitted outcome, without persisting anything (C1-C6).
#
# Reuses the exact hooks Task B built for this: _build_batch_ctx (shared
# context, optionally reused), the same per-action appliers
# _execute_curation_batch dispatches to, _cascade_batch (the one combined
# cascade, touches no file), and _derive_batch_review (the would-be review
# INCLUDING final products, no write). _finish_batch -- the engine's only
# writer of /stage5_fitting -- is never called.
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
        the combined cascade refit this window as a downstream dependent of
        some *other* directly-edited window, without any action of its own
        naming it. A window that is both directly edited *and* downstream of
        a sibling edit in the same batch (the case ``_cascade_closure``
        deliberately adds back, ``stage6_impl.py`` ~:2026) is ``"direct"``:
        it has an originating action, even though the cascade pass re-fit it
        again to pick up the sibling's refreshed background.
    action_indices : list of int
        0-based indices into ``ReviewPreviewResult.plan`` of every action
        that directly targeted this window. Empty for a purely-cascaded
        window.
    n_peaks_before, n_peaks_after : int
        Peak count in this window before the batch / after the cascade.
    chi2r_before, chi2r_after : float
        Reduced chi-squared before the batch / after the cascade.
    peaks : list of FinalPeak
        This window's rows from the would-be final-products table
        (:func:`_derive_batch_review`) -- calibrated frequencies and the
        three-term sigma budget, identical in shape and value to what a
        subsequent ``apply`` of the same plan would persist. Not the
        raw / stat-only ``RefitWindowResult.fitted_peaks``.
    """

    window_id: int
    origin: str
    action_indices: List[int] = field(default_factory=list)
    n_peaks_before: int = 0
    n_peaks_after: int = 0
    chi2r_before: float = 0.0
    chi2r_after: float = 0.0
    peaks: List["FinalPeak"] = field(default_factory=list)


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
    """

    windows: Dict[int, PreviewWindowResult] = field(default_factory=dict)
    plan: List["PlannedAction"] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class _PreviewRun:
    """Internal: everything one preview computed, including the pieces
    :func:`review_preview_impl` throws away but a :class:`ReviewSession`
    needs to stage for a possible immediately-following accept (D4): the
    finished (already cascaded, never persisted) batch context, its cascaded
    window ids, and the already-derived would-be review. ``ctx`` /
    ``cascaded_wids`` / ``review`` are ``None`` (empty) only for the
    bare-accept-only short circuit, which never opens the engine and so has
    nothing to stage.
    """

    result: ReviewPreviewResult
    ctx: Optional[_BatchCtx] = None
    cascaded_wids: List[int] = field(default_factory=list)
    review: Optional[Stage6Review] = None


def _run_review_preview(
    file_path: Union[Path, str],
    curation_path: Union[Path, str],
    *,
    frame: Optional[Frame] = None,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
    shared: Optional[_SharedFitCtx] = None,
) -> _PreviewRun:
    """Run a curation file's resolved plan to completion in memory and report
    the fitted outcome -- final-product numbers, post-cascade -- without
    writing anything to *file_path*. The body of :func:`review_preview_impl`,
    plus the internal state (D4) a :class:`ReviewSession` needs to persist a
    following accept without recomputing.

    Mirrors :func:`apply_curation_impl`'s parse / resolve / frame-convert
    prologue exactly, including the curation file's optional frame header
    (A3) and the A5 frame-mismatch advisory, then instead of delegating to
    :func:`_execute_curation_batch` (which persists), runs the same
    canonicalized action sequence against the same appliers, the same one
    combined cascade (:func:`_cascade_batch`), and the same derive-without-
    write step (:func:`_derive_batch_review`) that a live apply's persist
    would have used. ``_finish_batch`` -- the engine's only writer of
    ``/stage5_fitting`` -- is never called, and the undo baseline snapshot is
    never taken (``_open_batch(..., snapshot=False)``): a preview writes
    nothing, byte for byte.

    Epoch-gated exactly like a real batch (the same ``_open_batch`` call, so
    the same ``require_splice_compatible_environment`` check): a preview
    across an unacknowledged epoch boundary would show numbers whose accept
    is guaranteed to refuse.

    A plan consisting entirely of bare ``accept`` rows (no frequency, no fit
    touched) short-circuits before the gate: it does no fits, is not
    epoch-gated (a bare accept in a live apply is not gated either -- see
    :func:`review_accept_impl`), and returns an empty ``windows`` dict.

    Raises the same per-action attributed ``ValueError`` a live apply raises
    (tagged with the 1-based action index and its description), on the same
    failures, since it shares the same appliers.

    ``shared`` lets a caller with an already-built :class:`_SharedFitCtx`
    reuse it (``ReviewSession``, D3); a fresh one is built when omitted,
    exactly as before.
    """
    path = str(file_path)
    ops = parse_curation_file(curation_path)
    plan = _resolve_curation_plan(ops)

    resolved_frame: Frame = "raw"
    stamp: Optional[_CalibrationStamp] = None
    if any(_planned_action_has_freq(a) for a in plan):
        resolved_frame, stamp = _resolve_curation_frame(path, ops.header, frame)
        plan = [
            _planned_action_to_raw(a, frame=resolved_frame, stamp=stamp) for a in plan
        ]

    warnings = _frame_mismatch_warnings(
        path, plan, resolved_frame=resolved_frame, stamp=stamp
    )

    needs_fit = any(a.kind != "accept" or a.candidate is not None for a in plan)
    if not needs_fit:
        # C5: bare-accept-only (or empty) plan -- no fits, no gate, nothing to
        # report. Mirrors _execute_curation_batch's own cheap path, which
        # likewise never opens the engine for this shape.
        return _PreviewRun(
            result=ReviewPreviewResult(windows={}, plan=plan, warnings=warnings)
        )

    ctx = _open_batch(path, snap_tol_mhz=snap_tol_mhz, snapshot=False, shared=shared)

    # Snapshot every touched window's pre-batch stats now, before any action
    # mutates ctx.changeset.spectrum_fit in place -- this is the "before" a
    # multi-action batch on one window (or a cascade that revisits a directly
    # -edited window) must report, not any action's own intermediate result.
    before_stats: Dict[int, Tuple[int, float]] = {
        int(wf.window_id): (len(wf.fitted_peaks), float(wf.reduced_chi2))
        for wf in ctx.changeset.spectrum_fit.window_fits
        if wf.window_id is not None
    }

    action_indices: Dict[int, List[int]] = {}
    for original_index, action in _canonicalize_batch_plan(plan):
        try:
            if action.kind == "create":
                if action.anchor is None:
                    raise ValueError("create action requires an anchor frequency")
                created = _batch_apply_create(
                    ctx,
                    action.anchor,
                    replay_window_id=(
                        None
                        if action.window_id == _NEW_WINDOW_SENTINEL
                        else action.window_id
                    ),
                    snap_tol_mhz=snap_tol_mhz,
                )
                action_indices.setdefault(created.window_id, []).append(original_index)
            elif action.kind == "edit":
                _batch_apply_edit_action(
                    ctx,
                    action.window_id,
                    action.add,
                    action.remove,
                    snap_tol_mhz=snap_tol_mhz,
                )
                action_indices.setdefault(action.window_id, []).append(original_index)
            elif action.kind == "merge":
                _batch_apply_merge(
                    ctx, action.window_id, action.peaks, snap_tol_mhz=snap_tol_mhz
                )
                action_indices.setdefault(action.window_id, []).append(original_index)
            elif action.kind == "split":
                if action.peak is None:
                    raise ValueError("split action requires a peak frequency")
                _batch_apply_split(
                    ctx,
                    action.window_id,
                    action.peak,
                    action.into,
                    snap_tol_mhz=snap_tol_mhz,
                )
                action_indices.setdefault(action.window_id, []).append(original_index)
            elif action.kind == "accept":
                _batch_apply_accept(
                    ctx,
                    action.window_id,
                    action.candidate,
                    snap_tol_mhz=snap_tol_mhz,
                )
                if action.candidate is not None:
                    action_indices.setdefault(action.window_id, []).append(
                        original_index
                    )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"curation action {original_index + 1} "
                f"({describe_planned_action(action)}) failed: {exc}"
            ) from exc

    # Direct = every window some action touched this batch, snapshotted
    # BEFORE the cascade runs (mutated_wids only grows from here). Anything
    # _cascade_batch adds beyond this set arrived purely as a dependent.
    direct_wids = set(ctx.changeset.mutated_wids)
    cascaded_wids = set(_cascade_batch(ctx, snap_tol_mhz=snap_tol_mhz))

    review = _derive_batch_review(ctx, path)
    peaks_by_window: Dict[int, List["FinalPeak"]] = {}
    if review.final_products is not None:
        for peak in review.final_products.peaks:
            if peak.window_id is not None:
                peaks_by_window.setdefault(int(peak.window_id), []).append(peak)

    after_by_wid: Dict[int, Tuple[int, float]] = {
        int(wf.window_id): (len(wf.fitted_peaks), float(wf.reduced_chi2))
        for wf in ctx.changeset.spectrum_fit.window_fits
        if wf.window_id is not None
    }

    windows: Dict[int, PreviewWindowResult] = {}
    for wid in sorted(direct_wids | cascaded_wids):
        n_before, chi2r_before = before_stats.get(wid, (0, 0.0))
        n_after, chi2r_after = after_by_wid.get(wid, (0, 0.0))
        windows[wid] = PreviewWindowResult(
            window_id=wid,
            origin="direct" if wid in direct_wids else "cascaded",
            action_indices=sorted(action_indices.get(wid, [])),
            n_peaks_before=n_before,
            n_peaks_after=n_after,
            chi2r_before=chi2r_before,
            chi2r_after=chi2r_after,
            peaks=peaks_by_window.get(wid, []),
        )

    result = ReviewPreviewResult(windows=windows, plan=plan, warnings=warnings)
    return _PreviewRun(
        result=result, ctx=ctx, cascaded_wids=sorted(cascaded_wids), review=review
    )


def review_preview_impl(
    file_path: Union[Path, str],
    curation_path: Union[Path, str],
    *,
    frame: Optional[Frame] = None,
    snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
) -> ReviewPreviewResult:
    """Run a curation file's resolved plan to completion in memory and report
    the fitted outcome -- final-product numbers, post-cascade -- without
    writing anything to *file_path*. See :func:`_run_review_preview` for the
    full contract; this is the public entry point, which discards the
    internal staging state a :class:`ReviewSession` needs and a sessionless
    caller does not.
    """
    return _run_review_preview(
        file_path, curation_path, frame=frame, snap_tol_mhz=snap_tol_mhz
    ).result


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
    if kind == "create_window":
        # Carry the id the original create produced: replay re-derives the
        # geometry from the anchor and the base plan but pins the label, so the
        # rows keyed on that id (and any peaks a consumer bound to it) still
        # refer to the same window.
        return CurationOp("create", wid, [float(entry.frequency_mhz)], {}, 0)
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
    _shared: Optional["_SharedFitCtx"] = None,
) -> UndoResult:
    """Undo one or more recorded decisions by id, replaying the rest.

    Rollback is replay-from-baseline: the automatic Stage 5 fit (snapshotted
    before the first edit) is restored, the review state rebuilt from it, and
    every *surviving* decision re-applied through the same engine ``review
    apply`` uses. The undone ids are dropped; the result is the canonical replay
    of the remaining decisions, so decision ids are renumbered afterward.

    ``dry_run`` returns the removed/surviving split and the resolved replay plan
    without mutating.

    ``_shared`` is internal (see :func:`refit_window_impl`) -- safe to reuse
    across an undo's restore-then-replay because none of it (the baseline
    restore, the review rebuild, the surviving-decision replay) touches
    anything :class:`_SharedFitCtx` derives from (settings, tau calibration,
    the base window plan, or the calibration stamp).

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

    # Undoing a window creation orphans every decision made against that window:
    # replaying them would fail partway through, leaving the file half-rolled-back.
    # Refuse up front and name the ids the caller has to undo along with it.
    dropped_windows = {
        int(e.window_id) for e in removed if e.kind == "create_window"
    } - {int(e.window_id) for e in surviving if e.kind == "create_window"}
    orphaned = sorted(
        e.order_index for e in surviving if int(e.window_id) in dropped_windows
    )
    if orphaned:
        raise ValueError(
            f"cannot undo: decision(s) {orphaned} act on window(s) "
            f"{sorted(dropped_windows)}, which the undone 'create_window' "
            f"decision(s) installed. Undo them together."
        )

    has_fit_edits = any(e.kind in _FIT_EDIT_KINDS for e in log)
    baseline = _has_stage5_baseline(path)
    if has_fit_edits and not baseline:
        raise ValueError(
            "cannot undo: the automatic-fit baseline is unavailable (the fit may "
            "have been re-run after editing). Rebuild from the source and re-edit."
        )

    ops = [_decision_to_op(e) for e in surviving]
    plan = _resolve_curation_plan(ops)

    # Undo is restore-then-replay, so the restore happens before any replayed
    # edit could hit the epoch gate. Check first: otherwise a refusal partway
    # through would leave the file rolled all the way back to the automatic
    # fit, discarding the surviving decisions the caller asked to keep.
    if not dry_run and any(e.kind in _FIT_EDIT_KINDS for e in surviving):
        require_splice_compatible_environment(path)

    if dry_run:
        return UndoResult(
            removed=removed, surviving=surviving, plan=plan, applied=0, dry_run=True
        )

    # Restore the automatic fit (when fit-mutating edits existed), rebuild the
    # review afresh from it, then replay the surviving decisions onto it as one
    # batch (_execute_curation_batch): a single shared fit context, one
    # combined cascade, one persist -- instead of one full rebuild per
    # surviving decision.
    if baseline:
        _restore_stage5_baseline(path)
    with h5py.File(path, "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
    review_run_impl(path)
    applied = _execute_curation_batch(path, plan, shared=_shared)

    return UndoResult(
        removed=removed,
        surviving=surviving,
        plan=plan,
        applied=applied,
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
    with h5py.File(path, "r") as h5f:
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


def _current_calibration_stamp(
    path: str,
) -> Optional[Tuple[str, float, float, float, float, str]]:
    """The six-tuple a fresh :class:`FinalProducts` would be stamped with
    *right now*: ``(calibration_state, epsilon, sigma_epsilon,
    sigma_floor_khz, probe_freq_mhz, sideband)``.

    Derived straight from the file -- ``spur.clocks``, ``timebase_calibration``,
    ``/frequency_calibration`` and the FID header -- never from a persisted
    ``FinalProducts``. Comparing a stamp against this is the whole staleness
    check. ``None`` when the file has no FID header to derive a probe
    frequency / sideband from (see :func:`_fid_header_for_stamp`).
    """
    header = _fid_header_for_stamp(path)
    if header is None:
        return None
    probe_freq_mhz, sideband_value = header
    cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
    with h5py.File(path, "r") as h5f:
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
    return (
        cal_state,
        float(epsilon),
        float(sigma_eps),
        float(floor_khz),
        float(probe_freq_mhz),
        sideband_value,
    )


def _final_products_is_stale(fp: Optional[FinalProducts], path: str) -> bool:
    """Whether ``fp``'s calibration stamp no longer matches the file's
    currently-derived calibration (e.g. after a timebase re-run that never
    touched Stage 6). ``None`` (no table built yet) is never "stale" -- there
    is nothing to have gone stale. Likewise when the file has nothing to
    derive a current stamp from (:func:`_current_calibration_stamp` returns
    ``None``): with no grounds to declare staleness, trust the persisted
    table rather than force a rebuild that cannot succeed anyway."""
    if fp is None:
        return False
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


def _rebuild_final_products(path: str) -> Optional[FinalProducts]:
    """Rebuild the final-products table from the raw Stage 5 fit and the
    file's current calibration. Pure derivation, read-only -- touches no file
    and does not require a Stage 6 action. Returns ``None`` when there is no
    Stage 5 fit to derive from."""
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            return None
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        floor_khz = load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz
    fid = load_fid_from_pipeline_impl(path)
    sideband = Sideband.coerce(fid.sideband)
    cal_state, epsilon, sigma_eps = _derive_frequency_calibration(path)
    return _build_final_products(
        spectrum_fit,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        calibration_state=cal_state,
        epsilon=epsilon,
        sigma_epsilon=sigma_eps,
        sigma_floor_khz=floor_khz,
    )


def _current_final_products(
    existing: Optional[FinalProducts], path: str
) -> Optional[FinalProducts]:
    """Return final products consistent with the file's current calibration.

    ``existing`` is whatever a persisted ``Stage6Review`` carries (``None``
    until ``review run`` first builds a table). When it is present but its
    stamp no longer matches the file's current calibration -- most commonly a
    timebase re-run with no Stage 6 action at all -- rebuild from the raw
    Stage 5 fit rather than returning the stale table.

    Read-only: never writes to ``path``, so it is safe to call on a file the
    caller has open only for reading (or not open at all). Callers that want
    the rebuilt table to stick persist it themselves.
    """
    if existing is None or not _final_products_is_stale(existing, path):
        return existing
    return _rebuild_final_products(path)


def get_final_products_impl(file_path: Union[Path, str]) -> Optional[FinalProducts]:
    """Return the current Stage 6 final-products table, or ``None``.

    Rebuilds from the raw Stage 5 fit -- in memory, without persisting --
    when the persisted table's calibration stamp no longer matches the
    file's current calibration (e.g. a timebase re-run since the table was
    last built). See :func:`_current_final_products`.
    """
    path = str(file_path)
    existing = load_stage6_review_from_file(path).final_products
    return _current_final_products(existing, path)


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
                clock_lattice=pk.clock_lattice,
                derivation=pk.derivation,
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
    sideband = Sideband.coerce(fid.sideband)

    spur_centers_mhz: List[float] = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
    ]
    acquisition_us: float = float(spectrum_fit.parameters.get("acquisition_us", 0.0))
    merged_window_freqs = _auto_merged_window_freqs(spectrum_fit)

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
            auto_merged=wid in merged_window_freqs,
            merged_freqs=merged_window_freqs.get(wid, ()),
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
        created_windows=list(existing_review.created_windows),
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
    with h5py.File(path, "r") as h5f:
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
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size, _fit_ctx_stage_provenance(path))


def _resolve_curation_call(
    path: str, curation_path: Union[Path, str], frame: Optional[Frame]
) -> Tuple[List["PlannedAction"], Frame, Optional[_CalibrationStamp]]:
    """Parse + resolve + frame-convert a curation file into the ready-to-run
    plan -- the shared prologue :func:`apply_curation_impl` and
    :func:`_run_review_preview` each already inline for themselves. A THIRD
    inlined copy for :class:`ReviewSession`'s staged-plan comparison would
    make three, so it is factored out here instead (used only by new D3/D4
    code; the two existing inlined copies are left as they are).
    """
    ops = parse_curation_file(curation_path)
    plan = _resolve_curation_plan(ops)
    resolved_frame: Frame = "raw"
    stamp: Optional[_CalibrationStamp] = None
    if any(_planned_action_has_freq(a) for a in plan):
        resolved_frame, stamp = _resolve_curation_frame(path, ops.header, frame)
        plan = [
            _planned_action_to_raw(a, frame=resolved_frame, stamp=stamp) for a in plan
        ]
    return plan, resolved_frame, stamp


# ---------------------------------------------------------------------------
# D3/D4: the amortized review session.
# ---------------------------------------------------------------------------


@dataclass
class _StagedPreview:
    """A finished, cascaded, but never-persisted batch outcome retained by a
    :class:`ReviewSession` immediately after ``review_preview`` (D4) -- so an
    immediately-following ``review_apply`` of the identical plan against an
    unchanged base can persist it directly instead of re-running the
    appliers, the cascade, and the review derivation a second time. Dropped
    the moment anything about the base -- or the requested plan -- no longer
    matches (see ``ReviewSession.review_apply``).
    """

    fingerprint: _FitCtxFingerprint
    curation_path: str
    frame: Optional[Frame]
    resolved_plan: List["PlannedAction"]
    warnings: List[str]
    ctx: _BatchCtx
    cascaded_wids: List[int]
    review: Stage6Review


class ReviewSession:
    """Amortized Stage 6 review session (D3): a context manager holding one
    :class:`_SharedFitCtx` -- the ~420 ms active-FT reconstruction every
    fit-mutating Stage 6 verb otherwise rebuilds from scratch -- reused
    across every verb this session issues against the same file: ``edit``,
    ``merge``, ``split``, ``accept``, ``create``, ``undo``, ``preview`` and
    ``apply``. Hosting the whole verb set (not just preview/apply) is
    deliberate: an interactive single-window edit costs ~516 ms cold,
    essentially all of it the same setup a batch amortizes, so a session that
    only sped up the batch door would leave every interactive click paying
    the full price.

    Opened via :meth:`Pipeline.review_session`. Warm-up (building the shared
    context) is synchronous and happens in :meth:`__enter__`, blocking --
    there is no thread inside the library (settled decision 7 in
    ``scratch/preview-session-plan.md``).

    **Correctness never depends on reuse.** Every verb re-validates a cheap
    on-disk fingerprint (:func:`_compute_fit_ctx_fingerprint`, ~0.8 ms)
    before doing any work; a mismatch (a foreign writer touched the file, or
    this is the session's first use) rebuilds the shared context from
    scratch -- identically to what the sessionless functions this class
    wraps do on their own when no ``shared`` is passed. After each of the
    session's OWN writes, the fingerprint is RE-READ from disk, never
    predicted, so a foreign writer landing in the same instant is still
    caught on the session's next use.

    Retains ~26 MB of active-FT arrays for its lifetime (D5); lifetime is
    entirely caller-controlled (``with`` block, or explicit :meth:`close`) --
    there is no module-level cache, so a session that is never opened, or one
    that is closed, costs nothing beyond the object itself.

    Not thread-safe, holds no lock, and does not protect the file from a
    second writer (settled decision 7): single-writer discipline per file is
    the caller's, exactly as it is for every sessionless verb.
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
        """Release the retained shared context (D5). Idempotent."""
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
        """
        self._fingerprint = _compute_fit_ctx_fingerprint(self._path)

    # -- single-window verbs --------------------------------------------------

    def review_edit(
        self,
        window_id: int,
        *,
        add: Sequence[float] = (),
        remove: Sequence[float] = (),
        snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
        frame: Optional[Frame] = None,
    ) -> RefitWindowResult:
        """Session-hosted :func:`refit_window_impl`. See its docstring for
        the full contract; identical here except the shared fit context is
        reused (validated fresh first) rather than rebuilt."""
        shared = self._sync()
        self._drop_staged(base_changed=True)
        result = refit_window_impl(
            self._path,
            window_id,
            add=add,
            remove=remove,
            snap_tol_mhz=snap_tol_mhz,
            frame=frame,
            _shared=shared,
        )
        self._resync_after_write()
        return result

    def review_merge(
        self,
        window_id: int,
        peaks: Sequence[float],
        *,
        snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
        frame: Optional[Frame] = None,
    ) -> RefitWindowResult:
        """Session-hosted :func:`merge_peaks_impl`."""
        shared = self._sync()
        self._drop_staged(base_changed=True)
        result = merge_peaks_impl(
            self._path,
            window_id,
            peaks,
            snap_tol_mhz=snap_tol_mhz,
            frame=frame,
            _shared=shared,
        )
        self._resync_after_write()
        return result

    def review_split(
        self,
        window_id: int,
        peak: float,
        *,
        into: int = 2,
        snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
        frame: Optional[Frame] = None,
    ) -> RefitWindowResult:
        """Session-hosted :func:`split_peak_impl`."""
        shared = self._sync()
        self._drop_staged(base_changed=True)
        result = split_peak_impl(
            self._path,
            window_id,
            peak,
            into=into,
            snap_tol_mhz=snap_tol_mhz,
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
        snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
        frame: Optional[Frame] = None,
    ) -> Optional[RefitWindowResult]:
        """Session-hosted :func:`review_accept_impl`."""
        shared = self._sync()
        self._drop_staged(base_changed=True)
        result = review_accept_impl(
            self._path,
            window_id,
            candidate_freq=candidate_freq,
            snap_tol_mhz=snap_tol_mhz,
            frame=frame,
            _shared=shared,
        )
        self._resync_after_write()
        return result

    def review_create(
        self,
        anchor_mhz: float,
        *,
        snap_tol_mhz: float = REFIT_SNAP_TOL_MHZ,
        frame: Optional[Frame] = None,
    ) -> CreateWindowResult:
        """Session-hosted :func:`create_window_impl`."""
        shared = self._sync()
        self._drop_staged(base_changed=True)
        result = create_window_impl(
            self._path,
            anchor_mhz,
            snap_tol_mhz=snap_tol_mhz,
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
        """Session-hosted :func:`review_undo_impl`."""
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
        curation_path: Union[str, Path],
        *,
        frame: Optional[Frame] = None,
    ) -> ReviewPreviewResult:
        """Session-hosted :func:`review_preview_impl`. Stages its finished,
        cascaded, in-memory outcome (D4) so an immediately-following
        :meth:`review_apply` of the identical plan against an unchanged base
        can persist it directly rather than recomputing -- see that method.
        """
        shared = self._sync()
        # A fresh preview supersedes any earlier drift note.
        self._pending_base_changed = False
        run = _run_review_preview(self._path, curation_path, frame=frame, shared=shared)
        if run.ctx is not None and run.review is not None:
            assert self._fingerprint is not None
            self._staged = _StagedPreview(
                fingerprint=self._fingerprint,
                curation_path=str(curation_path),
                frame=frame,
                resolved_plan=list(run.result.plan),
                warnings=list(run.result.warnings),
                ctx=run.ctx,
                cascaded_wids=list(run.cascaded_wids),
                review=run.review,
            )
        else:
            self._staged = None
        return run.result

    def _persist_staged(self, staged: _StagedPreview) -> List[int]:
        """The write half of D4's staged-reuse accept: re-gate and take the
        undo baseline exactly as a live accept would (via :func:`_open_batch`
        -- the only function structurally permitted to take that baseline),
        then persist the ALREADY-cascaded fit and the ALREADY-derived review
        from the staged preview -- never re-cascading or re-deriving, so the
        persisted bytes are guaranteed to be exactly what the preview showed
        rather than a second computation trusted to agree with the first.
        """
        gate_ctx = _open_batch(
            self._path, snap_tol_mhz=REFIT_SNAP_TOL_MHZ, shared=self._shared
        )
        staged.ctx.baseline_taken = gate_ctx.baseline_taken
        return _finish_batch(
            staged.ctx,
            self._path,
            snap_tol_mhz=REFIT_SNAP_TOL_MHZ,
            cascaded=staged.cascaded_wids,
            precomputed_review=staged.review,
        )

    def review_apply(
        self,
        curation_path: Union[str, Path],
        *,
        frame: Optional[Frame] = None,
    ) -> CurationApplyResult:
        """Session-hosted :func:`apply_curation_impl`, with D4's staged
        reuse: when an immediately-preceding :meth:`review_preview` staged
        the identical plan (same curation file, same ``frame``, same
        resolved actions) against a base that has not moved since (the
        fingerprint captured at preview time still matches), this persists
        that finished result directly instead of re-running the appliers,
        the cascade, and the review derivation. Any mismatch -- a different
        plan, or a base that moved -- falls back to a full, ordinary apply,
        identical to the sessionless :func:`apply_curation_impl`.

        ``base_changed`` on the result is ``True`` only when a staged preview
        existed but had to be dropped because the base moved out from under
        it (a foreign write, or another mutating verb issued on this session
        in between) -- never merely because no preview preceded this call.
        """
        shared = self._sync()
        base_changed = self._pending_base_changed
        self._pending_base_changed = False

        staged = self._staged
        if staged is not None:
            same_request = (
                staged.curation_path == str(curation_path)
                and staged.frame == frame
                and staged.fingerprint == self._fingerprint
            )
            if same_request:
                plan, _, _ = _resolve_curation_call(self._path, curation_path, frame)
                if plan == staged.resolved_plan:
                    self._persist_staged(staged)
                    self._staged = None
                    self._resync_after_write()
                    return CurationApplyResult(
                        plan=plan,
                        warnings=staged.warnings,
                        applied=len(plan),
                        dry_run=False,
                        base_changed=base_changed,
                    )
            # Staged, but it does not match this call -- irrelevant now.
            self._staged = None

        result = apply_curation_impl(
            self._path, curation_path, frame=frame, _shared=shared
        )
        if base_changed:
            result = replace(result, base_changed=True)
        self._resync_after_write()
        return result
