"""
Shared implementation for Stage 6: review and candidate ledger.

Pass 1 — candidate ledger derivation and the ``review show`` read path.

The ledger is a pure function of the already-persisted Stage 5 audit trail
(``FittingResult.audit_trail``) and rescue events
(``FittingResult.rescue_events``).  It is derived on demand; no re-fitting
and no writes to the Stage 5 group.

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.data_structures import (
    AuditStep,
    FittingResult,
    LedgerCandidate,
    RescueCandidateInfo,
    RescueRoundInfo,
    Sideband,
    SpectrumFit,
)
from ..fitting.peak_model import molecular_frequency as _molecular_frequency
from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
from .stage0_impl import load_fid_from_pipeline_impl

import h5py
import numpy as np

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

# If a candidate's evidence is within this factor of the accept gate it
# passes the bar even when its raw SNR is below DEFAULT_DISPLAY_BAR.
_NEAR_GATE_FACTOR: float = 10.0

# Deduplicate candidates whose molecular frequencies are within this window.
_DEDUP_TOL_MHZ: float = 0.02  # 20 kHz; roughly half an active-FT bin at 13 µs


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
        # A ``tentative`` decision is the add-loop's *explicit* "held as a
        # marginal near-miss" flag (the patience mechanism) -- revivable by
        # definition, so it surfaces regardless of the AICc delta magnitude.
        # A ``reject`` is surfaced only when it was a near-gate miss: ``ev`` is
        # the raw positive AICc delta (>= 0 for a rejected K+1 model); a
        # *marginal* reject has a small delta, a *decisive* reject a large one.
        # Pass rejects whose delta is within ``_NEAR_GATE_FACTOR`` of the gate
        # value (0) -- the "within an order of magnitude of the bar" criterion
        # from §D of the planning doc.
        sites = candidate.get("sites") or []
        if any("tentative" in str(s) for s in sites):
            return True
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

    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)

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
            )
        )

    return sorted(all_candidates, key=lambda c: c.frequency_mhz)
