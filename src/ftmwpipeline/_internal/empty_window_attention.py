"""The ``empty_window_residual`` review item: a window the fit holds no line in.

Stage 5 can finish a window of its plan with no fitted line (its seeds rejected,
gated as spurs, or pruned by the per-node cleanup) while the residual on the
window's edge is still coherent. The structural replan then records the flag as
``not merged`` (an empty window has no fitted line straddling its boundary), and
nothing else points the user at the window. This module turns those records into
an attention reason (``dev-docs/CONTRACT_STRATEGY.md`` §Review attention).

It is advisory and read-only: it reads only what Stage 5 recorded (the thaw and
replan handshake records, the fit's own ``residual_edge_threshold``, the gated
spurs) plus the fitted plan and the Stage 3 peak list, and never changes a fit.

Also here: :func:`lineless_window_fit`, the empty
:class:`~ftmwpipeline.core.data_structures.FittingResult` the review surfaces
(``review show``, the report) draw such a window from -- its data on the
window's range, with nothing fitted, so the residual is the data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..core.data_structures import (
    AttentionReason,
    FittingResult,
    FitWindow,
    Peak,
    SpectralWindow,
    SpectrumFit,
    Stage6Review,
)

__all__ = [
    "EMPTY_WINDOW_KINDS",
    "EMPTY_WINDOW_RESIDUAL",
    "EMPTY_WINDOW_SPUR",
    "empty_window_reasons",
    "flagged_empty_edges",
    "flagged_lineless_ids",
    "lineless_window_fit",
    "review_lineless_window_fits",
    "superseded_window_ids",
]

#: The queued kind (``ATTENTION_KINDS``): at least one Stage 3 peak in the
#: window is not on a gated spur, so a line may be missing.
EMPTY_WINDOW_RESIDUAL = "empty_window_residual"

#: The advisory twin: every Stage 3 peak in the window sits on a gated spur, so
#: the edge residual is most likely the spur's skirt beyond its mask. Shown on
#: the window's status, not queued (``_ADVISORY_REASON_KINDS``).
EMPTY_WINDOW_SPUR = "empty_window_spur"

#: Both kinds an empty, edge-flagged window can carry.
EMPTY_WINDOW_KINDS: Tuple[str, ...] = (EMPTY_WINDOW_RESIDUAL, EMPTY_WINDOW_SPUR)

_EDGE_ORDER = ("low", "high")


def _over(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value > threshold


def flagged_empty_edges(
    spectrum_fit: SpectrumFit, window_id: int, threshold: float
) -> List[Tuple[str, float]]:
    """The edges of ``window_id`` Stage 5's edge handshake left flagged.

    Reads the thaw records (``dependent_window_id``) and then the structural
    replan records (``triggering_window_id``), in that order -- the order Stage
    5 runs them -- and keeps, per edge, the latest verdict: an accepted thaw
    resolves the edge; a rejected thaw or a replan record that was not applied
    flags it with its ``S_coh`` when that is above ``threshold``. Returns
    ``(side, s_coh)`` pairs, low edge first; empty when no edge stays flagged.
    """
    state: Dict[str, Optional[float]] = {}

    def _flag(side: str, s_coh: float) -> None:
        if not _over(s_coh, threshold):
            return
        prior = state.get(side)
        state[side] = s_coh if prior is None else max(prior, s_coh)

    for t in spectrum_fit.thaw_history:
        if int(t.dependent_window_id) != window_id:
            continue
        if t.accepted:
            state[str(t.edge_side)] = None
        else:
            _flag(str(t.edge_side), float(t.edge_coherence_before))
    for r in spectrum_fit.replan_history:
        if int(r.triggering_window_id) != window_id or r.accepted:
            continue
        _flag(str(r.edge_side), float(r.edge_coherence_before))
    out: List[Tuple[str, float]] = []
    for side in _EDGE_ORDER:
        s_coh = state.get(side)
        if s_coh is not None:
            out.append((side, float(s_coh)))
    return out


def _gated_spur_near(
    freq_mhz: float,
    gated_spurs: Sequence[Mapping[str, Any]],
    spur_centers_mhz: Sequence[float],
    tol_mhz: float,
) -> Optional[Tuple[float, str]]:
    """The nearest gated spur within ``tol_mhz`` of ``freq_mhz`` as
    ``(center_mhz, source)`` (``source`` empty when the fit recorded none), or
    ``None`` when no gated spur is that close."""
    best: Optional[Tuple[float, float, str]] = None
    for sp in gated_spurs:
        try:
            c = float(sp.get("center_mhz", float("nan")))
        except (TypeError, ValueError):
            continue
        sep = abs(c - freq_mhz)
        if math.isfinite(sep) and sep <= tol_mhz and (best is None or sep < best[0]):
            best = (sep, c, str(sp.get("source", "") or ""))
    if best is None:
        for c in spur_centers_mhz:
            sep = abs(float(c) - freq_mhz)
            if sep <= tol_mhz and (best is None or sep < best[0]):
                best = (sep, float(c), "")
    return None if best is None else (best[1], best[2])


def empty_window_reasons(
    spectrum_fit: SpectrumFit,
    plan_windows: Sequence[FitWindow],
    stage3_peaks: Sequence[Peak],
    *,
    excluded_window_ids: Set[int],
    spur_tol_mhz: float,
) -> Dict[int, AttentionReason]:
    """The empty-window reason of every window that carries one.

    A window of ``plan_windows`` (the fitted plan) is flagged when the fit holds
    no line in it, it is not in ``excluded_window_ids`` (the windows Stage 6
    created, and those a fit-changing decision was recorded on), and
    :func:`flagged_empty_edges` finds a flagged edge against the fit's own
    ``residual_edge_threshold``. A fit that recorded no threshold flags nothing.

    The reason is :data:`EMPTY_WINDOW_RESIDUAL` (queued) when at least one Stage
    3 peak the plan put in the window is not on a gated spur, or the window
    holds none; it is the advisory :data:`EMPTY_WINDOW_SPUR` when every one sits
    on a gated spur, whose skirt beyond its mask then most likely explains the
    edge residual.

    ``spur_tol_mhz`` is how close a Stage 3 peak must sit to a gated spur to be
    reported on it (the ``spur_adjacent`` tolerance).
    """
    raw_thr = spectrum_fit.parameters.get("residual_edge_threshold")
    try:
        threshold = float(raw_thr) if raw_thr is not None else float("nan")
    except (TypeError, ValueError):
        threshold = float("nan")
    if not math.isfinite(threshold) or threshold <= 0.0:
        return {}

    live = {
        int(wf.window_id)
        for wf in spectrum_fit.window_fits
        if wf.window_id is not None and wf.fitted_peaks
    }
    gated_spurs = list((spectrum_fit.diagnostics or {}).get("gated_spurs") or [])
    spur_centers = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", []) or []
    ]

    out: Dict[int, AttentionReason] = {}
    for win in plan_windows:
        wid = int(win.window_id)
        if wid in live or wid in excluded_window_ids:
            continue
        edges = flagged_empty_edges(spectrum_fit, wid, threshold)
        if not edges:
            continue

        candidates: List[Dict[str, Any]] = []
        for idx in win.free_peak_indices:
            if not 0 <= int(idx) < len(stage3_peaks):
                continue
            pk = stage3_peaks[int(idx)]
            freq = float(pk.frequency)
            spur = _gated_spur_near(freq, gated_spurs, spur_centers, spur_tol_mhz)
            cand: Dict[str, Any] = {
                "detection_index": int(idx),
                "frequency_mhz": freq,
            }
            if pk.snr is not None and math.isfinite(float(pk.snr)):
                cand["snr"] = float(pk.snr)
            cand["gated_spur"] = spur is not None
            if spur is not None:
                cand["spur_center_mhz"] = spur[0]
                cand["spur_source"] = spur[1]
            candidates.append(cand)
        candidates.sort(key=lambda c: float(c["frequency_mhz"]))

        worst = max(s for _, s in edges)
        edge_txt = " and ".join(f"{side} edge (S_coh {s:.2f})" for side, s in edges)
        cand_txt = ""
        if candidates:
            parts = []
            for c in candidates:
                bits = []
                if "snr" in c:
                    bits.append(f"SNR {c['snr']:.1f}")
                if c["gated_spur"]:
                    src = f" ({c['spur_source']})" if c["spur_source"] else ""
                    bits.append(
                        f"on the gated spur at {c['spur_center_mhz']:.4f} MHz{src}"
                    )
                extra = f" ({', '.join(bits)})" if bits else ""
                parts.append(f"{c['frequency_mhz']:.4f} MHz{extra}")
            cand_txt = f"; Stage 3 peak(s) here: {', '.join(parts)}"
        all_spur = bool(candidates) and all(c["gated_spur"] for c in candidates)
        if all_spur:
            kind = EMPTY_WINDOW_SPUR
            saturated = any("saturated" in c["spur_source"] for c in candidates)
            hint = (
                f"the edge residual is likely the {'saturated ' if saturated else ''}"
                "spur's skirt beyond its mask; if a line hides under the spur, "
                "create a window at it (review create) and add the line to the new "
                "window"
            )
        else:
            kind = EMPTY_WINDOW_RESIDUAL
            hint = (
                "a line may be missing: to fit it, create a window at it (review "
                "create) and add the line to the new window; to leave the window "
                "empty, mark it reviewed (review accept)"
            )
        out[wid] = AttentionReason(
            kind=kind,
            detail=(
                f"the fit holds no line in this window, but Stage 5 flagged a "
                f"coherent residual on its {edge_txt}, threshold "
                f"{threshold:g}{cand_txt} -- {hint}"
            ),
            severity=float(worst / threshold),
            locations=[float(c["frequency_mhz"]) for c in candidates],
            evidence={
                "edges": [{"side": side, "s_coh": float(s)} for side, s in edges],
                "residual_edge_threshold": float(threshold),
                "candidates": candidates,
            },
        )
    return out


def superseded_window_ids(
    plan_windows: Sequence[FitWindow], created_windows: Sequence[FitWindow]
) -> Set[int]:
    """Plan windows a Stage 6 created window took over: the same id (a widened
    window replaces its plan row) or an overlapping range. The Stage 5 records
    of such a window no longer describe what the fit holds there."""
    if not created_windows:
        return set()
    spans = [
        (int(c.window_id), min(c.freq_range), max(c.freq_range))
        for c in created_windows
    ]
    out: Set[int] = set()
    for w in plan_windows:
        wid = int(w.window_id)
        lo, hi = min(w.freq_range), max(w.freq_range)
        if any(cid == wid or (clo <= hi and lo <= chi) for cid, clo, chi in spans):
            out.add(wid)
    return out


def flagged_lineless_ids(review: Stage6Review, fit_window_ids: Set[int]) -> Set[int]:
    """Windows the review flagged with an empty-window kind
    (:data:`EMPTY_WINDOW_KINDS`) that the fit has no window result for -- the
    windows a bare ``review accept`` may name although the fit does not hold
    them."""
    return {
        int(wid)
        for wid, st in review.window_statuses.items()
        if int(wid) not in fit_window_ids
        and any(r.kind in EMPTY_WINDOW_KINDS for r in st.attention_reasons)
    }


def lineless_window_fit(window: FitWindow, shape: str) -> FittingResult:
    """An empty :class:`FittingResult` on ``window``'s range: no line, no
    baseline, no frozen contributor, so a plot of it draws the data and a
    residual equal to the data. For display only; never persisted."""
    lo, hi = (float(v) for v in window.freq_range)
    spectral = SpectralWindow(
        None,
        np.empty(0, dtype=float),
        np.empty(0, dtype=complex),
        (lo, hi),
        window_id=int(window.window_id),
    )
    return FittingResult(
        success=False,
        window=spectral,
        window_id=int(window.window_id),
        shape=str(shape),
    )


def review_lineless_window_fits(
    file_path: Union[str, Path],
    review: Stage6Review,
    spectrum_fit: SpectrumFit,
) -> Dict[int, FittingResult]:
    """Lineless fits for the review's flagged windows the fit has no result for.

    One :func:`lineless_window_fit` per window whose review status carries an
    attention reason while ``spectrum_fit`` has no window result for it (an
    ``empty_window_residual`` window), on its fitted-plan range. Empty when
    there is none, without reading the plan.
    """
    have = {
        int(wf.window_id) for wf in spectrum_fit.window_fits if wf.window_id is not None
    }
    wanted = sorted(
        wid
        for wid, st in review.window_statuses.items()
        if st.attention_reasons and int(wid) not in have
    )
    if not wanted:
        return {}
    from .fitted_plan import load_fitted_plan

    by_id = {int(w.window_id): w for w in load_fitted_plan(file_path).plan.windows}
    shape = str(spectrum_fit.parameters.get("shape", "lorentzian"))
    return {
        wid: lineless_window_fit(by_id[wid], shape) for wid in wanted if wid in by_id
    }
