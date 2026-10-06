"""The empty-window review items: a window the fit holds no line in.

Stage 5 can finish a window of its plan with no fitted line (its seeds rejected,
gated as spurs, or pruned by the per-node cleanup) while the residual on the
window's edge is still coherent. The structural replan then records the flag as
``not merged`` (an empty window has no fitted line straddling its boundary), and
nothing else points the user at the window. This module turns those records into
an attention reason (``dev-docs/CONTRACT_STRATEGY.md`` §Review attention):
``empty_window_residual`` (queued) or, when every Stage 3 peak in the window sits
on a gated spur, the advisory ``empty_window_spur``.

It is read-only: it reads only what Stage 5 recorded (the thaw and replan
handshake records, the fit's own ``residual_edge_threshold``, the gated spurs)
plus the fitted plan and the Stage 3 peak list, and never changes a fit.

Also here: :func:`lineless_window_fit`, the empty
:class:`~ftmwpipeline.core.data_structures.FittingResult` the review surfaces
(``review show``, the report) draw such a window from -- its data on the
window's range, with nothing fitted, so the residual is the data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from ..core.absent import Absent
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
    "last_refit_revision",
    "lineless_window_fit",
    "review_lineless_window_fits",
    "superseded_window_ids",
    "takeover_points",
]

#: The queued kind (``ATTENTION_KINDS``): at least one Stage 3 peak in the
#: window is not on a gated spur (or the window holds none), so a line may be
#: missing.
EMPTY_WINDOW_RESIDUAL = "empty_window_residual"

#: The advisory twin: every Stage 3 peak in the window sits on a gated spur, so
#: the edge residual is consistent with the spur's skirt beyond its mask. Shown
#: on the window's status, not queued (``_ADVISORY_REASON_KINDS``).
EMPTY_WINDOW_SPUR = "empty_window_spur"

#: Both kinds an empty, edge-flagged window can carry.
EMPTY_WINDOW_KINDS: Tuple[str, ...] = (EMPTY_WINDOW_RESIDUAL, EMPTY_WINDOW_SPUR)

_EDGE_ORDER = ("low", "high")


def _over(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value > threshold


def last_refit_revision(
    spectrum_fit: SpectrumFit,
    window_id: int,
    dependency_edges: Iterable[Tuple[int, int]] = (),
) -> int:
    """The plan revision after which Stage 5 last re-fit ``window_id``; ``0``
    when no structural merge re-fit it.

    A merge round re-fits its survivor, every window that transitively depends
    on it, and the primary of every accepted thaw whose record it dropped, with
    that primary's dependents (``plan_execution._affected_after_replan``). The
    fit keeps the replan records of earlier rounds even for a window re-fit
    later, so a reader has to know which of them still describe the current
    fit. An accepted record stores the round's re-fit set
    (``refit_window_ids``), which is read as is. A record from a file written
    before the set was stored falls back to the survivor and its dependency
    closure: the dependency edges are the fitted plan's ``(window_id,
    depends_on)`` pairs -- the plan the last merge produced, which every
    earlier round's dependency set is contained in for the windows it did not
    merge. That fallback misses a re-fit thawed primary that does not depend on
    the survivor.
    """
    children: Dict[int, List[int]] = {}
    for child, parent in dependency_edges:
        children.setdefault(int(parent), []).append(int(child))
    last = 0
    for r in spectrum_fit.replan_history:
        if not r.accepted:
            continue
        if r.refit_window_ids is not None:
            if window_id in r.refit_window_ids:
                last = max(last, int(r.revision_after))
            continue
        affected = {int(r.surviving_window_id)}
        frontier = [int(r.surviving_window_id)]
        while frontier:
            for c in children.get(frontier.pop(), []):
                if c not in affected:
                    affected.add(c)
                    frontier.append(c)
        if window_id in affected:
            last = max(last, int(r.revision_after))
    return last


def flagged_empty_edges(
    spectrum_fit: SpectrumFit,
    window_id: int,
    threshold: float,
    *,
    dependency_edges: Iterable[Tuple[int, int]] = (),
) -> List[Tuple[str, float]]:
    """The edges of ``window_id`` Stage 5's edge handshake left flagged on the
    window's current fit.

    Only records of the current fit count. The thaw records Stage 5 kept for
    the window all belong to it (Stage 5 drops a re-fit window's thaw records).
    The replan records it kept do not, so a record whose ``revision_before``
    predates the window's last re-fit (:func:`last_refit_revision`) is ignored.
    The rest are read in the order Stage 5 measured them -- the current fit's
    thaws, then the replan rounds that scanned it -- keeping per edge the
    latest verdict: an accepted thaw resolves the edge; a rejected thaw or a
    replan record that was not applied flags it with its ``S_coh`` when that is
    above ``threshold``. Returns ``(side, s_coh)`` pairs, low edge first; empty
    when no edge stays flagged.
    """
    refit = last_refit_revision(spectrum_fit, window_id, dependency_edges)
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
        if (
            int(r.triggering_window_id) != window_id
            or r.accepted
            or int(r.revision_before) < refit
        ):
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
) -> Optional[Tuple[float, Optional[str]]]:
    """The nearest gated spur within ``tol_mhz`` of ``freq_mhz`` as
    ``(center_mhz, source)`` (``source`` ``None`` when the fit recorded none),
    or ``None`` when no gated spur is that close."""
    best: Optional[Tuple[float, float, Optional[str]]] = None
    for sp in gated_spurs:
        try:
            c = float(sp.get("center_mhz", float("nan")))
        except (TypeError, ValueError):
            continue
        sep = abs(c - freq_mhz)
        if math.isfinite(sep) and sep <= tol_mhz and (best is None or sep < best[0]):
            src = sp.get("source")
            best = (sep, c, str(src) if src else None)
    if best is None:
        for c in spur_centers_mhz:
            sep = abs(float(c) - freq_mhz)
            if sep <= tol_mhz and (best is None or sep < best[0]):
                best = (sep, float(c), None)
    return None if best is None else (best[1], best[2])


def _stage3_snr(pk: Peak) -> Union[float, Absent]:
    """A Stage 3 peak's SNR, ``Absent.UNDEFINED`` when it has none: not
    recorded, not finite, or degenerate (a local noise that is not positive,
    for which earlier writers stored ``0.0``; ``absence_rules``)."""
    if pk.snr is None:
        return Absent.UNDEFINED
    snr = float(pk.snr)
    sd = pk.noise_std_local
    if not math.isfinite(snr) or (sd is not None and float(sd) <= 0.0):
        return Absent.UNDEFINED
    return snr


def _neighbour_distance(
    window: FitWindow, side: str, line_freqs: Sequence[float]
) -> Union[float, Absent]:
    """Distance (MHz) from ``window``'s ``side`` edge to the nearest fitted
    line beyond it; ``Absent.UNDEFINED`` when the fit holds none on that side."""
    lo, hi = min(window.freq_range), max(window.freq_range)
    if side == "low":
        gaps = [lo - f for f in line_freqs if f < lo]
    else:
        gaps = [f - hi for f in line_freqs if f > hi]
    return float(min(gaps)) if gaps else Absent.UNDEFINED


def takeover_points(
    window: FitWindow, locations: Sequence[float], edge_sides: Sequence[str]
) -> List[float]:
    """The frequencies a created window must cover to take ``window`` over:
    the flagged Stage 3 peaks (``locations``), or, when there are none, the
    window's flagged edges (its whole range when none is named)."""
    if locations:
        return [float(f) for f in locations]
    lo, hi = min(window.freq_range), max(window.freq_range)
    points = [lo if side == "low" else hi for side in edge_sides]
    return points or [lo, hi]


def superseded_window_ids(
    plan_windows: Sequence[FitWindow],
    created_windows: Sequence[FitWindow],
    points: Mapping[int, Sequence[float]],
) -> Set[int]:
    """Plan windows a Stage 6 created window took over: one of the same id (a
    widened window replaces its plan row), or created windows that together
    cover every point of ``points[window_id]`` (:func:`takeover_points`). A
    created window that merely overlaps the window without covering what was
    flagged there takes nothing over."""
    if not created_windows:
        return set()
    spans = [
        (int(c.window_id), min(c.freq_range), max(c.freq_range))
        for c in created_windows
    ]
    out: Set[int] = set()
    for w in plan_windows:
        wid = int(w.window_id)
        if any(cid == wid for cid, _, _ in spans):
            out.add(wid)
            continue
        pts = points.get(wid)
        if pts and all(any(lo <= p <= hi for _, lo, hi in spans) for p in pts):
            out.add(wid)
    return out


def empty_window_reasons(
    spectrum_fit: SpectrumFit,
    plan_windows: Sequence[FitWindow],
    stage3_peaks: Sequence[Peak],
    *,
    excluded_window_ids: Set[int],
    spur_tol_mhz: float,
    created_windows: Sequence[FitWindow] = (),
    dependency_edges: Iterable[Tuple[int, int]] = (),
) -> Dict[int, AttentionReason]:
    """The empty-window reason of every window that carries one.

    A window of ``plan_windows`` (the fitted plan, whose ``dependency_edges``
    are passed) is flagged when the fit holds no line in it, it is not in
    ``excluded_window_ids`` (those a fit-changing decision was recorded on), no
    created window took it over (:func:`superseded_window_ids`), and
    :func:`flagged_empty_edges` finds a flagged edge against the fit's own
    ``residual_edge_threshold``. A fit that recorded no threshold flags nothing.

    The reason is :data:`EMPTY_WINDOW_RESIDUAL` (queued) when at least one Stage
    3 peak the plan put in the window is not on a gated spur, or the window
    holds none; it is the advisory :data:`EMPTY_WINDOW_SPUR` when every one sits
    on a gated spur, whose skirt beyond its mask the edge residual is then
    consistent with.

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
    edges_list = [(int(a), int(b)) for a, b in dependency_edges]

    live_fits = [
        wf
        for wf in spectrum_fit.window_fits
        if wf.window_id is not None and wf.fitted_peaks
    ]
    live = {int(wf.window_id) for wf in live_fits if wf.window_id is not None}
    line_freqs = [float(p.frequency_mhz) for wf in live_fits for p in wf.fitted_peaks]
    gated_spurs = list((spectrum_fit.diagnostics or {}).get("gated_spurs") or [])
    spur_centers = [
        float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", []) or []
    ]

    out: Dict[int, AttentionReason] = {}
    for win in plan_windows:
        wid = int(win.window_id)
        if wid in live or wid in excluded_window_ids:
            continue
        edges = flagged_empty_edges(
            spectrum_fit, wid, threshold, dependency_edges=edges_list
        )
        if not edges:
            continue

        candidates: List[Dict[str, Any]] = []
        for idx in win.free_peak_indices:
            if not 0 <= int(idx) < len(stage3_peaks):
                continue
            pk = stage3_peaks[int(idx)]
            freq = float(pk.frequency)
            spur = _gated_spur_near(freq, gated_spurs, spur_centers, spur_tol_mhz)
            spur_source: Union[str, Absent] = Absent.UNDEFINED
            if spur is not None:
                spur_source = spur[1] if spur[1] is not None else Absent.NOT_RUN
            candidates.append(
                {
                    "detection_index": int(idx),
                    "frequency_mhz": freq,
                    "snr": _stage3_snr(pk),
                    "gated_spur": spur is not None,
                    "spur_center_mhz": (
                        spur[0] if spur is not None else Absent.UNDEFINED
                    ),
                    "spur_source": spur_source,
                }
            )
        candidates.sort(key=lambda c: float(c["frequency_mhz"]))

        locations = [float(c["frequency_mhz"]) for c in candidates]
        if superseded_window_ids(
            [win],
            created_windows,
            {wid: takeover_points(win, locations, [side for side, _ in edges])},
        ):
            continue

        worst = max(s for _, s in edges)
        edge_txt = " and ".join(f"{side} edge (S_coh {s:.2f})" for side, s in edges)
        cand_txt = ""
        if candidates:
            parts = []
            for c in candidates:
                bits = []
                if not isinstance(c["snr"], Absent):
                    bits.append(f"SNR {c['snr']:.1f}")
                if c["gated_spur"]:
                    src = c["spur_source"]
                    src_txt = f" ({src})" if isinstance(src, str) else ""
                    bits.append(
                        f"on the gated spur at {c['spur_center_mhz']:.4f} MHz{src_txt}"
                    )
                extra = f" ({', '.join(bits)})" if bits else ""
                parts.append(f"{c['frequency_mhz']:.4f} MHz{extra}")
            cand_txt = f"; Stage 3 peak(s) here: {', '.join(parts)}"
        all_spur = bool(candidates) and all(c["gated_spur"] for c in candidates)
        if all_spur:
            kind = EMPTY_WINDOW_SPUR
            saturated = any(
                isinstance(c["spur_source"], str) and "saturated" in c["spur_source"]
                for c in candidates
            )
            hint = (
                "the edge residual is consistent with the "
                f"{'saturated ' if saturated else ''}spur's skirt beyond its "
                "mask; if a line hides under the spur, create a window at it "
                "(review create) and add the line to the new window"
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
            locations=locations,
            evidence={
                "edges": [
                    {
                        "side": side,
                        "s_coh": float(s),
                        "neighbour_line_distance_mhz": _neighbour_distance(
                            win, side, line_freqs
                        ),
                    }
                    for side, s in edges
                ],
                "residual_edge_threshold": float(threshold),
                "candidates": candidates,
            },
        )
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
