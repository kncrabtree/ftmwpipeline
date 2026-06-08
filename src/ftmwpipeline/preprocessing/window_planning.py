"""
Stage 4 window-planning algorithm.

This module turns the promoted Stage 3 peak list into a :class:`WindowPlan` --
an ordered set of disjoint analysis windows, each carrying the peaks to fit
freely, the strong out-of-band lines whose leakage must be carried frozen, a
fit dependency DAG, and a difficulty class. It is *purely structural*: it makes
no fits and changes no spectrum (see
``dev-docs/planning/stage4-window-assignment.md``).

The algorithm is pure (peaks + arrays in, ``WindowPlan`` out) so it stays
unit-testable; file orchestration/persistence lives in
:mod:`ftmwpipeline._internal.stage4_impl`.

Outline (the plan's eight steps):

1. Compute the rolling complex-edge coherence statistic ``S_coh`` over the
   complex spectrum (:mod:`ftmwpipeline.preprocessing.edge_coherence`). Its
   contiguous above-threshold runs are the spectrum's *leakage-touched*
   regions: a strong line's coherent skirt fills a touched region several MHz
   wide, far wider than the per-point analytic leakage reach.
2. Each promoted peak proposes a tight window -- its core plus
   ``min_window_half_width_mhz``. Extent is deliberately *not* the leakage
   skirt: a strong line's coherent skirt is ~80-100 MHz wide, and its distant
   leakage is carried by other windows as a fixed contributor, not by widening
   this one.
3. Strong lines sharing one leakage-touched region are mutually coupled and
   merge into one *primary joint window* (the 2638 36350/36389 doublet);
   overlapping proposed windows then merge to a fixpoint -> disjoint fit
   windows covering each spectrum point at most once (the hard invariant).
4. A window that sits inside a strong line's leakage-touched region but is a
   *different* window than that line's gets the strong line attached as a
   fixed contributor -- the coherence statistic confirms a coherent skirt
   genuinely reaches it -- which adds a fit dependency edge.
5. Leakage-artifact detections -- a strong line's sidelobes that Stage 3's gap
   pass promoted as peaks -- are pruned from the free set once that line is a
   contributor (its analytic envelope explains them).
6. Difficulty is strong-line-driven: a window is HARD if it contains/near a
   strong line (in-band or as a fixed contributor) or exceeds the width cap;
   EASY otherwise. Too-wide hard windows get a proposed split at a
   complex-edge-clean interior point, or a ``needs_joint_treatment`` marker.
7. The dependency DAG is topologically ordered into parallel batches.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.data_structures import (
    FitWindow,
    FixedContributor,
    MergeRequest,
    Peak,
    PeakClassification,
    WindowDifficulty,
    WindowPlan,
)
from .edge_coherence import (
    DEFAULT_EDGE_M,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_TRIM_M,
    above_threshold_intervals,
    max_cumsum_statistic,
    rolling_coherence,
)
from .leakage import deramp_to_active_start

# Stage 4 parameter defaults. All configurable on the pipeline file.
DEFAULT_MAX_WINDOW_WIDTH_MHZ = 40.0
"""Width cap default. On 2638 a strong line's above-threshold skirt extends to
~40 MHz, so a single window wider than this is already dense/coupled."""

DEFAULT_MIN_FREEZE_SNR = 50.0
"""Freeze-eligibility SNR cutoff (O4-2): a fixed contributor below this is
flagged as a thaw-and-re-fit candidate rather than safely frozen."""

DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ = 2.0
"""Minimum half-width of a window built around an isolated weak line."""

DEFAULT_MAX_PEAKS_PER_WINDOW = 8
"""Per-window promoted-peak cap. A window holding more promoted peaks than the
Stage 5 fit can model jointly (``conservative.max_peaks``, default 8) is split at
its sparsest internal gaps. The default tracks the Stage 5 default so windows are
sized to be fittable; raise both together if Stage 5's ``max_peaks`` is raised. The
cap (with the width cap) is what keeps a dense, ultra-high-SNR spectrum from
collapsing into one unfittable mega-window -- without it the strong-cluster merge
and the overlapping per-peak proto-spans chain hundreds of real lines into a single
GHz-scale window the fit can only partially model."""

DEFAULT_MAX_EDGE_FREE_NEIGHBORS = 3
"""Cap on how many distinct primary windows per dependent window may have their
cycle-orphaned contributors converted to edge-free skirts (the issue-#3
leakage-subtraction recovery). When the Step-7 cycle-breaker drops a
fit-ordering edge it would otherwise discard the attached
:class:`~ftmwpipeline.core.data_structures.FixedContributor`; the few most
dominant orphans (ranked by aggregated predicted skirt) are instead kept as
``edge_free`` so their leakage is still subtracted. Capping the count keeps the
subtraction *targeted*: a dense ultra-high-SNR forest (655) drops dozens of
edges, and converting them all re-creates the global-crude over-subtraction that
regressed the bulk fit (see the Phase-1 negative in the stage5 cross-fixture
report). 3 spans the largest real adjacent cluster (the 360 w288 triplet) while
staying well short of the forest."""

DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD = 0.1
"""Tier-1 attachment threshold (in units of σ_c on the target window) for the
analytic-skirt-magnitude contributor-attachment rule. A strong promoted peak
is attached to a candidate window as a :class:`FixedContributor` when its
predicted mean |skirt| on that window's grid is at least this fraction of
σ_c(w). Default 0.1σ_c gives a median of 2 contributors per window on the
2638 fixture (p95 = 10, max = 16); see O5-10 in stage5-fitting.md."""


@dataclass
class _PPeak:
    """A promoted peak with its position resolved onto the ordered grid."""

    list_index: int  # position in the full Stage 3 peak list
    grid_index: int  # index into the ascending-frequency-ordered grid
    frequency: float
    snr: float
    intensity: float
    is_strong: bool


def _leakage_envelope_fraction(
    delta_f_hz: float, acquisition_us: float, tau_us: Optional[float]
) -> float:
    """Finite-T leakage envelope as a fraction of the source line's peak height.

    ``|S_env(Δf)| / |S(0)| = (1 + e^{-T/τ}) / (2π·|Δf|·τ_eff)`` -- the same
    analytic model used by :func:`estimate_leakage_reach`, evaluated here to
    decide whether a weaker nearby detection sits below a strong line's skirt.
    """
    t_s = acquisition_us * 1e-6
    if tau_us is None:
        tau_eff = t_s
        env_factor = 2.0
    else:
        tau = tau_us * 1e-6
        tau_eff = tau * (-math.expm1(-t_s / tau))
        env_factor = 1.0 + math.exp(-t_s / tau)
    df = abs(delta_f_hz)
    if df <= 0.0:
        return float("inf")
    return env_factor / (2.0 * np.pi * df * tau_eff)


def _ordered_grid(
    freqs: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrum arrays sorted by ascending frequency, plus the order.

    The persisted user spectrum may be descending in frequency (2638 is lower
    sideband). The planning algorithm works in ascending order throughout and
    maps results back to physical frequency at the end.
    """
    order = np.argsort(freqs)
    return freqs[order], complex_spectrum[order], rms_noise[order], order


def _nearest_grid_index(ordered_freqs: np.ndarray, frequency: float) -> int:
    """Index into ``ordered_freqs`` (ascending) of the closest frequency."""
    pos = int(np.searchsorted(ordered_freqs, frequency))
    pos = min(max(pos, 1), len(ordered_freqs) - 1)
    if abs(frequency - ordered_freqs[pos - 1]) <= abs(frequency - ordered_freqs[pos]):
        return pos - 1
    return pos


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping inclusive ``(lo, hi)`` index spans into disjoint ones."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: List[Tuple[int, int]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:  # overlap (touching counts as overlap)
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _split_run_to_cap(gidx_sorted: List[int], cap_idx: float) -> List[List[int]]:
    """Split a sorted grid-index run at its largest gap until every chunk spans at
    most ``cap_idx`` grid steps.

    Used to bound the strong-cluster merge: rather than force-merging every strong
    line that shares one leakage-touched run into one ``(min, max)`` span -- which
    on a dense, ultra-high-SNR spectrum is the whole band -- each strong run is cut
    at its sparsest points so no forced span exceeds the width cap. The cross-window
    coupling between the resulting chunks is carried by the fixed-contributor
    mechanism, not by widening the window.
    """
    if not gidx_sorted:
        return []
    if gidx_sorted[-1] - gidx_sorted[0] <= cap_idx:
        return [gidx_sorted]
    k = int(np.argmax(np.diff(np.asarray(gidx_sorted))))  # split after position k
    return _split_run_to_cap(gidx_sorted[: k + 1], cap_idx) + _split_run_to_cap(
        gidx_sorted[k + 1 :], cap_idx
    )


def _split_span_to_caps(
    lo: int,
    hi: int,
    member_gidx: List[int],
    cap_idx: float,
    max_peaks: int,
) -> List[Tuple[int, int]]:
    """Split a merged ``(lo, hi)`` grid span until each piece holds at most
    ``max_peaks`` promoted peaks AND spans at most ``cap_idx`` grid steps.

    Splits at the largest internal peak gap, placing the boundary at the gap
    midpoint so the resulting windows stay disjoint and each edge peak keeps half
    the gap as margin. ``member_gidx`` is the sorted promoted-peak grid indices
    inside ``[lo, hi]``. A dense forest is genuinely coupled across large spans (the
    bright lines' skirts overlap everywhere) but cannot be fit jointly past
    ``max_peaks``; the cross-window coupling is carried by the fixed contributors,
    the same mechanism that handles a strong line's distant skirt.
    """
    members = [g for g in member_gidx if lo <= g <= hi]
    if (hi - lo <= cap_idx and len(members) <= max_peaks) or len(members) <= 1:
        return [(lo, hi)]
    arr = np.asarray(members)
    k = int(np.argmax(np.diff(arr)))  # largest gap -> split after the k-th member
    mid = (members[k] + members[k + 1]) // 2
    return _split_span_to_caps(
        lo, mid, members, cap_idx, max_peaks
    ) + _split_span_to_caps(mid + 1, hi, members, cap_idx, max_peaks)


def _span_of(spans: List[Tuple[int, int]], idx: int) -> int:
    """Return the position of the span containing ``idx``, or -1 if none."""
    for i, (lo, hi) in enumerate(spans):
        if lo <= idx <= hi:
            return i
    return -1


def _topological_batches(
    window_ids: List[int], edges: List[Tuple[int, int]]
) -> Tuple[List[int], Dict[int, int], List[Tuple[int, int]]]:
    """Kahn topological sort of the fit DAG; assign parallel-batch indices.

    ``edges`` are ``(window, depends_on)``. Returns ``(topological_order,
    batch_by_window, kept_edges)``. A cycle (rare -- mutually-reaching strong
    lines normally merge into one window) is broken deterministically and the
    dropped edges are reported.
    """
    deps: Dict[int, set] = {w: set() for w in window_ids}
    for w, d in edges:
        if d != w and d in deps:
            deps[w].add(d)
    dependents: Dict[int, List[int]] = {w: [] for w in window_ids}
    for w in window_ids:
        for d in deps[w]:
            dependents[d].append(w)

    indeg = {w: len(deps[w]) for w in window_ids}
    batch = {w: 0 for w in window_ids}
    order: List[int] = []
    ready = sorted(w for w in window_ids if indeg[w] == 0)
    while ready:
        w = ready.pop(0)
        order.append(w)
        for d in deps[w]:
            batch[w] = max(batch[w], batch[d] + 1)
        for child in sorted(dependents[w]):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
        ready.sort()

    dropped: List[Tuple[int, int]] = []
    if len(order) != len(window_ids):
        # Cycle: place leftover nodes after everything else, drop their edges.
        leftover = sorted(w for w in window_ids if w not in order)
        next_batch = (max(batch.values()) + 1) if batch else 0
        for w in leftover:
            batch[w] = next_batch
            order.append(w)
            for d in deps[w]:
                dropped.append((w, d))
    kept = [(w, d) for (w, d) in edges if (w, d) not in dropped]
    return order, batch, kept


def build_window_plan(
    peaks: List[Peak],
    freqs: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    *,
    acquisition_us: float,
    tau_us: Optional[float] = None,
    edge_m: int = DEFAULT_EDGE_M,
    trim_m: int = DEFAULT_TRIM_M,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    max_window_width_mhz: float = DEFAULT_MAX_WINDOW_WIDTH_MHZ,
    min_freeze_snr: float = DEFAULT_MIN_FREEZE_SNR,
    min_window_half_width_mhz: float = DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ,
    magnitude_attachment_threshold: float = DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    max_edge_free_neighbors: int = DEFAULT_MAX_EDGE_FREE_NEIGHBORS,
    max_peaks_per_window: int = DEFAULT_MAX_PEAKS_PER_WINDOW,
    probe_freq_mhz: float = 0.0,
    start_us: float = 0.0,
) -> WindowPlan:
    """Build the Stage 4 fit plan from the promoted Stage 3 peaks.

    Parameters
    ----------
    peaks : list of Peak
        The full persisted Stage 3 peak list. Only peaks with
        ``properties['promoted']`` truthy are planned; ``free_peak_indices`` and
        fixed-contributor references index back into *this* list.
    freqs, complex_spectrum, rms_noise : np.ndarray
        The persisted user spectrum: frequency axis (MHz), complex FT, and the
        canonical Stage 2 per-point RMS noise. Equal length, 1D.
    acquisition_us : float
        Active FID acquisition ``T`` (µs) for the analytic leakage reach.
    tau_us : float, optional
        Assumed shared decay constant; ``None`` = undamped/boxcar limit.
    edge_m, trim_m : int
        Coherence-statistic band widths (rolling scan / trim refinement).
    edge_threshold : float
        ``S_coh`` threshold ``T_edge``.
    max_window_width_mhz : float
        Width cap; a wider window is HARD and gets a split proposal.
    min_freeze_snr : float
        Freeze-eligibility SNR cutoff for fixed contributors (O4-2).
    min_window_half_width_mhz : float
        Minimum half-width of a window built around an isolated weak line.
    max_peaks_per_window : int
        Per-window promoted-peak cap. The strong-cluster merge is bounded at
        ``max_window_width_mhz`` and the merged spans are then split at their
        sparsest internal gaps until each window holds at most this many promoted
        peaks (and is at most ``max_window_width_mhz`` wide). Tracks the Stage 5
        ``conservative.max_peaks`` so windows are sized to be fittable; see
        :data:`DEFAULT_MAX_PEAKS_PER_WINDOW`.
    magnitude_attachment_threshold : float
        Tier-1 contributor-attachment threshold in units of σ_c. A strong
        promoted peak is attached to a window's ``fixed_contributors`` when
        its predicted mean |skirt| on that window's grid is at least
        ``threshold * sigma_c(w)``. Default
        :data:`DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD`.
    max_edge_free_neighbors : int
        Cap on the number of distinct primary windows whose cycle-orphaned
        contributors are converted to edge-free skirts per dependent window
        (see :data:`DEFAULT_MAX_EDGE_FREE_NEIGHBORS`). Keeps the
        leakage-subtraction recovery targeted on a dense spectrum.
    probe_freq_mhz : float
        Probe (LO) frequency in MHz, used to de-ramp the spectrum to the
        active-region turn-on before the edge-coherence statistic (see
        :func:`~ftmwpipeline.preprocessing.leakage.deramp_to_active_start`).
    start_us : float
        Active-region start time ``t0`` in microseconds for that de-ramp.
        ``0`` (the default) makes the de-ramp the identity -- correct for a
        synthetic spectrum with no turn-on ramp.

    Returns
    -------
    WindowPlan
        The fit plan: disjoint windows, dependency DAG, topological order,
        parallel batches, parameters and diagnostics.

    Raises
    ------
    ValueError
        If the spectrum arrays differ in length or ``acquisition_us <= 0``.
    """
    if not (len(freqs) == len(complex_spectrum) == len(rms_noise)):
        raise ValueError("freqs, complex_spectrum and rms_noise must be equal length")
    if acquisition_us <= 0:
        raise ValueError("acquisition_us must be positive")

    parameters: Dict[str, Any] = {
        "edge_m": int(edge_m),
        "trim_m": int(trim_m),
        "edge_threshold": float(edge_threshold),
        "max_window_width_mhz": float(max_window_width_mhz),
        "min_freeze_snr": float(min_freeze_snr),
        "min_window_half_width_mhz": float(min_window_half_width_mhz),
        "magnitude_attachment_threshold": float(magnitude_attachment_threshold),
        "max_edge_free_neighbors": int(max_edge_free_neighbors),
        "max_peaks_per_window": int(max_peaks_per_window),
        "acquisition_us": float(acquisition_us),
        "tau_us": tau_us,
        "start_us": float(start_us),
        "probe_freq_mhz": float(probe_freq_mhz),
    }

    ofreqs, ospec, orms, _order = _ordered_grid(
        np.asarray(freqs, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        np.asarray(rms_noise, dtype=float),
    )
    # Reference the spectrum to the active-region turn-on: the pipeline FT is a
    # full-record rfft, so a strong line's truncation-leakage skirt carries an
    # exp(+/-i2pi f t0) ramp that makes the coherent edge statistic cancel on
    # genuine leakage. The de-ramp restores it (see leakage-detection-rework).
    ospec = deramp_to_active_start(ofreqs, ospec, probe_freq_mhz, start_us)
    n = ofreqs.size
    diagnostics: Dict[str, Any] = {}

    # Promoted peaks resolved onto the ordered grid.
    promoted: List[_PPeak] = []
    for li, p in enumerate(peaks):
        if not p.properties.get("promoted"):
            continue
        gi = _nearest_grid_index(ofreqs, p.frequency) if n else 0
        promoted.append(
            _PPeak(
                list_index=li,
                grid_index=gi,
                frequency=float(p.frequency),
                snr=float(p.snr) if p.snr is not None else 0.0,
                intensity=float(p.intensity),
                is_strong=(p.classification == PeakClassification.STRONG),
            )
        )
    if not promoted or n == 0:
        return WindowPlan(parameters=parameters, diagnostics=diagnostics)

    step_mhz = float(np.mean(np.diff(ofreqs))) if n > 1 else 1.0
    step_mhz = abs(step_mhz) or 1.0

    # --- Step 1: rolling coherence statistic + leakage-touched regions ------
    rolling = rolling_coherence(ospec, orms, band_m=edge_m)
    touched = above_threshold_intervals(rolling, edge_threshold)

    # --- Step 2: per-peak proposed windows (tight, uniform) -----------------
    # A window's extent is a peak's core plus min_window_half_width -- it is
    # NOT the leakage-touched run. A strong line's run is ~80-100 MHz wide; its
    # distant leakage is carried by other windows as a fixed contributor, not
    # by widening this window (see leakage-detection-rework.md).
    half_idx = max(int(round(min_window_half_width_mhz / step_mhz)), edge_m)
    proto_spans: List[Tuple[int, int]] = []
    for pk in promoted:
        lo = max(pk.grid_index - half_idx, 0)
        hi = min(pk.grid_index + half_idx, n - 1)
        proto_spans.append((lo, hi))

    # --- Step 3: strong-cluster grouping (primary joint windows) ------------
    # Strong lines that share one leakage-touched region are mutually coupled
    # (the S_coh statistic stays above threshold all the way between them) and
    # must be fit together -- the 2638 36350/36389 doublet is the reference
    # case. Force their proposed windows to merge, but BOUND the forced span at
    # ``max_window_width_mhz``: on a dense, ultra-high-SNR spectrum a strong
    # line's leakage keeps S_coh above threshold across the whole band, so an
    # unbounded force-merge would collapse hundreds of distinct lines into one
    # GHz-scale window. Each strong run is split at its sparsest gaps so no
    # forced span exceeds the cap; distant coupling is carried by the
    # fixed-contributor mechanism (Step 4), not by widening the window.
    cap_idx = max_window_width_mhz / step_mhz
    strong_in_interval: Dict[int, List[_PPeak]] = {}
    for pk in promoted:
        if not pk.is_strong:
            continue
        ti = _span_of(touched, pk.grid_index)
        if ti >= 0:
            strong_in_interval.setdefault(ti, []).append(pk)
    for ti, strong_pks in strong_in_interval.items():
        if len(strong_pks) <= 1:
            continue
        gidx = sorted(pk.grid_index for pk in strong_pks)
        for chunk in _split_run_to_cap(gidx, cap_idx):
            if len(chunk) > 1:
                proto_spans.append((min(chunk), max(chunk)))

    merged = _merge_spans(proto_spans)

    # --- Cap split: enforce <= max_peaks_per_window AND <= the width cap -----
    # The bounded strong-merge above stops a forced GHz span, but in a dense
    # forest the overlapping per-peak proto-spans (and the capped strong spans)
    # still chain into windows far wider and far more peak-dense than the Stage 5
    # fit can model. Split each merged span at its sparsest internal peak gaps
    # until every window holds at most ``max_peaks_per_window`` promoted peaks and
    # spans at most ``max_window_width_mhz``; the cross-window coupling is carried
    # by the fixed contributors.
    all_gidx = sorted(pk.grid_index for pk in promoted)
    capped: List[Tuple[int, int]] = []
    for lo, hi in merged:
        span_gidx = [g for g in all_gidx if lo <= g <= hi]
        capped.extend(
            _split_span_to_caps(lo, hi, span_gidx, cap_idx, max_peaks_per_window)
        )
    merged = capped

    # --- Build fit windows from the merged disjoint spans -------------------
    windows: List[FitWindow] = []
    members_by_window: Dict[int, List[_PPeak]] = {}
    for wid, (lo, hi) in enumerate(merged):
        members = [pk for pk in promoted if lo <= pk.grid_index <= hi]
        members_by_window[wid] = members
        strong = [pk for pk in members if pk.is_strong]
        fr = (float(ofreqs[lo]), float(ofreqs[hi]))
        lo_band = ospec[lo : min(lo + trim_m, n)]
        hi_band = ospec[max(hi - trim_m + 1, 0) : hi + 1]
        lo_sig = float(np.mean(orms[lo : min(lo + trim_m, n)]))
        hi_sig = float(np.mean(orms[max(hi - trim_m + 1, 0) : hi + 1]))
        lo_stat, _ = max_cumsum_statistic(lo_band, lo_sig)
        hi_stat, _ = max_cumsum_statistic(hi_band, hi_sig)
        windows.append(
            FitWindow(
                window_id=wid,
                freq_range=(min(fr), max(fr)),
                free_peak_indices=[pk.list_index for pk in members],
                diagnostics={
                    "grid_span": [int(lo), int(hi)],
                    "n_strong_in_band": len(strong),
                    "edge_statistic_lo": lo_stat,
                    "edge_statistic_hi": hi_stat,
                    "trimmed_width_mhz": abs(fr[1] - fr[0]),
                },
            )
        )

    return _finalize_plan(
        windows=windows,
        ofreqs=ofreqs,
        orms=orms,
        rolling=rolling,
        touched=touched,
        promoted=promoted,
        parameters=parameters,
        diagnostics=diagnostics,
        edge_m=edge_m,
        trim_m=trim_m,
        edge_threshold=edge_threshold,
        max_window_width_mhz=max_window_width_mhz,
        min_freeze_snr=min_freeze_snr,
        magnitude_attachment_threshold=magnitude_attachment_threshold,
        max_edge_free_neighbors=max_edge_free_neighbors,
        acquisition_us=acquisition_us,
        tau_us=tau_us,
        plan_revision=0,
    )


def _finalize_plan(
    *,
    windows: List[FitWindow],
    ofreqs: np.ndarray,
    orms: np.ndarray,
    rolling: np.ndarray,
    touched: List[Tuple[int, int]],
    promoted: List[_PPeak],
    parameters: Dict[str, Any],
    diagnostics: Dict[str, Any],
    edge_m: int,
    trim_m: int,
    edge_threshold: float,
    max_window_width_mhz: float,
    min_freeze_snr: float,
    magnitude_attachment_threshold: float,
    max_edge_free_neighbors: int,
    acquisition_us: float,
    tau_us: Optional[float],
    plan_revision: int,
) -> WindowPlan:
    """Run the spectrum-state-dependent finalization steps (4-7) of the plan.

    Given a window list whose ``freq_range``, ``free_peak_indices``, and
    ``diagnostics['grid_span']`` are set, this:

    * recomputes the fixed-contributor lists and dependency edges from the
      leakage-touched regions (step 4),
    * prunes leakage-artifact detections (step 5),
    * classifies difficulty + sets ``split_proposal`` / ``needs_joint_treatment``
      (step 6),
    * topo-sorts and computes parallel batches (step 7),
    * records plan-level diagnostics.

    Shared by :func:`build_window_plan` (initial plan) and :func:`replan`
    (after structural :class:`MergeRequest` application).
    """
    by_list_index = {pk.list_index: pk for pk in promoted}
    members_by_window = {
        w.window_id: [pk for pk in promoted if pk.list_index in w.free_peak_indices]
        for w in windows
    }
    strong_by_window = {
        w.window_id: [pk for pk in members_by_window[w.window_id] if pk.is_strong]
        for w in windows
    }
    strong_in_interval: Dict[int, List[_PPeak]] = {}
    for pk in promoted:
        if not pk.is_strong:
            continue
        ti = _span_of(touched, pk.grid_index)
        if ti >= 0:
            strong_in_interval.setdefault(ti, []).append(pk)

    # --- Step 4: fixed contributors + fit dependency edges ------------------
    # Recomputed from scratch so a merged window's now-internal contributors
    # drop off automatically.
    #
    # Tier-1 magnitude-based attachment (O5-10): for every (strong promoted
    # peak s, candidate window w) pair, predict the mean |skirt| s would
    # contribute to w's grid and attach s as a FixedContributor of w when
    # that prediction crosses ``magnitude_attachment_threshold * sigma_c(w)``.
    # Replaces the previous "touched-region overlap" gate, which missed the
    # long-tail cumulative-skirt bias diagnosed in scratch/stage5-validation/
    # (see dev-docs/planning/stage5-fitting.md O5-10).
    for w in windows:
        w.fixed_contributors = []

    # Which window owns each strong promoted peak (its primary window).
    primary_of_strong: Dict[int, int] = {}
    for w in windows:
        for s in strong_by_window[w.window_id]:
            primary_of_strong[s.list_index] = w.window_id

    edges: List[Tuple[int, int]] = []
    for w in windows:
        wlo, whi = w.diagnostics["grid_span"]
        w_center_mhz = 0.5 * (float(ofreqs[wlo]) + float(ofreqs[whi]))
        # Per-window noise reference: mean Stage 2 rms over the window's grid
        # span converted to per-quadrature sigma. ``sigma_c = rms / sqrt(2)``
        # matches the diagnostic convention (the |residual| / Rayleigh test).
        w_rms_mean = float(np.mean(orms[wlo : whi + 1]))
        sigma_c_w = w_rms_mean / math.sqrt(2.0) if w_rms_mean > 0 else 0.0
        threshold = magnitude_attachment_threshold * sigma_c_w

        primaries_attached: set = set()
        for li, primary_wid in primary_of_strong.items():
            if primary_wid == w.window_id:
                continue
            s_pk = by_list_index[li]
            df_mhz = abs(s_pk.frequency - w_center_mhz)
            if df_mhz <= 0.0:
                # In-grid same-frequency case (shouldn't happen for primary
                # vs dependent windows but guards the divide-by-zero in the
                # envelope helper).
                continue
            # |skirt|/|peak| = (1+e^{-T/τ})/(2π·Δf·τ_eff); intensity is the
            # peak FT magnitude (= 0.5·A·τ_eff), so predicted_mean_skirt
            # equals intensity · |skirt|/|peak| in the far-field limit, which
            # is the regime every cross-window contributor sits in.
            envelope_ratio = _leakage_envelope_fraction(
                df_mhz * 1e6, acquisition_us, tau_us
            )
            predicted_skirt = s_pk.intensity * envelope_ratio
            if predicted_skirt < threshold:
                continue
            w.fixed_contributors.append(
                FixedContributor(
                    peak_index=li,
                    primary_window_id=primary_wid,
                    frequency_mhz=s_pk.frequency,
                    freeze_eligible=s_pk.snr >= min_freeze_snr,
                )
            )
            primaries_attached.add(primary_wid)
        for primary_wid in primaries_attached:
            edges.append((w.window_id, primary_wid))

    # --- Step 5: prune leakage-artifact detections from the free set --------
    total_pruned = 0
    for w in windows:
        # Strong influences on this window: in-band strong free peaks +
        # out-of-band fixed contributors.
        sources: List[Tuple[float, float, float]] = []  # (freq, intensity, snr)
        for pk in strong_by_window[w.window_id]:
            sources.append((pk.frequency, pk.intensity, pk.snr))
        for fc in w.fixed_contributors:
            src = by_list_index.get(fc.peak_index)
            if src is not None:
                sources.append((src.frequency, src.intensity, src.snr))
        if not sources:
            continue
        kept: List[int] = []
        pruned: List[int] = []
        for li in w.free_peak_indices:
            pk = by_list_index[li]
            if pk.is_strong:
                kept.append(li)
                continue
            is_artifact = False
            for src_freq, src_int, src_snr in sources:
                df_mhz = abs(pk.frequency - src_freq)
                if df_mhz <= 0.0:
                    continue
                env = src_int * _leakage_envelope_fraction(
                    df_mhz * 1e6, acquisition_us, tau_us
                )
                # Below the strong line's analytic skirt -> a leakage artifact.
                if pk.intensity < env:
                    is_artifact = True
                    break
            if is_artifact:
                pruned.append(li)
            else:
                kept.append(li)
        if pruned:
            w.free_peak_indices = kept
            w.diagnostics["pruned_leakage_artifacts"] = pruned
            total_pruned += len(pruned)
    if total_pruned:
        diagnostics["n_pruned_leakage_artifacts"] = total_pruned

    # --- Step 6: difficulty classification + width-cap split proposal -------
    for w in windows:
        has_strong = len(strong_by_window[w.window_id]) > 0
        has_fixed = len(w.fixed_contributors) > 0
        too_wide = w.width_mhz > max_window_width_mhz
        # Edge-coherence test: a window whose edge band still carries coherent
        # leakage (rolling S_coh above threshold within trim_m of either edge)
        # is materially influenced by a strong line even when none is in-band.
        lo, hi = w.diagnostics["grid_span"]
        edge_lo = rolling[max(lo - trim_m, 0) : lo + trim_m + 1]
        edge_hi = rolling[max(hi - trim_m, 0) : hi + trim_m + 1]
        edge_fail = False
        edge_stat = 0.0
        for band in (edge_lo, edge_hi):
            finite_band = band[np.isfinite(band)]
            if finite_band.size:
                edge_stat = max(edge_stat, float(np.max(finite_band)))
        edge_fail = edge_stat > edge_threshold
        w.diagnostics["edge_coherence_statistic"] = edge_stat
        w.diagnostics["edge_coherence_fail"] = bool(edge_fail)
        w.difficulty = (
            WindowDifficulty.HARD
            if (has_strong or has_fixed or too_wide or edge_fail)
            else WindowDifficulty.EASY
        )
        w.diagnostics["width_cap_hit"] = bool(too_wide)
        if too_wide:
            interior = rolling[lo + edge_m : hi - edge_m + 1]
            finite = interior[np.isfinite(interior)]
            if finite.size and float(np.min(finite)) < edge_threshold:
                rel = int(np.argmin(np.where(np.isfinite(interior), interior, np.inf)))
                w.split_proposal = float(ofreqs[lo + edge_m + rel])
            else:
                w.needs_joint_treatment = True

    # --- Step 7: topological order + parallel batches -----------------------
    window_ids = [w.window_id for w in windows]
    topo, batch, kept_edges = _topological_batches(window_ids, edges)
    for w in windows:
        w.batch = batch[w.window_id]
    if len(kept_edges) != len(edges):
        dropped_edges = [e for e in edges if e not in kept_edges]
        diagnostics["dropped_cyclic_dependencies"] = [list(e) for e in dropped_edges]
        # The magnitude-based attachment rule can produce cycles (two or more
        # strong lines in separate windows each attaching the other as a
        # contributor). The cycle-breaker drops the dependency edge to keep the
        # DAG acyclic; the matching FixedContributor can no longer be evaluated
        # from its primary's converged fit (the primary is no longer guaranteed
        # to precede the dependent). Discarding it outright -- the previous
        # behaviour -- leaves a bright neighbour's leakage skirt subtracted from
        # nowhere, the in-window lines under-fit: the issue-#3 cross-fixture
        # failure mode. Instead, the few most *dominant* orphaned contributors
        # per window are converted to EDGE-FREE: kept on the window but flagged
        # so Stage 5 reads their frozen (amplitude, phase) self-contained from
        # the active FT rather than from the un-ordered primary fit. Conversion
        # is capped at the top ``max_edge_free_neighbors`` primary windows
        # (ranked by aggregated predicted skirt) -- a dense, ultra-high-SNR
        # forest drops dozens of edges, and a self-contained skirt for every one
        # re-creates the global-crude over-subtraction that regressed the bulk
        # fit; only genuinely dominant neighbours earn one. The rest are dropped
        # as before.
        drops_by_dep: Dict[int, set] = {}
        for w_id, p_id in dropped_edges:
            drops_by_dep.setdefault(w_id, set()).add(p_id)
        n_edge_free = 0
        for w in windows:
            doomed = drops_by_dep.get(w.window_id)
            if not doomed:
                continue
            wlo, whi = w.diagnostics["grid_span"]
            w_center_mhz = 0.5 * (float(ofreqs[wlo]) + float(ofreqs[whi]))
            # Aggregate the predicted leakage skirt each orphaned primary
            # contributes to this window, so the cap keeps the strongest
            # neighbours (the same analytic envelope the Tier-1 attachment uses).
            skirt_by_primary: Dict[int, float] = {}
            for fc in w.fixed_contributors:
                if fc.primary_window_id not in doomed:
                    continue
                src_pk = by_list_index.get(fc.peak_index)
                if src_pk is None:
                    continue
                df_mhz = abs(src_pk.frequency - w_center_mhz)
                if df_mhz <= 0.0:
                    continue
                pred = src_pk.intensity * _leakage_envelope_fraction(
                    df_mhz * 1e6, acquisition_us, tau_us
                )
                skirt_by_primary[fc.primary_window_id] = (
                    skirt_by_primary.get(fc.primary_window_id, 0.0) + pred
                )
            keep_primaries = set(
                sorted(
                    skirt_by_primary,
                    key=lambda p: skirt_by_primary[p],
                    reverse=True,
                )[:max_edge_free_neighbors]
            )
            kept_contribs: List[FixedContributor] = []
            for fc in w.fixed_contributors:
                if fc.primary_window_id not in doomed:
                    kept_contribs.append(fc)
                elif fc.primary_window_id in keep_primaries:
                    fc.edge_free = True
                    kept_contribs.append(fc)
                    n_edge_free += 1
                # else: drop the orphaned contributor entirely.
            w.fixed_contributors = kept_contribs
        if n_edge_free:
            diagnostics["n_edge_free_contributors"] = n_edge_free

    # Plan-level diagnostics: leakage-touched regions with no promoted peak --
    # an early-warning hint that Stage 3 may have missed a line.
    unexplained = []
    for lo, hi in touched:
        if not any(lo <= pk.grid_index <= hi for pk in promoted):
            unexplained.append((float(ofreqs[lo]), float(ofreqs[hi])))
    if unexplained:
        diagnostics["unexplained_coherent_regions_mhz"] = unexplained

    return WindowPlan(
        windows=windows,
        dependency_edges=sorted(set(kept_edges)),
        topological_order=topo,
        parameters=parameters,
        diagnostics=diagnostics,
        plan_revision=plan_revision,
    )


def _apply_merge(
    windows: List[FitWindow],
    req: MergeRequest,
    *,
    ofreqs: np.ndarray,
    step_mhz: float,
) -> List[FitWindow]:
    """Apply one :class:`MergeRequest` to a window list.

    The two named windows must be **adjacent in frequency** (no other
    window's ``freq_range`` lies between them); otherwise the disjoint-
    coverage invariant would be violated. The surviving merged window
    keeps the *lower* of the two window ids; the higher id disappears
    from the plan. The merged window's:

    * ``freq_range`` = union of the two,
    * ``free_peak_indices`` = ordered union of both lists,
    * ``diagnostics['grid_span']`` recomputed for the new freq_range,
    * other diagnostics carry over from the lower-id window with a
      ``"merged_from"`` note added.

    Fixed contributors and dependency edges are intentionally **not**
    recomputed here -- :func:`_finalize_plan` re-derives them from the
    new window list against the spectrum state, which handles the
    "contributor is now internal" case automatically.

    Returns a new list (does not mutate the input).
    """
    by_id = {w.window_id: w for w in windows}
    if req.window_a_id not in by_id:
        raise ValueError(f"merge request references unknown window {req.window_a_id}")
    if req.window_b_id not in by_id:
        raise ValueError(f"merge request references unknown window {req.window_b_id}")
    if req.window_a_id == req.window_b_id:
        raise ValueError("merge request must reference two distinct windows")

    a = by_id[req.window_a_id]
    b = by_id[req.window_b_id]
    # Normalize so `lo_w` is the lower-frequency one (and its id survives if
    # both ids are equally valid -- we still pick the lower id below).
    if a.freq_range[0] > b.freq_range[0]:
        a, b = b, a

    # Adjacency: nothing else may sit between a.freq_range[1] and b.freq_range[0].
    for other in windows:
        if other.window_id in (a.window_id, b.window_id):
            continue
        olo, ohi = other.freq_range
        if olo > a.freq_range[1] and ohi < b.freq_range[0]:
            raise ValueError(
                f"merge request {req.window_a_id}/{req.window_b_id}: window "
                f"{other.window_id} lies between them (not adjacent)"
            )

    survivor_id = min(a.window_id, b.window_id)
    new_lo = min(a.freq_range[0], b.freq_range[0])
    new_hi = max(a.freq_range[1], b.freq_range[1])

    # Ordered union of free peaks (preserve insertion order).
    seen: set = set()
    merged_free: List[int] = []
    for li in list(a.free_peak_indices) + list(b.free_peak_indices):
        if li not in seen:
            seen.add(li)
            merged_free.append(li)

    # New grid span on the ordered grid. Use searchsorted on the (ascending)
    # ofreqs to keep the half-open semantics build_window_plan uses.
    new_lo_idx = int(np.searchsorted(ofreqs, new_lo, side="left"))
    new_hi_idx = int(np.searchsorted(ofreqs, new_hi, side="right")) - 1
    new_hi_idx = max(new_hi_idx, new_lo_idx)

    # Carry diagnostics forward from the lower-id source for stability;
    # _finalize_plan overwrites the per-window fields it manages.
    base_diag = dict(by_id[survivor_id].diagnostics)
    base_diag["grid_span"] = [new_lo_idx, new_hi_idx]
    merged_from = sorted({a.window_id, b.window_id})
    prior = base_diag.get("merged_from")
    if prior:
        merged_from = sorted(set(prior) | set(merged_from))
    base_diag["merged_from"] = merged_from
    if req.reason:
        base_diag["merge_reason"] = req.reason

    merged_window = FitWindow(
        window_id=survivor_id,
        freq_range=(new_lo, new_hi),
        free_peak_indices=merged_free,
        # fixed_contributors / difficulty / batch are rebuilt by _finalize_plan.
        diagnostics=base_diag,
    )

    out: List[FitWindow] = []
    for w in windows:
        if w.window_id == a.window_id or w.window_id == b.window_id:
            if w.window_id == survivor_id:
                out.append(merged_window)
            # else: drop the absorbed window
        else:
            out.append(w)
    return out


def replan(
    plan: WindowPlan,
    requests: List[MergeRequest],
    peaks: List[Peak],
    freqs: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    *,
    acquisition_us: float,
    tau_us: Optional[float] = None,
    edge_m: int = DEFAULT_EDGE_M,
    trim_m: int = DEFAULT_TRIM_M,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    max_window_width_mhz: float = DEFAULT_MAX_WINDOW_WIDTH_MHZ,
    min_freeze_snr: float = DEFAULT_MIN_FREEZE_SNR,
    min_window_half_width_mhz: float = DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ,
    magnitude_attachment_threshold: float = DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    max_edge_free_neighbors: int = DEFAULT_MAX_EDGE_FREE_NEIGHBORS,
    probe_freq_mhz: float = 0.0,
    start_us: float = 0.0,
) -> WindowPlan:
    """Re-plan: apply structural change requests to an existing window plan.

    Stage 5 emits :class:`MergeRequest` objects when the residual edge-
    coherence check flags a boundary cut and no fixed contributor is
    available to thaw -- i.e. a real spectral feature crosses the window
    boundary. ``replan`` applies each request to the existing window list,
    then re-runs the bookkeeping tail of :func:`build_window_plan`
    (contributor / dependency / difficulty / batch recomputation, artifact
    pruning) against the new window list, and bumps
    :attr:`WindowPlan.plan_revision`.

    The spectrum-dependent state (the rolling complex-edge coherence
    statistic, the leakage-touched intervals, the promoted-peak grid
    indices) is recomputed from the same spectrum the original plan was
    built on -- callers must pass the same ``peaks``/``freqs``/
    ``complex_spectrum``/``rms_noise`` plus matching stage-4 parameters
    (the helper accepts everything :func:`build_window_plan` takes for
    that reason).

    Parameters
    ----------
    plan : WindowPlan
        The plan to revise.
    requests : list of MergeRequest
        Structural change requests, applied in order. An empty list
        produces a copy of ``plan`` with the revision counter bumped.
    peaks, freqs, complex_spectrum, rms_noise : ...
        Same inputs the plan was built from.
    acquisition_us, tau_us, edge_m, trim_m, edge_threshold,
    max_window_width_mhz, min_freeze_snr, min_window_half_width_mhz,
    probe_freq_mhz, start_us : ...
        Stage 4 parameters; see :func:`build_window_plan`.

    Returns
    -------
    WindowPlan
        A revised plan with ``plan_revision = plan.plan_revision + 1``,
        the merged windows, and freshly recomputed dependency edges,
        difficulty, batches, and topological order. The disjoint-coverage
        invariant is preserved.

    Raises
    ------
    ValueError
        If a merge request names an unknown window, names the same window
        twice, or names two windows that are not adjacent in frequency.
    """
    if not (len(freqs) == len(complex_spectrum) == len(rms_noise)):
        raise ValueError("freqs, complex_spectrum and rms_noise must be equal length")
    if acquisition_us <= 0:
        raise ValueError("acquisition_us must be positive")

    parameters: Dict[str, Any] = {
        "edge_m": int(edge_m),
        "trim_m": int(trim_m),
        "edge_threshold": float(edge_threshold),
        "max_window_width_mhz": float(max_window_width_mhz),
        "min_freeze_snr": float(min_freeze_snr),
        "min_window_half_width_mhz": float(min_window_half_width_mhz),
        "magnitude_attachment_threshold": float(magnitude_attachment_threshold),
        "max_edge_free_neighbors": int(max_edge_free_neighbors),
        "acquisition_us": float(acquisition_us),
        "tau_us": tau_us,
        "start_us": float(start_us),
        "probe_freq_mhz": float(probe_freq_mhz),
    }

    ofreqs, ospec, orms, _order = _ordered_grid(
        np.asarray(freqs, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        np.asarray(rms_noise, dtype=float),
    )
    ospec = deramp_to_active_start(ofreqs, ospec, probe_freq_mhz, start_us)
    n = ofreqs.size
    diagnostics: Dict[str, Any] = {}

    promoted: List[_PPeak] = []
    for li, p in enumerate(peaks):
        if not p.properties.get("promoted"):
            continue
        gi = _nearest_grid_index(ofreqs, p.frequency) if n else 0
        promoted.append(
            _PPeak(
                list_index=li,
                grid_index=gi,
                frequency=float(p.frequency),
                snr=float(p.snr) if p.snr is not None else 0.0,
                intensity=float(p.intensity),
                is_strong=(p.classification == PeakClassification.STRONG),
            )
        )

    step_mhz = float(np.mean(np.diff(ofreqs))) if n > 1 else 1.0
    step_mhz = abs(step_mhz) or 1.0

    rolling = rolling_coherence(ospec, orms, band_m=edge_m)
    touched = above_threshold_intervals(rolling, edge_threshold)

    # Deep-copy the windows so mutations during _finalize_plan don't leak
    # back into the caller's plan object.
    windows = [
        FitWindow(
            window_id=w.window_id,
            freq_range=w.freq_range,
            free_peak_indices=list(w.free_peak_indices),
            fixed_contributors=[],  # rebuilt by _finalize_plan
            difficulty=w.difficulty,
            batch=w.batch,
            split_proposal=w.split_proposal,
            needs_joint_treatment=w.needs_joint_treatment,
            diagnostics=dict(w.diagnostics),
        )
        for w in plan.windows
    ]

    for req in requests:
        windows = _apply_merge(windows, req, ofreqs=ofreqs, step_mhz=step_mhz)

    return _finalize_plan(
        windows=windows,
        ofreqs=ofreqs,
        orms=orms,
        rolling=rolling,
        touched=touched,
        promoted=promoted,
        parameters=parameters,
        diagnostics=diagnostics,
        edge_m=edge_m,
        trim_m=trim_m,
        edge_threshold=edge_threshold,
        max_window_width_mhz=max_window_width_mhz,
        min_freeze_snr=min_freeze_snr,
        magnitude_attachment_threshold=magnitude_attachment_threshold,
        max_edge_free_neighbors=max_edge_free_neighbors,
        acquisition_us=acquisition_us,
        tau_us=tau_us,
        plan_revision=plan.plan_revision + 1,
    )
