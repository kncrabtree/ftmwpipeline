"""
Stage 4 window-planning algorithm.

This module turns the promoted Stage 3 peak list into a :class:`WindowPlan` --
an ordered set of disjoint analysis windows, each carrying the peaks to fit
freely, the strong out-of-band lines whose leakage must be carried frozen, a
fit dependency DAG. It is *purely structural*: it makes
no fits and changes no spectrum.

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
6. Each window records a per-edge coherence diagnostic (does its edge band
   still carry coherent leakage) for inspection -- nothing downstream branches
   on it.
7. Candidate dependency edges are oriented strong->weak (acyclic by the
   window-strength total order) and a downward skirt is kept edge-bearing only
   when material -- its level clears ``skirt_level_keep`` or its
   order-p-irreducible curvature clears ``curvature_keep_sigma``; sub-threshold
   skirts fall to the dependent's baseline. The surviving DAG is then
   topologically ordered into parallel batches.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.data_structures import (
    FitWindow,
    FixedContributor,
    MergeRequest,
    Peak,
    PeakClassification,
    WindowPlan,
)
from ..fitting.peak_model import baseline_basis, effective_tau, h_T
from .edge_coherence import (
    DEFAULT_EDGE_M,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_TRIM_M,
    above_threshold_intervals,
    active_edge_coherence,
    max_cumsum_statistic,
)

# Stage 4 parameter defaults. All configurable on the pipeline file.
DEFAULT_MAX_WINDOW_WIDTH_MHZ = 40.0
"""Width cap default. On 2638 a strong line's above-threshold skirt extends to
~40 MHz, so a single window wider than this is already dense/coupled.

ASSESSED AND KEPT ABSOLUTE, 2026-08-19 (``dev-docs/SCIENCE_STRATEGY.md``
Requirement 8, task E4). Two reasons, recorded so a future audit does not
re-derive them. First, the bin-relative form of this cap **already exists and
already wins**: :data:`DEFAULT_MAX_WINDOW_WIDTH_POINTS` is positive by default
and *replaces* this value, for exactly the Requirement 8 reason its own
docstring gives. This is the fallback the caller selects by setting the points
cap to ``0``, so converting it would leave the knob with two bin-count
spellings and no absolute one. Second, on its own terms it is a genuine
spectral width -- an observed skirt extent in MHz on a real spectrum -- not a
bin count that was written down in MHz."""

DEFAULT_MAX_WINDOW_WIDTH_POINTS = 96
"""Width cap in active-FT grid points; ``0`` disables it so the cap is
``max_window_width_mhz``. When positive it *replaces* the MHz cap as
the bound on the strong-cluster merge and the cap split (the effective cap in
MHz is ``points * grid step``). A points cap is the statistically portable
form: the Stage 5 gates reason over bins (n_eff, per-bin sigma), and the
active-FT bin width varies with acquisition length across instruments, so a
fixed MHz cap yields different statistical window sizes per fixture while a
points cap holds them constant. 96 points (~8 MHz on the reference 2638
grid) is the small-window operating point the window-invariant accept gates
are calibrated against: small enough that dense ultra-high-SNR fixtures fit
in minutes (the conservative loop's NLS cost grows ~K^2 with window
population), large enough that every window keeps tens of informative bins
for the gate."""

DEFAULT_MIN_FREEZE_SNR = 50.0
"""Freeze-eligibility SNR cutoff (O4-2): a fixed contributor below this is
flagged as a thaw-and-re-fit candidate rather than safely frozen."""

DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ = 2.0
"""Minimum half-width of a window built around an isolated weak line. The MHz
form of the window margin; the points form (:data:`DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS`)
supersedes it whenever that is positive (mirroring the width-cap MHz/points pair).

ASSESSED AND KEPT ABSOLUTE, 2026-08-19 (``dev-docs/SCIENCE_STRATEGY.md``
Requirement 8, task E4), for the first of the two reasons on
:data:`DEFAULT_MAX_WINDOW_WIDTH_MHZ`: the margin's bin-relative definition
already exists and is already the default
(:data:`DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS`, 32 active-FT points), and this
is the fallback a caller selects by zeroing it. The requirement it serves --
a window must hold enough bins for a 4-parameter fit -- is a bin count, and it
is stated as one there. Note the two are NOT the same number expressed twice:
32 points is ~2.53 MHz on the reference grid, against 2.0 MHz here. Do not
"reconcile" them; the MHz form is legacy and inert by default."""

DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS = 32
"""The window margin in active-FT grid points: the empty noise budget kept on
each side of a window's outermost promoted peak. It is the half-width every peak
proposes for its proto-window *and* the budget the post-construction trim leaves
around the content -- one number, so a window's extent tracks its peaks instead
of carrying a fat empty pedestal. ``0`` defers to the MHz form
``min_window_half_width_mhz`` (the portable points form is preferred, mirroring
``max_window_width_points`` over ``max_window_width_mhz``).

Decoupled from ``edge_m`` (the rolling-coherence band): the legacy proto
half-width was ``max(min_window_half_width_mhz / step, edge_m)``, so ``edge_m=64``
always won and the MHz knob was inert -- and worse, the resulting 2*64 = 128-point
proto window was *wider than the 96-point content cap*, so the cap split was
perpetually re-cutting content that already fit (review findings F2/F3). The
coherent operating range is ``trim_m <= margin <= max_window_width_points / 2``:
at least ``trim_m`` (default 32) so the edge-coherence statistic samples the noise
margin rather than a peak, and at most half the content cap (default 96/2 = 48)
so a lone line's ``2*margin`` window never exceeds the cap. The default 32 is the
tight end of that range (== ``trim_m``): minimal noise dilution and NLS cost,
still ample to anchor the order-<=4 leakage-wing baseline."""

DEFAULT_MAX_PEAKS_PER_WINDOW = 0
"""Per-window promoted-peak cap; ``0`` (the default) means *no* peak cap -- a
window is bounded only by ``max_window_width_mhz``. A fragmenting peak cap split a
dense cluster into windows too narrow for the Stage 5 AICc-with-``n_eff`` gate to
behave: the perplexity ``n_eff`` collapsed on a few-point fragment, so the gate
both under-fit (parking real lines) and over-fit (packing weak near-resolution
peaks) on neighboring slices of one physical cluster. Bounding a window by width
alone gives the gate enough informative bins to self-regulate K, fixing both at
the root (cross-fixture: every issue-#3 fixture's SNR-aware pass improved or held).
The strong-cluster merge and the cap split are still bounded by
``max_window_width_mhz`` (default 40), which alone keeps a dense ultra-high-SNR
spectrum from collapsing into one GHz-scale mega-window -- the runaway the peak cap
was wrongly credited with preventing was the unbounded strong-cluster force-merge,
governed by the width cap. A positive value restores an explicit cap (power users /
diagnostics); it tracks the Stage 5 ``conservative.max_peaks`` and both should be
set together."""

DEFAULT_STAGE6_MIN_WINDOW_HALF_WIDTH_POINTS = 8
"""Floor on the noise margin each side of a **Stage-6-created** window's anchor.

Stage 6 can create a window for a line the detector missed (see
:func:`plan_stage6_window`). Such a window is built in whatever gap the base
plan left, so unlike a Stage 4 window it cannot always get the full
``min_window_half_width_points`` margin. This is the point below which shrinking
it further stops being worth doing: 8 points a side is 17 bins, i.e. 34 real
residual elements against the ~4 free parameters of a single line plus tau --
enough for the per-window sigma and the residual edge statistic to mean
something. Below it, absorbing the anchor into the adjacent window is the better
answer: that window brings an already-fit statistical context instead of a
starved new one."""

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

    ``|S_env(Δf)| / |S(0)| = (1 + e^{-T/τ}) / (2π·|Δf|·τ_eff)`` -- the
    finite-T leakage envelope of a damped cosine observed over acquisition
    ``T``, evaluated here to decide whether a weaker nearby detection sits
    below a strong line's skirt.
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
    """Split a merged ``(lo, hi)`` grid span until each piece's *peak content*
    spans at most ``cap_idx`` grid steps and (when ``max_peaks > 0``) holds at
    most ``max_peaks`` promoted peaks.

    Splits at the largest internal peak gap, placing the boundary at the gap
    midpoint so the resulting windows stay disjoint and each edge peak keeps half
    the gap as margin. ``member_gidx`` is the sorted promoted-peak grid indices
    inside ``[lo, hi]``. ``max_peaks <= 0`` disables the peak-count split so a span
    is bounded by ``cap_idx`` (the width cap) alone; the cross-window coupling is
    carried by the fixed contributors, the same mechanism that handles a strong
    line's distant skirt.

    The width bound is on the **peak content** (the span between the first and
    last promoted peak), not the padded ``(lo, hi)`` span: a cluster whose lines
    fit inside the cap must stay in one window even when its empty proto-margins
    (the ``edge_m``-sized half-extent every peak proposes) push the padded span
    over the cap. Bounding on the padded width instead bisects a content-fitting
    cluster at whatever sub-minimum interior gap happens to be largest -- the
    2638 33723.5-33724.6 cluster (0.94 MHz of content) split at a 0.39 MHz notch
    because its ~7 MHz of margin tipped the enclosing span past the cap (review
    finding F2).
    """
    members = [g for g in member_gidx if lo <= g <= hi]
    peaks_ok = max_peaks <= 0 or len(members) <= max_peaks
    content_idx = (members[-1] - members[0]) if len(members) >= 2 else 0
    if (content_idx <= cap_idx and peaks_ok) or len(members) <= 1:
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


# Tiered cycle-break (replaces the Kahn drop-leftover + edge-free demotion).
# The goal is a deterministic, tiered dependency DAG along which a Stage-6 edit
# cascades on a predictable path -- NOT to maximally decouple the windows. So:
# orient every candidate edge strong->weak (window strength = strongest
# promoted-peak intensity, ties broken by id -> a total order, hence acyclic),
# which levels the windows into tiers; keep a stronger (higher-tier) window's
# skirt as an edge-bearing contributor to weaker (lower-tier) windows whenever it
# is *material* -- i.e. it carries either significant skirt LEVEL (it consumes
# real baseline budget on the dependent, esp. a floor-dominated window) OR
# order-p-irreducible CURVATURE. These are the cascade paths and are kept
# generously; same-tier peer couplings the orientation cannot order are left to
# the joint thawed fit (Step C). A pure-curvature gate (level decays ~1/Δf, so
# curvature ~1/Δf**3 is steeply local) removes too many real relationships: a far
# but bright source's skirt is smooth (low curvature) yet large (high level), and
# dropping it onto the order-p baseline starves a floor-dominated dependent (the
# 655 w1010 baseline-budget failure). Level is the materiality measure; curvature
# is the secondary catch for steep-local skirts of modest level.
DEFAULT_SKIRT_LEVEL_KEEP = 150.0
"""S_level threshold (the skirt's total significance ``||skirt/sigma_c||`` over
the dependent window) above which a downward edge is kept edge-bearing. Materiality
measure: a skirt this significant consumes baseline budget the dependent may need
for its own line, so it is carried as a physical contributor rather than left to
the polynomial. The hard default mirrored by
``WindowPlanningSettings.contributor.skirt_level_keep``.

Calibrated on the 7-fixture A/B cascade refit: 150 vs 50 halves the contributor
load on the dense 655 (1982 -> 971),
shallows the cascade (6 -> 4 tiers), cuts fit time (656s -> 493s), and *reduces*
over-subtraction (655 peaks 1794 -> 1846) by pruning the long tail of weak-source
edges, while the budget-critical giant skirts (e.g. 655 w1010) survive at any bar."""

DEFAULT_CURVATURE_KEEP_SIGMA = 5.0
"""S_resid threshold (in the dependent's per-component sigma_c): a *secondary*
keep criterion for a skirt whose curvature an order-p baseline cannot absorb even
when its level is below ``DEFAULT_SKIRT_LEVEL_KEEP``. The hard default mirrored by
``WindowPlanningSettings.contributor.curvature_keep_sigma``."""

CURVATURE_BASELINE_ORDER = 4
"""Baseline order the curvature discriminator projects against -- the part of the
skirt this order of polynomial cannot absorb is what an explicit contributor must
carry. Matches the ``baseline.order`` default (presets/defaults.yaml)."""


def _skirt_significance(
    grid_freqs: np.ndarray,
    sigma_c: np.ndarray,
    src_freqs: List[float],
    src_intensities: List[float],
    acquisition_us: float,
    tau_us: Optional[float],
    order: int,
) -> Tuple[float, float]:
    """``(S_level, S_resid)`` of a source window's strong peaks projected into a
    dependent window grid, in units of the dependent's per-component noise:

    * ``S_level  = ||skirt / sigma_c||``                -- raw skirt significance
      (materiality: how much baseline budget the skirt consumes).
    * ``S_resid  = ||(skirt - B_p[skirt]) / sigma_c||`` -- the part an order-``p``
      baseline cannot absorb (steep-local curvature).

    Phase is unavailable at plan time, so each source line is synthesized at
    phase 0 (the coherent worst-case skirt). Both norms are invariant to the
    sideband sign (it flips ``skirt -> conj(skirt)``, leaving the magnitudes
    unchanged). The shape is the Lorentzian ``h_T``; this is a coarse keep/drop,
    not a precise fit, so a per-window shape is not threaded here.
    """
    if grid_freqs.size == 0 or not src_freqs:
        return 0.0, 0.0
    tau_model = float(tau_us) if tau_us else acquisition_us / 3.0
    tau_eff = effective_tau(tau_model, acquisition_us)
    center = 0.5 * (float(grid_freqs[0]) + float(grid_freqs[-1]))
    u = grid_freqs - center
    skirt = np.zeros(u.shape, dtype=np.complex128)
    for f, inten in zip(src_freqs, src_intensities):
        amp = 2.0 * float(inten) / tau_eff  # intensity = 0.5*A*tau_eff
        skirt = skirt + 0.5 * amp * h_T(
            u - (float(f) - center), tau_model, acquisition_us
        )
    sig = np.where(sigma_c > 0, sigma_c, np.nan)
    s_level = float(np.sqrt(np.nansum((np.abs(skirt) / sig) ** 2)))
    u_s = float(np.max(np.abs(u))) or 1.0
    basis = baseline_basis(u, order, u_s)
    coef, *_ = np.linalg.lstsq(basis, skirt, rcond=None)
    resid = skirt - basis @ coef
    s_resid = float(np.sqrt(np.nansum((np.abs(resid) / sig) ** 2)))
    return s_level, s_resid


def _orient_and_gate_contributors(
    windows: List[FitWindow],
    strong_by_window: Dict[int, List["_PPeak"]],
    ofreqs: np.ndarray,
    orms: np.ndarray,
    acquisition_us: float,
    tau_us: Optional[float],
    diagnostics: Dict[str, Any],
    level_keep: float,
    resid_keep: float,
    order: int,
) -> List[Tuple[int, int]]:
    """Orient candidate edges strong->weak and keep the material downward ones.

    Mutates each window's ``fixed_contributors`` in place to the surviving
    edge-bearing set (``edge_free=False``). A dependency edge ``(w, primary)``
    survives when ``primary`` is the stronger window (orientation -> tiered DAG)
    *and* its projected skirt is material into ``w``: ``S_level >= level_keep``
    (it consumes real baseline budget) OR ``S_resid >= resid_keep`` (steep-local
    curvature an order-p baseline cannot absorb). Reverse arcs are dropped; a
    sub-threshold downward skirt is left to the dependent's baseline. Returns the
    surviving edge list (acyclic by the strength total order).
    """
    strength: Dict[int, float] = {
        w.window_id: max(
            (pk.intensity for pk in strong_by_window.get(w.window_id, [])),
            default=0.0,
        )
        for w in windows
    }
    kept_edges: List[Tuple[int, int]] = []
    n_bearing = n_drop_base = n_drop_reverse = 0
    for w in windows:
        if not w.fixed_contributors:
            continue
        wid = w.window_id
        wlo, whi = w.diagnostics["grid_span"]
        grid_freqs = ofreqs[wlo : whi + 1]
        sigma_c = orms[wlo : whi + 1] / math.sqrt(2.0)
        by_primary: Dict[int, List[FixedContributor]] = {}
        for fc in w.fixed_contributors:
            by_primary.setdefault(fc.primary_window_id, []).append(fc)
        survivors: List[FixedContributor] = []
        for primary, group in by_primary.items():
            # Orientation: the weaker window depends on the stronger. A reverse
            # arc (this window is the stronger) is dropped -- the forward arc, if
            # any, is a contributor on the *primary*'s list, gated when we reach
            # that window.
            if (strength.get(primary, 0.0), primary) <= (strength[wid], wid):
                n_drop_reverse += len(group)
                continue
            src = strong_by_window.get(primary, [])
            s_level, s_resid = _skirt_significance(
                grid_freqs,
                sigma_c,
                [pk.frequency for pk in src],
                [pk.intensity for pk in src],
                acquisition_us,
                tau_us,
                order,
            )
            if s_level >= level_keep or s_resid >= resid_keep:
                for fc in group:
                    fc.edge_free = False
                survivors.extend(group)
                kept_edges.append((wid, primary))
                n_bearing += len(group)
            else:
                n_drop_base += len(group)
        w.fixed_contributors = survivors
    if n_bearing:
        diagnostics["n_edge_bearing_contributors"] = n_bearing
    if n_drop_base:
        diagnostics["n_dropped_baseline_contributors"] = n_drop_base
    if n_drop_reverse:
        diagnostics["n_dropped_reverse_contributors"] = n_drop_reverse
    return kept_edges


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
    min_window_half_width_points: int = DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS,
    magnitude_attachment_threshold: float = DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    skirt_level_keep: float = DEFAULT_SKIRT_LEVEL_KEEP,
    curvature_keep_sigma: float = DEFAULT_CURVATURE_KEEP_SIGMA,
    max_peaks_per_window: int = DEFAULT_MAX_PEAKS_PER_WINDOW,
    max_window_width_points: int = DEFAULT_MAX_WINDOW_WIDTH_POINTS,
) -> WindowPlan:
    """Build the Stage 4 fit plan from the promoted Stage 3 peaks.

    Parameters
    ----------
    peaks : list of Peak
        The full persisted Stage 3 peak list. Only peaks with
        ``properties['promoted']`` truthy are planned; ``free_peak_indices`` and
        fixed-contributor references index back into *this* list.
    freqs, complex_spectrum, rms_noise : np.ndarray
        The active-FT surface: frequency axis (MHz), complex FT, and the
        persisted Stage 2 per-point RMS noise. Equal length, 1D.
    acquisition_us : float
        Active FID acquisition ``T`` (µs) for the analytic leakage reach.
    tau_us : float, optional
        Assumed shared decay constant; ``None`` = undamped/boxcar limit.
    edge_m, trim_m : int
        Coherence-statistic band widths (rolling scan / trim refinement).
    edge_threshold : float
        ``S_coh`` threshold ``T_edge``.
    max_window_width_mhz : float
        Width cap; merged spans wider than this are split at their sparsest
        internal peak gaps. Superseded by ``max_window_width_points`` when that
        is positive.
    max_window_width_points : int
        Width cap in grid points; ``0`` (the default) defers to
        ``max_window_width_mhz``. The portable form of the cap -- see
        :data:`DEFAULT_MAX_WINDOW_WIDTH_POINTS`.
    min_freeze_snr : float
        Freeze-eligibility SNR cutoff for fixed contributors (O4-2).
    min_window_half_width_mhz : float
        The MHz form of the window margin (the noise budget each side of the
        outermost peak). Used only when ``min_window_half_width_points <= 0``.
    min_window_half_width_points : int
        The window margin in grid points -- the proto half-width *and* the
        post-construction trim budget. Supersedes ``min_window_half_width_mhz``
        when positive (the default). See
        :data:`DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS`.
    max_peaks_per_window : int
        Per-window promoted-peak cap; ``0`` (the default) disables it so a window
        is bounded only by ``max_window_width_mhz``. The strong-cluster merge is
        bounded at ``max_window_width_mhz`` and the merged spans are split at their
        sparsest internal gaps until each window is at most ``max_window_width_mhz``
        wide and (when positive) holds at most this many promoted peaks. A positive
        value tracks the Stage 5 ``conservative.max_peaks``; see
        :data:`DEFAULT_MAX_PEAKS_PER_WINDOW`.
    magnitude_attachment_threshold : float
        Tier-1 contributor-attachment threshold in units of σ_c. A strong
        promoted peak is attached to a window's ``fixed_contributors`` when
        its predicted mean |skirt| on that window's grid is at least
        ``threshold * sigma_c(w)``. Default
        :data:`DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD`.
    skirt_level_keep : float
        Keep an oriented downward skirt edge-bearing when its total
        significance ``S_level`` clears this (see
        :data:`DEFAULT_SKIRT_LEVEL_KEEP`); a sub-threshold skirt falls to the
        dependent's baseline polynomial.
    curvature_keep_sigma : float
        Secondary keep criterion in σ_c units: keep a downward skirt whose
        order-``p``-irreducible curvature ``S_resid`` clears this even when its
        level is below ``skirt_level_keep`` (see
        :data:`DEFAULT_CURVATURE_KEEP_SIGMA`).

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
        "min_window_half_width_points": int(min_window_half_width_points),
        "magnitude_attachment_threshold": float(magnitude_attachment_threshold),
        "skirt_level_keep": float(skirt_level_keep),
        "curvature_keep_sigma": float(curvature_keep_sigma),
        "max_peaks_per_window": int(max_peaks_per_window),
        "max_window_width_points": int(max_window_width_points),
        "acquisition_us": float(acquisition_us),
        "tau_us": tau_us,
    }

    ofreqs, ospec, orms, _order = _ordered_grid(
        np.asarray(freqs, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        np.asarray(rms_noise, dtype=float),
    )
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
    # The active FT is in the [0, T] frame (sliced active region), so the
    # coherent sum is scored directly -- no de-ramp (active_edge_coherence).
    rolling = active_edge_coherence(ospec, orms, band_m=edge_m)
    touched = above_threshold_intervals(rolling, edge_threshold)

    # --- Step 2: per-peak proposed windows (tight, uniform) -----------------
    # A window's extent is a peak's core plus the window margin -- it is NOT the
    # leakage-touched run. A strong line's run is ~80-100 MHz wide; its distant
    # leakage is carried by other windows as a fixed contributor, not by widening
    # this window (see leakage-detection-rework.md). The margin is the points
    # form when set (the default), else the MHz form -- decoupled from edge_m
    # (the coherence band), which previously shadowed it (review findings F2/F3).
    if min_window_half_width_points > 0:
        half_idx = int(min_window_half_width_points)
    else:
        half_idx = max(int(round(min_window_half_width_mhz / step_mhz)), 1)
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
    # A positive points cap is the portable form and supersedes the MHz cap
    # (see :data:`DEFAULT_MAX_WINDOW_WIDTH_POINTS`).
    cap_idx = (
        float(max_window_width_points)
        if max_window_width_points > 0
        else max_window_width_mhz / step_mhz
    )
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

    # --- Cap split: enforce <= the width cap (and, if set, the peak cap) ----
    # The bounded strong-merge above stops a forced GHz span, but in a dense
    # forest the overlapping per-peak proto-spans (and the capped strong spans)
    # still chain into windows wider than the width cap. Split each merged span at
    # its sparsest internal peak gaps until every window spans at most
    # ``max_window_width_mhz`` (and, when ``max_peaks_per_window > 0``, holds at
    # most that many promoted peaks); the cross-window coupling is carried by the
    # fixed contributors.
    all_gidx = sorted(pk.grid_index for pk in promoted)
    capped: List[Tuple[int, int]] = []
    for lo, hi in merged:
        span_gidx = [g for g in all_gidx if lo <= g <= hi]
        capped.extend(
            _split_span_to_caps(lo, hi, span_gidx, cap_idx, max_peaks_per_window)
        )
    merged = capped

    # --- Trim empty margins to the window-margin budget ---------------------
    # Pull each window's edges in to at most ``half_idx`` grid points beyond its
    # outermost promoted peak. The merge and cap split leave wide empty margins
    # (a lone line proposes a full +/- half_idx span that may overlap nothing; a
    # cap split places its boundary at a gap midpoint that can sit far from the
    # nearest peak), which leaves the feature off-center and dilutes the
    # per-window statistics with noise-only bins. Trimming shrinks each span to
    # its peak content plus the margin so the outermost peaks sit ``half_idx``
    # points from the edge -- enough noise to anchor the leakage-wing baseline,
    # no more. Shrinking preserves the disjoint-coverage invariant: a window
    # covers each spectrum point at most once, and a noise-only gap that opens
    # between two trimmed windows needs no coverage (distant leakage is carried
    # by fixed contributors, not by window width).
    trimmed: List[Tuple[int, int]] = []
    for lo, hi in merged:
        span_gidx = [g for g in all_gidx if lo <= g <= hi]
        if span_gidx:
            lo = max(lo, span_gidx[0] - half_idx)
            hi = min(hi, span_gidx[-1] + half_idx)
        trimmed.append((lo, hi))
    merged = trimmed

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
        trim_m=trim_m,
        edge_threshold=edge_threshold,
        min_freeze_snr=min_freeze_snr,
        magnitude_attachment_threshold=magnitude_attachment_threshold,
        skirt_level_keep=skirt_level_keep,
        curvature_keep_sigma=curvature_keep_sigma,
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
    trim_m: int,
    edge_threshold: float,
    min_freeze_snr: float,
    magnitude_attachment_threshold: float,
    skirt_level_keep: float,
    curvature_keep_sigma: float,
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
    * records the per-window edge-coherence diagnostic (step 6),
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
    # long-tail cumulative-skirt bias (O5-10).
    for w in windows:
        w.fixed_contributors = []

    # Which window owns each strong promoted peak (its primary window).
    primary_of_strong: Dict[int, int] = {}
    for w in windows:
        for s in strong_by_window[w.window_id]:
            primary_of_strong[s.list_index] = w.window_id

    for w in windows:
        wlo, whi = w.diagnostics["grid_span"]
        w_center_mhz = 0.5 * (float(ofreqs[wlo]) + float(ofreqs[whi]))
        # Per-window noise reference: mean Stage 2 rms over the window's grid
        # span converted to per-quadrature sigma. ``sigma_c = rms / sqrt(2)``
        # matches the diagnostic convention (the |residual| / Rayleigh test).
        w_rms_mean = float(np.mean(orms[wlo : whi + 1]))
        sigma_c_w = w_rms_mean / math.sqrt(2.0) if w_rms_mean > 0 else 0.0
        threshold = magnitude_attachment_threshold * sigma_c_w

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

    # --- Step 6: per-window edge-coherence diagnostic -----------------------
    # Record whether either edge band still carries coherent leakage (rolling
    # S_coh above threshold within trim_m of an edge) -- a hint that a strong
    # line materially influences the window even when none is in-band. Purely
    # diagnostic: nothing downstream branches on it.
    for w in windows:
        lo, hi = w.diagnostics["grid_span"]
        edge_lo = rolling[max(lo - trim_m, 0) : lo + trim_m + 1]
        edge_hi = rolling[max(hi - trim_m, 0) : hi + trim_m + 1]
        edge_stat = 0.0
        for band in (edge_lo, edge_hi):
            finite_band = band[np.isfinite(band)]
            if finite_band.size:
                edge_stat = max(edge_stat, float(np.max(finite_band)))
        w.diagnostics["edge_coherence_statistic"] = edge_stat
        w.diagnostics["edge_coherence_fail"] = bool(edge_stat > edge_threshold)

    # --- Step 7: orient + materiality-gate contributors, then topo-order ----
    # Orient every candidate edge strong->weak and keep it edge-bearing only
    # where the source's skirt is material into the dependent -- it carries
    # significant level (``skirt_level_keep``) or order-p-irreducible curvature
    # (``curvature_keep_sigma``); the rest fall to the (order-p) baseline. The
    # strength total order makes the surviving graph acyclic by construction, so
    # no edge is force-dropped.
    window_ids = [w.window_id for w in windows]
    kept_edges = _orient_and_gate_contributors(
        windows,
        strong_by_window,
        ofreqs,
        orms,
        acquisition_us,
        tau_us,
        diagnostics,
        skirt_level_keep,
        curvature_keep_sigma,
        CURVATURE_BASELINE_ORDER,
    )
    topo, batch, kept_edges = _topological_batches(window_ids, kept_edges)
    for w in windows:
        w.batch = batch[w.window_id]

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
        # fixed_contributors / batch are rebuilt by _finalize_plan.
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
    min_window_half_width_points: int = DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS,
    magnitude_attachment_threshold: float = DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    skirt_level_keep: float = DEFAULT_SKIRT_LEVEL_KEEP,
    curvature_keep_sigma: float = DEFAULT_CURVATURE_KEEP_SIGMA,
    max_window_width_points: int = DEFAULT_MAX_WINDOW_WIDTH_POINTS,
) -> WindowPlan:
    """Re-plan: apply structural change requests to an existing window plan.

    Stage 5 emits :class:`MergeRequest` objects when the residual edge-
    coherence check flags a boundary cut and no fixed contributor is
    available to thaw -- i.e. a real spectral feature crosses the window
    boundary. ``replan`` applies each request to the existing window list,
    then re-runs the bookkeeping tail of :func:`build_window_plan`
    (contributor / dependency / batch recomputation, artifact pruning) against
    the new window list, and bumps
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
    max_window_width_mhz, min_freeze_snr, min_window_half_width_mhz : ...
        Stage 4 parameters; see :func:`build_window_plan`.

    Returns
    -------
    WindowPlan
        A revised plan with ``plan_revision = plan.plan_revision + 1``,
        the merged windows, and freshly recomputed dependency edges,
        batches, and topological order. The disjoint-coverage invariant is
        preserved.

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
        "min_window_half_width_points": int(min_window_half_width_points),
        "magnitude_attachment_threshold": float(magnitude_attachment_threshold),
        "skirt_level_keep": float(skirt_level_keep),
        "curvature_keep_sigma": float(curvature_keep_sigma),
        "max_window_width_points": int(max_window_width_points),
        "acquisition_us": float(acquisition_us),
        "tau_us": tau_us,
    }

    ofreqs, ospec, orms, _order = _ordered_grid(
        np.asarray(freqs, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        np.asarray(rms_noise, dtype=float),
    )
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

    rolling = active_edge_coherence(ospec, orms, band_m=edge_m)
    touched = above_threshold_intervals(rolling, edge_threshold)

    # Deep-copy the windows so mutations during _finalize_plan don't leak
    # back into the caller's plan object.
    windows = [
        FitWindow(
            window_id=w.window_id,
            freq_range=w.freq_range,
            free_peak_indices=list(w.free_peak_indices),
            fixed_contributors=[],  # rebuilt by _finalize_plan
            batch=w.batch,
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
        trim_m=trim_m,
        edge_threshold=edge_threshold,
        min_freeze_snr=min_freeze_snr,
        magnitude_attachment_threshold=magnitude_attachment_threshold,
        skirt_level_keep=skirt_level_keep,
        curvature_keep_sigma=curvature_keep_sigma,
        acquisition_us=acquisition_us,
        tau_us=tau_us,
        plan_revision=plan.plan_revision + 1,
    )


# ---------------------------------------------------------------------------
# Stage 6: create a window for a line the detector missed.
#
# A curator who spots a real line in a region the base plan left uncovered has
# no window to edit -- Stage 4 built windows around *promoted* detections, and
# lowering the detection threshold to reach the line drops Stages 5 and 6 and
# destroys the whole curated edit set. The planner below builds one window for
# such an anchor, deliberately as a **purely additive** structural change:
#
# * it never renumbers an existing window (a consumer partitions peaks on
#   window_id to decide what an edit touched; renumbering would flag every peak
#   in the spectrum on every window addition),
# * its extent is a function of the anchor and the base plan alone -- never of
#   the current curated state -- so replaying an edit set in order reproduces
#   the same geometry, and
# * it carries only *inbound* dependency edges: the base plan's strong lines
#   leak into it, it leaks into nothing. That is what keeps it a leaf in the
#   fit DAG and lets it be added without re-fitting anything already fit.
#
# The one case where "additive" is not achievable is a gap too narrow to hold a
# fittable window at all. Rather than create a starved one, the adjacent window
# is widened to absorb the anchor -- reported honestly as ``mode="widened"`` so
# the caller (and the decision log) records that an existing window changed.
# ---------------------------------------------------------------------------


@dataclass
class Stage6WindowProposal:
    """One proposed Stage-6 structural change, ready to persist and fit.

    Attributes
    ----------
    window : FitWindow
        The window to install. For ``mode="created"`` this carries a **fresh**
        ``window_id`` (one past the plan's highest); for ``mode="widened"`` it
        is the existing window's id with a grown ``freq_range``.
    mode : str
        ``"created"`` when a new window was built in a gap, ``"widened"`` when
        the gap was too narrow and an existing window absorbed the anchor.
    depends_on : list of int
        ``window_id`` values this window reads frozen leakage from -- the
        inbound half of the dependency edges ``(window.window_id, primary)``.
        Empty for ``mode="widened"`` (its edges are unchanged).
    diagnostics : dict
        Why the proposal came out the way it did: the anchor, the gap it landed
        in, the realized half-widths, and the margin floor that was applied.
    """

    window: FitWindow
    mode: str
    depends_on: List[int]
    diagnostics: Dict[str, Any]


def _grid_span_of(
    ordered_freqs: np.ndarray, freq_range: Tuple[float, float]
) -> Tuple[int, int]:
    """Inclusive ascending-grid index span covered by a window's ``freq_range``.

    Derived from the frequencies rather than read from ``diagnostics`` so it is
    correct for a hand-edited plan and for a window this module itself created.
    """
    lo_mhz, hi_mhz = float(freq_range[0]), float(freq_range[1])
    if lo_mhz > hi_mhz:
        lo_mhz, hi_mhz = hi_mhz, lo_mhz
    lo = int(np.searchsorted(ordered_freqs, lo_mhz, side="left"))
    hi = int(np.searchsorted(ordered_freqs, hi_mhz, side="right")) - 1
    n = int(ordered_freqs.size)
    lo = min(max(lo, 0), n - 1)
    hi = min(max(hi, 0), n - 1)
    if hi < lo:
        hi = lo
    return lo, hi


def _grid_index_covering_window(
    spans: List[Tuple[int, int, int]], grid_index: int
) -> Optional[int]:
    """Return the id of the ``(lo, hi, wid)`` span in ``spans`` covering
    ``grid_index``, or ``None``.

    The single definition of "does a live window already cover this grid
    index" -- shared by :func:`plan_stage6_window` (which raises on a hit)
    and :func:`live_window_covering_anchor` (which answers the same question
    without raising, for a caller deciding whether a create is even needed).
    ``spans`` need not be sorted; the first covering span wins, and the caller
    invariant (plan windows are disjoint) means there is at most one.
    """
    for lo, hi, wid in spans:
        if lo <= grid_index <= hi:
            return wid
    return None


def live_window_covering_anchor(
    plan: WindowPlan,
    freqs: np.ndarray,
    anchor_mhz: float,
    *,
    live_window_ids: Optional[Iterable[int]] = None,
) -> Optional[int]:
    """Return the id of the live window that already covers ``anchor_mhz``.

    Uses the identical coverage predicate :func:`plan_stage6_window` uses to
    refuse a create -- the anchor is snapped to the nearest active-FT grid
    index (:func:`_nearest_grid_index`) and compared against each live
    window's grid span (:func:`_grid_span_of`), not a plain MHz-range test.
    This is *not* a search: it asks the planner's own containment question
    once, after the fact, for a caller (an implied-create resolver) deciding
    whether deriving a window is even necessary before calling
    :func:`plan_stage6_window` at all.

    Parameters
    ----------
    plan :
        The base :class:`WindowPlan` to check coverage against.
    freqs :
        The active-FT frequency axis (any order; sorted internally, exactly
        as :func:`plan_stage6_window` sorts its spectrum arrays).
    anchor_mhz :
        The candidate frequency, in MHz.
    live_window_ids :
        Restrict the check to these window ids (Stage 5's live set).
        ``None`` treats every plan window as live.

    Returns
    -------
    Optional[int]
        The covering window's id, or ``None`` if the anchor lies outside the
        analysis band or inside no live window.
    """
    ordered = np.sort(np.asarray(freqs, dtype=float))
    if ordered.size == 0:
        return None
    anchor = float(anchor_mhz)
    if anchor < float(ordered[0]) or anchor > float(ordered[-1]):
        return None
    gi = _nearest_grid_index(ordered, anchor)
    live: Optional[Set[int]] = (
        None if live_window_ids is None else {int(w) for w in live_window_ids}
    )
    candidates = [w for w in plan.windows if live is None or int(w.window_id) in live]
    spans = [
        (*_grid_span_of(ordered, w.freq_range), int(w.window_id)) for w in candidates
    ]
    return _grid_index_covering_window(spans, gi)


def plan_stage6_window(
    plan: WindowPlan,
    peaks: List[Peak],
    freqs: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    anchor_mhz: float,
    *,
    acquisition_us: float,
    tau_us: Optional[float] = None,
    min_window_half_width_mhz: float = DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ,
    min_window_half_width_points: int = DEFAULT_MIN_WINDOW_HALF_WIDTH_POINTS,
    min_freeze_snr: float = DEFAULT_MIN_FREEZE_SNR,
    magnitude_attachment_threshold: float = DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD,
    stage6_min_half_width_points: int = DEFAULT_STAGE6_MIN_WINDOW_HALF_WIDTH_POINTS,
    live_window_ids: Optional[Sequence[int]] = None,
) -> Stage6WindowProposal:
    """Propose a fit window covering ``anchor_mhz`` without disturbing the plan.

    Pure and deterministic: given the same base ``plan``, spectrum and anchor it
    always returns the same proposal, which is what lets a Stage 6 edit set be
    replayed. It does not mutate ``plan``.

    Geometry
    --------
    The anchor is placed on the active-FT grid and the **gap** it falls in --
    the run of grid points between the two nearest existing windows -- is the
    only room available, because Stage 4's hard invariant is that windows are
    disjoint. Inside that gap the window takes the plan's own margin
    (``min_window_half_width_points``, else the MHz form) on each side of the
    anchor. If the gap cannot hold that symmetrically the window is **shifted**,
    not shrunk, so a line near one end of a roomy gap still gets a full-size
    window (off-center). Only when the gap itself is too small does the window
    shrink, and only down to ``stage6_min_half_width_points`` a side.

    Below that floor the proposal switches to ``mode="widened"``: the nearer
    adjacent window grows to absorb the anchor plus the floor margin, bounded by
    the far neighbor so disjointness still holds. Its contributor set and
    dependency edges are carried over untouched -- the widening adds grid points,
    it does not re-derive the plan.

    Contributors
    ------------
    A created window is attached to the base plan's strong lines by the same
    Tier-1 magnitude rule Stage 4 uses (predicted mean |skirt| at least
    ``magnitude_attachment_threshold * sigma_c`` on this window's grid), reading
    the primaries straight off ``plan``. Every attachment is edge-bearing and
    inbound; the new window is a leaf, so no existing window acquires a
    dependency on it and none needs re-fitting. That is deliberate: a window
    created for a detection the automatic pass missed holds, by construction, a
    line too weak to clear the freeze bar, whose own leakage into its neighbors
    is negligible.

    Parameters
    ----------
    plan :
        The base :class:`WindowPlan` (not mutated).
    peaks :
        The full persisted Stage 3 peak list; ``free_peak_indices`` and
        contributor references index into it, exactly as in
        :func:`build_window_plan`.
    freqs, complex_spectrum, rms_noise :
        The active-FT surface, as for :func:`build_window_plan`.
    anchor_mhz :
        Molecular frequency (MHz) the window must cover.
    acquisition_us, tau_us :
        Active acquisition ``T`` (µs) and assumed decay constant, for the
        analytic leakage envelope.
    min_window_half_width_mhz, min_window_half_width_points :
        The plan's window margin (points form wins when positive), i.e. the
        extent a created window aims for on each side of the anchor.
    min_freeze_snr :
        Freeze-eligibility SNR cutoff stamped on each attached contributor.
    magnitude_attachment_threshold :
        Tier-1 contributor-attachment threshold in units of σ_c.
    stage6_min_half_width_points :
        The floor below which a created window is not worth having; see
        :data:`DEFAULT_STAGE6_MIN_WINDOW_HALF_WIDTH_POINTS`.
    live_window_ids :
        The plan windows that actually carry a Stage 5 fit. Stage 5 drops a
        window whose peaks all fail their gates, and a dropped window is not
        occupying its range in any meaningful sense: nothing is fit there, so a
        line there genuinely has no window and it is not a widening target
        either. Passing the ids keeps both decisions honest. ``None`` (the
        default) treats every plan window as live, which is the right reading
        for a caller reasoning about the plan alone.

    Returns
    -------
    Stage6WindowProposal

    Raises
    ------
    ValueError
        If the spectrum arrays differ in length, ``acquisition_us <= 0``, the
        anchor lies outside the spectrum, the anchor already falls inside an
        existing window (that frequency is an ordinary ``review edit --add``),
        or the plan has no windows to position against.
    """
    if not (len(freqs) == len(complex_spectrum) == len(rms_noise)):
        raise ValueError("freqs, complex_spectrum and rms_noise must be equal length")
    if acquisition_us <= 0:
        raise ValueError("acquisition_us must be positive")

    ofreqs, _ospec, orms, _order = _ordered_grid(
        np.asarray(freqs, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        np.asarray(rms_noise, dtype=float),
    )
    n = int(ofreqs.size)
    if n == 0:
        raise ValueError("the active-FT spectrum is empty")

    anchor = float(anchor_mhz)
    if anchor < float(ofreqs[0]) or anchor > float(ofreqs[-1]):
        raise ValueError(
            f"anchor {anchor:.4f} MHz is outside the analysis band "
            f"[{float(ofreqs[0]):.4f}, {float(ofreqs[-1]):.4f}] MHz"
        )
    gi = _nearest_grid_index(ofreqs, anchor)

    if not plan.windows:
        raise ValueError(
            "the window plan has no windows; re-run Stage 4 rather than "
            "creating a window against an empty plan"
        )

    # --- Locate the anchor against the existing (disjoint) window spans -----
    # Only *live* windows count: a window Stage 5 dropped covers no fit, so it
    # neither blocks an anchor nor can absorb one.
    live: Optional[Set[int]] = (
        None if live_window_ids is None else {int(w) for w in live_window_ids}
    )
    candidates = [w for w in plan.windows if live is None or int(w.window_id) in live]
    spans: List[Tuple[int, int, int]] = sorted(
        ((*_grid_span_of(ofreqs, w.freq_range), int(w.window_id)) for w in candidates),
        key=lambda t: (t[0], t[1]),
    )
    covering_wid = _grid_index_covering_window(spans, gi)
    if covering_wid is not None:
        wlo, whi = plan.window(covering_wid).freq_range
        raise ValueError(
            f"anchor {anchor:.4f} MHz already falls inside window {covering_wid} "
            f"([{min(wlo, whi):.4f}, {max(wlo, whi):.4f}] MHz). Add the peak "
            f"to that window with 'review edit --window {covering_wid} --add "
            f"{anchor:.4f}' instead; window creation is for a frequency no "
            f"window covers."
        )

    below = [(lo, hi, wid) for lo, hi, wid in spans if hi < gi]
    above = [(lo, hi, wid) for lo, hi, wid in spans if lo > gi]
    gap_lo = (max(hi for _lo, hi, _w in below) + 1) if below else 0
    gap_hi = (min(lo for lo, _hi, _w in above) - 1) if above else n - 1

    # --- Desired extent: the plan's own margin, shifted to fit the gap ------
    step_mhz = abs(float(np.mean(np.diff(ofreqs)))) if n > 1 else 1.0
    step_mhz = step_mhz or 1.0
    if min_window_half_width_points > 0:
        margin = int(min_window_half_width_points)
    else:
        margin = max(int(round(min_window_half_width_mhz / step_mhz)), 1)
    floor = max(int(stage6_min_half_width_points), 1)

    lo, hi = gi - margin, gi + margin
    if hi > gap_hi:
        lo -= hi - gap_hi
        hi = gap_hi
    if lo < gap_lo:
        hi = min(gap_hi, hi + (gap_lo - lo))
        lo = gap_lo

    diagnostics: Dict[str, Any] = {
        "stage6_anchor_mhz": anchor,
        "stage6_anchor_grid_index": int(gi),
        "stage6_gap_grid_span": [int(gap_lo), int(gap_hi)],
        "stage6_margin_points": int(margin),
        "stage6_min_half_width_points": int(floor),
    }

    if (gi - lo) < floor or (hi - gi) < floor:
        return _widen_for_stage6_anchor(
            candidates,
            peaks,
            ofreqs,
            gi=gi,
            gap_lo=gap_lo,
            gap_hi=gap_hi,
            below=below,
            above=above,
            floor=floor,
            diagnostics=diagnostics,
        )

    # --- Build the new window ----------------------------------------------
    new_wid = max(int(w.window_id) for w in plan.windows) + 1
    promoted = _promoted_ppeaks(peaks, ofreqs, n)
    members = [pk for pk in promoted if lo <= pk.grid_index <= hi]

    window = FitWindow(
        window_id=new_wid,
        freq_range=(float(ofreqs[lo]), float(ofreqs[hi])),
        free_peak_indices=[pk.list_index for pk in members],
        batch=0,
        diagnostics={
            "grid_span": [int(lo), int(hi)],
            "n_strong_in_band": sum(1 for pk in members if pk.is_strong),
            "trimmed_width_mhz": abs(float(ofreqs[hi]) - float(ofreqs[lo])),
            "stage6_created": True,
            **diagnostics,
        },
    )

    depends_on = _attach_stage6_contributors(
        window,
        candidates,
        promoted,
        ofreqs,
        orms,
        lo,
        hi,
        acquisition_us=acquisition_us,
        tau_us=tau_us,
        min_freeze_snr=min_freeze_snr,
        magnitude_attachment_threshold=magnitude_attachment_threshold,
    )
    # A leaf's batch only has to follow the windows it reads.
    by_wid = {int(w.window_id): w for w in candidates}
    dep_batches = [by_wid[p].batch for p in depends_on if p in by_wid]
    window.batch = (max(dep_batches) + 1) if dep_batches else 0

    return Stage6WindowProposal(
        window=window,
        mode="created",
        depends_on=depends_on,
        diagnostics=window.diagnostics,
    )


def _promoted_ppeaks(peaks: List[Peak], ofreqs: np.ndarray, n: int) -> List[_PPeak]:
    """The promoted Stage 3 peaks resolved onto the ascending grid."""
    promoted: List[_PPeak] = []
    for li, p in enumerate(peaks):
        if not p.properties.get("promoted"):
            continue
        promoted.append(
            _PPeak(
                list_index=li,
                grid_index=_nearest_grid_index(ofreqs, p.frequency) if n else 0,
                frequency=float(p.frequency),
                snr=float(p.snr) if p.snr is not None else 0.0,
                intensity=float(p.intensity),
                is_strong=(p.classification == PeakClassification.STRONG),
            )
        )
    return promoted


def _attach_stage6_contributors(
    window: FitWindow,
    source_windows: List[FitWindow],
    promoted: List[_PPeak],
    ofreqs: np.ndarray,
    orms: np.ndarray,
    lo: int,
    hi: int,
    *,
    acquisition_us: float,
    tau_us: Optional[float],
    min_freeze_snr: float,
    magnitude_attachment_threshold: float,
) -> List[int]:
    """Attach the base plan's strong lines that materially leak into ``window``.

    The same Tier-1 magnitude rule Step 4 of :func:`_finalize_plan` applies, run
    for one window against the plan's existing primaries. Returns the sorted
    primary window ids -- the window's inbound dependencies.
    """
    by_list_index = {pk.list_index: pk for pk in promoted}
    primary_of_strong: Dict[int, int] = {}
    for w in source_windows:
        for li in w.free_peak_indices:
            pk = by_list_index.get(li)
            if pk is not None and pk.is_strong:
                primary_of_strong[li] = int(w.window_id)

    w_center_mhz = 0.5 * (float(ofreqs[lo]) + float(ofreqs[hi]))
    w_rms_mean = float(np.mean(orms[lo : hi + 1]))
    sigma_c_w = w_rms_mean / math.sqrt(2.0) if w_rms_mean > 0 else 0.0
    threshold = magnitude_attachment_threshold * sigma_c_w

    contributors: List[FixedContributor] = []
    for li, primary_wid in sorted(primary_of_strong.items()):
        s_pk = by_list_index[li]
        df_mhz = abs(s_pk.frequency - w_center_mhz)
        if df_mhz <= 0.0:
            continue
        predicted_skirt = s_pk.intensity * _leakage_envelope_fraction(
            df_mhz * 1e6, acquisition_us, tau_us
        )
        if predicted_skirt < threshold:
            continue
        contributors.append(
            FixedContributor(
                peak_index=li,
                primary_window_id=primary_wid,
                frequency_mhz=s_pk.frequency,
                freeze_eligible=s_pk.snr >= min_freeze_snr,
            )
        )
    window.fixed_contributors = contributors
    return sorted({fc.primary_window_id for fc in contributors})


def _widen_for_stage6_anchor(
    source_windows: List[FitWindow],
    peaks: List[Peak],
    ofreqs: np.ndarray,
    *,
    gi: int,
    gap_lo: int,
    gap_hi: int,
    below: List[Tuple[int, int, int]],
    above: List[Tuple[int, int, int]],
    floor: int,
    diagnostics: Dict[str, Any],
) -> Stage6WindowProposal:
    """Absorb ``gi`` into the nearer adjacent window (the narrow-gap fallback).

    Reached only when the gap cannot hold a window with ``floor`` grid points of
    margin each side of the anchor -- a line sitting in a narrow crack between
    two windows. Widening beats creating a starved window: the neighbor already
    has a converged fit and a contributor set covering this region.

    The target is the window whose edge is nearest the anchor (lower
    ``window_id`` breaks a tie, so the choice is deterministic). Its span grows
    to reach ``floor`` points past the anchor, bounded by the *other* neighbor,
    so the plan stays disjoint. Contributors and dependency edges are carried
    over verbatim: this adds grid points to an existing window, it does not
    re-derive the plan.
    """
    n = int(ofreqs.size)
    dist_below = (gi - (gap_lo - 1)) if below else None
    dist_above = ((gap_hi + 1) - gi) if above else None

    if dist_below is None and dist_above is None:  # pragma: no cover - guarded above
        raise ValueError("no adjacent window to widen")
    if dist_above is None:
        pick_below = True
    elif dist_below is None:
        pick_below = False
    elif dist_below != dist_above:
        pick_below = dist_below < dist_above
    else:
        pick_below = below[-1][2] <= above[0][2]

    if pick_below:
        src_lo, src_hi, wid = below[-1]
        new_lo, new_hi = src_lo, min(gap_hi, max(gi + floor, gi))
    else:
        src_lo, src_hi, wid = above[0]
        new_lo, new_hi = max(gap_lo, min(gi - floor, gi)), src_hi
    new_lo = max(0, min(new_lo, n - 1))
    new_hi = max(0, min(new_hi, n - 1))

    src = next(w for w in source_windows if int(w.window_id) == int(wid))
    promoted = _promoted_ppeaks(peaks, ofreqs, n)
    members = [pk for pk in promoted if new_lo <= pk.grid_index <= new_hi]

    widened = FitWindow(
        window_id=int(wid),
        freq_range=(float(ofreqs[new_lo]), float(ofreqs[new_hi])),
        free_peak_indices=[pk.list_index for pk in members],
        fixed_contributors=list(src.fixed_contributors),
        batch=src.batch,
        diagnostics={
            **dict(src.diagnostics),
            "grid_span": [int(new_lo), int(new_hi)],
            "trimmed_width_mhz": abs(float(ofreqs[new_hi]) - float(ofreqs[new_lo])),
            "stage6_widened": True,
            "stage6_widened_from_grid_span": [int(src_lo), int(src_hi)],
            **diagnostics,
        },
    )
    return Stage6WindowProposal(
        window=widened,
        mode="widened",
        depends_on=[],
        diagnostics=widened.diagnostics,
    )
