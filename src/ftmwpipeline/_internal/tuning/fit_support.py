"""Stage 5 fit-window selection.

A Stage 5 knob sweep re-runs the full per-window fit for every grid value, which
on a dense spectrum (2638 has ~300+ windows) is far too slow to tune
interactively. So before the sweep the persisted Stage 4 plan is reduced *once*
on the working copy to a representative subset: the brightest windows, a seeded
random sample of the rest, and any windows pinned by frequency. The subset is
closed over joint-fit dependency components so jointly-fit windows stay correct;
fixed contributors sourced from dropped windows simply stay frozen at their
Stage-3-seeded values. The selection is deterministic and identical across every
grid value, so the per-value metric deltas reflect the knob, not resampling.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def window_snr_max(wf: object) -> float:
    """Brightest in-window fitted-peak SNR (0 if none). Mirrors the Stage 5
    validation's per-window SNR_max so the ε / pass gate the tuning surface
    reports matches the shipped health report."""
    import numpy as np

    snrs = [
        float(p.snr)
        for p in wf.fitted_peaks  # type: ignore[attr-defined]
        if p.snr is not None and np.isfinite(p.snr)
    ]
    return max(snrs) if snrs else 0.0


def window_fit_quality(wf: object) -> Dict[str, object]:
    """Per-window fit-quality row shared by the Stage 5 metric and plot: reduced
    χ²ᵣ, SNR_max, the SNR-normalized shape-error fraction ε, and the SNR-aware
    pass gate — all from the shipped :mod:`fitting.validation` so the tuning
    surface and the Stage 5 health report agree.

    The gate ``χ²ᵣ ≤ F + (κ·SNR)²`` is equivalent to ``ε ≤ κ`` in ε-space, so a
    plot of ε versus SNR carries the pass boundary as a flat line at ``κ``.
    """
    from ftmwpipeline.fitting.validation import (
        DEFAULT_CHI2R_NOISE_FLOOR,
        DEFAULT_SHAPE_ERROR_KAPPA,
        shape_error_fraction,
        snr_aware_chi2_pass,
    )

    chi2r = float(getattr(wf, "reduced_chi2", float("nan")))
    snr_max = window_snr_max(wf)
    return {
        "window_id": int(wf.window_id) if wf.window_id is not None else -1,  # type: ignore[attr-defined]
        "chi2r": chi2r,
        "snr_max": snr_max,
        "epsilon": shape_error_fraction(chi2r, snr_max, DEFAULT_CHI2R_NOISE_FLOOR),
        "passed": snr_aware_chi2_pass(
            chi2r, snr_max, DEFAULT_SHAPE_ERROR_KAPPA, DEFAULT_CHI2R_NOISE_FLOOR
        ),
        "n_peaks": len(wf.fitted_peaks),  # type: ignore[attr-defined]
    }


@dataclass(frozen=True)
class FitWindowSelection:
    """Which Stage 4 windows a fit-knob sweep should re-fit.

    ``top_snr`` keeps the N brightest windows (the strong lines you want to
    watch); ``sample`` adds a seeded random sample of the remaining windows for
    band-representative coverage; ``freqs`` pins the window nearest each
    frequency (MHz) so a specific line is always included. ``fit_all`` bypasses
    the reduction entirely (fit every window). The default (``top_snr=3``,
    ``sample=20``) keeps a sweep to a few tens of windows.
    """

    top_snr: int = 3
    sample: int = 20
    freqs: Tuple[float, ...] = ()
    sample_seed: int = 0
    fit_all: bool = False


def _spread_evenly(ids: List[int], value_by_id: Dict[int, float], k: int) -> List[int]:
    """Pick ``k`` ids spread evenly across their value range (here SNR), so a
    straddle sample covers the whole grid range rather than clumping."""
    if k <= 0 or not ids:
        return []
    ordered = sorted(ids, key=lambda i: value_by_id[i])
    if k >= len(ordered):
        return ordered
    if k == 1:
        return [ordered[len(ordered) // 2]]
    picks = {round(j * (len(ordered) - 1) / (k - 1)) for j in range(k)}
    return [ordered[i] for i in sorted(picks)]


def _close_components(keep: Set[int], edges: List[Tuple[int, int]]) -> Set[int]:
    """Grow ``keep`` to the full connected component (over ``edges``) of each
    member, so a window that may be jointly co-fit with a neighbor never loses
    that neighbor."""
    adj: Dict[int, Set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    out = set(keep)
    stack = list(keep)
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in out:
                out.add(y)
                stack.append(y)
    return out


def reduce_plan_for_fit(
    path: Path,
    selection: FitWindowSelection,
    spec: object = None,
    values: Optional[Sequence[Any]] = None,
) -> None:
    """Reduce the persisted Stage 4 window plan on ``path`` in place to the
    subset described by ``selection``.

    A no-op when ``selection`` is ``None`` / ``fit_all``, when no selection
    criteria are set, or when the plan already fits within the requested budget.
    Used as a :class:`~ftmwpipeline._internal.tuning.registry.KnobSpec` prepare
    hook: the engine calls it once on the freshly copied working file before the
    sweep loop, passing the knob ``spec`` and its resolved grid ``values``.

    **Knob-aware sampling.** For an SNR-threshold knob (``spec.select_hint ==
    "snr_threshold"``) part of the sample budget is spent on windows whose
    SNR_max straddles the grid range ``[min(values), max(values)]`` — the windows
    that actually flip as the threshold sweeps — so the knob is guaranteed to
    bite instead of looking inert because the random draw missed its regime. The
    remaining budget is the ordinary random sample.
    """
    if selection is None or selection.fit_all:
        return
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl,
        save_window_plan_impl,
    )

    plan = load_windows_impl(str(path))["plan"]
    budget = max(0, selection.top_snr) + max(0, selection.sample)
    # Already small enough that fitting the whole plan is cheap -> leave it.
    if budget and len(plan.windows) <= budget:
        return

    peaks = ftmw.load_peaks(path)

    def win_snr(w: object) -> float:
        snrs = [
            float(peaks[i].snr)
            for i in w.free_peak_indices  # type: ignore[attr-defined]
            if 0 <= i < len(peaks) and peaks[i].snr is not None
        ]
        return max(snrs) if snrs else 0.0

    snr_by_id = {w.window_id: win_snr(w) for w in plan.windows}
    keep: Set[int] = set()
    for w in sorted(plan.windows, key=lambda w: snr_by_id[w.window_id], reverse=True)[
        : max(0, selection.top_snr)
    ]:
        keep.add(w.window_id)
    for f in selection.freqs:
        nearest = min(
            plan.windows,
            key=lambda w: abs((w.freq_range[0] + w.freq_range[1]) / 2.0 - f),
        )
        keep.add(nearest.window_id)

    sample_budget = max(0, selection.sample)
    # Knob-aware straddle for SNR-threshold knobs: spend up to half the sample
    # budget on windows whose SNR flips across the grid range.
    hint = getattr(spec, "select_hint", None) if spec is not None else None
    if hint == "snr_threshold" and values and sample_budget:
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        if nums:
            lo, hi = min(nums), max(nums)
            straddle = [
                wid
                for wid in snr_by_id
                if wid not in keep and lo <= snr_by_id[wid] <= hi
            ]
            n_straddle = min(len(straddle), max(1, (sample_budget + 1) // 2))
            picked = _spread_evenly(straddle, snr_by_id, n_straddle)
            keep.update(picked)
            sample_budget -= len(picked)

    rest = [w.window_id for w in plan.windows if w.window_id not in keep]
    if sample_budget and rest:
        rng = random.Random(selection.sample_seed)
        rng.shuffle(rest)
        keep.update(rest[:sample_budget])

    if not keep:
        return  # no criteria selected anything -> leave the full plan

    keep = _close_components(keep, plan.dependency_edges)

    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [wid for wid in plan.topological_order if wid in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(path), plan)
