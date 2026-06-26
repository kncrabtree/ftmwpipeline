# Stage 5 — per-window cleanup folded into the fit walk (implementation summary)

**Status: shipped** on `stage6-cascade-refit` (commits `2c0f1f8`, `45f3b83`,
`bd51709`). This doc is the implementation summary; the per-twist investigation
is in git history and memory `stage5-inline-cleanup-recall-regression`. Companion:
[`stage6-cascade-refit.md`](stage6-cascade-refit.md) (the cascade gate that
surfaced this); divergence record ROADMAP **D16**.

## What it does

The Stage-5 fit is a dependency-DAG walk: each window's fit reads its converged
predecessors as a frozen-contributor background. The per-window cleanup (SNR-floor
**prune** of sub-floor dust + degenerate-overfit **VIF-collapse**) now runs **in
the walk's per-node tail**, before the DAG releases a window's dependents — so
every dependent is fit against an already-cleaned source. The previous design ran
the cleanup as a **global post-pass** (with its own second fork pool) after the
whole walk, which left every dependent fit against the *pre-cleanup* source and
its frozen-background snapshot stale relative to the source's final line list (the
`#3b` class; e.g. 655 `w918` dependents fitting a doubled, pre-collapse `w918`).

## How it is built

- **`refit_outcome`** (`fitting/plan_execution.py`) — the live-outcome refit
  primitive the cleanup drives. Reuses the node's frozen background / grids / spur
  mask / per-band τ off the `WindowOutcome` and rebuilds its `fit_window` kwargs
  exactly as `refit_window_core` does, so the node's fit context is inherited by
  construction (it physically cannot re-resolve τ / baseline / penalties wrong —
  the old post-pass `tau_maj`-vs-per-band re-resolution bug class is gone). The
  walk caches that context on each outcome (`stash_refit_context`).
- **Shared fitted-line view** (`fitting/result_conversion.py`: `FittedLineView` +
  `outcome_line_views` / `result_line_views`) — one decision surface the
  prune/collapse logic reads from *either* a `WindowOutcome` (in-walk) or a
  persisted `FittingResult`. The decision functions are pure and module-level:
  `_is_survival_dust_view` (dust predicate) and `_collapse_rank` (collapse
  eligibility: amplitude VIF, or the singular-covariance degeneracy the VIF gate
  is blind to) in `_internal/stage5_impl.py`.
- **`build_finalize_node`** (`_internal/stage5_impl.py`: `_prune_outcome` /
  `_collapse_outcome`) — the SNR-prune → VIF-collapse fixpoint in outcome space,
  returning a cleaned outcome or a `NodeCleanup` drop (window cascaded to empty).
  Injected into all three walk paths (`_walk_windows_dag`, the sequential
  authority, the legacy level walk) via a callback; `_fit_peaks_impl` aggregates
  per-window provenance into the `peak_survival` / `vif_collapse` diagnostics and
  inflates merged frequency errors at end-of-walk.

### §4 — frozen background from the ancestor's *fit*, not Stage 3

A dependent freezes each ancestor window's **current above-`min_freeze_snr` fitted
lines** that fall in its leakage range (`evaluate_ancestor_leakage`), keyed only
off the Stage-4 dependency edge — not Stage-3 contributor records. Stage 3
determines the *plan* (which windows depend on which) and the gold seeds; the
frozen-background *content* is whatever the ancestor actually fit. Consequences:
N ancestor lines → N frozen peaks (no Stage-3 nearest-match doubling), never stale
(it is the ancestor's current fit by construction), and the post-fit user-edit
cascade reduces to "re-evaluate a dependent from its source's current fit."
`fixed_parameters` is re-keyed by enumeration (`frozen_peak_{i}`, `peak_index = -1`).

### Baseline warm-start (C0 — refit reproducibility)

`fit_window` co-fits a complex leakage-wing baseline but always cold-started those
coefficients at zero. A refit warm-started the *peaks* from their converged values
but reset the *baseline* to zero, so the seed was never the full converged state;
on wide, low-SNR windows the near-degenerate baseline/position valley let an
identity refit slide untouched peaks (655 `w848` ~1.2 MHz). Fix: seed the baseline
from the persisted converged coefficients (`initial_baseline_coeffs` on
`fit_window`, wired through both `refit_window_core` and `refit_outcome`). The
baseline stays a **free parameter** — the model is unchanged; the refit just
starts at the true minimum. This superseded and removed the `freeze_baseline`
prototype (which dropped the baseline degree of freedom and blew up χ²ᵣ on
contributor-bearing windows). The automatic in-walk fit is byte-/recall-identical;
only refits change.

### §5 — dead-code retirement

The global post-pass functions (`apply_snr_survival_prune`,
`_survival_prune_window`, `_is_survival_dust`, `apply_vif_collapse`,
`_collapse_one`, orphaned `_merged_seed_for_pair`) are deleted; their unit tests
are repointed at the view-based decision functions
(`tests/unit/fitting/test_peak_survival_prune.py`). The spec's "move
`refit_window_core` to `stage5_impl` + wrap `refit_outcome`" was dropped as
obsolete: the cleanup already routes through `refit_outcome`, so the
`stage5 → stage6` import it would have removed never existed, and
`refit_window_core` cohabits stage6 with its sole caller `refit_window_impl`. The
window-level empty/spur-only drop (`apply_window_cleanup`) legitimately stays a
global post-pass.

## Result (7-fixture, `s4c` arm — `scratch/cascade/BASELINE.md`)

- **Inline cleanup (§§1–4)** vs the prior committed HEAD: 6/7 fixtures flat; 655
  (densest) isotopologue recall +9 (→0.885), χ²ᵣ p90 3.89→2.68, main recall held
  at 0.662, with *fewer* collapses (cleaner upstream over-splits less).
- **Baseline warm-start** is output-neutral on the automatic fit (5/7 fixtures
  byte-identical; 1512 +1 line, 655 +2; recall identical) and fixes refit
  reproducibility: 655 identity-refit movers 21→14 (worst 1.2 MHz → 84 kHz, no
  catastrophe), 363 0/401, 1231 4/206.

The residual ~14 movers on 655 are a characterized statistical floor — weak (snr
3–7) sub-resolution **blend members** reseeding into an equivalent-χ²
configuration within ~1–3× their own σ_f (peak-peak / multiplet degeneracy,
baseline-independent: freeze and relax give identical results on each). Not a bug;
left as-is.
