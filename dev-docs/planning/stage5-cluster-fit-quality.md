# Stage 5 — cluster fit-quality follow-ups (missing peaks, τ collapse, skirt shape)

Status: **the per-window peak cap is removed** (windows are bounded by width
alone); this resolves the dense-cluster under-fit *and* the narrow-window
over-fit at the root. The earlier candidate-local-accept lever (modes 1 + 4) is
**superseded and deferred** — it patched the same defect downstream at roughly
twice the dense-fixture cost, so it is set aside until the Stage 5 NLS is sped up
([[stage5-nls-performance]]; it survives as a re-appliable patch). **Mode 2 is
resolved** (655's τ-collapse is a leakage pedestal, fixed by the leakage-wing
baseline carrying it; see below); **mode 3 remains open.** Follows the
edge-free leakage-contributor subtraction
([`stage4-leakage-contributor-subtraction.md`](stage4-leakage-contributor-subtraction.md)),
which removed the dominant cross-window worst-ε driver. The Stage 5 plan
([`stage5-fitting.md`](stage5-fitting.md)) and the leakage-wing baseline
([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md)) remain
authoritative for everything this work does not change.

## What the diagnostic found (and how it revised the plan)

A per-window diagnostic over the seven issue-#3 fixtures' worst-ε windows
classified the failures and traced, for every uncovered residual line, which
gate dropped it (Stage 3 detection / promotion, Stage 4 assignment, or the
Stage 5 acceptance gate). The verdict collapsed the four nominally-separate
modes into a different structure than the original plan assumed:

- **Modes 1 and 4 are one defect.** Across 363 / 360 / 1231 / 1512 / 1019 the
  dominant worst-ε window is a dense cluster where Stage 4 assigned 5–7 free
  peaks and 5–7 *promoted* Stage-3 lines stand uncovered in the residual at
  SNR 18–45 — yet the conservative loop fits **K=1**. The lines are detected,
  promoted, *and* assigned: only the **Stage-5 AICc-with-`n_eff` accept gate**
  drops them. On a dense cluster the perplexity `n_eff` is small (≈10–75), so
  `n_eff·log(χ²/n_eff)` *undervalues* a 44 % χ² drop while the small-sample
  correction `2k(k+1)/(n_eff−k−1)` *explodes* (→ `+inf` at `n_eff≈10`); every
  real line is parked `[tentative]` and never committed (`patience=1` then ends
  the loop). Mode 1's "comparable-strength blend partner" and mode 4's "weaker
  line beside a bright one" are the same line dropped by the same global gate.

- **Mode 2's planned lever was falsified.** On the high-SNR dense fixture 655,
  τ collapses to ≈0.62 µs (band τ_maj 3.09). The plan proposed pinning τ at the
  band majority or adding a low-side τ-prior floor. Re-fitting with τ *pinned*
  at 3.09 (frozen background subtracted) makes χ²ᵣ dramatically **worse**
  (w385 55→347, w376 11.5→441); the windows genuinely fit better broad, and
  the candidate-local lever does not lift them either. τ-collapse on 655
  is therefore **not** an under-fit symptom a τ-prior fixes — most likely an
  unresolved blend or a band-mismatched leakage pedestal (the frozen skirt is
  evaluated at the dependent window's τ; a contributor from a faster-decaying
  band leaves a broad residual only a low-τ component absorbs). Mode 2 needs
  re-diagnosis before any lever; the τ-pin / τ-floor is out.

- **Mode 3 is separate and smaller** (2638 w274: coherent low-level residual
  sweeping a wide window that already passes the gate).

Diagnostic harness and full verdict: `scratch/stage5-cluster-fit-quality/`
(`diagnose.py`, `audit_probe.py`, `refit_probe.py`, `DIAGNOSTIC_VERDICT.md`).

## Resolution (modes 1 + 4) — remove the per-window peak cap

The `n_eff` starvation that drives modes 1 + 4 is **caused by the per-window
peak cap**, not by the accept gate alone. Two coupled caps defaulted to 8:
`stage4.clustering.max_peaks_per_window` (the Stage-4 split target) and
`stage5.conservative.max_peaks` (the add-loop cap), the first tracking the
second. On a dense forest the cap chops one physical cluster into several narrow,
few-point windows; each fragment has a small `n_eff`, and the AICc-with-`n_eff`
gate then misbehaves on *both* sides — it under-fits some fragments (parking real
lines) and over-fits others (packing weak near-resolution peaks). The over-fit
and the under-fit are the same starvation seen on neighbouring slices of one
cluster.

The fix is to **bound a window by `max_window_width_mhz` (default 40) alone** and
remove the peak cap: both `max_peaks_per_window` and `conservative.max_peaks`
default to `0` ( = "no cap"). With more informative bins per window the
perplexity `n_eff` is large enough that the gate self-regulates K, fixing
under- and over-fit together. A positive value restores an explicit cap (power
users / diagnostics); the two should then be set together.

**Structural evidence (363, 31246–31260 MHz).** This ≈11 MHz region holds ≈37
promoted peaks — far over any 8-peak cap — so the cap split it into seven
contiguous fragments (inter-window gap = one grid step). The fragments fit
inconsistently: **w197** (1.0 MHz, 3 free peaks) over-fit to **K=5**, while its
immediate neighbours **w196 / w198** (≈2 MHz, 7 free peaks each) under-fit to
**K=1** at χ²ᵣ **58 / 25**. Bounded by width alone the region forms a few wide
windows the gate resolves cleanly.

**Bounded, not a mega-window.** The width cap (and the width-bounded
strong-cluster merge) is what prevents a dense ultra-high-SNR spectrum from
collapsing into one GHz-scale window — *not* the peak cap. The documented 655
force-merge runaway was the unbounded strong-cluster merge, governed by the
width cap (see [[stage4-mega-window-and-chi2-snr-floor]]); with the peak cap
removed and the 40 MHz width cap intact, 655 stays bounded (widest window
≈ 38.7 MHz) and its SNR-aware pass *improves*. Windows now settle at K ≈ 2–19
on their own and bind on width, not peak count — the cap was almost pure
downside.

### Validation (SNR-aware metric, F=3.0, κ=0.05; bulk = windows snr_max < 100)

Full Stage 4 + Stage 5 re-fit. The table isolates the **cap effect** as a single
variable — the candidate-local lever is held *on* in both columns, the cap goes
8 → 40 — so the gain is attributable to the cap alone. It improves or holds the
overall pass on **every** issue-#3 fixture; the dense fixtures gain most and the
healthy 2638 control holds at 1.0.

| fixture | overall pass (cap 8 → 40) | bulk median χ²ᵣ |
|---|---|---|
| **363** | 0.779 → **0.871** | 1.96 → 1.46 |
| 2638 | 1.000 → 1.000 | 1.27 → 1.25 |
| 360 | 0.958 → **0.979** | 1.42 → 1.37 |
| **1231** | 0.886 → **0.969** | 1.34 → 1.27 |
| 1512 | 0.953 → **0.995** | 1.36 → 1.35 |
| 1019 | 0.967 → 0.967 | 1.29 → 1.29 |
| **655** | 0.856 → **0.914** | 1.21 → 1.13 |

Two refinements beyond the table, both in the shipped direction: (a) going fully
**unbounded** (the shipped default) pushes the densest fixture further — 363
reaches **0.886** — and stays bounded (363 windows ≤ 40 MHz; 655 ≤ 38.7 MHz);
(b) the shipped default additionally **drops** the candidate-local lever
(cap-only), which is neutral-to-small here — the lever's standalone contribution
is ≤ 0.017 on 363, so the cap is the dominant effect. The shipped `cap = 0`
default is confirmed on the 2638 control (overall pass **1.000**, max K = 13,
i.e. the uncapped loop does not over-fit a healthy window).

### Cost and the persisted-settings caveat

- **Cost.** Wide, high-K windows make the dense fixtures ≈5× slower (363 / 655
  ≈10–17 min; sparse fixtures are unaffected) — this is the *conservative loop*
  on the wider windows, measured as ≈97 % of 655's fit time (the mode-2 baseline
  refit is only ~3 %). The win is correctness; the cost is an optimization,
  consolidated in [[stage5-nls-performance]] (pin BLAS
  threads — the per-window matrices are small enough that threading hurts;
  banded/sparse Jacobian + `tr_solver='lsmr'`; warm-started knockout/merge
  refits; `x_scale='jac'`). Sequence it *after* the correctness changes so a
  perf change and a behaviour change are not conflated.
- **Persisted settings.** Resolved settings persist into the `.ftmw` file and
  *persisted outranks the hard default*, so an existing file that already stored
  `max_peaks = 8` keeps the old behaviour until it is re-assigned with an
  explicit override (or re-imported). New files get the width-bounded default
  automatically. This is the documented resolution contract, not a regression.

## Superseded / deferred — candidate-local accept (modes 1 + 4 lever)

Before the cap was identified as the root cause, a **candidate-local accept**
path was prototyped in `conservative_fit` (`ConservativeSubSettings.
local_accept_snr`, default 4.0): when the global AICc-with-`n_eff` gate rejected
a well-separated, F-significant candidate that stood ≥ 4 σ_c over the model in
its own residual bin, the loop accepted it anyway. It patched the *symptom* (the
gate dropping real lines on a starved window) rather than the *cause* (the
starvation). Removing the cap subsumes it: on 363 it lifted the overall pass
0.735 → 0.779 where the cap removal alone reaches 0.886, and adding it on top of
the cap removal buys only +0.017 (0.854 → 0.871 measured at cap-40) at ≈2× the
dense-fixture wall-clock (it adds peaks → more rescue/knockout sweeps).

It is therefore **deferred, not deleted**: revisit it once the NLS speedup makes
its cost negligible, to catch any residual narrow-window case the gate still
misses. The full implementation (impl + settings + serialization + scan knob +
viz) is preserved at
`scratch/stage5-cluster-fit-quality/leverA-mode1plus4.patch`. A second lever,
**evaluating the AICc on `n_data` instead of `n_eff`**, was falsified (runaway
peak-adding; the `n_eff` correction is the load-bearing anti-overfit guard the
sub-resolution work relies on — [[stage5-subresolution-overfit-discriminant]]).

## Resolved — mode 2 (τ collapse) is a leakage pedestal

655's τ-collapse is **not** a τ-prior deficit (the planned τ-pin/floor was
falsified, *and* re-anchoring the τ penalty at the true ~4.0 µs was also null on
the aggregate — the data drives τ to the 0.62 µs floor regardless of the
anchor). It is an **un-modeled smooth leakage pedestal**: the summed far-wings of
the hundreds of lines the discrete frozen contributors cannot individually
subtract, which the shared τ collapses to absorb. Confirmed it is *not* fixable
at the source — the under-subtraction is τ-independent (far-field skirt is
τ-independent; per-contributor source-τ leaves the same ~2.5e-5 residual), so a
better contributor model cannot capture a hundreds-line continuum per window.

The fix is the **leakage-wing baseline** absorbing the pedestal instead of τ
([[stage5-leakage-wing-baseline]]): three coupled changes — (1) it also triggers
on a smooth in-band residual (an order-`p` F-test), not just edge coherence;
(2) the default order is 4, enough to follow the pedestal ramp/curvature while
staying too smooth to mimic a line (windows are ≥ ~50 bins); (3) the refit
re-frees τ (anchored at the band majority), so τ relaxes to ~3 µs once the
baseline carries the pedestal. 655: 51 of 56 collapsed windows recover, overall
SNR-aware pass **0.919 → 0.950**, bulk median 1.15 → 1.11, genuinely strong
lines preserved, 2638 control unchanged. The τ-free refit is cheap (≈3 % of
655's fit time; 0 % on 2638) — the dense-fixture cost is the cap-removal
conservative loop, not this baseline ([[stage5-nls-performance]]).

The residual χ² floor on the few snr ≈ 10⁴–10⁵ strong-line windows is the
*SNR² model-fidelity floor* (the model is "off" by a fraction of a percent at
extreme SNR), which the κ·SNR² gate passes by design — a separate lineshape
matter, not the pedestal.

## Open — mode 3 (skirt shape)

- **O3 (mode 3).** For the imperfect-skirt windows, separate a *slope*
  (baseline order) from a *curved skirt remainder* (single-τ skirt model); try
  the cheapest sufficient lever first, A/B against the baseline-on pipeline so
  baseline and skirt do not double-count. Note the mode-2 baseline now carries
  much of the broad-skirt remainder, so re-scope O3 against the post-fix
  residual.

## Cross-cutting constraints (unchanged)

- **No numeric-default regressions.** Gate every change on the SNR-aware metric
  (`fitting/validation.py`, F=3.0, κ=0.05) and the healthy controls (2638
  holds at 1.0, dense-forest bulk medians flat-or-improved).
- **Use the Stage 2 σ array** for any noise normalization; never a local
  heuristic.
- **Verify against physics**, not the fit's own χ²ᵣ.
