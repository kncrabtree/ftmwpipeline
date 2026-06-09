# Plan: Stage 5 NLS performance

Status: **proposed.** No code written. Consolidates the performance levers for
the per-window nonlinear least-squares fit, which is the dominant wall-clock cost
on a dense fixture (655 ≈10 min, 363 ≈16 min).

The cost is the **conservative add-one-peak loop**, not any one stage: removing
the peak cap ([`stage5-cluster-fit-quality.md`](stage5-cluster-fit-quality.md))
made a dense fixture form a few wide, high-K windows instead of many tiny ones,
so each window's add-loop runs many `fit_window` solves over a larger system.
Instrumentation pins it: on 655 the conservative loop is ≈97 % of the 605 s fit;
the mode-2 leakage-wing baseline refit — despite firing on 294 windows — is only
**3 %** (16 s), and 0 % on 2638. So the levers target the add-loop's many small
dense solves, not the baseline.

Unblocking this also un-defers the candidate-local-accept lever
(`local_accept_snr`), set aside because its ≈2× cost was not worth +0.017 pass
*at today's speed*.

## What the cost actually is

Every solve is **pure NumPy/SciPy** — there is no explicit threading in the
package. The wall-clock is dominated by **thousands of small dense solves in the
conservative add-one-peak loop** — `scipy.optimize.least_squares(method="trf")`,
each iteration factoring the `(2M × 3K+1)` Jacobian (LAPACK QR / normal
equations), run per window, per conservative-loop K, and per knockout/merge
refit. On 655 this is ≈97 % of the fit.

*Not* the cost (measured, to forestall re-attribution): the active-FT FFT is one
~600k-point transform per fixture (negligible); the mode-2 leakage-wing baseline
(its `np.linalg.lstsq` trigger + τ-free refit) is ≈3 % on 655 and 0 % on 2638.

A multithreaded BLAS (OpenBLAS, no `OMP_/OPENBLAS_/MKL_NUM_THREADS` set) fans
each solve across all cores — so a fit can sit at ~1500% CPU on a 16-core box.

## Levers (roughly by return on effort)

0. **Batched / doubling conservative loop (highest ceiling — it changes the
   algorithm, not a constant factor).** The conservative loop adds one peak at a
   time: ~K sequential `fit_window` calls (cumulative ~O(K³) plus per-add
   residual-rescreen / candidate-pick / AICc overhead paid K times) — the
   dominant cost on the wide cap-removed windows. Since the cap removal, the
   candidate positions are *known upfront* (the promoted Stage-3 peaks in the
   window), so discovery is largely already done and the incremental
   one-at-a-time growth is partly vestigial. Replace it with **geometric-batch
   growth** ordered by residual prominence — fit the strongest 2, then 4, then 8,
   … (≈log K full fits, each warm-started from the previous so the strong,
   well-determined peaks anchor the fit before the weak/ambiguous ones join) —
   then run the existing knockout/merge cleanup **once** as the per-peak
   justification (the AICc gate moves from per-add to post-fit prune). Note the
   add/remove asymmetry: a high residual after the strong half means *more peaks
   are needed* (add by prominence), while *spurious* peaks are removed by
   knockout (not a symmetric binary remove — the spurious ones aren't the weakest
   contiguous half). Reuse the blend-aware seeder (already does K=1→2→3 mini-batch
   growth for blends). **Hard constraint:** this is a perf change, not a behaviour
   change — a joint multi-peak fit can settle in a *different* local minimum than
   the slow path, so the batched peak set must be validated to match the
   incremental loop's cross-fixture (SNR-aware metric + per-window peak-set
   diff), not assumed. (Operator's "binary-search the peak set" idea.)

1. **Test pinning BLAS threads (`OPENBLAS_NUM_THREADS=1`, near-free).** The
   per-window Jacobians are smallish (`2M × 3K+1`, M ≈ 50–1000 bins, K ≈ 1–20);
   on matrices this size BLAS thread spawn/sync overhead can *exceed* the FLOP
   saving, and 16 threads on a shared host thrash. Pinning would also stop one
   fit monopolizing the machine. **Whether it is a net win is unmeasured** — the
   one clean single-thread attempt here was confounded (and the active-FT FFT,
   once thought to want threads, is a single ~600k-point transform, so it is not
   the deciding factor either way). **Run a clean single-thread-vs-default timing
   A/B before changing a default.** A perf-only change: fit results must be
   byte-identical.

2. **Sparse / banded Jacobian + `tr_solver="lsmr"`.** A Lorentzian's derivative
   is negligible beyond a few FWHM, so a far-apart peak pair contributes almost
   nothing to the off-diagonal `JᵀJ` block — the system is **near block-banded**
   in frequency. Passing `jac_sparsity` (or `tr_solver="lsmr"`) lets TRF skip the
   dense normal-equations solve, the structural win for the new 24–40 MHz
   windows. *This is the same coupling structure that
   [`intra-window-clustering.md`](intra-window-clustering.md) reads from the
   **covariance** `(JᵀJ)⁻¹` after the fit:* block-diagonal `JᵀJ` ⇔
   block-diagonal covariance ⇔ decoupled peaks. That plan exploits it *post-fit*
   (decompose for AIC accounting); this lever exploits it *in-fit* (skip the
   coupling that is already ~0). A shared "frequency-coupling block structure"
   helper could serve both. *Caveat:* truncating the tails perturbs the result,
   so it needs an accuracy A/B, not just a timing one.

3. **Warm-start the O(K) knockout / merge refits.** Each drops one peak and
   re-seeds from scratch; seeding from the K-peak solution (minus that peak)
   slashes iterations on the sweep that dominates high-K windows.

4. **`x_scale="jac"`.** Better conditioning → fewer iterations everywhere,
   essentially free.

5. **Short-circuit provably-supported knockouts.** Skip the (K−1) refit when a
   peak's freeze-others Δχ² (already computed cheaply) is overwhelming.

## Cross-cutting constraint

Levers 1, 3, 4, 5 are **perf-only** — gate them on byte-identical fit results
plus wall-clock. Lever 2 changes the numbers slightly (tail truncation) and must
additionally pass the SNR-aware cross-fixture metric + the 2638 control, like any
fit-behaviour change.

## Task breakdown

Deferred. Start with lever 1 (a clean timing A/B; cheapest, reversible) before
the structural levers.
