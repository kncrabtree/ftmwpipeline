# Plan: Stage 5 NLS performance

Status: **partially resolved.** Lever 4 (`x_scale='jac'`) is shipped. Levers 1
(BLAS threads) and 0 (geometric-batch add-loop) were prototyped and measured to
be, respectively, null and a net regression — see *Measured outcomes* below.
Consolidates the performance levers for the per-window nonlinear least-squares
fit, the dominant wall-clock cost on a dense fixture (655 ≈10 min, 363 ≈16 min).

## Measured outcomes (issue-3 fixtures, capture/replay on real windows)

The cost model the lever ranking assumed — many *large* sequential solves whose
cumulative cost is `O(K³)` per window — **did not hold** when measured on a
captured subset of real 655 windows (`scratch/stage5-nls/`, `capture_windows.py`
+ `replay_bench.py` replay `conservative_fit` on its recorded call args, the
self-contained ~97 % path). On the heavy subset (K up to 39, median 12) the
add-loop already self-compresses to **~8 `fit_window` calls per window** (the
blend-aware seeder and patience/tentative batching, not one-per-candidate), and
each solve converges in **~11 nfev** — nothing approaches `DEFAULT_MAX_NFEV`. So
the per-window cost is iteration-and-overhead bound across *many small* solves,
not a few large ones.

- **Lever 4 — `x_scale='jac'` (SHIPPED).** Trust-region scaling by the Jacobian
  column norms. The fit parameters span amplitude ~1e3, offset ~1e-2 MHz, phase
  ~1, tau ~5, so uniform scaling conditions the step poorly. Measured **−36 %
  solver evaluations / −17 % wall** on the heavy 655 subset (a few percent on
  light windows) for an identical converged minimum. Cross-fixture metric holds
  (1019/1512 pass 1.000 unchanged, bulk χ²ᵣ unchanged; dense fixtures confirmed
  separately). Perf-with-negligible-perturbation: the fitted line set is
  unchanged within solver tolerance.
- **Lever 2b-assembly — vectorise the model/Jacobian over peaks (SHIPPED,
  byte-identical).** A cProfile of the replay pinned the per-window cost: the
  largest self-time is the line-shape assembly (`h_T` + `h_T_jacobian` ≈ 14 %)
  with another ~15 % of scattered Python dispatch (`PeakShape.coerce` called
  144 k times, the per-peak `h_T_shape` wrapper, `_unpack`), while the actual
  TRF solve (`svd`) is only ~6 %. `model_spectrum` and `model_jacobian` now
  evaluate every line in one broadcast over the `(K, M)` offset grid (one
  `h_T_shape` / `h_T_shape_jacobian` call, shape coerced once) instead of a
  Python loop of K per-peak calls. **Byte-identical** (peak-set fingerprint
  unchanged; full suite 1247 green) for **−8 % wall on the dense 655 heavy
  subset** (6.16→5.65 s). Note `model_spectrum` alone was ~0 (its self-overhead
  is below noise); `model_jacobian` carried it (two line-shape calls per peak +
  the column fills). The transcendental `exp` FLOPs themselves are unchanged --
  only banding (below) reduces those.
- **Lever 1 — pin BLAS threads (NULL, not pursued).** OpenBLAS (16 cores, no
  thread env) leaves these `2M × 3K+1` Jacobians **single-threaded already** —
  the matrices are below its threading threshold. Default vs
  `OPENBLAS_NUM_THREADS=1` are **byte-identical in result and within noise in
  time** on both light and heavy subsets (7.52 s vs 7.49 s). The full-pipeline
  ~1300 % CPU seen at default comes from large-array ops *outside*
  `conservative_fit` (the active-FT FFT, window materialisation), a small share
  of the 10-min wall — so pinning frees cores on a shared host but does not
  speed the fit.
- **Lever 0 — geometric-batch add-loop (NET REGRESSION, not shipped).**
  Prototyped (doubling batch by residual prominence, bisection-on-reject, same
  AICc gate; `scratch/stage5-nls/window_fit_xscale_plus_batch.patch`). On the
  heavy 655 subset it was **slower (8.4 s vs 7.5 s) and over-added peaks (116 vs
  103, count drift on 7/40 windows)**. Two compounding reasons: (a) the
  incremental loop re-evaluates the residual after *each* add, so sidelobe
  candidates vanish before they are considered — a batch ranked off a *stale*
  pre-fit residual seeds those sidelobes, and the per-batch joint AICc gate
  accepts them as a group the strong members carry; (b) the extra peaks inflate
  every subsequent joint Jacobian, so the call-count saving (fewer, larger
  solves) is outweighed. A prune-augmented batch (drop knockout-unsupported
  members per batch) could recover correctness but adds the solves back, and —
  given the loop is already only ~8 solves/window — cannot beat incremental.
  The premise (call-count is the cost) is the part that failed.
- **Lever 2a — frequency-window decomposition (FIXTURE-DEPENDENT, not shipped).**
  Prototyped (`scratch/stage5-nls/window_fit_2a_decompose.patch`): cluster the
  candidate offsets by line-coupling range, grow each cluster on its own
  sub-band with τ fixed, then one τ-free whole-window joint polish; gate each
  cluster against its sub-window null (the whole-window loop force-accepts only
  the single global strongest seed). The coupling range must be **amplitude- and
  shape-aware**: a Lorentzian dispersive wing falls only as ``1/Δ`` so a strong
  line couples to distant weak peaks (radius ∝ amplitude / SNR); a Gaussian wing
  falls as ``exp(-(Δ/σ)²)`` so its radius is a fixed few-FWHM regardless of
  amplitude. With a *fixed* FWHM cutoff a strong line is wrongly split off and
  its unmodelled wing is absorbed by spurious peaks in the neighbouring cluster
  (655 over-added 2→7 on a heavy window); the amplitude-aware radius fixed that
  (heavy-window peak-set mismatches 25→12 of 40). **But it does not generalise.**
  The decisive measurement is **wall vs nfev**: on the 655 heavy subset
  decomposition cut nfev −59 % (2291→946) but wall only −10 % (6.16→5.53 s) —
  the per-window cost is dominated by **fixed per-`fit_window` overhead + the
  O(M·K) model assembly** (the transcendental ``h_T`` evals over the grid), not
  the O(M·K²) solve. So decomposing the *solve* saves little while the
  per-cluster machinery (an extra null fit + seed + gate per cluster, plus the
  joint polish + knockout) *adds* overhead. It nets a marginal ~10–13 % only on
  the most expensive Lorentzian-dense windows (655); on **363 (Gaussian), where
  fits already converge fast, it is ~2× *slower* and over-adds (67→87 lines)**.
  It is also a behaviour change — the AICc accept gate's ``n_eff`` is global
  (perplexity over the whole-window model), so the per-cluster gate shifts
  marginal accept/reject decisions and relies on the downstream
  ``iterative_aicc_cleanup`` / merge / knockout to prune the over-adds (the
  cross-fixture metric held on 1019/1512/655 but is not byte-identical).
  Net: reverted. The amplitude/shape-aware coupling model is correct physics and
  worth keeping in mind, but window decomposition is the wrong place to spend it.
- **Stage-4 small-window retune (FIXTURE/REGION-DEPENDENT, not shipped).** A
  different route to the same end: shrink the Stage-4 windows themselves
  (`max_window_width_mhz` 40→~8) so each `fit_window` sees a smaller `M`, and —
  since over a narrow window a distant bright line's `1/Δ` skirt is nearly flat
  (pure pedestal the order-4 baseline soaks) — relax the contributor attachment
  (`magnitude_attachment_threshold` 0.1→~1.0) so the baseline carries it instead
  of thousands of explicit per-window subtractions. Characterisation
  (`scratch/stage5-nls/char_stage4.py`, `char_thresh.py`): default 655 windows
  are large (median 185 pts, p95 363); cap-8 → median 65 pts, but `n_windows`
  3× and contributor *attachments* 3510→8393 (the same ~230 bright lines, each
  attached to more windows — the brightest reaches 873). The current 0.1
  threshold attaches contributors a **median 434 MHz away** (p95 ~4 GHz) — flat
  pedestal the baseline absorbs; raising it to ~1–3 drops attachments ~9× while
  keeping the near (curvature-driving) ones. Band-fit sampling
  (`scratch/stage5-nls/band_fit.py`, a monkeypatched plan-load that fits only
  windows in a frequency band): on a **moderate-density Lorentzian** band cap-8 +
  thr-1.0 is a clean **−34 % wall, metric holds** (pass 0.987 vs 1.000, +5
  lines); but on a **dense-blend** band it is **slower and +67 % lines** (cutting
  a real blend), and on **363 (Gaussian/sparse)** it is **2× slower and +87 %
  lines**. Scaling the baseline order with the point count (the natural "k=4
  over-fits a 60-pt window" guess) **did not move the line counts** and slightly
  hurt the metric — order-4 was helping, not over-fitting. The over-add is the
  **same context-dependent-gate problem as 0 and 2a, arriving via Stage 4**: the
  conservative-fit AICc accept gate keys on a *global* `n_eff` (perplexity over
  the window) and the seeder/rescue are less constrained with fewer competing
  peaks, so *more, smaller windows systematically over-accept lines* — even where
  the splits are clean. It only nets positive where the speedup outweighs the
  over-add (moderate-density Lorentzian); it regresses dense blends and Gaussian.
  Reverted. **The structural blocker for every region-shrinking lever (0, 2a,
  small-window) is the window-size sensitivity of the accept gate** — see the
  context-invariant-gate sketch below.

**The cross-cutting lesson** from 0 and 2a: the per-window cost is **fixed
per-call overhead + O(M·K) assembly bound**, not the O(M·K²) solve and not the
call count. Reducing solve size (2a) or call count (0) therefore yields little,
while x_scale (fewer iterations of that fixed-cost-per-iteration assembly) is the
modest win that survives. The only remaining large lever is attacking the
**assembly cost itself** — evaluating each peak's ``h_T`` / Jacobian only over
its local support so per-iteration assembly drops from O(M·K) toward
O(M·support). That perturbs results (tail truncation) → needs the metric gate.

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

The lever descriptions above are the original intent; *Measured outcomes* at the
top records which survived contact with the fixtures. Lever 4 shipped; levers 0,
1, and 2a are closed (regression / null / fixture-dependent). The framing of
those levers — that the add-loop runs `O(K)` large sequential solves, that the
Jacobians are large enough to thread, and that the solve size is the cost — is
the part the measurement falsified. The cost is fixed per-call overhead + the
O(M·K) per-iteration model assembly.

## Task breakdown

- **Done:** lever 4 (`x_scale='jac'`, shipped in `fit_window`); levers 0, 1, and
  2a prototyped and closed (see *Measured outcomes*).
- **Next: attack the assembly cost (lever 2b — local-support model/Jacobian
  evaluation).** This is the one lever aimed at the measured bottleneck. On a
  wide 24–40 MHz window each peak's `h_T` (and its Jacobian columns) is
  evaluated over the *full* grid every residual/Jacobian call, but the line
  shape is ~0 beyond a few FWHM (Gaussian) / a wing that is cheap to bound
  (Lorentzian). Restricting each peak's contribution to a ±N-FWHM band around
  its centre drops the per-iteration assembly from O(M·K) toward O(M·support),
  and this is the term the wall-vs-nfev measurement showed dominates. Unlike 2a
  it does **not** restructure the selection or the gate — same peaks, same
  add-loop, just a faster model/Jacobian — so the only perturbation is tail
  truncation, gated on the cross-fixture metric + the 2638 control. Candidate
  implementation: a banded `model_spectrum` / `model_jacobian` that sums only
  each peak's in-band rows; validate against the dense (Lorentzian wing) and the
  bright-line (long-wing) cases where truncation bites hardest.
- **Lower priority:** lever 5 (short-circuit provably-supported knockouts) and
  trimming the rescue rounds' repeated full-window work (`iterative_aicc_cleanup`
  is `O(K²)` refits per round) — call-count multipliers in the orchestration, but
  each `fit_window` there is itself assembly-bound, so 2b compounds with them.
  Lever 3 (warm-start) offers little: the cleanup refits already seed from the
  K-fit peaks minus one.
