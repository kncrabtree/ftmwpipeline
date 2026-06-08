# Stage 5 — cluster fit-quality follow-ups (missing peaks, τ collapse, skirt shape)

Status: **planning.** Follows the edge-free leakage-contributor subtraction
([`stage4-leakage-contributor-subtraction.md`](stage4-leakage-contributor-subtraction.md)),
which removed the dominant cross-window worst-ε driver. The post-fix worst-ε
windows (`scratch/stage4-leakage-contributor/worst_eps_details/`, the 5
worst-epsilon windows per fixture from the issue-#3 rerun) cluster into three
remaining failure modes. The Stage 5 plan
([`stage5-fitting.md`](stage5-fitting.md)) and the leakage-wing baseline
([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md)) remain
authoritative for everything this work does not change.

These three modes are coupled (an under-fit cluster often *also* collapses τ;
an imperfect skirt leaves residual that mimics a missing peak), so the work is
scoped as one coordinated per-window fit-quality pass, but each mode has a
distinct lever and is investigated separately first.

## Failure mode 1 — windows missing real peaks

**Evidence.** 363 w197 (`f363_w0197.png`): the |X| panel shows a clear
doublet, the fit lands K=1, and the second line sits unmodeled in the residual
(χ²ᵣ 38, τ 9.2 µs — τ is fine, the model is just short a line). Several other
worst-ε windows share this signature: a resolved real line visible in |X| and
in the residual that no model peak covers.

**Root-cause candidates (diagnose first, do not assume).**
- Stage 3 never *promoted* the line (sub-resolution apex-snap merged the
  doublet into one detection, or the partner sits below the promotion gate), so
  the conservative loop is never seeded at that offset.
- The line is promoted but the conservative add-one-peak loop's significance /
  `min_separation_factor` gate rejected it as a blend of its neighbour.
- The residual-rescue B-loop should nominate a residual peak this obvious, and
  does not — its `prominence_threshold` / `rescue_significance` may be too
  strict at the window's SNR, or rescue is not reaching these windows.

**Approach.** For each missing-peak window, classify which of the three gates
dropped the line (is it in `load_peaks`? promoted? nominated by rescue?). The
likely lever is the **residual-rescue nomination** — a peak clearly standing in
the residual is exactly what the B-loop exists to catch, so the first
investigation is why it does not fire here (and whether seeding the conservative
loop from residual maxima, not only promoted offsets, closes the gap without
inflating false positives on the dense forest). Reuse the issue-#3 SNR-aware
gate to confirm no regression.

## Failure mode 2 — τ collapse on tight clusters

**Evidence.** 655 w385 (`f655_w0385.png`): K=2, **τ = 0.63 µs**, the cluster
structure washed into a broad blob (χ²ᵣ 55). 655 w391 (`f655_w0391.png`): K=1,
**τ = 0.63 µs**, a manifestly sharp line modeled far too broad, leaving a spike
at line centre. Both τ are far below the 655 band majority (~3–5 µs) — the
per-window fit walked into a low-τ (fast-decay, broad-line) basin where a few
broad components smear over many sharp lines.

**Root-cause candidates.**
- The shared-τ NLS on a tight cluster is degenerate: dropping τ broadens every
  line so K broad lines approximate K′>K sharp ones at lower χ² in a wrong
  basin. This is usually a **symptom of under-fitting the cluster** (too few
  peaks → the fit compensates with low τ), linking mode 2 to mode 1.
- The bidirectional Gaussian τ-prior penalty (anchored on the Stage 2b
  `τ_maj`) is too weak on the low side to hold τ near the band majority — the
  penalty λ / decay factor are first-try 2638 values (penalty-tuning debt,
  [`stage5-fitting.md`]).

**Approach.**
- Treat a converged **τ ≪ band τ_maj with a high residual** as a diagnostic
  signal, not an accepted fit: either re-fit with τ pinned at the band-local
  `τ_maj` (`fit_tau=False`) and let the peak count absorb the structure, or
  add a low-side τ-prior floor (strengthen the bidirectional penalty's
  short-τ arm; the prior asymmetric-τ prototype targeted the *high* side and
  was null on the bulk — the low side is the unexamined arm).
- Couple with mode 1: when τ-collapse is detected, prefer adding cluster peaks
  (rescue) over accepting the broad blob. Quantify on the 655 / 363 dense
  clusters; gate on the SNR-aware metric so the dense bulk does not regress.

## Failure mode 3 — imperfect frozen-skirt shape

**Evidence.** 2638 w274 (`f2638_w0274.png`): K=2, τ 3.6 µs, a coherent *sloped*
residual sweeps the whole window despite the edge-free contributor — the skirt
is subtracted but its shape does not quite match, leaving a low-order coherent
remainder.

**Root-cause candidates.**
- A contributor is missing: a bright neighbour whose skirt reaches the window
  was not attached (below the magnitude-attachment threshold) or fell outside
  the `max_edge_free_neighbors` cap.
- The frozen skirt is evaluated at the **dependent window's τ** (the `h_T`
  "one τ per window" contract). A contributor from a different band has a
  different physical τ, so its skirt shape is mismodeled — an edge-free
  contributor could carry its *own* band-local τ for the skirt evaluation.
- The leakage-wing baseline is `const` (order 0); a residual *slope* wants
  order 1, which the baseline can already fit but only fires by edge coherence.

**Approach.** For each imperfect-skirt window, separate the two signatures: a
*slope* points at the baseline order (does a linear baseline clear it without
harming the const-baseline wins?); a *curved skirt remainder* points at the
single-τ skirt model (try a per-contributor τ in `evaluate_edge_free_contributors`
/ the frozen-background evaluation). Check whether a missing contributor
explains it before changing the skirt model. A/B against the baseline-on
pipeline so baseline and skirt do not double-count (the same O2 discipline the
accept/reject gate already enforces).

## Failure mode 4 — weak peaks dropped in strong-line windows

**Evidence.** Even in otherwise-good strong-line fits, clear weaker peaks are
left completely unmodeled: 1019 w046 (`f1019_w0046.png`, K=1 at SNR 21071) shows
3 — possibly 4 — unfit lines in the |residual|; 655 w442 (`f655_w0442.png`,
K=2 at SNR 43814/13245) leaves a clear weaker line beside the doublet; 1019 w006
(`f1019_w0006.png`, SNR 5779) and 655 w380 leave secondary residual peaks. In
every case the unfit peaks stand *above 3σ_c* in the |residual| panel — they are
plainly fittable, not noise.

**The tell.** These windows' χ²ᵣ is large in absolute terms (2875, 55025, ...)
yet *far below* the SNR-aware allowance set by the dominant line
(`(κ·SNR_max)²` is ~10⁵–10⁶ here), so the window **passes the acceptance gate**
and nothing pushes the fit to add the weaker lines.

**Root-cause candidate.** The conservative add-one-peak loop and the
residual-rescue both judge a candidate by its *global* χ² reduction (the F-test
/ significance ratio over the whole window). A strong line's residual dominates
that denominator, so a weak line's real-but-small absolute reduction never
clears the relative threshold — it is dropped for "insufficient χ² reduction"
even though its own residual prominence is many σ. The global chi²-improvement
ratio is the wrong statistic in a high-dynamic-range window.

**Approach.** Add a **candidate-local** acceptance/nomination criterion:
residual prominence at the candidate offset measured against the Stage 2 σ over
a few local bins (a local SNR / Rayleigh test), independent of the window's
global χ². A candidate that clears the local-SNR bar is nominated and kept even
when its global χ²-ratio is buried by a brighter line. This is the natural
complement to mode 1 (both surface as missing peaks; mode 1's missing line is a
comparable-strength blend partner gated by detection/separation, mode 4's is a
weaker line gated by the global significance ratio). Validate that the local
criterion does not inflate false positives on the dense forest (655/363) — gate
on the SNR-aware metric and, on the VC fixtures, on catalog membership of the
recovered lines.

## Cross-cutting constraints

- **No numeric-default regressions.** Issue #3 confirmed the Stage 2–5 defaults
  generalize; gate every change on the SNR-aware metric
  (`fitting/validation.py`, F=3.0, κ=0.05) and on the healthy controls
  (2638 w105/w106 byte-stable, dense-forest bulk medians flat-or-improved),
  reusing `scratch/stage4-leakage-contributor/validate.py` and the issue-#3
  rerun harness.
- **Use the Stage 2 σ array** for any noise normalization; never a local
  heuristic.
- **Verify against physics**, not the fit's own χ²ᵣ: a recovered line on a VC
  fixture (655 / 1512) should match the vinyl-cyanide catalog; a τ should be
  band-physical.

## Open questions

- O1 — is mode 1 a detection (Stage 3) gap or a rescue-nomination gap? The
  lever differs (promote vs rescue).
- O2 — does τ-collapse resolve by adding peaks (mode-1 lever) or by a low-side
  τ floor, and which generalizes across the SNR range?
- O3 — does the imperfect skirt need a per-contributor τ, more contributors,
  or just a linear baseline? Cheapest sufficient lever first.
- O4 — does a candidate-local SNR/Rayleigh acceptance criterion recover the
  mode-4 weak lines without inflating false positives on the dense forest? Is it
  better placed in the conservative loop, the rescue nomination, or both?

## Tasks

1. Diagnostic pass: classify each post-fix worst-ε window into modes 1/2/3/4
   and, for the missing-peak modes, which gate dropped the line; record the
   per-window verdict.
2. Mode 1: residual-rescue nomination / seeding fix (+ SNR-aware no-regression).
3. Mode 2: τ-collapse detector + low-side prior or peak-add re-fit.
4. Mode 3: per-window slope-vs-curvature triage → baseline order / contributor
   set / per-contributor skirt τ.
5. Mode 4: candidate-local SNR/Rayleigh acceptance criterion in the
   conservative loop and/or rescue nomination, so a weak line is kept on its own
   residual prominence rather than a global χ²-ratio buried by a brighter line.
6. Cross-fixture rerun + controls; update this doc to an implementation
   overview; reconcile ROADMAP / STATUS.
