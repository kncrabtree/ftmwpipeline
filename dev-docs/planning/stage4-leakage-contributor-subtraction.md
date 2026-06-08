# Stage 4 — edge-free leakage-contributor subtraction (the cycle-breaker gap)

Status: **planning**. Resumes the deferred **O5-10 Tier-2** item
([`stage5-fitting.md`](stage5-fitting.md): "Tier-2 cumulative-tail background
deferred"), now with cross-fixture evidence (issue #3) and a validated
single-window prototype. The Stage-4 plan
([`stage4-window-assignment.md`](stage4-window-assignment.md)) and the Stage-5
leakage-wing baseline ([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md))
remain authoritative for everything this work does not touch.

## Problem

When a bright line sits just outside a window, its finite-T leakage skirt sweeps
across that window. The window's own model + the `const` leakage-wing baseline
cannot represent that skirt, so the window's lines are **under-fit, missed, or
mis-fit** and its residual is sloped or blown up.

The skirt *should* be subtracted: Stage 4's magnitude-attachment rule (O5-10
Tier-1, `window_planning.build_window_plan`) detects exactly this — it attaches
the strong line as a `FixedContributor` of the affected window when its predicted
skirt crosses `magnitude_attachment_threshold · σ_c`. But attaching a contributor
also creates a fit-ordering **dependency edge** (the dependent window must be fit
after the contributor's primary window). On dense / high-dynamic-range spectra
the strong-line neighbourhood is densely cyclic, so the **Step-7 cycle-breaker**
(`_topological_batches` → `dropped_cyclic_dependencies`) drops the edge **and its
`FixedContributor`** (`window_planning.py` ~728-746) to keep the DAG acyclic. The
needed leakage subtraction is discarded along with the edge.

### Evidence (issue #3, all seven fixtures)

The worst-ε window on every fixture has this signature — `fixed_contributors = 0`,
cycle-dropped edges, several promoted in-window lines under-fit, and a far
stronger line just outside. Severity scales with dynamic range:

| window | neighbour SNR (outside) | symptom | χ²ᵣ |
|---|---:|---|---:|
| 2638 w49 | 308 | 6 promoted → K=1 (5 unfit) | 4.5 |
| 363 w80 | 82 | 7 promoted → K=1 | 212 |
| 1231 w266 | 1 904 | 5 promoted; a fit peak SNR≈0 (mis-fit) | 74 |
| 1512 w167 | 7 160 | 3 promoted → K=1 | 77 |
| 655 w37 | **12 482** | 4 promoted (SNR 44–62) all missed | 256 |
| 1019 w56 | **20 685** | blown-up residual | **1 906** |

**Control (proves it is the skirt, not windowing or SNR):** 2638 w105/w106 also
sit on dropped edges with strong neighbours, but they fit K=4 — capturing their
lines — so their χ²ᵣ is just the SNR² fidelity floor; they are healthy. The
failure is the *unsubtracted skirt*, not the presence of a strong neighbour.

A second, coupled defect: the bounded-merge **splitter** can place a window
boundary *inside* a bright line's skirt (363 w349 is a 0.56 MHz sliver split
0.28 MHz from an SNR-37 line; 360 w287's boundary is 1.5 MHz from an SNR-657
line), maximising the leak into the orphaned sliver.

### Prototype (positive)

A targeted edge-free A/B on 360 w287 (`scratch/issue3-cross-fixture/w287_frozen_skirt_ab.py`):
re-fitting w287 with w288's three lines frozen in as a pre-computed skirt (no
dependency edge) drops the bare χ²ᵣ **108.7 → 2.61**, flattens the residual
slope, and resolves a genuine second line (K 1→2) — landing **below** even the
production const-baseline result (22.09). The frozen contributor is a decisively
better lever than the `const` baseline for an across-window skirt.

**Caveat from the 655 work** ([`../research/stage5-cross-fixture/report.md`](../research/stage5-cross-fixture/report.md)
§Phase 1): a *global, crude* frozen-skirt pre-subtraction — every strong line's
single-bin core phasor subtracted from every other window — was **NEGATIVE**
(bulk χ²ᵣ 2.40→4.71). Two reasons: (1) the shipped leakage-wing baseline already
covers in-window leakage, so a global pass double-counts; (2) single-bin phasor
reads are pedestal-contaminated on a dense spectrum (each core bin carries ~300
other lines' summed skirts → over-subtraction). The fix must therefore be
**targeted** (only genuinely-adjacent dominant neighbours) and use **robust
amplitude reads** (LSQ against the line template), not global single-bin phasors.

## Root cause, precisely

Leakage *subtraction* is coupled to fit *ordering*. A `FixedContributor` is
modelled with its primary window's fitted `(amp, freq, phase)`, which requires
the primary to be fit first — hence the edge, hence the cycle, hence the drop. But
a frozen contributor is **not re-fit**; it only needs a good estimate of the
neighbour line's parameters, which does not *have* to come from a strict
fit-ordering predecessor.

## Approach

Decouple subtraction from ordering and stop splitting inside skirts.

### 1. Edge-free frozen contributor (the load-bearing change)

Allow a `FixedContributor` whose parameters come from a **self-contained read**
of the strong line — so it carries **no dependency edge** and the cycle-breaker
has nothing to drop.

- **Parameter source.** Prefer the strong line's parameters as estimated
  directly from the active FT (its detected frequency + a robust amplitude/phase
  read), independent of whether its primary window has been fit. Options to
  evaluate (open question O1): a small LSQ of the line template against its core
  bins (robust to the pedestal — the 655 lesson), vs the primary window's fitted
  parameters when that window happens to precede without a cycle. The skirt shape
  uses the **dependent** window's τ (the existing `subtract_frozen_background`
  contract; `h_T` carries one τ per window).
- **No edge.** An edge-free contributor is excluded from the dependency DAG, so
  `_topological_batches` never sees it and never drops it. The existing
  `subtract_frozen_background` / `fit_window_with_fixed_contributors` machinery
  consumes it unchanged (the w287 prototype already exercised this path).
- **Targeting (avoid the 655 negative).** Attach edge-free skirts only for the
  few **dominant** neighbours of a window (cap by predicted contribution and/or
  count), not every strong line in the band. Coordinate with the leakage-wing
  baseline so the two do not double-count the same in-window leakage (open
  question O2).

### 2. Splitter must not cut inside a bright line's skirt

The bounded-merge recursive split (Step 3 cap-split) chooses the largest interior
peak gap; it should additionally **refuse a boundary within a strong line's
leakage reach** (snap to the nearest leakage-clean interior point — the
`split_proposal` already computes complex-edge-clean points; the cap-split path
does not consult them). This prevents manufacturing slivers like 363 w349.

### Non-goals

- **No numeric default changes.** Issue #3 confirmed the Stage 2–5 numeric
  defaults generalize; this is an algorithm rework, not a retune.
- The **leakage-wing baseline stays** (it correctly covers *in-window* leakage
  for per-line uncertainty); this adds *cross-window* bright-neighbour
  subtraction that the `const` baseline structurally cannot represent (a slope).
- **Window merge** (fit the strong line and its weak neighbour jointly) is the
  fallback for genuinely-blended cases where the two cannot be cleanly separated
  by subtraction; the edge-free frozen skirt is the primary mechanism.

## Interface surface

Mostly internal to Stage 4 planning + Stage 5 execution; the dual interface is
barely touched.

- **`preprocessing/window_planning.py`** — the attachment rule emits edge-free
  contributors; the cycle-breaker no longer needs to drop a needed subtraction
  (cyclic *fit-ordering* edges may still be broken, but the *contributor* survives
  as edge-free). Splitter gains the skirt-proximity guard.
- **`core/data_structures.py` `FixedContributor`** — likely a new flag (e.g.
  `edge_free: bool` / a `source` discriminator) and, if the parameters are read
  directly rather than from the primary fit, fields to carry the frozen
  `(amplitude, phase)` so execution needs no primary lookup. Serialization in
  `io/` updated to match (additive, back-compatible default).
- **`fitting/plan_execution.py`** — `subtract_frozen_background` already takes the
  frozen model peaks; the change is sourcing them for edge-free contributors. No
  new public function expected.
- **`assign_windows` (api / pipeline / cli)** — no new user-facing parameter
  anticipated; behaviour change only. A cross-interface consistency test still
  applies.

## Test plan

- **Per-window A/B regression** on the w287-class windows enumerated above (one
  per fixture): each must drop to a healthy χ²ᵣ and recover its missed/mis-fit
  lines (w287 108.7→2.61 is the template assertion).
- **Healthy-control no-regression**: 2638 w105/w106 (and other K-captured
  windows) must stay byte-stable / peak-count-identical.
- **Dense-forest no-regression**: the 655 bulk χ²ᵣ must **not** worsen (the
  global-crude form regressed 2.40→4.71 — the targeted form must avoid that).
  Gate on the SNR-aware metric (D10), not raw χ²ᵣ.
- **Cross-fixture worst-ε re-check**: re-run the issue-#3 harness; the worst-ε
  populations should shift from "missed/mis-fit beside a strong line" to the
  intrinsic SNR² fidelity floor.
- **Plan invariants**: windows stay disjoint; the DAG stays acyclic;
  `dropped_cyclic_dependencies` no longer carries edges whose contributor was
  needed (edge-free contributors are not in the DAG).
- Unit tests for the splitter skirt-proximity guard (no boundary inside a strong
  line's leakage reach) and the edge-free attachment/serialization round-trip.

## Open questions

- **O1 — amplitude-read method.** LSQ-of-template-on-core vs primary-fit
  parameters vs core-bin phasor. The 655 lesson rules out global single-bin
  phasor; quantify the targeted variants on the w287-class set.
- **O2 — double-counting with the leakage-wing baseline.** Does an edge-free
  skirt subtracted before the fit make the baseline fire less (good) or fight it
  (bad)? Sequence and A/B against the baseline-on pipeline.
- **O3 — neighbour selection / cap.** How many neighbours, by what magnitude /
  count cap, to stay targeted (the 0.1σ threshold saturated at SNR 30k on 655 →
  needs a dominance cap).
- **O4 — subtract vs merge boundary.** When is the line separable by subtraction
  vs when must the windows merge? A separation-in-resolution-elements criterion,
  reusing the #13 sub-resolution machinery.

## Tasks

1. `FixedContributor` edge-free representation + serialization round-trip (+ test).
2. Edge-free attachment in `build_window_plan`; keep the contributor through the
   cycle-breaker; targeting cap (O3).
3. Robust amplitude read for the frozen parameters (O1); A/B vs the prototype.
4. Splitter skirt-proximity guard (+ unit test).
5. Per-window + cross-fixture A/B harness run; baseline-interaction A/B (O2);
   dense-forest no-regression gate.
6. Cross-interface consistency test; ROADMAP / STATUS update; rewrite this doc as
   an implementation overview on completion.
