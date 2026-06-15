# Plan: Stage 4 — Window assignment

Status: **implemented (finalized).** All task-breakdown items landed; the
edge-statistic truncation-leakage defect found on 2638 was resolved by the D8
de-ramp rework (see [`leakage-detection-rework.md`](leakage-detection-rework.md)),
and this document describes Stage 4 as it now stands. Scope is Stage 4 only —
*classify and propose*, do not fit (fitting is Stage 5). Registered in
[`../ROADMAP.md`](../ROADMAP.md).

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the Stage 4 work it tracks. It builds directly on the
finalized Stage 3 → Stage 4 contract in
[`stage3-peak-detection.md`](stage3-peak-detection.md). The complex-edge
coherence test that anchors the algorithm's edge decision is derived,
calibrated, and verified in the research report
[`../research/complex-edge-coherence/report.md`](../research/complex-edge-coherence/report.md);
that report's locked statistic, band width, threshold, and σ source are
the parameters this stage uses.

## Objective

Turn the promoted Stage 3 peak list into a **fit plan**: a set of analysis
windows over the persisted user spectrum, each annotated with the peaks to fit
freely, the strong out-of-band lines whose leakage must be carried as a fixed
background, a fit dependency order, and a difficulty classification so Stage 5
can parallelize the easy windows and spend its budget on the hard ones. Stage 4
is purely structural: it makes no fits and changes no spectrum.

## Reuse map

Per the locked Stages 3–5 design, the refined `newfitting/` engine — which
held the *adaptive window selection* and *peak aggregation* — is permanently
lost. Stage 4 is therefore **recreate**, not port. The surviving
`~/github/bcfitting/src/bcfitting/ftmwfitting.py` contributes only philosophy
(predict extent from the strongest line, greedily expand while leakage is still
above baseline, sanity-check edges) and the analytic finite-T sinc-leakage
*model* (reused as the leakage-reach predictor, already ported in Stage 3 as
`preprocessing/leakage.py:estimate_leakage_reach`). No `newfitting` contract is
available to recreate against; the contract is defined here.

## The invariant (decided)

Two distinct notions, do not conflate them:

- **Fit window** — the contiguous frequency span whose points enter *that
  window's* least-squares residual. Fit windows are **disjoint and cover each
  spectrum point at most once**. This is the hard invariant (it guarantees no
  point's residual is double-counted and no peak is fit twice).
- **Contributor set** — the peaks whose model terms are *evaluated* over a
  window: **in-band/free** (the window's own peaks, free A/f/φ in Stage 5) and
  **out-of-band/fixed** (strong lines fit in *their* own window, parameters
  frozen, contributing only their leakage skirt). Contributor sets overlap
  across windows by design; that is how a strong line's leakage is represented
  everywhere without re-fitting it.

Guardrail: a fixed contributor is the strong line's **finite-T damped-cosine
term with frozen parameters**, evaluated in the dependent window's model
(consistent with the locked Stage 5 LS contract: residual = complex FFT(model)
vs complex FFT(window)). It is **not** analytic-sinc subtraction from the data
— the discarded `iterative_peak_subtraction` path is not revived.

Consequence: Stage 4's output is **not a flat partition**. It is an ordered set
of fit windows, each carrying `(free peaks, fixed out-of-band contributors)`
and a position in a fit **dependency DAG** (a window depends on the windows
that fit its fixed contributors). Stage 4 *classifies and proposes*; Stage 5
fits and **may revise** the separation — for strongly coupled regions the ideal
split may only be determinable at fit time.

## The edge-coherence test (locked)

The algorithm's load-bearing decision — does the proposed window edge
sit in clean noise, or is it still carrying a coherent leakage skirt? —
uses the complex-edge coherence statistic derived in
[the research report](../research/complex-edge-coherence/report.md):

$$
S_{\text{coh}}(z;\,\sigma) \;=\; \frac{\bigl|\sum_{k=1}^{M} z_k\bigr|}{\sigma\,\sqrt{M}}
$$

evaluated over an M-point edge band of complex spectrum values $z_k$
with per-bin complex noise RMS $\sigma$. Locked parameters:

- **Statistic:** `S_coh` (primary). The max-cumsum variant `S_cum` is
  used only to locate the trim point inside an above-threshold band.
- **Band width:** $M = 64$ for the rolling first-pass scan;
  $M \in [16, 32]$ for trim-point refinement after a flag.
- **Threshold:** `T_edge = 8` ($= \sqrt{M}$; the D8 recalibration —
  fires on coherent leakage of $\gtrsim 1\sigma$ per bin, since
  $T_{\text{edge}}/\sqrt{M}$ is the per-bin leakage in $\sigma$. See
  [`leakage-detection-rework.md`](leakage-detection-rework.md). The
  research report's original `3` flags sub-noise 0.38σ leakage — it is
  still null-safe (per-band false positives < 1%) but reads ~57% of
  2638 leakage-touched.
- **σ source:** the per-point `rms_noise[k]` array from the upstream
  noise stage (the high-pass scatter estimator; see the
  [noise SNR-scaling report](../research/noise-snr-scaling/report.md)).
  σ varies up to 6× across 2638 — use the *local* value (window mean
  of `rms_noise`), not a global median.
- **Threshold/M tunables:** both are configurable parameters on the
  pipeline file; the values above are the defaults and the
  empirically-validated operating point.

## Algorithm

Inputs: the promoted peaks (`Peak.properties['promoted']`) on the persisted
user spectrum; the canonical Stage 2 noise (per-point RMS on that grid); the
acquisition geometry (`start_us`, probe frequency) for the de-ramp; and the
**complex** user spectrum (real and imaginary parts, not just magnitude — the
edge test is a complex-domain coherence test and magnitude discards the
phase information it depends on).

1. **De-ramp + leakage-touched map.** Reference the complex spectrum to the
   active-region turn-on (`deramp_to_active_start`, D8 — see
   [`leakage-detection-rework.md`](leakage-detection-rework.md)), then roll
   `S_coh` at $M = 64$ and threshold it at $T_{\text{edge}} = 8$ into the
   contiguous **leakage-touched intervals**. That map drives strong-cluster
   grouping (step 3) and fixed-contributor attachment (step 6).
2. **Propose extents.** Each promoted peak proposes a *tight* window — its
   core plus `min_window_half_width_mhz`, **not** the leakage-touched run (a
   strong line's run is ~80–100 MHz wide; its distant leakage is carried into
   other windows as a fixed contributor, step 6). The max-cumsum variant
   `S_cum` locates the precise edge of a coherent stretch inside a flagged
   interval at a narrower band ($M \in [16, 32]$) when needed. (DC-offset-at-
   edges is the zero-distance degenerate case of the same test; the research
   report confirms no genuine flat pedestal is present on 2638.)
3. **Strong clusters.** Strong lines sharing one leakage-touched run
   (the rolling statistic stays above $T_{\text{edge}}$ between them)
   form a single **primary joint window** — they must be fit together;
   none can be a fixed background for the others. The 2638 fixture's
   36350/36389 pair (SNR 186 + 55, 39 MHz apart) is the canonical
   coupled-strong-line case: under the MAD/median Stage 2 σ the
   inter-line skirt is no longer absorbed into σ, so S_coh stays above
   $T_{\text{edge}} = 8$ throughout the gap and the pair merges into a
   single ~65 MHz joint window (`needs_joint_treatment`). See
   [`leakage-detection-rework.md`](leakage-detection-rework.md) and the
   audit at
   [`../research/stage4-poststage23-audit/report.md`](../research/stage4-poststage23-audit/report.md).
4. **Merge to fixpoint.** Overlapping proposed fit windows merge transitively
   in a deterministic order (by frequency, then descending strength), until
   stable → disjoint fit windows covering each point ≤ 1. The merge is a
   deterministic linear functional of the spectrum: same input → same
   partition, no randomness.
5. **Assign in-band peaks, pruning leakage artifacts.** A promoted peak is a
   free peak of the unique fit window containing its frequency **only if it is
   not attributable to a contributor's leakage**. Stage 3's gap pass is masked
   by the de-ramped leakage-touched map (D8), so strong-line skirts no longer
   leak through wholesale — but a few sidelobes can still survive in
   sub-threshold dips of that map (see
   [`stage3-peak-detection.md`](stage3-peak-detection.md) and
   [`../research/peak-detection/report.md`](../research/peak-detection/report.md) §8).
   Once that line is a contributor (free in-band or fixed) its leakage
   explains those detections, so they must not also be fit as independent
   lines. Pruning uses the analytic leakage envelope of the window's strong
   contributor(s); the residual after the strong term is what defines genuine
   free peaks.
6. **Attach fixed contributors.** For each window, a freeze-eligible strong
   line in-band of a *different* window is attached as a fixed contributor
   (reference to its primary window) when that line's leakage-touched interval
   (step 1) overlaps this window. The peak list answers "from which line?" —
   Stage 3's promoted strong lines give the candidate; the de-ramped
   leakage-touched map confirms that a coherent skirt actually arrives here.
   This adds a dependency edge.
7. **Classify difficulty (strong-line-driven, empirical).** Difficulty is
   *not* a promoted-peak-count threshold — that count can be inflated by
   residual leakage artifacts (gap-pass sidelobes that survive the de-ramped
   leakage mask) and in any case conflates a dense-but-easy region with a
   coupled-and-hard one, so it is an unreliable difficulty signal. A window is **hard** if it
   contains or is materially influenced by a strong line (has strong in-band
   peaks, or unresolved fixed contributors, or fails the edge-coherence test
   on its edges), or exceeds the width cap, or sits in a comparably-strong
   coupled cluster with no dominant line to freeze. Everything else is
   **easy/independent**. The only count-like cap is *width*; "too many peaks"
   is replaced by "contains/near a strong line".

   The width cap default uses the **contiguous-above-threshold extent**
   of `S_coh` as the natural scale: on 2638 the strong-line skirt
   extends to about ±20 MHz at $T_{\text{edge}} = 8$, so a single window
   ≳ 40 MHz wide is already in "dense / strongly-coupled" territory.
   `max_window_width_mhz` is a configurable parameter with that
   empirical scale as the default basis.

   For hard windows Stage 4 emits a *proposed* split (at a complex-edge-clean
   interior point, flagged as an approximation that knowingly cuts shared
   leakage) **and/or** a `needs_joint_treatment` marker. Stage 4 does not
   choose; it annotates.
8. **Emit the plan.** Topologically order the dependency DAG; independent
   windows form parallel batches. Each window carries: freq range, free peaks,
   fixed contributors (peak id + primary window id), difficulty class, batch
   id, and diagnostics (predicted vs trimmed extent, cap hits, split proposal).

## Data structures

The existing `core.data_structures.SpectralWindow` is close but insufficient:
it has `freq_range`/`peaks` but no free-vs-fixed contributor split, no
dependency edges, no difficulty class, no parallel-batch grouping. Stage 4
needs either an extended `SpectralWindow` or a new `WindowPlan` aggregate:

- per window: `window_id`, `freq_range`, `free_peaks` (refs into the Stage 3
  list), `fixed_contributors` (peak id + owning `window_id`), `difficulty`
  (`easy` | `hard`), `batch` (parallel group), `split_proposal`
  (optional), diagnostics (predicted vs trimmed extent in MHz, edge
  statistic values at the chosen trim points, width-cap hit flag).
- plan-level: the dependency DAG / topological order; the parallel batching.

`FittedPeak` already exists for Stage 5; fixed-contributor parameters become
known only after the contributor's primary window is fit (hence the ordering).

## Interface surface (dual-interface rule)

Logic in `_internal/stage4_impl.py`; thin identical wrappers:
`Pipeline.assign_windows()` / `api.assign_windows()` / CLI `windows run`,
plus `windows show` and `load_windows()`. Consumes the **promoted** peaks
only. Parameters (documented defaults, configurable on the file): the
edge-test M and threshold above (`edge_M = 64`, `edge_threshold = 8.0`,
`trim_M = 32`), `max_window_width_mhz` (default ≈ 40 on 2638-class
experiments), `min_freeze_snr` (freeze-eligibility cutoff — see O4-2),
assumed `τ` for reach prediction. These knobs are resolved through
`core/window_planning_settings.py` (`WindowPlanningSettings`) on the same
four-layer chain as the other stages (commit acb900a; see
[`settings-backfill.md`](settings-backfill.md)); the Gaussian-path defaults
audit kept `leakage.tau_us` on the boxcar value (commit 5a72c04). Stage
tracking: add `stage4_windows` to `PipelineStageTracker.STAGE_DEPENDENCIES`
(depends on `stage3_peaks`) and `STAGE_DATA_PATHS`; it is then automatically
invalidated by the existing canonical-settings-change mechanism.

## Serialization

The window plan is curation/coordination substrate (like the peak list), not a
heavy derived array, so it is **persisted** under `/stage4_windows` (flat,
hand-editable, loud validation), not recomputed on demand — consistent with the
SERIALIZATION spec's treatment of peaks. Round-trip + hand-edit contract as in
peak serialization, per `SERIALIZATION_STRATEGY.md`.

## Design questions (resolved)

The complex-edge prototype resolved the originally-listed research items
(statistic + M + threshold, σ source, shape vs peak-list classification,
strong-cluster grouping criterion, determinism, renegotiation-frequency
expectation); they are folded into the algorithm text above, with the full
derivation in the research report. Two design questions were settled at
implementation:

- **O4-2 Freeze-eligibility + error-propagation guard.** The empirical
  "does this window have a fixed contributor" question is answered by the
  edge-coherence test; the *parameter* side — which strong lines are stable
  enough to freeze without their uncertainty contaminating dependent windows —
  is governed by `min_freeze_snr` (default 50): a fixed contributor below it is
  flagged `freeze_eligible=False` for the Stage 5 thaw-and-re-fit handshake (the
  thaw protocol itself is Stage 5 work).
- **O4-6 Persist vs recompute.** The window plan is persisted (see
  *Serialization*), consistent with the serialization spec's treatment of the
  peak list.

## Renegotiation handshake with Stage 5

Stage 5 may find the proposed separation inadequate when coupling is
only visible at fit time. The handshake: Stage 5 emits a re-plan
request (merge two adjacent windows, or split a hard window at a
specified interior point), and Stage 4 exposes a re-plan entry point
that updates the persisted plan in place. Stage 4 does *not*
over-provision hard windows up front; the prototype confirmed the
statistic-trim resolution is roughly $M \cdot \Delta f \approx 1.5$ MHz
at $M = 64$ on 2638, so renegotiation is expected to fire on
coupling-only-visible-at-fit-time phenomena (rare), not on routine
edge-trim error (never). Lock the protocol in the Stage 5 plan; Stage
4 just needs to support the entry point.

## Test plan

- Synthetic spectra (known A/f/φ/τ) for: isolated strong line; weak line on a
  strong line's skirt (must become a fixed contributor, not free); two strong
  lines with overlapping reach (one primary joint window); a dense comparably-
  strong cluster (width cap triggers a flagged split).
- Edge-coherence statistic: regression tests against the
  research-report calibration — `S_coh` on synthetic clean edges sits
  at ~0.886 mean with p99 ≈ 2.15; OOB lines fire above $T_{\text{edge}}$
  with the predicted distance/SNR pattern; in-band centred and OOB
  decaying envelopes distinguishable by the spatial profile of the
  statistic, not by a single point estimate.
- Leakage-artifact pruning (step 5): a strong line's promoted sidelobes are
  excluded from the free set once it is a contributor; genuine nearby weak
  lines are retained.
- Invariant checks: fit windows disjoint and cover each point ≤ 1; every
  retained free peak in exactly one window's free set; dependency graph
  acyclic; topological order valid; batches independent.
- 2638 real data: sane window count and shape — the strong lines anchor
  windows, dense regions are flagged not exploded, promoted-only
  consumption. Reference numbers under the post-Stage-2/3-rework
  baseline: 391 windows at `T_edge = 8`, max width 65.67 MHz at the
  recoupled 36350/36389 joint window (`needs_joint_treatment`), all
  other widths 4–30 MHz. See step 3 and the audit at
  [`../research/stage4-poststage23-audit/report.md`](../research/stage4-poststage23-audit/report.md).
- Cross-interface identity (CLI/Pipeline/api); serialization round-trip +
  hand-edit; invalidation on Stage 1 canonical-settings change and on Stage 3
  re-detection.

## Implementation map

The stage is built from: the window-plan data structures (`WindowDifficulty`,
`FixedContributor`, `FitWindow`, `WindowPlan` in `core/data_structures.py`); the
edge-coherence statistic (`preprocessing/edge_coherence.py` —
`coherence_statistic`, `rolling_coherence`, `max_cumsum_statistic`,
`above_threshold_intervals`, calibrated against the research report);
strong-cluster grouping + merge-to-fixpoint
(`preprocessing/window_planning.py:build_window_plan`); fixed-contributor
attachment, leakage-artifact pruning, and the dependency-DAG topological/batch
ordering; strong-line-driven difficulty classification with the width-cap/split
proposal; `io/window_serialization.py` with `stage4_windows` stage tracking
wired into `file_manager.invalidate_downstream_stages` (also fired on Stage 3
re-detection); and the cross-interface wrappers
(`Pipeline.assign_windows/visualize_windows/load_windows`, `api.*`, CLI
`windows run`/`windows show`) over `visualization/window_visualization.py`.

## Implementation notes

- **Window extent is per-peak-proposed, statistic-grouped.** Each promoted
  peak proposes a *tight* window — its core plus the window margin (see "Window
  margin and the content-bounded cap split" below), **not** the leakage-touched
  run (a strong line's run is ~80–100 MHz wide;
  its distant leakage is carried elsewhere as a fixed contributor). Overlapping
  proposals merge to a fixpoint. The complex-edge coherence statistic supplies
  the *leakage-touched regions* used for strong-cluster grouping (strong lines
  sharing one touched region merge into a primary joint window) and for
  fixed-contributor attachment (a window inside a strong line's touched region
  but distinct from it gets that line frozen-in).
- **D8 — the `S_coh` de-ramp rework.** `S_coh` on the raw persisted spectrum
  cancelled on the oscillating truncation skirt and read noise-level over
  coherent leakage. Resolved by de-ramping the spectrum to the active-region
  turn-on before the statistic (`S_coh` itself was never wrong, only its
  input); `T_edge` recalibrated 3 → 8. The leakage-touched map, strong-cluster
  grouping, fixed-contributor attachment, and `edge_coherence_fail` all run on
  the de-ramped statistic. Stage 3's gap pass had a paired defect (sidelobes
  promoted as weak peaks), fixed in the same rework. Full implementation
  overview: [`leakage-detection-rework.md`](leakage-detection-rework.md).
- **O4-2 freeze-eligibility.** `min_freeze_snr` (default 50) is a parameter on
  the file; a fixed contributor below it is flagged `freeze_eligible=False`
  for the Stage 5 thaw-and-re-fit handshake. The thaw protocol itself is
  Stage 5 work.

## Window margin and the content-bounded cap split

Resolves Stage 6 review findings F2 (a single cluster bisected across two
windows, its lines piled on the inner edges) and F3 (the inert
`min_window_half_width_mhz`). Both traced to one incoherence: the per-peak proto
half-width was `max(min_window_half_width_mhz / step, edge_m)`, so `edge_m`
(the coherence band, 64) always won and the MHz knob never bound — and the
resulting 128-point proto window was *wider than the 96-point content cap*. The
cap split then perpetually re-cut content that already fit, choosing whatever
interior peak gap happened to be largest. On the 2638 33723.5–33724.6 cluster
(0.94 MHz of content) that gap was a 0.39 MHz intra-cluster notch, so the
cluster was split in half with each side leaning on the other's frozen skirt.

Three coordinated changes:

- **The cap split bounds peak *content*, not the padded span.** A window whose
  promoted peaks span ≤ the cap stays whole even when its empty proto-margins
  push the physical span over the cap. Only a genuinely over-cap *content* is
  split, at its sparsest interior gap. A content-fitting cluster is never
  bisected.
- **The window margin is a coherent points knob.**
  `min_window_half_width_points` (default 32) is the noise budget kept on each
  side of a window's outermost peak — used both as the proto half-width and as a
  post-construction **trim** that pulls each window's edges to ≤ the margin
  beyond its outermost peak. It supersedes the MHz form
  `min_window_half_width_mhz` (now the `points == 0` fallback, mirroring the
  `max_window_width_points` / `max_window_width_mhz` pair) and is decoupled from
  `edge_m`. The coherent operating range is `trim_m ≤ margin ≤
  max_window_width_points / 2`: at least `trim_m` (32) so the edge-coherence
  statistic samples the noise margin rather than a peak, and at most half the
  content cap (48) so a lone line's `2 × margin` window never exceeds the cap.
  The default 32 is the tight end — minimal noise dilution and NLS cost, ample
  for the order-≤4 leakage-wing baseline.
- **The trim removes empty pedestals and re-centres features.** A lone line's
  window is exactly `2 × margin`; a cluster's window is its content plus the
  margin each side. A noise-only gap may open between two trimmed windows — that
  is fine (disjoint coverage covers each spectrum point at most once; distant
  leakage is carried by fixed contributors, not window width). The
  difficulty/`too_wide` classifier is likewise judged on content, so it no
  longer flags (or proposes splitting) a correctly-sized padded window.

Cross-fixture re-fit (the seven issue-3 fixtures): recall up on both
ground-truth fixtures (1512 0.435→0.443, 655 0.518→0.521), `tier1_pass` up
everywhere, χ²ᵣ bulk median flat, 25–40% fewer windows (consolidation), no
mega-windows; the small fitted-line reductions are boundary-duplication
artifacts the joint fits resolve (recall rose, so not real losses).
