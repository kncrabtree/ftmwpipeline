# Stage 6 reports — corrected frequencies and the σ_f uncertainty budget

Status: **planning.** The favored fork after the Stage 6 review surface
(ROADMAP Priorities track 2). Builds on the final-products contract in
[`stage6-finalization.md`](stage6-finalization.md) §F and the timebase ε
measurement in [`instrument-clock-declaration.md`](instrument-clock-declaration.md)
§"Timebase scale correction". Acceptance test: the 1512 uncertainty-accuracy
goal (pull ≈ unit normal).

## The uncertainty problem — diagnosed and settled

The persisted σ_f (NLS) was ~60–77× too optimistic versus the vinyl-cyanide
catalog (1512 |Δf| median 16 kHz vs reported σ_f 0.2 kHz). The investigation
ruled out the obvious culprits and found the real cause:

- **Not a covariance bug.** `fit_window` inverts the *full joint* noise-weighted
  `JᵀJ` (all peaks + shared τ + baseline columns), so each per-line σ already
  marginalizes the off-diagonal correlations. The NLS σ_f is an honest
  *precision* (Cramér–Rao) estimate.
- **Not χ²-scaling.** Inflating by √χ²ᵣ closes only a fraction; the catalog
  residual is **SNR-independent** (~9–15 kHz flat from SNR 90 → 5900 while σ_f
  falls 0.33 → 0.06 kHz), so it is not a statistical/model-misfit error.
- **The dominant error is a systematic frequency bias from the unlocked
  digitizer sample clock** (ε ≈ 2 ppm). It scales the baseband, so for the
  lower sideband `Δf = ε·(probe − f_mol) = ε·f_baseband` — simultaneously a
  ~20 kHz offset and a ~2–3 kHz/GHz tilt. A single ε per acquisition collapses
  both: catalog regression gives ε = 2.28 ppm (1512) / 2.20 ppm (655), and the
  **clock spurs recover the same ε prior-free** (`calibrate_timebase`: 2.0–2.3
  ppm, σ 0.08–0.12; self-cal demonstrated in the clock-declaration doc).
  Applying it drops the catalog residual **21 → 6 kHz (1512) / 20 → 4 kHz
  (655)**, using no catalog.

So σ_f is "overly optimistic" because the reported number is *precision* while
the frequencies carry an *uncorrected systematic* the fit cannot see. The fix is
two coupled pieces — correct the bias, then budget the residual.

## A. Corrected frequencies (apply the persisted ε)

`calibrate_timebase` already measures and persists ε; reports **consume** it
(applying was deliberately deferred to here). Correction in the baseband frame:
`f_mol = probe − (probe − f_mol_measured)/(1 + ε)` (lower sideband; sideband-sign
generalized). The reported frequency is the corrected one; the raw fitted
frequency stays available as per-window drill-down.

**Calibration state** (per finalization §F, three states, reported never gated):
this instrument is all-clocks-Rb-locked except the digitizer → the
digitizer-self-cal case → ε applied; the residual flat offset (`δ_down`) is *not*
asserted by default (see §B — σ_floor ships at 0). Rb-locked absolute case →
ε ≡ 0. Free-running with no self-cal → flagged uncalibrated.

## B. The σ_f budget — what we ship, and what the user owns

**Design decision (settled).** We report only the uncertainty we can defend from
the measurement in hand, and we do **not** bake in a systematic we cannot
independently determine. The shipped budget is three terms:

`σ_f = sqrt(σ_stat² + (σ_ε · f_baseband)² + σ_floor²)`

1. **σ_stat** — the NLS diagonal (`frequency_error`), already persisted. Honest
   precision; dominant only at low SNR (≲ 30), exactly where it should be. Tracks
   the Cramér–Rao bound essentially perfectly: σ_stat vs `linewidth/√SNR` has
   Spearman ρ = +0.96 on isolated lines, so the statistical/precision physics the
   user expected is already *in* σ_stat — no separate term is needed for it.
2. **σ_ε · f_baseband** — the timebase ε uncertainty propagated through
   `Δf = ε·f_baseband`. σ_ε is the `calibrate_timebase` `sigma_epsilon`
   (~0.06–0.12 ppm) → ~1–1.5 kHz at 14 GHz baseband, falling to ~0.1 kHz near the
   probe. The *timebase parameter's* uncertainty, inherited by every line via the
   shared ε (not the line-fit covariance). Per-line (scales with baseband =
   probe − f_mol).
3. **σ_floor — a user-settable systematic floor, shipped default `0`.** Added in
   quadrature with the two measured terms. Users are advised to assess their own
   systematics (absolute frequency calibration, beam Doppler, lineshape
   reproducibility) and set σ_floor to fold them in. We ship `0` deliberately: the
   dominant unmodeled term is the per-acquisition absolute-frequency offset
   `δ_down` (below), which is **not independently determinable at the few-kHz
   level** with the instrumentation in hand, so a nonzero default would assert an
   accuracy we cannot stand behind. The reported σ_f is therefore an honest
   *precision*; the *accuracy* floor is the user's to declare.

### Why `δ_down` is not in the default (the systematic we can't measure)

There is a real per-acquisition, frequency-flat absolute offset of ~4–12 kHz
(`δ_down`) — additive (an absolute downconversion-LO / synthesizer reference
error), so the ε correction (a multiplicative baseband scale, zero at the probe)
cannot touch it, and it varies acquisition-to-acquisition. The evidence below
characterizes it and shows it cannot be recovered prior-free, which is exactly
why we neither correct it nor assert a value for it.

#### Catalog-free evidence (the measuring stick can't be trusted at the kHz level)

The catalog uncertainties (~0.35 kHz) are *model* precision — a global
Hamiltonian fit's parameter covariance projected onto each transition — not the
accuracy of any single measurement that fed the fit. Cavity-FTMW inputs are
typically eyeballed centers on ~10 kHz lines with ~4 kHz assigned. So the ~5 kHz
residual measured *against the catalog* conflates the catalog's own accuracy with
our errors and cannot, by itself, separate them.

The clean lever is **run-to-run comparison of the same molecule**, which needs no
ruler. Two vinyl-cyanide acquisitions (1512, 655) and two MTBE acquisitions
(360, 363) each measure the same physical lines twice, independently
ε-corrected; the difference of corrected frequencies is measuring-stick-free:

| pair | random reproducibility (offset-removed, high SNR) | constant offset | offset structure |
|---|---|---|---|
| 1512 ↔ 655 (vinyl cyanide) | ~2.6 kHz (SNR>300, n=3) | **+4.2 kHz** | flat (R²≈0) |
| 360 ↔ 363 (MTBE) | ~2.7 kHz (SNR>100, n=26) | **−12.2 kHz** | flat (slope +0.1 kHz/GHz, R²=0.01) |

This splits the apparent floor into the two terms above:

- **Random reproducibility shrinks with SNR** to ~2.6–2.7 kHz, agreeing across
  two independent pairs, and both fall *below* the ~5 kHz catalog residual. Most
  of the apparent catalog floor is the catalog's accuracy plus the constant
  offset; our measurement is better than the catalog can verify. (Rough
  quadrature: `5 ≈ √(catalog² + 4.2² + 2.7²)` ⇒ catalog accuracy ≈ 3 kHz —
  consistent with eyeballed cavity centers.)
- **A constant per-acquisition offset**, frequency-flat: no f_mol tilt ⇒ **not
  Doppler** (a fixed-velocity beam Doppler is `Δf=(v/c)·f_mol`, which must rise
  with f_mol); no baseband tilt ⇒ **not residual ε**. Both MTBE acquisitions are
  collision-free post-expansion (the temperature difference is backing pressure
  only, not a collisional pressure shift), so the MTBE −12 kHz is **instrumental**,
  not physical — confirming the offset is a genuine per-acquisition
  absolute-reference error, and giving a *clean instrumental pair* for the lattice
  test below.

#### The clock lattice cannot pin the absolute offset (tested)

ε is a multiplicative baseband scale; the gated spurs measure it cleanly. The
absolute offset `δ_down` is additive (the LO reference), so only a spur that
*passes through the downconversion mixer* (RF-native) can witness it. Two tests,
both fail:

- **Generic lattice intercept** — refit `df = a + ε·f_bb` with a free intercept.
  The per-fixture intercepts are nonzero (a stable low-baseband artefact) but
  their *differences* do not track the molecular offset (sign-wrong on 360/363,
  consistent-with-zero on 1512/655).
- **A specific RF-native spur, `6400 = probe − 3×upconvLO(11520)`.** Its
  deviation from the ε line is **sign-correct on both pairs** (δ_down does leak
  through) — but quantitatively wrong, for two compounding reasons, either fatal:
  1. **Upconv contamination.** It is the *3rd harmonic of the upconversion LO*
     downconverted by the downconversion LO, so it carries `δ_down − 3·δ_up`,
     not `δ_down`. A molecule never sees the upconversion chain (it emits at its
     own rest frequency, downconverted only), so the spur and the lines reference
     different δ's. Multi-spur 2-LO solves (adding `17920 = probe−2×upconv`,
     `12160 = probe−5×5760`) do not rescue it.
  2. **Comb coincidence** (the decisive objection). `6400 = 5×1280` and the
     downconversion LO is `40960 = 32×1280`, so 6400 sits on the **δ-free**
     downconv-derived 1280-comb. The measured tone is a power-weighted blend of
     the RF-native (δ-bearing) tooth and the comb (δ-free) tooth, which dilutes
     the deviation toward zero. The 2-LO reconstruction under-predicts even the
     clean instrumental pair (+0.5 vs +4.2 kHz) — the dilution signature. Every
     candidate spur (17920, 12160) lands on the same comb.

So **`δ_down` is irreducible from instrument self-calibration**: ε (scale) is
self-calibratable; the absolute offset is not, with this clock comb.

### How σ_floor works, and correcting `δ_down` (out of default scope)

- **σ_floor is a single user-settable field, shipped default `0`,** added in
  quadrature with the measured terms. A user who knows their instrument's
  accuracy floor (or has a trusted reference to calibrate it, via the pull below)
  sets it; we ship no value because we cannot determine the dominant systematic
  ourselves.
- **Correcting `δ_down`** (rather than widening σ_f for it) is possible *only*
  with an **in-spectrum absolute reference** measured per acquisition — a trusted
  catalog line, a live-injected Rb-locked tone, or a frequency-comb tooth. Absent
  that, `δ_down` is neither corrected nor budgeted by default; it is documented
  for the user. (Self-cal cannot recover it: §"clock lattice cannot pin it"; a
  bench frequency-counter check of the downconversion-LO synthesizer can confirm
  whether the flat offset is the LO at all vs a line-center estimation bias, but
  is an instrument diagnostic, not part of the pipeline.)

**Pull as a user calibration tool (not a shipped acceptance gate).** Pull
`= (f_reported − f_reference)/σ_f` ≈ unit normal is how a *user* with a trusted
reference calibrates their own σ_floor and assesses their systematics. With the
shipped default σ_floor = 0 the pull reads over-confident against a catalog —
expected and correct, because we report precision, not the user's accuracy floor.
The cross-fixture harness computes `ground_truth.pull`; keep it as the
calibration/diagnostic surface, reading it knowing a model-based catalog
contributes ~3 kHz of its own accuracy.

### Remaining tests (deferred) — characterizing `δ_down` (not blocking)

These would *characterize* `δ_down` further but are not required to ship (the
default already declines to assert it):

- **Cross-instrument comparison** — measure the same molecule on a second
  spectrometer and difference; the shared catalog cancels.
- **Bench tests on the home instrument** — counter the downconversion-LO
  (Valon 5009 → 5120 ×8 = 40960) output against the Rb reference to test the LO
  hypothesis directly; optionally a one-off locked-scope vinyl-cyanide run (ε≡0)
  as a third reference acquisition. A *live*-injected Rb-locked tone is the only
  one of these that could become a per-acquisition correction.
- **Locked digitizer (rejected as an operating mode)** — would set ε≡0 but the
  scope-lock test runs show massive `6250/4 = 1562.5` MHz harmonics across
  ~25 GHz (~30 µV); the unlocked clock + ε self-cal is the better operating point.
- **Leakage-pedestal asymmetry as a `δ_down` contributor (TESTED — not the
  driver).** A candidate mechanism for part of the per-acquisition offset: an
  asymmetric leakage pedestal under a line biases its fitted centre. The bias is
  set by *pedestal severity* = (noise floor) × (neighbouring strong-line leakage
  structure), so it varies acquisition-to-acquisition. Shot counts confirm the
  inputs differ: vinyl-cyanide 1512 = 17,860 vs 655 = 2,133,080 shots (≈119× →
  ≈11× noise); MTBE 360 = 775,860 vs 363 = 525,660 (≈1.5× → ≈1.2× noise). The
  pair offsets do **not** scale with the noise mismatch alone (the ≈11× VC
  mismatch gives the *smaller* +4.2 kHz; the noise-matched MTBE pair the *larger*
  −12.2 kHz) — but that is **not** a clean refutation, because the MTBE pair is at
  different rotational temperatures: cold 360 has a few much stronger low-J lines
  (more severe pedestals) while warm 363 spreads intensity over many weaker
  lines, so pedestal severity is *not* matched even though the noise is.

  **Test RUN (2026-06-16, `scratch/pedestal-test/`).** On the 7–10 isolated VC
  lines common to 1512 (noisy) and 655 (quiet) with SNR > 50 in both, each
  line's centre was measured under several pedestal-handling schemes and the
  inter-acquisition offset (1512 − 655, ε-corrected) compared. **Three
  independent pedestal *suppressors* converge** on the same offset: production
  per-window fit (adaptive leakage-wing baseline) **+4.4…+6.3 kHz**, an explicit
  polynomial-baseline refit **+4.7…+6.9 kHz**, and a Hann-apodized magnitude
  centroid **+6.0…+6.7 kHz (std ≤ 3.6, stable across window half-widths
  0.35–0.6 MHz)**. The pedestal-*exposed* estimators (no-baseline fit; boxcar
  centroid) are wildly unstable (per-line jumps 30–75 kHz; inter-acq std
  10–51 kHz) — confirming pedestals *do* bias raw centres, but they are not a
  clean estimator. **Conclusions:** (1) `δ_down` does **not** collapse under
  pedestal suppression — it survives at ~+5–7 kHz across three independent
  suppressors, so it is a genuine per-acquisition absolute-reference offset, not
  a pedestal artifact (confirms the §B reading). (2) The pedestal *bias itself*
  (Hann−boxcar centre shift) is consistently **larger in the dense 655**
  (median |20–32| kHz) than the noisy 1512 (|16–21| kHz) → driven by
  **neighbour-line leakage density, not absolute noise level** — supporting the
  temperature/line-strength framing and arguing *against* the original
  "noise-level-dependent" hypothesis. (3) The production per-window fit already
  agrees with the apodized (pedestal-suppressed) centre, so production `δ_down`
  is the post-mitigation residual. Caveats: n = 7–10 (1512 is shot-starved →
  few high-SNR isolated lines), ε applied as the §B catalog constants (2.28 /
  2.20 ppm), centroid is a crude estimator — the strength is the cross-method
  *agreement*, not any single number.

## C. The report feature (presentation over the persisted record)

Reports **render** the persisted record; they never recompute the fit. The
design (settled 2026-06-16) is a top-level **`report`** stage object, with three
content **levels** that all render one in-memory *report model* assembled from
the `.ftmw` — **assemble once, render many**. The report model is built from the
persisted Stage 6 `FinalProducts` table (§A/§F: corrected frequency + the
broken-out σ_f budget + amplitude/phase/SNR + per-peak origin), the per-stage
parameters and key results (Stages 0–5), the per-window audit/ledger/covariance
detail, the spur provenance (gated catalogue + ADC-image/clock identities), and
an optional catalog-match block. Generating any level **requires a built
`FinalProducts`** (i.e. `review run` has consolidated it); reports do not run the
fit. Catalog match is an optional `--catalog <file>` input to every level.

A content **level** largely constrains its **format**: the levels differ in
depth, and the medium follows. The pipeline emits an *unassigned* measured line
list, so spectroscopic fitting-software line-list formats (Pickett `.lin` /
SPFIT) are **out of scope** — they key on quantum-number assignments the pipeline
deliberately does not produce. Report formats are therefore presentation/data
formats only.

### Level 1 — `report table` (data export)

The `FinalProducts` table serialized for downstream use: **CSV** (with a
commented provenance header — calibration state, ε ± σ_ε, σ_floor, probe,
sideband, fixture id — so the file is self-describing as paper SI), a **JSON**
twin for programmatic consumption, and a **LaTeX table** (`booktabs`-style) for
direct inclusion in a paper's SI. Columns: calibrated frequency, σ_f and its
three components, raw frequency, baseband, amplitude, phase, SNR, origin,
window_id, clock-lattice flag. This is the cheapest level — `FinalProducts` is
already built and persisted (build-order step 1), so L1 is a serializer over it.

### Level 2 — `report summary` (methods + results document)

A **Markdown** report (the source-of-truth format; html/pdf are thin
conversions, not separate hand-built renderers) interleaving **static,
code-versioned algorithm prose** — a "methods section" that lives in the report
module so it stays in sync with the code, *not* pulled from the dev planning
docs — with per-experiment numbers from each stage (start time, FT band, noise σ,
τ, peak/window counts, χ² summary, calibration state). The full peak table is
emitted as the companion L1 CSV rather than dumped inline; a *summary* (counts,
band, χ² stats, strongest lines) appears inline, with `--include-table` to inline
the full table.

### Level 3 — `report full` (per-window document tree)

A **local linked HTML site** (dependency-light: hand-rolled HTML + matplotlib
PNGs, no new heavy dependency; PDF deferred): an index (summary + final table +
window links) → per-window pages, each with the figure from the existing
`fit show --window N` renderer, the audit/thaw/rescue history, raw + calibrated
frequencies, all window fit parameters + covariance, and the ledger candidates.
L3 is an **assembler** over existing artifacts + the visualization renderers, not
new analysis. With ~10³ windows it needs a manifest/index and a `--windows`
filter (all vs attention-only).

### σ_floor is persisted and reported (requirement, not optional)

The σ_floor the user declares **must be persisted in the `.ftmw` file** and
surfaced explicitly in the report — it is the user's accuracy declaration, part
of the provenance, so any reported σ_f is always reproducible from the record
alone (never dependent on a transient CLI flag or an out-of-file default). The
report shows σ_floor as its own line alongside the broken-out σ_f components, so
a reader sees whether the accuracy floor was declared (`> 0`) or left at the
shipped `0`.

Resolved (implemented): σ_floor is homed in a **file-level `/frequency_calibration`
provenance group** (option (a) — an instrument/experiment property, not a
per-window fit output, pairing with the ε / calibration-state record), with one
field `sigma_floor_khz` (default 0). It is reproducible from the record alone and
surfaced explicitly in the report.

## Build order

**Step 1 — persisted final-products contract (done).** Apply ε and the three-term
σ_f budget into the persisted `FinalProducts` table reports render
(`stage6_review` group), with **σ_floor default 0** homed file-level and folded in
quadrature; calibration state derived (rb_locked / self_calibrated /
uncalibrated). Built by `review run` (`--sigma-floor` sets the floor); report
precision, do not bake in `δ_down`. Dual-interface + tests landed.

**Step 2 — the cross-fixture / pull calibration surface.** Promote the
cross-fixture validation harness into the pull calibration/diagnostic surface
(how a user with a trusted reference sets their own σ_floor), keeping the
run-to-run (catalog-free) reproducibility path. Light, since the harness already
computes `ground_truth.pull`.

**Step 3 — the `report` feature (three levels, §C).** Top-level `report` object,
assemble-once / render-many over the persisted record. Sequence **L1 → L2 → L3**:
- **L1 `report table` — done.** `report table <file> --format {csv,json,latex}
  [--output PATH]` serializes the persisted `FinalProducts` to a CSV (commented
  provenance header), JSON, or a LaTeX `booktabs` table (state-aware caption);
  stdout by default. Dual-interface (`Pipeline.report_table` / `api.report_table`
  / `_internal/report_impl.py`); cross-interface + serializer tests landed.
- **L2 `report summary` — done.** `report summary <file> [--output PATH]
  [--include-table]` renders a Markdown methods + results document: static,
  code-versioned per-stage algorithm prose (a methods section living in the
  report module, not pulled from these planning docs) interleaved with the
  per-experiment numbers read from each persisted stage (start time, FT band +
  bin spacing, noise σ_x, τ_maj ± σ_τ and per-band τ, peak counts by class,
  planned/fit window counts, χ²ᵣ summary, thaw/replan/rescue tallies, ε ± σ_ε,
  σ_floor, calibration state). The full line list is the companion L1 export by
  default; `--include-table` inlines it. A strongest-lines table (top 10 by SNR)
  always appears inline. Assembled once from the persisted record, never
  recomputed. Dual-interface (`Pipeline.report_summary` / `api.report_summary` /
  `_internal/report_impl.py`); cross-interface + renderer tests landed.
  Each fitting stage (2–6) additionally carries: a **parameters table** (the
  knobs the stage ran with), the governing **equation(s)**, **detailed result
  tables** (per-band noise σ_x and noise-fraction; per-band τ + bimodality;
  Stage 3 peak-strength SNR percentiles + per-band candidate/promoted density;
  Stage 4 window-width and peaks-per-window percentiles; Stage 5 χ²ᵣ / shape-error
  ε% / σ_stat percentiles + a χ²ᵣ-and-gate breakdown binned by window brightness;
  Stage 6 the σ_f budget breakdown), and an **auto-flagged Concerns block** with
  recommendations (low noise fraction, bimodal τ, τ-twin/shape-mismatch fallback,
  capped/non-converged windows, SNR-aware-gate failures, uncalibrated state,
  σ_floor = 0). Read-only (the one replay is the deterministic active-FT rebuild
  for the per-band noise table). Markdown only; no emoji (Concerns use text tags).
  Possible follow-on statistic (deferred): a σ_f **distribution** percentile table
  in Stage 6 (parallels the Stage 5 σ_stat table; currently only medians).
- **L3 `report full`** (dependency-light linked HTML per-window site reusing the
  `fit show` figure renderer) — next.
Catalog match an optional `--catalog` cross-reference input to each (the
fast-follow after core L1, sharing the tolerance helper with Step 2's pull
surface): a geometric frequency-proximity annotation against a user catalog
(match within `N·√(σ_f² + σ_cat²)`, opaque-label echo), **never** assignment —
the pipeline emits unassigned lines, so no quantum-number logic.

Deferred / longer horizon: PDF export of L2/L3 (pandoc/weasyprint, only if
wanted); the frequency-calibration / σ_f **research report**
([`../research/frequency-calibration-uncertainty/PLAN.md`](../research/frequency-calibration-uncertainty/PLAN.md));
residual-tail characterization; the cross-instrument and bench tests that would
*characterize* `δ_down` (not blocking — the default declines to assert it); a
live-injected absolute reference as the only path to a per-acquisition `δ_down`
correction.
