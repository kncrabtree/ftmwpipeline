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

## C. The report itself (presentation over the persisted record)

Per finalization §F: one headline final-products table — corrected frequency +
the σ_f budget (components broken out) + amplitude/phase/SNR + per-peak origin;
raw frequencies are per-window drill-down. Plus decision/candidate ledgers, spur
provenance (the gated catalogue + the ADC-image/clock identities), and the
catalog-match summary when a catalog is supplied. Reports **render** the
persisted record; they do not recompute the fit.

### σ_floor is persisted and reported (requirement, not optional)

The σ_floor the user declares **must be persisted in the `.ftmw` file** and
surfaced explicitly in the report — it is the user's accuracy declaration, part
of the provenance, so any reported σ_f is always reproducible from the record
alone (never dependent on a transient CLI flag or an out-of-file default). The
report shows σ_floor as its own line alongside the broken-out σ_f components, so
a reader sees whether the accuracy floor was declared (`> 0`) or left at the
shipped `0`.

Open implementation choice (decide at build time): *where* it is declared and
homed. Candidates — (a) a file-level provenance field (one value per experiment,
sitting with the calibration state / `SourceMetadata`), or (b) a Stage 6
final-products parameter persisted with the σ_f-budget record. Leaning (a): it is
an instrument/experiment property, not a per-window fit output, and pairs
naturally with the ε / calibration-state record. Whichever is chosen, persistence
+ explicit report surfacing are required; the home is the only open question.

## Build order

1. **Apply ε + the σ_f budget into a persisted final-products table** (the
   contract reports render). Ship the three-term budget with **σ_floor default 0**
   and a user-settable σ_floor field folded in quadrature. Report precision; do
   not bake in `δ_down`.
2. **Promote the cross-fixture validation harness** into the pull
   calibration/diagnostic surface (how a user with a trusted reference sets their
   own σ_floor), keeping the run-to-run (catalog-free) reproducibility path.
3. **Render** the report (table + ledgers + provenance + catalog summary), with
   the σ_f components broken out and a note that σ_floor is the user's accuracy
   declaration.

Deferred / longer horizon: residual-tail characterization; the cross-instrument
and bench tests that would *characterize* `δ_down` (not blocking — the default
declines to assert it); a live-injected absolute reference as the only path to a
per-acquisition `δ_down` correction.
