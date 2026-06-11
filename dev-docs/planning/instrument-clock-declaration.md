# Instrument clock declaration — a deterministic spur prior

**Status: implemented** (declaration + lattice gate + drift lane +
annotation + SNR-scaled masks + Blackchirp auto-population + timebase
self-calibration; cross-fixture validation below). Implementation deltas
from the original design, all evidence-driven:

1. **Union nomination, not replacement.** With a declaration the sweep
   nominates from the union of the legacy integer-MHz anchor and the
   lattice points; off-lattice integer nominees keep the legacy bars.
   Replacing the integer sweep dropped real off-lattice gates (655's
   39990; 2638's 28460/30299/39820; the 363/360 ten-MHz-family tones).
2. **Locked-point band-power fallback with decay veto.** The drift lane
   also sweeps locked lattice points that produced no narrow/pair
   nominee (655's wandering 39040 = 320×6 bb fails both coherent lanes);
   such a nominee additionally faces the lattice decay veto, because a
   locked point can sit next to a real decaying line (measured: 655's
   36800 clears the band bar at 0.38 but decays at 0.39 — vetoed).
3. **Unlocked (digitizer) drift points are baseband-frame only.** An
   unlocked ADC clock injects at digitization; generating its family in
   the RF frame manufactured a false 6250×5 point at 31250 that gated
   part of 363's real K=6 molecular band.
4. **Skirt-consistent mask truncation.** The SNR-scaled mask widens only
   while the spectrum is consistent with the gated tone's own sinc-skirt
   envelope; the first bin significantly above it (real structure)
   truncates that side. Without this, the scaled mask of 1512's flat
   28440.26 tone (19 bins ≈ 1.6 MHz) swallowed four vinyl cyanide
   catalog lines 0.5–0.8 MHz away. With it, 1512 recall returns to the
   v9 reference (.443) while empty-window tones keep full-width masks.
   Partial w100 outcome: the 363 interference-doublet window improves
   K=0 χ²ᵣ ~90 → ~55 but still fails its bar — the full fix there is
   the Stage 4 spur-only window drop (still deferred).

Cross-fixture validation (full re-fit of the ship-audit fixtures with
the declaration in the `instrument_bc_2638` preset, vs the v9
reference): 363 .992 (=), 2638 .995 (wobble band), 360 1.0 (=),
1231 .988 (=), 1019 1.0 (=), 1512 .992 / recall .443 (=), 655 .977 /
recall .537 (=) — and 655's 39040 is gated (drift lane, mask 5 bins)
with zero catalog-line removals on 1512/655. The no-declaration path is
byte-stable (unit-tested SpurSet equality + full suite). The timebase
self-calibration measures ε on all seven fixtures (+0.80…+2.22 ppm,
σ ≈ 0.06–0.16 ppm, per-acquisition-epoch as expected) and passes the
all-Rb-locked null test on the second-instrument fixture (ε = 0 ± 0.05
ppm; see the succinimide notes in the team memory / scratch).

Parent:
[`stage5-spur-masking.md`](stage5-spur-masking.md) (the gate this feature
feeds); siblings: [`instrument-tunable-knobs.md`](instrument-tunable-knobs.md)
(the per-instrument preset surface this extends),
[`stage5-candidate-revival.md`](stage5-candidate-revival.md) (the other
post-stabilization UX track).

## Problem

The spur gate's hard frequency anchor is "integer MHz" — an empirical proxy
discovered on 2638. It works because the instrument's clock-derived tones
happen to land on integer MHz, but it is both too loose (13 500 integer-MHz
candidates in the 26.5–40 GHz band force conservative evidence bars, which
is why 655's drifting 39040 tone survives as a fitted "line") and too
opaque (it encodes no knowledge of *which* integers are plausible, or which
tones should drift).

The instrument's clock tree predicts the spur catalogue deterministically.
Verified against the seven-fixture gated catalogue
(`scratch/stage5-skirt/clock_lattice.py`; identities in
[`stage5-spur-masking.md`](stage5-spur-masking.md) § "Instrument
clock-lattice prior"):

- Clocks: upconversion LO 11520 MHz (= 2×5760, ×4 after lower-sideband
  mixing), downconversion LO 40960 MHz (= 8×5120, the probe), AWG
  16 GSa/s, scope 50 GSa/s as 8 interleaved 6.25-GSa/s ADCs. All but the
  scope are referenced to a 10 MHz Rb standard.
- Every cross-fixture recurring spur is a direct clock identity in one of
  two frames — baseband (f_bb = probe − f_mol: 11520, 5760, 5120, 2×5120,
  16000/2, 2×6250) or RF (5×5760, 6×5760) — and all of them sit on the
  **Rb-locked intermod lattice** of gcd = **320 MHz**. Note the bare
  fundamentals {5760, 5120, 16000} have gcd 640; the 320 lattice requires
  the AWG's half-rate product (16000/2 = 8000 — itself one of the
  strongest measured tones, and 32960 = bb 8000 is in the gated comb but
  is not a ×640 point), so 8000 is declared as its own fundamental
  (division is not a generated product the way harmonics are).
  Only 42 lattice points fall in band — a ~300× tighter prior than
  integer MHz.
- The scope clock is the one *unlocked* source, predicting a **drifting
  family** (frequency wanders relative to the Rb-locked axis). 655's
  39040 shows exactly that signature (~200 kHz asymmetric smear, erratic
  amplitude beat, decay-probe ratio 0.59 = pseudo-decaying) and defeats
  both narrowness lanes and the probe.

A user-declared clock tree turns the empirical proxy into physics the gate
can reason from — and makes the gate portable: a different instrument
(different clocks, different "shenanigans") should need only a different
declaration, not code changes.

## Design

### Declaration (settings)

A declarative list of clock sources in the Stage 5 `spur` settings
sub-block (four-layer resolution like every other knob; natural home is
the per-instrument preset YAML, persisted to the `.ftmw` like the rest of
`stage5_fit`):

```yaml
spur:
  clocks:
    - {freq_mhz: 5760,  locked: true,  label: upconv-lo}    # LO chain declared
    - {freq_mhz: 5120,  locked: true,  label: downconv-ref} # at the synth
    - {freq_mhz: 16000, locked: true,  label: awg}          # fundamental
    - {freq_mhz: 6250,  locked: false, label: scope-adc}
```

- Declare chain *fundamentals* (5760, 5120), not derived products (11520,
  40960) — harmonics are generated, so the products come for free.
- `locked` marks reference-locked sources. The probe frequency and
  sideband are already in the file (Stage 0 metadata) and are not
  duplicated here.
- Default: empty list → **behavior identical to today** (integer-MHz
  anchor). The declaration is strictly additive prior knowledge.
- Deliberately minimal schema. An ADC interleave factor (for predicting
  k×f_s/M image families) is a possible extension; start without it —
  the fundamental (6250) already generates the interleave grid.
- **Auto-population from Blackchirp metadata.** The experiment's
  `clocks.csv` records the synthesis chain directly (verified on 2638:
  `Clock.0 ×8 → DownLO 40960`, `Clock.0 ×2 → UpLO 11520` — one
  dual-output synth at 5120/5760, exactly the lattice fundamentals), and
  `header.csv` carries the digitizer sample rate. The Blackchirp loader
  should populate the declaration from these at import (user settings
  override; the scope clock's `locked` status is the one fact the
  metadata does not carry and defaults per the preset). `chirps.csv` +
  header likewise fully specify the chirp template (2638: 4895→1520 MHz
  over 1 µs, ×4 after the lower-sideband mix at 11520 → 26500→40000 ==
  the active region) for any future chirp-timing residual fit — the
  chirp is degenerate with chain dispersion for *absolute* ε (an LTI
  chain shifts a CW tone's phase but never its frequency, so spurs are
  the absolute reference), but with ε pinned by the spurs the chirp's
  matched-filter phase residual measures trigger timing and chain
  stability per acquisition, clipping-tolerant (zero crossings carry
  the information). Note Blackchirp's acquisition-time chirp phase
  correction is an *integer-sample* (20 ps) trigger-walk guard — the
  offline chirp fit complements rather than duplicates it.

### Lattice generation

From the locked clocks compute `g = gcd(freqs)` (320 MHz here once the
8000 half-clock is declared; gcd of
integers after a sanity check that declarations are integral MHz — Rb-locked
synthesizers are). The predicted **locked lattice** is the multiples of
`g` intersected with the analysis band, in *both* frames:

- molecular frame directly (RF-side harmonics radiating into the receiver);
- baseband mapped through probe/sideband (tones entering the digitizer).

Both frames reduce to "f_mol or f_bb is a multiple of g", a cheap test —
no enumeration needed. Tolerance: Rb-locked tones are exact to ≪1 Hz, so
the match tolerance is the *measurement* tolerance (~1 active-FT bin, the
same `integer_tol_mhz` role), not a clock tolerance.

Unlocked clocks contribute a **drifting lattice**: multiples of each
unlocked fundamental (in both frames) with a wider match window (the
drift scale, default a few bins — measurable from the 655 39040 smear,
~0.2 MHz). Membership marks a candidate as *expected to drift*; it never
sharpens a frequency test.

### Gate integration (the behavior changes)

1. **Nomination anchor.** With a declaration, the frequency sweep runs
   over lattice points instead of every integer MHz (42 vs 13 500 here);
   the integer-MHz sweep remains the no-declaration fallback. Off-lattice
   nomination paths are untouched (the probe-confirmed Stage 2b cluster
   lane — 12 of 363's 20 gated tones are off-lattice and must keep
   gating).
2. **Evidence-bar relaxation on-lattice.** The pair lane and the flat
   lane currently demand probe flatness ≥ 0.8 at amp_snr ≥ 10 because
   the prior is weak. On a locked-lattice point the prior odds are ~300×
   higher, so the burden of proof shifts. Strawman (to be calibrated, see
   Validation): an on-lattice narrow/pair nominee is gated unless the
   probe *clearly* decays (ratio below ~0.45 — well under every measured
   spur, above no measured line at usable SNR), and the drifting-family
   match additionally swaps the fixed-frequency demod probe for a
   drift-tolerant statistic — the band-power late/early ratio (measured:
   0.43 on the drifting 39040 vs 0.12–0.28 on real lines) or a
   late-frame band-power floor ≫ noise. This is the piece that gates
   655's 39040.
3. **Line-list annotation.** Any *fitted* peak whose frequency lands on
   the locked lattice (or in a drifting window) gets a `clock_lattice`
   flag in its persisted properties, rendered by `fit show` (table column
   + marker on the detail figure). No automatic removal — annotation is
   the conservative half of the feature and ships even if the bar
   relaxation needs more calibration. This is also the natural meeting
   point with [`stage5-candidate-revival.md`](stage5-candidate-revival.md):
   an annotated line is a one-command user removal.
4. **Mask-width interaction.** Gating strong tones surfaces the ±2-bin
   mask limitation (363 w100: the gated SNR-139 tone's sinc skirt leaks
   past the mask, K=0 χ²ᵣ 90). Scale the mask half-width with the gated
   tone's measured SNR (a sinc skirt falls ~1/(π·Δbins), so
   half-width ≈ SNR/(π·target_residual_snr) bins, capped) — or fall back
   to the Stage 4 spur-only window drop. In scope here because the
   lattice prior will gate more strong tones.

### Interface surface

Settings only (no new verbs): the `spur.clocks` block flows through the
existing four-layer resolver, `settings show/set/export`, and the preset
YAMLs; `fit show` gains the annotation rendering. Dual-interface work is
limited to the settings dataclass + serialization + the resolver, then the
gate reads the resolved block inside `stage5_impl`/`spur_detection` as
today.

### Serialization

- `spur.clocks` persists with `stage5_fit` processing parameters
  (list-of-dicts; absent = empty).
- Per-peak `clock_lattice` flag in the fitted-peak properties; gated-spur
  diagnostics entries gain the matched lattice identity (e.g.
  `lattice: "320x16 (bb)"`) for the audit trail.
- Legacy files: absent block resolves to empty; no migration.

## Validation plan

Re-run the seven-fixture sweep + full re-fit validation with the
declaration in the 2638-instrument preset:

- **Acceptance:** 655's 39040 gated; the four previously-gated narrow
  comb tones unchanged; **zero catalog-line removals** on 1512/655
  (vinyl cyanide ground truth — the hard arbiter, since the pass metric
  rewards fitting interference, measured on 363 w100).
- **Bar calibration:** sweep the on-lattice probe threshold and the
  drift statistic over the labeled populations
  (`scratch/stage5-skirt/decay_probe.py`, `pair_lane_sweep.py`,
  `probe_655_spurs4.py` band-power probe); the 0.45/band-power strawmen
  above must clear every labeled real line with margin or they don't ship.
- **Cross-instrument:** the planned different-instrument fixture (own
  clock tree, previously cleaned manually) is the real acceptance test —
  success = its spurs are handled by *declaring its clocks*, with no code
  change. Until then the no-declaration default must stay byte-stable on
  all fixtures.

## Test plan

Unit: gcd/lattice membership in both frames (incl. sideband mapping and
edge tolerance), locked vs drifting families, gate with/without
declaration (no-declaration = legacy behavior, byte-identical), bar
relaxation only on-lattice, annotation flag round-trip. Integration:
settings propagation cross-interface; an end-to-end fit on a synthetic
FID carrying a locked on-lattice tone + a drifting tone + an on-lattice
real (decaying) line, asserting gate/spare/annotate respectively.

## Timebase scale correction (measured — add to scope)

The unlocked scope clock also carries a *stable fractional offset*, and it
is measurable on the existing fixtures: regressing catalog position
residuals against baseband frequency
(`scratch/stage5-skirt/drift_regression.py`, vinyl cyanide truth) gives
**ε = 2.28×10⁻⁶ (1512) and 2.20×10⁻⁶ (655)** — the same scale error on
two independent acquisitions, monotonic across the band (≈0 kHz at 1.5 GHz
baseband → −22…−28 kHz at 13 GHz). Removing the linear term collapses
1512's catalog-match rms from 9.8 to 2.4 kHz. This is the "~10 kHz
instrument accuracy floor" previously attributed to the instrument — a
deterministic, correctable frequency-axis scale error, not a floor.

Scope addition: the unlocked-clock declaration carries an optional
`timebase_scale` (1 + ε) applied to the frequency axis (Stage 1 / active
FT). ε can be user-declared, fit from catalog matches, or — elegantly —
self-calibrated per fixture from the Rb-locked spur tones themselves
(their true frequencies are exact; their measured offsets give ε with no
catalog). Directly serves the 1512 uncertainty-accuracy goal.
Sequencing decision: the self-calibration **measures and persists** ε as
its own calibration record (`timebase run`/`show`,
`calibrate_timebase(...)`); *applying* (1 + ε) to the frequency axis is
deliberately deferred to the reports feature, which consumes the
persisted value in its corrected-frequency / uncertainty-budget output.
Correction semantics when consumed: f_true = f_measured / (1 + ε) in the
baseband frame, i.e. lower-sideband molecular frequencies correct as
f_mol = probe − (probe − f_mol_measured)/(1 + ε).

**Self-calibration demonstrated**
(`scratch/stage5-skirt/timebase_selfcal{2,3}.py`): demodulate the FID at
each Rb-locked tone's exact lattice frequency and locate the residual
offset by an ML fine-frequency scan of the block-averaged demod; ε =
offset / f_bb. On the **late record** (last ~45%, molecular lines decayed,
CW tones persist — kills line-pulling on dense spectra) the weighted mean
gives **ε = 2.30 ± 0.06 ppm on 1512 (catalog: 2.28) and 1.94 ± 0.09 ppm
on 655 (catalog: 2.20)**; the best single tone (the 5120 synth reference,
peak/noise 260–1300) matches the catalog at the 0.01–0.2 ppm level on
both. Statistical floor per strong tone is parts in 10⁸ (σ_f ≈ 60 Hz at
5.12 GHz, full record); practical accuracy is systematics-limited
(residual line pulling, weak-tone outliers — 1512's 11520 tone flips sign
at peak/noise 22). Production estimator: joint shared-ε fit across all
probe-flat Rb-locked tones, late-window or line-model-subtracted data,
outlier rejection via shared-ε consistency — conservatively ~0.1 ppm,
i.e. residual worst-case position error ≲ 1.3 kHz at 13 GHz baseband
(vs 28 kHz uncorrected). Scope-derived tones are excluded and
*identified* by the same machinery: they do not share the common ε
(measured: the 2×6250 tone sits tens of kHz off its nominal frequency
and wanders between fixtures — the drifting-family discriminant in one
measurement).

**Out-of-band calibration tones (measured; the primary ε anchors).** The
strongest Rb-locked tones sit *above* the chirp-driven molecular band
(baseband > ~14.5 GHz on the home instrument), a region the analysis trim
never sees and molecular emission cannot reach — the chirp does not
excite there. Wide-band survey + ML reads on all seven fixtures
(`scratch/stage5-skirt/survey_oob.py`, `timebase_selfcal{4,5,6}.py`):

- **15360 = 3×5120** is the strongest clock tone on *every* fixture
  (700–7400× the spectral floor; peak/noise up to ~7000; per-tone
  statistical σ(ε) down to ~6×10⁻¹⁰), with strong companions at
  **17920 = 56×320** and **17280 = 3×5760**. (A tone at exactly 16000 —
  the AWG fundamental — is weak-to-absent on these deeply averaged
  records; "the 16 GHz spur" resolves to the 15.36 GHz identity.)
- These tones are *window-stable* (full-record vs late-window reads agree
  to ±0.03–0.06 ppm), unlike in-band tones, whose early-record reads are
  molecular-pulled by up to ~0.3 ppm. Full-record reads of out-of-band
  tones are therefore the calibration anchors; no late-window dodge is
  needed.
- Scope-frame controls behave exactly as the frame physics predicts:
  18750 = 3×6250 and 25000 (Nyquist) read offsets ≈ 0 at exact rational
  sample-space frequencies on every fixture, while every Rb-locked tone
  carries the shared +ε·f offset — the drifting-family discriminant in a
  single measurement.
- ε is per-acquisition-epoch: 363/360 measure ~+0.8–0.9 ppm while the
  other five fixtures measure ~+1.8–2.3 ppm. The calibration must be
  per-file; a constant is wrong.

**Below-chirp region (measured; not usable for ε).** The baseband region
*below* the chirp start (< ~960 MHz) is also molecular-free and does
carry strong tones at 320/640/960 plus a 10-MHz comb
(`scratch/stage5-skirt/survey_lowbb.py`) — but their offsets are
kHz-scale and **not ε-scaled** (e.g. the 320 tone reads −5.5 kHz on both
1512 and 2638 where ε predicts +0.7 kHz; signs and magnitudes vary by
fixture and tone). They are a third tone family sitting genuinely
off-nominal, their ε leverage is ~50× weaker than the high-baseband
tones', and the shared-ε consistency rejection removes them cleanly
(verified on all seven fixtures). The production estimator keeps them
only as rejected table entries.

**Production estimator (validated prototype,
`scratch/stage5-skirt/timebase_selfcal5.py`):** sweep *all* locked-lattice
multiples k·g up to ~Nyquist on the full active record; per-tone ML scan;
per-tone error σ_tot² = σ_f² + (κ_sys·f_bb)² with κ_sys ≈ 0.2 ppm (the
measured tone-to-tone systematic floor — a fixed-kHz floor wrongly
rejects the highest-leverage tones); iterative weighted shared-ε fit with
4σ_tot consistency rejection. Converges to the strong-tone consensus on
all seven fixtures, auto-rejecting the below-chirp family and in-band
molecular contaminants. Honest accuracy: ~0.1–0.2 ppm systematic. The
catalog-truth cross-check agrees within ~0.2–0.3 ppm; the residual is a
real frame difference (fitted lines weight the early record where their
amplitude lives; tones weight the record uniformly) and belongs in the
reports feature's per-line uncertainty budget, not in the tone
calibration.

Falsified alongside (`scratch/stage5-skirt/split_fraction.py`): the
bright-line close multiplets are NOT clock-wander sidebands — fractional
splits span 1.4–160 ppm with no f_bb scaling, and the headline 655 w940
"doublets-are-quartets" separations (4–6 kHz at 37939.x) match real
catalog hyperfine. The ultra-high-SNR lineshape floor concentrates at
LOW baseband where wander broadening would be weakest, so drift is not
the bulk shape-error mechanism either; only the position error is.

## Time-domain interleave-offset subtraction (measured, out of scope here)

A complementary technique used on another instrument (16 interleaved ADCs):
estimate each interleave phase's DC offset by averaging a noise segment
mod M (and a second, deeper periodicity, e.g. mod 512), subtract the tiled
pattern from the record — annihilating the k·f_s/M offset-spur comb at the
source. Tested on the checked-in fixtures using the pre-chirp segments
(`scratch/stage5-skirt/interleave_subtract.py`, 655/360/2638/363):

- **No significant mod-M pattern** (M = 4…1024) in any pre-chirp segment,
  down to the estimation floor.
- **The estimator cannot reach the tones that do exist.** In these
  *averaged* records the interleave tones are tiny in the time domain
  (360's 2×6250 tone ≈ 1×10⁻⁷ amplitude) while the pre-chirp segments are
  short (0.34–0.64 µs → mod-8 mean noise ≈ 1.5×10⁻⁶ on the quietest
  fixture): subtracting the estimated pattern would *inject* a mod-8 comb
  ~10× larger than the tones it removes. The technique's operating regime
  is per-shot / pre-average data or long dedicated noise records, where
  offsets are large against the estimate's noise — not sub-µs pre-chirp
  on a deeply averaged record. (Only one tone in the whole gated
  catalogue, 28460 = 2×6250 baseband, is interleave-derived anyway.)
- **The transferable insight is the frame.** A scope-derived tone sits at
  an exact *rational frequency in sample space* (k/M cycles per sample),
  regardless of its drift in the Rb-locked frequency frame — the drifting
  family is perfectly-known-frequency in the right coordinates. If
  sample-space cleanup is ever needed, the better estimator on averaged
  records is a coherent full-record LSQ of the known comb frequencies
  (~600 k samples of processing gain, no noise segment required), not a
  short-segment pattern average.

**Why the technique is essential there and unnecessary here (measured).**
On the other instrument the offsets are *true analog offsets*: noise
~2000 LSB and per-phase offsets ~200 LSB in the stored data — far above
any quantization scale, so the 16-bit storage is benign (well dithered)
and an undithered-requantization mechanism is ruled out. The offsets sit
10× below the time-domain noise yet dominate the spectrum because they
are coherent: a comb tone gains √N_samples over the noise floor in the
FT (a 0.1·σ tone over ~10⁶ samples reads ~100× the floor), and shot
averaging lowers the noise under them further while leaving them fixed.
The estimator side is equally comfortable: a mod-16 average over an
L-sample noise segment has per-phase noise σ/√(L/16) — a few LSB against
200 — so the subtraction is clean. The home instrument's scope, by
contrast, runs internal interleave calibration that nulls its ADC
offsets, which is why no mod-M pattern exists in these fixtures and the
residual scope tones are at the 10⁻⁷ level. The operating criterion for
the cleanup is therefore simply **offset amplitude ≫ estimator noise
(σ/√(L/M))**, instrument-dependent, checkable in one measurement.

For the planned different-instrument fixture: if its records arrive
pre-average or with long noise segments, the mod-M subtraction belongs in
an optional Stage 0 cleanup keyed off the same clock declaration (the
unlocked-ADC entry supplies M); revisit when that fixture lands.

## Out of scope

- Signal-dependent ADC interleave *images* (k×f_s/M ± f_in) — they track
  their parent line's decay, so the decay probe already treats them as
  lines; masking them needs parent association, a separate design.
- Modeling/subtracting tones instead of masking (unchanged from parent
  doc).
- Auto-discovery of the clock tree from the data (the declaration is
  cheap and exact; inference is not).

## Open questions

1. Exact on-lattice bar semantics: relax the flat threshold, flip the
   default verdict, or score prior×evidence jointly? (Strawman above is
   the flip; calibration decides.)
2. Drift statistic: band-power late/early ratio vs late-floor-over-noise
   vs both. The measured separation (0.43 vs 0.12–0.28) is real but thin —
   may need more frames or a τ-aware expected-decay reference.
3. Should Stage 3 also annotate (not gate) detected peaks on the lattice?
   Cheap, but duplicates the Stage 5 annotation; decide when wiring
   `fit show`.
4. Whether `clocks` should eventually promote from the `spur` block to an
   instrument-level settings home shared with future consumers (e.g. a
   frequency-axis sanity check at import). Start in `spur`; promote when
   a second consumer exists.
