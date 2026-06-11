# Stage 5 spur masking

Detect clock / LO spurs and keep them out of the Stage 5 fit: exclude them
from peak nomination and from the χ²/residual sum, and drop windows whose only
content is a spur. The mechanism reuses the spur catalogue Stage 2b already
builds and persists, consumed through a gate that separates true CW spurs from
real lines.

## Motivation

The Stage 5 per-window audit (`research/stage5-gaussian-audit/report.md` § Step
5) found a `spur` bucket carrying ~106 excess χ²ᵣ units. A clock/LO spur is a
persistent CW tone: a single-bin (plus sinc-skirt) delta that **no finite-T
line shape can represent**, so it detonates χ² even when the fitter correctly
never places a peak on it (the w245 note: "excluded from the fit *and* the
residual/chi² calculation"). Some spurs instead attract a spurious fitted peak,
polluting the output line list. There is no spur handling in Stage 5 today; the
only `spur` logic in `src/` is Stage 2b's, which masks spur bins from the τ
histogram and never propagates downstream.

The prototype (`research/stage5-gaussian-audit/probe_spur_detector.py`)
established the fingerprint and the remediation on the 2638 fixture; the shipped
mechanism is that prototype productionised.

## Fingerprint (validated)

Two robust discriminators (the "energy in one quadrature" idea from the
original audit note is empirically false and is not used):

- **Exact integer-MHz center.** Every classified spur sits within a fraction of
  a bin of an integer MHz (clock harmonics).
- **Sub-resolution narrowness / temporal persistence.** A CW tone persists over
  the whole acquisition, so it is transform-limited by the full boxcar (first
  null ~1/T ≈ one active-FT bin) — narrower than a molecular line that decays in
  τ < T. In the frequency domain its peak bin is 10–25× its neighbours; in the
  time domain its STFT-frame magnitude is constant rather than decaying.

## Design

Two independent signals, combined by a **joint gate**:

1. **Temporal flatness (Stage 2b STFT).** Stage 2b's τ-calibration STFT
   classifies every bin by fitting exp-vs-constant across the STFT frames; a bin
   is `spur` when τ saturates at τ_max (`spur_by_tau`, no decay detected) or the
   constant model beats the exp by AICc > 2 (`spur_by_aicc`). A true CW tone is
   **flat in time** (frame magnitudes ~constant → τ saturates); this is the
   reliable signal. It groups adjacent sinc-skirt bins into a `SpurCluster`
   (`center_freq_mhz`, `bin_indices`) persisted at `/stage2b_*/spur_clusters`,
   giving a data-driven mask extent.
2. **Integer-MHz (+ narrowness) on the active-FT.** The frequency-domain test
   from the prototype.

Why both: the persisted STFT `cls == 1` set is unusable for masking as-is — it
holds ~135 clusters because the `spur_by_aicc` branch also fires on **erratic,
non-exponential bins** (damped beats / unresolved blends: e.g. 33421.1 and
38744.2 MHz, whose frame magnitudes bounce non-monotonically so the exp fits
badly and the constant wins by a thin margin). Those are not spurs. Two guards
remove them: they are **non-integer-MHz** (the integer gate rejects them), and
the persistence half should key on **flatness / saturation** (`spur_by_tau`, or
a slope-equivalence test) rather than the raw `cls == 1`, since true spurs are
flat while the beat/blend bins have large frame-to-frame spread. (No real line
trips saturation: real τ ≤ 9 µs ≪ τ_max = 100 µs, so a genuine line always
shows decay.) Persistence can in principle rescue a split-bin spur the
frequency test misses (one that rails τ to saturation while its split energy
fails the narrowness ratio); on the 2638 fixture this did not occur — the
saturated set is a subset of the narrow detections and the split-bin spurs
39830/39930 are caught by neither (see "Implementation status").

**Gated spur set** = integer-MHz active-FT bins that are *either* sub-resolution
narrow (frequency test) *or* flagged persistent by the Stage 2b catalogue. Each
gated spur carries the active-FT bin cluster to mask (from `SpurCluster.bin_indices`
mapped to the active-FT grid, with a fixed half-width fallback for split-bin
spurs the catalogue under-clusters).

**Remediation is a cluster mask, not a single bin.** Validated recovery of the
~106-unit bucket: single bin → 22.9, ±1 → 88, ±2 → 98, ±3 → 102.8. The default
half-width is ±2 bins (conservative; ±3 captures the last of the strongest
spur, w287). The gated mask is consumed in two places:

- **Peak nomination exclusion** — drop candidate offsets (Stage-3-seeded and
  rescue `find_residual_peaks`) that land on a spur cluster, so the fitter never
  places a peak on a spur.
- **χ²/residual mask** — exclude spur-cluster bins from the weighted residual
  sum in `fit_window`, so a window that shares a spur with real lines (w245,
  w287) no longer carries the spur's χ². This is the load-bearing change.

**Spur-only window drop (optional, Stage 4).** A window whose only content is a
gated spur (no non-spur peak) is dropped before Stage 5.

## Implementation

The mechanism is `fitting/spur_detection.py` → consumed by `fit_window` and
`plan_execution` → built once in `stage5_impl`, controlled by the `spur` settings
sub-block (default on) and flowing through all three interfaces via the existing
settings plumbing (no new subcommand).

### Spur gating (`fitting/spur_detection.py`, pure algorithm)
`detect_active_ft_spurs` runs the integer-MHz + narrowness detector on an
active-FT slice (arrays in, spur frequencies/clusters out). `gate_spurs` combines
those detections with the persisted Stage 2b `SpurCluster` catalogue — mapping
cluster bins to the active-FT grid — and emits the gated `SpurSet` /
`SpurMaskSpec` (gated spur set + per-spur bin cluster). No file IO; unit-tested in
isolation on synthetic spectra + synthetic catalogues
(`tests/unit/fitting/test_spur_detection.py`).

### Fit machinery (`window_fit.py`, `plan_execution.py`)
`fit_window` takes an optional per-bin spur mask and excludes masked bins from the
residual / Jacobian / χ² (reducing `n_data` so χ²ᵣ stays calibrated).
`plan_execution` drops nominated candidate offsets that fall on a spur cluster,
threads the spur mask into each window's `fit_window` call (including the rescue
and thaw paths), and drops a spur that was fit as a fixed contributor for a
downstream window.

### Stage 5 wiring (`_internal/stage5_impl.py`)
The gated spur set is built once (active-FT + persisted Stage 2b catalogue) and
the per-window mask + nomination exclusion pass through `execute_plan`. The Stage
2b catalogue's presence is auto-detected (same pattern as τ_maj); absent it, the
active-FT-only detector is used.

### Settings (`core/stage_fit_settings.py`)
The `spur` sub-block + `_HARD_DEFAULTS["spur"]` carries `enabled`,
`integer_tol_mhz`, `narrowness_ratio`, `snr_threshold`, `mask_half_width_bins`,
`use_stft_catalogue`; instrument-tunable (registered in
`instrument-tunable-knobs.md`).

### Persistence (`io/`)
The Stage 2b catalogue is persisted by Stage 2b; the spur bins the Stage 5 fit
masked are recorded for audit under the `stage5_fit` parameter group.

### Tests
Unit: the gating function (integer-MHz gate rejects a synthetic long-τ line at an
integer MHz; narrowness spares a real line; persistence catches a split-bin
spur); the `fit_window` mask reduces `n_data` and excludes the bin. Integration:
on 2638 the spur windows' χ²ᵣ drops toward noise, w245 recovers only the spur
portion, and no real line is masked. Plus cross-interface consistency.

## Validation (2638)

Sum Δχ²ᵣ = 111.5 over the classified spur windows; w245 keeps its 3 real lines;
zero real-line removals (the 354 spared integer-MHz lines stay unmasked). The
saturated-catalogue path is exercised end-to-end: re-running Stage 2b populates 4
saturated clusters (30720/32960/35840/39040) and the gate upgrades them to
`narrow+saturated` with χ²ᵣ recovery and real-line safety unchanged (625 = 625
peaks). On this fixture the saturated set is a subset of the narrow detections —
corroborative, not additive; the split-bin spurs 39830/39930 are gated by neither
detector. Full detail: `research/stage5-gaussian-audit/report.md`
§ "Production wiring & validation".

## Resolved decisions

- **Gate logic.** Union of (integer ∧ narrow) and (integer ∧ saturated), with
  integer-MHz the hard requirement on both. Validated zero real-line
  false-positive on 2638 (the 354 broad integer-MHz lines lack the narrowness
  signature and are never gated).
- **Mask extent.** Fixed ±N half-width (default ±2 bins) by nearest-bin in the
  window frame, implemented as a frequency half-width on `SpurMaskSpec` so it is
  invariant to the fit routines' internal grid re-sorting. Nomination exclusion
  uses a tighter ~1-bin tolerance so a real line a couple of bins from a spur
  survives nomination while the residual mask still spans ±N.
- **Persistence signal: flatness, not raw `cls == 1`.** Settled by measurement
  (audit report § "Flatness-exposure measurement"): of 59 integer-MHz `cls == 1`
  bins on 2638, 53 are erratic `spur_by_aicc` real lines (e.g. 38744 @ SNR 260)
  and only 6 are flat clock harmonics. The persistence half therefore keys on a
  per-cluster `saturated` flag (`spur_by_tau`), added to `SpurCluster` + its
  serialization (legacy catalogues default `saturated=False`, degrading to the
  frequency-domain detector).
- **Default-on.** Spur masking ships enabled (`spur.enabled=True`): the gate is
  integer-MHz-anchored and validated zero real-line FP, unlike the transitional
  default-off rescue.

## Time-domain decay-probe arbitration and the pair lane

Two later additions extend the joint gate (see the `spur_detection.py`
module docstring for the operative contract):

- **FID decay probe** (`make_decay_probe`): when the raw FID is available,
  every verdict is arbitrated in the time domain — the FID is demodulated at
  the nominee's baseband frequency, block-averaged into frames, and the
  late/early amplitude ratio separates a decaying line (vetoes a narrow
  nominee) from a flat CW tone (gates ANY Stage 2b cluster, integer or not,
  saturated or not — the bare `saturated` flag is trusted in neither
  direction; measured false positives both ways).
- **Pair lane** (`Spur.pair`, source `"narrow-pair"`): a CW tone whose
  frequency falls *between* two grid bins splits its power across them
  (each bin reads ~0.6–1.0 of the other), defeating the single-bin
  narrowness test — the 655 35840.0 spur (SNR ~300, probe ratio 1.00) was
  fit as a molecular line this way. The detector now also tests the
  integer bin + its stronger neighbour as a pair against the bins flanking
  the pair; because a blended doublet can mimic that signature, a pair
  nominee gates **only** with probe-flat confirmation (≥ `RATIO_FLAT` at
  `amp_snr ≥ MIN_SNR_FLAT`) and is never gated without a probe.
  Cross-fixture sweep: gates 655's 35840 plus three flat-confirmed tones
  on 363 (one is the known recurring 28058.1 interference tone); every
  real-line pair nominee on all seven fixtures — including SNR-10289
  (1512) and SNR-8004 (1019) lines — decays and is correctly skipped.

## Open follow-ups

- **Stage 4 spur-only window drop.** Optional cleanliness step (spur-only
  windows already fit to the null model and contribute ~noise χ²ᵣ).
- **Instrument clock-lattice prior (the principled comb prior).** The
  instrument's clock tree explains the gated catalogue deterministically:
  upconversion LO 11520 (= 2×5760, ×4 after mixing), downconversion LO
  40960 (= 8×5120), AWG 16000, scope 8× interleaved 6250 — all but the
  scope referenced to a 10 MHz Rb clock. Verified identities across the
  seven-fixture gated catalogue (f_bb = 40960 − f_mol): 29440 → 11520
  (LO leakage), 35200 → 5760 (pre-doubler), 35840 → 5120 (synth
  reference), 30720 → 2×5120, 32960 → 8000 = 16000/2 (on ALL seven
  fixtures), 28460 → 12500.0 = 2×6250 (scope ADC interleave); plus
  direct RF-side harmonics 28800 = 5×5760 and 34560 = 6×5760. Every
  cross-fixture recurring spur sits on the Rb-locked intermod lattice
  gcd(5760, 5120, 16000, 16000/2) = 320 MHz (the declared half-clock
  supplies the factor; the bare fundamentals alone give 640) — only 42
  lattice points in the
  26.5–40 GHz band vs the 13500 integer MHz the current gate sweeps
  (~300× tighter prior). Proposed design: a per-instrument **clock
  declaration** in settings (preset layer) generating the predicted
  lattice in BOTH frames (baseband through the LO + direct RF
  harmonics), used to (a) lower the spur-gate evidence bar on-lattice —
  this is what 655's 39040 (f_bb = 1920 = 6×320, probe-ambiguous at
  0.59) needs, (b) annotate fitted lines landing on lattice points as
  suspect in `fit show`, (c) mark unlocked-clock (scope-derived)
  families as the *drifting* population needing a drift-tolerant
  flatness statistic (655's 39040 measured: ~200 kHz asymmetric smear,
  erratic amplitude beat, late band-power ratio 0.43 vs 0.12–0.28 for
  real lines). A lattice hit raises prior odds but lines can sit on
  lattice points — the prior lowers the bar, the probe still arbitrates.
  Caveat: 12 of 363's 20 gated tones (the f_bb ≈ 12.9 GHz flat cluster)
  are off-lattice, so the lattice complements, never replaces, the
  probe-confirmed cluster lane. Analysis script:
  `scratch/stage5-skirt/clock_lattice.py`.
- **Strong-tone mask width / metric interaction.** Gating a strong CW
  tone whose sinc skirt extends past the ±2-bin mask leaves an unmasked
  skirt residual the null model cannot absorb (363 w100 after the
  28058.09 gate: K=0, χ²ᵣ 90 over the unmasked bins — previously that
  tone was *fit as a line*, which passed the SNR-aware metric while
  polluting the line list). Options: scale the mask half-width with tone
  SNR, or the Stage 4 spur-only window drop above. The pass metric
  rewards fitting interference; catalog truth is the arbiter.
- **Split-bin / weak clock harmonics (catalogue pollution).** On 2638 the
  ×10-MHz harmonics 39810/39830/39930 are gated by neither original
  detector: the active-FT bin grid (79.05 kHz) puts them half a bin off
  the integer so their energy splits across two bins, while the 10-frame
  STFT sees their flatness but at per-frame SNR < `t_sigma=5` so they
  classify `cls=0`, not `cls=1`. The pair lane now *nominates* this class,
  but on 2638 they stay below the probe's `MIN_SNR_FLAT` amplitude floor
  (e.g. 39810: ratio 1.14 at amp_snr 6) so they are still fit as weak
  lines (χ²ᵣ cost ~0, line-list pollution 3). Lowering the probe floor
  carries real-line false-positive risk for ~0 χ²ᵣ gain. Deferred;
  diagnosed in the audit report
  § "Why 39830 / 39930 fall through both detectors".
- **Flat-catalogue exercise.** *Done.* The 2638 fixture's persisted Stage 2b
  catalogue predated the `saturated` flag, so the shipped validation ran the
  frequency-domain detector alone. Re-running Stage 2b (Lorentzian + Gaussian
  twins) populates the flag — 4 saturated clusters (30720/32960/35840/39040),
  matching the flatness-exposure measurement — and the gate upgrades those to
  `narrow+saturated`. It does **not** add the split-bin spurs 39830/39930 the
  earlier write-ups expected: those rail τ to neither saturation nor pass the
  narrowness ratio on 2638, so the saturated set is a strict subset of the
  narrow detections (corroborative, not additive). No regression vs the
  frequency-domain run. See the audit report
  § "Saturated-catalogue path exercised end-to-end".

## Out of scope

- Modeling and subtracting the CW tone (a sinc fit) instead of masking — the
  mask recovers ~the full bucket far more cheaply.
- Cross-fixture spur calibration (per-instrument spur frequency catalogues).
  (Non-integer instrumental tones are no longer out of scope: the
  probe-confirmed flat-cluster lane gates them when Stage 2b carries a
  cluster — e.g. 363's 28057.46.)
