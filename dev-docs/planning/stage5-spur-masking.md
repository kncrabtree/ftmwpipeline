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
established the fingerprint and the remediation on the 2638 fixture; this plan
turns that into production behaviour.

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

1. **Temporal persistence (Stage 2b STFT).** Stage 2b's τ-calibration STFT
   already classifies every bin exponential-vs-constant (AICc) and labels a bin
   `spur` when the constant model wins or τ saturates at τ_max. It groups
   adjacent sinc-skirt bins into a `SpurCluster` (`center_freq_mhz`,
   `bin_indices`) and persists the catalogue at `/stage2b_*/spur_clusters`. The
   cluster `bin_indices` give a data-driven mask extent.
2. **Integer-MHz (+ narrowness) on the active-FT.** The frequency-domain test
   from the prototype.

Why both: the persisted STFT catalogue alone is unusable for masking — it holds
~135 clusters and the constant/saturation criterion conflates true spurs with
**long-τ strong real lines** (e.g. 33421.1 MHz at SNR 187, 38744.2 MHz at SNR
121 are in it). Masking those would delete signal. The integer-MHz requirement
is the guard against that false positive; persistence rescues the split-bin
spurs the frequency test misses (integer falls between bins, energy splits).
Neither detector is a superset of the other.

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

## Implementation surface

### Spur gating (`fitting/spur_detection.py`, new — pure algorithm)
- Integer-MHz + narrowness detector on an active-FT slice (lift from the
  prototype; arrays in, spur frequencies/clusters out).
- Joint gate combining the active-FT detections with the persisted Stage 2b
  `SpurCluster` catalogue; map cluster bins to the active-FT grid; emit the
  gated spur set + per-spur bin cluster.
- No file IO; unit-tested in isolation on synthetic spectra + synthetic
  catalogues.

### Fit machinery (`window_fit.py`, `plan_execution.py`)
- `fit_window`: accept an optional per-bin spur mask; exclude masked bins from
  the residual/Jacobian/χ² (reduce `n_data` accordingly so χ²ᵣ stays calibrated).
- `plan_execution`: drop nominated candidate offsets that fall on a spur
  cluster; thread the spur mask into each window's `fit_window` call.

### Stage 5 wiring (`_internal/stage5_impl.py`)
- Build the gated spur set once (active-FT + persisted Stage 2b catalogue),
  pass the per-window mask + nomination exclusion through `execute_plan`.
- Auto-detect the Stage 2b catalogue's presence (same pattern as τ_maj); fall
  back to the active-FT-only detector when absent.

### Stage 4 (`_internal/stage4_impl.py` / window planning) — optional
- Drop spur-only windows from the plan. Lower priority; can land after the
  Stage 5 mask.

### Settings (`core/stage_fit_settings.py`)
- New `spur` sub-block + `_HARD_DEFAULTS["spur"]`: `enabled`,
  `integer_tol_mhz`, `narrowness_ratio`, `snr_threshold`, `mask_half_width_bins`,
  `use_stft_catalogue`. Instrument-tunable (register in
  `instrument-tunable-knobs.md`).

### Persistence (`io/`)
- The Stage 2b catalogue is already persisted. Record which spur bins the Stage
  5 fit masked (audit), under the `stage5_fit` parameter group.

### Dual-interface
- Spur masking is internal to the fit, controlled by the `spur` settings block,
  so it flows through `pipeline.py` / `api.py` / CLI via the existing settings
  plumbing — no new subcommand. Add a cross-interface consistency test.

### Tests
- Unit: gating function (integer-MHz gate rejects a synthetic long-τ line at an
  integer MHz; narrowness spares a real line; persistence catches a split-bin
  spur); `fit_window` mask reduces `n_data` and excludes the bin.
- Integration: on the 2638 fixture, spur windows' χ²ᵣ drops toward noise on the
  clean spurs, w245 recovers only the spur portion, and no real line is masked.
- Cross-interface consistency.

## Validation

Acceptance, against `probe_spur_detector.py` + the per-window baseline at
`scratch/stage5-validation-rescue_prominence_threshold__1p5__gaussian/`:

- Spur bucket χ²ᵣ recovery ≈ the prototype's ±2–3 bin figure (~98–103 of ~106).
- Zero real-line false positives (the 354 spared integer-MHz lines stay
  unmasked; no fitted peak is removed by spur exclusion).
- Spur-only windows (88, 126, 193, 287, 372, ...) drop or fit cleanly; mixed
  windows (245) keep their real lines and lose the spur's χ².

## Implementation progress

- [ ] `fitting/spur_detection.py` detector + joint gate + unit tests
- [ ] `fit_window` spur mask (residual/Jacobian/`n_data`)
- [ ] `plan_execution` nomination exclusion + mask threading
- [ ] `stage5_impl` gated-set construction + Stage 2b auto-detect
- [ ] `spur` settings sub-block + `_HARD_DEFAULTS` + instrument-knobs row
- [ ] persistence of masked bins + cross-interface test
- [ ] integration validation on 2638; update the audit report with recovery
- [ ] (optional) Stage 4 spur-only window drop

## Open questions

- **Gate logic.** Union of (integer ∧ narrow) and (integer ∧ persistent), with
  integer-MHz as the hard requirement on both — vs intersection (stricter,
  fewer detections). The union catches more spurs while integer-MHz holds the
  false-positive rate near zero; a real line at *exactly* integer MHz remains a
  residual risk (rare; the narrowness sub-test rejects it on the frequency
  side).
- **Mask extent source.** Data-driven from `SpurCluster.bin_indices` (mapped to
  active-FT) vs a fixed ±N half-width. The catalogue may under-cluster split-bin
  spurs; a fixed-width fallback covers them.
- **Long-τ-line guard.** Integer-MHz (chosen) vs a τ-saturation exclusion
  (drop STFT spurs whose bin also carries a strong fitted line). Integer-MHz is
  simpler and the 2638 spurs are unambiguous clock harmonics.
- **Stage 4 spur-only drop** now or as a follow-up.

## Out of scope

- Modeling and subtracting the CW tone (a sinc fit) instead of masking — the
  mask recovers ~the full bucket far more cheaply.
- Non-integer instrumental tones (none observed on 2638; the gate would not
  catch them without relaxing the integer requirement).
- Cross-fixture spur calibration (per-instrument spur frequency catalogues).
