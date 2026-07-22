# Fixture 363

Real Blackchirp CP-FTMW experiment, checked in at
`examples/blackchirp_data/363/`. A second **different-sample** (MTBE chemistry)
multi-frame fixture, retained as an independent same-instrument cross-validation
point for the noise-estimation research (see the noise/SNR-scaling method doc,
`docs/source/methods/noise_snr_scaling.rst`).

## Instrument

Same spectrometer as the vinyl-cyanide fixtures: LO/probe **40960 MHz**, **lower
sideband**, **50 GS/s**, **750000** points, **15 µs** record, analysis band
**26500 - 40000 MHz**.

## Acquisition / frames

- **525 660** shots (frame 0); **two FID frames** retained (`fid/0.csv`,
  `fid/1.csv` — 525 660 / 359 890 shots). With only two frames it contributes a
  single-segment 1/√N slope estimate.

## Sample / ground truth

MTBE-related sample (from `~/github/mtbe-new`); **no line-list ground truth**.
Used only for noise behaviour, not fitting/uncertainty validation.

## Stage 5 (issue #3 cross-fixture audit)

**Flagged fit-quality anomaly.** Built through Stage 5 in its recommended shape
(`recommend_shape` → gaussian, vote 0.73), 363 fits poorly: Tier-1 SNR-aware pass
**0.19**, χ²ᵣ median **28** — far worse than the other six fixtures (0.69–0.99).
Residuals are structured over a dense high-amplitude forest (e.g. window 61: K=6,
χ²ᵣ 9.5). The Stage 5 *knob* behaviour is normal (const-order leakage baseline,
rescue accept 0.73), so the high χ²ᵣ is a fixture-level lineshape/blend-fidelity
problem, not a mistuned constant — candidate causes: gaussian-shape misfit on
MTBE, spur contamination, or unresolved blends. No ground truth to disambiguate.
Recommend a separate investigation; out of scope for the #3 knob audit. See the
Stage 5 fitting method doc, `docs/source/methods/stage5_fitting.rst`.
