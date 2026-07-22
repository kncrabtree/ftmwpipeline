# Fixture 360

Real Blackchirp CP-FTMW experiment, checked in at
`examples/blackchirp_data/360/`. A **different-sample** (MTBE chemistry, not
vinyl cyanide) multi-frame fixture, retained as an independent same-instrument
cross-validation point for the noise-estimation research (see the noise/SNR-scaling
method doc, `docs/source/methods/noise_snr_scaling.rst`).

## Instrument

Same spectrometer as the vinyl-cyanide fixtures: LO/probe **40960 MHz**, **lower
sideband**, **50 GS/s**, **750000** points, **15 µs** record, analysis band
**26500 - 40000 MHz**.

## Acquisition / frames

- **775 860** shots (frame 0); **three FID frames** retained (`fid/0.csv` …
  `fid/2.csv`) — progressive backups 359 860 / 719 860 / 775 860 shots — for the
  1/√N noise scaling test.

## Sample / ground truth

MTBE-related sample (from `~/github/mtbe-new`); **no line-list ground truth**.
Used only for noise behaviour (the noise floor is a sample-independent
spectrometer property), not for fitting/uncertainty validation.
