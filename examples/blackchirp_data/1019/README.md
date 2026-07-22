# Fixture 1019

Real Blackchirp CP-FTMW experiment, checked in at
`examples/blackchirp_data/1019/`.

## Instrument

Same instrument as 2638: downconversion LO / probe **40960 MHz**, **lower
sideband**; digitizer **50 GS/s** (spacing 2e-11 s), **750000** points,
**15 µs** record. AWG chirp sweeps **4895 → 1520 MHz** IF (RF coverage
36065 - 39440 MHz). Analysis band **26500 - 40000 MHz**.

Blackchirp build `v0.1-381-g1f52768` (beta). The `fidparams.csv` `sideband`
cell is the **integer code `1`** (older encoding), exercising the loader's
version-tolerant sideband path.

## Acquisition

- Chirp duration **3 µs** (the longest in the set).
- **1590380** shots co-averaged (the highest-SNR fixture).

## Sample

Not recorded in the experiment metadata (user-held).

## Ground truth

**Available** — known frequencies for some of the peaks (held outside the repo).
Suitable for cross-fixture frequency/intensity validation.

## Recommended processing

- `start_us = 4.35` (data-derived via `detect-start`; chirp end ~3.68 µs + the
  switch-bounce ringdown margin).
- No zero-padding (`zpf=0`) and `expf_us=None` (unapodized), `trim=(26500, 40000)`. The canonical FT stays raw so Stage 2/5 statistics are not corrupted; Stage 3 peak detection applies its own zero-padding internally.
- Run `calibrate_tau` before peak detection.
