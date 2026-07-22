# Fixture 655

Real Blackchirp CP-FTMW experiment, checked in at
`examples/blackchirp_data/655/`. **High-SNR** vinyl-cyanide fixture; the primary
fixture for the noise-estimation research (see the noise/SNR-scaling method doc,
`docs/source/methods/noise_snr_scaling.rst`).

## Instrument

Same instrument as 2638/1512: downconversion LO / probe **40960 MHz**, **lower
sideband**; digitizer **50 GS/s** (spacing 2e-11 s), **750000** points, **15 µs**
record. AWG chirp sweeps **4895 → 1520 MHz** IF (RF coverage 36065 - 39440 MHz).
Analysis band **26500 - 40000 MHz**. Blackchirp build `v0.1-337-g4748747`.

## Acquisition

- Chirp duration **2 µs**.
- **2 133 080** shots co-averaged — the highest-SNR fixture (max spectrum SNR
  ~10⁵–10⁶, ~120× the 1512 fixture). `processing.csv` records the experimenter's
  own settings, which match our recommended recipe (`FidStartUs=3.35`,
  `FidZeroPadFactor=0`, `FidExpfUs=0`, no window).

## Backup frames

**Six FID frames retained** (`fid/0.csv` … `fid/5.csv`): frame 0 is the final
fully-averaged FID (2 133 080 shots); frames 1–5 are progressive backups
(359 880 … 1 799 920 shots). These are load-bearing for the noise research — the
1/√N test and the frame-difference noise reference require them — so unlike the
earlier fixtures this one is **not** stripped to the primary frame.

## Sample

Vinyl cyanide (CH₂=CH–CN), with ¹³C isotopologues and ¹⁵N present, plus possible
vibrationally-excited states. Ground-truth catalogs for all five resolved species
(v=0 main + three ¹³C + ¹⁵N, 328 in-band lines) are tracked under
[`../vinyl-cyanide-reference/`](../vinyl-cyanide-reference/README.md) — raw SPCAT
`.cat` files in `catalogs/` and the parsed in-band union in `combined_lines.csv`.

## Recommended processing

- `start_us = 3.35`, `zpf = 0`, `expf_us = None`, `trim = (26500, 40000)`, then
  `calibrate_tau` before peak detection (same recipe as 1512).
- `recommend_shape` returns **lorentzian** (vinyl cyanide; 88% exp vote); fit
  Stage 5 with `shape="lorentzian"`. The bounded-merge Stage 4 is required here —
  the dense, ultra-high-SNR forest otherwise collapses into GHz mega-windows.

## Stage 5 validation

Measured with `validate-stage5-shape-error` on the canonical bounded-merge build
(lorentzian, 633 windows, 1675 fitted lines; κ=0.05, F=3). This is the **dense,
extreme-SNR** member of the set — it exercises the deficit-dominated regime of
the gate (and is why the raw χ²ᵣ Tier-1 gate had to be reformulated; see D10).

- **Tier 1 (SNR-aware health):** overall pass **0.69**. By `SNR_max` bin:
  - `<100` (499 windows): χ²ᵣ median **2.71**, pass 0.65 — this is the genuine
    dense-vinyl-cyanide **lineshape-fidelity floor** (window sizing, cross-window
    leakage, and τ were each falsified as levers for it; not a defect to drive to
    χ²ᵣ→1). The ~35% that fail are the real outlier tail (χ²ᵣ p95 ~373).
  - `100–1k` (93): χ²ᵣ median 5.2, pass 0.77, fractional deficit ε ≈ **0.8%**.
  - `1k–10k` (29) and `≥10k` (12): pass **1.0**, ε ≤ 0.5%. The gate correctly
    accepts the bright cores — a line fit to part-in-10⁵ shows χ²ᵣ up to ~7900
    purely because at SNR 10⁴–10⁵ a sub-percent deficit is hundreds of σ/bin.
- **Tier 3 (ground truth — the dense union):** matched against the 328-line
  5-species union (`../vinyl-cyanide-reference/combined_lines.csv`). Recall **0.39**
  (127/328). The automated mutual-nearest match within ½·FWHM is **noisier** on
  this dense, hyperfine-blended forest than on the sparse 1512 list, so its
  detrended residual scatter (~47 kHz) is **mismatch-contaminated** and is *not*
  the clean instrument accuracy floor — read that off 1512 (~9 kHz, see
  [`1512`](../1512/README.md)). The σ_f-overconfidence finding holds (matched residual
  ~56× the reported σ_f). A tighter, unambiguous-line match would isolate 655's
  own floor; the sparse 1512 read is the trustworthy one for now.
