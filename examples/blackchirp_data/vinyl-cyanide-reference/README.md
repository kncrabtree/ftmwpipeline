# Fixture 1512 — vinyl cyanide ground truth

Ground-truth line list for assessing the pipeline's **frequency and (especially)
uncertainty** estimates on fixture [1512](../1512/README.md).

## What 1512 contains

Vinyl cyanide (CH2=CH-CN, acrylonitrile), v=0, with hyperfine structure
resolved / partially resolved, **plus several other species** — some with
ground-truth data, some without. The validation is therefore **not** "match
every line in the spectrum." It is: for the lines that can be **unambiguously
attributed to vinyl cyanide**, how accurate are our fitted frequencies, and how
honest are our reported uncertainties?

## Source

`c053515_hfs.cat` — an SPCAT prediction for species tag **53515** (vinyl
cyanide), hyperfine-resolved. Lines **178–292** (115 transitions) are the subset
falling inside 1512's active band (26500–40000 MHz; actual span
**27520.004 – 39837.433 MHz**). All 115 carry tag 53515 (verified).

## Catalog format (verified, JPL/SPCAT fixed columns)

| field | columns (1-indexed) | Fortran | notes |
|---|---|---|---|
| frequency (MHz) | 1–13 | F13.4 | |
| uncertainty (MHz) | 14–21 | **F8.4** | width 8 — **not** 6.4f; value is right-justified with leading spaces |
| log10 intensity | 22–29 | F8.4 | nm²·MHz at 300 K |
| DR | 30–31 | I2 | degrees of freedom |
| E_low (cm⁻¹) | 32–41 | F10.4 | |
| g_up | 42–44 | I3 | |
| tag | 45–51 | I7 | 53515 |
| QN format | 52–55 | I4 | |
| quantum numbers | 56–79 | 12×I2 | 6 upper + 6 lower (J, Ka, Kc, F, …) |

Parse the first two fields as `float(line[0:13])` and `float(line[13:21])`.

## `lines.csv`

Columns: `freq_mhz, unc_mhz, log_intensity, e_low_cm-1, g_up, quantum_numbers`.
`quantum_numbers` is the verbatim catalog QN field (cols 56–79, six 2-char
upper + six 2-char lower). Catalog uncertainties are predicted, 0.0001–0.0021
MHz (0.1–2.1 kHz) — effectively exact relative to FTMW measurement precision, so
they serve as the reference for scoring our frequency accuracy.

## Caveats for the comparison

- **Hyperfine blends.** Some components are closely spaced (e.g. 27520.0037 and
  27520.0072 MHz — 3.5 kHz apart, far below resolution); these will not be
  separable and should be treated as blends, not individual targets.
- **Observability.** Weaker catalog lines (low `log_intensity`) may be below the
  noise floor or buried under other species; only observed, unblended,
  vinyl-cyanide-attributable lines enter the accuracy/uncertainty assessment.
- Disambiguation (which catalog lines are cleanly observed) is part of the test,
  to be settled in the fitting-test session, not pre-baked here.

## Multi-species catalogs (shared by 1512 and 655)

This directory is the vinyl-cyanide molecular ground truth for **all** VyCN
fixtures (1512 and the high-SNR 655). `lines.csv` above is the v=0 main
isotopologue only; the high-SNR 655 work resolves additional species.

`catalogs/` holds the raw SPCAT `.cat` source files (one per species); parse the
fixed columns as documented above. `combined_lines.csv` is the parsed in-band
(26500–40000 MHz) union of all five species — columns
`species,tag,predicted,freq_mhz,unc_mhz,log_intensity,qn` (328 lines):

| file | tag | species | in-band lines |
|---|---|---|---|
| `c053515_hfs.cat` | 53515 | v=0 main | 115 |
| `c054506_hfs.cat` | 54506 | ¹³C (a) | 56 |
| `c054507_hfs.cat` | 54507 | ¹³C (b) | 56 |
| `c054508_hfs.cat` | 54508 | ¹³C (c) | 56 |
| `c054509.cat` | 54509 | ¹⁵N (hfs from ¹⁴N absent) | 45 |

**Tier by predicted uncertainty.** The main and ¹³C catalogs are
calibration-grade (`unc_mhz` 0.1–2.1 kHz). The ¹⁵N catalog is mixed: some lines
are fit-quality (sub-kHz) but others are purely predicted with uncertainties up
to ~22 MHz (the `predicted` column / a negative raw tag flags these) — exclude
anything with `unc_mhz` above a few kHz from frequency/uncertainty scoring; it can
still serve for line attribution/exclusion.
