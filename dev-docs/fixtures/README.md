# Checked-in fixtures

Real Blackchirp experiments committed under `examples/blackchirp_data/<id>/`
for tests, manual runs, and cross-fixture validation. Each is stored stripped to
the primary FID frame only (backup frame CSVs + their `fidparams.csv` rows +
overlay spectra removed) to keep the repo light; the index-0 data is unchanged.

All current fixtures are from the **same instrument** (downconversion LO/probe
40960 MHz, lower sideband; AWG chirp sweeping 4895 → 1520 MHz IF, i.e. RF
36065 - 39440 MHz; digitizer 50 GS/s, 750000 points, 15 µs record; analysis band
26500 - 40000 MHz). They differ in sample and in chirp duration. A
*different-instrument* fixture and a *longer-acquisition* fixture (to separate
τ_L from τ_G) are still outstanding — see issues #5 and #6.

| fixture | chirp | shots | recommended `start_us` | ground truth | doc |
|---|---|---|---|---|---|
| 2638 | 1 µs | 501400 | 2.35 | manual window oracle | (primary reference; see `CLAUDE.md`) |
| 1019 | 3 µs | 1590380 | 4.35 | yes (some peak frequencies) | [1019.md](1019.md) |
| 1512 | 2 µs | 17860 | 3.35 | yes (vinyl cyanide; [catalog](1512-vinyl-cyanide-truth/README.md)) | [1512.md](1512.md) |
| 1231 | 3 µs | 74740 | 4.35 | no | [1231.md](1231.md) |

`start_us` is data-derived (`ftmwpipeline detect-start`, ≈ chirp_duration +
1.35 µs — the chirp end plus the switch-bounce ringdown). Recommended FT for
new analyses: `zpf=2`, `expf_us=None` (unapodized), `trim=(26500, 40000)`,
then `calibrate_tau` before peak detection.
