# Checked-in fixtures

Real Blackchirp experiments committed under `examples/blackchirp_data/<id>/`
for tests, manual runs, and cross-fixture validation.

**Backup FID frames are retained** where they exist. The progressive backup
frames (cumulative averages at increasing shot counts) are load-bearing
reproducibility artifacts for the noise-estimation research
([`../research/noise-snr-scaling/report.md`](../research/noise-snr-scaling/report.md)):
the 1/√N noise-scaling test and the frame-difference noise reference require them.
The frame CSVs are committed directly as plain blobs (the data is immutable and
compresses in the packfile, and plain blobs keep archives/tarballs reproducible
without extra tooling). Overlay/peakfind artifacts are not committed. (Earlier
fixtures were once stripped to the primary frame; 1019/2638 have since been
restored from `~/Downloads` and git history. 1512/1231 were acquired without
backups — single frame only.)

Fixtures are from the **same instrument** (downconversion LO/probe 40960 MHz,
lower sideband; AWG chirp sweeping 4895 → 1520 MHz IF, i.e. RF 36065 - 39440 MHz;
digitizer 50 GS/s, 750000 points, 15 µs record; analysis band 26500 - 40000 MHz).
They differ in sample and in chirp duration. A *different-instrument* fixture and
a *longer-acquisition* fixture (to separate τ_L from τ_G) are still outstanding —
see issues #5 and #6.

| fixture | chirp | shots | frames | recommended `start_us` | ground truth | doc |
|---|---|---|---|---|---|---|
| 2638 | 1 µs | 501400 | 3 | 2.35 | manual window oracle | (primary reference; see `CLAUDE.md`) |
| 1019 | 3 µs | 1590380 | 5 | 4.35 | yes (some peak frequencies) | [1019.md](1019.md) |
| 1512 | 2 µs | 17860 | 1 | 3.35 | yes (vinyl cyanide; [catalog](1512-vinyl-cyanide-truth/README.md)) | [1512.md](1512.md) |
| 1231 | 3 µs | 74740 | 1 | 4.35 | no | [1231.md](1231.md) |
| 655 | 2 µs | 2133080 | 6 | 3.35 | yes (vinyl cyanide, high SNR) | [655.md](655.md) |
| 360 | — | 775860 | 3 | 4.35 | no (MTBE sample) | [360.md](360.md) |
| 363 | — | 525660 | 2 | 4.35 | no (MTBE sample) | [363.md](363.md) |

`start_us` is data-derived (`ftmwpipeline detect-start`, ≈ chirp_duration +
1.35 µs — the chirp end plus the switch-bounce ringdown). Recommended FT for
new analyses: **no zero-padding** (`zpf=0`), `expf_us=None` (unapodized),
`trim=(26500, 40000)`, then `calibrate_tau` before peak detection. The pipeline
runs on the raw (un-padded) FT; zero-padding interpolates the bins and corrupts
the Stage 2/5 statistics, so a suggested `zpf` is never adopted. Stage 3 peak
detection applies its own internal zero-padding for position-finding only.
