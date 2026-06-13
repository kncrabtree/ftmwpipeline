# Scope-record import — segmented raw acquisitions

Status: **implemented** (loader, layout, segment persistence, interleave
cleanup, chirp-response spur lane — all three interfaces + tests);
**end-to-end fixture validation pending** (stages 0–5 on the succinimide
record: ringing-aware start detection, defaults portability, catalog
scoring). Implementation notes beyond the design below:

- The keysight-mat loader reproduces the bench frame average exactly and,
  with `interleave_factors=[16, 512]`, the bench-corrected average to
  float roundoff (the bench's mod-512 reference pattern is the *total*
  per-phase means; the sequential-residual application is equivalent).
- Interleave patterns persist in the `acquisition_segments` group (LSB
  units, estimated pre-volt-scaling) and the derived fs/M clocks ride the
  recommended-clock-sources layer (`interleave_m16` 8000 MHz,
  `interleave_m512` 250 MHz, locked).
- The chirp-response probe carries a comb-exclusion guard, found by
  measurement: the cleanup estimates its pattern on the pre-record and
  therefore nulls the fs/M combs there exactly — the stored pre-record's
  silence at those frequencies is manufactured, and an unguarded probe
  read the 16 GHz ADC image as "protect". On cleanup-comb frequencies the
  probe returns inconclusive and the lattice/narrowness lanes decide.

Driving fixture: `Succinimide_10.mat` (Keysight
UXR0204A, 128 GSa/s direct sampling, 8–18 GHz; bench characterization in
`scratch/succinimide/`). Part of the cross-instrument arc (ROADMAP
priority 3) together with
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)
(defaults portability) and the clock-declaration work
([`instrument-clock-declaration.md`](instrument-clock-declaration.md)).

## Problem

Direct-sampled scope acquisitions arrive as one long record: a quiet
pre-record, N chirp–FID frames at a fixed period, and a dead tail
(fixture: ~12.5 µs + 19 × 20 µs + ~13.75 µs at 7.8125 ps/sample). The
hardware did not average; the segmentation is operator knowledge, not
container metadata. The existing loaders (`blackchirp`, `csv`, `hdf5`)
accept only a single FID record.

The governing decision rule: **any transform that changes the science
record happens in-pipeline so provenance captures it** (frame averaging,
interleave-offset cleanup); offline preprocessing into a bare averaged
FID remains possible through the existing loaders but loses provenance
*and* the diagnostic segments. The pre-record is pipeline payload, not
packaging: the chirp-response spur anchor measured on the fixture
(scripts `scratch/succinimide/chirpoff_anchor.py`, `frame_coherence.py`,
`prerecord_flatness.py`) gives pre-record/FID band-power ratios of
0.59–1.68 for known interference (7 of 8 catalogued tones ≥ 0.9; clock
tones 0.81–1.01) vs ≤ 0.28 for the brightest chirp-responsive lines
(consistent with e^{-Δt/τ} bleed from the previous chirp at 20 µs
pulsing), with an ambiguous 0.50–0.74 band. Frame coherence and
in-pre-record flatness were both measured non-discriminating (every tone
is frame-coherent at the 50 kHz frame grid; amplitude modulation swamps
the flatness test) — the pre-record ratio is the only working
time-domain anchor on this instrument, and it requires the segment to
survive import.

## Design

Three layers with deliberately different generality.

### Layer 1 — thin vendor loader (`keysight-mat`)

MATLAB v7.3 (= HDF5, h5py-readable): `Channel_N/Data` (int16),
`XInc`/`XOrg` (sample clock), `YInc`/`YOrg` (vertical scaling),
`Frame/Model` + `Frame/Serial` (instrument identity, uint16-encoded
strings). `can_load` keys on the `.mat` extension + HDF5 signature +
`Channel_*/XInc`. Registered via the existing `BaseLoader` /
`register_loader` registry; channel selection is the only loader-level
parameter (default: the sole channel present). No ambition beyond
extracting the raw record and its sample clock; other vendors' formats
are future sibling loaders, not extensions of this one.

### Layer 2 — acquisition layout (loader-independent, the general piece)

A shared import step parameterized by what the operator knows:

- `pre_record_us`, `frame_period_us`, `n_frames` — the segment map;
- `frame` — `"avg"` (default: coherent mean of all frames) or an integer
  (import that single frame as the science FID, for per-frame /
  time-dependent analysis; one `.ftmw` per frame, batchable — mirrors
  the Blackchirp convention of one CSV per frame);
- `keep_frames` — default **False** (discard per-frame data after
  averaging); opt-in retention of the full frames dataset for future
  per-frame statistics.

Behavior: slice the record per the map; coherent mean (or single-frame
selection) → Stage 0 FID; persist the segment map plus the **pre-record
and tail segments** unconditionally (small; diagnostic payload). Plain
mean only — no robust averaging or frame rejection until a fixture
demands it. An autocorrelation check of the record against
`frame_period_us` is a validation assist (warn on mismatch), not a
substitute for the declared layout.

### Layer 3 — interleave-offset cleanup (declaration-keyed, opt-in)

The mod-M Stage-0 hook already motivated by the bench work: per-phase
means estimated on the pre-record quiet segment (reshape order `'F'`,
column means for M = 16 then the residual M = 512 layer), tiled
subtraction over the **full record before slicing/averaging**. The M
values come from the instrument clock declaration (interleave fields);
applied only when declared, recorded in provenance. Known limit, by
measurement: the 250 MHz comb is only reduced ~3–4× (partly
signal-path) — the clock-lattice gate owns the remainder.

### Declaration extension

The instrument declaration gains an acquisition-layout block (segment
map, chirp window within the frame, interleave M values) as the
recommended layer; explicit import parameters always win. Mirrors the
Blackchirp metadata auto-population precedent. The chirp window also
feeds ringing-aware start detection: this instrument needs the FID start
~3 µs after chirp end (chamber ringing to ~6.5 µs), far beyond the home
instrument's 0.67 µs — a declaration-provided start offset is the
minimal mechanism; start-detector robustness work stays out of scope
here.

### Spur lane — chirp-response anchor

A new probe built from the stored pre-record segment: band-power ratio
(pre-record vs FID window) at the nominee frequency, three-way verdict —
ratio ≥ hi ⇒ interference (gate), ≤ lo ⇒ chirp-responsive (protect),
middle ⇒ inconclusive, fall through to the other lanes (clock lattice,
cross-record recurrence, decay probe). Starting thresholds from the
fixture measurements: hi = 0.8, lo = 0.3; the pre-record's detection
floor is ~√N worse than the N-frame science average (≈ 5× here), so the
probe only arbitrates nominees bright enough to see pre-record — weaker
ones rely on the other lanes. Calibration of the ambiguous band awaits
the succinimide catalog (user-delivered, simulated into 8–18 GHz from
the group's 26.5–40 GHz constants); the six unidentified non-comb tones
at 9.2974 / 10.0774 / 14.9248 / 8.4762 / 16.9145 / 12.1328 GHz are the
test cases. A dedicated chirp-off diagnostic record (long, normal
TWTA/switch sequence, chirp amplitude zeroed) upgrades the probe to a
clean presence test when it arrives; the design must not depend on it.

## Serialization

Stage 0 gains acquisition-segment datasets alongside the FID: segment
map (attributes), `pre_record`, `tail`, and `frames` only under
`keep_frames`. `SourceMetadata` covers the source hash plus the layout
parameters, frame selection, and whether/with which M the offset cleanup
ran. Re-import with identical source and parameters stays a safe no-op;
re-import with a different frame selection or layout changes the Stage 0
data and follows the existing different-import path (downstream stages
invalidate; `force` semantics — open question below).

## Interface surface

Per the dual-interface rule, one `_internal` implementation surfaced
three ways:

- CLI: `data import --source X.mat --pre-record-us 12.5
  --frame-period-us 20 --n-frames 19 [--frame K] [--keep-frames]`, with
  declaration auto-population when the layout block is present;
  `data show` surfaces the segment map.
- `Pipeline.create(...)` / `api.import_data(...)`: the same layout
  parameters.
- Cross-interface consistency test alongside the existing suite.

## Test plan

- Unit: loader on a synthetic miniature `.mat` written with h5py;
  layout slicing/averaging on synthetic frames (known tone: average
  improves SNR by √N; single-frame selection picks the right slice);
  offset cleanup recovers and removes a synthetic mod-M pattern;
  segment-map and segment round-trip through the `.ftmw`.
- Integration: import of the real fixture is local-only (file is too
  large to check in); a truncated subset (pre-record + first frames,
  int16) as a checked-in example is an open question below.
- The chirp-response probe gets unit tests on synthetic
  pre-record/FID pairs (flat tone ⇒ gate; decaying bleed ⇒ protect;
  middle ⇒ fall-through).

## Out of scope

General DSP preprocessing (filtering, decimation, resampling),
multi-channel records, robust frame averaging / outlier rejection,
automatic frame-structure detection (assist only), and additional vendor
formats.

## Open questions

1. Re-import force semantics when only the frame selection or layout
   changes over an existing file (same source hash).
2. Checked-in CI fixture: truncated succinimide subset (~10–20 MB)
   under `examples/`, or keep the real-file test local-only.
3. **Resolved (implemented):** the chirp window is a declared record
   (`recommended_chirp_window` attr on `stage0_fid_data`:
   `chirp_start_us` / `chirp_end_us` / `start_margin_us`, record time
   base). The keysight-mat import takes the three values as explicit
   parameters (`--chirp-start-us` / `--chirp-end-us`
   / `--start-margin-us`); the Blackchirp loader derives them from the
   experiment config (`ChirpConfig PreGate + PreProtection` for the
   chirp start, `chirps.csv` durations for the end). At import a
   declared `chirp_end_us` stamps `recommended_processing.start_us =
   chirp_end + margin` unless the loader already recorded an
   experimenter start (Blackchirp `FidStartUs` outranks). Start
   detection treats a declared chirp end as authoritative — the sweep
   detector runs as a cross-check and warns on disagreement — closing
   the high-SNR chirp-end overshoot (issue #34: 655 recommends 3.27 µs
   against the detector's 4.22 µs overshoot; the succinimide record
   imports hands-off to start 7.0 µs). The chirp-response probe's
   FID-window definition can consume the same record (future wiring).
4. Succinimide catalog (user-delivered) → settles the ambiguous-band
   thresholds and classifies the six unidentified tones.
