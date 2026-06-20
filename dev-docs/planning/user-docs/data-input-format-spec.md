# Data input format specification

Design contract for the user-facing "bring your own data" surface: the formats a
user can shape arbitrary FTMW data into so that `ftmwpipeline` can import it
**without writing Python**, including how clock declarations are supplied.

Status: **implemented.** The layout below was signed off and built: the
`ftmw-hdf5` and `csv` loaders, the `--metadata` sidecar, and the `clocks`
declaration surface (CLI / `api` / `Pipeline`) all ship, with unit and
cross-interface tests. The remaining work is the user-facing Sphinx pages (the
Stage 0 page and a dedicated input-format reference page) that present this
contract to users. The resolved decisions are recorded in
[Resolved decisions](#resolved-decisions).

## Motivation and current state

`ftmwpipeline` ingests data through a loader registry
(`io/data_loaders/`, `BaseLoader` + `register_loader`). Two extension paths
were intended:

- **Path A — write a loader.** Subclass `BaseLoader`, implement
  `can_load` / `validate_source` / `load_fid` / `get_required_parameters` /
  `get_optional_parameters`, and `register_loader(...)`. This works today
  (`blackchirp`, `keysight-mat`) and stays the answer for proprietary binary
  formats. It requires Python.
- **Path B — shape data into a generic format.** The `csv` and `hdf5` loaders
  were meant to be the no-code path. **Both are non-functional stubs**: their
  `load_fid` raises `LoaderError("...not yet implemented...")`. So today there
  is no no-code import path.

Separately, **clock declarations** (the instrument clock fundamentals the Stage 5
spur gate uses to build its lattice prior) have a real persisted schema —
`ClockSource(freq_mhz: float, locked: bool, label: str)`, stored at
`/stage0_fid_data@recommended_clock_sources` — but the only ways to populate it
are a loader injecting `fid.metadata["clock_sources"]` (Blackchirp auto-extracts
from `clocks.csv`; Keysight-MAT injects interleave combs) or the ephemeral
`run --clocks` flag (frequencies only, no `locked`/`label`, not persisted as the
real schema). A custom-data user has no way to declare clocks.

This spec closes both gaps with a no-code path.

## Scope

In scope — a user can, without writing Python:

1. Import a **ready-to-process single (averaged) FID** from a documented
   **native HDF5** layout that embeds all acquisition metadata and, optionally,
   clock declarations.
2. Import the same FID from a minimal **CSV** of samples plus metadata supplied
   out-of-band.
3. Supply acquisition metadata and clock declarations through a **sidecar
   metadata file** that any loader (CSV included) consults.
4. Declare or edit clock sources on an **already-imported `.ftmw` file** through
   a new CLI/API surface, independent of the loader.

Out of scope (handled by dedicated loaders, documented on the Advanced
*scope-record import* page, **not** by the generic formats):

- Raw segmented oscilloscope records (pre-record + repeated frames), ADC
  interleave-offset cleanup, coherent frame averaging. These are
  instrument-specific reductions; the generic formats accept the **reduced**
  science FID, not the raw record. The native HDF5 layout deliberately does not
  carry an acquisition-layout/segmentation block.

## The FID contract every path must satisfy

All paths ultimately construct one `core.data_structures.FID`. The fields a
user must (or may) provide map to its constructor:

| FID field | Meaning | Required? | Default |
| --- | --- | --- | --- |
| `data` | real-valued time-domain voltage samples, 1-D | **yes** | — |
| `spacing` | sample period, **seconds** (user supplies `spacing_us`, µs) | **yes** | — |
| `probe_freq_mhz` | probe/LO frequency, MHz. Direct sampler → `0.0` | **yes** | — |
| `sideband` | `"upper"` or `"lower"` | no | `"upper"` |
| `shots` | number of averaged shots (provenance only) | no | `1` |
| `processing.rdc` | remove DC (subtract mean) before FT | no | `true` |

Frequency convention (from `FID.apply_molecular_frequency`): upper sideband →
`molecular = probe + baseband`; lower → `molecular = probe − baseband`. A
direct-sampling instrument sets `probe_freq_mhz = 0.0`, `sideband = "upper"`, so
`molecular = baseband` and the recorded frequency *is* the molecular frequency.

Optional import-time hints that may accompany any path (stored in
`fid.metadata`, consumed at Stage 0):

- **Chirp window** → recommended FID start. Fields mirror `ChirpWindow`:
  `chirp_end_us` (required to use this), optional `chirp_start_us`,
  `start_margin_us`. When present, the recommended start is
  `chirp_end_us + start_margin_us` and the start detector runs only as a
  cross-check.
- **Clock sources** → `recommended_clock_sources`. A list of
  `{freq_mhz, locked, label}` (see [Clock declarations](#clock-declarations)).

## Format 1 — native HDF5 (`ftmw-hdf5`)

The canonical no-code path: one self-describing HDF5 file the user writes with
h5py, MATLAB (`-v7.3`), or any HDF5 tool. Everything travels in one file; no
flags required.

```
mydata.h5
  attrs:
    ftmw_input_version  int      = 1          # format marker + version (REQUIRED)
    spacing_us          float64  = 0.02        # REQUIRED
    probe_freq_mhz      float64  = 40960.0     # REQUIRED (0.0 for direct sampler)
    sideband            string   = "lower"     # optional, default "upper"
    shots               int      = 100         # optional, default 1
    rdc                 bool/int = 1           # optional, default true
    chirp_end_us        float64  = 1.5         # optional (enables recommended start)
    chirp_start_us      float64  = 0.5         # optional
    start_margin_us     float64  = 0.5         # optional
    description         string   = "..."       # optional provenance, free text
  /fid                  float64[N]             # REQUIRED: 1-D real voltage samples
  /clock_sources/                              # OPTIONAL group (see below)
    freq_mhz            float64[K]
    locked             int8[K]                 # 1 = referenced/locked, 0 = free-running
    label              string[K]               # variable-length UTF-8
```

Rules:

- **Detection (`can_load`).** HDF5-openable **and** root attribute
  `ftmw_input_version` present. This is unambiguous against arbitrary HDF5/`.mat`
  files; this loader registers before any broader HDF5 matcher.
- **`/fid`** is cast to 1-D float64. Complex input is rejected with a clear
  error (the pipeline FT is a real FFT).
- **Attributes vs datasets.** Scalars are root attributes; only `/fid` and the
  optional `/clock_sources` group are datasets. HDF5 string attributes may be
  fixed- or variable-length UTF-8; both are accepted.
- **`/clock_sources`** is three equal-length parallel 1-D datasets (not a
  compound dtype — parallel datasets are far easier to emit from MATLAB).
  Omit the group entirely to declare no clocks. `label` may be empty strings.
- **Versioning.** `ftmw_input_version` is the compatibility gate; unknown
  (higher) versions fail with an explicit "unsupported input version" error
  rather than silently mis-parsing.
- **Validation** (`validate_source`) reports `n_points`, `duration_us`, the
  resolved acquisition parameters, and the clock-source count, with actionable
  errors for missing required attributes.

### `.mat` collision note

MATLAB `-v7.3` files are HDF5 and use the `.mat` extension; the `keysight-mat`
loader already claims `.mat` and gates on `Channel_*/XInc`. A native-input file
written from MATLAB should use `.h5` (or any non-`.mat` extension) so detection
is unambiguous, **or** be imported with an explicit `--format ftmw-hdf5`. The
`ftmw_input_version` attribute distinguishes it regardless; the extension only
affects auto-detection ordering.

## Format 2 — CSV (`csv`)

The minimal path for data that is already a column of numbers. The CSV carries
**samples only**; acquisition metadata and clocks come from flags and/or the
sidecar.

- **Layout.** One or more columns, optional header row. The voltage column is
  selected with `column` — an integer index (0-based) **or** a column name
  (requires a header row). Omitted → first column. Non-selected columns are
  ignored.
- **Required metadata** (not in the file): `spacing_us`, `probe_freq_mhz`.
  Supplied by CLI flags (`--spacing_us`, `--probe_freq_mhz`) or the sidecar.
- **Optional metadata:** `sideband` (default `"upper"`), `shots` (default `1`),
  `rdc` (default `true`), the chirp-window hints, and clock declarations — all
  via flags or sidecar.
- **Clocks in CSV.** A single-column CSV cannot embed clocks; a CSV user
  declares clocks through the sidecar or the `clocks` CLI on the imported file.

## Format 3 — sidecar metadata file (`--metadata`)

A companion JSON or YAML file any loader consults, so formats that cannot embed
metadata (CSV) — or users who prefer to keep acquisition parameters in
human-readable text next to the data — have a no-code home for it, including the
full clock-declaration schema.

```jsonc
// mydata.ftmwmeta.json  (or .yaml)
{
  "spacing_us": 0.02,
  "probe_freq_mhz": 40960.0,
  "sideband": "lower",
  "shots": 100,
  "rdc": true,
  "chirp_window": { "chirp_end_us": 1.5, "start_margin_us": 0.5 },
  "clock_sources": [
    { "freq_mhz": 5760.0, "locked": true,  "label": "synth" },
    { "freq_mhz": 6250.0, "locked": false, "label": "digitizer" }
  ]
}
```

- **Discovery.** Explicit `--metadata <path>`, or auto-discovered as
  `<source>.ftmwmeta.json` / `.yaml` adjacent to the source.
- **Applicability.** The sidecar augments whichever loader runs. A native-HDF5
  file that already carries everything does not need one; a sidecar may still
  override (see precedence).
- **Keys** are exactly the FID-contract fields plus `chirp_window` and
  `clock_sources`, all optional.

### Metadata precedence

When the same field is set in more than one place, highest wins:

```
explicit CLI flag / call kwarg  >  sidecar file  >  embedded in the data file  >  built-in default
```

This mirrors the pipeline's settings-resolution philosophy (explicit always
wins; the file's own embedded value beats only the hard default). Clock
declarations follow the same rule, then enter the existing Stage 5 resolver
chain (`explicit > persisted > preset > recommended > default`) as the
**recommended** layer.

## Clock declarations

One schema everywhere — the existing `ClockSource`:

| field | type | meaning |
| --- | --- | --- |
| `freq_mhz` | float | chain **fundamental** in MHz (declare 5760, not the 11520/40960 products — harmonics come for free) |
| `locked` | bool | `true` = referenced to the frequency standard (spans the intermod lattice); `false` = free-running (predicts a drifting tone family) |
| `label` | string | human-readable identifier; informational |

Three no-code entry points, all writing the same persisted
`recommended_clock_sources` record:

1. **Embedded** in a native-HDF5 file (`/clock_sources` group).
2. **Sidecar** `clock_sources` array.
3. **CLI/API on an existing `.ftmw`** (below) — for declaring clocks after
   import, or for any-format data where the user did neither of the above.

### `clocks` CLI (new meta-object)

Object-verb grammar, alongside `settings` / `scan`:

```
ftmwpipeline clocks show  exp.ftmw
ftmwpipeline clocks set   exp.ftmw  5760:locked:synth  6250:free:digitizer   # replaces
ftmwpipeline clocks add   exp.ftmw  16000:locked:awg                          # appends
ftmwpipeline clocks remove exp.ftmw 6250                                      # by frequency
ftmwpipeline clocks clear exp.ftmw
```

- Token grammar `freq[:locked|free[:label]]` (default `locked`, empty label).
- Writes the **recommended** declaration layer
  (`/stage0_fid_data@recommended_clock_sources`), not the persisted Stage 5
  layer — the declaration is a recommendation the resolver still ranks below an
  explicit `--clocks` and below persisted Stage 5 settings, preserving D11
  reproducibility.
- `clocks set/add/remove/clear` mutate Stage 0 metadata only; they do not
  invalidate downstream stages, but `clocks show` notes when a completed Stage 5
  fit predates the current declaration.

### API / Pipeline surface

Per the dual-interface rule, the same operation on all three interfaces,
delegating to one `_internal` implementation:

- functional: `ftmw.set_clock_sources(path, clocks, *, replace=True)`,
  `ftmw.get_clock_sources(path)`
- OO: `Pipeline.set_clock_sources(...)`, `Pipeline.get_clock_sources()`
- CLI: the `clocks` object above
- a cross-interface consistency test asserts identical results

## Implementation (shipped)

Built smallest-first per the `CLAUDE.md` checklist:

1. **Shared input-metadata resolver** — `io/input_metadata.py`: sidecar
   discovery/parse (JSON + YAML) and the per-field `explicit > sidecar >
   embedded > default` merge, returning a `ResolvedInputMetadata`. Used by both
   generic loaders.
2. **`ftmw-hdf5` loader** — `io/data_loaders/ftmw_hdf5.py` (`FtmwHdf5Loader`),
   detecting on the `ftmw_input_version` attribute; the dead `hdf5` stub was
   removed and the registry now lists `ftmw-hdf5`.
3. **`csv` loader** — `io/data_loaders/csv.py`, with the `column` selector
   (index or name) and shared-resolver metadata.
4. **Sidecar** — `--metadata` flag on `data import`, with
   `<source>.ftmwmeta.{json,yaml,yml}` auto-discovery.
5. **`clocks` surface** — `_internal/clocks_impl.py`, `cli/clocks_commands.py`
   (`show`/`set`/`add`/`remove`/`clear`), and `api` / `Pipeline`
   `get`/`set`/`remove`/`clear_clock_sources`, all writing the recommended
   declaration layer.
6. **Tests** — `tests/unit/io/test_generic_input_loaders.py` (round-trip,
   column selection, sidecar precedence, error cases) and
   `tests/integration/test_clocks_cross_interface.py` (CLI = api = Pipeline).
7. **American-English scan** of the new help/log/error strings (clean).

Remaining: the user-facing Sphinx pages — a dedicated input-format reference
page plus the Stage 0 page linking to it; the Advanced scope-record page stays
the home for raw-record loaders.

## Resolved decisions

Signed off; the spec above reflects them.

1. **Native-HDF5 format name** → `ftmw-hdf5`; the dead generic `hdf5` stub
   (never-built `/fid_data/voltage_data` layout) is **retired**.
2. **CSV column selection** → multi-column files are allowed; a `column` option
   (integer index or, with a header row, a column name; default first column)
   selects the voltage column. The `time,voltage` spacing-derivation form
   remains deferred — `spacing_us` always comes from a flag/sidecar in v1.
3. **Sidecar** → `<source>.ftmwmeta.json` / `.yaml` auto-discovery plus explicit
   `--metadata`; JSON and YAML both supported (`pyyaml>=6.0` already a runtime
   dependency).
4. **`clocks`** → a top-level meta-object, symmetric with `settings`/`scan`.
