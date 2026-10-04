# Specification: Machine Contract

Status of this document: **DRAFT normative specification**. It states
requirements, not current implementation state. Sections marked *(outline)* are
not yet complete enough to implement against; open questions are listed at the
end.

## Scope

The **machine contract** is the published, versioned surface that a program —
a script, a notebook helper, or a front end such as BlackQuill — may rely on
without reading prose, log text, private modules, or the HDF5 layout. It is
the same surface interactive users already have; this document adds the
guarantees a program needs and says what is *not* part of the contract.

It does not add a fourth interface. Every contract element is implemented once,
in the shared internal layer, and exposed identically through the functional
API, the `Pipeline` class, and the CLI (see the dual-interface rule in
[`API_STRATEGY.md`](API_STRATEGY.md) and [`CLI_STRATEGY.md`](CLI_STRATEGY.md)).
Interactive behavior is unchanged by anything here.

## Principles

1. **Declared, not discovered.** A program may rely on exactly what this
   contract lists. Anything else — the wording of a log line or error message,
   a module under `_internal`, an HDF5 group or attribute name, a dict key not
   listed here — may change without notice.
2. **Derived state is read, never re-derived.** Any value the pipeline derives
   (a calibrated frequency, a tolerance, a threshold, a line width) is offered
   through an accessor. A client must never have to recompute it from inputs,
   and never needs to read the file layout to obtain it.
3. **One answer per question.** Where two surfaces report the same quantity,
   they report the same value. An accessor that names the value another call
   applies (for example the display unit) is defined *as* that value.
4. **Typed, not textual.** Conditions a program must branch on — a failure
   class, a progress step, a warning — arrive as types, codes, and fields,
   never only as text.
5. **Versioned and additive.** The contract carries a version. Additions are
   backward compatible; anything else is a new version.

## Versioning and stability

- The package exposes a single integer **contract version**,
  `ftmwpipeline.CONTRACT_VERSION`, and a `capabilities()` call (*outline*,
  §Capabilities). A client gates on the contract version, never on
  `__version__` (every development build shares one `__version__`).
- Every machine-readable payload carries a **schema name** of the form
  `ftmw/<payload>@<n>` (for example `ftmw/final_products@1`). A schema name
  never changes meaning; an incompatible payload gets a new `<n>`.
- **Additive change** — a new accessor, a new optional field, a new code —
  raises `CONTRACT_VERSION` and keeps every existing schema valid. The first
  published contract is version 1; each release that adds or (before 1.0.0)
  changes contract elements raises it by one, so a client can gate on the
  version as well as on `capabilities()`.
- **Breaking change** — removing or renaming a field, changing a field's type
  or meaning, changing a definition — requires a new schema name, and the old
  one remains available until its retirement is announced.
- **Not contract**, regardless of how stable it has been in practice: log
  message text and templates, exception message text, the HDF5 group/attribute
  layout of `.ftmw`, anything under `ftmwpipeline._internal`, and any output
  not enumerated by this document.
- **Before 1.0.0** the contract may still change incompatibly, and existing
  result fields are migrated to it (the missing-value rule, typed errors)
  rather than grandfathered. From 1.0.0 the rules above are binding.
- **Retiring a path a client relies on today** (log templates, the drift-line
  prefix, direct HDF5 reads) happens only after the client confirms its
  migration has shipped. Until then the old path keeps working.
- The contract is enumerated in code (a manifest of accessors, schema names,
  codes, and declared keys/columns), so a test can assert that every declared
  element exists on all three interfaces and that nothing declared silently
  disappears.

## File model

- A `.ftmw` file is the unit of state. The contract does not include any state
  held outside it, other than an explicit `ReviewSession`.
- **Every write replaces the file.** Each stage run and each curation write
  ends by rewriting the file into a new file and atomically replacing the
  original, so the file's inode changes on every write. A client must not hold
  an open handle (h5py or otherwise) across contract calls.
- **Single writer.** There is no file locking. At most one writer may act on a
  file at a time; serializing writers is the client's responsibility. Reads
  concurrent with a write may observe either the old or the new file, never a
  partial one.
- A read never writes. A call documented as a read leaves the file
  byte-for-byte unchanged.

## Missing values

Two distinct meanings of "no value" exist and must survive every surface:

| Meaning | Python | JSON (wire) | Columnar (arrays) |
|---|---|---|---|
| present | the value | the value | value, status `0` |
| **not run** — the stage or quantity does not exist in this file (never computed, never tested) | `Absent.NOT_RUN` | `null` and sibling `"<field>_absent": "not_run"` | `nan` (or the column's fill), status `1` |
| **undefined** — computed, but the quantity has no value (e.g. χ²ᵣ with zero degrees of freedom, a failed K−1 refit) | `Absent.UNDEFINED` | `null` and sibling `"<field>_absent": "undefined"` | `nan`, status `2` |

- `Absent` is a public enum. Contract fields never use `None`, `nan`, `-1`,
  an empty string, or the `"__None__"` storage sentinel to mean either case.
  Existing result fields that encode absence in those ways today (for example
  `knockout_p_value`, `PreviewWindowResult.chi2r_*`, the `-1` fills in
  `read_table`) are migrated before 1.0.0.
- On the wire a field keeps one type (its value type or `null`); the reason
  lives only in the sibling key, which is omitted when the value is present.
- **Non-finite floats.** JSON cannot carry `nan` or `±inf`. A non-finite
  value in a named contract field is written as `null` with sibling
  `"<field>_absent": "undefined"` (it is a computed quantity without a finite
  value); inline in a JSON list it is written as `null`, and the reason is
  carried by the array's status column where one is declared. Arrays written
  as `.npy` keep their `nan` values.
- A key named `"<field>_absent"` is reserved: a payload never contains it
  unless `<field>` is `null` for one of the two reasons above.
- Typed error attributes follow the same rule (an epoch the file never
  recorded is `Absent.NOT_RUN`, not `None`).
- Columnar reads return a `uint8` status column `<column>__status` alongside any
  column that can be absent, using the codes above.
- Unset *settings* are not "absent values": a setting that is unset reads as
  `None` and is documented per setting in the typed registry (§Status and
  settings).

**Migration rules for pre-contract fields** (Wave 3):

- **Storage keeps its encodings.** The file format does not change.
  Conversion happens once, at the contract boundary, through
  `_internal/absence_rules.py`. So a quantity exposed on two surfaces (a
  `FinalPeak` field and its `fit_peaks` column) reports one status.
- **Status codes are derived at read time.**
  - A fill that the reader synthesized because a column predates the file is
    `NOT_RUN`.
  - A stored `nan`, `inf` or sentinel in a present column is classified per
    field. The per-field rules are documented with each table in
    `docs/source/machine_contract.rst`.
  - Knockout: the test did not run (no record, a negative tri-state, or a
    `nan` Δχ²) → every knockout field is `NOT_RUN`. Otherwise each non-finite
    knockout value is `UNDEFINED`. This rule takes precedence over the
    synthesized-fill rule: a test that ran on a file predating a knockout
    column reads that column `UNDEFINED`.
  - An uncertainty that does not exist because its parameter was held fixed
    is `UNDEFINED`, matching `FinalPeak.decay_time_error_us`.
- **Honest uncertainty.** When a line has no statistical frequency error, its
  `sigma_stat_khz` and `sigma_f_khz` are `UNDEFINED`. They are never computed
  with the statistical term silently dropped.
- **`read_metadata`.** The keys of a stage that has not run are omitted, as
  documented, so `.get()` callers keep working. A key that is present without
  a value is `Absent`.
- **Pre-contract return types stay.**
  - `get_final_products` still returns `None` in Python before Stage 6.
  - Its CLI envelope is `{"value": null, "value_absent": "not_run"}`.
  - Dict-shaped accessors (`get_pipeline_info`, `read_metadata`) hold
    `Absent` values instead of `None`.
- **Unchanged.** Storage types (`FittedPeak`, `KnockoutInfo`, the HDF5
  encodings) keep their `Optional`, `nan` and `-1` forms. Settings echoes stay
  `None`.

## Accessors

Each accessor below is a read: it never writes the file, and it is exposed on
the functional API, on `Pipeline`, and through the CLI `read` object with
`--format json`. Through the CLI, an array-valued accessor writes a `.npy`
file to `--output` (self-describing dtype, shape and byte order) and prints the
JSON envelope with the array fields replaced by their file name. Each states what it returns, what it is guaranteed to
equal, and when it is absent.

Rules every accessor follows:

- **One CLI verb per accessor: `ftmwpipeline read <accessor name>`**, spelled
  exactly as the functional-API name, printing the accessor's JSON envelope.
  Verbs that predate the contract (`info`, `review log`, `settings show`, …)
  keep their human output and are not contract.
- **The payload carries its schema in Python too.** A dict payload has a
  `"schema"` key; a dataclass payload declares `__ftmw_schema__`. The API,
  `Pipeline` and the CLI therefore return the same stamped object. Schema
  names are constants in `ftmwpipeline.contract`. Accessors that predate the
  contract keep their Python return types; for them the schema is applied by
  the CLI envelope: a dict or dataclass is stamped directly, a list becomes
  `{"schema", "items": [...]}`, a scalar `{"schema", "value": x}`.
- **Entity tables are lists of records.** An accessor that returns one row per
  window, FID or line returns a list of records (dataclasses or dicts), so an
  absent field travels as `Absent` per row (`null` plus its `_absent` sibling
  on the wire). Numeric series (samples, spectra, models) are arrays and go to
  `.npy` through the CLI. The columnar form of a table, with `<column>__status`
  columns, belongs to `read_table`.
- A refusal names the command to run as a bare CLI verb (`windows run`,
  `data import`), never a full shell line.

### Already present — declared as contract

These exist today; the contract freezes their names and the listed fields.

- `frequency_calibration(path)` → `CalibrationStamp`: calibration state, ε,
  σ_ε, σ floor, probe frequency, sideband — exactly what the final-products
  table applies.
- `refit_snap_tol_mhz(path)` → the curation snap tolerance in MHz for this
  file, and the public constant `REFIT_SNAP_TOL_BINS`.
- `read_metadata(path)`: the declared keys are `file.format_version`,
  `file.completed_stages`, `fid.n_points`, `fid.duration_us`,
  `fid.probe_freq_mhz`, `fid.sideband`, `fid.shots`, `ft.start_us`,
  `ft.end_us`, `ft.trim_min_mhz`, `ft.trim_max_mhz`, `ft.units_power`,
  `ft.acquisition_us` (and their `stage1.` aliases), `stage3.n_peaks`,
  `stage4.n_windows`, `stage5.n_fitted_peaks`, `stage5.acquisition_us`,
  `stage5.shape`, and the `tau.` / `tau_g.` / `timebase.` scalars enumerated in
  the manifest. These are the **stage counts** accessor.
- `read_tables(path)` / `read_table(path, table, columns=...)`: the tables and
  columns enumerated in the manifest (at minimum `fit_peaks`: `window_id`,
  `frequency_mhz`, `decay_rate`, `shape`; `windows`: `window_id`, `freq_min`,
  `freq_max`).
- `settings_defaults()` / `settings_show(path)` and their row fields.
- `get_final_products(path)` and `ReviewSession`, with `FinalPeak`'s identity
  and quality fields declared: `peak_uid`, `window_id`, `origin`,
  `derivation`, `clock_lattice`, and the knockout result
  `knockout_p_value`, `knockout_supported`, `knockout_aicc_delta`. Fields that
  encode absence with `None` today move to `Absent` (each with a notice).
- `review_log(path)` → `DecisionLogEntry` rows with the fields `order_index`,
  `window_id`, `frequency_mhz`, `kind`, `provenance`, `evidence`. Names, types
  and the `kind` / `provenance` vocabularies are frozen, so a client may hash
  the log as an edit-set identity.
- The curation result types (`RefitWindowResult`, `PreviewWindowResult`,
  `AppliedWindowResult`, …) and their `converged` flag.
- `get_pipeline_info(path)` / `info()`: the environment and epoch fields
  `stage_environments`, `last_written_with`, `environment_drift`,
  `runtime_environment_drift`, `current_environment`,
  `environment_acknowledged`, and `warnings` (where an analysis-epoch
  difference is reported). Warnings gain a `code` in the typed-errors phase.
- `compute_display_ft(path, pad_factor=...)` → `ComplexFT`: `freq_array`
  (ascending, in the raw frame, the native active-FT grid Stage 5 fits
  zero-filled to `pad_factor`× its density, from its first to its last bin
  inside the Stage 1 trim, so it contains every active bin), `complex_spectrum`
  aligned to it, and
  `metadata` with `amplitude_scale`, `units_label`, `pad_factor`. Displayed
  magnitude is `abs(complex_spectrum) * amplitude_scale`.

### FID samples

`fid_samples(path)` → `{"samples": 1-D float64 array, "stored_dtype": str}`:
the Stage 0 FID samples, in stored order, with **values equal to the stored
values** — no scaling, windowing or mean removal. The pipeline always stores the
samples as `float64`, so no conversion happens; should a stored dtype ever be
narrower, the values are promoted losslessly to `float64`, and `stored_dtype`
always reports what is on disk. The stored samples are write-once: nothing
after import rewrites them, so this array is stable for the life of the file.

The pipeline does not define a digest of these samples for clients. A client
that needs a spectrum identity hashes this array itself. If the pipeline ever
publishes a digest, it will carry a distinct, versioned name and never be
presented as a replacement for a client-defined identity.

### Analysis fingerprint

`analysis_fingerprint(path)` → `{"schema": "ftmw/analysis_fingerprint@1",
"digest": "<64 hex>"}`.

`ftmw/analysis_fingerprint@1` is **frozen once published**: its definition
never changes. A different definition is published under `@2` alongside it.

**Coverage rule.** The fingerprint covers **every input that affects the
file's scientific output**, as each stage resolved and used it. Two files with
the same FID samples and the same fingerprint must produce the same results
under the same analysis epoch. Concretely, `@1` covers:

- every stage's settings, at the values the stage actually ran with
  (Stages 1–5, including values a stage resolved from a recommended layer,
  such as a Stage 2b shape recommendation or a detected start time);
- the timebase calibration's inputs (its knobs and the clock declaration it
  used);
- the frequency-calibration inputs: the declared `sigma_floor_khz`, and the
  clock declaration the calibration state is derived from
  (`review.calibration_clocks`). The final products derive that state on read,
  through the same rule the timebase uses: a non-empty Stage 5 `spur.clocks`,
  otherwise the recommended declaration. So the hashed value is the
  declaration in effect when the fingerprint is read. These are the only
  Stage 6 inputs that shape the final products:
  - `review run`'s `bar`, `kappa`, `noise_floor` and
    `attention_candidate_evidence` only route attention;
  - the snap tolerance and refit options are arguments of individual curation
    actions, which are excluded below;
- the acquisition parameters the analysis used (probe frequency, sideband,
  sample spacing), including any import-time overrides;
- the stored acquisition segments (`data.acquisition_segments`), which the
  Stage 5 spur gate reads:
  - their layout scalars;
  - the SHA-256 of each array as little-endian float64 in C order;
  - `null` when the import stored none;
- the analysis epoch each persisted stage was produced under.

Every stage key is always present. A stage that has not run contributes
`Absent.NOT_RUN` under its key, so the digest changes as stages complete; a key
is never omitted.

The guarantee runs one way: the same digest means the same results. Different
digests do not imply different results. `@1` hashes every recorded input,
including a knob that a sibling setting makes ineffective in a particular run
(for example `tau.stft.tau_max_factor` when `tau.stft.tau_max_us` is set). It
never infers that a knob had no effect.

**Incomplete provenance.** A file written before a stage persisted everything
it resolved cannot yield a complete fingerprint. The accessor then raises a
typed `incomplete_provenance` error naming the missing inputs (re-running the
stage persists them). It never computes a digest over incomplete inputs.

**Reading the records.**

- A stage is read only when it is **complete** in the stage tracker. An
  incomplete stage is `NOT_RUN`, whatever records remain on disk. A sparse
  record written by `settings set` before a re-run is one example.
- A complete stage raises `incomplete_provenance` when any of these hold:
  - it has no analysis-epoch stamp;
  - one of its records is absent, has no field-set version, or is at an older
    version (`is_pre_provenance`);
  - one of its records is at a **newer** version than this build knows;
  - a field its current version requires is missing (e.g. a version-2 timebase
    record without `clock_sources`).

  The error lists every such input as a dotted canonical key, such as
  `tau.stft`, `timebase.clock_sources` or `ft.analysis_epoch`.
- **`review.sigma_floor_khz`.** When a completed review has no
  `/frequency_calibration` record, the value is `0.0`: no floor was declared,
  and Stage 6 has always applied `0.0` in that case. A record that is present
  but unversioned is pre-provenance and raises, as above.
- **The shape recommendation is not hashed itself.** Its verdict reaches the
  science only through its consumers. Stage 3 records the gap-pass shape and
  decay time it took (`peaks.consumed`). Stage 5 persists the resolved fit
  shape. The fingerprint covers the verdict there, where it was used. The
  recommendation's own record is provenance for display. Its stale epoch stamp
  is never read.
- **Stage 2b.**
  - `tau` and `tau_g` come from each twin's own producer record, never from the
    shared `stage2b_tau` recipe.
  - A twin's Stage 1 inputs are covered by `ft`, because a Stage 1 re-run
    invalidates the twins.
  - So are Stage 2's, because the twins depend on Stage 2.
- **Consumed blocks** are hashed as recorded, under `<stage>.consumed`.
- **`data`** holds the acquisition parameters the analysis read:
  - the probe frequency, sideband and sample spacing;
  - any import-time overrides;
  - `data.analysis_epoch`, the import's stamp.

**Excluded:**

- the FID samples themselves (spectrum identity belongs to the client; see
  §FID samples), and the import selections that only choose which samples
  were stored;
- curation decisions (the review log is public; a client hashes it if it needs
  an edit-set identity);
- write timestamps, preset names, audit attributes;
- the package version and environment stamps. Science-relevant code changes
  are marked by the analysis epoch, which *is* covered. Environment drift is
  reported separately.

**Canonical form.** A JSON object keyed by the canonical stage names
(`data`, `ft`, `noise`, `tau`, `tau_g`, `timebase`, `peaks`, `windows`, `fit`,
`review`), each value an object of that stage's inputs nested by settings group
(`ft.trim_min_mhz`, `fit.peak_survival.vif_collapse_threshold`,
`tau.stft.n_seg`, `timebase.kappa_sys`, `review.sigma_floor_khz`), plus
`<stage>.analysis_epoch`. A stage that has not run is `null` with sibling
`"<stage>_absent": "not_run"`. The acquisition parameters live under `data`.
Keys are sorted by code point at every level; finite floats as the shortest
decimal string that round-trips to the same IEEE-754 binary64 value (as
Python's `repr` produces, e.g. `0.1`, `1e-05`, `25.0`); `-0.0` spelled `-0.0`;
non-finite values as the strings `"nan"`, `"inf"`, `"-inf"`; integers without a
decimal point; booleans as `true`/`false`; tuples as arrays; a setting that was
resolved to *unset* as `null`; structured values (`ShapeSpec`, clock sources)
in their documented typed-JSON form. Strings escape only `"` (as `\"`), `\`
(as `\\`) and U+0000–U+001F (as `\u00xx`, lowercase hex). Every other
character is written literally. UTF-8, no insignificant whitespace. The
digest is SHA-256 over that byte string, lowercase hex.

**Recording rules** (what every stage must persist so the fingerprint can be
computed from the file alone, through each stage's codec, never by walking the
layout):

- **The persisted layer is authoritative.** A stage persists the values it ran
  with, resolved. Once it has, no recommended or detected layer written later
  can change what it is read as having used: a setting it resolved to *unset*
  is recorded as unset and never falls through to another layer.
- **Field-set version.** Every persisted settings record carries the version
  of its field set. A record at the current version has every field present,
  so `None` means "resolved to unset"; a record without a version, or at an
  older version missing fields, is pre-provenance and raises
  `incomplete_provenance` naming the fields.
- **One record per producer.** Each result has its own settings record: the
  Stage 2b Lorentzian (`tau`) and Gaussian (`tau_g`) calibrations and the
  shape recommendation (`tau.recommendation`) never share or overwrite one.
- **Consumed upstream values are recorded by the consumer** under
  `<stage>.consumed`: the values a stage took from another stage's result
  (Stage 3's gap-pass decay time and shape from Stage 2b; Stage 5's per-band
  decay-time anchor from Stage 2b and the ε it used for the spur window from
  the timebase). Re-running the upstream stage does not invalidate the
  consumer; the consumer's record still says what it used.
- **Every completed stage records its analysis epoch.** Recording it is part
  of completing the stage, never best-effort.
- **Only effective knobs are inputs.** A registered setting that cannot change
  the output is a defect to fix, not an input to hash.

### Display units

`display_units(path)` → `{"amplitude_scale": float, "units_label": str,
"units_power": int}`. `units_power` always resolves (the Stage 1 chain ends
in a hard default).

**Guarantee:** on the same file, at every stage — including before Stage 1 has
persisted anything — `amplitude_scale` and `units_label` are exactly the pair
`compute_display_ft(path)` applies to its returned spectrum. The value is
resolved through the Stage 1 chain (persisted > import-time recommended > hard
default), the same chain that selects the spectrum being labelled.

### Fit thresholds

`fit_thresholds(path)` → the thresholds the persisted Stage 5 fit actually
applied, by name, at minimum:

- `peak_survival_snr_floor` — the survival SNR floor (derived: the Stage 3
  promotion cutoff times the survival factor, unless overridden), and
- `vif_collapse_threshold`.

Every field is `Absent.NOT_RUN` when no Stage 5 fit exists. A fit persisted
before a threshold was recorded reports that threshold as `Absent.NOT_RUN`,
never a guessed default.

### Final products: per-line fit fields

Each `FinalPeak` additionally carries, from the Stage 5 fit of its window:

- `decay_time_us` and `decay_time_error_us` (the fitted τ and its 1σ error;
  `Absent.UNDEFINED` when τ was held fixed and has no error),
- `shape` (the line shape the window was fitted with),
- `fwhm_mhz` — the feature FWHM of the finite-record line shape:
  `fitting.validation.feature_fwhm(τ, stage5.acquisition_us, shape)`, the
  same call and record length a client makes today, so displayed widths do
  not move,
- `detection_index` (the Stage 3 index that seeded the line; provenance, not
  identity), and
- `fit_window_mhz` — the `(low, high)` frequency bounds of the fit window, in
  the same frame as `frequency_mhz`.

These fields are always present on a table read through the contract. A stored
table that predates them is rebuilt **in memory** when read; the file is not
touched (§File model: a read never writes). They use `Absent` from the start
(no `None`/`nan` absence): a line with no Stage 5 fit record behind it (for
example a line whose window cannot be identified) reports each field as
`Absent.UNDEFINED`. `fit_window_mhz` is in the calibrated frame, like
`frequency_mhz`; a line from a window created during review reports that
window's bounds. On the wire a pair is a two-element array.

### Window status

`window_status(path)` → one row per Stage 4 plan window and per created
window: `window_id`, `freq_min_mhz`, `freq_max_mhz`, `created` (bool),
`n_fitted_peaks`, and `live` (bool). A window is **live** when the Stage 5 fit
holds at least one fitted line in it. Before Stage 5, `n_fitted_peaks` and
`live` are `Absent.NOT_RUN`. While Stage 5 is `partial` (after a cancel), a
window the fit has not reached reports both as `Absent.NOT_RUN`, never 0. Also available as a `read_table` table.

### Source preview

`preview_source(source, format_name=None)` → without importing: the detected
format and the source's **FID table**, one row per FID with `index`,
`channel`, `n_points`, `spacing_us`, `probe_freq_mhz`, `sideband` (decoded to
`"upper"`/`"lower"` from whatever encoding the source uses), and `shots`, plus
the declared chirp window when the source carries one. Every loader that can
hold more than one FID reports all of them; a multi-channel source reports
one row per channel, and `channel` names the value import would need.

The preview reports only what the source declares. A field the source does
not declare is `Absent.NOT_RUN` — never the default import would apply. A
declared value that cannot be read (a chirp window that fails to parse, a
non-finite sidecar value) is `Absent.UNDEFINED`.

### Window model

`window_model(path, window_id, *, grid="active", components=False)` → the
fitted model of one window, evaluated so a client can draw it — and check it —
without reimplementing the line shape:

- `schema`: `ftmw/window_model@1`; `window_id`; `grid`; `frame` — always
  `"raw"`, the fit's frame (the calibrated frame applies only to reported line
  frequencies);
- `frequency_mhz`: the grid, restricted to the window's fit range;
- `data`: complex, the spectrum on that grid — for `"active"`, exactly the data
  the fit compared with;
- `model`: complex, everything the fit compared with the data: the window's
  fitted lines, plus the frozen contributions of neighbours the fit held fixed,
  plus the window's fitted baseline when it has one;
- `fixed`: complex, the frozen neighbour contribution alone, drawn at the
  window's shared fitted decay time (the far-wing approximation of
  `stage5_fitting.rst`; the fit evaluates it the same way);
- `baseline`: complex, the fitted baseline alone, or `Absent.NOT_RUN` when the
  window was fitted without one;
- `sigma`, `excluded` (active grid only; `Absent.UNDEFINED` on the display
  grid): the per-bin noise the fit weighted by and a boolean mask of the bins
  the fit left out (gated spurs), so that
  `sum(|data − model|² / (sigma²/2))` over the bins not excluded is the fit's
  χ²;
- with `components=True`, one complex array per fitted line, keyed by
  `peak_uid`; `model = Σ components + fixed + baseline`.

`grid="active"` is the **native active-FT grid that Stage 5 fits** — not
`compute_ft`'s grid, which covers the full record at a different bin spacing,
scale and phase origin. `grid="display"` is `compute_display_ft`'s zero-filled
grid, which contains every active bin exactly; the model overlays that
spectrum point for point. Amplitudes are in the units of the corresponding
spectrum, so `display_units` applies to both. Magnitudes are `abs()` of the
complex arrays. The evaluation is the same code path every plot uses.

An unknown `window_id` raises `not_found`; a file without a Stage 5 fit raises
`stage_not_run`. A window fitted before analysis epoch 4 with frozen
contributors and a free decay time held their skirt at a starting decay time
it did not record; its model cannot be reproduced and raises
`incomplete_provenance` (re-run `fit run`).

### Spectrum model

`spectrum_model(path, *, grid="active")` → `{"schema":
"ftmw/spectrum_model@1", "grid", "frame": "raw", "frequency_mhz", "data",
"model", "residual"}`:

- `frequency_mhz` / `data`: the whole grid and spectrum (the native active FT,
  or `compute_display_ft`'s);
- `model`: complex, every line of the persisted fit (including curation edits)
  evaluated **once** over the whole grid at its window's fitted decay time,
  plus each window's fitted baseline evaluated **only inside that window's fit
  range** (a baseline is a local leakage-wing polynomial and is meaningless
  outside it; where fit ranges overlap, a bin takes the baseline of the window
  whose centre is nearest);
- `residual`: `data − model`, point for point.

Inside a window, `spectrum_model` differs from `window_model` by design: it
carries every neighbour's full line rather than the frozen contribution the
fit held, so its residual is not the fit's χ² residual. A file without a
Stage 5 fit raises `stage_not_run`. Through the CLI, the arrays go to `.npy`
under `--output`.

## Errors *(outline)*

- Every refusal a program may need to route on raises a member of the public
  exception family rooted at `PipelineFileError`, carrying a stable `code`
  string and typed attributes; `to_dict()` returns
  `{"schema": "ftmw/error@1", "code", "message", ...attributes}`.
- Codes (initial set): `stage_not_run` (`StageDependencyError`:
  `missing_dependencies` — canonical stage names, §Status and settings —
  and `command`), `bad_setting` (`path`, `expected`,
  `value`), `not_found` (window/peak/file: `kind`, `ids` — every id a request
named that does not exist, e.g. all unknown window ids of a curation batch),
`incomplete_provenance` (`missing`), `file_exists`, `file_incompatible`
  (`file_version`, `supported_version`), `file_corrupt`, `epoch_mismatch` (`file_epoch`, `current_epoch`),
  `cancelled`, `callback_failed`, `algorithm_failed` (`stage`).
- Each typed error remains a subclass of the built-in it replaced (most are
  `ValueError`), so existing `except` clauses keep working.
- A `.ftmw` path that does not exist raises `not_found` (`kind: "file"`),
  which is also a `FileNotFoundError`; a path that exists but cannot be opened
  as a pipeline file raises `file_corrupt`.
- The CLI maps codes to exit codes in one place: `file_corrupt` and
  `algorithm_failed` exit 2, `cancelled` exits 130, every other code exits 1.
  Under `--format json`, and always for `read` accessors, the error dict is
  written to stderr. The code set is introduced wave by wave. A code whose
  class exists but that no refusal raises yet (`algorithm_failed` until a
  stage needs it; `cancelled` and `callback_failed` before Wave 5) is
  documented as reserved. `capabilities()` lists only codes with a class.
- Status calls (`get_pipeline_info`, `list_available_stages`) raise for a file
  they cannot open. `validate_pipeline` and `Pipeline.validate` agree:
  - a file that cannot be opened raises the typed error `Pipeline.open`
    raises (`not_found`, `file_corrupt`, `file_incompatible`);
  - an openable file gets a report, whose `valid` and `errors` describe the
    file's integrity.

  Validation reports problems in a pipeline file. It does not stand in for
  opening one.
- **Which refusals are typed.** A refusal is typed when a program calling a
  public interface could act on its reason:
  - a setting value (`bad_setting`: an unknown path, a wrong type, an
    out-of-range value, a bad choice);
  - a missing entity (`not_found`);
  - a missing stage (`stage_not_run`);
  - a stage algorithm that cannot produce a result from valid inputs
    (`algorithm_failed`).

  Argument checks inside kernels and storage codecs, which a correct caller
  cannot trigger, stay built-in exceptions. They are bugs, not routes.

## Events and cancellation *(outline)*

- Every long operation (stage runs, `run_pipeline`, curation apply/preview,
  scans) accepts an optional `events` callback and an optional cancel token.
- The callback is synchronous and is always invoked on the calling thread,
  never inside a pool worker.
- Events (each with a schema name): stage start; stage end with the summary
  numbers today's completion log line carries; per-window progress
  (`index`, `total`, `window_id`, elapsed); `invalidated` (stage keys);
  warnings with codes (`epoch`, `environment_drift`, `slow_window`,
  `frame_mismatch`, …).
- A callback that raises aborts the operation with a typed `callback_failed`
  error, the callback's exception chained.
- Cancellation is checked between windows and between stages. A cancelled
  operation raises a typed `cancelled` error. Every completed stage stays as
  written, and a stage interrupted mid-run leaves nothing behind, with one
  exception: **Stage 5 keeps the windows it completed.** The `cancelled`
  error lists them (`completed_windows`), and `status` reports Stage 5 as
  `partial`. A later `fit run` with the same settings fits only the remaining
  windows and must reach the same result as an uninterrupted run. Changed
  settings discard the partial fit and start over. While Stage 5 is
  `partial`, Stage 6 and the final products are `not_run`: they need a
  complete fit.
- Log output is rendered from these events, so interactive users see the same
  lines as today.

## Status and settings

- One canonical **stage vocabulary**, the public enum `ftmwpipeline.Stage`,
  whose values are the CLI object names: `data`, `ft`, `noise`, `tau`,
  `tau_g`, `timebase`, `peaks`, `windows`, `fit`, `review`. Every contract
  payload that names a stage uses these values.
- **Mappings.** Each spelling has one read-only mapping in `contract.py`:
  - `STAGE_KEYS`: the storage key;
  - `STAGE_SETTINGS_PREFIX`: the settings / preset prefix, such as `stage2b`
    (it covers both `tau` and `tau_g`), or `None` for a stage with no settings
    record;
  - `STAGE_KNOB_PREFIX`: the tuning-registry knob prefix, or `None`.

  Each has a `*_for_*` inverse where the mapping is one-to-one. They are
  exposed in `capabilities()` under `"stages"`, as
  `[{"stage", "storage_key", "settings_prefix", "knob_prefix", "depends_on"}]`.
- **`status(path)`** returns `{"schema": "ftmw/status@1", "stages": [{"stage",
  "state", "depends_on"}], "runnable": [...], "rerun_order": [...]}`.
  - `state` is `complete`, `partial` (Stage 5 after a cancel; no producer
    yet) or `not_run`.
  - `depends_on` lists canonical stage names.
  - `runnable` lists the stages that are not complete but whose dependencies
    all are.
  - `rerun_order` lists every stage in the dependency-respecting order a full
    refresh follows: a fixed topological order with ties broken by the enum's
    order.
  - It is file-bound on all three interfaces, and the CLI verb is
    `read status`.
- **No stale results persist.** A write that makes a stored result
  inconsistent with its inputs does one of two things in the same write:
  - rebuilds that result, as the final-products table is rebuilt after a
    timebase or σ-floor change;
  - or deletes it and its downstream results.

  "Stale" is therefore never a state a client can observe. A call that only
  reads never writes. `compute_ft(..., from_saved_params=True)` recomputes
  the spectrum without persisting.
- A timebase change does not invalidate Stage 5. The fit uses ε only to
  classify clock spurs, and that classification is deliberately not redone.
  The change reaches the calibrated frequencies and their errors through the
  final-products rebuild.
- **Every stage-running call reports the stages it invalidated.** Its result
  carries `invalidated`, a list of canonical stage names in `rerun_order`
  order, which is empty when nothing was invalidated. On the CLI, the human
  output names them and `--format json` includes them.
- Settings registry rows (`SettingRow`, from `settings_show` /
  `settings_defaults`) add these fields:
  - `type`: one of `"float"`, `"int"`, `"bool"`, `"str"`, `"choice"`,
    `"float_pair"`, `"float_list"`, `"str_list"`, `"shape_spec"`,
    `"clock_sources"`;
  - `nullable` (`bool`);
  - `units` (a string such as `"MHz"` or `"us"`, or `None`);
  - `choices` (a list, or `None`);
  - `bounds` (`{"min", "max", "min_inclusive", "max_inclusive"}`, or `None`).

  Every value round-trips as typed JSON, including `ShapeSpec` with its
  parameters, and clock sources. Bounds and choices come from the knob
  metadata; a knob without them reports `None`, never a guess.
- **Every verb types an unopenable file.** A path that exists but is not a
  readable pipeline file raises `file_corrupt` (exit 2) from every verb, not
  only from those that open through `open_pipeline_file`.

## Curation as data

`review_apply` / `review_preview` accept an in-memory sequence of typed,
JSON-able curation actions as an alternative to a curation-file path, with the
same validation, frame handling, and results. The file path remains for people.

**`CurationAction`** is a public frozen dataclass in `core/curation.py`, 1:1 with a
curation-file row:

| field | type | meaning |
|---|---|---|
| `action` | `"add"` \| `"remove"` \| `"accept"` \| `"create"` | the row action |
| `window_id` | `int` or `None` | the window. `None` means "derive it" on `add` / `remove` (the file's `auto`), and "a new window" on `create`. It is required on `accept`. |
| `freq_mhz` | `float` or `None` | the row's one frequency: `add` / `remove`, or the `create` anchor |
| `peak_uid` | `int` or `None` | `remove` only, in place of `freq_mhz` (the file's `uid:N`) |
| `candidate_mhz` | `float` or `None` | `accept` only: a candidate to revive (`candidate=F`) |
| `frame` | `"raw"` \| `"calibrated"` or `None` | the frame of `freq_mhz` / `candidate_mhz`; see below |
| `epsilon` | `float` or `None` | only with `frame="calibrated"`: the epsilon the frequency was computed under (a file's `# epsilon:` stamp), checked for drift at apply time exactly as a file's |

- **Construction validates** the same arity rules the parser enforces:
  - one frequency, or one `peak_uid` on `remove`;
  - no frequency on `accept`;
  - no `peak_uid` outside `remove`.

  A violation raises `bad_setting` naming the field.
- **Wire form.** `to_dict()` gives
  `{"schema": "ftmw/curation_action@1", "action", "window_id", "freq_mhz",
  "peak_uid", "candidate_mhz", "frame", "epsilon"}`. Every key is always present, and an
  unused field is `null`; these are request fields, not absent results.
  `CurationAction.from_dict()` is the inverse.
- **Row correspondence.** `to_row()` gives the CSV row. Parsing a file yields
  the same actions as `from_dict` of their dicts: one grammar, two spellings.
- **Frames.** Each action's `frame` is resolved on its own: `None` takes the
  call's `frame=`, and then the call's rule applies (raw on an `epsilon == 0`
  file; required on a `self_calibrated` file when the action carries a
  frequency). A batch may mix frames. An explicit action frame that disagrees
  with an explicit call `frame=` is refused, as a file header that disagrees
  with it is. The pipeline converts each action to raw before resolution,
  exactly as it does a file's frequencies.
- **Calls.**
  - `review_apply(path, curation_path=None, *, actions=None, ...)` and
    `review_preview(...)` take exactly one of `curation_path` or `actions`,
    else `bad_setting` (`path` `"actions"`). `curation_path` keeps its name and
    position for existing callers.
  - `Pipeline` and the review session take the same.
  - The CLI's `review apply` / `review preview` accept `--actions FILE`, a
    JSON array of action dicts (`-` for stdin), as an alternative to the
    curation CSV.
- **Results are identical.** The same actions given as a file and as data give
  equal results, decision logs and files.

**Frames are explicit.** Every curation action and every review call that
takes a frequency declares its frame (`"raw"` or `"calibrated"`) as a typed,
documented parameter. The stated default is `None`, meaning:
- raw on a file whose `epsilon` is 0;
- an error (`bad_setting`, `path` `"frame"`) on a `self_calibrated` file when
  the call carries a frequency.

The conversion stays inside the pipeline. Clients never convert frequencies
themselves or probe signatures to discover the parameter.

## Serialization *(outline)*

A public `ftmwpipeline.serialize.to_jsonable(obj)` covers every contract result
and domain type, applies the missing-value rule above, and stamps the schema
name. Enums serialize as their `.value`. Inline complex numbers are
`{"real", "imag"}`. CLI `--format json` on every verb routes through it.

Through the CLI, `--output` for a `read` accessor names a directory; each
array field is written there as `<field path>.npy` and the JSON envelope
names the file in the array's place. An accessor whose result holds arrays
refuses to run without `--output`.

## Capabilities *(outline)*

`capabilities()` → `{"schema": "ftmw/capabilities@1", "contract_version": int, "schemas": [...],
"accessors": [...], "codes": [...]}`, read from the same manifest the
contract tests check.

## Open questions

None open. Resolved: CLI arrays are `.npy` (§Accessors); the whole-spectrum
model is §Spectrum model; a time-domain model is not wanted.
