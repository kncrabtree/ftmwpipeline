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

## Accessors

Each accessor below is a read: it never writes the file, and it is exposed on
the functional API, on `Pipeline`, and through the CLI `read` object with
`--format json`. Through the CLI, an array-valued accessor writes a `.npy`
file to `--output` (self-describing dtype, shape and byte order) and prints the
JSON envelope with the array fields replaced by their file name. Each states what it returns, what it is guaranteed to
equal, and when it is absent.

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
  (ascending, in the raw frame, trimmed to `compute_ft`'s band at
  `pad_factor`× its density), `complex_spectrum` aligned to it, and
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
- the frequency-calibration inputs (the declared `sigma_floor_khz`). This is
  the only Stage 6 input that shapes the final products: `review run`'s
  `bar`, `kappa`, `noise_floor` and `attention_candidate_evidence` only route
  attention, and the snap tolerance and refit options are arguments of
  individual curation actions, which are excluded below;
- the acquisition parameters the analysis used (probe frequency, sideband,
  sample spacing), including any import-time overrides;
- the analysis epoch each persisted stage was produced under.

Every stage key is always present. A stage that has not run contributes
`Absent.NOT_RUN` under its key, so the digest changes as stages complete; a key
is never omitted.

**Incomplete provenance.** A file written before a stage persisted everything
it resolved cannot yield a complete fingerprint. The accessor then raises a
typed `incomplete_provenance` error naming the missing inputs (re-running the
stage persists them). It never computes a digest over incomplete inputs.

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

**Canonical form.** A JSON object keyed by registry path
(`stage1.trim`, `stage5.peak_survival.vif_collapse_threshold`,
`timebase.kappa_sys`, …), keys sorted by code point; finite floats as the
shortest decimal string that round-trips to the same IEEE-754 binary64 value
(as Python's `repr` produces, e.g. `0.1`, `1e-05`, `25.0`); `-0.0` spelled
`-0.0`; non-finite values as the strings `"nan"`, `"inf"`, `"-inf"`; integers
without a decimal point; booleans as `true`/`false`;
tuples as arrays; unset as `null`; `Absent` per the wire rule; structured
values (`ShapeSpec`, clock sources) in their documented typed-JSON form.
UTF-8, no insignificant whitespace. The digest is SHA-256 over that byte
string, lowercase hex.

Values are read through each stage's codec, never by walking the file layout.
Every stage must persist what it resolved at run time. A stage that today
resolves a value from a layer it does not persist (a recommendation, a
detected value) must persist the value it used, so the fingerprint can be
computed from the file alone.

### Display units

`display_units(path)` → `{"amplitude_scale": float, "units_label": str,
"units_power": int | None}`.

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
touched (§File model: a read never writes).

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
`n_points`, `spacing_us`, `probe_freq_mhz`, `sideband` (decoded to
`"upper"`/`"lower"` from whatever encoding the source uses), and `shots`, plus
the declared chirp window when the source carries one. Every loader that can
hold more than one FID reports all of them.

### Window model

`window_model(path, window_id, *, grid="active", components=False)` → the
fitted model of one window, evaluated so a client can draw it over the
spectrum without reimplementing the line shape:

- `schema`: `ftmw/window_model@1`; `window_id`; `frame` (the frequency frame of
  the grid, matching `compute_display_ft`'s axis);
- `frequency_mhz`: the grid, restricted to the window's fit range;
- `model`: complex, the full window model — the window's fitted lines plus the
  frozen contributions of neighbours that the fit held fixed;
- `fixed`: complex, the frozen neighbour contribution alone;
- with `components=True`, one complex array per fitted line, keyed by
  `peak_uid`.

`grid="active"` evaluates on the native active-FT grid. There, `model` is
exactly what the fit compared with the data; the residual is
`compute_ft` data minus `model`. `grid="display"` evaluates on
`compute_display_ft`'s zero-filled grid, so the model overlays that spectrum
point for point. Amplitudes are in the same units as the corresponding
spectrum, so `display_units` applies to both. Magnitudes are `abs()` of the
complex arrays. The evaluation is the same code path the fit-detail plots
use.

An unknown `window_id` raises `not_found`; a file without a Stage 5 fit raises
`stage_not_run`.

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
  written to stderr. The code set is introduced wave by wave; codes not yet
  implemented are not listed by `capabilities()`.
- Status calls (`get_pipeline_info`, `list_available_stages`) raise for a file
  they cannot open; `validate_pipeline` and `Pipeline.validate` must agree on
  whether an unopenable file raises or returns a report.

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

## Status and settings *(outline)*

- One canonical **stage vocabulary**, the public enum `ftmwpipeline.Stage`,
  whose values are the CLI object names: `data`, `ft`, `noise`, `tau`,
  `tau_g`, `timebase`, `peaks`, `windows`, `fit`, `review`. It carries a
  documented mapping to every other spelling (settings prefix, knob label,
  internal storage key). Every contract payload that names a stage uses these
  values.
- `status(path)` → each stage's state (`complete`, `partial` — Stage 5 after a
  cancel — or `not_run`), the dependency graph, and the re-run order a full
  refresh would follow.
- **No stale results persist.** A write that makes a stored result
  inconsistent with its inputs either rebuilds that result in the same write
  (as the final-products table is rebuilt after a timebase or σ-floor change)
  or deletes it and its downstream results. "Stale" is therefore never a
  state a client can observe.
- A timebase change does not invalidate Stage 5. The fit uses ε only to
  classify clock spurs, and that classification is deliberately not redone.
  The change reaches the calibrated frequencies and their errors through the
  final-products rebuild.
- Every stage-running call reports the stages it invalidated.
- Settings registry rows add `type`, `nullable`, `units`, and `choices` or
  `bounds`; every value round-trips as typed JSON, including `ShapeSpec` with
  its parameters and clock sources.

## Curation as data *(outline)*

`review_apply` / `review_preview` accept an in-memory sequence of typed,
JSON-able curation actions as an alternative to a curation-file path, with the
same validation, frame handling, and results. The file path remains for people.

**Frames are explicit.** Every curation action and every review call that
takes a frequency declares its frame (`"raw"` or `"calibrated"`) as a typed,
documented parameter, with a stated default. The conversion stays inside the
pipeline; clients never convert frequencies themselves or probe signatures to
discover the parameter.

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

1. **Whole-spectrum model** (wanted by BlackQuill; lands before 1.0.0, after
   the accessors above). `spectrum_model(path, grid=...)` — every final line evaluated over
   the full grid, with its residual. Overlapping windows make "sum of window
   models" double-count the frozen neighbours, so it is its own evaluation.

Resolved: CLI arrays are `.npy` (§Accessors). A time-domain model is not
wanted.
