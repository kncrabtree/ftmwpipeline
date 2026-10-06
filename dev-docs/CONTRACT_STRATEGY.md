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
  - **Degenerate statistics are `UNDEFINED`.** Earlier writers stored some
    undefined quantities as ordinary numbers:
    - an F-test with no residual degrees of freedom or a non-positive χ²
      (stored as F = 0, p = 1);
    - the doublet `orth_evidence` fallback and the disabled line-evidence
      escape (stored as 0.0);
    - edge coherence of an empty residual or of zero σ (stored as 0.0);
    - a Stage 3 SNR whose local noise σ is ≤ 0 (stored as 0.0).

    Each reads as `UNDEFINED` wherever the stored inputs show it was
    degenerate, so old files report it too. Writers store `nan` for them from
    now on. Gate decisions do not change.
  - A count, creation time or plan revision attribute missing from a stage
    group reads as `NOT_RUN`. It is never filled with `0` or `"unknown"`.
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
  `window_id`, `frequency_mhz`, `kind`, `provenance`, `evidence`, `serial`.
  Names, types and the `kind` / `provenance` vocabularies are frozen, so a
  client may hash the log as an edit-set identity. `serial` (contract version
  16) is the decision's stable id: minted when the row is recorded, never
  reused or renumbered within the fit's lineage (`fit run` starts a new one),
  `Absent.NOT_RUN` on a row a pre-engine build recorded. It is what
  `review_undo` takes and what a peak's `derivation` holds; `order_index` is
  the row's position only. Every row's `evidence` carries `action_index`, the
  `serial` of the first row of the same user action (a one-row action, a bare
  accept included, carries its own); undo and a log-prefix apply replay one
  action group at a time. The key is part of `evidence`, so a hash taken over
  `evidence` changes with it (contract version 15). A `remove` row's
  `frequency_mhz` and a merge's `merged_from` are the fitted (raw-frame)
  frequencies of the peaks the request resolved to, never the frequencies
  sent (contract version 16; older rows hold the frequencies sent). Rows are
  immutable: a replay (undo, log-prefix apply) keeps every surviving row
  verbatim, never what the replayed fit resolves, and only `order_index` is
  recomputed; a surviving row that would replay as a different action, or
  that the replay would not record, is refused (`curation_conflict`,
  `replay_diverged`). The snap tolerance is a
  property of the file: no call takes one (contract version 16).
- Stage 6 writes refuse a file the replay engine cannot curate (contract
  version 16): `curation_conflict` with `predates_peak_identity` (a fitted
  peak has no `peak_uid`) or `predates_replay_engine` (the review or the undo
  baseline was written without it), `ids` `[]`, before anything is resolved,
  fitted or written; nothing is converted. `Stage6Review.refit_required`
  carries the reason on `get_review_status` (and `review show --json` /
  `review log --json` carry it as `refit_required`), and `review_log` and the
  HTML report flag the file without writing. `fit run` is the one remedy and
  discards the curation. A review from a newer engine raises
  `PipelineCompatibilityError` on a write, and `refit_required` is then
  `file_incompatible` (the remedy is upgrading). A bare `review_edit` (neither `add` nor
  `remove`) is `bad_setting`, `path` `add`, on every interface.
- The curation result types (`RefitWindowResult`, `PreviewWindowResult`,
  `AppliedWindowResult`, …) and their `converged` flag: a bool, `Absent.NOT_RUN`
  where `chi2r_after` is, and `Absent.UNDEFINED` for a window left with no peak
  (no solver ran; no non-convergence warning, and the report's non-converged
  count excludes it).
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
  - the snap tolerance is a property of the file (no call takes one), and
    refit options are arguments of individual curation actions, which are
    excluded below;
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
  an edit-set identity; from contract version 15 its rows' `evidence` carries
  `action_index`, so a hash over `evidence` differs from an earlier one);
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

`window_status(path)` → one row per window of the fitted plan (below) and per
created window: `window_id`, `freq_min_mhz`, `freq_max_mhz`, `created` (bool),
`n_fitted_peaks`, `live` (bool), and `merged_from` (ids, ascending: a tuple in
Python, an array on the wire). A window is **live** when the Stage 5 fit
holds at least one fitted line in it. Before Stage 5, `n_fitted_peaks` and
`live` are `Absent.NOT_RUN`. While Stage 5 is `partial` (after a cancel), a
window the fit has not reached reports both as `Absent.NOT_RUN`, never 0. Also available as a `read_table` table.

**Windows after a structural merge.** Once a complete Stage 5 fit exists,
every window geometry the contract reports for the fit is the geometry the fit
was made on: `window_status`, the window model, and the windows every Stage 6
call resolves, edits and refits. A structural replan can merge a window into its
neighbour:
- the survivor keeps the lower id and the merged range;
- the absorbed id no longer appears as a window and is never minted again;
- a row gains `merged_from`, the ids folded into it, empty if none were.

The Stage 4 product stays the plan as planned: `load_windows` and the
`read_table` tables `windows`, `window_free_peaks` and `window_contributors`
report the Stage 4 windows whatever a fit later merged. The windows a fit was
made on are reported by `window_status` (and by the fit's own tables); a client
that joins fit rows to windows joins them to `window_status`.

Before a complete fit, while Stage 5 is `partial` included, rows follow the
Stage 4 plan.

### Review attention

`review run` routes windows to a human: each window it flags gets a
`WindowReviewStatus` in `get_review_status(path).window_statuses` (Python
`Pipeline.review_status()`), keyed by window id, whose `attention_reasons` are
`AttentionReason` records:
- `kind`, from the closed vocabulary `attention_kind`: `worst_eps`,
  `auto_merged_review`, `candidate_bearing`, `spur_adjacent`, `edge_boundary`,
  `flat_decay`, `empty_window_residual`, `empty_window_spur`. Kinds are only
  ever added.
- `severity` (float; higher asks for a look sooner);
- `locations`: the molecular frequencies (MHz) the reason points at, empty for
  a window-wide reason;
- `evidence`: a dict of the kind's declared keys (below), empty for a kind that
  declares none;
- `detail`: a human sentence. Its text is not contract.

`auto_merged_review`, `flat_decay` and `empty_window_spur` are advisory: they
stay on the status but do not by themselves put the window in the queue
(`needs_attention`).
Attention is advice. It never changes a fitted number, a final product or the
analysis fingerprint.

The CLI reports the same records: `review show --attention --json` gives one
row per queued window with its top reason (`window_id`, `label`, `kind`,
`severity`, `detail`) and every reason in `reasons` (`kind`, `severity`,
`detail`, `locations`, `evidence`); `review show --window N --json` gives the
window's `attention_reasons` in the same form.

**`empty_window_residual` and `empty_window_spur`.** A window of the fitted plan
in which the fit holds no line, while Stage 5 measured a coherent residual on
its edge. One of the two flags a window when all of these hold:
- the window is a window of the fitted plan, not one Stage 6 created, and no
  created window has taken it over (below);
- the current fit holds no line in it;
- no Stage 6 decision that changes the fit has been recorded on it;
- Stage 5's residual edge-coherence handshake left at least one of its edges
  flagged on the window's current fit: a structural-replan record the window
  triggered that was not applied, or a thaw record of the window that was not
  accepted, with `S_coh` above the fit's own `residual_edge_threshold`. An edge
  a later accepted thaw resolved does not count. A replan record measured
  before the window was last re-fit does not count either: a merge re-fits its
  survivor and every window that transitively depends on it (the fitted plan's
  dependency edges), so a record whose `revision_before` precedes the last such
  merge's `revision_after` describes a fit that no longer exists. The records
  that remain are read in the order Stage 5 measured them, the current fit's
  thaws before the replan rounds that scanned it.

Which of the two depends on the Stage 3 peaks the plan put in the window. When
every one of them sits on a gated spur (`gated_spur` below), the edge residual
is consistent with the spur's skirt beyond its mask, and the reason is the
advisory `empty_window_spur`, whose detail names the spur. Otherwise (at least
one peak off every gated spur, or no peak at all) a line may be missing, and the
reason is `empty_window_residual`, which queues the window. Everything else
below holds for both.

The trigger reads only what Stage 5 recorded; a fit run with the thaw and the
replan disabled records no flags and raises none. The window has a status
although the fit has no window result for it, so `window_statuses` may hold ids
that `load_fit` does not. Its `severity` is the strongest flagged edge's `S_coh`
over the threshold, its `locations` the frequencies of the Stage 3 peaks the
plan put in the window, and its `evidence`, in which every record carries every
key and a missing value is `Absent` (§Missing values):
- `edges`: one `{"side": "low" | "high", "s_coh": float,
  "neighbour_line_distance_mhz": float}` per flagged edge, low first.
  `neighbour_line_distance_mhz` is the distance from that edge to the nearest
  fitted line beyond it, `UNDEFINED` when the fit holds none on that side;
- `residual_edge_threshold`: the threshold the fit applied;
- `candidates`: one `{"detection_index": int, "frequency_mhz": float, "snr":
  float, "gated_spur": bool, "spur_center_mhz": float, "spur_source": str}` per
  Stage 3 peak the plan put in the window, ascending in frequency. `snr` is the
  Stage 3 SNR, `UNDEFINED` when Stage 3 recorded none or it is degenerate (a
  local noise that is not positive, stored by earlier writers as `0.0`).
  `gated_spur` is true when the peak sits within the `spur_adjacent` tolerance
  of a spur the fit gated; `spur_center_mhz` and `spur_source` name the nearest
  such spur, both `UNDEFINED` when the peak is on none, and `spur_source`
  `NOT_RUN` when the fit gated the spur without recording its source.

**Taking the window over.** The fit holds no window to edit, so the item
resolves when created windows take the window over: a created window with the
same id, or created windows that together cover every flagged Stage 3 peak
(every flagged edge, when the window holds no peak). A created window that
overlaps the window without covering what was flagged there leaves the item on
the window. The item is dropped in the batch that creates the covering window,
and with it a status that then carries nothing, reviewed or not. A bare
`review accept` may name the window, alone or in any batch (with other edits,
in a preview, replayed by `review undo`), and marks it reviewed, as for any
kind. The report gives such a window a page wherever it gives a window of its
queue state one (a queued window under either window filter, an advisory one
only when every window gets a page): the data and the residual on the window's
range (the residual is the data, since no model was fitted), with the
candidates marked.

A fit made before fits stored their plan does not hold a merged window's
geometry. Its rows follow the merges its replan record names, and a Stage 6 call
that would refit those windows, or create one inside a merged range, is refused
(`curation_conflict`, reason `fit_plan_unavailable`) rather than run on geometry
the fit was not made on.

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

A source its format's loader refuses — whether the format was named or
detected — raises `bad_setting` (`path` `"source"`) with the loader's message,
as `import_data` does for a source its loader refuses (an unknown sidecar key,
a missing required value such as `spacing_us`, an unknown `column`).

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
  `value`), `not_found` (`kind` — `window`, `peak`, `file` or `decision` — and
`ids`: every id a request named that does not exist, e.g. all unknown window
ids of a curation batch),
`incomplete_provenance` (`missing`), `file_exists`, `file_incompatible`
  (`file_version`, `supported_version`), `file_corrupt`, `epoch_mismatch` (`file_epoch`, `current_epoch`),
  `cancelled` (`stage`, `completed_stages`, `completed_windows`;
  §Events and cancellation), `callback_failed` (`event_schema`, `completed_windows`),
  `algorithm_failed` (`stage`), `write_conflict` (§Crash safety),
  `curation_conflict` (`reason`, `ids`).
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
    file's integrity;
  - a file that becomes unreadable while the report is built raises the
    same typed error; the report does not swallow it.

  The report, and `list_available_stages`, name stages by their canonical
  names (§Status and settings).

  Validation reports problems in a pipeline file. It does not stand in for
  opening one.
- **Which refusals are typed.** A refusal is typed when a program calling a
  public interface could act on its reason:
  - a setting value (`bad_setting`: an unknown path, a wrong type, an
    out-of-range value, a bad choice);
  - a missing entity (`not_found`);
  - a missing stage (`stage_not_run`);
  - a stage algorithm that cannot produce a result from valid inputs
    (`algorithm_failed`);
  - a valid curation request that conflicts with the file's review state
    (`curation_conflict`). Its `reason` is a stable slug, such as
    `line_already_fitted`, `targets_span_windows`, `orphans_created_window`,
    `baseline_unavailable` or `replay_conflict`. `ids` names the windows,
    peaks or decisions involved.
- **`bad_setting.path` names what the caller wrote:**
  - a registry setting is its registry path;
  - an argument is its name (`add`, `remove`, `format`, `source`);
  - a curation-file cell is `curation[line <n>].<column>`;
  - a field of the i-th `CurationAction` is `actions[<i>].<field>`.

  Argument checks inside kernels and storage codecs, which a correct caller
  cannot trigger, stay built-in exceptions. They are bugs, not routes.

## Events and cancellation

This section is built in three steps:
- Wave 5.1 builds the events and cancellation.
- Wave 5.1b builds §Crash safety.
- Wave 5.2 builds §Stage 5 partial fits.

Until 5.2, cancelling Stage 5 leaves nothing behind, like any other stage.

### Python surface

- Every long operation takes two optional keyword arguments on all three
  interfaces (the CLI equivalents are under §CLI below):
  - `events: Callable[[Event], None] | None`;
  - `cancel: CancelToken | None`. `CancelToken` is a protocol with one method,
    `is_set() -> bool`, so a `threading.Event` qualifies.
- The long operations:
  - every stage run: `import_data`, `detect_start_time`, `compute_ft`,
    `estimate_noise`, `calibrate_tau`, `recommend_shape`, `calibrate_timebase`,
    `detect_peaks`, `assign_windows`, `fit_peaks`, `review_run`;
  - the curation and refit calls: `review_apply`, `review_preview`,
    `review_accept`, `review_edit`, `review_create`, `review_undo`;
  - `report_run`, `scan_run`, `scan_all` and `run_pipeline`.

  `scan_run`'s existing `progress` callback keeps working, and passing `events`
  as well is allowed.
- The callback is synchronous and always runs on the calling thread, never
  inside a pool worker. Events from pool work are passed back to the parent
  first.
- An event is a frozen dataclass that carries its `schema`. `to_jsonable`
  serializes it like any other contract type. `Event` is the union of the types
  below.

### Event types

Every event has `schema`, `operation` (the CLI verb, such as `"fit run"`) and
`stage` (a canonical stage name, or `null` where none applies).

| Type | Schema | Further fields |
|---|---|---|
| `StageStarted` | `ftmw/stage_started@1` | — |
| `StageFinished` | `ftmw/stage_finished@1` | `elapsed_s`; `summary` |
| `WindowProgress` | `ftmw/window_progress@1` | `phase`; `round`; `index`; `total`; `window_id`; `n_peaks`; `chi2r`; `elapsed_s`; `dropped` |
| `ScanProgress` | `ftmw/scan_progress@1` | `knob`; `value`; `index`; `total` |
| `Invalidated` | `ftmw/invalidated@1` | `stages` |
| `PipelineWarning` | `ftmw/warning@1` | `code`; `message`; code-specific fields |

- **`StageFinished.summary`** has exactly the keys of the same verb's
  `ftmw/run_result@1` summary. The two come from one builder.
- **`WindowProgress`:**
  - Events come in **passes**. A pass is identified by its `(phase, round)`
    pair.
  - `phase` is one of four values:
    - `"initial"`: the fit's first walk, or a Stage 6 call's own windows;
    - `"replan"`: a structural replan round;
    - `"fallback"`: a sequential re-walk after the parallel walk of the same
      round fell back (`walk_fallback`). This pass reports windows that the
      round's earlier pass already reported.
    - `"cascade"`: Stage 6's re-fit of dependent windows.
  - `round` is `0` for the initial walk, its fallback, and every Stage 6 pass.
    It is the replan round's number, starting at 1, for a replan round and its
    fallback.
  - `index` counts finished windows within the pass, starting at 1. Windows can
    finish out of id order.
  - `total` is the number of windows in the pass. It is fixed when the pass
    begins.
  - `elapsed_s` is the window's own fitting time.
  - A dropped window has `dropped: true`, and its `n_peaks` and `chi2r` are
    Absent.

  Stage 6 refits and cascades emit `WindowProgress` with `stage: "review"`.
- **`Invalidated.stages`** are canonical names in `rerun_order`. The event is
  emitted once per call that invalidates something, after the call's write
  is durable (§Crash safety) and just before `StageFinished`. An invalidation
  that never landed is therefore never announced. The event matches the
  result's `invalidated` field.
- **`ScanProgress`** is emitted once per scanned value; `knob` is a registry
  path.
- **`PipelineWarning.code`** comes from the vocabulary `warning_code`.
  Additions are additive.

  | Code | Further fields | Emitted when |
  |---|---|---|
  | `slow_window` | `window_id`, `elapsed_s`, `threshold_s` | a window takes longer than the threshold |
  | `epoch_acknowledged` | `file_epoch`, `current_epoch` | a call edits a fit under an epoch acknowledgement |
  | `environment_drift` | `fields` | a long operation opens a file whose recorded environment differs from the running one (once per operation) |
  | `frame_mismatch` | `actions` (indices) | the existing curation advisory |
  | `walk_fallback` | `reason`, `n_windows` | the fit walk falls back to the sequential walk |
  | `timebase_skipped` | — | `run_pipeline` continues past a failed timebase calibration |

### Ordering

- For each stage, `StageStarted` comes first, then the stage's other events,
  then `Invalidated` (if any), then `StageFinished`.
- `StageFinished` is emitted only after the stage's results are written.
- A cancelled or failed stage emits no `StageFinished`.

### Callback failure

A callback that raises aborts the operation with `callback_failed`
(`CallbackFailedError`). The error's `event_schema` names the schema of the
event being delivered, and the callback's exception is chained as `__cause__`.
What is left in the file is the same as after a cancel at that point.

### Cancellation

- **Where cancellation is checked:**
  - before every stage, and between stages;
  - between windows: the Stage 5 walk, Stage 6 refits and cascades, and the
    report's per-window rendering;
  - between scan values.
- A stage with no windows (noise, FT, tau, …) checks only before it starts. Once
  a stage has begun its final write, it completes, and the cancel is honoured
  at the next check point.
- A cancelled operation raises `cancelled` (`OperationCancelledError`) with
  these fields:
  - `stage`: the stage that was interrupted, or `null` when the cancel fell
    between stages;
  - `completed_stages`: the stages this operation finished and wrote, in order;
  - `completed_windows`: a list of window ids. It is always `[]` until Wave 5.2.
- Every completed stage stays as written. A stage interrupted mid-run leaves the
  file exactly as it was before that stage began: nothing new, nothing deleted,
  and no invalidation.
- A curation batch (`review_apply`, and a refit with its cascade) is one unit:
  a cancel discards the whole batch.
- **Stage 5 in parallel.** A cancel stops the walk without waiting for windows
  that are still fitting:
  - the workers are terminated;
  - the pool is shut down without waiting;
  - the broken-pool state that termination causes is absorbed.

  Cancel latency is the parent's poll interval (about 0.2 s). The sequential
  walk (`jobs=1`, or no `fork`) honours a cancel after the current window. No
  check is made inside a single window's least-squares fit.
- **`run_pipeline`.** A cancel or a `callback_failed` raises; it is not folded
  into the result dict. For any other failure:
  - the result's `error` becomes that error's `ftmw/error@1` dict instead of a
    string;
  - a failure that is not a typed error is reported with the declared fallback
    code `pipeline_error`;
  - `failed_stage` is the canonical stage of the failing step, `None` when
    that step is not a stage (start detection, the report) and `tau_g` for a
    failing Gaussian tau step, matching the events and a cancel's "null for a
    step that is not a stage";
  - `failed_step` (additive) is the progress label of the failing step
    (`import`, `start detection`, `FT`, ..., `report`), `None` on success;
  - `completed_stages` is the canonical stages written, in order and
    de-duplicated, the list a cancel's `completed_stages` holds; start
    detection and the report add nothing. The `run --json` summary carries
    `failed_step` as a scalar and `completed_stages` as one comma-joined
    string.
- **Twin calibration.** A `tau run` that also builds the twin calibration
  reports the twin inside its own stage, with no events of its own. The
  `tau_g` (or `tau`) twin still appears in `completed_stages` once it is
  written.

### Log rendering

- These lines are rendered from events by a single renderer, at today's logger
  name, level and text:
  - stage start and end lines;
  - the `window %d/%d` progress line;
  - the per-window detail, dropped and slow-window lines;
  - the invalidation warning;
  - the walk-fallback warning.
- Per-window lines are now logged by the parent process. Their ordering among
  other parent lines may therefore differ from today's; their text does not.
- All other log lines stay ordinary logging. Log text is still not contract
  (§Versioning).

### CLI

- **Ctrl-C.** The first Ctrl-C sets the cancel token. The verb then exits 130,
  printing the `cancelled` error (as `ftmw/error@1` on stderr under `--json`).
  A second Ctrl-C raises `KeyboardInterrupt` at once.
- **`--events`.** Every long verb accepts `--events`, which writes each event to
  stderr as one JSON line. Under `--json`, an error dict follows as the last
  stderr line.

### Crash safety

- **Every call that writes a pipeline file writes atomically.** This covers stage
  runs, curation, settings, clocks and stamps. The call does all its writes in a
  temporary copy beside the file, then replaces the original with one
  `os.replace`.
- A process killed at any point, `SIGKILL` included, leaves one of two files: the
  file exactly as it was before the call, or the file as the call completed it.
  It never leaves a mix, a stage marked complete over missing or partial
  results, or a file that will not open.
- A reader that already has the file open keeps the version it opened.
- If the replace fails, the call raises and the file is unchanged. This happens,
  for example, on a platform that refuses to replace a file another process
  holds open.
- Within `run_pipeline`, each stage is its own atomic write, so a kill keeps
  every stage that finished before it.
- **Concurrent writers.** A call's copy is taken when its write begins. If the
  file on disk changed after that, because another process wrote it, the call
  raises `write_conflict` (`WriteConflictError`, exit 1) instead of replacing
  the file, and the other write stands. Writes from one process to one file
  are serialized.
- A cancel, a `callback_failed` or any other failure discards the copy, so
  the file is left exactly as it was before the call.
- Compaction happens when the copy is made: the copy is written compacted.
  The space a write frees is reclaimed by the next write.
- **Temporary copies.** The copy is created in the same directory as the target,
  so that `os.replace` stays on one filesystem. It is named
  `.<target basename>.ftmw-tmp.<hostname>.<pid>`, where `<pid>` is the decimal
  process id of the writer and the basename includes its extension. Compaction
  uses the same pattern.
  - A kill between creating the copy and the replace leaves it behind.
  - Before making its own copy, every write removes leftover copies of the same
    target that were made on the same host by a process that is no longer
    running.
  - Copies from other hosts are never touched. Neither are copies whose pid is
    alive, even if that pid has been reused.
  - A client that knows no write to the file is in progress may delete every
    copy matching the pattern.

### Stage 5 partial fits

- **Writing a partial fit.**
  - When a cancel or a `callback_failed` interrupts Stage 5, the windows that
    finished are written as a partial fit in one atomic write. "Finished"
    means the window's whole per-window pass ran.
  - The same write discards the previous fit and everything downstream of it.
  - If no window had finished, nothing is written and the file stays as it was,
    with any previous fit kept.
  - No partial fit is written during the walk. A process killed during
    Stage 5 therefore leaves the file as it was before the call
    (§Crash safety).
  - `cancelled.completed_windows` lists the window ids that were written.
    So does `callback_failed.completed_windows`, which is `[]` everywhere
    except a Stage 5 interruption.
  - The write's invalidations are delivered as one `Invalidated` (`fit` and
    everything downstream) after the partial write is durable and before the
    error is raised. It is never delivered to a callback that has just failed.
- **`status` while partial.**
  - `fit` reports `partial` and appears in `runnable`.
  - `review` reports `not_run`.
- **What reads and writes do while partial:**
  - Every accessor that reads the fit or the final products behaves as it does
    before Stage 5 has run. There is no partial line list.
  - `window_status` rows of written windows carry `n_fitted_peaks` and
    `live`; the other rows keep them `not_run`.
  - `review run` and every curation call refuse with `stage_not_run`.
- **Discard.** Anything that invalidates a complete fit also discards a partial
  one:
  - an upstream stage re-run;
  - `settings set` / `settings unset` on a `stage5` setting;
  - a forced re-import.

  Clocks and the timebase leave a partial fit alone, as they leave a complete
  fit.
- **Resume.** `fit_peaks` (CLI: `fit run`) resumes a partial fit by default
  and refits only the windows not yet written. `restart=True` (CLI:
  `--restart`) discards the partial fit and starts over. The call starts over,
  and says why, when:
  - the requested settings differ from the partial fit's recorded effective
    settings: the resolved Stage 5 settings, the consumed upstream values, or
    `ANALYSIS_EPOCH`;
  - or the partial fit lacks the provenance needed for that comparison.

  It never resumes on a guess. `ANALYSIS_EPOCH` is the only guard against a
  code change between the cancel and the resume. Every change that moves
  per-window fit numerics raises it, so a resume never mixes windows fitted
  by different numerics.
- **What a resume produces.** After the remaining windows, the fit finishes as
  usual: structural replan, cleanup and sorting over the whole fit. The result
  equals an uninterrupted run with the same settings, to the same standard as
  the parallel and sequential walks equal each other:
  - the same lines, `peak_uid`s and window structure;
  - parameters equal to floating-point rounding.

  An accepted thaw during a resume refits every window sequentially, as an
  uninterrupted run with an accepted thaw already does.
- **Resume summary.** `fit run`'s `run_result` summary, and therefore its
  `StageFinished.summary`, carries:
  - `resumed` (`bool`);
  - `windows_carried` (`int`, `0` unless resumed);
  - `restart_reason`: `null`, `"restart_requested"`, `"settings_changed"`,
    `"incomplete_provenance"`, or `"thaw_refit"`.

  `restart_reason` is `null` both for a fresh fit with nothing to resume and
  for a clean resume.
- **Progress on a resume.** `WindowProgress.index` continues from the count
  already written, and `total` stays the full window count.

## Status and settings

- One canonical **stage vocabulary**, the public enum `ftmwpipeline.Stage`,
  whose values are the CLI object names: `data`, `ft`, `noise`, `tau`,
  `tau_g`, `timebase`, `peaks`, `windows`, `fit`, `review`. Every contract
  payload that names a stage uses these values. The one per-stage environment
  entry that is not a stage, the shape recommendation, is published as
  `tau_shape` (`contract.PROVENANCE_NAMES`, `canonical_provenance_name`); every
  published drift or environment surface (`stage_environments`, the
  `environment_drift` message, the report's environment rows) names its keys
  this way, and a key that is neither is kept as recorded.
- **Mappings.** Each spelling has one read-only mapping in `contract.py`:
  - `STAGE_KEYS`: the storage key;
  - `STAGE_SETTINGS_PREFIX`: the settings / preset prefix, such as `stage2b`
    (it covers both `tau` and `tau_g`), or `None` for a stage with no settings
    record;
  - `STAGE_KNOB_PREFIX`: the tuning-registry knob prefix, or `None`.

  Each has a `*_for_*` inverse where the mapping is one-to-one. Only the
  storage key is one-to-one: `tau` and `tau_g` share both the `stage2b`
  settings prefix and the `stage2b` knob prefix. They are
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
  output names them. A verb with a JSON output mode includes them there.
  Stage verbs gain JSON output with the serializer rollout (Wave 8).
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

## Serialization

A public `ftmwpipeline.serialize.to_jsonable(obj)` covers every contract result
and domain type, applies the missing-value rule above, and stamps the schema
name. Enums serialize as their `.value`. Inline complex numbers are
`{"real", "imag"}`.

**Machine-readable CLI output.** Every CLI verb accepts `--json`, which makes
its output machine-readable, routed through `to_jsonable` (never
`json.dumps(..., default=str)`). This is one uniform switch: `--format` already
means the report file format (`report`) and the source format (`data import`,
`run`) on some verbs. `--format json`, where a verb accepts it today (`info`,
`review snap-tolerance`, `timebase state`, the `read` accessors), stays a
synonym. Where `--format` names the format of a file the verb writes
(`report run`'s table, `read table` / `report table` with `--output`), `--json`
leaves it alone. A typed error is printed as its `ftmw/error@1` dict under
`--json`, under those synonyms, and under a `--format json` that formats what
`read table` / `read meta` / `report table` print; never because a written
file's format is `json` (`report run --format json`, a dump with `--output`).
Under
`--json`:
- a `read` accessor prints its envelope, as today;
- a stage-running or curation verb prints
  `{"schema": "ftmw/run_result@1", "verb": "<object> <verb>", "stage": <canonical name or null>,
  "invalidated": [...], "summary": {...}}`. `summary` holds the scalars the
  human output reports (counts, chosen values, paths written), and at most
  flat objects of counts, never arrays. `stage` is `null` for a write that is
  not one stage (`settings`, `clocks`, `start run`, `run`, `report run`), and
  stage names inside `summary` are canonical;
- any other verb prints its natural payload through `to_jsonable`;
- an error prints its `ftmw/error@1` dict on stderr, as `--format json` does
  today.

Through the CLI, `--output` for a `read` accessor names a directory; each
array field is written there as `<field path>.npy` and the JSON envelope
names the file in the array's place. An accessor whose result holds arrays
refuses to run without `--output`.

## Capabilities

`capabilities()` → `{"schema": "ftmw/capabilities@1", "contract_version": int,
"schemas": [...], "accessors": [...], "codes": [...], "stages": [...],
"metadata_keys": [...], "tables": {name: [columns]}, "fields": {type: [fields]},
"vocabularies": {name: [values]}, "file_bound": {accessor: bool},
"pipeline_names": {accessor: name}}`. Every group of the manifest the contract
tests check appears, so a client can discover the surface without importing the
package. `read capabilities` prints it, and no file is needed.

## Open questions

None open. Resolved: CLI arrays are `.npy` (§Accessors); the whole-spectrum
model is §Spectrum model; a time-domain model is not wanted.
