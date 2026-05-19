# Specification: Python API

Status of this document: **normative specification**. It states requirements,
not current implementation state. Where the code diverges, see the divergence
log in [`ROADMAP.md`](ROADMAP.md).

## Scope

The Python API for ftmwpipeline. Two surfaces are provided over the same
implementation:

- an object-oriented **Pipeline class** bound to one `.ftmw` file;
- a stateless **functional API** operating on `.ftmw` file paths.

Both are thin; all stage logic resides in shared internal implementation
functions (see [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) and
[`CLI_STRATEGY.md`](CLI_STRATEGY.md) for the other consumers of that shared
core).

## Principles

1. **One file, one experiment.** A `.ftmw` file holds one complete analysis
   progressing through pipeline stages. Users address experiments by file
   path, never by experiment ID or cache directory.
2. **Explicit creation vs. opening.** Creating a new analysis and working with
   an existing one are distinct operations with distinct failure modes.
3. **Safe re-execution.** Re-running code (e.g. a notebook cell) must not
   destroy completed downstream work.
4. **Single implementation.** The class API, functional API, and CLI must
   produce identical results for identical inputs; they share one core.
5. **No silent behavior.** Any action that could discard work requires
   explicit intent and informs the user of what happened.

## Pipeline class

A `Pipeline` instance is bound to exactly one `.ftmw` file for its lifetime.

### Construction

- `Pipeline.create(path, source, *, format_name=None, fid_index=None,
  force=False, **loader_params) -> Pipeline` — create a new analysis from raw
  data.
- `Pipeline.open(path) -> Pipeline` — open an existing analysis.
- `Pipeline(path)` — convenience constructor: open if the file exists,
  otherwise raise a clear error directing the user to `create`.

`create` semantics:

| Situation | Required behavior |
|---|---|
| File does not exist | Create it from `source`. |
| File exists, identical source | Load existing; inform the user it was reused. Do not reprocess. |
| File exists, different source | Raise `PipelineExistsError` with actionable options. |
| File exists, `force=True` | Overwrite, with a warning. |

Source identity is determined from recorded provenance (see Provenance).

`open` raises `FileNotFoundError` with guidance (how to create one, including
the CLI form) when the file is absent, and a corruption error when the file is
present but unreadable as a pipeline file.

### Stage methods

Each implemented stage exposes a method that loads its inputs from the file,
applies the stage with user-overridable parameters, persists results/parameters
to the same file, and returns the stage result object:

- `load_data() -> FID`
- `compute_ft(...) -> ComplexFT`
- `visualize_ft(...)`
- `estimate_noise(...) -> NoiseResult`
- `visualize_noise(...)`
- `detect_peaks(...) -> list[Peak]`

Future stages (window assignment, fitting) follow the same contract and are
added without changing existing signatures.

**Canonical FT settings.**  The user-chosen Stage 1 FT processing parameters —
`start_us`, `end_us`, `zpf`, `expf_us`, `window_function`, `units_power`,
`rdc`, and the frequency `trim` range — are persisted in the `.ftmw` file as
the experiment's canonical settings.  All later stages operate on the spectrum
they define; no stage carries its own trim or zpf.  Resolution order for each
setting is: **explicit caller override > persisted canonical > import-time
recommended default**.  Passing explicit overrides to `compute_ft` stores them
as the new canonical state and invalidates any downstream stage results (they
must be re-run).

### Introspection

- `info() -> dict` — provenance, validity, completed stages, next available
  stages. Must reflect the current on-disk state, including stages completed
  after the instance was constructed.
- `validate() -> dict` — integrity report.

Stage methods must reject execution when a required predecessor stage is not
complete, with an error naming the missing dependency.

## Functional API

A stateless surface where every function takes the `.ftmw` path as its first
argument and shares the class API's implementation. It is the appropriate
surface for batch and scripting use. Provided operations mirror the class:
import, load, compute FT, visualize, estimate noise, save parameters, and
introspection.

The canonical functional namespace is `ftmwpipeline.api`
(`import ftmwpipeline.api as ftmw`); the package top level is intentionally not
flooded with these functions. Only the whole-experiment convenience wrappers
`process_experiment` and `batch_process_experiments` (and the `Pipeline` class)
are exported at the package top level.

## Provenance

Every `.ftmw` file records, for the data it was created from: source path,
source modification time, a content/identity hash, import timestamp, format
name, and loader parameters. This record is the basis for safe-reimport
detection and reproducibility, and must be sufficient to detect whether a
re-import refers to the same source.

## Interactive use

The API must be safe to drive from notebooks: opening is idempotent,
re-creating with an identical source is non-destructive, and parameter
exploration (recomputing a stage with new parameters) never requires
re-importing source data.

## Extensibility

Adding a stage must not require changes to existing stage signatures, the file
format of prior stages, or the interface-sharing structure. Each new stage adds
one shared implementation plus thin class/functional/CLI wrappers and its own
serialization and tests.
