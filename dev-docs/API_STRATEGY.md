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
core). The scientific invariants the API must uphold — faithful raw data, the
unbiased canonical spectrum, reproducibility — are specified in
[`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md) and are not restated here.

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

Three entry points with distinct, non-overlapping roles:

- **create** — start a new analysis from a raw source.
- **open** — attach to an existing analysis.
- **smart constructor** (`Pipeline(path)`) — open if the file exists, otherwise
  raise a clear error directing the user to create.

The role separation and the failure modes are the contract; the exact keyword
parameters (source selection, input format, loader options, force-overwrite)
are the code's to define.

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

Each stage exposes a method that loads its inputs from the file, applies the
stage with user-overridable parameters, persists its results and parameters
back to the same file, and returns the stage's result object; visualizing a
stage's output is a sibling method. Every stage method must reject execution
when a required predecessor is incomplete, raising an error that names the
missing dependency.

This contract is uniform and fixed: adding a stage adds a method that obeys it
without altering any existing signature. The roster of stages, their parameters,
and their return types are the code's to define and
[`../STATUS.md`](../STATUS.md)'s to record; the requirement here is the shared
shape, not the list.

**Canonical FT settings.** That the canonical transform is unapodized,
un-windowed, and native-length is a scientific requirement specified in
[`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md). The API consequence is that the
user-chosen Stage 1 settings are data selection and display/scaling only, and
are persisted as the experiment's canonical settings; every later stage operates
on the spectrum they define and none carries its own trim. Resolution order per
setting is **explicit caller override > persisted canonical > import-time
recommended default**. Passing an explicit override stores it as the new
canonical state and invalidates any downstream stage results, which must be
re-run.

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
flooded with these functions. The `Pipeline` class and the `api` module are
exported at the package top level; the whole-experiment workflow is
`api.run_pipeline` / `Pipeline.build` (and the `run` CLI verb), which drive a
raw source through every stage.

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
