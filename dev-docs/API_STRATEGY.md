# Specification: Python API

Status of this document: **normative specification**. It states requirements,
not current implementation state.

## Scope

The Python API for ftmwpipeline. Two surfaces are provided over the same
implementation:

- an object-oriented **Pipeline class** bound to one `.ftmw` file;
- a stateless **functional API** operating on `.ftmw` file paths.

Both are thin; all stage logic resides in shared internal implementation
functions (see [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) and
[`CLI_STRATEGY.md`](CLI_STRATEGY.md) for the other consumers of that shared
core). The scientific invariants the API must uphold — faithful raw data, the
unbiased active FT, reproducibility — are specified in
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
and their return types are the code's to define; the requirement here is the
shared shape, not the list.

**Active-FT settings.** That the measured transform is unapodized, un-windowed,
and native-length is a scientific requirement specified in
[`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md), which also fixes the vocabulary:
the **active FT** is the single grid every measuring stage consumes, the
**magnitude-display FT** is the 2x zero-padded display spectrum, and the term
"canonical spectrum" is retired. The API consequence is that the user-chosen
Stage 1 settings are data selection and display/scaling only, and are persisted
as the experiment's active-region settings; every later stage operates on the
spectrum they define and none carries its own trim. Resolution order per setting
is **explicit caller override > persisted > import-time recommended default**.
Passing an explicit override stores it as the new persisted state and
invalidates any downstream stage results, which must be re-run.

### Introspection

- `info() -> dict` — provenance, validity, completed stages, next available
  stages. Must reflect the current on-disk state, including stages completed
  after the instance was constructed.
- `validate() -> dict` — integrity report.

Stage methods must reject execution when a required predecessor stage is not
complete, with an error naming the missing dependency.

### Amortized sessions

Some stages expose an additional, optional session surface for repeated
in-process calls against the same file — `Pipeline.review_session()` (Stage 6)
is the first of these. A session is a context manager that holds one
expensive, file-derived context (built once, synchronously, on entry) and
reuses it across every verb issued through it, instead of each call rebuilding
that context from scratch. It never changes what a verb returns or persists —
correctness does not depend on the session amortizing anything: every verb
re-validates a cheap on-disk fingerprint before trusting the cached context
and transparently rebuilds it, exactly like the sessionless method it wraps,
on any mismatch. Only latency differs; results are identical either way.

**This is deliberately a `Pipeline`-class-only surface**, not mirrored on the
functional API or the CLI:

- The functional API is defined above as *stateless* ("every function takes
  the `.ftmw` path as its first argument") — a call that hands back a live,
  in-process handle with retained memory does not fit that contract, and
  scripting/batch use (the functional API's stated purpose) rarely benefits
  from amortizing a single process's worth of edits against one file the way
  an interactive client issuing many small edits does.
- The CLI is stateless *between invocations* by its own principle
  (`CLI_STRATEGY.md`, Principles #4): each invocation is a separate process,
  so there is no process-lifetime handle for a session to hold open. A CLI
  command that wanted the same amortization would need a long-lived server
  process, which is out of scope here.

Neither omission is a cross-interface divergence: the CLI and the functional
API still produce results identical to the class API's session-hosted calls
for identical inputs (`API_STRATEGY.md`, Principles #4) — a session is a pure
latency optimization for a long-lived, in-process, file-bound caller (exactly
what `Pipeline` already is), not a new behavior. A client that wants the
amortization uses `Pipeline`; one that does not, or that cannot (a fresh CLI
process per call), pays full price per call and gets the same answer.

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

## Errors

A caller that must respond differently to different failures must be able to
tell them apart **by type**, never by matching the text of a message. Message
wording is a rendering for humans and is free to change; the exception type and
its attributes are the contract.

Concretely:

- Every failure mode a caller is expected to *route on* has a named exception
  class, exported from the package top level, carrying the facts of the failure
  as attributes so the caller can report the condition without re-parsing the
  message it was told not to parse.
- The classes form one family under a common base (`PipelineFileError`), so a
  caller may also catch the whole category.
- A class introduced for a failure that previously raised a builtin must
  subclass that builtin, so existing handlers keep working. Adding a type is
  always additive.
- The named refusals include: creating over a file with a different source
  (`PipelineExistsError`), running a stage whose predecessor is incomplete
  (`StageDependencyError`), a corrupt file (`PipelineCorruptionError`), a file
  format newer than the reader (`PipelineCompatibilityError`), and a Stage 6
  edit that would splice a fit across an analysis-epoch boundary
  (`AnalysisEpochMismatchError`).

Where a diagnostic is delivered as human-readable *lines* rather than an
exception (the environment-drift lists in `info()` / `validate()`), the machine-
readable part must be documented as a contract and the rest declared prose. The
convention is a `"<field_name>: "` prefix; nothing after the colon is stable.

## Public constants

Where an external tool must make the same decision the pipeline makes — most
importantly, resolving a requested frequency to "the peak at *f*" — the
tolerance that decision uses is published, not private.

A published constant must be the *single definition* the pipeline itself uses:
every public parameter defaulting to that behavior takes its default from the
constant, so reading it is provably reading the value the verbs will apply. A
public name that merely agrees with hardcoded literals is not compliant, and the
test suite must pin the identity rather than the equality.

Publishing is read-only by intent. Callers are expected to read the value at
call time rather than persist it; the constant's meaning is stable even when its
value is refined.

`REFIT_SNAP_TOL_MHZ` (Stage 6 curation snap tolerance) is published on these
terms, canonically in `core/curation.py` and re-exported at the package top
level.

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
