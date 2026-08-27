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

The session object's own methods are a public contract, so its **class is
published** — exported at the package top level — even though the session is
obtained from the `Pipeline` method and never constructed directly. A caller
that must name the type (a type annotation, an `isinstance` check) would
otherwise have to import from `_internal`, which no public contract may
require. Publishing the name does not make the constructor a supported entry
point: the class is documented as obtained from its `Pipeline` method, and the
implementation stays wherever the verbs it hosts live.

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

The one class of exception is **introspection that is not about a file**: the
tunable-knob registry and the settings registry describe the pipeline itself,
not any experiment, and a path argument they would ignore would be a lie about
what the answer depends on. Such a function takes no path (`scan_list`,
`settings_defaults`), and its `Pipeline` counterpart is a staticmethod for
parity. These must be genuinely file-independent: a function whose answer can
differ per file belongs to the path-first rule, however convenient a fileless
default would be.

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

### A mutating verb refuses a value it cannot coerce

A verb that persists a caller-supplied value into the file must **reject** a
value it cannot coerce to the target's declared type, rather than storing a
best-effort reading of it. A silently mistyped value is worse than an error by
the same argument the derived-state rule makes: the caller believes it set one
thing, the next run computes against another, and neither side has a signal.
Type-driven coercion is not optional strictness — it is what makes the write
mean what the caller said.

Two requirements follow, and both are part of the published contract:

- **The native typed value is accepted.** A caller holding a `float`, a
  `bool`, or a list does not have to render it to a string and hope the parser
  reads it back the same way. Strings remain accepted — they are what a CLI
  has — but they are a second encoding of the same contract, not the only one.
- **Every accepted string encoding is documented at the verb**, per target
  type, so an integrator has a spelling to render *to* instead of guessing one
  from the parser's behavior. Where a value has no string spelling, that is
  stated too, along with the verb that expresses it instead.

`settings_set` / `settings_unset` are governed by this: values coerce to each
settings field's declared type (element-typed and arity-checked for tuple
fields), an uncoercible value raises without touching the file, and `None` —
the unset request — is deliberately spelled as the native value or as
`settings_unset`, never as a magic string.

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

### A constant whose value is per-file is published as its definition

Some published constants are not numbers at all: a tolerance expressing a
spectral distance is *defined* as a multiple of the active-FT bin spacing
(`SCIENCE_STRATEGY.md` Requirement 8), so it has no value until a file is
named. Publishing a resolved frequency for such a constant would republish
exactly the frozen-in-MHz mistake, and inviting the caller to resolve the
definition itself would recreate the two-reads-can-disagree surface the
single-definition rule exists to close.

The compliant shape is both halves, and only these two:

- **the definition** — the bin count — published as the constant, with its
  docstring stating that it must not be multiplied by a spacing the caller
  derived; and
- **one accessor** returning the resolved value for a named file, meeting the
  "derived state is read" requirements below.

The verbs' parameters then default to `None`, not to a number — a float default
would be one acquisition length's answer frozen into every other file's call —
and the resolution happens once, at the public boundary, with the resolved
value passed inward as a required argument. The resolved value must not also be
stamped onto result objects: one read is the point.

`REFIT_SNAP_TOL_BINS` (Stage 6 curation snap tolerance, 0.625 active-FT bins) is
published on these terms, canonically in `core/curation.py` and re-exported at
the package top level, with `api.refit_snap_tol_mhz` /
`Pipeline.refit_snap_tol_mhz` / `review snap-tolerance` as the accessor.

The same rule governs published *vocabularies* — the string sets a caller must
be able to pin a parser on, such as the frame names and the frequency-
calibration states. A published vocabulary is a named type, not a set of
literals a consumer transcribes from prose, and its members are stable: a
member may gain meaning but must not be renamed or silently reused.

## Derived state is read, never re-derived

Where the pipeline derives a fact about a file rather than storing it — most
importantly *which frame* the file's frequencies are in and by how much they
are corrected — that derivation is published as a read-only accessor. The
requirement is the constants rule applied to a computation: a consumer that
re-implemented the derivation from the persisted parts would re-implement the
precondition logic too, and would drift from it.

Such an accessor must be:

- **derived at call time**, from the same inputs the verbs consult, so it
  cannot disagree with what the pipeline will actually apply, and cannot
  report a value that has gone stale;
- **total**, degrading to the documented default for every input a file may
  legitimately lack, so it answers at any stage — including a file that has
  been through nothing but the import — rather than raising;
- **read-only**, safe to call on a file the caller may not write to.

Totality is bounded by what a legitimate file may lack, not by what an
arbitrary HDF5 file may lack. An accessor may refuse on a file that carries
none of the inputs its answer is *about* — but the refusal must be documented
at the accessor, as a decision, so a caller knows which of "a defined default"
and "an error" it is holding. Where refusing and defaulting are both defensible,
prefer whichever the pipeline itself would do: fabricating a value the verbs
would never actually use is worse than an error.

A stage's own persisted artifact (`load_*`) does not satisfy this: it requires
the stage to have run and describes that run, not the file's current state.

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
