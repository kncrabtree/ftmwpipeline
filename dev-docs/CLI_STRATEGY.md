# Specification: Command-Line Interface

Status of this document: **normative specification**. It states requirements,
not current implementation state. Command names below are the intended
contract.

## Scope

The `ftmwpipeline` command-line interface. The CLI is a thin wrapper over the
shared implementation core (the same core used by the Python API, per
[`API_STRATEGY.md`](API_STRATEGY.md)). It contains no analysis logic of its
own.

## Principles

1. **Thin wrapper.** CLI commands parse arguments, call shared implementation
   functions, format output, and map errors to exit codes. Nothing more.
2. **File-centric.** Every operation acts on a `.ftmw` file given as the first
   positional argument after the verb. No experiment IDs, no cache directories.
3. **Single responsibility.** One invocation performs one pipeline-stage action
   or one utility operation.
4. **Composable and scriptable.** Commands are stateless between invocations
   (state lives in the file), chainable, and automation-friendly.
5. **Single implementation.** CLI and Python API must produce identical
   results; divergence is a defect.

## Command form

The CLI is an **object-verb** grammar: the first token names what is acted on
(a pipeline stage, or a cross-cutting tool), the second names the action.

```
ftmwpipeline <object> <verb> <pipeline_file.ftmw> [options]
```

Option names must match the corresponding Python API parameter names. A small
set of bare utility commands (no object) is the only exception (see *Utility
commands*).

### Stage objects

The pipeline stages are the primary objects. Every stage object accepts the
same two verbs — `run` (execute the stage) and `show` (visualize its output) —
so the surface is uniform and discoverable. Each stage object also carries a
**`stageN` synonym** fully interchangeable with its name (e.g. `ft run` ≡
`stage1 run`), so a user who thinks in stage numbers and one who thinks in names
reach the same command. An analysis stage requires its predecessor to be
complete and must fail with a clear dependency message otherwise.

Three rules govern the stage surface; the roster of stage objects itself is the
code's:

- **Execute is uniform `run`.** The one deliberate exception is data import: it
  *creates* a file from a raw source rather than running on an existing one, so
  its execute verb names that act and takes the source — not a `.ftmw` — as
  input. A pre-stage step that is not itself a numbered stage may omit the
  `stageN` synonym.
- **Output variants are `--kind` options, not extra commands.** A stage with
  more than one view selects among them with `--kind`; the default is the
  stage's primary view. Likewise a within-stage algorithm variant is a flag on
  `run`, not a separate object.
- **A stage-specific action beyond run/show is a named verb on that object**,
  used only where the action genuinely has no run/show form (for example a
  fit-quality check).

### Cross-cutting objects

Tooling that operates *across* stages — for example settings management,
parameter scans, or driving the whole pipeline in one call — is grouped under
its own object with named action subcommands, rather than folded into a stage.
Where such an object addresses a particular stage or a field within one, it does
so with an optional dotted **selector** (a stage name or a sub-block / field
path, e.g. `noise` or `noise.window_mhz`) — the same selector grammar the knob
registry uses. The requirement is structural: each cross-cutting object groups
its actions under its own object name and uses explicit named subcommands, so a
subcommand token never collides with the file positional. The set of such
objects is the code's.

### Utility commands

File-global operations that are not stage actions — environment or
single-file utilities such as an installation check, a formats listing, a
version report, or file provenance/status — remain **bare commands** (no
object). Pipeline-*file* integrity is reported through the provenance/status
command (and the Python `.validate()`); the installation check is reserved for
*environment* validation, not file integrity.

## Output and errors

- Human-readable progress and result summaries on success.
- A machine-readable (JSON) output mode for information/introspection commands,
  for integration and scripting.
- Errors are actionable: state what failed and the concrete command to fix it
  (e.g. the import command to run when a file is missing, or the predecessor
  stage to run when a dependency is unmet).
- Plain-text output only. No decorative emoji or Unicode ornamentation.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | User error (bad parameters, missing file, unmet dependency) |
| 2 | Processing error (corrupt data, algorithm failure) |
| 130 | Interrupted by user |

## Naming

Commands use an **object-verb** form: the object (a pipeline stage or a
cross-cutting tool) first, the action second. Stage objects share a uniform
two-verb vocabulary (`run` / `show`) plus their `stageN` synonym; output
variants are `--kind` options, not new commands; cross-cutting tools group their
actions under their own object and scope with a selector. These rules are the
contract; new commands must follow the same scheme — a new stage adds an object
with `run`/`show` (and a `stageN` synonym); a new cross-cutting tool adds an
object with named action subcommands.

## Scripting

The CLI must support batch use: stable exit codes, parseable output, and
behavior independent of working directory beyond the paths given. The uniform
`<stage> run` form makes whole-pipeline driving a simple loop over stage
objects. Any file artifacts a command writes must go to an explicit,
user-controlled location, not implicitly to the current directory.

## Testing

CLI behavior is validated by integration tests that invoke real commands as a
subprocess and by cross-interface tests asserting parity with the Python API
(see [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)).
