# Specification: Command-Line Interface

Status of this document: **normative specification**. It states requirements,
not current implementation state. Command names below are the intended
contract; where the code currently differs, see the divergence log in
[`ROADMAP.md`](ROADMAP.md).

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
so the surface is uniform and discoverable: every stage runs with `<object>
run` and is viewed with `<object> show`. An analysis stage requires its
predecessor to be complete and must fail with a clear dependency message
otherwise.

Each stage object also has a **`stageN` synonym** fully interchangeable with its
name (`ft run` ≡ `stage1 run`), so a user who thinks in stage numbers and one
who thinks in names reach the same command.

| Stage | Object (synonym) | Execute | Visualize |
|---|---|---|---|
| 0 Data import | `data` (`stage0`) | `import` | `show` |
| — Start detection | `start` | `run` | `show` |
| 1 FT processing | `ft` (`stage1`) | `run` | `show` |
| 2 Noise estimation | `noise` (`stage2`) | `run` | `show` |
| 2b τ calibration | `tau` (`stage2b`) | `run` | `show` |
| 3 Peak detection | `peaks` (`stage3`) | `run` | `show` |
| 4 Window assignment | `windows` (`stage4`) | `run` | `show` |
| 5 Fitting | `fit` (`stage5`) | `run` | `show` |

Notes:

- **Execute is uniform `run`**, with one deliberate exception: stage 0 *creates*
  a file rather than running on one, so its execute verb is `import`
  (`data import <source-path-or-dir> [--format <name>]`). `start` has no `stageN`
  synonym — start detection is the pre-FT step that produces `start_us` for
  stage 1, not a numbered stage.
- **A stage with output variants selects them with `--kind`**, not extra
  commands: `tau show --kind heatmap|distribution`. The default is the stage's
  primary view.
- **Stage-specific actions beyond run/show are named verbs on the object.** The
  only current case is the Stage 5 fit-quality report, `fit check` (the
  SNR-aware shape-error validation).
- **`tau run --gaussian`** selects the pure-Gaussian τ calibration variant.

### Meta objects

Cross-cutting tooling that operates *across* stages is grouped under its own
object and scoped by an optional dotted **selector** (a stage name or a
sub-block / field path, e.g. `noise`, `noise.window_mhz`, `stage2b.gaussian` —
the same selector grammar the knob registry uses). These do not belong to one
stage, so they are not stage objects.

| Object | Verb | Purpose |
|---|---|---|
| `settings` | `show <file> [selector]` | Resolved value + provenance layer (`.ftmw` / `.yml:<name>` / `recommended` / `default`) per setting |
| | `set <file> <knob> <value>` | Persist a chosen value into the `.ftmw` |
| | `export <file> <out.yml> [selector]` | Write the chosen values to a `.yml` preset block |
| `scan` | `list <file> [selector]` | List the tunable knobs |
| | `run <file> <knob>` | Sweep one knob across a grid and report the metric table (+ CSV / plot) |
| | `all <file> [selector]` | Sweep every knob in a stage / sub-block on its default grid |

The `settings` and `scan` verbs are explicit subcommands, so there is no
collision between a subcommand token and the file positional.

### Utility commands

File-global operations that are not stage actions remain **bare commands** (no
object):

| Command | Purpose |
|---|---|
| `info` | Pipeline-file provenance and stage status (`--format text\|json`) |
| `formats` | List available data formats |
| `validate` | Installation/environment check |
| `version` | Version and package information |

Pipeline-file integrity is reported through `info` (and the Python
`.validate()`); `validate` is reserved for installation checks.

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
variants are `--kind` options, not new commands; cross-cutting tools
(`settings`, `scan`) group their actions under their own object and scope with a
selector. The tables above are the contract; new commands must follow the same
scheme — a new stage adds an object with `run`/`show` (and a `stageN` synonym);
a new cross-cutting tool adds an object with named action subcommands.

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
