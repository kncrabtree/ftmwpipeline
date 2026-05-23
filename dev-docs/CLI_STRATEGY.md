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
   positional argument. No experiment IDs, no cache directories.
3. **Single responsibility.** One command performs one pipeline stage or one
   utility operation.
4. **Composable and scriptable.** Commands are stateless between invocations
   (state lives in the file), chainable, and automation-friendly.
5. **Single implementation.** CLI and Python API must produce identical
   results; divergence is a defect.

## Command form

```
ftmwpipeline <command> <pipeline_file.ftmw> [options]
```

Option names must match the corresponding Python API parameter names.

### Stage commands

One command per pipeline stage; analysis stages require their predecessor to be
complete and must fail with a clear dependency message otherwise.

| Stage | Command(s) |
|---|---|
| 0 Data import | `import-data`, `visualize-data` |
| 1 FT processing | `compute-ft`, `visualize-ft` |
| 2 Noise estimation | `estimate-noise`, `visualize-noise` |
| 3 Peak detection | `detect-peaks`, `visualize-peaks` |
| 4 Window assignment | `assign-windows`, `visualize-windows` |
| 5 Fitting | `fit-peaks`, `visualize-fit` |

### Utility commands

| Command | Purpose |
|---|---|
| `info` | Pipeline-file provenance and stage status (`--format text|json`) |
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

Commands use a verb-object form expressing intent and aligned with pipeline
stages; related commands share a consistent prefix (`visualize-*`). Creation
(`import-data`) is clearly distinguished from analysis. The command tables
above are the contract; new commands must follow the same scheme.

## Scripting

The CLI must support batch use: stable exit codes, parseable output, and
behavior independent of working directory beyond the paths given. Any file
artifacts a command writes must go to an explicit, user-controlled location,
not implicitly to the current directory.

## Testing

CLI behavior is validated by integration tests that invoke real commands as a
subprocess and by cross-interface tests asserting parity with the Python API
(see [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)).
