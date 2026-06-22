# AGENTS.md

Canonical agent/contributor instructions for `ftmwpipeline`. This file holds
the durable invariants and how to operate the repo; it deliberately does **not**
restate program content that lives elsewhere and would drift.

## What this is

`ftmwpipeline` is a Python package for FTMW (Fourier Transform Microwave)
spectroscopy signal processing and peak fitting: it drives a raw free-induction
decay through a sequence of stages (import → FT → noise → τ calibration → peak
detection → window assignment → time-domain fitting → timebase → review/report)
to a fitted line list with honest uncertainties. Each experiment is one
self-contained, portable `.ftmw` (HDF5) file.

Where the detailed truth lives — consult these rather than trusting any summary:

- **`README.md`** — the scientific goals, for the user's-eye view. The code
  itself is the authority on *what is implemented*.
- **`dev-docs/SCIENCE_STRATEGY.md`** and the other **`dev-docs/*_STRATEGY.md`**
  specs — normative, timeless requirements (science, API, CLI, serialization,
  testing). The authority on *what must be true*.
- **`dev-docs/ROADMAP.md`** — coordination doc and the code-vs-spec divergence
  log.
- **`dev-docs/planning/`** — per-stage plans (see `planning/README.md` for the
  lifecycle); read the relevant one before changing a stage's behavior.

## Commands

The dev environment is the conda env `ftmwpipeline-dev` (from
`environment-dev.yml`, the superset with tooling + an editable install). The
package itself is pure-pip installable (`pip install -e ".[dev]"`); conda is a
local convenience, not a requirement.

```bash
conda env create -f environment-dev.yml                                      # one-time
conda run -n ftmwpipeline-dev python -m pytest -o addopts="" -m "not slow"    # inner-loop suite
conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts=""   # full suite
conda run -n ftmwpipeline-dev python -m pytest -o addopts="" tests/unit/io/test_fid_serialization.py::test_name
conda run -n ftmwpipeline-dev ftmwpipeline validate                          # installation check
```

`pyproject.toml` hardwires `--cov` flags into pytest `addopts`, so `-p no:cov`
alone breaks argument parsing — pass `-o addopts=""` to run without coverage
(pytest-cov is in the dev env, so plain `pytest` with coverage also works).
Test markers: `slow`, `integration`, `unit`, `performance`.

mypy is configured strict (`disallow_untyped_defs`, etc.); new code in `src/`
must be fully type-annotated. Line length is 88. The repo predates a clean
black/isort pass, so most files are not black-clean — a "would reformat" on a
file you did not touch is pre-existing debt, not your regression. Keep *new*
code locally black/isort/mypy-clean.

**Never commit run artifacts.** Tests must write only to pytest `tmp_path`. Any
command or test that emits files (plots, exports, scratch `.ftmw`/`.h5`,
coverage HTML) must be directed to an untracked location, never a tracked path.
(Machine-local mechanics for this are in `CLAUDE.local.md`.)

## The dual-interface invariant

This is the single most important architectural rule. Three user-facing
interfaces must behave **identically** and must not duplicate logic:

1. **CLI** — `src/ftmwpipeline/cli/` (object-verb grammar; entry point
   `ftmwpipeline.cli:main`).
2. **Pipeline class** — `src/ftmwpipeline/pipeline.py` (file-bound, OO).
3. **Functional API** — `src/ftmwpipeline/api.py` (stateless, path-first).

All three are **thin wrappers**: the real logic lives once under
`src/ftmwpipeline/_internal/`. The functional API and CLI generally delegate
through the `Pipeline` class, which delegates to `_internal`. When adding or
changing stage behavior, edit the `_internal` impl and let all three interfaces
inherit it — never patch one interface in isolation. Cross-interface
consistency tests gate every stage change; run them after one. The grammar and
the contracts are normative in `dev-docs/API_STRATEGY.md` and
`dev-docs/CLI_STRATEGY.md`.

## Divergence discipline

The known code-vs-spec divergences are resolved — code and specs currently
agree. If you find a *new* mismatch, do not silently paper over it: log it in
`dev-docs/ROADMAP.md` and resolve it deliberately, either amending the spec or
changing the code.

## Extending the pipeline (new stage)

Follow the established pattern, in order: add the algorithm/data structure → add
`_internal/stageN_impl.py` with dependency checking and HDF5 storage → register
the stage name + dependencies with the stage tracker → add serialization under
`io/` → expose it identically through `pipeline.py`, `api.py`, and a
`cli/` subcommand → add unit tests *and* a cross-interface consistency test.
Before starting, create the stage's planning doc in `dev-docs/planning/` and
register it in `dev-docs/ROADMAP.md`. The normative requirements for each piece
are in the `dev-docs/*_STRATEGY.md` specs.
