# Development scripts

Unsupported maintenance and validation utilities for working on
`ftmwpipeline` itself. These are **not** part of the package's public API or
CLI — they are not installed, not versioned as an interface, and may change or
be removed without notice. For normal use, drive the pipeline through the
`ftmwpipeline` CLI, the `Pipeline` class, or `ftmwpipeline.api` instead.

All of these run from the repository root in the `ftmwpipeline-dev` conda
environment and read the checked-in example experiments under
`examples/blackchirp_data/`. Anything they emit goes to the untracked
`scratch/` directory — never a tracked path.

## Tools

- **`build_fixture.py`** — Drive one example experiment end-to-end (Stages 0–6,
  optionally through timebase calibration and the Stage 6 review) and write a
  fresh `.ftmw`. Always builds from the raw data rather than patching an
  existing file, so a stale persisted setting can never mask a code default.
  This is the canonical way to (re)build a fixture for validation or
  benchmarking, and it produces the inputs the other dev tools consume.

  ```bash
  python scripts/development/build_fixture.py 2638 --output scratch/2638.ftmw
  ```

- **`catalog_recall.py`** — Score a finished fit's line list against a reference
  truth catalog (the `combined_lines.csv` format) with an injective
  nearest-match, sub-resolution blends collapsed, and a species/tag tier split,
  so the measure is not inflated by an over-splitting fit. The vinyl-cyanide
  example experiments (1512 and 655) share the catalog under
  `examples/blackchirp_data/vinyl-cyanide-reference/`.

  ```bash
  python scripts/development/catalog_recall.py FIT.ftmw \
      --catalog examples/blackchirp_data/vinyl-cyanide-reference/combined_lines.csv \
      --main-tag 53515
  ```
