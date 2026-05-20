# Research reports

Standalone write-ups that document and justify load-bearing algorithmic
choices in the FTMW pipeline. Each report lives in its own subdirectory
with a self-contained reproducibility script (`prototype.py`) and the
figures it generates (`figures/`).

These are **archived findings**, not specifications. They explain *why*
the pipeline does what it does — the physics, the calibration, the
empirical sanity checks — at a level of detail that the normative
spec docs (`../*_STRATEGY.md`) and per-stage plans (`../planning/`)
deliberately omit. New reports are added when a load-bearing
algorithmic decision is made and want to be explained for future
readers.

## Available reports

- [Complex-edge coherence statistic](complex-edge-coherence/report.md) —
  how the windowing stage decides where one analysis window ends and
  the next begins. Derives the phase-coherent edge test, calibrates
  the null distribution and threshold, and verifies on synthetic
  ground truth and the 2638 fixture.
- [Noise-estimation heuristic audit](noise-heuristic-audit/report.md) —
  five load-bearing constants in the noise-estimation stage examined
  against synthetic ground truth and the 2638 fixture. Three preserved
  with empirical justification; two replaced with sample-count
  defaults derived from closed-form Rayleigh stability bounds.

## Conventions for new reports

- One topic per subdirectory. Pick a timeless directory name (the
  algorithm, the question), not the stage number or the planning-doc
  open-question label.
- The report (`report.md`) is the only durable narrative artifact.
  Write timelessly: refer to pipeline stages by concept ("the
  windowing stage", "the noise-estimation stage"), not by number.
- Reproducibility script (`prototype.py`) regenerates every figure
  in `figures/` and reads any inputs from the conventional fixture
  locations (e.g. `scratch/exp_2638.ftmw`). Document missing-input
  fallbacks at the top of the script.
- Large intermediate artifacts (sweep `.npz` blobs > a few MB) stay
  out of git — the script is enough to regenerate them on demand.
  Only the figures the report references are committed.
