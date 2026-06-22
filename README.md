# ftmwpipeline

A Python package for FTMW (Fourier Transform Microwave) spectroscopy signal
processing and peak fitting.

Each experiment is processed as a single, portable `.ftmw` file that progresses
through pipeline stages. The same functionality is available through a Python
API and a command-line interface.

## Status

The full pipeline is implemented and tested — **data import**, **FT
processing**, **noise estimation**, **τ calibration**, **peak detection**,
**window assignment**, **time-domain fitting**, **timebase calibration**, and
the **review / report** surface (pipeline Stages 0–6) — exposed identically
through the CLI, the `Pipeline` class, and the functional API.

For the precise, verified current state see [`STATUS.md`](STATUS.md). For
direction and specifications see [`dev-docs/ROADMAP.md`](dev-docs/ROADMAP.md).

## Installation

From a clone, using conda:

```bash
# Development environment (tests, linters, docs, viz tooling)
conda env create -f environment-dev.yml
conda activate ftmwpipeline-dev

# or the minimal runtime environment
conda env create -f environment.yml
conda activate ftmwpipeline
```

Both environment files install the package itself in editable mode.

## Quick start

### Python — Pipeline class

```python
from ftmwpipeline import Pipeline

# Create a new analysis from raw data (here: a Blackchirp experiment directory)
pipe = Pipeline.create("exp_2638.ftmw", source="examples/blackchirp_data/2638/")

fid = pipe.load_data()
complex_ft = pipe.compute_ft(trim=(26500, 40000))
noise = pipe.estimate_noise()

print(pipe.info())   # provenance, completed stages, next available stages
```

Open an existing analysis with `Pipeline.open("exp_2638.ftmw")`.

### Python — functional API

```python
import ftmwpipeline.api as ftmw

ftmw.import_data("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
complex_ft = ftmw.compute_ft("exp_2638.ftmw", trim=(26500, 40000))
noise = ftmw.estimate_noise("exp_2638.ftmw")
```

### Python — whole-experiment convenience

```python
from ftmwpipeline import Pipeline

# Drive a raw source through every stage (import → FT → noise → tau →
# peaks → windows → fit → timebase → review).
result = Pipeline.build(
    "examples/blackchirp_data/2638/",
    trim=(26500, 40000),
    output="exp_2638.ftmw",
)
```

### Command line

Run the whole experiment in one command (import → FT → noise → tau → peaks →
windows → fit → timebase → review):

```bash
ftmwpipeline run examples/blackchirp_data/2638/ --output exp_2638.ftmw --trim 26500:40000
```

Or drive it stage by stage with the object-verb grammar (each stage object takes
`run`/`show`):

```bash
ftmwpipeline data import   exp_2638.ftmw examples/blackchirp_data/2638/
ftmwpipeline ft run        exp_2638.ftmw --trim 26500:40000
ftmwpipeline noise run     exp_2638.ftmw
ftmwpipeline tau run       exp_2638.ftmw
ftmwpipeline peaks run     exp_2638.ftmw
ftmwpipeline windows run   exp_2638.ftmw
ftmwpipeline fit run       exp_2638.ftmw
ftmwpipeline report run    exp_2638.ftmw --output-dir scratch/report

ftmwpipeline info exp_2638.ftmw            # provenance and stage status
ftmwpipeline validate                      # check installation
ftmwpipeline version
```

## Example data

`examples/blackchirp_data/2638/` is a real Blackchirp experiment included for
testing and trying the pipeline. The canonical FT is unapodized and
native-length; only trim to the active region 26500–40000 MHz.

## Tests

```bash
conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts=""
```

(The `-o addopts=""` is required because coverage flags are configured in
`pyproject.toml`.)

## License

MIT — see [LICENSE](LICENSE).
