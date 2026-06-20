# ftmwpipeline

A Python package for FTMW (Fourier Transform Microwave) spectroscopy signal
processing and peak fitting.

Each experiment is processed as a single, portable `.ftmw` file that progresses
through pipeline stages. The same functionality is available through a Python
API and a command-line interface.

## Status

Implemented and tested: **data import**, **FT processing**, and **noise
estimation** (pipeline Stages 0–2), via the CLI, the `Pipeline` class, and the
functional API. Peak detection, window assignment, and fitting are not yet
implemented.

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
from ftmwpipeline import process_experiment

result = process_experiment(
    "examples/blackchirp_data/2638/",
    "exp_2638.ftmw",
    ft_params={"trim": (26500, 40000)},
)
```

### Command line

```bash
ftmwpipeline data import   exp_2638.ftmw examples/blackchirp_data/2638/
ftmwpipeline ft run        exp_2638.ftmw --trim 26500:40000
ftmwpipeline ft show       exp_2638.ftmw --trim 26500:40000 --no-interactive
ftmwpipeline noise run     exp_2638.ftmw
ftmwpipeline noise show    exp_2638.ftmw

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
