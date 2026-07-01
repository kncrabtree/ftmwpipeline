# ftmwpipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Documentation Status](https://readthedocs.org/projects/ftmwpipeline/badge/?version=devel)](https://ftmwpipeline.readthedocs.io/en/devel/)

A Python package for FTMW (Fourier Transform Microwave) spectroscopy signal
processing and peak fitting — from a raw free-induction decay to a fitted line
list with honest uncertainties.

The design goal is a **statistically faithful** measurement, not just a
plausible-looking spectrum:

- **Unbiased spectrum.** The canonical Fourier transform is computed with no
  apodization, time-domain windowing, or zero-padding. Those operations trade
  frequency resolution, bias the line shape, and interpolate bins so the noise
  and χ² statistics no longer mean what they should. Keeping the transform
  native-length preserves resolution and the line shape, and keeps the per-bin
  statistics valid for everything downstream.
- **Rigorous from noise to fit.** A region-aware, per-bin noise estimate sets
  detection thresholds; lines are fit in the time domain over finite-duration
  windows; and fitted parameters carry uncertainties propagated **with their
  correlations**, not as independent error bars. A reported precision reflects
  what the data actually constrains.
- **Reproducible and portable.** Each experiment is one self-contained `.ftmw`
  file. Given the file and a compatible package version, the analysis
  reproduces the same result on any machine — no external preset or side
  artifact can silently change it.
- **One analysis, three interfaces.** A command-line interface, an
  object-oriented `Pipeline` class, and a stateless functional API are thin
  wrappers over one implementation and produce numerically identical results;
  use whichever fits your workflow.

## Installation

`ftmwpipeline` is currently in open beta. To install,

```bash
pip install --pre ftmwpipeline
```

This pulls in the scientific Python stack (NumPy, SciPy, Matplotlib, pandas,
h5py) and the `blackchirp` loader dependency. Testing and documentation tooling
are available as extras (`[dev]`, `[docs]`).

To work from a clone, install in editable mode — directly with pip, or via the
provided conda environment files:

```bash
pip install -e ".[dev]"
# or
conda env create -f environment-dev.yml && conda activate ftmwpipeline-dev
```

## Quick start

Process a raw experiment end to end in one command:

```bash
ftmwpipeline run path/to/experiment/ --output exp.ftmw --trim 26500:40000
```

`--trim` selects the active spectral band (MHz). The same build from Python,
through the class API:

```python
from ftmwpipeline import Pipeline

result = Pipeline.build("path/to/experiment/", trim=(26500, 40000), output="exp.ftmw")
pipe = Pipeline.open(result["pipeline_file"])
print(pipe.info())   # provenance, completed stages, next available stages
```

or driven stage by stage, with the same calls available on the functional API
(`import ftmwpipeline.api as ftmw`):

```python
pipe = Pipeline.create("exp.ftmw", source="path/to/experiment/")
pipe.compute_ft(trim=(26500, 40000))
pipe.estimate_noise()
pipe.calibrate_tau()
pipe.detect_peaks()
pipe.assign_windows()
pipe.fit_peaks()
```

At the command line each stage runs with the object-verb grammar
(`<stage> run` / `<stage> show`):

```bash
ftmwpipeline data import  exp.ftmw path/to/experiment/
ftmwpipeline ft run       exp.ftmw --trim 26500:40000
ftmwpipeline noise run    exp.ftmw
ftmwpipeline tau run      exp.ftmw
ftmwpipeline peaks run    exp.ftmw
ftmwpipeline windows run  exp.ftmw
ftmwpipeline fit run      exp.ftmw
ftmwpipeline report run   exp.ftmw --output-dir report
```

## Documentation

Full documentation — installation, a worked quickstart on bundled example data,
a guide to each pipeline stage, and the CLI/API reference — is at
[ftmwpipeline.readthedocs.io/en/devel](https://ftmwpipeline.readthedocs.io/en/devel/).

## License

MIT — see [LICENSE](LICENSE).
