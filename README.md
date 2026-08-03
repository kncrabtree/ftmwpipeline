# ftmwpipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Documentation Status](https://readthedocs.org/projects/ftmwpipeline/badge/?version=devel)](https://ftmwpipeline.readthedocs.io/en/devel/)

A Python package for FTMW (Fourier Transform Microwave) spectroscopy signal
processing and peak fitting. No zero-padding, no window functions, just
statistically-grounded analysis based on the complex FT.

- **Unbiased spectrum.** The Fourier transform is computed with no
  apodization, time-domain windowing, or zero-padding. Those operations trade
  frequency resolution for leakage suppression, bias the line shape, and
  correlate frequency bins so the noise and χ² statistics no longer carry
  their meaning. Keeping the transform native-length preserves resolution
  and the line shape, and keeps the per-bin statistics valid.
- **Rigorous from noise to fit.** A region-aware, per-bin noise estimate sets
  detection thresholds; lines are fit in the complex domain to analytic lineshape
  models over finite-duration windows. Fitted parameters carry meaningful
  uncertainties.
- **Reproducible and portable.** Each FID, its analysis settings, and results
  are contained in an `.ftmw` HDF5 file. The results can be exported into an HTML
  file with inline graphs for easy browsing and sharing, as well as `.csv` for
  tabular results.
- **One analysis, three interfaces.** A command-line interface, an
  object-oriented `Pipeline` class, and a stateless functional API are thin
  wrappers over one implementation and produce numerically identical results.

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
