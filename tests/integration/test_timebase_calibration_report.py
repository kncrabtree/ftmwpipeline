"""Regeneration guard for the timebase-calibration methods note.

Re-runs the documentation harness — the synthetic recovery and Cramer-Rao
checks plus the 2638 application (import through the FT, then calibrate) — and
checks the results against the committed ``results.json``. Regenerate by hand
when the estimator changes::

    python docs/source/methods/timebase_calibration/generate.py

Marked ``slow``: the 2638 build dominates. Run the default fast suite with
``-m "not slow"`` to skip it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCDIR = _ROOT / "docs" / "source" / "methods" / "timebase_calibration"
_RESULTS = _DOCDIR / "results.json"

_spec = importlib.util.spec_from_file_location(
    "timebase_generate", _DOCDIR / "generate.py"
)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def committed():
    return json.loads(_RESULTS.read_text())


@pytest.fixture(scope="module")
def regenerated_2638():
    import os

    cwd = os.getcwd()
    os.chdir(_ROOT)
    try:
        return generate.run_2638()
    finally:
        os.chdir(cwd)


def test_results_file_is_committed():
    assert _RESULTS.is_file(), "results.json missing; run generate.py"


def test_synthetic_recovery_matches(committed):
    rec = generate.synthetic_recovery(seed=generate.SEED)
    for got, ref in zip(rec["recovered"], committed["synthetic_recovery"]["recovered"]):
        assert got["epsilon"] == pytest.approx(ref["epsilon"], abs=1e-9)


def test_2638_epsilon_matches(regenerated_2638, committed):
    d = regenerated_2638
    ref = committed["2638"]
    assert d["lattice_g_mhz"] == pytest.approx(ref["lattice_g_mhz"])
    assert d["epsilon_ppm"] == pytest.approx(ref["epsilon_ppm"], abs=0.05)
    assert d["sigma_epsilon_ppm"] == pytest.approx(ref["sigma_epsilon_ppm"], rel=0.2)
    assert d["n_used"] == ref["n_used"]
    # The scale error is a real, non-null tilt dominated by the out-of-band tones.
    assert d["epsilon_ppm"] > 1.0
    assert d["f_bb_max_mhz"] > 10000.0
