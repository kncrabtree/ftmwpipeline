"""Regeneration guard for the high-SNR noise methods note.

Re-runs the documentation harness over the checked-in fixtures and asserts the
key results still match the committed ``results.json`` within tolerance. If the
estimator (or a fixture) changes such that the note's claims no longer hold,
this fails and the note must be reviewed and regenerated::

    python docs/source/methods/noise_snr_scaling/generate.py

Marked ``slow``: it loads the multi-frame fixtures (a few seconds total). Run the
default fast suite with ``-m "not slow"`` to skip it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCDIR = _ROOT / "docs" / "source" / "methods" / "noise_snr_scaling"
_RESULTS = _DOCDIR / "results.json"

_spec = importlib.util.spec_from_file_location(
    "noise_snr_generate", _DOCDIR / "generate.py"
)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def regenerated():
    # Run from the repo root so the relative fixture paths resolve; no figures.
    import os

    cwd = os.getcwd()
    os.chdir(_ROOT)
    try:
        return generate.run(fixtures=list(generate.FIXTURES), figures=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def committed():
    return json.loads(_RESULTS.read_text())


def test_results_file_is_committed():
    assert _RESULTS.is_file(), "results.json missing; run generate.py"


def test_overestimate_and_snr_match(regenerated, committed):
    """Per-fixture peak SNR and naive/scatter overestimate are reproduced."""
    for fx, exp in committed["per_fixture"].items():
        got = regenerated["per_fixture"][fx]
        assert got["snr"] == pytest.approx(exp["snr"], rel=0.05)
        assert got["overestimate_naive_over_scatter"] == pytest.approx(
            exp["overestimate_naive_over_scatter"], rel=0.05
        )


def test_high_snr_fixture_shows_the_pedestal_failure(regenerated):
    """655: the naive level estimate flattens while the scatter falls as 1/sqrt(N)."""
    d = regenerated["per_fixture"]["655"]
    assert d["overestimate_naive_over_scatter"] > 3.0
    assert abs(d["slope_naive"]) < 0.1  # flat: pedestal-bound
    assert d["slope_scatter"] < -0.3  # falls with shot count


def test_scatter_slopes_track_one_over_sqrt_n(regenerated, committed):
    """Every multi-frame fixture's scatter slope stays near the committed value."""
    for fx, exp in committed["per_fixture"].items():
        if "slope_scatter" not in exp:
            continue
        got = regenerated["per_fixture"][fx]
        assert got["slope_scatter"] == pytest.approx(exp["slope_scatter"], abs=0.05)


def test_cr_table_summary_matches(regenerated, committed):
    for key in ("r_max", "c_max", "n_points"):
        assert regenerated["cr_table"][key] == pytest.approx(
            committed["cr_table"][key], rel=1e-6
        )
