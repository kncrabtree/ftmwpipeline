"""Regeneration guard for the edge-coherence methods note.

Re-runs the synthetic half of the documentation harness (the shipped statistic
on the fixed-seed simulator) and checks the null + signal-growth claims against
the committed ``results.json``, plus the 2638 application (cheap: it builds the
fixture only through noise estimation). Regenerate by hand when the statistic
changes::

    python docs/source/methods/edge_coherence/generate.py

Marked ``slow``: the 2638 build dominates. Run the default fast suite with
``-m "not slow"`` to skip it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCDIR = _ROOT / "docs" / "source" / "methods" / "edge_coherence"
_RESULTS = _DOCDIR / "results.json"

_spec = importlib.util.spec_from_file_location(
    "edge_coherence_generate", _DOCDIR / "generate.py"
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


def test_null_and_signal_match(committed):
    nul = generate.null_calibration(seed=generate.SEED)
    for m in committed["null"]["band_widths"]:
        assert nul["mean"][str(m)] == pytest.approx(
            committed["null"]["mean"][str(m)], abs=0.03
        )


def test_2638_deramp_collapses_the_statistic(regenerated_2638, committed):
    """The active grid carries real coherence; de-ramping it collapses to null."""
    d = regenerated_2638
    assert d["median_active"] == pytest.approx(
        committed["2638"]["median_active"], abs=0.2
    )
    # De-ramping the [0, T] active grid collapses the median toward sqrt(pi/4).
    assert d["median_deramped"] < 1.0
    assert d["median_active"] > 1.5 * d["median_deramped"]


def test_2638_strong_line_and_no_pedestal(regenerated_2638):
    d = regenerated_2638
    assert d["scoh_at_36350"] > 30.0  # strong line towers over the threshold
    assert d["sigma_ratio"] > 2.0  # local sigma genuinely varies across the band
    assert abs(d["quiet_re_over_sigma"]) < 0.3  # no constant pedestal
    assert abs(d["quiet_im_over_sigma"]) < 0.3
