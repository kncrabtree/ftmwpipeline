"""Regeneration guard for the matched-filter detection methods note.

Re-runs the synthetic half of the documentation harness (the shipped
matched-filter and Blackman-Harris kernels on the fixed-seed simulator) and
asserts the recall claims still match the committed ``results.json`` within
tolerance, plus the projection-equals-apodized-FFT identity. The 2638 half is
checked against the committed values rather than rebuilt: a full Stage 5 fit on
the fixture is already exercised by the Stage 5 integration suite and would blow
the documentation-test time budget here. Regenerate the fixture numbers and
figures by hand when the detector changes::

    python docs/source/methods/matched_filter_detection/generate.py

Marked ``slow``: the synthetic sweep is a few seconds. Run the default fast
suite with ``-m "not slow"`` to skip it.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCDIR = _ROOT / "docs" / "source" / "methods" / "matched_filter_detection"
_RESULTS = _DOCDIR / "results.json"

_spec = importlib.util.spec_from_file_location(
    "matched_filter_generate", _DOCDIR / "generate.py"
)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def committed():
    return json.loads(_RESULTS.read_text())


def test_results_file_is_committed():
    assert _RESULTS.is_file(), "results.json missing; run generate.py"


def test_projection_identity_holds():
    assert generate.projection_fft_identity()["rel_error"] < 1e-10


def test_synthetic_recall_matches(committed):
    """The matched-filter and BH recall grids reproduce within tolerance."""
    regen = generate.synthetic_sweep()
    for det in ("matched_filter", "blackman_harris"):
        got = np.array(regen["recall"][det], float)
        exp = np.array(committed["synthetic_sweep"]["recall"][det], float)
        assert got.shape == exp.shape
        np.testing.assert_allclose(got, exp, atol=0.12)


def test_matched_filter_holds_the_narrow_regime(committed):
    """The matched filter keeps high recall where the BH detector collapses."""
    sweep = committed["synthetic_sweep"]
    snr = sweep["snr_axis"].index(4.0)
    fwhm = sweep["fwhm_axis"].index(1.34)
    mf = sweep["recall"]["matched_filter"][snr][fwhm]
    bh = sweep["recall"]["blackman_harris"][snr][fwhm]
    assert mf > 0.9
    assert mf > bh + 0.3


def test_2638_gap_pass_recovers_real_lines(committed):
    """On the fixture the matched-filter gap pass lifts real-line recall."""
    d = committed["2638"]
    assert d["n_fit_peaks"] > 0
    assert d["recall_primary_plus_gap"] > d["recall_primary_only"]
    assert d["n_gap_only_real_lines"] >= 1
