"""Fast, data-free invariants behind the Stage 5 fitting methods note.

These guard the three synthetic claims the :doc:`methods note
<methods/stage5_fitting>` rests on without touching a fixture:

* the shipped penalized accept gate is **window-size invariant** while the
  classical F-test it replaced manufactures significance from quiet padding;
* the :math:`\\sigma_\\text{eff}` weighting **discounts** an absorber sitting
  under a bright model component while leaving a real line's evidence intact;
* a single cosine fit to a close blend leaves a large reduced-chi-squared
  **signature**, so a blend is always detectable.

The full seven-fixture cross-fixture roll-up lives in the ``slow`` suite
(``tests/integration/test_stage5_fitting_report.py``).
"""

import importlib.util
from pathlib import Path

import pytest

_GEN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "methods"
    / "stage5_fitting"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("stage5_fitting_generate", _GEN)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def test_penalized_gate_is_window_size_invariant():
    """The F-test p-value drops with padding; the penalized verdict does not."""
    inv = generate.gate_window_invariance()
    n_data = inv["n_data"]
    assert n_data == sorted(n_data)  # padding increases the window

    # The F-test manufactures significance: the marginal absorber's p-value
    # falls by orders of magnitude as the window pads.
    p_absorber = inv["pvalue"]["absorber"]
    assert all(b < a for a, b in zip(p_absorber, p_absorber[1:]))
    assert p_absorber[0] / p_absorber[-1] > 50.0

    # The penalized gate returns a single verdict for each candidate, with no
    # window-size axis: the absorber (Delta chi2 below the bar) is rejected, the
    # real line (above the bar) accepted, regardless of padding.
    assert inv["penalized_accept"]["absorber"] is False
    assert inv["penalized_accept"]["real"] is True
    assert inv["bar"] == pytest.approx(2.0 * inv["lambda"] * inv["delta_k"])


def test_sigma_eff_discounts_the_absorber():
    """Equal raw evidence; sigma_eff keeps the real line and crushes the absorber."""
    se = generate.sigma_eff_localization()
    # Both candidates start above the accept bar on raw evidence.
    assert se["raw_chi2"] > se["bar"]
    # A real line over quiet bins is essentially unchanged and accepted.
    assert se["eff_chi2_real"] == pytest.approx(se["raw_chi2"], rel=0.05)
    assert se["accept_real"] is True
    # An absorber under a bright line is discounted far below the bar, rejected.
    assert se["eff_chi2_absorber"] < 0.1 * se["bar"]
    assert se["accept_absorber"] is False


def test_blend_leaves_single_cosine_signature():
    """A close blend recovers to sub-kHz (K=2) and is always detectable (K=1)."""
    bl = generate.blend_study()
    one = bl["single_cosine_rchi2"]
    two = bl["two_line_rchi2"]
    err = bl["recovery_max_err_khz"]
    # The single-cosine fit is elevated by at least an order of magnitude at
    # every separation; the joint two-line fit sits near one.
    assert all(r is not None and r > 10.0 for r in one)
    assert all(r is not None and r < 1.5 for r in two)
    # K-known recovery holds below a kHz across the whole sweep, including the
    # tightest 0.3-FWHM blend.
    assert all(e is not None and e < 1.0 for e in err)
