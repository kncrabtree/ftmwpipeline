"""Fast, data-free invariants behind the edge-coherence methods note.

These guard the two claims the :doc:`methods note <methods/edge_coherence>`
rests on without touching the fixture:

* the closed-form null mean ``sqrt(pi/4)`` of ``S_coh`` and its
  ``M``-independence;
* the ``sqrt(M)`` growth of the statistic over a coherent band, which fixes the
  ``T_edge`` threshold (a 1-sigma-per-bin leakage reaches the default 8 at the
  default band width 64).

The full 2638 regeneration lives in the ``slow`` suite
(``tests/integration/test_edge_coherence_report.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np

from ftmwpipeline.preprocessing.edge_coherence import NULL_MEAN

_GEN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "methods"
    / "edge_coherence"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("edge_coherence_generate", _GEN)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def test_null_mean_is_sqrt_pi_over_4_and_m_independent():
    """The clean-noise mean sits on sqrt(pi/4) at every band width."""
    nul = generate.null_calibration(seed=generate.SEED)
    means = [nul["mean"][str(m)] for m in nul["band_widths"]]
    for mean in means:
        assert mean == __import__("pytest").approx(NULL_MEAN, abs=0.03)
    # M-independence: the spread across band widths is tiny.
    assert max(means) - min(means) < 0.03


def test_signal_grows_as_sqrt_m():
    """A coherent band's S_coh grows as (L/sigma) sqrt(M); 1 sigma reaches 8 at M=64."""
    grw = generate.signal_growth(seed=generate.SEED)
    widths = grw["band_widths"]
    vals = grw["value"]["1.0"]  # L/sigma = 1
    # Monotone increasing with band width.
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # At M=64 a 1-sigma-per-bin band sits at the operating threshold (~8).
    i64 = widths.index(64)
    assert vals[i64] == __import__("pytest").approx(8.0, abs=1.2)
    # sqrt(M) scaling: doubling M scales the (coherent) value by ~sqrt(2).
    i16, i32 = widths.index(16), widths.index(32)
    assert vals[i32] / vals[i16] == __import__("pytest").approx(np.sqrt(2.0), rel=0.25)
