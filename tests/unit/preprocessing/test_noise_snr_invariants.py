"""Fast, data-free invariants behind the high-SNR noise note.

These guard the two claims the :doc:`methods note <methods/noise_snr_scaling>`
rests on without touching the fixtures:

* the analytic Rician correction table ``C(R)`` baked into the estimator is
  reproducible from the documented fixed-seed Monte-Carlo procedure;
* the scatter estimator follows a ``1/sqrt(N)`` noise (slope ~ -0.5) while a
  naive level estimate flattens (slope ~ 0) on a constant pedestal.

The full real-fixture regeneration lives in the ``slow`` suite
(``tests/integration/test_noise_snr_report.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ftmwpipeline.preprocessing.noise_estimation import (
    _SCATTER_C_TAB,
    _SCATTER_R_TAB,
)

# Load the doc harness by path (it lives under docs/, not an importable package).
_GEN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "methods"
    / "noise_snr_scaling"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("noise_snr_generate", _GEN)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def test_cr_table_matches_baked_constants():
    """The regenerated C(R) table reproduces the constants the estimator ships."""
    r_tab, c_tab = generate.build_cr_table(seed=0)
    assert r_tab.shape == _SCATTER_R_TAB.shape
    assert c_tab.shape == _SCATTER_C_TAB.shape
    # Same fixed seed and procedure -> bit-for-bit (allow a hair for any BLAS drift).
    np.testing.assert_allclose(r_tab, _SCATTER_R_TAB, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(c_tab, _SCATTER_C_TAB, rtol=1e-6, atol=1e-9)


def test_cr_table_spans_the_rician_regime():
    """C(R) runs from ~1.0 under strong lines to ~1.47 in the Rayleigh limit."""
    _, c_tab = generate.build_cr_table(seed=0)
    assert c_tab.min() < 1.02
    assert c_tab.max() > 1.45


def test_synthetic_sqrtn_slope_split():
    """Scatter follows 1/sqrt(N) (slope ~ -0.5); naive level flattens (~0)."""
    slopes = generate.synthetic_sqrtn_slopes(seed=0)
    assert slopes["slope_scatter"] == pytest.approx(-0.5, abs=0.05)
    assert abs(slopes["slope_naive"]) < 0.05
