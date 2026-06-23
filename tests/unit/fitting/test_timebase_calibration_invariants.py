"""Fast, data-free invariants behind the timebase-calibration methods note.

These guard the two claims the :doc:`methods note <methods/timebase_calibration>`
rests on without touching the fixture:

* the estimator recovers a planted scale error ``eps_true`` to well within its
  formal uncertainty and holds that recovery as the noise is raised;
* each kept tone's reported uncertainty equals the Cramer-Rao single-tone bound
  ``(sqrt(6)/pi) / (T snr)`` when the systematic floor is switched off.

The full 2638 regeneration lives in the ``slow`` suite
(``tests/integration/test_timebase_calibration_report.py``).
"""

import importlib.util
from pathlib import Path

import pytest

_GEN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "methods"
    / "timebase_calibration"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("timebase_generate", _GEN)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def test_recovers_planted_epsilon_across_noise():
    """The estimator returns eps_true within a fraction of its formal sigma."""
    rec = generate.synthetic_recovery(seed=generate.SEED)
    assert rec["eps_true"] == pytest.approx(2.0e-6)
    for row in rec["recovered"]:
        assert row["n_used"] >= 3
        # Recovered eps is within the formal uncertainty of the truth.
        assert abs(row["epsilon"] - rec["eps_true"]) < row["sigma_epsilon"]
        # And within 0.05 ppm in absolute terms.
        assert abs(row["epsilon"] - rec["eps_true"]) < 0.05e-6


def test_per_tone_sigma_is_the_cramer_rao_bound():
    """With the systematic floor off, each tone's sigma is (sqrt6/pi)/(T snr)."""
    crb = generate.crb_check(seed=generate.SEED)
    assert crb["n"] >= 3
    for ratio in crb["ratios"]:
        assert ratio == pytest.approx(1.0, abs=1e-6)
