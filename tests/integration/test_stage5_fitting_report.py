"""Regeneration guard for the Stage 5 fitting methods note.

Re-runs the synthetic half of the documentation harness (the shipped gate,
sigma_eff, and blend studies on fixed-seed simulators) and checks it against the
committed ``results.json``, then checks the cross-fixture snapshot's invariants.
The seven-fixture roll-up is a snapshot, not regenerated here -- a full Stage 5
fit on all seven fixtures is minutes -- but one light fixture (the reference) is
rebuilt live and checked against its snapshot so the snapshot cannot silently
rot. Regenerate the full snapshot by hand when the fit changes::

    python docs/source/methods/stage5_fitting/generate.py

Marked ``slow``: the live 2638 rebuild dominates. Run the default fast suite with
``-m "not slow"`` to skip it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCDIR = _ROOT / "docs" / "source" / "methods" / "stage5_fitting"
_RESULTS = _DOCDIR / "results.json"

_spec = importlib.util.spec_from_file_location(
    "stage5_fitting_generate", _DOCDIR / "generate.py"
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


def test_synthetic_claims_match(committed):
    """The data-free studies reproduce the committed numbers."""
    inv = generate.gate_window_invariance()
    assert inv["bar"] == pytest.approx(committed["gate_invariance"]["bar"])
    assert inv["pvalue"]["absorber"][-1] == pytest.approx(
        committed["gate_invariance"]["pvalue"]["absorber"][-1], rel=1e-6
    )

    se = generate.sigma_eff_localization()
    assert se["eff_chi2_absorber"] == pytest.approx(
        committed["sigma_eff"]["eff_chi2_absorber"], rel=1e-6
    )
    assert se["accept_absorber"] is False

    bl = generate.blend_study()
    # The blend study carries noise; check the structural claim, not the digits.
    assert all(r > 10.0 for r in bl["single_cosine_rchi2"])
    assert all(r < 1.5 for r in bl["two_line_rchi2"])


def test_cross_fixture_snapshot_invariants(committed):
    """The committed seven-fixture snapshot still encodes the note's claims."""
    cf = committed["cross_fixture"]
    fx = cf["fixtures"]
    # All seven fixtures are present and fit a healthy number of windows.
    assert set(fx) == set(generate.FIXTURES)
    for name, rec in fx.items():
        assert rec["n_windows"] > 0
        assert 0.0 <= rec["pass_rate"] <= 1.0
    # The SNR-aware gate clears the great majority of windows on every fixture.
    assert all(rec["pass_rate"] >= 0.6 for rec in fx.values())
    # Recall is reported on the two vinyl-cyanide fixtures and is nonzero.
    for name in generate.CATALOG_FIXTURES:
        assert fx[name].get("recall", 0.0) > 0.0
    # The scatter cloud spans the full SNR range the note claims (>=3 decades).
    snr = cf["windows"]["snr_max"]
    assert max(snr) / max(min(s for s in snr if s > 0), 1e-6) > 1e3


def test_2638_snapshot_matches_live_rebuild(committed):
    """Rebuild the reference fixture live and check it against the snapshot."""
    import os
    import tempfile

    from ftmwpipeline._internal.stage5_validation_impl import (
        validate_stage5_shape_error_impl,
    )

    cwd = os.getcwd()
    os.chdir(_ROOT)
    try:
        with tempfile.TemporaryDirectory(prefix="stage5-note-test-") as td:
            path = generate._build_fixture("2638", Path(td) / "2638")
            rep = validate_stage5_shape_error_impl(path)
    finally:
        os.chdir(cwd)

    snap = committed["cross_fixture"]["fixtures"]["2638"]
    t1 = rep["tier1"]
    # The fit is deterministic, so the live rebuild reproduces the snapshot.
    assert t1["n_windows"] == snap["n_windows"]
    assert t1["chi2r_median"] == pytest.approx(snap["chi2r_median"], rel=0.02)
    assert t1["pass_rate"] == pytest.approx(snap["pass_rate"], abs=0.02)
