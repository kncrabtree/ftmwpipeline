"""Fast, data-free invariants behind the matched-filter detection note.

These guard the two claims the :doc:`methods note <methods/matched_filter_detection>`
rests on without building any fixture:

* the sigma-weighted Lorentzian projection at every frequency equals the
  exponentially-apodized FFT (the identity that makes the matched filter one
  ``O(N log N)`` transform, not N projections);
* in the narrow-line regime real FTMW data sits in (FWHM ~ 1-2 bins), the
  matched-filter kernel recovers weak lines a strong-window (Blackman-Harris)
  detector loses.

The full real-fixture (2638) regeneration lives in the ``slow`` suite
(``tests/integration/test_matched_filter_report.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np

# Load the doc harness by path (it lives under docs/, not an importable package).
_GEN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "methods"
    / "matched_filter_detection"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("matched_filter_generate", _GEN)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


def test_projection_equals_apodized_fft():
    """The explicit O(N^2) projection equals the apodized FFT to precision."""
    ident = generate.projection_fft_identity(seed=generate.SEED)
    assert ident["rel_error"] < 1e-10


def test_matched_filter_beats_blackman_harris_in_narrow_regime():
    """At FWHM/bin = 1.34, SNR = 4 the matched filter recovers what BH loses."""
    mf_recall = []
    bh_recall = []
    for trial in range(4):
        sim = generate.simulate_fid(
            fwhm_bins=1.34, true_snr=4.0, seed=generate.SEED + 1000 * trial
        )
        tol = max(1.0, 1.34 / 2.0) * sim["bin_mhz"]
        mf = generate.matched_filter_candidates(
            sim, tau_basis_us=2.0 * sim["tau_truth_us"]
        )
        bh = generate.blackman_harris_candidates(sim)
        mf_recall.append(generate._recall_fp(mf, sim["truth_mhz"], tol)[0])
        bh_recall.append(generate._recall_fp(bh, sim["truth_mhz"], tol)[0])
    mf_mean = float(np.mean(mf_recall))
    bh_mean = float(np.mean(bh_recall))
    assert mf_mean > 0.9  # matched filter recovers nearly every weak line
    assert mf_mean > bh_mean + 0.3  # and clearly beats the strong-window detector
