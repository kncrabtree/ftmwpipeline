"""Per-window peak / covariance sort by ascending frequency.

:func:`~ftmwpipeline.fitting.result_conversion.sort_fitting_result_by_frequency`
reorders a window's fitted peaks by molecular frequency and applies the matching
block permutation to the parameter covariance. These tests pin the property that
matters: a covariance/correlation entry identified *by peak frequency* carries
the same value after the sort -- every peak keeps its own block, so ``sqrt(diag)``
still equals that peak's stored errors and off-diagonal cross-terms are preserved.
"""

from __future__ import annotations

import numpy as np

from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.fitting.result_conversion import (
    build_covariance_param_labels,
    sort_fitting_result_by_frequency,
)


def _encoded_covariance(dim: int) -> np.ndarray:
    """Symmetric matrix whose every unordered (i, j) entry is unique.

    ``M[i, j] = 1000 * min(i, j) + max(i, j)`` -- symmetric, and distinct for
    each unordered index pair, so a permutation bug cannot accidentally land the
    right number in the wrong cell.
    """
    m = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            m[i, j] = 1000.0 * min(i, j) + max(i, j)
    return m


def _make_result(freqs, *, fit_tau, baseline_order=None) -> FittingResult:
    """A FittingResult with the given (fit-order) peak frequencies + a covariance.

    Each peak's amplitude/frequency/phase errors are set to ``sqrt`` of its own
    covariance-block diagonal, mirroring production (the documented invariant on
    :class:`FittingResult`), so the diagonal/error correspondence is testable.
    """
    n = len(freqs)
    labels = build_covariance_param_labels(n, fit_tau=fit_tau, baseline_order=baseline_order)
    cov = _encoded_covariance(len(labels))
    fr = FittingResult()
    peaks = []
    for i, f in enumerate(freqs):
        a = labels.index(f"amplitude_{i}")
        o = labels.index(f"offset_{i}")
        p = labels.index(f"phase_{i}")
        peaks.append(
            FittedPeak(
                peak_id=i,  # records the *seed* identity; survives the reorder
                frequency_mhz=float(f),
                amplitude=1.0,
                phase=0.0,
                amplitude_error=float(np.sqrt(cov[a, a])),
                frequency_error=float(np.sqrt(cov[o, o])),
                phase_error=float(np.sqrt(cov[p, p])),
            )
        )
    fr.fitted_peaks = peaks
    fr.covariance = cov
    fr.covariance_param_labels = labels
    return fr


def _entry_by_identity(fr: FittingResult, peak_id_a, kind_a, peak_id_b, kind_b) -> float:
    """Covariance entry between two parameters located by peak *identity*.

    ``peak_id`` is the stable seed id carried on the FittedPeak, so the same
    physical parameter pair is found regardless of the current row order.
    """
    labels = list(fr.covariance_param_labels)
    pos = {pk.peak_id: i for i, pk in enumerate(fr.fitted_peaks)}
    ia = labels.index(f"{kind_a}_{pos[peak_id_a]}")
    ib = labels.index(f"{kind_b}_{pos[peak_id_b]}")
    return float(fr.covariance[ia, ib])


def test_sorts_peaks_and_preserves_each_peak_block():
    fr = _make_result([30.0, 10.0, 20.0], fit_tau=True)
    # Capture the diagonal-error correspondence by identity before the sort.
    before = {pk.peak_id: (pk.amplitude_error, pk.frequency_error, pk.phase_error) for pk in fr.fitted_peaks}

    sort_fitting_result_by_frequency(fr)

    # Peaks now ascending by frequency.
    assert [pk.frequency_mhz for pk in fr.fitted_peaks] == [10.0, 20.0, 30.0]
    # Labels are positional, so they are unchanged.
    assert list(fr.covariance_param_labels) == build_covariance_param_labels(3, fit_tau=True, baseline_order=None)
    # Every peak still owns its block: sqrt(diag at its new position) equals the
    # errors it carried before (the values travelled with the peak).
    labels = list(fr.covariance_param_labels)
    for pos, pk in enumerate(fr.fitted_peaks):
        a = labels.index(f"amplitude_{pos}")
        o = labels.index(f"offset_{pos}")
        p = labels.index(f"phase_{pos}")
        exp_a, exp_f, exp_p = before[pk.peak_id]
        assert np.sqrt(fr.covariance[a, a]) == exp_a
        assert np.sqrt(fr.covariance[o, o]) == exp_f
        assert np.sqrt(fr.covariance[p, p]) == exp_p
        # And the peak's own stored errors are untouched by the reorder.
        assert (pk.amplitude_error, pk.frequency_error, pk.phase_error) == before[pk.peak_id]


def test_offdiagonal_crossterms_preserved_by_identity():
    fr = _make_result([30.0, 10.0, 20.0], fit_tau=True)
    # cov(amplitude of the 10 MHz peak, phase of the 30 MHz peak) -- by identity.
    before = _entry_by_identity(fr, 1, "amplitude", 0, "phase")
    tau_before = float(fr.covariance[-1, -1])

    sort_fitting_result_by_frequency(fr)

    after = _entry_by_identity(fr, 1, "amplitude", 0, "phase")
    assert after == before
    # The shared-tau tail entry is unmoved.
    assert float(fr.covariance[-1, -1]) == tau_before


def test_full_matrix_is_a_pure_permutation():
    fr = _make_result([30.0, 10.0, 20.0], fit_tau=True, baseline_order=1)
    original = np.array(fr.covariance, copy=True)
    sort_fitting_result_by_frequency(fr)
    # The sorted matrix is a symmetric permutation of the original: same multiset
    # of entries, and the tau + baseline tail block is bit-for-bit unchanged.
    assert sorted(fr.covariance.flatten().tolist()) == sorted(original.flatten().tolist())
    tail = 3 * 3  # 3 peaks * 3 params; tau + baseline tail begins here
    np.testing.assert_array_equal(fr.covariance[tail:, tail:], original[tail:, tail:])


def test_idempotent_and_noop_when_already_sorted():
    fr = _make_result([10.0, 20.0, 30.0], fit_tau=True)
    cov0 = np.array(fr.covariance, copy=True)
    sort_fitting_result_by_frequency(fr)  # already ascending -> untouched
    np.testing.assert_array_equal(fr.covariance, cov0)

    fr2 = _make_result([30.0, 10.0, 20.0], fit_tau=True)
    sort_fitting_result_by_frequency(fr2)
    cov_once = np.array(fr2.covariance, copy=True)
    order_once = [pk.frequency_mhz for pk in fr2.fitted_peaks]
    sort_fitting_result_by_frequency(fr2)  # second call is a no-op
    np.testing.assert_array_equal(fr2.covariance, cov_once)
    assert [pk.frequency_mhz for pk in fr2.fitted_peaks] == order_once


def test_sorts_peaks_when_no_covariance():
    fr = _make_result([30.0, 10.0, 20.0], fit_tau=True)
    fr.covariance = None
    fr.covariance_param_labels = None
    sort_fitting_result_by_frequency(fr)
    assert [pk.frequency_mhz for pk in fr.fitted_peaks] == [10.0, 20.0, 30.0]


def test_malformed_covariance_leaves_both_untouched():
    fr = _make_result([30.0, 10.0, 20.0], fit_tau=True)
    # Truncate the covariance so its layout no longer matches the peak count.
    fr.covariance = np.asarray(fr.covariance)[:4, :4]
    order_before = [pk.frequency_mhz for pk in fr.fitted_peaks]
    cov_before = np.array(fr.covariance, copy=True)
    sort_fitting_result_by_frequency(fr)
    # Neither peaks nor covariance reordered -- they stay in correspondence.
    assert [pk.frequency_mhz for pk in fr.fitted_peaks] == order_before
    np.testing.assert_array_equal(fr.covariance, cov_before)
