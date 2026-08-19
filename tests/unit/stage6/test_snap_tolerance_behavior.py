"""The curation verbs snap at the tolerance the accessor publishes -- at any
acquisition length.

``tests/unit/stage6/test_snap_tolerance_contract.py`` pins the *shape* of the
surface (one definition, one derivation). This module pins the behaviour, and
in particular the property that a suite at a single acquisition length cannot
check: the snap tolerance is a count of active-FT bins, so **the same absolute
offset must resolve differently on two files acquired at different lengths**.

Under the old flat 50 kHz it did not -- an offset either snapped on both files
or missed on both, which is precisely the failure
``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8 exists to remove.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.unit]


def _isolated_fitted_peak(
    path: Path, *, min_separation_mhz: float
) -> Tuple[int, float]:
    """A ``(window_id, frequency_mhz)`` with no other fitted peak nearer than
    ``min_separation_mhz``, so a snap test is about the tolerance and not about
    nearest-wins between two candidates."""
    with h5py.File(path, "r") as h5f:
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    everything = [
        float(p.frequency_mhz)
        for wf in spectrum_fit.window_fits
        for p in wf.fitted_peaks
    ]
    for wf in spectrum_fit.window_fits:
        if wf.window_id is None:
            continue
        for peak in wf.fitted_peaks:
            f = float(peak.frequency_mhz)
            others = [g for g in everything if g != f]
            if not others or min(abs(g - f) for g in others) > min_separation_mhz:
                return int(wf.window_id), f
    pytest.skip("no sufficiently isolated fitted peak in the fixture")


def test_verbs_snap_at_exactly_the_published_tolerance(stage5_small_file):
    """Just inside resolves; just outside is a hard error on ``remove``.

    This is the pairing guarantee an integrator builds on: the number
    :func:`ftmwpipeline.api.refit_snap_tol_mhz` reports is the number the verb
    actually uses, to the boundary.
    """
    tol = ftmw.refit_snap_tol_mhz(stage5_small_file)
    window_id, freq = _isolated_fitted_peak(
        stage5_small_file, min_separation_mhz=4 * tol
    )

    with pytest.raises(ValueError, match="(?i)no fitted peak|closest"):
        ftmw.review_edit(
            stage5_small_file, window_id, remove=[freq + 1.05 * tol], frame="raw"
        )

    result = ftmw.review_edit(
        stage5_small_file, window_id, remove=[freq + 0.95 * tol], frame="raw"
    )
    assert result.n_peaks_after == result.n_peaks_before - 1


@pytest.mark.slow
def test_same_offset_snaps_at_a_short_acquisition_and_misses_at_a_long_one(
    stage5_small_file, stage5_short_active_file
):
    """The cross-length invariant, on two real builds of the same experiment.

    ``end_us=8.0`` gives T_active = 5.65 us against the reference 12.65 us, so
    the resolved tolerance is 110.6 kHz against 49.4 kHz. An 80 kHz offset
    therefore lands *inside* the tolerance on the short build and *outside* it
    on the long one -- the same typed frequency, the same line, two answers,
    because the two files resolve different numbers of bins into 80 kHz.

    Assert the resolved values bracket the offset rather than hardcoding them:
    the point is the ordering, not the arithmetic (which
    ``test_snap_tolerance_contract`` checks exactly).
    """
    offset_mhz = 0.080
    tol_long = ftmw.refit_snap_tol_mhz(stage5_small_file)
    tol_short = ftmw.refit_snap_tol_mhz(stage5_short_active_file)
    assert tol_short > offset_mhz > tol_long, (tol_short, offset_mhz, tol_long)

    wid_long, freq_long = _isolated_fitted_peak(
        stage5_small_file, min_separation_mhz=4 * tol_short
    )
    wid_short, freq_short = _isolated_fitted_peak(
        stage5_short_active_file, min_separation_mhz=4 * tol_short
    )

    with pytest.raises(ValueError, match="(?i)no fitted peak|closest"):
        ftmw.review_edit(
            stage5_small_file, wid_long, remove=[freq_long + offset_mhz], frame="raw"
        )

    result = ftmw.review_edit(
        stage5_short_active_file,
        wid_short,
        remove=[freq_short + offset_mhz],
        frame="raw",
    )
    assert result.n_peaks_after == result.n_peaks_before - 1
