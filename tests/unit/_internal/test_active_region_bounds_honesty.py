"""The declared active-region length may never exceed the recording.

``1 / T_active`` is the active-FT bin spacing every bin-relative tolerance in
the pipeline resolves against (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8).
Two things derive the active region independently: ``active_acquisition_us``
computes ``end_us - start_us`` from the declared bounds, and
``active_region_bounds`` clamps a sample slice to the record. If those two can
disagree, the pipeline resolves its tolerances against the bin spacing of a
spectrum that does not exist -- silently, since nothing crashes and every
number stays finite.

These tests pin the agreement rather than assume it, and pin the refusal that
keeps a bad bound from being written in the first place.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.shared_utils import active_acquisition_us
from ftmwpipeline.fitting.active_ft import (
    active_ft_bin_spacing_mhz,
    active_region_bounds,
)

pytestmark = [pytest.mark.unit]

_DT_US = 0.002
_N_TOTAL = 6325  # 12.65 us, the reference active length


def _declared_vs_actual(start_us: float, end_us: float) -> tuple[float, float]:
    """The bin spacing each of the two derivations reports."""
    duration_us = _N_TOTAL * _DT_US
    acquisition_us = active_acquisition_us(duration_us, start_us, end_us)
    declared = active_ft_bin_spacing_mhz(acquisition_us)
    lo, hi = active_region_bounds(_N_TOTAL, _DT_US, start_us, end_us)
    actual = 1.0 / ((hi - lo) * _DT_US)
    return declared, actual


@pytest.mark.parametrize(
    "start_us,end_us",
    [
        (0.0, 12.65),  # the whole record, named explicitly
        (2.35, 12.65),  # bounds on sample boundaries
        (1.234567, 11.765433),  # bounds off sample boundaries
        (0.000013, 12.649987),  # off-boundary at both ends
    ],
)
def test_the_two_derivations_agree_within_the_record(start_us, end_us):
    """Sample quantization is the only difference allowed."""
    declared, actual = _declared_vs_actual(start_us, end_us)
    # One sample out of ~6300 is the quantization floor; anything larger means
    # the two derivations have genuinely diverged.
    assert declared == pytest.approx(actual, rel=1.0 / _N_TOTAL)


@pytest.mark.parametrize("end_us", [12.66, 20.0, 100.0, 1e6])
def test_an_end_past_the_record_cannot_inflate_the_length(end_us):
    """The case that used to diverge without limit.

    At ``end_us = 100`` on a 12.65 us record the declared spacing was 0.01 MHz
    against an actual 0.079 MHz -- every tolerance 8x too tight.
    """
    declared, actual = _declared_vs_actual(0.0, end_us)
    assert declared == pytest.approx(actual, rel=1.0 / _N_TOTAL)


def test_the_clamp_is_what_does_it():
    """Stated directly, so the invariant survives a refactor of the helper."""
    duration_us = _N_TOTAL * _DT_US
    assert active_acquisition_us(duration_us, 0.0, 1e9) == pytest.approx(duration_us)
    assert active_acquisition_us(duration_us, 2.35, 1e9) == pytest.approx(
        duration_us - 2.35
    )
    # An in-range bound is untouched -- the clamp must not become a floor.
    assert active_acquisition_us(duration_us, 2.35, 10.0) == pytest.approx(7.65)


@pytest.fixture
def imported(tmp_path):
    """A bare imported .ftmw of known duration."""
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = _DT_US
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(_N_TOTAL) * 0.01))
    out = tmp_path / "exp.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def test_stage1_refuses_an_end_past_the_record(imported):
    """Refused, not trimmed: leaving end_us unset already means 'all of it'."""
    with pytest.raises(Exception, match="past the end of the recording"):
        ftmw.compute_ft(str(imported), start_us=0.0, end_us=100.0)


def test_stage1_accepts_the_records_own_duration(imported):
    """The commonest explicit spelling of 'the whole record' must not trip the
    half-sample tolerance on a float-represented duration."""
    ftmw.compute_ft(str(imported), start_us=0.0, end_us=_N_TOTAL * _DT_US)
    assert ftmw.refit_snap_tol_mhz(str(imported)) == pytest.approx(
        0.625 / (_N_TOTAL * _DT_US)
    )


def test_a_bad_window_cannot_reach_the_published_tolerance(imported):
    """End to end: the reproduction from the audit, inverted."""
    honest = ftmw.refit_snap_tol_mhz(str(imported))
    with pytest.raises(Exception):
        ftmw.compute_ft(str(imported), start_us=0.0, end_us=100.0)
    assert ftmw.refit_snap_tol_mhz(str(imported)) == pytest.approx(honest)
