"""The Stage 6 read-path budget, and the cheap readers' equivalence to the
full loader (T1-T3 of ``scratch/stage6-performance-plan.md``).

Stage 6 used to spend most of its wall clock re-reading the whole fitted-peak
table for values costing a fraction of a millisecond. ``5a80b8f`` replaced
those reads with per-column reads (``io/fitting_serialization.py``), which is
a **pure performance** change: reintroducing a full load breaks no assertion
anywhere else in the suite, because the value returned is identical either
way. These tests are the ones that would notice.

Three things are pinned here:

- **T1, the load budget.** A counter around
  :func:`~ftmwpipeline.io.fitting_serialization.load_spectrum_fit_from_hdf5`,
  attributed to its calling frame, asserts that the read-only verbs perform
  *no* full loads at all. A regression names the function that reintroduced
  one.
- **T2, reader equivalence.** Each cheap reader returns exactly what the full
  load returned on the same file. This is the correctness half: T1 alone
  would pass just as well if a reader returned nothing.
- **T3, a file predating ``peak_uid``.** The uid readers must report an empty
  set per window rather than raising, which is what makes a ``uid:N`` target
  against such a file resolve to "no such peak" instead of a crash.

Equivalence is pinned, never speed -- a timing assertion on this box would be
flaky, an equivalence assertion is not. Uses ``stage5_multi_file`` (the
12-window build) from ``conftest.py``; the shared 3-window fixture typically
keeps only one live window, which would make a per-window comparison
vacuous. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

from ftmwpipeline._internal import stage5_impl as s5mod
from ftmwpipeline._internal import stage6_impl as s6mod
from ftmwpipeline._internal.stage6_impl import (
    _persisted_acquisition_us,
    apply_curation_impl,
    refit_snap_tol_mhz_impl,
)
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    read_fit_parameters,
    read_fit_peak_frequencies_by_window,
    read_fit_peak_uids_by_window,
    read_fit_window_coverage,
)

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# The load counter
# ---------------------------------------------------------------------------

# Every module that imported the full loader into its own namespace. Patching
# the definition in ``io.fitting_serialization`` would miss all of them, since
# each holds its own reference from import time.
_LOADER_HOLDERS = (s6mod, s5mod)


class _LoadCounter:
    """Records every full ``SpectrumFit`` load, by calling frame."""

    def __init__(self) -> None:
        self.callers: List[str] = []

    @property
    def total(self) -> int:
        return len(self.callers)

    def summary(self) -> str:
        if not self.callers:
            return "no full loads"
        return ", ".join(sorted(set(self.callers)))


@pytest.fixture
def load_counter(monkeypatch) -> _LoadCounter:
    counter = _LoadCounter()

    def counted(h5_group):
        # frame 1 is the caller; the wrapper itself is frame 0.
        frame = sys._getframe(1)
        counter.callers.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
        return load_spectrum_fit_from_hdf5(h5_group)

    for module in _LOADER_HOLDERS:
        monkeypatch.setattr(module, "load_spectrum_fit_from_hdf5", counted)
    return counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_group(path: Path):
    """The full loader's result on *path* (caller closes the file)."""
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _a_fitted_peak(path: Path) -> Tuple[int, float, int]:
    """``(window_id, frequency_mhz, peak_uid)`` of the first fitted peak."""
    sf = _fit_group(path)
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for peak in wf.fitted_peaks:
            if peak.peak_uid is not None:
                return int(wf.window_id), float(peak.frequency_mhz), int(peak.peak_uid)
    pytest.skip("No fitted peak carrying a peak_uid in the built subset")


def _write_curation(tmp_path: Path, text: str) -> str:
    p = tmp_path / "budget_curation.csv"
    p.write_text(text)
    return str(p)


def _strip_peak_uid_column(path: Path) -> None:
    """Make *path* look like a fit written before ``peak_uid`` existed."""
    with h5py.File(str(path), "r+") as h5f:
        windows = h5f["stage5_fitting"]["windows"]
        for name in windows:
            peaks = windows[name]["peaks"]
            if "peak_uid" in peaks:
                del peaks["peak_uid"]


# ---------------------------------------------------------------------------
# T1 -- the load budget
# ---------------------------------------------------------------------------


def test_snap_tolerance_performs_no_full_load(stage5_multi_file, load_counter):
    """``review snap-tolerance`` reads one attribute, not the peak table."""
    value = refit_snap_tol_mhz_impl(str(stage5_multi_file))

    assert value > 0.0
    assert load_counter.total == 0, (
        f"review snap-tolerance performed {load_counter.total} full "
        f"SpectrumFit load(s) ({load_counter.summary()}); it needs only the "
        f"acquisition_us attribute"
    )


def test_dry_run_without_a_create_performs_no_full_load(
    stage5_multi_file, tmp_path, load_counter
):
    """An ordinary dry run resolves entirely from per-column reads.

    A plan carrying no ``create`` never opens the fit engine, so every value
    it needs -- window coverage, each window's fitted frequencies, the snap
    tolerance -- comes from a column read.
    """
    wid, freq, _uid = _a_fitted_peak(stage5_multi_file)
    curation = _write_curation(tmp_path, f"remove,{wid},{freq},\n")

    result = apply_curation_impl(stage5_multi_file, curation, dry_run=True)

    assert result.dry_run is True and result.applied == 0
    assert load_counter.total == 0, (
        f"a no-create dry run performed {load_counter.total} full "
        f"SpectrumFit load(s) ({load_counter.summary()})"
    )


def test_dry_run_with_a_uid_target_performs_no_full_load(
    stage5_multi_file, tmp_path, load_counter
):
    """The ``uid:N`` advisory path is a column read too.

    It is the one path that reads a second per-window column (``peak_uid``),
    so it is the one most likely to reach for the full loader by accident.
    """
    wid, _freq, uid = _a_fitted_peak(stage5_multi_file)
    curation = _write_curation(tmp_path, f"remove,{wid},uid:{uid},\n")

    result = apply_curation_impl(stage5_multi_file, curation, dry_run=True)

    assert result.dry_run is True
    assert load_counter.total == 0, (
        f"a uid-targeted dry run performed {load_counter.total} full "
        f"SpectrumFit load(s) ({load_counter.summary()})"
    )


# ---------------------------------------------------------------------------
# T2 -- reader equivalence
# ---------------------------------------------------------------------------


def test_cheap_readers_match_the_full_load(stage5_multi_file):
    """Every cheap reader returns exactly what the full load returned."""
    sf = _fit_group(stage5_multi_file)

    expected_freqs: Dict[int, List[float]] = {
        int(wf.window_id): [float(p.frequency_mhz) for p in wf.fitted_peaks]
        for wf in sf.window_fits
        if wf.window_id is not None
    }
    expected_uids = {
        int(wf.window_id): {
            int(p.peak_uid) for p in wf.fitted_peaks if p.peak_uid is not None
        }
        for wf in sf.window_fits
        if wf.window_id is not None
    }
    expected_coverage = [
        (
            int(wf.window_id),
            (
                None
                if wf.window is None
                else tuple(float(v) for v in wf.window.freq_range)
            ),
            expected_uids[int(wf.window_id)],
        )
        for wf in sf.window_fits
        if wf.window_id is not None
    ]
    expected_coverage.sort(key=lambda row: row[0])

    with h5py.File(str(stage5_multi_file), "r") as h5f:
        group = h5f["stage5_fitting"]
        assert read_fit_parameters(group) == sf.parameters
        # Order matters: a caller breaking a tie with min() depends on the
        # per-window row order matching the full loader's.
        assert read_fit_peak_frequencies_by_window(group) == expected_freqs
        assert read_fit_peak_uids_by_window(group) == expected_uids
        coverage = [
            (row.window_id, row.freq_range, row.peak_uids)
            for row in read_fit_window_coverage(group)
        ]

    assert coverage == expected_coverage
    assert _persisted_acquisition_us(str(stage5_multi_file)) == float(
        sf.parameters.get("acquisition_us", 0.0)
    )


# ---------------------------------------------------------------------------
# T3 -- a fit predating peak_uid
# ---------------------------------------------------------------------------


def test_uid_readers_tolerate_a_file_without_the_column(stage5_multi_file):
    """No ``peak_uid`` dataset means an empty set per window, not a raise."""
    _strip_peak_uid_column(stage5_multi_file)

    with h5py.File(str(stage5_multi_file), "r") as h5f:
        group = h5f["stage5_fitting"]
        by_uid = read_fit_peak_uids_by_window(group)
        coverage = read_fit_window_coverage(group)

    assert by_uid, "the fixture should still carry windows"
    assert all(uids == set() for uids in by_uid.values())
    assert all(row.peak_uids == set() for row in coverage)
    # The frequencies are untouched -- only the identity column went away.
    assert any(row.freq_range is not None for row in coverage)


def test_uid_target_against_a_file_without_the_column_is_unmatched(
    stage5_multi_file, tmp_path
):
    """A ``uid:N`` remove on such a file reports the uid unmatched.

    Both rungs are pinned, because the missing column must not change
    either one: the dry run *advises* that the edit will fail (it advises,
    it never refuses), and the live apply *raises* naming the uid. What must
    never happen is the third outcome -- resolving to some other peak
    because the window's uid set came back empty.
    """
    wid, _freq, uid = _a_fitted_peak(stage5_multi_file)
    _strip_peak_uid_column(stage5_multi_file)
    curation = _write_curation(tmp_path, f"remove,{wid},uid:{uid},\n")

    preview = apply_curation_impl(stage5_multi_file, curation, dry_run=True)

    assert preview.applied == 0
    # The advisory names the target it could not resolve. Its wording on this
    # file is "window N has no fitted peaks" -- the window does have fitted
    # peaks, it has no *uids*, an inaccuracy inherited from the empty-set
    # representation and left alone here; what this test pins is that the uid
    # is named and the edit is flagged, not the phrasing.
    assert any(f"uid:{uid}" in w for w in preview.warnings), preview.warnings

    with pytest.raises(ValueError) as excinfo:
        apply_curation_impl(stage5_multi_file, curation)

    assert str(uid) in str(excinfo.value)
