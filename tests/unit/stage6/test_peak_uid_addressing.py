"""``review edit --remove`` (and a curation file's ``remove`` row) addressing
a peak by its ``peak_uid`` instead of by frequency.

Grammar (:func:`ftmwpipeline.core.curation.parse_peak_token`): a ``"uid:N"``
token names the fitted peak in the target window whose
:attr:`~ftmwpipeline.core.data_structures.FittedPeak.peak_uid` equals ``N``;
anything else is a molecular frequency, exactly as every curation verb has
always accepted. ``remove`` accepts either, mixed; ``add`` and
``accept --candidate`` stay frequency-only.

Resolution is a substitution, not a second matching path: a ``uid:N`` token
is turned into the named peak's own ``frequency_mhz`` before the existing
nearest-fitted-peak snap ever runs, and that peak is at distance 0 from
itself, so it always wins the match -- including when another fitted peak
sits within the ordinary snap tolerance of it (see
``test_remove_by_uid_wins_over_close_neighbor`` below, built via an inferred
split so the window has two genuinely close peaks to distinguish).

Uses the ``stage5_multi_file`` fixture (several live fitted windows) from
``conftest.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    parse_curation_file,
    refit_snap_tol_mhz_impl,
    refit_window_impl,
)
from ftmwpipeline._internal.stage6_impl import _resolve_curation_plan
from ftmwpipeline.cli.review_commands import cmd_review_edit
from ftmwpipeline.core.curation import PeakUidToken, parse_peak_token
from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# parse_peak_token: pure grammar, no fixture
# ---------------------------------------------------------------------------


def test_parse_peak_token_frequency_float_and_str():
    assert parse_peak_token(27549.3259) == 27549.3259
    assert parse_peak_token("27549.3259") == 27549.3259


def test_parse_peak_token_uid():
    assert parse_peak_token("uid:15425022") == PeakUidToken(15425022)
    assert parse_peak_token("uid:0") == PeakUidToken(0)


def test_parse_peak_token_malformed_frequency_names_token():
    with pytest.raises(ValueError, match=re.escape("abc")):
        parse_peak_token("abc")


@pytest.mark.parametrize("token", ["uid:", "uid:abc", "uid:-3", "uid:1.5"])
def test_parse_peak_token_malformed_uid_names_token(token):
    """Every malformed uid spelling raises ValueError naming the offending
    token, rather than silently falling through to a frequency parse."""
    with pytest.raises(ValueError, match=re.escape(token)):
        parse_peak_token(token)


# ---------------------------------------------------------------------------
# parse_curation_file: the "uid:" grammar in a remove row's freqs column
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "curation.csv"
    p.write_text(text)
    return str(p)


def test_parse_remove_row_accepts_uid_token(tmp_path):
    """A remove row's one token may be a plain frequency or a 'uid:N'
    identifier."""
    ops = parse_curation_file(_write(tmp_path, "remove,12,uid:15425022,\n"))
    assert len(ops) == 1
    op = ops[0]
    assert op.action == "remove" and op.window_id == 12
    assert op.freqs == [PeakUidToken(15425022)]


def test_parse_remove_requires_exactly_one_token(tmp_path):
    """remove shares add's arity: exactly one token per row, whether a
    frequency or a uid. Naming several peaks is a matter of writing several
    rows -- a run of add/remove rows on one window coalesces into a single
    refit (_resolve_curation_plan) -- not a longer freqs column."""
    with pytest.raises(ValueError, match="remove needs exactly one frequency"):
        parse_curation_file(_write(tmp_path, "remove,12,,\n"))
    with pytest.raises(ValueError, match="remove needs exactly one frequency"):
        parse_curation_file(_write(tmp_path, "remove,12,uid:15425022;27549.3259,\n"))


def test_parse_add_row_rejects_uid_token(tmp_path):
    """'add' is unaffected by this grammar -- a 'uid:' token there still hits
    the ordinary non-numeric-frequency parse error."""
    with pytest.raises(ValueError, match="non-numeric frequency"):
        parse_curation_file(_write(tmp_path, "add,5,uid:3,\n"))


# ---------------------------------------------------------------------------
# End-to-end resolution: refit_window_impl / api / Pipeline / CLI
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _fitted_wf(path: Path, wid: int) -> FittingResult:
    sf = _load_spectrum_fit(path)
    return next(wf for wf in sf.window_fits if wf.window_id == wid)


def _first_peak(path: Path) -> Tuple[int, FittedPeak]:
    """(window_id, FittedPeak) of the first fitted peak with a stamped uid."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), p
    all_uids = [p.peak_uid for wf in sf.window_fits for p in wf.fitted_peaks]
    raise AssertionError(
        "expected stage5_multi_file to carry at least one fitted peak with "
        f"a stamped peak_uid; found {len(all_uids)} fitted peak(s) across "
        f"{len(sf.window_fits)} window(s), all with peak_uid=None"
    )


def _window_with_n_peaks(path: Path, n: int) -> FittingResult:
    sf = _load_spectrum_fit(path)
    wf = next(
        (
            w
            for w in sf.window_fits
            if w.window_id is not None and len(w.fitted_peaks) >= n
        ),
        None,
    )
    counts = sorted(
        len(w.fitted_peaks) for w in sf.window_fits if w.window_id is not None
    )
    assert wf is not None, (
        f"expected stage5_multi_file to carry a window with at least {n} "
        f"fitted peaks; found {len(counts)} window(s) with fitted-peak "
        f"counts {counts}"
    )
    return wf


def _fitted_by_window(path: Path) -> Dict[int, List[float]]:
    sf = _load_spectrum_fit(path)
    return {
        int(wf.window_id): sorted(
            round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks
        )
        for wf in sf.window_fits
        if wf.window_id is not None
    }


@pytest.mark.integration
def test_remove_by_uid_removes_named_peak(stage5_multi_file):
    path = stage5_multi_file
    wid, target = _first_peak(path)
    n_before = len(_fitted_wf(path, wid).fitted_peaks)

    result = refit_window_impl(str(path), wid, remove=[f"uid:{target.peak_uid}"])

    assert result.n_peaks_after == n_before - 1
    assert target.peak_uid not in {p.peak_uid for p in result.fitted_peaks}


@pytest.mark.integration
def test_remove_by_uid_wins_over_close_neighbor(stage5_multi_file):
    """The test that proves uid addressing is not just frequency addressing
    wearing a hat: build a window with two peaks within the ordinary snap
    tolerance of each other (via an inferred split -- see
    test_curation_intent.py for the same construction), remove one by uid,
    and confirm the OTHER survives untouched."""
    path = stage5_multi_file
    wid, parent = _first_peak(path)
    parent_freq = float(parent.frequency_mhz)
    snap_tol = refit_snap_tol_mhz_impl(path)

    # An add within snap tolerance of a fitted peak is read as a split of
    # that peak -- see _infer_curation_intent -- so this seeds exactly two
    # close fitted peaks, both within snap_tol of each other.
    refit_window_impl(str(path), wid, add=[parent_freq + 0.3 * snap_tol])

    wf = _fitted_wf(path, wid)
    peaks_sorted = sorted(wf.fitted_peaks, key=lambda p: float(p.frequency_mhz))
    close_pair = None
    for a, b in zip(peaks_sorted, peaks_sorted[1:]):
        if (
            a.peak_uid is not None
            and b.peak_uid is not None
            and abs(float(b.frequency_mhz) - float(a.frequency_mhz)) <= snap_tol
        ):
            close_pair = (a, b)
            break
    assert close_pair is not None, (
        "expected the inferred split (add at parent_freq + 0.3 * snap_tol) to "
        f"leave two fitted peaks within {snap_tol:.6f} MHz of each other in "
        f"window {wid}; got "
        f"{[float(p.frequency_mhz) for p in peaks_sorted]} (snap_tol="
        f"{snap_tol:.6f} MHz)"
    )
    target, other = close_pair
    n_before = len(wf.fitted_peaks)

    result = refit_window_impl(str(path), wid, remove=[f"uid:{target.peak_uid}"])

    assert result.n_peaks_after == n_before - 1
    remaining_uids = {p.peak_uid for p in result.fitted_peaks}
    assert target.peak_uid not in remaining_uids
    assert other.peak_uid in remaining_uids


@pytest.mark.integration
def test_remove_mixed_uid_and_frequency_removes_both(stage5_multi_file):
    path = stage5_multi_file
    wf = _window_with_n_peaks(path, 2)
    wid = int(wf.window_id)
    peaks_sorted = sorted(wf.fitted_peaks, key=lambda p: float(p.frequency_mhz))
    by_uid, by_freq = peaks_sorted[0], peaks_sorted[1]
    assert by_uid.peak_uid is not None
    n_before = len(wf.fitted_peaks)

    result = refit_window_impl(
        str(path),
        wid,
        remove=[f"uid:{by_uid.peak_uid}", float(by_freq.frequency_mhz)],
    )

    assert result.n_peaks_after == n_before - 2
    remaining_uids = {p.peak_uid for p in result.fitted_peaks}
    assert by_uid.peak_uid not in remaining_uids


@pytest.mark.integration
def test_remove_by_uid_unknown_raises_naming_uid_and_window(stage5_multi_file):
    path = stage5_multi_file
    wid, _peak = _first_peak(path)

    with pytest.raises(ValueError) as excinfo:
        refit_window_impl(str(path), wid, remove=["uid:999999999"])

    msg = str(excinfo.value)
    assert "999999999" in msg
    assert str(wid) in msg


@pytest.mark.integration
def test_add_rejects_uid_token(stage5_multi_file):
    path = stage5_multi_file
    wid, _peak = _first_peak(path)

    with pytest.raises(ValueError, match="add takes a frequency"):
        refit_window_impl(str(path), wid, add=["uid:5"])


@pytest.mark.integration
def test_curation_file_remove_uid_round_trips(stage5_multi_file, tmp_path):
    """Two remove rows on the same window -- one 'uid:N', one a plain
    frequency -- coalesce into a single edit action that removes both (a run
    of add/remove rows on one window is always one row per target; see
    _resolve_curation_plan)."""
    path = stage5_multi_file
    wf = _window_with_n_peaks(path, 2)
    wid = int(wf.window_id)
    peaks_sorted = sorted(wf.fitted_peaks, key=lambda p: float(p.frequency_mhz))
    by_uid, by_freq = peaks_sorted[0], peaks_sorted[1]
    assert by_uid.peak_uid is not None
    n_before = len(wf.fitted_peaks)

    cur = tmp_path / "uid_remove.csv"
    cur.write_text(
        f"remove,{wid},uid:{by_uid.peak_uid},\n"
        f"remove,{wid},{float(by_freq.frequency_mhz)},\n"
    )

    ops = parse_curation_file(str(cur))
    plan = _resolve_curation_plan(ops)
    assert len(plan) == 1
    assert plan[0].kind == "edit" and plan[0].window_id == wid
    assert plan[0].remove == [
        PeakUidToken(by_uid.peak_uid),
        float(by_freq.frequency_mhz),
    ]

    apply_curation_impl(path, cur)

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before - 2
    assert by_uid.peak_uid not in {p.peak_uid for p in wf_after.fitted_peaks}


@pytest.mark.integration
def test_curation_file_dry_run_flags_unknown_uid(stage5_multi_file, tmp_path):
    """--dry-run must catch a 'uid:N' that names no fitted peak in the
    window -- the exact class of resolution failure the preview exists to
    surface -- and the file must stay untouched."""
    path = stage5_multi_file
    wid, _peak = _first_peak(path)

    cur = tmp_path / "bad_uid.csv"
    cur.write_text(f"remove,{wid},uid:999999999,\n")

    before = _fitted_by_window(path)
    dry = apply_curation_impl(path, cur, dry_run=True)
    assert any(
        "999999999" in w and str(wid) in w and "the edit will fail" in w
        for w in dry.warnings
    ), dry.warnings
    assert _fitted_by_window(path) == before

    with pytest.raises(ValueError, match="999999999"):
        apply_curation_impl(path, cur)
    assert _fitted_by_window(path) == before


# ---------------------------------------------------------------------------
# Cross-interface consistency: api / Pipeline / CLI must persist identically
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_review_edit_remove_uid_cross_interface(stage5_multi_file, tmp_path):
    paths = {k: tmp_path / f"uid_{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_multi_file, p)

    wid, target = _first_peak(paths["api"])
    token = f"uid:{target.peak_uid}"

    ftmw.review_edit(paths["api"], wid, remove=[token])
    Pipeline.open(paths["pipe"]).review_edit(wid, remove=[token])
    rc = cmd_review_edit(
        argparse.Namespace(
            file_path=str(paths["cli"]), window=wid, add=None, remove=[token]
        )
    )
    assert rc == 0

    ref = _fitted_by_window(paths["api"])
    assert _fitted_by_window(paths["pipe"]) == ref
    assert _fitted_by_window(paths["cli"]) == ref
