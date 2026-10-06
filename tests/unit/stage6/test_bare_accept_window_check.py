"""A bare accept refuses a window the file does not have.

It used to record a "reviewed" status and a decision-log entry for any id at
all. The windows it may name are the Stage 5 fit's (created windows included);
with no complete fit it is refused with ``stage_not_run``, like every other
curation call.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    review_accept_impl,
)
from ftmwpipeline.file_manager import NotFoundError, StageDependencyError
from ftmwpipeline.io.fitting_serialization import read_fit_window_coverage
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file

pytestmark = [
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]

_UNKNOWN = 987654


def _window_ids(path) -> list:
    with h5py.File(path, "r") as h5f:
        return [c.window_id for c in read_fit_window_coverage(h5f["stage5_fitting"])]


def _review_state(path):
    review = load_stage6_review_from_file(str(path))
    return dict(review.window_statuses), list(review.decision_log)


def test_bare_accept_of_an_unknown_window_is_refused(stage5_small_file):
    before = _review_state(stage5_small_file)
    with pytest.raises(KeyError, match=str(_UNKNOWN)):
        review_accept_impl(str(stage5_small_file), _UNKNOWN)
    assert _review_state(stage5_small_file) == before


def test_bare_accept_of_a_real_window_still_records(stage5_small_file):
    wid = _window_ids(stage5_small_file)[0]
    review_accept_impl(str(stage5_small_file), wid)
    statuses, log = _review_state(stage5_small_file)
    assert statuses[wid].provenance == "reviewed"
    assert [(e.window_id, e.kind) for e in log] == [(wid, "accept")]


def test_curation_file_with_an_unknown_window_applies_nothing(
    stage5_small_file, tmp_path
):
    """A bare-accept-only file is all or nothing: the valid row above the bad
    one is not left applied."""
    wid = _window_ids(stage5_small_file)[0]
    cur = tmp_path / "accepts.csv"
    cur.write_text(f"accept,{wid},,\naccept,{_UNKNOWN},,\n")
    before = _review_state(stage5_small_file)
    with pytest.raises(NotFoundError, match=str(_UNKNOWN)):
        apply_curation_impl(str(stage5_small_file), cur)
    assert _review_state(stage5_small_file) == before


def test_unknown_window_in_a_fitting_batch_is_refused(stage5_small_file, tmp_path):
    """A bare accept riding in a batch with a fit-changing row goes through the
    batch engine, which checks the batch's own fit."""
    with h5py.File(stage5_small_file, "r") as h5f:
        window = next(
            c
            for c in read_fit_window_coverage(h5f["stage5_fitting"])
            if c.freq_range is not None
        )
    lo, hi = window.freq_range
    candidate = 0.5 * (lo + hi)
    cur = tmp_path / "mixed.csv"
    cur.write_text(
        f"accept,{window.window_id},,candidate={candidate!r}\n" f"accept,{_UNKNOWN},,\n"
    )
    before = _review_state(stage5_small_file)
    with pytest.raises(NotFoundError, match=str(_UNKNOWN)):
        apply_curation_impl(str(stage5_small_file), cur)
    assert _review_state(stage5_small_file) == before


def test_before_stage5_a_bare_accept_is_refused(stage5_small_file, tmp_path):
    """No complete fit: a bare accept (and a bare-accept-only curation file) is
    refused with ``stage_not_run`` and records nothing."""
    wid = _window_ids(stage5_small_file)[0]
    with h5py.File(stage5_small_file, "a") as h5f:
        del h5f["stage5_fitting"]
    before = _review_state(stage5_small_file)
    with pytest.raises(StageDependencyError) as info:
        review_accept_impl(str(stage5_small_file), wid)
    assert info.value.code == "stage_not_run"
    assert info.value.to_dict()["command"] == "fit run"
    cur = tmp_path / "accepts.csv"
    cur.write_text(f"accept,{wid},,\n")
    with pytest.raises(StageDependencyError):
        apply_curation_impl(str(stage5_small_file), cur)
    assert _review_state(stage5_small_file) == before
