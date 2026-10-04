"""A curation batch naming ids the file lacks refuses with every one of them.

``not_found`` lists all unknown window ids (or peak uids) at once, so a
program fixes the whole request in one round trip. It is still a ``KeyError``
and a ``PipelineFileError`` carrying ``kind`` and ``ids``.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline._internal.stage6_impl import apply_curation_impl, review_preview_impl
from ftmwpipeline.file_manager import NotFoundError, PipelineFileError
from ftmwpipeline.io.fitting_serialization import read_fit_window_coverage

pytestmark = [pytest.mark.unit]

_BAD_WINDOWS = (987654, 987655)
_BAD_UIDS = (900001, 900002)


def _good_window(path) -> int:
    with h5py.File(path, "r") as h5f:
        return next(
            c.window_id for c in read_fit_window_coverage(h5f["stage5_fitting"])
        )


def test_every_unknown_window_is_listed_at_once(stage5_small_file, tmp_path):
    wid = _good_window(stage5_small_file)
    cur = tmp_path / "c.csv"
    cur.write_text(
        f"accept,{wid},,\naccept,{_BAD_WINDOWS[0]},,\naccept,{_BAD_WINDOWS[1]},,\n"
    )
    with pytest.raises(NotFoundError) as exc:
        apply_curation_impl(str(stage5_small_file), cur)
    assert exc.value.kind == "window"
    assert exc.value.ids == list(_BAD_WINDOWS)
    assert isinstance(exc.value, (KeyError, PipelineFileError))
    assert exc.value.to_dict()["code"] == "not_found"


def test_every_unknown_window_is_listed_through_the_batch_engine(
    stage5_small_file, tmp_path
):
    with h5py.File(stage5_small_file, "r") as h5f:
        window = next(
            c
            for c in read_fit_window_coverage(h5f["stage5_fitting"])
            if c.freq_range is not None
        )
    lo, hi = window.freq_range
    cur = tmp_path / "c.csv"
    cur.write_text(
        f"accept,{window.window_id},,candidate={0.5 * (lo + hi)!r}\n"
        f"accept,{_BAD_WINDOWS[0]},,\naccept,{_BAD_WINDOWS[1]},,\n"
    )
    with pytest.raises(NotFoundError) as exc:
        apply_curation_impl(str(stage5_small_file), cur)
    assert exc.value.ids == list(_BAD_WINDOWS)
    with pytest.raises(NotFoundError) as prev:
        review_preview_impl(str(stage5_small_file), cur)
    assert prev.value.ids == list(_BAD_WINDOWS)


def test_every_unknown_peak_uid_is_listed_at_once(stage5_small_file, tmp_path):
    cur = tmp_path / "c.csv"
    cur.write_text("".join(f"remove,,uid:{u},\n" for u in _BAD_UIDS))
    with pytest.raises(NotFoundError) as exc:
        apply_curation_impl(str(stage5_small_file), cur)
    assert exc.value.kind == "peak"
    assert exc.value.ids == list(_BAD_UIDS)
