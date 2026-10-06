"""A window with no peaks has no convergence outcome.

The null model reports ``success=False`` although no solver ran, so a window
left with zero peaks (an explicitly created one, or one whose last peak a
curation removes) reports ``converged`` as ``Absent.UNDEFINED`` -- not
``False`` -- and neither the CLI nor the report counts it as non-converged.
Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import h5py
import pytest

from ftmwpipeline._internal.report_impl import _count_nonconverged
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    review_preview_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_preview
from ftmwpipeline.contract import Absent
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.serialize import to_jsonable

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]

_WARNING = "did not converge"


def _window_peaks(path: Path) -> Tuple[int, List[float]]:
    """The first fitted window and its peak frequencies (raw frame)."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    for wf in sf.window_fits:
        if wf.window_id is not None and wf.fitted_peaks:
            return int(wf.window_id), [float(p.frequency_mhz) for p in wf.fitted_peaks]
    pytest.skip("fixture has no fitted window with peaks")


def _curation(tmp_path: Path, wid: int, freqs: List[float]) -> str:
    cur = tmp_path / "cur.csv"
    cur.write_text("".join(f"remove,{wid},{f},\n" for f in freqs))
    return str(cur)


def test_removing_every_peak_reports_converged_undefined(stage5_multi_file, tmp_path):
    wid, freqs = _window_peaks(stage5_multi_file)
    cur = _curation(tmp_path, wid, freqs)
    w = review_preview_impl(stage5_multi_file, cur, frame="raw").windows[wid]
    assert w.n_peaks_after == 0
    assert w.converged is Absent.UNDEFINED


def test_wire_form_of_a_real_result_is_null_plus_absent_marker(
    stage5_multi_file, tmp_path
):
    wid, freqs = _window_peaks(stage5_multi_file)
    cur = _curation(tmp_path, wid, freqs)
    w = review_preview_impl(stage5_multi_file, cur, frame="raw").windows[wid]
    wire = json.loads(json.dumps(to_jsonable(w)))
    assert wire["converged"] is None
    assert wire["converged_absent"] == "undefined"


def test_apply_removing_every_peak_reports_undefined_and_no_warning(
    stage5_multi_file, tmp_path, capsys
):
    wid, freqs = _window_peaks(stage5_multi_file)
    cur = _curation(tmp_path, wid, freqs)
    rc = cmd_review_apply(
        argparse.Namespace(
            file_path=str(stage5_multi_file),
            curation_file=cur,
            verbose=False,
            frame="raw",
        )
    )
    assert rc == 0
    assert _WARNING not in capsys.readouterr().out


def test_python_apply_reports_converged_undefined(stage5_multi_file, tmp_path):
    wid, freqs = _window_peaks(stage5_multi_file)
    cur = _curation(tmp_path, wid, freqs)
    result = apply_curation_impl(stage5_multi_file, cur, frame="raw")
    assert result.windows[wid].n_peaks_after == 0
    assert result.windows[wid].converged is Absent.UNDEFINED


def test_cli_preview_prints_no_nonconvergence_warning(
    stage5_multi_file, tmp_path, capsys
):
    wid, freqs = _window_peaks(stage5_multi_file)
    cur = _curation(tmp_path, wid, freqs)
    rc = cmd_review_preview(
        argparse.Namespace(
            file_path=str(stage5_multi_file),
            curation_file=cur,
            verbose=False,
            frame="raw",
        )
    )
    assert rc == 0
    assert _WARNING not in capsys.readouterr().out


def test_window_that_keeps_a_peak_still_reports_a_bool(stage5_multi_file, tmp_path):
    wid, freqs = _window_peaks(stage5_multi_file)
    if len(freqs) < 2:
        pytest.skip("window has a single peak")
    cur = _curation(tmp_path, wid, freqs[:1])
    w = review_preview_impl(stage5_multi_file, cur, frame="raw").windows[wid]
    assert isinstance(w.converged, bool)


def test_report_does_not_count_a_no_peak_window_as_nonconverged():
    wfs = [
        SimpleNamespace(fitted_peaks=[], success=False),
        SimpleNamespace(fitted_peaks=[object()], success=False),
        SimpleNamespace(fitted_peaks=[object()], success=True),
    ]
    assert _count_nonconverged(wfs) == 1


def test_explicitly_created_empty_window_reports_converged_undefined(
    stage5_multi_file, tmp_path
):
    from ftmwpipeline._internal.stage4_impl import load_windows_impl

    plan = load_windows_impl(str(stage5_multi_file))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    anchor = next(
        (0.5 * (h1 + l2) for (_, h1), (l2, _) in zip(spans, spans[1:]) if l2 - h1 > 4),
        None,
    )
    if anchor is None:
        pytest.skip("no gap wide enough to create into")
    cur = tmp_path / "cur.csv"
    cur.write_text(f"create,new,{anchor},\n")
    result = review_preview_impl(stage5_multi_file, str(cur))
    assert result.windows
    for w in result.windows.values():
        assert w.n_peaks_after == 0
        assert w.converged is Absent.UNDEFINED
