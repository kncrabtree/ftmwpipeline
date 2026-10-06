"""Contract 18: a created window's anchor and extent in the calibrated frame.

``PlannedWindowResult`` (a curation's ``created_windows``),
``CreateWindowResult`` and the ``created_window_*`` fields of
``PreviewWindowResult`` / ``RefitWindowResult`` report a created window's
anchor and extent raw-frame, beside calibrated frequencies a UI shows and may
not convert. Each carries a calibrated companion converted exactly as
``window_status(frame="calibrated")`` converts a window's bounds, so the
companion of an installed window equals that window's calibrated
``window_status`` bounds. With no calibration to apply (``epsilon == 0``) the
frames coincide and the companion equals the raw value -- never an absence;
it is ``Absent.NOT_RUN`` only where its raw sibling is.

Only a ``self_calibrated`` file tells the frames apart, so the fixture is
stamped like ``test_window_status_frame.py``'s (helper duplicated per this
suite's precedent). Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline, to_jsonable
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("every_write_is_reference"),
]

EPS = 2.2e-6


def _make_self_calibrated(path: Path, *, epsilon: float = EPS) -> None:
    """Declare an unlocked digitizer and stamp a passing timebase result."""
    persisted = load_stage_fit_settings_from_h5(str(path))
    assert persisted is not None
    settings = replace(
        persisted,
        spur=replace(
            persisted.spur,
            clocks=(
                ClockSource(5120.0, locked=True),
                ClockSource(6250.0, locked=False),
            ),
        ),
    )
    with atomic_write(str(path)):
        save_stage_fit_settings_to_h5(str(path), settings)
    result = TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=0.1e-6,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=320.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        save_timebase_calibration_to_hdf5(result, h5f.create_group(GROUP_PATH))


def _gap_anchor(path: Path) -> float:
    """A raw frequency in the band but outside every planned window."""
    plan = load_windows_impl(str(path))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        if lo2 - hi1 > 4.0:
            return 0.5 * (hi1 + lo2)
    pytest.skip("no gap between planned windows wide enough to create into")


@pytest.fixture
def sc_file(stage5_multi_file: Path) -> Path:
    _make_self_calibrated(stage5_multi_file)
    assert ftmw.frequency_calibration(str(stage5_multi_file)).epsilon == EPS
    return stage5_multi_file


def _copy(src: Path, tmp_path: Path, name: str) -> str:
    dst = tmp_path / name
    shutil.copy(src, dst)
    return str(dst)


def _status_bounds(path: str, frame: str) -> Dict[int, Any]:
    payload = ftmw.window_status(path, frame=frame)
    return {w.window_id: (w.freq_min_mhz, w.freq_max_mhz) for w in payload["windows"]}


def _cli_json(argv: List[str], capsys: Any) -> Dict[str, Any]:
    capsys.readouterr()
    rc = main(argv + ["--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    payload: Dict[str, Any] = json.loads(out)
    return payload


def test_planned_structure_matches_window_status_and_agrees_dry_and_live(
    sc_file: Path, tmp_path: Path
) -> None:
    anchor = _gap_anchor(sc_file)
    cur = tmp_path / "implied.csv"
    cur.write_text(f"add,,{anchor},\n")

    dry = ftmw.review_apply(str(sc_file), str(cur), dry_run=True, frame="raw")
    preview = ftmw.review_preview(str(sc_file), str(cur), frame="raw")
    live_path = _copy(sc_file, tmp_path, "live.ftmw")
    live = ftmw.review_apply(live_path, str(cur), frame="raw")

    assert len(live.created_windows) == 1
    pw = live.created_windows[0]
    # Dry run, preview and live apply agree field for field.
    assert dry.created_windows == live.created_windows
    assert preview.created_windows == live.created_windows
    w = preview.windows[pw.window_id]
    assert w.created_window_freq_range == pw.freq_range
    assert w.created_window_freq_range_calibrated == pw.freq_range_calibrated

    # The calibrated extent is the installed window's calibrated
    # window_status bounds; the raw one its raw bounds.
    assert _status_bounds(live_path, "calibrated")[pw.window_id] == (
        pw.freq_range_calibrated
    )
    assert _status_bounds(live_path, "raw")[pw.window_id] == pw.freq_range
    assert pw.freq_range_calibrated != pw.freq_range
    assert pw.anchor_mhz == anchor
    assert pw.anchor_calibrated_mhz != pw.anchor_mhz
    lo, hi = pw.freq_range_calibrated
    assert lo <= pw.anchor_calibrated_mhz <= hi

    # The decision-log evidence is unchanged: a raw snapshot only.
    created = ftmw.review_log(live_path)[0].evidence["created_window"]
    assert (created["freq_min_mhz"], created["freq_max_mhz"]) == pw.freq_range
    assert not any("calibrated" in k for k in created)


def test_create_and_edit_results_carry_the_same_companions(
    sc_file: Path, tmp_path: Path
) -> None:
    anchor = _gap_anchor(sc_file)
    cur = tmp_path / "create.csv"
    cur.write_text(f"create,new,{anchor},\n")
    planned = ftmw.review_apply(
        str(sc_file), str(cur), dry_run=True, frame="raw"
    ).created_windows[0]

    created = Pipeline.open(_copy(sc_file, tmp_path, "c.ftmw")).review_create(
        anchor, frame="raw"
    )
    assert created.anchor_calibrated_mhz == planned.anchor_calibrated_mhz
    assert created.freq_range_calibrated == planned.freq_range_calibrated
    assert created.freq_range == planned.freq_range

    edited = ftmw.review_edit(
        _copy(sc_file, tmp_path, "e.ftmw"), None, add=[anchor], frame="raw"
    )
    assert edited.created_window_freq_range == planned.freq_range
    assert edited.created_window_freq_range_calibrated == (
        planned.freq_range_calibrated
    )


def test_cli_json_carries_the_companions(
    sc_file: Path, tmp_path: Path, capsys: Any
) -> None:
    anchor = _gap_anchor(sc_file)
    cur = tmp_path / "implied.csv"
    cur.write_text(f"add,,{anchor},\n")
    expected = to_jsonable(
        ftmw.review_apply(
            str(sc_file), str(cur), dry_run=True, frame="raw"
        ).created_windows
    )
    assert expected[0]["freq_range_calibrated"] != expected[0]["freq_range"]

    base = [str(sc_file), str(cur), "--frame", "raw"]
    preview = _cli_json(["review", "preview", *base], capsys)
    dry = _cli_json(["review", "apply", *base, "--dry-run"], capsys)
    live_path = _copy(sc_file, tmp_path, "live.ftmw")
    live = _cli_json(["review", "apply", live_path, str(cur), "--frame", "raw"], capsys)
    for payload in (preview, dry, live):
        assert payload["created_windows"] == expected
    (window,) = preview["windows"]
    assert window["created_window_freq_range_calibrated"] == (
        expected[0]["freq_range_calibrated"]
    )

    create = _cli_json(
        [
            "review",
            "create",
            _copy(sc_file, tmp_path, "c.ftmw"),
            "--at",
            str(anchor),
            "--frame",
            "raw",
        ],
        capsys,
    )
    summary = create["summary"]
    assert summary["anchor_calibrated_mhz"] == expected[0]["anchor_calibrated_mhz"]
    assert [
        summary["freq_lo_calibrated_mhz"],
        summary["freq_hi_calibrated_mhz"],
    ] == expected[0]["freq_range_calibrated"]


def test_companion_is_absent_only_where_its_raw_sibling_is(
    stage5_multi_file: Path, tmp_path: Path
) -> None:
    path = str(stage5_multi_file)
    assert ftmw.frequency_calibration(path).epsilon == 0.0
    anchor = _gap_anchor(stage5_multi_file)
    cur = tmp_path / "implied.csv"
    cur.write_text(f"add,,{anchor},\n")
    (pw,) = ftmw.review_preview(path, str(cur)).created_windows
    # No calibration to apply: the frames coincide -- a value, not an absence.
    assert pw.freq_range_calibrated == pw.freq_range
    assert pw.anchor_calibrated_mhz == pw.anchor_mhz

    # A window the batch only edits carries no created structure at all.
    wid, uid = _first_peak(stage5_multi_file)
    edit = tmp_path / "ordinary.csv"
    edit.write_text(f"remove,{wid},uid:{uid},\n")
    w = ftmw.review_preview(path, str(edit)).windows[wid]
    assert w.created_window_freq_range is Absent.NOT_RUN
    assert w.created_window_freq_range_calibrated is Absent.NOT_RUN
    wire = to_jsonable(w)
    assert wire["created_window_freq_range_calibrated"] is None
    assert wire["created_window_freq_range_calibrated_absent"] == "not_run"


def _first_peak(path: Path) -> Tuple[int, int]:
    """``(window_id, peak_uid)`` of the first fitted line with a uid."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), int(p.peak_uid)
    pytest.skip("no fitted line with a stamped peak_uid")
