"""``window_status(path, frame=...)``: bounds in the raw or calibrated frame.

Spec: ``docs/source/machine_contract.rst`` §Window status, "Frame of the
bounds". The calibrated bounds of a fitted window must equal the
``fit_window_mhz`` the final products report for each of its lines -- a
client compares the two directly and never converts a frequency itself. Only
a ``self_calibrated`` file (``epsilon != 0``) tells the frames apart, so the
fixture is stamped like ``test_frame_parameter.py``'s.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, BadSettingError
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage6_impl import review_run_impl
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)

EPS = 2.2e-6


def make_self_calibrated(path: Path, *, epsilon: float = EPS) -> None:
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


@pytest.fixture(scope="module")
def sc_reviewed(_stage5_small_built: Path, tmp_path_factory) -> str:
    """The small Stage 5 build, self_calibrated, then reviewed (read-only)."""
    fp = tmp_path_factory.mktemp("ws_frame") / "sc_reviewed.ftmw"
    shutil.copy(_stage5_small_built, fp)
    make_self_calibrated(fp)
    review_run_impl(str(fp))
    assert ftmw.frequency_calibration(str(fp)).state == "self_calibrated"
    return str(fp)


def _bounds(payload):
    return {w.window_id: (w.freq_min_mhz, w.freq_max_mhz) for w in payload["windows"]}


def test_calibrated_bounds_equal_each_lines_fit_window(sc_reviewed):
    calibrated = ftmw.window_status(sc_reviewed, frame="calibrated")
    assert calibrated["frame"] == "calibrated"
    bounds = _bounds(calibrated)
    fp = ftmw.get_final_products(sc_reviewed)
    assert fp is not None
    checked = 0
    for p in fp.peaks:
        if isinstance(p.window_id, Absent) or isinstance(p.fit_window_mhz, Absent):
            continue
        assert tuple(p.fit_window_mhz) == bounds[p.window_id]
        checked += 1
    assert checked > 0


def test_raw_is_the_default_and_differs_from_calibrated(sc_reviewed):
    default = ftmw.window_status(sc_reviewed)
    assert default == ftmw.window_status(sc_reviewed, frame="raw")
    assert default["frame"] == "raw"
    raw = _bounds(default)
    cal = _bounds(ftmw.window_status(sc_reviewed, frame="calibrated"))
    assert raw.keys() == cal.keys()
    assert all(raw[w] != cal[w] for w in raw)
    # Raw bounds are the stored ones: the read_table form.
    table = ftmw.read_table(sc_reviewed, "window_status")
    assert list(table["freq_min_mhz"]) == [lo for lo, _ in raw.values()]


def test_only_the_bounds_change_with_the_frame(sc_reviewed):
    raw = ftmw.window_status(sc_reviewed)["windows"]
    cal = ftmw.window_status(sc_reviewed, frame="calibrated")["windows"]
    assert [w.window_id for w in raw] == [w.window_id for w in cal]
    for r, c in zip(raw, cal):
        assert replace(c, freq_min_mhz=r.freq_min_mhz, freq_max_mhz=r.freq_max_mhz) == r


def test_frames_coincide_when_epsilon_is_zero(stage5_small_source):
    path = str(stage5_small_source)
    assert ftmw.frequency_calibration(path).epsilon == 0.0
    assert _bounds(ftmw.window_status(path, frame="calibrated")) == _bounds(
        ftmw.window_status(path)
    )


@pytest.mark.parametrize("bad", ["Calibrated", "", None, 1])
def test_an_unknown_frame_is_refused(bad, stage5_small_source):
    with pytest.raises(BadSettingError) as info:
        ftmw.window_status(str(stage5_small_source), frame=bad)
    assert info.value.to_dict()["path"] == "frame"
