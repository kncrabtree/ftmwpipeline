"""Stage 1's persisted record is authoritative.

Once Stage 1 has written ``processing_parameters/ft_processing`` at the current
field-set version, it holds the concrete values the FT was computed with, and
every reader resolves through the one Stage 1 resolver without falling through
to the recommended layer. A later stamp to that layer (``start run``, a
chirp-window declaration) changes nothing Stage 1 is read as having used. An
older record keeps the fall-through it was written under.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.active_ft_support import compute_persisted_active_ft
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage1_impl import (
    _resolve_settings,
    compute_ft_impl,
    ft_settings_provenance,
    persisted_ft_settings,
)
from ftmwpipeline._internal.start_detection_impl import (
    detect_start_time_impl,
    resolve_start_provenance,
    stamped_start_note,
)
from ftmwpipeline.core.data_structures import FID, ChirpWindow, Sideband
from ftmwpipeline.core.settings import (
    FT_PROCESSING_FIELD_SET_VERSION,
    FT_PROCESSING_PATH,
    RECOMMENDED_PATH,
)
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.file_manager import (
    SourceMetadata,
    create_pipeline_file,
    update_processing_parameters,
)
from ftmwpipeline.io.provenance import FIELD_SET_VERSION_ATTR
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_recommended_chirp_window,
)

_FAST = StartDetectionSettings(step_us=0.05, sweep_max_us=7.0)
_TRIM = (900.0, 950.0)


def _make_fid(duration_us: float = 20.0, dt_s: float = 4e-9) -> FID:
    n = int(round(duration_us * 1e-6 / dt_s))
    t = np.arange(n) * dt_s
    t_us = t * 1e6
    data = 5.0 * np.exp(-t_us / 10.0) * np.cos(2 * np.pi * 80e6 * t)
    return FID(data=data, spacing=dt_s, probe_freq_mhz=1000.0, sideband=Sideband.LOWER)


def _create_ftmw(tmp_path: Path, name: str = "s1.ftmw") -> str:
    p = str(tmp_path / name)
    src = SourceMetadata(source_path=tmp_path, format_name="test")
    with atomic_write(p):
        create_pipeline_file(p, _make_fid(), src, force=True)
    return p


def _record(p: str) -> dict:
    with h5py.File(p, "r") as h5f:
        return dict(h5f[FT_PROCESSING_PATH].attrs)


def _completed(p: str) -> list:
    with h5py.File(p, "r") as h5f:
        return sorted(json.loads(h5f["pipeline_stages"].attrs["completed_stages"]))


def _make_pre_provenance(p: str, **attrs: object) -> None:
    """Rewrite the Stage 1 record as an older writer left it: no version."""
    with h5py.File(p, "a") as h5f:
        group = h5f[FT_PROCESSING_PATH]
        del group.attrs[FIELD_SET_VERSION_ATTR]
        for key, value in attrs.items():
            group.attrs[key] = value


# Mutation caught: _persist_ft_settings writes unset (__None__) bounds, skips
# the version stamp, or stamps below the current field-set version.
def test_stage1_records_the_concrete_window_it_ran_with(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    duration = float(ftmw.load_fid(p).duration_us)
    ftmw.compute_ft(p)

    attrs = _record(p)
    assert attrs[FIELD_SET_VERSION_ATTR] == FT_PROCESSING_FIELD_SET_VERSION
    assert attrs["start_us"] == 0.0
    assert attrs["end_us"] == duration
    # No trim is recorded as unset, which a current record means literally.
    assert attrs["trim_min_mhz"] == "__None__"
    prov = ft_settings_provenance(p)
    assert prov is not None and prov.is_current


# Mutation caught: the concrete window is off by a sample (e.g. end = duration
# - dt), which would move a fresh run.
def test_concrete_window_selects_the_same_spectrum(tmp_path: Path) -> None:
    """0.0 / the duration select exactly what unset bounds selected."""
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    with atomic_write(p):
        concrete = compute_persisted_active_ft(p)
    _make_pre_provenance(p, start_us="__None__", end_us="__None__")
    with atomic_write(p):
        unset = compute_persisted_active_ft(p)
    np.testing.assert_array_equal(concrete.complex_spectrum, unset.complex_spectrum)
    np.testing.assert_array_equal(concrete.freq_mhz, unset.freq_mhz)


# Mutation caught: resolve_ft_settings_h5 reads the recommended layer on a
# current record, or a consumer (compute_ft_impl / persisted_ft_settings /
# compute_persisted_active_ft) goes back to its own reader.
def test_a_later_recommended_start_changes_nothing_stage1_used(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, trim=_TRIM)
    duration = float(ftmw.load_fid(p).duration_us)
    with atomic_write(p):
        before = compute_persisted_active_ft(p).complex_spectrum.copy()

    with atomic_write(p):
        update_processing_parameters(p, {"start_us": 3.0, "end_us": 12.0})

    assert _resolve_settings(p, None).start_us == 0.0
    assert compute_ft_impl(p)["resolved_settings"].active_window_us() == (
        0.0,
        duration,
    )
    assert persisted_ft_settings(p, "x", duration).active_window_us() == (
        0.0,
        duration,
    )
    with atomic_write(p):
        np.testing.assert_array_equal(
            compute_persisted_active_ft(p).complex_spectrum, before
        )
    # The recommendation is still stored, for display.
    with h5py.File(p, "r") as h5f:
        assert h5f[RECOMMENDED_PATH].attrs["start_us"] == 3.0


# Mutation caught: an unset trim in a current record falls through to the
# recommended trim.
def test_an_unset_trim_does_not_fall_through(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    with h5py.File(p, "a") as h5f:
        h5f[RECOMMENDED_PATH].attrs["trim_min_mhz"] = _TRIM[0]
        h5f[RECOMMENDED_PATH].attrs["trim_max_mhz"] = _TRIM[1]
    assert _resolve_settings(p, None).trim is None
    assert compute_ft_impl(p).get("trim_range") is None


# Mutation caught: a start stamp after Stage 1 alters the resolved start or
# invalidates Stage 1, or the 'ft run --start-us' warning is lost.
def test_start_run_after_stage1_warns_and_leaves_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    completed = _completed(p)
    with atomic_write(p):
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
    with caplog.at_level(logging.WARNING):
        out = detect_start_time_impl(p, settings=_FAST, stamp=True)
    assert out["stamped"] is True
    assert "does not change it" in caplog.text
    assert _resolve_settings(p, None).start_us == 0.0
    assert _completed(p) == completed


# Mutation caught: the 'start run' note claims a later FT run inherits the
# stamp after Stage 1 has persisted its authoritative record.
def test_stamped_start_note_is_true_before_and_after_stage1(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    before = stamped_start_note(p, 2.0)
    assert "will inherit it" in before

    ftmw.compute_ft(p)  # start_us 0.0, authoritative
    after = stamped_start_note(p, 2.0)
    assert "inherit" not in after
    assert "start_us = 0.000 us" in after
    assert f"ft run {p} --start-us 2.000" in after

    same = stamped_start_note(p, 0.0)
    assert "inherit" not in same and "already runs" in same


def test_start_run_cli_prints_the_note_for_the_file_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from ftmwpipeline.cli.main import main

    p = _create_ftmw(tmp_path)
    with atomic_write(p):
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
    assert main(["start", "run", p]) == 0
    assert "will inherit it" in capsys.readouterr().out

    ftmw.compute_ft(p, start_us=0.0)
    assert main(["start", "run", p]) == 0
    out = capsys.readouterr().out
    assert "will inherit it" not in out
    assert "--start-us" in out


# Mutation caught: an explicit re-run is swallowed, or leaves a stale downstream
# result when the effective window changes.
def test_explicit_rerun_adopts_a_new_start_and_invalidates(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    assert "stage2_noise_result" in _completed(p)
    ftmw.compute_ft(p, start_us=2.0)
    assert _record(p)["start_us"] == 2.0
    assert "stage2_noise_result" not in _completed(p)


# Mutation caught: a current record with no recommendation classifies as
# 'manual' because start_us is now concrete.
def test_nothing_recommended_reads_as_none_provenance(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    prov = resolve_start_provenance(p)
    assert prov.source == "none"
    assert prov.start_us == 0.0


# Mutation caught: the authoritative rule is applied to old records, or
# Stage 2b/timebase read 0.0 again instead of the shared resolver.
def test_pre_provenance_record_keeps_its_fall_through(tmp_path: Path) -> None:
    """An older record with an unset start resolves it from the recommended
    layer, and every reader -- Stage 2-5's FT and the Stage 2b / timebase
    reader -- agrees on the value."""
    p = _create_ftmw(tmp_path)
    duration = float(ftmw.load_fid(p).duration_us)
    ftmw.compute_ft(p, trim=_TRIM)
    _make_pre_provenance(p, start_us="__None__")
    with atomic_write(p):
        update_processing_parameters(p, {"start_us": 3.0})

    assert compute_ft_impl(p)["resolved_settings"].start_us == 3.0
    assert persisted_ft_settings(p, "x", duration).start_us == 3.0


# Mutation caught: the re-persist compares raw None against 0.0/duration and
# wipes downstream stages on an old file.
def test_upgrading_an_unchanged_old_record_invalidates_nothing(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    _make_pre_provenance(p, start_us="__None__", end_us="__None__")
    completed = _completed(p)

    ftmw.compute_ft(p)

    assert _completed(p) == completed
    prov = ft_settings_provenance(p)
    assert prov is not None and prov.is_current
    assert _record(p)["start_us"] == 0.0


# Mutation caught: 'settings unset' records None, reopening the fall-through.
def test_settings_unset_re_resolves_from_the_recommendation(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    with atomic_write(p):
        update_processing_parameters(p, {"start_us": 1.25})
    ftmw.settings_set(p, "stage1.start_us", "2.5")
    assert _record(p)["start_us"] == 2.5
    ftmw.settings_unset(p, "stage1.start_us")
    assert _record(p)["start_us"] == 1.25


# Mutation caught: FTSettings.to_attrs / from_attrs lose the concrete window or
# turn "no trim" into something other than None.
def test_record_round_trips_through_its_codec_at_the_current_version(
    tmp_path: Path,
) -> None:
    from ftmwpipeline.core.settings import FTSettings

    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, start_us=1.5, end_us=12.0, trim=_TRIM)
    attrs = _record(p)
    back = FTSettings.from_attrs(attrs)
    assert attrs[FIELD_SET_VERSION_ATTR] == FT_PROCESSING_FIELD_SET_VERSION
    assert (back.start_us, back.end_us, back.trim) == (1.5, 12.0, _TRIM)
    assert FTSettings.from_attrs(back.to_attrs()) == back

    q = _create_ftmw(tmp_path, "notrim.ftmw")
    ftmw.compute_ft(q, start_us=1.5)
    none_back = FTSettings.from_attrs(_record(q))
    assert none_back.trim is None
    assert none_back.start_us == 1.5


# Mutation caught: ft_settings_provenance counts version 1 / a missing version
# as current, so old records are silently treated as authoritative.
@pytest.mark.parametrize("version", [None, 1])
def test_old_record_is_reported_pre_provenance(tmp_path: Path, version: object) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    with h5py.File(p, "a") as h5f:
        attrs = h5f[FT_PROCESSING_PATH].attrs
        if version is None:
            del attrs[FIELD_SET_VERSION_ATTR]
        else:
            attrs[FIELD_SET_VERSION_ATTR] = version
    prov = ft_settings_provenance(p)
    assert prov is not None
    assert prov.is_pre_provenance and not prov.is_current


# Mutation caught: a v1/unversioned record is treated as authoritative, so its
# recommended-layer fall-through for trim is lost.
def test_old_record_with_unset_trim_still_falls_through(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    _make_pre_provenance(p)
    with h5py.File(p, "a") as h5f:
        h5f[RECOMMENDED_PATH].attrs["trim_min_mhz"] = _TRIM[0]
        h5f[RECOMMENDED_PATH].attrs["trim_max_mhz"] = _TRIM[1]
    assert _resolve_settings(p, None).trim == _TRIM


# Mutation caught: Stage 1 resolves the recommended trim lazily, so a
# recommendation present at run time is not recorded and later edits leak in.
def test_a_recommended_trim_present_at_the_run_is_recorded_then_frozen(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    with h5py.File(p, "a") as h5f:
        h5f[RECOMMENDED_PATH].attrs["trim_min_mhz"] = _TRIM[0]
        h5f[RECOMMENDED_PATH].attrs["trim_max_mhz"] = _TRIM[1]
    ftmw.compute_ft(p)
    attrs = _record(p)
    assert (attrs["trim_min_mhz"], attrs["trim_max_mhz"]) == _TRIM
    with h5py.File(p, "a") as h5f:
        h5f[RECOMMENDED_PATH].attrs["trim_min_mhz"] = 910.0
        h5f[RECOMMENDED_PATH].attrs["trim_max_mhz"] = 960.0
    assert _resolve_settings(p, None).trim == _TRIM
    duration = float(ftmw.load_fid(p).duration_us)
    assert persisted_ft_settings(p, "x", duration).trim == _TRIM


# Mutation caught: a consumer (timebase / Stage 2b / shape recommendation)
# reads its own view instead of persisted_ft_settings.
def test_consumers_read_the_window_from_the_persisted_record(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, start_us=2.0, end_us=11.0, trim=_TRIM)
    with atomic_write(p):
        update_processing_parameters(p, {"start_us": 4.0, "end_us": 9.0})
    duration = float(ftmw.load_fid(p).duration_us)
    got = persisted_ft_settings(p, "timebase_calibration", duration)
    assert got.active_window_us() == (2.0, 11.0)


# Mutation caught: persisted_ft_settings returns defaults when Stage 1 has not
# persisted instead of raising the dependency error.
def test_persisted_reader_requires_a_stage1_record(tmp_path: Path) -> None:
    from ftmwpipeline.file_manager import StageDependencyError

    p = _create_ftmw(tmp_path)
    with pytest.raises(StageDependencyError):
        persisted_ft_settings(p, "timebase_calibration", 20.0)


# Mutation caught: compute_ft swallows a failed persist (the epoch stamp is
# part of completing Stage 1) and returns success on the old window.
def test_a_failed_stage1_stamp_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    before = _record(p)

    def _fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "ftmwpipeline.io.environment_serialization.save_stage_environment", _fail
    )
    with pytest.raises(RuntimeError, match="disk full"):
        ftmw.compute_ft(p, start_us=2.0)
    assert _record(p)["start_us"] == before["start_us"]


# Mutation caught: a Stage 1 re-run removes the Stage 2b results but leaves the
# shape recommendation's record, which then names a verdict no stage uses.
def test_stage1_rerun_withdraws_the_shape_recommendation_record(
    tmp_path: Path,
) -> None:
    from ftmwpipeline.io.tau_calibration_settings_serialization import (
        SHAPE_RECOMMENDATION_SETTINGS_PATH,
    )

    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    with h5py.File(p, "a") as h5f:
        h5f.create_group("stage2b_tau_calibration")
        h5f.create_group(SHAPE_RECOMMENDATION_SETTINGS_PATH)
        stages = h5f["pipeline_stages"]
        completed = json.loads(stages.attrs["completed_stages"])
        stages.attrs["completed_stages"] = json.dumps(
            completed + ["stage2b_tau_calibration"]
        )
    ftmw.compute_ft(p, start_us=2.0)
    with h5py.File(p, "r") as h5f:
        assert "stage2b_tau_calibration" not in h5f
        assert SHAPE_RECOMMENDATION_SETTINGS_PATH not in h5f


# Mutation caught: compute_ft(from_saved_params=True) persists (a read that
# writes), or recomputes a different spectrum than the stored settings give.
def test_from_saved_params_only_reads(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, trim=_TRIM)
    with open(p, "rb") as fh:
        before = fh.read()
    ft = ftmw.compute_ft(p, from_saved_params=True)
    with open(p, "rb") as fh:
        assert fh.read() == before
    ref = compute_ft_impl(p)["complex_ft"]
    np.testing.assert_array_equal(ft.freq_array, ref.freq_array)
    np.testing.assert_array_equal(ft.complex_spectrum, ref.complex_spectrum)
    assert ft.invalidated == ()


# Mutation caught: a from_saved_params call completes Stage 1 on a fresh file.
def test_from_saved_params_does_not_complete_stage1(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, from_saved_params=True)
    assert "stage1_complex_ft" not in _completed(p)
    assert ft_settings_provenance(p) is None


# Mutation caught: a start stamp that moves the start a pre-provenance record
# still falls through to leaves the stages built on the old spectrum standing.
def test_start_stamp_through_an_old_record_invalidates_downstream(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    _make_pre_provenance(p, start_us="__None__")
    with atomic_write(p):
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
    out = detect_start_time_impl(p, settings=_FAST, stamp=True)
    assert out["stamped"] is True
    assert _resolve_settings(p, None).start_us == out["start_us"]
    assert out["invalidated"] == ["noise"]
    assert out["start_detection"].invalidated == ("noise",)
    assert "stage2_noise_result" not in _completed(p)
    assert "stage1_complex_ft" in _completed(p)


# Mutation caught: a stamp under an authoritative record reports (or performs)
# an invalidation although nothing Stage 1 used moved.
def test_start_stamp_under_a_current_record_invalidates_nothing(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    with atomic_write(p):
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
    out = detect_start_time_impl(p, settings=_FAST, stamp=True)
    assert out["invalidated"] == []
    assert "stage2_noise_result" in _completed(p)


# Mutation caught: a Stage 1 re-run that changes the window does not report
# what it invalidated, or reports storage keys / an unordered list.
def test_stage1_rerun_reports_canonical_invalidated(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    same = ftmw.compute_ft(p)
    assert same.invalidated == ()
    moved = ftmw.compute_ft(p, start_us=2.0)
    assert moved.invalidated == ("noise",)
    with atomic_write(p):
        result = compute_ft_impl(p, persist=True)
    assert result["invalidated"] == []
