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
from ftmwpipeline._internal.stage1_impl import (
    _resolve_settings,
    compute_ft_impl,
    ft_settings_provenance,
    persisted_ft_settings,
)
from ftmwpipeline._internal.start_detection_impl import (
    detect_start_time_impl,
    resolve_start_provenance,
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


def test_concrete_window_selects_the_same_spectrum(tmp_path: Path) -> None:
    """0.0 / the duration select exactly what unset bounds selected."""
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    concrete = compute_persisted_active_ft(p)
    _make_pre_provenance(p, start_us="__None__", end_us="__None__")
    unset = compute_persisted_active_ft(p)
    np.testing.assert_array_equal(concrete.complex_spectrum, unset.complex_spectrum)
    np.testing.assert_array_equal(concrete.freq_mhz, unset.freq_mhz)


def test_a_later_recommended_start_changes_nothing_stage1_used(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p, trim=_TRIM)
    duration = float(ftmw.load_fid(p).duration_us)
    before = compute_persisted_active_ft(p).complex_spectrum.copy()

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
    np.testing.assert_array_equal(
        compute_persisted_active_ft(p).complex_spectrum, before
    )
    # The recommendation is still stored, for display.
    with h5py.File(p, "r") as h5f:
        assert h5f[RECOMMENDED_PATH].attrs["start_us"] == 3.0


def test_an_unset_trim_does_not_fall_through(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    with h5py.File(p, "a") as h5f:
        h5f[RECOMMENDED_PATH].attrs["trim_min_mhz"] = _TRIM[0]
        h5f[RECOMMENDED_PATH].attrs["trim_max_mhz"] = _TRIM[1]
    assert _resolve_settings(p, None).trim is None
    assert compute_ft_impl(p).get("trim_range") is None


def test_start_run_after_stage1_warns_and_leaves_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    completed = _completed(p)
    write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
    with caplog.at_level(logging.WARNING):
        out = detect_start_time_impl(p, settings=_FAST, stamp=True)
    assert out["stamped"] is True
    assert "does not change it" in caplog.text
    assert _resolve_settings(p, None).start_us == 0.0
    assert _completed(p) == completed


def test_explicit_rerun_adopts_a_new_start_and_invalidates(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    assert "stage2_noise_result" in _completed(p)
    ftmw.compute_ft(p, start_us=2.0)
    assert _record(p)["start_us"] == 2.0
    assert "stage2_noise_result" not in _completed(p)


def test_nothing_recommended_reads_as_none_provenance(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    prov = resolve_start_provenance(p)
    assert prov.source == "none"
    assert prov.start_us == 0.0


def test_pre_provenance_record_keeps_its_fall_through(tmp_path: Path) -> None:
    """An older record with an unset start resolves it from the recommended
    layer, and every reader -- Stage 2-5's FT and the Stage 2b / timebase
    reader -- agrees on the value."""
    p = _create_ftmw(tmp_path)
    duration = float(ftmw.load_fid(p).duration_us)
    ftmw.compute_ft(p, trim=_TRIM)
    _make_pre_provenance(p, start_us="__None__")
    update_processing_parameters(p, {"start_us": 3.0})

    assert compute_ft_impl(p)["resolved_settings"].start_us == 3.0
    assert persisted_ft_settings(p, "x", duration).start_us == 3.0


def test_upgrading_an_unchanged_old_record_invalidates_nothing(
    tmp_path: Path,
) -> None:
    p = _create_ftmw(tmp_path)
    ftmw.compute_ft(p)
    ftmw.estimate_noise(p)
    _make_pre_provenance(p, start_us="__None__", end_us="__None__")
    completed = _completed(p)

    ftmw.compute_ft(p, from_saved_params=True)

    assert _completed(p) == completed
    prov = ft_settings_provenance(p)
    assert prov is not None and prov.is_current
    assert _record(p)["start_us"] == 0.0


def test_settings_unset_re_resolves_from_the_recommendation(tmp_path: Path) -> None:
    p = _create_ftmw(tmp_path)
    update_processing_parameters(p, {"start_us": 1.25})
    ftmw.settings_set(p, "stage1.start_us", "2.5")
    assert _record(p)["start_us"] == 2.5
    ftmw.settings_unset(p, "stage1.start_us")
    assert _record(p)["start_us"] == 1.25
