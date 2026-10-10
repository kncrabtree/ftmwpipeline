"""Explicit load parameters outrank the source; inapplicable ones are refused.

Contract 20. A chirp parameter passed explicitly (``chirp_start_us``,
``chirp_end_us``, ``start_margin_us``) replaces, field by field, the chirp
window the source declares (Blackchirp ``chirps.csv``, a sidecar, embedded
attributes); when any is passed, the derived start ``chirp_end + margin``
outranks a start the source records (Blackchirp ``FidStartUs``). An explicit
margin or chirp start with no chirp end anywhere is refused, as is a parameter
the format does not accept -- ``bad_setting`` with ``path`` the parameter name,
before anything is written. A missing or unparsable ``FidStartUs`` declares no
start; an explicit ``0`` is kept.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.io.data_loaders import get_format_info
from ftmwpipeline.io.data_loaders.base import (
    CHIRP_WINDOW_EXPLICIT_KEY,
    LoadParameterError,
    resolve_chirp_window,
)
from ftmwpipeline.io.data_loaders.blackchirp import BlackChirpLoader
from ftmwpipeline.io.data_loaders.registry import get_loader
from ftmwpipeline.io.input_metadata import resolve_input_metadata
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_recommended_chirp_window,
)

_DATA_2638 = Path(__file__).resolve().parents[3] / "examples/blackchirp_data/2638"
_GUARD = StartDetectionSettings().guard_margin_us

pytestmark = pytest.mark.skipif(
    not _DATA_2638.exists(), reason="examples/blackchirp_data/2638 not present"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def bc(tmp_path: Path) -> Path:
    """A private copy of the 2638 Blackchirp experiment (FidStartUs 2.35,
    chirp end 1.60 us) to edit."""
    dest = tmp_path / "2638"
    shutil.copytree(_DATA_2638, dest)
    return dest


def _set_fid_start(exp: Path, value: Optional[str]) -> None:
    """Rewrite FidStartUs in processing.csv; ``None`` removes the row."""
    proc = exp / "fid" / "processing.csv"
    lines = proc.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("FidStartUs;"):
            if value is None:
                continue
            line = f"FidStartUs;{value}"
        out.append(line)
    proc.write_text("\n".join(out) + "\n")


def _recommended_start(path: Path) -> Optional[float]:
    with h5py.File(path, "r") as h5f:
        raw = h5f["stage0_fid_data/recommended_processing"].attrs["start_us"]
    if isinstance(raw, (bytes, str)):
        return None
    return float(raw)


def _csv(tmp_path: Path, sidecar: Optional[Dict[str, Any]] = None) -> Path:
    path = tmp_path / "volts.csv"
    rng = np.random.default_rng(3)
    path.write_text("v\n" + "\n".join(f"{x:.6f}" for x in rng.normal(size=512)))
    if sidecar is not None:
        (tmp_path / "volts.csv.ftmwmeta.json").write_text(json.dumps(sidecar))
    return path


def _native(tmp_path: Path, **attrs: Any) -> Path:
    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        for key, value in attrs.items():
            f.attrs[key] = value
        f.create_dataset("fid", data=np.cos(np.arange(512) * 0.1))
    return path


def _mat(tmp_path: Path) -> Path:
    """A minimal Keysight MAT (2 us pre + 3 x 5 us frames + 1 us tail, 1 GS/s)."""
    xinc = 1e-9
    n = int(round((2.0 + 3 * 5.0 + 1.0) * 1e-6 / xinc))
    data = np.random.default_rng(7).integers(-500, 500, size=n, dtype=np.int16)
    path = tmp_path / "scope.mat"
    with h5py.File(path, "w") as f:
        ch = f.create_group("Channel_1")
        ch.create_dataset("Data", data=data.reshape(1, -1))
        ch.create_dataset("XInc", data=np.array([[xinc]]))
        ch.create_dataset("XOrg", data=np.array([[0.0]]))
        ch.create_dataset("YInc", data=np.array([[1.0]]))
        ch.create_dataset("YOrg", data=np.array([[0.0]]))
    return path


_MAT_LAYOUT = {"pre_record_us": 2.0, "frame_period_us": 5.0, "n_frames": 3}


# ---------------------------------------------------------------------------
# The merge rule
# ---------------------------------------------------------------------------


class TestResolveChirpWindow:
    def test_nothing_explicit_keeps_the_declaration(self) -> None:
        declared = {"chirp_end_us": 1.6, "chirp_start_us": 0.6}
        merged, explicit = resolve_chirp_window({}, declared)
        assert merged == declared and explicit is False
        assert resolve_chirp_window({"chirp_end_us": None}, None) == (None, False)

    def test_explicit_wins_field_by_field(self) -> None:
        merged, explicit = resolve_chirp_window(
            {"chirp_end_us": 3.6, "start_margin_us": 1.0},
            {"chirp_end_us": 1.6, "chirp_start_us": 0.6},
        )
        assert explicit is True
        assert merged == {
            "chirp_end_us": 3.6,
            "chirp_start_us": 0.6,
            "start_margin_us": 1.0,
        }

    def test_margin_alone_attaches_to_the_declared_end(self) -> None:
        merged, explicit = resolve_chirp_window(
            {"start_margin_us": 0.5}, {"chirp_end_us": 1.6}
        )
        assert merged == {"chirp_end_us": 1.6, "start_margin_us": 0.5}
        assert explicit is True

    @pytest.mark.parametrize("name", ["start_margin_us", "chirp_start_us"])
    def test_nothing_to_attach_to_is_refused(self, name: str) -> None:
        with pytest.raises(LoadParameterError) as exc:
            resolve_chirp_window({name: 0.5}, None)
        assert exc.value.parameter == name and exc.value.value == 0.5

    def test_non_finite_is_refused(self) -> None:
        with pytest.raises(LoadParameterError) as exc:
            resolve_chirp_window({"chirp_end_us": float("nan")}, None)
        assert exc.value.parameter == "chirp_end_us"


# ---------------------------------------------------------------------------
# Blackchirp
# ---------------------------------------------------------------------------


class TestBlackchirp:
    def test_bq_case_explicit_end_and_margin_beat_fid_start(
        self, bc: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "bq.ftmw"
        ftmw.import_data(out, bc, chirp_end_us=3.6, start_margin_us=1.0)
        assert _recommended_start(out) == pytest.approx(4.6)
        cw = read_recommended_chirp_window(str(out))
        assert cw is not None
        assert cw.chirp_end_us == pytest.approx(3.6)
        assert cw.start_margin_us == pytest.approx(1.0)

    def test_margin_alone_combines_with_chirps_csv(
        self, bc: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "margin.ftmw"
        ftmw.import_data(out, bc, start_margin_us=1.0)
        assert _recommended_start(out) == pytest.approx(1.6 + 1.0)
        cw = read_recommended_chirp_window(str(out))
        assert cw is not None and cw.chirp_end_us == pytest.approx(1.6)

    def test_without_explicit_params_fid_start_still_wins(
        self, bc: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "plain.ftmw"
        ftmw.import_data(out, bc)
        assert _recommended_start(out) == pytest.approx(2.35)

    def test_loader_marks_only_explicit_windows(self, bc: Path) -> None:
        loader = BlackChirpLoader()
        assert CHIRP_WINDOW_EXPLICIT_KEY not in loader.load_fid(bc).metadata
        fid = loader.load_fid(bc, chirp_end_us=3.0)
        assert fid.metadata[CHIRP_WINDOW_EXPLICIT_KEY] is True
        assert fid.metadata["chirp_window"]["chirp_end_us"] == 3.0

    def test_missing_fid_start_gives_the_chirp_derived_start(
        self, bc: Path, tmp_path: Path
    ) -> None:
        _set_fid_start(bc, None)
        assert BlackChirpLoader().load_fid(bc).processing.start_us is None
        out = tmp_path / "missing.ftmw"
        ftmw.import_data(out, bc)
        assert _recommended_start(out) == pytest.approx(1.6 + _GUARD)

    def test_unparsable_fid_start_declares_no_start(self, bc: Path) -> None:
        _set_fid_start(bc, "not-a-number")
        assert BlackChirpLoader().load_fid(bc).processing.start_us is None

    def test_an_explicit_zero_fid_start_is_kept(self, bc: Path, tmp_path: Path) -> None:
        _set_fid_start(bc, "0")
        assert BlackChirpLoader().load_fid(bc).processing.start_us == 0.0
        out = tmp_path / "zero.ftmw"
        ftmw.import_data(out, bc)
        assert _recommended_start(out) == 0.0

    def test_margin_without_chirps_csv_is_refused(
        self, bc: Path, tmp_path: Path
    ) -> None:
        (bc / "chirps.csv").unlink()
        out = tmp_path / "nochirp.ftmw"
        with pytest.raises(BadSettingError) as exc:
            ftmw.import_data(out, bc, start_margin_us=1.0)
        assert exc.value.path == "start_margin_us"
        assert exc.value.value == 1.0
        assert not out.exists()
        # An explicit chirp end supplies the window chirps.csv does not.
        ftmw.import_data(out, bc, chirp_end_us=2.0)
        assert _recommended_start(out) == pytest.approx(2.0 + _GUARD)


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------


class TestGeneric:
    def test_csv_margin_alone_combines_with_the_sidecar_end(
        self, tmp_path: Path
    ) -> None:
        src = _csv(
            tmp_path, {"spacing_us": 0.02, "chirp_window": {"chirp_end_us": 1.5}}
        )
        out = tmp_path / "csv.ftmw"
        ftmw.import_data(out, src, start_margin_us=0.25)
        assert _recommended_start(out) == pytest.approx(1.75)
        cw = read_recommended_chirp_window(str(out))
        assert cw is not None and cw.start_margin_us == pytest.approx(0.25)

    def test_csv_margin_with_no_chirp_end_is_refused(self, tmp_path: Path) -> None:
        src = _csv(tmp_path)
        out = tmp_path / "csv.ftmw"
        with pytest.raises(BadSettingError) as exc:
            ftmw.import_data(out, src, spacing_us=0.02, start_margin_us=0.25)
        assert exc.value.path == "start_margin_us"
        assert not out.exists()

    def test_hdf5_explicit_end_beats_embedded_and_keeps_its_margin(
        self, tmp_path: Path
    ) -> None:
        src = _native(tmp_path, chirp_end_us=1.0, start_margin_us=0.5)
        out = tmp_path / "h5.ftmw"
        ftmw.import_data(out, src, chirp_end_us=2.0)
        assert _recommended_start(out) == pytest.approx(2.5)

    def test_resolver_merges_explicit_over_sidecar_over_embedded(self) -> None:
        resolved = resolve_input_metadata(
            explicit={"chirp_window": {"start_margin_us": 0.3}},
            sidecar={"spacing_us": 0.02, "chirp_window": {"chirp_end_us": 1.2}},
            embedded={"chirp_window": {"chirp_end_us": 9.0, "chirp_start_us": 0.1}},
        )
        # The sidecar block wins whole over the embedded one (as before); the
        # explicit field then replaces the field it names.
        assert resolved.chirp_window == {"chirp_end_us": 1.2, "start_margin_us": 0.3}
        assert resolved.chirp_window_explicit is True

    def test_keysight_margin_without_end_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "mat.ftmw"
        with pytest.raises(BadSettingError) as exc:
            ftmw.import_data(out, _mat(tmp_path), start_margin_us=0.4, **_MAT_LAYOUT)
        assert exc.value.path == "start_margin_us"
        assert not out.exists()


# ---------------------------------------------------------------------------
# Inapplicable parameters
# ---------------------------------------------------------------------------


def _sources(tmp_path: Path, bc: Path) -> Dict[str, Path]:
    return {
        "blackchirp": bc,
        "csv": _csv(tmp_path),
        "ftmw-hdf5": _native(tmp_path),
        "keysight-mat": _mat(tmp_path),
    }


_FOREIGN = {
    "blackchirp": {"n_frames": 3, "spacing_us": 0.02, "metadata": "x.json"},
    "csv": {"channel": "Channel_1", "fid_index": 1, "pre_record_us": 1.0},
    "ftmw-hdf5": {"column": 0, "keep_frames": True, "interleave_factors": [16]},
    "keysight-mat": {"spacing_us": 0.02, "fid_index": 0, "metadata": "x.json"},
}


class TestInapplicable:
    def test_every_loader_declares_the_chirp_parameters(self) -> None:
        for name in _FOREIGN:  # the built-in formats (tests may register more)
            accepted = get_loader(name).accepted_parameters()
            for chirp in ("chirp_start_us", "chirp_end_us", "start_margin_us"):
                assert chirp in accepted, (name, chirp)
            info = get_format_info(name)
            assert set(accepted) == set(info["required_parameters"]) | set(
                info["optional_parameters"]
            )

    @pytest.mark.parametrize("auto", [False, True], ids=["named", "auto"])
    def test_a_foreign_parameter_is_refused_before_writing(
        self, bc: Path, tmp_path: Path, auto: bool
    ) -> None:
        sources = _sources(tmp_path, bc)
        for fmt, foreign in _FOREIGN.items():
            accepted = get_loader(fmt).accepted_parameters()
            for param, value in foreign.items():
                out = tmp_path / f"{fmt}-{param}.ftmw"
                # Otherwise-valid parameters for the format, plus the foreign
                # one (fid_index is import_data's own argument; same path).
                kwargs: Dict[str, Any] = {
                    "keysight-mat": dict(_MAT_LAYOUT),
                    "csv": {"spacing_us": 0.02},
                }.get(fmt, {})
                with pytest.raises(BadSettingError) as exc:
                    ftmw.import_data(
                        out,
                        sources[fmt],
                        format_name=None if auto else fmt,
                        **{**kwargs, param: value},
                    )
                err = exc.value
                assert err.path == param, (fmt, param)
                assert err.value == value, (fmt, param)
                for name in accepted:
                    assert name in err.expected, (fmt, param, name)
                assert err.to_dict()["code"] == "bad_setting"
                assert not out.exists(), (fmt, param)

    def test_none_is_not_passed(self, bc: Path, tmp_path: Path) -> None:
        out = tmp_path / "none.ftmw"
        ftmw.import_data(out, bc, n_frames=None, channel=None)
        assert out.exists()
