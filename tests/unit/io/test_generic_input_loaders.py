"""Unit tests for the generic data-input loaders and the metadata resolver.

Covers the native ``ftmw-hdf5`` loader, the ``csv`` loader (column selection),
the sidecar metadata file, and the explicit > sidecar > embedded > default
precedence rule.
"""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.io.data_loaders import detect_format, load_fid
from ftmwpipeline.io.data_loaders.base import LoaderError
from ftmwpipeline.io.data_loaders.csv import CSVLoader
from ftmwpipeline.io.data_loaders.ftmw_hdf5 import FtmwHdf5Loader
from ftmwpipeline.io.input_metadata import resolve_input_metadata


def _signal(n: int = 1024) -> np.ndarray:
    return np.cos(2 * np.pi * np.arange(n) * 0.01).astype(float)


def _write_native(
    path,
    *,
    n: int = 1024,
    version: int = 1,
    clocks: bool = False,
    with_fid: bool = True,
    **attrs,
):
    sig = _signal(n)
    with h5py.File(path, "w") as f:
        f.attrs["ftmw_input_version"] = version
        for key, value in attrs.items():
            f.attrs[key] = value
        if with_fid:
            f.create_dataset("fid", data=sig)
        if clocks:
            g = f.create_group("clock_sources")
            g.create_dataset("freq_mhz", data=np.array([5760.0, 6250.0]))
            g.create_dataset("locked", data=np.array([1, 0], dtype=np.int8))
            dt = h5py.string_dtype("utf-8")
            g.create_dataset(
                "label",
                data=np.array(["synth", "digitizer"], dtype=object).astype(dt),
            )
    return sig


# ---------------------------------------------------------------------------
# Resolver precedence
# ---------------------------------------------------------------------------


class TestResolverPrecedence:
    def test_explicit_beats_sidecar_beats_embedded(self):
        resolved = resolve_input_metadata(
            explicit={"spacing_us": 0.01},
            sidecar={"spacing_us": 0.02, "probe_freq_mhz": 100.0, "sideband": "lower"},
            embedded={"spacing_us": 0.03, "probe_freq_mhz": 200.0, "shots": 9},
        )
        assert resolved.spacing_us == 0.01  # explicit wins
        assert resolved.probe_freq_mhz == 100.0  # sidecar over embedded
        assert resolved.sideband == "lower"  # sidecar
        assert resolved.shots == 9  # embedded (only layer that set it)

    def test_defaults_applied(self):
        # Only spacing_us is required; probe_freq_mhz defaults to 0 (a direct
        # sampler whose baseband is the molecular frequency).
        resolved = resolve_input_metadata(embedded={"spacing_us": 0.02})
        assert resolved.probe_freq_mhz == 0.0
        assert resolved.sideband == "upper"
        assert resolved.shots == 1
        assert resolved.clock_sources is None
        assert resolved.spacing_s == pytest.approx(0.02e-6)

    def test_missing_required_raises(self):
        # spacing_us is the only required field; supplying probe but not spacing
        # still raises.
        with pytest.raises(LoaderError, match="spacing_us"):
            resolve_input_metadata(explicit={"probe_freq_mhz": 1.0})

    def test_probe_freq_optional(self):
        # A CSV-style load that omits probe_freq_mhz succeeds with the 0 default.
        resolved = resolve_input_metadata(explicit={"spacing_us": 0.02})
        assert resolved.probe_freq_mhz == 0.0

    def test_bad_sideband_raises(self):
        with pytest.raises(LoaderError, match="sideband"):
            resolve_input_metadata(
                explicit={"spacing_us": 1.0, "probe_freq_mhz": 1.0, "sideband": "x"}
            )


# ---------------------------------------------------------------------------
# Native ftmw-hdf5
# ---------------------------------------------------------------------------


class TestFtmwHdf5Loader:
    def test_detect_and_round_trip(self, tmp_path):
        path = tmp_path / "mydata.h5"
        sig = _write_native(
            path, spacing_us=0.02, probe_freq_mhz=40960.0, sideband="lower", shots=100
        )
        assert detect_format(path) == "ftmw-hdf5"
        fid = load_fid(path)
        assert fid.n_points == len(sig)
        np.testing.assert_allclose(fid.data, sig)
        assert fid.spacing == pytest.approx(0.02e-6)
        assert fid.probe_freq_mhz == 40960.0
        assert fid.sideband == Sideband.LOWER
        assert fid.shots == 100

    def test_embedded_clocks(self, tmp_path):
        path = tmp_path / "mydata.h5"
        _write_native(path, spacing_us=0.02, probe_freq_mhz=40960.0, clocks=True)
        fid = load_fid(path)
        clocks = fid.metadata["clock_sources"]
        assert [c["freq_mhz"] for c in clocks] == [5760.0, 6250.0]
        assert [c["locked"] for c in clocks] == [True, False]
        assert [c["label"] for c in clocks] == ["synth", "digitizer"]

    def test_explicit_overrides_embedded(self, tmp_path):
        path = tmp_path / "mydata.h5"
        _write_native(path, spacing_us=0.02, probe_freq_mhz=40960.0, sideband="upper")
        fid = load_fid(path, sideband="lower", shots=7)
        assert fid.sideband == Sideband.LOWER
        assert fid.shots == 7

    def test_missing_version_attr_not_detected(self, tmp_path):
        path = tmp_path / "plain.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("fid", data=_signal())
        assert detect_format(path) is None
        assert FtmwHdf5Loader().can_load(path) is False

    def test_unsupported_version_raises(self, tmp_path):
        path = tmp_path / "future.h5"
        _write_native(path, version=99, spacing_us=0.02, probe_freq_mhz=1.0)
        with pytest.raises(LoaderError, match="Unsupported"):
            load_fid(path, format_name="ftmw-hdf5")

    def test_missing_fid_dataset_raises(self, tmp_path):
        path = tmp_path / "nofid.h5"
        _write_native(path, with_fid=False, spacing_us=0.02, probe_freq_mhz=1.0)
        with pytest.raises(LoaderError, match="/fid"):
            load_fid(path, format_name="ftmw-hdf5")

    def test_missing_required_metadata_raises(self, tmp_path):
        path = tmp_path / "nometa.h5"
        _write_native(path)  # no spacing/probe attrs
        with pytest.raises(LoaderError, match="spacing_us"):
            load_fid(path, format_name="ftmw-hdf5")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def test_single_column_with_flags(self, tmp_path):
        path = tmp_path / "data.csv"
        sig = _signal()
        pd.DataFrame({"v": sig}).to_csv(path, index=False)
        assert detect_format(path) == "csv"
        fid = load_fid(path, spacing_us=0.02, probe_freq_mhz=40960.0)
        np.testing.assert_allclose(fid.data, sig)
        assert fid.spacing == pytest.approx(0.02e-6)

    def test_column_by_name(self, tmp_path):
        path = tmp_path / "data.csv"
        sig = _signal()
        pd.DataFrame({"t": np.arange(len(sig)), "volts": sig}).to_csv(path, index=False)
        fid = load_fid(path, spacing_us=0.02, probe_freq_mhz=1.0, column="volts")
        np.testing.assert_allclose(fid.data, sig)

    def test_column_by_index(self, tmp_path):
        path = tmp_path / "data.csv"
        sig = _signal()
        pd.DataFrame({"t": np.arange(len(sig)), "volts": sig}).to_csv(path, index=False)
        fid = load_fid(path, spacing_us=0.02, probe_freq_mhz=1.0, column=1)
        np.testing.assert_allclose(fid.data, sig)

    def test_bad_column_name_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        with pytest.raises(LoaderError, match="not found"):
            load_fid(path, spacing_us=0.02, probe_freq_mhz=1.0, column="missing")

    def test_missing_required_metadata_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        with pytest.raises(LoaderError, match="spacing_us"):
            load_fid(path, format_name="csv")


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


class TestSidecar:
    def test_json_sidecar_auto_discovered(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        (tmp_path / "data.csv.ftmwmeta.json").write_text(
            json.dumps(
                {
                    "spacing_us": 0.02,
                    "probe_freq_mhz": 40960.0,
                    "sideband": "lower",
                    "clock_sources": [
                        {"freq_mhz": 5120.0, "locked": True, "label": "lo"}
                    ],
                }
            )
        )
        fid = load_fid(path, format_name="csv")
        assert fid.sideband == Sideband.LOWER
        assert fid.metadata["clock_sources"][0]["freq_mhz"] == 5120.0

    def test_yaml_sidecar_explicit_path(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        meta = tmp_path / "side.yaml"
        meta.write_text("spacing_us: 0.02\nprobe_freq_mhz: 100.0\nsideband: lower\n")
        fid = load_fid(path, format_name="csv", metadata=str(meta))
        assert fid.probe_freq_mhz == 100.0
        assert fid.sideband == Sideband.LOWER

    def test_explicit_flag_overrides_sidecar(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        (tmp_path / "data.csv.ftmwmeta.json").write_text(
            json.dumps({"spacing_us": 0.02, "probe_freq_mhz": 1.0})
        )
        fid = load_fid(path, format_name="csv", probe_freq_mhz=999.0)
        assert fid.probe_freq_mhz == 999.0

    def test_unknown_sidecar_key_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        (tmp_path / "data.csv.ftmwmeta.json").write_text(
            json.dumps({"spacing_us": 0.02, "probe_freq_mhz": 1.0, "typo": 5})
        )
        with pytest.raises(LoaderError, match="Unknown key"):
            load_fid(path, format_name="csv")

    def test_missing_explicit_sidecar_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"v": _signal()}).to_csv(path, index=False)
        with pytest.raises(LoaderError, match="not found"):
            load_fid(
                path,
                format_name="csv",
                spacing_us=0.02,
                probe_freq_mhz=1.0,
                metadata=str(tmp_path / "nope.json"),
            )
