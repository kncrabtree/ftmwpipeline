"""``preview_source``: what a data source holds, found without importing it.

Checks the promises of ``dev-docs/CONTRACT_STRATEGY.md`` §Source preview: the
detected format, one FID-table row per FID the source holds (``index``,
``n_points``, ``spacing_us``, ``probe_freq_mhz``, ``sideband`` decoded to
``"upper"``/``"lower"``, ``shots``), the declared chirp window, absence as
:class:`~ftmwpipeline.contract.Absent`, typed refusals, and that nothing is
imported or written. Sources are synthetic and live in ``tmp_path``.
"""

import hashlib
import json
import logging
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.contract import Absent, FidPreviewRow
from ftmwpipeline.file_manager import NotFoundError, PipelineFileNotFoundError
from ftmwpipeline.io.data_loaders import validate_source
from ftmwpipeline.io.data_loaders.blackchirp import BlackChirpLoader
from ftmwpipeline.serialize import to_jsonable

COLUMNS = ("n_points", "spacing_us", "probe_freq_mhz", "sideband", "shots")
PRESENT, NOT_RUN, UNDEFINED = 0, 1, 2


# ---------------------------------------------------------------------------
# Synthetic sources
# ---------------------------------------------------------------------------


def make_blackchirp(tmp_path: Path, rows, *, name="exp", columns=None) -> Path:
    """A Blackchirp-shaped directory whose fidparams.csv holds ``rows``.

    ``rows`` is a list of dicts keyed by fidparams column name; ``columns``
    restricts which columns are written (to model older files).
    """
    exp = tmp_path / name
    fid_dir = exp / "fid"
    fid_dir.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame[list(columns)]
    frame.to_csv(fid_dir / "fidparams.csv", sep=";", index=False)
    for i in range(len(rows)):
        pd.DataFrame({"data": ["0", "1k", "2z"]}).to_csv(
            fid_dir / f"{i}.csv", index=False
        )
    (exp / "version.csv").write_text(";\nkey;value\nBCMajorVersion;1\n")
    return exp


def bc_row(index=0, sideband="LowerSideband", **kw):
    row = {
        "index": index,
        "spacing": 2e-11,
        "probefreq": 40960.0,
        "vmult": 0.001,
        "shots": 1000 + index,
        "sideband": sideband,
        "size": 750000,
    }
    row.update(kw)
    return row


def make_csv(tmp_path: Path, sidecar=None, name="data.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame({"v": np.cos(np.arange(64) * 0.1)}).to_csv(path, index=False)
    if sidecar is not None:
        (tmp_path / f"{name}.ftmwmeta.json").write_text(json.dumps(sidecar))
    return path


def make_hdf5(tmp_path: Path, name="native.h5", **attrs) -> Path:
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        for key, value in attrs.items():
            f.attrs[key] = value
        f.create_dataset("fid", data=np.cos(np.arange(128) * 0.05))
    return path


def make_keysight(tmp_path: Path, n_samples=1000, xinc=1e-9) -> Path:
    path = tmp_path / "scope.mat"
    with h5py.File(path, "w") as f:
        ch = f.create_group("Channel_1")
        data = np.arange(n_samples, dtype=np.int16) % 100
        ch.create_dataset("Data", data=data.reshape(1, -1))
        ch.create_dataset("XInc", data=np.array([[xinc]]))
        ch.create_dataset("XOrg", data=np.array([[0.0]]))
        ch.create_dataset("YInc", data=np.array([[1.0]]))
        ch.create_dataset("YOrg", data=np.array([[0.0]]))
    return path


def tree_md5(path: Path) -> dict:
    """md5 of every file under ``path`` (or of the file itself)."""
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in files}


def table(result):
    """Columnar view of the record list (``<col>__status`` is 0/1/2 like read_table)."""
    rows = result["fids"]
    assert all(isinstance(r, FidPreviewRow) for r in rows)
    cols = {"index": [r.index for r in rows]}
    for col in COLUMNS:
        values = [getattr(r, col) for r in rows]
        cols[col + "__status"] = [
            (
                {Absent.NOT_RUN: NOT_RUN, Absent.UNDEFINED: UNDEFINED}[v]
                if isinstance(v, Absent)
                else PRESENT
            )
            for v in values
        ]
        cols[col] = values
    return cols


def declare(tmp_path: Path, **kw) -> dict:
    """Preview of a CSV whose sidecar declares exactly ``kw``."""
    return ftmw.preview_source(make_csv(tmp_path, kw))


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


class TestPayloadShape:
    def test_schema_keys_and_one_record_per_fid(self, tmp_path):
        src = make_blackchirp(tmp_path, [bc_row(0), bc_row(1), bc_row(2)])
        r = ftmw.preview_source(src)
        assert r["schema"] == "ftmw/source_preview@1"
        assert set(r) == {
            "schema",
            "source",
            "format",
            "n_fids",
            "fids",
            "chirp_window",
        }
        assert r["format"] == "blackchirp"
        assert r["n_fids"] == 3 == len(r["fids"])
        assert [row.index for row in r["fids"]] == [0, 1, 2]
        assert all(isinstance(row, FidPreviewRow) for row in r["fids"])
        assert {f.name for f in fields(FidPreviewRow)} == {
            "index",
            *COLUMNS,
            "channel",
        }

    def test_payload_is_json_serializable_without_nan(self, tmp_path):
        src = make_csv(tmp_path)  # everything but n_points absent
        wire = to_jsonable(ftmw.preview_source(src))
        json.dumps(wire, allow_nan=False)
        row = wire["fids"][0]
        assert row["spacing_us"] is None and row["spacing_us_absent"] == "not_run"

    def test_result_is_plain_python(self, tmp_path):
        src = make_blackchirp(tmp_path, [bc_row(0)])
        for row in ftmw.preview_source(src)["fids"]:
            for f in fields(row):
                assert not isinstance(getattr(row, f.name), np.generic)


# ---------------------------------------------------------------------------
# Blackchirp: every FID, both sideband encodings
# ---------------------------------------------------------------------------


class TestBlackchirp:
    def test_every_fid_is_reported_with_its_own_values(self, tmp_path):
        rows = [
            bc_row(0, shots=500),
            bc_row(1, shots=60, size=1000, spacing=4e-11, probefreq=20480.0),
            bc_row(2, shots=7, sideband="UpperSideband"),
        ]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["index"] == [0, 1, 2]
        assert cols["n_points"] == [750000, 1000, 750000]
        assert cols["shots"] == [500, 60, 7]
        assert cols["probe_freq_mhz"] == [40960.0, 20480.0, 40960.0]
        assert cols["sideband"] == ["lower", "lower", "upper"]
        for col in COLUMNS:
            assert cols[col + "__status"] == [PRESENT] * 3

    def test_spacing_is_converted_from_seconds_to_microseconds(self, tmp_path):
        rows = [bc_row(0, spacing=2.44140625e-11), bc_row(1, spacing=1e-9)]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["spacing_us"][0] == pytest.approx(2.44140625e-5)
        assert cols["spacing_us"][1] == pytest.approx(1e-3)

    @pytest.mark.parametrize(
        "encoded",
        [["LowerSideband", "UpperSideband"], [1, 0]],
        ids=["enum-names", "integer-codes"],
    )
    def test_both_sideband_encodings_decode(self, tmp_path, encoded):
        rows = [bc_row(i, sideband=s) for i, s in enumerate(encoded)]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["sideband"] == ["lower", "upper"]
        assert cols["sideband__status"] == [PRESENT, PRESENT]

    def test_index_is_row_position_not_the_files_index_column(self, tmp_path):
        rows = [bc_row(7), bc_row(3), bc_row(9)]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["index"] == [0, 1, 2]

    def test_unrecognised_sideband_is_undefined(self, tmp_path):
        rows = [bc_row(0, sideband="LowerSideband"), bc_row(1, sideband="Sideways")]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["sideband__status"] == [PRESENT, UNDEFINED]
        assert cols["sideband"][0] == "lower"

    def test_non_finite_numbers_are_undefined(self, tmp_path):
        rows = [bc_row(0), bc_row(1, spacing=float("inf"), probefreq=float("nan"))]
        cols = table(ftmw.preview_source(make_blackchirp(tmp_path, rows)))
        assert cols["spacing_us__status"] == [PRESENT, UNDEFINED]
        assert cols["probe_freq_mhz__status"] == [PRESENT, UNDEFINED]
        wire = to_jsonable(ftmw.preview_source(tmp_path / "exp"))["fids"]
        assert wire[1]["spacing_us"] is None
        assert wire[1]["spacing_us_absent"] == "undefined"
        assert wire[1]["probe_freq_mhz_absent"] == "undefined"

    def test_columns_an_older_file_lacks_are_not_run(self, tmp_path):
        src = make_blackchirp(
            tmp_path,
            [bc_row(0), bc_row(1)],
            columns=("index", "spacing", "probefreq", "vmult"),
        )
        cols = table(ftmw.preview_source(src))
        assert cols["n_points__status"] == [NOT_RUN] * 2
        assert cols["shots__status"] == [NOT_RUN] * 2
        assert cols["sideband__status"] == [NOT_RUN] * 2
        assert cols["spacing_us__status"] == [PRESENT] * 2
        assert cols["probe_freq_mhz__status"] == [PRESENT] * 2

    def test_chirp_window_absent_when_source_declares_none(self, tmp_path):
        src = make_blackchirp(tmp_path, [bc_row(0)])  # no chirps.csv / header.csv
        assert ftmw.preview_source(src)["chirp_window"] is Absent.NOT_RUN
        wire = to_jsonable(ftmw.preview_source(src))
        assert wire["chirp_window"] is None
        assert wire["chirp_window_absent"] == "not_run"

    def test_chirp_window_of_the_checked_in_experiment(self, exp_2638_path):
        pytest.importorskip("blackchirp")
        window = ftmw.preview_source(exp_2638_path)["chirp_window"]
        assert window["chirp_start_us"] == pytest.approx(0.6)
        assert window["chirp_end_us"] == pytest.approx(1.6)
        assert window["start_margin_us"] is Absent.NOT_RUN

    def test_checked_in_multi_fid_experiment_reports_all_fids(self, exp_2638_path):
        r = ftmw.preview_source(exp_2638_path)
        fid_files = sorted((exp_2638_path / "fid").glob("[0-9]*.csv"))
        assert r["n_fids"] == len(fid_files) == 3
        cols = table(r)
        assert cols["index"] == [0, 1, 2]
        assert cols["sideband"] == ["lower"] * 3
        assert cols["shots"] == [501400, 179860, 359860]
        assert cols["n_points"] == [750000] * 3
        assert cols["spacing_us"] == [pytest.approx(2e-5)] * 3
        assert cols["probe_freq_mhz"] == [40960.0] * 3

    def test_checked_in_integer_encoded_experiment(self):
        path = Path("examples/blackchirp_data/1019")
        if not path.exists():
            pytest.skip("experiment 1019 not available")
        n_rows = len(pd.read_csv(path / "fid" / "fidparams.csv", sep=";"))
        cols = table(ftmw.preview_source(path))
        assert n_rows > 1
        assert cols["index"] == list(range(n_rows))
        assert cols["sideband"] == ["lower"] * n_rows  # stored as the integer 1


@pytest.fixture
def exp_2638_path() -> Path:
    path = Path("examples/blackchirp_data/2638")
    if not path.exists():
        pytest.skip("Experiment 2638 data not available")
    return path


# ---------------------------------------------------------------------------
# Single-FID loaders return exactly one row
# ---------------------------------------------------------------------------


class TestSingleFidLoaders:
    def test_csv_with_sidecar(self, tmp_path):
        src = make_csv(
            tmp_path,
            {
                "spacing_us": 0.02,
                "probe_freq_mhz": 40960.0,
                "sideband": "lower",
                "shots": 250,
                "chirp_window": {"chirp_end_us": 1.5, "chirp_start_us": 0.5},
            },
        )
        r = ftmw.preview_source(src)
        assert r["format"] == "csv" and r["n_fids"] == 1
        cols = table(r)
        assert cols["index"] == [0]
        assert cols["n_points"] == [64]
        assert cols["spacing_us"] == [0.02]
        assert cols["probe_freq_mhz"] == [40960.0]
        assert cols["sideband"] == ["lower"]
        assert cols["shots"] == [250]
        for col in COLUMNS:
            assert cols[col + "__status"] == [PRESENT]
        assert r["chirp_window"]["chirp_end_us"] == 1.5
        assert r["chirp_window"]["chirp_start_us"] == 0.5
        assert r["chirp_window"]["start_margin_us"] is Absent.NOT_RUN

    def test_csv_without_sidecar_reports_absent_spacing(self, tmp_path):
        r = ftmw.preview_source(make_csv(tmp_path))
        cols = table(r)
        assert cols["n_points"] == [64]
        assert cols["spacing_us__status"] == [NOT_RUN]
        assert to_jsonable(r)["fids"][0]["spacing_us_absent"] == "not_run"
        assert r["chirp_window"] is Absent.NOT_RUN

    def test_hdf5_embedded_attributes(self, tmp_path):
        src = make_hdf5(
            tmp_path,
            spacing_us=0.002,
            probe_freq_mhz=12000.0,
            sideband="lower",
            shots=40,
            chirp_end_us=2.0,
        )
        r = ftmw.preview_source(src)
        assert r["format"] == "ftmw-hdf5" and r["n_fids"] == 1
        cols = table(r)
        assert cols["n_points"] == [128]
        assert cols["spacing_us"] == [0.002]
        assert cols["probe_freq_mhz"] == [12000.0]
        assert cols["sideband"] == ["lower"]
        assert cols["shots"] == [40]
        assert r["chirp_window"]["chirp_end_us"] == 2.0
        assert r["chirp_window"]["chirp_start_us"] is Absent.NOT_RUN

    def test_hdf5_without_spacing_or_window(self, tmp_path):
        r = ftmw.preview_source(make_hdf5(tmp_path))
        assert table(r)["spacing_us__status"] == [NOT_RUN]
        assert r["chirp_window"] is Absent.NOT_RUN

    def test_keysight(self, tmp_path):
        r = ftmw.preview_source(make_keysight(tmp_path, n_samples=1000, xinc=1e-9))
        assert r["format"] == "keysight-mat" and r["n_fids"] == 1
        (row,) = r["fids"]
        assert row.index == 0
        assert row.channel == "Channel_1"
        assert row.spacing_us == pytest.approx(1e-3)
        # Depend on load-time acquisition-layout parameters.
        assert row.n_points is Absent.UNDEFINED
        assert row.shots is Absent.UNDEFINED
        # The record does not declare these; import's fixed direct-sampler
        # values are not the source's declaration.
        assert row.probe_freq_mhz is Absent.NOT_RUN
        assert row.sideband is Absent.NOT_RUN
        assert r["chirp_window"] is Absent.NOT_RUN


# ---------------------------------------------------------------------------
# Import defaults are never reported as declared values
# ---------------------------------------------------------------------------


class TestImportDefaultsAreNotDeclarations:
    @pytest.mark.parametrize("make", ["csv", "hdf5"])
    def test_silent_source_reports_not_run_for_all_defaulted_fields(
        self, tmp_path, make
    ):
        src = make_csv(tmp_path) if make == "csv" else make_hdf5(tmp_path)
        (row,) = ftmw.preview_source(src)["fids"]
        assert row.probe_freq_mhz is Absent.NOT_RUN  # not 0 MHz
        assert row.sideband is Absent.NOT_RUN  # not "upper"
        assert row.shots is Absent.NOT_RUN  # not 1
        assert row.spacing_us is Absent.NOT_RUN

    def test_import_still_applies_its_defaults(self, tmp_path):
        from ftmwpipeline.io.data_loaders import load_fid

        src = make_csv(tmp_path, {"spacing_us": 0.02})
        fid = load_fid(src, "csv")
        assert fid.probe_freq_mhz == 0.0
        assert fid.sideband.value == "upper"
        assert fid.shots == 1
        (row,) = ftmw.preview_source(src)["fids"]
        assert row.spacing_us == 0.02
        assert row.probe_freq_mhz is Absent.NOT_RUN

    def test_each_declared_field_is_reported_independently(self, tmp_path):
        one = declare(tmp_path, spacing_us=0.02, shots=7)
        (row,) = one["fids"]
        assert row.shots == 7
        assert row.probe_freq_mhz is Absent.NOT_RUN
        assert row.sideband is Absent.NOT_RUN

    def test_explicit_zero_and_upper_are_declarations(self, tmp_path):
        r = declare(
            tmp_path, spacing_us=0.02, probe_freq_mhz=0.0, sideband="upper", shots=1
        )
        (row,) = r["fids"]
        assert (row.probe_freq_mhz, row.sideband, row.shots) == (0.0, "upper", 1)

    def test_hdf5_sidecar_overrides_embedded_and_silence_stays_not_run(self, tmp_path):
        src = make_hdf5(tmp_path, spacing_us=0.002, shots=40)
        (tmp_path / "native.h5.ftmwmeta.json").write_text(json.dumps({"shots": 9}))
        (row,) = ftmw.preview_source(src)["fids"]
        assert row.shots == 9
        assert row.spacing_us == 0.002
        assert row.probe_freq_mhz is Absent.NOT_RUN
        assert row.sideband is Absent.NOT_RUN


# ---------------------------------------------------------------------------
# Keysight: one row per channel
# ---------------------------------------------------------------------------


def make_multichannel_keysight(tmp_path: Path, channels) -> Path:
    """``channels`` maps group name -> XInc."""
    path = tmp_path / "multi.mat"
    with h5py.File(path, "w") as f:
        for name, xinc in channels.items():
            ch = f.create_group(name)
            ch.create_dataset("Data", data=np.zeros((1, 100), dtype=np.int16))
            ch.create_dataset("XInc", data=np.array([[xinc]]))
            ch.create_dataset("XOrg", data=np.array([[0.0]]))
            ch.create_dataset("YInc", data=np.array([[1.0]]))
            ch.create_dataset("YOrg", data=np.array([[0.0]]))
    return path


class TestKeysightChannels:
    def test_one_row_per_channel_with_its_own_spacing(self, tmp_path):
        src = make_multichannel_keysight(
            tmp_path, {"Channel_3": 2e-9, "Channel_1": 1e-9, "Channel_2": 4e-9}
        )
        r = ftmw.preview_source(src)
        assert r["n_fids"] == 3
        rows = r["fids"]
        assert [row.index for row in rows] == [0, 1, 2]
        assert [row.channel for row in rows] == ["Channel_1", "Channel_2", "Channel_3"]
        assert [row.spacing_us for row in rows] == pytest.approx([1e-3, 4e-3, 2e-3])
        for row in rows:
            assert row.n_points is Absent.UNDEFINED
            assert row.shots is Absent.UNDEFINED

    def test_channel_value_is_what_import_needs(self, tmp_path):
        from ftmwpipeline.io.data_loaders import load_fid

        src = make_multichannel_keysight(
            tmp_path, {"Channel_1": 1e-9, "Channel_4": 5e-9}
        )
        params = dict(pre_record_us=0.0, frame_period_us=0.1, n_frames=1)
        with pytest.raises(Exception, match="Multiple channels"):
            load_fid(src, "keysight-mat", **params)
        for row in ftmw.preview_source(src)["fids"]:
            fid = load_fid(src, "keysight-mat", channel=row.channel, **params)
            assert fid.spacing * 1e6 == pytest.approx(row.spacing_us)

    def test_channelless_sources_report_not_run(self, tmp_path):
        sources = [
            make_blackchirp(tmp_path, [bc_row(0), bc_row(1)]),
            make_csv(tmp_path, {"spacing_us": 0.02}),
            make_hdf5(tmp_path, spacing_us=0.002),
        ]
        for src in sources:
            for row in ftmw.preview_source(src)["fids"]:
                assert row.channel is Absent.NOT_RUN

    def test_channel_travels_on_the_wire(self, tmp_path):
        src = make_multichannel_keysight(
            tmp_path, {"Channel_1": 1e-9, "Channel_2": 1e-9}
        )
        wire = to_jsonable(ftmw.preview_source(src))["fids"]
        assert [w["channel"] for w in wire] == ["Channel_1", "Channel_2"]
        csv_wire = to_jsonable(ftmw.preview_source(make_csv(tmp_path)))["fids"][0]
        assert csv_wire["channel"] is None
        assert csv_wire["channel_absent"] == "not_run"


# ---------------------------------------------------------------------------
# Chirp window: declared but unreadable is UNDEFINED
# ---------------------------------------------------------------------------


class TestChirpWindowValidation:
    @pytest.mark.parametrize(
        "bad", ["abc", float("nan"), float("inf"), True, [1.0], {"a": 1}]
    )
    def test_csv_sidecar_invalid_chirp_value_is_undefined(self, tmp_path, bad):
        src = make_csv(
            tmp_path,
            {
                "spacing_us": 0.02,
                "chirp_window": {"chirp_end_us": 1.5, "chirp_start_us": bad},
            },
        )
        window = ftmw.preview_source(src)["chirp_window"]
        assert window["chirp_end_us"] == 1.5
        assert window["chirp_start_us"] is Absent.UNDEFINED
        assert window["start_margin_us"] is Absent.NOT_RUN

    def test_csv_sidecar_invalid_chirp_end_is_undefined(self, tmp_path):
        src = make_csv(
            tmp_path, {"spacing_us": 0.02, "chirp_window": {"chirp_end_us": "soon"}}
        )
        window = ftmw.preview_source(src)["chirp_window"]
        assert window["chirp_end_us"] is Absent.UNDEFINED

    @pytest.mark.parametrize("block", ["1.5", [1, 2], 3.0, {}, {"chirp_start_us": 1.0}])
    def test_csv_sidecar_unusable_block_is_undefined_as_a_whole(self, tmp_path, block):
        src = make_csv(tmp_path, {"spacing_us": 0.02, "chirp_window": block})
        assert ftmw.preview_source(src)["chirp_window"] is Absent.UNDEFINED

    def test_undefined_window_travels_on_the_wire(self, tmp_path):
        src = make_csv(tmp_path, {"spacing_us": 0.02, "chirp_window": "bad"})
        wire = to_jsonable(ftmw.preview_source(src))
        assert wire["chirp_window"] is None
        assert wire["chirp_window_absent"] == "undefined"

    def test_integer_chirp_values_are_valid(self, tmp_path):
        src = make_csv(
            tmp_path, {"spacing_us": 0.02, "chirp_window": {"chirp_end_us": 2}}
        )
        assert ftmw.preview_source(src)["chirp_window"]["chirp_end_us"] == 2.0

    @pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf")])
    def test_hdf5_attribute_invalid_chirp_value_is_undefined(self, tmp_path, bad):
        src = make_hdf5(
            tmp_path, spacing_us=0.002, chirp_end_us=2.0, chirp_start_us=bad
        )
        r = ftmw.preview_source(src)
        assert r["n_fids"] == 1  # a bad window must not refuse the preview
        assert r["chirp_window"]["chirp_end_us"] == 2.0
        assert r["chirp_window"]["chirp_start_us"] is Absent.UNDEFINED

    def test_hdf5_attribute_invalid_chirp_end_is_undefined(self, tmp_path):
        src = make_hdf5(tmp_path, spacing_us=0.002, chirp_end_us=float("nan"))
        window = ftmw.preview_source(src)["chirp_window"]
        assert window["chirp_end_us"] is Absent.UNDEFINED

    def test_hdf5_invalid_sidecar_window_wins_and_is_undefined(self, tmp_path):
        src = make_hdf5(tmp_path, chirp_end_us=2.0)
        (tmp_path / "native.h5.ftmwmeta.json").write_text(
            json.dumps({"chirp_window": {"chirp_end_us": "x"}})
        )
        window = ftmw.preview_source(src)["chirp_window"]
        assert window["chirp_end_us"] is Absent.UNDEFINED

    def test_blackchirp_declared_but_unparsable_is_undefined_and_logged(
        self, tmp_path, caplog
    ):
        pytest.importorskip("blackchirp")
        src = make_blackchirp(tmp_path, [bc_row(0)])
        (src / "chirps.csv").write_text("garbage\n")  # declared, unreadable
        with caplog.at_level(logging.DEBUG, logger="ftmwpipeline.io.data_loaders"):
            r = ftmw.preview_source(src)
        assert r["chirp_window"] is Absent.UNDEFINED
        assert any(
            rec.levelno == logging.DEBUG and "unreadable" in rec.getMessage()
            for rec in caplog.records
        )

    def test_blackchirp_without_chirps_csv_is_not_run(self, tmp_path):
        src = make_blackchirp(tmp_path, [bc_row(0)])
        assert ftmw.preview_source(src)["chirp_window"] is Absent.NOT_RUN

    def test_blackchirp_unparsable_without_blackchirp_package_is_undefined(
        self, tmp_path, monkeypatch
    ):
        src = make_blackchirp(tmp_path, [bc_row(0)])
        (src / "chirps.csv").write_text("Chirp;DurationUs\n0;1.0\n")

        def boom(*a, **k):
            raise RuntimeError("cannot parse")

        monkeypatch.setattr(BlackChirpLoader, "_parse_chirp_window", staticmethod(boom))
        assert ftmw.preview_source(src)["chirp_window"] is Absent.UNDEFINED


# ---------------------------------------------------------------------------
# Format selection
# ---------------------------------------------------------------------------


class TestFormatSelection:
    def test_named_format_matches_detection(self, tmp_path):
        src = make_csv(tmp_path, {"spacing_us": 0.02})
        assert ftmw.preview_source(src, "csv") == ftmw.preview_source(src)
        assert ftmw.preview_source(src, format_name="csv")["format"] == "csv"

    def test_pipeline_accessor_is_static_and_equal(self, tmp_path):
        src = make_csv(tmp_path, {"spacing_us": 0.02})
        assert Pipeline.preview_source(src) == ftmw.preview_source(src)
        assert Pipeline.preview_source(src, "csv") == ftmw.preview_source(src, "csv")

    def test_accepts_str_and_path(self, tmp_path):
        src = make_csv(tmp_path)
        a, b = ftmw.preview_source(src), ftmw.preview_source(str(src))
        assert to_jsonable(a)["fids"] == to_jsonable(b)["fids"]
        assert a["format"] == b["format"] == "csv"


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_missing_source_is_not_found_file(self, tmp_path):
        missing = tmp_path / "nowhere"
        with pytest.raises(PipelineFileNotFoundError) as info:
            ftmw.preview_source(missing)
        assert info.value.code == "not_found"
        assert info.value.kind == "file"
        assert info.value.ids == [str(missing)]
        assert isinstance(info.value, FileNotFoundError)

    def test_unknown_format_name_is_not_found_format(self, tmp_path):
        src = make_csv(tmp_path)
        with pytest.raises(NotFoundError) as info:
            ftmw.preview_source(src, "no-such-format")
        assert info.value.code == "not_found"
        assert info.value.kind == "format"
        assert info.value.ids == ["no-such-format"]
        assert info.value.to_dict()["schema"] == "ftmw/error@1"

    def test_undetectable_source_is_not_found_format_with_no_ids(self, tmp_path):
        odd = tmp_path / "notes.txt"
        odd.write_text("hello")
        with pytest.raises(NotFoundError) as info:
            ftmw.preview_source(odd)
        assert info.value.code == "not_found"
        assert info.value.kind == "format"
        assert info.value.ids == []

    def test_empty_directory_is_not_found_format(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(NotFoundError) as info:
            ftmw.preview_source(empty)
        assert info.value.kind == "format"

    def test_source_that_does_not_fit_the_named_format_is_value_error(self, tmp_path):
        src = make_csv(tmp_path)
        with pytest.raises(ValueError):
            ftmw.preview_source(src, "blackchirp")
        with pytest.raises(ValueError):
            ftmw.preview_source(src, "keysight-mat")

    def test_refusals_agree_on_pipeline(self, tmp_path):
        with pytest.raises(PipelineFileNotFoundError):
            Pipeline.preview_source(tmp_path / "nowhere")
        with pytest.raises(NotFoundError):
            Pipeline.preview_source(make_csv(tmp_path), "no-such-format")


# ---------------------------------------------------------------------------
# Nothing imported, nothing written
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_sources_are_byte_identical_and_nothing_is_created(self, tmp_path):
        sources = [
            make_blackchirp(tmp_path, [bc_row(0), bc_row(1)]),
            make_csv(tmp_path, {"spacing_us": 0.02}),
            make_hdf5(tmp_path, spacing_us=0.002),
            make_keysight(tmp_path),
        ]
        listing_before = sorted(str(p) for p in tmp_path.rglob("*"))
        digests = [tree_md5(s) for s in sources]
        for src in sources:
            ftmw.preview_source(src)
            Pipeline.preview_source(src)
        assert [tree_md5(s) for s in sources] == digests
        assert sorted(str(p) for p in tmp_path.rglob("*")) == listing_before
        assert not list(tmp_path.rglob("*.ftmw"))

    def test_checked_in_experiment_is_byte_identical(self, exp_2638_path):
        before = tree_md5(exp_2638_path)
        ftmw.preview_source(exp_2638_path)
        assert tree_md5(exp_2638_path) == before

    def test_no_fid_is_loaded(self, tmp_path, monkeypatch):
        def refuse(*args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("preview_source must not import FID data")

        from ftmwpipeline.io.data_loaders.csv import CSVLoader
        from ftmwpipeline.io.data_loaders.ftmw_hdf5 import FtmwHdf5Loader
        from ftmwpipeline.io.data_loaders.keysight_mat import KeysightMatLoader

        for cls in (BlackChirpLoader, CSVLoader, FtmwHdf5Loader, KeysightMatLoader):
            monkeypatch.setattr(cls, "load_fid", refuse)
        for src in (
            make_blackchirp(tmp_path, [bc_row(0), bc_row(1)]),
            make_csv(tmp_path),
            make_hdf5(tmp_path),
            make_keysight(tmp_path),
        ):
            assert ftmw.preview_source(src)["n_fids"] >= 1


# ---------------------------------------------------------------------------
# validate_source is unchanged
# ---------------------------------------------------------------------------


class TestValidateSourceUnchanged:
    def test_validate_source_still_reports_one_fid_entry_and_no_fid_table(
        self, tmp_path
    ):
        src = make_blackchirp(tmp_path, [bc_row(0), bc_row(1), bc_row(2)])
        result = validate_source(src)
        assert result["valid"] is True
        assert set(result) == {"format", "valid", "metadata", "options", "errors"}
        assert "fids" not in result and "fids" not in result["metadata"]
        # It still describes FID 0 alone (bc_row(0) holds 1000 shots) ...
        assert result["metadata"]["shots"] == 1000
        # ... while the preview is a separate capability that sees all three.
        assert ftmw.preview_source(src)["n_fids"] == 3
