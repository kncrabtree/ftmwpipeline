"""``preview_source`` is the same on API, Pipeline and CLI.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
file-less source-preview accessor (``dev-docs/CONTRACT_STRATEGY.md`` §Source
preview): ``api.preview_source``, ``Pipeline.preview_source`` and
``ftmwpipeline read preview_source SOURCE --format json`` give equal results,
refuse identically, and leave the source byte-identical. The payload carries
no arrays, so no ``.npy`` comparison applies.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import Absent, capabilities
from ftmwpipeline.serialize import to_jsonable

pytestmark = [pytest.mark.integration]

SCHEMA = "ftmw/source_preview@1"


def _cli(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _wire(result):
    return to_jsonable(result, schema=SCHEMA)


def _csv_source(tmp_path, *, sidecar=True) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame({"v": np.cos(np.arange(64) * 0.1)}).to_csv(path, index=False)
    if sidecar:
        (tmp_path / "data.csv.ftmwmeta.json").write_text(
            json.dumps(
                {
                    "spacing_us": 0.02,
                    "probe_freq_mhz": 40960.0,
                    "sideband": "lower",
                    "shots": 12,
                    "chirp_window": {"chirp_end_us": 1.5},
                }
            )
        )
    return path


def _md5_tree(path: Path) -> dict:
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in files}


# ---- manifest -------------------------------------------------------------


def test_preview_source_is_declared_file_less_with_its_schema():
    assert "preview_source" in MANIFEST.accessors
    assert MANIFEST.file_bound["preview_source"] is False
    assert SCHEMA in MANIFEST.schemas
    cap = capabilities()
    assert "preview_source" in cap["accessors"]
    assert SCHEMA in cap["schemas"]


# ---- equality across interfaces -------------------------------------------


def test_multi_fid_blackchirp_agrees_across_interfaces(exp_2638_data_path, capsys):
    via_api = ftmw.preview_source(exp_2638_data_path)
    via_pipeline = Pipeline.preview_source(exp_2638_data_path)
    rc, out, err = _cli(
        ["read", "preview_source", exp_2638_data_path, "--format", "json"], capsys
    )
    assert rc == 0 and err == ""
    via_cli = json.loads(out)

    assert via_api == via_pipeline
    assert _wire(via_api) == via_cli
    assert via_cli["schema"] == SCHEMA
    assert via_cli["format"] == "blackchirp"
    assert via_cli["n_fids"] == 3
    assert [row["index"] for row in via_cli["fids"]] == [0, 1, 2]
    assert [row["sideband"] for row in via_cli["fids"]] == ["lower"] * 3


def test_csv_with_absent_fields_agrees_across_interfaces(tmp_path, capsys):
    # No sidecar: spacing is NOT_RUN, the chirp window is NOT_RUN. The absence
    # encoding must be identical on every interface.
    src = str(_csv_source(tmp_path, sidecar=False))
    via_api = ftmw.preview_source(src)
    via_pipeline = Pipeline.preview_source(src)
    rc, out, _ = _cli(["read", "preview_source", src, "--format", "json"], capsys)
    via_cli = json.loads(out)

    assert rc == 0
    assert via_api["chirp_window"] is Absent.NOT_RUN
    assert _wire(via_api) == _wire(via_pipeline) == via_cli
    (row,) = via_cli["fids"]
    assert row["spacing_us"] is None and row["spacing_us_absent"] == "not_run"
    # Import defaults are not declarations: silent fields are not_run too.
    for field in ("probe_freq_mhz", "sideband", "shots", "channel"):
        assert row[field] is None and row[field + "_absent"] == "not_run"
    assert via_cli["chirp_window"] is None
    assert via_cli["chirp_window_absent"] == "not_run"


def test_source_format_flag_matches_format_name(tmp_path, capsys):
    src = str(_csv_source(tmp_path))
    rc, out, _ = _cli(
        ["read", "preview_source", src, "--source-format", "csv", "--format", "json"],
        capsys,
    )
    assert rc == 0
    assert json.loads(out) == _wire(ftmw.preview_source(src, "csv"))
    assert json.loads(out) == _wire(Pipeline.preview_source(src, format_name="csv"))
    assert json.loads(out)["chirp_window"]["chirp_end_us"] == 1.5


def test_default_cli_format_is_json(tmp_path, capsys):
    src = str(_csv_source(tmp_path))
    rc, out, _ = _cli(["read", "preview_source", src], capsys)
    assert rc == 0
    assert json.loads(out)["schema"] == SCHEMA


def test_cli_needs_no_output_directory(tmp_path, capsys):
    # Plain lists, not arrays: no --output is needed.
    src = str(_csv_source(tmp_path))
    rc, out, err = _cli(["read", "preview_source", src, "--format", "json"], capsys)
    assert rc == 0 and err == ""
    assert not list(tmp_path.glob("*.npy"))


# ---- read leaves the source byte-identical --------------------------------


def test_read_leaves_source_byte_identical_on_every_interface(
    exp_2638_data_path, tmp_path, capsys
):
    sources = [Path(exp_2638_data_path), _csv_source(tmp_path)]
    before = [_md5_tree(s) for s in sources]
    listing = sorted(str(p) for p in tmp_path.rglob("*"))
    for src in sources:
        ftmw.preview_source(src)
        Pipeline.preview_source(src)
        assert _cli(["read", "preview_source", str(src)], capsys)[0] == 0
    assert [_md5_tree(s) for s in sources] == before
    assert sorted(str(p) for p in tmp_path.rglob("*")) == listing
    assert not list(tmp_path.rglob("*.ftmw"))


# ---- refusals agree -------------------------------------------------------


def test_missing_source_refuses_identically(tmp_path, capsys):
    missing = str(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError) as api_exc:
        ftmw.preview_source(missing)
    with pytest.raises(FileNotFoundError) as pipe_exc:
        Pipeline.preview_source(missing)
    rc, out, err = _cli(["read", "preview_source", missing, "--format", "json"], capsys)
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "not_found" and payload["kind"] == "file"
    assert payload == api_exc.value.to_dict() == pipe_exc.value.to_dict()


def test_unknown_format_refuses_identically(tmp_path, capsys):
    src = str(_csv_source(tmp_path))
    with pytest.raises(ValueError) as api_exc:
        ftmw.preview_source(src, "no-such-format")
    with pytest.raises(ValueError) as pipe_exc:
        Pipeline.preview_source(src, "no-such-format")
    rc, out, err = _cli(
        [
            "read",
            "preview_source",
            src,
            "--source-format",
            "no-such-format",
            "--format",
            "json",
        ],
        capsys,
    )
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["code"] == "bad_setting" and payload["path"] == "format"
    assert payload["value"] == "no-such-format"
    assert payload == api_exc.value.to_dict() == pipe_exc.value.to_dict()


def test_undetectable_source_refuses_identically(tmp_path, capsys):
    odd = tmp_path / "notes.txt"
    odd.write_text("hello")
    with pytest.raises(ValueError) as api_exc:
        ftmw.preview_source(odd)
    rc, out, err = _cli(
        ["read", "preview_source", str(odd), "--format", "json"], capsys
    )
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["code"] == "bad_setting" and payload["path"] == "format"
    assert payload["value"] is None
    assert payload == api_exc.value.to_dict()


def test_source_that_does_not_fit_the_format_refuses_identically(tmp_path, capsys):
    """A loader refusal is ``bad_setting`` (path ``source``), message kept."""
    from ftmwpipeline.file_manager import BadSettingError

    src = str(_csv_source(tmp_path))
    with pytest.raises(BadSettingError) as api_exc:
        ftmw.preview_source(src, "blackchirp")
    with pytest.raises(BadSettingError) as pipe_exc:
        Pipeline.preview_source(src, "blackchirp")
    assert isinstance(api_exc.value, ValueError)
    assert api_exc.value.path == "source"
    assert api_exc.value.value == src
    assert "Not a valid Blackchirp experiment directory" in str(api_exc.value)
    rc, out, err = _cli(
        ["read", "preview_source", src, "--source-format", "blackchirp"], capsys
    )
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "bad_setting" and payload["path"] == "source"
    assert payload == api_exc.value.to_dict() == pipe_exc.value.to_dict()


def test_undefined_chirp_window_and_keysight_channels_agree_across_interfaces(
    tmp_path, capsys
):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"v": np.arange(8.0)}).to_csv(bad, index=False)
    (tmp_path / "bad.csv.ftmwmeta.json").write_text(
        json.dumps({"spacing_us": 0.02, "chirp_window": {"chirp_end_us": "soon"}})
    )
    rc, out, _ = _cli(["read", "preview_source", str(bad)], capsys)
    via_cli = json.loads(out)
    assert rc == 0
    assert via_cli["chirp_window"]["chirp_end_us"] is None
    assert via_cli["chirp_window"]["chirp_end_us_absent"] == "undefined"
    assert (
        via_cli
        == _wire(ftmw.preview_source(bad))
        == _wire(Pipeline.preview_source(bad))
    )

    mat = tmp_path / "scope.mat"
    with h5py.File(mat, "w") as f:
        for name in ("Channel_2", "Channel_1"):
            ch = f.create_group(name)
            ch.create_dataset("Data", data=np.zeros((1, 10), dtype=np.int16))
            ch.create_dataset("XInc", data=np.array([[1e-9]]))
    rc, out, _ = _cli(["read", "preview_source", str(mat)], capsys)
    via_cli = json.loads(out)
    assert rc == 0
    assert [r["channel"] for r in via_cli["fids"]] == ["Channel_1", "Channel_2"]
    assert via_cli == _wire(ftmw.preview_source(mat))
