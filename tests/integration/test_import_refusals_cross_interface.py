"""Import refusals are typed and agree across interfaces (CONTRACT_STRATEGY, Errors).

* A source path that does not exist is ``not_found`` (``kind="file"``, ``ids``
  the path), which is also the ``FileNotFoundError`` it always was.
* A source that exists but that the resolved format's loader does not accept is
  ``bad_setting`` with ``path`` ``"source"`` (the source path as ``value``),
  which is also the ``ValueError`` it always was.

``import_data``, ``Pipeline.create``, ``run_pipeline`` and the CLI (under
``--json``) raise or report the same error, and a refused import writes no file.

Neither case needs a built experiment: an empty directory is a source that
exists and that the ``blackchirp`` loader refuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.stage0_impl import import_data_impl
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import (
    BadSettingError,
    PipelineFileError,
    PipelineFileNotFoundError,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def missing_source(tmp_path) -> Path:
    return tmp_path / "no_such_source"


@pytest.fixture
def refused_source(tmp_path) -> Path:
    src = tmp_path / "empty_experiment"
    src.mkdir()
    return src


def _imports(tmp_path: Path, source: Path, **kw: Any):
    """Every way to start an import, as (name, zero-arg callable)."""
    return [
        (
            "impl",
            lambda: import_data_impl(str(tmp_path / "impl.ftmw"), source=source, **kw),
        ),
        ("api", lambda: ftmw.import_data(tmp_path / "api.ftmw", source=source, **kw)),
        (
            "pipeline",
            lambda: Pipeline.create(tmp_path / "pipe.ftmw", source=source, **kw),
        ),
    ]


def test_a_missing_source_is_not_found_on_every_interface(tmp_path, missing_source):
    for name, call in _imports(tmp_path, missing_source):
        with pytest.raises(PipelineFileNotFoundError) as exc:
            call()
        err = exc.value
        assert isinstance(err, FileNotFoundError), name
        assert isinstance(err, PipelineFileError), name
        assert err.kind == "file", name
        assert err.ids == [str(missing_source)], name
        assert "Source path does not exist" in str(err), name
        d = err.to_dict()
        assert d["code"] == "not_found" and d["kind"] == "file", name
    assert not list(tmp_path.glob("*.ftmw"))


def test_a_refused_source_is_bad_setting_on_the_source_on_every_interface(
    tmp_path, refused_source
):
    for name, call in _imports(tmp_path, refused_source, format_name="blackchirp"):
        with pytest.raises(BadSettingError) as exc:
            call()
        err = exc.value
        assert isinstance(err, ValueError), name
        assert err.path == "source", name
        assert err.value == str(refused_source), name
        assert "blackchirp" in err.expected, name
        assert "Source validation failed" in str(err), name
        d = err.to_dict()
        assert d["code"] == "bad_setting" and d["path"] == "source", name
    assert not list(tmp_path.glob("*.ftmw"))


def test_an_undetectable_or_unknown_format_is_still_named_format(
    tmp_path, refused_source
):
    with pytest.raises(BadSettingError) as exc:
        ftmw.import_data(tmp_path / "a.ftmw", source=refused_source)
    assert exc.value.path == "format"
    with pytest.raises(BadSettingError) as exc:
        ftmw.import_data(
            tmp_path / "b.ftmw", source=refused_source, format_name="bogus"
        )
    assert exc.value.path == "format"


def _cli_error(capsys, *argv: Any):
    capsys.readouterr()
    rc = main([str(a) for a in argv] + ["--json"])
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    return rc, json.loads(lines[-1])


def test_the_cli_reports_the_same_errors_under_json(
    tmp_path, capsys, missing_source, refused_source
):
    rc, payload = _cli_error(
        capsys, "data", "import", tmp_path / "cli_a.ftmw", missing_source
    )
    assert rc == 1
    assert payload["code"] == "not_found" and payload["kind"] == "file"
    assert payload["ids"] == [str(missing_source)]

    rc, payload = _cli_error(
        capsys,
        "data",
        "import",
        tmp_path / "cli_b.ftmw",
        refused_source,
        "--format",
        "blackchirp",
    )
    assert rc == 1
    assert payload["code"] == "bad_setting" and payload["path"] == "source"
    assert payload["value"] == str(refused_source)
    assert not list(tmp_path.glob("*.ftmw"))


def test_run_pipeline_reports_the_typed_code(tmp_path, missing_source):
    """``run_pipeline`` carries the failure's error dict: a missing source is
    ``not_found`` now, not the ``pipeline_error`` fallback of an untyped one."""
    result: Dict[str, Any] = ftmw.run_pipeline(
        missing_source,
        output=tmp_path / "run.ftmw",
        trim=(26500, 40000),
        progress=False,
    )
    assert result["status"] == "error"
    assert result["failed_stage"] == "data"
    assert result["error"]["schema"] == "ftmw/error@1"
    assert result["error"]["code"] == "not_found"
    assert result["error"]["kind"] == "file"
    assert result["completed_stages"] == []


# ---- a source that validates but its loader refuses -------------------------


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "volts.csv"
    path.write_text("v\n1.0\n0.5\n-0.25\n0.125\n")
    return path


def _loader_refusals(tmp_path: Path):
    """(label, loader params, message fragment) for refusals made at load time,
    after the source has passed validation."""
    sidecar = tmp_path / "side.json"
    sidecar.write_text(json.dumps({"spacing_us": 0.02, "no_such_key": 1}))
    return [
        ("missing spacing_us", {"probe_freq_mhz": 40960.0}, "spacing_us"),
        (
            "unknown column",
            {"spacing_us": 0.02, "probe_freq_mhz": 40960.0, "column": "nosuch"},
            "nosuch",
        ),
        ("unknown sidecar key", {"metadata": str(sidecar)}, "no_such_key"),
    ]


def test_a_loader_refusal_is_bad_setting_on_the_source_on_every_interface(
    tmp_path,
):
    """Not ``RuntimeError("Failed to load FID data")``: the loader's refusal is
    ``bad_setting`` (path ``source``) with the loader's message kept."""
    src = _csv(tmp_path)
    for label, params, fragment in _loader_refusals(tmp_path):
        for name, call in _imports(tmp_path, src, format_name="csv", **params):
            with pytest.raises(BadSettingError) as exc:
                call()
            err = exc.value
            assert isinstance(err, ValueError), (label, name)
            assert err.path == "source", (label, name)
            assert err.value == str(src), (label, name)
            assert "csv" in err.expected, (label, name)
            assert fragment in str(err), (label, name)
            assert err.to_dict()["code"] == "bad_setting", (label, name)
    assert not list(tmp_path.glob("*.ftmw"))


def test_the_cli_reports_a_loader_refusal_as_the_error_dict(tmp_path, capsys):
    src = _csv(tmp_path)
    rc, payload = _cli_error(
        capsys,
        "data",
        "import",
        tmp_path / "cli_c.ftmw",
        src,
        "--format",
        "csv",
        "--probe_freq_mhz",
        "40960",
    )
    assert rc == 1
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "bad_setting" and payload["path"] == "source"
    assert payload["value"] == str(src)
    assert "spacing_us" in payload["message"]
    with pytest.raises(BadSettingError) as exc:
        ftmw.import_data(
            tmp_path / "api_c.ftmw",
            source=src,
            format_name="csv",
            probe_freq_mhz=40960.0,
        )
    assert payload == exc.value.to_dict()
    assert not list(tmp_path.glob("*.ftmw"))


def test_the_cli_does_not_swallow_a_keyboard_interrupt(tmp_path, monkeypatch):
    """A second Ctrl-C (a ``KeyboardInterrupt``) is not turned into exit 1 with
    "Operation canceled by user"; it propagates like every other verb's."""
    import ftmwpipeline.cli.data_commands as data_commands

    def _interrupt(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(data_commands, "import_data_impl", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        main(["data", "import", str(tmp_path / "k.ftmw"), str(_csv(tmp_path))])


def test_a_cancelled_import_exits_130_with_the_cancelled_error(
    tmp_path, capsys, monkeypatch
):
    import ftmwpipeline.cli.data_commands as data_commands
    from ftmwpipeline.file_manager import OperationCancelledError

    def _cancelled(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise OperationCancelledError("data")

    monkeypatch.setattr(data_commands, "import_data_impl", _cancelled)
    rc, payload = _cli_error(
        capsys, "data", "import", tmp_path / "c.ftmw", _csv(tmp_path)
    )
    assert rc == 130
    assert payload["code"] == "cancelled"
