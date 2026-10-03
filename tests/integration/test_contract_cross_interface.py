"""``capabilities`` is the same on API, Pipeline and CLI; CLI contract plumbing.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
machine contract's file-less accessor, plus the CLI's error and array rules
(``dev-docs/CONTRACT_STRATEGY.md`` §Accessors, §Errors).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import CONTRACT_VERSION, MANIFEST, Pipeline
from ftmwpipeline.cli.contract_commands import (
    EXIT_CODES,
    register_accessor,
)
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import capabilities
from ftmwpipeline.file_manager import (
    NotFoundError,
    PipelineCorruptionError,
    PipelineFileError,
)

pytestmark = [pytest.mark.integration]


def _cli(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_capabilities_agree_across_interfaces(capsys):
    via_api = ftmw.capabilities()
    via_pipeline = Pipeline.capabilities()
    rc, out, _ = _cli(["read", "capabilities", "--format", "json"], capsys)
    assert rc == 0
    via_cli = json.loads(out)
    assert via_api == via_pipeline == via_cli == capabilities()


def test_capabilities_payload_content(capsys):
    cap = ftmw.capabilities()
    assert cap["schema"] == "ftmw/capabilities@1"
    assert cap["contract_version"] == CONTRACT_VERSION
    assert cap["accessors"] == list(MANIFEST.accessors)
    assert cap["codes"] == list(MANIFEST.codes)
    assert cap["schemas"] == list(MANIFEST.schemas)
    json.dumps(cap, allow_nan=False)


def test_capabilities_cli_default_format_and_no_file(capsys):
    rc, out, err = _cli(["read", "capabilities"], capsys)
    assert rc == 0 and err == ""
    assert json.loads(out)["schema"] == "ftmw/capabilities@1"


# ---- plumbing, via a standalone parser with test-only accessors -----------


def _run(accessor, argv, *, schema=None, takes_file=False):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="verb")
    register_accessor(
        sub, "probe", accessor, help="test", schema=schema, takes_file=takes_file
    )
    args = parser.parse_args(argv)
    return args.func(args)


def _raiser(exc):
    def accessor(*a, **k):
        raise exc

    return accessor


def test_contract_error_goes_to_stderr_as_json_with_mapped_exit(capsys):
    exc = NotFoundError("window", [4, 9])
    rc = _run(_raiser(exc), ["probe", "--format", "json"])
    cap = capsys.readouterr()
    assert cap.out == ""
    payload = json.loads(cap.err)
    assert payload == exc.to_dict()
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"
    assert payload["ids"] == [4, 9]
    assert rc == EXIT_CODES["not_found"]


def test_corrupt_file_maps_to_processing_exit(capsys):
    rc = _run(_raiser(PipelineCorruptionError("f", "bad")), ["probe"])
    assert json.loads(capsys.readouterr().err)["code"] == "file_corrupt"
    assert rc == EXIT_CODES["file_corrupt"] == 2


def test_every_manifest_code_has_an_exit_code():
    for code in MANIFEST.codes:
        assert EXIT_CODES[code] in (1, 2)


def test_unlisted_code_defaults_to_user_error(capsys):
    # The base class's fallback code is deliberately not in EXIT_CODES.
    assert "pipeline_error" not in EXIT_CODES
    assert _run(_raiser(PipelineFileError("x")), ["probe"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "pipeline_error"


def test_plain_result_is_stamped_valid_json(capsys):
    rc = _run(
        lambda: {"v": float("nan"), "n": 1},
        ["probe"],
        schema="ftmw/probe@1",
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["schema"] == "ftmw/probe@1" and out["v"] == "nan"


def test_npy_written_to_output_and_named_in_envelope(tmp_path, capsys):
    samples = np.linspace(0.0, 1.0, 5)
    cplx = (np.arange(3) + 1j * np.arange(3)).astype(np.complex128)
    rc = _run(
        lambda: {"samples": samples, "c": cplx, "stored_dtype": "float64"},
        ["probe", "--output", str(tmp_path / "out")],
        schema="ftmw/probe@1",
    )
    env = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert env["samples"] == "samples.npy" and env["c"] == "c.npy"
    np.testing.assert_array_equal(np.load(tmp_path / "out" / "samples.npy"), samples)
    loaded = np.load(tmp_path / "out" / "c.npy")
    assert loaded.dtype == np.complex128
    np.testing.assert_array_equal(loaded, cplx)


def test_array_without_output_is_a_user_error_not_inline(capsys):
    rc = _run(lambda: {"samples": np.zeros(3)}, ["probe"], schema="ftmw/probe@1")
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    assert cap.err.strip()


def test_output_not_written_when_no_arrays(tmp_path, capsys):
    out = tmp_path / "unused"
    rc = _run(lambda: {"a": 1}, ["probe", "-o", str(out)], schema="ftmw/probe@1")
    capsys.readouterr()
    assert rc == 0 and not out.exists()
