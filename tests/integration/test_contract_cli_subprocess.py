"""The contract's CLI behaviour, observed from outside the process.

Every test runs ``python -m ftmwpipeline`` (or a few lines of Python that wire
a probe accessor into the real CLI plumbing) as a SUBPROCESS and checks what a
calling program sees: the bytes on stdout/stderr and the exit code.
``PYTHONPATH`` is derived from the imported package, so the subprocess runs the
same source tree as the test (a worktree or the main checkout).

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Errors (codes, exit-code table, the
missing / corrupt file rule) and §Accessors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline
from ftmwpipeline import MANIFEST, Pipeline
from ftmwpipeline._internal.read_impl import (
    READ_TABLES,
    format_metadata_impl,
    read_metadata_impl,
)
from ftmwpipeline.contract import capabilities

pytestmark = [pytest.mark.integration]

#: Directory that contains the imported ``ftmwpipeline`` package.
_PKG_ROOT = str(Path(ftmwpipeline.__file__).resolve().parent.parent)


def _env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _PKG_ROOT + (os.pathsep + existing if existing else "")
    return env


def _cli(*argv: str):
    return subprocess.run(
        [sys.executable, "-m", "ftmwpipeline", *argv],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )


#: A file-bound probe accessor wired through the real ``register_accessor``.
#: ``Pipeline.open`` is what the first real file-bound accessor will do.
_PROBE = textwrap.dedent("""
    import argparse, sys
    from types import SimpleNamespace
    from ftmwpipeline import Pipeline
    from ftmwpipeline.cli import contract_commands
    from ftmwpipeline.cli.contract_commands import register_accessor

    # register_accessor reads file binding from MANIFEST.file_bound; a probe is
    # not a manifest accessor, so give the plumbing a stand-in.
    contract_commands.MANIFEST = SimpleNamespace(file_bound={"probe": True})

    def accessor(path):
        Pipeline.open(path)
        return {"opened": True}

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="verb")
    register_accessor(sub, "probe", accessor, help="probe", schema="ftmw/probe@1")
    args = parser.parse_args(sys.argv[1:])
    sys.exit(args.func(args))
    """)


def _probe(*argv: str):
    return subprocess.run(
        [sys.executable, "-c", _PROBE, *argv],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )


def _error_json(stderr: str) -> dict:
    """The error dict: the one JSON object line on stderr."""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    assert lines, "nothing on stderr"
    return json.loads(lines[-1])


@pytest.fixture
def junk_file(tmp_path) -> Path:
    path = tmp_path / "junk.ftmw"
    path.write_bytes(b"this is definitely not an HDF5 file\n" * 50)
    return path


@pytest.fixture
def tiny_file(tmp_path) -> Path:
    """A real, minimal Stage-0 ``.ftmw`` (imported from a synthetic HDF5)."""
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(6325) * 0.01))
    out = tmp_path / "tiny.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


# ---- read capabilities ----------------------------------------------------


def test_read_capabilities_is_valid_json_with_exit_zero():
    r = _cli("read", "capabilities")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["schema"] == "ftmw/capabilities@1"
    assert payload["contract_version"] == ftmwpipeline.CONTRACT_VERSION
    assert payload == capabilities()


def test_read_capabilities_lists_only_implemented_codes():
    payload = json.loads(_cli("read", "capabilities").stdout)
    assert payload["codes"] == list(MANIFEST.codes)
    # A code is listed once its class exists (spec: reserved codes such as
    # algorithm_failed are documented as such); cancelled has one since 5.1.
    assert "cancelled" in payload["codes"]
    assert "callback_failed" in payload["codes"]
    assert "write_conflict" in payload["codes"]
    assert "curation_conflict" in payload["codes"]
    assert "internal_error" in payload["codes"]


# ---- file-bound accessor (probe), JSON on stderr --------------------------


def test_file_bound_accessor_on_missing_path_is_not_found_exit_1(tmp_path):
    missing = tmp_path / "absent.ftmw"
    r = _probe("probe", str(missing))
    assert r.returncode == 1
    assert r.stdout == ""
    err = _error_json(r.stderr)
    assert err["schema"] == "ftmw/error@1"
    assert err["code"] == "not_found"
    assert err["kind"] == "file" and err["ids"] == [str(missing)]
    assert "Traceback" not in r.stderr


def test_file_bound_accessor_on_junk_file_is_file_corrupt_exit_2(junk_file):
    r = _probe("probe", str(junk_file))
    assert r.returncode == 2
    assert r.stdout == ""
    err = _error_json(r.stderr)
    assert err["schema"] == "ftmw/error@1" and err["code"] == "file_corrupt"
    assert "Traceback" not in r.stderr


def test_file_bound_accessor_success_envelope(tiny_file):
    r = _probe("probe", str(tiny_file))
    assert r.returncode == 0
    assert json.loads(r.stdout) == {"schema": "ftmw/probe@1", "opened": True}


# ---- read meta / table / list: shared exit-code mapping, text unchanged ---


def test_read_meta_missing_path_exits_1_with_unchanged_text(tmp_path):
    missing = tmp_path / "absent.ftmw"
    r = _cli("read", "meta", str(missing))
    assert r.returncode == 1
    assert r.stdout == (
        f"Error: Pipeline file not found: {missing}\n\n"
        f"To create a new pipeline:\n"
        f"  ftmwpipeline data import {missing} path/to/data/\n"
    )
    assert "Traceback" not in r.stderr


def test_read_meta_junk_file_is_file_corrupt_exit_2(junk_file):
    r = _cli("read", "meta", str(junk_file))
    assert r.returncode == 2
    assert r.stdout.startswith(f"Error: Failed to open pipeline file {junk_file}")
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("verb", ["table", "list"])
def test_read_table_and_list_share_the_mapping(junk_file, tmp_path, verb):
    extra = [READ_TABLES[0]] if verb == "table" else []
    r = _cli("read", verb, str(junk_file), *extra)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    missing = str(tmp_path / "absent.ftmw")
    assert _cli("read", verb, missing, *extra).returncode == 1


def test_read_meta_human_output_is_unchanged_on_success(tiny_file):
    r = _cli("read", "meta", str(tiny_file))
    assert r.returncode == 0
    expected = format_metadata_impl(read_metadata_impl(tiny_file), "csv")
    assert r.stdout == expected and r.stdout.strip()
