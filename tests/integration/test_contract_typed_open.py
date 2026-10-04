"""Every verb that takes a ``.ftmw`` types an unopenable file.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Errors and §Status and settings (last
bullet). A path that does not exist is ``not_found`` (CLI exit 1); a path that
exists but is not HDF5 is ``file_corrupt`` (CLI exit 2) -- never a raw
``OSError``, a ``RuntimeError("X failed: Unable to open ...")`` re-wrap, or
exit 1.

Mutation guarding all tests here: remove the ``@requires_pipeline_file``
decorator (or the ``require_pipeline_file`` call) from the verb's impl; the
open then fails inside the impl with an untyped error and the exit code or
exception type below changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as api
from ftmwpipeline import Pipeline
from ftmwpipeline.cli import main
from ftmwpipeline.file_manager import (
    PipelineCorruptionError,
    PipelineFileNotFoundError,
    require_pipeline_file,
)

pytestmark = [pytest.mark.integration]

#: Leaf CLI verbs that take a ``.ftmw`` as their first positional argument
#: (path inserted after the verb words), with any further required arguments.
_VERBS = [
    ["noise", "run"],
    ["noise", "show"],
    ["tau", "run"],
    ["tau", "recommend"],
    ["tau", "show"],
    ["timebase", "run"],
    ["timebase", "show"],
    ["timebase", "state"],
    ["peaks", "run"],
    ["peaks", "show"],
    ["windows", "run"],
    ["windows", "show"],
    ["fit", "run"],
    ["fit", "show"],
    ["fit", "check"],
    ["review", "log"],
    ["review", "rank", "--by", "chi2r"],
    ["review", "preview", "ACTIONS"],
    ["review", "apply", "ACTIONS"],
    ["review", "undo", "--id", "1"],
    ["review", "snap-tolerance"],
    ["review", "show"],
    ["review", "accept", "--window", "1"],
    ["review", "edit"],
    ["review", "create", "--at", "1"],
    ["review", "acknowledge-environment"],
    ["report", "table"],
    ["report", "run"],
    ["report", "diff"],
    ["scan", "all"],
    ["settings", "show"],
    ["settings", "set", "stage3.min_snr", "1"],
    ["settings", "unset", "stage3.min_snr"],
]


def _argv(verb: list, path: Path, actions: Path) -> list:
    argv = (
        verb[:2]
        + [str(path)]
        + [str(actions) if a == "ACTIONS" else a for a in verb[2:]]
    )
    return argv


@pytest.fixture
def junk(tmp_path) -> Path:
    p = tmp_path / "junk.ftmw"
    p.write_bytes(b"this is definitely not an HDF5 file\n" * 50)
    return p


@pytest.fixture
def actions(tmp_path) -> Path:
    p = tmp_path / "actions.json"
    p.write_text("[]")
    return p


@pytest.fixture
def tiny_file(tmp_path) -> Path:
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(6325) * 0.01))
    out = tmp_path / "tiny.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


@pytest.mark.parametrize("verb", _VERBS, ids=lambda v: " ".join(v[:2]))
def test_cli_verb_on_junk_file_exits_2_with_typed_message(verb, junk, actions, capsys):
    # Mutation: drop the guard from this verb's impl -> exit 1 / untyped text.
    rc = main(_argv(verb, junk, actions))
    cap = capsys.readouterr()
    assert rc == 2
    assert f"Failed to open pipeline file {junk}" in cap.out + cap.err


@pytest.mark.parametrize("verb", _VERBS, ids=lambda v: " ".join(v[:2]))
def test_cli_verb_on_missing_path_is_not_found_exit_1(verb, tmp_path, actions, capsys):
    # Mutation: drop the guard -> an untyped message, or a different failure.
    missing = tmp_path / "absent.ftmw"
    rc = main(_argv(verb, missing, actions))
    cap = capsys.readouterr()
    assert rc == 1
    assert "Pipeline file not found" in cap.out + cap.err


def test_cli_junk_file_under_json_reports_file_corrupt_dict(junk, capsys):
    # Mutation: drop the guard from report table -> not an ftmw/error@1 dict.
    rc = main(["report", "table", str(junk), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 2
    payload = json.loads(cap.err.strip().splitlines()[-1])
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "file_corrupt"


@pytest.mark.parametrize("name", ["detect_peaks", "assign_windows", "fit_peaks"])
def test_api_stage_on_junk_and_missing(name, junk, tmp_path):
    # Mutation: drop the guard -> RuntimeError("... failed: Unable to open").
    fn = getattr(api, name)
    with pytest.raises(PipelineCorruptionError) as ei:
        fn(junk)
    assert ei.value.code == "file_corrupt"
    with pytest.raises(PipelineFileNotFoundError) as ei2:
        fn(tmp_path / "absent.ftmw")
    assert ei2.value.code == "not_found"


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.detect_peaks(),
        lambda p: p.assign_windows(),
        lambda p: p.fit_peaks(),
        lambda p: p.review_log(),
        lambda p: p.report_table(),
    ],
    ids=["detect_peaks", "assign_windows", "fit_peaks", "review_log", "report_table"],
)
def test_pipeline_wrappers_let_the_typed_error_propagate(call, tiny_file):
    # Mutation: drop the guard, or re-wrap in the Pipeline method's
    # ``except Exception`` -> RuntimeError / OSError instead of the typed error.
    p = Pipeline.open(tiny_file)
    tiny_file.write_bytes(b"not hdf5 any more\n" * 100)
    with pytest.raises(PipelineCorruptionError):
        call(p)


def test_guard_passes_a_valid_file_and_returns_its_path(tiny_file):
    # Mutation: make the guard refuse source-metadata-only files -> raises.
    assert require_pipeline_file(tiny_file) == tiny_file


def test_guard_does_not_type_a_transient_permission_error(tmp_path, monkeypatch):
    # Mutation: report every OSError as file_corrupt -> a retryable lock
    # failure would be misreported.
    p = tmp_path / "locked.ftmw"
    p.write_bytes(b"x")

    def boom(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(h5py, "File", boom)
    with pytest.raises(PermissionError):
        require_pipeline_file(p)
