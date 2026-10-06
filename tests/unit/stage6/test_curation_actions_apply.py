"""Curation as data: the same batch given as a file and as ``actions=`` gives
equal results, decision logs and files, on every interface.

Each test names the mutation it catches. Fixtures: the shared post-fit small
build (``stage5_small_source``); every file is a tmp_path copy.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import BadSettingError, CurationAction
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_preview
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.pipeline import Pipeline

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


def _fitted(path: Path) -> Dict[int, List[float]]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): sorted(
            round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks
        )
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _log(path: Path) -> List[Tuple[Any, ...]]:
    return [
        (e.order_index, e.window_id, e.frequency_mhz, e.kind, e.provenance)
        for e in load_stage6_review_from_file(str(path)).decision_log
    ]


def _plan(path: Path) -> List[CurationAction]:
    by_w = _fitted(path)
    wid, freqs = next((w, f) for w, f in by_w.items() if f)
    f = freqs[0]
    return [
        CurationAction("remove", freq_mhz=f),  # window derived
        CurationAction("add", window_id=wid, freq_mhz=f + 0.4),
        CurationAction("accept", window_id=wid),
    ]


def _write_rows(tmp_path: Path, actions: List[CurationAction]) -> str:
    p = tmp_path / "plan.csv"
    p.write_text("\n".join(a.to_row() for a in actions) + "\n")
    return str(p)


def _copy(src: Path, tmp_path: Path, name: str) -> Path:
    dst = tmp_path / f"{name}.ftmw"
    shutil.copy(src, dst)
    return dst


def test_dry_run_file_equals_actions(stage5_small_source, tmp_path):
    # Mutation: the actions path builds different ops (window sentinel, order)
    # than the file parser -> the resolved plans differ.
    fp = _copy(stage5_small_source, tmp_path, "a")
    acts = _plan(fp)
    csv = _write_rows(tmp_path, acts)
    r_file = ftmw.review_apply(str(fp), csv, dry_run=True)
    r_act = ftmw.review_apply(str(fp), actions=acts, dry_run=True)
    r_dict = ftmw.review_apply(
        str(fp), actions=[a.to_dict() for a in acts], dry_run=True
    )
    assert r_file == r_act == r_dict
    assert r_file.plan  # not vacuous


def test_live_apply_equal_on_api_pipeline_and_cli(
    stage5_small_source, tmp_path, capsys
):
    # Mutation: any one interface drops/reorders actions or skips the decision
    # log -> fitted state or log diverges from the file apply.
    paths = {
        k: _copy(stage5_small_source, tmp_path, k)
        for k in ("file", "api", "pipe", "cli")
    }
    acts = _plan(paths["file"])
    csv = _write_rows(tmp_path, acts)
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps([a.to_dict() for a in acts]))

    r_file = ftmw.review_apply(str(paths["file"]), csv)
    r_api = ftmw.review_apply(str(paths["api"]), actions=acts)
    r_pipe = Pipeline.open(paths["pipe"]).review_apply(actions=acts)
    rc = cmd_review_apply(
        argparse.Namespace(
            file_path=str(paths["cli"]),
            curation_file=None,
            actions=str(plan_json),
            dry_run=False,
        )
    )
    assert rc == 0
    assert r_file == r_api == r_pipe
    assert r_file.applied == len(r_file.plan) > 0

    ref_fit, ref_log = _fitted(paths["file"]), _log(paths["file"])
    assert ref_log
    for k in ("api", "pipe", "cli"):
        assert _fitted(paths[k]) == ref_fit, k
        assert _log(paths[k]) == ref_log, k


def test_cli_actions_from_stdin(stage5_small_source, tmp_path, monkeypatch):
    # Mutation: '-' not mapped to stdin.
    ref = _copy(stage5_small_source, tmp_path, "ref")
    got = _copy(stage5_small_source, tmp_path, "got")
    acts = _plan(ref)
    ftmw.review_apply(str(ref), _write_rows(tmp_path, acts))
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps([a.to_dict() for a in acts]))
    )
    rc = cmd_review_apply(
        argparse.Namespace(
            file_path=str(got), curation_file=None, actions="-", dry_run=False
        )
    )
    assert rc == 0
    assert _fitted(got) == _fitted(ref) and _log(got) == _log(ref)


def test_preview_file_equals_actions_on_api_pipeline_session_cli(
    stage5_small_source, tmp_path, capsys
):
    # Mutation: preview ignores actions= on one interface; or the session's
    # staged-preview reuse mis-keys on the action tuple (apply after preview
    # must equal a plain file apply).
    fp = _copy(stage5_small_source, tmp_path, "p")
    acts = _plan(fp)
    csv = _write_rows(tmp_path, acts)
    ref = ftmw.review_preview(str(fp), csv)
    assert ftmw.review_preview(str(fp), actions=acts) == ref
    assert Pipeline.open(fp).review_preview(actions=acts) == ref

    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps([a.to_dict() for a in acts]))
    assert (
        cmd_review_preview(
            argparse.Namespace(
                file_path=str(fp), curation_file=None, actions=str(plan_json)
            )
        )
        == 0
    )

    via_file = _copy(stage5_small_source, tmp_path, "via_file")
    via_sess = _copy(stage5_small_source, tmp_path, "via_sess")
    ftmw.review_apply(str(via_file), csv)
    with Pipeline.open(via_sess).review_session() as sess:
        assert sess.review_preview(actions=acts) == ref
        sess.review_apply(actions=acts)
    assert _fitted(via_sess) == _fitted(via_file)
    assert _log(via_sess) == _log(via_file)


def test_preview_does_not_write(stage5_small_source, tmp_path):
    # Mutation: preview with actions persists the batch.
    fp = _copy(stage5_small_source, tmp_path, "np")
    before = (_fitted(fp), _log(fp))
    ftmw.review_preview(str(fp), actions=_plan(fp))
    assert (_fitted(fp), _log(fp)) == before


def test_exactly_one_of_file_or_actions(stage5_small_source, tmp_path, capsys):
    # Mutation: both / neither tolerated on any interface.
    fp = _copy(stage5_small_source, tmp_path, "one")
    acts = _plan(fp)
    csv = _write_rows(tmp_path, acts)
    for call in (
        lambda: ftmw.review_apply(str(fp)),
        lambda: ftmw.review_apply(str(fp), csv, actions=acts),
        lambda: ftmw.review_preview(str(fp)),
        lambda: ftmw.review_preview(str(fp), csv, actions=acts),
        lambda: Pipeline.open(fp).review_apply(),
        lambda: Pipeline.open(fp).review_apply(csv, actions=acts),
    ):
        with pytest.raises(BadSettingError) as ei:
            call()
        assert ei.value.path == "actions"
    plan_json = tmp_path / "plan.json"
    plan_json.write_text("[]")
    # Through the CLI entry point: the verb lets the typed refusal reach
    # main, which exits 1 and writes it to stderr.
    from ftmwpipeline.cli.main import main as cli_main

    capsys.readouterr()
    for argv in (
        ["review", "apply", str(fp)],
        ["review", "apply", str(fp), str(csv), "--actions", str(plan_json)],
    ):
        assert cli_main(argv) == 1
        assert "actions" in capsys.readouterr().err
    assert _log(fp) == []  # nothing applied


def test_cli_actions_json_must_be_an_array(stage5_small_source, tmp_path):
    # Mutation: a JSON object accepted as a batch.
    fp = _copy(stage5_small_source, tmp_path, "obj")
    bad = tmp_path / "bad.json"
    bad.write_text('{"action": "accept", "window_id": 1}')
    with pytest.raises(BadSettingError) as ei:
        cmd_review_apply(
            argparse.Namespace(
                file_path=str(fp), curation_file=None, actions=str(bad), dry_run=False
            )
        )
    assert ei.value.path == "actions"
    bad.write_text("not json")
    with pytest.raises(BadSettingError) as ei:
        cmd_review_apply(
            argparse.Namespace(
                file_path=str(fp), curation_file=None, actions=str(bad), dry_run=False
            )
        )
    assert ei.value.path == "actions"
