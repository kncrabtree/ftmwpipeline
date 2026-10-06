"""
Stage 6 write refusals, identical on the CLI, the Pipeline class and the
functional API (and through a ``ReviewSession``).

Two kinds of write are refused before anything is resolved, fitted or written:

* a bare ``review edit`` -- no ``add`` and no ``remove`` -- which would refit a
  window and record no decision: ``bad_setting`` (``path`` ``"add"``), with or
  without a window;
* every Stage 6 write of a file the replay engine cannot curate (refuse and
  flag): ``curation_conflict`` ``predates_peak_identity`` when a fitted peak
  carries no ``peak_uid``, ``predates_replay_engine`` when the review or the
  undo baseline was written by a pre-engine build. A review from a newer
  engine is ``file_incompatible``. Reads still work and flag the file
  (``Stage6Review.refit_required``); ``fit run`` makes it writable again.

The refused files are built from a curated engine file by stripping exactly
what a pre-engine build would not have written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal import stage6_impl
from ftmwpipeline._internal.stage6_impl import LINEAGE_ID_ATTR, STAGE5_BASELINE_GROUP
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.absent import Absent
from ftmwpipeline.core.data_structures import ENGINE_VERSION
from ftmwpipeline.file_manager import (
    BadSettingError,
    CurationConflictError,
    PipelineCompatibilityError,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_and_clear_add(path: Path) -> Tuple[int, float]:
    """A fitted window and an in-window frequency far from its peaks."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if w.fitted_peaks and w.window is not None)
    assert wf.window is not None and wf.window_id is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    clear = max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min(abs(x - q) for q in peaks),
    )
    return int(wf.window_id), clear


@pytest.fixture(scope="module")
def curated(baseline_2638_stage5_small, tmp_path_factory) -> Path:
    """A file the engine curated: one add, so a review and a baseline exist."""
    path = tmp_path_factory.mktemp("stage6_refusals") / "curated.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid, clear = _window_and_clear_add(path)
    ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    return path


def _strip_engine_version(h5f: h5py.File) -> None:
    del h5f["stage6_review"].attrs["engine_version"]
    # A pre-engine build recorded no serials either.
    grp = h5f["stage6_review"]["decision_log"]
    rows = json.loads(str(grp.attrs["data"]))
    for row in rows:
        row.pop("serial", None)
    grp.attrs["data"] = json.dumps(rows)


def _strip_lineage(h5f: h5py.File) -> None:
    del h5f[STAGE5_BASELINE_GROUP].attrs[LINEAGE_ID_ATTR]


def _null_uids(h5f: h5py.File) -> None:
    col = h5f["stage5_fitting"]["peaks"]["peak_uid"]
    col[...] = np.full(col.shape, -1, dtype=col.dtype)


#: variant -> (the file surgery, the refusal reason)
VARIANTS: Dict[str, Tuple[List[Callable[[h5py.File], None]], str]] = {
    "engine_version": ([_strip_engine_version], "predates_replay_engine"),
    "lineage_id": ([_strip_lineage], "predates_replay_engine"),
    "peak_uid": ([_null_uids], "predates_peak_identity"),
    # Both hold: peak identity wins.
    "peak_uid_and_engine": (
        [_null_uids, _strip_engine_version],
        "predates_peak_identity",
    ),
}


def _variant(curated: Path, tmp_path: Path, name: str) -> Tuple[Path, str]:
    path = tmp_path / f"{name}.ftmw"
    shutil.copy(curated, path)
    surgery, reason = VARIANTS[name]
    with h5py.File(str(path), "a") as h5f:
        for fn in surgery:
            fn(h5f)
    return path, reason


@pytest.fixture
def no_fit(monkeypatch):
    """Fail the test if a window is refit: a refusal comes before any fit."""

    def refit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a refused write fitted a window")

    monkeypatch.setattr(stage6_impl, "refit_window_core", refit)


def _cli_error(argv: List[str], capsys: Any) -> Tuple[int, Dict[str, Any]]:
    capsys.readouterr()
    rc = main(argv + ["--json"])
    err = capsys.readouterr().err.strip().splitlines()
    return rc, json.loads(err[-1])


# ---------------------------------------------------------------------------
# The bare edit
# ---------------------------------------------------------------------------


def test_bare_edit_is_refused_on_every_interface(curated, tmp_path, capsys, no_fit):
    path = tmp_path / "bare.ftmw"
    shutil.copy(curated, path)
    wid, _ = _window_and_clear_add(path)
    before = _sha(path)

    for window in (None, wid):
        with pytest.raises(BadSettingError) as exc:
            ftmw.review_edit(str(path), window)
        assert exc.value.path == "add"
        with pytest.raises(BadSettingError) as exc:
            Pipeline.open(path).review_edit(window)
        assert exc.value.path == "add"
        with Pipeline.open(path).review_session() as session:
            with pytest.raises(BadSettingError) as exc:
                session.review_edit(window)
        assert exc.value.path == "add"
        argv = ["review", "edit", str(path)] + (
            [] if window is None else ["--window", str(window)]
        )
        rc, payload = _cli_error(argv, capsys)
        assert rc == 1
        assert (payload["code"], payload["path"]) == ("bad_setting", "add")

    assert _sha(path) == before


# ---------------------------------------------------------------------------
# Refuse and flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_write_is_refused_on_every_interface(
    curated, tmp_path, capsys, no_fit, variant
):
    path, reason = _variant(curated, tmp_path, variant)
    wid, clear = _window_and_clear_add(path)
    csv = tmp_path / "accept.csv"
    csv.write_text(f"accept,{wid},,\n")
    before = _sha(path)

    api_writes: List[Callable[[], Any]] = [
        lambda: ftmw.review_edit(str(path), wid, add=[clear], frame="raw"),
        lambda: ftmw.review_accept(str(path), wid),
        lambda: ftmw.review_create(str(path), 30000.0, frame="raw"),
        lambda: ftmw.review_apply(str(path), csv),
        lambda: ftmw.review_apply(str(path), csv, dry_run=True),
        lambda: ftmw.review_undo(str(path), [0]),
        lambda: ftmw.review_run(str(path)),
    ]
    pipe = Pipeline.open(path)
    pipe_writes: List[Callable[[], Any]] = [
        lambda: pipe.review_edit(wid, add=[clear], frame="raw"),
        lambda: pipe.review_accept(wid),
        lambda: pipe.review_create(30000.0, frame="raw"),
        lambda: pipe.review_apply(csv),
        lambda: pipe.review_undo([0]),
        lambda: pipe.review_run(),
    ]
    for write in api_writes + pipe_writes:
        with pytest.raises(CurationConflictError) as exc:
            write()
        assert (exc.value.reason, exc.value.ids) == (reason, [])

    with Pipeline.open(path).review_session() as session:
        for session_write in (
            lambda: session.review_edit(wid, add=[clear], frame="raw"),
            lambda: session.review_accept(wid),
            lambda: session.review_create(30000.0, frame="raw"),
            lambda: session.review_undo([0]),
            lambda: session.review_apply(csv),
        ):
            with pytest.raises(CurationConflictError) as exc:
                session_write()
            assert exc.value.reason == reason

    for argv in (
        ["review", "edit", str(path), "--window", str(wid), "--add", str(clear)],
        ["review", "accept", str(path), "--window", str(wid)],
        ["review", "create", str(path), "--at", "30000.0"],
        ["review", "apply", str(path), str(csv)],
        ["review", "undo", str(path), "--id", "0"],
        ["review", "run", str(path)],
    ):
        rc, payload = _cli_error(argv, capsys)
        assert rc == 1, argv
        assert (payload["code"], payload["reason"]) == ("curation_conflict", reason)

    assert _sha(path) == before


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_reads_flag_the_file_and_write_nothing(curated, tmp_path, capsys, variant):
    path, reason = _variant(curated, tmp_path, variant)
    before = _sha(path)

    assert ftmw.get_review_status(str(path)).refit_required == reason
    assert Pipeline.open(path).review_status().refit_required == reason
    log = ftmw.review_log(str(path))
    assert [e.kind for e in log] == ["add"]
    if variant.endswith("engine") or variant == "engine_version":
        # Rows a pre-engine build recorded carry no serial; nothing is mapped.
        assert log[0].serial is Absent.NOT_RUN
    capsys.readouterr()
    assert main(["review", "log", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refit_required"] == reason
    (entry,) = payload["entries"]
    assert main(["review", "show", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["refit_required"] == reason
    if log[0].serial is Absent.NOT_RUN:
        assert (entry["serial"], entry["serial_absent"]) == (None, "not_run")
    else:
        assert entry["serial"] == log[0].serial
    assert main(["review", "log", str(path)]) == 0
    assert main(["review", "show", str(path)]) == 0
    assert "fit run" in capsys.readouterr().err

    assert _sha(path) == before


def test_an_admitted_file_is_not_flagged(curated, capsys):
    assert ftmw.get_review_status(str(curated)).refit_required is None
    capsys.readouterr()
    assert main(["review", "log", str(curated), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["refit_required"] is None


def test_a_newer_engine_is_incompatible(curated, tmp_path, capsys):
    path = tmp_path / "newer.ftmw"
    shutil.copy(curated, path)
    with h5py.File(str(path), "a") as h5f:
        h5f["stage6_review"].attrs["engine_version"] = ENGINE_VERSION + 1
    wid, clear = _window_and_clear_add(path)
    before = _sha(path)

    # Flagged on read like a refused file, with the error a write raises: a
    # client reading None would conclude it can write.
    assert ftmw.get_review_status(str(path)).refit_required == "file_incompatible"
    assert Pipeline.open(path).review_status().refit_required == "file_incompatible"
    capsys.readouterr()
    assert main(["review", "show", str(path)]) == 0
    assert "upgrade ftmwpipeline" in capsys.readouterr().err
    with pytest.raises(PipelineCompatibilityError):
        ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    with pytest.raises(PipelineCompatibilityError):
        Pipeline.open(path).review_accept(wid)
    rc, payload = _cli_error(["review", "run", str(path)], capsys)
    assert payload["code"] == "file_incompatible" and rc != 0
    assert _sha(path) == before


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fit_run_makes_a_refused_file_writable(curated, tmp_path, variant):
    """``fit run`` writes a fit with peak identity and drops the review and
    the baseline: the next write starts a new lineage, serials from 0."""
    path, _ = _variant(curated, tmp_path, variant)
    ftmw.fit_peaks(str(path))
    assert ftmw.get_review_status(str(path)).refit_required is None

    wid, clear = _window_and_clear_add(path)
    ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    assert [(e.kind, e.serial) for e in ftmw.review_log(str(path))] == [("add", 0)]


def test_a_uidless_fit_is_refused_before_any_curation_exists(
    baseline_2638_stage5_small, tmp_path, capsys, no_fit
):
    """No review and no baseline yet: the fit alone makes the file unwritable,
    and a refused first write snapshots nothing."""
    path = tmp_path / "uidless.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    with h5py.File(str(path), "a") as h5f:
        _null_uids(h5f)
    wid, clear = _window_and_clear_add(path)
    before = _sha(path)

    assert ftmw.get_review_status(str(path)).refit_required == "predates_peak_identity"
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    assert (exc.value.reason, exc.value.ids) == ("predates_peak_identity", [])
    rc, payload = _cli_error(["review", "run", str(path)], capsys)
    assert (rc, payload["reason"]) == (1, "predates_peak_identity")
    assert _sha(path) == before
    with h5py.File(str(path), "r") as h5f:
        assert STAGE5_BASELINE_GROUP not in h5f and "stage6_review" not in h5f


def test_a_review_run_only_pre_engine_file_is_refused(
    baseline_2638_stage5_small, tmp_path, no_fit
):
    """A review with statuses and no decisions (only ``review run`` ever ran)
    is pre-engine like any other: no exemption."""
    path = tmp_path / "run_only.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    ftmw.review_run(str(path))
    assert ftmw.get_review_status(str(path)).refit_required is None
    with h5py.File(str(path), "a") as h5f:
        del h5f["stage6_review"].attrs["engine_version"]
    wid, clear = _window_and_clear_add(path)
    before = _sha(path)

    assert ftmw.review_log(str(path)) == []
    assert ftmw.get_review_status(str(path)).refit_required == "predates_replay_engine"
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    assert exc.value.reason == "predates_replay_engine"
    with pytest.raises(CurationConflictError):
        ftmw.review_run(str(path))
    assert _sha(path) == before


@pytest.mark.parametrize("variant", ["engine_version", "lineage_id"])
def test_a_staged_preview_cannot_be_persisted_on_a_refused_file(
    curated, tmp_path, variant
):
    """A preview is a read and works; the apply that would persist it is a
    write and is refused, writing nothing."""
    path, reason = _variant(curated, tmp_path, variant)
    wid, _ = _window_and_clear_add(path)
    csv = tmp_path / "accept.csv"
    csv.write_text(f"accept,{wid},,\n")
    before = _sha(path)

    with Pipeline.open(path).review_session() as session:
        session.review_preview(csv)
        assert _sha(path) == before
        with pytest.raises(CurationConflictError) as exc:
            session.review_apply(csv)
        assert exc.value.reason == reason
    assert _sha(path) == before


# ---------------------------------------------------------------------------
# uid-addressed resolution: every refusal before the first fit
# ---------------------------------------------------------------------------


def _fits(path: Path) -> Dict[int, Any]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(w.window_id): w for w in sf.window_fits if w.window_id is not None}


def _clear_in(path: Path, wid: int) -> float:
    wf = _fits(path)[wid]
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min((abs(x - q) for q in peaks), default=1e9),
    )


def _edit_on(interface: str, path: Path, capsys: Any, wid: int, **kw: Any) -> Any:
    """``review edit`` through *interface*; the CurationConflictError (or the
    CLI's payload) it is refused with."""
    add = [str(f) for f in kw.get("add", [])]
    remove = [str(f) for f in kw.get("remove", [])]
    if interface == "cli":
        argv = ["review", "edit", str(path), "--window", str(wid), "--frame", "raw"]
        argv += [a for f in add for a in ("--add", f)]
        argv += [a for f in remove for a in ("--remove", f)]
        rc, payload = _cli_error(argv, capsys)
        assert rc == 1
        return payload["reason"], payload["ids"]
    with pytest.raises(CurationConflictError) as exc:
        if interface == "api":
            ftmw.review_edit(str(path), wid, add=add, remove=remove, frame="raw")
        elif interface == "pipeline":
            Pipeline.open(path).review_edit(wid, add=add, remove=remove, frame="raw")
        else:
            with Pipeline.open(path).review_session() as session:
                session.review_edit(wid, add=add, remove=remove, frame="raw")
    return exc.value.reason, exc.value.ids


def _undo_on(interface: str, path: Path, capsys: Any, ids: List[int]) -> Any:
    if interface == "cli":
        argv = ["review", "undo", str(path)] + [
            a for i in ids for a in ("--id", str(i))
        ]
        rc, payload = _cli_error(argv, capsys)
        assert rc == 1
        return payload["reason"], payload["ids"]
    with pytest.raises(CurationConflictError) as exc:
        if interface == "api":
            ftmw.review_undo(str(path), ids)
        elif interface == "pipeline":
            Pipeline.open(path).review_undo(ids)
        else:
            with Pipeline.open(path).review_session() as session:
                session.review_undo(ids)
    return exc.value.reason, exc.value.ids


INTERFACES = ["api", "pipeline", "session", "cli"]


@pytest.mark.parametrize("interface", INTERFACES)
def test_a_seed_off_the_named_window_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, no_fit, interface
):
    path = tmp_path / "outside.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wa, wb = sorted(w for w, wf in _fits(path).items() if wf.fitted_peaks)[:2]
    before = _sha(path)
    reason, ids = _edit_on(interface, path, capsys, wa, add=[_clear_in(path, wb)])
    assert (reason, ids) == ("target_outside_window", [wa])
    assert _sha(path) == before


@pytest.mark.parametrize("interface", INTERFACES)
def test_a_birth_on_a_held_uid_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, no_fit, interface
):
    path = tmp_path / "clash.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid, clear = _window_and_clear_add(path)
    before = _sha(path)
    reason, _ = _edit_on(interface, path, capsys, wid, add=[clear, clear])
    assert reason == "line_already_fitted"
    assert _sha(path) == before


@pytest.mark.parametrize("interface", INTERFACES)
def test_a_cascade_into_an_unavailable_window_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, monkeypatch, interface
):
    """The cascade's ``fit_plan_unavailable`` is structural: an edit whose
    cascade reaches a window whose fitted geometry the file does not hold is
    refused before the edited window is refit, not after. The small fixture
    has no cascade edge, so one is given to it, into a window marked
    unavailable (what a merged fit that predates the stored plan holds)."""
    import dataclasses

    path = tmp_path / "unavailable.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    src, dep = sorted(w for w, wf in _fits(path).items() if wf.fitted_peaks)[:2]
    real = stage6_impl._build_shared_fit_ctx

    def blocked(*args: Any, **kwargs: Any) -> Any:
        shared = real(*args, **kwargs)
        sources = dict(shared.base_cascade_sources)
        sources[dep] = tuple(sources.get(dep, ())) + (src,)
        return dataclasses.replace(
            shared,
            base_cascade_sources=sources,
            unavailable_window_ids=frozenset({dep}),
        )

    monkeypatch.setattr(stage6_impl, "_build_shared_fit_ctx", blocked)
    monkeypatch.setattr(stage6_impl, "refit_window_core", _forbidden_fit)
    before = _sha(path)
    reason, ids = _edit_on(interface, path, capsys, src, add=[_clear_in(path, src)])
    assert (reason, ids) == ("fit_plan_unavailable", [dep])
    assert _sha(path) == before


def _forbidden_fit(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a refused write fitted a window")


@pytest.mark.parametrize("interface", INTERFACES)
def test_an_undo_orphaning_a_birth_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, monkeypatch, interface
):
    """``orphans_peak``: undoing an add whose peak a kept remove names."""
    path = tmp_path / "orphan.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid, clear = _window_and_clear_add(path)
    ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    (add,) = ftmw.review_log(str(path))
    ftmw.review_edit(str(path), wid, remove=[f"uid:{add.born_uids[0]}"])
    remove = ftmw.review_log(str(path))[-1]
    monkeypatch.setattr(stage6_impl, "refit_window_core", _forbidden_fit)
    before = _sha(path)
    reason, ids = _undo_on(interface, path, capsys, [add.serial])
    assert (reason, ids) == ("orphans_peak", [remove.serial])
    assert _sha(path) == before


@pytest.mark.parametrize("interface", INTERFACES)
def test_an_edit_of_an_unavailable_window_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, monkeypatch, interface
):
    import dataclasses

    path = tmp_path / "unavailable_edit.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid = sorted(w for w, wf in _fits(path).items() if wf.fitted_peaks)[0]
    real = stage6_impl._build_shared_fit_ctx

    def blocked(*args: Any, **kwargs: Any) -> Any:
        shared = real(*args, **kwargs)
        return dataclasses.replace(shared, unavailable_window_ids=frozenset({wid}))

    monkeypatch.setattr(stage6_impl, "_build_shared_fit_ctx", blocked)
    monkeypatch.setattr(stage6_impl, "refit_window_core", _forbidden_fit)
    before = _sha(path)
    reason, ids = _edit_on(interface, path, capsys, wid, add=[_clear_in(path, wid)])
    assert (reason, ids) == ("fit_plan_unavailable", [wid])
    assert _sha(path) == before


@pytest.mark.parametrize("interface", INTERFACES)
def test_a_target_the_window_holds_twice_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, monkeypatch, interface
):
    """``ambiguous_peak``: a window that holds one uid twice (forged here)
    cannot say which peak a request naming it means."""
    path = tmp_path / "twice.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid = next(w for w, wf in sorted(_fits(path).items()) if len(wf.fitted_peaks) > 1)
    p0, p1 = sorted(_fits(path)[wid].fitted_peaks, key=lambda p: p.frequency_mhz)[:2]
    with h5py.File(str(path), "a") as h5f:
        col = h5f["stage5_fitting"]["peaks"]["peak_uid"]
        uids = col[...]
        uids[uids == int(p1.peak_uid)] = int(p0.peak_uid)
        col[...] = uids
    monkeypatch.setattr(stage6_impl, "refit_window_core", _forbidden_fit)
    before = _sha(path)
    reason, ids = _edit_on(interface, path, capsys, wid, remove=[f"uid:{p0.peak_uid}"])
    assert (reason, ids) == ("ambiguous_peak", [int(p0.peak_uid)])
    assert _sha(path) == before


@pytest.mark.parametrize("interface", INTERFACES)
def test_an_undo_replaying_a_gone_target_is_refused_before_any_fit(
    baseline_2638_stage5_small, tmp_path, capsys, monkeypatch, interface
):
    """``replay_diverged``: a row whose target its window does not hold at
    its place in the log (forged here) refuses the replay."""
    path = tmp_path / "gone.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wa, wb = sorted(w for w, wf in _fits(path).items() if wf.fitted_peaks)[:2]
    victim = int(_fits(path)[wa].fitted_peaks[0].peak_uid)
    ftmw.review_edit(str(path), wa, remove=[f"uid:{victim}"])
    ftmw.review_accept(str(path), wb)
    with h5py.File(str(path), "a") as h5f:
        grp = h5f["stage6_review"]["decision_log"]
        rows = json.loads(str(grp.attrs["data"]))
        rows[0]["targets"] = [victim + 7]
        grp.attrs["data"] = json.dumps(rows)
    last = ftmw.review_log(str(path))[-1].serial
    monkeypatch.setattr(stage6_impl, "refit_window_core", _forbidden_fit)
    before = _sha(path)
    reason, ids = _undo_on(interface, path, capsys, [last])
    assert (reason, ids) == ("replay_diverged", [ftmw.review_log(str(path))[0].serial])
    assert _sha(path) == before
