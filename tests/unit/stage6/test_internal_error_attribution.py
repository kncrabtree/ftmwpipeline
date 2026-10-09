"""An unexpected failure inside one curation action names that action.

When an untyped exception (not a typed refusal) escapes ``review_preview`` /
``review_apply`` while one action of a curation batch is being processed, it is
reported as ``internal_error`` (``ftmw/error@1``) carrying the action's request
positions (``action_indices``), so a client can mark that row failed and
resubmit the rest. The batch stays all-or-nothing: nothing is written.

The failure is forced by monkeypatching an internal step to raise for one
specific action, in each phase a batch processes an action in: the resolve
phase (before any fit), the symbolic pass over the new log, and the fit of the
action's rows in the write's replay.

Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import BadSettingError, InternalError, Pipeline
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.curation import CurationAction
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = pytest.mark.integration

#: What the patched step raises: a plain error no typed refusal wraps.
_BOOM = "synthetic failure for one action"


def _clear_freqs(lo: float, hi: float, peaks: List[float], n: int) -> List[float]:
    """*n* in-window frequencies well clear of *peaks* and of each other."""
    grid = [lo + (hi - lo) * t / 40 for t in range(4, 37)]
    chosen: List[float] = []
    for _ in range(n):
        taken = peaks + chosen
        chosen.append(max(grid, key=lambda x: min(abs(x - q) for q in taken)))
    return chosen


def _batch(path: Path) -> Tuple[int, int, List[Tuple[int, float]]]:
    """Two fitted windows A and B, and a batch of three adds: two on A (they
    coalesce into one edit, request positions 0 and 1) and one on B
    (position 2). Returns ``(A, B, rows)``."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    fitted = [w for w in sf.window_fits if w.fitted_peaks and w.window is not None]
    assert len(fitted) >= 2, "fixture must hold two fitted windows"
    rows: List[Tuple[int, float]] = []
    ids: List[int] = []
    for wf, n in zip(fitted[:2], (2, 1)):
        assert wf.window is not None and wf.window_id is not None
        lo, hi = sorted(float(v) for v in wf.window.freq_range)
        peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
        ids.append(int(wf.window_id))
        rows += [(int(wf.window_id), f) for f in _clear_freqs(lo, hi, peaks, n)]
    return ids[0], ids[1], rows


def _csv(tmp_path: Path, rows: List[Tuple[int, float]]) -> Path:
    cur = tmp_path / "batch.csv"
    cur.write_text("".join(f"add,{w},{f!r},\n" for w, f in rows))
    return cur


def _actions(rows: List[Tuple[int, float]]) -> List[CurationAction]:
    return [
        CurationAction(action="add", window_id=w, freq_mhz=f, frame="raw")
        for w, f in rows
    ]


def _copy(src: Path, tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copy(src, dst)
    return dst


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _fail_fit_on(monkeypatch: Any, window_id: int) -> None:
    """The fit phase: fitting a refit step on *window_id* raises."""
    real = s6._apply_refit_steps

    def boom(ctx: Any, steps: Any, **kw: Any) -> Any:
        if steps and int(steps[0].window_id) == window_id:
            raise TypeError(_BOOM)
        return real(ctx, steps, **kw)

    monkeypatch.setattr(s6, "_apply_refit_steps", boom)


def _fail_resolve_on(
    monkeypatch: Any, window_id: int, exc: Callable[[], Exception]
) -> None:
    """The resolve phase: resolving an action on *window_id* raises."""
    real = s6._resolve_action

    def boom(ctx: Any, state: Any, index: int, action: Any, **kw: Any) -> Any:
        if action.kind != "create" and int(action.window_id) == window_id:
            raise exc()
        return real(ctx, state, index, action, **kw)

    monkeypatch.setattr(s6, "_resolve_action", boom)


def _fail_walk_on(monkeypatch: Any, window_id: int) -> None:
    """The symbolic pass over the new log: replaying a row on *window_id*
    raises (the resolve phase never asks this question)."""
    real = s6._refuse_unavailable_fit_plan

    def boom(unavailable: Any, wids: Any, where: str) -> Any:
        wids = list(wids)
        if where.startswith("replaying decision") and window_id in wids:
            raise KeyError(_BOOM)
        return real(unavailable, wids, where)

    monkeypatch.setattr(s6, "_refuse_unavailable_fit_plan", boom)


def _assert_internal(exc: BaseException, indices: List[int], cause: type) -> None:
    assert isinstance(exc, InternalError)
    assert isinstance(exc, RuntimeError)
    assert exc.code == "internal_error"
    assert exc.action_indices == indices
    assert _BOOM in str(exc)
    # The 1-based tag of the action, as the typed per-action refusals carry.
    numbers = ", ".join(str(i + 1) for i in indices)
    assert f"curation action{'s' if len(indices) > 1 else ''} {numbers} (" in str(exc)
    assert isinstance(exc.__cause__, cause)
    d = exc.to_dict()
    assert d["schema"] == "ftmw/error@1"
    assert d["code"] == "internal_error"
    assert d["action_indices"] == indices
    assert "action_indices_absent" not in d
    assert d["message"] == str(exc)


# ---------------------------------------------------------------------------
# Fit phase
# ---------------------------------------------------------------------------


def test_a_fit_failure_on_a_coalesced_edit_names_both_rows(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    a, _b, rows = _batch(src)
    _fail_fit_on(monkeypatch, a)
    cur = _csv(tmp_path, rows)

    with pytest.raises(InternalError) as preview:
        ftmw.review_preview(str(src), str(cur), frame="raw")
    _assert_internal(preview.value, [0, 1], TypeError)

    path = _copy(src, tmp_path, "apply.ftmw")
    before = _md5(path)
    with pytest.raises(InternalError) as apply:
        ftmw.review_apply(str(path), str(cur), frame="raw")
    _assert_internal(apply.value, [0, 1], TypeError)
    assert apply.value.to_dict() == preview.value.to_dict()
    assert _md5(path) == before


def test_a_fit_failure_on_a_single_action_names_it(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    _a, b, rows = _batch(src)
    _fail_fit_on(monkeypatch, b)
    path = _copy(src, tmp_path, "apply.ftmw")
    before = _md5(path)

    with pytest.raises(InternalError) as preview:
        ftmw.review_preview(str(path), actions=_actions(rows))
    _assert_internal(preview.value, [2], TypeError)
    with pytest.raises(InternalError) as apply:
        ftmw.review_apply(str(path), actions=_actions(rows))
    _assert_internal(apply.value, [2], TypeError)
    assert _md5(path) == before


def test_the_request_order_not_the_canonical_order_is_named(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """The batch is resolved in canonical (window) order; the error names
    the positions the caller wrote."""
    src = stage5_multi_source
    a, _b, rows = _batch(src)
    _fail_fit_on(monkeypatch, a)
    reordered = [rows[2], rows[0], rows[1]]
    with pytest.raises(InternalError) as err:
        ftmw.review_preview(str(src), actions=_actions(reordered))
    _assert_internal(err.value, [1, 2], TypeError)


def test_a_fit_failure_on_an_implied_create_names_the_add(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """An add no window covers implies a create; a failure fitting it names
    the add."""
    src = stage5_multi_source
    _a, _b, rows = _batch(src)
    ctx = s6._build_shared_fit_ctx(str(src)).fit_ctx
    assert ctx.trim_range is not None
    windows = [
        sorted(float(v) for v in w.freq_range)
        for w in s6.effective_window_plan(str(src)).windows
    ]
    t_lo, t_hi = sorted(float(v) for v in ctx.trim_range)
    anchor = next(
        f
        for f in (t_lo + (t_hi - t_lo) * k / 50 for k in range(5, 46))
        if all(f < lo - 10.0 or f > hi + 10.0 for lo, hi in windows)
    )

    def boom(ctx: Any, create: Any, **kw: Any) -> Any:
        raise RuntimeError(_BOOM)

    monkeypatch.setattr(s6, "_fit_planned_create", boom)
    batch = _actions(rows[:1]) + [
        CurationAction(action="add", freq_mhz=anchor, frame="raw")
    ]
    with pytest.raises(InternalError) as err:
        ftmw.review_preview(str(src), actions=batch)
    _assert_internal(err.value, [1], RuntimeError)


# ---------------------------------------------------------------------------
# Resolve phase and the symbolic pass
# ---------------------------------------------------------------------------


def test_a_resolve_failure_names_the_action(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    a, _b, rows = _batch(src)
    _fail_resolve_on(monkeypatch, a, lambda: IndexError(_BOOM))
    cur = _csv(tmp_path, rows)

    with pytest.raises(InternalError) as preview:
        ftmw.review_preview(str(src), str(cur), frame="raw")
    _assert_internal(preview.value, [0, 1], IndexError)
    path = _copy(src, tmp_path, "apply.ftmw")
    before = _md5(path)
    with pytest.raises(InternalError) as apply:
        ftmw.review_apply(str(path), str(cur), frame="raw")
    assert apply.value.to_dict() == preview.value.to_dict()
    assert _md5(path) == before


def test_a_symbolic_pass_failure_on_a_new_row_names_the_action(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    _a, b, rows = _batch(src)
    _fail_walk_on(monkeypatch, b)
    with pytest.raises(InternalError) as err:
        ftmw.review_preview(str(src), actions=_actions(rows))
    _assert_internal(err.value, [2], KeyError)


def test_a_typed_error_passes_through_untouched(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """A typed error raised while fitting one action keeps its type and
    attributes: only an untyped failure becomes internal_error."""
    from ftmwpipeline.file_manager import AlgorithmFailedError

    src = stage5_multi_source
    a, _b, rows = _batch(src)
    original = AlgorithmFailedError("review", _BOOM)

    def boom(ctx: Any, steps: Any, **kw: Any) -> Any:
        raise original

    monkeypatch.setattr(s6, "_apply_refit_steps", boom)
    with pytest.raises(AlgorithmFailedError) as err:
        ftmw.review_preview(str(src), actions=_actions(rows))
    assert err.value is original


def test_a_keyboard_interrupt_is_never_caught(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    a, _b, rows = _batch(src)

    def boom(ctx: Any, steps: Any, **kw: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(s6, "_apply_refit_steps", boom)
    with pytest.raises(KeyboardInterrupt):
        ftmw.review_preview(str(src), actions=_actions(rows))


def test_a_failure_replaying_a_prior_row_is_not_attributed() -> None:
    """Only a request's new rows map to an action: a row the log already
    held (a serial outside the attribution) re-raises as it is."""
    action = s6.PlannedAction(kind="edit", window_id=0, action_indices=[3])
    attribution = {7: (0, action)}
    exc = TypeError(_BOOM)
    assert s6._raise_attributed_failure(attribution, [5, 6], exc) is None
    assert s6._raise_attributed_failure(None, [7], exc) is None
    with pytest.raises(InternalError) as err:
        s6._raise_attributed_failure(attribution, [6, 7], exc)
    assert err.value.action_indices == [3]
    assert err.value.__cause__ is exc


# ---------------------------------------------------------------------------
# bad_setting inside a batch
# ---------------------------------------------------------------------------


def test_a_bad_setting_from_one_action_names_it(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    src = stage5_multi_source
    _a, b, rows = _batch(src)
    _fail_resolve_on(
        monkeypatch, b, lambda: BadSettingError("peaks", "something", [1.0])
    )
    with pytest.raises(BadSettingError) as err:
        ftmw.review_preview(str(src), actions=_actions(rows))
    assert err.value.action_indices == [2]
    assert err.value.path == "peaks"
    d = err.value.to_dict()
    assert d["action_indices"] == [2] and "action_indices_absent" not in d


def test_an_out_of_band_create_names_its_action_and_field(
    stage5_multi_source: Path, tmp_path: Path
) -> None:
    src = stage5_multi_source
    _a, _b, rows = _batch(src)
    batch = _actions(rows) + [
        CurationAction(action="create", freq_mhz=1000.0, frame="raw")
    ]
    with pytest.raises(BadSettingError) as err:
        ftmw.review_preview(str(src), actions=batch)
    assert err.value.path == "actions[3].freq_mhz"
    assert err.value.action_indices == [3]


def test_a_malformed_action_field_names_its_action(
    stage5_multi_source: Path,
) -> None:
    with pytest.raises(BadSettingError) as err:
        ftmw.review_preview(
            str(stage5_multi_source),
            actions=[
                {"action": "accept", "window_id": 0},
                {"action": "add", "window_id": 0},
            ],
        )
    assert err.value.path.startswith("actions[1].")
    assert err.value.action_indices == [1]


def test_a_bad_setting_outside_a_batch_is_not_run(
    stage5_multi_source: Path, tmp_path: Path
) -> None:
    path = _copy(stage5_multi_source, tmp_path, "verb.ftmw")
    with pytest.raises(BadSettingError) as err:
        ftmw.review_create(str(path), 1000.0, frame="raw")
    assert err.value.action_indices is None
    assert err.value.to_dict()["action_indices_absent"] == "not_run"


# ---------------------------------------------------------------------------
# Every interface
# ---------------------------------------------------------------------------


def _cli_error(argv: List[str], capsys: Any) -> Tuple[int, Dict[str, Any]]:
    capsys.readouterr()
    rc = main(argv + ["--json"])
    err = capsys.readouterr().err.strip().splitlines()[-1]
    payload: Dict[str, Any] = json.loads(err)
    return rc, payload


def test_every_interface_reports_the_same_internal_error(
    stage5_multi_source: Path, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    src = stage5_multi_source
    a, _b, rows = _batch(src)
    _fail_fit_on(monkeypatch, a)
    cur = _csv(tmp_path, rows)

    with pytest.raises(InternalError) as api:
        ftmw.review_preview(str(src), str(cur), frame="raw")
    with pytest.raises(InternalError) as pipe:
        Pipeline.open(str(src)).review_preview(str(cur), frame="raw")
    assert pipe.value.to_dict() == api.value.to_dict()

    rc, payload = _cli_error(
        ["review", "preview", str(src), str(cur), "--frame", "raw"], capsys
    )
    assert rc == 2
    assert payload == api.value.to_dict()

    path = _copy(src, tmp_path, "cli.ftmw")
    before = _md5(path)
    rc, payload = _cli_error(
        ["review", "apply", str(path), str(cur), "--frame", "raw"], capsys
    )
    assert rc == 2
    assert payload["code"] == "internal_error"
    assert payload["action_indices"] == [0, 1]
    assert _md5(path) == before

    pipe_path = _copy(src, tmp_path, "pipe.ftmw")
    with pytest.raises(InternalError) as pipe_apply:
        Pipeline.open(str(pipe_path)).review_apply(str(cur), frame="raw")
    assert pipe_apply.value.to_dict() == api.value.to_dict()
    assert _md5(pipe_path) == before
