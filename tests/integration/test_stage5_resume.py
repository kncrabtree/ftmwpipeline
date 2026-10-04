"""Resuming a Stage 5 partial fit (Wave 5.2).

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Stage 5 partial fits. ``fit_peaks`` / ``fit run`` resumes a partial fit by
default and refits only the windows not yet written; the result equals an
uninterrupted run (the same lines, ``peak_uid`` values and window structure,
parameters to floating-point rounding -- the standard the parallel and
sequential walks meet, ``test_stage5_fitting.py``). It starts over, and says
why, when:

- ``restart=True`` (``restart_requested``);
- the requested settings, the consumed upstream values or ``ANALYSIS_EPOCH``
  differ from the partial fit's (``settings_changed``);
- the partial fit lacks the provenance for that comparison
  (``incomplete_provenance``);
- a thaw was accepted (``thaw_refit``).

It never resumes on a guess. A partial fit is also data from a file that may
come from anyone: reading one never runs code from it.

Writes only to pytest ``tmp_path``. The full-2638 equality checks are
``slow``.
"""

from __future__ import annotations

import json
import multiprocessing
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.fitting.plan_execution as plan_execution
from ftmwpipeline import (
    OperationCancelledError,
    PipelineWarning,
    StageFinished,
    WindowProgress,
)
from ftmwpipeline._internal import stage5_partial_impl
from ftmwpipeline._internal.stage5_impl import FIT_RUN_SUMMARY_KEYS
from ftmwpipeline.core.environment import ANALYSIS_EPOCH
from ftmwpipeline.core.stage_fit_settings import (
    ClockSource,
    StageFitSettings,
    TauSubSettings,
)
from tests._events_support import Recorder, Token, cancel_on_nth

pytestmark = pytest.mark.integration

_FORK = "fork" in multiprocessing.get_all_start_methods()
needs_fork = pytest.mark.skipif(not _FORK, reason="needs fork")
_RESUME_KEYS = ("resumed", "windows_carried", "restart_reason")
_PROV = "stage5_partial/provenance"
_OVERRIDES = {"tau_maj_override_us": 3.0, "sigma_tau_override_us": 0.5}


# ---- helpers ------------------------------------------------------------------------


def _copy(src: Path, tmp_path: Path, name: str = "work.ftmw") -> Path:
    dest = tmp_path / name
    shutil.copy(src, dest)
    return dest


def _state(path: Path, stage: str = "fit") -> str:
    return next(r["state"] for r in ftmw.status(path)["stages"] if r["stage"] == stage)


def _has_partial(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        return "stage5_partial" in h5f


def _kept_ids(path: Path) -> List[int]:
    with h5py.File(path, "r") as h5f:
        return sorted(int(w) for w in h5f["stage5_partial/window_ids"][()])


def _cancel_after(
    fp: Path, n: int = 1, *, jobs: int = 1, **kwargs: Any
) -> OperationCancelledError:
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, n, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=rec, cancel=tok, **kwargs)
    return info.value


def _fit(fp: Path, **kwargs: Any) -> Tuple[Dict[str, Any], Recorder]:
    rec = Recorder()
    kwargs.setdefault("jobs", 1)
    ftmw.fit_peaks(fp, events=rec, **kwargs)
    (fin,) = rec.of(StageFinished)
    return dict(fin.summary), rec


def _counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The summary without the resume fields: what the fit found."""
    return {k: summary[k] for k in FIT_RUN_SUMMARY_KEYS if k not in _RESUME_KEYS}


def _tau(window: Any) -> Dict[str, Any]:
    return window.shared_parameters.get("tau_us", {})


def assert_same_fit(path_a: Path, path_b: Path) -> None:
    """The same fit: identical structure and identities, parameters equal to
    the tolerance the parallel and sequential walks are held to."""
    a, b = ftmw.load_fit(path_a), ftmw.load_fit(path_b)
    assert a.n_windows == b.n_windows
    assert a.n_fitted_peaks == b.n_fitted_peaks
    assert a.final_plan_revision == b.final_plan_revision
    by_a = {w.window_id: w for w in a.window_fits}
    by_b = {w.window_id: w for w in b.window_fits}
    assert set(by_a) == set(by_b)
    for wid, wa in by_a.items():
        wb = by_b[wid]
        if wa.window is not None and wb.window is not None:
            assert tuple(wa.window.freq_range) == pytest.approx(
                tuple(wb.window.freq_range), abs=1e-6
            ), wid
        assert _tau(wa).get("fitted") == _tau(wb).get("fitted"), wid
        va, vb = _tau(wa).get("value"), _tau(wb).get("value")
        if va is not None and vb is not None:
            assert va == pytest.approx(vb, rel=1e-6), wid
        pa = sorted(wa.fitted_peaks, key=lambda p: p.frequency_mhz)
        pb = sorted(wb.fitted_peaks, key=lambda p: p.frequency_mhz)
        assert len(pa) == len(pb), f"window {wid} peak count"
        for x, y in zip(pa, pb):
            assert x.peak_uid == y.peak_uid, wid
            assert x.origin == y.origin, wid
            assert x.window_id == y.window_id
            assert x.detection_index == y.detection_index
            assert x.frequency_mhz == pytest.approx(y.frequency_mhz, abs=1e-6)
            assert x.amplitude == pytest.approx(y.amplitude, rel=1e-6, abs=1e-9)


# ---- fixtures: a reference fit and a ready-made partial fit -----------------------------


@pytest.fixture(scope="module")
def reference(baseline_2638_stage4_small, tmp_path_factory):
    """The uninterrupted sequential fit (read-only) and its summary."""
    fp = tmp_path_factory.mktemp("resume_ref") / "ref.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)
    summary, _ = _fit(fp)
    return fp, summary


@pytest.fixture(scope="module")
def partial_source(baseline_2638_stage4_small, tmp_path_factory):
    """A file whose sequential fit was cancelled after its first window
    (read-only: tests copy it)."""
    fp = tmp_path_factory.mktemp("resume_partial") / "partial.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)
    err = _cancel_after(fp)
    return fp, err.completed_windows


@pytest.fixture
def partial(partial_source, tmp_path):
    fp, kept = partial_source
    return _copy(fp, tmp_path), kept


# ---- a resume equals an uninterrupted fit ------------------------------------------------


@pytest.mark.parametrize(
    "cancel_jobs, resume_jobs",
    [
        (1, 1),
        pytest.param(2, 2, marks=needs_fork),
        pytest.param(2, 1, marks=needs_fork),
        pytest.param(1, 2, marks=needs_fork),
    ],
)
def test_a_resume_fits_only_the_rest_and_equals_an_uninterrupted_fit(
    cancel_jobs, resume_jobs, reference, baseline_2638_stage4_small, tmp_path
):
    ref, ref_summary = reference
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    all_ids = {r.window_id for r in ftmw.window_status(fp)["windows"]}
    err = _cancel_after(fp, jobs=cancel_jobs)
    kept = err.completed_windows
    assert kept and set(kept) < all_ids

    summary, rec = _fit(fp, jobs=resume_jobs)
    assert summary["resumed"] is True
    assert summary["windows_carried"] == len(kept)
    assert summary["restart_reason"] is None
    assert set(summary) == set(FIT_RUN_SUMMARY_KEYS)
    assert (ref_summary["resumed"], ref_summary["windows_carried"]) == (False, 0)
    assert ref_summary["restart_reason"] is None
    assert _counts(summary) == _counts(ref_summary)

    # Progress continues from the count already written; total is the full count.
    progress = rec.of(WindowProgress)
    assert {e.phase for e in progress} == {"initial"}
    assert {e.total for e in progress} == {len(all_ids)}
    assert [e.index for e in progress] == list(range(len(kept) + 1, len(all_ids) + 1))
    refit = {e.window_id for e in progress}
    assert not refit & set(kept)
    assert refit | set(kept) == all_ids
    assert not [e for e in rec.of(PipelineWarning) if e.code == "walk_fallback"]

    assert_same_fit(fp, ref)
    assert _state(fp) == "complete"
    assert not _has_partial(fp)


def test_a_fit_after_a_resume_has_nothing_left_to_resume(partial, reference):
    fp, _kept = partial
    _fit(fp)
    again, rec = _fit(fp)
    assert (again["resumed"], again["windows_carried"], again["restart_reason"]) == (
        False,
        0,
        None,
    )
    assert [e.index for e in rec.of(WindowProgress)][0] == 1
    assert_same_fit(fp, reference[0])


@pytest.mark.parametrize("jobs", [1, pytest.param(2, marks=needs_fork)])
def test_a_cancel_after_the_last_window_keeps_all_of_them(
    jobs, reference, baseline_2638_stage4_small, tmp_path, monkeypatch
):
    """The cancel falls on the structural-replan check, after every window of
    the initial walk finished: all of them are kept and the resume redoes only
    the replan."""
    ref, _ref_summary = reference
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    rounds: List[int] = []
    real_round = plan_execution._dispatch_structural_round

    def counting_round(*args: Any, **kwargs: Any) -> Any:
        rounds.append(1)
        return real_round(*args, **kwargs)

    monkeypatch.setattr(plan_execution, "_dispatch_structural_round", counting_round)

    tok = Token()

    def cancel_on_last(event: Any) -> None:
        if isinstance(event, WindowProgress) and event.index == event.total:
            tok.set()

    rec = Recorder(cancel_on_last)
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=rec, cancel=tok)
    kept = info.value.completed_windows
    total = rec.of(WindowProgress)[0].total
    assert len(kept) == total and kept == sorted(
        {e.window_id for e in rec.of(WindowProgress)}
    )
    assert rounds == []  # the cancel came before any replan round dispatched
    with h5py.File(fp, "r") as h5f:
        walk = json.loads(bytes(h5f[_PROV][()]).decode())["walk"]
    assert walk["phase"] == "replan" and walk["n_windows"] == total
    assert _state(fp) == "partial"

    monkeypatch.undo()
    summary, rec2 = _fit(fp, jobs=jobs)
    assert (summary["resumed"], summary["windows_carried"]) == (True, total)
    assert summary["restart_reason"] is None
    assert not rec2.of(WindowProgress)  # no window left to fit
    assert_same_fit(fp, ref)


# ---- starting over, and saying why -------------------------------------------------------


def test_restart_discards_the_partial_fit(partial, reference):
    fp, _kept = partial
    summary, rec = _fit(fp, restart=True)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "restart_requested"
    assert [e.index for e in rec.of(WindowProgress)][0] == 1
    assert_same_fit(fp, reference[0])
    assert not _has_partial(fp)


def test_a_restart_with_nothing_to_resume_has_no_reason(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    summary, _ = _fit(fp, restart=True)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] is None


def test_the_overrides_of_the_call_that_differ_start_over(
    partial, baseline_2638_stage4_small, tmp_path
):
    fp, _kept = partial
    summary, _ = _fit(fp, **_OVERRIDES)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "settings_changed"
    # ... and what it produced is the fit those settings give, not a mix of
    # the partial fit's windows and the new settings.
    fresh = _copy(baseline_2638_stage4_small, tmp_path, "fresh.ftmw")
    _fit(fresh, **_OVERRIDES)
    assert_same_fit(fp, fresh)


def test_the_settings_argument_that_differs_starts_over(partial):
    fp, _kept = partial
    changed = StageFitSettings(tau=TauSubSettings(tau0_us=4.0))
    summary, _ = _fit(fp, settings=changed)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "settings_changed"


def test_the_same_settings_resume(baseline_2638_stage4_small, tmp_path):
    """The comparison is of values: the same setting given again (even as an
    int where the file reads back a float) is not a difference."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp, settings=StageFitSettings(tau=TauSubSettings(tau0_us=4.0)))
    summary, _ = _fit(fp, settings=StageFitSettings(tau=TauSubSettings(tau0_us=4)))
    assert (summary["resumed"], summary["restart_reason"]) == (True, None)


def test_the_settings_of_a_cancelled_fit_are_resumed_by_a_plain_run(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp, **_OVERRIDES)
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["restart_reason"]) == (True, None)
    # Giving them again resumes too.
    fp2 = _copy(baseline_2638_stage4_small, tmp_path, "two.ftmw")
    _cancel_after(fp2, **_OVERRIDES)
    summary, _ = _fit(fp2, **_OVERRIDES)
    assert (summary["resumed"], summary["restart_reason"]) == (True, None)


def test_a_different_analysis_epoch_in_the_partial_fit_starts_over(partial, reference):
    fp, _kept = partial
    _edit_provenance(fp, lambda p: p.update(analysis_epoch=json.dumps(-1)))
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "settings_changed"
    assert_same_fit(fp, reference[0])


def test_a_different_running_analysis_epoch_starts_over(partial, monkeypatch):
    fp, _kept = partial
    monkeypatch.setattr(stage5_partial_impl, "ANALYSIS_EPOCH", ANALYSIS_EPOCH + 1)
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "settings_changed"


def test_a_changed_consumed_upstream_value_starts_over(partial):
    """The fit consumes values from other stages (the timebase epsilon among
    them): a partial fit made before they changed is not resumed."""
    fp, _kept = partial
    clocks = [
        ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
        ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
    ]
    ftmw.calibrate_timebase(fp, clocks=clocks)
    assert _state(fp) == "partial"  # the timebase leaves it alone ...
    summary, _ = _fit(fp)
    assert summary["resumed"] is False  # ... but the next fit does not resume it
    assert summary["restart_reason"] == "settings_changed"


# ---- provenance that cannot be trusted ---------------------------------------------------


def _set_bytes(h5f: h5py.File, path: str, data: Any) -> None:
    if path in h5f:
        del h5f[path]
    if isinstance(data, (bytes, str)):
        raw = data.encode() if isinstance(data, str) else data
        data = np.frombuffer(raw, dtype=np.uint8)
    h5f.create_dataset(path, data=data)


def _edit_provenance(fp: Path, edit: Callable[[Dict[str, Any]], None]) -> None:
    with h5py.File(fp, "a") as h5f:
        prov = json.loads(bytes(h5f[_PROV][()]).decode())
        edit(prov)
        _set_bytes(h5f, _PROV, json.dumps(prov))


def _edit_graph(fp: Path, wid: int, edit: Callable[[Dict[str, Any]], None]) -> None:
    path = f"stage5_partial/windows/w{wid}/graph"
    with h5py.File(fp, "a") as h5f:
        graph = json.loads(bytes(h5f[path][()]).decode())
        edit(graph)
        _set_bytes(h5f, path, json.dumps(graph))


def _outcome_node(graph: Dict[str, Any]) -> Dict[str, Any]:
    return next(n for n in graph["nodes"] if n.get("$o") == "WindowOutcome")


def _drop_key(key: str) -> Callable[[Dict[str, Any]], None]:
    return lambda p: p.pop(key)


def _tamper_cases() -> Dict[str, Callable[[h5py.File, int], None]]:
    def prov_bytes(data: Any) -> Callable[[h5py.File, int], None]:
        return lambda h, wid: _set_bytes(h, _PROV, data)

    def edit_prov(edit: Callable[[Dict[str, Any]], None]):
        def go(h: h5py.File, wid: int) -> None:
            prov = json.loads(bytes(h[_PROV][()]).decode())
            edit(prov)
            _set_bytes(h, _PROV, json.dumps(prov))

        return go

    def edit_graph(edit: Callable[[Dict[str, Any]], None]):
        def go(h: h5py.File, wid: int) -> None:
            path = f"stage5_partial/windows/w{wid}/graph"
            graph = json.loads(bytes(h[path][()]).decode())
            edit(graph)
            _set_bytes(h, path, json.dumps(graph))

        return go

    def unknown_type(graph: Dict[str, Any]) -> None:
        _outcome_node(graph)["$o"] = "os.system"

    def dropped_field(graph: Dict[str, Any]) -> None:
        del _outcome_node(graph)["f"]["fit"]

    def set_walk_thaw(prov: Dict[str, Any]) -> None:
        prov["walk"]["accepted_thaw"] = "no"

    return {
        "provenance_missing": lambda h, wid: h.__delitem__(_PROV),
        "provenance_not_utf8_json": prov_bytes(b"\xff\xfe{"),
        "provenance_not_an_object": prov_bytes(b"[1, 2]"),
        "provenance_wrong_dtype": prov_bytes(np.array([1.5])),
        "provenance_lacks_settings": edit_prov(_drop_key("settings")),
        "provenance_lacks_consumed": edit_prov(_drop_key("consumed")),
        "provenance_lacks_context": edit_prov(_drop_key("context")),
        "provenance_lacks_walk": edit_prov(_drop_key("walk")),
        "provenance_walk_malformed": edit_prov(set_walk_thaw),
        "provenance_other_layout_version": edit_prov(
            lambda p: p.update(format_version=99)
        ),
        "graph_not_json": lambda h, wid: _set_bytes(
            h, f"stage5_partial/windows/w{wid}/graph", b"not json"
        ),
        "graph_names_an_unlisted_type": edit_graph(unknown_type),
        "graph_lacks_a_field": edit_graph(dropped_field),
        "buffer_is_text": lambda h, wid: _set_bytes(
            h,
            f"stage5_partial/windows/w{wid}/buf0",
            np.array(["abc"], dtype=h5py.string_dtype()),
        ),
        "window_ids_name_an_unknown_window": lambda h, wid: _set_bytes(
            h, "stage5_partial/window_ids", np.array([10**6], dtype=np.int64)
        ),
        "window_ids_repeat": lambda h, wid: _set_bytes(
            h, "stage5_partial/window_ids", np.array([wid, wid], dtype=np.int64)
        ),
        "window_ids_empty": lambda h, wid: _set_bytes(
            h, "stage5_partial/window_ids", np.array([], dtype=np.int64)
        ),
        "windows_group_missing": lambda h, wid: h.__delitem__("stage5_partial/windows"),
        "partial_group_is_a_dataset": lambda h, wid: (
            h.__delitem__("stage5_partial"),
            h.create_dataset("stage5_partial", data=np.arange(3)),
        ),
    }


_TAMPER = _tamper_cases()


@pytest.mark.parametrize("case", sorted(_TAMPER))
def test_a_partial_fit_that_cannot_be_trusted_is_not_resumed(case, partial, reference):
    """Missing, unreadable or malformed provenance or window data: the call
    starts over (never resumes on a guess), says ``incomplete_provenance``,
    and produces the correct fit -- nothing escapes."""
    fp, kept = partial
    with h5py.File(fp, "a") as h5f:
        _TAMPER[case](h5f, kept[0])
    summary, rec = _fit(fp)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "incomplete_provenance"
    assert [e.index for e in rec.of(WindowProgress)][0] == 1
    assert _state(fp) == "complete"
    assert not _has_partial(fp)
    assert_same_fit(fp, reference[0])


def _field(name: str, value: Any) -> Callable[[Dict[str, Any]], None]:
    def edit(graph: Dict[str, Any]) -> None:
        _outcome_node(graph)["f"][name] = value

    return edit


# A well-formed graph that holds values of the wrong kind is not caught when the
# partial fit is read: it fails later, inside the walk, with a raw error.
# Suspected bug (reported with Wave 5.2): the decoder checks the graph's shape,
# not the types of the fields it sets, and a stray IndexError is not among the
# errors the resume catches.
@pytest.mark.xfail(strict=True, reason="decoded field types are not validated")
@pytest.mark.parametrize(
    "edit",
    [
        _field("fit", 7),
        _field("offset_grid_mhz", "abc"),
        _field("edge_coherence_low", {"$g": "<c16", "v": [1]}),
    ],
    ids=["fit_is_an_int", "grid_is_text", "short_complex_scalar"],
)
def test_wrongly_typed_window_data_is_not_resumed(edit, partial, reference):
    fp, kept = partial
    _edit_graph(fp, kept[0], edit)
    summary, _ = _fit(fp)
    assert summary["restart_reason"] == "incomplete_provenance"
    assert_same_fit(fp, reference[0])


# ---- reading a partial fit never runs code from the file -------------------------------------


def _text_datasets(group: h5py.Group) -> List[h5py.Dataset]:
    found: List[h5py.Dataset] = []
    group.visititems(
        lambda name, obj: found.append(obj) if isinstance(obj, h5py.Dataset) else None
    )
    return found


def test_the_partial_fit_holds_no_object_serialized_payload(partial):
    fp, _kept = partial
    with h5py.File(fp, "r") as h5f:
        datasets = _text_datasets(h5f["stage5_partial"])
        assert datasets
        for ds in datasets:
            dt = ds.dtype
            # Plain numeric data only: no object, vlen, reference, compound,
            # opaque or string datasets that could carry a pickle.
            assert dt.kind in "buifc", (ds.name, dt)
            assert not dt.hasobject and dt.names is None, ds.name
            assert h5py.check_dtype(vlen=dt) is None, ds.name
            assert h5py.check_dtype(ref=dt) is None, ds.name
        # The text datasets are UTF-8 JSON, and not a pickle stream.
        for name in ("provenance",):
            raw = bytes(h5f[f"stage5_partial/{name}"][()])
            assert not raw.startswith(b"\x80")
            assert isinstance(json.loads(raw.decode("utf-8")), dict)
        for wname in h5f["stage5_partial/windows"]:
            raw = bytes(h5f[f"stage5_partial/windows/{wname}/graph"][()])
            assert not raw.startswith(b"\x80")
            assert "root" in json.loads(raw.decode("utf-8"))
            attrs = h5f[f"stage5_partial/windows/{wname}"].attrs
            assert all(isinstance(v, (str, int, float)) for v in attrs.values())


def test_the_codec_never_unpickles():
    """The only way a pickle could be read is by the modules that read the
    partial fit naming it: they do not."""
    import ftmwpipeline._internal.stage5_partial_impl as impl
    import ftmwpipeline.io.stage5_partial_serialization as ser

    for module in (impl, ser):
        text = Path(module.__file__).read_text()
        for needle in (
            "import pickle",
            "from pickle",
            "allow_pickle",
            "marshal",
            "shelve",
            "cloudpickle",
            "dill",
            "np.load",
            "numpy.load",
            "eval(",
            "exec(",
        ):
            assert needle not in text, (module.__name__, needle)


def test_a_pickle_planted_in_a_partial_fit_is_not_executed(
    partial, reference, tmp_path
):
    fp, kept = partial
    sentinel = tmp_path / "executed.txt"

    class Payload:
        def __reduce__(self) -> Tuple[Any, ...]:
            return (sentinel.write_text, ("pwned",))

    blob = pickle.dumps(Payload())
    with h5py.File(fp, "a") as h5f:
        _set_bytes(h5f, f"stage5_partial/windows/w{kept[0]}/graph", blob)
        _set_bytes(h5f, f"stage5_partial/windows/w{kept[0]}/buf0", blob)
        _set_bytes(h5f, _PROV, blob)
    summary, _ = _fit(fp)
    assert not sentinel.exists()
    assert summary["restart_reason"] == "incomplete_provenance"
    assert_same_fit(fp, reference[0])


def test_a_graph_naming_a_callable_is_not_executed(partial, tmp_path):
    fp, kept = partial
    sentinel = tmp_path / "executed.txt"

    def hostile(graph: Dict[str, Any]) -> None:
        graph["nodes"].append(
            {"$o": "builtins.exec", "f": {"code": f"open({str(sentinel)!r}, 'w')"}}
        )
        graph["nodes"][0] = {"$o": "os.system", "f": {}}

    _edit_graph(fp, kept[0], hostile)
    summary, _ = _fit(fp)
    assert not sentinel.exists()
    assert summary["restart_reason"] == "incomplete_provenance"


# ---- an accepted thaw -------------------------------------------------------------------


def _fake_thaw_on_first_call() -> Callable[..., Any]:
    """``attempt_thaw_round`` with one accepted thaw on its first call (a thaw
    that accepts never happens on the 2638 fixture), the real one after."""
    real = plan_execution.attempt_thaw_round
    state = {"calls": 0}

    def wrapper(win: Any, outcome: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == 1:
            event = plan_execution.ThawEvent(
                win.window_id, win.window_id, 0, 1.0, "low", 9.0, 1.0, True, "test"
            )
            outcome.thaw_events.append(event)
            return [event]
        return real(win, outcome, **kwargs)

    return wrapper


def test_a_thaw_accepted_during_a_resume_refits_every_window(partial, monkeypatch):
    fp, kept = partial
    all_ids = {r.window_id for r in ftmw.window_status(fp)["windows"]}
    monkeypatch.setattr(
        plan_execution, "attempt_thaw_round", _fake_thaw_on_first_call()
    )
    summary, rec = _fit(fp, jobs=1)
    assert summary["resumed"] is False
    assert summary["windows_carried"] == 0
    assert summary["restart_reason"] == "thaw_refit"

    (warning,) = [e for e in rec.of(PipelineWarning) if e.code == "walk_fallback"]
    assert warning.details == {"reason": "accepted thaw", "n_windows": len(all_ids)}
    fallback = [e for e in rec.of(WindowProgress) if e.phase == "fallback"]
    assert [e.index for e in fallback] == list(range(1, len(all_ids) + 1))
    assert {e.total for e in fallback} == {len(all_ids)}
    assert {e.window_id for e in fallback} == all_ids
    assert _state(fp) == "complete"
    assert not _has_partial(fp)


def test_a_partial_fit_that_saw_an_accepted_thaw_is_not_resumed(
    baseline_2638_stage4_small, tmp_path, monkeypatch
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(plan_execution, "attempt_thaw_round", _fake_thaw_on_first_call())
        err = _cancel_after(fp, 2)
    assert len(err.completed_windows) == 2
    with h5py.File(fp, "r") as h5f:
        walk = json.loads(bytes(h5f[_PROV][()]).decode())["walk"]
    assert walk["accepted_thaw"] is True

    summary, rec = _fit(fp)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "thaw_refit"
    assert [e.index for e in rec.of(WindowProgress)][0] == 1
    assert _state(fp) == "complete"


# ---- clocks ----------------------------------------------------------------------------


def test_clocks_alone_do_not_stop_a_resume(partial):
    """Clocks leave the partial fit alone; with no timebase they gate nothing,
    so the settings the fit resolves are the same ones."""
    fp, kept = partial
    ftmw.set_clock_sources(
        fp, [ClockSource(freq_mhz=5760.0, locked=True, label="upconv")]
    )
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["windows_carried"]) == (True, len(kept))
    assert summary["restart_reason"] is None


# ---- the whole 2638 plan (slow) -----------------------------------------------------------


@pytest.fixture(scope="module")
def full_reference(baseline_2638_stage4, tmp_path_factory):
    fp = tmp_path_factory.mktemp("resume_full_ref") / "full_ref.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    summary, rec = _fit(fp, jobs=None)
    return fp, summary, rec.of(WindowProgress)


@pytest.mark.slow
@needs_fork
def test_resuming_the_full_2638_fit_equals_an_uninterrupted_fit(
    full_reference, baseline_2638_stage4, tmp_path
):
    ref, ref_summary, ref_progress = full_reference
    fp = _copy(baseline_2638_stage4, tmp_path)
    tok = Token()

    def cancel_at_40_percent(event: Any) -> None:
        if (
            isinstance(event, WindowProgress)
            and event.phase == "initial"
            and event.index >= int(0.4 * event.total)
        ):
            tok.set()

    first = Recorder(cancel_at_40_percent)
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=None, events=first, cancel=tok)
    kept = info.value.completed_windows
    total = first.of(WindowProgress)[0].total
    assert 0 < len(kept) < total
    assert kept == sorted({e.window_id for e in first.of(WindowProgress)})
    assert _state(fp) == "partial"

    summary, rec = _fit(fp, jobs=None)
    assert (summary["resumed"], summary["windows_carried"]) == (True, len(kept))
    assert summary["restart_reason"] is None
    initial = [e for e in rec.of(WindowProgress) if e.phase == "initial"]
    assert [e.index for e in initial] == list(range(len(kept) + 1, total + 1))
    assert {e.total for e in initial} == {total}
    assert _counts(summary) == _counts(ref_summary)
    assert_same_fit(fp, ref)
    assert not _has_partial(fp)


@pytest.mark.slow
@needs_fork
def test_a_cancel_during_the_replan_keeps_every_initial_window(
    full_reference, baseline_2638_stage4, tmp_path
):
    ref, ref_summary, ref_progress = full_reference
    if not any(e.phase == "replan" for e in ref_progress):
        pytest.skip("the 2638 plan no longer replans")
    fp = _copy(baseline_2638_stage4, tmp_path)
    tok = Token()

    def cancel_in_replan(event: Any) -> None:
        if isinstance(event, WindowProgress) and event.phase == "replan":
            tok.set()

    first = Recorder(cancel_in_replan)
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=None, events=first, cancel=tok)
    initial_ids = {
        e.window_id for e in first.of(WindowProgress) if e.phase == "initial"
    }
    total = first.of(WindowProgress)[0].total
    assert len(initial_ids) == total
    assert set(info.value.completed_windows) == initial_ids
    with h5py.File(fp, "r") as h5f:
        walk = json.loads(bytes(h5f[_PROV][()]).decode())["walk"]
    assert walk["phase"] == "replan"

    summary, rec = _fit(fp, jobs=None)
    assert (summary["resumed"], summary["windows_carried"]) == (True, total)
    assert summary["restart_reason"] is None
    assert not [e for e in rec.of(WindowProgress) if e.phase == "initial"]
    assert [e for e in rec.of(WindowProgress) if e.phase == "replan"]
    assert _counts(summary) == _counts(ref_summary)
    assert_same_fit(fp, ref)
