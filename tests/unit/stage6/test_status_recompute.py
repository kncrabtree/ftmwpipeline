"""
Statuses are recomputed on every Stage 6 write, under recorded parameters.

A window's status is a pure function of the window fits, the recorded
``review run`` parameters and the decision log, so after every write the
stored statuses must be exactly what a fresh ``review run`` with the same
parameters computes on the written file, and what the full-replay reference
computes from the log. ``review run`` records the parameters it ran with (one
left out keeps its recorded value), and every other write -- an edit, a bare
accept, an apply, an undo -- reuses them.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.replay_reference import replay_full
from ftmwpipeline.cli import main
from ftmwpipeline.core.data_structures import FittingResult, ReviewParams
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import (
    _status_to_dict,
    load_stage6_review_from_file,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]

#: Parameters that route attention differently from the defaults on the
#: fixtures: a near-zero kappa fails the shape-error gate almost everywhere.
_STRICT = ReviewParams(
    bar=2.0, attention_candidate_evidence=3.0, kappa=1e-4, noise_floor=0.5
)


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(wf.window_id): wf for wf in sf.window_fits if wf.window_id is not None}


def _freqs(path: Path, wid: int) -> List[float]:
    return sorted(float(p.frequency_mhz) for p in _fits(path)[wid].fitted_peaks)


def _clear(path: Path, wid: int) -> float:
    """The in-window position furthest from every fitted peak."""
    wf = _fits(path)[wid]
    assert wf.window is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    ps = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min((abs(x - q) for q in ps), default=1e9),
    )


def _statuses(path: Path) -> Dict[int, Dict[str, Any]]:
    review = load_stage6_review_from_file(str(path))
    return {w: _status_to_dict(s) for w, s in review.window_statuses.items()}


def _fresh_review_run(path: Path, params: ReviewParams) -> Dict[int, Dict[str, Any]]:
    """The statuses a fresh ``review run`` with *params* writes on a copy."""
    copy = path.with_name(f"fresh_{path.name}")
    shutil.copy(path, copy)
    ftmw.review_run(
        copy,
        bar=params.bar,
        attention_candidate_evidence=params.attention_candidate_evidence,
        kappa=params.kappa,
        noise_floor=params.noise_floor,
    )
    out = _statuses(copy)
    copy.unlink()
    return out


def _assert_statuses_are_fresh(path: Path, params: ReviewParams) -> None:
    """The stored statuses are a fresh ``review run``'s under the recorded
    parameters, which are *params*, and the reference's for the stored log."""
    review = load_stage6_review_from_file(str(path))
    assert review.review_params == params
    stored = _statuses(path)
    assert stored == _fresh_review_run(path, params)
    reference = replay_full(path, review.decision_log, params)
    assert {
        w: _status_to_dict(s) for w, s in reference.review.window_statuses.items()
    } == stored


def _multi_window_ids(path: Path, n: int, min_peaks: int) -> List[int]:
    fits = _fits(path)
    wids = [w for w in sorted(fits) if len(fits[w].fitted_peaks) >= min_peaks]
    if len(wids) < n:
        pytest.skip(f"fixture has fewer than {n} windows with {min_peaks}+ peaks")
    return wids[:n]


# ---------------------------------------------------------------------------
# Recorded parameters
# ---------------------------------------------------------------------------


def test_review_run_records_its_parameters(stage5_small_file):
    """A fresh review records the defaults; a parameter passed overrides its
    recorded value and one left out keeps it."""
    fp = stage5_small_file
    ftmw.review_run(fp)
    assert load_stage6_review_from_file(str(fp)).review_params == (
        s6.DEFAULT_REVIEW_PARAMS
    )
    ftmw.review_run(fp, kappa=0.2, noise_floor=4.0)
    ftmw.review_run(fp, bar=3.0)
    assert load_stage6_review_from_file(str(fp)).review_params == ReviewParams(
        bar=3.0,
        attention_candidate_evidence=s6.DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
        kappa=0.2,
        noise_floor=4.0,
    )


def test_a_review_without_recorded_parameters_uses_the_defaults(stage5_small_file):
    """A stored review that records none reads ``None`` and routes attention
    under the defaults; the next write records them."""
    fp = stage5_small_file
    ftmw.review_run(fp, kappa=0.2)
    with h5py.File(str(fp), "a") as h5f:
        del h5f["stage6_review"].attrs["review_params"]
    assert load_stage6_review_from_file(str(fp)).review_params is None
    (wid,) = _multi_window_ids(fp, 1, 1)
    ftmw.review_accept(fp, wid)
    _assert_statuses_are_fresh(fp, s6.DEFAULT_REVIEW_PARAMS)


def test_review_run_parameters_on_every_interface(stage5_small_source, tmp_path):
    """The functional API, the Pipeline class and the CLI record the same
    parameters, and leave out the same ones."""
    paths = {k: tmp_path / f"{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_small_source, p)
    ftmw.review_run(paths["api"], kappa=0.2, noise_floor=4.0)
    Pipeline.open(paths["pipe"]).review_run(kappa=0.2, noise_floor=4.0)
    assert (
        main(
            ["review", "run", str(paths["cli"]), "--kappa", "0.2", "--noise-floor", "4"]
        )
        == 0
    )
    expected = ReviewParams(
        bar=s6.DEFAULT_DISPLAY_BAR,
        attention_candidate_evidence=s6.DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
        kappa=0.2,
        noise_floor=4.0,
    )
    for name, p in paths.items():
        assert load_stage6_review_from_file(str(p)).review_params == expected, name
    assert (
        _statuses(paths["api"]) == _statuses(paths["pipe"]) == _statuses(paths["cli"])
    )


@pytest.mark.parametrize(
    "name, flag, value",
    [
        ("bar", "--bar", math.nan),
        ("attention_candidate_evidence", "--attention-bar", math.inf),
        ("kappa", "--kappa", -0.1),
        ("noise_floor", "--noise-floor", -math.inf),
    ],
)
def test_a_bad_parameter_is_refused_on_every_interface(
    stage5_small_file, capsys, name, flag, value
):
    """Every later write reuses what ``review run`` records, so a value that is
    not finite and non-negative is refused (``bad_setting`` naming the
    argument) with the recorded parameters and the file untouched."""
    fp = stage5_small_file
    ftmw.review_run(fp, kappa=0.2)
    recorded = load_stage6_review_from_file(str(fp)).review_params
    before = fp.read_bytes()
    with pytest.raises(BadSettingError) as api_exc:
        ftmw.review_run(fp, **{name: value})
    assert api_exc.value.path == name
    with pytest.raises(BadSettingError) as pipe_exc:
        Pipeline.open(fp).review_run(**{name: value})
    assert pipe_exc.value.path == name
    capsys.readouterr()
    assert main(["review", "run", str(fp), f"{flag}={value}", "--json"]) == 1
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert (payload["code"], payload["path"]) == ("bad_setting", name)
    assert fp.read_bytes() == before
    assert load_stage6_review_from_file(str(fp)).review_params == recorded


def test_the_cli_reports_the_recorded_parameters(stage5_small_file, capsys):
    """``review show --json`` carries the recorded parameters in every mode,
    as ``get_review_status`` does; the text forms print them."""
    fp = stage5_small_file
    ftmw.review_run(fp, kappa=0.2)
    expected = asdict(ftmw.get_review_status(fp).review_params)
    assert expected["kappa"] == 0.2
    (wid,) = _multi_window_ids(fp, 1, 1)
    for mode in ([], ["--attention"], ["--candidates"], ["--window", str(wid)]):
        capsys.readouterr()
        assert main(["review", "show", str(fp), *mode, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["review_params"] == expected, mode
    for mode in (["--attention"], ["--window", str(wid)]):
        capsys.readouterr()
        assert main(["review", "show", str(fp), *mode]) == 0
        assert "kappa=0.2, " in capsys.readouterr().out, mode

    # A review that records none: null, and the text names the defaults.
    with h5py.File(str(fp), "a") as h5f:
        del h5f["stage6_review"].attrs["review_params"]
    capsys.readouterr()
    assert main(["review", "show", str(fp), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["review_params"] is None
    assert main(["review", "show", str(fp), "--attention"]) == 0
    assert "(defaults)" in capsys.readouterr().out


def test_bare_accepts_recorded_together_equal_one_by_one(stage5_multi_source, tmp_path):
    """A bare-accept-only apply records its rows in one write, and leaves
    the log and statuses one ``review accept`` per row leaves; an undo that
    replays a bare-accept-only log keeps the survivors' rows verbatim."""
    one, together = tmp_path / "one.ftmw", tmp_path / "together.ftmw"
    for p in (one, together):
        shutil.copy(stage5_multi_source, p)
        ftmw.review_run(p, kappa=_STRICT.kappa)
    wids = _multi_window_ids(one, 3, 1)
    for wid in wids:
        ftmw.review_accept(one, wid)
    ftmw.review_apply(
        together,
        actions=[{"action": "accept", "window_id": w} for w in wids],
        frame="raw",
    )
    params = load_stage6_review_from_file(str(one)).review_params
    assert params is not None
    assert ftmw.review_log(one) == ftmw.review_log(together)
    assert _statuses(one) == _statuses(together)
    _assert_statuses_are_fresh(together, params)

    log = ftmw.review_log(together)
    ftmw.review_undo(together, [log[1].serial])
    assert ftmw.review_log(together) == [
        replace(log[0], order_index=0),
        replace(log[2], order_index=1),
    ]
    _assert_statuses_are_fresh(together, params)


# ---------------------------------------------------------------------------
# Every write leaves fresh statuses
# ---------------------------------------------------------------------------


def test_every_write_leaves_the_statuses_of_a_fresh_review_run(stage5_multi_file):
    """Under recorded non-default parameters: an edit, a remove, a bare accept,
    an apply, an inferred split and undos each leave exactly the statuses a
    fresh ``review run`` with those parameters computes, and the reference's."""
    fp = stage5_multi_file
    w1, w2, w3 = _multi_window_ids(fp, 3, 2)
    tol = s6.refit_snap_tol_mhz_impl(str(fp))
    ftmw.review_run(
        fp,
        bar=_STRICT.bar,
        attention_candidate_evidence=_STRICT.attention_candidate_evidence,
        kappa=_STRICT.kappa,
        noise_floor=_STRICT.noise_floor,
    )
    # The parameters matter on this fixture: the defaults route differently.
    assert _statuses(fp) != _fresh_review_run(fp, s6.DEFAULT_REVIEW_PARAMS)
    _assert_statuses_are_fresh(fp, _STRICT)

    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_edit(fp, w2, remove=[_freqs(fp, w2)[0]], frame="raw")
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_accept(fp, w3)
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_apply(
        fp,
        actions=[
            {"action": "add", "window_id": w3, "freq_mhz": _clear(fp, w3)},
            {"action": "remove", "window_id": w1, "freq_mhz": _freqs(fp, w1)[-1]},
            {"action": "accept", "window_id": w2},
        ],
        frame="raw",
    )
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_edit(fp, w2, add=[_freqs(fp, w2)[-1] + 0.5 * tol], frame="raw")
    _assert_statuses_are_fresh(fp, _STRICT)

    log = ftmw.review_log(fp)
    ftmw.review_undo(fp, [log[1].serial])
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_undo(fp, [e.serial for e in ftmw.review_log(fp)])
    _assert_statuses_are_fresh(fp, _STRICT)
    assert {s["provenance"] for s in _statuses(fp).values()} == {"auto"}


def test_a_staged_preview_persists_fresh_statuses(stage5_multi_file):
    """A ``ReviewSession`` apply that persists its staged preview writes the
    statuses a fresh ``review run`` computes."""
    fp = stage5_multi_file
    (w1,) = _multi_window_ids(fp, 1, 2)
    ftmw.review_run(fp, kappa=_STRICT.kappa)
    params = load_stage6_review_from_file(str(fp)).review_params
    assert params is not None and params.kappa == _STRICT.kappa
    actions = [{"action": "add", "window_id": w1, "freq_mhz": _clear(fp, w1)}]
    with Pipeline.open(fp).review_session() as session:
        session.review_preview(actions=actions, frame="raw")
        session.review_apply(actions=actions, frame="raw")
    _assert_statuses_are_fresh(fp, params)


def test_undo_keeps_the_recorded_parameters(stage5_multi_file):
    """An undo replays from the automatic fit under the parameters ``review
    run`` recorded, never the defaults."""
    fp = stage5_multi_file
    w1, w2 = _multi_window_ids(fp, 2, 2)
    ftmw.review_run(
        fp,
        bar=_STRICT.bar,
        attention_candidate_evidence=_STRICT.attention_candidate_evidence,
        kappa=_STRICT.kappa,
        noise_floor=_STRICT.noise_floor,
    )
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_accept(fp, w2)
    ftmw.review_undo(fp, [ftmw.review_log(fp)[0].serial])
    _assert_statuses_are_fresh(fp, _STRICT)
    assert _statuses(fp) != _fresh_review_run(fp, s6.DEFAULT_REVIEW_PARAMS)


def _gap_anchor(path: Path) -> float:
    """The strongest Stage 3 peak no plan window covers: a create there
    installs a new window."""
    plan = ftmw.load_windows(str(path))
    spans = [(min(w.freq_range), max(w.freq_range)) for w in plan.windows]
    free = [
        p
        for p in ftmw.load_peaks(str(path))
        if p.snr is not None
        and all(not (lo - 5.0 <= p.frequency <= hi + 5.0) for lo, hi in spans)
    ]
    if not free:
        pytest.skip("no free Stage 3 peak to create a window at")
    return float(max(free, key=lambda p: float(p.snr)).frequency)


def test_a_create_and_its_undo_leave_fresh_statuses(stage5_multi_file):
    """A created window gets a status under the recorded parameters, and
    undoing the create removes it; both leave a fresh ``review run``'s
    statuses."""
    fp = stage5_multi_file
    (w1,) = _multi_window_ids(fp, 1, 2)
    anchor = _gap_anchor(fp)
    ftmw.review_run(
        fp,
        bar=_STRICT.bar,
        attention_candidate_evidence=_STRICT.attention_candidate_evidence,
        kappa=_STRICT.kappa,
        noise_floor=_STRICT.noise_floor,
    )
    before = _statuses(fp)

    created = ftmw.review_create(fp, anchor, frame="raw")
    after = _statuses(fp)
    assert created.window_id in after and created.window_id not in before
    _assert_statuses_are_fresh(fp, _STRICT)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    _assert_statuses_are_fresh(fp, _STRICT)

    log = ftmw.review_log(fp)
    (create_row,) = [e for e in log if e.kind == "create_window"]
    ftmw.review_undo(fp, [create_row.serial])
    assert created.window_id not in _statuses(fp)
    _assert_statuses_are_fresh(fp, _STRICT)

    ftmw.review_undo(fp, [e.serial for e in ftmw.review_log(fp)])
    assert _statuses(fp) == before
