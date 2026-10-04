"""Stage 5 partial fits and resume (Wave 5.2).

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Stage 5 partial fits. A cancel (or a raising callback) during the fit keeps
the windows that finished; the next ``fit run`` refits only the others and
produces the fit an uninterrupted run produces, unless its inputs changed (or
it is told to start over), in which case it starts over and says why.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    OperationCancelledError,
    StageDependencyError,
    StageFinished,
    StageStarted,
    WindowProgress,
)
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.contract import FIT_RESTART_REASONS, MANIFEST, Absent
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import Recorder, Token, cancel_on_nth, content_digest

pytestmark = pytest.mark.integration

_SUMMARY = ("resumed", "windows_carried", "restart_reason")


def _copy(src: Path, tmp_path: Path, name: str = "work.ftmw") -> Path:
    dest = tmp_path / name
    shutil.copy(src, dest)
    return dest


def _state(path: Path, stage: str = "fit") -> str:
    rows = ftmw.status(path)["stages"]
    return next(r["state"] for r in rows if r["stage"] == stage)


def _cancel_after_first_window(fp: Path, **kwargs: Any) -> OperationCancelledError:
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, 1, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=1, events=rec, cancel=tok, **kwargs)
    return info.value


def _fit(fp: Path, **kwargs: Any) -> Tuple[Dict[str, Any], Recorder]:
    rec = Recorder()
    ftmw.fit_peaks(fp, jobs=1, events=rec, **kwargs)
    (fin,) = rec.of(StageFinished)
    return dict(fin.summary), rec


def _table(fp: Path) -> List[Tuple[Any, ...]]:
    fit = ftmw.load_fit(fp)
    return [
        (
            w.window_id,
            tuple(w.window.freq_range) if w.window is not None else None,
            tuple((p.peak_uid, p.origin) for p in w.fitted_peaks),
            tuple(p.frequency_mhz for p in w.fitted_peaks),
            tuple(p.amplitude for p in w.fitted_peaks),
            w.shared_parameters.get("tau_us", {}).get("value"),
        )
        for w in sorted(fit.window_fits, key=lambda r: r.window_id)
    ]


@pytest.fixture
def partial_file(baseline_2638_stage4_small, tmp_path):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    err = _cancel_after_first_window(fp)
    return fp, err


# ---- writing a partial fit -----------------------------------------------------------


def test_a_cancel_keeps_the_finished_windows_as_a_partial_fit(partial_file):
    fp, err = partial_file
    assert err.code == "cancelled" and err.stage == "fit"
    assert len(err.completed_windows) == 1
    assert err.to_dict()["completed_windows"] == err.completed_windows

    status = ftmw.status(fp)
    states = {r["stage"]: r["state"] for r in status["stages"]}
    assert states["fit"] == "partial"
    assert states["review"] == "not_run"
    assert "fit" in status["runnable"] and "review" not in status["runnable"]

    rows = {r.window_id: r for r in ftmw.window_status(fp)["windows"]}
    (kept,) = err.completed_windows
    assert isinstance(rows[kept].n_fitted_peaks, int)
    assert rows[kept].live == (rows[kept].n_fitted_peaks > 0)
    for wid, row in rows.items():
        if wid != kept:
            assert row.n_fitted_peaks is Absent.NOT_RUN
            assert row.live is Absent.NOT_RUN


def test_a_partial_fit_is_not_a_fit(partial_file):
    fp, _err = partial_file
    with pytest.raises(StageDependencyError) as info:
        ftmw.load_fit(fp)
    assert info.value.code == "stage_not_run"
    with pytest.raises(StageDependencyError) as info:
        ftmw.review_run(fp)
    assert info.value.code == "stage_not_run"


def test_a_partial_fit_discards_the_previous_fit(baseline_2638_stage4_small, tmp_path):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    ftmw.fit_peaks(fp, jobs=1)
    assert _state(fp) == "complete"
    _cancel_after_first_window(fp)
    assert _state(fp) == "partial"
    with pytest.raises(StageDependencyError):
        ftmw.load_fit(fp)


def test_a_cancel_before_any_window_finishes_writes_nothing(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    ftmw.fit_peaks(fp, jobs=1)  # a previous fit, which must be kept
    before = content_digest(fp)
    tok = Token()
    rec = Recorder(cancel_on_nth(StageStarted, 1, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=1, events=rec, cancel=tok)
    assert info.value.completed_windows == []
    assert content_digest(fp) == before
    assert _state(fp) == "complete"


# ---- resume --------------------------------------------------------------------------


def test_a_resume_fits_only_the_rest_and_equals_an_uninterrupted_fit(
    baseline_2638_stage4_small, tmp_path, partial_file
):
    ref = _copy(baseline_2638_stage4_small, tmp_path, "ref.ftmw")
    ref_summary, _ = _fit(ref)
    fp, err = partial_file

    summary, rec = _fit(fp)
    assert summary["resumed"] is True
    assert summary["windows_carried"] == len(err.completed_windows)
    assert summary["restart_reason"] is None
    assert (ref_summary["resumed"], ref_summary["restart_reason"]) == (False, None)
    assert ref_summary["windows_carried"] == 0

    # Progress continues from the carried count; total stays the full count.
    initial = [e for e in rec.of(WindowProgress) if e.phase == "initial"]
    n_windows = initial[0].total
    assert [e.index for e in initial] == list(
        range(len(err.completed_windows) + 1, n_windows + 1)
    )
    assert not {e.window_id for e in initial} & set(err.completed_windows)

    assert _table(fp) == _table(ref)
    assert _state(fp) == "complete"
    with h5py.File(fp, "r") as h5f:
        assert "stage5_partial" not in h5f


def test_restart_discards_the_partial_fit(partial_file):
    fp, _err = partial_file
    summary, rec = _fit(fp, restart=True)
    assert (summary["resumed"], summary["windows_carried"]) == (False, 0)
    assert summary["restart_reason"] == "restart_requested"
    assert [e.index for e in rec.of(WindowProgress) if e.phase == "initial"][0] == 1


def test_changed_settings_start_over(partial_file):
    fp, _err = partial_file
    summary, _ = _fit(fp, tau_maj_override_us=3.0, sigma_tau_override_us=0.5)
    assert (summary["resumed"], summary["restart_reason"]) == (
        False,
        "settings_changed",
    )


def test_missing_provenance_starts_over(partial_file):
    fp, _err = partial_file
    with h5py.File(fp, "a") as h5f:
        del h5f["stage5_partial/provenance"]
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["restart_reason"]) == (
        False,
        "incomplete_provenance",
    )


def test_a_restart_with_nothing_to_resume_has_no_reason(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    summary, _ = _fit(fp, restart=True)
    assert (summary["resumed"], summary["restart_reason"]) == (False, None)


def test_the_settings_of_a_cancelled_fit_are_resumed_by_a_plain_run(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after_first_window(fp, tau_maj_override_us=3.0, sigma_tau_override_us=0.5)
    summary, _ = _fit(fp)
    assert (summary["resumed"], summary["restart_reason"]) == (True, None)


# ---- discard -------------------------------------------------------------------------


def test_a_stage5_setting_change_discards_the_partial_fit(partial_file):
    fp, _err = partial_file
    result = ftmw.settings_set(fp, "stage5.conservative.max_peaks", 7)
    assert "fit" in result.invalidated
    assert _state(fp) == "not_run"
    with h5py.File(fp, "r") as h5f:
        assert "stage5_partial" not in h5f


def test_an_upstream_rerun_discards_the_partial_fit(partial_file):
    fp, _err = partial_file
    ftmw.assign_windows(fp)
    assert _state(fp) == "not_run"
    rows = ftmw.window_status(fp)["windows"]
    assert all(r.n_fitted_peaks is Absent.NOT_RUN for r in rows)


def test_the_fingerprint_treats_a_partial_fit_as_not_run(partial_file):
    from ftmwpipeline._internal.fingerprint_impl import canonical_fingerprint_inputs
    from ftmwpipeline.file_manager import IncompleteProvenanceError

    fp, _err = partial_file
    try:
        inputs = canonical_fingerprint_inputs(fp)
    except IncompleteProvenanceError as exc:  # a fixture without full provenance
        pytest.skip(f"fingerprint unavailable on the fixture: {exc}")
    assert inputs["fit"] is None
    assert inputs["fit_absent"] == "not_run"


# ---- summary and interfaces ----------------------------------------------------------


def test_the_summary_keys_and_vocabulary_are_declared():
    from ftmwpipeline._internal.stage5_impl import FIT_RUN_SUMMARY_KEYS

    assert set(_SUMMARY) <= set(FIT_RUN_SUMMARY_KEYS)
    assert set(MANIFEST.vocabularies["restart_reason"]) == set(FIT_RESTART_REASONS)
    assert set(FIT_RESTART_REASONS) == {
        "restart_requested",
        "settings_changed",
        "incomplete_provenance",
        "thaw_refit",
    }


def test_pipeline_and_cli_restart_agree(partial_file, tmp_path, capsys):
    fp, _err = partial_file
    other = _copy(fp, tmp_path, "other.ftmw")

    rec = Recorder()
    Pipeline.open(fp).fit_peaks(jobs=1, restart=True, events=rec)
    (fin,) = rec.of(StageFinished)

    capsys.readouterr()
    rc = cli_main(["fit", "run", str(other), "--jobs", "1", "--restart", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    run_result = json.loads(out)
    assert set(run_result["summary"]) == set(fin.summary)
    for key in _SUMMARY:
        assert run_result["summary"][key] == fin.summary[key]
    assert run_result["summary"]["restart_reason"] == "restart_requested"
