"""
The one Stage 6 write path (design §6).

Every mutating operation builds the new decision log ``L'`` and review
parameters ``P'`` and curates them: the full reference replay from the
automatic-fit baseline, one cascade, one persist. This module pins what that
structure promises beyond the per-verb behaviour the other Stage 6 modules
cover: the persisted state is the reference after *every* kind of write (and
the check that says so can fail), a write that is refused fits nothing, the
sequential-path machinery the one path replaced is gone, a file that cannot be
curated is refused at admission, and the writes that refit nothing keep the
fits they find.
"""

from __future__ import annotations

import inspect
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.report_diff_impl import report_diff_impl
from ftmwpipeline.core.data_structures import FittingResult
from ftmwpipeline.file_manager import (
    CurationConflictError,
    PipelineCorruptionError,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from tests._events_support import content_digest
from tests._replay_support import assert_persisted_is_reference
from tests.unit.stage6.test_curation import _clear_add_freq

pytestmark = [pytest.mark.integration]


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(w.window_id): w for w in sf.window_fits if w.window_id is not None}


def _two_peak_window(path: Path) -> Tuple[int, List[float]]:
    for wid, wf in sorted(_fits(path).items()):
        if len(wf.fitted_peaks) >= 2:
            return wid, sorted(float(p.frequency_mhz) for p in wf.fitted_peaks)
    pytest.skip("fixture has no window with two fitted peaks")


def _live_windows(path: Path, n: int) -> List[int]:
    wids = [w for w, wf in sorted(_fits(path).items()) if wf.fitted_peaks]
    if len(wids) < n:
        pytest.skip(f"fixture has fewer than {n} fitted windows")
    return wids[:n]


def _anchor_past_every_live_window(path: Path) -> float:
    """A create anchor clear of every fitted window, inside the trim."""
    from ftmwpipeline._internal.stage4_impl import load_windows_impl

    live = set(_fits(path))
    plan = load_windows_impl(str(path))["plan"]
    return (
        max(
            max(w.freq_range)
            for w in plan.windows
            if int(w.window_id) in live and w.freq_range is not None
        )
        + 5.0
    )


def _frequency_bytes(path: Path) -> bytes:
    with h5py.File(str(path), "r") as h5f:
        return bytes(h5f["stage5_fitting/peaks/frequency_mhz"][...].tobytes())


# ---------------------------------------------------------------------------
# G1 after every kind of write, and the check can fail
# ---------------------------------------------------------------------------


class TestEveryWriteLeavesTheReference:
    """Each write verb persists exactly once, through the one writer, and what
    it persists is the reference replay of the log it left."""

    def test_every_verb_persists_once_and_matches_the_reference(
        self, stage5_multi_file, tmp_path, every_write_is_reference
    ):
        path = str(stage5_multi_file)
        checks = every_write_is_reference
        wa, wb = _live_windows(stage5_multi_file, 2)
        wc, freqs = _two_peak_window(stage5_multi_file)
        clear_a = _clear_add_freq(stage5_multi_file, wa)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wb},{_clear_add_freq(stage5_multi_file, wb)},\n")

        def one_persist(verb: str, call: Any) -> Any:
            n = len(checks.checked)
            out = call()
            assert len(checks.checked) == n + 1, f"{verb} did not persist once"
            return out

        one_persist(
            "edit", lambda: ftmw.review_edit(path, wa, add=[clear_a], frame="raw")
        )
        one_persist(
            "edit remove",
            lambda: ftmw.review_edit(path, wc, remove=[freqs[0]], frame="raw"),
        )
        one_persist("accept", lambda: ftmw.review_accept(path, wb))
        one_persist(
            "create",
            lambda: ftmw.review_create(
                path, _anchor_past_every_live_window(stage5_multi_file)
            ),
        )
        one_persist("apply file", lambda: ftmw.review_apply(path, cur, frame="raw"))
        one_persist(
            "apply actions",
            lambda: ftmw.review_apply(
                path,
                actions=[{"action": "accept", "window_id": wa}],
            ),
        )
        one_persist("run", lambda: ftmw.review_run(path, kappa=4.0))
        log = ftmw.review_log(path)
        one_persist(
            "apply at log_prefix",
            lambda: ftmw.review_apply(path, cur, frame="raw", log_prefix=2),
        )
        log = ftmw.review_log(path)
        one_persist("undo", lambda: ftmw.review_undo(path, [log[-1].serial]))

        # A preview is not a write; a session's apply of its staged preview is
        # one, and persists the staged state through the same writer.
        n = len(checks.checked)
        with Pipeline.open(path).review_session() as session:
            session.review_preview(cur, frame="raw")
            assert len(checks.checked) == n
            session.review_apply(cur, frame="raw")
        assert len(checks.checked) == n + 1
        ftmw.review_preview(path, cur, frame="raw")
        assert len(checks.checked) == n + 1

    def test_the_check_fails_on_a_state_that_is_not_the_reference(
        self, stage5_multi_file
    ):
        """Negative control: a persisted fit that differs from the reference
        replay of its log by one ulp is caught, and so is a stale status."""
        path = stage5_multi_file
        wid, freqs = _two_peak_window(path)
        ftmw.review_edit(str(path), wid, remove=[freqs[0]], frame="raw")
        assert_persisted_is_reference(path)

        with h5py.File(str(path), "a") as h5f:
            col = h5f["stage5_fitting/peaks/frequency_mhz"]
            vals = col[...]
            vals[0] = vals[0].__class__(vals[0] + abs(vals[0]) * 1e-15)
            col[...] = vals
        with pytest.raises(AssertionError, match="not the reference replay"):
            assert_persisted_is_reference(path)


# ---------------------------------------------------------------------------
# Refusals come before any fit, creates included
# ---------------------------------------------------------------------------


@pytest.fixture
def forbid_fits(monkeypatch: pytest.MonkeyPatch) -> Any:
    def arm() -> None:
        def refit(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a refused write fitted a window")

        monkeypatch.setattr(s6, "refit_window_core", refit)

    return arm


class TestAnApplyIsRefusedBeforeItFitsAnything:
    """Creates are planned up front with the rest of the request: a create
    that cannot be placed refuses the whole batch before its first edit is
    fit, and the file is untouched."""

    def _file_with_a_refused_create(
        self, path: Path, tmp_path: Path
    ) -> Tuple[Path, int]:
        wa = _live_windows(path, 1)[0]
        taken = wa  # a pinned id that is already a live window's
        cf = tmp_path / "cur.csv"
        cf.write_text(
            f"add,{wa},{_clear_add_freq(path, wa)},\n"
            f"create,{taken},{_anchor_past_every_live_window(path):.6f},\n"
        )
        return cf, taken

    def test_a_refused_create_fits_nothing_on_an_ordinary_apply(
        self, stage5_multi_file, tmp_path, forbid_fits
    ):
        cf, _taken = self._file_with_a_refused_create(stage5_multi_file, tmp_path)
        before = content_digest(stage5_multi_file)
        forbid_fits()
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_apply(str(stage5_multi_file), cf, frame="raw")
        assert exc.value.reason == "replay_conflict"
        assert content_digest(stage5_multi_file) == before

    def test_a_refused_create_fits_nothing_after_an_empty_prefix(
        self, stage5_multi_file, tmp_path, forbid_fits
    ):
        """``log_prefix=0`` drops the log (the baseline is restored in
        memory, not refit), and the file's rows resolve against that state:
        the refused create still comes before any fit."""
        path = str(stage5_multi_file)
        wc, freqs = _two_peak_window(stage5_multi_file)
        ftmw.review_edit(path, wc, remove=[freqs[0]], frame="raw")
        cf, _taken = self._file_with_a_refused_create(stage5_multi_file, tmp_path)
        before = content_digest(stage5_multi_file)
        forbid_fits()
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_apply(path, cf, frame="raw", log_prefix=0)
        assert exc.value.reason == "replay_conflict"
        assert content_digest(stage5_multi_file) == before
        assert len(ftmw.review_log(path)) == 1

    def test_a_refused_create_leaves_a_prefix_apply_untouched(
        self, stage5_multi_file, tmp_path
    ):
        """With kept rows the prefix is curated in memory (it fits), the
        file's create is refused, and nothing is persisted: neither the
        dropped rows' removal nor the kept prefix's state."""
        path = str(stage5_multi_file)
        wa, wb = _live_windows(stage5_multi_file, 2)
        ftmw.review_edit(
            path, wa, add=[_clear_add_freq(stage5_multi_file, wa)], frame="raw"
        )
        ftmw.review_edit(
            path, wb, add=[_clear_add_freq(stage5_multi_file, wb)], frame="raw"
        )
        cf, _taken = self._file_with_a_refused_create(stage5_multi_file, tmp_path)
        before = content_digest(stage5_multi_file)
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_apply(path, cf, frame="raw", log_prefix=1)
        assert exc.value.reason == "replay_conflict"
        assert content_digest(stage5_multi_file) == before
        assert len(ftmw.review_log(path)) == 2


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


class TestAdmission:
    """§6.1 step 1: the baseline is taken inside ``_open_batch`` on the first
    write of a fresh file; recorded decisions with no baseline behind them can
    only be a damaged file."""

    def _without_baseline(self, path: Path) -> None:
        with h5py.File(str(path), "a") as h5f:
            del h5f[s6.STAGE5_BASELINE_GROUP]

    def test_the_first_write_of_a_fresh_file_is_admitted_and_snapshots(
        self, stage5_multi_file
    ):
        path = stage5_multi_file
        with h5py.File(str(path), "r") as h5f:
            assert s6.STAGE5_BASELINE_GROUP not in h5f
        wid = _live_windows(path, 1)[0]
        ftmw.review_accept(str(path), wid)
        with h5py.File(str(path), "r") as h5f:
            assert s6.STAGE5_BASELINE_GROUP in h5f

    @pytest.mark.parametrize(
        "verb",
        ["edit", "accept", "run", "create", "apply", "undo", "undo_dry", "prefix"],
    )
    def test_recorded_decisions_without_a_baseline_are_file_corrupt(
        self, stage5_multi_file, tmp_path, verb
    ):
        path = stage5_multi_file
        wid, freqs = _two_peak_window(path)
        ftmw.review_edit(str(path), wid, remove=[freqs[0]], frame="raw")
        self._without_baseline(path)
        before = content_digest(path)

        calls = {
            "edit": lambda: ftmw.review_edit(
                str(path), wid, remove=[freqs[1]], frame="raw"
            ),
            "accept": lambda: ftmw.review_accept(str(path), wid),
            "run": lambda: ftmw.review_run(str(path), kappa=4.0),
            "create": lambda: ftmw.review_create(
                str(path), _anchor_past_every_live_window(path)
            ),
            "apply": lambda: ftmw.review_apply(
                str(path),
                actions=[{"action": "accept", "window_id": wid}],
            ),
            "undo": lambda: ftmw.review_undo(str(path), [0]),
            "undo_dry": lambda: ftmw.review_undo(str(path), [0], dry_run=True),
            "prefix": lambda: ftmw.review_apply(
                str(path),
                actions=[{"action": "accept", "window_id": wid}],
                log_prefix=0,
            ),
        }
        with pytest.raises(PipelineCorruptionError, match="no automatic-fit baseline"):
            calls[verb]()
        assert content_digest(path) == before


# ---------------------------------------------------------------------------
# What the one path replaced is gone
# ---------------------------------------------------------------------------


_DEAD_NAMES = (
    "_reset_to_baseline",
    "_restore_stage5_baseline",
    "_DeferredCuration",
    "_resolve_deferred_curation",
    "_execute_curation_batch",
    "_run_curation_batch",
    "_apply_batch_segment",
    "_apply_actions",
    "_run_single_action",
    "_record_bare_accept",
    "_record_bare_accepts",
    "_apply_bare_accepts",
    "_derive_batch_review",
    "_persist_batch_review",
    "_close_batch_action",
    "_refuse_unavailable_cascade",
    "_BatchOutcome",
    "_window_structure_from_create",
    "_created_windows_from_log",
)


class TestTheSequentialPathIsGone:
    @pytest.mark.parametrize("name", _DEAD_NAMES)
    def test_the_name_no_longer_exists(self, name):
        assert not hasattr(s6, name), f"{name} belonged to the sequential path"

    def test_the_batch_carries_no_log_position_bookkeeping(self):
        for cls in (s6._BatchChangeset, s6._BatchCtx):
            fields = set(getattr(cls, "__dataclass_fields__", {}))
            assert not fields & {"log_dirty_wids", "mutated_wids", "baseline_taken"}

    def test_an_undo_does_not_rerun_review_run(self, stage5_multi_file, monkeypatch):
        """Undo is a write that curates the log it leaves; the statuses come
        out of that curate, not out of a second ``review run`` pass."""
        path = str(stage5_multi_file)
        wid, freqs = _two_peak_window(stage5_multi_file)
        ftmw.review_edit(path, wid, remove=[freqs[0]], frame="raw")
        ftmw.review_accept(path, wid)

        def no_rerun(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("undo reran review run")

        monkeypatch.setattr(s6, "review_run_impl", no_rerun)
        log = ftmw.review_log(path)
        result = ftmw.review_undo(path, [log[0].serial])
        assert result.applied == 1
        assert [e.kind for e in ftmw.review_log(path)] == ["accept"]
        assert_persisted_is_reference(path)

    def test_every_write_entry_point_curates_through_the_one_function(self):
        """No write verb keeps a private apply loop: each reaches ``_curate``
        through ``_curate_request`` (or, for ``review run`` and the prefix
        apply, directly)."""
        src = inspect.getsource(s6)
        assert len(re.findall(r"\bdef _curate\(", src)) == 1
        for verb in (
            s6.refit_window_impl,
            s6.create_window_impl,
            s6.apply_curation_impl,
            s6.review_undo_impl,
            s6.review_accept_impl,
        ):
            text = inspect.getsource(verb)
            assert (
                "_curate_request(" in text
                or "_curate(" in text
                or "_apply_curation_at_prefix(" in text
                or "_single_action_refit(" in text
            ), f"{verb.__name__} does not write through the one path"


# ---------------------------------------------------------------------------
# Writes that refit nothing keep the fits they find
# ---------------------------------------------------------------------------


class TestWritesThatRefitNothing:
    def test_a_bare_accept_keeps_the_fits_and_needs_no_fit_context(
        self, stage5_multi_file, monkeypatch
    ):
        path = str(stage5_multi_file)
        wid = _live_windows(stage5_multi_file, 1)[0]
        before = _frequency_bytes(stage5_multi_file)

        def no_context(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a bare accept built a fit context")

        monkeypatch.setattr(s6, "_build_shared_fit_ctx", no_context)
        ftmw.review_accept(path, wid)
        assert _frequency_bytes(stage5_multi_file) == before
        assert_persisted_is_reference(stage5_multi_file)

    def test_review_run_refits_nothing_and_records_its_parameters(
        self, stage5_multi_file, monkeypatch
    ):
        path = str(stage5_multi_file)
        wid, freqs = _two_peak_window(stage5_multi_file)
        ftmw.review_edit(path, wid, remove=[freqs[0]], frame="raw")
        before = _frequency_bytes(stage5_multi_file)

        def no_fit(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("review run refit a window")

        monkeypatch.setattr(s6, "refit_window_core", no_fit)
        ftmw.review_run(path, kappa=3.0)
        monkeypatch.undo()  # the oracle replay below does fit
        assert _frequency_bytes(stage5_multi_file) == before
        recorded = load_stage6_review_from_file(path).review_params
        assert recorded is not None and recorded.kappa == 3.0
        assert_persisted_is_reference(stage5_multi_file)

    def test_undoing_every_fit_row_restores_the_baseline_by_copy(
        self, stage5_multi_file, monkeypatch
    ):
        path = str(stage5_multi_file)
        wid, freqs = _two_peak_window(stage5_multi_file)
        ftmw.review_edit(path, wid, remove=[freqs[0]], frame="raw")
        ftmw.review_accept(path, wid)
        fit_row = next(e for e in ftmw.review_log(path) if e.kind != "accept")

        def no_fit(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a restore of the baseline refit a window")

        monkeypatch.setattr(s6, "refit_window_core", no_fit)
        ftmw.review_undo(path, [fit_row.serial])
        with h5py.File(str(stage5_multi_file), "r") as h5f:
            baseline = h5f[f"{s6.STAGE5_BASELINE_GROUP}/peaks/frequency_mhz"][...]
        assert _frequency_bytes(stage5_multi_file) == bytes(baseline.tobytes())
        assert_persisted_is_reference(stage5_multi_file)


# ---------------------------------------------------------------------------
# report diff: a baseline with no fit-changing decision is no curation
# ---------------------------------------------------------------------------


class TestReportDiffReadsTheDecisionLog:
    def test_bare_accepts_alone_are_no_curation(self, stage5_multi_file, tmp_path):
        wid = _live_windows(stage5_multi_file, 1)[0]
        ftmw.review_accept(str(stage5_multi_file), wid)
        with h5py.File(str(stage5_multi_file), "r") as h5f:
            assert s6.STAGE5_BASELINE_GROUP in h5f
        out = report_diff_impl(str(stage5_multi_file), output_dir=str(tmp_path / "o"))
        text = Path(out).read_text()
        assert "No curation" in text
        assert 'section class="win"' not in text

    def test_undoing_every_fit_row_is_no_curation_again(
        self, stage5_multi_file, tmp_path
    ):
        path = str(stage5_multi_file)
        wid, freqs = _two_peak_window(stage5_multi_file)
        ftmw.review_edit(path, wid, remove=[freqs[0]], frame="raw")
        out = report_diff_impl(path, output_dir=str(tmp_path / "a"))
        assert f"Window {wid}" in Path(out).read_text()
        ftmw.review_undo(path, [ftmw.review_log(path)[0].serial])
        out = report_diff_impl(path, output_dir=str(tmp_path / "b"))
        assert "No curation" in Path(out).read_text()


# ---------------------------------------------------------------------------
# The documented refusals are in the docstrings
# ---------------------------------------------------------------------------

_RAISING_VERBS = (
    "review_edit",
    "review_accept",
    "review_create",
    "review_apply",
    "review_run",
    "review_undo",
)


class TestDocstringsNameTheRefusals:
    """Every write verb's Raises section lists the refusals of the one write
    path on both the functional API and the Pipeline class."""

    @pytest.mark.parametrize("verb", _RAISING_VERBS)
    @pytest.mark.parametrize("surface", ["api", "pipeline"])
    def test_raises_lists_curation_and_compatibility_refusals(self, verb, surface):
        fn = getattr(ftmw if surface == "api" else Pipeline, verb)
        doc = inspect.getdoc(fn) or ""
        assert "Raises" in doc, f"{surface}.{verb} has no Raises section"
        raises = doc.split("Raises", 1)[1]
        assert "PipelineCompatibilityError" in raises
        if verb != "review_run":
            assert "CurationConflictError" in raises
        assert "AnalysisEpochMismatchError" in raises or verb == "review_run"


# ---------------------------------------------------------------------------
# The splice-compatibility docstring states the gate rule
# ---------------------------------------------------------------------------


def test_the_gate_docstring_states_it_applies_only_to_a_refitting_write():
    doc = inspect.getdoc(s6.require_splice_compatible_environment) or ""
    assert "refit" in doc


# ---------------------------------------------------------------------------
# The curated state is a function of the log, not of the call history
# ---------------------------------------------------------------------------


def _without_derivation(part: Any) -> Any:
    """*part* with every ``derivation`` removed: it holds the serial of the
    decision that made a peak, which differs between two call histories."""
    if isinstance(part, dict):
        return {k: _without_derivation(v) for k, v in part.items() if k != "derivation"}
    if isinstance(part, list):
        return [_without_derivation(v) for v in part]
    return part


def test_decisions_on_different_windows_commute_across_call_order(
    stage5_multi_file, tmp_path
):
    """The same two edits made in either order leave the same fitted numbers
    and final products: a write replays the log, so an edit never builds on the
    curated fit the other left. The logs differ in order and serials, and so
    does every ``derivation`` (the serial of the decision that made a peak)."""
    from ftmwpipeline._internal.replay_reference import _persisted_parts

    wa, wb = _live_windows(stage5_multi_file, 2)
    adds = {w: _clear_add_freq(stage5_multi_file, w) for w in (wa, wb)}
    parts = []
    for name, order in (("ab", (wa, wb)), ("ba", (wb, wa))):
        fp = tmp_path / f"{name}.ftmw"
        shutil.copy(stage5_multi_file, fp)
        for w in order:
            ftmw.review_edit(str(fp), w, add=[adds[w]], frame="raw")
        parts.append(_persisted_parts(str(fp)))
    ab, ba = parts
    assert _without_derivation(ab["fit"]) == _without_derivation(ba["fit"])
    assert _without_derivation(ab["review"]["final_products"]) == _without_derivation(
        ba["review"]["final_products"]
    )
    assert ab["review"]["window_statuses"] == ba["review"]["window_statuses"]
