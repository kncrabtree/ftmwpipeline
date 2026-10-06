"""Structural invariants of the Stage 6 write path.

Every Stage 6 write -- the interactive single-window verbs, a curation file, an
undo, ``review run``, a session's staged preview -- runs through one path:
``_open_batch`` (admission and the undo baseline), resolution of the request
into decision rows, ``_curate`` (the structural pass, the epoch gate when the
write refits, the full reference replay) and ``_finish_batch`` (one fit
persist, one review persist). The point of routing everything through it is
that the guards are enforced by structure rather than remembered at each call
site.

These tests hold that structure in place. The behavioral ones check that each
entry point is in fact gated; the static ones check that no *new* entry point
can quietly appear beside the engine rather than through it -- which is how the
gate came to be missed on the candidate-``accept`` path once already.
"""

from __future__ import annotations

import ast
import inspect
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Set

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl
from ftmwpipeline.core.environment import ANALYSIS_EPOCH, capture_environment

pytestmark = [pytest.mark.integration]

# The module functions that write through the engine. Kept as a literal list
# so that adding an entry point without deciding how it is gated fails
# ``test_no_engine_entry_point_is_unlisted`` rather than passing silently.
ENGINE_ENTRY_POINTS: Set[str] = {
    "refit_window_impl",
    "merge_peaks_impl",
    "split_peak_impl",
    "review_accept_impl",
    "create_window_impl",
    "apply_curation_impl",
    "review_undo_impl",
    "_apply_curation_at_prefix",
    "_review_run",
    "_run_review_preview",
}


def _module_ast() -> ast.Module:
    return ast.parse(inspect.getsource(stage6_impl))


def _functions_calling(names: Set[str]) -> Dict[str, List[str]]:
    """Map each module-level function to the sought call names it makes.

    Nested ``def``/``lambda`` bodies count toward their enclosing module-level
    function, which is what the single-verb wrappers use.
    """
    found: Dict[str, List[str]] = {}
    for node in _module_ast().body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        hits = [
            sub.func.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id in names
        ]
        if hits:
            found[node.name] = hits
    return found


class TestEngineIsSingleSourced:
    def test_only_finish_batch_persists_the_fit(self):
        """One writer for ``/stage5_fitting``, reachable only through the engine.

        A second place that persisted a fit could skip the replay, the review
        refresh, or the gate -- so there is exactly one, and it is the write
        path's own closing step.
        """
        writers = set(
            _functions_calling(
                {"save_spectrum_fit_to_hdf5", "update_spectrum_fit_windows_in_hdf5"}
            )
        )
        assert writers == {"_finish_batch"}, (
            "Stage 6 must persist /stage5_fitting in exactly one place "
            f"(_finish_batch); found writers: {sorted(writers)}"
        )

    def test_only_open_batch_takes_the_undo_baseline(self):
        """The baseline snapshot is the engine's, so it cannot be forgotten."""
        snappers = set(_functions_calling({"_snapshot_stage5_baseline"}))
        assert snappers == {"_open_batch"}, (
            "the undo baseline must be taken in exactly one place (_open_batch); "
            f"found: {sorted(snappers)}"
        )

    def test_curate_is_the_only_path_to_finish_batch(self):
        """Every persisted state is one ``_curate`` computed: a function that
        persists curates first, or persists a session's staged preview (itself
        a ``_curate`` result, made by ``_curate_request``)."""
        persisters = set(_functions_calling({"_finish_batch"}))
        curators = set(_functions_calling({"_curate"}))
        stray = persisters - curators - {"_persist_staged"}
        assert (
            not stray
        ), f"these functions persist a state they did not curate: {sorted(stray)}"

    def test_no_engine_entry_point_is_unlisted(self):
        """Every function that writes through the engine is covered by the gate
        test below.

        A new verb routed through the engine is gated automatically -- but it
        still has to be listed here, so that its gating is asserted rather than
        assumed.
        """
        openers = set(
            _functions_calling({"_curate_request", "_curate", "_single_action_refit"})
        )
        # The two shared helpers every verb reaches the engine through.
        openers -= {"_curate_request", "_single_action_refit"}
        unlisted = openers - ENGINE_ENTRY_POINTS
        assert not unlisted, (
            f"these functions write through the Stage 6 engine but are not "
            f"listed in ENGINE_ENTRY_POINTS, so nothing asserts they are "
            f"epoch-gated: {sorted(unlisted)}"
        )

    def test_the_epoch_gate_is_enforced_by_the_engine(self):
        """The gate lives in ``_curate``, which applies it to a write that
        refits; a dry run that would install a window reports its structure
        through the same gate."""
        gaters = set(_functions_calling({"require_splice_compatible_environment"}))
        assert gaters == {"_curate", "_resolve_created_window_structure"}, (
            "the epoch gate belongs to _curate (plus the dry run's structural "
            f"report of a create); found: {sorted(gaters)}"
        )


#: Every Stage 6 write: the engine entry points that are public verbs, plus
#: ``review run`` and a ReviewSession's persist of a staged preview.
WRITE_ENTRY_POINTS: Set[str] = (
    ENGINE_ENTRY_POINTS
    - {"_apply_curation_at_prefix", "_review_run", "_run_review_preview"}
) | {
    "review_run_impl",
    "_persist_staged",
    "_apply",
}


class TestEveryWriteRefusesAPreEngineFile:
    def test_every_write_entry_point_calls_the_engine_gate(self):
        """Refuse-and-flag is structural: each write checks the file before it
        resolves, fits or writes anything. (A log-prefix apply is reached
        only through ``apply_curation_impl``, which checks first.)"""
        tree = _module_ast()
        callers: Set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_require_engine_file"
                for sub in ast.walk(node)
            ):
                callers.add(node.name)
        missing = WRITE_ENTRY_POINTS - callers
        assert not missing, (
            "these Stage 6 writes do not call _require_engine_file, so a "
            f"pre-engine file could be written: {sorted(missing)}"
        )

    def test_a_session_apply_gates_before_it_resolves(self):
        """``ReviewSession._apply`` resolves the request on its staged-reuse
        path, so the gate must be its first statement, not only the
        ``_persist_staged`` it reaches after resolving."""
        session = next(
            node
            for node in _module_ast().body
            if isinstance(node, ast.ClassDef) and node.name == "ReviewSession"
        )
        apply = next(
            node
            for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply"
        )
        body = apply.body[1:] if ast.get_docstring(apply) else apply.body
        first = body[0]
        assert (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Call)
            and isinstance(first.value.func, ast.Name)
            and first.value.func.id == "_require_engine_file"
        ), "ReviewSession._apply must call _require_engine_file first"


def _force_fit_epoch(path: Path, epoch: int) -> None:
    """Rewrite the recorded Stage 5 epoch, simulating a fit from another epoch."""
    with h5py.File(path, "a") as f:
        g = f.require_group("pipeline_stages")
        blob = json.loads(str(g.attrs.get("stage_environments", "{}")))
        rec = capture_environment().to_dict()
        rec["analysis_epoch"] = epoch
        blob["stage5_fitting"] = rec
        g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)


class TestEveryEntryPointIsGated:
    """Each fit-mutating entry point refuses across an epoch change, and a
    refusal writes nothing at all."""

    @pytest.fixture
    def fitted(self, stage5_small_source, tmp_path) -> Path:
        dst = tmp_path / "fitted.ftmw"
        shutil.copy(stage5_small_source, dst)
        return dst

    @staticmethod
    def _a_fitted_peak(path: Path):
        from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

        with h5py.File(path, "r") as f:
            sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
        wf = next(w for w in sf.window_fits if len(w.fitted_peaks) >= 2)
        return int(wf.window_id), [float(p.frequency_mhz) for p in wf.fitted_peaks]

    @staticmethod
    def _just_past_a_live_window(path: Path) -> float:
        """Two active-FT bins above the highest live window's upper edge: a
        valid create anchor (its structural checks pass, so the write reaches
        the gate)."""
        from ftmwpipeline._internal.stage4_impl import load_windows_impl
        from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

        with h5py.File(path, "r") as f:
            sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
        live = {int(w.window_id) for w in sf.window_fits if w.window_id is not None}
        plan = load_windows_impl(str(path))["plan"]
        top = max(
            (w for w in plan.windows if int(w.window_id) in live),
            key=lambda w: max(w.freq_range),
        )
        lo, hi = min(top.freq_range), max(top.freq_range)
        span = top.diagnostics["grid_span"]
        return hi + 2 * (hi - lo) / (int(span[1]) - int(span[0]))

    def _operations(
        self, path: Path, tmp_path: Path
    ) -> Dict[str, Callable[[], object]]:
        wid, freqs = self._a_fitted_peak(path)
        p = str(path)

        edit_csv = tmp_path / "edit.csv"
        edit_csv.write_text(f"remove,{wid},{freqs[0]:.6f},\n")
        accept_csv = tmp_path / "accept.csv"
        accept_csv.write_text(f"accept,{wid},,candidate={freqs[0]:.6f}\n")

        return {
            "review_edit": lambda: ftmw.review_edit(p, wid, remove=[freqs[0]]),
            # merge_peaks_impl / split_peak_impl are not CLI/api/Pipeline verbs
            # any more (curation-intent inference and decision-log replay are
            # the only callers left), but they are still listed
            # ENGINE_ENTRY_POINTS and so must still be independently gated --
            # called directly against the internal impl rather than through
            # ftmw.review_merge/review_split, which no longer exist.
            "merge_peaks_impl": lambda: stage6_impl.merge_peaks_impl(p, wid, freqs[:2]),
            "split_peak_impl": lambda: stage6_impl.split_peak_impl(p, wid, freqs[0]),
            "review_accept_candidate": lambda: ftmw.review_accept(
                p, wid, candidate_freq=freqs[0]
            ),
            "review_create": lambda: ftmw.review_create(
                p, self._just_past_a_live_window(path)
            ),
            "review_apply_edit": lambda: ftmw.review_apply(p, edit_csv),
            "review_apply_candidate_accept": lambda: ftmw.review_apply(p, accept_csv),
            "review_preview_edit": lambda: ftmw.review_preview(p, edit_csv),
        }

    def test_every_operation_is_gated_and_writes_nothing(self, fitted, tmp_path):
        ops = self._operations(fitted, tmp_path)
        # Every listed engine entry point is exercised, except review_undo_impl
        # and _apply_curation_at_prefix, which need recorded decisions and are
        # covered in test_environment_gate, and _review_run, which refits
        # nothing and so is not gated (below).
        assert len(ops) >= len(ENGINE_ENTRY_POINTS) - 3

        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        for name, op in ops.items():
            with pytest.raises(ValueError, match="analysis epoch"):
                op()
            with h5py.File(fitted, "r") as f:
                assert "stage5_fitting_baseline" not in f, (
                    f"{name} was refused but still took the undo baseline; a "
                    f"refused edit must leave the file untouched"
                )
            assert ftmw.review_log(str(fitted)) == [], f"{name} recorded a decision"

    def test_a_bare_accept_is_not_gated(self, fitted):
        """It changes no fitted number, so it is not a splice and is not gated."""
        wid, _ = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        assert ftmw.review_accept(str(fitted), wid) is None
        assert [e.kind for e in ftmw.review_log(str(fitted))] == ["accept"]

    def test_writes_that_refit_nothing_are_not_gated(self, fitted):
        """``review run``, a bare accept and an undo of every fit-changing
        decision refit nothing, so they stay available on an unacknowledged
        file; an edit, which refits, is refused. A write that refits nothing
        keeps the fits it finds."""
        wid, freqs = self._a_fitted_peak(fitted)
        p = str(fitted)
        ftmw.review_edit(p, wid, remove=[freqs[0]], frame="raw")
        ftmw.review_accept(p, wid)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)

        with h5py.File(fitted, "r") as f:
            fit_before = f["stage5_fitting/peaks/frequency_mhz"][...].tobytes()
        ftmw.review_run(p, kappa=7.0)
        ftmw.review_accept(p, wid)
        with h5py.File(fitted, "r") as f:
            assert f["stage5_fitting/peaks/frequency_mhz"][...].tobytes() == fit_before
        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_edit(p, wid, remove=[freqs[1]], frame="raw")
        edits = [e.serial for e in ftmw.review_log(p) if e.kind != "accept"]
        ftmw.review_undo(p, edits)
        assert [e.kind for e in ftmw.review_log(p)] == ["accept", "accept"]
        with h5py.File(fitted, "r") as f:
            restored = f["stage5_fitting/peaks/frequency_mhz"][...].tobytes()
            baseline = f["stage5_fitting_baseline/peaks/frequency_mhz"][...].tobytes()
        assert restored == baseline


class TestArgumentChecksPrecedeAnyWrite:
    """A malformed call must be refused before the engine touches the file."""

    @pytest.fixture
    def fitted(self, stage5_small_source, tmp_path) -> Path:
        dst = tmp_path / "fitted.ftmw"
        shutil.copy(stage5_small_source, dst)
        return dst

    @staticmethod
    def _a_window(path: Path) -> int:
        from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

        with h5py.File(path, "r") as f:
            sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
        return int(next(w for w in sf.window_fits if w.fitted_peaks).window_id)

    @pytest.mark.parametrize(
        "name,call",
        [
            (
                "merge_needs_two",
                lambda p, w: stage6_impl.merge_peaks_impl(p, w, [1.0]),
            ),
            (
                "split_needs_two",
                lambda p, w: stage6_impl.split_peak_impl(p, w, 1.0, into=1),
            ),
        ],
    )
    def test_bad_arity_writes_nothing(self, fitted, name, call):
        wid = self._a_window(fitted)
        with pytest.raises(ValueError):
            call(str(fitted), wid)
        with h5py.File(fitted, "r") as f:
            assert "stage5_fitting_baseline" not in f, (
                f"{name}: the call was rejected but the undo baseline was "
                f"already taken"
            )
