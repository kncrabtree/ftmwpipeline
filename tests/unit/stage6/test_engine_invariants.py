"""Structural invariants of the Stage 6 fit-editing engine.

Every Stage 6 operation that changes a fitted number -- the interactive
single-window verbs, a curation file, an undo replay -- runs through one engine:
``_open_batch`` (epoch gate, undo baseline, shared fit context), the per-action
appliers, then ``_finish_batch`` (one cascade, one fit persist, one review
persist). The point of routing everything through it is that the guards are
enforced by structure rather than remembered at each call site.

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

# The module functions that open the engine. Kept as a literal list so that
# adding an entry point without deciding how it is gated fails
# ``test_no_engine_entry_point_is_unlisted`` rather than passing silently.
ENGINE_ENTRY_POINTS: Set[str] = {
    "refit_window_impl",
    "merge_peaks_impl",
    "split_peak_impl",
    "review_accept_impl",
    "create_window_impl",
    "apply_curation_impl",
    "review_undo_impl",
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

        A second place that persisted a fit could skip the cascade, the review
        refresh, or the gate -- so there is exactly one, and it is the engine's
        own closing step.
        """
        writers = set(_functions_calling({"save_spectrum_fit_to_hdf5"}))
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

    def test_no_engine_entry_point_is_unlisted(self):
        """Every function that opens a batch is covered by the gate test below.

        A new verb routed through the engine is gated automatically -- but it
        still has to be listed here, so that its gating is asserted rather than
        assumed.
        """
        openers = set(
            _functions_calling({"_run_single_action", "_execute_curation_batch"})
        )
        openers -= {"_open_batch", "_finish_batch"}
        unlisted = openers - ENGINE_ENTRY_POINTS
        assert not unlisted, (
            f"these functions open the Stage 6 engine but are not listed in "
            f"ENGINE_ENTRY_POINTS, so nothing asserts they are epoch-gated: "
            f"{sorted(unlisted)}"
        )

    def test_the_epoch_gate_is_enforced_by_the_engine(self):
        """The gate lives in ``_open_batch``; only undo may also pre-check.

        Undo restores the baseline *before* it replays, so it has to refuse
        before that rollback rather than when the replay opens its batch.
        """
        gaters = set(_functions_calling({"require_splice_compatible_environment"}))
        assert gaters == {"_open_batch", "review_undo_impl"}, (
            "the epoch gate belongs to _open_batch (plus review_undo_impl, which "
            f"must refuse before it rolls back); found: {sorted(gaters)}"
        )


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
            "review_merge": lambda: ftmw.review_merge(p, wid, freqs[:2]),
            "review_split": lambda: ftmw.review_split(p, wid, freqs[0]),
            "review_accept_candidate": lambda: ftmw.review_accept(
                p, wid, candidate_freq=freqs[0]
            ),
            "review_create": lambda: ftmw.review_create(p, freqs[0] + 0.5),
            "review_apply_edit": lambda: ftmw.review_apply(p, edit_csv),
            "review_apply_candidate_accept": lambda: ftmw.review_apply(p, accept_csv),
        }

    def test_every_operation_is_gated_and_writes_nothing(self, fitted, tmp_path):
        ops = self._operations(fitted, tmp_path)
        # Every listed engine entry point is exercised, except review_undo_impl,
        # which needs recorded decisions and is covered in test_environment_gate.
        assert len(ops) >= len(ENGINE_ENTRY_POINTS) - 1

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
            ("merge_needs_two", lambda p, w: ftmw.review_merge(p, w, [1.0])),
            ("split_needs_two", lambda p, w: ftmw.review_split(p, w, 1.0, into=1)),
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
