"""
The analysis-epoch gate on Stage 6 splice operations.

A Stage 6 edit re-fits *one* window and writes it back into a ``SpectrumFit``
whose other windows were fit earlier, and the cascade then re-fits the
dependents. If the fitting code changed in between, the result holds two
different models inside one product -- a mixture no per-file version stamp can
express and no report can honestly caveat, because it is *within* the artifact.

That makes splicing the one operation class the environment policy blocks.
Extending a file forward only warns; reads are never gated. The gate keys on
:data:`~ftmwpipeline.core.environment.ANALYSIS_EPOCH` alone, and an unknown
epoch (a fit written before environment stamping) is treated as compatible so an
upgrade never strands an existing file.

The override is a persisted acknowledgement rather than a per-call flag, so a
file curated across an epoch boundary carries that fact in its own record.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.core.environment as envmod
from ftmwpipeline.core.environment import ANALYSIS_EPOCH, capture_environment
from ftmwpipeline.file_manager import AnalysisEpochMismatchError, PipelineFileError

pytestmark = [pytest.mark.integration]


def _force_fit_epoch(path: Path, epoch) -> None:
    """Rewrite the recorded Stage 5 epoch, simulating a fit from another epoch."""
    with h5py.File(path, "a") as f:
        g = f.require_group("pipeline_stages")
        blob = json.loads(str(g.attrs.get("stage_environments", "{}")))
        rec = capture_environment().to_dict()
        rec["analysis_epoch"] = epoch
        blob["stage5_fitting"] = rec
        g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)


class TestSpliceGate:
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
        wf = next(w for w in sf.window_fits if w.fitted_peaks)
        return int(wf.window_id), float(wf.fitted_peaks[0].frequency_mhz)

    def test_legacy_fit_is_not_blocked(self, fitted):
        """A fit written before stamping has an unknown epoch; editing it must
        still work, or an upgrade would strand every existing file."""
        wid, freq = self._a_fitted_peak(fitted)
        result = ftmw.review_edit(str(fitted), wid, remove=[freq])
        assert result.n_peaks_after == result.n_peaks_before - 1

    def test_edit_blocked_across_an_epoch_change(self, fitted):
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_edit(str(fitted), wid, remove=[freq])

    def test_create_blocked_across_an_epoch_change(self, fitted):
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_create(str(fitted), 26622.0)

    def test_a_blocked_edit_writes_nothing(self, fitted):
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError):
            ftmw.review_edit(str(fitted), wid, remove=[freq])
        with h5py.File(fitted, "r") as f:
            assert "stage5_fitting_baseline" not in f
        assert ftmw.review_log(str(fitted)) == []

    def test_the_error_names_the_remedy(self, fitted):
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError) as exc:
            ftmw.review_edit(str(fitted), wid, remove=[freq])
        msg = str(exc.value)
        assert "fit run" in msg
        assert "acknowledge-environment" in msg

    def test_acknowledgement_unblocks(self, fitted):
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        ack = ftmw.review_acknowledge_environment(str(fitted), reason="reviewed")
        assert ack.mismatch is True
        result = ftmw.review_edit(str(fitted), wid, remove=[freq])
        assert result.n_peaks_after == result.n_peaks_before - 1

    def test_acknowledgement_reports_when_it_was_unnecessary(self, fitted):
        ack = ftmw.review_acknowledge_environment(str(fitted))
        assert ack.mismatch is False

    def test_acknowledgement_does_not_carry_to_a_later_epoch(self, fitted, monkeypatch):
        """The ack names the environment it was given under, so upgrading again
        asks again rather than inheriting a stale acceptance."""
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        ftmw.review_acknowledge_environment(str(fitted))
        monkeypatch.setattr(envmod, "ANALYSIS_EPOCH", ANALYSIS_EPOCH + 7)
        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_edit(str(fitted), wid, remove=[freq])

    def test_undo_is_gated_before_it_rolls_back(self, fitted):
        """Undo restores the baseline and then replays, so a refusal partway
        through would discard the decisions the caller asked to keep. It must
        refuse before touching anything."""
        wid, freq = self._a_fitted_peak(fitted)
        ftmw.review_edit(str(fitted), wid, remove=[freq])
        ftmw.review_edit(str(fitted), wid, add=[freq])
        assert len(ftmw.review_log(str(fitted))) == 2

        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_undo(str(fitted), [1])

        # The decision log is intact -- nothing was rolled back.
        assert len(ftmw.review_log(str(fitted))) == 2

    def test_apply_is_gated_before_it_applies_anything(self, fitted, tmp_path):
        wid, freq = self._a_fitted_peak(fitted)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"remove,{wid},{freq:.6f},\n")
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)

        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_apply(str(fitted), cf)
        assert ftmw.review_log(str(fitted)) == []

    def test_apply_of_only_candidate_accepts_is_gated(self, fitted, tmp_path):
        """An ``accept`` carrying a candidate adds a peak, so it splices.

        The plan is all-``accept``, which is what the caller-side "any
        non-accept action" pre-check reads as non-mutating -- the batch engine
        has to gate itself or this slips through and writes.
        """
        wid, freq = self._a_fitted_peak(fitted)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"accept,{wid},,candidate={freq:.6f}\n")
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)

        with pytest.raises(ValueError, match="analysis epoch"):
            ftmw.review_apply(str(fitted), cf)
        # Refused before the baseline snapshot, so nothing was written at all.
        with h5py.File(fitted, "r") as f:
            assert "stage5_fitting_baseline" not in f
        assert ftmw.review_log(str(fitted)) == []

    def test_the_refusal_carries_a_type_not_just_prose(self, fitted):
        """The refusal is routable without reading its message.

        A caller that must handle *this* refusal differently from every other
        pipeline error (offer a re-run, not a generic failure) had only the
        message text to key on before this type existed. Rewording the message
        would then silently downgrade that recovery path, and no test on the
        caller's side could catch it -- so the type is the contract and the
        prose is not.
        """
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(AnalysisEpochMismatchError) as exc:
            ftmw.review_edit(str(fitted), wid, remove=[freq])

        err = exc.value
        assert err.file_epoch == ANALYSIS_EPOCH + 1
        assert err.current_epoch == ANALYSIS_EPOCH
        assert Path(err.filepath) == fitted
        assert err.file_environment.analysis_epoch == ANALYSIS_EPOCH + 1
        assert err.current_environment.analysis_epoch == ANALYSIS_EPOCH

    def test_the_type_stays_catchable_the_old_ways(self, fitted):
        """Additive, not breaking: it is still a ValueError, and now also a
        PipelineFileError, so both existing habits keep working."""
        assert issubclass(AnalysisEpochMismatchError, ValueError)
        assert issubclass(AnalysisEpochMismatchError, PipelineFileError)

        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(ValueError):
            ftmw.review_edit(str(fitted), wid, remove=[freq])

    def test_undo_refuses_with_the_same_type(self, fitted):
        wid, freq = self._a_fitted_peak(fitted)
        ftmw.review_edit(str(fitted), wid, remove=[freq])
        ftmw.review_edit(str(fitted), wid, add=[freq])
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(AnalysisEpochMismatchError):
            ftmw.review_undo(str(fitted), [1])

    def test_the_type_survives_review_apply(self, fitted, tmp_path):
        """``review apply`` re-raises per-action failures as plain ValueError.

        The gate fires *outside* that wrapper, so the typed refusal reaches the
        caller intact. That placement is load-bearing for anyone routing on the
        type, and nothing else pins it -- moving the gate inside the batch's
        try block would downgrade this to an untyped error with the tests all
        still green.
        """
        wid, freq = self._a_fitted_peak(fitted)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"remove,{wid},{freq:.6f},\n")
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)

        with pytest.raises(AnalysisEpochMismatchError):
            ftmw.review_apply(str(fitted), cf)

    def test_the_message_is_unchanged_by_the_typing(self, fitted):
        """Typing the refusal must not reword it: the CLI prints str(exc)."""
        wid, freq = self._a_fitted_peak(fitted)
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        with pytest.raises(AnalysisEpochMismatchError) as exc:
            ftmw.review_edit(str(fitted), wid, remove=[freq])
        msg = str(exc.value)
        assert msg.startswith(
            "This file's Stage 5 fit was produced under analysis epoch "
        )
        assert "fit run" in msg
        assert "acknowledge-environment" in msg

    def test_dry_run_apply_is_not_gated(self, fitted, tmp_path):
        """A preview writes nothing, so it is a read and must never be gated."""
        wid, freq = self._a_fitted_peak(fitted)
        cf = tmp_path / "cur.csv"
        cf.write_text(f"remove,{wid},{freq:.6f},\n")
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)

        result = ftmw.review_apply(str(fitted), cf, dry_run=True)
        assert result.applied == 0
        assert result.plan

    def test_acknowledgement_is_surfaced_in_info_and_the_table(self, fitted, tmp_path):
        _force_fit_epoch(fitted, ANALYSIS_EPOCH + 1)
        ftmw.review_acknowledge_environment(str(fitted), reason="known change")

        assert ftmw.get_pipeline_info(str(fitted))["environment_acknowledged"] is True

        ftmw.review_run(str(fitted))
        csv_text = ftmw.report_table(str(fitted), fmt="csv")
        assert "environment_ack" in csv_text
        assert "known change" in csv_text
