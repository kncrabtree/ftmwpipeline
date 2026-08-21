"""
Unit / integration tests for Stage 6 decision recording and review_accept.

Tests cover:
- review edit → decision log entries with kind="add" and kind="remove"
- merge_peaks_impl → decision log entry with kind="merge"
- split_peak_impl → decision log entry with kind="split"
- review_accept_impl (no candidate) → provenance="reviewed", kind="accept" entry
- review_accept_impl (with candidate) → delegates to add, provenance="user-edited"
- HDF5 round-trip for decision log entries
- Cross-interface: api.review_accept / Pipeline.review_accept
- Cross-interface: api.get_review_status / Pipeline.review_status
- review show --attention smoke test
- review show --window N --output PATH creates file

The ``stage5_small_file`` fixture (3-window 2638 subset) is used for tests that
only need at least one window with a peak.  ``stage5_multi_peak_file`` (full
2638 fit reused from a scratch artefact) is used for tests that need a window
with >=2 peaks; those tests are skipped when the artefact is absent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    RefitWindowResult,
    merge_peaks_impl,
    refit_window_impl,
    review_accept_impl,
    split_peak_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_show
from ftmwpipeline.core.data_structures import (
    DecisionLogEntry,
    FittingResult,
    SpectrumFit,
    Stage6Review,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_file,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors test_review_verbs.py)
# ---------------------------------------------------------------------------


_CROSS_FIXTURE_2638 = Path("scratch/issue3-cross-fixture/2638/exp_2638.ftmw")


@pytest.fixture(scope="module")
def stage5_multi_peak_file(tmp_path_factory):
    """Full 2638 Stage-5 fit for tests needing multi-peak windows."""
    src = _CROSS_FIXTURE_2638
    if not src.exists():
        pytest.skip(
            "Cross-fixture 2638 file not found at scratch/issue3-cross-fixture/2638/."
            " Build it via 'ftmwpipeline fit run' on that experiment first."
        )
    tmp = tmp_path_factory.mktemp("stage5_multi_peak_decisions")
    fp = tmp / "2638_multi_peak_decisions.ftmw"
    shutil.copy(src, fp)
    return fp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sf(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _first_window_with_n_peaks(sf: SpectrumFit, n: int) -> Optional[FittingResult]:
    for wf in sf.window_fits:
        if len(wf.fitted_peaks) >= n:
            return wf
    return None


def _load_review(path: Path) -> Stage6Review:
    return load_stage6_review_from_file(str(path))


# ---------------------------------------------------------------------------
# Decision log: add
# ---------------------------------------------------------------------------


class TestEditRecordsAddDecision:
    """review edit add → decision log entry kind='add', provenance='user-edited'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_multi_peak_file, tmp_path):
        self.path = tmp_path / "edit_add_decision.ftmw"
        shutil.copy(stage5_multi_peak_file, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_edit_records_add_entry(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        center: Optional[float] = None
        if self.wf.window and self.wf.window.freq_range:
            lo, hi = self.wf.window.freq_range
            center = (lo + hi) / 2.0
        if center is None:
            pytest.skip("Window has no freq_range")

        ftmw.review_edit(self.path, wid, add=[center])
        review = _load_review(self.path)
        assert len(review.decision_log) >= 1
        add_entries = [
            e for e in review.decision_log if e.kind == "add" and e.window_id == wid
        ]
        assert (
            len(add_entries) >= 1
        ), f"Expected kind='add' entry; got {[e.kind for e in review.decision_log]}"
        entry = add_entries[0]
        assert entry.provenance == "user"
        assert entry.window_id == wid

    def test_edit_add_provenance_user_edited(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        center: Optional[float] = None
        if self.wf.window and self.wf.window.freq_range:
            lo, hi = self.wf.window.freq_range
            center = (lo + hi) / 2.0
        if center is None:
            pytest.skip("Window has no freq_range")

        ftmw.review_edit(self.path, wid, add=[center])
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "user-edited"

    def test_edit_no_add_no_remove_no_log_entry(self):
        """Identity refit (no add, no remove) must not produce a log entry."""
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id

        refit_window_impl(self.path, wid)
        review = _load_review(self.path)
        assert len(review.decision_log) == 0, (
            f"Identity refit should produce zero log entries; "
            f"got {review.decision_log}"
        )


# ---------------------------------------------------------------------------
# Decision log: remove
# ---------------------------------------------------------------------------


class TestEditRecordsRemoveDecision:
    """review edit remove → decision log entry kind='remove'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_multi_peak_file, tmp_path):
        self.path = tmp_path / "edit_remove_decision.ftmw"
        shutil.copy(stage5_multi_peak_file, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 2)

    def test_edit_records_remove_entry(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks")
        wid = self.wf.window_id
        target_freq = float(
            sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0].frequency_mhz
        )

        ftmw.review_edit(self.path, wid, remove=[target_freq])
        review = _load_review(self.path)
        remove_entries = [
            e for e in review.decision_log if e.kind == "remove" and e.window_id == wid
        ]
        assert (
            len(remove_entries) >= 1
        ), f"Expected kind='remove' entry; got {[e.kind for e in review.decision_log]}"

    def test_edit_remove_provenance_user_edited(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks")
        wid = self.wf.window_id
        target_freq = float(
            sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0].frequency_mhz
        )

        ftmw.review_edit(self.path, wid, remove=[target_freq])
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "user-edited"


# ---------------------------------------------------------------------------
# Decision log: merge
# ---------------------------------------------------------------------------


class TestMergeRecordsMergeDecision:
    """merge_peaks_impl → decision log entry kind='merge'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_multi_peak_file, tmp_path):
        self.path = tmp_path / "merge_decision.ftmw"
        shutil.copy(stage5_multi_peak_file, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 2)

    def test_merge_records_kind_merge(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks")
        wid = self.wf.window_id
        sorted_peaks = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        freqs = [
            float(sorted_peaks[0].frequency_mhz),
            float(sorted_peaks[1].frequency_mhz),
        ]

        merge_peaks_impl(str(self.path), wid, freqs)
        review = _load_review(self.path)
        merge_entries = [
            e for e in review.decision_log if e.kind == "merge" and e.window_id == wid
        ]
        assert (
            len(merge_entries) >= 1
        ), f"Expected kind='merge' entry; got {[e.kind for e in review.decision_log]}"
        # Evidence dict should contain chi2r metrics
        entry = merge_entries[0]
        assert "chi2r_before" in entry.evidence
        assert "chi2r_after" in entry.evidence
        assert "merged_from" in entry.evidence

    def test_merge_provenance_user_edited(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks")
        wid = self.wf.window_id
        sorted_peaks = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        freqs = [
            float(sorted_peaks[0].frequency_mhz),
            float(sorted_peaks[1].frequency_mhz),
        ]

        merge_peaks_impl(str(self.path), wid, freqs)
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "user-edited"

    def test_merge_only_one_log_entry(self):
        """merge_peaks_impl must record exactly one log entry, not one per freq."""
        if self.wf is None:
            pytest.skip("No window with >=2 peaks")
        wid = self.wf.window_id
        sorted_peaks = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        freqs = [
            float(sorted_peaks[0].frequency_mhz),
            float(sorted_peaks[1].frequency_mhz),
        ]

        merge_peaks_impl(str(self.path), wid, freqs)
        review = _load_review(self.path)
        wid_entries = [e for e in review.decision_log if e.window_id == wid]
        assert len(wid_entries) == 1, (
            f"Expected exactly 1 log entry after merge; got {len(wid_entries)}: "
            f"{[(e.kind, e.frequency_mhz) for e in wid_entries]}"
        )


# ---------------------------------------------------------------------------
# Decision log: split
# ---------------------------------------------------------------------------


class TestSplitRecordsSplitDecision:
    """split_peak_impl → decision log entry kind='split'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "split_decision.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_split_records_kind_split(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        target_freq = float(
            sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0].frequency_mhz
        )

        split_peak_impl(str(self.path), wid, target_freq, into=2)
        review = _load_review(self.path)
        split_entries = [
            e for e in review.decision_log if e.kind == "split" and e.window_id == wid
        ]
        assert (
            len(split_entries) >= 1
        ), f"Expected kind='split' entry; got {[e.kind for e in review.decision_log]}"
        entry = split_entries[0]
        assert "split_into" in entry.evidence
        assert int(entry.evidence["split_into"]) == 2  # type: ignore[arg-type]
        assert abs(entry.frequency_mhz - target_freq) < 0.1

    def test_split_provenance_user_edited(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        target_freq = float(
            sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0].frequency_mhz
        )

        split_peak_impl(str(self.path), wid, target_freq, into=2)
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "user-edited"

    def test_split_only_one_log_entry(self):
        """split_peak_impl must record exactly one log entry."""
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        target_freq = float(
            sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0].frequency_mhz
        )

        split_peak_impl(str(self.path), wid, target_freq, into=2)
        review = _load_review(self.path)
        wid_entries = [e for e in review.decision_log if e.window_id == wid]
        assert (
            len(wid_entries) == 1
        ), f"Expected exactly 1 log entry after split; got {len(wid_entries)}"


# ---------------------------------------------------------------------------
# review_accept_impl: no candidate
# ---------------------------------------------------------------------------


class TestReviewAcceptNoCandidate:
    """review_accept_impl (no candidate) → provenance='reviewed', kind='accept'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "accept_no_cand.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_accept_returns_none(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id

        result = review_accept_impl(str(self.path), wid)
        assert result is None

    def test_accept_provenance_reviewed(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id

        review_accept_impl(str(self.path), wid)
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "reviewed"

    def test_accept_records_kind_accept(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id

        review_accept_impl(str(self.path), wid)
        review = _load_review(self.path)
        accept_entries = [
            e for e in review.decision_log if e.kind == "accept" and e.window_id == wid
        ]
        assert (
            len(accept_entries) >= 1
        ), f"Expected kind='accept' entry; got {[e.kind for e in review.decision_log]}"

    def test_accept_fit_unchanged(self):
        """Accepting as-is must not modify the fitted peaks."""
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        n_before = len(self.wf.fitted_peaks)

        review_accept_impl(str(self.path), wid)
        sf_after = _load_sf(self.path)
        wf_after = next(w for w in sf_after.window_fits if w.window_id == wid)
        assert len(wf_after.fitted_peaks) == n_before


# ---------------------------------------------------------------------------
# review_accept_impl: with candidate
# ---------------------------------------------------------------------------


class TestReviewAcceptWithCandidate:
    """review_accept_impl(candidate_freq=F) → delegates to add, provenance='user-edited'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "accept_with_cand.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_accept_with_candidate_returns_result(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        center: Optional[float] = None
        if self.wf.window and self.wf.window.freq_range:
            lo, hi = self.wf.window.freq_range
            center = (lo + hi) / 2.0
        if center is None:
            pytest.skip("Window has no freq_range")

        result = review_accept_impl(str(self.path), wid, candidate_freq=center)
        assert isinstance(result, RefitWindowResult)

    def test_accept_with_candidate_provenance_user_edited(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        center: Optional[float] = None
        if self.wf.window and self.wf.window.freq_range:
            lo, hi = self.wf.window.freq_range
            center = (lo + hi) / 2.0
        if center is None:
            pytest.skip("Window has no freq_range")

        review_accept_impl(str(self.path), wid, candidate_freq=center)
        review = _load_review(self.path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "user-edited"

    def test_accept_with_candidate_records_add_entry(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        center: Optional[float] = None
        if self.wf.window and self.wf.window.freq_range:
            lo, hi = self.wf.window.freq_range
            center = (lo + hi) / 2.0
        if center is None:
            pytest.skip("Window has no freq_range")

        review_accept_impl(str(self.path), wid, candidate_freq=center)
        review = _load_review(self.path)
        add_entries = [
            e for e in review.decision_log if e.kind == "add" and e.window_id == wid
        ]
        assert len(add_entries) >= 1


# ---------------------------------------------------------------------------
# HDF5 round-trip for decision log
# ---------------------------------------------------------------------------


class TestDecisionLogRoundtripHDF5:
    """Decision log entries survive a save/load round-trip through HDF5."""

    def test_roundtrip(self, tmp_path):
        from ftmwpipeline.core.data_structures import (
            AttentionReason,
            WindowReviewStatus,
        )

        entry = DecisionLogEntry(
            order_index=0,
            window_id=42,
            frequency_mhz=27500.1234,
            kind="merge",
            provenance="user",
            evidence={"chi2r_before": 1.5, "chi2r_after": 1.1, "split_into": 2},
        )
        status = WindowReviewStatus(
            window_id=42,
            provenance="user-edited",
            attention_reasons=[
                AttentionReason(kind="worst_eps", detail="chi2r too high", severity=3.5)
            ],
            invalidated=False,
        )
        review = Stage6Review(
            window_statuses={42: status},
            decision_log=[entry],
        )

        h5_path = tmp_path / "roundtrip.h5"
        with h5py.File(str(h5_path), "w") as h5f:
            grp = h5f.create_group("stage6_review")
            save_stage6_review_to_hdf5(review, grp)

        with h5py.File(str(h5_path), "r") as h5f:
            loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])

        assert len(loaded.decision_log) == 1
        e = loaded.decision_log[0]
        assert e.kind == "merge"
        assert e.window_id == 42
        assert abs(e.frequency_mhz - 27500.1234) < 1e-9
        assert e.provenance == "user"
        assert "chi2r_before" in e.evidence

        assert 42 in loaded.window_statuses
        s = loaded.window_statuses[42]
        assert s.provenance == "user-edited"
        assert len(s.attention_reasons) == 1
        assert s.attention_reasons[0].kind == "worst_eps"


# ---------------------------------------------------------------------------
# Cross-interface: review_accept
# ---------------------------------------------------------------------------


class TestCrossInterfaceAccept:
    """api.review_accept and Pipeline.review_accept both mark provenance='reviewed'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_file, tmp_path):
        self.stage5_small = stage5_small_file
        self.tmp = tmp_path
        self.sf = _load_sf(stage5_small_file)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_api_accept_sets_reviewed(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        path = self.tmp / "api_accept.ftmw"
        shutil.copy(self.stage5_small, path)

        result = ftmw.review_accept(path, wid)
        assert result is None
        review = _load_review(path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "reviewed"

    def test_pipeline_accept_sets_reviewed(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        path = self.tmp / "pip_accept.ftmw"
        shutil.copy(self.stage5_small, path)

        result = Pipeline.open(path).review_accept(wid)
        assert result is None
        review = _load_review(path)
        status = review.window_statuses.get(wid)
        assert status is not None
        assert status.provenance == "reviewed"


# ---------------------------------------------------------------------------
# Cross-interface: review_status / get_review_status
# ---------------------------------------------------------------------------


class TestCrossInterfaceReviewStatus:
    """api.get_review_status and Pipeline.review_status return the same state."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "review_status.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)

    def test_both_interfaces_return_same_provenance(self):
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id

        # Write a provenance entry via accept.
        review_accept_impl(str(self.path), wid)

        api_review = ftmw.get_review_status(self.path)
        pip_review = Pipeline.open(self.path).review_status()

        api_status = api_review.window_statuses.get(wid)
        pip_status = pip_review.window_statuses.get(wid)

        assert api_status is not None
        assert pip_status is not None
        assert api_status.provenance == pip_status.provenance == "reviewed"

    def test_empty_review_returns_stage6review(self):
        """get_review_status before review run returns an empty Stage6Review."""
        api_review = ftmw.get_review_status(self.path)
        assert isinstance(api_review, Stage6Review)
        assert len(api_review.decision_log) == 0


# ---------------------------------------------------------------------------
# review show --attention smoke test
# ---------------------------------------------------------------------------


class TestReviewShowAttentionSmoke:
    """review show --attention does not error after review run."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "show_attention.ftmw"
        shutil.copy(stage5_small_source, self.path)

    def test_show_attention_no_error(self):
        ftmw.review_run(self.path)
        args = argparse.Namespace(
            file_path=str(self.path),
            window=None,
            candidates=False,
            bar=4.0,
            verbose=False,
            attention=True,
            output=None,
        )
        ret = cmd_review_show(args)
        assert ret == 0

    def test_show_summary_includes_label(self, capsys):
        ftmw.review_run(self.path)
        args = argparse.Namespace(
            file_path=str(self.path),
            window=None,
            candidates=False,
            bar=4.0,
            verbose=False,
            attention=False,
            output=None,
        )
        ret = cmd_review_show(args)
        assert ret == 0
        out = capsys.readouterr().out
        # The summary table must now include a "label" column header.
        assert "label" in out


# ---------------------------------------------------------------------------
# review show --window N --output PATH
# ---------------------------------------------------------------------------


class TestReviewShowOutputFile:
    """review show --window N --output PATH creates the file."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "show_output.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_sf(self.path)
        self.wf = _first_window_with_n_peaks(self.sf, 1)
        self.tmp = tmp_path

    def test_output_creates_file(self):
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        if self.wf is None:
            pytest.skip("No window with peaks")
        wid = self.wf.window_id
        out_path = self.tmp / "window_fit.png"

        args = argparse.Namespace(
            file_path=str(self.path),
            window=wid,
            candidates=False,
            bar=4.0,
            verbose=False,
            attention=False,
            output=str(out_path),
        )
        ret = cmd_review_show(args)
        assert ret == 0
        assert out_path.exists(), f"Expected output file at {out_path}"
