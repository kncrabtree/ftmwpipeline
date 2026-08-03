"""
Stage 6 per-peak derivation tag (issue #41).

``FittedPeak.derivation`` / ``FinalPeak.derivation`` name the curation decision
that created or altered a peak's *identity*; ``None`` means the peak carried
through the refit unchanged.  The point of the tag is that a consumer binding
external state to individual peaks (line assignments, say) can read the
derivation instead of reconstructing it by pairing peak sets across an edit.

The contract these tests pin down:

- an automatic fit tags nothing (every peak is identity-preserved);
- an ``add`` tags exactly the created peak with the ``order_index`` of the
  decision that created it, and leaves its neighbours in the window untagged;
- merge / split products carry the coarser decision's id;
- the tag survives the HDF5 round trip and reaches ``FinalProducts``;
- ``review undo`` renumbers the log and the tags together, so a tag always
  indexes a decision that is actually in the log.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import pytest

from ftmwpipeline._internal.stage6_impl import (
    get_final_products_impl,
    merge_peaks_impl,
    refit_window_impl,
    review_log_impl,
    review_run_impl,
    review_undo_impl,
    split_peak_impl,
)
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _window_with_range(sf: SpectrumFit):
    return next(
        (
            w
            for w in sf.window_fits
            if w.window_id is not None
            and w.window is not None
            and w.window.freq_range is not None
        ),
        None,
    )


@pytest.fixture
def working_file(stage5_small_file, tmp_path) -> Path:
    dst = tmp_path / "working.ftmw"
    shutil.copy(stage5_small_file, dst)
    return dst


class TestAutomaticFitCarriesNoTag:
    def test_every_automatic_peak_is_untagged(self, working_file):
        sf = _load_spectrum_fit(working_file)
        assert sf.fitted_peaks, "fixture has no fitted peaks"
        assert all(p.derivation is None for p in sf.fitted_peaks)

    def test_identity_refit_tags_nothing(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        refit_window_impl(str(working_file), wf.window_id, add=[], remove=[])
        after = _load_spectrum_fit(working_file)
        assert all(p.derivation is None for p in after.fitted_peaks)


class TestAddTagsOnlyTheCreatedPeak:
    def test_added_peak_carries_the_decision_id(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        center = 0.5 * (lo + hi)

        result = refit_window_impl(str(working_file), wf.window_id, add=[center])

        log = review_log_impl(str(working_file))
        assert len(log) == 1 and log[0].kind == "add"
        expected = log[0].order_index

        tagged = [p for p in result.fitted_peaks if p.derivation is not None]
        assert len(tagged) == 1, (
            "exactly the created peak should be tagged; got "
            f"{[(p.frequency_mhz, p.derivation) for p in result.fitted_peaks]}"
        )
        assert tagged[0].derivation == expected
        # The tag and the user-origin flag mark the same peak.
        assert tagged[0].origin == "user"

    def test_untouched_peaks_stay_untagged(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        result = refit_window_impl(
            str(working_file), wf.window_id, add=[0.5 * (lo + hi)]
        )
        untagged = [p for p in result.fitted_peaks if p.derivation is None]
        assert len(untagged) == len(result.fitted_peaks) - 1

    def test_two_adds_in_one_edit_get_distinct_ids(self, working_file):
        """One decision is recorded per add frequency, so the tags differ."""
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        span = hi - lo
        result = refit_window_impl(
            str(working_file),
            wf.window_id,
            add=[lo + 0.3 * span, lo + 0.7 * span],
        )
        log = review_log_impl(str(working_file))
        assert [e.kind for e in log] == ["add", "add"]
        tags = sorted(
            p.derivation for p in result.fitted_peaks if p.derivation is not None
        )
        assert tags == [e.order_index for e in log]

    def test_tag_survives_the_hdf5_round_trip(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        refit_window_impl(str(working_file), wf.window_id, add=[0.5 * (lo + hi)])

        reloaded = _load_spectrum_fit(working_file)
        tagged = [p for p in reloaded.fitted_peaks if p.derivation is not None]
        assert len(tagged) == 1
        assert tagged[0].derivation == 0

    def test_second_edit_leaves_the_first_tag_intact(self, working_file):
        """A later refit re-converges the earlier peak but does not re-identify
        it, so its tag still names the decision that created it."""
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        span = hi - lo
        refit_window_impl(str(working_file), wf.window_id, add=[lo + 0.3 * span])
        result = refit_window_impl(
            str(working_file), wf.window_id, add=[lo + 0.7 * span]
        )
        tags = sorted(
            p.derivation for p in result.fitted_peaks if p.derivation is not None
        )
        assert tags == [0, 1]


class TestSplitProductsCarryTheSplitDecision:
    def test_split_products_share_the_split_decision_id(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = next(
            (w for w in sf.window_fits if w.window_id is not None and w.fitted_peaks),
            None,
        )
        if wf is None:
            pytest.skip("no window with a fitted peak")
        target = float(wf.fitted_peaks[0].frequency_mhz)

        result = split_peak_impl(str(working_file), wf.window_id, target, into=2)

        log = review_log_impl(str(working_file))
        assert len(log) == 1 and log[0].kind == "split"
        tagged = [p for p in result.fitted_peaks if p.derivation is not None]
        assert len(tagged) == 2
        assert {p.derivation for p in tagged} == {log[0].order_index}


class TestMergeProductCarriesTheMergeDecision:
    def test_merge_product_tagged(self, working_file):
        sf = _load_spectrum_fit(working_file)
        wf = next(
            (
                w
                for w in sf.window_fits
                if w.window_id is not None and len(w.fitted_peaks) >= 2
            ),
            None,
        )
        if wf is None:
            pytest.skip("no window with >= 2 fitted peaks")
        freqs = [float(p.frequency_mhz) for p in wf.fitted_peaks[:2]]

        result = merge_peaks_impl(str(working_file), wf.window_id, freqs)

        log = review_log_impl(str(working_file))
        assert len(log) == 1 and log[0].kind == "merge"
        tagged = [p for p in result.fitted_peaks if p.derivation is not None]
        assert len(tagged) == 1
        assert tagged[0].derivation == log[0].order_index


class TestFinalProductsCarryTheTag:
    def test_final_peak_derivation_matches_the_fitted_peak(self, working_file):
        review_run_impl(str(working_file))
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        refit_window_impl(str(working_file), wf.window_id, add=[0.5 * (lo + hi)])

        products = get_final_products_impl(str(working_file))
        assert products is not None
        tagged = [p for p in products.peaks if p.derivation is not None]
        assert len(tagged) == 1
        assert tagged[0].derivation == 0

        # And the tag indexes a real decision.
        log = review_log_impl(str(working_file))
        assert any(e.order_index == tagged[0].derivation for e in log)

    def test_pre_curation_table_is_all_none(self, working_file):
        review_run_impl(str(working_file))
        products = get_final_products_impl(str(working_file))
        assert products is not None
        assert all(p.derivation is None for p in products.peaks)


class TestUndoRenumbersTagsWithTheLog:
    def test_tag_still_indexes_a_live_decision_after_undo(self, working_file):
        review_run_impl(str(working_file))
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        span = hi - lo
        refit_window_impl(str(working_file), wf.window_id, add=[lo + 0.3 * span])
        refit_window_impl(str(working_file), wf.window_id, add=[lo + 0.7 * span])
        assert len(review_log_impl(str(working_file))) == 2

        # Drop the FIRST decision; the second survives and is replayed, so it
        # renumbers from 1 to 0 -- and its peak's tag must follow.
        review_undo_impl(str(working_file), [0])

        log = review_log_impl(str(working_file))
        assert len(log) == 1
        live_ids = {e.order_index for e in log}
        after = _load_spectrum_fit(working_file)
        tags = [p.derivation for p in after.fitted_peaks if p.derivation is not None]
        assert tags, "the surviving decision's peak lost its tag on replay"
        assert set(tags) <= live_ids

    def test_undoing_everything_clears_all_tags(self, working_file):
        review_run_impl(str(working_file))
        sf = _load_spectrum_fit(working_file)
        wf = _window_with_range(sf)
        assert wf is not None
        lo, hi = wf.window.freq_range
        refit_window_impl(str(working_file), wf.window_id, add=[0.5 * (lo + hi)])

        review_undo_impl(str(working_file), [0])

        assert review_log_impl(str(working_file)) == []
        after = _load_spectrum_fit(working_file)
        assert all(p.derivation is None for p in after.fitted_peaks)
