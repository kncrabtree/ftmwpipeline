"""
Unit / integration tests for Stage 6 review_run (Pass 2 attention routing).

Tests cover:
- review_run_impl: basic run, idempotency, provenance preservation
- Stage6Review serialization round-trip
- Cross-interface consistency (api vs Pipeline)
- Attention-reason firing: edge_boundary
- candidate_bearing on a multi-peak file (skipped when absent)

All tests that invoke review_run_impl work on per-test copies of the
module-scoped stage5_small_file fixture (since review_run writes to the file).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    RANK_METRICS,
    RankedWindow,
    ReviewRunResult,
    rank_windows_impl,
    review_run_impl,
)
from ftmwpipeline.core.data_structures import (
    AttentionReason,
    DecisionLogEntry,
    Stage6Review,
    WindowReviewStatus,
)
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_file,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Shared fixture: 2638 Stage-5-small (3 windows) -- provided by conftest.py
# ---------------------------------------------------------------------------


_CROSS_FIXTURE_2638 = Path("scratch/issue3-cross-fixture/2638/exp_2638.ftmw")


@pytest.fixture(scope="module")
def stage5_multi_peak_file(tmp_path_factory):
    """Full 2638 Stage-5 fit; skip when the scratch artefact is absent."""
    src = _CROSS_FIXTURE_2638
    if not src.exists():
        pytest.skip(
            "Cross-fixture 2638 file not found at scratch/issue3-cross-fixture/2638/."
            " Build it via 'ftmwpipeline fit run' on that experiment first."
        )
    tmp = tmp_path_factory.mktemp("stage5_multi_peak_review_run")
    fp = tmp / "2638_multi_peak_review_run.ftmw"
    shutil.copy(src, fp)
    return fp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_review_run_basic(stage5_small_file, tmp_path):
    """review_run_impl completes and writes a valid Stage6Review group."""
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    result = review_run_impl(str(fp))

    assert isinstance(result, ReviewRunResult)
    assert result.n_windows > 0
    assert result.n_attention >= 0
    assert isinstance(result.reason_counts, dict)

    # HDF5 group must exist.
    with h5py.File(str(fp), "r") as h5f:
        assert "stage6_review" in h5f
        review = load_stage6_review_from_hdf5(h5f["stage6_review"])

    assert isinstance(review, Stage6Review)
    assert len(review.window_statuses) == result.n_windows
    assert len(review.decision_log) == 0


def test_review_run_idempotency(stage5_small_file, tmp_path):
    """Running review_run_impl twice yields identical window statuses."""
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    result1 = review_run_impl(str(fp))
    result2 = review_run_impl(str(fp))

    assert result1.n_windows == result2.n_windows
    assert result1.n_attention == result2.n_attention
    assert result1.reason_counts == result2.reason_counts

    review2 = load_stage6_review_from_file(str(fp))
    for wid, status in review2.window_statuses.items():
        assert status.provenance == "auto"


def test_provenance_preservation(stage5_small_file, tmp_path):
    """Provenance "reviewed" is preserved across a re-run."""
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    # First run to initialize.
    result = review_run_impl(str(fp))
    assert result.n_windows > 0

    # Manually set one window's provenance to "reviewed".
    review = load_stage6_review_from_file(str(fp))
    first_wid = next(iter(review.window_statuses))
    review.window_statuses[first_wid].provenance = "reviewed"

    # Write back the modified review.
    with h5py.File(str(fp), "a") as h5f:
        if "stage6_review" in h5f:
            del h5f["stage6_review"]
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)

    # Re-run and check provenance survived.
    review_run_impl(str(fp))
    review2 = load_stage6_review_from_file(str(fp))
    assert review2.window_statuses[first_wid].provenance == "reviewed"

    # Other windows should remain "auto".
    for wid, status in review2.window_statuses.items():
        if wid != first_wid:
            assert status.provenance == "auto"


def test_serialization_roundtrip(tmp_path):
    """Stage6Review serializes and deserializes without data loss."""
    reasons = [
        AttentionReason(kind="worst_eps", detail="chi2r=5.0 fails gate", severity=3.14),
        AttentionReason(kind="edge_boundary", detail="1 peak near edge", severity=1.0),
    ]
    statuses = {
        7: WindowReviewStatus(
            window_id=7,
            provenance="reviewed",
            attention_reasons=reasons,
            invalidated=False,
        ),
        12: WindowReviewStatus(
            window_id=12,
            provenance="auto",
            attention_reasons=[],
            invalidated=False,
        ),
    }
    log = [
        DecisionLogEntry(
            order_index=0,
            window_id=7,
            frequency_mhz=27000.1234,
            kind="add",
            provenance="user",
            evidence={"chi2r_before": 5.0, "chi2r_after": 1.2},
        )
    ]
    original = Stage6Review(window_statuses=statuses, decision_log=log)

    fp = tmp_path / "roundtrip.h5"
    with h5py.File(str(fp), "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(original, grp)

    with h5py.File(str(fp), "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])

    assert set(loaded.window_statuses.keys()) == {7, 12}

    s7 = loaded.window_statuses[7]
    assert s7.provenance == "reviewed"
    assert len(s7.attention_reasons) == 2
    assert s7.attention_reasons[0].kind == "worst_eps"
    assert abs(s7.attention_reasons[0].severity - 3.14) < 1e-9
    assert not s7.invalidated

    s12 = loaded.window_statuses[12]
    assert s12.provenance == "auto"
    assert len(s12.attention_reasons) == 0

    assert len(loaded.decision_log) == 1
    e = loaded.decision_log[0]
    assert e.window_id == 7
    assert e.kind == "add"
    assert abs(e.frequency_mhz - 27000.1234) < 1e-6
    assert e.evidence["chi2r_before"] == 5.0


def test_cross_interface(stage5_small_file, tmp_path):
    """api.review_run and Pipeline.review_run produce identical results."""
    fp_api = tmp_path / "api.ftmw"
    fp_pipe = tmp_path / "pipeline.ftmw"
    shutil.copy(stage5_small_file, fp_api)
    shutil.copy(stage5_small_file, fp_pipe)

    result_api = ftmw.review_run(str(fp_api))
    result_pipe = Pipeline.open(fp_pipe).review_run()

    assert result_api.n_windows == result_pipe.n_windows
    assert result_api.n_attention == result_pipe.n_attention
    assert result_api.reason_counts == result_pipe.reason_counts

    # Both files must have a stage6_review group.
    for fp in (fp_api, fp_pipe):
        with h5py.File(str(fp), "r") as h5f:
            assert "stage6_review" in h5f


def test_attention_reasons_edge_boundary(stage5_small_file, tmp_path):
    """edge_boundary reason fires when a peak sits at a window edge."""
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )

    fp = tmp_path / "edge_test.ftmw"
    shutil.copy(stage5_small_file, fp)

    # Load the fit, force a peak to the window edge in the first window.
    with h5py.File(str(fp), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    # Find a window with a valid freq_range and at least one fitted peak.
    target_wf = None
    for wf in sf.window_fits:
        if (
            wf.window is not None
            and wf.window.freq_range is not None
            and wf.fitted_peaks
        ):
            target_wf = wf
            break

    if target_wf is None:
        pytest.skip("No suitable window found in the 3-window fixture")

    # Move first peak to the window's lower edge.
    flo, fhi = target_wf.window.freq_range
    original_freq = target_wf.fitted_peaks[0].frequency_mhz
    # Directly set the frequency to the edge.
    target_wf.fitted_peaks[0].frequency_mhz = flo

    with h5py.File(str(fp), "a") as h5f:
        del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(sf, grp)

    result = review_run_impl(str(fp))
    review = load_stage6_review_from_file(str(fp))

    # The modified window must have an edge_boundary reason.
    wid = target_wf.window_id if target_wf.window_id is not None else -1
    assert wid in review.window_statuses
    kinds = {r.kind for r in review.window_statuses[wid].attention_reasons}
    assert (
        "edge_boundary" in kinds
    ), f"Expected edge_boundary reason for window {wid}; got {kinds}"


def test_attention_reasons_spur_adjacent(stage5_small_file, tmp_path):
    """spur_adjacent fires for a fitted line sitting on a gated spur node, and
    stays silent when every gated spur is far from every fitted line."""
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )

    def _run_with_spurs(spur_centers):
        fp = tmp_path / f"spur_{abs(hash(tuple(spur_centers))) % 100000}.ftmw"
        shutil.copy(stage5_small_file, fp)
        with h5py.File(str(fp), "r") as h5f:
            sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        target = _first_window_with_peak(sf)
        if target is None:
            pytest.skip("No suitable window found in the 3-window fixture")
        peak_f = float(target.fitted_peaks[0].frequency_mhz)
        sf.parameters["spur_centers_mhz"] = [c(peak_f) for c in spur_centers]
        with h5py.File(str(fp), "a") as h5f:
            del h5f["stage5_fitting"]
            save_spectrum_fit_to_hdf5(sf, h5f.create_group("stage5_fitting"))
        review_run_impl(str(fp))
        review = load_stage6_review_from_file(str(fp))
        wid = target.window_id if target.window_id is not None else -1
        return wid, review

    # Positive: a gated spur exactly on the first peak (plus a decoy 100 MHz off).
    wid, review = _run_with_spurs([lambda f: f, lambda f: f + 100.0])
    kinds = {r.kind for r in review.window_statuses[wid].attention_reasons}
    assert (
        "spur_adjacent" in kinds
    ), f"expected spur_adjacent for window {wid}; got {kinds}"

    # Negative: every gated spur is 50 MHz away -- no window should flag.
    _, review_clean = _run_with_spurs([lambda f: f + 50.0])
    all_kinds = {
        r.kind
        for st in review_clean.window_statuses.values()
        for r in st.attention_reasons
    }
    assert "spur_adjacent" not in all_kinds


def _first_window_with_peak(sf):
    for wf in sf.window_fits:
        if (
            wf.window is not None
            and wf.window.freq_range is not None
            and wf.fitted_peaks
        ):
            return wf
    return None


def test_overfit_vif_retired(stage5_small_file, tmp_path):
    """overfit_vif is retired: a non-identifiable amplitude (VIF >> 1) no longer
    raises a standalone flag. Its content now routes to the end-of-Stage-5 merge
    (``auto_merged_review``) or the SNR-aware gate (``worst_eps``); degeneracy is
    discoverable on demand via ``review rank --by max-vif``."""
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )

    fp = tmp_path / "vif_test.ftmw"
    shutil.copy(stage5_small_file, fp)

    with h5py.File(str(fp), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    target_wf = _first_window_with_peak(sf)
    if target_wf is None:
        pytest.skip("No suitable window found in the 3-window fixture")

    # Force a degenerate amplitude: error == amplitude, healthy SNR -> VIF == snr.
    p = target_wf.fitted_peaks[0]
    p.amplitude_error = abs(float(p.amplitude))
    p.snr = 10.0  # VIF = (amp_err/amp) * snr = 10

    with h5py.File(str(fp), "a") as h5f:
        del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(sf, grp)

    review_run_impl(str(fp))
    review = load_stage6_review_from_file(str(fp))

    wid = target_wf.window_id if target_wf.window_id is not None else -1
    kinds = {r.kind for r in review.window_statuses[wid].attention_reasons}
    assert "overfit_vif" not in kinds, f"overfit_vif should be retired; got {kinds}"


def test_low_snr_retired(stage5_small_file, tmp_path):
    """A borderline-SNR fitted peak is NOT flagged: low_snr is retired as a
    reason (weak windows are surfaced on demand via ranking, not flagged)."""
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )

    fp = tmp_path / "lowsnr_test.ftmw"
    shutil.copy(stage5_small_file, fp)

    with h5py.File(str(fp), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    target_wf = _first_window_with_peak(sf)
    if target_wf is None:
        pytest.skip("No suitable window found in the 3-window fixture")

    # A peak just above the survival floor must no longer flag the window.
    target_wf.fitted_peaks[0].snr = 3.3

    with h5py.File(str(fp), "a") as h5f:
        del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(sf, grp)

    result = review_run_impl(str(fp))
    assert "low_snr" not in result.reason_counts


def test_doublet_eps_gt_kappa_retired(stage5_multi_peak_file, tmp_path):
    """The AICc-preferred-doublet observation is no longer an attention trigger."""
    fp = tmp_path / "no_doublet_flag.ftmw"
    shutil.copy(stage5_multi_peak_file, fp)

    result = review_run_impl(str(fp))

    assert "doublet_eps_gt_kappa" not in result.reason_counts, (
        "doublet_eps_gt_kappa should be retired as an attention reason; "
        f"got {result.reason_counts}"
    )


def test_multi_peak_candidate_bearing(stage5_multi_peak_file, tmp_path):
    """On a full 2638 fit, at least some windows have candidate_bearing flags."""
    fp = tmp_path / "multi.ftmw"
    shutil.copy(stage5_multi_peak_file, fp)

    result = review_run_impl(str(fp))

    # A full 2638 run always has some revivable candidates above bar=4.
    candidate_count = result.reason_counts.get("candidate_bearing", 0)
    assert (
        candidate_count > 0
    ), f"Expected candidate_bearing windows in full 2638 fit; got {result.reason_counts}"


# ---------------------------------------------------------------------------
# review rank
# ---------------------------------------------------------------------------


def test_rank_min_snr_sorted_ascending(stage5_small_file, tmp_path):
    """min-snr ranks worst (lowest SNR) first; values are non-decreasing."""
    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)

    ranked = rank_windows_impl(str(fp), by="min-snr")
    assert ranked and all(isinstance(r, RankedWindow) for r in ranked)
    assert all(r.metric == "min-snr" for r in ranked)
    values = [r.value for r in ranked]
    assert values == sorted(values)  # ascending: weakest first


def test_rank_max_vif_sorted_descending(stage5_small_file, tmp_path):
    """max-vif ranks worst (highest VIF) first; values are non-increasing."""
    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)

    ranked = rank_windows_impl(str(fp), by="max_vif")  # underscore form accepted
    values = [r.value for r in ranked]
    assert values == sorted(values, reverse=True)


def test_rank_top_limits(stage5_small_file, tmp_path):
    """--top limits the result; top<=0 returns all."""
    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)

    all_rows = rank_windows_impl(str(fp), by="chi2r")
    assert len(rank_windows_impl(str(fp), by="chi2r", top=1)) == 1
    assert len(rank_windows_impl(str(fp), by="chi2r", top=0)) == len(all_rows)


def test_rank_unknown_metric_raises(stage5_small_file, tmp_path):
    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)
    with pytest.raises(ValueError, match="unknown rank metric"):
        rank_windows_impl(str(fp), by="bogus")


def test_rank_all_metrics_run(stage5_small_file, tmp_path):
    """Every registered metric runs without error and stays in-range/sorted."""
    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)
    for name, (_desc, lower_is_worse) in RANK_METRICS.items():
        ranked = rank_windows_impl(str(fp), by=name)
        values = [r.value for r in ranked]
        assert values == sorted(values, reverse=not lower_is_worse)


def test_rank_cross_interface(stage5_small_file, tmp_path):
    """api.rank_windows and Pipeline.rank_windows agree."""
    import ftmwpipeline.api as ftmw_api

    fp = tmp_path / "rank.ftmw"
    shutil.copy(stage5_small_file, fp)

    a = ftmw_api.rank_windows(str(fp), "min-snr", top=3)
    p = Pipeline.open(fp).rank_windows("min-snr", top=3)
    assert [(r.window_id, round(r.value, 6)) for r in a] == [
        (r.window_id, round(r.value, 6)) for r in p
    ]
