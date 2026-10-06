"""The ``empty_window_residual`` review item (pure trigger, no file).

``dev-docs/CONTRACT_STRATEGY.md`` §Review attention: a fitted-plan window the fit
holds no line in, not taken over or edited in Stage 6, whose edge Stage 5's
thaw / replan handshake left flagged above the fit's own
``residual_edge_threshold``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import pytest

from ftmwpipeline._internal.empty_window_attention import (
    EMPTY_WINDOW_KINDS,
    EMPTY_WINDOW_RESIDUAL,
    EMPTY_WINDOW_SPUR,
    empty_window_reasons,
    flagged_empty_edges,
    flagged_lineless_ids,
    last_refit_revision,
    lineless_window_fit,
    superseded_window_ids,
    takeover_points,
)
from ftmwpipeline._internal.stage6_impl import (
    PlannedAction,
    _batch_known_window_ids,
    _settle_empty_window_reasons,
)
from ftmwpipeline.core.absent import Absent
from ftmwpipeline.core.data_structures import (
    _ADVISORY_REASON_KINDS,
    ATTENTION_KINDS,
    AttentionReason,
    FittedPeak,
    FittingResult,
    FitWindow,
    Peak,
    ReplanInfo,
    SpectrumFit,
    Stage6Review,
    ThawInfo,
    WindowReviewStatus,
)
from ftmwpipeline.io.stage6_review_serialization import (
    _status_from_dict,
    _status_to_dict,
)

pytestmark = [pytest.mark.unit]

THR = 8.0
TOL = 0.2  # MHz; the spur_adjacent tolerance a caller passes


def _replan(
    wid: int,
    side: str,
    s_coh: float,
    accepted: bool = False,
    *,
    revision: int = 0,
    survivor: Optional[int] = None,
    refit: Optional[Sequence[int]] = None,
) -> ReplanInfo:
    return ReplanInfo(
        triggering_window_id=wid,
        partner_window_id=wid + 1,
        surviving_window_id=wid if survivor is None else survivor,
        edge_side=side,
        edge_coherence_before=s_coh,
        revision_before=revision,
        revision_after=revision + 1 if accepted else revision,
        accepted=accepted,
        reason="" if accepted else "not merged: the window's fit holds no line",
        refit_window_ids=None if refit is None else tuple(refit),
    )


def _thaw(wid: int, side: str, before: float, accepted: bool) -> ThawInfo:
    return ThawInfo(
        dependent_window_id=wid,
        primary_window_id=wid - 1,
        contributor_peak_index=0,
        contributor_frequency_mhz=1000.0,
        edge_side=side,
        edge_coherence_before=before,
        edge_coherence_after=1.0 if accepted else before,
        accepted=accepted,
    )


def _live_fit(wid: int, freq: float) -> FittingResult:
    wf = FittingResult(window_id=wid)
    wf.fitted_peaks = [
        FittedPeak(detection_index=0, frequency_mhz=freq, amplitude=1.0, window_id=wid)
    ]
    return wf


def _fit(
    *,
    replans: Sequence[ReplanInfo] = (),
    thaws: Sequence[ThawInfo] = (),
    window_fits: Sequence[FittingResult] = (),
    threshold: Optional[float] = THR,
    spurs: Sequence[float] = (),
) -> SpectrumFit:
    params = {} if threshold is None else {"residual_edge_threshold": threshold}
    params["spur_centers_mhz"] = list(spurs)
    params["shape"] = "gaussian"
    return SpectrumFit(
        window_fits=list(window_fits),
        replan_history=list(replans),
        thaw_history=list(thaws),
        parameters=params,
        diagnostics={
            "gated_spurs": [
                {"center_mhz": f, "source": "flat+saturated"} for f in spurs
            ]
        },
    )


def _plan() -> List[FitWindow]:
    return [
        FitWindow(window_id=1, freq_range=(990.0, 995.0), free_peak_indices=[0]),
        FitWindow(window_id=2, freq_range=(1000.0, 1005.0), free_peak_indices=[2, 1]),
        FitWindow(window_id=3, freq_range=(1010.0, 1015.0), free_peak_indices=[3]),
    ]


def _peaks() -> List[Peak]:
    return [
        Peak(992.5, 1.0, snr=12.0),
        Peak(1002.5, 5.0, snr=69.0),
        Peak(1001.0, 0.5, snr=None),
        Peak(1012.0, 1.0, snr=9.0),
    ]


def _reasons(fit: SpectrumFit, excluded: Sequence[int] = ()) -> dict:
    return empty_window_reasons(
        fit,
        _plan(),
        _peaks(),
        excluded_window_ids=set(excluded),
        spur_tol_mhz=TOL,
    )


# ---- the kind --------------------------------------------------------------


def test_kinds_are_declared_queued_and_advisory():
    assert EMPTY_WINDOW_RESIDUAL == "empty_window_residual"
    assert EMPTY_WINDOW_SPUR == "empty_window_spur"
    assert set(EMPTY_WINDOW_KINDS) <= set(ATTENTION_KINDS)
    assert EMPTY_WINDOW_RESIDUAL not in _ADVISORY_REASON_KINDS
    assert EMPTY_WINDOW_SPUR in _ADVISORY_REASON_KINDS
    advisory = WindowReviewStatus(
        window_id=2,
        attention_reasons=[AttentionReason(EMPTY_WINDOW_SPUR, "x", 2.0)],
    )
    assert not advisory.needs_attention
    st = WindowReviewStatus(
        window_id=2,
        attention_reasons=[AttentionReason(EMPTY_WINDOW_RESIDUAL, "x", 2.0)],
    )
    assert st.needs_attention


# ---- the trigger -----------------------------------------------------------


def test_replan_flag_on_an_empty_window_triggers():
    fit = _fit(replans=[_replan(2, "low", 17.6), _replan(2, "high", 11.4)])
    out = _reasons(fit)
    assert set(out) == {2}
    r = out[2]
    assert r.kind == EMPTY_WINDOW_RESIDUAL
    assert r.severity == pytest.approx(17.6 / THR)
    assert r.evidence["edges"] == [
        {
            "side": "low",
            "s_coh": 17.6,
            "neighbour_line_distance_mhz": Absent.UNDEFINED,
        },
        {
            "side": "high",
            "s_coh": 11.4,
            "neighbour_line_distance_mhz": Absent.UNDEFINED,
        },
    ]
    assert r.evidence["residual_edge_threshold"] == THR
    # Candidates ascend in frequency, whatever the plan's index order.
    assert [c["detection_index"] for c in r.evidence["candidates"]] == [2, 1]
    assert r.locations == [1001.0, 1002.5]


def test_edge_at_or_below_threshold_does_not_trigger():
    assert _reasons(_fit(replans=[_replan(2, "low", THR)])) == {}
    assert _reasons(_fit(replans=[_replan(2, "low", 3.0)])) == {}
    assert _reasons(_fit(replans=[_replan(2, "low", float("nan"))])) == {}


def test_a_window_holding_a_line_does_not_trigger():
    fit = _fit(replans=[_replan(2, "low", 20.0)], window_fits=[_live_fit(2, 1002.5)])
    assert _reasons(fit) == {}


def test_a_window_result_with_no_line_still_triggers():
    """A window result that holds no line counts as empty."""
    fit = _fit(
        replans=[_replan(2, "low", 20.0)],
        window_fits=[FittingResult(window_id=2)],
    )
    assert set(_reasons(fit)) == {2}


def test_excluded_windows_do_not_trigger():
    fit = _fit(replans=[_replan(2, "low", 20.0)])
    assert _reasons(fit, excluded=[2]) == {}


def test_a_fit_without_a_threshold_flags_nothing():
    assert _reasons(_fit(replans=[_replan(2, "low", 20.0)], threshold=None)) == {}


def test_an_accepted_replan_is_not_a_flag():
    assert _reasons(_fit(replans=[_replan(2, "low", 20.0, accepted=True)])) == {}


def test_rejected_thaw_flags_and_accepted_thaw_resolves():
    flagged = _fit(thaws=[_thaw(2, "high", 12.0, accepted=False)])
    assert flagged_empty_edges(flagged, 2, THR) == [("high", 12.0)]
    resolved = _fit(thaws=[_thaw(2, "high", 12.0, accepted=True)])
    assert flagged_empty_edges(resolved, 2, THR) == []


def test_replan_after_an_accepted_thaw_flags_again():
    """Records are read in Stage 5's order: thaw, then replan."""
    fit = _fit(
        thaws=[_thaw(2, "low", 12.0, accepted=True)],
        replans=[_replan(2, "low", 15.0)],
    )
    assert flagged_empty_edges(fit, 2, THR) == [("low", 15.0)]


def test_strongest_record_per_edge_wins():
    fit = _fit(replans=[_replan(2, "low", 10.0), _replan(2, "low", 14.0)])
    assert flagged_empty_edges(fit, 2, THR) == [("low", 14.0)]


def test_other_windows_records_are_ignored():
    fit = _fit(replans=[_replan(3, "low", 20.0)])
    assert flagged_empty_edges(fit, 2, THR) == []


def test_gated_spur_and_snr_evidence():
    fit = _fit(replans=[_replan(2, "low", 20.0)], spurs=[1002.55])
    cands = _reasons(fit)[2].evidence["candidates"]
    by_idx = {c["detection_index"]: c for c in cands}
    assert by_idx[1]["gated_spur"] is True
    assert by_idx[1]["spur_center_mhz"] == 1002.55
    assert by_idx[1]["spur_source"] == "flat+saturated"
    assert by_idx[1]["snr"] == 69.0
    assert by_idx[2]["gated_spur"] is False
    # Every candidate carries every key; a missing value is Absent.
    assert set(by_idx[1]) == set(by_idx[2])
    assert by_idx[2]["spur_center_mhz"] is Absent.UNDEFINED
    assert by_idx[2]["spur_source"] is Absent.UNDEFINED
    assert by_idx[2]["snr"] is Absent.UNDEFINED  # Stage 3 recorded none
    assert "gated spur at 1002.5500 MHz" in _reasons(fit)[2].detail
    # One peak of the window is not on a gated spur: the item is queued.
    assert _reasons(fit)[2].kind == EMPTY_WINDOW_RESIDUAL


def test_spur_without_a_recorded_source_reads_not_run():
    fit = _fit(replans=[_replan(2, "low", 20.0)])
    fit.parameters["spur_centers_mhz"] = [1002.5]  # gated, no diagnostic record
    cand = {c["detection_index"]: c for c in _reasons(fit)[2].evidence["candidates"]}[1]
    assert cand["gated_spur"] is True
    assert cand["spur_center_mhz"] == 1002.5
    assert cand["spur_source"] is Absent.NOT_RUN


def test_degenerate_stage3_snr_reads_undefined():
    peaks = _peaks()
    peaks[1] = Peak(1002.5, 5.0, snr=0.0, noise_std_local=0.0)
    out = empty_window_reasons(
        _fit(replans=[_replan(2, "low", 20.0)]),
        _plan(),
        peaks,
        excluded_window_ids=set(),
        spur_tol_mhz=TOL,
    )
    cand = {c["detection_index"]: c for c in out[2].evidence["candidates"]}[1]
    assert cand["snr"] is Absent.UNDEFINED


def test_edge_carries_the_distance_to_the_nearest_live_line():
    fit = _fit(
        replans=[_replan(2, "low", 20.0), _replan(2, "high", 12.0)],
        window_fits=[_live_fit(1, 992.5)],
    )
    edges = {e["side"]: e for e in _reasons(fit)[2].evidence["edges"]}
    assert edges["low"]["neighbour_line_distance_mhz"] == pytest.approx(7.5)
    assert edges["high"]["neighbour_line_distance_mhz"] is Absent.UNDEFINED


def test_every_peak_on_a_gated_spur_is_advisory():
    fit = _fit(replans=[_replan(2, "low", 20.0)], spurs=[1001.0, 1002.5])
    reason = _reasons(fit)[2]
    assert reason.kind == EMPTY_WINDOW_SPUR
    assert all(c["gated_spur"] for c in reason.evidence["candidates"])
    assert "consistent with the saturated spur's skirt beyond its mask" in (
        reason.detail
    )
    assert reason.severity == pytest.approx(20.0 / THR)
    assert not WindowReviewStatus(2, attention_reasons=[reason]).needs_attention


def test_a_window_with_no_stage3_peak_is_queued():
    plan = [FitWindow(window_id=2, freq_range=(1000.0, 1005.0))]
    out = empty_window_reasons(
        _fit(replans=[_replan(2, "low", 20.0)], spurs=[1002.5]),
        plan,
        _peaks(),
        excluded_window_ids=set(),
        spur_tol_mhz=TOL,
    )
    assert out[2].kind == EMPTY_WINDOW_RESIDUAL
    assert out[2].evidence["candidates"] == [] and out[2].locations == []


def test_evidence_is_json_and_round_trips():
    """Absent values are stored as null plus a ``__status`` code and come back
    as the same members."""
    reason = _reasons(_fit(replans=[_replan(2, "low", 20.0)], spurs=[1002.5]))[2]
    assert any(
        isinstance(v, Absent) for c in reason.evidence["candidates"] for v in c.values()
    )
    st = WindowReviewStatus(window_id=2, attention_reasons=[reason])
    blob = json.loads(json.dumps(_status_to_dict(st), allow_nan=False))
    back = _status_from_dict(blob)
    assert back.attention_reasons[0] == reason


def test_legacy_reason_without_evidence_loads_empty():
    st = _status_from_dict(
        {
            "window_id": 4,
            "attention_reasons": [{"kind": "worst_eps", "detail": "d", "severity": 1}],
        }
    )
    assert st.attention_reasons[0].evidence == {}


# ---- supersession and acceptance ------------------------------------------


def test_superseded_by_same_id_or_by_covering_the_flagged_peaks():
    plan = _plan()
    points = {2: [1001.0, 1002.5]}
    # Overlapping window 2 without covering its peaks takes nothing over.
    overlap = [FitWindow(window_id=9, freq_range=(1002.0, 1007.0))]
    assert superseded_window_ids(plan, overlap, points) == set()
    covering = [FitWindow(window_id=9, freq_range=(1000.5, 1003.0))]
    assert superseded_window_ids(plan, covering, points) == {2}
    # Two created windows may cover the peaks between them.
    split = [
        FitWindow(window_id=9, freq_range=(1000.5, 1001.5)),
        FitWindow(window_id=10, freq_range=(1002.0, 1003.0)),
    ]
    assert superseded_window_ids(plan, split, points) == {2}
    same_id = [FitWindow(window_id=3, freq_range=(1011.0, 1016.0))]
    assert superseded_window_ids(plan, same_id, {}) == {3}
    assert superseded_window_ids(plan, [], points) == set()


def test_takeover_points_fall_back_to_the_flagged_edges():
    win = _plan()[1]
    assert takeover_points(win, [1002.5], ["low"]) == [1002.5]
    assert takeover_points(win, [], ["high"]) == [1005.0]
    assert takeover_points(win, [], []) == [1000.0, 1005.0]


def test_an_overlapping_created_window_keeps_the_item():
    fit = _fit(replans=[_replan(2, "low", 20.0)])
    out = empty_window_reasons(
        fit,
        _plan(),
        _peaks(),
        excluded_window_ids=set(),
        spur_tol_mhz=TOL,
        created_windows=[FitWindow(window_id=9, freq_range=(1003.0, 1007.0))],
    )
    assert set(out) == {2}
    gone = empty_window_reasons(
        fit,
        _plan(),
        _peaks(),
        excluded_window_ids=set(),
        spur_tol_mhz=TOL,
        created_windows=[FitWindow(window_id=9, freq_range=(1000.0, 1004.0))],
    )
    assert gone == {}


# ---- records that no longer describe the fit ------------------------------


def test_a_replan_record_from_before_the_windows_refit_is_ignored():
    """Window 2 depends on window 1; a round-1 merge with 1 as survivor re-fit
    window 2, so its round-1 'not merged' record is stale."""
    stale = _replan(2, "low", 20.0, revision=0)
    merge = _replan(1, "high", 15.0, accepted=True, revision=0, survivor=1)
    deps = [(2, 1)]
    fit = _fit(replans=[stale, merge])
    assert last_refit_revision(fit, 2, deps) == 1
    assert flagged_empty_edges(fit, 2, THR, dependency_edges=deps) == []
    out = empty_window_reasons(
        fit,
        _plan(),
        _peaks(),
        excluded_window_ids=set(),
        spur_tol_mhz=TOL,
        dependency_edges=deps,
    )
    assert out == {}
    # Without the dependency the merge did not re-fit window 2.
    assert flagged_empty_edges(fit, 2, THR) == [("low", 20.0)]
    # A record of a later round (scanning the re-fit) still counts.
    fresh = _replan(2, "high", 11.0, revision=1)
    fit2 = _fit(replans=[stale, merge, fresh])
    assert flagged_empty_edges(fit2, 2, THR, dependency_edges=deps) == [("high", 11.0)]


def test_transitive_dependents_of_the_survivor_count_as_refit():
    merge = _replan(1, "high", 15.0, accepted=True, revision=2, survivor=1)
    fit = _fit(replans=[merge])
    assert last_refit_revision(fit, 3, [(2, 1), (3, 2)]) == 3
    assert last_refit_revision(fit, 3, [(2, 1)]) == 0


def test_the_stored_refit_set_covers_a_rewritten_thawed_primary():
    """Window 4 is the primary of an accepted thaw whose dependent the merge
    re-fit; the round re-fit window 4 too, though it depends on no survivor.
    The stored set says so, and its round-0 record is stale."""
    stale = _replan(4, "low", 20.0, revision=0)
    merge = _replan(1, "high", 15.0, accepted=True, revision=0, refit=(1, 2, 4))
    deps = [(2, 1)]
    fit = _fit(replans=[stale, merge])
    assert last_refit_revision(fit, 4, deps) == 1
    assert flagged_empty_edges(fit, 4, THR, dependency_edges=deps) == []
    # The stored set is read as is: the dependency closure is not added to it.
    assert last_refit_revision(fit, 3, [(3, 1)]) == 0


def test_a_record_without_the_stored_set_falls_back_to_the_closure():
    stale = _replan(4, "low", 20.0, revision=0)
    merge = _replan(1, "high", 15.0, accepted=True, revision=0)
    assert merge.refit_window_ids is None
    fit = _fit(replans=[stale, merge])
    assert last_refit_revision(fit, 2, [(2, 1)]) == 1
    # The closure cannot see the thawed primary, so its record still counts.
    assert last_refit_revision(fit, 4, [(2, 1)]) == 0
    assert flagged_empty_edges(fit, 4, THR, dependency_edges=[(2, 1)]) == [
        ("low", 20.0)
    ]


# ---- a bare accept of a lineless window in a batch -------------------------


def test_batch_may_name_a_flagged_lineless_window_for_a_bare_accept_only():
    ctx = SimpleNamespace(changeset=SimpleNamespace(lineless_reviewable=frozenset({5})))
    bare = PlannedAction(kind="accept", window_id=5)
    with_candidate = PlannedAction(kind="accept", window_id=5, candidate=1.0)
    edit = PlannedAction(kind="edit", window_id=5, add=[1.0])
    assert _batch_known_window_ids(ctx, [bare], {1}) == {1, 5}
    assert _batch_known_window_ids(ctx, [with_candidate], {1}) == {1}
    assert _batch_known_window_ids(ctx, [edit], {1}) == {1}
    other = PlannedAction(kind="accept", window_id=6)
    assert _batch_known_window_ids(ctx, [other], {1}) == {1}


def test_flagged_lineless_ids():
    review = Stage6Review(
        window_statuses={
            2: WindowReviewStatus(
                2, attention_reasons=[AttentionReason(EMPTY_WINDOW_RESIDUAL, "", 1.0)]
            ),
            3: WindowReviewStatus(
                3, attention_reasons=[AttentionReason("worst_eps", "", 1.0)]
            ),
            4: WindowReviewStatus(
                4, attention_reasons=[AttentionReason(EMPTY_WINDOW_SPUR, "", 1.0)]
            ),
        }
    )
    assert flagged_lineless_ids(review, {7}) == {2, 4}
    assert flagged_lineless_ids(review, {2, 4}) == set()


def test_lineless_window_fit_draws_the_plan_range():
    wf = lineless_window_fit(_plan()[1], "gaussian")
    assert wf.window_id == 2
    assert wf.window is not None and wf.window.freq_range == (1000.0, 1005.0)
    assert wf.fitted_peaks == [] and wf.shape == "gaussian"
    assert not np.isfinite(wf.reduced_chi2)


# ---- a batch drops the reason it resolved ---------------------------------


def _flagged_statuses(provenance: str = "auto") -> dict:
    return {
        2: WindowReviewStatus(
            2,
            provenance=provenance,
            attention_reasons=[AttentionReason(EMPTY_WINDOW_RESIDUAL, "", 2.0)],
        )
    }


def test_settle_drops_a_status_a_created_window_took_over():
    statuses = _flagged_statuses()
    _settle_empty_window_reasons(
        statuses,
        _fit(),
        [FitWindow(window_id=9, freq_range=(1000.0, 1005.0))],
        base_plan_windows=_plan(),
        edited_window_ids=set(),
    )
    assert 2 not in statuses


def test_settle_keeps_a_reviewed_status_without_the_reason():
    statuses = _flagged_statuses("reviewed")
    _settle_empty_window_reasons(
        statuses,
        _fit(window_fits=[_live_fit(2, 1002.5)]),
        [],
        base_plan_windows=_plan(),
        edited_window_ids=set(),
    )
    assert statuses[2].provenance == "reviewed"
    assert statuses[2].attention_reasons == []


def test_settle_drops_the_advisory_kind_too():
    statuses = {
        2: WindowReviewStatus(
            2, attention_reasons=[AttentionReason(EMPTY_WINDOW_SPUR, "", 2.0)]
        )
    }
    _settle_empty_window_reasons(
        statuses,
        _fit(),
        [FitWindow(window_id=9, freq_range=(999.0, 1006.0))],
        base_plan_windows=_plan(),
        edited_window_ids=set(),
    )
    assert 2 not in statuses


def test_settle_keeps_the_item_when_a_created_window_misses_its_peak():
    statuses = {
        2: WindowReviewStatus(
            2,
            attention_reasons=[
                AttentionReason(EMPTY_WINDOW_RESIDUAL, "", 2.0, locations=[1002.5])
            ],
        )
    }
    _settle_empty_window_reasons(
        statuses,
        _fit(),
        [FitWindow(window_id=9, freq_range=(1003.0, 1007.0))],
        base_plan_windows=_plan(),
        edited_window_ids=set(),
    )
    assert [r.kind for r in statuses[2].attention_reasons] == [EMPTY_WINDOW_RESIDUAL]


def test_settle_drops_a_reviewed_lineless_status_once_taken_over():
    statuses = _flagged_statuses("reviewed")
    _settle_empty_window_reasons(
        statuses,
        _fit(),
        [FitWindow(window_id=9, freq_range=(999.0, 1006.0))],
        base_plan_windows=_plan(),
        edited_window_ids=set(),
    )
    assert 2 not in statuses


def test_settle_leaves_an_unresolved_item_alone():
    statuses = _flagged_statuses()
    _settle_empty_window_reasons(
        statuses, _fit(), [], base_plan_windows=_plan(), edited_window_ids=set()
    )
    assert [r.kind for r in statuses[2].attention_reasons] == [EMPTY_WINDOW_RESIDUAL]
