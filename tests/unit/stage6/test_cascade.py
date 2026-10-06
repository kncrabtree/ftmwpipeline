"""Unit tests for the contributor-edit cascade graph + refresh helpers.

These exercise the pure machinery in isolation (no fit context, no fixture
build): the plan-derived source graph, the transitive closure, the topological
order, and the window-level frozen-background refresh. The end-to-end propagation
(refit fires the cascade, identity is a no-op, undo returns to the automatic
baseline) is validated against the dense 655 hub in
``scratch/cascade/cascade_lab_wl.py`` and the committed scratch harness; here we
pin the building blocks so a regression in the graph/refresh logic is caught fast.
"""

from __future__ import annotations

import pytest

from ftmwpipeline._internal.stage6_impl import (
    _base_cascade_sources,
    _cascade_closure,
    _cascade_refit_dependents,
    _cascade_sources,
    _cascade_succs,
    _cascade_topo,
    _non_edge_free_primaries,
    _refresh_frozen_from_sources,
)
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    FitWindow,
    FixedContributor,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.file_manager import CurationConflictError


def _wf(wid, peaks=(), frozen=()):
    """A FittingResult with the given fitted peaks and frozen contributors.

    ``peaks``: iterable of (freq_mhz, amplitude, phase, snr).
    ``frozen``: iterable of (primary_window_id, freq_mhz, amplitude, phase).
    """
    wf = FittingResult(window_id=wid)
    wf.fitted_peaks = [
        FittedPeak(
            detection_index=i,
            frequency_mhz=f,
            amplitude=a,
            phase=ph,
            snr=s,
            window_id=wid,
        )
        for i, (f, a, ph, s) in enumerate(peaks)
    ]
    wf.fixed_parameters = {
        f"frozen_peak_{i}": {
            "peak_index": -1,
            "primary_window_id": p,
            "frequency_mhz": f,
            "amplitude": a,
            "phase": ph,
            "freeze_eligible": True,
        }
        for i, (p, f, a, ph) in enumerate(frozen)
    }
    return wf


def _fw(wid, contributors=()):
    """A FitWindow whose ``fixed_contributors`` are (primary, freq, edge_free)."""
    return FitWindow(
        window_id=wid,
        freq_range=(0.0, 1.0),
        fixed_contributors=[
            FixedContributor(
                peak_index=-1,
                primary_window_id=p,
                frequency_mhz=f,
                edge_free=ef,
            )
            for (p, f, ef) in contributors
        ],
    )


class TestNonEdgeFreePrimaries:
    def test_none_window_returns_none(self):
        assert _non_edge_free_primaries(None) is None

    def test_excludes_edge_free(self):
        fw = _fw(5, [(1, 10.0, False), (2, 20.0, True), (3, 30.0, False)])
        assert _non_edge_free_primaries(fw) == {1, 3}

    def test_empty_contributors(self):
        assert _non_edge_free_primaries(_fw(5)) == set()


def _plan(*windows):
    return WindowPlan(windows=list(windows))


class TestCascadeSources:
    """The cascade graph is plan-derived: each window's non-edge-free
    ``fixed_contributors`` primaries, restricted to the windows the automatic
    fit holds, ordered as the baseline fit froze them."""

    def test_sources_are_the_plan_contributors(self):
        # 2 and 3 read 1; 3 also reads 2.
        plan = _plan(
            _fw(1),
            _fw(2, [(1, 100.0, False)]),
            _fw(3, [(1, 100.0, False), (2, 100.5, False)]),
        )
        frozen = {1: (), 2: (1,), 3: (1, 2)}
        sources = _base_cascade_sources(plan, frozen)
        assert sources == {1: (), 2: (1,), 3: (1, 2)}
        succs = _cascade_succs(sources)
        assert succs == {1: {2, 3}, 2: {3}, 3: set()}

    def test_a_source_the_baseline_fit_froze_nothing_from_is_still_an_edge(self):
        # Window 2's baseline fit froze no line of 1 (none above threshold),
        # but the plan says 2 reads 1: the edge is there, after the frozen
        # ones, so a later edit that gives 1 a strong line reaches 2.
        plan = _plan(_fw(1), _fw(3), _fw(2, [(1, 100.0, False), (3, 99.0, False)]))
        sources = _base_cascade_sources(plan, {1: (), 3: (), 2: (3,)})
        assert sources[2] == (3, 1)

    def test_refresh_order_follows_the_baseline_fit(self):
        plan = _plan(_fw(1), _fw(2), _fw(3, [(1, 100.0, False), (2, 100.5, False)]))
        assert _base_cascade_sources(plan, {1: (), 2: (), 3: (2, 1)})[3] == (2, 1)

    def test_edge_free_contributor_is_not_an_edge(self):
        plan = _plan(_fw(1), _fw(2, [(1, 100.0, True)]))
        sources = _base_cascade_sources(plan, {1: (), 2: (1,)})
        assert sources[2] == ()
        assert _cascade_succs(sources)[1] == set()

    def test_a_window_the_fit_dropped_is_no_source(self):
        # Stage 5 dropped window 4: it is in the plan but not in the fit.
        plan = _plan(_fw(1), _fw(4), _fw(2, [(1, 100.0, False), (4, 101.0, False)]))
        assert _base_cascade_sources(plan, {1: (), 2: (1,)})[2] == (1,)

    def test_unavailable_window_keeps_its_fit_edges(self):
        # No plan window says which of 2's frozen primaries are edge-free.
        plan = _plan(_fw(1), _fw(2, [(1, 100.0, True)]))
        assert _base_cascade_sources(plan, {1: (), 2: (1,)}, {2})[2] == (1,)
        # Nor for a fitted window with no plan entry at all.
        assert _base_cascade_sources(_plan(_fw(1)), {1: (), 2: (1,)})[2] == (1,)

    def test_created_window_reads_live_and_earlier_created_windows(self):
        base = {1: (), 2: (1,)}
        created = [
            _fw(11, [(1, 100.0, False), (12, 101.0, False), (10, 99.0, False)]),
            _fw(10, [(2, 100.0, False), (1, 100.2, False), (2, 100.4, False)]),
            # A widened base window: its base contributors, no new edge.
            _fw(2, [(1, 100.0, False)]),
        ]
        sources = _cascade_sources(base, created)
        assert sources[10] == (2, 1)
        assert sources[11] == (1, 10)  # 12 is not earlier than 11
        assert sources[2] == (1,)
        assert _cascade_succs(sources)[10] == {11}


class TestClosureAndTopo:
    def test_transitive_closure_excludes_edited(self):
        succs = {1: {2}, 2: {3}, 3: set(), 4: set()}
        assert _cascade_closure([1], succs) == {2, 3}

    def test_closure_of_leaf_is_empty(self):
        assert _cascade_closure([3], {1: {2}, 2: {3}, 3: set()}) == set()

    def test_merged_closure_of_two_edits(self):
        succs = {1: {2}, 2: set(), 10: {11}, 11: set()}
        assert _cascade_closure([1, 10], succs) == {2, 11}

    def test_topo_orders_predecessors_first(self):
        preds = {2: {1}, 3: {2}}
        order = _cascade_topo({2, 3}, preds)
        assert order.index(2) < order.index(3)

    def test_topo_raises_on_a_cycle(self):
        """The graph is acyclic by construction; a cycle is an invariant
        violation, never walked in some arbitrary order."""
        with pytest.raises(RuntimeError, match="cycle"):
            _cascade_topo({1, 2, 3}, {2: {1}, 3: {2}, 1: {3}})
        # A cycle through a window outside the set does not count.
        assert _cascade_topo({2, 3}, {2: {1}, 3: {2}, 1: {3}}) == [2, 3]


class TestRefreshFrozenFromSources:
    def _src(self, wid, peaks):
        return _wf(wid, peaks=peaks)

    def _frozen(self, wf):
        return [
            v for k, v in wf.fixed_parameters.items() if k.startswith("frozen_peak_")
        ]

    def test_rebuilds_from_current_source_above_threshold(self):
        # Source 1 currently fits two peaks; one is below min_freeze_snr.
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0), (100.5, 0.2, 0.0, 10.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0)])  # stale snapshot
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {1: src, 2: dep},
            min_freeze_snr=50.0,
        )
        froz = self._frozen(dep)
        assert len(froz) == 1  # the snr=10 peak is dropped
        assert froz[0]["frequency_mhz"] == 100.0
        assert froz[0]["amplitude"] == 1.0  # refreshed from the current fit

    def test_removed_source_peak_drops_contributor(self):
        # Source 1 now fits ONE peak; the dependent had two frozen from it.
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0), (1, 106.0, 0.5, 0.0)])
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {1: src, 2: dep},
            min_freeze_snr=50.0,
        )
        assert len(self._frozen(dep)) == 1

    def test_split_source_adds_contributor(self):
        # Source 1 split one line into two children; both above threshold.
        src = self._src(1, [(99.98, 0.5, 0.0, 3000.0), (100.02, 0.5, 0.0, 3000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {1: src, 2: dep},
            min_freeze_snr=50.0,
        )
        assert len(self._frozen(dep)) == 2

    def test_a_source_the_dependent_holds_nothing_from_contributes_again(self):
        # An earlier edit left source 1 with no line above threshold, so the
        # dependent's refreshed background holds no entry from it. Once 1 has
        # a strong line again the refresh rebuilds it from the source list,
        # not from the entries the dependent happens to hold.
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2)
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {1: src, 2: dep},
            min_freeze_snr=50.0,
        )
        froz = self._frozen(dep)
        assert [(e["primary_window_id"], e["frequency_mhz"]) for e in froz] == [
            (1, 100.0)
        ]

    def test_sources_are_rebuilt_in_source_list_order(self):
        src1 = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        src3 = self._src(3, [(90.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])
        fwm = {2: _fw(2, [(1, 100.0, False), (3, 90.0, False)])}
        _refresh_frozen_from_sources(
            dep, (3, 1), fwm, {1: src1, 3: src3, 2: dep}, min_freeze_snr=50.0
        )
        assert [e["primary_window_id"] for e in self._frozen(dep)] == [3, 1]

    def test_edge_free_contributor_preserved_verbatim(self):
        # Dependent 2 reads source 1 (cascade edge) and source 9 (edge-free).
        src1 = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0), (9, 200.0, 0.3, 1.23)])
        fwm = {2: _fw(2, [(1, 100.0, False), (9, 200.0, True)])}
        _refresh_frozen_from_sources(
            dep, (1,), fwm, {1: src1, 2: dep}, min_freeze_snr=50.0
        )
        ef = [e for e in self._frozen(dep) if e["primary_window_id"] == 9]
        assert len(ef) == 1
        assert ef[0]["amplitude"] == 0.3  # untouched (read from data, not the fit)
        assert ef[0]["phase"] == 1.23

    def test_dropped_source_contributes_nothing(self):
        # Source 1 is absent from fit_map (merged away) -> no skirt.
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {2: dep},
            min_freeze_snr=50.0,
        )
        assert self._frozen(dep) == []

    def test_non_frozen_entries_preserved(self):
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0)])
        dep.fixed_parameters["other_meta"] = {"keep": 1}
        _refresh_frozen_from_sources(
            dep,
            (1,),
            {2: _fw(2, [(1, 100.0, False)])},
            {1: src, 2: dep},
            min_freeze_snr=50.0,
        )
        assert dep.fixed_parameters["other_meta"] == {"keep": 1}


class TestCascadeIntoAnUnavailableWindow:
    """A merged fit with no stored plan: survivor 1's fit holds window 3's line
    as a frozen contributor, but survivor 1's Stage 4 window (narrower than the
    window its fit was made on) does not name 3, or names it edge-free. An edit
    of 3 must still reach 1 and be refused, not leave 1's fit stale."""

    def _cascade(self, stage4_window_1, unavailable):
        fits = [
            _wf(1, frozen=[(3, 100.0, 1.0, 0.0)]),
            _wf(3, peaks=[(100.0, 1.0, 0.0, 5000.0)]),
        ]
        sources = _base_cascade_sources(
            _plan(stage4_window_1, _fw(3)), {1: (3,), 3: ()}, unavailable
        )
        return _cascade_refit_dependents(
            spectrum_fit=SpectrumFit(window_fits=fits),
            edited_wids=[3],
            sources=sources,
            fit_window_map={1: stage4_window_1, 3: _fw(3)},
            fit_ctx=None,
            resolved=None,
            shape_enum=None,
            persisted_cal=None,
            tau_maj_us=None,
            sigma_tau_us=None,
            tau_source="none",
            peak_frequencies_mhz=[],
            min_freeze_snr=0.0,
            snap_tol_mhz=0.0,
            unavailable_window_ids=unavailable,
        )

    @pytest.mark.parametrize(
        "stage4_window_1",
        [_fw(1), _fw(1, [(3, 100.0, True)])],
        ids=["not_a_stage4_contributor", "edge_free_in_stage4"],
    )
    def test_the_edit_is_refused(self, stage4_window_1):
        with pytest.raises(CurationConflictError) as info:
            self._cascade(stage4_window_1, frozenset({1}))
        assert (info.value.reason, info.value.ids) == ("fit_plan_unavailable", [1])

    def test_an_available_window_keeps_its_plan_edges(self):
        # The same fit with its geometry available: the plan window says 1 does
        # not read 3 through a cascade edge, so nothing cascades.
        assert self._cascade(_fw(1, [(3, 100.0, True)]), frozenset()) == []
