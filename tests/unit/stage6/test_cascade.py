"""Unit tests for the contributor-edit cascade graph + refresh helpers.

These exercise the pure machinery in isolation (no fit context, no fixture
build): the reverse dependency map, the transitive closure, the topological
order, and the window-level frozen-background refresh. The end-to-end propagation
(refit fires the cascade, identity is a no-op, undo returns to the automatic
baseline) is validated against the dense 655 hub in
``scratch/cascade/cascade_lab_wl.py`` and the committed scratch harness; here we
pin the building blocks so a regression in the graph/refresh logic is caught fast.
"""

from __future__ import annotations

from ftmwpipeline._internal.stage6_impl import (
    _cascade_closure,
    _cascade_succs,
    _cascade_topo,
    _non_edge_free_primaries,
    _refresh_frozen_window_level,
)
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    FitWindow,
    FixedContributor,
)


def _wf(wid, peaks=(), frozen=()):
    """A FittingResult with the given fitted peaks and frozen contributors.

    ``peaks``: iterable of (freq_mhz, amplitude, phase, snr).
    ``frozen``: iterable of (primary_window_id, freq_mhz, amplitude, phase).
    """
    wf = FittingResult(window_id=wid)
    wf.fitted_peaks = [
        FittedPeak(
            peak_id=i,
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


class TestCascadeSuccs:
    def test_reverse_map_from_frozen_parameters(self):
        # 2 and 3 depend on 1; 3 also depends on 2.
        wfs = [
            _wf(1, peaks=[(100.0, 1.0, 0.0, 5000.0)]),
            _wf(2, frozen=[(1, 100.0, 1.0, 0.0)]),
            _wf(3, frozen=[(1, 100.0, 1.0, 0.0), (2, 100.5, 0.5, 0.0)]),
        ]
        fwm = {
            1: _fw(1),
            2: _fw(2, [(1, 100.0, False)]),
            3: _fw(3, [(1, 100.0, False), (2, 100.5, False)]),
        }
        succs = _cascade_succs(wfs, fwm)
        assert succs[1] == {2, 3}
        assert succs[2] == {3}
        assert succs[3] == set()

    def test_edge_free_contributor_is_not_an_edge(self):
        # 2 reads 1 only via an edge-free contributor -> cascade-immune.
        wfs = [_wf(1), _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])]
        fwm = {1: _fw(1), 2: _fw(2, [(1, 100.0, True)])}
        assert _cascade_succs(wfs, fwm)[1] == set()

    def test_unknown_plan_window_treats_all_as_edges(self):
        # No plan FitWindow for 2 -> conservative: the contributor is an edge.
        wfs = [_wf(1), _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])]
        assert _cascade_succs(wfs, {})[1] == {2}


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


class TestRefreshFrozenWindowLevel:
    def _src(self, wid, peaks):
        return _wf(wid, peaks=peaks)

    def test_rebuilds_from_current_source_above_threshold(self):
        # Source 1 currently fits two peaks; one is below min_freeze_snr.
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0), (100.5, 0.2, 0.0, 10.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0)])  # stale snapshot
        _refresh_frozen_window_level(
            dep, {2: _fw(2, [(1, 100.0, False)])}, {1: src, 2: dep}, min_freeze_snr=50.0
        )
        froz = [
            v for k, v in dep.fixed_parameters.items() if k.startswith("frozen_peak_")
        ]
        assert len(froz) == 1  # the snr=10 peak is dropped
        assert froz[0]["frequency_mhz"] == 100.0
        assert froz[0]["amplitude"] == 1.0  # refreshed from the current fit

    def test_removed_source_peak_drops_contributor(self):
        # Source 1 now fits ONE peak; the dependent had two frozen from it.
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0), (1, 106.0, 0.5, 0.0)])
        _refresh_frozen_window_level(
            dep, {2: _fw(2, [(1, 100.0, False)])}, {1: src, 2: dep}, min_freeze_snr=50.0
        )
        froz = [k for k in dep.fixed_parameters if k.startswith("frozen_peak_")]
        assert len(froz) == 1

    def test_split_source_adds_contributor(self):
        # Source 1 split one line into two children; both above threshold.
        src = self._src(1, [(99.98, 0.5, 0.0, 3000.0), (100.02, 0.5, 0.0, 3000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])
        _refresh_frozen_window_level(
            dep, {2: _fw(2, [(1, 100.0, False)])}, {1: src, 2: dep}, min_freeze_snr=50.0
        )
        froz = [k for k in dep.fixed_parameters if k.startswith("frozen_peak_")]
        assert len(froz) == 2

    def test_edge_free_contributor_preserved_verbatim(self):
        # Dependent 2 reads source 1 (cascade edge) and source 9 (edge-free).
        src1 = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0), (9, 200.0, 0.3, 1.23)])
        fwm = {2: _fw(2, [(1, 100.0, False), (9, 200.0, True)])}
        _refresh_frozen_window_level(dep, fwm, {1: src1, 2: dep}, min_freeze_snr=50.0)
        froz = [
            v for k, v in dep.fixed_parameters.items() if k.startswith("frozen_peak_")
        ]
        ef = [e for e in froz if e["primary_window_id"] == 9]
        assert len(ef) == 1
        assert ef[0]["amplitude"] == 0.3  # untouched (read from data, not the fit)
        assert ef[0]["phase"] == 1.23

    def test_dropped_source_contributes_nothing(self):
        # Source 1 is absent from fit_map (merged away) -> no skirt.
        dep = _wf(2, frozen=[(1, 100.0, 1.0, 0.0)])
        _refresh_frozen_window_level(
            dep, {2: _fw(2, [(1, 100.0, False)])}, {2: dep}, min_freeze_snr=50.0
        )
        froz = [k for k in dep.fixed_parameters if k.startswith("frozen_peak_")]
        assert froz == []

    def test_non_frozen_entries_preserved(self):
        src = self._src(1, [(100.0, 1.0, 0.0, 5000.0)])
        dep = _wf(2, frozen=[(1, 100.0, 0.9, 0.0)])
        dep.fixed_parameters["other_meta"] = {"keep": 1}
        _refresh_frozen_window_level(
            dep, {2: _fw(2, [(1, 100.0, False)])}, {1: src, 2: dep}, min_freeze_snr=50.0
        )
        assert dep.fixed_parameters["other_meta"] == {"keep": 1}
