"""Unit tests for the Stage 4 window-plan data structures."""

import pytest

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)


class TestFitWindow:
    def test_defaults_and_properties(self):
        w = FitWindow(window_id=2, freq_range=(100.0, 140.0))
        assert w.width_mhz == 40.0
        assert w.n_free_peaks == 0
        assert w.batch == 0

    def test_carries_contributors(self):
        fc = FixedContributor(peak_index=7, primary_window_id=0, frequency_mhz=123.4)
        w = FitWindow(
            window_id=1,
            freq_range=(120.0, 130.0),
            free_peak_indices=[3, 4, 5],
            fixed_contributors=[fc],
        )
        assert w.n_free_peaks == 3
        assert w.fixed_contributors[0].peak_index == 7
        assert w.fixed_contributors[0].freeze_eligible is True
        assert "fixed=1" in repr(w)


class TestWindowPlan:
    def test_counts_and_lookup(self):
        windows = [
            FitWindow(window_id=0, freq_range=(0.0, 10.0)),
            FitWindow(
                window_id=1,
                freq_range=(10.0, 20.0),
                batch=1,
            ),
        ]
        plan = WindowPlan(
            windows=windows,
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )
        assert plan.n_windows == 2
        assert plan.n_batches == 2
        assert plan.window(1).batch == 1
        with pytest.raises(KeyError):
            plan.window(99)

    def test_empty_plan(self):
        plan = WindowPlan()
        assert plan.n_windows == 0
        assert plan.n_batches == 0
