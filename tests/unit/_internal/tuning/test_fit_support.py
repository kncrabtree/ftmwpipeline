"""Unit tests for the Stage 5 fit-window selection helpers (pure logic; no
pipeline build needed)."""

from dataclasses import dataclass, field
from typing import Any, List

from ftmwpipeline._internal.tuning.fit_support import (
    FitWindowSelection,
    _close_components,
    _spread_evenly,
    window_fit_quality,
    window_snr_max,
)


def test_spread_evenly_covers_endpoints():
    vals = {i: float(i) for i in range(10)}
    out = _spread_evenly(list(range(10)), vals, 3)
    assert len(out) == 3
    assert out[0] == 0 and out[-1] == 9  # spans the SNR range, not clumped


def test_spread_evenly_degenerate_k():
    vals = {i: float(i) for i in (1, 2, 3, 4)}
    assert _spread_evenly([1, 2, 3, 4], vals, 10) == [1, 2, 3, 4]  # k >= n
    assert _spread_evenly([], vals, 3) == []
    assert len(_spread_evenly([1, 2, 3, 4], vals, 1)) == 1


def test_close_components_pulls_in_joint_group():
    edges = [(0, 1), (1, 2), (3, 4)]  # {0,1,2}, {3,4}, 5 isolated
    assert _close_components({0}, edges) == {0, 1, 2}
    assert _close_components({3}, edges) == {3, 4}
    assert _close_components({5}, edges) == {5}


@dataclass
class _FakePeak:
    snr: Any
    frequency_error: Any = None


@dataclass
class _FakeWF:
    reduced_chi2: float
    window_id: int
    fitted_peaks: List[Any]
    shared_parameters: dict = field(default_factory=dict)


def test_window_snr_max_picks_brightest():
    wf = _FakeWF(2.0, 1, [_FakePeak(5.0), _FakePeak(30.0), _FakePeak(None)])
    assert window_snr_max(wf) == 30.0
    assert window_snr_max(_FakeWF(2.0, 1, [])) == 0.0


def test_window_fit_quality_eps_and_pass():
    # noise-dominated window (chi2r below the F=3 floor): eps 0, passes.
    q = window_fit_quality(_FakeWF(2.0, 3, [_FakePeak(8.0)]))
    assert q["window_id"] == 3 and q["n_peaks"] == 1
    assert q["epsilon"] == 0.0 and q["passed"] is True
    # high chi2r at low SNR clears the SNR-aware allowance -> genuine misfit.
    assert window_fit_quality(_FakeWF(50.0, 4, [_FakePeak(8.0)]))["passed"] is False
    # at high SNR the same chi2r sits under the (kappa*SNR)^2 floor -> passes.
    assert window_fit_quality(_FakeWF(50.0, 5, [_FakePeak(400.0)]))["passed"] is True


def test_fit_window_selection_defaults():
    sel = FitWindowSelection()
    assert sel.top_snr == 3 and sel.sample == 20 and not sel.fit_all
