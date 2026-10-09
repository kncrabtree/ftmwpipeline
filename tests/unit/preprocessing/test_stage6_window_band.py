"""A Stage 6 window stays inside the analysis band.

``plan_stage6_window`` positions a created window on the active-FT grid, which
is not trimmed: on 2638 it spans 15960-40960 MHz against a 26500-40000 MHz
band. With no window between the anchor and the end of the grid, the gap the
window may occupy used to run to the end of the grid, so a window created
within a half-width of a band edge reached past it -- onto bins the replay path
never estimates noise for, which surfaced as a bare ``amp_floor`` ValueError
inside the fit. ``band_mhz`` clamps the gap to the band's grid points.

Pure planner tests on a synthetic grid; the end-to-end Stage 6 path is pinned in
``tests/unit/stage6/test_create_window.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FitWindow, WindowPlan
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.fitting.plan_execution import materialize_window
from ftmwpipeline.preprocessing.window_planning import (
    Stage6WindowProposal,
    plan_stage6_window,
)

#: 100.0 .. 199.9 MHz at 0.1 MHz, against a narrower analysis band.
FREQS = 100.0 + 0.1 * np.arange(1000)
BAND = (110.0, 190.0)


def _plan(*ranges: Tuple[float, float]) -> WindowPlan:
    windows = [
        FitWindow(window_id=i, freq_range=r, batch=0) for i, r in enumerate(ranges)
    ]
    return WindowPlan(windows=windows, topological_order=list(range(len(windows))))


def _propose(
    plan: WindowPlan,
    anchor: float,
    band: Optional[Tuple[float, float]] = BAND,
) -> Stage6WindowProposal:
    return plan_stage6_window(
        plan,
        [],
        FREQS,
        np.ones_like(FREQS, dtype=complex),
        np.ones_like(FREQS),
        anchor,
        acquisition_us=10.0,
        band_mhz=band,
    )


def _extent(proposal: Stage6WindowProposal) -> Tuple[float, float]:
    lo, hi = proposal.window.freq_range
    return min(lo, hi), max(lo, hi)


@pytest.mark.parametrize("anchor", [110.0, 110.3, 111.5, 188.5, 189.8, 190.0])
def test_a_created_window_near_a_band_edge_stays_inside_the_band(anchor):
    proposal = _propose(_plan((140.0, 150.0)), anchor)
    lo, hi = _extent(proposal)
    assert proposal.mode == "created"
    assert BAND[0] <= lo <= anchor <= hi <= BAND[1]


def test_it_is_shifted_not_shrunk():
    """Same width at the edge as in open space: only the position changes."""
    plan = _plan((140.0, 150.0))
    edge = _propose(plan, 110.2).window.diagnostics["grid_span"]
    open_ = _propose(plan, 125.0).window.diagnostics["grid_span"]
    assert edge[1] - edge[0] == open_[1] - open_[0]


def test_without_a_band_the_window_reaches_past_it():
    """The pre-fix geometry, kept as the control that makes the above mean
    something: an unclamped edge window does leave the band."""
    lo, _hi = _extent(_propose(_plan((140.0, 150.0)), 110.3, band=None))
    assert lo < BAND[0]


def test_a_window_away_from_the_edges_is_unchanged_by_the_band():
    plan = _plan((140.0, 150.0))
    with_band = _propose(plan, 125.0).window
    without = _propose(plan, 125.0, band=None).window
    assert with_band.freq_range == without.freq_range
    assert with_band.diagnostics["grid_span"] == without.diagnostics["grid_span"]


def test_a_widened_window_near_a_band_edge_stays_inside_the_band():
    """A window hugging the band edge leaves too narrow a gap: the neighbor
    widens toward the edge, and stops at it."""
    proposal = _propose(_plan((110.8, 120.0)), 110.3)
    lo, hi = _extent(proposal)
    assert proposal.mode == "widened"
    assert BAND[0] <= lo <= 110.3 <= hi


@pytest.mark.parametrize("anchor", [109.95, 190.05, 105.0, 195.0])
def test_an_anchor_outside_the_bands_grid_points_is_refused(anchor):
    with pytest.raises(BadSettingError, match="outside the analysis band"):
        _propose(_plan((140.0, 150.0)), anchor)


def test_a_band_holding_no_grid_point_is_an_error():
    with pytest.raises(ValueError, match="holds no active-FT grid point"):
        _propose(_plan((140.0, 150.0)), 125.0, band=(250.0, 260.0))


# ---------------------------------------------------------------------------
# materialize_window: the backstop
# ---------------------------------------------------------------------------


def _active_ft() -> SimpleNamespace:
    return SimpleNamespace(
        freq_mhz=FREQS, complex_spectrum=np.ones_like(FREQS, dtype=complex)
    )


def test_a_window_slicing_non_finite_noise_is_refused_by_name():
    rms = np.ones_like(FREQS)
    rms[FREQS < BAND[0]] = np.nan
    win = FitWindow(window_id=7, freq_range=(107.0, 113.0))
    with pytest.raises(ValueError, match="window 7 .* no finite noise estimate"):
        materialize_window(win, _active_ft(), rms, sideband="upper")  # type: ignore[arg-type]


def test_a_window_inside_the_band_materializes():
    rms = np.ones_like(FREQS)
    rms[FREQS < BAND[0]] = np.nan
    win = FitWindow(window_id=7, freq_range=(110.0, 116.0))
    _f, _u, _z, sig, _c = materialize_window(
        win, _active_ft(), rms, sideband="upper"  # type: ignore[arg-type]
    )
    assert np.isfinite(sig).all()
