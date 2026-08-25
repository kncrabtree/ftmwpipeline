"""
The auto-merge frequency-error widening, and its survival through curation.

A Stage 5 auto-merge collapses a degenerate sub-resolution pair into one line
whose position is honestly known only to within the amplitude-weighted spread
of the components it absorbed, so its ``frequency_error`` is widened to
``sqrt(formal**2 + spread**2)``. The spread was previously recorded only in
``FittedPeak.extra_errors``, which is not serialized -- so the widening
survived exactly until the first Stage 6 refit of that window recomputed the
error from its own covariance, at which point the reported uncertainty on a
blended line silently *shrank* by one to two orders of magnitude (BlackQuill,
2026-08-25). Nothing had to touch the line for that to happen: a cascade refit
triggered by an edit in a neighboring window did it too.

These tests pin the whole lifecycle: the formula, Stage 5 stamping it on the
peak, the column round-tripping, the re-application at every refit, the cases
where the spread must NOT be carried (removed line, split products, added
lines), and the recovery of the spread from window-level diagnostics for a
file fitted before the per-peak column existed.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List

import h5py
import pytest

from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage5_impl import _inflate_merged_frequency_errors
from ftmwpipeline._internal.stage6_impl import (
    _seed_unresolved_spreads_from_diagnostics,
    refit_window_impl,
)
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    SpectrumFit,
    widen_for_unresolved_spread,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _window_peaks(path: Path, wid: int) -> List[FittedPeak]:
    wf = next(w for w in _load_fit(path).window_fits if w.window_id == wid)
    return list(wf.fitted_peaks)


def _busiest_window(path: Path) -> int:
    """The window with the most fitted peaks -- the one whose refit is the
    least degenerate, and the one a spread is most usefully planted in."""
    fit = _load_fit(path)
    wf = max(
        (w for w in fit.window_fits if w.window_id is not None),
        key=lambda w: len(w.fitted_peaks),
    )
    return int(wf.window_id)


def _plant_spread(path: Path, wid: int, spread_mhz: float) -> float:
    """Stamp *spread_mhz* on the first peak of *wid*, exactly as a Stage 5
    auto-merge would have, and persist it. Returns that peak's frequency.

    Planting rather than fitting a real blend: the merge itself is Stage 5's
    business and is covered there, and a fixture that happens to contain a
    degenerate pair would make this module's subject incidental to it.
    """
    from ftmwpipeline.io.fitting_serialization import (
        update_spectrum_fit_windows_in_hdf5,
    )

    fit = _load_fit(path)
    wf = next(w for w in fit.window_fits if w.window_id == wid)
    pk = wf.fitted_peaks[0]
    pk.unresolved_spread_mhz = spread_mhz
    pk.frequency_error = widen_for_unresolved_spread(pk.frequency_error, spread_mhz)
    with h5py.File(str(path), "a") as h5f:
        update_spectrum_fit_windows_in_hdf5(fit, h5f["stage5_fitting"], [wid])
    return float(pk.frequency_mhz)


def _peak_nearest(peaks: List[FittedPeak], freq: float) -> FittedPeak:
    return min(peaks, key=lambda p: abs(float(p.frequency_mhz) - freq))


def _clear_add_freq(path: Path, wid: int) -> float:
    """The in-window frequency furthest from every fitted peak.

    Not the window center: Stage 4 builds a window around the detection that
    seeded it, so the center is an existing peak's birth position and an add
    there is refused (two lines cannot be born at one position).
    """
    import numpy as np

    from ftmwpipeline._internal.stage4_impl import load_windows_impl

    plan = load_windows_impl(str(path))["plan"]
    w = next(x for x in plan.windows if int(x.window_id) == wid)
    lo, hi = sorted(w.freq_range)
    peaks = [float(p.frequency_mhz) for p in _window_peaks(path, wid)]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


class TestWidenFormula:
    def test_adds_in_quadrature(self):
        assert widen_for_unresolved_spread(0.003, 0.004) == pytest.approx(0.005)

    def test_no_spread_is_a_pass_through(self):
        assert widen_for_unresolved_spread(0.003, None) == pytest.approx(0.003)
        assert widen_for_unresolved_spread(0.003, 0.0) == pytest.approx(0.003)
        assert widen_for_unresolved_spread(None, None) is None

    def test_absent_formal_error_counts_as_zero(self):
        """A peak with no covariance error still gets the spread: the position
        is no better known for the covariance being unavailable."""
        assert widen_for_unresolved_spread(None, 0.004) == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# Stage 5 stamps it on the peak
# ---------------------------------------------------------------------------


class TestStage5Stamping:
    def test_inflation_records_the_spread_and_widens_the_error(self):
        pk = FittedPeak(
            detection_index=0,
            frequency_mhz=30000.0,
            amplitude=1.0,
            frequency_error=0.0002,
            window_id=3,
        )
        wf = FittingResult(window_id=3)
        wf.fitted_peaks = [pk]
        _inflate_merged_frequency_errors(
            wf,
            [{"merged_frequency_mhz": 30000.0, "unresolved_spread_mhz": 0.012}],
        )
        out = wf.fitted_peaks[0]
        assert out.unresolved_spread_mhz == pytest.approx(0.012)
        assert out.frequency_error == pytest.approx(math.hypot(0.0002, 0.012))

    def test_unmerged_peaks_are_untouched(self):
        pk = FittedPeak(
            detection_index=0,
            frequency_mhz=30000.0,
            amplitude=1.0,
            frequency_error=0.0002,
            window_id=3,
        )
        wf = FittingResult(window_id=3)
        wf.fitted_peaks = [pk]
        _inflate_merged_frequency_errors(wf, [])
        assert wf.fitted_peaks[0].unresolved_spread_mhz is None
        assert wf.fitted_peaks[0].frequency_error == pytest.approx(0.0002)


# ---------------------------------------------------------------------------
# It survives a refit -- the defect itself
# ---------------------------------------------------------------------------


class TestSurvivesRefit:
    SPREAD = 0.012  # MHz, ~0.15 resolution elements on this fixture

    def test_refit_re_applies_the_widening_against_its_own_formal_error(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        """The headline claim, pinned against the same refit's own formal
        error rather than against a remembered number: refit the SAME window
        twice, once with a spread planted and once without, and the widened
        error must be exactly the bare one widened by the spread.

        A test that only asserted "the error did not change" would pass on a
        refit that never re-fit anything, and one that compared against the
        pre-refit value would fail for the legitimate reason that the formal
        error moved.
        """
        bare = tmp_path / "bare.ftmw"
        planted = tmp_path / "planted.ftmw"
        shutil.copy(stage5_small_source, bare)
        shutil.copy(stage5_small_source, planted)
        wid = _busiest_window(bare)
        freq = _plant_spread(planted, wid, self.SPREAD)

        refit_window_impl(str(bare), wid)
        refit_window_impl(str(planted), wid)

        pk_bare = _peak_nearest(_window_peaks(bare, wid), freq)
        pk_planted = _peak_nearest(_window_peaks(planted, wid), freq)

        assert pk_bare.unresolved_spread_mhz is None
        assert pk_planted.unresolved_spread_mhz == pytest.approx(self.SPREAD)
        assert pk_bare.frequency_error is not None
        assert pk_planted.frequency_error == pytest.approx(
            math.hypot(float(pk_bare.frequency_error), self.SPREAD), rel=1e-9
        )
        # And the widening is the whole story here, as it is on real merges:
        # the formal error is far smaller than the spread it is widened by.
        assert float(pk_bare.frequency_error) < self.SPREAD

    def test_neighbors_in_the_same_window_are_not_widened(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        planted = tmp_path / "planted.ftmw"
        shutil.copy(stage5_small_source, planted)
        wid = _busiest_window(planted)
        freq = _plant_spread(planted, wid, self.SPREAD)

        refit_window_impl(str(planted), wid)

        others = [
            p
            for p in _window_peaks(planted, wid)
            if abs(float(p.frequency_mhz) - freq) > 1e-6
        ]
        assert others, "this window needs a second peak for the test to mean anything"
        assert all(p.unresolved_spread_mhz is None for p in others)

    def test_a_second_refit_does_not_widen_twice(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        """The re-application must run against the covariance's formal error,
        never against a value that already carries the spread -- otherwise
        each successive edit inflates the line further."""
        planted = tmp_path / "planted.ftmw"
        shutil.copy(stage5_small_source, planted)
        wid = _busiest_window(planted)
        freq = _plant_spread(planted, wid, self.SPREAD)

        refit_window_impl(str(planted), wid)
        once = _peak_nearest(_window_peaks(planted, wid), freq).frequency_error
        refit_window_impl(str(planted), wid)
        twice = _peak_nearest(_window_peaks(planted, wid), freq).frequency_error

        assert once is not None and twice is not None
        assert float(twice) == pytest.approx(float(once), rel=1e-6)

    def test_a_cascade_refit_keeps_it(
        self,
        stage5_multi_source: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The case a user cannot see coming: the merged line's window is
        never named, and is re-fit only because a neighbor's edit propagated
        into it. Needs the multi-window build; the small one is a single
        window, which nothing can cascade into."""
        path = tmp_path / "cascade.ftmw"
        shutil.copy(stage5_multi_source, path)
        fit = _load_fit(path)
        wids = [int(w.window_id) for w in fit.window_fits if w.window_id is not None]
        assert len(wids) >= 2, "need two windows to cascade between"
        primary, dependent = wids[0], wids[1]
        freq = _plant_spread(path, dependent, self.SPREAD)

        orig_succs = s6._cascade_succs

        def fake_succs(window_fits, fit_window_map):
            d = orig_succs(window_fits, fit_window_map)
            d.setdefault(primary, set()).add(dependent)
            return d

        monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

        # Spy on the fitter so the test cannot pass vacuously: if the cascade
        # never re-fits the dependent window, its spread survives for the
        # uninteresting reason that nothing touched it.
        orig_core = s6.refit_window_core
        refit_wids: List[int] = []

        def spy_core(fit_ctx, fit_win, wf, **kwargs):
            refit_wids.append(int(fit_win.window_id))
            return orig_core(fit_ctx, fit_win, wf, **kwargs)

        monkeypatch.setattr(s6, "refit_window_core", spy_core)

        # An identity refit is deliberately NOT dirty (it moves no leakage
        # skirt), so nothing would cascade; the neighbor has to really edit.
        refit_window_impl(str(path), primary, add=[_clear_add_freq(path, primary)])
        assert dependent in refit_wids, (
            "the faked dependency never fired -- this run cannot distinguish "
            "'the spread was re-applied' from 'the window was never re-fit'"
        )
        after = _peak_nearest(_window_peaks(path, dependent), freq)

        assert after.unresolved_spread_mhz == pytest.approx(self.SPREAD), (
            "the dependent window was re-fit by the cascade and lost its "
            "auto-merge widening -- the exact defect, by the path a user "
            "never sees"
        )
        assert after.frequency_error is not None
        assert float(after.frequency_error) >= self.SPREAD


# ---------------------------------------------------------------------------
# Where the spread must NOT be carried
# ---------------------------------------------------------------------------


class TestNotCarried:
    SPREAD = 0.012

    def test_removing_the_line_takes_its_spread_with_it(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "removed.ftmw"
        shutil.copy(stage5_small_source, path)
        wid = _busiest_window(path)
        freq = _plant_spread(path, wid, self.SPREAD)

        refit_window_impl(str(path), wid, remove=[freq])

        assert all(
            p.unresolved_spread_mhz is None for p in _window_peaks(path, wid)
        ), "a surviving line inherited the removed multiplet's spread"

    def test_an_added_line_carries_no_spread(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        """A line the user adds is a new line, not the collapsed multiplet --
        including the split case, where the add asserts the components are
        resolved after all."""
        path = tmp_path / "added.ftmw"
        shutil.copy(stage5_small_source, path)
        wid = _busiest_window(path)
        freq = _plant_spread(path, wid, self.SPREAD)

        target = _clear_add_freq(path, wid)
        assert abs(target - freq) > self.SPREAD, (
            "the add landed on the planted line; this fixture cannot "
            "distinguish an added line from the multiplet"
        )

        refit_window_impl(str(path), wid, add=[target])

        peaks = _window_peaks(path, wid)
        added = _peak_nearest(peaks, target)
        assert added.unresolved_spread_mhz is None
        # ...and the planted line kept its own.
        assert _peak_nearest(peaks, freq).unresolved_spread_mhz == pytest.approx(
            self.SPREAD
        )


# ---------------------------------------------------------------------------
# Recovery for files written before the per-peak column
# ---------------------------------------------------------------------------


def _fit_with_collapse_record(
    *, peak_freq: float, target: float, spread: float, derivation=None
) -> SpectrumFit:
    pk = FittedPeak(
        detection_index=0,
        frequency_mhz=peak_freq,
        amplitude=1.0,
        frequency_error=0.02,
        window_id=7,
        derivation=derivation,
    )
    wf = FittingResult(window_id=7)
    wf.fitted_peaks = [pk]
    fit = SpectrumFit(fitted_peaks=[pk], window_fits=[wf])
    fit.diagnostics["vif_collapse"] = {
        "collapses": [
            {
                "window_id": 7,
                "merged_frequency_mhz": target,
                "unresolved_spread_mhz": spread,
            }
        ]
    }
    return fit


class TestLegacyRecovery:
    TOL = 0.01

    def test_seeds_the_spread_from_diagnostics(self):
        fit = _fit_with_collapse_record(
            peak_freq=30000.0, target=30000.0005, spread=0.013
        )
        n = _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL)
        assert n == 1
        assert fit.window_fits[0].fitted_peaks[
            0
        ].unresolved_spread_mhz == pytest.approx(0.013)

    def test_does_not_rewrite_the_stored_error(self):
        """The stored error on an uncurated legacy file already carries the
        widening; re-applying it here would count the spread twice."""
        fit = _fit_with_collapse_record(peak_freq=30000.0, target=30000.0, spread=0.013)
        _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL)
        assert fit.window_fits[0].fitted_peaks[0].frequency_error == pytest.approx(0.02)

    def test_is_idempotent(self):
        fit = _fit_with_collapse_record(peak_freq=30000.0, target=30000.0, spread=0.013)
        assert (
            _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL) == 1
        )
        assert (
            _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL) == 0
        )

    def test_skips_a_peak_out_of_tolerance(self):
        """The merged line was removed or split away; whatever is left nearby
        is a different line and must not inherit its spread."""
        fit = _fit_with_collapse_record(peak_freq=30000.0, target=30000.5, spread=0.013)
        assert (
            _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL) == 0
        )
        assert fit.window_fits[0].fitted_peaks[0].unresolved_spread_mhz is None

    def test_skips_a_peak_a_stage6_decision_altered(self):
        fit = _fit_with_collapse_record(
            peak_freq=30000.0, target=30000.0, spread=0.013, derivation=4
        )
        assert (
            _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL) == 0
        )

    def test_no_merges_is_a_no_op(self):
        fit = _fit_with_collapse_record(peak_freq=30000.0, target=30000.0, spread=0.013)
        fit.diagnostics.pop("vif_collapse")
        assert (
            _seed_unresolved_spreads_from_diagnostics(fit, snap_tol_mhz=self.TOL) == 0
        )
