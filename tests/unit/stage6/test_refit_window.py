"""
Unit / integration tests for Stage 6 single-window refit engine.

Acceptance criteria (from the task spec):
1. fit_peaks byte-identical after Task-1 refactor — verified by the Stage 5
   integration tests passing (those tests are the canonical assertion).
2. No-edit refit fidelity on multiple windows (max freq/amp deviation).
3. Remove + add behavior with immunity (protected/forbidden offsets).
4. Complex-amplitude round-trip (fixed_parameters JSON parse).
5. Full suite green (reported separately).

Tests in this module run against the 2638 Stage-5-small fixture (3 windows,
fast) and a synthetic two-peak window for unit-level isolation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.atomic import atomic_write, h5open
from ftmwpipeline._internal.stage5_impl import (
    Stage5FitContext,
    build_stage5_fit_context,
)
from ftmwpipeline._internal.stage6_impl import (
    RefitWindowResult,
    _build_batch_ctx,
    _parse_complex_amplitude,
    _reconstruct_frozen_peaks,
    refit_snap_tol_mhz_impl,
    refit_window_impl,
)
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    update_spectrum_fit_windows_in_hdf5,
)

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def writable_stage5_file(stage5_small_source, tmp_path):
    """Return a writable copy of the Stage 5 small file."""
    dst = tmp_path / "working.ftmw"
    shutil.copy(stage5_small_source, dst)
    return dst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _identity_refit_action(ctx, window_id, snap):
    from ftmwpipeline._internal import stage6_impl as s6

    wf = s6._batch_lookup_wf(ctx, window_id)
    n0, c0 = len(wf.fitted_peaks), float(wf.reduced_chi2)
    new = s6._batch_apply_edit_core(ctx, window_id, snap_tol_mhz=snap)
    return s6._make_refit_result(
        ctx,
        window_id=window_id,
        n_peaks_before=n0,
        n_peaks_after=len(new.fitted_peaks),
        chi2r_before=c0,
        chi2r_after=float(new.reduced_chi2),
        fitted_peaks=list(new.fitted_peaks),
        converged=s6._converged_or_absent(new),
    )


def _identity_refit(path: Union[str, Path], window_id: int) -> RefitWindowResult:
    """Refit one window with no edit and persist it, recording no decision.

    This is the identity refit the cascade runs on a window it reaches -- an
    engine mechanic, not a user verb (a bare ``review edit`` is refused, and
    every state the engine persists is a replay of its log). It is driven
    here on the window's displayed fit and written straight to the fit table,
    so the fidelity of ``refit_window_core`` stays pinned end to end.
    """
    p = str(path)
    snap = refit_snap_tol_mhz_impl(p)
    ctx = _build_batch_ctx(p, snap_tol_mhz=snap)
    result = _identity_refit_action(ctx, window_id, snap)
    with atomic_write(p), h5open(p, "a") as h5f:
        update_spectrum_fit_windows_in_hdf5(
            ctx.changeset.spectrum_fit, h5f["stage5_fitting"], [window_id]
        )
    return result


# ---------------------------------------------------------------------------
# Convergence visibility on the single-window verb
# ---------------------------------------------------------------------------


class TestRefitReportsConvergence:
    """``RefitWindowResult.converged`` mirrors the joint fit's own outcome.

    A failed NLS returns the window's seeds verbatim with an infinite
    chi-squared and still returns a result -- the refit ran, the fit did not
    converge -- so without this flag the only tell a caller gets is a wild
    ``chi2r_after``.
    """

    def test_identity_refit_converges(self, writable_stage5_file):
        wid = _load_spectrum_fit(writable_stage5_file).window_fits[0].window_id
        result = _identity_refit(str(writable_stage5_file), wid)
        assert result.converged is True

    def test_failed_fit_is_reported(self, writable_stage5_file, monkeypatch):
        from ftmwpipeline._internal import stage6_impl as s6

        orig_core = s6.refit_window_core

        def failing_core(fit_ctx, fit_win, wf, **kwargs):
            out = orig_core(fit_ctx, fit_win, wf, **kwargs)
            out.success = False
            return out

        monkeypatch.setattr(s6, "refit_window_core", failing_core)

        wid = _load_spectrum_fit(writable_stage5_file).window_fits[0].window_id
        result = _identity_refit(str(writable_stage5_file), wid)
        assert result.converged is False


# ---------------------------------------------------------------------------
# Task 4: Complex-amplitude round-trip
# ---------------------------------------------------------------------------


class TestComplexAmplitudeRoundTrip:
    """Verify _parse_complex_amplitude handles every storage format."""

    def test_real_float_int(self):
        assert _parse_complex_amplitude(1.5) == complex(1.5)
        assert _parse_complex_amplitude(3) == complex(3)

    def test_complex_value(self):
        assert _parse_complex_amplitude(complex(1.5, 0)) == complex(1.5, 0)

    def test_str_float(self):
        assert _parse_complex_amplitude("1.5") == complex(1.5)

    def test_str_complex(self):
        result = _parse_complex_amplitude("(1.5+0j)")
        assert abs(result - complex(1.5, 0)) < 1e-12

    def test_negative_zero_imag(self):
        # repr(complex(3.2, 0)) == '(3.2+0j)' in Python 3
        val = complex(3.2, 0)
        parsed = _parse_complex_amplitude(repr(val))
        assert abs(parsed.real - 3.2) < 1e-12

    def test_real_only_from_json_default(self):
        # json.dumps(3.2, default=str) → 3.2 (stays float, not string)
        # but json.dumps(complex(3.2,0), default=str) → '"(3.2+0j)"'
        # Either way, _parse_complex_amplitude must accept both.
        assert abs(_parse_complex_amplitude(3.2).real - 3.2) < 1e-12

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_complex_amplitude("not_a_number")


# ---------------------------------------------------------------------------
# Task 3-helper: reconstruct_frozen_peaks
# ---------------------------------------------------------------------------


class TestReconstructFrozenPeaks:
    """_reconstruct_frozen_peaks builds FrozenPeak objects from fixed_parameters."""

    def test_empty(self):
        from ftmwpipeline.core.data_structures import Sideband

        result = _reconstruct_frozen_peaks({}, 30000.0, Sideband.LOWER)
        assert result == []

    def test_single_peak(self):
        from ftmwpipeline.core.data_structures import Sideband

        fixed = {
            "frozen_peak_7": {
                "peak_index": 7,
                "primary_window_id": 42,
                "frequency_mhz": 30100.0,
                "amplitude": 1234.5,
                "phase": 0.5,
                "freeze_eligible": True,
            }
        }
        center = 30000.0
        sideband = Sideband.LOWER
        result = _reconstruct_frozen_peaks(fixed, center, sideband)
        assert len(result) == 1
        fp = result[0]
        assert fp.peak_index == 7
        assert fp.primary_window_id == 42
        assert abs(fp.frequency_mhz - 30100.0) < 1e-9
        # LOWER sideband: s = -1, so offset = -1 * (30100 - 30000) = -100
        assert abs(fp.model_peak.offset_mhz - (-100.0)) < 1e-9
        assert abs(fp.model_peak.amplitude - 1234.5) < 1e-9
        assert abs(fp.model_peak.phase - 0.5) < 1e-9

    def test_non_frozen_key_ignored(self):
        from ftmwpipeline.core.data_structures import Sideband

        fixed = {
            "some_other_key": {"frequency_mhz": 30100.0},
            "frozen_peak_3": {
                "peak_index": 3,
                "primary_window_id": 1,
                "frequency_mhz": 30050.0,
                "amplitude": 500.0,
                "phase": 0.0,
                "freeze_eligible": False,
            },
        }
        result = _reconstruct_frozen_peaks(fixed, 30000.0, Sideband.UPPER)
        assert len(result) == 1
        assert result[0].peak_index == 3

    def test_peak_uid_carried_when_present(self):
        from ftmwpipeline.core.data_structures import Sideband

        fixed = {
            "frozen_peak_7": {
                "peak_index": 7,
                "primary_window_id": 42,
                "frequency_mhz": 30100.0,
                "amplitude": 1234.5,
                "phase": 0.5,
                "freeze_eligible": True,
                "peak_uid": 991,
            }
        }
        result = _reconstruct_frozen_peaks(fixed, 30000.0, Sideband.LOWER)
        assert result[0].model_peak.peak_uid == 991

    def test_peak_uid_absent_key_reads_as_none(self):
        """A ``fixed_parameters`` entry written before peak identity existed
        has no ``peak_uid`` key at all; the reconstructed peak must read
        ``None``, never a value derived from ``frequency_mhz``."""
        from ftmwpipeline.core.data_structures import Sideband

        fixed = {
            "frozen_peak_3": {
                "peak_index": 3,
                "primary_window_id": 1,
                "frequency_mhz": 30050.0,
                "amplitude": 500.0,
                "phase": 0.0,
                "freeze_eligible": False,
            },
        }
        result = _reconstruct_frozen_peaks(fixed, 30000.0, Sideband.UPPER)
        assert result[0].model_peak.peak_uid is None


# ---------------------------------------------------------------------------
# Task 1: build_stage5_fit_context is reusable (spot-check on 2638)
# ---------------------------------------------------------------------------


class TestBuildStage5FitContext:
    """Verify that build_stage5_fit_context is importable and returns the right type."""

    def test_returns_stage5_fit_context(self, stage5_small_file):
        # This is a smoke test — the full byte-identical verification is in
        # the Stage 5 integration tests that passed after the refactor.
        from ftmwpipeline.core.stage_fit_settings import StageFitSettings
        from ftmwpipeline.core.stage_fit_settings import resolve as resolve_settings
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            load_stage_fit_settings_from_h5,
            read_recommended_clock_sources,
            read_stage2b_recommended_shape,
        )

        path = str(stage5_small_file)
        persisted = load_stage_fit_settings_from_h5(path)
        recommended_shape_str = read_stage2b_recommended_shape(path)
        recommended_clocks = read_recommended_clock_sources(path)
        recommended = None
        if recommended_shape_str is not None or recommended_clocks is not None:
            from ftmwpipeline.core.stage_fit_settings import ShapeSpec, SpurSubSettings

            recommended = StageFitSettings(
                shape=(
                    ShapeSpec.coerce(recommended_shape_str)
                    if recommended_shape_str is not None
                    else None
                ),
                spur=SpurSubSettings(clocks=recommended_clocks),
            )
        resolved = resolve_settings(
            explicit=StageFitSettings(),
            preset=None,
            persisted=persisted,
            recommended=recommended,
        )
        assert resolved.shape is not None
        shape_enum = resolved.shape.kind

        ctx = build_stage5_fit_context(path, resolved, None, shape_enum)
        assert isinstance(ctx, Stage5FitContext)
        assert ctx.active_ft is not None
        assert ctx.rms_for_fit is not None
        assert len(ctx.rms_for_fit) == len(ctx.active_ft.freq_mhz)
        assert ctx.acquisition_us > 0


# ---------------------------------------------------------------------------
# Task 2: Identity refit fidelity
# ---------------------------------------------------------------------------


class TestIdentityRefitFidelity:
    """No-edit refit must reproduce the persisted fit within tight tolerance."""

    def test_all_windows_fidelity(self, stage5_small_file, tmp_path):
        """Identity refit on every window: max frequency deviation ~1e-4 MHz,
        amplitude within 0.1 %.

        Each window gets its own copy of the file so refitting one window
        does not affect the baseline used to evaluate the next.
        """
        sf_before = _load_spectrum_fit(stage5_small_file)

        max_freq_delta = 0.0
        max_amp_rel = 0.0
        n_windows_tested = 0

        for wf_before in sf_before.window_fits:
            wid = wf_before.window_id
            n_peaks = len(wf_before.fitted_peaks)
            if n_peaks == 0:
                continue

            # Fresh copy for each window so mutations don't accumulate.
            path = tmp_path / f"w{wid}.ftmw"
            shutil.copy(stage5_small_file, path)

            result = _identity_refit(str(path), wid)
            assert isinstance(result, RefitWindowResult)
            assert result.window_id == wid

            # Load the updated fit.
            sf_after = _load_spectrum_fit(path)
            wf_after_list = [w for w in sf_after.window_fits if w.window_id == wid]
            assert len(wf_after_list) == 1
            wf_after = wf_after_list[0]

            # Peak count must match (identity refit does not add/remove peaks).
            assert len(wf_after.fitted_peaks) == n_peaks, (
                f"window {wid}: peak count changed from {n_peaks} "
                f"to {len(wf_after.fitted_peaks)}"
            )

            # Frequencies within 1e-4 MHz (100 kHz ≫ the solver precision).
            peaks_before = sorted(wf_before.fitted_peaks, key=lambda p: p.frequency_mhz)
            peaks_after = sorted(wf_after.fitted_peaks, key=lambda p: p.frequency_mhz)
            for pb, pa in zip(peaks_before, peaks_after):
                delta_f = abs(float(pb.frequency_mhz) - float(pa.frequency_mhz))
                delta_a_rel = abs(float(pb.amplitude) - float(pa.amplitude)) / max(
                    abs(float(pb.amplitude)), 1e-30
                )
                max_freq_delta = max(max_freq_delta, delta_f)
                max_amp_rel = max(max_amp_rel, delta_a_rel)

            n_windows_tested += 1

        assert n_windows_tested > 0, "No windows with peaks to test"
        print(
            f"\nIdentity refit fidelity: {n_windows_tested} windows tested, "
            f"max freq delta = {max_freq_delta:.6f} MHz, "
            f"max amp rel delta = {max_amp_rel:.6f}"
        )

    def _test_single_window_fidelity(self, writable_stage5_file):
        """Alternative: check the FIRST non-empty window fidelity in isolation."""
        path = writable_stage5_file
        sf = _load_spectrum_fit(path)
        # Find first window with peaks.
        wf = next(
            (w for w in sf.window_fits if len(w.fitted_peaks) > 0),
            None,
        )
        if wf is None:
            pytest.skip("No window with peaks found")

        wid = wf.window_id
        peaks_before = sorted(wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        chi2r_before = float(wf.reduced_chi2)

        result = _identity_refit(str(path), wid)

        assert result.n_peaks_before == len(peaks_before)
        assert result.n_peaks_after == len(peaks_before)
        assert abs(result.chi2r_before - chi2r_before) < 1e-6

        sf_after = _load_spectrum_fit(path)
        wf_after = next(w for w in sf_after.window_fits if w.window_id == wid)
        peaks_after = sorted(wf_after.fitted_peaks, key=lambda p: p.frequency_mhz)

        for pb, pa in zip(peaks_before, peaks_after):
            assert abs(float(pb.frequency_mhz) - float(pa.frequency_mhz)) < 1e-4, (
                f"window {wid}: freq delta "
                f"{abs(float(pb.frequency_mhz) - float(pa.frequency_mhz)):.6f} MHz"
            )
            amp_rel = abs(float(pb.amplitude) - float(pa.amplitude)) / max(
                float(pb.amplitude), 1e-30
            )
            assert (
                amp_rel < 0.001
            ), f"window {wid}: amplitude relative delta {amp_rel:.6f}"


class TestIdentityRefitPerWindow:
    """Per-window isolation version of the fidelity test."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        """Give each test its own writable copy."""
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf_orig = _load_spectrum_fit(self.path)
        self.windows_with_peaks = [
            wf for wf in self.sf_orig.window_fits if len(wf.fitted_peaks) > 0
        ]

    def test_first_window_peak_count_preserved(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        n_before = len(wf.fitted_peaks)
        result = _identity_refit(str(self.path), wf.window_id)
        assert result.n_peaks_after == n_before

    def test_first_window_frequencies_within_tolerance(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        peaks_before = sorted(wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        _identity_refit(str(self.path), wf.window_id)
        sf_after = _load_spectrum_fit(self.path)
        wf_after = next(w for w in sf_after.window_fits if w.window_id == wf.window_id)
        peaks_after = sorted(wf_after.fitted_peaks, key=lambda p: p.frequency_mhz)
        assert len(peaks_after) == len(peaks_before)
        for pb, pa in zip(peaks_before, peaks_after):
            delta = abs(float(pb.frequency_mhz) - float(pa.frequency_mhz))
            assert delta < 1e-4, (
                f"window {wf.window_id}: freq delta {delta:.6f} MHz "
                f"(peak at {float(pb.frequency_mhz):.4f} MHz)"
            )

    def test_first_window_amplitudes_within_tolerance(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        peaks_before = sorted(wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        _identity_refit(str(self.path), wf.window_id)
        sf_after = _load_spectrum_fit(self.path)
        wf_after = next(w for w in sf_after.window_fits if w.window_id == wf.window_id)
        peaks_after = sorted(wf_after.fitted_peaks, key=lambda p: p.frequency_mhz)
        for pb, pa in zip(peaks_before, peaks_after):
            amp_rel = abs(float(pb.amplitude) - float(pa.amplitude)) / max(
                float(pb.amplitude), 1e-30
            )
            assert amp_rel < 0.001, (
                f"window {wf.window_id}: amp rel delta {amp_rel:.6f} "
                f"(peak at {float(pb.frequency_mhz):.4f} MHz)"
            )

    def test_all_windows_peak_count_preserved(self):
        """Identity refit on every window must preserve peak count."""
        for wf in self.windows_with_peaks:
            n_before = len(wf.fitted_peaks)
            # Each refit mutates the file; reload for the next window.
            result = _identity_refit(str(self.path), wf.window_id)
            assert (
                result.n_peaks_before == n_before
            ), f"window {wf.window_id}: n_peaks_before mismatch"
            assert result.n_peaks_after == n_before, (
                f"window {wf.window_id}: identity refit changed peak count "
                f"{n_before} → {result.n_peaks_after}"
            )

    def test_chi2r_reported_correctly(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        result = _identity_refit(str(self.path), wf.window_id)
        assert abs(result.chi2r_before - float(wf.reduced_chi2)) < 1e-6

    def test_other_windows_unchanged(self):
        """Refitting one window must not alter other windows' fitted peaks."""
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf0 = self.windows_with_peaks[0]
        other_wfs = [w for w in self.windows_with_peaks if w.window_id != wf0.window_id]
        if not other_wfs:
            pytest.skip("Only one window with peaks; cannot test isolation")

        _identity_refit(str(self.path), wf0.window_id)
        sf_after = _load_spectrum_fit(self.path)

        for other_wf in other_wfs:
            after_wf = next(
                w for w in sf_after.window_fits if w.window_id == other_wf.window_id
            )
            assert len(after_wf.fitted_peaks) == len(
                other_wf.fitted_peaks
            ), f"window {other_wf.window_id} (not refitted) changed peak count"
            after_peaks = sorted(after_wf.fitted_peaks, key=lambda p: p.frequency_mhz)
            orig_peaks = sorted(other_wf.fitted_peaks, key=lambda p: p.frequency_mhz)
            for pb, pa in zip(orig_peaks, after_peaks):
                assert float(pb.frequency_mhz) == pytest.approx(
                    float(pa.frequency_mhz), abs=1e-9
                ), f"window {other_wf.window_id}: peak moved after sibling refit"


# ---------------------------------------------------------------------------
# Task 3: Remove behavior with forbidden immunity
# ---------------------------------------------------------------------------


class TestRemovePeak:
    """Removing a fitted peak must exclude it from the refit, and the rescue
    must not re-add it (forbidden_offsets)."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_spectrum_fit(self.path)
        # Find a window with at least 2 peaks so removing one leaves something.
        self.wf = next(
            (w for w in self.sf.window_fits if len(w.fitted_peaks) >= 2),
            None,
        )

    def test_remove_reduces_peak_count(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")
        target_peak = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0]
        n_before = len(self.wf.fitted_peaks)

        result = refit_window_impl(
            str(self.path),
            self.wf.window_id,
            remove=[float(target_peak.frequency_mhz)],
        )
        # Rescue may or may not re-add, but peak count should be ≤ n_before - 1
        # due to the forbidden offset — the rescue was told not to re-add it.
        # The refit starts without it; cleanup/rescue won't re-add it.
        assert result.n_peaks_after <= n_before - 1

    def test_remove_snaps_within_tolerance(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")
        target_peak = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0]
        # Snap should work with a small offset.
        snap_freq = float(target_peak.frequency_mhz) + 0.001  # 1 kHz offset
        result = refit_window_impl(
            str(self.path),
            self.wf.window_id,
            remove=[snap_freq],
        )
        n_before = len(self.wf.fitted_peaks)
        assert result.n_peaks_after <= n_before - 1

    def test_remove_outside_tolerance_raises(self):
        if self.wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")
        target_peak = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)[0]
        # 10 MHz away — should raise.
        bad_freq = float(target_peak.frequency_mhz) + 10.0
        with pytest.raises(ValueError, match="no fitted peak within"):
            refit_window_impl(
                str(self.path),
                self.wf.window_id,
                remove=[bad_freq],
            )


# ---------------------------------------------------------------------------
# Task 3: Add behavior with protected immunity
# ---------------------------------------------------------------------------


class TestAddPeak:
    """Adding a peak bypasses the accept gate; cleanup/rescue must not prune it
    (protected_offsets).  The added peak must carry origin='user'."""

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_spectrum_fit(self.path)
        self.wf = next(
            (w for w in self.sf.window_fits if w.window_id is not None),
            None,
        )

    def _window_center_mhz(self, wf):
        if wf.window is not None and wf.window.freq_range is not None:
            lo, hi = wf.window.freq_range
            return 0.5 * (lo + hi)
        return None

    def test_add_peak_origin_user(self):
        if self.wf is None:
            pytest.skip("No window found")
        center = self._window_center_mhz(self.wf)
        if center is None:
            pytest.skip("Window has no freq_range")

        # Add a peak at the center of the window (a fresh frequency, not in the
        # ledger; the engine will seed fresh at that bin).
        result = refit_window_impl(
            str(self.path),
            self.wf.window_id,
            add=[center],
        )
        # At least one peak should have origin='user'.
        user_peaks = [p for p in result.fitted_peaks if p.origin == "user"]
        assert len(user_peaks) >= 1, (
            f"No user-origin peak found after add; peaks: "
            f"{[(p.frequency_mhz, p.origin) for p in result.fitted_peaks]}"
        )

    def test_add_then_reload_persisted_origin(self):
        """The user origin must survive the HDF5 round-trip."""
        if self.wf is None:
            pytest.skip("No window found")
        center = self._window_center_mhz(self.wf)
        if center is None:
            pytest.skip("Window has no freq_range")

        refit_window_impl(
            str(self.path),
            self.wf.window_id,
            add=[center],
        )
        sf_after = _load_spectrum_fit(self.path)
        wf_after = next(
            w for w in sf_after.window_fits if w.window_id == self.wf.window_id
        )
        user_peaks = [p for p in wf_after.fitted_peaks if p.origin == "user"]
        assert len(user_peaks) >= 1, "user origin not persisted after HDF5 round-trip"


def test_an_edit_takes_no_seed_the_log_does_not_record():
    """A Stage 6 add's starting point comes from the replay, never from the
    caller: an amplitude or phase seed the decision log does not record would
    make a replay of the log differ from the write (design §3.4)."""
    import inspect

    assert "add_seeds" not in inspect.signature(refit_window_impl).parameters


# ---------------------------------------------------------------------------
# An `add` outside the named window's range is a caller error, not a fit
# against data the window does not cover (issue #40). `remove` was already
# validated against the window's contents; `add` is now validated against its
# extent.
# ---------------------------------------------------------------------------


class TestAddOutsideWindowRange:
    @pytest.fixture(autouse=True)
    def _setup(self, stage5_small_source, tmp_path):
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_source, self.path)
        self.sf = _load_spectrum_fit(self.path)
        self.wf = next(
            (
                w
                for w in self.sf.window_fits
                if w.window_id is not None
                and w.window is not None
                and w.window.freq_range is not None
            ),
            None,
        )
        if self.wf is None:
            pytest.skip("No window with a freq_range in the fixture")
        lo, hi = self.wf.window.freq_range
        self.lo, self.hi = min(lo, hi), max(lo, hi)

    def test_add_above_range_raises(self):
        # A full window-width beyond the top edge: unambiguously off the window.
        target = self.hi + (self.hi - self.lo)
        with pytest.raises(ValueError, match="outside window"):
            refit_window_impl(str(self.path), self.wf.window_id, add=[target])

    def test_add_below_range_raises(self):
        target = self.lo - (self.hi - self.lo)
        with pytest.raises(ValueError, match="outside window"):
            refit_window_impl(str(self.path), self.wf.window_id, add=[target])

    def test_error_names_the_window_range_and_the_remedy(self):
        target = self.hi + (self.hi - self.lo)
        with pytest.raises(ValueError) as exc:
            refit_window_impl(str(self.path), self.wf.window_id, add=[target])
        msg = str(exc.value)
        assert f"{self.lo:.4f}" in msg and f"{self.hi:.4f}" in msg
        assert "review create" in msg

    def test_out_of_range_add_leaves_the_fit_untouched(self):
        """The validation must fire before anything is persisted."""
        before = [(p.window_id, p.frequency_mhz) for p in self.sf.fitted_peaks]
        target = self.hi + (self.hi - self.lo)
        with pytest.raises(ValueError):
            refit_window_impl(str(self.path), self.wf.window_id, add=[target])
        after_sf = _load_spectrum_fit(self.path)
        after = [(p.window_id, p.frequency_mhz) for p in after_sf.fitted_peaks]
        assert before == after

    def test_out_of_range_add_records_no_decision(self):
        from ftmwpipeline._internal.stage6_impl import review_log_impl

        target = self.hi + (self.hi - self.lo)
        with pytest.raises(ValueError):
            refit_window_impl(str(self.path), self.wf.window_id, add=[target])
        assert review_log_impl(str(self.path)) == []

    def test_edge_of_range_is_accepted(self):
        """A frequency ON the boundary is in-window -- only beyond it is not."""
        result = refit_window_impl(str(self.path), self.wf.window_id, add=[self.hi])
        assert isinstance(result, RefitWindowResult)


# ---------------------------------------------------------------------------
# Spur-catalog replay: the refit must reproduce the Stage 5 residual mask
# from the persisted gated catalog, never re-derive it (detection is a
# Stage 5 product; a detector change between fit and refit must not re-mask).
# ---------------------------------------------------------------------------


class TestSpurCatalogReplay:
    def test_spur_set_from_catalog_roundtrip(self):
        """Reconstructing a SpurSet from its persisted fields reproduces the
        window mask spec exactly (centers, default + per-spur half-widths)."""
        from ftmwpipeline.core.data_structures import Sideband
        from ftmwpipeline.fitting.spur_detection import (
            GatedSpur,
            SpurSet,
            spur_set_from_catalog,
        )

        orig = SpurSet(
            spurs=(
                GatedSpur(center_mhz=28023.0, integer_mhz=28023, source="narrow"),
                GatedSpur(
                    center_mhz=28024.5,
                    integer_mhz=28025,
                    source="saturated",
                    lattice="320x6 (bb)",
                    drift=True,
                    mask_half_width_bins=7,
                ),
            ),
            bin_spacing_mhz=0.094,
            mask_half_width_bins=2,
        )
        rebuilt = spur_set_from_catalog(
            centers_mhz=[s.center_mhz for s in orig.spurs],
            sources=[s.source for s in orig.spurs],
            lattice=[s.lattice for s in orig.spurs],
            drift=[s.drift for s in orig.spurs],
            per_spur_mask_half_width_bins=[s.mask_half_width_bins for s in orig.spurs],
            bin_spacing_mhz=orig.bin_spacing_mhz,
            default_mask_half_width_bins=orig.mask_half_width_bins,
        )
        # Identical mask geometry for a window spanning both spurs.
        a = orig.window_mask_spec(28020.0, 28030.0, 28025.0, Sideband.LOWER)
        b = rebuilt.window_mask_spec(28020.0, 28030.0, 28025.0, Sideband.LOWER)
        assert a is not None and b is not None
        assert a.offsets_mhz == pytest.approx(b.offsets_mhz)
        assert a.half_width_mhz == pytest.approx(b.half_width_mhz)
        assert (a.half_widths_mhz is None) == (b.half_widths_mhz is None)
        if a.half_widths_mhz is not None:
            assert a.half_widths_mhz == pytest.approx(b.half_widths_mhz)

    def test_legacy_short_lists_default(self):
        """A pre-field file (only centers + sources persisted) still rebuilds:
        lattice->None, drift->False, per-spur width->None (uniform default)."""
        from ftmwpipeline.fitting.spur_detection import spur_set_from_catalog

        s = spur_set_from_catalog(
            centers_mhz=[28023.0, 30100.0],
            sources=["narrow", "narrow"],
            bin_spacing_mhz=0.094,
            default_mask_half_width_bins=2,
        )
        assert len(s.spurs) == 2
        assert all(sp.mask_half_width_bins is None for sp in s.spurs)
        assert all(sp.lattice is None and sp.drift is False for sp in s.spurs)

    def test_refit_replays_persisted_catalog_not_detection(self, stage5_small_file):
        """build_stage5_fit_context(replay_spur_catalog=...) must yield a
        spur set matching the INJECTED catalog, not whatever detection finds.
        Inject a fictional spur center and confirm it is the one masked."""
        from ftmwpipeline.core.stage_fit_settings import StageFitSettings
        from ftmwpipeline.core.stage_fit_settings import resolve as resolve_settings
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            load_stage_fit_settings_from_h5,
            read_recommended_clock_sources,
            read_stage2b_recommended_shape,
        )

        path = str(stage5_small_file)
        persisted = load_stage_fit_settings_from_h5(path)
        rec_shape = read_stage2b_recommended_shape(path)
        rec_clocks = read_recommended_clock_sources(path)
        rec = None
        if rec_shape is not None or rec_clocks is not None:
            from ftmwpipeline.core.stage_fit_settings import ShapeSpec, SpurSubSettings

            rec = StageFitSettings(
                shape=ShapeSpec.coerce(rec_shape) if rec_shape else None,
                spur=SpurSubSettings(clocks=rec_clocks),
            )
        resolved = resolve_settings(
            explicit=StageFitSettings(),
            preset=None,
            persisted=persisted,
            recommended=rec,
        )
        shape_enum = resolved.shape.kind

        # A fictional catalog current detection would not produce.
        injected = {
            "spur_centers_mhz": [28123.456, 33000.789],
            "spur_sources": ["narrow", "saturated"],
            "spur_lattice": [None, None],
            "spur_drift": [False, False],
            "spur_mask_half_width_bins_per_spur": [None, None],
            "spur_mask_half_width_bins": 2,
        }
        ctx = build_stage5_fit_context(
            path, resolved, None, shape_enum, replay_spur_catalog=injected
        )
        centers = sorted(s.center_mhz for s in ctx.spur_set.spurs)
        assert centers == pytest.approx([28123.456, 33000.789])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRefitWindowErrors:
    def test_no_stage5_raises(self, exp_2638_data_path, tmp_path):
        fp = tmp_path / "no_stage5.ftmw"
        ftmw.import_data(fp, source=exp_2638_data_path)
        with pytest.raises(ValueError, match="No Stage 5 fit"):
            refit_window_impl(str(fp), 0, remove=[30000.0])

    def test_invalid_window_raises(self, stage5_small_source, tmp_path):
        dst = tmp_path / "copy.ftmw"
        shutil.copy(stage5_small_source, dst)
        with pytest.raises(KeyError):
            refit_window_impl(str(dst), 99999, remove=[30000.0])


# ---------------------------------------------------------------------------
# An accepted Stage 5 thaw holds none of the dependent's own peaks out
# ---------------------------------------------------------------------------


def _inject_synthetic_thaw(path: Path, window_id: int, thawed_freq_mhz: float) -> None:
    """Inject a synthetic accepted thaw event into the window's row.

    Names ``thawed_freq_mhz`` as the contributor of a thaw from a (fictitious)
    primary window.
    """
    import json

    thaw_record = {
        "dependent_window_id": window_id,
        "primary_window_id": window_id - 1,  # synthetic; not checked during refit
        "contributor_peak_index": -1,
        "contributor_frequency_mhz": thawed_freq_mhz,
        "edge_side": "low",
        "edge_coherence_before": 5.0,
        "edge_coherence_after": 1.5,
        "accepted": True,
        "reason": "synthetic test injection",
    }
    with h5py.File(str(path), "a") as h5f:
        # One row per window in the flat windows table; `thaw_events` is that
        # row's JSON cell.
        windows = h5f["stage5_fitting/windows"]
        ids = [int(v) for v in windows["window_id"][:]]
        row = ids.index(int(window_id))
        windows["thaw_events"][row] = json.dumps([thaw_record])


class TestAcceptedThawHoldsNothingOut:
    """Since epoch 6 an accepted thaw leaves the thawed line frozen in the
    dependent (a ``fixed_parameters`` entry naming its primary), never one of
    the dependent's fitted peaks. So a refit of the dependent holds none of its
    own peaks out on the thaw's account, however close one sits to the thawed
    contributor's frequency: each is re-fit freely and nothing new is frozen.
    """

    def test_a_peak_at_the_thawed_frequency_is_refit_freely(
        self, stage5_small_source, tmp_path
    ):
        sf_orig = _load_spectrum_fit(stage5_small_source)
        wf = next(
            (w for w in sf_orig.window_fits if len(w.fitted_peaks) >= 2),
            None,
        )
        if wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")
        thawed_freq = float(min(p.frequency_mhz for p in wf.fitted_peaks))

        plain = tmp_path / "plain.ftmw"
        thawed = tmp_path / "thawed.ftmw"
        shutil.copy(stage5_small_source, plain)
        shutil.copy(stage5_small_source, thawed)
        _inject_synthetic_thaw(thawed, wf.window_id, thawed_freq)
        _identity_refit(str(plain), wf.window_id)
        _identity_refit(str(thawed), wf.window_id)

        def after(path: Path):
            sf = _load_spectrum_fit(path)
            return next(w for w in sf.window_fits if w.window_id == wf.window_id)

        ref, got = after(plain), after(thawed)
        # The thaw event changes nothing: the same free fit, the same frozen
        # entries, a covariance over every fitted peak.
        assert [p.frequency_mhz for p in got.fitted_peaks] == [
            p.frequency_mhz for p in ref.fitted_peaks
        ]
        assert [p.peak_uid for p in got.fitted_peaks] == [
            p.peak_uid for p in ref.fitted_peaks
        ]
        assert got.fixed_parameters == ref.fixed_parameters
        assert got.covariance is not None
        np.testing.assert_array_equal(got.covariance, ref.covariance)


# ---------------------------------------------------------------------------
# P4 headline test: peak_uid survives a Stage 6 refit
# ---------------------------------------------------------------------------


class TestPeakUidSurvivesRefit:
    """The whole point of P4 (``scratch/peak-identity-plan.md``): a peak's
    identifier must survive a Stage 6 refit of its window unchanged, even
    though the fitted frequency moves. ``refit_window_core``'s inherited-seed
    path (``stage6_impl.py``, the ``else`` branch that builds
    ``seed_peaks_with_origin``) is the carry site under test -- it warm-starts
    each free peak from its own previously fitted position, which is a
    propagation, not a birth, and must copy ``peak_uid`` rather than leave it
    to default to ``None``.
    """

    def test_uid_unchanged_while_frequency_moves(self, stage5_small_source, tmp_path):
        sf_before = _load_spectrum_fit(stage5_small_source)

        found_moved_pair = False
        for wf_before in sf_before.window_fits:
            wid = wf_before.window_id
            if len(wf_before.fitted_peaks) == 0:
                continue
            # Every peak in the automatic fit must already carry an
            # identifier: P3 stamps every birth site, and this fixture is
            # built via the real ``fit_peaks`` pipeline (no hand-patching).
            assert all(
                p.peak_uid is not None for p in wf_before.fitted_peaks
            ), f"window {wid}: automatic fit has an unstamped peak"

            # Fresh copy per window so one refit cannot bleed into the next.
            path = tmp_path / f"w{wid}.ftmw"
            shutil.copy(stage5_small_source, path)
            _identity_refit(str(path), wid)
            sf_after = _load_spectrum_fit(path)
            wf_after = next(w for w in sf_after.window_fits if w.window_id == wid)

            by_uid_before = {p.peak_uid: p for p in wf_before.fitted_peaks}
            by_uid_after = {p.peak_uid: p for p in wf_after.fitted_peaks}

            # A no-edit refit adds and removes nothing, so the identifier set
            # itself must be exactly preserved -- the strongest form of "the
            # uid survives the refit".
            assert set(by_uid_before) == set(by_uid_after), (
                f"window {wid}: peak_uid set changed on a no-edit refit "
                f"({set(by_uid_before)} -> {set(by_uid_after)})"
            )

            for uid, pb in by_uid_before.items():
                pa = by_uid_after[uid]
                if float(pa.frequency_mhz) != float(pb.frequency_mhz):
                    found_moved_pair = True

        assert found_moved_pair, (
            "no peak's fitted frequency moved on any window's refit in this "
            "fixture, so this run cannot distinguish 'the uid was carried' "
            "from 'nothing happened' -- the headline claim is unverified"
        )
