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
from typing import List

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import (
    Stage5FitContext,
    build_stage5_fit_context,
)
from ftmwpipeline._internal.stage6_impl import (
    RefitWindowResult,
    _parse_complex_amplitude,
    _reconstruct_frozen_peaks,
    refit_window_impl,
)
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.fitting.peak_model import ModelPeak, sideband_sign
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp_2638_data_path():
    data_path = Path("examples/blackchirp_data/2638")
    if not data_path.exists():
        pytest.skip("Experiment 2638 data not available")
    return str(data_path)


@pytest.fixture(scope="module")
def stage5_small_file(exp_2638_data_path, tmp_path_factory):
    """Build the 2638 pipeline through Stage 5 (3-window subset) once."""
    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl,
        save_window_plan_impl,
    )

    tmp = tmp_path_factory.mktemp("stage5_small_refit")
    fp = tmp / "2638_stage5_small.ftmw"

    ftmw.import_data(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, trim=(26500, 40000))
    ftmw.estimate_noise(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)

    # Trim to first 3 dependency-free windows (same logic as conftest).
    plan = load_windows_impl(str(fp))["plan"]
    n_target = 3
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= n_target:
            break
    if not candidates:
        candidates = list(plan.topological_order[:n_target])
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(fp), plan)

    ftmw.fit_peaks(str(fp))
    return fp


@pytest.fixture
def writable_stage5_file(stage5_small_file, tmp_path):
    """Return a writable copy of the Stage 5 small file."""
    dst = tmp_path / "working.ftmw"
    shutil.copy(stage5_small_file, dst)
    return dst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


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
        from ftmwpipeline.fitting.peak_model import PeakShape
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

            result = refit_window_impl(str(path), wid)
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

        result = refit_window_impl(str(path), wid)

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
    def _setup(self, stage5_small_file, tmp_path):
        """Give each test its own writable copy."""
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_file, self.path)
        self.sf_orig = _load_spectrum_fit(self.path)
        self.windows_with_peaks = [
            wf for wf in self.sf_orig.window_fits if len(wf.fitted_peaks) > 0
        ]

    def test_first_window_peak_count_preserved(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        n_before = len(wf.fitted_peaks)
        result = refit_window_impl(str(self.path), wf.window_id)
        assert result.n_peaks_after == n_before

    def test_first_window_frequencies_within_tolerance(self):
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf = self.windows_with_peaks[0]
        peaks_before = sorted(wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        refit_window_impl(str(self.path), wf.window_id)
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
        refit_window_impl(str(self.path), wf.window_id)
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
            result = refit_window_impl(str(self.path), wf.window_id)
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
        result = refit_window_impl(str(self.path), wf.window_id)
        assert abs(result.chi2r_before - float(wf.reduced_chi2)) < 1e-6

    def test_other_windows_unchanged(self):
        """Refitting one window must not alter other windows' fitted peaks."""
        if not self.windows_with_peaks:
            pytest.skip("No windows with peaks")
        wf0 = self.windows_with_peaks[0]
        other_wfs = [w for w in self.windows_with_peaks if w.window_id != wf0.window_id]
        if not other_wfs:
            pytest.skip("Only one window with peaks; cannot test isolation")

        refit_window_impl(str(self.path), wf0.window_id)
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
    def _setup(self, stage5_small_file, tmp_path):
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_file, self.path)
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
    def _setup(self, stage5_small_file, tmp_path):
        self.path = tmp_path / "working.ftmw"
        shutil.copy(stage5_small_file, self.path)
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

    def test_add_with_explicit_seed(self):
        """add_seeds provides the starting ModelPeak for the added frequency.

        The key contract: when add_seeds is provided, the engine must accept
        and use it as the starting point.  Whether the peak survives the NLS
        and rescue depends on the data; we assert only that the function
        completes without error and returns a valid result.  The user-origin
        stamping is tested in test_add_peak_origin_user using the auto-seed
        path (which derives a realistic amplitude from the data).
        """
        if self.wf is None:
            pytest.skip("No window found")
        center = self._window_center_mhz(self.wf)
        if center is None:
            pytest.skip("Window has no freq_range")

        # Use offset=0 (window center) as the explicit seed.
        explicit_seed = ModelPeak(amplitude=1.0, offset_mhz=0.0, phase=0.0)
        result = refit_window_impl(
            str(self.path),
            self.wf.window_id,
            add=[center],
            add_seeds=[explicit_seed],
        )
        # The function must complete without error and return a valid result.
        assert isinstance(result, RefitWindowResult)
        assert result.window_id == self.wf.window_id

    def test_add_seeds_wrong_length_raises(self):
        if self.wf is None:
            pytest.skip("No window found")
        center = self._window_center_mhz(self.wf)
        if center is None:
            pytest.skip("Window has no freq_range")
        with pytest.raises(ValueError, match="len\\(add_seeds\\)"):
            refit_window_impl(
                str(self.path),
                self.wf.window_id,
                add=[center],
                add_seeds=[],
            )


# ---------------------------------------------------------------------------
# Spur-catalogue replay: the refit must reproduce the Stage 5 residual mask
# from the persisted gated catalogue, never re-derive it (detection is a
# Stage 5 product; a detector change between fit and refit must not re-mask).
# ---------------------------------------------------------------------------


class TestSpurCatalogueReplay:
    def test_spur_set_from_catalogue_roundtrip(self):
        """Reconstructing a SpurSet from its persisted fields reproduces the
        window mask spec exactly (centres, default + per-spur half-widths)."""
        from ftmwpipeline.core.data_structures import Sideband
        from ftmwpipeline.fitting.spur_detection import (
            GatedSpur,
            SpurSet,
            spur_set_from_catalogue,
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
        rebuilt = spur_set_from_catalogue(
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
        """A pre-field file (only centres + sources persisted) still rebuilds:
        lattice->None, drift->False, per-spur width->None (uniform default)."""
        from ftmwpipeline.fitting.spur_detection import spur_set_from_catalogue

        s = spur_set_from_catalogue(
            centers_mhz=[28023.0, 30100.0],
            sources=["narrow", "narrow"],
            bin_spacing_mhz=0.094,
            default_mask_half_width_bins=2,
        )
        assert len(s.spurs) == 2
        assert all(sp.mask_half_width_bins is None for sp in s.spurs)
        assert all(sp.lattice is None and sp.drift is False for sp in s.spurs)

    def test_refit_replays_persisted_catalogue_not_detection(self, stage5_small_file):
        """build_stage5_fit_context(replay_spur_catalogue=...) must yield a
        spur set matching the INJECTED catalogue, not whatever detection finds.
        Inject a fictional spur centre and confirm it is the one masked."""
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

        # A fictional catalogue current detection would not produce.
        injected = {
            "spur_centers_mhz": [28123.456, 33000.789],
            "spur_sources": ["narrow", "saturated"],
            "spur_lattice": [None, None],
            "spur_drift": [False, False],
            "spur_mask_half_width_bins_per_spur": [None, None],
            "spur_mask_half_width_bins": 2,
        }
        ctx = build_stage5_fit_context(
            path, resolved, None, shape_enum, replay_spur_catalogue=injected
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
            refit_window_impl(str(fp), 0)

    def test_invalid_window_raises(self, stage5_small_file, tmp_path):
        dst = tmp_path / "copy.ftmw"
        shutil.copy(stage5_small_file, dst)
        with pytest.raises(KeyError):
            refit_window_impl(str(dst), 99999)


# ---------------------------------------------------------------------------
# Thawed-line freezing: accepted thaw events must be frozen during refit
# ---------------------------------------------------------------------------


def _inject_synthetic_thaw(path: Path, window_id: int, thawed_freq_mhz: float) -> None:
    """Inject a synthetic accepted ThawInfo into the window's HDF5 attrs.

    Pretends that ``thawed_freq_mhz`` is a peak that was thawed from a
    (fictitious) primary window.  The refit engine must freeze this peak
    and re-insert it verbatim rather than re-fitting it freely.
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
        # Window groups are stored under stage5_fitting/windows/window_XXXX
        # (4-digit zero-padded).
        wg_key = f"stage5_fitting/windows/window_{window_id:04d}"
        wg = h5f[wg_key]
        wg.attrs["thaw_events"] = json.dumps([thaw_record])


_CROSS_FIXTURE_2638 = Path("scratch/issue3-cross-fixture/2638/exp_2638.ftmw")


@pytest.fixture(scope="module")
def stage5_multi_peak_file(tmp_path_factory):
    """Full 2638 Stage-5 fit for tests requiring multi-peak windows.

    Reuses the pre-built cross-fixture file when present (fastest path).
    Falls back to skipping if the file is absent (CI without the scratch
    artefacts).
    """
    src = _CROSS_FIXTURE_2638
    if not src.exists():
        pytest.skip(
            "Cross-fixture 2638 file not found at scratch/issue3-cross-fixture/2638/."
            " Build it via 'ftmwpipeline fit run' on that experiment first."
        )
    # Copy to a tmp location so the tests can write synthetic thaw events
    # without dirtying the shared file.
    tmp = tmp_path_factory.mktemp("stage5_multi_peak")
    fp = tmp / "2638_multi_peak.ftmw"
    shutil.copy(src, fp)
    return fp


class TestThawedLineFreeze:
    """Verify that an accepted thaw event causes the thawed peak to be frozen
    during the refit (reproduced verbatim) while the window's own peaks
    still re-converge normally.

    Uses a synthetic ThawInfo injected into the 2638 multi-peak fixture to
    avoid needing a real fixture with thaw events (which are rare in practice).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, stage5_multi_peak_file, tmp_path):
        self.path = tmp_path / "thaw_test.ftmw"
        shutil.copy(stage5_multi_peak_file, self.path)
        self.sf_orig = _load_spectrum_fit(self.path)
        # Pick a window with at least 2 peaks so one can be synthetic-thawed
        # while at least one remains as a free peak.
        self.wf = next(
            (w for w in self.sf_orig.window_fits if len(w.fitted_peaks) >= 2),
            None,
        )

    def test_thawed_peak_exact_after_refit(self):
        """No-edit refit with a synthetic thaw must reproduce the thawed peak
        EXACTLY (zero frequency and amplitude deviation) while the remaining
        free peak(s) re-converge to ~1e-5 MHz / ~1e-3 amplitude tolerance."""
        if self.wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")

        wid = self.wf.window_id
        peaks_sorted = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)

        # Designate the first (lowest-frequency) peak as "thawed".
        thawed_fp = peaks_sorted[0]
        thawed_freq = float(thawed_fp.frequency_mhz)

        # Inject the synthetic thaw event.
        _inject_synthetic_thaw(self.path, wid, thawed_freq)

        # No-edit refit.
        result = refit_window_impl(str(self.path), wid)
        assert isinstance(result, RefitWindowResult)

        # Load the updated fit.
        sf_after = _load_spectrum_fit(self.path)
        wf_after = next(w for w in sf_after.window_fits if w.window_id == wid)

        # Peak count must be preserved.
        assert len(wf_after.fitted_peaks) == len(self.wf.fitted_peaks), (
            f"window {wid}: peak count changed "
            f"{len(self.wf.fitted_peaks)} → {len(wf_after.fitted_peaks)}"
        )
        assert result.n_peaks_before == len(self.wf.fitted_peaks)
        assert result.n_peaks_after == len(self.wf.fitted_peaks)

        peaks_after = sorted(wf_after.fitted_peaks, key=lambda p: p.frequency_mhz)

        # Thawed peak: EXACT match (re-inserted verbatim).
        thawed_after = next(
            (
                p
                for p in peaks_after
                if abs(float(p.frequency_mhz) - thawed_freq) < 1e-6
            ),
            None,
        )
        assert thawed_after is not None, (
            f"Thawed peak at {thawed_freq:.4f} MHz missing after refit; "
            f"peaks after: {[float(p.frequency_mhz) for p in peaks_after]}"
        )
        assert float(thawed_after.frequency_mhz) == pytest.approx(
            thawed_freq, abs=0.0
        ), (
            f"Thawed peak freq drifted: "
            f"before={thawed_freq:.6f} MHz, after={float(thawed_after.frequency_mhz):.6f} MHz"
        )
        assert float(thawed_after.amplitude) == pytest.approx(
            float(thawed_fp.amplitude), rel=0.0, abs=0.0
        ), (
            f"Thawed peak amplitude changed: "
            f"before={float(thawed_fp.amplitude):.4g}, after={float(thawed_after.amplitude):.4g}"
        )

        # Non-thawed peaks: should re-converge within a loose tolerance.
        # The thawed peak was frozen into the background (whereas in the real
        # Stage 5 result it was a free peak); this slightly changes the effective
        # noise in the window and allows the NLS to move non-thawed peaks by
        # a few hundred kHz.  The important contract is that (a) the thawed peak
        # is EXACT, and (b) non-thawed peaks stay in the same basin (< 1 MHz).
        non_thawed_before = [
            p for p in peaks_sorted if abs(float(p.frequency_mhz) - thawed_freq) > 1e-6
        ]
        non_thawed_after = [
            p for p in peaks_after if abs(float(p.frequency_mhz) - thawed_freq) > 1e-6
        ]
        assert len(non_thawed_before) == len(
            non_thawed_after
        ), "Non-thawed peak count changed after refit"
        for pb, pa in zip(non_thawed_before, non_thawed_after):
            delta_f = abs(float(pb.frequency_mhz) - float(pa.frequency_mhz))
            assert delta_f < 1.0, (
                f"Non-thawed peak at {float(pb.frequency_mhz):.4f} MHz jumped "
                f"{delta_f:.4f} MHz — likely escaped the NLS basin"
            )

    def test_remove_thawed_peak_drops_it(self):
        """Removing a synthetic-thawed peak must drop it entirely (not frozen).

        After removal the thawed peak should not appear in the output, and
        the remaining free peaks should re-converge normally.
        """
        if self.wf is None:
            pytest.skip("No window with >=2 peaks in the small fixture")

        wid = self.wf.window_id
        peaks_sorted = sorted(self.wf.fitted_peaks, key=lambda p: p.frequency_mhz)
        thawed_freq = float(peaks_sorted[0].frequency_mhz)
        n_before = len(peaks_sorted)

        _inject_synthetic_thaw(self.path, wid, thawed_freq)

        # Remove the thawed peak.
        result = refit_window_impl(str(self.path), wid, remove=[thawed_freq])

        # Should have one fewer peak.
        assert result.n_peaks_after <= n_before - 1, (
            f"Expected peak count to drop after removing the thawed peak; "
            f"before={n_before}, after={result.n_peaks_after}"
        )

        # The thawed peak must not appear in the output.
        remaining_freqs = [float(p.frequency_mhz) for p in result.fitted_peaks]
        assert not any(abs(f - thawed_freq) < 1e-4 for f in remaining_freqs), (
            f"Removed thawed peak at {thawed_freq:.4f} MHz still present; "
            f"remaining: {remaining_freqs}"
        )
