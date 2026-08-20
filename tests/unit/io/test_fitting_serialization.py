"""
Unit tests for Stage 5 fitting-result HDF5 serialization.

Covers:

* round-trip contract (save -> load returns an equivalent
  :class:`SpectrumFit`);
* hand-edit (save -> edit a fitted peak's frequency -> load returns the
  edited value);
* loud validation of malformed groups (missing required attrs/datasets,
  mismatched peak-column lengths, invalid audit-decision and edge-side
  labels in the JSON history);
* NaN-encoded ``None`` uncertainties round-trip;
* recomputable arrays (``fitted_spectrum``, ``window``) come back as
  ``None`` -- the SERIALIZATION spec's lightweight-file invariant
  forbids persisting them;
* the ``stage5_fitting`` stage key is registered with the right
  dependencies (``stage0_fid_data`` AND ``stage4_windows``).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    AuditStep,
    DoubletAlternativeInfo,
    FittedPeak,
    FittingResult,
    KnockoutInfo,
    ReplanInfo,
    RescueCandidateInfo,
    RescueRoundInfo,
    SpectralWindow,
    SpectrumFit,
    ThawInfo,
)
from ftmwpipeline.file_manager import PipelineStageTracker
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    save_spectrum_fit_to_hdf5,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _sample_audit() -> list[AuditStep]:
    return [
        AuditStep(
            n_peaks_before=0,
            candidate_offset_mhz=-0.12,
            chi2_before=1234.5,
            chi2_after=200.1,
            f_statistic=80.0,
            p_value=1e-30,
            aic_before=2050.0,
            aic_after=1900.0,
            separation_ok=True,
            decision="seed",
            reason="K=1 seed fit",
        ),
        AuditStep(
            n_peaks_before=1,
            candidate_offset_mhz=0.34,
            chi2_before=200.1,
            chi2_after=180.0,
            f_statistic=12.0,
            p_value=0.002,
            aic_before=1900.0,
            aic_after=1880.0,
            separation_ok=True,
            decision="accept",
            reason="+1 line(s)",
        ),
        AuditStep(
            n_peaks_before=2,
            candidate_offset_mhz=0.5,
            chi2_before=180.0,
            chi2_after=float("nan"),
            f_statistic=float("nan"),
            p_value=float("nan"),
            aic_before=1880.0,
            aic_after=float("nan"),
            separation_ok=True,
            decision="spur-drop",
            reason="converged peak on a gated spur, no primary backing",
        ),
    ]


def _sample_window_thaw() -> list[ThawInfo]:
    return [
        ThawInfo(
            dependent_window_id=1,
            primary_window_id=0,
            contributor_peak_index=0,
            contributor_frequency_mhz=36100.0,
            edge_side="low",
            edge_coherence_before=4.5,
            edge_coherence_after=1.2,
            accepted=True,
            reason="co-fit cleared the flagged edge",
        ),
    ]


def _sample_fitted_peak(
    *, peak_id: int, window_id: int, freq_mhz: float, amplitude: float = 0.5
) -> FittedPeak:
    return FittedPeak(
        peak_id=peak_id,
        frequency_mhz=freq_mhz,
        amplitude=amplitude,
        decay_rate=0.2,
        phase=0.3,
        frequency_error=1e-4,
        amplitude_error=0.01,
        decay_rate_error=2e-3,
        phase_error=0.02,
        snr=120.0,
        chi_squared=15.4,
        window_id=window_id,
        knockout=KnockoutInfo(
            delta_chi2=80.0, expected_delta_chi2=75.0, supported=True
        ),
    )


def _make_window_fit(
    window_id: int,
    fitted_peaks: list[FittedPeak],
    *,
    audit: list[AuditStep],
    thaw_events: list[ThawInfo],
    tau_us: float = 5.0,
    tau_error: float | None = 0.05,
) -> FittingResult:
    fr = FittingResult(
        success=True,
        fitted_spectrum=np.ones(8, dtype=np.complex128) * (0.1 + 0.2j),
        cost=12.34,
        iterations=len(audit),
        aic=1880.0,
        reduced_chi2=1.02,
        window=None,
        window_id=window_id,
    )
    fr.fitted_peaks = list(fitted_peaks)
    fr.shared_parameters["tau_us"] = {
        "value": tau_us,
        "error": tau_error,
        "peak_ids": [p.peak_id for p in fitted_peaks],
    }
    if window_id == 1:
        fr.fixed_parameters["frozen_peak_0"] = {
            "peak_index": 0,
            "primary_window_id": 0,
            "frequency_mhz": 36100.0,
            "amplitude": 1.5,
            "phase": 0.1,
            "freeze_eligible": True,
        }
    fr.residuals = np.zeros(8, dtype=np.complex128)
    fr.quality_metrics = {
        "edge_coherence_low": 0.8,
        "edge_coherence_high": 0.9,
        "n_fixed_contributors": float(len(fr.fixed_parameters)),
    }
    fr.audit_trail = list(audit)
    fr.thaw_events = list(thaw_events)
    return fr


def _sample_spectrum_fit() -> SpectrumFit:
    win0_peaks = [_sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.012)]
    win1_peaks = [_sample_fitted_peak(peak_id=1, window_id=1, freq_mhz=36110.045)]
    win0 = _make_window_fit(0, win0_peaks, audit=_sample_audit(), thaw_events=[])
    win1 = _make_window_fit(
        1, win1_peaks, audit=_sample_audit()[:1], thaw_events=_sample_window_thaw()
    )
    return SpectrumFit(
        window_fits=[win0, win1],
        fitted_peaks=sorted(win0_peaks + win1_peaks, key=lambda p: p.frequency_mhz),
        thaw_history=_sample_window_thaw(),
        replan_history=[
            ReplanInfo(
                triggering_window_id=1,
                partner_window_id=2,
                surviving_window_id=1,
                edge_side="high",
                edge_coherence_before=5.7,
                revision_before=0,
                revision_after=1,
                accepted=True,
                reason="boundary cut a real feature",
            )
        ],
        final_plan_revision=1,
        parameters={"tau0_us": 5.0, "residual_edge_threshold": 1.5},
        diagnostics={"note": "synthetic"},
    )


def _roundtrip(fit: SpectrumFit, path) -> SpectrumFit:
    with h5py.File(path, "w") as h5f:
        g = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(fit, g)
    with h5py.File(path, "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_basic_round_trip(self, tmp_path):
        fit = _sample_spectrum_fit()
        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        # Top-level attrs.
        assert loaded.n_windows == fit.n_windows
        assert loaded.n_fitted_peaks == fit.n_fitted_peaks
        assert loaded.final_plan_revision == fit.final_plan_revision
        assert loaded.parameters == fit.parameters
        assert loaded.diagnostics == fit.diagnostics

        # Plan-level histories.
        assert len(loaded.thaw_history) == len(fit.thaw_history)
        for got, want in zip(loaded.thaw_history, fit.thaw_history):
            assert got.dependent_window_id == want.dependent_window_id
            assert got.edge_side == want.edge_side
            assert got.accepted == want.accepted
        assert len(loaded.replan_history) == len(fit.replan_history)
        assert loaded.replan_history[0].surviving_window_id == 1

        # Per-window fits in window_id order.
        assert [w.window_id for w in loaded.window_fits] == [0, 1]
        for got, want in zip(loaded.window_fits, fit.window_fits):
            assert got.success == want.success
            assert got.cost == pytest.approx(want.cost)
            assert got.aic == pytest.approx(want.aic)
            assert got.reduced_chi2 == pytest.approx(want.reduced_chi2)
            assert got.iterations == want.iterations
            # Per-window tau survives via shared_parameters.
            assert got.shared_parameters["tau_us"]["value"] == pytest.approx(
                want.shared_parameters["tau_us"]["value"]
            )
            assert got.shared_parameters["tau_us"]["error"] == pytest.approx(
                want.shared_parameters["tau_us"]["error"]
            )
            # Audit trail.
            assert len(got.audit_trail) == len(want.audit_trail)
            for ga, wa in zip(got.audit_trail, want.audit_trail):
                assert ga.decision == wa.decision
                assert ga.n_peaks_before == wa.n_peaks_before
            # Quality metrics.
            assert got.quality_metrics["edge_coherence_low"] == pytest.approx(
                want.quality_metrics["edge_coherence_low"]
            )

        # Merged peak list is rebuilt and sorted by frequency.
        freqs = [p.frequency_mhz for p in loaded.fitted_peaks]
        assert freqs == sorted(freqs)
        assert len(loaded.fitted_peaks) == 2

    def test_per_peak_fields_round_trip(self, tmp_path):
        fit = _sample_spectrum_fit()
        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        for got, want in zip(loaded.fitted_peaks, fit.fitted_peaks):
            assert got.peak_id == want.peak_id
            assert got.window_id == want.window_id
            assert got.frequency_mhz == pytest.approx(want.frequency_mhz)
            assert got.amplitude == pytest.approx(want.amplitude)
            assert got.phase == pytest.approx(want.phase)
            assert got.decay_rate == pytest.approx(want.decay_rate)
            assert got.frequency_error == pytest.approx(want.frequency_error)
            assert got.amplitude_error == pytest.approx(want.amplitude_error)
            assert got.snr == pytest.approx(want.snr)
            assert isinstance(got.knockout, KnockoutInfo)
            assert got.knockout.supported == want.knockout.supported
            assert got.knockout.delta_chi2 == pytest.approx(want.knockout.delta_chi2)

    def test_none_uncertainties_round_trip_as_none(self, tmp_path):
        """``None`` optional fields are NaN-encoded and round-trip as None."""
        peak = FittedPeak(
            peak_id=3,
            frequency_mhz=36120.0,
            amplitude=0.7,
            decay_rate=None,
            phase=None,
            frequency_error=None,
            amplitude_error=None,
            phase_error=None,
            decay_rate_error=None,
            snr=None,
            chi_squared=None,
            window_id=2,
            knockout=None,
        )
        win = _make_window_fit(2, [peak], audit=[], thaw_events=[], tau_error=None)
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        got = loaded.fitted_peaks[0]
        assert got.peak_id == 3
        assert got.decay_rate is None
        assert got.phase is None
        assert got.frequency_error is None
        assert got.amplitude_error is None
        assert got.phase_error is None
        assert got.snr is None
        assert got.chi_squared is None
        assert got.knockout is None

        # tau_error None survives through the shared_parameters dict.
        tau_entry = loaded.window_fits[0].shared_parameters["tau_us"]
        assert tau_entry["error"] is None

    def test_recomputable_arrays_not_persisted(self, tmp_path):
        """fitted_spectrum is recomputable -- comes back as None. The window's
        freq_range is persisted (lightweight) so a loaded SpectralWindow has
        no spectrum arrays but the right freq_range."""
        fit = _sample_spectrum_fit()
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        for window_fit in loaded.window_fits:
            assert window_fit.fitted_spectrum is None
            # The fixture _make_window_fit sets window=None, so freq_range
            # round-trips as NaN -> loaded window stays None.
            assert window_fit.window is None

    def test_window_freq_range_round_trips_when_attached(self, tmp_path):
        """A FittingResult with an attached SpectralWindow round-trips the
        freq_range; the loaded window's arrays are empty (recomputable) but
        freq_range is preserved so visualization can locate the window."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        win.window = SpectralWindow(
            parent_ft=None,
            freq_array=np.linspace(36099.4, 36100.6, 50),
            complex_spectrum=np.zeros(50, dtype=np.complex128),
            freq_range=(36099.4, 36100.6),
            window_id=0,
        )
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        loaded_win = loaded.window_fits[0].window
        assert isinstance(loaded_win, SpectralWindow)
        assert loaded_win.freq_range == pytest.approx((36099.4, 36100.6))
        # Arrays are not persisted -- empty on load (caller recomputes).
        assert loaded_win.freq_array.size == 0
        assert loaded_win.complex_spectrum.size == 0

    def test_overwrite_existing_group(self, tmp_path):
        """A second save into the same group replaces, not appends."""
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Smaller fit overwrites.
            smaller = SpectrumFit(
                window_fits=[fit.window_fits[0]],
                fitted_peaks=fit.window_fits[0].fitted_peaks,
            )
            save_spectrum_fit_to_hdf5(smaller, g)
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.n_windows == 1
        assert [w.window_id for w in loaded.window_fits] == [0]


# ---------------------------------------------------------------------------
# Hand-edit
# ---------------------------------------------------------------------------
class TestHandEdit:
    def test_edit_peak_frequency_survives_round_trip(self, tmp_path):
        """A curator nudges a peak frequency in HDF5; the load picks it up."""
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
        # Hand-edit window 0's only peak.
        with h5py.File(path, "a") as h5f:
            arr = h5f["stage5_fitting/windows/window_0000/peaks/frequency_mhz"]
            arr[0] = 36100.250  # nudge +238 kHz
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        win0 = loaded.window_fit(0)
        assert win0.fitted_peaks[0].frequency_mhz == pytest.approx(36100.250)
        # Merged global peak list rebuilt -> still sorted, includes the edit.
        sorted_freqs = [p.frequency_mhz for p in loaded.fitted_peaks]
        assert sorted_freqs == sorted(sorted_freqs)
        assert 36100.250 in sorted_freqs

    def test_drop_a_peak_via_resize(self, tmp_path):
        """Re-saving with one peak fewer is a valid hand-edit path: write
        new columns, then load returns the truncated list. (Direct in-place
        h5py resize on an existing dataset is harder; we just re-save.)"""
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
        # Curator drops window 1's only peak.
        fit.window_fits[1].fitted_peaks = []
        with h5py.File(path, "a") as h5f:
            save_spectrum_fit_to_hdf5(fit, h5f["stage5_fitting"])
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.window_fit(1).fitted_peaks == []
        # Merged list now has only window 0's peak.
        assert len(loaded.fitted_peaks) == 1
        assert loaded.fitted_peaks[0].window_id == 0


# ---------------------------------------------------------------------------
# Loud validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_missing_windows_subgroup_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        with h5py.File(path, "w") as h5f:
            h5f.create_group("stage5_fitting")  # no /windows
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="missing required 'windows'"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_missing_window_attr_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            del g["windows/window_0000"].attrs["aic"]
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="missing required attribute 'aic'"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_missing_peak_column_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            del g["windows/window_0000/peaks/snr"]
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="missing required peak column"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_mismatched_peak_column_lengths_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            peaks_group = g["windows/window_0000/peaks"]
            # Replace one column with a wrong-length array.
            del peaks_group["snr"]
            peaks_group.create_dataset("snr", data=np.zeros(3))
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="mismatched lengths"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_unknown_audit_decision_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            bad = [
                {
                    "n_peaks_before": 0,
                    "candidate_offset_mhz": 0.0,
                    "chi2_before": 0.0,
                    "chi2_after": 0.0,
                    "f_statistic": 0.0,
                    "p_value": 1.0,
                    "aic_before": 0.0,
                    "aic_after": 0.0,
                    "separation_ok": True,
                    "decision": "frobnicate",
                    "reason": "",
                }
            ]
            g["windows/window_0000"].attrs["audit_trail"] = json.dumps(bad)
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="unknown decision"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_invalid_thaw_edge_side_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            bad = [
                {
                    "dependent_window_id": 1,
                    "primary_window_id": 0,
                    "contributor_peak_index": 0,
                    "contributor_frequency_mhz": 0.0,
                    "edge_side": "left",  # invalid
                    "edge_coherence_before": 0.0,
                    "edge_coherence_after": 0.0,
                    "accepted": False,
                    "reason": "",
                }
            ]
            g.attrs["thaw_history"] = json.dumps(bad)
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="invalid edge_side"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_malformed_json_attribute_raises(self, tmp_path):
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            g.attrs["parameters"] = "not-json-{["
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="not valid JSON"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_missing_window_id_on_fit_raises(self, tmp_path):
        """A FittingResult without window_id cannot be saved."""
        win = _make_window_fit(0, [], audit=[], thaw_events=[])
        win.window_id = None  # corrupt
        fit = SpectrumFit(window_fits=[win])
        with h5py.File(tmp_path / "fit.h5", "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            with pytest.raises(ValueError, match="window_id is required"):
                save_spectrum_fit_to_hdf5(fit, g)


# ---------------------------------------------------------------------------
# Stage tracker registration
# ---------------------------------------------------------------------------
class TestStageRegistration:
    def test_stage5_fitting_dependencies(self):
        deps = PipelineStageTracker.STAGE_DEPENDENCIES
        assert "stage5_fitting" in deps
        assert set(deps["stage5_fitting"]) == {"stage0_fid_data", "stage4_windows"}

    def test_stage5_fitting_data_path(self):
        paths = PipelineStageTracker.STAGE_DATA_PATHS
        assert paths.get("stage5_fitting") == "stage5_fitting"

    def test_stage5_fitting_only_available_after_deps(self):
        tracker = PipelineStageTracker(completed_stages=["stage0_fid_data"])
        # Missing stage4_windows -> stage5_fitting not available yet.
        assert "stage5_fitting" not in tracker.get_available_stages()
        tracker.mark_completed("stage1_complex_ft")
        tracker.mark_completed("stage2_noise_result")
        tracker.mark_completed("stage3_peaks")
        tracker.mark_completed("stage4_windows")
        assert "stage5_fitting" in tracker.get_available_stages()

    def test_stage6_review_dependencies(self):
        deps = PipelineStageTracker.STAGE_DEPENDENCIES
        assert "stage6_review" in deps
        assert deps["stage6_review"] == ["stage5_fitting"]

    def test_stage6_review_data_path(self):
        paths = PipelineStageTracker.STAGE_DATA_PATHS
        assert paths.get("stage6_review") == "stage6_review"

    def test_stage6_review_only_available_after_stage5(self):
        tracker = PipelineStageTracker(
            completed_stages=["stage0_fid_data", "stage4_windows"]
        )
        assert "stage6_review" not in tracker.get_available_stages()
        tracker.mark_completed("stage5_fitting")
        assert "stage6_review" in tracker.get_available_stages()


# ---------------------------------------------------------------------------
# Rescue-rounds persistence (Task C)
# ---------------------------------------------------------------------------
def _sample_rescue_round(
    *,
    window_id: int,
    round_idx: int,
    n_initial_peaks: int = 2,
    n_rescue_added: int = 1,
    n_pruned_total: int = 0,
    n_pruned_rescue_origin: int = 0,
    n_merged: int = 0,
    chi2_before: float = 12.0,
    chi2_after: float = 8.0,
    tau_us_before: float = 3.0,
    tau_us_after: float = 3.05,
    accepted: bool = True,
    reason: str = "joint refit consolidated rescue contribution",
    candidates: list[RescueCandidateInfo] | None = None,
) -> RescueRoundInfo:
    return RescueRoundInfo(
        window_id=window_id,
        round_idx=round_idx,
        n_initial_peaks=n_initial_peaks,
        n_rescue_added=n_rescue_added,
        n_pruned_total=n_pruned_total,
        n_pruned_rescue_origin=n_pruned_rescue_origin,
        n_merged=n_merged,
        chi2_before=chi2_before,
        chi2_after=chi2_after,
        tau_us_before=tau_us_before,
        tau_us_after=tau_us_after,
        accepted=accepted,
        reason=reason,
        candidates=candidates or [],
    )


class TestRescueRoundsRoundTrip:
    def test_empty_rescue_history_round_trips(self, tmp_path):
        """A fit with no rescue activity round-trips with empty lists."""
        fit = _sample_spectrum_fit()
        assert fit.rescue_history == []
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        assert loaded.rescue_history == []
        for wf in loaded.window_fits:
            assert wf.rescue_events == []

    def test_single_round_round_trip(self, tmp_path):
        """A window with one rescue round (with candidates) round-trips."""
        fit = _sample_spectrum_fit()
        candidates = [
            RescueCandidateInfo(frequency_mhz=-0.18, magnitude=0.42, snr=3.6),
            RescueCandidateInfo(frequency_mhz=0.27, magnitude=0.31, snr=2.7),
        ]
        round0 = _sample_rescue_round(
            window_id=0,
            round_idx=0,
            n_rescue_added=2,
            candidates=candidates,
        )
        fit.window_fits[0].rescue_events = [round0]
        fit.rescue_history = [round0]

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert len(loaded.rescue_history) == 1
        got = loaded.rescue_history[0]
        assert got.window_id == 0
        assert got.round_idx == 0
        assert got.n_initial_peaks == 2
        assert got.n_rescue_added == 2
        assert got.accepted is True
        assert got.reason == round0.reason
        assert got.chi2_before == pytest.approx(12.0)
        assert got.chi2_after == pytest.approx(8.0)
        assert got.tau_us_after == pytest.approx(3.05)
        assert [c.frequency_mhz for c in got.candidates] == pytest.approx([-0.18, 0.27])
        assert got.candidates[0].snr == pytest.approx(3.6)

        # Per-window mirror.
        assert len(loaded.window_fits[0].rescue_events) == 1
        assert loaded.window_fits[0].rescue_events[0].window_id == 0

    def test_multi_round_with_failsafe_round_trip(self, tmp_path):
        """A multi-round chain with the rescue-origin pruning failsafe set."""
        fit = _sample_spectrum_fit()
        round0 = _sample_rescue_round(
            window_id=1,
            round_idx=0,
            n_initial_peaks=1,
            n_rescue_added=2,
            n_pruned_total=0,
            n_pruned_rescue_origin=0,
            n_merged=0,
            chi2_before=20.0,
            chi2_after=14.0,
            candidates=[
                RescueCandidateInfo(frequency_mhz=-0.5, magnitude=0.8, snr=5.1),
                RescueCandidateInfo(frequency_mhz=0.6, magnitude=0.7, snr=4.4),
            ],
        )
        round1 = _sample_rescue_round(
            window_id=1,
            round_idx=1,
            n_initial_peaks=3,
            n_rescue_added=1,
            n_pruned_total=1,
            n_pruned_rescue_origin=1,  # the failsafe firing
            n_merged=0,
            chi2_before=14.0,
            chi2_after=14.0,
            accepted=False,
            reason="joint refit consolidated rescue contribution (rescue-origin prune)",
            candidates=[
                RescueCandidateInfo(frequency_mhz=0.12, magnitude=0.35, snr=2.9),
            ],
        )
        fit.window_fits[1].rescue_events = [round0, round1]
        fit.rescue_history = [round0, round1]

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert [r.round_idx for r in loaded.rescue_history] == [0, 1]
        failsafe = loaded.rescue_history[1]
        assert failsafe.n_pruned_rescue_origin == 1
        assert failsafe.accepted is False
        assert failsafe.window_id == 1
        # Candidates and counts on the first round survive.
        first = loaded.rescue_history[0]
        assert first.n_rescue_added == 2
        assert len(first.candidates) == 2
        assert first.candidates[1].snr == pytest.approx(4.4)
        # Per-window mirror has both rounds for window 1.
        win1 = loaded.window_fits[1]
        assert [r.round_idx for r in win1.rescue_events] == [0, 1]

    def test_legacy_file_loads_with_empty_rescue_history(self, tmp_path):
        """A file written without the rescue_history / rescue_events attrs
        loads with empty lists, not an error."""
        path = tmp_path / "fit.h5"
        fit = _sample_spectrum_fit()
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a pre-rescue-persistence file: strip the new attrs.
            del g.attrs["rescue_history"]
            for name in g["windows"]:
                wg = g[f"windows/{name}"]
                if "rescue_events" in wg.attrs:
                    del wg.attrs["rescue_events"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.rescue_history == []
        for wf in loaded.window_fits:
            assert wf.rescue_events == []


# ---------------------------------------------------------------------------
# tau_fitted flag (disambiguates frozen-by-gate vs singular-covariance)
# ---------------------------------------------------------------------------
class TestTauFittedRoundTrip:
    def _make_pair(
        self,
        *,
        fitted_0: bool | None,
        fitted_1: bool | None,
        tau_error_0: float | None,
        tau_error_1: float | None,
    ) -> SpectrumFit:
        peak0 = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.012)
        peak1 = _sample_fitted_peak(peak_id=1, window_id=1, freq_mhz=36110.045)
        win0 = _make_window_fit(
            0, [peak0], audit=[], thaw_events=[], tau_error=tau_error_0
        )
        win1 = _make_window_fit(
            1, [peak1], audit=[], thaw_events=[], tau_error=tau_error_1
        )
        win0.shared_parameters["tau_us"]["fitted"] = fitted_0
        win1.shared_parameters["tau_us"]["fitted"] = fitted_1
        return SpectrumFit(
            window_fits=[win0, win1],
            fitted_peaks=[peak0, peak1],
        )

    def test_true_and_false_round_trip(self, tmp_path):
        """``fitted=True`` and ``fitted=False`` survive save/load unchanged."""
        fit = self._make_pair(
            fitted_0=True,
            fitted_1=False,
            tau_error_0=0.05,
            tau_error_1=None,
        )
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        assert loaded.window_fits[0].shared_parameters["tau_us"]["fitted"] is True
        assert loaded.window_fits[1].shared_parameters["tau_us"]["fitted"] is False

    def test_singular_cov_round_trips_as_fitted_true(self, tmp_path):
        """``fitted=True`` with ``error=None`` (singular cov at tau slot) is
        preserved; the loader does not collapse it to fitted=False."""
        fit = self._make_pair(
            fitted_0=True,
            fitted_1=True,
            tau_error_0=None,
            tau_error_1=0.04,
        )
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        entry0 = loaded.window_fits[0].shared_parameters["tau_us"]
        assert entry0["fitted"] is True
        assert entry0["error"] is None
        entry1 = loaded.window_fits[1].shared_parameters["tau_us"]
        assert entry1["fitted"] is True
        assert entry1["error"] == pytest.approx(0.04)

    def test_pre_flag_file_uses_backward_compat(self, tmp_path):
        """A file written before the tau_fitted attr existed loads with
        ``fitted=True`` when tau_error is finite, ``None`` (unknown) when
        tau_error is NaN. Disambiguation isn't possible from those files;
        ``None`` signals that to the consumer.
        """
        path = tmp_path / "fit.h5"
        fit = self._make_pair(
            fitted_0=True,
            fitted_1=False,
            tau_error_0=0.05,
            tau_error_1=None,
        )
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a pre-flag file: strip tau_fitted from each window.
            for name in g["windows"]:
                wg = g[f"windows/{name}"]
                if "tau_fitted" in wg.attrs:
                    del wg.attrs["tau_fitted"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        # Window 0 had finite tau_error -> backward-compat infers fitted=True.
        assert loaded.window_fits[0].shared_parameters["tau_us"]["fitted"] is True
        # Window 1 had tau_error=None -> backward-compat cannot tell; None.
        assert loaded.window_fits[1].shared_parameters["tau_us"]["fitted"] is None

    def test_unknown_sentinel_round_trips_as_backward_compat(self, tmp_path):
        """A window with ``fitted=None`` (e.g. loaded from a pre-flag file
        and re-saved) writes the -1 sentinel and reloads via the same
        backward-compat rule. Round-trip is idempotent on re-save.
        """
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[], tau_error=None)
        win.shared_parameters["tau_us"]["fitted"] = None
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        loaded = _roundtrip(fit, tmp_path / "fit.h5")
        entry = loaded.window_fits[0].shared_parameters["tau_us"]
        # tau_error None + fitted None -> backward-compat resolves to None.
        assert entry["fitted"] is None


# ---------------------------------------------------------------------------
# Clock-lattice annotation persistence
# ---------------------------------------------------------------------------
class TestClockLatticeRoundTrip:
    """The ``clock_lattice`` string persists and rehydrates correctly."""

    def test_annotated_peak_round_trips(self, tmp_path):
        """A peak with ``clock_lattice`` set survives save -> load unchanged."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        peak.clock_lattice = "320x6 (bb)"
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        got = loaded.fitted_peaks[0]
        assert got.clock_lattice == "320x6 (bb)"
        # Also survives in the per-window peak list.
        assert loaded.window_fits[0].fitted_peaks[0].clock_lattice == "320x6 (bb)"

    def test_unannotated_peak_round_trips_as_none(self, tmp_path):
        """A peak without a lattice annotation loads with ``clock_lattice=None``."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        assert peak.clock_lattice is None
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert loaded.fitted_peaks[0].clock_lattice is None

    def test_mixed_annotation_in_one_window(self, tmp_path):
        """One annotated + one unannotated peak in the same window round-trips."""
        pk_a = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        pk_b = _sample_fitted_peak(peak_id=1, window_id=0, freq_mhz=36105.0)
        pk_a.clock_lattice = "6250x3 (bb, drift)"
        # pk_b.clock_lattice stays None
        win = _make_window_fit(0, [pk_a, pk_b], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[pk_a, pk_b])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        by_id = {p.peak_id: p for p in loaded.fitted_peaks}
        assert by_id[0].clock_lattice == "6250x3 (bb, drift)"
        assert by_id[1].clock_lattice is None

    def test_legacy_file_without_column_loads_none(self, tmp_path):
        """A file written before the ``clock_lattice`` column existed loads
        with ``None`` on every peak -- no error, backward-compatible."""
        path = tmp_path / "fit.h5"
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a file that pre-dates the column.
            del g["windows/window_0000/peaks/clock_lattice"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.fitted_peaks[0].clock_lattice is None


# ---------------------------------------------------------------------------
# Origin provenance persistence
# ---------------------------------------------------------------------------
class TestOriginRoundTrip:
    """The ``origin`` provenance field persists and rehydrates correctly."""

    def test_default_origin_is_auto(self):
        """A freshly constructed FittedPeak carries origin='auto'."""
        p = FittedPeak(peak_id=0, frequency_mhz=1.0, amplitude=1.0)
        assert p.origin == "auto"

    def test_auto_origin_round_trips(self, tmp_path):
        """A peak with the default ``origin='auto'`` survives save -> load."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        assert peak.origin == "auto"
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert loaded.fitted_peaks[0].origin == "auto"
        assert loaded.window_fits[0].fitted_peaks[0].origin == "auto"

    def test_user_origin_round_trips(self, tmp_path):
        """A peak with ``origin='user'`` survives save -> load unchanged."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        peak.origin = "user"
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert loaded.fitted_peaks[0].origin == "user"

    def test_mixed_origins_in_one_window(self, tmp_path):
        """One 'auto' + one 'user' peak in the same window both round-trip."""
        pk_a = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        pk_b = _sample_fitted_peak(peak_id=1, window_id=0, freq_mhz=36105.0)
        pk_b.origin = "user"
        win = _make_window_fit(0, [pk_a, pk_b], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[pk_a, pk_b])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        by_id = {p.peak_id: p for p in loaded.fitted_peaks}
        assert by_id[0].origin == "auto"
        assert by_id[1].origin == "user"

    def test_legacy_file_without_column_loads_as_auto(self, tmp_path):
        """A file written before the ``origin`` column existed loads with
        ``'auto'`` on every peak -- no error, backward-compatible."""
        path = tmp_path / "fit.h5"
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a file that pre-dates the column.
            del g["windows/window_0000/peaks/origin"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.fitted_peaks[0].origin == "auto"


# ---------------------------------------------------------------------------
# Peak identity (peak_uid) persistence
# ---------------------------------------------------------------------------
class TestPeakUidRoundTrip:
    """The ``peak_uid`` point-space identity persists and rehydrates
    correctly. Never re-derived on either side of the boundary -- these
    tests only check that the stamped value survives, not what it is."""

    def test_stamped_peak_round_trips(self, tmp_path):
        """A peak with ``peak_uid`` set survives save -> load unchanged."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        peak.peak_uid = 361_000_00
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        got = loaded.fitted_peaks[0]
        assert got.peak_uid == 361_000_00
        # Also survives in the per-window peak list.
        assert loaded.window_fits[0].fitted_peaks[0].peak_uid == 361_000_00

    def test_unstamped_peak_round_trips_as_none(self, tmp_path):
        """A peak with no identifier loads with ``peak_uid=None``."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        assert peak.peak_uid is None
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert loaded.fitted_peaks[0].peak_uid is None

    def test_mixed_identity_in_one_window(self, tmp_path):
        """One stamped + one unstamped peak in the same window both
        round-trip independently."""
        pk_a = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        pk_b = _sample_fitted_peak(peak_id=1, window_id=0, freq_mhz=36105.0)
        pk_a.peak_uid = 42
        # pk_b.peak_uid stays None
        win = _make_window_fit(0, [pk_a, pk_b], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[pk_a, pk_b])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        by_id = {p.peak_id: p for p in loaded.fitted_peaks}
        assert by_id[0].peak_uid == 42
        assert by_id[1].peak_uid is None

    def test_legacy_file_without_column_loads_none(self, tmp_path):
        """A file written before the ``peak_uid`` column existed loads with
        ``None`` on every peak -- no error, backward-compatible, and no
        identifier is backfilled from the fitted frequency."""
        path = tmp_path / "fit.h5"
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a file that pre-dates the column.
            del g["windows/window_0000/peaks/peak_uid"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.fitted_peaks[0].peak_uid is None


# ---------------------------------------------------------------------------
# DoubletAlternativeInfo serialization
# ---------------------------------------------------------------------------
def _make_doublet_alt(
    *,
    freq_a: float = 36100.1,
    freq_b: float = 36100.4,
    merged_success: bool = True,
) -> DoubletAlternativeInfo:
    return DoubletAlternativeInfo(
        frequency_a_mhz=freq_a,
        frequency_b_mhz=freq_b,
        amplitude_a=0.8,
        amplitude_b=0.5,
        separation_res_elements=0.9,
        amp_ratio=0.625,
        chi2r_production=1.4,
        chi2r_merged=float("nan") if not merged_success else 0.9,
        delta_chi2_raw=50.0 if merged_success else float("nan"),
        delta_aicc=-2.5,
        merged_frequency_mhz=freq_a + 0.15 if merged_success else float("nan"),
        merged_amplitude=1.3 if merged_success else float("nan"),
        merged_phase=0.1 if merged_success else float("nan"),
        merged_tau_us=4.5 if merged_success else float("nan"),
        merged_success=merged_success,
        orth_evidence_delta_chi2=12.0,
        orth_evidence_n_params=3,
        support_bins=8,
    )


class TestDoubletAlternativeRoundTrip:
    def test_round_trip_with_successful_merge(self, tmp_path):
        """DoubletAlternativeInfo with a successful merge survives HDF5 round-trip."""
        alt = _make_doublet_alt(merged_success=True)
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        win.doublet_alternatives = [alt]
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        alts = loaded.window_fits[0].doublet_alternatives
        assert len(alts) == 1
        got = alts[0]
        assert got.frequency_a_mhz == pytest.approx(alt.frequency_a_mhz)
        assert got.frequency_b_mhz == pytest.approx(alt.frequency_b_mhz)
        assert got.amplitude_a == pytest.approx(alt.amplitude_a)
        assert got.amplitude_b == pytest.approx(alt.amplitude_b)
        assert got.separation_res_elements == pytest.approx(alt.separation_res_elements)
        assert got.chi2r_production == pytest.approx(alt.chi2r_production)
        assert got.chi2r_merged == pytest.approx(alt.chi2r_merged)
        assert got.delta_chi2_raw == pytest.approx(alt.delta_chi2_raw)
        assert got.delta_aicc == pytest.approx(alt.delta_aicc)
        assert got.merged_success == alt.merged_success
        assert got.orth_evidence_delta_chi2 == pytest.approx(
            alt.orth_evidence_delta_chi2
        )
        assert got.orth_evidence_n_params == alt.orth_evidence_n_params
        assert got.support_bins == alt.support_bins

    def test_round_trip_with_failed_merge_nans(self, tmp_path):
        """DoubletAlternativeInfo with NaN fields (merge failed) round-trips cleanly."""
        alt = _make_doublet_alt(merged_success=False)
        assert np.isnan(alt.chi2r_merged)
        assert np.isnan(alt.delta_chi2_raw)

        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        win.doublet_alternatives = [alt]
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        got = loaded.window_fits[0].doublet_alternatives[0]
        assert np.isnan(got.chi2r_merged)
        assert np.isnan(got.delta_chi2_raw)
        assert got.merged_success is False

    def test_empty_doublet_alternatives_round_trips(self, tmp_path):
        """A window with no doublet alternatives saves and loads as an empty list."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        assert win.doublet_alternatives == []
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        assert loaded.window_fits[0].doublet_alternatives == []

    def test_legacy_file_missing_attr_loads_as_empty(self, tmp_path):
        """Files written before doublet_alternatives existed load with an empty list."""
        path = tmp_path / "fit.h5"
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            # Simulate a file written before the attr existed.
            if "doublet_alternatives" in g["windows/window_0000"].attrs:
                del g["windows/window_0000"].attrs["doublet_alternatives"]
        with h5py.File(path, "r") as h5f:
            loaded = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert loaded.window_fits[0].doublet_alternatives == []

    def test_two_alts_in_one_window_round_trip(self, tmp_path):
        """Multiple DoubletAlternativeInfo objects per window all survive."""
        alt1 = _make_doublet_alt(freq_a=36100.1, freq_b=36100.4, merged_success=True)
        alt2 = _make_doublet_alt(freq_a=36200.1, freq_b=36200.35, merged_success=False)
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        win.doublet_alternatives = [alt1, alt2]
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        alts = loaded.window_fits[0].doublet_alternatives
        assert len(alts) == 2
        assert alts[0].frequency_a_mhz == pytest.approx(alt1.frequency_a_mhz)
        assert alts[1].frequency_a_mhz == pytest.approx(alt2.frequency_a_mhz)
        assert alts[0].merged_success is True
        assert alts[1].merged_success is False


# ---------------------------------------------------------------------------
# Per-window parameter covariance
# ---------------------------------------------------------------------------
def _make_window_fit_with_covariance(
    window_id: int,
    n_peaks: int,
    *,
    fit_tau: bool = True,
    baseline_order: int | None = None,
) -> FittingResult:
    """Build a FittingResult with a synthetic covariance matrix attached."""
    from ftmwpipeline.fitting.result_conversion import build_covariance_param_labels

    peaks = [
        _sample_fitted_peak(peak_id=i, window_id=window_id, freq_mhz=36100.0 + i * 0.5)
        for i in range(n_peaks)
    ]
    labels = build_covariance_param_labels(
        n_peaks=n_peaks, fit_tau=fit_tau, baseline_order=baseline_order
    )
    d = len(labels)
    # Use a positive-definite symmetric matrix so the values are physically
    # plausible; a diagonal matrix is simplest.
    rng = np.random.default_rng(42)
    cov = np.diag(rng.uniform(1e-6, 1e-3, size=d))

    fr = _make_window_fit(window_id, peaks, audit=[], thaw_events=[])
    fr.covariance = cov
    fr.covariance_param_labels = labels
    return fr


class TestCovarianceRoundTrip:
    def test_covariance_round_trips(self, tmp_path):
        """A FittingResult with a covariance matrix persists and loads back
        with the exact same values and labels."""
        from ftmwpipeline.fitting.result_conversion import build_covariance_param_labels

        win = _make_window_fit_with_covariance(
            0, n_peaks=2, fit_tau=True, baseline_order=1
        )
        expected_labels = build_covariance_param_labels(
            n_peaks=2, fit_tau=True, baseline_order=1
        )
        fit = SpectrumFit(window_fits=[win], fitted_peaks=list(win.fitted_peaks))

        loaded = _roundtrip(fit, tmp_path / "fit.h5")

        wf = loaded.window_fits[0]
        assert wf.covariance is not None
        assert wf.covariance_param_labels == expected_labels
        np.testing.assert_array_equal(wf.covariance, win.covariance)

    def test_covariance_none_round_trips_as_none(self, tmp_path):
        """A FittingResult with no covariance writes no dataset and loads back
        as None (back-compat with older files that never had a covariance)."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        assert win.covariance is None
        assert win.covariance_param_labels is None

        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])
        path = tmp_path / "fit.h5"
        loaded = _roundtrip(fit, path)

        # No dataset written.
        with h5py.File(path, "r") as h5f:
            assert "covariance" not in h5f["stage5_fitting/windows/window_0000"]

        wf = loaded.window_fits[0]
        assert wf.covariance is None
        assert wf.covariance_param_labels is None

    def test_covariance_no_tau_no_baseline_labels(self, tmp_path):
        """Labels are peak-only when fit_tau=False and baseline_order=None."""
        from ftmwpipeline.fitting.result_conversion import build_covariance_param_labels

        labels = build_covariance_param_labels(
            n_peaks=3, fit_tau=False, baseline_order=None
        )
        assert labels == [
            "amplitude_0",
            "offset_0",
            "phase_0",
            "amplitude_1",
            "offset_1",
            "phase_1",
            "amplitude_2",
            "offset_2",
            "phase_2",
        ]

    def test_covariance_labels_with_tau_and_baseline(self, tmp_path):
        """Labels include tau + baseline blocks in the right order."""
        from ftmwpipeline.fitting.result_conversion import build_covariance_param_labels

        labels = build_covariance_param_labels(
            n_peaks=1, fit_tau=True, baseline_order=2
        )
        assert labels == [
            "amplitude_0",
            "offset_0",
            "phase_0",
            "tau",
            "baseline_re_0",
            "baseline_re_1",
            "baseline_re_2",
            "baseline_im_0",
            "baseline_im_1",
            "baseline_im_2",
        ]

    def test_malformed_covariance_non_square_raises(self, tmp_path):
        """A non-square covariance dataset raises ValueError on load."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        path = tmp_path / "fit.h5"
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            wg = g["windows/window_0000"]
            # Inject a non-square matrix and a matching-length label list.
            wg.create_dataset("covariance", data=np.zeros((3, 4), dtype="f8"))
            wg.attrs["covariance_param_labels"] = json.dumps(
                ["amplitude_0", "offset_0", "phase_0"]
            )
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="not square"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_malformed_covariance_label_length_mismatch_raises(self, tmp_path):
        """Label list length != matrix dimension raises ValueError on load."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        path = tmp_path / "fit.h5"
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            wg = g["windows/window_0000"]
            wg.create_dataset("covariance", data=np.eye(3, dtype="f8"))
            # Wrong number of labels (2 instead of 3).
            wg.attrs["covariance_param_labels"] = json.dumps(
                ["amplitude_0", "offset_0"]
            )
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="label count"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    def test_malformed_covariance_amplitude_count_mismatch_raises(self, tmp_path):
        """amplitude_* label count != fitted_peaks count raises ValueError."""
        peak = _sample_fitted_peak(peak_id=0, window_id=0, freq_mhz=36100.0)
        win = _make_window_fit(0, [peak], audit=[], thaw_events=[])
        fit = SpectrumFit(window_fits=[win], fitted_peaks=[peak])

        path = tmp_path / "fit.h5"
        with h5py.File(path, "w") as h5f:
            g = h5f.create_group("stage5_fitting")
            save_spectrum_fit_to_hdf5(fit, g)
            wg = g["windows/window_0000"]
            # 2 peaks worth of labels but only 1 fitted peak.
            labels_2peaks = [
                "amplitude_0",
                "offset_0",
                "phase_0",
                "amplitude_1",
                "offset_1",
                "phase_1",
            ]
            wg.create_dataset("covariance", data=np.eye(6, dtype="f8"))
            wg.attrs["covariance_param_labels"] = json.dumps(labels_2peaks)
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="amplitude label"):
                load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
