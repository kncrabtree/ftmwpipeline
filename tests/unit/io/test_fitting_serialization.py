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
    FittedPeak,
    FittingResult,
    KnockoutInfo,
    ReplanInfo,
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
