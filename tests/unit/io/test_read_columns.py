"""
Unit tests for the read-only bulk column accessors in ``io.*_serialization``.

These are the narrow counterparts to the full ``load_*_from_hdf5`` loaders: a
consumer that only wants a few columns must get exactly the values the full
loader would have produced, without the object-graph reconstruction. So the
tests here are mostly *parity* tests -- read the columns, load the full record,
assert they agree -- plus coverage of column selection, the back-compat fills
for columns older files do not carry, and loud validation of malformed groups.
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
    FitWindow,
    FixedContributor,
    KnockoutInfo,
    Peak,
    PeakClassification,
    ReplanInfo,
    RescueCandidateInfo,
    RescueRoundInfo,
    SpectralWindow,
    SpectrumFit,
    ThawInfo,
    WindowPlan,
)
from ftmwpipeline.fitting.tau_calibration import (
    BandMajority,
    FrequencyThird,
    GMMBimodality,
    SpurCluster,
    TauCalibrationResult,
)
from ftmwpipeline.io.fitting_serialization import (
    FIT_AUDIT_COLUMN_SPECS,
    FIT_DOUBLET_COLUMN_SPECS,
    FIT_PEAK_COLUMN_SPECS,
    FIT_REPLAN_COLUMN_SPECS,
    FIT_RESCUE_COLUMN_SPECS,
    FIT_THAW_COLUMN_SPECS,
    FIT_WINDOW_COLUMN_SPECS,
    load_spectrum_fit_from_hdf5,
    read_fit_audit_columns,
    read_fit_doublet_columns,
    read_fit_peak_columns,
    read_fit_replan_columns,
    read_fit_rescue_columns,
    read_fit_scalars,
    read_fit_thaw_columns,
    read_fit_window_columns,
    save_spectrum_fit_to_hdf5,
)
from ftmwpipeline.io.peak_serialization import (
    PEAK_COLUMN_SPECS,
    load_peaks_from_hdf5,
    read_peak_columns,
    read_peak_scalars,
    save_peaks_to_hdf5,
)
from ftmwpipeline.io.tau_calibration_serialization import (
    TAU_BAND_COLUMN_SPECS,
    TAU_CONTRIBUTOR_COLUMN_SPECS,
    TAU_SPUR_COLUMN_SPECS,
    TAU_THIRD_COLUMN_SPECS,
    load_tau_calibration_from_hdf5,
    read_tau_band_columns,
    read_tau_contributor_columns,
    read_tau_scalars,
    read_tau_spur_columns,
    read_tau_third_columns,
    save_tau_calibration_to_hdf5,
)
from ftmwpipeline.io.window_serialization import (
    WINDOW_CONTRIBUTOR_COLUMN_SPECS,
    WINDOW_FREE_PEAK_COLUMN_SPECS,
    WINDOW_PLAN_COLUMN_SPECS,
    load_window_plan_from_hdf5,
    read_window_contributor_columns,
    read_window_free_peak_columns,
    read_window_plan_columns,
    read_window_plan_scalars,
    save_window_plan_to_hdf5,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _fitted_peak(detection_index: int, window_id: int, freq_mhz: float) -> FittedPeak:
    return FittedPeak(
        detection_index=detection_index,
        frequency_mhz=freq_mhz,
        amplitude=0.5 + 0.1 * detection_index,
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
        clock_lattice="320x6 (bb)" if detection_index == 0 else None,
        origin="user" if detection_index == 2 else "auto",
        flat_decay=detection_index == 1,
        derivation=7 if detection_index == 2 else None,
        peak_uid=4213 if detection_index == 0 else None,
    )


def _window_fit(
    window_id: int,
    peaks: list[FittedPeak],
    *,
    shape: str = "lorentzian",
    tau_us: float = 5.0,
    freq_range: tuple[float, float] | None = None,
) -> FittingResult:
    window = None
    if freq_range is not None:
        window = SpectralWindow(
            parent_ft=None,
            freq_array=np.array([], dtype=float),
            complex_spectrum=np.array([], dtype=np.complex128),
            freq_range=freq_range,
            window_id=window_id,
        )
    fr = FittingResult(
        success=window_id != 2,
        fitted_spectrum=None,
        cost=12.34 + window_id,
        iterations=3 + window_id,
        aic=1880.0 - window_id,
        reduced_chi2=1.02,
        window=window,
        window_id=window_id,
        shape=shape,
    )
    fr.fitted_peaks = list(peaks)
    fr.shared_parameters["tau_us"] = {
        "value": tau_us,
        "error": 0.05,
        "fitted": True,
        "detection_indices": [p.detection_index for p in peaks],
    }
    fr.quality_metrics = {
        "edge_coherence_low": 0.8,
        "edge_coherence_high": 0.9,
    }
    # One audit step per fitted peak, so the trail's length varies by window and
    # the long-form table's grouping is actually exercised.
    fr.audit_trail = [
        AuditStep(
            n_peaks_before=i,
            candidate_offset_mhz=-0.12 + 0.1 * i,
            chi2_before=1234.5,
            chi2_after=200.1,
            f_statistic=80.0,
            p_value=1e-30,
            aic_before=2050.0,
            aic_after=1900.0,
            separation_ok=True,
            decision="seed" if i == 0 else "accept",
            reason=f"step {i}",
            n_eff=42.0,
            aicc_delta=-9.5,
        )
        for i in range(len(peaks))
    ]
    if peaks:
        fr.doublet_alternatives = [
            DoubletAlternativeInfo(
                frequency_a_mhz=peaks[0].frequency_mhz,
                frequency_b_mhz=peaks[0].frequency_mhz + 0.01,
                amplitude_a=1.0,
                amplitude_b=0.5,
                separation_res_elements=0.8,
                amp_ratio=0.5,
                chi2r_production=1.1,
                chi2r_merged=1.4,
                delta_chi2_raw=12.0,
                delta_aicc=-3.5,
                merged_frequency_mhz=peaks[0].frequency_mhz + 0.005,
                merged_amplitude=1.4,
                merged_phase=0.25,
                merged_tau_us=5.1,
                merged_success=True,
                orth_evidence_delta_chi2=8.0,
                orth_evidence_n_params=3,
                support_bins=6,
            )
        ]
    return fr


def _sample_thaw() -> ThawInfo:
    return ThawInfo(
        dependent_window_id=1,
        primary_window_id=0,
        contributor_peak_index=3,
        contributor_frequency_mhz=36105.5,
        edge_side="low",
        edge_coherence_before=9.1,
        edge_coherence_after=1.4,
        accepted=True,
        reason="edge incoherent with the contributor frozen",
    )


def _sample_replan() -> ReplanInfo:
    return ReplanInfo(
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


def _sample_rescue() -> RescueRoundInfo:
    return RescueRoundInfo(
        window_id=0,
        round_idx=0,
        n_initial_peaks=2,
        n_rescue_added=1,
        n_pruned_total=0,
        n_pruned_rescue_origin=0,
        n_merged=0,
        chi2_before=300.0,
        chi2_after=210.0,
        tau_us_before=5.0,
        tau_us_after=5.2,
        accepted=True,
        reason="residual carried a real line",
        candidates=[
            RescueCandidateInfo(frequency_mhz=36111.0, magnitude=0.4, snr=7.5),
            RescueCandidateInfo(frequency_mhz=36113.0, magnitude=0.2, snr=4.0),
        ],
    )


def _sample_fit() -> SpectrumFit:
    # Window ids and molecular frequencies are deliberately anti-correlated so
    # the row ordering (by frequency, not by window) is actually exercised.
    win0 = _window_fit(
        0,
        [_fitted_peak(0, 0, 36110.045), _fitted_peak(1, 0, 36112.500)],
        shape="gaussian",
        freq_range=(36105.0, 36115.0),
    )
    win1 = _window_fit(
        1,
        [_fitted_peak(2, 1, 36100.012)],
        tau_us=6.5,
        freq_range=(36095.0, 36105.0),
    )
    win2 = _window_fit(2, [], tau_us=4.0)
    peaks = win0.fitted_peaks + win1.fitted_peaks
    return SpectrumFit(
        window_fits=[win0, win1, win2],
        fitted_peaks=sorted(peaks, key=lambda p: p.frequency_mhz),
        thaw_history=[_sample_thaw()],
        replan_history=[_sample_replan()],
        rescue_history=[_sample_rescue()],
        final_plan_revision=1,
        parameters={"tau0_us": 5.0, "acquisition_us": 12.73},
        diagnostics={"note": "synthetic"},
    )


def _sample_plan() -> WindowPlan:
    return WindowPlan(
        windows=[
            FitWindow(
                window_id=0,
                freq_range=(26500.0, 26520.0),
                free_peak_indices=[0, 1, 2],
                batch=0,
                diagnostics={"edge_coherence_fail": False},
            ),
            FitWindow(
                window_id=1,
                freq_range=(26530.0, 26536.0),
                free_peak_indices=[5],
                fixed_contributors=[
                    FixedContributor(
                        peak_index=1,
                        primary_window_id=0,
                        frequency_mhz=26510.0,
                        freeze_eligible=False,
                        edge_free=True,
                    )
                ],
                batch=1,
            ),
        ],
        dependency_edges=[(1, 0)],
        topological_order=[0, 1],
        parameters={"edge_m": 64},
        diagnostics={},
    )


def _sample_peaks() -> list[Peak]:
    return [
        Peak(
            frequency=26501.5,
            intensity=1200.0,
            index=41,
            snr=25.0,
            noise_std_local=48.0,
            classification=PeakClassification.STRONG,
            detection_pass="primary",
            internal_snr=26.0,
            internal_frequency=26501.4,
            leakage_pedestal=3.0,
        ),
        Peak(
            frequency=26510.25,
            intensity=200.0,
            index=87,
            snr=4.0,
            noise_std_local=48.0,
            classification=PeakClassification.WEAK,
            detection_pass="gap",
            internal_snr=4.2,
            internal_frequency=26510.2,
            leakage_pedestal=2.0,
        ),
    ]


@pytest.fixture
def fit_file(tmp_path):
    path = tmp_path / "fit.h5"
    with h5py.File(path, "w") as h5f:
        save_spectrum_fit_to_hdf5(_sample_fit(), h5f.create_group("stage5_fitting"))
    return path


@pytest.fixture
def plan_file(tmp_path):
    path = tmp_path / "plan.h5"
    with h5py.File(path, "w") as h5f:
        save_window_plan_to_hdf5(_sample_plan(), h5f.create_group("stage4_windows"))
    return path


def _sample_tau(*, with_bands: bool = True) -> TauCalibrationResult:
    """A small but complete decay-time calibration.

    Band majorities are optional on disk (the calibration can run without
    them), so ``with_bands=False`` builds the shape a bounds reader must still
    handle -- an empty table, not an error.
    """
    n = 24
    freqs = np.linspace(27000.0, 39000.0, n)
    bands: tuple = ()
    if with_bands:
        bands = (
            # Deliberately out of frequency order on disk, so the reader's
            # ordering is actually exercised.
            BandMajority("high", 35000.0, 39000.0, 8, 5.4, 0.9),
            BandMajority("low", 27000.0, 31000.0, 9, 7.7, 1.2),
            BandMajority("mid", 31000.0, 35000.0, 7, 6.1, 1.1),
        )
    return TauCalibrationResult(
        tau_maj_us=6.33,
        sigma_tau_us=1.62,
        n_contributors=n,
        n_spur_bins=9,
        spur_clusters=(
            SpurCluster(
                center_freq_mhz=28000.0,
                peak_bin_index=1000,
                n_bins=9,
                bin_indices=tuple(range(996, 1005)),
            ),
        ),
        bimodality=GMMBimodality(
            n=n,
            mu1=6.0,
            sigma1=1.0,
            mu_a=5.0,
            sigma_a=0.8,
            mu_b=7.5,
            sigma_b=1.2,
            pi_a=0.6,
            aic1=120.0,
            aic2=110.5,
            delta_aic=9.5,
            two_component_preferred=True,
            dominant_weight=0.6,
        ),
        pearson_r_log_snr_vs_tau=-0.30,
        pearson_r_freq_vs_tau=-0.32,
        frequency_thirds=(
            FrequencyThird("low", 27000.0, 31000.0, 8, 7.34),
            FrequencyThird("mid", 31000.0, 35000.0, 8, 6.36),
            FrequencyThird("high", 35000.0, 39000.0, 8, 6.06),
        ),
        contributor_bin_indices=np.arange(100, 100 + n, dtype=np.int64),
        contributor_taus_us=np.linspace(5.0, 7.0, n),
        contributor_snrs=np.linspace(10.0, 100.0, n),
        contributor_freqs_mhz=freqs,
        band_majorities=bands,
        n_seg=10,
        t_sigma=5.0,
        tau_max_us=63.25,
        rss_gate_factor=5.0,
        sample_dt_us=0.020,
        start_us=2.35,
        end_us=15.0,
        probe_freq_mhz=40000.0,
        sideband="lower",
        trim_lo_mhz=26500.0,
        trim_hi_mhz=40000.0,
        sigma_x_full=4.68e-7,
        sigma_frame=1.48e-7,
        snr_weighted=True,
        preconditions_passed=False,
        preconditions_notes=("ok", "strongly bimodal", "spread too wide"),
    )


def _write_tau(path, result: TauCalibrationResult, *, vote: bool = True):
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("stage2b_tau_calibration")
        save_tau_calibration_to_hdf5(result, group)
        group.attrs["creation_time"] = "2026-08-10T00:00:00"
        if vote:
            # Written by the Stage 2b impl, not the serializer: what the file
            # is set to use, what the vote favored, and the vote itself.
            group.attrs["shape"] = "lorentzian"
            group.attrs["recommended_shape"] = "gaussian"
            group.attrs["shape_vote_rates"] = json.dumps(
                {"exp": 0.12, "gauss": 0.66, "voigt": 0.22}
            )
    return path


@pytest.fixture
def tau_file(tmp_path):
    return _write_tau(tmp_path / "tau.h5", _sample_tau())


@pytest.fixture
def tau_file_no_bands(tmp_path):
    return _write_tau(tmp_path / "tau_nobands.h5", _sample_tau(with_bands=False))


@pytest.fixture
def peaks_file(tmp_path):
    path = tmp_path / "peaks.h5"
    with h5py.File(path, "w") as h5f:
        save_peaks_to_hdf5(
            _sample_peaks(),
            h5f.create_group("stage3_peaks"),
            parameters={"promotion_min_snr": 5.0, "internal_min_snr": 3.0},
        )
    return path


# ---------------------------------------------------------------------------
# Stage 5 fitted peaks
# ---------------------------------------------------------------------------
class TestReadFitPeakColumns:
    def test_matches_the_full_loader_row_for_row(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        expected = fit.fitted_peaks
        assert len(cols["detection_index"]) == len(expected)
        np.testing.assert_array_equal(
            cols["detection_index"], [p.detection_index for p in expected]
        )
        np.testing.assert_allclose(
            cols["frequency_mhz"], [p.frequency_mhz for p in expected]
        )
        np.testing.assert_allclose(cols["amplitude"], [p.amplitude for p in expected])
        np.testing.assert_allclose(cols["decay_rate"], [p.decay_rate for p in expected])
        np.testing.assert_array_equal(
            cols["window_id"], [p.window_id for p in expected]
        )
        assert list(cols["origin"]) == [p.origin for p in expected]
        assert list(cols["flat_decay"]) == [p.flat_decay for p in expected]

    def test_rows_are_ordered_by_molecular_frequency(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(h5f["stage5_fitting"])
        freqs = cols["frequency_mhz"]
        assert list(freqs) == sorted(freqs)
        # The window that owns the lowest-frequency peak is not the first window
        # group on disk, so this is a real ordering assertion.
        assert cols["window_id"][0] == 1

    def test_shape_is_broadcast_from_the_owning_window(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(
                h5f["stage5_fitting"], columns=["window_id", "shape"]
            )
        by_window = dict(zip(cols["window_id"], cols["shape"]))
        assert by_window == {0: "gaussian", 1: "lorentzian"}

    def test_a_peak_with_no_window_id_of_its_own_is_stamped_with_its_group(
        self, tmp_path
    ):
        """The writer must not put ``-1`` where the owning group has an id.

        ``FittedPeak.window_id`` is optional, but a peak is *stored inside* a
        window group, so the group always answers the grouping question. A
        ``-1`` in the column would let a reader that trusts the column drop the
        row out of its window with no error raised.
        """
        orphan = _fitted_peak(9, 0, 36111.0)
        orphan.window_id = None
        fit = _sample_fit()
        fit.window_fits[0].fitted_peaks = [orphan]

        path = tmp_path / "orphan.h5"
        with h5py.File(path, "w") as h5f:
            save_spectrum_fit_to_hdf5(fit, h5f.create_group("stage5_fitting"))
        with h5py.File(path, "r") as h5f:
            stored = h5f["stage5_fitting/peaks/window_id"][:]
            offsets = h5f["stage5_fitting/windows/peak_offset"][:]
            counts = h5f["stage5_fitting/windows/peak_count"][:]
        # Window 0's slice of the shared peak table -- the orphan row.
        start, count = int(offsets[0]), int(counts[0])
        assert list(stored[start : start + count]) == [0]

    def test_a_stored_minus_one_window_id_is_backfilled_from_the_group(self, fit_file):
        """Files written before the writer stamped the group's id still group.

        The tap and the full loader must agree on which window owns the row --
        the loader takes it from the group, so the tap does too. Without the
        backfill the row is silently ungrouped: no exception, no warning, just
        a peak that appears to belong to no window.
        """
        # Reach behind the writer to recreate the on-disk state an older
        # version produced for a peak whose own window_id was None.
        with h5py.File(fit_file, "r+") as h5f:
            h5f["stage5_fitting/peaks/window_id"][0] = -1

        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(
                h5f["stage5_fitting"], columns=["detection_index", "window_id"]
            )
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        assert -1 not in set(cols["window_id"])
        # Grouping agrees with the loader's, which reads it off the group.
        loader_grouping = {
            p.detection_index: wf.window_id
            for wf in fit.window_fits
            for p in wf.fitted_peaks
        }
        assert dict(zip(cols["detection_index"], cols["window_id"])) == loader_grouping
        # And the loader's own per-peak field is backfilled the same way, so
        # the two never disagree regardless of which one a consumer reads.
        assert all(p.window_id is not None for p in fit.fitted_peaks)

    def test_none_valued_fields_come_back_as_their_sentinels(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(
                h5f["stage5_fitting"],
                columns=["detection_index", "clock_lattice", "derivation"],
            )
        by_id = dict(zip(cols["detection_index"], cols["clock_lattice"]))
        assert by_id[0] == "320x6 (bb)"
        assert by_id[1] == ""  # None -> empty string, not None
        derivation = dict(zip(cols["detection_index"], cols["derivation"]))
        assert derivation[2] == 7
        assert derivation[1] == -1  # None -> -1, not None

    def test_peak_uid_column_round_trips_and_absent_rows_are_sentinel(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(
                h5f["stage5_fitting"], columns=["detection_index", "peak_uid"]
            )
        peak_uid = dict(zip(cols["detection_index"], cols["peak_uid"]))
        assert peak_uid[0] == 4213
        assert peak_uid[1] == -1  # None -> -1, not None
        assert peak_uid[2] == -1

    def test_row_count_raises_naming_detection_index_when_absent(self, fit_file):
        """Neither ``detection_index`` nor ``peak_id`` present: the row-count
        helper must raise, naming the current column name."""
        with h5py.File(fit_file, "r+") as h5f:
            h5f["stage5_fitting/peaks"].move("detection_index", "something_else")

        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="detection_index"):
                read_fit_peak_columns(h5f["stage5_fitting"])

    def test_column_selection_is_honored_in_order(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(
                h5f["stage5_fitting"], columns=["snr", "frequency_mhz"]
            )
        assert list(cols) == ["snr", "frequency_mhz"]

    def test_selecting_without_frequency_still_orders_by_frequency(self, fit_file):
        """The ordering column is read even when the caller does not want it."""
        with h5py.File(fit_file, "r") as h5f:
            selected = read_fit_peak_columns(
                h5f["stage5_fitting"], columns=["detection_index"]
            )
            everything = read_fit_peak_columns(h5f["stage5_fitting"])
        np.testing.assert_array_equal(
            selected["detection_index"], everything["detection_index"]
        )
        assert "frequency_mhz" not in selected

    def test_unknown_column_raises(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="unknown column"):
                read_fit_peak_columns(h5f["stage5_fitting"], columns=["nope"])

    def test_empty_column_list_raises(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="no columns requested"):
                read_fit_peak_columns(h5f["stage5_fitting"], columns=[])

    def test_missing_windows_subgroup_raises(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as h5f:
            h5f.create_group("stage5_fitting")
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="missing required 'windows'"):
                read_fit_window_columns(h5f["stage5_fitting"])

    def test_missing_required_column_raises(self, fit_file):
        with h5py.File(fit_file, "a") as h5f:
            del h5f["stage5_fitting/peaks/amplitude"]
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="missing required column 'amplitude'"):
                read_fit_peak_columns(h5f["stage5_fitting"])

    def test_mismatched_column_length_raises(self, fit_file):
        with h5py.File(fit_file, "a") as h5f:
            group = h5f["stage5_fitting/peaks"]
            del group["snr"]
            group.create_dataset("snr", data=np.array([1.0], dtype="f8"))
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="has length 1, expected"):
                read_fit_peak_columns(h5f["stage5_fitting"])

    def test_absent_optional_column_reads_back_as_its_fill(self, fit_file):
        """An older file with no ``derivation`` column reads as all -1 (None)."""
        with h5py.File(fit_file, "a") as h5f:
            del h5f["stage5_fitting/peaks/derivation"]
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(h5f["stage5_fitting"], columns=["derivation"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert list(cols["derivation"]) == [-1, -1, -1]
        assert all(p.derivation is None for p in fit.fitted_peaks)

    def test_absent_peak_uid_column_reads_back_as_none(self, fit_file):
        """An older file with no ``peak_uid`` column reads as all -1 (None)."""
        with h5py.File(fit_file, "a") as h5f:
            del h5f["stage5_fitting/peaks/peak_uid"]
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(h5f["stage5_fitting"], columns=["peak_uid"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        assert list(cols["peak_uid"]) == [-1, -1, -1]
        assert all(p.peak_uid is None for p in fit.fitted_peaks)

    def test_every_declared_column_is_readable(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_peak_columns(h5f["stage5_fitting"])
        assert list(cols) == list(FIT_PEAK_COLUMN_SPECS)
        assert len({len(v) for v in cols.values()}) == 1


# ---------------------------------------------------------------------------
# Stage 5 per-window scalars
# ---------------------------------------------------------------------------
class TestReadFitWindowColumns:
    def test_matches_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_window_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        expected = fit.window_fits
        np.testing.assert_array_equal(
            cols["window_id"], [w.window_id for w in expected]
        )
        np.testing.assert_array_equal(cols["success"], [w.success for w in expected])
        np.testing.assert_allclose(cols["aic"], [w.aic for w in expected])
        np.testing.assert_allclose(
            cols["reduced_chi2"], [w.reduced_chi2 for w in expected]
        )
        np.testing.assert_allclose(
            cols["tau_us"],
            [w.shared_parameters["tau_us"]["value"] for w in expected],
        )
        np.testing.assert_array_equal(
            cols["n_peaks"], [len(w.fitted_peaks) for w in expected]
        )
        assert list(cols["shape"]) == [w.shape for w in expected]

    def test_bounds_match_the_reconstructed_windows(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_window_columns(
                h5f["stage5_fitting"], columns=["window_id", "freq_min", "freq_max"]
            )
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        for i, wf in enumerate(fit.window_fits):
            if wf.window is None:
                # No persisted freq_range -> NaN bounds, not a fabricated value.
                assert np.isnan(cols["freq_min"][i])
                assert np.isnan(cols["freq_max"][i])
            else:
                assert cols["freq_min"][i] == pytest.approx(wf.window.freq_range[0])
                assert cols["freq_max"][i] == pytest.approx(wf.window.freq_range[1])

    def test_rows_are_ordered_by_window_id(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_window_columns(h5f["stage5_fitting"])
        assert list(cols["window_id"]) == [0, 1, 2]

    def test_missing_required_attr_raises(self, fit_file):
        with h5py.File(fit_file, "a") as h5f:
            del h5f["stage5_fitting/windows/tau_us"]
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="missing required column 'tau_us'"):
                read_fit_window_columns(h5f["stage5_fitting"])

    def test_every_declared_column_is_readable(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_window_columns(h5f["stage5_fitting"])
        assert list(cols) == list(FIT_WINDOW_COLUMN_SPECS)
        assert len({len(v) for v in cols.values()}) == 1


class TestReadFitEventLogs:
    """The fit's decision record, presented as tables."""

    def test_audit_matches_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_audit_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        expected = [
            (wf.window_id, i, step.decision, step.reason)
            for wf in fit.window_fits
            for i, step in enumerate(wf.audit_trail)
        ]
        assert (
            list(
                zip(
                    cols["window_id"],
                    cols["step_index"],
                    cols["decision"],
                    cols["reason"],
                )
            )
            == expected
        )
        np.testing.assert_allclose(
            cols["p_value"],
            [s.p_value for wf in fit.window_fits for s in wf.audit_trail],
        )

    def test_audit_step_index_restarts_per_window(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_audit_columns(
                h5f["stage5_fitting"], columns=["window_id", "step_index"]
            )
        # window 0 has two peaks (two steps), window 1 has one, window 2 none.
        assert list(zip(cols["window_id"], cols["step_index"])) == [
            (0, 0),
            (0, 1),
            (1, 0),
        ]

    def test_doublets_match_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_doublet_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        expected = [
            (wf.window_id, d.frequency_a_mhz, d.amp_ratio)
            for wf in fit.window_fits
            for d in wf.doublet_alternatives
        ]
        assert (
            list(zip(cols["window_id"], cols["frequency_a_mhz"], cols["amp_ratio"]))
            == expected
        )

    def test_doublet_optional_fields_take_their_fills(self, fit_file):
        """An older record that omits a field reads back as its documented fill."""
        with h5py.File(fit_file, "a") as h5f:
            windows = h5f["stage5_fitting/windows"]
            for row in range(windows["doublet_alternatives"].shape[0]):
                raw = windows["doublet_alternatives"][row]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                blobs = json.loads(raw) if raw else []
                for blob in blobs:
                    del blob["chi2r_merged"]
                    del blob["support_bins"]
                windows["doublet_alternatives"][row] = json.dumps(blobs)
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_doublet_columns(h5f["stage5_fitting"])
        assert np.isnan(cols["chi2r_merged"]).all()
        assert cols["support_bins"].tolist() == [0] * len(cols["support_bins"])

    def test_thaw_matches_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_thaw_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        events = fit.thaw_history
        np.testing.assert_array_equal(
            cols["dependent_window_id"], [e.dependent_window_id for e in events]
        )
        assert list(cols["edge_side"]) == [e.edge_side for e in events]
        assert list(cols["accepted"]) == [e.accepted for e in events]
        assert list(cols["reason"]) == [e.reason for e in events]

    def test_replans_match_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_replan_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        events = fit.replan_history
        np.testing.assert_array_equal(
            cols["triggering_window_id"], [e.triggering_window_id for e in events]
        )
        np.testing.assert_array_equal(
            cols["revision_after"], [e.revision_after for e in events]
        )

    def test_rescues_match_the_full_loader(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            cols = read_fit_rescue_columns(h5f["stage5_fitting"])
            fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        rounds = fit.rescue_history
        np.testing.assert_array_equal(cols["window_id"], [r.window_id for r in rounds])
        np.testing.assert_allclose(cols["chi2_after"], [r.chi2_after for r in rounds])
        # n_candidates counts a nested list, which is not a stored field.
        np.testing.assert_array_equal(
            cols["n_candidates"], [len(r.candidates) for r in rounds]
        )

    def test_empty_logs_read_as_empty_tables(self, tmp_path):
        path = tmp_path / "bare.h5"
        bare = SpectrumFit(window_fits=[], fitted_peaks=[])
        with h5py.File(path, "w") as h5f:
            save_spectrum_fit_to_hdf5(bare, h5f.create_group("stage5_fitting"))
        with h5py.File(path, "r") as h5f:
            group = h5f["stage5_fitting"]
            for reader in (
                read_fit_audit_columns,
                read_fit_doublet_columns,
                read_fit_thaw_columns,
                read_fit_replan_columns,
                read_fit_rescue_columns,
            ):
                cols = reader(group)
                assert all(len(v) == 0 for v in cols.values())

    def test_every_declared_column_is_readable(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            group = h5f["stage5_fitting"]
            for reader, specs in (
                (read_fit_audit_columns, FIT_AUDIT_COLUMN_SPECS),
                (read_fit_doublet_columns, FIT_DOUBLET_COLUMN_SPECS),
                (read_fit_thaw_columns, FIT_THAW_COLUMN_SPECS),
                (read_fit_replan_columns, FIT_REPLAN_COLUMN_SPECS),
                (read_fit_rescue_columns, FIT_RESCUE_COLUMN_SPECS),
            ):
                cols = reader(group)
                assert list(cols) == list(specs)
                assert len({len(v) for v in cols.values()}) == 1

    def test_a_record_missing_a_required_field_raises(self, fit_file):
        with h5py.File(fit_file, "a") as h5f:
            group = h5f["stage5_fitting"]
            blobs = json.loads(group.attrs["thaw_history"])
            del blobs[0]["edge_side"]
            group.attrs["thaw_history"] = json.dumps(blobs)
        with h5py.File(fit_file, "r") as h5f:
            with pytest.raises(ValueError, match="missing required field 'edge_side'"):
                read_fit_thaw_columns(h5f["stage5_fitting"])


class TestReadFitScalars:
    def test_reports_the_plan_level_scalars(self, fit_file):
        with h5py.File(fit_file, "r") as h5f:
            scalars = read_fit_scalars(h5f["stage5_fitting"])
        assert scalars["n_windows"] == 3
        assert scalars["n_fitted_peaks"] == 3
        assert scalars["final_plan_revision"] == 1
        assert scalars["acquisition_us"] == pytest.approx(12.73)

    def test_acquisition_us_is_none_when_the_fit_did_not_record_it(self, fit_file):
        with h5py.File(fit_file, "a") as h5f:
            h5f["stage5_fitting"].attrs["parameters"] = json.dumps({"tau0_us": 5.0})
        with h5py.File(fit_file, "r") as h5f:
            assert read_fit_scalars(h5f["stage5_fitting"])["acquisition_us"] is None


# ---------------------------------------------------------------------------
# Stage 4 window plan
# ---------------------------------------------------------------------------
class TestReadWindowPlanColumns:
    def test_matches_the_full_loader(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_plan_columns(h5f["stage4_windows"])
            plan = load_window_plan_from_hdf5(h5f["stage4_windows"])

        np.testing.assert_array_equal(
            cols["window_id"], [w.window_id for w in plan.windows]
        )
        np.testing.assert_allclose(
            cols["freq_min"], [w.freq_range[0] for w in plan.windows]
        )
        np.testing.assert_allclose(
            cols["freq_max"], [w.freq_range[1] for w in plan.windows]
        )
        np.testing.assert_array_equal(cols["batch"], [w.batch for w in plan.windows])
        np.testing.assert_array_equal(
            cols["n_free_peaks"], [len(w.free_peak_indices) for w in plan.windows]
        )
        np.testing.assert_array_equal(
            cols["n_fixed_contributors"],
            [len(w.fixed_contributors) for w in plan.windows],
        )

    def test_bounds_only_selection(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_plan_columns(
                h5f["stage4_windows"], columns=["freq_min", "freq_max"]
            )
        assert list(cols) == ["freq_min", "freq_max"]
        assert len(cols["freq_min"]) == 2

    def test_missing_windows_subgroup_raises(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as h5f:
            h5f.create_group("stage4_windows")
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="missing required 'windows'"):
                read_window_plan_columns(h5f["stage4_windows"])

    def test_missing_required_attr_raises(self, plan_file):
        with h5py.File(plan_file, "a") as h5f:
            del h5f["stage4_windows/windows/window_0001"].attrs["batch"]
        with h5py.File(plan_file, "r") as h5f:
            with pytest.raises(ValueError, match="missing required attribute 'batch'"):
                read_window_plan_columns(h5f["stage4_windows"])

    def test_missing_count_dataset_raises(self, plan_file):
        with h5py.File(plan_file, "a") as h5f:
            del h5f["stage4_windows/windows/window_0000/free_peak_indices"]
        with h5py.File(plan_file, "r") as h5f:
            with pytest.raises(ValueError, match="free_peak_indices"):
                read_window_plan_columns(
                    h5f["stage4_windows"], columns=["n_free_peaks"]
                )

    def test_every_declared_column_is_readable(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_plan_columns(h5f["stage4_windows"])
        assert list(cols) == list(WINDOW_PLAN_COLUMN_SPECS)

    def test_scalars(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            scalars = read_window_plan_scalars(h5f["stage4_windows"])
        assert scalars["n_windows"] == 2
        assert scalars["n_dependency_edges"] == 1


class TestReadWindowLongTables:
    """The plan's ragged per-window sets, flattened to long form."""

    def test_free_peaks_match_the_full_loader(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_free_peak_columns(h5f["stage4_windows"])
            plan = load_window_plan_from_hdf5(h5f["stage4_windows"])

        expected = [
            (w.window_id, index) for w in plan.windows for index in w.free_peak_indices
        ]
        assert list(zip(cols["window_id"], cols["peak_index"])) == expected

    def test_contributors_match_the_full_loader(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_contributor_columns(h5f["stage4_windows"])
            plan = load_window_plan_from_hdf5(h5f["stage4_windows"])

        expected = [
            (
                w.window_id,
                c.peak_index,
                c.primary_window_id,
                c.frequency_mhz,
                c.freeze_eligible,
                c.edge_free,
            )
            for w in plan.windows
            for c in w.fixed_contributors
        ]
        assert (
            list(
                zip(
                    cols["window_id"],
                    cols["peak_index"],
                    cols["primary_window_id"],
                    cols["frequency_mhz"],
                    cols["freeze_eligible"],
                    cols["edge_free"],
                )
            )
            == expected
        )

    def test_window_id_is_broadcast_over_each_windows_rows(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_free_peak_columns(h5f["stage4_windows"])
        # window 0 has three free peaks, window 1 has one.
        assert list(cols["window_id"]) == [0, 0, 0, 1]

    def test_selecting_without_window_id_still_reads(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_free_peak_columns(
                h5f["stage4_windows"], columns=["peak_index"]
            )
        assert list(cols) == ["peak_index"]
        assert list(cols["peak_index"]) == [0, 1, 2, 5]

    def test_absent_edge_free_column_reads_as_its_fill(self, plan_file):
        """Legacy plans predate the edge-free attachment; it defaults False."""
        with h5py.File(plan_file, "a") as h5f:
            for name in h5f["stage4_windows/windows"]:
                del h5f[f"stage4_windows/windows/{name}/fixed_edge_free"]
        with h5py.File(plan_file, "r") as h5f:
            cols = read_window_contributor_columns(h5f["stage4_windows"])
        assert not cols["edge_free"].any()

    def test_every_declared_column_is_readable(self, plan_file):
        with h5py.File(plan_file, "r") as h5f:
            group = h5f["stage4_windows"]
            assert list(read_window_free_peak_columns(group)) == list(
                WINDOW_FREE_PEAK_COLUMN_SPECS
            )
            assert list(read_window_contributor_columns(group)) == list(
                WINDOW_CONTRIBUTOR_COLUMN_SPECS
            )


# ---------------------------------------------------------------------------
# Stage 2b decay-time calibration
# ---------------------------------------------------------------------------
class TestReadTauColumns:
    def test_bands_match_the_full_loader(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_band_columns(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])

        bands = sorted(result.band_majorities, key=lambda b: b.freq_lo_mhz)
        assert list(cols["label"]) == [b.label for b in bands]
        np.testing.assert_allclose(cols["tau_maj_us"], [b.tau_maj_us for b in bands])
        np.testing.assert_allclose(
            cols["sigma_tau_us"], [b.sigma_tau_us for b in bands]
        )
        np.testing.assert_allclose(cols["freq_lo_mhz"], [b.freq_lo_mhz for b in bands])
        np.testing.assert_allclose(cols["freq_hi_mhz"], [b.freq_hi_mhz for b in bands])
        # The bare on-disk ``n`` reads back under a name that says what it counts.
        np.testing.assert_array_equal(cols["n_contributors"], [b.n for b in bands])

    def test_bands_are_ordered_by_frequency(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_band_columns(h5f["stage2b_tau_calibration"])
        assert list(cols["freq_lo_mhz"]) == sorted(cols["freq_lo_mhz"])

    def test_thirds_match_the_full_loader(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_third_columns(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])

        thirds = sorted(result.frequency_thirds, key=lambda t: t.freq_lo_mhz)
        assert list(cols["label"]) == [t.label for t in thirds]
        np.testing.assert_allclose(
            cols["median_tau_us"], [t.median_tau_us for t in thirds]
        )
        np.testing.assert_array_equal(cols["n_contributors"], [t.n for t in thirds])

    def test_contributors_match_the_full_loader(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_contributor_columns(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])

        np.testing.assert_array_equal(cols["bin_index"], result.contributor_bin_indices)
        np.testing.assert_allclose(cols["tau_us"], result.contributor_taus_us)
        np.testing.assert_allclose(cols["snr"], result.contributor_snrs)
        np.testing.assert_allclose(cols["freq_mhz"], result.contributor_freqs_mhz)

    def test_contributor_columns_are_singular(self, tau_file):
        """The read surface names one row's field, not the stored array."""
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_contributor_columns(h5f["stage2b_tau_calibration"])
        assert list(cols) == ["bin_index", "freq_mhz", "tau_us", "snr"]

    def test_spurs_match_the_full_loader(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_spur_columns(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])

        clusters = result.spur_clusters
        np.testing.assert_allclose(
            cols["center_freq_mhz"], [c.center_freq_mhz for c in clusters]
        )
        np.testing.assert_array_equal(
            cols["peak_bin_index"], [c.peak_bin_index for c in clusters]
        )
        np.testing.assert_array_equal(cols["n_bins"], [c.n_bins for c in clusters])
        assert list(cols["saturated"]) == [c.saturated for c in clusters]

    def test_absent_saturated_column_reads_as_its_fill(self, tau_file):
        """Legacy files predate the flat/CW-tone flag; it defaults to False."""
        with h5py.File(tau_file, "a") as h5f:
            del h5f["stage2b_tau_calibration/spur_clusters/saturated"]
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_spur_columns(h5f["stage2b_tau_calibration"])
        assert not cols["saturated"].any()

    def test_column_selection_is_honored_in_order(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            cols = read_tau_contributor_columns(
                h5f["stage2b_tau_calibration"], columns=["tau_us", "freq_mhz"]
            )
        assert list(cols) == ["tau_us", "freq_mhz"]

    def test_no_band_majorities_reads_as_an_empty_table(self, tau_file_no_bands):
        with h5py.File(tau_file_no_bands, "r") as h5f:
            cols = read_tau_band_columns(h5f["stage2b_tau_calibration"])
        assert list(cols) == list(TAU_BAND_COLUMN_SPECS)
        assert all(len(v) == 0 for v in cols.values())

    def test_missing_band_majorities_subgroup_raises(self, tau_file):
        with h5py.File(tau_file, "a") as h5f:
            del h5f["stage2b_tau_calibration/band_majorities"]
        with h5py.File(tau_file, "r") as h5f:
            with pytest.raises(ValueError, match="band_majorities"):
                read_tau_band_columns(h5f["stage2b_tau_calibration"])

    def test_missing_contributors_subgroup_raises(self, tau_file):
        with h5py.File(tau_file, "a") as h5f:
            del h5f["stage2b_tau_calibration/contributors"]
        with h5py.File(tau_file, "r") as h5f:
            with pytest.raises(ValueError, match="contributors"):
                read_tau_contributor_columns(h5f["stage2b_tau_calibration"])

    def test_unknown_column_raises(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            with pytest.raises(ValueError, match="unknown column"):
                read_tau_band_columns(
                    h5f["stage2b_tau_calibration"], columns=["tau_us"]
                )

    def test_every_declared_column_is_readable(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            group = h5f["stage2b_tau_calibration"]
            assert list(read_tau_band_columns(group)) == list(TAU_BAND_COLUMN_SPECS)
            assert list(read_tau_third_columns(group)) == list(TAU_THIRD_COLUMN_SPECS)
            assert list(read_tau_contributor_columns(group)) == list(
                TAU_CONTRIBUTOR_COLUMN_SPECS
            )
            assert list(read_tau_spur_columns(group)) == list(TAU_SPUR_COLUMN_SPECS)

    def test_scalars_carry_the_calibration_and_the_shape_vote(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            scalars = read_tau_scalars(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])
        assert scalars["tau_maj_us"] == pytest.approx(result.tau_maj_us)
        assert scalars["sigma_tau_us"] == pytest.approx(result.sigma_tau_us)
        assert scalars["n_contributors"] == result.n_contributors
        assert scalars["preconditions_passed"] is False
        assert scalars["n_bands"] == len(result.band_majorities)
        assert scalars["n_spur_clusters"] == len(result.spur_clusters)
        assert scalars["shape"] == "lorentzian"
        assert scalars["recommended_shape"] == "gaussian"
        assert scalars["shape_vote_gauss"] == pytest.approx(0.66)

    def test_scalars_carry_the_precondition_verdict_and_its_reasons(self, tau_file):
        with h5py.File(tau_file, "r") as h5f:
            scalars = read_tau_scalars(h5f["stage2b_tau_calibration"])
            result = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])
        assert scalars["preconditions_notes"] == list(result.preconditions_notes)
        assert scalars["bimodality_two_component_preferred"] is True
        assert scalars["bimodality_delta_aic"] == pytest.approx(
            result.bimodality.delta_aic
        )
        assert scalars["pearson_r_freq_vs_tau"] == pytest.approx(
            result.pearson_r_freq_vs_tau
        )
        assert scalars["method"] == "sliding_active_window_stft"

    def test_scalars_tolerate_a_group_with_no_shape_vote(self, tau_file):
        with h5py.File(tau_file, "a") as h5f:
            group = h5f["stage2b_tau_calibration"]
            for name in ("shape", "recommended_shape", "shape_vote_rates"):
                del group.attrs[name]
        with h5py.File(tau_file, "r") as h5f:
            scalars = read_tau_scalars(h5f["stage2b_tau_calibration"])
        assert "shape" not in scalars
        assert not any(k.startswith("shape_vote_") for k in scalars)
        assert scalars["tau_maj_us"] > 0


# ---------------------------------------------------------------------------
# Stage 3 peak list
# ---------------------------------------------------------------------------
class TestReadPeakColumns:
    def test_matches_the_full_loader(self, peaks_file):
        with h5py.File(peaks_file, "r") as h5f:
            cols = read_peak_columns(h5f["stage3_peaks"])
            peaks = load_peaks_from_hdf5(h5f["stage3_peaks"])

        np.testing.assert_allclose(cols["frequency"], [p.frequency for p in peaks])
        np.testing.assert_allclose(cols["intensity"], [p.intensity for p in peaks])
        np.testing.assert_array_equal(cols["index"], [p.index for p in peaks])
        np.testing.assert_allclose(cols["snr"], [p.snr for p in peaks])
        assert list(cols["classification"]) == [p.classification.value for p in peaks]
        assert list(cols["detection_pass"]) == [
            p.properties["detection_pass"] for p in peaks
        ]

    def test_promoted_matches_the_loader_derivation(self, peaks_file):
        with h5py.File(peaks_file, "r") as h5f:
            cols = read_peak_columns(h5f["stage3_peaks"], columns=["promoted"])
            peaks = load_peaks_from_hdf5(h5f["stage3_peaks"])
        assert list(cols["promoted"]) == [p.properties["promoted"] for p in peaks]
        assert list(cols["promoted"]) == [True, False]

    def test_promoted_is_all_false_without_a_cutoff(self, tmp_path):
        path = tmp_path / "nocutoff.h5"
        with h5py.File(path, "w") as h5f:
            save_peaks_to_hdf5(
                _sample_peaks(), h5f.create_group("stage3_peaks"), parameters={}
            )
        with h5py.File(path, "r") as h5f:
            cols = read_peak_columns(h5f["stage3_peaks"], columns=["promoted"])
        assert not cols["promoted"].any()

    def test_missing_required_column_raises(self, peaks_file):
        with h5py.File(peaks_file, "a") as h5f:
            del h5f["stage3_peaks/frequency"]
        with h5py.File(peaks_file, "r") as h5f:
            with pytest.raises(ValueError, match="missing required column 'frequency'"):
                read_peak_columns(h5f["stage3_peaks"])

    def test_every_declared_column_is_readable(self, peaks_file):
        with h5py.File(peaks_file, "r") as h5f:
            cols = read_peak_columns(h5f["stage3_peaks"])
        assert list(cols) == list(PEAK_COLUMN_SPECS)

    def test_scalars(self, peaks_file):
        with h5py.File(peaks_file, "r") as h5f:
            scalars = read_peak_scalars(h5f["stage3_peaks"])
        assert scalars["n_peaks"] == 2
        assert scalars["promotion_min_snr"] == pytest.approx(5.0)
        assert scalars["internal_min_snr"] == pytest.approx(3.0)
