"""
Unit tests for the read-only data tap (``_internal.read_impl``).

Covers the table registry and its name normalization, the missing-stage error
(actionable, naming the command to run), the cheap top-level scalars, and the
CSV / TSV / JSON renderers -- including the round-trip-exact float convention
that distinguishes this surface from the presentation-formatted
``report table``.
"""

from __future__ import annotations

import csv
import io
import json

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.read_impl import (
    READ_TABLES,
    format_metadata_impl,
    format_table_impl,
    normalize_table_name,
    read_metadata_impl,
    read_table_impl,
    read_tables_impl,
    write_text_impl,
)
from ftmwpipeline._internal.shared_utils import active_acquisition_us
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    FitWindow,
    Peak,
    PeakClassification,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.file_manager import PipelineCompatibilityError
from ftmwpipeline.fitting.tau_calibration import (
    BandMajority,
    GMMBimodality,
    TauCalibrationResult,
)
from ftmwpipeline.io.fitting_serialization import save_spectrum_fit_to_hdf5
from ftmwpipeline.io.peak_serialization import save_peaks_to_hdf5
from ftmwpipeline.io.tau_calibration_serialization import save_tau_calibration_to_hdf5
from ftmwpipeline.io.window_serialization import save_window_plan_to_hdf5


def _fit() -> SpectrumFit:
    peak = FittedPeak(
        peak_id=0,
        frequency_mhz=36100.012345678,
        amplitude=0.5,
        decay_rate=0.2,
        phase=0.3,
        frequency_error=1e-4,
        amplitude_error=0.01,
        decay_rate_error=2e-3,
        phase_error=0.02,
        snr=None,  # -> NaN on disk; exercises the non-finite JSON path
        chi_squared=15.4,
        window_id=0,
    )
    window = FittingResult(
        success=True,
        fitted_spectrum=None,
        cost=1.0,
        iterations=2,
        aic=10.0,
        reduced_chi2=1.1,
        window=None,
        window_id=0,
        shape="lorentzian",
    )
    window.fitted_peaks = [peak]
    window.shared_parameters["tau_us"] = {
        "value": 5.0,
        "error": 0.05,
        "fitted": True,
        "peak_ids": [0],
    }
    return SpectrumFit(
        window_fits=[window],
        fitted_peaks=[peak],
        parameters={"acquisition_us": 12.73},
    )


def _tau(shape: str) -> TauCalibrationResult:
    """A minimal decay-time calibration; ``shape`` distinguishes the two twins."""
    n = 4
    return TauCalibrationResult(
        tau_maj_us=6.0 if shape == "lorentzian" else 6.4,
        sigma_tau_us=1.5,
        n_contributors=n,
        n_spur_bins=0,
        spur_clusters=(),
        bimodality=GMMBimodality(
            n=n,
            mu1=6.0,
            sigma1=1.0,
            mu_a=5.0,
            sigma_a=0.8,
            mu_b=7.5,
            sigma_b=1.2,
            pi_a=0.6,
            aic1=1.0,
            aic2=0.5,
            delta_aic=0.5,
            two_component_preferred=False,
            dominant_weight=0.6,
        ),
        pearson_r_log_snr_vs_tau=-0.3,
        pearson_r_freq_vs_tau=-0.3,
        frequency_thirds=(),
        contributor_bin_indices=np.arange(n, dtype=np.int64),
        contributor_taus_us=np.linspace(5.0, 7.0, n),
        contributor_snrs=np.linspace(10.0, 40.0, n),
        contributor_freqs_mhz=np.linspace(27000.0, 39000.0, n),
        band_majorities=(
            BandMajority("low", 26500.0, 33000.0, 2, 7.0, 1.0),
            BandMajority("high", 33000.0, 40000.0, 2, 5.5, 0.9),
        ),
        n_seg=10,
        t_sigma=5.0,
        tau_max_us=60.0,
        rss_gate_factor=5.0,
        sample_dt_us=0.02,
        start_us=2.27,
        end_us=15.0,
        probe_freq_mhz=40960.0,
        sideband="lower",
        trim_lo_mhz=26500.0,
        trim_hi_mhz=40000.0,
        sigma_x_full=4.7e-7,
        sigma_frame=1.5e-7,
        snr_weighted=True,
        preconditions_passed=True,
        preconditions_notes=("ok", "ok", "ok"),
    )


@pytest.fixture
def ftmw_file(tmp_path):
    """A minimal but structurally faithful pipeline file."""
    path = tmp_path / "exp.ftmw"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["ftmw_format_version"] = "1.0"
        h5f.attrs["created_with_ftmwpipeline"] = "0.1.0b3"
        stages = h5f.create_group("pipeline_stages")
        stages.attrs["completed_stages"] = json.dumps(
            ["stage0_fid_data", "stage3_peaks", "stage4_windows", "stage5_fitting"]
        )
        source = h5f.create_group("source_metadata")
        source.attrs["source_path"] = "/data/2638"
        source.attrs["format_name"] = "blackchirp"
        source.attrs["import_timestamp"] = "2026-07-21T16:21:31"
        source.attrs["source_hash"] = "fc6e7187cad130ad"
        acquisition = h5f.create_group("stage0_fid_data/acquisition")
        acquisition.attrs["n_points"] = 750000
        acquisition.attrs["duration_us"] = 15.0
        acquisition.attrs["probe_freq_mhz"] = 40960.0
        acquisition.attrs["sideband"] = "lower"
        h5f["stage0_fid_data"].attrs["recommended_start_detection"] = json.dumps(
            {
                "guard_margin_us": 0.67,
                "chirp_end_us": 1.6,
                "chirp_detected": True,
                "floor": 4240.5,
                "plateau": 3430286.5,
                "band_min_mhz": "__None__",
                "resolved_band_min_mhz": "__None__",
                "resolved_band_max_mhz": "__None__",
            }
        )
        ft = h5f.create_group("processing_parameters/ft_processing")
        ft.attrs["start_us"] = 2.27
        ft.attrs["end_us"] = 15.0
        ft.attrs["trim_min_mhz"] = 26500.0
        ft.attrs["trim_max_mhz"] = 40000.0
        ft.attrs["units_power"] = 6

        for group_name, shape in (
            ("stage2b_tau_calibration", "lorentzian"),
            ("stage2b_tau_G_calibration", "gaussian"),
        ):
            group = h5f.create_group(group_name)
            save_tau_calibration_to_hdf5(_tau(shape), group)
            group.attrs["shape"] = shape
            group.attrs["creation_time"] = "2026-08-10T00:00:00"

        save_peaks_to_hdf5(
            [
                Peak(
                    frequency=26501.5,
                    intensity=1200.0,
                    index=41,
                    snr=25.0,
                    noise_std_local=48.0,
                    classification=PeakClassification.STRONG,
                    detection_pass="primary",
                )
            ],
            h5f.create_group("stage3_peaks"),
            parameters={"promotion_min_snr": 5.0},
        )
        save_window_plan_to_hdf5(
            WindowPlan(
                windows=[
                    FitWindow(
                        window_id=0,
                        freq_range=(26500.0, 26520.0),
                        free_peak_indices=[0],
                        batch=0,
                    )
                ],
                dependency_edges=[],
                topological_order=[0],
            ),
            h5f.create_group("stage4_windows"),
        )
        save_spectrum_fit_to_hdf5(_fit(), h5f.create_group("stage5_fitting"))
    return path


class TestTableRegistry:
    @pytest.mark.parametrize("table", READ_TABLES)
    def test_every_registered_table_reads(self, ftmw_file, table):
        cols = read_table_impl(ftmw_file, table)
        assert cols
        assert len({len(v) for v in cols.values()}) == 1

    def test_hyphens_and_underscores_are_the_same_table(self, ftmw_file):
        by_hyphen = read_table_impl(ftmw_file, "fit-peaks")
        by_underscore = read_table_impl(ftmw_file, "fit_peaks")
        assert list(by_hyphen) == list(by_underscore)

    def test_name_normalization_is_case_insensitive(self):
        assert normalize_table_name("  FIT-Peaks ") == "fit_peaks"

    def test_unknown_table_names_the_valid_ones(self, ftmw_file):
        with pytest.raises(ValueError, match="unknown table 'bogus'"):
            read_table_impl(ftmw_file, "bogus")

    def test_missing_stage_error_names_the_command_to_run(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage5_fitting"]
        with pytest.raises(ValueError, match=r"Run fit_peaks\(\) / 'fit run' first"):
            read_table_impl(ftmw_file, "fit_peaks")

    def test_missing_file_error_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data import"):
            read_table_impl(tmp_path / "absent.ftmw", "peaks")


class TestCompatibilityGate:
    """A future-format file must be refused, not misread."""

    def _stamp(self, path, version: str) -> None:
        with h5py.File(path, "a") as h5f:
            h5f.attrs["ftmw_format_version"] = version

    def test_future_major_version_is_refused(self, ftmw_file):
        self._stamp(ftmw_file, "99.0")
        for call in (
            lambda: read_table_impl(ftmw_file, "peaks"),
            lambda: read_tables_impl(ftmw_file),
            lambda: read_metadata_impl(ftmw_file),
        ):
            with pytest.raises(PipelineCompatibilityError):
                call()

    def test_future_minor_version_still_reads(self, ftmw_file):
        """A newer MINOR may carry fields this version ignores -- readable."""
        self._stamp(ftmw_file, "1.99")
        assert read_table_impl(ftmw_file, "peaks")["frequency"].size == 1

    def test_the_gate_does_not_leak_an_open_file_handle(self, ftmw_file):
        self._stamp(ftmw_file, "99.0")
        with pytest.raises(PipelineCompatibilityError):
            read_table_impl(ftmw_file, "peaks")
        # The refusal must close its handle, or this exclusive open would fail.
        with h5py.File(ftmw_file, "a") as h5f:
            assert h5f is not None


class TestReadTables:
    def test_lists_every_table_with_its_columns(self, ftmw_file):
        listing = read_tables_impl(ftmw_file)
        assert list(listing) == list(READ_TABLES)
        assert listing["fit_peaks"]["available"] is True
        assert listing["fit_peaks"]["n_rows"] == 1
        assert "frequency_mhz" in listing["fit_peaks"]["columns"]
        assert listing["windows"]["group"] == "stage4_windows"

    def test_marks_tables_whose_stage_has_not_run(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage5_fitting"]
        listing = read_tables_impl(ftmw_file)
        assert listing["fit_peaks"]["available"] is False
        assert listing["fit_peaks"]["n_rows"] is None
        # The column roster is still reported, so a caller can plan a read.
        assert listing["fit_peaks"]["columns"]
        assert listing["peaks"]["available"] is True


class TestReadMetadata:
    def test_reports_dotted_scalars(self, ftmw_file):
        meta = read_metadata_impl(ftmw_file)
        assert meta["file.format_version"] == "1.0"
        assert meta["file.created_with"] == "0.1.0b3"
        assert meta["source.format_name"] == "blackchirp"
        assert meta["fid.n_points"] == 750000
        assert meta["stage3.n_peaks"] == 1
        assert meta["stage4.n_windows"] == 1
        assert meta["stage5.n_fitted_peaks"] == 1

    def test_acquisition_us_is_reachable_without_loading_the_fit(self, ftmw_file):
        assert read_metadata_impl(ftmw_file)["stage5.acquisition_us"] == pytest.approx(
            12.73
        )

    def test_completed_stages_is_a_sorted_list(self, ftmw_file):
        stages = read_metadata_impl(ftmw_file)["file.completed_stages"]
        assert stages == sorted(stages)
        assert "stage5_fitting" in stages

    def test_absent_stage_contributes_no_keys(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage5_fitting"]
        meta = read_metadata_impl(ftmw_file)
        assert not any(key.startswith("stage5.") for key in meta)
        assert meta["stage3.n_peaks"] == 1

    def test_values_are_plain_python_scalars(self, ftmw_file):
        for key, value in read_metadata_impl(ftmw_file).items():
            assert not isinstance(value, np.generic), key

    def test_ft_window_carries_the_derived_active_length(self, ftmw_file):
        meta = read_metadata_impl(ftmw_file)
        assert meta["ft.start_us"] == pytest.approx(2.27)
        assert meta["ft.end_us"] == pytest.approx(15.0)
        assert meta["ft.trim_min_mhz"] == pytest.approx(26500.0)
        assert meta["ft.acquisition_us"] == pytest.approx(12.73)

    def test_ft_acquisition_matches_the_shared_helper(self, ftmw_file):
        """The read tap must not derive T its own way."""
        meta = read_metadata_impl(ftmw_file)
        assert meta["ft.acquisition_us"] == active_acquisition_us(
            meta["fid.duration_us"], meta["ft.start_us"], meta["ft.end_us"]
        )

    def test_ft_acquisition_matches_what_the_fit_recorded(self, ftmw_file):
        meta = read_metadata_impl(ftmw_file)
        assert meta["ft.acquisition_us"] == pytest.approx(meta["stage5.acquisition_us"])

    def test_stage1_is_an_exact_synonym_for_the_ft_section(self, ftmw_file):
        """``ft`` and ``stage1`` name the same stage everywhere else.

        The CLI declares the two object names interchangeable and
        ``settings show`` spells these same persisted knobs ``stage1.``, so a
        consumer must not have to know which surface it came from.
        """
        meta = read_metadata_impl(ftmw_file)
        ft_keys = {k.split(".", 1)[1] for k in meta if k.startswith("ft.")}
        stage1_keys = {k.split(".", 1)[1] for k in meta if k.startswith("stage1.")}
        assert ft_keys == stage1_keys
        assert ft_keys  # the fixture has run Stage 1, or this proves nothing
        for name in ft_keys:
            assert meta[f"stage1.{name}"] == meta[f"ft.{name}"]
        # The name the settings view uses for the knob that prompted this.
        assert meta["stage1.units_power"] == meta["ft.units_power"]

    def test_ft_section_absent_when_stage1_has_not_run(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["processing_parameters/ft_processing"]
        meta = read_metadata_impl(ftmw_file)
        assert not any(key.startswith("ft.") for key in meta)
        # The synonym must vanish with it, not linger as a half-present section.
        assert not any(key.startswith("stage1.") for key in meta)

    def test_start_record_reports_the_sweep_outcome(self, ftmw_file):
        meta = read_metadata_impl(ftmw_file)
        assert meta["start.chirp_end_us"] == pytest.approx(1.6)
        assert meta["start.chirp_detected"] is True
        assert meta["start.guard_margin_us"] == pytest.approx(0.67)
        # The JSON None sentinel must decode to None, not the literal string.
        assert meta["start.resolved_band_min_mhz"] is None

    def test_start_section_absent_when_the_sweep_has_not_run(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage0_fid_data"].attrs["recommended_start_detection"]
        meta = read_metadata_impl(ftmw_file)
        assert not any(key.startswith("start.") for key in meta)

    def test_both_tau_twins_are_reported_separately(self, ftmw_file):
        meta = read_metadata_impl(ftmw_file)
        assert meta["tau.tau_maj_us"] == pytest.approx(6.0)
        assert meta["tau_g.tau_maj_us"] == pytest.approx(6.4)
        assert meta["tau.shape"] == "lorentzian"
        assert meta["tau_g.shape"] == "gaussian"
        assert meta["tau.n_bands"] == 2

    def test_tau_g_section_absent_without_the_gaussian_twin(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage2b_tau_G_calibration"]
        meta = read_metadata_impl(ftmw_file)
        assert not any(key.startswith("tau_g.") for key in meta)
        assert meta["tau.tau_maj_us"] == pytest.approx(6.0)


class TestTableCoverage:
    """Every persisted artifact the surface claims must actually be reachable."""

    def test_the_registry_covers_each_stage_group(self, ftmw_file):
        groups = {entry["group"] for entry in read_tables_impl(ftmw_file).values()}
        assert groups == {
            "stage2b_tau_calibration",
            "stage2b_tau_G_calibration",
            "stage3_peaks",
            "stage4_windows",
            "stage5_fitting",
        }

    def test_no_two_tables_share_a_column_roster_by_accident(self, ftmw_file):
        """A copy-paste in the registry would point two names at one spec."""
        listing = read_tables_impl(ftmw_file)
        for name, entry in listing.items():
            twin = name.replace("tau_g_", "tau_", 1) if "tau_g_" in name else None
            same = [
                other
                for other, e in listing.items()
                if other != name
                and e["columns"] == entry["columns"]
                and e["group"] == entry["group"]
            ]
            assert not same, f"{name} duplicates {same}"
            if twin is not None:
                # The twins deliberately share a roster, in different groups.
                assert listing[twin]["columns"] == entry["columns"]
                assert listing[twin]["group"] != entry["group"]

    def test_tables_with_no_recorded_count_are_still_available(self, ftmw_file):
        listing = read_tables_impl(ftmw_file)
        assert listing["fit_audit"]["available"] is True
        assert listing["fit_audit"]["n_rows"] is None
        assert read_table_impl(ftmw_file, "fit_audit")["window_id"].size >= 0


class TestTauTables:
    def test_the_two_twins_are_distinct_tables(self, ftmw_file):
        primary = read_table_impl(ftmw_file, "tau_bands")
        gaussian = read_table_impl(ftmw_file, "tau_g_bands")
        assert list(primary) == list(gaussian)
        # Same synthetic bands in both twins; what differs is the group read.
        listing = read_tables_impl(ftmw_file)
        assert listing["tau_bands"]["group"] == "stage2b_tau_calibration"
        assert listing["tau_g_bands"]["group"] == "stage2b_tau_G_calibration"

    def test_row_counts_come_from_the_right_subgroup_attribute(self, ftmw_file):
        listing = read_tables_impl(ftmw_file)
        assert listing["tau_bands"]["n_rows"] == 2
        assert listing["tau_contributors"]["n_rows"] == 4

    def test_missing_gaussian_twin_is_reported_as_unavailable(self, ftmw_file):
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f["stage2b_tau_G_calibration"]
        listing = read_tables_impl(ftmw_file)
        assert listing["tau_g_bands"]["available"] is False
        assert listing["tau_bands"]["available"] is True
        with pytest.raises(ValueError, match="tau run --gaussian"):
            read_table_impl(ftmw_file, "tau_g_bands")


class TestFormatting:
    def test_csv_has_a_header_and_one_row_per_record(self, ftmw_file):
        table = read_table_impl(ftmw_file, "windows")
        rows = list(csv.reader(io.StringIO(format_table_impl(table, "csv"))))
        assert rows[0] == list(table)
        assert len(rows) == 1 + len(table["window_id"])

    def test_csv_columns_follow_the_requested_order(self, ftmw_file):
        table = read_table_impl(
            ftmw_file, "fit_peaks", columns=["shape", "frequency_mhz"]
        )
        header = format_table_impl(table, "csv").splitlines()[0]
        assert header == "shape,frequency_mhz"

    def test_floats_round_trip_exactly(self, ftmw_file):
        table = read_table_impl(ftmw_file, "fit_peaks", columns=["frequency_mhz"])
        text = format_table_impl(table, "csv")
        value = float(text.splitlines()[1])
        assert value == table["frequency_mhz"][0]

    def test_tsv_uses_tabs(self, ftmw_file):
        table = read_table_impl(ftmw_file, "windows", columns=["window_id", "batch"])
        assert format_table_impl(table, "tsv").splitlines()[0] == "window_id\tbatch"

    def test_booleans_render_as_words(self, ftmw_file):
        table = read_table_impl(ftmw_file, "fit_windows", columns=["success"])
        assert format_table_impl(table, "csv").splitlines()[1] == "true"

    def test_json_is_a_list_of_objects_with_null_for_non_finite(self, ftmw_file):
        table = read_table_impl(ftmw_file, "fit_peaks", columns=["peak_id", "snr"])
        records = json.loads(format_table_impl(table, "json"))
        assert records == [{"peak_id": 0, "snr": None}]

    def test_unknown_format_raises(self, ftmw_file):
        table = read_table_impl(ftmw_file, "windows")
        with pytest.raises(ValueError, match="unknown format"):
            format_table_impl(table, "yaml")

    def test_metadata_csv_is_key_value_rows(self, ftmw_file):
        text = format_metadata_impl(read_metadata_impl(ftmw_file), "csv")
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0] == ["key", "value"]
        assert dict(rows[1:])["source.format_name"] == "blackchirp"

    def test_metadata_csv_joins_list_values(self, ftmw_file):
        text = format_metadata_impl(read_metadata_impl(ftmw_file), "csv")
        stages = dict(list(csv.reader(io.StringIO(text)))[1:])["file.completed_stages"]
        assert "stage5_fitting" in stages.split(";")

    def test_metadata_json_is_an_object(self, ftmw_file):
        blob = json.loads(format_metadata_impl(read_metadata_impl(ftmw_file), "json"))
        assert blob["stage5.acquisition_us"] == pytest.approx(12.73)


class TestWriteText:
    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "out.csv"
        written = write_text_impl("a,b\n1,2\n", target)
        assert written == target
        assert target.read_text() == "a,b\n1,2\n"
