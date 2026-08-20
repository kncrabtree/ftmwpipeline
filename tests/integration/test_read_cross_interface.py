"""
Cross-interface consistency for the read-only data tap.

The read surface is only useful if it is a faithful, cheap substitute for the
full loaders, so this module asserts two things on real 2638 data:

* **Parity with the full loaders.** The columns ``read_table`` returns are
  numerically identical to what ``load_peaks`` / ``load_windows`` / ``load_fit``
  reconstruct, row for row and in the same order.
* **Parity across interfaces.** The Pipeline class, the functional API, and the
  ``read`` CLI (as a subprocess, parsed back from its CSV) agree exactly.

The mandatory cross-interface category in ``dev-docs/TESTING_STRATEGY.md`` is
what this satisfies for the read surface.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.read_impl import READ_TABLES

from ._stage2b_helpers import skip_auto_recommend_settings


def _run_read(args: list) -> str:
    """Run an ``ftmwpipeline read`` command and return its stdout."""
    result = subprocess.run(
        ["ftmwpipeline", "read"] + args,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CLI command failed: ftmwpipeline read {' '.join(args)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def _parse_csv(text: str) -> dict:
    """Parse a ``read table`` CSV dump back into ``{column: [cells]}``."""
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    return {name: [row[i] for row in body] for i, name in enumerate(header)}


def _available_tables(path: str) -> list:
    """The tables this file actually carries."""
    return [
        name for name, entry in ftmw.read_tables(path).items() if entry["available"]
    ]


@pytest.fixture(scope="module")
def tau_calibrated_2638(baseline_2638_stage2, tmp_path_factory):
    """A file carrying both decay-time calibrations, built once per module.

    The Stage 5 baseline does not run Stage 2b, so the ``tau_*`` tables need
    their own file. Both twins are built here because they live in separate
    groups and the read surface addresses them as separate tables --
    ``tau_g_*`` would otherwise never be exercised on real data.
    ``auto_recommend`` is off: the 3-way shape classifier costs ~50 s per call
    on this fixture and nothing here depends on its verdict.
    """
    tmp = tmp_path_factory.mktemp("read_tau")
    path = tmp / "read_tau_2638.ftmw"
    shutil.copy(baseline_2638_stage2, path)
    for shape in ("lorentzian", "gaussian"):
        ftmw.calibrate_tau(
            str(path), shape=shape, settings=skip_auto_recommend_settings()
        )
    return path


@pytest.fixture
def table_source(baseline_2638_stage5_small, tau_calibrated_2638):
    """Resolve a table name to the fixture file that carries it."""

    def pick(table: str) -> str:
        if table.startswith("tau_"):
            return str(tau_calibrated_2638)
        return str(baseline_2638_stage5_small)

    return pick


@pytest.mark.integration
@pytest.mark.cross_interface
class TestReadMatchesFullLoaders:
    """The cheap read must reproduce the full loaders exactly."""

    def test_fit_peaks_match_load_fit(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "fit_peaks")
        peaks = ftmw.load_fit(path).fitted_peaks

        assert len(cols["detection_index"]) == len(peaks)
        np.testing.assert_array_equal(
            cols["detection_index"], [p.detection_index for p in peaks]
        )
        np.testing.assert_array_equal(
            cols["frequency_mhz"], [p.frequency_mhz for p in peaks]
        )
        np.testing.assert_array_equal(cols["amplitude"], [p.amplitude for p in peaks])
        np.testing.assert_array_equal(cols["decay_rate"], [p.decay_rate for p in peaks])
        np.testing.assert_array_equal(cols["snr"], [p.snr for p in peaks])
        np.testing.assert_array_equal(cols["window_id"], [p.window_id for p in peaks])
        assert list(cols["origin"]) == [p.origin for p in peaks]

    def test_fit_peak_shape_matches_its_window(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "fit_peaks", columns=["window_id", "shape"])
        fit = ftmw.load_fit(path)
        shape_of = {w.window_id: w.shape for w in fit.window_fits}
        assert [shape_of[w] for w in cols["window_id"]] == list(cols["shape"])

    def test_fit_windows_match_load_fit(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "fit_windows")
        windows = ftmw.load_fit(path).window_fits

        np.testing.assert_array_equal(cols["window_id"], [w.window_id for w in windows])
        np.testing.assert_array_equal(cols["success"], [w.success for w in windows])
        np.testing.assert_array_equal(cols["aic"], [w.aic for w in windows])
        np.testing.assert_array_equal(
            cols["reduced_chi2"], [w.reduced_chi2 for w in windows]
        )
        np.testing.assert_array_equal(
            cols["tau_us"],
            [w.shared_parameters["tau_us"]["value"] for w in windows],
        )
        np.testing.assert_array_equal(
            cols["n_peaks"], [len(w.fitted_peaks) for w in windows]
        )

    def test_window_bounds_match_load_windows(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "windows")
        windows = ftmw.load_windows(path).windows

        np.testing.assert_array_equal(cols["window_id"], [w.window_id for w in windows])
        np.testing.assert_array_equal(
            cols["freq_min"], [w.freq_range[0] for w in windows]
        )
        np.testing.assert_array_equal(
            cols["freq_max"], [w.freq_range[1] for w in windows]
        )
        np.testing.assert_array_equal(cols["batch"], [w.batch for w in windows])

    def test_window_free_peaks_match_load_windows(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "window_free_peaks")
        expected = [
            (w.window_id, index)
            for w in ftmw.load_windows(path).windows
            for index in w.free_peak_indices
        ]
        assert list(zip(cols["window_id"], cols["peak_index"])) == expected

    def test_window_contributors_match_load_windows(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "window_contributors")
        expected = [
            (w.window_id, c.peak_index, c.frequency_mhz)
            for w in ftmw.load_windows(path).windows
            for c in w.fixed_contributors
        ]
        assert (
            list(zip(cols["window_id"], cols["peak_index"], cols["frequency_mhz"]))
            == expected
        )

    def test_fit_audit_matches_load_fit(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "fit_audit")
        expected = [
            (wf.window_id, i, step.decision)
            for wf in ftmw.load_fit(path).window_fits
            for i, step in enumerate(wf.audit_trail)
        ]
        assert expected, "fixture must record some audit steps to be meaningful"
        assert (
            list(zip(cols["window_id"], cols["step_index"], cols["decision"]))
            == expected
        )

    def test_fit_plan_histories_match_load_fit(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        fit = ftmw.load_fit(path)
        for table, events, key in (
            ("fit_thaw", fit.thaw_history, "dependent_window_id"),
            ("fit_replans", fit.replan_history, "triggering_window_id"),
            ("fit_rescues", fit.rescue_history, "window_id"),
        ):
            cols = ftmw.read_table(path, table)
            assert len(cols[key]) == len(events), table
            np.testing.assert_array_equal(
                cols[key], [getattr(e, key) for e in events], err_msg=table
            )

    def test_peaks_match_load_peaks(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        cols = ftmw.read_table(path, "peaks")
        peaks = ftmw.load_peaks(path)

        np.testing.assert_array_equal(cols["frequency"], [p.frequency for p in peaks])
        np.testing.assert_array_equal(cols["snr"], [p.snr for p in peaks])
        assert list(cols["promoted"]) == [
            bool(p.properties.get("promoted")) for p in peaks
        ]

    def test_acquisition_us_matches_the_fits_parameters(
        self, baseline_2638_stage5_small
    ):
        path = str(baseline_2638_stage5_small)
        meta = ftmw.read_metadata(path)
        assert meta["stage5.acquisition_us"] == pytest.approx(
            ftmw.load_fit(path).parameters["acquisition_us"]
        )

    def test_reading_nothing_is_not_recomputing(self, baseline_2638_stage5_small):
        """The read surface must not write to the file it reads."""
        path = baseline_2638_stage5_small
        before = path.stat().st_mtime_ns
        for table in _available_tables(str(path)):
            ftmw.read_table(str(path), table)
        ftmw.read_metadata(str(path))
        assert path.stat().st_mtime_ns == before

    @pytest.mark.parametrize(
        "table,shape", [("tau_bands", "lorentzian"), ("tau_g_bands", "gaussian")]
    )
    def test_tau_bands_match_load_tau_calibration(
        self, tau_calibrated_2638, table, shape
    ):
        path = str(tau_calibrated_2638)
        cols = ftmw.read_table(path, table)
        bands = sorted(
            ftmw.load_tau_calibration(path, shape=shape).band_majorities,
            key=lambda b: b.freq_lo_mhz,
        )
        assert list(cols["label"]) == [b.label for b in bands]
        np.testing.assert_array_equal(cols["tau_maj_us"], [b.tau_maj_us for b in bands])
        np.testing.assert_array_equal(
            cols["sigma_tau_us"], [b.sigma_tau_us for b in bands]
        )
        np.testing.assert_array_equal(
            cols["freq_lo_mhz"], [b.freq_lo_mhz for b in bands]
        )
        np.testing.assert_array_equal(
            cols["freq_hi_mhz"], [b.freq_hi_mhz for b in bands]
        )

    @pytest.mark.parametrize(
        "table,shape",
        [("tau_contributors", "lorentzian"), ("tau_g_contributors", "gaussian")],
    )
    def test_tau_contributors_match_load_tau_calibration(
        self, tau_calibrated_2638, table, shape
    ):
        path = str(tau_calibrated_2638)
        cols = ftmw.read_table(path, table)
        result = ftmw.load_tau_calibration(path, shape=shape)
        np.testing.assert_array_equal(cols["bin_index"], result.contributor_bin_indices)
        np.testing.assert_array_equal(cols["tau_us"], result.contributor_taus_us)
        np.testing.assert_array_equal(cols["freq_mhz"], result.contributor_freqs_mhz)
        np.testing.assert_array_equal(cols["snr"], result.contributor_snrs)

    @pytest.mark.parametrize(
        "prefix,shape", [("tau", "lorentzian"), ("tau_g", "gaussian")]
    )
    def test_tau_scalars_match_load_tau_calibration(
        self, tau_calibrated_2638, prefix, shape
    ):
        meta = ftmw.read_metadata(str(tau_calibrated_2638))
        result = ftmw.load_tau_calibration(str(tau_calibrated_2638), shape=shape)
        assert meta[f"{prefix}.tau_maj_us"] == pytest.approx(result.tau_maj_us)
        assert meta[f"{prefix}.sigma_tau_us"] == pytest.approx(result.sigma_tau_us)
        assert meta[f"{prefix}.n_contributors"] == result.n_contributors

    def test_the_two_tau_twins_are_read_from_different_groups(
        self, tau_calibrated_2638
    ):
        """A stale-group bug would make the twins identical; they are not."""
        meta = ftmw.read_metadata(str(tau_calibrated_2638))
        assert meta["tau.tau_maj_us"] != meta["tau_g.tau_maj_us"]
        lorentzian = ftmw.read_table(str(tau_calibrated_2638), "tau_contributors")
        gaussian = ftmw.read_table(str(tau_calibrated_2638), "tau_g_contributors")
        assert not np.array_equal(lorentzian["tau_us"], gaussian["tau_us"])


@pytest.mark.integration
@pytest.mark.cross_interface
class TestReadAcrossInterfaces:
    """Pipeline class, functional API, and CLI must agree exactly."""

    @pytest.mark.parametrize("table", READ_TABLES)
    def test_identical_columns(self, table_source, table):
        path = table_source(table)
        assert table in _available_tables(path)
        via_api = ftmw.read_table(path, table)
        via_pipeline = Pipeline.open(path).read_table(table)
        via_cli = _parse_csv(_run_read(["table", path, table]))

        assert list(via_api) == list(via_pipeline) == list(via_cli)
        for name, expected in via_api.items():
            np.testing.assert_array_equal(via_pipeline[name], expected)
            if expected.dtype.kind == "f":
                # repr() round-trips a float exactly, so this is an equality
                # assertion, not a tolerance one.
                np.testing.assert_array_equal(
                    np.array([float(c) for c in via_cli[name]]), expected
                )
            elif expected.dtype.kind == "b":
                assert [c == "true" for c in via_cli[name]] == list(expected)
            elif expected.dtype.kind in "iu":
                assert [int(c) for c in via_cli[name]] == list(expected)
            else:
                assert via_cli[name] == list(expected)

    def test_identical_column_selection(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        columns = ["frequency_mhz", "decay_rate", "shape"]
        via_api = ftmw.read_table(path, "fit_peaks", columns=columns)
        via_pipeline = Pipeline.open(path).read_table("fit_peaks", columns)
        via_cli = _parse_csv(
            _run_read(["table", path, "fit-peaks", "--columns", ",".join(columns)])
        )

        assert list(via_api) == columns
        assert list(via_pipeline) == columns
        assert list(via_cli) == columns
        np.testing.assert_array_equal(
            via_pipeline["frequency_mhz"], via_api["frequency_mhz"]
        )
        np.testing.assert_array_equal(
            np.array([float(c) for c in via_cli["frequency_mhz"]]),
            via_api["frequency_mhz"],
        )

    def test_selection_is_a_subset_of_the_full_read(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        everything = ftmw.read_table(path, "fit_peaks")
        subset = ftmw.read_table(path, "fit_peaks", columns=["decay_rate"])
        np.testing.assert_array_equal(subset["decay_rate"], everything["decay_rate"])

    def test_identical_metadata(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        via_api = ftmw.read_metadata(path)
        via_pipeline = Pipeline.open(path).read_metadata()
        via_cli = json.loads(_run_read(["meta", path, "--format", "json"]))

        assert via_api == via_pipeline
        assert via_cli["stage5.n_fitted_peaks"] == via_api["stage5.n_fitted_peaks"]
        for key in ("stage5.acquisition_us", "ft.acquisition_us"):
            assert via_cli[key] == pytest.approx(via_api[key])

    def test_identical_tau_metadata(self, tau_calibrated_2638):
        path = str(tau_calibrated_2638)
        via_api = ftmw.read_metadata(path)
        via_cli = json.loads(_run_read(["meta", path, "--format", "json"]))
        assert via_api == Pipeline.open(path).read_metadata()
        for key in ("tau.tau_maj_us", "tau_g.tau_maj_us", "tau.sigma_tau_us"):
            assert via_cli[key] == pytest.approx(via_api[key])

    def test_ft_and_fit_agree_on_the_active_record_length(
        self, baseline_2638_stage5_small
    ):
        """``ft.acquisition_us`` is the Stage 1 value the fit later recorded."""
        meta = ftmw.read_metadata(str(baseline_2638_stage5_small))
        assert meta["ft.acquisition_us"] == pytest.approx(meta["stage5.acquisition_us"])

    def test_identical_table_listing(self, baseline_2638_stage5_small):
        path = str(baseline_2638_stage5_small)
        assert ftmw.read_tables(path) == Pipeline.open(path).read_tables()
        listing = _run_read(["list", path])
        for table in READ_TABLES:
            assert table in listing

    def test_cli_writes_to_the_requested_file(
        self, baseline_2638_stage5_small, tmp_path
    ):
        out = tmp_path / "lines.csv"
        _run_read(
            [
                "table",
                str(baseline_2638_stage5_small),
                "fit_peaks",
                "--columns",
                "frequency_mhz",
                "--output",
                str(out),
            ]
        )
        written = _parse_csv(out.read_text())
        np.testing.assert_array_equal(
            np.array([float(c) for c in written["frequency_mhz"]]),
            ftmw.read_table(
                str(baseline_2638_stage5_small), "fit_peaks", ["frequency_mhz"]
            )["frequency_mhz"],
        )


@pytest.mark.integration
class TestReadErrors:
    def test_unavailable_stage_reports_the_command_to_run(
        self, baseline_2638_stage4, tmp_path
    ):
        import shutil

        path = tmp_path / "stage4_only.ftmw"
        shutil.copy(baseline_2638_stage4, path)
        with pytest.raises(ValueError, match="fit run"):
            ftmw.read_table(str(path), "fit_peaks")

        result = subprocess.run(
            ["ftmwpipeline", "read", "table", str(path), "fit_peaks"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 1
        assert "fit run" in result.stdout

    def test_unknown_column_reports_the_valid_ones(self, baseline_2638_stage5_small):
        with pytest.raises(ValueError, match="unknown column"):
            ftmw.read_table(
                str(baseline_2638_stage5_small), "fit_peaks", columns=["nope"]
            )

    def test_api_and_cli_refuse_the_same_future_format_file(
        self, baseline_2638_stage5_small, tmp_path
    ):
        """The one gate the read tap shares with the full loaders."""
        import shutil

        import h5py

        from ftmwpipeline.file_manager import PipelineCompatibilityError

        path = tmp_path / "future.ftmw"
        shutil.copy(baseline_2638_stage5_small, path)
        with h5py.File(path, "a") as h5f:
            h5f.attrs["ftmw_format_version"] = "99.0"

        with pytest.raises(PipelineCompatibilityError):
            ftmw.read_table(str(path), "fit_peaks")

        result = subprocess.run(
            ["ftmwpipeline", "read", "table", str(path), "fit_peaks"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 1
        assert "Error:" in result.stdout
