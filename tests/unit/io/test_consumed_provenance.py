"""
Recording-rule tests for the consumed-values task.

Complements ``test_consumed_records.py`` (codec round trips): these check the
field-set version bumps and the pre-provenance reading, that "None means unset"
survives a current-version record, and that Stage 6 resolves the clock
declaration under the Stage 5 empty-versus-unset rule. Each test names the
mutation it catches.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage6_impl import (
    _resolve_calibration_clocks,
    _write_stage6_review_only,
)
from ftmwpipeline.core.data_structures import FinalProducts, Stage6Review
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings
from ftmwpipeline.core.stage_fit_settings import ClockSource, StageFitSettings
from ftmwpipeline.core.stage_fit_settings import resolve as resolve_stage_fit
from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings
from ftmwpipeline.io.peak_detection_settings_serialization import (
    STAGE3_PEAKS_FIELD_SET_VERSION,
    STAGE3_PEAKS_SETTINGS_PATH,
    Stage3Consumed,
    load_peak_detection_consumed_from_h5,
    peak_detection_settings_provenance,
    save_peak_detection_settings_to_h5,
)
from ftmwpipeline.io.provenance import write_field_set_version
from ftmwpipeline.io.stage6_review_serialization import (
    read_final_products_calibration_clocks,
)
from ftmwpipeline.io.stage_fit_settings_serialization import (
    STAGE5_FIT_FIELD_SET_VERSION,
    STAGE_FIT_PATH,
    Stage5Consumed,
    load_stage_fit_consumed_from_h5,
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
    stage_fit_settings_provenance,
)
from ftmwpipeline.io.timebase_serialization import (
    TIMEBASE_FIELD_SET_VERSION,
    load_timebase_calibration_from_hdf5,
    save_timebase_calibration_to_hdf5,
    timebase_calibration_provenance,
)
from ftmwpipeline.io.window_planning_settings_serialization import (
    load_window_planning_settings_from_h5,
    save_window_planning_settings_to_h5,
)

from .test_consumed_records import CLOCKS, _timebase


@pytest.fixture
def empty_ftmw(tmp_path: Any) -> str:
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


@pytest.fixture
def imported(tmp_path: Any) -> str:
    """A bare imported .ftmw: Stage 0 and nothing else."""
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(1024) * 0.01))
    out = tmp_path / "exp.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return str(out)


S3_CONSUMED = Stage3Consumed(9.5, "gaussian", "stage2b_tau_G_maj")
S5_CONSUMED = Stage5Consumed("none", None, None, None, None, None, 5.0)


class TestVersionsAndPreProvenance:
    def test_a_saved_record_is_current_and_the_versions_were_bumped(
        self, empty_ftmw: str
    ) -> None:
        # Mutation: leave a field-set version at 1 after adding the fields.
        # Version 1 is what a record written before ``consumed`` carries (the
        # next test), so it must now read as older than current.
        assert STAGE3_PEAKS_FIELD_SET_VERSION >= 2
        assert STAGE5_FIT_FIELD_SET_VERSION >= 2
        assert TIMEBASE_FIELD_SET_VERSION >= 2
        with atomic_write(empty_ftmw):
            save_peak_detection_settings_to_h5(
                empty_ftmw, PeakDetectionSettings(), consumed=S3_CONSUMED
            )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=S5_CONSUMED
            )
        p3 = peak_detection_settings_provenance(empty_ftmw)
        p5 = stage_fit_settings_provenance(empty_ftmw)
        assert p3 is not None and p3.is_current
        assert p5 is not None and p5.is_current

    def test_a_version_1_record_is_pre_provenance_and_has_no_consumed(
        self, empty_ftmw: str
    ) -> None:
        # Mutation: the loader invents a consumed value for an old record, or
        # the provenance reader calls a version-1 record current.
        with atomic_write(empty_ftmw):
            save_peak_detection_settings_to_h5(
                empty_ftmw, PeakDetectionSettings(), consumed=S3_CONSUMED
            )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=S5_CONSUMED
            )
        with h5py.File(empty_ftmw, "a") as h5f:
            for path in (STAGE3_PEAKS_SETTINGS_PATH, STAGE_FIT_PATH):
                grp = h5f[path]
                del grp["consumed"]
                write_field_set_version(grp.attrs, 1)
        for prov in (
            peak_detection_settings_provenance(empty_ftmw),
            stage_fit_settings_provenance(empty_ftmw),
        ):
            assert prov is not None and prov.is_pre_provenance
            assert prov.version == 1
        assert load_peak_detection_consumed_from_h5(empty_ftmw) is None
        assert load_stage_fit_consumed_from_h5(empty_ftmw) is None
        # The settings themselves still load: old files open and run.
        assert load_stage_fit_settings_from_h5(empty_ftmw) is not None

    def test_a_record_with_no_version_is_pre_provenance(self, empty_ftmw: str) -> None:
        # Mutation: treating a missing version attr as current.
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=S5_CONSUMED
            )
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE_FIT_PATH].attrs["field_set_version"]
        prov = stage_fit_settings_provenance(empty_ftmw)
        assert prov is not None and prov.is_pre_provenance and prov.version is None

    def test_timebase_record_versions(self, empty_ftmw: str) -> None:
        # Mutation: not bumping TIMEBASE_FIELD_SET_VERSION for clock_sources;
        # or loading a record written without the attr as a typed declaration.
        with h5py.File(empty_ftmw, "a") as h5f:
            grp = h5f.create_group("timebase_calibration")
            save_timebase_calibration_to_hdf5(_timebase(clock_sources=CLOCKS), grp)
            assert timebase_calibration_provenance(grp).is_current
            # What a version-1 record looks like: no clock_sources, version 1.
            del grp.attrs["clock_sources"]
            write_field_set_version(grp.attrs, 1)
            assert timebase_calibration_provenance(grp).is_pre_provenance
            assert load_timebase_calibration_from_hdf5(grp).clock_sources is None

    def test_timebase_clock_labels_and_lock_survive(self, empty_ftmw: str) -> None:
        # Mutation: persisting only freq_mhz (the parameters_used shape),
        # dropping locked/label.
        with h5py.File(empty_ftmw, "a") as h5f:
            grp = h5f.create_group("timebase_calibration")
            save_timebase_calibration_to_hdf5(_timebase(clock_sources=CLOCKS), grp)
            loaded = load_timebase_calibration_from_hdf5(grp).clock_sources
        assert loaded is not None
        assert [(c.locked, c.label) for c in loaded] == [
            (True, "DownLO"),
            (False, "digitizer"),
        ]

    def test_an_empty_timebase_declaration_is_not_none(self, empty_ftmw: str) -> None:
        # Mutation: encoding () as absent, so "ran with no clocks" reads as
        # "unrecorded".
        with h5py.File(empty_ftmw, "a") as h5f:
            grp = h5f.create_group("timebase_calibration")
            save_timebase_calibration_to_hdf5(_timebase(clock_sources=()), grp)
            assert load_timebase_calibration_from_hdf5(grp).clock_sources == ()


class TestNoneMeansUnset:
    def test_stage5_unset_knobs_round_trip_as_none(self, empty_ftmw: str) -> None:
        # Mutation: the codec substitutes a default for a None on write or read
        # (e.g. fit_tau -> True), so an explicit unset becomes a value.
        resolved = resolve_stage_fit()
        settings = replace(
            resolved,
            tau=replace(
                resolved.tau,
                fit_tau=None,
                tau_maj_override_us=None,
                sigma_tau_override_us=None,
            ),
            peak_survival=replace(resolved.peak_survival, snr_survival_floor=None),
        )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(empty_ftmw, settings)
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.tau.fit_tau is None
        assert loaded.tau.tau_maj_override_us is None
        assert loaded.tau.sigma_tau_override_us is None
        assert loaded.peak_survival.snr_survival_floor is None

    def test_stage5_resolved_fit_tau_persists_true(self, empty_ftmw: str) -> None:
        # Mutation: removing fit_tau from the hard defaults, so the persisted
        # record says None where the fit used True.
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(empty_ftmw, resolve_stage_fit())
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        # The codec reads bool knobs back as numpy bools, hence not ``is True``.
        assert loaded is not None
        assert loaded.tau.fit_tau is not None and bool(loaded.tau.fit_tau)

    def test_stage4_leakage_tau_none_round_trips(self, empty_ftmw: str) -> None:
        # Mutation: the codec drops or defaults an unset leakage.tau_us.
        settings = WindowPlanningSettings()
        assert settings.leakage.tau_us is None
        with atomic_write(empty_ftmw):
            save_window_planning_settings_to_h5(empty_ftmw, settings)
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None and loaded.leakage.tau_us is None
        set_value = replace(settings, leakage=replace(settings.leakage, tau_us=6.0))
        with atomic_write(empty_ftmw):
            save_window_planning_settings_to_h5(empty_ftmw, set_value)
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None and loaded.leakage.tau_us == 6.0


def _persist_clocks(path: str, clocks: Optional[Tuple[ClockSource, ...]]) -> None:
    resolved = resolve_stage_fit()
    with atomic_write(path):
        save_stage_fit_settings_to_h5(
            path, replace(resolved, spur=replace(resolved.spur, clocks=clocks))
        )


RECOMMENDED = [
    {"freq_mhz": 5120.0, "locked": True},
    {"freq_mhz": 6250.0, "locked": False},
]
PERSISTED = (ClockSource(100.0, locked=True, label="persisted"),)


class TestStage6ClockRule:
    def test_recommended_applies_with_no_stage5_record(self, imported: str) -> None:
        # Mutation: ignoring the recommended layer before Stage 5 has run.
        ftmw.set_clock_sources(imported, RECOMMENDED)
        got = _resolve_calibration_clocks(imported)
        assert [c.freq_mhz for c in got] == [5120.0, 6250.0]

    def test_recommended_applies_when_persisted_clocks_unset(
        self, imported: str
    ) -> None:
        # Mutation: treating an unset (None) persisted declaration as "set but
        # empty", which would hide a declaration no stage has overridden.
        _persist_clocks(imported, None)
        ftmw.set_clock_sources(imported, RECOMMENDED)
        assert len(_resolve_calibration_clocks(imported)) == 2

    def test_persisted_empty_falls_through_like_the_timebase(
        self, imported: str
    ) -> None:
        # Mutation: treating a persisted empty declaration as authoritative,
        # which disagrees with the timebase resolver and drops a measured
        # epsilon after a post-fit ``clocks set``.
        from ftmwpipeline._internal.timebase_impl import _resolve_clock_sources

        _persist_clocks(imported, ())
        ftmw.set_clock_sources(imported, RECOMMENDED)
        got = _resolve_calibration_clocks(imported)
        assert [c.freq_mhz for c in got] == [5120.0, 6250.0]
        assert got == tuple(_resolve_clock_sources(imported, None))

    def test_persisted_nonempty_beats_recommended(self, imported: str) -> None:
        # Mutation: recommended layer outranking the persisted one.
        _persist_clocks(imported, PERSISTED)
        ftmw.set_clock_sources(imported, RECOMMENDED)
        assert _resolve_calibration_clocks(imported) == PERSISTED

    def test_the_table_records_the_declaration_it_was_derived_under(
        self, imported: str
    ) -> None:
        # Mutation: Stage 6 recording the persisted Stage 5 layer instead of
        # the resolved declaration the calibration state was derived under.
        _persist_clocks(imported, ())
        ftmw.set_clock_sources(imported, RECOMMENDED)
        with atomic_write(imported):
            _write_stage6_review_only(
                Stage6Review(final_products=FinalProducts()), imported
            )
        with h5py.File(imported, "r") as h5f:
            got = read_final_products_calibration_clocks(h5f["stage6_review"])
        assert got == _resolve_calibration_clocks(imported)
        assert got is not None and len(got) == 2

    def test_every_review_write_records_the_declaration(self, imported: str) -> None:
        # Mutation: a writer that bypasses _write_stage6_review_only (or drops
        # calibration_clocks=) leaves the table without its declaration.
        _persist_clocks(imported, PERSISTED)
        with atomic_write(imported):
            _write_stage6_review_only(
                Stage6Review(final_products=FinalProducts()), imported
            )
        with h5py.File(imported, "r") as h5f:
            got = read_final_products_calibration_clocks(h5f["stage6_review"])
        assert got == PERSISTED

    def test_no_table_means_no_declaration_recorded(self, imported: str) -> None:
        # Mutation: recording a declaration for a review with no final
        # products, claiming a derivation that never happened.
        with atomic_write(imported):
            _write_stage6_review_only(Stage6Review(final_products=None), imported)
        with h5py.File(imported, "r") as h5f:
            assert read_final_products_calibration_clocks(h5f["stage6_review"]) is None
