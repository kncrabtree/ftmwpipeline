"""Stage 2b: one settings record per producer.

The Lorentzian calibration (``tau``), the Gaussian calibration (``tau_g``) and
the shape recommendation (``tau.recommendation``) each persist their own
resolved record; none overwrites another, so each record says what its own
result used. ``processing_parameters/stage2b_tau`` stays the shared recipe the
next run resolves against (what ``settings show stage2b`` displays).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import shape_recommendation_impl, stage2b_impl
from ftmwpipeline.core.tau_calibration_settings import (
    TauCalibrationSettings,
    resolve,
)
from ftmwpipeline.fitting.tau_calibration import DEFAULT_TAU_MAX_FACTOR
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_stage2b_recommended_shape,
)
from ftmwpipeline.io.tau_calibration_settings_serialization import (
    SHAPE_RECOMMENDATION_SETTINGS_PATH,
    STAGE2B_LORENTZIAN_SETTINGS_PATH,
    load_shape_recommendation_record,
    load_tau_calibration_settings_from_h5,
    load_tau_producer_settings_from_h5,
    tau_producer_settings_provenance,
)

pytestmark = [pytest.mark.integration]


def _settings(n_seg: int, **stft: Any) -> TauCalibrationSettings:
    s = TauCalibrationSettings()
    s.stft.n_seg = n_seg
    for k, v in stft.items():
        setattr(s.stft, k, v)
    s.recommendation.auto_recommend = False
    return s


@pytest.fixture
def f2b(baseline_2638_stage2: Path, tmp_path: Path) -> str:
    dst = tmp_path / "w.ftmw"
    shutil.copyfile(baseline_2638_stage2, dst)
    return str(dst)


def _n_seg(f: str, producer: str) -> int:
    rec = load_tau_producer_settings_from_h5(f, producer)
    assert rec is not None, f"no {producer} record"
    return rec["stft"]["n_seg"]


class TestNoOverwrite:
    def test_gaussian_then_lorentzian_keep_their_own_records(self, f2b) -> None:
        # Mutation: both twins write one shared record (last writer wins) ->
        # the gaussian record would read 12.
        stage2b_impl.calibrate_tau_impl(f2b, shape="gaussian", settings=_settings(8))
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(12))
        assert _n_seg(f2b, "gaussian") == 8
        assert _n_seg(f2b, "lorentzian") == 12

    def test_recommendation_after_twins_leaves_twin_records(self, f2b) -> None:
        # Mutation: the recommender writes the twin record path.
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(12))
        stage2b_impl.calibrate_tau_impl(f2b, shape="gaussian", settings=_settings(8))
        shape_recommendation_impl.recommend_shape_impl(f2b, settings=_settings(6))
        assert _n_seg(f2b, "lorentzian") == 12
        assert _n_seg(f2b, "gaussian") == 8
        assert _n_seg(f2b, "recommendation") == 6

    def test_every_record_is_current_version(self, f2b) -> None:
        # The record round-trips at the current field-set version.
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(12))
        stage2b_impl.calibrate_tau_impl(f2b, shape="gaussian", settings=_settings(8))
        shape_recommendation_impl.recommend_shape_impl(f2b, settings=_settings(6))
        for p in ("lorentzian", "gaussian", "recommendation"):
            prov = tau_producer_settings_provenance(f2b, p)
            assert prov is not None and prov.is_current


class TestAutoRecommendRun:
    def test_default_run_records_every_producer_consistently(self, f2b) -> None:
        """A default calibration (auto_recommend on) leaves the recommendation
        record and the cross-built twin's record in place.

        Mutation: the cross-built twin resets the recommendation (deleting its
        record), or does not record itself."""
        s = TauCalibrationSettings()
        s.stft.n_seg = 12
        stage2b_impl.calibrate_tau_impl(f2b, settings=s)

        rec = load_shape_recommendation_record(f2b)
        assert rec is not None
        assert rec.recommended_shape == read_stage2b_recommended_shape(f2b)
        assert _n_seg(f2b, "lorentzian") == 12
        assert rec.settings["stft"]["n_seg"] == 12
        if rec.recommended_shape in ("lorentzian", "gaussian"):
            twin = load_tau_producer_settings_from_h5(f2b, rec.recommended_shape)
            assert twin is not None


class TestWithdrawal:
    def test_primary_calibration_withdraws_and_attr_agree(self, f2b) -> None:
        """Record presence tracks the verdict in effect: with auto_recommend
        off, a primary calibration clears both the attr and the record.

        Mutation: the record is not deleted -> a stale verdict stays recorded
        while Stage 5 reads None."""
        shape_recommendation_impl.recommend_shape_impl(f2b)
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(10))
        assert read_stage2b_recommended_shape(f2b) is None
        assert load_shape_recommendation_record(f2b) is None
        with h5py.File(f2b, "r") as h5f:
            assert SHAPE_RECOMMENDATION_SETTINGS_PATH not in h5f


class TestOldFile:
    def test_file_without_producer_records_opens_and_runs(self, f2b) -> None:
        """An old file (results, no producer records) reads None, not a guess,
        and the next run of a producer writes its record.

        Mutation: readers invent defaults for an absent record."""
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(12))
        with h5py.File(f2b, "a") as h5f:
            del h5f[STAGE2B_LORENTZIAN_SETTINGS_PATH]
        assert load_tau_producer_settings_from_h5(f2b, "lorentzian") is None
        assert tau_producer_settings_provenance(f2b, "lorentzian") is None
        # The result still loads.
        assert stage2b_impl.load_tau_calibration_impl(f2b)["tau_calibration"]
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(9))
        assert _n_seg(f2b, "lorentzian") == 9


class TestSharedRecipe:
    def test_settings_show_and_set_still_drive_the_next_run(self, f2b) -> None:
        """``settings show stage2b`` reports the recipe the last producer
        resolved; ``settings set`` then steers a producer that runs with no
        explicit settings, and that producer's record says so.

        Mutation: the recipe stops being written (show stays stale), or the
        twin ignores the persisted layer."""
        stage2b_impl.calibrate_tau_impl(f2b, settings=_settings(12))
        rows = {r.path: r for r in ftmw.settings_show(f2b, "stage2b.stft")}
        assert rows["stage2b.stft.n_seg"].value == 12

        ftmw.settings_set(f2b, "stage2b.stft.n_seg", 9)
        recipe = load_tau_calibration_settings_from_h5(f2b)
        assert recipe is not None and recipe.stft.n_seg == 9

        s = TauCalibrationSettings()
        s.recommendation.auto_recommend = False
        stage2b_impl.calibrate_tau_impl(f2b, shape="gaussian", settings=s)
        assert _n_seg(f2b, "gaussian") == 9


class TestTauMaxFactor:
    def test_default_factor_is_the_old_constant(self) -> None:
        # Mutation: the registered default drifts from the kernel constant,
        # moving every fresh run's clip.
        assert resolve().stft.tau_max_factor == DEFAULT_TAU_MAX_FACTOR == 5.0

    @pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
    def test_default_run_equals_the_pre_change_kernel_call(
        self, f2b, monkeypatch, shape
    ) -> None:
        """With the default factor the kernel result is bit-identical to the
        call that never passed a factor (the module constant path).

        Mutation: the default factor differs from the constant, or the impl
        passes a different factor than resolved."""
        name = (
            "extract_tau_majority"
            if shape == "lorentzian"
            else ("extract_tau_G_majority")
        )
        real = getattr(stage2b_impl, name)
        pairs: List[Any] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            with_factor = real(*args, **kwargs)
            legacy_kwargs = {k: v for k, v in kwargs.items() if k != "tau_max_factor"}
            legacy = real(*args, **legacy_kwargs)
            pairs.append((with_factor, legacy))
            return with_factor

        monkeypatch.setattr(stage2b_impl, name, spy)
        stage2b_impl.calibrate_tau_impl(f2b, shape=shape, settings=_settings(12))
        assert len(pairs) == 1
        new, legacy = pairs[0]
        assert new.tau_max_us == legacy.tau_max_us
        assert new.tau_maj_us == legacy.tau_maj_us
        assert new.sigma_tau_us == legacy.sigma_tau_us
        np.testing.assert_array_equal(
            new.contributor_taus_us, legacy.contributor_taus_us
        )

    @pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
    def test_effective_clip_is_persisted_with_each_result(self, f2b, shape) -> None:
        """The factor reaches the kernel and the effective clip rides on the
        result codec; the record says what was set.

        Mutation: the factor is dropped before the kernel (clip stays 5x)."""
        stage2b_impl.calibrate_tau_impl(
            f2b, shape=shape, settings=_settings(12, tau_max_factor=3.0)
        )
        res = stage2b_impl.load_tau_calibration_impl(f2b, shape=shape)[
            "tau_calibration"
        ]
        stage2b_impl.calibrate_tau_impl(f2b, shape=shape, settings=_settings(12))
        default = stage2b_impl.load_tau_calibration_impl(f2b, shape=shape)[
            "tau_calibration"
        ]
        assert default.tau_max_us > 0
        assert res.tau_max_us == pytest.approx(
            3.0 / DEFAULT_TAU_MAX_FACTOR * default.tau_max_us, rel=1e-9
        )
        rec = load_tau_producer_settings_from_h5(f2b, shape)
        assert rec is not None
        assert rec["stft"]["tau_max_factor"] == 3.0
        assert rec["stft"]["tau_max_us"] is None

    def test_explicit_tau_max_us_is_the_effective_clip(self, f2b) -> None:
        # Mutation: the factor overrides an explicit tau_max_us.
        stage2b_impl.calibrate_tau_impl(
            f2b, settings=_settings(12, tau_max_us=40.0, tau_max_factor=3.0)
        )
        res = stage2b_impl.load_tau_calibration_impl(f2b)["tau_calibration"]
        assert res.tau_max_us == 40.0
        rec = load_tau_producer_settings_from_h5(f2b, "lorentzian")
        assert rec is not None and rec["stft"]["tau_max_us"] == 40.0

    def test_recommendation_verdict_carries_its_effective_clip(self, f2b) -> None:
        # Mutation: the recommender ignores the factor / does not store the clip.
        out = shape_recommendation_impl.recommend_shape_impl(
            f2b, settings=_settings(12, tau_max_factor=3.0)
        )
        rec = load_shape_recommendation_record(f2b)
        assert rec is not None
        assert rec.tau_max_us == out["shape_recommendation"].tau_max_us
        assert rec.settings["stft"]["tau_max_factor"] == 3.0
        default = shape_recommendation_impl.recommend_shape_impl(
            f2b, settings=_settings(12)
        )["shape_recommendation"]
        assert rec.tau_max_us == pytest.approx(
            3.0 / DEFAULT_TAU_MAX_FACTOR * default.tau_max_us, rel=1e-9
        )


class TestRecommendationWithoutGroup:
    def test_verdict_persists_with_no_2b_group(
        self, baseline_2638_stage1: Path, tmp_path: Path
    ) -> None:
        """The verdict lives in the record even with no Stage 2b group to
        carry the attr.

        Mutation: the record is written only when a group is stamped."""
        dst = tmp_path / "nogroup.ftmw"
        shutil.copyfile(baseline_2638_stage1, dst)
        f = str(dst)
        out = shape_recommendation_impl.recommend_shape_impl(f)
        assert out["groups_written"] == []
        rec = load_shape_recommendation_record(f)
        assert rec is not None
        assert rec.recommended_shape == out["shape_recommendation"].recommended_shape
        prov = tau_producer_settings_provenance(f, "recommendation")
        assert prov is not None and prov.is_current
