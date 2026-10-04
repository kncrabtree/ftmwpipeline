"""Integration: every consumed TauCalibrationSettings field actually drives the kernel.

Stage 2b exposes three user-facing kernels -- ``extract_tau_majority``
(pure-exp), ``extract_tau_G_majority`` (Gaussian twin), and
``compute_shape_recommendation`` (3-way L/G/V vote) -- and each has ~10
documented knobs. Most fields fall through to ``DEFAULT_*`` constants
inside :mod:`ftmwpipeline.fitting.tau_calibration` if the orchestrator
fails to forward them, so a missed wiring would silently regress to
hard defaults with no visible error.

These tests intercept each kernel call from its impl wrapper and assert
that every routed :class:`TauCalibrationSettings` field reaches the
kernel's kwargs bag. A test failing on "key missing" indicates a phantom
field (the dataclass / YAML accept the value, the persisted record
carries it, but the calibration ignores it).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from ftmwpipeline._internal import (
    shape_recommendation_impl,
    stage2b_impl,
)
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.core.tau_calibration_settings import (
    RecommendationSubSettings,
    StftSubSettings,
    TauCalibrationSettings,
)

pytestmark = [
    pytest.mark.integration,
]


class _CalibIntercepted(RuntimeError):
    """Sentinel raised from the mocked kernel to stop the orchestrator."""


def _intercept(return_value: Any = None) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    captured: Dict[str, Any] = {}

    def _fake_kernel(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        raise _CalibIntercepted("intercepted")

    return _fake_kernel, captured


# ---------------------------------------------------------------------------
# calibrate_tau_impl -> extract_tau_majority
# ---------------------------------------------------------------------------
def _sub_set(sub_name: str, field_name: str, value: Any) -> Callable[..., None]:
    def setter(s: TauCalibrationSettings) -> None:
        setattr(getattr(s, sub_name), field_name, value)

    return setter


# (label, setter, expected_kernel_kwarg, expected_value)
CALIBRATE_TAU_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("stft.n_seg", _sub_set("stft", "n_seg", 12), "n_seg", 12),
    ("stft.t_sigma", _sub_set("stft", "t_sigma", 7.5), "t_sigma", 7.5),
    ("stft.tau_max_us", _sub_set("stft", "tau_max_us", 60.0), "tau_max_us", 60.0),
    (
        "stft.tau_max_factor",
        _sub_set("stft", "tau_max_factor", 9.0),
        "tau_max_factor",
        9.0,
    ),
    (
        "stft.rss_gate_factor",
        _sub_set("stft", "rss_gate_factor", 3.0),
        "rss_gate_factor",
        3.0,
    ),
    (
        "stft.relative_gate_fraction",
        _sub_set("stft", "relative_gate_fraction", 0.08),
        "relative_gate_fraction",
        0.08,
    ),
    ("stft.sigma_x_full", _sub_set("stft", "sigma_x_full", 1.5), "sigma_x_full", 1.5),
    ("polish.polish", _sub_set("polish", "polish", False), "polish", False),
    (
        "polish.polish_n_iter",
        _sub_set("polish", "polish_n_iter", 3),
        "polish_n_iter",
        3,
    ),
    (
        "polish.polish_snr_cap",
        _sub_set("polish", "polish_snr_cap", 14.0),
        "polish_snr_cap",
        14.0,
    ),
    (
        "polish.polish_noise_debias",
        _sub_set("polish", "polish_noise_debias", True),
        "polish_noise_debias",
        True,
    ),
    (
        "aggregation.min_contributors",
        _sub_set("aggregation", "min_contributors", 333),
        "min_contributors",
        333,
    ),
    (
        "aggregation.sigma_tau_fraction_max",
        _sub_set("aggregation", "sigma_tau_fraction_max", 0.33),
        "sigma_tau_fraction_max",
        0.33,
    ),
    (
        "aggregation.bimodality_dominant_fraction",
        _sub_set("aggregation", "bimodality_dominant_fraction", 0.85),
        "bimodality_dominant_fraction",
        0.85,
    ),
    (
        "aggregation.spur_cluster_multiplier",
        _sub_set("aggregation", "spur_cluster_multiplier", 2.0),
        "spur_cluster_multiplier",
        2.0,
    ),
    (
        "aggregation.sigma_tau_floor_us",
        _sub_set("aggregation", "sigma_tau_floor_us", 1.25),
        "sigma_tau_floor_us",
        1.25,
    ),
    (
        "band.compute_band_majorities",
        _sub_set("band", "compute_band_majorities", False),
        "compute_band_majorities_flag",
        False,
    ),
    (
        "band.min_contributors_per_band",
        _sub_set("band", "min_contributors_per_band", 17),
        "min_contributors_per_band",
        17,
    ),
    (
        "band.band_edges_mhz",
        _sub_set("band", "band_edges_mhz", (30000.0, 36000.0)),
        "band_edges_mhz",
        (30000.0, 36000.0),
    ),
    (
        "band.band_labels",
        _sub_set("band", "band_labels", ("A", "B", "C")),
        "band_labels",
        ("A", "B", "C"),
    ),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    CALIBRATE_TAU_FIELDS,
    ids=[c[0] for c in CALIBRATE_TAU_FIELDS],
)
def test_calibrate_tau_field_reaches_kernel(
    baseline_2638_stage2: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    variant = tmp_path / f"propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage2, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage2b_impl, "extract_tau_majority", mock)

    s = TauCalibrationSettings()
    setter(s)

    with pytest.raises(_CalibIntercepted):
        stage2b_impl.calibrate_tau_impl(str(variant), settings=s)

    assert "kwargs" in captured, "kernel mock did not capture kwargs"
    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to extract_tau_majority. Phantom field."
    )
    assert kwargs[key] == value, (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


# ---------------------------------------------------------------------------
# calibrate_tau_impl(shape="gaussian") -> extract_tau_G_majority
# ---------------------------------------------------------------------------
CALIBRATE_TAU_G_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("stft.n_seg", _sub_set("stft", "n_seg", 6), "n_seg", 6),
    ("stft.t_sigma", _sub_set("stft", "t_sigma", 4.0), "t_sigma", 4.0),
    ("stft.tau_max_us", _sub_set("stft", "tau_max_us", 80.0), "tau_max_us", 80.0),
    (
        "stft.tau_max_factor",
        _sub_set("stft", "tau_max_factor", 7.0),
        "tau_max_factor",
        7.0,
    ),
    (
        "stft.rss_gate_factor",
        _sub_set("stft", "rss_gate_factor", 3.5),
        "rss_gate_factor",
        3.5,
    ),
    (
        "stft.relative_gate_fraction",
        _sub_set("stft", "relative_gate_fraction", 0.09),
        "relative_gate_fraction",
        0.09,
    ),
    (
        "aggregation.sigma_tau_fraction_max",
        _sub_set("aggregation", "sigma_tau_fraction_max", 0.25),
        "sigma_tau_fraction_max",
        0.25,
    ),
    (
        "aggregation.bimodality_dominant_fraction",
        _sub_set("aggregation", "bimodality_dominant_fraction", 0.65),
        "bimodality_dominant_fraction",
        0.65,
    ),
    (
        "aggregation.spur_cluster_multiplier",
        _sub_set("aggregation", "spur_cluster_multiplier", 1.5),
        "spur_cluster_multiplier",
        1.5,
    ),
    (
        "aggregation.sigma_tau_floor_us",
        _sub_set("aggregation", "sigma_tau_floor_us", 0.9),
        "sigma_tau_floor_us",
        0.9,
    ),
    (
        "band.compute_band_majorities",
        _sub_set("band", "compute_band_majorities", False),
        "compute_band_majorities_flag",
        False,
    ),
    (
        "band.min_contributors_per_band",
        _sub_set("band", "min_contributors_per_band", 11),
        "min_contributors_per_band",
        11,
    ),
    (
        "band.band_edges_mhz",
        _sub_set("band", "band_edges_mhz", (29000.0, 34000.0)),
        "band_edges_mhz",
        (29000.0, 34000.0),
    ),
    ("gaussian.snr_min", _sub_set("gaussian", "snr_min", 22.0), "snr_min", 22.0),
    (
        "gaussian.tau_G_bound_lo",
        _sub_set("gaussian", "tau_G_bound_lo", 0.8),
        "tau_G_bound_lo",
        0.8,
    ),
    (
        "gaussian.tau_G_bound_hi",
        _sub_set("gaussian", "tau_G_bound_hi", 150.0),
        "tau_G_bound_hi",
        150.0,
    ),
    (
        "gaussian.tau_G_seeds",
        _sub_set("gaussian", "tau_G_seeds", (75.0, 25.0, 7.0)),
        "tau_G_seeds",
        (75.0, 25.0, 7.0),
    ),
    (
        "gaussian.delta_chi2r_min",
        _sub_set("gaussian", "delta_chi2r_min", 2.5),
        "delta_chi2r_min",
        2.5,
    ),
    (
        "gaussian.tau_G_upper_fraction",
        _sub_set("gaussian", "tau_G_upper_fraction", 0.85),
        "tau_G_upper_fraction",
        0.85,
    ),
    (
        "gaussian.min_contributors",
        _sub_set("gaussian", "min_contributors", 30),
        "min_contributors",
        30,
    ),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    CALIBRATE_TAU_G_FIELDS,
    ids=[c[0] for c in CALIBRATE_TAU_G_FIELDS],
)
def test_calibrate_tau_G_field_reaches_kernel(
    baseline_2638_stage2: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    variant = tmp_path / f"propagation_G_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage2, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage2b_impl, "extract_tau_G_majority", mock)

    s = TauCalibrationSettings()
    setter(s)

    with pytest.raises(_CalibIntercepted):
        stage2b_impl.calibrate_tau_impl(str(variant), shape="gaussian", settings=s)

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to extract_tau_G_majority. Phantom field."
    )
    assert kwargs[key] == value, (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


def test_gaussian_run_routes_aggregation_min_contributors(
    baseline_2638_stage2: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``aggregation.min_contributors`` routes onto the Gaussian block.

    The shared ``--min-contributors`` flag lands on ``aggregation``, but a
    Gaussian run reads ``gaussian.min_contributors``. The impl routes it across
    so every interface treats the flag identically; the caller's bundle is left
    untouched (routed on a copy).
    """
    variant = tmp_path / "propagation_G_route.ftmw"
    shutil.copyfile(baseline_2638_stage2, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage2b_impl, "extract_tau_G_majority", mock)

    s = TauCalibrationSettings()
    s.aggregation.min_contributors = 37  # only the aggregation field is set

    with pytest.raises(_CalibIntercepted):
        stage2b_impl.calibrate_tau_impl(str(variant), shape="gaussian", settings=s)

    assert captured["kwargs"]["min_contributors"] == 37
    # The caller's bundle is not mutated by the routing.
    assert s.aggregation.min_contributors == 37
    assert s.gaussian.min_contributors is None


# ---------------------------------------------------------------------------
# recommend_shape_impl -> compute_shape_recommendation
# ---------------------------------------------------------------------------
RECOMMEND_SHAPE_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("stft.n_seg", _sub_set("stft", "n_seg", 14), "n_seg", 14),
    ("stft.t_sigma", _sub_set("stft", "t_sigma", 6.0), "t_sigma", 6.0),
    ("stft.tau_max_us", _sub_set("stft", "tau_max_us", 50.0), "tau_max_us", 50.0),
    (
        "stft.tau_max_factor",
        _sub_set("stft", "tau_max_factor", 3.0),
        "tau_max_factor",
        3.0,
    ),
    (
        "stft.rss_gate_factor",
        _sub_set("stft", "rss_gate_factor", 4.5),
        "rss_gate_factor",
        4.5,
    ),
    (
        "stft.relative_gate_fraction",
        _sub_set("stft", "relative_gate_fraction", 0.07),
        "relative_gate_fraction",
        0.07,
    ),
    ("stft.sigma_x_full", _sub_set("stft", "sigma_x_full", 2.5), "sigma_x_full", 2.5),
    (
        "recommendation.snr_min",
        _sub_set("recommendation", "snr_min", 18.0),
        "snr_min",
        18.0,
    ),
    (
        "recommendation.tau_bound_lo",
        _sub_set("recommendation", "tau_bound_lo", 0.3),
        "tau_bound_lo",
        0.3,
    ),
    (
        "recommendation.tau_bound_hi",
        _sub_set("recommendation", "tau_bound_hi", 120.0),
        "tau_bound_hi",
        120.0,
    ),
    (
        "recommendation.tau_G_seeds",
        _sub_set("recommendation", "tau_G_seeds", (80.0, 30.0)),
        "tau_G_seeds",
        (80.0, 30.0),
    ),
    (
        "recommendation.pure_margin_threshold",
        _sub_set("recommendation", "pure_margin_threshold", 0.15),
        "pure_margin_threshold",
        0.15,
    ),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    RECOMMEND_SHAPE_FIELDS,
    ids=[c[0] for c in RECOMMEND_SHAPE_FIELDS],
)
def test_recommend_shape_field_reaches_kernel(
    baseline_2638_stage1: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    variant = tmp_path / f"propagation_R_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage1, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(
        shape_recommendation_impl,
        "compute_shape_recommendation",
        mock,
    )

    s = TauCalibrationSettings()
    setter(s)

    with pytest.raises(_CalibIntercepted):
        shape_recommendation_impl.recommend_shape_impl(str(variant), settings=s)

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to compute_shape_recommendation. "
        f"Phantom field."
    )
    assert kwargs[key] == value, (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


# ---------------------------------------------------------------------------
# `settings=` and `preset=` compose
# ---------------------------------------------------------------------------
class TestSettingsPresetComposition:
    """Passing both ``settings=`` and ``preset=`` to any of the three Stage 2b
    impls composes: the ``settings`` bundle is the explicit layer (wins for the
    field it sets) and the ``preset`` fills a field the explicit layer leaves
    unset (the persisted record outranks the preset, per D11)."""

    def test_calibrate_tau_composes(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "compose.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        preset_yaml = tmp_path / "preset.yml"
        preset_yaml.write_text("stage2b:\n  stft:\n    t_sigma: 4.5\n")

        mock, captured = _intercept()
        monkeypatch.setattr(stage2b_impl, "extract_tau_majority", mock)

        s = TauCalibrationSettings(stft=StftSubSettings(n_seg=7))

        with pytest.raises(_CalibIntercepted):
            stage2b_impl.calibrate_tau_impl(
                str(variant),
                settings=s,
                preset=str(preset_yaml),
            )

        kwargs = captured["kwargs"]
        assert (
            kwargs["n_seg"] == 7
        ), "explicit settings= n_seg must win over the preset/default"
        assert (
            kwargs["t_sigma"] == 4.5
        ), "preset must fill t_sigma, which the explicit settings= left unset"

    def test_calibrate_tau_G_composes(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "compose_G.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        preset_yaml = tmp_path / "preset_G.yml"
        preset_yaml.write_text("stage2b:\n  stft:\n    t_sigma: 4.5\n")

        mock, captured = _intercept()
        monkeypatch.setattr(stage2b_impl, "extract_tau_G_majority", mock)

        s = TauCalibrationSettings(stft=StftSubSettings(n_seg=7))

        with pytest.raises(_CalibIntercepted):
            stage2b_impl.calibrate_tau_impl(
                str(variant),
                shape="gaussian",
                settings=s,
                preset=str(preset_yaml),
            )

        kwargs = captured["kwargs"]
        assert (
            kwargs["n_seg"] == 7
        ), "explicit settings= n_seg must win over the preset/default"
        assert (
            kwargs["t_sigma"] == 4.5
        ), "preset must fill t_sigma, which the explicit settings= left unset"

    def test_recommend_shape_composes(
        self,
        baseline_2638_stage1: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "compose_R.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)

        preset_yaml = tmp_path / "preset_R.yml"
        preset_yaml.write_text("stage2b:\n  recommendation:\n    tau_bound_lo: 0.3\n")

        mock, captured = _intercept()
        monkeypatch.setattr(
            shape_recommendation_impl,
            "compute_shape_recommendation",
            mock,
        )

        s = TauCalibrationSettings(
            recommendation=RecommendationSubSettings(snr_min=18.0)
        )

        with pytest.raises(_CalibIntercepted):
            shape_recommendation_impl.recommend_shape_impl(
                str(variant),
                settings=s,
                preset=str(preset_yaml),
            )

        kwargs = captured["kwargs"]
        assert (
            kwargs["snr_min"] == 18.0
        ), "explicit settings= snr_min must win over the preset/default"
        assert kwargs["tau_bound_lo"] == 0.3, (
            "preset must fill tau_bound_lo, which the explicit settings= " "left unset"
        )


# ---------------------------------------------------------------------------
# Persisted settings inherit on no-kwargs follow-up
# ---------------------------------------------------------------------------
class TestPersistedLayerInherit:
    """A no-kwargs follow-up call must inherit the previously resolved
    settings from the persisted ``processing_parameters/stage2b_tau`` block."""

    def test_calibrate_tau_inherits_persisted_n_seg(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            save_tau_calibration_settings_to_h5,
        )

        variant = tmp_path / "inherit.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        # Stamp a sentinel persisted record. Use n_seg=12 to distinguish from
        # the hard default 10.
        persisted = TauCalibrationSettings()
        persisted.stft.n_seg = 12
        with atomic_write(str(variant)):
            save_tau_calibration_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage2b_impl, "extract_tau_majority", mock)

        with pytest.raises(_CalibIntercepted):
            stage2b_impl.calibrate_tau_impl(str(variant))

        assert captured["kwargs"]["n_seg"] == 12, (
            "no-kwargs follow-up did not inherit the persisted n_seg; "
            "the persisted layer of the resolver is misrouted"
        )


class TestExplicitSettingsOverridePersisted:
    """A passed ``settings=`` bundle is the explicit override: it outranks a
    value already persisted in the ``.ftmw`` (D11 ``explicit > persisted``)."""

    def test_settings_beats_persisted_n_seg(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.core.tau_calibration_settings import StftSubSettings
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            save_tau_calibration_settings_to_h5,
        )

        variant = tmp_path / "override.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        persisted = TauCalibrationSettings()
        persisted.stft.n_seg = 12
        with atomic_write(str(variant)):
            save_tau_calibration_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage2b_impl, "extract_tau_majority", mock)

        with pytest.raises(_CalibIntercepted):
            stage2b_impl.calibrate_tau_impl(
                str(variant),
                settings=TauCalibrationSettings(stft=StftSubSettings(n_seg=7)),
            )

        assert captured["kwargs"]["n_seg"] == 7, (
            "an explicit settings= bundle must override the persisted n_seg; "
            "settings= is misrouted below the persisted layer"
        )


# ---------------------------------------------------------------------------
# Auto-recommend: calibrate_tau / calibrate_tau_G fire compute_shape_recommendation
# when ``recommendation.auto_recommend`` is True (the hard default)
# ---------------------------------------------------------------------------
class TestAutoRecommend:
    """``RecommendationSubSettings.auto_recommend`` controls whether
    ``calibrate_tau`` and ``calibrate_tau_G`` invoke ``run_shape_recommendation``
    as part of the calibration flow. Default is ``True`` so the Stage 5
    resolver's *recommended* layer fires on every fresh Stage 2b run."""

    def _intercept_recommend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Dict[str, Any]:
        """Patch ``run_shape_recommendation`` in both Stage 2b orchestrators.

        Returns a ``{"called": bool, ...}`` capture dict that the orchestrator
        flips when (and only when) the auto-recommend pass fires.
        """
        captured: Dict[str, Any] = {"called": False, "file_path": None}

        def _fake_recommend(file_path: str, *args: Any, **kwargs: Any) -> Any:
            captured["called"] = True
            captured["file_path"] = file_path
            return {
                "status": "success",
                "shape_recommendation": None,
                "groups_written": [],
            }

        monkeypatch.setattr(
            stage2b_impl,
            "run_shape_recommendation",
            _fake_recommend,
        )
        return captured

    def test_calibrate_tau_auto_recommends_by_default(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "auto.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        captured = self._intercept_recommend(monkeypatch)

        stage2b_impl.calibrate_tau_impl(str(variant))

        assert captured["called"], (
            "default-on auto_recommend did not invoke run_shape_recommendation "
            "after calibrate_tau"
        )
        assert captured["file_path"] == str(variant)

    def test_calibrate_tau_skips_recommend_when_disabled(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "noauto.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        captured = self._intercept_recommend(monkeypatch)

        s = TauCalibrationSettings()
        s.recommendation.auto_recommend = False
        stage2b_impl.calibrate_tau_impl(str(variant), settings=s)

        assert not captured["called"], (
            "auto_recommend=False still invoked run_shape_recommendation; "
            "the flag does not gate the auto-run pass"
        )

    def test_calibrate_tau_G_auto_recommends_by_default(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "auto_g.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        captured = self._intercept_recommend(monkeypatch)

        stage2b_impl.calibrate_tau_impl(str(variant), shape="gaussian")

        assert captured["called"], (
            "default-on auto_recommend did not invoke run_shape_recommendation "
            "after calibrate_tau_impl(shape='gaussian')"
        )

    def test_calibrate_tau_G_skips_recommend_when_disabled(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "noauto_g.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        captured = self._intercept_recommend(monkeypatch)

        s = TauCalibrationSettings()
        s.recommendation.auto_recommend = False
        stage2b_impl.calibrate_tau_impl(str(variant), shape="gaussian", settings=s)

        assert not captured["called"]

    def test_calibrate_tau_builds_recommended_gaussian_twin(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
    ) -> None:
        """2638 votes gaussian, so `calibrate_tau` (the Lorentzian entry) must
        also build the gaussian twin -- otherwise Stage 5's gaussian fit finds
        no matching calibration and silently falls back to T_active/3."""
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            read_stage2b_recommended_shape,
        )

        variant = tmp_path / "twin.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        stage2b_impl.calibrate_tau_impl(str(variant))  # real vote, no mock

        assert read_stage2b_recommended_shape(str(variant)) == "gaussian"
        assert stage2b_impl.tau_calibration_present(str(variant))
        assert stage2b_impl.tau_calibration_present(str(variant), shape="gaussian"), (
            "calibrate_tau on a gaussian-voting fixture did not build the "
            "matching tau_G twin"
        )

    def test_cross_build_skipped_when_auto_recommend_off(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
    ) -> None:
        """With auto_recommend off there is no vote, so only the requested twin
        is built (no implicit cross-build)."""
        variant = tmp_path / "noauto_twin.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        s = TauCalibrationSettings()
        s.recommendation.auto_recommend = False
        stage2b_impl.calibrate_tau_impl(str(variant), settings=s)

        assert stage2b_impl.tau_calibration_present(str(variant))
        assert not stage2b_impl.tau_calibration_present(str(variant), shape="gaussian")


# ---------------------------------------------------------------------------
# One record per producer: each Stage 2b result says what it used
# ---------------------------------------------------------------------------
def _n_seg_settings(n_seg: int) -> TauCalibrationSettings:
    s = TauCalibrationSettings()
    s.stft.n_seg = n_seg
    s.recommendation.auto_recommend = False
    return s


class TestPerProducerRecords:
    """The two tau twins and the shape recommendation each persist their own
    resolved settings record; running one never changes what another is read
    as having used. The shared ``stage2b_tau`` recipe still follows the most
    recent run."""

    def test_running_one_producer_leaves_the_others_records(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
    ) -> None:
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            load_tau_calibration_settings_from_h5,
            load_tau_producer_settings_from_h5,
            tau_producer_settings_provenance,
        )

        variant = tmp_path / "records.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        f = str(variant)

        stage2b_impl.calibrate_tau_impl(f, settings=_n_seg_settings(12))
        stage2b_impl.calibrate_tau_impl(
            f, shape="gaussian", settings=_n_seg_settings(8)
        )
        shape_recommendation_impl.recommend_shape_impl(f, settings=_n_seg_settings(6))

        used = {
            p: load_tau_producer_settings_from_h5(f, p)
            for p in ("lorentzian", "gaussian", "recommendation")
        }
        assert {p: rec["stft"]["n_seg"] for p, rec in used.items()} == {
            "lorentzian": 12,
            "gaussian": 8,
            "recommendation": 6,
        }
        for p in used:
            prov = tau_producer_settings_provenance(f, p)
            assert prov is not None and prov.is_current
        # The recipe (persisted layer for the next run) follows the last run.
        recipe = load_tau_calibration_settings_from_h5(f)
        assert recipe is not None and recipe.stft.n_seg == 6

    def test_twin_record_matches_its_result(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
    ) -> None:
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            load_tau_producer_settings_from_h5,
        )

        variant = tmp_path / "twin_record.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        f = str(variant)
        s = _n_seg_settings(12)
        s.stft.tau_max_factor = 4.0
        out = stage2b_impl.calibrate_tau_impl(f, settings=s)
        result = out["tau_calibration"]

        used = load_tau_producer_settings_from_h5(f, "lorentzian")
        assert used is not None
        assert used["stft"]["n_seg"] == result.n_seg == 12
        assert used["stft"]["tau_max_factor"] == 4.0
        assert used["stft"]["tau_max_us"] is None
        # The effective clip rides on the result codec.
        loaded = stage2b_impl.load_tau_calibration_impl(f)["tau_calibration"]
        span_us = result.end_us - result.start_us
        assert loaded.tau_max_us == pytest.approx(4.0 * span_us, rel=1e-3)

    def test_recommendation_persists_without_a_tau_group(
        self,
        baseline_2638_stage1: Path,
        tmp_path: Path,
    ) -> None:
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            load_shape_recommendation_record,
        )

        variant = tmp_path / "rec_only.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)
        f = str(variant)
        out = shape_recommendation_impl.recommend_shape_impl(f)
        assert out["groups_written"] == []

        rec = load_shape_recommendation_record(f)
        assert rec is not None
        verdict = out["shape_recommendation"]
        assert rec.recommended_shape == verdict.recommended_shape
        assert rec.vote_rates == pytest.approx(verdict.vote_rates)
        assert rec.tau_max_us == verdict.tau_max_us
        assert rec.consumed["trim_lo_mhz"] == 26500.0
        assert rec.consumed["trim_hi_mhz"] == 40000.0

    def test_primary_calibration_withdraws_the_recommendation_record(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
    ) -> None:
        from ftmwpipeline.io.tau_calibration_settings_serialization import (
            load_shape_recommendation_record,
        )

        variant = tmp_path / "withdraw.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        f = str(variant)
        shape_recommendation_impl.recommend_shape_impl(f)
        assert load_shape_recommendation_record(f) is not None

        # A primary calibration resets the verdict; with auto_recommend off no
        # new one is produced, so no recommendation record remains.
        stage2b_impl.calibrate_tau_impl(f, settings=_n_seg_settings(10))
        assert load_shape_recommendation_record(f) is None
