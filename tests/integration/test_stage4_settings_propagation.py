"""Integration: every consumed WindowPlanningSettings field reaches the kernel.

Stage 4's ``build_window_plan`` exposes 8 user-facing kwargs across four
sub-dataclasses (``coherence``, ``clustering``, ``contributor``, ``leakage``).
Most fall through to module-level ``DEFAULT_*`` constants if the orchestrator
fails to forward them, so a missed wiring would silently regress to those
defaults with no visible error.

These tests intercept the kernel call from
``_internal/stage4_impl.assign_windows_impl`` and assert that every routed
:class:`WindowPlanningSettings` field reaches the kernel's kwargs bag.
A test failing on "key missing" indicates a phantom field (the dataclass /
YAML accept the value, the persisted record carries it, but the planner
ignores it).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from ftmwpipeline._internal import stage4_impl
from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings

pytestmark = [
    pytest.mark.integration,
]


class _CalibIntercepted(RuntimeError):
    """Sentinel raised from the mocked kernel to stop the orchestrator."""


def _intercept() -> Tuple[Callable[..., Any], Dict[str, Any]]:
    captured: Dict[str, Any] = {}

    def _fake_kernel(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        raise _CalibIntercepted("intercepted")

    return _fake_kernel, captured


def _sub_set(sub_name: str, field_name: str, value: Any) -> Callable[..., None]:
    def setter(s: WindowPlanningSettings) -> None:
        setattr(getattr(s, sub_name), field_name, value)

    return setter


# (label, setter, expected_kernel_kwarg, expected_value)
PROPAGATION_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("coherence.edge_m", _sub_set("coherence", "edge_m", 128), "edge_m", 128),
    ("coherence.trim_m", _sub_set("coherence", "trim_m", 16), "trim_m", 16),
    (
        "coherence.edge_threshold",
        _sub_set("coherence", "edge_threshold", 6.0),
        "edge_threshold",
        6.0,
    ),
    (
        "clustering.max_window_width_mhz",
        _sub_set("clustering", "max_window_width_mhz", 25.0),
        "max_window_width_mhz",
        25.0,
    ),
    (
        "clustering.min_window_half_width_mhz",
        _sub_set("clustering", "min_window_half_width_mhz", 3.5),
        "min_window_half_width_mhz",
        3.5,
    ),
    (
        "contributor.min_freeze_snr",
        _sub_set("contributor", "min_freeze_snr", 25.0),
        "min_freeze_snr",
        25.0,
    ),
    (
        "contributor.magnitude_attachment_threshold",
        _sub_set("contributor", "magnitude_attachment_threshold", 0.05),
        "magnitude_attachment_threshold",
        0.05,
    ),
    ("leakage.tau_us", _sub_set("leakage", "tau_us", 5.0), "tau_us", 5.0),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    PROPAGATION_FIELDS,
    ids=[c[0] for c in PROPAGATION_FIELDS],
)
def test_assign_windows_field_reaches_kernel(
    baseline_2638_stage3: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    """Each WindowPlanningSettings field forwards into build_window_plan's kwargs."""
    variant = tmp_path / f"propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage3, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage4_impl, "build_window_plan", mock)

    s = WindowPlanningSettings()
    setter(s)

    with pytest.raises(_CalibIntercepted):
        stage4_impl.assign_windows_impl(str(variant), settings=s)

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to build_window_plan. Phantom field."
    )
    assert (
        kwargs[key] == pytest.approx(value)
        if isinstance(value, float)
        else kwargs[key] == value
    ), (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


class TestMutualExclusion:
    """Passing both ``settings=`` and ``preset=`` to ``assign_windows_impl``
    must raise ``ValueError``, matching Stages 5, 2b, 2, and 3."""

    def test_settings_and_preset_both_raises(
        self,
        baseline_2638_stage3: Path,
        tmp_path: Path,
    ) -> None:
        variant = tmp_path / "both.ftmw"
        shutil.copyfile(baseline_2638_stage3, variant)
        s = WindowPlanningSettings()
        with pytest.raises(ValueError, match=r"mutually|alternative"):
            stage4_impl.assign_windows_impl(
                str(variant),
                settings=s,
                preset="instrument_bc_2638",
            )


class TestPersistedLayerInherit:
    """A no-kwargs follow-up call must inherit the previously resolved
    settings from the persisted ``processing_parameters/stage4_windows`` block."""

    def test_assign_windows_inherits_persisted_edge_m(
        self,
        baseline_2638_stage3: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.window_planning_settings_serialization import (
            save_window_planning_settings_to_h5,
        )

        variant = tmp_path / "inherit.ftmw"
        shutil.copyfile(baseline_2638_stage3, variant)

        persisted = WindowPlanningSettings()
        persisted.coherence.edge_m = 128
        save_window_planning_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage4_impl, "build_window_plan", mock)

        with pytest.raises(_CalibIntercepted):
            stage4_impl.assign_windows_impl(str(variant))

        assert captured["kwargs"]["edge_m"] == 128, (
            "no-kwargs follow-up did not inherit the persisted "
            "edge_m; the persisted layer of the resolver is misrouted"
        )


class TestExplicitSettingsOverridePersisted:
    """A passed ``settings=`` bundle is the explicit override: it outranks a
    value already persisted in the ``.ftmw`` (D11 ``explicit > persisted``)."""

    def test_settings_beats_persisted_edge_m(
        self,
        baseline_2638_stage3: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.window_planning_settings_serialization import (
            save_window_planning_settings_to_h5,
        )

        variant = tmp_path / "override.ftmw"
        shutil.copyfile(baseline_2638_stage3, variant)

        persisted = WindowPlanningSettings()
        persisted.coherence.edge_m = 99
        save_window_planning_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage4_impl, "build_window_plan", mock)

        s = WindowPlanningSettings()
        s.coherence.edge_m = 128

        with pytest.raises(_CalibIntercepted):
            stage4_impl.assign_windows_impl(str(variant), settings=s)

        assert captured["kwargs"]["edge_m"] == 128, (
            "an explicit settings= bundle must override the persisted "
            "edge_m; settings= is misrouted below the persisted layer"
        )
