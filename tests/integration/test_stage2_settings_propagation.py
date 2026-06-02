"""Integration: every consumed NoiseSettings field reaches the kernel.

Stage 2's scatter estimator exposes its knobs as flat ``NoiseSettings`` fields.
Most fall through to the kernel's signature defaults if the orchestrator fails
to forward them, so a missed wiring would silently regress to those defaults
with no visible error.

These tests intercept the kernel call from the impl wrapper and assert that
every routed :class:`NoiseSettings` field reaches the kernel's kwargs bag. A
test failing on "key missing" indicates a phantom field (the dataclass / YAML
accept the value, the persisted record carries it, but the calibration ignores
it).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from ftmwpipeline._internal import stage2_impl
from ftmwpipeline.core.noise_settings import NoiseSettings

pytestmark = [
    pytest.mark.integration,
    # The propagation tests parametrize over the legacy per-knob kwarg
    # path on purpose -- the deprecation warning fires there as expected
    # and is uninformative for these tests.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
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


def _set(field_name: str, value: Any) -> Callable[..., None]:
    def setter(s: NoiseSettings) -> None:
        setattr(s, field_name, value)
    return setter


# (label, setter, expected_kernel_kwarg, expected_value) for the scatter path.
SCATTER_PROPAGATION_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("window_mhz", _set("window_mhz", 60.0), "window_mhz", 60.0),
    ("pedestal_mhz", _set("pedestal_mhz", 40.0), "pedestal_mhz", 40.0),
    ("line_k", _set("line_k", 6.0), "line_k", 6.0),
    ("n_iter", _set("n_iter", 5), "n_iter", 5),
    ("region_aware", _set("region_aware", False), "region_aware", False),
    ("smoothing_mhz", _set("smoothing_mhz", 400.0), "smoothing_mhz", 400.0),
    ("smoothing_percentile",
     _set("smoothing_percentile", 25.0), "smoothing_percentile", 25.0),
    ("convolve_mhz", _set("convolve_mhz", 100.0), "convolve_mhz", 100.0),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    SCATTER_PROPAGATION_FIELDS,
    ids=[c[0] for c in SCATTER_PROPAGATION_FIELDS],
)
def test_scatter_field_reaches_kernel(
    baseline_2638_stage1: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    """Each scatter NoiseSettings field forwards into estimate_noise_scatter."""
    variant = tmp_path / f"scatter_propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage1, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage2_impl, "estimate_active_ft_noise", mock)

    s = NoiseSettings()
    setter(s)

    with pytest.raises(ValueError, match=r"intercepted"):
        stage2_impl.compute_noise_estimation_impl(
            str(variant), settings=s
        )

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to estimate_noise_scatter. Phantom field."
    )
    if isinstance(value, bool):
        assert kwargs[key] is value
    else:
        assert kwargs[key] == pytest.approx(value)


class TestMutualExclusion:
    """Passing both ``settings=`` and ``preset=`` to ``compute_noise_estimation_impl``
    must raise ``ValueError``, matching Stages 5 and 2b."""

    def test_settings_and_preset_both_raises(
        self, baseline_2638_stage1: Path, tmp_path: Path,
    ) -> None:
        variant = tmp_path / "both_scatter.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)
        s = NoiseSettings()
        with pytest.raises(ValueError, match=r"mutually|alternative"):
            stage2_impl.compute_noise_estimation_impl(
                str(variant), settings=s,
                preset="instrument_bc_2638",
            )


class TestPersistedLayerInherit:
    """A no-kwargs follow-up call must inherit the previously resolved
    settings from the persisted ``processing_parameters/stage2_noise`` block."""

    def test_scatter_inherits_persisted_window(
        self,
        baseline_2638_stage1: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.noise_settings_serialization import (
            save_noise_settings_to_h5,
        )

        variant = tmp_path / "scatter_inherit.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)

        persisted = NoiseSettings()
        persisted.window_mhz = 137.0
        save_noise_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage2_impl, "estimate_active_ft_noise", mock)

        with pytest.raises(ValueError, match=r"intercepted"):
            stage2_impl.compute_noise_estimation_impl(str(variant))

        assert captured["kwargs"]["window_mhz"] == 137.0, (
            "no-kwargs scatter follow-up did not inherit the persisted "
            "window_mhz; the persisted layer of the resolver is misrouted"
        )
