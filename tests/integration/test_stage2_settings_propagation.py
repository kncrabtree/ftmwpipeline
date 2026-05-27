"""Integration: every consumed NoiseSettings field reaches the kernel.

Stage 2's ``estimate_noise_adaptive`` exposes ~10 knobs spread across
the four sub-dataclasses (``binning``, ``skewness``, ``smoothing``,
``skirt_exclusion``). Most fall through to either module-level constants
or function-signature defaults if the orchestrator fails to forward them,
so a missed wiring would silently regress to those defaults with no
visible error.

These tests intercept the kernel call from the impl wrapper and assert
that every routed :class:`NoiseSettings` field reaches the kernel's
kwargs bag. A test failing on "key missing" indicates a phantom field
(the dataclass / YAML accept the value, the persisted record carries it,
but the calibration ignores it).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from ftmwpipeline._internal import stage2_impl
from ftmwpipeline.core.noise_settings import NoiseSettings

pytestmark = [pytest.mark.integration]


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
    def setter(s: NoiseSettings) -> None:
        setattr(getattr(s, sub_name), field_name, value)
    return setter


# (label, setter, expected_kernel_kwarg, expected_value)
PROPAGATION_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    ("binning.subdivision_threshold",
     _sub_set("binning", "subdivision_threshold", 0.05),
     "subdivision_threshold", 0.05),
    ("binning.abs_min_bin_size",
     _sub_set("binning", "abs_min_bin_size", 500),
     "abs_min_bin_size", 500),
    ("binning.min_bin_fraction",
     _sub_set("binning", "min_bin_fraction", 1 / 32),
     "min_bin_fraction", 1 / 32),
    ("binning.min_noise_fraction",
     _sub_set("binning", "min_noise_fraction", 0.8),
     "min_noise_fraction", 0.8),
    ("skewness.skew_target",
     _sub_set("skewness", "skew_target", 0.75),
     "skew_target", 0.75),
    ("skewness.inc",
     _sub_set("skewness", "inc", 0.02),
     "inc", 0.02),
    ("smoothing.smoothing_window_mhz",
     _sub_set("smoothing", "smoothing_window_mhz", 100.0),
     "smoothing_window_mhz", 100.0),
    ("skirt_exclusion.strong_peak_snr",
     _sub_set("skirt_exclusion", "strong_peak_snr", 25.0),
     "strong_peak_snr", 25.0),
    ("skirt_exclusion.skirt_exclusion_k",
     _sub_set("skirt_exclusion", "skirt_exclusion_k", 2.0),
     "skirt_exclusion_k", 2.0),
    ("skirt_exclusion.max_skirt_exclusion_mhz",
     _sub_set("skirt_exclusion", "max_skirt_exclusion_mhz", 300.0),
     "max_skirt_exclusion_mhz", 300.0),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    PROPAGATION_FIELDS,
    ids=[c[0] for c in PROPAGATION_FIELDS],
)
def test_estimate_noise_field_reaches_kernel(
    baseline_2638_stage1: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    """Each NoiseSettings field forwards into estimate_noise_adaptive's kwargs."""
    variant = tmp_path / f"propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage1, variant)

    mock, captured = _intercept()
    monkeypatch.setattr(stage2_impl, "estimate_noise_adaptive", mock)

    s = NoiseSettings()
    setter(s)

    # The orchestrator wraps the kernel call in a try/except that
    # re-raises as ``ValueError("Noise estimation failed: ...")``. The
    # mock's ``_CalibIntercepted`` exception is therefore visible only as
    # the wrapped ValueError; we still get the captured kwargs.
    with pytest.raises(ValueError, match=r"intercepted"):
        stage2_impl.compute_noise_estimation_impl(str(variant), settings=s)

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to estimate_noise_adaptive. Phantom field."
    )
    assert kwargs[key] == pytest.approx(value), (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


class TestMutualExclusion:
    """Passing both ``settings=`` and ``preset=`` to ``compute_noise_estimation_impl``
    must raise ``ValueError``, matching Stages 5 and 2b."""

    def test_settings_and_preset_both_raises(
        self, baseline_2638_stage1: Path, tmp_path: Path,
    ) -> None:
        variant = tmp_path / "both.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)
        s = NoiseSettings()
        with pytest.raises(ValueError, match=r"mutually|alternative"):
            stage2_impl.compute_noise_estimation_impl(
                str(variant), settings=s, preset="instrument_bc_2638",
            )

    def test_from_saved_params_with_settings_raises(
        self, baseline_2638_stage1: Path, tmp_path: Path,
    ) -> None:
        """The legacy ``from_saved_params=True`` flag is incompatible with
        the new ``settings=`` / ``preset=`` kwargs."""
        variant = tmp_path / "both_legacy.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)
        s = NoiseSettings()
        with pytest.raises(ValueError, match=r"legacy persistence contract"):
            stage2_impl.compute_noise_estimation_impl(
                str(variant), settings=s, from_saved_params=True,
            )


class TestPersistedLayerInherit:
    """A no-kwargs follow-up call must inherit the previously resolved
    settings from the persisted ``processing_parameters/stage2_noise`` block."""

    def test_estimate_noise_inherits_persisted_smoothing(
        self,
        baseline_2638_stage1: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.noise_settings_serialization import (
            save_noise_settings_to_h5,
        )

        variant = tmp_path / "inherit.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)

        persisted = NoiseSettings()
        persisted.smoothing.smoothing_window_mhz = 123.0
        save_noise_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept()
        monkeypatch.setattr(stage2_impl, "estimate_noise_adaptive", mock)

        with pytest.raises(ValueError, match=r"intercepted"):
            stage2_impl.compute_noise_estimation_impl(str(variant))

        assert captured["kwargs"]["smoothing_window_mhz"] == 123.0, (
            "no-kwargs follow-up did not inherit the persisted "
            "smoothing_window_mhz; the persisted layer of the resolver "
            "is misrouted"
        )

    def test_from_saved_params_uses_legacy_block_not_resolver(
        self,
        baseline_2638_stage1: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``from_saved_params=True`` reads the legacy ``noise_estimation``
        block; the new ``stage2_noise`` block is ignored under that flag."""
        from ftmwpipeline._internal.stage2_impl import save_noise_parameters_impl
        from ftmwpipeline.io.noise_settings_serialization import (
            save_noise_settings_to_h5,
        )

        variant = tmp_path / "legacy_vs_resolver.ftmw"
        shutil.copyfile(baseline_2638_stage1, variant)

        # Legacy block carries one value, new block carries a different value.
        save_noise_parameters_impl(
            str(variant), {"smoothing_window_mhz": 50.0},
        )
        new_block = NoiseSettings()
        new_block.smoothing.smoothing_window_mhz = 200.0
        save_noise_settings_to_h5(str(variant), new_block)

        mock, captured = _intercept()
        monkeypatch.setattr(stage2_impl, "estimate_noise_adaptive", mock)

        with pytest.raises(ValueError, match=r"intercepted"):
            stage2_impl.compute_noise_estimation_impl(
                str(variant), from_saved_params=True,
            )

        assert captured["kwargs"]["smoothing_window_mhz"] == 50.0, (
            "from_saved_params=True should read the legacy 'noise_estimation' "
            "block, not the new 'stage2_noise' resolver-persisted block"
        )
