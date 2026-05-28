"""Integration: every consumed PeakDetectionSettings field reaches the kernel.

Stage 3 spreads its ~14 knobs across four sub-dataclasses
(``promotion``, ``savgol``, ``primary_pass``, ``gap_pass``). Most of the
fields would fall through to either module-level constants or function-
signature defaults if the orchestrator failed to forward them, so a missed
wiring would silently regress to those defaults with no visible error.

These tests intercept the relevant call sites from
``_internal/stage3_impl.detect_peaks_impl`` and assert that every routed
:class:`PeakDetectionSettings` field reaches its kernel's kwargs bag.
Some fields land directly on ``preprocessing.peak_detection.detect_peaks``;
others drive the orchestrator-internal helpers ``_spectrum_from_fid``,
``_mf_gap_spectrum``, ``_grid_aware_sg_window``, or the
``leakage_touched_intervals`` call. Each test mocks the right hook for
its field.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from ftmwpipeline._internal import stage3_impl
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings

pytestmark = [
    pytest.mark.integration,
    # Propagation tests deliberately exercise the legacy per-knob kwarg
    # path; suppress the expected deprecation noise.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


class _CalibIntercepted(RuntimeError):
    """Sentinel raised from a mocked kernel to stop the orchestrator."""


def _intercept_kernel() -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """Mock that captures kwargs and raises to short-circuit the run."""
    captured: Dict[str, Any] = {}

    def _fake_kernel(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        raise _CalibIntercepted("intercepted")

    return _fake_kernel, captured


def _spy(real_callable: Callable[..., Any]) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """Wrap a callable to capture its kwargs while delegating to the real impl."""
    captured: Dict[str, Any] = {}

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        captured.setdefault("calls", []).append(
            {"args": args, "kwargs": dict(kwargs)}
        )
        return real_callable(*args, **kwargs)

    return _wrapper, captured


def _sub_set(sub_name: str, field_name: str, value: Any) -> Callable[..., None]:
    def setter(s: PeakDetectionSettings) -> None:
        setattr(getattr(s, sub_name), field_name, value)
    return setter


# Fields that reach ``preprocessing.peak_detection.detect_peaks`` directly.
# (label, setter, expected_kernel_kwarg, expected_value)
KERNEL_FIELDS: list[tuple[str, Callable[..., None], str, Any]] = [
    # promotion.min_snr does not flow into detect_peaks (the orchestrator
    # passes ``min(internal_min_snr, promotion)`` as ``min_snr``); checked
    # separately below.
    ("promotion.weak_medium_snr",
     _sub_set("promotion", "weak_medium_snr", 12.0),
     "weak_medium_snr", 12.0),
    ("promotion.medium_strong_snr",
     _sub_set("promotion", "medium_strong_snr", 75.0),
     "medium_strong_snr", 75.0),
    ("savgol.sg_window",
     _sub_set("savgol", "sg_window", 9),
     "sg_window", 9),
    ("savgol.sg_order",
     _sub_set("savgol", "sg_order", 5),
     "sg_order", 5),
    ("primary_pass.min_exclusion_mhz",
     _sub_set("primary_pass", "min_exclusion_mhz", 0.5),
     "min_exclusion_mhz", 0.5),
    ("gap_pass.run_gap_pass",
     _sub_set("gap_pass", "run_gap_pass", False),
     "run_gap_pass", False),
]


@pytest.mark.parametrize(
    "label, setter, key, value",
    KERNEL_FIELDS,
    ids=[c[0] for c in KERNEL_FIELDS],
)
def test_detect_peaks_field_reaches_kernel(
    baseline_2638_stage2: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    key: str,
    value: Any,
) -> None:
    """Each direct-kernel field forwards into ``detect_peaks``'s kwargs."""
    variant = tmp_path / f"propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage2, variant)

    mock, captured = _intercept_kernel()
    monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

    s = PeakDetectionSettings()
    setter(s)

    with pytest.raises(_CalibIntercepted):
        stage3_impl.detect_peaks_impl(str(variant), settings=s)

    kwargs = captured["kwargs"]
    assert key in kwargs, (
        f"{label}: dataclass value {value!r} was set but the orchestrator "
        f"did not forward {key!r} to detect_peaks. Phantom field."
    )
    assert kwargs[key] == value, (
        f"{label}: forwarded value mismatch -- expected {value!r}, "
        f"got {kwargs[key]!r} as kernel kwarg {key!r}."
    )


class TestPromotionFlooring:
    """``promotion.min_snr`` and ``promotion.internal_min_snr`` together
    determine the kernel's ``min_snr`` floor as ``min(internal, promotion)``."""

    def test_promotion_min_snr_caps_kernel_floor(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "promotion_min_snr.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        mock, captured = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        # Set promotion below the default internal cap (2.0): kernel sees
        # the lower of the two = promotion.
        s.promotion.min_snr = 1.0
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["kwargs"]["min_snr"] == pytest.approx(1.0)

    def test_internal_min_snr_caps_kernel_floor(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "internal_min_snr.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        mock, captured = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        # Default promotion = 3.0; lower internal cap to 1.5. Kernel floor
        # is ``min(internal, promotion)`` = 1.5.
        s.promotion.internal_min_snr = 1.5
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["kwargs"]["min_snr"] == pytest.approx(1.5)


class TestPrimaryPassSpectrum:
    """``primary_pass.primary_window`` + ``primary_pass.detection_zpf`` drive
    ``_spectrum_from_fid``."""

    def test_primary_window_reaches_spectrum_from_fid(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "primary_window.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl._spectrum_from_fid)
        monkeypatch.setattr(stage3_impl, "_spectrum_from_fid", spy)
        # Stop the run after the spy fires (downstream noise estimation is
        # slow); intercept detect_peaks.
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.primary_pass.primary_window = "hann"
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        # _spectrum_from_fid is called once for the primary pass.
        assert captured["calls"], "spy never fired"
        primary_call = captured["calls"][0]
        assert primary_call["kwargs"]["window_function"] == "hann"

    def test_detection_zpf_reaches_spectrum_from_fid(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "detection_zpf.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl._spectrum_from_fid)
        monkeypatch.setattr(stage3_impl, "_spectrum_from_fid", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.primary_pass.detection_zpf = 0
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        primary_call = captured["calls"][0]
        assert primary_call["kwargs"]["zpf"] == 0


class TestGapPassSpectrum:
    """``gap_pass.gap_active_zpf`` drives ``_mf_gap_spectrum``."""

    def test_gap_active_zpf_reaches_mf_gap_spectrum(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "gap_active_zpf.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl._mf_gap_spectrum)
        monkeypatch.setattr(stage3_impl, "_mf_gap_spectrum", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.gap_pass.gap_active_zpf = 3
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["calls"], "spy never fired"
        assert captured["calls"][0]["kwargs"]["zpf_active"] == 3


class TestSavgolCoverage:
    """``savgol.sg_fwhm_coverage`` and ``savgol.sg_min_window`` drive
    ``_grid_aware_sg_window``."""

    def test_sg_fwhm_coverage_reaches_grid_aware(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "sg_fwhm_coverage.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl._grid_aware_sg_window)
        monkeypatch.setattr(stage3_impl, "_grid_aware_sg_window", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.savgol.sg_fwhm_coverage = 6.0
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["calls"], "spy never fired"
        assert captured["calls"][0]["kwargs"]["fwhm_coverage"] == 6.0

    def test_sg_min_window_reaches_grid_aware(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "sg_min_window.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl._grid_aware_sg_window)
        monkeypatch.setattr(stage3_impl, "_grid_aware_sg_window", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.savgol.sg_min_window = 9
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["calls"], "spy never fired"
        assert captured["calls"][0]["kwargs"]["min_window"] == 9


class TestLeakageThreshold:
    """``gap_pass.gap_mask_edge_threshold`` drives the
    ``leakage_touched_intervals`` call."""

    def test_gap_mask_edge_threshold_reaches_leakage(
        self, baseline_2638_stage2: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        variant = tmp_path / "gap_mask_threshold.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        spy, captured = _spy(stage3_impl.leakage_touched_intervals)
        monkeypatch.setattr(stage3_impl, "leakage_touched_intervals", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        s = PeakDetectionSettings()
        s.gap_pass.gap_mask_edge_threshold = 5.5
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant), settings=s)
        assert captured["calls"], "spy never fired"
        assert captured["calls"][0]["kwargs"]["threshold"] == 5.5


class TestMutualExclusion:
    """Passing both ``settings=`` and ``preset=`` to ``detect_peaks_impl``
    must raise ``ValueError``, matching Stages 5, 2b, and 2."""

    def test_settings_and_preset_both_raises(
        self, baseline_2638_stage2: Path, tmp_path: Path,
    ) -> None:
        variant = tmp_path / "both.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        s = PeakDetectionSettings()
        with pytest.raises(ValueError, match=r"mutually|alternative"):
            stage3_impl.detect_peaks_impl(
                str(variant), settings=s, preset="instrument_bc_2638",
            )


class TestGapPassTauFeeder:
    """Shape-aware τ-feeder for the gap-pass matched filter.

    The gap-pass matched-filter ``tau_basis_us`` precedence is:

      1. Stage 2b Gaussian twin ``tau_G_maj`` when ``recommended_shape``
         is ``'gaussian'`` and the twin is present.
      2. Stage 2b Lorentzian ``tau_maj`` when available.
      3. Stage 1 user apodization ``expf_us`` for the pre-calibration path.
      4. Historical 5.0 µs default.

    These tests pin each layer of the precedence with monkeypatched
    helper functions and verify the right value reaches
    ``_mf_gap_spectrum``.
    """

    @staticmethod
    def _patch_tau_lookup(
        monkeypatch: pytest.MonkeyPatch,
        *,
        tau_maj_us: Any = None,
        tau_G_maj_us: Any = None,
        recommended_shape: Any = None,
    ) -> None:
        """Pin Stage 2b loader results without materialising HDF5 groups.

        Each ``None`` means "no calibration present"; a float means
        ``tau_calibration_present`` returns True and ``load_..._impl``
        returns an object whose ``tau_maj_us`` attribute is that value.
        """
        class _Stub:
            def __init__(self, value: float) -> None:
                self.tau_maj_us = value

        monkeypatch.setattr(
            stage3_impl,
            "tau_calibration_present",
            lambda fp: tau_maj_us is not None,
        )
        monkeypatch.setattr(
            stage3_impl,
            "tau_G_calibration_present",
            lambda fp: tau_G_maj_us is not None,
        )
        monkeypatch.setattr(
            stage3_impl,
            "read_stage2b_recommended_shape",
            lambda fp: recommended_shape,
        )
        if tau_maj_us is not None:
            monkeypatch.setattr(
                stage3_impl,
                "load_tau_calibration_impl",
                lambda fp: {"tau_calibration": _Stub(float(tau_maj_us))},
            )
        if tau_G_maj_us is not None:
            monkeypatch.setattr(
                stage3_impl,
                "load_tau_G_calibration_impl",
                lambda fp: {"tau_G_calibration": _Stub(float(tau_G_maj_us))},
            )

    def _run_with_spy(
        self,
        variant: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Dict[str, Any]:
        spy, captured = _spy(stage3_impl._mf_gap_spectrum)
        monkeypatch.setattr(stage3_impl, "_mf_gap_spectrum", spy)
        mock, _ = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)
        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant))
        assert captured["calls"], "spy never fired"
        return captured["calls"][0]["kwargs"]

    def test_gaussian_recommended_uses_tau_G_maj(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recommended Gaussian + twin present → τ_G_maj."""
        variant = tmp_path / "tau_feed_gaussian.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        self._patch_tau_lookup(
            monkeypatch,
            tau_maj_us=6.4,
            tau_G_maj_us=7.7,
            recommended_shape="gaussian",
        )
        kwargs = self._run_with_spy(variant, monkeypatch)
        assert kwargs["tau_basis_us"] == 7.7

    def test_lorentzian_recommended_uses_tau_maj(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recommended Lorentzian (or no recommendation) + Lorentzian
        twin present → τ_maj, even when Gaussian twin also exists."""
        variant = tmp_path / "tau_feed_lorentzian.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        self._patch_tau_lookup(
            monkeypatch,
            tau_maj_us=6.4,
            tau_G_maj_us=7.7,
            recommended_shape="lorentzian",
        )
        kwargs = self._run_with_spy(variant, monkeypatch)
        assert kwargs["tau_basis_us"] == 6.4

    def test_no_recommendation_uses_tau_maj(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No recommended_shape stamp → falls through to Lorentzian τ_maj
        even when the Gaussian twin is present. Preserves the pre-Phase-A
        behaviour for files without an auto-recommend pass."""
        variant = tmp_path / "tau_feed_no_rec.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        self._patch_tau_lookup(
            monkeypatch,
            tau_maj_us=6.4,
            tau_G_maj_us=7.7,
            recommended_shape=None,
        )
        kwargs = self._run_with_spy(variant, monkeypatch)
        assert kwargs["tau_basis_us"] == 6.4

    def test_gaussian_recommended_without_twin_falls_back(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recommended Gaussian but no Gaussian twin → falls through
        to Lorentzian τ_maj rather than raising."""
        variant = tmp_path / "tau_feed_gauss_no_twin.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)
        self._patch_tau_lookup(
            monkeypatch,
            tau_maj_us=6.4,
            tau_G_maj_us=None,
            recommended_shape="gaussian",
        )
        kwargs = self._run_with_spy(variant, monkeypatch)
        assert kwargs["tau_basis_us"] == 6.4


class TestPersistedLayerInherit:
    """A no-kwargs follow-up call must inherit the previously resolved
    settings from the persisted ``processing_parameters/stage3_peaks`` block."""

    def test_detect_peaks_inherits_persisted_sg_window(
        self,
        baseline_2638_stage2: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ftmwpipeline.io.peak_detection_settings_serialization import (
            save_peak_detection_settings_to_h5,
        )

        variant = tmp_path / "inherit.ftmw"
        shutil.copyfile(baseline_2638_stage2, variant)

        persisted = PeakDetectionSettings()
        persisted.savgol.sg_window = 17
        save_peak_detection_settings_to_h5(str(variant), persisted)

        mock, captured = _intercept_kernel()
        monkeypatch.setattr(stage3_impl, "detect_peaks", mock)

        with pytest.raises(_CalibIntercepted):
            stage3_impl.detect_peaks_impl(str(variant))

        assert captured["kwargs"]["sg_window"] == 17, (
            "no-kwargs follow-up did not inherit the persisted "
            "sg_window; the persisted layer of the resolver is misrouted"
        )
