"""Integration: every consumed StageFitSettings field actually drives the fit.

The Stage 5 settings architecture exposes ~30 knobs spread across the
``StageFitSettings`` sub-dataclasses (``tau``, ``conservative``, ``seeder``,
``penalties``, ``rescue``, ``thaw``). Most of them have ``DEFAULT_*``
fallbacks inside the fitting modules, so a field that the driver fails to
forward silently falls back to the hard default with no visible error.
That makes it easy to ship phantom fields: the dataclass / YAML accept the
value and the persisted ``stage5_fit`` record carries it, but the fit
ignores it.

These tests run the driver up to its ``execute_plan`` call and intercept the
kwargs the planner would receive. For each StageFitSettings field the driver
is supposed to forward, the test sets a sentinel value and asserts the
sentinel appears in the captured ``conservative_kwargs`` (or
``rescue_kwargs``). That covers every routed field cheaply -- only one Stage
5 startup happens per case (session-scoped baseline + an immediate raise
inside the mock).

The headline τ-penalty λ knob also gets one real-fit integration check
that runs the full pipeline at two extremes and confirms the persisted
``reduced_chi2`` differs -- this is the slow end-to-end guarantee that
the kwargs route through to the LSQ, not just through to the planner.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage5_impl  # noqa: F401  -- monkeypatch target
from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.core.stage_fit_settings import (
    ShapeSpec,
    SpurSubSettings,
    StageFitSettings,
)
from ftmwpipeline.fitting.spur_detection import SpurSet

pytestmark = [
    pytest.mark.integration,
    # Propagation tests deliberately exercise the legacy per-knob kwarg
    # path; suppress the expected deprecation noise.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


# ---------------------------------------------------------------------------
# Mock-based: capture conservative_kwargs / rescue_kwargs and assert per field
# ---------------------------------------------------------------------------
class _PlanIntercepted(RuntimeError):
    """Sentinel raised from the mocked execute_plan to stop the driver."""


def _intercept_execute_plan() -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """Return (mock_fn, captured_kwargs_dict).

    The mock raises :class:`_PlanIntercepted` so the driver halts after the
    one call, leaving the captured kwargs available for inspection.
    """
    captured: Dict[str, Any] = {}

    def _fake_execute_plan(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _PlanIntercepted("intercepted")

    return _fake_execute_plan, captured


def _base_settings(shape: PeakShape = PeakShape.LORENTZIAN) -> StageFitSettings:
    return StageFitSettings(shape=ShapeSpec(kind=shape))


# Each propagation case: (field path, setter, sentinel_value, captured_key,
# kwargs_bag). ``kwargs_bag`` is "conservative" or "rescue" — which
# bag the driver should forward the value into.
def _t(sub: str, field: str) -> Callable[[StageFitSettings, Any], None]:
    def setter(s: StageFitSettings, v: Any) -> None:
        setattr(getattr(s, sub), field, v)
    return setter


PROPAGATION_FIELDS: list[tuple[str, Callable[..., None], Any, str, str]] = [
    # (description, setter, sentinel_value, captured_key, bag)
    ("tau.tau_penalty_lambda", _t("tau", "tau_penalty_lambda"), 12345.0,
     "tau_penalty_lambda", "conservative"),
    ("tau.tau_penalty_n_sigma", _t("tau", "tau_penalty_n_sigma"), 9.5,
     "tau_penalty_n_sigma", "conservative"),
    ("tau.max_decay_factor", _t("tau", "max_decay_factor"), 7.5,
     "max_decay_factor", "conservative"),
    ("conservative.significance", _t("conservative", "significance"), 0.123,
     "significance", "conservative"),
    ("conservative.max_peaks", _t("conservative", "max_peaks"), 17,
     "max_peaks", "conservative"),
    ("conservative.patience", _t("conservative", "patience"), 4,
     "patience", "conservative"),
    ("conservative.min_separation_factor",
     _t("conservative", "min_separation_factor"), 1.7,
     "min_separation_factor", "conservative"),
    ("conservative.min_pair_separation_factor",
     _t("conservative", "min_pair_separation_factor"), 0.77,
     "min_pair_separation_factor", "conservative"),
    ("conservative.weak_window_snr_threshold",
     _t("conservative", "weak_window_snr_threshold"), 33.0,
     "weak_window_snr_threshold", "conservative"),
    ("conservative.n_eff_kind",
     _t("conservative", "n_eff_kind"), "ess_lp_log1p_snr",
     "n_eff_kind", "conservative"),
    ("seeder.seeder_rchi2", _t("seeder", "seeder_rchi2"), 2.5,
     "seeder_rchi2_threshold", "conservative"),
    ("seeder.seeder_straddle_factor",
     _t("seeder", "seeder_straddle_factor"), 0.65,
     "seeder_straddle_factor", "conservative"),
    ("seeder.seeder_max_k", _t("seeder", "seeder_max_k"), 5,
     "seeder_max_k", "conservative"),
    ("penalties.phase_penalty_lambda",
     _t("penalties", "phase_penalty_lambda"), 999.0,
     "phase_penalty_lambda", "conservative"),
    ("penalties.phase_penalty_cutoff_fwhm",
     _t("penalties", "phase_penalty_cutoff_fwhm"), 3.5,
     "phase_penalty_cutoff_fwhm", "conservative"),
    ("penalties.amp_penalty_lambda",
     _t("penalties", "amp_penalty_lambda"), 25.0,
     "amp_penalty_lambda", "conservative"),
    ("penalties.amp_max_headroom",
     _t("penalties", "amp_max_headroom"), 6.0,
     "amp_max_headroom", "conservative"),
    # Rescue knobs: cleanup_significance feeds both rescue_significance and
    # knockout_significance (they share the same dataclass field today).
    ("rescue.cleanup_significance",
     _t("rescue", "cleanup_significance"), 0.0123,
     "rescue_significance", "rescue"),
    ("rescue.cleanup_significance",
     _t("rescue", "cleanup_significance"), 0.0123,
     "knockout_significance", "rescue"),
    ("rescue.merge_separation_factor",
     _t("rescue", "merge_separation_factor"), 1.25,
     "merge_separation_factor", "rescue"),
    ("rescue.structural_merge_factor",
     _t("rescue", "structural_merge_factor"), 0.13,
     "structural_merge_factor", "rescue"),
]


@pytest.mark.parametrize(
    "label, setter, value, key, bag",
    PROPAGATION_FIELDS,
    ids=[f"{c[0]}->{c[4]}.{c[3]}" for c in PROPAGATION_FIELDS],
)
def test_setting_field_reaches_planner(
    baseline_2638_stage4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    setter: Callable[..., None],
    value: Any,
    key: str,
    bag: str,
) -> None:
    """Each StageFitSettings field forwards into the right kwarg bag.

    Set the dataclass field to a sentinel; the driver's call to
    ``execute_plan`` must carry that sentinel in either ``conservative_kwargs``
    or ``rescue_kwargs``. A test that fails on "key missing" indicates a
    phantom dataclass field (the driver doesn't route it).
    """
    variant = tmp_path / f"propagation_{key}.ftmw"
    shutil.copyfile(baseline_2638_stage4, variant)

    mock, captured = _intercept_execute_plan()
    monkeypatch.setattr(stage5_impl, "execute_plan", mock)

    s = _base_settings()
    setter(s, value)

    # Call ``fit_peaks_impl`` directly so the mock's sentinel propagates --
    # the Pipeline / api wrappers re-raise as ``RuntimeError`` and would
    # swallow the sentinel's type.
    with pytest.raises(_PlanIntercepted):
        stage5_impl.fit_peaks_impl(str(variant), settings=s)

    bag_kwarg = f"{bag}_kwargs"
    assert bag_kwarg in captured, (
        f"execute_plan was not given a {bag_kwarg} argument; the driver "
        f"changed shape since this test was written."
    )
    bag_dict = captured[bag_kwarg]
    if bag_dict is None and bag == "rescue":
        pytest.fail(
            f"{bag_kwarg} is None -- rescue rounds may be disabled in the "
            f"resolved settings (defaults set max_rounds > 0, so this "
            f"shouldn't happen on a fresh fit). Field: {label}"
        )
    assert key in bag_dict, (
        f"{label}: dataclass value {value!r} was set but the driver did "
        f"not forward {key!r} into {bag_kwarg}; phantom field detected."
    )
    assert bag_dict[key] == value, (
        f"{label}: forwarded value mismatch -- expected {value!r}, got "
        f"{bag_dict[key]!r} in {bag_kwarg}[{key!r}]"
    )


# ---------------------------------------------------------------------------
# Spur masking: settings reach the driver as a SpurSet (or None when off)
# ---------------------------------------------------------------------------
def test_spur_settings_build_spur_set(
    baseline_2638_stage4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spur.enabled=True`` makes the driver pass a built SpurSet to the plan.

    The mask half-width knob must survive into the SpurSet geometry.
    """
    variant = tmp_path / "spur_on.ftmw"
    shutil.copyfile(baseline_2638_stage4, variant)

    mock, captured = _intercept_execute_plan()
    monkeypatch.setattr(stage5_impl, "execute_plan", mock)

    s = _base_settings()
    s.spur = SpurSubSettings(enabled=True, mask_half_width_bins=3)
    with pytest.raises(_PlanIntercepted):
        stage5_impl.fit_peaks_impl(str(variant), settings=s)

    assert "spur_set" in captured
    spur_set = captured["spur_set"]
    # The propagation contract is that enabling builds a SpurSet carrying the
    # configured geometry; the gated *count* depends on the data (an apodized
    # fixture smears spurs below the narrowness gate) and is validated
    # separately on the unapodized fixture.
    assert isinstance(spur_set, SpurSet)
    assert spur_set.mask_half_width_bins == 3


def test_spur_disabled_passes_none(
    baseline_2638_stage4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spur.enabled=False`` passes ``spur_set=None`` (masking off)."""
    variant = tmp_path / "spur_off.ftmw"
    shutil.copyfile(baseline_2638_stage4, variant)

    mock, captured = _intercept_execute_plan()
    monkeypatch.setattr(stage5_impl, "execute_plan", mock)

    s = _base_settings()
    s.spur = SpurSubSettings(enabled=False)
    with pytest.raises(_PlanIntercepted):
        stage5_impl.fit_peaks_impl(str(variant), settings=s)

    assert captured.get("spur_set", "MISSING") is None


# ---------------------------------------------------------------------------
# End-to-end smoke: λ at two extremes must produce different persisted fits
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tau_penalty_lambda_drives_real_fit(
    baseline_2638_stage4: Path,
    tmp_path: Path,
) -> None:
    """The τ-penalty λ sweep needs the kwarg to actually reach the LSQ.

    Two extremes (λ=0 disables the penalty, λ=10000 over-constrains τ).
    With wiring intact, at least one window must show a different
    ``reduced_chi2`` between the two persisted fits. Mock-based tests
    above confirm the kwarg reaches the planner; this test guarantees the
    planner forwards it through to the inner LSQ.
    """
    def _fit(tag: str, lam: float) -> list[tuple[int, float]]:
        variant = tmp_path / f"{tag}.ftmw"
        shutil.copyfile(baseline_2638_stage4, variant)
        s = _base_settings()
        s.tau.tau_penalty_lambda = lam
        ftmw.fit_peaks(str(variant), settings=s)
        rows: list[tuple[int, float]] = []
        with h5py.File(variant, "r") as h5:
            windows = h5["/stage5_fitting/windows"]
            for name in sorted(windows.keys()):
                wg = windows[name]
                rows.append(
                    (int(wg.attrs["window_id"]), float(wg.attrs["reduced_chi2"]))
                )
        return rows

    rows_low = _fit("lam_0", 0.0)
    rows_high = _fit("lam_10000", 10000.0)

    by_low = dict(rows_low)
    by_high = dict(rows_high)
    common = set(by_low) & set(by_high)
    diffs = sum(1 for wid in common if by_low[wid] != by_high[wid])
    assert diffs > 0, (
        f"varying tau.tau_penalty_lambda from 0 to 10000 produced "
        f"bit-identical fits across {len(common)} windows -- the LSQ is "
        f"not receiving the kwarg"
    )


# ---------------------------------------------------------------------------
# Per-band τ₀: every window must seed at its band's τ_maj, not the band-wide
# ---------------------------------------------------------------------------
def test_per_band_tau_routes_tau0_per_window(
    baseline_2638_stage4_small: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-band routing forwards each window's band-local τ_maj into ``tau0_us``.

    Stage 2b τ_G stamps three per-band majorities on the 2638 fixture; the
    band-low majority differs from the band-wide value, so a per-window
    fixed-τ window (``fit_tau=False``) in the low band that seeded at the
    band-wide ``tau_maj_us`` would land on the wrong τ. The fix routes the
    per-band ``tau_maj_us`` into the per-window ``tau0_us`` seed; this test
    intercepts the first ``_fit_one_window`` call and asserts the kwarg
    matches the band-local τ_maj rather than the band-wide one.
    """
    import shutil

    from ftmwpipeline.fitting import plan_execution
    from ftmwpipeline._internal.stage2b_g_impl import (
        calibrate_tau_G_impl,
        load_tau_G_calibration_impl,
    )

    variant = tmp_path / "per_band_tau0.ftmw"
    shutil.copyfile(baseline_2638_stage4_small, variant)
    # Stage 2b τ_G must be present for per-band routing to activate.
    calibrate_tau_G_impl(str(variant))
    tc = load_tau_G_calibration_impl(str(variant))["tau_G_calibration"]
    assert tc.band_majorities, "Stage 2b τ_G did not produce band_majorities"

    captured: Dict[str, Any] = {}

    def _fake_fit_one_window(win, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["window_id"] = int(win.window_id)
        captured["window_freq_lo"] = float(win.freq_range[0])
        captured["window_freq_hi"] = float(win.freq_range[1])
        captured["tau0_us"] = float(kwargs["tau0_us"])
        captured["conservative_kwargs"] = dict(kwargs["conservative_kwargs"])
        raise _PlanIntercepted("intercepted on first window")

    monkeypatch.setattr(plan_execution, "_fit_one_window", _fake_fit_one_window)

    s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
    s.tau.per_band_tau = True
    with pytest.raises(_PlanIntercepted):
        stage5_impl.fit_peaks_impl(str(variant), settings=s)

    # The captured window's centre frequency tells us which band it sits in.
    centre = 0.5 * (captured["window_freq_lo"] + captured["window_freq_hi"])
    matching_band = None
    for band in tc.band_majorities:
        if band.freq_lo_mhz <= centre < band.freq_hi_mhz:
            matching_band = band
            break
    assert matching_band is not None, (
        f"window centre {centre} MHz is outside every band majority "
        f"({[(b.label, b.freq_lo_mhz, b.freq_hi_mhz) for b in tc.band_majorities]})"
    )
    assert captured["tau0_us"] == pytest.approx(matching_band.tau_maj_us), (
        f"window {captured['window_id']} (centre {centre:.1f} MHz, "
        f"band {matching_band.label}) saw tau0_us={captured['tau0_us']:.3f} "
        f"but the band-local τ_maj is {matching_band.tau_maj_us:.3f}. "
        f"Band-wide τ_maj is {tc.tau_maj_us:.3f} -- if those match, the "
        f"per-band τ₀ routing did not fire."
    )
    # Sanity: assert conservative_kwargs's tau_maj_us was also routed.
    assert captured["conservative_kwargs"]["tau_maj_us"] == pytest.approx(
        matching_band.tau_maj_us
    )
