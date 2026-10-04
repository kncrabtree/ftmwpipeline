"""Unit tests for the shared window-model evaluator (fitting.model_eval)."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FittingResult
from ftmwpipeline.fitting import model_eval
from ftmwpipeline.fitting.model_eval import (
    evaluate_window_baseline,
    evaluate_window_model,
    fitted_model_peaks,
    frozen_model_peaks,
    window_model_peaks,
)
from ftmwpipeline.visualization import fit_detail

CENTER = 20000.0
TAU = 5.0
ACQ = 12.0


def _fit(frozen=True, baseline=False):
    wf = FittingResult(window_id=1, shape="lorentzian")
    wf.fitted_peaks = [
        SimpleNamespace(frequency_mhz=CENTER + 0.05, amplitude=1.0, phase=0.3),
        SimpleNamespace(frequency_mhz=CENTER - 0.10, amplitude=0.5, phase=None),
    ]
    if frozen:
        wf.fixed_parameters = {
            "frozen_peak_0": {
                "frequency_mhz": CENTER + 0.4,
                "amplitude": 2.0,
                "phase": -0.2,
            },
            "other_key": {"frequency_mhz": 0.0, "amplitude": 9.0},
        }
    if baseline:
        wf.quality_metrics = {
            "baseline_applied": 1.0,
            "baseline_order": 1,
            "baseline_offset_scale": 0.5,
            "baseline_coeff0_re": 0.1,
            "baseline_coeff0_im": -0.2,
            "baseline_coeff1_re": 0.3,
            "baseline_coeff1_im": 0.4,
        }
    return wf


GRID = CENTER + np.linspace(-0.6, 0.6, 241)


def _eval(wf, peaks, sideband="upper", freqs=GRID):
    return evaluate_window_model(freqs, wf, peaks, TAU, ACQ, sideband, CENTER, wf.shape)


def test_plots_use_the_shared_evaluator():
    assert fit_detail.evaluate_window_model is model_eval.evaluate_window_model
    assert fit_detail.window_model_peaks is model_eval.window_model_peaks
    for name in ("_eval_window_baseline", "_window_model_peaks", "_eval_model"):
        assert not hasattr(fit_detail, name)
    src = inspect.getsource(fit_detail.prepare_window_panels)
    assert "window_model_peaks(" in src and "evaluate_window_model(" in src


def test_peak_order_fitted_then_frozen():
    wf = _fit()
    fitted = fitted_model_peaks(wf, "upper", CENTER)
    frozen = frozen_model_peaks(wf, "upper", CENTER)
    assert len(fitted) == 2 and len(frozen) == 1
    assert window_model_peaks(wf, "upper", CENTER) == fitted + frozen
    assert fitted[0].offset_mhz == pytest.approx(0.05)
    assert fitted[1].phase == 0.0  # missing phase -> 0
    assert frozen[0].offset_mhz == pytest.approx(0.4)
    assert frozen[0].amplitude == 2.0


def test_lower_sideband_flips_offsets():
    wf = _fit()
    up = window_model_peaks(wf, "upper", CENTER)
    lo = window_model_peaks(wf, "lower", CENTER)
    for a, b in zip(up, lo):
        assert b.offset_mhz == pytest.approx(-a.offset_mhz)


def test_no_frozen_gives_empty_frozen_list():
    assert frozen_model_peaks(_fit(frozen=False), "upper", CENTER) == []


def test_baseline_zero_when_not_applied():
    out = evaluate_window_baseline(_fit(), np.linspace(-1, 1, 5))
    assert out.dtype == np.complex128 and not out.any()


def test_baseline_polynomial_value():
    u = np.array([-0.5, 0.0, 0.25])
    out = evaluate_window_baseline(_fit(baseline=True), u)
    x = u / 0.5
    expect = (0.1 - 0.2j) + (0.3 + 0.4j) * x
    np.testing.assert_allclose(out, expect, rtol=0, atol=1e-15)


def test_model_is_complex128_and_baseline_only_without_peaks():
    wf = _fit(baseline=True)
    m = _eval(wf, [])
    assert m.dtype == np.complex128
    u = GRID - CENTER
    np.testing.assert_array_equal(m, evaluate_window_baseline(wf, u))


def test_no_double_counting_components_sum_to_model():
    wf = _fit()
    peaks = window_model_peaks(wf, "upper", CENTER)
    total = _eval(wf, peaks)
    parts = sum(_eval(wf, [p]) for p in peaks)
    np.testing.assert_allclose(total, parts, rtol=0, atol=1e-12 * abs(total).max())
    fitted_only = _eval(wf, fitted_model_peaks(wf, "upper", CENTER))
    fixed = _eval(wf, frozen_model_peaks(wf, "upper", CENTER))
    np.testing.assert_allclose(
        total, fitted_only + fixed, rtol=0, atol=1e-12 * abs(total).max()
    )
    assert abs(fixed).max() > 0


def test_model_is_deterministic():
    wf = _fit(baseline=True)
    peaks = window_model_peaks(wf, "upper", CENTER)
    np.testing.assert_array_equal(_eval(wf, peaks), _eval(wf, peaks))


# ---- term-by-term evaluation and the whole-spectrum model ----------------


def _windowed(wid, center, half, *, tau=TAU, peaks=(), baseline=None, uid0=100):
    wf = FittingResult(window_id=wid, shape="lorentzian")
    wf.window = SimpleNamespace(freq_range=(center - half, center + half))
    wf.shared_parameters = {"tau_us": {"value": tau}}
    wf.fitted_peaks = [
        SimpleNamespace(
            frequency_mhz=center + off, amplitude=amp, phase=0.1, peak_uid=uid0 + i
        )
        for i, (off, amp) in enumerate(peaks)
    ]
    if baseline is not None:
        wf.quality_metrics = {
            "baseline_applied": 1.0,
            "baseline_order": 0,
            "baseline_offset_scale": half,
            "baseline_coeff0_re": baseline,
            "baseline_coeff0_im": 0.0,
        }
    return wf


def test_window_terms_sum_to_the_drawn_model():
    wf = _fit(baseline=True)
    wf.window = SimpleNamespace(freq_range=(CENTER - 0.6, CENTER + 0.6))
    wf.shared_parameters = {"tau_us": {"value": TAU}}
    terms = model_eval.evaluate_window_terms(
        GRID, wf, acquisition_us=ACQ, sideband="upper", components=True
    )
    drawn = _eval(wf, window_model_peaks(wf, "upper", CENTER))
    np.testing.assert_array_equal(terms.model, drawn)
    total = sum(terms.components) + terms.fixed + terms.baseline
    np.testing.assert_allclose(total, drawn, rtol=0, atol=1e-12 * abs(drawn).max())
    # The frozen term is drawn at the window's own fitted tau.
    np.testing.assert_array_equal(
        terms.fixed, _eval(_fit(), frozen_model_peaks(wf, "upper", CENTER))
    )


def test_window_terms_baseline_absent_without_one():
    wf = _fit()
    wf.window = SimpleNamespace(freq_range=(CENTER - 0.6, CENTER + 0.6))
    wf.shared_parameters = {"tau_us": {"value": TAU}}
    terms = model_eval.evaluate_window_terms(
        GRID, wf, acquisition_us=ACQ, sideband="upper"
    )
    assert terms.baseline is None and terms.components is None


def test_spectrum_model_baselines_stay_inside_their_windows():
    a = _windowed(1, CENTER - 0.4, 0.3, baseline=1.0)
    b = _windowed(2, CENTER + 0.4, 0.3, baseline=10.0)
    fit = SimpleNamespace(window_fits=[a, b])
    freqs = CENTER + np.linspace(-1.0, 1.0, 401)
    m = model_eval.evaluate_spectrum_model(
        freqs, fit, acquisition_us=ACQ, sideband="upper"
    )
    in_a = (freqs >= a.window.freq_range[0]) & (freqs <= a.window.freq_range[1])
    in_b = (freqs >= b.window.freq_range[0]) & (freqs <= b.window.freq_range[1])
    np.testing.assert_array_equal(m[in_a], 1.0)
    np.testing.assert_array_equal(m[in_b], 10.0)
    assert not m[~(in_a | in_b)].any()


def test_spectrum_model_overlap_takes_the_nearest_centre():
    a = _windowed(1, CENTER - 0.2, 0.3, baseline=1.0)
    b = _windowed(2, CENTER + 0.2, 0.3, baseline=10.0)
    fit = SimpleNamespace(window_fits=[a, b])
    freqs = np.array([CENTER - 0.05, CENTER + 0.05])
    m = model_eval.evaluate_spectrum_model(
        freqs, fit, acquisition_us=ACQ, sideband="upper"
    )
    np.testing.assert_array_equal(m, [1.0, 10.0])


def test_spectrum_model_evaluates_a_shared_line_once():
    a = _windowed(1, CENTER, 0.3, peaks=[(0.05, 1.0)])
    b = _windowed(2, CENTER + 0.6, 0.3, peaks=[(-0.55, 1.0)])  # same uid 100
    assert a.fitted_peaks[0].peak_uid == b.fitted_peaks[0].peak_uid
    fit = SimpleNamespace(window_fits=[a, b])
    assert model_eval.spectrum_line_owners(fit) == {100: 1}
    one = SimpleNamespace(window_fits=[a])
    freqs = CENTER + np.linspace(-1.0, 1.0, 101)
    kw = dict(acquisition_us=ACQ, sideband="upper")
    np.testing.assert_array_equal(
        model_eval.evaluate_spectrum_model(freqs, fit, **kw),
        model_eval.evaluate_spectrum_model(freqs, one, **kw),
    )
