"""The Stage 5 fit plots draw the shared model evaluator (fitting.model_eval).

``fit_visualization`` used to carry its own model, which ignored the line shape
and the baseline. Both of its figures now route through ``model_eval``; the
persisted-grid amplitude scale and phase ramp are a separate, explicit step.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FittingResult
from ftmwpipeline.fitting import model_eval
from ftmwpipeline.visualization import fit_visualization as fv

pytestmark = [pytest.mark.unit]

CENTER = 20000.0


def _fit():
    wf = FittingResult(window_id=1, shape="gaussian")
    wf.window = SimpleNamespace(freq_range=(CENTER - 0.5, CENTER + 0.5))
    wf.shared_parameters = {"tau_us": {"value": 4.0}}
    wf.fitted_peaks = [
        SimpleNamespace(
            frequency_mhz=CENTER + 0.05, amplitude=1.0, phase=0.3, peak_uid=7
        )
    ]
    wf.quality_metrics = {
        "baseline_applied": 1.0,
        "baseline_order": 0,
        "baseline_offset_scale": 0.5,
        "baseline_coeff0_re": 0.2,
        "baseline_coeff0_im": -0.1,
    }
    return SimpleNamespace(window_fits=[wf])


def test_overview_model_is_the_spectrum_model_then_the_frame_step():
    fit = _fit()
    f = CENTER + np.linspace(-1.0, 1.0, 201)
    ramp = np.exp(1j * np.linspace(0.0, 1.0, f.size))
    got = fv._window_model_on_persisted_grid(f, fit, "upper", 12.0, 3.0, ramp)
    expect = (
        model_eval.evaluate_spectrum_model(
            f, fit, acquisition_us=12.0, sideband="upper"
        )
        * 3.0
        * ramp
    )
    np.testing.assert_array_equal(got, expect)


def test_no_private_model_left_in_the_plots():
    src = inspect.getsource(fv)
    assert "model_spectrum(" not in src
    assert "ModelPeak(" not in src
    detail = inspect.getsource(fv._plot_per_window_detail)
    assert "evaluate_window_terms(" in detail
