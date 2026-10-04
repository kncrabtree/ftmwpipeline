"""Typed setting refusals of the fit, fit-show and window-creation calls (Wave 4).

Built on the session-scoped few-window Stage 5 baseline (copied, never mutated).
"""

import shutil

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import load_fit_impl
from ftmwpipeline.core.stage_fit_settings import StageFitSettings, TauSubSettings
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


@pytest.fixture
def fit_file(baseline_2638_stage5_small, tmp_path):
    fp = tmp_path / "fit.ftmw"
    shutil.copy(baseline_2638_stage5_small, fp)
    return str(fp)


def _assert_bad_setting(exc, path):
    err = exc.value
    assert err.path == path
    assert isinstance(err, ValueError)  # the built-in it replaced still catches
    d = err.to_dict()
    assert d["code"] == "bad_setting"
    assert d["path"] == path


@pytest.mark.parametrize(
    "kwargs, path",
    [
        # Mutation: ShapeSpec.coerce's plain ValueError escapes unwrapped.
        ({"shape": "bogus"}, "shape"),
        # Mutation: the pair check raises a plain ValueError.
        ({"tau_maj_override_us": 3.0}, "tau_maj_override_us"),
        (
            {"tau_maj_override_us": 3.0, "sigma_tau_override_us": -1.0},
            "sigma_tau_override_us",
        ),
        (
            {"settings": StageFitSettings(tau=TauSubSettings(tau0_us=-1.0))},
            "stage5.tau.tau0_us",
        ),
        (
            {"settings": StageFitSettings(tau=TauSubSettings(max_decay_factor=0.5))},
            "stage5.tau.max_decay_factor",
        ),
    ],
)
@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_fit_peaks_bad_setting(fit_file, kwargs, path, via):
    with pytest.raises(BadSettingError) as exc:
        if via == "api":
            ftmw.fit_peaks(fit_file, **kwargs)
        else:
            Pipeline.open(fit_file).fit_peaks(**kwargs)
    _assert_bad_setting(exc, path)


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_fit_show_unknown_ids_and_freqs_all_at_once(fit_file, via):
    # Mutation: only the first unknown id is reported / plain ValueError.
    def call(**kw):
        if via == "api":
            return ftmw.show_fit(fit_file, **kw)
        return Pipeline.open(fit_file).show_fit(**kw)

    with pytest.raises(BadSettingError) as exc:
        call(window_ids=[10**6, 10**6 + 1])
    _assert_bad_setting(exc, "window_ids")
    assert exc.value.value == [10**6, 10**6 + 1]

    with pytest.raises(BadSettingError) as exc:
        call(freqs=[1.0, 2.0])
    _assert_bad_setting(exc, "freqs")
    assert exc.value.value == [1.0, 2.0]


def test_fit_show_bad_apodization_names_the_cli_knob(fit_file):
    # Mutation: make_apodization raises an untyped ValueError.
    wid = int(load_fit_impl(fit_file)["fit"].window_fits[0].window_id)
    with pytest.raises(BadSettingError) as exc:
        ftmw.show_fit(fit_file, window_ids=[wid], apodize="notawindow")
    _assert_bad_setting(exc, "apodize")


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_review_create_anchor_inside_a_window(fit_file, via):
    # Mutation: plan_stage6_window raises a plain ValueError for an anchor inside a window.
    def call(anchor):
        if via == "api":
            return ftmw.review_create(fit_file, anchor)
        return Pipeline.open(fit_file).review_create(anchor)

    wf = load_fit_impl(fit_file)["fit"].window_fits[0]
    lo, hi = wf.window.freq_range
    with pytest.raises(BadSettingError) as exc:
        call(0.5 * (lo + hi))  # already inside an existing window
    _assert_bad_setting(exc, "anchor_mhz")
