"""``window_model`` and ``spectrum_model`` (CONTRACT_STRATEGY §Window model,
§Spectrum model).

The window payload alone reproduces the fit's chi-squared on every window; the
display grid carries every active bin with the same model; refusals are typed;
the reads never write. Runs on the session-scoped reviewed 2638 build, read
only (the Stage 1-only build for the stage-not-run refusal).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline
from ftmwpipeline._internal import model_impl
from ftmwpipeline.contract import SPECTRUM_MODEL_SCHEMA, WINDOW_MODEL_SCHEMA
from ftmwpipeline.file_manager import NotFoundError, StageDependencyError

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def stage5(stage5_reviewed_2638):
    return str(stage5_reviewed_2638)


def _chi2(payload) -> float:
    keep = ~payload["excluded"]
    r = (payload["data"] - payload["model"])[keep]
    return float(np.sum(np.abs(r) ** 2 / (payload["sigma"][keep] ** 2 / 2)))


def test_payload_reproduces_the_fit_chi2_on_every_window(stage5):
    ctx = model_impl._load_model_context(stage5, "window_model", "active")
    assert ctx.fit.window_fits
    for wf in ctx.fit.window_fits:
        payload = model_impl._window_payload(ctx, int(wf.window_id), components=True)
        assert _chi2(payload) == pytest.approx(2.0 * wf.cost, rel=1e-10, abs=1e-9)
        recon = sum(payload["components"].values()) + payload["fixed"]
        if isinstance(payload["baseline"], np.ndarray):
            recon = recon + payload["baseline"]
        scale = max(1.0, float(np.max(np.abs(payload["model"]), initial=0.0)))
        np.testing.assert_allclose(recon, payload["model"], rtol=0, atol=1e-12 * scale)


def test_api_payload_equals_the_shared_path(stage5):
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    via_api = ftmw.window_model(stage5, wid)
    via_pipe = Pipeline.open(stage5).window_model(wid)
    ctx = model_impl._load_model_context(stage5, "window_model", "active")
    direct = model_impl._window_payload(ctx, wid, components=False)
    assert via_api["schema"] == WINDOW_MODEL_SCHEMA
    assert via_api["frame"] == "raw" and via_api["grid"] == "active"
    assert via_api["components"] is Absent.NOT_RUN
    for key in ("frequency_mhz", "data", "model", "fixed", "sigma", "excluded"):
        np.testing.assert_array_equal(via_api[key], direct[key])
        np.testing.assert_array_equal(via_pipe[key], direct[key])


def test_display_grid_overlays_the_active_bins(stage5):
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    active = ftmw.window_model(stage5, wid)
    display = ftmw.window_model(stage5, wid, grid="display")
    assert display["sigma"] is Absent.UNDEFINED
    assert display["excluded"] is Absent.UNDEFINED
    idx = np.searchsorted(display["frequency_mhz"], active["frequency_mhz"])
    np.testing.assert_array_equal(
        display["frequency_mhz"][idx], active["frequency_mhz"]
    )
    np.testing.assert_array_equal(display["model"][idx], active["model"])
    scale = float(np.max(np.abs(active["data"])))
    np.testing.assert_allclose(
        display["data"][idx], active["data"], rtol=0, atol=1e-12 * scale
    )


def test_spectrum_model_covers_the_whole_active_grid(stage5):
    sm = ftmw.spectrum_model(stage5)
    assert sm["schema"] == SPECTRUM_MODEL_SCHEMA and sm["frame"] == "raw"
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    wm = ftmw.window_model(stage5, wid)
    assert np.all(np.diff(sm["frequency_mhz"]) > 0)
    assert np.isin(wm["frequency_mhz"], sm["frequency_mhz"]).all()
    np.testing.assert_array_equal(sm["residual"], sm["data"] - sm["model"])
    display = ftmw.spectrum_model(stage5, grid="display")
    assert display["frequency_mhz"].size == 2 * sm["frequency_mhz"].size - 1


def test_reads_never_write(stage5):
    before = _md5(stage5)
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    ftmw.window_model(stage5, wid, components=True)
    ftmw.window_model(stage5, wid, grid="display")
    ftmw.spectrum_model(stage5)
    assert _md5(stage5) == before


def test_unknown_window_is_not_found(stage5):
    with pytest.raises(NotFoundError) as exc:
        ftmw.window_model(stage5, 10**6)
    assert exc.value.to_dict()["code"] == "not_found"


def test_without_a_fit_is_stage_not_run(baseline_2638_stage1):
    for call in (
        lambda p: ftmw.window_model(p, 1),
        lambda p: ftmw.spectrum_model(p),
    ):
        with pytest.raises(StageDependencyError) as exc:
            call(str(baseline_2638_stage1))
        assert exc.value.to_dict()["command"] == "fit run"


def test_unknown_grid_is_refused(stage5):
    with pytest.raises(ValueError):
        ftmw.spectrum_model(stage5, grid="persisted")
