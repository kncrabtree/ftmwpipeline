"""``window_model`` and ``spectrum_model`` (CONTRACT_STRATEGY §Window model,
§Spectrum model).

The window payload alone reproduces the fit's chi-squared on every window; the
display grid carries every active bin with the same model; refusals are typed;
the reads never write. Runs on the session-scoped reviewed 2638 build, read
only (the Stage 1-only build for the stage-not-run refusal).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline
from ftmwpipeline._internal import model_impl
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import SPECTRUM_MODEL_SCHEMA, WINDOW_MODEL_SCHEMA
from ftmwpipeline.file_manager import NotFoundError, StageDependencyError
from ftmwpipeline.fitting import model_eval

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


# ---- the whole Gaussian 2638 fit: spur windows, baseline windows ----------


@pytest.fixture(scope="module")
def gctx(stage5_gaussian_2638):
    return model_impl._load_model_context(
        str(stage5_gaussian_2638), "window_model", "active"
    )


def _has_baseline(wf) -> bool:
    return float((wf.quality_metrics or {}).get("baseline_applied", 0.0)) >= 0.5


def _active_payloads(ctx, components=False):
    return {
        int(wf.window_id): model_impl._window_payload(
            ctx, int(wf.window_id), components=components
        )
        for wf in ctx.fit.window_fits
    }


def test_chi2_reproduces_on_spur_baseline_and_plain_windows(gctx):
    payloads = _active_payloads(gctx, components=True)
    by_id = {int(w.window_id): w for w in gctx.fit.window_fits}
    spur = [w for w, p in payloads.items() if p["excluded"].any()]
    base = [w for w, wf in by_id.items() if _has_baseline(wf)]
    plain = [w for w, wf in by_id.items() if not _has_baseline(wf) and w not in spur]
    # The fixture must actually contain every kind the promise is about.
    assert spur and base and plain
    for wid in spur + base + plain[:10]:
        p = payloads[wid]
        assert _chi2(p) == pytest.approx(2.0 * by_id[wid].cost, rel=1e-9, abs=1e-9)
    # A spur window's chi-squared is only reproduced because the mask is applied.
    p = payloads[spur[0]]
    keep_all = np.sum(np.abs(p["data"] - p["model"]) ** 2 / (p["sigma"] ** 2 / 2))
    assert keep_all > _chi2(p)


def test_model_is_components_plus_fixed_plus_baseline(gctx):
    by_id = {int(w.window_id): w for w in gctx.fit.window_fits}
    for wid, p in _active_payloads(gctx, components=True).items():
        recon = sum(p["components"].values()) + p["fixed"]
        if _has_baseline(by_id[wid]):
            assert isinstance(p["baseline"], np.ndarray)
            recon = recon + p["baseline"]
        scale = max(1.0, float(np.max(np.abs(p["model"]), initial=0.0)))
        np.testing.assert_allclose(recon, p["model"], rtol=0, atol=1e-12 * scale)
        uids = [q.peak_uid for q in by_id[wid].fitted_peaks]
        assert sorted(p["components"]) == sorted(int(u) for u in uids)


def test_baseline_is_not_run_without_one(gctx, stage5):
    without = [w for w in gctx.fit.window_fits if not _has_baseline(w)]
    assert without
    for wf in without[:10]:
        p = model_impl._window_payload(gctx, int(wf.window_id), components=False)
        assert p["baseline"] is Absent.NOT_RUN
    # And the same on the small Lorentzian build, through the public API.
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    assert ftmw.window_model(stage5, wid)["baseline"] is Absent.NOT_RUN


def test_display_grid_on_a_baseline_window(stage5_gaussian_2638, gctx):
    path = str(stage5_gaussian_2638)
    wid = next(int(w.window_id) for w in gctx.fit.window_fits if _has_baseline(w))
    active = ftmw.window_model(path, wid, components=True)
    display = ftmw.window_model(path, wid, grid="display", components=True)
    assert display["sigma"] is Absent.UNDEFINED
    assert display["excluded"] is Absent.UNDEFINED
    idx = np.searchsorted(display["frequency_mhz"], active["frequency_mhz"])
    np.testing.assert_array_equal(
        display["frequency_mhz"][idx], active["frequency_mhz"]
    )
    for key in ("model", "fixed", "baseline"):
        np.testing.assert_array_equal(display[key][idx], active[key])


def test_spectrum_model_lines_once_and_baselines_inside_their_windows(gctx):
    fit = gctx.fit
    owners = model_eval.spectrum_line_owners(fit)
    by_id = {int(w.window_id): w for w in fit.window_fits}
    holders: dict = {}
    for w in fit.window_fits:
        for q in w.fitted_peaks:
            holders.setdefault(int(q.peak_uid), set()).add(int(w.window_id))
    # A line held by several windows (a thaw copy) has exactly one owner, a
    # window that holds it; every other line is drawn by its only window.
    assert set(owners) == {u for u, h in holders.items() if len(h) > 1}
    for uid, wid in owners.items():
        assert wid in holders[uid]

    freq = gctx.grid.freq_mhz
    kw = dict(acquisition_us=gctx.acquisition_us, sideband=gctx.sideband)
    full = model_eval.evaluate_spectrum_model(freq, fit, **kw)

    def _no_baseline(wf):
        c = copy.copy(wf)
        c.quality_metrics = {**(wf.quality_metrics or {}), "baseline_applied": 0.0}
        return c

    lines = model_eval.evaluate_spectrum_model(
        freq,
        SimpleNamespace(window_fits=[_no_baseline(w) for w in fit.window_fits]),
        **kw,
    )
    diff = full - lines

    ranges = {w: model_eval.window_fit_range_mhz(wf) for w, wf in by_id.items()}
    covered = np.zeros(freq.size, dtype=int)
    for lo, hi in ranges.values():
        covered += (freq >= lo) & (freq <= hi)
    # Baselines are zero everywhere no window reaches ...
    assert not diff[covered == 0].any()
    # ... and inside a window that no other window reaches they are the
    # window's own baseline.
    checked = 0
    for wid, wf in by_id.items():
        if not _has_baseline(wf):
            continue
        lo, hi = ranges[wid]
        sel = (freq >= lo) & (freq <= hi) & (covered == 1)
        if not sel.any():
            continue
        p = model_impl._window_payload(gctx, wid, components=False)
        in_range = (freq >= lo) & (freq <= hi)
        own = p["baseline"][sel[in_range]]
        np.testing.assert_allclose(
            diff[sel], own, rtol=0, atol=1e-12 * max(1.0, float(np.abs(own).max()))
        )
        checked += 1
    assert checked > 0


def test_spectrum_model_display_matches_active_at_coincident_bins(
    stage5_gaussian_2638,
):
    path = str(stage5_gaussian_2638)
    active = ftmw.spectrum_model(path)
    display = ftmw.spectrum_model(path, grid="display")
    idx = np.searchsorted(display["frequency_mhz"], active["frequency_mhz"])
    np.testing.assert_array_equal(
        display["frequency_mhz"][idx], active["frequency_mhz"]
    )
    np.testing.assert_array_equal(display["model"][idx], active["model"])
    np.testing.assert_array_equal(
        display["residual"], display["data"] - display["model"]
    )


# ---- the CLI: return codes, byte-identical reads, typed refusals ----------


def test_cli_reads_return_zero_and_leave_the_file_byte_identical(
    stage5, tmp_path, capsys
):
    before = _md5(stage5)
    wid = int(ftmw.load_fit(stage5).window_fits[0].window_id)
    for argv in (
        ["read", "window_model", stage5, str(wid), "--components"],
        ["read", "window_model", stage5, str(wid), "--grid", "display"],
        ["read", "spectrum_model", stage5],
        ["read", "spectrum_model", stage5, "--grid", "display"],
    ):
        rc = main(argv + ["--format", "json", "--output", str(tmp_path / "o")])
        assert rc == 0, capsys.readouterr().err
        capsys.readouterr()
    assert _md5(stage5) == before


def test_cli_unknown_window_is_a_typed_not_found(stage5, tmp_path, capsys):
    rc = main(
        ["read", "window_model", stage5, str(10**6), "--format", "json"]
        + ["--output", str(tmp_path / "o")]
    )
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    payload = json.loads(cap.err)
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"
    assert payload["kind"] == "window"


def test_cli_without_a_fit_names_the_fit_run_verb(
    baseline_2638_stage1, tmp_path, capsys
):
    path = str(baseline_2638_stage1)
    for argv in (
        ["read", "window_model", path, "1"],
        ["read", "spectrum_model", path],
    ):
        rc = main(argv + ["--format", "json", "--output", str(tmp_path / "o")])
        cap = capsys.readouterr()
        assert rc == 1 and cap.out == ""
        assert json.loads(cap.err)["command"] == "fit run"


def test_cli_bad_grid_is_a_usage_error(stage5, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["read", "spectrum_model", stage5, "--grid", "persisted"])
    assert exc.value.code == 2
    capsys.readouterr()
    with pytest.raises(ValueError):
        Pipeline.open(stage5).spectrum_model(grid="persisted")
    wid = _first(stage5)
    with pytest.raises(ValueError):
        Pipeline.open(stage5).window_model(wid, grid="persisted")


def _first(path) -> int:
    return int(ftmw.load_fit(path).window_fits[0].window_id)
