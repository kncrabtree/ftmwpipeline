"""
Stage 5 integration tests on real experiment 2638 data.

* cross-interface consistency: CLI == Pipeline == functional API
* serialization round-trip + hand-edit on the real fit
* invalidation on Stage 4 re-assignment and on a Stage 1 canonical-settings
  change

The full 328-window 2638 fit is heavy; marked slow + integration. The
finer-grained Stage 5 prototype-fixtures + scenario tests live alongside the
algorithm modules (``tests/unit/fitting/``).
"""

import json
import shutil
import subprocess

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.data_structures import SpectrumFit

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _assert_fits_equivalent(a: SpectrumFit, b: SpectrumFit) -> None:
    """Two SpectrumFit aggregates carry the same scientific content.

    The active-FT and the solver path are deterministic from the persisted
    inputs (FID + canonical Stage 1 settings + plan), so two independent
    Stage 5 runs against the same .ftmw file must produce the same windows,
    the same per-window peak count, and the same per-peak parameters up to
    a tight tolerance.
    """
    assert a.n_windows == b.n_windows
    assert a.n_fitted_peaks == b.n_fitted_peaks
    assert a.final_plan_revision == b.final_plan_revision

    by_a = {w.window_id: w for w in a.window_fits}
    by_b = {w.window_id: w for w in b.window_fits}
    assert set(by_a) == set(by_b)
    for wid in by_a:
        wa = by_a[wid]
        wb = by_b[wid]
        assert len(wa.fitted_peaks) == len(wb.fitted_peaks), (
            f"window {wid} peak count differs: {len(wa.fitted_peaks)} vs "
            f"{len(wb.fitted_peaks)}"
        )
        # Shared tau and its disambiguating "fitted" flag must match exactly:
        # the flag is set by result_conversion from the inner WindowFitResult,
        # so all three interfaces (CLI / Pipeline / api) -- thin wrappers over
        # the same impl -- have to land on the same value per window.
        ta = wa.shared_parameters.get("tau_us", {})
        tb = wb.shared_parameters.get("tau_us", {})
        assert ta.get("fitted") == tb.get("fitted"), (
            f"window {wid} tau_us.fitted differs: "
            f"{ta.get('fitted')!r} vs {tb.get('fitted')!r}"
        )
        wa_peaks = sorted(wa.fitted_peaks, key=lambda p: p.frequency_mhz)
        wb_peaks = sorted(wb.fitted_peaks, key=lambda p: p.frequency_mhz)
        for pa, pb in zip(wa_peaks, wb_peaks):
            assert pa.frequency_mhz == pytest.approx(pb.frequency_mhz, abs=1e-6)
            assert pa.amplitude == pytest.approx(pb.amplitude, rel=1e-6, abs=1e-9)
            assert pa.window_id == pb.window_id
            assert pa.peak_id == pb.peak_id


def test_cross_interface_consistency(baseline_2638_stage4, temp_ftmw_dir):
    """CLI == Pipeline == functional API for fit_peaks."""
    pfile = temp_ftmw_dir / "p.ftmw"
    ffile = temp_ftmw_dir / "f.ftmw"
    cfile = temp_ftmw_dir / "c.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage4, fp)

    fit_pipe = Pipeline(pfile).fit_peaks()
    fit_func = ftmw.fit_peaks(ffile)
    res = subprocess.run(
        ["ftmwpipeline", "fit-peaks", str(cfile)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    fit_cli = ftmw.load_fit(cfile)

    _assert_fits_equivalent(fit_pipe, fit_func)
    _assert_fits_equivalent(fit_pipe, fit_cli)


def test_serialization_round_trip_and_hand_edit(baseline_2638_stage4, temp_ftmw_dir):
    """Save -> load returns an equivalent fit; an in-place peak edit survives."""
    fp = temp_ftmw_dir / "ser.ftmw"
    shutil.copy(baseline_2638_stage4, fp)

    fit = ftmw.fit_peaks(fp)
    reloaded = ftmw.load_fit(fp)
    _assert_fits_equivalent(fit, reloaded)

    # Hand-edit the first window's first fitted peak's frequency; reload sees it.
    first_window = reloaded.window_fits[0]
    if not first_window.fitted_peaks:
        pytest.skip("first window has no fitted peaks; cannot exercise hand-edit")
    wid = first_window.window_id
    with h5py.File(fp, "a") as h5f:
        arr = h5f[f"stage5_fitting/windows/window_{wid:04d}/peaks/frequency_mhz"]
        original = float(arr[0])
        arr[0] = original + 0.001  # nudge +1 kHz
    again = ftmw.load_fit(fp)
    edited_peak = again.window_fit(wid).fitted_peaks[0]
    assert edited_peak.frequency_mhz == pytest.approx(original + 0.001)


def test_reassign_windows_invalidates_stage5(baseline_2638_stage4, temp_ftmw_dir):
    """Re-running Stage 4 drops the stale Stage 5 fit."""
    fp = temp_ftmw_dir / "inv.ftmw"
    shutil.copy(baseline_2638_stage4, fp)

    ftmw.fit_peaks(fp)
    with h5py.File(fp, "r") as h5f:
        assert "stage5_fitting" in h5f
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage5_fitting" in completed

    # Re-run Stage 4 (it invalidates downstream Stage 5 by registered dep).
    ftmw.assign_windows(fp)
    with h5py.File(fp, "r") as h5f:
        assert "stage5_fitting" not in h5f
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage5_fitting" not in completed

    with pytest.raises(Exception):
        ftmw.load_fit(fp)


def test_stage1_change_invalidates_stage5(baseline_2638_stage4, temp_ftmw_dir):
    """Changing canonical Stage 1 settings invalidates Stage 5 transitively."""
    fp = temp_ftmw_dir / "s1.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    ftmw.fit_peaks(fp)

    # Different Stage 1 settings cascade through Stage 2/3/4 invalidation.
    ftmw.compute_ft(fp, zpf=1, expf_us=5.0, trim=(26500, 40000))
    with h5py.File(fp, "r") as h5f:
        assert "stage5_fitting" not in h5f
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage5_fitting" not in completed
        assert "stage4_windows" not in completed
        assert "stage3_peaks" not in completed


def test_fit_peaks_gaussian_cross_interface(
    baseline_2638_stage4, temp_ftmw_dir,
):
    """fit_peaks(shape='gaussian') is identical across CLI / Pipeline / api.

    Drives all three interfaces on a baseline that has the Gaussian τ_G
    calibration pre-staged, then asserts the persisted fits agree. Also
    confirms the persisted ``/stage5_fitting`` root group carries the new
    ``shape='gaussian'`` attribute and per-window subgroups inherit it.
    """
    pfile = temp_ftmw_dir / "gp.ftmw"
    ffile = temp_ftmw_dir / "gf.ftmw"
    cfile = temp_ftmw_dir / "gc.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage4, fp)
        ftmw.calibrate_tau_G(fp)

    fit_pipe = Pipeline(pfile).fit_peaks(shape="gaussian")
    fit_func = ftmw.fit_peaks(ffile, shape="gaussian")
    res = subprocess.run(
        ["ftmwpipeline", "fit-peaks", str(cfile), "--shape", "gaussian"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    fit_cli = ftmw.load_fit(cfile)

    _assert_fits_equivalent(fit_pipe, fit_func)
    _assert_fits_equivalent(fit_pipe, fit_cli)

    # The persisted root and per-window shape attributes are 'gaussian'.
    for fp in (pfile, ffile, cfile):
        with h5py.File(fp, "r") as h5f:
            root_shape = h5f["stage5_fitting"].attrs.get("shape")
            root_shape_str = (
                root_shape.decode("utf-8")
                if isinstance(root_shape, bytes)
                else str(root_shape)
            )
            assert root_shape_str == "gaussian", (
                f"{fp.name}: /stage5_fitting shape attr is {root_shape_str!r}"
            )
            windows_group = h5f["stage5_fitting/windows"]
            for wname in windows_group:
                w_shape = windows_group[wname].attrs.get("shape", "lorentzian")
                w_shape_str = (
                    w_shape.decode("utf-8")
                    if isinstance(w_shape, bytes)
                    else str(w_shape)
                )
                assert w_shape_str == "gaussian", (
                    f"{fp.name}/{wname} shape attr is {w_shape_str!r}"
                )


def test_fit_peaks_gaussian_persists_and_loads_shape(
    baseline_2638_stage4, temp_ftmw_dir,
):
    """A Gaussian fit round-trips: every loaded FittingResult carries shape='gaussian'."""
    fp = temp_ftmw_dir / "rg.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    ftmw.calibrate_tau_G(fp)
    ftmw.fit_peaks(fp, shape="gaussian")
    fit = ftmw.load_fit(fp)
    # Every window's FittingResult must carry the persisted shape; older
    # files would default to 'lorentzian' on the missing attribute.
    assert fit.window_fits, "no windows in the loaded fit"
    for wf in fit.window_fits:
        assert wf.shape == "gaussian", (
            f"window {wf.window_id} shape={wf.shape!r} (expected 'gaussian')"
        )
    # The fit's parameters dict echoes the same shape the driver used.
    assert fit.parameters.get("shape") == "gaussian"


def test_calibrate_tau_G_cross_interface(
    baseline_2638_stage4, temp_ftmw_dir,
):
    """calibrate_tau_G is identical across CLI / Pipeline / api on 2638."""
    pfile = temp_ftmw_dir / "tgp.ftmw"
    ffile = temp_ftmw_dir / "tgf.ftmw"
    cfile = temp_ftmw_dir / "tgc.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage4, fp)

    tc_pipe = Pipeline(pfile).calibrate_tau_G()
    tc_func = ftmw.calibrate_tau_G(ffile)
    res = subprocess.run(
        ["ftmwpipeline", "calibrate-tau-G", str(cfile)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    tc_cli = ftmw.load_tau_G_calibration(cfile)

    # The STFT + per-bin Voigt path is deterministic from the persisted
    # FID and Stage 1 settings; all three interfaces must land on
    # numerically identical aggregates.
    for other in (tc_func, tc_cli):
        assert other.tau_maj_us == pytest.approx(tc_pipe.tau_maj_us, rel=1e-12)
        assert other.sigma_tau_us == pytest.approx(tc_pipe.sigma_tau_us, rel=1e-12)
        assert other.n_contributors == tc_pipe.n_contributors
        assert len(other.band_majorities) == len(tc_pipe.band_majorities)


def test_cli_visualize_fit_writes_output(baseline_2638_stage4, temp_ftmw_dir):
    """visualize-fit --no-interactive --output writes the image and exits 0."""
    fp = temp_ftmw_dir / "viz.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    ftmw.fit_peaks(fp)

    out = temp_ftmw_dir / "fit_overview.png"
    res = subprocess.run(
        [
            "ftmwpipeline",
            "visualize-fit",
            str(fp),
            "--no-interactive",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG
