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

import hashlib
import json
import math
import shutil
import subprocess

import h5py
import numpy as np
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


def test_cross_interface_consistency(baseline_2638_stage4_small, temp_ftmw_dir):
    """CLI == Pipeline == functional API for fit_peaks.

    Runs on the small (3-window) Stage-4 baseline -- cross-interface
    bit-identity is a property of the dispatch shape, not the window count,
    so a small plan exercises it just as well as the full 382-window one.
    """
    pfile = temp_ftmw_dir / "p.ftmw"
    ffile = temp_ftmw_dir / "f.ftmw"
    cfile = temp_ftmw_dir / "c.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage4_small, fp)

    fit_pipe = Pipeline(pfile).fit_peaks()
    fit_func = ftmw.fit_peaks(ffile)
    res = subprocess.run(
        ["ftmwpipeline", "fit", "run", str(cfile)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    fit_cli = ftmw.load_fit(cfile)

    _assert_fits_equivalent(fit_pipe, fit_func)
    _assert_fits_equivalent(fit_pipe, fit_cli)


def test_validate_stage5_shape_error_cross_interface(
    baseline_2638_stage4_small, temp_ftmw_dir, tmp_path
):
    """validate-stage5-shape-error: CLI == Pipeline == api, and read-only.

    The command is a deterministic read over the persisted fit, so the
    Pipeline and functional-API reports must be dict-identical; the CLI
    (print-only) must run cleanly and report the same Tier-1 headline. The
    validation must not mutate the file.
    """
    fitted = temp_ftmw_dir / "fitted.ftmw"
    shutil.copy(baseline_2638_stage4_small, fitted)
    fit = ftmw.fit_peaks(fitted)

    # A tiny ground-truth catalog drawn from the fit's own frequencies so Tier 3
    # has guaranteed matches and the catalog code path is exercised too.
    freqs = sorted(p.frequency_mhz for p in fit.fitted_peaks)[:5]
    assert freqs, "need at least one fitted peak to build a ground-truth catalog"
    gt = tmp_path / "ground_truth.csv"
    gt.write_text("freq_mhz,unc_mhz\n" + "".join(f"{f:.6f},0.001\n" for f in freqs))

    pfile = temp_ftmw_dir / "vp.ftmw"
    ffile = temp_ftmw_dir / "vf.ftmw"
    cfile = temp_ftmw_dir / "vc.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(fitted, fp)

    before = hashlib.sha256(pfile.read_bytes()).hexdigest()
    rep_pipe = Pipeline(pfile).validate_stage5_shape_error(ground_truth=gt)
    rep_func = ftmw.validate_stage5_shape_error(ffile, ground_truth=gt)
    after = hashlib.sha256(pfile.read_bytes()).hexdigest()

    assert rep_pipe == rep_func
    assert before == after, "validation must not mutate the .ftmw file"
    assert rep_pipe["tier3"] is not None
    assert rep_pipe["tier3"]["n_matched"] >= 1
    assert rep_pipe["parameters"]["kappa"] == pytest.approx(0.05)

    res = subprocess.run(
        [
            "ftmwpipeline",
            "fit",
            "check",
            str(cfile),
            "--ground-truth",
            str(gt),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    t1 = rep_pipe["tier1"]
    assert f"{t1['n_pass']}/{t1['n_windows']} windows pass" in res.stdout


def test_serialization_round_trip_and_hand_edit(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """Save -> load returns an equivalent fit; an in-place peak edit survives.

    Round-trip + hand-edit semantics are window-count-independent, so the
    small (3-window) baseline is sufficient.
    """
    fp = temp_ftmw_dir / "ser.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)

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


def test_baseline_audit_persists(baseline_2638_stage4_small, temp_ftmw_dir):
    """A default fit records the leakage-wing baseline settings + per-window
    audit trail on the persisted fit (window-count-independent contract)."""
    fp = temp_ftmw_dir / "baseline_audit.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)

    fit = ftmw.fit_peaks(fp)
    # Settings audit on the plan-level parameters.
    assert fit.parameters.get("baseline_enabled") is True
    assert fit.parameters.get("baseline_order") == 4
    assert fit.parameters.get("baseline_edge_threshold") == pytest.approx(3.5)
    assert fit.parameters.get("baseline_smooth_threshold") == pytest.approx(50.0)
    n_fired = fit.parameters.get("n_baseline_windows")
    assert isinstance(n_fired, int) and n_fired >= 0

    # Every window records whether the baseline fired; fired windows carry the
    # fitted complex coefficients + the triggering S_coh.
    reloaded = ftmw.load_fit(fp)
    for wf in reloaded.window_fits:
        qa = wf.quality_metrics or {}
        assert "baseline_applied" in qa
        if qa.get("baseline_applied", 0.0) > 0.5:
            order = int(qa["baseline_order"])
            for k in range(order + 1):
                assert f"baseline_coeff{k}_re" in qa
                assert f"baseline_coeff{k}_im" in qa
            assert qa["baseline_edge_coherence"] > 3.5


def test_per_window_covariance_persisted(baseline_2638_stage4_small, temp_ftmw_dir):
    """After a Stage 5 fit, at least one window has a persisted covariance
    matrix, and sqrt(diag) of the amplitude entries matches the stored
    amplitude_error for that window (ordering sanity check).

    Uses the small (3-window) baseline so the fit is fast.
    """
    fp = temp_ftmw_dir / "cov_check.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)

    fit = ftmw.fit_peaks(fp)
    reloaded = ftmw.load_fit(fp)

    # Every window's peaks are ordered by ascending frequency (so the line list,
    # the report / fit-show tables, and the covariance / correlation heatmap all
    # read in one order). The amplitude/offset/phase diagonal checks below then
    # confirm each peak kept its own covariance block through that reorder.
    for wf in reloaded.window_fits:
        freqs = [p.frequency_mhz for p in wf.fitted_peaks]
        assert freqs == sorted(
            freqs
        ), f"window {wf.window_id} peaks are not frequency-ordered: {freqs}"

    # At least one window should have a non-None covariance (the 3-window
    # small plan has real lines with finite JᵀJ).
    windows_with_cov = [wf for wf in reloaded.window_fits if wf.covariance is not None]
    assert (
        windows_with_cov
    ), "no windows with persisted covariance in the 3-window small fit"

    # For each window with a covariance, verify the amplitude / offset / phase
    # diagonal entries match the stored per-peak errors (sqrt round-trip). The
    # diagonal entry for peak ``i`` must equal *that* peak's error -- the check
    # that the frequency sort permuted the covariance in lockstep with the peaks.
    for wf in windows_with_cov:
        cov = wf.covariance
        labels = wf.covariance_param_labels
        assert cov is not None
        assert labels is not None
        assert cov.shape[0] == cov.shape[1] == len(labels)

        for kind, attr in (
            ("amplitude", "amplitude_error"),
            ("offset", "frequency_error"),
            ("phase", "phase_error"),
        ):
            indices = [i for i, lbl in enumerate(labels) if lbl.startswith(f"{kind}_")]
            assert len(indices) == len(wf.fitted_peaks)
            for peak_idx, col_idx in enumerate(indices):
                cov_err = math.sqrt(float(cov[col_idx, col_idx]))
                stored = getattr(wf.fitted_peaks[peak_idx], attr)
                if stored is not None:
                    assert cov_err == pytest.approx(stored, rel=1e-6), (
                        f"window {wf.window_id} peak {peak_idx}: "
                        f"sqrt(cov[{kind},{kind}])={cov_err} != {attr}={stored}"
                    )


def _inject_stage5_marker(fp) -> None:
    """Write a minimal ``stage5_fitting`` group + mark it complete.

    Stage 5 invalidation keys off the group's existence and the
    ``completed_stages`` JSON list, not the fit's contents — so a placeholder
    is enough to exercise the invalidation cascade without paying for an
    ~2-minute real fit. This is a deliberate shortcut for the invalidation
    tests; functional tests must run a real fit.
    """
    with h5py.File(fp, "a") as h5f:
        if "stage5_fitting" not in h5f:
            g = h5f.create_group("stage5_fitting")
            g.attrs["stage_name"] = "stage5_fitting"
            g.attrs["n_windows"] = 0
            g.attrs["n_fitted_peaks"] = 0
        completed = json.loads(
            h5f["pipeline_stages"].attrs.get("completed_stages", "[]")
        )
        if "stage5_fitting" not in completed:
            completed.append("stage5_fitting")
        h5f["pipeline_stages"].attrs["completed_stages"] = json.dumps(completed)


def test_reassign_windows_invalidates_stage5(baseline_2638_stage4, temp_ftmw_dir):
    """Re-running Stage 4 drops the stale Stage 5 marker."""
    fp = temp_ftmw_dir / "inv.ftmw"
    shutil.copy(baseline_2638_stage4, fp)

    _inject_stage5_marker(fp)
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


def test_stage1_change_invalidates_stage5(baseline_2638_stage4, temp_ftmw_dir):
    """Changing canonical Stage 1 settings invalidates Stage 5 transitively."""
    fp = temp_ftmw_dir / "s1.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    _inject_stage5_marker(fp)

    # Different Stage 1 settings cascade through Stage 2/3/4 invalidation.
    ftmw.compute_ft(fp, start_us=2.0, trim=(26500, 40000))
    with h5py.File(fp, "r") as h5f:
        assert "stage5_fitting" not in h5f
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage5_fitting" not in completed
        assert "stage4_windows" not in completed
        assert "stage3_peaks" not in completed


def test_fit_peaks_gaussian_cross_interface(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """fit_peaks(shape='gaussian') is identical across CLI / Pipeline / api.

    Drives all three interfaces on a small baseline that has the Gaussian
    τ_G calibration pre-staged, then asserts the persisted fits agree. Also
    confirms the persisted ``/stage5_fitting`` root group carries the new
    ``shape='gaussian'`` attribute and per-window subgroups inherit it.

    Stage 2b cross-interface identity is covered separately by
    ``test_calibrate_tau_G_cross_interface``; here we calibrate once on
    a shared file and copy, so the three test files differ only at Stage 5.
    """
    from tests.integration._stage2b_helpers import skip_auto_recommend_settings

    pfile = temp_ftmw_dir / "gp.ftmw"
    ffile = temp_ftmw_dir / "gf.ftmw"
    cfile = temp_ftmw_dir / "gc.ftmw"
    # Build Stage 2b once on the small baseline, then copy to all three.
    # ``shape='gaussian'`` is pinned explicitly on the fit_peaks calls below,
    # so the Stage 2b auto-recommend verdict is not needed -- skip the ~50s
    # 3-way classifier pass on this fixture.
    staged = temp_ftmw_dir / "gaussian_staged.ftmw"
    shutil.copy(baseline_2638_stage4_small, staged)
    ftmw.calibrate_tau(
        staged, shape="gaussian", settings=skip_auto_recommend_settings()
    )
    for fp in (pfile, ffile, cfile):
        shutil.copy(staged, fp)

    fit_pipe = Pipeline(pfile).fit_peaks(shape="gaussian")
    fit_func = ftmw.fit_peaks(ffile, shape="gaussian")
    res = subprocess.run(
        ["ftmwpipeline", "fit", "run", str(cfile), "--shape", "gaussian"],
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
            assert (
                root_shape_str == "gaussian"
            ), f"{fp.name}: /stage5_fitting shape attr is {root_shape_str!r}"
            windows_group = h5f["stage5_fitting/windows"]
            for wname in windows_group:
                w_shape = windows_group[wname].attrs.get("shape", "lorentzian")
                w_shape_str = (
                    w_shape.decode("utf-8")
                    if isinstance(w_shape, bytes)
                    else str(w_shape)
                )
                assert (
                    w_shape_str == "gaussian"
                ), f"{fp.name}/{wname} shape attr is {w_shape_str!r}"


def test_fit_peaks_gaussian_persists_and_loads_shape(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """A Gaussian fit round-trips: every loaded FittingResult carries shape='gaussian'.

    Shape-persistence is a per-window attribute round-trip; the small (3-window)
    baseline exercises every code path of the full fixture.
    """
    from tests.integration._stage2b_helpers import skip_auto_recommend_settings

    fp = temp_ftmw_dir / "rg.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)
    # ``shape='gaussian'`` is pinned explicitly below, so the Stage 2b
    # auto-recommend verdict is not needed -- skip the ~50s classifier pass.
    ftmw.calibrate_tau(fp, shape="gaussian", settings=skip_auto_recommend_settings())
    ftmw.fit_peaks(fp, shape="gaussian")
    fit = ftmw.load_fit(fp)
    # Every window's FittingResult must carry the persisted shape; older
    # files would default to 'lorentzian' on the missing attribute.
    assert fit.window_fits, "no windows in the loaded fit"
    for wf in fit.window_fits:
        assert (
            wf.shape == "gaussian"
        ), f"window {wf.window_id} shape={wf.shape!r} (expected 'gaussian')"
    # The fit's parameters dict echoes the same shape the driver used.
    assert fit.parameters.get("shape") == "gaussian"


def test_calibrate_tau_G_cross_interface(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """calibrate_tau_G is identical across CLI / Pipeline / api on 2638.

    calibrate_tau_G operates on the FT (same in small + full baselines) and
    the FID; window count is irrelevant to this proof. The Stage 2b
    auto-recommend pass (~50s per interface on 2638) is unrelated to the
    τ_G identity assertion; opt out via the helper.
    """
    from tests.integration._stage2b_helpers import (
        skip_auto_recommend_preset_yaml,
        skip_auto_recommend_settings,
    )

    pfile = temp_ftmw_dir / "tgp.ftmw"
    ffile = temp_ftmw_dir / "tgf.ftmw"
    cfile = temp_ftmw_dir / "tgc.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage4_small, fp)

    skip = skip_auto_recommend_settings()
    skip_yaml = skip_auto_recommend_preset_yaml(temp_ftmw_dir)
    tc_pipe = Pipeline(pfile).calibrate_tau(shape="gaussian", settings=skip)
    tc_func = ftmw.calibrate_tau(ffile, shape="gaussian", settings=skip)
    res = subprocess.run(
        [
            "ftmwpipeline",
            "tau",
            "run",
            "--gaussian",
            str(cfile),
            "--preset",
            str(skip_yaml),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    tc_cli = ftmw.load_tau_calibration(cfile, shape="gaussian")

    # The STFT + per-bin Voigt path is deterministic from the persisted
    # FID and Stage 1 settings; all three interfaces must land on
    # numerically identical aggregates.
    for other in (tc_func, tc_cli):
        assert other.tau_maj_us == pytest.approx(tc_pipe.tau_maj_us, rel=1e-12)
        assert other.sigma_tau_us == pytest.approx(tc_pipe.sigma_tau_us, rel=1e-12)
        assert other.n_contributors == tc_pipe.n_contributors
        assert len(other.band_majorities) == len(tc_pipe.band_majorities)


def test_cli_visualize_fit_writes_output(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """visualize-fit --no-interactive --output writes the image and exits 0.

    Renders work the same on a small fit; the test gate is exit code +
    non-zero PNG output, not visual fidelity.
    """
    fp = temp_ftmw_dir / "viz.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)
    ftmw.fit_peaks(fp)

    out = temp_ftmw_dir / "fit_overview.png"
    res = subprocess.run(
        [
            "ftmwpipeline",
            "fit",
            "show",
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


def test_recommend_shape_persists_and_feeds_resolver(
    baseline_2638_stage4_small,
    temp_ftmw_dir,
):
    """The 3-way recommendation lands on the Stage 2b attr and Stage 5 picks it up.

    Runs ``calibrate_tau_G`` then ``recommend_shape`` on the small 2638
    baseline; verifies (a) the verdict is well-formed, (b) the
    ``recommended_shape`` attr lands on every Stage 2b group present,
    and (c) the Stage 5 resolver's persisted ``stage5_fit`` carries
    that shape after a no-arg ``fit_peaks`` call. The 2638 fixture is
    Gaussian-dominant on the strong contributor bins, so the verdict
    is expected to be ``"gaussian"``. ``calibrate_tau_G`` already
    auto-runs the recommendation by default; the explicit
    ``recommend_shape`` call here exercises the standalone surface
    (back-compat) and re-stamps the same verdict.
    """
    from ftmwpipeline.io.stage_fit_settings_serialization import (
        read_stage2b_recommended_shape,
    )
    from tests.integration._stage2b_helpers import skip_auto_recommend_settings

    fp = temp_ftmw_dir / "recommend.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)

    # Disable the calibrate_tau_G auto-recommend pass so the explicit
    # recommend_shape call below is the one whose stamp the test verifies
    # (otherwise the auto pass would have already stamped the same verdict).
    ftmw.calibrate_tau(fp, shape="gaussian", settings=skip_auto_recommend_settings())
    rec = ftmw.recommend_shape(fp)
    assert rec.n_contributors > 0
    assert sum(rec.vote_rates.values()) == pytest.approx(1.0, abs=1e-6)
    assert rec.recommended_shape == "gaussian"

    # Attr lands on the Gauss twin's group (the only Stage 2b group present
    # on this fixture path -- the Lorentzian twin is not run here).
    with h5py.File(fp, "r") as h5:
        assert "stage2b_tau_G_calibration" in h5
        attr = h5["stage2b_tau_G_calibration"].attrs.get("recommended_shape")
        assert attr is not None
        decoded = attr.decode("utf-8") if isinstance(attr, bytes) else str(attr)
        assert decoded == "gaussian"

    # The shared reader picks up the Gauss-twin attr when the Lorentzian
    # group is absent.
    assert read_stage2b_recommended_shape(str(fp)) == "gaussian"

    # Stage 5 resolver respects the recommendation: a no-arg fit_peaks
    # on a file with no explicit shape and no persisted shape inherits
    # the recommended one.
    ftmw.fit_peaks(fp)
    with h5py.File(fp, "r") as h5:
        shape_attr = h5["processing_parameters/stage5_fit"].attrs.get("shape")
        if shape_attr is None:
            # Sub-group form
            shape_attr = h5["processing_parameters/stage5_fit/shape"].attrs["kind"]
        decoded_shape = (
            shape_attr.decode("utf-8")
            if isinstance(shape_attr, bytes)
            else str(shape_attr)
        )
        assert decoded_shape == "gaussian"


# ---------------------------------------------------------------------------
# Doublet-alternative observation-only invariant + guaranteed-trigger fixture
# ---------------------------------------------------------------------------


def _build_synthetic_doublet_stage4(tmp_dir):
    """Build a Stage-4 .ftmw file whose single window contains two fitted peaks
    guaranteed to trigger the doublet-alternative adjudication pass.

    Acquisition: 10 µs at 100 MSa/s (1000 pts), upper sideband, probe 10000 MHz.
    Resolution element = 1/T_active = 0.1 MHz.

    Two molecular tones at 10001.00 and 10001.10 MHz (separation = 1.0 resolution
    elements, below the default k_res = 1.5 trigger threshold) with amplitude
    ratio 0.5 at SNR ≈ 50 (strong) / 25 (weak). Gaussian noise is added so the
    noise estimator returns a finite, realistic σ. Both tones are well above the
    Stage 3 promotion threshold and land in the same Stage 4 window, so the
    adjudication pass receives a window with exactly the qualifying pair.
    """
    from ftmwpipeline.core.data_structures import FID, Sideband
    from ftmwpipeline.file_manager import SourceMetadata, create_pipeline_file

    rng = np.random.default_rng(20260612)

    # --- FID parameters ---
    dt_us = 0.01  # µs per sample → 100 MSa/s
    dt_s = dt_us * 1e-6
    n_pts = 1000  # 10 µs total
    probe_mhz = 10000.0
    tau_us = 5.0  # decay constant; longer tau → narrower lines → both peaks keep
    # separate identity above the min-pair-separation floor (max(0.5*fwhm, 1/T))

    # Two molecular frequencies (upper sideband: IF = f_mol - probe)
    f_strong_mhz = 10001.00
    f_weak_mhz = 10001.10
    # Amplitude chosen so SNR ≈ 50 (strong) / 25 (weak) after noise addition.
    amp_strong = 1.0
    amp_weak = 0.5

    t_us = np.arange(n_pts) * dt_us
    fid_signal = amp_strong * np.exp(-t_us / tau_us) * np.cos(
        2.0 * np.pi * (f_strong_mhz - probe_mhz) * t_us
    ) + amp_weak * np.exp(-t_us / tau_us) * np.cos(
        2.0 * np.pi * (f_weak_mhz - probe_mhz) * t_us
    )
    # Add white noise small enough for SNR >> 3 on both lines.
    noise_sigma = 2e-3
    fid_data = fid_signal + rng.normal(0.0, noise_sigma, n_pts)

    fid = FID(
        data=fid_data,
        spacing=dt_s,
        probe_freq_mhz=probe_mhz,
        sideband=Sideband.UPPER,
        shots=1,
    )
    source_meta = SourceMetadata(
        source_path=tmp_dir / "synthetic_doublet_source",
        format_name="synthetic",
    )

    fp = tmp_dir / "synth_doublet.ftmw"
    create_pipeline_file(fp, fid, source_meta, force=True)

    # Stage 1: unapodized FT trimmed to the active band.
    ftmw.compute_ft(fp, start_us=0.0, end_us=10.0, trim=(9998.0, 10003.0))
    # Stage 2: noise estimation.
    ftmw.estimate_noise(fp)
    # Stage 3: peak detection — the two close tones are above SNR 3 and detected.
    ftmw.detect_peaks(fp)
    # Stage 4: window assignment — both peaks land in the same window.
    ftmw.assign_windows(fp)

    return fp


def test_doublet_alternative_observation_only(
    tmp_path,
):
    """The doublet-alternative pass is observation-only: fitted peaks are identical
    whether the pass is enabled or disabled; records are absent when disabled; and
    when enabled on a fixture whose pair separation is within the trigger band,
    at least one record fires with finite adjudication statistics.

    Uses a synthetic FID with two tones separated by exactly 1.0 resolution
    element and amplitude ratio 0.5. This guarantees the adjudication trigger
    fires on every run, so the finite-statistics assertions below are not
    conditional on the fixture's peak content.
    """
    from ftmwpipeline.core.stage_fit_settings import (
        DoubletAlternativeSubSettings,
        StageFitSettings,
    )

    stage4_file = _build_synthetic_doublet_stage4(tmp_path)

    def _run(tag: str, enabled: bool) -> SpectrumFit:
        fp = tmp_path / f"doublet_{tag}.ftmw"
        shutil.copy(stage4_file, fp)
        s = StageFitSettings()
        s.doublet_alternative = DoubletAlternativeSubSettings(enabled=enabled)
        return ftmw.fit_peaks(fp, settings=s)

    fit_on = _run("on", enabled=True)
    fit_off = _run("off", enabled=False)

    # Fitted peaks must be identical: the pass is observation-only and must
    # not change the production fit in any window.
    assert fit_on.n_windows == fit_off.n_windows
    assert fit_on.n_fitted_peaks == fit_off.n_fitted_peaks

    by_on = {w.window_id: w for w in fit_on.window_fits}
    by_off = {w.window_id: w for w in fit_off.window_fits}
    assert set(by_on) == set(by_off)
    for wid in by_on:
        wa = by_on[wid]
        wb = by_off[wid]
        assert len(wa.fitted_peaks) == len(wb.fitted_peaks), (
            f"window {wid}: enabled={True} has {len(wa.fitted_peaks)} peaks, "
            f"enabled={False} has {len(wb.fitted_peaks)}"
        )
        for pa, pb in zip(
            sorted(wa.fitted_peaks, key=lambda p: p.frequency_mhz),
            sorted(wb.fitted_peaks, key=lambda p: p.frequency_mhz),
        ):
            assert pa.frequency_mhz == pytest.approx(
                pb.frequency_mhz, abs=1e-9
            ), f"window {wid}: peak frequency differs between enabled/disabled pass"

    # When the pass is disabled, no records appear on any window.
    n_alts_off = sum(
        len(getattr(w, "doublet_alternatives", []) or []) for w in fit_off.window_fits
    )
    assert (
        n_alts_off == 0
    ), f"doublet_alternative pass is disabled but {n_alts_off} records appeared"

    # When the pass is enabled on the synthetic fixture, the pair whose
    # separation is 1.0 resolution element must produce at least one record.
    all_alts = [
        alt
        for w in fit_on.window_fits
        for alt in (getattr(w, "doublet_alternatives", None) or [])
    ]
    assert len(all_alts) >= 1, (
        "doublet-alternative pass is enabled and the fixture contains a qualifying "
        "pair (separation = 1.0 res. element, ratio = 0.5), but no adjudication "
        "records were produced — the trigger did not fire"
    )

    # The qualifying pair must have a successful merged refit with finite stats.
    # A NaN result means the refit raised an error that was swallowed silently.
    successful = [a for a in all_alts if a.merged_success]
    assert len(successful) >= 1, (
        f"found {len(all_alts)} adjudication record(s) but none has merged_success=True; "
        "a swallowed refit error (e.g. duplicate kwarg) would produce this pattern"
    )
    rec = successful[0]
    assert math.isfinite(rec.chi2r_merged), (
        f"merged chi2r is not finite ({rec.chi2r_merged!r}); "
        "indicates a failed refit whose exception was silently caught"
    )
    assert math.isfinite(rec.delta_aicc), (
        f"delta_aicc is not finite ({rec.delta_aicc!r}); "
        "indicates a failed refit or degenerate AICc from a bad parameter bag"
    )
    assert math.isfinite(
        rec.orth_evidence_delta_chi2
    ), f"orth_evidence_delta_chi2 is not finite ({rec.orth_evidence_delta_chi2!r})"
    # The merged peak must land between the two production peaks: the amplitude-
    # weighted centroid of a ratio-0.5 pair lies between the two frequencies.
    assert rec.frequency_a_mhz < rec.merged_frequency_mhz < rec.frequency_b_mhz, (
        f"merged_frequency_mhz={rec.merged_frequency_mhz:.6f} is not strictly between "
        f"frequency_a={rec.frequency_a_mhz:.6f} and frequency_b={rec.frequency_b_mhz:.6f}"
    )


def test_no_auto_peak_below_snr_floor_after_fit(
    baseline_2638_stage4_small, temp_ftmw_dir
):
    """After a default Stage 5 fit, no auto-origin peak has snr < 3.2.

    The peak-survival prune is default-on; every surviving auto-origin fitted
    peak must have snr >= 3.2 (or None/NaN snr, which the prune conservatively
    keeps). User-origin peaks are exempt but cannot appear in a fresh automatic
    fit.
    """
    import math

    fp = temp_ftmw_dir / "snr_floor_check.ftmw"
    shutil.copy(baseline_2638_stage4_small, fp)

    fit = ftmw.fit_peaks(fp)
    for peak in fit.fitted_peaks:
        if peak.origin != "user":
            snr = peak.snr
            if snr is not None and not (isinstance(snr, float) and math.isnan(snr)):
                assert snr >= 3.2, (
                    f"auto peak at {peak.frequency_mhz:.4f} MHz survived with "
                    f"snr={snr:.3f} < 3.2 (window {peak.window_id})"
                )

    # The diagnostics dict must carry the peak_survival key.
    ps = fit.diagnostics.get("peak_survival")
    assert ps is not None
    assert "snr_floor" in ps
    assert "n_pruned" in ps
    assert "pruned" in ps
    assert "dropped_window_ids" in ps
    # The survival floor tracks the Stage 3 promotion cutoff (default 3.0) times
    # ``snr_survival_factor`` (default 1.1) -> 3.3.
    assert ps["snr_floor"] == pytest.approx(3.3)
