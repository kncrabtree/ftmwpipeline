"""Cross-fixture Stage 5 characterization harness (issues #2/#3/#4, Theme T1).

The backbone the cross-fixture validation reads off. For each of the seven
same-instrument production fixtures it builds the canonical pipeline through
Stage 5 and emits, per fixture:

* **Tier 1 -- distribution health.** The post-fit per-window reduced-chi2
  distribution (median / p95 / max, plus the >10 / >4 / <=1.5 occupancy). This
  is the headline Stage-5-healthy gate (planning-doc Tier 1).

* **Tier 2 -- gate firing.** Whether the rescue/merge machinery is doing
  meaningful work: the merge fire rate (windows whose rescue chain collapsed at
  least one close pair), the ``n_pruned_rescue_origin`` failsafe total (the
  knockout undoing a peak the rescue just added), a limit-cycle count (rescue
  rounds that add peaks and merge them back with chi2 unmoved), and the
  rescue/thaw/replan history sizes.

* **The tau comparison (Theme T2's input).** STFT ``calibrate_tau`` tau_maj vs
  the Gaussian ``calibrate_tau_G`` majority vs a model-free decay read straight
  off the raw FID as a sliding-window RMS. A per-line demodulated envelope was
  tried first and rejected: vinyl cyanide's hyperfine splitting makes the
  "isolated" strong lines unresolved multiplets, so narrowband demodulation
  preserves the hyperfine *beat* and reads it as a spuriously fast decay. The
  sliding RMS over the whole FID averages out the carrier and the inter-line
  beats, leaving the gross ``exp(-t/tau)`` power-decay envelope -- robust on the
  strong-line fixtures (where one or a few lines dominate the FID power) and
  flagged low-confidence where the spectrum is noise-dominated. The STFT-vs-RMS
  gap is the per-fixture bias the production tau-source decision (#3) turns on.

Everything is read off the genuine production objects (``SpectrumFit``,
``TauCalibrationResult``, the raw ``FID``) -- no hand-recomputed fit statistics.

Self-contained: builds each fixture into a gitignored per-fixture tree under
``--output-root`` (default ``scratch/issue3-cross-fixture/<fixture>/``) holding
``exp_<id>.ftmw``, ``summary.json``, ``ground_truth.{json,csv}``, and ``viz/``;
the cross-fixture ``rollup.{json,csv}`` land in the root. ``fit_peaks`` (and the
per-bin Voigt ``calibrate_tau_G``) are the slow stages; ``--reuse`` reloads the
persisted Stage 5 fit and tau calibrations instead of recomputing them. The
driver code is tracked; everything it writes under ``scratch/`` is not. Run from
the repo root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-cross-fixture/stage5_cross_fixture.py \
        [--output-root DIR] [--reuse] [--no-viz] [--fixtures 2638,655]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.fitting.tau_calibration import TauCalibrationResult
from ftmwpipeline.fitting.validation import (
    DEFAULT_CHI2R_NOISE_FLOOR,
    DEFAULT_SHAPE_ERROR_KAPPA,
    shape_error_fraction,
    snr_aware_chi2_pass,
)

# SNR-ascending so the table reads low-to-high; same set the o2/o4 recipe uses.
FIXTURES = ["363", "2638", "360", "1231", "1512", "1019", "655"]
TRIM = (26500.0, 40000.0)

# Model-free FID-RMS tau: sliding-RMS window width (raw-FID samples). Wide enough
# to average the carrier and the inter-line beats, narrow vs the decay timescale.
RMS_WIN_PTS = 4000
# The RMS decay tau is only trusted inside this physical range (us), with a
# log-linear decay fit at least this well-determined (R^2), and only when the
# early FID power clears the tail-noise floor by at least this dynamic range
# (otherwise the FID is noise-dominated and the slope is meaningless).
RMS_TAU_LO_US, RMS_TAU_HI_US = 0.3, 20.0
# Trust the RMS tau only on a strong fixture whose dominant line decays as a
# single exponential: amplitude SNR at start >= RMS_DYNRANGE_MIN (excludes the
# weak, noise-dominated fixtures where STFT is the right tool) and a log-linear
# decay R^2 >= RMS_FIT_R2_MIN (excludes beating doublets like 2638, whose
# envelope oscillates and is not a single exponential).
RMS_FIT_R2_MIN = 0.85
RMS_FIT_FLOOR_FRAC = 0.1  # fit the decay down to this fraction of the peak level
RMS_DYNRANGE_MIN = 4.0  # early/tail amplitude SNR below which tau is untrusted
# Limit-cycle test: a rescue round that adds and merges peaks with chi2 unmoved.
LIMIT_CYCLE_CHI2_REL = 0.02


def build(
    fid: str, out_dir: Path, reuse: bool = False
) -> tuple[str, SpectrumFit, TauCalibrationResult, dict]:
    """Canonical production pipeline through Stage 5 for one fixture.

    import -> detect_start_time(stamp) -> compute_ft(trim) ->
    estimate_noise(scatter) -> calibrate_tau -> recommend_shape -> [calibrate_tau_G
    if gaussian] -> detect_peaks -> assign_windows -> fit_peaks(shape). The fit
    runs in each fixture's *recommended* shape (the per-line L/G/V AICc vote):
    2638 is gaussian (consuming the ``calibrate_tau_G`` band majorities), 655 is
    lorentzian (the STFT ``calibrate_tau`` band majorities). Fitting the wrong
    shape inflates chi2r via the Lorentzian-core/Gaussian-wing residual, so the
    cross-fixture Tier-1 numbers are only meaningful in the correct shape.
    ``per_band_tau`` rides on the hard-default True (the resolver picks it up),
    so the per-band majorities route per window without an explicit kwarg.

    The ``exp_<id>.ftmw`` is written into ``out_dir`` (the issue-3 per-fixture
    tree, gitignored). With ``reuse=True`` and an already-fit file present,
    reload the persisted ``SpectrumFit`` and tau calibration instead of
    re-running the slow NLS. Returns the start-detection diagnostics as the
    fourth element so the per-fixture summary can record the detected
    ``start_us`` (chirp_end + guard).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = str(out_dir / f"exp_{fid}.ftmw")
    if reuse and Path(fp).exists():
        try:
            fit = ftmw.load_fit(fp)
            cal = ftmw.load_tau_calibration(fp)
            # Start-detection diagnostics are not re-derivable from the fit;
            # re-run the cheap detector against the persisted FID for the
            # provenance record (no stamping needed on reuse).
            sd = ftmw.detect_start_time(fp, band=TRIM, stamp=False)
            start = dict(
                start_us=float(sd.start_us),
                chirp_end_us=float(sd.chirp_end_us),
                guard_us=round(float(sd.start_us) - float(sd.chirp_end_us), 4),
                chirp_detected=bool(sd.chirp_detected),
            )
            return fp, fit, cal, start
        except Exception:  # noqa: BLE001
            pass  # fall through to a clean rebuild
    ftmw.import_data(fp, source=f"examples/blackchirp_data/{fid}", force=True)
    sd = ftmw.detect_start_time(fp, band=TRIM, stamp=True)
    start = dict(
        start_us=float(sd.start_us),
        chirp_end_us=float(sd.chirp_end_us),
        guard_us=round(float(sd.start_us) - float(sd.chirp_end_us), 4),
        chirp_detected=bool(sd.chirp_detected),
    )
    ftmw.compute_ft(fp, trim=TRIM)
    ftmw.estimate_noise(fp)  # scatter default
    cal = ftmw.calibrate_tau(fp)
    # recommend_shape returns "lorentzian" (exp wins) / "gaussian" (gauss wins) /
    # None (no clear winner -- voigt mass reported but unrepresentable here);
    # fall through to the Lorentzian default in the no-winner case.
    shape = ftmw.recommend_shape(fp).recommended_shape or "lorentzian"
    if shape == "gaussian":
        ftmw.calibrate_tau_G(fp)  # the gaussian-path tau_G band majorities
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)
    fit = ftmw.fit_peaks(fp, shape=shape)
    return fp, fit, cal, start


# SNR_max bins -- the natural breakdown (noise-dominated bulk -> floor-limited
# bright cores), matching ``_internal/stage5_validation_impl.py``.
SNR_BIN_EDGES = (100.0, 1000.0, 10000.0)
SNR_BIN_LABELS = ("<100", "100-1k", "1k-10k", ">=10k")


def _window_snr_max(wf) -> float:
    """Brightest in-window fitted-peak SNR (0 for an empty / SNR-less window)."""
    snrs = [
        float(p.snr)
        for p in wf.fitted_peaks
        if p.snr is not None and np.isfinite(p.snr)
    ]
    return max(snrs) if snrs else 0.0


def _snr_bin(snr: float) -> str:
    for i, edge in enumerate(SNR_BIN_EDGES):
        if snr < edge:
            return SNR_BIN_LABELS[i]
    return SNR_BIN_LABELS[-1]


def tier1_metrics(
    fit: SpectrumFit,
    kappa: float = DEFAULT_SHAPE_ERROR_KAPPA,
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR,
) -> dict:
    """SNR-aware per-window acceptance (planning-doc Tier 1, post-D10).

    The raw chi2r gate (median<=1.5/p95<=4/max<=10) is a model-fidelity-vs-SNR
    floor, not a health metric, so it is unachievable at extreme SNR. The gate
    is now ``chi2r <= F + (kappa*SNR_max)**2`` per window (F the noise-regime
    allowance) with the fractional deficit ``eps`` reported, binned by the
    brightest in-window peak SNR.
    """
    rows = []
    for wf in fit.window_fits:
        chi2r = float(getattr(wf, "reduced_chi2", np.inf))
        snr_max = _window_snr_max(wf)
        rows.append(
            dict(
                chi2r=chi2r,
                snr_max=snr_max,
                eps=shape_error_fraction(chi2r, snr_max, noise_floor),
                snr_bin=_snr_bin(snr_max),
                passed=snr_aware_chi2_pass(chi2r, snr_max, kappa, noise_floor),
            )
        )
    if not rows:
        return dict(n_windows=0)

    finite = np.array([r["chi2r"] for r in rows if np.isfinite(r["chi2r"])])
    n_pass = sum(r["passed"] for r in rows)
    bins = {}
    for label in SNR_BIN_LABELS:
        brows = [r for r in rows if r["snr_bin"] == label]
        if not brows:
            continue
        bfin = np.array([r["chi2r"] for r in brows if np.isfinite(r["chi2r"])])
        bins[label] = dict(
            n=len(brows),
            chi2r_median=round(float(np.median(bfin)), 3) if bfin.size else None,
            eps_median=round(float(np.median([r["eps"] for r in brows])), 5),
            pass_rate=round(sum(r["passed"] for r in brows) / len(brows), 3),
        )
    return dict(
        n_windows=len(rows),
        kappa=kappa,
        noise_floor=noise_floor,
        pass_rate=round(n_pass / len(rows), 3),
        n_pass=int(n_pass),
        # Raw distribution kept for reference (the superseded gate's numbers).
        chi2r_median=round(float(np.median(finite)), 3) if finite.size else None,
        chi2r_p95=round(float(np.percentile(finite, 95)), 3) if finite.size else None,
        chi2r_max=round(float(finite.max()), 3) if finite.size else None,
        snr_bins=bins,
        tier1_pass=bool(n_pass == len(rows)),
    )


def tier2_metrics(fit: SpectrumFit) -> dict:
    """Rescue/merge gate firing rates (planning-doc Tier 2)."""
    n_windows = len(fit.window_fits)
    rounds = list(fit.rescue_history)
    merged_windows = {r.window_id for r in rounds if r.n_merged > 0}
    n_origin_pruned = sum(int(r.n_pruned_rescue_origin) for r in rounds)
    limit_cycle = 0
    for r in rounds:
        denom = max(abs(r.chi2_before), 1e-12)
        unmoved = abs(r.chi2_after - r.chi2_before) / denom < LIMIT_CYCLE_CHI2_REL
        if unmoved and r.n_rescue_added > 0 and r.n_merged > 0:
            limit_cycle += 1
    return dict(
        merge_fire_windows=len(merged_windows),
        merge_fire_rate=round(len(merged_windows) / n_windows, 3) if n_windows else 0.0,
        n_pruned_rescue_origin=int(n_origin_pruned),
        limit_cycle_rounds=int(limit_cycle),
        n_rescue_rounds=len(rounds),
        n_rescue_accepted=sum(1 for r in rounds if r.accepted),
        n_thaw=len(fit.thaw_history),
        n_thaw_accepted=sum(1 for t in fit.thaw_history if t.accepted),
        n_replan=len(fit.replan_history),
        final_plan_revision=int(fit.final_plan_revision),
    )


def _fid_rms_tau_us(y: np.ndarray, dt_us: float, start_us: float) -> dict:
    """Model-free decay constant from the gross sliding-RMS of the raw FID.

    The real FID is a sum of decaying tones on a fast carrier. A sliding RMS
    over ``RMS_WIN_PTS`` samples averages out the carrier and the inter-line
    beats (so it does *not* mistake hyperfine beating for fast decay the way a
    narrowband per-line demodulation does), leaving the gross ``exp(-t/tau)``
    power envelope; ``log(RMS)`` is then fit log-linearly over the clean decay
    region (above the tail-noise floor, down to ``RMS_FIT_FLOOR_FRAC`` of the
    peak). Trustworthy only when the early FID power clears the tail noise by
    ``RMS_DYNRANGE_MIN`` -- otherwise the FID is noise-dominated. Returns the
    fitted ``tau_us`` (NaN when non-physical, ill-determined, or low dynamic
    range) plus the diagnostics (``r2``, ``dyn_range``, ``trusted``).

    The FID is first truncated to the active region (``start_us`` onward, which
    drops the chirp burst) and its DC offset is subtracted. The order matters:
    these spectra carry a true DC offset, and estimating it over the full FID
    would let the large, asymmetric chirp bias the mean -- leaving a residual
    constant under the RMS that never decays and caps the dynamic range (this is
    what flattened the earlier measure to dyn-range ~0.4 on 2638).
    """
    i0 = int(round(start_us / dt_us))
    yy = np.asarray(y[i0:], dtype=float)
    yy = yy - yy.mean()  # subtract the active-region DC (chirp already excluded)
    t = np.arange(yy.size) * dt_us
    kernel = np.ones(RMS_WIN_PTS) / RMS_WIN_PTS
    ms = np.convolve(yy * yy, kernel, mode="same")  # sliding mean-square
    # The flat broadband-noise pedestal adds in quadrature to the decaying
    # coherent power (ms = S^2 e^{-2t/tau} + N^2). Subtract the mean-square noise
    # floor before fitting, else the curve flattens toward N and the slope reads
    # tau too long. Fit only where the coherent power clears the noise.
    tail_from = int((9.0 - start_us) / dt_us)  # 9-15 us tail ~ noise
    noise2 = float(np.median(ms[tail_from:])) if tail_from < ms.size else float(ms.min())
    sig = np.sqrt(np.clip(ms - noise2, 0.0, None))
    e0 = float(np.median(sig[: max(1, int(0.5 / dt_us))]))  # first ~0.5 us level
    dyn_range = float(np.sqrt(ms[: max(1, int(0.5 / dt_us))].mean() / noise2)) if noise2 > 0 else float("inf")
    good = (t > 0.3) & (ms > 2.0 * noise2) & (sig > RMS_FIT_FLOOR_FRAC * e0)
    out = dict(tau_rms_us=None, r2=None, dyn_range=round(dyn_range, 1),
               trusted=False)
    if int(good.sum()) < 200:
        return out
    x, ly = t[good], np.log(sig[good])
    slope, intercept = np.polyfit(x, ly, 1)
    resid = ly - (slope * x + intercept)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
    tau = -1.0 / slope if slope < 0 else float("nan")
    out["r2"] = round(r2, 3)
    physical = np.isfinite(tau) and RMS_TAU_LO_US <= tau <= RMS_TAU_HI_US
    trusted = bool(physical and r2 >= RMS_FIT_R2_MIN and dyn_range >= RMS_DYNRANGE_MIN)
    out["trusted"] = trusted
    if physical:
        out["tau_rms_us"] = round(float(tau), 3)
    return out


def tau_comparison(fp: str, fit: SpectrumFit, cal: TauCalibrationResult,
                   reuse: bool) -> dict:
    """STFT tau_maj vs Gaussian tau_G vs model-free demodulated-envelope tau."""
    # Gaussian-shape majority: slow per-bin Voigt fits -- reuse if persisted.
    tau_g = None
    try:
        cal_g = None
        if reuse:
            try:
                cal_g = ftmw.load_tau_G_calibration(fp)
            except Exception:  # noqa: BLE001
                cal_g = None
        if cal_g is None:
            cal_g = ftmw.calibrate_tau_G(fp)
        tau_g = round(float(cal_g.tau_maj_us), 4)
    except Exception as exc:  # noqa: BLE001
        tau_g = None
        print(f"    (tau_G unavailable: {exc!r})", flush=True)

    # Model-free decay from the gross sliding-RMS of the raw FID.
    fid = ftmw.load_fid(fp)
    y = np.asarray(fid.data, dtype=float)
    dt_us = float(fid.spacing) * 1e6
    start_us = float(cal.start_us)
    rms = _fid_rms_tau_us(y, dt_us, start_us)
    return dict(
        tau_stft_us=round(float(cal.tau_maj_us), 4),
        sigma_tau_stft_us=round(float(cal.sigma_tau_us), 4),
        tau_G_us=tau_g,
        tau_rms_us=rms["tau_rms_us"],
        tau_rms_trusted=rms["trusted"],
        tau_rms_r2=rms["r2"],
        tau_rms_dyn_range=rms["dyn_range"],
    )


def _pct(arr: np.ndarray, q: float) -> float | None:
    a = np.asarray([v for v in arr if np.isfinite(v)], dtype=float)
    return round(float(np.percentile(a, q)), 4) if a.size else None


def _med(arr) -> float | None:
    a = np.asarray([v for v in arr if np.isfinite(v)], dtype=float)
    return round(float(np.median(a)), 4) if a.size else None


# ---------------------------------------------------------------------------
# Provenance / Stage 3 / Stage 4 / Stage 5 collectors (issue #3 summary.json)
# ---------------------------------------------------------------------------


def provenance_metrics(
    cal: TauCalibrationResult, shape: str, start: dict, tau_dict: dict
) -> dict:
    """Detected start, trim, recommended shape, per-band tau majorities, and the
    three-way tau-source comparison (STFT vs Gaussian-tau_G vs model-free RMS)."""
    band_maj = [
        dict(
            lo=round(float(b.freq_lo_mhz), 1),
            hi=round(float(b.freq_hi_mhz), 1),
            tau_maj_us=round(float(b.tau_maj_us), 4),
            sigma_tau_us=round(float(b.sigma_tau_us), 4),
            n=int(b.n),
        )
        for b in cal.band_majorities
    ]
    thirds = [
        dict(
            label=t.label,
            lo=round(float(t.freq_lo_mhz), 1),
            hi=round(float(t.freq_hi_mhz), 1),
            median_tau_us=round(float(t.median_tau_us), 4),
            n=int(t.n),
        )
        for t in cal.frequency_thirds
    ]
    return dict(
        start_us=round(float(start["start_us"]), 4),
        chirp_end_us=round(float(start["chirp_end_us"]), 4),
        guard_us=start["guard_us"],
        chirp_detected=start["chirp_detected"],
        trim_mhz=list(TRIM),
        recommended_shape=shape,
        tau_band_majorities=band_maj,
        tau_frequency_thirds=thirds,
        tau_source_comparison=tau_dict,
    )


def stage3_metrics(peaks) -> dict:
    """Stage 3 detection counts + promoted-peak SNR distribution."""
    n_detected = len(peaks)
    promoted = [p for p in peaks if p.properties.get("promoted", False)]
    snrs = [
        float(p.snr) for p in promoted if p.snr is not None and np.isfinite(p.snr)
    ]
    snr_arr = np.asarray(snrs, dtype=float)
    return dict(
        n_detected=int(n_detected),
        n_promoted=int(len(promoted)),
        promoted_snr_min=round(float(snr_arr.min()), 3) if snr_arr.size else None,
        promoted_snr_median=_med(snr_arr),
        promoted_snr_p95=_pct(snr_arr, 95),
        promoted_snr_max=round(float(snr_arr.max()), 3) if snr_arr.size else None,
    )


# A "mega-window" is a force-merged / very wide window: above this width the
# window has lumped spatially-distinct lines (the GHz-mega-window failure mode
# the bounded merge fixed -- post-fix nothing should exceed ~40 MHz).
MEGA_WINDOW_MHZ = 60.0


def stage4_metrics(plan) -> dict:
    """Window count, difficulty split, width distribution, mega-window flag."""
    widths = np.asarray([float(w.width_mhz) for w in plan.windows], dtype=float)
    n_hard = sum(
        1 for w in plan.windows if w.difficulty.value == "hard"
    )
    mega = [
        dict(window_id=int(w.window_id), width_mhz=round(float(w.width_mhz), 2))
        for w in plan.windows
        if float(w.width_mhz) >= MEGA_WINDOW_MHZ
    ]
    return dict(
        n_windows=len(plan.windows),
        n_hard=int(n_hard),
        n_easy=int(len(plan.windows) - n_hard),
        width_p50_mhz=_med(widths),
        width_p95_mhz=_pct(widths, 95),
        width_max_mhz=round(float(widths.max()), 2) if widths.size else None,
        n_mega_windows=len(mega),
        mega_windows=mega,
        plan_revision=int(plan.plan_revision),
    )


def stage5_metrics(fit: SpectrumFit) -> dict:
    """Per-window chi2r occupancy, sigma_f distribution, free-tau count, and the
    line count, beyond the Tier-1/Tier-2 already collected separately."""
    chi2r = np.asarray(
        [float(getattr(wf, "reduced_chi2", np.inf)) for wf in fit.window_fits],
        dtype=float,
    )
    fin = chi2r[np.isfinite(chi2r)]
    occ_gt10 = int(np.sum(fin > 10)) if fin.size else 0
    occ_gt4 = int(np.sum(fin > 4)) if fin.size else 0
    occ_le15 = int(np.sum(fin <= 1.5)) if fin.size else 0
    # A window fit tau freely iff its shared tau carries a finite stderr (the
    # frozen / weak-window path records error=None).
    n_free_tau = 0
    for wf in fit.window_fits:
        terr = wf.shared_parameters.get("tau_us", {}).get("error")
        if terr is not None and np.isfinite(float(terr)):
            n_free_tau += 1
    # sigma_f distribution over all fitted lines (frequency_error MHz -> kHz).
    sigfs = np.asarray(
        [
            float(p.frequency_error) * 1e3
            for p in fit.fitted_peaks
            if p.frequency_error is not None and np.isfinite(p.frequency_error)
        ],
        dtype=float,
    )
    return dict(
        n_fitted_lines=len(fit.fitted_peaks),
        chi2r_occ_gt10=occ_gt10,
        chi2r_occ_gt4=occ_gt4,
        chi2r_occ_le1p5=occ_le15,
        n_free_tau_windows=n_free_tau,
        sigma_f_median_khz=_med(sigfs),
        sigma_f_p95_khz=_pct(sigfs, 95),
        n_sigma_f=int(sigfs.size),
    )


# ---------------------------------------------------------------------------
# Tier 3 -- ground-truth scoring (issue #3 work item 4)
# ---------------------------------------------------------------------------

VC_TRUTH = Path("dev-docs/fixtures/1512-vinyl-cyanide-truth")
# Per-fixture catalog source. 1512 = the clean single-species v=0 list; 655 =
# the dense 5-species union. 1019's truth is held outside the repo (see
# dev-docs/fixtures/1019.md) -- scored only if a local file is dropped in.
CATALOGS = {
    "1512": ("lines.csv", VC_TRUTH / "lines.csv"),
    "655": ("combined_lines.csv", VC_TRUTH / "combined_lines.csv"),
}
# Calibration-grade catalog uncertainty ceiling (MHz). Lines predicted to worse
# than this (the 15N predicted tail in the union, up to ~22 MHz) are excluded
# from frequency/uncertainty scoring but kept for attribution/recall.
CAT_UNC_CEILING_MHZ = 0.005
# Mutual-nearest match tolerance (MHz). ~0.5*FWHM for a tau~4 us Lorentzian
# (FWHM 1/(pi*4) ~ 80 kHz) plus the ~10-16 kHz instrument accuracy floor.
MATCH_TOL_MHZ = 0.05


def _load_catalog(fid: str) -> dict | None:
    """Parse the per-fixture catalog into in-band (freq, unc, log_int) rows."""
    import csv

    entry = CATALOGS.get(fid)
    if entry is None:
        return None
    label, path = entry
    if not Path(path).exists():
        return None
    lo, hi = TRIM
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                f = float(row["freq_mhz"])
            except (KeyError, ValueError):
                continue
            if not (lo <= f <= hi):
                continue
            unc = float(row.get("unc_mhz") or "nan")
            log_int = float(row.get("log_intensity") or "nan")
            rows.append(dict(freq_mhz=f, unc_mhz=unc, log_intensity=log_int))
    rows.sort(key=lambda r: r["freq_mhz"])
    return dict(label=label, path=str(path), rows=rows)


def _mutual_nearest(fit_f: np.ndarray, cat_f: np.ndarray, tol: float):
    """Mutual-nearest matches within ``tol`` MHz. Returns list of (i_fit, j_cat)."""
    if fit_f.size == 0 or cat_f.size == 0:
        return []
    # nearest cat for each fit, nearest fit for each cat; keep mutual pairs.
    fit_to_cat = np.array([int(np.argmin(np.abs(cat_f - f))) for f in fit_f])
    cat_to_fit = np.array([int(np.argmin(np.abs(fit_f - c))) for c in cat_f])
    pairs = []
    for i, j in enumerate(fit_to_cat):
        if cat_to_fit[j] == i and abs(fit_f[i] - cat_f[j]) <= tol:
            pairs.append((i, int(j)))
    return pairs


def score_ground_truth(fid: str, fit: SpectrumFit) -> dict | None:
    """Score fitted frequency AND uncertainty against the catalog (Tier 3).

    Mutual-nearest match within ``MATCH_TOL_MHZ``; report recall, the frequency
    residual distribution, the detrended instrument accuracy floor (linear
    clock drift + offset), and the pull (= residual / sigma_f) distribution.
    The dense union match is noisier than the sparse single-species one (blended
    hyperfine multiplets), so the per-fixture record carries the caveat.
    """
    cat = _load_catalog(fid)
    if cat is None or not cat["rows"]:
        return None
    lo, hi = TRIM
    fit_peaks = [
        p
        for p in fit.fitted_peaks
        if p.frequency_mhz is not None and lo <= float(p.frequency_mhz) <= hi
    ]
    fit_f = np.asarray([float(p.frequency_mhz) for p in fit_peaks], dtype=float)
    cat_rows = cat["rows"]
    cat_f = np.asarray([r["freq_mhz"] for r in cat_rows], dtype=float)
    # Calibration-grade subset for accuracy/uncertainty scoring.
    cal_mask = np.asarray(
        [
            (np.isfinite(r["unc_mhz"]) and r["unc_mhz"] <= CAT_UNC_CEILING_MHZ)
            for r in cat_rows
        ]
    )

    pairs = _mutual_nearest(fit_f, cat_f, MATCH_TOL_MHZ)
    matched_cat = {j for _, j in pairs}
    # Recall against the full in-band catalog (detectability-bounded) and against
    # the calibration-grade subset.
    n_cat = len(cat_rows)
    n_cat_cal = int(cal_mask.sum())
    matched_cal = sum(1 for j in matched_cat if cal_mask[j])

    # Residuals + pulls on the calibration-grade matched pairs only.
    resid_khz, pulls, sigfs_khz, fmatch_ghz = [], [], [], []
    for i, j in pairs:
        if not cal_mask[j]:
            continue
        r_mhz = fit_f[i] - cat_f[j]
        resid_khz.append(r_mhz * 1e3)
        fmatch_ghz.append(cat_f[j] / 1e3)
        sf = fit_peaks[i].frequency_error
        if sf is not None and np.isfinite(sf) and sf > 0:
            sigfs_khz.append(float(sf) * 1e3)
            pulls.append(r_mhz / float(sf))
    resid = np.asarray(resid_khz, dtype=float)
    fghz = np.asarray(fmatch_ghz, dtype=float)
    pulls_a = np.asarray(pulls, dtype=float)

    detrend = None
    if resid.size >= 5:
        x = fghz - fghz.mean()
        slope, offset = np.polyfit(x, resid, 1)  # kHz per GHz, kHz at mean f
        det = resid - (slope * x + offset)
        detrend = dict(
            slope_khz_per_ghz=round(float(slope), 3),
            offset_khz=round(float(resid.mean()), 3),
            raw_rms_khz=round(float(np.sqrt(np.mean(resid**2))), 3),
            detrended_rms_khz=round(float(np.sqrt(np.mean(det**2))), 3),
        )

    pull_block = None
    if pulls_a.size:
        pull_block = dict(
            n=int(pulls_a.size),
            pull_median=round(float(np.median(pulls_a)), 3),
            pull_std=round(float(np.std(pulls_a)), 3),
            frac_within_1sigma=round(float(np.mean(np.abs(pulls_a) <= 1)), 3),
            frac_within_2sigma=round(float(np.mean(np.abs(pulls_a) <= 2)), 3),
            sigma_f_median_khz=_med(sigfs_khz),
            abs_resid_median_khz=round(float(np.median(np.abs(resid))), 3)
            if resid.size
            else None,
        )
        # The headline overconfidence factor: median |residual| / median sigma_f.
        if pull_block["sigma_f_median_khz"]:
            pull_block["accuracy_over_precision"] = round(
                float(np.median(np.abs(resid))) / pull_block["sigma_f_median_khz"], 1
            )

    # Spurious fitted lines: in-band fitted peaks with NO catalog match. For a
    # single-species catalog (1512) this conflates real other-species lines with
    # true false positives, so it is reported, not interpreted as precision.
    n_unmatched_fit = len(fit_peaks) - len({i for i, _ in pairs})

    return dict(
        catalog=cat["label"],
        n_catalog_inband=n_cat,
        n_catalog_calgrade=n_cat_cal,
        n_fitted_inband=len(fit_peaks),
        n_matched=len(pairs),
        n_matched_calgrade=matched_cal,
        recall_all=round(len(matched_cat) / n_cat, 3) if n_cat else None,
        recall_calgrade=round(matched_cal / n_cat_cal, 3) if n_cat_cal else None,
        n_unmatched_fitted=int(n_unmatched_fit),
        match_tol_mhz=MATCH_TOL_MHZ,
        freq_resid_abs_median_khz=round(float(np.median(np.abs(resid))), 3)
        if resid.size
        else None,
        freq_resid_abs_p95_khz=_pct(np.abs(resid), 95),
        detrend=detrend,
        pull=pull_block,
        note=(
            "dense multi-species union; mutual-nearest match is mismatch-noisy "
            "on blended hyperfine multiplets -- detrended RMS is contaminated, "
            "not the clean instrument floor"
            if fid == "655"
            else "sparse single-species (v=0) list -- the clean accuracy read"
        ),
    )


def _write_ground_truth_files(out_dir: Path, fid: str, gt: dict | None) -> None:
    """Persist ground_truth.{json,csv} in the per-fixture dir."""
    import csv

    (out_dir / "ground_truth.json").write_text(
        json.dumps(gt if gt is not None else {"available": False}, indent=2)
    )
    with open(out_dir / "ground_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        if gt is None:
            w.writerow(["available"])
            w.writerow([False])
            return
        w.writerow(["metric", "value"])
        for k, v in gt.items():
            if isinstance(v, (dict, list)):
                w.writerow([k, json.dumps(v)])
            else:
                w.writerow([k, v])


# ---------------------------------------------------------------------------
# Visualization driver (subprocess into generate_validation.py)
# ---------------------------------------------------------------------------

VIZ_SCRIPT = Path("scripts/development/stage5-validation/generate_validation.py")


def run_viz(fp: str, viz_dir: Path, seed: int = 1234) -> dict:
    """Render the issue-3 viz set: 3 top-SNR, 3 worst-by-eps, 4 random windows.

    Three separate invocations into disjoint subdirs (top-snr / worst-eps /
    random) so each gets its own overview.png + INDEX.md without clobbering.
    Returns a per-selector ok/failed map; viz failure is logged, not fatal.
    """
    import subprocess

    selectors = [
        ("top-snr", ["--top-snr", "3"]),
        ("worst-eps", ["--worst-eps", "3"]),
        ("random", ["--random-sample", "4", "--seed", str(seed)]),
    ]
    out = {}
    for name, sel in selectors:
        sub = viz_dir / name
        cmd = [
            sys.executable,
            str(VIZ_SCRIPT),
            "--ftmw-path",
            fp,
            "--output-dir",
            str(sub),
            *sel,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            out[name] = "ok"
        except subprocess.CalledProcessError as exc:  # noqa: BLE001
            out[name] = f"FAILED: {exc.stderr[-400:] if exc.stderr else exc}"
            print(f"    (viz {name} failed: {out[name]})", flush=True)
    return out


def capture(
    fid: str, root: Path, reuse: bool = False, do_viz: bool = True, seed: int = 1234
) -> dict:
    """Build one fixture, collect the full summary, and write the per-fixture
    tree (summary.json, ground_truth.{json,csv}, viz/)."""
    out_dir = root / fid
    fp, fit, cal, start = build(fid, out_dir, reuse=reuse)
    shape = str(fit.parameters.get("shape", "lorentzian"))
    tau = tau_comparison(fp, fit, cal, reuse=reuse)
    t1 = tier1_metrics(fit)
    t2 = tier2_metrics(fit)
    peaks = ftmw.load_peaks(fp)
    plan = ftmw.load_windows(fp)
    prov = provenance_metrics(cal, shape, start, tau)
    s3 = stage3_metrics(peaks)
    s4 = stage4_metrics(plan)
    s5 = stage5_metrics(fit)
    gt = score_ground_truth(fid, fit)
    _write_ground_truth_files(out_dir, fid, gt)

    viz = {}
    if do_viz:
        viz = run_viz(fp, out_dir / "viz", seed=seed)

    summary = dict(
        fixture=fid,
        ftmw_path=fp,
        shape=shape,
        provenance=prov,
        stage3=s3,
        stage4=s4,
        stage5=s5,
        tier1=t1,
        tier2=t2,
        ground_truth=gt,
        viz=viz,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    bulk = (t1.get("snr_bins") or {}).get("<100", {})
    print(
        f"[{fid}] shape={shape} win={s4['n_windows']}"
        f"(mega={s4['n_mega_windows']}) lines={s5['n_fitted_lines']} "
        f"| det/prom={s3['n_detected']}/{s3['n_promoted']} "
        f"| T1 pass={t1.get('pass_rate')} bulk_med={bulk.get('chi2r_median')} "
        f"| free_tau={s5['n_free_tau_windows']} sigf_med={s5['sigma_f_median_khz']}kHz "
        f"| tau stft={tau['tau_stft_us']} G={tau['tau_G_us']} "
        f"rms={tau['tau_rms_us']}{'' if tau['tau_rms_trusted'] else '(unt)'}"
        + (
            f" | GT recall={gt.get('recall_all')} "
            f"resid_med={gt.get('freq_resid_abs_median_khz')}kHz"
            if gt
            else " | GT n/a"
        ),
        flush=True,
    )
    return summary


def write_rollup(results: list[dict], root: Path) -> None:
    """Cross-fixture rollup.{json,csv} (one row per fixture)."""
    import csv

    (root / "rollup.json").write_text(json.dumps(results, indent=2))
    cols = [
        "fixture",
        "shape",
        "start_us",
        "n_detected",
        "n_promoted",
        "prom_snr_median",
        "prom_snr_max",
        "n_windows",
        "n_hard",
        "width_p95_mhz",
        "width_max_mhz",
        "n_mega_windows",
        "n_fitted_lines",
        "tier1_pass",
        "chi2r_bulk_median",
        "chi2r_bulk_eps_pct",
        "chi2r_p95",
        "n_free_tau",
        "sigma_f_median_khz",
        "merge_fire_rate",
        "n_rescue_accepted",
        "n_rescue_rounds",
        "n_thaw_accepted",
        "n_replan",
        "n_pruned_rescue_origin",
        "limit_cycle_rounds",
        "tau_stft_us",
        "tau_G_us",
        "tau_rms_us",
        "tau_rms_trusted",
        "gt_recall_all",
        "gt_resid_median_khz",
        "gt_detrended_rms_khz",
        "gt_acc_over_prec",
    ]
    with open(root / "rollup.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            if "error" in r:
                w.writerow([r["fixture"]] + ["ERROR"] * (len(cols) - 1))
                continue
            prov, s3, s4, s5 = (
                r["provenance"],
                r["stage3"],
                r["stage4"],
                r["stage5"],
            )
            t1, t2, tau = r["tier1"], r["tier2"], prov["tau_source_comparison"]
            bulk = (t1.get("snr_bins") or {}).get("<100", {})
            gt = r.get("ground_truth") or {}
            det = (gt.get("detrend") or {}) if gt else {}
            w.writerow(
                [
                    r["fixture"],
                    r["shape"],
                    prov["start_us"],
                    s3["n_detected"],
                    s3["n_promoted"],
                    s3["promoted_snr_median"],
                    s3["promoted_snr_max"],
                    s4["n_windows"],
                    s4["n_hard"],
                    s4["width_p95_mhz"],
                    s4["width_max_mhz"],
                    s4["n_mega_windows"],
                    s5["n_fitted_lines"],
                    t1.get("pass_rate"),
                    bulk.get("chi2r_median"),
                    round((bulk.get("eps_median") or 0) * 100, 3),
                    t1.get("chi2r_p95"),
                    s5["n_free_tau_windows"],
                    s5["sigma_f_median_khz"],
                    t2["merge_fire_rate"],
                    t2["n_rescue_accepted"],
                    t2["n_rescue_rounds"],
                    t2["n_thaw_accepted"],
                    t2["n_replan"],
                    t2["n_pruned_rescue_origin"],
                    t2["limit_cycle_rounds"],
                    tau["tau_stft_us"],
                    tau["tau_G_us"],
                    tau["tau_rms_us"],
                    tau["tau_rms_trusted"],
                    gt.get("recall_all"),
                    gt.get("freq_resid_abs_median_khz"),
                    det.get("detrended_rms_khz"),
                    (gt.get("pull") or {}).get("accuracy_over_precision"),
                ]
            )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Issue #3 cross-fixture assessment: build all 7 same-instrument "
            "fixtures through Stage 5 at default settings, emit a per-fixture "
            "summary tree + a cross-fixture rollup. Assess-and-recommend only; "
            "does not change shipped defaults."
        )
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="scratch/issue3-cross-fixture",
        help="Parent dir for the per-fixture tree (gitignored).",
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="Reload persisted fits/calibrations instead of re-running fit_peaks.",
    )
    ap.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip the per-fixture visualization rendering (faster).",
    )
    ap.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="Comma-separated subset of fixture ids (default: all 7).",
    )
    ap.add_argument(
        "--viz-seed",
        type=int,
        default=1234,
        help="Seed for the random-window viz selector (default 1234).",
    )
    args = ap.parse_args()

    root = Path(args.output_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    fixtures = (
        [f.strip() for f in args.fixtures.split(",") if f.strip()]
        if args.fixtures
        else list(FIXTURES)
    )

    results = []
    for fid in fixtures:
        try:
            results.append(
                capture(
                    fid,
                    root,
                    reuse=args.reuse,
                    do_viz=not args.no_viz,
                    seed=args.viz_seed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[{fid}] FAILED: {exc!r}", flush=True)
            results.append(dict(fixture=fid, error=repr(exc)))

    write_rollup(results, root)
    print(f"\nwrote {root / 'rollup.json'} and {root / 'rollup.csv'}", flush=True)

    # Headline cross-fixture table to stdout (the ASSESSMENT.md is authored
    # separately from the rollup once the run lands).
    print("\n=== cross-fixture rollup (SNR-ascending) ===")
    print(
        f"{'fix':>5} {'shape':>6} {'win':>5} {'mega':>5} {'lines':>6} "
        f"{'T1pass':>7} {'bulkmed':>8} {'p95':>9} {'freeT':>6} "
        f"{'sigf':>7} {'stft':>6} {'tauG':>6} {'rms':>6} {'recall':>7}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['fixture']:>5}  ERROR: {r['error'][:60]}")
            continue
        s4, s5, t1 = r["stage4"], r["stage5"], r["tier1"]
        tau = r["provenance"]["tau_source_comparison"]
        bulk = (t1.get("snr_bins") or {}).get("<100", {})
        gt = r.get("ground_truth") or {}

        def _f(v, w=6, p=2):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

        print(
            f"{r['fixture']:>5} {r['shape']:>6} {s4['n_windows']:>5} "
            f"{s4['n_mega_windows']:>5} {s5['n_fitted_lines']:>6} "
            f"{_f(t1.get('pass_rate'),7,3)} {_f(bulk.get('chi2r_median'),8)} "
            f"{_f(t1.get('chi2r_p95'),9)} {s5['n_free_tau_windows']:>6} "
            f"{_f(s5['sigma_f_median_khz'],7)} {_f(tau['tau_stft_us'])} "
            f"{_f(tau['tau_G_us'])} {_f(tau['tau_rms_us'])} "
            f"{_f(gt.get('recall_all'),7,3)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
