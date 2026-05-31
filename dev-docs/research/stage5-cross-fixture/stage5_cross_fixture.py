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

Self-contained: builds each fixture into a gitignored scratch path and writes
the roll-up JSON next to this script under ``data/``. ``fit_peaks`` (and the
per-bin Voigt ``calibrate_tau_G``) are the slow stages; ``--reuse`` reloads the
persisted Stage 5 fit and tau calibrations instead of recomputing them. Run from
the repo root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-cross-fixture/stage5_cross_fixture.py [--reuse]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.fitting.tau_calibration import TauCalibrationResult

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# SNR-ascending so the table reads low-to-high; same set the o2/o4 recipe uses.
FIXTURES = ["363", "2638", "360", "1231", "1512", "1019", "655"]
TRIM = (26500.0, 40000.0)
SCRATCH = Path("scratch/stage5_cross_fixture")

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


def build(fid: str, reuse: bool = False) -> tuple[str, SpectrumFit, TauCalibrationResult]:
    """Canonical production pipeline through Stage 5 for one fixture.

    import -> detect_start_time(stamp) -> compute_ft(zpf=0, expf=None, trim) ->
    estimate_noise(scatter) -> calibrate_tau -> detect_peaks -> assign_windows ->
    fit_peaks. With ``reuse=True`` and an already-fit file present, reload the
    persisted ``SpectrumFit`` and tau calibration instead of re-running the slow
    ``fit_peaks`` NLS.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fp = str(SCRATCH / f"exp_{fid}.ftmw")
    if reuse and Path(fp).exists():
        try:
            return fp, ftmw.load_fit(fp), ftmw.load_tau_calibration(fp)
        except Exception:  # noqa: BLE001
            pass  # fall through to a clean rebuild
    ftmw.import_data(fp, source=f"examples/blackchirp_data/{fid}", force=True)
    ftmw.detect_start_time(fp, band=TRIM, stamp=True)
    ftmw.compute_ft(fp, zpf=0, expf_us=None, trim=TRIM)
    ftmw.estimate_noise(fp)  # scatter default
    cal = ftmw.calibrate_tau(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)
    fit = ftmw.fit_peaks(fp)
    return fp, fit, cal


def tier1_metrics(fit: SpectrumFit) -> dict:
    """Per-window reduced-chi2 distribution health (planning-doc Tier 1)."""
    chi2 = np.array(
        [
            float(wf.reduced_chi2)
            for wf in fit.window_fits
            if np.isfinite(getattr(wf, "reduced_chi2", np.inf))
        ]
    )
    if chi2.size == 0:
        return dict(n_windows=0)
    return dict(
        n_windows=int(chi2.size),
        chi2r_median=round(float(np.median(chi2)), 3),
        chi2r_p95=round(float(np.percentile(chi2, 95)), 3),
        chi2r_max=round(float(chi2.max()), 3),
        n_gt10=int((chi2 > 10).sum()),
        n_gt4=int((chi2 > 4).sum()),
        n_le1p5=int((chi2 <= 1.5).sum()),
        frac_le1p5=round(float((chi2 <= 1.5).mean()), 3),
        # Tier-1 acceptance per the planning doc (median<=1.5, p95<=4, max<=10).
        tier1_pass=bool(
            np.median(chi2) <= 1.5
            and np.percentile(chi2, 95) <= 4.0
            and chi2.max() <= 10.0
        ),
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


def capture(fid: str, reuse: bool = False) -> dict:
    fp, fit, cal = build(fid, reuse=reuse)
    t1 = tier1_metrics(fit)
    t2 = tier2_metrics(fit)
    tau = tau_comparison(fp, fit, cal, reuse=reuse)
    out = dict(
        fixture=fid,
        n_windows=t1.get("n_windows", 0),
        n_fitted_peaks=len(fit.fitted_peaks),
        tier1=t1,
        tier2=t2,
        tau=tau,
    )
    print(
        f"[{fid}] win={t1.get('n_windows')} peaks={len(fit.fitted_peaks)} "
        f"| T1 chi2r med={t1.get('chi2r_median')} p95={t1.get('chi2r_p95')} "
        f"max={t1.get('chi2r_max')} pass={t1.get('tier1_pass')} "
        f"| T2 merge={t2['merge_fire_rate']} origin_pruned={t2['n_pruned_rescue_origin']} "
        f"lc={t2['limit_cycle_rounds']} "
        f"| tau stft={tau['tau_stft_us']} G={tau['tau_G_us']} "
        f"rms={tau['tau_rms_us']}{'' if tau['tau_rms_trusted'] else '(untrusted)'}",
        flush=True,
    )
    return out


def main() -> int:
    reuse = "--reuse" in sys.argv[1:]
    results = []
    for fid in FIXTURES:
        try:
            results.append(capture(fid, reuse=reuse))
        except Exception as exc:  # noqa: BLE001
            print(f"[{fid}] FAILED: {exc!r}", flush=True)
            results.append(dict(fixture=fid, error=repr(exc)))
    (DATA / "stage5_cross_fixture.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {DATA / 'stage5_cross_fixture.json'}")

    # Tier 1 distribution-health summary.
    print("\n=== Tier 1: per-window chi2_r health ===")
    print(f"{'fix':>5} {'win':>5} {'median':>7} {'p95':>7} {'max':>9} "
          f"{'>10':>4} {'>4':>4} {'<=1.5':>6} {'pass':>5}")
    for r in results:
        t1 = r.get("tier1")
        if not t1 or not t1.get("n_windows"):
            continue
        print(f"{r['fixture']:>5} {t1['n_windows']:>5} {t1['chi2r_median']:>7.2f} "
              f"{t1['chi2r_p95']:>7.2f} {t1['chi2r_max']:>9.2f} {t1['n_gt10']:>4} "
              f"{t1['n_gt4']:>4} {t1['n_le1p5']:>6} "
              f"{'PASS' if t1['tier1_pass'] else 'FAIL':>5}")

    # Tier 2 gate-firing summary.
    print("\n=== Tier 2: rescue/merge gate firing ===")
    print(f"{'fix':>5} {'merge%':>7} {'origin_pr':>10} {'lim_cyc':>8} "
          f"{'rescue':>7} {'thaw':>9} {'replan':>7}")
    for r in results:
        t2 = r.get("tier2")
        if not t2:
            continue
        print(f"{r['fixture']:>5} {t2['merge_fire_rate']:>7.2f} "
              f"{t2['n_pruned_rescue_origin']:>10} {t2['limit_cycle_rounds']:>8} "
              f"{t2['n_rescue_accepted']}/{t2['n_rescue_rounds']:<5} "
              f"{t2['n_thaw_accepted']}/{t2['n_thaw']:<7} {t2['n_replan']:>7}")

    # Tau-source comparison: the Theme T2 / issue #3 bias lever.
    print("\n=== tau: STFT vs Gaussian vs model-free FID-RMS (us) ===")
    print(f"{'fix':>5} {'stft':>7} {'sig':>6} {'tau_G':>7} {'rms':>7} "
          f"{'r2':>5} {'dynR':>7} {'trust':>5}  stft/rms")
    for r in results:
        tau = r.get("tau")
        if not tau:
            continue
        rms = tau["tau_rms_us"]
        ratio = (round(tau["tau_stft_us"] / rms, 2) if rms and rms > 0 else None)
        print(f"{r['fixture']:>5} {tau['tau_stft_us']:>7.2f} "
              f"{tau['sigma_tau_stft_us']:>6.2f} "
              f"{(tau['tau_G_us'] if tau['tau_G_us'] is not None else float('nan')):>7.2f} "
              f"{(rms if rms is not None else float('nan')):>7.2f} "
              f"{(tau['tau_rms_r2'] if tau['tau_rms_r2'] is not None else float('nan')):>5.2f} "
              f"{tau['tau_rms_dyn_range']:>7.1f} "
              f"{'yes' if tau['tau_rms_trusted'] else 'no':>5}  {ratio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
