"""STFT vs LSQ-fit-and-histogram tau cross-comparison on 2638.

Loads the persisted Stage 5 fit from the unapodized 2638 pipeline file
(built by hand with ``expf_us=None`` so no apodization runs and Stage 2b
is not consumed), extracts per-window tau from EASY K=1 windows that pass
the difficulty / SNR / uncertainty / chi-squared / non-saturating filter
gates, and fits a Gaussian to the resulting distribution. The result is
the LSQ counterpart to the STFT majority tau documented in
[report.md](report.md) § "LSQ cross-validation".

Run from the repository root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-tau-calibration/lsq_comparison.py

Outputs:

- ``data/lsq_comparison.json``  — headline numbers + per-window table.
- ``figures/12_lsq_histogram.png`` — histogram + Gaussian overlay.
- ``figures/13_lsq_freq_third.png`` — band-third median plot.

Assumes the unapodized fixture and Stages 0-5 have already been run via:

    from ftmwpipeline.core.stage_fit_settings import StageFitSettings
    from ftmwpipeline.core.tau_calibration_settings import TauCalibrationSettings
    ftmw.import_data('scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw',
                     source='examples/blackchirp_data/2638/', force=True)
    ftmw.compute_ft(fpath, trim=(26500, 40000))
    ftmw.estimate_noise(fpath)
    tau_s = TauCalibrationSettings()
    tau_s.band.compute_band_majorities = True
    ftmw.calibrate_tau(fpath, settings=tau_s)   # band majorities for per-band tau routing
    ftmw.detect_peaks(fpath)
    ftmw.assign_windows(fpath)
    fit_s = StageFitSettings()
    fit_s.tau.tau0_us = 6.325   # T_active / 2
    ftmw.fit_peaks(fpath, settings=fit_s)

See ``dev-docs/planning/stage2b-tau-calibration.md`` § Phase 4 for the
gate and motivation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import (
    SpectrumFit,
    WindowDifficulty,
    WindowPlan,
)
from ftmwpipeline.fitting.tau_calibration import extract_tau_majority
from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

FIXTURE = Path("scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw")

# Filter gates. Planning doc § Phase 4 specified "EASY K=1 SNR ≥ 20, σ_τ/τ <
# 0.10, χ²_r < 2"; that gate empties to zero on 2638 (see report). Effective
# gate after the methodological loosening: per-window max free-peak SNR ≥ 10
# AND χ²_r < 3 (still K≥1 with no fixed contributors, free tau fit, well-
# determined σ_τ/τ < 0.10, bounds non-saturating). The looser thresholds
# admit roughly 2× more windows without polluting the sample with weak / poor
# fits.
SNR_GATE = 10.0
TAU_REL_ERR_GATE = 0.10
RCHI2_GATE = 3.0
# Tau-bound non-saturation margins. tau bounds were (tau0/k, tau0*k) with
# tau0 = 6.325 µs and k = DEFAULT_MAX_DECAY_FACTOR = 5 → (1.265, 31.625) µs.
TAU_BOUND_LO = 6.325 / 5.0
TAU_BOUND_HI = 6.325 * 5.0
TAU_LO_SATURATE_FACTOR = 1.05   # tau must be ≥ 1.05 * tau_lo
TAU_HI_SATURATE_FACTOR = 0.95   # tau must be ≤ 0.95 * tau_hi

# Trim band — same as Stage 1 compute_ft trim.
TRIM_LO_MHZ = 26500.0
TRIM_HI_MHZ = 40000.0

# Arithmetic frequency thirds across the trim band (4500 MHz each), replacing
# the asymmetric horn-band split used in the original Phase 2 STFT report
# (26.6-33.6 / 33.6-36.7 / 36.7-39.9 GHz: the "low" third covered ~7 GHz, the
# mid/high ~3 GHz each). Arithmetic thirds give an unbiased frequency-dependence
# read on whichever side has fewer LSQ samples.
ARITHMETIC_THIRDS = [
    ("low",  TRIM_LO_MHZ, TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("mid",  TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0,
             TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("high", TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0, TRIM_HI_MHZ),
]

# STFT majority headlines from report.md § "Polish design" — the two
# polish endpoints (polish=False and polish=True with no SNR cap) used as
# the reference for the per-band bias-flip analysis.
TAU_STFT_POLISH_OFF = 6.328
SIGMA_TAU_STFT_POLISH_OFF = 1.617
TAU_STFT_POLISH_ON = 5.512
SIGMA_TAU_STFT_POLISH_ON = 1.585

logger = logging.getLogger("lsq-comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Per-window record
# ---------------------------------------------------------------------------
@dataclass
class WindowRecord:
    window_id: int
    freq_lo_mhz: float
    freq_hi_mhz: float
    difficulty: str
    n_free_fit: int
    n_fixed_contributors: int
    tau_us: float
    tau_err_us: Optional[float]
    reduced_chi2: float
    max_free_peak_snr: Optional[float]
    saturates_lo: bool
    saturates_hi: bool
    passes_tau_fit_attempted: bool  # fit_tau=True in the LSQ (shared_parameters["tau_us"]["fitted"])
    passes_finite_tau_err: bool     # tau_err is a finite number (cov non-singular)
    passes_no_fixed: bool
    passes_snr: bool
    passes_tau_err_rel: bool        # only meaningful when passes_finite_tau_err
    passes_rchi2: bool
    passes_non_saturating: bool

    @property
    def passes_strict(self) -> bool:
        """Original gate: finite tau uncertainty + all per-window quality cuts."""
        return (
            self.passes_finite_tau_err
            and self.passes_no_fixed
            and self.passes_snr
            and self.passes_tau_err_rel
            and self.passes_rchi2
            and self.passes_non_saturating
        )

    @property
    def passes_expanded(self) -> bool:
        """Loosened gate: tau_fit_attempted (covariance may be singular)
        + all per-window quality cuts EXCEPT the tau-error gate (which is
        meaningless when σ_τ is NaN).
        """
        return (
            self.passes_tau_fit_attempted
            and self.passes_no_fixed
            and self.passes_snr
            and self.passes_rchi2
            and self.passes_non_saturating
        )

    @property
    def is_singular_cov(self) -> bool:
        """tau was fit freely but covariance was singular at the tau slot."""
        return self.passes_tau_fit_attempted and not self.passes_finite_tau_err

    @property
    def center_freq_mhz(self) -> float:
        return 0.5 * (self.freq_lo_mhz + self.freq_hi_mhz)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def collect_records(
    fit: SpectrumFit, plan: WindowPlan,
) -> list[WindowRecord]:
    """Build the per-window filter table from a SpectrumFit + Stage 4 plan.

    Filter spirit: the planning doc's "EASY-difficulty K=1 with free-peak
    SNR ≥ 20, no fixed contributors" reduces to **zero** windows on 2638 (only
    2 EASY K=1 windows clear SNR≥20 and both fail the tau_err gate). The
    practical realization on a dense W-band spectrum is "K≥1 (any difficulty)
    with no fixed contributors, free tau fitted, strong-peak SNR≥10, good fit,
    bounds non-saturating". The "tau actually fit" gate reads
    ``shared_parameters["tau_us"]["fitted"]`` from the persisted fit -- True
    means fit_tau=True ran in the LSQ. It has two flavors: *strict* (finite
    σ_τ) and *expanded* (singular-cov windows also retained; these are
    windows where fit_tau=True ran but the tau slot of J^T J was non-positive
    at the optimum — typically tight blends where the tau column becomes
    degenerate with the amplitude/phase columns).
    """
    plan_by_id = {fw.window_id: fw for fw in plan.windows}
    records: list[WindowRecord] = []
    for result in fit.window_fits:
        wid = result.window_id
        fw = plan_by_id.get(wid) if wid is not None else None
        if fw is None:
            continue
        n_free = len(result.fitted_peaks)
        n_fixed = len(result.fixed_parameters)
        shared = result.shared_parameters.get("tau_us", {})
        tau_us = float(shared.get("value", float("nan")))
        tau_err = shared.get("error")
        tau_err_v: Optional[float] = (
            float(tau_err) if tau_err is not None and np.isfinite(tau_err) else None
        )
        tau_fitted = shared.get("fitted")
        rchi2 = float(result.reduced_chi2)
        snrs = [
            float(p.snr) for p in result.fitted_peaks
            if p.snr is not None and np.isfinite(p.snr)
        ]
        max_snr = max(snrs) if snrs else None

        sat_lo = tau_us < TAU_LO_SATURATE_FACTOR * TAU_BOUND_LO
        sat_hi = tau_us > TAU_HI_SATURATE_FACTOR * TAU_BOUND_HI
        passes_tau_fit_attempted = bool(tau_fitted) if tau_fitted is not None else (
            tau_err_v is not None
        )
        passes_finite_tau_err = tau_err_v is not None
        passes_no_fixed = n_fixed == 0
        passes_snr = (max_snr is not None) and (max_snr >= SNR_GATE)
        passes_tau_err_rel = (
            tau_err_v is not None
            and tau_us > 0
            and (tau_err_v / tau_us) < TAU_REL_ERR_GATE
        )
        passes_rchi2 = np.isfinite(rchi2) and rchi2 < RCHI2_GATE
        passes_non_saturating = not (sat_lo or sat_hi)

        records.append(
            WindowRecord(
                window_id=int(wid),
                freq_lo_mhz=float(fw.freq_range[0]),
                freq_hi_mhz=float(fw.freq_range[1]),
                difficulty=fw.difficulty.value,
                n_free_fit=n_free,
                n_fixed_contributors=n_fixed,
                tau_us=tau_us,
                tau_err_us=tau_err_v,
                reduced_chi2=rchi2,
                max_free_peak_snr=max_snr,
                saturates_lo=sat_lo,
                saturates_hi=sat_hi,
                passes_tau_fit_attempted=passes_tau_fit_attempted,
                passes_finite_tau_err=passes_finite_tau_err,
                passes_no_fixed=passes_no_fixed,
                passes_snr=passes_snr,
                passes_tau_err_rel=passes_tau_err_rel,
                passes_rchi2=passes_rchi2,
                passes_non_saturating=passes_non_saturating,
            )
        )
    return records


# ---------------------------------------------------------------------------
# STFT contributor third-medians (polish=False + polish=True)
# ---------------------------------------------------------------------------
def _extract_stft_contributor_thirds(fixture: Path) -> dict[str, dict[str, dict]]:
    """Re-run the STFT calibration on a fixture twice (polish off / on) and
    return per-arithmetic-third medians of the contributor tau set for each.

    Output structure:

        {"polish_off": {"low": {...}, "mid": {...}, "high": {...}},
         "polish_on":  {...}}

    Each per-third dict carries ``n``, ``median_us``, ``iqr_us``,
    ``tau_maj_us`` (the SNR-weighted majority across the *full* trim band
    for that polish setting), and ``sigma_tau_us``.
    """
    fid = load_fid_from_pipeline_impl(str(fixture))
    sample_dt_us = float(fid.spacing * 1e6)
    sideband = fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    # Recover start/end from the persisted Stage 1 settings via api.compute_ft
    # metadata round-trip would be ideal; the unapodized fixture used
    # start_us = 2.35, end_us = 15.0 by Stage 0 default. Pull from the FT
    # processing_params for robustness.
    ft = ftmw.compute_ft(str(fixture))   # cached canonical settings
    pp = ft.metadata["processing_params"]
    start_us = float(pp.start_us)
    end_us = float(pp.end_us)

    out: dict[str, dict[str, dict]] = {}
    for label, polish in (("polish_off", False), ("polish_on", True)):
        # Pass polish_snr_cap=None explicitly: the production default is
        # cap=9 (a third operating point), but this script's purpose is
        # the polish=OFF vs polish=ON endpoint comparison underlying the
        # bias-flip analysis.
        cal = extract_tau_majority(
            np.asarray(fid.data, dtype=float),
            sample_dt_us,
            start_us=start_us,
            end_us=end_us,
            probe_freq_mhz=float(fid.probe_freq_mhz),
            sideband=sideband,
            trim_lo_mhz=TRIM_LO_MHZ,
            trim_hi_mhz=TRIM_HI_MHZ,
            polish=polish,
            polish_snr_cap=None,
        )
        freqs = np.asarray(cal.contributor_freqs_mhz, dtype=float)
        taus = np.asarray(cal.contributor_taus_us, dtype=float)
        thirds: dict[str, dict] = {}
        for name, lo, hi in ARITHMETIC_THIRDS:
            mask = (freqs >= lo) & (freqs < hi)
            s = taus[mask]
            thirds[name] = {
                "n": int(s.size),
                "median_us": float(np.median(s)) if s.size else float("nan"),
                "iqr_us": (
                    float(np.percentile(s, 75) - np.percentile(s, 25))
                    if s.size else float("nan")
                ),
                "band_lo_mhz": float(lo),
                "band_hi_mhz": float(hi),
            }
        # Also store band-wide stats for context.
        thirds["_band_total"] = {
            "tau_maj_us": float(cal.tau_maj_us),
            "sigma_tau_us": float(cal.sigma_tau_us),
            "n_contributors": int(cal.n_contributors),
        }
        out[label] = thirds
    return out


# ---------------------------------------------------------------------------
# Gaussian fit by MLE (mean, std) — robust for the EASY-K=1 sample
# ---------------------------------------------------------------------------
def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def _gaussian_histfit(
    taus: np.ndarray,
) -> tuple[float, float, float, float]:
    """Gaussian descriptors of the contributor sample.

    Returns ``(mu_mle, sigma_mle, median, iqr_over_1349)`` so the report
    can pick whichever the reader prefers. The histogram is shown with
    the MLE Gaussian overlaid; the headline ``tau_lsq`` is the MLE mean.
    """
    mu = float(np.mean(taus))
    sigma = float(np.std(taus, ddof=1)) if taus.size > 1 else float("nan")
    med = float(np.median(taus))
    q25, q75 = np.percentile(taus, [25, 75])
    iqr_spread = float((q75 - q25) / 1.349)
    return mu, sigma, med, iqr_spread


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_histogram(
    taus_strict: np.ndarray,
    taus_expanded: np.ndarray,
    mu_s: float,
    sigma_s: float,
    median_s: float,
    mu_e: float,
    sigma_e: float,
    median_e: float,
    out: Path,
) -> None:
    """Histogram with the strict population overlaid on the expanded
    population (which includes singular-covariance tau-fit windows)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sample = taus_expanded if taus_expanded.size else taus_strict
    lo = float(sample.min()) - 0.5
    hi = float(sample.max()) + 0.5
    bin_edges = np.linspace(lo, hi, max(12, min(28, sample.size // 3)))
    # Expanded behind, strict in front.
    if taus_expanded.size:
        ax.hist(
            taus_expanded, bins=bin_edges, density=True,
            color="tab:cyan", alpha=0.45, edgecolor="white",
            label=f"LSQ expanded (N={taus_expanded.size})",
        )
    ax.hist(
        taus_strict, bins=bin_edges, density=True,
        color="steelblue", alpha=0.7, edgecolor="white",
        label=f"LSQ strict (N={taus_strict.size})",
    )
    xs = np.linspace(lo, hi, 400)
    if np.isfinite(mu_s) and np.isfinite(sigma_s) and sigma_s > 0:
        ax.plot(xs, _gauss_pdf(xs, mu_s, sigma_s), "-", color="steelblue", lw=1.6,
                label=fr"Strict Gaussian: $\mu={mu_s:.2f}$, $\sigma={sigma_s:.2f}$")
    if np.isfinite(mu_e) and np.isfinite(sigma_e) and sigma_e > 0 and taus_expanded.size > 5:
        ax.plot(xs, _gauss_pdf(xs, mu_e, sigma_e), "-", color="tab:cyan", lw=1.6,
                label=fr"Expanded Gaussian: $\mu={mu_e:.2f}$, $\sigma={sigma_e:.2f}$")
    for label, val, color in (
        ("STFT polish=OFF (6.33)", TAU_STFT_POLISH_OFF, "tab:orange"),
        ("STFT polish=ON (5.51)", TAU_STFT_POLISH_ON, "tab:green"),
    ):
        ax.axvline(val, color=color, ls="--", lw=1.5, label=label)
    ax.axvline(median_s, color="navy", ls=":", lw=1.5,
               label=fr"LSQ strict median={median_s:.2f}")
    if np.isfinite(median_e):
        ax.axvline(median_e, color="teal", ls="-.", lw=1.5,
                   label=fr"LSQ expanded median={median_e:.2f}")
    ax.set_xlabel(r"per-window $\tau$ (µs) — strict: finite $\sigma_\tau$ + $\sigma_\tau/\tau<0.10$; expanded: + singular-cov fits")
    ax.set_ylabel("density")
    ax.set_title(
        f"Phase 4 LSQ histogram on 2638  (strict N={taus_strict.size}, expanded N={taus_expanded.size})"
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_freq_third(
    strict_centers: np.ndarray,
    strict_taus: np.ndarray,
    strict_tau_errs: np.ndarray,
    singular_centers: np.ndarray,
    singular_taus: np.ndarray,
    third_summary: dict,
    out: Path,
) -> None:
    """Frequency-resolved tau scatter with two LSQ subpopulations
    (finite-σ_τ "strict" with error bars, singular-cov "expanded" with no
    error bars in a different color) plus per-third median overlays for
    strict / expanded / STFT(polish=OFF) / STFT(polish=ON) on the same
    arithmetic bands.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.errorbar(
        strict_centers, strict_taus, yerr=strict_tau_errs,
        fmt="o", ms=6, alpha=0.85, color="tab:blue", ecolor="tab:blue",
        elinewidth=1.0, capsize=3, zorder=3,
        label=r"LSQ strict — finite $\sigma_\tau$ ($\pm\sigma_\tau$)",
    )
    if singular_centers.size:
        ax.scatter(
            singular_centers, singular_taus,
            s=36, alpha=0.55, color="tab:cyan", edgecolor="black", linewidth=0.4,
            marker="s", zorder=2,
            label=r"LSQ expanded — singular cov ($\sigma_\tau$ unavailable)",
        )
    band_colors = ("tab:orange", "tab:green", "tab:red")
    for (key, info), color in zip(third_summary.items(), band_colors):
        lo, hi = info["band_lo_mhz"], info["band_hi_mhz"]
        # LSQ medians: strict (solid thick), expanded (solid thin offset down)
        med_s = info["median_lsq_strict_us"]
        med_e = info["median_lsq_expanded_us"]
        n_s = info["n_lsq_strict"]
        n_e = info["n_lsq_expanded"]
        if np.isfinite(med_s):
            ax.hlines(
                med_s, lo, hi, color=color, lw=3.0,
                label=f"LSQ strict {key}={med_s:.2f} (N={n_s})",
            )
        if np.isfinite(med_e):
            ax.hlines(
                med_e, lo, hi, color=color, lw=1.5, alpha=0.55,
                linestyles=(0, (8, 2, 2, 2)),
                label=f"LSQ expanded {key}={med_e:.2f} (N={n_e})",
            )
        po = info["stft_polish_off"]["median_us"]
        pn = info["stft_polish_on"]["median_us"]
        if np.isfinite(po):
            ax.hlines(po, lo, hi, color=color, ls=":", lw=2.0, alpha=0.85,
                      label=f"STFT polish=OFF {key}={po:.2f} (N={info['stft_polish_off']['n']})")
        if np.isfinite(pn):
            ax.hlines(pn, lo, hi, color=color, ls="--", lw=1.4, alpha=0.5,
                      label=f"STFT polish=ON  {key}={pn:.2f} (N={info['stft_polish_on']['n']})")
    ax.set_xlabel("window centre frequency (MHz)")
    ax.set_ylabel(r"$\tau$ (µs)")
    ax.set_title("Phase 4: LSQ vs STFT τ — arithmetic-third comparison on 2638")
    ax.legend(fontsize=7.0, loc="upper right", ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Loading SpectrumFit + WindowPlan from %s", FIXTURE)
    fit = ftmw.load_fit(FIXTURE)
    plan = ftmw.load_windows(FIXTURE)
    logger.info(
        "Loaded %d windows; final_plan_revision=%d, n_fitted_peaks=%d",
        fit.n_windows, fit.final_plan_revision, fit.n_fitted_peaks,
    )

    records = collect_records(fit, plan)
    logger.info("Built %d window records", len(records))

    # Gate funnel — count both the strict (finite σ_τ) and expanded
    # (includes singular-covariance tau-fit windows) paths.
    n_total = len(records)
    n_tau_attempted = sum(r.passes_tau_fit_attempted for r in records)
    n_finite_tau_err = sum(r.passes_finite_tau_err for r in records)
    n_singular = n_tau_attempted - n_finite_tau_err
    strict = [r for r in records if r.passes_strict]
    expanded = [r for r in records if r.passes_expanded]
    singular_only = [r for r in expanded if r.is_singular_cov]

    funnel = {
        "n_total_windows": n_total,
        "n_tau_fit_attempted": n_tau_attempted,
        "n_finite_tau_err": n_finite_tau_err,
        "n_singular_cov": n_singular,
        "n_strict_pass": len(strict),
        "n_expanded_pass": len(expanded),
        "n_singular_pass": len(singular_only),
    }
    logger.info("Gate funnel: %s", funnel)

    if len(strict) < 10:
        logger.warning(
            "Strict sample has %d < 10 windows; the Gaussian fit is unreliable",
            len(strict),
        )

    def _arrs(sample: list[WindowRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not sample:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty
        return (
            np.array([r.tau_us for r in sample], dtype=float),
            np.array([r.tau_err_us if r.tau_err_us is not None else float("nan")
                      for r in sample], dtype=float),
            np.array([r.center_freq_mhz for r in sample], dtype=float),
        )

    taus_strict, tau_errs_strict, centers_strict = _arrs(strict)
    taus_expanded, _, centers_expanded = _arrs(expanded)
    taus_sing, _, centers_sing = _arrs(singular_only)
    rchi2s_strict = np.array([r.reduced_chi2 for r in strict], dtype=float)

    # Headline = strict; expanded reported alongside.
    mu, sigma, median, iqr_spread = _gaussian_histfit(taus_strict)
    if taus_expanded.size > 1:
        mu_e, sigma_e, median_e, iqr_e = _gaussian_histfit(taus_expanded)
    else:
        mu_e = sigma_e = median_e = iqr_e = float("nan")
    logger.info(
        "Strict LSQ:    mu=%.3f sigma=%.3f median=%.3f IQR/1.349=%.3f N=%d",
        mu, sigma, median, iqr_spread, taus_strict.size,
    )
    logger.info(
        "Expanded LSQ:  mu=%.3f sigma=%.3f median=%.3f IQR/1.349=%.3f N=%d "
        "(of which singular-cov: N=%d)",
        mu_e, sigma_e, median_e, iqr_e, taus_expanded.size, taus_sing.size,
    )

    # Keep legacy variable names for compatibility with downstream blocks.
    taus = taus_strict
    tau_errs = tau_errs_strict
    centers = centers_strict
    rchi2s = rchi2s_strict

    # Arithmetic-third analysis. Compute LSQ medians on the passing windows
    # AND STFT medians on the freshly-extracted contributor data (both polish
    # settings) so the comparison shares the same band edges. A horn-band
    # split (low 7 GHz, mid+high 3.1 GHz each) would be asymmetric and
    # apples-to-oranges against an arithmetic-third LSQ.
    logger.info("Running Stage 2b STFT extraction (polish=False and polish=True) "
                "for STFT-vs-LSQ arithmetic-third comparison")
    stft_thirds_by_polish = _extract_stft_contributor_thirds(FIXTURE)

    third_summary: dict[str, dict] = {}
    for name, lo, hi in ARITHMETIC_THIRDS:
        m_strict = (centers_strict >= lo) & (centers_strict < hi)
        m_expanded = (centers_expanded >= lo) & (centers_expanded < hi)
        s_strict = taus_strict[m_strict]
        s_expanded = taus_expanded[m_expanded]
        third_summary[name] = {
            "band_lo_mhz": float(lo),
            "band_hi_mhz": float(hi),
            "n_lsq_strict": int(s_strict.size),
            "median_lsq_strict_us": float(np.median(s_strict)) if s_strict.size else float("nan"),
            "iqr_lsq_strict_us": (
                float(np.percentile(s_strict, 75) - np.percentile(s_strict, 25))
                if s_strict.size else float("nan")
            ),
            "n_lsq_expanded": int(s_expanded.size),
            "median_lsq_expanded_us": float(np.median(s_expanded)) if s_expanded.size else float("nan"),
            "iqr_lsq_expanded_us": (
                float(np.percentile(s_expanded, 75) - np.percentile(s_expanded, 25))
                if s_expanded.size else float("nan")
            ),
            # Back-compat with downstream consumers that read median_lsq_us.
            "median_lsq_us": float(np.median(s_strict)) if s_strict.size else float("nan"),
            "n_lsq": int(s_strict.size),
            "stft_polish_off": stft_thirds_by_polish["polish_off"][name],
            "stft_polish_on": stft_thirds_by_polish["polish_on"][name],
        }

    # Comparison against the STFT headlines.
    def compare(label: str, tau_stft: float, sigma_stft: float) -> dict:
        rel = abs(mu - tau_stft) / tau_stft
        return {
            "label": label,
            "tau_stft_us": tau_stft,
            "sigma_tau_stft_us": sigma_stft,
            "tau_lsq_us": mu,
            "sigma_tau_lsq_us": sigma,
            "abs_diff_us": abs(mu - tau_stft),
            "rel_diff": rel,
            "passes_10pct_gate": rel < 0.10,
        }

    headline = {
        "fixture": str(FIXTURE),
        "tau0_us_used": float(fit.parameters.get("tau0_us", float("nan"))),
        "tau_bounds_us": (TAU_BOUND_LO, TAU_BOUND_HI),
        "strict": {
            "n_passed_windows": int(taus_strict.size),
            "tau_lsq_mu_us": float(mu),
            "tau_lsq_sigma_us": float(sigma),
            "tau_lsq_median_us": float(median),
            "tau_lsq_iqr_over_1349_us": float(iqr_spread),
            "median_reduced_chi2": (
                float(np.median(rchi2s_strict)) if rchi2s_strict.size else float("nan")
            ),
        },
        "expanded": {
            "n_passed_windows": int(taus_expanded.size),
            "n_singular_cov": int(taus_sing.size),
            "tau_lsq_mu_us": float(mu_e),
            "tau_lsq_sigma_us": float(sigma_e),
            "tau_lsq_median_us": float(median_e),
            "tau_lsq_iqr_over_1349_us": float(iqr_e),
        },
        "stft_polish_off": compare(
            "STFT polish=False", TAU_STFT_POLISH_OFF, SIGMA_TAU_STFT_POLISH_OFF
        ),
        "stft_polish_on": compare(
            "STFT polish=True", TAU_STFT_POLISH_ON, SIGMA_TAU_STFT_POLISH_ON
        ),
        "funnel": funnel,
        "freq_thirds": third_summary,
        "passed_window_records_strict": [
            {
                "window_id": r.window_id,
                "freq_lo_mhz": r.freq_lo_mhz,
                "freq_hi_mhz": r.freq_hi_mhz,
                "tau_us": r.tau_us,
                "tau_err_us": r.tau_err_us,
                "reduced_chi2": r.reduced_chi2,
                "max_free_peak_snr": r.max_free_peak_snr,
            }
            for r in strict
        ],
        "passed_window_records_singular_only": [
            {
                "window_id": r.window_id,
                "freq_lo_mhz": r.freq_lo_mhz,
                "freq_hi_mhz": r.freq_hi_mhz,
                "tau_us": r.tau_us,
                "reduced_chi2": r.reduced_chi2,
                "max_free_peak_snr": r.max_free_peak_snr,
                "n_free_fit": r.n_free_fit,
            }
            for r in singular_only
        ],
    }

    out_json = DATA / "lsq_comparison.json"
    out_json.write_text(json.dumps(headline, indent=2))
    logger.info("Headline JSON written to %s", out_json)

    plot_histogram(
        taus_strict, taus_expanded,
        mu, sigma, median,
        mu_e, sigma_e, median_e,
        FIG / "12_lsq_histogram.png",
    )
    plot_freq_third(
        centers_strict, taus_strict, tau_errs_strict,
        centers_sing, taus_sing,
        third_summary,
        FIG / "13_lsq_freq_third.png",
    )
    logger.info("Figures written to %s", FIG)

    # Final stdout banner so the operator sees the headline call.
    print()
    print("====== Phase 4 LSQ comparison headline ======")
    print(f"  Strict:   N={taus_strict.size}  mean={mu:.3f}+/-{sigma:.3f}  median={median:.3f}  IQR/1.349={iqr_spread:.3f}")
    print(f"  Expanded: N={taus_expanded.size}  mean={mu_e:.3f}+/-{sigma_e:.3f}  median={median_e:.3f}  IQR/1.349={iqr_e:.3f}"
          f"  (+{taus_sing.size} singular-cov)")
    print(f"  STFT polish=False (6.328): strict-mean {abs(mu-TAU_STFT_POLISH_OFF)/TAU_STFT_POLISH_OFF*100:+.1f}%   strict-median {(median-TAU_STFT_POLISH_OFF)/TAU_STFT_POLISH_OFF*100:+.1f}%   expanded-median {(median_e-TAU_STFT_POLISH_OFF)/TAU_STFT_POLISH_OFF*100:+.1f}%")
    print(f"  STFT polish=True  (5.512): strict-mean {abs(mu-TAU_STFT_POLISH_ON)/TAU_STFT_POLISH_ON*100:+.1f}%   strict-median {(median-TAU_STFT_POLISH_ON)/TAU_STFT_POLISH_ON*100:+.1f}%   expanded-median {(median_e-TAU_STFT_POLISH_ON)/TAU_STFT_POLISH_ON*100:+.1f}%")
    print()
    print("  Arithmetic-third medians (band edges {:.1f}-{:.1f}-{:.1f}-{:.1f} GHz):".format(
        TRIM_LO_MHZ / 1e3,
        (TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0) / 1e3,
        (TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0) / 1e3,
        TRIM_HI_MHZ / 1e3,
    ))
    print(f"    {'third':>5} {'LSQ strict':>11} {'(N)':>5} {'LSQ expanded':>13} {'(N)':>5} {'STFT off':>9} {'(N)':>6} {'STFT on':>9} {'(N)':>6}")
    for name in ("low", "mid", "high"):
        info = third_summary[name]
        po = info["stft_polish_off"]; pn = info["stft_polish_on"]
        print(
            f"    {name:>5} {info['median_lsq_strict_us']:>11.2f} "
            f"{info['n_lsq_strict']:>5} {info['median_lsq_expanded_us']:>13.2f} "
            f"{info['n_lsq_expanded']:>5} {po['median_us']:>9.2f} {po['n']:>6} "
            f"{pn['median_us']:>9.2f} {pn['n']:>6}"
        )
    print("=============================================")


if __name__ == "__main__":
    main()
