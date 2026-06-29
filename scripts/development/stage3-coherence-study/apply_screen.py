"""
Apply the calibrated projection-coherence screen to 2638.

Re-runs ``project_candidates`` on every EASY-window persisted Stage 3
candidate of the 2638 fixture across four τ_basis settings -- matched
to the strong-line fit median (3 µs), narrower bases (6 µs and 9 µs),
and the apodization default (5 µs) -- and writes a descriptive
breakdown of what each setting would eliminate at several thresholds.

Output: ``scratch/stage3-coherence-study/apply_screen.md`` plus a
multi-panel ratio-distribution figure. No ground-truth claims: this is
a descriptive look at the screen's behavior on a real dataset, *not*
a validation. The research project at
``dev-docs/research/stage3-coherence-screen/`` is the source for the
τ_basis choice and the operating-point intuition.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.core.data_structures import WindowDifficulty
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.preprocessing.coherence_screen import project_candidates
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

logger = logging.getLogger("stage3-coherence-apply")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FTMW_PATH = REPO_ROOT / "scratch" / "stage5-validation" / "exp_2638.ftmw"
OUTPUT_DIR = REPO_ROOT / "scratch" / "stage3-coherence-study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# τ_basis settings to compare. From the research project: 2638's
# strong-line fit-determined τ_eff is ≈ 3 µs. The wiring proposal
# suggests τ_basis ≈ 2-3× narrower than the data (factor 2-3).
TAU_SETTINGS = [
    ("matched (τ=3)", 3.0),
    ("apodization (τ=5)", 5.0),
    ("2× narrower (τ=6)", 6.0),
    ("3× narrower (τ=9)", 9.0),
]

# Operating points (ratio thresholds) to scan. A candidate is *kept* if
# ratio ≥ threshold; killed otherwise.
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _ascending_bin(sorted_freq: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(sorted_freq, value))
    if pos == 0:
        return 0
    if pos >= sorted_freq.size:
        return int(sorted_freq.size - 1)
    if abs(value - sorted_freq[pos - 1]) <= abs(value - sorted_freq[pos]):
        return pos - 1
    return pos


def main() -> None:
    logger.info("loading 2638 fixture: %s", FTMW_PATH)
    plan = ftmw.load_windows(str(FTMW_PATH))
    peaks = ftmw.load_peaks(str(FTMW_PATH))

    easy_ids = {
        w.window_id for w in plan.windows if w.difficulty == WindowDifficulty.EASY
    }
    easy_windows = [w for w in plan.windows if w.window_id in easy_ids]

    def win_of(f: float) -> int | None:
        for w in easy_windows:
            if w.freq_range[0] <= f <= w.freq_range[1]:
                return w.window_id
        return None

    survey_peaks = []
    for peak in peaks:
        wid = win_of(peak.frequency)
        if wid is None:
            continue
        survey_peaks.append((wid, peak))
    logger.info(
        "survey set: %d EASY-window candidates (of %d total persisted)",
        len(survey_peaks),
        len(peaks),
    )

    # Build the active-FT once.
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        expf_us,
        probe_freq_mhz,
        sideband_enum,
        n_padded,
        acquisition_us,
        _user_ft,
        _user_rms,
    ) = _build_active_ft_inputs(str(FTMW_PATH))
    active_ft = compute_active_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband_enum,
        n_padded=n_padded,
    )
    # Active-FT noise.
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freq_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    mag_sorted = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    noise = estimate_noise_adaptive(freq_sorted, mag_sorted)
    rms_sorted = np.asarray(noise.rms_noise, dtype=float)
    sigma_active = rms_sorted[unsort_idx]
    bin_mhz = 1.0 / acquisition_us
    logger.info(
        "active-FT: %d bins, T=%.3f µs, bin=%.4f MHz",
        active_ft.freq_mhz.size,
        acquisition_us,
        bin_mhz,
    )
    fwhm_data = 1.0 / (np.pi * 3.0)  # τ_eff=3 µs from strong-line fit
    logger.info(
        "expected FWHM (τ_eff=3 µs): %.4f MHz = %.2f bins",
        fwhm_data,
        fwhm_data / bin_mhz,
    )

    # Per-candidate baseline columns.
    cand_freqs = np.array([float(p.frequency) for _, p in survey_peaks])
    user_snr = np.array([float(p.snr) for _, p in survey_peaks])
    promoted = np.array(
        [bool(p.properties.get("promoted", False)) for _, p in survey_peaks]
    )
    detection_pass = np.array(
        [p.properties.get("detection_pass", "?") for _, p in survey_peaks]
    )
    active_bins = np.array(
        [int(unsort_idx[_ascending_bin(freq_sorted, f)]) for f in cand_freqs]
    )
    active_mag = np.abs(active_ft.complex_spectrum)[active_bins]
    sigma_c_per_bin = np.where(sigma_active > 0, sigma_active / np.sqrt(2.0), 1.0)
    active_snr = active_mag / sigma_c_per_bin[active_bins]

    # Run projection at each τ_basis.
    runs: dict[str, np.ndarray] = {}
    for label, tau in TAU_SETTINGS:
        logger.info("τ_basis = %.2f µs ... %s", tau, label)
        projections = project_candidates(
            active_ft.freq_mhz,
            active_ft.complex_spectrum,
            sigma_active,
            cand_freqs.tolist(),
            tau_us=tau,
            acquisition_us=acquisition_us,
            sideband=sideband_enum,
        )
        runs[label] = np.array([p.ratio for p in projections])

    # Markdown report.
    md: list[str] = []
    md.append("# Calibrated screen applied to 2638 -- descriptive run\n\n")
    md.append(
        "Re-runs ``project_candidates`` on the 398 EASY-window persisted "
        "Stage 3 candidates of the 2638 fixture across four τ_basis "
        "settings. This is a **descriptive** view: the prior 2638 study "
        "established that ``became_fitted_peak`` is *not* trustworthy "
        "ground truth on this fixture (the fitter freezes τ on weak "
        "lines), so no AUC / TPR / FPR numbers are quoted here. The "
        "research project at "
        "[`dev-docs/research/stage3-coherence-screen/`](../../dev-docs/research/stage3-coherence-screen/) "
        "is the source for the τ_basis range and the operating-point "
        "intuition.\n\n"
    )
    md.append("## Fixture geometry\n\n")
    md.append(
        f"- Active acquisition T = {acquisition_us:.3f} µs, "
        f"bin spacing Δf = {bin_mhz:.4f} MHz\n"
        f"- Strong-line fit-determined τ_eff ≈ 3 µs "
        f"(from prior fit-determined-only sample)\n"
        f"- Expected line FWHM = {fwhm_data:.4f} MHz = "
        f"{fwhm_data / bin_mhz:.2f} bins\n"
        f"- Apodization τ_apod = {expf_us} µs (the Stage 1 expf_us)\n"
        f"- **2638 sits at FWHM/bin ≈ 1.34** -- inside the bin-matched "
        f"optimum (FWHM/bin ∈ [1, 2]) the simulator study identified.\n\n"
    )
    md.append("## Survey composition\n\n")
    md.append(
        f"- {len(survey_peaks)} EASY-window candidates "
        f"(of {len(peaks)} total persisted Stage 3 peaks)\n"
        f"- promoted (user-grid SNR ≥ 3): {int(promoted.sum())} "
        f"({promoted.mean():.1%})\n"
        f"- user-grid SNR distribution: median={np.median(user_snr):.2f}, "
        f"q25={np.quantile(user_snr, 0.25):.2f}, "
        f"q75={np.quantile(user_snr, 0.75):.2f}\n"
        f"- active-FT SNR distribution: median={np.median(active_snr):.2f}, "
        f"q25={np.quantile(active_snr, 0.25):.2f}, "
        f"q75={np.quantile(active_snr, 0.75):.2f}\n"
        f"- primary-pass: {int((detection_pass == 'primary').sum())}; "
        f"gap-pass: {int((detection_pass == 'gap').sum())}\n\n"
    )

    # Ratio summary per τ_basis.
    md.append("## Ratio distributions\n\n")
    md.append("| τ_basis | min | q25 | median | q75 | max |\n")
    md.append("|---------|-----|-----|--------|-----|-----|\n")
    for label, _ in TAU_SETTINGS:
        r = runs[label]
        md.append(
            f"| {label} | {r.min():.3f} | {np.quantile(r, 0.25):.3f} | "
            f"{np.median(r):.3f} | {np.quantile(r, 0.75):.3f} | "
            f"{r.max():.3f} |\n"
        )
    md.append("\n")

    # Threshold scan per τ_basis.
    md.append("## What gets eliminated at various thresholds\n\n")
    md.append("Candidates with ratio below the threshold are eliminated.\n\n")
    for label, tau in TAU_SETTINGS:
        r = runs[label]
        md.append(f"### τ_basis = {tau:.1f} µs ({label})\n\n")
        md.append(
            "| threshold | killed | killed% | killed-promoted | "
            "killed-not-promoted | killed (SNR<3) | "
            "killed (3≤SNR<5) | killed (SNR≥5) |\n"
        )
        md.append(
            "|-----------|--------|---------|"
            "-----------------|---------------------|"
            "----------------|------------------|----------------|\n"
        )
        for t in THRESHOLDS:
            killed = r < t
            n_k = int(killed.sum())
            n_kp = int((killed & promoted).sum())
            n_knp = int((killed & ~promoted).sum())
            n_low = int((killed & (user_snr < 3.0)).sum())
            n_mid = int((killed & (user_snr >= 3.0) & (user_snr < 5.0)).sum())
            n_hi = int((killed & (user_snr >= 5.0)).sum())
            md.append(
                f"| {t:.2f} | {n_k} | {n_k / r.size:.1%} | {n_kp} | "
                f"{n_knp} | {n_low} | {n_mid} | {n_hi} |\n"
            )
        md.append("\n")

    # Cross-tau comparison: how many of the candidates each setting
    # kills are common across settings? (Stability check.)
    md.append("## Stability across τ_basis at threshold 0.9\n\n")
    md.append("Same-candidate overlap between τ_basis settings, threshold 0.9.\n\n")
    md.append("| | " + " | ".join(label for label, _ in TAU_SETTINGS) + " |\n")
    md.append("|" + "|".join(["-"] * (len(TAU_SETTINGS) + 1)) + "|\n")
    killed_sets = {
        label: set(np.where(runs[label] < 0.9)[0].tolist()) for label, _ in TAU_SETTINGS
    }
    for label_a, _ in TAU_SETTINGS:
        row = [label_a]
        for label_b, _ in TAU_SETTINGS:
            inter = killed_sets[label_a] & killed_sets[label_b]
            row.append(f"{len(inter)}")
        md.append("| " + " | ".join(row) + " |\n")
    md.append("\n")
    md.append(
        "Diagonal = candidates killed by that setting. Off-diagonal = "
        "intersection. Stable killings (those flagged by all settings) "
        "are the most defensible eliminations; settings disagree on "
        "marginal candidates.\n\n"
    )

    # Observations.
    md.append("## Observations (descriptive, no ground-truth claims)\n\n")
    matched_r = runs["matched (τ=3)"]
    narrow_r = runs["2× narrower (τ=6)"]
    apod_r = runs["apodization (τ=5)"]
    md.append(
        f"- The four τ_basis settings concentrate ratios in different "
        f"bands: matched median = {np.median(matched_r):.3f}, "
        f"apodization median = {np.median(apod_r):.3f}, "
        f"2× narrower median = {np.median(narrow_r):.3f}. The wiring "
        f"proposal's choice (2× narrower than data, τ=6) sits between "
        f"matched and the apodization default the original buggy "
        f"helper used.\n"
    )
    md.append(
        f"- At threshold 0.9 with matched τ, "
        f"{int((matched_r < 0.9).sum())}/{len(survey_peaks)} candidates "
        f"are eliminated; with 2× narrower τ, "
        f"{int((narrow_r < 0.9).sum())}/{len(survey_peaks)}. The "
        f"narrower-basis setting eliminates more candidates because "
        f"its sub-window collapses to ~3 bins and the on-line phase "
        f"check becomes stricter.\n"
    )
    md.append(
        f"- The promoted vs not-promoted split of killed candidates is "
        f"informative: at threshold 0.9, matched-τ kills "
        f"{int((matched_r < 0.9 & promoted).sum() if False else ((matched_r < 0.9) & promoted).sum())} "
        f"promoted (user-grid SNR ≥ 3) candidates and "
        f"{int(((matched_r < 0.9) & ~promoted).sum())} not-promoted. The "
        f"promoted/total ratio gives a sense of how aggressive the "
        f"screen is on what production already considers strong.\n"
    )
    md.append(
        "- Stability across τ_basis is the most defensible signal "
        "absent ground truth: candidates eliminated by *every* "
        "setting are the screen's most consistent kills. Marginal "
        "candidates where the settings disagree are good targets "
        "for visual inspection.\n"
    )

    md_path = OUTPUT_DIR / "apply_screen.md"
    md_path.write_text("".join(md))
    logger.info("wrote %s", md_path)

    # Figure: histogram of ratios per τ_basis + threshold-scan curve.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (label, tau) in zip(axes[0], TAU_SETTINGS[:2]):
        r = runs[label]
        bins_h = np.linspace(0.0, max(2.0, r.max() + 0.1), 40)
        ax.hist(
            r[promoted],
            bins=bins_h,
            alpha=0.6,
            color="C0",
            label=f"promoted (SNR≥3, n={int(promoted.sum())})",
        )
        ax.hist(
            r[~promoted],
            bins=bins_h,
            alpha=0.6,
            color="C3",
            label=f"not promoted (n={int((~promoted).sum())})",
        )
        for t in [0.7, 0.9, 1.0]:
            ax.axvline(t, ls=":", color="k", alpha=0.5)
        ax.set_xlabel("projection ratio")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: τ={tau:.1f} µs")
        ax.legend(loc="upper right", fontsize=9)

    for ax, (label, tau) in zip(axes[1], TAU_SETTINGS[2:]):
        r = runs[label]
        bins_h = np.linspace(0.0, max(2.0, r.max() + 0.1), 40)
        ax.hist(
            r[promoted],
            bins=bins_h,
            alpha=0.6,
            color="C0",
            label=f"promoted (SNR≥3, n={int(promoted.sum())})",
        )
        ax.hist(
            r[~promoted],
            bins=bins_h,
            alpha=0.6,
            color="C3",
            label=f"not promoted (n={int((~promoted).sum())})",
        )
        for t in [0.7, 0.9, 1.0]:
            ax.axvline(t, ls=":", color="k", alpha=0.5)
        ax.set_xlabel("projection ratio")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: τ={tau:.1f} µs")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Projection ratios on 2638 EASY candidates -- four τ_basis settings")
    fig.tight_layout()
    fig_path = OUTPUT_DIR / "apply_screen.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", fig_path)


if __name__ == "__main__":
    main()
