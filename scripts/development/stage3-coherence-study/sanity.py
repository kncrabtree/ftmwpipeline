"""
Stage 3 projection-coherence study -- sanity checks.

Three controls for the projection helper, run on the 2638 fixture:

1. Strong-peak control. Every candidate at active-FT SNR >= 10 should
   project to ratio close to 1 (the basis matches the data exactly). Pulls
   the rows from ``candidates.csv``; no recomputation.

2. Pure-noise control. Pick a set of frequencies inside the trimmed
   spectrum but at least 5 FWHM from any persisted Stage 3 candidate (so
   the bin is genuinely "between lines"). Run the projection there;
   expect the ratios to sit well below 1.

3. Padded-vs-active cross-check. Project five strong, isolated peaks
   against (a) the active-FT and (b) the persisted user FT. Report the
   per-candidate ratio differences. If the two are indistinguishable,
   the "active-FT only" requirement in the module can relax to a doc
   note.

Outputs ``scratch/stage3-coherence-study/sanity.md`` plus a small CSV per
control.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.preprocessing.coherence_screen import project_candidates
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

logger = logging.getLogger("stage3-coherence-sanity")


DEFAULT_FTMW_PATH = REPO_ROOT / "scratch" / "stage5-validation" / "exp_2638.ftmw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "stage3-coherence-study"
DEFAULT_CANDIDATES_CSV = DEFAULT_OUTPUT_DIR / "candidates.csv"


def _ascending_bin(sorted_freq: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(sorted_freq, value))
    if pos == 0:
        return 0
    if pos >= sorted_freq.size:
        return int(sorted_freq.size - 1)
    if abs(value - sorted_freq[pos - 1]) <= abs(value - sorted_freq[pos]):
        return pos - 1
    return pos


def strong_peak_control(candidates_csv: Path, output_md: list[str]) -> None:
    rows = list(csv.DictReader(candidates_csv.open()))
    output_md.append("## 1. Strong-peak control\n")
    output_md.append(
        "Every candidate with active-FT SNR >= 10 should project to ratio "
        "close to 1 (the basis matches the data; the projection recovers "
        "the amplitude). A failure here means the basis tau / sub-window "
        "choice is wrong.\n"
    )
    strong = [r for r in rows if float(r["active_snr"]) >= 10.0]
    ratios = np.array([float(r["ratio"]) for r in strong])
    if ratios.size == 0:
        output_md.append("- no strong candidates in survey\n")
        return
    output_md.append(
        f"- n={ratios.size}, ratio min={ratios.min():.3f} max={ratios.max():.3f} "
        f"median={float(np.median(ratios)):.3f}\n"
    )
    n_below_0p8 = int((ratios < 0.8).sum())
    output_md.append(f"- candidates with ratio < 0.8: {n_below_0p8}/{ratios.size}\n")
    if n_below_0p8 == 0:
        output_md.append("- **PASS**: no strong-peak failures.\n\n")
    else:
        output_md.append(
            "- **FAIL**: strong peaks projecting below 0.8 indicate a "
            "basis/data mismatch; inspect those rows.\n\n"
        )


def pure_noise_control(
    ftmw_path: Path,
    candidates_csv: Path,
    output_md: list[str],
    *,
    n_samples: int = 200,
    seed: int = 12345,
) -> None:
    output_md.append("## 2. Pure-noise control\n")
    output_md.append(
        f"Sample {n_samples} frequencies inside the trimmed spectrum at "
        "least 5 FWHM from any persisted Stage 3 candidate. Apply the "
        "projection. Expect the ratio distribution to sit below 1 (no "
        "coherent line under the basis).\n"
    )

    peaks_list = ftmw.load_peaks(str(ftmw_path))
    peak_freqs = np.sort(np.array([float(p.frequency) for p in peaks_list]))

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
    ) = _build_active_ft_inputs(str(ftmw_path))
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
    tau_us = float(expf_us) if expf_us is not None else float(acquisition_us / 3.0)
    fwhm_mhz = 1.0 / (np.pi * tau_us)
    exclusion_mhz = 5.0 * fwhm_mhz

    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freq_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    mag_sorted = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    noise = estimate_noise_adaptive(freq_sorted, mag_sorted)
    rms_sorted = np.asarray(noise.rms_noise, dtype=float)
    sigma_active = rms_sorted[unsort_idx]

    # Sample candidates: random ascending-grid positions, reject those
    # within ``exclusion_mhz`` of any persisted peak. Bound to the active-FT
    # frequency range.
    rng = np.random.default_rng(seed)
    f_lo = float(freq_sorted[10])
    f_hi = float(freq_sorted[-10])
    sampled: list[float] = []
    attempts = 0
    while len(sampled) < n_samples and attempts < 50 * n_samples:
        attempts += 1
        f = float(rng.uniform(f_lo, f_hi))
        # nearest peak
        pos = int(np.searchsorted(peak_freqs, f))
        candidates = []
        if pos > 0:
            candidates.append(abs(f - peak_freqs[pos - 1]))
        if pos < peak_freqs.size:
            candidates.append(abs(f - peak_freqs[pos]))
        if candidates and min(candidates) < exclusion_mhz:
            continue
        sampled.append(f)
    output_md.append(
        f"- exclusion radius: 5*FWHM = {exclusion_mhz:.3f} MHz; "
        f"sampled {len(sampled)} of {n_samples} (rejected {attempts - len(sampled)})\n"
    )

    projections = project_candidates(
        active_ft.freq_mhz,
        active_ft.complex_spectrum,
        sigma_active,
        sampled,
        tau_us=tau_us,
        acquisition_us=acquisition_us,
        sideband=sideband_enum,
    )
    ratios = np.array([p.ratio for p in projections])
    active_snrs = np.array([p.detected_snr_active for p in projections])
    output_md.append(
        f"- ratio: min={ratios.min():.3f} max={ratios.max():.3f} "
        f"median={float(np.median(ratios)):.3f} mean={float(ratios.mean()):.3f}\n"
    )
    output_md.append(
        f"- active-FT SNR: median={float(np.median(active_snrs)):.3f} "
        f"max={float(active_snrs.max()):.3f}\n"
    )
    n_below_0p9 = int((ratios < 0.9).sum())
    output_md.append(
        f"- ratios below 0.9: {n_below_0p9}/{ratios.size} ({n_below_0p9/ratios.size:.2%})\n"
    )

    # Dump rows for follow-up analysis.
    out_csv = DEFAULT_OUTPUT_DIR / "pure_noise_samples.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "candidate_freq_mhz",
                "active_bin",
                "coherent_amp",
                "coherent_snr",
                "detected_snr_active",
                "ratio",
            ]
        )
        for p in projections:
            w.writerow(
                [
                    f"{p.candidate_freq_mhz:.6f}",
                    p.active_bin,
                    f"{p.coherent_amp:.6e}",
                    f"{p.coherent_snr:.6f}",
                    f"{p.detected_snr_active:.6f}",
                    f"{p.ratio:.6f}",
                ]
            )
    output_md.append(f"- per-sample CSV: `{out_csv.name}`\n\n")


def padded_vs_active(
    ftmw_path: Path,
    candidates_csv: Path,
    output_md: list[str],
    *,
    n_peaks: int = 5,
) -> None:
    output_md.append("## 3. Padded user-FT vs active-FT cross-check\n")
    output_md.append(
        "Project five strong, isolated peaks against both the active-FT "
        "and the persisted user FT (same h_T basis, same sigma-weighting; "
        "just swap the spectrum and the frequency grid). The ratios "
        "should differ systematically -- if not, the active-FT-only "
        "requirement in the helper can relax to a doc note.\n"
    )
    rows = list(csv.DictReader(candidates_csv.open()))
    strong = [
        r for r in rows if float(r["active_snr"]) >= 20.0 and r["close_pair"] != "True"
    ]
    strong.sort(key=lambda r: float(r["active_snr"]), reverse=True)
    strong = strong[:n_peaks]
    if not strong:
        output_md.append("- no strong isolated candidates available\n\n")
        return

    # Active-FT (same as the survey).
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
        user_ft,
        user_rms,
    ) = _build_active_ft_inputs(str(ftmw_path))
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
    tau_us = float(expf_us) if expf_us is not None else float(acquisition_us / 3.0)

    # Active-FT noise.
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freq_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    mag_sorted = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    noise_a = estimate_noise_adaptive(freq_sorted, mag_sorted)
    sigma_active = noise_a.rms_noise[unsort_idx]

    # User-FT noise: estimate on the (trimmed) user spectrum.
    user_freq = user_ft.freq_array
    user_mag = user_ft.magnitude_spectrum
    # estimate_noise_adaptive needs ascending freqs.
    user_sort = np.argsort(user_freq)
    user_unsort = np.argsort(user_sort)
    noise_u = estimate_noise_adaptive(
        np.ascontiguousarray(user_freq[user_sort]),
        np.ascontiguousarray(user_mag[user_sort]),
    )
    sigma_user = np.asarray(noise_u.rms_noise, dtype=float)[user_unsort]

    cand_freqs = [float(r["frequency_mhz"]) for r in strong]
    proj_active = project_candidates(
        active_ft.freq_mhz,
        active_ft.complex_spectrum,
        sigma_active,
        cand_freqs,
        tau_us=tau_us,
        acquisition_us=acquisition_us,
        sideband=sideband_enum,
    )
    proj_user = project_candidates(
        np.asarray(user_freq, dtype=float),
        np.asarray(user_ft.complex_spectrum, dtype=complex),
        sigma_user,
        cand_freqs,
        tau_us=tau_us,
        acquisition_us=acquisition_us,
        sideband=sideband_enum,
    )

    output_md.append(
        "| freq_mhz | active_snr | ratio(active) | ratio(user FT) | " "Δratio |\n"
    )
    output_md.append(
        "|---------|-----------|--------------|---------------|--------|\n"
    )
    diffs: list[float] = []
    for r, pa, pu in zip(strong, proj_active, proj_user):
        d = pa.ratio - pu.ratio
        diffs.append(d)
        output_md.append(
            f"| {float(r['frequency_mhz']):.4f} | "
            f"{float(r['active_snr']):.2f} | {pa.ratio:.4f} | "
            f"{pu.ratio:.4f} | {d:+.4f} |\n"
        )
    diffs_arr = np.array(diffs)
    output_md.append(
        f"\nΔratio: mean={diffs_arr.mean():+.4f} "
        f"absmax={float(np.max(np.abs(diffs_arr))):.4f}\n"
    )
    if float(np.max(np.abs(diffs_arr))) < 0.05:
        output_md.append(
            "\n**Verdict**: padded vs active ratios differ by < 5%. The "
            "active-FT-only requirement could relax to a documentation "
            "note.\n\n"
        )
    else:
        output_md.append(
            "\n**Verdict**: padded vs active ratios differ by more than "
            "5%. The active-FT requirement is load-bearing.\n\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftmw", type=Path, default=DEFAULT_FTMW_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output: list[str] = ["# Stage 3 coherence screen -- sanity checks\n\n"]
    strong_peak_control(args.candidates, output)
    pure_noise_control(args.ftmw, args.candidates, output)
    padded_vs_active(args.ftmw, args.candidates, output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "sanity.md"
    md_path.write_text("".join(output))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
