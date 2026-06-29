"""
Stage 3 projection-coherence study -- candidate survey.

Loads the 2638 fixture, builds the active-portion FT, scopes every persisted
Stage 3 candidate inside an EASY window, projects each candidate onto the
finite-T Lorentzian basis from
:mod:`ftmwpipeline.preprocessing.coherence_screen`, and matches each one to
the persisted Stage 5 consolidated fit (`±1` active-FT bin) to get a
ground-truth proxy for "the candidate became a fitted peak".

The fixture's production Stage 3 already detects down to
``DEFAULT_INTERNAL_MIN_SNR=2.0`` on the internal grid -- so the "lowered
floor" requirement in the study prompt is already satisfied by the persisted
peak list; no re-detection or .ftmw copy is needed. The user-grid SNR at
which each peak ends up after snap-back varies (some bins re-measure below 2),
so this script optionally subsets by ``snr >= --min_snr`` if the analyst
wants to restrict the study to a stricter band.

Outputs ``scratch/stage3-coherence-study/candidates.csv`` with one row per
EASY-window candidate: window id, frequencies / magnitudes / SNRs on both
grids, projection-ratio columns from the helper, the ground-truth
``became_fitted_peak`` flag, and the ``close_pair`` flag (any other survey
candidate within ``close_pair_bins`` active-FT bins). Run from anywhere; all
paths are absolute or derived from this file's location.
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
from ftmwpipeline.core.data_structures import (
    FitWindow,
    SpectrumFit,
    WindowDifficulty,
)
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.preprocessing.coherence_screen import project_candidates
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

logger = logging.getLogger("stage3-coherence-survey")


DEFAULT_FTMW_PATH = REPO_ROOT / "scratch" / "stage5-validation" / "exp_2638.ftmw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "stage3-coherence-study"


def _ascending_bin(freq_sorted_ascending: np.ndarray, value: float) -> int:
    """Index of the nearest entry in an ascending array to ``value``."""
    pos = int(np.searchsorted(freq_sorted_ascending, value))
    if pos == 0:
        return 0
    if pos >= freq_sorted_ascending.size:
        return int(freq_sorted_ascending.size - 1)
    if abs(value - freq_sorted_ascending[pos - 1]) <= abs(
        value - freq_sorted_ascending[pos]
    ):
        return pos - 1
    return pos


def _window_containing(freq_mhz: float, windows: list[FitWindow]) -> int | None:
    """Return the id of the window whose ``freq_range`` brackets ``freq_mhz``."""
    for w in windows:
        lo, hi = w.freq_range
        if lo <= freq_mhz <= hi:
            return w.window_id
    return None


def run_survey(
    ftmw_path: Path,
    output_dir: Path,
    *,
    min_user_snr: float | None,
    close_pair_bins: int,
    window_fwhm_factor: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading 2638 fixture: %s", ftmw_path)
    plan = ftmw.load_windows(str(ftmw_path))
    peaks = ftmw.load_peaks(str(ftmw_path))
    fit: SpectrumFit = ftmw.load_fit(str(ftmw_path))

    easy_ids = {
        w.window_id for w in plan.windows if w.difficulty == WindowDifficulty.EASY
    }
    easy_windows = [w for w in plan.windows if w.window_id in easy_ids]
    logger.info(
        "plan: %d windows (%d EASY, %d HARD); fit: %d window_fits, %d fitted peaks",
        plan.n_windows,
        len(easy_windows),
        plan.n_windows - len(easy_windows),
        len(fit.window_fits),
        fit.n_fitted_peaks,
    )

    # Build the active-FT once (the spectrum the projection test consumes).
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
    if expf_us is None:
        # Stage 1 wasn't apodized; fall back to T/3, the same default
        # tau0_us Stage 5 uses when expf_us is missing.
        tau_us = acquisition_us / 3.0
        logger.warning(
            "expf_us is None; falling back to tau_us = T/3 = %.3f us for the "
            "projection basis",
            tau_us,
        )
    else:
        tau_us = float(expf_us)

    # Per-bin active-FT noise: Stage 2 adaptive estimator on the sorted
    # active-FT magnitude. Reindex back to the active-FT bin order so each
    # ``sigma_arr[i]`` matches ``active_ft.complex_spectrum[i]``.
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freq_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    mag_sorted = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    noise = estimate_noise_adaptive(freq_sorted, mag_sorted)
    rms_sorted = np.asarray(noise.rms_noise, dtype=float)
    sigma_active = rms_sorted[unsort_idx]
    logger.info(
        "active-FT: %d bins, df=%.4f MHz, sigma median=%.3e",
        active_ft.freq_mhz.size,
        1.0 / acquisition_us,
        float(np.median(rms_sorted)),
    )

    # Survey set: persisted Stage 3 candidates whose user-grid SNR clears
    # ``min_user_snr`` (None => all detected) AND that land inside an EASY
    # window. The user-grid SNR is what production reports; using it as the
    # filter keeps the survey consistent with the production promotion flag.
    survey_peaks = []
    for peak in peaks:
        if min_user_snr is not None and peak.snr < min_user_snr:
            continue
        wid = _window_containing(peak.frequency, easy_windows)
        if wid is None:
            continue
        survey_peaks.append((wid, peak))
    logger.info(
        "survey set: %d candidates across %d EASY windows (min_user_snr=%s)",
        len(survey_peaks),
        len({wid for wid, _ in survey_peaks}),
        min_user_snr,
    )

    # Build the active-FT bin lookup for every fitted peak (ground truth).
    # Fitted peaks live on the user-grid frequency axis; we snap to the
    # nearest active-FT bin and store the set of bins per window.
    fit_by_wid: dict[int, list[int]] = {}
    for wf in fit.window_fits:
        bins_for_window: list[int] = []
        for fp in wf.fitted_peaks:
            b = _ascending_bin(freq_sorted, float(fp.frequency_mhz))
            bins_for_window.append(int(unsort_idx[b]))
        if bins_for_window:
            fit_by_wid[wf.window_id] = bins_for_window
    logger.info(
        "fitted-peak active-FT bins built for %d windows; %d total fitted peaks",
        len(fit_by_wid),
        sum(len(v) for v in fit_by_wid.values()),
    )

    # Pre-compute each candidate's active-FT bin and active-FT magnitude/SNR
    # (cheap, scalar per peak) before the projection.
    cand_freqs = np.array([float(p.frequency) for _, p in survey_peaks])
    active_bin_sorted = np.array(
        [_ascending_bin(freq_sorted, f) for f in cand_freqs], dtype=int
    )
    active_bin_native = unsort_idx[active_bin_sorted].astype(int)
    active_mag_native = np.abs(active_ft.complex_spectrum)[active_bin_native]
    safe_sigma = np.where(sigma_active > 0.0, sigma_active, 1.0)
    sigma_c_per_bin = safe_sigma / np.sqrt(2.0)
    active_snr_native = active_mag_native / sigma_c_per_bin[active_bin_native]

    # Run the projection.
    logger.info(
        "running projection (tau=%.3f us, T=%.3f us, window=%.1f FWHM)",
        tau_us,
        acquisition_us,
        window_fwhm_factor,
    )
    projections = project_candidates(
        active_ft.freq_mhz,
        active_ft.complex_spectrum,
        sigma_active,
        cand_freqs.tolist(),
        tau_us=tau_us,
        acquisition_us=acquisition_us,
        sideband=sideband_enum,
        window_fwhm_factor=window_fwhm_factor,
    )

    # close_pair: any other survey candidate within close_pair_bins active-FT
    # bins. Compute by sorting candidates within each window by frequency and
    # walking the sorted list.
    close_pair_flag = np.zeros(len(survey_peaks), dtype=bool)
    by_wid: dict[int, list[int]] = {}
    for i, (wid, _) in enumerate(survey_peaks):
        by_wid.setdefault(wid, []).append(i)
    for wid, idxs in by_wid.items():
        sub_bins = active_bin_native[idxs]
        # all-pairs distance in bins
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if abs(int(sub_bins[a]) - int(sub_bins[b])) <= close_pair_bins:
                    close_pair_flag[idxs[a]] = True
                    close_pair_flag[idxs[b]] = True

    # Ground truth: did any fitted peak in this candidate's window land within
    # ±1 active-FT bin of the candidate?
    became_peak = np.zeros(len(survey_peaks), dtype=bool)
    for i, (wid, _) in enumerate(survey_peaks):
        cand_b = int(active_bin_native[i])
        for fp_b in fit_by_wid.get(wid, []):
            if abs(fp_b - cand_b) <= 1:
                became_peak[i] = True
                break

    csv_path = output_dir / "candidates.csv"
    header = [
        "window_id",
        "frequency_mhz",
        "user_grid_intensity",
        "user_grid_snr",
        "user_grid_promoted",
        "internal_snr",
        "active_bin",
        "active_bin_freq_mhz",
        "active_magnitude",
        "active_snr",
        "coherent_amp",
        "coherent_snr",
        "detected_snr_active",
        "ratio",
        "projection_n_bins",
        "fwhm_mhz",
        "window_half_width_mhz",
        "close_pair",
        "became_fitted_peak",
        "detection_pass",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, (wid, peak) in enumerate(survey_peaks):
            pr = projections[i]
            writer.writerow(
                [
                    wid,
                    f"{peak.frequency:.6f}",
                    f"{peak.intensity:.6e}",
                    f"{peak.snr:.4f}",
                    bool(peak.properties.get("promoted", False)),
                    f"{peak.properties.get('internal_snr', float('nan')):.4f}",
                    int(active_bin_native[i]),
                    f"{active_ft.freq_mhz[active_bin_native[i]]:.6f}",
                    f"{active_mag_native[i]:.6e}",
                    f"{active_snr_native[i]:.4f}",
                    f"{pr.coherent_amp:.6e}",
                    f"{pr.coherent_snr:.4f}",
                    f"{pr.detected_snr_active:.4f}",
                    f"{pr.ratio:.6f}",
                    pr.projection_n_bins,
                    f"{pr.fwhm_mhz:.6f}",
                    f"{pr.window_half_width_mhz:.6f}",
                    bool(close_pair_flag[i]),
                    bool(became_peak[i]),
                    peak.properties.get("detection_pass", ""),
                ]
            )

    logger.info(
        "wrote %s (%d rows). became_peak fraction=%.3f, close_pair fraction=%.3f",
        csv_path,
        len(survey_peaks),
        float(became_peak.mean()) if len(survey_peaks) else 0.0,
        float(close_pair_flag.mean()) if len(survey_peaks) else 0.0,
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ftmw",
        type=Path,
        default=DEFAULT_FTMW_PATH,
        help="Path to the .ftmw fixture (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to drop candidates.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--min-user-snr",
        type=float,
        default=2.0,
        help="Minimum user-grid SNR to keep (None disables; default 2.0)",
    )
    parser.add_argument(
        "--close-pair-bins",
        type=int,
        default=2,
        help="Active-FT bin distance threshold for the close_pair flag "
        "(default 2 bins ≈ 2 FWHM on 2638)",
    )
    parser.add_argument(
        "--window-fwhm-factor",
        type=float,
        default=5.0,
        help="Projection sub-window half-width in FWHM units (default 5.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    min_snr = args.min_user_snr if args.min_user_snr > 0 else None
    run_survey(
        args.ftmw,
        args.output_dir,
        min_user_snr=min_snr,
        close_pair_bins=args.close_pair_bins,
        window_fwhm_factor=args.window_fwhm_factor,
    )


if __name__ == "__main__":
    main()
