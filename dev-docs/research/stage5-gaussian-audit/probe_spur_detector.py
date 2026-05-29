"""Spur-detector prototype for Stage 5 structural work item 1.

The Step 5 cross-reference flagged a ``spur`` bucket carrying ~106 excess
chi2r units (``report.md`` § Step 5): clock/LO spurs flow into the fitter, and a
single-bin CW delta cannot be represented by any finite-T line shape, so it
detonates chi2 even when the spur is (correctly) never fitted -- the w245 note
("excluded from the fit *and* the residual/chi2 calculation") is the
load-bearing part.

This throwaway prototype validates the detector *before* any production wiring:
it scans the active-FT for spur bins, then measures (a) the chi2r recovery if
those bins are masked from each window's residual sum and (b) the
false-positive risk -- does any flagged bin sit on a real fitted line?

Fingerprint (verified empirically on the fixture, not assumed):

* **exact integer-MHz center** -- every classified spur sits within a fraction
  of a bin of an integer MHz (29440, 30720, 32960, 34560, 35200, 35840, ...).
* **sub-resolution narrowness** -- a persistent CW tone is transform-limited by
  the full boxcar (first null ~1/T ~ one bin), so its peak bin is 10-25x its
  neighbours, whereas a real finite-T molecular line has a coherent leakage
  skirt where adjacent bins are comparable. Both gates are required, so a
  genuine line that happens to land near an integer MHz (it still has a skirt)
  is not masked.

The "energy in only one quadrature" criterion from the earlier note is *not*
used: both probed spurs show comparable Re/Im (the spur's phase relative to t0
is arbitrary), so it is not a reliable discriminator.

Run via the conda env::

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-gaussian-audit/probe_spur_detector.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from ftmwpipeline.fitting.plan_execution import materialize_window
from ftmwpipeline.fitting.validation import feature_fwhm

# Reuse the fixture loader from the shape-escalation probe (same audit dir).
from probe_shape_escalation import DEFAULT_FIXTURE, load_fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).parent / "data"
SPUR_CSV = DATA_DIR / "spur_detection.csv"
WINDOW_CSV = DATA_DIR / "spur_window_recovery.csv"

# Band carrying molecular signal on 2638 (matches the persisted Stage 1 trim).
BAND_MHZ: Tuple[float, float] = (26500.0, 40000.0)

# Detection thresholds (initial values; the prototype reports the separation so
# they can be tuned against the data before production wiring).
INTEGER_TOL_MHZ = 0.04   # ~half a bin (bin spacing ~79 kHz)
NARROWNESS_RATIO = 0.30   # max(neighbour)/peak below this => sub-resolution
SNR_THRESHOLD = 5.0       # peak-bin magnitude / local sigma_c floor

# Windows the user classified as spur (or noted a spur in), for recall scoring.
CLASSIFIED_SPUR_WINDOWS = {88, 126, 193, 264, 287, 372, 385, 386, 387, 390}
CLASSIFIED_SPUR_NOTED = {74, 245, 389}  # spur present but already unfitted

logger = logging.getLogger("spur-detector")


@dataclass
class Spur:
    integer_mhz: int
    bin_freq_mhz: float
    bin_index: int
    magnitude: float
    snr: float
    narrowness_ratio: float
    nearest_peak_freq_mhz: Optional[float]
    nearest_peak_sep_mhz: Optional[float]
    nearest_peak_window_id: Optional[int]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_spurs(
    freqs_sorted: np.ndarray,
    spec_sorted: np.ndarray,
    sig_c_sorted: np.ndarray,
    *,
    band: Tuple[float, float] = BAND_MHZ,
    integer_tol_mhz: float = INTEGER_TOL_MHZ,
    narrowness_ratio: float = NARROWNESS_RATIO,
    snr_threshold: float = SNR_THRESHOLD,
) -> Tuple[List[Spur], List[dict]]:
    """Flag integer-MHz, sub-resolution-narrow bins on the sorted active-FT.

    Returns ``(spurs, rejected)`` where ``rejected`` records integer-MHz
    candidates that cleared the SNR floor but failed the narrowness gate --
    i.e. real lines near an integer MHz the detector correctly did *not* flag.
    """
    mag = np.abs(spec_sorted)
    lo, hi = band
    spurs: List[Spur] = []
    rejected: List[dict] = []
    for f_int in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
        k = int(np.argmin(np.abs(freqs_sorted - f_int)))
        if abs(float(freqs_sorted[k]) - f_int) > integer_tol_mhz:
            continue
        peak = float(mag[k])
        sig = float(sig_c_sorted[k]) if sig_c_sorted[k] > 0 else float("nan")
        snr = peak / sig if sig > 0 else 0.0
        if snr < snr_threshold:
            continue
        left = float(mag[k - 1]) if k > 0 else 0.0
        right = float(mag[k + 1]) if k < mag.size - 1 else 0.0
        ratio = max(left, right) / peak if peak > 0 else 1.0
        record = dict(
            integer_mhz=f_int, bin_freq=float(freqs_sorted[k]), k=k,
            magnitude=peak, snr=snr, ratio=ratio,
        )
        if ratio <= narrowness_ratio:
            spurs.append(
                Spur(
                    integer_mhz=f_int,
                    bin_freq_mhz=float(freqs_sorted[k]),
                    bin_index=k,
                    magnitude=peak,
                    snr=snr,
                    narrowness_ratio=ratio,
                    nearest_peak_freq_mhz=None,
                    nearest_peak_sep_mhz=None,
                    nearest_peak_window_id=None,
                )
            )
        else:
            rejected.append(record)
    return spurs, rejected


# ---------------------------------------------------------------------------
# chi2 recovery per window
# ---------------------------------------------------------------------------
# Mask half-widths (in bins) swept per affected window. A strong CW tone is a
# full-window sinc whose skirt sits several sigma above noise for +/-2-3 bins
# (the "small" neighbour bins are still ~8 sigma because sigma is tiny), so a
# single-bin mask under-recovers; the sweep shows where chi2r plateaus.
MASK_HALF_WIDTHS: Tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class WindowRecovery:
    window_id: int
    freq_lo_mhz: float
    freq_hi_mhz: float
    n_spur_bins: int
    chi2r_persisted: float
    chi2r_full_recomputed: float
    chi2r_by_halfwidth: dict  # half-width (bins) -> chi2r after masking


def _full_model(wf, offset_grid, center_mhz, s, tau_shared, acq, shape):
    peaks: List[ModelPeak] = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center_mhz)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in wf.fitted_peaks
    ]
    for key, fp in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        peaks.append(
            ModelPeak(
                amplitude=float(fp["amplitude"]),
                offset_mhz=float(s * (float(fp["frequency_mhz"]) - center_mhz)),
                phase=float(fp.get("phase", 0.0) or 0.0),
            )
        )
    if not peaks or tau_shared <= 0:
        return np.zeros_like(offset_grid, dtype=np.complex128)
    return model_spectrum(offset_grid, peaks, tau_shared, acq, shape=shape)


def measure_recovery(ctx, spurs: Sequence[Spur]) -> List[WindowRecovery]:
    plan_by_id = {w.window_id: w for w in ctx.plan.windows}
    fit_by_id = {wf.window_id: wf for wf in ctx.fit.window_fits}
    s = sideband_sign(ctx.sideband)
    spur_freqs = np.array([sp.bin_freq_mhz for sp in spurs], dtype=float)
    out: List[WindowRecovery] = []
    bin_tol = INTEGER_TOL_MHZ  # a bin in-window is "the spur" if within tol
    for wid, window in sorted(plan_by_id.items()):
        wf = fit_by_id.get(wid)
        if wf is None:
            continue
        lo, hi = window.freq_range
        flo, fhi = min(lo, hi), max(lo, hi)
        if not np.any((spur_freqs >= flo) & (spur_freqs <= fhi)):
            continue
        freq_slice, offset_grid, z_slice, sig_slice, center = materialize_window(
            window, ctx.active_ft, ctx.active_noise_arr, sideband=ctx.sideband
        )
        tau_shared = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
        model = _full_model(
            wf, offset_grid, center, s, tau_shared, ctx.acquisition_us, ctx.shape
        )
        sig_ri = np.asarray(sig_slice, float) / np.sqrt(2.0)
        r = (np.asarray(z_slice) - model) / sig_ri
        per_bin = r.real ** 2 + r.imag ** 2
        # spur bins in this window's slice
        in_win = [sp for sp in spurs if flo <= sp.bin_freq_mhz <= fhi]
        spur_bin_mask = np.zeros(freq_slice.size, dtype=bool)
        for sp in in_win:
            j = int(np.argmin(np.abs(freq_slice - sp.bin_freq_mhz)))
            if abs(float(freq_slice[j]) - sp.bin_freq_mhz) <= bin_tol:
                spur_bin_mask[j] = True
        n_spur = int(spur_bin_mask.sum())
        spur_centers = np.where(spur_bin_mask)[0]
        m = freq_slice.size
        n_params = 3 * len(wf.fitted_peaks) + (1 if tau_shared > 0 else 0)
        chi2_full = float(per_bin.sum())
        dof_full = max(2 * m - n_params, 1)
        by_hw: dict = {}
        for hw in MASK_HALF_WIDTHS:
            mask = np.zeros(m, dtype=bool)
            for j in spur_centers:
                mask[max(0, j - hw):min(m, j + hw + 1)] = True
            dof = max(2 * (m - int(mask.sum())) - n_params, 1)
            by_hw[hw] = float(per_bin[~mask].sum()) / dof
        out.append(
            WindowRecovery(
                window_id=wid,
                freq_lo_mhz=flo,
                freq_hi_mhz=fhi,
                n_spur_bins=n_spur,
                chi2r_persisted=float(wf.reduced_chi2),
                chi2r_full_recomputed=chi2_full / dof_full,
                chi2r_by_halfwidth=by_hw,
            )
        )
    return out


def _annotate_nearest_peaks(ctx, spurs: List[Spur]) -> None:
    """Fill each spur's nearest *fitted* peak (false-positive check)."""
    s = sideband_sign(ctx.sideband)
    peaks: List[Tuple[float, int]] = []
    for wf in ctx.fit.window_fits:
        for p in wf.fitted_peaks:
            peaks.append((float(p.frequency_mhz), int(wf.window_id)))
    if not peaks:
        return
    pf = np.array([p[0] for p in peaks])
    pw = [p[1] for p in peaks]
    for sp in spurs:
        d = np.abs(pf - sp.integer_mhz)
        j = int(np.argmin(d))
        sp.nearest_peak_freq_mhz = float(pf[j])
        sp.nearest_peak_sep_mhz = float(d[j])
        sp.nearest_peak_window_id = pw[j]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(fixture: Path) -> None:
    ctx = load_fixture(fixture)
    sort_idx = np.argsort(ctx.active_ft.freq_mhz)
    freqs_sorted = np.ascontiguousarray(ctx.active_ft.freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(ctx.active_ft.complex_spectrum[sort_idx])
    sig_c_sorted = np.ascontiguousarray(ctx.active_noise_arr[sort_idx]) / np.sqrt(2.0)
    bin_khz = float(np.median(np.diff(freqs_sorted))) * 1000.0
    logger.info("active-FT bin spacing ~ %.1f kHz", bin_khz)

    spurs, rejected = detect_spurs(freqs_sorted, spec_sorted, sig_c_sorted)
    _annotate_nearest_peaks(ctx, spurs)

    logger.info("detected %d spur bins (integer-MHz + narrowness)", len(spurs))
    print("\n=== detected spurs ===")
    print(f"{'int MHz':>8} {'bin freq':>11} {'snr':>8} {'ratio':>6} "
          f"{'near peak':>11} {'sep MHz':>9} {'win':>5}")
    for sp in spurs:
        print(
            f"{sp.integer_mhz:>8} {sp.bin_freq_mhz:>11.4f} {sp.snr:>8.1f} "
            f"{sp.narrowness_ratio:>6.3f} "
            f"{(sp.nearest_peak_freq_mhz or float('nan')):>11.4f} "
            f"{(sp.nearest_peak_sep_mhz or float('nan')):>9.4f} "
            f"{sp.nearest_peak_window_id if sp.nearest_peak_window_id is not None else -1:>5}"
        )

    # The genuine false-positive guard is the narrowness gate: it must never
    # flag a *real* molecular line at an integer MHz. The 354 spared candidates
    # below (real lines that cleared the SNR floor but are broad, ratio > gate)
    # are that evidence. Proximity of a detected spur to a *fitted* peak is NOT
    # a false positive -- on a pure-spur window a spurious peak was placed ON
    # the spur, which is exactly what nomination-exclusion would remove.
    fwhm_typ = feature_fwhm(ctx.tau0_us, ctx.acquisition_us, shape=ctx.shape)
    near_fit = [
        sp for sp in spurs
        if sp.nearest_peak_sep_mhz is not None
        and sp.nearest_peak_sep_mhz < fwhm_typ
    ]
    print(f"\ntypical FWHM ~ {fwhm_typ*1000:.1f} kHz")
    print(f"detected spurs with a fitted peak within 1 FWHM "
          f"(= spurious peak placed on the spur, nomination-exclusion target): "
          f"{len(near_fit)}")
    print(f"real integer-MHz lines spared by the narrowness gate (false-positive "
          f"guard working): {len(rejected)} "
          f"(e.g. strongest spared: "
          f"{max(rejected, key=lambda r: r['snr'])['integer_mhz']} MHz "
          f"snr={max(rejected, key=lambda r: r['snr'])['snr']:.0f} "
          f"ratio={max(rejected, key=lambda r: r['snr'])['ratio']:.2f})")

    # Recall vs the user's classifications.
    recovery = measure_recovery(ctx, spurs)
    detected_windows = {r.window_id for r in recovery if r.n_spur_bins > 0}
    missed = CLASSIFIED_SPUR_WINDOWS - detected_windows
    extra = detected_windows - CLASSIFIED_SPUR_WINDOWS - CLASSIFIED_SPUR_NOTED
    print(f"\n=== recall vs classifications ===")
    print(f"classified spur windows: {sorted(CLASSIFIED_SPUR_WINDOWS)}")
    print(f"detected (>=1 spur bin):  {sorted(detected_windows)}")
    print(f"missed classified spur windows: {sorted(missed) or 'none'}")
    print(f"detected in non-spur-classified windows: {sorted(extra) or 'none'} "
          f"(may be spurs the user left to prominence-luck)")

    print(f"\n=== chi2r recovery per affected window (by mask half-width, bins) ===")
    hw_hdr = " ".join(f"{'+-'+str(hw):>7}" for hw in MASK_HALF_WIDTHS)
    print(f"{'win':>5} {'persisted':>10} {'recomp':>9} {hw_hdr}")
    totals = {hw: 0.0 for hw in MASK_HALF_WIDTHS}
    for r in sorted(recovery, key=lambda r: r.chi2r_full_recomputed, reverse=True):
        row = " ".join(f"{r.chi2r_by_halfwidth[hw]:>7.2f}" for hw in MASK_HALF_WIDTHS)
        print(f"{r.window_id:>5} {r.chi2r_persisted:>10.3f} "
              f"{r.chi2r_full_recomputed:>9.2f} {row}")
        for hw in MASK_HALF_WIDTHS:
            totals[hw] += r.chi2r_full_recomputed - r.chi2r_by_halfwidth[hw]
    print(f"\nsum(chi2r) reduction by mask half-width:")
    for hw in MASK_HALF_WIDTHS:
        print(f"  +-{hw} bins: {totals[hw]:6.1f}")
    print("(report's spur bucket = ~106 excess chi2r over 10 classified windows;"
          " that counts sum(chi2r-1), not the recoverable amount)")

    # CSVs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SPUR_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["integer_mhz", "bin_freq_mhz", "snr", "narrowness_ratio",
                    "nearest_peak_freq_mhz", "nearest_peak_sep_mhz",
                    "nearest_peak_window_id"])
        for sp in spurs:
            w.writerow([sp.integer_mhz, f"{sp.bin_freq_mhz:.4f}", f"{sp.snr:.2f}",
                        f"{sp.narrowness_ratio:.4f}",
                        f"{sp.nearest_peak_freq_mhz:.4f}",
                        f"{sp.nearest_peak_sep_mhz:.4f}",
                        sp.nearest_peak_window_id])
    with WINDOW_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_id", "freq_lo_mhz", "freq_hi_mhz", "n_spur_bins",
                    "chi2r_persisted", "chi2r_full_recomputed"]
                   + [f"chi2r_mask_hw{hw}" for hw in MASK_HALF_WIDTHS])
        for r in recovery:
            w.writerow([r.window_id, f"{r.freq_lo_mhz:.4f}", f"{r.freq_hi_mhz:.4f}",
                        r.n_spur_bins, f"{r.chi2r_persisted:.6f}",
                        f"{r.chi2r_full_recomputed:.6f}"]
                       + [f"{r.chi2r_by_halfwidth[hw]:.6f}" for hw in MASK_HALF_WIDTHS])
    logger.info("wrote %s and %s", SPUR_CSV, WINDOW_CSV)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = ap.parse_args()
    if not args.fixture.exists():
        raise SystemExit(f"missing fixture {args.fixture}")
    run(args.fixture)


if __name__ == "__main__":
    main()
