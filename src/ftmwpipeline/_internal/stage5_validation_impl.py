"""Stage 5 cross-fixture validation (``validate-stage5-shape-error``).

Read-only assessment of an already-persisted Stage 5 fit against the
cross-fixture acceptance framework (``dev-docs/planning/stage5-cross-fixture-
validation.md``). Consumes ``/stage5_fitting`` via the same validated loader the
functional API uses (:func:`load_fit_impl`) and emits three tiers:

* **Tier 1 -- SNR-aware distribution health.** Per window, the gate
  ``chi2r <= F + (kappa * SNR_max)**2`` with the fractional model deficit
  ``eps = sqrt(max(chi2r - F, 0)) / SNR_max`` reported alongside, binned by the
  brightest in-window peak SNR. At extreme SNR the per-window reduced chi-squared
  is a model-fidelity floor, not a noise statistic, so the raw ``chi2r <=
  threshold`` gate is meaningless; the SNR-aware form collapses to ``chi2r <= F``
  (the noise-regime allowance) in the noise-dominated regime and grows with
  ``SNR^2`` in the deficit-dominated regime. See
  :func:`ftmwpipeline.fitting.validation.snr_aware_chi2_pass`.

* **Tier 2 -- gate firing.** The rescue/merge machinery's activity: merge fire
  rate, the ``n_pruned_rescue_origin`` failsafe, a limit-cycle count, and the
  rescue/thaw/replan history sizes.

* **Tier 3 -- known-line ground truth (optional).** When a catalog CSV is
  supplied, matches fitted lines to it and reports recall, a (caveated)
  precision, the frequency-residual statistics, the reported-sigma honesty, and
  the instrument frequency-accuracy floor (a constant, SNR-independent offset +
  drift the per-line LSQ sigma cannot capture -- reported, never failed on).

Nothing here mutates the file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.data_structures import FittingResult, SpectrumFit
from ..fitting.validation import (
    DEFAULT_CHI2R_NOISE_FLOOR,
    DEFAULT_SHAPE_ERROR_KAPPA,
    feature_fwhm,
    shape_error_fraction,
    snr_aware_chi2_pass,
)
from .stage5_impl import load_fit_impl

# SNR_max bin edges / labels -- the natural breakdown the cross-fixture report
# reads off (noise-dominated bulk through the floor-limited bright cores).
_SNR_BIN_EDGES = (100.0, 1000.0, 10000.0)
_SNR_BIN_LABELS = ("<100", "100-1k", "1k-10k", ">=10k")
# A rescue round that adds and re-merges peaks with chi-squared unmoved (matches
# the cross-fixture harness's limit-cycle definition).
_LIMIT_CYCLE_CHI2_REL = 0.02


def _window_snr_max(wf: FittingResult) -> float:
    """Brightest in-window fitted-peak SNR (0 for an empty / SNR-less window)."""
    snrs = [
        float(p.snr)
        for p in wf.fitted_peaks
        if p.snr is not None and np.isfinite(p.snr)
    ]
    return max(snrs) if snrs else 0.0


def _snr_bin_index(snr: float) -> int:
    """Index into :data:`_SNR_BIN_LABELS` for a window's ``SNR_max``."""
    for i, edge in enumerate(_SNR_BIN_EDGES):
        if snr < edge:
            return i
    return len(_SNR_BIN_EDGES)


def _tier1(fit: SpectrumFit, kappa: float, noise_floor: float) -> Dict[str, Any]:
    """SNR-aware per-window acceptance, binned by ``SNR_max``."""
    per_window: List[Dict[str, Any]] = []
    for wf in fit.window_fits:
        chi2r = float(getattr(wf, "reduced_chi2", float("inf")))
        snr_max = _window_snr_max(wf)
        per_window.append(
            {
                "window_id": int(wf.window_id) if wf.window_id is not None else -1,
                "reduced_chi2": chi2r,
                "snr_max": round(snr_max, 2),
                "epsilon": round(shape_error_fraction(chi2r, snr_max, noise_floor), 5),
                "snr_bin": _SNR_BIN_LABELS[_snr_bin_index(snr_max)],
                "pass": snr_aware_chi2_pass(chi2r, snr_max, kappa, noise_floor),
            }
        )

    n_windows = len(per_window)
    if n_windows == 0:
        return {"n_windows": 0, "kappa": kappa, "noise_floor": noise_floor}

    # SNR-binned aggregates.
    bins: List[Dict[str, Any]] = []
    for label in _SNR_BIN_LABELS:
        rows = [w for w in per_window if w["snr_bin"] == label]
        if not rows:
            continue
        finite = np.array(
            [w["reduced_chi2"] for w in rows if np.isfinite(w["reduced_chi2"])]
        )
        eps = np.array([w["epsilon"] for w in rows])
        bins.append(
            {
                "snr_bin": label,
                "n": len(rows),
                "chi2r_median": (
                    round(float(np.median(finite)), 3) if finite.size else None
                ),
                "epsilon_median": round(float(np.median(eps)), 5),
                "pass_rate": round(sum(w["pass"] for w in rows) / len(rows), 3),
            }
        )

    finite_all = np.array(
        [w["reduced_chi2"] for w in per_window if np.isfinite(w["reduced_chi2"])]
    )
    n_pass = sum(w["pass"] for w in per_window)
    return {
        "n_windows": n_windows,
        "kappa": kappa,
        "noise_floor": noise_floor,
        "pass_rate": round(n_pass / n_windows, 3),
        "n_pass": int(n_pass),
        "n_fail": int(n_windows - n_pass),
        # Raw distribution kept for reference (the superseded gate's numbers).
        "chi2r_median": (
            round(float(np.median(finite_all)), 3) if finite_all.size else None
        ),
        "chi2r_p95": (
            round(float(np.percentile(finite_all, 95)), 3) if finite_all.size else None
        ),
        "chi2r_max": (round(float(finite_all.max()), 3) if finite_all.size else None),
        "snr_bins": bins,
        "windows": per_window,
    }


def _tier2(fit: SpectrumFit) -> Dict[str, Any]:
    """Rescue / merge / thaw gate firing rates (planning-doc Tier 2)."""
    n_windows = len(fit.window_fits)
    rounds = list(fit.rescue_history)
    merged_windows = {r.window_id for r in rounds if r.n_merged > 0}
    n_origin_pruned = sum(int(r.n_pruned_rescue_origin) for r in rounds)
    limit_cycle = 0
    for r in rounds:
        denom = max(abs(r.chi2_before), 1e-12)
        unmoved = abs(r.chi2_after - r.chi2_before) / denom < _LIMIT_CYCLE_CHI2_REL
        if unmoved and r.n_rescue_added > 0 and r.n_merged > 0:
            limit_cycle += 1
    return {
        "merge_fire_windows": len(merged_windows),
        "merge_fire_rate": (
            round(len(merged_windows) / n_windows, 3) if n_windows else 0.0
        ),
        "n_pruned_rescue_origin": int(n_origin_pruned),
        "limit_cycle_rounds": int(limit_cycle),
        "n_rescue_rounds": len(rounds),
        "n_rescue_accepted": sum(1 for r in rounds if r.accepted),
        "n_thaw": len(fit.thaw_history),
        "n_thaw_accepted": sum(1 for t in fit.thaw_history if t.accepted),
        "n_replan": len(fit.replan_history),
        "final_plan_revision": int(fit.final_plan_revision),
    }


def _read_catalog_freqs(path: str) -> List[Dict[str, float]]:
    """Parse a ground-truth CSV into ``[{freq_mhz, unc_mhz, log_intensity}, ...]``.

    Accepts both the parsed ``combined_lines.csv`` / ``lines.csv`` schemas (a
    ``freq_mhz`` column is required; ``unc_mhz`` and ``log_intensity`` are
    optional). The files are pre-filtered to the analysis band, so every row is
    treated as an in-band ground-truth line.
    """
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "freq_mhz" not in reader.fieldnames:
            raise ValueError(
                f"ground-truth CSV {path!r} must have a 'freq_mhz' column; "
                f"found {reader.fieldnames!r}"
            )
        for row in reader:
            try:
                freq = float(row["freq_mhz"])
            except (TypeError, ValueError):
                continue
            entry = {"freq_mhz": freq}
            if row.get("unc_mhz") not in (None, ""):
                entry["unc_mhz"] = float(row["unc_mhz"])
            if row.get("log_intensity") not in (None, ""):
                entry["log_intensity"] = float(row["log_intensity"])
            rows.append(entry)
    rows.sort(key=lambda r: r["freq_mhz"])
    return rows


def _window_tau_map(fit: SpectrumFit) -> Dict[int, float]:
    """``window_id -> fitted tau_us`` from each window's shared parameters."""
    out: Dict[int, float] = {}
    for wf in fit.window_fits:
        if wf.window_id is None:
            continue
        tau_entry = wf.shared_parameters.get("tau_us")
        if tau_entry is None:
            continue
        val = tau_entry.get("value")
        if val is not None and np.isfinite(val) and float(val) > 0.0:
            out[int(wf.window_id)] = float(val)
    return out


def _nearest_indices(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Index into sorted ``ref`` of the nearest value to each ``query`` entry."""
    pos = np.searchsorted(ref, query)
    pos = np.clip(pos, 1, len(ref) - 1)
    left = ref[pos - 1]
    right = ref[pos]
    choose_left = (query - left) <= (right - query)
    nearest: np.ndarray = np.where(choose_left, pos - 1, pos)
    return nearest


def _tier3(
    fit: SpectrumFit,
    ground_truth: str,
    match_tol_fwhm: float,
) -> Dict[str, Any]:
    """Match fitted lines to a ground-truth catalog and assess accuracy.

    Mutual-nearest-neighbour matching within ``match_tol_fwhm * FWHM`` (FWHM from
    each peak's window tau via :func:`feature_fwhm`). Reports recall, a caveated
    precision (the spectrum legitimately carries real lines absent from the
    catalog -- vibrational satellites, unmodelled species -- so unmatched fitted
    peaks are not necessarily spurious), the frequency-residual statistics, the
    reported-sigma honesty, and the detrended instrument accuracy floor.
    """
    catalog = _read_catalog_freqs(ground_truth)
    n_catalog = len(catalog)
    peaks = [p for p in fit.fitted_peaks if np.isfinite(p.frequency_mhz)]
    if n_catalog == 0 or not peaks:
        return {
            "ground_truth": str(ground_truth),
            "n_catalog": n_catalog,
            "n_fitted": len(peaks),
            "n_matched": 0,
            "recall": 0.0,
            "note": "no catalog lines or no fitted peaks to match",
        }

    acquisition_us = float(fit.parameters.get("acquisition_us", 0.0) or 0.0)
    shape = str(fit.parameters.get("shape", "lorentzian"))
    tau_map = _window_tau_map(fit)
    tau_default = (
        float(np.median(list(tau_map.values())))
        if tau_map
        else float(fit.parameters.get("tau_maj_us", 0.0) or 0.0)
    )

    def _fwhm_for(window_id: Optional[int]) -> float:
        tau = (
            tau_map.get(int(window_id), tau_default)
            if window_id is not None
            else tau_default
        )
        if tau <= 0.0 or acquisition_us <= 0.0:
            return 0.0
        return feature_fwhm(tau, acquisition_us, shape=shape)

    # Sorted fitted/catalog frequency arrays for mutual-nearest matching.
    order = np.argsort([p.frequency_mhz for p in peaks])
    peaks = [peaks[i] for i in order]
    ff = np.array([p.frequency_mhz for p in peaks])
    cf = np.array([c["freq_mhz"] for c in catalog])
    tol = np.array([match_tol_fwhm * _fwhm_for(p.window_id) for p in peaks])

    c2f = _nearest_indices(cf, ff)  # nearest fitted index for each catalog line
    f2c = _nearest_indices(ff, cf)  # nearest catalog index for each fitted peak

    matches: List[Tuple[int, int]] = []  # (catalog_idx, fitted_idx)
    for ci in range(n_catalog):
        fi = int(c2f[ci])
        if int(f2c[fi]) == ci and abs(ff[fi] - cf[ci]) <= tol[fi] and tol[fi] > 0.0:
            matches.append((ci, fi))

    n_matched = len(matches)
    # Residuals (kHz) and reported-sigma honesty on the matched pairs.
    resid_khz = np.array([(ff[fi] - cf[ci]) * 1e3 for ci, fi in matches])
    match_freqs = np.array([cf[ci] for ci, _ in matches])
    reported_err_khz = np.array(
        [(peaks[fi].frequency_error or np.nan) * 1e3 for _, fi in matches]
    )

    result: Dict[str, Any] = {
        "ground_truth": str(ground_truth),
        "n_catalog": n_catalog,
        "n_fitted": len(peaks),
        "n_matched": n_matched,
        "recall": round(n_matched / n_catalog, 3),
        "precision": round(n_matched / len(peaks), 3),
        "precision_note": (
            "loose upper bound on spuriousness: the spectrum carries real lines "
            "absent from this catalog, so an unmatched fitted peak is not "
            "necessarily spurious"
        ),
        "match_tol_fwhm": match_tol_fwhm,
    }
    if n_matched >= 2:
        result["freq_residual_khz"] = {
            "median": round(float(np.median(resid_khz)), 4),
            "mean": round(float(np.mean(resid_khz)), 4),
            "rms": round(float(np.sqrt(np.mean(resid_khz**2))), 4),
            "mad": round(float(np.median(np.abs(resid_khz - np.median(resid_khz)))), 4),
            "max_abs": round(float(np.max(np.abs(resid_khz))), 4),
        }
        # Instrument accuracy floor: remove a linear-in-frequency drift + a
        # constant offset (free-running digitizer clock, 1-2 ppm) and report the
        # residual scatter -- the SNR-independent floor the per-line sigma_f
        # cannot capture.
        slope, intercept = np.polyfit(match_freqs, resid_khz, 1)
        detrended = resid_khz - (slope * match_freqs + intercept)
        result["accuracy_floor"] = {
            "raw_rms_khz": round(float(np.sqrt(np.mean(resid_khz**2))), 4),
            "detrended_rms_khz": round(float(np.std(detrended)), 4),
            "drift_slope_khz_per_ghz": round(float(slope * 1e3), 4),
            "offset_khz": round(float(intercept), 4),
            "note": (
                "the detrended scatter is an instrument property (clock drift), "
                "not a pipeline defect; report it, do not fail on it"
            ),
        }
        # Reported-sigma honesty: how many reported sigma the residual sits at.
        good = np.isfinite(reported_err_khz) & (reported_err_khz > 0)
        if good.any():
            ratio = np.abs(resid_khz[good]) / reported_err_khz[good]
            result["sigma_f_honesty"] = {
                "median_reported_sigma_khz": round(
                    float(np.median(reported_err_khz[good])), 4
                ),
                "median_residual_over_sigma": round(float(np.median(ratio)), 2),
                "note": (
                    "median residual / reported sigma_f; >> 1 means sigma_f is an "
                    "honest LSQ precision but overconfident as absolute accuracy"
                ),
            }
    return result


def validate_stage5_shape_error_impl(
    file_path: str,
    *,
    kappa: Optional[float] = None,
    noise_floor: Optional[float] = None,
    ground_truth: Optional[str] = None,
    match_tol_fwhm: float = 0.5,
) -> Dict[str, Any]:
    """Assess a persisted Stage 5 fit against the SNR-aware acceptance framework.

    Read-only. Loads ``/stage5_fitting`` via :func:`load_fit_impl` and returns a
    Tier 1 (SNR-aware health) / Tier 2 (gate firing) / Tier 3 (ground truth, when
    ``ground_truth`` is given) report. Raises ``ValueError`` if the file has no
    Stage 5 fit.

    Parameters
    ----------
    file_path : str
        Path to a ``.ftmw`` file with a completed Stage 5 fit.
    kappa : float, optional
        Tolerated fractional model deficit for the SNR-aware gate; defaults to
        :data:`ftmwpipeline.fitting.validation.DEFAULT_SHAPE_ERROR_KAPPA`.
    noise_floor : float, optional
        Noise-regime allowance ``F`` in ``chi2r <= F + (kappa*SNR_max)**2``;
        defaults to
        :data:`ftmwpipeline.fitting.validation.DEFAULT_CHI2R_NOISE_FLOOR`.
    ground_truth : str, optional
        Path to a catalog CSV (a ``freq_mhz`` column) to run Tier 3 against.
    match_tol_fwhm : float, default 0.5
        Tier-3 match tolerance in units of the per-window line FWHM.

    Returns
    -------
    dict
        ``{status, parameters, tier1, tier2, tier3}`` (``tier3`` is ``None`` when
        no catalog was supplied).
    """
    kappa_v = DEFAULT_SHAPE_ERROR_KAPPA if kappa is None else float(kappa)
    floor_v = DEFAULT_CHI2R_NOISE_FLOOR if noise_floor is None else float(noise_floor)
    if ground_truth is not None and not Path(ground_truth).exists():
        raise FileNotFoundError(f"ground-truth catalog not found: {ground_truth}")

    fit = load_fit_impl(file_path)["fit"]

    tier3 = (
        _tier3(fit, ground_truth, match_tol_fwhm) if ground_truth is not None else None
    )
    return {
        "status": "success",
        "parameters": {
            "kappa": kappa_v,
            "noise_floor": floor_v,
            "match_tol_fwhm": match_tol_fwhm,
            "ground_truth": str(ground_truth) if ground_truth else None,
            "shape": str(fit.parameters.get("shape", "lorentzian")),
            "acquisition_us": float(fit.parameters.get("acquisition_us", 0.0) or 0.0),
        },
        "tier1": _tier1(fit, kappa_v, floor_v),
        "tier2": _tier2(fit),
        "tier3": tier3,
    }
