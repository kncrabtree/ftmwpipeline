"""
Shared implementation for Stage 3: Peak detection.

Orchestration only -- the detection algorithm lives in
``ftmwpipeline.preprocessing.peak_detection``. Stage 3 does **not** own any FT
settings: the spectrum the user chose (Stage 1 canonical ``ft_processing``,
incl. ``trim``) is authoritative. Detection runs *internally* at ``zpf=1`` on
two recomputed spectra -- an apodized primary (robust position finding) and an
unapodized full-resolution gap spectrum (weak-line recovery) -- because apex
localization is best at the native grid. The primary pass applies a strong
window function (default Blackman-Harris) chosen purely to suppress
truncation-leakage sidelobes so the strong-line list it produces -- which
seeds the gap pass's leakage mask -- is not itself polluted by sidelobes; see
``dev-docs/research/peak-detection/report.md`` for the calibration. The
primary apodization is independent of the user's Stage 1 settings and affects
only *which positions* are found, never any reported amplitude or SNR. Every
detected peak is then snapped
back onto the user's persisted spectrum by physical frequency: its amplitude
is re-measured on the user-settings ``ComplexFT`` and its SNR against the
canonical Stage 2 noise, so the stored/returned result is expressed entirely
on the user grid. The internal-grid values are kept under ``properties`` for
diagnostics. Wrapped identically by the CLI, Pipeline class, and functional
API.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..core.data_structures import ComplexFT, Peak
from ..preprocessing.noise_estimation import estimate_noise_adaptive
from ..preprocessing.peak_detection import (
    DEFAULT_INTERNAL_MIN_SNR,
    DEFAULT_MEDIUM_STRONG_SNR,
    DEFAULT_MIN_SNR,
    DEFAULT_WEAK_MEDIUM_SNR,
    classify_by_snr,
    detect_peaks,
)
from ..io.noise_result_serialization import load_noise_result_from_hdf5
from ..io.peak_serialization import (
    load_peaks_from_hdf5,
    save_peaks_to_hdf5,
)
from ..file_manager import invalidate_downstream_stages
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion

logger = logging.getLogger(__name__)

# Apex localization runs at the native grid; the user's persisted zpf governs
# only the reported/stored spectrum (snap-back re-measures there).
_DETECTION_ZPF = 1

# Primary-pass apodization: a strong window function suppresses truncation
# sidelobes so the primary pass's strong-line list (which seeds the gap-pass
# leakage mask) is clean. Blackman-Harris is the calibrated default -- on the
# 2638 fixture it removes ~5x the sidelobe-suspect detections that the mild
# Stage-1 exponential filter leaves behind. See
# dev-docs/research/peak-detection/report.md sections 3 and 6.
DEFAULT_PRIMARY_WINDOW = "blackmanharris"


def _active_acquisition_us(
    fid_duration_us: float, start_us: Optional[float], end_us: Optional[float]
) -> float:
    """Effective acquisition length T (µs) of the analysed FID window."""
    lo = 0.0 if start_us is None else float(start_us)
    hi = fid_duration_us if end_us is None else float(end_us)
    return max(hi - lo, 0.0)


def _spectrum_from_fid(
    fid: Any,
    base_pp: Any,
    trim_range: Optional[Tuple[float, float]],
    expf_us: Optional[float],
    window_function: Optional[str] = None,
) -> ComplexFT:
    """Recompute a ComplexFT from the FID at the internal detection grid.

    Always uses :data:`_DETECTION_ZPF` (native resolution -- best for apex
    localization). With ``expf_us=None`` and ``window_function=None`` this
    yields the unapodized boxcar spectrum used by the gap pass; passing a
    ``window_function`` (e.g. ``"blackmanharris"``) yields the
    leakage-suppressed primary spectrum. Window bounds, units, rdc and the
    persisted ``trim`` are held identical to the user's settings so both
    detection spectra share a consistent physical-frequency axis with the
    user spectrum. The primary apodization is deliberately *not* tied to the
    user's Stage 1 ``winf``/``expf_us``: it is a Stage 3 position-finding
    choice only (see the module docstring).
    """
    preprocessed = fid.preprocess(
        start_us=base_pp.start_us,
        end_us=base_pp.end_us,
        zpf=_DETECTION_ZPF,
        expf_us=expf_us,
        window_function=window_function,
        rdc=base_pp.rdc,
        units_power=base_pp.units_power,
    )
    spectrum, freqs = preprocessed.compute_fft()
    cft = ComplexFT.from_spectrum(spectrum, freqs)
    if trim_range is not None:
        cft = cft.trim_to_range(trim_range[0], trim_range[1])
    return cft


def _nearest_index(sorted_pairs: Tuple[np.ndarray, np.ndarray], value: float) -> int:
    """Index into the original array of the frequency closest to ``value``.

    ``sorted_pairs`` is ``(order, sorted_freq)`` precomputed once; handles
    ascending or descending frequency axes (2638 is descending).
    """
    order, sorted_f = sorted_pairs
    pos = int(np.searchsorted(sorted_f, value))
    pos = min(max(pos, 1), len(sorted_f) - 1)
    if abs(value - sorted_f[pos - 1]) <= abs(value - sorted_f[pos]):
        pos -= 1
    return int(order[pos])


def _snap_to_user_grid(
    internal_peaks: List[Peak],
    user_ft: ComplexFT,
    user_rms: np.ndarray,
    weak_medium_snr: float,
    medium_strong_snr: float,
    promotion_min_snr: float,
) -> List[Peak]:
    """Re-express internal-grid detections on the persisted user spectrum.

    For each detection: snap by physical frequency to the nearest user-grid
    point, re-measure amplitude on the user spectrum and SNR against the
    canonical Stage 2 noise, and reclassify. De-duplicates collisions on the
    user grid (keeps the strongest). Internal-grid SNR/frequency are preserved
    under ``properties`` for curation/diagnosis, and a ``promoted`` flag marks
    whether the user-grid SNR meets the Stage 4 promotion cutoff. All
    detections are kept (promotion is a downstream gate, not a filter here).
    """
    user_freq = user_ft.freq_array
    user_mag = user_ft.magnitude_spectrum
    order = np.argsort(user_freq)
    sorted_pairs = (order, user_freq[order])

    by_user_idx: Dict[int, Peak] = {}
    for p in internal_peaks:
        ui = _nearest_index(sorted_pairs, p.frequency)
        intensity = float(user_mag[ui])
        sd = float(user_rms[ui])
        snr = intensity / sd if sd > 0 else 0.0
        snapped = Peak(
            frequency=float(user_freq[ui]),
            intensity=intensity,
            index=int(ui),
            snr=snr,
            noise_std_local=sd,
            classification=classify_by_snr(
                snr, weak_medium_snr, medium_strong_snr
            ),
            detection_pass=p.properties.get("detection_pass"),
            internal_frequency=p.frequency,
            internal_snr=p.snr,
            promoted=snr >= promotion_min_snr,
        )
        prev = by_user_idx.get(ui)
        if prev is None or snapped.intensity > prev.intensity:
            by_user_idx[ui] = snapped

    return sorted(by_user_idx.values(), key=lambda q: q.frequency)


def _load_canonical_noise(file_path: str, user_ft: ComplexFT) -> np.ndarray:
    """Reconstruct the canonical Stage 2 noise on the user spectrum grid."""
    with h5py.File(file_path, "r") as h5f:
        noise = load_noise_result_from_hdf5(
            h5f["stage2_noise_result"],
            user_ft.freq_array,
            user_ft.magnitude_spectrum,
        )
    return np.asarray(noise.rms_noise, dtype=float)


def detect_peaks_impl(
    file_path: str,
    min_snr: Optional[float] = None,
    weak_medium_snr: Optional[float] = None,
    medium_strong_snr: Optional[float] = None,
    sg_window: Optional[int] = None,
    sg_order: Optional[int] = None,
    primary_window: Optional[str] = None,
    tau_us: Optional[float] = None,
    min_exclusion_mhz: Optional[float] = None,
    run_gap_pass: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run Stage 3 two-pass peak detection and persist the result.

    Requires Stage 1 (canonical FT settings) and Stage 2 (noise) completed.
    Detection operates on the user's persisted spectrum: there is no Stage 3
    ``trim``/``zpf`` -- those come from the Stage 1 canonical record.

    ``min_snr`` is the **promotion cutoff** on the user-grid SNR -- which
    peaks move on to Stage 4 -- not the detection floor. Detection always runs
    aggressively on the internal zpf=1 grids at
    ``min(DEFAULT_INTERNAL_MIN_SNR, promotion)`` (cheap, and recovers real
    peaks the user-grid re-measure would otherwise miss). *Every* detected
    peak is persisted with a ``promoted`` flag; the promotion cutoff is stored
    so Stage 4 / curation can re-threshold without re-running detection.

    ``primary_window`` selects the apodization window for the primary pass
    (any scipy.signal window name accepted by ``FID.preprocess``, e.g.
    ``"blackmanharris"``, ``"blackman"``, ``"hann"``). It defaults to
    :data:`DEFAULT_PRIMARY_WINDOW` -- a strong window chosen to suppress
    truncation sidelobes; weaker windows leave sidelobe contamination in the
    strong-line list that seeds the gap-pass mask. It affects only which
    positions the primary pass finds, never any reported amplitude or SNR.

    Parameters left as ``None`` fall back to documented defaults. Returns the
    full peak list (user grid) plus diagnostics; also writes ``/stage3_peaks``
    and marks the stage done.
    """
    promotion_v: float = DEFAULT_MIN_SNR if min_snr is None else float(min_snr)
    internal_min_snr: float = min(DEFAULT_INTERNAL_MIN_SNR, promotion_v)
    weak_medium_v: float = (
        DEFAULT_WEAK_MEDIUM_SNR if weak_medium_snr is None else float(weak_medium_snr)
    )
    medium_strong_v: float = (
        DEFAULT_MEDIUM_STRONG_SNR
        if medium_strong_snr is None
        else float(medium_strong_snr)
    )
    sg_window_v: int = 11 if sg_window is None else int(sg_window)
    sg_order_v: int = 3 if sg_order is None else int(sg_order)
    primary_window_v: str = (
        DEFAULT_PRIMARY_WINDOW if primary_window is None else str(primary_window)
    )
    min_excl_v: float = 0.0 if min_exclusion_mhz is None else float(min_exclusion_mhz)
    run_gap_v: bool = True if run_gap_pass is None else bool(run_gap_pass)
    tau_v: Optional[float] = tau_us

    params: Dict[str, Any] = {
        "promotion_min_snr": promotion_v,
        "internal_min_snr": internal_min_snr,
        "weak_medium_snr": weak_medium_v,
        "medium_strong_snr": medium_strong_v,
        "sg_window": sg_window_v,
        "sg_order": sg_order_v,
        "primary_window": primary_window_v,
        "tau_us": tau_v,
        "min_exclusion_mhz": min_excl_v,
        "run_gap_pass": run_gap_v,
        "detection_zpf": _DETECTION_ZPF,
        "settings_source": "stage1_canonical",
    }

    with h5py.File(file_path, "r") as h5f:
        if (
            "processing_parameters" not in h5f
            or "ft_processing" not in h5f["processing_parameters"]
        ):
            raise ValueError(
                "Stage 1 (FT computation) must be completed before peak "
                "detection. Run compute_ft()/compute-ft first."
            )
        if "stage2_noise_result" not in h5f:
            raise ValueError(
                "Stage 2 (noise estimation) must be completed before peak "
                "detection. Run estimate_noise()/estimate-noise first."
            )

    # The user's persisted spectrum is authoritative for reported results.
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]
    trim_range = stage1.get("trim_range")
    user_rms = _load_canonical_noise(file_path, user_ft)

    fid = load_fid_from_pipeline_impl(file_path)
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # Internal detection grids (native zpf=1), identical physical axis. The
    # primary pass applies a strong window (default Blackman-Harris) for
    # sidelobe suppression; the gap pass is unapodized (full resolution).
    primary_ft = _spectrum_from_fid(
        fid, base_pp, trim_range, expf_us=None, window_function=primary_window_v
    )
    gap_ft = _spectrum_from_fid(fid, base_pp, trim_range, expf_us=None)
    primary_noise = estimate_noise_adaptive(
        primary_ft.freq_array, primary_ft.magnitude_spectrum
    )
    gap_noise = estimate_noise_adaptive(
        gap_ft.freq_array, gap_ft.magnitude_spectrum
    )

    internal_peaks: List[Peak] = detect_peaks(
        primary_ft.freq_array,
        primary_ft.magnitude_spectrum,
        primary_noise.rms_noise,
        gap_ft.freq_array,
        gap_ft.magnitude_spectrum,
        gap_noise.rms_noise,
        min_snr=internal_min_snr,
        weak_medium_snr=weak_medium_v,
        medium_strong_snr=medium_strong_v,
        sg_window=sg_window_v,
        sg_order=sg_order_v,
        acquisition_us=acquisition_us,
        tau_us=tau_v,
        min_exclusion_mhz=min_excl_v,
        run_gap_pass=run_gap_v,
    )

    # Snap onto the user grid: physical frequency + re-measured amplitude/SNR.
    peaks = _snap_to_user_grid(
        internal_peaks,
        user_ft,
        user_rms,
        weak_medium_v,
        medium_strong_v,
        promotion_v,
    )

    full_params = {**params, "acquisition_us": acquisition_us}
    save_peaks_impl(file_path, peaks, parameters=full_params)
    save_peak_parameters_impl(file_path, full_params)
    _update_stage_completion(file_path, "stage3_peaks")
    # Re-detection supersedes any Stage 4 window plan built on the old peaks.
    invalidate_downstream_stages(file_path, "stage3_peaks")
    n_promoted = sum(1 for p in peaks if p.properties.get("promoted"))
    logger.info(
        "Stage 3: detected %d peaks (user grid); %d promoted at SNR>=%.3g",
        len(peaks),
        n_promoted,
        promotion_v,
    )

    n_primary = sum(
        1 for p in peaks if p.properties.get("detection_pass") == "primary"
    )
    return {
        "status": "success",
        "peaks": peaks,
        "n_peaks": len(peaks),
        "n_promoted": n_promoted,
        "promotion_min_snr": promotion_v,
        "internal_min_snr": internal_min_snr,
        "n_primary": n_primary,
        "n_gap": len(peaks) - n_primary,
        "parameters_used": params,
        "acquisition_us": acquisition_us,
        "user_ft": user_ft,
        "user_rms": user_rms,
        "primary_ft": primary_ft,
        "gap_ft": gap_ft,
        "primary_noise": primary_noise,
        "gap_noise": gap_noise,
    }


def save_peaks_impl(
    file_path: str,
    peaks: List[Peak],
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a peak list to ``/stage3_peaks`` (overwriting any existing)."""
    with h5py.File(file_path, "a") as h5f:
        if "stage3_peaks" in h5f:
            del h5f["stage3_peaks"]
        grp = h5f.create_group("stage3_peaks")
        save_peaks_to_hdf5(peaks, grp, parameters=parameters)
    logger.info("Saved %d peaks to %s", len(peaks), file_path)


def load_peaks_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted Stage 3 peak list (validates structure loudly)."""
    with h5py.File(file_path, "r") as h5f:
        if "stage3_peaks" not in h5f:
            raise ValueError(
                "No Stage 3 peak results found. Run detect_peaks()/"
                "detect-peaks first."
            )
        grp = h5f["stage3_peaks"]
        peaks = load_peaks_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
        promo_attr = grp.attrs.get("promotion_min_snr")
        parameters: Dict[str, Any] = {}
        if "parameters" in grp.attrs:
            try:
                parameters = json.loads(grp.attrs["parameters"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse saved Stage 3 parameters")
    promotion_min_snr: Optional[float] = None
    if promo_attr is not None and not np.isnan(float(promo_attr)):
        promotion_min_snr = float(promo_attr)
    n_promoted = sum(1 for p in peaks if p.properties.get("promoted"))
    return {
        "peaks": peaks,
        "n_peaks": len(peaks),
        "n_promoted": n_promoted,
        "promotion_min_snr": promotion_min_snr,
        "creation_time": creation_time,
        "parameters_used": parameters,
    }


def visualize_peaks_impl(
    file_path: str,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    y_max_factor: Optional[float] = None,
    backend: str = "matplotlib",
    interactive: bool = True,
    show_snr_histogram: bool = False,
) -> Any:
    """Overlay the persisted classified peaks on the user's spectrum.

    Peaks are stored on the user grid (frequency + re-measured amplitude), so
    the overlay is the user's persisted spectrum with the canonical Stage 2
    noise -- exactly the surface the peaks were scored on. Requires Stage 3
    completed. With ``show_snr_histogram`` a second panel shows the user-grid
    SNR distribution with the promotion cutoff marked (curation view).
    """
    loaded = load_peaks_impl(file_path)
    peaks: List[Peak] = loaded["peaks"]
    promotion_min_snr = loaded.get("promotion_min_snr")

    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    try:
        user_rms = _load_canonical_noise(file_path, user_ft)
    except Exception as e:  # Stage 2 invalidated/missing -> degrade loudly.
        logger.warning(
            "Canonical Stage 2 noise unavailable (%s); estimating on the "
            "user spectrum for display only.",
            e,
        )
        user_rms = estimate_noise_adaptive(
            user_ft.freq_array, user_ft.magnitude_spectrum
        ).rms_noise

    from ..visualization.peak_visualization import plot_peak_detection

    if title is None:
        name = Path(file_path).stem
        fr = (user_ft.freq_array.min(), user_ft.freq_array.max())
        title = (
            f"Pipeline {name} - Stage 3 Peak Detection "
            f"({fr[0]:.0f}-{fr[1]:.0f} MHz, {len(peaks)} peaks)"
        )

    return plot_peak_detection(
        frequencies=user_ft.freq_array,
        magnitudes=user_ft.magnitude_spectrum,
        rms_noise=user_rms,
        peaks=peaks,
        figsize=figsize if figsize is not None else (16, 6),
        title=title,
        y_max_factor=y_max_factor if y_max_factor is not None else 25.0,
        backend=backend,
        snr_histogram=show_snr_histogram,
        promotion_min_snr=promotion_min_snr,
    )


def save_peak_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """Save Stage 3 detection parameters for reuse (JSON under the group)."""
    with h5py.File(file_path, "a") as h5f:
        grp = h5f.require_group("processing_parameters")
        if "peak_detection" in grp:
            del grp["peak_detection"]
        pk = grp.create_group("peak_detection")
        pk.attrs["parameters"] = json.dumps(parameters, default=str)
        pk.attrs["last_updated"] = datetime.now().isoformat()
    logger.info("Saved Stage 3 parameters to %s", file_path)
