"""
Shared implementation for Stage 3: Peak detection.

Orchestration only -- the detection algorithm lives in
``ftmwpipeline.preprocessing.peak_detection``. This module recomputes the two
spectra the two-pass detector needs from the FID + Stage 1 parameters
(an *apodized* primary spectrum and an *unapodized* full-resolution gap
spectrum), estimates per-point noise on each grid (they differ, so noise must
be estimated on the same grid it is detected on -- not reused from the Stage 2
result, which spans the untrimmed spectrum), runs the detector, and persists
the classified peak list. Wrapped identically by the CLI, Pipeline class, and
functional API.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import h5py

from ..core.data_structures import ComplexFT, Peak
from ..preprocessing.noise_estimation import estimate_noise_adaptive
from ..preprocessing.peak_detection import (
    DEFAULT_MEDIUM_STRONG_SNR,
    DEFAULT_MIN_SNR,
    DEFAULT_WEAK_MEDIUM_SNR,
    detect_peaks,
)
from ..io.peak_serialization import (
    load_peaks_from_hdf5,
    save_peaks_to_hdf5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion

logger = logging.getLogger(__name__)


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
    zpf: Optional[int] = None,
) -> ComplexFT:
    """Recompute a ComplexFT from the FID at a chosen apodization.

    ``expf_us=None`` (and no window function) yields the unapodized,
    full-resolution boxcar spectrum used by the gap pass; a finite ``expf_us``
    yields the leakage-suppressed primary spectrum. ``zpf`` overrides the
    zero-padding factor (None -> the saved/recommended ``base_pp.zpf``); the
    same value is used for both spectra so they share a consistent grid.
    All other preprocessing (window bounds, units, trim) is held identical.
    """
    preprocessed = fid.preprocess(
        start_us=base_pp.start_us,
        end_us=base_pp.end_us,
        zpf=base_pp.zpf if zpf is None else zpf,
        expf_us=expf_us,
        window_function=None if expf_us is None else base_pp.winf,
        rdc=base_pp.rdc,
        units_power=base_pp.units_power,
    )
    spectrum, freqs = preprocessed.compute_fft()
    cft = ComplexFT.from_spectrum(spectrum, freqs)
    if trim_range is not None:
        cft = cft.trim_to_range(trim_range[0], trim_range[1])
    return cft


def _saved_stage3_params(file_path: str) -> Dict[str, Any]:
    """Best-effort read of previously persisted Stage 3 parameters."""
    try:
        with h5py.File(file_path, "r") as h5f:
            pk = "processing_parameters/peak_detection"
            if pk in h5f and "parameters" in h5f[pk].attrs:
                return cast(Dict[str, Any], json.loads(h5f[pk].attrs["parameters"]))
    except Exception as e:  # pragma: no cover - best-effort reuse
        logger.warning("Could not read saved Stage 3 parameters: %s", e)
    return {}


def _resolve_trim(
    explicit: Optional[Tuple[float, float]],
    saved: Dict[str, Any],
    stage1_trim: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Trim precedence: explicit arg > saved Stage 3 trim > Stage 1 trim.

    Stage 1 is intentionally lightweight and does not persist the FT trim (or
    ``zpf``), so detection would otherwise run on the full untrimmed,
    recommended-zpf spectrum. Stage 3 therefore owns its own trim and zpf.
    """
    if explicit is not None:
        return explicit
    tr = saved.get("trim")
    if tr and tr[0] is not None and tr[1] is not None:
        return (float(tr[0]), float(tr[1]))
    return stage1_trim


def _resolve_zpf(explicit: Optional[int], saved: Dict[str, Any], base_zpf: int) -> int:
    """zpf precedence: explicit arg > saved Stage 3 zpf > Stage 1/recommended."""
    if explicit is not None:
        return int(explicit)
    sz = saved.get("zpf")
    if sz is not None:
        return int(sz)
    return int(base_zpf)


def detect_peaks_impl(
    file_path: str,
    min_snr: Optional[float] = None,
    weak_medium_snr: Optional[float] = None,
    medium_strong_snr: Optional[float] = None,
    sg_window: Optional[int] = None,
    sg_order: Optional[int] = None,
    apodization_us: Optional[float] = None,
    tau_us: Optional[float] = None,
    min_exclusion_mhz: Optional[float] = None,
    run_gap_pass: Optional[bool] = None,
    trim: Optional[Tuple[float, float]] = None,
    zpf: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Stage 3 two-pass peak detection and persist the result.

    Requires Stage 1 (FT params) and Stage 2 (noise) completed. Parameters
    left as ``None`` fall back to documented defaults. Returns the peak list
    and diagnostics; also writes ``/stage3_peaks`` and marks the stage done.
    """
    min_snr_v: float = DEFAULT_MIN_SNR if min_snr is None else float(min_snr)
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
    min_excl_v: float = 0.0 if min_exclusion_mhz is None else float(min_exclusion_mhz)
    run_gap_v: bool = True if run_gap_pass is None else bool(run_gap_pass)
    tau_v: Optional[float] = tau_us

    params: Dict[str, Any] = {
        "min_snr": min_snr_v,
        "weak_medium_snr": weak_medium_v,
        "medium_strong_snr": medium_strong_v,
        "sg_window": sg_window_v,
        "sg_order": sg_order_v,
        "apodization_us": apodization_us,
        "tau_us": tau_v,
        "min_exclusion_mhz": min_excl_v,
        "run_gap_pass": run_gap_v,
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

    # Saved Stage 1 FT settings drive both recomputed spectra.
    stage1 = compute_ft_impl(file_path=file_path)
    base_pp = stage1["complex_ft"].metadata["processing_params"]
    saved_s3 = _saved_stage3_params(file_path)
    trim_range = _resolve_trim(trim, saved_s3, stage1.get("trim_range"))
    zpf_v = _resolve_zpf(zpf, saved_s3, base_pp.zpf)
    params["trim"] = list(trim_range) if trim_range is not None else None
    params["zpf"] = zpf_v
    if trim_range is None:
        logger.warning(
            "No trim set for Stage 3; detecting on the full untrimmed "
            "spectrum (DC edges may dominate). Pass trim=(min,max)."
        )

    fid = load_fid_from_pipeline_impl(file_path)
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # Primary apodization: caller override, else saved Stage 1 expf, else 5 us
    # (a finite value is required so the primary differs from the gap pass).
    primary_apod = apodization_us
    if primary_apod is None:
        primary_apod = base_pp.expf_us if base_pp.expf_us else 5.0

    # Apodized primary + unapodized full-resolution gap, on an identical grid.
    primary_ft = _spectrum_from_fid(
        fid, base_pp, trim_range, expf_us=primary_apod, zpf=zpf_v
    )
    gap_ft = _spectrum_from_fid(fid, base_pp, trim_range, expf_us=None, zpf=zpf_v)

    # Per-point noise on each grid (must match the grid it is detected on).
    primary_noise = estimate_noise_adaptive(
        primary_ft.freq_array, primary_ft.magnitude_spectrum
    )
    gap_noise = estimate_noise_adaptive(gap_ft.freq_array, gap_ft.magnitude_spectrum)

    peaks: List[Peak] = detect_peaks(
        primary_ft.freq_array,
        primary_ft.magnitude_spectrum,
        primary_noise.rms_noise,
        gap_ft.freq_array,
        gap_ft.magnitude_spectrum,
        gap_noise.rms_noise,
        min_snr=min_snr_v,
        weak_medium_snr=weak_medium_v,
        medium_strong_snr=medium_strong_v,
        sg_window=sg_window_v,
        sg_order=sg_order_v,
        acquisition_us=acquisition_us,
        tau_us=tau_v,
        min_exclusion_mhz=min_excl_v,
        run_gap_pass=run_gap_v,
    )

    full_params = {**params, "acquisition_us": acquisition_us}
    save_peaks_impl(file_path, peaks, parameters=full_params)
    # Persist parameters (incl. resolved trim) for reuse by visualize/reruns.
    save_peak_parameters_impl(file_path, full_params)
    _update_stage_completion(file_path, "stage3_peaks")
    logger.info("Stage 3: detected %d peaks", len(peaks))

    n_primary = sum(1 for p in peaks if p.properties.get("detection_pass") == "primary")
    return {
        "status": "success",
        "peaks": peaks,
        "n_peaks": len(peaks),
        "n_primary": n_primary,
        "n_gap": len(peaks) - n_primary,
        "parameters_used": params,
        "acquisition_us": acquisition_us,
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
        parameters: Dict[str, Any] = {}
        if "parameters" in grp.attrs:
            try:
                parameters = json.loads(grp.attrs["parameters"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse saved Stage 3 parameters")
    return {
        "peaks": peaks,
        "n_peaks": len(peaks),
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
) -> Any:
    """Overlay the persisted classified peaks on the gap (full-res) spectrum.

    Recomputes the unapodized spectrum + its per-point noise (same grid Stage 3
    detected the gap pass on) for context, then delegates to the spectrum
    visualization. Requires Stage 3 to be completed.
    """
    loaded = load_peaks_impl(file_path)
    peaks: List[Peak] = loaded["peaks"]
    used = loaded["parameters_used"]
    saved_trim = used.get("trim")
    explicit_trim = (float(saved_trim[0]), float(saved_trim[1])) if saved_trim else None

    stage1 = compute_ft_impl(file_path=file_path)
    base_pp = stage1["complex_ft"].metadata["processing_params"]
    saved_s3 = _saved_stage3_params(file_path)
    trim_range = _resolve_trim(explicit_trim, saved_s3, stage1.get("trim_range"))
    # Display the same grid detection used (same zpf).
    zpf_v = _resolve_zpf(used.get("zpf"), saved_s3, base_pp.zpf)
    fid = load_fid_from_pipeline_impl(file_path)
    gap_ft = _spectrum_from_fid(fid, base_pp, trim_range, expf_us=None, zpf=zpf_v)
    gap_noise = estimate_noise_adaptive(gap_ft.freq_array, gap_ft.magnitude_spectrum)

    from ..visualization.peak_visualization import plot_peak_detection

    if title is None:
        name = Path(file_path).stem
        fr = (gap_ft.freq_array.min(), gap_ft.freq_array.max())
        title = (
            f"Pipeline {name} - Stage 3 Peak Detection "
            f"({fr[0]:.0f}-{fr[1]:.0f} MHz, {len(peaks)} peaks)"
        )

    return plot_peak_detection(
        frequencies=gap_ft.freq_array,
        magnitudes=gap_ft.magnitude_spectrum,
        rms_noise=gap_noise.rms_noise,
        peaks=peaks,
        figsize=figsize if figsize is not None else (16, 6),
        title=title,
        y_max_factor=y_max_factor if y_max_factor is not None else 25.0,
        backend=backend,
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
