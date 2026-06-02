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
from ..core.peak_detection_settings import (
    PeakDetectionSettings,
    load_preset as load_peak_detection_preset,
    resolve as resolve_peak_detection_settings,
)
from ..preprocessing.edge_coherence import DEFAULT_EDGE_M, rolling_coherence
from ..preprocessing.leakage import deramp_to_active_start
from ..preprocessing.noise_estimation import (
    estimate_noise_scatter,
)
from ..preprocessing.peak_detection import (
    classify_by_snr,
    detect_peaks,
)
from .active_ft_support import build_active_grid_with_noise
from .deprecation import warn_legacy_kwargs
from ..io.noise_result_serialization import load_noise_result_from_hdf5
from ..io.peak_detection_settings_serialization import (
    load_peak_detection_settings_from_h5,
    save_peak_detection_settings_to_h5,
)
from ..io.peak_serialization import (
    load_peaks_from_hdf5,
    save_peaks_to_hdf5,
)
from ..io.stage_fit_settings_serialization import (
    read_stage2b_recommended_shape,
)
from ..file_manager import invalidate_downstream_stages
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl
from .stage2_impl import _update_stage_completion
from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
from .stage2b_g_impl import (
    load_tau_G_calibration_impl,
    tau_G_calibration_present,
)

logger = logging.getLogger(__name__)

# Apex localization runs at the native grid; the user's persisted zpf governs
# only the reported/stored spectrum (snap-back re-measures there).
_DETECTION_ZPF = 1

# Gap-pass detector: matched-filter (exp-apodized active-region FFT) on a
# zero-padded grid that lands the Lorentzian FWHM in SavGol's sweet spot
# (≈ 3 bins). Active-region zpf chosen so FWHM_bins from the Stage 1
# apodization is ≥ ~3 — see dev-docs/research/matched-filter-detection §10
# (revised) and the reassessment script under
# scratch/matched-filter-detection/.
_GAP_ACTIVE_ZPF = 2

# Grid-aware Savitzky-Golay window: cover ~4 line-FWHM in frequency, with
# the SavGol-minimum floor at 5 bins for order-3 polynomial stability. The
# coefficient ≈ 4 reproduces the empirical primary-pass default
# (sg_window=11 at FWHM ≈ 2.67 bins on the zpf=1 internal grid) and gives
# sg_window=13 on the MF gap-pass grid (FWHM ≈ 3.2 bins at active zpf=2).
_SG_FWHM_COVERAGE = 4.0
_SG_MIN_WINDOW = 5

# Primary-pass apodization: a strong window function suppresses truncation
# sidelobes so the primary pass's strong-line list (which seeds the gap-pass
# leakage mask) is clean. Blackman-Harris is the calibrated default -- on the
# 2638 fixture it removes ~5x the sidelobe-suspect detections that the mild
# Stage-1 exponential filter leaves behind. See
# dev-docs/research/peak-detection/report.md sections 3 and 6.
DEFAULT_PRIMARY_WINDOW = "blackmanharris"

# Continuous leakage-aware detection floor (both passes). Neither pass uses a
# hard ``S_coh`` mask: the strong/cluster lines *are* what generate the
# coherence, so a hard cutoff would delete them. Instead each pass raises its
# detection floor continuously by the local coherent-leakage amplitude
# ``k * (S_coh / sqrt(M)) * sigma`` -- a genuine line towers over it, a strong
# line's truncation-skirt ripple (which is that leakage) does not. This keeps
# both passes from re-detecting skirts as weak lines once their internal noise
# floor is the honest (scatter) one. M is the edge-coherence band width.
#
# The two passes need *different* k because they run on opposite-leakage
# spectra. The PRIMARY pass is Blackman-Harris apodized, which annihilates the
# truncation leakage (S_coh ~0.2 across the band, below the noise-only null);
# the window IS the primary's leakage suppression, so the floor is only a
# surgical correction at the rare cluster cores. ``primary k = 1`` is set by
# direct visual validation on the sparse high-SNR fixture 1019 (k=2 there starts
# clipping real cluster lines); catalog recall on 1512/655 is flat across
# k in [1, 4], so it costs no real lines on the dense fixtures either.
#
# The GAP pass is the matched filter, matched to the line shape for weak-line
# sensitivity and so retaining the full leakage (S_coh ~4-15 typical, strong
# wings into the thousands). Here the floor carries ALL the leakage suppression,
# so it needs a larger ``gap k = 3``; at k=1 the floor sits at the wing level
# and the gap pass floods with skirt ripple. ``gap k = 3`` is set by direct
# visual validation on 1512 (the flood collapses and the survivors are genuine
# catalogued / clean-region lines).
#
# This continuous floor replaces the former hard gap-mask ``S_coh`` cutoff,
# which over-killed real lines sitting on strong wings (recall improves). See
# dev-docs/research/stage3-snr-corner/report.md.
PRIMARY_LEAKAGE_FLOOR_K = 1.0
GAP_LEAKAGE_FLOOR_K = 3.0
_LEAKAGE_M = DEFAULT_EDGE_M


def _leakage_floor_amp(
    freq_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    sigma: np.ndarray,
    probe_freq_mhz: float,
    start_us: float,
    k: float,
    band_m: int = _LEAKAGE_M,
) -> np.ndarray:
    """Continuous leakage-aware additive floor ``k * (S_coh / sqrt(M)) * sigma``.

    De-ramps the spectrum to the active-region turn-on
    (:func:`deramp_to_active_start`; ``start_us=0`` is the identity, for the
    active-region-only matched-filter gap FT) before the rolling complex-edge
    coherence, so genuine coherent leakage is exposed. NaN band edges (no full
    M-band centred) contribute no floor. ``k <= 0`` disables it (zeros).
    """
    if k <= 0:
        return np.zeros_like(sigma, dtype=float)
    deramped = deramp_to_active_start(
        freq_mhz, complex_spectrum, probe_freq_mhz, start_us
    )
    scoh = rolling_coherence(deramped, sigma, band_m=band_m)
    return (
        k * (np.nan_to_num(scoh, nan=0.0) / np.sqrt(band_m)) * sigma
    )


def _active_acquisition_us(
    fid_duration_us: float, start_us: Optional[float], end_us: Optional[float]
) -> float:
    """Effective acquisition length T (µs) of the analysed FID window."""
    lo = 0.0 if start_us is None else float(start_us)
    hi = fid_duration_us if end_us is None else float(end_us)
    return max(hi - lo, 0.0)


def _grid_aware_sg_window(
    freq_step_mhz: float,
    fwhm_mhz: float,
    *,
    fwhm_coverage: float = _SG_FWHM_COVERAGE,
    min_window: int = _SG_MIN_WINDOW,
) -> int:
    """Pick sg_window covering ~K line-FWHM, rounded up to odd, floor min_window."""
    if freq_step_mhz <= 0 or fwhm_mhz <= 0:
        return int(min_window)
    target_bins = int(round(fwhm_coverage * fwhm_mhz / freq_step_mhz))
    if target_bins % 2 == 0:
        target_bins += 1
    return max(int(min_window), target_bins)


def _mf_gap_spectrum(
    fid: Any,
    base_pp: Any,
    trim_range: Optional[Tuple[float, float]],
    tau_basis_us: float,
    zpf_active: int = _GAP_ACTIVE_ZPF,
) -> ComplexFT:
    """Matched-filter active-region FFT for the gap-pass detector.

    The FID's [base_pp.start_us, base_pp.end_us] active region is exp-apodized
    at ``tau_basis_us`` (Stage 1's user apodization), zero-padded by
    ``zpf_active`` (default 2) so the FWHM in bins lands in SavGol's
    operating range, then rfft'd. The phase reference is the active-region
    turn-on (t=0 maps to start_us), so callers running coherence statistics
    on this spectrum (e.g., ``leakage_touched_intervals``) must pass
    ``start_us=0.0`` to skip the de-ramp that the full-record-rfft path
    needs. See dev-docs/research/matched-filter-detection/report.md §10
    (revised) for the calibration.
    """
    sample_dt_us = fid.spacing * 1e6
    start_idx = int(round((base_pp.start_us or 0.0) / sample_dt_us))
    if base_pp.end_us is None:
        end_idx = len(fid.data)
    else:
        end_idx = int(round(base_pp.end_us / sample_dt_us))
    end_idx = min(end_idx, len(fid.data))
    n_active = end_idx - start_idx
    if n_active <= 0:
        raise ValueError("active region must have positive length")
    n_padded = n_active * (2 ** int(zpf_active))

    # Exp-apodize the active region, mean-remove (matching Stage 1's rdc),
    # pad with zeros to n_padded, then rfft. compute_active_ft itself runs
    # an rfft on n_active points only and ignores n_padded for the FFT — we
    # need zpf > 0 here so we inline the padded rfft directly.
    active = fid.data[start_idx:end_idx].astype(float, copy=True)
    t_rel = np.arange(n_active) * sample_dt_us
    active *= np.exp(-t_rel / float(tau_basis_us))
    active -= active.mean()
    padded = np.zeros(n_padded, dtype=float)
    padded[:n_active] = active
    spectrum = sample_dt_us * np.fft.rfft(padded)
    f_bb = np.fft.rfftfreq(n_padded, d=sample_dt_us)
    from ..fitting.peak_model import sideband_sign
    s = sideband_sign(fid.sideband)
    freq_mhz = float(fid.probe_freq_mhz) + s * f_bb

    cft = ComplexFT.from_spectrum(spectrum.astype(np.complex128), freq_mhz.astype(float))
    if trim_range is not None:
        cft = cft.trim_to_range(trim_range[0], trim_range[1])
    return cft


def _spectrum_from_fid(
    fid: Any,
    base_pp: Any,
    trim_range: Optional[Tuple[float, float]],
    expf_us: Optional[float],
    window_function: Optional[str] = None,
    *,
    zpf: int = _DETECTION_ZPF,
) -> ComplexFT:
    """Recompute a ComplexFT from the FID at the internal detection grid.

    Defaults to ``zpf=_DETECTION_ZPF`` (native resolution -- best for apex
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
        zpf=int(zpf),
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


def _snap_to_active_grid(
    internal_peaks: List[Peak],
    snap_ft: ComplexFT,
    snap_rms: np.ndarray,
    weak_medium_snr: float,
    medium_strong_snr: float,
    promotion_min_snr: float,
) -> List[Peak]:
    """Re-express internal-grid detections on the canonical active FT.

    For each detection: snap by physical frequency to the nearest active-FT
    grid point, re-measure amplitude on the active FT and SNR against the
    active-FT authority noise, and reclassify. De-duplicates collisions on the
    active grid (keeps the strongest). Internal-grid SNR/frequency are
    preserved under ``properties`` for curation/diagnosis, and a ``promoted``
    flag marks whether the active-grid SNR meets the Stage 4 promotion cutoff.
    All detections are kept (promotion is a downstream gate, not a filter
    here). The active FT -- not the front-zeroed full-record spectrum -- is the
    single grid on which detection results are scored and reported.
    """
    snap_freq = snap_ft.freq_array
    snap_mag = snap_ft.magnitude_spectrum
    order = np.argsort(snap_freq)
    sorted_pairs = (order, snap_freq[order])

    by_idx: Dict[int, Peak] = {}
    for p in internal_peaks:
        ui = _nearest_index(sorted_pairs, p.frequency)
        intensity = float(snap_mag[ui])
        sd = float(snap_rms[ui])
        snr = intensity / sd if sd > 0 else 0.0
        snapped = Peak(
            frequency=float(snap_freq[ui]),
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
        prev = by_idx.get(ui)
        if prev is None or snapped.intensity > prev.intensity:
            by_idx[ui] = snapped

    return sorted(by_idx.values(), key=lambda q: q.frequency)


def _load_canonical_noise(file_path: str, user_ft: ComplexFT) -> np.ndarray:
    """Reconstruct the canonical Stage 2 noise on the user spectrum grid."""
    with h5py.File(file_path, "r") as h5f:
        noise = load_noise_result_from_hdf5(
            h5f["stage2_noise_result"],
            user_ft.freq_array,
            user_ft.magnitude_spectrum,
        )
    return np.asarray(noise.rms_noise, dtype=float)


def _build_explicit_from_kwargs(
    *,
    min_snr: Optional[float],
    weak_medium_snr: Optional[float],
    medium_strong_snr: Optional[float],
    sg_window: Optional[int],
    sg_order: Optional[int],
    primary_window: Optional[str],
    min_exclusion_mhz: Optional[float],
    run_gap_pass: Optional[bool],
) -> PeakDetectionSettings:
    """Bundle the legacy per-knob kwargs into an explicit-layer settings instance.

    The remaining knobs that the resolver covers
    (``internal_min_snr``, ``sg_fwhm_coverage``, ``sg_min_window``,
    ``detection_zpf``, ``gap_active_zpf``, ``primary_leakage_floor_k``,
    ``gap_leakage_floor_k``)
    are not on the public Stage 3 signature; they flow through ``settings=``
    / ``preset=`` only.
    """
    explicit = PeakDetectionSettings()
    explicit.promotion.min_snr = min_snr
    explicit.promotion.weak_medium_snr = weak_medium_snr
    explicit.promotion.medium_strong_snr = medium_strong_snr
    explicit.savgol.sg_window = sg_window
    explicit.savgol.sg_order = sg_order
    explicit.primary_pass.primary_window = primary_window
    explicit.primary_pass.min_exclusion_mhz = min_exclusion_mhz
    explicit.gap_pass.run_gap_pass = run_gap_pass
    return explicit


def _required(value: Any, name: str) -> Any:
    """Coerce a post-resolve field that must be filled (hard default present)."""
    if value is None:
        raise AssertionError(
            f"resolved PeakDetectionSettings.{name} is None; missing hard default"
        )
    return value


def detect_peaks_impl(
    file_path: str,
    min_snr: Optional[float] = None,
    weak_medium_snr: Optional[float] = None,
    medium_strong_snr: Optional[float] = None,
    sg_window: Optional[int] = None,
    sg_order: Optional[int] = None,
    primary_window: Optional[str] = None,
    min_exclusion_mhz: Optional[float] = None,
    run_gap_pass: Optional[bool] = None,
    *,
    settings: Optional[PeakDetectionSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Stage 3 two-pass peak detection and persist the result.

    Requires Stage 1 (canonical FT settings) and Stage 2 (noise) completed.
    Detection operates on the user's persisted spectrum: there is no Stage 3
    ``trim``/``zpf`` -- those come from the Stage 1 canonical record.

    ``min_snr`` is the **promotion cutoff** on the user-grid SNR -- which
    peaks move on to Stage 4 -- not the detection floor. Detection always runs
    aggressively on the internal zpf=1 grids at
    ``min(internal_min_snr, promotion)`` (cheap, and recovers real
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

    ``settings`` / ``preset`` populate the same layer of the four-layer
    resolution chain (``explicit > preset > persisted > recommended``);
    passing both raises ``ValueError``. Knobs beyond the legacy per-knob
    signature -- ``internal_min_snr``, ``sg_fwhm_coverage``, ``sg_min_window``,
    ``detection_zpf``, ``gap_active_zpf``, ``primary_leakage_floor_k``,
    ``gap_leakage_floor_k`` --
    flow through ``settings=`` / ``preset=`` only.

    Parameters left as ``None`` fall back to documented defaults. Returns the
    full peak list (user grid) plus diagnostics; also writes ``/stage3_peaks``
    and marks the stage done.
    """
    warn_legacy_kwargs(
        func_name="detect_peaks",
        legacy_kwargs={
            "min_snr": min_snr,
            "weak_medium_snr": weak_medium_snr,
            "medium_strong_snr": medium_strong_snr,
            "sg_window": sg_window,
            "sg_order": sg_order,
            "primary_window": primary_window,
            "min_exclusion_mhz": min_exclusion_mhz,
            "run_gap_pass": run_gap_pass,
        },
        migration_hint=(
            "use settings=PeakDetectionSettings(...) or preset='name' to "
            "drive Stage 3 from the settings resolver"
        ),
    )

    if preset is not None and settings is not None:
        raise ValueError(
            "'preset' and 'settings' are alternative ways to populate "
            "the preset layer of the peak-detection-settings chain; pass "
            "exactly one (or override individual fields via explicit kwargs)"
        )

    explicit = _build_explicit_from_kwargs(
        min_snr=min_snr,
        weak_medium_snr=weak_medium_snr,
        medium_strong_snr=medium_strong_snr,
        sg_window=sg_window,
        sg_order=sg_order,
        primary_window=primary_window,
        min_exclusion_mhz=min_exclusion_mhz,
        run_gap_pass=run_gap_pass,
    )
    preset_layer: Optional[PeakDetectionSettings] = settings
    preset_name: Optional[str] = None
    if preset is not None:
        preset_layer = load_peak_detection_preset(preset)
        preset_name = str(preset)
    persisted_layer = load_peak_detection_settings_from_h5(file_path)

    resolved = resolve_peak_detection_settings(
        explicit=explicit,
        preset=preset_layer,
        persisted=persisted_layer,
        recommended=None,
    )

    promotion = resolved.promotion
    savgol = resolved.savgol
    primary = resolved.primary_pass
    gap = resolved.gap_pass

    promotion_v: float = float(_required(promotion.min_snr, "promotion.min_snr"))
    internal_floor: float = float(
        _required(promotion.internal_min_snr, "promotion.internal_min_snr")
    )
    internal_min_snr: float = min(internal_floor, promotion_v)
    weak_medium_v: float = float(
        _required(promotion.weak_medium_snr, "promotion.weak_medium_snr")
    )
    medium_strong_v: float = float(
        _required(promotion.medium_strong_snr, "promotion.medium_strong_snr")
    )
    sg_window_v: int = int(_required(savgol.sg_window, "savgol.sg_window"))
    sg_order_v: int = int(_required(savgol.sg_order, "savgol.sg_order"))
    sg_fwhm_coverage_v: float = float(
        _required(savgol.sg_fwhm_coverage, "savgol.sg_fwhm_coverage")
    )
    sg_min_window_v: int = int(
        _required(savgol.sg_min_window, "savgol.sg_min_window")
    )
    primary_window_v: str = str(
        _required(primary.primary_window, "primary_pass.primary_window")
    )
    min_excl_v: float = float(
        _required(primary.min_exclusion_mhz, "primary_pass.min_exclusion_mhz")
    )
    detection_zpf_v: int = int(
        _required(primary.detection_zpf, "primary_pass.detection_zpf")
    )
    primary_leakage_floor_k_v: float = float(
        _required(
            primary.primary_leakage_floor_k,
            "primary_pass.primary_leakage_floor_k",
        )
    )
    run_gap_v: bool = bool(_required(gap.run_gap_pass, "gap_pass.run_gap_pass"))
    gap_active_zpf_v: int = int(
        _required(gap.gap_active_zpf, "gap_pass.gap_active_zpf")
    )
    gap_leakage_floor_k_v: float = float(
        _required(gap.gap_leakage_floor_k, "gap_pass.gap_leakage_floor_k")
    )

    params: Dict[str, Any] = {
        "promotion_min_snr": promotion_v,
        "internal_min_snr": internal_min_snr,
        "weak_medium_snr": weak_medium_v,
        "medium_strong_snr": medium_strong_v,
        "sg_window": sg_window_v,
        "sg_order": sg_order_v,
        "primary_window": primary_window_v,
        "primary_leakage_floor_k": primary_leakage_floor_k_v,
        "gap_leakage_floor_k": gap_leakage_floor_k_v,
        "internal_noise_method": "scatter",
        "min_exclusion_mhz": min_excl_v,
        "run_gap_pass": run_gap_v,
        "detection_zpf": detection_zpf_v,
        "gap_active_zpf": gap_active_zpf_v,
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

    # The canonical active FT is the single grid on which detections are
    # scored and reported. The full-record persisted FT is rebuilt only for
    # its canonical processing params / trim (a display artifact otherwise).
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft: ComplexFT = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]
    trim_range = stage1.get("trim_range")
    snap_ft, snap_rms = build_active_grid_with_noise(file_path, trim_range)

    fid = load_fid_from_pipeline_impl(file_path)
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )

    # Primary detection runs on the leakage-suppressed apodized spectrum
    # (Stage 3 internal zpf=1 grid). The gap pass runs on the matched-filter
    # active-region FFT: exp-apodized at the Stage 1 user apodization,
    # zero-padded so the FWHM lands in SavGol's operating range. The grid-
    # aware sg_window for the gap pass comes from
    # ``_grid_aware_sg_window`` driven by the line FWHM = 1/(π · expf_us).
    primary_ft = _spectrum_from_fid(
        fid,
        base_pp,
        trim_range,
        expf_us=None,
        window_function=primary_window_v,
        zpf=detection_zpf_v,
    )
    # Gap-pass matched-filter tau: shape-aware feeder.
    #
    # Precedence:
    #   1. Stage 2b Gaussian twin ``tau_G_maj`` when ``recommended_shape``
    #      is ``'gaussian'`` and the twin is present -- matches the
    #      envelope the production fit uses on Gaussian-shape data.
    #   2. Stage 2b Lorentzian ``tau_maj`` when available (physical
    #      molecular decay for exp-envelope data; the matched filter's
    #      FWHM equals the true line FWHM).
    #   3. Stage 1 user apodization ``expf_us`` for the pre-calibration
    #      path.
    #   4. Historical 5.0 µs default.
    #
    # The matched filter itself remains a pure-exp apodization in all
    # cases; substituting tau_G into an exp filter is a mismatched-filter
    # variant whose SNR cost is modest on the 2638 fixture (see
    # dev-docs/research/stage3-gaussian-audit/ -- the apodization-shape
    # question is tracked separately from the time-constant choice).
    recommended_shape = read_stage2b_recommended_shape(file_path)
    if (
        recommended_shape == "gaussian"
        and tau_G_calibration_present(file_path)
    ):
        tau_basis_us = float(
            load_tau_G_calibration_impl(file_path)["tau_G_calibration"].tau_maj_us
        )
    elif tau_calibration_present(file_path):
        tau_basis_us = float(
            load_tau_calibration_impl(file_path)["tau_calibration"].tau_maj_us
        )
    elif base_pp.expf_us:
        tau_basis_us = float(base_pp.expf_us)
    else:
        tau_basis_us = 5.0
    gap_ft = _mf_gap_spectrum(
        fid,
        base_pp,
        trim_range,
        tau_basis_us=tau_basis_us,
        zpf_active=gap_active_zpf_v,
    )
    # Record the resolved gap-pass τ in the diagnostics dict so callers
    # (and persisted ``/stage3_peaks`` consumers) can see which τ branch
    # of the shape-aware feeder fired.
    params["tau_basis_us"] = float(tau_basis_us)
    params["tau_basis_source"] = (
        "stage2b_tau_G_maj"
        if recommended_shape == "gaussian" and tau_G_calibration_present(file_path)
        else "stage2b_tau_maj"
        if tau_calibration_present(file_path)
        else "stage1_expf_us"
        if base_pp.expf_us
        else "default_5us"
    )
    line_fwhm_mhz = 1.0 / (np.pi * tau_basis_us)
    gap_freq_step = abs(gap_ft.freq_array[1] - gap_ft.freq_array[0])
    gap_sg_window_v = _grid_aware_sg_window(
        gap_freq_step,
        line_fwhm_mhz,
        fwhm_coverage=sg_fwhm_coverage_v,
        min_window=sg_min_window_v,
    )
    # Internal detection noise is the same honest scatter estimator the
    # persisted Stage 2 uses (consistency with the snap-back SNR scale). The
    # scatter floor is lower than the legacy adaptive one in leakage-pedestal
    # regions; the primary pass is kept safe by the leakage-aware floor below
    # rather than by adaptive's incidental pedestal inflation.
    primary_noise = estimate_noise_scatter(
        primary_ft.freq_array, primary_ft.magnitude_spectrum
    )
    gap_noise = estimate_noise_scatter(
        gap_ft.freq_array, gap_ft.magnitude_spectrum
    )

    # Continuous leakage-aware floors for both passes (k * S_coh/sqrt(M) * sigma,
    # the local coherent-leakage amplitude). The primary FT is a full-record
    # rfft, so it is de-ramped to the active-region turn-on (start_us) before
    # the coherence statistic; the matched-filter gap FT is active-region-only,
    # so its phase reference is already the turn-on (start_us=0.0, identity).
    primary_leakage_amp = _leakage_floor_amp(
        primary_ft.freq_array,
        primary_ft.complex_spectrum,
        primary_noise.rms_noise,
        fid.probe_freq_mhz,
        base_pp.start_us or 0.0,
        primary_leakage_floor_k_v,
    )
    gap_leakage_amp = _leakage_floor_amp(
        gap_ft.freq_array,
        gap_ft.complex_spectrum,
        gap_noise.rms_noise,
        fid.probe_freq_mhz,
        0.0,
        gap_leakage_floor_k_v,
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
        gap_sg_window=gap_sg_window_v,
        sg_order=sg_order_v,
        primary_leakage_amp=primary_leakage_amp,
        gap_leakage_amp=gap_leakage_amp,
        min_exclusion_mhz=min_excl_v,
        run_gap_pass=run_gap_v,
    )

    # Snap onto the active grid: physical frequency + re-measured amplitude/SNR
    # against the active-FT authority noise.
    peaks = _snap_to_active_grid(
        internal_peaks,
        snap_ft,
        snap_rms,
        weak_medium_v,
        medium_strong_v,
        promotion_v,
    )

    full_params = {**params, "acquisition_us": acquisition_us}
    save_peaks_impl(file_path, peaks, parameters=full_params)
    save_peak_parameters_impl(file_path, full_params)
    # Persist the resolved PeakDetectionSettings to
    # ``processing_parameters/stage3_peaks``. The legacy JSON-encoded
    # ``processing_parameters/peak_detection`` block is kept by
    # ``save_peak_parameters_impl`` above as a back-compat shim; the new
    # canonical record below is what the resolver's persisted layer reads.
    save_peak_detection_settings_to_h5(
        file_path, resolved, preset_name=preset_name,
    )
    _update_stage_completion(file_path, "stage3_peaks")
    # Re-detection supersedes any Stage 4 window plan built on the old peaks.
    invalidate_downstream_stages(file_path, "stage3_peaks")
    n_promoted = sum(1 for p in peaks if p.properties.get("promoted"))
    logger.info(
        "Stage 3: detected %d peaks (active grid); %d promoted at SNR>=%.3g",
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
        "active_ft": snap_ft,
        "active_rms": snap_rms,
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
    """Overlay the persisted classified peaks on the canonical active FT.

    Peaks are scored and stored on the active FT (frequency + re-measured
    amplitude), so the overlay is the active FT with the active-FT authority
    noise -- exactly the surface the peaks were scored on. Requires Stage 3
    completed. With ``show_snr_histogram`` a second panel shows the active-grid
    SNR distribution with the promotion cutoff marked (curation view).
    """
    loaded = load_peaks_impl(file_path)
    peaks: List[Peak] = loaded["peaks"]
    promotion_min_snr = loaded.get("promotion_min_snr")

    stage1 = compute_ft_impl(file_path=file_path)
    trim_range = stage1.get("trim_range")
    active_ft, active_rms = build_active_grid_with_noise(file_path, trim_range)

    from ..visualization.peak_visualization import plot_peak_detection

    if title is None:
        name = Path(file_path).stem
        fr = (active_ft.freq_array.min(), active_ft.freq_array.max())
        title = (
            f"Pipeline {name} - Stage 3 Peak Detection "
            f"({fr[0]:.0f}-{fr[1]:.0f} MHz, {len(peaks)} peaks)"
        )

    return plot_peak_detection(
        frequencies=active_ft.freq_array,
        magnitudes=active_ft.magnitude_spectrum,
        rms_noise=active_rms,
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
