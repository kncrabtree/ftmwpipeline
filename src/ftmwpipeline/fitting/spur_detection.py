"""Clock / LO spur detection and masking for the Stage 5 fit.

A clock or local-oscillator spur is a persistent CW tone: a single-bin
(plus sinc-skirt) delta that no finite-T line shape can represent, so it
detonates a window's chi-squared even when the fitter correctly never
places a peak on it. This module detects spurs and produces the per-window
bin mask the fit machinery uses to (a) drop nominated peak candidates that
land on a spur and (b) exclude spur bins from the weighted residual sum.

Two independent discriminators are combined by a **joint gate** whose hard
requirement is an exact integer-MHz center (clock harmonics):

* **frequency-domain narrowness** -- on the active-FT a CW tone is
  transform-limited by the full boxcar (first null ~ one bin), so its peak
  bin towers over its neighbours, whereas a real finite-T molecular line
  has a coherent leakage skirt. This is the zero-false-positive primary
  (it spares real lines that merely sit near an integer MHz).
* **temporal flatness** -- the Stage 2b STFT τ-calibration flags bins
  whose magnitude is flat across the STFT frames (``spur_by_tau``: τ
  rails to ``tau_max``). That flag is persisted per :class:`SpurCluster`
  as :attr:`SpurCluster.saturated`. It catches split-bin spurs the
  narrowness test misses. The *raw* ``cls == 1`` catalogue is **not**
  usable here -- its AICc branch also fires on erratic beat / blend bins
  that are not spurs (measured on 2638: 53 of 59 integer-MHz ``cls == 1``
  bins are erratic real lines); only the saturated subset is trusted.

The gate is therefore ``integer-MHz ∧ (narrow ∨ saturated)``. See
``dev-docs/planning/stage5-spur-masking.md`` and
``dev-docs/research/stage5-gaussian-audit/report.md`` §§ "Spur-detection
prototype", "Flatness-exposure measurement".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, cast

import numpy as np

from .peak_model import sideband_sign
from .tau_calibration import SpurCluster

__all__ = [
    "Spur",
    "GatedSpur",
    "SpurMaskSpec",
    "SpurSet",
    "DEFAULT_INTEGER_TOL_MHZ",
    "DEFAULT_NARROWNESS_RATIO",
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_MASK_HALF_WIDTH_BINS",
    "detect_active_ft_spurs",
    "gate_spurs",
    "build_spur_set",
]

# Detection thresholds (validated on the 2638 fixture; instrument-tunable
# via the Stage 5 ``spur`` settings sub-block).
DEFAULT_INTEGER_TOL_MHZ = 0.04  # ~half a bin (active-FT spacing ~79 kHz)
DEFAULT_NARROWNESS_RATIO = 0.30  # max(neighbour)/peak below this => narrow
DEFAULT_SNR_THRESHOLD = 5.0  # peak-bin magnitude / local sigma_c floor
DEFAULT_MASK_HALF_WIDTH_BINS = 2  # residual-mask half-width in active-FT bins


@dataclass(frozen=True)
class Spur:
    """One integer-MHz, sub-resolution-narrow bin on the active-FT."""

    integer_mhz: int
    center_mhz: float
    bin_index: int
    magnitude: float
    snr: float
    narrowness_ratio: float


@dataclass(frozen=True)
class GatedSpur:
    """A spur that passed the joint gate, with its provenance.

    ``source`` is one of ``"narrow"`` (frequency-domain only),
    ``"saturated"`` (Stage 2b flat-cluster only), or ``"narrow+saturated"``
    (both detectors agree). ``snr`` / ``narrowness_ratio`` are NaN when the
    spur came only from the persisted flat catalogue (no active-FT
    measurement).
    """

    center_mhz: float
    integer_mhz: int
    source: str
    snr: float = float("nan")
    narrowness_ratio: float = float("nan")


@dataclass(frozen=True)
class SpurMaskSpec:
    """Per-window spur mask in the window's baseband-offset frame.

    :func:`~ftmwpipeline.fitting.window_fit.fit_window` rebuilds the boolean
    bin mask from this spec and its own offset grid: a bin is masked when it
    lies within ``half_width_mhz`` of any entry of ``offsets_mhz``. Carrying
    offsets + a half-width (rather than a bin-index mask) makes the spec
    invariant to the internal grid re-sorting the fit routines perform.
    """

    offsets_mhz: Tuple[float, ...]
    half_width_mhz: float

    def bin_mask(self, offset_grid_mhz: np.ndarray) -> np.ndarray:
        """Boolean mask over ``offset_grid_mhz`` (True = masked spur bin)."""
        u = np.asarray(offset_grid_mhz, dtype=float)
        mask = np.zeros(u.shape, dtype=bool)
        if not self.offsets_mhz or self.half_width_mhz <= 0.0:
            return cast(np.ndarray, mask)
        for off in self.offsets_mhz:
            mask |= np.abs(u - off) <= self.half_width_mhz
        return cast(np.ndarray, mask)


@dataclass(frozen=True)
class SpurSet:
    """Gated spur catalogue + mask geometry for one Stage 5 fit.

    Built once per fit by :func:`build_spur_set`; consumed by the plan
    executor to derive each window's :class:`SpurMaskSpec` (residual mask)
    and to drop nominated candidates that land on a spur.

    Attributes
    ----------
    spurs : tuple of GatedSpur
        Gated spurs (molecular center frequencies), sorted by frequency.
    bin_spacing_mhz : float
        Active-FT bin spacing (MHz).
    mask_half_width_bins : int
        Residual-mask half-width in active-FT bins.
    """

    spurs: Tuple[GatedSpur, ...]
    bin_spacing_mhz: float
    mask_half_width_bins: int = DEFAULT_MASK_HALF_WIDTH_BINS

    def __bool__(self) -> bool:
        return len(self.spurs) > 0

    @property
    def centers_mhz(self) -> np.ndarray:
        return cast(
            np.ndarray, np.asarray([s.center_mhz for s in self.spurs], dtype=float)
        )

    @property
    def mask_half_width_mhz(self) -> float:
        # ``(N + 0.5)`` bins masks a spur's center bin +/-N robustly under
        # rounding while never reaching the (N+1)th bin.
        return (self.mask_half_width_bins + 0.5) * self.bin_spacing_mhz

    @property
    def nomination_tol_mhz(self) -> float:
        # Drop a candidate only when it sits essentially *on* the spur core
        # (within one bin), so a real line a couple of bins away survives
        # nomination even though the residual mask still spans +/-N bins.
        return max(self.bin_spacing_mhz, DEFAULT_INTEGER_TOL_MHZ)

    def window_mask_spec(
        self,
        freq_lo_mhz: float,
        freq_hi_mhz: float,
        center_mhz: float,
        sideband: Any,
    ) -> Optional[SpurMaskSpec]:
        """Spur mask for a window spanning ``[freq_lo, freq_hi]`` (molecular).

        Returns ``None`` when no gated spur falls inside the window.
        """
        lo, hi = (freq_lo_mhz, freq_hi_mhz)
        if lo > hi:
            lo, hi = hi, lo
        s = sideband_sign(sideband)
        offsets = [
            float(s * (sp.center_mhz - center_mhz))
            for sp in self.spurs
            if lo <= sp.center_mhz <= hi
        ]
        if not offsets:
            return None
        return SpurMaskSpec(
            offsets_mhz=tuple(offsets),
            half_width_mhz=self.mask_half_width_mhz,
        )

    def excluded_offsets_for_window(
        self,
        freq_lo_mhz: float,
        freq_hi_mhz: float,
        center_mhz: float,
        sideband: Any,
    ) -> List[float]:
        """Spur baseband offsets to reject during peak nomination."""
        spec = self.window_mask_spec(freq_lo_mhz, freq_hi_mhz, center_mhz, sideband)
        return list(spec.offsets_mhz) if spec is not None else []

    def candidate_on_spur(self, freq_mhz: float) -> bool:
        """Whether a candidate at ``freq_mhz`` (molecular) sits on a spur."""
        tol = self.nomination_tol_mhz
        return any(abs(freq_mhz - c) <= tol for c in self.centers_mhz)


# ---------------------------------------------------------------------------
# Frequency-domain detector
# ---------------------------------------------------------------------------
def detect_active_ft_spurs(
    freqs_sorted_mhz: np.ndarray,
    complex_spectrum_sorted: np.ndarray,
    sigma_c_sorted: np.ndarray,
    *,
    band: Tuple[float, float],
    integer_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
    narrowness_ratio: float = DEFAULT_NARROWNESS_RATIO,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> List[Spur]:
    """Flag integer-MHz, sub-resolution-narrow bins on the sorted active-FT.

    Parameters
    ----------
    freqs_sorted_mhz, complex_spectrum_sorted, sigma_c_sorted
        Active-FT molecular frequency grid (ascending), its complex
        spectrum, and the per-bin *complex* noise RMS ``sigma_c`` -- all
        sorted by frequency. ``sigma_c`` is the canonical Stage 2 noise
        (the SNR floor uses ``sigma_c`` directly, matching the prototype).
    band
        ``(lo, hi)`` molecular-frequency analysis range (MHz).
    integer_tol_mhz, narrowness_ratio, snr_threshold
        Gate thresholds (see module-level defaults).
    """
    freqs = np.asarray(freqs_sorted_mhz, dtype=float)
    mag = np.abs(np.asarray(complex_spectrum_sorted))
    sig = np.asarray(sigma_c_sorted, dtype=float)
    lo, hi = band
    spurs: List[Spur] = []
    for f_int in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
        k = int(np.argmin(np.abs(freqs - f_int)))
        if abs(float(freqs[k]) - f_int) > integer_tol_mhz:
            continue
        peak = float(mag[k])
        sigma = float(sig[k]) if sig[k] > 0 else float("nan")
        snr = peak / sigma if sigma > 0 else 0.0
        if snr < snr_threshold:
            continue
        left = float(mag[k - 1]) if k > 0 else 0.0
        right = float(mag[k + 1]) if k < mag.size - 1 else 0.0
        ratio = max(left, right) / peak if peak > 0 else 1.0
        if ratio <= narrowness_ratio:
            spurs.append(
                Spur(
                    integer_mhz=f_int,
                    center_mhz=float(freqs[k]),
                    bin_index=k,
                    magnitude=peak,
                    snr=snr,
                    narrowness_ratio=ratio,
                )
            )
    return spurs


# ---------------------------------------------------------------------------
# Joint gate
# ---------------------------------------------------------------------------
def gate_spurs(
    active_ft_spurs: Sequence[Spur],
    saturated_clusters: Sequence[SpurCluster],
    *,
    integer_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
    merge_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
) -> List[GatedSpur]:
    """Combine the frequency-domain spurs with the saturated flat-cluster set.

    The frequency-domain spurs already satisfy ``integer-MHz ∧ narrow``.
    The flat clusters are filtered to those whose center is integer-MHz
    (``saturated`` was already enforced by the caller). Detections within
    ``merge_tol_mhz`` of each other are merged into one :class:`GatedSpur`
    carrying the combined provenance.
    """
    by_int: dict[int, GatedSpur] = {}

    def _add(center: float, source: str, snr: float, ratio: float) -> None:
        f_int = int(round(center))
        existing = by_int.get(f_int)
        if existing is None:
            by_int[f_int] = GatedSpur(
                center_mhz=center,
                integer_mhz=f_int,
                source=source,
                snr=snr,
                narrowness_ratio=ratio,
            )
            return
        merged_source = existing.source
        if source not in existing.source:
            merged_source = "+".join(sorted(set(existing.source.split("+")) | {source}))
        # Prefer the frequency-domain center / SNR (a measured bin) over the
        # catalogue center when both are present.
        keep_freq = existing
        if source == "narrow":
            keep_freq = GatedSpur(
                center_mhz=center,
                integer_mhz=f_int,
                source=merged_source,
                snr=snr,
                narrowness_ratio=ratio,
            )
        else:
            keep_freq = GatedSpur(
                center_mhz=existing.center_mhz,
                integer_mhz=f_int,
                source=merged_source,
                snr=existing.snr,
                narrowness_ratio=existing.narrowness_ratio,
            )
        by_int[f_int] = keep_freq

    for sp in active_ft_spurs:
        _add(sp.center_mhz, "narrow", sp.snr, sp.narrowness_ratio)
    for cl in saturated_clusters:
        if not cl.saturated:
            continue
        center = float(cl.center_freq_mhz)
        if abs(center - round(center)) > integer_tol_mhz:
            continue
        _add(center, "saturated", float("nan"), float("nan"))

    gated = sorted(by_int.values(), key=lambda g: g.center_mhz)
    return gated


# ---------------------------------------------------------------------------
# Driver: build the SpurSet for a fit
# ---------------------------------------------------------------------------
def build_spur_set(
    freqs_sorted_mhz: np.ndarray,
    complex_spectrum_sorted: np.ndarray,
    sigma_c_sorted: np.ndarray,
    *,
    band: Tuple[float, float],
    saturated_clusters: Sequence[SpurCluster] = (),
    integer_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
    narrowness_ratio: float = DEFAULT_NARROWNESS_RATIO,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    mask_half_width_bins: int = DEFAULT_MASK_HALF_WIDTH_BINS,
    use_stft_catalogue: bool = True,
) -> SpurSet:
    """Detect + gate spurs and package them with the mask geometry.

    ``saturated_clusters`` is the persisted Stage 2b ``spur_clusters``
    catalogue (only entries with ``saturated=True`` contribute). Pass an
    empty sequence (or ``use_stft_catalogue=False``) to run the
    frequency-domain detector alone -- the auto-detect fallback when no
    Stage 2b calibration is present.
    """
    freqs = np.asarray(freqs_sorted_mhz, dtype=float)
    active_spurs = detect_active_ft_spurs(
        freqs,
        complex_spectrum_sorted,
        sigma_c_sorted,
        band=band,
        integer_tol_mhz=integer_tol_mhz,
        narrowness_ratio=narrowness_ratio,
        snr_threshold=snr_threshold,
    )
    clusters = tuple(saturated_clusters) if use_stft_catalogue else ()
    gated = gate_spurs(
        active_spurs,
        clusters,
        integer_tol_mhz=integer_tol_mhz,
    )
    if freqs.size >= 2:
        bin_spacing = float(np.median(np.abs(np.diff(freqs))))
    else:
        bin_spacing = 0.0
    return SpurSet(
        spurs=tuple(gated),
        bin_spacing_mhz=bin_spacing,
        mask_half_width_bins=int(mask_half_width_bins),
    )
