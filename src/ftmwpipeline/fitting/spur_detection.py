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

The nomination gate is ``integer-MHz ∧ (narrow ∨ saturated)``, plus a
**pair lane** for tones that fall *between* two grid bins: the split power
defeats the single-bin narrowness test, so the two-bin pair is tested
against its second neighbours instead -- but that signature is ambiguous
against a blended doublet, so pair nominees gate only with time-domain
flatness confirmation. When the raw FID is available, every verdict is
additionally **arbitrated by a direct time-domain decay probe**
(:func:`make_decay_probe`): the FID is demodulated
at the candidate's baseband frequency and block-averaged into frames, and
the late/early amplitude ratio separates a decaying molecular line
(ratio ~ ``exp(-dT/tau)`` << 1) from a flat CW tone (ratio ~ 1). The probe
fixes both error directions the frequency-domain gate alone carries:

* a real line whose center happens to fall within ``integer_tol`` of an
  integer MHz **and** whose ``tau`` is long enough that its on-grid profile
  passes the narrowness test (measured: a bin-centered ``tau ~ 0.7 T`` line
  has a neighbour ratio ~ 0.23) is *vetoed* out of the gate when the probe
  sees it decay;
* a genuinely flat tone at a NON-integer frequency (LO/IF intermodulation
  rather than a clock harmonic) nominated by the Stage 2b cluster
  catalogue is gated when the probe confirms flatness -- the ``saturated``
  flag alone is not trusted in either direction (measured false positives
  on decaying lines and false negatives on real tones).

See ``dev-docs/planning/stage5-spur-masking.md`` and
``dev-docs/research/stage5-gaussian-audit/report.md`` §§ "Spur-detection
prototype", "Flatness-exposure measurement".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

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
    "DEFAULT_DECAY_RATIO_LINE",
    "DEFAULT_DECAY_RATIO_FLAT",
    "DEFAULT_DECAY_MIN_SNR_VETO",
    "DEFAULT_DECAY_MIN_SNR_FLAT",
    "DEFAULT_DECAY_N_FRAMES",
    "detect_active_ft_spurs",
    "make_decay_probe",
    "gate_spurs",
    "build_spur_set",
]

# Detection thresholds (validated on the 2638 fixture; instrument-tunable
# via the Stage 5 ``spur`` settings sub-block).
DEFAULT_INTEGER_TOL_MHZ = 0.04  # ~half a bin (active-FT spacing ~79 kHz)
DEFAULT_NARROWNESS_RATIO = 0.30  # max(neighbour)/peak below this => narrow
DEFAULT_SNR_THRESHOLD = 5.0  # peak-bin magnitude / local sigma_c floor
DEFAULT_MASK_HALF_WIDTH_BINS = 2  # residual-mask half-width in active-FT bins

# Decay-probe arbitration thresholds (cross-fixture calibration, 7 fixtures:
# every true clock-comb spur measured ratio >= 0.70; every decaying real
# line at usable probe SNR measured <= 0.65, with the contested cases far
# from the bars -- the falsely-narrow-gated SNR-1410 line at 0.29 and the
# flat 28057.46 / 29985.35 tones at 0.97-1.20).
DEFAULT_DECAY_RATIO_LINE = 0.6  # below this (and SNR ok) a nominee decays
DEFAULT_DECAY_RATIO_FLAT = 0.8  # above this (and SNR ok) a nominee is flat
DEFAULT_DECAY_MIN_SNR_VETO = 3.0  # min frame-amplitude SNR to veto a narrow spur
DEFAULT_DECAY_MIN_SNR_FLAT = 10.0  # min frame-amplitude SNR to gate a flat tone
DEFAULT_DECAY_N_FRAMES = 8  # frames over the active record


@dataclass(frozen=True)
class Spur:
    """One integer-MHz, sub-resolution-narrow bin on the active-FT.

    ``pair`` marks a *two-bin* nominee: a CW tone whose frequency falls
    between two grid bins splits its power across them (each reads ~0.6-1.0
    of the other), so the single-bin neighbour test fails even though the
    pair together is transform-limited (second neighbours fall back to the
    sinc skirt). Pair nominees are a weaker frequency-domain signature --
    a blended doublet can mimic them -- so the joint gate only accepts them
    with explicit time-domain flatness confirmation (never on the
    frequency-domain evidence alone).
    """

    integer_mhz: int
    center_mhz: float
    bin_index: int
    magnitude: float
    snr: float
    narrowness_ratio: float
    pair: bool = False


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
            continue
        # Pair lane: a tone between two bins splits its power, so neither
        # bin passes the single-bin test. Treat the integer bin plus its
        # stronger neighbour as the tone and test the bins flanking the
        # pair instead. This signature alone is NOT gate-worthy (a blended
        # doublet looks identical); the joint gate requires time-domain
        # flatness confirmation for ``pair`` nominees.
        j = k - 1 if left >= right else k + 1
        if j < 0 or j >= mag.size:
            continue
        pair_peak = max(peak, float(mag[j]))
        lo_i, hi_i = (k, j) if j > k else (j, k)
        second_left = float(mag[lo_i - 1]) if lo_i > 0 else 0.0
        second_right = float(mag[hi_i + 1]) if hi_i < mag.size - 1 else 0.0
        pair_ratio = (
            max(second_left, second_right) / pair_peak if pair_peak > 0 else 1.0
        )
        if pair_ratio <= narrowness_ratio:
            k_top = k if peak >= float(mag[j]) else j
            sigma_top = float(sig[k_top]) if sig[k_top] > 0 else float("nan")
            spurs.append(
                Spur(
                    integer_mhz=f_int,
                    center_mhz=float(freqs[k_top]),
                    bin_index=k_top,
                    magnitude=pair_peak,
                    snr=pair_peak / sigma_top if sigma_top > 0 else 0.0,
                    narrowness_ratio=pair_ratio,
                    pair=True,
                )
            )
    return spurs


# ---------------------------------------------------------------------------
# Time-domain decay probe
# ---------------------------------------------------------------------------
def make_decay_probe(
    fid_data: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Any,
    n_frames: int = DEFAULT_DECAY_N_FRAMES,
    n_noise_probes: int = 16,
) -> Callable[[float], Tuple[float, float]]:
    """Build a per-frequency FID decay probe ``f_mol -> (ratio, amp_snr)``.

    Demodulates the real FID's active slice ``[start_us, end_us)`` at the
    candidate's baseband frequency, block-averages the complex demod into
    ``n_frames`` frames, and returns

    * ``ratio`` -- median frame amplitude of the last third over the first
      third. A molecular line decays (``~ exp(-dT/tau)``); a CW tone is ~ 1.
    * ``amp_snr`` -- the first-third amplitude over the noise floor of a
      frame-mean amplitude, measured once from the median frame amplitude
      over ``n_noise_probes`` quasi-random in-band frequencies (lines are
      sparse, so the median lands on empty bins).

    The probe is the time-domain arbiter for the spur gate
    (:func:`gate_spurs`): it tests the candidate against the physics that
    defines a spur -- a tone that does not decay -- instead of frequency-
    domain proxies for it.
    """
    x = np.asarray(fid_data, dtype=float)
    i0 = max(int(round(start_us / sample_dt_us)), 0)
    i1 = min(int(round(end_us / sample_dt_us)), x.size)
    seg = x[i0:i1]
    t_us = np.arange(i0, i1) * sample_dt_us
    s = sideband_sign(sideband)
    third = max(n_frames // 3, 1)

    def _amps(f_bb_mhz: float) -> np.ndarray:
        y = seg * np.exp(-2j * np.pi * f_bb_mhz * t_us)
        return cast(
            np.ndarray,
            np.asarray(
                [np.abs(fr.mean()) for fr in np.array_split(y, n_frames)], dtype=float
            ),
        )

    def _to_bb(f_mol: float) -> float:
        # ``f_mol = probe + s_bb * f_bb`` with ``s_bb = -s`` of the
        # molecular-offset sign convention: lower sideband means
        # ``f_bb = probe - f_mol``.
        return float(abs(probe_freq_mhz - f_mol) if s < 0 else f_mol - probe_freq_mhz)

    # Frame-amplitude noise floor: median frame amplitude over quasi-random
    # baseband offsets across the Nyquist band (avoiding DC). Lines are
    # sparse, so the median of medians lands on empty bins.
    rng = np.random.default_rng(20260610)
    nyquist_mhz = 0.5 / sample_dt_us
    probes = rng.uniform(0.05 * nyquist_mhz, 0.95 * nyquist_mhz, size=n_noise_probes)
    noise_amp = float(np.median([np.median(_amps(f)) for f in probes]))

    def probe(f_mol_mhz: float) -> Tuple[float, float]:
        amps = _amps(_to_bb(float(f_mol_mhz)))
        early = float(np.median(amps[:third]))
        late = float(np.median(amps[-third:]))
        ratio = late / early if early > 0 else float("nan")
        snr = early / noise_amp if noise_amp > 0 else float("inf")
        return ratio, snr

    return probe


# ---------------------------------------------------------------------------
# Joint gate
# ---------------------------------------------------------------------------
def gate_spurs(
    active_ft_spurs: Sequence[Spur],
    saturated_clusters: Sequence[SpurCluster],
    *,
    integer_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
    merge_tol_mhz: float = DEFAULT_INTEGER_TOL_MHZ,
    decay_probe: Optional[Callable[[float], Tuple[float, float]]] = None,
) -> List[GatedSpur]:
    """Combine the frequency-domain spurs with the Stage 2b cluster set.

    Without ``decay_probe`` this is the legacy frequency-domain-only gate:
    the active-FT spurs already satisfy ``integer-MHz ∧ narrow``, and the
    clusters contribute only ``saturated`` entries at integer-MHz centers.

    With ``decay_probe`` (see :func:`make_decay_probe`) every verdict is
    arbitrated in the time domain:

    * a narrow nominee whose probe DECAYS
      (``ratio < DEFAULT_DECAY_RATIO_LINE`` at
      ``amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO``) is a real line that
      happens to sit near an integer MHz -- vetoed;
    * a *pair* nominee (split-power two-bin tone,
      :attr:`Spur.pair`) is gated ONLY when its probe is FLAT
      (``ratio >= DEFAULT_DECAY_RATIO_FLAT`` at
      ``amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT``) -- source
      ``"narrow-pair"``. Its frequency-domain signature alone is ambiguous
      against a blended doublet, so without a probe (or without flatness
      confirmation) it is skipped entirely;
    * ANY cluster (saturated or not, integer or not) whose probe is FLAT
      (``ratio >= DEFAULT_DECAY_RATIO_FLAT`` at
      ``amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT``) is a confirmed CW tone --
      gated with source ``"flat"``. The bare ``saturated`` flag is not
      trusted without probe confirmation (measured false positives on
      strong erratic lines).

    Detections within ``merge_tol_mhz`` of each other are merged into one
    :class:`GatedSpur` carrying the combined provenance.
    """
    gated: List[GatedSpur] = []

    def _add(center: float, source: str, snr: float, ratio: float) -> None:
        for i, existing in enumerate(gated):
            if abs(existing.center_mhz - center) <= merge_tol_mhz:
                merged_source = existing.source
                if source not in existing.source.split("+"):
                    merged_source = "+".join(
                        sorted(set(existing.source.split("+")) | {source})
                    )
                # Prefer the frequency-domain center / SNR (a measured bin)
                # over the catalogue center when both are present.
                if source == "narrow":
                    gated[i] = GatedSpur(
                        center_mhz=center,
                        integer_mhz=int(round(center)),
                        source=merged_source,
                        snr=snr,
                        narrowness_ratio=ratio,
                    )
                else:
                    gated[i] = GatedSpur(
                        center_mhz=existing.center_mhz,
                        integer_mhz=existing.integer_mhz,
                        source=merged_source,
                        snr=existing.snr,
                        narrowness_ratio=existing.narrowness_ratio,
                    )
                return
        gated.append(
            GatedSpur(
                center_mhz=center,
                integer_mhz=int(round(center)),
                source=source,
                snr=snr,
                narrowness_ratio=ratio,
            )
        )

    for sp in active_ft_spurs:
        if sp.pair:
            # A two-bin (split-power) nominee: the frequency-domain
            # signature is ambiguous against a blended doublet, so it is
            # gated only on positive time-domain flatness confirmation --
            # never on frequency-domain evidence alone (and never without
            # a probe).
            if decay_probe is None:
                continue
            decay_ratio, amp_snr = decay_probe(sp.center_mhz)
            if (
                np.isfinite(decay_ratio)
                and decay_ratio >= DEFAULT_DECAY_RATIO_FLAT
                and amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT
            ):
                _add(sp.center_mhz, "narrow-pair", sp.snr, sp.narrowness_ratio)
            continue
        if decay_probe is not None:
            decay_ratio, amp_snr = decay_probe(sp.center_mhz)
            if (
                np.isfinite(decay_ratio)
                and decay_ratio < DEFAULT_DECAY_RATIO_LINE
                and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
            ):
                # The tone decays: a real line near an integer MHz, not a
                # clock spur. Veto.
                continue
        _add(sp.center_mhz, "narrow", sp.snr, sp.narrowness_ratio)

    for cl in saturated_clusters:
        center = float(cl.center_freq_mhz)
        if decay_probe is not None:
            decay_ratio, amp_snr = decay_probe(center)
            if (
                np.isfinite(decay_ratio)
                and decay_ratio >= DEFAULT_DECAY_RATIO_FLAT
                and amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT
            ):
                source = "flat+saturated" if cl.saturated else "flat"
                _add(center, source, float("nan"), float("nan"))
            continue
        # Legacy (no probe): only saturated clusters at integer MHz.
        if not cl.saturated:
            continue
        if abs(center - round(center)) > integer_tol_mhz:
            continue
        _add(center, "saturated", float("nan"), float("nan"))

    return sorted(gated, key=lambda g: g.center_mhz)


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
    decay_probe: Optional[Callable[[float], Tuple[float, float]]] = None,
) -> SpurSet:
    """Detect + gate spurs and package them with the mask geometry.

    ``saturated_clusters`` is the persisted Stage 2b ``spur_clusters``
    catalogue. Pass an empty sequence (or ``use_stft_catalogue=False``) to
    run the frequency-domain detector alone -- the auto-detect fallback
    when no Stage 2b calibration is present. ``decay_probe`` (see
    :func:`make_decay_probe`) enables the time-domain arbitration of every
    verdict (:func:`gate_spurs`); without it the legacy frequency-domain
    gate applies (clusters contribute only saturated integer-MHz entries).
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
        decay_probe=decay_probe,
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
