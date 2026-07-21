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
  bin towers over its neighbors, whereas a real finite-T molecular line
  has a coherent leakage skirt. This is the zero-false-positive primary
  (it spares real lines that merely sit near an integer MHz).
* **temporal flatness** -- the Stage 2b STFT τ-calibration flags bins
  whose magnitude is flat across the STFT frames (``spur_by_tau``: τ
  rails to ``tau_max``). That flag is persisted per :class:`SpurCluster`
  as :attr:`SpurCluster.saturated`. It catches split-bin spurs the
  narrowness test misses. The *raw* ``cls == 1`` catalog is **not**
  usable here -- its AICc branch also fires on erratic beat / blend bins
  that are not spurs (measured on 2638: 53 of 59 integer-MHz ``cls == 1``
  bins are erratic real lines); only the saturated subset is trusted.

The nomination gate is ``integer-MHz ∧ (narrow ∨ saturated)``, plus a
**pair lane** for tones that fall *between* two grid bins: the split power
defeats the single-bin narrowness test, so the two-bin pair is tested
against its second neighbors instead -- but that signature is ambiguous
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
  has a neighbor ratio ~ 0.23) is *vetoed* out of the gate when the probe
  sees it decay;
* a genuinely flat tone at a NON-integer frequency (LO/IF intermodulation
  rather than a clock harmonic) nominated by the Stage 2b cluster
  catalog is gated when the probe confirms flatness -- the ``saturated``
  flag alone is not trusted in either direction (measured false positives
  on decaying lines and false negatives on real tones).

See ``dev-docs/research/stage5-gaussian-audit/report.md`` §§ "Spur-detection
prototype", "Flatness-exposure measurement".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import numpy as np

from .clock_lattice import ClockLattice, LatticePoint
from .peak_model import sideband_sign
from .tau_calibration import SpurCluster

__all__ = [
    "Spur",
    "GatedSpur",
    "SpurMaskSpec",
    "SpurSet",
    "DEFAULT_INTEGER_TOL_MHZ",
    "DEFAULT_MATCH_BIN_FRACTION",
    "DEFAULT_EPS_WIDEN_N_SIGMA",
    "DEFAULT_NARROWNESS_RATIO",
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_MASK_HALF_WIDTH_BINS",
    "DEFAULT_DECAY_RATIO_LINE",
    "DEFAULT_DECAY_RATIO_FLAT",
    "DEFAULT_DECAY_MIN_SNR_VETO",
    "DEFAULT_DECAY_MIN_SNR_FLAT",
    "DEFAULT_DECAY_N_FRAMES",
    "DEFAULT_LATTICE_DECAY_RATIO",
    "DEFAULT_DRIFT_BAND_RATIO",
    "DEFAULT_DRIFT_MIN_SNR",
    "DEFAULT_CHIRP_RESPONSE_GATE_RATIO",
    "DEFAULT_CHIRP_RESPONSE_PROTECT_RATIO",
    "detect_active_ft_spurs",
    "detect_drift_spurs",
    "make_decay_probe",
    "make_band_power_probe",
    "make_chirp_response_probe",
    "gate_spurs",
    "build_spur_set",
]

# Detection thresholds (validated on the 2638 fixture; instrument-tunable
# via the Stage 5 ``spur`` settings sub-block).
DEFAULT_INTEGER_TOL_MHZ = 0.04  # ~half a bin (active-FT spacing ~79 kHz)
# Nearest-bin match window as a fraction of the active-FT bin spacing
# ``Delta_f = 1/T_active``. 0.5 == half a bin: correct at any active length,
# whereas the fixed ``DEFAULT_INTEGER_TOL_MHZ`` (0.04) is only ~half a bin at
# the reference ``T_active ~ 13 us``. The match tolerance is the larger of the
# two (0.04 is an absolute floor), so long/reference records are byte-neutral
# and short records get a correctly-sized window.
DEFAULT_MATCH_BIN_FRACTION = 0.5
# Clock scale-error (eps) match-window widening: a scale error displaces a
# measured tone by ``eps * f_bb`` (baseband frequency), so the eps-aware match
# window is widened by ``(|eps| + N * |sigma_eps|) * f_bb``. N is the sigma
# multiplier for the eps uncertainty term.
DEFAULT_EPS_WIDEN_N_SIGMA = 3.0
DEFAULT_NARROWNESS_RATIO = 0.30  # max(neighbor)/peak below this => narrow
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

# Chirp-response anchor thresholds. Measured on the reference instrument
# (Keysight UXR0204A, 19-frame scope record): known interference tones have
# pre/FID amplitude ratios 0.59–1.68 (7 of 8 cataloged >= 0.9; clock tones
# 0.81–1.01); known chirp-responsive molecular lines have ratios <= 0.28,
# consistent with e^{-Δt/τ} bleed from the previous chirp. The ambiguous
# band (0.50–0.74) is left as inconclusive (fall-through to other lanes).
# The pre-record floor is ~√N_frames worse than the science average's, so
# only nominees bright enough to see pre-record are arbitrated; weaker ones
# fall through unchanged.
DEFAULT_CHIRP_RESPONSE_GATE_RATIO = 0.8  # >= this AND pre_snr >= 5 -> confirmed CW
DEFAULT_CHIRP_RESPONSE_PROTECT_RATIO = (
    0.3  # <= this (+ detectability) -> chirp-responsive
)

# Clock-lattice prior thresholds. On a locked-lattice point the prior odds
# are ~300x higher than at an arbitrary integer MHz, so the evidence burden
# flips: an on-lattice narrow/pair nominee is gated UNLESS the decay probe
# *clearly* decays (ratio below this bar), rather than requiring positive
# flatness confirmation. Calibration anchors (7 fixtures): every true CW
# comb spur's decay-probe ratio >= 0.70; every real line at usable probe
# SNR <= 0.65; the contested falsely-narrow case was 0.29 -- so 0.45 clears
# every real line with margin and gates every comb tone.
DEFAULT_LATTICE_DECAY_RATIO = 0.45
# Drift-family band-power statistic. A coherent demod is meaningless under
# the ~200 kHz wander of an unlocked-clock tone (the real 39040 tone reads
# a pseudo-decay ratio 0.59), so a drift nominee is arbitrated by the
# late/early *incoherent* band-power ratio instead: the drifting 655 tone
# at 39039.997 (bb 1920 = 6x320) measures band-ratio 0.43 while real lines
# measure 0.12-0.28. The gate fires when band_ratio >= this AND band_snr
# >= DEFAULT_DRIFT_MIN_SNR.
DEFAULT_DRIFT_BAND_RATIO = 0.35
DEFAULT_DRIFT_MIN_SNR = 10.0


@dataclass(frozen=True)
class Spur:
    """One integer-MHz, sub-resolution-narrow bin on the active-FT.

    ``pair`` marks a *two-bin* nominee: a CW tone whose frequency falls
    between two grid bins splits its power across them (each reads ~0.6-1.0
    of the other), so the single-bin neighbor test fails even though the
    pair together is transform-limited (second neighbors fall back to the
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
    # Clock-lattice provenance: ``lattice`` is the matched lattice identity
    # (e.g. ``"320x6 (bb)"``) when the nominee came from a clock-declaration
    # sweep, else ``None`` (legacy integer-MHz sweep). ``drift`` marks a
    # band-power-arbitrated nominee (the unlocked-clock drifting family, or a
    # locked point that wandered -- see ``locked_fallback``). ``drift`` and
    # ``locked_fallback`` together: the band-power lane nominated a *locked*
    # point because no narrow/pair nominee covered it (655's 39040 = locked
    # 320x6 wandering under the scope clock). A locked point can sit near a
    # genuinely decaying molecular line, so the gate additionally applies the
    # lattice decay veto to ``locked_fallback`` nominees (the unlocked
    # drifting family keeps no decay veto -- it pseudo-decays).
    lattice: Optional[str] = None
    drift: bool = False
    locked_fallback: bool = False


@dataclass(frozen=True)
class GatedSpur:
    """A spur that passed the joint gate, with its provenance.

    ``source`` is one of ``"narrow"`` (frequency-domain only),
    ``"saturated"`` (Stage 2b flat-cluster only), or ``"narrow+saturated"``
    (both detectors agree). ``snr`` / ``narrowness_ratio`` are NaN when the
    spur came only from the persisted flat catalog (no active-FT
    measurement).
    """

    center_mhz: float
    integer_mhz: int
    source: str
    snr: float = float("nan")
    narrowness_ratio: float = float("nan")
    # Clock-lattice provenance carried through from the nominee.
    lattice: Optional[str] = None
    drift: bool = False
    # Per-spur residual-mask half-width (bins) when the SNR-scaled mask is
    # active; ``None`` falls back to the :class:`SpurSet` uniform default.
    mask_half_width_bins: Optional[int] = None


@dataclass(frozen=True)
class SpurMaskSpec:
    """Per-window spur mask in the window's baseband-offset frame.

    :func:`~ftmwpipeline.fitting.window_fit.fit_window` rebuilds the boolean
    bin mask from this spec and its own offset grid: a bin is masked when it
    lies within ``half_width_mhz`` of any entry of ``offsets_mhz``. Carrying
    offsets + a half-width (rather than a bin-index mask) makes the spec
    invariant to the internal grid re-sorting the fit routines perform.

    ``half_widths_mhz`` is the per-offset half-width parallel to
    ``offsets_mhz`` (the SNR-scaled mask: a strong gated tone's sinc skirt
    leaks past a fixed +/-2-bin mask, so its mask is widened ~1/(pi*Delta)).
    ``None`` (the default) means every offset uses the uniform
    ``half_width_mhz`` -- the legacy behavior, byte-identical when no clock
    declaration / SNR-scaling is in play.
    """

    offsets_mhz: Tuple[float, ...]
    half_width_mhz: float
    half_widths_mhz: Optional[Tuple[float, ...]] = None

    def bin_mask(self, offset_grid_mhz: np.ndarray) -> np.ndarray:
        """Boolean mask over ``offset_grid_mhz`` (True = masked spur bin)."""
        u = np.asarray(offset_grid_mhz, dtype=float)
        mask = np.zeros(u.shape, dtype=bool)
        if not self.offsets_mhz:
            return cast(np.ndarray, mask)
        if self.half_widths_mhz is None:
            if self.half_width_mhz <= 0.0:
                return cast(np.ndarray, mask)
            for off in self.offsets_mhz:
                mask |= np.abs(u - off) <= self.half_width_mhz
            return cast(np.ndarray, mask)
        for off, hw in zip(self.offsets_mhz, self.half_widths_mhz):
            if hw <= 0.0:
                continue
            mask |= np.abs(u - off) <= hw
        return cast(np.ndarray, mask)


@dataclass(frozen=True)
class SpurSet:
    """Gated spur catalog + mask geometry for one Stage 5 fit.

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
    flat_decay_flags : tuple of float
        Molecular-frequency centers of cluster nominees kept-but-flagged: a
        Stage-2b flat-cluster pick whose coherent decay is ambiguous
        (``DEFAULT_DECAY_RATIO_LINE..DEFAULT_DECAY_RATIO_FLAT``) -- a real line
        and a CW tone are indistinguishable there, so the line is fit but
        surfaced for review rather than masked.
    """

    spurs: Tuple[GatedSpur, ...]
    bin_spacing_mhz: float
    mask_half_width_bins: int = DEFAULT_MASK_HALF_WIDTH_BINS
    flat_decay_flags: Tuple[float, ...] = ()

    def __bool__(self) -> bool:
        return len(self.spurs) > 0

    def flat_decay_match(self, freq_mhz: float) -> bool:
        """Whether a fitted peak at ``freq_mhz`` (molecular) was flagged
        ``flat_decay`` (kept-but-ambiguous cluster nominee)."""
        tol = self.nomination_tol_mhz
        return any(abs(freq_mhz - c) <= tol for c in self.flat_decay_flags)

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

    def _spur_half_width_mhz(self, spur: GatedSpur) -> float:
        """Mask half-width (MHz) for one spur (per-spur override or default)."""
        bins = (
            spur.mask_half_width_bins
            if spur.mask_half_width_bins is not None
            else self.mask_half_width_bins
        )
        return (bins + 0.5) * self.bin_spacing_mhz

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
        in_window = [sp for sp in self.spurs if lo <= sp.center_mhz <= hi]
        if not in_window:
            return None
        offsets = tuple(float(s * (sp.center_mhz - center_mhz)) for sp in in_window)
        # Emit per-offset widths only when some spur carries an SNR-scaled
        # override; otherwise the uniform legacy field keeps the spec
        # byte-identical to the no-declaration path.
        half_widths: Optional[Tuple[float, ...]] = None
        if any(sp.mask_half_width_bins is not None for sp in in_window):
            half_widths = tuple(self._spur_half_width_mhz(sp) for sp in in_window)
        return SpurMaskSpec(
            offsets_mhz=offsets,
            half_width_mhz=self.mask_half_width_mhz,
            half_widths_mhz=half_widths,
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


def spur_set_from_catalog(
    *,
    centers_mhz: Sequence[float],
    sources: Sequence[str],
    lattice: Sequence[Optional[str]] = (),
    drift: Sequence[bool] = (),
    per_spur_mask_half_width_bins: Sequence[Optional[int]] = (),
    bin_spacing_mhz: float,
    default_mask_half_width_bins: int,
) -> SpurSet:
    """Reconstruct a :class:`SpurSet` from a persisted Stage 5 spur catalog.

    The gated spur catalog is a **Stage 5 product**: it is derived once
    during the fit (frequency-domain detector + time-domain decay/chirp
    arbitration + clock lattice) and persisted on the ``SpectrumFit``.  A
    later stage that needs the same per-window residual mask -- a Stage 6
    user-directed window refit -- must *replay* that catalog rather than
    re-running the detector, otherwise a detection-code change between the
    original fit and the refit silently re-masks the window and the refit no
    longer reproduces the fit it is editing.

    Only the fields that drive the residual mask
    (:meth:`SpurSet.window_mask_spec`) and candidate nomination
    (:meth:`SpurSet.candidate_on_spur`) are reconstructed: the per-spur
    center, source, lattice/drift provenance, and the per-spur mask
    half-width override.  Measurement-only fields (``snr`` /
    ``narrowness_ratio``) are not persisted and come back ``NaN`` -- they are
    diagnostics, never consumed by the mask.  ``bin_spacing_mhz`` is recovered
    from the (deterministically rebuilt) active-FT grid, not persisted.

    Parallel sequences shorter than ``centers_mhz`` (legacy files written
    before a field existed) default per entry: ``lattice`` -> ``None``,
    ``drift`` -> ``False``, ``per_spur_mask_half_width_bins`` -> ``None``
    (the uniform ``default_mask_half_width_bins`` then applies).
    """

    def _at(seq: Sequence[Any], i: int, default: Any) -> Any:
        return seq[i] if i < len(seq) else default

    spurs_list: List[GatedSpur] = []
    for i in range(len(centers_mhz)):
        lat = _at(lattice, i, None)
        hw = _at(per_spur_mask_half_width_bins, i, None)
        spurs_list.append(
            GatedSpur(
                center_mhz=float(centers_mhz[i]),
                integer_mhz=int(round(float(centers_mhz[i]))),
                source=str(_at(sources, i, "narrow")),
                lattice=str(lat) if lat else None,
                drift=bool(_at(drift, i, False)),
                mask_half_width_bins=int(hw) if hw is not None else None,
            )
        )
    spurs = tuple(spurs_list)
    return SpurSet(
        spurs=spurs,
        bin_spacing_mhz=float(bin_spacing_mhz),
        mask_half_width_bins=int(default_mask_half_width_bins),
    )


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
    bin_fraction: float = DEFAULT_MATCH_BIN_FRACTION,
    narrowness_ratio: float = DEFAULT_NARROWNESS_RATIO,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    lattice_points: Optional[Sequence[LatticePoint]] = None,
) -> List[Spur]:
    """Flag sub-resolution-narrow bins on the sorted active-FT.

    Parameters
    ----------
    freqs_sorted_mhz, complex_spectrum_sorted, sigma_c_sorted
        Active-FT molecular frequency grid (ascending), its complex
        spectrum, and the per-bin *complex* noise RMS ``sigma_c`` -- all
        sorted by frequency. ``sigma_c`` is the persisted Stage 2 noise
        (the SNR floor uses ``sigma_c`` directly, matching the prototype).
    band
        ``(lo, hi)`` molecular-frequency analysis range (MHz).
    integer_tol_mhz, narrowness_ratio, snr_threshold
        Gate thresholds (see module-level defaults). ``integer_tol_mhz`` is
        an absolute *floor* on the nearest-bin match window (see
        ``bin_fraction``), not the window itself.
    bin_fraction
        Nearest-bin match window as a fraction of the active-FT bin spacing
        ``Delta_f = median(|diff(freqs)|)``. In integer mode the match
        tolerance is ``max(integer_tol_mhz, bin_fraction * Delta_f)`` -- the
        bin-width term makes the window correct at any active length
        (``DEFAULT_INTEGER_TOL_MHZ`` alone is ~half a bin only at the
        reference ``T_active ~ 13 us``), while the floor keeps long/reference
        records byte-neutral. When ``freqs`` has fewer than two samples the
        spacing is unknown and the tolerance falls back to ``integer_tol_mhz``.
    lattice_points
        When ``None`` (the default) the frequency anchor is the legacy
        integer-MHz sweep and every entry uses the bin-width-derived default
        tolerance above. When a clock declaration is present, pass its
        **non-drift** :class:`~ftmwpipeline.fitting.clock_lattice.LatticePoint`
        nominations and the sweep runs over those frequencies instead
        (same nearest-bin lookup, narrow + pair tests); each produced
        :class:`Spur` is stamped with ``lattice=point.identity``. In lattice
        mode each point's match window is ``max(point.window_mhz,
        bin_fraction * Delta_f)`` -- the point's own window (which may already
        carry the upstream eps widening) floored by the bin width, rather than
        the scalar tolerance. The ``integer_mhz`` field keeps
        ``int(round(center))`` semantics.
    """
    freqs = np.asarray(freqs_sorted_mhz, dtype=float)
    mag = np.abs(np.asarray(complex_spectrum_sorted))
    sig = np.asarray(sigma_c_sorted, dtype=float)
    lo, hi = band
    bin_spacing = float(np.median(np.abs(np.diff(freqs)))) if freqs.size >= 2 else 0.0
    # The bin-width-derived default: the fixed ``integer_tol_mhz`` is an
    # absolute floor, plus a half-bin (bin_fraction) term so short records get
    # a correctly-sized window (byte-neutral where 0.5*Delta_f <= 0.04).
    default_tol = (
        max(integer_tol_mhz, bin_fraction * bin_spacing)
        if bin_spacing > 0
        else integer_tol_mhz
    )
    # Sweep entries: (target frequency, identity-or-None, match_tol). The
    # integer sweep uses ``None`` + ``default_tol`` (legacy: integer_mhz = the
    # swept integer); the lattice sweep stamps the lattice identity
    # (integer_mhz = round(center)) and each point's own bin-width-floored
    # match window.
    sweep: List[Tuple[float, Optional[str], float]]
    if lattice_points is None:
        sweep = [
            (float(f_int), None, default_tol)
            for f_int in range(int(math.ceil(lo)), int(math.floor(hi)) + 1)
        ]
    else:
        sweep = [
            (
                float(p.freq_mhz),
                p.identity,
                (
                    max(p.window_mhz, bin_fraction * bin_spacing)
                    if bin_spacing > 0
                    else p.window_mhz
                ),
            )
            for p in lattice_points
        ]
    spurs: List[Spur] = []
    for f_target, identity, match_tol in sweep:
        k = int(np.argmin(np.abs(freqs - f_target)))
        if abs(float(freqs[k]) - f_target) > match_tol:
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
            center = float(freqs[k])
            spurs.append(
                Spur(
                    integer_mhz=int(round(center)),
                    center_mhz=center,
                    bin_index=k,
                    magnitude=peak,
                    snr=snr,
                    narrowness_ratio=ratio,
                    lattice=identity,
                )
            )
            continue
        # Pair lane: a tone between two bins splits its power, so neither
        # bin passes the single-bin test. Treat the integer bin plus its
        # stronger neighbor as the tone and test the bins flanking the
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
            center = float(freqs[k_top])
            spurs.append(
                Spur(
                    integer_mhz=int(round(center)),
                    center_mhz=center,
                    bin_index=k_top,
                    magnitude=pair_peak,
                    snr=pair_peak / sigma_top if sigma_top > 0 else 0.0,
                    narrowness_ratio=pair_ratio,
                    pair=True,
                    lattice=identity,
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
# Drifting-family detector
# ---------------------------------------------------------------------------
def detect_drift_spurs(
    freqs_sorted_mhz: np.ndarray,
    complex_spectrum_sorted: np.ndarray,
    sigma_c_sorted: np.ndarray,
    drift_points: Sequence[LatticePoint],
    *,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    window_override_mhz: Optional[float] = None,
    locked_fallback: bool = False,
) -> List[Spur]:
    """Nominate the strongest bin near each drifting-lattice point.

    A drifting (unlocked-clock) tone wanders ~200 kHz off its nominal
    lattice frequency, so the narrowness / pair tests (which assume a
    bin-locked tone) do not apply. Instead, within each point's match window
    the bin of maximum ``|X| / sigma_c`` is taken; if that SNR clears
    ``snr_threshold`` it is nominated as a ``drift`` :class:`Spur`
    (``narrowness_ratio = 1.0``, no frequency-domain narrowness claim). The
    gate arbitrates these with the band-power probe, never on
    frequency-domain evidence alone.

    ``window_override_mhz`` widens the per-point match window (default: each
    point's own ``window_mhz``). It is the wide drift scale used when this
    detector is run over *locked* points as the band-power fallback lane for
    a locked tone that wandered off its narrow window (e.g. 655's 39040, a
    locked ``320x6`` baseband point that drifts under the unlocked scope
    clock and so defeats both the narrow and pair tests).

    ``locked_fallback`` stamps the produced spurs (set when this detector is
    run over *locked* points). A locked point can sit near a real decaying
    molecular line, so the gate applies the lattice decay veto to these (the
    genuine unlocked drift family carries ``locked_fallback=False`` and keeps
    no decay veto).
    """
    freqs = np.asarray(freqs_sorted_mhz, dtype=float)
    mag = np.abs(np.asarray(complex_spectrum_sorted))
    sig = np.asarray(sigma_c_sorted, dtype=float)
    spurs: List[Spur] = []
    for p in drift_points:
        window = p.window_mhz if window_override_mhz is None else window_override_mhz
        sel = np.abs(freqs - p.freq_mhz) <= window
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_window = np.where(sig[idx] > 0, mag[idx] / sig[idx], 0.0)
        kk = int(idx[int(np.argmax(snr_window))])
        peak = float(mag[kk])
        sigma = float(sig[kk]) if sig[kk] > 0 else float("nan")
        snr = peak / sigma if sigma > 0 else 0.0
        if snr < snr_threshold:
            continue
        spurs.append(
            Spur(
                integer_mhz=int(round(float(freqs[kk]))),
                center_mhz=float(freqs[kk]),
                bin_index=kk,
                magnitude=peak,
                snr=snr,
                narrowness_ratio=1.0,
                lattice=p.identity,
                drift=True,
                locked_fallback=locked_fallback,
            )
        )
    return spurs


# ---------------------------------------------------------------------------
# Drift-tolerant band-power probe
# ---------------------------------------------------------------------------
def make_band_power_probe(
    fid_data: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Any,
    half_mhz: float,
    n_frames: int = DEFAULT_DECAY_N_FRAMES,
    n_noise_probes: int = 3,
) -> Callable[[float], Tuple[float, float]]:
    """Build a drift-tolerant per-frequency band-power probe.

    A coherent demod (:func:`make_decay_probe`) is meaningless under the
    ~200 kHz wander of an unlocked-clock tone, so the drifting-family gate
    uses the *incoherent* band power instead. The active record is split
    into ``n_frames`` frames; each frame's real FFT gives the RMS magnitude
    over the bins within ``half_mhz`` of the candidate's baseband frequency
    (the prototype's ``k-1 .. k+1`` window). The per-frame RMS is
    noise-subtracted (a fixed off-band median reference) and normalized, and
    the probe returns

    * ``band_ratio`` -- median of the last third over the first third of the
      normalized excess. A drifting CW tone persists (the real 655 tone at
      39039.997 measures 0.43); a molecular line decays (real lines measure
      0.12-0.28).
    * ``band_snr`` -- the first-third excess over the noise-frame level.

    The per-frame FFTs are cached once at build time (frames are independent
    of the probe frequency).
    """
    x = np.asarray(fid_data, dtype=float)
    i0 = max(int(round(start_us / sample_dt_us)), 0)
    i1 = min(int(round(end_us / sample_dt_us)), x.size)
    seg = x[i0:i1]
    s = sideband_sign(sideband)
    third = max(n_frames // 3, 1)

    # Cache the per-frame rFFTs once -- frames don't depend on the probe.
    frame_fft: List[np.ndarray] = []
    frame_fax: List[np.ndarray] = []
    for fr in np.array_split(seg, n_frames):
        n = fr.size
        frame_fft.append(np.abs(np.fft.rfft(fr)))
        frame_fax.append(np.fft.rfftfreq(n, d=sample_dt_us))

    def _to_bb(f_mol: float) -> float:
        return float(abs(probe_freq_mhz - f_mol) if s < 0 else f_mol - probe_freq_mhz)

    def _band_power(f_bb_mhz: float) -> np.ndarray:
        out: np.ndarray = np.empty(n_frames, dtype=float)
        for i in range(n_frames):
            fax = frame_fax[i]
            mag = frame_fft[i]
            k = int(np.argmin(np.abs(fax - f_bb_mhz)))
            lo_k = max(k - 1, 0)
            hi_k = min(k + 2, mag.size)
            # Widen to +/- half_mhz when that spans more than the 3-bin core.
            half = np.flatnonzero(np.abs(fax - f_bb_mhz) <= half_mhz)
            if half.size > (hi_k - lo_k):
                seg_mag = mag[half]
            else:
                seg_mag = mag[lo_k:hi_k]
            out[i] = float(np.sqrt(np.mean(seg_mag**2))) if seg_mag.size else 0.0
        return out

    # Noise reference: median band power over a few off-band probe
    # frequencies (fixed seed, like make_decay_probe). Lines are sparse, so
    # the median lands on empty bins.
    rng = np.random.default_rng(20260610)
    nyquist_mhz = 0.5 / sample_dt_us
    noise_bbs = rng.uniform(0.05 * nyquist_mhz, 0.95 * nyquist_mhz, size=n_noise_probes)
    noise = np.median([_band_power(f) for f in noise_bbs], axis=0)

    def probe(f_mol_mhz: float) -> Tuple[float, float]:
        bp = _band_power(_to_bb(float(f_mol_mhz)))
        excess = np.sqrt(np.maximum(bp**2 - noise**2, 0.0))
        peak = float(excess.max())
        norm = excess / peak if peak > 0 else excess
        early = float(np.median(norm[:third]))
        late = float(np.median(norm[-third:]))
        band_ratio = late / early if early > 0 else float("nan")
        noise_level = float(np.median(noise))
        early_abs = float(np.median(bp[:third]))
        band_snr = early_abs / noise_level if noise_level > 0 else float("inf")
        return band_ratio, band_snr

    return probe


# ---------------------------------------------------------------------------
# Chirp-response anchor probe
# ---------------------------------------------------------------------------
def make_chirp_response_probe(
    pre_record: np.ndarray,
    fid_data: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Any,
    pre_guard_front_us: float = 1.0,
    pre_guard_back_us: float = 0.5,
    excluded_comb_mhz: Optional[Sequence[float]] = None,
) -> Callable[[float], Tuple[float, float, float]]:
    """Build a pre-record / FID amplitude ratio probe.

    The pre-record (the quiet segment before the first chirp frame in a
    segmented scope acquisition) is the time-domain anchor for the chirp-
    response verdict. Molecular emission requires chirp excitation, so a true
    line is absent pre-record (or present only as weak ``e^{-Δt/τ}`` bleed
    from the previous frame). Clock/LO interference does not care about the
    chirp and appears pre-record at full amplitude.

    Both spectra are computed as rfft amplitude spectra (``|FT| / N``). The
    tone amplitude at the candidate's baseband frequency is read as the maximum
    within ±3 bins; the local pre-record floor is the median within ±50 MHz.

    Parameters
    ----------
    pre_record
        Quiet pre-record voltage samples (real, float-convertible). Guards
        ``pre_guard_front_us`` (from the start, to skip residual ringing) and
        ``pre_guard_back_us`` (from the end, to skip the chirp onset transient)
        are removed before the spectrum is computed.
    fid_data
        Full FID array (real, float-convertible). The active slice
        ``[start_us, end_us)`` is used, matching the :func:`make_decay_probe`
        convention.
    sample_dt_us
        Sample interval in µs (same clock for both arrays).
    start_us, end_us
        Active-record bounds for the FID side (µs).
    probe_freq_mhz
        LO / probe frequency (MHz) for the sideband transform.
    sideband
        Sideband convention; passed to :func:`~ftmwpipeline.fitting.peak_model.sideband_sign`.
    pre_guard_front_us, pre_guard_back_us
        Guard intervals (µs) trimmed from the pre-record before computing its
        amplitude spectrum. Defaults: 1.0 µs front (ringing), 0.5 µs back.
    excluded_comb_mhz
        Optional list of comb spacings (MHz) whose integer multiples are
        excluded from the pre-record/FID ratio verdict. When a candidate's
        baseband frequency lies within ``tol`` of a nonzero multiple of any
        listed spacing, the probe returns ``(nan, nan, nan)`` (the
        inconclusive sentinel), letting the existing lanes (clock lattice,
        narrowness, decay) decide.

        The rationale: the interleave-offset cleanup
        (:func:`~ftmwpipeline.io.data_loaders.keysight_mat.apply_interleave_cleanup`)
        estimates per-phase DC offsets from the pre-record itself and
        subtracts the tiled pattern, which nulls the ``fs/M`` comb lines in
        the stored pre-record EXACTLY (by construction: the pre-record's own
        per-phase means go to zero), while the FID frames retain the
        signal-path remainder of those combs. The result is a manufactured
        absence: pre_amp ≈ 0, ratio ≈ 0, which the protect veto mistakes for
        a chirp-responsive molecular line. At cleanup-comb frequencies the
        absence is not physics evidence, so the probe must be silent.

    Returns
    -------
    Callable[[float], tuple[float, float, float]]
        ``probe(f_mol_mhz) -> (ratio, pre_snr, fid_detectability)`` where

        * ``ratio`` = ``pre_amp / fid_amp`` -- the key discriminant. A CW tone
          reads ≈ 1; a chirp-responsive line reads ≤ e^{-Δt/τ} (≤ 0.28 on the
          reference fixture).
        * ``pre_snr`` = ``pre_amp / pre_floor`` -- the candidate's
          significance in the pre-record.
        * ``fid_detectability`` = ``fid_amp / pre_floor`` -- how far above the
          pre-record floor the FID amplitude sits; used to ensure the probe
          only arbitrates candidates that WOULD have been visible pre-record if
          they were CW.

        Zero/NaN edges, and frequencies on an excluded comb, are handled
        conservatively (``nan`` is returned) so the gate falls through to the
        existing lanes when the probe cannot decide. The NaN sentinel is safe:
        ``_chirp_verdict`` in :func:`gate_spurs` tests ``np.isfinite(ratio)``
        before any threshold comparison, so NaN falls through to
        ``"inconclusive"`` without triggering either guard.
    """
    s = sideband_sign(sideband)

    def _to_bb(f_mol: float) -> float:
        return float(abs(probe_freq_mhz - f_mol) if s < 0 else f_mol - probe_freq_mhz)

    # Pre-record side: trim guards and compute rfft amplitude spectrum (|FT|/N).
    pre = np.asarray(pre_record, dtype=float)
    guard_front = max(int(round(pre_guard_front_us / sample_dt_us)), 0)
    guard_back = max(int(round(pre_guard_back_us / sample_dt_us)), 0)
    pre_trimmed = pre[
        guard_front : pre.size - guard_back if guard_back > 0 else pre.size
    ]
    pre_n = pre_trimmed.size
    if pre_n < 4:
        # Pre-record too short after trimming; return a probe that is always
        # inconclusive (ratio = nan) so the gate falls through.
        def _null_probe(f_mol_mhz: float) -> Tuple[float, float, float]:
            return float("nan"), float("nan"), float("nan")

        return _null_probe

    pre_fft = np.abs(np.fft.rfft(pre_trimmed)) / pre_n
    pre_fax = np.fft.rfftfreq(pre_n, d=sample_dt_us)  # baseband frequencies (MHz)

    # FID side: active slice, same amplitude convention.
    fid = np.asarray(fid_data, dtype=float)
    i0 = max(int(round(start_us / sample_dt_us)), 0)
    i1 = min(int(round(end_us / sample_dt_us)), fid.size)
    fid_seg = fid[i0:i1]
    fid_n = fid_seg.size
    if fid_n < 4:

        def _null_probe(f_mol_mhz: float) -> Tuple[float, float, float]:
            return float("nan"), float("nan"), float("nan")

        return _null_probe

    fid_fft = np.abs(np.fft.rfft(fid_seg)) / fid_n
    fid_fax = np.fft.rfftfreq(fid_n, d=sample_dt_us)

    # Bin window (±3 bins of the candidate) at the pre-record frequency resolution.
    half_bins = 3
    # Floor search window: ±50 MHz of the candidate's baseband frequency.
    floor_half_mhz = 50.0

    # Comb-exclusion tolerance: at least 0.5 MHz OR 4 pre-record bin widths,
    # whichever is larger. This covers the DC comb lines whose pre-record
    # absence was manufactured by the interleave-offset cleanup.
    pre_bin_width_mhz = float(pre_fax[1]) if pre_fax.size > 1 else 0.0
    _comb_tol_mhz = max(0.5, 4.0 * pre_bin_width_mhz)
    # Pre-compute the list of (spacing, tol) pairs once; the closure captures
    # only these scalars plus the already-computed spectra.
    _comb_spacings: List[float] = (
        [float(sp) for sp in excluded_comb_mhz if float(sp) > 0.0]
        if excluded_comb_mhz is not None
        else []
    )

    def _on_excluded_comb(f_bb: float) -> bool:
        """Return True when ``f_bb`` (MHz, baseband) sits on an excluded comb."""
        for sp in _comb_spacings:
            n = f_bb / sp
            nearest = round(n)
            if nearest != 0 and abs(n - nearest) * sp <= _comb_tol_mhz:
                return True
        return False

    def probe(f_mol_mhz: float) -> Tuple[float, float, float]:
        f_bb = _to_bb(float(f_mol_mhz))

        # Interleave-cleanup comb exclusion: the cleanup nulls the fs/M comb
        # lines in the stored pre-record exactly (the pre-record's own per-
        # phase means go to zero), while the FID frames retain the signal-path
        # remainder. The resulting pre_amp ≈ 0 and ratio ≈ 0 would incorrectly
        # trigger the protect veto. Return the inconclusive sentinel instead so
        # the existing lanes (clock lattice, narrowness, decay) decide.
        if _on_excluded_comb(f_bb):
            return float("nan"), float("nan"), float("nan")

        # -- Pre-record amplitude --
        k_pre = int(np.argmin(np.abs(pre_fax - f_bb)))
        lo_pre = max(k_pre - half_bins, 0)
        hi_pre = min(k_pre + half_bins + 1, pre_fft.size)
        pre_amp = float(np.max(pre_fft[lo_pre:hi_pre])) if hi_pre > lo_pre else 0.0

        # -- Pre-record local floor (median within ±50 MHz) --
        floor_mask = np.abs(pre_fax - f_bb) <= floor_half_mhz
        floor_vals = pre_fft[floor_mask]
        pre_floor = float(np.median(floor_vals)) if floor_vals.size > 0 else 0.0

        # -- FID amplitude --
        k_fid = int(np.argmin(np.abs(fid_fax - f_bb)))
        lo_fid = max(k_fid - half_bins, 0)
        hi_fid = min(k_fid + half_bins + 1, fid_fft.size)
        fid_amp = float(np.max(fid_fft[lo_fid:hi_fid])) if hi_fid > lo_fid else 0.0

        # Guard against zero/NaN edges conservatively.
        if fid_amp <= 0.0 or pre_floor <= 0.0:
            return float("nan"), float("nan"), float("nan")

        ratio = pre_amp / fid_amp
        pre_snr = pre_amp / pre_floor
        fid_detectability = fid_amp / pre_floor
        return ratio, pre_snr, fid_detectability

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
    lattice_decay_ratio: float = DEFAULT_LATTICE_DECAY_RATIO,
    band_power_probe: Optional[Callable[[float], Tuple[float, float]]] = None,
    drift_band_ratio: float = DEFAULT_DRIFT_BAND_RATIO,
    drift_min_snr: float = DEFAULT_DRIFT_MIN_SNR,
    chirp_response_probe: Optional[
        Callable[[float], Tuple[float, float, float]]
    ] = None,
    chirp_response_gate_ratio: float = DEFAULT_CHIRP_RESPONSE_GATE_RATIO,
    chirp_response_protect_ratio: float = DEFAULT_CHIRP_RESPONSE_PROTECT_RATIO,
    flat_decay_out: Optional[List[float]] = None,
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

    Clock-lattice prior (only for nominees carrying :attr:`Spur.lattice`):

    * a narrow on-locked-lattice nominee is gated UNLESS the decay probe
      *clearly* decays (``ratio < lattice_decay_ratio`` at
      ``amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO``) -- the ~300x stronger prior
      flips the burden of proof. Off-lattice nominees keep the legacy
      ``DEFAULT_DECAY_RATIO_LINE`` (0.6) veto unchanged;
    * a pair on-locked-lattice nominee with a probe is gated UNLESS it
      clearly decays (same bar) -- vs the legacy pair which requires
      positive flatness. Without a probe, on-lattice pairs are still
      skipped (conservative, like legacy);
    * a drift nominee (:attr:`Spur.drift`) requires ``band_power_probe``: a
      coherent demod is meaningless under the ~200 kHz wander (the real
      39039.997 tone reads pseudo-decay 0.59), so it is gated when
      ``band_ratio >= drift_band_ratio`` AND ``band_snr >= drift_min_snr``
      (source ``"drift"``; the real tone measures band-ratio 0.43, real
      lines 0.12-0.28). A :attr:`Spur.locked_fallback` nominee (a locked
      point band-power-arbitrated because no narrow/pair nominee covered it)
      is additionally decay-vetoed: a locked point can sit near a real
      decaying line (655's 36800 clears the band bar but clearly decays), so
      it is dropped when ``decay_ratio < lattice_decay_ratio`` at usable
      probe SNR. The genuine unlocked drift family keeps no decay veto.
      Without ``band_power_probe`` drift nominees are skipped.

    Chirp-response anchor (only when ``chirp_response_probe`` is supplied,
    built from the persisted pre-record segment via
    :func:`make_chirp_response_probe`). The probe is evaluated for each
    nominee BEFORE the decay probe, because modulated carriers can mimic
    FID decay and fool the coherent demod:

    * **gate-confirm**: ``ratio >= chirp_response_gate_ratio`` AND
      ``pre_snr >= 5`` -- the tone is present pre-record at full amplitude,
      confirming external interference. The candidate is gated regardless of
      the decay-probe verdict (overrides a decay-probe "decays" veto).
    * **protect**: ``ratio <= chirp_response_protect_ratio`` AND
      ``fid_detectability >= 5 / chirp_response_protect_ratio`` -- the tone
      is bright enough that a CW counterpart would have been detectable
      pre-record, but it is absent (or very weak). The candidate is
      chirp-responsive; its gating is vetoed regardless of the decay probe.
    * **inconclusive**: anything else (ratio in the ambiguous band, or the
      probe returned NaN) -- fall through to the existing arbitration
      unchanged.

    The probe runs only when ``chirp_response_probe`` is not ``None`` (files
    without acquisition segments load with ``None``, and behavior is
    byte-identical to before).

    Detections within ``merge_tol_mhz`` of each other are merged into one
    :class:`GatedSpur`; a non-``None`` lattice identity is preserved across
    the merge.
    """
    gated: List[GatedSpur] = []

    def _add(
        center: float,
        source: str,
        snr: float,
        ratio: float,
        lattice: Optional[str] = None,
        drift: bool = False,
    ) -> None:
        for i, existing in enumerate(gated):
            if abs(existing.center_mhz - center) <= merge_tol_mhz:
                merged_source = existing.source
                if source not in existing.source.split("+"):
                    merged_source = "+".join(
                        sorted(set(existing.source.split("+")) | {source})
                    )
                merged_lattice = existing.lattice or lattice
                merged_drift = existing.drift or drift
                # Prefer the frequency-domain center / SNR (a measured bin)
                # over the catalog center when both are present.
                if source == "narrow":
                    gated[i] = GatedSpur(
                        center_mhz=center,
                        integer_mhz=int(round(center)),
                        source=merged_source,
                        snr=snr,
                        narrowness_ratio=ratio,
                        lattice=merged_lattice,
                        drift=merged_drift,
                    )
                else:
                    gated[i] = GatedSpur(
                        center_mhz=existing.center_mhz,
                        integer_mhz=existing.integer_mhz,
                        source=merged_source,
                        snr=existing.snr,
                        narrowness_ratio=existing.narrowness_ratio,
                        lattice=merged_lattice,
                        drift=merged_drift,
                    )
                return
        gated.append(
            GatedSpur(
                center_mhz=center,
                integer_mhz=int(round(center)),
                source=source,
                snr=snr,
                narrowness_ratio=ratio,
                lattice=lattice,
                drift=drift,
            )
        )

    # Chirp-response probe SNR floor for both the gate-confirm and protect lanes.
    _CR_PRE_SNR_FLOOR = 5.0

    def _chirp_verdict(center_mhz: float) -> str:
        """Return ``"gate"``, ``"protect"``, or ``"inconclusive"``."""
        if chirp_response_probe is None:
            return "inconclusive"
        cr_ratio, cr_pre_snr, cr_fid_det = chirp_response_probe(center_mhz)
        if not np.isfinite(cr_ratio):
            return "inconclusive"
        # Gate-confirm: pre-record amplitude comparable to FID -> CW interference.
        if cr_ratio >= chirp_response_gate_ratio and cr_pre_snr >= _CR_PRE_SNR_FLOOR:
            return "gate"
        # Protect: tone is bright enough to have been seen pre-record as CW, but
        # is absent -- it is chirp-responsive (a molecular line).
        detect_floor = _CR_PRE_SNR_FLOOR / max(chirp_response_protect_ratio, 1e-9)
        if cr_ratio <= chirp_response_protect_ratio and cr_fid_det >= detect_floor:
            return "protect"
        return "inconclusive"

    def _clearly_decays(center_mhz: float) -> bool:
        """True when the coherent decay probe confirms a real (decaying) line.

        A chirp-response gate-confirm rests on the premise that a tone present
        in the pre-record is external CW. That premise fails for a strong
        molecular line whose emission persists into the pre-record (measured on
        the succinimide UXR fixture: the brightest lines read pre-record ratio
        ~0.9-1.2). A line that *clearly decays* in the active FID is molecular
        regardless of any pre-record presence, so it must override the
        gate-confirm and fall through to the decay veto. The bar is the legacy
        off-lattice line bar (a coherent demod of a modulated/drifting carrier
        does not produce a clean exponential at this depth -- those are
        arbitrated in the drift/band-power lane, not here).
        """
        if decay_probe is None:
            return False
        decay_ratio, amp_snr = decay_probe(center_mhz)
        return bool(
            np.isfinite(decay_ratio)
            and decay_ratio < DEFAULT_DECAY_RATIO_LINE
            and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
        )

    for sp in active_ft_spurs:
        # Chirp-response anchor: run before the decay probe. Modulated carriers
        # can mimic FID decay (pseudo-decay failure mode), so the pre-record
        # presence check ranks above the coherent-demod verdict.
        cr = _chirp_verdict(sp.center_mhz)
        if cr == "protect":
            # The tone is absent pre-record despite being strong enough to
            # appear there if it were CW: chirp-responsive, not a spur.
            continue

        if sp.drift:
            # Drifting-family nominee: arbitrated by the band-power probe
            # (the coherent demod is meaningless under the wander).
            if band_power_probe is None and cr != "gate":
                continue
            # A locked-point band-power fallback can land near a real
            # decaying molecular line (655's 36800: band_ratio 0.38 clears
            # the drift bar but decay_ratio 0.39 clearly decays). Apply the
            # lattice decay veto to ``locked_fallback`` nominees; the genuine
            # unlocked drift family pseudo-decays and keeps no decay veto.
            # A chirp-response gate-confirm overrides: the pre-record confirms
            # CW presence, so there is no decay to worry about.
            if sp.locked_fallback and decay_probe is not None and cr != "gate":
                decay_ratio, amp_snr = decay_probe(sp.center_mhz)
                if (
                    np.isfinite(decay_ratio)
                    and decay_ratio < lattice_decay_ratio
                    and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
                ):
                    continue
            if cr == "gate":
                # Chirp-response gate-confirm: pre-record confirms CW, skip
                # the band-power check.
                _add(
                    sp.center_mhz,
                    "drift",
                    sp.snr,
                    sp.narrowness_ratio,
                    lattice=sp.lattice,
                    drift=True,
                )
                continue
            if band_power_probe is None:
                continue
            band_ratio, band_snr = band_power_probe(sp.center_mhz)
            if (
                np.isfinite(band_ratio)
                and band_ratio >= drift_band_ratio
                and band_snr >= drift_min_snr
            ):
                _add(
                    sp.center_mhz,
                    "drift",
                    sp.snr,
                    sp.narrowness_ratio,
                    lattice=sp.lattice,
                    drift=True,
                )
            continue
        if sp.pair:
            # A two-bin (split-power) nominee: the frequency-domain
            # signature is ambiguous against a blended doublet.
            # Chirp-response gate-confirm overrides the flatness requirement
            # (the pre-record confirms CW presence).
            if cr == "gate":
                _add(
                    sp.center_mhz,
                    "narrow-pair",
                    sp.snr,
                    sp.narrowness_ratio,
                    lattice=sp.lattice,
                )
                continue
            if decay_probe is None:
                continue
            decay_ratio, amp_snr = decay_probe(sp.center_mhz)
            if sp.lattice is not None:
                # On-locked-lattice pair: the strong prior flips the burden
                # -- gate unless the probe clearly decays.
                decays = (
                    np.isfinite(decay_ratio)
                    and decay_ratio < lattice_decay_ratio
                    and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
                )
                if not decays:
                    _add(
                        sp.center_mhz,
                        "narrow-pair",
                        sp.snr,
                        sp.narrowness_ratio,
                        lattice=sp.lattice,
                    )
                continue
            # Off-lattice pair: legacy positive-flatness requirement.
            if (
                np.isfinite(decay_ratio)
                and decay_ratio >= DEFAULT_DECAY_RATIO_FLAT
                and amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT
            ):
                _add(sp.center_mhz, "narrow-pair", sp.snr, sp.narrowness_ratio)
            continue
        # Narrow (non-pair, non-drift) nominee.
        if cr == "gate" and not _clearly_decays(sp.center_mhz):
            # Pre-record confirms CW presence and the line does not clearly
            # decay: gate (the gate-confirm overrides the modulated-carrier
            # pseudo-decay). A clearly-decaying line is a real molecular line
            # that happens to persist into the pre-record -- it falls through
            # to the decay veto below.
            _add(
                sp.center_mhz, "narrow", sp.snr, sp.narrowness_ratio, lattice=sp.lattice
            )
            continue
        if decay_probe is not None:
            decay_ratio, amp_snr = decay_probe(sp.center_mhz)
            # On a locked-lattice point the prior odds are ~300x higher, so
            # the veto bar tightens (clearly-decaying only); off-lattice
            # keeps the legacy 0.6 bar.
            veto_ratio = (
                lattice_decay_ratio
                if sp.lattice is not None
                else DEFAULT_DECAY_RATIO_LINE
            )
            if (
                np.isfinite(decay_ratio)
                and decay_ratio < veto_ratio
                and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
            ):
                # The tone decays: a real line, not a clock spur. Veto.
                continue
        _add(sp.center_mhz, "narrow", sp.snr, sp.narrowness_ratio, lattice=sp.lattice)

    for cl in saturated_clusters:
        center = float(cl.center_freq_mhz)
        # Chirp-response anchor applies to cluster nominees too.
        cr = _chirp_verdict(center)
        if cr == "protect":
            continue
        # Decay-driven verdict for a Stage-2b flat-cluster nominee. Unlike the
        # narrow lane (a single-bin integer-MHz tone is a clock spur regardless
        # of decay), the cluster catalog over-nominates *strong real lines* that
        # look flat-topped to the STFT clustering. The coherent decay is the
        # arbiter, in three tiers:
        #   * clearly decays (< LINE)        -> a real line, keep clean;
        #   * flat (>= FLAT)                 -> a CW tone, gate;
        #   * ambiguous (LINE..FLAT)         -> indistinguishable by decay alone
        #                                       (the line bleeds into the
        #                                       pre-record at a near-CW ratio) ->
        #                                       keep but flag ``flat_decay`` for
        #                                       human review.
        # The chirp gate-confirm only decides when the probe is inconclusive
        # (NaN / sub-threshold SNR); it never independently removes a
        # non-flat cluster.
        if decay_probe is not None:
            decay_ratio, amp_snr = decay_probe(center)
            if np.isfinite(decay_ratio):
                if (
                    decay_ratio < DEFAULT_DECAY_RATIO_LINE
                    and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
                ):
                    continue  # clearly decays -> real line, keep clean
                if (
                    decay_ratio >= DEFAULT_DECAY_RATIO_FLAT
                    and amp_snr >= DEFAULT_DECAY_MIN_SNR_FLAT
                ):
                    source = "flat+saturated" if cl.saturated else "flat"
                    _add(center, source, float("nan"), float("nan"))
                    continue
                if (
                    DEFAULT_DECAY_RATIO_LINE <= decay_ratio < DEFAULT_DECAY_RATIO_FLAT
                    and amp_snr >= DEFAULT_DECAY_MIN_SNR_VETO
                ):
                    if flat_decay_out is not None:
                        flat_decay_out.append(center)
                    continue  # ambiguous -> keep, flag for review
            # Inconclusive probe (NaN or sub-threshold SNR): fall back to the
            # chirp gate-confirm.
            if cr == "gate":
                source = "flat+saturated" if cl.saturated else "flat"
                _add(center, source, float("nan"), float("nan"))
            continue
        # Legacy (no probe): chirp gate-confirm, else saturated integer-MHz.
        if cr == "gate":
            source = "flat+saturated" if cl.saturated else "flat"
            _add(center, source, float("nan"), float("nan"))
            continue
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
    use_stft_catalog: bool = True,
    decay_probe: Optional[Callable[[float], Tuple[float, float]]] = None,
    lattice: Optional[ClockLattice] = None,
    timebase_epsilon: Optional[float] = None,
    timebase_sigma_epsilon: float = 0.0,
    eps_widen_n_sigma: float = DEFAULT_EPS_WIDEN_N_SIGMA,
    lattice_decay_ratio: float = DEFAULT_LATTICE_DECAY_RATIO,
    band_power_probe: Optional[Callable[[float], Tuple[float, float]]] = None,
    drift_band_ratio: float = DEFAULT_DRIFT_BAND_RATIO,
    drift_min_snr: float = DEFAULT_DRIFT_MIN_SNR,
    mask_target_residual_snr: float = 0.0,
    mask_max_half_width_bins: int = 32,
    chirp_response_probe: Optional[
        Callable[[float], Tuple[float, float, float]]
    ] = None,
    chirp_response_gate_ratio: float = DEFAULT_CHIRP_RESPONSE_GATE_RATIO,
    chirp_response_protect_ratio: float = DEFAULT_CHIRP_RESPONSE_PROTECT_RATIO,
) -> SpurSet:
    """Detect + gate spurs and package them with the mask geometry.

    ``saturated_clusters`` is the persisted Stage 2b ``spur_clusters``
    catalog. Pass an empty sequence (or ``use_stft_catalog=False``) to
    run the frequency-domain detector alone -- the auto-detect fallback
    when no Stage 2b calibration is present. ``decay_probe`` (see
    :func:`make_decay_probe`) enables the time-domain arbitration of every
    verdict (:func:`gate_spurs`); without it the legacy frequency-domain
    gate applies (clusters contribute only saturated integer-MHz entries).

    ``lattice`` is the optional clock-declaration prior
    (:class:`~ftmwpipeline.fitting.clock_lattice.ClockLattice`). It is
    strictly additive prior knowledge, so when given the nomination anchor
    is the UNION of:

    * the legacy integer-MHz sweep (every off-lattice integer tone still
      gates, on the legacy bars -- decay veto < 0.6, pair needs flat);
    * the locked-lattice sweep (lattice identity + flipped on-lattice bar
      via ``lattice_decay_ratio``); a bin found by both is shadowed by the
      lattice-stamped nominee (identity + flipped bar win);
    * the unlocked-clock drifting lane (``band_power_probe`` arbitrates it --
      see :func:`make_band_power_probe`);
    * a locked-point band-power fallback: a *locked* tone can itself wander
      under the unlocked scope clock (655's 39040 = 320x6 (bb), defeating
      both narrow and pair tests), so each locked point with no narrow/pair
      nominee in its drift window is nominated drift-style and arbitrated by
      the band-power probe.

    When ``None`` the legacy integer-MHz behavior applies and every lattice knob
    is inert.

    ``timebase_epsilon`` / ``timebase_sigma_epsilon`` are the persisted
    scope-timebase clock scale error and its uncertainty. When ``lattice`` is
    given and ``timebase_epsilon`` is not ``None``, each lattice point's search
    frequency is **shifted** to the measured position ``f_true*(1+eps)`` (a
    displacement of ``s * eps * f_bb`` in the molecular frame, ``s`` the
    sideband sign) and its match window is widened only by the eps *uncertainty*
    term ``eps_widen_n_sigma * |sigma_eps| * f_bb``. The shift is essential, not
    a mere widening: :func:`detect_active_ft_spurs` inspects the single bin
    nearest the *predicted* lattice frequency, so once a scale error pushes a
    tone past ~half a bin it lands in a *different* bin and a wider tolerance
    alone never reaches it -- only moving the search anchor onto the measured
    position does (verified empirically; see the C3 planning doc). When
    ``timebase_epsilon is None`` (no eps available) the points are untouched --
    byte-identical to the pre-eps path.

    ``mask_target_residual_snr`` (> 0) enables the SNR-scaled residual mask:
    a gated tone of SNR ``snr`` gets a per-spur mask half-width
    ``clamp(ceil(snr / (pi * target)), mask_half_width_bins,
    mask_max_half_width_bins)`` bins (a sinc skirt falls ~1/(pi*Delta_bins),
    so a strong tone's skirt that leaks past the fixed +/-2-bin mask is
    covered). For flat-lane / saturated nominees (no measured frequency-
    domain SNR) the bin SNR is read from the active FT in hand so a strong
    flat tone's skirt is masked too (363 w100). Drift spurs additionally get
    at least ``ceil(window_mhz / bin_spacing) + mask_half_width_bins`` bins.
    0 (the default) keeps the fixed uniform width.
    """
    freqs = np.asarray(freqs_sorted_mhz, dtype=float)
    lattice_points = lattice.nomination_points() if lattice is not None else None
    if (
        lattice is not None
        and lattice_points is not None
        and timebase_epsilon is not None
    ):
        # Eps-aware relocation. A clock scale error displaces a measured tone
        # to ``f_measured = f_true*(1+eps)`` in baseband, i.e. by
        # ``s * eps * f_bb`` in the molecular frame (s = sideband sign). The
        # nearest-bin detector (:func:`detect_active_ft_spurs`) picks the single
        # bin nearest the *predicted* lattice frequency, so widening the match
        # tolerance alone cannot find a tone that has moved to a different bin --
        # the search anchor itself must move. Shift each point's search
        # frequency onto the measured position, and widen its window only by the
        # eps *uncertainty* term ``N * |sigma_eps| * f_bb`` (the deterministic
        # displacement is absorbed by the shift, not the window). Applied to ALL
        # points (locked + drift); the drift lane already searches a wide window
        # so the shift is harmless there. When ``timebase_epsilon is None`` the
        # points are untouched (byte-identical to the pre-eps path).
        eps = float(timebase_epsilon)
        sigma_term = eps_widen_n_sigma * abs(float(timebase_sigma_epsilon))
        s_bb = lattice.sideband_sign
        shifted: List[LatticePoint] = []
        for point in lattice_points:
            f_bb = lattice.baseband_mhz(point.freq_mhz)
            shifted.append(
                replace(
                    point,
                    freq_mhz=point.freq_mhz + s_bb * eps * f_bb,
                    window_mhz=point.window_mhz + sigma_term * f_bb,
                )
            )
        lattice_points = shifted
    if lattice_points is None:
        # No declaration: the legacy integer-MHz nomination anchor.
        active_spurs = detect_active_ft_spurs(
            freqs,
            complex_spectrum_sorted,
            sigma_c_sorted,
            band=band,
            integer_tol_mhz=integer_tol_mhz,
            narrowness_ratio=narrowness_ratio,
            snr_threshold=snr_threshold,
            lattice_points=None,
        )
    else:
        # A declaration is strictly additive prior knowledge: nominate from
        # the UNION of the legacy integer-MHz sweep (keeps every off-lattice
        # integer tone, gated on the legacy bars) and the locked-lattice
        # sweep (flipped on-lattice bar + lattice identity). Where the same
        # bin is found by both, the lattice-stamped nominee wins (it carries
        # the identity and the flipped bar).
        locked_points = [p for p in lattice_points if not p.drift]
        lattice_spurs = detect_active_ft_spurs(
            freqs,
            complex_spectrum_sorted,
            sigma_c_sorted,
            band=band,
            integer_tol_mhz=integer_tol_mhz,
            narrowness_ratio=narrowness_ratio,
            snr_threshold=snr_threshold,
            lattice_points=locked_points,
        )
        integer_spurs = detect_active_ft_spurs(
            freqs,
            complex_spectrum_sorted,
            sigma_c_sorted,
            band=band,
            integer_tol_mhz=integer_tol_mhz,
            narrowness_ratio=narrowness_ratio,
            snr_threshold=snr_threshold,
            lattice_points=None,
        )
        # Merge by bin index: a lattice nominee shadows the integer nominee
        # at the same bin (lattice identity + flipped bar win).
        lattice_bins = {sp.bin_index for sp in lattice_spurs}
        active_spurs = lattice_spurs + [
            sp for sp in integer_spurs if sp.bin_index not in lattice_bins
        ]
        # Drifting (unlocked-clock) lane: nominate the strongest bin in each
        # drift point's wide window.
        drift_points = [p for p in lattice_points if p.drift]
        active_spurs = active_spurs + detect_drift_spurs(
            freqs,
            complex_spectrum_sorted,
            sigma_c_sorted,
            drift_points,
            snr_threshold=snr_threshold,
        )
        # Locked-point band-power fallback: a locked tone can itself wander
        # under the unlocked scope clock (655's 39040 = locked 320x6 (bb)),
        # defeating both the narrow and pair tests. For each locked point
        # that produced no narrow/pair nominee, nominate the strongest bin
        # in the wide drift window and arbitrate it with the band-power
        # probe (source "drift"), keeping the locked identity.
        if lattice is not None and band_power_probe is not None:
            nominated_bins = {sp.bin_index for sp in active_spurs}
            drift_win = lattice.drift_window_mhz
            ungated_locked = [
                p
                for p in locked_points
                if not _has_bin_within(freqs, nominated_bins, p.freq_mhz, drift_win)
            ]
            active_spurs = active_spurs + detect_drift_spurs(
                freqs,
                complex_spectrum_sorted,
                sigma_c_sorted,
                ungated_locked,
                snr_threshold=snr_threshold,
                window_override_mhz=drift_win,
                locked_fallback=True,
            )
    clusters = tuple(saturated_clusters) if use_stft_catalog else ()
    flat_decay_flags: List[float] = []
    gated = gate_spurs(
        active_spurs,
        clusters,
        integer_tol_mhz=integer_tol_mhz,
        decay_probe=decay_probe,
        lattice_decay_ratio=lattice_decay_ratio,
        band_power_probe=band_power_probe,
        drift_band_ratio=drift_band_ratio,
        drift_min_snr=drift_min_snr,
        chirp_response_probe=chirp_response_probe,
        chirp_response_gate_ratio=chirp_response_gate_ratio,
        chirp_response_protect_ratio=chirp_response_protect_ratio,
        flat_decay_out=flat_decay_flags,
    )
    if freqs.size >= 2:
        bin_spacing = float(np.median(np.abs(np.diff(freqs))))
    else:
        bin_spacing = 0.0
    base_bins = int(mask_half_width_bins)
    if mask_target_residual_snr > 0.0:
        drift_win = lattice.drift_window_mhz if lattice is not None else 0.0
        mag = np.abs(np.asarray(complex_spectrum_sorted))
        sig = np.asarray(sigma_c_sorted, dtype=float)
        gated = [
            _scale_spur_mask(
                g,
                target=mask_target_residual_snr,
                base_bins=base_bins,
                cap_bins=int(mask_max_half_width_bins),
                bin_spacing_mhz=bin_spacing,
                drift_window_mhz=drift_win,
                measured_snr=_measure_bin_snr(freqs, mag, sig, g.center_mhz),
                freqs_sorted_mhz=freqs,
                mag_sorted=mag,
                sigma_c_sorted=sig,
            )
            for g in gated
        ]
    return SpurSet(
        spurs=tuple(gated),
        bin_spacing_mhz=bin_spacing,
        mask_half_width_bins=base_bins,
        flat_decay_flags=tuple(flat_decay_flags),
    )


def _has_bin_within(
    freqs_sorted_mhz: np.ndarray,
    bin_indices: "set[int]",
    center_mhz: float,
    half_window_mhz: float,
) -> bool:
    """Whether any of ``bin_indices`` falls within ``half_window`` of center."""
    for k in bin_indices:
        if abs(float(freqs_sorted_mhz[k]) - center_mhz) <= half_window_mhz:
            return True
    return False


def _measure_bin_snr(
    freqs_sorted_mhz: np.ndarray,
    mag_sorted: np.ndarray,
    sigma_c_sorted: np.ndarray,
    center_mhz: float,
) -> float:
    """Bin SNR ``|X| / sigma_c`` at the active-FT bin nearest ``center``.

    The mask-scaling fallback for a gated spur whose own
    :attr:`GatedSpur.snr` is non-finite (the flat-lane / saturated catalog
    nominees never measured a frequency-domain bin): read the active FT in
    hand directly so a strong flat tone's sinc skirt gets a scaled mask too
    (363 w100's SNR-139 flat 28057.46 tone). Returns ``nan`` when the bin
    has no usable noise.
    """
    if freqs_sorted_mhz.size == 0:
        return float("nan")
    k = int(np.argmin(np.abs(freqs_sorted_mhz - center_mhz)))
    sigma = float(sigma_c_sorted[k])
    if sigma > 0:
        return float(mag_sorted[k]) / sigma
    return float("nan")


# Skirt-consistency truncation of the SNR-scaled mask: a bin is real
# structure (not the tone's own leakage) when its magnitude exceeds both
# this multiple of the predicted sinc-skirt envelope and this many sigma_c.
_MASK_STRUCT_ENVELOPE_FACTOR = 2.0
_MASK_STRUCT_MIN_SNR = 5.0


def _skirt_consistent_half_width(
    freqs_sorted_mhz: np.ndarray,
    mag_sorted: np.ndarray,
    sigma_c_sorted: np.ndarray,
    center_mhz: float,
    base_bins: int,
    want_bins: int,
) -> int:
    """Truncate a scaled mask where the spectrum stops looking like the
    tone's own sinc skirt.

    The wide mask exists to absorb the gated tone's leakage; a bin whose
    magnitude clearly exceeds the predicted skirt envelope
    (``peak / (pi * Delta_bins)``) is real structure -- typically a
    molecular line -- that the mask must not eat. Measured case: 1512's
    flat tone at 28440.26 sits 0.5-0.8 MHz from four vinyl cyanide catalog
    lines; an unguarded SNR-scaled mask (19 bins ~ 1.6 MHz) swallowed all
    four, while the truncated mask stops below them. Both sides truncate
    independently and the symmetric half-width is their minimum, floored
    at ``base_bins`` (the legacy core mask, which is never reduced).
    """
    if want_bins <= base_bins or mag_sorted.size == 0:
        return want_bins
    k = int(np.argmin(np.abs(freqs_sorted_mhz - center_mhz)))
    peak = float(mag_sorted[k])
    if not peak > 0:
        return want_bins
    half = want_bins
    for d in range(base_bins + 1, want_bins + 1):
        skirt = peak / (math.pi * d)
        for j in (k - d, k + d):
            if 0 <= j < mag_sorted.size:
                sigma = float(sigma_c_sorted[j])
                if (
                    float(mag_sorted[j]) > _MASK_STRUCT_ENVELOPE_FACTOR * skirt
                    and sigma > 0
                    and float(mag_sorted[j]) > _MASK_STRUCT_MIN_SNR * sigma
                ):
                    return max(d - 1, base_bins)
    return half


def _scale_spur_mask(
    spur: GatedSpur,
    *,
    target: float,
    base_bins: int,
    cap_bins: int,
    bin_spacing_mhz: float,
    drift_window_mhz: float,
    measured_snr: float = float("nan"),
    freqs_sorted_mhz: Optional[np.ndarray] = None,
    mag_sorted: Optional[np.ndarray] = None,
    sigma_c_sorted: Optional[np.ndarray] = None,
) -> GatedSpur:
    """Assign a :class:`GatedSpur` an SNR-scaled per-spur mask half-width.

    A sinc skirt falls ~1/(pi*Delta_bins), so a tone of amplitude ``snr``
    (relative to noise) drops to the target residual SNR at
    ``snr / (pi * target)`` bins; clamp to ``[base_bins, cap_bins]``. A
    drift spur wanders across its match window, so it gets at least
    ``ceil(window_mhz / bin_spacing) + base_bins`` bins (still capped).

    ``measured_snr`` is the bin SNR read from the active FT (see
    :func:`_measure_bin_snr`); it scales the mask when the spur's own
    :attr:`GatedSpur.snr` is non-finite (the flat-lane / saturated nominees),
    so a strong flat tone's skirt still earns a wide mask. The persisted
    :attr:`GatedSpur.snr` field is left unchanged (its NaN-vs-finite value
    distinguishes flat-catalog from frequency-domain provenance).

    When the sorted active-FT arrays are supplied, the scaled width is
    truncated where the spectrum stops being skirt-consistent
    (:func:`_skirt_consistent_half_width`), so a wide mask never swallows
    a real line near the tone. The drift floor is applied after truncation
    (a wandering tone's smear is its own structure).
    """
    bins = base_bins
    snr_for_scale = spur.snr if np.isfinite(spur.snr) else measured_snr
    if np.isfinite(snr_for_scale) and snr_for_scale > 0:
        scaled = int(math.ceil(snr_for_scale / (math.pi * target)))
        bins = max(base_bins, min(scaled, cap_bins))
        if (
            bins > base_bins
            and freqs_sorted_mhz is not None
            and mag_sorted is not None
            and sigma_c_sorted is not None
        ):
            bins = _skirt_consistent_half_width(
                freqs_sorted_mhz,
                mag_sorted,
                sigma_c_sorted,
                spur.center_mhz,
                base_bins,
                bins,
            )
    if spur.drift and bin_spacing_mhz > 0 and drift_window_mhz > 0:
        drift_bins = int(math.ceil(drift_window_mhz / bin_spacing_mhz)) + base_bins
        bins = max(bins, min(drift_bins, cap_bins))
    return GatedSpur(
        center_mhz=spur.center_mhz,
        integer_mhz=spur.integer_mhz,
        source=spur.source,
        snr=spur.snr,
        narrowness_ratio=spur.narrowness_ratio,
        lattice=spur.lattice,
        drift=spur.drift,
        mask_half_width_bins=bins,
    )
