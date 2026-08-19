"""Instrument clock-lattice spur prior for the Stage 5 fit.

A reference-locked synthesizer chain emits CW tones at exact integer
combinations of its fundamentals; those combinations are exactly the
multiples of the fundamentals' greatest common divisor (the *locked
lattice*). On the home instrument the locked fundamentals
``{5760, 5120, 16000}`` MHz share ``gcd = 320`` MHz, so every Rb-locked
clock harmonic in band lands on a multiple of 320 -- a ~300x tighter
frequency prior than "any integer MHz".

A clock harmonic enters the receiver in one of two frames:

* the **baseband** frame, ``f_bb = s_bb * (f_probe - f_mol)`` -- the tone
  the digitizer sees after the LO mix (with ``s_bb`` the lower/upper
  sideband sign);
* the **RF (molecular)** frame, ``f_mol`` directly -- a harmonic radiating
  into the receiver front end.

A multiple of the lattice GCD in *either* frame is a lattice point. The
membership test is therefore pure modular arithmetic on ``g`` (no
enumeration): ``f_mol mod g`` near 0 (RF frame) or ``f_bb mod g`` near 0
(baseband frame).

The one *unlocked* clock (a free-running digitizer ADC) is referenced to
nothing, so its harmonics **drift** relative to the Rb axis (the home
instrument's 6250-MHz ADC produces the wandering 39040 tone). Unlocked
clocks contribute a separate *drifting lattice*: multiples of the unlocked
fundamental, matched with a wide window (the drift scale, ~0.2 MHz) and
flagged as expected-to-drift so the gate swaps its fixed-frequency demod
probe for a drift-tolerant band-power statistic.

An unlocked clock is a *digitizer* clock: it injects its harmonics at the
point of digitization, so its drifting points exist in the **baseband
frame only** (``f_bb = s_bb * (f_probe - f_mol)`` a multiple of the
fundamental). An ADC clock does not radiate into the RF front end, so it
has no molecular-frame (RF) family -- generating one manufactures false
predictions on top of real molecular bands (measured: a spurious
``6250x5 (rf)`` point at 31250 lands on the 363 fixture's K=6 molecular
anchor band). Locked synthesizer clocks keep both frames (their harmonics
both enter the digitizer after the LO mix *and* radiate at RF).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..core.stage_fit_settings import ClockSource
from .peak_model import sideband_sign

__all__ = [
    "LatticePoint",
    "ClockLattice",
    "build_clock_lattice",
]

logger = logging.getLogger(__name__)

# A declared locked fundamental is accepted as integral-MHz when it is within
# this of an integer (Rb-locked synthesizers are exact to << 1 Hz).
#
# Legitimately absolute (dev-docs/SCIENCE_STRATEGY.md Requirement 8): this is
# a float-comparison tolerance on a user-declared clock fundamental (a value
# a person typed when declaring an instrument clock source), not a spectral
# distance on any FT grid -- it owes nothing to the active-FT bin spacing and
# must not be converted to a bin-relative quantity.
_INTEGRAL_TOL_MHZ = 1e-6


@dataclass(frozen=True)
class LatticePoint:
    """One predicted clock-lattice frequency in the molecular frame.

    Attributes
    ----------
    freq_mhz : float
        Molecular-frame frequency of the predicted tone.
    identity : str
        Human-readable clock identity, e.g. ``"320x6 (bb)"`` (locked GCD
        multiple in the baseband frame), ``"320x80 (rf)"`` (RF frame), or
        ``"6250x3 (bb, drift)"`` (an unlocked-clock harmonic; drifting
        points are baseband-only).
    drift : bool
        Whether the point belongs to the drifting (unlocked-clock) family.
    window_mhz : float
        The match half-window: the measurement tolerance (~one active-FT
        bin) for locked points, the drift scale for drifting points.
    """

    freq_mhz: float
    identity: str
    drift: bool
    window_mhz: float


@dataclass(frozen=True)
class ClockLattice:
    """The locked + drifting lattice predicted by a clock declaration.

    Built by :func:`build_clock_lattice`. ``nomination_points`` enumerates
    the in-band lattice frequencies (the Stage 5 nomination anchor that
    replaces the integer-MHz sweep); ``match`` is the cheap modular
    membership test used to annotate an arbitrary frequency;
    ``baseband_mhz`` exposes a molecular-frame tone's baseband frequency
    (used to size the eps-aware spur match window).
    """

    g_mhz: int
    locked_freqs_mhz: Tuple[int, ...]
    unlocked_freqs_mhz: Tuple[float, ...]
    probe_freq_mhz: float
    sideband_sign: float
    band: Tuple[float, float]
    tol_mhz: float
    drift_window_mhz: float

    # -- frame helpers ---------------------------------------------------
    def _f_bb(self, f_mol: float) -> float:
        """Baseband frequency of a molecular-frame tone (>= 0)."""
        # f_bb = s * (f_mol - probe) per the sideband convention; the
        # digitizer sees |f_bb|.
        return abs(self.sideband_sign * (f_mol - self.probe_freq_mhz))

    def baseband_mhz(self, f_mol_mhz: float) -> float:
        """Baseband frequency (>= 0) of a molecular-frame tone.

        Public wrapper over :meth:`_f_bb`: the displacement a clock scale
        error ``eps`` imparts to a tone is ``eps * f_bb``, so the eps-aware
        spur match window (Stage 5) needs each lattice point's baseband
        frequency.
        """
        return self._f_bb(float(f_mol_mhz))

    def _f_mol_from_bb(self, f_bb: float) -> float:
        """Molecular frequency of a baseband tone (inverse of ``_f_bb``)."""
        # f_mol = probe + s * f_bb.  Lower sideband (s < 0): f_mol descends
        # as f_bb rises, so probe - f_bb.
        return float(self.probe_freq_mhz + self.sideband_sign * f_bb)

    # -- nomination ------------------------------------------------------
    def nomination_points(self) -> List[LatticePoint]:
        """Sorted in-band lattice points (the Stage 5 nomination anchor).

        Locked multiples ``k*g`` (``k >= 1``) are mapped into the band from
        both frames; the baseband identity is preferred when a point lands
        in both frames (e.g. when the probe is itself on-lattice, every bb
        point is also an rf point). Drifting points (unlocked-clock
        harmonics) follow, with any within ``tol`` of a locked point dropped
        (locked wins).
        """
        lo, hi = self.band if self.band[0] <= self.band[1] else self.band[::-1]
        # Collect locked points keyed by rounded molecular frequency so the
        # bb/rf degeneracy dedupes and bb wins.
        locked: List[LatticePoint] = []
        if self.g_mhz > 0:
            seen_bb: List[float] = []
            # Baseband frame: f_bb = k*g maps to f_mol = probe + s*k*g.
            k = 1
            while True:
                f_bb = k * self.g_mhz
                f_mol = self._f_mol_from_bb(float(f_bb))
                if self.sideband_sign < 0:
                    # f_mol descends with k; stop once below the band.
                    if f_mol < lo:
                        break
                    if f_mol <= hi:
                        locked.append(
                            LatticePoint(
                                freq_mhz=f_mol,
                                identity=f"{self.g_mhz}x{k} (bb)",
                                drift=False,
                                window_mhz=self.tol_mhz,
                            )
                        )
                        seen_bb.append(f_mol)
                else:
                    if f_mol > hi:
                        break
                    if f_mol >= lo:
                        locked.append(
                            LatticePoint(
                                freq_mhz=f_mol,
                                identity=f"{self.g_mhz}x{k} (bb)",
                                drift=False,
                                window_mhz=self.tol_mhz,
                            )
                        )
                        seen_bb.append(f_mol)
                k += 1
            # RF frame: f_mol = k*g directly.
            k_lo = int(math.ceil(lo / self.g_mhz))
            k_hi = int(math.floor(hi / self.g_mhz))
            for kk in range(max(k_lo, 1), k_hi + 1):
                f_mol = float(kk * self.g_mhz)
                if any(abs(f_mol - b) <= self.tol_mhz for b in seen_bb):
                    # Already a baseband point (degenerate); bb identity wins.
                    continue
                locked.append(
                    LatticePoint(
                        freq_mhz=f_mol,
                        identity=f"{self.g_mhz}x{kk} (rf)",
                        drift=False,
                        window_mhz=self.tol_mhz,
                    )
                )
        # Drifting points: per unlocked clock, harmonics in the BASEBAND
        # frame only (an unlocked ADC clock injects at digitization and does
        # not radiate RF, so it has no molecular-frame family); dropped if
        # they coincide with a locked point.
        drift: List[LatticePoint] = []
        for c in self.unlocked_freqs_mhz:
            if c <= 0:
                continue
            k = 1
            while True:
                f_bb_drift = float(k * c)
                if f_bb_drift > self._f_bb_max():
                    break
                f_mol = self._f_mol_from_bb(f_bb_drift)
                if lo <= f_mol <= hi:
                    drift.append(
                        LatticePoint(
                            freq_mhz=f_mol,
                            identity=f"{_clk(c)}x{k} (bb, drift)",
                            drift=True,
                            window_mhz=self.drift_window_mhz,
                        )
                    )
                k += 1
        locked_freqs = [p.freq_mhz for p in locked]
        drift = [
            p
            for p in drift
            if not any(abs(p.freq_mhz - lf) <= self.tol_mhz for lf in locked_freqs)
        ]
        return sorted(locked + drift, key=lambda p: p.freq_mhz)

    def _f_bb_max(self) -> float:
        """Largest baseband frequency reachable within the band."""
        lo, hi = self.band if self.band[0] <= self.band[1] else self.band[::-1]
        return max(self._f_bb(lo), self._f_bb(hi))

    # -- membership ------------------------------------------------------
    def match(self, f_mol_mhz: float) -> Optional[LatticePoint]:
        """Locked-or-drift membership test for an arbitrary frequency.

        Returns the matched :class:`LatticePoint` (with the actual harmonic
        index ``k`` stamped into its identity) or ``None``. Locked match is
        modular distance to a multiple of ``g`` in either frame (``k = 0``
        excluded); drift match is per-unlocked-clock modular distance within
        the drift window. Locked beats drift.
        """
        f_mol = float(f_mol_mhz)
        if self.g_mhz > 0:
            # RF frame.
            r_rf, k_rf = _mod_dist(f_mol, self.g_mhz)
            if k_rf >= 1 and r_rf <= self.tol_mhz:
                return LatticePoint(
                    freq_mhz=f_mol,
                    identity=f"{self.g_mhz}x{k_rf} (rf)",
                    drift=False,
                    window_mhz=self.tol_mhz,
                )
            # Baseband frame.
            f_bb = self._f_bb(f_mol)
            r_bb, k_bb = _mod_dist(f_bb, self.g_mhz)
            if k_bb >= 1 and r_bb <= self.tol_mhz:
                return LatticePoint(
                    freq_mhz=f_mol,
                    identity=f"{self.g_mhz}x{k_bb} (bb)",
                    drift=False,
                    window_mhz=self.tol_mhz,
                )
        for c in self.unlocked_freqs_mhz:
            if c <= 0:
                continue
            # Baseband frame only: an unlocked ADC clock injects at
            # digitization and does not radiate RF (see ``nomination_points``
            # / the module docstring).
            f_bb = self._f_bb(f_mol)
            r_bb, k_bb = _mod_dist(f_bb, c)
            if k_bb >= 1 and r_bb <= self.drift_window_mhz:
                return LatticePoint(
                    freq_mhz=f_mol,
                    identity=f"{_clk(c)}x{k_bb} (bb, drift)",
                    drift=True,
                    window_mhz=self.drift_window_mhz,
                )
        return None


def _clk(freq_mhz: float) -> str:
    """Compact clock-fundamental label (int when integral, else trimmed)."""
    if abs(freq_mhz - round(freq_mhz)) <= _INTEGRAL_TOL_MHZ:
        return str(int(round(freq_mhz)))
    return f"{freq_mhz:g}"


def _mod_dist(value: float, base: float) -> Tuple[float, int]:
    """Modular distance of ``value`` to the nearest multiple of ``base``.

    Returns ``(min(r, base - r), k)`` where ``k`` is the nearest multiple
    index. ``k`` may be 0 (caller excludes the DC term).
    """
    if base <= 0:
        return float("inf"), 0
    q = value / base
    k = int(round(q))
    return abs(value - k * base), k


def _gcd_int(values: Sequence[int]) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, int(v))
    return g


def build_clock_lattice(
    clocks: Optional[Sequence[ClockSource]],
    *,
    probe_freq_mhz: float,
    sideband: Any,
    band: Tuple[float, float],
    tol_mhz: float,
    drift_window_mhz: float,
) -> Optional[ClockLattice]:
    """Build the :class:`ClockLattice` from a clock declaration.

    Returns ``None`` when ``clocks`` is empty or ``None`` (no declaration ->
    legacy integer-MHz anchor). Locked fundamentals must be integral MHz
    (Rb-locked synthesizers are exact); a non-integral locked entry is
    warned and skipped. With no usable locked clock there is no locked
    lattice (``g = 0``), but unlocked clocks still contribute a drifting
    lattice.
    """
    if not clocks:
        return None
    s = sideband_sign(sideband)
    locked_int: List[int] = []
    unlocked: List[float] = []
    for c in clocks:
        if c.locked:
            f = float(c.freq_mhz)
            if abs(f - round(f)) > _INTEGRAL_TOL_MHZ:
                logger.warning(
                    "clock-lattice: skipping non-integral locked clock %.6f MHz "
                    "(label=%r); locked fundamentals must be integral MHz",
                    f,
                    c.label,
                )
                continue
            locked_int.append(int(round(f)))
        else:
            unlocked.append(float(c.freq_mhz))
    g = _gcd_int(locked_int)
    return ClockLattice(
        g_mhz=int(g),
        locked_freqs_mhz=tuple(locked_int),
        unlocked_freqs_mhz=tuple(unlocked),
        probe_freq_mhz=float(probe_freq_mhz),
        sideband_sign=float(s),
        band=(float(band[0]), float(band[1])),
        tol_mhz=float(tol_mhz),
        drift_window_mhz=float(drift_window_mhz),
    )
