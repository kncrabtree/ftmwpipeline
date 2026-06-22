"""Scope-timebase self-calibration from Rb-locked clock spur tones.

The digitizer (scope) clock is the one instrument source not referenced to
the Rb frequency standard. It carries a stable fractional scale error ``eps``:
every measured frequency reads ``f_true * (1 + eps)``. The Rb-locked clock
tree predicts exact CW tones at every multiple of ``g = gcd(locked clock
fundamentals)`` baseband MHz (``g = 320`` on the home instrument). A detected
lattice tone at nominal baseband ``k * g`` is measured at an offset
``df = eps * f_bb``, so each tone gives one estimate ``eps = df / f_bb`` and a
joint weighted fit over the lattice yields ``eps`` with parts-in-10^7 formal
precision.

The strongest lattice tones sit *above* the chirp-driven molecular band
(measured: baseband ``15360 = 3 * 5120`` MHz at peak/noise up to ~7000, plus
``17920``, ``17280``) -- molecular-free, so the full active record is usable
without late-windowing to dodge line pulling. Below ~1 GHz baseband a third
tone family appears with kHz-scale offsets that are *not* eps-scaled (measured:
the 320 MHz tone reads -5.5 kHz on two fixtures while eps predicts +0.7 kHz).
The shared-eps consistency rejection removes these: a tone whose measured
offset is more than ``4 * sigma_tot`` from the joint ``eps * f_bb`` line is
dropped from the fit.

The unlocked-clock multiples (e.g. the scope's own ``k * 6250`` comb) are
measured as *drift controls*: they sit at exact rational frequencies in sample
space, so their measured offsets read ~0, **not** ``eps * f_bb``. That contrast
-- a locked tone tracks the common eps line while an unlocked tone reads ~0 --
is the drifting-family discriminant in a single measurement. Controls never
enter the eps fit.

Correction semantics (consumed elsewhere, *not* applied here): with the
measured ``eps``, ``f_true = f_measured / (1 + eps)``. For a lower-sideband
instrument ``f_mol = probe - f_bb``, so the molecular-frame correction is
``f_mol_true = probe - (probe - f_mol_meas) / (1 + eps)``. This module only
*measures and persists* ``eps``; applying it to the frequency axis is
deliberately out of scope.

Per-fixture measured eps differs by acquisition epoch (the scope clock drifts
between sessions), so eps is a per-file quantity, never a package constant.
Accuracy is systematics-limited at ~0.1-0.2 ppm (tone-to-tone scatter); the
catalog-truth cross-check agrees within ~0.2-0.3 ppm. The estimator and its
operating points are ported from the validated prototypes
``scratch/stage5-skirt/timebase_selfcal{4,5,6}.py``; see
``dev-docs/planning/instrument-clock-declaration.md`` for the design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import gcd
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TimebaseToneRead",
    "TimebaseCalibrationResult",
    "calibrate_timebase_from_fid",
    "DEFAULT_KAPPA_SYS",
    "DEFAULT_SNR_MIN",
    "DEFAULT_N_BLOCKS",
    "DEFAULT_SCAN_HALF_RANGE_MHZ",
    "DEFAULT_SCAN_STEP_MHZ",
    "DEFAULT_NYQUIST_FRACTION",
    "DEFAULT_MAX_REJECT_ITERS",
    "DEFAULT_REJECT_SIGMA",
    "DEFAULT_EDGE_PIN_STEPS",
    "DEFAULT_N_NOISE_PROBES",
]

# Operating points ported from the validated prototypes
# (scratch/stage5-skirt/timebase_selfcal{4,5,6}.py).
DEFAULT_KAPPA_SYS = 0.2e-6  # systematic per-tone fractional floor (sigma_tot)
DEFAULT_SNR_MIN = 8.0  # peak/noise gate for a tone to count as detected
DEFAULT_N_BLOCKS = 4096  # block-average count before the ML fine scan
DEFAULT_SCAN_HALF_RANGE_MHZ = 0.1  # ML fine scan spans +-this around nominal
DEFAULT_SCAN_STEP_MHZ = 0.0001  # 0.1 kHz scan step
DEFAULT_NYQUIST_FRACTION = 0.98  # highest lattice index sits below this * Nyquist
DEFAULT_MAX_REJECT_ITERS = 10  # iterative consistency-rejection cap
DEFAULT_REJECT_SIGMA = 4.0  # |df - eps*f| > this * sigma_tot => reject
DEFAULT_EDGE_PIN_STEPS = 1.5  # maxima within this many steps of an edge are pins
DEFAULT_N_NOISE_PROBES = 8  # off-lattice scans for the noise reference


@dataclass(frozen=True)
class TimebaseToneRead:
    """One demodulated tone the timebase calibration measured.

    Attributes
    ----------
    f_bb_mhz : float
        Nominal baseband frequency of the tone (MHz). For a locked lattice
        tone this is ``k * lattice_g_mhz``; for a drift control it is a
        multiple of an unlocked-clock fundamental.
    k : int
        Lattice index (``f_bb_mhz / lattice_g_mhz`` for locked tones;
        ``f_bb_mhz / fundamental`` for controls). ``0`` if not on a lattice.
    df_mhz : float
        Measured frequency offset from nominal (MHz). Positive means the
        tone reads high.
    sigma_mhz : float
        Total per-tone uncertainty ``hypot(sigma_f, kappa_sys * f_bb)`` (MHz),
        where ``sigma_f = (sqrt(6) / pi) / (T * snr)`` is the ML position
        uncertainty.
    snr : float
        Peak-to-noise ratio of the demodulated tone against the off-lattice
        noise reference.
    used : bool
        Whether the tone entered the final shared-eps fit (a detected locked
        tone that survived consistency rejection).
    drift_control : bool
        ``True`` for an unlocked-clock multiple measured as a control; never
        used in the eps fit.
    """

    f_bb_mhz: float
    k: int
    df_mhz: float
    sigma_mhz: float
    snr: float
    used: bool
    drift_control: bool


@dataclass(frozen=True)
class TimebaseCalibrationResult:
    """Outcome of one scope-timebase self-calibration pass.

    Attributes
    ----------
    epsilon : float
        Fractional scope-timebase scale error (dimensionless). Every measured
        frequency reads ``f_true * (1 + epsilon)``; correct via
        ``f_true = f_measured / (1 + epsilon)``. ``0.0`` when the
        preconditions fail.
    sigma_epsilon : float
        Formal 1-sigma uncertainty on ``epsilon``
        (``1 / sqrt(sum (f / sigma_tot)^2)`` over kept tones). ``inf`` when
        no tones were kept.
    n_used : int
        Number of locked lattice tones in the final eps fit.
    n_detected : int
        Number of locked lattice tones detected above ``snr_min`` (before
        consistency rejection).
    lattice_g_mhz : float
        GCD of the locked clock fundamentals (baseband MHz); the lattice
        spacing. ``0.0`` when no locked clocks were declared.
    tone_reads : tuple of TimebaseToneRead
        Per-tone diagnostic table (locked detections plus drift controls).
    kappa_sys : float
        Systematic per-tone fractional floor used in ``sigma_tot``.
    snr_min : float
        Peak/noise gate applied for tone detection.
    sample_dt_us : float
        FID sample spacing the calibration used (microseconds).
    start_us, end_us : float
        Active-region bounds the calibration sliced (microseconds).
    span_us : float
        Active-region duration ``T`` used for ``sigma_f`` (microseconds).
    preconditions_passed : bool
        ``True`` only when locked clocks were declared, the lattice is valid,
        and at least three tones survived to the fit.
    preconditions_notes : tuple of str
        Per-precondition diagnostics ("ok" or the failing reason).
    """

    epsilon: float
    sigma_epsilon: float
    n_used: int
    n_detected: int
    lattice_g_mhz: float
    tone_reads: Tuple[TimebaseToneRead, ...]
    kappa_sys: float
    snr_min: float
    sample_dt_us: float
    start_us: float
    end_us: float
    span_us: float
    preconditions_passed: bool
    preconditions_notes: Tuple[str, ...] = field(default_factory=tuple)


def _lattice_gcd_mhz(
    locked_freqs_mhz: Sequence[float],
    notes: List[str],
) -> float:
    """GCD of locked clock fundamentals, in baseband MHz.

    Each fundamental must be (close to) an integer number of MHz so the GCD
    is well defined. Non-integral fundamentals are warned-and-skipped. Returns
    ``0.0`` (with a note appended) when no usable integral fundamental
    remains.
    """
    integral: List[int] = []
    for f in locked_freqs_mhz:
        rounded = round(float(f))
        if abs(float(f) - rounded) > 1e-6 or rounded <= 0:
            logger.warning(
                "locked clock fundamental %s MHz is not a positive integer "
                "number of MHz; skipping it for the lattice GCD",
                f,
            )
            notes.append(f"skipped non-integral locked fundamental {f} MHz for lattice")
            continue
        integral.append(int(rounded))
    if not integral:
        return 0.0
    g = integral[0]
    for v in integral[1:]:
        g = gcd(g, v)
    return float(g)


def _build_block_demod(
    t: np.ndarray,
    n_blocks: int,
    scan_half_range_mhz: float,
    scan_step_mhz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the block-time vector and the ML fine-scan matrix.

    Returns ``(block_indices, t_blocks, scan_matrix)`` where ``scan_matrix`` is
    ``exp(-2j*pi * outer(grid, t_blocks))`` over the fine-frequency ``grid``.
    """
    blocks = np.array_split(t, n_blocks)
    t_blocks = np.array([c.mean() for c in blocks])
    grid = np.arange(
        -scan_half_range_mhz,
        scan_half_range_mhz + scan_step_mhz / 2,
        scan_step_mhz,
    )
    scan_matrix = np.exp(-2j * np.pi * np.outer(grid, t_blocks))
    return grid, t_blocks, scan_matrix


def calibrate_timebase_from_fid(
    fid_data: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    locked_freqs_mhz: Sequence[float],
    unlocked_freqs_mhz: Sequence[float] = (),
    kappa_sys: float = DEFAULT_KAPPA_SYS,
    snr_min: float = DEFAULT_SNR_MIN,
    n_blocks: int = DEFAULT_N_BLOCKS,
    scan_half_range_mhz: float = DEFAULT_SCAN_HALF_RANGE_MHZ,
    scan_step_mhz: float = DEFAULT_SCAN_STEP_MHZ,
    nyquist_fraction: float = DEFAULT_NYQUIST_FRACTION,
    max_reject_iters: int = DEFAULT_MAX_REJECT_ITERS,
    reject_sigma: float = DEFAULT_REJECT_SIGMA,
    edge_pin_steps: float = DEFAULT_EDGE_PIN_STEPS,
    n_noise_probes: int = DEFAULT_N_NOISE_PROBES,
    rng_seed: int = 1,
) -> TimebaseCalibrationResult:
    """Measure the scope-timebase scale error ``eps`` from Rb-locked tones.

    Demodulates the active FID at every Rb-locked lattice frequency
    (multiples of ``gcd(locked_freqs_mhz)`` up to ``nyquist_fraction *
    Nyquist``), reads each tone's residual offset by an ML fine-frequency scan
    of the block-averaged demod, and runs an iterative consistency-rejected
    weighted fit of the shared ``eps = df / f_bb``. Multiples of each
    ``unlocked_freqs_mhz`` fundamental are measured the same way as drift
    controls (``drift_control=True``) but never enter the eps fit -- they sit
    at exact rationals in sample space and read offset ~0 rather than
    ``eps * f_bb``, which is exactly the drifting-family discriminant.

    Parameters
    ----------
    fid_data : np.ndarray
        Raw real time-domain FID samples.
    sample_dt_us : float
        Sample spacing (microseconds).
    start_us, end_us : float
        Active-region bounds (microseconds); the slice ``[start_us, end_us)``
        is used, with time re-zeroed to the slice start.
    locked_freqs_mhz : sequence of float
        Rb-locked clock fundamentals (MHz). Their integer-MHz GCD defines the
        lattice. Empty => preconditions fail.
    unlocked_freqs_mhz : sequence of float, optional
        Unlocked clock fundamentals (MHz) to measure as drift controls.
    kappa_sys : float
        Systematic per-tone fractional floor folded into ``sigma_tot``.
    snr_min : float
        Peak/noise gate for a tone to count as detected.

    Returns
    -------
    TimebaseCalibrationResult
    """
    notes: List[str] = []

    x = np.asarray(fid_data, dtype=float)
    dt = float(sample_dt_us)
    nyq = 0.5 / dt

    i0 = int(float(start_us) / dt)
    i1 = min(int(float(end_us) / dt), x.size)
    i0 = max(i0, 0)
    seg = x[i0:i1]
    t = (np.arange(i0, i1) - i0) * dt
    span = float(t[-1]) if t.size > 1 else 0.0

    g = _lattice_gcd_mhz(locked_freqs_mhz, notes)

    def _empty_result(passed: bool) -> TimebaseCalibrationResult:
        return TimebaseCalibrationResult(
            epsilon=0.0,
            sigma_epsilon=float("inf"),
            n_used=0,
            n_detected=0,
            lattice_g_mhz=g,
            tone_reads=tuple(),
            kappa_sys=float(kappa_sys),
            snr_min=float(snr_min),
            sample_dt_us=dt,
            start_us=float(start_us),
            end_us=float(end_us),
            span_us=span,
            preconditions_passed=passed,
            preconditions_notes=tuple(notes),
        )

    if g <= 0:
        notes.append(
            "no locked clocks declared (or none integral): cannot build the "
            "Rb-locked lattice; declare spur.clocks with locked fundamentals"
        )
        return _empty_result(False)
    if span <= 0 or seg.size < 16:
        notes.append("active region too short to demodulate any tone")
        return _empty_result(False)

    grid, _t_blocks, scan_matrix = _build_block_demod(
        t, n_blocks, scan_half_range_mhz, scan_step_mhz
    )

    def scan(fbb: float) -> Tuple[float, float]:
        """ML fine-frequency scan of the block-averaged demod at ``fbb``.

        Returns ``(offset_mhz, peak_amplitude)`` where ``offset_mhz`` is the
        residual frequency offset (with quadratic peak interpolation) from the
        nominal demod frequency ``fbb``.
        """
        z = seg * np.exp(-2j * np.pi * fbb * t)
        zb = np.array([c.mean() for c in np.array_split(z, n_blocks)])
        amp = np.abs(scan_matrix @ zb)
        j = int(np.argmax(amp))
        if 0 < j < grid.size - 1:
            denom = amp[j - 1] - 2 * amp[j] + amp[j + 1]
            joff = 0.5 * (amp[j - 1] - amp[j + 1]) / denom if denom != 0 else 0.0
        else:
            joff = 0.0
        return float(grid[j] + joff * scan_step_mhz), float(amp[j])

    # Noise reference: 25th percentile of the scan peak at off-lattice
    # quasi-random frequencies (fixed seed, matching the prototype).
    rng = np.random.default_rng(rng_seed)
    k_hi_probe = max(int(nyq / g) - 2, 9)
    probe_amps = [
        scan(g * (k + 0.5) + rng.uniform(-20.0, 20.0))[1]
        for k in rng.integers(8, k_hi_probe, size=n_noise_probes)
    ]
    noise_ref = float(np.percentile(probe_amps, 25))
    if not noise_ref > 0:
        notes.append("noise reference is non-positive; cannot form SNRs")
        return _empty_result(False)

    sigma_f_coef = (np.sqrt(6.0) / np.pi) / span  # MHz at snr = 1

    # --- locked lattice tones ----------------------------------------------
    k_max = int(np.floor((nyquist_fraction * nyq) / g))
    detected: List[TimebaseToneRead] = []
    edge_tol = edge_pin_steps * scan_step_mhz
    for k in range(1, k_max + 1):
        fbb = k * g
        df, pk = scan(fbb)
        snr = pk / noise_ref
        if snr < snr_min:
            continue
        # Edge-pinned maxima are scan artifacts, not tones.
        if abs(abs(df) - scan_half_range_mhz) < edge_tol:
            continue
        sigma_f = sigma_f_coef / snr
        sigma_tot = float(np.hypot(sigma_f, kappa_sys * fbb))
        detected.append(
            TimebaseToneRead(
                f_bb_mhz=float(fbb),
                k=int(k),
                df_mhz=float(df),
                sigma_mhz=sigma_tot,
                snr=float(snr),
                used=False,
                drift_control=False,
            )
        )

    n_detected = len(detected)

    # --- iterative consistency-rejected shared-eps fit ---------------------
    # Seed eps from the single maximum-weight tone -- weight (f/sigma_tot)^2
    # saturates at (1/kappa_sys)^2 for strong tones, so the seed is the
    # strongest *high-leverage* tone (the out-of-band 3x5120 anchor on the
    # home instrument), never a strong low-baseband contaminant (its leverage
    # is ~50x smaller). Each iteration then re-selects the consistent set from
    # ALL detected tones (membership is recomputed, not destructively shrunk,
    # so one bad iterate cannot walk the fit out of the consensus basin) and
    # refits the weighted mean over that set, to a fixed point. Measured on
    # the seven fixtures: converges to the strong-tone consensus everywhere
    # while rejecting the low-baseband non-eps family and in-band molecular
    # contaminants.
    work: List[TimebaseToneRead] = []
    epsilon = 0.0
    if detected:
        f_all = np.array([w.f_bb_mhz for w in detected])
        d_all = np.array([w.df_mhz for w in detected])
        s_all = np.array([w.sigma_mhz for w in detected])
        w2_all = (f_all / s_all) ** 2
        seed = int(np.argmax(w2_all))
        epsilon = float(d_all[seed] / f_all[seed])
        keep_mask: np.ndarray = np.zeros(len(detected), dtype=bool)
        keep_mask[seed] = True
        for _ in range(int(max_reject_iters)):
            new_mask = np.abs(d_all - epsilon * f_all) <= reject_sigma * s_all
            if not new_mask.any():
                break
            new_eps = float(
                np.sum(w2_all[new_mask] * (d_all[new_mask] / f_all[new_mask]))
                / np.sum(w2_all[new_mask])
            )
            converged = bool(np.array_equal(new_mask, keep_mask))
            keep_mask = new_mask
            epsilon = new_eps
            if converged:
                break
        work = [w for w, m in zip(detected, keep_mask) if m]

    kept_freqs = set(id(w) for w in work)
    if len(work) >= 1:
        f_kept = np.array([w.f_bb_mhz for w in work])
        s_kept = np.array([w.sigma_mhz for w in work])
        sigma_epsilon = float(1.0 / np.sqrt(np.sum((f_kept / s_kept) ** 2)))
    else:
        sigma_epsilon = float("inf")

    # --- drift controls (unlocked-clock multiples) -------------------------
    controls: List[TimebaseToneRead] = []
    for fund in unlocked_freqs_mhz:
        fund_f = float(fund)
        if not fund_f > 0:
            continue
        k_ctrl_max = int(np.floor((nyquist_fraction * nyq) / fund_f))
        for k in range(1, k_ctrl_max + 1):
            fbb = k * fund_f
            df, pk = scan(fbb)
            snr = pk / noise_ref
            if snr < snr_min:
                continue
            if abs(abs(df) - scan_half_range_mhz) < edge_tol:
                continue
            sigma_f = sigma_f_coef / snr
            sigma_tot = float(np.hypot(sigma_f, kappa_sys * fbb))
            controls.append(
                TimebaseToneRead(
                    f_bb_mhz=float(fbb),
                    k=int(k),
                    df_mhz=float(df),
                    sigma_mhz=sigma_tot,
                    snr=float(snr),
                    used=False,
                    drift_control=True,
                )
            )

    # Re-stamp ``used`` on the detected locked tones that survived rejection.
    tone_reads: List[TimebaseToneRead] = []
    for tone in detected:
        used = id(tone) in kept_freqs
        tone_reads.append(
            TimebaseToneRead(
                f_bb_mhz=tone.f_bb_mhz,
                k=tone.k,
                df_mhz=tone.df_mhz,
                sigma_mhz=tone.sigma_mhz,
                snr=tone.snr,
                used=used,
                drift_control=False,
            )
        )
    tone_reads.extend(controls)

    n_used = sum(1 for tr in tone_reads if tr.used)
    if n_used < 3:
        notes.append(
            f"only {n_used} locked tones survived to the eps fit (need >= 3); "
            "eps estimate is unreliable"
        )
        passed = False
    else:
        notes.append("ok")
        passed = True

    return TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=sigma_epsilon,
        n_used=n_used,
        n_detected=n_detected,
        lattice_g_mhz=g,
        tone_reads=tuple(tone_reads),
        kappa_sys=float(kappa_sys),
        snr_min=float(snr_min),
        sample_dt_us=dt,
        start_us=float(start_us),
        end_us=float(end_us),
        span_us=span,
        preconditions_passed=passed,
        preconditions_notes=tuple(notes),
    )
