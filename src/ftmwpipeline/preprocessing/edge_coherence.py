"""
Complex-edge coherence statistic for Stage 4 window assignment.

The Stage 4 windowing stage must decide, for any candidate window edge, whether
the points on the noise side of that edge are clean background or are still
carrying a strong line's coherent leakage skirt. A magnitude-domain test fails
here: the finite-acquisition leakage envelope is *phase-coherent* and its
magnitude passes through zero at every sinc zero, so the spectrum can look like
noise (in magnitude) while still carrying fully coherent leakage. The
discriminator is the phase, and this module provides the complex-domain test
that uses it.

For an M-point band of complex spectrum values ``z`` with per-bin complex noise
RMS ``sigma`` the statistic is

    S_coh(z; sigma) = |sum_k z_k| / (sigma * sqrt(M))

Under the null (band carries only noise) ``S_coh`` has mean ``sqrt(pi/4) ~=
0.886`` and is independent of M; under a coherent leakage tail of per-bin
amplitude ``L`` it grows like ``(L/sigma) * sqrt(M)``. The threshold ``T_edge``
therefore corresponds to a per-bin leakage of ``L/sigma = T_edge/sqrt(M)``: the
default ``T_edge = 8`` at ``M = 64`` flags coherent leakage that is at least
~1σ per bin (the D8 recalibration -- see
``dev-docs/planning/leakage-detection-rework.md``). The research report's
original ``3`` is still safe on the null (< 1% per-band false positives) but
flags sub-noise leakage. The statistic's derivation and calibration are in
``dev-docs/research/complex-edge-coherence/report.md``.

The functions here are pure (arrays in, arrays out) so they stay unit-testable;
file orchestration lives in :mod:`ftmwpipeline._internal.stage4_impl`.
"""

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

# Locked operating point (research report sections 3 and 5). All three are
# Stage 4 parameters configurable on the pipeline file; these are the
# empirically-validated defaults.
DEFAULT_EDGE_M = 64
"""Band width for the rolling first-pass scan (null-tightest, cache-sized)."""

DEFAULT_TRIM_M = 32
"""Band width for trim-point refinement after a flag (finer spatial scale)."""

DEFAULT_EDGE_THRESHOLD = 8.0
"""``S_coh`` threshold ``T_edge`` separating leakage-touched from line-free
regions. ``8 = sqrt(M)`` at the default ``M = 64`` -- fires on coherent leakage
of at least ~1σ per bin (D8 recalibration). The Stage 3 gap mask passes a
higher value; see leakage-detection-rework.md."""

# Closed-form null moments of S_coh (M-independent), for regression tests.
NULL_MEAN = float(np.sqrt(np.pi / 4.0))  # ~= 0.8862
NULL_VAR = float(1.0 - np.pi / 4.0)  # ~= 0.2146


def coherence_statistic(z: np.ndarray, sigma: float) -> float:
    """Complex-edge coherence statistic ``S_coh`` of one M-point band.

    Parameters
    ----------
    z : np.ndarray
        Complex spectrum values of the band (1D, length M).
    sigma : float
        Per-bin complex noise RMS for the band (a single scalar; callers pass
        the local window-mean of the per-point ``rms_noise`` array).

    Returns
    -------
    float
        ``|sum z| / (sigma * sqrt(M))``. Returns ``0.0`` for an empty band or
        a non-positive ``sigma`` (degenerate, nothing to test).
    """
    z = np.asarray(z)
    m = z.size
    if m == 0 or sigma <= 0.0:
        return 0.0
    return float(np.abs(np.sum(z)) / (sigma * np.sqrt(m)))


def max_cumsum_statistic(z: np.ndarray, sigma: float) -> Tuple[float, int]:
    """Max-cumsum variant ``S_cum`` and the index of its hot spot.

    ``S_cum = max_t |sum_{k<=t} z_k| / (sigma * sqrt(t))``. Where ``S_coh``
    asks "is this band coherent on average", ``S_cum`` amplifies a coherent
    *sub-stretch* — used by Stage 4 to locate the precise edge of a coherent
    region inside an already-flagged band.

    Parameters
    ----------
    z : np.ndarray
        Complex spectrum values of the band (1D).
    sigma : float
        Per-bin complex noise RMS for the band.

    Returns
    -------
    tuple of (float, int)
        ``(S_cum, t_hot)`` where ``t_hot`` is the 0-based index of the band
        element at which the running statistic is maximized — i.e. the edge of
        the coherent stretch. ``(0.0, 0)`` for an empty band / degenerate
        sigma.
    """
    z = np.asarray(z)
    m = z.size
    if m == 0 or sigma <= 0.0:
        return 0.0, 0
    partial = np.abs(np.cumsum(z))
    t = np.arange(1, m + 1, dtype=float)
    vals = partial / (sigma * np.sqrt(t))
    hot = int(np.argmax(vals))
    return float(vals[hot]), hot


def rolling_coherence(
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    band_m: int = DEFAULT_EDGE_M,
) -> np.ndarray:
    """Rolling ``S_coh`` across a spectrum, one value per band center.

    A length-M band slides over the spectrum; the statistic of the band
    starting at index ``s`` is assigned to its center ``s + M//2``. The per-band
    ``sigma`` is the mean of ``rms_noise`` over the band (the report's locally-
    varying noise — never a global median). Positions with no full band
    centered on them (the first/last ~M/2 points) are ``NaN``.

    Parameters
    ----------
    complex_spectrum : np.ndarray
        Complex spectrum (1D). Must be the complex FT, not magnitude — the test
        is a phase-coherence test.
    rms_noise : np.ndarray
        Per-point noise RMS, same length as ``complex_spectrum`` (the persisted
        Stage 2 ``rms_noise`` array).
    band_m : int, default 64
        Band width M.

    Returns
    -------
    np.ndarray
        Float array the same length as ``complex_spectrum``; entry ``c`` is the
        ``S_coh`` of the band centered at ``c``, or ``NaN`` near the edges. If
        the spectrum is shorter than ``band_m`` the single whole-spectrum
        statistic is placed at the midpoint and all other entries are ``NaN``.

    Raises
    ------
    ValueError
        If the arrays differ in length or ``band_m`` is not positive.
    """
    z = np.asarray(complex_spectrum, dtype=complex)
    sd = np.asarray(rms_noise, dtype=float)
    if z.shape != sd.shape:
        raise ValueError("complex_spectrum and rms_noise must have equal length")
    if z.ndim != 1:
        raise ValueError("complex_spectrum must be 1-dimensional")
    if band_m <= 0:
        raise ValueError("band_m must be positive")

    n = z.size
    out: np.ndarray = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    if n < band_m:
        sigma = float(np.mean(sd)) if sd.size else 0.0
        out[n // 2] = coherence_statistic(z, sigma)
        return out

    # Band [s, s+M): complex sum via cumulative sums.
    zc = np.concatenate(([0.0 + 0.0j], np.cumsum(z)))
    band_sum = zc[band_m:] - zc[:-band_m]  # length n - M + 1
    sc = np.concatenate(([0.0], np.cumsum(sd)))
    band_sigma = (sc[band_m:] - sc[:-band_m]) / band_m  # length n - M + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.abs(band_sum) / (band_sigma * np.sqrt(band_m))
    stat = np.where(band_sigma > 0.0, stat, 0.0)

    centers = np.arange(band_sum.size) + band_m // 2
    out[centers] = stat
    return out


def active_edge_coherence(
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    *,
    band_m: int = DEFAULT_EDGE_M,
) -> np.ndarray:
    """Rolling ``S_coh`` on the active FT -- the single edge-coherence
    entry point shared by Stages 3 and 4.

    The active FT is the ``dt_us * rfft`` of just the
    ``[start_us, end_us]`` active region, so it is already in the ``[0, T]``
    reference frame: the active signal begins at the transform's own time
    origin. The turn-on phase ramp ``exp(-i 2pi f t0)`` a *full-record* rfft
    would carry -- and that a coherent sum must be de-ramped to remove (see
    :func:`ftmwpipeline.preprocessing.leakage.deramp_to_active_start`) -- is
    absent here by construction, so the coherent sum is taken directly with no
    de-ramp. Routing the Stage 3 detection floor and the Stage 4
    leakage-touched map through this one function keeps that frame convention in
    a single place (de-ramping the active grid would wind the band phase through
    several turns and collapse ``S_coh`` to the null).

    Parameters
    ----------
    complex_spectrum : np.ndarray
        Complex active FT (1-D), in the order the caller scores it (a monotone
        frequency grid, ascending or descending, so each band is contiguous).
    rms_noise : np.ndarray
        Per-bin noise RMS aligned with ``complex_spectrum``.
    band_m : int, default :data:`DEFAULT_EDGE_M`
        Band width M.

    Returns
    -------
    np.ndarray
        Rolling ``S_coh`` (see :func:`rolling_coherence`); ``NaN`` near the
        band-less edges.
    """
    return rolling_coherence(complex_spectrum, rms_noise, band_m=band_m)


def above_threshold_intervals(
    rolling: np.ndarray,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> List[Tuple[int, int]]:
    """Contiguous index runs where the rolling statistic exceeds ``threshold``.

    These are the spectrum's "leakage-touched" regions: a strong line and its
    coherent skirt drive ``S_coh`` above threshold over a contiguous stretch,
    and the report shows the threshold partitions leakage-touched from
    line-free regions cleanly. ``NaN`` entries (spectrum edges) are treated as
    below threshold.

    Parameters
    ----------
    rolling : np.ndarray
        Rolling ``S_coh`` array from :func:`rolling_coherence`.
    threshold : float, default 8.0
        ``S_coh`` threshold ``T_edge``.

    Returns
    -------
    list of tuple of int
        ``(lo, hi)`` inclusive index pairs, ordered by ``lo``, one per
        contiguous above-threshold run.
    """
    r = np.asarray(rolling, dtype=float)
    mask = np.isfinite(r) & (r > threshold)
    if not mask.any():
        return []

    # Run boundaries from the diff of the boolean mask.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def coherence_curve(
    freqs: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Reconstruct the plotted ``S_coh`` curve that drove a Stage 4 partition.

    De-ramps the full-record spectrum to the active turn-on, rolls ``S_coh``
    over the persisted ``edge_m`` band on the frequency-sorted grid, and reads
    the ``edge_threshold`` -- both knobs falling back to the module
    :data:`DEFAULT_EDGE_M` / :data:`DEFAULT_EDGE_THRESHOLD`, so every plotting
    surface shares one set of defaults. This is the display companion to
    :func:`active_edge_coherence` (which scores the already-``[0, T]`` active
    FT and needs no de-ramp).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (MHz) of the persisted user spectrum.
    complex_spectrum : np.ndarray
        Complex full-record FT on ``freqs``.
    rms_noise : np.ndarray
        Per-point noise RMS aligned with ``freqs``.
    parameters : mapping, optional
        Stage 4 plan parameters; ``edge_m``, ``edge_threshold``,
        ``probe_freq_mhz``, and ``start_us`` are read with safe defaults.

    Returns
    -------
    tuple
        ``(ordered_freq, rolling, threshold, edge_m)`` -- the ascending
        frequency grid, the rolling ``S_coh`` on it, the ``T_edge`` threshold,
        and the band width used.
    """
    # Local import: leakage imports from this module, so importing it at module
    # scope would cycle.
    from .leakage import deramp_to_active_start

    params = parameters or {}
    freqs_arr = np.asarray(freqs, dtype=float)
    spec = np.asarray(complex_spectrum, dtype=complex)
    order = np.argsort(freqs_arr)
    edge_m = int(params.get("edge_m", DEFAULT_EDGE_M))
    threshold = float(params.get("edge_threshold", DEFAULT_EDGE_THRESHOLD))
    referenced = deramp_to_active_start(
        freqs_arr,
        spec,
        float(params.get("probe_freq_mhz", 0.0)),
        float(params.get("start_us", 0.0)),
    )
    rolling = rolling_coherence(
        referenced[order],
        np.asarray(rms_noise, dtype=float)[order],
        band_m=edge_m,
    )
    return freqs_arr[order], rolling, threshold, edge_m
