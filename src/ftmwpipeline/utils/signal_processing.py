"""FID apodization and windowing utilities.

The FT the main pipeline persists is unconditionally unapodized,
un-windowed, and native-length -- apodization trades resolution and biases the
line shape, and zero-padding interpolates the bins and corrupts the per-bin
noise/χ² statistics the later stages depend on. These helpers keep the
FID-domain apodization, matched-filter, and zero-padding capability available in
one place for the things that legitimately need it -- the windowed fit-view
visualization, the Stage 3 detection spectra, and the research prototypes -- so
there is a single source of truth and the main pipeline stays apodization-free.
"""

from typing import Any, Optional, cast

import numpy as np
from scipy.signal import get_window

# Apodization vocabulary for the windowed views. Window specs are forwarded to
# ``scipy.signal.get_window`` (the same vocabulary Blackchirp's ``BCFid.ft``
# uses), so any scipy window is available: 'boxcar', 'hann', 'hamming',
# 'blackman', 'blackmanharris', 'bartlett', 'kaiser:14', 'gaussian:50',
# 'tukey:0.3', etc. (':' separates a parameterized window's float args). These
# symmetric windows taper the FID's high-SNR start as well as the trailing
# edge -- they are the conventional leakage-suppression tools, offered so a
# result can be compared against the symmetric-window version a user is
# accustomed to. The one custom, non-scipy spec is 'exp' (exp(-t/W)): the FID
# matched filter, which keeps the start intact and only damps the trailing
# edge. (This pipeline's robust fit is the intended alternative to apodizing.)
APODIZATION_EXAMPLES = ("exp", "boxcar", "hann", "hamming", "blackman", "kaiser:14")


def make_apodization(
    spec: str,
    t_us: np.ndarray,
    *,
    width_us: Optional[float] = None,
    default_width_us: Optional[float] = None,
) -> np.ndarray:
    """Real apodization window ``w(t)`` on the active-region time grid.

    Parameters
    ----------
    spec : str
        ``"exp"`` (or ``"exponential"``) for the custom FID matched filter
        ``exp(-t/W)`` (start-anchored). Anything else is forwarded to
        :func:`scipy.signal.get_window`: a bare name (``"hann"``, ``"hamming"``,
        ``"blackman"``, ``"boxcar"``, ...) or a parameterized window written with
        ``':'``-separated float args (``"kaiser:14"``, ``"gaussian:50"``,
        ``"tukey:0.3"`` -> ``("kaiser", 14.0)`` etc.).
    t_us : np.ndarray
        Active-region time grid (µs, ``t = 0`` at the active start).
    width_us : float, optional
        Time constant ``W`` (µs) for ``exp``; falls back to ``default_width_us``.
    default_width_us : float, optional
        Width used when ``width_us`` is ``None`` (e.g. the window's fitted τ, so
        ``exp`` defaults to the matched filter).
    """
    t = np.asarray(t_us, dtype=float)
    key = spec.strip()
    low = key.lower()
    if low in ("exp", "exponential") and ":" not in key:
        w = width_us if width_us is not None else default_width_us
        if w is None or w <= 0:
            raise ValueError("exp apodization needs a positive width (--apodize-us)")
        return cast(np.ndarray, np.exp(-t / float(w)))
    if ":" in key:
        parts = key.split(":")
        try:
            gw_spec: Any = (parts[0], *(float(p) for p in parts[1:]))
        except ValueError as e:
            raise ValueError(f"bad apodization spec {spec!r}: {e}") from e
    else:
        gw_spec = low
    try:
        return cast(np.ndarray, get_window(gw_spec, t.size).astype(float))
    except Exception as e:  # scipy raises ValueError on unknown / underspecified
        raise ValueError(
            f"unknown apodization {spec!r}: {e} (use 'exp' or a "
            f"scipy.signal.get_window spec, e.g. {APODIZATION_EXAMPLES})"
        ) from e


def matched_filter_window(
    t_us: np.ndarray,
    tau_us: float,
    *,
    shape: str = "lorentzian",
) -> np.ndarray:
    """Start-anchored matched-filter envelope on the active-region time grid.

    A Lorentzian instrument (``exp(-t/τ)`` FID) is matched by ``exp(-t/τ)``; a
    Gaussian one (``exp(-(t/τ)²)`` FID) by ``exp(-(t/τ)²)``. The same envelope
    convention the Stage 5 fit and the Stage 3 gap-pass detector use, kept here
    as the single definition.

    Parameters
    ----------
    t_us : np.ndarray
        Active-region time grid (µs, ``t = 0`` at the active start).
    tau_us : float
        Matched decay constant (µs).
    shape : str, default ``"lorentzian"``
        ``"gaussian"`` selects the ``exp(-(t/τ)²)`` envelope; anything else the
        ``exp(-t/τ)`` Lorentzian envelope.
    """
    t = np.asarray(t_us, dtype=float)
    if tau_us <= 0:
        raise ValueError("matched-filter window needs a positive tau_us")
    if shape == "gaussian":
        return cast(np.ndarray, np.exp(-((t / float(tau_us)) ** 2)))
    return cast(np.ndarray, np.exp(-t / float(tau_us)))


def apodize_fid(
    data: np.ndarray,
    time_us: np.ndarray,
    *,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    expf_us: Optional[float] = None,
    window_function: Optional[str] = None,
    zpf: int = 0,
    rdc: bool = True,
) -> np.ndarray:
    """Apodize and zero-pad a real FID, returning the processed time series.

    The capability the main FT path deliberately dropped, kept here for the
    windowed views and research prototypes. The sequence (active-region only,
    everything outside ``[start_us, end_us]`` zeroed first) is:

    1. exponential matched filter ``exp(-t/expf_us)``,
    2. symmetric window ``window_function`` via :func:`make_apodization`,
    3. DC removal,
    4. zero-pad to ``2**(floor(log2(N)) + 1 + zpf)`` when ``zpf > 0``.

    Parameters
    ----------
    data : np.ndarray
        Real FID samples.
    time_us : np.ndarray
        Sample times (µs), same length as ``data``.
    start_us, end_us : float, optional
        Active-region bounds; ``None`` means the corresponding record edge.
    expf_us : float, optional
        Exponential matched-filter time constant (µs). ``None`` disables it.
    window_function : str, optional
        Symmetric window name (any :func:`make_apodization` ``get_window`` spec).
        ``None`` disables it.
    zpf : int, default 0
        Zero-padding factor (powers of two). ``0`` leaves the length native.
    rdc : bool, default True
        Remove the active-region DC offset after windowing.
    """
    data = np.asarray(data, dtype=float)
    time_us = np.asarray(time_us, dtype=float)

    start_idx = 0
    end_idx = len(data)
    if start_us is not None:
        start_idx = int(np.searchsorted(time_us, start_us))
    if end_us is not None:
        end_idx = int(np.searchsorted(time_us, end_us))

    windowed = data.copy()
    if start_idx > 0:
        windowed[:start_idx] = 0.0
    if end_idx < len(windowed):
        windowed[end_idx:] = 0.0

    active = start_idx < end_idx
    if expf_us is not None and active:
        active_time = time_us[start_idx:end_idx]
        rel = active_time - active_time[0]
        windowed[start_idx:end_idx] *= np.exp(-rel / float(expf_us))

    if window_function is not None and active:
        rel = time_us[start_idx:end_idx] - time_us[start_idx]
        windowed[start_idx:end_idx] *= make_apodization(window_function, rel)

    if rdc and active:
        windowed[start_idx:end_idx] -= np.mean(windowed[start_idx:end_idx])

    final = windowed
    if zpf > 0:
        if len(final) == 0:
            n_padded = 2**zpf
        else:
            n_padded = 2 ** (int(np.log2(len(final))) + 1 + zpf)
        padded = np.zeros(n_padded, dtype=float)
        padded[: len(final)] = final
        final = padded
    return cast(np.ndarray, final)
