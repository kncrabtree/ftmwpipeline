"""Shared construction of the canonical active-portion FT.

The active FT -- the ``dt_us * rfft`` of just the ``[start_us, end_us]`` active
samples, unpadded -- is the single domain in which all real processing
(Stages 2/3/5) measures noise, scores, and fits. The front-zeroed, full-length
persisted Stage 1 spectrum exists only for the Stage 0/1 start-time comparison
view; it is never the substrate for noise, detection, or fitting.

This module owns the file-bound construction of that active FT from the
persisted FID plus the canonical Stage 1 settings, so Stage 2 (noise authority),
Stage 3 (detection snap/score grid), and Stage 5 (fit) all build the *same*
spectrum from one place. The pure array math lives in
:func:`ftmwpipeline.fitting.active_ft.compute_active_ft`; the per-bin noise
estimator is :func:`ftmwpipeline.preprocessing.noise_estimation.estimate_active_ft_noise`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import h5py
import numpy as np

from ..core.data_structures import ComplexFT, Sideband
from ..core.noise_settings import resolve as resolve_noise_settings
from ..fitting.active_ft import ActiveFTResult, compute_active_ft
from ..io.noise_result_serialization import load_noise_result_from_hdf5
from ..io.noise_settings_serialization import load_noise_settings_from_h5
from ..preprocessing.noise_estimation import NoiseResult, estimate_active_ft_noise
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import compute_ft_impl


def _resolve_sideband(value: Sideband | str) -> Sideband:
    """Coerce a stored sideband (enum or string) to the :class:`Sideband` enum."""
    if isinstance(value, Sideband):
        return value
    return Sideband(str(value).lower())


def _active_acquisition_us(
    fid_duration_us: float, start_us: Optional[float], end_us: Optional[float]
) -> float:
    """Effective acquisition length T (µs) of the analysed FID window."""
    lo = 0.0 if start_us is None else float(start_us)
    hi = fid_duration_us if end_us is None else float(end_us)
    return max(hi - lo, 0.0)


def compute_canonical_active_ft(
    file_path: str,
    *,
    expf_us: Optional[float] = None,
) -> ActiveFTResult:
    """Build the canonical active FT for a pipeline file.

    Loads the persisted FID and the canonical Stage 1 processing parameters,
    then rffts just the ``[start_us, end_us]`` active region (no zero-padding)
    via :func:`compute_active_ft`. ``expf_us`` defaults to ``None`` (boxcar /
    unapodized) -- the authority domain. Pass the Stage 1 ``expf_us`` only when
    a caller deliberately wants the apodized active FT.

    Requires Stage 1 (canonical FT settings) to be present.
    """
    fid = load_fid_from_pipeline_impl(file_path)
    stage1 = compute_ft_impl(file_path=file_path)
    user_ft = stage1["complex_ft"]
    base_pp = user_ft.metadata["processing_params"]

    sample_dt_us = float(fid.spacing * 1e6)
    start_us = float(base_pp.start_us) if base_pp.start_us is not None else 0.0
    end_us = (
        float(base_pp.end_us)
        if base_pp.end_us is not None
        else float(fid.duration_us)
    )
    acquisition_us = _active_acquisition_us(
        fid.duration_us, base_pp.start_us, base_pp.end_us
    )
    if acquisition_us <= 0:
        raise ValueError(
            f"Stage 1 canonical settings produce a non-positive active "
            f"acquisition length ({acquisition_us} us)"
        )

    # n_padded: the zero-padded FFT length the persisted full-record FT uses.
    # Recorded on the result as ``alpha`` for diagnostics only; the active FT
    # itself is unpadded.
    n_active_estimate = int(round(acquisition_us / sample_dt_us))
    n_padded = max(
        2 ** (int(np.log2(max(n_active_estimate, 1))) + 1 + int(base_pp.zpf)),
        n_active_estimate,
    )

    return compute_active_ft(
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        expf_us=expf_us,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=_resolve_sideband(fid.sideband),
        n_padded=n_padded,
    )


def estimate_canonical_active_ft_noise(
    file_path: str,
    **scatter_kwargs: Any,
) -> Tuple[NoiseResult, ActiveFTResult]:
    """Build the canonical unapodized active FT and estimate its per-bin σ.

    The Stage 2 noise authority: returns the scatter-estimated
    :class:`NoiseResult` on the active-FT bin order together with the
    :class:`ActiveFTResult` it was measured on (so the caller can persist the
    σ against the same grid it lives on).
    """
    active_ft = compute_canonical_active_ft(file_path, expf_us=None)
    noise = estimate_active_ft_noise(
        active_ft.freq_mhz, active_ft.complex_spectrum, **scatter_kwargs
    )
    return noise, active_ft


def _persisted_scatter_knobs(file_path: str) -> dict:
    """Resolve the persisted Stage 2 scatter knobs (non-None only).

    Returns the resolved scatter parameters so the active-FT authority noise
    is measured with the *same* knobs Stage 2 used on the full-record FT.
    Fields left None by the resolver fall through to
    :func:`estimate_noise_scatter`'s defaults (= Stage 2's hard defaults).
    """
    resolved = resolve_noise_settings(
        explicit=None,
        preset=None,
        persisted=load_noise_settings_from_h5(file_path),
        recommended=None,
    )
    candidates = {
        "window_mhz": resolved.window_mhz,
        "pedestal_mhz": resolved.pedestal_mhz,
        "line_k": resolved.line_k,
        "n_iter": resolved.n_iter,
        "region_aware": resolved.region_aware,
        "smoothing_mhz": resolved.smoothing_mhz,
        "smoothing_percentile": resolved.smoothing_percentile,
        "convolve_mhz": resolved.convolve_mhz,
    }
    return {k: v for k, v in candidates.items() if v is not None}


def build_trimmed_active_ft(
    file_path: str,
    trim_range: Optional[Tuple[float, float]] = None,
) -> ComplexFT:
    """Build the canonical unapodized active FT as a (trimmed) :class:`ComplexFT`.

    The single active-grid surface every later stage scores, plans, fits, and
    estimates noise on: the ``dt_us*rfft`` of the active region, wrapped as a
    ComplexFT (so it carries the ``freq_array`` / ``magnitude_spectrum``
    interface), trimmed to the analysis band when ``trim_range`` is given.
    """
    active = compute_canonical_active_ft(file_path, expf_us=None)
    cft = ComplexFT.from_spectrum(active.complex_spectrum, active.freq_mhz)
    if trim_range is not None:
        cft = cft.trim_to_range(trim_range[0], trim_range[1])
    return cft


def build_active_grid_with_noise(
    file_path: str,
    trim_range: Optional[Tuple[float, float]] = None,
) -> Tuple[ComplexFT, np.ndarray]:
    """Build the canonical active FT (trimmed) and its per-bin authority σ.

    The single active-grid scoring/planning surface for Stages 3, 4, and 5:
    the unapodized active FT plus the scatter-estimated per-bin σ_x on that
    same grid, measured with the persisted Stage 2 scatter knobs so it matches
    the canonical noise estimator. σ aligns with ``cft`` element-for-element.

    ``trim_range`` restricts both to the analysis band (mirroring the
    persisted FT's trim), so the authority σ is measured region-aware over the
    same band Stage 2 used.
    """
    cft = build_trimmed_active_ft(file_path, trim_range)
    noise = estimate_active_ft_noise(
        cft.freq_array,
        cft.complex_spectrum,
        **_persisted_scatter_knobs(file_path),
    )
    return cft, np.asarray(noise.rms_noise, dtype=float)


def load_canonical_active_noise(
    file_path: str,
) -> Tuple[ActiveFTResult, np.ndarray]:
    """Load the persisted Stage 2 σ on the canonical active-FT grid.

    Rebuilds the canonical unapodized active FT (the grid Stage 2 measured and
    persisted on) and reads the stored σ back element-for-element. Returns the
    active FT and its per-bin σ_x. This is the single read path every
    σ-consuming stage (3 snap/score, 5 fit weighting) goes through, so they all
    share one grid.

    Raises
    ------
    ValueError
        If Stage 2 (noise estimation) has not been completed.
    """
    active_ft = compute_canonical_active_ft(file_path, expf_us=None)
    with h5py.File(file_path, "r") as h5f:
        if "stage2_noise_result" not in h5f:
            raise ValueError(
                "Stage 2 (noise estimation) must be completed before the "
                "canonical noise can be loaded. Run estimate_noise() first."
            )
        noise = load_noise_result_from_hdf5(
            h5f["stage2_noise_result"],
            active_ft.freq_mhz,
            np.abs(active_ft.complex_spectrum),
        )
    return active_ft, np.asarray(noise.rms_noise, dtype=float)
