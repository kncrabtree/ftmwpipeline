"""The fitted-model accessors: ``window_model`` and ``spectrum_model``.

Both deliver the persisted Stage 5 fit evaluated on a spectrum grid, so a client
can draw the model -- and check it -- without reimplementing the line shape
(``dev-docs/CONTRACT_STRATEGY.md`` §Window model, §Spectrum model). Every model
array comes from :mod:`ftmwpipeline.fitting.model_eval`, the evaluator the
fit-detail figures draw with, so the delivered model and the drawn one cannot
drift apart.

Two grids are offered:

* ``"active"`` -- the native active-portion FT that Stage 5 fits, restricted to
  the Stage 1 analysis band (the bins a window can reach), with the per-bin
  noise the fit weighted by and the gated-spur mask it applied. Rebuilt from the
  FID and the persisted settings exactly as the fit built it.
* ``"display"`` -- :func:`~ftmwpipeline._internal.stage5_impl.compute_display_ft_impl`'s
  zero-filled grid. It is the same transform (scale and phase origin) at
  ``pad_factor`` x the density and contains every active bin, so the model
  overlays it point for point.

Both reads leave the file byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from ..contract import SPECTRUM_MODEL_SCHEMA, WINDOW_MODEL_SCHEMA, Absent
from ..core.data_structures import FittingResult, Sideband, SpectrumFit
from ..file_manager import (
    BadSettingError,
    IncompleteProvenanceError,
    NotFoundError,
    StageDependencyError,
)
from ..fitting.model_eval import (
    FROZEN_PEAK_PREFIX,
    evaluate_spectrum_model,
    evaluate_window_terms,
    window_center_mhz,
    window_fit_range_mhz,
)
from ..fitting.peak_model import sideband_sign
from .atomic import h5open

__all__ = [
    "MODEL_GRIDS",
    "window_model_impl",
    "spectrum_model_impl",
]

#: The grids a model accessor evaluates on.
MODEL_GRIDS: Tuple[str, ...] = ("active", "display")

#: The frame every model payload is in: the fit's (raw) frame. The calibrated
#: frame applies only to reported line frequencies.
_MODEL_FRAME = "raw"


@dataclass(frozen=True)
class _ModelGrid:
    """One evaluation grid: ascending frequencies, the spectrum on them, and
    (active grid only) the fit's per-bin noise."""

    name: str
    freq_mhz: np.ndarray
    data: np.ndarray
    sigma: Optional[np.ndarray]


@dataclass(frozen=True)
class _ModelContext:
    fit: SpectrumFit
    grid: _ModelGrid
    sideband: Sideband
    acquisition_us: float
    spur_set: Optional[Any]  # SpurSet on the active grid, else None
    fit_epoch: Optional[int] = None  # analysis epoch the Stage 5 fit ran under


#: The first analysis epoch whose fits draw frozen contributors at the window's
#: fitted decay time (ROADMAP D18). Earlier free-tau fits held them at an
#: unrecorded starting tau, so their model cannot be reproduced.
FROZEN_SKIRT_FOLLOWS_TAU_EPOCH = 4


def _fit_epoch(file_path: str) -> Optional[int]:
    """The analysis epoch the persisted Stage 5 fit was produced under."""

    from ..io.environment_serialization import load_stage_environments

    with h5open(file_path, "r") as h5f:
        env = load_stage_environments(h5f).get("stage5_fitting")
    return None if env is None else env.analysis_epoch


def _require_fit(file_path: Union[str, Path], accessor: str) -> None:
    """Gate on the file (typed not-found / corrupt / incompatible) and Stage 5."""
    from .read_impl import _open

    with _open(file_path) as h5f:
        if "stage5_fitting" not in h5f:
            raise StageDependencyError(
                accessor,
                ["stage5_fitting"],
                Path(str(file_path)),
                command="fit run",
                message=(
                    f"No Stage 5 fit found in {file_path}; {accessor} needs one. "
                    "Run 'fit run' first."
                ),
            )


def _fit_spur_set(fit: SpectrumFit, sorted_active_freq: np.ndarray) -> Optional[Any]:
    """The gated-spur set the Stage 5 fit masked with, replayed from the file.

    The catalog is the fit's, at full precision (see
    :func:`~ftmwpipeline._internal.stage5_impl.gated_spur_catalog`). The mask
    geometry (bin spacing, default half width) is the fit's.
    """
    from .stage5_impl import gated_spur_catalog, replay_spur_set

    params: Mapping[str, Any] = fit.parameters or {}
    if not params.get("spur_masking_enabled", False):
        return None
    catalog = gated_spur_catalog(params, fit.diagnostics)
    spur_set = replay_spur_set(catalog, sorted_active_freq)
    return spur_set if spur_set else None


def _load_model_context(
    file_path: Union[str, Path], accessor: str, grid: str
) -> _ModelContext:
    """Load the fit and build the requested grid, read-only."""
    from ..fitting.active_ft import compute_active_ft
    from .active_ft_support import build_active_grid_with_noise
    from .stage5_impl import (
        _build_active_ft_inputs,
        _display_ft_from_inputs,
        load_fit_impl,
    )

    if grid not in MODEL_GRIDS:
        raise BadSettingError(
            "grid",
            f"one of: {', '.join(MODEL_GRIDS)}",
            grid,
            message=f"unknown grid {grid!r}; choose one of {', '.join(MODEL_GRIDS)}",
        )
    _require_fit(file_path, accessor)
    path = str(file_path)
    fit: SpectrumFit = load_fit_impl(path)["fit"]
    inputs = _build_active_ft_inputs(path)
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        n_raw,
        acquisition_us,
        _user_ft,
        trim_range,
    ) = inputs

    spur_set: Optional[Any] = None
    if grid == "active":
        # The fit's own construction: the active FT in the analysis band, and
        # the noise estimated over that band with the persisted Stage 2 knobs
        # (``build_stage5_fit_context``'s in-band ``rms_for_fit``).
        active = compute_active_ft(
            fid_samples,
            sample_dt_us,
            start_us=start_us,
            end_us=end_us,
            probe_freq_mhz=probe_freq_mhz,
            sideband=sideband,
            n_raw=n_raw,
        )
        cft, rms = build_active_grid_with_noise(path, trim_range, active=active)
        order = np.argsort(cft.freq_array)
        model_grid = _ModelGrid(
            name=grid,
            freq_mhz=np.ascontiguousarray(np.asarray(cft.freq_array)[order]),
            data=np.ascontiguousarray(
                np.asarray(cft.complex_spectrum, dtype=np.complex128)[order]
            ),
            sigma=np.ascontiguousarray(np.asarray(rms, dtype=float)[order]),
        )
        spur_set = _fit_spur_set(fit, np.sort(np.asarray(active.freq_mhz)))
    else:
        display = _display_ft_from_inputs(path, inputs)
        model_grid = _ModelGrid(
            name=grid,
            freq_mhz=np.asarray(display.freq_array, dtype=float),
            data=np.asarray(display.complex_spectrum, dtype=np.complex128),
            sigma=None,
        )
    return _ModelContext(
        fit=fit,
        grid=model_grid,
        sideband=Sideband.coerce(sideband),
        acquisition_us=float(acquisition_us),
        spur_set=spur_set,
        fit_epoch=_fit_epoch(path),
    )


def _window_fit(fit: SpectrumFit, window_id: int) -> FittingResult:
    for wf in fit.window_fits:
        if wf.window_id is not None and int(wf.window_id) == int(window_id):
            return wf
    raise NotFoundError(
        "window",
        [int(window_id)],
        message=f"window {window_id} has no Stage 5 fit in this file",
    )


def _excluded_bins(
    ctx: _ModelContext, wf: FittingResult, freq: np.ndarray
) -> np.ndarray:
    """The bins of ``freq`` (the window's slice) the fit left out of its cost.

    The window's gated-spur mask, exactly as ``fit_window`` applied it: a mask
    that would drop every bin is ignored there, so it is here.
    """
    mask = np.zeros(freq.shape, dtype=bool)
    if ctx.spur_set is None:
        return mask
    lo, hi = window_fit_range_mhz(wf)
    center = window_center_mhz(wf)
    spec = ctx.spur_set.window_mask_spec(lo, hi, center, ctx.sideband)
    if spec is None:
        return mask
    u = sideband_sign(ctx.sideband) * (freq - center)
    mask = np.asarray(spec.bin_mask(u), dtype=bool)
    if mask.all():
        mask = np.zeros(freq.shape, dtype=bool)
    return mask


def window_model_impl(
    file_path: Union[str, Path],
    window_id: int,
    *,
    grid: str = "active",
    components: bool = False,
) -> Dict[str, Any]:
    """The ``ftmw/window_model@1`` payload of one window (see the module doc).

    Raises
    ------
    StageDependencyError
        No Stage 5 fit in the file (``command`` ``"fit run"``).
    NotFoundError
        ``window_id`` has no Stage 5 fit (``kind`` ``"window"``).
    ValueError
        ``grid`` is not one of :data:`MODEL_GRIDS`.
    """
    ctx = _load_model_context(file_path, "window_model", grid)
    return _window_payload(ctx, window_id, components=components)


def _window_payload(
    ctx: _ModelContext, window_id: int, *, components: bool
) -> Dict[str, Any]:
    """:func:`window_model_impl` on an already-built context."""
    wf = _window_fit(ctx.fit, window_id)
    _require_reproducible_frozen_skirt(ctx, wf, window_id)
    lo, hi = window_fit_range_mhz(wf)
    in_range = (ctx.grid.freq_mhz >= lo) & (ctx.grid.freq_mhz <= hi)
    freq = np.ascontiguousarray(ctx.grid.freq_mhz[in_range])
    terms = evaluate_window_terms(
        freq,
        wf,
        acquisition_us=ctx.acquisition_us,
        sideband=ctx.sideband,
        components=components,
    )

    sigma: Union[np.ndarray, Absent] = Absent.UNDEFINED
    excluded: Union[np.ndarray, Absent] = Absent.UNDEFINED
    if ctx.grid.sigma is not None:
        sigma = np.ascontiguousarray(ctx.grid.sigma[in_range])
        excluded = _excluded_bins(ctx, wf, freq)

    comp: Union[Dict[int, np.ndarray], Absent] = Absent.NOT_RUN
    if components:
        uids = [p.peak_uid for p in wf.fitted_peaks]
        if terms.components is None or any(u is None for u in uids):
            # A line without an identity cannot be keyed.
            comp = Absent.UNDEFINED
        else:
            comp = {int(u): c for u, c in zip(uids, terms.components) if u is not None}

    return {
        "schema": WINDOW_MODEL_SCHEMA,
        "window_id": int(window_id),
        "grid": ctx.grid.name,
        "frame": _MODEL_FRAME,
        "frequency_mhz": freq,
        "data": np.ascontiguousarray(ctx.grid.data[in_range]),
        "model": terms.model,
        "fixed": terms.fixed,
        "baseline": terms.baseline if terms.baseline is not None else Absent.NOT_RUN,
        "sigma": sigma,
        "excluded": excluded,
        "components": comp,
    }


def _require_reproducible_frozen_skirt(
    ctx: _ModelContext, wf: FittingResult, window_id: int
) -> None:
    """Refuse a window whose fit held its frozen contributors at an unrecorded tau.

    Before :data:`FROZEN_SKIRT_FOLLOWS_TAU_EPOCH`, a window with frozen
    contributors and a free decay time subtracted their skirt once, at the
    starting tau, and never recorded it; drawing it at the fitted tau would
    present a model the fit never minimised.
    """
    has_frozen = any(k.startswith(FROZEN_PEAK_PREFIX) for k in wf.fixed_parameters)
    tau_free = (wf.shared_parameters.get("tau_us") or {}).get("fitted") is not False
    epoch = ctx.fit_epoch
    if not (has_frozen and tau_free):
        return
    if epoch is not None and epoch >= FROZEN_SKIRT_FOLLOWS_TAU_EPOCH:
        return
    raise IncompleteProvenanceError(
        [f"stage5_fitting.window[{int(window_id)}].frozen_skirt_tau_us"],
        message=(
            f"Window {int(window_id)} was fitted with frozen contributors and a "
            f"free decay time under analysis epoch {epoch}, which held their "
            f"skirt at a starting decay time it did not record, so its model "
            f"cannot be reproduced. Re-run 'fit run'."
        ),
    )


def spectrum_model_impl(
    file_path: Union[str, Path], *, grid: str = "active"
) -> Dict[str, Any]:
    """The ``ftmw/spectrum_model@1`` payload (see the module doc).

    Every line of the persisted fit, once, over the whole grid at its window's
    fitted decay time, plus each window's baseline inside its own fit range.

    Raises
    ------
    StageDependencyError
        No Stage 5 fit in the file (``command`` ``"fit run"``).
    ValueError
        ``grid`` is not one of :data:`MODEL_GRIDS`.
    """
    ctx = _load_model_context(file_path, "spectrum_model", grid)
    freq = ctx.grid.freq_mhz
    model = evaluate_spectrum_model(
        freq, ctx.fit, acquisition_us=ctx.acquisition_us, sideband=ctx.sideband
    )
    data = ctx.grid.data
    return {
        "schema": SPECTRUM_MODEL_SCHEMA,
        "grid": ctx.grid.name,
        "frame": _MODEL_FRAME,
        "frequency_mhz": freq,
        "data": data,
        "model": model,
        "residual": data - model,
    }
