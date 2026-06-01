"""Knob registry: the data behind the tuning sweep engine.

A :class:`KnobSpec` declares everything parameter-specific the knob-agnostic
engine needs: how to set the knob and re-run the affected stage (``run``), how
to reduce that stage's result to one or more named metric columns (``metric``),
an optional plot adapter, and an optional recommender (or a simple
``direction`` for the built-in best-value pick). Knobs are addressed by a
dotted settings path (e.g. ``stage2.scatter.window_mhz``) that mirrors the
table in ``dev-docs/planning/instrument-tunable-knobs.md``.

``run`` callables import :mod:`ftmwpipeline.api` lazily so this module carries
no import-time dependency on the API/Pipeline layer (which depends back on the
stage impls), keeping the package import graph acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .plots import (
    plot_noise_sweep,
    plot_spectra_ladder,
    plot_start_detection,
    plot_tau_trend,
)

# A stage runner: given a (writable) .ftmw path and a knob value, set the knob
# and re-run the affected stage, returning the stage result object.
RunFn = Callable[[Path, Any], Any]
# A metric reducer: stage result -> ordered named metric columns.
MetricFn = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class KnobSpec:
    """One tunable parameter and how to sweep/measure/plot/recommend it."""

    path: str
    """Dotted settings path, e.g. ``"stage2.scatter.window_mhz"``."""
    stage: str
    """Human stage label, e.g. ``"stage2_noise"`` / ``"start_detection"``."""
    requires: str
    """Last pipeline stage that must already be present on the input file."""
    help: str
    """One-line physical meaning (mirrors the knob registry)."""
    inst_sensitivity: str
    """Instrument-sensitivity rating: ``"Y"`` / ``"N"`` / ``"maybe"``."""
    default_grid: Tuple[Any, ...]
    """Sweep values used when the caller passes no explicit grid."""
    run: RunFn
    """Set the knob to a value and re-run its stage; return the stage result."""
    metric: MetricFn
    """Reduce a stage result to named metric columns for the table/CSV."""
    metric_columns: Tuple[str, ...]
    """Column order for the metrics emitted by :attr:`metric`."""
    primary_metric: Optional[str] = None
    """Metric column the built-in recommender optimizes (None disables it)."""
    direction: str = "none"
    """``"min"`` / ``"max"`` for the built-in recommender, or ``"none"``."""
    recommend: Optional[Callable[["list[SweepRowLike]"], "Recommendation"]] = None
    """Override recommender; wins over :attr:`direction` when set."""
    plot: Optional[Callable[..., Any]] = None
    """Optional plot adapter ``(spec, rows, ctx) -> Figure | None``; ``None``
    => table-only output for this knob."""
    see_also: Optional[str] = None
    """Optional pointer to a related knob/visualization, shown in non-quiet
    output (e.g. a detection knob pointing at the spectrum-impact knob)."""

    def grid(self, override: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
        """Resolve the grid to sweep: explicit override, else the default."""
        if override is not None:
            return tuple(override)
        return self.default_grid


# Forward-declared structural aliases for recommender typing (the concrete
# types live in engine.py; using Any here avoids a circular import).
SweepRowLike = Any
Recommendation = Any


# ---------------------------------------------------------------------------
# Stage runners (lazy api import to keep the import graph acyclic)
# ---------------------------------------------------------------------------

def _run_start(field_name: str) -> RunFn:
    """Re-run start detection with a single ``StartDetectionSettings`` field set.

    Uses ``stamp=False`` so the sweep never mutates the file's recommended
    ``start_us`` layer; the engine still operates on a working copy.
    """

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

        return ftmw.detect_start_time(path, stamp=False, **{field_name: value})

    return run


def _run_noise(method: str, kwarg: str) -> RunFn:
    """Re-run noise estimation with a single estimator knob set."""

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

        return ftmw.estimate_noise(path, method=method, **{kwarg: value})

    return run


@dataclass(frozen=True)
class FtAtStart:
    """A FT computed at a particular window start, with the start (and the
    chirp-end it was referenced to, if any) retained for the ladder plot.

    Both the ``stage1.start_us`` and ``start.guard_margin_us`` knobs sweep the
    *same* thing — the spectrum as a function of where the FT window starts.
    They differ only in how the start is specified: ``start_us`` absolutely,
    ``guard_margin_us`` as an offset past the (separately detected) chirp end.
    """

    ft: Any
    start_us: float
    chirp_end_us: Optional[float] = None


def _run_ft_start(path: Path, value: Any) -> Any:
    """Recompute the FT at an absolute window start; returns an ``FtAtStart``.

    Other FT settings (trim, zpf, apodization) are inherited from the file's
    resolution chain, so the sweep isolates the effect of ``start_us``.
    """
    import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

    start = float(value)
    return FtAtStart(ft=ftmw.compute_ft(path, start_us=start), start_us=start)


def _run_guard() -> RunFn:
    """Sweep the guard margin: the FT window start is ``chirp_end + guard``.

    The chirp end is detected once per sweep (memoised on the working file's
    identity) since the guard does not affect detection — it only shifts the
    start past the chirp. So the sweep costs one detection plus a fast FT per
    value, not a full detection per value.
    """
    cache: Dict[Any, float] = {}

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        chirp_end = cache.get(key)
        if chirp_end is None:
            cache.clear()
            chirp_end = float(ftmw.detect_start_time(path, stamp=False).chirp_end_us)
            cache[key] = chirp_end
        start = chirp_end + float(value)
        return FtAtStart(
            ft=ftmw.compute_ft(path, start_us=start),
            start_us=start,
            chirp_end_us=chirp_end,
        )

    return run


def _run_tau(sub_block: str, field_name: str) -> RunFn:
    """Re-run the Stage 2b tau calibration with a single settings field set.

    Builds a ``TauCalibrationSettings`` bundle (the preset layer) carrying just
    the one sub-block field, so any tau knob — including the ``polish`` fields
    that ``calibrate_tau`` does not expose as kwargs — is sweepable uniformly.
    """

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle
        from ftmwpipeline.core import tau_calibration_settings as tcs

        sub_cls = {
            "stft": tcs.StftSubSettings,
            "polish": tcs.PolishSubSettings,
        }[sub_block]
        bundle = tcs.TauCalibrationSettings(**{sub_block: sub_cls(**{field_name: value})})
        return ftmw.calibrate_tau(path, settings=bundle)

    return run


# ---------------------------------------------------------------------------
# Metric reducers
# ---------------------------------------------------------------------------

def _metric_start(result: Any) -> Dict[str, Any]:
    return {
        "start_us": round(float(result.start_us), 4),
        "chirp_end_us": round(float(result.chirp_end_us), 4),
        "chirp_detected": bool(result.chirp_detected),
    }


def _metric_ft_band_floor(result: Any) -> Dict[str, Any]:
    """Percentiles of |FT| over the (already-trimmed) active band: a ladder of
    floor markers (p1..p50) plus the peak, so the chirp/ringdown residue is
    readable for both dense and diffuse spectra."""
    import numpy as np

    mag = np.abs(np.asarray(result.ft.complex_spectrum))
    pcts: "np.ndarray" = np.percentile(mag, [1, 5, 10, 20, 50, 100])
    return {
        "p1": round(float(pcts[0]), 5),
        "p5": round(float(pcts[1]), 5),
        "p10": round(float(pcts[2]), 5),
        "p20": round(float(pcts[3]), 5),
        "p50": round(float(pcts[4]), 5),
        "max": round(float(pcts[5]), 5),
    }


def _metric_tau(result: Any) -> Dict[str, Any]:
    return {
        "tau_maj_us": round(float(result.tau_maj_us), 4),
        "sigma_tau_us": round(float(result.sigma_tau_us), 4),
        "n_contributors": int(result.n_contributors),
    }


def _metric_noise(result: Any) -> Dict[str, Any]:
    import numpy as np

    sigma = np.asarray(result.rms_noise, dtype=float)
    mask = np.asarray(result.noise_mask)
    finite = sigma[np.isfinite(sigma)]
    median_sigma = float(np.median(finite)) if finite.size else float("nan")
    return {
        "median_sigma": median_sigma,
        "noise_fraction": round(float(np.mean(mask)) if mask.size else float("nan"), 4),
    }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, KnobSpec] = {}


def _register(spec: KnobSpec) -> None:
    if spec.path in _REGISTRY:
        raise ValueError(f"duplicate knob path: {spec.path!r}")
    _REGISTRY[spec.path] = spec


# Start handling (pre-Stage 1) — requires Stage 0 (FID).
#
# The guard margin is a spectrum-impact knob, not a detection knob: it only
# shifts the FT window start past the (separately detected) chirp end, so it
# shows the same stacked-spectra ladder as stage1.start_us. The detection knobs
# below (sweep_max_us, min_chirp_drop_ratio) are the ones that move the chirp
# end, so they show the Sigma|FT| detection curve.
_DETECTION_SEE_ALSO = (
    "start.guard_margin_us / stage1.start_us — stack the resulting spectra to "
    "see how the detected start affects the FT (chirp/ringdown residue)."
)
_register(KnobSpec(
    path="start.guard_margin_us",
    stage="start_detection",
    requires="stage0_fid_data",
    help="Margin past the chirp end for switch-bounce ringdown (instrument-specific).",
    inst_sensitivity="Y",
    default_grid=(0.3, 0.5, 0.67, 0.85, 1.0),
    run=_run_guard(),
    metric=_metric_ft_band_floor,
    metric_columns=("p1", "p5", "p10", "p20", "p50", "max"),
    plot=plot_spectra_ladder,
))
_register(KnobSpec(
    path="start.sweep_max_us",
    stage="start_detection",
    requires="stage0_fid_data",
    help="Upper bound of the start-time sweep (must clear chirp end + floor tail).",
    inst_sensitivity="Y",
    default_grid=(5.0, 6.0, 7.5, 9.0),
    run=_run_start("sweep_max_us"),
    metric=_metric_start,
    metric_columns=("start_us", "chirp_end_us", "chirp_detected"),
    plot=plot_start_detection,
    see_also=_DETECTION_SEE_ALSO,
))
_register(KnobSpec(
    path="start.min_chirp_drop_ratio",
    stage="start_detection",
    requires="stage0_fid_data",
    help="Min plateau/floor ratio for a chirp collapse to be considered present.",
    inst_sensitivity="Y",
    default_grid=(5.0, 10.0, 20.0, 40.0),
    run=_run_start("min_chirp_drop_ratio"),
    metric=_metric_start,
    metric_columns=("start_us", "chirp_end_us", "chirp_detected"),
    plot=plot_start_detection,
    see_also=_DETECTION_SEE_ALSO,
))

# FT window start time (Stage 1) — sweep the actual start and stack the
# resulting active-band spectra. Requires Stage 1 settings to be resolvable
# (built once); each value recomputes the FT.
_register(KnobSpec(
    path="stage1.start_us",
    stage="stage1_ft",
    requires="stage0_fid_data",
    help="FID window start time for the FT; stack the active-band spectra to "
         "judge the chirp/ringdown residue.",
    inst_sensitivity="Y",
    default_grid=(1.5, 1.7, 1.85, 2.0, 2.15, 2.3, 2.45, 2.6),
    run=_run_ft_start,
    metric=_metric_ft_band_floor,
    metric_columns=("p1", "p5", "p10", "p20", "p50", "max"),
    plot=plot_spectra_ladder,
))

# Stage 2 noise — scatter estimator (the canonical default). Requires Stage 1.
_register(KnobSpec(
    path="stage2.scatter.window_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="Width of the per-region scatter-MAD window (scale over which sigma(f) is constant).",
    inst_sensitivity="Y",
    default_grid=(40.0, 60.0, 80.0, 120.0, 160.0),
    run=_run_noise("scatter", "window_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))
_register(KnobSpec(
    path="stage2.scatter.pedestal_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="High-pass running-median width isolating the smooth leakage pedestal.",
    inst_sensitivity="Y",
    default_grid=(10.0, 20.0, 40.0, 80.0),
    run=_run_noise("scatter", "pedestal_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))
_register(KnobSpec(
    path="stage2.scatter.smoothing_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="Broad lower-envelope median sigma smoothing width (0 disables).",
    inst_sensitivity="Y",
    default_grid=(0.0, 400.0, 800.0, 1200.0),
    run=_run_noise("scatter", "smoothing_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))
# Stage 2 noise — adaptive estimator smoothing window (legacy method).
_register(KnobSpec(
    path="stage2.smoothing.smoothing_window_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="Adaptive estimator: moving-window size for per-point sigma interpolation.",
    inst_sensitivity="Y",
    default_grid=(150.0, 300.0, 600.0),
    run=_run_noise("adaptive", "smoothing_window_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))

# Stage 2b tau calibration — requires Stages 0-2. Each value re-runs the STFT
# calibration (the slowest stage), so default grids are kept modest.
_register(KnobSpec(
    path="stage2b.stft.n_seg",
    stage="stage2b_tau",
    requires="stage2_noise_result",
    help="Number of non-overlapping STFT frames (window = T_full / n_seg).",
    inst_sensitivity="Y",
    default_grid=(6, 8, 10, 14, 20),
    run=_run_tau("stft", "n_seg"),
    metric=_metric_tau,
    metric_columns=("tau_maj_us", "sigma_tau_us", "n_contributors"),
    plot=plot_tau_trend,
))
_register(KnobSpec(
    path="stage2b.stft.t_sigma",
    stage="stage2b_tau",
    requires="stage2_noise_result",
    help="Above-threshold SNR gate for per-frame signal detection (contributor floor).",
    inst_sensitivity="Y",
    default_grid=(3.0, 4.0, 5.0, 6.0, 8.0),
    run=_run_tau("stft", "t_sigma"),
    metric=_metric_tau,
    metric_columns=("tau_maj_us", "sigma_tau_us", "n_contributors"),
    plot=plot_tau_trend,
))
_register(KnobSpec(
    path="stage2b.polish.polish_snr_cap",
    stage="stage2b_tau",
    requires="stage2_noise_result",
    help="SNR above which the Gauss-Newton polish is skipped (avoid over-correction).",
    inst_sensitivity="Y",
    default_grid=(5.0, 7.0, 9.0, 12.0),
    run=_run_tau("polish", "polish_snr_cap"),
    metric=_metric_tau,
    metric_columns=("tau_maj_us", "sigma_tau_us", "n_contributors"),
    plot=plot_tau_trend,
))
_register(KnobSpec(
    path="stage2b.polish.polish_noise_debias",
    stage="stage2b_tau",
    requires="stage2_noise_result",
    help="Apply Rician-unbiased magnitude on high-SNR frames (removes residual bias).",
    inst_sensitivity="Y",
    default_grid=(False, True),
    run=_run_tau("polish", "polish_noise_debias"),
    metric=_metric_tau,
    metric_columns=("tau_maj_us", "sigma_tau_us", "n_contributors"),
))


def get_knob(path: str) -> KnobSpec:
    """Look up a knob by its dotted path, or raise ``KeyError`` with a hint."""
    try:
        return _REGISTRY[path]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"unknown tuning knob {path!r}; registered knobs: {known}"
        ) from None


def list_knobs(stage: Optional[str] = None) -> Tuple[KnobSpec, ...]:
    """All registered knobs (optionally filtered to one ``stage``), path-sorted."""
    specs = sorted(_REGISTRY.values(), key=lambda s: s.path)
    if stage is not None:
        specs = [s for s in specs if s.stage == stage]
    return tuple(specs)
