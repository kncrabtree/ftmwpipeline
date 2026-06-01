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
    """Optional plot adapter; ``None`` => table-only output for this knob."""

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


# ---------------------------------------------------------------------------
# Metric reducers
# ---------------------------------------------------------------------------

def _metric_start(result: Any) -> Dict[str, Any]:
    return {
        "start_us": round(float(result.start_us), 4),
        "chirp_end_us": round(float(result.chirp_end_us), 4),
        "chirp_detected": bool(result.chirp_detected),
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


# Start detection (pre-Stage 1) — requires Stage 0 (FID).
_register(KnobSpec(
    path="start.guard_margin_us",
    stage="start_detection",
    requires="stage0_fid_data",
    help="Margin past the chirp end for switch-bounce ringdown (instrument-specific).",
    inst_sensitivity="Y",
    default_grid=(0.3, 0.5, 0.67, 0.85, 1.0),
    run=_run_start("guard_margin_us"),
    metric=_metric_start,
    metric_columns=("start_us", "chirp_end_us", "chirp_detected"),
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
