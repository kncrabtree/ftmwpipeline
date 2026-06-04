"""Knob registry: the data behind the tuning sweep engine.

A :class:`KnobSpec` declares everything parameter-specific the knob-agnostic
engine needs: how to set the knob and re-run the affected stage (``run``), how
to reduce that stage's result to one or more named metric columns (``metric``),
an optional plot adapter, and an optional recommender (or a simple
``direction`` for the built-in best-value pick). Knobs are addressed by a
dotted settings path (e.g. ``stage2.window_mhz``) that mirrors the
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
    plot_ft_band_stack,
    plot_noise_sweep,
    plot_peak_detection,
    plot_shape_vote,
    plot_spectra_ladder,
    plot_start_detection,
    plot_tau_trend,
    plot_window_planning,
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
    """Dotted settings path, e.g. ``"stage2.window_mhz"``."""
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
    tier: str = "primary"
    """``"primary"`` (shown in the default ``tune list``) or ``"advanced"``
    (revealed only with ``--all`` / ``include_advanced``). Lets the surface
    expose every knob while keeping the default view a short starting point."""

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

    Sets the field on a defaults bundle and passes it as ``settings=`` so any
    field is sweepable uniformly — only a subset of fields are exposed as
    per-knob kwargs on ``detect_start_time``. Uses ``stamp=False`` so the sweep
    never mutates the file's recommended ``start_us`` layer; the engine still
    operates on a working copy.
    """

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle
        from dataclasses import replace
        from ftmwpipeline.core.start_detection_settings import (
            StartDetectionSettings,
        )

        settings = replace(StartDetectionSettings(), **{field_name: value})
        return ftmw.detect_start_time(path, stamp=False, settings=settings)

    return run


def _run_noise(field_name: str) -> RunFn:
    """Re-run noise estimation with a single ``NoiseSettings`` field set.

    Sets the one field on a ``NoiseSettings`` bundle and passes it as
    ``settings=``, so the sweep drives the estimator through the settings
    resolver rather than the (deprecated) per-knob kwargs.
    """

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle
        from ftmwpipeline.core import noise_settings as ns

        bundle = ns.NoiseSettings(**{field_name: value})
        return ftmw.estimate_noise(path, settings=bundle)

    return run


@dataclass(frozen=True)
class FtAtStart:
    """A FT computed at a particular window start, with the start (and the
    chirp-end it was referenced to, if any) retained for the ladder plot.

    Both the ``stage1.start_us`` and ``stage0.guard_margin_us`` knobs sweep the
    *same* thing — the spectrum as a function of where the FT window starts.
    They differ only in how the start is specified: ``start_us`` absolutely,
    ``guard_margin_us`` as an offset past the (separately detected) chirp end.
    """

    ft: Any
    # start_us is set for the start ladder (start_us / guard knobs); the FT
    # band/window knobs (trim / end_us) leave it None — their plot is the
    # spectrum stack, which does not mark a window start.
    start_us: Optional[float] = None
    chirp_end_us: Optional[float] = None


def _run_ft_start(path: Path, value: Any) -> Any:
    """Recompute the FT at an absolute window start; returns an ``FtAtStart``.

    Other FT settings (trim, zpf, apodization) are inherited from the file's
    resolution chain, so the sweep isolates the effect of ``start_us``.
    """
    import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

    start = float(value)
    return FtAtStart(ft=ftmw.compute_ft(path, start_us=start), start_us=start)


def _run_ft_end() -> RunFn:
    """Recompute the FT with the window end set to each value; isolates the
    effect of ``end_us`` (FID truncation), inheriting the rest of the chain."""

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

        return FtAtStart(ft=ftmw.compute_ft(path, end_us=float(value)))

    return run


def _run_ft_trim(edge: str) -> RunFn:
    """Sweep one edge of the FT frequency trim, holding the other at the file's
    canonical value. ``edge`` is ``"min"`` or ``"max"``.

    The canonical trim edges are read once from the persisted FT (its
    ``freq_array`` spans the active band; the array is sideband-ordered, so the
    edges are its min/max, not its endpoints) and memoised for the sweep.
    """
    cache: Dict[Any, Tuple[float, float]] = {}

    def run(path: Path, value: Any) -> Any:
        import numpy as np
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle

        key = str(path)
        edges = cache.get(key)
        if edges is None:
            freqs = np.asarray(ftmw.compute_ft(path).freq_array, dtype=float)
            edges = (float(freqs.min()), float(freqs.max()))
            cache.clear()
            cache[key] = edges
        lo, hi = edges
        trim = (float(value), hi) if edge == "min" else (lo, float(value))
        return FtAtStart(ft=ftmw.compute_ft(path, trim=trim))

    return run


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

        # Key on the path only: compute_ft below rewrites the working file each
        # value (so its mtime/size are not stable), but the FID the detection
        # reads is invariant within a sweep. cache holds just the latest file.
        key = str(path)
        chirp_end = cache.get(key)
        if chirp_end is None:
            chirp_end = float(ftmw.detect_start_time(path, stamp=False).chirp_end_us)
            cache.clear()
            cache[key] = chirp_end
        start = chirp_end + float(value)
        return FtAtStart(
            ft=ftmw.compute_ft(path, start_us=start),
            start_us=start,
            chirp_end_us=chirp_end,
        )

    return run


def _run_tau(sub_block: str, field_name: str) -> RunFn:
    """Re-run a Stage 2b tau extraction with a single settings field set.

    Builds a ``TauCalibrationSettings`` bundle (the preset layer) carrying just
    the one sub-block field, so any tau knob — including fields the orchestrators
    do not expose as kwargs — is sweepable uniformly. The sub-block selects the
    orchestrator: ``gaussian`` drives ``calibrate_tau_G`` (Gaussian τ_G),
    ``recommendation`` drives ``recommend_shape`` (the exp/gauss/voigt vote), and
    every other sub-block drives the exponential ``calibrate_tau``. All three
    return through the same settings resolver, so the bundle's single override
    composes with the file's persisted/default layers.
    """

    def run(path: Path, value: Any) -> Any:
        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle
        from ftmwpipeline.core import tau_calibration_settings as tcs

        sub_cls = {
            "stft": tcs.StftSubSettings,
            "polish": tcs.PolishSubSettings,
            "aggregation": tcs.AggregationSubSettings,
            "band": tcs.BandSubSettings,
            "gaussian": tcs.GaussianSubSettings,
            "recommendation": tcs.RecommendationSubSettings,
        }[sub_block]
        bundle = tcs.TauCalibrationSettings(
            **{sub_block: sub_cls(**{field_name: value})}
        )
        if sub_block == "gaussian":
            return ftmw.calibrate_tau_G(path, settings=bundle)
        if sub_block == "recommendation":
            return ftmw.recommend_shape(path, settings=bundle)
        return ftmw.calibrate_tau(path, settings=bundle)

    return run


def _run_peaks(sub_block: str, field_name: str) -> RunFn:
    """Re-run Stage 3 peak detection with a single settings field set.

    Builds a ``PeakDetectionSettings`` bundle (the preset layer) carrying just
    the one sub-block field, so any Stage 3 knob is sweepable uniformly. Drives
    the internal ``detect_peaks_impl`` directly (rather than the public
    ``detect_peaks``) so the result carries the active FT, per-pass peak lists,
    and per-pass noise the spectrum-overlay plot needs — the public surface
    returns only the peak list.
    """

    def run(path: Path, value: Any) -> Any:
        from ftmwpipeline._internal.stage3_impl import detect_peaks_impl  # lazy
        from ftmwpipeline.core import peak_detection_settings as pds

        sub_cls = {
            "promotion": pds.PromotionSubSettings,
            "savgol": pds.SavgolSubSettings,
            "primary_pass": pds.PrimaryPassSubSettings,
            "gap_pass": pds.GapPassSubSettings,
        }[sub_block]
        bundle = pds.PeakDetectionSettings(
            **{sub_block: sub_cls(**{field_name: value})}
        )
        return detect_peaks_impl(str(path), settings=bundle)

    return run


def _run_windows(sub_block: str, field_name: str) -> RunFn:
    """Re-run Stage 4 window assignment with a single settings field set.

    Builds a ``WindowPlanningSettings`` bundle (the preset layer) carrying just
    the one sub-block field, so any Stage 4 knob is sweepable uniformly. The
    ``assign_windows_impl`` result carries the plan, the active FT, and the
    active σ the boundary/coherence overlay plot needs.
    """

    def run(path: Path, value: Any) -> Any:
        from ftmwpipeline._internal.stage4_impl import assign_windows_impl  # lazy
        from ftmwpipeline.core import window_planning_settings as wps

        sub_cls = {
            "coherence": wps.CoherenceSubSettings,
            "clustering": wps.ClusteringSubSettings,
            "contributor": wps.ContributorSubSettings,
            "leakage": wps.LeakageSubSettings,
        }[sub_block]
        bundle = wps.WindowPlanningSettings(
            **{sub_block: sub_cls(**{field_name: value})}
        )
        return assign_windows_impl(str(path), settings=bundle)

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


def _metric_shape(result: Any) -> Dict[str, Any]:
    """Reduce a ``ShapeRecommendation`` to the exp/gauss/voigt vote rates plus
    the verdict, for the shape-recommendation knobs."""
    rates = dict(getattr(result, "vote_rates", {}) or {})
    rec = getattr(result, "recommended_shape", None)
    return {
        "recommended_shape": "none" if rec in (None, "") else str(rec),
        "exp": round(float(rates.get("exp", float("nan"))), 4),
        "gauss": round(float(rates.get("gauss", float("nan"))), 4),
        "voigt": round(float(rates.get("voigt", float("nan"))), 4),
        "n_contributors": int(getattr(result, "n_contributors", 0)),
    }


def _metric_peaks(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 3 result to the peaks *passed to Stage 4*: counts by SNR
    band plus the SNR distribution.

    Stage 3 stores every line it detects, but what a user tunes for is which
    lines advance — so the table summarises only the promoted peaks. Two
    cross-cutting views: counts split into the weak / medium / strong SNR bands
    (whose edges are themselves the tunable ``weak_medium_snr`` /
    ``medium_strong_snr`` knobs), and the SNR distribution (min, p10/p25/p50/p90,
    max — ``snr_min`` tracks the promotion cutoff, a handy sanity check). The
    detection-pool size and the internal/promoted bookkeeping are not surfaced.
    """
    import numpy as np

    peaks = result.get("peaks", [])
    promoted = [p for p in peaks if p.properties.get("promoted")]

    def _band(p: Any) -> Optional[str]:
        cls = getattr(p, "classification", None)
        return getattr(cls, "value", None)

    snrs = np.asarray(
        [float(p.snr) for p in promoted if p.snr is not None], dtype=float
    )

    def _pct(q: float) -> float:
        return round(float(np.percentile(snrs, q)), 3) if snrs.size else 0.0

    return {
        "n_total": len(promoted),
        "n_strong": sum(1 for p in promoted if _band(p) == "strong"),
        "n_medium": sum(1 for p in promoted if _band(p) == "medium"),
        "n_weak": sum(1 for p in promoted if _band(p) == "weak"),
        "snr_min": round(float(snrs.min()), 3) if snrs.size else 0.0,
        "snr_p10": _pct(10),
        "snr_p25": _pct(25),
        "snr_p50": _pct(50),
        "snr_p90": _pct(90),
        "snr_max": round(float(snrs.max()), 3) if snrs.size else 0.0,
    }


def _metric_windows(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 4 result to the plan's shape: window counts split by
    difficulty, the contributor/dependency bookkeeping, and the window-width
    distribution.

    What a user tunes Stage 4 for is how the band partitions — how many windows,
    how many are HARD (oversized / leakage-touched), how many out-of-window lines
    are frozen as fixed contributors, how many windows the knob forces a split
    on, and how wide the windows run. The width tail (p95/max) is the headline
    for the width-cap knob; the HARD/split counts track the coherence and cap
    knobs.
    """
    import numpy as np

    plan = result["plan"]
    widths = np.asarray([w.width_mhz for w in plan.windows], dtype=float)
    if widths.size == 0:
        widths = np.zeros(1)
    n_split = sum(1 for w in plan.windows if w.split_proposal is not None)
    return {
        "n_windows": int(result["n_windows"]),
        "n_hard": int(result["n_hard"]),
        "n_easy": int(result["n_easy"]),
        "n_free": int(result["n_free_peaks"]),
        "n_fixed": int(result["n_fixed_contributors"]),
        "n_dep": int(result["n_dependencies"]),
        "n_split": int(n_split),
        "width_p50": round(float(np.median(widths)), 3),
        "width_p95": round(float(np.percentile(widths, 95)), 3),
        "width_max": round(float(widths.max()), 3),
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
# live under the stage0 prefix (pre-Stage-1 start detection on the raw FID).
# below (sweep_max_us, min_chirp_drop_ratio) are the ones that move the chirp
# end, so they show the Sigma|FT| detection curve.
_DETECTION_SEE_ALSO = (
    "stage0.guard_margin_us / stage1.start_us — stack the resulting spectra to "
    "see how the detected start affects the FT (chirp/ringdown residue)."
)
_register(KnobSpec(
    path="stage0.guard_margin_us",
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
    path="stage0.sweep_max_us",
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
    path="stage0.min_chirp_drop_ratio",
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

# Stage 0 advanced detection internals (chirp-end localisation + sweep). The
# band_min_mhz / band_max_mhz integration-band override is intentionally not a
# sweep knob — the detector ignores it unless both edges are set, so neither
# sweeps meaningfully alone (reach them via settings= / preset=).
_START_COLS = ("start_us", "chirp_end_us", "chirp_detected")
for _path, _field, _help, _grid, _inst in (
    ("stage0.step_us", "step_us",
     "Start-time sweep step (us): the resolution of the Sigma|FT| curve.",
     (0.01, 0.02, 0.05), "maybe"),
    ("stage0.floor_factor", "floor_factor",
     "Multiple of the floor at which Sigma|FT| is considered settled (chirp end).",
     (2.0, 3.0, 5.0), "maybe"),
    ("stage0.floor_tail_us", "floor_tail_us",
     "Deep-tail width (us) whose median defines the settled floor.",
     (0.5, 1.0, 2.0), "N"),
):
    _register(KnobSpec(
        path=_path, stage="start_detection", requires="stage0_fid_data",
        help=_help, inst_sensitivity=_inst, default_grid=_grid,
        run=_run_start(_field), metric=_metric_start,
        metric_columns=_START_COLS, plot=plot_start_detection, tier="advanced",
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
    default_grid=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    run=_run_ft_start,
    metric=_metric_ft_band_floor,
    metric_columns=("p1", "p5", "p10", "p20", "p50", "max"),
    plot=plot_spectra_ladder,
))

# FT frequency trim + window end. The trim default grids are MHz-absolute and
# 2638-shaped; pass --grid for another instrument's band. zpf / expf_us /
# window_function are intentionally NOT exposed — the canonical analysis runs a
# raw, unapodized FT (they corrupt the Stage 2/5 noise and fit statistics);
# units_power is a display-scale choice surfaced by the resolved-settings view.
_FT_BAND_COLS = ("p1", "p5", "p10", "p20", "p50", "max")
_register(KnobSpec(
    path="stage1.trim_min_mhz",
    stage="stage1_ft",
    requires="stage0_fid_data",
    help="Lower edge of the FT frequency trim (MHz, absolute; --grid for your band).",
    inst_sensitivity="Y",
    default_grid=(26000.0, 27000.0, 28000.0, 30000.0),
    run=_run_ft_trim("min"),
    metric=_metric_ft_band_floor,
    metric_columns=_FT_BAND_COLS,
    plot=plot_ft_band_stack,
))
_register(KnobSpec(
    path="stage1.trim_max_mhz",
    stage="stage1_ft",
    requires="stage0_fid_data",
    help="Upper edge of the FT frequency trim (MHz, absolute; --grid for your band).",
    inst_sensitivity="Y",
    default_grid=(36000.0, 38000.0, 40000.0),
    run=_run_ft_trim("max"),
    metric=_metric_ft_band_floor,
    metric_columns=_FT_BAND_COLS,
    plot=plot_ft_band_stack,
))
_register(KnobSpec(
    path="stage1.end_us",
    stage="stage1_ft",
    requires="stage0_fid_data",
    help="FID window end time (us): truncates the record before the FT.",
    inst_sensitivity="Y",
    default_grid=(5.0, 10.0, 15.0),
    run=_run_ft_end(),
    metric=_metric_ft_band_floor,
    metric_columns=_FT_BAND_COLS,
    plot=plot_ft_band_stack,
))

# Stage 2 noise — scatter estimator (the canonical default). Requires Stage 1.
_register(KnobSpec(
    path="stage2.window_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="Width of the per-region scatter-MAD window (scale over which sigma(f) is constant).",
    inst_sensitivity="Y",
    default_grid=(40.0, 60.0, 80.0, 120.0, 160.0),
    run=_run_noise("window_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))
_register(KnobSpec(
    path="stage2.pedestal_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="High-pass running-median width isolating the smooth leakage pedestal.",
    inst_sensitivity="Y",
    default_grid=(10.0, 20.0, 40.0, 80.0),
    run=_run_noise("pedestal_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))
_register(KnobSpec(
    path="stage2.smoothing_mhz",
    stage="stage2_noise",
    requires="stage1_complex_ft",
    help="Broad lower-envelope median sigma smoothing width (0 disables).",
    inst_sensitivity="Y",
    default_grid=(0.0, 400.0, 800.0, 1200.0),
    run=_run_noise("smoothing_mhz"),
    metric=_metric_noise,
    metric_columns=("median_sigma", "noise_fraction"),
    plot=plot_noise_sweep,
))

# Stage 2 advanced — remaining scatter knobs. All re-run Stage 2 and report the
# same sigma trend + sigma(f)-over-spectrum overlay.
_NOISE_COLS = ("median_sigma", "noise_fraction")
for _field, _help, _grid, _inst in (
    ("line_k",
     "Robust-sigma multiple above which a bin is flagged a line (excluded).",
     (4.0, 6.0, 8.0, 12.0), "maybe"),
    ("n_iter",
     "Self-mask refinement iterations of the scatter estimator.",
     (1, 2, 3, 5), "N"),
    ("region_aware",
     "Use the region-aware Rician correction (else a fixed mid-regime factor).",
     (False, True), "maybe"),
    ("smoothing_percentile",
     "Percentile of the broad sigma smoothing (50=median; lower=lower-envelope).",
     (25.0, 50.0, 75.0), "maybe"),
    ("convolve_mhz",
     "Gaussian sigma (MHz) of the second, step-removing smoothing pass (0=off).",
     (0.0, 100.0, 200.0, 400.0), "maybe"),
):
    _register(KnobSpec(
        path=f"stage2.{_field}", stage="stage2_noise",
        requires="stage1_complex_ft", help=_help, inst_sensitivity=_inst,
        default_grid=_grid, run=_run_noise(_field),
        metric=_metric_noise, metric_columns=_NOISE_COLS, plot=plot_noise_sweep,
        tier="advanced",
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
    tier="advanced",
))

# --- Stage 2b: advanced STFT gates (exp twin) -----------------------------
_TAU_COLS = ("tau_maj_us", "sigma_tau_us", "n_contributors")
_TAU_TWIN_SEE_ALSO = (
    "stage2b.gaussian.* is the Gaussian τ_G twin and stage2b.recommendation.* "
    "is the exp-vs-gauss shape vote — all three share the STFT contributor pool."
)
for _path, _field, _help, _grid, _inst in (
    ("stage2b.stft.tau_max_us",
     "tau_max_us",
     "Hard upper clip on recovered τ (saturation → spur candidate); unset → derived.",
     (20.0, 40.0, 80.0), "maybe"),
    ("stage2b.stft.tau_max_factor",
     "tau_max_factor",
     "τ_max as a multiple of the full-record duration when tau_max_us is unset.",
     (3.0, 5.0, 8.0, 12.0), "maybe"),
    ("stage2b.stft.rss_gate_factor",
     "rss_gate_factor",
     "Bad-fit gate strength (relative-or-absolute residual hybrid).",
     (3.0, 5.0, 8.0, 12.0), "maybe"),
    ("stage2b.stft.relative_gate_fraction",
     "relative_gate_fraction",
     "Relative-RSS fraction below which a per-frame fit is accepted.",
     (0.02, 0.05, 0.10, 0.20), "maybe"),
):
    _register(KnobSpec(
        path=_path, stage="stage2b_tau", requires="stage2_noise_result",
        help=_help, inst_sensitivity=_inst, default_grid=_grid,
        run=_run_tau("stft", _field), metric=_metric_tau,
        metric_columns=_TAU_COLS, plot=plot_tau_trend, tier="advanced",
    ))

# --- Stage 2b: advanced polish knobs --------------------------------------
for _path, _field, _help, _grid in (
    ("stage2b.polish.polish_n_iter", "polish_n_iter",
     "Gauss-Newton polish iterations per eligible contributor.", (1, 2, 3)),
    ("stage2b.polish.polish_top_n", "polish_top_n",
     "Polish only the top-N contributors by SNR (unset → all).",
     (200, 500, 1000, 2000)),
):
    _register(KnobSpec(
        path=_path, stage="stage2b_tau", requires="stage2_noise_result",
        help=_help, inst_sensitivity="N", default_grid=_grid,
        run=_run_tau("polish", _field), metric=_metric_tau,
        metric_columns=_TAU_COLS, plot=plot_tau_trend, tier="advanced",
    ))

# --- Stage 2b: advanced aggregation / acceptance knobs --------------------
for _path, _field, _help, _grid, _inst in (
    ("stage2b.aggregation.min_contributors", "min_contributors",
     "Minimum contributor count for the calibration to pass preconditions.",
     (100, 200, 400, 800), "N"),
    ("stage2b.aggregation.sigma_tau_fraction_max", "sigma_tau_fraction_max",
     "Max σ_τ/τ_maj for the calibration to pass preconditions.",
     (0.10, 0.20, 0.30), "N"),
    ("stage2b.aggregation.bimodality_dominant_fraction",
     "bimodality_dominant_fraction",
     "Dominant-mode fraction above which a bimodal histogram still passes.",
     (0.6, 0.7, 0.8), "N"),
    ("stage2b.aggregation.sigma_tau_floor_us", "sigma_tau_floor_us",
     "Floor on the reported σ_τ (guards against over-tight spreads).",
     (0.0, 0.5, 1.0), "maybe"),
    ("stage2b.aggregation.spur_cluster_multiplier", "spur_cluster_multiplier",
     "Scale on the spur-cluster width (wider → more bins flagged as spurs).",
     (1.0, 1.5, 2.0), "maybe"),
):
    _register(KnobSpec(
        path=_path, stage="stage2b_tau", requires="stage2_noise_result",
        help=_help, inst_sensitivity=_inst, default_grid=_grid,
        run=_run_tau("aggregation", _field), metric=_metric_tau,
        metric_columns=_TAU_COLS, plot=plot_tau_trend, tier="advanced",
    ))

# --- Stage 2b: multi-band majorities --------------------------------------
_register(KnobSpec(
    path="stage2b.band.min_contributors_per_band",
    stage="stage2b_tau", requires="stage2_noise_result",
    help="Min contributors for a band to use its own τ majority (else band-wide).",
    inst_sensitivity="Y", default_grid=(25, 50, 100, 200),
    run=_run_tau("band", "min_contributors_per_band"), metric=_metric_tau,
    metric_columns=_TAU_COLS, plot=plot_tau_trend, see_also=_TAU_TWIN_SEE_ALSO,
))
_register(KnobSpec(
    path="stage2b.band.compute_band_majorities",
    stage="stage2b_tau", requires="stage2_noise_result",
    help="Compute per-band τ majorities (the τ-vs-frequency band steps).",
    inst_sensitivity="Y", default_grid=(False, True),
    run=_run_tau("band", "compute_band_majorities"), metric=_metric_tau,
    metric_columns=_TAU_COLS, plot=plot_tau_trend, tier="advanced",
))

# --- Stage 2b: Gaussian τ_G twin (calibrate_tau_G) ------------------------
_register(KnobSpec(
    path="stage2b.gaussian.snr_min",
    stage="stage2b_tau", requires="stage2_noise_result",
    help="Gaussian τ_G: per-bin SNR floor for a contributor to enter the fit.",
    inst_sensitivity="Y", default_grid=(10.0, 15.0, 20.0, 30.0),
    run=_run_tau("gaussian", "snr_min"), metric=_metric_tau,
    metric_columns=_TAU_COLS, plot=plot_tau_trend, see_also=_TAU_TWIN_SEE_ALSO,
))
for _path, _field, _help, _grid in (
    ("stage2b.gaussian.tau_G_bound_lo", "tau_G_bound_lo",
     "Gaussian τ_G lower fit bound (us).", (0.2, 0.5, 1.0)),
    ("stage2b.gaussian.tau_G_bound_hi", "tau_G_bound_hi",
     "Gaussian τ_G upper fit bound (us).", (50.0, 100.0, 200.0)),
    ("stage2b.gaussian.delta_chi2r_min", "delta_chi2r_min",
     "Min χ²ᵣ improvement of the Gaussian over the exp fit to count a bin.",
     (0.5, 1.0, 2.0)),
    ("stage2b.gaussian.tau_G_upper_fraction", "tau_G_upper_fraction",
     "Fraction of the τ_G bound above which a fit is treated as railed.",
     (0.5, 0.7, 0.9)),
    ("stage2b.gaussian.min_contributors", "min_contributors",
     "Minimum Gaussian-eligible contributor count for τ_G preconditions.",
     (25, 50, 100)),
):
    _register(KnobSpec(
        path=_path, stage="stage2b_tau", requires="stage2_noise_result",
        help=_help, inst_sensitivity="maybe", default_grid=_grid,
        run=_run_tau("gaussian", _field), metric=_metric_tau,
        metric_columns=_TAU_COLS, plot=plot_tau_trend, tier="advanced",
    ))

# --- Stage 2b: exp-vs-gauss shape recommendation (recommend_shape) --------
_SHAPE_COLS = ("recommended_shape", "exp", "gauss", "voigt", "n_contributors")
_register(KnobSpec(
    path="stage2b.recommendation.pure_margin_threshold",
    stage="stage2b_tau", requires="stage2_noise_result",
    help="Min SNR-weighted vote margin for a pure shape to win (else 'none').",
    inst_sensitivity="maybe", default_grid=(0.05, 0.10, 0.15, 0.20),
    run=_run_tau("recommendation", "pure_margin_threshold"),
    metric=_metric_shape, metric_columns=_SHAPE_COLS, plot=plot_shape_vote,
    see_also=_TAU_TWIN_SEE_ALSO,
))
for _path, _field, _help, _grid in (
    ("stage2b.recommendation.snr_min", "snr_min",
     "Shape vote: per-bin SNR floor for a contributor to vote.",
     (10.0, 15.0, 20.0, 30.0)),
    ("stage2b.recommendation.tau_bound_lo", "tau_bound_lo",
     "Shape vote: lower τ fit bound shared by the per-bin model fits (us).",
     (0.2, 0.5, 1.0)),
    ("stage2b.recommendation.tau_bound_hi", "tau_bound_hi",
     "Shape vote: upper τ fit bound shared by the per-bin model fits (us).",
     (50.0, 100.0, 200.0)),
):
    _register(KnobSpec(
        path=_path, stage="stage2b_tau", requires="stage2_noise_result",
        help=_help, inst_sensitivity="maybe", default_grid=_grid,
        run=_run_tau("recommendation", _field), metric=_metric_shape,
        metric_columns=_SHAPE_COLS, plot=plot_shape_vote, tier="advanced",
    ))


# ---------------------------------------------------------------------------
# Stage 3 peak detection — requires Stages 1-2 (Stage 2b optional: when present
# the gap matched filter is anchored on τ_maj and shape-matched, else it falls
# back to the user apodization). Every Stage 3 sweep re-runs detect_peaks and
# renders the same view: a count/SNR trend over per-value × per-region spectrum
# panels showing which peaks each value finds, drops, and promotes.
# ---------------------------------------------------------------------------
_PEAK_COLS = (
    "n_total", "n_strong", "n_medium", "n_weak",
    "snr_min", "snr_p10", "snr_p25", "snr_p50", "snr_p90", "snr_max",
)
_PEAK_SEE_ALSO = (
    "the table counts the peaks passed to Stage 4 by SNR band (weak/medium/"
    "strong); the panels show where they land — read alongside the Stage 2 noise "
    "floor (stage2.*) and the primary's own apodized floor "
    "(stage3.primary_pass.noise_*)."
)


def _peak_knob(
    path: str, sub_block: str, field_name: str, help_: str,
    inst: str, grid: Tuple[Any, ...], tier: str = "primary",
    see_also: Optional[str] = None,
) -> None:
    _register(KnobSpec(
        path=path, stage="stage3_peaks", requires="stage2_noise_result",
        help=help_, inst_sensitivity=inst, default_grid=grid,
        run=_run_peaks(sub_block, field_name), metric=_metric_peaks,
        metric_columns=_PEAK_COLS, plot=plot_peak_detection, tier=tier,
        see_also=see_also,
    ))


# Primary tier — the Y-rated detection-shaping knobs.
_peak_knob(
    "stage3.promotion.min_snr", "promotion", "min_snr",
    "User-grid promotion cutoff: peaks at/above this SNR advance to Stage 4.",
    "Y", (2.0, 2.5, 3.0, 4.0, 5.0), see_also=_PEAK_SEE_ALSO,
)
_peak_knob(
    "stage3.promotion.internal_min_snr", "promotion", "internal_min_snr",
    "Internal detection floor on the zpf grids (recovers lines apodization smears).",
    "Y", (1.5, 2.0, 2.5, 3.0), see_also=_PEAK_SEE_ALSO,
)
_peak_knob(
    "stage3.primary_pass.min_exclusion_mhz", "primary_pass", "min_exclusion_mhz",
    "Half-width (MHz) around each primary peak the gap pass excludes from its mask.",
    "Y", (0.0, 0.05, 0.1, 0.2, 0.5), see_also=_PEAK_SEE_ALSO,
)
_peak_knob(
    "stage3.primary_pass.primary_leakage_floor_k", "primary_pass",
    "primary_leakage_floor_k",
    "Scale on the primary leakage-aware floor k·(S_coh/√M)·σ (0 disables).",
    "Y", (0.0, 0.5, 1.0, 2.0, 3.0), see_also=_PEAK_SEE_ALSO,
)
_peak_knob(
    "stage3.gap_pass.gap_leakage_floor_k", "gap_pass", "gap_leakage_floor_k",
    "Scale on the gap leakage-aware floor k·(S_coh/√M)·σ (0 disables; replaces "
    "the former hard S_coh mask).",
    "Y", (0.0, 1.0, 2.0, 3.0, 5.0), see_also=_PEAK_SEE_ALSO,
)

# Advanced — classification edges (move only the weak/medium/strong labels).
_peak_knob(
    "stage3.promotion.weak_medium_snr", "promotion", "weak_medium_snr",
    "Weak/medium SNR classification boundary.",
    "Y", (5.0, 10.0, 15.0, 20.0), tier="advanced",
)
_peak_knob(
    "stage3.promotion.medium_strong_snr", "promotion", "medium_strong_snr",
    "Medium/strong SNR classification boundary.",
    "Y", (30.0, 50.0, 75.0, 100.0), tier="advanced",
)

# Advanced — Savitzky-Golay apex localiser (algorithmic conditioning).
for _path, _field, _help, _grid in (
    ("stage3.savgol.sg_window", "sg_window",
     "Primary-pass Savitzky-Golay window (bins, odd).", (7, 9, 11, 15)),
    ("stage3.savgol.sg_order", "sg_order",
     "Savitzky-Golay polynomial order.", (2, 3, 4)),
    ("stage3.savgol.sg_fwhm_coverage", "sg_fwhm_coverage",
     "Gap-pass window target in line-FWHM units (window auto-derived).",
     (3.0, 4.0, 5.0)),
    ("stage3.savgol.sg_min_window", "sg_min_window",
     "Minimum Savitzky-Golay window (polynomial stability floor).", (5, 7, 9)),
):
    _peak_knob(_path, "savgol", _field, _help, "N", _grid, tier="advanced")

# Advanced — primary-pass apodization + zpf (position-finding only).
_peak_knob(
    "stage3.primary_pass.primary_window", "primary_pass", "primary_window",
    "Primary-pass apodization window (sidelobe suppression; affects positions only).",
    "N", ("blackmanharris", "blackman", "hann", "hamming"), tier="advanced",
)
_peak_knob(
    "stage3.primary_pass.detection_zpf", "primary_pass", "detection_zpf",
    "Zero-padding factor for the active-region primary spectrum.",
    "N", (1, 2, 3), tier="advanced",
)

# Advanced — the primary pass's own apodized-domain σ (scatter estimator on the
# Blackman-Harris spectrum). Mirrors the Stage 2 NoiseSettings knobs; see
# stage2.* for the unapodized authority twin.
for _path, _field, _help, _grid, _inst in (
    ("stage3.primary_pass.noise_window_mhz", "noise_window_mhz",
     "Apodized-domain σ: scatter-MAD window width (MHz).",
     (40.0, 60.0, 80.0, 120.0, 160.0), "Y"),
    ("stage3.primary_pass.noise_pedestal_mhz", "noise_pedestal_mhz",
     "Apodized-domain σ: high-pass running-median width (MHz).",
     (10.0, 20.0, 40.0, 80.0), "Y"),
    ("stage3.primary_pass.noise_smoothing_mhz", "noise_smoothing_mhz",
     "Apodized-domain σ: broad lower-envelope median width (MHz; 0=off).",
     (0.0, 400.0, 800.0, 1200.0), "Y"),
    ("stage3.primary_pass.noise_line_k", "noise_line_k",
     "Apodized-domain σ: robust-σ multiple above which a bin self-masks.",
     (4.0, 6.0, 8.0, 12.0), "maybe"),
    ("stage3.primary_pass.noise_smoothing_percentile", "noise_smoothing_percentile",
     "Apodized-domain σ: percentile of the broad smoothing (50=median).",
     (25.0, 50.0, 75.0), "maybe"),
    ("stage3.primary_pass.noise_convolve_mhz", "noise_convolve_mhz",
     "Apodized-domain σ: step-removing second-pass Gaussian σ (MHz; 0=off).",
     (0.0, 100.0, 200.0, 400.0), "N"),
    ("stage3.primary_pass.noise_n_iter", "noise_n_iter",
     "Apodized-domain σ: self-mask refinement iterations.",
     (1, 2, 3, 5), "N"),
    ("stage3.primary_pass.noise_region_aware", "noise_region_aware",
     "Apodized-domain σ: region-aware Rician correction switch.",
     (False, True), "N"),
):
    _peak_knob(_path, "primary_pass", _field, _help, _inst, _grid, tier="advanced")

# Advanced — gap pass structural toggles.
_peak_knob(
    "stage3.gap_pass.run_gap_pass", "gap_pass", "run_gap_pass",
    "Enable the matched-filter gap pass (recovers weak apodization-suppressed lines).",
    "N", (False, True), tier="advanced",
)
_peak_knob(
    "stage3.gap_pass.gap_active_zpf", "gap_pass", "gap_active_zpf",
    "Zero-padding factor for the matched-filter active-region FFT.",
    "N", (1, 2, 3), tier="advanced",
)


# ---------------------------------------------------------------------------
# Stage 4 window assignment — requires Stage 3 (peaks). Every Stage 4 sweep
# re-runs assign_windows and renders the same view: a window-count trend, a
# band-wide boundary-shift overlay (each value's window edges coloured by
# value), then per-value × per-region zoom panels showing how the partition,
# its difficulty, and the driving S_coh statistic move across the grid.
# ---------------------------------------------------------------------------
_WINDOW_COLS = (
    "n_windows", "n_hard", "n_easy", "n_free", "n_fixed", "n_dep", "n_split",
    "width_p50", "width_p95", "width_max",
)
_WINDOW_SEE_ALSO = (
    "the table reports the plan shape (window/hard/contributor counts + the "
    "width distribution); the panels show how the boundaries move — read "
    "alongside the Stage 3 promoted peaks (stage3.*) that seed the partition."
)


def _window_knob(
    path: str, sub_block: str, field_name: str, help_: str,
    inst: str, grid: Tuple[Any, ...], tier: str = "primary",
    see_also: Optional[str] = None,
) -> None:
    _register(KnobSpec(
        path=path, stage="stage4_windows", requires="stage3_peaks",
        help=help_, inst_sensitivity=inst, default_grid=grid,
        run=_run_windows(sub_block, field_name), metric=_metric_windows,
        metric_columns=_WINDOW_COLS, plot=plot_window_planning, tier=tier,
        see_also=see_also,
    ))


# Primary tier — the Y-rated partition-shaping knobs (grids lifted from the
# tracked stage4-gaussian-audit probes).
_window_knob(
    "stage4.coherence.edge_threshold", "coherence", "edge_threshold",
    "S_coh cutoff (T_edge) for flagging leakage-touched regions that force "
    "window boundaries.",
    "Y", (4.0, 6.0, 8.0, 10.0, 12.0), see_also=_WINDOW_SEE_ALSO,
)
_window_knob(
    "stage4.clustering.max_window_width_mhz", "clustering",
    "max_window_width_mhz",
    "Width cap (MHz) above which a window is HARD and gains a split proposal.",
    "Y", (20.0, 30.0, 40.0, 60.0, 80.0), see_also=_WINDOW_SEE_ALSO,
)
_window_knob(
    "stage4.contributor.magnitude_attachment_threshold", "contributor",
    "magnitude_attachment_threshold",
    "Tier-1 contributor attachment: predicted mean-skirt threshold (σ_c units).",
    "Y", (0.05, 0.075, 0.1, 0.15, 0.2), see_also=_WINDOW_SEE_ALSO,
)
_window_knob(
    "stage4.contributor.min_freeze_snr", "contributor", "min_freeze_snr",
    "SNR floor for fixed-contributor freeze-eligibility (below = thaw candidate).",
    "Y", (20.0, 35.0, 50.0, 75.0, 100.0), see_also=_WINDOW_SEE_ALSO,
)
_window_knob(
    "stage4.leakage.tau_us", "leakage", "tau_us",
    "Decay constant (µs) for the analytic leakage-skirt envelope; None = boxcar "
    "(undamped) limit. Reach the Stage 2b τ anchors via settings=/preset=.",
    "Y", (None, 3.0, 6.0, 12.0), see_also=_WINDOW_SEE_ALSO,
)

# Advanced — coherence band scales and the isolated-peak / per-window caps.
_window_knob(
    "stage4.coherence.edge_m", "coherence", "edge_m",
    "Band width (bins) for the rolling complex-edge coherence statistic.",
    "N", (32, 48, 64, 96, 128), tier="advanced",
)
_window_knob(
    "stage4.coherence.trim_m", "coherence", "trim_m",
    "Band width (bins) for coherence refinement after a leakage-region flag.",
    "N", (16, 24, 32, 48), tier="advanced",
)
_window_knob(
    "stage4.clustering.min_window_half_width_mhz", "clustering",
    "min_window_half_width_mhz",
    "Minimum half-width (MHz) of an isolated-peak proposed window.",
    "maybe", (1.0, 2.0, 3.0, 4.0), tier="advanced",
)
_window_knob(
    "stage4.clustering.max_peaks_per_window", "clustering",
    "max_peaks_per_window",
    "Per-window promoted-peak cap (windows over it are split).",
    "N", (8, 12, 16, 24), tier="advanced",
)


def get_knob(path: str) -> KnobSpec:
    """Look up a knob by its dotted path, or raise ``KeyError`` with a hint."""
    try:
        return _REGISTRY[path]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"unknown tuning knob {path!r}; registered knobs: {known}"
        ) from None


def list_knobs(
    selector: Optional[str] = None,
    *,
    include_advanced: bool = False,
) -> Tuple[KnobSpec, ...]:
    """Registered knobs, path-sorted.

    ``selector`` filters by dotted-path prefix: a knob matches when its path
    equals ``selector`` or begins with ``selector + "."`` (e.g. ``"stage2b"`` or
    ``"stage2b.gaussian"``); the legacy ``stage``-label match is kept as a
    fallback. ``include_advanced=False`` (the default) hides ``tier ==
    "advanced"`` knobs so the default listing stays a short starting point.
    """
    specs = sorted(_REGISTRY.values(), key=lambda s: s.path)
    if not include_advanced:
        specs = [s for s in specs if s.tier != "advanced"]
    if selector is not None:
        specs = [
            s for s in specs
            if s.path == selector
            or s.path.startswith(selector + ".")
            or s.stage == selector
        ]
    return tuple(specs)
