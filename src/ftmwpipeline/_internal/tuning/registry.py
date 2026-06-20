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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

from ...core import (
    noise_settings,
    peak_detection_settings,
    tau_calibration_settings,
    window_planning_settings,
)
from ...core.knob_metadata import field_knob_meta
from .fit_support import reduce_plan_for_fit
from .plots import (
    plot_fit_quality,
    plot_ft_band_stack,
    plot_noise_sweep,
    plot_peak_detection,
    plot_rescue,
    plot_shape_vote,
    plot_spectra_ladder,
    plot_spur,
    plot_start_detection,
    plot_tau_trend,
    plot_thaw,
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
    """``"primary"`` (shown in the default ``scan list``) or ``"advanced"``
    (revealed only with ``--all`` / ``include_advanced``). Lets the surface
    expose every knob while keeping the default view a short starting point."""
    prepare: Optional[Callable[..., None]] = None
    """Optional one-time conditioning of the working copy before the sweep
    (``(work_path, FitWindowSelection, spec, values) -> None``). Stage 5 fit
    knobs set this to reduce the window plan to a representative subset so each
    value re-fits only a few windows; ``None`` => the working copy is swept
    as-is."""
    select_hint: Optional[str] = None
    """Optional hint to the ``prepare`` window-selector. ``"snr_threshold"``
    marks a knob that gates per-window behaviour on the window's SNR, so the
    selector straddle-samples windows across the grid's SNR range (otherwise the
    knob can look inert when the random sample misses its regime)."""

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
        from dataclasses import replace

        import ftmwpipeline.api as ftmw  # lazy: avoid import cycle
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

    Other FT settings (trim, units_power) are inherited from the file's
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


def _run_fit(sub_block: str, field_name: str) -> RunFn:
    """Re-run Stage 5 fitting with a single ``StageFitSettings`` field set.

    Builds a one-field settings bundle (the preset layer) and drives
    ``fit_peaks_impl`` so any fit knob is sweepable uniformly. The working copy's
    window plan has already been reduced to a representative subset by the knob's
    ``prepare`` hook, so each call fits only those windows.
    """

    def run(path: Path, value: Any) -> Any:
        from ftmwpipeline._internal.stage5_impl import fit_peaks_impl  # lazy
        from ftmwpipeline.core import stage_fit_settings as sfs

        sub_cls = {
            "tau": sfs.TauSubSettings,
            "seeder": sfs.SeederSubSettings,
            "conservative": sfs.ConservativeSubSettings,
            "penalties": sfs.PenaltySubSettings,
            "rescue": sfs.RescueSubSettings,
            "thaw": sfs.ThawSubSettings,
            "spur": sfs.SpurSubSettings,
            "baseline": sfs.BaselineSubSettings,
            "doublet_alternative": sfs.DoubletAlternativeSubSettings,
            "peak_survival": sfs.PeakSurvivalSubSettings,
        }[sub_block]
        bundle = sfs.StageFitSettings(**{sub_block: sub_cls(**{field_name: value})})
        return fit_peaks_impl(str(path), settings=bundle)

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


def _metric_fit(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 5 fit to the fit-quality lens: the SNR-normalised
    shape-error fraction ε (and the SNR-aware fail count) as the honest quality
    headline, the structural counts a fit knob actually moves (peaks, free-τ
    windows, median freq uncertainty), and raw χ²ᵣ kept only as a de-emphasised
    secondary so the SNR² floor never masquerades as misfit.
    """
    import numpy as np

    from .fit_support import window_fit_quality

    fit = result["fit"]
    rows = [window_fit_quality(wf) for wf in fit.window_fits]
    eps = np.asarray([r["epsilon"] for r in rows], dtype=float)
    chi2r = np.asarray(
        [r["chi2r"] for r in rows if np.isfinite(r["chi2r"])], dtype=float
    )
    n_fail = sum(1 for r in rows if not r["passed"])
    n_peaks = sum(cast(int, r["n_peaks"]) for r in rows)

    n_free_tau = 0
    sig_f = []
    for wf in fit.window_fits:
        tau = (wf.shared_parameters or {}).get("tau_us")
        if tau is not None and tau.get("error") is not None:
            n_free_tau += 1
        for p in wf.fitted_peaks:
            fe = getattr(p, "frequency_error", None)
            if fe is not None and np.isfinite(fe):
                sig_f.append(float(fe))

    def pct(a: Any, q: float) -> float:
        return round(float(np.percentile(a, q)), 5) if len(a) else 0.0

    return {
        "eps_p50": pct(eps, 50),
        "eps_p95": pct(eps, 95),
        "n_fail": n_fail,
        "n_peaks": n_peaks,
        "n_free_tau": n_free_tau,
        "sigma_f_khz": round(float(np.median(sig_f)) * 1e3, 4) if sig_f else 0.0,
        "chi2r_p50": round(float(np.median(chi2r)), 3) if chi2r.size else 0.0,
        "chi2r_p95": pct(chi2r, 95),
    }


def _fit_eps_summary(fit: Any) -> Tuple[float, int]:
    """``(eps_p50, n_peaks)`` for a ``SpectrumFit`` — the net fit-quality context
    the rescue / spur / thaw metrics carry so the structural change a knob makes
    can be read against whether the band's misfit moved."""
    import numpy as np

    from .fit_support import window_fit_quality

    rows = [window_fit_quality(wf) for wf in fit.window_fits]
    eps = np.asarray([r["epsilon"] for r in rows], dtype=float)
    eps_p50 = round(float(np.percentile(eps, 50)), 5) if eps.size else 0.0
    n_peaks = sum(cast(int, r["n_peaks"]) for r in rows)
    return eps_p50, n_peaks


def _metric_rescue(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 5 fit to the residual-rescue B-loop's work: peaks added by
    accepted rounds, the rescue-origin pruned count (the failsafe — rescue adding
    lines a later refit undoes), merges, the windows and rounds touched, and the
    median χ² reduction across accepted rounds, plus the net ε / peak count so the
    churn can be read against whether the fit actually improved."""
    import numpy as np

    fit = result["fit"]
    rh = fit.rescue_history
    acc = [r for r in rh if r.accepted]
    drop = [
        (r.chi2_before - r.chi2_after) / r.chi2_before
        for r in acc
        if r.chi2_before > 0
        and np.isfinite(r.chi2_before)
        and np.isfinite(r.chi2_after)
    ]
    eps_p50, n_peaks = _fit_eps_summary(fit)
    return {
        "n_added": sum(r.n_rescue_added for r in acc),
        "n_pruned_rescue": sum(r.n_pruned_rescue_origin for r in rh),
        "n_merged": sum(r.n_merged for r in rh),
        "n_win": len({r.window_id for r in acc}),
        "n_rounds": len(rh),
        "chi2_drop_pct": round(float(np.median(drop)) * 100.0, 2) if drop else 0.0,
        "eps_p50": eps_p50,
        "n_peaks": n_peaks,
    }


def _metric_spur(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 5 fit to the spur gate's verdict: how many integer-MHz tones
    were masked, split by source (narrow / saturated), the mask half-width in
    bins, and the net ε / peak count. The catalogue is band-level (computed on the
    full active FT), so these are unaffected by the fit's plan reduction."""
    fit = result["fit"]
    params = fit.parameters
    srcs = list(params.get("spur_sources") or [])
    centers = params.get("spur_centers_mhz") or []
    eps_p50, n_peaks = _fit_eps_summary(fit)
    return {
        "n_spurs": int(params.get("n_spurs_gated", len(centers))),
        "n_narrow": sum(1 for s in srcs if "narrow" in s),
        "n_saturated": sum(1 for s in srcs if "saturated" in s),
        "mask_hw_bins": int(params.get("spur_mask_half_width_bins", 0) or 0),
        "eps_p50": eps_p50,
        "n_peaks": n_peaks,
    }


def _metric_thaw(result: Any) -> Dict[str, Any]:
    """Reduce a Stage 5 fit to the residual-edge renegotiation handshake: thaw and
    replan attempt / accept counts, the final plan revision, the flagged-edge
    S_coh tail (the trigger surface), the median coherence reduction on accepted
    thaws, and the net ε. Thaw is near-dormant on clean spectra (acceptance ~0),
    so the attempt counts and the flagged-edge tail carry the readout even when
    nothing is accepted."""
    import numpy as np

    fit = result["fit"]
    th = fit.thaw_history
    rp = fit.replan_history
    flagged = [
        t.edge_coherence_before for t in th if np.isfinite(t.edge_coherence_before)
    ]
    red = [
        t.edge_coherence_before - t.edge_coherence_after
        for t in th
        if t.accepted
        and np.isfinite(t.edge_coherence_before)
        and np.isfinite(t.edge_coherence_after)
    ]
    eps_p50, _ = _fit_eps_summary(fit)
    return {
        "n_thaw": len(th),
        "n_thaw_acc": sum(1 for t in th if t.accepted),
        "n_replan": len(rp),
        "n_replan_acc": sum(1 for x in rp if x.accepted),
        "rev": int(fit.final_plan_revision),
        "coh_flag_p95": round(float(np.percentile(flagged, 95)), 3) if flagged else 0.0,
        "coh_red_p50": round(float(np.median(red)), 3) if red else 0.0,
        "eps_p50": eps_p50,
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
_register(
    KnobSpec(
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
    )
)
_register(
    KnobSpec(
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
    )
)
_register(
    KnobSpec(
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
    )
)

# Stage 0 advanced detection internals (chirp-end localisation + sweep). The
# band_min_mhz / band_max_mhz integration-band override is intentionally not a
# sweep knob — the detector ignores it unless both edges are set, so neither
# sweeps meaningfully alone (reach them via settings= / preset=).
_START_COLS = ("start_us", "chirp_end_us", "chirp_detected")
_grid: Tuple[Any, ...]
for _path, _field, _help, _grid, _inst in (
    (
        "stage0.step_us",
        "step_us",
        "Start-time sweep step (us): the resolution of the Sigma|FT| curve.",
        (0.01, 0.02, 0.05),
        "maybe",
    ),
    (
        "stage0.floor_factor",
        "floor_factor",
        "Multiple of the floor at which Sigma|FT| is considered settled (chirp end).",
        (2.0, 3.0, 5.0),
        "maybe",
    ),
    (
        "stage0.floor_tail_us",
        "floor_tail_us",
        "Deep-tail width (us) whose median defines the settled floor.",
        (0.5, 1.0, 2.0),
        "N",
    ),
):
    _register(
        KnobSpec(
            path=_path,
            stage="start_detection",
            requires="stage0_fid_data",
            help=_help,
            inst_sensitivity=_inst,
            default_grid=_grid,
            run=_run_start(_field),
            metric=_metric_start,
            metric_columns=_START_COLS,
            plot=plot_start_detection,
            tier="advanced",
        )
    )

# FT window start time (Stage 1) — sweep the actual start and stack the
# resulting active-band spectra. Requires Stage 1 settings to be resolvable
# (built once); each value recomputes the FT.
_register(
    KnobSpec(
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
    )
)

# FT frequency trim + window end. The trim default grids are MHz-absolute and
# 2638-shaped; pass --grid for another instrument's band. The canonical FT is
# unconditionally unapodized and native-length, so there are no apodization /
# zero-pad knobs to expose; units_power is a display-scale choice surfaced by
# the resolved-settings view.
_FT_BAND_COLS = ("p1", "p5", "p10", "p20", "p50", "max")
_register(
    KnobSpec(
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
    )
)
_register(
    KnobSpec(
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
    )
)
_register(
    KnobSpec(
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
    )
)

# Stage 2 noise — scatter estimator (the canonical default). Requires Stage 1.
# The descriptors (help / tier / inst_sensitivity / grid) are read from the
# NoiseSettings field metadata — the single knob declaration site — so the
# registry carries only the sweep behavior (run / metric / plot). All knobs
# re-run Stage 2 and report the same sigma trend + sigma(f)-over-spectrum
# overlay.
_NOISE_COLS = ("median_sigma", "noise_fraction")
for _field in (
    "window_mhz",
    "pedestal_mhz",
    "smoothing_mhz",
    "line_k",
    "n_iter",
    "region_aware",
    "smoothing_percentile",
    "convolve_mhz",
):
    _km = field_knob_meta(noise_settings.NoiseSettings, _field)
    assert _km.grid is not None, f"stage2.{_field} registered without a sweep grid"
    _register(
        KnobSpec(
            path=f"stage2.{_field}",
            stage="stage2_noise",
            requires="stage1_complex_ft",
            help=_km.help,
            inst_sensitivity=_km.inst_sensitivity,
            default_grid=_km.grid,
            run=_run_noise(_field),
            metric=_metric_noise,
            metric_columns=_NOISE_COLS,
            plot=plot_noise_sweep,
            tier=_km.tier,
        )
    )

# Stage 2b tau calibration — requires Stages 0-2. Each value re-runs the STFT
# calibration (the slowest stage), so default grids are kept modest. The
# descriptors (help / tier / inst_sensitivity / grid) are read from the
# TauCalibrationSettings field metadata — the single knob declaration site —
# so the registry carries only the sweep behavior (run / metric / plot).
_TAU_COLS = ("tau_maj_us", "sigma_tau_us", "n_contributors")
_SHAPE_COLS = ("recommended_shape", "exp", "gauss", "voigt", "n_contributors")
_TAU_TWIN_SEE_ALSO = (
    "stage2b.gaussian.* is the Gaussian τ_G twin and stage2b.recommendation.* "
    "is the exp-vs-gauss shape vote — all three share the STFT contributor pool."
)


def _tau_knob(
    sub_block: str,
    field_name: str,
    *,
    metric: MetricFn,
    metric_columns: Tuple[str, ...],
    plot: Optional[Callable[..., Any]],
    see_also: Optional[str] = None,
) -> KnobSpec:
    """Build a Stage 2b ``KnobSpec`` reading its descriptors from the field.

    ``help`` / ``tier`` / ``inst_sensitivity`` / ``default_grid`` come from the
    :class:`TauCalibrationSettings` field metadata (single source); only the
    sweep behavior (``run`` / ``metric`` / ``plot``) and the structural
    ``path`` / ``requires`` are supplied here.
    """
    km = field_knob_meta(
        tau_calibration_settings.TauCalibrationSettings,
        f"{sub_block}.{field_name}",
    )
    assert (
        km.grid is not None
    ), f"stage2b.{sub_block}.{field_name} registered without a sweep grid"
    return KnobSpec(
        path=f"stage2b.{sub_block}.{field_name}",
        stage="stage2b_tau",
        requires="stage2_noise_result",
        help=km.help,
        inst_sensitivity=km.inst_sensitivity,
        default_grid=km.grid,
        run=_run_tau(sub_block, field_name),
        metric=metric,
        metric_columns=metric_columns,
        plot=plot,
        tier=km.tier,
        see_also=see_also,
    )


# Tau-metric knobs (every sub-block except recommendation): one trend view.
for _sub, _field in (
    ("stft", "n_seg"),
    ("stft", "t_sigma"),
    ("stft", "tau_max_us"),
    ("stft", "tau_max_factor"),
    ("stft", "rss_gate_factor"),
    ("stft", "relative_gate_fraction"),
    ("polish", "polish_n_iter"),
    ("polish", "polish_top_n"),
    ("polish", "polish_snr_cap"),
    ("polish", "polish_noise_debias"),
    ("aggregation", "min_contributors"),
    ("aggregation", "sigma_tau_fraction_max"),
    ("aggregation", "bimodality_dominant_fraction"),
    ("aggregation", "sigma_tau_floor_us"),
    ("aggregation", "spur_cluster_multiplier"),
    ("band", "compute_band_majorities"),
):
    _register(
        _tau_knob(
            _sub,
            _field,
            metric=_metric_tau,
            metric_columns=_TAU_COLS,
            plot=plot_tau_trend,
        )
    )

# Tau-metric knobs that point at the shared-pool twins via see_also.
_register(
    _tau_knob(
        "band",
        "min_contributors_per_band",
        metric=_metric_tau,
        metric_columns=_TAU_COLS,
        plot=plot_tau_trend,
        see_also=_TAU_TWIN_SEE_ALSO,
    )
)
_register(
    _tau_knob(
        "gaussian",
        "snr_min",
        metric=_metric_tau,
        metric_columns=_TAU_COLS,
        plot=plot_tau_trend,
        see_also=_TAU_TWIN_SEE_ALSO,
    )
)
for _field in (
    "tau_G_bound_lo",
    "tau_G_bound_hi",
    "delta_chi2r_min",
    "tau_G_upper_fraction",
    "min_contributors",
):
    _register(
        _tau_knob(
            "gaussian",
            _field,
            metric=_metric_tau,
            metric_columns=_TAU_COLS,
            plot=plot_tau_trend,
        )
    )

# Shape-vote knobs (recommendation sub-block): the shape-vote view.
_register(
    _tau_knob(
        "recommendation",
        "pure_margin_threshold",
        metric=_metric_shape,
        metric_columns=_SHAPE_COLS,
        plot=plot_shape_vote,
        see_also=_TAU_TWIN_SEE_ALSO,
    )
)
for _field in ("snr_min", "tau_bound_lo", "tau_bound_hi"):
    _register(
        _tau_knob(
            "recommendation",
            _field,
            metric=_metric_shape,
            metric_columns=_SHAPE_COLS,
            plot=plot_shape_vote,
        )
    )


# ---------------------------------------------------------------------------
# Stage 3 peak detection — requires Stages 1-2 (Stage 2b optional: when present
# the gap matched filter is anchored on τ_maj and shape-matched, else it falls
# back to the user apodization). Every Stage 3 sweep re-runs detect_peaks and
# renders the same view: a count/SNR trend over per-value × per-region spectrum
# panels showing which peaks each value finds, drops, and promotes.
# ---------------------------------------------------------------------------
_PEAK_COLS = (
    "n_total",
    "n_strong",
    "n_medium",
    "n_weak",
    "snr_min",
    "snr_p10",
    "snr_p25",
    "snr_p50",
    "snr_p90",
    "snr_max",
)
_PEAK_SEE_ALSO = (
    "the table counts the peaks passed to Stage 4 by SNR band (weak/medium/"
    "strong); the panels show where they land — read alongside the Stage 2 noise "
    "floor (stage2.*) and the primary's own apodized floor "
    "(stage3.primary_pass.noise_*)."
)


def _peak_knob(
    sub_block: str,
    field_name: str,
    *,
    metric: MetricFn,
    metric_columns: Tuple[str, ...],
    plot: Optional[Callable[..., Any]],
    see_also: Optional[str] = None,
) -> None:
    """Register a Stage 3 ``KnobSpec`` reading its descriptors from the field.

    ``help`` / ``tier`` / ``inst_sensitivity`` / ``default_grid`` come from the
    :class:`PeakDetectionSettings` field metadata (single source); only the
    sweep behavior (``run`` / ``metric`` / ``plot``) and the structural
    ``path`` / ``requires`` / ``see_also`` are supplied here.
    """
    km = field_knob_meta(
        peak_detection_settings.PeakDetectionSettings,
        f"{sub_block}.{field_name}",
    )
    assert (
        km.grid is not None
    ), f"stage3.{sub_block}.{field_name} registered without a sweep grid"
    _register(
        KnobSpec(
            path=f"stage3.{sub_block}.{field_name}",
            stage="stage3_peaks",
            requires="stage2_noise_result",
            help=km.help,
            inst_sensitivity=km.inst_sensitivity,
            default_grid=km.grid,
            run=_run_peaks(sub_block, field_name),
            metric=metric,
            metric_columns=metric_columns,
            plot=plot,
            tier=km.tier,
            see_also=see_also,
        )
    )


# Detection-shaping knobs that point at the noise-floor twins via see_also.
for _sub, _field in (
    ("promotion", "min_snr"),
    ("promotion", "internal_min_snr"),
    ("primary_pass", "min_exclusion_mhz"),
    ("primary_pass", "primary_leakage_floor_k"),
    ("gap_pass", "gap_leakage_floor_k"),
):
    _peak_knob(
        _sub,
        _field,
        metric=_metric_peaks,
        metric_columns=_PEAK_COLS,
        plot=plot_peak_detection,
        see_also=_PEAK_SEE_ALSO,
    )

# Every other Stage 3 knob: classification edges, the SavGol localiser, the
# primary-pass apodization + zpf, the apodized-domain σ scatter knobs, and the
# gap-pass structural toggles. All descriptors come from the field.
for _sub, _field in (
    ("promotion", "weak_medium_snr"),
    ("promotion", "medium_strong_snr"),
    ("savgol", "sg_window"),
    ("savgol", "sg_order"),
    ("savgol", "sg_fwhm_coverage"),
    ("savgol", "sg_min_window"),
    ("primary_pass", "primary_window"),
    ("primary_pass", "detection_zpf"),
    ("primary_pass", "noise_window_mhz"),
    ("primary_pass", "noise_pedestal_mhz"),
    ("primary_pass", "noise_smoothing_mhz"),
    ("primary_pass", "noise_line_k"),
    ("primary_pass", "noise_smoothing_percentile"),
    ("primary_pass", "noise_convolve_mhz"),
    ("primary_pass", "noise_n_iter"),
    ("primary_pass", "noise_region_aware"),
    ("gap_pass", "run_gap_pass"),
    ("gap_pass", "gap_active_zpf"),
):
    _peak_knob(
        _sub,
        _field,
        metric=_metric_peaks,
        metric_columns=_PEAK_COLS,
        plot=plot_peak_detection,
    )


# ---------------------------------------------------------------------------
# Stage 4 window assignment — requires Stage 3 (peaks). Every Stage 4 sweep
# re-runs assign_windows and renders the same view: a window-count trend, a
# band-wide boundary-shift overlay (each value's window edges coloured by
# value), then per-value × per-region zoom panels showing how the partition,
# its difficulty, and the driving S_coh statistic move across the grid.
# ---------------------------------------------------------------------------
_WINDOW_COLS = (
    "n_windows",
    "n_hard",
    "n_easy",
    "n_free",
    "n_fixed",
    "n_dep",
    "n_split",
    "width_p50",
    "width_p95",
    "width_max",
)
_WINDOW_SEE_ALSO = (
    "the table reports the plan shape (window/hard/contributor counts + the "
    "width distribution); the panels show how the boundaries move — read "
    "alongside the Stage 3 promoted peaks (stage3.*) that seed the partition."
)


def _window_knob(
    sub_block: str,
    field_name: str,
    *,
    see_also: Optional[str] = None,
) -> None:
    """Register a Stage 4 ``KnobSpec`` reading its descriptors from the field.

    ``help`` / ``tier`` / ``inst_sensitivity`` / ``default_grid`` come from the
    :class:`WindowPlanningSettings` field metadata (single source); only the
    sweep behavior (``run`` / ``metric`` / ``plot``) and the structural
    ``path`` / ``requires`` / ``see_also`` are supplied here.
    """
    km = field_knob_meta(
        window_planning_settings.WindowPlanningSettings,
        f"{sub_block}.{field_name}",
    )
    assert (
        km.grid is not None
    ), f"stage4.{sub_block}.{field_name} registered without a sweep grid"
    _register(
        KnobSpec(
            path=f"stage4.{sub_block}.{field_name}",
            stage="stage4_windows",
            requires="stage3_peaks",
            help=km.help,
            inst_sensitivity=km.inst_sensitivity,
            default_grid=km.grid,
            run=_run_windows(sub_block, field_name),
            metric=_metric_windows,
            metric_columns=_WINDOW_COLS,
            plot=plot_window_planning,
            tier=km.tier,
            see_also=see_also,
        )
    )


# Partition-shaping knobs that point back at the plan-shape view via see_also.
# (``leakage.tau_us`` is demoted to advanced on the field: the boxcar default
# only widens windows, absorbed downstream by split proposals, so it is a
# low-leverage control whose fate — keep, auto-feed the Stage 2b τ, or remove —
# is deferred to the cross-fixture audit, issue #6.)
for _sub, _field in (
    ("coherence", "edge_threshold"),
    ("clustering", "max_window_width_mhz"),
    ("contributor", "magnitude_attachment_threshold"),
    ("contributor", "min_freeze_snr"),
    ("leakage", "tau_us"),
    ("clustering", "min_window_half_width_points"),
    ("clustering", "max_window_width_points"),
):
    _window_knob(_sub, _field, see_also=_WINDOW_SEE_ALSO)

# Every other Stage 4 knob: the coherence band scales, the isolated-peak MHz
# margin, and the per-window cap. All descriptors come from the field.
for _sub, _field in (
    ("coherence", "edge_m"),
    ("coherence", "trim_m"),
    ("clustering", "min_window_half_width_mhz"),
    ("clustering", "max_peaks_per_window"),
):
    _window_knob(_sub, _field)


# ---------------------------------------------------------------------------
# Stage 5 fitting — requires Stage 4 (windows). Each sweep re-runs fit_peaks on
# a working copy whose plan the ``prepare`` hook has reduced to a representative
# window subset (top-SNR + seeded sample + frequency pins), so a value costs a
# few tens of fits, not the whole band. This is the *fit-quality* family
# (tau / conservative / penalties / seeder / baseline); rescue / spur / thaw get
# their own dedicated plots. The honest quality lens is the SNR-normalised
# shape-error fraction ε (pass ⇔ ε ≤ κ), not the SNR²-floored χ²ᵣ.
# ---------------------------------------------------------------------------
_FIT_COLS = (
    "eps_p50",
    "eps_p95",
    "n_fail",
    "n_peaks",
    "n_free_tau",
    "sigma_f_khz",
    "chi2r_p50",
    "chi2r_p95",
)
_FIT_SEE_ALSO = (
    "the headline is the SNR-normalised shape-error ε (pass ⇔ ε ≤ κ=0.05), not "
    "χ²ᵣ (which rides an SNR² floor); n_peaks / n_free_tau / sigma_f_khz track "
    "what the knob structurally moved. Sweeps a reduced window subset — widen it "
    "with --fit-top-snr / --fit-sample / --fit-freqs (or --fit-all)."
)


def _fit_knob(
    path: str,
    sub_block: str,
    field_name: str,
    help_: str,
    inst: str,
    grid: Tuple[Any, ...],
    tier: str = "primary",
    see_also: Optional[str] = None,
    select_hint: Optional[str] = None,
    metric: MetricFn = _metric_fit,
    metric_columns: Tuple[str, ...] = _FIT_COLS,
    plot: Optional[Callable[..., Any]] = plot_fit_quality,
) -> None:
    _register(
        KnobSpec(
            path=path,
            stage="stage5_fitting",
            requires="stage4_windows",
            help=help_,
            inst_sensitivity=inst,
            default_grid=grid,
            run=_run_fit(sub_block, field_name),
            metric=metric,
            metric_columns=metric_columns,
            plot=plot,
            tier=tier,
            see_also=see_also,
            prepare=reduce_plan_for_fit,
            select_hint=select_hint,
        )
    )


# Primary — the Y-rated fit-quality knobs (grids lifted from the
# stage5-gaussian-audit probes where one exists).
_fit_knob(
    "stage5.tau.fit_tau_min_snr",
    "tau",
    "fit_tau_min_snr",
    "In-window SNR above which τ is freed (the free-τ floor is the max of this "
    "and conservative.weak_window_snr_threshold; 10 = the weak-window floor).",
    "Y",
    (10.0, 25.0, 50.0, 100.0),
    see_also=_FIT_SEE_ALSO,
    select_hint="snr_threshold",
)
_fit_knob(
    "stage5.conservative.weak_window_snr_threshold",
    "conservative",
    "weak_window_snr_threshold",
    "In-window SNR floor for free-τ eligibility (hold τ fixed below).",
    "Y",
    (5.0, 10.0, 15.0, 20.0),
    see_also=_FIT_SEE_ALSO,
    select_hint="snr_threshold",
)
_fit_knob(
    "stage5.baseline.edge_threshold",
    "baseline",
    "edge_threshold",
    "S_coh threshold (max residual edge) gating the leakage-wing baseline refit.",
    "Y",
    (2.5, 3.5, 5.0, 8.0),
    see_also=_FIT_SEE_ALSO,
)
_fit_knob(
    "stage5.baseline.smooth_threshold",
    "baseline",
    "smooth_threshold",
    "Smooth-residual F-test (chi2-drop/dof) gating the baseline on an in-band "
    "leakage pedestal.",
    "Y",
    (20.0, 50.0, 100.0, 200.0),
    see_also=_FIT_SEE_ALSO,
)

# Advanced — tau shaping (the penalty / bounds / routing knobs).
_g: Tuple[Any, ...]
for _p, _f, _h, _g, _inst in (
    (
        "stage5.tau.tau0_us",
        "tau0_us",
        "Starting shared decay τ₀ (µs); None = runtime fallback (Stage 2b / T/3).",
        (None, 3.0, 5.0, 8.0),
        "maybe",
    ),
    (
        "stage5.tau.max_decay_factor",
        "max_decay_factor",
        "τ bounds multiplier: τ ∈ [τ₀/k, τ₀·k].",
        (3.0, 5.0, 8.0),
        "N",
    ),
    (
        "stage5.tau.tau_penalty_lambda",
        "tau_penalty_lambda",
        "Strength of the bidirectional Gaussian prior on τ.",
        (10.0, 50.0, 100.0),
        "N",
    ),
    (
        "stage5.tau.tau_penalty_n_sigma",
        "tau_penalty_n_sigma",
        "τ-bound half-width in units of σ_τ from Stage 2b.",
        (3.0, 5.0, 8.0),
        "N",
    ),
):
    _fit_knob(_p, "tau", _f, _h, _inst, _g, tier="advanced")
_fit_knob(
    "stage5.tau.per_band_tau",
    "tau",
    "per_band_tau",
    "Route τ to per-band majorities (True) or a single band-wide τ (False).",
    "maybe",
    (False, True),
    tier="advanced",
)

# Advanced — the conservative add-one-peak loop.
for _p, _f, _h, _g in (
    (
        "stage5.conservative.significance",
        "significance",
        "F-test significance α for add-one-peak acceptance.",
        (0.01, 0.05, 0.1),
    ),
    (
        "stage5.conservative.max_peaks",
        "max_peaks",
        "Hard cap on the final peak count per window; 0 = no cap "
        "(candidate/patience-bounded).",
        (0, 8, 16),
    ),
    (
        "stage5.conservative.patience",
        "patience",
        "Consecutive-rejection patience before the add loop stops.",
        (1, 2, 3),
    ),
    (
        "stage5.conservative.min_separation_factor",
        "min_separation_factor",
        "Minimum peak separation (FWHM units; unresolvable below).",
        (0.5, 1.0, 1.5),
    ),
    (
        "stage5.conservative.min_pair_separation_factor",
        "min_pair_separation_factor",
        "Post-escalation pair-separation floor (FWHM units).",
        (0.25, 0.5, 0.75),
    ),
    (
        "stage5.conservative.min_pair_separation_resolution_factor",
        "min_pair_separation_resolution_factor",
        "Resolution-referenced pair floor (1/T_active elements).",
        (0.5, 1.0, 1.5),
    ),
    (
        "stage5.conservative.max_nfev",
        "max_nfev",
        "Solver evaluation cap per window.",
        (1000, 2000, 4000),
    ),
):
    _fit_knob(_p, "conservative", _f, _h, "N", _g, tier="advanced")

# Advanced — the soft penalties.
for _p, _f, _h, _g in (
    (
        "stage5.penalties.phase_penalty_lambda",
        "phase_penalty_lambda",
        "Phase-difference soft-penalty strength.",
        (50.0, 100.0, 200.0),
    ),
    (
        "stage5.penalties.phase_penalty_cutoff_fwhm",
        "phase_penalty_cutoff_fwhm",
        "Phase-penalty range (FWHM units; zero in quadrature).",
        (1.0, 2.0, 3.0),
    ),
    (
        "stage5.penalties.amp_penalty_lambda",
        "amp_penalty_lambda",
        "Amplitude-floor soft-penalty strength.",
        (5.0, 10.0, 20.0),
    ),
    (
        "stage5.penalties.amp_max_headroom",
        "amp_max_headroom",
        "Hard amplitude ceiling as a multiple of 2·max_data/τ_eff_min.",
        (2.0, 3.0, 5.0),
    ),
):
    _fit_knob(_p, "penalties", _f, _h, "N", _g, tier="advanced")

# Advanced — the blend-aware re-seeder.
for _p, _f, _h, _g in (
    (
        "stage5.seeder.seeder_rchi2",
        "seeder_rchi2",
        "χ²ᵣ threshold that triggers the K=2/3 blend-aware re-seed.",
        (1.2, 1.5, 2.0),
    ),
    (
        "stage5.seeder.seeder_straddle_factor",
        "seeder_straddle_factor",
        "Re-seed offset spacing in line-FWHM units.",
        (0.5, 1.0, 1.5),
    ),
    (
        "stage5.seeder.seeder_max_k",
        "seeder_max_k",
        "Maximum blend-escalation depth.",
        (2, 3, 4),
    ),
):
    _fit_knob(_p, "seeder", _f, _h, "N", _g, tier="advanced")

# Advanced — the leakage-wing baseline shape / switch.
_fit_knob(
    "stage5.baseline.order",
    "baseline",
    "order",
    "Baseline polynomial order (0 = const, 1 = linear; higher overfits).",
    "maybe",
    (0, 1),
    tier="advanced",
)
_fit_knob(
    "stage5.baseline.enabled",
    "baseline",
    "enabled",
    "Master switch for the evidence-triggered leakage-wing baseline term.",
    "N",
    (False, True),
    tier="advanced",
)

# Advanced — doublet-alternative observation pass (observation-only; never
# changes fitted peaks).
_fit_knob(
    "stage5.doublet_alternative.enabled",
    "doublet_alternative",
    "enabled",
    "Master switch for the post-fit doublet-alternative observation pass "
    "(attaches records, never modifies fitted peaks).",
    "N",
    (False, True),
    tier="advanced",
)
_fit_knob(
    "stage5.doublet_alternative.k_res",
    "doublet_alternative",
    "k_res",
    "Sub-resolution separation threshold (1/T_active elements) for doublet "
    "adjudication; pairs closer than k_res are evaluated.",
    "N",
    (1.0, 1.5, 2.0, 2.5),
    tier="advanced",
)
_fit_knob(
    "stage5.doublet_alternative.r_min",
    "doublet_alternative",
    "r_min",
    "Minimum amplitude ratio for the weaker member to trigger doublet evaluation "
    "(suppresses ghost pairs beside strong lines).",
    "N",
    (0.02, 0.05, 0.1),
    tier="advanced",
)


# ---------------------------------------------------------------------------
# Stage 5 rescue / spur / thaw families — same ``_run_fit`` runner and window
# reduction as the fit-quality family, but each reads a distinct persisted
# renegotiation history and gets its own provenance plot (rescue adds residual
# lines, spur masks integer-MHz tones, thaw re-co-fits contested window edges),
# so they carry their own metric columns + plot adapter.
# ---------------------------------------------------------------------------
_RESCUE_COLS = (
    "n_added",
    "n_pruned_rescue",
    "n_merged",
    "n_win",
    "n_rounds",
    "chi2_drop_pct",
    "eps_p50",
    "n_peaks",
)
_RESCUE_SEE_ALSO = (
    "n_added / chi2_drop_pct say whether rescue earns its keep; n_pruned_rescue "
    "is the failsafe (lines rescue added that a later refit undid). The panels "
    "show where on the band rescue fires and the candidate SNRs vs the gate. "
    "Sweeps a reduced window subset — widen with --fit-* (or --fit-all)."
)
_SPUR_COLS = (
    "n_spurs",
    "n_narrow",
    "n_saturated",
    "mask_hw_bins",
    "eps_p50",
    "n_peaks",
)
_SPUR_SEE_ALSO = (
    "the gated-spur catalogue is band-level (computed on the full active FT), so "
    "spur counts are immune to the window reduction; the overlay shows which "
    "integer-MHz tones each value masks, coloured by the last value still "
    "gating them."
)
_THAW_COLS = (
    "n_thaw",
    "n_thaw_acc",
    "n_replan",
    "n_replan_acc",
    "rev",
    "coh_flag_p95",
    "coh_red_p50",
    "eps_p50",
)
_THAW_SEE_ALSO = (
    "thaw is near-dormant on clean spectra (n_thaw_acc ~0 on 2638); the attempt "
    "counts + the flagged-edge S_coh tail (coh_flag_p95) carry the readout. The "
    "before→after panel shows whether the co-fit moved the contested edge. Note "
    "literal boundary moves are not persisted — only the coherence handshake."
)


# Rescue — primary: the two residual-detection gates (Y-rated). Advanced: the
# safety cap, the cleanup F-test, and the merge / overfit-absorber factors.
_fit_knob(
    "stage5.rescue.snr_threshold",
    "rescue",
    "snr_threshold",
    "Residual-peak detection floor (nominates generously; the F-test gates "
    "acceptance).",
    "Y",
    (2.0, 2.5, 3.0, 4.0),
    see_also=_RESCUE_SEE_ALSO,
    metric=_metric_rescue,
    metric_columns=_RESCUE_COLS,
    plot=plot_rescue,
)
_fit_knob(
    "stage5.rescue.prominence_threshold",
    "rescue",
    "prominence_threshold",
    "Residual-peak prominence threshold for candidate nomination.",
    "Y",
    (1.5, 2.0, 3.0, 4.0),
    see_also=_RESCUE_SEE_ALSO,
    metric=_metric_rescue,
    metric_columns=_RESCUE_COLS,
    plot=plot_rescue,
)
for _p, _f, _h, _g in (
    (
        "stage5.rescue.max_rounds",
        "max_rounds",
        "Maximum residual-rescue iterations per window (safety cap).",
        (1, 3, 5, 8),
    ),
    (
        "stage5.rescue.cleanup_significance",
        "cleanup_significance",
        "F-test significance for the remove-and-refit post-rescue cleanup.",
        (0.01, 0.05, 0.1),
    ),
    (
        "stage5.rescue.merge_separation_factor",
        "merge_separation_factor",
        "AICc-gated merge threshold above resolution (FWHM units).",
        (0.25, 0.5, 0.75),
    ),
    (
        "stage5.rescue.structural_merge_factor",
        "structural_merge_factor",
        "Sub-resolution merge floor: pairs closer than this (FWHM units) collapse "
        "unconditionally.",
        (0.25, 0.5, 0.75),
    ),
    (
        "stage5.rescue.overfit_amp_ratio_band",
        "overfit_amp_ratio_band",
        "Upper bound (1/T_active elements) of the amplitude-ratio merge tier that "
        "collapses supra-resolution shape-error absorbers.",
        (1.0, 1.5, 2.0),
    ),
    (
        "stage5.rescue.overfit_amp_ratio_threshold",
        "overfit_amp_ratio_threshold",
        "Amplitude ratio above which a pair in the band collapses as an absorber "
        "(0 disables).",
        (0.0, 4.0, 6.0, 10.0),
    ),
):
    _fit_knob(
        _p,
        "rescue",
        _f,
        _h,
        "N",
        _g,
        tier="advanced",
        see_also=_RESCUE_SEE_ALSO,
        metric=_metric_rescue,
        metric_columns=_RESCUE_COLS,
        plot=plot_rescue,
    )

# Spur — primary: the integer-MHz / narrowness gate + the mask half-width (all
# Y-rated). Advanced: the master switch, the frequency-domain SNR floor, and the
# Stage 2b saturated-catalogue toggle.
for _p, _f, _h, _g in (
    (
        "stage5.spur.integer_tol_mhz",
        "integer_tol_mhz",
        "Max distance (MHz) from an integer MHz for the spur gate's hard integer "
        "requirement (~½ active-FT bin).",
        (0.02, 0.04, 0.08, 0.16),
    ),
    (
        "stage5.spur.narrowness_ratio",
        "narrowness_ratio",
        "max(neighbour)/peak below which an integer-MHz bin is sub-resolution "
        "narrow (a CW tone vs a real line with a skirt).",
        (0.2, 0.3, 0.4, 0.5),
    ),
    (
        "stage5.spur.mask_half_width_bins",
        "mask_half_width_bins",
        "Residual-mask half-width (active-FT bins) around a detected spur.",
        (1, 2, 3, 4),
    ),
):
    _fit_knob(
        _p,
        "spur",
        _f,
        _h,
        "Y",
        _g,
        see_also=_SPUR_SEE_ALSO,
        metric=_metric_spur,
        metric_columns=_SPUR_COLS,
        plot=plot_spur,
    )
for _p, _f, _h, _g in (
    (
        "stage5.spur.enabled",
        "enabled",
        "Master switch for clock/LO-spur detection + masking.",
        (False, True),
    ),
    (
        "stage5.spur.snr_threshold",
        "snr_threshold",
        "Peak-bin / σ_c floor for the frequency-domain spur detector.",
        (3.0, 5.0, 8.0, 12.0),
    ),
    (
        "stage5.spur.use_stft_catalogue",
        "use_stft_catalogue",
        "Consume the persisted Stage 2b flat-spur (saturated) catalogue as the "
        "gate's persistence half; False = frequency-domain detector only.",
        (False, True),
    ),
):
    _fit_knob(
        _p,
        "spur",
        _f,
        _h,
        "N",
        _g,
        tier="advanced",
        see_also=_SPUR_SEE_ALSO,
        metric=_metric_spur,
        metric_columns=_SPUR_COLS,
        plot=plot_spur,
    )

# Thaw — primary: the residual-edge S_coh trigger (Y-rated). Advanced: the
# thaw / replan round caps and the edge-detection band width.
_fit_knob(
    "stage5.thaw.residual_edge_threshold",
    "thaw",
    "residual_edge_threshold",
    "S_coh threshold for a residual-edge-coherence boundary violation (the "
    "thaw / replan trigger).",
    "Y",
    (4.0, 6.0, 8.0, 10.0, 12.0),
    see_also=_THAW_SEE_ALSO,
    metric=_metric_thaw,
    metric_columns=_THAW_COLS,
    plot=plot_thaw,
)
for _p, _f, _h, _g in (
    (
        "stage5.thaw.max_thaw_rounds",
        "max_thaw_rounds",
        "Maximum local-thaw iterations (re-fit a frozen contributor; 0 disables).",
        (0, 1, 2, 3),
    ),
    (
        "stage5.thaw.max_replan_rounds",
        "max_replan_rounds",
        "Maximum structural-replan iterations (window-boundary merges; 0 disables).",
        (0, 1, 2, 3),
    ),
    (
        "stage5.thaw.residual_edge_m",
        "residual_edge_m",
        "Band width (bins) for residual edge-coherence detection.",
        (16, 32, 48, 64),
    ),
):
    _fit_knob(
        _p,
        "thaw",
        _f,
        _h,
        "N",
        _g,
        tier="advanced",
        see_also=_THAW_SEE_ALSO,
        metric=_metric_thaw,
        metric_columns=_THAW_COLS,
        plot=plot_thaw,
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
            s
            for s in specs
            if s.path == selector
            or s.path.startswith(selector + ".")
            or s.stage == selector
        ]
    return tuple(specs)
