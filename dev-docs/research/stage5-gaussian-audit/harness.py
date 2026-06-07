"""Shared harness for the Stage 5 Gaussian-path parameter optimization audit.

Mirrors the Stage 4 sibling under
``dev-docs/research/stage4-gaussian-audit/harness.py`` but drives
``fit_peaks`` instead of ``assign_windows`` and emits *per-window*
records so Step 5 of the optimization loop (cross-reference flagged
windows against the variant grid) can filter the CSV by ``variant_id``
without a join.

Fixture layout per shape, matching the Stage 4 audit:

* Stages 0-2 with ``trim=(26500, 40000)`` on the canonical unapodized,
  native-length FT.
* Stage 2b Lorentzian + Gaussian twins both present.
* ``recommended_shape`` stamped on the requested shape.
* Stage 3 peak list persisted using the shape-aware τ-feeder.
* Stage 4 windows persisted using the boxcar ``leakage.tau_us``
  default (the Stage 4 audit's verdict).

The cached fixtures live at
``scratch/stage5-gaussian-audit/exp_2638_unapodized_{lorentzian,gaussian}.ftmw``;
per-variant working copies land under
``scratch/stage5-gaussian-audit/runs/<variant_id>.ftmw``. ``variant_id``
is ``baseline__<shape>`` when every knob carries its hard default and
``<knob>__<value>__<shape>`` otherwise -- the dedupe lets the five
probes share the baseline row across knobs (saves five fits per shape).
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage2b_g_impl import tau_G_calibration_present
from ftmwpipeline._internal.stage2b_impl import tau_calibration_present
from ftmwpipeline.core.stage_fit_settings import StageFitSettings, _HARD_DEFAULTS
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_DIR = REPO_ROOT / "scratch" / "stage5-gaussian-audit"
RUNS_DIR = SCRATCH_DIR / "runs"
EXAMPLE_2638 = REPO_ROOT / "examples" / "blackchirp_data" / "2638"
# Re-use Stages 0-3 fixtures from the Stage 4 audit when available so we
# don't pay the import+FT+τ+detect cost twice (the two audits build the
# same fixture; only Stage 4 differs in how the per-variant runs are set up).
STAGE4_CACHE = REPO_ROOT / "scratch" / "stage4-gaussian-audit"

FT_TRIM_MHZ: Tuple[float, float] = (26500.0, 40000.0)

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
PER_WINDOW_CSV = DATA_DIR / "probe_all_per_window.csv"
SUMMARY_CSV = DATA_DIR / "probe_all_variants_summary.csv"

logger = logging.getLogger("stage5-gaussian-audit")


PER_WINDOW_FIELDS: List[str] = [
    "variant_id",
    "knob",
    "knob_value",
    "shape",
    "window_id",
    "freq_lo_mhz",
    "freq_hi_mhz",
    "chi2r",
    "aic",
    "n_fitted_peaks",
    "n_fixed_contributors",
    "success",
]

SUMMARY_FIELDS: List[str] = [
    "variant_id",
    "knob",
    "knob_value",
    "shape",
    "n_windows",
    "chi2r_median",
    "chi2r_p95",
    "chi2r_max",
    "n_chi2r_gt_5",
    "n_chi2r_gt_10",
    "n_fitted_peaks_total",
    "n_fixed_total",
    "n_success",
    "runtime_s",
]


# Knob -> (sub-block, field, hard-default) lookup; lets variant_id resolve
# whether a value is the baseline without round-tripping through resolve().
KNOB_SPEC: Dict[str, Tuple[str, str, Any]] = {
    "tau.fit_tau_min_snr": ("tau", "fit_tau_min_snr", _HARD_DEFAULTS["tau"]["fit_tau_min_snr"]),
    "conservative.weak_window_snr_threshold": (
        "conservative",
        "weak_window_snr_threshold",
        _HARD_DEFAULTS["conservative"]["weak_window_snr_threshold"],
    ),
    "rescue.snr_threshold": (
        "rescue",
        "snr_threshold",
        _HARD_DEFAULTS["rescue"]["snr_threshold"],
    ),
    "rescue.prominence_threshold": (
        "rescue",
        "prominence_threshold",
        _HARD_DEFAULTS["rescue"]["prominence_threshold"],
    ),
    "thaw.residual_edge_threshold": (
        "thaw",
        "residual_edge_threshold",
        _HARD_DEFAULTS["thaw"]["residual_edge_threshold"],
    ),
}


@dataclass
class WindowRow:
    variant_id: str
    knob: str
    knob_value: Any
    shape: str
    window_id: int
    freq_lo_mhz: float
    freq_hi_mhz: float
    chi2r: float
    aic: float
    n_fitted_peaks: int
    n_fixed_contributors: int
    success: bool


@dataclass
class VariantSummary:
    variant_id: str
    knob: str
    knob_value: Any
    shape: str
    n_windows: int
    chi2r_median: float
    chi2r_p95: float
    chi2r_max: float
    n_chi2r_gt_5: int
    n_chi2r_gt_10: int
    n_fitted_peaks_total: int
    n_fixed_total: int
    n_success: int
    runtime_s: float


def _fmt_knob_value(value: Any) -> str:
    """Stable filename-safe rendering of a knob value."""
    if value is None:
        return "None"
    if isinstance(value, float):
        # 1.5 -> "1p5", 50.0 -> "50p0".
        return f"{value:g}".replace(".", "p").replace("-", "neg")
    return str(value).replace(".", "p").replace("/", "_")


def variant_id_for(
    knob: str,
    value: Any,
    shape: str,
) -> str:
    """``baseline__<shape>`` when ``value`` equals the knob's hard default;
    otherwise ``<knob>__<value>__<shape>``.

    The dedupe lets every probe share the baseline fit across knobs --
    Step 1's run-all driver pays the baseline cost once per shape rather
    than once per knob × shape.
    """
    spec = KNOB_SPEC.get(knob)
    if spec is None:
        raise ValueError(f"unknown knob {knob!r}; add to KNOB_SPEC")
    _, _, default_value = spec
    if value == default_value:
        return f"baseline__{shape}"
    knob_short = knob.replace(".", "_")
    return f"{knob_short}__{_fmt_knob_value(value)}__{shape}"


def settings_for(knob: str, value: Any) -> StageFitSettings:
    """Build a sparse :class:`StageFitSettings` pinning the one knob.

    Other knobs stay ``None`` so the resolver falls through to the
    hard default (every probe sweeps one knob in isolation).
    """
    spec = KNOB_SPEC.get(knob)
    if spec is None:
        raise ValueError(f"unknown knob {knob!r}")
    sub_name, field_name, _ = spec
    s = StageFitSettings()
    setattr(getattr(s, sub_name), field_name, value)
    return s


# ---------------------------------------------------------------------------
# Fixture preparation
# ---------------------------------------------------------------------------
def prepare_fixture(shape: str, *, force_rebuild: bool = False) -> Path:
    """Return a Stages 0-4 prepped fixture for ``shape`` (cached).

    Identical to the Stage 4 audit's fixture except that Stage 4 itself is
    also pre-run (the Stage 5 sweep doesn't need to rerun window planning,
    only the fit). Stage 4 uses the boxcar ``leakage.tau_us`` default (the
    Stage 4 audit verdict).
    """
    if shape not in {"lorentzian", "gaussian"}:
        raise ValueError(f"shape must be 'lorentzian' or 'gaussian', got {shape!r}")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    fp = SCRATCH_DIR / f"exp_2638_unapodized_{shape}.ftmw"
    if fp.exists() and not force_rebuild:
        logger.info("Reusing cached fixture %s", fp)
        write_stage2b_recommended_shape(str(fp), shape=shape)
        return fp

    # Stages 0-3 are identical between the Stage 4 and Stage 5 audits.
    # If the Stage 4 audit's fixture is present, copy it and run only
    # Stage 4 on top -- skips ~4 min of import+FT+τ+detect work.
    stage4_src = STAGE4_CACHE / f"exp_2638_unapodized_{shape}.ftmw"
    if stage4_src.exists():
        logger.info("Seeding %s from Stage 4 audit cache %s", fp, stage4_src)
        shutil.copy(stage4_src, fp)
        write_stage2b_recommended_shape(str(fp), shape=shape)
        logger.info("  running assign_windows ...")
        ftmw.assign_windows(str(fp))
        logger.info("Fixture %s ready (recommended_shape=%s)", fp.name, shape)
        return fp

    logger.info(
        "Preparing %s fixture at %s (import + FT + noise + 2x Stage 2b + Stage 3 + Stage 4)",
        shape, fp,
    )
    ftmw.import_data(str(fp), source=str(EXAMPLE_2638), force=True)
    ftmw.compute_ft(str(fp), trim=FT_TRIM_MHZ)
    ftmw.estimate_noise(str(fp))
    if not tau_calibration_present(str(fp)):
        logger.info("  running calibrate_tau (Lorentzian twin) ...")
        ftmw.calibrate_tau(str(fp))
    if not tau_G_calibration_present(str(fp)):
        logger.info("  running calibrate_tau_G (Gaussian twin) ...")
        ftmw.calibrate_tau_G(str(fp))
    write_stage2b_recommended_shape(str(fp), shape=shape)
    logger.info("  running detect_peaks (shape-aware τ-feeder) ...")
    ftmw.detect_peaks(str(fp))
    logger.info("  running assign_windows ...")
    ftmw.assign_windows(str(fp))
    logger.info("Fixture %s ready (recommended_shape=%s)", fp.name, shape)
    return fp


# ---------------------------------------------------------------------------
# CSV bootstrap / append helpers
# ---------------------------------------------------------------------------
def reset_csvs() -> None:
    """Truncate both audit CSVs and write the headers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PER_WINDOW_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(PER_WINDOW_FIELDS)
    with SUMMARY_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(SUMMARY_FIELDS)
    logger.info("Reset CSVs at %s + %s", PER_WINDOW_CSV, SUMMARY_CSV)


def existing_variant_ids() -> set[str]:
    """Variant ids already present in the per-window CSV (dedupe key)."""
    if not PER_WINDOW_CSV.exists():
        return set()
    out: set[str] = set()
    with PER_WINDOW_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.add(row["variant_id"])
    return out


def _append_per_window(rows: Iterable[WindowRow]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PER_WINDOW_CSV.exists()
    with PER_WINDOW_CSV.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(PER_WINDOW_FIELDS)
        for r in rows:
            w.writerow([
                r.variant_id, r.knob, r.knob_value, r.shape,
                r.window_id, f"{r.freq_lo_mhz:.4f}", f"{r.freq_hi_mhz:.4f}",
                f"{r.chi2r:.6f}", f"{r.aic:.4f}",
                r.n_fitted_peaks, r.n_fixed_contributors,
                int(r.success),
            ])


def _append_summary(summary: VariantSummary) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(SUMMARY_FIELDS)
        w.writerow([
            summary.variant_id, summary.knob, summary.knob_value, summary.shape,
            summary.n_windows,
            f"{summary.chi2r_median:.6f}",
            f"{summary.chi2r_p95:.6f}",
            f"{summary.chi2r_max:.6f}",
            summary.n_chi2r_gt_5, summary.n_chi2r_gt_10,
            summary.n_fitted_peaks_total, summary.n_fixed_total,
            summary.n_success,
            f"{summary.runtime_s:.2f}",
        ])


def _per_window_rows_for_variant(variant_id: str) -> List[WindowRow]:
    """Load existing per-window rows for a variant_id (used for baseline reuse)."""
    if not PER_WINDOW_CSV.exists():
        return []
    rows: List[WindowRow] = []
    with PER_WINDOW_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["variant_id"] != variant_id:
                continue
            rows.append(WindowRow(
                variant_id=row["variant_id"],
                knob=row["knob"],
                knob_value=row["knob_value"],
                shape=row["shape"],
                window_id=int(row["window_id"]),
                freq_lo_mhz=float(row["freq_lo_mhz"]),
                freq_hi_mhz=float(row["freq_hi_mhz"]),
                chi2r=float(row["chi2r"]),
                aic=float(row["aic"]),
                n_fitted_peaks=int(row["n_fitted_peaks"]),
                n_fixed_contributors=int(row["n_fixed_contributors"]),
                success=bool(int(row["success"])),
            ))
    return rows


def _summarize(
    variant_id: str,
    knob: str,
    knob_value: Any,
    shape: str,
    rows: List[WindowRow],
    runtime_s: float,
) -> VariantSummary:
    chi2r = np.array([
        r.chi2r for r in rows if np.isfinite(r.chi2r)
    ])
    n_peaks = sum(r.n_fitted_peaks for r in rows)
    n_fixed = sum(r.n_fixed_contributors for r in rows)
    n_success = sum(1 for r in rows if r.success)
    if chi2r.size == 0:
        return VariantSummary(
            variant_id=variant_id, knob=knob, knob_value=knob_value,
            shape=shape, n_windows=len(rows),
            chi2r_median=float("nan"), chi2r_p95=float("nan"),
            chi2r_max=float("nan"),
            n_chi2r_gt_5=0, n_chi2r_gt_10=0,
            n_fitted_peaks_total=n_peaks, n_fixed_total=n_fixed,
            n_success=n_success, runtime_s=runtime_s,
        )
    return VariantSummary(
        variant_id=variant_id, knob=knob, knob_value=knob_value,
        shape=shape, n_windows=len(rows),
        chi2r_median=float(np.median(chi2r)),
        chi2r_p95=float(np.percentile(chi2r, 95)),
        chi2r_max=float(chi2r.max()),
        n_chi2r_gt_5=int(np.sum(chi2r > 5.0)),
        n_chi2r_gt_10=int(np.sum(chi2r > 10.0)),
        n_fitted_peaks_total=n_peaks,
        n_fixed_total=n_fixed,
        n_success=n_success,
        runtime_s=runtime_s,
    )


# ---------------------------------------------------------------------------
# fit_peaks runner
# ---------------------------------------------------------------------------
def _extract_window_rows(
    fit, variant_id: str, knob: str, knob_value: Any, shape: str,
) -> List[WindowRow]:
    rows: List[WindowRow] = []
    for wf in fit.window_fits:
        win = wf.window
        wid = wf.window_id if wf.window_id is not None else (
            win.window_id if win is not None else -1
        )
        if win is not None:
            fr = win.freq_range
            freq_lo = float(fr[0])
            freq_hi = float(fr[1])
        else:
            freq_lo = float("nan")
            freq_hi = float("nan")
        n_fixed = len([
            k for k in wf.fixed_parameters
            if k.startswith("frozen_peak_")
        ])
        rows.append(WindowRow(
            variant_id=variant_id,
            knob=knob,
            knob_value=knob_value,
            shape=shape,
            window_id=int(wid),
            freq_lo_mhz=freq_lo,
            freq_hi_mhz=freq_hi,
            chi2r=float(wf.reduced_chi2),
            aic=float(wf.aic),
            n_fitted_peaks=int(wf.n_peaks_fitted),
            n_fixed_contributors=int(n_fixed),
            success=bool(wf.success),
        ))
    return rows


def run_variant(
    fixture: Path,
    *,
    shape: str,
    knob: str,
    knob_value: Any,
    reuse_baseline: bool = True,
) -> VariantSummary:
    """Run ``fit_peaks`` on a working copy of ``fixture`` with the one
    knob pinned to ``knob_value``.

    Per-window rows are appended to :data:`PER_WINDOW_CSV` and a summary
    row to :data:`SUMMARY_CSV`. If ``reuse_baseline`` is True and the
    knob value is the hard default, this re-uses an existing
    ``baseline__<shape>`` per-window block (skipping the fit entirely)
    and still emits the summary row keyed to this knob's name.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    vid = variant_id_for(knob, knob_value, shape)

    # Cached baseline reuse -- saves up to 5 redundant fits per shape.
    if reuse_baseline and vid.startswith("baseline__"):
        cached = _per_window_rows_for_variant(vid)
        if cached:
            logger.info(
                "Reusing cached baseline rows for %s (n=%d) under %s=%s",
                vid, len(cached), knob, knob_value,
            )
            # Re-tag the rows for this probe so the summary CSV knows which
            # knob asked for it; the variant_id stays "baseline__<shape>"
            # (shared key, but the knob column lets the picker tell which
            # baseline row to compare its grid against).
            relabeled = [
                WindowRow(
                    variant_id=r.variant_id, knob=knob, knob_value=knob_value,
                    shape=r.shape, window_id=r.window_id,
                    freq_lo_mhz=r.freq_lo_mhz, freq_hi_mhz=r.freq_hi_mhz,
                    chi2r=r.chi2r, aic=r.aic,
                    n_fitted_peaks=r.n_fitted_peaks,
                    n_fixed_contributors=r.n_fixed_contributors,
                    success=r.success,
                )
                for r in cached
            ]
            summary = _summarize(vid, knob, knob_value, shape, relabeled, 0.0)
            _append_summary(summary)
            return summary

    work = RUNS_DIR / f"{vid}.ftmw"
    shutil.copy(fixture, work)
    write_stage2b_recommended_shape(str(work), shape=shape)

    settings = settings_for(knob, knob_value)
    logger.info(
        "fit_peaks  variant=%s  knob=%s  value=%s  shape=%s",
        vid, knob, knob_value, shape,
    )
    t0 = time.perf_counter()
    fit = ftmw.fit_peaks(str(work), settings=settings)
    runtime = time.perf_counter() - t0
    logger.info(
        "  done in %.1f s: %d windows, %d fitted peaks",
        runtime, fit.n_windows, fit.n_fitted_peaks,
    )

    rows = _extract_window_rows(fit, vid, knob, knob_value, shape)
    _append_per_window(rows)
    summary = _summarize(vid, knob, knob_value, shape, rows, runtime)
    _append_summary(summary)
    return summary


__all__ = [
    "FT_TRIM_MHZ",
    "KNOB_SPEC",
    "PER_WINDOW_CSV",
    "PER_WINDOW_FIELDS",
    "RUNS_DIR",
    "SCRATCH_DIR",
    "SUMMARY_CSV",
    "SUMMARY_FIELDS",
    "VariantSummary",
    "WindowRow",
    "existing_variant_ids",
    "prepare_fixture",
    "reset_csvs",
    "run_variant",
    "settings_for",
    "variant_id_for",
]
