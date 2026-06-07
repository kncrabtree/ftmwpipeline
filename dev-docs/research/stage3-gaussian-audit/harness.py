"""Shared harness for the Stage 3 Gaussian-path defaults audit.

Each per-knob probe under ``dev-docs/research/stage3-gaussian-audit/`` calls
into this module so the heavy steps (fixture prep, Stage 2b L+G calibration,
recommended-shape stamping) happen once per shape and the probes themselves
are thin sweep drivers.

The 2638 fixture is the only one we have. The harness prepares two
per-shape working copies of it -- ``exp_2638_unapodized_lorentzian.ftmw``
and ``exp_2638_unapodized_gaussian.ftmw`` -- under
``scratch/stage3-gaussian-audit/``. Probes that exhaust the heavy
fixture prep then re-run ``detect_peaks`` per grid point with the
target knob override; each call is on the order of seconds, not
minutes.

Fixture layout per working copy:

* Stages 0-2 done with ``trim=(26500, 40000)`` on the canonical
  unapodized, native-length FT (the spectrum the Stage 2b STFT
  classifier consumes).
* Stage 2b Lorentzian (``calibrate_tau``) + Gaussian
  (``calibrate_tau_G``) both present so the shape-aware τ-feeder for
  the gap pass can pick either branch.
* The ``recommended_shape`` attr is force-stamped on both Stage 2b
  groups so the resolver picks the requested shape regardless of what
  the 3-way classifier voted.

Why both calibrations on both copies: the τ-feeder reads both
``tau_calibration`` and ``tau_G_calibration`` independently and routes
on the ``recommended_shape`` attr. Carrying both anchors on every
working copy keeps the harness symmetric and the τ-substitution probe
free of cross-fixture noise.
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage2b_g_impl import tau_G_calibration_present
from ftmwpipeline._internal.stage2b_impl import tau_calibration_present
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_DIR = REPO_ROOT / "scratch" / "stage3-gaussian-audit"
EXAMPLE_2638 = REPO_ROOT / "examples" / "blackchirp_data" / "2638"

# Stage 1 settings shared with the production unapodized fixture used by
# the gaussian-shape research dir; matches what Stage 2b expects.
FT_TRIM_MHZ: tuple[float, float] = (26500.0, 40000.0)

logger = logging.getLogger("stage3-gaussian-audit")


@dataclass(frozen=True)
class Stage3RunResult:
    """Summary of one ``detect_peaks`` invocation under a fixed knob value."""

    shape: str
    knob_label: str
    knob_value: Any
    tau_basis_us: Optional[float]
    n_peaks: int
    n_promoted: int
    n_primary: int
    n_gap: int
    n_weak: int
    n_medium: int
    n_strong: int
    n_unclassified: int
    snr_median: float
    snr_p25: float
    snr_p75: float
    snr_p95: float
    snr_max: float
    # Sidelobe-suspect proxy: peaks whose internal-grid SNR cleared the
    # internal floor by ≥ 2x but whose user-grid SNR is below the
    # promotion cutoff. A high count flags a noisy gap-pass mask.
    n_internal_high_user_low: int
    runtime_s: float


def summarise(
    result: Mapping[str, Any],
    *,
    shape: str,
    knob_label: str,
    knob_value: Any,
    runtime_s: float,
) -> Stage3RunResult:
    peaks = result["peaks"]
    snrs = np.asarray(
        [float(p.snr) if p.snr is not None else 0.0 for p in peaks],
        dtype=float,
    )
    if snrs.size == 0:
        snrs = np.zeros(1)
    promotion_floor = float(result.get("promotion_min_snr", 0.0))
    n_weak = sum(
        1 for p in peaks
        if p.classification is not None
        and p.classification.value == "weak"
    )
    n_medium = sum(
        1 for p in peaks
        if p.classification is not None
        and p.classification.value == "medium"
    )
    n_strong = sum(
        1 for p in peaks
        if p.classification is not None
        and p.classification.value == "strong"
    )
    n_unclassified = sum(
        1 for p in peaks if p.classification is None
    )
    n_internal_high_user_low = 0
    for p in peaks:
        internal_snr = p.properties.get("internal_snr")
        if internal_snr is None:
            continue
        try:
            i_snr = float(internal_snr)
        except (TypeError, ValueError):
            continue
        if i_snr >= 2.0 * promotion_floor and (p.snr or 0.0) < promotion_floor:
            n_internal_high_user_low += 1

    return Stage3RunResult(
        shape=shape,
        knob_label=knob_label,
        knob_value=knob_value,
        tau_basis_us=float(result["parameters_used"].get("tau_basis_us"))
        if result["parameters_used"].get("tau_basis_us") is not None
        else None,
        n_peaks=int(result["n_peaks"]),
        n_promoted=int(result["n_promoted"]),
        n_primary=int(result["n_primary"]),
        n_gap=int(result["n_gap"]),
        n_weak=n_weak,
        n_medium=n_medium,
        n_strong=n_strong,
        n_unclassified=n_unclassified,
        snr_median=float(np.median(snrs)),
        snr_p25=float(np.percentile(snrs, 25)),
        snr_p75=float(np.percentile(snrs, 75)),
        snr_p95=float(np.percentile(snrs, 95)),
        snr_max=float(snrs.max()),
        n_internal_high_user_low=int(n_internal_high_user_low),
        runtime_s=runtime_s,
    )


def prepare_fixture(shape: str, *, force_rebuild: bool = False) -> Path:
    """Return a Stages 0-2b-prepped fixture for ``shape`` (cached).

    Both Stage 2b twins (Lorentzian and Gaussian) are run on each copy
    so the τ-feeder can read either. The persisted
    ``recommended_shape`` attr is force-stamped to match the requested
    shape regardless of what the 3-way auto-recommend voted.
    """
    if shape not in {"lorentzian", "gaussian"}:
        raise ValueError(f"shape must be 'lorentzian' or 'gaussian', got {shape!r}")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    fp = SCRATCH_DIR / f"exp_2638_unapodized_{shape}.ftmw"
    if fp.exists() and not force_rebuild:
        logger.info("Reusing cached fixture %s", fp)
        # Re-stamp recommended_shape in case a prior probe forced
        # the other value.
        write_stage2b_recommended_shape(str(fp), shape=shape)
        return fp

    logger.info(
        "Preparing %s fixture at %s (this takes ~3 min: import + FT + "
        "noise + 2x Stage 2b)", shape, fp,
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
    logger.info("Fixture %s ready (recommended_shape=%s)", fp.name, shape)
    return fp


def run_detect_peaks(
    fixture: Path,
    *,
    shape: str,
    settings: Any,
    knob_label: str,
    knob_value: Any,
) -> Stage3RunResult:
    """Run ``detect_peaks`` on a copy of ``fixture`` with the given settings.

    Each call copies the fixture into a per-call scratch file so the
    persisted ``/stage3_peaks`` / ``processing_parameters/stage3_peaks``
    from one grid point doesn't leak into the next.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    safe_value = str(knob_value).replace(".", "p").replace("/", "_")
    work_fp = SCRATCH_DIR / "runs" / f"{shape}_{knob_label}_{safe_value}.ftmw"
    work_fp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, work_fp)
    write_stage2b_recommended_shape(str(work_fp), shape=shape)

    t0 = time.perf_counter()
    detect_kwargs: dict[str, Any] = {}
    if settings is not None:
        detect_kwargs["settings"] = settings
    # Call the internal impl directly so we get the rich result dict
    # (the public API returns a Pipeline.PeakDetectionResult that hides
    # the per-pass counts we want for the audit).
    from ftmwpipeline._internal.stage3_impl import detect_peaks_impl
    result = detect_peaks_impl(str(work_fp), **detect_kwargs)
    runtime = time.perf_counter() - t0

    summary = summarise(
        result,
        shape=shape,
        knob_label=knob_label,
        knob_value=knob_value,
        runtime_s=runtime,
    )
    return summary


def sweep(
    fixture: Path,
    *,
    shape: str,
    knob_label: str,
    knob_values: Iterable[Any],
    build_settings: Callable[[Any], Any],
) -> List[Stage3RunResult]:
    """Run ``detect_peaks`` once per knob value; return summaries."""
    out: List[Stage3RunResult] = []
    for v in knob_values:
        s = build_settings(v)
        logger.info(
            "  sweep[%s=%s, shape=%s] ...", knob_label, v, shape,
        )
        out.append(
            run_detect_peaks(
                fixture, shape=shape, settings=s,
                knob_label=knob_label, knob_value=v,
            )
        )
    return out


_CSV_FIELDS: list[str] = [
    "shape", "knob_label", "knob_value", "tau_basis_us",
    "n_peaks", "n_promoted", "n_primary", "n_gap",
    "n_weak", "n_medium", "n_strong", "n_unclassified",
    "snr_median", "snr_p25", "snr_p75", "snr_p95", "snr_max",
    "n_internal_high_user_low", "runtime_s",
]


def write_csv(rows: Iterable[Stage3RunResult], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in rows:
            w.writerow([getattr(r, field) for field in _CSV_FIELDS])
    logger.info("Wrote %s", csv_path)
    return csv_path


def plot_knob_sweep(
    rows: List[Stage3RunResult],
    *,
    x_label: str,
    out_path: Path,
    title: str,
) -> Path:
    """Standard 4-panel sweep figure: counts / promoted / SNR p95 / sidelobe.

    Expects ``rows`` to contain entries from one or more shapes; lines
    are coloured by shape. ``knob_value`` is the x axis on every panel.
    """
    import matplotlib.pyplot as plt  # local import to avoid harness-level dep

    shapes = sorted({r.shape for r in rows})
    colours = {"lorentzian": "tab:blue", "gaussian": "tab:orange"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax_total, ax_prom), (ax_snr, ax_side) = axes
    for shape in shapes:
        sub = [r for r in rows if r.shape == shape]
        sub.sort(key=lambda r: r.knob_value)
        x = np.asarray([float(r.knob_value) for r in sub])
        ax_total.plot(x, [r.n_peaks for r in sub], "-o",
                      color=colours.get(shape, "k"), label=shape)
        ax_prom.plot(x, [r.n_promoted for r in sub], "-o",
                     color=colours.get(shape, "k"), label=shape)
        ax_snr.plot(x, [r.snr_p95 for r in sub], "-o",
                    color=colours.get(shape, "k"), label=shape)
        ax_side.plot(x, [r.n_internal_high_user_low for r in sub], "-o",
                     color=colours.get(shape, "k"), label=shape)

    for ax, ylab in (
        (ax_total, "n_peaks (total)"),
        (ax_prom, "n_promoted"),
        (ax_snr, "SNR p95"),
        (ax_side, "sidelobe-suspect proxy"),
    ):
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_snr.set_yscale("log")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


__all__ = [
    "FT_TRIM_MHZ",
    "SCRATCH_DIR",
    "Stage3RunResult",
    "prepare_fixture",
    "run_detect_peaks",
    "summarise",
    "sweep",
    "write_csv",
    "plot_knob_sweep",
]
