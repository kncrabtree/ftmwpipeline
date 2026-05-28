"""Shared harness for the Stage 4 Gaussian-path defaults audit.

Mirrors the Stage 3 sibling under
``dev-docs/research/stage3-gaussian-audit/harness.py``. Each per-knob
probe stands up a thin sweep driver against this module so the heavy
steps (fixture prep, Stage 2b L+G calibration, Stage 3 detection,
recommended-shape stamping) happen once per shape.

Fixture layout per working copy:

* Stages 0-2 with ``zpf=2``, ``trim=(26500, 40000)``, ``expf_us=None``.
* Stage 2b Lorentzian + Gaussian twins both present so the shape-aware
  Stage 3 τ-feeder (and the prospective Stage 4 τ-feeder) can pick
  either anchor.
* ``recommended_shape`` stamped on both Stage 2b groups so resolvers
  pick the requested shape.
* Stage 3 peak list persisted using the shape-aware τ-feeder.

The two cached shape fixtures live at
``scratch/stage4-gaussian-audit/exp_2638_unapodized_{lorentzian,gaussian}.ftmw``.
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage2b_g_impl import (
    load_tau_G_calibration_impl,
    tau_G_calibration_present,
)
from ftmwpipeline._internal.stage2b_impl import (
    load_tau_calibration_impl,
    tau_calibration_present,
)
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_DIR = REPO_ROOT / "scratch" / "stage4-gaussian-audit"
EXAMPLE_2638 = REPO_ROOT / "examples" / "blackchirp_data" / "2638"

FT_ZPF = 2
FT_TRIM_MHZ: tuple[float, float] = (26500.0, 40000.0)

logger = logging.getLogger("stage4-gaussian-audit")


@dataclass(frozen=True)
class Stage4RunResult:
    """Summary of one ``assign_windows`` invocation under a fixed knob value."""

    shape: str
    knob_label: str
    knob_value: Any
    leakage_tau_us: Optional[float]
    n_windows: int
    n_hard: int
    n_easy: int
    n_batches: int
    n_free_peaks: int
    n_fixed_contributors: int
    n_dependencies: int
    width_median_mhz: float
    width_p25_mhz: float
    width_p75_mhz: float
    width_p95_mhz: float
    width_max_mhz: float
    n_needs_joint: int
    n_split_proposed: int
    runtime_s: float


def summarise(
    result: Mapping[str, Any],
    *,
    shape: str,
    knob_label: str,
    knob_value: Any,
    runtime_s: float,
) -> Stage4RunResult:
    plan = result["plan"]
    widths = np.asarray([w.width_mhz for w in plan.windows], dtype=float)
    if widths.size == 0:
        widths = np.zeros(1)
    n_needs_joint = sum(1 for w in plan.windows if w.needs_joint_treatment)
    n_split_proposed = sum(
        1 for w in plan.windows if w.split_proposal is not None
    )
    params = result.get("parameters_used") or {}
    tau_v = params.get("tau_us")
    return Stage4RunResult(
        shape=shape,
        knob_label=knob_label,
        knob_value=knob_value,
        leakage_tau_us=float(tau_v) if tau_v is not None else None,
        n_windows=int(result["n_windows"]),
        n_hard=int(result["n_hard"]),
        n_easy=int(result["n_easy"]),
        n_batches=int(result["n_batches"]),
        n_free_peaks=int(result["n_free_peaks"]),
        n_fixed_contributors=int(result["n_fixed_contributors"]),
        n_dependencies=int(result["n_dependencies"]),
        width_median_mhz=float(np.median(widths)),
        width_p25_mhz=float(np.percentile(widths, 25)),
        width_p75_mhz=float(np.percentile(widths, 75)),
        width_p95_mhz=float(np.percentile(widths, 95)),
        width_max_mhz=float(widths.max()),
        n_needs_joint=int(n_needs_joint),
        n_split_proposed=int(n_split_proposed),
        runtime_s=runtime_s,
    )


def prepare_fixture(shape: str, *, force_rebuild: bool = False) -> Path:
    """Return a Stages 0–3 prepped fixture for ``shape`` (cached).

    Stages 2b Lorentzian and Gaussian are both run. The
    ``recommended_shape`` attr is force-stamped to ``shape`` so the
    Stage 3 gap-pass τ-feeder picks the matching anchor; the
    persisted Stage 3 peak list reflects that choice.
    """
    if shape not in {"lorentzian", "gaussian"}:
        raise ValueError(f"shape must be 'lorentzian' or 'gaussian', got {shape!r}")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    fp = SCRATCH_DIR / f"exp_2638_unapodized_{shape}.ftmw"
    if fp.exists() and not force_rebuild:
        logger.info("Reusing cached fixture %s", fp)
        write_stage2b_recommended_shape(str(fp), shape=shape)
        return fp

    logger.info(
        "Preparing %s fixture at %s (import + FT + noise + 2x Stage 2b + Stage 3)",
        shape, fp,
    )
    ftmw.import_data(str(fp), source=str(EXAMPLE_2638), force=True)
    ftmw.compute_ft(str(fp), zpf=FT_ZPF, expf_us=None, trim=FT_TRIM_MHZ)
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
    logger.info("Fixture %s ready (recommended_shape=%s)", fp.name, shape)
    return fp


def get_stage2b_tau(fixture: Path, *, shape: str) -> float:
    """Read the Stage 2b ``τ_maj`` / ``τ_G_maj`` from the fixture.

    ``shape='lorentzian'`` → pure-exp ``τ_maj``;
    ``shape='gaussian'`` → Gaussian-twin ``τ_G_maj``.
    """
    if shape == "lorentzian":
        return float(
            load_tau_calibration_impl(str(fixture))["tau_calibration"].tau_maj_us
        )
    if shape == "gaussian":
        return float(
            load_tau_G_calibration_impl(str(fixture))[
                "tau_G_calibration"
            ].tau_maj_us
        )
    raise ValueError(f"unknown shape {shape!r}")


def run_assign_windows(
    fixture: Path,
    *,
    shape: str,
    settings: Any,
    knob_label: str,
    knob_value: Any,
) -> Stage4RunResult:
    """Run ``assign_windows`` on a copy of ``fixture`` with given settings."""
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
    from ftmwpipeline._internal.stage4_impl import assign_windows_impl
    result = assign_windows_impl(str(work_fp), **detect_kwargs)
    runtime = time.perf_counter() - t0
    return summarise(
        result, shape=shape, knob_label=knob_label,
        knob_value=knob_value, runtime_s=runtime,
    )


_CSV_FIELDS: list[str] = [
    "shape", "knob_label", "knob_value", "leakage_tau_us",
    "n_windows", "n_hard", "n_easy", "n_batches",
    "n_free_peaks", "n_fixed_contributors", "n_dependencies",
    "width_median_mhz", "width_p25_mhz", "width_p75_mhz",
    "width_p95_mhz", "width_max_mhz",
    "n_needs_joint", "n_split_proposed", "runtime_s",
]


def write_csv(rows: Iterable[Stage4RunResult], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in rows:
            w.writerow([getattr(r, field) for field in _CSV_FIELDS])
    logger.info("Wrote %s", csv_path)
    return csv_path


def plot_knob_sweep(
    rows: List[Stage4RunResult],
    *,
    x_label: str,
    out_path: Path,
    title: str,
) -> Path:
    """Standard 4-panel sweep figure for Stage 4: n_windows, n_hard,
    width p95, n_fixed_contributors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = sorted({r.shape for r in rows})
    colours = {"lorentzian": "tab:blue", "gaussian": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax_n, ax_h), (ax_w, ax_c) = axes
    for shape in shapes:
        sub = [r for r in rows if r.shape == shape]
        sub.sort(key=lambda r: r.knob_value if r.knob_value is not None else -1)
        x = np.asarray(
            [float(r.knob_value) if r.knob_value is not None else -1.0
             for r in sub]
        )
        ax_n.plot(x, [r.n_windows for r in sub], "-o",
                  color=colours.get(shape, "k"), label=shape)
        ax_h.plot(x, [r.n_hard for r in sub], "-o",
                  color=colours.get(shape, "k"), label=shape)
        ax_w.plot(x, [r.width_p95_mhz for r in sub], "-o",
                  color=colours.get(shape, "k"), label=shape)
        ax_c.plot(x, [r.n_fixed_contributors for r in sub], "-o",
                  color=colours.get(shape, "k"), label=shape)

    for ax, ylab in (
        (ax_n, "n_windows"),
        (ax_h, "n_hard"),
        (ax_w, "width p95 (MHz)"),
        (ax_c, "n_fixed_contributors"),
    ):
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


__all__ = [
    "FT_ZPF",
    "FT_TRIM_MHZ",
    "SCRATCH_DIR",
    "Stage4RunResult",
    "get_stage2b_tau",
    "plot_knob_sweep",
    "prepare_fixture",
    "run_assign_windows",
    "summarise",
    "write_csv",
]
