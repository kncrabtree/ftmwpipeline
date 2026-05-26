"""Generate Stage 5 validation artifacts for the 2638 fixture.

Layout: this script lives in the tracked ``scripts/development/`` tree
but reads its fixture from and writes its outputs to the untracked
``scratch/stage5-validation/`` directory at the repo root. The fixture
``exp_2638.ftmw`` must exist there beforehand (build it by running the
pipeline through ``fit_peaks`` on the BlackChirp 2638 source data
checked in at ``examples/blackchirp_data/2638/``).

Run from the repository root:

    conda run -n ftmwpipeline-dev python scripts/development/stage5-validation/generate_validation.py

Produces, under ``scratch/stage5-validation/``:

* ``overview.png`` -- full-spectrum data+model overlay + magnitude residual.
* ``INDEX.md``     -- catalog of every sampled window with one-line context.
* ``window_NNN/detail.png``         -- consolidated final fit (Figure 1).
* ``window_NNN/audit-trail.png``    -- rescue audit trail (Figure 2).
* ``window_NNN/detail-rr<n>.png``   -- one per consolidated B-loop round.
* ``window_NNN/report.md``          -- per-window initial-fit rollup.
* ``window_NNN/report-rr.md``       -- per-window rescue chain rollup.

Re-runnable. Reads the persisted Stage 5 fit from the .ftmw -- does not re-fit
the initial pass; does run the rescue chain afresh per window so the audit
diagnostics are live (``RescueRoundDiagnostics`` is not yet persisted).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive for batch runs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage1_impl import _read_settings_layer
from ftmwpipeline._internal.stage3_impl import (
    _active_acquisition_us,
    _load_canonical_noise,
)
from ftmwpipeline.core.settings import FTSettings
from ftmwpipeline._internal.stage5_impl import (
    _build_active_ft_inputs,
    _resolve_sideband,
)
from ftmwpipeline._internal.stage1_impl import compute_ft_impl
from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
from ftmwpipeline.core.data_structures import (
    AuditStep,
    FitWindow,
    FittedPeak,
    FittingResult,
    SpectrumFit,
    WindowDifficulty,
    WindowPlan,
)
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from ftmwpipeline.fitting.plan_execution import (
    FrozenPeak,
    _peaks_to_candidate_offsets,
    fit_window_with_fixed_contributors,
    materialize_window,
)
from ftmwpipeline.fitting.residual_rescue import (
    DEFAULT_RESCUE_MAX_ROUNDS,
    DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    DEFAULT_RESCUE_SNR_THRESHOLD,
    ConsolidatedRescueOutcome,
    RescueRoundDiagnostics,
    rescue_and_consolidate,
)
from ftmwpipeline.fitting.residual_screening import (
    ResidualPeakCandidate,
    find_residual_peaks,
)
from ftmwpipeline.fitting.window_fit import (
    ConservativeFitResult,
    WindowFitResult,
)
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive
from ftmwpipeline.visualization.fit_visualization import plot_spectrum_fit


SCRIPT_DIR = Path(__file__).resolve().parent
# Script lives at <repo>/scripts/development/stage5-validation/; outputs go to
# the gitignored <repo>/scratch/stage5-validation/ (or any other subdir picked
# via --output-dir) alongside the .ftmw fixture. The actual paths are resolved
# inside ``main`` after CLI parsing — these defaults exist for any helpers /
# tests that import the module without driving the CLI.
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "scratch" / "stage5-validation"
FTMW_PATH = OUTPUT_DIR / "exp_2638.ftmw"

# Model curves in the detail and audit-trail figures are plotted on a grid this
# many times finer than the data grid so the lineshape reads as a smooth curve
# instead of a polyline. Residuals, histograms, and χ² stats always use the
# data grid (one model evaluation per data bin) so they remain fit-faithful.
MODEL_OVERSAMPLE = 8


# ---------------------------------------------------------------------------
# Display-style helpers
# ---------------------------------------------------------------------------
# The fitting algorithms run in active-FT native amplitude units
# (``dt_us * rfft(active)``) which are very small (~1e-6). The persisted
# FTSettings record carries the user's preferred display convention -- a
# ``units_power`` that scales the magnitude (10**units_power, e.g. 6 -> µV)
# and a ``trim`` range that restricts the active region. Neither affects
# any algorithm; both are display-only transforms applied to the harness's
# figures.

UNITS_LABEL_BY_POWER = {
    0: "V",
    3: "mV",
    6: "µV",
    9: "nV",
    12: "pV",
}


@dataclass(frozen=True)
class DisplayStyle:
    """Display-only transforms read from the persisted FTSettings."""

    amplitude_scale: float
    """Multiplicative factor applied to plotted amplitudes (10**units_power)."""

    units_label: str
    """Y-axis amplitude unit (e.g. 'µV')."""

    trim_mhz: Optional[Tuple[float, float]]
    """``(lo, hi)`` MHz trim for the overview spectrum; ``None`` = full range."""


def _load_display_style(ftmw_path: Path) -> DisplayStyle:
    """Read the persisted FTSettings (canonical /processing_parameters/ft_processing)
    and return the display transforms it implies. Falls back to (scale=1,
    units='', trim=None) if nothing is persisted (older fixtures)."""
    settings = _read_settings_layer(
        str(ftmw_path), "/processing_parameters/ft_processing"
    )
    if settings is None:
        return DisplayStyle(amplitude_scale=1.0, units_label="", trim_mhz=None)
    units_power = settings.units_power
    if units_power is None:
        scale = 1.0
        label = ""
    else:
        scale = 10.0 ** int(units_power)
        label = UNITS_LABEL_BY_POWER.get(int(units_power), f"·10^{units_power} V")
    trim = settings.trim
    return DisplayStyle(amplitude_scale=scale, units_label=label, trim_mhz=trim)

# Deliberate sample (matches the choices reported in the conversation).
EASY_SAMPLE: Tuple[int, ...] = (215, 16, 104, 337, 64, 63)
HARD_SAMPLE: Tuple[int, ...] = (127, 260, 148, 68, 132, 198)
NAMED_SAMPLE: Tuple[Tuple[int, str], ...] = (
    (209, "34154 MHz anomaly (anomalously low de-ramped S_coh)"),
    (269, "36350 MHz half of the canonical decoupled doublet"),
    (271, "36389 MHz half of the canonical decoupled doublet"),
)


def _fmt_opt(v: Optional[float], digits: int = 3) -> str:
    return f"{v:.{digits}g}" if v is not None else "-"


def _fmt_pm(value: Optional[float], err: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    err_part = f"+/-{err:.{digits}g}" if err is not None else "+/-?"
    return f"{value:.{digits}g} {err_part}"


def _format_spectroscopic(
    value: Optional[float], err: Optional[float], *, n_digits: int = 2,
) -> str:
    """PDG-style ``value(err_digits)`` formatter.

    Uses 2 digits of error by default. Bumps to 3 digits when the 2-digit
    error rounds into the 10-19 range (the leading-1 case where the
    per-digit precision is worst: a "13" carries ~7% rounding error,
    "134" carries ~0.7%). Value precision is matched to the error's last
    displayed digit.

    Examples
    --------
    >>> _format_spectroscopic(3.06546, 0.0134)
    '3.0655(134)'
    >>> _format_spectroscopic(3.06546, 0.0234)
    '3.065(23)'
    >>> _format_spectroscopic(3.06546, 0.0094)
    '3.066(94)'
    """
    import math
    if value is None or not math.isfinite(value):
        return "-"
    if err is None or not math.isfinite(err) or err <= 0:
        return f"{value:.6g}"
    k = math.floor(math.log10(err))
    last_pos = k - n_digits + 1
    scale = 10.0 ** last_pos
    err_int = int(round(err / scale))
    # Round-up to next order (e.g., 0.999 -> 100 with 2 digits): pull back one digit.
    if err_int >= 10 ** n_digits:
        last_pos += 1
        scale = 10.0 ** last_pos
        err_int = int(round(err / scale))
    # PDG leading-1: bump 2-digit "1X" forms to 3-digit "1XY" forms.
    if 10 <= err_int < 20:
        last_pos -= 1
        scale = 10.0 ** last_pos
        err_int = int(round(err / scale))
    value_round = round(value / scale) * scale
    decimals = max(0, -last_pos)
    return f"{value_round:.{decimals}f}({err_int})"


def _format_spectroscopic_sci(
    value: Optional[float], err: Optional[float], *, n_digits: int = 2,
) -> str:
    """Spectroscopic formatter with auto scientific notation.

    Falls back to :func:`_format_spectroscopic` (plain decimal form) when
    ``|value|`` is in ``[1e-3, 1e6)``; otherwise factors out the value's
    order of magnitude and formats the mantissa spectroscopically:

    >>> _format_spectroscopic_sci(4.18e-6, 3.1e-7)
    '4.18(31)e-06'
    """
    import math
    if value is None or not math.isfinite(value):
        return "-"
    if value == 0.0:
        return _format_spectroscopic(value, err, n_digits=n_digits)
    if 1e-3 <= abs(value) < 1e6:
        return _format_spectroscopic(value, err, n_digits=n_digits)
    exp = int(math.floor(math.log10(abs(value))))
    scale = 10.0 ** exp
    m_value = value / scale
    m_err: Optional[float]
    if err is not None and math.isfinite(err) and err > 0:
        m_err = err / scale
    else:
        m_err = None
    body = _format_spectroscopic(m_value, m_err, n_digits=n_digits)
    return f"{body}e{exp:+03d}"


def _peak_table(peaks: Sequence[FittedPeak]) -> str:
    lines = [
        "| peak_id | freq (MHz) | amplitude | decay_rate (1/us) | phase (rad) | SNR | knockout |",
        "|---:|---|---|---|---|---:|:---:|",
    ]
    for p in peaks:
        ko = "-"
        if p.knockout is not None:
            ok = "ok" if p.knockout.supported else "WEAK"
            ko = f"{ok} ({p.knockout.delta_chi2:.2g}/{p.knockout.expected_delta_chi2:.2g})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(p.peak_id),
                    _fmt_pm(p.frequency_mhz, p.frequency_error),
                    _fmt_pm(p.amplitude, p.amplitude_error, digits=4),
                    _fmt_pm(p.decay_rate, p.decay_rate_error, digits=4),
                    _fmt_pm(p.phase, p.phase_error, digits=4),
                    _fmt_opt(p.snr, digits=3),
                    ko,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _audit_table(wf: FittingResult) -> str:
    if not wf.audit_trail:
        return "_no audit trail_"
    lines = [
        "| # | decision | off (MHz) | chi2 before -> after | F | p | AIC before -> after | sep_ok | reason |",
        "|---:|---|---:|---|---:|---|---|:---:|---|",
    ]
    for i, step in enumerate(wf.audit_trail):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    step.decision,
                    f"{step.candidate_offset_mhz:+.4f}",
                    f"{step.chi2_before:.3g} -> {step.chi2_after:.3g}",
                    f"{step.f_statistic:.3g}",
                    f"{step.p_value:.2e}",
                    f"{step.aic_before:.3g} -> {step.aic_after:.3g}",
                    "yes" if step.separation_ok else "no",
                    (step.reason or "").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _thaw_table(wf: FittingResult) -> str:
    if not wf.thaw_events:
        return "_no thaw events on this window_"
    lines = [
        "| dep | primary | contrib idx | contrib f (MHz) | side | S_coh before -> after | accepted | reason |",
        "|---:|---:|---:|---|---|---|:---:|---|",
    ]
    for e in wf.thaw_events:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(e.dependent_window_id),
                    str(e.primary_window_id),
                    str(e.contributor_peak_index),
                    f"{e.contributor_frequency_mhz:.4f}",
                    e.edge_side,
                    f"{e.edge_coherence_before:.3g} -> {e.edge_coherence_after:.3g}",
                    "yes" if e.accepted else "no",
                    (e.reason or "").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _difficulty_block(window: FitWindow) -> str:
    rows = [
        f"- difficulty: **{window.difficulty.value}**",
        f"- batch: {window.batch}",
        f"- width: {window.width_mhz:.3f} MHz",
        f"- needs_joint_treatment: {window.needs_joint_treatment}",
        f"- split_proposal: {window.split_proposal}",
    ]
    if window.diagnostics:
        rows.append("- diagnostics:")
        for k, v in sorted(window.diagnostics.items()):
            rows.append(f"    - `{k}`: {v}")
    return "\n".join(rows)


def _shared_block(wf: FittingResult) -> str:
    rows = []
    for name, info in sorted(wf.shared_parameters.items()):
        val = info.get("value")
        err = info.get("stderr") or info.get("error")
        units = info.get("units", "")
        rows.append(f"- `{name}`: {_fmt_pm(val, err, digits=4)} {units}".rstrip())
    if not rows:
        rows.append("- _none_")
    return "\n".join(rows)


def _fixed_block(window: FitWindow, peaks: Sequence) -> str:
    if not window.fixed_contributors:
        return "_no fixed contributors_"
    by_idx = {i: p for i, p in enumerate(peaks)}
    lines = [
        "| peak_idx | freq (MHz) | primary window | freeze_eligible |",
        "|---:|---|---:|:---:|",
    ]
    for fc in window.fixed_contributors:
        f_mhz = fc.frequency_mhz
        p = by_idx.get(fc.peak_index)
        if p is not None and not f_mhz:
            f_mhz = float(p.frequency)
        lines.append(
            f"| {fc.peak_index} | {f_mhz:.4f} | {fc.primary_window_id} | "
            f"{'yes' if fc.freeze_eligible else 'NO'} |"
        )
    return "\n".join(lines)


def _compute_window_residual(
    wf: FittingResult,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    sideband,
    acquisition_us: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Recompute the per-window slice + model + residual the plot uses.

    Mirrors ``fit_visualization._plot_per_window_detail`` so the detector and
    the visualization see the same data. Returns
    ``(f_slice, residual, sigma_slice, model_slice, fwhm_mhz)`` where
    ``fwhm_mhz = 1 / (pi * tau_us)`` is the expected Lorentzian width.
    """
    if wf.window is None:
        raise ValueError(f"window fit {wf.window_id} has no SpectralWindow")
    s = sideband_sign(sideband)
    lo, hi = wf.window.freq_range
    mask = (frequencies >= min(lo, hi)) & (frequencies <= max(lo, hi))
    f_slice = frequencies[mask]
    z_slice = complex_spectrum[mask]
    sigma_slice = rms_noise[mask]
    center = 0.5 * (lo + hi)
    u_slice = s * (f_slice - center)

    tau_us = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    peaks = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in wf.fitted_peaks
    ]
    frozen_peaks: List[ModelPeak] = []
    for key, fp_data in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        frozen_peaks.append(
            ModelPeak(
                amplitude=float(fp_data["amplitude"]),
                offset_mhz=float(
                    s * (float(fp_data["frequency_mhz"]) - center)
                ),
                phase=float(fp_data.get("phase", 0.0) or 0.0),
            )
        )
    all_peaks = peaks + frozen_peaks
    shape_str = getattr(wf, "shape", "lorentzian")
    if all_peaks and tau_us > 0:
        model_slice = model_spectrum(
            u_slice, all_peaks, tau_us, acquisition_us, shape=shape_str,
        )
    else:
        model_slice = np.zeros_like(z_slice)
    residual = z_slice - model_slice
    fwhm_mhz = 1.0 / (np.pi * tau_us) if tau_us > 0 else 0.0
    return f_slice, residual, sigma_slice, model_slice, float(fwhm_mhz)


def _residual_candidate_block(
    candidates: Sequence[ResidualPeakCandidate],
) -> str:
    if not candidates:
        return "_no |residual| candidates above 4σ_c with prominence ≥ 2σ_c_"
    lines = [
        "| # | freq (MHz) | |residual| | SNR | prom (σ_c) | nearest existing | sep (MHz) | flag |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for i, c in enumerate(candidates):
        near_freq = (
            f"{c.nearest_existing_freq_mhz:.4f}"
            if c.nearest_existing_freq_mhz is not None
            else "-"
        )
        near_sep = (
            f"{c.nearest_existing_separation_mhz:.4f}"
            if c.nearest_existing_separation_mhz is not None
            else "-"
        )
        if c.near_existing:
            flag = (
                f"near peak {c.nearest_existing_peak_id}"
                if c.nearest_existing_peak_id is not None
                else "near existing"
            )
        else:
            flag = "new candidate"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"C{i}",
                    f"{c.frequency_mhz:.4f}",
                    f"{c.magnitude:.4g}",
                    f"{c.snr:.2f}",
                    f"{c.prominence_sigma_c:.2f}",
                    near_freq,
                    near_sep,
                    flag,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _overlay_residual_candidates(
    fig,
    candidates: Sequence[ResidualPeakCandidate],
) -> None:
    """Mark candidates on the |residual| panel of the per-window detail fig.

    The 4x2 grid in ``_plot_per_window_detail`` creates ``ax_mag_res`` as
    the 6th ``add_subplot`` call (index 5 in ``fig.axes``). The lookup is
    indexed; if the grid changes, update this offset.
    """
    if not candidates:
        return
    if len(fig.axes) <= 5:
        return
    ax_mag = fig.axes[5]
    for i, c in enumerate(candidates):
        color = "tab:gray" if c.near_existing else "tab:orange"
        ax_mag.axvline(
            c.frequency_mhz, color=color, lw=0.8, ls=":", alpha=0.7, zorder=1,
        )
        ax_mag.plot(
            c.frequency_mhz,
            c.magnitude,
            marker="v",
            markersize=7,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=4,
        )
        ax_mag.annotate(
            f"C{i}: {c.snr:.1f}σ",
            xy=(c.frequency_mhz, c.magnitude),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            color=color,
        )


def _run_window_rescue(
    window: FitWindow,
    wf: FittingResult,
    *,
    active_ft,
    active_noise_arr: np.ndarray,
    sideband,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    peak_frequencies_mhz: Sequence[float],
    init_conservative_kwargs: dict,
    rescue_conservative_kwargs: dict,
    rescue_kwargs: dict,
    max_rescue_rounds: int = DEFAULT_RESCUE_MAX_ROUNDS,
) -> Tuple[ConsolidatedRescueOutcome, ConservativeFitResult, float]:
    """Reproduce the initial fit for this window and drive the full B-loop.

    Returns ``(consolidated, initial_fit, center_mhz)``. ``consolidated``
    is the :class:`ConsolidatedRescueOutcome` from
    :func:`rescue_and_consolidate` -- it carries the per-round chain
    (rescue + joint refit + knockout sweep) and the final consolidated
    :class:`ConservativeFitResult`. ``initial_fit`` is the unmodified
    pre-rescue fit (carried back so the report can compare against it).

    Re-running the initial fit is intentional: the persisted fit is a
    :class:`FittingResult` (no offset-frame ``ModelPeak`` state), so the
    rescue cannot pick it up directly. The repeat call reproduces the
    persisted state exactly (no stochastic seeding) and adds <1s per
    window for the 2638 fixture.
    """
    _, offset_grid, z_offset, sig_slice, center_mhz = materialize_window(
        window, active_ft, active_noise_arr, sideband=sideband,
    )
    s = sideband_sign(sideband)
    frozen_peaks: List[FrozenPeak] = []
    for key, fp_data in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        contrib_freq = float(fp_data["frequency_mhz"])
        contrib_amp = float(fp_data["amplitude"])
        contrib_phase = float(fp_data.get("phase", 0.0) or 0.0)
        frozen_peaks.append(
            FrozenPeak(
                peak_index=int(fp_data.get("peak_index", -1)),
                primary_window_id=int(fp_data.get("primary_window_id", -1)),
                model_peak=ModelPeak(
                    amplitude=contrib_amp,
                    offset_mhz=float(s * (contrib_freq - center_mhz)),
                    phase=contrib_phase,
                ),
                frequency_mhz=contrib_freq,
                freeze_eligible=True,
            )
        )
    candidate_offsets = _peaks_to_candidate_offsets(
        window, peak_frequencies_mhz, center_mhz, sideband,
    )
    initial_fit, background, _full_fitted, _full_residual = (
        fit_window_with_fixed_contributors(
            offset_grid, z_offset, sig_slice, frozen_peaks, candidate_offsets,
            tau0_us, acquisition_us,
            fit_tau=fit_tau, **init_conservative_kwargs,
        )
    )
    consolidated = rescue_and_consolidate(
        offset_grid,
        z_offset - background,
        sig_slice,
        initial_fit,
        tau0_us,
        acquisition_us,
        max_rescue_rounds=max_rescue_rounds,
        conservative_kwargs=rescue_conservative_kwargs,
        **rescue_kwargs,
    )
    return consolidated, initial_fit, center_mhz


def _build_consolidated_fittingresult(
    window: FitWindow,
    original_wf: FittingResult,
    joint_fit: WindowFitResult,
    sideband,
    center_mhz: float,
    label: str,
) -> FittingResult:
    """Synthetic :class:`FittingResult` whose ``fitted_peaks`` are the
    *consolidated* peaks at a given B-loop round.

    Used by the per-round ``detail-rr<n>.png`` artifact: at the end of
    round ``n``, the consolidated state is ``initial_peaks + rescue
    contributions from rounds 0..n``, jointly refit with knockout
    pruning. The frozen-contributor background and the original
    quality metrics are carried over from ``original_wf`` so the
    overlay reads at the same full-data scale as ``detail.png``.
    """
    s = sideband_sign(sideband)
    tau_us = float(joint_fit.tau_us)
    decay_rate = (1.0 / tau_us) if tau_us > 0.0 else None
    # Propagate tau_us uncertainty to decay_rate uncertainty via d(1/tau)/dtau.
    decay_rate_err: Optional[float] = None
    if joint_fit.tau_error is not None and tau_us > 0.0:
        decay_rate_err = float(joint_fit.tau_error) / (tau_us * tau_us)
    fitted_peaks: List[FittedPeak] = []
    for i, pk in enumerate(joint_fit.peaks):
        mol_freq = center_mhz + s * pk.offset_mhz
        err = (
            joint_fit.peak_errors[i]
            if i < len(joint_fit.peak_errors)
            else None
        )
        freq_err = (
            float(err.offset_mhz)
            if err is not None and np.isfinite(err.offset_mhz)
            else None
        )
        amp_err = (
            float(err.amplitude)
            if err is not None and np.isfinite(err.amplitude)
            else None
        )
        phase_err = (
            float(err.phase)
            if err is not None and np.isfinite(err.phase)
            else None
        )
        fitted_peaks.append(
            FittedPeak(
                peak_id=f"{label}_{window.window_id}_{i}",
                frequency_mhz=float(mol_freq),
                amplitude=float(pk.amplitude),
                decay_rate=decay_rate,
                phase=float(pk.phase),
                window_id=window.window_id,
                frequency_error=freq_err,
                amplitude_error=amp_err,
                decay_rate_error=decay_rate_err,
                phase_error=phase_err,
            )
        )
    n_residual = max(joint_fit.n_data - joint_fit.n_params, 1)
    new_wf = FittingResult(
        success=joint_fit.success,
        cost=float(joint_fit.chi_squared),
        iterations=0,
        aic=float(joint_fit.aic),
        reduced_chi2=float(joint_fit.chi_squared / n_residual),
        window=original_wf.window,
        window_id=original_wf.window_id,
        shape=getattr(original_wf, "shape", "lorentzian"),
    )
    new_wf.fitted_peaks = fitted_peaks
    new_wf.shared_parameters["tau_us"] = {
        "value": tau_us,
        "error": (
            float(joint_fit.tau_error)
            if joint_fit.tau_error is not None
            and np.isfinite(joint_fit.tau_error)
            else None
        ),
        "peak_ids": [p.peak_id for p in fitted_peaks],
    }
    # Carry the frozen background through -- the rescue does NOT touch it
    # (the rescue operates on data_minus_bg), and the full-data overlay
    # needs the frozen contributors to read at the right scale.
    new_wf.fixed_parameters = dict(original_wf.fixed_parameters)
    new_wf.quality_metrics = dict(original_wf.quality_metrics)
    return new_wf


def _peak_provenance(
    initial_fit: ConservativeFitResult,
    consolidated: ConsolidatedRescueOutcome,
    consolidated_wf: FittingResult,
    sideband,
    center_mhz: float,
) -> List[Tuple[Optional[str], Optional[float], Optional[str]]]:
    """For each peak in ``consolidated_wf.fitted_peaks``, return its
    ``(origin_label, p_value, decision_tag)`` for display.

    - ``origin_label`` is ``"init"`` or ``"r{N}"`` -- the round whose
      ``rescue.fit.peaks`` contained the nearest match in offset to this
      consolidated peak. ``None`` when no source's added-peak list lands
      within 1.5 FWHM (peak arose purely from joint-refit reshuffling).
    - ``p_value`` is the **knockout test** F-test p of this peak against the
      final consolidated fit -- i.e. the K-peak fit vs (K-1)-peak fit
      produced by knocking the peak out. This is the strongest per-peak
      significance the pipeline produces (it tests the peak against every
      other peak fully fitted to convergence, not just the partial model
      at the time the peak was added). ``0.0`` indicates the F-test p
      underflowed (very significant). ``None`` indicates the peak has no
      knockout entry (shouldn't happen for a consolidated fit, but a guard).
    - ``decision_tag`` is set to ``"accept"`` so the display formatter
      treats ``p == 0`` as ``<1e-15`` (extreme significance) rather than
      a placeholder ``(seed)``. The decision categories from the audit
      trail are no longer the source of the p-value here -- the knockout
      is a fresher, more authoritative test.
    """
    s = sideband_sign(sideband)
    tau_us = float(consolidated.fit.fit.tau_us)
    fwhm = 1.0 / (np.pi * tau_us) if tau_us > 0.0 else 0.1
    tol = 1.5 * fwhm

    sources: List[Tuple[str, List[ModelPeak], List]] = [
        ("init", list(initial_fit.fit.peaks), list(initial_fit.audit_trail)),
    ]
    for i, diag in enumerate(consolidated.rounds):
        if diag.accepted and diag.rescue.fit.peaks:
            sources.append(
                (f"r{i}", list(diag.rescue.fit.peaks), list(diag.rescue.audit))
            )

    # Knockout p-values for each consolidated peak. consolidated.fit
    # (the post-rescue ConservativeFitResult) carries the knockout list run
    # on the final peak set; KnockoutResult.peak_index aligns with
    # ``consolidated.fit.fit.peaks`` order, which is the same order our
    # ``consolidated_wf.fitted_peaks`` was built in. Map by peak_index.
    knockout_by_index = {
        ko.peak_index: ko for ko in consolidated.fit.knockouts
    }

    out: List[Tuple[Optional[str], Optional[float], Optional[str]]] = []
    for fp_idx, fp in enumerate(consolidated_wf.fitted_peaks):
        peak_off = s * (float(fp.frequency_mhz) - center_mhz)
        # 1. Source attribution: closest match in any source's added-peaks list.
        best_label: Optional[str] = None
        best_d = float("inf")
        for label, src_peaks, _src_audit in sources:
            for src_pk in src_peaks:
                d = abs(src_pk.offset_mhz - peak_off)
                if d < best_d:
                    best_d = d
                    best_label = label
        if best_label is None or best_d > tol:
            best_label = None
        # 2. Knockout p-value for this peak (the authoritative per-peak stat).
        ko = knockout_by_index.get(fp_idx)
        if ko is None or not np.isfinite(ko.p_value):
            ko_p: Optional[float] = None
        else:
            ko_p = float(ko.p_value)
        # 3. ``decision`` is forced to "accept" so the formatter treats
        # p == 0 as "<1e-15" (underflow = extreme significance) rather than
        # the seed-placeholder "(seed:...)" path. The knockout is always a
        # real F-test; it never has the seed's null-vs-K=1 placeholder.
        out.append((best_label, ko_p, "accept" if ko_p is not None else None))
    return out


def _format_p_origin(
    label: Optional[str], p_value: Optional[float], decision: Optional[str],
) -> str:
    """Render the per-peak 'p (origin)' column for the peak listing axes.

    Encodes both the F-test p-value (when one was recorded) and the source
    label (init / r0 / r1 / ...). p=0.0 is overloaded in the audit: the
    seed of a fresh conservative loop records its p as a placeholder ``0``
    when ``null_chi2`` is undefined, while a strongly-significant accept
    can also underflow to ``0`` from ``1 - F_cdf``. Disambiguate by the
    decision string:

    - seed/tentative with p == 0 -> "(seed:label)" (no meaningful F-test).
    - accept/promote with p == 0 -> "<1e-15 (label)" (the F-test p
      underflowed, evidence was extreme).
    - any decision with p > 0 -> "{p:.1e} (label)" (tentative gets a "t"
      suffix marker).
    """
    if label is None:
        return "(joint)"
    if p_value is None:
        return f"(seed:{label})"
    if p_value <= 0.0:
        if decision in ("accept", "promote"):
            return f"<1e-15 ({label})"
        return f"(seed:{label})"
    body = f"{p_value:.1e}"
    if decision == "tentative":
        return f"{body} t ({label})"
    return f"{body} ({label})"


def _peak_labels(n: int) -> List[str]:
    """Generate alpha labels: A..Z, then AA..AZ, BA..BZ, ..."""
    out = []
    for i in range(n):
        if i < 26:
            out.append(chr(ord("A") + i))
        else:
            j = i - 26
            out.append(chr(ord("A") + j // 26) + chr(ord("A") + j % 26))
    return out


def _full_spectrum_model(
    frequencies: np.ndarray,
    consolidated_wf: FittingResult,
    other_window_fits: Sequence[FittingResult],
    sideband,
    acquisition_us: float,
) -> np.ndarray:
    """Sum of all windows' models on the active-FT grid.

    The current window contributes ``consolidated_wf`` (post-rescue final);
    every other window contributes its persisted initial fit. Built on the
    same grid the data lives on (no amplitude rescale, no phase re-roll --
    the harness plots in active-FT native units).
    """
    s = sideband_sign(sideband)
    total = np.zeros(frequencies.shape, dtype=np.complex128)
    f = np.asarray(frequencies, dtype=float)
    all_window_fits: List[FittingResult] = [consolidated_wf, *other_window_fits]
    for wf in all_window_fits:
        if wf.window is None:
            continue
        center = 0.5 * (wf.window.freq_range[0] + wf.window.freq_range[1])
        tau_us = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
        if tau_us <= 0:
            continue
        peaks: List[ModelPeak] = [
            ModelPeak(
                amplitude=float(p.amplitude),
                offset_mhz=float(s * (p.frequency_mhz - center)),
                phase=float(p.phase if p.phase is not None else 0.0),
            )
            for p in wf.fitted_peaks
        ]
        # Frozen contributors leakage (carry through for non-current windows;
        # the current window's consolidated WF already has its frozen
        # contributors stored in fixed_parameters).
        for key, fp_data in wf.fixed_parameters.items():
            if not key.startswith("frozen_peak_"):
                continue
            peaks.append(
                ModelPeak(
                    amplitude=float(fp_data["amplitude"]),
                    offset_mhz=float(
                        s * (float(fp_data["frequency_mhz"]) - center)
                    ),
                    phase=float(fp_data.get("phase", 0.0) or 0.0),
                )
            )
        if not peaks:
            continue
        u = s * (f - center)
        total += model_spectrum(u, peaks, tau_us, acquisition_us)
    return total


def _plot_consolidated_detail(
    window: FitWindow,
    consolidated_wf: FittingResult,
    other_window_fits: Sequence[FittingResult],
    *,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    sideband,
    acquisition_us: float,
    peak_provenance: Sequence[Tuple[Optional[str], Optional[float], Optional[str]]],
    title: str,
    style: DisplayStyle,
) -> plt.Figure:
    """Render Figure 1 (landscape letter): consolidated per-window detail.

    Layout (top-to-bottom):
      - Row 1: full active-FT magnitude overview with current window axvspan.
      - Row 2 (3 cols, sharex): Re/Im/|z| residuals + transparent vlines at
        every fitted peak's molecular frequency.
      - Row 3 (3 cols, sharex): Re/Im/|z| data+model + same vlines.
      - Row 4: |z| residual histogram (left, 1 col) + peak-listing axes
        (right, 2 cols) with PDG-style frequency uncertainties and the
        acceptance-step p-value per fitted peak.
    """
    s = sideband_sign(sideband)
    lo, hi = window.freq_range
    mask = (frequencies >= min(lo, hi)) & (frequencies <= max(lo, hi))
    f_slice = frequencies[mask]
    z_slice = complex_spectrum[mask]
    sigma_slice = rms_noise[mask]
    center = 0.5 * (lo + hi)
    u_slice = s * (f_slice - center)

    tau_us = float(consolidated_wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    peaks_in_window = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in consolidated_wf.fitted_peaks
    ]
    frozen_peaks: List[ModelPeak] = []
    for key, fp_data in consolidated_wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        frozen_peaks.append(
            ModelPeak(
                amplitude=float(fp_data["amplitude"]),
                offset_mhz=float(
                    s * (float(fp_data["frequency_mhz"]) - center)
                ),
                phase=float(fp_data.get("phase", 0.0) or 0.0),
            )
        )
    all_peaks = peaks_in_window + frozen_peaks
    shape_str = getattr(consolidated_wf, "shape", "lorentzian")
    if all_peaks and tau_us > 0.0:
        model_slice = model_spectrum(
            u_slice, all_peaks, tau_us, acquisition_us, shape=shape_str,
        )
    else:
        model_slice = np.zeros_like(z_slice)
    # Fine-grid model for the smooth-curve overlay on row 3; residuals,
    # histograms, and quality stats continue to use ``model_slice`` (data grid).
    if f_slice.size >= 2:
        n_fine = (f_slice.size - 1) * MODEL_OVERSAMPLE + 1
        f_fine = np.linspace(float(f_slice.min()), float(f_slice.max()), n_fine)
        u_fine = s * (f_fine - center)
        if all_peaks and tau_us > 0.0:
            model_fine = model_spectrum(
                u_fine, all_peaks, tau_us, acquisition_us, shape=shape_str,
            )
        else:
            model_fine = np.zeros_like(f_fine, dtype=np.complex128)
    else:
        f_fine = f_slice.copy()
        model_fine = model_slice.copy()
    residual = z_slice - model_slice
    sigma_c_slice = sigma_slice / np.sqrt(2.0)
    band = 3.0 * float(np.median(sigma_c_slice))

    # Display-only amplitude scale (e.g., 10^6 -> µV). Algorithms above all
    # ran in active-FT native units; the figure just rescales for readability.
    amp_scale = float(style.amplitude_scale)
    units_lbl = style.units_label
    amp_unit_suffix = f" ({units_lbl})" if units_lbl else ""

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=11)
    gs = GridSpec(
        nrows=4, ncols=3, figure=fig,
        height_ratios=[1.0, 1.6, 1.6, 1.6],
        hspace=0.45, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_overview = fig.add_subplot(gs[0, :])
    ax_re_res = fig.add_subplot(gs[1, 0])
    ax_im_res = fig.add_subplot(gs[1, 1], sharex=ax_re_res)
    ax_mag_res = fig.add_subplot(gs[1, 2], sharex=ax_re_res)
    ax_re_dat = fig.add_subplot(gs[2, 0], sharex=ax_re_res)
    ax_im_dat = fig.add_subplot(gs[2, 1], sharex=ax_re_res)
    ax_mag_dat = fig.add_subplot(gs[2, 2], sharex=ax_re_res)
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_peaks = fig.add_subplot(gs[3, 1:])
    ax_peaks.set_axis_off()

    # --- Row 1: full-spectrum overview ----------------------------------
    # Show data magnitude only -- the model overlay is omitted here because
    # at the full-spectrum scale (~13.5 GHz width on ~1400 px) every fitted
    # line collapses into a thin spike that overlaps the data spike
    # pixel-for-pixel; the model line adds no information and only obscures
    # the data. The per-window data+model panels carry the model overlay.
    if style.trim_mhz is not None:
        t_lo, t_hi = style.trim_mhz
        ov_mask = (frequencies >= min(t_lo, t_hi)) & (
            frequencies <= max(t_lo, t_hi)
        )
        ov_freqs = frequencies[ov_mask]
        ov_data = complex_spectrum[ov_mask]
    else:
        ov_freqs = frequencies
        ov_data = complex_spectrum
    ax_overview.plot(
        ov_freqs, np.abs(ov_data) * amp_scale,
        color="0.3", lw=0.5, label="data |X|",
    )
    # axvspan is narrow on the full-spectrum scale so it can render
    # invisibly when the window is a few MHz out of a 13 GHz axis.
    # Add a pair of edge vlines as a visibility backstop.
    ax_overview.axvspan(
        min(lo, hi), max(lo, hi),
        color="tab:green", alpha=0.35, zorder=0, label="this window",
    )
    ax_overview.axvline(min(lo, hi), color="tab:green", lw=0.9, alpha=0.75, zorder=1)
    ax_overview.axvline(max(lo, hi), color="tab:green", lw=0.9, alpha=0.75, zorder=1)
    ax_overview.set_xlim(ov_freqs[0], ov_freqs[-1])
    ax_overview.set_ylabel(f"|X(f)|{amp_unit_suffix}", fontsize=9)
    ax_overview.set_title("full-spectrum context", fontsize=9)
    ax_overview.tick_params(axis="both", labelsize=8)
    ax_overview.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # --- Rows 2 & 3: residual + data+model panels with peak vlines ------
    fitted_freqs = [float(p.frequency_mhz) for p in consolidated_wf.fitted_peaks]
    labels = _peak_labels(len(fitted_freqs))

    # Assign each label a row height (in axes fraction above the top of each
    # panel) so blended peaks don't collide. Peaks sorted by frequency; if a
    # peak falls within 1 FWHM of the previous label's freq AND would land in
    # the same row, bump it to the next row. Cycles through 3 rows.
    fwhm_label = 1.0 / (np.pi * tau_us) if tau_us > 0.0 else 0.0
    n_rows = 3
    label_row_offsets = [0] * len(fitted_freqs)
    sort_order = sorted(range(len(fitted_freqs)), key=lambda i: fitted_freqs[i])
    last_freq_in_row = [-float("inf")] * n_rows
    for idx in sort_order:
        f_here = fitted_freqs[idx]
        chosen_row = 0
        if fwhm_label > 0:
            for r in range(n_rows):
                if f_here - last_freq_in_row[r] >= fwhm_label:
                    chosen_row = r
                    break
            else:
                # All rows have a too-close prev label; pick the row whose
                # last label is furthest left (least recent).
                chosen_row = int(
                    min(range(n_rows), key=lambda r: last_freq_in_row[r])
                )
        label_row_offsets[idx] = chosen_row
        last_freq_in_row[chosen_row] = f_here

    # Vertical offset per label row in points above the axes top.
    row_y_offsets_pts = [2.0, 12.0, 22.0]

    def _vlines_at_peaks(ax: plt.Axes, with_labels: bool) -> None:
        for f_pk in fitted_freqs:
            ax.axvline(
                f_pk, color="tab:gray", lw=0.7, ls="-", alpha=0.45, zorder=1,
            )
        if not with_labels:
            return
        for f_pk, lbl, row in zip(fitted_freqs, labels, label_row_offsets):
            y_off = row_y_offsets_pts[row]
            ax.annotate(
                lbl,
                xy=(f_pk, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, y_off), textcoords="offset points",
                fontsize=7, ha="center", va="bottom",
                color="0.25",
            )

    # Residuals (row 2). band lines included. Scale amplitudes by amp_scale.
    band_scaled = band * amp_scale

    def _plot_res(ax: plt.Axes, vals: np.ndarray, color: str, mag: bool) -> None:
        if not mag:
            ax.axhline(0.0, color="0.5", lw=0.4)
        ax.plot(f_slice, vals * amp_scale, color=color, lw=0.7)
        if band_scaled > 0.0:
            ax.axhline(band_scaled, color="0.3", lw=0.5, ls="--")
            if not mag:
                ax.axhline(-band_scaled, color="0.3", lw=0.5, ls="--")

    _vlines_at_peaks(ax_re_res, with_labels=True)
    _vlines_at_peaks(ax_im_res, with_labels=True)
    _vlines_at_peaks(ax_mag_res, with_labels=True)
    _plot_res(ax_re_res, np.real(residual), "tab:red", mag=False)
    _plot_res(ax_im_res, np.imag(residual), "tab:blue", mag=False)
    _plot_res(ax_mag_res, np.abs(residual), "tab:purple", mag=True)
    ax_re_res.set_ylabel(f"Re residual{amp_unit_suffix}", fontsize=9)
    ax_im_res.set_ylabel(f"Im residual{amp_unit_suffix}", fontsize=9)
    ax_mag_res.set_ylabel(f"|residual|{amp_unit_suffix}", fontsize=9)
    for ax in (ax_re_res, ax_im_res, ax_mag_res):
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", labelbottom=False)

    # Data+model (row 3). data as thin gray line + black markers; model thick
    # on the fine grid for a smooth lineshape.
    def _plot_data(
        ax: plt.Axes, dvals: np.ndarray, mvals_fine: np.ndarray, mcolor: str,
    ) -> None:
        ax.plot(f_slice, dvals * amp_scale, color="#00000044", lw=0.5, zorder=1)
        ax.plot(
            f_slice, dvals * amp_scale, marker="o", linestyle="None", markersize=2.0,
            markerfacecolor="black", markeredgecolor="black", zorder=2,
        )
        ax.plot(f_fine, mvals_fine * amp_scale, color=mcolor, lw=1.2, zorder=3)

    _vlines_at_peaks(ax_re_dat, with_labels=False)
    _vlines_at_peaks(ax_im_dat, with_labels=False)
    _vlines_at_peaks(ax_mag_dat, with_labels=False)
    _plot_data(ax_re_dat, np.real(z_slice), np.real(model_fine), "tab:red")
    _plot_data(ax_im_dat, np.imag(z_slice), np.imag(model_fine), "tab:blue")
    _plot_data(ax_mag_dat, np.abs(z_slice), np.abs(model_fine), "tab:purple")
    ax_re_dat.set_ylabel(f"Re{amp_unit_suffix}", fontsize=9)
    ax_im_dat.set_ylabel(f"Im{amp_unit_suffix}", fontsize=9)
    ax_mag_dat.set_ylabel(f"|X|{amp_unit_suffix}", fontsize=9)
    for ax in (ax_re_dat, ax_im_dat, ax_mag_dat):
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel("frequency (MHz)", fontsize=9)

    # --- Row 4: |residual| histogram + Rayleigh ------------------------
    mag_res = np.abs(residual) * amp_scale
    sigma_c = (float(np.median(sigma_slice)) / np.sqrt(2.0)) * amp_scale
    if sigma_c > 0.0 and mag_res.size > 0:
        n_bins = max(10, min(40, mag_res.size // 5))
        ax_hist.hist(
            mag_res, bins=n_bins, density=True,
            color="0.75", edgecolor="0.3", linewidth=0.4, label="|residual|",
        )
        x_max = max(float(mag_res.max()), 5.0 * sigma_c)
        x = np.linspace(0.0, x_max, 400)
        rayleigh = (x / (sigma_c ** 2)) * np.exp(-(x ** 2) / (2.0 * sigma_c ** 2))
        ax_hist.plot(
            x, rayleigh, color="tab:purple", lw=1.0,
            label=r"Rayleigh($\sigma/\sqrt{2}$)",
        )
        ax_hist.axvline(
            3.0 * sigma_c, color="tab:red", lw=0.6, ls="--",
            label=r"$3\sigma_c$",
        )
        ax_hist.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_hist.set_xlabel(f"|residual|{amp_unit_suffix}", fontsize=9)
    ax_hist.set_ylabel("density", fontsize=9)
    ax_hist.tick_params(axis="both", labelsize=8)
    ax_hist.set_title("|residual| vs noise", fontsize=9)

    # --- Row 4 right: peak listing ------------------------------------
    ax_peaks.set_title("Fitted peaks (consolidated)", fontsize=9, loc="left")
    n_peaks = len(consolidated_wf.fitted_peaks)
    line_h = 1.0 / max(n_peaks + 1, 8)  # ~8 lines comfortably, more compress
    amp_col_header = (
        f"amplitude ({units_lbl})" if units_lbl else "amplitude"
    )
    header = (
        f"  peak |  frequency (MHz)     |  {amp_col_header:<15}  |  p_KO (origin)"
    )
    ax_peaks.text(
        0.02, 0.98, header,
        transform=ax_peaks.transAxes, family="monospace", fontsize=8.5,
        va="top", color="0.3",
    )
    ax_peaks.text(
        0.02, 0.98 - 0.5 * line_h,
        "  " + "-" * (len(header) - 2),
        transform=ax_peaks.transAxes, family="monospace", fontsize=8.5,
        va="top", color="0.5",
    )
    for i, (pk, lbl) in enumerate(zip(consolidated_wf.fitted_peaks, labels)):
        freq_s = _format_spectroscopic(
            float(pk.frequency_mhz), pk.frequency_error,
        )
        amp_val = float(pk.amplitude) * amp_scale
        amp_err_scaled = (
            float(pk.amplitude_error) * amp_scale
            if pk.amplitude_error is not None
            else None
        )
        amp_s = _format_spectroscopic_sci(amp_val, amp_err_scaled)
        prov = (
            peak_provenance[i] if i < len(peak_provenance)
            else (None, None, None)
        )
        p_s = _format_p_origin(*prov)
        line = f"   {lbl:>2}  | {freq_s:>20} | {amp_s:>15} | {p_s}"
        y = 0.98 - (i + 1.5) * line_h
        ax_peaks.text(
            0.02, y, line,
            transform=ax_peaks.transAxes, family="monospace", fontsize=8.5,
            va="top",
        )

    return fig


def _plot_audit_trail_figure(
    window: FitWindow,
    consolidated_wf: FittingResult,
    *,
    initial_fit: ConservativeFitResult,
    consolidated: ConsolidatedRescueOutcome,
    f_slice: np.ndarray,
    z_slice: np.ndarray,
    model_slice: np.ndarray,
    f_fine: np.ndarray,
    model_fine: np.ndarray,
    sideband,
    center_mhz: float,
    peak_provenance: Sequence[Tuple[Optional[str], Optional[float], Optional[str]]],
    title: str,
    style: DisplayStyle,
) -> plt.Figure:
    """Render Figure 2 (portrait letter): rescue audit trail.

    Top: the window's magnitude spectrum with the final consolidated model
    overlay (data scaled by ``style.amplitude_scale``).

    Bottom: per-round audit panel. **Chronological flow goes bottom-to-top**
    so the final consolidated peaks sit immediately below the spectrum --
    the dotted vlines that connect final peaks up to the spectrum then have
    the shortest possible reach. Layout (bottom row to top row):

      * **initial fit** (y = Y_INIT) -- blue circles at each initial peak.
      * **round i candidates row** -- detector candidates (green
        triangles) and the rescue's conservative-fit accepted peaks
        (open green circles).
      * **round i merge row** -- knockout-pruned peaks (red X; red border
        marks rescue-origin pruning, the failsafe diagnostic). Joint-refit
        K transition annotated on the left.
      * **round i+1** ... (each round adds a fixed ``Y_PER_ROUND`` units).
      * **final consolidated** (y = Y_FINAL) -- purple circles, p-value
        annotations under each, and dotted vlines reaching UP to the
        spectrum.

    Fixed vertical increment per row: figure height grows with the number
    of rounds. Small chains leave trailing blank space at the bottom rather
    than stretching individual rows.
    """
    amp_scale = float(style.amplitude_scale)
    units_lbl = style.units_label
    amp_unit_suffix = f" ({units_lbl})" if units_lbl else ""

    s = sideband_sign(sideband)
    lo, hi = window.freq_range
    n_rounds = len(consolidated.rounds)

    # Bottom-up y-layout. y=0 is initial (bottom); each round adds
    # Y_PER_ROUND = 2 (candidates + merge); final sits Y_FINAL_OFFSET above
    # the last round.
    Y_INIT = 0.0
    Y_PER_ROUND = 2.0
    Y_FINAL_OFFSET = 1.0
    Y_FINAL = Y_INIT + 1.0 + n_rounds * Y_PER_ROUND + Y_FINAL_OFFSET
    # Fixed audit inches per y-unit; figure height grows with content.
    INCH_PER_Y_UNIT = 0.45
    SPECTRUM_INCHES = 3.0
    audit_inches = (Y_FINAL + 1.5) * INCH_PER_Y_UNIT
    fig_height = max(7.0, SPECTRUM_INCHES + audit_inches + 1.0)

    fig = plt.figure(figsize=(8.5, fig_height))
    fig.suptitle(title, fontsize=10)
    title_inches = 0.5
    bottom_inches = 0.6
    gs = GridSpec(
        nrows=2, ncols=1, figure=fig,
        height_ratios=[SPECTRUM_INCHES, audit_inches],
        hspace=0.0,
        left=0.09, right=0.97,
        top=1.0 - title_inches / fig_height,
        bottom=bottom_inches / fig_height,
    )
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_audit = fig.add_subplot(gs[1, 0], sharex=ax_spec)

    # --- Top: fitted window magnitude + final model -----------------
    ax_spec.plot(
        f_slice, np.abs(z_slice) * amp_scale,
        color="0.25", lw=0.7, label="data |X|", zorder=2,
    )
    ax_spec.plot(
        f_slice, np.abs(z_slice) * amp_scale,
        marker="o", linestyle="None", markersize=2.2,
        markerfacecolor="0.15", markeredgecolor="0.15", zorder=3,
    )
    ax_spec.plot(
        f_fine, np.abs(model_fine) * amp_scale,
        color="tab:purple", lw=1.4, label="consolidated model |X|", zorder=4,
    )
    ax_spec.set_ylabel(f"|X(f)|{amp_unit_suffix}", fontsize=9)
    ax_spec.set_xlim(min(lo, hi), max(lo, hi))
    ax_spec.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_spec.tick_params(axis="y", labelsize=8)
    ax_spec.spines["bottom"].set_visible(False)
    ax_spec.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_audit.spines["top"].set_visible(False)

    # --- Bottom: audit panel (bottom-up flow) -----------------------
    ax_audit.set_xlim(min(lo, hi), max(lo, hi))
    ax_audit.set_ylim(-0.6, Y_FINAL + 0.8)
    ax_audit.tick_params(axis="x", labelsize=9)
    ax_audit.set_xlabel("frequency (MHz)", fontsize=9)
    ax_audit.set_yticks([])
    ax_audit.set_ylabel(
        "rescue audit  (bottom: initial, top: final)", fontsize=9,
    )

    def _label_left(y: float, text: str, color: str = "0.25") -> None:
        ax_audit.annotate(
            text,
            xy=(0.0, y), xycoords=("axes fraction", "data"),
            xytext=(2, 0), textcoords="offset points",
            ha="left", va="center", fontsize=7.5, color=color,
            bbox=dict(
                boxstyle="round,pad=0.15", fc="white", ec="0.85", lw=0.4, alpha=0.9,
            ),
        )

    def _hline(y: float) -> None:
        ax_audit.axhline(y, color="0.65", lw=0.5, ls=":")

    final_freqs = [float(p.frequency_mhz) for p in consolidated_wf.fitted_peaks]

    # --- Initial row (bottom) -----------------------------------------
    init_peak_freqs = [
        center_mhz + s * pk.offset_mhz for pk in initial_fit.fit.peaks
    ]
    for f_pk in init_peak_freqs:
        ax_audit.plot(
            f_pk, Y_INIT,
            marker="o", markersize=8, markerfacecolor="tab:blue",
            markeredgecolor="black", markeredgewidth=0.6,
            zorder=3,
        )
    _label_left(Y_INIT, f"initial: K={initial_fit.n_peaks}", color="tab:blue")

    # --- Per-round bands (going UP as i increases) -------------------
    for i, diag in enumerate(consolidated.rounds):
        y_separator = 0.7 + i * Y_PER_ROUND
        y_candidates = 1.0 + i * Y_PER_ROUND
        y_merge = 2.0 + i * Y_PER_ROUND
        _hline(y_separator)
        _label_left(
            y_separator,
            f"round {i} ({'accepted' if diag.accepted else 'rejected'})",
            color="0.35",
        )

        rescue = diag.rescue
        for c in rescue.candidates:
            f_c = center_mhz + s * float(c.frequency_mhz)
            ax_audit.plot(
                f_c, y_candidates,
                marker="v", markersize=6, markerfacecolor="tab:green",
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=3,
            )
        for pk in rescue.fit.peaks:
            f_pk = center_mhz + s * pk.offset_mhz
            ax_audit.plot(
                f_pk, y_candidates,
                marker="o", markersize=8, markerfacecolor="none",
                markeredgecolor="tab:green", markeredgewidth=1.2,
                zorder=4,
            )
        _label_left(
            y_candidates,
            f"  candidates: {len(rescue.candidates)}, "
            f"{rescue.fit.n_peaks} fit-accepted",
        )

        if diag.joint_fit is not None:
            for ko in diag.joint_knockouts:
                if ko.supported:
                    continue
                f_ko = center_mhz + s * float(ko.offset_mhz)
                is_rescue_origin = (
                    diag.n_initial_peaks
                    <= ko.peak_index
                    < diag.n_initial_peaks + diag.n_rescue_added
                )
                edge_color = "tab:red" if is_rescue_origin else "0.3"
                edge_width = 1.5 if is_rescue_origin else 0.8
                ax_audit.plot(
                    f_ko, y_merge,
                    marker="X", markersize=10, markerfacecolor="tab:red",
                    markeredgecolor=edge_color, markeredgewidth=edge_width,
                    zorder=3,
                )
            joint_k = diag.joint_fit.n_peaks
            pruned_k = diag.pruned_fit.n_peaks if diag.pruned_fit else joint_k
            failsafe = (
                f"  [!fail: {diag.n_pruned_rescue_origin} rescue-origin pruned]"
                if diag.n_pruned_rescue_origin > 0
                else ""
            )
            _label_left(
                y_merge,
                f"  merge: K {diag.n_initial_peaks}+{diag.n_rescue_added}={joint_k} "
                f"-> {pruned_k} (knockout pruned {diag.n_pruned_total}){failsafe}",
            )
        else:
            _label_left(
                y_merge,
                f"  no merge (rescue produced 0 peaks)",
                color="0.5",
            )

    # --- Final row (top, touching spectrum) -------------------------
    _hline(Y_FINAL - 0.5)
    for f_pk in final_freqs:
        ax_audit.plot(
            f_pk, Y_FINAL,
            marker="o", markersize=10, markerfacecolor="tab:purple",
            markeredgecolor="black", markeredgewidth=0.7,
            zorder=4,
        )
    _label_left(
        Y_FINAL,
        f"final consolidated: K={len(final_freqs)}",
        color="tab:purple",
    )
    y_final = Y_FINAL  # backward-compat for downstream code

    # Dotted vertical lines from final peaks UP to the spectrum panel,
    # plus per-peak p-value annotation just BELOW the marker (away from
    # the spectrum, into the audit panel's free space). When peaks are
    # close in frequency (within ~1 FWHM), stagger the offset so labels
    # don't overlap. With non-inverted y, "below" is negative offset.
    tau_us_final = float(consolidated.fit.fit.tau_us)
    fwhm_final = (
        1.0 / (np.pi * tau_us_final) if tau_us_final > 0.0 else 0.0
    )
    annot_offsets = [-12.0] * len(final_freqs)
    order = sorted(range(len(final_freqs)), key=lambda i: final_freqs[i])
    last_used_offset = -12.0
    for idx_in_order, idx in enumerate(order):
        if idx_in_order == 0:
            annot_offsets[idx] = -12.0
            last_used_offset = -12.0
            continue
        prev_idx = order[idx_in_order - 1]
        if (
            fwhm_final > 0.0
            and abs(final_freqs[idx] - final_freqs[prev_idx]) < fwhm_final
        ):
            annot_offsets[idx] = -26.0 if last_used_offset == -12.0 else -12.0
            last_used_offset = annot_offsets[idx]
        else:
            annot_offsets[idx] = -12.0
            last_used_offset = -12.0

    for f_pk, prov, y_off in zip(final_freqs, peak_provenance, annot_offsets):
        for ax in (ax_spec, ax_audit):
            ax.axvline(
                f_pk, color="tab:purple", lw=0.6, ls=":", alpha=0.5,
                zorder=1,
            )
        _, p_val, decision = prov
        if p_val is None:
            label = "joint"
        elif p_val <= 0.0:
            label = "<1e-15" if decision in ("accept", "promote") else "seed"
        else:
            label = f"{p_val:.1e}"
        ax_audit.annotate(
            label,
            xy=(f_pk, Y_FINAL), xytext=(0, y_off), textcoords="offset points",
            ha="center", va="top", fontsize=7, color="tab:purple",
        )

    return fig


def _save_audit_trail_figure_wrapper(
    out_dir: Path,
    window: FitWindow,
    *,
    consolidated: ConsolidatedRescueOutcome,
    original_wf: FittingResult,
    center_mhz: float,
    freqs_sorted: np.ndarray,
    spec_sorted: np.ndarray,
    rms_sorted: np.ndarray,
    sideband,
    acquisition_us: float,
    style: DisplayStyle,
) -> None:
    """Write ``audit-trail.png`` (Figure 2)."""
    consolidated_wf = _build_consolidated_fittingresult(
        window, original_wf, consolidated.fit.fit, sideband, center_mhz,
        label="final",
    )
    # Compute window slice + final consolidated model on slice.
    s = sideband_sign(sideband)
    lo, hi = window.freq_range
    mask = (freqs_sorted >= min(lo, hi)) & (freqs_sorted <= max(lo, hi))
    f_slice = freqs_sorted[mask]
    z_slice = spec_sorted[mask]
    center = 0.5 * (lo + hi)
    u_slice = s * (f_slice - center)
    tau_us = float(consolidated.fit.fit.tau_us)
    all_peaks: List[ModelPeak] = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in consolidated_wf.fitted_peaks
    ]
    for key, fp_data in consolidated_wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        all_peaks.append(
            ModelPeak(
                amplitude=float(fp_data["amplitude"]),
                offset_mhz=float(s * (float(fp_data["frequency_mhz"]) - center)),
                phase=float(fp_data.get("phase", 0.0) or 0.0),
            )
        )
    shape_str = getattr(consolidated_wf, "shape", "lorentzian")
    if all_peaks and tau_us > 0.0:
        model_slice = model_spectrum(
            u_slice, all_peaks, tau_us, acquisition_us, shape=shape_str,
        )
    else:
        model_slice = np.zeros_like(z_slice)
    # Fine-grid model curve for the smooth overlay on the audit-trail spectrum
    # panel. χ²ᵣ and other stats are computed elsewhere from ``model_slice``.
    if f_slice.size >= 2:
        n_fine = (f_slice.size - 1) * MODEL_OVERSAMPLE + 1
        f_fine = np.linspace(float(f_slice.min()), float(f_slice.max()), n_fine)
        u_fine = s * (f_fine - center)
        if all_peaks and tau_us > 0.0:
            model_fine = model_spectrum(
                u_fine, all_peaks, tau_us, acquisition_us, shape=shape_str,
            )
        else:
            model_fine = np.zeros_like(f_fine, dtype=np.complex128)
    else:
        f_fine = f_slice.copy()
        model_fine = model_slice.copy()
    provenance = _peak_provenance(
        consolidated.initial_fit, consolidated, consolidated_wf,
        sideband, center,
    )
    n_rounds_total = len(consolidated.rounds)
    n_rounds_accepted = sum(1 for d in consolidated.rounds if d.accepted)
    final_chi2 = float(consolidated.fit.fit.chi_squared)
    final_n_residual = max(
        consolidated.fit.fit.n_data - consolidated.fit.fit.n_params, 1
    )
    final_rchi2 = final_chi2 / final_n_residual
    title = (
        f"Window {window.window_id} -- rescue audit trail  "
        f"[{lo:.2f}, {hi:.2f}] MHz  "
        f"K {consolidated.initial_fit.n_peaks} -> "
        f"{len(consolidated_wf.fitted_peaks)}  "
        f"chi2_r {float(consolidated.initial_fit.fit.chi_squared) / max(1, consolidated.initial_fit.fit.n_data - consolidated.initial_fit.fit.n_params):.2f} -> "
        f"{final_rchi2:.2f}  "
        f"({n_rounds_accepted}/{n_rounds_total} rounds accepted)"
    )
    fig = _plot_audit_trail_figure(
        window, consolidated_wf,
        initial_fit=consolidated.initial_fit,
        consolidated=consolidated,
        f_slice=f_slice, z_slice=z_slice, model_slice=model_slice,
        f_fine=f_fine, model_fine=model_fine,
        sideband=sideband, center_mhz=center,
        peak_provenance=provenance,
        title=title,
        style=style,
    )
    fig.savefig(out_dir / "audit-trail.png", dpi=130)
    plt.close(fig)


def _save_consolidated_detail(
    out_dir: Path,
    window: FitWindow,
    *,
    consolidated: ConsolidatedRescueOutcome,
    original_wf: FittingResult,
    other_window_fits: Sequence[FittingResult],
    center_mhz: float,
    freqs_sorted: np.ndarray,
    spec_sorted: np.ndarray,
    rms_sorted: np.ndarray,
    sideband,
    acquisition_us: float,
    style: DisplayStyle,
) -> None:
    """Write ``detail.png`` showing the final consolidated fit (Figure 1)."""
    consolidated_wf = _build_consolidated_fittingresult(
        window, original_wf, consolidated.fit.fit, sideband, center_mhz,
        label="final",
    )
    provenance = _peak_provenance(
        consolidated.initial_fit, consolidated, consolidated_wf,
        sideband, center_mhz,
    )
    lo, hi = window.freq_range
    final_chi2 = float(consolidated.fit.fit.chi_squared)
    final_n_residual = max(
        consolidated.fit.fit.n_data - consolidated.fit.fit.n_params, 1
    )
    final_rchi2 = final_chi2 / final_n_residual
    n_rounds_total = len(consolidated.rounds)
    n_rounds_accepted = sum(1 for d in consolidated.rounds if d.accepted)
    rounds_note = (
        f"{n_rounds_accepted}/{n_rounds_total} rescue rounds accepted"
        if n_rounds_total > 0
        else "no rescue rounds"
    )
    title = (
        f"Window {window.window_id}  [{lo:.2f}, {hi:.2f}] MHz  "
        f"K={len(consolidated_wf.fitted_peaks)}  "
        f"chi2_r={final_rchi2:.2f}  "
        f"tau={float(consolidated.fit.fit.tau_us):.3g} us  "
        f"(consolidated, {rounds_note})"
    )
    fig = _plot_consolidated_detail(
        window, consolidated_wf, other_window_fits,
        frequencies=freqs_sorted,
        complex_spectrum=spec_sorted,
        rms_noise=rms_sorted,
        sideband=sideband,
        acquisition_us=acquisition_us,
        peak_provenance=provenance,
        title=title,
        style=style,
    )
    fig.savefig(out_dir / "detail.png", dpi=130)
    plt.close(fig)


def _save_rescue_artifacts(
    out_dir: Path,
    window: FitWindow,
    wf: FittingResult,
    *,
    consolidated: ConsolidatedRescueOutcome,
    initial_fit: ConservativeFitResult,
    center_mhz: float,
    freqs_sorted: np.ndarray,
    spec_sorted: np.ndarray,
    rms_sorted: np.ndarray,
    sideband,
    acquisition_us: float,
) -> None:
    """Write per-round ``detail-rr<n>.png`` figures and a single
    ``report-rr.md`` rollup in ``out_dir``.

    Each ``detail-rr<n>.png`` mirrors the original ``detail.png`` layout
    (full window data, model overlay, residual panel) but with the
    *consolidated* fit at the end of round ``n`` (the cumulative
    initial-plus-rescue peaks, jointly refit and knockout-pruned). Read
    side-by-side with ``detail.png`` (initial only), the chain shows the
    convergence trajectory of the rescue B-loop.

    The single ``report-rr.md`` covers the whole chain: one section per
    round with the rescue / joint-refit / knockout stats, plus the
    failsafe diagnostic (``n_pruned_rescue_origin`` -- nonzero means the
    joint refit may not be escaping pathological basins).
    """
    lo, hi = window.freq_range

    for diag in consolidated.rounds:
        joint = diag.pruned_fit
        if joint is None:
            continue  # nothing new to plot (rescue accepted 0, or all pruned)
        round_wf = _build_consolidated_fittingresult(
            window, wf, joint, sideband, center_mhz,
            label=f"rr{diag.round_idx}",
        )
        round_fit = SpectrumFit(
            window_fits=[round_wf],
            parameters={"source": f"rescue_consolidated_round{diag.round_idx}"},
        )
        init_rchi2 = float(initial_fit.fit.chi_squared) / max(
            1, initial_fit.fit.n_data - initial_fit.fit.n_params,
        )
        round_rchi2 = float(joint.chi_squared) / max(
            1, joint.n_data - joint.n_params,
        )
        title = (
            f"Window {window.window_id} RESCUE round {diag.round_idx}  "
            f"[{lo:.2f}, {hi:.2f}] MHz  "
            f"K {diag.n_initial_peaks}+{diag.n_rescue_added}->"
            f"{len(joint.peaks)} peaks  "
            f"chi2_r init {init_rchi2:.2f} -> after round{diag.round_idx} "
            f"{round_rchi2:.2f}  "
            f"(pruned by knockout: {diag.n_pruned_total} total / "
            f"{diag.n_pruned_rescue_origin} rescue-origin)"
        )
        fig = plot_spectrum_fit(
            frequencies=freqs_sorted,
            complex_spectrum=spec_sorted,
            rms_noise=rms_sorted,
            fit=round_fit,
            sideband=sideband,
            acquisition_us=acquisition_us,
            window_id=window.window_id,
            figsize=(11, 13),
            title=title,
        )
        fig.savefig(out_dir / f"detail-rr{diag.round_idx}.png", dpi=120)
        plt.close(fig)

    (out_dir / "report-rr.md").write_text(
        _rescue_report(window, consolidated, initial_fit)
    )


def _rescue_report(
    window: FitWindow,
    consolidated: ConsolidatedRescueOutcome,
    initial_fit: ConservativeFitResult,
) -> str:
    """Rollup of the B-loop chain: initial fit + one section per round."""
    init_chi2 = float(initial_fit.fit.chi_squared)
    init_n_residual = max(initial_fit.fit.n_data - initial_fit.fit.n_params, 1)
    init_rchi2 = init_chi2 / init_n_residual
    final_fit = consolidated.fit
    final_chi2 = float(final_fit.fit.chi_squared)
    final_n_residual = max(final_fit.fit.n_data - final_fit.fit.n_params, 1)
    final_rchi2 = final_chi2 / final_n_residual

    parts = [
        f"# Window {window.window_id} - Residual Rescue (B-loop)",
        "",
        "_Each round: detect candidates in the current residual, fit "
        "them on the residual (rescue) with frozen tau, then joint-refit "
        "the (initial + rescue) union with all parameters thawed "
        "(starting tau from the rescue's apodization-aware value), then "
        "knockout-prune any unsupported peaks. The next round operates "
        "on the residual of the consolidated fit. Terminates when the "
        "rescue accepts nothing, the joint refit fails, or all peaks "
        "get pruned._",
        "",
        f"**Loop terminated:** `{consolidated.terminated_reason}` "
        f"after {len(consolidated.rounds)} round(s)",
        "",
        "## Initial fit (round -1, baseline)",
        f"- n_peaks: {initial_fit.n_peaks}",
        f"- chi-squared: {init_chi2:.4g}",
        f"- reduced chi-squared: {init_rchi2:.4g}",
        f"- AIC: {float(initial_fit.fit.aic):.4g}",
        f"- tau (us): {float(initial_fit.fit.tau_us):.4g}",
        "",
        "## Consolidated final state",
        f"- n_peaks: **{final_fit.n_peaks}** "
        f"(net change vs initial: {final_fit.n_peaks - initial_fit.n_peaks:+d})",
        f"- chi-squared: {final_chi2:.4g}",
        f"- reduced chi-squared: **{final_rchi2:.4g}** "
        f"(initial {init_rchi2:.4g})",
        f"- AIC: {float(final_fit.fit.aic):.4g}",
        f"- tau (us): {float(final_fit.fit.tau_us):.4g}",
        "",
    ]

    if not consolidated.rounds:
        parts.append(
            "_No rescue rounds executed -- the rescue chain produced "
            "no candidates on its first pass._"
        )
        return "\n".join(parts) + "\n"

    for diag in consolidated.rounds:
        parts.extend(_rescue_round_section(diag))
    return "\n".join(parts) + "\n"


def _rescue_round_section(diag: RescueRoundDiagnostics) -> List[str]:
    """One per-round section of the rescue rollup report."""
    rescue = diag.rescue
    rounded_status = "ACCEPTED" if diag.accepted else "REJECTED"
    parts = [
        f"## Round {diag.round_idx} ({rounded_status})",
        f"- reason: _{diag.reason}_",
        f"- inherited peaks: {diag.n_initial_peaks}",
        f"- detector candidates: {len(rescue.candidates)}",
        f"- rescue accepted: **{diag.n_rescue_added}** peak(s) "
        f"({len(rescue.audit)} audit entries)",
    ]
    if diag.joint_fit is not None:
        joint_rchi2 = float(diag.joint_fit.chi_squared) / max(
            1, diag.joint_fit.n_data - diag.joint_fit.n_params,
        )
        parts.append(
            f"- joint refit: success={diag.joint_fit.success}, "
            f"K={len(diag.joint_fit.peaks)}, "
            f"chi2_r={joint_rchi2:.4g}, "
            f"tau={float(diag.joint_fit.tau_us):.4g} us"
        )
    parts.append(
        f"- knockout pruned: {diag.n_pruned_total} total, "
        f"**{diag.n_pruned_rescue_origin}** of rescue origin "
        f"(failsafe: nonzero rescue-origin pruning means the joint refit "
        f"may not be escaping a pathological basin)"
    )
    parts.append(
        f"- chi^2 movement: {diag.chi2_before:.4g} -> {diag.chi2_after:.4g} "
        f"(tau {diag.tau_us_before:.4g} -> {diag.tau_us_after:.4g} us)"
    )

    if rescue.candidates:
        parts.append("")
        parts.append("### Round candidates")
        parts.append("| # | offset (MHz) | |residual| | SNR | prom (sigma_c) |")
        parts.append("|---:|---:|---|---:|---:|")
        for i, c in enumerate(rescue.candidates):
            parts.append(
                f"| C{i} | {c.frequency_mhz:+.4f} | {c.magnitude:.4g} | "
                f"{c.snr:.2f} | {c.prominence_sigma_c:.2f} |"
            )

    if diag.joint_fit is not None and diag.pruned_knockouts:
        parts.append("")
        parts.append("### Knockout sweep on consolidated fit")
        parts.append("| peak | offset (MHz) | delta chi^2 | expected | supported |")
        parts.append("|---:|---:|---:|---:|:---|")
        for ko in diag.pruned_knockouts:
            parts.append(
                f"| {ko.peak_index} | {ko.offset_mhz:+.4f} | "
                f"{ko.delta_chi2:.4g} | {ko.expected_delta_chi2:.4g} | "
                f"{'yes' if ko.supported else 'NO'} |"
            )

    parts.append("")
    return parts


def _window_report(
    window: FitWindow,
    wf: FittingResult,
    peaks_loaded: Sequence,
    note: Optional[str] = None,
    residual_candidates: Sequence[ResidualPeakCandidate] = (),
    residual_summary: Optional[str] = None,
) -> str:
    chi2_total = float(wf.reduced_chi2) * max(int(wf.iterations), 1) if False else None
    parts = [
        f"# Window {window.window_id}",
        "",
    ]
    if note:
        parts.append(f"> {note}")
        parts.append("")
    parts += [
        "## Window plan (Stage 4)",
        f"- freq range: [{window.freq_range[0]:.4f}, {window.freq_range[1]:.4f}] MHz",
        f"- free peaks (Stage 3 indices): {window.free_peak_indices}",
        _difficulty_block(window),
        "",
        "## Fixed contributors",
        _fixed_block(window, peaks_loaded),
        "",
        "## Fit statistics (Stage 5)",
        f"- success: **{wf.success}**",
        f"- iterations: {wf.iterations}",
        f"- cost: {wf.cost:.4g}",
        f"- AIC: {wf.aic:.4g}",
        f"- reduced chi-squared: **{wf.reduced_chi2:.4g}**",
        f"- edge_coherence_low: {_fmt_opt(wf.quality_metrics.get('edge_coherence_low'))}",
        f"- edge_coherence_high: {_fmt_opt(wf.quality_metrics.get('edge_coherence_high'))}",
        "",
        "## Shared parameters",
        _shared_block(wf),
        "",
        "## Fitted peaks",
        _peak_table(wf.fitted_peaks),
        "",
        "## Audit trail (conservative add-one-peak loop)",
        _audit_table(wf),
        "",
        "## Thaw events on this window",
        _thaw_table(wf),
        "",
        "## Residual peak candidates (prototype detector)",
    ]
    if residual_summary:
        parts.append(residual_summary)
    parts.append(_residual_candidate_block(residual_candidates))
    parts.append("")
    return "\n".join(parts)


def _save_window_artifacts(
    out_dir: Path,
    window: FitWindow,
    wf: FittingResult,
    peaks_loaded: Sequence,
    *,
    frequencies,
    complex_spectrum,
    rms_noise,
    fit: SpectrumFit,
    sideband,
    acquisition_us: float,
    note: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Recompute the per-window residual so the detector and the visualization
    # see the same data. ``existing_freqs`` aggregates fitted + frozen
    # contributors -- a residual peak co-located with either is a
    # fit-quality flag rather than a missed line.
    f_slice, residual, sigma_slice, _model_slice, fwhm_mhz = (
        _compute_window_residual(
            wf, frequencies, complex_spectrum, rms_noise,
            sideband, acquisition_us,
        )
    )
    existing_freqs: List[float] = [
        float(p.frequency_mhz) for p in wf.fitted_peaks
    ]
    existing_ids: List[int] = [int(p.peak_id) for p in wf.fitted_peaks]
    for key, fp_data in wf.fixed_parameters.items():
        if key.startswith("frozen_peak_"):
            existing_freqs.append(float(fp_data["frequency_mhz"]))
            existing_ids.append(-1)
    candidates = find_residual_peaks(
        f_slice,
        residual,
        sigma_slice,
        snr_threshold=DEFAULT_RESCUE_SNR_THRESHOLD,
        prominence_threshold=DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
        fwhm_mhz=fwhm_mhz if fwhm_mhz > 0 else None,
        existing_freqs_mhz=existing_freqs,
        existing_peak_ids=existing_ids,
    )
    sigma_c_med = float(np.median(sigma_slice)) / np.sqrt(2.0)
    residual_summary = (
        f"_detector settings (aligned with rescue): "
        f"SNR ≥ {DEFAULT_RESCUE_SNR_THRESHOLD}, "
        f"prominence ≥ {DEFAULT_RESCUE_PROMINENCE_THRESHOLD}σ_c, "
        f"FWHM = {fwhm_mhz:.4f} MHz (=1/πτ); "
        f"median σ_c = {sigma_c_med:.3g}; "
        f"{len(candidates)} candidate(s)_"
    )

    # Detail PNG is now emitted by ``_save_consolidated_detail`` (Figure 1
    # showing the post-rescue consolidated fit). Here we keep the residual
    # detector summary live so ``report.md`` continues to surface what the
    # initial fit's residual still had above threshold -- useful pre-rescue
    # context even when the rescue then folded those candidates in.

    (out_dir / "report.md").write_text(
        _window_report(
            window, wf, peaks_loaded, note=note,
            residual_candidates=candidates,
            residual_summary=residual_summary,
        )
    )


def _category_block(
    category: str,
    entries: Iterable[Tuple[int, FitWindow, FittingResult, Optional[str], Path]],
) -> str:
    parts = [f"## {category}", ""]
    for window_id, window, wf, note, rel_dir in entries:
        snr_max = max(((p.snr or 0.0) for p in wf.fitted_peaks), default=0.0)
        suffix = f" -- _{note}_" if note else ""
        parts.append(
            f"- [`{rel_dir.name}/`]({rel_dir.name}/report.md) "
            f"**window {window_id}** -- [{window.freq_range[0]:.2f}, "
            f"{window.freq_range[1]:.2f}] MHz, free={window.n_free_peaks}, "
            f"fixed={len(window.fixed_contributors)}, "
            f"chi2_r={wf.reduced_chi2:.2f}, peaks={len(wf.fitted_peaks)}, "
            f"snr_max={snr_max:.1f}{suffix}"
        )
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    import argparse

    arg_parser = argparse.ArgumentParser(
        description=(
            "Generate Stage 5 validation artifacts for the 2638 fixture. "
            "Default: emits the deliberate EASY + HARD + NAMED samples. "
            "Pass --window-id N (repeatable) to restrict to specific windows."
        )
    )
    arg_parser.add_argument(
        "--window-id",
        type=int,
        action="append",
        dest="window_ids",
        help=(
            "Restrict artifact generation to this window id. Repeat the "
            "flag to select multiple windows. When provided, overrides "
            "the EASY/HARD/NAMED sample lists; the overview / INDEX.md "
            "are still emitted but cover only the requested windows."
        ),
    )
    arg_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output / fixture directory. Accepts an absolute "
            "path or a path relative to the repo root. Defaults to "
            "``scratch/stage5-validation``."
        ),
    )
    arg_parser.add_argument(
        "--fixture-name",
        type=str,
        default=None,
        help=(
            "Override the .ftmw fixture filename inside the output dir. "
            "Defaults to ``exp_2638.ftmw``."
        ),
    )
    arg_parser.add_argument(
        "--named-window",
        action="append",
        dest="named_windows",
        default=[],
        metavar="ID:NOTE",
        help=(
            "Annotate a --window-id as a NAMED case in the INDEX.md "
            "(repeatable). Format: ``--named-window 234:'34154 anomaly'``. "
            "Only consulted when --window-id is used to override the "
            "default EASY/HARD/NAMED samples."
        ),
    )
    args = arg_parser.parse_args()

    # Re-bind the module-level OUTPUT_DIR / FTMW_PATH based on CLI flags.
    # The directory may not exist yet on first --output-dir invocation; the
    # fixture must.
    global OUTPUT_DIR, FTMW_PATH
    if args.output_dir is not None:
        odir = Path(args.output_dir)
        if not odir.is_absolute():
            odir = REPO_ROOT / odir
        OUTPUT_DIR = odir
    if args.fixture_name is not None:
        FTMW_PATH = OUTPUT_DIR / args.fixture_name
    else:
        FTMW_PATH = OUTPUT_DIR / FTMW_PATH.name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUTPUT_DIR}")
    print(f"fixture:    {FTMW_PATH}")

    # --named-window overrides NAMED_SAMPLE for the --window-id path.
    cli_named: List[Tuple[int, str]] = []
    for entry in (args.named_windows or []):
        if ":" not in entry:
            raise SystemExit(
                f"--named-window expects 'ID:NOTE', got {entry!r}"
            )
        sid, note = entry.split(":", 1)
        try:
            cli_named.append((int(sid), note.strip()))
        except ValueError as exc:
            raise SystemExit(f"invalid --named-window id {sid!r}: {exc}")

    if not FTMW_PATH.exists():
        raise SystemExit(
            f"missing {FTMW_PATH}; build it first by running the pipeline "
            "through fit_peaks (see the conversation log)."
        )

    plan: WindowPlan = ftmw.load_windows(str(FTMW_PATH))
    fit: SpectrumFit = ftmw.load_fit(str(FTMW_PATH))
    peaks_loaded = ftmw.load_peaks(str(FTMW_PATH))
    display_style = _load_display_style(FTMW_PATH)
    print(f"display style: {display_style}")

    # Validation plots the actual ACTIVE-FT (the spectrum the fit consumes)
    # with the model in its native units. No amplitude rescale / phase re-roll
    # is needed -- model amplitudes and phases live in the active-FT
    # ``dt_us * rfft(active)`` frame natively. The persisted FT is in a
    # different amplitude scale and a different phase frame; reconciling them
    # for production overlays is a separate concern (see TODO at top).
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        expf_us,
        probe_freq_mhz,
        sideband_enum,
        n_padded,
        acquisition_us,
        _user_ft,
        _user_rms,
    ) = _build_active_ft_inputs(str(FTMW_PATH))
    active_ft = compute_active_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband_enum,
        n_padded=n_padded,
    )
    # Sort active-FT by molecular frequency so the plot axis is ascending
    # (active-FT bin order is rfft order; for the lower sideband that maps to
    # descending molecular frequency).
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freqs_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(active_ft.complex_spectrum[sort_idx])
    # Active-FT noise: Stage 2 estimator on the sorted active-FT magnitude
    # (same way ``stage5_impl.fit_peaks_impl`` measures it for the fit).
    active_noise = estimate_noise_adaptive(
        freqs_sorted, np.abs(spec_sorted).astype(np.float64)
    )
    rms_sorted = np.asarray(active_noise.rms_noise, dtype=float)
    # materialize_window consumes the active-FT in its native (unsorted) bin
    # order; re-index the noise back to that order.
    active_noise_arr = rms_sorted[unsort_idx]
    sideband = sideband_enum

    fit_by_id = {wf.window_id: wf for wf in fit.window_fits}
    plan_by_id = {w.window_id: w for w in plan.windows}

    # Rescue-pass inputs: rebuild conservative_kwargs from the persisted
    # SpectrumFit parameters so the rescue's re-fit reproduces the persisted
    # fit exactly (no stochastic seeding -- the rescue then runs on top).
    params = fit.parameters or {}
    tau0_us_v = float(params.get("tau0_us", acquisition_us / 3.0))
    fit_tau_v = bool(params.get("fit_tau", True))
    # Carry the persisted line-shape selector forward so the harness's
    # re-fit on a Gaussian-fitted file reproduces the Gaussian model
    # (otherwise conservative_fit defaults to Lorentzian and the rescue's
    # re-fit silently drifts off the persisted fit).
    shape_str = str(params.get("shape", "lorentzian"))
    # Single shared conservative_kwargs (used both to reproduce the
    # initial fit and to drive the rescue's per-round conservative loop).
    # ``rescue_max_peaks`` is a rescue-only override on the per-round K
    # cap; the persisted initial fit's max_peaks lives in conservative_fit's
    # own DEFAULT_MAX_PEAKS=8, which we keep -- the rescue is the one place
    # we want generous headroom because the F-test gates are what should
    # decide K, not an arbitrary integer.
    init_conservative_kwargs = {
        "max_decay_factor": float(params.get("max_decay_factor", 5.0)),
        "tau_apodization_us": expf_us if expf_us else None,
        "shape": shape_str,
    }
    # Forward the persisted Stage 2b τ_maj / σ_τ pair so the rescue's
    # bidirectional Gaussian-prior penalty matches what fit_peaks_impl
    # uses in production. Falls through harmlessly when the persisted
    # parameters don't carry one (no calibration was active).
    tau_maj_persisted = params.get("tau_maj_us")
    sigma_tau_persisted = params.get("sigma_tau_us")
    if tau_maj_persisted is not None and sigma_tau_persisted is not None:
        init_conservative_kwargs["tau_maj_us"] = float(tau_maj_persisted)
        init_conservative_kwargs["sigma_tau_us"] = float(sigma_tau_persisted)
    rescue_conservative_kwargs = dict(init_conservative_kwargs)
    rescue_kwargs = {
        "snr_threshold": DEFAULT_RESCUE_SNR_THRESHOLD,
        "prominence_threshold": DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
        "rescue_max_peaks": 32,
        # Shape-error-aware sigma inflation for the rescue's screening
        # pipeline. epsilon = fractional Lorentzian-vs-true-lineshape
        # residual per unit parent amplitude (per-bin, not
        # chi^2_r-aggregated -- those are ~4-8x different in scale).
        # Empirical sweep on the 2638 fixture (diag_phase1_merge_gate.py):
        # epsilon=0.05 stops the w148/w269 rescue-merge limit cycle
        # without affecting borderline real-peak rescues (w16/w104/w127/
        # w337) or clean controls (w63/w64). Re-calibrate per dataset.
        "shape_error_epsilon": 0.05,
    }
    peak_frequencies_mhz = [float(p.frequency) for p in peaks_loaded]

    # --- overview ----------------------------------------------------------
    overview_title = (
        f"2638 Stage 5 overview - {fit.n_windows} windows, "
        f"{fit.n_fitted_peaks} fitted peaks, plan revision {fit.final_plan_revision} "
        f"(active-FT)"
    )
    fig = plot_spectrum_fit(
        frequencies=freqs_sorted,
        complex_spectrum=spec_sorted,
        rms_noise=rms_sorted,
        fit=fit,
        sideband=sideband,
        acquisition_us=acquisition_us,
        figsize=(18, 8),
        title=overview_title,
    )
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=120)
    plt.close(fig)

    # --- per-window artifacts ---------------------------------------------
    def emit(window_id: int, note: Optional[str]) -> Path:
        window = plan_by_id[window_id]
        wf = fit_by_id[window_id]
        rel_dir = OUTPUT_DIR / f"window_{window_id:03d}"
        rel_dir.mkdir(parents=True, exist_ok=True)
        # Rescue artifacts: reproduce the initial fit, run the rescue, write
        # detail-rr.png + report-rr.md alongside the original detail.png /
        # report.md (suffix only; the user reads them side-by-side).
        try:
            consolidated, initial_fit, center_mhz = _run_window_rescue(
                window,
                wf,
                active_ft=active_ft,
                active_noise_arr=active_noise_arr,
                sideband=sideband,
                acquisition_us=acquisition_us,
                tau0_us=tau0_us_v,
                fit_tau=fit_tau_v,
                peak_frequencies_mhz=peak_frequencies_mhz,
                init_conservative_kwargs=init_conservative_kwargs,
                rescue_conservative_kwargs=rescue_conservative_kwargs,
                rescue_kwargs=rescue_kwargs,
                max_rescue_rounds=DEFAULT_RESCUE_MAX_ROUNDS,
            )
            _save_rescue_artifacts(
                rel_dir,
                window,
                wf,
                consolidated=consolidated,
                initial_fit=initial_fit,
                center_mhz=center_mhz,
                freqs_sorted=freqs_sorted,
                spec_sorted=spec_sorted,
                rms_sorted=rms_sorted,
                sideband=sideband,
                acquisition_us=acquisition_us,
            )
            # Consolidated final-fit detail (Figure 1). Built on the rescue
            # outcome so the model shown is post-rescue, post-knockout.
            other_window_fits = [
                wf_other for wf_other in fit.window_fits
                if wf_other.window_id != window_id
            ]
            _save_consolidated_detail(
                rel_dir,
                window,
                consolidated=consolidated,
                original_wf=wf,
                other_window_fits=other_window_fits,
                center_mhz=center_mhz,
                freqs_sorted=freqs_sorted,
                spec_sorted=spec_sorted,
                rms_sorted=rms_sorted,
                sideband=sideband,
                acquisition_us=acquisition_us,
                style=display_style,
            )
            _save_audit_trail_figure_wrapper(
                rel_dir,
                window,
                consolidated=consolidated,
                original_wf=wf,
                center_mhz=center_mhz,
                freqs_sorted=freqs_sorted,
                spec_sorted=spec_sorted,
                rms_sorted=rms_sorted,
                sideband=sideband,
                acquisition_us=acquisition_us,
                style=display_style,
            )
        except Exception as exc:  # noqa: BLE001
            (rel_dir / "report-rr.md").write_text(
                f"# Window {window_id} - Rescue FAILED\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```\n"
            )
        _save_window_artifacts(
            rel_dir,
            window,
            wf,
            peaks_loaded,
            frequencies=freqs_sorted,
            complex_spectrum=spec_sorted,
            rms_noise=rms_sorted,
            fit=fit,
            sideband=sideband,
            acquisition_us=acquisition_us,
            note=note,
        )
        return rel_dir

    easy_entries: List[Tuple[int, FitWindow, FittingResult, Optional[str], Path]] = []
    hard_entries: List[Tuple[int, FitWindow, FittingResult, Optional[str], Path]] = []
    named_entries: List[Tuple[int, FitWindow, FittingResult, Optional[str], Path]] = []

    if args.window_ids:
        # --window-id overrides the deliberate samples. Bucket the
        # requested ids by their difficulty in the plan so the INDEX.md
        # categories still make sense; named-case annotations come from
        # --named-window when provided, otherwise fall back to the
        # built-in NAMED_SAMPLE.
        named_lookup = dict(cli_named) if cli_named else dict(NAMED_SAMPLE)
        for wid in args.window_ids:
            if wid not in plan_by_id:
                raise SystemExit(f"window_id {wid} not in plan")
            if wid not in fit_by_id:
                raise SystemExit(f"window_id {wid} not in persisted fit")
            note = named_lookup.get(wid)
            rel = emit(wid, note)
            entry = (wid, plan_by_id[wid], fit_by_id[wid], note, rel)
            if note is not None:
                named_entries.append(entry)
            elif plan_by_id[wid].difficulty == WindowDifficulty.HARD:
                hard_entries.append(entry)
            else:
                easy_entries.append(entry)
    else:
        for wid in EASY_SAMPLE:
            rel = emit(wid, None)
            easy_entries.append(
                (wid, plan_by_id[wid], fit_by_id[wid], None, rel)
            )
        for wid in HARD_SAMPLE:
            rel = emit(wid, None)
            hard_entries.append(
                (wid, plan_by_id[wid], fit_by_id[wid], None, rel)
            )
        for wid, note in NAMED_SAMPLE:
            rel = emit(wid, note)
            named_entries.append(
                (wid, plan_by_id[wid], fit_by_id[wid], note, rel)
            )

    # --- INDEX.md ----------------------------------------------------------
    n_thaw = len(fit.thaw_history)
    n_thaw_ok = sum(1 for e in fit.thaw_history if e.accepted)
    n_replan = len(fit.replan_history)
    n_replan_ok = sum(1 for e in fit.replan_history if e.accepted)
    n_easy = sum(1 for w in plan.windows if w.difficulty == WindowDifficulty.EASY)
    n_hard = sum(1 for w in plan.windows if w.difficulty == WindowDifficulty.HARD)

    body = [
        "# 2638 Stage 5 validation",
        "",
        "Initial deliberate sample for visual inspection.",
        "",
        "## Run summary",
        f"- windows: **{fit.n_windows}** ({n_easy} easy / {n_hard} hard)",
        f"- fitted peaks: **{fit.n_fitted_peaks}**",
        f"- final plan revision: {fit.final_plan_revision} _(structural replans accepted: {n_replan_ok}/{n_replan})_",
        f"- thaw attempts: **{n_thaw_ok}/{n_thaw} accepted**",
        "- parameters used:",
    ]
    for k, v in sorted((fit.parameters or {}).items()):
        body.append(f"    - `{k}`: {v}")
    body += [
        "",
        f"![overview](overview.png)",
        "",
    ]
    body.append(_category_block("Easy windows", easy_entries))
    body.append(_category_block("Hard windows", hard_entries))
    body.append(_category_block("Named cases", named_entries))
    (OUTPUT_DIR / "INDEX.md").write_text("\n".join(body))

    print(f"overview: {OUTPUT_DIR / 'overview.png'}")
    print(f"index:    {OUTPUT_DIR / 'INDEX.md'}")
    for entries in (easy_entries, hard_entries, named_entries):
        for wid, _w, _wf, _note, rel in entries:
            print(f"  window {wid:>3} -> {rel}/")


if __name__ == "__main__":
    main()
