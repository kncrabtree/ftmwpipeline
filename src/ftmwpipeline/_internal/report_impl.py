"""Shared implementation for the Stage 6 ``report`` object.

Reports render the persisted Stage 6 record; they never recompute the fit.

Level 1 (``report table``, data export): the consolidated :class:`FinalProducts`
table serialized to CSV, JSON, or a LaTeX ``booktabs`` table for paper SI. All
three are flat serializations of the same persisted table -- *assemble once,
render many* -- so they stay consistent by construction.

This module also holds the **static, code-versioned algorithm prose** -- a
methods section that lives here so it stays in sync with the code, not pulled
from the planning docs -- interleaved with the per-experiment numbers read from
each persisted stage. :func:`_render_markdown` builds that document; the Level-3
HTML report (``report run``) folds it into its methods page.

Robustness: every numeric field is rendered through width-bounded, non-finite
guarded formatters, so a degenerate fit (e.g. an amplitude-collapsed phantom
with a runaway uncertainty) never dumps a hundred-digit number into the table.
Amplitudes are reported in a dynamically chosen SI unit (V/mV/uV/nV/...) so the
magnitudes read sensibly, and amplitude/phase/SNR carry their uncertainties.
LaTeX uses concise value(uncertainty) notation. See
``dev-docs/planning/stage6-reports.md`` §C.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from ..core.data_structures import FinalPeak, FinalProducts, SpectrumFit
from ..fitting.validation import (
    DEFAULT_CHI2R_NOISE_FLOOR,
    DEFAULT_SHAPE_ERROR_KAPPA,
    shape_error_fraction,
    snr_aware_chi2_pass,
)
from ..io.peak_serialization import load_peaks_from_hdf5
from ..io.stage6_review_serialization import load_stage6_review_from_file
from .catalog_xref import CatalogCrossRef, CatalogMatch, load_cross_ref

VALID_FORMATS = ("csv", "json", "latex")

# SI amplitude units, descending by multiplier (base value in volts).
_AMP_UNITS: List[Tuple[str, float]] = [
    ("V", 1.0),
    ("mV", 1e-3),
    ("uV", 1e-6),
    ("nV", 1e-9),
    ("pV", 1e-12),
    ("fV", 1e-15),
]


# ---------------------------------------------------------------------------
# Width-bounded, non-finite-guarded number formatting
# ---------------------------------------------------------------------------


def _nonfinite(x: float) -> Optional[str]:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return None


def _g(x: Optional[float], sig: int = 4) -> str:
    """General bounded format (``%g``): scientific for huge/tiny, never runaway."""
    if x is None:
        return ""
    xf = float(x)
    nf = _nonfinite(xf)
    if nf is not None:
        return nf
    if xf == 0.0:
        return "0"
    return f"{xf:.{sig}g}"


def _freq(x: Optional[float]) -> str:
    """Fixed 6-dp MHz (≈1 Hz), guarded against non-finite / runaway values."""
    if x is None:
        return ""
    xf = float(x)
    nf = _nonfinite(xf)
    if nf is not None:
        return nf
    if abs(xf) >= 1e9:  # a sane frequency is never this large -- bound the width
        return f"{xf:.6g}"
    return f"{xf:.6f}"


def _scaled(x: Optional[float], unit_value: float, sig: int = 4) -> str:
    """Format ``x / unit_value`` (amplitude in the chosen SI unit)."""
    if x is None:
        return ""
    return _g(float(x) / unit_value, sig)


def _concise(value: float, sigma: Optional[float], fallback_sig: int = 8) -> str:
    """Concise ``value(uncertainty)`` notation (uncertainty in last-digit units).

    Rounds the uncertainty to two significant figures and the value to the same
    decimal place, e.g. ``26613.6131(15)`` (= 26613.6131 ± 0.0015). Falls back
    to a plain bounded format when the uncertainty is missing / non-positive,
    and to a compact scientific pair when either value or uncertainty is
    degenerate (non-finite or astronomically large), so it never runs away.
    """
    v = float(value)
    nf = _nonfinite(v)
    if nf is not None:
        return nf
    if sigma is None:
        return f"{v:.{fallback_sig}g}"
    s = float(sigma)
    if _nonfinite(s) is not None or s <= 0.0:
        return f"{v:.{fallback_sig}g}"

    exp = math.floor(math.log10(s))
    if exp > 5 or exp < -12:  # degenerate uncertainty -- compact, bounded pair
        return f"{_g(v, 6)}({_g(s, 2)})"

    nsig = 2
    last = exp - (nsig - 1)
    factor = 10.0**last
    unc = int(round(s / factor))
    if unc >= 10**nsig:  # rounding carry (e.g. 99.6 -> 100)
        unc //= 10
        last += 1
        factor = 10.0**last
    vr = round(v / factor) * factor
    if last < 0:
        return f"{vr:.{-last}f}({unc})"
    return f"{int(round(vr))}({int(round(s))})"


# ---------------------------------------------------------------------------
# Amplitude unit selection (outlier-robust)
# ---------------------------------------------------------------------------


def _amplitude_unit(products: FinalProducts) -> Tuple[str, float]:
    """Pick an SI amplitude unit so the smallest real amplitude reads sensibly.

    Targets the smallest *real* amplitude into ``>= 0.01`` in the chosen unit
    (the largest prefix that keeps it so). Phantom/degenerate amplitudes far
    below the strongest line (``< 1e-4 * max``) are excluded so a single
    collapsed peak cannot poison the unit for the whole table.
    """
    amps = [
        abs(float(p.amplitude))
        for p in products.peaks
        if p.amplitude is not None
        and math.isfinite(float(p.amplitude))
        and float(p.amplitude) != 0.0
    ]
    if not amps:
        return ("V", 1.0)
    a_max = max(amps)
    floor = a_max * 1e-4
    real = [a for a in amps if a >= floor] or amps
    a_min = min(real)
    target = a_min * 100.0  # unit <= target keeps a_min/unit >= 0.01
    for name, val in _AMP_UNITS:  # descending value: first <= target is largest
        if val <= target:
            return (name, val)
    return _AMP_UNITS[-1]


def _unit_latex(name: str) -> str:
    return r"$\mu$V" if name == "uV" else name


# ---------------------------------------------------------------------------
# Provenance header
# ---------------------------------------------------------------------------


def _provenance(
    products: FinalProducts,
    file_path: Union[Path, str],
    amp_unit: str,
    xref: Optional[CatalogCrossRef] = None,
) -> List[Tuple[str, str]]:
    out = [
        ("experiment", Path(file_path).stem),
        ("calibration_state", products.calibration_state),
        (
            "epsilon_ppm",
            f"{products.epsilon * 1e6:+.3f} +- {products.sigma_epsilon * 1e6:.3f}",
        ),
        ("sigma_floor_khz", f"{products.sigma_floor_khz:.3f}"),
        ("probe_freq_mhz", f"{products.probe_freq_mhz:.4f}"),
        ("sideband", products.sideband),
        ("amplitude_unit", amp_unit),
        ("n_peaks", str(len(products.peaks))),
    ]
    if xref is not None:
        out += [
            ("catalog", Path(xref.catalog_path).name),
            ("catalog_n_sigma", f"{xref.n_sigma:g}"),
            (
                "catalog_matched",
                f"{xref.n_matched}/{xref.n_total} ({xref.match_rate * 100:.0f}%)",
            ),
        ]
        if xref.pull_mean is not None:
            std = "n/a" if xref.pull_std is None else f"{xref.pull_std:.2f}"
            out.append(("catalog_pull_mean_std", f"{xref.pull_mean:.2f} / {std}"))
    return out


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "frequency_mhz",
    "sigma_f_khz",
    "sigma_stat_khz",
    "sigma_eps_khz",
    "sigma_floor_khz",
    "frequency_raw_mhz",
    "f_baseband_mhz",
    "amplitude",
    "amplitude_err",
    "phase_rad",
    "phase_err_rad",
    "snr",
    "snr_err",
    "origin",
    "window_id",
]


# Proximity-annotation columns appended when a ``--catalog`` is supplied.
_CATALOG_COLUMNS = [
    "catalog_label",
    "catalog_freq_mhz",
    "catalog_delta_khz",
    "catalog_pull",
]


def _catalog_csv_cells(m: Optional[CatalogMatch]) -> List[str]:
    if m is None:
        return ["", "", "", ""]
    return [m.label, _freq(m.frequency_mhz), _g(m.delta_khz), _g(m.pull)]


def _csv_row(p: FinalPeak, unit_value: float) -> List[str]:
    return [
        _freq(p.frequency_mhz),
        _g(p.sigma_f_khz),
        _g(p.sigma_stat_khz),
        _g(p.sigma_eps_khz),
        _g(p.sigma_floor_khz),
        _freq(p.frequency_raw_mhz),
        _freq(p.f_baseband_mhz),
        _scaled(p.amplitude, unit_value),
        _scaled(p.amplitude_error, unit_value),
        _g(p.phase),
        _g(p.phase_error),
        _g(p.snr),
        _g(p.snr_error),
        p.origin,
        "" if p.window_id is None else str(p.window_id),
    ]


def _render_csv(
    products: FinalProducts,
    file_path: Union[Path, str],
    xref: Optional[CatalogCrossRef] = None,
) -> str:
    uname, uval = _amplitude_unit(products)
    comments = ["# ftmwpipeline final products"]
    comments += [
        f"# {k}: {v}" for k, v in _provenance(products, file_path, uname, xref)
    ]
    cols = list(_CSV_COLUMNS) + (_CATALOG_COLUMNS if xref is not None else [])
    # The stdlib writer quotes any free-text cell (a catalog label or origin) that
    # carries a comma / quote / newline, so column alignment survives such values.
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for i, p in enumerate(products.peaks):
        row = _csv_row(p, uval)
        if xref is not None:
            row = row + _catalog_csv_cells(xref.matches[i])
        writer.writerow(row)
    return "\n".join(comments) + "\n" + buf.getvalue()


def _jnum(x: Optional[float]) -> Optional[float]:
    """JSON-safe float: non-finite -> null (so the JSON stays strict-valid)."""
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


def _catalog_json(m: Optional[CatalogMatch]) -> Optional[dict]:
    if m is None:
        return None
    return {
        "label": m.label,
        "frequency_mhz": _jnum(m.frequency_mhz),
        "sigma_cat_khz": _jnum(m.sigma_cat_khz),
        "delta_khz": _jnum(m.delta_khz),
        "pull": _jnum(m.pull),
    }


def _peak_json(
    p: FinalPeak,
    unit_value: float,
    match: Optional[CatalogMatch] = None,
    *,
    with_catalog: bool = False,
) -> dict:
    payload: Dict[str, Any] = {
        "frequency_mhz": _jnum(p.frequency_mhz),
        "sigma_f_khz": _jnum(p.sigma_f_khz),
        "sigma_stat_khz": _jnum(p.sigma_stat_khz),
        "sigma_eps_khz": _jnum(p.sigma_eps_khz),
        "sigma_floor_khz": _jnum(p.sigma_floor_khz),
        "frequency_raw_mhz": _jnum(p.frequency_raw_mhz),
        "f_baseband_mhz": _jnum(p.f_baseband_mhz),
        "amplitude": _jnum(None if p.amplitude is None else p.amplitude / unit_value),
        "amplitude_error": _jnum(
            None if p.amplitude_error is None else p.amplitude_error / unit_value
        ),
        "phase_rad": _jnum(p.phase),
        "phase_err_rad": _jnum(p.phase_error),
        "snr": _jnum(p.snr),
        "snr_error": _jnum(p.snr_error),
        "origin": p.origin,
        "window_id": p.window_id,
    }
    if with_catalog:
        payload["catalog"] = _catalog_json(match)
    return payload


def _catalog_metadata(xref: CatalogCrossRef) -> dict:
    return {
        "path": xref.catalog_path,
        "n_sigma": xref.n_sigma,
        "n_catalog": xref.n_catalog,
        "n_matched": xref.n_matched,
        "n_total": xref.n_total,
        "match_rate": _jnum(xref.match_rate),
        "pull_mean": _jnum(xref.pull_mean),
        "pull_std": _jnum(xref.pull_std),
    }


def _render_json(
    products: FinalProducts,
    file_path: Union[Path, str],
    xref: Optional[CatalogCrossRef] = None,
) -> str:
    uname, uval = _amplitude_unit(products)
    metadata: Dict[str, Any] = {
        "experiment": Path(file_path).stem,
        "calibration_state": products.calibration_state,
        "epsilon": _jnum(products.epsilon),
        "sigma_epsilon": _jnum(products.sigma_epsilon),
        "sigma_floor_khz": _jnum(products.sigma_floor_khz),
        "probe_freq_mhz": _jnum(products.probe_freq_mhz),
        "sideband": products.sideband,
        "amplitude_unit": uname,
        "n_peaks": len(products.peaks),
    }
    if xref is not None:
        metadata["catalog"] = _catalog_metadata(xref)
        peaks = [
            _peak_json(p, uval, xref.matches[i], with_catalog=True)
            for i, p in enumerate(products.peaks)
        ]
    else:
        peaks = [_peak_json(p, uval) for p in products.peaks]
    return json.dumps({"metadata": metadata, "peaks": peaks}, indent=2) + "\n"


def _latex_caption(
    products: FinalProducts,
    amp_unit: str,
    xref: Optional[CatalogCrossRef] = None,
) -> str:
    state = products.calibration_state
    if state == "self_calibrated":
        freq = (
            "Frequencies corrected for the free-running digitizer timebase scale "
            f"error ($\\epsilon = {products.epsilon * 1e6:+.3f}$~ppm)"
        )
    elif state == "rb_locked":
        freq = "Frequencies on the instrument's Rb-locked absolute scale"
    else:
        freq = (
            "Frequencies uncalibrated (free-running digitizer, no timebase "
            "self-calibration)"
        )
    cat = ""
    if xref is not None:
        cat = (
            f" The catalog column echoes the nearest entry within "
            f"${xref.n_sigma:g}\\sigma$ (proximity only, not an assignment); "
            f"{xref.n_matched}/{xref.n_total} lines matched."
        )
    return (
        f"Fitted line list. {freq}; $\\sigma_f$ is the reported precision budget. "
        f"Uncertainties are in units of the last digit; amplitude in "
        f"{_unit_latex(amp_unit)}.{cat}"
    )


def _latex_escape(text: str) -> str:
    """Escape the LaTeX specials that can appear in an opaque catalog label."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _render_latex(
    products: FinalProducts,
    file_path: Union[Path, str],
    xref: Optional[CatalogCrossRef] = None,
) -> str:
    uname, uval = _amplitude_unit(products)
    cols = [
        (
            "Frequency (MHz)",
            lambda p, m: _concise(p.frequency_mhz, p.sigma_f_khz * 1e-3),
        ),
        (
            f"Amplitude ({_unit_latex(uname)})",
            lambda p, m: _concise(
                p.amplitude / uval,
                None if p.amplitude_error is None else p.amplitude_error / uval,
            ),
        ),
        (
            "SNR",
            lambda p, m: "--" if p.snr is None else _concise(p.snr, p.snr_error),
        ),
    ]
    if xref is not None:
        # Proximity annotation: echo the opaque label and the offset (kHz). The
        # column header marks the geometric tolerance; the label is never an
        # assignment.
        cols.append(
            (
                r"Catalog ($\Delta$/kHz)",
                lambda p, m: (
                    "--"
                    if m is None
                    else f"{_latex_escape(m.label)} ({_g(m.delta_khz, 2)})"
                ),
            )
        )
    spec = "r" * len(cols)
    out = ["% ftmwpipeline final products -- requires \\usepackage{booktabs}"]
    out += [f"% {k}: {v}" for k, v in _provenance(products, file_path, uname, xref)]
    out.append(r"\begin{table}")
    out.append(r"  \centering")
    out.append(r"  \caption{" + _latex_caption(products, uname, xref) + r"}")
    out.append(r"  \begin{tabular}{" + spec + "}")
    out.append(r"    \toprule")
    out.append("    " + " & ".join(h for h, _ in cols) + r" \\")
    out.append(r"    \midrule")
    matches = xref.matches if xref is not None else [None] * len(products.peaks)
    for p, m in zip(products.peaks, matches):
        out.append("    " + " & ".join(fmt(p, m) for _, fmt in cols) + r" \\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


_RENDERERS = {"csv": _render_csv, "json": _render_json, "latex": _render_latex}


def report_table_impl(
    file_path: Union[Path, str],
    *,
    fmt: str = "csv",
    output: Optional[Union[Path, str]] = None,
    catalog: Optional[Union[Path, str]] = None,
    catalog_n_sigma: float = 3.0,
) -> str:
    """Render the persisted Level-1 final-products table to *fmt* and return it.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    fmt :
        One of ``"csv"`` / ``"json"`` / ``"latex"`` (default ``"csv"``).
    output :
        When given, also write the rendered text to this path.
    catalog :
        Optional path to a frequency catalog (CSV: ``frequency_mhz`` plus an
        optional uncertainty in kHz and an optional opaque label). When given,
        each line is proximity-flagged against the nearest catalog entry within
        ``catalog_n_sigma * sqrt(sigma_f^2 + sigma_cat^2)`` and the match
        (label / frequency / offset / pull) is added to the output. This is a
        cross-check echo of the catalog's opaque label, **never an assignment**;
        it never alters the fit.
    catalog_n_sigma :
        Match tolerance in combined sigmas (default ``3``).

    Returns
    -------
    str
        The rendered table.

    Raises
    ------
    ValueError
        If *fmt* is unknown, no final-products table is present (the Stage 6
        ``review run`` consolidation has not been run), or *catalog* is given
        but unreadable / empty.
    """
    key = str(fmt).lower()
    if key not in _RENDERERS:
        raise ValueError(
            f"unknown report format {fmt!r}; choose one of {VALID_FORMATS}"
        )

    products = load_stage6_review_from_file(str(file_path)).final_products
    if products is None:
        raise ValueError(
            "No final-products table found in this file. Run 'review run' first "
            "to consolidate the calibrated final products."
        )

    xref = load_cross_ref(products.peaks, catalog, catalog_n_sigma)
    text = _RENDERERS[key](products, file_path, xref)
    if output is not None:
        Path(output).write_text(text)
    return text


# ===========================================================================
# Level 2 -- methods + results document (Markdown)
# ===========================================================================
#
# Assembled once from the persisted record, then rendered. Each stage section
# carries four things: static, code-versioned algorithm prose; a table of the
# key parameters the stage actually ran with (read from the persisted
# settings); the governing equation(s); the detailed per-experiment results
# (including per-band breakdowns); and an automatically flagged list of
# concerns with recommendations. The prose / equations / parameter meanings are
# static (timeless); the numbers and concern flags are computed from the record.


@dataclass
class _BandTau:
    """Per-band tau majority (Stage 2b ``band_majorities`` entry)."""

    label: str
    lo_mhz: float
    hi_mhz: float
    n: int
    tau_us: float
    sigma_tau_us: float


@dataclass
class _NoiseBand:
    """Per-band Stage 2 noise summary over the trimmed active grid."""

    lo_mhz: float
    hi_mhz: float
    median_sigma: float
    noise_fraction: float
    n_bins: int


@dataclass
class _Concern:
    """One flagged metric with its recommendation.

    ``severity`` is ``"warning"`` (likely affects results) or ``"note"``
    (worth knowing, usually benign / by-design).
    """

    severity: str
    message: str
    recommendation: str


@dataclass
class _SummaryModel:
    """Per-experiment numbers gathered from the persisted stages for L2.

    Optional fields are ``None`` / empty when the originating stage is absent
    (Stage 2b is an optional dependency; the rest are present whenever
    ``FinalProducts`` exists, since ``review run`` requires Stages 0--5).
    """

    # Provenance
    experiment: str
    source_path: str
    source_format: str
    # Stage 0 -- FID / start time
    probe_freq_mhz: float
    sideband: str
    start_us: Optional[float]
    end_us: Optional[float]
    duration_us: float
    n_points: int
    shots: Optional[int]
    # Stage 1 -- FT band
    band_lo_mhz: Optional[float]
    band_hi_mhz: Optional[float]
    ft_bin_khz: Optional[float]
    ft_n_bins: Optional[int]
    # Stage 2 -- noise
    noise_median: Optional[float]
    noise_min: Optional[float]
    noise_max: Optional[float]
    # Stage 2b -- tau (optional)
    tau_maj_us: Optional[float]
    sigma_tau_us: Optional[float]
    n_contributors: Optional[int]
    band_taus: List[_BandTau] = field(default_factory=list)
    # Stage 3 -- peak detection
    n_peaks_total: int = 0
    n_strong: int = 0
    n_medium: int = 0
    n_weak: int = 0
    promotion_min_snr: Optional[float] = None
    weak_medium_snr: Optional[float] = None
    medium_strong_snr: Optional[float] = None
    # Stage 4 -- window assignment
    n_windows_planned: int = 0
    # Stage 5 -- fitting
    shape: Optional[str] = None
    n_windows_fit: int = 0
    n_fitted_peaks: int = 0
    chi2_median: Optional[float] = None
    chi2_min: Optional[float] = None
    chi2_max: Optional[float] = None
    n_thaw: int = 0
    n_thaw_accepted: int = 0
    n_replan: int = 0
    n_rescue_rounds: int = 0
    # Stage 6 -- final products (carries state / eps / floor / peaks)
    products: Optional[FinalProducts] = None

    # --- enriched detail (all default-empty so a hand-built model is valid) --
    # Stage 2
    noise_params: Dict[str, Any] = field(default_factory=dict)
    noise_fraction_overall: Optional[float] = None
    noise_bands: List[_NoiseBand] = field(default_factory=list)
    # Stage 2b
    tau_params: Dict[str, Any] = field(default_factory=dict)
    tau_preconditions_passed: Optional[bool] = None
    tau_preconditions_notes: List[str] = field(default_factory=list)
    tau_bimodal: Optional[bool] = None
    tau_delta_aic: Optional[float] = None
    tau_dominant_weight: Optional[float] = None
    tau_mu_a: Optional[float] = None
    tau_mu_b: Optional[float] = None
    tau_pearson_freq: Optional[float] = None
    recommended_shape: Optional[str] = None
    shape_vote_rates: Dict[str, float] = field(default_factory=dict)
    tau_exp_present: bool = False
    tau_G_present: bool = False
    # Stage 3
    det_params: Dict[str, Any] = field(default_factory=dict)
    det_pass_counts: Dict[str, int] = field(default_factory=dict)
    n_promoted: int = 0
    snr_pctiles: Dict[str, float] = field(default_factory=dict)
    snr_pctiles_promoted: Dict[str, float] = field(default_factory=dict)
    density_chunks: List[Tuple[float, float, int, int]] = field(default_factory=list)
    # Stage 4
    window_params: Dict[str, Any] = field(default_factory=dict)
    width_min_mhz: Optional[float] = None
    width_median_mhz: Optional[float] = None
    width_max_mhz: Optional[float] = None
    width_pctiles: Dict[str, float] = field(default_factory=dict)
    ppw_median: Optional[float] = None
    ppw_max: Optional[int] = None
    ppw_pctiles: Dict[str, float] = field(default_factory=dict)
    # Stage 5
    fit_params: Dict[str, Any] = field(default_factory=dict)
    tau0_us: Optional[float] = None
    tau_calibration_source: Optional[str] = None
    n_nonconverged: int = 0
    n_spurs_gated: int = 0
    n_empty_dropped: int = 0
    n_pruned: int = 0
    n_baseline_windows: int = 0
    n_windows_gate_checked: int = 0
    n_windows_fail_gate: int = 0
    worst_windows: List[Tuple[int, float, float, float]] = field(default_factory=list)
    chi2r_pctiles: Dict[str, float] = field(default_factory=dict)
    eps_pctiles: Dict[str, float] = field(default_factory=dict)
    sigma_stat_pctiles: Dict[str, float] = field(default_factory=dict)
    snr_bin_rows: List[Tuple[str, int, float, float, float, float, float]] = field(
        default_factory=list
    )
    # Stage 6
    timebase_n_used: Optional[int] = None
    timebase_lattice_g: Optional[float] = None
    median_sigma_stat: Optional[float] = None
    median_sigma_eps: Optional[float] = None
    median_sigma_f: Optional[float] = None
    sigma_eps_pctiles: Dict[str, float] = field(default_factory=dict)
    sigma_f_pctiles: Dict[str, float] = field(default_factory=dict)
    n_stat_dominated: int = 0
    n_eps_dominated: int = 0
    # Raw distributions backing the percentile tables (for histogram rendering;
    # not used by the Markdown report, which shows percentiles only).
    snr_values_promoted: List[float] = field(default_factory=list)
    chi2r_values: List[float] = field(default_factory=list)
    eps_values: List[float] = field(default_factory=list)
    sigma_stat_values: List[float] = field(default_factory=list)
    sigma_eps_values: List[float] = field(default_factory=list)
    sigma_f_values: List[float] = field(default_factory=list)


def _h5_attr_json(group: h5py.Group, key: str) -> dict:
    """Parse a JSON ``parameters`` attr off *group*, tolerating absence."""
    raw = group.attrs.get(key)
    if raw is None:
        return {}
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError):
        return {}


def _noise_band_table(
    file_path: str, trim: Optional[Tuple[float, float]], n_bands: int = 8
) -> Tuple[Optional[float], List[_NoiseBand]]:
    """Per-band median sigma_x and noise fraction over the trimmed active grid.

    Replays the canonical trimmed active FT (a deterministic rfft of the
    persisted active record) and reads the persisted Stage 2 sigma back
    element-for-element -- the sanctioned noise read path, not a recompute of
    the estimate. Returns ``(overall_noise_fraction, bands)``; ``(None, [])``
    when the active grid / sigma cannot be aligned.
    """
    from ..io.noise_result_serialization import load_noise_result_from_hdf5
    from .active_ft_support import build_trimmed_active_ft

    try:
        cft = build_trimmed_active_ft(file_path, trim)
        freqs = np.asarray(cft.freq_array, dtype=float)
        mags = np.asarray(cft.magnitude_spectrum, dtype=float)
        with h5py.File(file_path, "r") as h5f:
            noise = load_noise_result_from_hdf5(h5f["stage2_noise_result"], freqs, mags)
        sigma = np.asarray(noise.rms_noise, dtype=float)
        mask = np.asarray(noise.noise_mask, dtype=bool)
    except (KeyError, ValueError, OSError):
        return None, []
    if freqs.size == 0 or sigma.size != freqs.size or mask.size != freqs.size:
        return None, []

    order = np.argsort(freqs)
    freqs, sigma, mask = freqs[order], sigma[order], mask[order]
    overall = float(mask.mean())
    lo, hi = float(freqs[0]), float(freqs[-1])
    edges = np.linspace(lo, hi, n_bands + 1)
    bands: List[_NoiseBand] = []
    for i in range(n_bands):
        sel = (freqs >= edges[i]) & (
            (freqs < edges[i + 1]) if i < n_bands - 1 else (freqs <= edges[i + 1])
        )
        if not sel.any():
            continue
        finite = sigma[sel][np.isfinite(sigma[sel])]
        bands.append(
            _NoiseBand(
                lo_mhz=float(edges[i]),
                hi_mhz=float(edges[i + 1]),
                median_sigma=float(np.median(finite)) if finite.size else float("nan"),
                noise_fraction=float(mask[sel].mean()),
                n_bins=int(sel.sum()),
            )
        )
    return overall, bands


_NONE_SENTINEL = "__None__"

# The 3-way discriminator votes over exp/gauss/voigt models; the report names
# the exponential model by its line shape (Lorentzian) for reader continuity
# with ``recommended_shape``.
_VOTE_DISPLAY_SHAPE = {"exp": "lorentzian", "gauss": "gaussian", "voigt": "voigt"}


def _decode_recommended_shape(attr: Any) -> Optional[str]:
    """Decode the persisted ``recommended_shape`` attr, mapping the sentinel."""
    if attr is None:
        return None
    decoded = attr.decode() if isinstance(attr, bytes) else str(attr)
    return None if decoded == _NONE_SENTINEL else decoded


def _decode_vote_rates(attr: Any) -> Dict[str, float]:
    """Decode the JSON shape-vote attr into display-named fractions."""
    if attr is None:
        return {}
    decoded = attr.decode() if isinstance(attr, bytes) else str(attr)
    try:
        rates = json.loads(decoded)
    except (ValueError, TypeError):
        return {}
    return {_VOTE_DISPLAY_SHAPE.get(str(k), str(k)): float(v) for k, v in rates.items()}


def _assemble_summary(file_path: Union[Path, str]) -> _SummaryModel:
    """Gather the per-stage numbers for the L2 report from the persisted file.

    Reads each stage's persisted group directly (never recomputes the fit;
    the only replay is the deterministic active-FT rebuild for the per-band
    noise table). Requires a built :class:`FinalProducts` (i.e. ``review run``
    has consolidated the calibrated table); raises :class:`ValueError`
    otherwise.
    """
    from collections import Counter

    from ..io.fid_serialization import load_fid_from_hdf5
    from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
    from ..io.tau_calibration_serialization import load_tau_calibration_from_hdf5
    from ..io.timebase_serialization import load_timebase_calibration_from_hdf5
    from ..io.window_serialization import load_window_plan_from_hdf5

    path = str(file_path)
    review = load_stage6_review_from_file(path)
    products = review.final_products
    if products is None:
        raise ValueError(
            "No final-products table found in this file. Run 'review run' first "
            "to consolidate the calibrated final products."
        )

    timebase_n_used: Optional[int] = None
    timebase_lattice_g: Optional[float] = None

    with h5py.File(path, "r") as h5f:
        # --- provenance -----------------------------------------------------
        src = h5f.get("source_metadata")
        source_path = ""
        source_format = ""
        if src is not None:
            source_path = str(src.attrs.get("source_path", ""))
            source_format = str(src.attrs.get("format_name", ""))

        # --- Stage 0: FID / start time -------------------------------------
        fid = load_fid_from_hdf5(h5f["stage0_fid_data"])
        proc = fid.processing
        sideband = (
            fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
        )

        # --- Stage 1: FT band (display knobs persisted under ft_processing) -
        band_lo: Optional[float] = None
        band_hi: Optional[float] = None
        ft_processing = h5f.get("processing_parameters/ft_processing")
        if ft_processing is not None:
            lo = ft_processing.attrs.get("trim_min_mhz")
            hi = ft_processing.attrs.get("trim_max_mhz")
            band_lo = None if lo is None else float(lo)
            band_hi = None if hi is None else float(hi)
        ft_bin_khz = 1.0e3 / fid.duration_us if fid.duration_us else None
        ft_n_bins = (
            int(round((band_hi - band_lo) / (ft_bin_khz * 1e-3)))
            if band_lo is not None and band_hi is not None and ft_bin_khz
            else None
        )

        # --- Stage 2: noise (verbatim sigma_x array stored on the group) ----
        noise_median = noise_min = noise_max = None
        noise_params: Dict[str, Any] = {}
        noise_grp = h5f.get("stage2_noise_result")
        if noise_grp is not None and "rms_noise_full" in noise_grp:
            sigma_full = np.asarray(noise_grp["rms_noise_full"][:], dtype=float)
            sigma_full = sigma_full[np.isfinite(sigma_full)]
            if sigma_full.size:
                noise_median = float(np.median(sigma_full))
                noise_min = float(sigma_full.min())
                noise_max = float(sigma_full.max())
            bi = noise_grp.get("bin_info")
            if bi is not None:
                noise_params = {k: bi.attrs[k] for k in bi.attrs.keys()}

        # --- Stage 2b: tau (optional dependency) ----------------------------
        tau_maj = sigma_tau = None
        n_contrib = None
        band_taus: List[_BandTau] = []
        tau_params: Dict[str, Any] = {}
        tau_pre_passed: Optional[bool] = None
        tau_pre_notes: List[str] = []
        tau_bimodal: Optional[bool] = None
        tau_delta_aic = tau_dom_w = tau_mu_a = tau_mu_b = tau_pear = None
        recommended_shape: Optional[str] = None
        shape_vote_rates: Dict[str, float] = {}
        if "stage2b_tau_calibration" in h5f:
            tau_grp = h5f["stage2b_tau_calibration"]
            rs_attr = tau_grp.attrs.get("recommended_shape")
            recommended_shape = _decode_recommended_shape(rs_attr)
            shape_vote_rates = _decode_vote_rates(tau_grp.attrs.get("shape_vote_rates"))
            tau = load_tau_calibration_from_hdf5(tau_grp)
            tau_maj = float(tau.tau_maj_us)
            sigma_tau = float(tau.sigma_tau_us)
            n_contrib = int(tau.n_contributors)
            tau_pre_passed = bool(tau.preconditions_passed)
            tau_pre_notes = [str(s) for s in (tau.preconditions_notes or [])]
            tau_pear = float(tau.pearson_r_freq_vs_tau)
            bm = tau.bimodality
            tau_bimodal = bool(getattr(bm, "two_component_preferred", False))
            tau_delta_aic = float(getattr(bm, "delta_aic", float("nan")))
            tau_dom_w = float(getattr(bm, "dominant_weight", float("nan")))
            tau_mu_a = float(getattr(bm, "mu_a", float("nan")))
            tau_mu_b = float(getattr(bm, "mu_b", float("nan")))
            tau_params = {
                "n_seg": tau.n_seg,
                "sample_dt_us": tau.sample_dt_us,
                "rss_gate_factor": tau.rss_gate_factor,
                "snr_weighted": tau.snr_weighted,
                "n_spur_bins": tau.n_spur_bins,
            }
            for bmaj in tau.band_majorities:
                band_taus.append(
                    _BandTau(
                        label=str(bmaj.label),
                        lo_mhz=float(bmaj.freq_lo_mhz),
                        hi_mhz=float(bmaj.freq_hi_mhz),
                        n=int(bmaj.n),
                        tau_us=float(bmaj.tau_maj_us),
                        sigma_tau_us=float(bmaj.sigma_tau_us),
                    )
                )

        # --- Stage 3: peak detection ---------------------------------------
        n_total = n_strong = n_medium = n_weak = 0
        det_pass_counts: Dict[str, int] = {}
        det_snrs: List[float] = []
        det_freqs: List[float] = []
        if "stage3_peaks" in h5f:
            peaks = load_peaks_from_hdf5(h5f["stage3_peaks"])
            n_total = len(peaks)
            passes: Counter = Counter()
            for pk in peaks:
                label = getattr(pk.classification, "value", str(pk.classification))
                if label == "strong":
                    n_strong += 1
                elif label == "medium":
                    n_medium += 1
                elif label == "weak":
                    n_weak += 1
                dp = (pk.properties or {}).get("detection_pass")
                if dp is not None:
                    passes[str(dp)] += 1
                snr_v = getattr(pk, "snr", None)
                if snr_v is not None and math.isfinite(float(snr_v)):
                    det_snrs.append(float(snr_v))
                    det_freqs.append(float(pk.frequency))
            det_pass_counts = dict(passes)
        det = _h5_attr_json(
            h5f.get("processing_parameters/peak_detection", h5f), "parameters"
        )

        # --- Stage 4: window assignment ------------------------------------
        n_windows_planned = 0
        window_params: Dict[str, Any] = {}
        widths: List[float] = []
        ppw: List[int] = []
        if "stage4_windows" in h5f:
            plan = load_window_plan_from_hdf5(h5f["stage4_windows"])
            n_windows_planned = plan.n_windows
            window_params = dict(plan.parameters)
            widths = [
                float(w.width_mhz)
                for w in plan.windows
                if getattr(w, "width_mhz", None) is not None
            ]
            ppw = [int(getattr(w, "n_free_peaks", 0)) for w in plan.windows]

        # Which Stage 2b twins exist (drives the Stage 5 tau-source diagnosis).
        tau_exp_present = "stage2b_tau_calibration" in h5f
        tau_G_present = "stage2b_tau_G_calibration" in h5f

        # --- Stage 5: fitting ----------------------------------------------
        fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

        # --- Stage 6: timebase (for parameter table / concerns) ------------
        if "timebase_calibration" in h5f:
            try:
                tb = load_timebase_calibration_from_hdf5(h5f["timebase_calibration"])
                timebase_n_used = int(tb.n_used)
                timebase_lattice_g = float(tb.lattice_g_mhz)
            except (KeyError, ValueError):
                pass

    # ---- Stage 3 derived: SNR percentiles + spectral density chunks ------
    promo = _opt_float(det.get("promotion_min_snr"))
    snr_pctiles = _percentiles(det_snrs)
    snr_pctiles_promoted = _percentiles(
        [s for s in det_snrs if promo is None or s >= promo]
    )
    n_promoted = (
        len(det_snrs) if promo is None else sum(1 for s in det_snrs if s >= promo)
    )
    density_chunks: List[Tuple[float, float, int, int]] = []
    if det_freqs and band_lo is not None and band_hi is not None:
        fa = np.asarray(det_freqs, dtype=float)
        sa = np.asarray(det_snrs, dtype=float)
        edges = np.linspace(band_lo, band_hi, 9)
        for i in range(8):
            sel = (fa >= edges[i]) & (
                (fa < edges[i + 1]) if i < 7 else (fa <= edges[i + 1])
            )
            n_all = int(sel.sum())
            n_pro = n_all if promo is None else int((sel & (sa >= promo)).sum())
            density_chunks.append((float(edges[i]), float(edges[i + 1]), n_all, n_pro))

    # ---- Stage 4 derived: width + peaks-per-window percentiles -----------
    width_pctiles = _percentiles(widths)
    ppw_pctiles = _percentiles([float(x) for x in ppw])
    width_min = float(min(widths)) if widths else None
    width_med = float(np.median(widths)) if widths else None
    width_max = float(max(widths)) if widths else None
    ppw_med = float(np.median(ppw)) if ppw else None
    ppw_max = int(max(ppw)) if ppw else None

    # ---- Stage 5 derived: chi2 distribution + SNR-aware gate -------------
    chi = [
        float(wf.reduced_chi2)
        for wf in fit.window_fits
        if wf.reduced_chi2 is not None and math.isfinite(float(wf.reduced_chi2))
    ]
    chi_median = float(np.median(chi)) if chi else None
    chi_min = min(chi) if chi else None
    chi_max = max(chi) if chi else None
    shape = str(fit.parameters.get("shape")) if fit.parameters.get("shape") else None
    n_nonconverged = sum(1 for wf in fit.window_fits if not wf.success)

    # Per-window SNR_max from the consolidated final lines, keyed by window.
    snr_by_window: Dict[int, float] = {}
    for p in products.peaks:
        if p.window_id is None or p.snr is None or not math.isfinite(float(p.snr)):
            continue
        snr_by_window[p.window_id] = max(
            snr_by_window.get(p.window_id, 0.0), float(p.snr)
        )
    # One row per gate-checked window: (wid, chi2r, snr_max, eps, passed, bin).
    gate_rows: List[Tuple[int, float, float, float, bool, str]] = []
    for wf in fit.window_fits:
        wid = wf.window_id
        c2 = wf.reduced_chi2
        snr_max = None if wid is None else snr_by_window.get(wid)
        if wid is None or c2 is None or not math.isfinite(float(c2)) or snr_max is None:
            continue
        eps = float(shape_error_fraction(float(c2), snr_max))
        passed = snr_aware_chi2_pass(float(c2), snr_max)
        gate_rows.append((int(wid), float(c2), snr_max, eps, passed, _snr_bin(snr_max)))
    n_gate_checked = len(gate_rows)
    fails = sorted(
        ((r[0], r[1], r[2], r[3]) for r in gate_rows if not r[4]),
        key=lambda t: t[3],
        reverse=True,
    )

    # Fit-quality distributions for the Stage 5 statistics tables.
    chi2r_pctiles = _percentiles([r[1] for r in gate_rows])
    eps_pctiles = _percentiles([100.0 * r[3] for r in gate_rows])  # percent
    snr_bin_rows: List[Tuple[str, int, float, float, float, float, float]] = []
    for label in _SNR_BIN_LABELS:
        brows = [r for r in gate_rows if r[5] == label]
        if not brows:
            continue
        c2s = np.asarray([r[1] for r in brows], dtype=float)
        epss = np.asarray([100.0 * r[3] for r in brows], dtype=float)
        snr_bin_rows.append(
            (
                label,
                len(brows),
                float(np.median(c2s)),
                float(np.percentile(c2s, 90)),
                float(c2s.max()),
                float(np.mean([1.0 if r[4] else 0.0 for r in brows])),
                float(np.median(epss)),
            )
        )

    diag = fit.diagnostics or {}
    n_spurs_gated = int(fit.parameters.get("n_spurs_gated", 0) or 0)
    n_baseline_windows = int(fit.parameters.get("n_baseline_windows", 0) or 0)
    n_empty_dropped = int((diag.get("window_cleanup") or {}).get("n_empty_dropped", 0))
    n_pruned = int((diag.get("peak_survival") or {}).get("n_pruned", 0))

    # ---- Stage 6 derived: sigma_f budget breakdown ----------------------
    s_stat = [
        float(p.sigma_stat_khz)
        for p in products.peaks
        if p.sigma_stat_khz is not None and math.isfinite(float(p.sigma_stat_khz))
    ]
    # The Stage 5 NLS frequency-precision distribution (statistical sigma).
    sigma_stat_pctiles = _percentiles(s_stat)
    s_eps = [
        float(p.sigma_eps_khz)
        for p in products.peaks
        if p.sigma_eps_khz is not None and math.isfinite(float(p.sigma_eps_khz))
    ]
    s_f = [
        float(p.sigma_f_khz)
        for p in products.peaks
        if p.sigma_f_khz is not None and math.isfinite(float(p.sigma_f_khz))
    ]
    n_stat_dom = sum(
        1
        for p in products.peaks
        if p.sigma_stat_khz is not None
        and p.sigma_eps_khz is not None
        and float(p.sigma_stat_khz) >= float(p.sigma_eps_khz)
    )
    n_eps_dom = len(products.peaks) - n_stat_dom

    return _SummaryModel(
        experiment=Path(path).stem,
        source_path=source_path,
        source_format=source_format,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        start_us=None if proc.start_us is None else float(proc.start_us),
        end_us=None if proc.end_us is None else float(proc.end_us),
        duration_us=float(fid.duration_us),
        n_points=int(fid.n_points),
        shots=None if fid.shots is None else int(fid.shots),
        band_lo_mhz=band_lo,
        band_hi_mhz=band_hi,
        ft_bin_khz=ft_bin_khz,
        ft_n_bins=ft_n_bins,
        noise_median=noise_median,
        noise_min=noise_min,
        noise_max=noise_max,
        tau_maj_us=tau_maj,
        sigma_tau_us=sigma_tau,
        n_contributors=n_contrib,
        band_taus=band_taus,
        n_peaks_total=n_total,
        n_strong=n_strong,
        n_medium=n_medium,
        n_weak=n_weak,
        promotion_min_snr=_opt_float(det.get("promotion_min_snr")),
        weak_medium_snr=_opt_float(det.get("weak_medium_snr")),
        medium_strong_snr=_opt_float(det.get("medium_strong_snr")),
        n_windows_planned=n_windows_planned,
        shape=shape,
        n_windows_fit=fit.n_windows,
        n_fitted_peaks=fit.n_fitted_peaks,
        chi2_median=chi_median,
        chi2_min=chi_min,
        chi2_max=chi_max,
        n_thaw=len(fit.thaw_history),
        n_thaw_accepted=sum(1 for e in fit.thaw_history if e.accepted),
        n_replan=len(fit.replan_history),
        n_rescue_rounds=len(fit.rescue_history),
        products=products,
        # enriched detail
        noise_params=noise_params,
        noise_fraction_overall=None,  # filled below (needs the active-grid replay)
        noise_bands=[],
        tau_params=tau_params,
        tau_preconditions_passed=tau_pre_passed,
        tau_preconditions_notes=tau_pre_notes,
        tau_bimodal=tau_bimodal,
        tau_delta_aic=tau_delta_aic,
        tau_dominant_weight=tau_dom_w,
        tau_mu_a=tau_mu_a,
        tau_mu_b=tau_mu_b,
        tau_pearson_freq=tau_pear,
        recommended_shape=recommended_shape,
        shape_vote_rates=shape_vote_rates,
        tau_exp_present=tau_exp_present,
        tau_G_present=tau_G_present,
        det_params=det,
        det_pass_counts=det_pass_counts,
        n_promoted=n_promoted,
        snr_pctiles=snr_pctiles,
        snr_pctiles_promoted=snr_pctiles_promoted,
        density_chunks=density_chunks,
        window_params=window_params,
        width_min_mhz=width_min,
        width_median_mhz=width_med,
        width_max_mhz=width_max,
        width_pctiles=width_pctiles,
        ppw_median=ppw_med,
        ppw_max=ppw_max,
        ppw_pctiles=ppw_pctiles,
        fit_params=dict(fit.parameters),
        tau0_us=_opt_float(fit.parameters.get("tau0_us")),
        tau_calibration_source=(
            str(fit.parameters.get("tau_calibration_source"))
            if fit.parameters.get("tau_calibration_source") is not None
            else None
        ),
        n_nonconverged=n_nonconverged,
        n_spurs_gated=n_spurs_gated,
        n_empty_dropped=n_empty_dropped,
        n_pruned=n_pruned,
        n_baseline_windows=n_baseline_windows,
        n_windows_gate_checked=n_gate_checked,
        n_windows_fail_gate=len(fails),
        worst_windows=fails[:5],
        chi2r_pctiles=chi2r_pctiles,
        eps_pctiles=eps_pctiles,
        sigma_stat_pctiles=sigma_stat_pctiles,
        snr_bin_rows=snr_bin_rows,
        timebase_n_used=timebase_n_used,
        timebase_lattice_g=timebase_lattice_g,
        median_sigma_stat=float(np.median(s_stat)) if s_stat else None,
        median_sigma_eps=float(np.median(s_eps)) if s_eps else None,
        median_sigma_f=float(np.median(s_f)) if s_f else None,
        sigma_eps_pctiles=_percentiles(s_eps),
        sigma_f_pctiles=_percentiles(s_f),
        n_stat_dominated=n_stat_dom,
        n_eps_dominated=n_eps_dom,
        snr_values_promoted=[s for s in det_snrs if promo is None or s >= promo],
        chi2r_values=[r[1] for r in gate_rows],
        eps_values=[100.0 * r[3] for r in gate_rows],
        sigma_stat_values=s_stat,
        sigma_eps_values=s_eps,
        sigma_f_values=s_f,
    )


def _opt_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _percentiles(values: List[float]) -> Dict[str, float]:
    """p10/p25/median/p75/p90/max of *values* (empty -> ``{}``)."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {}
    arr = np.asarray(finite, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "max": float(arr.max()),
    }


# Brightest-in-window SNR bins -- the natural breakdown for a fit whose χ²ᵣ
# tracks SNR² (noise-dominated bulk -> shape-floor-limited bright cores). Mirrors
# the cross-fixture harness / `_internal/stage5_validation_impl.py`.
_SNR_BIN_EDGES = (100.0, 1000.0, 10000.0)
_SNR_BIN_LABELS = ("<100", "100-1k", "1k-10k", ">=10k")


def _snr_bin(snr_max: float) -> str:
    """Label the brightest-in-window SNR into one of :data:`_SNR_BIN_LABELS`."""
    for edge, label in zip(_SNR_BIN_EDGES, _SNR_BIN_LABELS):
        if snr_max < edge:
            return label
    return _SNR_BIN_LABELS[-1]


def assemble_summary_model(file_path: Union[Path, str]) -> _SummaryModel:
    """Assemble the L2 model, including the per-band noise table replay.

    Kept separate from :func:`_assemble_summary` (which reads only the
    persisted groups) so the active-FT replay -- the one non-trivial cost -- is
    opt-in and easy to stub in tests.
    """
    model = _assemble_summary(file_path)
    trim = (
        (model.band_lo_mhz, model.band_hi_mhz)
        if model.band_lo_mhz is not None and model.band_hi_mhz is not None
        else None
    )
    overall, bands = _noise_band_table(str(file_path), trim)
    model.noise_fraction_overall = overall
    model.noise_bands = bands
    return model


# ---------------------------------------------------------------------------
# Static, code-versioned methods prose (one block per stage)
# ---------------------------------------------------------------------------

_METHODS = {
    "stage0": (
        "The free-induction decay is imported verbatim and the coherent signal "
        "start time is detected data-driven (the chirp excitation and ring-down "
        "are excluded), so every later stage works from the molecular-emission "
        "portion of the record."
    ),
    "stage1": (
        "The canonical Fourier transform is **unapodized, un-windowed, and "
        "native-length**: no exponential apodization, FID window, or zero-padding "
        "is applied. Apodization would trade resolution and bias the line shape, "
        "and zero-padding would interpolate the bins and corrupt the noise and "
        "χ² statistics; the robust per-window fit (Stage 5) is the intended "
        "alternative. The spectrum is trimmed to the active band the molecule "
        "emits into. The bin spacing is therefore 1 / T_record."
    ),
    "stage2": (
        "Noise is estimated on the canonical active FT with a high-pass, "
        "region-aware, Rician-corrected scatter MAD, broadly lower-envelope "
        "smoothed. This is immune to the leakage-pedestal σ inflation that "
        "afflicts level-based estimators on high-SNR, line-dense spectra. It "
        "emits the per-bin complex-RMS σ_x that every later stage scores, plans, "
        "and fits against -- the single noise authority. The per-band table "
        "below shows where the floor and the line density vary across the band."
    ),
    "stage2b": (
        "The decay constant τ is calibrated data-driven from a "
        "sliding-active-window short-time FT on the raw FID, yielding a robust "
        "majority τ_maj ± σ_τ (overall and per frequency band) plus a lineshape "
        "vote. Stage 3's matched filter uses τ_maj as its basis, and Stage 5 "
        "anchors its bidirectional Gaussian τ penalty on it -- but only when the "
        "calibration passes its preconditions (single dominant decay population, "
        "bounded spread); otherwise the later stages fall back to a default. "
        "This stage is optional."
    ),
    "stage3": (
        "Peaks are detected in two passes over the active FT: a primary "
        "Blackman-Harris pass and a matched-filter gap pass (exp-apodized, "
        "τ-basis from Stage 2b) that recovers lines between the primary pass's "
        "windows. Detection uses a continuous, leakage-aware noise floor on both "
        "passes; sub-bin positions come from a throwaway internal zero-padded "
        "grid. Candidates are classified weak / medium / strong by SNR."
    ),
    "stage4": (
        "Detected peaks are grouped into disjoint spectral fitting windows. "
        "Window edges are driven by peak clustering (coherent-point margins and "
        "a content-fit width cap), not by the baseline, so each window holds a "
        "self-contained group of lines to fit jointly. A width cap bounds the "
        "forest so no single window grows unbounded."
    ),
    "stage5": (
        "Each window is fit independently with a robust complex-FFT-residual "
        "non-linear least squares: all peaks share a common decay τ and a "
        "leakage-wing baseline, with each line's frequency, amplitude, and phase "
        "free. The fit is noise-weighted by the Stage 2 σ authority, with thaw "
        "(parameter release), structural replan (merge), and residual-rescue "
        "rounds gated on honest statistical evidence. Spurious instrument tones "
        "are gated out via an FID decay probe and the declared clock lattice. "
        "Window quality is judged by an SNR-aware χ² gate, since at high SNR a "
        "sub-percent lineshape deficit alone drives χ²ᵣ up."
    ),
    "stage6": (
        "Frequencies are corrected for the free-running digitizer timebase scale "
        "error ε (self-calibrated from the clock spurs, multiplicative in the "
        "baseband). The reported σ_f is a three-term precision budget: the NLS "
        "statistical precision, the ε-uncertainty propagated through the "
        "baseband, and a user-declared systematic floor (shipped default 0). "
        "σ_f is a precision, not an accuracy: the per-acquisition absolute offset "
        "δ_down is not independently determinable from instrument "
        "self-calibration and is the user's to declare via σ_floor."
    ),
}

_EQUATIONS = {
    "stage2": r"$$\mathrm{SNR}_k = \frac{|X_k|}{\sigma_{x,k}}, \qquad "
    r"\sigma_x = \sqrt{2}\,\sigma_c$$",
    "stage2b": r"$$S(t)\;\propto\;e^{-t/\tau}, \qquad "
    r"\Delta\nu_{\mathrm{FWHM}} = \frac{1}{\pi\tau}$$",
    "stage3": r"$$\text{weak} \;<\; \mathrm{SNR}_{wm} \;\le\; \text{medium} "
    r"\;<\; \mathrm{SNR}_{ms} \;\le\; \text{strong}$$",
    "stage5": r"$$r_k = \frac{X_k - \hat{X}_k}{\sigma_{x,k}}, \quad "
    r"\chi^2_\nu = \frac{1}{\nu}\sum_k |r_k|^2, \quad "
    r"\text{pass} \iff \chi^2_\nu \le F + (\kappa\,\mathrm{SNR}_{\max})^2$$",
    "stage6": r"$$f_{\mathrm{mol}} = f_p - \frac{f_p - f_{\mathrm{meas}}}{1+\varepsilon}"
    r", \qquad \sigma_f = \sqrt{\sigma_{\mathrm{stat}}^2 "
    r"+ (\sigma_\varepsilon\, f_{bb})^2 + \sigma_{\mathrm{floor}}^2}$$",
}

# Per-stage parameter specs: (param_key, display_label, meaning). Only keys
# present (and non-None) in the stage's persisted params are rendered.
_PARAM_SPECS: Dict[str, List[Tuple[str, str, str]]] = {
    "stage2": [
        ("window_mhz", "window_mhz", "Local scatter-MAD window width"),
        ("pedestal_mhz", "pedestal_mhz", "Leakage-pedestal exclusion half-width"),
        ("line_k", "line_k", "Line-bin rejection threshold (k·σ, high-pass)"),
        ("n_iter", "n_iter", "Region-aware refinement iterations"),
        ("smoothing_mhz", "smoothing_mhz", "Lower-envelope σ smoothing width"),
        ("convolve_mhz", "convolve_mhz", "Final σ smoothing kernel width"),
        ("region_aware", "region_aware", "Per-band (region-local) σ estimate"),
    ],
    "stage2b": [
        ("n_seg", "n_seg", "Sliding active-window STFT segments"),
        ("sample_dt_us", "sample_dt_us", "FID sample spacing"),
        ("rss_gate_factor", "rss_gate_factor", "Per-segment fit-quality gate"),
        ("snr_weighted", "snr_weighted", "SNR-weighted τ majority vote"),
        ("n_spur_bins", "n_spur_bins", "Spur bins excluded from the vote"),
    ],
    "stage3": [
        ("promotion_min_snr", "promotion_min_snr", "Min SNR to promote to a detection"),
        (
            "internal_min_snr",
            "internal_min_snr",
            "Internal padded-grid detection floor",
        ),
        ("weak_medium_snr", "weak_medium_snr", "weak / medium class boundary"),
        ("medium_strong_snr", "medium_strong_snr", "medium / strong class boundary"),
        ("primary_window", "primary_window", "Primary-pass apodization window"),
        ("gap_shape", "gap_shape", "Gap-pass matched-filter line shape"),
        ("gap_leakage_floor_k", "gap_leakage_floor_k", "Gap-pass leakage floor (k)"),
        ("detection_zpf", "detection_zpf", "Internal zero-pad factor (sub-bin)"),
        ("tau_basis_us", "tau_basis_us", "Matched-filter decay basis τ (µs)"),
    ],
    "stage4": [
        ("edge_m", "edge_m", "Window edge margin (bins)"),
        ("trim_m", "trim_m", "Post-construction trim margin (bins)"),
        ("edge_threshold", "edge_threshold", "SNR threshold for edge attachment"),
        ("max_window_width_mhz", "max_window_width_mhz", "Window width cap (MHz)"),
        (
            "max_window_width_points",
            "max_window_width_points",
            "Window width cap (bins)",
        ),
        (
            "min_window_half_width_points",
            "min_window_half_width_points",
            "Minimum coherent half-width (bins)",
        ),
        ("min_freeze_snr", "min_freeze_snr", "SNR above which a contributor is frozen"),
    ],
    "stage5": [
        ("shape", "shape", "Line shape fit per window"),
        ("tau0_us", "tau0_us", "Per-window starting decay constant τ₀ (µs)"),
        ("tau_calibration_source", "tau_calibration_source", "Where τ₀ came from"),
        ("fit_tau", "fit_tau", "τ is a free shared parameter"),
        ("max_decay_factor", "max_decay_factor", "Upper bound on τ (× T_active)"),
        ("baseline_order", "baseline_order", "Leakage-wing baseline polynomial order"),
        ("rescue_snr_threshold", "rescue_snr_threshold", "Residual-rescue SNR trigger"),
        (
            "max_residual_rescue_rounds",
            "max_residual_rescue_rounds",
            "Rescue round cap",
        ),
        ("spur_masking_enabled", "spur_masking_enabled", "Gate instrument spurs"),
    ],
}

_CAL_STATE_PHRASE = {
    "self_calibrated": (
        "Frequencies are ε-corrected for the free-running digitizer "
        "timebase (self-calibrated from the clock spurs)."
    ),
    "rb_locked": (
        "Frequencies are on the instrument's Rb-locked absolute scale (ε = 0)."
    ),
    "uncalibrated": (
        "Frequencies are **uncalibrated** -- the digitizer is free-running and "
        "no timebase self-calibration is available."
    ),
}


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _md_num(x: Optional[float], sig: int = 4) -> str:
    return "n/a" if x is None else _g(x, sig)


def _md_int(x: Optional[int]) -> str:
    return "n/a" if x is None else f"{x:,}"


def _fmt_param(value: Any) -> str:
    """Compact, bounded rendering of a parameter value for the params table."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return _g(float(value), 4)
    return str(value)


def _md_table(
    headers: List[str], rows: List[List[str]], aligns: List[str]
) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def _params_block(
    params: Dict[str, Any], specs: List[Tuple[str, str, str]]
) -> List[str]:
    rows = [
        [f"`{label}`", _fmt_param(params[key]), meaning]
        for key, label, meaning in specs
        if key in params and params[key] is not None
    ]
    if not rows:
        return []
    out = ["**Parameters**", ""]
    out += _md_table(["Parameter", "Value", "Meaning"], rows, [":---", "---:", ":---"])
    out.append("")
    return out


def _concerns_block(concerns: List[_Concern]) -> List[str]:
    out = ["**Concerns**", ""]
    if not concerns:
        out.append("*None flagged.*")
        out.append("")
        return out
    for c in concerns:
        tag = "Warning" if c.severity == "warning" else "Note"
        out.append(f"- **{tag} —** {c.message}")
        out.append(f"  *Recommendation:* {c.recommendation}")
    out.append("")
    return out


def _percentile_table(
    title: str,
    metric_label: str,
    named: List[Tuple[str, Dict[str, float]]],
    sig: int = 3,
) -> List[str]:
    """A p10/p25/median/p75/p90/max table, one row per named distribution."""
    named = [(n, p) for n, p in named if p]
    if not named:
        return []
    rows = [
        [
            name,
            _g(p["p10"], sig),
            _g(p["p25"], sig),
            _g(p["p50"], sig),
            _g(p["p75"], sig),
            _g(p["p90"], sig),
            _g(p["max"], sig),
        ]
        for name, p in named
    ]
    out = [f"**{title}**", ""]
    out += _md_table(
        [metric_label, "p10", "p25", "median", "p75", "p90", "max"],
        rows,
        [":---", "---:", "---:", "---:", "---:", "---:", "---:"],
    )
    out.append("")
    return out


def _strongest_lines(products: FinalProducts, n: int = 10) -> List[FinalPeak]:
    real = [
        p for p in products.peaks if p.snr is not None and math.isfinite(float(p.snr))
    ]
    real.sort(key=lambda p: float(p.snr), reverse=True)  # type: ignore[arg-type]
    return real[:n]


def _pull_interpretation(xref: CatalogCrossRef) -> str:
    """One-line read on the pull spread: optimistic / honest / conservative σ_f."""
    n = len(xref.pull_values)
    if n < 2 or xref.pull_std is None:
        return (
            f"Too few matched lines ({n}) to calibrate the σ_f budget from the "
            "pull spread."
        )
    std = xref.pull_std
    mean = xref.pull_mean if xref.pull_mean is not None else 0.0
    if std > 1.3:
        verdict = (
            "the pull spread is wider than unity, so the reported σ_f looks "
            "**optimistic** (under-estimated). Consider declaring a σ_floor "
            "(`review run --sigma-floor`) from this calibration."
        )
    elif std < 0.7:
        verdict = (
            "the pull spread is narrower than unity, so the reported σ_f looks "
            "**conservative** (over-estimated)."
        )
    else:
        verdict = (
            "the pull spread is consistent with unity, so the reported σ_f budget "
            "looks honest."
        )
    bias = ""
    if abs(mean) > 0.5:
        bias = (
            f" The mean pull is {mean:+.2f} (a systematic ~{abs(mean):.2f}σ_f "
            "frequency offset the precision budget does not capture)."
        )
    return (
        f"Pull = (f_fit − f_cat) / σ_f over {n} matched lines: mean {mean:+.2f}, "
        f"std {std:.2f}. With σ_f honest the pull is ~unit-normal; here {verdict}"
        f"{bias}"
    )


def _catalog_section(xref: CatalogCrossRef, peaks: List[FinalPeak]) -> List[str]:
    """The Level-2 catalog cross-reference section (match rate + worst-pull table).

    Proximity annotation only -- echoes the nearest catalog label per line, never
    an assignment. Doubles as the σ_f pull-calibration surface (item 2): the pull
    distribution validates the reported precision budget, it does not gate it.
    """
    name = Path(xref.catalog_path).name
    out = ["## Catalog cross-reference", ""]
    out.append(
        "Each reported line is flagged against the nearest entry in "
        f"`{name}` ({xref.n_catalog:,} entries) within "
        f"{xref.n_sigma:g}·√(σ_f² + σ_cat²). This **echoes the catalog's opaque "
        "label as a cross-check; it is never an assignment** and never enters "
        "the fit."
    )
    out.append("")
    out.append(
        f"- **Match rate:** {xref.n_matched:,} of {xref.n_total:,} lines "
        f"({xref.match_rate * 100:.0f}%) matched within {xref.n_sigma:g}σ."
    )
    out.append("")
    # Pull-calibration surface.
    out.append("**Pull calibration (σ_f validation)**")
    out.append("")
    out.append(_pull_interpretation(xref))
    out.append("")
    # Worst-pull table: the lines straining the budget hardest.
    worst = sorted(
        (
            (p, m)
            for p, m in zip(peaks, xref.matches)
            if m is not None and math.isfinite(m.pull)
        ),
        key=lambda pm: abs(pm[1].pull),
        reverse=True,
    )[:10]
    if worst:
        out.append("**Largest pulls**")
        out.append("")
        rows = [
            [
                _freq(p.frequency_mhz),
                _g(p.sigma_f_khz, 3),
                m.label,
                _freq(m.frequency_mhz),
                _g(m.delta_khz, 3),
                _g(m.pull, 3),
            ]
            for p, m in worst
        ]
        out += _md_table(
            [
                "Line f (MHz)",
                "σ_f (kHz)",
                "Catalog",
                "Catalog f (MHz)",
                "Δ (kHz)",
                "pull",
            ],
            rows,
            ["---:", "---:", ":---", "---:", "---:", "---:"],
        )
        out.append("")
    return out


def _md_line_table(peaks: List[FinalPeak], unit_value: float, unit_name: str) -> str:
    head = (
        f"| Frequency (MHz) | σ_f (kHz) | Amplitude ({unit_name}) | SNR | "
        f"Origin | Window |"
    )
    rule = "| ---: | ---: | ---: | ---: | :--- | ---: |"
    rows = [head, rule]
    for p in peaks:
        rows.append(
            "| "
            + " | ".join(
                [
                    _freq(p.frequency_mhz),
                    _g(p.sigma_f_khz, 3),
                    _scaled(p.amplitude, unit_value),
                    _md_num(p.snr, 3),
                    p.origin,
                    "" if p.window_id is None else str(p.window_id),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Per-stage concern logic (each returns a list, possibly empty)
# ---------------------------------------------------------------------------


def _concerns_stage2(m: _SummaryModel) -> List[_Concern]:
    out: List[_Concern] = []
    if m.noise_fraction_overall is not None and m.noise_fraction_overall < 0.5:
        out.append(
            _Concern(
                "warning",
                f"Only {m.noise_fraction_overall*100:.0f}% of bins were classed as "
                "noise -- the band is very line-dense, which can bias the σ "
                "floor.",
                "Confirm the analysis band is not dominated by a forest of "
                "lines; consider narrowing the trim or reviewing Stage 2 knobs.",
            )
        )
    dense = [b for b in m.noise_bands if b.noise_fraction < 0.6]
    if dense:
        worst = min(dense, key=lambda b: b.noise_fraction)
        out.append(
            _Concern(
                "note",
                f"{len(dense)} band(s) are line-dense (lowest "
                f"{worst.noise_fraction*100:.0f}% noise near "
                f"{worst.lo_mhz:.0f}-{worst.hi_mhz:.0f} MHz).",
                "The region-aware estimator handles this, but check those bands "
                "in `noise show` if their sigma looks high.",
            )
        )
    return out


def _concerns_stage2b(m: _SummaryModel) -> List[_Concern]:
    out: List[_Concern] = []
    if m.tau_preconditions_passed is False:
        note = "; ".join(n for n in m.tau_preconditions_notes if n and n != "ok")
        out.append(
            _Concern(
                "note",
                "Stage 2b preconditions did not pass"
                + (f" ({note})" if note else "")
                + " -- the decay is flagged as multi-population, so a single "
                "band-wide τ_maj is a coarse summary.",
                "Expected when the decay genuinely has several τ populations "
                "(the per-band τ above shows the variation); if a single τ is "
                "expected, inspect `tau show` for spur or blend contamination. "
                "This does not by itself stop Stage 5 from consuming the "
                "calibration -- see the Stage 5 τ source.",
            )
        )
    if (
        m.recommended_shape
        and m.shape
        and m.recommended_shape.lower() != m.shape.lower()
    ):
        out.append(
            _Concern(
                "note",
                f"The Stage 2b lineshape vote favored `{m.recommended_shape}` but "
                f"Stage 5 fit `{m.shape}`.",
                "Usually fine (the vote is advisory and the bare-prototype vote "
                "can over-call Voigt); re-fit with the voted shape to compare "
                "chi-squared if in doubt.",
            )
        )
    return out


def _concerns_stage4(m: _SummaryModel) -> List[_Concern]:
    out: List[_Concern] = []
    cap = _opt_float(m.window_params.get("max_window_width_mhz"))
    if (
        cap is not None
        and m.width_max_mhz is not None
        and m.width_max_mhz >= 0.98 * cap
    ):
        out.append(
            _Concern(
                "note",
                f"The widest window ({m.width_max_mhz:.1f} MHz) sits at the width "
                f"cap ({cap:.0f} MHz).",
                "A capped window can truncate a wide blend; review it with "
                "`fit show` / `review rank --by width_mhz`.",
            )
        )
    return out


def _concerns_stage5(m: _SummaryModel) -> List[_Concern]:
    out: List[_Concern] = []
    # tau-source diagnosis: a 'none' source means tau0 fell back to T_active/3
    # rather than the measured tau_maj. The usual cause is a shape/twin
    # mismatch -- the fit shape needs its own Stage 2b twin calibration.
    if m.tau_calibration_source == "none":
        shape = (m.shape or "").lower()
        measured = (
            f" (the measured τ_maj is {_md_num(m.tau_maj_us, 3)} µs)"
            if m.tau_maj_us is not None
            else ""
        )
        if shape == "gaussian" and not m.tau_G_present:
            extra = (
                " There is a Lorentzian τ calibration on file, but the Gaussian "
                "fit needs its own τ_G twin."
                if m.tau_exp_present
                else ""
            )
            out.append(
                _Concern(
                    "warning",
                    f"Stage 5 fit a **gaussian** shape but no τ_G calibration "
                    f"(`stage2b_tau_G_calibration`) is present, so τ₀ fell back to "
                    f"T_active/3 = {_md_num(m.tau0_us, 3)} µs{measured}." + extra,
                    "Run `calibrate_tau(shape='gaussian')` (the Gaussian τ "
                    "variant) before `fit_peaks` for a τ-anchored gaussian fit; "
                    "the default `calibrate_tau` only builds the Lorentzian "
                    "calibration.",
                )
            )
        elif shape != "gaussian" and not m.tau_exp_present:
            out.append(
                _Concern(
                    "warning",
                    f"Stage 5 has no Stage 2b τ calibration, so τ₀ fell back to "
                    f"T_active/3 = {_md_num(m.tau0_us, 3)} µs.",
                    "Run `calibrate_tau` before `fit_peaks` for a τ-anchored fit.",
                )
            )
        else:
            out.append(
                _Concern(
                    "note",
                    f"Stage 5 τ₀ used the T_active/3 fallback "
                    f"({_md_num(m.tau0_us, 3)} µs){measured}; the matching Stage 2b "
                    "twin was not consumed.",
                    "Confirm the shape-matching τ calibration "
                    "(`calibrate_tau`, with `shape='gaussian'` for the τ_G "
                    "variant) ran before `fit_peaks`.",
                )
            )
    if m.n_nonconverged:
        out.append(
            _Concern(
                "warning",
                f"{m.n_nonconverged} window(s) did not converge.",
                "Inspect them with `fit show --window N`; they may need a manual "
                "window edit or a re-fit.",
            )
        )
    if m.n_windows_fail_gate:
        wid_list = ", ".join(str(w[0]) for w in m.worst_windows)
        out.append(
            _Concern(
                "warning",
                f"{m.n_windows_fail_gate} of {m.n_windows_gate_checked} window(s) "
                "fail the SNR-aware χ² gate (genuine misfit, not the high-SNR "
                f"shape floor). Worst: windows {wid_list}.",
                "Review those windows with `fit show`; a real misfit usually means "
                "a missing line, an unmodeled blend, or a spur under the line.",
            )
        )
    return out


def _concerns_stage6(m: _SummaryModel) -> List[_Concern]:
    out: List[_Concern] = []
    products = m.products
    state = products.calibration_state if products else ""
    if state == "uncalibrated":
        out.append(
            _Concern(
                "warning",
                "Frequencies are uncalibrated -- the free-running digitizer "
                "timebase error is neither corrected nor budgeted.",
                "Run `tau calibrate-timebase` (declare the clock lattice) to "
                "self-calibrate ε, or treat the frequencies as scale-uncertain.",
            )
        )
    if products is not None and products.sigma_floor_khz == 0.0:
        out.append(
            _Concern(
                "note",
                "σ_floor = 0: the reported σ_f is a precision budget only. "
                "The per-acquisition absolute offset (δ_down) is not included.",
                "If you need an accuracy figure, set `review run --sigma-floor "
                "<kHz>` from your own systematics assessment (or a trusted-line "
                "pull calibration).",
            )
        )
    if m.timebase_n_used is not None and m.timebase_n_used < 5:
        out.append(
            _Concern(
                "note",
                f"ε was fit from only {m.timebase_n_used} clock tone(s).",
                "A small tone count widens σ_ε; confirm the clock declaration "
                "captured the available lattice points.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def _section(
    L: List[str],
    title: str,
    prose: str,
    *,
    params: Optional[List[str]] = None,
    equation: Optional[str] = None,
    results: Optional[List[str]] = None,
    detail: Optional[List[str]] = None,
    concerns: Optional[List[_Concern]] = None,
) -> None:
    L.append(f"### {title}")
    L.append("")
    L.append(prose)
    L.append("")
    if params:
        L.extend(params)
    if equation:
        L.append(equation)
        L.append("")
    if results:
        L.append("*Results:* " + "; ".join(results) + ".")
        L.append("")
    if detail:
        L.extend(detail)
    if concerns is not None:
        L.extend(_concerns_block(concerns))


def _render_markdown(
    model: _SummaryModel,
    file_path: Union[Path, str],
    *,
    cross_ref: Optional[CatalogCrossRef] = None,
) -> str:
    m = model
    products = m.products
    assert products is not None  # guaranteed by _assemble_summary
    uname, uval = _amplitude_unit(products)
    state = products.calibration_state
    cal_phrase = _CAL_STATE_PHRASE.get(state, state)
    eps_ppm = products.epsilon * 1e6
    seps_ppm = products.sigma_epsilon * 1e6

    L: List[str] = []
    L.append(f"# FTMW pipeline report -- {m.experiment}")
    L.append("")
    L.append(
        "Generated by `ftmwpipeline`. This report renders the persisted analysis "
        "record; it does not recompute the fit."
    )
    L.append("")

    # --- Summary --------------------------------------------------------
    L.append("## Summary")
    L.append("")
    src = m.source_path or "(unknown)"
    fmt = m.source_format or "unknown"
    L.append(f"- **Source:** `{src}` ({fmt} format)")
    L.append(f"- **Probe:** {_md_num(m.probe_freq_mhz, 8)} MHz, {m.sideband} sideband")
    if m.band_lo_mhz is not None and m.band_hi_mhz is not None:
        L.append(
            f"- **Active band:** {_md_num(m.band_lo_mhz, 7)}"
            f"--{_md_num(m.band_hi_mhz, 7)} MHz"
        )
    L.append(
        f"- **Lines reported:** {len(products.peaks):,} "
        f"({m.n_fitted_peaks:,} fitted across {m.n_windows_fit:,} windows)"
    )
    L.append(
        f"- **Calibration state:** `{state}` "
        f"(ε = {eps_ppm:+.3f} ± {seps_ppm:.3f} ppm, "
        f"σ_floor = {products.sigma_floor_khz:.3f} kHz)"
    )
    if m.chi2_median is not None:
        L.append(
            f"- **Reduced χ²:** median {_md_num(m.chi2_median, 3)} "
            f"(range {_md_num(m.chi2_min, 3)}--{_md_num(m.chi2_max, 3)})"
        )

    # Top-level concern roll-up.
    all_concerns = (
        _concerns_stage2(m)
        + _concerns_stage2b(m)
        + _concerns_stage4(m)
        + _concerns_stage5(m)
        + _concerns_stage6(m)
    )
    n_warn = sum(1 for c in all_concerns if c.severity == "warning")
    n_note = sum(1 for c in all_concerns if c.severity == "note")
    if all_concerns:
        L.append(
            f"- **Concerns flagged:** {n_warn} warning(s), {n_note} note(s) "
            "(see the per-stage sections)"
        )
    else:
        L.append("- **Concerns flagged:** none")
    if cross_ref is not None:
        pull_note = ""
        if cross_ref.pull_std is not None:
            pull_note = f", pull std {cross_ref.pull_std:.2f}"
        L.append(
            f"- **Catalog match:** {cross_ref.n_matched:,}/{cross_ref.n_total:,} "
            f"lines ({cross_ref.match_rate * 100:.0f}%) within "
            f"{cross_ref.n_sigma:g}σ{pull_note}"
        )
    L.append("")
    L.append(cal_phrase)
    L.append("")

    # --- Methods + results, stage by stage ------------------------------
    L.append("## Methods and results")
    L.append("")

    # Stage 0 (kept simple).
    _section(
        L,
        "Stage 0 -- start-time detection",
        _METHODS["stage0"],
        results=[
            f"start = {_md_num(m.start_us, 4)} µs",
            f"record {_md_num(m.duration_us, 4)} µs ({_md_int(m.n_points)} points)",
        ]
        + ([f"{m.shots:,} shots"] if m.shots is not None else []),
    )

    # Stage 1 (kept simple).
    ft_results = []
    if m.band_lo_mhz is not None and m.band_hi_mhz is not None:
        ft_results.append(
            f"active band {_md_num(m.band_lo_mhz, 7)}--{_md_num(m.band_hi_mhz, 7)} MHz"
        )
    if m.ft_n_bins is not None:
        ft_results.append(f"{m.ft_n_bins:,} bins")
    if m.ft_bin_khz is not None:
        ft_results.append(f"{_md_num(m.ft_bin_khz, 4)} kHz/bin")
    ft_results.append("unapodized, native-length")
    _section(L, "Stage 1 -- Fourier transform", _METHODS["stage1"], results=ft_results)

    # Stage 2 (enriched: per-band table + concerns).
    noise_results = []
    if m.noise_median is not None:
        noise_results.append(
            f"median σ_x = {_md_num(m.noise_median, 3)} "
            f"(range {_md_num(m.noise_min, 3)}--{_md_num(m.noise_max, 3)})"
        )
    if m.noise_fraction_overall is not None:
        noise_results.append(f"{m.noise_fraction_overall*100:.1f}% of bins are noise")
    band_detail: List[str] = []
    if m.noise_bands:
        band_detail.append("**Per-band noise**")
        band_detail.append("")
        rows = [
            [
                f"{b.lo_mhz:.0f}--{b.hi_mhz:.0f}",
                _g(b.median_sigma, 3),
                f"{b.noise_fraction*100:.1f}%",
                f"{b.n_bins:,}",
            ]
            for b in m.noise_bands
        ]
        band_detail += _md_table(
            ["Band (MHz)", "Median σ_x", "Noise fraction", "Bins"],
            rows,
            ["---:", "---:", "---:", "---:"],
        )
        band_detail.append("")
    _section(
        L,
        "Stage 2 -- noise estimation",
        _METHODS["stage2"],
        params=_params_block(m.noise_params, _PARAM_SPECS["stage2"]),
        equation=_EQUATIONS["stage2"],
        results=noise_results,
        detail=band_detail,
        concerns=_concerns_stage2(m),
    )

    # Stage 2b (enriched).
    if m.tau_maj_us is not None:
        tau_results = [
            f"τ_maj = {_md_num(m.tau_maj_us, 3)} ± {_md_num(m.sigma_tau_us, 2)} µs "
            f"from {_md_int(m.n_contributors)} contributors",
        ]
        if m.recommended_shape:
            votes = ", ".join(
                f"{k} {v*100:.0f}%"
                for k, v in sorted(
                    m.shape_vote_rates.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            tau_results.append(
                f"lineshape vote → {m.recommended_shape}"
                + (f" ({votes})" if votes else "")
            )
        tau_detail: List[str] = []
        if m.band_taus:
            tau_detail.append("**Per-band τ**")
            tau_detail.append("")
            rows = [
                [
                    f"{bt.label} ({bt.lo_mhz:.0f}--{bt.hi_mhz:.0f})",
                    f"{bt.tau_us:.3g}",
                    f"{bt.sigma_tau_us:.2g}",
                    f"{bt.n:,}",
                ]
                for bt in m.band_taus
            ]
            tau_detail += _md_table(
                ["Band (MHz)", "τ (µs)", "σ_τ (µs)", "n"],
                rows,
                [":---", "---:", "---:", "---:"],
            )
            tau_detail.append("")
        if m.tau_bimodal is not None:
            tau_detail.append(
                f"Decay-population test: "
                f"{'two-component preferred' if m.tau_bimodal else 'single dominant'}"
                f" (ΔAIC = {_md_num(m.tau_delta_aic, 3)}, dominant weight "
                f"{_md_num(m.tau_dominant_weight, 2)}); frequency–τ correlation "
                f"r = {_md_num(m.tau_pearson_freq, 2)}."
            )
            tau_detail.append("")
        _section(
            L,
            "Stage 2b -- τ calibration",
            _METHODS["stage2b"],
            params=_params_block(m.tau_params, _PARAM_SPECS["stage2b"]),
            equation=_EQUATIONS["stage2b"],
            results=tau_results,
            detail=tau_detail,
            concerns=_concerns_stage2b(m),
        )
    else:
        _section(
            L,
            "Stage 2b -- τ calibration",
            _METHODS["stage2b"],
            results=["not run (Stage 5 used its τ fallback)"],
            concerns=[],
        )

    # Stage 3 (enriched).
    det_results = [
        f"{m.n_peaks_total:,} candidate peaks ({m.n_strong:,} strong, "
        f"{m.n_medium:,} medium, {m.n_weak:,} weak)"
    ]
    if m.promotion_min_snr is not None and m.n_peaks_total:
        frac = 100.0 * m.n_promoted / m.n_peaks_total
        det_results.append(
            f"{m.n_promoted:,} promoted at SNR ≥ {_md_num(m.promotion_min_snr, 3)} "
            f"({frac:.0f}% of candidates)"
        )
    if m.det_pass_counts:
        det_results.append(
            "by pass ("
            + ", ".join(f"{k} {v:,}" for k, v in sorted(m.det_pass_counts.items()))
            + ")"
        )
    det_detail: List[str] = []
    det_detail += _percentile_table(
        "Peak-strength distribution (SNR)",
        "Population",
        [
            ("all candidates", m.snr_pctiles),
            ("promoted", m.snr_pctiles_promoted),
        ],
    )
    if m.density_chunks:
        det_detail.append("**Spectral density**")
        det_detail.append("")
        rows = [
            [
                f"{lo:.0f}--{hi:.0f}",
                f"{n_all:,}",
                f"{n_pro:,}",
            ]
            for lo, hi, n_all, n_pro in m.density_chunks
        ]
        det_detail += _md_table(
            ["Band (MHz)", "Candidates", "Promoted"],
            rows,
            ["---:", "---:", "---:"],
        )
        det_detail.append("")
    _section(
        L,
        "Stage 3 -- peak detection",
        _METHODS["stage3"],
        params=_params_block(m.det_params, _PARAM_SPECS["stage3"]),
        equation=_EQUATIONS["stage3"],
        results=det_results,
        detail=det_detail,
        concerns=[],
    )

    # Stage 4 (enriched).
    win_results = [f"{m.n_windows_planned:,} disjoint windows planned"]
    if m.width_median_mhz is not None:
        win_results.append(
            f"width {m.width_min_mhz:.2g}--{m.width_max_mhz:.2g} MHz "
            f"(median {m.width_median_mhz:.2g})"
        )
    if m.ppw_median is not None:
        win_results.append(
            f"{m.ppw_median:.0f} peaks/window median ({m.ppw_max:,} max)"
        )
    win_detail: List[str] = []
    win_detail += _percentile_table(
        "Window distribution",
        "Metric",
        [
            ("width (MHz)", m.width_pctiles),
            ("peaks / window", m.ppw_pctiles),
        ],
    )
    _section(
        L,
        "Stage 4 -- window assignment",
        _METHODS["stage4"],
        params=_params_block(m.window_params, _PARAM_SPECS["stage4"]),
        results=win_results,
        detail=win_detail,
        concerns=_concerns_stage4(m),
    )

    # Stage 5 (enriched).
    fit_results = [
        f"{m.n_windows_fit:,} windows fit",
        f"{m.n_fitted_peaks:,} fitted lines",
    ]
    if m.shape:
        fit_results.append(f"{m.shape} shape, τ₀ = {_md_num(m.tau0_us, 3)} µs")
    if m.chi2_median is not None:
        fit_results.append(
            f"χ²ᵣ median {_md_num(m.chi2_median, 3)} "
            f"(range {_md_num(m.chi2_min, 3)}--{_md_num(m.chi2_max, 3)})"
        )
    fit_results.append(
        f"{m.n_thaw_accepted:,}/{m.n_thaw:,} thaws accepted, {m.n_replan:,} replans, "
        f"{m.n_rescue_rounds:,} rescue rounds"
    )
    fit_detail = [
        f"Bookkeeping: {m.n_spurs_gated:,} spurs gated, {m.n_empty_dropped:,} empty "
        f"windows dropped, {m.n_pruned:,} peaks pruned, {m.n_baseline_windows:,} "
        f"windows carried a leakage-wing baseline.",
        "",
        f"SNR-aware quality gate: {m.n_windows_gate_checked - m.n_windows_fail_gate:,}"
        f"/{m.n_windows_gate_checked:,} windows pass "
        f"(κ = {DEFAULT_SHAPE_ERROR_KAPPA}, F = {DEFAULT_CHI2R_NOISE_FLOOR}).",
        "",
    ]
    # Per-window fit-quality distributions.
    fit_detail += _percentile_table(
        "Fit-quality distributions",
        "Metric",
        [
            ("reduced χ² (per window)", m.chi2r_pctiles),
            ("shape-error ε (% per bin)", m.eps_pctiles),
            ("freq precision σ_stat (kHz)", m.sigma_stat_pctiles),
        ],
    )
    # The χ²ᵣ ~ SNR² structure: χ²ᵣ and the gate broken out by window brightness.
    if m.snr_bin_rows:
        fit_detail.append("**Fit quality by window brightness (SNR_max)**")
        fit_detail.append("")
        rows = [
            [
                label,
                f"{n:,}",
                _g(med, 3),
                _g(p90, 3),
                _g(mx, 3),
                f"{pass_rate*100:.0f}%",
                _g(med_eps, 2),
            ]
            for (label, n, med, p90, mx, pass_rate, med_eps) in m.snr_bin_rows
        ]
        fit_detail += _md_table(
            ["SNR_max", "windows", "median χ²ᵣ", "p90", "max", "pass", "median ε(%)"],
            rows,
            [":---", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
        fit_detail.append("")
    _section(
        L,
        "Stage 5 -- per-window fitting",
        _METHODS["stage5"],
        params=_params_block(m.fit_params, _PARAM_SPECS["stage5"]),
        equation=_EQUATIONS["stage5"],
        results=fit_results,
        detail=fit_detail,
        concerns=_concerns_stage5(m),
    )

    # Stage 6 (enriched: budget breakdown).
    cal_results = [
        f"state `{state}`",
        f"ε = {eps_ppm:+.3f} ± {seps_ppm:.3f} ppm",
        f"σ_floor = {products.sigma_floor_khz:.3f} kHz",
        f"{len(products.peaks):,} calibrated lines",
    ]
    budget_detail: List[str] = []
    if m.median_sigma_f is not None:
        budget_detail.append(
            f"σ_f budget (median): σ_stat = {_md_num(m.median_sigma_stat, 3)} "
            f"kHz, σ_ε = {_md_num(m.median_sigma_eps, 3)} kHz, "
            f"σ_f = {_md_num(m.median_sigma_f, 3)} kHz. "
            f"{m.n_stat_dominated:,} lines are precision-dominated, "
            f"{m.n_eps_dominated:,} are timebase-dominated."
        )
        budget_detail.append("")
    # Full σ_f budget distribution (parallels the Stage 5 σ_stat percentiles).
    budget_detail += _percentile_table(
        "σ_f budget distribution (kHz)",
        "Component",
        [
            ("σ_stat (NLS precision)", m.sigma_stat_pctiles),
            ("σ_ε (timebase)", m.sigma_eps_pctiles),
            ("σ_f (total)", m.sigma_f_pctiles),
        ],
    )
    if m.timebase_n_used is not None:
        budget_detail.append(
            f"Timebase: ε from {m.timebase_n_used:,} clock tone(s)"
            + (
                f", lattice spacing {m.timebase_lattice_g:.0f} MHz"
                if m.timebase_lattice_g
                else ""
            )
            + "."
        )
        budget_detail.append("")
    _section(
        L,
        "Stage 6 -- calibration and finalization",
        _METHODS["stage6"],
        equation=_EQUATIONS["stage6"],
        results=cal_results,
        detail=budget_detail,
        concerns=_concerns_stage6(m),
    )

    # --- Catalog cross-reference (optional) -----------------------------
    if cross_ref is not None:
        L.extend(_catalog_section(cross_ref, list(products.peaks)))

    # --- Strongest lines ------------------------------------------------
    L.append("## Strongest lines")
    L.append("")
    top = _strongest_lines(products, 10)
    if top:
        L.append(_md_line_table(top, uval, uname))
    else:
        L.append("_No lines with finite SNR._")
    L.append("")

    # --- Full line list -------------------------------------------------
    L.append("## Final line list")
    L.append("")
    L.append(
        f"The full {len(products.peaks):,}-line calibrated table is the companion "
        "data export -- run `report table --format csv` (or json / latex)."
    )
    L.append("")

    return "\n".join(L) + "\n"
