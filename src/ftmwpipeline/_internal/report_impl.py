"""Shared implementation for the Stage 6 ``report`` object.

Reports render the persisted Stage 6 record; they never recompute the fit. This
module holds Level 1 (data export): the consolidated :class:`FinalProducts`
table serialized to CSV, JSON, or a LaTeX ``booktabs`` table for paper SI. All
three are flat serializations of the same persisted table -- *assemble once,
render many* -- so they stay consistent by construction.

Robustness: every numeric field is rendered through width-bounded, non-finite
guarded formatters, so a degenerate fit (e.g. an amplitude-collapsed phantom
with a runaway uncertainty) never dumps a hundred-digit number into the table.
Amplitudes are reported in a dynamically chosen SI unit (V/mV/uV/nV/...) so the
magnitudes read sensibly, and amplitude/phase/SNR carry their uncertainties.
LaTeX uses concise value(uncertainty) notation. See
``dev-docs/planning/stage6-reports.md`` §C.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..core.data_structures import FinalPeak, FinalProducts
from ..io.stage6_review_serialization import load_stage6_review_from_file

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
    products: FinalProducts, file_path: Union[Path, str], amp_unit: str
) -> List[Tuple[str, str]]:
    return [
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


def _render_csv(products: FinalProducts, file_path: Union[Path, str]) -> str:
    uname, uval = _amplitude_unit(products)
    lines = ["# ftmwpipeline final products"]
    lines += [f"# {k}: {v}" for k, v in _provenance(products, file_path, uname)]
    lines.append(",".join(_CSV_COLUMNS))
    for p in products.peaks:
        lines.append(",".join(_csv_row(p, uval)))
    return "\n".join(lines) + "\n"


def _jnum(x: Optional[float]) -> Optional[float]:
    """JSON-safe float: non-finite -> null (so the JSON stays strict-valid)."""
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


def _peak_json(p: FinalPeak, unit_value: float) -> dict:
    return {
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


def _render_json(products: FinalProducts, file_path: Union[Path, str]) -> str:
    uname, uval = _amplitude_unit(products)
    payload = {
        "metadata": {
            "experiment": Path(file_path).stem,
            "calibration_state": products.calibration_state,
            "epsilon": _jnum(products.epsilon),
            "sigma_epsilon": _jnum(products.sigma_epsilon),
            "sigma_floor_khz": _jnum(products.sigma_floor_khz),
            "probe_freq_mhz": _jnum(products.probe_freq_mhz),
            "sideband": products.sideband,
            "amplitude_unit": uname,
            "n_peaks": len(products.peaks),
        },
        "peaks": [_peak_json(p, uval) for p in products.peaks],
    }
    return json.dumps(payload, indent=2) + "\n"


def _latex_caption(products: FinalProducts, amp_unit: str) -> str:
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
    return (
        f"Fitted line list. {freq}; $\\sigma_f$ is the reported precision budget. "
        f"Uncertainties are in units of the last digit; amplitude in "
        f"{_unit_latex(amp_unit)}."
    )


def _render_latex(products: FinalProducts, file_path: Union[Path, str]) -> str:
    uname, uval = _amplitude_unit(products)
    cols = [
        (
            "Frequency (MHz)",
            lambda p: _concise(p.frequency_mhz, p.sigma_f_khz * 1e-3),
        ),
        (
            f"Amplitude ({_unit_latex(uname)})",
            lambda p: _concise(
                p.amplitude / uval,
                None if p.amplitude_error is None else p.amplitude_error / uval,
            ),
        ),
        (
            "SNR",
            lambda p: "--" if p.snr is None else _concise(p.snr, p.snr_error),
        ),
    ]
    spec = "r" * len(cols)
    out = ["% ftmwpipeline final products -- requires \\usepackage{booktabs}"]
    out += [f"% {k}: {v}" for k, v in _provenance(products, file_path, uname)]
    out.append(r"\begin{table}")
    out.append(r"  \centering")
    out.append(r"  \caption{" + _latex_caption(products, uname) + r"}")
    out.append(r"  \begin{tabular}{" + spec + "}")
    out.append(r"    \toprule")
    out.append("    " + " & ".join(h for h, _ in cols) + r" \\")
    out.append(r"    \midrule")
    for p in products.peaks:
        out.append("    " + " & ".join(fmt(p) for _, fmt in cols) + r" \\")
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

    Returns
    -------
    str
        The rendered table.

    Raises
    ------
    ValueError
        If *fmt* is unknown, or no final-products table is present (the Stage 6
        ``review run`` consolidation has not been run).
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

    text = _RENDERERS[key](products, file_path)
    if output is not None:
        Path(output).write_text(text)
    return text
