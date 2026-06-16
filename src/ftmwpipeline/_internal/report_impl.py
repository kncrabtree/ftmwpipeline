"""Shared implementation for the Stage 6 ``report`` object.

Reports render the persisted Stage 6 record; they never recompute the fit.

Level 1 (``report table``, data export): the consolidated :class:`FinalProducts`
table serialized to CSV, JSON, or a LaTeX ``booktabs`` table for paper SI. All
three are flat serializations of the same persisted table -- *assemble once,
render many* -- so they stay consistent by construction.

Level 2 (``report summary``, methods + results document): a Markdown report
interleaving **static, code-versioned algorithm prose** -- a methods section
that lives here so it stays in sync with the code, not pulled from the planning
docs -- with the per-experiment numbers read from each persisted stage. The full
line list is the companion Level-1 CSV; only a summary (counts, band, χ² stats,
strongest lines) appears inline unless ``--include-table`` inlines it.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

from ..core.data_structures import FinalPeak, FinalProducts, SpectrumFit
from ..io.peak_serialization import load_peaks_from_hdf5
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


# ===========================================================================
# Level 2 -- methods + results document (Markdown)
# ===========================================================================
#
# Assembled once from the persisted record, then rendered. The "methods"
# prose is static and code-versioned (it describes the algorithms as they
# are, timelessly); the "results" numbers are read from each persisted stage.


@dataclass
class _BandTau:
    """Per-band τ majority (Stage 2b ``band_majorities`` entry)."""

    label: str
    lo_mhz: float
    hi_mhz: float
    n: int
    tau_us: float
    sigma_tau_us: float


@dataclass
class _SummaryModel:
    """Per-experiment numbers gathered from the persisted stages for L2.

    Optional fields are ``None`` when the originating stage is absent (Stage 2b
    is an optional dependency; the rest are present whenever ``FinalProducts``
    exists, since ``review run`` requires Stages 0--5).
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
    tau_source: Optional[str]
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


def _h5_attr_json(group: h5py.Group, key: str) -> dict:
    """Parse a JSON ``parameters`` attr off *group*, tolerating absence."""
    raw = group.attrs.get(key)
    if raw is None:
        return {}
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError):
        return {}


def _assemble_summary(file_path: Union[Path, str]) -> _SummaryModel:
    """Gather the per-stage numbers for the L2 report from the persisted file.

    Reads each stage's persisted group directly (never recomputes). Requires a
    built :class:`FinalProducts` (i.e. ``review run`` has consolidated the
    calibrated table); raises :class:`ValueError` otherwise.
    """
    from ..io.fid_serialization import load_fid_from_hdf5
    from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
    from ..io.tau_calibration_serialization import load_tau_calibration_from_hdf5
    from ..io.window_serialization import load_window_plan_from_hdf5

    path = str(file_path)
    review = load_stage6_review_from_file(path)
    products = review.final_products
    if products is None:
        raise ValueError(
            "No final-products table found in this file. Run 'review run' first "
            "to consolidate the calibrated final products."
        )

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
        # The canonical FT is native-length, so the bin spacing is 1 / T_record.
        ft_bin_khz = 1.0e3 / fid.duration_us if fid.duration_us else None
        ft_n_bins = (
            int(round((band_hi - band_lo) / (ft_bin_khz * 1e-3)))
            if band_lo is not None and band_hi is not None and ft_bin_khz
            else None
        )

        # --- Stage 2: noise (verbatim sigma_x array stored on the group) ----
        noise_median = noise_min = noise_max = None
        noise_grp = h5f.get("stage2_noise_result")
        if noise_grp is not None and "rms_noise_full" in noise_grp:
            sigma = np.asarray(noise_grp["rms_noise_full"][:], dtype=float)
            sigma = sigma[np.isfinite(sigma)]
            if sigma.size:
                noise_median = float(np.median(sigma))
                noise_min = float(sigma.min())
                noise_max = float(sigma.max())

        # --- Stage 2b: tau (optional dependency) ----------------------------
        tau_maj = sigma_tau = None
        n_contrib = None
        tau_source = None
        band_taus: List[_BandTau] = []
        if "stage2b_tau_calibration" in h5f:
            tau = load_tau_calibration_from_hdf5(h5f["stage2b_tau_calibration"])
            tau_maj = float(tau.tau_maj_us)
            sigma_tau = float(tau.sigma_tau_us)
            n_contrib = int(tau.n_contributors)
            for bm in tau.band_majorities:
                band_taus.append(
                    _BandTau(
                        label=str(bm.label),
                        lo_mhz=float(bm.freq_lo_mhz),
                        hi_mhz=float(bm.freq_hi_mhz),
                        n=int(bm.n),
                        tau_us=float(bm.tau_maj_us),
                        sigma_tau_us=float(bm.sigma_tau_us),
                    )
                )

        # --- Stage 3: peak detection ---------------------------------------
        n_total = n_strong = n_medium = n_weak = 0
        if "stage3_peaks" in h5f:
            peaks = load_peaks_from_hdf5(h5f["stage3_peaks"])
            n_total = len(peaks)
            for pk in peaks:
                label = getattr(pk.classification, "value", str(pk.classification))
                if label == "strong":
                    n_strong += 1
                elif label == "medium":
                    n_medium += 1
                elif label == "weak":
                    n_weak += 1
        det = _h5_attr_json(
            h5f.get("processing_parameters/peak_detection", h5f), "parameters"
        )

        # --- Stage 4: window assignment ------------------------------------
        n_windows_planned = 0
        if "stage4_windows" in h5f:
            plan = load_window_plan_from_hdf5(h5f["stage4_windows"])
            n_windows_planned = plan.n_windows

        # --- Stage 5: fitting ----------------------------------------------
        fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    chi = [
        float(wf.reduced_chi2)
        for wf in fit.window_fits
        if wf.reduced_chi2 is not None and math.isfinite(float(wf.reduced_chi2))
    ]
    chi_median = float(np.median(chi)) if chi else None
    chi_min = min(chi) if chi else None
    chi_max = max(chi) if chi else None
    shape = str(fit.parameters.get("shape")) if fit.parameters.get("shape") else None

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
        tau_source=tau_source,
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
    )


def _opt_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


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
        "emits the per-bin complex-RMS σ_x that every later stage scores, "
        "plans, and fits against -- the single noise authority."
    ),
    "stage2b": (
        "The decay constant τ is calibrated data-driven from a "
        "sliding-active-window short-time FT on the raw FID, yielding a robust "
        "majority τ_maj ± σ_τ (overall and per frequency band). Stage 3's "
        "matched filter uses τ_maj as its basis, and Stage 5 anchors its "
        "bidirectional Gaussian τ penalty on it. This stage is optional; absent "
        "it, the later stages fall back to defaults."
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
        "self-contained group of lines to fit jointly."
    ),
    "stage5": (
        "Each window is fit independently with a robust complex-FFT-residual "
        "non-linear least squares: all peaks share a common decay τ and a "
        "leakage-wing baseline, with each line's frequency, amplitude, and phase "
        "free. The fit is noise-weighted by the Stage 2 σ authority, with thaw "
        "(parameter release), structural replan (merge), and residual-rescue "
        "rounds gated on honest statistical evidence. Spurious instrument tones "
        "are gated out via an FID decay probe and the declared clock lattice."
    ),
    "stage6": (
        "Frequencies are corrected for the free-running digitizer timebase scale "
        "error ε (self-calibrated from the clock spurs, multiplicative in the "
        "baseband). The reported σ_f is a three-term precision budget, "
        "σ_f = √(σ_stat² + (σ_ε·f_baseband)² + σ_floor²): the NLS statistical "
        "precision, the ε-uncertainty propagated through the baseband, and a "
        "user-declared systematic floor (shipped default 0). σ_f is a precision, "
        "not an accuracy: the per-acquisition absolute offset δ_down is not "
        "independently determinable from instrument self-calibration and is the "
        "user's to declare via σ_floor."
    ),
}

_CAL_STATE_PHRASE = {
    "self_calibrated": (
        "Frequencies are ε-corrected for the free-running digitizer timebase "
        "(self-calibrated from the clock spurs)."
    ),
    "rb_locked": (
        "Frequencies are on the instrument's Rb-locked absolute scale (ε ≡ 0)."
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


def _strongest_lines(products: FinalProducts, n: int = 10) -> List[FinalPeak]:
    real = [
        p for p in products.peaks if p.snr is not None and math.isfinite(float(p.snr))
    ]
    real.sort(key=lambda p: float(p.snr), reverse=True)  # type: ignore[arg-type]
    return real[:n]


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


def _render_markdown(
    model: _SummaryModel,
    file_path: Union[Path, str],
    *,
    include_table: bool,
) -> str:
    products = model.products
    assert products is not None  # guaranteed by _assemble_summary
    uname, uval = _amplitude_unit(products)
    state = products.calibration_state
    cal_phrase = _CAL_STATE_PHRASE.get(state, state)

    L: List[str] = []
    L.append(f"# FTMW pipeline report -- {model.experiment}")
    L.append("")
    L.append(
        "Generated by `ftmwpipeline`. This report renders the persisted analysis "
        "record; it does not recompute the fit."
    )
    L.append("")

    # --- Summary --------------------------------------------------------
    L.append("## Summary")
    L.append("")
    src = model.source_path or "(unknown)"
    fmt = model.source_format or "unknown"
    L.append(f"- **Source:** `{src}` ({fmt} format)")
    L.append(
        f"- **Probe:** {_md_num(model.probe_freq_mhz, 8)} MHz, "
        f"{model.sideband} sideband"
    )
    if model.band_lo_mhz is not None and model.band_hi_mhz is not None:
        L.append(
            f"- **Active band:** {_md_num(model.band_lo_mhz, 7)}"
            f"--{_md_num(model.band_hi_mhz, 7)} MHz"
        )
    L.append(
        f"- **Lines reported:** {len(products.peaks):,} "
        f"({model.n_fitted_peaks:,} fitted across {model.n_windows_fit:,} windows)"
    )
    eps_ppm = products.epsilon * 1e6
    seps_ppm = products.sigma_epsilon * 1e6
    L.append(
        f"- **Calibration state:** `{state}` "
        f"(ε = {eps_ppm:+.3f} ± {seps_ppm:.3f} ppm, "
        f"σ_floor = {products.sigma_floor_khz:.3f} kHz)"
    )
    if model.chi2_median is not None:
        L.append(
            f"- **Reduced χ²:** median {_md_num(model.chi2_median, 3)} "
            f"(range {_md_num(model.chi2_min, 3)}--{_md_num(model.chi2_max, 3)})"
        )
    L.append("")
    L.append(cal_phrase)
    L.append("")

    # --- Methods + results, stage by stage ------------------------------
    L.append("## Methods and results")
    L.append("")

    def section(title: str, prose: str, results: List[str]) -> None:
        L.append(f"### {title}")
        L.append("")
        L.append(prose)
        L.append("")
        if results:
            L.append("*Results:* " + "; ".join(results) + ".")
            L.append("")

    section(
        "Stage 0 -- start-time detection",
        _METHODS["stage0"],
        [
            f"start = {_md_num(model.start_us, 4)} µs",
            f"record {_md_num(model.duration_us, 4)} µs "
            f"({_md_int(model.n_points)} points)",
        ]
        + ([f"{model.shots:,} shots"] if model.shots is not None else []),
    )

    ft_results = []
    if model.band_lo_mhz is not None and model.band_hi_mhz is not None:
        ft_results.append(
            f"active band {_md_num(model.band_lo_mhz, 7)}"
            f"--{_md_num(model.band_hi_mhz, 7)} MHz"
        )
    if model.ft_n_bins is not None:
        ft_results.append(f"{model.ft_n_bins:,} bins")
    if model.ft_bin_khz is not None:
        ft_results.append(f"{_md_num(model.ft_bin_khz, 4)} kHz/bin")
    ft_results.append("unapodized, native-length")
    section("Stage 1 -- Fourier transform", _METHODS["stage1"], ft_results)

    noise_results = []
    if model.noise_median is not None:
        noise_results.append(
            f"median σ_x = {_md_num(model.noise_median, 3)} "
            f"(range {_md_num(model.noise_min, 3)}--{_md_num(model.noise_max, 3)})"
        )
    section("Stage 2 -- noise estimation", _METHODS["stage2"], noise_results)

    if model.tau_maj_us is not None:
        tau_results = [
            f"τ_maj = {_md_num(model.tau_maj_us, 3)} ± "
            f"{_md_num(model.sigma_tau_us, 2)} µs "
            f"from {_md_int(model.n_contributors)} contributors"
        ]
        if model.band_taus:
            per_band = ", ".join(
                f"{bt.label} {_md_num(bt.tau_us, 3)} µs" for bt in model.band_taus
            )
            tau_results.append(f"per band ({per_band})")
        section("Stage 2b -- τ calibration", _METHODS["stage2b"], tau_results)
    else:
        section(
            "Stage 2b -- τ calibration",
            _METHODS["stage2b"],
            ["not run (Stage 5 used its τ fallback)"],
        )

    det_results = [
        f"{model.n_peaks_total:,} candidate peaks "
        f"({model.n_strong:,} strong, {model.n_medium:,} medium, "
        f"{model.n_weak:,} weak)"
    ]
    if model.promotion_min_snr is not None:
        det_results.append(f"promotion SNR ≥ {_md_num(model.promotion_min_snr, 3)}")
    section("Stage 3 -- peak detection", _METHODS["stage3"], det_results)

    section(
        "Stage 4 -- window assignment",
        _METHODS["stage4"],
        [f"{model.n_windows_planned:,} disjoint windows planned"],
    )

    fit_results = [
        f"{model.n_windows_fit:,} windows fit",
        f"{model.n_fitted_peaks:,} fitted lines",
    ]
    if model.shape:
        fit_results.append(f"{model.shape} line shape")
    if model.chi2_median is not None:
        fit_results.append(
            f"χ²ᵣ median {_md_num(model.chi2_median, 3)} "
            f"(range {_md_num(model.chi2_min, 3)}--{_md_num(model.chi2_max, 3)})"
        )
    fit_results.append(
        f"{model.n_thaw_accepted:,}/{model.n_thaw:,} thaws accepted, "
        f"{model.n_replan:,} replans, {model.n_rescue_rounds:,} rescue rounds"
    )
    section("Stage 5 -- per-window fitting", _METHODS["stage5"], fit_results)

    section(
        "Stage 6 -- calibration and finalization",
        _METHODS["stage6"],
        [
            f"state `{state}`",
            f"ε = {eps_ppm:+.3f} ± {seps_ppm:.3f} ppm",
            f"σ_floor = {products.sigma_floor_khz:.3f} kHz",
            f"{len(products.peaks):,} calibrated lines",
        ],
    )

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
    if include_table:
        L.append(
            f"All {len(products.peaks):,} calibrated lines " f"(amplitude in {uname}):"
        )
        L.append("")
        L.append(_md_line_table(list(products.peaks), uval, uname))
        L.append("")
    else:
        L.append(
            f"The full {len(products.peaks):,}-line calibrated table is the "
            "companion data export -- run `report table --format csv` (or json / "
            "latex), or pass `--include-table` to inline it here."
        )
        L.append("")

    return "\n".join(L) + "\n"


def report_summary_impl(
    file_path: Union[Path, str],
    *,
    output: Optional[Union[Path, str]] = None,
    include_table: bool = False,
) -> str:
    """Render the persisted Level-2 methods + results summary (Markdown).

    Interleaves static, code-versioned algorithm prose with the per-experiment
    numbers read from each persisted stage. The full line list is the companion
    Level-1 export unless *include_table* inlines it.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    output :
        When given, also write the rendered Markdown to this path.
    include_table :
        Inline the full calibrated line table rather than a summary plus a
        pointer to the companion ``report table`` export.

    Returns
    -------
    str
        The rendered Markdown document.

    Raises
    ------
    ValueError
        If no final-products table is present (``review run`` has not been run).
    """
    model = _assemble_summary(file_path)
    text = _render_markdown(model, file_path, include_table=include_table)
    if output is not None:
        Path(output).write_text(text)
    return text
