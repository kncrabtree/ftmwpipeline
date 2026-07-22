"""Regenerate the figures and key results for the Stage 5 fitting methods note.

This harness reproduces, from fixed-seed simulators and the checked-in example
fixtures, the quantitative claims in :doc:`stage5_fitting`:

* the **context-invariant accept gate** -- the shipped penalized-Delta-chi^2
  bar ``Delta chi2 > 2 lambda Delta k`` is independent of how many quiet bins
  pad a window, while the classical nested-model F-test the bar replaced
  manufactures significance from that padding;
* the **sigma_eff localization** -- the same gate, scored against the
  fidelity-inflated per-bin noise ``sigma_eff^2 = sigma^2 + (kappa |model|)^2``,
  discounts residual evidence sitting *under* a bright model component (the
  irreducible lineshape floor) while keeping full weight where the model is
  small (a real missing line);
* **blend recovery vs detectability** -- a K-known joint fit recovers a close
  blend to ~kHz down to half a line width, and a single-cosine fit to the same
  blend leaves a large, smooth reduced-chi^2 elevation, so blends are always
  detectable and the only failure mode is sequential initialization (which the
  blend-aware seeder fixes);
* the **SNR-aware health gate across fixtures** -- on the seven checked-in
  experiments the per-window reduced chi^2 tracks ``F + (kappa SNR)^2`` across
  roughly three decades of signal-to-noise, with catalog recall on the two
  vinyl-cyanide fixtures.

The gate, the F-test, the sigma_eff chi^2, the window fitter, and the SNR-aware
validator are all the shipped ``ftmwpipeline`` functions, so the note validates
the production code rather than a prototype.

Run as a script to (re)write ``figures/*.png`` and ``results.json`` beside this
file::

    python generate.py                  # synthetic + cross-fixture (minutes)
    python generate.py --no-cross-fixture   # synthetic only (seconds)
    python generate.py --no-figures

The committed ``results.json`` is the regression target for
``tests/integration/test_stage5_fitting_report.py`` (the ``slow`` marker); the
cross-fixture roll-up is a snapshot (a full seven-fixture fit is minutes, too
heavy to regenerate in the test), checked for invariants rather than rebuilt.
``gate_window_invariance``, ``sigma_eff_localization``, and ``blend_study`` back
the fast, data-free unit checks in
``tests/unit/fitting/test_stage5_fitting_invariants.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    model_spectrum,
)
from ftmwpipeline.fitting.validation import (
    DEFAULT_CHI2R_NOISE_FLOOR,
    DEFAULT_GATE_PENALTY_LAMBDA,
    DEFAULT_GATE_SIGMA_EFF_KAPPA,
    DEFAULT_SHAPE_ERROR_KAPPA,
    calculate_chi_squared_improvement,
    feature_fwhm,
    sigma_eff_chi2,
)
from ftmwpipeline.fitting.window_fit import fit_window

SEED = 20260620
EXAMPLES = Path("examples/blackchirp_data")
TRIM = (26500.0, 40000.0)
FIXTURES = ["2638", "1019", "1231", "1512", "360", "363", "655"]
# The one checked-in ground-truth catalog (the resolved vinyl-cyanide species'
# in-band union). 1512 and 655 are both vinyl cyanide and match against it.
CATALOG = Path("examples/blackchirp_data/vinyl-cyanide-reference/combined_lines.csv")
CATALOG_FIXTURES = ("1512", "655")

# Reference acquisition / line geometry (the 2638-class fixture scale).
T_ACTIVE_US = 12.65
TAU_US = 4.0
BIN_MHZ = 0.012

# One added line costs three parameters (amplitude, offset, phase); tau is
# shared and does not count toward Delta k.
DELTA_K = 3


def _bar() -> float:
    """The shipped penalized-gate accept bar on Delta chi^2 (floor scaling off)."""
    assert DEFAULT_GATE_PENALTY_LAMBDA is not None
    return 2.0 * float(DEFAULT_GATE_PENALTY_LAMBDA) * DELTA_K


def _complex_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian band with E[|n|^2] = sigma^2."""
    s = sigma / np.sqrt(2.0)
    noise: np.ndarray = rng.normal(0.0, s, n) + 1j * rng.normal(0.0, s, n)
    return noise


# ---------------------------------------------------------------------------
# 1. The accept gate is window-size invariant; the F-test is not.
# ---------------------------------------------------------------------------
def gate_window_invariance() -> Dict[str, Any]:
    """F-test p-value vs window padding, against the flat penalized-gate verdict.

    A fixed local candidate carries a fixed amount of evidence ``delta_chi2``
    (the likelihood-ratio statistic over the bins it informs). The window around
    it is padded with progressively more quiet noise bins. The classical nested
    F-test divides that local evidence by the *whole-window* reduced chi^2; as
    padding drives that denominator toward one, the F-statistic rises and the
    p-value falls -- significance manufactured from empty bins. The shipped gate
    compares ``delta_chi2`` to the fixed bar ``2 lambda delta_k`` and is
    independent of the padding.
    """
    bar = _bar()
    n_params_more = 2 * DELTA_K + 1  # K=2 lines + shared tau
    m_feature = 16  # informative complex bins under the feature
    pad_each_side = [0, 8, 24, 56, 120, 280]

    # Two candidates, each with fixed local evidence:
    #  - "absorber": a marginal candidate (delta_chi2 < bar) sitting on a bright
    #    line whose K+1 model still carries an irreducible lineshape misfit
    #    ``c_floor`` -- the case the F-test wrongly accepts on a padded window;
    #  - "real": genuine evidence (delta_chi2 > bar), accepted by both.
    cands = {
        "absorber": {"delta_chi2": 20.0, "c_floor": 30.0},
        "real": {"delta_chi2": 60.0, "c_floor": 0.0},
    }

    out: Dict[str, Any] = {
        "lambda": float(DEFAULT_GATE_PENALTY_LAMBDA),
        "delta_k": DELTA_K,
        "bar": bar,
        "n_data": [2 * (m_feature + 2 * p) for p in pad_each_side],
        "pvalue": {},
        "fstat": {},
        "penalized_accept": {},
    }
    for name, c in cands.items():
        dchi2 = c["delta_chi2"]
        ps: List[float] = []
        fs: List[float] = []
        for p in pad_each_side:
            m = m_feature + 2 * p
            n_data = 2 * m
            # K+1 model: ~1 per quiet bin plus the irreducible local misfit.
            chi2_more = float(n_data - n_params_more) + c["c_floor"]
            chi2_less = chi2_more + dchi2
            pval, fstat, _ = calculate_chi_squared_improvement(
                chi2_less, chi2_more, DELTA_K, n_data, n_params_more
            )
            ps.append(float(pval))
            fs.append(float(fstat))
        out["pvalue"][name] = ps
        out["fstat"][name] = fs
        out["penalized_accept"][name] = bool(dchi2 > bar)
    return out


# ---------------------------------------------------------------------------
# 2. sigma_eff localizes the gate: irreducible misfit under a bright line is
#    discounted; a real line over quiet bins keeps its evidence.
# ---------------------------------------------------------------------------
def sigma_eff_localization() -> Dict[str, Any]:
    """A candidate of fixed raw evidence is discounted under a bright component.

    Two candidates carry the *same* raw chi^2 evidence. One sits where the model
    is small (a real missing line); the other sits under a bright fitted line's
    core, where it is absorbing the irreducible lineshape floor. Scoring the gate
    against ``sigma_eff^2 = sigma^2 + (kappa |model|)^2`` leaves the first intact
    and crushes the second below the accept bar.
    """
    kappa = float(DEFAULT_GATE_SIGMA_EFF_KAPPA)
    bar = _bar()
    sigma = 1.0
    m = 9  # a few bins around the candidate
    # A common candidate residual bump (3 informative bins) with raw chi^2 well
    # above the bar.
    bump: np.ndarray = np.zeros(m, dtype=complex)
    amp = 4.0
    bump[m // 2 - 1 : m // 2 + 2] = amp  # 3 bins
    raw = float(sigma_eff_chi2(bump, sigma, np.zeros(m, dtype=complex), 0.0))

    # Case A: a real line -- the K+1 model is small where the candidate sits.
    model_quiet: np.ndarray = np.zeros(m, dtype=complex)
    eff_real = float(sigma_eff_chi2(bump, sigma, model_quiet, kappa))

    # Case B: an absorber -- the candidate sits under a bright fitted line's core
    # (on-line |model|/sigma = snr_bright).
    snr_bright = 400.0
    model_bright: np.ndarray = np.zeros(m, dtype=complex)
    model_bright[m // 2 - 1 : m // 2 + 2] = snr_bright * sigma
    eff_absorber = float(sigma_eff_chi2(bump, sigma, model_bright, kappa))

    return {
        "kappa": kappa,
        "bar": bar,
        "snr_bright": snr_bright,
        "raw_chi2": raw,
        "eff_chi2_real": eff_real,
        "eff_chi2_absorber": eff_absorber,
        "accept_real": bool(eff_real > bar),
        "accept_absorber": bool(eff_absorber > bar),
    }


# ---------------------------------------------------------------------------
# 3. Blend recovery vs detectability.
# ---------------------------------------------------------------------------
def blend_study(seed: int = SEED) -> Dict[str, Any]:
    """Sweep a two-line blend over separation in line widths.

    For each separation, a 1:1 in-phase blend at on-line SNR ``snr`` is
    synthesized and (a) fit jointly with two lines started at the true positions
    -- the K-known recovery error -- and (b) fit with a single line -- the
    detectability signature (an elevated single-cosine reduced chi^2). Recovery
    holds to ~kHz down to about half a line width; the single-cosine elevation is
    large and smooth across the whole range.
    """
    rng = np.random.default_rng(seed)
    fwhm = feature_fwhm(TAU_US, T_ACTIVE_US, shape="lorentzian")
    seps_fwhm = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    snr = 120.0
    n_trials = 12

    half_span = 3.0  # MHz; window half-width about the blend center
    u = np.arange(-half_span, half_span + BIN_MHZ, BIN_MHZ)
    # On-line response of one line is 0.5 * A * tau_eff; set sigma=1 and choose A
    # so that |X(0)| = snr.
    tau_eff = effective_tau(TAU_US, T_ACTIVE_US)
    amp = 2.0 * snr / tau_eff
    sigma = 1.0

    out: Dict[str, Any] = {
        "fwhm_mhz": float(fwhm),
        "snr": snr,
        "sep_fwhm": seps_fwhm,
        "recovery_max_err_khz": [],
        "single_cosine_rchi2": [],
        "two_line_rchi2": [],
    }
    for sf in seps_fwhm:
        sep = sf * fwhm
        truth = [
            ModelPeak(amplitude=amp, offset_mhz=-sep / 2.0, phase=0.0),
            ModelPeak(amplitude=amp, offset_mhz=+sep / 2.0, phase=0.0),
        ]
        clean = model_spectrum(u, truth, TAU_US, T_ACTIVE_US, shape="lorentzian")
        errs: List[float] = []
        r2_one: List[float] = []
        r2_two: List[float] = []
        for _ in range(n_trials):
            data = clean + _complex_noise(u.size, sigma, rng)
            # (a) K-known recovery: start at the true positions.
            two = fit_window(u, data, sigma, truth, TAU_US, T_ACTIVE_US, fit_tau=False)
            if two.success and len(two.peaks) == 2:
                rec = sorted(pk.offset_mhz for pk in two.peaks)
                err = max(abs(rec[0] - (-sep / 2.0)), abs(rec[1] - (sep / 2.0)))
                errs.append(err * 1e3)  # kHz
                r2_two.append(float(two.reduced_chi2))
            # (b) single-cosine detectability: one line at the centroid.
            one = fit_window(
                u,
                data,
                sigma,
                [ModelPeak(amplitude=amp, offset_mhz=0.0, phase=0.0)],
                TAU_US,
                T_ACTIVE_US,
                fit_tau=False,
            )
            if one.success:
                r2_one.append(float(one.reduced_chi2))
        out["recovery_max_err_khz"].append(
            round(float(np.median(errs)), 4) if errs else None
        )
        out["single_cosine_rchi2"].append(
            round(float(np.median(r2_one)), 3) if r2_one else None
        )
        out["two_line_rchi2"].append(
            round(float(np.median(r2_two)), 3) if r2_two else None
        )
    return out


# ---------------------------------------------------------------------------
# 4. SNR-aware health gate across the seven fixtures (snapshot).
# ---------------------------------------------------------------------------
def _build_fixture(fixture: str, workdir: Path) -> str:
    """Build one checked-in fixture fresh through Stage 5 (no persisted knobs)."""
    import ftmwpipeline.api as ftmw

    workdir.mkdir(parents=True, exist_ok=True)
    path = str(workdir / f"exp_{fixture}.ftmw")
    ftmw.import_data(path, source=str(EXAMPLES / fixture), force=True)
    ftmw.detect_start_time(path, band=TRIM, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)
    ftmw.calibrate_tau(path)
    ftmw.detect_peaks(path)
    ftmw.assign_windows(path)
    ftmw.fit_peaks(path)
    return path


def cross_fixture(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Build all seven fixtures and score the shipped SNR-aware health gate."""
    import tempfile

    from ftmwpipeline._internal.stage5_validation_impl import (
        validate_stage5_shape_error_impl,
    )

    tmp = workdir or Path(tempfile.mkdtemp(prefix="stage5-note-"))
    out: Dict[str, Any] = {
        "kappa": DEFAULT_SHAPE_ERROR_KAPPA,
        "noise_floor": DEFAULT_CHI2R_NOISE_FLOOR,
        "fixtures": {},
        "windows": {"snr_max": [], "reduced_chi2": [], "fixture": [], "pass": []},
    }
    cat = str(CATALOG) if CATALOG.exists() else None
    for fx in FIXTURES:
        path = _build_fixture(fx, tmp / fx)
        gt = cat if fx in CATALOG_FIXTURES else None
        rep = validate_stage5_shape_error_impl(path, ground_truth=gt)
        t1 = rep["tier1"]
        t3 = rep.get("tier3")
        rec = {
            "n_windows": t1.get("n_windows"),
            "pass_rate": t1.get("pass_rate"),
            "chi2r_median": t1.get("chi2r_median"),
            "chi2r_p95": t1.get("chi2r_p95"),
            "chi2r_max": t1.get("chi2r_max"),
            "snr_bins": t1.get("snr_bins"),
        }
        if t3 is not None:
            rec["recall"] = t3.get("recall")
            rec["n_catalog"] = t3.get("n_catalog")
            rec["n_matched"] = t3.get("n_matched")
            rec["accuracy_floor"] = t3.get("accuracy_floor")
        out["fixtures"][fx] = rec
        for w in t1.get("windows", []):
            c = w["reduced_chi2"]
            if not np.isfinite(c):
                continue
            out["windows"]["snr_max"].append(round(float(w["snr_max"]), 3))
            out["windows"]["reduced_chi2"].append(round(float(c), 4))
            out["windows"]["fixture"].append(fx)
            out["windows"]["pass"].append(bool(w["pass"]))
        print(
            f"  {fx}: windows={rec['n_windows']} pass={rec['pass_rate']} "
            f"chi2r_med={rec['chi2r_median']} "
            f"recall={rec.get('recall', '-')}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figdir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(results: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        AGGIE_GOLD,
        BRAND_CYCLE,
        DOUBLE_DECKER,
        aggie_blue_cmap,
        apply_bare_style,
        apply_color_cycle,
    )

    apply_color_cycle(matplotlib)
    figdir = _figdir()

    # fig1: F-test p-value vs window padding; the penalized verdict is flat.
    inv = results["gate_invariance"]
    n_data = np.array(inv["n_data"], float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        n_data,
        inv["pvalue"]["absorber"],
        "o-",
        color=DOUBLE_DECKER,
        lw=1.8,
        label="F-test p (lineshape absorber)",
    )
    ax.plot(
        n_data,
        inv["pvalue"]["real"],
        "s-",
        color=AGGIE_BLUE,
        lw=1.8,
        label="F-test p (real line)",
    )
    ax.axhline(0.05, color="#999999", ls=":", lw=1.0, label=r"$\alpha=0.05$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("window size (real residual elements)")
    ax.set_ylabel("F-test p-value")
    apply_bare_style(ax)
    verdict = (
        f"penalized gate (flat in window size):\n"
        f"reject absorber, accept real\n"
        f"bar $2\\lambda\\Delta k={inv['bar']:.0f}$"
    )
    ax.text(
        0.97,
        0.46,
        verdict,
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="center",
        color=AGGIE_BLUE,
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_gate_window_invariance.png", dpi=130)
    plt.close(fig)

    # fig2: sigma_eff localization -- raw vs effective evidence, by location.
    se = results["sigma_eff"]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["real line\n(quiet bins)", "absorber\n(under bright line)"]
    raw = [se["raw_chi2"], se["raw_chi2"]]
    eff = [se["eff_chi2_real"], se["eff_chi2_absorber"]]
    x = np.arange(2)
    w = 0.38
    ax.bar(x - w / 2, raw, w, color=AGGIE_GOLD, label=r"raw $\Delta\chi^2$")
    ax.bar(
        x + w / 2,
        eff,
        w,
        color=AGGIE_BLUE,
        label=r"$\sigma_\mathrm{eff}$-weighted $\Delta\chi^2$",
    )
    ax.axhline(
        se["bar"],
        color=DOUBLE_DECKER,
        ls="--",
        lw=1.2,
        label=rf"accept bar $={se['bar']:.0f}$",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta\chi^2$")
    apply_bare_style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_sigma_eff_localization.png", dpi=130)
    plt.close(fig)

    # fig3: blend recovery (top) and single-cosine detectability (bottom).
    bl = results["blend"]
    sep = np.array(bl["sep_fwhm"], float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(
        sep,
        bl["recovery_max_err_khz"],
        "o-",
        color=AGGIE_BLUE,
        lw=1.8,
    )
    ax1.axvline(0.5, color="#999999", ls=":", lw=1.0)
    ax1.set_ylabel("recovery error (kHz)")
    ax1.set_yscale("log")
    apply_bare_style(ax1)
    ax2.plot(
        sep,
        bl["single_cosine_rchi2"],
        "s-",
        color=DOUBLE_DECKER,
        lw=1.8,
        label="single cosine (K=1)",
    )
    ax2.plot(
        sep,
        bl["two_line_rchi2"],
        "o-",
        color=AGGIE_BLUE,
        lw=1.8,
        label="joint two-line (K=2)",
    )
    ax2.axhline(1.0, color="#999999", ls=":", lw=1.0)
    ax2.axvline(0.5, color="#999999", ls=":", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("blend separation (line widths, FWHM)")
    ax2.set_ylabel(r"reduced $\chi^2$")
    apply_bare_style(ax2)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_blend_recovery_detectability.png", dpi=130)
    plt.close(fig)

    # fig4: SNR-aware health gate across fixtures (snapshot).
    if "cross_fixture" in results:
        cf = results["cross_fixture"]
        w = cf["windows"]
        snr = np.array(w["snr_max"], float)
        chi = np.array(w["reduced_chi2"], float)
        fxs = np.array(w["fixture"])
        kappa = float(cf["kappa"])
        floor = float(cf["noise_floor"])
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for fx, color in zip(FIXTURES, BRAND_CYCLE):
            sel = fxs == fx
            if not np.any(sel):
                continue
            ax.scatter(
                snr[sel],
                np.clip(chi[sel], 1e-2, None),
                s=10,
                color=color,
                alpha=0.55,
                edgecolors="none",
                label=fx,
            )
        xs = np.logspace(0, np.log10(max(snr.max(), 10.0)), 200)
        ax.plot(
            xs,
            floor + (kappa * xs) ** 2,
            color=AGGIE_BLUE,
            lw=2.0,
            label=rf"$F+(\kappa\,\mathrm{{SNR}})^2$, $\kappa={kappa:g}$",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"per-window max SNR")
        ax.set_ylabel(r"reduced $\chi^2$")
        apply_bare_style(ax)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(figdir / "fig4_snr_aware_gate.png", dpi=130)
        plt.close(fig)
    # touch the cmap import so lint does not flag it on the no-cross-fixture path
    _ = aggie_blue_cmap


def run(do_cross_fixture: bool = True, figures: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    results["gate_invariance"] = gate_window_invariance()
    results["sigma_eff"] = sigma_eff_localization()
    results["blend"] = blend_study()
    inv = results["gate_invariance"]
    print(
        f"gate bar 2*lambda*dk = {inv['bar']:.0f}; "
        f"F-test p(absorber) {inv['pvalue']['absorber'][0]:.3g} -> "
        f"{inv['pvalue']['absorber'][-1]:.3g} as the window pads",
        flush=True,
    )
    se = results["sigma_eff"]
    print(
        f"sigma_eff: real {se['eff_chi2_real']:.1f} > bar, "
        f"absorber {se['eff_chi2_absorber']:.2f} << bar {se['bar']:.0f}",
        flush=True,
    )
    if do_cross_fixture:
        print("cross-fixture (building 7 fixtures through Stage 5):", flush=True)
        results["cross_fixture"] = cross_fixture()
    if figures:
        make_figures(results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cross-fixture", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "results.json")
    )
    args = ap.parse_args()
    results = run(
        do_cross_fixture=not args.no_cross_fixture, figures=not args.no_figures
    )
    if not args.no_cross_fixture:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
