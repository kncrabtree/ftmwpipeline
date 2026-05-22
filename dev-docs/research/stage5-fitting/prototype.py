"""
Stage 5 per-window fitting — research prototype.

Reproducibility script for the report in ``report.md``. It establishes the
fitting model and its conventions, then probes the failure modes that the
production Stage 5 algorithm must survive — above all, strong blended lines.

Run from the repo root via the project conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage5-fitting/prototype.py

Outputs land in ``figures/`` alongside this script. No external fixture is
needed: every spectrum here is synthetic, built analytically on an rfft grid
from the finite-T damped-cosine model so the ground truth is exact.

What this prototype investigates (task 1 of dev-docs/planning/stage5-fitting.md)
-------------------------------------------------------------------------------
A. The model. ``h_T`` is the closed-form complex FFT of a finite-T damped
   cosine; fig 01 confirms it against a literal numerical FFT of a synthesized
   FID. The demodulation/sideband mapping is derived in the module docstring
   below and exercised in fig 02 — using the wrong sideband sign conjugates
   every leakage skirt and biases the fit.
B. The fitter. A complex-domain least-squares core and a conservative
   add-one-peak loop (F-test + AIC, peak-separation constraint, a patience
   parameter for the underfitting failure mode).
C. Blending. The central investigation: blend recovery as a function of
   separation (down to ~0.5 FWHM), relative phase, and intensity ratio
   (including a 3:5:1 nitrogen-quadrupole-style triplet).
D. Fixed-contributor skirt fidelity. If a "strong line" carried frozen into a
   neighbour window is itself an unrecognised blend, its mis-estimated skirt
   biases the neighbour's weak lines. Fig 04 quantifies that bias.
E. Validation. Fig 05 shows the two views the production loop must expose:
   the add-one-peak audit trail (every F-test decision) and the knockout test
   (remove a fitted line, confirm the residual grows and reacquires it).

Math conventions
----------------
A molecular line is a damped cosine in the FID, active over [t0, t0+T] of a
longer zero-padded record. In *baseband* (scope) frequency offset Δf from line
centre its rfft-domain response is

    X(Δf) ≈ (A/2) · e^{iφ} · h_T(Δf; τ)
    h_T(Δf; τ) = [1 - exp(-(1/τ + i 2π Δf) T)] / (1/τ + i 2π Δf)

(the active-region turn-on ramp exp(-i 2π f_bb t0) is dropped here — the
prototype works in the de-ramped frame throughout, as production Stage 5 does;
see ``deramp`` and report §2).

Sideband mapping. The persisted spectrum is on a *molecular* grid f; the line
sits at baseband frequency f_bb = s·(f - f_probe), with s = -1 for the lower
sideband (2638) and s = +1 for the upper. A window is fit about a reference
molecular frequency f_c; every peak j is fit by its signed baseband offset
δ_j = s·(f_j - f_c), and the window grid is converted to u = s·(f - f_c). The
model is Σ_j (A_j/2) e^{iφ_j} h_T(u - δ_j; τ). Because h_T(-Δf) = conj(h_T(Δf))
exactly, using the wrong s conjugates the model — fig 02.

Noise convention: each spectrum bin carries N(0, σ/√2) + i N(0, σ/√2), so
E[|n|²] = σ². σ is the per-bin complex RMS, matching the per-point noise the
noise-estimation stage reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

# ----- physical defaults broadly matching the 2638 fixture -----------------
T_US = 12.65            # active acquisition length (15 us record - 2.35 us turn-on)
TAU_US = 5.0            # effective decay constant (expf_us = 5.0 dominates)
DF_MHZ = 0.0122         # FT bin spacing (~12 kHz: 2638 zpf=2 grid)
PROBE_MHZ = 40960.0     # 2638 probe/LO frequency
RNG_SEED = 20260521

HERE = Path(__file__).parent
FIGDIR = HERE / "figures"


# ---------------------------------------------------------------------------
# A. The model
# ---------------------------------------------------------------------------
def h_T(delta_f_mhz: np.ndarray, tau_us: float, T_us: float = T_US) -> np.ndarray:
    """Closed-form complex FFT of a finite-T damped cosine.

    ``h_T(Δf; τ)`` on a baseband frequency-offset grid Δf (MHz). This is the
    exact rfft-domain line shape: at centre ``h_T(0) = τ_eff = τ(1-e^{-T/τ})``,
    and far from centre the magnitude decays as the 1/|Δf| truncation skirt.

    Worked entirely in µs/MHz: a frequency in MHz times a time in µs is
    dimensionless, so ``h_T`` carries units of µs and ``h_T(0)`` is τ_eff in µs.
    """
    df_mhz = np.asarray(delta_f_mhz, dtype=float)
    denom = (1.0 / tau_us) + 1j * 2.0 * np.pi * df_mhz          # 1/µs
    return (1.0 - np.exp(-denom * T_us)) / denom                # µs


def h_T_jac(
    delta_f_mhz: np.ndarray, tau_us: float, T_us: float = T_US
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic derivatives of ``h_T`` w.r.t. Δf (MHz) and τ (µs).

    Returns ``(dh/d(Δf_MHz), dh/d(τ_us))``. Used to validate the production
    Jacobian against finite differences.
    """
    df_mhz = np.asarray(delta_f_mhz, dtype=float)
    denom = (1.0 / tau_us) + 1j * 2.0 * np.pi * df_mhz
    e = np.exp(-denom * T_us)
    # h = (1 - e) / denom ; d/dz [(1-e)/z] with e = exp(-zT):
    #   dh/dz = (T e)/z - (1 - e)/z^2
    dh_dz = (T_us * e) / denom - (1.0 - e) / denom**2
    # z = 1/τ + i2π Δf : dz/d(Δf_MHz) = i2π,  dz/d(τ_us) = -1/τ².
    dh_ddf_mhz = dh_dz * (1j * 2.0 * np.pi)
    dh_dtau_us = dh_dz * (-1.0 / tau_us**2)
    return dh_ddf_mhz, dh_dtau_us


def numerical_fft_response(
    delta_f_mhz: np.ndarray,
    tau_us: float,
    T_us: float = T_US,
    fs_mhz: float = 50.0,
) -> np.ndarray:
    """Build a damped-cosine FID, rfft it, and return the line shape near centre.

    A literal numerical realisation of the model: synthesize the time-domain
    damped cosine over [0, T], rfft the zero-padded record, and read the bins
    near the line. Returned interpolated onto ``delta_f_mhz`` for direct
    comparison with the closed-form :func:`h_T`. Confirms h_T *is* the FFT of
    the model (fig 01).
    """
    dt_us = 1.0 / fs_mhz                       # sample spacing in µs
    n_active = int(round(T_us / dt_us))
    # Park the line at a baseband frequency well inside Nyquist.
    f0_mhz = 5.0
    n_pad = 1 << (int(np.log2(n_active)) + 4)  # generous zero-pad: dense grid
    t = np.arange(n_active) * dt_us            # µs
    fid = np.cos(2.0 * np.pi * f0_mhz * t) * np.exp(-t / tau_us)
    rec = np.zeros(n_pad)
    rec[:n_active] = fid
    spec = np.fft.rfft(rec) * dt_us  # dt factor -> continuous-transform units (µs)
    freqs_mhz = np.fft.rfftfreq(n_pad, d=dt_us)
    # Response near +f0, as a function of offset; phase 0, amp 1 => X = 0.5 h_T.
    offs_mhz = freqs_mhz - f0_mhz
    keep = np.abs(offs_mhz) <= float(np.max(np.abs(delta_f_mhz))) + 1.0
    interp_re = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).real)
    interp_im = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).imag)
    return interp_re + 1j * interp_im


def deramp(freqs_mhz: np.ndarray, X: np.ndarray, t0_us: float) -> np.ndarray:
    """Reference a spectrum to the active-region turn-on (prototype copy of
    ``preprocessing.leakage.deramp_to_active_start``)."""
    return X * np.exp(2j * np.pi * np.abs(freqs_mhz) * t0_us)


@dataclass
class Peak:
    """A line in baseband-offset coordinates: amplitude, signed offset, phase."""

    amp: float
    offset_mhz: float
    phase: float


def model_spectrum(u_mhz: np.ndarray, peaks: list[Peak], tau_us: float) -> np.ndarray:
    """Sum of finite-T damped-cosine responses on a baseband-offset grid u."""
    X = np.zeros_like(u_mhz, dtype=np.complex128)
    for pk in peaks:
        X += 0.5 * pk.amp * np.exp(1j * pk.phase) * h_T(u_mhz - pk.offset_mhz, tau_us)
    return X


def make_window(
    u_mhz: np.ndarray,
    peaks: list[Peak],
    tau_us: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Analytic window spectrum (sum of h_T responses) plus complex Gaussian noise."""
    X = model_spectrum(u_mhz, peaks, tau_us)
    noise = rng.normal(0.0, sigma / np.sqrt(2.0), X.shape) + 1j * rng.normal(
        0.0, sigma / np.sqrt(2.0), X.shape
    )
    return X + noise


def window_grid(half_width_mhz: float = 2.0, df_mhz: float = DF_MHZ) -> np.ndarray:
    """Symmetric baseband-offset grid u (MHz) at the FT bin spacing."""
    n = int(round(half_width_mhz / df_mhz))
    return np.arange(-n, n + 1) * df_mhz


def feature_fwhm_mhz(tau_us: float, T_us: float = T_US) -> float:
    """FWHM (MHz) of the magnitude line shape |h_T| — the resolution scale."""
    u = np.linspace(-1.0, 1.0, 200001)
    mag = np.abs(h_T(u, tau_us, T_us))
    half = mag.max() / 2.0
    above = np.where(mag >= half)[0]
    return float(u[above[-1]] - u[above[0]])


def tau_eff_us(tau_us: float, T_us: float = T_US) -> float:
    """Effective on-line gain τ_eff = τ(1 - e^{-T/τ}) — sets the SNR scale."""
    return tau_us * (1.0 - np.exp(-T_us / tau_us))


def amp_for_snr(snr: float, tau_us: float, sigma: float) -> float:
    """Amplitude A whose on-line response (A/2)·τ_eff gives the requested SNR."""
    return 2.0 * snr * sigma / tau_eff_us(tau_us)


# ---------------------------------------------------------------------------
# B. The fitter
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """Outcome of a fixed-K least-squares window fit."""

    success: bool
    peaks: list[Peak]
    tau_us: float
    chi2: float                       # noise-weighted Σ (residual/σ)²
    n_data: int                       # 2·M stacked real points
    n_params: int
    cov: np.ndarray | None = None     # parameter covariance (param order below)

    @property
    def aic(self) -> float:
        """AIC = 2k + n·ln(χ²/n)."""
        if self.chi2 <= 0 or self.n_data <= 0:
            return np.inf
        return 2 * self.n_params + self.n_data * np.log(self.chi2 / self.n_data)

    @property
    def reduced_chi2(self) -> float:
        dof = self.n_data - self.n_params
        return self.chi2 / dof if dof > 0 else np.inf


def _pack(peaks: list[Peak], tau_us: float, fit_tau: bool) -> np.ndarray:
    p = []
    for pk in peaks:
        p += [pk.amp, pk.offset_mhz, pk.phase]
    if fit_tau:
        p.append(tau_us)
    return np.asarray(p, dtype=float)


def _unpack(
    p: np.ndarray, k: int, tau_fixed: float, fit_tau: bool
) -> tuple[list[Peak], float]:
    peaks = [Peak(p[3 * i], p[3 * i + 1], p[3 * i + 2]) for i in range(k)]
    tau = p[3 * k] if fit_tau else tau_fixed
    return peaks, tau


def fit_window(
    u_mhz: np.ndarray,
    z: np.ndarray,
    sigma: float,
    init_peaks: list[Peak],
    tau0_us: float,
    *,
    fit_tau: bool = True,
    tau_bounds: tuple[float, float] = (TAU_US / 5.0, TAU_US * 5.0),
) -> FitResult:
    """Complex-domain least-squares fit of a fixed number of lines.

    Residual = stacked Re/Im of (data - model)/σ. Parameter order per peak is
    ``(amp, offset_mhz, phase)``, optionally followed by a shared ``tau_us``.
    """
    k = len(init_peaks)
    u_lo, u_hi = float(u_mhz.min()), float(u_mhz.max())

    m = len(u_mhz)
    # The complex noise has per-bin RMS σ, so Re and Im each carry variance
    # σ²/2. Weighting the stacked Re/Im residual by σ/√2 makes every element
    # unit-variance, so χ² ~ n_data and the F-test is properly calibrated.
    sig_ri = sigma / np.sqrt(2.0)

    def residual(p: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(p, k, tau0_us, fit_tau)
        r = (z - model_spectrum(u_mhz, peaks, tau)) / sig_ri
        return np.concatenate([r.real, r.imag])

    def jac(p: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of the stacked residual. Column per parameter;
        rows [0:m] are Re, [m:2m] are Im. ∂r/∂p = -(∂model/∂p)/(σ/√2)."""
        peaks, tau = _unpack(p, k, tau0_us, fit_tau)
        J = np.zeros((2 * m, len(p)))
        dmodel_dtau = np.zeros(m, dtype=complex)
        for i, pk in enumerate(peaks):
            du = u_mhz - pk.offset_mhz
            H = h_T(du, tau)
            dHdf, dHdt = h_T_jac(du, tau)
            e = np.exp(1j * pk.phase)
            cols = {
                3 * i: 0.5 * e * H,                       # ∂model/∂A
                3 * i + 1: 0.5 * pk.amp * e * (-dHdf),    # ∂model/∂offset
                3 * i + 2: 0.5j * pk.amp * e * H,         # ∂model/∂phase
            }
            for col, dmodel in cols.items():
                g = -dmodel / sig_ri
                J[:m, col] = g.real
                J[m:, col] = g.imag
            dmodel_dtau += 0.5 * pk.amp * e * dHdt
        if fit_tau:
            g = -dmodel_dtau / sig_ri
            J[:m, 3 * k] = g.real
            J[m:, 3 * k] = g.imag
        return J

    lo, hi = [], []
    for _ in range(k):
        lo += [0.0, u_lo, -4.0 * np.pi]
        hi += [np.inf, u_hi, 4.0 * np.pi]
    if fit_tau:
        lo.append(tau_bounds[0])
        hi.append(tau_bounds[1])
    p0 = _pack(init_peaks, tau0_us, fit_tau)
    p0 = np.clip(p0, lo, hi)

    try:
        sol = least_squares(
            residual, p0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=400
        )
    except Exception:
        return FitResult(False, init_peaks, tau0_us, np.inf, 2 * len(u_mhz),
                         len(p0))

    peaks, tau = _unpack(sol.x, k, tau0_us, fit_tau)
    for pk in peaks:
        pk.phase = (pk.phase + np.pi) % (2.0 * np.pi) - np.pi
    chi2 = float(2.0 * sol.cost)  # least_squares cost = 0.5 Σ r²
    n_data = 2 * len(u_mhz)
    n_params = len(sol.x)

    cov = None
    try:
        jtj = sol.jac.T @ sol.jac
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        pass
    return FitResult(sol.success, peaks, tau, chi2, n_data, n_params, cov)


def _f_test_p(chi2_old: float, chi2_new: float, d_params: int, n_data: int,
              n_params_new: int) -> float:
    """p-value of the nested-model F-test (smaller p -> the added line helps)."""
    diff = chi2_old - chi2_new
    df_complex = n_data - n_params_new
    if diff <= 0 or d_params <= 0 or df_complex <= 0:
        return 1.0
    f_stat = (diff / d_params) / (chi2_new / df_complex)
    return float(1.0 - f_dist.cdf(f_stat, d_params, df_complex))


@dataclass
class AddStep:
    """One iteration of the conservative add-one-peak loop (the audit trail)."""

    n_before: int
    cand_offset_mhz: float
    chi2_before: float
    chi2_after: float
    p_value: float
    aic_before: float
    aic_after: float
    decision: str            # "accept" | "tentative" | "reject" | "promote"
    note: str = ""


@dataclass
class ConservativeResult:
    """Result of the conservative add-one-peak loop."""

    fit: FitResult
    accepted: list[Peak]
    trail: list[AddStep] = field(default_factory=list)


def conservative_fit(
    u_mhz: np.ndarray,
    z: np.ndarray,
    sigma: float,
    candidate_offsets: list[float],
    tau0_us: float,
    *,
    fit_tau: bool = True,
    significance: float = 0.05,
    min_separation_mhz: float = 0.0,
    max_peaks: int = 8,
    patience: int = 0,
) -> ConservativeResult:
    """Conservative incremental peak addition with F-test + AIC and patience.

    Strongest-residual-first. A candidate is accepted only if the χ² drop
    passes the F-test *and* the AIC decreases. ``patience`` is the number of
    consecutive rejections tolerated before the loop stops: ``patience=0`` is
    the strict loop — it halts at the first rejection. A rejected candidate is
    held *tentatively*; if a later candidate makes the accumulated batch
    significant against the last accepted model the whole batch is promoted —
    this is what rescues the underfitting case where peak N+1 alone looks
    insignificant but N+1 and N+2 together do not (fig 05).
    """
    remaining = sorted(
        candidate_offsets, key=lambda o: -np.abs(np.interp(o, u_mhz, np.abs(z)))
    )
    trail: list[AddStep] = []

    def init_peak(off: float, residual: np.ndarray) -> Peak:
        amp = 2.0 * np.abs(np.interp(off, u_mhz, np.abs(residual)))
        amp /= max(tau_eff_us(tau0_us), 1e-9)
        ph = float(np.interp(off, u_mhz, np.angle(residual)))
        return Peak(max(amp, 1e-6), off, ph)

    if not remaining:
        return ConservativeResult(
            FitResult(False, [], tau0_us, np.inf, 2 * len(u_mhz), 0), [], trail
        )

    seed = remaining.pop(0)
    fit = fit_window(u_mhz, z, sigma, [init_peak(seed, z)], tau0_us,
                     fit_tau=fit_tau)
    accepted = list(fit.peaks)
    tentative: list[Peak] = []

    consecutive_rejects = 0
    while remaining and len(accepted) + len(tentative) < max_peaks:
        resid = z - model_spectrum(u_mhz, accepted + tentative, fit.tau_us)
        cand = remaining.pop(0)
        too_close = any(
            abs(cand - pk.offset_mhz) < min_separation_mhz
            for pk in accepted + tentative
        )
        if too_close:
            trail.append(AddStep(len(accepted), cand, fit.chi2, fit.chi2, 1.0,
                                 fit.aic, fit.aic, "reject", "separation"))
            continue

        # Trial = accepted + the whole tentative batch + this candidate, tested
        # against the last *accepted* model (fit). A tentative batch is thus
        # judged as a unit, so peaks that are jointly but not individually
        # significant are still found.
        trial_init = accepted + tentative + [init_peak(cand, resid)]
        trial = fit_window(u_mhz, z, sigma, trial_init, fit.tau_us,
                           fit_tau=fit_tau)
        d_params = trial.n_params - fit.n_params
        p = _f_test_p(fit.chi2, trial.chi2, d_params, trial.n_data,
                      trial.n_params)
        passes = (p < significance) and (trial.aic < fit.aic)

        if passes:
            decision = "promote" if tentative else "accept"
            trail.append(AddStep(len(accepted), cand, fit.chi2, trial.chi2, p,
                                 fit.aic, trial.aic, decision,
                                 f"+{len(tentative)+1} lines"))
            accepted = list(trial.peaks)
            tentative = []
            fit = trial
            consecutive_rejects = 0
        else:
            trail.append(AddStep(len(accepted), cand, fit.chi2, trial.chi2, p,
                                 fit.aic, trial.aic, "tentative"))
            tentative.append(init_peak(cand, resid))
            consecutive_rejects += 1
            if consecutive_rejects > patience:
                break  # strict loop (patience=0) stops at the first rejection

    return ConservativeResult(fit, accepted, trail)


# ---------------------------------------------------------------------------
# Fig 01 — h_T closed form vs numerical FFT, and the Jacobian
# ---------------------------------------------------------------------------
def fig01_model_validation() -> dict:
    u = window_grid(2.0)
    analytic = h_T(u, TAU_US)
    numeric = numerical_fft_response(u, TAU_US)
    rel_err = np.abs(analytic - numeric) / np.abs(analytic).max()

    # Jacobian check: analytic vs central finite difference.
    eps_f, eps_t = 1e-6, 1e-4
    dh_df_a, dh_dt_a = h_T_jac(u, TAU_US)
    dh_df_n = (h_T(u + eps_f, TAU_US) - h_T(u - eps_f, TAU_US)) / (2 * eps_f)
    dh_dt_n = (h_T(u, TAU_US + eps_t) - h_T(u, TAU_US - eps_t)) / (2 * eps_t)
    jac_df_err = np.abs(dh_df_a - dh_df_n).max() / np.abs(dh_df_a).max()
    jac_dt_err = np.abs(dh_dt_a - dh_dt_n).max() / np.abs(dh_dt_a).max()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Fig 01 — finite-T model: closed form vs numerical FFT")
    axes[0, 0].plot(u, analytic.real, color="#022851", label="h_T closed form")
    axes[0, 0].plot(u, numeric.real, "--", color="#ffbf00", label="numerical FFT")
    axes[0, 0].set_title("real part"); axes[0, 0].legend()
    axes[0, 0].set_xlabel("baseband offset Δf (MHz)")
    axes[0, 1].plot(u, analytic.imag, color="#022851", label="h_T closed form")
    axes[0, 1].plot(u, numeric.imag, "--", color="#ffbf00", label="numerical FFT")
    axes[0, 1].set_title("imag part"); axes[0, 1].legend()
    axes[0, 1].set_xlabel("baseband offset Δf (MHz)")
    axes[1, 0].semilogy(u, np.maximum(rel_err, 1e-16), color="#c01230")
    axes[1, 0].set_title(f"closed-form vs FFT relative error (max {rel_err.max():.1e})")
    axes[1, 0].set_xlabel("baseband offset Δf (MHz)")
    axes[1, 1].plot(u, dh_df_a.real, color="#022851", label="∂h/∂Δf analytic")
    axes[1, 1].plot(u, dh_df_n.real, ":", color="#ffbf00", label="finite diff")
    axes[1, 1].set_title(
        f"Jacobian check — ∂/∂Δf err {jac_df_err:.1e}, ∂/∂τ err {jac_dt_err:.1e}"
    )
    axes[1, 1].legend(); axes[1, 1].set_xlabel("baseband offset Δf (MHz)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_model_validation.png", dpi=110)
    plt.close(fig)
    return {
        "model_rel_err": float(rel_err.max()),
        "jac_df_err": float(jac_df_err),
        "jac_dt_err": float(jac_dt_err),
        "fwhm_khz": feature_fwhm_mhz(TAU_US) * 1e3,
    }


# ---------------------------------------------------------------------------
# Fig 02 — the sideband sign
# ---------------------------------------------------------------------------
def fig02_sideband(rng: np.random.Generator) -> dict:
    """A line on each sideband, fit with the correct and the wrong sign s.

    The window spectrum is built on the molecular grid; the fitter is handed
    the grid converted to baseband offset with either the correct or the
    flipped sign. The wrong sign conjugates the model -> biased frequency and
    phase even though the magnitude can still look plausible.
    """
    tau, sigma, snr = TAU_US, 1.0, 60.0
    amp = amp_for_snr(snr, tau, sigma)
    fwhm = feature_fwhm_mhz(tau)
    true_off = 0.35      # signed baseband offset of the line from window centre
    true_phase = 1.1

    rows = []
    for sb_name, s in (("lower", -1.0), ("upper", +1.0)):
        f_c = PROBE_MHZ + s * 8.0           # window centre, molecular MHz
        f_line = f_c + s * true_off         # line, molecular MHz
        u_true = window_grid(2.0)           # baseband offset grid (truth frame)
        z = make_window(u_true, [Peak(amp, true_off, true_phase)], tau, sigma, rng)
        f_grid = f_c + s * u_true           # molecular grid the data lives on

        for sign_name, s_fit in (("correct", s), ("flipped", -s)):
            u_fit = s_fit * (f_grid - f_c)
            order = np.argsort(u_fit)
            res = conservative_fit(
                u_fit[order], z[order], sigma, [0.0, true_off, -true_off],
                tau, min_separation_mhz=0.5 * fwhm,
            )
            pk = min(res.accepted, key=lambda p: abs(p.offset_mhz - true_off)) \
                if res.accepted else Peak(0, 99, 0)
            f_fit = f_c + s_fit * pk.offset_mhz
            rows.append({
                "sideband": sb_name, "sign": sign_name,
                "freq_err_khz": (f_fit - f_line) * 1e3,
                "phase_err": (pk.phase - true_phase + np.pi) % (2 * np.pi) - np.pi,
                "n_peaks": len(res.accepted),
            })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig 02 — sideband sign: wrong s conjugates the model")
    labels = [f"{r['sideband']}\n{r['sign']}" for r in rows]
    ferr = [abs(r["freq_err_khz"]) for r in rows]
    perr = [abs(r["phase_err"]) for r in rows]
    colors = ["#266041" if r["sign"] == "correct" else "#c01230" for r in rows]
    axes[0].bar(labels, ferr, color=colors)
    axes[0].set_ylabel("|frequency error| (kHz)")
    axes[0].set_title("frequency recovery")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[1].bar(labels, perr, color=colors)
    axes[1].set_ylabel("|phase error| (rad)")
    axes[1].set_title("phase recovery")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_sideband_sign.png", dpi=110)
    plt.close(fig)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Fig 03 — the blending sweep
# ---------------------------------------------------------------------------
def fig03_blending(rng: np.random.Generator) -> dict:
    """Two-line blend: joint-fit recovery, and the 1-vs-2-line F-test.

    Two questions, kept separate because they have different answers:

    * **Recovery** (top row) — *given* that the window is fit with the true
      K=2, how accurately are the two frequencies recovered? Median over noise
      draws of the per-line offset error.
    * **Blend signature** (bottom row) — how badly does a *single* damped
      cosine fail to model the pair? The reduced χ² of the best 1-cosine fit.
      A single cosine has a fixed line shape, so a blend it cannot match
      leaves an elevated reduced χ² — the smooth, observable signal a
      blend-aware seeder triggers on. (The 1-vs-2-line F-test p-value is also
      recorded; it sits at the numerical floor for every separation here, i.e.
      the second line is *always* statistically required once the pair is fit
      with a proper K=2 initialisation — detectability is never the limit.)

    Swept over separation, relative phase, and intensity ratio (1:1, 3:1, and
    5:1 — the faint-component limit of the nitrogen-quadrupole 3:5:1 triplet).
    """
    tau, sigma = TAU_US, 1.0
    fwhm = feature_fwhm_mhz(tau)
    seps_fwhm = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])
    phase_cases = [("in phase", 0.0), ("quadrature", np.pi / 2), ("anti", np.pi)]
    ratio_cases = [("1:1", 1.0), ("3:1", 1.0 / 3.0), ("5:1", 0.2)]
    n_trials = 16
    snr_strong = 120.0

    results: dict = {}
    for ph_name, dphi in phase_cases:
        for ra_name, ratio in ratio_cases:
            freq_err, pvals, rchi2_1 = [], [], []
            for sep_fwhm in seps_fwhm:
                sep = sep_fwhm * fwhm
                a_strong = amp_for_snr(snr_strong, tau, sigma)
                a_weak = a_strong * ratio
                ferrs, ps, r1s = [], [], []
                for _ in range(n_trials):
                    true = [
                        Peak(a_strong, -sep / 2, 0.3),
                        Peak(a_weak, +sep / 2, 0.3 + dphi),
                    ]
                    u = window_grid(2.5)
                    z = make_window(u, true, tau, sigma, rng)
                    fit2 = fit_window(
                        u, z, sigma,
                        [Peak(a_strong, -sep / 2, 0.3),
                         Peak(a_weak, sep / 2, 0.3 + dphi)], tau,
                    )
                    fit1 = fit_window(
                        u, z, sigma,
                        [Peak(a_strong + a_weak, 0.0, 0.3)], tau,
                    )
                    ps.append(_f_test_p(
                        fit1.chi2, fit2.chi2,
                        fit2.n_params - fit1.n_params,
                        fit2.n_data, fit2.n_params,
                    ))
                    r1s.append(fit1.reduced_chi2)
                    if fit2.success and len(fit2.peaks) == 2:
                        matched = _match(fit2.peaks, true)
                        ferrs.append(
                            np.median([abs(m.offset_mhz - t.offset_mhz)
                                       for m, t in matched]) * 1e3
                        )
                freq_err.append(np.median(ferrs) if ferrs else np.nan)
                pvals.append(float(np.median(ps)))
                rchi2_1.append(float(np.median(r1s)))
            results[(ph_name, ra_name)] = {
                "sep_fwhm": seps_fwhm, "freq_err_khz": np.array(freq_err),
                "pval": np.array(pvals), "rchi2_1": np.array(rchi2_1),
            }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Fig 03 — blend recovery and detectability vs separation, phase, "
        f"intensity ratio (FWHM ≈ {fwhm*1e3:.0f} kHz, strong line SNR "
        f"{snr_strong:.0f})"
    )
    for col, (ph_name, _) in enumerate(phase_cases):
        for ra_name, _ in ratio_cases:
            r = results[(ph_name, ra_name)]
            axes[0, col].plot(r["sep_fwhm"], r["freq_err_khz"], "o-", label=ra_name)
            axes[1, col].plot(r["sep_fwhm"], r["rchi2_1"], "o-", label=ra_name)
        axes[0, col].set_title(f"relative phase: {ph_name}")
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].axhline(1.0, color="#022851", ls="--",
                             label="reduced χ² = 1 (good single fit)")
        axes[1, col].set_xlabel("separation (units of FWHM)")
        for row in (0, 1):
            axes[row, col].axvline(0.5, color="#888", ls=":")
    axes[0, 0].set_ylabel("K=2 fit: median |frequency error| (kHz)")
    axes[1, 0].set_ylabel("single-cosine fit: reduced χ²")
    axes[0, 0].legend(title="intensity ratio", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_blending_sweep.png", dpi=110)
    plt.close(fig)
    return {"fwhm_khz": fwhm * 1e3, "results": results}


def _match(fitted: list[Peak], true: list[Peak]) -> list[tuple[Peak, Peak]]:
    """Greedy nearest-offset pairing of fitted peaks to true peaks."""
    out, used = [], set()
    for t in true:
        best, bi = None, -1
        for i, fpk in enumerate(fitted):
            if i in used:
                continue
            if best is None or abs(fpk.offset_mhz - t.offset_mhz) < abs(
                best.offset_mhz - t.offset_mhz
            ):
                best, bi = fpk, i
        if best is not None:
            used.add(bi)
            out.append((best, t))
    return out


# ---------------------------------------------------------------------------
# Fig 04 — fixed-contributor skirt corruption
# ---------------------------------------------------------------------------
def fig04_fixed_contributor(rng: np.random.Generator) -> dict:
    """A strong "contributor" that is secretly a blend, frozen into a neighbour.

    The contributor sits out of band; in the neighbour window a weak line is
    fit while the contributor's skirt is carried frozen. We compare carrying
    the *true* (2-line) skirt against the *mis-fit* (1-line) skirt, as the
    contributor's own blend separation is swept. Mis-estimating the skirt
    biases the weak line — the failure mode that breaks the fixed-contributor
    model.
    """
    tau, sigma = TAU_US, 1.0
    fwhm = feature_fwhm_mhz(tau)
    seps_fwhm = np.array([0.0, 0.5, 0.8, 1.2, 2.0])
    n_trials = 20
    snr_contrib, snr_weak = 250.0, 12.0
    contrib_gap = 8.0          # MHz from weak window centre to the contributor
    a_c = amp_for_snr(snr_contrib, tau, sigma)
    a_w = amp_for_snr(snr_weak, tau, sigma)
    u = window_grid(1.0)       # the (narrow) weak-line window

    bias_true, bias_misfit = [], []
    for sep_fwhm in seps_fwhm:
        sep = sep_fwhm * fwhm
        # The contributor is really two lines (a blend) near -contrib_gap.
        c1 = Peak(a_c * 0.6, -contrib_gap - sep / 2, 0.7)
        c2 = Peak(a_c * 0.4, -contrib_gap + sep / 2, 2.0)
        bt, bm = [], []
        for _ in range(n_trials):
            weak = Peak(a_w, 0.05, 1.4)
            z = make_window(u, [weak, c1, c2], tau, sigma, rng)
            # (a) carry the TRUE two-line skirt frozen.
            true_skirt = model_spectrum(u, [c1, c2], tau)
            ft = fit_window(u, z - true_skirt, sigma, [Peak(a_w, 0.05, 1.4)],
                            tau, fit_tau=False)
            # (b) fit the contributor as ONE line, carry that skirt frozen.
            uc = window_grid(2.5)
            zc = make_window(uc + (-contrib_gap), [c1, c2], tau, sigma, rng)
            # fit a single cosine to the blended contributor
            one = fit_window(uc, zc, sigma,
                             [Peak(a_c, 0.0, 1.0)], tau, fit_tau=False)
            misfit_peak = Peak(one.peaks[0].amp,
                               one.peaks[0].offset_mhz - contrib_gap,
                               one.peaks[0].phase) if one.peaks else c1
            misfit_skirt = model_spectrum(u, [misfit_peak], tau)
            fm = fit_window(u, z - misfit_skirt, sigma,
                            [Peak(a_w, 0.05, 1.4)], tau, fit_tau=False)
            if ft.peaks:
                bt.append(abs(ft.peaks[0].offset_mhz - weak.offset_mhz) * 1e3)
            if fm.peaks:
                bm.append(abs(fm.peaks[0].offset_mhz - weak.offset_mhz) * 1e3)
        bias_true.append(np.median(bt) if bt else np.nan)
        bias_misfit.append(np.median(bm) if bm else np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(seps_fwhm, bias_true, "o-", color="#266041",
            label="true 2-line skirt carried frozen")
    ax.plot(seps_fwhm, bias_misfit, "s-", color="#c01230",
            label="contributor mis-fit as 1 line")
    ax.axvline(0.5, color="#888", ls=":")
    ax.set_xlabel("contributor's own blend separation (units of FWHM)")
    ax.set_ylabel("median weak-line |frequency error| (kHz)")
    ax.set_title(
        "Fig 04 — fixed-contributor skirt fidelity\n"
        "an unrecognised blend in the frozen contributor biases the weak line"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_fixed_contributor.png", dpi=110)
    plt.close(fig)
    return {
        "seps_fwhm": seps_fwhm.tolist(),
        "bias_true_khz": [float(x) for x in bias_true],
        "bias_misfit_khz": [float(x) for x in bias_misfit],
    }


# ---------------------------------------------------------------------------
# Fig 05 — the add-one-peak audit trail and the knockout test
# ---------------------------------------------------------------------------
def _chi2(u: np.ndarray, z: np.ndarray, sigma: float,
          peaks: list[Peak], tau: float) -> float:
    """Noise-weighted χ² of a model against window data (unit-variance stack)."""
    r = (z - model_spectrum(u, peaks, tau)) / (sigma / np.sqrt(2.0))
    return float(np.sum(r.real**2 + r.imag**2))


def fig05_audit_knockout(rng: np.random.Generator) -> dict:
    """Two validation views the production loop must expose.

    Left — the **audit trail**: the conservative add-one-peak loop on a
    representative multi-line window, every iteration's χ² drop and F-test
    p-value with its accept/reject decision, so the decision flow can be
    inspected and signed off.

    Right — the **knockout test**: with the converged fit in hand, each line
    is removed in turn while every other parameter stays frozen. A genuinely
    supported line leaves a large χ² increase and a residual that reacquires
    that line's shape at its frequency; the panel overlays the flat full-fit
    residual against a single-line knockout to show the absent peak
    reappearing.
    """
    tau, sigma = TAU_US, 1.0
    fwhm = feature_fwhm_mhz(tau)
    true = [
        Peak(amp_for_snr(180.0, tau, sigma), -0.85, 0.3),
        Peak(amp_for_snr(22.0, tau, sigma), -0.10, 2.1),
        Peak(amp_for_snr(45.0, tau, sigma), 0.55, 5.0),
        Peak(amp_for_snr(13.0, tau, sigma), 1.05, 1.2),
    ]
    u = window_grid(2.0)
    z = make_window(u, true, tau, sigma, rng)
    cands = [p.offset_mhz for p in true]
    res = conservative_fit(u, z, sigma, cands, tau,
                           min_separation_mhz=0.3 * fwhm, patience=1)
    fit = res.fit
    peaks = sorted(fit.peaks, key=lambda p: p.offset_mhz)

    # Knockout: drop each line, keep the rest frozen, measure the χ² increase.
    chi_full = _chi2(u, z, sigma, peaks, fit.tau_us)
    knock = []
    for i in range(len(peaks)):
        kept = [p for j, p in enumerate(peaks) if j != i]
        knock.append({
            "offset": peaks[i].offset_mhz,
            "dchi2": _chi2(u, z, sigma, kept, fit.tau_us) - chi_full,
        })

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Fig 05 — conservative-loop audit trail and knockout validation")

    # --- left: audit trail ------------------------------------------------
    ax = axes[0]
    steps = res.trail
    if steps:
        cols = {"accept": "#266041", "promote": "#266041",
                "tentative": "#ffbf00", "reject": "#c01230"}
        pv = [max(s.p_value, 1e-300) for s in steps]
        ax.bar(range(len(steps)), pv, color=[cols[s.decision] for s in steps])
        ax.axhline(0.05, color="#022851", ls="--", label="significance 0.05")
        ax.set_yscale("log")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(
            [f"+{s.cand_offset_mhz:+.2f}\n{s.decision}" for s in steps],
            fontsize=8,
        )
        ax.set_ylabel("F-test p-value (added line)")
    ax.set_title(
        f"add-one-peak audit trail — {len(res.accepted)} lines accepted "
        f"(true: {len(true)})"
    )
    ax.legend(fontsize=8)

    # --- right: knockout --------------------------------------------------
    ax = axes[1]
    full_resid = np.abs(z - model_spectrum(u, peaks, fit.tau_us))
    # Knock out the line with the median χ² increase for a legible bump.
    drop = int(np.argsort([k["dchi2"] for k in knock])[len(knock) // 2])
    kept = [p for j, p in enumerate(peaks) if j != drop]
    knock_resid = np.abs(z - model_spectrum(u, kept, fit.tau_us))
    ax.plot(u, full_resid, color="#266041", lw=1.0,
            label="full-fit residual (flat, ~noise)")
    ax.plot(u, knock_resid, color="#c01230", lw=1.0,
            label=f"line at {peaks[drop].offset_mhz:+.2f} MHz knocked out")
    ax.axvline(peaks[drop].offset_mhz, color="#888", ls=":")
    ax.set_xlabel("baseband offset (MHz)")
    ax.set_ylabel("|residual|")
    ax.set_title(
        "knockout test — the absent line reappears in the residual\n"
        + ", ".join(f"Δχ²={k['dchi2']:.0f}@{k['offset']:+.2f}" for k in knock)
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_audit_knockout.png", dpi=110)
    plt.close(fig)
    return {
        "n_true": len(true),
        "n_accepted": len(res.accepted),
        "n_steps": len(steps),
        "knockout_dchi2": [round(k["dchi2"], 1) for k in knock],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print(f"[stage5] figures -> {FIGDIR}")
    print(f"[stage5] feature FWHM ≈ {feature_fwhm_mhz(TAU_US)*1e3:.1f} kHz "
          f"(T={T_US} us, tau={TAU_US} us)")

    m = fig01_model_validation()
    print(f"[fig01] model closed-form vs FFT rel err {m['model_rel_err']:.2e}; "
          f"Jacobian err ∂Δf {m['jac_df_err']:.1e} ∂τ {m['jac_dt_err']:.1e}")

    sb = fig02_sideband(rng)
    for r in sb["rows"]:
        print(f"[fig02] {r['sideband']:>5} sideband / {r['sign']:>7} sign: "
              f"freq err {r['freq_err_khz']:+8.2f} kHz, "
              f"phase err {r['phase_err']:+.3f} rad")

    bl = fig03_blending(rng)
    for (ph, ra), r in bl["results"].items():
        i_half = int(np.argmin(np.abs(r["sep_fwhm"] - 0.5)))
        print(f"[fig03] {ph:>11} / {ra:>3}: at 0.5 FWHM "
              f"1-vs-2 p {r['pval'][i_half]:.2e}, "
              f"K=2 freq err {r['freq_err_khz'][i_half]:.2f} kHz")

    fc = fig04_fixed_contributor(rng)
    print(f"[fig04] weak-line bias true-skirt {fc['bias_true_khz']}")
    print(f"[fig04] weak-line bias misfit-skirt {fc['bias_misfit_khz']}")

    ak = fig05_audit_knockout(rng)
    print(f"[fig05] audit trail: {ak['n_steps']} steps, "
          f"{ak['n_accepted']}/{ak['n_true']} lines accepted")
    print(f"[fig05] knockout Δχ² per line: {ak['knockout_dchi2']}")
    print("[stage5] done")


if __name__ == "__main__":
    main()
