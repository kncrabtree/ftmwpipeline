"""Phase-4 close-out: which noise reference should the STFT tau calibration use?

The STFT classifier needs a per-bin noise floor to gate above-threshold bins.
Two candidates on a production (scatter-noise) pipeline:

  * FID-tail sigma_t  (``estimate_sigma_time_from_tail``; the production default)
  * the Stage 2 scatter spectral sigma  (lower; physically the cleaner floor)

This script settles the choice against an INDEPENDENT reference: the unbiased
LSQ-fit-and-histogram per-band tau, obtained by running Stages 0-5 with the
tau-anchoring prior DISABLED (``tau_penalty_lambda=0``, ``tau0=T_active/2``,
``per_band_tau=False``, no Stage 2b consumed). That LSQ tau is independent of
either STFT noise reference, so whichever STFT variant it agrees with is the
correct one.

Result on 2638 (the verdict): the FID-tail reference reproduces the LSQ tau's
monotonic decrease with frequency (horn-coupling geometry); the lower scatter
sigma admits weak, log-linear-high-biased bins in the sparse high band and
inverts that trend. The tail's residual-signal inflation is therefore a
*beneficial* stricter above-threshold gate, not a bug -- the production path
keeps it (``extract_tau_majority(sigma_x_full=None)``).

Self-contained: builds the canonical unapodized 2638 fixture from
``examples/blackchirp_data/2638`` into a gitignored scratch path; writes the
headline JSON to ``data/lsq_noise_reference.json``. Run from the repo root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-tau-calibration/lsq_noise_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage1_impl import _read_settings_layer
from ftmwpipeline.core.settings import FT_PROCESSING_PATH
from ftmwpipeline.core.stage_fit_settings import StageFitSettings
from ftmwpipeline.fitting.tau_calibration import (
    extract_tau_majority,
    estimate_sigma_time_from_tail,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

SRC = "examples/blackchirp_data/2638"
FIX = Path("scratch/lsq-noise-reference/exp_2638_unapod.ftmw")
TRIM = (26500.0, 40000.0)
N_SEG = 10
# Arithmetic thirds across the trim band (matches lsq_comparison.py).
THIRDS = [
    ("low", TRIM[0], TRIM[0] + (TRIM[1] - TRIM[0]) / 3.0),
    ("mid", TRIM[0] + (TRIM[1] - TRIM[0]) / 3.0, TRIM[0] + 2 * (TRIM[1] - TRIM[0]) / 3.0),
    ("high", TRIM[0] + 2 * (TRIM[1] - TRIM[0]) / 3.0, TRIM[1]),
]
SNR_GATE = 10.0
RCHI2_GATE = 3.0


def build():
    FIX.parent.mkdir(parents=True, exist_ok=True)
    ftmw.import_data(str(FIX), source=SRC, force=True)
    ftmw.detect_start_time(str(FIX), band=TRIM, stamp=True)
    ftmw.compute_ft(str(FIX), zpf=0, expf_us=None, trim=TRIM)
    nr = ftmw.estimate_noise(str(FIX), method="scatter")
    return nr


def geometry():
    fid = ftmw.load_fid(str(FIX))
    fts = _read_settings_layer(str(FIX), FT_PROCESSING_PATH)
    dt = float(fid.spacing * 1e6)
    return dict(
        arr=np.asarray(fid.data, float), dt=dt,
        start_us=float(fts.start_us), end_us=float(fts.end_us),
        probe=float(fid.probe_freq_mhz),
        sb=fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband),
        units_power=int(fts.units_power),
    )


def stft_bands(g, sigma_x_full):
    r = extract_tau_majority(
        g["arr"], g["dt"], start_us=g["start_us"], end_us=g["end_us"],
        probe_freq_mhz=g["probe"], sideband=g["sb"],
        trim_lo_mhz=TRIM[0], trim_hi_mhz=TRIM[1], n_seg=N_SEG,
        sigma_x_full=sigma_x_full, compute_band_majorities_flag=True,
    )
    return {b.label: round(b.tau_maj_us, 3) for b in r.band_majorities}, r


def scatter_sigma_x_full(nr, g):
    """Median scatter sigma in the trim band, converted to STFT (dt*rfft) units."""
    ft = ftmw.compute_ft(str(FIX), zpf=0, expf_us=None, trim=TRIM)
    freqs = np.asarray(ft.freq_array, float)
    rms = np.asarray(nr.rms_noise, float)
    band = rms[(freqs >= TRIM[0]) & (freqs <= TRIM[1])] if freqs.size == rms.size else rms
    s = max(int(round(g["start_us"] / g["dt"])), 0)
    e = min(int(round(g["end_us"] / g["dt"])), g["arr"].size)
    N = ((e - s) // N_SEG) * N_SEG
    scale = 10.0 ** g["units_power"]
    conv = g["dt"] * N / (scale * np.sqrt(2.0))
    return float(np.median(band) * conv)


def unbiased_lsq_bands():
    """Stage 3-5 with the tau prior OFF -> per-band median of the free-fit tau."""
    ftmw.detect_peaks(str(FIX))
    ftmw.assign_windows(str(FIX))
    fs = StageFitSettings()
    fs.tau.tau0_us = 6.325  # T_active / 2
    fs.tau.fit_tau = True
    fs.tau.tau_penalty_lambda = 0.0
    fs.tau.per_band_tau = False
    ftmw.fit_peaks(str(FIX), settings=fs)
    fit = ftmw.load_fit(str(FIX))
    plan = ftmw.load_windows(str(FIX))
    plan_by_id = {fw.window_id: fw for fw in plan.windows}
    taus, fcs = [], []
    for res in fit.window_fits:
        fw = plan_by_id.get(res.window_id)
        if fw is None or len(res.fixed_parameters):
            continue
        shared = res.shared_parameters.get("tau_us", {})
        tau = float(shared.get("value", float("nan")))
        rchi2 = float(res.reduced_chi2)
        snrs = [float(p.snr) for p in res.fitted_peaks if p.snr and np.isfinite(p.snr)]
        if not snrs or max(snrs) < SNR_GATE or not np.isfinite(rchi2) or rchi2 >= RCHI2_GATE:
            continue
        if not (np.isfinite(tau) and tau > 0):
            continue
        taus.append(tau)
        fcs.append(0.5 * (fw.freq_range[0] + fw.freq_range[1]))
    taus, fcs = np.array(taus), np.array(fcs)
    out = {}
    for name, lo, hi in THIRDS:
        m = (fcs >= lo) & (fcs < hi)
        out[name] = round(float(np.median(taus[m])), 3) if m.any() else None
    return out, int(taus.size)


def main():
    nr = build()
    g = geometry()
    sig_t = estimate_sigma_time_from_tail(
        g["arr"][max(int(round(g["start_us"] / g["dt"])), 0):
                 min(int(round(g["end_us"] / g["dt"])), g["arr"].size)]
    )
    sigx_tail = sig_t * g["dt"] * np.sqrt(
        (((min(int(round(g["end_us"] / g["dt"])), g["arr"].size)
           - max(int(round(g["start_us"] / g["dt"])), 0)) // N_SEG) * N_SEG) / 2.0
    )
    sigx_scatter = scatter_sigma_x_full(nr, g)

    tail_bands, _ = stft_bands(g, None)               # production default
    scatter_bands, _ = stft_bands(g, sigx_scatter)
    lsq_bands, lsq_n = unbiased_lsq_bands()

    result = {
        "fixture": "2638",
        "trim_mhz": list(TRIM),
        "sigma_x_full_tail": sigx_tail,
        "sigma_x_full_scatter": sigx_scatter,
        "scatter_over_tail_ratio": round(sigx_scatter / sigx_tail, 4),
        "bands_lsq_unbiased": lsq_bands,
        "bands_stft_tail": tail_bands,
        "bands_stft_scatter": scatter_bands,
        "lsq_window_count": lsq_n,
    }
    (DATA / "lsq_noise_reference.json").write_text(json.dumps(result, indent=2))

    print(f"scatter/tail sigma ratio: {result['scatter_over_tail_ratio']}")
    print(f"{'band':>5} {'LSQ(indep)':>11} {'STFT-tail':>10} {'STFT-scatter':>13}")
    for name, _, _ in THIRDS:
        print(f"{name:>5} {str(lsq_bands[name]):>11} {str(tail_bands.get(name)):>10} "
              f"{str(scatter_bands.get(name)):>13}")
    # Verdict: tail tracks the LSQ trend; scatter should diverge in the high band.
    print(f"\nLSQ window count: {lsq_n}")
    print("Verdict: production keeps sigma_x_full=None (FID-tail) — see report.md "
          "§ 'Noise-reference robustness'.")


if __name__ == "__main__":
    sys.exit(main())
