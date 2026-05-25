"""Polish-step validation for the STFT tau calibration.

Companion to the Phase 1 prototype (``prototype.py``). The Phase 1 study
established the +3-5 % log-linear-weighting bias on the case-1 grid and
left three candidate fixes in § "Open questions for Phase 4". This
script implements and validates the cheapest of them -- one Gauss-Newton
NLS step per contributor bin -- against the production
``extract_tau_majority`` entry point.

Two synthetic harnesses:

1. **Case-1 grid replay.** Reproduces the Phase-1 case-1 (T_full, tau,
   N_seg) sweep with polish ON (`polish=True`, the new production
   default) and OFF, reporting per-cell median bias.

2. **2638-shape multi-line.** Builds a 2638-shape synthetic with a
   controlled `tau(f)` gradient (7.5 us low molecular freq -> 6 us
   high) and a `SNR(f)` gradient (1x low -> 3x high; matching the
   chirp-induced excitation-time gradient noted in
   ``../residual-rescue/report.md``). Compares the SNR-weighted
   majority against the analytically-known SNR-weighted-expected tau.

Both harnesses also report the behaviour of the opt-in
``polish_noise_debias`` knob (Rician-unbiased magnitude
``sqrt(|S|^2 - 2 sigma^2)``): it closes case-1 to sub-percent but
over-corrects on multi-line spectra and on real 2638 (FID-tail sigma
overstates the per-bin noise by ~2x because of residual signal in the
tail). Documented as the reason the debias is opt-in only.

Run from repo root in the dev env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage5-tau-calibration/polish_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from prototype import (  # type: ignore  # noqa: E402
    make_single_line_fid,
    synth_fid,
    SAMPLE_DT_US as PROTO_SAMPLE_DT_US,
)

from ftmwpipeline.fitting.tau_calibration import (  # noqa: E402
    extract_tau_majority,
)

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("polish_validation")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Case-1 grid replay
# ---------------------------------------------------------------------------
def case1_polish_replay() -> dict:
    """Reproduce the Phase-1 case-1 grid at N_seg=10 with polish ON / OFF.

    Reports the (T_full, tau) bias table for both paths. Polish-OFF
    reproduces the published Phase-1 numbers; polish-ON is the new
    production default.
    """
    logger.info("=== case-1 polish replay ===")
    T_fulls = [5.0, 10.0, 12.65, 20.0, 30.0, 40.0]
    tau_truths = [3.0, 5.0, 7.5, 12.0, 20.0]
    n_seg = 10
    snr = 100.0
    n_trials = 8
    rows: list[dict] = []

    for T_full in T_fulls:
        for tau_truth in tau_truths:
            N = int(round(T_full / PROTO_SAMPLE_DT_US))
            if (N // n_seg) < 4:
                continue
            # polish_snr_cap=None on the "on" / debias variants: this
            # script documents the polish-OFF vs polish-ON-no-cap endpoint
            # comparison the +3-5 % bias study was built around. The
            # production default (cap=9) is a third operating point swept
            # separately by polish_snr_cap_validation.py.
            for label, kwargs in (
                ("off", dict(polish=False)),
                ("on", dict(polish=True, polish_n_iter=1, polish_snr_cap=None)),
                (
                    "on_noise_debias",
                    dict(polish=True, polish_n_iter=1, polish_snr_cap=None,
                         polish_noise_debias=True),
                ),
            ):
                errs = []
                for trial in range(n_trials):
                    rng = np.random.default_rng(
                        20260525 + 1000 * trial + int(T_full * 100),
                    )
                    fid, meta = make_single_line_fid(
                        T_full, tau_truth, snr, rng=rng, n_seg=n_seg,
                    )
                    result = extract_tau_majority(
                        fid, PROTO_SAMPLE_DT_US,
                        start_us=0.0,
                        end_us=meta["n_samples"] * PROTO_SAMPLE_DT_US,
                        probe_freq_mhz=40000.0,
                        sideband="lower",
                        trim_lo_mhz=0.0, trim_hi_mhz=1e6,
                        sigma_time=meta["sigma_time"],
                        n_seg=n_seg, min_contributors=1,
                        **kwargs,
                    )
                    if (
                        np.isfinite(result.tau_maj_us)
                        and result.tau_maj_us > 0
                    ):
                        errs.append(
                            100.0 * (result.tau_maj_us - tau_truth) / tau_truth
                        )
                if errs:
                    rows.append({
                        "T_full": T_full,
                        "tau_truth": tau_truth,
                        "polish": label,
                        "median_err_pct": float(np.median(errs)),
                        "iqr_err_pct": float(
                            np.percentile(errs, 75) - np.percentile(errs, 25)
                        ),
                        "trials": len(errs),
                    })

    np.savez(DATA / "polish_case1.npz", rows=np.asarray(json.dumps(rows)))

    # Side-by-side heatmaps.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, label, title in zip(
        axes,
        ("off", "on", "on_noise_debias"),
        (
            "polish=OFF (legacy log-linear)",
            "polish=ON (new default)",
            "polish=ON + noise_debias (opt-in)",
        ),
    ):
        Z = np.full((len(tau_truths), len(T_fulls)), np.nan)
        for r in rows:
            if r["polish"] != label:
                continue
            i = tau_truths.index(r["tau_truth"])
            j = T_fulls.index(r["T_full"])
            Z[i, j] = r["median_err_pct"]
        im = ax.imshow(
            Z, aspect="auto", origin="lower",
            cmap="RdBu_r", vmin=-15, vmax=15,
            extent=[-0.5, len(T_fulls) - 0.5, -0.5, len(tau_truths) - 0.5],
        )
        ax.set_xticks(range(len(T_fulls)))
        ax.set_xticklabels([f"{t:g}" for t in T_fulls])
        ax.set_yticks(range(len(tau_truths)))
        ax.set_yticklabels([f"{t:g}" for t in tau_truths])
        ax.set_xlabel("T_full (us)")
        ax.set_ylabel("tau_truth (us)")
        ax.set_title(title)
        for i in range(len(tau_truths)):
            for j in range(len(T_fulls)):
                if not np.isnan(Z[i, j]):
                    ax.text(
                        j, i, f"{Z[i, j]:+.1f}",
                        ha="center", va="center",
                        color="white" if abs(Z[i, j]) > 8 else "black",
                        fontsize=7,
                    )
        plt.colorbar(im, ax=ax, label="median err (%)")
    fig.suptitle("Polish-step bias closure (N_seg=10, SNR=100, trials=8)")
    fig.tight_layout()
    fig.savefig(FIG / "10_polish_case1.png", dpi=110)
    plt.close(fig)

    return {"rows": rows}


# ---------------------------------------------------------------------------
# 2638-shape multi-line synthetic
# ---------------------------------------------------------------------------
def case2638_polish() -> dict:
    """2638-shape synthetic with controlled tau(f), SNR(f).

    Builds a 50-line FID at 50 GS/s, T_full=15 us active=12.65 us,
    tau gradient 7.5 -> 6.0 us across the trim band, SNR ratio hi/lo
    = 3x. Reports each polish variant's recovered SNR-weighted
    majority against the analytically-known SNR-weighted-expected
    tau.
    """
    logger.info("=== 2638-shape multi-line polish ===")
    sample_dt_us = 2e-5
    t_full_us = 15.0
    active_end_us = 12.65
    probe_mhz = 40000.0
    trim_lo = 26500.0
    trim_hi = 40000.0
    n_seg = 10
    n_lines = 50
    tau_lo = 7.5
    tau_hi = 6.0
    snr_baseline = 30.0
    snr_ratio_hi = 3.0
    n_trials = 8

    def build_one(rng: np.random.Generator):
        n_active = int(round(active_end_us / sample_dt_us))
        n_active = (n_active // n_seg) * n_seg
        f_bb = np.linspace(100.0, 13400.0, n_lines)  # MHz
        bin_per_mhz = n_active * sample_dt_us
        line_bins = np.round(f_bb * bin_per_mhz).astype(int).tolist()
        f_mol = probe_mhz - f_bb
        f_norm = np.clip(
            (f_mol - trim_lo) / (trim_hi - trim_lo), 0.0, 1.0,
        )
        line_taus = tau_lo + (tau_hi - tau_lo) * f_norm
        line_snrs = snr_baseline * (
            1.0 + (snr_ratio_hi - 1.0) * f_norm
        )
        fid_active, sigma_t, _ = synth_fid(
            n_samples=n_active, sample_dt_us=sample_dt_us,
            line_bins=line_bins,
            line_taus_us=line_taus.tolist(),
            line_snrs=line_snrs.tolist(),
            rng=rng,
        )
        n_total = int(round(t_full_us / sample_dt_us))
        fid = np.zeros(n_total, dtype=float)
        fid[:n_active] = fid_active
        if n_total > n_active:
            fid[n_active:] = rng.normal(scale=sigma_t, size=n_total - n_active)
        expected = float(np.sum(line_snrs * line_taus) / np.sum(line_snrs))
        return fid, sigma_t, expected

    results: dict[str, list[float]] = {}
    expected_truth: float | None = None
    for trial in range(n_trials):
        rng = np.random.default_rng(20260525 + 7 * trial)
        fid, sigma_t, expected = build_one(rng)
        if expected_truth is None:
            expected_truth = expected
        # polish_snr_cap=None: same rationale as case1_polish_replay above.
        for label, kwargs in (
            ("off", dict(polish=False)),
            ("on", dict(polish=True, polish_n_iter=1, polish_snr_cap=None)),
            (
                "on_noise_debias",
                dict(polish=True, polish_n_iter=1, polish_snr_cap=None,
                     polish_noise_debias=True),
            ),
        ):
            r = extract_tau_majority(
                fid, sample_dt_us,
                start_us=0.0, end_us=active_end_us,
                probe_freq_mhz=probe_mhz, sideband="lower",
                trim_lo_mhz=trim_lo, trim_hi_mhz=trim_hi,
                sigma_time=sigma_t,
                n_seg=n_seg, min_contributors=1,
                **kwargs,
            )
            results.setdefault(label, []).append(float(r.tau_maj_us))

    assert expected_truth is not None
    summary: dict = {
        "expected_snr_weighted_tau_us": expected_truth,
        "tau_lo_us": tau_lo, "tau_hi_us": tau_hi,
        "snr_baseline": snr_baseline, "snr_ratio_hi": snr_ratio_hi,
        "n_lines": n_lines, "n_trials": n_trials,
        "variants": {},
    }
    for label, vals in results.items():
        med = float(np.median(vals))
        summary["variants"][label] = {
            "median_tau_us": med,
            "err_pct_vs_truth": 100.0 * (med - expected_truth) / expected_truth,
        }
    (DATA / "polish_2638shape.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["polish=OFF", "polish=ON\n(new default)",
              "polish=ON +\nnoise_debias"]
    keys = ["off", "on", "on_noise_debias"]
    medians = [
        float(np.median(results[k])) for k in keys
    ]
    iqrs_lo = [
        float(np.percentile(results[k], 25)) for k in keys
    ]
    iqrs_hi = [
        float(np.percentile(results[k], 75)) for k in keys
    ]
    err_lo = [m - lo for m, lo in zip(medians, iqrs_lo)]
    err_hi = [hi - m for m, hi in zip(medians, iqrs_hi)]
    ax.errorbar(
        range(len(labels)), medians,
        yerr=[err_lo, err_hi],
        fmt="o", capsize=4,
    )
    ax.axhline(
        expected_truth, color="k", ls="--", lw=1,
        label=f"SNR-weighted truth = {expected_truth:.3f} us",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("recovered tau_maj (us)")
    ax.set_title(
        f"2638-shape multi-line synthetic (tau(f) {tau_lo}->{tau_hi}, "
        f"SNR ratio {snr_ratio_hi}x)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "11_polish_2638shape.png", dpi=110)
    plt.close(fig)
    return summary


def main() -> None:
    rows = case1_polish_replay()
    summary = case2638_polish()
    print("Case-1 grid -- polish=OFF vs polish=ON medians (%):")
    by_pair: dict[tuple[float, float], dict[str, float]] = {}
    for r in rows["rows"]:
        by_pair.setdefault((r["T_full"], r["tau_truth"]), {})
        by_pair[(r["T_full"], r["tau_truth"])][r["polish"]] = r["median_err_pct"]
    for (T_full, tau_truth), v in sorted(by_pair.items()):
        print(
            f"  T_full={T_full:>5.2f} tau={tau_truth:>4.1f}: "
            f"off={v.get('off', float('nan')):+.2f}  "
            f"on={v.get('on', float('nan')):+.2f}  "
            f"debias={v.get('on_noise_debias', float('nan')):+.2f}"
        )
    print("\n2638-shape multi-line:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
