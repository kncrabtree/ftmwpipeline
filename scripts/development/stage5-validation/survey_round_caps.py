"""Stage 5 rescue round-cap calibration survey.

For each cap in ``--caps`` (default 3, 5, 7), runs the residual-rescue
chain on every Nth window (default 7) of the 2638 fixture and records
per-window:

* ``cap``                -- the ``max_rescue_rounds`` setting used.
* ``window_id``          -- the FitWindow id.
* ``n_rounds_executed``  -- ``len(ConsolidatedRescueOutcome.rounds)``.
* ``terminated_reason``  -- ``"no candidates"`` / ``"joint failed"`` /
                            ``"all pruned"`` / ``"max rounds reached"``.
* ``init_chi2_r``        -- reduced chi-squared of the persisted initial
                            fit (before rescue).
* ``final_chi2_r``       -- reduced chi-squared after the rescue chain.
* ``n_peaks_initial`` / ``n_peaks_final`` -- peak counts.

Writes a single CSV at ``scratch/stage5-rescue-cap-survey/survey.csv``
suitable for picking the smallest cap that satisfies:

* No window terminates at ``"max rounds reached"`` with chi^2_r > 2 *
  median(final_chi2_r).
* >= 90% of windows terminate at round <= ceil(cap / 2).

No PNGs or per-window markdown -- this is a calibration sweep, not a
visualization run. The deliberate-sample visual harness lives at
:mod:`generate_validation`.

Run from repo root::

    conda run -n ftmwpipeline-dev python \\
        scripts/development/stage5-validation/survey_round_caps.py \\
        --stride 7 --caps 3,5,7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.core.data_structures import (
    FittingResult,
    FitWindow,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.residual_rescue import (
    DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    DEFAULT_RESCUE_SNR_THRESHOLD,
    ConsolidatedRescueOutcome,
)
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

# Reuse the harness's _run_window_rescue helper -- the survey runs the
# same rescue path the validation harness exercises, just without the
# PNG/MD output. ``generate_validation`` sits next to this script; Python
# adds the script directory to ``sys.path[0]`` at startup so the bare
# module name resolves when this file is run as a script.
from generate_validation import _run_window_rescue  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "scratch" / "stage5-validation"
FTMW_PATH = FIXTURE_DIR / "exp_2638.ftmw"
SURVEY_DIR = REPO_ROOT / "scratch" / "stage5-rescue-cap-survey"


def _reduced_chi2(chi2: float, n_data: int, n_params: int) -> float:
    return float(chi2) / max(1, int(n_data) - int(n_params))


def _row(
    cap: int,
    wid: int,
    consolidated: ConsolidatedRescueOutcome,
) -> dict:
    init = consolidated.initial_fit.fit
    final = consolidated.fit.fit
    return {
        "cap": cap,
        "window_id": wid,
        "n_rounds_executed": len(consolidated.rounds),
        "terminated_reason": consolidated.terminated_reason,
        "init_chi2_r": _reduced_chi2(init.chi_squared, init.n_data, init.n_params),
        "final_chi2_r": _reduced_chi2(
            final.chi_squared, final.n_data, final.n_params
        ),
        "n_peaks_initial": consolidated.initial_fit.n_peaks,
        "n_peaks_final": consolidated.fit.n_peaks,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stride",
        type=int,
        default=7,
        help="Survey every Nth window from plan.windows (default: 7).",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default="3,5,7",
        help="Comma-separated list of max_rescue_rounds values (default: 3,5,7).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SURVEY_DIR / "survey.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    caps: List[int] = [int(x) for x in args.caps.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not FTMW_PATH.exists():
        raise SystemExit(f"missing fixture: {FTMW_PATH}")

    plan: WindowPlan = ftmw.load_windows(str(FTMW_PATH))
    fit: SpectrumFit = ftmw.load_fit(str(FTMW_PATH))
    peaks_loaded = ftmw.load_peaks(str(FTMW_PATH))

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
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freqs_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(active_ft.complex_spectrum[sort_idx])
    active_noise = estimate_noise_adaptive(
        freqs_sorted, np.abs(spec_sorted).astype(np.float64)
    )
    rms_sorted = np.asarray(active_noise.rms_noise, dtype=float)
    active_noise_arr = rms_sorted[unsort_idx]

    fit_by_id = {wf.window_id: wf for wf in fit.window_fits}
    plan_by_id = {w.window_id: w for w in plan.windows}

    params = fit.parameters or {}
    tau0_us_v = float(params.get("tau0_us", acquisition_us / 3.0))
    fit_tau_v = bool(params.get("fit_tau", True))
    init_conservative_kwargs = {
        "max_decay_factor": float(params.get("max_decay_factor", 5.0)),
        "tau_apodization_us": expf_us if expf_us else None,
    }
    rescue_conservative_kwargs = dict(init_conservative_kwargs)
    rescue_kwargs = {
        "snr_threshold": DEFAULT_RESCUE_SNR_THRESHOLD,
        "prominence_threshold": DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
        "rescue_max_peaks": 32,
        "shape_error_epsilon": 0.05,
    }
    peak_frequencies_mhz = [float(p.frequency) for p in peaks_loaded]

    # Every Nth window from the plan's window list (ordered by ascending
    # window_id since plan.windows is built that way).
    survey_ids = [w.window_id for w in plan.windows][:: args.stride]
    print(
        f"survey: {len(survey_ids)} windows (stride={args.stride}) "
        f"x {len(caps)} caps = {len(survey_ids) * len(caps)} fits"
    )

    rows: List[dict] = []
    for cap in caps:
        print(f"--- cap = {cap} ---")
        for i, wid in enumerate(survey_ids):
            window: FitWindow = plan_by_id[wid]
            wf: FittingResult = fit_by_id[wid]
            try:
                consolidated, _initial_fit, _center_mhz = _run_window_rescue(
                    window,
                    wf,
                    active_ft=active_ft,
                    active_noise_arr=active_noise_arr,
                    sideband=sideband_enum,
                    acquisition_us=acquisition_us,
                    tau0_us=tau0_us_v,
                    fit_tau=fit_tau_v,
                    peak_frequencies_mhz=peak_frequencies_mhz,
                    init_conservative_kwargs=init_conservative_kwargs,
                    rescue_conservative_kwargs=rescue_conservative_kwargs,
                    rescue_kwargs=rescue_kwargs,
                    max_rescue_rounds=cap,
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                row = {
                    "cap": cap,
                    "window_id": wid,
                    "n_rounds_executed": -1,
                    "terminated_reason": f"ERROR: {type(exc).__name__}: {exc}",
                    "init_chi2_r": float("nan"),
                    "final_chi2_r": float("nan"),
                    "n_peaks_initial": -1,
                    "n_peaks_final": -1,
                }
                rows.append(row)
                print(f"  w{wid:>3}: ERROR {exc}")
                continue
            row = _row(cap, wid, consolidated)
            rows.append(row)
            if (i + 1) % 10 == 0 or i == len(survey_ids) - 1:
                print(
                    f"  [{i + 1:>3}/{len(survey_ids)}] w{wid:>3} "
                    f"rounds={row['n_rounds_executed']:>2} "
                    f"reason={row['terminated_reason']:<22} "
                    f"chi2_r {row['init_chi2_r']:.2f}->{row['final_chi2_r']:.2f}"
                )

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")

    # Quick acceptance-criterion check per cap.
    print()
    print("--- per-cap summary ---")
    for cap in caps:
        cap_rows = [r for r in rows if r["cap"] == cap and r["n_rounds_executed"] >= 0]
        if not cap_rows:
            print(f"cap={cap}: no successful rows")
            continue
        finals = np.asarray([r["final_chi2_r"] for r in cap_rows], dtype=float)
        median_chi2_r = float(np.median(finals))
        max_reached = [
            r for r in cap_rows if r["terminated_reason"] == "max rounds reached"
        ]
        bad_max = [r for r in max_reached if r["final_chi2_r"] > 2.0 * median_chi2_r]
        threshold_n = max(1, (cap + 1) // 2)
        early = [r for r in cap_rows if r["n_rounds_executed"] <= threshold_n]
        early_pct = 100.0 * len(early) / len(cap_rows)
        print(
            f"cap={cap:>2}: n={len(cap_rows):>3} "
            f"median chi2_r={median_chi2_r:.2f} "
            f"p95={float(np.percentile(finals, 95)):.2f}  "
            f"max-reached={len(max_reached)} (with chi2_r>2*med: {len(bad_max)})  "
            f"converged-by-round-{threshold_n}: {len(early)}/{len(cap_rows)} "
            f"({early_pct:.0f}%)"
        )


if __name__ == "__main__":
    main()
