"""Cross-instrument validation of the Stage 3 gap-pass grid knobs (O4) and the
WEAK/MEDIUM/STRONG SNR classification tiers (O2), issue #10.

Two bounded checks over the seven production fixtures (SNR span ~3 orders of
magnitude):

* **O4** -- the matched-filter gap pass zero-pads the active region by
  ``_GAP_ACTIVE_ZPF = 2`` so the Lorentzian magnitude FWHM lands at ~3 bins
  (SavGol's operating range), and ``_grid_aware_sg_window`` picks an odd
  ``sg_window`` covering ~4 FWHM (floor 5). Both were calibrated on 2638; the
  FWHM-in-bins scales with the molecular tau (Stage 2b ``tau_maj``) and the
  active duration, which vary across instruments. This measures, per fixture,
  the *actual production* gap-grid ``Delta f``, ``FWHM_bins``, and resolved
  ``sg_window`` by instrumenting the real Stage 3 run, and checks they stay in
  range (``FWHM_bins >~ 3``, ``sg_window`` odd ``>= 5``).

* **O2** -- the detected-peak SNR distribution per fixture, and whether the
  fixed tier boundaries ``DEFAULT_WEAK_MEDIUM_SNR = 10`` /
  ``DEFAULT_MEDIUM_STRONG_SNR = 50`` separate the population meaningfully
  across the SNR span, or collapse (e.g. 655 -> all-strong, 363 -> all-weak).

Both quantities are read off the genuine production path: each fixture is built
canonically (import -> detect_start_time -> compute_ft zpf=0/expf=None/trim ->
estimate_noise scatter -> calibrate_tau), then ``detect_peaks_impl`` is run and
its returned ``gap_ft`` / ``parameters_used`` / promoted ``peaks`` are mined --
no hand-recomputation of the grid.

Self-contained: builds each fixture into a gitignored scratch path and writes
the headline JSON next to this script under ``data/``. Run from the repo root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage3-snr-corner/o2_o4_validation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage3_impl import (
    _GAP_ACTIVE_ZPF,
    _SG_FWHM_COVERAGE,
    _SG_MIN_WINDOW,
    _grid_aware_sg_window,
    detect_peaks_impl,
)
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_stage2b_recommended_shape,
)
from ftmwpipeline.preprocessing.peak_detection import (
    DEFAULT_MEDIUM_STRONG_SNR,
    DEFAULT_WEAK_MEDIUM_SNR,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

FIXTURES = ["363", "2638", "360", "1231", "1512", "1019", "655"]
TRIM = (26500.0, 40000.0)
SCRATCH = Path("scratch/stage3_o2o4")

T1 = DEFAULT_WEAK_MEDIUM_SNR  # 10.0
T2 = DEFAULT_MEDIUM_STRONG_SNR  # 50.0


def build(fid: str, reuse: bool = False) -> str:
    """Canonical production pipeline through Stage 2b for one fixture.

    With ``reuse=True`` and an already-built file present, reload the persisted
    Stage 2b calibration instead of re-running the slow ``calibrate_tau`` NLS.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fp = str(SCRATCH / f"exp_{fid}.ftmw")
    if reuse and Path(fp).exists():
        try:
            return fp, ftmw.load_tau_calibration(fp)
        except Exception:  # noqa: BLE001
            pass  # fall through to a clean rebuild
    ftmw.import_data(fp, source=f"examples/blackchirp_data/{fid}", force=True)
    ftmw.detect_start_time(fp, band=TRIM, stamp=True)
    ftmw.compute_ft(fp, zpf=0, expf_us=None, trim=TRIM)
    ftmw.estimate_noise(fp)  # scatter default
    cal = ftmw.calibrate_tau(fp)
    return fp, cal


def o4_metrics(res: dict) -> dict:
    """Gap-grid spacing, FWHM-in-bins, and resolved sg_window from the real run."""
    gap_ft = res["gap_ft"]
    params = res["parameters_used"]
    tau_us = float(params["tau_basis_us"])
    delta_f = float(abs(gap_ft.freq_array[1] - gap_ft.freq_array[0]))
    fwhm_mhz = 1.0 / (np.pi * tau_us)
    fwhm_bins = fwhm_mhz / delta_f
    sg_window = _grid_aware_sg_window(
        delta_f,
        fwhm_mhz,
        fwhm_coverage=_SG_FWHM_COVERAGE,
        min_window=_SG_MIN_WINDOW,
    )
    return dict(
        tau_basis_us=round(tau_us, 4),
        tau_basis_source=params.get("tau_basis_source"),
        gap_active_zpf=int(params.get("gap_active_zpf", _GAP_ACTIVE_ZPF)),
        gap_delta_f_khz=round(delta_f * 1e3, 4),
        line_fwhm_khz=round(fwhm_mhz * 1e3, 4),
        fwhm_bins=round(fwhm_bins, 3),
        sg_window=int(sg_window),
        sg_covers_fwhm=round(sg_window * delta_f / fwhm_mhz, 2),
        fwhm_ok=bool(fwhm_bins >= 3.0),
        sg_ok=bool(sg_window >= 5 and sg_window % 2 == 1),
    )


def o2_metrics(res: dict) -> dict:
    """Promoted-peak SNR distribution and weak/medium/strong tier occupancy."""
    peaks = res["peaks"]
    snr = np.array(
        [
            float(p.snr)
            for p in peaks
            if p.properties.get("promoted") and np.isfinite(p.snr)
        ]
    )
    snr_all = np.array([float(p.snr) for p in peaks if np.isfinite(p.snr)])
    if snr.size == 0:
        return dict(n_promoted=0)
    pct = {
        f"p{q}": round(float(np.percentile(snr, q)), 2)
        for q in (5, 25, 50, 75, 90, 95, 99)
    }
    n_weak = int(np.sum(snr < T1))
    n_med = int(np.sum((snr >= T1) & (snr < T2)))
    n_strong = int(np.sum(snr >= T2))
    n = snr.size
    return dict(
        n_detected=int(snr_all.size),
        n_promoted=n,
        snr_min=round(float(snr.min()), 2),
        snr_max=round(float(snr.max()), 2),
        percentiles=pct,
        tiers_at_10_50=dict(weak=n_weak, medium=n_med, strong=n_strong),
        tier_frac=dict(
            weak=round(n_weak / n, 3),
            medium=round(n_med / n, 3),
            strong=round(n_strong / n, 3),
        ),
    )


def adaptive_zpf(tau_us: float, acquisition_us: float) -> int:
    """Smallest power-of-two active zpf giving FWHM_bins >= 3.

    ``FWHM_bins = 2^zpf * T_active / (pi * tau)``, so solving for FWHM_bins >= 3
    gives ``zpf = ceil(log2(3 * pi * tau / T_active))``, floored at the
    production value 2 (never coarsen below the 2638-calibrated grid).
    """
    if acquisition_us <= 0 or tau_us <= 0:
        return _GAP_ACTIVE_ZPF
    return max(
        _GAP_ACTIVE_ZPF, math.ceil(math.log2(3.0 * math.pi * tau_us / acquisition_us))
    )


# Fixtures whose lines have a published catalog (same vinyl-cyanide molecule +
# instrument band as 1512): scorable for recall against the union catalog.
VC_FIXTURES = {"1512", "655"}
# Active-region zpf values swept for the gap pass: 0 (no padding) through 3.
ZPF_SWEEP = [0, 1, 2, 3]
# Match tolerance for a promoted peak against a catalog line (MHz).
VC_MATCH_TOL_MHZ = 0.05


def vc_union_truth() -> "np.ndarray | None":
    """Union of all vinyl-cyanide catalogued lines (all species) in the band.

    ``combined_lines.csv`` is the parsed in-band union of the five resolved
    species (v=0 main + 3x 13C + 15N); the molecule and instrument band are
    shared by fixtures 1512 and 655, so the same union scores both.
    """
    p = Path("dev-docs/fixtures/1512-vinyl-cyanide-truth/combined_lines.csv")
    if not p.exists():
        return None
    freqs = []
    for row in p.read_text().strip().splitlines()[1:]:
        cols = row.split(",")
        try:  # header: species,tag,predicted,freq_mhz,...
            freqs.append(float(cols[3]))
        except (ValueError, IndexError):
            pass
    arr = np.array([f for f in freqs if TRIM[0] <= f <= TRIM[1]])
    return np.unique(np.round(arr, 4)) if arr.size else None


def o4_zpf_sweep(fid: str, tau_us: float, acquisition_us: float) -> dict:
    """Sweep the gap-pass active zpf down and up around the production value.

    The fixed production ``zpf=2`` was calibrated to land FWHM_bins ~3, but the
    detection optimum is empirical: zpf=0 under-samples the intrinsically coarse
    active-region FT (Δf = 1/T_active), while zpf>=3 over-pads and drives the
    grid-aware SavGol window wide enough to over-smooth weak lines. Scored on
    the vinyl-cyanide catalog union (1512, 655) at a 50 kHz tolerance -- the
    only direct quality metric, since FWHM_bins itself is not one. Also reports
    the adaptive (FWHM_bins>=3) zpf the naive sizing rule would pick, to show it
    lands in the over-padded falloff.
    """
    fp = str(SCRATCH / f"exp_{fid}.ftmw")
    truth = vc_union_truth() if fid in VC_FIXTURES else None

    def run(zpf: int) -> dict:
        s = PeakDetectionSettings()
        s.gap_pass.gap_active_zpf = zpf
        peaks = ftmw.detect_peaks(fp, settings=s)
        prom = [p for p in peaks if p.properties.get("promoted")]
        gap = sum(1 for p in prom if p.properties.get("detection_pass") == "gap")
        fwhm_bins = (2**zpf) * acquisition_us / (math.pi * tau_us)
        rec = None
        if truth is not None and prom:
            pf = np.array([p.frequency for p in prom])
            hit = int(sum(np.min(np.abs(pf - t)) <= VC_MATCH_TOL_MHZ for t in truth))
            rec = dict(
                hit=hit, n_truth=int(truth.size), recall=round(hit / truth.size, 3)
            )
        return dict(
            zpf=zpf,
            fwhm_bins=round(fwhm_bins, 3),
            n_total=len(peaks),
            n_promoted=len(prom),
            n_gap_promoted=int(gap),
            vc_recall=rec,
        )

    return dict(
        sweep=[run(z) for z in ZPF_SWEEP],
        production_zpf=_GAP_ACTIVE_ZPF,
        adaptive_zpf=adaptive_zpf(tau_us, acquisition_us),
    )


def capture(fid: str, reuse: bool = False) -> dict:
    fp, cal = build(fid, reuse=reuse)
    # Pin the gap-pass zpf to the production default explicitly: detect_peaks
    # persists the resolved settings, so a prior A/B run (below) leaves an
    # overridden gap_active_zpf in the file's persisted layer. The preset layer
    # (settings=) outranks persisted, immunising the canonical measurement.
    prod = PeakDetectionSettings()
    prod.gap_pass.gap_active_zpf = _GAP_ACTIVE_ZPF
    res = detect_peaks_impl(fp, settings=prod)
    shape = read_stage2b_recommended_shape(fp)
    o4 = o4_metrics(res)
    o2 = o2_metrics(res)
    acq = float(res["acquisition_us"])
    # Sweep the gap-pass zpf down and up on every fixture; the catalog-truth
    # fixtures (1512, 655) carry the recall metric that adjudicates the optimum.
    sweep = o4_zpf_sweep(fid, o4["tau_basis_us"], acq)
    out = dict(
        fixture=fid,
        recommended_shape=shape,
        tau_maj_us=round(float(cal.tau_maj_us), 4),
        sigma_tau_us=round(float(cal.sigma_tau_us), 4),
        acquisition_us=round(acq, 4),
        o4=o4,
        o4_zpf_sweep=sweep,
        o2=o2,
    )
    t = o2.get("tiers_at_10_50", {})
    print(
        f"[{fid}] shape={shape} tau={cal.tau_maj_us:.2f}us "
        f"| O4 FWHM_bins={o4['fwhm_bins']} sg={o4['sg_window']} "
        f"(fwhm_ok={o4['fwhm_ok']} sg_ok={o4['sg_ok']}) "
        f"| O2 promoted={o2['n_promoted']} "
        f"w/m/s={t.get('weak')}/{t.get('medium')}/{t.get('strong')} "
        f"snr[{o2.get('snr_min')}-{o2.get('snr_max')}]",
        flush=True,
    )
    return out


def main() -> int:
    reuse = "--reuse" in sys.argv[1:]
    results = []
    for fid in FIXTURES:
        try:
            results.append(capture(fid, reuse=reuse))
        except Exception as exc:  # noqa: BLE001
            print(f"[{fid}] FAILED: {exc!r}", flush=True)
            results.append(dict(fixture=fid, error=repr(exc)))
    (DATA / "o2_o4_validation.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {DATA / 'o2_o4_validation.json'}")

    # O4 acceptance summary.
    print("\n=== O4: gap-grid / SavGol ===")
    print(
        f"{'fix':>5} {'tau_us':>7} {'dF_kHz':>7} {'FWHM_bins':>10} "
        f"{'sg':>4} {'covers':>7} {'ok':>4}"
    )
    for r in results:
        if "o4" not in r:
            continue
        o = r["o4"]
        ok = "PASS" if (o["fwhm_ok"] and o["sg_ok"]) else "FAIL"
        print(
            f"{r['fixture']:>5} {o['tau_basis_us']:>7.2f} "
            f"{o['gap_delta_f_khz']:>7.2f} {o['fwhm_bins']:>10.2f} "
            f"{o['sg_window']:>4} {o['sg_covers_fwhm']:>7.2f} {ok:>4}"
        )

    # O4 zpf sweep: detection quality vs active zpf (down and up around 2).
    # production zpf=2; the naive FWHM_bins>=3 rule picks adaptive_zpf (>=3).
    print("\n=== O4: gap-pass zpf sweep (promoted / gap_prom / VC-recall) ===")
    print(
        f"{'fix':>5} {'prod':>4} {'adapt':>5}  "
        + "  ".join(f"zpf{z}" for z in ZPF_SWEEP)
    )
    for r in results:
        sw = r.get("o4_zpf_sweep")
        if not sw:
            continue
        cells = []
        for s in sw["sweep"]:
            rec = s["vc_recall"]
            tag = f"{s['n_promoted']}/{s['n_gap_promoted']}" + (
                f"/{rec['recall']}" if rec else ""
            )
            cells.append(f"{tag:>14}")
        print(
            f"{r['fixture']:>5} {sw['production_zpf']:>4} "
            f"{sw['adaptive_zpf']:>5}  " + "".join(cells)
        )

    # O2 tier-occupancy summary.
    print("\n=== O2: SNR tiers at 10 / 50 (promoted) ===")
    print(
        f"{'fix':>5} {'nprom':>6} {'p50':>7} {'p90':>7} {'max':>9} "
        f"{'weak':>6} {'med':>6} {'strong':>7}"
    )
    for r in results:
        o = r.get("o2", {})
        if not o.get("n_promoted"):
            continue
        t = o["tiers_at_10_50"]
        print(
            f"{r['fixture']:>5} {o['n_promoted']:>6} "
            f"{o['percentiles']['p50']:>7.1f} {o['percentiles']['p90']:>7.1f} "
            f"{o['snr_max']:>9.1f} {t['weak']:>6} {t['medium']:>6} "
            f"{t['strong']:>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
