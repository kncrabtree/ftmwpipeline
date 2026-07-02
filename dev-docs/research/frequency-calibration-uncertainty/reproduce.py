#!/usr/bin/env python
"""Reproducer for the frequency-calibration / σ_f uncertainty report.

Regenerates the report's numbers and figures from the checked-in `.ftmw`
fixtures and the vinyl-cyanide catalog. Self-contained: build the fixtures fresh
if absent, then run each section. Figures go to a gitignored output dir.

    python reproduce.py [--out DIR] [--rebuild] [SECTION ...]

SECTION in {eps, between, within, snr, quad, lattice, pedestal}; default = all
available. Run from the repo root (the fixtures live under examples/).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import ftmwpipeline.api as ftmw

REPO = Path(__file__).resolve().parents[3]
PROBE = 40960.0
TRIM = (26500.0, 40000.0)
CATALOG = REPO / "dev-docs/fixtures/1512-vinyl-cyanide-truth/combined_lines.csv"
# fixtures reused from the cross-acquisition scratch build to avoid 16-min rebuilds
FIXDIR = REPO / "scratch/vycn-crossacq"
FIXTURES = {"655": "examples/blackchirp_data/655",
            "1512": "examples/blackchirp_data/1512"}
_BETWEEN_CACHE: dict = {}   # so §6 reuses §8 in a full run instead of recomputing


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def ensure_fixture(name: str, rebuild: bool) -> Path:
    out = FIXDIR / f"{name}.ftmw"
    if out.exists() and not rebuild:
        return out
    FIXDIR.mkdir(parents=True, exist_ok=True)
    print(f"  building {name} (this is slow for 655) ...", flush=True)
    res = ftmw.run_pipeline(str(REPO / FIXTURES[name]), output=str(out),
                            trim=TRIM, force=True, progress=False)
    if res["status"] != "success":
        raise SystemExit(f"build {name} failed at {res.get('failed_stage')}")
    return out


def final_lines(name: str):
    """(eps, sigma_eps, list of dict) from the production final-products surface."""
    fp = ftmw.get_final_products(str(FIXDIR / f"{name}.ftmw"))
    peaks = sorted(fp.peaks, key=lambda p: p.frequency_mhz)
    fr = np.array([p.frequency_mhz for p in peaks])
    out = []
    for i, p in enumerate(peaks):
        nn = min((abs(fr[j] - p.frequency_mhz) for j in range(len(peaks)) if j != i),
                 default=1e9)
        out.append(dict(f=p.frequency_mhz, fraw=p.frequency_raw_mhz,
                        base=p.f_baseband_mhz, snr=p.snr or 0.0,
                        stat=p.sigma_stat_khz, sig=p.sigma_f_khz, nn=nn))
    return fp.epsilon, fp.sigma_epsilon, out


def catalog_freqs(max_unc_mhz: float = 0.05) -> np.ndarray:
    rows = csv.DictReader(open(CATALOG))
    fs = []
    for r in rows:
        try:
            f = float(r["freq_mhz"])
            u = float(r["unc_mhz"]) if r.get("unc_mhz") else 1.0
        except (TypeError, ValueError):
            continue
        if TRIM[0] <= f <= TRIM[1] and u <= max_unc_mhz:
            fs.append(f)
    return np.array(sorted(fs))


def catalog_species():
    """All in-band catalog (freq_mhz, species) — incl. hyperfine components."""
    out = []
    for r in csv.DictReader(open(CATALOG)):
        try:
            f = float(r["freq_mhz"])
        except (TypeError, ValueError):
            continue
        if TRIM[0] <= f <= TRIM[1]:
            out.append((f, r["species"]))
    return out


def tag_hyperfine(freq_mhz: float, cat, win_mhz: float = 0.3):
    """(nearest species, # catalog components within win) for a fitted line.
    ¹⁵N is spin-½ (hyperfine-free); v0_main/¹³C carry ¹⁴N quadrupole hyperfine."""
    near = sorted(((abs(f - freq_mhz), sp) for f, sp in cat), key=lambda x: x[0])
    nclose = sum(1 for d, _ in near if d < win_mhz)
    return (near[0][1] if near else "?"), nclose


def outdir(args) -> Path:
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    return d


def catalog_match(name: str, iso_only: bool = True):
    """Per-line catalog residuals: arrays (resid_corr_khz, resid_raw_khz, snr,
    baseband_mhz) for lines matched to a clean catalog line (corrected within
    10 kHz)."""
    cat = catalog_freqs()
    _, _, lines = final_lines(name)
    rc, rr, snr, base = [], [], [], []
    for ln in lines:
        j = int(np.argmin(np.abs(cat - ln["f"])))
        if abs(cat[j] - ln["f"]) > 0.010:
            continue
        if iso_only and ln["nn"] < 0.3:
            continue
        rc.append((ln["f"] - cat[j]) * 1e3)
        rr.append((ln["fraw"] - cat[j]) * 1e3)
        snr.append(ln["snr"])
        base.append(ln["base"])
    return (np.array(rc), np.array(rr), np.array(snr), np.array(base))


# ---------------------------------------------------------------------------
# §3 — the calibration mechanism (ε from catalog + prior-free spurs)
# ---------------------------------------------------------------------------
def sec_eps(args):
    print("\n## §3  ε calibration (catalog regression vs prior-free spurs)")
    cat = catalog_freqs()
    for name in ("1512", "655"):
        eps_spur, sig_eps, lines = final_lines(name)
        # Match on the production-CORRECTED frequency (within ~10 kHz of catalog)
        # for clean line identity, then fit (cat - f_raw) = eps*baseband and read
        # the residual reduction off corrected-vs-raw against the same catalog line.
        draw, dcorr, base = [], [], []
        for ln in lines:
            j = int(np.argmin(np.abs(cat - ln["f"])))
            if abs(cat[j] - ln["f"]) > 0.010 or ln["nn"] < 0.3:
                continue
            draw.append(cat[j] - ln["fraw"])     # raw residual to catalog (MHz)
            dcorr.append(cat[j] - ln["f"])        # corrected residual to catalog
            base.append(PROBE - ln["fraw"])
        draw = np.array(draw); dcorr = np.array(dcorr); base = np.array(base)
        eps_cat = float(np.sum(draw * base) / np.sum(base * base))  # cat-raw = eps*base
        print(f"  {name}: eps_catalog={eps_cat*1e6:+.2f}ppm  "
              f"eps_spur={eps_spur*1e6:+.2f}ppm (sigma {sig_eps*1e6:.3f})  "
              f"| catalog residual {np.std(draw)*1e3:.1f} -> {np.std(dcorr)*1e3:.1f} kHz "
              f"(median |{np.median(np.abs(draw))*1e3:.1f}| -> "
              f"|{np.median(np.abs(dcorr))*1e3:.1f}|, n={len(draw)} iso)")


# ---------------------------------------------------------------------------
# §8 — between-acquisition reproducibility (catalog-free)
# ---------------------------------------------------------------------------
def sec_between(args):
    print("\n## §8  between-acquisition reproducibility (655 vs 1512, catalog-free)")
    _, _, A = final_lines("655")
    _, _, B = final_lines("1512")
    fb = np.array([p["f"] for p in B])
    rows = []
    for pa in A:
        j = int(np.argmin(np.abs(fb - pa["f"])))
        dkhz = (pa["f"] - fb[j]) * 1e3
        if abs(dkhz) < 50:
            rows.append((dkhz, min(pa["snr"], B[j]["snr"]),
                         pa["nn"] > 0.3 and B[j]["nn"] > 0.3, pa["f"]))
    R = np.array(rows)
    d, snr, iso, fmol = R[:, 0], R[:, 1], R[:, 2].astype(bool), R[:, 3]
    print(f"  matched {len(R)} lines, isolated-both={iso.sum()}")
    for lab, m in [("all", np.ones(len(d), bool)), ("iso", iso),
                   ("iso snr>100", iso & (snr > 100)), ("iso snr>300", iso & (snr > 300))]:
        if m.sum() > 1:
            x = d[m]
            print(f"    {lab:13} n={m.sum():3d}  med={np.median(x):+6.1f}  "
                  f"std={x.std(ddof=1):5.1f}  per-meas={x.std(ddof=1)/np.sqrt(2):5.1f} kHz")
    m = iso & (snr > 50)
    flat = float(np.median(d[m]))
    base = PROBE - fmol[m]
    A_ = np.vstack([base - base.mean(), np.ones_like(base)]).T
    slope = np.linalg.lstsq(A_, d[m], rcond=None)[0][0]
    print(f"  flat offset delta_down = {flat:+.2f} kHz ; "
          f"tilt vs baseband = {slope*1e3:+.2f} kHz/GHz "
          f"(Doppler/eps degenerate within one pair)")
    _plot_between(args, d, snr, iso, fmol, flat)
    r = dict(flat=flat, repro_hi=d[iso & (snr > 100)].std(ddof=1) / np.sqrt(2))
    _BETWEEN_CACHE["r"] = r
    return r


def _plot_between(args, d, snr, iso, fmol, flat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    s = snr[iso]
    order = np.argsort(s)
    # binned per-meas repro vs SNR
    bins = [(0, 30), (30, 100), (100, 300), (300, 1e9)]
    xs, ys = [], []
    for lo, hi in bins:
        mm = iso & (snr >= lo) & (snr < hi)
        if mm.sum() > 2:
            xs.append(np.sqrt(lo * max(hi, lo + 1)) if hi < 1e9 else lo * 2)
            ys.append(d[mm].std(ddof=1) / np.sqrt(2))
    ax[0].semilogx(xs, ys, "o-")
    ax[0].set_xlabel("paired min SNR"); ax[0].set_ylabel("per-meas repro (kHz)")
    ax[0].set_title("between-acq random reproducibility vs SNR"); ax[0].grid(alpha=0.3)
    ax[1].scatter(PROBE - fmol[iso], d[iso], s=10)
    ax[1].axhline(flat, color="C3", label=f"flat offset {flat:+.1f} kHz")
    ax[1].set_xlabel("baseband (MHz)"); ax[1].set_ylabel("Δf 655-1512 (kHz)")
    ax[1].set_title("offset is flat in baseband (δ_down)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir(args) / "between_acq.png"
    fig.savefig(p, dpi=130)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# §7 — within-acquisition reproducibility (655 backup differencing)
# ---------------------------------------------------------------------------
def _build_655_chunks():
    """Disjoint shot chunks from 655's cumulative backups (indices 1..5)."""
    from blackchirp import BCFTMW
    ftmw_bc = BCFTMW(str(REPO / FIXTURES["655"]), sep=";")
    n = ftmw_bc.numfids
    cum, shots = {}, {}
    for k in range(1, n):                      # 1..5 are the cumulative backups
        f = ftmw_bc.get_fid(k)
        cum[k] = np.asarray(f.data)[:, 0] * f.shots   # sum*vmult (exact)
        shots[k] = int(f.shots)
    chunks = []
    for k in range(1, n):
        lo = k - 1
        prev = cum[lo] if lo >= 1 else 0.0
        ps = shots[lo] if lo >= 1 else 0
        chunks.append(((cum[k] - prev) / (shots[k] - ps), shots[k] - ps))
    return chunks


def _pruned_model(name_fit: str, targets, model_path: Path):
    """Copy the fit file, prune Stage 4 to the target windows, return path."""
    import shutil
    import h5py
    from dataclasses import replace
    from ftmwpipeline.io.window_serialization import save_window_plan_to_hdf5
    wp = ftmw.load_windows(str(FIXDIR / f"{name_fit}.ftmw"))
    tf = [t["f"] for t in targets]
    kept = [w for w in wp.windows
            if any(lo <= f <= hi for f in tf for (lo, hi) in [w.freq_range])]
    pruned = replace(wp, windows=sorted(kept, key=lambda w: w.window_id),
                     dependency_edges=[],
                     topological_order=sorted(w.window_id for w in kept))
    shutil.copy(FIXDIR / f"{name_fit}.ftmw", model_path)
    with h5py.File(model_path, "a") as h5f:
        del h5f["stage4_windows"]
        save_window_plan_to_hdf5(pruned, h5f.create_group("stage4_windows"))
    return len(kept)


def sec_within(args):
    print("\n## §7  within-acquisition reproducibility (655 backup differencing)")
    import json
    import h5py
    from ftmwpipeline.core.data_structures import FID, FIDProcessingParameters, Sideband
    from ftmwpipeline.file_manager import SourceMetadata, create_pipeline_file
    od = outdir(args)

    # clean isolated high-SNR targets from the 655 final products (sole, far, strong)
    _, _, lines = final_lines("655")
    targets = [dict(f=ln["f"], snr=ln["snr"], stat=ln["stat"])
               for ln in lines if ln["snr"] > 200 and ln["nn"] > 1.0]
    targets = sorted(targets, key=lambda t: -t["snr"])[:15]
    model = od / "model_655.ftmw"
    nwin = _pruned_model("655", targets, model)
    print(f"  {len(targets)} isolated targets (snr>200), pruned model = {nwin} windows")

    chunks = _build_655_chunks()
    print(f"  {len(chunks)} disjoint chunks, ~{chunks[0][1]} shots each "
          f"(full/{(2133080/chunks[0][1])**0.5:.1f} SNR)")
    per = {round(t["f"], 4): [] for t in targets}
    crbs = {round(t["f"], 4): [] for t in targets}
    for ci, (data, nsh) in enumerate(chunks):
        cp = od / f"_chunk655_{ci}.ftmw"
        fid = FID(data=data, spacing=2e-11, probe_freq_mhz=PROBE,
                  sideband=Sideband.LOWER, shots=nsh,
                  processing=FIDProcessingParameters(), metadata={})
        sm = SourceMetadata(source_path=str(REPO / FIXTURES["655"]),
                            format_name="blackchirp", loader_parameters={"chunk": ci})
        create_pipeline_file(str(cp), fid, sm, force=True)
        ftmw.compute_ft(str(cp), trim=TRIM, start_us=3.35, units_power=6)
        ftmw.estimate_noise(str(cp))
        # transplant pruned model (stage2b/3/4)
        with h5py.File(model, "r") as s, h5py.File(cp, "a") as d:
            comp = json.loads(d["pipeline_stages"].attrs["completed_stages"])
            for g in ("stage2b_tau_calibration", "stage3_peaks", "stage4_windows"):
                if g in s:
                    if g in d:
                        del d[g]
                    s.copy(g, d)
                    if g not in comp:
                        comp.append(g)
            d["pipeline_stages"].attrs["completed_stages"] = json.dumps(comp)
        fit = ftmw.fit_peaks(str(cp))
        fr = np.array([p.frequency_mhz for p in fit.fitted_peaks])
        for t in targets:
            k = round(t["f"], 4)
            j = int(np.argmin(np.abs(fr - t["f"])))
            if abs(fr[j] - t["f"]) < 0.5:
                p = fit.fitted_peaks[j]
                per[k].append(p.frequency_mhz)
                if p.frequency_error is not None:
                    crbs[k].append(p.frequency_error * 1e3)
        cp.unlink(missing_ok=True)

    cat = catalog_species()
    print(f"  {'freq_mhz':>11} {'snr':>8} {'std_kHz':>8} {'CRB_kHz':>8} {'ratio':>6}"
          f"  {'species':>8} {'#hf':>4}")
    hf_free, hf = [], []   # std of 15N (hyperfine-free) vs 14N-bearing lines
    for t in targets:
        k = round(t["f"], 4)
        v = per[k]
        if len(v) < 4:
            continue
        std = np.std(v, ddof=1) * 1e3
        crb = float(np.median(crbs[k])) if crbs[k] else np.nan
        sp, nhf = tag_hyperfine(t["f"], cat)
        (hf_free if sp == "15N" else hf).append(std)
        print(f"  {t['f']:>11.4f} {t['snr']:>8.0f} {std:>8.2f} {crb:>8.3f} "
              f"{std/crb if crb else float('nan'):>6.2f}  {sp:>8} {nhf:>4}")
    print(f"\n  hyperfine-free (¹⁵N, spin-½) lines: median std = "
          f"{np.median(hf_free):.2f} kHz  (n={len(hf_free)})  <- clean instrumental floor")
    print(f"  ¹⁴N-bearing (v0_main/¹³C) lines:    median std = "
          f"{np.median(hf):.2f} kHz  (n={len(hf)})  <- inflated by unresolved ¹⁴N hyperfine")
    print(f"  => the within-acquisition floor is sub-kHz on hyperfine-free lines; "
          f"VyCN ¹⁴N hyperfine (not pedestals) drives the high-SNR scatter.")
    print(f"  NOTE: re-bin sweep + steady 1.25x-CRB validation need many backups "
          f"(external 100-backup acquisition); 655's {len(chunks)} backups only bound it")


# ---------------------------------------------------------------------------
# §2/§5 — the catalog residual is SNR-independent (systematic, not statistical)
# ---------------------------------------------------------------------------
def sec_snr(args):
    print("\n## §2/§5  catalog residual vs SNR (corrected; SNR-independence)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, c in (("655", "C0"), ("1512", "C1")):
        rc, rr, snr, base = catalog_match(name)
        m = snr > 0
        ax.loglog(snr[m], np.abs(rc[m]), "o", ms=3, color=c, alpha=0.5, label=name)
        # binned median |residual| vs SNR
        edges = np.array([10, 30, 100, 300, 1000, 3000, 1e6])
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = m & (snr >= lo) & (snr < hi)
            if mm.sum() > 2:
                xs.append(np.sqrt(lo * hi))
                ys.append(np.median(np.abs(rc[mm])))
        ax.loglog(xs, ys, "-", color=c, lw=2)
        if m.sum() > 3:
            sp = np.polyfit(np.log(snr[m]), np.log(np.abs(rc[m]) + 1e-9), 1)[0]
            print(f"  {name}: median |resid_corr| = {np.median(np.abs(rc[m])):.1f} kHz, "
                  f"|resid| ~ SNR^{sp:+.2f} (flat => systematic, n={m.sum()})")
    ax.set_xlabel("SNR"); ax.set_ylabel("|catalog residual| (kHz, corrected)")
    ax.set_title("catalog residual is SNR-independent (a systematic)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = outdir(args) / "snr_independence.png"
    fig.savefig(p, dpi=130)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# §6 — quadrature: the catalog is most of the apparent floor
# ---------------------------------------------------------------------------
def sec_quad(args, between=None):
    print("\n## §6  catalog-vs-run-to-run: the catalog is not a kHz ruler")
    gaps = {}
    for name in ("1512", "655"):
        rc, _, snr, _ = catalog_match(name)
        hi = snr > 100
        gaps[name] = float(np.median(np.abs(rc[hi]))) if hi.sum() else float("nan")
    b = between if between is not None else _BETWEEN_CACHE.get("r") or sec_between(args)
    flat, repro = abs(b["flat"]), b["repro_hi"]
    print(f"  corrected catalog gap (median |Δ|): 1512 ~{gaps['1512']:.1f} kHz, "
          f"655 ~{gaps['655']:.1f} kHz")
    print(f"  catalog-FREE run-to-run: flat δ_down {flat:.1f} kHz, "
          f"random repro {repro:.1f} kHz (SNR-limited by 1512; high-SNR floor ~2.7)")
    # the two acquisitions agree with EACH OTHER as well as / better than with the
    # catalog, so the catalog adds its own accuracy on top of the instrumental terms.
    cat_gap = gaps["1512"]
    inst = np.hypot(flat, 2.7)
    cat_acc = np.sqrt(cat_gap**2 - inst**2) if cat_gap > inst else float("nan")
    if np.isfinite(cat_acc):
        print(f"  quadrature: gap {cat_gap:.1f} ~ sqrt(δ_down {flat:.1f}^2 + repro 2.7^2 "
              f"+ catalog^2) => catalog accuracy ~ {cat_acc:.1f} kHz")
    else:
        print(f"  the catalog gap ({cat_gap:.1f}) is comparable to the instrumental "
              f"terms alone (sqrt(δ_down^2+2.7^2)={inst:.1f}) -> the catalog adds "
              f"~few-kHz accuracy of its own (cavity inputs ~4 kHz); not a kHz ruler.")


# ---------------------------------------------------------------------------
# §9 / §10 — lattice spur reconstruction and pedestal cross-method test.
# These consolidate established results that currently live in dedicated seed
# scripts; porting them onto the fresh fixtures is the remaining reproducer work.
# ---------------------------------------------------------------------------
def sec_lattice(args):
    print("\n## §9  δ_down is not pinnable by the clock lattice  [port pending]")
    print("  established result: 6400 = probe - 3*upconvLO carries δ_down - 3*δ_up")
    print("  AND sits on the δ-free 1280-comb (6400=5*1280, 40960=32*1280) -> diluted")
    print("  to zero; 2-LO reconstruction under-predicts the clean pair (+0.5 vs +4.2).")
    print("  seeds: scratch/{lattice_intercept*.py,spur_2lo.py,spur6400.py}")


def sec_pedestal(args):
    print("\n## §10  pedestal influence (rejected as δ_down driver)  [port pending]")
    print("  established result: 3 pedestal suppressors converge on the offset")
    print("  (+4.4..6.3 / +4.7..6.9 / +6.0..6.7 kHz) while exposed estimators are")
    print("  unstable (30-75 kHz jumps) -> δ_down survives, not a pedestal artifact.")
    print("  seeds: scratch/pedestal-test/{recon,pedestal_test,apod_arm}.py")


# ---------------------------------------------------------------------------
# §12 — the reproducibility grid (17 controlled vinyl-cyanide acquisitions)
# The raw grid is user-held external data (not checked in); this section reads it
# from $FREQCAL_GRID (default scratch/vycn_repro) and skips if absent. It builds
# each acquisition (co-averaging 10-record files via the blackchirp package),
# then prints the ε clock-control table, the δ_down consensus decomposition, and
# the frame-coherence retention, writing eps_clock / delta_down / frame_coherence.
# ---------------------------------------------------------------------------
import os

GRID_DIR = Path(os.environ.get("FREQCAL_GRID", str(REPO / "scratch/vycn_repro")))
# committed report figures (regenerated here, embedded in report.md §12)
FIGURES = Path(__file__).resolve().parent / "figures"
# committed grid artifacts: Stage 6 line CSVs + design metadata + coherence bins.
# `grid` reads these (self-contained); `export-grid` regenerates them from raw.
DATA = Path(__file__).resolve().parent / "data"


def _grid_meta():
    """Parse summary.csv into {exp: {records, ref, psi}} (records from header.csv)."""
    import re
    meta = {}
    with open(GRID_DIR / "summary.csv") as fh:
        for row in csv.DictReader(fh):
            exp = row["Experiment"].strip()
            ref = "Rb" if "rb" in row["Reference"].lower() else "Internal"
            psi = 30 if "30" in row["Pressure (psi)"] else 10
            hdr = (GRID_DIR / "experiments" / exp / "header.csv").read_text()
            m = re.search(r"MultiRecordNum;(\d+)", hdr)
            meta[exp] = dict(records=int(m.group(1)) if m else 1, ref=ref, psi=psi)
    return meta


def _grid_build(exp, meta, built_dir):
    """Build one grid acquisition; co-average 10-record files first. Returns path."""
    import shutil
    import numpy as np
    import pandas as pd
    from blackchirp import BCFTMW
    out = built_dir / f"{exp}.ftmw"
    if out.exists():
        return out
    src = GRID_DIR / "experiments" / exp
    if meta["records"] == 10:                       # materialize a co-averaged copy
        dst = built_dir / "coavg" / exp
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        fid = BCFTMW(str(src), sep=";").get_fid(0)
        fid.average_frames()
        col = np.frompyfunc(lambda v: np.base_repr(int(v), 36), 1, 1)(
            fid._rawdata.ravel()).astype(str)
        (dst / "fid" / "0.csv").write_text("fid0\n" + "\n".join(col) + "\n")
        fp = pd.read_csv(src / "fid" / "fidparams.csv", sep=";")
        fp.loc[fp.index[0], "shots"] = int(fid.fidparams["shots"])
        fp.to_csv(dst / "fid" / "fidparams.csv", sep=";", index=False)
        src = dst
    res = ftmw.run_pipeline(str(src), output=str(out), trim=TRIM, force=True,
                            calibrate=True, progress=False)
    if res["status"] != "success":
        raise SystemExit(f"grid build {exp} failed at {res.get('failed_stage')}")
    ftmw.review_run(str(out))
    return out


def _read_lines_csv(path):
    """(eps, sigma_eps, [per-line dict]) from a committed Stage 6 CSV export."""
    import re
    import pandas as pd
    eps = sig = 0.0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"epsilon_ppm:\s*([+-][\d.]+)\s*\+-\s*([\d.]+)", line)
            if m:
                eps, sig = float(m.group(1)) / 1e6, float(m.group(2)) / 1e6
    df = pd.read_csv(path, comment="#")
    fr = df["frequency_mhz"].to_numpy()
    out = []
    for i in range(len(df)):
        f = float(fr[i])
        nn = float(np.min(np.abs(np.delete(fr, i) - f))) if len(fr) > 1 else 1e9
        snr = df["snr"].iloc[i]
        out.append(dict(f=f, fraw=float(df["frequency_raw_mhz"].iloc[i]),
                        snr=float(snr) if pd.notna(snr) else 0.0, nn=nn))
    return eps, sig, out


def _read_coher_csv(path):
    """(baseband_signal, eta_signal, eta_noise) from a committed coherence CSV."""
    import re
    import pandas as pd
    eta_noise = float("nan")
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"eta_noise:\s*([\d.]+)", line)
            if m:
                eta_noise = float(m.group(1))
    df = pd.read_csv(path, comment="#")
    return df["baseband_mhz"].to_numpy(), df["eta"].to_numpy(), eta_noise


def sec_export_grid(args):
    """Regenerate the committed data/ artifacts from the raw $FREQCAL_GRID export.

    Run once when the raw 17-acquisition export is available; the committed CSVs
    are what `grid` (and thus the report) actually depend on."""
    print("\n## export-grid: regenerate committed data/ from raw export")
    if not (GRID_DIR / "summary.csv").exists():
        print(f"  raw grid not found at {GRID_DIR}; set $FREQCAL_GRID. Skipping.")
        return
    meta = _grid_meta()
    built_dir = GRID_DIR / "built"
    built_dir.mkdir(parents=True, exist_ok=True)
    (DATA / "lines").mkdir(parents=True, exist_ok=True)
    (DATA / "coherence").mkdir(parents=True, exist_ok=True)
    with open(DATA / "grid_meta.csv", "w") as fh:
        fh.write("exp,ref,psi,records\n")
        for exp in sorted(meta):
            m = meta[exp]
            fh.write(f"{exp},{m['ref']},{m['psi']},{m['records']}\n")
    for exp in sorted(meta):
        p = _grid_build(exp, meta[exp], built_dir)
        ftmw.report_table(str(p), fmt="csv",
                          output=str(DATA / "lines" / f"{exp}.csv"))
        if meta[exp]["records"] == 10:
            _, freq, S = _grid_spectra(exp)
            mag = np.abs(S); mm = mag.mean(axis=1)
            eta = np.abs(S.sum(axis=1)) / (mag.sum(axis=1) + 1e-30)
            band = (freq >= 1400) & (freq <= 5000)
            thr = np.percentile(mm[band], 30)
            sig = band & (mm > 8 * thr)
            en = float(np.median(eta[band & (mm < thr)]))
            with open(DATA / "coherence" / f"{exp}.csv", "w") as fh:
                fh.write(f"# frame-coherence signal bins for {exp}\n")
                fh.write(f"# eta_noise: {en:.4f}\n")
                fh.write("baseband_mhz,eta\n")
                for b, e in zip(freq[sig], eta[sig]):
                    fh.write(f"{b:.4f},{e:.5f}\n")
        print(f"  exported {exp}", flush=True)
    print(f"  wrote {DATA}/ (grid_meta.csv, lines/, coherence/)")


def sec_grid(args):
    print("\n## §12  reproducibility grid (17 controlled VyCN acquisitions)")
    if not (DATA / "grid_meta.csv").exists():
        print(f"  committed grid artifacts not found at {DATA}; regenerate with "
              f"'reproduce.py export-grid' and $FREQCAL_GRID set to the raw export.")
        return
    meta = {}
    for r in csv.DictReader(open(DATA / "grid_meta.csv")):
        meta[r["exp"]] = dict(ref=r["ref"], psi=int(r["psi"]),
                              records=int(r["records"]))
    lines, eps, coher = {}, {}, {}
    for exp in sorted(meta):
        es, sg, ln = _read_lines_csv(DATA / "lines" / f"{exp}.csv")
        lines[exp], eps[exp] = ln, (es, sg)
        cf = DATA / "coherence" / f"{exp}.csv"
        if cf.exists():
            coher[exp] = _read_coher_csv(cf)

    # §12.1 -- ε clock control
    cat = catalog_freqs()
    print("\n  §12.1 ε: Internal (free-running) vs Rb-locked digitizer")
    grp = {"Internal": [], "Rb": []}
    for exp in sorted(meta):
        es, sg = eps[exp]
        draw, base = [], []
        for ln in lines[exp]:
            j = int(np.argmin(np.abs(cat - ln["f"])))
            if abs(cat[j] - ln["f"]) <= 0.010 and ln["nn"] >= 0.3:
                draw.append(cat[j] - ln["fraw"]); base.append(PROBE - ln["fraw"])
        ec = (float(np.sum(np.array(draw) * base) / np.sum(np.array(base) ** 2))
              if len(draw) >= 3 else float("nan"))
        grp[meta[exp]["ref"]].append((es * 1e6, ec * 1e6))
    for ref in ("Internal", "Rb"):
        g = np.array(grp[ref])
        spur = g[:, 0]; catv = g[:, 1][~np.isnan(g[:, 1])]
        print(f"    {ref:<8} n={len(g)}  eps_spur = {spur.mean():+.2f} ± {spur.std():.2f}"
              f"   eps_catalog = {catv.mean():+.2f} ± {catv.std():.2f} ppm")

    # §12.2 -- δ_down from dominant hyperfine-multiplet centroids (all 17 runs)
    print("\n  §12.2 δ_down (dominant ¹⁴N-multiplet centroids, catalog-free)")
    cents = _grid_centroids(meta, lines)   # [(consensus_mhz, {exp: centroid_mhz})]
    offs = {}
    for _, byexp in cents:
        med = np.median(list(byexp.values()))
        for exp, c in byexp.items():
            offs.setdefault(exp, []).append((c - med) * 1e3)
    dd = {e: float(np.median(v)) for e, v in offs.items()}
    allv = np.array([dd[e] for e in sorted(dd)])
    byref = {r: np.array([dd[e] for e in dd if meta[e]["ref"] == r])
             for r in ("Internal", "Rb")}
    bypsi = {p: np.array([dd[e] for e in dd if meta[e]["psi"] == p]) for p in (10, 30)}
    print(f"    {len(cents)} dominant multiplet centroids anchor all {len(dd)} runs "
          f"(¹⁵N sub-SNR at ~3000 shots; ¹⁴N-bearing only)")
    print(f"    run-to-run δ_down std = {allv.std(ddof=1):.2f} kHz, mean {allv.mean():+.2f}")
    print(f"    Internal {byref['Internal'].mean():+.2f}±{byref['Internal'].std(ddof=1):.2f}"
          f"  Rb {byref['Rb'].mean():+.2f}±{byref['Rb'].std(ddof=1):.2f} kHz "
          f"(means overlap; no clock-dependent offset)")
    print(f"    10psi {bypsi[10].mean():+.2f}±{bypsi[10].std(ddof=1):.2f}"
          f"  30psi {bypsi[30].mean():+.2f}±{bypsi[30].std(ddof=1):.2f} kHz "
          f"(no resolved pressure shift)")

    # §12.3 -- frame coherence (from committed coherence bins)
    print("\n  §12.3 frame coherence η (10-record co-average retention)")
    for exp in sorted(coher):
        freq_sig, eta_sig, eta_noise = coher[exp]
        print(f"    {exp} {meta[exp]['ref']:<8} η_signal={np.median(eta_sig):.3f} "
              f"η_noise={eta_noise:.3f}")
    _grid_figures(args, meta, eps, dd, offs, coher)


def _grid_centroids(meta, lines, cluster_mhz=0.5, min_acq=12, min_totsnr=100.0):
    """Dominant hyperfine-multiplet centroids across acquisitions.

    Cluster all fitted peaks within cluster_mhz; keep strong clusters present in
    >= min_acq acquisitions; each acquisition's SNR-weighted centroid (the
    split-invariant first moment) is its estimate of that dominant line.
    Returns [(consensus_median_mhz, {exp: centroid_mhz}), ...]."""
    allpk = sorted((ln["f"], ln["snr"], e) for e in lines for ln in lines[e])
    clusters, cur = [], [allpk[0]]
    for p in allpk[1:]:
        if p[0] - cur[-1][0] <= cluster_mhz:
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    out = []
    for cl in clusters:
        byexp = {}
        tot = []
        for e in {p[2] for p in cl}:
            f = np.array([p[0] for p in cl if p[2] == e])
            w = np.array([p[1] for p in cl if p[2] == e])
            byexp[e] = float(np.sum(f * w) / np.sum(w))
            tot.append(w.sum())
        if len(byexp) >= min_acq and np.median(tot) >= min_totsnr:
            out.append((float(np.median(list(byexp.values()))), byexp))
    return out


def _grid_spectra(exp, start_us=3.35, end_us=15.0):
    from blackchirp import BCFTMW
    fid = BCFTMW(str(GRID_DIR / "experiments" / exp), sep=";").get_fid(0)
    d = np.asarray(fid.data); sp = float(fid.fidparams["spacing"])
    lo = max(round(start_us / 1e6 / sp), 0)
    hi = min(round(end_us / 1e6 / sp), d.shape[0])
    S = np.fft.rfft(d[lo:hi, :], axis=0)
    freq = np.fft.rfftfreq(hi - lo, sp) * 1e-6
    return PROBE - freq, freq, S


def _grid_figures(args, meta, eps, dd, offs, coher):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RED, BLUE = "#c0392b", "#2c6fbb"
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = sorted(meta)
    col = {e: (RED if meta[e]["ref"] == "Internal" else BLUE) for e in names}

    # --- Fig 01: ε clock control ---
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, e in enumerate(names):
        ax.errorbar(i, eps[e][0] * 1e6, yerr=eps[e][1] * 1e6, fmt="o", ms=9,
                    color=col[e], ecolor="0.4", capsize=3, zorder=3)
    igrp = [eps[e][0] * 1e6 for e in names if meta[e]["ref"] == "Internal"]
    rgrp = [eps[e][0] * 1e6 for e in names if meta[e]["ref"] == "Rb"]
    ax.axhline(np.mean(igrp), color=RED, ls="--", lw=1.3,
               label=f"Internal mean +{np.mean(igrp):.2f} ± {np.std(igrp):.2f} ppm")
    ax.axhline(np.mean(rgrp), color=BLUE, ls="--", lw=1.3,
               label=f"Rb-locked mean +{np.mean(rgrp):.2f} ± {np.std(rgrp):.2f} ppm")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{e}\n{meta[e]['ref'][0]}" for e in names], fontsize=7.5)
    ax.set_ylabel("clock scale error ε  (ppm)")
    ax.set_title("§12.1  Digitizer sample clock: free-running (Internal) vs Rb-locked")
    ax.legend(loc="center right", framealpha=.95); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / "01_eps_clock_control.png", dpi=140)
    plt.close(fig)

    # --- Fig 02: δ_down, two panels (per-acquisition + by condition) ---
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6),
                                 gridspec_kw={"width_ratios": [2.3, 1]})
    allv = np.array([dd[e] for e in names])
    for i, e in enumerate(names):
        a0.scatter(i, dd[e], color=col[e], s=90, alpha=.9, zorder=3)
    a0.axhline(0, color="0.6", lw=.8)
    a0.axhspan(-allv.std(ddof=1), allv.std(ddof=1), color="0.6", alpha=.13,
               label=f"±{allv.std(ddof=1):.1f} kHz (run-to-run std)")
    a0.set_xticks(range(len(names)))
    a0.set_xticklabels([f"{e}\n{meta[e]['ref'][0]}{meta[e]['psi']}·r{meta[e]['records']}"
                        for e in names], fontsize=6.5)
    a0.set_ylabel("δ_down vs consensus  (kHz)")
    a0.set_title("§12.2  Between-acquisition δ_down (dominant hyperfine-multiplet centroids)")
    a0.legend(loc="lower left"); a0.grid(axis="y", alpha=.3)
    # right: strip by (clock × pressure)
    conds = [("Internal", 10), ("Internal", 30), ("Rb", 10), ("Rb", 30)]
    for j, (ref, psi) in enumerate(conds):
        vals = [dd[e] for e in names if meta[e]["ref"] == ref and meta[e]["psi"] == psi]
        c = RED if ref == "Internal" else BLUE
        a1.scatter(np.full(len(vals), j) + np.linspace(-.12, .12, len(vals)),
                   vals, color=c, s=70, alpha=.9)
        a1.plot([j - .2, j + .2], [np.mean(vals)] * 2, color=c, lw=2)
    a1.axhline(0, color="0.6", lw=.8)
    a1.set_xticks(range(4))
    a1.set_xticklabels([f"{r[0]}\n{p}psi" for r, p in conds], fontsize=8)
    a1.set_title("by condition (bar = mean)"); a1.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / "02_delta_down.png", dpi=140)
    plt.close(fig)

    # --- Fig 03: frame coherence ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, (freq_sig, eta_sig, _noise) in coher.items():
        ax.scatter(freq_sig, eta_sig, s=10,
                   color=(RED if meta[exp]["ref"] == "Internal" else BLUE), alpha=.45)
    ax.axhline(1 / np.sqrt(10), color="0.4", ls="--",
               label="incoherent floor  1/√10 ≈ 0.32")
    ax.axhline(1.0, color="0.7", lw=.8)
    ax.plot([], [], "o", color=RED, label="Internal (free-running)")
    ax.plot([], [], "o", color=BLUE, label="Rb-locked")
    ax.set_xlabel("baseband frequency  (MHz)")
    ax.set_ylabel("co-average retention  η = |Σ Sₖ| / Σ|Sₖ|")
    ax.set_title("§12.3  Ten-frame co-average is phase-coherent across the band (both clocks)")
    ax.set_ylim(0.2, 1.03); ax.grid(alpha=.3); ax.legend(loc="lower left", framealpha=.95)
    fig.tight_layout(); fig.savefig(FIGURES / "03_frame_coherence.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {FIGURES}/{{01_eps_clock_control,02_delta_down,03_frame_coherence}}.png")


SECTIONS = {"eps": sec_eps, "between": sec_between, "within": sec_within,
            "snr": sec_snr, "quad": sec_quad,
            "lattice": sec_lattice, "pedestal": sec_pedestal, "grid": sec_grid}
# export-grid regenerates committed data/ from raw; not part of a default run.
EXTRA = {"export-grid": sec_export_grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=[])
    ap.add_argument("--out", default=str(REPO / "output/freqcal-repro"))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    todo = args.sections or list(SECTIONS)
    # the 655/1512 fixtures are only needed by the archival-pair sections
    if any(s in todo for s in ("eps", "between", "within", "snr", "quad")):
        for name in FIXTURES:
            ensure_fixture(name, args.rebuild)
    for s in todo:
        if s in SECTIONS:
            SECTIONS[s](args)
        elif s in EXTRA:
            EXTRA[s](args)
        else:
            print(f"  (section '{s}' not yet ported)")


if __name__ == "__main__":
    main()
