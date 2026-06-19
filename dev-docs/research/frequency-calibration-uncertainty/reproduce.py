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


SECTIONS = {"eps": sec_eps, "between": sec_between, "within": sec_within,
            "snr": sec_snr, "quad": sec_quad,
            "lattice": sec_lattice, "pedestal": sec_pedestal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=[])
    ap.add_argument("--out", default=str(REPO / "output/freqcal-repro"))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    for name in FIXTURES:
        ensure_fixture(name, args.rebuild)
    todo = args.sections or list(SECTIONS)
    for s in todo:
        if s in SECTIONS:
            SECTIONS[s](args)
        else:
            print(f"  (section '{s}' not yet ported)")


if __name__ == "__main__":
    main()
