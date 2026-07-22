"""Honest catalog recall / precision for a fitted ``.ftmw``.

A development validation utility: scores a finished fit's line list against a
reference catalog with an **injective** (one-to-one) nearest match over a fixed
analysis band, sub-resolution blends collapsed to single targets, and a
species/tag tier split. These three guards keep the measure from rewarding an
over-splitting fit -- a naive all-lines, non-injective, fit-extent-coupled match
inflates recall precisely where the fit splits one line into many.

The catalog is the ``combined_lines.csv`` truth format (columns ``species, tag,
predicted, freq_mhz, unc_mhz, ...``): well-determined lines only (``unc_mhz`` <=
``--unc-max``, dropping wide predicted positions). ``--main-tag`` is the primary
(v=0 main-isotopologue) tier; everything else is reported separately as the
isotopologue tier (mostly sub-noise on lower-SNR fixtures). Precision is the
count of fitted peaks matching no catalog target in any tier -- a soft upper
bound on false positives, since the spectrum may hold other un-cataloged species.

Usage:
    python scripts/development/catalog_recall.py FIT.ftmw \\
        --catalog examples/blackchirp_data/vinyl-cyanide-reference/combined_lines.csv \\
        --main-tag 53515

The VyCN fixtures 1512 and 655 share that catalog. For a possible future
user-facing form this would reuse the shared catalog reader behind
``report --catalog`` (CSV header-units + Pickett/SPCAT ``.cat``); this dev tool
stays self-contained on the truth-CSV format.
"""

from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

import numpy as np

import ftmwpipeline.api as ftmw


def load_catalog(
    path: str, band: Tuple[float, float], unc_max: float, main_tag: str
) -> Tuple[List[float], List[float]]:
    """Return ``(main_freqs, iso_freqs)`` within the band and uncertainty cap."""
    main: List[float] = []
    iso: List[float] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("freq_mhz"):
                continue
            f, u = float(r["freq_mhz"]), float(r.get("unc_mhz") or 0.0)
            if not (band[0] <= f <= band[1] and u <= unc_max):
                continue
            (main if r.get("tag") == main_tag else iso).append(f)
    return main, iso


def collapse_blends(freqs: List[float], res: float) -> List[Tuple[float, float]]:
    """Cluster frequencies within ``res`` into single ``(centroid, span)`` targets."""
    if not freqs:
        return []
    fs = sorted(freqs)
    clusters: List[List[float]] = [[fs[0]]]
    for f in fs[1:]:
        if f - clusters[-1][-1] <= res:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    return [(float(np.mean(c)), c[-1] - c[0]) for c in clusters]


def match_injective(
    targets: List[Tuple[float, float]], peaks: np.ndarray, tol: float
) -> Tuple[int, List[float], set]:
    """Global-greedy one-to-one match (each target's tol widened by half its span).

    Returns ``(n_matched, miss_centroids, matched_peak_indices)``.
    """
    pairs = []
    for ti, (cen, span) in enumerate(targets):
        eff = tol + 0.5 * span
        for pj, pf in enumerate(peaks):
            d = abs(float(pf) - cen)
            if d <= eff:
                pairs.append((d, ti, pj))
    pairs.sort()
    tused: set = set()
    pused: set = set()
    for _d, ti, pj in pairs:
        if ti in tused or pj in pused:
            continue
        tused.add(ti)
        pused.add(pj)
    misses = [targets[ti][0] for ti in range(len(targets)) if ti not in tused]
    return len(tused), misses, pused


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit", help="built .ftmw with a completed fit")
    ap.add_argument("--catalog", required=True, help="combined_lines.csv truth file")
    ap.add_argument(
        "--band",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=(26500.0, 40000.0),
        help="analysis band MHz (the recall denominator span)",
    )
    ap.add_argument(
        "--unc-max", type=float, default=0.005, help="max catalog unc (MHz)"
    )
    ap.add_argument("--main-tag", required=True, help="primary-tier catalog tag")
    args = ap.parse_args()

    fit = ftmw.load_fit(args.fit)
    peaks = np.array(sorted(float(p.frequency_mhz) for p in fit.fitted_peaks))
    acq = (getattr(fit, "parameters", None) or {}).get("acquisition_us") or 13.0
    res = 1.0 / acq
    tol = max(0.05, res)

    main_lines, iso_lines = load_catalog(
        args.catalog, (args.band[0], args.band[1]), args.unc_max, args.main_tag
    )
    tgt_a = collapse_blends(main_lines, res)
    tgt_b = collapse_blends(iso_lines, res)
    m_a, miss_a, _ = match_injective(tgt_a, peaks, tol)
    m_b, _, _ = match_injective(tgt_b, peaks, tol)
    all_t = collapse_blends(main_lines + iso_lines, res)
    _, _, used_all = match_injective(all_t, peaks, tol)
    n_fp = len(peaks) - len(used_all)

    print(
        f"{args.fit}  band {args.band[0]:.0f}-{args.band[1]:.0f}  "
        f"unc<={args.unc_max*1000:.0f}kHz  res={res*1000:.0f}kHz tol={tol*1000:.0f}kHz"
    )
    print(f"  npeaks={len(peaks)}")
    den_a = max(1, len(tgt_a))
    den_b = max(1, len(tgt_b))
    print(f"  main (tag {args.main_tag}): {m_a}/{len(tgt_a)}  recall={m_a/den_a:.3f}")
    print(f"  isotopologue           : {m_b}/{len(tgt_b)}  recall={m_b/den_b:.3f}")
    print(f"  peaks w/ no catalog match (soft FP bound): {n_fp}")
    if miss_a:
        print("  main misses: " + " ".join(f"{c:.3f}" for c in sorted(miss_a)))


if __name__ == "__main__":
    main()
