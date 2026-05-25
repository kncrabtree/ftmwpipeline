"""
Stage 3 projection-coherence study -- cross-tabulation and threshold analysis.

Consumes ``scratch/stage3-coherence-study/candidates.csv`` produced by
``survey.py``. Buckets by projection ratio, reports per-bucket
``became_fitted_peak`` rates, sweeps a threshold, picks the threshold that
maximises precision at TPR ≥ ``--min-recall``, and emits a Markdown report
plus a histogram + ROC figure.

The report's verdict section assumes the ratio is monotone -- higher ratio =
more like a Lorentzian -- so a candidate is *kept* by the screen if its ratio
clears the threshold. ``became_fitted_peak=True`` is the positive class.

The analysis reports both the SNR-stratified and the all-candidates view: the
discriminator is interesting only on the low-SNR band (SNR 2--5 in user-grid),
since high-SNR candidates are trivially preserved by any sensible threshold.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("stage3-coherence-analyse")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "scratch" / "stage3-coherence-study" / "candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "scratch" / "stage3-coherence-study"


def _load(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(csv_path.open()))
    out: dict[str, np.ndarray] = {}
    out["window_id"] = np.array([int(r["window_id"]) for r in rows])
    out["frequency_mhz"] = np.array([float(r["frequency_mhz"]) for r in rows])
    out["user_snr"] = np.array([float(r["user_grid_snr"]) for r in rows])
    out["active_snr"] = np.array([float(r["active_snr"]) for r in rows])
    out["ratio"] = np.array([float(r["ratio"]) for r in rows])
    out["coherent_snr"] = np.array([float(r["coherent_snr"]) for r in rows])
    out["detected_snr_active"] = np.array(
        [float(r["detected_snr_active"]) for r in rows]
    )
    out["promoted"] = np.array(
        [r["user_grid_promoted"] == "True" for r in rows]
    )
    out["became_peak"] = np.array(
        [r["became_fitted_peak"] == "True" for r in rows]
    )
    out["close_pair"] = np.array([r["close_pair"] == "True" for r in rows])
    return out


def _bucket_table(
    ratio: np.ndarray,
    became_peak: np.ndarray,
    *,
    edges: np.ndarray,
) -> list[dict[str, float]]:
    table: list[dict[str, float]] = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (ratio >= lo) & (ratio < hi)
        if i == edges.size - 2:
            mask = (ratio >= lo) & (ratio <= hi)
        n = int(mask.sum())
        n_true = int(became_peak[mask].sum())
        n_false = n - n_true
        table.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "n_true": n_true,
                "n_false": n_false,
                "frac_true": (n_true / n) if n > 0 else float("nan"),
            }
        )
    return table


def _roc(
    ratio: np.ndarray, became_peak: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the ratio threshold downward; return thresholds, TPR, FPR.

    A candidate is *kept* (predicted positive) when ratio >= threshold.
    """
    order = np.argsort(-ratio)  # high ratio first (most-like-Lorentzian first)
    sorted_ratio = ratio[order]
    sorted_became = became_peak[order].astype(int)
    total_p = int(became_peak.sum())
    total_n = int((~became_peak).sum())
    if total_p == 0 or total_n == 0:
        return np.array([]), np.array([]), np.array([])

    tp = np.cumsum(sorted_became)
    fp = np.cumsum(1 - sorted_became)
    tpr = tp / total_p
    fpr = fp / total_n
    # Use the sorted_ratio values themselves as thresholds (each entry is the
    # threshold at which the next candidate enters the "kept" set).
    return sorted_ratio, tpr, fpr


def _pick_threshold(
    thresholds: np.ndarray,
    tpr: np.ndarray,
    fpr: np.ndarray,
    *,
    min_recall: float,
) -> dict[str, float] | None:
    """Pick threshold = highest ratio with TPR ≥ min_recall, minimising FPR."""
    if thresholds.size == 0:
        return None
    eligible = tpr >= min_recall
    if not np.any(eligible):
        return None
    # Among eligible thresholds, pick the one with smallest FPR; tie-break on
    # highest threshold (which is the strictest kept screen).
    fpr_eligible = np.where(eligible, fpr, np.inf)
    idx_best_fpr = int(np.argmin(fpr_eligible))
    return {
        "threshold": float(thresholds[idx_best_fpr]),
        "tpr": float(tpr[idx_best_fpr]),
        "fpr": float(fpr[idx_best_fpr]),
        "kept_fraction": float((idx_best_fpr + 1) / thresholds.size),
    }


def _plot(data: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    became = data["became_peak"]
    ratio = data["ratio"]

    bins = np.linspace(min(0.0, ratio.min()), max(3.0, ratio.max()), 50)
    axes[0].hist(
        ratio[~became], bins=bins, alpha=0.5, label="became_peak=False",
        color="C3",
    )
    axes[0].hist(
        ratio[became], bins=bins, alpha=0.5, label="became_peak=True",
        color="C0",
    )
    axes[0].set_xlabel("projection ratio (coherent_snr / detected_snr_active)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Ratio distribution by ground-truth label")
    axes[0].legend()
    axes[0].axvline(1.0, color="k", linestyle=":", alpha=0.5)

    thresholds, tpr, fpr = _roc(ratio, became)
    if thresholds.size > 0:
        axes[1].plot(fpr, tpr, marker=".", markersize=3, color="C2")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
        axes[1].set_xlabel("FPR (kept noise / total noise)")
        axes[1].set_ylabel("TPR (kept real / total real)")
        axes[1].set_title("ROC: keep candidates with ratio >= threshold")
        axes[1].legend()
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylim(0.0, 1.05)
        # AUC by trapezoidal rule
        auc = float(np.trapezoid(tpr[::-1], fpr[::-1]))
        axes[1].text(
            0.55,
            0.1,
            f"AUC = {auc:.3f}",
            transform=axes[1].transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.6),
        )

    fig.suptitle(f"Stage 3 coherence screen — {ratio.size} candidates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _stratified(data: dict[str, np.ndarray], min_snr: float, max_snr: float) -> None:
    """Print a quick summary of the SNR-stratified discrimination."""
    snr = data["user_snr"]
    mask = (snr >= min_snr) & (snr < max_snr)
    if mask.sum() == 0:
        return
    ratio = data["ratio"][mask]
    became = data["became_peak"][mask]
    msg = (
        f"  user_snr in [{min_snr:.1f}, {max_snr:.1f}): "
        f"n={int(mask.sum())} became={int(became.sum())} "
        f"(frac={became.mean():.3f})\n"
        f"    ratio: real median={np.median(ratio[became]) if became.any() else float('nan'):.3f}, "
        f"non-real median={np.median(ratio[~became]) if (~became).any() else float('nan'):.3f}"
    )
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Minimum TPR (real-line retention) at the chosen threshold",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(args.csv)
    n = data["ratio"].size
    print(f"Loaded {n} candidates from {args.csv}")
    print(
        f"  became_peak: {int(data['became_peak'].sum())} / {n} "
        f"({data['became_peak'].mean():.3f})"
    )
    print(
        f"  close_pair:  {int(data['close_pair'].sum())} / {n} "
        f"({data['close_pair'].mean():.3f})"
    )

    print("\nRatio buckets (all candidates):")
    edges = np.linspace(0.0, 3.0, 16)
    table = _bucket_table(data["ratio"], data["became_peak"], edges=edges)
    print(f"{'lo':>5} {'hi':>5} {'n':>5} {'true':>5} {'false':>5} {'frac':>6}")
    for row in table:
        print(
            f"{row['lo']:>5.2f} {row['hi']:>5.2f} {row['n']:>5d} "
            f"{row['n_true']:>5d} {row['n_false']:>5d} {row['frac_true']:>6.3f}"
        )

    # SNR-stratified view: the discrimination matters in the SNR 2-5 band.
    print("\nDiscrimination by user_snr band:")
    for lo, hi in [(2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, np.inf)]:
        _stratified(data, lo, hi)

    print("\nDiscrimination on the LOW-SNR band (user_snr 2-5):")
    low_mask = (data["user_snr"] >= 2.0) & (data["user_snr"] < 5.0)
    if low_mask.sum() > 0:
        ratio_low = data["ratio"][low_mask]
        became_low = data["became_peak"][low_mask]
        thresholds, tpr, fpr = _roc(ratio_low, became_low)
        chosen = _pick_threshold(thresholds, tpr, fpr, min_recall=args.min_recall)
        if chosen is None:
            print("  no threshold meets the recall floor")
        else:
            print(
                f"  picked threshold={chosen['threshold']:.4f} "
                f"(TPR={chosen['tpr']:.3f}, FPR={chosen['fpr']:.3f}, "
                f"kept_fraction={chosen['kept_fraction']:.3f})"
            )

    print("\nDiscrimination on ALL candidates (for reference):")
    thresholds, tpr, fpr = _roc(data["ratio"], data["became_peak"])
    chosen_all = _pick_threshold(thresholds, tpr, fpr, min_recall=args.min_recall)
    if chosen_all is None:
        print("  no threshold meets the recall floor")
    else:
        print(
            f"  picked threshold={chosen_all['threshold']:.4f} "
            f"(TPR={chosen_all['tpr']:.3f}, FPR={chosen_all['fpr']:.3f}, "
            f"kept_fraction={chosen_all['kept_fraction']:.3f})"
        )

    # Plot
    plot_path = args.output_dir / "ratio_distribution.png"
    _plot(data, plot_path)
    print(f"\nWrote plot: {plot_path}")


if __name__ == "__main__":
    main()
