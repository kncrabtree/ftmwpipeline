"""Build a side-by-side INDEX.md cross-linking Lorentzian and Gaussian artifacts.

Consumes:
- ``dev-docs/research/gaussian-shape/data/exp_2638_unapodized_per_window.csv``
- ``scratch/gaussian-shape-validation/{lorentzian,gaussian}/window_NNN/``

Emits:
- ``scratch/gaussian-shape-validation/INDEX.md`` -- one row per window with
  K, χ²ᵣ, AIC for each shape, ΔAIC, and links to the Lorentzian and Gaussian
  per-window artifacts (detail.png, report.md, audit-trail.png).

Grouping:
- Original stage5-validation set (matches scratch/stage5-validation/INDEX.md).
- Lorentzian-wins (potential clock spurs / collisional regime).
- Strongly-Gaussian-wins (Part-A-style shape-error windows).
- Other (catch-all if the selected set drifts later).

Run from repo root:
    conda run -n ftmwpipeline-dev python \
        scripts/development/stage5-validation/build_shape_comparison_index.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = (
    REPO_ROOT
    / "dev-docs"
    / "research"
    / "gaussian-shape"
    / "data"
    / "exp_2638_unapodized_per_window.csv"
)
OUT_DIR = REPO_ROOT / "scratch" / "gaussian-shape-validation"
L_DIR = OUT_DIR / "lorentzian"
G_DIR = OUT_DIR / "gaussian"

# Original stage5-validation deliberate sample (matches the order in
# scratch/stage5-validation/INDEX.md so a reader can hop directly between
# the apodized-fit validation set and this one).
ORIGINAL_VALIDATION = (
    215,
    16,
    104,
    337,
    64,
    63,
    127,
    260,
    148,
    68,
    132,
    198,
    209,
    269,
    271,
)
LORENTZIAN_WINS = (218, 284, 71, 312, 67, 198, 291, 185, 342)
# Part-A shape-error windows from the Voigt-deficit prototype. w141 is in the
# fixture but not in our re-run set (we picked the original validation set
# instead); list here for cross-reference only.
PART_A_REFERENCE = (141, 213, 310, 355)


def _load_rows() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            wid = int(r["window_id"])
            rows[wid] = {
                "wid": wid,
                "flo": float(r["freq_lo_mhz"]),
                "fhi": float(r["freq_hi_mhz"]),
                "np_L": int(r["n_peaks_lorentz"]),
                "np_G": int(r["n_peaks_gauss"]),
                "c2_L": float(r["chi2r_lorentz"]),
                "c2_G": float(r["chi2r_gauss"]),
                "aic_L": float(r["aic_lorentz"]),
                "aic_G": float(r["aic_gauss"]),
                "dAIC": float(r["delta_aic_lorentz_minus_gauss"]),
            }
    return rows


def _window_links(wid: int) -> tuple[str, str]:
    """Build markdown links to L / G window artifacts (relative to OUT_DIR)."""
    name = f"window_{wid:03d}"
    l_dir = L_DIR / name
    g_dir = G_DIR / name
    l_links: list[str] = []
    g_links: list[str] = []
    if (l_dir / "detail.png").exists():
        l_links.append(f"[detail](lorentzian/{name}/detail.png)")
    if (l_dir / "audit-trail.png").exists():
        l_links.append(f"[audit](lorentzian/{name}/audit-trail.png)")
    if (l_dir / "report.md").exists():
        l_links.append(f"[report](lorentzian/{name}/report.md)")
    if (g_dir / "detail.png").exists():
        g_links.append(f"[detail](gaussian/{name}/detail.png)")
    if (g_dir / "audit-trail.png").exists():
        g_links.append(f"[audit](gaussian/{name}/audit-trail.png)")
    if (g_dir / "report.md").exists():
        g_links.append(f"[report](gaussian/{name}/report.md)")
    return " · ".join(l_links) or "_missing_", " · ".join(g_links) or "_missing_"


def _verdict(d_aic: float) -> str:
    if d_aic > 5.0:
        return "**G** ✓"
    if d_aic < -5.0:
        return "**L** ✓"
    return "tied"


def _row(rows: dict[int, dict], wid: int) -> str:
    r = rows.get(wid)
    if r is None:
        return f"| {wid} | _not in fixture_ | | | | | | | |"
    l_links, g_links = _window_links(wid)
    return (
        f"| {wid} | {r['flo']:.2f}–{r['fhi']:.2f} | "
        f"{r['np_L']} / {r['np_G']} | "
        f"{r['c2_L']:.2f} / {r['c2_G']:.2f} | "
        f"{r['dAIC']:+.1f} | {_verdict(r['dAIC'])} | "
        f"{l_links} | {g_links} |"
    )


def _section(title: str, wids, rows, intro: str | None = None) -> list[str]:
    out = [f"## {title}", ""]
    if intro:
        out.append(intro)
        out.append("")
    out.extend(
        [
            "| wid | freq range (MHz) | K (L / G) | χ²ᵣ (L / G) | ΔAIC | wins | Lorentzian artifacts | Gaussian artifacts |",
            "|---:|---|:---:|:---:|---:|:---:|---|---|",
        ]
    )
    for wid in wids:
        out.append(_row(rows, wid))
    out.append("")
    return out


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing {CSV_PATH}; run compare_shapes.py first to produce it."
        )
    rows = _load_rows()
    selected = set(ORIGINAL_VALIDATION) | set(LORENTZIAN_WINS)
    missing = [w for w in selected if w not in rows]
    if missing:
        print(f"WARNING: windows not present in per-window CSV: {missing}")

    body: list[str] = []
    body.extend(
        [
            "# Stage 5 Lorentzian-vs-Gaussian validation index",
            "",
            "Side-by-side per-window artifacts for the 2638 unapodized fixture.",
            "Each row links to the Lorentzian and Gaussian detail.png / "
            "audit-trail.png / report.md emitted by the validation harness "
            "(`scripts/development/stage5-validation/generate_validation.py`).",
            "Aggregate stats and the comparison figure live under "
            "[`../../dev-docs/research/gaussian-shape/`](../../dev-docs/research/gaussian-shape/).",
            "",
            "ΔAIC = AIC(L) − AIC(G). **Positive** = Gaussian preferred; "
            "**negative** = Lorentzian preferred. ±5 is the conventional "
            "decisive-evidence threshold.",
            "",
        ]
    )
    body.extend(
        _section(
            "Original stage5-validation set",
            ORIGINAL_VALIDATION,
            rows,
            intro=(
                "Same window IDs as `scratch/stage5-validation/INDEX.md` "
                "(though the unapodized fixture's window plan has slightly "
                "different freq boundaries because Stage 4 was re-run after "
                "rebuilding without apodization)."
            ),
        )
    )
    body.extend(
        _section(
            "Lorentzian-wins (potential clock spurs / collisional regime)",
            LORENTZIAN_WINS,
            rows,
            intro=(
                "Sorted by ΔAIC (most-negative first in the underlying CSV). "
                "Two patterns to look for in the per-window plots: "
                "(a) narrow lines with χ²ᵣ_L ≈ 1 that the Gaussian "
                "model over-broadens (collisional / rotationally-cold "
                "regime — true Lorentzian shape), and "
                "(b) bins where Gaussian χ²ᵣ blows up dramatically — "
                "candidate clock spurs whose CW time-domain content the "
                "Gaussian envelope cannot accommodate. **w218** is the "
                "extreme case (χ²ᵣ_G = 213)."
            ),
        )
    )
    body.extend(
        _section(
            "Part A reference (Voigt-deficit prototype targets)",
            PART_A_REFERENCE,
            rows,
            intro=(
                "The four windows the per-window joint `(τ_L, τ_G)` LSQ in "
                "`dev-docs/planning/stage5-voigt-deficit.md` flagged as the "
                "strongest Gaussian-preference candidates on 2638. Three of "
                "the four (w141, w213, w310) win Gaussian by large ΔAIC; "
                "w355 went the other way (the open-question case the "
                "planning doc calls out for deep-dive)."
            ),
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "INDEX.md"
    out_path.write_text("\n".join(body) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
