"""Step 2 -- pick the leading variant per knob from the variants_summary CSV.

Reads ``data/probe_all_variants_summary.csv``, groups by ``(knob, shape)``,
and applies the heuristic from the session plan:

    lowest p95 chi2r subject to median chi2r <= baseline_median + 5 %

The baseline per shape is the ``baseline__<shape>`` row of the same knob's
group (which exists because each probe's grid includes the knob's hard
default and the harness re-uses ``baseline__<shape>`` across knobs).

Outputs:

* prints a one-line per-knob/shape leader to stdout.
* writes ``report.md`` with a markdown table and the choice for Step 3
  (the single leading variant whose per-window artifacts get emitted).

The "single leader" choice tie-breaks across knobs by picking the
(knob, shape) with the largest absolute p95 chi2r improvement vs. its
baseline -- the variant most likely to surface classification-relevant
differences in the per-window walkthrough. If the gain is below a small
epsilon, falls back to the Gaussian baseline (the documented
end-state shape).
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).parent
DATA = HERE / "data"
SUMMARY_CSV = DATA / "probe_all_variants_summary.csv"
REPORT_MD = HERE / "report.md"

# Acceptance band on median chi2r vs baseline (allow up to +5%).
MEDIAN_BUDGET_FRACTION = 0.05
# Below this absolute p95 chi2r improvement, treat the variant as
# "indistinguishable from baseline" for the Step 3 leader choice.
MIN_P95_GAIN = 0.05

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pick-leading-variants")


@dataclass
class SummaryRow:
    variant_id: str
    knob: str
    knob_value: str
    shape: str
    n_windows: int
    chi2r_median: float
    chi2r_p95: float
    chi2r_max: float
    n_chi2r_gt_5: int
    n_chi2r_gt_10: int
    n_fitted_peaks_total: int
    n_fixed_total: int


def _parse_value(s: str) -> Any:
    try:
        return float(s)
    except ValueError:
        return s


def _load_rows() -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with SUMMARY_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(SummaryRow(
                variant_id=r["variant_id"],
                knob=r["knob"],
                knob_value=r["knob_value"],
                shape=r["shape"],
                n_windows=int(r["n_windows"]),
                chi2r_median=float(r["chi2r_median"]),
                chi2r_p95=float(r["chi2r_p95"]),
                chi2r_max=float(r["chi2r_max"]),
                n_chi2r_gt_5=int(r["n_chi2r_gt_5"]),
                n_chi2r_gt_10=int(r["n_chi2r_gt_10"]),
                n_fitted_peaks_total=int(r["n_fitted_peaks_total"]),
                n_fixed_total=int(r["n_fixed_total"]),
            ))
    return rows


def _group_key(r: SummaryRow) -> Tuple[str, str]:
    return (r.knob, r.shape)


def _is_baseline(r: SummaryRow) -> bool:
    return r.variant_id.startswith("baseline__")


def _pick_leader_for_group(
    rows: List[SummaryRow],
) -> Tuple[Optional[SummaryRow], Optional[SummaryRow]]:
    """Return ``(baseline_row, leader_row)`` for one (knob, shape) group.

    ``leader_row`` is the row with the lowest p95 chi2r whose median
    chi2r is within +5 % of the baseline's median. ``leader_row`` may
    equal ``baseline_row`` if no variant beats it.
    """
    baseline = next((r for r in rows if _is_baseline(r)), None)
    if baseline is None:
        return None, None
    cap = baseline.chi2r_median * (1.0 + MEDIAN_BUDGET_FRACTION)
    eligible = [r for r in rows if r.chi2r_median <= cap]
    if not eligible:
        return baseline, baseline
    leader = min(eligible, key=lambda r: r.chi2r_p95)
    return baseline, leader


def _format_leader_line(
    baseline: SummaryRow, leader: SummaryRow,
) -> str:
    same = leader.variant_id == baseline.variant_id
    if same:
        return (
            f"  {leader.knob:>40s} / {leader.shape:>10s}: "
            f"baseline holds (med={baseline.chi2r_median:.3f}, "
            f"p95={baseline.chi2r_p95:.3f})"
        )
    d_med = leader.chi2r_median - baseline.chi2r_median
    d_p95 = leader.chi2r_p95 - baseline.chi2r_p95
    return (
        f"  {leader.knob:>40s} / {leader.shape:>10s}: "
        f"{leader.knob_value:>6s} (med={leader.chi2r_median:.3f} "
        f"[{d_med:+.3f}], p95={leader.chi2r_p95:.3f} [{d_p95:+.3f}])"
    )


def main() -> None:
    rows = _load_rows()
    by_group: Dict[Tuple[str, str], List[SummaryRow]] = defaultdict(list)
    for r in rows:
        by_group[_group_key(r)].append(r)

    leaders: List[Tuple[SummaryRow, SummaryRow]] = []
    print("====== Per-(knob, shape) leaders ======")
    for key in sorted(by_group):
        baseline, leader = _pick_leader_for_group(by_group[key])
        if baseline is None or leader is None:
            print(f"  {key}: no baseline row -- run probes first")
            continue
        print(_format_leader_line(baseline, leader))
        leaders.append((baseline, leader))

    # Step 3 input -- the single leader to walk through per-window.
    # Pick the (knob, shape) pair with the largest absolute p95 chi2r
    # improvement; fall back to the Gaussian baseline if no variant
    # clears MIN_P95_GAIN (the prior probes' empirical regime).
    best_pair: Optional[Tuple[SummaryRow, SummaryRow]] = None
    best_gain = 0.0
    for baseline, leader in leaders:
        gain = baseline.chi2r_p95 - leader.chi2r_p95
        if gain > best_gain:
            best_gain = gain
            best_pair = (baseline, leader)
    if best_pair is None or best_gain < MIN_P95_GAIN:
        gauss_baseline = next(
            (r for r in rows if r.variant_id == "baseline__gaussian"),
            None,
        )
        if gauss_baseline is None:
            raise SystemExit(
                "no baseline__gaussian row in summary; cannot pick a leader"
            )
        chosen = gauss_baseline
        rationale = (
            f"No variant beat baseline by >= {MIN_P95_GAIN} on p95 chi2r; "
            f"falling back to baseline__gaussian (the documented "
            f"end-state shape)."
        )
    else:
        baseline, leader = best_pair
        chosen = leader
        rationale = (
            f"Largest p95 chi2r improvement: {best_gain:.3f} "
            f"({leader.knob}={leader.knob_value}, shape={leader.shape})."
        )

    print()
    print(f"Step 3 leader: {chosen.variant_id}")
    print(f"  rationale: {rationale}")
    print(f"  fixture:   scratch/stage5-gaussian-audit/runs/{chosen.variant_id}.ftmw")

    # ---------------------------------------------------------------
    # report.md
    # ---------------------------------------------------------------
    md_lines: List[str] = [
        "# Stage 5 Gaussian-path parameter optimization audit -- progress",
        "",
        "Step 1 (parameter sweep) and Step 2 (leader selection) report.",
        "Step 4 (per-window classification) is the user's out-of-session",
        "task. Step 5 (cross-reference) updates this report once the",
        "classifications land.",
        "",
        "## Step 2 -- per-(knob, shape) leaders",
        "",
        "Selection heuristic: lowest p95 chi2r subject to median chi2r",
        "<= baseline_median * 1.05.",
        "",
        "| knob | shape | leader value | median chi2r (Delta) | p95 chi2r (Delta) | n>5 | n>10 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for baseline, leader in leaders:
        same = leader.variant_id == baseline.variant_id
        leader_v = "baseline" if same else leader.knob_value
        d_med = leader.chi2r_median - baseline.chi2r_median
        d_p95 = leader.chi2r_p95 - baseline.chi2r_p95
        md_lines.append(
            f"| `{leader.knob}` | {leader.shape} | {leader_v} | "
            f"{leader.chi2r_median:.3f} ({d_med:+.3f}) | "
            f"{leader.chi2r_p95:.3f} ({d_p95:+.3f}) | "
            f"{leader.n_chi2r_gt_5} | {leader.n_chi2r_gt_10} |"
        )

    md_lines += [
        "",
        "## Step 3 -- leading variant for per-window walkthrough",
        "",
        f"**Chosen:** `{chosen.variant_id}`",
        "",
        f"- knob: `{chosen.knob}`",
        f"- value: `{chosen.knob_value}`",
        f"- shape: `{chosen.shape}`",
        f"- chi2r median / p95 / max: "
        f"{chosen.chi2r_median:.3f} / "
        f"{chosen.chi2r_p95:.3f} / "
        f"{chosen.chi2r_max:.3f}",
        f"- n_windows > 5: {chosen.n_chi2r_gt_5}; > 10: {chosen.n_chi2r_gt_10}",
        f"- rationale: {rationale}",
        "",
        "Fixture: "
        f"`scratch/stage5-gaussian-audit/runs/{chosen.variant_id}.ftmw`",
        "",
        "Per-window artifacts emitted by",
        "`scripts/development/stage5-validation/generate_validation.py` "
        "with `--all-windows --variant-id <vid>`.",
        "",
        "## Step 4 -- user classification (out-of-session)",
        "",
        "Open the emitted",
        f"`scratch/stage5-validation-{chosen.variant_id}/windows.toml`",
        "and fill `classification = \"...\"` only for windows that look",
        "wrong. Leave the rest empty (the convention keeps the typing",
        "budget small on the full 386-391 windows).",
        "",
        "## Step 5 -- cross-reference (next session)",
        "",
        "_Pending; populated once Step 4 lands._",
        "",
    ]
    REPORT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
