"""Knob-agnostic parameter-sweep engine.

Given a :class:`~ftmwpipeline._internal.tuning.registry.KnobSpec` and an input
``.ftmw`` already built through the knob's upstream stage, the engine sweeps the
knob across a grid on a working copy (never mutating the input), reduces each
run to the knob's metric columns, and returns a :class:`SweepResult` carrying
the table rows, a CSV path, an optional plot path, a best-effort recommendation,
and human instructions for applying the chosen value. No knob-specific logic
lives here — it all comes from the ``KnobSpec``.
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, cast

from .registry import KnobSpec


@dataclass(frozen=True)
class PlotContext:
    """Side data a plot adapter may need beyond the swept rows.

    ``ftmw_path`` is the working-copy ``.ftmw`` (built through the knob's
    upstream stage), so an adapter can load source data — e.g. the FID — that
    the per-value stage result does not carry.
    """

    ftmw_path: Path


@dataclass(frozen=True)
class SweepRow:
    """One grid point: the knob value, its measured metric columns, and the
    full stage result.

    ``metrics`` is what the table/CSV report; ``result`` is the raw stage output
    (e.g. a ``NoiseResult`` / ``StartDetectionResult``) so plot adapters can
    render rich per-knob diagnostics. ``result`` is never serialized.
    """

    value: Any
    metrics: Dict[str, Any]
    result: Any = None


@dataclass(frozen=True)
class Recommendation:
    """A best-value pick over the swept grid."""

    value: Any
    metric: str
    metric_value: Any
    reason: str


@dataclass(frozen=True)
class SweepResult:
    """Outcome of a knob sweep across all interfaces."""

    knob: str
    metric_columns: Tuple[str, ...]
    rows: List[SweepRow]
    recommendation: Optional[Recommendation]
    apply_instructions: str
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None

    def as_table(self) -> str:
        """Render the sweep as an aligned text table (always available)."""
        headers = [self.knob.split(".")[-1], *self.metric_columns]
        rows = [
            [_fmt(r.value), *[_fmt(r.metrics.get(c)) for c in self.metric_columns]]
            for r in self.rows
        ]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows)) if rows
            else len(headers[i])
            for i in range(len(headers))
        ]
        sep = "  "
        lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
        lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
        for row in rows:
            lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(row))))
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _safe(path: str) -> str:
    return path.replace(".", "_")


def _make_default_reporter(
    spec: KnobSpec,
    values: Sequence[Any],
    stream: Optional[TextIO] = None,
) -> Callable[[int, int, Any], None]:
    """A terminal progress reporter: a header naming the knob and the grid,
    then a single in-place line that updates as each value completes.

    Writes to stderr by default so it never mixes with the result table a
    caller may print to stdout. Used when ``run_scan`` is not ``quiet`` and no
    custom callback is supplied.
    """
    out = stream if stream is not None else sys.stderr
    leaf = spec.path.split(".")[-1]
    out.write(
        f"Scanning {spec.path} ({spec.stage}) over {len(values)} value(s): "
        f"{', '.join(_fmt(v) for v in values)}\n"
    )
    if spec.see_also:
        out.write(f"  see also: {spec.see_also}\n")
    out.flush()

    def report(done: int, total: int, value: Any) -> None:
        end = "\n" if done == total else ""
        out.write(f"\r  [{done}/{total}] {leaf}={_fmt(value)} done   {end}")
        out.flush()

    return report


def _recommend(spec: KnobSpec, rows: List[SweepRow]) -> Optional[Recommendation]:
    """Best-effort recommendation: a knob-supplied recommender, else the
    built-in min/max over ``primary_metric``; ``None`` when the knob opts out."""
    if not rows:
        return None
    if spec.recommend is not None:
        return cast(Optional[Recommendation], spec.recommend(rows))
    if spec.direction not in ("min", "max") or spec.primary_metric is None:
        return None
    metric = spec.primary_metric
    candidates = [
        r for r in rows
        if isinstance(r.metrics.get(metric), (int, float))
    ]
    if not candidates:
        return None
    pick = (min if spec.direction == "min" else max)(
        candidates, key=lambda r: float(r.metrics[metric])
    )
    return Recommendation(
        value=pick.value,
        metric=metric,
        metric_value=pick.metrics[metric],
        reason=f"{spec.direction}({metric}) over the swept grid",
    )


def _apply_instructions(spec: KnobSpec, rec: Optional[Recommendation]) -> str:
    """How to persist a chosen value: onto the .ftmw or into a preset YAML.

    This is the seam to the deferred preset-emit UX; it tells the user the two
    routes rather than performing the write.
    """
    chosen = _fmt(rec.value) if rec is not None else "<value>"
    leaf = spec.path.split(".")[-1]
    stage_block = spec.path.split(".")[0]
    note = (
        "No automatic recommendation for this knob — inspect the table"
        + (" / plot" if spec.plot is not None else "")
        + " and choose a value."
        if rec is None else
        f"Recommended: {spec.path} = {chosen} ({rec.reason})."
    )
    return (
        f"{note}\n"
        f"To apply a chosen value:\n"
        f"  - persist it onto this experiment by re-running the stage with the "
        f"knob set (the result is saved into the .ftmw), e.g.\n"
        f"      {leaf}={chosen}\n"
        f"  - or record it in an instrument preset YAML under the "
        f"'{stage_block}:' block for routine reuse across experiments."
    )


def run_scan(
    spec: KnobSpec,
    ftmw_path: Path,
    *,
    grid: Optional[Sequence[Any]] = None,
    output_dir: Optional[Path] = None,
    reuse: bool = False,
    make_plot: bool = True,
    interactive: bool = False,
    quiet: bool = False,
    progress: Optional[Callable[[int, int, Any], None]] = None,
) -> SweepResult:
    """Sweep ``spec`` across ``grid`` on a working copy of ``ftmw_path``.

    Parameters
    ----------
    spec :
        The knob to sweep.
    ftmw_path :
        Input ``.ftmw``, already built through ``spec.requires``. Never mutated.
    grid :
        Values to sweep; defaults to ``spec.default_grid``.
    output_dir :
        Where the CSV/plot/working-copy land. Defaults to the current working
        directory; nothing is written outside it.
    reuse :
        Reuse an existing working copy instead of re-copying the input.
    make_plot :
        Render the knob's plot adapter if it has one.
    interactive :
        Show the figure interactively (CLI-only) instead of writing a file.
    quiet :
        Suppress the default terminal progress indicator (header + per-value
        line on stderr). Progress is shown by default on every surface; pass
        ``quiet=True`` (or ``-q`` on the CLI) to silence it.
    progress :
        Optional custom callback invoked as ``progress(done, total, value)``
        after each grid value completes. Overrides the default reporter; with
        a callback set, ``quiet`` is ignored.
    """
    ftmw_path = Path(ftmw_path)
    out = Path(output_dir) if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    work_dir = out / ".tune_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{ftmw_path.stem}__{_safe(spec.path)}.ftmw"
    if not (reuse and work.exists()):
        shutil.copy2(ftmw_path, work)

    values = spec.grid(grid)
    reporter = progress
    if reporter is None and not quiet:
        reporter = _make_default_reporter(spec, values)

    rows: List[SweepRow] = []
    last_result: Any = None
    total = len(values)
    for i, value in enumerate(values):
        last_result = spec.run(work, value)
        rows.append(SweepRow(
            value=value,
            metrics=dict(spec.metric(last_result)),
            result=last_result,
        ))
        if reporter is not None:
            reporter(i + 1, total, value)

    csv_path = out / f"tune_{_safe(spec.path)}_{ftmw_path.stem}.csv"
    _write_csv(csv_path, spec, rows)

    rec = _recommend(spec, rows)
    plot_path = None
    if make_plot and spec.plot is not None:
        plot_path = _render_plot(
            spec, rows, out, ftmw_path.stem, interactive, PlotContext(ftmw_path=work)
        )

    return SweepResult(
        knob=spec.path,
        metric_columns=spec.metric_columns,
        rows=rows,
        recommendation=rec,
        apply_instructions=_apply_instructions(spec, rec),
        csv_path=csv_path,
        plot_path=plot_path,
    )


def _write_csv(path: Path, spec: KnobSpec, rows: List[SweepRow]) -> None:
    leaf = spec.path.split(".")[-1]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([leaf, *spec.metric_columns])
        for r in rows:
            writer.writerow([r.value, *[r.metrics.get(c) for c in spec.metric_columns]])


def _render_plot(
    spec: KnobSpec,
    rows: List[SweepRow],
    out: Path,
    stem: str,
    interactive: bool,
    ctx: "PlotContext",
) -> Optional[Path]:
    fig = spec.plot(spec, rows, ctx)  # type: ignore[misc]
    if fig is None:
        return None
    if interactive:
        import matplotlib.pyplot as plt  # lazy

        plt.show()
        return None
    plot_path = out / f"tune_{_safe(spec.path)}_{stem}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt  # lazy

    plt.close(fig)
    return plot_path
