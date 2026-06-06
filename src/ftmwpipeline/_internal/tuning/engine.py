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

from ...io.noise_settings_serialization import STAGE2_NOISE_SETTINGS_PATH
from ...io.peak_detection_settings_serialization import STAGE3_PEAKS_SETTINGS_PATH
from ...io.stage_fit_settings_serialization import STAGE_FIT_PATH
from ...io.tau_calibration_settings_serialization import STAGE2B_TAU_SETTINGS_PATH
from ...io.window_planning_settings_serialization import STAGE4_WINDOWS_SETTINGS_PATH
from .fit_support import FitWindowSelection
from .registry import KnobSpec

# Persisted settings group per stage prefix. A sweep clears the knob's stage
# block before each per-value run so the swept value (which enters the resolver
# at the preset layer) is not shadowed by the previous value's persisted
# settings: under the canonical precedence ``persisted > preset``, re-running a
# stage on a working copy that already carries a persisted block would otherwise
# pin every value to the first run's settings. Stages 0/1 are exempt -- start
# detection runs with ``stamp=False`` and the FT resolver has no preset layer.
_STAGE_SETTINGS_GROUPS = {
    "stage2": STAGE2_NOISE_SETTINGS_PATH,
    "stage2b": STAGE2B_TAU_SETTINGS_PATH,
    "stage3": STAGE3_PEAKS_SETTINGS_PATH,
    "stage4": STAGE4_WINDOWS_SETTINGS_PATH,
    "stage5": STAGE_FIT_PATH,
}


def _clear_persisted_stage_settings(work: Path, knob_path: str) -> None:
    """Delete the knob's stage settings block from the working copy, if present.

    Keeps each swept value authoritative under the ``persisted > preset``
    precedence: the swept value is injected at the preset layer, so any settings
    a prior value persisted to the working copy must be removed first or they
    would outrank it. Only the settings group is removed -- a knob-prepared
    working copy (e.g. a trimmed Stage 5 window plan) is untouched. A no-op for
    stages with no preset-driven settings block.
    """
    group = _STAGE_SETTINGS_GROUPS.get(knob_path.split(".")[0])
    if group is None:
        return
    import h5py

    with h5py.File(work, "a") as h5f:
        if group in h5f:
            del h5f[group]


@dataclass(frozen=True)
class PlotContext:
    """Side data a plot adapter may need beyond the swept rows.

    ``ftmw_path`` is the working-copy ``.ftmw`` (built through the knob's
    upstream stage), so an adapter can load source data — e.g. the FID — that
    the per-value stage result does not carry.

    The ``zoom_*`` fields let the user steer the per-region zoom panels of the
    region-based adapters (Stage 3 peak detection, Stage 4 window planning):
    ``zoom_regions`` pins explicit ``(lo_mhz, hi_mhz)`` windows and overrides the
    adapter's divergence auto-selection; when it is empty the adapter
    auto-selects as usual but honours ``n_zoom`` / ``zoom_width_mhz`` (when set)
    for how many regions to pick and how wide each is. Adapters with no zoom
    panels ignore these.
    """

    ftmw_path: Path
    zoom_regions: Tuple[Tuple[float, float], ...] = ()
    n_zoom: Optional[int] = None
    zoom_width_mhz: Optional[float] = None


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
            (
                max(len(headers[i]), *(len(row[i]) for row in rows))
                if rows
                else len(headers[i])
            )
            for i in range(len(headers))
        ]
        sep = "  "
        lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
        lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
        for row in rows:
            lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(row))))
        return "\n".join(lines)


@dataclass(frozen=True)
class BatchItem:
    """One knob's outcome within a batch scan.

    Exactly one of ``result`` / ``error`` is set: ``result`` on success, or
    ``error`` (the exception message) when that knob's scan failed — e.g. its
    required stage is absent on the file. A batch never aborts on one failure.
    """

    knob: str
    result: Optional[SweepResult] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None


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
    candidates = [r for r in rows if isinstance(r.metrics.get(metric), (int, float))]
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

    Points the user at the ``settings`` change-grammar rather than performing
    the write here.
    """
    chosen = _fmt(rec.value) if rec is not None else "<value>"
    stage_block = spec.path.split(".")[0]
    note = (
        "No automatic recommendation for this knob — inspect the table"
        + (" / plot" if spec.plot is not None else "")
        + " and choose a value."
        if rec is None
        else f"Recommended: {spec.path} = {chosen} ({rec.reason})."
    )
    return (
        f"{note}\n"
        f"To apply a chosen value:\n"
        f"  - persist it onto this experiment (invalidates downstream stages):\n"
        f"      ftmwpipeline settings set <file.ftmw> {spec.path} {chosen}\n"
        f"  - or capture this experiment's chosen values as a reusable preset:\n"
        f"      ftmwpipeline settings export <file.ftmw> <out.yml> {stage_block}"
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
    zoom_regions: Optional[Sequence[Tuple[float, float]]] = None,
    n_zoom: Optional[int] = None,
    zoom_width_mhz: Optional[float] = None,
    fit_top_snr: int = 3,
    fit_sample: int = 20,
    fit_freqs: Optional[Sequence[float]] = None,
    fit_sample_seed: int = 0,
    fit_all: bool = False,
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
    zoom_regions :
        Explicit ``(lo_mhz, hi_mhz)`` windows for the region-based plot adapters
        (Stage 3 / Stage 4). When given, they replace the divergence
        auto-selection; otherwise the adapter auto-selects.
    n_zoom, zoom_width_mhz :
        How many regions to auto-select and how wide each is, when
        ``zoom_regions`` is not given. ``None`` keeps the adapter's defaults.
    fit_top_snr, fit_sample, fit_freqs, fit_sample_seed, fit_all :
        Window selection for the fit knobs that carry a ``prepare`` hook
        (Stage 5): re-fit only the ``fit_top_snr`` brightest windows plus a
        seeded ``fit_sample`` random sample plus the windows nearest each
        ``fit_freqs`` value, rather than the whole plan. ``fit_all`` re-fits
        every window. Knobs without a prepare hook ignore these.
    progress :
        Optional custom callback invoked as ``progress(done, total, value)``
        after each grid value completes. Overrides the default reporter; with
        a callback set, ``quiet`` is ignored.
    """
    ftmw_path = Path(ftmw_path)
    out = Path(output_dir) if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    work_dir = out / ".scan_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    values = spec.grid(grid)
    work = work_dir / f"{ftmw_path.stem}__{_safe(spec.path)}.ftmw"
    if not (reuse and work.exists()):
        shutil.copy2(ftmw_path, work)
        # A knob may reduce/condition the working copy once before the sweep
        # (Stage 5 fit knobs trim the window plan to a representative subset so
        # each value re-fits only a handful of windows). It gets the knob spec
        # and resolved grid so SNR-threshold knobs can straddle-sample. Runs only
        # on a fresh copy — a reused working file was already prepared.
        if spec.prepare is not None:
            spec.prepare(
                work,
                FitWindowSelection(
                    top_snr=fit_top_snr,
                    sample=fit_sample,
                    freqs=tuple(fit_freqs) if fit_freqs else (),
                    sample_seed=fit_sample_seed,
                    fit_all=fit_all,
                ),
                spec,
                values,
            )
    reporter = progress
    if reporter is None and not quiet:
        reporter = _make_default_reporter(spec, values)

    rows: List[SweepRow] = []
    last_result: Any = None
    total = len(values)
    for i, value in enumerate(values):
        _clear_persisted_stage_settings(work, spec.path)
        last_result = spec.run(work, value)
        rows.append(
            SweepRow(
                value=value,
                metrics=dict(spec.metric(last_result)),
                result=last_result,
            )
        )
        if reporter is not None:
            reporter(i + 1, total, value)

    csv_path = out / f"scan_{_safe(spec.path)}_{ftmw_path.stem}.csv"
    _write_csv(csv_path, spec, rows)

    rec = _recommend(spec, rows)
    plot_path = None
    if make_plot and spec.plot is not None:
        ctx = PlotContext(
            ftmw_path=work,
            zoom_regions=tuple(zoom_regions) if zoom_regions else (),
            n_zoom=n_zoom,
            zoom_width_mhz=zoom_width_mhz,
        )
        plot_path = _render_plot(spec, rows, out, ftmw_path.stem, interactive, ctx)

    return SweepResult(
        knob=spec.path,
        metric_columns=spec.metric_columns,
        rows=rows,
        recommendation=rec,
        apply_instructions=_apply_instructions(spec, rec),
        csv_path=csv_path,
        plot_path=plot_path,
    )


def run_scan_batch(
    specs: Sequence[KnobSpec],
    ftmw_path: Path,
    *,
    output_dir: Optional[Path] = None,
    reuse: bool = False,
    make_plot: bool = True,
    quiet: bool = False,
    zoom_regions: Optional[Sequence[Tuple[float, float]]] = None,
    n_zoom: Optional[int] = None,
    zoom_width_mhz: Optional[float] = None,
    fit_top_snr: int = 3,
    fit_sample: int = 20,
    fit_freqs: Optional[Sequence[float]] = None,
    fit_sample_seed: int = 0,
    fit_all: bool = False,
) -> List[BatchItem]:
    """Sweep every knob in ``specs`` sequentially, each on its default grid.

    A convenience over :func:`run_scan` for reviewing a whole stage / sub-block
    at once (pair with :func:`registry.list_knobs` and its selector). Each knob
    runs independently on its own working copy of ``ftmw_path`` (never mutated);
    a knob whose scan raises — e.g. its required stage is absent — is recorded as
    a failed :class:`BatchItem` and the batch continues. Per-knob progress is the
    same stderr header :func:`run_scan` prints unless ``quiet=True``.
    """
    items: List[BatchItem] = []
    for spec in specs:
        try:
            result = run_scan(
                spec,
                ftmw_path,
                output_dir=output_dir,
                reuse=reuse,
                make_plot=make_plot,
                quiet=quiet,
                zoom_regions=zoom_regions,
                n_zoom=n_zoom,
                zoom_width_mhz=zoom_width_mhz,
                fit_top_snr=fit_top_snr,
                fit_sample=fit_sample,
                fit_freqs=fit_freqs,
                fit_sample_seed=fit_sample_seed,
                fit_all=fit_all,
            )
            items.append(BatchItem(knob=spec.path, result=result))
        except Exception as e:  # one knob's failure must not abort the batch
            items.append(BatchItem(knob=spec.path, error=str(e)))
    return items


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
    plot_path = out / f"scan_{_safe(spec.path)}_{stem}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt  # lazy

    plt.close(fig)
    return plot_path
