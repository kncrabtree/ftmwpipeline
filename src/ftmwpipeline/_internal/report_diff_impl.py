"""Post-curation before/after diff report (the ``report diff`` deliverable).

A targeted, read-only HTML document that shows -- side by side -- every window
that is *materially* different between the automatic Stage 5 fit and the current
curated fit. It folds in both the windows the user edited and the dependents the
contributor-edit cascade changed, so the changes can be evaluated in one place
before they are committed, without maintaining and diffing two separate files.

The two states already live on disk: ``/stage5_fitting`` is the current
(curated) fit and ``/stage5_fitting_baseline`` is the automatic-fit snapshot
taken before the first edit (see :func:`ftmwpipeline._internal.stage6_impl.
_snapshot_stage5_baseline`). A window-by-window diff of the two is the report.

The report is rendered with the same per-window painter the Level-3 report uses
(:func:`render_fit_panels_impl` over a detail bundle whose ``fit`` is swapped for
the baseline), so the before/after panels are visually identical in style to the
rest of the reporting. Only the ``|X|`` + model panel is shown per side; the
report is a focused comparison, not the full per-window page.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py

from ..core.data_structures import FittingResult, SpectrumFit
from ..fitting.validation import DEFAULT_CHI2R_NOISE_FLOOR, shape_error_fraction
from .stage5_impl import _resolve_detail_bundle, render_fit_panels_impl
from .stage6_impl import STAGE5_BASELINE_GROUP

# A matched peak counts as "moved" when it shifts at least this fraction of a
# resolution element; below it the move is fit jitter, not a curation effect.
_DEFAULT_SHIFT_RES: float = 0.25
# A window's reduced chi-squared counts as changed at this relative magnitude.
_DEFAULT_CHI2R_REL: float = 0.10
# A window's shape-error fraction counts as changed at this absolute magnitude.
_DEFAULT_EPS_ABS: float = 0.01


@dataclasses.dataclass
class _WindowDiff:
    """The baseline -> current diff for one window."""

    window_id: int
    n_base: int
    n_cur: int
    chi2r_base: float
    chi2r_cur: float
    eps_base: float
    eps_cur: float
    max_shift_khz: float
    n_added: int
    n_removed: int
    material: bool

    @property
    def freq_range(self) -> Optional[Tuple[float, float]]:
        return self._freq_range

    _freq_range: Optional[Tuple[float, float]] = None


def _window_snr_max(wf: Optional[FittingResult]) -> float:
    if wf is None:
        return 0.0
    snrs = [
        float(p.snr)
        for p in wf.fitted_peaks
        if p.snr is not None and math.isfinite(float(p.snr))
    ]
    return max(snrs, default=0.0)


def _window_eps(wf: Optional[FittingResult]) -> float:
    if wf is None:
        return 0.0
    chi2r = float(getattr(wf, "reduced_chi2", float("nan")))
    if not math.isfinite(chi2r):
        return 0.0
    return shape_error_fraction(chi2r, _window_snr_max(wf), DEFAULT_CHI2R_NOISE_FLOOR)


def _match_shift_khz(
    base_wf: Optional[FittingResult],
    cur_wf: Optional[FittingResult],
    tol_mhz: float,
) -> Tuple[float, int, int]:
    """Greedy injective nearest-frequency match between the two peak sets.

    Returns ``(max_matched_shift_khz, n_added, n_removed)``: the largest
    frequency move over matched pairs (kHz), and the counts of peaks present only
    in the current / only in the baseline fit (within ``tol_mhz``).
    """
    base = sorted(
        float(p.frequency_mhz) for p in (base_wf.fitted_peaks if base_wf else [])
    )
    cur = sorted(
        float(p.frequency_mhz) for p in (cur_wf.fitted_peaks if cur_wf else [])
    )
    used_cur: set = set()
    max_shift = 0.0
    matched = 0
    for fb in base:
        best_k = -1
        best_d = tol_mhz
        for k, fc in enumerate(cur):
            if k in used_cur:
                continue
            d = abs(fc - fb)
            if d <= best_d:
                best_d, best_k = d, k
        if best_k >= 0:
            used_cur.add(best_k)
            matched += 1
            max_shift = max(max_shift, best_d * 1e3)
    n_removed = len(base) - matched
    n_added = len(cur) - matched
    return max_shift, n_added, n_removed


def _diff_window(
    wid: int,
    base_wf: Optional[FittingResult],
    cur_wf: Optional[FittingResult],
    *,
    res_element_mhz: float,
    shift_res: float,
    chi2r_rel: float,
    eps_abs: float,
) -> _WindowDiff:
    n_base = len(base_wf.fitted_peaks) if base_wf else 0
    n_cur = len(cur_wf.fitted_peaks) if cur_wf else 0
    chi2r_base = (
        float(getattr(base_wf, "reduced_chi2", float("nan")))
        if base_wf
        else float("nan")
    )
    chi2r_cur = (
        float(getattr(cur_wf, "reduced_chi2", float("nan"))) if cur_wf else float("nan")
    )
    eps_base = _window_eps(base_wf)
    eps_cur = _window_eps(cur_wf)
    tol_mhz = res_element_mhz if res_element_mhz > 0 else 0.05
    max_shift_khz, n_added, n_removed = _match_shift_khz(base_wf, cur_wf, tol_mhz)

    shift_thresh_khz = (
        shift_res * res_element_mhz * 1e3 if res_element_mhz > 0 else 25.0
    )
    chi2r_changed = False
    if math.isfinite(chi2r_base) and math.isfinite(chi2r_cur):
        denom = max(abs(chi2r_base), 1.0)
        chi2r_changed = abs(chi2r_cur - chi2r_base) / denom >= chi2r_rel
    eps_changed = abs(eps_cur - eps_base) >= eps_abs
    material = (
        (n_cur != n_base)
        or (max_shift_khz >= shift_thresh_khz)
        or chi2r_changed
        or eps_changed
        or (base_wf is None) != (cur_wf is None)
    )

    fr = None
    src = cur_wf or base_wf
    if src is not None and src.window is not None and src.window.freq_range is not None:
        fr = src.window.freq_range
    return _WindowDiff(
        window_id=wid,
        n_base=n_base,
        n_cur=n_cur,
        chi2r_base=chi2r_base,
        chi2r_cur=chi2r_cur,
        eps_base=eps_base,
        eps_cur=eps_cur,
        max_shift_khz=max_shift_khz,
        n_added=n_added,
        n_removed=n_removed,
        material=material,
        _freq_range=fr,
    )


def _load_baseline_fit(path: str) -> Optional[SpectrumFit]:
    """Load the automatic-fit baseline snapshot, or ``None`` if not present."""
    from ..io.fitting_serialization import load_spectrum_fit_from_hdf5

    with h5py.File(path, "r") as h5f:
        if STAGE5_BASELINE_GROUP not in h5f:
            return None
        return load_spectrum_fit_from_hdf5(h5f[STAGE5_BASELINE_GROUP])


def _panel_png_data_uri(bundle: object, wid: int, dpi: int) -> Optional[str]:
    """Render the ``|X|`` + model panel for one window from ``bundle`` to a
    base64 PNG data URI, or ``None`` when the window has no such panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report_html_impl import _figure_png_bytes

    try:
        panels = render_fit_panels_impl("", wid, bundle=bundle, with_overview=False)  # type: ignore[arg-type]
    except (KeyError, ValueError, AttributeError):
        return None
    fig = panels.get("mag")
    uri: Optional[str] = None
    try:
        if fig is not None:
            raw = _figure_png_bytes(fig, dpi=dpi)
            uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    finally:
        for f in panels.values():
            plt.close(f)
    return uri


_DIFF_CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1c2024}
header{background:#1c2530;color:#fff;padding:18px 26px}
header h1{margin:0 0 4px;font-size:20px}
header .sub{opacity:.8;font-size:13px}
main{max-width:1180px;margin:0 auto;padding:18px 26px 60px}
.notice{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:28px;
 text-align:center;color:#586069;font-size:15px;margin-top:24px}
section.win{background:#fff;border:1px solid #d8dde3;border-radius:8px;
 margin:18px 0;overflow:hidden}
section.win>h2{margin:0;padding:11px 16px;font-size:15px;background:#eef1f4;
 border-bottom:1px solid #d8dde3}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;padding:11px 16px;font-size:13px;
 border-bottom:1px solid #eceff2}
.stats b{font-weight:600}
.delta-up{color:#b42318}.delta-dn{color:#067647}.delta-0{color:#586069}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
 font-weight:600}
.tag.add{background:#e7f6ec;color:#067647}.tag.rm{background:#fdeceb;color:#b42318}
.tag.shift{background:#fff4e5;color:#9a5b00}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:0}
.panels figure{margin:0;padding:10px 12px}
.panels figure:first-child{border-right:1px solid #eceff2}
.panels figcaption{font-size:12px;font-weight:600;color:#586069;
 text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.panels img{width:100%;height:auto;display:block;background:#fff}
.panels .missing{color:#9aa1a8;font-style:italic;padding:30px 0;text-align:center}
"""


def _delta_span(delta: float, *, lower_is_better: bool, fmt: str) -> str:
    if abs(delta) < 10**-6:
        cls = "delta-0"
    elif (delta < 0) == lower_is_better:
        cls = "delta-dn"
    else:
        cls = "delta-up"
    sign = "+" if delta > 0 else ""
    return f'<span class="{cls}">{sign}{format(delta, fmt)}</span>'


def _window_section(
    d: _WindowDiff,
    before_uri: Optional[str],
    after_uri: Optional[str],
) -> str:
    if d.freq_range is not None:
        lo, hi = d.freq_range
        title = f"Window {d.window_id} &mdash; {lo:.3f}&ndash;{hi:.3f} MHz"
    else:
        title = f"Window {d.window_id}"

    tags: List[str] = []
    if d.n_added:
        tags.append(f'<span class="tag add">+{d.n_added} peak</span>')
    if d.n_removed:
        tags.append(f'<span class="tag rm">&minus;{d.n_removed} peak</span>')
    if d.max_shift_khz >= 1.0:
        tags.append(f'<span class="tag shift">shift {d.max_shift_khz:.0f} kHz</span>')

    def _fchi(v: float) -> str:
        return f"{v:.3g}" if math.isfinite(v) else "&mdash;"

    dchi = (
        d.chi2r_cur - d.chi2r_base
        if math.isfinite(d.chi2r_cur) and math.isfinite(d.chi2r_base)
        else float("nan")
    )
    dchi_str = (
        _delta_span(dchi, lower_is_better=True, fmt=".3g")
        if math.isfinite(dchi)
        else "&mdash;"
    )
    deps = d.eps_cur - d.eps_base
    stats = (
        f'<div class="stats">'
        f"<span>peaks <b>{d.n_base} &rarr; {d.n_cur}</b></span>"
        f"<span>&chi;&sup2;&#7523; <b>{_fchi(d.chi2r_base)} &rarr; {_fchi(d.chi2r_cur)}</b> ({dchi_str})</span>"
        f"<span>&epsilon; <b>{d.eps_base*100:.2f}% &rarr; {d.eps_cur*100:.2f}%</b> "
        f"({_delta_span(deps*100, lower_is_better=True, fmt='+.2f')}%)</span>"
        f'<span>{"".join(tags)}</span>'
        f"</div>"
    )

    def _panel(uri: Optional[str], label: str) -> str:
        if uri is None:
            body = '<div class="missing">(window absent)</div>'
        else:
            body = f'<img src="{uri}" alt="{label} window {d.window_id}">'
        return f"<figure><figcaption>{label}</figcaption>{body}</figure>"

    panels = (
        f'<div class="panels">'
        f"{_panel(before_uri, 'Before (automatic)')}"
        f"{_panel(after_uri, 'After (curated)')}"
        f"</div>"
    )
    return f'<section class="win"><h2>{title}</h2>{stats}{panels}</section>'


def report_diff_impl(
    file_path: Union[Path, str],
    *,
    output_dir: Optional[Union[Path, str]] = None,
    dpi: int = 110,
    shift_res: float = _DEFAULT_SHIFT_RES,
    chi2r_rel: float = _DEFAULT_CHI2R_REL,
    eps_abs: float = _DEFAULT_EPS_ABS,
) -> str:
    """Render the post-curation before/after diff report; return its path.

    Compares the automatic-fit baseline snapshot (``/stage5_fitting_baseline``)
    with the current curated fit (``/stage5_fitting``) and writes a self-contained
    HTML document with a side-by-side ``|X|``-and-model panel for every window
    that differs materially (a peak added/removed, a peak moved at least
    ``shift_res`` resolution elements, or the reduced chi-squared / shape-error
    fraction changed beyond ``chi2r_rel`` / ``eps_abs``). When no baseline
    snapshot exists (no curation edits have been made), a valid report stating
    that is written instead.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    output_dir :
        Directory to write the report into (created if absent); defaults to the
        current working directory.
    dpi :
        Resolution for the per-window panels.
    shift_res, chi2r_rel, eps_abs :
        Materiality thresholds (resolution-element fraction for a peak move,
        relative reduced-chi-squared change, absolute shape-error-fraction change).

    Returns
    -------
    str
        Path to the generated ``<stem>_diff.html`` file.

    Raises
    ------
    ValueError
        When Stage 5 has not been run (no current fit to diff).
    """
    path = str(file_path)
    out_dir = Path.cwd() if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    out_path = out_dir / f"{stem}_diff.html"

    baseline_fit = _load_baseline_fit(path)
    if baseline_fit is None:
        out_path.write_text(
            _shell(
                stem,
                "No curation changes",
                '<div class="notice">No curation edits have been made: the current '
                "fit is the automatic fit, so there is nothing to compare. Make a "
                "<code>review</code> edit, then re-run <code>report diff</code>.</div>",
            )
        )
        return str(out_path)

    bundle = _resolve_detail_bundle(path)
    current_fit: SpectrumFit = bundle.fit
    res_element_mhz = 1.0 / bundle.acquisition_us if bundle.acquisition_us > 0 else 0.0

    cur_by_id: Dict[int, FittingResult] = {
        int(wf.window_id): wf
        for wf in current_fit.window_fits
        if wf.window_id is not None
    }
    base_by_id: Dict[int, FittingResult] = {
        int(wf.window_id): wf
        for wf in baseline_fit.window_fits
        if wf.window_id is not None
    }
    all_ids = sorted(set(cur_by_id) | set(base_by_id))

    diffs = [
        _diff_window(
            wid,
            base_by_id.get(wid),
            cur_by_id.get(wid),
            res_element_mhz=res_element_mhz,
            shift_res=shift_res,
            chi2r_rel=chi2r_rel,
            eps_abs=eps_abs,
        )
        for wid in all_ids
    ]
    material = [d for d in diffs if d.material]
    # Worst-first: the larger fit-quality swing is the more important to eyeball.
    material.sort(
        key=lambda d: (
            abs(d.chi2r_cur - d.chi2r_base)
            if math.isfinite(d.chi2r_cur) and math.isfinite(d.chi2r_base)
            else 0.0
        ),
        reverse=True,
    )

    if not material:
        out_path.write_text(
            _shell(
                stem,
                "No material changes",
                '<div class="notice">The curated fit differs from the automatic fit '
                "only below the materiality thresholds (no peak added or removed, no "
                "peak moved appreciably, fit quality unchanged).</div>",
            )
        )
        return str(out_path)

    baseline_bundle = dataclasses.replace(bundle, fit=baseline_fit)
    sections: List[str] = []
    for d in material:
        before = (
            _panel_png_data_uri(baseline_bundle, d.window_id, dpi)
            if d.window_id in base_by_id
            else None
        )
        after = (
            _panel_png_data_uri(bundle, d.window_id, dpi)
            if d.window_id in cur_by_id
            else None
        )
        sections.append(_window_section(d, before, after))

    n_added = sum(d.n_added for d in material)
    n_removed = sum(d.n_removed for d in material)
    sub = (
        f"{len(material)} of {len(all_ids)} windows changed materially "
        f"&middot; +{n_added} / &minus;{n_removed} peaks"
    )
    out_path.write_text(_shell(stem, sub, "".join(sections)))
    return str(out_path)


def _shell(stem: str, subtitle: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(stem)} &mdash; curation diff</title>"
        f"<style>{_DIFF_CSS}</style></head><body>"
        f"<header><h1>{html.escape(stem)} &mdash; curation diff</h1>"
        f'<div class="sub">automatic fit &rarr; curated fit &middot; {subtitle}</div>'
        "</header><main>"
        f"{body}"
        "</main></body></html>"
    )
