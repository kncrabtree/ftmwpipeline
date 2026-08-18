"""
Stage 6 review commands.

Implements the ``review run``/``show``/``rank``/``edit``/``create``/``merge``/
``split``/``accept``/``apply``/``log``/``undo`` subcommands.  Thin wrappers over
:mod:`ftmwpipeline._internal.stage6_impl` -- identical behavior to
:class:`~ftmwpipeline.Pipeline` and the functional API.
"""

import argparse
from typing import Any, List, Optional, Sequence

import h5py

from .._internal.stage6_impl import (
    DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    DEFAULT_DISPLAY_BAR,
    RANK_METRICS,
    RefitWindowResult,
    ReviewRunResult,
    _normalize_metric,
    acknowledge_environment_impl,
    apply_curation_impl,
    create_window_impl,
    describe_planned_action,
    get_candidate_ledger_impl,
    get_final_products_impl,
    get_review_status_impl,
    merge_peaks_impl,
    rank_windows_impl,
    refit_window_impl,
    review_accept_impl,
    review_log_impl,
    review_preview_impl,
    review_run_impl,
    review_undo_impl,
    split_peak_impl,
)
from ..core.curation import REFIT_SNAP_TOL_MHZ, Frame
from ..core.data_structures import FittingResult, LedgerCandidate, Stage6Review
from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
from .utils import add_stage_object, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def _add_frame_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--frame``, identically, on every verb that takes a
    caller-supplied frequency (dest name matches the Python parameter name --
    the dual-interface invariant)."""
    parser.add_argument(
        "--frame",
        dest="frame",
        choices=("raw", "calibrated"),
        default=None,
        metavar="FRAME",
        help=(
            "Frame the frequency argument(s) are expressed in: 'raw' (the "
            "Stage 5 fit / ledger frame) or 'calibrated' (the corrected "
            "molecular frame the final-products table reports). Omitting it "
            "defaults to 'raw', matching today's behavior; on a "
            "self_calibrated file, omitting it while a frequency is given "
            "is refused rather than silently assumed."
        ),
    )


def _load_window_fits(file_path: str) -> List[FittingResult]:
    """Return per-window FittingResult objects from the persisted Stage 5 fit."""
    with h5py.File(file_path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found. Run 'fit run' first.")
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return list(spectrum_fit.window_fits)


def _fmt_mhz(v: float) -> str:
    return f"{v:.4f}"


def _combined_label(status: Optional["WindowReviewStatus"]) -> str:  # type: ignore[name-defined]
    """Build a short combined label from provenance + attention count."""
    if status is None:
        return "auto"
    prov = status.provenance
    n_attn = len(status.attention_reasons)
    if n_attn:
        return f"{prov}·needs-attention[{n_attn}]"
    return f"{prov}·—"


def cmd_review_show(args: argparse.Namespace) -> int:
    """Show the Stage 6 review surface.

    Without flags: per-window summary (fitted peak count, candidate count,
    provenance label).
    ``--attention``: windows needing attention, ranked by severity.
    ``--candidates``: tabulated candidate ledger for all (or a chosen) window.
    ``--window N``: detail for one window (fitted peaks, candidates, review
    status, decision log).
    ``--window N --output PATH``: render the window fit to a PNG file.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    bar: float = getattr(args, "bar", DEFAULT_DISPLAY_BAR)
    window_filter: Optional[int] = getattr(args, "window", None)
    show_candidates: bool = getattr(args, "candidates", False)
    show_attention: bool = getattr(args, "attention", False)
    output_path: Optional[str] = getattr(args, "output", None)
    output_dir: Optional[str] = getattr(args, "output_dir", None)

    try:
        window_fits = _load_window_fits(file_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    review: Stage6Review = get_review_status_impl(file_path)

    if window_filter is not None:
        wf_filtered = [wf for wf in window_fits if wf.window_id == window_filter]
        if not wf_filtered:
            print(f"Error: window_id={window_filter} not found in the Stage 5 fit.")
            return 1
        window_fits = wf_filtered

    # ---- batch-render windows to a directory (--output-dir DIR) --------------
    # One invocation resolves the detail bundle (FT, noise, padded grids) once
    # and reuses it across every window, so rendering the attention set is a
    # single file load instead of one per `--window N --output` call. The
    # window set follows the active filter: the attention queue under
    # ``--attention`` (severity-ranked, worst-first filenames), one window
    # under ``--window N``, otherwise every fitted window.
    if output_dir is not None:
        return _render_windows_to_dir(
            file_path, window_fits, review, output_dir, attention_only=show_attention
        )

    # ---- render window fit to file (--window N --output PATH) ----------------
    if window_filter is not None and output_path is not None:
        try:
            import matplotlib

            from .._internal.stage5_impl import render_fit_detail_impl

            matplotlib.use("Agg")
            fig = render_fit_detail_impl(file_path, window_filter)
            if fig is None:
                print(
                    f"Error: render_fit_detail_impl returned None for window {window_filter}."
                )
                return 1
            fig.savefig(output_path, dpi=130, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
            print(f"Saved window {window_filter} fit plot to {output_path}")
        except ImportError:
            print("Error: matplotlib is required for --output rendering.")
            return 1
        except (ValueError, KeyError) as exc:
            print(f"Error rendering window {window_filter}: {exc}")
            return 1
        return 0

    # ---- attention-only table (--attention) ----------------------------------
    if show_attention:
        attention_rows = []
        for wf in sorted(window_fits, key=lambda w: w.window_id or 0):
            wid = wf.window_id if wf.window_id is not None else -1
            status = review.window_statuses.get(wid)
            if status is None or not status.needs_attention:
                continue
            top_reason = max(status.attention_reasons, key=lambda r: r.severity)
            attention_rows.append((top_reason.severity, wid, status, top_reason))

        attention_rows.sort(key=lambda t: t[0], reverse=True)
        if not attention_rows:
            print("No windows flagged for attention.")
            return 0

        print(f"{'win':>5}  {'label':<30}  {'top reason':<18}  detail")
        print("-" * 90)
        for _, wid, status, top_reason in attention_rows:
            label = _combined_label(status)
            print(
                f"{wid:>5}  {label:<30}  {top_reason.kind:<18}  {top_reason.detail[:60]}"
            )
        return 0

    # ---- per-window summary (default) ----------------------------------------
    if not show_candidates and window_filter is None:
        print(
            f"{'win':>5}  {'freq_lo':>12}  {'freq_hi':>12}  "
            f"{'peaks':>6}  {'candidates':>10}  {'label'}"
        )
        print("-" * 74)
        for wf in sorted(window_fits, key=lambda w: w.window_id or 0):
            wid = wf.window_id if wf.window_id is not None else -1
            cands = get_candidate_ledger_impl(file_path, window_id=wid, bar=bar)
            flo = fhi = float("nan")
            if wf.window is not None and wf.window.freq_range is not None:
                flo, fhi = wf.window.freq_range
            status = review.window_statuses.get(wid)
            label = _combined_label(status)
            print(
                f"{wid:>5}  {_fmt_mhz(flo):>12}  {_fmt_mhz(fhi):>12}  "
                f"{wf.n_peaks_fitted:>6}  {len(cands):>10}  {label}"
            )
        return 0

    # ---- per-window detail (--window N) without --candidates -----------------
    if window_filter is not None and not show_candidates:
        wf = window_fits[0]
        wid = wf.window_id if wf.window_id is not None else -1
        flo = fhi = float("nan")
        if wf.window is not None and wf.window.freq_range is not None:
            flo, fhi = wf.window.freq_range
        print(
            f"Window {wid}  [{_fmt_mhz(flo)}, {_fmt_mhz(fhi)}] MHz  "
            f"chi2r={wf.reduced_chi2:.3f}"
        )

        # Review status block
        status = review.window_statuses.get(wid)
        print()
        print(f"  Review status: {_combined_label(status)}")
        if status is not None and status.attention_reasons:
            for reason in status.attention_reasons:
                print(f"    [{reason.kind}] {reason.detail}")

        # Decision log entries for this window
        log_entries = [e for e in review.decision_log if e.window_id == wid]
        if log_entries:
            print(f"  Decision log ({len(log_entries)} entries):")
            for entry in log_entries:
                ev_str = (
                    ", ".join(
                        f"{k}={v:.4g}"
                        for k, v in entry.evidence.items()
                        if isinstance(v, float)
                    )
                    if entry.evidence
                    else ""
                )
                print(
                    f"    #{entry.order_index}  kind={entry.kind}  "
                    f"freq={_fmt_mhz(entry.frequency_mhz)}  {ev_str}"
                )

        print()
        print(f"  Fitted peaks ({wf.n_peaks_fitted}):")
        if wf.fitted_peaks:
            print(
                f"  {'peak_id':>8}  {'freq (MHz)':>14}  {'amp':>10}  "
                f"{'snr':>8}  {'origin':>6}"
            )
            print("  " + "-" * 54)
            for p in sorted(wf.fitted_peaks, key=lambda pk: pk.frequency_mhz):
                snr_str = f"{p.snr:.1f}" if p.snr is not None else "  n/a"
                print(
                    f"  {str(p.peak_id):>8}  {_fmt_mhz(p.frequency_mhz):>14}  "
                    f"{p.amplitude:>10.3e}  {snr_str:>8}  {p.origin:>6}"
                )
        else:
            print("  (none)")

        # Doublet alternatives: the AICc-preferred-doublet observation is no
        # longer an attention trigger (it confirms a real doublet, not an
        # actionable overfit), but the per-pair adjudication stays visible here
        # for the reviewer's drill-down.
        doublet_alts = [
            da for da in getattr(wf, "doublet_alternatives", []) if da.merged_success
        ]
        if doublet_alts:
            print()
            print(f"  Doublet alternatives ({len(doublet_alts)}):")
            print(
                f"  {'pair (MHz)':>27}  {'sep (res)':>9}  "
                f"{'amp_ratio':>9}  {'dAICc':>9}  verdict"
            )
            print("  " + "-" * 70)
            for da in doublet_alts:
                verdict = "doublet" if da.delta_aicc > 0 else "merge"
                print(
                    f"  {_fmt_mhz(da.frequency_a_mhz):>12}/"
                    f"{_fmt_mhz(da.frequency_b_mhz):<12}  "
                    f"{da.separation_res_elements:>9.2f}  "
                    f"{da.amp_ratio:>9.2f}  {da.delta_aicc:>9.3g}  {verdict}"
                )

        print()
        cands = get_candidate_ledger_impl(file_path, window_id=wid, bar=bar)
        print(f"  Candidates ({len(cands)}, bar={bar:.1f}):")
        if cands:
            _print_candidate_table(cands, indent=2)
        else:
            print("  (none above bar)")
        return 0

    # ---- candidate table (--candidates [--window N]) -------------------------
    try:
        cands = get_candidate_ledger_impl(file_path, window_id=window_filter, bar=bar)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    scope = f"window {window_filter}" if window_filter is not None else "all windows"
    print(f"Candidate ledger ({scope}, bar={bar:.1f}): {len(cands)} candidate(s)")
    if cands:
        print()
        _print_candidate_table(cands, indent=0)
    return 0


def _render_windows_to_dir(
    file_path: str,
    window_fits: List[FittingResult],
    review: Stage6Review,
    output_dir: str,
    *,
    attention_only: bool,
) -> int:
    """Render a set of window detail figures into ``output_dir`` (one bundle).

    Resolves the detail bundle once and reuses it for every window. With
    ``attention_only`` the set is the flagged windows, ordered worst-severity
    first and named ``<rank>_w<id>.png`` so a file browser sorts them in
    review order; otherwise every window in ``window_fits`` is rendered as
    ``w<id>.png``.
    """
    import os

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .._internal.stage5_impl import (
            _resolve_detail_bundle,
            render_fit_detail_impl,
        )
    except ImportError:
        print("Error: matplotlib is required for --output-dir rendering.")
        return 1

    # Build the (rank, window_id, filename) work list.
    targets: List[tuple] = []
    if attention_only:
        rows = []
        for wf in window_fits:
            wid = wf.window_id if wf.window_id is not None else -1
            status = review.window_statuses.get(wid)
            if status is None or not status.needs_attention:
                continue
            severity = max(r.severity for r in status.attention_reasons)
            rows.append((severity, wid))
        rows.sort(key=lambda t: (-t[0], t[1]))
        if not rows:
            print("No windows flagged for attention; nothing to render.")
            return 0
        width = len(str(len(rows)))
        for rank, (_, wid) in enumerate(rows, start=1):
            targets.append((wid, f"{rank:0{width}d}_w{wid}.png"))
    else:
        for wf in sorted(window_fits, key=lambda w: w.window_id or 0):
            wid = wf.window_id if wf.window_id is not None else -1
            targets.append((wid, f"w{wid}.png"))

    os.makedirs(output_dir, exist_ok=True)
    bundle = _resolve_detail_bundle(file_path)
    print(f"Rendering {len(targets)} window(s) to {output_dir} ...")
    n_ok = 0
    for wid, fname in targets:
        out = os.path.join(output_dir, fname)
        try:
            fig = render_fit_detail_impl(file_path, wid, bundle=bundle)
        except (ValueError, KeyError) as exc:
            print(f"  window {wid}: skipped ({exc})")
            continue
        if fig is None:
            print(f"  window {wid}: skipped (no figure)")
            continue
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        n_ok += 1
    print(f"Rendered {n_ok}/{len(targets)} window(s) to {output_dir}")
    return 0


def _print_candidate_table(cands: List[LedgerCandidate], indent: int = 0) -> None:
    pad = " " * indent
    print(
        f"{pad}{'win':>5}  {'freq (MHz)':>14}  {'evidence':>10}  "
        f"{'kind':>13}  {'sites':<30}  {'reasons'}"
    )
    print(pad + "-" * 90)
    for c in cands:
        sites_str = ",".join(c.decision_sites)[:28]
        reasons_str = "; ".join(c.reasons)[:40]
        print(
            f"{pad}{c.window_id:>5}  {_fmt_mhz(c.frequency_mhz):>14}  "
            f"{c.best_evidence:>10.3f}  {c.evidence_kind:>13}  "
            f"{sites_str:<30}  {reasons_str}"
        )


def cmd_review_edit(args: argparse.Namespace) -> int:
    """Re-fit one window with user-directed add/remove edits.

    Prints a before/after summary: peak counts, χ²ᵣ, which peaks were added
    or removed, and the origin of each resulting peak.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    window_id: int = args.window
    add_freqs: List[float] = list(args.add or [])
    remove_freqs: List[float] = list(args.remove or [])

    if not add_freqs and not remove_freqs:
        print(
            "Warning: no --add or --remove frequencies given; "
            "performing identity refit (no-op edit)."
        )

    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result: RefitWindowResult = refit_window_impl(
            file_path,
            window_id,
            add=add_freqs,
            remove=remove_freqs,
            snap_tol_mhz=getattr(args, "snap_tol_mhz", REFIT_SNAP_TOL_MHZ),
            frame=frame,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"review edit  window={result.window_id}  "
        f"peaks {result.n_peaks_before} → {result.n_peaks_after}  "
        f"chi2r {result.chi2r_before:.4g} → {result.chi2r_after:.4g}"
    )
    if add_freqs:
        print(f"  Added seeds ({len(add_freqs)}): {[f'{f:.4f}' for f in add_freqs]}")
    if remove_freqs:
        print(f"  Removed ({len(remove_freqs)}): {[f'{f:.4f}' for f in remove_freqs]}")
    if result.fitted_peaks:
        print(f"  {'freq (MHz)':>14}  {'amp':>10}  {'snr':>8}  {'origin':>6}")
        print("  " + "-" * 46)
        for p in sorted(result.fitted_peaks, key=lambda pk: pk.frequency_mhz):
            snr_str = f"{p.snr:.1f}" if p.snr is not None else "  n/a"
            print(
                f"  {_fmt_mhz(p.frequency_mhz):>14}  "
                f"{p.amplitude:>10.3e}  {snr_str:>8}  {p.origin:>6}"
            )
    return 0


def cmd_review_acknowledge_environment(args: argparse.Namespace) -> int:
    """Record acceptance of an analysis-epoch mismatch for Stage 6 editing."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    reason: str = getattr(args, "reason", "") or ""

    try:
        info = acknowledge_environment_impl(file_path, reason=reason)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    current = info.acknowledged_environment
    fit_env = info.fit_environment
    print("review acknowledge-environment")
    print(f"  Acknowledged: {current.summary()}")
    if fit_env is not None:
        print(f"  Stage 5 fit:  {fit_env.summary()}")
    if not info.mismatch:
        print(
            "  Note: no analysis-epoch mismatch was present; the edit verbs "
            "were not blocked. The acknowledgement is recorded anyway."
        )
    else:
        print(
            "  Stage 6 edits may now proceed. The curated fit will mix two "
            "analysis environments, and the reports will say so."
        )
    if reason:
        print(f"  Reason: {reason}")
    return 0


def cmd_review_create(args: argparse.Namespace) -> int:
    """Install a fit window covering a frequency no window covers.

    Prints the installed window's id, whether it was created or an adjacent
    window was widened, its extent, and the next command to run.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    anchor: float = args.anchor
    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = create_window_impl(
            file_path,
            anchor,
            snap_tol_mhz=getattr(args, "snap_tol_mhz", REFIT_SNAP_TOL_MHZ),
            frame=frame,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    lo, hi = result.freq_range
    print(
        f"review create  window={result.window_id}  {result.mode}  "
        f"anchor {result.anchor_mhz:.4f} MHz"
    )
    print(
        f"  Range: [{_fmt_mhz(lo)}, {_fmt_mhz(hi)}] MHz  "
        f"({result.n_points} points, {result.n_contributors} frozen contributor(s))"
    )
    if result.depends_on:
        print(f"  Reads leakage from window(s): {result.depends_on}")
    if result.mode == "widened":
        print(
            "  The gap was too narrow for a fittable window, so this existing "
            "window was widened to cover the anchor."
        )
    print(f"  Fitted peaks in the window: {result.n_peaks}")
    print(
        f"  Next: ftmwpipeline review edit {args.file_path} "
        f"--window {result.window_id} --add {anchor:.4f}"
    )
    return 0


def cmd_review_merge(args: argparse.Namespace) -> int:
    """Collapse ≥2 fitted peaks in a window into one.

    Prints a before/after summary: peak counts, χ²ᵣ, and the merged product.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    window_id: int = args.window
    peak_freqs: List[float] = list(args.peaks or [])

    if len(peak_freqs) < 2:
        print(
            f"Error: --peaks requires at least 2 frequencies; "
            f"got {len(peak_freqs)}."
        )
        return 1

    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = merge_peaks_impl(
            file_path,
            window_id,
            peak_freqs,
            snap_tol_mhz=getattr(args, "snap_tol_mhz", REFIT_SNAP_TOL_MHZ),
            frame=frame,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"review merge  window={result.window_id}  "
        f"peaks {result.n_peaks_before} → {result.n_peaks_after}  "
        f"chi2r {result.chi2r_before:.4g} → {result.chi2r_after:.4g}"
    )
    print(f"  Merged from ({len(peak_freqs)}): {[f'{f:.4f}' for f in peak_freqs]}")
    user_peaks = [p for p in result.fitted_peaks if p.origin == "user"]
    if user_peaks:
        print("  Merged product(s):")
        for p in sorted(user_peaks, key=lambda pk: pk.frequency_mhz):
            snr_str = f"{p.snr:.1f}" if p.snr is not None else "  n/a"
            print(
                f"    {_fmt_mhz(p.frequency_mhz):>14} MHz  "
                f"{p.amplitude:>10.3e}  snr={snr_str}  origin={p.origin}"
            )
    return 0


def cmd_review_split(args: argparse.Namespace) -> int:
    """Replace one fitted peak with K peaks spread across one resolution element.

    Prints a before/after summary: peak counts, χ²ᵣ, and the split products.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    window_id: int = args.window
    peak_freq: float = args.peak
    into: int = args.into

    if into < 2:
        print(f"Error: --into must be >= 2; got {into}.")
        return 1

    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = split_peak_impl(
            file_path,
            window_id,
            peak_freq,
            into=into,
            snap_tol_mhz=getattr(args, "snap_tol_mhz", REFIT_SNAP_TOL_MHZ),
            frame=frame,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"review split  window={result.window_id}  "
        f"peaks {result.n_peaks_before} → {result.n_peaks_after}  "
        f"chi2r {result.chi2r_before:.4g} → {result.chi2r_after:.4g}"
    )
    print(f"  Split {peak_freq:.4f} MHz into {into}")
    user_peaks = [p for p in result.fitted_peaks if p.origin == "user"]
    if user_peaks:
        print("  Split product(s):")
        for p in sorted(user_peaks, key=lambda pk: pk.frequency_mhz):
            snr_str = f"{p.snr:.1f}" if p.snr is not None else "  n/a"
            print(
                f"    {_fmt_mhz(p.frequency_mhz):>14} MHz  "
                f"{p.amplitude:>10.3e}  snr={snr_str}  origin={p.origin}"
            )
    return 0


def cmd_review_accept(args: argparse.Namespace) -> int:
    """Accept a window as-is or accept a specific revived candidate.

    Without ``--candidate``: marks the window ``provenance=reviewed`` and
    records an ``"accept"`` decision log entry.  The fit is unchanged.

    With ``--candidate F``: adds the candidate peak at ``F`` MHz via a
    single-window refit, marking the window ``provenance=user-edited``.
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    window_id: int = args.window
    candidate_freq: Optional[float] = getattr(args, "candidate", None)
    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = review_accept_impl(
            file_path,
            window_id,
            candidate_freq=candidate_freq,
            snap_tol_mhz=getattr(args, "snap_tol_mhz", REFIT_SNAP_TOL_MHZ),
            frame=frame,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    if result is None:
        print(f"review accept  window={window_id}  provenance→reviewed")
    else:
        print(
            f"review accept  window={result.window_id}  "
            f"candidate accepted: peaks {result.n_peaks_before} → {result.n_peaks_after}  "
            f"chi2r {result.chi2r_before:.4g} → {result.chi2r_after:.4g}"
        )
    return 0


def cmd_review_run(args: argparse.Namespace) -> int:
    """Build or refresh the Stage 6 attention-routing curation layer."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    bar: float = getattr(args, "bar", DEFAULT_DISPLAY_BAR)
    attention_bar: float = getattr(
        args, "attention_bar", DEFAULT_ATTENTION_CANDIDATE_EVIDENCE
    )
    sigma_floor_khz: Optional[float] = getattr(args, "sigma_floor_khz", None)

    try:
        result = review_run_impl(
            file_path,
            bar=bar,
            attention_candidate_evidence=attention_bar,
            sigma_floor_khz=sigma_floor_khz,
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"review run: {result.n_windows} window(s), "
        f"{result.n_attention} needing attention"
    )
    if result.reason_counts:
        for kind, count in sorted(result.reason_counts.items()):
            print(f"  {kind}: {count}")

    fp = get_final_products_impl(file_path)
    if fp is not None:
        print(
            f"final products: {len(fp.peaks)} peak(s), "
            f"calibration {fp.calibration_state}"
            + (f" (eps={fp.epsilon * 1e6:+.3f} ppm)" if fp.epsilon else "")
            + f", sigma_floor={fp.sigma_floor_khz:.3f} kHz"
        )
    return 0


def cmd_review_apply(args: argparse.Namespace) -> int:
    """Apply a curation file of batched review edits (with optional dry-run)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    dry_run: bool = getattr(args, "dry_run", False)
    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = apply_curation_impl(
            file_path, args.curation_file, dry_run=dry_run, frame=frame
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    header = (
        "review apply (dry run): resolved plan" if dry_run else "review apply: plan"
    )
    print(header)
    if not result.plan:
        print("  (no actions)")
    for i, action in enumerate(result.plan, start=1):
        print(f"  {i:>3}. {describe_planned_action(action)}")
    if result.warnings:
        print("warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    if dry_run:
        print(f"{len(result.plan)} action(s) would be applied (nothing written).")
    else:
        print(f"applied {result.applied} action(s).")
    return 0


def cmd_review_preview(args: argparse.Namespace) -> int:
    """Preview a curation file's fitted outcome, run to completion in memory
    -- nothing is written."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    frame: Optional[Frame] = getattr(args, "frame", None)

    try:
        result = review_preview_impl(file_path, args.curation_file, frame=frame)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    print("review preview (nothing written):")
    if not result.windows:
        print("  (no fit-mutating actions; nothing to preview)")
        return 0
    for wid in sorted(result.windows):
        w = result.windows[wid]
        actions = ",".join(str(i + 1) for i in w.action_indices) or "-"
        print(
            f"  window {wid:>4}  [{w.origin:>8}]  actions={actions:<8}  "
            f"peaks {w.n_peaks_before}->{w.n_peaks_after}  "
            f"chi2r {w.chi2r_before:.3f}->{w.chi2r_after:.3f}"
        )
    return 0


def cmd_review_log(args: argparse.Namespace) -> int:
    """List the persisted Stage 6 decision log (read-only)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)

    try:
        entries = review_log_impl(file_path)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    print("review log (user decisions, execution order):")
    if not entries:
        print("  (no recorded decisions)")
        return 0
    # Size the action column to the widest kind present ("create_window" is
    # longer than the others), so the table stays aligned as the vocabulary grows.
    kw = max(7, max(len(e.kind) for e in entries))
    print(f"  {'id':>4}  {'action':>{kw}}  {'window':>6}  {'freq (MHz)':>12}")
    print("  " + "-" * (kw + 29))
    for e in entries:
        print(
            f"  {e.order_index:>4}  {e.kind:>{kw}}  {e.window_id:>6}  "
            f"{e.frequency_mhz:>12.4f}"
        )
    return 0


def cmd_review_undo(args: argparse.Namespace) -> int:
    """Undo one or more recorded decisions by id, replaying the rest."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    ids: List[int] = getattr(args, "ids", None) or []
    dry_run: bool = getattr(args, "dry_run", False)
    if not ids:
        print("Error: provide at least one --id to undo (see 'review log').")
        return 1

    try:
        result = review_undo_impl(file_path, ids, dry_run=dry_run)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    print("review undo (dry run)" if dry_run else "review undo")
    print("removing:")
    for e in result.removed:
        print(
            f"  id {e.order_index}: {e.kind} window {e.window_id} "
            f"@ {e.frequency_mhz:.4f} MHz"
        )
    print("replay of surviving decisions:")
    if not result.plan:
        print("  (none -- fully reverted to the automatic fit)")
    for i, action in enumerate(result.plan, start=1):
        print(f"  {i:>3}. {describe_planned_action(action)}")
    if dry_run:
        print(f"{len(result.removed)} decision(s) would be undone (nothing written).")
    else:
        print(
            f"undid {len(result.removed)} decision(s); "
            f"replayed {result.applied} action(s)."
        )
    return 0


def cmd_review_rank(args: argparse.Namespace) -> int:
    """Rank windows by a persisted per-window statistic (read-only)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    by: str = args.by
    top: Optional[int] = getattr(args, "top", None)

    try:
        ranked = rank_windows_impl(file_path, by=by, top=top)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    desc = RANK_METRICS[_normalize_metric(by)][0]
    review = get_review_status_impl(file_path)
    print(f"review rank by {_normalize_metric(by)} ({desc}), worst first:")
    if not ranked:
        print("  (no windows with this metric defined)")
        return 0
    print(
        f"  {'win':>5}  {'value':>12}  {'freq_lo':>12}  {'freq_hi':>12}  "
        f"{'peaks':>6}  {'chi2r':>8}  label"
    )
    print("  " + "-" * 78)
    for rw in ranked:
        status = review.window_statuses.get(rw.window_id)
        print(
            f"  {rw.window_id:>5}  {rw.value:>12.4g}  {_fmt_mhz(rw.freq_lo):>12}  "
            f"{_fmt_mhz(rw.freq_hi):>12}  {rw.n_peaks:>6}  "
            f"{rw.reduced_chi2:>8.2f}  {_combined_label(status)}"
        )
    return 0


def register_review_commands(subparsers: Any) -> None:
    """Register review (Stage 6) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "review",
        synonym="stage6",
        help="Stage 6: review fitted model and candidate ledger",
        description=(
            "Stage 6 review surface.\n\n"
            "Inspect the automatic fit, view per-window summaries, explore\n"
            "the candidate ledger, and apply user-directed edits (add, remove,\n"
            "merge, split peaks) -- one at a time or batched from a curation file,\n"
            "and undo them by id.  Create a window for a line no window covers.\n\n"
            "Verbs: run, show, rank, edit, create, merge, split, accept, apply,\n"
            "preview, log, undo"
        ),
    )

    p_run = verbs.add_parser(
        "run",
        help="Build or refresh the Stage 6 attention-routing layer",
        description=(
            "Compute per-window advisory attention reasons and persist the\n"
            "Stage 6 review state to the .ftmw file.\n\n"
            "Existing provenance (reviewed/user-edited) and the decision log\n"
            "are preserved; only attention reasons are refreshed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_run.add_argument(
        "--bar",
        dest="bar",
        type=float,
        default=DEFAULT_DISPLAY_BAR,
        metavar="BAR",
        help=(
            f"Display bar for candidate-bearing detection "
            f"(default {DEFAULT_DISPLAY_BAR:.1f})."
        ),
    )
    p_run.add_argument(
        "--attention-bar",
        dest="attention_bar",
        type=float,
        default=DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
        metavar="EV",
        help=(
            f"Evidence threshold for flagging a candidate-bearing window "
            f"(stiffer than --bar; default "
            f"{DEFAULT_ATTENTION_CANDIDATE_EVIDENCE:.1f})."
        ),
    )
    p_run.add_argument(
        "--sigma-floor",
        dest="sigma_floor_khz",
        type=float,
        default=None,
        metavar="KHZ",
        help=(
            "Declare the systematic frequency-accuracy floor (kHz), persisted "
            "in-file and folded into the sigma_f budget. Omit to keep the "
            "persisted value (default 0)."
        ),
    )
    p_run.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_run.set_defaults(func=cmd_review_run)

    metric_lines = "\n".join(
        f"  {name:<20} {desc}" for name, (desc, _) in sorted(RANK_METRICS.items())
    )
    p_rank = verbs.add_parser(
        "rank",
        help="Rank all windows by a per-window statistic (read-only)",
        description=(
            "Rank every fit window worst-first by a chosen persisted statistic\n"
            "-- on-demand exploration decoupled from the attention flags. Read\n"
            "only; nothing is written.\n\nMetrics:\n" + metric_lines
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rank.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_rank.add_argument(
        "--by",
        dest="by",
        required=True,
        metavar="METRIC",
        help="Ranking metric (e.g. min-snr, max-vif, chi2r; see description).",
    )
    p_rank.add_argument(
        "--top",
        dest="top",
        type=int,
        default=20,
        metavar="N",
        help="Show at most N windows (default 20; use 0 for all).",
    )
    p_rank.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_rank.set_defaults(func=cmd_review_rank)

    # ---- review apply --------------------------------------------------------
    p_apply = verbs.add_parser(
        "apply",
        help="Apply a curation file of batched review edits",
        description=(
            "Replay a curation file (CSV) of batched edits through the same\n"
            "impls the interactive verbs use.\n\n"
            "Columns: action,window,freqs,params -- where action is one of\n"
            "add/remove/merge/split/accept/create, freqs is a ';'-separated list\n"
            "of molecular MHz, and params is ';'-separated key=value (into=K for\n"
            "split, candidate=F for accept). Blank lines and '#' comments are\n"
            "ignored; an optional header row is skipped.\n\n"
            "A 'create' row takes the anchor frequency in freqs and either a\n"
            "window id or 'new' in the window column; 'new' means 'whichever id\n"
            "this produces', while a named id pins the id the created window\n"
            "takes -- which is how a file generated from the decision log keeps\n"
            "each created window's identity stable across a replay.\n\n"
            "A run of add/remove rows on one window coalesces into a single\n"
            "refit; merge/split/accept/create stand alone. With --dry-run the\n"
            "resolved plan and any frequency-resolution warnings print without\n"
            "writing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_apply.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_apply.add_argument("curation_file", help="Path to the curation CSV to apply.")
    p_apply.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print the resolved plan and warnings without modifying the file.",
    )
    _add_frame_argument(p_apply)
    p_apply.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_apply.set_defaults(func=cmd_review_apply)

    # ---- review preview --------------------------------------------------------
    p_preview = verbs.add_parser(
        "preview",
        help="Preview a curation file's fitted outcome (writes nothing)",
        description=(
            "Run a curation file's resolved plan to completion in memory --\n"
            "the same parse, frame conversion, appliers, and one combined\n"
            "cascade 'apply' uses -- and report the fitted outcome without\n"
            "writing anything to the file.\n\n"
            "The result is keyed by window id, read after the cascade (not\n"
            "per-action, since a cascade can supersede an action's own\n"
            "numbers), and reports final-product numbers: calibrated\n"
            "frequency and the three-term sigma budget, identical to what a\n"
            "subsequent 'apply' of the same plan would persist.\n\n"
            "Epoch-gated exactly like 'apply', except a plan of entirely bare\n"
            "accept rows, which touches no fit and so is not gated either."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_preview.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_preview.add_argument("curation_file", help="Path to the curation CSV to preview.")
    _add_frame_argument(p_preview)
    p_preview.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_preview.set_defaults(func=cmd_review_preview)

    # ---- review log ----------------------------------------------------------
    p_log = verbs.add_parser(
        "log",
        help="List the persisted decision log (read-only)",
        description=(
            "List the Stage 6 decision log -- every recorded user edit in\n"
            "execution order, keyed by id (order_index). Read only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_log.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_log.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_log.set_defaults(func=cmd_review_log)

    # ---- review undo ---------------------------------------------------------
    p_undo = verbs.add_parser(
        "undo",
        help="Undo recorded decisions by id, replaying the rest",
        description=(
            "Undo one or more decisions (by the id from 'review log').\n\n"
            "Rollback is replay-from-baseline: the automatic Stage 5 fit is\n"
            "restored and every surviving decision is re-applied, so decision\n"
            "ids are renumbered afterward. Use --dry-run to preview. Requires\n"
            "the automatic-fit baseline (unavailable if the fit was re-run after\n"
            "editing -- rebuild and re-edit in that case)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_undo.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_undo.add_argument(
        "--id",
        dest="ids",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="Decision id to undo; repeat for several: --id 2 --id 4.",
    )
    p_undo.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print what would be undone and the replay plan without writing.",
    )
    p_undo.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_undo.set_defaults(func=cmd_review_undo)

    p_show = verbs.add_parser(
        "show",
        help="Show per-window summary or candidate ledger",
        description=(
            "Show the Stage 6 review surface.\n\n"
            "Without flags: per-window table (fitted peak count, candidate count).\n"
            "--candidates: tabulated candidate ledger.\n"
            "--window N: detail for one window (fitted peaks + candidates).\n"
            "--window N --candidates: candidates for that window only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_show.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_show.add_argument(
        "--window",
        dest="window",
        type=int,
        default=None,
        metavar="N",
        help="Restrict output to this window_id.",
    )
    p_show.add_argument(
        "--candidates",
        dest="candidates",
        action="store_true",
        default=False,
        help="Show the candidate ledger (revivable lines) instead of the summary.",
    )
    p_show.add_argument(
        "--bar",
        dest="bar",
        type=float,
        default=DEFAULT_DISPLAY_BAR,
        metavar="BAR",
        help=(
            f"Display bar: candidates below this evidence / SNR threshold are "
            f"hidden (default {DEFAULT_DISPLAY_BAR:.1f})."
        ),
    )
    p_show.add_argument(
        "--attention",
        dest="attention",
        action="store_true",
        default=False,
        help=(
            "Show only windows flagged for attention, ranked by severity. "
            "Requires 'review run' to have been called."
        ),
    )
    p_show.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Render the window fit to this file (PNG/PDF). "
            "Requires --window N and Stage 5 completed."
        ),
    )
    p_show.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Batch-render windows to this directory (one detail PNG each), "
            "resolving the FT/noise bundle once. With --attention renders the "
            "flagged set worst-first; with --window N renders that one; "
            "otherwise renders every window."
        ),
    )
    p_show.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_show.set_defaults(func=cmd_review_show)

    # ---- review accept -------------------------------------------------------
    p_accept = verbs.add_parser(
        "accept",
        help="Accept a window as-is or accept a revived candidate",
        description=(
            "Accept a window in the Stage 6 review.\n\n"
            "Without --candidate: marks the window as reviewed (looked, no change).\n"
            "With --candidate F: adds peak at F MHz, marks window as user-edited."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_accept.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_accept.add_argument(
        "--window",
        dest="window",
        type=int,
        required=True,
        metavar="N",
        help="window_id to accept.",
    )
    p_accept.add_argument(
        "--candidate",
        dest="candidate",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Accept by adding a candidate peak at this molecular MHz. "
            "Delegates to 'review edit --add F'."
        ),
    )
    p_accept.add_argument(
        "--snap-tol-mhz",
        dest="snap_tol_mhz",
        type=float,
        default=REFIT_SNAP_TOL_MHZ,
        metavar="MHZ",
        help=(
            "Tolerance for snapping a requested frequency to an existing peak "
            f"or ledger candidate (default {REFIT_SNAP_TOL_MHZ} MHz = "
            f"{REFIT_SNAP_TOL_MHZ * 1e3:.0f} kHz)."
        ),
    )
    _add_frame_argument(p_accept)
    p_accept.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_accept.set_defaults(func=cmd_review_accept)

    # ---- review edit ---------------------------------------------------------
    p_edit = verbs.add_parser(
        "edit",
        help="Re-fit one window with user add/remove edits",
        description=(
            "User-directed single-window refit.\n\n"
            "Removes named peaks and/or adds new ones, then re-converges the\n"
            "window's NLS.  Added peaks carry origin=user and are immune to\n"
            "automatic cleanup/rescue pruning.  Removed peaks are not re-added.\n"
            "Other windows are untouched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_edit.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_edit.add_argument(
        "--window",
        dest="window",
        type=int,
        required=True,
        metavar="N",
        help="window_id to refit.",
    )
    p_edit.add_argument(
        "--add",
        dest="add",
        type=float,
        action="append",
        default=None,
        metavar="F",
        help=(
            "Molecular MHz frequency of a peak to add. "
            "Repeat the flag for several: --add F1 --add F2 ..."
        ),
    )
    p_edit.add_argument(
        "--remove",
        dest="remove",
        type=float,
        action="append",
        default=None,
        metavar="F",
        help=(
            "Molecular MHz frequency of a fitted peak to remove. "
            "Repeat the flag for several: --remove F1 --remove F2 ..."
        ),
    )
    p_edit.add_argument(
        "--snap-tol-mhz",
        dest="snap_tol_mhz",
        type=float,
        default=REFIT_SNAP_TOL_MHZ,
        metavar="MHZ",
        help=(
            "Tolerance for snapping a requested frequency to an existing peak "
            f"or ledger candidate (default {REFIT_SNAP_TOL_MHZ} MHz = "
            f"{REFIT_SNAP_TOL_MHZ * 1e3:.0f} kHz)."
        ),
    )
    _add_frame_argument(p_edit)
    p_edit.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_edit.set_defaults(func=cmd_review_edit)

    # ---- review create -------------------------------------------------------
    p_create = verbs.add_parser(
        "create",
        help="Install a fit window for a line no window covers",
        description=(
            "Create a fit window covering a frequency the automatic pass left\n"
            "uncovered -- the line the detector missed.\n\n"
            "Purely structural and purely additive: no existing window is\n"
            "renumbered, re-fit, or thawed, and the whole curated edit set\n"
            "stands (re-running detection to reach the line would drop Stages\n"
            "5 and 6 and destroy it).\n\n"
            "This installs the window only.  Put the line in it with a separate\n"
            "'review edit --window N --add F', so the decision log records the\n"
            "two operations distinctly.\n\n"
            "If the gap is too narrow to hold a fittable window, the adjacent\n"
            "window is widened instead; the output says which happened."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_create.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_create.add_argument(
        "--at",
        dest="anchor",
        type=float,
        required=True,
        metavar="F",
        help="Molecular MHz frequency the new window must cover.",
    )
    p_create.add_argument(
        "--snap-tol-mhz",
        dest="snap_tol_mhz",
        type=float,
        default=REFIT_SNAP_TOL_MHZ,
        metavar="MHZ",
        help=(
            "Tolerance for snapping a requested frequency to an existing peak "
            f"or ledger candidate (default {REFIT_SNAP_TOL_MHZ} MHz = "
            f"{REFIT_SNAP_TOL_MHZ * 1e3:.0f} kHz)."
        ),
    )
    _add_frame_argument(p_create)
    p_create.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_create.set_defaults(func=cmd_review_create)

    # ---- review acknowledge-environment --------------------------------------
    p_ack = verbs.add_parser(
        "acknowledge-environment",
        help="Accept an analysis-epoch mismatch so Stage 6 editing can proceed",
        description=(
            "Record acceptance of an analysis-environment mismatch.\n\n"
            "A Stage 6 edit re-fits one window and splices it into a fit whose\n"
            "other windows came from an earlier run. When that fit was produced\n"
            "under a different analysis epoch, the edit verbs refuse, because\n"
            "the result would hold two different fitting models.\n\n"
            "The cleaner fix is usually to re-run 'fit run' so the whole fit\n"
            "comes from one environment. Use this when you accept the mixture.\n\n"
            "The acknowledgement is stored in the .ftmw, not passed per command,\n"
            "so the curated result carries the fact in its own record and the\n"
            "reports say the curation crossed an epoch boundary. It names the\n"
            "environment it was given under, so a later upgrade asks again."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ack.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_ack.add_argument(
        "--reason",
        dest="reason",
        default="",
        metavar="TEXT",
        help="Optional note stored alongside the acknowledgement.",
    )
    p_ack.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_ack.set_defaults(func=cmd_review_acknowledge_environment)

    # ---- review merge --------------------------------------------------------
    p_merge = verbs.add_parser(
        "merge",
        help="Collapse ≥2 fitted peaks in a window into one",
        description=(
            "Collapse a set of fitted peaks into a single peak.\n\n"
            "The replacement is seeded at the SNR-weighted centroid of the\n"
            "removed peaks.  When the pair matches a persisted doublet-alternative\n"
            "record with a successful merged refit, the recorded merged seed is\n"
            "used instead.  All products carry origin=user."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_merge.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_merge.add_argument(
        "--window",
        dest="window",
        type=int,
        required=True,
        metavar="N",
        help="window_id containing the peaks to merge.",
    )
    p_merge.add_argument(
        "--peaks",
        dest="peaks",
        type=float,
        action="append",
        default=None,
        required=True,
        metavar="F",
        help=(
            "Molecular MHz frequency of a peak to collapse; repeat the flag "
            "for each (≥2): --peaks F1 --peaks F2 ..."
        ),
    )
    p_merge.add_argument(
        "--snap-tol-mhz",
        dest="snap_tol_mhz",
        type=float,
        default=REFIT_SNAP_TOL_MHZ,
        metavar="MHZ",
        help=(
            "Tolerance for snapping a requested frequency to an existing peak "
            f"or ledger candidate (default {REFIT_SNAP_TOL_MHZ} MHz = "
            f"{REFIT_SNAP_TOL_MHZ * 1e3:.0f} kHz)."
        ),
    )
    _add_frame_argument(p_merge)
    p_merge.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_merge.set_defaults(func=cmd_review_merge)

    # ---- review split --------------------------------------------------------
    p_split = verbs.add_parser(
        "split",
        help="Replace one fitted peak with K peaks",
        description=(
            "Replace one fitted peak with K peaks (default K=2).\n\n"
            "The replacement peaks are spread symmetrically about the named\n"
            "peak by ±½ of one Fourier resolution element (1/acquisition_us MHz).\n"
            "All products carry origin=user."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_split.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_split.add_argument(
        "--window",
        dest="window",
        type=int,
        required=True,
        metavar="N",
        help="window_id containing the peak to split.",
    )
    p_split.add_argument(
        "--peak",
        dest="peak",
        type=float,
        required=True,
        metavar="F",
        help="Molecular MHz frequency of the peak to split.",
    )
    p_split.add_argument(
        "--into",
        dest="into",
        type=int,
        default=2,
        metavar="K",
        help="Number of replacement peaks (default 2, must be ≥2).",
    )
    p_split.add_argument(
        "--snap-tol-mhz",
        dest="snap_tol_mhz",
        type=float,
        default=REFIT_SNAP_TOL_MHZ,
        metavar="MHZ",
        help=(
            "Tolerance for snapping a requested frequency to an existing peak "
            f"or ledger candidate (default {REFIT_SNAP_TOL_MHZ} MHz = "
            f"{REFIT_SNAP_TOL_MHZ * 1e3:.0f} kHz)."
        ),
    )
    _add_frame_argument(p_split)
    p_split.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_split.set_defaults(func=cmd_review_split)
