"""
Stage 6 review commands.

Implements the ``review show``, ``review edit``, ``review merge``,
``review split``, and ``review run`` subcommands.  Thin wrappers over
:mod:`ftmwpipeline._internal.stage6_impl` -- identical behaviour to
:class:`~ftmwpipeline.Pipeline` and the functional API.
"""

import argparse
from typing import Any, List, Optional, Sequence

from .._internal.stage6_impl import (
    DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    DEFAULT_DISPLAY_BAR,
    RefitWindowResult,
    ReviewRunResult,
    get_candidate_ledger_impl,
    get_review_status_impl,
    merge_peaks_impl,
    refit_window_impl,
    review_accept_impl,
    review_run_impl,
    split_peak_impl,
)
from ..core.data_structures import FittingResult, LedgerCandidate, Stage6Review
from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
from .utils import add_stage_object, setup_logging

import h5py


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


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

    # ---- render window fit to file (--window N --output PATH) ----------------
    if window_filter is not None and output_path is not None:
        try:
            from .._internal.stage5_impl import render_fit_detail_impl
            import matplotlib

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

    try:
        result: RefitWindowResult = refit_window_impl(
            file_path,
            window_id,
            add=add_freqs,
            remove=remove_freqs,
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

    try:
        result = merge_peaks_impl(file_path, window_id, peak_freqs)
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

    try:
        result = split_peak_impl(file_path, window_id, peak_freq, into=into)
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

    try:
        result = review_accept_impl(file_path, window_id, candidate_freq=candidate_freq)
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

    try:
        result = review_run_impl(
            file_path, bar=bar, attention_candidate_evidence=attention_bar
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
            "merge, split peaks).\n\n"
            "Verbs: run, show, edit, merge, split, accept"
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
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_run.set_defaults(func=cmd_review_run)

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
        nargs="*",
        default=[],
        metavar="F",
        help=(
            "Molecular MHz frequencies of peaks to add. "
            "Accepts one or more values: --add F1 F2 ..."
        ),
    )
    p_edit.add_argument(
        "--remove",
        dest="remove",
        type=float,
        nargs="*",
        default=[],
        metavar="F",
        help=(
            "Molecular MHz frequencies of fitted peaks to remove. "
            "Accepts one or more values: --remove F1 F2 ..."
        ),
    )
    p_edit.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_edit.set_defaults(func=cmd_review_edit)

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
        nargs="+",
        required=True,
        metavar="F",
        help="Molecular MHz frequencies of the peaks to collapse (≥2).",
    )
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
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_split.set_defaults(func=cmd_review_split)
