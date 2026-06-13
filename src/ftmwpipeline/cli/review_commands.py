"""
Stage 6 review commands.

Implements the ``review show`` subcommand (Pass 1: candidate ledger display).
Thin wrapper over :mod:`ftmwpipeline._internal.stage6_impl` -- identical
behaviour to :class:`~ftmwpipeline.Pipeline` and the functional API.

Pass 2 verbs (``review run``, ``review edit``, ``review accept``,
``review merge``, ``review split``) are not implemented here.
"""

import argparse
from typing import Any, List, Optional

from .._internal.stage6_impl import DEFAULT_DISPLAY_BAR, get_candidate_ledger_impl
from ..core.data_structures import FittingResult, LedgerCandidate
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


def cmd_review_show(args: argparse.Namespace) -> int:
    """Show the Stage 6 review surface.

    Without flags: per-window summary (fitted peak count and candidate count).
    ``--candidates``: tabulated candidate ledger for all (or a chosen) window.
    ``--window N``: detail for one window (fitted peaks + its candidates).
    """
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    bar: float = getattr(args, "bar", DEFAULT_DISPLAY_BAR)
    window_filter: Optional[int] = getattr(args, "window", None)
    show_candidates: bool = getattr(args, "candidates", False)

    try:
        window_fits = _load_window_fits(file_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if window_filter is not None:
        window_fits = [wf for wf in window_fits if wf.window_id == window_filter]
        if not window_fits:
            print(f"Error: window_id={window_filter} not found in the Stage 5 fit.")
            return 1

    # ---- per-window summary (default) ----------------------------------------
    if not show_candidates and window_filter is None:
        # Header
        print(
            f"{'win':>5}  {'freq_lo':>12}  {'freq_hi':>12}  "
            f"{'peaks':>6}  {'candidates':>10}"
        )
        print("-" * 56)
        for wf in sorted(window_fits, key=lambda w: w.window_id or 0):
            wid = wf.window_id if wf.window_id is not None else -1
            cands = get_candidate_ledger_impl(file_path, window_id=wid, bar=bar)
            flo = fhi = float("nan")
            if wf.window is not None and wf.window.freq_range is not None:
                flo, fhi = wf.window.freq_range
            print(
                f"{wid:>5}  {_fmt_mhz(flo):>12}  {_fmt_mhz(fhi):>12}  "
                f"{wf.n_peaks_fitted:>6}  {len(cands):>10}"
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


def register_review_commands(subparsers: Any) -> None:
    """Register review (Stage 6 Pass 1) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "review",
        synonym="stage6",
        help="Stage 6: review fitted model and candidate ledger (show)",
        description=(
            "Stage 6 review surface (Pass 1).\n\n"
            "Inspect the automatic fit, view per-window summaries, and explore\n"
            "the candidate ledger of revivable lines that the automatic pipeline\n"
            "considered but did not accept.\n\n"
            "Pass 2 verbs (edit / merge / split / accept / run) are not yet\n"
            "implemented."
        ),
    )

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
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_show.set_defaults(func=cmd_review_show)
