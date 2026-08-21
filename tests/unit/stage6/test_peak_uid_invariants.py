"""
Cross-cutting ``peak_uid`` invariants over a whole Stage 5 fit.

A ``peak_uid`` is the point-space position (in hundredths of a point of the
active FT) a fitted peak was seeded at, stamped once at birth and carried
unchanged through every subsequent fit and curation edit -- it is never
re-derived from a fitted position. The other Stage 6 test modules each pin one
birth site (a merge product, a split product, an undo replay); this module
pins invariants that only make sense over a WHOLE fit or a whole cascade:
global uniqueness across every window, and survival through a cascade refit
that a window's own edit never asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import h5py
import pytest

from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage6_impl import apply_curation_impl
from ftmwpipeline.core.data_structures import FittedPeak
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5


def _peaks_by_window(path: Path) -> Dict[int, List[FittedPeak]]:
    """Every window's fitted peaks (not just their frequencies), keyed by
    ``window_id`` -- the identifiers live on the peaks themselves."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): list(wf.fitted_peaks)
        for wf in sf.window_fits
        if wf.window_id is not None
    }


@pytest.mark.integration
def test_peak_uid_unique_across_whole_fit(stage5_multi_file):
    """Every stamped ``peak_uid`` is globally unique across the whole
    ``SpectrumFit`` -- not merely unique within the window that produced it."""
    by_window = _peaks_by_window(stage5_multi_file)

    total_peaks = sum(len(peaks) for peaks in by_window.values())
    nonempty_windows = [wid for wid, peaks in by_window.items() if peaks]
    assert total_peaks >= 5 and len(nonempty_windows) >= 2, (
        f"fixture too sparse to pin the invariant: {total_peaks} peaks across "
        f"{len(nonempty_windows)} non-empty windows (of {len(by_window)} total) "
        "-- the assertion below would pass vacuously"
    )

    owners: Dict[int, List[int]] = {}
    unstamped: List[tuple] = []
    for wid, peaks in by_window.items():
        for p in peaks:
            if p.peak_uid is None:
                unstamped.append((wid, p.frequency_mhz))
            else:
                owners.setdefault(p.peak_uid, []).append(wid)

    assert (
        not unstamped
    ), f"peaks with no stamped peak_uid, as (window_id, frequency_mhz): {unstamped}"

    collisions = {uid: wids for uid, wids in owners.items() if len(wids) > 1}
    assert not collisions, "peak_uid collided across windows: " + "; ".join(
        f"peak_uid={uid} appeared in windows {sorted(wids)}"
        for uid, wids in collisions.items()
    )


@pytest.mark.integration
def test_cascade_refit_preserves_dependent_peak_uid(
    stage5_multi_file, tmp_path, monkeypatch
):
    """A window downstream of an edited one is refit by the cascade even
    though nobody edited it directly. ``_cascade_refit_dependents``
    (stage6_impl.py) rebuilds the dependent's frozen background and calls
    ``refit_window_core`` with no add and no remove, so the dependent's free
    peaks are warm-started from its own persisted peak set -- their
    ``peak_uid`` values must come through unchanged.

    The fixture's windows are dependency-free by construction, so the
    dependency is faked exactly as ``test_cascade_downstream_of_two_edits_refit_once``
    does in ``test_curation.py``: monkeypatch ``_cascade_succs`` to add ``w2``
    as a successor of ``w0``, and spy on ``refit_window_core`` to confirm the
    cascade actually reached it. Everything downstream of that graph
    (closure, topological order, the refit itself) stays real.

    The faked edge carries no frozen background with it -- ``w2`` has no
    ``frozen_peak_*`` entries naming ``w0``, so ``_refresh_frozen_window_level``
    rebuilds nothing and the cascade reduces to an identity refit. That is the
    case this pins: a window refit purely because something upstream changed
    keeps its identifiers. A refit whose frozen background genuinely moved is
    the same ``refit_window_core`` call with a different fixed model.

    The edit is a ``remove``, not an ``add``: a window's center is the birth
    position of the peak Stage 4 built the window around, so an ill-chosen
    ``add`` frequency can collide with an existing peak's identity and get
    refused as a duplicate identifier. A ``remove`` triggers the same cascade
    with no such risk.
    """
    by_window = _peaks_by_window(stage5_multi_file)

    w0 = next(
        (wid for wid, peaks in sorted(by_window.items()) if len(peaks) >= 2),
        None,
    )
    assert w0 is not None, "need a window with >= 2 fitted peaks to remove one from"

    w2 = next(
        (wid for wid, peaks in sorted(by_window.items()) if wid != w0 and peaks),
        None,
    )
    assert (
        w2 is not None
    ), "need a second window with fitted peaks to act as the faked dependent"

    before_uids = sorted(p.peak_uid for p in by_window[w2])

    orig_succs = s6._cascade_succs

    def fake_succs(window_fits, fit_window_map):
        d = orig_succs(window_fits, fit_window_map)
        d.setdefault(w0, set()).add(w2)
        return d

    monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

    orig_core = s6.refit_window_core
    refit_ids: List[int] = []

    def spy_core(fit_ctx, fit_win, wf, **kwargs):
        refit_ids.append(int(fit_win.window_id))
        return orig_core(fit_ctx, fit_win, wf, **kwargs)

    monkeypatch.setattr(s6, "refit_window_core", spy_core)

    remove_freq = sorted(float(p.frequency_mhz) for p in by_window[w0])[0]
    cur = tmp_path / "remove.csv"
    cur.write_text(f"remove,{w0},{remove_freq},\n")
    apply_curation_impl(stage5_multi_file, cur)

    assert w2 in refit_ids, (
        f"cascade never reached the faked dependent w2={w2}; "
        f"refit_window_core was called for windows {refit_ids}"
    )

    after_uids = sorted(p.peak_uid for p in _peaks_by_window(stage5_multi_file)[w2])
    assert after_uids == before_uids, (
        f"w2={w2}'s peak_uid set changed across the cascade refit it never "
        f"asked for: before={before_uids} after={after_uids}"
    )
