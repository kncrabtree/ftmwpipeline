"""Report generation reloads the fit O(N), not O(N^2), in the window count.

The profiling pass found the report loop reloading the whole Stage 5 fit (and
the raw FID) once *per window* via ``get_candidate_ledger_impl``; on a
277-window fit that fired the per-window peak-column loader ~277^2 = 77,560
times. The fix threads the already-loaded fit/sideband bundle into the per-window
ledger derivation so the report deserializes the fit a bounded number of times
regardless of window count.

This guards that win deterministically: count the per-window peak-column loader
(``_peaks_from_rows``, which fires exactly once per window per full fit
deserialization) across one ``report_run`` and assert the implied number of full
deserializations stays a small constant. The regression makes it scale with the
window count, so total calls would jump from ~k*N to ~N^2.
"""

from __future__ import annotations

import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.io.fitting_serialization as fserial

from .conftest import BuiltPipeline

pytestmark = [pytest.mark.performance, pytest.mark.slow]

# A full fit deserialization calls _peaks_from_rows once per window. The report
# loads the fit a small constant number of times (measured: 2 -- table + HTML).
# The O(N^2) regression reloads per window, so deserializations ~ N. This ceiling
# sits clear of the constant and well under N for any non-trivial fit.
MAX_FIT_DESERIALIZATIONS = 4


def test_report_reload_is_linear_in_windows(
    built_pipeline: BuiltPipeline, tmp_path, monkeypatch
) -> None:
    n_windows = built_pipeline.fit.n_windows
    assert n_windows > 1, "fixture must fit more than one window to be meaningful"

    calls = {"n": 0}
    original = fserial._peaks_from_rows

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    # Same-module call site: load_spectrum_fit_from_hdf5 resolves
    # _peaks_from_rows through the module namespace, so every deserialization
    # path is counted regardless of which module imported the loader.
    monkeypatch.setattr(fserial, "_peaks_from_rows", counting)

    out_dir = tmp_path / "report"
    out_dir.mkdir()
    ftmw.report_run(built_pipeline.path, output_dir=out_dir, jobs=1)

    deserializations = calls["n"] / n_windows
    print(
        f"\nreport _peaks_from_rows calls: {calls['n']} over {n_windows} windows "
        f"=> {deserializations:.2f} full fit deserializations "
        f"(N^2 regression would be ~{n_windows * n_windows})"
    )

    assert calls["n"] >= n_windows, (
        "expected at least one full fit deserialization during the report; "
        f"saw {calls['n']} calls over {n_windows} windows"
    )
    assert deserializations <= MAX_FIT_DESERIALIZATIONS, (
        f"report deserialized the fit ~{deserializations:.1f} times per render "
        f"({calls['n']} loader calls / {n_windows} windows) -- the per-window "
        "reload regressed from O(N) toward O(N^2)"
    )
