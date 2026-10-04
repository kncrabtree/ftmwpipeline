"""Degenerate statistics read ``UNDEFINED``, old files included.

Written against ``dev-docs/CONTRACT_STRATEGY.md`` §Missing values, the migration
rule "Degenerate statistics are ``UNDEFINED``": an F-test with no residual
degrees of freedom or a non-positive chi-squared, the doublet orthogonal-evidence
fallback, an undefined residual edge coherence and a Stage 3 SNR without positive
local noise were stored by earlier writers as ordinary numbers. Writers store
``nan`` for them now; every ``read_table`` column reads them as ``nan`` with
status ``2`` wherever what the file stores shows the statistic was degenerate,
whichever writer produced the file.

Each table is built twice with h5py/the serializers in ``tmp_path`` -- once in
the earlier writers' encoding (the ordinary numbers), once in the current
writers' (``nan``, taken from the functions that now return it) -- and the two
must read identically. A statistic that is a genuine number (a measured
non-improvement, a zero evidence on a usable support, a clamped SNR of 0 on a
positive noise) must keep reading as present in both.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.absence_rules import (
    degenerate_edge_coherence,
    degenerate_f_test,
    degenerate_internal_snr,
    degenerate_orth_evidence,
    degenerate_stage3_snr,
)
from ftmwpipeline._internal.read_impl import read_table_impl
from ftmwpipeline.contract import Absent
from ftmwpipeline.core.data_structures import (
    FittingResult,
    FitWindow,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.fitting.plan_execution import residual_edge_coherence
from ftmwpipeline.fitting.validation import calculate_chi_squared_improvement
from ftmwpipeline.io.fitting_serialization import (
    read_fit_window_quality_recorded,
    save_spectrum_fit_to_hdf5,
)
from ftmwpipeline.io.window_serialization import save_window_plan_to_hdf5

# The builders are shared with the status-column suite.
from tests.unit._internal.test_read_status_columns import (
    _doublet,
    _peak,
    _step,
)

NR = Absent.NOT_RUN.status
UD = Absent.UNDEFINED.status
NAN = float("nan")
STYLES = ("earlier", "current")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _result(window_id, *, audit=(), doublets=(), quality=None):
    result = FittingResult(
        success=True,
        fitted_spectrum=None,
        cost=1.0,
        iterations=2,
        aic=10.0,
        reduced_chi2=1.1,
        window=None,
        window_id=window_id,
        shape="lorentzian",
    )
    result.fitted_peaks = [_peak(window_id, window_id=window_id)]
    result.audit_trail = list(audit)
    result.doublet_alternatives = list(doublets)
    result.quality_metrics = dict(quality or {})
    return result


def _build(path, results):
    peaks = [p for r in results for p in r.fitted_peaks]
    fit = SpectrumFit(
        window_fits=list(results),
        fitted_peaks=peaks,
        parameters={"acquisition_us": 12.73},
    )
    ids = [r.window_id for r in results]
    with h5py.File(path, "w") as h5f:
        h5f.attrs["ftmw_format_version"] = "1.0"
        save_window_plan_to_hdf5(
            WindowPlan(
                windows=[
                    FitWindow(window_id=i, freq_range=(99.0, 120.0), batch=0)
                    for i in ids
                ],
                dependency_edges=[],
                topological_order=ids,
            ),
            h5f.create_group("stage4_windows"),
        )
        save_spectrum_fit_to_hdf5(fit, h5f.create_group("stage5_fitting"))
    return path


# ---------------------------------------------------------------------------
# The shared predicates
# ---------------------------------------------------------------------------
class TestPredicates:
    def test_f_test_zero_and_unity_is_degenerate_unless_a_non_improvement(self):
        # (F, p, chi2_before, chi2_after)
        assert degenerate_f_test(0.0, 1.0, 2.0, 1.5)  # improvement, yet F = 0
        assert degenerate_f_test(0.0, 1.0, 2.0, 0.0)  # non-positive chi2
        assert degenerate_f_test(0.0, 1.0, 2.0, -1.0)
        assert degenerate_f_test(0.0, 1.0, 2.0, np.nan)  # no chi2 to vouch for it
        assert not degenerate_f_test(0.0, 1.0, 2.0, 2.0)  # genuine, equal
        assert not degenerate_f_test(0.0, 1.0, 2.0, 2.5)  # genuine, worse
        assert not degenerate_f_test(3.0, 0.01, 2.0, 1.5)  # a real test
        assert not degenerate_f_test(0.0, 0.5, 2.0, 1.5)  # F = 0 alone is not enough
        assert not degenerate_f_test(1.0, 1.0, 2.0, 1.5)

    def test_f_test_takes_arrays(self):
        out = degenerate_f_test(
            np.array([0.0, 0.0, 3.0]),
            np.array([1.0, 1.0, 0.01]),
            np.array([2.0, 2.0, 2.0]),
            np.array([1.5, 2.5, 1.5]),
        )
        assert out.tolist() == [True, False, False]

    def test_edge_coherence_exact_zero_only(self):
        assert degenerate_edge_coherence(0.0)
        assert not degenerate_edge_coherence(1e-12)
        assert not degenerate_edge_coherence(np.nan)  # nan is the ordinary rule's
        assert degenerate_edge_coherence(np.array([0.0, 2.0])).tolist() == [True, False]

    def test_orth_evidence_needs_zero_value_and_zero_support(self):
        assert degenerate_orth_evidence(0.0, 0)
        assert not degenerate_orth_evidence(0.0, 10)
        assert not degenerate_orth_evidence(4.0, 0)
        out = degenerate_orth_evidence(np.array([0.0, 0.0]), np.array([0, 5]))
        assert out.tolist() == [True, False]

    def test_stage3_snr_degenerate_where_noise_not_positive(self):
        assert degenerate_stage3_snr(0.0)
        assert degenerate_stage3_snr(-1.0)
        assert not degenerate_stage3_snr(1.0)
        assert not degenerate_stage3_snr(np.nan)  # an unrecorded noise is not 0

    def test_internal_snr_needs_a_contributing_internal_pass(self):
        assert degenerate_internal_snr(0.0, 100.0)
        assert degenerate_internal_snr(np.nan, 100.0)
        assert not degenerate_internal_snr(12.0, 100.0)
        assert not degenerate_internal_snr(np.nan, np.nan)  # internal pass absent
        assert not degenerate_internal_snr(0.0, np.nan)


# ---------------------------------------------------------------------------
# fit_audit: the F-test
# ---------------------------------------------------------------------------
def _ftest(style, chi2_before, chi2_after, dof_change, n_data, n_params_new):
    """``(f_statistic, p_value)`` as the writers of *style* stored the test."""
    if style == "earlier":
        # (F, p) = (0, 1) for a non-improvement and for a degenerate test alike.
        return 0.0, 1.0
    p, f, _ = calculate_chi_squared_improvement(
        chi2_before, chi2_after, dof_change, n_data, n_params_new
    )
    return f, p


def _audit(style):
    def step(decision, before, after, dof, n_data, n_params, **over):
        f, p = _ftest(style, before, after, dof, n_data, n_params)
        return _step(
            decision,
            chi2_before=before,
            chi2_after=after,
            f_statistic=f,
            p_value=p,
            **over,
        )

    return [
        _step("seed", n_eff=NAN, aicc_delta=NAN),
        # degenerate: no residual degrees of freedom, though chi2 fell
        step("accept", 2.0, 1.5, 3, 10, 10),
        # genuine non-improvement (chi2 rose): the test ran, F = 0 and p = 1
        step("reject", 2.0, 2.5, 3, 800, 10),
        # degenerate: a non-positive chi2 after
        step("reject", 2.0, 0.0, 3, 800, 10),
        # separation reject: the F-test never ran
        _step(
            "reject",
            separation_ok=False,
            f_statistic=0.0,
            p_value=1.0,
            n_eff=NAN,
            aicc_delta=NAN,
        ),
        # spur drop: no F-test exists
        _step(
            "spur-drop",
            chi2_after=NAN,
            f_statistic=NAN,
            p_value=NAN,
            aic_after=NAN,
            n_eff=NAN,
            aicc_delta=NAN,
        ),
    ]


@pytest.mark.parametrize("style", STYLES)
def test_audit_f_test_degenerate_rows_read_undefined(tmp_path, style):
    f = _build(tmp_path / f"{style}.ftmw", [_result(0, audit=_audit(style))])
    t = read_table_impl(f, "fit_audit")
    assert list(t["f_statistic__status"]) == [0, UD, 0, UD, NR, NR]
    assert list(t["p_value__status"]) == [0, UD, 0, UD, NR, NR]
    for col in ("f_statistic", "p_value"):
        got = t[col]
        # A degenerate or never-run test has no value; a measured one does.
        assert np.isnan(got[[1, 3, 4, 5]]).all(), col
        assert np.isfinite(got[[0, 2]]).all(), col
    # The genuine non-improvement keeps its measured F = 0, p = 1.
    assert (t["f_statistic"][2], t["p_value"][2]) == (0.0, 1.0)
    assert t["f_statistic"][0] == 3.0 and t["p_value"][0] == 0.01
    # The column selection path reads the same masked values.
    only = read_table_impl(f, "fit_audit", ["p_value", "p_value__status"])
    assert list(only["p_value__status"]) == [0, UD, 0, UD, NR, NR]
    assert np.isnan(only["p_value"][1])


def test_audit_earlier_and_current_files_read_identically(tmp_path):
    reads = {}
    for style in STYLES:
        f = _build(tmp_path / f"{style}.ftmw", [_result(0, audit=_audit(style))])
        reads[style] = read_table_impl(f, "fit_audit")
    assert set(reads["earlier"]) == set(reads["current"])
    for col, a in reads["earlier"].items():
        b = reads["current"][col]
        if a.dtype.kind == "f":
            np.testing.assert_array_equal(a, b, err_msg=col)
        else:
            assert list(a) == list(b), col


def test_audit_rows_that_never_carried_an_f_test_are_not_degenerate(tmp_path):
    """A spur drop and a knockout-null step are excluded from the old-file rule:
    their F = 0, p = 1 is not a degenerate test."""
    steps = [
        _step(
            "spur-drop",
            chi2_after=1.5,
            f_statistic=0.0,
            p_value=1.0,
            n_eff=NAN,
            aicc_delta=NAN,
        ),
        # The reversed knockout test's p-value is kept as stored.
        _step("knockout-null", chi2_after=1.5, f_statistic=0.0, p_value=1.0),
    ]
    f = _build(tmp_path / "x.ftmw", [_result(0, audit=steps)])
    t = read_table_impl(f, "fit_audit")
    assert list(t["f_statistic__status"]) == [NR, NR]  # not UNDEFINED
    assert t["p_value__status"][0] == NR  # not UNDEFINED
    assert t["p_value__status"][1] == 0  # not UNDEFINED
    assert t["p_value"][1] == 1.0


def test_audit_current_knockout_null_nan_p_is_still_undefined(tmp_path):
    steps = [_step("knockout-null", f_statistic=NAN, p_value=NAN)]
    f = _build(tmp_path / "k.ftmw", [_result(0, audit=steps)])
    t = read_table_impl(f, "fit_audit")
    assert (t["f_statistic__status"][0], t["p_value__status"][0]) == (NR, UD)


# ---------------------------------------------------------------------------
# fit_windows: residual edge coherence
# ---------------------------------------------------------------------------
def _edge_windows(style):
    # The current writers' edge statistic of an empty residual is nan.
    undef = 0.0 if style == "earlier" else residual_edge_coherence(np.array([]), 1.0)[0]
    return [
        _result(0, quality={"edge_coherence_low": undef, "edge_coherence_high": 2.0}),
        _result(1, quality={"edge_coherence_low": undef, "edge_coherence_high": undef}),
        _result(2, quality={}),  # a window the fit never evaluated
        _result(3, quality={"edge_coherence_low": 1.2, "edge_coherence_high": undef}),
    ]


@pytest.mark.parametrize("style", STYLES)
def test_fit_windows_edge_coherence_statuses(tmp_path, style):
    f = _build(tmp_path / f"{style}.ftmw", _edge_windows(style))
    t = read_table_impl(f, "fit_windows")
    assert list(t["window_id"]) == [0, 1, 2, 3]
    assert list(t["edge_coherence_low__status"]) == [UD, UD, NR, 0]
    assert list(t["edge_coherence_high__status"]) == [0, UD, NR, UD]
    assert np.isnan(t["edge_coherence_low"][[0, 1, 2]]).all()
    assert np.isnan(t["edge_coherence_high"][[1, 2, 3]]).all()
    assert t["edge_coherence_high"][0] == 2.0 and t["edge_coherence_low"][3] == 1.2
    assert t["edge_coherence_low__status"].dtype == np.uint8


def test_fit_windows_earlier_and_current_files_read_identically(tmp_path):
    reads = [
        read_table_impl(
            _build(tmp_path / f"{s}.ftmw", _edge_windows(s)),
            "fit_windows",
            [
                "edge_coherence_low",
                "edge_coherence_low__status",
                "edge_coherence_high",
                "edge_coherence_high__status",
            ],
        )
        for s in STYLES
    ]
    for col in reads[0]:
        np.testing.assert_array_equal(reads[0][col], reads[1][col], err_msg=col)


def test_fit_window_quality_recorded_distinguishes_computed_from_never_fit(tmp_path):
    f = _build(tmp_path / "q.ftmw", _edge_windows("current"))
    with h5py.File(f, "r") as h5f:
        rec = read_fit_window_quality_recorded(
            h5f["stage5_fitting"], ["edge_coherence_low", "edge_coherence_high"]
        )
    assert rec["edge_coherence_low"].tolist() == [True, True, False, True]
    assert rec["edge_coherence_high"].tolist() == [True, True, False, True]


def test_fit_window_quality_recorded_without_a_quality_column(tmp_path):
    f = _build(tmp_path / "q.ftmw", _edge_windows("current"))
    with h5py.File(f, "r+") as h5f:
        del h5f["stage5_fitting/windows/quality_metrics"]
        rec = read_fit_window_quality_recorded(h5f["stage5_fitting"], ["k"])
    assert rec["k"].tolist() == [False] * 4


# ---------------------------------------------------------------------------
# fit_doublets: orthogonal evidence
# ---------------------------------------------------------------------------
def _doublets(style):
    undef = 0.0 if style == "earlier" else NAN  # a test that never ran
    return [
        _doublet(orth_evidence_delta_chi2=undef, support_bins=0),
        # A genuine zero: the template had support and found no evidence.
        _doublet(orth_evidence_delta_chi2=0.0, support_bins=10),
        # The refit failed: nan on a usable support is undefined in both.
        _doublet(orth_evidence_delta_chi2=NAN, support_bins=10, merged_success=False),
        _doublet(orth_evidence_delta_chi2=4.0, support_bins=10),
    ]


@pytest.mark.parametrize("style", STYLES)
def test_doublet_orth_evidence_statuses(tmp_path, style):
    f = _build(tmp_path / f"{style}.ftmw", [_result(0, doublets=_doublets(style))])
    t = read_table_impl(f, "fit_doublets")
    assert list(t["orth_evidence_delta_chi2__status"]) == [UD, 0, UD, 0]
    got = t["orth_evidence_delta_chi2"]
    assert np.isnan(got[[0, 2]]).all()
    assert got[1] == 0.0 and got[3] == 4.0
    assert list(t["support_bins"]) == [0, 10, 10, 10]  # the value of a column kept


def test_doublet_earlier_and_current_files_read_identically(tmp_path):
    cols = ["orth_evidence_delta_chi2", "orth_evidence_delta_chi2__status"]
    reads = [
        read_table_impl(
            _build(tmp_path / f"{s}.ftmw", [_result(0, doublets=_doublets(s))]),
            "fit_doublets",
            cols,
        )
        for s in STYLES
    ]
    for col in cols:
        np.testing.assert_array_equal(reads[0][col], reads[1][col], err_msg=col)


def test_doublet_support_bins_is_not_listed_in_a_column_selection_it_did_not_ask_for(
    tmp_path,
):
    """The mask reads ``support_bins`` internally; it does not leak into the
    selection."""
    f = _build(tmp_path / "s.ftmw", [_result(0, doublets=_doublets("earlier"))])
    t = read_table_impl(f, "fit_doublets", ["orth_evidence_delta_chi2"])
    assert "support_bins" not in t
    assert np.isnan(t["orth_evidence_delta_chi2"][0])


# ---------------------------------------------------------------------------
# peaks: Stage 3 SNR and internal SNR
# ---------------------------------------------------------------------------
def _peaks_file(path, style):
    undef = 0.0 if style == "earlier" else NAN
    snr = [9.0, undef, 7.0, 25.0, 0.0, 5.0, undef]
    noise = [1.0, 0.0, 1.0, 1.0, 1.0, NAN, 0.0]
    ifreq = [NAN, NAN, 100.0, 101.0, NAN, NAN, 102.0]
    isnr = [NAN, NAN, undef, 12.0, NAN, NAN, undef]
    n = len(snr)
    f = _build(path, [_result(0)])
    with h5py.File(f, "a") as h5f:
        g = h5f.create_group("stage3_peaks")
        g.attrs["n_peaks"] = n
        g.create_dataset("frequency", data=np.arange(n, dtype=float))
        g.create_dataset("intensity", data=np.ones(n))
        g.create_dataset("index", data=np.arange(n))
        g.create_dataset("snr", data=snr)
        g.create_dataset("noise_std_local", data=noise)
        g.create_dataset("classification", data=["weak"] * n)
        g.create_dataset("detection_pass", data=["primary"] * n)
        g.create_dataset("internal_snr", data=isnr)
        g.create_dataset("internal_frequency", data=ifreq)
    return f


@pytest.mark.parametrize("style", STYLES)
def test_peak_snr_statuses(tmp_path, style):
    f = _peaks_file(tmp_path / f"{style}.ftmw", style)
    t = read_table_impl(f, "peaks")
    # Row meaning: 0 ordinary; 1 zero local noise; 2 internal pass, internal
    # noise zero; 3 internal pass, ordinary; 4 SNR clamped to 0 on positive
    # noise; 5 noise not recorded; 6 zero noise and internal-pass zero noise.
    assert list(t["snr__status"]) == [0, UD, 0, 0, 0, 0, UD]
    assert list(t["internal_snr__status"]) == [NR, NR, UD, 0, NR, NR, UD]
    assert np.isnan(t["snr"][[1, 6]]).all()
    assert t["snr"][4] == 0.0  # a clamped excess is a measured zero, not undefined
    assert t["snr"][5] == 5.0  # an unrecorded noise does not make the SNR undefined
    assert np.isnan(t["internal_snr"][[2, 6]]).all()
    assert t["internal_snr"][3] == 12.0
    assert t["snr__status"].dtype == np.uint8


def test_peaks_earlier_and_current_files_read_identically(tmp_path):
    cols = ["snr", "snr__status", "internal_snr", "internal_snr__status"]
    reads = [
        read_table_impl(_peaks_file(tmp_path / f"{s}.ftmw", s), "peaks", cols)
        for s in STYLES
    ]
    for col in cols:
        np.testing.assert_array_equal(reads[0][col], reads[1][col], err_msg=col)


def test_peak_snr_column_selection_without_the_needed_columns(tmp_path):
    """Asking for ``snr`` alone still masks it; the helpers' inputs are read
    internally."""
    f = _peaks_file(tmp_path / "e.ftmw", "earlier")
    t = read_table_impl(f, "peaks", ["snr"])
    assert set(t) == {"snr"}
    assert np.isnan(t["snr"][[1, 6]]).all()
    t = read_table_impl(f, "peaks", ["internal_snr__status"])
    assert list(t["internal_snr__status"]) == [NR, NR, UD, 0, NR, NR, UD]
