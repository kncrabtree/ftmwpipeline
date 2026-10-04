"""``<column>__status`` companions on the columnar tables (Wave 3).

Written against ``dev-docs/CONTRACT_STRATEGY.md`` §Missing values and its
migration rules: the value column keeps its stored fill, a ``uint8`` status
column says whether that fill is a real value (``0``), a quantity that was never
computed (``1``) or one computed with no finite value (``2``); a column that
predates the file is ``NOT_RUN``. Files are built with h5py in ``tmp_path``.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.absence_rules import clock_lattice_or_absent
from ftmwpipeline._internal.read_impl import (
    read_table_impl,
    read_tables_impl,
)
from ftmwpipeline.contract import MANIFEST, Absent
from ftmwpipeline.core.data_structures import (
    AuditStep,
    DoubletAlternativeInfo,
    FittedPeak,
    FittingResult,
    FitWindow,
    KnockoutInfo,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.io.fitting_serialization import save_spectrum_fit_to_hdf5
from ftmwpipeline.io.window_serialization import save_window_plan_to_hdf5

NR = Absent.NOT_RUN.status
UD = Absent.UNDEFINED.status
NAN = float("nan")


def _peak(k, *, knockout=None, **over):
    kw = dict(
        detection_index=k,
        frequency_mhz=100.0 + k,
        amplitude=0.5,
        decay_rate=0.2,
        phase=0.3,
        frequency_error=1e-4,
        amplitude_error=0.01,
        decay_rate_error=2e-3,
        phase_error=0.02,
        snr=10.0,
        chi_squared=1.0,
        window_id=0,
        knockout=knockout,
    )
    kw.update(over)
    return FittedPeak(**kw)


def _step(decision, **over):
    kw = dict(
        n_peaks_before=1,
        candidate_offset_mhz=0.1,
        chi2_before=2.0,
        chi2_after=1.5,
        f_statistic=3.0,
        p_value=0.01,
        aic_before=5.0,
        aic_after=4.0,
        separation_ok=True,
        decision=decision,
        n_eff=10.0,
        aicc_delta=-1.0,
    )
    kw.update(over)
    return AuditStep(**kw)


def _build(path, peaks, audit=()):
    result = FittingResult(
        success=True,
        fitted_spectrum=None,
        cost=1.0,
        iterations=2,
        aic=10.0,
        reduced_chi2=1.1,
        window=None,
        window_id=0,
        shape="lorentzian",
    )
    result.fitted_peaks = list(peaks)
    result.audit_trail = list(audit)
    fit = SpectrumFit(
        window_fits=[result],
        fitted_peaks=list(peaks),
        parameters={"acquisition_us": 12.73},
    )
    with h5py.File(path, "w") as h5f:
        h5f.attrs["ftmw_format_version"] = "1.0"
        save_window_plan_to_hdf5(
            WindowPlan(
                windows=[FitWindow(window_id=0, freq_range=(99.0, 120.0), batch=0)],
                dependency_edges=[],
                topological_order=[0],
            ),
            h5f.create_group("stage4_windows"),
        )
        save_spectrum_fit_to_hdf5(fit, h5f.create_group("stage5_fitting"))
    return path


def _ko(supported, delta, p=0.01, aicc=2.0):
    return KnockoutInfo(
        delta_chi2=delta,
        expected_delta_chi2=1.0,
        supported=supported,
        p_value=p,
        n_eff=5.0,
        aicc_delta=aicc,
    )


def test_knockout_block_shares_one_not_run_rule(tmp_path):
    f = _build(
        tmp_path / "k.ftmw",
        [
            _peak(0),  # no knockout record
            _peak(1, knockout=_ko(True, 4.0)),
            _peak(2, knockout=_ko(True, 4.0, p=NAN, aicc=float("inf"))),
        ],
    )
    t = read_table_impl(f, "fit_peaks")
    for col in (
        "knockout_supported",
        "knockout_delta_chi2",
        "knockout_expected_delta_chi2",
        "chi_squared",
        "knockout_p_value",
        "knockout_n_eff",
        "knockout_aicc_delta",
    ):
        st = t[col + "__status"]
        assert st.dtype == np.uint8
        assert st[0] == NR, col
    assert list(t["knockout_supported"][:1]) == [-1]
    assert list(t["knockout_p_value__status"]) == [NR, 0, UD]
    assert list(t["knockout_aicc_delta__status"]) == [NR, 0, UD]
    assert list(t["knockout_supported__status"]) == [NR, 0, 0]


def test_fit_peak_undefined_and_never_merged(tmp_path):
    f = _build(
        tmp_path / "p.ftmw",
        [_peak(0, decay_rate_error=NAN, frequency_error=NAN), _peak(1)],
    )
    t = read_table_impl(f, "fit_peaks")
    assert list(t["decay_rate_error__status"]) == [UD, 0]
    assert list(t["frequency_error__status"]) == [UD, 0]
    assert list(t["snr__status"]) == [0, 0]
    assert list(t["unresolved_spread_mhz__status"]) == [NR, NR]
    assert list(t["derivation__status"]) == [NR, NR]
    assert list(t["peak_uid__status"]) == [NR, NR]
    assert np.isnan(t["decay_rate_error"][0])  # the value keeps its fill


def test_synthesized_column_is_not_run_except_under_the_knockout_rule(tmp_path):
    f = _build(tmp_path / "s.ftmw", [_peak(0, knockout=_ko(True, 4.0))])
    with h5py.File(f, "r+") as h5f:
        del h5f["stage5_fitting/peaks/knockout_p_value"]
        del h5f["stage5_fitting/windows/tau_error"]
    # The knockout rule takes precedence over the synthesized fill: the test
    # ran, so a p-value the file predates is UNDEFINED, as on FinalPeak.
    assert list(
        read_table_impl(f, "fit_peaks", ["knockout_p_value__status"])[
            "knockout_p_value__status"
        ]
    ) == [UD]
    assert list(
        read_table_impl(f, "fit_windows", ["tau_error__status"])["tau_error__status"]
    ) == [NR]


def test_fit_windows_status(tmp_path):
    f = _build(tmp_path / "w.ftmw", [_peak(0)])
    with h5py.File(f, "r+") as h5f:
        h5f["stage5_fitting/windows/reduced_chi2"][0] = float("inf")
        h5f["stage5_fitting/windows/tau_error"][0] = NAN
    t = read_table_impl(f, "fit_windows")
    assert list(t["reduced_chi2__status"]) == [UD]
    assert list(t["tau_error__status"]) == [UD]
    assert t["aic__status"][0] == 0


def test_fit_audit_masks_unrun_gates(tmp_path):
    steps = [
        _step("seed", n_eff=NAN, aicc_delta=NAN),
        _step("accept"),
        _step(
            "reject",
            separation_ok=False,
            f_statistic=0.0,
            p_value=1.0,
            n_eff=NAN,
            aicc_delta=NAN,
        ),
        _step(
            "spur-drop",
            chi2_after=NAN,
            f_statistic=NAN,
            p_value=NAN,
            aic_after=NAN,
            n_eff=NAN,
            aicc_delta=NAN,
        ),
        _step("knockout-null", f_statistic=NAN, p_value=NAN),
    ]
    f = _build(tmp_path / "a.ftmw", [_peak(0)], steps)
    t = read_table_impl(f, "fit_audit")
    assert list(t["n_eff__status"]) == [NR, 0, NR, NR, 0]
    assert list(t["f_statistic__status"]) == [0, 0, NR, NR, NR]
    assert list(t["p_value__status"]) == [0, 0, NR, NR, UD]
    assert list(t["chi2_after__status"]) == [0, 0, 0, NR, 0]
    # The fabricated 0.0 / 1.0 of the separation reject are NaN, not values.
    assert np.isnan(t["f_statistic"][2]) and np.isnan(t["p_value"][2])
    assert t["f_statistic"][0] == 3.0
    only = read_table_impl(f, "fit_audit", ["p_value"])
    assert np.isnan(only["p_value"][2])


def test_peaks_promoted_without_a_cutoff_is_not_run(tmp_path):
    f = _build(tmp_path / "c.ftmw", [_peak(0)])
    with h5py.File(f, "a") as h5f:
        g = h5f.create_group("stage3_peaks")
        g.attrs["n_peaks"] = 2
        g.create_dataset("frequency", data=[1.0, 2.0])
        g.create_dataset("intensity", data=[1.0, 1.0])
        g.create_dataset("index", data=[3, -1])
        g.create_dataset("snr", data=[9.0, NAN])
        g.create_dataset("noise_std_local", data=[1.0, 1.0])
        g.create_dataset("classification", data=["ok", ""])
        g.create_dataset("detection_pass", data=["a", "a"])
    t = read_table_impl(f, "peaks")
    assert list(t["promoted__status"]) == [NR, NR]
    assert list(t["index__status"]) == [0, NR]
    assert list(t["classification__status"]) == [0, NR]
    assert list(t["snr__status"]) == [0, UD]
    assert list(t["internal_snr__status"]) == [NR, NR]  # synthesized
    with h5py.File(f, "a") as h5f:
        h5f["stage3_peaks"].attrs["promotion_min_snr"] = 5.0
    t = read_table_impl(f, "peaks")
    assert list(t["promoted__status"]) == [0, 0]
    assert list(t["promoted"]) == [True, False]


def test_listing_declares_status_columns_and_counts_rows(tmp_path):
    f = _build(tmp_path / "l.ftmw", [_peak(0)], [_step("seed")])
    listing = read_tables_impl(f)
    for table in ("fit_peaks", "fit_windows", "fit_audit", "fit_doublets", "peaks"):
        cols = listing[table]["columns"]
        assert any(c.endswith("__status") for c in cols), table
    assert set(MANIFEST.tables["fit_peaks"]) == set(listing["fit_peaks"]["columns"])
    # Event logs record no count attribute; the count is computed, an int.
    assert listing["fit_audit"]["n_rows"] == 1
    assert listing["fit_doublets"]["n_rows"] == 0
    assert listing["peaks"]["available"] is False
    assert listing["peaks"]["n_rows"] is Absent.NOT_RUN
    assert listing["fit_audit"]["group"] == "stage5_fitting"


def test_unknown_status_column_is_refused(tmp_path):
    f = _build(tmp_path / "u.ftmw", [_peak(0)])
    with pytest.raises(ValueError, match="unknown column"):
        read_table_impl(f, "fit_peaks", ["shape__status"])
    json.dumps(read_tables_impl(f), default=str)


def _doublet(**over):
    kw = dict(
        frequency_a_mhz=100.0,
        frequency_b_mhz=100.1,
        amplitude_a=1.0,
        amplitude_b=0.5,
        separation_res_elements=1.2,
        amp_ratio=0.5,
        chi2r_production=1.0,
        chi2r_merged=1.4,
        delta_chi2_raw=3.0,
        delta_aicc=2.0,
        merged_frequency_mhz=100.05,
        merged_amplitude=1.4,
        merged_phase=0.1,
        merged_tau_us=5.0,
        merged_success=True,
        orth_evidence_delta_chi2=4.0,
        orth_evidence_n_params=3,
        support_bins=10,
    )
    kw.update(over)
    return DoubletAlternativeInfo(**kw)


def test_fit_doublets_failed_refit_is_undefined(tmp_path):
    # Mutation: dropping fit_doublets from the status layout removes the
    # companion columns (KeyError below); treating NaN as present gives 0.
    peaks = [_peak(0)]
    f = _build(tmp_path / "d.ftmw", peaks)
    result = FittingResult(
        success=True,
        fitted_spectrum=None,
        cost=1.0,
        iterations=2,
        aic=10.0,
        reduced_chi2=1.1,
        window=None,
        window_id=0,
        shape="lorentzian",
    )
    result.fitted_peaks = peaks
    result.doublet_alternatives = [
        _doublet(),
        _doublet(
            chi2r_merged=NAN,
            delta_chi2_raw=NAN,
            delta_aicc=NAN,
            merged_frequency_mhz=NAN,
            merged_amplitude=NAN,
            merged_phase=NAN,
            merged_tau_us=NAN,
            orth_evidence_delta_chi2=NAN,
            merged_success=False,
        ),
    ]
    fit = SpectrumFit(
        window_fits=[result], fitted_peaks=peaks, parameters={"acquisition_us": 12.7}
    )
    with h5py.File(f, "r+") as h5f:
        del h5f["stage5_fitting"]
        save_spectrum_fit_to_hdf5(fit, h5f.create_group("stage5_fitting"))
    t = read_table_impl(f, "fit_doublets")
    for col in (
        "chi2r_merged",
        "delta_chi2_raw",
        "delta_aicc",
        "merged_frequency_mhz",
        "merged_amplitude",
        "merged_phase",
        "merged_tau_us",
        "orth_evidence_delta_chi2",
    ):
        assert list(t[col + "__status"]) == [0, UD], col
        assert t[col + "__status"].dtype == np.uint8
    assert np.isnan(t["delta_aicc"][1])


def test_clock_lattice_rule_and_table(tmp_path, monkeypatch):
    # Mutation: ignoring the declaration (always UNDEFINED for '') or always
    # NOT_RUN turns one of the asserts below red.
    assert clock_lattice_or_absent("", declared=False) is Absent.NOT_RUN
    assert clock_lattice_or_absent("", declared=True) is Absent.UNDEFINED
    assert clock_lattice_or_absent("10 MHz", declared=True) == "10 MHz"
    f = _build(tmp_path / "cl.ftmw", [_peak(0)])
    t = read_table_impl(f, "fit_peaks", ["clock_lattice__status"])
    assert list(t["clock_lattice__status"]) == [NR]  # no declaration persisted
    import ftmwpipeline._internal.read_impl as ri

    monkeypatch.setattr(ri, "fit_declares_clocks", lambda _h5f: True)
    t = read_table_impl(f, "fit_peaks", ["clock_lattice__status"])
    assert list(t["clock_lattice__status"]) == [UD]  # declared, off the lattice


def test_detection_index_and_window_fills(tmp_path):
    # Mutation: detection_index -1 mapped to NOT_RUN, or tau_fitted/freq fills
    # reported present.
    f = _build(tmp_path / "di.ftmw", [_peak(-1), _peak(3)])
    with h5py.File(f, "r+") as h5f:
        h5f["stage5_fitting/windows/tau_fitted"][0] = -1
        h5f["stage5_fitting/windows/freq_min"][0] = NAN
        h5f["stage5_fitting/windows/freq_max"][0] = 120.0
        h5f["stage5_fitting/windows/tau_us"][0] = 0.0
    t = read_table_impl(f, "fit_peaks", ["detection_index", "detection_index__status"])
    assert list(t["detection_index"]) == [-1, 3]
    assert list(t["detection_index__status"]) == [UD, 0]
    w = read_table_impl(f, "fit_windows")
    assert list(w["tau_fitted__status"]) == [NR]
    assert list(w["freq_min__status"]) == [NR]
    assert list(w["freq_max__status"]) == [0]
    assert list(w["tau_us__status"]) == [UD]
    assert w["tau_fitted"][0] == -1


def test_read_list_prints_row_counts(tmp_path, capsys):
    # Mutation: reverting cmd_read_list to `is not None` prints the Absent
    # repr / "available" instead of the integer count.
    from ftmwpipeline.cli import main

    f = _build(tmp_path / "rl.ftmw", [_peak(0)], [_step("seed")])
    assert main(["read", "list", str(f)]) == 0
    out = capsys.readouterr().out
    assert "fit_audit" in out and "1 row(s)" in out
    assert "fit_peaks" in out and "knockout_p_value__status" in out
