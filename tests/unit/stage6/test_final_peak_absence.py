"""Absence encodings of the Stage 6 results (machine contract, Wave 3).

``FinalPeak``'s pre-contract fields, ``CalibrationStamp``'s header fields and
the curation results' chi2r / convergence / created-window fields carry
:class:`~ftmwpipeline.contract.Absent` instead of ``None`` / ``nan`` (normative:
``dev-docs/CONTRACT_STRATEGY.md`` §Missing values, "Migration rules for
pre-contract fields"). Storage keeps its encodings; a stored table written
before the status keys existed is decoded field by field without a rebuild.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import pytest

from ftmwpipeline._internal.stage6_impl import (
    PreviewWindowResult,
    _build_final_products,
    frequency_calibration_impl,
)
from ftmwpipeline.cli.review_commands import _fmt_chi2r
from ftmwpipeline.contract import Absent
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    KnockoutInfo,
    Sideband,
    SpectrumFit,
    Stage6Review,
)
from ftmwpipeline.file_manager import StageDependencyError
from ftmwpipeline.io.stage6_review_serialization import (
    FINAL_PEAK_ABSENT_FIELDS,
    fit_declares_clocks,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.serialize import to_jsonable

PROBE = 40960.0


def _peak(**kw) -> FittedPeak:
    base = dict(
        detection_index=0,
        frequency_mhz=30000.0,
        amplitude=1.0,
        phase=0.5,
        frequency_error=0.0002,
        amplitude_error=0.01,
        phase_error=0.02,
        snr=100.0,
        window_id=3,
        origin="auto",
        peak_uid=123,
    )
    base.update(kw)
    return FittedPeak(**base)


def _ko(**kw) -> KnockoutInfo:
    base = dict(
        delta_chi2=12.0,
        expected_delta_chi2=9.0,
        supported=True,
        p_value=1e-3,
        n_eff=200.0,
        aicc_delta=5.0,
    )
    base.update(kw)
    return KnockoutInfo(**base)


def _build(peaks, *, declared=False, sigma_eps=0.0, floor=0.0):
    return _build_final_products(
        SpectrumFit(fitted_peaks=list(peaks)),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state="rb_locked",
        epsilon=0.0,
        sigma_epsilon=sigma_eps,
        sigma_floor_khz=floor,
        clocks_declared=declared,
    )


# ---------------------------------------------------------------------------
# Builder rules
# ---------------------------------------------------------------------------


def test_no_frequency_error_leaves_both_sigmas_undefined():
    """Honest uncertainty: the total is never computed without its statistical
    term (it used to read sigma_stat 0.0 and a total missing that term)."""
    fp = _build([_peak(frequency_error=None)], sigma_eps=1e-7, floor=2.0)
    p = fp.peaks[0]
    assert p.sigma_stat_khz is Absent.UNDEFINED
    assert p.sigma_f_khz is Absent.UNDEFINED
    # The other budget terms are still reported.
    assert p.sigma_floor_khz == 2.0
    assert p.sigma_eps_khz > 0.0


def test_present_frequency_error_keeps_the_three_term_budget():
    fp = _build([_peak()], sigma_eps=1e-7, floor=2.0)
    p = fp.peaks[0]
    assert p.sigma_stat_khz == pytest.approx(0.2)
    assert p.sigma_f_khz == pytest.approx(
        math.sqrt(0.2**2 + p.sigma_eps_khz**2 + 2.0**2)
    )


def test_missing_errors_snr_and_phase_are_undefined():
    p = _build(
        [_peak(amplitude_error=None, phase_error=None, snr=None, phase=None)]
    ).peaks[0]
    assert p.amplitude_error is Absent.UNDEFINED
    assert p.phase_error is Absent.UNDEFINED
    assert p.snr is Absent.UNDEFINED
    assert p.snr_error is Absent.UNDEFINED
    assert p.phase is Absent.UNDEFINED


def test_snr_error_undefined_for_zero_amplitude():
    p = _build([_peak(amplitude=0.0)]).peaks[0]
    assert p.snr == 100.0
    assert p.snr_error is Absent.UNDEFINED


def test_identity_fields_none_is_not_run():
    p = _build([_peak(window_id=None, peak_uid=None, derivation=None)]).peaks[0]
    assert p.window_id is Absent.NOT_RUN
    assert p.peak_uid is Absent.NOT_RUN
    assert p.derivation is Absent.NOT_RUN
    q = _build([_peak(derivation=4)]).peaks[0]
    assert q.derivation == 4 and q.peak_uid == 123 and q.window_id == 3


@pytest.mark.parametrize(
    "declared, stored, expected",
    [
        (False, None, Absent.NOT_RUN),
        (True, None, Absent.UNDEFINED),
        (True, "320x6 (bb)", "320x6 (bb)"),
    ],
)
def test_clock_lattice_rule(declared, stored, expected):
    p = _build([_peak(clock_lattice=stored)], declared=declared).peaks[0]
    assert p.clock_lattice == expected


def test_knockout_rules():
    never, nan_delta, failed, inf_aicc = _build(
        [
            _peak(knockout=None),
            _peak(knockout=_ko(delta_chi2=float("nan"))),
            _peak(knockout=_ko(p_value=float("nan"), aicc_delta=float("nan"))),
            _peak(knockout=_ko(aicc_delta=float("inf"), supported=False)),
        ]
    ).peaks
    for p in (never, nan_delta):
        assert p.knockout_p_value is Absent.NOT_RUN
        assert p.knockout_supported is Absent.NOT_RUN
        assert p.knockout_aicc_delta is Absent.NOT_RUN
    assert failed.knockout_supported is True
    assert failed.knockout_p_value is Absent.UNDEFINED
    assert failed.knockout_aicc_delta is Absent.UNDEFINED
    assert inf_aicc.knockout_supported is False
    assert inf_aicc.knockout_p_value == pytest.approx(1e-3)
    assert inf_aicc.knockout_aicc_delta is Absent.UNDEFINED


def test_wire_form_has_no_bare_null():
    fp = _build([_peak(frequency_error=None, knockout=None, peak_uid=None, snr=None)])
    line = to_jsonable(fp)["peaks"][0]
    assert line["sigma_f_khz"] is None
    assert line["sigma_f_khz_absent"] == "undefined"
    assert line["knockout_p_value_absent"] == "not_run"
    assert line["peak_uid_absent"] == "not_run"
    assert line["snr_error_absent"] == "undefined"
    for key, value in line.items():
        if value is None:
            assert key + "_absent" in line, key


# ---------------------------------------------------------------------------
# Persistence: value + "<field>__status", and the pre-status decoder
# ---------------------------------------------------------------------------


def _roundtrip(products, tmp_path: Path, *, mutate=None, clocks=None):
    out = tmp_path / "review.h5"
    with h5py.File(out, "w") as h5f:
        if clocks is not None:
            spur = h5f.require_group("processing_parameters/stage5_fit/spur")
            spur.attrs["clocks"] = clocks
        save_stage6_review_to_hdf5(
            Stage6Review(final_products=products), h5f.create_group("stage6_review")
        )
    if mutate is not None:
        with h5py.File(out, "r+") as h5f:
            grp = h5f["stage6_review/final_products"]
            raw = json.loads(str(grp.attrs["data"]))
            for peak in raw["peaks"]:
                mutate(peak)
            grp.attrs["data"] = json.dumps(raw)
    with h5py.File(out, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.final_products is not None
    return loaded.final_products.peaks


def test_every_absent_field_roundtrips_with_its_status(tmp_path):
    fp = _build(
        [
            _peak(
                frequency_error=None,
                amplitude_error=None,
                phase=None,
                snr=None,
                window_id=None,
                peak_uid=None,
                knockout=None,
            ),
            _peak(knockout=_ko(p_value=float("nan"))),
            _peak(derivation=2, clock_lattice="320x6 (bb)", knockout=_ko()),
        ],
        declared=True,
    )
    got = _roundtrip(fp, tmp_path)
    for a, b in zip(fp.peaks, got):
        for name, _ in FINAL_PEAK_ABSENT_FIELDS:
            va, vb = getattr(a, name), getattr(b, name)
            if isinstance(va, Absent):
                assert vb is va, name
            else:
                assert vb == va and type(vb) is type(va), name


def test_stored_record_keeps_value_and_status(tmp_path):
    fp = _build([_peak(knockout=None, frequency_error=None)])
    seen = {}
    _roundtrip(fp, tmp_path, mutate=lambda d: seen.update(d))
    assert seen["knockout_p_value"] is None
    assert seen["knockout_p_value__status"] == 1
    assert seen["sigma_f_khz"] is None and seen["sigma_f_khz__status"] == 2
    assert seen["window_id"] == 3 and seen["window_id__status"] == 0


def _strip_status(peak: dict) -> None:
    """Make a current record look like one written before the status keys."""
    for name, _ in FINAL_PEAK_ABSENT_FIELDS:
        peak.pop(name + "__status", None)


def test_pre_status_record_decodes_field_by_field(tmp_path):
    fp = _build(
        [
            _peak(knockout=None, peak_uid=None, snr=None, phase_error=None),
            _peak(derivation=1, knockout=_ko(p_value=float("nan"))),
        ]
    )

    def legacy(peak: dict) -> None:
        _strip_status(peak)
        # The old writer stored None for absence and nan for a failed refit.
        if peak["knockout_supported"] is not None:
            peak["knockout_p_value"] = float("nan")

    no_ko, failed = _roundtrip(fp, tmp_path, mutate=legacy)
    assert no_ko.knockout_p_value is Absent.NOT_RUN
    assert no_ko.knockout_supported is Absent.NOT_RUN
    assert no_ko.knockout_aicc_delta is Absent.NOT_RUN
    assert no_ko.peak_uid is Absent.NOT_RUN
    assert no_ko.derivation is Absent.NOT_RUN
    assert no_ko.snr is Absent.UNDEFINED
    assert no_ko.snr_error is Absent.UNDEFINED
    assert no_ko.phase_error is Absent.UNDEFINED
    assert no_ko.sigma_stat_khz == pytest.approx(0.2)
    assert failed.knockout_supported is True
    assert failed.knockout_p_value is Absent.UNDEFINED
    assert failed.knockout_aicc_delta == pytest.approx(5.0)
    assert failed.derivation == 1


def test_pre_status_zero_sigma_stat_is_the_old_sentinel(tmp_path):
    """The old writer stored sigma_stat 0.0 (and a total missing that term)
    for a line with no frequency error; it reads as UNDEFINED, both sigmas."""
    fp = _build([_peak()], floor=2.0)

    def legacy(peak: dict) -> None:
        _strip_status(peak)
        peak["sigma_stat_khz"] = 0.0
        peak["sigma_f_khz"] = 2.0

    (p,) = _roundtrip(fp, tmp_path, mutate=legacy)
    assert p.sigma_stat_khz is Absent.UNDEFINED
    assert p.sigma_f_khz is Absent.UNDEFINED


@pytest.mark.parametrize(
    "clocks, expected",
    [
        (None, Absent.NOT_RUN),
        ("[]", Absent.NOT_RUN),
        ('[{"freq_mhz": 50000.0, "locked": false, "label": "dig"}]', Absent.UNDEFINED),
    ],
)
def test_pre_status_clock_lattice_reads_the_fit_declaration(tmp_path, clocks, expected):
    fp = _build([_peak(clock_lattice=None)])
    (p,) = _roundtrip(fp, tmp_path, mutate=_strip_status, clocks=clocks)
    assert p.clock_lattice is expected


def test_fit_declares_clocks(tmp_path):
    out = tmp_path / "f.h5"
    with h5py.File(out, "w") as h5f:
        assert fit_declares_clocks(h5f) is False
        spur = h5f.require_group("processing_parameters/stage5_fit/spur")
        spur.attrs["clocks"] = "__None__"
        assert fit_declares_clocks(h5f) is False
        spur.attrs["clocks"] = '[{"freq_mhz": 10.0, "locked": true, "label": "x"}]'
        assert fit_declares_clocks(h5f) is True


# ---------------------------------------------------------------------------
# CalibrationStamp, curation results, typed errors
# ---------------------------------------------------------------------------


def test_calibration_stamp_without_fid_header_is_not_run(tmp_path):
    path = tmp_path / "bare.ftmw"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["x"] = 1
    stamp = frequency_calibration_impl(path)
    assert stamp.probe_freq_mhz is Absent.NOT_RUN
    assert stamp.sideband is Absent.NOT_RUN
    wire = to_jsonable(stamp)
    assert wire["probe_freq_mhz"] is None
    assert wire["probe_freq_mhz_absent"] == "not_run"
    assert wire["sideband_absent"] == "not_run"


def test_preview_window_defaults_are_not_run():
    w = PreviewWindowResult(window_id=1, origin="direct")
    for name in (
        "chi2r_before",
        "chi2r_after",
        "converged",
        "created_window_mode",
        "created_window_freq_range",
        "created_window_n_points",
        "created_window_n_contributors",
        "created_window_depends_on",
    ):
        assert getattr(w, name) is Absent.NOT_RUN, name


def test_review_cli_chi2r_rendering():
    assert _fmt_chi2r(Absent.NOT_RUN, ".3f") == "-"
    assert _fmt_chi2r(Absent.UNDEFINED, ".3f") == "undefined"
    assert _fmt_chi2r(1.23456, ".3f") == "1.235"


def test_stage_dependency_error_without_command_is_not_run():
    e = StageDependencyError("s", ["stage3_peaks"], Path("x.ftmw"))
    assert e.command is None  # the attribute keeps None
    d = json.loads(json.dumps(e.to_dict(), allow_nan=False))
    assert d["command"] is None
    assert d["command_absent"] == "not_run"
    named = StageDependencyError("s", ["stage3_peaks"], Path("x"), command="peaks run")
    assert named.to_dict()["command"] == "peaks run"
    assert "command_absent" not in named.to_dict()


@pytest.mark.integration
def test_refit_context_carries_the_fit_clock_lattice(stage5_small_source):
    """A Stage 6 refit annotates its lines against the fit's clock lattice, so
    its replayed context must carry the lattice whenever the fit declared
    clocks. Mutation: building the lattice only on the detection path (refit
    lines would read clock_lattice UNDEFINED -- off-lattice -- untested)."""
    from ftmwpipeline._internal.stage6_impl import _build_shared_fit_ctx
    from ftmwpipeline.io.stage6_review_serialization import (
        fit_declares_clocks as _declares,
    )

    path = str(stage5_small_source)
    with h5py.File(path, "r") as h5f:
        declared = _declares(h5f)
    lattice = _build_shared_fit_ctx(path).fit_ctx.clock_lattice
    assert (lattice is not None) == declared
