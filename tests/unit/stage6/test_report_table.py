"""Unit + cross-interface tests for the Stage 6 ``report table`` (Level 1).

Serializer unit tests build a :class:`FinalProducts` directly and persist it to
a tmp file (no fixture fit needed); the cross-interface test runs the real
``review run`` consolidation on the small 2638 fixture and checks api ==
Pipeline == impl byte-for-byte.
"""

from __future__ import annotations

import csv as csvmod
import io
import json
import shutil
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_impl import _amplitude_unit, report_table_impl
from ftmwpipeline.core.data_structures import (
    FinalPeak,
    FinalProducts,
    Stage6Review,
)
from ftmwpipeline.io.stage6_review_serialization import save_stage6_review_to_hdf5
from ftmwpipeline.pipeline import Pipeline


def _products() -> FinalProducts:
    return FinalProducts(
        peaks=[
            FinalPeak(
                frequency_mhz=30000.021920,
                frequency_raw_mhz=30000.000000,
                f_baseband_mhz=10960.0,
                sigma_f_khz=1.234,
                sigma_stat_khz=0.200,
                sigma_eps_khz=1.096,
                sigma_floor_khz=0.500,
                amplitude=0.0123,
                phase=0.45,
                snr=100.0,
                origin="auto",
                window_id=3,
                amplitude_error=0.0005,
                phase_error=0.02,
                snr_error=4.0,
                clock_lattice="320x6 (bb)",
            ),
            FinalPeak(
                frequency_mhz=39000.019800,
                frequency_raw_mhz=39000.000000,
                f_baseband_mhz=1960.0,
                sigma_f_khz=0.530,
                sigma_stat_khz=0.100,
                sigma_eps_khz=0.196,
                sigma_floor_khz=0.500,
                amplitude=0.02,
                phase=None,
                snr=None,
                origin="user",
                window_id=None,
                amplitude_error=None,
                phase_error=None,
                snr_error=None,
            ),
        ],
        calibration_state="self_calibrated",
        epsilon=2.2e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=0.5,
        probe_freq_mhz=40960.0,
        sideband="lower",
    )


def _write_products_file(tmp_path) -> Path:
    fp = tmp_path / "exp.ftmw"
    review = Stage6Review(final_products=_products())
    with h5py.File(fp, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(review, grp)
    return fp


def test_unknown_format_raises(tmp_path):
    fp = _write_products_file(tmp_path)
    with pytest.raises(ValueError):
        report_table_impl(str(fp), fmt="xml")


def test_missing_products_raises(tmp_path):
    fp = tmp_path / "empty.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with pytest.raises(ValueError, match="review run"):
        report_table_impl(str(fp), fmt="csv")


def test_csv_structure(tmp_path):
    fp = _write_products_file(tmp_path)
    text = report_table_impl(str(fp), fmt="csv")

    # Provenance header is commented and carries the calibration state + unit.
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("calibration_state: self_calibrated" in ln for ln in header)
    assert any("epsilon_ppm: +2.200 +- 0.100" in ln for ln in header)
    assert any("sigma_floor_khz: 0.500" in ln for ln in header)
    assert any("amplitude_unit: V" in ln for ln in header)
    assert any("n_peaks: 2" in ln for ln in header)

    # Data rows parse (skipping comment lines) and carry the canonical columns.
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    rows = list(csvmod.DictReader(io.StringIO(body)))
    assert len(rows) == 2
    assert rows[0]["frequency_mhz"] == "30000.021920"
    assert rows[0]["origin"] == "auto"
    assert rows[0]["window_id"] == "3"
    # Errors are present (amplitudes already O(0.01) -> unit V, unscaled).
    assert rows[0]["amplitude"] == "0.0123"
    assert rows[0]["amplitude_err"] == "0.0005"
    assert rows[0]["phase_err_rad"] == "0.02"
    assert rows[0]["snr_err"] == "4"
    # None phase/snr/errors/window_id render empty.
    assert rows[1]["phase_rad"] == ""
    assert rows[1]["snr"] == ""
    assert rows[1]["amplitude_err"] == ""
    assert rows[1]["snr_err"] == ""
    assert rows[1]["window_id"] == ""
    # The clock-lattice annotation: identity on-lattice, empty off-lattice.
    assert rows[0]["clock_lattice"] == "320x6 (bb)"
    assert rows[1]["clock_lattice"] == ""


def test_json_structure(tmp_path):
    fp = _write_products_file(tmp_path)
    payload = json.loads(report_table_impl(str(fp), fmt="json"))
    assert payload["metadata"]["calibration_state"] == "self_calibrated"
    assert payload["metadata"]["n_peaks"] == 2
    assert payload["metadata"]["epsilon"] == pytest.approx(2.2e-6)
    assert payload["metadata"]["amplitude_unit"] == "V"
    assert len(payload["peaks"]) == 2
    p0 = payload["peaks"][0]
    assert p0["frequency_mhz"] == pytest.approx(30000.02192)
    assert p0["phase_rad"] == pytest.approx(0.45)
    assert "phase" not in p0  # renamed to phase_rad
    assert p0["amplitude_error"] == pytest.approx(0.0005)
    assert p0["snr_error"] == pytest.approx(4.0)
    assert payload["peaks"][1]["phase_rad"] is None
    assert payload["peaks"][1]["amplitude_error"] is None
    assert p0["clock_lattice"] == "320x6 (bb)"
    assert payload["peaks"][1]["clock_lattice"] is None


def test_latex_structure(tmp_path):
    fp = _write_products_file(tmp_path)
    text = report_table_impl(str(fp), fmt="latex")
    assert r"\begin{tabular}" in text
    assert r"\toprule" in text and r"\bottomrule" in text
    assert "booktabs" in text
    # self_calibrated -> caption states the correction + epsilon.
    assert "timebase scale error" in text
    assert "+2.200" in text
    # Two data rows (lines ending with the LaTeX row terminator inside tabular).
    data_rows = [ln for ln in text.splitlines() if ln.strip().endswith(r"\\")]
    # header row + 2 data rows all end with \\
    assert len(data_rows) == 3


def test_output_writes_file(tmp_path):
    fp = _write_products_file(tmp_path)
    out = tmp_path / "out.csv"
    text = report_table_impl(str(fp), fmt="csv", output=str(out))
    assert out.read_text() == text


# ---------------------------------------------------------------------------
# Robustness: catastrophic values + dynamic amplitude units
# ---------------------------------------------------------------------------


def _mk_peak(freq=30000.0, amp=1.0e-6, **kw) -> FinalPeak:
    base = dict(
        frequency_mhz=freq,
        frequency_raw_mhz=freq,
        f_baseband_mhz=abs(40960.0 - freq),
        sigma_f_khz=1.0,
        sigma_stat_khz=1.0,
        sigma_eps_khz=0.0,
        sigma_floor_khz=0.0,
        amplitude=amp,
        phase=0.1,
        snr=50.0,
        origin="auto",
        window_id=1,
        amplitude_error=amp * 0.02,
        phase_error=0.01,
        snr_error=1.0,
    )
    base.update(kw)
    return FinalPeak(**base)


def _write(products: FinalProducts, tmp_path) -> Path:
    fp = tmp_path / "exp.ftmw"
    with h5py.File(fp, "w") as h5f:
        save_stage6_review_to_hdf5(
            Stage6Review(final_products=products), h5f.create_group("stage6_review")
        )
    return fp


def test_catastrophic_values_bounded(tmp_path):
    """A degenerate fit (runaway sigma, inf error, denormal amplitude) must not
    dump a hundred-digit field, and JSON must stay strict-valid."""
    products = FinalProducts(
        peaks=[
            _mk_peak(
                freq=38992.894750,
                amp=1.7e-109,
                sigma_f_khz=9.5e105,
                sigma_stat_khz=9.5e105,
                snr=0.0,
                amplitude_error=float("inf"),
                snr_error=None,
            ),
            _mk_peak(freq=30000.0, amp=1.0e-6),
        ],
        calibration_state="rb_locked",
        sideband="lower",
        probe_freq_mhz=40960.0,
    )
    fp = _write(products, tmp_path)

    csv = report_table_impl(str(fp), fmt="csv")
    for line in csv.splitlines():
        if line.startswith("#"):
            continue
        for field in line.split(","):
            assert len(field) <= 20, f"runaway field: {field!r}"
    assert "e+" in csv  # the blown-up sigma rendered in scientific
    assert "inf" in csv  # the inf amplitude error

    # JSON must parse strictly (non-finite -> null, never NaN/Infinity tokens).
    payload = json.loads(report_table_impl(str(fp), fmt="json"))
    assert payload["peaks"][0]["amplitude_error"] is None  # inf -> null

    tex = report_table_impl(str(fp), fmt="latex")  # must not crash
    assert r"\begin{tabular}" in tex
    for line in tex.splitlines():
        assert len(line) <= 200


def test_amplitude_unit_selection_outlier_robust():
    products = FinalProducts(
        peaks=[
            _mk_peak(amp=3.0e-7),
            _mk_peak(amp=1.5e-6),
            _mk_peak(amp=1.0e-100),  # phantom -- must not drag the unit to fV
        ],
    )
    name, val = _amplitude_unit(products)
    assert name == "uV"
    assert val == pytest.approx(1e-6)


def test_amplitude_unit_in_csv_and_scaled(tmp_path):
    products = FinalProducts(peaks=[_mk_peak(amp=3.0e-7), _mk_peak(amp=1.5e-6)])
    fp = _write(products, tmp_path)
    text = report_table_impl(str(fp), fmt="csv")
    assert any("amplitude_unit: uV" in ln for ln in text.splitlines())
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    rows = list(csvmod.DictReader(io.StringIO(body)))
    # 3.0e-7 V -> 0.3 uV ; 1.5e-6 V -> 1.5 uV
    assert rows[0]["amplitude"] == "0.3"
    assert rows[1]["amplitude"] == "1.5"


# ---------------------------------------------------------------------------
# Catalog cross-reference (--catalog)
# ---------------------------------------------------------------------------


def _catalog_file(tmp_path) -> Path:
    # First peak at 30000.021920 (sigma_f 1.234 kHz) -- catalog entry 0.5 kHz
    # away matches at N=3; the 39000 line has no nearby entry.
    f = tmp_path / "cat.csv"
    f.write_text(
        "frequency_mhz,uncertainty_khz,label\n"
        "30000.0214,0.0,line-A\n"
        "35000.0,0.0,line-Z\n"
    )
    return f


def test_csv_catalog_columns(tmp_path):
    fp = _write_products_file(tmp_path)
    cat = _catalog_file(tmp_path)
    text = report_table_impl(str(fp), fmt="csv", catalog=str(cat))

    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("catalog: cat.csv" in ln for ln in header)
    assert any("catalog_matched: 1/2 (50%)" in ln for ln in header)

    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    rows = list(csvmod.DictReader(io.StringIO(body)))
    assert rows[0]["catalog_label"] == "line-A"
    assert float(rows[0]["catalog_freq_mhz"]) == pytest.approx(30000.0214)
    assert rows[0]["catalog_delta_khz"] != ""
    # Second line has no match -> empty catalog cells.
    assert rows[1]["catalog_label"] == ""
    assert rows[1]["catalog_pull"] == ""


def test_csv_no_catalog_omits_columns(tmp_path):
    fp = _write_products_file(tmp_path)
    text = report_table_impl(str(fp), fmt="csv")
    assert "catalog_label" not in text


def test_json_catalog_block(tmp_path):
    fp = _write_products_file(tmp_path)
    cat = _catalog_file(tmp_path)
    payload = json.loads(report_table_impl(str(fp), fmt="json", catalog=str(cat)))
    meta = payload["metadata"]["catalog"]
    assert meta["n_matched"] == 1 and meta["n_total"] == 2
    assert meta["n_sigma"] == 3.0
    assert payload["peaks"][0]["catalog"]["label"] == "line-A"
    assert payload["peaks"][1]["catalog"] is None  # key present, no match


def test_latex_catalog_column(tmp_path):
    fp = _write_products_file(tmp_path)
    cat = _catalog_file(tmp_path)
    text = report_table_impl(str(fp), fmt="latex", catalog=str(cat))
    assert "Catalog" in text
    assert "line-A" in text
    assert "not an assignment" in text  # caption disclaimer


def test_latex_label_escaped(tmp_path):
    fp = _write_products_file(tmp_path)
    f = tmp_path / "cat.csv"
    f.write_text("frequency_mhz,uncertainty_khz,label\n30000.0214,0.0,a_b&c%\n")
    text = report_table_impl(str(fp), fmt="latex", catalog=str(f))
    assert r"a\_b\&c\%" in text


def test_catalog_nsigma_tightens_match(tmp_path):
    fp = _write_products_file(tmp_path)
    # 30000.0214 is ~0.4 kHz from the 30000.02192 line; sigma_f 1.234 kHz.
    # N=0.1 -> tol 0.12 kHz -> no match.
    cat = _catalog_file(tmp_path)
    payload = json.loads(
        report_table_impl(str(fp), fmt="json", catalog=str(cat), catalog_n_sigma=0.1)
    )
    assert payload["metadata"]["catalog"]["n_matched"] == 0


def test_catalog_missing_file_raises(tmp_path):
    fp = _write_products_file(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        report_table_impl(str(fp), fmt="csv", catalog=str(tmp_path / "nope.csv"))


# ---------------------------------------------------------------------------
# Cross-interface on the small 2638 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def stage5_small_file(stage5_reviewed_file):
    """Tests here need the post-review file; alias the shared reviewed build."""
    return stage5_reviewed_file


@pytest.mark.integration
@pytest.mark.parametrize("fmt", ["csv", "json", "latex"])
def test_report_table_cross_interface(stage5_small_file, tmp_path, fmt):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    via_impl = report_table_impl(str(fp), fmt=fmt)
    via_api = ftmw.report_table(str(fp), fmt=fmt)
    via_pipe = Pipeline.open(fp).report_table(fmt=fmt)

    assert via_impl == via_api == via_pipe
    assert len(via_impl) > 0
