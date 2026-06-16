"""Unit + cross-interface tests for the Stage 6 ``report summary`` (Level 2).

The Markdown renderer is tested against a hand-built :class:`_SummaryModel` (no
fixture fit needed); the missing-products error path uses a minimal file; and a
cross-interface test runs the real ``review run`` consolidation on the small
2638 fixture and checks api == Pipeline == impl byte-for-byte.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_impl import (
    _BandTau,
    _render_markdown,
    _SummaryModel,
    report_summary_impl,
)
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import review_run_impl
from ftmwpipeline.core.data_structures import FinalPeak, FinalProducts
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Hand-built model for renderer unit tests
# ---------------------------------------------------------------------------


def _peak(freq, snr, amp=1.0e-6, **kw) -> FinalPeak:
    base = dict(
        frequency_mhz=freq,
        frequency_raw_mhz=freq,
        f_baseband_mhz=abs(40960.0 - freq),
        sigma_f_khz=1.0,
        sigma_stat_khz=0.5,
        sigma_eps_khz=0.8,
        sigma_floor_khz=0.0,
        amplitude=amp,
        phase=0.1,
        snr=snr,
        origin="auto",
        window_id=7,
        amplitude_error=amp * 0.02,
        phase_error=0.01,
        snr_error=1.0,
    )
    base.update(kw)
    return FinalPeak(**base)


def _products(state="self_calibrated", n=3) -> FinalProducts:
    peaks = [_peak(30000.0 + 100.0 * i, snr=10.0 * (i + 1)) for i in range(n)]
    return FinalProducts(
        peaks=peaks,
        calibration_state=state,
        epsilon=2.2e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=0.0,
        probe_freq_mhz=40960.0,
        sideband="lower",
    )


def _model(**over) -> _SummaryModel:
    base = dict(
        experiment="exp_demo",
        source_path="examples/blackchirp_data/2638",
        source_format="blackchirp",
        probe_freq_mhz=40960.0,
        sideband="lower",
        start_us=2.35,
        end_us=15.0,
        duration_us=15.0,
        n_points=750000,
        shots=501400,
        band_lo_mhz=26500.0,
        band_hi_mhz=40000.0,
        ft_bin_khz=66.6667,
        ft_n_bins=202500,
        noise_median=2.4e-7,
        noise_min=1.5e-7,
        noise_max=3.9e-7,
        tau_maj_us=5.96,
        sigma_tau_us=1.51,
        n_contributors=4417,
        tau_source=None,
        band_taus=[
            _BandTau("low", 26500.0, 31000.0, 792, 7.62, 1.18),
            _BandTau("mid", 31000.0, 35500.0, 1459, 6.16, 1.26),
            _BandTau("high", 35500.0, 40000.0, 2166, 5.29, 1.13),
        ],
        n_peaks_total=4635,
        n_strong=142,
        n_medium=144,
        n_weak=4349,
        promotion_min_snr=3.0,
        weak_medium_snr=10.0,
        medium_strong_snr=50.0,
        n_windows_planned=369,
        shape="lorentzian",
        n_windows_fit=228,
        n_fitted_peaks=455,
        chi2_median=1.43,
        chi2_min=0.63,
        chi2_max=98.5,
        n_thaw=14,
        n_thaw_accepted=0,
        n_replan=3,
        n_rescue_rounds=439,
        products=_products(),
    )
    base.update(over)
    return _SummaryModel(**base)


def test_markdown_structure_and_methods_prose():
    md = _render_markdown(_model(), "exp_demo.ftmw", include_table=False)

    # Title + the seven stage method sections are present.
    assert md.startswith("# FTMW pipeline report -- exp_demo")
    for heading in (
        "## Summary",
        "### Stage 0 -- start-time detection",
        "### Stage 1 -- Fourier transform",
        "### Stage 2 -- noise estimation",
        "### Stage 2b -- τ calibration",
        "### Stage 3 -- peak detection",
        "### Stage 4 -- window assignment",
        "### Stage 5 -- per-window fitting",
        "### Stage 6 -- calibration and finalization",
        "## Strongest lines",
        "## Final line list",
    ):
        assert heading in md, heading

    # Static, code-versioned prose (not pulled from dev docs) is present.
    assert "unapodized, un-windowed, and native-length" in md
    assert "single noise authority" in md
    assert "three-term precision budget" in md

    # Per-experiment results numbers are interleaved.
    assert "start = 2.35 µs" in md
    assert "750,000 points" in md
    assert "26500--40000 MHz" in md
    assert "τ_maj = 5.96" in md
    assert "4,635 candidate peaks" in md
    assert "369 disjoint windows" in md
    assert "228 windows fit" in md
    assert "χ²ᵣ median 1.43" in md
    assert "ε = +2.200 ± 0.100 ppm" in md


def test_summary_calibration_state_phrasing():
    md_self = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "ε-corrected for the free-running digitizer timebase" in md_self

    md_rb = _render_markdown(
        _model(products=_products(state="rb_locked")), "x.ftmw", include_table=False
    )
    assert "Rb-locked absolute scale" in md_rb

    md_unc = _render_markdown(
        _model(products=_products(state="uncalibrated")),
        "x.ftmw",
        include_table=False,
    )
    assert "uncalibrated" in md_unc


def test_strongest_lines_sorted_and_capped():
    # 15 peaks with increasing SNR -> table shows the top 10 by SNR, descending.
    prod = FinalProducts(
        peaks=[_peak(30000.0 + i, snr=float(i + 1)) for i in range(15)],
        calibration_state="self_calibrated",
        epsilon=0.0,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
        probe_freq_mhz=40960.0,
        sideband="lower",
    )
    md = _render_markdown(_model(products=prod), "x.ftmw", include_table=False)
    table = md.split("## Strongest lines")[1].split("## Final line list")[0]
    data_rows = [ln for ln in table.splitlines() if ln.startswith("| 3")]
    assert len(data_rows) == 10  # capped at 10
    # First data row is the highest SNR (15), last is SNR 6.
    snr_col = [ln.split("|")[4].strip() for ln in data_rows]
    assert snr_col[0] == "15" and snr_col[-1] == "6"


def test_include_table_inlines_full_list():
    prod = _products(n=5)
    md_ptr = _render_markdown(_model(products=prod), "x.ftmw", include_table=False)
    md_full = _render_markdown(_model(products=prod), "x.ftmw", include_table=True)

    # Pointer form references the companion export; inlined form does not.
    assert "companion data export" in md_ptr
    assert "report table --format csv" in md_ptr

    full_section = md_full.split("## Final line list")[1]
    assert "companion data export" not in full_section
    data_rows = [ln for ln in full_section.splitlines() if ln.startswith("| 3")]
    assert len(data_rows) == 5  # all five lines inlined


def test_tau_absent_renders_fallback_note():
    md = _render_markdown(
        _model(tau_maj_us=None, sigma_tau_us=None, n_contributors=None, band_taus=[]),
        "x.ftmw",
        include_table=False,
    )
    assert "### Stage 2b -- τ calibration" in md
    assert "not run" in md
    assert "τ_maj =" not in md


def test_missing_products_raises(tmp_path):
    fp = tmp_path / "empty.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with pytest.raises(ValueError, match="review run"):
        report_summary_impl(str(fp))


# ---------------------------------------------------------------------------
# Cross-interface on the small 2638 fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage5_small_file(tmp_path_factory):
    data_path = Path("examples/blackchirp_data/2638")
    if not data_path.exists():
        pytest.skip("Experiment 2638 data not available")
    tmp = tmp_path_factory.mktemp("stage5_small_summary")
    fp = tmp / "2638_summary.ftmw"

    ftmw.import_data(fp, source=str(data_path))
    ftmw.compute_ft(fp, trim=(26500, 40000))
    ftmw.estimate_noise(fp)
    ftmw.calibrate_tau(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)

    plan = load_windows_impl(str(fp))["plan"]
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= 3:
            break
    if not candidates:
        candidates = list(plan.topological_order[:3])
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(fp), plan)

    ftmw.fit_peaks(str(fp))
    review_run_impl(str(fp))
    return fp


@pytest.mark.integration
def test_report_summary_cross_interface(stage5_small_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    via_impl = report_summary_impl(str(fp))
    via_api = ftmw.report_summary(str(fp))
    via_pipe = Pipeline.open(fp).report_summary()

    assert via_impl == via_api == via_pipe
    # Real numbers from the persisted stages made it into the document.
    assert "# FTMW pipeline report" in via_impl
    assert "### Stage 5 -- per-window fitting" in via_impl
    assert "## Strongest lines" in via_impl
    assert "τ_maj =" in via_impl  # Stage 2b was run on this fixture


@pytest.mark.integration
def test_report_summary_output_writes_file(stage5_small_file, tmp_path):
    out = tmp_path / "summary.md"
    text = report_summary_impl(str(stage5_small_file), output=str(out))
    assert out.read_text() == text
    assert text.startswith("# FTMW pipeline report")
