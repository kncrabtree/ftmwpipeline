"""Unit + cross-interface tests for the Stage 6 ``report summary`` (Level 2).

The Markdown renderer is tested against a hand-built :class:`_SummaryModel`
(no fixture fit needed) -- covering the per-stage parameter tables, equations,
detailed result tables, and the automatically flagged concerns. The
missing-products error path uses a minimal file; a cross-interface test runs the
real ``review run`` consolidation on the small 2638 fixture and checks api ==
Pipeline == impl byte-for-byte.
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
    _NoiseBand,
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


def _products(state="self_calibrated", n=3, sigma_floor_khz=0.0) -> FinalProducts:
    peaks = [_peak(30000.0 + 100.0 * i, snr=10.0 * (i + 1)) for i in range(n)]
    return FinalProducts(
        peaks=peaks,
        calibration_state=state,
        epsilon=2.2e-6,
        sigma_epsilon=0.1e-6,
        sigma_floor_khz=sigma_floor_khz,
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
        shape="gaussian",
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
        # enriched detail
        noise_params={
            "window_mhz": 80.0,
            "pedestal_mhz": 20.0,
            "line_k": 8.0,
            "n_iter": 3,
            "smoothing_mhz": 800.0,
            "convolve_mhz": 200.0,
            "region_aware": True,
        },
        noise_fraction_overall=0.985,
        noise_bands=[
            _NoiseBand(26500.0, 33250.0, 2.0e-7, 0.991, 21347),
            _NoiseBand(33250.0, 40000.0, 3.1e-7, 0.978, 21347),
        ],
        tau_params={"n_seg": 10, "snr_weighted": True, "n_spur_bins": 649},
        tau_preconditions_passed=False,
        tau_preconditions_notes=[
            "ok",
            "strongly bimodal (delta_aic=598.0) and dominant cluster weight "
            "0.59 < 0.70",
        ],
        tau_bimodal=True,
        tau_delta_aic=598.0,
        tau_dominant_weight=0.59,
        tau_mu_a=5.5,
        tau_mu_b=7.6,
        tau_pearson_freq=-0.43,
        recommended_shape="gaussian",
        det_params={
            "promotion_min_snr": 3.0,
            "internal_min_snr": 2.0,
            "weak_medium_snr": 10.0,
            "medium_strong_snr": 50.0,
            "primary_window": "blackmanharris",
            "gap_shape": "lorentzian",
            "gap_leakage_floor_k": 3.0,
            "detection_zpf": 2,
            "tau_basis_us": 5.958,
        },
        det_pass_counts={"primary": 4526, "gap": 109},
        window_params={
            "edge_m": 64,
            "trim_m": 32,
            "edge_threshold": 8.0,
            "max_window_width_mhz": 40.0,
            "max_window_width_points": 96,
            "min_window_half_width_points": 32,
            "min_freeze_snr": 50.0,
        },
        width_min_mhz=3.6,
        width_median_mhz=5.1,
        width_max_mhz=13.0,
        ppw_median=1.0,
        ppw_max=12,
        n_hard_windows=4,
        fit_params={
            "shape": "gaussian",
            "tau0_us": 4.217,
            "tau_calibration_source": "none",
            "fit_tau": True,
            "max_decay_factor": 5.0,
            "baseline_order": 4,
            "rescue_snr_threshold": 2.5,
            "max_residual_rescue_rounds": 5,
            "spur_masking_enabled": True,
        },
        tau0_us=4.217,
        tau_calibration_source="none",
        tau_exp_present=True,
        tau_G_present=False,
        n_promoted=747,
        snr_pctiles={
            "p10": 1.28,
            "p25": 1.6,
            "p50": 1.97,
            "p75": 2.5,
            "p90": 4.35,
            "max": 395.0,
        },
        snr_pctiles_promoted={
            "p10": 3.19,
            "p25": 3.67,
            "p50": 5.65,
            "p75": 33.0,
            "p90": 78.9,
            "max": 395.0,
        },
        density_chunks=[
            (26500.0, 33250.0, 2342, 372),
            (33250.0, 40000.0, 2293, 375),
        ],
        width_pctiles={
            "p10": 5.06,
            "p25": 5.06,
            "p50": 5.06,
            "p75": 7.35,
            "p90": 9.57,
            "max": 12.6,
        },
        ppw_pctiles={
            "p10": 1.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 3.0,
            "p90": 4.0,
            "max": 12.0,
        },
        n_nonconverged=0,
        n_spurs_gated=9,
        n_empty_dropped=98,
        n_pruned=72,
        n_baseline_windows=37,
        n_windows_gate_checked=228,
        n_windows_fail_gate=0,
        worst_windows=[],
        chi2r_pctiles={
            "p10": 0.94,
            "p25": 1.11,
            "p50": 1.37,
            "p75": 1.8,
            "p90": 2.91,
            "max": 34.7,
        },
        eps_pctiles={
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.04,
            "max": 2.13,
        },
        sigma_stat_pctiles={
            "p10": 0.45,
            "p25": 0.81,
            "p50": 3.62,
            "p75": 8.6,
            "p90": 11.0,
            "max": 26.2,
        },
        sigma_eps_pctiles={
            "p10": 0.2,
            "p25": 0.39,
            "p50": 0.6,
            "p75": 0.84,
            "p90": 0.95,
            "max": 1.05,
        },
        sigma_f_pctiles={
            "p10": 0.74,
            "p25": 1.08,
            "p50": 3.66,
            "p75": 8.62,
            "p90": 11.0,
            "max": 26.2,
        },
        snr_bin_rows=[
            ("<100", 245, 1.3, 1.95, 5.41, 1.0, 0.0),
            ("100-1k", 31, 4.14, 9.68, 34.7, 1.0, 0.67),
            ("1k-10k", 1, 8.62, 8.62, 8.62, 1.0, 0.096),
        ],
        timebase_n_used=11,
        timebase_lattice_g=640.0,
        median_sigma_stat=2.16,
        median_sigma_eps=0.58,
        median_sigma_f=2.26,
        n_stat_dominated=376,
        n_eps_dominated=79,
    )
    base.update(over)
    return _SummaryModel(**base)


def test_markdown_structure_and_methods_prose():
    md = _render_markdown(_model(), "exp_demo.ftmw", include_table=False)

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


def test_parameter_tables_present():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    # Every enriched stage emits a Parameters table with its real knobs.
    assert md.count("**Parameters**") >= 4
    assert "| `window_mhz` | 80 | Local scatter-MAD window width |" in md
    assert "| `promotion_min_snr` | 3 | Min SNR to promote to a detection |" in md
    assert "`max_window_width_mhz`" in md
    assert "`tau_calibration_source` | none |" in md


def test_equations_present():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert r"\sigma_{x,k}" in md  # Stage 2 SNR equation
    assert r"\Delta\nu_{\mathrm{FWHM}}" in md  # Stage 2b linewidth equation
    assert r"\kappa\,\mathrm{SNR}_{\max}" in md  # Stage 5 SNR-aware gate
    assert r"f_{\mathrm{mol}}" in md  # Stage 6 epsilon correction


def test_per_band_noise_table():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "**Per-band noise**" in md
    assert "| Band (MHz) | Median σ_x | Noise fraction | Bins |" in md
    assert "| 26500--33250 |" in md
    assert "99.1%" in md  # band noise fraction rendered as a percentage
    assert "98.5% of bins are noise" in md  # overall


def test_snr_aware_gate_and_budget_detail():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "SNR-aware quality gate: 228/228 windows pass" in md
    assert "9 spurs gated" in md
    assert "σ_f budget (median): σ_stat = 2.16 kHz, σ_ε = 0.58 kHz" in md
    assert "376 lines are precision-dominated" in md
    assert "ε from 11 clock tone(s)" in md


def test_concern_tau_source_fallback_gaussian_no_twin():
    # Base model: gaussian fit, exp tau twin present but no tau_G twin, source
    # 'none' -> the real diagnosis is a Stage 5 warning about the missing twin.
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "Stage 5 fit a **gaussian** shape but no τ_G calibration" in md
    assert "fell back to T_active/3 = 4.22 µs" in md
    assert "the measured τ_maj is 5.96 µs" in md
    assert "Run `calibrate_tau_G`" in md
    # The Stage 2b precondition failure is re-framed as a *note*, with no false
    # claim that it caused the Stage 5 fallback.
    assert "Stage 2b preconditions did not pass" in md
    assert "the τ majority is not trustworthy, so Stage 5 fell back" not in md
    # ... plus the sigma_floor=0 note from Stage 6.
    assert "σ_floor = 0:" in md
    # Roll-up: 1 warning (tau_G), 2 notes (preconditions, sigma_floor).
    assert "**Concerns flagged:** 1 warning(s), 2 note(s)" in md


def test_concern_tau_source_fallback_lorentzian_no_calibration():
    m = _model(
        shape="lorentzian",
        fit_params={"shape": "lorentzian", "tau0_us": 4.217},
        tau_calibration_source="none",
        tau_exp_present=False,
        tau_G_present=False,
        tau_maj_us=None,
    )
    md = _render_markdown(m, "x.ftmw", include_table=False)
    assert "no Stage 2b τ calibration" in md
    assert "Run `calibrate_tau` before `fit_peaks`" in md


def test_no_tau_concern_when_calibration_consumed():
    m = _model(tau_calibration_source="persisted", tau0_us=5.96, tau_G_present=True)
    md = _render_markdown(m, "x.ftmw", include_table=False)
    assert "fell back to T_active/3" not in md
    assert "no τ_G calibration" not in md


def test_concern_uncalibrated_and_gate_failures():
    m = _model(
        products=_products(state="uncalibrated"),
        tau_preconditions_passed=True,
        tau_preconditions_notes=[],
        tau_calibration_source="persisted",
        tau0_us=5.96,
        n_windows_fail_gate=2,
        n_windows_gate_checked=228,
        worst_windows=[(281, 98.5, 3.0, 3.3), (44, 40.0, 4.0, 1.5)],
        n_nonconverged=1,
    )
    md = _render_markdown(m, "x.ftmw", include_table=False)
    assert "Frequencies are uncalibrated" in md
    assert "fail the SNR-aware χ² gate" in md
    assert "Worst: windows 281, 44" in md
    assert "1 window(s) did not converge" in md


def test_no_concerns_when_clean():
    m = _model(
        products=_products(state="rb_locked", sigma_floor_khz=2.0),
        tau_preconditions_passed=True,
        tau_preconditions_notes=[],
        tau_calibration_source="persisted",
        tau0_us=5.96,
        tau_G_present=True,
        recommended_shape="gaussian",
        timebase_n_used=11,
    )
    md = _render_markdown(m, "x.ftmw", include_table=False)
    assert "**Concerns flagged:** none" in md
    # A clean stage still prints its concerns block as "None flagged."
    assert "*None flagged.*" in md


def test_stage3_peak_strength_and_density_tables():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "**Peak-strength distribution (SNR)**" in md
    assert "| Population | p10 | p25 | median | p75 | p90 | max |" in md
    assert "| all candidates |" in md
    assert "| promoted |" in md
    assert "747 promoted at SNR ≥ 3" in md
    assert "**Spectral density**" in md
    assert "| Band (MHz) | Candidates | Promoted |" in md


def test_stage5_fit_quality_distributions():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    # The percentile table covers chi2r, the gate statistic, and freq precision.
    assert "**Fit-quality distributions**" in md
    assert "| reduced χ² (per window) |" in md
    assert "| shape-error ε (% per bin) |" in md
    assert "| freq precision σ_stat (kHz) |" in md
    # The SNR-binned breakdown shows the chi2r ~ SNR^2 structure + per-bin gate.
    assert "**Fit quality by window brightness (SNR_max)**" in md
    assert "| SNR_max | windows | median χ²ᵣ | p90 | max | pass | median ε(%) |" in md
    assert "| <100 |" in md
    assert "| 100-1k |" in md


def test_stage6_sigma_f_budget_distribution_table():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    # The σ_f budget distribution parallels the Stage 5 σ_stat percentile table.
    assert "**σ_f budget distribution (kHz)**" in md
    assert "| Component | p10 | p25 | median | p75 | p90 | max |" in md
    assert "| σ_stat (NLS precision) |" in md
    assert "| σ_ε (timebase) |" in md
    assert "| σ_f (total) |" in md


def test_stage4_window_distribution_table():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "**Window distribution**" in md
    assert "| Metric | p10 | p25 | median | p75 | p90 | max |" in md
    assert "| width (MHz) |" in md
    assert "| peaks / window |" in md


def test_no_emoji_in_output():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    for emoji in ("⚠️", "ℹ️", "✅", "❌"):
        assert emoji not in md
    # Concern severity is rendered as a text tag instead.
    assert "**Warning —**" in md


def test_summary_calibration_state_phrasing():
    md_self = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "ε-corrected for the free-running digitizer timebase" in md_self

    md_rb = _render_markdown(
        _model(products=_products(state="rb_locked")), "x.ftmw", include_table=False
    )
    assert "Rb-locked absolute scale" in md_rb


def test_strongest_lines_sorted_and_capped():
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
    snr_col = [ln.split("|")[4].strip() for ln in data_rows]
    assert snr_col[0] == "15" and snr_col[-1] == "6"


def test_include_table_inlines_full_list():
    prod = _products(n=5)
    md_ptr = _render_markdown(_model(products=prod), "x.ftmw", include_table=False)
    md_full = _render_markdown(_model(products=prod), "x.ftmw", include_table=True)

    assert "companion data export" in md_ptr
    assert "report table --format csv" in md_ptr

    full_section = md_full.split("## Final line list")[1]
    assert "companion data export" not in full_section
    data_rows = [ln for ln in full_section.splitlines() if ln.startswith("| 3")]
    assert len(data_rows) == 5


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
# Catalog cross-reference section + pull calibration
# ---------------------------------------------------------------------------


def _xref(products, n_sigma=3.0):
    from ftmwpipeline._internal.catalog_xref import CatalogEntry, build_cross_ref

    # A catalog entry on each peak frequency (exact) -> every line matches.
    cat = [
        CatalogEntry(p.frequency_mhz, 0.0, f"L{i}")
        for i, p in enumerate(products.peaks)
    ]
    return build_cross_ref(products.peaks, cat, catalog_path="/x/ref.csv", n_sigma=n_sigma)


def test_catalog_section_and_summary_bullet():
    prod = _products(n=4)
    xref = _xref(prod)
    md = _render_markdown(
        _model(products=prod), "x.ftmw", include_table=False, cross_ref=xref
    )
    assert "## Catalog cross-reference" in md
    assert "**Catalog match:**" in md  # summary bullet
    assert "ref.csv" in md
    assert "4 of 4 lines (100%)" in md
    assert "Pull calibration" in md
    assert "Largest pulls" in md
    assert "never an assignment" in md


def test_no_catalog_section_when_absent():
    md = _render_markdown(_model(), "x.ftmw", include_table=False)
    assert "## Catalog cross-reference" not in md
    assert "**Catalog match:**" not in md


def test_pull_interpretation_flags_optimistic():
    from ftmwpipeline._internal.catalog_xref import CatalogEntry, build_cross_ref
    from ftmwpipeline._internal.report_impl import _pull_interpretation

    # Lines sit far (in sigma) from the catalog -> wide pull spread -> optimistic.
    peaks = _products(n=6).peaks
    cat = [
        CatalogEntry(p.frequency_mhz - (0.01 if i % 2 else -0.01), 0.0, f"L{i}")
        for i, p in enumerate(peaks)
    ]  # +-10 kHz with sigma_f 1 kHz -> pull ~ +-10
    xref = build_cross_ref(peaks, cat, catalog_path="c", n_sigma=100.0)
    text = _pull_interpretation(xref)
    assert "optimistic" in text




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
    # Real numbers + the enriched pieces from the persisted record made it in.
    assert "# FTMW pipeline report" in via_impl
    assert "### Stage 5 -- per-window fitting" in via_impl
    assert "**Per-band noise**" in via_impl
    assert "**Parameters**" in via_impl
    assert "SNR-aware quality gate:" in via_impl
    assert "τ_maj =" in via_impl  # Stage 2b was run on this fixture


@pytest.mark.integration
def test_report_summary_is_read_only(stage5_small_file, tmp_path):
    """A report renders the record; it must not mutate the .ftmw file."""
    import hashlib

    fp = tmp_path / "ro.ftmw"
    shutil.copy(stage5_small_file, fp)
    before = hashlib.md5(fp.read_bytes()).hexdigest()
    report_summary_impl(str(fp))
    assert hashlib.md5(fp.read_bytes()).hexdigest() == before


@pytest.mark.integration
def test_report_summary_output_writes_file(stage5_small_file, tmp_path):
    out = tmp_path / "summary.md"
    text = report_summary_impl(str(stage5_small_file), output=str(out))
    assert out.read_text() == text
    assert text.startswith("# FTMW pipeline report")
