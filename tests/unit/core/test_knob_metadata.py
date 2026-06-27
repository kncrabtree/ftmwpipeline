"""Unit tests for the per-knob field metadata and the data-driven argspec.

``knob_field`` is the single declaration site for a tunable setting: it carries
the help / tier / instrument-sensitivity / sweep grid and (opt-in) the CLI flag
spec. These tests pin the accessors and the nested-aware CLI generation that
``add_settings_args`` / ``settings_from_namespace`` build on top of it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pytest

from ftmwpipeline.cli._argspec import add_settings_args, settings_from_namespace
from ftmwpipeline.core.knob_metadata import (
    field_knob_meta,
    iter_knob_fields,
    knob_field,
    knob_meta,
)
from ftmwpipeline.core.noise_settings import NoiseSettings
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings
from ftmwpipeline.core.stage_fit_settings import StageFitSettings
from ftmwpipeline.core.tau_calibration_settings import TauCalibrationSettings
from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings


# A synthetic nested settings class exercising the sub-block walk (the real
# nested stages are not CLI-wired yet; this locks the machinery now).
@dataclass
class _Sub:
    alpha: Optional[float] = knob_field(
        help="alpha knob", cli=True, argtype=float, tier="primary"
    )
    beta: Optional[int] = knob_field(help="beta knob (no flag)", grid=(1, 2))


@dataclass
class _Nested:
    sub: _Sub = None  # type: ignore[assignment]
    flat: Optional[bool] = knob_field(help="flat flag", cli=True, is_flag=True)
    plain: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sub is None:
            self.sub = _Sub()


class TestKnobMeta:
    def test_knob_field_default_is_none(self) -> None:
        assert _Sub().alpha is None and _Sub().beta is None

    def test_knob_meta_reads_descriptors(self) -> None:
        f = next(
            f for s, f in iter_knob_fields(NoiseSettings) if f.name == "window_mhz"
        )
        km = knob_meta(f)
        assert km is not None
        assert km.tier == "primary"
        assert km.inst_sensitivity == "Y"
        assert km.grid == (40.0, 60.0, 80.0, 120.0, 160.0)
        assert km.cli is not None and km.cli.argtype is float

    def test_knob_meta_none_for_plain_field(self) -> None:
        f = next(f for s, f in iter_knob_fields(_Nested) if f.name == "plain")
        assert knob_meta(f) is None

    def test_cli_none_when_not_exposed(self) -> None:
        km = field_knob_meta(_Sub, "beta")
        assert km.cli is None  # beta is a knob but not CLI-exposed

    def test_field_knob_meta_nested_path(self) -> None:
        km = field_knob_meta(_Nested, "sub.alpha")
        assert km.help == "alpha knob" and km.cli is not None

    def test_field_knob_meta_missing_raises(self) -> None:
        with pytest.raises(KeyError):
            field_knob_meta(NoiseSettings, "no_such_field")

    def test_iter_knob_fields_descends_one_level(self) -> None:
        pairs = [(sub, f.name) for sub, f in iter_knob_fields(_Nested)]
        assert ("sub", "alpha") in pairs and ("sub", "beta") in pairs
        assert (None, "flat") in pairs and (None, "plain") in pairs


class TestArgspecNested:
    def _parser(self, cls: type) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        add_settings_args(p, cls)
        return p

    def test_only_cli_fields_get_flags(self) -> None:
        p = self._parser(_Nested)
        opts = {a.dest for a in p._actions if a.dest != "help"}
        # alpha (sub, cli) and flat (top, cli) only; beta/plain have no flag.
        assert opts == {"sub.alpha", "flat"}

    def test_nested_roundtrip(self) -> None:
        p = self._parser(_Nested)
        ns = p.parse_args(["--alpha", "1.5", "--flat"])
        s = settings_from_namespace(ns, _Nested)
        assert s.sub.alpha == 1.5
        assert s.flat is True
        assert s.sub.beta is None  # untouched knob stays unset

    def test_empty_parse_is_all_unset(self) -> None:
        p = self._parser(_Nested)
        s = settings_from_namespace(p.parse_args([]), _Nested)
        assert s.sub.alpha is None and s.flat is None


class TestStage2FlagParity:
    """The generated `noise run` flags equal the retired hand-rolled set."""

    def test_noise_flags_match_legacy_surface(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, NoiseSettings)
        flags = {opt for a in p._actions for opt in a.option_strings}
        expected = {
            "--window-mhz",
            "--pedestal-mhz",
            "--line-k",
            "--n-iter",
            "--region-aware",
            "--no-region-aware",  # BooleanOptionalAction keeps the legacy form
            "--smoothing-mhz",
            "--smoothing-percentile",
            "--convolve-mhz",
        }
        assert flags == expected

    def test_every_noise_field_is_cli_exposed(self) -> None:
        # Stage 2 exposes all knobs on the CLI; none are settings-set-only.
        for f in NoiseSettings().__dataclass_fields__.values():
            km = knob_meta(f)
            assert km is not None and km.cli is not None, f.name


class TestRegistryFieldSingleSource:
    """The Stage 2 knob registry sources its descriptors from the field.

    After the unification the field metadata is the single declaration site;
    the registry must echo it rather than carry its own literals.
    """

    def test_stage2_registry_echoes_field_metadata(self) -> None:
        from ftmwpipeline._internal.tuning.registry import get_knob

        for name in (
            "window_mhz",
            "pedestal_mhz",
            "smoothing_mhz",
            "line_k",
            "n_iter",
            "region_aware",
            "smoothing_percentile",
            "convolve_mhz",
        ):
            spec = get_knob(f"stage2.{name}")
            km = field_knob_meta(NoiseSettings, name)
            assert spec.help == km.help
            assert spec.tier == km.tier
            assert spec.inst_sensitivity == km.inst_sensitivity
            assert spec.default_grid == km.grid


class TestStage2bFlagParity:
    """The generated `tau run` flags equal the curated Stage 2b CLI surface.

    Every Stage 2b knob lives in a sub-block, so each generated flag's ``dest``
    is sub-block-qualified (``"<sub>.<field>"``) to avoid leaf-name collisions
    (``min_contributors`` lives in both ``aggregation`` and ``gaussian``).
    """

    def test_stage2b_flags_match_curated_surface(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, TauCalibrationSettings)
        flags = {opt for a in p._actions for opt in a.option_strings}
        expected = {
            "--n-seg",
            "--t-sigma",
            "--tau-max-us",
            "--rss-gate-factor",
            "--sigma-time",
            "--min-contributors",
            "--sigma-tau-fraction-max",
            "--bimodality-dominant-fraction",
            "--snr-min",
            "--tau-g-bound-lo",
            "--tau-g-bound-hi",
            "--delta-chi2r-min",
            "--tau-g-upper-fraction",
        }
        assert flags == expected

    def test_stage2b_dests_are_subblock_qualified(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, TauCalibrationSettings)
        dests = {a.dest for a in p._actions if a.dest != "help"}
        expected = {
            "stft.n_seg",
            "stft.t_sigma",
            "stft.tau_max_us",
            "stft.rss_gate_factor",
            "stft.sigma_time",
            "aggregation.min_contributors",
            "aggregation.sigma_tau_fraction_max",
            "aggregation.bimodality_dominant_fraction",
            "gaussian.snr_min",
            "gaussian.tau_G_bound_lo",
            "gaussian.tau_G_bound_hi",
            "gaussian.delta_chi2r_min",
            "gaussian.tau_G_upper_fraction",
        }
        assert dests == expected


class TestStage2bRegistrySingleSource:
    """The Stage 2b knob registry sources its descriptors from the field.

    A representative sample across all six sub-blocks (including the two
    same-named knobs ``band.min_contributors_per_band`` and
    ``gaussian.min_contributors``) must echo the field metadata rather than
    carry its own literals.
    """

    def test_stage2b_registry_echoes_field_metadata(self) -> None:
        from ftmwpipeline._internal.tuning.registry import get_knob

        for tail in (
            "stft.n_seg",
            "polish.polish_snr_cap",
            "aggregation.sigma_tau_fraction_max",
            "band.min_contributors_per_band",
            "gaussian.min_contributors",
            "recommendation.pure_margin_threshold",
        ):
            spec = get_knob(f"stage2b.{tail}")
            km = field_knob_meta(TauCalibrationSettings, tail)
            assert spec.help == km.help
            assert spec.tier == km.tier
            assert spec.inst_sensitivity == km.inst_sensitivity
            assert spec.default_grid == km.grid


class TestStage3FlagParity:
    """The generated `peaks run` flags equal the curated Stage 3 CLI surface.

    Every Stage 3 knob lives in a sub-block, so each generated flag's ``dest``
    is sub-block-qualified (``"<sub>.<field>"``). The gap-pass toggle is a
    tri-state ``BooleanOptionalAction``, so it spells both ``--gap-pass`` and
    the historical ``--no-gap-pass``.
    """

    def test_stage3_flags_match_curated_surface(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, PeakDetectionSettings)
        flags = {opt for a in p._actions for opt in a.option_strings}
        expected = {
            "--min-snr",
            "--weak-medium-snr",
            "--medium-strong-snr",
            "--sg-window",
            "--sg-order",
            "--primary-window",
            "--min-exclusion-mhz",
            "--gap-pass",
            "--no-gap-pass",
        }
        assert flags == expected

    def test_stage3_dests_are_subblock_qualified(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, PeakDetectionSettings)
        dests = {a.dest for a in p._actions if a.dest != "help"}
        expected = {
            "promotion.min_snr",
            "promotion.weak_medium_snr",
            "promotion.medium_strong_snr",
            "savgol.sg_window",
            "savgol.sg_order",
            "primary_pass.primary_window",
            "primary_pass.min_exclusion_mhz",
            "gap_pass.run_gap_pass",
        }
        assert dests == expected


class TestStage3RegistrySingleSource:
    """The Stage 3 knob registry sources its descriptors from the field.

    A representative sample across all four sub-blocks (including the
    registry-only knobs that carry no CLI flag) must echo the field metadata
    rather than carry its own literals.
    """

    def test_stage3_registry_echoes_field_metadata(self) -> None:
        from ftmwpipeline._internal.tuning.registry import get_knob

        for tail in (
            "promotion.min_snr",
            "promotion.internal_min_snr",
            "savgol.sg_window",
            "primary_pass.noise_window_mhz",
            "gap_pass.run_gap_pass",
            "gap_pass.gap_leakage_floor_k",
        ):
            spec = get_knob(f"stage3.{tail}")
            km = field_knob_meta(PeakDetectionSettings, tail)
            assert spec.help == km.help
            assert spec.tier == km.tier
            assert spec.inst_sensitivity == km.inst_sensitivity
            assert spec.default_grid == km.grid


class TestStage4FlagParity:
    """The generated `windows run` flags equal the curated Stage 4 CLI surface.

    Every Stage 4 knob lives in a sub-block, so each generated flag's ``dest``
    is sub-block-qualified (``"<sub>.<field>"``). None of the eleven knobs is a
    boolean, so the surface is exactly eleven plain ``--flag VALUE`` options.
    """

    def test_stage4_flags_match_curated_surface(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, WindowPlanningSettings)
        flags = {opt for a in p._actions for opt in a.option_strings}
        expected = {
            "--edge-m",
            "--trim-m",
            "--edge-threshold",
            "--max-window-width-mhz",
            "--min-window-half-width-mhz",
            "--min-window-half-width-points",
            "--max-peaks-per-window",
            "--max-window-width-points",
            "--min-freeze-snr",
            "--magnitude-attachment-threshold",
            "--skirt-level-keep",
            "--curvature-keep-sigma",
            "--tau-us",
        }
        assert flags == expected

    def test_stage4_dests_are_subblock_qualified(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, WindowPlanningSettings)
        dests = {a.dest for a in p._actions if a.dest != "help"}
        expected = {
            "coherence.edge_m",
            "coherence.trim_m",
            "coherence.edge_threshold",
            "clustering.max_window_width_mhz",
            "clustering.min_window_half_width_mhz",
            "clustering.min_window_half_width_points",
            "clustering.max_peaks_per_window",
            "clustering.max_window_width_points",
            "contributor.min_freeze_snr",
            "contributor.magnitude_attachment_threshold",
            "contributor.skirt_level_keep",
            "contributor.curvature_keep_sigma",
            "leakage.tau_us",
        }
        assert dests == expected


class TestStage4RegistrySingleSource:
    """The Stage 4 knob registry sources its descriptors from the field.

    A representative sample across all four sub-blocks must echo the field
    metadata rather than carry its own literals.
    """

    def test_stage4_registry_echoes_field_metadata(self) -> None:
        from ftmwpipeline._internal.tuning.registry import get_knob

        for tail in (
            "coherence.edge_threshold",
            "coherence.edge_m",
            "clustering.max_window_width_mhz",
            "clustering.max_peaks_per_window",
            "contributor.min_freeze_snr",
            "leakage.tau_us",
        ):
            spec = get_knob(f"stage4.{tail}")
            km = field_knob_meta(WindowPlanningSettings, tail)
            assert spec.help == km.help
            assert spec.tier == km.tier
            assert spec.inst_sensitivity == km.inst_sensitivity
            assert spec.default_grid == km.grid


class TestStage5FlagParity:
    """The generated `fit run` flags equal the curated Stage 5 CLI surface.

    Every Stage 5 knob lives in a sub-block, so each generated flag's ``dest``
    is sub-block-qualified (``"<sub>.<field>"``). Two of the ten cli=True knobs
    are tri-state booleans rendered with ``BooleanOptionalAction`` (each spells
    both ``--x`` and ``--no-x``), so the flag surface is twelve strings.
    """

    def test_stage5_flags_match_curated_surface(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, StageFitSettings)
        flags = {opt for a in p._actions for opt in a.option_strings}
        expected = {
            "--tau0-us",
            "--fit-tau",
            "--no-fit-tau",
            "--max-decay-factor",
            "--per-band-tau",
            "--no-per-band-tau",
            "--residual-edge-threshold",
            "--residual-edge-m",
            "--max-thaw-rounds",
            "--max-replan-rounds",
            "--max-residual-rescue-rounds",
            "--rescue-snr-threshold",
        }
        assert flags == expected

    def test_stage5_dests_are_subblock_qualified(self) -> None:
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, StageFitSettings)
        dests = {a.dest for a in p._actions if a.dest != "help"}
        expected = {
            "tau.tau0_us",
            "tau.fit_tau",
            "tau.max_decay_factor",
            "tau.per_band_tau",
            "thaw.residual_edge_threshold",
            "thaw.residual_edge_m",
            "thaw.max_thaw_rounds",
            "thaw.max_replan_rounds",
            "rescue.max_rounds",
            "rescue.snr_threshold",
        }
        assert dests == expected

    def test_stage5_keepers_get_no_generated_flag(self) -> None:
        """The three kept explicit args and the flag-less rescue prominence knob
        are not tagged ``cli=True`` -- they must not appear in the generated
        surface (``shape`` / the τ-override pair are hand-rolled; prominence had
        no historical flag)."""
        p = argparse.ArgumentParser(add_help=False)
        add_settings_args(p, StageFitSettings)
        dests = {a.dest for a in p._actions if a.dest != "help"}
        for absent in (
            "shape",
            "tau.tau_maj_override_us",
            "tau.sigma_tau_override_us",
            "rescue.prominence_threshold",
        ):
            assert absent not in dests


class TestStage5RegistrySingleSource:
    """The Stage 5 knob registry sources its descriptors from the field.

    A representative sample across several sub-blocks (the largest registry
    block) must echo the field metadata rather than carry its own literals.
    """

    def test_stage5_registry_echoes_field_metadata(self) -> None:
        from ftmwpipeline._internal.tuning.registry import get_knob

        for tail in (
            "tau.fit_tau_min_snr",
            "tau.tau0_us",
            "tau.per_band_tau",
            "conservative.weak_window_snr_threshold",
            "conservative.max_peaks",
            "penalties.phase_penalty_lambda",
            "seeder.seeder_rchi2",
            "baseline.edge_threshold",
            "doublet_alternative.k_res",
            "rescue.snr_threshold",
            "rescue.max_rounds",
            "spur.integer_tol_mhz",
            "thaw.residual_edge_threshold",
            "thaw.residual_edge_m",
        ):
            spec = get_knob(f"stage5.{tail}")
            km = field_knob_meta(StageFitSettings, tail)
            assert spec.help == km.help
            assert spec.tier == km.tier
            assert spec.inst_sensitivity == km.inst_sensitivity
            assert spec.default_grid == km.grid
