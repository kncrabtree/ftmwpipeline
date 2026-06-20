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
        f = next(f for s, f in iter_knob_fields(NoiseSettings) if f.name == "window_mhz")
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
