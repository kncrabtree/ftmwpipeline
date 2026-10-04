"""Declared ``choices`` / ``bounds`` hold wherever a stage's settings resolve.

``settings set`` refuses a value outside a field's declaration; the stage
resolvers hold the merged result to the same declaration, so a value arriving
through a ``settings=`` object, a preset or a persisted record is refused as
``bad_setting`` with the registry path too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import pytest

from ftmwpipeline.core import noise_settings as noise_mod
from ftmwpipeline.core import peak_detection_settings as peak_mod
from ftmwpipeline.core import settings as ft_mod
from ftmwpipeline.core import settings_framework as sf
from ftmwpipeline.core import stage_fit_settings as fit_mod
from ftmwpipeline.core import tau_calibration_settings as tau_mod
from ftmwpipeline.core import window_planning_settings as window_mod
from ftmwpipeline.core.knob_metadata import (
    check_declared_typing,
    field_typing,
    make_bounds,
)
from ftmwpipeline.file_manager import BadSettingError

_N_EFF = "stage5.conservative.n_eff_kind"
_N_EFF_CHOICES = ("perplexity_log1p_snr", "kish_mag_sq", "kish_mag", "hard_radius")

_SUBBLOCK_MODULES = [noise_mod, tau_mod, peak_mod, window_mod, fit_mod]


def _bad_fit() -> fit_mod.StageFitSettings:
    return fit_mod.StageFitSettings(
        conservative=fit_mod.ConservativeSubSettings(n_eff_kind="bogus")
    )


# --- existing valid inputs still resolve ------------------------------------
@pytest.mark.parametrize("mod", _SUBBLOCK_MODULES, ids=lambda m: m.__name__)
def test_the_packaged_defaults_preset_resolves(mod: Any) -> None:
    mod.resolve(preset=mod.load_preset("defaults"))
    mod.resolve()


def test_the_ft_hard_defaults_resolve() -> None:
    ft_mod.resolve(None, None, None)


@pytest.mark.parametrize("choice", _N_EFF_CHOICES)
def test_every_declared_choice_resolves(choice: str) -> None:
    explicit = fit_mod.StageFitSettings(
        conservative=fit_mod.ConservativeSubSettings(n_eff_kind=choice)
    )
    assert fit_mod.resolve(explicit=explicit).conservative.n_eff_kind == choice


def test_an_unset_field_falls_through_unchecked_to_the_default() -> None:
    resolved = fit_mod.resolve(explicit=fit_mod.StageFitSettings())
    assert resolved.conservative.n_eff_kind == "perplexity_log1p_snr"


# --- a violation from any layer is bad_setting -----------------------------
@pytest.mark.parametrize("layer", ["explicit", "preset", "persisted", "recommended"])
def test_a_bad_choice_from_any_layer_is_bad_setting(layer: str) -> None:
    """Mutation: drop the check from ``fill_resolved_subblocks``."""
    with pytest.raises(BadSettingError) as excinfo:
        fit_mod.resolve(**{layer: _bad_fit()})
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert err.code == "bad_setting"
    assert err.path == _N_EFF
    assert err.value == "bogus"
    assert err.expected == "one of " + ", ".join(repr(c) for c in _N_EFF_CHOICES)


def test_a_bad_choice_in_a_preset_yaml_is_bad_setting() -> None:
    preset = fit_mod.from_yaml_dict({"conservative": {"n_eff_kind": "nope"}})
    with pytest.raises(BadSettingError) as excinfo:
        fit_mod.resolve(preset=preset)
    assert excinfo.value.path == _N_EFF
    assert excinfo.value.value == "nope"


def test_a_bad_value_overridden_by_a_higher_layer_is_not_used() -> None:
    """Only the value that wins is held to the declaration."""
    good = fit_mod.StageFitSettings(
        conservative=fit_mod.ConservativeSubSettings(n_eff_kind="kish_mag")
    )
    resolved = fit_mod.resolve(explicit=good, preset=_bad_fit())
    assert resolved.conservative.n_eff_kind == "kish_mag"


# --- every resolver checks, under its registry prefix -----------------------
@pytest.mark.parametrize(
    "mod, prefix",
    [
        (tau_mod, "stage2b"),
        (peak_mod, "stage3"),
        (window_mod, "stage4"),
        (fit_mod, "stage5"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_each_subblock_resolver_checks_under_its_prefix(
    monkeypatch: pytest.MonkeyPatch, mod: Any, prefix: str
) -> None:
    calls: List[Tuple[type, str]] = []
    monkeypatch.setattr(
        sf, "check_declared_typing", lambda s, p: calls.append((type(s), p))
    )
    resolved = mod.resolve()
    assert calls == [(type(resolved), prefix)]


@pytest.mark.parametrize(
    "mod, prefix, call",
    [
        (ft_mod, "stage1", lambda: ft_mod.resolve(None, None, None)),
        (noise_mod, "stage2", lambda: noise_mod.resolve()),
    ],
    ids=["stage1", "stage2"],
)
def test_each_flat_resolver_checks_under_its_prefix(
    monkeypatch: pytest.MonkeyPatch, mod: Any, prefix: str, call: Any
) -> None:
    calls: List[str] = []
    monkeypatch.setattr(mod, "check_declared_typing", lambda s, p: calls.append(p))
    call()
    assert calls == [prefix]


# --- the walk: registry path forms -------------------------------------------
@dataclass
class _Sub:
    ratio: Optional[float] = field_typing(bounds=make_bounds(min=0.0, max=1.0))
    plain: Optional[float] = None


@dataclass
class _Flat:
    kind: Optional[str] = field_typing(choices=("a", "b"))
    sub: _Sub = field(default_factory=_Sub)


def test_a_scalar_field_is_named_prefix_dot_field() -> None:
    with pytest.raises(BadSettingError) as excinfo:
        check_declared_typing(_Flat(kind="c"), "stageX")
    assert excinfo.value.path == "stageX.kind"


def test_a_subblock_field_is_named_prefix_dot_sub_dot_field() -> None:
    with pytest.raises(BadSettingError) as excinfo:
        check_declared_typing(_Flat(sub=_Sub(ratio=1.5)), "stageX")
    assert excinfo.value.path == "stageX.sub.ratio"
    assert excinfo.value.expected == "a value in [0.0, 1.0]"


def test_unset_and_undeclared_fields_are_not_checked() -> None:
    check_declared_typing(_Flat(), "stageX")
    check_declared_typing(_Flat(kind="a", sub=_Sub(ratio=0.5, plain=1e99)), "stageX")
