"""Resolved-settings inspection: per-field value + provenance for a ``.ftmw``.

Where the knob registry answers *what is tunable*, this module answers *what
value is actually in effect for this experiment, and which layer supplied it*.
It is the shared core behind the ``settings show`` verb (issue #28): it walks
every stage settings dataclass field, runs the same precedence chain the stage
resolvers use, and reports the winning layer.

Source of truth is the **settings dataclasses themselves** (``dataclasses.fields``
over each stage class and its sub-blocks), not the knob registry -- so fields the
registry omits because they do not sweep meaningfully in isolation
(``stage1.units_power``) still surface here, which is exactly the resolved view a
user needs. The registry is
consulted only to *enrich* a row that corresponds to a registered knob (tier,
one-line help); rows with no registered knob still appear, tiered advanced.

Precedence (mirrors the corrected D11 order in the stage resolvers, minus the
``explicit`` layer, which has no per-invocation kwargs in the verb context):

    persisted (.ftmw)  >  preset (.yml)  >  recommended  >  hard default

The headline ``start_us`` is not special-cased: it is an ordinary Stage 1 FT
setting whose resolved value lives in ``processing_parameters/ft_processing``,
so the generic FT walk attributes it to persisted (a user-stamped/detected value
in the file) vs recommended (the ``chirp_end + guard_margin`` detector output)
vs default exactly as the issue's worked example requires.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from ...core import noise_settings as noise_mod
from ...core import peak_detection_settings as peak_mod
from ...core import settings as ft_mod
from ...core import stage_fit_settings as fit_mod
from ...core import tau_calibration_settings as tau_mod
from ...core import window_planning_settings as window_mod
from ...core.stage_fit_settings import ShapeSpec, SpurSubSettings
from ...io.noise_settings_serialization import load_noise_settings_from_h5
from ...io.peak_detection_settings_serialization import (
    load_peak_detection_settings_from_h5,
)
from ...io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    read_recommended_clock_sources,
    read_stage2b_recommended_shape,
)
from ...io.tau_calibration_settings_serialization import (
    load_tau_calibration_settings_from_h5,
)
from ...io.window_planning_settings_serialization import (
    load_window_planning_settings_from_h5,
)
from ..stage1_impl import _read_settings_layer
from .registry import get_knob

# Provenance layer labels (the ``source`` column). The preset label is rendered
# as ``f"{SOURCE_PRESET_PREFIX}{name}"`` so a reader sees which preset won.
SOURCE_FTMW = ".ftmw"
SOURCE_RECOMMENDED = "recommended"
SOURCE_DEFAULT = "default"
SOURCE_PRESET_PREFIX = ".yml:"


@dataclass(frozen=True)
class SettingRow:
    """One resolved setting: its value, the layer that supplied it, and help.

    ``path`` is the dotted settings path (e.g. ``stage2b.gaussian.snr_min``),
    matching the registry's selector convention. ``source`` is one of
    :data:`SOURCE_FTMW` / :data:`SOURCE_RECOMMENDED` / :data:`SOURCE_DEFAULT`, or
    ``".yml:<name>"`` when a named preset supplied the value. ``hard_default`` is
    the resolved hard default for reference (what ``source == "default"`` uses).
    ``tier`` / ``help`` are the registry enrichment (``"advanced"`` / ``""`` when
    no knob is registered for the path).
    """

    path: str
    value: Any
    source: str
    hard_default: Any
    tier: str
    help: str


@dataclass(frozen=True)
class _StageSpec:
    """How to source every resolution layer for one stage's settings class."""

    prefix: str
    cls: type
    load_persisted: Callable[[str], Optional[Any]]
    make_default: Callable[[], Any]
    load_preset: Optional[Callable[[Union[str, Path]], Any]]
    load_recommended: Optional[Callable[[str], Optional[Any]]]


def _ft_recommended(file_path: str) -> Optional[Any]:
    """Stage 1 recommended layer: Stage 0's import-time ``recommended_processing``."""
    return _read_settings_layer(file_path, ft_mod.RECOMMENDED_PATH)


def _ft_persisted(file_path: str) -> Optional[Any]:
    """Stage 1 persisted layer: the canonical ``ft_processing`` record."""
    return _read_settings_layer(file_path, ft_mod.FT_PROCESSING_PATH)


def _fit_recommended(file_path: str) -> Optional[Any]:
    """Stage 5 recommended layer: shape from Stage 2b + clocks from Stage 0 import.

    Stage 5's resolver merges two sources of recommendations:
    - Line shape from Stage 2b's L/G/V discriminator (``recommended_shape``).
    - Instrument clock declaration from the Blackchirp loader (stored at import
      on ``stage0_fid_data`` as ``recommended_clock_sources``).
    Returns ``None`` when neither source has stamped a recommendation.
    """
    shape_str = read_stage2b_recommended_shape(file_path)
    clocks = read_recommended_clock_sources(file_path)
    if shape_str is None and clocks is None:
        return None
    return fit_mod.StageFitSettings(
        shape=ShapeSpec.coerce(shape_str) if shape_str is not None else None,
        spur=SpurSubSettings(clocks=clocks),
    )


# Stage settings classes the verb covers, in display order. Stage 0 start
# detection is intentionally absent: it is a pre-Stage-1 dataclass whose *output*
# (a recommended start_us) is what matters, and that resolves through the Stage 1
# FT walk above. The FT resolver has no preset layer (Stage 1 settings are not
# preset-driven), so its ``load_preset`` is ``None``.
_STAGE_SPECS: Tuple[_StageSpec, ...] = (
    _StageSpec(
        prefix="stage1",
        cls=ft_mod.FTSettings,
        load_persisted=_ft_persisted,
        make_default=lambda: ft_mod.resolve(None, None, None),
        load_preset=None,
        load_recommended=_ft_recommended,
    ),
    _StageSpec(
        prefix="stage2",
        cls=noise_mod.NoiseSettings,
        load_persisted=load_noise_settings_from_h5,
        make_default=noise_mod.resolve,
        load_preset=noise_mod.load_preset,
        load_recommended=None,
    ),
    _StageSpec(
        prefix="stage2b",
        cls=tau_mod.TauCalibrationSettings,
        load_persisted=load_tau_calibration_settings_from_h5,
        make_default=tau_mod.resolve,
        load_preset=tau_mod.load_preset,
        load_recommended=None,
    ),
    _StageSpec(
        prefix="stage3",
        cls=peak_mod.PeakDetectionSettings,
        load_persisted=load_peak_detection_settings_from_h5,
        make_default=peak_mod.resolve,
        load_preset=peak_mod.load_preset,
        load_recommended=None,
    ),
    _StageSpec(
        prefix="stage4",
        cls=window_mod.WindowPlanningSettings,
        load_persisted=load_window_planning_settings_from_h5,
        make_default=window_mod.resolve,
        load_preset=window_mod.load_preset,
        load_recommended=None,
    ),
    _StageSpec(
        prefix="stage5",
        cls=fit_mod.StageFitSettings,
        load_persisted=load_stage_fit_settings_from_h5,
        make_default=fit_mod.resolve,
        load_preset=fit_mod.load_preset,
        load_recommended=_fit_recommended,
    ),
)


def _enumerate_fields(cls: type) -> List[Tuple[Optional[str], str]]:
    """Walk a settings class into ``(sub_block, field)`` pairs.

    A dataclass-valued field is a sub-block: it expands into one pair per
    sub-field, ``(sub_name, field_name)``. A scalar field yields ``(None,
    field_name)``. Discovery uses a default instance so a sub-block is detected
    by value, keeping the walk in lockstep with the resolver's own structure.
    """
    inst = cls()
    out: List[Tuple[Optional[str], str]] = []
    for f in fields(cls):
        value = getattr(inst, f.name)
        if is_dataclass(value) and not isinstance(value, type):
            for sub_f in fields(value):
                out.append((f.name, sub_f.name))
        else:
            out.append((None, f.name))
    return out


def _layer_value(layer: Optional[Any], sub: Optional[str], name: str) -> Any:
    """Value of one field at one resolution layer, or ``None`` if unset/absent."""
    if layer is None:
        return None
    obj = layer if sub is None else getattr(layer, sub, None)
    if obj is None:
        return None
    return getattr(obj, name, None)


def _matches(path: str, selector: Optional[str]) -> bool:
    """Path-prefix selector match, identical to ``list_knobs``' rule."""
    if selector is None:
        return True
    return path == selector or path.startswith(selector + ".")


def _enrich(path: str) -> Tuple[str, str]:
    """``(tier, help)`` for a path: the registered knob's, else advanced/empty."""
    try:
        spec = get_knob(path)
    except KeyError:
        return "advanced", ""
    return spec.tier, spec.help


def resolve_settings_view(
    file_path: Union[str, Path],
    selector: Optional[str] = None,
    *,
    include_advanced: bool = False,
    preset: Optional[Union[str, Path]] = None,
) -> Tuple[SettingRow, ...]:
    """Resolved value + provenance for every covered setting of ``file_path``.

    Walks each stage's settings dataclass, runs the precedence chain
    ``persisted (.ftmw) > preset (.yml) > recommended > hard default`` per field,
    and returns structured :class:`SettingRow` rows (the CLI/Pipeline/api layers
    format them). Returning data, not text, keeps the dual-interface surfaces
    thin.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw``. Layers that have not been written (a stage not yet
        run) simply fall through; the file need not have reached any stage.
    selector :
        Dotted-path prefix filter (``stage2b`` / ``stage2b.gaussian``); ``None``
        returns every row.
    include_advanced :
        Include advanced-tier rows. ``False`` (default) keeps the view a short
        primary-knob starting point, matching ``scan list``.
    preset :
        Bare preset name or YAML path. When given, its per-stage blocks populate
        the preset layer (and provenance reads ``".yml:<name>"``); with no
        preset the layer is empty. Because persisted outranks preset (D11), a
        named preset changes a resolved value only for fields the file has not
        persisted.
    """
    path_str = str(file_path)
    preset_label = None if preset is None else f"{SOURCE_PRESET_PREFIX}{preset}"
    rows: List[SettingRow] = []
    for spec in _STAGE_SPECS:
        persisted = spec.load_persisted(path_str)
        recommended = spec.load_recommended(path_str) if spec.load_recommended else None
        preset_layer = (
            spec.load_preset(preset)
            if preset is not None and spec.load_preset is not None
            else None
        )
        default_inst = spec.make_default()

        for sub, name in _enumerate_fields(spec.cls):
            tail = name if sub is None else f"{sub}.{name}"
            path = f"{spec.prefix}.{tail}"
            if not _matches(path, selector):
                continue
            tier, help_text = _enrich(path)
            if tier == "advanced" and not include_advanced:
                continue

            hard_default = _layer_value(default_inst, sub, name)
            persisted_v = _layer_value(persisted, sub, name)
            preset_v = _layer_value(preset_layer, sub, name)
            recommended_v = _layer_value(recommended, sub, name)
            if persisted_v is not None:
                source, value = SOURCE_FTMW, persisted_v
            elif preset_v is not None:
                # preset_label is non-None whenever preset_layer is non-None.
                source = preset_label or SOURCE_PRESET_PREFIX
                value = preset_v
            elif recommended_v is not None:
                source, value = SOURCE_RECOMMENDED, recommended_v
            else:
                source, value = SOURCE_DEFAULT, hard_default

            rows.append(
                SettingRow(
                    path=path,
                    value=value,
                    source=source,
                    hard_default=hard_default,
                    tier=tier,
                    help=help_text,
                )
            )
    return tuple(rows)
