"""Change-grammar for the ``settings`` meta-object: ``set`` and ``export``.

Two mutating companions to the read-only :mod:`settings_inspection` view:

* :func:`set_setting` persists one chosen value into the ``.ftmw`` (the
  persisted layer), so the experiment reproduces it from the file alone. Because
  the stored stage results were computed against the old value, the affected
  stage and everything downstream are invalidated -- the file never carries
  results inconsistent with its persisted settings.
* :func:`export_settings` writes the file's chosen (persisted) values to a
  ``.yml`` preset block, the portable form a sibling experiment loads via
  ``--preset``.

Stage 1 is special. The canonical FT is unapodized and native-length (there are
no apodization knobs). Its data-selection knobs (``start_us`` / ``end_us`` /
``trim`` / ``units_power`` / ``rdc``) are settable but, since the FT is
recomputed on demand from these settings, changing them invalidates every
downstream stage. Presets do not carry Stage 1, so it is excluded from
:func:`export_settings`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml  # type: ignore[import-untyped]

from ...core import noise_settings as noise_mod
from ...core import peak_detection_settings as peak_mod
from ...core import settings as ft_mod
from ...core import stage_fit_settings as fit_mod
from ...core import tau_calibration_settings as tau_mod
from ...core import window_planning_settings as window_mod
from ...core.stage_fit_settings import ClockSource, ShapeSpec, coerce_clock_sources
from ...io.noise_settings_serialization import (
    load_noise_settings_from_h5,
    save_noise_settings_to_h5,
)
from ...io.peak_detection_settings_serialization import (
    load_peak_detection_settings_from_h5,
    save_peak_detection_settings_to_h5,
)
from ...io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ...io.tau_calibration_settings_serialization import (
    load_tau_calibration_settings_from_h5,
    save_tau_calibration_settings_to_h5,
)
from ...io.window_planning_settings_serialization import (
    load_window_planning_settings_from_h5,
    save_window_planning_settings_to_h5,
)


@dataclass(frozen=True)
class _MutSpec:
    """How to load / save / serialize one stage's settings, plus its tracker
    stage names (for inclusive invalidation) and its preset block key."""

    prefix: str
    cls: type
    load: Callable[[str], Optional[Any]]
    save: Callable[[str, Any], None]
    to_yaml_dict: Callable[[Any], Dict[str, Any]]
    block_key: str
    own_stages: Tuple[str, ...]


_MUT_SPECS: Dict[str, _MutSpec] = {
    "stage2": _MutSpec(
        "stage2",
        noise_mod.NoiseSettings,
        load_noise_settings_from_h5,
        save_noise_settings_to_h5,
        noise_mod.to_yaml_dict,
        "stage2",
        ("stage2_noise_result",),
    ),
    "stage2b": _MutSpec(
        "stage2b",
        tau_mod.TauCalibrationSettings,
        load_tau_calibration_settings_from_h5,
        save_tau_calibration_settings_to_h5,
        tau_mod.to_yaml_dict,
        "stage2b",
        ("stage2b_tau_calibration", "stage2b_tau_G_calibration"),
    ),
    "stage3": _MutSpec(
        "stage3",
        peak_mod.PeakDetectionSettings,
        load_peak_detection_settings_from_h5,
        save_peak_detection_settings_to_h5,
        peak_mod.to_yaml_dict,
        "stage3",
        ("stage3_peaks",),
    ),
    "stage4": _MutSpec(
        "stage4",
        window_mod.WindowPlanningSettings,
        load_window_planning_settings_from_h5,
        save_window_planning_settings_to_h5,
        window_mod.to_yaml_dict,
        "stage4",
        ("stage4_windows",),
    ),
    "stage5": _MutSpec(
        "stage5",
        fit_mod.StageFitSettings,
        load_stage_fit_settings_from_h5,
        save_stage_fit_settings_to_h5,
        fit_mod.to_yaml_dict,
        "stage5",
        ("stage5_fitting",),
    ),
}


@dataclass(frozen=True)
class SetResult:
    """Outcome of :func:`set_setting`."""

    path: str
    value: Any
    invalidated: Tuple[str, ...]


@dataclass(frozen=True)
class ExportResult:
    """Outcome of :func:`export_settings`."""

    out_path: str
    paths: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Knob-path resolution and value coercion
# ---------------------------------------------------------------------------
def _split_knob(knob: str) -> Tuple[str, Optional[str], str]:
    """Split a dotted knob into ``(prefix, sub_block_or_None, field)``."""
    parts = knob.split(".")
    if len(parts) == 2:
        return parts[0], None, parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f"malformed knob {knob!r}; expected 'stageN.field' or " f"'stageN.sub.field'"
    )


def _owner_and_field_type(cls: type, sub: Optional[str], field: str) -> Any:
    """Resolve the (sub-)dataclass that owns ``field`` and return the field's
    declared type (with ``Optional`` unwrapped). Raises ``KeyError`` when the
    field does not exist on the settings class."""
    owner = cls
    if sub is not None:
        sub_value = getattr(cls(), sub, None)
        if not is_dataclass(sub_value):
            raise KeyError(sub)
        owner = type(sub_value)
    hints = get_type_hints(owner)
    if field not in hints:
        raise KeyError(field)
    hint = hints[field]
    if get_origin(hint) is Union:
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return hint


def _coerce(type_hint: Any, raw: str) -> Any:
    """Coerce the CLI string ``raw`` to the field's declared type."""
    origin = get_origin(type_hint)
    if type_hint is ShapeSpec:
        return ShapeSpec.coerce(raw)
    if origin is tuple and ClockSource in get_args(type_hint):
        # The clock declaration sets as a JSON list:
        # settings set stage5.spur.clocks '[{"freq_mhz": 5760, ...}, ...]'
        return coerce_clock_sources(raw)
    if type_hint is bool:
        low = raw.strip().lower()
        if low in {"true", "1", "yes", "on"}:
            return True
        if low in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"cannot parse boolean from {raw!r}")
    if type_hint is int:
        return int(raw)
    if type_hint is float:
        return float(raw)
    if type_hint is str:
        return raw
    if origin in (tuple, list):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        coerced = [_coerce_scalar(p) for p in parts]
        return tuple(coerced) if origin is tuple else coerced
    raise ValueError(
        f"setting a value of type {getattr(type_hint, '__name__', type_hint)!r} "
        f"is not supported by 'settings set'"
    )


def _coerce_scalar(raw: str) -> Any:
    """Best-effort scalar coercion for tuple/list elements (number else text)."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------
def set_setting(file_path: Union[str, Path], knob: str, raw_value: str) -> SetResult:
    """Persist ``raw_value`` for ``knob`` into the ``.ftmw`` and invalidate the
    affected stage and everything downstream.

    ``knob`` is a dotted settings path (``stage2.window_mhz`` /
    ``stage2b.gaussian.snr_min`` / ``stage5.shape``). The value is coerced to the
    field's declared type. The canonical FT is unapodized and native-length, so
    there are no FT apodization knobs to set.
    """
    path = str(file_path)
    prefix, sub, field = _split_knob(knob)

    if prefix == "stage1":
        return _set_stage1(path, knob, field, raw_value)

    spec = _MUT_SPECS.get(prefix)
    if spec is None:
        raise ValueError(
            f"unknown settings stage {prefix!r} in knob {knob!r}; settable "
            f"stages are {sorted(_MUT_SPECS)} (and stage1 windowing knobs)"
        )
    try:
        field_type = _owner_and_field_type(spec.cls, sub, field)
    except KeyError:
        raise ValueError(
            f"unknown setting {knob!r}; no such field on {prefix} settings"
        ) from None
    value = _coerce(field_type, raw_value)

    settings = spec.load(path) or spec.cls()
    _assign(settings, sub, field, value)
    spec.save(path, settings)

    invalidated = _invalidate_inclusive(path, spec.own_stages)
    return SetResult(path=knob, value=value, invalidated=invalidated)


def _set_stage1(path: str, knob: str, field: str, raw_value: str) -> SetResult:
    """Persist a Stage 1 windowing knob into ``ft_processing`` and invalidate
    every downstream stage (the FT is recomputed on demand from these settings)."""
    from ..stage1_impl import _persist_canonical_settings, _resolve_settings

    try:
        field_type = _owner_and_field_type(ft_mod.FTSettings, None, field)
    except KeyError:
        raise ValueError(
            f"unknown setting {knob!r}; no such field on stage1 FT settings"
        ) from None
    value = _coerce(field_type, raw_value)

    resolved = _resolve_settings(path, None)
    setattr(resolved, field, value)
    # _persist_canonical_settings rewrites ft_processing and invalidates every
    # FT-dependent stage when the record changes.
    completed_before = _completed_stages(path)
    _persist_canonical_settings(path, resolved)
    invalidated = tuple(sorted(completed_before - _completed_stages(path)))
    return SetResult(path=knob, value=value, invalidated=invalidated)


def _assign(settings: Any, sub: Optional[str], field: str, value: Any) -> None:
    target = settings if sub is None else getattr(settings, sub)
    setattr(target, field, value)


# ---------------------------------------------------------------------------
# Stage invalidation
# ---------------------------------------------------------------------------
def _completed_stages(path: str) -> set:
    import h5py

    with h5py.File(path, "r") as h5f:
        if "pipeline_stages" not in h5f:
            return set()
        raw = h5f["pipeline_stages"].attrs.get("completed_stages", "[]")
    return set(json.loads(raw))


def _invalidate_inclusive(path: str, own_stages: Tuple[str, ...]) -> Tuple[str, ...]:
    """Drop ``own_stages`` (results + completion) and every stage that depends on
    them, returning the invalidated stage names sorted."""
    import h5py

    from ...file_manager import PipelineStageTracker, invalidate_downstream_stages

    dependents: List[str] = []
    for stage in own_stages:
        dependents.extend(invalidate_downstream_stages(path, stage))

    paths = PipelineStageTracker.STAGE_DATA_PATHS
    dropped: List[str] = []
    with h5py.File(path, "a") as h5f:
        if "pipeline_stages" not in h5f:
            return tuple(sorted(set(dependents)))
        stages_group = h5f["pipeline_stages"]
        completed = json.loads(stages_group.attrs.get("completed_stages", "[]"))
        for stage in own_stages:
            data_path = paths.get(stage, stage)
            if data_path in h5f:
                del h5f[data_path]
            if stage in completed:
                completed.remove(stage)
                dropped.append(stage)
        if dropped:
            stages_group.attrs["completed_stages"] = json.dumps(completed)
            stages_group.attrs["last_updated"] = datetime.now().isoformat()
    return tuple(sorted(set(dependents) | set(dropped)))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export_settings(
    file_path: Union[str, Path],
    out_path: Union[str, Path],
    selector: Optional[str] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ExportResult:
    """Write the file's chosen (persisted) Stage 2--5 values to a ``.yml`` preset.

    Each stage's persisted settings are serialized through its sparse
    ``to_yaml_dict`` into the matching ``stageN:`` block, filtered by the dotted
    ``selector`` (a stage or sub-block / field prefix). Stage 1 is excluded --
    presets do not carry FT settings. The result loads back via the stages'
    ``--preset`` path.
    """
    src = str(file_path)
    out = Path(out_path)

    document: Dict[str, Any] = {}
    document["name"] = name or out.stem
    if description is not None:
        document["description"] = description

    exported: List[str] = []
    for spec in _MUT_SPECS.values():
        persisted = spec.load(src)
        if persisted is None:
            continue
        block = spec.to_yaml_dict(persisted)
        block = _filter_block(spec.prefix, block, selector)
        if block:
            document[spec.block_key] = _to_native(block)
            exported.extend(_flatten_paths(spec.prefix, block))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        yaml.safe_dump(document, fh, sort_keys=False, default_flow_style=False)
    return ExportResult(out_path=str(out), paths=tuple(exported))


def _filter_block(
    prefix: str, block: Dict[str, Any], selector: Optional[str]
) -> Dict[str, Any]:
    """Keep only the block entries whose dotted path matches ``selector``."""
    if selector is None:
        return block
    out: Dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, dict):
            kept = {
                fk: fv
                for fk, fv in value.items()
                if _selector_match(f"{prefix}.{key}.{fk}", selector)
            }
            if kept:
                out[key] = kept
        elif _selector_match(f"{prefix}.{key}", selector):
            out[key] = value
    return out


def _flatten_paths(prefix: str, block: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key, value in block.items():
        if isinstance(value, dict):
            paths.extend(f"{prefix}.{key}.{fk}" for fk in value)
        else:
            paths.append(f"{prefix}.{key}")
    return paths


def _selector_match(path: str, selector: str) -> bool:
    return path == selector or path.startswith(selector + ".")


def _to_native(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays (as HDF5 returns them) to native
    Python types so ``yaml.safe_dump`` can represent the preset block."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value
