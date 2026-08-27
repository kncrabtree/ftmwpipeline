"""Change-grammar for the ``settings`` meta-object: ``set`` and ``export``.

Two mutating companions to the read-only :mod:`settings_inspection` view:

* :func:`set_setting` persists one chosen value into the ``.ftmw`` (the
  persisted layer), so the experiment reproduces it from the file alone. Because
  the stored stage results were computed against the old value, the affected
  stage and everything downstream are invalidated -- the file never carries
  results inconsistent with its persisted settings.
* :func:`unset_setting` clears one field back to ``None``, so the resolver's
  own layers (preset / recommended / hard default) decide it again. It is the
  inverse of :func:`set_setting`, invalidates the same stages, and is what a
  ``None`` value means -- there is deliberately no *string* that encodes it.
* :func:`export_settings` writes the file's chosen (persisted) values to a
  ``.yml`` preset block, the portable form a sibling experiment loads via
  ``--preset``.

Stage 1 is special. The FT is unapodized and native-length (there are
no apodization knobs) and DC removal is unconditional (no ``rdc`` knob). Its
data-selection knobs (``start_us`` / ``end_us`` / ``trim`` / ``units_power``)
are settable but, since the FT is recomputed on demand from these settings,
changing them invalidates every downstream stage. Presets do not carry Stage 1,
so it is excluded from :func:`export_settings`.

Value encoding
--------------

:func:`set_setting` takes ``Any``. A **native** Python value of the field's
declared type is the direct form and is what an in-process caller should pass;
a **string** is the CLI's form and is parsed per the table below. Either way the
value is validated against the field's declared type and a value that cannot be
coerced raises ``ValueError`` -- nothing is stored on a parse failure.

===============================  =========================================
field type                       accepted encodings
===============================  =========================================
``float`` / ``int``              the number; or a string ``float``/``int``
                                 parses (an ``int`` field also accepts an
                                 integral float, since JSON has one number
                                 type)
``bool``                         ``True``/``False``; or one of
                                 ``true/false``, ``1/0``, ``yes/no``,
                                 ``on/off`` (case-insensitive)
``str``                          the string, verbatim
``Tuple[X, ...]`` /              a list/tuple of natives; or a JSON array
``Tuple[X, Y]``                  (``"[1.0, 2.0]"``); or comma-joined
                                 scalars without brackets (``"1.0,2.0"``).
                                 Every element is coerced to the declared
                                 element type ``X``, and a fixed-arity
                                 tuple checks its length.
``ShapeSpec`` (``stage5.shape``) a :class:`ShapeSpec`, a ``PeakShape``, a
                                 mapping, or the kind string
                                 (``"gaussian"``)
``clocks``                       a sequence of :class:`ClockSource` or of
(``stage5.spur.clocks``)         mappings; or the JSON array of objects
===============================  =========================================

``None`` is the one value with no string spelling: passing the native ``None``
unsets the field (equivalently :func:`unset_setting`), while the *string*
``"None"`` is only ever the four-character text -- valid for a ``str`` field,
a coercion error for any other. Every settings field is ``Optional`` because
``None`` is precisely how a field says "I am not fixed here; resolve me from a
lower layer", so a string that silently meant ``None`` would make the unset
state unspellable in the other direction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, is_dataclass
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


def _field_hint(cls: type, sub: Optional[str], field: str) -> Tuple[Any, bool]:
    """``(declared type with Optional unwrapped, is_optional)`` for one field.

    ``is_optional`` is what decides whether the field can be unset; every
    settings field is ``Optional`` today (that is how a field says "resolve me
    from a lower layer"), but the check is made rather than assumed.
    Raises ``KeyError`` when the field does not exist on the settings class.
    """
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
        args = get_args(hint)
        optional = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], optional
        return hint, optional
    return hint, False


def _type_name(type_hint: Any) -> str:
    """Readable name for a type hint, for error messages.

    A subscripted generic renders in full (``Tuple[float, float]``), since its
    bare ``__name__`` (``Tuple``) drops the very part an arity or element-type
    error is about.
    """
    if get_origin(type_hint) is not None:
        return str(type_hint).replace("typing.", "")
    name = getattr(type_hint, "__name__", None)
    return str(name) if name else str(type_hint).replace("typing.", "")


def _coerce(type_hint: Any, raw: Any) -> Any:
    """Coerce ``raw`` to the field's declared type, or raise ``ValueError``.

    ``raw`` is either a native Python value of the declared type (the in-process
    form) or a string (the CLI form); see the module docstring's encoding table.
    A value that does not parse as the declared type raises rather than being
    stored as-is -- a silently mistyped element is a value the next run would
    fit against.
    """
    origin = get_origin(type_hint)
    if type_hint is ShapeSpec:
        return ShapeSpec.coerce(raw)
    if origin is tuple and ClockSource in get_args(type_hint):
        # The clock declaration takes the native list-of-mappings or its JSON
        # rendering: settings set stage5.spur.clocks '[{"freq_mhz": 5760}, ...]'
        return coerce_clock_sources(raw)
    if type_hint is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in {"true", "1", "yes", "on"}:
                return True
            if low in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"cannot parse boolean from {raw!r}")
    if type_hint is int:
        return _coerce_int(raw)
    if type_hint is float:
        return _coerce_float(raw)
    if type_hint is str:
        if isinstance(raw, str):
            return raw
        raise ValueError(f"expected a string, got {_type_name(type(raw))}: {raw!r}")
    if origin in (tuple, list):
        return _coerce_sequence(type_hint, origin, raw)
    raise ValueError(
        f"setting a value of type {_type_name(type_hint)!r} "
        f"is not supported by 'settings set'"
    )


def _coerce_int(raw: Any) -> int:
    """Coerce to ``int``. An integral float is accepted (JSON has one number
    type, so ``3.0`` is how a JSON caller spells ``3``); a fractional one is
    not, and neither is a bool."""
    if isinstance(raw, bool):
        raise ValueError(f"expected an integer, got a boolean: {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise ValueError(f"expected an integer, got {raw!r}")
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            raise ValueError(f"cannot parse an integer from {raw!r}") from None
    raise ValueError(f"cannot parse an integer from {raw!r}")


def _coerce_float(raw: Any) -> float:
    """Coerce to ``float``; a bool is not a number here."""
    if isinstance(raw, bool):
        raise ValueError(f"expected a number, got a boolean: {raw!r}")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            raise ValueError(f"cannot parse a number from {raw!r}") from None
    raise ValueError(f"cannot parse a number from {raw!r}")


def _split_sequence(raw: Any) -> List[Any]:
    """Elements of a sequence value, from any of its accepted encodings.

    Native list/tuple passes through. A bracketed string is read as a JSON
    array (the natural rendering for a JSON-speaking caller), falling back to a
    comma split of its contents when it is not valid JSON; an unbracketed
    string is comma-joined scalars, the CLI's shorthand.
    """
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if not isinstance(raw, str):
        raise ValueError(f"expected a sequence, got {_type_name(type(raw))}: {raw!r}")
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Bracketed but not JSON -- the hand-typed ``[K, Ka]`` form. Treat
            # the brackets as decoration and comma-split, so element coercion
            # reports the offending element rather than a JSON syntax position.
            text = text[1:-1]
        else:
            if isinstance(parsed, list):
                return parsed
            raise ValueError(f"expected a JSON array, got {raw!r}")
    return [p.strip() for p in text.split(",") if p.strip()]


def _coerce_sequence(type_hint: Any, origin: Any, raw: Any) -> Any:
    """Coerce to a ``tuple``/``list`` field, element-typed and arity-checked."""
    items = _split_sequence(raw)
    args = [a for a in get_args(type_hint) if a is not Ellipsis]
    homogeneous = origin is list or Ellipsis in get_args(type_hint) or len(args) <= 1
    if not homogeneous and len(items) != len(args):
        raise ValueError(
            f"expected {len(args)} value(s) for a "
            f"{_type_name(type_hint)} field, got {len(items)}: {raw!r}"
        )
    coerced: List[Any] = []
    for index, item in enumerate(items):
        if not args:
            coerced.append(_coerce_scalar(item))
            continue
        elem_type = args[0] if homogeneous else args[index]
        try:
            coerced.append(_coerce(elem_type, item))
        except ValueError as e:
            raise ValueError(f"element {index} of {raw!r}: {e}") from None
    return tuple(coerced) if origin is tuple else coerced


def _coerce_scalar(raw: Any) -> Any:
    """Best-effort scalar coercion for an *untyped* sequence element (a bare
    ``tuple``/``list`` annotation, which no settings field currently uses)."""
    if not isinstance(raw, str):
        return raw
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
def set_setting(file_path: Union[str, Path], knob: str, value: Any) -> SetResult:
    """Persist ``value`` for ``knob`` into the ``.ftmw`` and invalidate the
    affected stage and everything downstream.

    ``knob`` is a dotted settings path (``stage2.window_mhz`` /
    ``stage2b.gaussian.snr_min`` / ``stage5.shape``). ``value`` is either a
    native Python value of the field's declared type or a string in one of the
    encodings the module docstring tabulates; it is coerced to that type, and a
    value that does not parse raises ``ValueError`` without touching the file.
    ``None`` unsets the field (see :func:`unset_setting`). The FT is unapodized
    and native-length, so there are no FT apodization knobs to set.
    """
    path = str(file_path)
    prefix, sub, field = _split_knob(knob)

    if prefix == "stage1":
        return _set_stage1(path, knob, field, value)

    spec = _MUT_SPECS.get(prefix)
    if spec is None:
        raise ValueError(
            f"unknown settings stage {prefix!r} in knob {knob!r}; settable "
            f"stages are {sorted(_MUT_SPECS)} (and stage1 windowing knobs)"
        )
    try:
        field_type, optional = _field_hint(spec.cls, sub, field)
    except KeyError:
        raise ValueError(
            f"unknown setting {knob!r}; no such field on {prefix} settings"
        ) from None
    coerced = _coerce_or_unset(knob, field_type, optional, value)

    settings = spec.load(path) or spec.cls()
    _assign(settings, sub, field, coerced)
    spec.save(path, settings)

    invalidated = _invalidate_inclusive(path, spec.own_stages)
    return SetResult(path=knob, value=coerced, invalidated=invalidated)


def unset_setting(file_path: Union[str, Path], knob: str) -> SetResult:
    """Clear ``knob``'s persisted value, restoring the resolver's own layers.

    The inverse of :func:`set_setting`: the field goes back to ``None``, so the
    next run resolves it from the preset / recommended / hard-default chain
    instead of the value the file had fixed. Stages are invalidated exactly as
    they are for a set, since the effective value changes either way. The
    returned :class:`SetResult` carries ``value=None``.
    """
    return set_setting(file_path, knob, None)


def _coerce_or_unset(knob: str, field_type: Any, optional: bool, value: Any) -> Any:
    """Coerce ``value``, treating ``None`` as the unset request."""
    if value is None:
        if not optional:
            raise ValueError(
                f"{knob!r} is not optional and cannot be unset "
                f"(declared {_type_name(field_type)})"
            )
        return None
    return _coerce(field_type, value)


def _set_stage1(path: str, knob: str, field: str, value: Any) -> SetResult:
    """Persist a Stage 1 windowing knob into ``ft_processing`` and invalidate
    every downstream stage (the FT is recomputed on demand from these settings)."""
    from ..stage1_impl import _persist_ft_settings, _resolve_settings

    try:
        field_type, optional = _field_hint(ft_mod.FTSettings, None, field)
    except KeyError:
        raise ValueError(
            f"unknown setting {knob!r}; no such field on stage1 FT settings"
        ) from None
    coerced = _coerce_or_unset(knob, field_type, optional, value)

    resolved = _resolve_settings(path, None)
    setattr(resolved, field, coerced)
    # _persist_ft_settings rewrites ft_processing and invalidates every
    # FT-dependent stage when the record changes.
    completed_before = _completed_stages(path)
    _persist_ft_settings(path, resolved)
    invalidated = tuple(sorted(completed_before - _completed_stages(path)))
    return SetResult(path=knob, value=coerced, invalidated=invalidated)


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
