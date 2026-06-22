"""
Shared resolve/preset/round-trip scaffolding for the sub-block stage settings.

The Stage 2b/3/4/5 settings dataclasses
(:class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`,
:class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`,
:class:`~ftmwpipeline.core.window_planning_settings.WindowPlanningSettings`,
:class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`) all share the
same architecture: a top-level dataclass holding one sub-dataclass per HDF5
subgroup / YAML block, every field ``Optional`` with ``None`` meaning *unset*,
a four-layer resolution chain (``explicit > persisted > preset > recommended >
hard default``), and a ``__None__``-sentinel dict round-trip driving both the
HDF5 attrs and the sparse YAML preset interchange.

The structural pieces (the layer walk, the sub-block merge, the attrs/YAML loops,
the preset file resolution) live here once, parameterized by the owning settings
class, its ordered ``sub_names``, its nested ``hard_defaults`` map, and per-module
value codecs (``encode``/``decode``/``yaml_encode``/``yaml_coerce`` callables that
take ``(field_name, value)``). The only per-module specializations left in the
settings modules are the codecs (tuple coercion, the ``PeakShape``/``ClockSource``
encodings) and the preset-wrapper block extraction.

The ``__None__`` marker mirrors the one used by ``io.fid_serialization`` for
optional HDF5 attrs. This module is dependency-free within the package (stdlib +
PyYAML) so it can be imported from ``core`` without cycles.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type, TypeVar, Union, cast

import yaml  # type: ignore[import-untyped]

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
NONE = "__None__"

# (field_name, value) -> form  /  (field_name, raw) -> value
Codec = Callable[[str, Any], Any]

# A settings dataclass instance (the top-level or any sub-block).
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Default value codecs (None <-> __None__, bytes decode); identity otherwise
# ---------------------------------------------------------------------------
def default_encode(field_name: str, value: Any) -> Any:
    """Encode a field value for the attrs/dict form (``None`` -> ``__None__``)."""
    return NONE if value is None else value


def default_decode(field_name: str, value: Any) -> Any:
    """Inverse of :func:`default_encode` (``__None__`` -> ``None``, bytes decode)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value == NONE:
        return None
    return value


def identity_coerce(field_name: str, value: Any) -> Any:
    """YAML round-trip pass-through (no per-field coercion)."""
    return value


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def first_set_field(name: str, *layers: Any) -> Any:
    """Walk layers left-to-right, returning the first non-``None`` field value."""
    for layer in layers:
        if layer is None:
            continue
        value = getattr(layer, name)
        if value is not None:
            return value
    return None


def resolve_sub(
    sub_name: str,
    settings_cls: type,
    hard_defaults: Dict[str, Dict[str, Any]],
    layers: Sequence[Any],
) -> Any:
    """Per-sub-dataclass field-merge with hard-default fallback."""
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    template = getattr(settings_cls(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = first_set_field(f.name, *sub_layers)
        if value is None:
            value = hard_defaults.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def fill_resolved_subblocks(
    merged: T,
    settings_cls: type,
    sub_names: Sequence[str],
    hard_defaults: Dict[str, Dict[str, Any]],
    layers: Sequence[Any],
) -> T:
    """Resolve every sub-block on *merged* in place and return it."""
    for sub_name in sub_names:
        setattr(
            merged, sub_name, resolve_sub(sub_name, settings_cls, hard_defaults, layers)
        )
    return merged


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip (drives both HDF5 and YAML serialization)
# ---------------------------------------------------------------------------
def sub_to_attrs(sub: Any, encode: Codec) -> Dict[str, Any]:
    return {f.name: encode(f.name, getattr(sub, f.name)) for f in fields(sub)}


def subblocks_to_attrs(
    settings: Any, sub_names: Sequence[str], encode: Codec
) -> Dict[str, Any]:
    """Nested attrs dict (one key per sub-dataclass)."""
    return {sn: sub_to_attrs(getattr(settings, sn), encode) for sn in sub_names}


def sub_from_attrs(cls: type, attrs: Dict[str, Any], decode: Codec) -> Any:
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in attrs:
            continue
        kwargs[f.name] = decode(f.name, attrs[f.name])
    return cls(**kwargs)


def subblocks_from_attrs(
    settings: T,
    settings_cls: type,
    sub_names: Sequence[str],
    attrs: Dict[str, Any],
    decode: Codec,
) -> T:
    """Fill *settings*'s sub-blocks from a nested attrs dict (tolerant of gaps)."""
    for sub_name in sub_names:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(settings_cls(), sub_name)
        setattr(settings, sub_name, sub_from_attrs(type(template), sub_attrs, decode))
    return settings


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def yaml_sub_to_mapping(sub: Any, yaml_encode: Codec) -> Dict[str, Any]:
    """YAML view: drop ``None`` fields entirely (presets are sparse)."""
    out: Dict[str, Any] = {}
    for f in fields(sub):
        value = getattr(sub, f.name)
        if value is None:
            continue
        out[f.name] = yaml_encode(f.name, value)
    return out


def subblocks_to_yaml_dict(
    settings: Any,
    sub_names: Sequence[str],
    yaml_encode: Codec,
    *,
    top: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    out: Dict[str, Any] = dict(top) if top else {}
    for sub_name in sub_names:
        sub_dict = yaml_sub_to_mapping(getattr(settings, sub_name), yaml_encode)
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def subblocks_from_yaml_dict(
    settings: T,
    settings_cls: type,
    sub_names: Sequence[str],
    data: Optional[Dict[str, Any]],
    yaml_coerce: Codec,
    *,
    allowed_top: Sequence[str] = (),
) -> T:
    """Populate *settings*'s sub-blocks from a YAML-shaped mapping.

    Unknown sub-block fields and unknown top-level keys raise ``ValueError``
    so typos surface loudly. ``allowed_top`` lists extra top-level keys a
    caller handles itself (e.g. Stage 5's ``shape``) on top of the always-
    accepted ``name``/``description`` preset-metadata keys. The caller is
    responsible for applying those extra keys before/after this call.
    """
    for sub_name in sub_names:
        if sub_name not in data:  # type: ignore[operator]
            continue
        block = data[sub_name]  # type: ignore[index]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(settings_cls(), sub_name)
        valid_names = {f.name for f in fields(template)}
        unknown = set(block) - valid_names
        if unknown:
            raise ValueError(
                f"unknown {sub_name!r} fields in preset: {sorted(unknown)} "
                f"(valid: {sorted(valid_names)})"
            )
        kwargs = {k: yaml_coerce(k, v) for k, v in block.items()}
        setattr(settings, sub_name, type(template)(**kwargs))
    allowed = set(sub_names) | {"name", "description"} | set(allowed_top)
    extra_top = set(data) - allowed  # type: ignore[arg-type]
    if extra_top:
        raise ValueError(
            f"unknown top-level preset keys: {sorted(extra_top)} "
            f"(allowed: {sorted(allowed)})"
        )
    return settings


def load_yaml_source(source: Union[str, Path]) -> Any:
    """Load a YAML document from a file path or an inline string."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text()
    else:
        text = str(source)
    return yaml.safe_load(text)


def dump_yaml(yaml_dict: Dict[str, Any]) -> str:
    """Serialize a sparse settings dict to a YAML string."""
    text: Any = yaml.safe_dump(yaml_dict, sort_keys=False, default_flow_style=False)
    return cast(str, text)


# ---------------------------------------------------------------------------
# Preset resolution (bare name vs filesystem path)
# ---------------------------------------------------------------------------
def looks_like_path(name_or_path: Union[str, Path]) -> bool:
    """Heuristic: does ``name_or_path`` reference a file rather than a bare name?

    A bare preset name is a single identifier (e.g. ``instrument_bc_2638``)
    that resolves against the packaged ``ftmwpipeline.presets`` resources.
    Anything else -- a path with separators, a string ending in ``.yaml`` /
    ``.yml``, or an absolute path -- gets loaded directly.
    """
    if isinstance(name_or_path, Path):
        return True
    s = str(name_or_path)
    return ("/" in s) or ("\\" in s) or s.endswith((".yaml", ".yml"))


def read_preset_root(name_or_path: Union[str, Path]) -> Dict[str, Any]:
    """Resolve a preset by bare name or path and return its parsed YAML root.

    Bare names resolve against the packaged ``ftmwpipeline.presets`` resources;
    paths load directly. Raises ``FileNotFoundError`` for a missing path or an
    unknown packaged name, and ``ValueError`` if the document root is not a
    mapping.
    """
    if looks_like_path(name_or_path):
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"preset file not found: {path}")
        text = path.read_text()
    else:
        from importlib.resources import files

        candidate = files("ftmwpipeline.presets") / f"{name_or_path}.yaml"
        if not candidate.is_file():
            available = sorted(
                p.name[:-5]
                for p in files("ftmwpipeline.presets").iterdir()
                if p.name.endswith(".yaml")
            )
            raise FileNotFoundError(
                f"no packaged preset named {name_or_path!r}; " f"available: {available}"
            )
        text = candidate.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"preset YAML root must be a mapping; got {type(data)} from "
            f"{name_or_path}"
        )
    return data


def extract_stage_block(
    data: Dict[str, Any], block_key: str, name_or_path: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """Return the named stage block (with ``name``/``description`` copied in).

    Returns ``None`` when the block is absent (a stage-spanning preset that
    omits this stage), so the caller can hand back an empty settings bundle.
    Raises ``ValueError`` if the block is present but not a mapping.
    """
    inner = data.get(block_key)
    if inner is None:
        return None
    if not isinstance(inner, dict):
        raise ValueError(
            f"preset {block_key!r} block must be a mapping; got {type(inner)} "
            f"from {name_or_path}"
        )
    block = dict(inner)
    for meta in ("name", "description"):
        if meta in data and meta not in block:
            block[meta] = data[meta]
    return block


def load_subblock_preset(
    settings_cls: Type[T],
    block_key: str,
    name_or_path: Union[str, Path],
    from_yaml_dict: Callable[[Optional[Dict[str, Any]]], T],
) -> T:
    """Load a single-stage-block preset, returning an empty bundle if absent.

    The common ``load_preset`` body for the sub-block settings modules that
    wrap their settings in exactly one named ``stageN:`` block. Stage 5's
    loader (legacy ``fit:`` wrapper, flat fallback) is bespoke and calls
    :func:`read_preset_root` directly instead.
    """
    data = read_preset_root(name_or_path)
    block = extract_stage_block(data, block_key, name_or_path)
    if block is None:
        return settings_cls()
    return from_yaml_dict(block)


def any_field_set(settings: Any, sub_names: Sequence[str]) -> bool:
    """True if any field across the named sub-blocks is set (non-``None``)."""
    for sub_name in sub_names:
        sub = getattr(settings, sub_name)
        if any(getattr(sub, f.name) is not None for f in fields(sub)):
            return True
    return False


__all__ = [
    "NONE",
    "Codec",
    "default_encode",
    "default_decode",
    "identity_coerce",
    "first_set_field",
    "resolve_sub",
    "fill_resolved_subblocks",
    "sub_to_attrs",
    "subblocks_to_attrs",
    "sub_from_attrs",
    "subblocks_from_attrs",
    "yaml_sub_to_mapping",
    "subblocks_to_yaml_dict",
    "subblocks_from_yaml_dict",
    "load_yaml_source",
    "dump_yaml",
    "looks_like_path",
    "read_preset_root",
    "extract_stage_block",
    "load_subblock_preset",
    "any_field_set",
]
