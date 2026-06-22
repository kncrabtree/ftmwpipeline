"""Shared HDF5 persistence scaffolding for the ``*_settings_serialization`` modules.

Every stage's resolved-settings record lives under ``processing_parameters/<stage>``
and is written the same way: wipe any prior group, stamp ``creation_time`` and an
optional ``preset_name`` audit attr, then dump the field values the settings
dataclass's ``to_attrs`` produced. The only per-stage variation is the *shape* of
that attrs dict -- a flat field map (Stage 2's single estimator) versus one HDF5
subgroup per sub-dataclass (Stages 2b/3/4/5), plus a couple of value-encoding
quirks (Stage 2b's tuple fields, Stage 5's ``shape`` subgroup). Those ride in as
``write_attr`` / ``read_sub_attrs`` / ``extra_top`` callbacks; the open/wipe/stamp
and the present-probe live here once.

A ``to_attrs`` value that is a ``dict`` becomes an HDF5 subgroup (one per
sub-dataclass, or Stage 5's set ``shape``); a scalar value becomes a top-level
attr (every Stage 2 field, or Stage 5's unset-``shape`` sentinel). That single
rule covers all five serializers' write paths.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import h5py


def decode_attr(value: Any) -> Any:
    """Decode an HDF5 attribute value (handles bytes -> str)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _default_write_attr(grp: h5py.Group, name: str, value: Any) -> None:
    grp.attrs[name] = value


def read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    """Decode every attr on *grp* (bytes -> str); the default sub-block reader."""
    return {key: decode_attr(raw) for key, raw in grp.attrs.items()}


def save_settings(
    file_path: str,
    path: str,
    attrs: Dict[str, Any],
    *,
    preset_name: Optional[str] = None,
    write_attr: Callable[[h5py.Group, str, Any], None] = _default_write_attr,
) -> None:
    """Persist a settings *attrs* dict to ``path``, overwriting any prior group.

    ``creation_time`` is always stamped; ``preset_name`` is recorded when given.
    Each ``attrs`` entry whose value is a ``dict`` is written as a subgroup (its
    fields via *write_attr*); scalar-valued entries are written as top-level
    attrs. This covers the flat (Stage 2) and sub-block (Stages 2b/3/4/5) layouts
    and Stage 5's ``shape`` subgroup-or-sentinel.
    """
    with h5py.File(file_path, "a") as h5f:
        if path in h5f:
            del h5f[path]
        grp = h5f.create_group(path)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        for key, value in attrs.items():
            if isinstance(value, dict):
                sub_grp = grp.create_group(key)
                for field_name, field_value in value.items():
                    write_attr(sub_grp, field_name, field_value)
            else:
                grp.attrs[key] = value


def load_subblock_settings(
    file_path: str,
    path: str,
    sub_names: Sequence[str],
    from_attrs: Callable[[Dict[str, Any]], Any],
    *,
    read_sub: Callable[[h5py.Group], Dict[str, Any]] = read_sub_attrs,
    extra_top: Optional[Callable[[h5py.Group, Dict[str, Any]], None]] = None,
) -> Optional[Any]:
    """Load a sub-block settings record from ``path``, or ``None`` if absent.

    Reads one attrs dict per sub-block (missing blocks default to ``{}`` so a
    partial group still loads), optionally pulling extra top-level state first
    via *extra_top* (Stage 5's ``shape``), then hands the assembled nested dict
    to *from_attrs*.
    """
    with h5py.File(file_path, "r") as h5f:
        if path not in h5f:
            return None
        grp = h5f[path]
        attrs_dict: Dict[str, Any] = {}
        if extra_top is not None:
            extra_top(grp, attrs_dict)
        for sub_name in sub_names:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                attrs_dict[sub_name] = read_sub(grp[sub_name])
            else:
                attrs_dict[sub_name] = {}
    return from_attrs(attrs_dict)


def load_flat_settings(
    file_path: str,
    path: str,
    from_attrs: Callable[[Dict[str, Any]], Any],
    *,
    audit_attrs: Sequence[str],
) -> Optional[Any]:
    """Load a flat (single-group) settings record from ``path``, or ``None``.

    Reads every top-level attr except the *audit_attrs* bookkeeping keys and
    hands the field map to *from_attrs* (the Stage 2 layout).
    """
    with h5py.File(file_path, "r") as h5f:
        if path not in h5f:
            return None
        grp = h5f[path]
        attrs_dict: Dict[str, Any] = {
            key: decode_attr(raw)
            for key, raw in grp.attrs.items()
            if key not in audit_attrs
        }
    return from_attrs(attrs_dict)


def settings_block_present(file_path: str, path: str) -> bool:
    """Lightweight: does the file carry a persisted settings group at ``path``?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return path in h5f
    except (OSError, KeyError):
        return False


__all__ = [
    "decode_attr",
    "read_sub_attrs",
    "save_settings",
    "load_subblock_settings",
    "load_flat_settings",
    "settings_block_present",
]
