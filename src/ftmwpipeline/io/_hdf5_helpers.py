"""Shared HDF5 (de)serialization primitives for the ``io.*_serialization`` modules.

Small helpers factored out of the per-stage serializers so the same arithmetic
and conventions are written once: JSON-attribute parsing, NaN-sentinel float
coercion (``Optional[float]`` <-> stored ``float``), group wiping, and the
per-stage attribute header stamp.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import h5py
import numpy as np


def load_json_attr(
    h5_group: h5py.Group, name: str, default: Any, *, label: Optional[str] = None
) -> Any:
    """Parse a JSON-encoded attribute, falling back to ``default`` if absent.

    Raises ``ValueError`` (mentioning the group path, and *label* when given) if
    the attribute is present but not decodable JSON.
    """
    raw = h5_group.attrs.get(name)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        prefix = f"{label} " if label else ""
        raise ValueError(
            f"{prefix}attribute {name!r} on {h5_group.name} is not valid JSON"
        ) from exc


def nan_if_none(value: Optional[float]) -> float:
    """``float(value)``, or a NaN sentinel when *value* is ``None``."""
    if value is None:
        return float("nan")
    return float(value)


def none_if_nan(value: float) -> Optional[float]:
    """Inverse of :func:`nan_if_none`: NaN -> ``None``, finite value otherwise."""
    f = float(value)
    return None if np.isnan(f) else f


def opt_float(d: Dict[str, Any], key: str) -> Optional[float]:
    """``float(d[key])``, or ``None`` when the key is absent or ``None``."""
    v = d.get(key)
    return None if v is None else float(v)


def reset_group(h5_group: h5py.Group, *, attrs: bool = False) -> None:
    """Delete all child links of *h5_group* (and its attrs when ``attrs=True``)."""
    for key in list(h5_group.keys()):
        del h5_group[key]
    if attrs:
        for key in list(h5_group.attrs.keys()):
            del h5_group.attrs[key]


def stamp_stage_header(h5_group: h5py.Group, stage_name: str, **counts: int) -> None:
    """Stamp the standard per-stage header attrs onto *h5_group*.

    Writes the *counts* attrs (e.g. ``n_windows``/``n_peaks``) first, then
    ``creation_time`` and ``stage_name`` — matching the per-stage write order so
    the on-disk attribute set is unchanged.
    """
    for name, value in counts.items():
        h5_group.attrs[name] = int(value)
    h5_group.attrs["creation_time"] = datetime.now().isoformat()
    h5_group.attrs["stage_name"] = stage_name
