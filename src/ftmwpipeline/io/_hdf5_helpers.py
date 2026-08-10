"""Shared HDF5 (de)serialization primitives for the ``io.*_serialization`` modules.

Small helpers factored out of the per-stage serializers so the same arithmetic
and conventions are written once: JSON-attribute parsing, NaN-sentinel float
coercion (``Optional[float]`` <-> stored ``float``), group wiping, the
per-stage attribute header stamp, and the bulk column-read primitives the
read-only accessors (``io.*_serialization.read_*_columns``) share.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Bulk column reads (the read-only accessor path)
# ---------------------------------------------------------------------------
#
# The full ``load_*_from_hdf5`` loaders rebuild whole object graphs and are the
# right tool for curation. A consumer that only wants a few columns pays that
# cost for nothing -- the expense is h5py's per-item Python overhead, paid
# thousands of times, not the bytes. These primitives back the narrow
# ``read_*_columns`` accessors: each requested column is one whole-dataset (or
# one attribute) read, and nothing else on the group is touched.
#
# A column is declared as ``(dtype, fill)``. ``dtype`` is a numpy dtype string,
# or ``"str"`` for a variable-length UTF-8 column (returned as an object array
# of ``str``), or ``"bool"``. ``fill`` is the value substituted when the column
# is absent -- older files predating it -- or :data:`REQUIRED`, which makes an
# absent column a loud :class:`ValueError` exactly as the full loader would.


class _Required:
    """Sentinel: this column must be present on disk."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "REQUIRED"


#: Marks a column spec whose column must be present (see :data:`ColumnSpec`).
REQUIRED = _Required()

#: ``(dtype, fill_or_REQUIRED)`` -- see the module note above.
ColumnSpec = Tuple[str, Any]


def resolve_column_selection(
    columns: Optional[Sequence[str]],
    available: Sequence[str],
    *,
    table: str,
) -> List[str]:
    """Validate a caller's column selection against *available*.

    ``None`` selects every column, in the canonical order. An explicit
    selection keeps the caller's order (so a CSV dump is column-ordered as
    asked) and rejects unknown names loudly, listing what is valid -- a typo
    must not silently yield a narrower table.
    """
    if columns is None:
        return list(available)
    requested = list(columns)
    unknown = [c for c in requested if c not in available]
    if unknown:
        raise ValueError(
            f"unknown column(s) {unknown} for table {table!r}; "
            f"available columns are {list(available)}"
        )
    if not requested:
        raise ValueError(f"no columns requested for table {table!r}")
    return requested


def decode_str_array(raw: Any) -> np.ndarray:
    """Decode an HDF5 string column into an object array of ``str``."""
    return np.array(
        [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw],
        dtype=object,
    )


def empty_column(dtype: str, n: int, fill: Any = None) -> np.ndarray:
    """An ``n``-element column of *dtype*, filled with *fill*."""
    if dtype == "str":
        return np.full(n, "" if fill is None else str(fill), dtype=object)
    if dtype == "bool":
        return np.full(n, False if fill is None else bool(fill), dtype=bool)
    return np.full(n, 0 if fill is None else fill, dtype=dtype)


def coerce_column(raw: Any, dtype: str) -> np.ndarray:
    """Coerce a raw h5py column read to this module's column conventions."""
    if dtype == "str":
        return decode_str_array(raw)
    if dtype == "bool":
        return np.asarray(raw).astype(bool)
    return np.asarray(raw).astype(dtype, copy=False)


def read_dataset_column(
    h5_group: h5py.Group,
    name: str,
    spec: ColumnSpec,
    n_rows: Optional[int],
    *,
    where: str,
    dataset: Optional[str] = None,
) -> np.ndarray:
    """Read one whole dataset column from *h5_group*, or synthesize its fill.

    *n_rows* is the row count every column of the group must agree on; pass
    ``None`` for the first (anchor) column read, whose own length defines it.

    *dataset* is the on-disk name when it differs from the public column name
    (a few stages stored their columns under plural names). Errors quote both,
    so a reader chasing the message can find the dataset in the file.

    Raises ``ValueError`` when a :data:`REQUIRED` column is absent, when a
    present column's length disagrees with *n_rows*, or when an anchor column is
    absent (its fill has no length to stand in for) -- the same loud-validation
    contract the full loaders hold. Membership is probed by catching the miss
    rather than testing first, so the common hit path is one h5py lookup, not
    two; on a file with hundreds of window groups that halves the traffic.
    """
    dtype, fill = spec
    on_disk = dataset if dataset is not None else name
    label = name if on_disk == name else f"{name} (dataset {on_disk!r})"
    try:
        h5_dataset = h5_group[on_disk]
    except KeyError:
        if fill is REQUIRED:
            raise ValueError(f"{where} missing required column {label!r}") from None
        if n_rows is None:
            raise ValueError(
                f"{where} missing column {label!r}, which is needed to determine "
                f"the row count"
            ) from None
        return empty_column(dtype, n_rows, fill)
    raw = h5_dataset[:]
    if n_rows is not None and len(raw) != n_rows:
        raise ValueError(
            f"{where} column {label!r} has length {len(raw)}, expected {n_rows}"
        )
    return coerce_column(raw, dtype)


_MISSING = object()


def read_attr_value(
    h5_group: h5py.Group,
    name: str,
    spec: ColumnSpec,
    *,
    where: str,
) -> Any:
    """Read one scalar attribute from *h5_group*, or return its fill."""
    dtype, fill = spec
    raw = h5_group.attrs.get(name, _MISSING)
    if raw is _MISSING:
        if fill is REQUIRED:
            raise ValueError(f"{where} missing required attribute {name!r}")
        return fill
    if dtype == "str":
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    if dtype == "bool":
        return bool(raw)
    return np.asarray(raw).astype(dtype).item()


def stack_columns(
    chunks: Dict[str, List[np.ndarray]],
    specs: Dict[str, ColumnSpec],
    keep: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Concatenate per-group column chunks into whole-table columns."""
    out: Dict[str, np.ndarray] = {}
    for name in keep:
        pieces = chunks[name]
        dtype = specs[name][0]
        if pieces:
            out[name] = np.concatenate(pieces)
        else:
            out[name] = empty_column(dtype, 0)
    return out


def record_row(
    record: Dict[str, Any],
    specs: Dict[str, ColumnSpec],
    keep: Iterable[str],
    rows: Dict[str, List[Any]],
    *,
    where: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON-decoded record's fields to the accumulating *rows*.

    The event logs a fit persists (the conservative add-loop audit, the thaw and
    replan histories, the rescue rounds, the doublet alternatives) are
    JSON-encoded lists of uniform dicts rather than parallel arrays, because
    they are written once and read as a narrative. Presenting them as columns
    costs a JSON parse -- there is no cheaper path to them, and this makes that
    explicit rather than pretending otherwise -- but the result is still a flat
    table a consumer can export.

    *extra* supplies values not in the record itself (the owning ``window_id``,
    say). A field absent from the record takes its declared fill; a
    :data:`REQUIRED` field raises.
    """
    for name in keep:
        dtype, fill = specs[name]
        if extra is not None and name in extra:
            rows[name].append(extra[name])
            continue
        if name in record:
            rows[name].append(record[name])
            continue
        if fill is REQUIRED:
            raise ValueError(f"{where} missing required field {name!r}")
        rows[name].append(fill)


def build_columns(
    rows: Dict[str, List[Any]],
    specs: Dict[str, ColumnSpec],
    keep: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Turn per-row scalar lists (one per group visited) into typed columns."""
    out: Dict[str, np.ndarray] = {}
    for name in keep:
        dtype = specs[name][0]
        values = rows[name]
        if not values:
            out[name] = empty_column(dtype, 0)
        elif dtype == "str":
            out[name] = np.array(values, dtype=object)
        else:
            out[name] = np.array(values, dtype=dtype)
    return out


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
