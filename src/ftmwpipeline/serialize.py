"""JSON serialization for machine-contract payloads.

:func:`to_jsonable` is the one converter every contract result and domain type
goes through (and that CLI ``--format json`` routes through). It applies the
contract's single missing-value rule (``dev-docs/CONTRACT_STRATEGY.md``,
§Missing values):

- a field whose value is :class:`~ftmwpipeline.contract.Absent` is written as
  ``null`` with a sibling ``"<field>_absent": "not_run" | "undefined"``; the
  sibling is omitted when the value is present;
- columns that can be absent are built with :func:`absent_column`, which gives
  ``nan`` (or the column's fill) plus a ``uint8`` ``<column>__status`` column.

- a key named ``"<field>_absent"`` is reserved for that sibling: a source
  object that supplies one is rejected (:class:`ValueError`);
- JSON has no ``nan``/``inf``: a non-finite float that is the value of a named
  field (mapping key or dataclass field) is written as ``null`` with sibling
  ``"<field>_absent": "undefined"``; inside a list it is written as ``null``
  (a declared status column carries the reason); with no field or list to
  hold it (the top level) it raises :class:`TypeError`, like a nameless
  :class:`~ftmwpipeline.contract.Absent`. Arrays written through a sink (the
  CLI's ``.npy`` files) keep their ``nan`` values.

Conversions: ``None``/``bool``/``int``/``str`` pass through; enums become their
``.value`` (checked first, so a ``str``- or ``int``-valued enum is never
written as ``"Cls.MEMBER"``; also for mapping keys); finite floats (Python and
numpy) become Python floats; numpy scalars become Python scalars; complex
numbers become ``{"real": x, "imag": y}`` (each part a named field);
dataclasses become dicts of their fields (or of the items their
``__ftmw_items__()`` returns, where the wire form differs from the fields, as
for :class:`~ftmwpipeline.contract.PipelineWarning`); :class:`~pathlib.Path` a string;
tuples, lists and sets lists; ``date``/``datetime`` ISO-8601 strings; a
:class:`~ftmwpipeline.file_manager.PipelineFileError` its ``to_dict()``.
Anything else raises :class:`TypeError` -- there is no ``str()`` fallback.

Schema stamping: pass ``schema=`` for the envelope, or give a dataclass a
``__ftmw_schema__`` class attribute so every instance is stamped wherever it
appears. The ``"schema"`` key comes first.

Arrays: by default numpy arrays are written inline as (nested) lists. Pass an
``arrays`` sink to take them out of the JSON instead -- the CLI uses
:class:`ArrayCollector` to write each one as ``.npy`` to ``--output`` and puts
the file name in the JSON in its place.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import math
from pathlib import PurePath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from .contract import SCHEMA_NAME_RE, STATUS_PRESENT, Absent
from .file_manager import PipelineFileError

#: Location of a value inside the payload: the keys / list indices from the root.
JsonPath = Tuple[Union[str, int], ...]

#: An array sink: given an array's location and the array, return what the JSON
#: holds in its place (e.g. a file name). Called for every ``ndarray`` with
#: ``ndim >= 1``; 0-d arrays are always written inline as scalars.
ArraySink = Callable[[JsonPath, np.ndarray], Any]

#: Suffix of the sibling key that carries an absence reason.
ABSENT_SUFFIX = "_absent"

#: Suffix of a columnar status column.
STATUS_SUFFIX = "__status"


def to_jsonable(
    obj: Any,
    *,
    schema: Optional[str] = None,
    arrays: Optional[ArraySink] = None,
) -> Any:
    """Convert ``obj`` to a structure ``json.dumps`` accepts, per the contract.

    Parameters
    ----------
    obj
        A contract result, domain type, or plain container of them.
    schema : str, optional
        Schema name (``ftmw/<payload>@<n>``) to stamp on the top-level object as
        its first key. The converted ``obj`` must be a JSON object.
    arrays : callable, optional
        Array sink (see :data:`ArraySink`). ``None`` writes arrays inline.

    Returns
    -------
    Any
        ``None``, ``bool``, ``int``, ``float`` (finite), ``str``, ``list`` or
        ``dict`` with ``str`` keys, recursively -- or whatever the sink returned
        in an array's place.

    Raises
    ------
    TypeError
        For an unsupported type, a non-string-able mapping key, an
        :class:`Absent` that is not the value of a named field (top level or a
        list element) -- its reason could not survive, so build a column with
        :func:`absent_column` instead -- or a non-finite float at the top level.
    ValueError
        For a malformed schema name, a conflicting ``"schema"`` key, or a
        source key ending in ``"_absent"`` (reserved for absence siblings).
    """
    result = _convert(obj, (), arrays)
    if schema is not None:
        result = _stamp(result, schema)
    return result


def absent_column(
    values: Iterable[Any],
    *,
    dtype: npt.DTypeLike = np.float64,
    fill: Any = math.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the columnar form of a column that can be absent.

    Each element is a value or an :class:`Absent`. Absent elements are written
    as ``fill`` (``nan`` by default; pass the column's fill for an integer
    column) and flagged in the returned status array.

    Returns
    -------
    (values, status)
        ``values`` as ``dtype``; ``status`` as ``uint8``: ``0`` present, ``1``
        not run, ``2`` undefined.
    """
    items = list(values)
    status = np.full(len(items), STATUS_PRESENT, dtype=np.uint8)
    out: List[Any] = []
    for i, item in enumerate(items):
        if isinstance(item, Absent):
            status[i] = item.status
            out.append(fill)
        else:
            out.append(item)
    return np.asarray(out, dtype=dtype), status


def with_status_columns(
    columns: Mapping[str, Iterable[Any]],
    *,
    absent_capable: Sequence[str],
    fills: Optional[Mapping[str, Any]] = None,
    dtypes: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> Dict[str, np.ndarray]:
    """Columnar table with a ``<column>__status`` column per absent-capable column.

    Every column named in ``absent_capable`` goes through :func:`absent_column`
    and gains its status column (even when nothing in it is absent: the status
    column's presence is declared, not data-dependent). Other columns are
    converted with ``np.asarray`` and must not contain :class:`Absent`.
    """
    fills = fills or {}
    dtypes = dtypes or {}
    unknown = set(absent_capable) - set(columns)
    if unknown:
        raise KeyError(f"absent_capable names unknown columns: {sorted(unknown)}")
    out: Dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name in absent_capable:
            vals, status = absent_column(
                values,
                dtype=dtypes.get(name, np.float64),
                fill=fills.get(name, math.nan),
            )
            out[name] = vals
            out[name + STATUS_SUFFIX] = status
        else:
            items = list(values)
            if any(isinstance(v, Absent) for v in items):
                raise TypeError(
                    f"column {name!r} holds Absent but is not declared absent-capable"
                )
            dtype = dtypes.get(name)
            out[name] = np.asarray(items) if dtype is None else np.asarray(items, dtype)
    return out


class ArrayCollector:
    """An :data:`ArraySink` that takes arrays out of the JSON for ``.npy`` output.

    Each array is recorded under a file name derived from its location in the
    payload (keys and indices joined by ``"."``, e.g. ``"samples.npy"``,
    ``"components.peak_ab12.npy"``), optionally prefixed, and that file name
    replaces the array in the JSON. The caller then writes
    ``arrays[name]`` to ``<output>/<name>`` with :func:`numpy.save`.

    Attributes
    ----------
    arrays : dict of str to ndarray
        File name -> array, in encounter order.
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.arrays: Dict[str, np.ndarray] = {}

    def __call__(self, path: JsonPath, array: np.ndarray) -> str:
        stem = ".".join(str(p) for p in path) or "array"
        stem = stem.replace("/", "_").replace("\\", "_")
        name = f"{self.prefix}{stem}.npy"
        n = 1
        while name in self.arrays:
            n += 1
            name = f"{self.prefix}{stem}-{n}.npy"
        self.arrays[name] = array
        return name


# --------------------------------------------------------------------------
# Implementation
# --------------------------------------------------------------------------


def _stamp(result: Any, schema: str) -> Dict[str, Any]:
    if not SCHEMA_NAME_RE.match(schema):
        raise ValueError(f"malformed schema name: {schema!r}")
    if not isinstance(result, dict):
        raise TypeError(
            f"schema {schema!r} can only stamp a JSON object, "
            f"not {type(result).__name__}"
        )
    existing = result.get("schema")
    if existing is not None and existing != schema:
        raise ValueError(f"payload already stamped {existing!r}, not {schema!r}")
    stamped: Dict[str, Any] = {"schema": schema}
    stamped.update((k, v) for k, v in result.items() if k != "schema")
    return stamped


def _is_nonfinite(x: Any) -> bool:
    """True for a real float (Python, numpy, or 0-d float array) that is nan/inf."""
    if isinstance(x, np.ndarray):
        if x.ndim != 0 or x.dtype.kind != "f":
            return False
        x = x.item()
    if isinstance(x, (float, np.floating)):
        return not math.isfinite(float(x))
    return False


def _key(k: Any) -> str:
    # Enum before str/int: a str- or int-valued Enum key is written as its value.
    if isinstance(k, Absent):
        raise TypeError("Absent cannot be used as a JSON object key")
    if isinstance(k, enum.Enum):
        return _key(k.value)
    if isinstance(k, str):
        return str(k)
    if isinstance(k, (bool, np.bool_)):
        return "true" if k else "false"
    if isinstance(k, (int, np.integer)):
        return str(int(k))
    if isinstance(k, PurePath):
        return str(k)
    raise TypeError(f"cannot use {type(k).__name__} as a JSON object key")


def _object(
    items: Iterable[Tuple[Any, Any]], path: JsonPath, arrays: Optional[ArraySink]
) -> Dict[str, Any]:
    converted: List[Tuple[str, Any, Optional[Absent]]] = []
    keys = set()
    for k, v in items:
        key = _key(k)
        if key in keys:
            raise ValueError(f"duplicate key {key!r} at {_where(path)}")
        keys.add(key)
        if isinstance(v, Absent):
            converted.append((key, None, v))
        elif _is_nonfinite(v):
            # A named field with no finite value: computed, but undefined.
            converted.append((key, None, Absent.UNDEFINED))
        else:
            converted.append((key, _convert(v, path + (key,), arrays), None))
    # "<field>_absent" is reserved for the absence sibling this function
    # writes; a source object may never supply one.
    for key in keys:
        if key.endswith(ABSENT_SUFFIX):
            base = key[: -len(ABSENT_SUFFIX)]
            if base in keys:
                raise ValueError(
                    f"key {key!r} at {_where(path)} is reserved for the absence "
                    f"reason of {base!r}; the source object may not supply it"
                )
            raise ValueError(
                f"key {key!r} at {_where(path)} is reserved for an absence "
                f"reason, but the payload has no field {base!r}"
            )
    out: Dict[str, Any] = {}
    for key, value, reason in converted:
        out[key] = value
        if reason is not None:
            out[key + ABSENT_SUFFIX] = reason.value
    return out


def _where(path: JsonPath) -> str:
    return "/" + "/".join(str(p) for p in path)


def _element(obj: Any, path: JsonPath, arrays: Optional[ArraySink]) -> Any:
    """Convert a list element: a non-finite float is ``null`` (no field to
    carry a reason; a declared status column carries it)."""
    if _is_nonfinite(obj):
        return None
    return _convert(obj, path, arrays)


def _is_complex_ft(obj: Any) -> bool:
    from .core.data_structures import ComplexFT

    return isinstance(obj, ComplexFT)


def _npy_safe(arr: np.ndarray, path: JsonPath) -> np.ndarray:
    """An array a sink can save without pickling.

    An object array of strings (a text column) becomes fixed-width unicode, which
    ``.npy`` stores natively; any other object array has no self-describing
    form and is refused.
    """
    if arr.dtype.kind != "O":
        return arr
    if all(isinstance(v, str) for v in arr.flat):
        return arr.astype(str)
    raise TypeError(
        f"array at {_where(path)} holds non-string objects; it has no .npy form"
    )


def _convert(obj: Any, path: JsonPath, arrays: Optional[ArraySink]) -> Any:
    # Order matters: Absent before generic Enum, Enum before str/int/float (a
    # str- or int-valued Enum is written as its value), bool before int, numpy
    # before the Python builtins they subclass.
    if obj is None:
        return None
    if isinstance(obj, Absent):
        raise TypeError(
            f"Absent at {_where(path)} is not the value of a named field; its "
            "reason cannot be carried. Use a field, or absent_column() for arrays."
        )
    if isinstance(obj, enum.Enum):
        return _convert(obj.value, path, arrays)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            raise TypeError(
                f"non-finite float at {_where(path)} is not the value of a named "
                "field; its reason cannot be carried. Use a field (written as "
                "null with an 'undefined' sibling) or an array."
            )
        return x
    if isinstance(obj, (complex, np.complexfloating)):
        c = complex(obj)
        return _object((("real", c.real), ("imag", c.imag)), path, arrays)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _convert(obj.item(), path, arrays)
        if arrays is not None:
            return arrays(path, _npy_safe(obj, path))
        if obj.dtype.kind == "O" and any(isinstance(v, Absent) for v in obj.flat):
            raise TypeError(
                f"array at {_where(path)} holds Absent; use absent_column()"
            )
        return [_element(v, path + (i,), None) for i, v in enumerate(obj.tolist())]
    if isinstance(obj, PipelineFileError):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # A dataclass whose wire form differs from its fields (PipelineWarning
        # flattens ``details``) supplies its items through ``__ftmw_items__``.
        wire_items = getattr(obj, "__ftmw_items__", None)
        items: Iterable[Tuple[str, Any]] = (
            wire_items()
            if callable(wire_items)
            else ((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))
        )
        result = _object(items, path, arrays)
        schema = getattr(type(obj), "__ftmw_schema__", None)
        return _stamp(result, schema) if isinstance(schema, str) else result
    if isinstance(obj, Mapping):
        return _object(obj.items(), path, arrays)
    if _is_complex_ft(obj):
        # ComplexFT is a plain class, not a dataclass: its contract fields are
        # the two arrays and the metadata mapping.
        return _object(
            (
                ("freq_array", obj.freq_array),
                ("complex_spectrum", obj.complex_spectrum),
                ("metadata", obj.metadata),
            ),
            path,
            arrays,
        )
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_element(v, path + (i,), arrays) for i, v in enumerate(obj)]
    if isinstance(obj, (set, frozenset)):
        converted = [_element(v, path, arrays) for v in obj]
        try:
            return sorted(converted)
        except TypeError:
            raise TypeError(
                f"set at {_where(path)} has no canonical order; convert it first"
            ) from None
    raise TypeError(f"{type(obj).__name__} at {_where(path)} is not JSON-able")


__all__ = [
    "ABSENT_SUFFIX",
    "STATUS_SUFFIX",
    "ArrayCollector",
    "ArraySink",
    "JsonPath",
    "absent_column",
    "to_jsonable",
    "with_status_columns",
]
