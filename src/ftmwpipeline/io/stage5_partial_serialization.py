"""Persistence of a Stage 5 partial fit (``/stage5_partial``).

A cancelled (or callback-failed) Stage 5 run keeps the windows whose whole
per-window pass had run, so the next ``fit run`` refits only the others
(``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation → §Stage 5 partial
fits). The group is a private resume checkpoint, not contract layout: no
accessor reads it as a fit, and every write that discards a fit discards it.

What a resume needs is each finished window's live
:class:`~ftmwpipeline.fitting.plan_execution.WindowOutcome` exactly as the walk
left it -- a dependent reads its ancestors' outcomes, and the post-walk steps
(structural replan, history aggregation, cleanup, conversion, sort) read every
field of it -- plus the window's
:class:`~ftmwpipeline.fitting.plan_execution.WalkRecord`. A ``WindowOutcome`` is
a graph of a closed set of dataclasses, enums, numpy arrays and plain values,
with private attributes the walk stashes (``_center_mhz``, ``_spur_mask``,
``_ck_for_window``, ...) and aliasing the walk relies on (a rescue event is the
same object in the outcome and in the plan-level history, and is refreshed in
place). It is persisted with a small typed graph codec:

- every object is encoded by its type and its instance attributes, and only the
  types in :data:`_DATACLASSES` / :data:`_ENUMS` can be encoded or decoded;
- shared (and cyclic) references are kept: a list, dict, dataclass instance or
  array is written once and referenced thereafter;
- numbers are written as JSON (``repr`` round-trips a float exactly; ``NaN`` /
  ``Infinity`` are kept), numpy arrays and scalars by dtype (numeric dtypes
  only) with their bytes in typed datasets.

Decoding never runs code from the file: it parses JSON, builds instances of the
allow-listed types with ``object.__new__`` and sets their attributes, and makes
numeric arrays. Nothing is pickled. A value outside the allow-list cannot be
written; the window is then not kept (the resume refits it).

Layout::

    /stage5_partial
        provenance        [u1]   UTF-8 JSON: what the resume compares + walk state
        window_ids        [i8]   finished windows, in the order they finished
        n_fitted_peaks    [i8]   each one's fitted-line count (0: emptied)
        windows/
            w<id>/
                graph     [u1]   UTF-8 JSON: the window's encoded graph
                buf<k>    [...]  the graph's array bytes, one dataset per dtype
                .attrs: dtypes (JSON list of the buffers' dtype strings)
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import typing
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import h5py
import numpy as np

from ..core.data_structures import Sideband
from ..core.peak_shape import PeakShape
from ..file_manager import STAGE5_PARTIAL_PATH
from ..fitting.active_ft import PointMap
from ..fitting.doublet_alternative import DoubletAdjudication
from ..fitting.peak_model import ModelPeak
from ..fitting.plan_execution import (
    CarriedWindows,
    FrozenPeak,
    PartialWalk,
    RescueEvent,
    ThawEvent,
    WalkRecord,
    WindowOutcome,
)
from ..fitting.residual_screening import ResidualPeakCandidate
from ..fitting.spur_detection import SpurMaskSpec
from ..fitting.window_fit import (
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    ParameterErrors,
    WindowFitResult,
)

#: Layout version of ``/stage5_partial``. A different version is not resumed
#: from (the resume starts over with ``incomplete_provenance``).
PARTIAL_FORMAT_VERSION = 1

_WINDOWS = "windows"
_PROVENANCE = "provenance"
_WINDOW_IDS = "window_ids"
_N_FITTED = "n_fitted_peaks"

#: The only dataclasses the codec writes and builds, by name.
_DATACLASSES: Dict[str, type] = {
    cls.__qualname__: cls
    for cls in (
        WindowOutcome,
        WalkRecord,
        ConservativeFitResult,
        WindowFitResult,
        ModelPeak,
        ParameterErrors,
        AddStep,
        KnockoutResult,
        FrozenPeak,
        ThawEvent,
        RescueEvent,
        ResidualPeakCandidate,
        DoubletAdjudication,
        SpurMaskSpec,
        PointMap,
    )
}
_DATACLASS_NAMES: Dict[type, str] = {cls: name for name, cls in _DATACLASSES.items()}

#: The only enums the codec writes and builds, by name (members by name).
_ENUMS: Dict[str, type] = {cls.__qualname__: cls for cls in (PeakShape, Sideband)}
_ENUM_NAMES: Dict[type, str] = {cls: name for name, cls in _ENUMS.items()}

#: Numeric dtype kinds an array or numpy scalar may have.
_NUMERIC_KINDS = frozenset("biufc")

#: The private attributes the walk stashes on a dataclass beyond its declared
#: fields (``plan_execution``: the window centre, the spur mask, the per-window
#: fit conditions, the edge-free verdict), with their types. No other
#: undeclared attribute is written or read.
_STASHED: Dict[type, Dict[str, Any]] = {
    WindowOutcome: {
        "_center_mhz": float,
        "_spur_mask": Optional[SpurMaskSpec],
        "_ck_for_window": Dict[str, Any],
        "_acquisition_us": float,
        "_residual_edge_m": int,
        "_n_eff_kind": str,
        "_edge_free_accepted": bool,
    },
}


def _attribute_types(cls: type) -> Dict[str, Any]:
    """Every attribute ``cls`` may carry (declared fields and stashes), typed."""
    hints = typing.get_type_hints(cls)
    types = {f.name: hints.get(f.name, Any) for f in dataclasses.fields(cls)}
    types.update(_STASHED.get(cls, {}))
    return types


#: The attributes, and their types, of each allow-listed dataclass.
_ATTRIBUTES: Dict[type, Dict[str, Any]] = {
    cls: _attribute_types(cls) for cls in _DATACLASSES.values()
}
_ATTRIBUTE_NAMES: Dict[type, FrozenSet[str]] = {
    cls: frozenset(types) for cls, types in _ATTRIBUTES.items()
}


class PartialCodecError(ValueError):
    """A value cannot be written to, or read from, a partial fit."""


# ---------------------------------------------------------------------------
# The graph codec
# ---------------------------------------------------------------------------


class _Encoder:
    """Encode one object graph to JSON-able nodes plus typed array buffers."""

    def __init__(self) -> None:
        self.nodes: List[Any] = []
        self._memo: Dict[int, int] = {}
        # Keep every memoized object alive so its id() is never reused mid-walk.
        self._alive: List[Any] = []
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._sizes: Dict[str, int] = {}

    def buffers(self) -> Tuple[List[str], List[np.ndarray]]:
        names = list(self._buffers)
        return names, [
            (
                np.concatenate(self._buffers[n])
                if self._buffers[n]
                else np.zeros(0, dtype=np.dtype(n))
            )
            for n in names
        ]

    def _ref(self, x: Any) -> Optional[Dict[str, int]]:
        idx = self._memo.get(id(x))
        return None if idx is None else {"$r": idx}

    def _reserve(self, x: Any) -> int:
        idx = len(self.nodes)
        self.nodes.append(None)
        self._memo[id(x)] = idx
        self._alive.append(x)
        return idx

    def enc(self, x: Any) -> Any:  # noqa: C901 - one dispatch over the value kinds
        if x is None or type(x) is str or type(x) is bool:
            return x
        if isinstance(x, enum.Enum):
            name = _ENUM_NAMES.get(type(x))
            if name is None:
                raise PartialCodecError(f"cannot persist enum {type(x).__qualname__}")
            return {"$e": name, "v": x.name}
        if isinstance(x, np.generic):
            if x.dtype.kind not in _NUMERIC_KINDS:
                raise PartialCodecError(f"cannot persist numpy {x.dtype}")
            if x.dtype.kind == "c":
                return {"$g": x.dtype.str, "v": [float(x.real), float(x.imag)]}
            return {"$g": x.dtype.str, "v": x.item()}
        if type(x) is int or type(x) is float:
            return x
        if type(x) is complex:
            return {"$c": [x.real, x.imag]}
        if isinstance(x, tuple):
            if type(x) is not tuple:
                raise PartialCodecError(f"cannot persist {type(x).__qualname__}")
            return {"$t": [self.enc(v) for v in x]}
        ref = self._ref(x)
        if ref is not None:
            return ref
        if isinstance(x, np.ndarray):
            return self._enc_array(x)
        if type(x) is list:
            idx = self._reserve(x)
            self.nodes[idx] = {"$l": [self.enc(v) for v in x]}
            return {"$r": idx}
        if type(x) is dict:
            idx = self._reserve(x)
            self.nodes[idx] = {"$d": [[self.enc(k), self.enc(v)] for k, v in x.items()]}
            return {"$r": idx}
        name = _DATACLASS_NAMES.get(type(x))
        if name is not None and dataclasses.is_dataclass(x):
            attrs = vars(x)
            stray = sorted(set(attrs) - _ATTRIBUTE_NAMES[type(x)])
            if stray:
                raise PartialCodecError(
                    f"cannot persist {name} attribute(s) {stray}: not declared"
                )
            idx = self._reserve(x)
            self.nodes[idx] = {
                "$o": name,
                "f": {str(k): self.enc(v) for k, v in attrs.items()},
            }
            return {"$r": idx}
        raise PartialCodecError(
            f"cannot persist {type(x).__module__}.{type(x).__qualname__}"
        )

    def _enc_array(self, x: np.ndarray) -> Dict[str, int]:
        if x.dtype.kind not in _NUMERIC_KINDS:
            raise PartialCodecError(f"cannot persist an array of {x.dtype}")
        idx = self._reserve(x)
        key = x.dtype.str
        flat = np.ascontiguousarray(x).reshape(-1)
        offset = self._sizes.get(key, 0)
        self._buffers.setdefault(key, []).append(flat.copy())
        self._sizes[key] = offset + flat.size
        self.nodes[idx] = {"$a": [key, offset, [int(n) for n in x.shape]]}
        return {"$r": idx}


class _Decoder:
    """Rebuild a graph written by :class:`_Encoder` (allow-listed types only).

    Every malformed value -- a number that is not one, a key that cannot be a
    key, an attribute the type does not declare -- is a
    :class:`PartialCodecError`; nothing else escapes.
    """

    def __init__(self, nodes: List[Any], buffers: Dict[str, np.ndarray]) -> None:
        self._nodes = nodes
        self._buffers = buffers
        self._built: Dict[int, Any] = {}
        #: Every dataclass instance built, for the post-decode type check.
        self.objects: List[Any] = []

    def dec(self, v: Any) -> Any:  # noqa: C901 - one dispatch over the value kinds
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return float(v)
        if isinstance(v, list):
            raise PartialCodecError("bare list in a partial-fit graph")
        if not isinstance(v, dict) or len(v) == 0:
            raise PartialCodecError("malformed partial-fit graph value")
        if "$r" in v:
            return self._node(v["$r"])
        if "$t" in v:
            return tuple(self.dec(x) for x in _as_list(v["$t"]))
        if "$c" in v:
            real, imag = _pair(v["$c"])
            return complex(real, imag)
        if "$e" in v:
            cls = _ENUMS.get(str(v["$e"]))
            if cls is None:
                raise PartialCodecError(f"unknown enum {v['$e']!r}")
            member = v.get("v")
            if not isinstance(member, str):
                raise PartialCodecError(f"malformed {v['$e']} member")
            try:
                return cls[member]  # type: ignore[index]
            except KeyError:
                raise PartialCodecError(f"unknown {v['$e']} member {member!r}")
        if "$g" in v:
            dtype = _numeric_dtype(v["$g"])
            raw = v.get("v")
            try:
                if dtype.kind == "c":
                    real, imag = _pair(raw)
                    return np.array(complex(real, imag), dtype=dtype)[()]
                if not isinstance(raw, (int, float)) or (
                    dtype.kind in "biu" and not isinstance(raw, int)
                ):
                    raise PartialCodecError(f"malformed numpy {dtype} scalar")
                return np.array(raw, dtype=dtype)[()]
            except (OverflowError, ValueError, TypeError) as exc:
                raise PartialCodecError(f"malformed numpy {dtype} scalar") from exc
        raise PartialCodecError("malformed partial-fit graph value")

    def _node(self, idx: Any) -> Any:  # noqa: C901 - one dispatch over node kinds
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or not 0 <= idx < len(self._nodes)
        ):
            raise PartialCodecError("dangling partial-fit graph reference")
        if idx in self._built:
            return self._built[idx]
        node = self._nodes[idx]
        if not isinstance(node, dict):
            raise PartialCodecError("malformed partial-fit graph node")
        if "$l" in node:
            out_list: List[Any] = []
            self._built[idx] = out_list
            out_list.extend(self.dec(x) for x in _as_list(node["$l"]))
            return out_list
        if "$d" in node:
            out_dict: Dict[Any, Any] = {}
            self._built[idx] = out_dict
            for pair in _as_list(node["$d"]):
                kv = _as_list(pair)
                if len(kv) != 2:
                    raise PartialCodecError("malformed dict entry")
                key, value = self.dec(kv[0]), self.dec(kv[1])
                try:
                    out_dict[key] = value
                except TypeError as exc:  # an unhashable key
                    raise PartialCodecError("malformed dict key") from exc
            return out_dict
        if "$o" in node:
            cls = _DATACLASSES.get(str(node["$o"]))
            if cls is None:
                raise PartialCodecError(f"unknown type {node['$o']!r}")
            fields = node.get("f")
            if not isinstance(fields, dict):
                raise PartialCodecError("malformed object node")
            allowed = _ATTRIBUTE_NAMES[cls]
            stray = sorted(str(n) for n in fields if n not in allowed)
            if stray:
                raise PartialCodecError(
                    f"{cls.__qualname__} has no attribute(s) {stray}"
                )
            missing = [f.name for f in dataclasses.fields(cls) if f.name not in fields]
            if missing:
                raise PartialCodecError(f"{cls.__qualname__} lacks {missing}")
            obj: Any = object.__new__(cls)
            self._built[idx] = obj
            self.objects.append(obj)
            for name, raw in fields.items():
                object.__setattr__(obj, name, self.dec(raw))
            return obj
        if "$a" in node:
            spec = _as_list(node["$a"])
            if len(spec) != 3:
                raise PartialCodecError("malformed array node")
            key, offset, shape = str(spec[0]), spec[1], _as_list(spec[2])
            buf = self._buffers.get(key)
            if buf is None or not isinstance(offset, int) or isinstance(offset, bool):
                raise PartialCodecError("array buffer missing")
            if not all(
                isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
            ):
                raise PartialCodecError("malformed array shape")
            count = math.prod(shape)
            if offset < 0 or offset + count > buf.size:
                raise PartialCodecError("array out of its buffer")
            arr = np.array(buf[offset : offset + count]).reshape(shape)
            self._built[idx] = arr
            return arr
        raise PartialCodecError("malformed partial-fit graph node")


def _pair(v: Any) -> Tuple[float, float]:
    """A ``[real, imag]`` pair of JSON numbers."""
    pair = _as_list(v)
    if len(pair) != 2 or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in pair
    ):
        raise PartialCodecError("malformed complex value")
    try:
        return float(pair[0]), float(pair[1])
    except OverflowError as exc:
        raise PartialCodecError("malformed complex value") from exc


# ---------------------------------------------------------------------------
# Post-decode type check
# ---------------------------------------------------------------------------


def _is_real(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating, np.bool_))


def _conforms(value: Any, tp: Any, seen: set) -> bool:  # noqa: C901 - one dispatch
    """Whether ``value`` has the shape of type annotation ``tp``.

    Numbers are taken loosely (a numpy scalar for a float, an int for a float),
    since the walk stores both; containers are checked element by element; a
    dataclass or enum by class. An annotation it does not know (``Any``) passes.
    """
    if tp is Any or tp is object:
        return True
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        return any(_conforms(value, a, seen) for a in args)
    if tp is type(None):
        return value is None
    if tp is float:
        return _is_real(value)
    if tp is int:
        return isinstance(value, (int, np.integer, np.bool_))
    if tp is bool:
        return isinstance(value, (bool, np.bool_, int, np.integer))
    if tp is str:
        return isinstance(value, str)
    if tp is np.ndarray:
        return isinstance(value, np.ndarray)
    if origin in (list, tuple, dict) or tp in (list, tuple, dict):
        container = origin or tp
        if not isinstance(value, container):
            return False
        if id(value) in seen or not args:
            return True
        seen.add(id(value))
        if container is dict:
            kt, vt = args
            return all(
                _conforms(k, kt, seen) and _conforms(v, vt, seen)
                for k, v in value.items()
            )
        if container is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            return len(value) == len(args) and all(
                _conforms(v, a, seen) for v, a in zip(value, args)
            )
        return all(_conforms(v, args[0], seen) for v in value)
    if isinstance(tp, type):
        return isinstance(value, tp)
    return True


def _check_types(objects: List[Any]) -> None:
    """Every attribute of every decoded instance has its declared type.

    Raises
    ------
    PartialCodecError
        On the first attribute that does not.
    """
    for obj in objects:
        cls = type(obj)
        for name, tp in _ATTRIBUTES[cls].items():
            if name not in vars(obj):
                continue  # an undeclared stash the walk did not set
            if not _conforms(vars(obj)[name], tp, set()):
                raise PartialCodecError(f"{cls.__qualname__}.{name} is not a {tp!r}")
        if cls is WindowOutcome:
            _check_outcome_shapes(obj)


#: The per-bin arrays of a :class:`WindowOutcome`, all on its offset grid.
_OUTCOME_GRID_ARRAYS = (
    "offset_grid_mhz",
    "complex_spectrum",
    "rms_noise",
    "background",
    "full_fitted_spectrum",
    "full_residual",
)


def _check_outcome_shapes(outcome: WindowOutcome) -> None:
    """The arrays the walk indexes together agree: every per-bin array of the
    outcome and its fit is 1-D on the outcome's grid, the grid is real, and
    the fit has one error record per peak."""
    n = None
    for name in _OUTCOME_GRID_ARRAYS:
        arr = getattr(outcome, name)
        if arr.ndim != 1 or (n is not None and arr.shape[0] != n):
            raise PartialCodecError(f"WindowOutcome.{name} is not on the grid")
        n = arr.shape[0]
    if outcome.offset_grid_mhz.dtype.kind not in "iuf":
        raise PartialCodecError("WindowOutcome.offset_grid_mhz is not real")
    inner = outcome.fit.fit
    for name in ("fitted_spectrum", "residual"):
        arr = getattr(inner, name)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise PartialCodecError(f"WindowFitResult.{name} is not on the grid")
    if len(inner.peak_errors) != len(inner.peaks):
        raise PartialCodecError("WindowFitResult has peaks without errors")


def _as_list(v: Any) -> List[Any]:
    if not isinstance(v, list):
        raise PartialCodecError("malformed partial-fit graph value")
    return v


def _numeric_dtype(text: Any) -> np.dtype:
    try:
        dtype = np.dtype(str(text))
    except (TypeError, ValueError) as exc:
        raise PartialCodecError(f"bad dtype {text!r}") from exc
    if dtype.kind not in _NUMERIC_KINDS or dtype.hasobject:
        raise PartialCodecError(f"non-numeric dtype {text!r}")
    return dtype


def encode_graph(root: Any) -> Tuple[str, List[str], List[np.ndarray]]:
    """Encode ``root`` to ``(graph JSON, buffer dtypes, buffers)``.

    Raises
    ------
    PartialCodecError
        When the graph holds a value outside the allow-list.
    """
    enc = _Encoder()
    top = enc.enc(root)
    names, buffers = enc.buffers()
    text = json.dumps({"root": top, "nodes": enc.nodes}, allow_nan=True)
    return text, names, buffers


def decode_graph(text: str, names: List[str], buffers: List[np.ndarray]) -> Any:
    """Inverse of :func:`encode_graph`.

    Raises
    ------
    PartialCodecError
        When the graph is malformed or names a type outside the allow-list.
    """
    try:
        doc = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise PartialCodecError(f"unreadable partial-fit graph: {exc}") from exc
    if not isinstance(doc, dict) or "root" not in doc:
        raise PartialCodecError("malformed partial-fit graph")
    if len(names) != len(buffers):
        raise PartialCodecError("partial-fit buffers do not match their dtypes")
    typed: Dict[str, np.ndarray] = {}
    for name, buf in zip(names, buffers):
        dtype = _numeric_dtype(name)
        arr = np.asarray(buf)
        if arr.dtype != dtype or arr.ndim != 1:
            raise PartialCodecError(f"partial-fit buffer is not 1-D {name}")
        typed[str(name)] = arr
    decoder = _Decoder(_as_list(doc.get("nodes")), typed)
    try:
        root = decoder.dec(doc["root"])
        _check_types(decoder.objects)
    except RecursionError as exc:
        raise PartialCodecError("partial-fit graph nested too deeply") from exc
    return root


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


def _text_dataset(group: h5py.Group, name: str, text: str) -> None:
    group.create_dataset(name, data=np.frombuffer(text.encode("utf-8"), dtype=np.uint8))


def _read_text(group: h5py.Group, name: str) -> str:
    node = group.get(name)
    if not isinstance(node, h5py.Dataset) or node.dtype != np.uint8:
        raise PartialCodecError(f"missing {name}")
    try:
        return bytes(np.asarray(node[()], dtype=np.uint8)).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PartialCodecError(f"{name} is not UTF-8") from exc


def stage5_partial_present(h5f: h5py.File) -> bool:
    """Whether the file holds a Stage 5 partial fit."""
    return STAGE5_PARTIAL_PATH in h5f


def delete_stage5_partial(h5f: h5py.File) -> bool:
    """Delete the partial fit (if any); return whether one was deleted."""
    if STAGE5_PARTIAL_PATH in h5f:
        del h5f[STAGE5_PARTIAL_PATH]
        return True
    return False


@dataclasses.dataclass(frozen=True)
class EncodedWindow:
    """One finished window, encoded and ready to write.

    ``preds`` are the finished windows whose outcome this one read: when any of
    them is not kept, neither is this one.
    """

    window_id: int
    graph: str
    dtypes: List[str]
    buffers: List[np.ndarray]
    n_fitted_peaks: int
    preds: Tuple[int, ...] = ()


def encode_stage5_partial(
    partial: PartialWalk,
) -> Tuple[List[EncodedWindow], List[int]]:
    """Encode every window of ``partial`` (no file access).

    Returns ``(encoded, refused)``: the encoded windows, in the order they
    finished, and the ids of the windows that are not kept (a resume refits
    them): those whose graph holds a value the codec cannot write -- or that
    fail to encode for any other reason -- and every finished window that read
    the outcome of a window not kept (``partial.preds``), so the kept set holds
    every window a kept window read.
    """
    encoded: List[EncodedWindow] = []
    refused: List[int] = []
    dropped: set = set()
    for wid in partial.order:
        outcome = partial.outcomes.get(wid)
        record = partial.records.get(wid)
        preds = tuple(sorted(int(p) for p in partial.preds.get(wid, ())))
        if record is None or any(p in dropped for p in preds):
            refused.append(int(wid))
            dropped.add(int(wid))
            continue
        try:
            text, names, buffers = encode_graph({"outcome": outcome, "record": record})
            n_fitted = 0 if outcome is None else int(len(outcome.fit.fit.peaks))
        except Exception:  # noqa: BLE001 - any failure refuses only this window
            refused.append(int(wid))
            dropped.add(int(wid))
            continue
        encoded.append(EncodedWindow(int(wid), text, names, buffers, n_fitted, preds))
    return encoded, refused


def write_stage5_partial(
    h5f: h5py.File, encoded: List[EncodedWindow], provenance: Dict[str, Any]
) -> List[int]:
    """Write the encoded windows as the file's partial fit, replacing any.

    ``provenance`` is stored as JSON (with :data:`PARTIAL_FORMAT_VERSION`).
    A window whose write fails is not kept (its group is removed), and nor is
    any window that read it (:attr:`EncodedWindow.preds`). Returns the window
    ids written, in the order they finished.
    """
    delete_stage5_partial(h5f)
    group = h5f.create_group(STAGE5_PARTIAL_PATH)
    prov = dict(provenance)
    prov["format_version"] = PARTIAL_FORMAT_VERSION
    _text_dataset(group, _PROVENANCE, json.dumps(prov, sort_keys=True))
    windows = group.create_group(_WINDOWS)
    kept: List[EncodedWindow] = []
    dropped: set = set()
    for e in encoded:
        if any(p in dropped for p in e.preds):
            dropped.add(e.window_id)
            continue
        name = f"w{e.window_id}"
        try:
            wg = windows.create_group(name)
            _text_dataset(wg, "graph", e.graph)
            wg.attrs["dtypes"] = json.dumps(e.dtypes)
            for k, buf in enumerate(e.buffers):
                kwargs: Dict[str, Any] = {}
                if buf.size >= 1024:
                    kwargs = {"compression": "gzip", "compression_opts": 1}
                wg.create_dataset(f"buf{k}", data=buf, **kwargs)
        except Exception:  # noqa: BLE001 - any failure drops only this window
            dropped.add(e.window_id)
            if name in windows:
                del windows[name]
            continue
        kept.append(e)
    group.create_dataset(
        _WINDOW_IDS, data=np.asarray([e.window_id for e in kept], dtype=np.int64)
    )
    group.create_dataset(
        _N_FITTED,
        data=np.asarray([e.n_fitted_peaks for e in kept], dtype=np.int64),
    )
    return [e.window_id for e in kept]


def read_stage5_partial_provenance(h5f: h5py.File) -> Optional[Dict[str, Any]]:
    """The partial fit's recorded provenance, or ``None`` when it is missing,
    unreadable or written by another layout version."""
    group = h5f.get(STAGE5_PARTIAL_PATH)
    if not isinstance(group, h5py.Group):
        return None
    try:
        prov = json.loads(_read_text(group, _PROVENANCE))
    except (PartialCodecError, ValueError, RecursionError):
        return None
    if (
        not isinstance(prov, dict)
        or prov.get("format_version") != PARTIAL_FORMAT_VERSION
    ):
        return None
    return prov


def read_stage5_partial_counts(h5f: h5py.File) -> Optional[Dict[int, int]]:
    """``{window_id: n_fitted_peaks}`` of the partial fit's windows, or ``None``
    when the file holds no (readable) partial fit."""
    group = h5f.get(STAGE5_PARTIAL_PATH)
    if not isinstance(group, h5py.Group):
        return None
    ids = group.get(_WINDOW_IDS)
    counts = group.get(_N_FITTED)
    if not isinstance(ids, h5py.Dataset) or not isinstance(counts, h5py.Dataset):
        return None
    try:
        wids = np.asarray(ids[()]).reshape(-1)
        ns = np.asarray(counts[()]).reshape(-1)
    except (OSError, TypeError, ValueError):
        return None
    if (
        wids.shape != ns.shape
        or wids.dtype.kind not in "iu"
        or ns.dtype.kind not in "iu"
    ):
        return None
    return {int(w): int(n) for w, n in zip(wids, ns)}


def read_stage5_partial_windows(h5f: h5py.File) -> CarriedWindows:
    """Rebuild every window of the partial fit.

    Raises
    ------
    PartialCodecError
        When any window cannot be read back (the resume then starts over).
    """
    group = h5f.get(STAGE5_PARTIAL_PATH)
    if not isinstance(group, h5py.Group):
        raise PartialCodecError("no partial fit")
    ids = group.get(_WINDOW_IDS)
    windows = group.get(_WINDOWS)
    if not isinstance(ids, h5py.Dataset) or not isinstance(windows, h5py.Group):
        raise PartialCodecError("partial fit lacks its windows")
    raw_ids = np.asarray(ids[()]).reshape(-1)
    if raw_ids.dtype.kind not in "iu":
        raise PartialCodecError("partial fit window ids are not integers")
    order = [int(w) for w in raw_ids]
    if len(set(order)) != len(order):
        raise PartialCodecError("partial fit repeats a window")
    outcomes: Dict[int, Optional[WindowOutcome]] = {}
    records: Dict[int, WalkRecord] = {}
    for wid in order:
        wg = windows.get(f"w{wid}")
        if not isinstance(wg, h5py.Group):
            raise PartialCodecError(f"partial fit lacks window {wid}")
        try:
            names = json.loads(wg.attrs.get("dtypes", "[]"))
        except (TypeError, ValueError) as exc:
            raise PartialCodecError(f"window {wid}: bad dtypes") from exc
        if not isinstance(names, list):
            raise PartialCodecError(f"window {wid}: bad dtypes")
        buffers = []
        for k in range(len(names)):
            ds = wg.get(f"buf{k}")
            if not isinstance(ds, h5py.Dataset):
                raise PartialCodecError(f"window {wid}: missing buffer {k}")
            buffers.append(np.asarray(ds[()]))
        bundle = decode_graph(_read_text(wg, "graph"), [str(n) for n in names], buffers)
        if not isinstance(bundle, dict):
            raise PartialCodecError(f"window {wid}: malformed bundle")
        outcome = bundle.get("outcome")
        record = bundle.get("record")
        if not isinstance(record, WalkRecord):
            raise PartialCodecError(f"window {wid}: no walk record")
        if outcome is not None and (
            not isinstance(outcome, WindowOutcome) or int(outcome.window_id) != wid
        ):
            raise PartialCodecError(f"window {wid}: malformed outcome")
        outcomes[wid] = outcome
        records[wid] = record
    return CarriedWindows(order=order, outcomes=outcomes, records=records)


__all__ = [
    "PARTIAL_FORMAT_VERSION",
    "EncodedWindow",
    "PartialCodecError",
    "decode_graph",
    "delete_stage5_partial",
    "encode_graph",
    "encode_stage5_partial",
    "read_stage5_partial_counts",
    "read_stage5_partial_provenance",
    "read_stage5_partial_windows",
    "stage5_partial_present",
    "write_stage5_partial",
]
