"""Per-knob field metadata: the single declaration site for a tunable setting.

A settings-dataclass field declared with :func:`knob_field` carries, in its
``dataclasses.field`` metadata, everything the surrounding machinery needs to
treat the field as a tunable knob:

* the one-line ``help`` (shared by the CLI flag, ``settings show``, and the
  ``scan`` registry),
* the ``scan`` descriptors ``tier`` / ``inst_sensitivity`` / ``grid``,
* an optional generated CLI flag (``cli=True`` plus the argparse spec).

The field default is always ``None`` (the unset sentinel) so concrete values
come from the resolution chain, never the dataclass default.

This is the stage-agnostic spelling.
:func:`ftmwpipeline.core.settings.cli_field` predates it (Stage 1's
``FTSettings``) and now delegates here, so the whole settings surface declares a
knob once and the CLI, the ``scan`` registry, and ``settings show`` all read
the same metadata.

Stdlib-only so every ``core/*_settings.py`` module (and the knob registry) can
import it without cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Iterator, Optional, Tuple

# Metadata sub-keys on a ``dataclasses.Field``.
_KNOB_KEY = "knob"
_CLI_KEY = "cli"
_TYPING_KEY = "typing"


@dataclass(frozen=True)
class CliSpec:
    """argparse spec for a knob that is exposed as a generated CLI flag."""

    flag: Optional[str]
    argtype: Optional[Callable[[str], Any]]
    metavar: Optional[str]
    is_flag: bool


@dataclass(frozen=True)
class FieldTyping:
    """Machine-readable typing a field states about itself.

    Every member is ``None`` unless the declaration states it -- nothing is
    inferred from a name or a default. ``bounds`` is the contract mapping
    ``{"min", "max", "min_inclusive", "max_inclusive"}``.
    """

    units: Optional[str] = None
    choices: Optional[Tuple[Any, ...]] = None
    bounds: Optional[dict[str, Any]] = None


def make_bounds(
    *,
    min: Optional[float] = None,
    max: Optional[float] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> dict[str, Any]:
    """A bounds mapping in the contract's shape (``None`` end = unbounded)."""
    return {
        "min": min,
        "max": max,
        "min_inclusive": min_inclusive,
        "max_inclusive": max_inclusive,
    }


def _typing_meta(
    units: Optional[str],
    choices: Optional[Tuple[Any, ...]],
    bounds: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if units is None and choices is None and bounds is None:
        return {}
    return {_TYPING_KEY: {"units": units, "choices": choices, "bounds": bounds}}


@dataclass(frozen=True)
class KnobMeta:
    """Resolved knob declaration read off a field's metadata.

    ``cli`` is ``None`` for a knob that has no generated CLI flag (reachable via
    ``settings set`` / preset / ``settings=`` only).
    """

    help: str
    tier: str
    inst_sensitivity: str
    grid: Optional[Tuple[Any, ...]]
    cli: Optional[CliSpec]


def knob_field(
    *,
    help: str,
    tier: str = "advanced",
    inst_sensitivity: str = "N",
    grid: Optional[Tuple[Any, ...]] = None,
    cli: bool = False,
    flag: Optional[str] = None,
    argtype: Optional[Callable[[str], Any]] = None,
    metavar: Optional[str] = None,
    is_flag: bool = False,
    units: Optional[str] = None,
    choices: Optional[Tuple[Any, ...]] = None,
    bounds: Optional[dict[str, Any]] = None,
) -> Any:
    """Declare a settings-dataclass field as a tunable knob.

    The default is always ``None`` (the unset sentinel); concrete values come
    from the resolution chain, never the dataclass default, so precedence stays
    intact.

    Parameters
    ----------
    help:
        One-line physical meaning. The single source for the CLI help, the
        ``settings show`` help column, and the ``scan`` registry help.
    tier:
        ``"primary"`` (shown in the default ``scan list`` / ``settings show``)
        or ``"advanced"`` (revealed with ``--all``).
    inst_sensitivity:
        Instrument-sensitivity rating ``"Y"`` / ``"N"`` / ``"maybe"`` for the
        per-instrument calibration audit.
    grid:
        Default sweep values for ``scan`` when the caller passes no grid.
    cli:
        When ``True`` the field gets a generated CLI flag (opt-in: most fields
        stay reachable only through ``settings set`` / preset / ``settings=``).
    flag:
        Explicit long flag (e.g. to preserve a legacy name); else derived as
        ``"--" + name.replace("_", "-")``.
    argtype:
        Callable passed to argparse ``type=`` (coercion lives in metadata,
        never inferred from the annotation -- robust under
        ``from __future__ import annotations``).
    metavar:
        Optional argparse metavar.
    is_flag:
        Tri-state boolean rendered with ``BooleanOptionalAction``
        (``--x`` / ``--no-x``).
    units, choices, bounds:
        Machine-readable typing (see :class:`FieldTyping`), stated only where
        unambiguous; ``None`` otherwise.
    """
    meta: dict[str, Any] = {
        _KNOB_KEY: {
            "help": help,
            "tier": tier,
            "inst_sensitivity": inst_sensitivity,
            "grid": grid,
        }
    }
    if cli:
        meta[_CLI_KEY] = {
            "flag": flag,
            "argtype": argtype,
            "metavar": metavar,
            "is_flag": is_flag,
        }
    meta.update(_typing_meta(units, choices, bounds))
    return field(default=None, metadata=meta)


def field_typing(
    *,
    units: Optional[str] = None,
    choices: Optional[Tuple[Any, ...]] = None,
    bounds: Optional[dict[str, Any]] = None,
) -> Any:
    """Declare typing for a settings field that is *not* a ``knob_field``.

    Same ``None`` default and typing metadata as :func:`knob_field`, without the
    knob (no help, tier or scan grid), so the field stays out of the knob
    registry.
    """
    return field(default=None, metadata=_typing_meta(units, choices, bounds))


def field_typing_meta(f: Any) -> FieldTyping:
    """:class:`FieldTyping` of a ``dataclasses.Field`` (all ``None`` if unstated)."""
    t = f.metadata.get(_TYPING_KEY)
    if t is None:
        return FieldTyping()
    return FieldTyping(units=t["units"], choices=t["choices"], bounds=t["bounds"])


def knob_meta(f: Any) -> Optional[KnobMeta]:
    """:class:`KnobMeta` for a ``dataclasses.Field``, or ``None`` if it is not a
    ``knob_field`` (a plain field carries no knob metadata)."""
    km = f.metadata.get(_KNOB_KEY)
    if km is None:
        return None
    cli_raw = f.metadata.get(_CLI_KEY)
    cli = (
        None
        if cli_raw is None
        else CliSpec(
            flag=cli_raw["flag"],
            argtype=cli_raw["argtype"],
            metavar=cli_raw["metavar"],
            is_flag=cli_raw["is_flag"],
        )
    )
    return KnobMeta(
        help=km["help"],
        tier=km["tier"],
        inst_sensitivity=km["inst_sensitivity"],
        grid=km["grid"],
        cli=cli,
    )


def iter_knob_fields(cls: type) -> Iterator[Tuple[Optional[str], Any]]:
    """Yield ``(sub_block_or_None, Field)`` for every field of a settings class.

    A dataclass-valued field is a sub-block: it expands into one pair per
    sub-field, ``(sub_name, sub_field)``. A scalar field yields
    ``(None, field)``. Sub-blocks are detected by value on a default instance,
    keeping the walk in lockstep with the resolver's own structure (mirrors
    ``settings_inspection._enumerate_fields`` but yields the ``Field`` objects
    so their metadata is reachable).
    """
    inst = cls()
    for f in fields(cls):
        value = getattr(inst, f.name)
        if is_dataclass(value) and not isinstance(value, type):
            for sub_f in fields(value):
                yield (f.name, sub_f)
        else:
            yield (None, f)


def field_knob_meta(cls: type, dotted_tail: str) -> KnobMeta:
    """:class:`KnobMeta` for ``"field"`` or ``"subblock.field"`` within ``cls``.

    The lookup key is the settings path tail (the part after the ``stageN.``
    prefix), matching the knob registry's selector convention. Raises
    ``KeyError`` when the path does not resolve to a ``knob_field``.
    """
    parts = dotted_tail.split(".")
    inst = cls()
    if len(parts) == 1:
        target_fields = fields(cls)
        leaf = parts[0]
    elif len(parts) == 2:
        sub, leaf = parts
        sub_inst = getattr(inst, sub, None)
        if not is_dataclass(sub_inst) or isinstance(sub_inst, type):
            raise KeyError(f"{cls.__name__} has no sub-block {sub!r}")
        target_fields = fields(sub_inst)
    else:
        raise KeyError(f"unsupported knob path depth: {dotted_tail!r}")
    for f in target_fields:
        if f.name == leaf:
            km = knob_meta(f)
            if km is None:
                raise KeyError(f"{cls.__name__}.{dotted_tail} is not a knob_field")
            return km
    raise KeyError(f"{cls.__name__} has no field {dotted_tail!r}")


# ---------------------------------------------------------------------------
# Enforcing declared choices / bounds
# ---------------------------------------------------------------------------
def _bounds_text(bounds: dict[str, Any]) -> str:
    """Interval notation for a bounds mapping: ``[0.0, 1.0)``, ``(0, inf)``."""
    lo, hi = bounds.get("min"), bounds.get("max")
    left = "[" if bounds.get("min_inclusive", True) and lo is not None else "("
    right = "]" if bounds.get("max_inclusive", True) and hi is not None else ")"
    return f"{left}{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}{right}"


def _outside_bounds(number: Any, bounds: dict[str, Any]) -> bool:
    """``True`` when a numeric ``number`` violates ``bounds`` (nan always does)."""
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return False
    if isinstance(number, float) and math.isnan(number):
        return True
    lo, hi = bounds.get("min"), bounds.get("max")
    if lo is not None and (
        number < lo or (number == lo and not bounds.get("min_inclusive", True))
    ):
        return True
    return hi is not None and (
        number > hi or (number == hi and not bounds.get("max_inclusive", True))
    )


def check_field_typing(path: str, value: Any, raw: Any, typing: FieldTyping) -> None:
    """Enforce a field's declared ``choices`` and ``bounds`` on ``value``.

    ``choices`` is membership of the whole value. ``bounds`` applies to a
    numeric scalar, or to every numeric element of a tuple / list value; a
    value of any other kind is not bounded. A violation raises
    :class:`~ftmwpipeline.file_manager.BadSettingError` with ``path`` (the
    registry path), the declared constraint as ``expected`` and ``raw`` (what
    the caller supplied) as ``value``. ``None`` (unset) is never checked here.
    """
    # Imported lazily: ``file_manager`` imports from ``core``.
    from ..file_manager import BadSettingError

    if typing.choices is not None and value not in typing.choices:
        listed = ", ".join(repr(c) for c in typing.choices)
        raise BadSettingError(
            path,
            f"one of {listed}",
            raw,
            message=f"{path}: {raw!r} is not one of {listed}",
        )
    if typing.bounds is not None:
        items = value if isinstance(value, (tuple, list)) else (value,)
        if any(_outside_bounds(item, typing.bounds) for item in items):
            interval = _bounds_text(typing.bounds)
            raise BadSettingError(
                path,
                f"a value in {interval}",
                raw,
                message=f"{path}: {raw!r} is outside {interval}",
            )


def check_declared_typing(settings: Any, prefix: str) -> None:
    """Enforce every declared ``choices`` / ``bounds`` on a settings object.

    Walks the settings dataclass the way the registry does -- a scalar field is
    ``<prefix>.<field>``, a sub-block field ``<prefix>.<sub>.<field>`` -- and
    checks each set (non-``None``) value with :func:`check_field_typing`. The
    stage resolvers call it on the merged result, so a value from any layer
    (``settings=`` object, preset, persisted record) is held to the same
    declaration ``settings set`` enforces.
    """
    for f in fields(settings):
        value = getattr(settings, f.name)
        if is_dataclass(value) and not isinstance(value, type):
            for sub_f in fields(value):
                sub_value = getattr(value, sub_f.name)
                if sub_value is not None:
                    check_field_typing(
                        f"{prefix}.{f.name}.{sub_f.name}",
                        sub_value,
                        sub_value,
                        field_typing_meta(sub_f),
                    )
        elif value is not None:
            check_field_typing(f"{prefix}.{f.name}", value, value, field_typing_meta(f))


__all__ = [
    "check_declared_typing",
    "check_field_typing",
    "CliSpec",
    "FieldTyping",
    "KnobMeta",
    "knob_field",
    "field_typing",
    "field_typing_meta",
    "make_bounds",
    "knob_meta",
    "iter_knob_fields",
    "field_knob_meta",
]
