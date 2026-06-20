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

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Iterator, Optional, Tuple

# Metadata sub-keys on a ``dataclasses.Field``.
_KNOB_KEY = "knob"
_CLI_KEY = "cli"


@dataclass(frozen=True)
class CliSpec:
    """argparse spec for a knob that is exposed as a generated CLI flag."""

    flag: Optional[str]
    argtype: Optional[Callable[[str], Any]]
    metavar: Optional[str]
    is_flag: bool


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
    return field(default=None, metadata=meta)


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


__all__ = [
    "CliSpec",
    "KnobMeta",
    "knob_field",
    "knob_meta",
    "iter_knob_fields",
    "field_knob_meta",
]
