"""Generate argparse options from a settings dataclass's field metadata.

Single source of truth: the per-knob CLI flags for a stage are derived from the
settings dataclass's ``knob_field`` metadata (see
:mod:`ftmwpipeline.core.knob_metadata`) rather than hand-written in each
subcommand. The per-subcommand shell (parser registration, ``file_path``,
``--output``, ``--verbose``, ``set_defaults(func=...)``) is *not* generated --
only the settings option group.

Works for flat settings classes (``FTSettings`` / ``NoiseSettings``) and for the
nested, sub-block settings classes (``PeakDetectionSettings`` etc.): the walk
descends one level into sub-block dataclasses. Only fields the maintainer tagged
``cli=True`` get a flag; everything else stays reachable through
``settings set`` / a preset / a ``settings=`` instance.
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple, Type

from ..core.knob_metadata import iter_knob_fields, knob_meta
from ..core.settings import FTSettings


def _dest(sub: Optional[str], name: str) -> str:
    """argparse ``dest`` for a (possibly nested) field.

    Flat fields keep their bare name so existing direct reads (``args.trim`` in
    the FT command) still work; sub-block fields are namespaced
    (``"subblock.field"``) so leaf-name collisions across sub-blocks cannot
    clash and the reconstruction can route the value back to its sub-block.
    """
    return name if sub is None else f"{sub}.{name}"


def _cli_fields(cls: Type[Any]) -> List[Tuple[Optional[str], Any]]:
    """``(sub_block_or_None, Field)`` for every ``cli=True`` knob of ``cls``."""
    out: List[Tuple[Optional[str], Any]] = []
    for sub, f in iter_knob_fields(cls):
        km = knob_meta(f)
        if km is not None and km.cli is not None:
            out.append((sub, f))
    return out


def add_settings_args(
    parser: argparse.ArgumentParser, cls: Type[Any] = FTSettings
) -> None:
    """Add one CLI option per ``cli=True`` knob of ``cls`` to ``parser``.

    The argparse ``default`` is forced to ``None`` (the unset sentinel) so the
    resolution chain is preserved regardless of the dataclass default.
    """
    for sub, f in _cli_fields(cls):
        km = knob_meta(f)
        assert km is not None and km.cli is not None  # _cli_fields guarantees it
        flag = km.cli.flag or "--" + f.name.replace("_", "-")
        kwargs: Dict[str, Any] = {
            "dest": _dest(sub, f.name),
            "default": None,
            "help": km.help,
        }
        if km.cli.is_flag:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            if km.cli.argtype is not None:
                kwargs["type"] = km.cli.argtype
            if km.cli.metavar is not None:
                kwargs["metavar"] = km.cli.metavar
        parser.add_argument(flag, **kwargs)


def settings_from_namespace(
    args: argparse.Namespace, cls: Type[Any] = FTSettings
) -> Any:
    """Reconstruct a sparse settings instance from parsed CLI args.

    Only ``cli=True`` fields are read; everything else stays at the unset
    sentinel so the resolution chain fills it. Sub-block fields are gathered and
    each populated sub-block is instantiated, so a nested settings class round-
    trips with the same sparse semantics as a flat one.
    """
    top: Dict[str, Any] = {}
    sub_values: Dict[str, Dict[str, Any]] = {}
    for sub, f in _cli_fields(cls):
        value = getattr(args, _dest(sub, f.name), None)
        if sub is None:
            top[f.name] = value
        else:
            sub_values.setdefault(sub, {})[f.name] = value

    kwargs: Dict[str, Any] = dict(top)
    if sub_values:
        template = cls()
        for sub, field_vals in sub_values.items():
            sub_cls = type(getattr(template, sub))
            kwargs[sub] = sub_cls(**field_vals)
    return cls(**kwargs)
