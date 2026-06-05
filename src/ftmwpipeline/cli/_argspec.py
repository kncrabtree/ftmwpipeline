"""
Generate argparse options from a settings dataclass's field metadata.

Single source of truth: the CLI flags for FT settings are derived from
``FTSettings``' ``cli_field`` metadata rather than hand-written in each
subcommand. The per-subcommand shell (parser registration, ``file_path``,
``--output``, ``--verbose``, ``set_defaults(func=...)``) is *not* generated --
only the settings option group.
"""

import argparse
from dataclasses import fields
from typing import Any, Dict, Type

from ..core.settings import FTSettings


def add_settings_args(
    parser: argparse.ArgumentParser, cls: Type[Any] = FTSettings
) -> None:
    """Add one CLI option per ``cli_field`` of ``cls`` to ``parser``.

    The argparse ``default`` is forced to ``None`` (the unset sentinel) so the
    resolution chain is preserved regardless of the dataclass default.
    """
    for f in fields(cls):
        meta = f.metadata.get("cli")
        if meta is None:
            continue
        flag = meta["flag"] or "--" + f.name.replace("_", "-")
        kwargs: Dict[str, Any] = {
            "dest": f.name,
            "default": None,
            "help": meta["help"],
        }
        if meta["is_flag"]:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            if meta["argtype"] is not None:
                kwargs["type"] = meta["argtype"]
            if meta["metavar"] is not None:
                kwargs["metavar"] = meta["metavar"]
        parser.add_argument(flag, **kwargs)


def settings_from_namespace(
    args: argparse.Namespace, cls: Type[Any] = FTSettings
) -> Any:
    """Reconstruct a sparse settings instance from parsed CLI args.

    Only fields carrying ``cli`` metadata are read; everything else stays at
    the unset sentinel so the resolution chain fills it.
    """
    values = {
        f.name: getattr(args, f.name, None) for f in fields(cls) if "cli" in f.metadata
    }
    return cls(**values)
