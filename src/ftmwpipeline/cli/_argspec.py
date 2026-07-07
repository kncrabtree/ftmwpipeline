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
import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from ..core.knob_metadata import iter_knob_fields, knob_meta
from ..core.settings import FTSettings
from ..core.start_detection_settings import StartDetectionSettings


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


def _namespaced_flag_body(sub: Optional[str], name: str) -> str:
    """Deterministic flag body for a namespaced (prefixed) flag.

    Ignores any custom ``km.cli.flag`` override so the add-side flag and the
    reconstruct-side ``dest`` stay in lockstep purely from the field path --
    two stages could otherwise pick the same hand-picked legacy flag name and
    collide once namespaced onto a shared parser (e.g. ``run``).
    """
    return (f"{sub}." if sub else "") + name.replace("_", "-")


def add_settings_args(
    parser: "argparse._ActionsContainer",
    cls: Type[Any] = FTSettings,
    *,
    prefix: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    """Add one CLI option per ``cli=True`` knob of ``cls`` to ``parser``.

    The argparse ``default`` is forced to ``None`` (the unset sentinel) so the
    resolution chain is preserved regardless of the dataclass default.

    ``prefix``, when given, namespaces every generated flag/dest under
    ``"{prefix}."`` (e.g. ``--ft.start-us`` / dest ``"ft.start_us"``) so
    several stages' knob surfaces can share one parser (the ``run`` command)
    without leaf-name collisions; the flag body is then always derived
    deterministically from the field path (see :func:`_namespaced_flag_body`),
    ignoring any custom ``flag=`` override. ``prefix=None`` (the default)
    reproduces the exact pre-existing per-stage subcommand behavior.

    ``exclude`` (top-level field names only) skips fields that a caller wants
    to keep as a distinct, non-namespaced flag (e.g. ``run`` excludes
    ``FTSettings.trim`` because ``--trim`` stays the canonical top-level flag).
    """
    excluded = set(exclude or ())
    for sub, f in _cli_fields(cls):
        if sub is None and f.name in excluded:
            continue
        km = knob_meta(f)
        assert km is not None and km.cli is not None  # _cli_fields guarantees it
        if prefix is None:
            flag = km.cli.flag or "--" + f.name.replace("_", "-")
            dest = _dest(sub, f.name)
        else:
            flag = f"--{prefix}.{_namespaced_flag_body(sub, f.name)}"
            dest = f"{prefix}.{_dest(sub, f.name)}"
        kwargs: Dict[str, Any] = {
            "dest": dest,
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
    args: argparse.Namespace,
    cls: Type[Any] = FTSettings,
    *,
    prefix: Optional[str] = None,
) -> Any:
    """Reconstruct a sparse settings instance from parsed CLI args.

    Only ``cli=True`` fields are read; everything else stays at the unset
    sentinel so the resolution chain fills it. Sub-block fields are gathered and
    each populated sub-block is instantiated, so a nested settings class round-
    trips with the same sparse semantics as a flat one.

    ``prefix`` must match whatever :func:`add_settings_args` used to build the
    namespace (each dest read as ``"{prefix}.{existing_dest}"``); a field an
    excluding caller never added a flag for simply reads back ``None`` via the
    ``getattr`` default, so it stays unset without any special-casing here.
    """
    top: Dict[str, Any] = {}
    sub_values: Dict[str, Dict[str, Any]] = {}
    for sub, f in _cli_fields(cls):
        existing_dest = _dest(sub, f.name)
        dest = existing_dest if prefix is None else f"{prefix}.{existing_dest}"
        value = getattr(args, dest, None)
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


# ---------------------------------------------------------------------------
# StartDetectionSettings: a flat frozen dataclass with concrete hard defaults
# and no ``knob_field`` metadata (see core/start_detection_settings.py), so
# the metadata-driven generation above does not apply. This is the single
# shared generator + reader both the `start` subcommand
# (cli/start_commands.py) and the `run` command's `--start.*` passthrough
# (cli/run_commands.py) build on, so the two never drift out of sync.
# ---------------------------------------------------------------------------

#: One-line help text for each StartDetectionSettings field.
START_FIELD_HELP: Dict[str, str] = {
    "sweep_max_us": "Upper bound of the start-time sweep, us (default 7.5; "
    "capped to the FID duration).",
    "step_us": "Sweep step, us (default 0.02).",
    "floor_factor": "Chirp-end = first start where the integrated |FT| falls "
    "below factor*floor (default 3.0).",
    "floor_tail_us": "Width of the deep-tail window used for the floor "
    "estimate, us (default 1.0).",
    "guard_margin_us": "Margin added past the chirp end for the "
    "switch-bounce ringdown, us (default 0.67; instrument-specific).",
    "min_chirp_drop_ratio": "Minimum plateau/floor ratio for the chirp "
    "collapse to be considered present (default 10.0).",
    "band_min_mhz": "Lower bound of an explicit integration band override, "
    "MHz (default: Stage 1 trim, else full spectrum).",
    "band_max_mhz": "Upper bound of an explicit integration band override, "
    "MHz (default: Stage 1 trim, else full spectrum).",
}


def add_start_detection_args(
    parser: "argparse._ActionsContainer",
    *,
    prefix: Optional[str] = None,
    exclude: Iterable[str] = frozenset(),
) -> None:
    """Add one ``--[{prefix}.]<field>`` float flag per ``StartDetectionSettings``
    field not in ``exclude``.

    Field-driven over ``dataclasses.fields`` (there is no ``knob_field``
    metadata to walk here). ``prefix`` namespaces the flag/dest the same way
    :func:`add_settings_args` does (``--start.sweep-max-us`` / dest
    ``"start.sweep_max_us"``); ``prefix=None`` reproduces the bare
    ``--sweep-max-us`` style the ``start`` subcommand uses. ``exclude`` lets a
    caller keep a subset of fields as a distinct, hand-written flag (e.g. the
    ``start`` subcommand excludes ``band_min_mhz``/``band_max_mhz`` in favor
    of its paired ``--band MIN MAX`` convenience flag).
    """
    excluded = set(exclude)
    for f in dataclasses.fields(StartDetectionSettings):
        if f.name in excluded:
            continue
        flag_body = f.name.replace("_", "-")
        flag = f"--{flag_body}" if prefix is None else f"--{prefix}.{flag_body}"
        dest = f.name if prefix is None else f"{prefix}.{f.name}"
        parser.add_argument(
            flag,
            dest=dest,
            type=float,
            default=None,
            help=START_FIELD_HELP.get(f.name, f"Start-detection knob: {f.name}."),
        )


def start_settings_from_namespace(
    args: argparse.Namespace,
    *,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Collect the non-``None`` ``StartDetectionSettings`` overrides parsed
    from ``args``.

    Reads dest ``[{prefix}.]{field}`` for every field, matching whatever
    :func:`add_start_detection_args` used to build the namespace. Only fields
    the user actually set are included (fields the caller ``exclude``d from
    generation, or simply left unset, read back ``None`` and are omitted); the
    caller passes the resulting sparse dict to ``StartDetectionSettings(**...)``
    so the rest fill in from its concrete defaults.
    """
    overrides: Dict[str, float] = {}
    for f in dataclasses.fields(StartDetectionSettings):
        dest = f.name if prefix is None else f"{prefix}.{f.name}"
        value = getattr(args, dest, None)
        if value is not None:
            overrides[f.name] = value
    return overrides
