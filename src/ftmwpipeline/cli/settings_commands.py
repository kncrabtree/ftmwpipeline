"""CLI subcommands for the cross-cutting ``settings`` meta-object.

``settings show <file> [selector]`` reports, per setting, the value actually in
effect for an experiment and the layer that supplied it (``.ftmw`` / ``.yml:<name>``
/ ``recommended`` / ``default``) -- the resolved-view counterpart to ``scan
list``'s tunable-knob listing. The shared core is
``_internal.tuning.resolve_settings_view``; this module only formats its rows,
reusing the ``scan list`` table layout (prefix-elided knob column, primary/advanced
tiering, dotted-path selector).

The ``set`` / ``export`` verbs (persist a chosen value to the ``.ftmw`` / write a
``.yml`` preset block) attach to the same ``settings`` object group; see
``CLI_STRATEGY.md`` and ``planning/tune-settings-verb.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ..core.stage_fit_settings import ShapeSpec
from .utils import elide_path, print_error


def _fmt_value(value: Any) -> str:
    """Render a resolved setting value for the table.

    ``None`` reads ``unset`` (the field falls through every layer); a
    :class:`ShapeSpec` shows its line-shape kind; tuples (the FT ``trim``) join
    their elements; floats use compact ``%g``.
    """
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, ShapeSpec):
        return str(value.kind.value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_fmt_value(v) for v in value)
    return str(value)


def cmd_settings_show(args: argparse.Namespace) -> int:
    """Print the resolved value + provenance for each covered setting.

    Shows primary-tier rows by default; ``--all`` reveals advanced ones. A
    positional ``selector`` filters by dotted-path prefix (e.g. ``stage2b`` /
    ``stage2b.gaussian``). ``--preset`` populates the ``.yml`` provenance layer
    for the named preset (with no ``--preset`` the preset layer is empty, since
    no preset is bound to a file).
    """
    from .._internal.tuning import resolve_settings_view

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"
    if not Path(file_path).exists():
        print_error(f"Pipeline file not found: {file_path}")
        return 1

    selector = getattr(args, "selector", None)
    show_all = bool(getattr(args, "all", False))
    preset = getattr(args, "preset", None)

    try:
        rows = resolve_settings_view(
            file_path,
            selector,
            include_advanced=show_all,
            preset=preset,
        )
    except FileNotFoundError as e:
        # A bad --preset name reports the available presets.
        print_error(str(e))
        return 1

    if not rows:
        suffix = f" matching {selector!r}." if selector else "."
        print(
            "No settings"
            + suffix
            + ("" if show_all else " (try --all for advanced settings).")
        )
        return 0

    headers = ("knob", "value", "source", "default")
    table = [
        (r.path, _fmt_value(r.value), r.source, _fmt_value(r.hard_default))
        for r in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        for i in range(len(headers))
    ]
    sep = "  "

    def _line(cells: Any) -> str:
        return sep.join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    print(_line(headers))
    print(_line(tuple("-" * widths[i] for i in range(len(headers)))))
    prev_path: Optional[str] = None
    for r, row in zip(rows, table):
        stage = r.path.split(".")[0]
        if prev_path is not None and stage != prev_path.split(".")[0]:
            print()  # blank line between stages
        knob_cell = elide_path(r.path, prev_path)
        print(_line((knob_cell,) + row[1:]))
        prev_path = r.path

    print()
    if not show_all:
        hidden = [
            r
            for r in resolve_settings_view(
                file_path, selector, include_advanced=True, preset=preset
            )
            if r.tier == "advanced"
        ]
        if hidden:
            print(f"{len(hidden)} advanced setting(s) hidden; use --all to show them.")
    print(
        "source: .ftmw (persisted in the file) > .yml:<name> (preset) > "
        "recommended > default."
    )
    print(
        "A value persisted in the .ftmw outranks a preset; pass --preset <name> "
        "to see what a preset would seed for fields the file has not fixed."
    )
    print(
        "Change a value with 'settings set <file> <knob> <value>'; capture the "
        "file's chosen values as a preset with 'settings export <file> <out.yml>'."
    )
    return 0


def cmd_settings_set(args: argparse.Namespace) -> int:
    """Persist a chosen value into the .ftmw and invalidate stale stages."""
    from .._internal.tuning import set_setting

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"
    if not Path(file_path).exists():
        print_error(f"Pipeline file not found: {file_path}")
        return 1

    try:
        result = set_setting(file_path, args.knob, args.value)
    except ValueError as e:
        print_error(str(e))
        return 1

    print(f"Set {result.path} = {_fmt_value(result.value)} (persisted to .ftmw).")
    if result.invalidated:
        print(
            "Invalidated downstream stage(s): "
            + ", ".join(result.invalidated)
            + " -- re-run them to refresh."
        )
    else:
        print("No completed stages depended on this setting.")
    return 0


def cmd_settings_export(args: argparse.Namespace) -> int:
    """Write the file's chosen Stage 2-5 values to a .yml preset block."""
    from .._internal.tuning import export_settings

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"
    if not Path(file_path).exists():
        print_error(f"Pipeline file not found: {file_path}")
        return 1

    result = export_settings(
        file_path,
        args.out_path,
        getattr(args, "selector", None),
        name=getattr(args, "name", None),
        description=getattr(args, "description", None),
    )
    if not result.paths:
        sel = getattr(args, "selector", None)
        suffix = f" matching {sel!r}" if sel else ""
        print(
            f"No persisted Stage 2-5 values{suffix} to export; "
            f"wrote an empty preset to {result.out_path}."
        )
        return 0
    print(
        f"Exported {len(result.paths)} setting(s) to {result.out_path}. "
        f"Load with --preset {result.out_path}."
    )
    return 0


def register_settings_commands(subparsers: Any) -> None:
    """Register the ``settings`` object group and its verbs."""
    settings = subparsers.add_parser(
        "settings",
        help="Inspect and change resolved pipeline settings for an experiment",
        description=(
            "Cross-cutting settings surface. 'settings show' reports, per "
            "setting, the value in effect for a .ftmw and the layer that "
            "supplied it (.ftmw / .yml preset / recommended / default)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    settings_sub = settings.add_subparsers(
        dest="settings_command", help="settings subcommands"
    )

    p_show = settings_sub.add_parser(
        "show",
        help="Show the resolved value and provenance of each setting",
    )
    p_show.add_argument(
        "file_path",
        help="Path to the .ftmw experiment to inspect",
    )
    p_show.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="Filter by dotted-path prefix, e.g. stage2b or stage2b.gaussian",
    )
    p_show.add_argument(
        "--all",
        action="store_true",
        help="Include advanced-tier settings (hidden by default)",
    )
    p_show.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Populate the .yml preset provenance layer from this preset "
        "(bare name or path to a YAML file)",
    )
    p_show.set_defaults(func=cmd_settings_show)

    p_set = settings_sub.add_parser(
        "set",
        help="Persist a chosen value into the .ftmw (invalidates stale stages)",
        description=(
            "Persist KNOB = VALUE into the experiment's settings. The affected "
            "stage and every downstream stage are invalidated so the file stays "
            "self-consistent; re-run them to refresh. The canonical FT is "
            "unapodized and native-length -- there are no FT apodization knobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_set.add_argument("file_path", help="Path to the .ftmw experiment")
    p_set.add_argument(
        "knob",
        help="Dotted settings path, e.g. stage2.window_mhz or "
        "stage5.tau.max_decay_factor",
    )
    p_set.add_argument("value", help="Value to persist (coerced to the field type)")
    p_set.set_defaults(func=cmd_settings_set)

    p_export = settings_sub.add_parser(
        "export",
        help="Write the file's chosen Stage 2-5 values to a .yml preset block",
        description=(
            "Serialize the experiment's persisted Stage 2-5 settings into a "
            "YAML preset (the portable form a sibling experiment loads via "
            "--preset). Stage 1 is excluded -- presets do not carry FT settings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export.add_argument("file_path", help="Path to the .ftmw experiment")
    p_export.add_argument("out_path", help="Destination .yml preset file")
    p_export.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="Restrict to a dotted prefix, e.g. stage5 or stage5.rescue",
    )
    p_export.add_argument(
        "--name",
        type=str,
        default=None,
        help="Preset name metadata (defaults to the output file stem)",
    )
    p_export.add_argument(
        "--description",
        type=str,
        default=None,
        help="Preset description metadata",
    )
    p_export.set_defaults(func=cmd_settings_export)

    # 'settings' with no subcommand prints its help.
    def _settings_help(args: argparse.Namespace) -> int:
        settings.print_help()
        return 1

    settings.set_defaults(func=_settings_help)
