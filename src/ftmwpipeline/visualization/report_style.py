"""Shared figure styling for the report and CLI plots.

The spine-free "bare" look is the default house style: no spines, no tick
marks (labels kept), and a faint major grid behind the data. Titles are
opt-in -- a plot function takes ``title=None`` to use its own default (the CLI
path), ``title=""`` to suppress it entirely (the report path, where the page
heading already labels the figure), or any truthy string to show that text.
"""

from typing import Any, Optional

_BARE_GRID_COLOR = "#dbe0e6"


def apply_bare_style(ax: Any) -> None:
    """Strip *ax* to a spine-free, tick-mark-free look with a light major grid."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, which="major", color=_BARE_GRID_COLOR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def resolve_title(title: Optional[str], default: str) -> str:
    """Resolve the opt-in title convention.

    ``None`` -> *default* (CLI/direct callers); ``""`` -> ``""`` (suppressed, the
    report path); any other string -> itself. Callers draw the title only when
    the result is truthy.
    """
    return default if title is None else title
