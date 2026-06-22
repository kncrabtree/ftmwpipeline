"""Shared figure styling for the report, CLI, and documentation plots.

The spine-free "bare" look is the default house style: no spines, no tick
marks (labels kept), and a faint major grid behind the data. Titles are
opt-in -- a plot function takes ``title=None`` to use its own default (the CLI
path), ``title=""`` to suppress it entirely (the report path, where the page
heading already labels the figure), or any truthy string to show that text.

Colors are the UC Davis brand palette
(https://communicationsguide.ucdavis.edu/brand-guide/colors): ``AGGIE_BLUE`` and
``AGGIE_GOLD`` are the primaries, ``BRAND_CYCLE`` is the categorical line/marker
cycle (set as the default via :func:`apply_color_cycle`), and
:func:`aggie_blue_cmap` / :func:`aggie_gold_cmap` are single-hue gradient
colormaps for simple magnitude plots. Genuinely divergent or
perceptually-uniform data should use a purpose-built colormap (e.g. ``viridis``)
instead of a brand gradient.
"""

from typing import Any, List, Optional

import numpy as np

_BARE_GRID_COLOR = "#dbe0e6"

# -- UC Davis brand palette -------------------------------------------------
# Primaries.
AGGIE_BLUE = "#022851"
AGGIE_GOLD = "#FFBF00"

# Single-hue tint scales, darkest -> lightest (the brand "Aggie Blue scale" and
# "Aggie Gold scale"). Used to build gradient colormaps.
AGGIE_BLUE_SCALE: List[str] = [
    "#022851",
    "#033266",
    "#1d4776",
    "#355b85",
    "#4f7094",
    "#6884a3",
    "#8198b2",
    "#9aadc2",
    "#b3c1d1",
    "#cdd6e0",
]
AGGIE_GOLD_SCALE: List[str] = [
    "#ffbf00",
    "#ffc519",
    "#ffcc33",
    "#ffd24c",
    "#ffd966",
    "#ffdf80",
    "#ffe599",
    "#ffecb2",
    "#fff2cc",
    "#fff9e5",
]

# Named colors from the brand's expanded/secondary palette (official names).
DOUBLE_DECKER = "#c10230"  # red
GUNROCK = "#0047ba"  # blue
QUAD = "#3dae2b"  # green
POPPY = "#f18a00"  # orange
PINOT = "#76236c"  # purple
ARBORETUM = "#00c4b3"  # teal
REDBUD = "#c6007e"  # magenta
MERLOT = "#79242f"  # burgundy
REDWOOD = "#266041"  # dark green
CABERNET = "#481268"  # deep violet
TAHOE = "#00b2e3"  # cyan
SUNFLOWER = "#ffdc00"  # yellow

# Categorical cycle (one strong representative per hue family), ordered for
# contrast on a white background. Aggie Blue leads; gold is reserved for fills
# and highlights (it is low-contrast as a line color on white).
BRAND_CYCLE: List[str] = [
    AGGIE_BLUE,
    DOUBLE_DECKER,
    QUAD,
    POPPY,
    PINOT,
    ARBORETUM,
    GUNROCK,
    REDBUD,
    MERLOT,
    REDWOOD,
]


def apply_color_cycle(mpl: Any) -> None:
    """Set the brand categorical cycle as the matplotlib default.

    Pass the ``matplotlib`` module; sets ``axes.prop_cycle`` on ``rcParams`` so
    unspecified line/marker colors come from :data:`BRAND_CYCLE`.
    """
    from cycler import cycler

    mpl.rcParams["axes.prop_cycle"] = cycler(color=BRAND_CYCLE)


def aggie_blue_cmap() -> Any:
    """Light-to-dark single-hue gradient colormap from the Aggie Blue scale."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("aggie_blue", AGGIE_BLUE_SCALE[::-1])


def aggie_gold_cmap() -> Any:
    """Light-to-dark single-hue gradient colormap from the Aggie Gold scale."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("aggie_gold", AGGIE_GOLD_SCALE[::-1])


def apply_bare_style(ax: Any) -> None:
    """Strip *ax* to a spine-free, tick-mark-free look with a light major grid."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, which="major", color=_BARE_GRID_COLOR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def set_log_spectrum_ylim(
    ax: Any, rms_noise: np.ndarray, top: float, y_max_factor: float
) -> None:
    """Apply the shared log-magnitude y-axis for a spectrum panel.

    Sets a log y-scale with the lower limit a decade below the median noise
    (floored at ``1e-12``) and the upper limit ``y_max_factor``-scaled headroom
    above the tallest feature ``top`` -- so the noise floor and the 100s-of-x
    stronger lines are both legible. ``y_max_factor`` only sets the headroom;
    its baseline of 25 leaves a 1.2x minimum so small factors do not clip peaks.
    Shared by the Stage 3 peak and Stage 4 window spectrum panels.
    """
    median_rms = float(np.median(rms_noise)) if len(rms_noise) else 1.0
    floor = max(median_rms * 0.1, 1e-12)
    ax.set_yscale("log")
    ax.set_ylim(floor, top * max(y_max_factor / 25.0, 1.2))


def resolve_title(title: Optional[str], default: str) -> str:
    """Resolve the opt-in title convention.

    ``None`` -> *default* (CLI/direct callers); ``""`` -> ``""`` (suppressed, the
    report path); any other string -> itself. Callers draw the title only when
    the result is truthy.
    """
    return default if title is None else title
