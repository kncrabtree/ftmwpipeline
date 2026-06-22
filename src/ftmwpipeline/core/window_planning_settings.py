"""
Canonical Stage 4 window-planning settings.

``WindowPlanningSettings`` is the single source of truth for the Stage 4
parameters across every surface:

* the public API signatures (``Pipeline.assign_windows`` /
  ``ftmwpipeline.api.assign_windows``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage4_windows``,
* the YAML preset interchange format.

The dataclass mirrors the architectural template established by Stages 5,
2b, 2, and 3: every field is ``Optional`` with ``None`` meaning *unset*
(fall through the resolution chain). A *resolved* instance (produced by
:func:`resolve`) has every field filled with a hard default if no layer
supplied a value.

The dataclass is structured into four sub-dataclasses grouping the knobs
by what they configure: ``coherence``, ``clustering``, ``contributor``,
and ``leakage``. The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage4_windows`` so each sub-block is
independently inspectable.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 4 today -- Stage 4 has no automatic upstream recommender. The slot
is kept in the signature so a future cross-stage recommender (e.g. a
Stage 2b τ_maj-driven ``tau_us`` suggestion) can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the ``DEFAULT_*`` constants in
:mod:`ftmwpipeline.preprocessing.window_planning` and
:mod:`ftmwpipeline.preprocessing.edge_coherence`. Those constants are
still imported by the kernel as its parameter defaults; once every
consumer reads from a resolved ``WindowPlanningSettings``, the constants
become docstring-only and can be removed.

``leakage.tau_us`` is allowed to remain ``None`` after resolution: the
hard default is intentionally ``None`` (boxcar / undamped limit -- the
analytic leakage envelope reduces to ``2/(2π·Δf·T)``). A future
auto-feeder from the persisted Stage 2b ``τ_maj`` would fill this slot
via the resolver's *recommended* layer; until then, users override it
explicitly via ``settings.leakage.tau_us = ...``.

This module is dependency-free within the package (stdlib + PyYAML for
preset interchange) so it can be imported from ``core`` without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from . import settings_framework as sf
from .knob_metadata import knob_field


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per HDF5 subgroup / YAML block)
# ---------------------------------------------------------------------------
@dataclass
class CoherenceSubSettings:
    """Rolling edge-coherence statistic knobs (drive the spatial partition).

    ``edge_m`` and ``trim_m`` are the band widths used by the rolling-scan
    and trim-refinement coherence statistics; ``edge_threshold`` is the
    ``S_coh`` cutoff ``T_edge`` above which a frequency interval is
    considered leakage-touched.
    """

    edge_m: Optional[int] = knob_field(
        help="Band width (bins) for the rolling complex-edge coherence statistic.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(32, 48, 64, 96, 128),
        cli=True,
        argtype=int,
    )
    trim_m: Optional[int] = knob_field(
        help="Band width (bins) for coherence refinement after a leakage-region "
        "flag.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(16, 24, 32, 48),
        cli=True,
        argtype=int,
    )
    edge_threshold: Optional[float] = knob_field(
        help="S_coh cutoff (T_edge) for flagging leakage-touched regions that "
        "force window boundaries.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(4.0, 6.0, 8.0, 10.0, 12.0),
        cli=True,
        argtype=float,
    )


@dataclass
class ClusteringSubSettings:
    """Window-extent decisions.

    ``max_window_width_mhz`` is the width cap above which a window's peak
    content is split at its sparsest gaps; ``min_window_half_width_mhz``
    is the minimum half-width of a window built around an isolated weak line.
    ``max_peaks_per_window`` is the per-window promoted-peak cap; ``0`` (the
    default) disables it so a window is bounded only by ``max_window_width_mhz``.
    Bounding by width alone keeps the Stage 5 AICc-with-``n_eff`` gate's effective
    sample size large enough to self-regulate K on dense clusters; a fragmenting
    peak cap starved it and drove both under- and over-fit. The width cap (and the
    width-bounded strong-cluster merge) is what prevents a dense ultra-high-SNR
    spectrum from collapsing into one GHz-scale mega-window. A positive value
    restores an explicit cap and tracks the Stage 5 ``conservative.max_peaks``.
    ``max_window_width_points`` is the same width cap expressed in active-FT
    grid points -- the statistically portable form (bin width varies with
    acquisition length across instruments); ``0`` defers to the MHz cap, a
    positive value supersedes it. The hard default (96 points, ~8 MHz on the
    reference 2638 grid) is the small-window operating point the Stage 5
    window-invariant accept gates are calibrated against.
    ``min_window_half_width_points`` is the window margin in grid points -- the
    noise budget kept on each side of a window's outermost peak (the proto
    half-width and the post-construction trim budget). It supersedes the MHz form
    ``min_window_half_width_mhz`` when positive (the default), mirroring the
    width-cap MHz/points pair, and is decoupled from ``edge_m`` (the coherence
    band). The coherent range is ``trim_m <= margin <= max_window_width_points / 2``
    (so the edge statistic samples the noise margin and a lone line's window never
    exceeds the content cap); the hard default 32 is the tight end (== ``trim_m``).
    """

    max_window_width_mhz: Optional[float] = knob_field(
        help="Width cap (MHz); a window's peak content wider than this is split "
        "at its sparsest gaps.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(20.0, 30.0, 40.0, 60.0, 80.0),
        cli=True,
        argtype=float,
    )
    min_window_half_width_mhz: Optional[float] = knob_field(
        help="MHz form of the window margin; used only when "
        "min_window_half_width_points is 0 (the points form is the active "
        "default).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1.0, 2.0, 3.0, 4.0),
        cli=True,
        argtype=float,
    )
    min_window_half_width_points: Optional[int] = knob_field(
        help="Window margin in active-FT grid points -- the noise budget each "
        "side of a window's outermost peak (proto half-width and trim budget). "
        "Supersedes min_window_half_width_mhz when positive. Coherent range: "
        "trim_m..max_window_width_points/2.",
        tier="advanced",
        inst_sensitivity="Y",
        grid=(24, 32, 40, 48),
        cli=True,
        argtype=int,
    )
    max_peaks_per_window: Optional[int] = knob_field(
        help="Per-window promoted-peak cap; 0 = no cap (width-bounded). Windows "
        "over a positive cap are split at their sparsest gaps.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0, 8, 16, 32),
        cli=True,
        argtype=int,
    )
    max_window_width_points: Optional[int] = knob_field(
        help="Width cap in active-FT grid points (the portable form; bin width "
        "varies across instruments). 0 = defer to max_window_width_mhz; positive "
        "supersedes it.",
        tier="advanced",
        inst_sensitivity="Y",
        grid=(0, 64, 96, 128, 256),
        cli=True,
        argtype=int,
    )


@dataclass
class ContributorSubSettings:
    """Fixed-contributor freeze/attach decisions.

    ``min_freeze_snr`` is the freeze-eligibility SNR cutoff (a fixed
    contributor below this is flagged as a thaw-and-re-fit candidate
    rather than safely frozen); ``magnitude_attachment_threshold`` is the
    analytic-skirt-magnitude attachment rule (in units of σ_c on the
    target window) governing which strong promoted peaks are attached to
    a window's ``fixed_contributors``.
    """

    min_freeze_snr: Optional[float] = knob_field(
        help="SNR floor for fixed-contributor freeze-eligibility (below = thaw "
        "candidate).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(20.0, 35.0, 50.0, 75.0, 100.0),
        cli=True,
        argtype=float,
    )
    magnitude_attachment_threshold: Optional[float] = knob_field(
        help="Tier-1 contributor attachment: predicted mean-skirt threshold "
        "(σ_c units).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.05, 0.075, 0.1, 0.15, 0.2),
        cli=True,
        argtype=float,
    )


@dataclass
class LeakageSubSettings:
    """Leakage-envelope parameter.

    ``tau_us`` is the assumed shared time-domain decay constant used by
    the analytic finite-T leakage envelope. ``None`` (the hard default)
    means undamped/boxcar -- the envelope reduces to ``2/(2π·Δf·T)``. A
    future auto-feeder may populate this from the persisted Stage 2b
    ``τ_maj`` via the resolver's recommended layer.
    """

    tau_us: Optional[float] = knob_field(
        help="Decay constant (µs) for the analytic leakage-skirt envelope; None "
        "= boxcar (undamped) limit. A single band-wide scalar — Stage 2b τ is not "
        "auto-fed here; set it explicitly via the grid / settings= / preset=.",
        tier="advanced",
        inst_sensitivity="Y",
        grid=(None, 3.0, 6.0, 12.0),
        cli=True,
        argtype=float,
    )


@dataclass
class WindowPlanningSettings:
    """Stage 4 window-planning settings (see module docstring)."""

    coherence: CoherenceSubSettings = field(default_factory=CoherenceSubSettings)
    clustering: ClusteringSubSettings = field(default_factory=ClusteringSubSettings)
    contributor: ContributorSubSettings = field(default_factory=ContributorSubSettings)
    leakage: LeakageSubSettings = field(default_factory=LeakageSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        return not sf.any_field_set(self, _SUB_NAMES)


# Sub-dataclass field names on WindowPlanningSettings, in HDF5/YAML order.
_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


# Hard defaults per sub-dataclass. These mirror the ``DEFAULT_*`` constants
# in ``preprocessing/window_planning.py`` and
# ``preprocessing/edge_coherence.py``. Kept as inline literals (rather than
# imported from ``preprocessing/``) to keep ``core`` dependency-free from
# ``preprocessing``; the kernel module's constants are the readable canonical
# source and these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "coherence": {
        "edge_m": 64,
        "trim_m": 32,
        "edge_threshold": 8.0,
    },
    "clustering": {
        "max_window_width_mhz": 40.0,
        "min_window_half_width_mhz": 2.0,
        "min_window_half_width_points": 32,
        "max_peaks_per_window": 0,
        "max_window_width_points": 96,
    },
    "contributor": {
        "min_freeze_snr": 50.0,
        "magnitude_attachment_threshold": 0.1,
    },
    "leakage": {
        # ``tau_us`` legitimately stays None (boxcar / undamped limit).
    },
}


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def resolve(
    explicit: Optional[WindowPlanningSettings] = None,
    preset: Optional[WindowPlanningSettings] = None,
    persisted: Optional[WindowPlanningSettings] = None,
    recommended: Optional[WindowPlanningSettings] = None,
) -> WindowPlanningSettings:
    """Merge the four layers by precedence into a resolved ``WindowPlanningSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The
    ``recommended`` layer is reserved for a future upstream recommender;
    Stage 4 call sites currently pass ``None`` there.
    """
    layers = (explicit, persisted, preset, recommended)
    return sf.fill_resolved_subblocks(
        WindowPlanningSettings(),
        WindowPlanningSettings,
        _SUB_NAMES,
        _HARD_DEFAULTS,
        layers,
    )


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip
# ---------------------------------------------------------------------------
def to_attrs(settings: WindowPlanningSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields.
    """
    return sf.subblocks_to_attrs(settings, _SUB_NAMES, sf.default_encode)


def from_attrs(attrs: Dict[str, Any]) -> WindowPlanningSettings:
    return sf.subblocks_from_attrs(
        WindowPlanningSettings(),
        WindowPlanningSettings,
        _SUB_NAMES,
        attrs,
        sf.default_decode,
    )


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def to_yaml_dict(settings: WindowPlanningSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    return sf.subblocks_to_yaml_dict(settings, _SUB_NAMES, sf.default_encode)


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> WindowPlanningSettings:
    if data is None:
        return WindowPlanningSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    return sf.subblocks_from_yaml_dict(
        WindowPlanningSettings(),
        WindowPlanningSettings,
        _SUB_NAMES,
        data,
        sf.identity_coerce,
    )


def from_yaml(source: Union[str, Path]) -> WindowPlanningSettings:
    return from_yaml_dict(sf.load_yaml_source(source))


def load_preset(name_or_path: Union[str, Path]) -> WindowPlanningSettings:
    """Load a Stage 4 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"defaults"`` ->
    ``ftmwpipeline/presets/defaults.yaml``); paths load directly.
    Preset YAML wraps the Stage 4 settings inside a top-level ``stage4:``
    block (alongside optional ``stage2:`` / ``stage2b:`` / ``stage3:`` /
    ``stage5:`` blocks for other stages).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`WindowPlanningSettings` (no fields set) when
    the preset carries no ``stage4:`` block, so a stage-spanning preset
    that omits Stage 4 loads cleanly without producing spurious overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    WindowPlanningSettings
        The parsed preset; unset fields stay ``None`` so the resolver can
        fall through to higher-precedence layers.

    Raises
    ------
    FileNotFoundError
        If a bare name does not match any packaged preset, or the
        supplied path does not exist.
    """
    return sf.load_subblock_preset(
        WindowPlanningSettings, "stage4", name_or_path, from_yaml_dict
    )


def to_yaml(settings: WindowPlanningSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    return sf.dump_yaml(to_yaml_dict(settings))


__all__ = [
    "CoherenceSubSettings",
    "ClusteringSubSettings",
    "ContributorSubSettings",
    "LeakageSubSettings",
    "WindowPlanningSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
