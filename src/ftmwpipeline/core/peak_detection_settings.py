"""
Standard Stage 3 peak-detection settings.

``PeakDetectionSettings`` is the single source of truth for the Stage 3
parameters across every surface:

* the public API signatures (``Pipeline.detect_peaks`` /
  ``ftmwpipeline.api.detect_peaks``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted record in ``processing_parameters/stage3_peaks``,
* the YAML preset interchange format.

The dataclass mirrors :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`,
:class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`,
and :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`: every field
is ``Optional`` with ``None`` meaning *unset* (fall through the resolution
chain). A *resolved* instance (produced by :func:`resolve`) has every
field filled with a hard default if no layer supplied a value.

The dataclass is structured into four sub-dataclasses grouping the knobs
by what they configure: ``promotion``, ``savgol``, ``primary_pass``, and
``gap_pass``. The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage3_peaks`` so each sub-block is independently
inspectable.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 3 today -- the Stage 2b ``τ_maj`` value the gap pass uses for its
matched-filter ``tau_basis_us`` is a *runtime value*, not a settings knob,
so Stage 3's impl reads it directly from the persisted
``stage2b_tau_calibration`` block rather than routing it through the
resolver. The layer is kept in the signature so a future cross-stage
recommender can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the module-level constants in
:mod:`ftmwpipeline.preprocessing.peak_detection`
(``DEFAULT_MIN_SNR``, ``DEFAULT_INTERNAL_MIN_SNR``,
``DEFAULT_WEAK_MEDIUM_SNR``, ``DEFAULT_MEDIUM_STRONG_SNR``) and in
:mod:`ftmwpipeline._internal.stage3_impl`
(``DEFAULT_PRIMARY_WINDOW``, ``PRIMARY_LEAKAGE_FLOOR_K``,
``GAP_LEAKAGE_FLOOR_K``, ``_DETECTION_ZPF``, ``_GAP_ACTIVE_ZPF``,
``_SG_FWHM_COVERAGE``, ``_SG_MIN_WINDOW``), plus the hardcoded ``sg_window=11`` and
``sg_order=3`` defaults inside ``detect_peaks_impl``. Those constants
are still imported by the kernel and orchestrator as their parameter
defaults; once every consumer reads from a resolved
``PeakDetectionSettings``, the constants become docstring-only.

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
class PromotionSubSettings:
    """SNR cutoffs governing detection-floor and user-grid promotion.

    ``min_snr`` is the user-grid promotion cutoff (which peaks move to
    Stage 4); ``internal_min_snr`` caps the floor at which detection
    actually runs on the internal zpf=1 grids (effective floor is
    ``min(internal_min_snr, min_snr)``). ``weak_medium_snr`` and
    ``medium_strong_snr`` are the classification bin edges.
    """

    min_snr: Optional[float] = knob_field(
        help="User-grid promotion cutoff: peaks at/above this SNR advance to "
        "Stage 4.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(2.0, 2.5, 3.0, 4.0, 5.0),
        cli=True,
        argtype=float,
    )
    internal_min_snr: Optional[float] = knob_field(
        help="Internal detection floor on the zpf grids (recovers lines "
        "apodization smears).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(1.5, 2.0, 2.5, 3.0),
    )
    weak_medium_snr: Optional[float] = knob_field(
        help="Weak/medium SNR classification boundary.",
        inst_sensitivity="Y",
        grid=(5.0, 10.0, 15.0, 20.0),
        cli=True,
        argtype=float,
    )
    medium_strong_snr: Optional[float] = knob_field(
        help="Medium/strong SNR classification boundary.",
        inst_sensitivity="Y",
        grid=(30.0, 50.0, 75.0, 100.0),
        cli=True,
        argtype=float,
    )


@dataclass
class SavgolSubSettings:
    """Savitzky-Golay second-derivative apex localizer knobs.

    ``sg_window`` is the primary-pass fixed window (odd, > sg_order).
    The gap-pass window is computed at runtime from the actual grid
    spacing and the line FWHM via ``sg_fwhm_coverage * FWHM / freq_step``
    rounded up to odd, floored at ``sg_min_window``.
    """

    sg_window: Optional[int] = knob_field(
        help="Primary-pass Savitzky-Golay window (bins, odd).",
        inst_sensitivity="N",
        grid=(7, 9, 11, 15),
        cli=True,
        argtype=int,
    )
    sg_order: Optional[int] = knob_field(
        help="Savitzky-Golay polynomial order.",
        inst_sensitivity="N",
        grid=(2, 3, 4),
        cli=True,
        argtype=int,
    )
    sg_fwhm_coverage: Optional[float] = knob_field(
        help="Gap-pass window target in line-FWHM units (window auto-derived).",
        inst_sensitivity="N",
        grid=(3.0, 4.0, 5.0),
    )
    sg_min_window: Optional[int] = knob_field(
        help="Minimum Savitzky-Golay window (polynomial stability floor).",
        inst_sensitivity="N",
        grid=(5, 7, 9),
    )


@dataclass
class PrimaryPassSubSettings:
    """Primary-pass apodization + zpf knobs.

    The primary pass runs on the active-region ``dt·rfft`` frame (the same frame
    as the gap pass and the active FT), zero-padded by
    ``detection_zpf``, on a strongly-windowed spectrum (``primary_window``) to
    suppress truncation sidelobes;
    ``min_exclusion_mhz`` is the half-width around every primary detection
    that the gap pass excludes from its mask. ``primary_leakage_floor_k``
    scales the continuous leakage-aware detection floor ``k·(S_coh/√M)·σ``
    added to the primary-pass threshold so that a strong line's coherent
    skirt ripple is not re-detected as weak lines (the primary pass has no
    hard leakage mask -- a hard mask would delete the strong lines that
    generate the coherence). ``0`` disables the floor.

    The ``noise_*`` fields are the scatter-estimator knobs for the primary's
    **own** per-bin σ, measured on its apodized active-FT spectrum (the second
    Stage 3 noise level, distinct from the unapodized Stage 2 authority: the
    Blackman-Harris window suppresses the leakage that inflates the boxcar
    authority σ on dense spectra, so the primary floor is genuinely lower and
    must be measured on its own spectrum, not propagated). They mirror the
    Stage 2 :class:`~ftmwpipeline.core.noise_settings.NoiseSettings` knobs and
    default to the same values; expose them here so the apodized-domain floor is
    tunable through the same Stage 3 settings the rest of the pass uses.
    """

    primary_window: Optional[str] = knob_field(
        help="Primary-pass apodization window (sidelobe suppression; affects "
        "positions only).",
        inst_sensitivity="N",
        grid=("blackmanharris", "blackman", "hann", "hamming"),
        cli=True,
        argtype=str,
    )
    min_exclusion_mhz: Optional[float] = knob_field(
        help="Half-width (MHz) around each primary peak the gap pass excludes "
        "from its mask.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.0, 0.05, 0.1, 0.2, 0.5),
        cli=True,
        argtype=float,
    )
    detection_zpf: Optional[int] = knob_field(
        help="Zero-padding factor for the active-region primary spectrum.",
        inst_sensitivity="N",
        grid=(1, 2, 3),
    )
    primary_leakage_floor_k: Optional[float] = knob_field(
        help="Scale on the primary leakage-aware floor k·(S_coh/√M)·σ " "(0 disables).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.0, 0.5, 1.0, 2.0, 3.0),
    )
    noise_window_mhz: Optional[float] = knob_field(
        help="Apodized-domain σ: scatter-MAD window width (MHz).",
        inst_sensitivity="Y",
        grid=(40.0, 60.0, 80.0, 120.0, 160.0),
    )
    noise_pedestal_mhz: Optional[float] = knob_field(
        help="Apodized-domain σ: high-pass running-median width (MHz).",
        inst_sensitivity="Y",
        grid=(10.0, 20.0, 40.0, 80.0),
    )
    noise_line_k: Optional[float] = knob_field(
        help="Apodized-domain σ: robust-σ multiple above which a bin " "self-masks.",
        inst_sensitivity="maybe",
        grid=(4.0, 6.0, 8.0, 12.0),
    )
    noise_n_iter: Optional[int] = knob_field(
        help="Apodized-domain σ: self-mask refinement iterations.",
        inst_sensitivity="N",
        grid=(1, 2, 3, 5),
    )
    noise_region_aware: Optional[bool] = knob_field(
        help="Apodized-domain σ: region-aware Rician correction switch.",
        inst_sensitivity="N",
        grid=(False, True),
    )
    noise_smoothing_mhz: Optional[float] = knob_field(
        help="Apodized-domain σ: broad lower-envelope median width (MHz; " "0=off).",
        inst_sensitivity="Y",
        grid=(0.0, 400.0, 800.0, 1200.0),
    )
    noise_smoothing_percentile: Optional[float] = knob_field(
        help="Apodized-domain σ: percentile of the broad smoothing (50=median).",
        inst_sensitivity="maybe",
        grid=(25.0, 50.0, 75.0),
    )
    noise_convolve_mhz: Optional[float] = knob_field(
        help="Apodized-domain σ: step-removing second-pass Gaussian σ (MHz; " "0=off).",
        inst_sensitivity="N",
        grid=(0.0, 100.0, 200.0, 400.0),
    )


@dataclass
class GapPassSubSettings:
    """Matched-filter gap-pass knobs.

    ``run_gap_pass`` enables/disables the pass. ``gap_active_zpf`` is the
    zero-padding factor for the active-region rfft (chosen so the Lorentzian
    FWHM lands at ~3 bins on the resulting grid; SavGol's operating range).
    ``gap_leakage_floor_k`` scales the continuous leakage-aware detection floor
    ``k·(S_coh/√M)·σ`` added to the gap-pass threshold -- the same mechanism
    the primary pass uses (``primary_leakage_floor_k``) -- so a strong line's
    coherent skirt ripple is not re-detected as weak lines. It replaces the
    former hard ``S_coh``-cutoff mask. ``0`` disables the floor.

    ``tau_basis_us`` (the matched filter's exponential time constant) is
    *not* a Stage 3 knob -- it is the upstream-feeder value the Stage 3
    impl picks up from the persisted Stage 2b ``τ_maj`` (or falls back to
    the Stage 1 user apodization).
    """

    run_gap_pass: Optional[bool] = knob_field(
        help="Enable the matched-filter gap pass (recovers weak "
        "apodization-suppressed lines).",
        inst_sensitivity="N",
        grid=(False, True),
        cli=True,
        is_flag=True,
        flag="--gap-pass",
    )
    gap_active_zpf: Optional[int] = knob_field(
        help="Zero-padding factor for the matched-filter active-region FFT.",
        inst_sensitivity="N",
        grid=(1, 2, 3),
    )
    gap_leakage_floor_k: Optional[float] = knob_field(
        help="Scale on the gap leakage-aware floor k·(S_coh/√M)·σ (0 disables; "
        "replaces the former hard S_coh mask).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.0, 1.0, 2.0, 3.0, 5.0),
    )


@dataclass
class PeakDetectionSettings:
    """Stage 3 peak-detection settings (see module docstring)."""

    promotion: PromotionSubSettings = field(default_factory=PromotionSubSettings)
    savgol: SavgolSubSettings = field(default_factory=SavgolSubSettings)
    primary_pass: PrimaryPassSubSettings = field(default_factory=PrimaryPassSubSettings)
    gap_pass: GapPassSubSettings = field(default_factory=GapPassSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        return not sf.any_field_set(self, _SUB_NAMES)


# Sub-dataclass field names on PeakDetectionSettings, in HDF5/YAML order.
_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


# Hard defaults per sub-dataclass. These mirror the module-level constants
# in ``preprocessing/peak_detection.py`` and ``_internal/stage3_impl.py``.
# Kept as inline literals (rather than imported from those modules) to
# keep ``core`` dependency-free from ``preprocessing`` / ``_internal``;
# the kernel modules' constants are the readable standard source and
# these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "promotion": {
        "min_snr": 3.0,
        "internal_min_snr": 2.0,
        "weak_medium_snr": 10.0,
        "medium_strong_snr": 50.0,
    },
    "savgol": {
        "sg_window": 11,
        "sg_order": 3,
        "sg_fwhm_coverage": 4.0,
        "sg_min_window": 5,
    },
    "primary_pass": {
        "primary_window": "blackmanharris",
        "min_exclusion_mhz": 0.0,
        "detection_zpf": 2,
        "primary_leakage_floor_k": 1.0,
        # Apodized-domain scatter knobs (mirror NoiseSettings hard defaults).
        "noise_window_mhz": 80.0,
        "noise_pedestal_mhz": 20.0,
        "noise_line_k": 8.0,
        "noise_n_iter": 3,
        "noise_region_aware": True,
        "noise_smoothing_mhz": 800.0,
        "noise_smoothing_percentile": 50.0,
        "noise_convolve_mhz": 200.0,
    },
    "gap_pass": {
        "run_gap_pass": True,
        "gap_active_zpf": 2,
        "gap_leakage_floor_k": 3.0,
    },
}


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def resolve(
    explicit: Optional[PeakDetectionSettings] = None,
    preset: Optional[PeakDetectionSettings] = None,
    persisted: Optional[PeakDetectionSettings] = None,
    recommended: Optional[PeakDetectionSettings] = None,
) -> PeakDetectionSettings:
    """Merge the four layers by precedence into a resolved ``PeakDetectionSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The
    ``recommended`` layer is reserved for a future upstream recommender;
    Stage 3 call sites currently pass ``None`` there.
    """
    layers = (explicit, persisted, preset, recommended)
    return sf.fill_resolved_subblocks(
        PeakDetectionSettings(),
        PeakDetectionSettings,
        _SUB_NAMES,
        _HARD_DEFAULTS,
        layers,
    )


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip
# ---------------------------------------------------------------------------
def to_attrs(settings: PeakDetectionSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields.
    """
    return sf.subblocks_to_attrs(settings, _SUB_NAMES, sf.default_encode)


def from_attrs(attrs: Dict[str, Any]) -> PeakDetectionSettings:
    return sf.subblocks_from_attrs(
        PeakDetectionSettings(),
        PeakDetectionSettings,
        _SUB_NAMES,
        attrs,
        sf.default_decode,
    )


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def to_yaml_dict(settings: PeakDetectionSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    return sf.subblocks_to_yaml_dict(settings, _SUB_NAMES, sf.default_encode)


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> PeakDetectionSettings:
    if data is None:
        return PeakDetectionSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    return sf.subblocks_from_yaml_dict(
        PeakDetectionSettings(),
        PeakDetectionSettings,
        _SUB_NAMES,
        data,
        sf.identity_coerce,
    )


def from_yaml(source: Union[str, Path]) -> PeakDetectionSettings:
    return from_yaml_dict(sf.load_yaml_source(source))


def load_preset(name_or_path: Union[str, Path]) -> PeakDetectionSettings:
    """Load a Stage 3 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"defaults"`` ->
    ``ftmwpipeline/presets/defaults.yaml``); paths load directly.
    Preset YAML wraps the Stage 3 settings inside a top-level ``stage3:``
    block (alongside optional ``stage2:`` / ``stage2b:`` / ``stage5:``
    blocks for other stages).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`PeakDetectionSettings` (no fields set) when
    the preset carries no ``stage3:`` block, so a stage-spanning preset
    that omits Stage 3 loads cleanly without producing spurious overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    PeakDetectionSettings
        The parsed preset; unset fields stay ``None`` so the resolver can
        fall through to higher-precedence layers.

    Raises
    ------
    FileNotFoundError
        If a bare name does not match any packaged preset, or the
        supplied path does not exist.
    """
    return sf.load_subblock_preset(
        PeakDetectionSettings, "stage3", name_or_path, from_yaml_dict
    )


def to_yaml(settings: PeakDetectionSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    return sf.dump_yaml(to_yaml_dict(settings))


__all__ = [
    "PromotionSubSettings",
    "SavgolSubSettings",
    "PrimaryPassSubSettings",
    "GapPassSubSettings",
    "PeakDetectionSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
