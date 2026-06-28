"""
Canonical Stage 5 fit settings.

``StageFitSettings`` is the single source of truth for the Stage 5 fitting
parameters across every surface:

* the public API signatures (``Pipeline.fit_peaks`` / ``api.fit_peaks``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage5_fit``,
* the YAML preset interchange format.

Every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has
every field filled with the hard default if no layer supplied a value.

The dataclass is structured into one top-level setting (``shape``) plus six
sub-dataclasses grouping the knobs by what they configure
(``tau``, ``seeder``, ``conservative``, ``penalties``, ``rescue``, ``thaw``).
The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage5_fit`` so each sub-block is independently
inspectable.

The ``_HARD_DEFAULTS`` nested dict mirrors the ``DEFAULT_*`` constants in the
fitting modules. Those constants are still imported by the fitting functions
as their parameter defaults; once every consumer reads from a resolved
``StageFitSettings``, the constants become docstring-only and can be removed.

This module is dependency-free within the package (stdlib + the local
``__None__`` HDF5 marker convention shared with ``io.fid_serialization``,
plus PyYAML for the preset interchange) so it can be imported from ``core``
without cycles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import settings_framework as sf
from .knob_metadata import knob_field
from .peak_shape import PeakShape

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"


# ---------------------------------------------------------------------------
# Instrument clock declaration (spur.clocks entries)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockSource:
    """One declared instrument clock source.

    Declare chain *fundamentals* (e.g. the 5760 / 5120 MHz synthesizer
    outputs), not derived products (11520, 40960) -- harmonics are
    generated, so the products come for free. ``locked`` marks a source
    referenced to the instrument's frequency standard (Rb): the locked
    fundamentals span the exact intermod lattice (multiples of their GCD),
    while *unlocked* sources (a free-running digitizer clock) predict a
    drifting tone family. See
    ``dev-docs/planning/instrument-clock-declaration.md``.
    """

    freq_mhz: float
    locked: bool = True
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freq_mhz": float(self.freq_mhz),
            "locked": bool(self.locked),
            "label": str(self.label),
        }


def coerce_clock_sources(
    value: Any,
) -> Optional[Tuple[ClockSource, ...]]:
    """Coerce a clocks-like value into a tuple of :class:`ClockSource`.

    Accepts ``None`` (unset), an empty/non-empty sequence of
    :class:`ClockSource` or mappings (the YAML list-of-dicts form), or a
    JSON string (the HDF5-attr encoding). Other inputs raise ``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == _NONE:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"cannot parse clock declaration from {value!r}") from e
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"clock declaration must be a sequence; got {type(value)}")
    out: List[ClockSource] = []
    for entry in value:
        if isinstance(entry, ClockSource):
            out.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"clock entry must be a mapping with 'freq_mhz'; got {entry!r}"
            )
        if "freq_mhz" not in entry:
            raise ValueError(f"clock entry missing 'freq_mhz': {entry!r}")
        unknown = set(entry) - {"freq_mhz", "locked", "label"}
        if unknown:
            raise ValueError(
                f"unknown clock entry keys {sorted(unknown)} in {entry!r} "
                f"(valid: freq_mhz, locked, label)"
            )
        freq = float(entry["freq_mhz"])
        if not freq > 0:
            raise ValueError(f"clock freq_mhz must be positive; got {freq}")
        out.append(
            ClockSource(
                freq_mhz=freq,
                locked=bool(entry.get("locked", True)),
                label=str(entry.get("label", "")),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Shape spec (discriminated union scaffold)
# ---------------------------------------------------------------------------
@dataclass
class ShapeSpec:
    """Per-line envelope shape discriminator.

    ``kind`` is the :class:`PeakShape` enum -- ``LORENTZIAN`` /
    ``GAUSSIAN`` today, with ``VOIGT`` anticipated. Shape-specific parameter
    blocks (e.g. ``voigt_params: Optional[VoigtParams]``) attach here when
    Voigt support lands; no API churn elsewhere is required.
    """

    kind: PeakShape = PeakShape.LORENTZIAN

    @classmethod
    def coerce(cls, value: Any) -> Optional["ShapeSpec"]:
        """Coerce a shape-like value into a :class:`ShapeSpec` (or ``None``).

        Accepts the dataclass itself, a :class:`PeakShape`, a string member
        (``"gaussian"``), a mapping like ``{"kind": "gaussian"}``, or ``None``.
        Other inputs raise ``ValueError``.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, PeakShape):
            return cls(kind=value)
        if isinstance(value, str):
            return cls(kind=PeakShape.coerce(value))
        if isinstance(value, Mapping):
            if "kind" not in value:
                raise ValueError(
                    f"ShapeSpec mapping must carry a 'kind' key; got {value!r}"
                )
            return cls(kind=PeakShape.coerce(value["kind"]))
        raise ValueError(f"cannot coerce {value!r} to ShapeSpec")


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per HDF5 subgroup / YAML block)
# ---------------------------------------------------------------------------
@dataclass
class TauSubSettings:
    """Stage 5 τ handling (initial guess, bounds, free-vs-fixed, anchoring)."""

    tau0_us: Optional[float] = knob_field(
        help="Starting shared decay τ₀ (µs); None = runtime fallback "
        "(Stage 2b / T/3).",
        tier="advanced",
        inst_sensitivity="maybe",
        grid=(None, 3.0, 5.0, 8.0),
        cli=True,
        argtype=float,
    )
    fit_tau: Optional[bool] = knob_field(
        help="Hold the per-window tau fixed at tau0 (default: free).",
        tier="advanced",
        inst_sensitivity="N",
        cli=True,
        is_flag=True,
        flag="--fit-tau",
    )
    max_decay_factor: Optional[float] = knob_field(
        help="τ bounds multiplier: τ ∈ [τ₀/k, τ₀·k].",
        tier="advanced",
        inst_sensitivity="N",
        grid=(3.0, 5.0, 8.0),
        cli=True,
        argtype=float,
    )
    fit_tau_min_snr: Optional[float] = knob_field(
        help="In-window SNR above which τ is freed (the free-τ floor is the max "
        "of this and conservative.weak_window_snr_threshold; 10 = the "
        "weak-window floor).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(10.0, 25.0, 50.0, 100.0),
    )
    tau_penalty_lambda: Optional[float] = knob_field(
        help="Strength of the bidirectional Gaussian prior on τ.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(10.0, 50.0, 100.0),
    )
    tau_penalty_n_sigma: Optional[float] = knob_field(
        help="τ-bound half-width in units of σ_τ from Stage 2b.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(3.0, 5.0, 8.0),
    )
    tau_maj_override_us: Optional[float] = None
    sigma_tau_override_us: Optional[float] = None
    per_band_tau: Optional[bool] = knob_field(
        help="Route τ to per-band majorities (True) or a single band-wide τ "
        "(False).",
        tier="advanced",
        inst_sensitivity="maybe",
        grid=(False, True),
        cli=True,
        is_flag=True,
        flag="--per-band-tau",
    )


@dataclass
class SeederSubSettings:
    """Conservative-fit blend-aware seeder thresholds."""

    seeder_rchi2: Optional[float] = knob_field(
        help="χ²ᵣ threshold that triggers the K=2/3 blend-aware re-seed.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1.2, 1.5, 2.0),
    )
    seeder_straddle_factor: Optional[float] = knob_field(
        help="Re-seed offset spacing in line-FWHM units.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.5, 1.0, 1.5),
    )
    seeder_max_k: Optional[int] = knob_field(
        help="Maximum blend-escalation depth.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(2, 3, 4),
    )


@dataclass
class ConservativeSubSettings:
    """Add-one-peak loop gates and per-call caps."""

    significance: Optional[float] = knob_field(
        help="F-test significance α for add-one-peak acceptance.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.01, 0.05, 0.1),
    )
    max_peaks: Optional[int] = knob_field(
        help="Hard cap on the final peak count per window; 0 = no cap "
        "(candidate/patience-bounded).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0, 8, 16),
    )
    patience: Optional[int] = knob_field(
        help="Consecutive-rejection patience before the add loop stops.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1, 2, 3),
    )
    min_separation_factor: Optional[float] = knob_field(
        help="Minimum peak separation (FWHM units; unresolvable below).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.5, 1.0, 1.5),
    )
    min_pair_separation_factor: Optional[float] = knob_field(
        help="Post-escalation pair-separation floor (FWHM units).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.25, 0.5, 0.75),
    )
    min_pair_separation_resolution_factor: Optional[float] = knob_field(
        help="Resolution-referenced pair floor (1/T_active elements).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.5, 1.0, 1.5),
    )
    n_eff_kind: Optional[str] = None
    weak_window_snr_threshold: Optional[float] = knob_field(
        help="In-window SNR floor for free-τ eligibility (hold τ fixed below).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(5.0, 10.0, 15.0, 20.0),
    )
    max_nfev: Optional[int] = knob_field(
        help="Solver evaluation cap per window.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1000, 2000, 4000),
    )


@dataclass
class PenaltySubSettings:
    """Phase / amplitude soft-penalty weights."""

    phase_penalty_lambda: Optional[float] = knob_field(
        help="Phase-difference soft-penalty strength.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(50.0, 100.0, 200.0),
    )
    phase_penalty_cutoff_fwhm: Optional[float] = knob_field(
        help="Phase-penalty range (FWHM units; zero in quadrature).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1.0, 2.0, 3.0),
    )
    amp_penalty_lambda: Optional[float] = knob_field(
        help="Amplitude-floor soft-penalty strength.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(5.0, 10.0, 20.0),
    )
    amp_max_headroom: Optional[float] = knob_field(
        help="Hard amplitude ceiling as a multiple of 2·max_data/τ_eff_min.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(2.0, 3.0, 5.0),
    )


@dataclass
class RescueSubSettings:
    """Residual-rescue B-loop knobs."""

    max_rounds: Optional[int] = knob_field(
        help="Maximum residual-rescue iterations per window (safety cap).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1, 3, 5, 8),
        cli=True,
        argtype=int,
        flag="--max-residual-rescue-rounds",
    )
    snr_threshold: Optional[float] = knob_field(
        help="Residual-peak detection floor (nominates generously; the F-test "
        "gates acceptance).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(2.0, 2.5, 3.0, 4.0),
        cli=True,
        argtype=float,
        flag="--rescue-snr-threshold",
    )
    prominence_threshold: Optional[float] = knob_field(
        help="Residual-peak prominence threshold for candidate nomination.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(1.5, 2.0, 3.0, 4.0),
    )
    cleanup_significance: Optional[float] = knob_field(
        help="F-test significance for the remove-and-refit post-rescue cleanup.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.01, 0.05, 0.1),
    )
    merge_separation_factor: Optional[float] = knob_field(
        help="AICc-gated merge threshold above resolution (FWHM units).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.25, 0.5, 0.75),
    )
    structural_merge_factor: Optional[float] = knob_field(
        help="Sub-resolution merge floor: pairs closer than this (FWHM units) "
        "collapse unconditionally.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.25, 0.5, 0.75),
    )
    overfit_amp_ratio_band: Optional[float] = knob_field(
        help="Upper bound (1/T_active elements) of the amplitude-ratio merge tier "
        "that collapses supra-resolution shape-error absorbers.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1.0, 1.5, 2.0),
    )
    overfit_amp_ratio_threshold: Optional[float] = knob_field(
        help="Amplitude ratio above which a pair in the band collapses as an "
        "absorber (0 disables).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.0, 4.0, 6.0, 10.0),
    )
    final_add_snr_threshold: Optional[float] = knob_field(
        help="Strong residual-candidate SNR above which a final warm-started "
        "add-from-convergence is attempted (recovers companion lines the "
        "mid-fit seeder rejected; the conservative AICc gate still decides). "
        "0 disables.",
        tier="advanced",
        inst_sensitivity="Y",
        grid=(8.0, 10.0, 12.0, 15.0),
    )


@dataclass
class ThawSubSettings:
    """Local-thaw + structural-replan orchestration."""

    max_thaw_rounds: Optional[int] = knob_field(
        help="Maximum local-thaw iterations (re-fit a frozen contributor; 0 "
        "disables).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0, 1, 2, 3),
        cli=True,
        argtype=int,
    )
    max_replan_rounds: Optional[int] = knob_field(
        help="Maximum structural-replan iterations (window-boundary merges; 0 "
        "disables).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0, 1, 2, 3),
        cli=True,
        argtype=int,
    )
    residual_edge_threshold: Optional[float] = knob_field(
        help="S_coh threshold for a residual-edge-coherence boundary violation "
        "(the thaw / replan trigger).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(4.0, 6.0, 8.0, 10.0, 12.0),
        cli=True,
        argtype=float,
    )
    residual_edge_m: Optional[int] = knob_field(
        help="Band width (bins) for residual edge-coherence detection.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(16, 32, 48, 64),
        cli=True,
        argtype=int,
    )


@dataclass
class BaselineSubSettings:
    """Leakage-wing complex-baseline nuisance term knobs.

    An evidence-triggered low-order complex baseline ``B(u) = Σ_{k≤p}
    (a_k + i b_k)(u/u_s)^k`` added to a window's fit to absorb the coherent
    residual a neighboring strong line's mismodeled leakage skirt -- or the
    summed far-wings of the many lines the discrete contributors cannot fully
    subtract -- leaves behind. Fires where ``residual_edge_coherence`` exceeds
    ``edge_threshold`` (a coherent edge wing) OR where an order-``p`` polynomial
    explains the residual above ``smooth_threshold`` chi-squared per added dof (a
    smooth in-band leakage pedestal); fit jointly with the free lines and a
    re-freed ``tau`` so the flexibility is priced into the per-line
    uncertainties. See ``dev-docs/planning/stage5-leakage-wing-baseline.md``.
    """

    enabled: Optional[bool] = knob_field(
        help="Master switch for the evidence-triggered leakage-wing baseline " "term.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(False, True),
    )
    order: Optional[int] = knob_field(
        help="Baseline polynomial order (0 = const, 1 = linear; higher overfits).",
        tier="advanced",
        inst_sensitivity="maybe",
        grid=(0, 1),
    )
    edge_threshold: Optional[float] = knob_field(
        help="S_coh threshold (max residual edge) gating the leakage-wing "
        "baseline refit.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(2.5, 3.5, 5.0, 8.0),
    )
    smooth_threshold: Optional[float] = knob_field(
        help="Smooth-residual F-test (chi2-drop/dof) gating the baseline on an "
        "in-band leakage pedestal.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(20.0, 50.0, 100.0, 200.0),
    )


@dataclass
class SpurSubSettings:
    """Clock/LO-spur detection + masking knobs.

    A spur is a persistent CW tone (clock harmonic): a single-bin delta no
    finite-T line shape can represent. The gate is
    ``integer-MHz ∧ (frequency-domain narrow ∨ Stage 2b flat/saturated)``;
    detected spurs are dropped from peak nomination and excluded from the
    residual / chi-squared. See ``dev-docs/planning/stage5-spur-masking.md``.

    ``clocks`` is the declarative instrument clock tree
    (:class:`ClockSource` entries). When non-empty it replaces the
    integer-MHz nomination anchor with the locked-clock intermod *lattice*
    (multiples of the locked fundamentals' GCD, tested in both the
    molecular and baseband frames), flips the on-lattice evidence burden
    (``lattice_decay_ratio``), and adds a drifting-tone lane for unlocked
    clocks (``drift_*`` knobs). Empty/unset = legacy integer-MHz behavior.
    See ``dev-docs/planning/instrument-clock-declaration.md``.
    """

    enabled: Optional[bool] = knob_field(
        help="Master switch for clock/LO-spur detection + masking.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(False, True),
    )
    integer_tol_mhz: Optional[float] = knob_field(
        help="Max distance (MHz) from an integer MHz for the spur gate's hard "
        "integer requirement (~½ active-FT bin).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.02, 0.04, 0.08, 0.16),
    )
    narrowness_ratio: Optional[float] = knob_field(
        help="max(neighbor)/peak below which an integer-MHz bin is "
        "sub-resolution narrow (a CW tone vs a real line with a skirt).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.2, 0.3, 0.4, 0.5),
    )
    snr_threshold: Optional[float] = knob_field(
        help="Peak-bin / σ_c floor for the frequency-domain spur detector.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(3.0, 5.0, 8.0, 12.0),
    )
    mask_half_width_bins: Optional[int] = knob_field(
        help="Residual-mask half-width (active-FT bins) around a detected spur.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(1, 2, 3, 4),
    )
    use_stft_catalog: Optional[bool] = knob_field(
        help="Consume the persisted Stage 2b flat-spur (saturated) catalog as "
        "the gate's persistence half; False = frequency-domain detector only.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(False, True),
    )
    clocks: Optional[Tuple[ClockSource, ...]] = None
    lattice_decay_ratio: Optional[float] = None
    drift_window_mhz: Optional[float] = None
    drift_band_ratio: Optional[float] = None
    drift_min_snr: Optional[float] = None
    mask_target_residual_snr: Optional[float] = None
    mask_max_half_width_bins: Optional[int] = None
    chirp_response_gate_ratio: Optional[float] = None
    chirp_response_protect_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        # Tolerant ingestion: YAML hands list-of-dicts, HDF5 hands a JSON
        # string; normalize both to the canonical tuple-of-ClockSource.
        if self.clocks is not None and not (
            isinstance(self.clocks, tuple)
            and all(isinstance(c, ClockSource) for c in self.clocks)
        ):
            self.clocks = coerce_clock_sources(self.clocks)


@dataclass
class DoubletAlternativeSubSettings:
    """Post-fit doublet-alternative adjudication pass knobs.

    An observation-only pass that, for each adjacent fitted pair with
    separation ``<= k_res * (1/T_active)`` and weak/strong amplitude ratio
    ``>= r_min``, refits the window with the pair collapsed to one peak and
    records statistics (Δχ², ΔAICc, orthogonal-evidence score). The pass
    never changes any fitted peak. ``DEFAULT_DOUBLET_K_RES`` and
    ``DEFAULT_DOUBLET_R_MIN`` in
    :mod:`ftmwpipeline.fitting.doublet_alternative` are the canonical source
    for the default values mirrored in :data:`_HARD_DEFAULTS`.
    """

    enabled: Optional[bool] = knob_field(
        help="Master switch for the post-fit doublet-alternative observation pass "
        "(attaches records, never modifies fitted peaks).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(False, True),
    )
    k_res: Optional[float] = knob_field(
        help="Sub-resolution separation threshold (1/T_active elements) for "
        "doublet adjudication; pairs closer than k_res are evaluated.",
        tier="advanced",
        inst_sensitivity="N",
        grid=(1.0, 1.5, 2.0, 2.5),
    )
    r_min: Optional[float] = knob_field(
        help="Minimum amplitude ratio for the weaker member to trigger doublet "
        "evaluation (suppresses ghost pairs beside strong lines).",
        tier="advanced",
        inst_sensitivity="N",
        grid=(0.02, 0.05, 0.1),
    )


@dataclass
class PeakSurvivalSubSettings:
    """Post-fit peak-survival pass: SNR-floor prune, degenerate-overfit collapse,
    bright-neighbor sidelobe prune, and the degenerate merge trial.

    After Stage 5 produces the fitted :class:`~ftmwpipeline.core.data_structures.SpectrumFit`,
    four automatic cuts run (``origin == "user"`` peaks are immune to all):

    1. **SNR-floor prune.** Drop every automatic-origin fitted peak whose
       post-fit ``snr`` falls below the survival floor, then drop windows left
       empty; partially-pruned windows are refitted so survivors stay honest.
       Finite SNR only -- ``None``/NaN snr peaks are kept unconditionally. The
       floor **tracks the Stage 3 promotion cutoff**: by default it is that
       cutoff scaled by ``snr_survival_factor`` (so a stricter detection
       threshold raises the survival bar in step, and a survivor never sits
       below the SNR that admitted it). Set ``snr_survival_floor`` to pin an
       absolute floor instead, which overrides the factor.
    2. **Degenerate-pair merge.** Merge a close pair into one line, within
       ``collapse_max_separation_res`` resolution elements (``1 / T_active``),
       when a member is not individually constrained -- either its amplitude
       variance-inflation factor ``VIF = (amplitude_error / amplitude) * snr``
       reaches ``vif_collapse_threshold`` (the singular, brightness-invariant
       degeneracy) **or** its fractional amplitude uncertainty
       ``amplitude_error / amplitude`` reaches ``collapse_frac_unc_threshold``
       (the sub-resolution over-split band at VIF 4-25 the VIF gate alone
       leaves behind). The fractional-uncertainty path applies only within the
       tighter ``collapse_frac_unc_max_separation_res`` (deep sub-resolution),
       since a high fractional uncertainty at a marginally-resolvable separation
       is modest SNR on a real doublet, not an over-split.
       Policy: a sub-resolution *split* is a high-bar claim a prior-free fit
       cannot support, and the ambiguous band is ~92% over-splits, so the
       default is to merge (VIF<4 identifiable = keep, VIF>=4 degenerate =
       merge) and let the user opt into a split with catalog support. The merge
       overrides chi^2 / AICc unconditionally (at high SNR chi^2 is a lineshape
       floor that rewards the spurious split; a high post-merge chi^2 is the
       irreducible unresolved-structure floor, never evidence for two resolvable
       lines) and the sweep iterates to a fixpoint, including over the all-free
       relax that can re-split a dense window. A chained fold is bounded so it
       cannot cross a resolvable gap into a real neighbor. Merged windows are
       flagged ``auto_merged_review`` for overrule.
    3. **Bright-neighbor sidelobe prune.** Remove an automatic-origin peak that
       sits inside a *brighter* line's lineshape skirt -- separation within
       ``min(sidelobe_prune_max_separation_res, SHAPE_ERROR_REACH_KAPPA *
       snr_bright / snr_self)`` resolution elements (the finite-T sinc shadow,
       brightness-scaled, the same reach the Stage 6 candidate ledger uses) --
       then refit. Such a peak is the bright line's lineshape artifact, not a
       molecular line; removing it **raises** ``chi2r`` (the artifact was
       absorbing the bright line's non-Lorentzian lineshape error), and that is
       accepted: the goal is a reliable line list, not a low ``chi2r`` -- the
       residual lineshape error then surfaces honestly through ``epsilon`` /
       ``worst_eps`` rather than as a spurious line. The reach is capped because a
       very bright / very faint pair's unbounded reach extends past where a
       feature is resolved (and could be a real faint line); beyond the cap the
       Blackman-Harris apodization pass is the realness arbiter. A
       comparable-brightness pair has a sub-``kappa`` reach and is left to the
       collapse (2), so the two passes do not overlap.
    4. **Degenerate merge trial.** In the band
       ``(collapse_frac_unc_max_separation_res, collapse_max_separation_res]`` --
       past the unconditional collapse's reach -- a close pair where BOTH members
       are clearly degenerate (fractional amplitude uncertainty >=
       ``degenerate_trial_frac``; a sidelobe never qualifies, its bright parent is
       well determined) is trial-merged to its centroid and the merge **kept only
       if** ``chi2r`` does not rise by more than ``degenerate_trial_chi2r_rel_tol``
       (relative). The merge holding distinguishes a genuine over-split (kept) from
       a real but poorly-conditioned doublet (rejected, ``chi2r`` blows up) -- the
       one signal that separates them in this marginally-resolved band. Accepted
       merges flag ``auto_merged_review`` like (2).

    Defaults: ``enabled`` True, ``snr_survival_factor`` 1.1 (the floor tracks
    1.1x the Stage 3 promotion cutoff; ``snr_survival_floor`` unset = no absolute
    override), ``vif_collapse_threshold`` 25.0, ``collapse_frac_unc_threshold``
    0.15, ``collapse_frac_unc_max_separation_res`` 0.5,
    ``collapse_max_separation_res`` 1.0, ``sidelobe_prune_max_separation_res`` 2.5
    (0.0 disables), ``degenerate_trial_frac`` 0.5 (0.0 disables),
    ``degenerate_trial_chi2r_rel_tol`` 0.5. See
    ``dev-docs/planning/stage6-peak-survival.md``.
    """

    enabled: Optional[bool] = None
    # Absolute survival floor; unset (None) = derive from the Stage 3 promotion
    # cutoff via ``snr_survival_factor``. A set value overrides the factor.
    snr_survival_floor: Optional[float] = None
    snr_survival_factor: Optional[float] = None
    vif_collapse_threshold: Optional[float] = None
    collapse_frac_unc_threshold: Optional[float] = None
    collapse_frac_unc_max_separation_res: Optional[float] = None
    collapse_max_separation_res: Optional[float] = None
    sidelobe_prune_max_separation_res: Optional[float] = None
    degenerate_trial_frac: Optional[float] = None
    degenerate_trial_chi2r_rel_tol: Optional[float] = None
    drop_empty_windows: Optional[bool] = None
    drop_spur_only_windows: Optional[bool] = None


@dataclass
class StageFitSettings:
    """Stage 5 fit settings (see module docstring)."""

    shape: Optional[ShapeSpec] = None
    tau: TauSubSettings = field(default_factory=TauSubSettings)
    seeder: SeederSubSettings = field(default_factory=SeederSubSettings)
    conservative: ConservativeSubSettings = field(
        default_factory=ConservativeSubSettings
    )
    penalties: PenaltySubSettings = field(default_factory=PenaltySubSettings)
    rescue: RescueSubSettings = field(default_factory=RescueSubSettings)
    thaw: ThawSubSettings = field(default_factory=ThawSubSettings)
    spur: SpurSubSettings = field(default_factory=SpurSubSettings)
    baseline: BaselineSubSettings = field(default_factory=BaselineSubSettings)
    doublet_alternative: DoubletAlternativeSubSettings = field(
        default_factory=DoubletAlternativeSubSettings
    )
    peak_survival: PeakSurvivalSubSettings = field(
        default_factory=PeakSurvivalSubSettings
    )

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        if self.shape is not None:
            return False
        return not sf.any_field_set(self, _SUB_NAMES)


# Sub-dataclass field names on StageFitSettings, in HDF5/YAML order.
_SUB_NAMES = (
    "tau",
    "seeder",
    "conservative",
    "penalties",
    "rescue",
    "thaw",
    "spur",
    "baseline",
    "doublet_alternative",
    "peak_survival",
)


# Hard defaults per sub-dataclass. These mirror the ``DEFAULT_*`` constants
# in ``fitting/window_fit.py``, ``fitting/residual_rescue.py``,
# ``fitting/plan_execution.py``, ``fitting/validation.py`` and
# ``_internal/stage5_impl.py``. Kept as inline literals (rather than imported
# from fitting/) to keep ``core`` dependency-free from ``fitting``; the
# fitting modules' constants are the readable canonical source and these
# must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "shape": {"kind": PeakShape.LORENTZIAN},
    "tau": {
        "max_decay_factor": 5.0,
        # The free-τ floor: τ is freed above max(this, weak_window_snr_threshold).
        # Defaults to the weak-window floor (10) so behavior is unchanged until
        # tuned upward; see fitting.window_fit.DEFAULT_FIT_TAU_MIN_SNR.
        "fit_tau_min_snr": 10.0,
        "tau_penalty_lambda": 50.0,
        "tau_penalty_n_sigma": 5.0,
        "per_band_tau": True,
        # ``tau0_us`` / ``fit_tau`` / overrides legitimately stay None
        # (tau0_us derives at runtime from Stage 2b / expf_us / T_active/3;
        # fit_tau defaults to True inside the impl; overrides are unset by
        # design until a user supplies the atomic pair).
    },
    "seeder": {
        "seeder_rchi2": 1.5,
        "seeder_straddle_factor": 1.0,
        "seeder_max_k": 3,
    },
    "conservative": {
        "significance": 0.05,
        "max_peaks": 0,
        "patience": 1,
        "min_separation_factor": 1.0,
        "min_pair_separation_factor": 0.5,
        "min_pair_separation_resolution_factor": 1.0,
        "n_eff_kind": "perplexity_log1p_snr",
        "weak_window_snr_threshold": 10.0,
        "max_nfev": 2000,
    },
    "penalties": {
        "phase_penalty_lambda": 100.0,
        "phase_penalty_cutoff_fwhm": 2.0,
        "amp_penalty_lambda": 10.0,
        "amp_max_headroom": 3.0,
    },
    "rescue": {
        "max_rounds": 5,
        "snr_threshold": 2.5,
        "prominence_threshold": 2.0,
        "cleanup_significance": 0.05,
        "merge_separation_factor": 0.5,
        "structural_merge_factor": 0.5,
        "overfit_amp_ratio_band": 1.5,
        "overfit_amp_ratio_threshold": 6.0,
        "final_add_snr_threshold": 10.0,
    },
    "thaw": {
        "max_thaw_rounds": 2,
        "max_replan_rounds": 2,
        "residual_edge_threshold": 8.0,
        "residual_edge_m": 32,
    },
    "spur": {
        # Spur masking defaults on: the gate is integer-MHz-anchored and
        # validated zero real-line false-positive on 2638. These mirror the
        # ``DEFAULT_*`` constants in ``fitting/spur_detection.py``.
        "enabled": True,
        "integer_tol_mhz": 0.04,
        "narrowness_ratio": 0.30,
        "snr_threshold": 5.0,
        "mask_half_width_bins": 2,
        "use_stft_catalog": True,
        # Clock declaration: empty = no declaration (legacy integer-MHz
        # anchor). The lattice/drift/mask-scaling knobs below only act when
        # a declaration is present, except mask scaling which also applies
        # to legacy gating when explicitly enabled (> 0).
        "clocks": (),
        "lattice_decay_ratio": 0.45,
        "drift_window_mhz": 0.3,
        "drift_band_ratio": 0.35,
        "drift_min_snr": 10.0,
        # SNR-scaled residual-mask half-width: half_width_bins ~=
        # snr / (pi * mask_target_residual_snr), floored at
        # mask_half_width_bins and capped at mask_max_half_width_bins.
        # 0 disables (fixed legacy width). Default off until the
        # cross-fixture calibration flips it (see the planning doc).
        "mask_target_residual_snr": 0.0,
        "mask_max_half_width_bins": 32,
        # Chirp-response pre-record anchor thresholds. The probe only runs
        # when acquisition segments are present (scope records); files without
        # segments behave byte-identically to before.
        "chirp_response_gate_ratio": 0.8,
        "chirp_response_protect_ratio": 0.3,
    },
    "baseline": {
        # Leakage-wing baseline defaults on. It fires on a coherent edge wing
        # (edge-coh > 3.5) OR a smooth in-band leakage pedestal (the order-p
        # F-test, ``smooth_threshold``). Order 4 follows the pedestal's ramp /
        # curvature while staying far too smooth to mimic a narrow line (every
        # window is >> 4 active-FT bins); the F-significance trigger is the
        # overfit guardrail. Mirrors ``DEFAULT_BASELINE_*`` in
        # ``fitting/plan_execution.py``.
        "enabled": True,
        "order": 4,
        "edge_threshold": 3.5,
        "smooth_threshold": 50.0,
    },
    "doublet_alternative": {
        # Doublet-alternative adjudication defaults on. ``k_res`` and ``r_min``
        # mirror ``DEFAULT_DOUBLET_K_RES`` / ``DEFAULT_DOUBLET_R_MIN`` in
        # ``fitting/doublet_alternative.py``.
        "enabled": True,
        "k_res": 1.5,
        "r_min": 0.05,
    },
    "peak_survival": {
        # Peak-survival pass defaults on. The SNR floor sits at the Stage-3
        # detection threshold (3.2 sigma): sub-floor automatic-origin peaks are
        # dust that slipped through the conservative gate. The VIF collapse
        # MERGES any amplitude-degenerate close pair: amplitude VIF >=
        # ``vif_collapse_threshold`` within < ``collapse_max_separation_res``
        # resolution elements; the sweep iterates to convergence. user-origin
        # peaks are immune. POLICY (prior-free): declaring a sub-resolution
        # *split* is a high-bar claim that needs catalog/physical support the
        # pipeline does not have, so the default is to merge the clearly-
        # degenerate ones (``review`` re-splits with catalog support). The VIF
        # threshold is set well above the identifiability floor because the two
        # error directions are NOT symmetric: merging a *resolved* doublet
        # destroys a real line (a single Lorentzian at the centroid fits the
        # trough between the two peaks -- observed catastrophe: 360 w36, a methyl
        # A/E doublet at 0.68 res, snr 120+82, merged -> chi2r 386 with the
        # dominant line gone), whereas leaving an over-split merely ships an extra
        # peak. A resolved doublet's amplitudes ARE individually constrained, so
        # its VIF stays moderate (<= ~20 across the fixtures: w36 17, w383 4,
        # w1096's real line 23), while a genuinely sub-resolution degenerate
        # over-split has unconstrained amplitudes -> VIF >> that (>= ~40, up to
        # 1e6: w139 40, w201 98, w134 476, w124 1e6). The default sits in that
        # gap so only unambiguous degeneracy collapses on VIF alone. The
        # fractional-uncertainty criterion (``collapse_frac_unc_threshold``)
        # catches the remaining VIF 4-25 over-splits, but only deep sub-resolution
        # (``collapse_frac_unc_max_separation_res``) so it cannot merge the
        # marginally-resolvable doublets the VIF gate protects.
        # ``collapse_max_separation_res`` 1.0 = the VIF/singular path considers
        # pairs up to one full resolution element; wider pairs are resolved and
        # kept.
        "enabled": True,
        # The survival floor tracks the Stage 3 promotion cutoff: the effective
        # floor is that cutoff times ``snr_survival_factor``. ``snr_survival_floor``
        # has no hard default (None = derive from the factor); set it to pin an
        # absolute floor that overrides the factor.
        "snr_survival_factor": 1.1,
        "vif_collapse_threshold": 25.0,
        # Second collapse criterion, on the *weak member*'s fractional amplitude
        # uncertainty ``amp_err / amp`` (= VIF / snr). The VIF gate alone leaves
        # a band of sub-resolution over-splits at VIF 4-25 (separations well
        # inside the guard, 0.2-0.5 res) unmerged: amplitudes too uncertain to
        # claim two lines, but not singular enough to clear 25. A weak member
        # with ``amp_err / amp`` >= this fraction is not individually
        # constrained, so the pair collapses. Calibrated to leave the resolvable
        # methyl A/E doublet 360 w36 (frac 14.3%, snr 120+82) just below the bar
        # while folding the genuine over-splits above it; merged windows still
        # flag ``auto_merged_review`` for catalog-supported re-split.
        "collapse_frac_unc_threshold": 0.15,
        # The fractional-uncertainty criterion fires only *deep* sub-resolution
        # (<= half a resolution element). The VIF/singular criterion can merge up
        # to the full ``collapse_max_separation_res`` because a singular
        # amplitude is unambiguous degeneracy at any sub-resolution separation;
        # a merely-high fractional uncertainty is not -- at 0.6-0.9 res a high
        # ``amp_err/amp`` is modest SNR on a *resolvable* doublet, not an
        # over-split, and merging it destroys a real line (363 w404 0.61 res,
        # w61 0.86 res -> post-merge chi2r misfit). 0.5 res = the unresolvability
        # boundary: the genuine over-splits cluster at 0.2-0.5 res, the
        # resolvable doublets at >= ~0.6 res.
        "collapse_frac_unc_max_separation_res": 0.5,
        "collapse_max_separation_res": 1.0,
        # Bright-neighbor lineshape-sidelobe prune. A faint fitted peak inside a
        # brighter line's near-field lineshape skirt (``sep_res <=
        # SHAPE_ERROR_REACH_KAPPA * snr_bright / snr_self``, the same predicate the
        # candidate ledger / F-2 install filter use) is the bright line's lineshape
        # artifact, not a molecular line, and is removed -- overriding chi2r (the
        # artifact was absorbing real lineshape error; a higher post-prune chi2r is
        # the honest lineshape floor, surfaced via the eps attention flag). The
        # brightness-scaled reach runs out to tens of resolution elements for a
        # very bright / very faint pair, where a feature is fully resolved and may
        # be a real faint line; this caps the prune to the near-field skirt (a few
        # resolution elements), beyond which the Blackman-Harris apodization pass is
        # the realness arbiter. 0.0 disables. Comparable-brightness degenerate pairs
        # have a sub-kappa reach and are handled by the VIF collapse, not here.
        "sidelobe_prune_max_separation_res": 2.5,
        # Degenerate-pair merge TRIAL (Type A over-splits the unconditional collapse
        # leaves in the (collapse_frac_unc_max_separation_res, collapse_max_
        # separation_res] band). A pair where BOTH members have fractional amplitude
        # uncertainty >= ``degenerate_trial_frac`` (a bright-line sidelobe never
        # qualifies -- its bright parent is well determined, so this does not
        # overlap the sidelobe prune) is trial-merged to its centroid and the merge
        # KEPT only if the post-merge chi2r does not rise by more than
        # ``degenerate_trial_chi2r_rel_tol`` (relative). The merge holding tells a
        # genuine over-split (kept) from a real but poorly-conditioned doublet
        # (rejected, chi2r blows up): 360 w13 1.57->1.92 (held, merged); 363 w61
        # 2.53->7.39 (rejected, kept split). Accepted merges flag auto_merged_review.
        # ``degenerate_trial_frac`` 0.0 disables the trial.
        "degenerate_trial_frac": 0.5,
        "degenerate_trial_chi2r_rel_tol": 0.5,
        # End-of-Stage-5 window cleanup: drop windows with no surviving fitted
        # peak (K=0 -- pure noise, no product), and drop a single-line window
        # whose sole peak sits on a confidently-instrumental gated spur (a
        # ``flat``/``saturated`` decay-probe verdict -- e.g. an ADC image or a
        # declared-clock tone leaking past its residual mask). Conservative:
        # single-line + instrumental-spur identity only, never an ambiguous
        # multi-line window; user-origin peaks are immune.
        "drop_empty_windows": True,
        "drop_spur_only_windows": True,
    },
}


# ---------------------------------------------------------------------------
# Value codecs: PeakShape -> member value, structured clock declaration ->
# JSON (attrs) / list-of-dicts (YAML); everything else uses the framework
# defaults. Decode is plain -- the shape lives at the top level and the
# clock tuple is rebuilt by ``SpurSubSettings.__post_init__``.
# ---------------------------------------------------------------------------
def _encode_value(field_name: str, value: Any) -> Any:
    if value is None:
        return _NONE
    if isinstance(value, PeakShape):
        return value.value
    if isinstance(value, tuple) and all(isinstance(c, ClockSource) for c in value):
        # Structured clock declaration -> compact JSON string (HDF5 attrs
        # are scalar; the dataclass __post_init__ decodes it on the way in).
        return json.dumps([c.to_dict() for c in value])
    return value


def _yaml_encode(field_name: str, value: Any) -> Any:
    if isinstance(value, tuple) and all(isinstance(c, ClockSource) for c in value):
        # Structured clock declaration -> plain list-of-dicts so
        # ``yaml.safe_dump`` renders the documented preset form.
        return [c.to_dict() for c in value]
    return value


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def resolve(
    explicit: Optional[StageFitSettings] = None,
    preset: Optional[StageFitSettings] = None,
    persisted: Optional[StageFitSettings] = None,
    recommended: Optional[StageFitSettings] = None,
) -> StageFitSettings:
    """Merge the four layers by precedence into a resolved ``StageFitSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The shape
    discriminator is resolved separately: the first non-``None`` ``ShapeSpec``
    across the layers wins, then the ``LORENTZIAN`` hard default.
    """
    layers = (explicit, persisted, preset, recommended)
    shape_resolved: Optional[ShapeSpec] = None
    for layer in layers:
        if layer is not None and layer.shape is not None:
            shape_resolved = layer.shape
            break
    if shape_resolved is None:
        shape_resolved = ShapeSpec(kind=_HARD_DEFAULTS["shape"]["kind"])
    merged = StageFitSettings(shape=shape_resolved)
    return sf.fill_resolved_subblocks(
        merged, StageFitSettings, _SUB_NAMES, _HARD_DEFAULTS, layers
    )


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip (drives both HDF5 and YAML serialization)
# ---------------------------------------------------------------------------
def to_attrs(settings: StageFitSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass + ``shape``).

    The shape is encoded as a nested dict ``{"kind": "<value>"}`` (or the
    ``__None__`` sentinel when unset) so future shape-specific parameter
    blocks can attach inside the same subgroup. Sub-dataclass values use
    ``__None__`` for unset fields.
    """
    if settings.shape is None:
        out: Dict[str, Any] = {"shape": _NONE}
    else:
        out = {"shape": {"kind": settings.shape.kind.value}}
    out.update(sf.subblocks_to_attrs(settings, _SUB_NAMES, _encode_value))
    return out


def from_attrs(attrs: Dict[str, Any]) -> StageFitSettings:
    """Inverse of :func:`to_attrs` (tolerant of missing sub-blocks)."""
    shape_raw = attrs.get("shape")
    shape_spec: Optional[ShapeSpec]
    if shape_raw is None or (isinstance(shape_raw, str) and shape_raw == _NONE):
        shape_spec = None
    elif isinstance(shape_raw, dict):
        shape_spec = ShapeSpec.coerce(shape_raw)
    elif isinstance(shape_raw, (str, PeakShape)):
        shape_spec = ShapeSpec.coerce(shape_raw)
    else:
        raise ValueError(f"cannot decode shape attrs from {shape_raw!r}")
    settings = StageFitSettings(shape=shape_spec)
    return sf.subblocks_from_attrs(
        settings, StageFitSettings, _SUB_NAMES, attrs, sf.default_decode
    )


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def to_yaml_dict(settings: StageFitSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``).

    Sparseness lets a preset author write only the fields they want to
    override. Round-trip through :func:`from_yaml_dict` reproduces the
    same dataclass (unset fields stay ``None``).
    """
    top: Dict[str, Any] = {}
    if settings.shape is not None:
        top["shape"] = settings.shape.kind.value
    return sf.subblocks_to_yaml_dict(settings, _SUB_NAMES, _yaml_encode, top=top)


def from_yaml_dict(data: Optional[Dict[str, Any]]) -> StageFitSettings:
    """Build a :class:`StageFitSettings` from a YAML-shaped mapping.

    Accepts a ``shape:`` shorthand string (``shape: gaussian``) at the top
    level. Each sub-dataclass block is a mapping of field name -> value;
    unknown keys raise ``ValueError`` so typos surface loudly.
    """
    if data is None:
        return StageFitSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = StageFitSettings()
    if "shape" in data:
        settings.shape = ShapeSpec.coerce(data["shape"])
    return sf.subblocks_from_yaml_dict(
        settings,
        StageFitSettings,
        _SUB_NAMES,
        data,
        sf.identity_coerce,
        allowed_top=("shape",),
    )


def from_yaml(source: Union[str, Path]) -> StageFitSettings:
    """Load a :class:`StageFitSettings` from a YAML file path or text."""
    return from_yaml_dict(sf.load_yaml_source(source))


def load_preset(name_or_path: Union[str, Path]) -> StageFitSettings:
    """Load a Stage 5 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"defaults"`` ->
    ``ftmwpipeline/presets/defaults.yaml``); paths load
    directly. Preset YAML may wrap the Stage 5 settings inside a
    top-level ``stage5:`` block (the supported convention, allowing parallel
    ``stage2b:`` / ``stage2:`` blocks for other stages), a legacy
    ``fit:`` block (accepted for back-compat with presets written before
    the per-stage wrapper landed), or carry the settings flat at the top
    level; all three forms parse identically. ``stage5:`` and ``fit:``
    must not both appear in the same file.

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the
    preset author. Sibling stage blocks (``stage2b:``, etc.) are
    ignored here; they belong to other stages' settings loaders.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    StageFitSettings
        The parsed preset; unset fields stay ``None`` so the resolver
        can fall through to higher-precedence layers.

    Raises
    ------
    FileNotFoundError
        If a bare name does not match any packaged preset, or the
        supplied path does not exist.
    """
    data = sf.read_preset_root(name_or_path)
    # Per-stage top-level blocks are the current convention; ``fit:`` is
    # the legacy spelling kept as a back-compat shim for presets written
    # before the per-stage wrapper landed (see
    # ``dev-docs/planning/settings-backfill.md`` § "Back-compat shims").
    has_fit = "fit" in data and isinstance(data["fit"], dict)
    has_stage5 = "stage5" in data and isinstance(data["stage5"], dict)
    if has_fit and has_stage5:
        raise ValueError(
            f"preset {name_or_path!r} carries both 'fit:' (legacy) and "
            f"'stage5:' (current) wrappers; pick one"
        )
    inner_block: Optional[Dict[str, Any]] = None
    if has_stage5:
        inner_block = dict(data["stage5"])
    elif has_fit:
        import warnings as _warnings

        _warnings.warn(
            f"preset {name_or_path!r}: top-level 'fit:' wrapper is "
            "deprecated; rename it to 'stage5:' (per-stage block "
            "convention -- see dev-docs/planning/settings-backfill.md "
            "back-compat shim #1)",
            DeprecationWarning,
            stacklevel=2,
        )
        inner_block = dict(data["fit"])
    if inner_block is not None:
        for meta in ("name", "description"):
            if meta in data and meta not in inner_block:
                inner_block[meta] = data[meta]
        return from_yaml_dict(inner_block)
    # No per-stage wrapper: treat the document root as Stage 5 settings,
    # but strip out sibling stage blocks (``stage2b:`` etc.) so they
    # don't trip ``from_yaml_dict``'s unknown-key rejection.
    flat = {
        k: v
        for k, v in data.items()
        if k not in ("stage2", "stage2b", "stage3", "stage4")
    }
    return from_yaml_dict(flat)


def to_yaml(settings: StageFitSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    return sf.dump_yaml(to_yaml_dict(settings))


__all__ = [
    "ShapeSpec",
    "TauSubSettings",
    "SeederSubSettings",
    "ConservativeSubSettings",
    "PenaltySubSettings",
    "RescueSubSettings",
    "ThawSubSettings",
    "BaselineSubSettings",
    "DoubletAlternativeSubSettings",
    "PeakSurvivalSubSettings",
    "StageFitSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
