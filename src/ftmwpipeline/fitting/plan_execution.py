"""
Plan-level execution for Stage 5 fitting.

Drive a Stage 4 :class:`~ftmwpipeline.core.data_structures.WindowPlan`
through to a fitted line list. This module owns the three pieces that make up
plan execution:

* **Fixed-contributor evaluation.** A
  :class:`~ftmwpipeline.core.data_structures.FixedContributor` names a strong line
  fit freely in its ``primary_window_id``. In a dependent window that line
  contributes only its frozen ``h_T`` skirt -- the line itself is not re-fit. The
  free-peak core (:func:`~ftmwpipeline.fitting.window_fit.fit_window` /
  :func:`~ftmwpipeline.fitting.window_fit.conservative_fit`) fits free peaks only,
  so the contributor's frozen model is *subtracted from the window data as a
  frozen background* before delegating to the conservative loop. After the fit,
  the background is added back to reconstruct the full model and residual.

* **DAG / batch execution order.** Windows are fit in
  :attr:`WindowPlan.topological_order` (equivalently, in ascending
  :attr:`FitWindow.batch`); a window's fixed contributors are pulled from the
  already-fit primary windows it depends on, so the dependency invariant is
  enforced by the walk order. Within a batch the windows are mutually
  independent, which keeps this pure-Python walk safe to parallelise later
  without changing its semantics.

* **Local thaw renegotiation.** After each window fit, a complex-edge coherence
  statistic (:mod:`ftmwpipeline.preprocessing.edge_coherence`) is run on the *fit
  residual* at the two window edges. Above-threshold coherence on an edge that
  faces a frozen contributor means the contributor's skirt was carried badly
  (or the contributor was not safe to freeze in the first place -- the
  ``freeze_eligible=False`` case the Stage 4 plan flagged). The contributor is
  then *thawed*: unfrozen and co-fit jointly with the dependent window and its
  primary window. This is the canonical 36350/36389 doublet case. Rounds are
  bounded for guaranteed termination.

The fit frame is the **active-portion FT**
(:mod:`ftmwpipeline.fitting.active_ft`) -- the rfft of just the active FID
samples, which is already in the ``[0, T]`` form ``h_T`` models. Each window is
sliced from the active-FT and its molecular-frequency grid is relabeled to a
signed baseband offset
(:func:`~ftmwpipeline.fitting.peak_model.to_baseband_offset`). No de-ramp is
needed: the active samples are referenced to ``t = 0`` directly.

This module fits a whole :class:`WindowPlan` -- traversing the dependency DAG,
fitting each window, freezing out-of-band contributors, and running the
edge-coherence renegotiation handshake (local thaw plus, for structural
coupling, a Stage 4 ``replan``). The light dataclasses defined here
(:class:`FrozenPeak`, :class:`ThawEvent`, :class:`WindowOutcome`,
:class:`PlanFitOutcome`) are the algorithm-side records that
:mod:`ftmwpipeline.fitting.result_conversion` composes into the persistent
:class:`~ftmwpipeline.core.data_structures.FittedPeak` /
:class:`~ftmwpipeline.core.data_structures.FittingResult` / the
``SpectrumFit`` aggregate.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    MergeRequest,
    Peak,
    Sideband,
    WindowPlan,
)
from ftmwpipeline.preprocessing.edge_coherence import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_TRIM_M,
    coherence_statistic,
)
from ftmwpipeline.preprocessing.window_planning import replan as stage4_replan

from ..utils.parallelism import resolve_worker_count
from . import validation
from .active_ft import ActiveFTResult
from .doublet_alternative import DoubletAdjudication, adjudicate_close_pairs
from .peak_model import (
    ModelPeak,
    PeakShape,
    model_spectrum,
    sideband_sign,
    to_baseband_offset,
)
from .residual_rescue import rescue_and_consolidate
from .spur_detection import GatedSpur, SpurMaskSpec, SpurSet
from .window_fit import (
    DEFAULT_MAX_DECAY_FACTOR,
    ConservativeFitResult,
    WindowFitConstraints,
    WindowFitResult,
    conservative_fit,
    derive_window_fit_constraints,
    evaluate_baseline,
    fit_window,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FrozenPeak",
    "ThawEvent",
    "RescueEvent",
    "ReplanContext",
    "ReplanEvent",
    "WindowOutcome",
    "PlanFitOutcome",
    "DEFAULT_RESIDUAL_EDGE_THRESHOLD",
    "DEFAULT_RESIDUAL_EDGE_M",
    "DEFAULT_MAX_THAW_ROUNDS",
    "DEFAULT_MAX_REPLAN_ROUNDS",
    "DEFAULT_BASELINE_ENABLED",
    "DEFAULT_BASELINE_ORDER",
    "DEFAULT_BASELINE_EDGE_THRESHOLD",
    "evaluate_fixed_contributor",
    "evaluate_edge_free_contributors",
    "subtract_frozen_background",
    "fit_window_with_fixed_contributors",
    "materialize_window",
    "residual_edge_coherence",
    "select_contributor_to_thaw",
    "local_thaw_cofit",
    "attempt_thaw_round",
    "execute_plan",
    "build_window_outcome",
    "fit_seeds_window_outcome",
]

NoiseLike = Union[float, np.ndarray]
SidebandLike = Union[Sideband, str]

# Residual edge-coherence renegotiation defaults (O5-6 -- starting points).
DEFAULT_RESIDUAL_EDGE_THRESHOLD = DEFAULT_EDGE_THRESHOLD
"""``S_coh`` threshold above which a residual edge triggers a thaw attempt."""

DEFAULT_RESIDUAL_EDGE_M = DEFAULT_TRIM_M
"""Band width (in spectrum bins) of the residual-edge coherence test."""

DEFAULT_MAX_THAW_ROUNDS = 2
"""Maximum thaw rounds per window, for guaranteed termination."""

DEFAULT_MAX_REPLAN_ROUNDS = 2
"""Maximum structural-replan rounds per :func:`execute_plan` call. Each round
applies at most one merge per disjoint window pair, so a chain of N adjacent
windows wanting to coalesce resolves in O(log2(N)) rounds."""

DEFAULT_BASELINE_ENABLED = True
"""Whether the evidence-triggered leakage-wing baseline term is applied."""

DEFAULT_BASELINE_ORDER = 4
"""Baseline polynomial order ``p``. A dense ultra-high-SNR spectrum carries a
smooth leakage *pedestal* -- the summed far-wings of the hundreds of lines the
discrete frozen contributors cannot fully subtract -- that ramps and curves
across a window; a constant cannot follow it, so the shared ``tau`` collapses to
absorb it (655 mode 2). Order 4 follows the pedestal while staying far too smooth
to mimic a line (every window is >= ~50 active-FT bins, >> the order), and the
F-significance trigger (:data:`DEFAULT_BASELINE_SMOOTH_THRESHOLD`) only commits it
where it is statistically warranted."""

DEFAULT_BASELINE_SMOOTH_THRESHOLD = 50.0
"""Smooth-residual trigger: the baseline also fires when an order-``p`` complex
polynomial explains the post-fit residual at more than this chi-squared drop per
added real degree of freedom (an F-test numerator). This catches the smooth
in-band leakage pedestal the edge-coherence trigger misses (it tests only the two
edges). On 655 the collapse windows score >~2000 here while clean windows sit
near the noise floor (~8), so the test self-gates against overfitting."""

DEFAULT_EDGE_FREE_ACCEPT_FRACTION = 0.95
"""Acceptance margin for an edge-free leakage skirt. The window is fit twice --
without the edge-free contributors (the byte-stable path a healthy window keeps,
its in-window leakage already covered by the const leakage-wing baseline) and
with them -- and the skirt is adopted only if it reduces the noise-weighted
residual sum of squares to at most this fraction of the no-skirt fit's. This
keeps the subtraction *evidence-triggered*: an orphaned bright neighbor's skirt
(360 w287: SSR drops ~10x) is adopted, while on a window the skirt would harm
(2638 w106, where the const baseline already handles the leakage) the no-skirt
fit stands unchanged, so the skirt never fights the baseline (open question O2).
The broken windows improve by far more than 5%, so the exact margin is not
sensitive."""

DEFAULT_BASELINE_EDGE_THRESHOLD = 3.5
"""``S_coh`` threshold (max of the two edges) above which a window's fit is
refit with the complex baseline enabled. A dedicated threshold well below the
thaw default (8.0): the thaw addresses a *missing real line* at the edge, the
baseline a *wrong skirt shape*. Empirically settled on 2638 (recall 0.89, zero
harmful fires); 2638-tuned, so instrument-tunable calibration debt."""

DEFAULT_EDGE_FREE_FREQ_REFINE = True
"""Whether the edge-free contributor read refines the line frequencies.

The joint complex least-squares read of
:func:`evaluate_edge_free_contributors` solves only the lines' linear
``(amplitude, phase)`` with the frequencies held at their Stage 3 detected
positions. On an ultra-high-SNR line a sub-bin frequency error mis-phases the
sharp core template enough that the linear solve recovers a substantially
low amplitude (the unmatched core residual is paid instead), and the
subtracted far-wing skirt then under-predicts the real pedestal by a factor
the ``kappa_skirt`` fidelity budget cannot cover -- the leftover coherent
wing is harvested as spurious peaks. With this enabled, a bounded
variable-projection refinement (frequencies free within ~1.5 grid steps,
amplitudes/phases re-solved linearly at each trial) is run per contributor
group before the final solve, recovering the core to a few percent and the
wing prediction to within the fidelity budget. ``False`` preserves the
fixed-frequency read."""


# ---------------------------------------------------------------------------
# Light per-window records
# ---------------------------------------------------------------------------
@dataclass
class FrozenPeak:
    """A :class:`FixedContributor` materialized in a dependent window's fit frame.

    The contributor's fitted ``(amplitude, frequency, phase)`` -- read from its
    ``primary_window``'s converged free-peak fit -- is mapped into the dependent
    window's signed baseband-offset coordinate; the frozen model term is just
    ``model_spectrum([model_peak], tau, T)`` evaluated on the dependent window's
    grid (note the shared ``tau`` is the dependent window's, not the primary's
    -- the leakage shape ``h_T`` carries one ``tau`` per window).

    Attributes
    ----------
    peak_index : int
        Stage 3 peak index of the strong line (links back to the plan).
    primary_window_id : int
        Window in which this line was fit freely.
    model_peak : ModelPeak
        The frozen model term, in the *dependent* window's offset frame.
    frequency_mhz : float
        Molecular frequency of the line (MHz), carried for diagnostics.
    freeze_eligible : bool
        Stage 4's eligibility flag, propagated for the thaw selection heuristic.
    edge_free : bool
        Propagated from the :class:`FixedContributor`. ``True`` means the
        ``model_peak`` was read self-contained from the active FT (no primary
        fit), so this contributor is excluded from the local-thaw handshake
        (a thaw needs the primary's converged fit, which an edge-free
        contributor has no dependency on).
    """

    peak_index: int
    primary_window_id: int
    model_peak: ModelPeak
    frequency_mhz: float
    freeze_eligible: bool = True
    edge_free: bool = False


@dataclass
class ThawEvent:
    """One local-thaw renegotiation record.

    Attributes
    ----------
    dependent_window_id : int
        Window whose post-fit residual edge triggered the thaw.
    primary_window_id : int
        Window the thawed contributor was originally fit in.
    contributor_peak_index : int
        Stage 3 peak index of the thawed line.
    contributor_frequency_mhz : float
        Molecular frequency of the thawed line (MHz).
    edge_side : str
        ``"low"`` or ``"high"`` -- which edge of ``dependent_window_id`` flagged.
    edge_coherence_before : float
        Residual ``S_coh`` on that edge before the thaw, in the same units as
        :func:`~ftmwpipeline.preprocessing.edge_coherence.coherence_statistic`.
    edge_coherence_after : float
        Residual ``S_coh`` on that edge after the joint co-fit. ``NaN`` if the
        co-fit did not converge.
    accepted : bool
        Whether the co-fit converged and lowered the flagged-edge coherence to
        at or below the threshold (i.e. the thaw improved things).
    reason : str
        Free-text note (which heuristic picked the contributor, etc.).
    """

    dependent_window_id: int
    primary_window_id: int
    contributor_peak_index: int
    contributor_frequency_mhz: float
    edge_side: str
    edge_coherence_before: float
    edge_coherence_after: float
    accepted: bool
    reason: str = ""


@dataclass
class RescueEvent:
    """One residual-rescue + joint-refit consolidation round on a window.

    Emitted by the rescue B-loop (:func:`~ftmwpipeline.fitting.residual_rescue.rescue_and_consolidate`)
    when the orchestrator's per-window pass runs with ``max_residual_rescue_rounds > 0``.
    One :class:`RescueEvent` per round actually executed (zero events on
    windows where rescue is disabled or the first round nominated no
    candidates).

    Attributes
    ----------
    window_id : int
        :class:`FitWindow` identifier this round belongs to.
    round_idx : int
        Zero-based round counter within the window's rescue chain.
    n_initial_peaks : int
        Number of peaks the round inherited from the previous round (the
        first round inherits from the initial fit).
    n_candidates : int
        Detector candidates the rescue passed to ``conservative_fit``
        (i.e., what :func:`attempt_residual_rescue` reports as
        ``candidates``).
    n_rescue_added : int
        Peaks the rescue's conservative loop actually accepted (the union
        with the previous round's peaks is what the joint refit fits).
    n_pruned_by_knockout : int
        Peaks the joint-refit's knockout sweep flagged as unsupported
        (dropped before the consolidated fit was finalised).
    n_pruned_rescue_origin : int
        Of the pruned peaks, how many came from *this round's* rescue
        (the failsafe diagnostic -- a high count signals the joint refit
        may not have escaped a pathological basin and is undoing the
        rescue's contribution; v1 logs only).
    n_merged : int
        Close-peak pairs the merge cleanup collapsed before the knockout
        sweep.
    chi2_before, chi2_after : float
        Noise-weighted chi-squared of the previous round's fit and the
        consolidated fit, both evaluated against the same data slice.
        Equal when ``accepted`` is False.
    tau_us_before, tau_us_after : float
        Shared decay constant before and after the round.
    accepted : bool
        Whether the round's contribution replaced the previous round's
        fit (False when the rescue nominated nothing, the joint refit
        failed, or pruning would have emptied the model).
    reason : str
        Free-text annotation -- which termination case fired, how many
        peaks pruned, etc.
    candidates : list of ResidualPeakCandidate
        Detector candidates the rescue passed to ``conservative_fit``.
        Carried by reference so the result-conversion layer can produce
        the persistent
        :class:`~ftmwpipeline.core.data_structures.RescueCandidateInfo`
        records without re-running the rescue.
    """

    window_id: int
    round_idx: int
    n_initial_peaks: int
    n_candidates: int
    n_rescue_added: int
    n_pruned_by_knockout: int
    n_pruned_rescue_origin: int
    n_merged: int
    chi2_before: float
    chi2_after: float
    tau_us_before: float
    tau_us_after: float
    accepted: bool
    reason: str = ""
    # Persistence-ready candidate detail: the candidates the rescue
    # passed to conservative_fit. Carried by reference to the working
    # ResidualPeakCandidate records so the converter can drop them onto
    # the persistent RescueCandidateInfo twins without re-running the
    # rescue. Empty on rounds where attempt_residual_rescue returned no
    # candidates.
    candidates: list = field(default_factory=list)


@dataclass
class ReplanContext:
    """Inputs needed by Stage 5 to ask Stage 4 for a structural re-plan.

    Stage 5's :func:`execute_plan` only operates on the active-FT, but Stage
    4's :func:`~ftmwpipeline.preprocessing.window_planning.replan` needs the
    Stage 3 peak list and the *persisted* Stage 1 spectrum (for the
    leakage-touched region recomputation that drives the bookkeeping tail of
    the plan). This dataclass bundles that context so the production
    orchestrator (``_internal/stage5_impl.py``) can hand one object
    through; tests can pass ``None`` to disable structural renegotiation.

    Stage 4 parameters (``edge_m``, ``edge_threshold``, ``max_window_width_mhz``,
    ``min_freeze_snr``, ``acquisition_us``, ``tau_us``, ``start_us``,
    ``probe_freq_mhz``, ``min_window_half_width_mhz``, ``trim_m``) are read
    from :attr:`WindowPlan.parameters` automatically -- they are persisted as
    part of the plan and need not be threaded again.

    Attributes
    ----------
    peaks : list of Peak
        Stage 3 promoted peak list. Replan keys off ``properties['promoted']``
        and ``classification`` exactly as :func:`build_window_plan` does.
    active_freq_mhz : np.ndarray
        Molecular frequency axis of the active FT (the grid Stage 4 planned
        on -- trimmed to the analysis band).
    active_complex_spectrum : np.ndarray
        Complex active FT on ``active_freq_mhz`` (same array/convention Stage 4
        was built on).
    active_rms_noise : np.ndarray
        Active-FT authority per-bin RMS on ``active_freq_mhz``.
    max_replan_rounds : int, default :data:`DEFAULT_MAX_REPLAN_ROUNDS`
        Cap on the structural-replan outer loop.
    """

    peaks: list[Peak]
    active_freq_mhz: np.ndarray
    active_complex_spectrum: np.ndarray
    active_rms_noise: np.ndarray
    max_replan_rounds: int = DEFAULT_MAX_REPLAN_ROUNDS


@dataclass
class ReplanEvent:
    """One structural-replan record.

    Parallels :class:`ThawEvent` but captures a *plan-structural* change
    rather than a local co-fit. Emitted when the residual edge-coherence
    check flags a window edge that has no fixed contributor to thaw and a
    frequency-adjacent neighbor exists for a :class:`MergeRequest`.

    Attributes
    ----------
    triggering_window_id : int
        Window whose flagged edge prompted the merge. May or may not be the
        ``surviving_window_id`` after the merge (the survivor is always the
        lower-id of the pair).
    partner_window_id : int
        The adjacent window the trigger asked to merge with.
    surviving_window_id : int
        ``min(triggering_window_id, partner_window_id)`` -- the id that
        carries the merged window in the revised plan.
    edge_side : str
        ``"low"`` or ``"high"`` -- which edge of ``triggering_window_id``
        flagged.
    edge_coherence_before : float
        Residual ``S_coh`` on the flagged edge before the merge.
    revision_before : int
        :attr:`WindowPlan.plan_revision` before the merge.
    revision_after : int
        :attr:`WindowPlan.plan_revision` after the merge (= ``before + 1``
        for an accepted merge; equal to ``before`` for a no-op).
    accepted : bool
        Whether the merge was applied and the refit completed (so the
        revised plan stands).
    reason : str
        Free-text annotation.
    """

    triggering_window_id: int
    partner_window_id: int
    surviving_window_id: int
    edge_side: str
    edge_coherence_before: float
    revision_before: int
    revision_after: int
    accepted: bool
    reason: str = ""


@dataclass
class WindowOutcome:
    """Stage 5 result for one window.

    Holds enough state for downstream consumers -- the
    :class:`~ftmwpipeline.core.data_structures.FittingResult` wiring, the
    visualization, and the next batch's fixed-contributor lookup -- to work from
    one object per window. The free-peak fit is left intact in ``fit``; the full
    model / residual (free peaks plus frozen contributors) are reconstructed in
    ``full_fitted_spectrum`` / ``full_residual`` for plotting and the residual
    edge-coherence check.

    Attributes
    ----------
    window_id : int
        :class:`FitWindow` identifier.
    fit : ConservativeFitResult
        The free-peak fit (against ``data - frozen_background``).
    fixed_peaks : list of FrozenPeak
        The frozen contributors used.
    offset_grid_mhz : np.ndarray
        Baseband-offset grid of the window (the fit frame).
    complex_spectrum : np.ndarray
        Active-FT complex window data (the data passed to the fit, *before*
        the frozen-background subtraction).
    rms_noise : np.ndarray
        Per-bin complex noise RMS over the window.
    background : np.ndarray
        Frozen-contributor model on ``offset_grid_mhz``.
    full_fitted_spectrum : np.ndarray
        ``fit.fitted_spectrum + background`` -- the complete model on the grid.
    full_residual : np.ndarray
        ``complex_spectrum - full_fitted_spectrum``.
    thaw_events : list of ThawEvent
        Per-window thaw history (chronological).
    rescue_events : list of RescueEvent
        Per-window residual-rescue history (chronological). Empty unless
        ``max_residual_rescue_rounds > 0`` was set on the executor call.
    edge_coherence_low : float
        Final residual ``S_coh`` at the low-frequency edge.
    edge_coherence_high : float
        Final residual ``S_coh`` at the high-frequency edge.
    """

    window_id: int
    fit: ConservativeFitResult
    fixed_peaks: list[FrozenPeak]
    offset_grid_mhz: np.ndarray
    complex_spectrum: np.ndarray
    rms_noise: np.ndarray
    background: np.ndarray
    full_fitted_spectrum: np.ndarray
    full_residual: np.ndarray
    thaw_events: list[ThawEvent] = field(default_factory=list)
    rescue_events: list[RescueEvent] = field(default_factory=list)
    edge_coherence_low: float = 0.0
    edge_coherence_high: float = 0.0
    # Leakage-wing baseline nuisance term (when fired). ``baseline_applied``
    # records whether the evidence-triggered complex baseline refit replaced
    # this window's fit; ``baseline_order`` / ``baseline_coeffs`` /
    # ``baseline_offset_scale`` carry the fitted term, and
    # ``baseline_edge_coherence`` is the triggering ``max(low, high)`` S_coh
    # measured on the pre-baseline residual. The peak uncertainties on
    # ``fit.fit`` already reflect the joint (peaks + baseline) covariance.
    baseline_applied: bool = False
    baseline_order: Optional[int] = None
    baseline_coeffs: Optional[np.ndarray] = None
    baseline_offset_scale: Optional[float] = None
    baseline_edge_coherence: float = float("nan")
    # Doublet-alternative adjudication records for close pairs in this window.
    # Populated by the optional observation-only pass; empty when the pass is
    # disabled or no qualifying pair was found.
    doublet_adjudications: list = field(default_factory=list)


@dataclass
class PlanFitOutcome:
    """Stage 5 outcome for a whole :class:`WindowPlan`.

    Attributes
    ----------
    window_outcomes : dict of int -> WindowOutcome
        Per-window results, keyed by ``window_id``. After structural
        renegotiation, keys reflect the *final* plan (absorbed window ids
        are absent).
    thaw_history : list of ThawEvent
        Every thaw attempt, in execution order.
    rescue_history : list of RescueEvent
        Every residual-rescue round across all windows, in execution
        order. Empty unless ``max_residual_rescue_rounds > 0`` was set
        on the executor call.
    replan_history : list of ReplanEvent
        Every structural-replan attempt, in execution order. Empty when
        ``execute_plan`` was called without a :class:`ReplanContext`.
    final_plan_revision : int
        :attr:`WindowPlan.plan_revision` of the plan the per-window
        outcomes describe. ``0`` if no replan happened.
    """

    window_outcomes: dict[int, WindowOutcome]
    thaw_history: list[ThawEvent] = field(default_factory=list)
    rescue_history: list[RescueEvent] = field(default_factory=list)
    replan_history: list[ReplanEvent] = field(default_factory=list)
    final_plan_revision: int = 0


# ---------------------------------------------------------------------------
# Fixed-contributor evaluation
# ---------------------------------------------------------------------------
def evaluate_fixed_contributor(
    contributor: FixedContributor,
    primary_outcome: WindowOutcome,
    *,
    dependent_center_mhz: float,
    sideband: SidebandLike,
) -> FrozenPeak:
    """Materialize a :class:`FixedContributor` in a dependent window's fit frame.

    Looks up the contributor's fitted :class:`ModelPeak` in
    ``primary_outcome.fit.peaks`` by the nearest-frequency match (the
    contributor's persisted ``frequency_mhz`` is the Stage 3 detection; the
    primary fit's free peak is its refined location), then remaps that line's
    signed baseband offset from the *primary* window's coordinate to the
    *dependent* window's coordinate. Because both windows share the same
    sideband and probe, the remap is the affine ``delta_dep = s*(f_c_primary -
    f_c_dependent) + delta_primary``; ``amplitude``/``phase``/``tau`` are
    physical and unchanged.

    Both windows are sliced from the same active-FT (already in the
    ``[0, T]`` form), so the relabel is a pure grid shift -- no extra phase
    term enters here.

    Parameters
    ----------
    contributor : FixedContributor
        The plan-level record naming the line.
    primary_outcome : WindowOutcome
        Converged outcome of the contributor's primary window.
    dependent_center_mhz : float
        Reference (molecular) frequency of the dependent window.
    sideband : Sideband or str
        Sideband configuration (must match both windows).

    Returns
    -------
    FrozenPeak
        The frozen model term, in the dependent window's offset frame.

    Raises
    ------
    ValueError
        If the primary outcome has no fitted peaks (the contributor cannot be
        evaluated without a converged primary fit).
    """
    if not primary_outcome.fit.peaks:
        raise ValueError(
            f"primary window {contributor.primary_window_id} has no fitted "
            f"peaks; cannot evaluate fixed contributor "
            f"(peak {contributor.peak_index} at {contributor.frequency_mhz} MHz)"
        )

    s = sideband_sign(sideband)

    # Pick the primary fit's peak nearest the persisted contributor frequency.
    # The primary window's offset is signed-baseband from its own center, so we
    # compare in primary-offset space by mapping the contributor frequency the
    # same way.
    primary_center = _window_center_mhz(primary_outcome)
    contributor_delta_primary = s * (contributor.frequency_mhz - primary_center)
    nearest = min(
        primary_outcome.fit.peaks,
        key=lambda pk: abs(pk.offset_mhz - contributor_delta_primary),
    )

    # Use the *refined* fit frequency, not the Stage 3 detection: the primary's
    # converged offset is the best estimate of where the line actually sits, and
    # the frozen skirt must be evaluated there (a few-kHz Stage 3 vs fit
    # disagreement otherwise places the skirt at the wrong location in the
    # dependent frame).
    fitted_freq_mhz = primary_center + s * nearest.offset_mhz
    delta_dep = s * (fitted_freq_mhz - dependent_center_mhz)

    return FrozenPeak(
        peak_index=contributor.peak_index,
        primary_window_id=contributor.primary_window_id,
        model_peak=ModelPeak(
            amplitude=nearest.amplitude,
            offset_mhz=float(delta_dep),
            phase=nearest.phase,
        ),
        frequency_mhz=fitted_freq_mhz,
        freeze_eligible=contributor.freeze_eligible,
    )


def _window_center_mhz(outcome: WindowOutcome) -> float:
    """Recover the molecular reference frequency from a window outcome.

    A window's offset grid is ``u = s*(f - f_c)``; one valid recovery is
    ``f_c = f_first - s*u_first`` once you know one paired (``f``, ``u``)
    point. We store ``f_c`` alongside the outcome implicitly through the
    ``offset_grid_mhz`` and the molecular grid -- but we did not persist the
    molecular grid on the outcome. The dependable invariant is that the
    outcome's offset grid was built with this center: callers that need the
    center must supply it (the plan executor does, see :func:`execute_plan`).
    Here we lean on a small helper: the center is cached as a private attribute
    on the outcome when one is produced by the plan executor.
    """
    center = getattr(outcome, "_center_mhz", None)
    if center is None:
        raise ValueError(
            "WindowOutcome is missing its molecular reference frequency. "
            "WindowOutcome instances produced outside execute_plan must set the "
            "_center_mhz attribute before being used as a fixed-contributor "
            "primary."
        )
    return float(center)


def evaluate_edge_free_contributors(
    contributors: Sequence[FixedContributor],
    active_freq_mhz: np.ndarray,
    active_complex_spectrum: np.ndarray,
    *,
    dependent_center_mhz: float,
    sideband: SidebandLike,
    tau_us: float,
    acquisition_us: float,
    shape: "PeakShape | str" = "lorentzian",
    read_half_width_mhz: Optional[float] = None,
) -> list[FrozenPeak]:
    """Materialize edge-free contributors via a self-contained active-FT read.

    Edge-free contributors carry no fit-ordering edge, so their frozen
    parameters cannot be read from a primary window's converged fit. Instead a
    **joint complex least-squares** of the finite-T line template against the
    strong lines' core bins on the active FT recovers each line's
    ``(amplitude, phase)``. Solving the co-located lines together makes the read
    robust to the leakage pedestal -- the global *single-bin phasor* read was
    NEGATIVE on the dense 655 spectrum (each core bin carries ~300 other lines'
    summed skirts; see ``dev-docs/research/stage5-cross-fixture/report.md``
    §1).

    The read uses the dependent window's ``tau_us`` -- the same decay the frozen
    skirt is later drawn with by :func:`subtract_frozen_background` -- so the
    recovered amplitude and the subtracted skirt stay self-consistent (reading
    at a different tau than the skirt is drawn at biases the amplitude; verified
    on the 360 w287 A/B).

    Contributors are grouped by ``primary_window_id`` so each genuinely-adjacent
    cluster (e.g. the 360 w288 triplet) is solved as one joint system. The
    returned :class:`FrozenPeak` s are in the *dependent* window's offset frame
    (``δ = s·(f_line - f_c_dependent)``); ``amplitude``/``phase`` are physical
    and frame-independent.

    Parameters
    ----------
    contributors : sequence of FixedContributor
        The edge-free contributors of one dependent window. Non-edge-free
        entries are ignored.
    active_freq_mhz, active_complex_spectrum : np.ndarray
        The full active-FT molecular grid and complex spectrum (the lines are
        read from their own core bins, which lie outside the dependent window).
    dependent_center_mhz : float
        Reference (molecular) frequency of the dependent window.
    sideband : Sideband or str
        Sideband configuration.
    tau_us : float
        Decay constant for the read template -- the dependent window's shared
        tau (the same value the frozen skirt is drawn with).
    acquisition_us : float
        Active acquisition length ``T`` (µs).
    shape : PeakShape or str
        Line shape for the template (must match the fit's shape).
    read_half_width_mhz : float, optional
        Half-width of the per-line core read region (MHz). ``None`` (default)
        uses ``8 / T`` (eight resolution elements), the 360 w287 A/B sweet
        spot.
    """
    s = sideband_sign(sideband)
    freq = np.asarray(active_freq_mhz, dtype=float)
    z = np.asarray(active_complex_spectrum, dtype=np.complex128)
    if read_half_width_mhz is None:
        read_half_width_mhz = 8.0 / acquisition_us

    by_primary: dict[int, list[FixedContributor]] = {}
    for c in contributors:
        if not c.edge_free:
            continue
        by_primary.setdefault(c.primary_window_id, []).append(c)

    frozen: list[FrozenPeak] = []
    for primary_id, group in by_primary.items():
        line_freqs = [c.frequency_mhz for c in group]
        # Union of each line's core bins.
        mask = np.zeros(freq.shape, dtype=bool)
        for f0 in line_freqs:
            mask |= np.abs(freq - f0) <= read_half_width_mhz
        n_core = int(np.count_nonzero(mask))
        if n_core < len(line_freqs):
            # Too few bins to solve (line(s) off the grid edge): skip the group
            # -- no skirt is better than a wild read.
            continue
        f_core = freq[mask]
        z_core = z[mask]

        def _design(freqs_at: Sequence[float]) -> np.ndarray:
            d = np.empty((f_core.size, len(freqs_at)), dtype=np.complex128)
            for j, f0 in enumerate(freqs_at):
                u_local = s * (f_core - f0)
                d[:, j] = model_spectrum(
                    u_local,
                    [ModelPeak(1.0, 0.0, 0.0)],
                    tau_us,
                    acquisition_us,
                    shape=shape,
                )
            return cast(np.ndarray, d)

        if DEFAULT_EDGE_FREE_FREQ_REFINE and f_core.size > 2 * len(line_freqs):
            # Variable-projection frequency refinement: the detected positions
            # carry sub-bin errors that mis-phase the sharp core template and
            # bias the linear amplitude read low (see
            # :data:`DEFAULT_EDGE_FREE_FREQ_REFINE`). Frequencies move within
            # a sub-bin-scale bound (never far enough to swap identities
            # within the group); amplitudes/phases stay linear per trial.
            step = float(np.median(np.abs(np.diff(np.sort(f_core)))))
            bound = 1.5 * step
            if len(line_freqs) > 1:
                seps = np.diff(np.sort(np.asarray(line_freqs)))
                min_sep = float(seps.min()) if seps.size else np.inf
                bound = min(bound, 0.45 * min_sep) if np.isfinite(min_sep) else bound
            if bound > 0.0:

                def _vp_residual(deltas: np.ndarray) -> np.ndarray:
                    d = _design([f0 + dd for f0, dd in zip(line_freqs, deltas)])
                    g_trial, *_ = np.linalg.lstsq(d, z_core, rcond=None)
                    r = np.asarray(z_core - d @ g_trial)
                    return cast(np.ndarray, np.concatenate([r.real, r.imag]))

                try:
                    from scipy.optimize import least_squares

                    sol = least_squares(
                        _vp_residual,
                        np.zeros(len(line_freqs)),
                        bounds=(-bound, bound),
                        method="trf",
                        max_nfev=60,
                    )
                    line_freqs = [f0 + dd for f0, dd in zip(line_freqs, sol.x)]
                except Exception:
                    pass  # keep the detected positions; the linear read stands

        design = _design(line_freqs)
        coeffs, *_ = np.linalg.lstsq(design, z_core, rcond=None)
        for c, f0, g in zip(group, line_freqs, coeffs):
            delta_dep = s * (f0 - dependent_center_mhz)
            frozen.append(
                FrozenPeak(
                    peak_index=c.peak_index,
                    primary_window_id=primary_id,
                    model_peak=ModelPeak(
                        amplitude=float(np.abs(g)),
                        offset_mhz=float(delta_dep),
                        phase=float(np.angle(g)),
                    ),
                    frequency_mhz=float(f0),
                    freeze_eligible=c.freeze_eligible,
                    edge_free=True,
                )
            )
    return frozen


def _noise_weighted_ssr(residual: np.ndarray, rms_noise: NoiseLike) -> float:
    """Noise-weighted residual sum of squares ``Σ |z|² / σ²`` (complex σ_x).

    The shared misfit scalar for the edge-free accept/reject A/B: lower is a
    better fit. A scalar or per-bin ``rms_noise`` is broadcast; zero/negative
    σ bins are dropped from the sum.
    """
    z = np.asarray(residual, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(z.shape, float(sigma))
    good = sigma > 0.0
    if not np.any(good):
        return float("inf")
    return float(np.sum((np.abs(z[good]) ** 2) / (sigma[good] ** 2)))


def subtract_frozen_background(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    fixed_peaks: Sequence[FrozenPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = "lorentzian",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the frozen-contributor background and subtract it from the data.

    Returns ``(background, data_minus_background)`` -- both complex, on the same
    grid. With no contributors the background is all zeros and the data is
    returned unchanged. The shared ``tau`` is the dependent window's, not the
    primary's: ``h_T`` carries one ``tau`` per window.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid of the dependent window (MHz).
    complex_spectrum : np.ndarray
        Active-FT complex window data, same shape as the grid.
    fixed_peaks : sequence of FrozenPeak
        Contributors to evaluate.
    tau_us : float
        Dependent window's shared decay constant (microseconds).
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    """
    if not fixed_peaks:
        zero = np.zeros(offset_grid_mhz.shape, dtype=np.complex128)
        return zero, np.asarray(complex_spectrum, dtype=np.complex128)
    bg = model_spectrum(
        offset_grid_mhz,
        [fp.model_peak for fp in fixed_peaks],
        tau_us,
        acquisition_us,
        shape=shape,
    )
    return bg, np.asarray(complex_spectrum, dtype=np.complex128) - bg


def fit_window_with_fixed_contributors(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    fixed_peaks: Sequence[FrozenPeak],
    candidate_offsets: Sequence[float],
    tau0_us: float,
    acquisition_us: float,
    early_baseline_order: Optional[int] = None,
    early_baseline_smooth_threshold: Optional[float] = None,
    **conservative_kwargs: Any,
) -> tuple[ConservativeFitResult, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative free-peak fit of a window with frozen contributors.

    Subtracts the frozen-contributor background from the window data and runs
    :func:`~ftmwpipeline.fitting.window_fit.conservative_fit` on the difference.
    The returned ``full_fitted_spectrum`` is the free-peak model plus the
    background, and ``full_residual`` is the data minus that full model -- the
    correct things to plot and to test for residual edge coherence.

    ``**conservative_kwargs`` are forwarded verbatim to :func:`conservative_fit`
    (e.g. ``fit_tau``, ``significance``, ``min_separation_factor``, ...).

    Returns
    -------
    tuple
        ``(fit_result, background, full_fitted_spectrum, full_residual)``.
    """
    shape = conservative_kwargs.get("shape", "lorentzian")
    background, data_minus_bg = subtract_frozen_background(
        offset_grid_mhz,
        complex_spectrum,
        fixed_peaks,
        tau0_us,
        acquisition_us,
        shape=shape,
    )
    # The sigma_eff skirt budget (``kappa_skirt * |background|``) deliberately
    # does NOT gate this conservative fit. The budget exists to discount the
    # subtracted frozen background's extrapolation error -- coherent fringes
    # the *residual rescue* harvests as spurious lines (it injects the budget
    # itself in ``_apply_rescue_to_outcome``). The conservative loop's
    # candidates are the window's Stage 3 *promoted detections*: the matched
    # filter's leakage-aware floor already vouches they are lines, and on a
    # deep-skirt window the budget (proportional to the dominant |background|)
    # otherwise swallows their entire evidence -- a raw delta chi-squared of
    # ~1e5 reads as ~0 in budgeted currency and every real riding line dies
    # (the same real-line-killing failure the budget-currency seed-knockout
    # measured; raw currency is correct for Stage-3-backed candidates).
    #
    # Early leakage-wing baseline, re-run-on-trigger: when the fit's residual
    # still carries a smooth pedestal an order-p complex polynomial explains
    # (the summed far-wings of many lines the discrete contributors cannot
    # fully subtract), the whole conservative fit is re-run with the baseline
    # as a joint nuisance term. A peaks-only fit on such a window buys
    # chi-squared by collapsing the shared tau onto the pedestal, and every
    # downstream decision then runs on a broken model: separations measured
    # in the ballooned FWHM block real candidates, the rescue's joint refit
    # slides peaks onto the window bounds, and the sequenced-last baseline
    # refit can only relax tau for whatever peak set survived. The trigger is
    # the same smooth-residual F-statistic the late baseline uses, evaluated
    # on the POST-fit residual: a null-residual (pre-fit) trigger cannot tell
    # a pedestal from a bright line's own profile and mis-fires on every
    # strong-line window (soaking line wings, biasing tau), while the
    # post-fit residual of a healthy bright-line fit carries only sharp core
    # structure the polynomial does not explain. Triggered windows pay one
    # extra conservative pass; healthy windows pay nothing.
    if os.environ.get("FTMW_DEBUG_FRINGE_DIR"):
        ctx = dict(validation._fringe_window_ctx or {})
        ctx["bg_u"] = np.asarray(offset_grid_mhz, dtype=float)
        ctx["bg"] = np.asarray(background, dtype=np.complex128)
        validation._fringe_window_ctx = ctx
    fit_result = conservative_fit(
        offset_grid_mhz,
        data_minus_bg,
        rms_noise,
        candidate_offsets,
        tau0_us,
        acquisition_us,
        gate_background=background,
        **conservative_kwargs,
    )
    if (
        early_baseline_order is not None
        and early_baseline_smooth_threshold is not None
        and fit_result.fit.baseline_order is None
    ):
        u_arr = np.asarray(offset_grid_mhz, dtype=float)
        first_residual = np.asarray(
            data_minus_bg, dtype=np.complex128
        ) - model_spectrum(
            u_arr,
            fit_result.fit.peaks,
            fit_result.fit.tau_us,
            acquisition_us,
            shape=fit_result.fit.shape,
        )
        smooth_stat = _smooth_residual_stat(
            u_arr,
            first_residual,
            np.asarray(rms_noise, dtype=float),
            int(early_baseline_order),
        )
        if smooth_stat > float(early_baseline_smooth_threshold):
            retry_kwargs = dict(conservative_kwargs)
            retry_kwargs["baseline_order"] = int(early_baseline_order)
            retry = conservative_fit(
                offset_grid_mhz,
                data_minus_bg,
                rms_noise,
                candidate_offsets,
                tau0_us,
                acquisition_us,
                gate_background=background,
                **retry_kwargs,
            )
            # Adopt only on a strict raw-chi-squared win: the re-run spends
            # 2(order+1) extra parameters, so a tie means the pedestal read
            # was spurious and the peaks-only fit stands.
            if retry.fit.success and retry.fit.chi_squared < fit_result.fit.chi_squared:
                fit_result = retry
    # conservative_fit sorts its grid internally; its fitted_spectrum is on
    # *that* sorted grid. Re-evaluate the model on the caller's input grid so
    # the returned arrays line up bin-for-bin with the inputs. The fit's
    # baseline term (when one was fit jointly) is part of the model: leaving
    # it out would re-expose the carried pedestal in ``full_residual`` and
    # spuriously fire the edge-coherence / rescue triggers downstream.
    full_free = model_spectrum(
        offset_grid_mhz,
        fit_result.fit.peaks,
        fit_result.fit.tau_us,
        acquisition_us,
        shape=fit_result.fit.shape,
    ) + evaluate_baseline(fit_result.fit, offset_grid_mhz)
    full_fitted = full_free + background
    full_residual = np.asarray(complex_spectrum, dtype=np.complex128) - full_fitted
    return fit_result, background, full_fitted, full_residual


# ---------------------------------------------------------------------------
# Residual edge-coherence
# ---------------------------------------------------------------------------
def residual_edge_coherence(
    residual: np.ndarray,
    rms_noise: NoiseLike,
    band_m: int = DEFAULT_RESIDUAL_EDGE_M,
) -> tuple[float, float]:
    """Complex-edge coherence ``S_coh`` of the residual on each window edge.

    The window's residual ``z`` is sliced into its first and last ``band_m``
    bins; :func:`~ftmwpipeline.preprocessing.edge_coherence.coherence_statistic`
    is evaluated on each slice with the local mean of ``rms_noise``. A coherent
    leakage tail on either edge shows up as an above-threshold value; a clean
    residual gives values near the null mean ``sqrt(pi/4) ~= 0.886``.

    The returned ``(low, high)`` ordering follows the input order, i.e. ``low``
    is the first-band statistic regardless of whether the grid is ascending or
    descending in molecular frequency.

    Parameters
    ----------
    residual : np.ndarray
        Complex residual on the window grid.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS (scalar broadcast or per-bin array).
    band_m : int, default :data:`DEFAULT_RESIDUAL_EDGE_M`
        Number of edge bins per band. Clamped down to the residual length when
        the window is shorter than ``2 * band_m``.

    Returns
    -------
    tuple of float
        ``(low_edge_S_coh, high_edge_S_coh)``.
    """
    z = np.asarray(residual, dtype=np.complex128)
    n = z.size
    if n == 0:
        return 0.0, 0.0
    m = max(1, min(int(band_m), n // 2 if n >= 2 else n))

    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(n, float(sigma))
    elif sigma.shape != z.shape:
        raise ValueError("rms_noise must be a scalar or match the residual shape")

    low_sigma = float(np.mean(sigma[:m]))
    high_sigma = float(np.mean(sigma[-m:]))
    low = coherence_statistic(z[:m], low_sigma)
    high = coherence_statistic(z[-m:], high_sigma)
    return low, high


# ---------------------------------------------------------------------------
# Shared window-outcome construction core
# ---------------------------------------------------------------------------
def build_window_outcome(
    fit_result: ConservativeFitResult,
    window_id: int,
    fixed_peaks: list[FrozenPeak],
    offset_grid: np.ndarray,
    z_slice: np.ndarray,
    sig_slice: np.ndarray,
    background: np.ndarray,
    full_fitted: np.ndarray,
    full_residual: np.ndarray,
    residual_edge_m: int,
    center_mhz: float,
    spur_mask: "Optional[Any]",
) -> WindowOutcome:
    """Construct a :class:`WindowOutcome` from already-computed fit pieces.

    This is the shared tail of :func:`_fit_one_window` and
    :func:`fit_seeds_window_outcome`: given the free-peak fit result, the
    frozen background, and the full model / residual on the window grid, it
    computes the residual edge-coherence statistics, assembles the
    :class:`WindowOutcome`, and stashes the window center and spur mask as
    dynamic attributes so downstream consumers can locate the window without
    re-materialising it.

    The caller is responsible for any post-construction extras (e.g.
    mirroring the early-baseline fields in :func:`_fit_one_window`, or
    converting the outcome to a :class:`FittingResult` in
    :func:`fit_seeds_window_outcome`).

    Parameters
    ----------
    fit_result :
        The free-peak fit against ``z_slice - background``, wrapped as a
        :class:`ConservativeFitResult`.
    window_id :
        :class:`FitWindow` identifier.
    fixed_peaks :
        Frozen contributors used in this window.
    offset_grid :
        Baseband-offset grid (MHz) of the window.
    z_slice :
        Complex active-FT data over the window (before background subtraction).
    sig_slice :
        Per-bin complex noise RMS over the window.
    background :
        Frozen-contributor model on ``offset_grid``.
    full_fitted :
        ``free_model + background`` — the complete predicted spectrum.
    full_residual :
        ``z_slice - full_fitted``.
    residual_edge_m :
        Band width (bins) for the residual edge-coherence test.
    center_mhz :
        Molecular center of the window (MHz); stashed as ``outcome._center_mhz``.
    spur_mask :
        Per-window spur mask (or ``None``); stashed as ``outcome._spur_mask``.

    Returns
    -------
    WindowOutcome
        Fully populated outcome with ``_center_mhz`` and ``_spur_mask``
        dynamic attributes set.
    """
    low_coh, high_coh = residual_edge_coherence(
        full_residual, sig_slice, band_m=residual_edge_m
    )
    outcome = WindowOutcome(
        window_id=window_id,
        fit=fit_result,
        fixed_peaks=fixed_peaks,
        offset_grid_mhz=offset_grid,
        complex_spectrum=z_slice,
        rms_noise=sig_slice,
        background=background,
        full_fitted_spectrum=full_fitted,
        full_residual=full_residual,
        edge_coherence_low=low_coh,
        edge_coherence_high=high_coh,
    )
    # Stash the center so this outcome can serve as a primary for downstream
    # windows. See _window_center_mhz for the contract.
    outcome._center_mhz = center_mhz  # type: ignore[attr-defined]
    # Stash the per-window spur mask so the rescue pass (and the refit) mask
    # the same bins.
    outcome._spur_mask = spur_mask  # type: ignore[attr-defined]
    return outcome


def fit_seeds_window_outcome(
    offset_grid: np.ndarray,
    z_slice: np.ndarray,
    sig_slice: np.ndarray,
    center_mhz: float,
    background: np.ndarray,
    data_minus_bg: np.ndarray,
    fixed_peaks: list[FrozenPeak],
    seed_peaks: list[ModelPeak],
    tau0_us: float,
    acquisition_us: float,
    fw_kwargs: dict[str, Any],
    spur_mask: "Optional[Any]",
    n_eff_kind: str,
    residual_edge_m: int,
    window_id: int,
) -> WindowOutcome:
    """Bare given-set refit core: one ``fit_window`` + ``knockout_test`` → ``WindowOutcome``.

    Runs a single joint NLS over ``seed_peaks`` (no conservative add-one-peak
    search, no rescue, no thaw), wraps the result in a
    :class:`ConservativeFitResult` (empty ``audit_trail``), computes the full
    model and residual, and delegates to :func:`build_window_outcome`.

    This captures exactly what :func:`~ftmwpipeline._internal.stage6_impl.refit_window_impl`'s
    inner fit body does (lines that mirror :func:`_fit_one_window`).  Both the
    Stage 6 user-directed refit and the forthcoming survival-pass refits route
    through this core; the conservative add-one-peak *search* in
    :func:`_fit_one_window` stays as orchestration above it.

    Parameters
    ----------
    offset_grid :
        Baseband-offset grid (MHz) of the window.
    z_slice :
        Complex active-FT data over the window (before background subtraction).
    sig_slice :
        Per-bin complex noise RMS over the window.
    center_mhz :
        Molecular center of the window (MHz).
    background :
        Frozen-contributor model on ``offset_grid``.
    data_minus_bg :
        ``z_slice - background``; pre-computed by the caller so the frozen
        background derivation is not repeated.
    fixed_peaks :
        Frozen contributors used in this window (carried verbatim onto the
        :class:`WindowOutcome`).
    seed_peaks :
        Initial :class:`ModelPeak` objects for the NLS.
    tau0_us :
        Starting decay constant (µs) for the fit.
    acquisition_us :
        Active-FID length (µs).
    fw_kwargs :
        Keyword arguments forwarded to :func:`~ftmwpipeline.fitting.window_fit.fit_window`
        (must include ``fit_tau``, ``shape``, penalty lambdas, etc.).
        If ``"spur_mask"`` is present it is stripped for the inner
        :func:`~ftmwpipeline.fitting.window_fit.knockout_test` refit.
    spur_mask :
        Per-window spur mask forwarded to ``knockout_test`` directly.
    n_eff_kind :
        Effective-count kind string for ``knockout_test``.
    residual_edge_m :
        Band width (bins) for the residual edge-coherence test.
    window_id :
        :class:`FitWindow` identifier.

    Returns
    -------
    WindowOutcome
        Outcome with ``_center_mhz`` and ``_spur_mask`` stashed; caller
        converts it to a :class:`FittingResult` as needed.
    """
    from .peak_model import model_spectrum
    from .window_fit import evaluate_baseline, knockout_test

    fit_result: WindowFitResult = fit_window(
        offset_grid,
        data_minus_bg,
        sig_slice,
        initial_peaks=seed_peaks,
        tau0_us=tau0_us,
        acquisition_us=acquisition_us,
        **fw_kwargs,
    )

    # ``knockout_test`` passes tau0_us / acquisition_us positionally and
    # spur_mask explicitly; strip spur_mask from the inner kwargs so it does
    # not collide on the multi-peak refit path.
    knockout_inner = {k: v for k, v in fw_kwargs.items() if k != "spur_mask"}
    knockouts = knockout_test(
        offset_grid,
        data_minus_bg,
        sig_slice,
        fit_result,
        acquisition_us,
        fit_kwargs_inner=knockout_inner,
        spur_mask=spur_mask,
        n_eff_kind=n_eff_kind,
    )

    # Wrap in ConservativeFitResult (empty audit_trail -- this is a joint
    # refit, not a conservative loop; audit provenance lives in the caller).
    conservative_result = ConservativeFitResult(
        fit=fit_result,
        audit_trail=[],
        knockouts=knockouts,
    )

    free_model = model_spectrum(
        offset_grid,
        conservative_result.fit.peaks,
        conservative_result.fit.tau_us,
        acquisition_us,
        shape=conservative_result.fit.shape,
    ) + evaluate_baseline(conservative_result.fit, offset_grid)
    full_fitted = free_model + background
    full_residual = z_slice - full_fitted

    return build_window_outcome(
        conservative_result,
        window_id,
        fixed_peaks,
        offset_grid,
        z_slice,
        sig_slice,
        background,
        full_fitted,
        full_residual,
        residual_edge_m,
        center_mhz,
        spur_mask,
    )


# ---------------------------------------------------------------------------
# Thaw selection + local co-fit
# ---------------------------------------------------------------------------
def select_contributor_to_thaw(
    window: FitWindow,
    fixed_peaks: Sequence[FrozenPeak],
    edge_side: str,
) -> Optional[FrozenPeak]:
    """Pick the contributor most likely responsible for a flagged residual edge.

    The selection rules, in order:

    1. Restrict to contributors on the same side of the window as the flagged
       edge. The molecular-frequency edge is ``window.freq_range[0]`` for the
       low side and ``window.freq_range[1]`` for the high side; a contributor
       is "on" that side if its frequency lies outside the window in that
       direction (the typical case for a fixed contributor -- they live in
       neighboring windows).
    2. Prefer ``freeze_eligible=False`` contributors -- Stage 4 already flagged
       these as not safe to freeze (O4-2).
    3. Among the remaining candidates, pick the one nearest the flagged edge
       in molecular frequency (the closest skirt is the most likely culprit).

    Returns ``None`` if no contributor matches the side, in which case the
    plan executor records a no-op thaw event and stops trying.

    Parameters
    ----------
    window : FitWindow
        The dependent window with the flagged edge.
    fixed_peaks : sequence of FrozenPeak
        Currently-frozen contributors in this window.
    edge_side : str
        ``"low"`` or ``"high"`` -- the flagged residual edge.
    """
    if edge_side not in ("low", "high"):
        raise ValueError("edge_side must be 'low' or 'high'")

    # Edge-free contributors carry no fit-ordering edge and no primary-fit
    # dependency, so they cannot be thawed (a thaw co-fits the dependent with
    # the contributor's converged primary, which an edge-free contributor does
    # not have). Exclude them from the thaw pool; their mismodeled skirt, if
    # any, is left to the leakage-wing baseline.
    thawable = [fp for fp in fixed_peaks if not fp.edge_free]
    lo, hi = window.freq_range
    if edge_side == "low":
        side_candidates = [fp for fp in thawable if fp.frequency_mhz <= lo]
        edge_freq = lo
    else:
        side_candidates = [fp for fp in thawable if fp.frequency_mhz >= hi]
        edge_freq = hi

    if not side_candidates:
        return None

    not_eligible = [fp for fp in side_candidates if not fp.freeze_eligible]
    pool = not_eligible if not_eligible else side_candidates
    return min(pool, key=lambda fp: abs(fp.frequency_mhz - edge_freq))


def local_thaw_cofit(
    dependent: WindowOutcome,
    primary: WindowOutcome,
    *,
    thawed: FrozenPeak,
    sideband: SidebandLike,
    tau0_us: float,
    acquisition_us: float,
    fit_tau: bool = True,
    max_decay_factor: float = DEFAULT_MAX_DECAY_FACTOR,
    shape: "PeakShape | str" = "lorentzian",
    spur_set: Optional[SpurSet] = None,
) -> tuple[WindowFitResult, np.ndarray]:
    """Joint co-fit of two windows with one contributor unfrozen.

    The two windows' active-FT data are concatenated under a shared baseband
    coordinate centered at the primary window's reference frequency: the
    primary's free peaks keep their offsets unchanged; the dependent window's
    grid and free-peak offsets are remapped into the shared frame by adding
    ``shift = s*(dep_center - primary_center)``. The thawed contributor was
    already a free peak in the primary's fit (that is what "thaw" means: a line
    fit freely in its primary is also constrained by the dependent's data),
    so it is **not** added as a separate peak -- doing so would double-count
    the line. Other frozen contributors of either window stay frozen as a
    background subtraction on their respective slices.

    The joint fit's peak list layout is

        joint.peaks[:n_primary] + joint.peaks[n_primary:]
        = (primary free peaks, refined) + (dependent free peaks in primary frame)

    and the returned index points to the thawed peak's slot within
    ``joint.peaks[:n_primary]`` (the primary peak nearest the contributor's
    persisted frequency).

    Returns
    -------
    tuple
        ``(joint_fit, thawed_peak_indices)``. ``thawed_peak_indices`` is a
        length-1 array selecting the row of ``joint.peaks`` that is the
        thawed contributor.
    """
    s = sideband_sign(sideband)
    primary_center = _window_center_mhz(primary)
    dep_center = _window_center_mhz(dependent)

    # Build the combined data / grid in the primary's offset frame.
    primary_u = np.asarray(primary.offset_grid_mhz, dtype=float)
    primary_data = np.asarray(primary.complex_spectrum, dtype=np.complex128)
    dep_u = np.asarray(dependent.offset_grid_mhz, dtype=float)
    dep_data = np.asarray(dependent.complex_spectrum, dtype=np.complex128)

    # u_primary = s*(f - f_c_primary); u_dep_in_primary = s*(f - f_c_primary)
    # = u_dep + s*(f_c_dep - f_c_primary). So shift the dependent grid by that
    # constant offset to remap it into the primary's frame.
    shift = s * (dep_center - primary_center)
    dep_u_in_primary = dep_u + shift

    # Subtract any *other* frozen contributors from each side as background, so
    # the joint fit only handles free peaks plus the thawed line.
    primary_other = [fp for fp in primary.fixed_peaks if fp is not thawed]
    dep_other = [
        fp
        for fp in dependent.fixed_peaks
        if fp.peak_index != thawed.peak_index
        or fp.primary_window_id != thawed.primary_window_id
    ]
    _, primary_clean = subtract_frozen_background(
        primary_u,
        primary_data,
        primary_other,
        tau0_us,
        acquisition_us,
        shape=shape,
    )
    _, dep_clean = subtract_frozen_background(
        dep_u_in_primary,
        dep_data,
        dep_other,
        tau0_us,
        acquisition_us,
        shape=shape,
    )

    grid = np.concatenate([primary_u, dep_u_in_primary])
    data = np.concatenate([primary_clean, dep_clean])

    sigma_primary = np.asarray(primary.rms_noise, dtype=float)
    if sigma_primary.ndim == 0:
        sigma_primary = np.full(primary_u.size, float(sigma_primary))
    sigma_dep = np.asarray(dependent.rms_noise, dtype=float)
    if sigma_dep.ndim == 0:
        sigma_dep = np.full(dep_u.size, float(sigma_dep))
    sigma = np.concatenate([sigma_primary, sigma_dep])

    # Initial peak list. The thawed contributor is *already* a free peak in
    # the primary's fit -- "thaw" means promoting that primary free peak so the
    # dependent's data also constrains it. We must NOT add it as a separate
    # peak (that would double-count the line). So: primary's free peaks
    # (unchanged) + dependent's free peaks (remapped into the primary frame).
    primary_peaks = [
        ModelPeak(pk.amplitude, pk.offset_mhz, pk.phase) for pk in primary.fit.peaks
    ]
    dep_peaks_in_primary = [
        ModelPeak(pk.amplitude, pk.offset_mhz + shift, pk.phase)
        for pk in dependent.fit.peaks
    ]
    init = primary_peaks + dep_peaks_in_primary

    # Identify the thawed peak's index in the primary list -- the closest match
    # to the contributor's molecular frequency, mapped into the primary frame.
    contributor_offset_primary = s * (thawed.frequency_mhz - primary_center)
    thawed_index = min(
        range(len(primary_peaks)),
        key=lambda i: abs(primary_peaks[i].offset_mhz - contributor_offset_primary),
    )

    # Widen the offset bounds to the combined span.
    lo = float(grid.min())
    hi = float(grid.max())

    # Spur mask in the joint (primary) frame: any gated spur whose offset
    # s*(f - primary_center) lands inside the combined grid span.
    joint_spur_mask: Optional[SpurMaskSpec] = None
    if spur_set is not None and spur_set:
        kept: List[Tuple[float, GatedSpur]] = []
        for sp in spur_set.spurs:
            off = float(s * (sp.center_mhz - primary_center))
            if lo <= off <= hi:
                kept.append((off, sp))
        if kept:
            half_widths: Optional[Tuple[float, ...]] = None
            if any(sp.mask_half_width_bins is not None for _, sp in kept):
                half_widths = tuple(spur_set._spur_half_width_mhz(sp) for _, sp in kept)
            joint_spur_mask = SpurMaskSpec(
                offsets_mhz=tuple(off for off, _ in kept),
                half_width_mhz=spur_set.mask_half_width_mhz,
                half_widths_mhz=half_widths,
            )

    joint = fit_window(
        grid,
        data,
        sigma,
        init,
        tau0_us,
        acquisition_us,
        fit_tau=fit_tau,
        max_decay_factor=max_decay_factor,
        offset_bounds=(lo, hi),
        shape=shape,
        spur_mask=joint_spur_mask,
    )
    return joint, np.array([thawed_index], dtype=int)


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------
def materialize_window(
    fit_window_spec: FitWindow,
    active_ft: ActiveFTResult,
    rms_noise: np.ndarray,
    *,
    sideband: SidebandLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Slice a window from the active-FT and move it into the fit frame.

    The active-FT is already in the ``[0, T]`` form ``h_T`` models, so this
    is just a band slice plus a molecular -> signed-baseband-offset grid
    relabel -- no de-ramp.

    Returns ``(freq_slice, offset_grid, complex_slice, rms_slice, center_mhz)``.

    The molecular frequency reference is the midpoint of
    ``fit_window_spec.freq_range`` -- the natural symmetric choice, also the
    one cached on each :class:`WindowOutcome` as ``_center_mhz``. No
    baseline-context margin is applied here; the slice is the bare Stage 4
    freq_range.
    """
    lo, hi = fit_window_spec.freq_range
    if lo > hi:
        lo, hi = hi, lo
    freq_array_mhz = active_ft.freq_mhz
    complex_spectrum = active_ft.complex_spectrum
    mask = (freq_array_mhz >= lo) & (freq_array_mhz <= hi)
    if not np.any(mask):
        raise ValueError(
            f"window {fit_window_spec.window_id} freq_range "
            f"({lo}, {hi}) MHz has no overlap with the active-FT spectrum"
        )
    freq_slice = freq_array_mhz[mask]
    z_slice = complex_spectrum[mask]
    sig_slice = rms_noise[mask]
    # Reference frequency: the midpoint of the window's freq_range. This is the
    # natural symmetric choice and keeps free-peak offsets balanced around zero.
    center_mhz = 0.5 * (lo + hi)
    offset_grid, z_offset = to_baseband_offset(
        freq_slice,
        z_slice,
        center_mhz=center_mhz,
        sideband=sideband,
    )
    return freq_slice, offset_grid, z_offset, sig_slice, center_mhz


def _peaks_to_candidate_offsets(
    fit_window_spec: FitWindow,
    peak_frequencies_mhz: Sequence[float],
    center_mhz: float,
    sideband: SidebandLike,
) -> list[float]:
    """Free-peak frequencies (MHz) -> signed baseband offsets in this window."""
    s = sideband_sign(sideband)
    return [
        float(s * (float(peak_frequencies_mhz[idx]) - center_mhz))
        for idx in fit_window_spec.free_peak_indices
    ]


def _debug_phase(wid: int, phase: str, outcome: "WindowOutcome") -> None:
    """``FTMW_DEBUG_PHASES=1``: per-window K / chi2 at each phase boundary."""
    if not os.environ.get("FTMW_DEBUG_PHASES"):
        return
    inner = outcome.fit.fit
    print(
        f"[phase] w{wid} {phase}: K={inner.n_peaks} chi2={inner.chi_squared:.0f} "
        f"tau={inner.tau_us:.2f} "
        f"offsets={[round(p.offset_mhz, 3) for p in inner.peaks]}",
        flush=True,
    )


def execute_plan(
    plan: WindowPlan,
    active_ft: ActiveFTResult,
    rms_noise: np.ndarray,
    peak_frequencies_mhz: Sequence[float],
    *,
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool = True,
    residual_edge_threshold: float = DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    residual_edge_m: int = DEFAULT_RESIDUAL_EDGE_M,
    max_thaw_rounds: int = DEFAULT_MAX_THAW_ROUNDS,
    conservative_kwargs: Optional[dict[str, Any]] = None,
    replan_context: Optional[ReplanContext] = None,
    max_residual_rescue_rounds: int = 0,
    rescue_kwargs: Optional[dict[str, Any]] = None,
    window_tau_overrides: Optional[dict[int, tuple[float, float]]] = None,
    shape: "PeakShape | str" = "lorentzian",
    spur_set: Optional[SpurSet] = None,
    baseline_enabled: bool = DEFAULT_BASELINE_ENABLED,
    baseline_order: int = DEFAULT_BASELINE_ORDER,
    baseline_edge_threshold: float = DEFAULT_BASELINE_EDGE_THRESHOLD,
    baseline_smooth_threshold: float = DEFAULT_BASELINE_SMOOTH_THRESHOLD,
    doublet_kwargs: Optional[dict[str, Any]] = None,
    jobs: Optional[int] = None,
) -> PlanFitOutcome:
    """Walk a Stage 4 :class:`WindowPlan` and fit every window on the active-FT.

    The algorithm:

    1. Visit windows in :attr:`WindowPlan.topological_order` (a window's
       fixed-contributor primaries are guaranteed already-fit when it is
       reached). Windows in the same batch are mutually independent.
    2. For each window: slice the active-FT into the fit frame (just a grid
       relabel -- the active-FT is already in the ``[0, T]`` form), evaluate
       each :class:`FixedContributor` against its primary's
       :class:`WindowOutcome` to produce :class:`FrozenPeak` s, then call
       :func:`fit_window_with_fixed_contributors`.
    3. Compute :func:`residual_edge_coherence` on the full residual. For any
       edge above ``residual_edge_threshold``, pick a contributor via
       :func:`select_contributor_to_thaw` and run a :func:`local_thaw_cofit` of
       this window with its primary. Accept if the co-fit converges and the
       flagged-edge residual coherence drops to or below the threshold; rebuild
       both windows' outcomes from the joint fit. Bounded by
       ``max_thaw_rounds``.
    4. **Structural renegotiation** (when ``replan_context`` is provided).
       After the main walk, scan every outcome for flagged residual edges
       that have *no* fixed contributor on that side -- thaw cannot help
       them; a real feature crosses the boundary. Emit a
       :class:`~ftmwpipeline.core.data_structures.MergeRequest` pairing
       each such window with its frequency-adjacent neighbor, dedup by
       sorted pair, and route them through
       :func:`~ftmwpipeline.preprocessing.window_planning.replan`. Drop
       outcomes for the affected windows (mergers + transitive
       downstream), re-walk them in topo order on the revised plan. Bounded
       by ``replan_context.max_replan_rounds``.

    The full plan-level outcome is returned; this function does **no file IO**.
    Pure inputs in, pure outputs out.

    Parameters
    ----------
    plan : WindowPlan
        Stage 4 fit plan.
    active_ft : ActiveFTResult
        Active-portion FT of the experiment, computed by
        :func:`~ftmwpipeline.fitting.active_ft.compute_active_ft` from the
        FID + canonical Stage 1 settings. Carries the molecular frequency
        grid, complex spectrum, and ``alpha = N_active/N_padded``.
    rms_noise : np.ndarray
        Per-bin complex noise RMS on the **active-FT** grid, same shape as
        ``active_ft.complex_spectrum``.
    peak_frequencies_mhz : sequence of float
        Frequencies of *all* Stage 3 peaks (indexed by
        ``FitWindow.free_peak_indices`` and ``FixedContributor.peak_index``).
    sideband, acquisition_us, tau0_us : ...
        Pipeline-level parameters -- sideband sign, active acquisition length
        ``T``, and default shared decay. ``start_us`` is *not* needed
        (the active-FT is in the ``[0, T]`` form natively).
    fit_tau : bool, default True
        Forwarded to :func:`conservative_fit` for each window.
    residual_edge_threshold : float
        ``S_coh`` threshold for the residual edge-coherence test (O5-6).
    residual_edge_m : int
        Band width (in bins) of the residual edge-coherence test.
    max_thaw_rounds : int
        Maximum thaw attempts per window per call.
    conservative_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`conservative_fit`.
    replan_context : ReplanContext, optional
        Inputs for structural renegotiation. When ``None`` (the default),
        the structural outer loop is skipped -- existing tests that build
        synthetic :class:`ActiveFTResult` s directly need no extra context.
    max_residual_rescue_rounds : int, default 0
        Cap on per-window residual-rescue + joint-refit cycles. ``0``
        disables the rescue pass entirely (escape hatch for diagnostic
        re-fits); a positive value runs the B-loop on every window's
        post-thaw fit with that round cap.
    rescue_kwargs : dict, optional
        Tuning knobs for the rescue loop, forwarded to
        :func:`~ftmwpipeline.fitting.residual_rescue.rescue_and_consolidate`
        (``snr_threshold``, ``prominence_threshold``,
        ``rescue_significance``, ``knockout_significance``). The round-cap
        lives separately on
        ``max_residual_rescue_rounds``. Ignored when
        ``max_residual_rescue_rounds == 0``.
    window_tau_overrides : dict[int, (float, float)], optional
        Per-window override of the ``(tau_maj_us, sigma_tau_us)`` pair
        in ``conservative_kwargs`` and the per-window ``tau0_us`` seed.
        When set, every fit on a window whose id is present in the map
        uses the overriding pair (with all other conservative_kwargs
        entries unchanged) AND seeds its τ-parameter at the band-local
        ``tau_maj_us``. Used by Stage 5 when a per-band tau calibration
        is plumbed (resolved ``StageFitSettings.tau.per_band_tau``);
        windows missing from the map keep the band-wide
        ``tau_maj_us`` / ``sigma_tau_us`` from ``conservative_kwargs``
        and the band-wide ``tau0_us`` (or ``None`` if no calibration is
        wired). The ``tau0_us`` part is what gives weak windows with
        ``fit_tau=False`` their band-local fixed τ.
    spur_set : SpurSet, optional
        Gated clock/LO spurs (:mod:`ftmwpipeline.fitting.spur_detection`).
        For each window the executor derives a per-window
        :class:`~ftmwpipeline.fitting.spur_detection.SpurMaskSpec`, drops
        nominated candidate offsets that land on a spur, and threads the
        mask into every inner fit so spur bins are excluded from the
        residual / chi-squared. ``None`` (default) disables spur masking.
    baseline_enabled : bool, default :data:`DEFAULT_BASELINE_ENABLED`
        Apply the evidence-triggered leakage-wing baseline term. After thaw
        and rescue, any window whose residual ``max(edge_low, edge_high)``
        exceeds ``baseline_edge_threshold`` is refit jointly with a complex
        baseline of order ``baseline_order`` (the lines' tau held fixed) so a
        neighbor's mismodeled leakage skirt is absorbed as a smooth nuisance
        and the per-line uncertainties are priced from the joint covariance.
    baseline_order : int, default :data:`DEFAULT_BASELINE_ORDER`
        Baseline polynomial order ``p`` (0 = const, 1 = linear).
    baseline_edge_threshold : float, default
        :data:`DEFAULT_BASELINE_EDGE_THRESHOLD`
        ``S_coh`` threshold gating the baseline refit (a dedicated threshold
        well below the thaw default).

    Returns
    -------
    PlanFitOutcome
        Per-window outcomes, the chronological thaw / rescue / replan
        histories, and the final plan revision.

    Raises
    ------
    ValueError
        If a window's ``freq_range`` does not overlap the active-FT spectrum,
        or a fixed contributor references a primary window that has not been
        fit (which should be impossible given the topological walk).
    """
    if conservative_kwargs is None:
        conservative_kwargs = {}
    conservative_kwargs = dict(conservative_kwargs)
    conservative_kwargs.setdefault("shape", shape)
    if window_tau_overrides is None:
        window_tau_overrides = {}

    noise = np.asarray(rms_noise, dtype=float)
    if noise.shape != active_ft.complex_spectrum.shape:
        raise ValueError("rms_noise must match active_ft.complex_spectrum shape")

    outcomes: dict[int, WindowOutcome] = {}
    thaw_history: list[ThawEvent] = []
    rescue_history: list[RescueEvent] = []
    replan_history: list[ReplanEvent] = []

    # --- Initial walk over the plan as given --------------------------------
    # Cross-window parallel: levelize the DAG and fit each antichain concurrently
    # (barrier between levels). Falls back to the in-process sequential walk when
    # the pool is unavailable or forced off (``_FIT_WINDOW_WORKERS == 1``). The
    # post-replan re-walk below stays sequential (small affected set + the
    # structural-renegotiation bookkeeping is inherently serial).
    _t_initial = time.monotonic()
    _walk_windows_parallel(
        plan,
        plan.topological_order or [w.window_id for w in plan.windows],
        active_ft=active_ft,
        noise=noise,
        peak_frequencies_mhz=peak_frequencies_mhz,
        outcomes=outcomes,
        thaw_history=thaw_history,
        rescue_history=rescue_history,
        sideband=sideband,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us,
        fit_tau=fit_tau,
        residual_edge_threshold=residual_edge_threshold,
        residual_edge_m=residual_edge_m,
        max_thaw_rounds=max_thaw_rounds,
        conservative_kwargs=conservative_kwargs,
        max_residual_rescue_rounds=max_residual_rescue_rounds,
        rescue_kwargs=rescue_kwargs,
        window_tau_overrides=window_tau_overrides,
        spur_set=spur_set,
        baseline_enabled=baseline_enabled,
        baseline_order=baseline_order,
        baseline_edge_threshold=baseline_edge_threshold,
        baseline_smooth_threshold=baseline_smooth_threshold,
        doublet_kwargs=doublet_kwargs,
        jobs=jobs,
    )
    logger.info("initial walk: %.1fs", time.monotonic() - _t_initial)

    # --- Structural renegotiation loop -------------------------------------
    _t_replan = time.monotonic()
    if replan_context is not None:
        for _ in range(replan_context.max_replan_rounds):
            pending = _dispatch_structural_round(
                outcomes, plan, residual_edge_threshold
            )
            if not pending:
                break
            requests = _dedup_merge_requests([p.request for p in pending])
            try:
                new_plan = _do_replan(plan, requests, replan_context)
            except ValueError as exc:
                # Record the failure(s) and stop -- the plan stays as is.
                for trig in pending:
                    replan_history.append(
                        ReplanEvent(
                            triggering_window_id=trig.window_id,
                            partner_window_id=trig.partner_id,
                            surviving_window_id=min(trig.window_id, trig.partner_id),
                            edge_side=trig.edge_side,
                            edge_coherence_before=trig.edge_coherence,
                            revision_before=plan.plan_revision,
                            revision_after=plan.plan_revision,
                            accepted=False,
                            reason=f"replan failed: {exc}",
                        )
                    )
                break

            affected = _affected_after_replan(new_plan, requests)
            new_by_id = {w.window_id: w for w in new_plan.windows}
            # Drop outcomes that need refit (affected) AND outcomes whose
            # window was absorbed by a merge (the partner id no longer
            # appears in the revised plan).
            for wid in list(outcomes.keys()):
                if wid not in new_by_id or wid in affected:
                    outcomes.pop(wid, None)
            affected_order = [
                wid for wid in new_plan.topological_order if wid in affected
            ]
            # Re-fit the affected set through the same parallel walk as the
            # initial walk -- the affected windows form their own antichain DAG
            # (their non-affected primaries are already in ``outcomes`` and
            # treated as satisfied), so this parallelizes the structural-
            # renegotiation tail too. Small affected sets fall back to the
            # in-process sequential path inside the walk (level width < 2).
            _walk_windows_parallel(
                new_plan,
                affected_order,
                active_ft=active_ft,
                noise=noise,
                peak_frequencies_mhz=peak_frequencies_mhz,
                outcomes=outcomes,
                thaw_history=thaw_history,
                rescue_history=rescue_history,
                sideband=sideband,
                acquisition_us=acquisition_us,
                tau0_us=tau0_us,
                fit_tau=fit_tau,
                residual_edge_threshold=residual_edge_threshold,
                residual_edge_m=residual_edge_m,
                max_thaw_rounds=max_thaw_rounds,
                conservative_kwargs=conservative_kwargs,
                max_residual_rescue_rounds=max_residual_rescue_rounds,
                rescue_kwargs=rescue_kwargs,
                window_tau_overrides=window_tau_overrides,
                spur_set=spur_set,
                baseline_enabled=baseline_enabled,
                baseline_order=baseline_order,
                baseline_edge_threshold=baseline_edge_threshold,
                baseline_smooth_threshold=baseline_smooth_threshold,
                doublet_kwargs=doublet_kwargs,
                jobs=jobs,
            )

            applied_pairs = {
                tuple(sorted([r.window_a_id, r.window_b_id])) for r in requests
            }
            for trig in pending:
                pair = tuple(sorted([trig.window_id, trig.partner_id]))
                accepted = pair in applied_pairs and min(pair) in new_by_id
                replan_history.append(
                    ReplanEvent(
                        triggering_window_id=trig.window_id,
                        partner_window_id=trig.partner_id,
                        surviving_window_id=min(trig.window_id, trig.partner_id),
                        edge_side=trig.edge_side,
                        edge_coherence_before=trig.edge_coherence,
                        revision_before=plan.plan_revision,
                        revision_after=new_plan.plan_revision,
                        accepted=accepted,
                        reason=trig.reason,
                    )
                )
            plan = new_plan
    if replan_context is not None and replan_history:
        logger.info(
            "replan tail: %.1fs (%d attempts, %d accepted)",
            time.monotonic() - _t_replan,
            len(replan_history),
            sum(1 for r in replan_history if r.accepted),
        )

    return PlanFitOutcome(
        window_outcomes=outcomes,
        thaw_history=thaw_history,
        rescue_history=rescue_history,
        replan_history=replan_history,
        final_plan_revision=plan.plan_revision,
    )


def _build_doublet_refit_kwargs(
    ck_for_window: dict[str, Any],
    outcome: "WindowOutcome",
    acquisition_us: float,
) -> dict[str, Any]:
    """Build the ``refit_kwargs`` dict for the doublet-alternative merged refit.

    The merged refit must reproduce the production fit's penalty and constraint
    conditions. :func:`~ftmwpipeline.fitting.window_fit.derive_window_fit_constraints`
    is the source of truth for what ``fit_kwargs_inner`` contains; we derive it
    here from ``ck_for_window`` and the window's data so the penalty kwargs
    (tau, phase, amp) and fit_tau are consistent with what the production fit saw.
    ``spur_mask`` is handled separately at the call site (not in refit_kwargs).
    """
    data_minus_bg = np.asarray(
        outcome.complex_spectrum, dtype=np.complex128
    ) - np.asarray(outcome.background, dtype=np.complex128)
    inner = outcome.fit.fit
    constraints = derive_window_fit_constraints(
        data_minus_bg,
        outcome.rms_noise,
        float(inner.tau_us),
        acquisition_us,
        fit_tau=bool(inner.fit_tau),
        **{
            k: ck_for_window[k]
            for k in (
                "min_separation_factor",
                "max_decay_factor",
                "amp_max_headroom",
                "amp_penalty_lambda",
                "phase_penalty_lambda",
                "phase_penalty_cutoff_fwhm",
                "tau_penalty_lambda",
                "tau_penalty_n_sigma",
                "weak_window_snr_threshold",
                "fit_tau_min_snr",
                "tau_apodization_us",
                "tau_maj_us",
                "sigma_tau_us",
                "tau_anchor_us",
                "tau_penalty_sigma_lo_factor",
                "shape",
            )
            if k in ck_for_window
        },
    )
    refit_kwargs: dict[str, Any] = dict(constraints.fit_kwargs_inner)
    refit_kwargs.setdefault("shape", inner.shape)
    return refit_kwargs


def _process_one_window(
    win: FitWindow,
    *,
    n_done: int,
    n_total: int,
    active_ft: ActiveFTResult,
    noise: np.ndarray,
    peak_frequencies_mhz: Sequence[float],
    outcomes: dict[int, WindowOutcome],
    thaw_history: list[ThawEvent],
    rescue_history: list[RescueEvent],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    residual_edge_threshold: float,
    residual_edge_m: int,
    max_thaw_rounds: int,
    conservative_kwargs: dict[str, Any],
    max_residual_rescue_rounds: int,
    rescue_kwargs: Optional[dict[str, Any]],
    window_tau_overrides: dict[int, tuple[float, float]],
    spur_set: Optional[SpurSet],
    baseline_enabled: bool,
    baseline_order: int,
    baseline_edge_threshold: float,
    baseline_smooth_threshold: float,
    doublet_kwargs: Optional[dict[str, Any]],
) -> None:
    """Process one window end to end: conservative fit -> bounded thaw loop ->
    residual-rescue B-loop -> leakage-wing baseline -> doublet adjudication.

    Mutates ``outcomes[wid]`` (and, only on an accepted thaw, the contributor's
    primary outcome) and appends to ``thaw_history`` / ``rescue_history``.
    Extracted verbatim from :func:`_walk_windows_in_order` so the one per-window
    unit drives both the sequential walk and the per-level parallel pool.
    ``n_done`` / ``n_total`` are progress-logging context only.
    """
    wid = win.window_id
    t_start = time.monotonic()
    if os.environ.get("FTMW_DEBUG_FRINGE_DIR"):
        validation._fringe_window_ctx = {
            "wid": wid,
            "center": 0.5 * (win.freq_range[0] + win.freq_range[1]),
            "sign": sideband_sign(sideband),
        }
    ck_for_window = conservative_kwargs
    tau0_us_for_window = tau0_us
    if wid in window_tau_overrides:
        tau_maj_w, sigma_tau_w = window_tau_overrides[wid]
        ck_for_window = dict(conservative_kwargs)
        ck_for_window["tau_maj_us"] = float(tau_maj_w)
        ck_for_window["sigma_tau_us"] = float(sigma_tau_w)
        tau0_us_for_window = float(tau_maj_w)
    outcome = _fit_one_window(
        win,
        active_ft,
        noise,
        peak_frequencies_mhz=peak_frequencies_mhz,
        outcomes=outcomes,
        sideband=sideband,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us_for_window,
        fit_tau=fit_tau,
        residual_edge_m=residual_edge_m,
        conservative_kwargs=ck_for_window,
        spur_set=spur_set,
        early_baseline_order=baseline_order if baseline_enabled else None,
        early_baseline_smooth_threshold=(
            baseline_smooth_threshold if baseline_enabled else None
        ),
    )
    outcomes[wid] = outcome
    _debug_phase(wid, "conservative", outcome)

    for _ in range(max_thaw_rounds):
        edge_events = attempt_thaw_round(
            win,
            outcome,
            outcomes=outcomes,
            sideband=sideband,
            acquisition_us=acquisition_us,
            tau0_us=tau0_us_for_window,
            fit_tau=fit_tau,
            residual_edge_threshold=residual_edge_threshold,
            residual_edge_m=residual_edge_m,
            shape=conservative_kwargs.get("shape", "lorentzian"),
            spur_set=spur_set,
        )
        if not edge_events:
            break
        for event in edge_events:
            thaw_history.append(event)
        outcome = outcomes[wid]
        if not any(e.accepted for e in edge_events):
            break
    _debug_phase(wid, "post-thaw", outcome)

    if max_residual_rescue_rounds > 0:
        events = _apply_rescue_to_outcome(
            win,
            outcome,
            acquisition_us=acquisition_us,
            tau0_us=tau0_us_for_window,
            residual_edge_m=residual_edge_m,
            conservative_kwargs=ck_for_window,
            max_residual_rescue_rounds=max_residual_rescue_rounds,
            rescue_kwargs=rescue_kwargs or {},
            spur_set=spur_set,
        )
        for ev in events:
            rescue_history.append(ev)
        outcome = outcomes[wid]
    _debug_phase(wid, "post-rescue", outcome)

    if baseline_enabled:
        _apply_baseline_to_outcome(
            outcome,
            acquisition_us=acquisition_us,
            residual_edge_m=residual_edge_m,
            baseline_order=baseline_order,
            baseline_edge_threshold=baseline_edge_threshold,
            baseline_smooth_threshold=baseline_smooth_threshold,
            tau0_us=tau0_us_for_window,
            conservative_kwargs=ck_for_window,
        )
    _debug_phase(wid, "post-baseline", outcome)

    if doublet_kwargs is not None:
        try:
            inner = outcome.fit.fit
            spur_mask_for_doublet = getattr(outcome, "_spur_mask", None)
            # Derive refit_kwargs that reproduce the production fit conditions:
            # the same tau-penalty, phase/amp-penalty, and fit_window-level
            # kwargs that conservative_fit uses via derive_window_fit_constraints.
            # Build from ck_for_window (includes per-window tau overrides).
            refit_kwargs_for_doublet = _build_doublet_refit_kwargs(
                ck_for_window, outcome, acquisition_us
            )
            adjudications: list[DoubletAdjudication] = adjudicate_close_pairs(
                offset_grid_mhz=outcome.offset_grid_mhz,
                complex_spectrum=outcome.complex_spectrum - outcome.background,
                rms_noise=outcome.rms_noise,
                fit=inner,
                acquisition_us=acquisition_us,
                shape=ck_for_window.get("shape", "lorentzian"),
                k_res=doublet_kwargs["k_res"],
                r_min=doublet_kwargs["r_min"],
                frozen_background=outcome.background,
                spur_mask=spur_mask_for_doublet,
                refit_kwargs=refit_kwargs_for_doublet,
            )
            outcome.doublet_adjudications = adjudications
        except Exception:
            logger.warning(
                "doublet-alternative pass failed on window %d; skipping",
                wid,
                exc_info=False,
            )
            outcome.doublet_adjudications = []

    elapsed = time.monotonic() - t_start
    final = outcomes[wid]
    log = logger.warning if elapsed > 60.0 else logger.info
    log(
        "window %d/%d w%d [%.1f-%.1f MHz]: %d peaks, chi2r=%.3g, %.1fs",
        n_done,
        n_total,
        wid,
        win.freq_range[0],
        win.freq_range[1],
        final.fit.n_peaks,
        final.fit.fit.reduced_chi2,
        elapsed,
    )


def _walk_windows_in_order(
    plan: WindowPlan,
    order: Sequence[int],
    *,
    active_ft: ActiveFTResult,
    noise: np.ndarray,
    peak_frequencies_mhz: Sequence[float],
    outcomes: dict[int, WindowOutcome],
    thaw_history: list[ThawEvent],
    rescue_history: list[RescueEvent],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    residual_edge_threshold: float,
    residual_edge_m: int,
    max_thaw_rounds: int,
    conservative_kwargs: dict[str, Any],
    max_residual_rescue_rounds: int = 0,
    rescue_kwargs: Optional[dict[str, Any]] = None,
    window_tau_overrides: Optional[dict[int, tuple[float, float]]] = None,
    spur_set: Optional[SpurSet] = None,
    baseline_enabled: bool = DEFAULT_BASELINE_ENABLED,
    baseline_order: int = DEFAULT_BASELINE_ORDER,
    baseline_edge_threshold: float = DEFAULT_BASELINE_EDGE_THRESHOLD,
    baseline_smooth_threshold: float = DEFAULT_BASELINE_SMOOTH_THRESHOLD,
    doublet_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Fit each window in ``order``, run the bounded local-thaw loop, and
    (when ``max_residual_rescue_rounds > 0``) the residual-rescue B-loop.

    Mutates ``outcomes``, ``thaw_history``, and ``rescue_history`` in place.
    Shared by the initial walk and the post-replan re-walk of affected
    windows.

    ``window_tau_overrides`` (optional) maps window_id to a
    ``(tau_maj_us, sigma_tau_us)`` pair that overrides the same keys in
    ``conservative_kwargs`` AND the per-window ``tau0_us`` seed for that
    window only -- used by the per-band Stage 5 path so each window sees
    its band-local tau anchor in both the prior penalty (``tau_maj_us``)
    and the τ-parameter starting point. Without the ``tau0_us`` override
    a weak window with ``fit_tau=False`` would sit pinned at the band-
    wide ``tau0_us`` regardless of band; the override routes the seed to
    the band-local majority so fixed-τ windows in different bands land
    at different τ.

    Order of work per window:

    1. Initial conservative fit (with frozen contributors).
    2. Bounded local-thaw loop -- a thawed contributor can promote a frozen
       line to a free peak that the rescue should then see in the model.
    3. Residual-rescue B-loop, when ``max_residual_rescue_rounds > 0``. The
       rescue operates on the post-thaw outcome so any contributor's line
       that thaw promoted is already part of the model.
    4. Leakage-wing baseline refit, when ``baseline_enabled``. Operates on
       the final (post-thaw, post-rescue) residual so removing the wing
       de-biases the rescue-confirmed lines and the trigger sees the
       cleanest residual; sequenced last and independent of rescue.
    """
    if window_tau_overrides is None:
        window_tau_overrides = {}
    by_id = {w.window_id: w for w in plan.windows}
    n_total = len(order)
    for n_done, wid in enumerate(order, start=1):
        win = by_id[wid]
        _process_one_window(
            win,
            n_done=n_done,
            n_total=n_total,
            active_ft=active_ft,
            noise=noise,
            peak_frequencies_mhz=peak_frequencies_mhz,
            outcomes=outcomes,
            thaw_history=thaw_history,
            rescue_history=rescue_history,
            sideband=sideband,
            acquisition_us=acquisition_us,
            tau0_us=tau0_us,
            fit_tau=fit_tau,
            residual_edge_threshold=residual_edge_threshold,
            residual_edge_m=residual_edge_m,
            max_thaw_rounds=max_thaw_rounds,
            conservative_kwargs=conservative_kwargs,
            max_residual_rescue_rounds=max_residual_rescue_rounds,
            rescue_kwargs=rescue_kwargs,
            window_tau_overrides=window_tau_overrides,
            spur_set=spur_set,
            baseline_enabled=baseline_enabled,
            baseline_order=baseline_order,
            baseline_edge_threshold=baseline_edge_threshold,
            baseline_smooth_threshold=baseline_smooth_threshold,
            doublet_kwargs=doublet_kwargs,
        )


# ---------------------------------------------------------------------------
# Cross-window parallel walk: antichain levelization + fork-per-level pool
# ---------------------------------------------------------------------------
# ``None`` => auto (cpu_count - 2); ``1`` => force the in-process sequential walk
# (the scientific-equivalence reference). Any int pins the worker count for tests.
_FIT_WINDOW_WORKERS: Optional[int] = None
# Set in the parent before each level's pool forks; read by the worker entry via
# fork inheritance (the large shared ``active_ft``/``noise`` and the earlier-level
# ``outcomes`` ride along on the fork, never pickled per task).
_WORKER_FIT_CTX: Optional[dict[str, Any]] = None


def _levelize(
    order: Sequence[int],
    by_id: dict[int, FitWindow],
    dependency_edges: Sequence[tuple[int, int]] = (),
) -> list[list[int]]:
    """Layer the window DAG into antichains (levels) over ``order``.

    The fit-ordering dependency of a window is authoritative in its
    ``fixed_contributors``: a contributor that is **not** ``edge_free`` reads its
    ``primary_window_id``'s fitted outcome (:func:`evaluate_fixed_contributor`),
    so that primary must be fit first; an ``edge_free`` contributor is
    materialized self-contained from the shared active FT and imposes no ordering.
    ``dependency_edges`` (``(child, parent)`` pairs) is folded in as a belt-and-
    braces superset -- the plan does not always populate it, so the contributor
    scan is the primary source. Returns a list of levels; level *k* holds exactly
    the windows whose every dependency is placed in a level ``< k`` (longest-path
    layering), so the windows within a level are mutually independent and safe to
    fit concurrently. ``order`` ordering is preserved within each level for
    deterministic logging. Only dependencies with both endpoints in ``order``
    constrain the layering (the post-replan re-walk passes a subset).
    """
    in_order = list(order)
    idset = set(in_order)
    preds: dict[int, set[int]] = {w: set() for w in in_order}
    for wid in in_order:
        win = by_id.get(wid)
        if win is None:
            continue
        for contributor in win.fixed_contributors:
            if contributor.edge_free:
                continue
            primary = contributor.primary_window_id
            if primary in idset and primary != wid:
                preds[wid].add(primary)
    for child, parent in dependency_edges:
        if child in idset and parent in idset and child != parent:
            preds[child].add(parent)
    levels: list[list[int]] = []
    placed: set[int] = set()
    remaining = list(in_order)
    guard = 0
    while remaining and guard <= len(in_order) + 1:
        guard += 1
        ready = [w for w in remaining if preds[w] <= placed]
        if not ready:
            # A cycle slipped through the cycle-breaker -- degrade to a single
            # sequential level rather than drop windows.
            ready = list(remaining)
        levels.append(ready)
        placed.update(ready)
        ready_set = set(ready)
        remaining = [w for w in remaining if w not in ready_set]
    return levels


def _fit_window_worker(
    task: tuple[int, int],
) -> tuple[int, WindowOutcome, list[ThawEvent], list[RescueEvent]]:
    """Process-pool entry point: fit one window from the fork-inherited context.

    Pins BLAS to a single thread (the per-window solve is single-threaded; this
    avoids N-workers x M-BLAS-threads oversubscription). Works on a shallow copy
    of the inherited ``outcomes`` so concurrent tasks reused on the same worker
    process never see each other's writes -- within a level the windows are an
    antichain, so the copy only needs the earlier-level primaries (present in the
    inherited dict). Returns the window's outcome plus its local thaw / rescue
    events for the parent to merge.
    """
    import os

    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_var] = "1"
    ctx = _WORKER_FIT_CTX
    assert ctx is not None  # set in the parent before the pool forks
    n_done, wid = task
    local_outcomes: dict[int, WindowOutcome] = dict(ctx["outcomes"])
    thaw_local: list[ThawEvent] = []
    rescue_local: list[RescueEvent] = []
    _process_one_window(
        ctx["by_id"][wid],
        n_done=n_done,
        n_total=ctx["n_total"],
        outcomes=local_outcomes,
        thaw_history=thaw_local,
        rescue_history=rescue_local,
        **ctx["shared"],
    )
    return wid, local_outcomes[wid], thaw_local, rescue_local


def _walk_windows_parallel(
    plan: WindowPlan,
    order: Sequence[int],
    *,
    active_ft: ActiveFTResult,
    noise: np.ndarray,
    peak_frequencies_mhz: Sequence[float],
    outcomes: dict[int, WindowOutcome],
    thaw_history: list[ThawEvent],
    rescue_history: list[RescueEvent],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    residual_edge_threshold: float,
    residual_edge_m: int,
    max_thaw_rounds: int,
    conservative_kwargs: dict[str, Any],
    max_residual_rescue_rounds: int = 0,
    rescue_kwargs: Optional[dict[str, Any]] = None,
    window_tau_overrides: Optional[dict[int, tuple[float, float]]] = None,
    spur_set: Optional[SpurSet] = None,
    baseline_enabled: bool = DEFAULT_BASELINE_ENABLED,
    baseline_order: int = DEFAULT_BASELINE_ORDER,
    baseline_edge_threshold: float = DEFAULT_BASELINE_EDGE_THRESHOLD,
    baseline_smooth_threshold: float = DEFAULT_BASELINE_SMOOTH_THRESHOLD,
    doublet_kwargs: Optional[dict[str, Any]] = None,
    jobs: Optional[int] = None,
) -> None:
    """Cross-window parallel form of :func:`_walk_windows_in_order`.

    Levelizes ``order`` into antichains and fits each level concurrently across a
    forking process pool, with a barrier between levels so every window's
    edged-contributor primaries (which sit in strictly earlier levels) are already
    fit when it runs. A fresh pool is forked **per level** after the prior level's
    outcomes are merged into ``outcomes``, so each level's workers inherit the
    complete earlier-level results via fork. Falls back to the in-process
    sequential walk when the pool is disabled (``_FIT_WINDOW_WORKERS == 1``), only
    one worker is available, the platform lacks ``fork``, or a level has a single
    window.

    Correctness gate is *scientific equivalence*, not byte-identity: structure
    (windows / peak counts / merges / thaws) is identical to the sequential walk;
    continuous params drift only at the fork+BLAS=1 ULP scale. The one structural
    hazard -- an accepted thaw mutates its primary window in place, which a worker
    can only do to its fork-private copy -- is guarded: if any worker in a level
    reports an accepted thaw, the whole level is re-fit sequentially in the parent
    (this never fires on the validated fixtures, where thaw-accept = 0).
    """
    if window_tau_overrides is None:
        window_tau_overrides = {}

    shared_kwargs: dict[str, Any] = dict(
        active_ft=active_ft,
        noise=noise,
        peak_frequencies_mhz=peak_frequencies_mhz,
        sideband=sideband,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us,
        fit_tau=fit_tau,
        residual_edge_threshold=residual_edge_threshold,
        residual_edge_m=residual_edge_m,
        max_thaw_rounds=max_thaw_rounds,
        conservative_kwargs=conservative_kwargs,
        max_residual_rescue_rounds=max_residual_rescue_rounds,
        rescue_kwargs=rescue_kwargs,
        window_tau_overrides=window_tau_overrides,
        spur_set=spur_set,
        baseline_enabled=baseline_enabled,
        baseline_order=baseline_order,
        baseline_edge_threshold=baseline_edge_threshold,
        baseline_smooth_threshold=baseline_smooth_threshold,
        doublet_kwargs=doublet_kwargs,
    )

    import multiprocessing

    max_workers = resolve_worker_count(jobs, override=_FIT_WINDOW_WORKERS)
    pool_available = (
        max_workers >= 2 and "fork" in multiprocessing.get_all_start_methods()
    )
    # No pool (forced off, single worker, or no fork): run the original
    # topological-order sequential walk -- the scientific-equivalence reference.
    if not pool_available:
        _walk_windows_in_order(
            plan,
            order,
            outcomes=outcomes,
            thaw_history=thaw_history,
            rescue_history=rescue_history,
            **shared_kwargs,
        )
        return

    by_id = {w.window_id: w for w in plan.windows}
    levels = _levelize(order, by_id, plan.dependency_edges)
    n_total = len(order)

    def _run_sequential(level: Sequence[int], n_done0: int) -> None:
        nd = n_done0
        for wid in level:
            nd += 1
            _process_one_window(
                by_id[wid],
                n_done=nd,
                n_total=n_total,
                outcomes=outcomes,
                thaw_history=thaw_history,
                rescue_history=rescue_history,
                **shared_kwargs,
            )

    if len(levels) > 1 or any(len(lv) > 1 for lv in levels):
        logger.info(
            "fit walk: %d windows in %d levels (max width %d), %d workers",
            n_total,
            len(levels),
            max(len(lv) for lv in levels) if levels else 0,
            max_workers,
        )

    n_done = 0
    for li, level in enumerate(levels):
        t_level = time.monotonic()
        # A width-1 level forks nothing -- run it in-process (no pool overhead).
        if len(level) < 2:
            _run_sequential(level, n_done)
            n_done += len(level)
            continue

        # Fork a fresh pool for this level so workers inherit every earlier-level
        # outcome merged into ``outcomes`` below. Only the (n_done, wid) tuples go
        # out; the shared arrays and outcomes ride the fork.
        global _WORKER_FIT_CTX
        _WORKER_FIT_CTX = {
            "by_id": by_id,
            "outcomes": outcomes,
            "n_total": n_total,
            "shared": shared_kwargs,
        }
        tasks = [(n_done + i + 1, wid) for i, wid in enumerate(level)]
        results: dict[int, tuple[int, WindowOutcome, list, list]] = {}
        try:
            from concurrent.futures import ProcessPoolExecutor

            ctx_mp = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(max_workers, len(level)), mp_context=ctx_mp
            ) as ex:
                for res in ex.map(_fit_window_worker, tasks):
                    results[res[0]] = res
        finally:
            _WORKER_FIT_CTX = None

        any_thaw_accepted = any(
            any(ev.accepted for ev in res[2]) for res in results.values()
        )
        if any_thaw_accepted:
            # An accepted thaw mutates the contributor's primary window in place;
            # a worker only mutated its fork-private copy. Re-fit the whole level
            # sequentially in the parent so the primary mutation is real and
            # cross-window ordering matches the sequential walk. Discard the
            # parallel outcomes for this level.
            logger.warning(
                "accepted thaw in a parallel level; re-fitting %d windows "
                "sequentially for cross-window correctness",
                len(level),
            )
            _run_sequential(level, n_done)
        else:
            for wid in level:
                _, outcome, thaws, rescues = results[wid]
                outcomes[wid] = outcome
                thaw_history.extend(thaws)
                rescue_history.extend(rescues)
        n_done += len(level)
        logger.info(
            "  level %d/%d: %d windows on %d workers, %.1fs",
            li + 1,
            len(levels),
            len(level),
            min(max_workers, len(level)),
            time.monotonic() - t_level,
        )


# ---------------------------------------------------------------------------
# Structural renegotiation: dispatch + replan + affected-window bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class _PendingMerge:
    """Internal dispatcher record: one window's flagged-edge request."""

    window_id: int
    partner_id: int
    edge_side: str
    edge_coherence: float
    reason: str
    request: MergeRequest


def _find_adjacent_window(
    plan: WindowPlan, win: FitWindow, side: str
) -> Optional[FitWindow]:
    """Return the immediately-adjacent window in ``side`` direction, or None.

    Adjacency is in molecular frequency: the ``"low"``-side neighbor is the
    window whose ``freq_range[1]`` is the largest value still ``<=
    win.freq_range[0]``; ``"high"`` mirrors. Returns ``None`` when ``win`` is
    at the plan's outer boundary on that side.
    """
    if side not in ("low", "high"):
        raise ValueError("side must be 'low' or 'high'")
    candidates = [w for w in plan.windows if w.window_id != win.window_id]
    if side == "low":
        below = [w for w in candidates if w.freq_range[1] <= win.freq_range[0]]
        if not below:
            return None
        return max(below, key=lambda w: w.freq_range[1])
    above = [w for w in candidates if w.freq_range[0] >= win.freq_range[1]]
    if not above:
        return None
    return min(above, key=lambda w: w.freq_range[0])


def _dispatch_structural_round(
    outcomes: dict[int, WindowOutcome],
    plan: WindowPlan,
    residual_edge_threshold: float,
) -> list[_PendingMerge]:
    """Scan outcomes for flagged edges that warrant a structural merge.

    A flagged edge is acted on as a merge only when both:

    * **no fixed contributor on that side** -- otherwise the local thaw
      handshake (already applied in the main walk) is the right tool and
      will have either resolved it or recorded its failure;
    * **an adjacent window exists** in that direction in the current plan.

    Multiple windows may emit overlapping requests (e.g. window A's high
    edge and window B's low edge both naming the same A-B merge); the
    deduplication happens at :func:`_dedup_merge_requests`.
    """
    by_id = {w.window_id: w for w in plan.windows}
    pending: list[_PendingMerge] = []
    for wid, outcome in outcomes.items():
        win = by_id.get(wid)
        if win is None:
            continue
        for edge_side, edge_coh in (
            ("low", outcome.edge_coherence_low),
            ("high", outcome.edge_coherence_high),
        ):
            if not np.isfinite(edge_coh) or edge_coh <= residual_edge_threshold:
                continue
            # Only edge-bearing contributors mean "thaw owns this edge". An
            # edge-free contributor cannot be thawed, so a still-flagged edge
            # beside one is fair game for a structural merge (a real feature may
            # cross the boundary that the frozen skirt cannot represent).
            side_contribs = [
                fp
                for fp in outcome.fixed_peaks
                if not fp.edge_free
                and (
                    fp.frequency_mhz <= win.freq_range[0]
                    if edge_side == "low"
                    else fp.frequency_mhz >= win.freq_range[1]
                )
            ]
            if side_contribs:
                continue  # thaw owns this edge; structural merge is not the tool
            adjacent = _find_adjacent_window(plan, win, edge_side)
            if adjacent is None:
                continue
            reason = (
                f"residual edge-coherence {edge_coh:.2f} on {edge_side} side, "
                f"no contributor to thaw; merging with adjacent "
                f"window {adjacent.window_id}"
            )
            pending.append(
                _PendingMerge(
                    window_id=win.window_id,
                    partner_id=adjacent.window_id,
                    edge_side=edge_side,
                    edge_coherence=float(edge_coh),
                    reason=reason,
                    request=MergeRequest(
                        window_a_id=win.window_id,
                        window_b_id=adjacent.window_id,
                        reason=reason,
                    ),
                )
            )
    return pending


def _dedup_merge_requests(
    requests: Sequence[MergeRequest],
) -> list[MergeRequest]:
    """Drop duplicate merge requests by sorted ``(a, b)`` pair."""
    seen: set[tuple[int, int]] = set()
    out: list[MergeRequest] = []
    for req in requests:
        lo = min(req.window_a_id, req.window_b_id)
        hi = max(req.window_a_id, req.window_b_id)
        pair: tuple[int, int] = (lo, hi)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(req)
    return out


_REPLAN_PARAM_KEYS = (
    "edge_m",
    "trim_m",
    "edge_threshold",
    "max_window_width_mhz",
    "max_window_width_points",
    "min_freeze_snr",
    "min_window_half_width_mhz",
    "min_window_half_width_points",
    "magnitude_attachment_threshold",
    "max_edge_free_neighbors",
    "acquisition_us",
    "tau_us",
)


def _do_replan(
    plan: WindowPlan,
    requests: list[MergeRequest],
    ctx: ReplanContext,
) -> WindowPlan:
    """Forward the requests to Stage 4's :func:`replan` with parameters
    sourced from ``plan.parameters`` (Stage 4 records them on the plan when
    it builds it; we pull them back off the plan here so the caller does not
    have to thread Stage 4 parameters separately).
    """
    kwargs = {k: plan.parameters[k] for k in _REPLAN_PARAM_KEYS if k in plan.parameters}
    return stage4_replan(
        plan,
        requests,
        ctx.peaks,
        ctx.active_freq_mhz,
        ctx.active_complex_spectrum,
        ctx.active_rms_noise,
        **kwargs,
    )


def _affected_after_replan(
    new_plan: WindowPlan,
    requests: Sequence[MergeRequest],
) -> set[int]:
    """Set of ``window_id`` s whose outcomes must be re-fit after a replan.

    Includes every merge survivor plus any window in ``new_plan`` that
    transitively depends on one of them.
    """
    affected: set[int] = set()
    new_ids = {w.window_id for w in new_plan.windows}
    for req in requests:
        survivor = min(req.window_a_id, req.window_b_id)
        if survivor in new_ids:
            affected.add(survivor)
    changed = True
    while changed:
        changed = False
        for child, parent in new_plan.dependency_edges:
            if parent in affected and child not in affected:
                affected.add(child)
                changed = True
    return affected


def _fit_one_window(
    win: FitWindow,
    active_ft: ActiveFTResult,
    noise: np.ndarray,
    *,
    peak_frequencies_mhz: Sequence[float],
    outcomes: dict[int, WindowOutcome],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    residual_edge_m: int,
    conservative_kwargs: dict[str, Any],
    spur_set: Optional[SpurSet] = None,
    early_baseline_order: Optional[int] = None,
    early_baseline_smooth_threshold: Optional[float] = None,
    protected_offsets: Optional[Sequence[float]] = None,
    protected_tol_mhz: float = 0.0,
    forbidden_offsets: Optional[Sequence[float]] = None,
    forbidden_tol_mhz: float = 0.0,
) -> WindowOutcome:
    """Fit one window with its frozen contributors; build its WindowOutcome."""
    _, offset_grid, z_slice, sig_slice, center_mhz = materialize_window(
        win,
        active_ft,
        noise,
        sideband=sideband,
    )

    # Per-window spur mask: spurs that fall inside this window, in the
    # window's baseband-offset frame. Stashed on the outcome so the thaw and
    # rescue passes mask the same bins, and threaded into the fit below.
    spur_mask: Optional[SpurMaskSpec] = None
    if spur_set is not None and spur_set:
        lo, hi = win.freq_range
        spur_mask = spur_set.window_mask_spec(lo, hi, center_mhz, sideband)

    fixed_peaks: list[FrozenPeak] = []
    edge_free_contributors: list[FixedContributor] = []
    for contributor in win.fixed_contributors:
        # A fixed contributor whose frequency lands on a spur is a spurious
        # frozen term: the spur is no longer fit as a peak in its primary
        # window (nomination exclusion), so it has no fitted peak to freeze
        # here. Drop it -- the spur's own bins are masked from this window's
        # residual anyway.
        if (
            spur_set is not None
            and spur_set
            and spur_set.candidate_on_spur(contributor.frequency_mhz)
        ):
            continue
        if contributor.edge_free:
            # Read self-contained from the active FT below -- no primary
            # outcome required (the whole point of an edge-free contributor).
            edge_free_contributors.append(contributor)
            continue
        primary_outcome = outcomes.get(contributor.primary_window_id)
        if primary_outcome is None:
            raise ValueError(
                f"window {win.window_id} depends on un-fit primary window "
                f"{contributor.primary_window_id} (topological_order broken)"
            )
        fixed_peaks.append(
            evaluate_fixed_contributor(
                contributor,
                primary_outcome,
                dependent_center_mhz=center_mhz,
                sideband=sideband,
            )
        )
    edge_free_peaks: list[FrozenPeak] = []
    if edge_free_contributors:
        edge_free_peaks = evaluate_edge_free_contributors(
            edge_free_contributors,
            active_ft.freq_mhz,
            active_ft.complex_spectrum,
            dependent_center_mhz=center_mhz,
            sideband=sideband,
            tau_us=tau0_us,
            acquisition_us=acquisition_us,
            shape=conservative_kwargs.get("shape", "lorentzian"),
        )

    candidate_offsets = _peaks_to_candidate_offsets(
        win, peak_frequencies_mhz, center_mhz, sideband
    )
    # Drop nominated candidates that land on a spur core so the fitter never
    # seeds a peak on a spur. The residual mask still spans +/-N bins; the
    # tighter nomination tolerance spares a real line a couple of bins away.
    ck_for_fit = conservative_kwargs
    if spur_mask is not None and spur_mask.offsets_mhz:
        tol = spur_set.nomination_tol_mhz  # type: ignore[union-attr]
        candidate_offsets = [
            o
            for o in candidate_offsets
            if not any(abs(o - so) <= tol for so in spur_mask.offsets_mhz)
        ]
        ck_for_fit = dict(conservative_kwargs)
        ck_for_fit["spur_mask"] = spur_mask
    # Inject per-window immunity params when provided (no-op when None).
    if protected_offsets is not None or forbidden_offsets is not None:
        if ck_for_fit is conservative_kwargs:
            ck_for_fit = dict(conservative_kwargs)
        if protected_offsets is not None:
            ck_for_fit["protected_offsets"] = protected_offsets
            ck_for_fit["protected_tol_mhz"] = protected_tol_mhz
        if forbidden_offsets is not None:
            ck_for_fit["forbidden_offsets"] = forbidden_offsets
            ck_for_fit["forbidden_tol_mhz"] = forbidden_tol_mhz
    # Edge-bearing contributors are always carried. Edge-free contributors are
    # evidence-triggered: fit the window without them first (the byte-stable
    # path for a healthy window whose leakage the const baseline already
    # handles), then only adopt the edge-free skirt if it clearly reduces the
    # noise-weighted residual -- so an orphaned bright neighbor's skirt is
    # subtracted where it helps without fighting the baseline elsewhere (O2).
    fit_result, background, full_fitted, full_residual = (
        fit_window_with_fixed_contributors(
            offset_grid,
            z_slice,
            sig_slice,
            fixed_peaks,
            candidate_offsets,
            tau0_us,
            acquisition_us,
            fit_tau=fit_tau,
            early_baseline_order=early_baseline_order,
            early_baseline_smooth_threshold=early_baseline_smooth_threshold,
            **ck_for_fit,
        )
    )
    edge_free_accepted = False
    if edge_free_peaks:
        ef_result, ef_bg, ef_full, ef_residual = fit_window_with_fixed_contributors(
            offset_grid,
            z_slice,
            sig_slice,
            fixed_peaks + edge_free_peaks,
            candidate_offsets,
            tau0_us,
            acquisition_us,
            fit_tau=fit_tau,
            early_baseline_order=early_baseline_order,
            early_baseline_smooth_threshold=early_baseline_smooth_threshold,
            **ck_for_fit,
        )
        ssr_without = _noise_weighted_ssr(full_residual, sig_slice)
        ssr_with = _noise_weighted_ssr(ef_residual, sig_slice)
        # The arbitration must price model complexity in the same currency as
        # the accept gate. The frozen skirt adds zero free parameters, but the
        # no-skirt fit can spend *peaks* absorbing the un-subtracted skirt
        # energy (pedestal + wing fringes) -- under a permissive gate its SSR
        # approaches the with-skirt fit's and a raw-SSR margin then rejects
        # the skirt, leaving the pedestal for the rescue to "explain" with
        # spurious lines (the dependent-window flood). Under the penalized
        # gate, score both sides as the gate does (``SSR + 2*lambda*k``) so
        # peak-bought residual reduction is charged for; the legacy gate keeps
        # the raw-SSR fraction rule it was calibrated with.
        penalty_lambda = validation.DEFAULT_GATE_PENALTY_LAMBDA
        if penalty_lambda is not None:
            pen = 2.0 * float(penalty_lambda)
            score_with = ssr_with + pen * float(ef_result.fit.n_params)
            score_without = ssr_without + pen * float(fit_result.fit.n_params)
            adopt_skirt = score_with <= score_without
        else:
            adopt_skirt = ssr_with <= DEFAULT_EDGE_FREE_ACCEPT_FRACTION * ssr_without
        if os.environ.get("FTMW_FORCE_SKIRT"):
            # Forensics only: force the edge-free skirt adoption so the full
            # downstream machinery (rescue, blend-split, baseline re-runs)
            # can be observed on the with-skirt branch of the A/B.
            adopt_skirt = True
        if os.environ.get("FTMW_DEBUG_SKIRT"):
            off_with = [round(p.offset_mhz, 3) for p in ef_result.fit.peaks]
            off_without = [round(p.offset_mhz, 3) for p in fit_result.fit.peaks]
            print(
                f"[skirt] w{win.window_id}: n_ef={len(edge_free_peaks)} "
                f"ssr_with={ssr_with:.1f} ssr_without={ssr_without:.1f} "
                f"k_with={ef_result.fit.n_params} "
                f"k_without={fit_result.fit.n_params} adopt={adopt_skirt} "
                f"peaks_with={off_with} peaks_without={off_without}",
                flush=True,
            )
            validation.debug_fringe_dump(
                "skirtab",
                u=offset_grid,
                z=z_slice,
                sigma=sig_slice,
                bg_with=ef_bg,
                bg_without=background,
                ssr_with=ssr_with,
                ssr_without=ssr_without,
                adopt=int(adopt_skirt),
            )
        if adopt_skirt:
            fit_result = ef_result
            background = ef_bg
            full_fitted = ef_full
            full_residual = ef_residual
            fixed_peaks = fixed_peaks + edge_free_peaks
            edge_free_accepted = True
    outcome = build_window_outcome(
        fit_result,
        win.window_id,
        fixed_peaks,
        offset_grid,
        z_slice,
        sig_slice,
        background,
        full_fitted,
        full_residual,
        residual_edge_m,
        center_mhz,
        spur_mask,
    )
    # An early (conservative-phase) leakage-wing baseline is part of the
    # persisted model: mirror it onto the outcome's audit fields so
    # serialization and rendering carry it even when the post-rescue
    # baseline pass finds nothing further to do. The trigger here was the
    # smooth-residual statistic, not an edge S_coh; the post-rescue pass
    # overwrites these (including the coherence) if it re-fires.
    if fit_result.fit.baseline_order is not None:
        outcome.baseline_applied = True
        outcome.baseline_order = fit_result.fit.baseline_order
        outcome.baseline_coeffs = fit_result.fit.baseline_coeffs
        outcome.baseline_offset_scale = fit_result.fit.baseline_offset_scale
        outcome.baseline_edge_coherence = float(
            max(outcome.edge_coherence_low, outcome.edge_coherence_high)
            if np.isfinite(outcome.edge_coherence_low)
            and np.isfinite(outcome.edge_coherence_high)
            else 0.0
        )
    # Diagnostic: whether the evidence-triggered edge-free skirt was adopted.
    outcome._edge_free_accepted = edge_free_accepted  # type: ignore[attr-defined]
    return outcome


def attempt_thaw_round(
    win: FitWindow,
    outcome: WindowOutcome,
    *,
    outcomes: dict[int, WindowOutcome],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool = True,
    residual_edge_threshold: float = DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    residual_edge_m: int = DEFAULT_RESIDUAL_EDGE_M,
    shape: "PeakShape | str" = "lorentzian",
    spur_set: Optional[SpurSet] = None,
) -> list[ThawEvent]:
    """Run one round of the residual edge-coherence check on a window and thaw.

    Returns the :class:`ThawEvent` records produced this round (one per
    above-threshold edge that was acted on). An empty return means no edge was
    over threshold (no thaw needed); a return with all ``accepted=False`` means
    we tried but the co-fit did not improve things.

    The public surface :func:`execute_plan` walks this in a bounded loop. This
    function is exposed for callers that want manual control over the
    renegotiation loop (and the tests that verify a single round).

    Parameters
    ----------
    win : FitWindow
        The dependent window to check.
    outcome : WindowOutcome
        The (already-fit) outcome of ``win``; mutated in place if a thaw is
        accepted.
    outcomes : dict of int -> WindowOutcome
        All known outcomes, by id; the contributor's primary outcome must be
        present.
    sideband, acquisition_us, tau0_us, fit_tau : ...
        Pipeline-level parameters propagated to :func:`local_thaw_cofit`.
    residual_edge_threshold, residual_edge_m : ...
        See :func:`execute_plan`.
    """
    events: list[ThawEvent] = []
    for edge_side, edge_coh in (
        ("low", outcome.edge_coherence_low),
        ("high", outcome.edge_coherence_high),
    ):
        if not np.isfinite(edge_coh) or edge_coh <= residual_edge_threshold:
            continue
        contributor = select_contributor_to_thaw(win, outcome.fixed_peaks, edge_side)
        if contributor is None:
            # No contributor to blame on this edge -- record as a no-op event
            # so the diagnostic shows we tried; do not stop the bounded loop.
            event = ThawEvent(
                dependent_window_id=win.window_id,
                primary_window_id=-1,
                contributor_peak_index=-1,
                contributor_frequency_mhz=float("nan"),
                edge_side=edge_side,
                edge_coherence_before=float(edge_coh),
                edge_coherence_after=float(edge_coh),
                accepted=False,
                reason="no frozen contributor on the flagged edge side",
            )
        else:
            event = _perform_thaw(
                win,
                outcome,
                contributor,
                edge_side=edge_side,
                edge_coherence_before=float(edge_coh),
                outcomes=outcomes,
                sideband=sideband,
                acquisition_us=acquisition_us,
                tau0_us=tau0_us,
                fit_tau=fit_tau,
                residual_edge_threshold=residual_edge_threshold,
                residual_edge_m=residual_edge_m,
                shape=shape,
                spur_set=spur_set,
            )
        events.append(event)
        outcome.thaw_events.append(event)
    return events


def _perform_thaw(
    dep_window: FitWindow,
    dep_outcome: WindowOutcome,
    thawed: FrozenPeak,
    *,
    edge_side: str,
    edge_coherence_before: float,
    outcomes: dict[int, WindowOutcome],
    sideband: SidebandLike,
    acquisition_us: float,
    tau0_us: float,
    fit_tau: bool,
    residual_edge_threshold: float,
    residual_edge_m: int,
    shape: "PeakShape | str" = "lorentzian",
    spur_set: Optional[SpurSet] = None,
) -> ThawEvent:
    """Do the joint co-fit, decide accept/reject, and rebuild the outcomes."""
    primary_outcome = outcomes[thawed.primary_window_id]

    joint, thawed_idx_arr = local_thaw_cofit(
        dep_outcome,
        primary_outcome,
        thawed=thawed,
        sideband=sideband,
        tau0_us=tau0_us,
        acquisition_us=acquisition_us,
        fit_tau=fit_tau,
        shape=shape,
        spur_set=spur_set,
    )
    if not joint.success:
        return ThawEvent(
            dependent_window_id=dep_window.window_id,
            primary_window_id=thawed.primary_window_id,
            contributor_peak_index=thawed.peak_index,
            contributor_frequency_mhz=thawed.frequency_mhz,
            edge_side=edge_side,
            edge_coherence_before=edge_coherence_before,
            edge_coherence_after=float("nan"),
            accepted=False,
            reason="joint co-fit did not converge",
        )

    # Split joint.peaks back into primary and dependent (the thawed peak is
    # *inside* the primary list -- it was already a free peak there before).
    s = sideband_sign(sideband)
    primary_center = _window_center_mhz(primary_outcome)
    dep_center = _window_center_mhz(dep_outcome)
    n_primary_peaks = len(primary_outcome.fit.peaks)
    n_dep_peaks = len(dep_outcome.fit.peaks)
    expected = n_primary_peaks + n_dep_peaks
    if len(joint.peaks) != expected:
        return ThawEvent(
            dependent_window_id=dep_window.window_id,
            primary_window_id=thawed.primary_window_id,
            contributor_peak_index=thawed.peak_index,
            contributor_frequency_mhz=thawed.frequency_mhz,
            edge_side=edge_side,
            edge_coherence_before=edge_coherence_before,
            edge_coherence_after=float("nan"),
            accepted=False,
            reason="joint co-fit peak count mismatch",
        )

    thawed_idx = int(thawed_idx_arr[0])
    primary_peaks = list(joint.peaks[:n_primary_peaks])
    dep_peaks_in_primary = list(joint.peaks[n_primary_peaks:])
    thawed_peak_primary = primary_peaks[thawed_idx]

    # Remap the dependent frame: shift = s*(dep_center - primary_center).
    shift = s * (dep_center - primary_center)
    dep_peaks = [
        ModelPeak(pk.amplitude, pk.offset_mhz - shift, pk.phase)
        for pk in dep_peaks_in_primary
    ]
    thawed_in_dep = ModelPeak(
        thawed_peak_primary.amplitude,
        thawed_peak_primary.offset_mhz - shift,
        thawed_peak_primary.phase,
    )

    # Provisional dependent residual after a (hypothetical) install: free peaks
    # become dep_peaks + thawed_in_dep, the thawed contributor drops from the
    # frozen background, every other frozen contributor stays at the new tau.
    dep_other_peaks = [
        fp
        for fp in dep_outcome.fixed_peaks
        if not (
            fp.peak_index == thawed.peak_index
            and fp.primary_window_id == thawed.primary_window_id
        )
    ]
    provisional_dep_full = model_spectrum(
        dep_outcome.offset_grid_mhz,
        dep_peaks + [thawed_in_dep],
        joint.tau_us,
        acquisition_us,
        shape=joint.shape,
    ) + _frozen_subset_model(
        dep_outcome.offset_grid_mhz,
        dep_other_peaks,
        joint.tau_us,
        acquisition_us,
        shape=joint.shape,
    )
    provisional_dep_residual = dep_outcome.complex_spectrum - provisional_dep_full
    dep_low_after, dep_high_after = residual_edge_coherence(
        provisional_dep_residual, dep_outcome.rms_noise, band_m=residual_edge_m
    )
    edge_after = dep_low_after if edge_side == "low" else dep_high_after
    accepted = edge_after <= residual_edge_threshold

    if not accepted:
        return ThawEvent(
            dependent_window_id=dep_window.window_id,
            primary_window_id=thawed.primary_window_id,
            contributor_peak_index=thawed.peak_index,
            contributor_frequency_mhz=thawed.frequency_mhz,
            edge_side=edge_side,
            edge_coherence_before=edge_coherence_before,
            edge_coherence_after=float(edge_after),
            accepted=False,
            reason="co-fit did not lower the flagged-edge coherence below threshold",
        )

    # Accept: install the new fits. The primary's peak list already includes
    # the thawed line (the joint fit refined it), so just refresh its peaks;
    # the dependent gains the thawed line as a free peak and drops it from
    # the frozen contributors.
    _install_cofit_outcome(
        primary_outcome,
        new_peaks=primary_peaks,
        tau_us=joint.tau_us,
        acquisition_us=acquisition_us,
        residual_edge_m=residual_edge_m,
        cofit_was_tau_free=bool(joint.tau_was_fit),
    )
    _install_cofit_outcome(
        dep_outcome,
        new_peaks=dep_peaks + [thawed_in_dep],
        tau_us=joint.tau_us,
        acquisition_us=acquisition_us,
        residual_edge_m=residual_edge_m,
        drop_contributor=thawed,
        cofit_was_tau_free=bool(joint.tau_was_fit),
    )

    return ThawEvent(
        dependent_window_id=dep_window.window_id,
        primary_window_id=thawed.primary_window_id,
        contributor_peak_index=thawed.peak_index,
        contributor_frequency_mhz=thawed.frequency_mhz,
        edge_side=edge_side,
        edge_coherence_before=edge_coherence_before,
        edge_coherence_after=float(edge_after),
        accepted=True,
        reason="joint co-fit lowered the flagged-edge coherence below threshold",
    )


def _frozen_subset_model(
    grid: np.ndarray,
    fixed_peaks: Sequence[FrozenPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = "lorentzian",
) -> np.ndarray:
    """Convenience: re-evaluate a subset of FrozenPeaks at one tau."""
    if not fixed_peaks:
        zero: np.ndarray = np.zeros(grid.shape, dtype=np.complex128)
        return zero
    return model_spectrum(
        grid,
        [fp.model_peak for fp in fixed_peaks],
        tau_us,
        acquisition_us,
        shape=shape,
    )


def _install_cofit_outcome(
    outcome: WindowOutcome,
    *,
    new_peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    residual_edge_m: int,
    drop_contributor: Optional[FrozenPeak] = None,
    cofit_was_tau_free: bool = False,
) -> None:
    """Update a WindowOutcome in place after an accepted co-fit.

    Replaces the free-peak fit's peaks/tau and recomputes the full model,
    residual, and edge-coherence statistics; optionally drops a thawed
    contributor from ``fixed_peaks``. The conservative-fit audit trail and
    knockouts are preserved (they describe the original free-peak fit; the
    accepted-thaw record lives on the :class:`ThawEvent`).

    When the co-fit ran with tau free (``cofit_was_tau_free=True``), the
    persisted tau came from a tau-free LSQ -- promote ``tau_was_fit`` so
    downstream consumers see this window's tau as data-determined.
    """
    if drop_contributor is not None:
        outcome.fixed_peaks = [
            fp
            for fp in outcome.fixed_peaks
            if not (
                fp.peak_index == drop_contributor.peak_index
                and fp.primary_window_id == drop_contributor.primary_window_id
            )
        ]
    # Replace the free-peak fit's peaks/tau and reconstruct the full model.
    new_peak_list = [ModelPeak(p.amplitude, p.offset_mhz, p.phase) for p in new_peaks]
    outcome.fit.fit.peaks = new_peak_list
    outcome.fit.fit.tau_us = tau_us
    if cofit_was_tau_free:
        outcome.fit.fit.tau_was_fit = True
    # The thaw co-fit carries no baseline term; clear any prior baseline so
    # the installed fit's model and its persisted coefficients stay
    # consistent (the post-rescue baseline pass re-derives one if the
    # residual still warrants it).
    outcome.fit.fit.baseline_order = None
    outcome.fit.fit.baseline_coeffs = None
    outcome.fit.fit.baseline_offset_scale = None
    outcome.baseline_applied = False
    shape_resolved = outcome.fit.fit.shape
    free_model = model_spectrum(
        outcome.offset_grid_mhz,
        new_peak_list,
        tau_us,
        acquisition_us,
        shape=shape_resolved,
    )
    outcome.fit.fit.fitted_spectrum = free_model
    background = _frozen_subset_model(
        outcome.offset_grid_mhz,
        outcome.fixed_peaks,
        tau_us,
        acquisition_us,
        shape=shape_resolved,
    )
    outcome.background = background
    outcome.full_fitted_spectrum = free_model + background
    outcome.full_residual = outcome.complex_spectrum - outcome.full_fitted_spectrum
    outcome.fit.fit.residual = outcome.complex_spectrum - free_model - background
    low, high = residual_edge_coherence(
        outcome.full_residual, outcome.rms_noise, band_m=residual_edge_m
    )
    outcome.edge_coherence_low = low
    outcome.edge_coherence_high = high


def _apply_rescue_to_outcome(
    win: FitWindow,
    outcome: WindowOutcome,
    *,
    acquisition_us: float,
    tau0_us: float,
    residual_edge_m: int,
    conservative_kwargs: dict[str, Any],
    max_residual_rescue_rounds: int,
    rescue_kwargs: dict[str, Any],
    spur_set: Optional[SpurSet] = None,
    protected_offsets: Optional[Sequence[float]] = None,
    protected_tol_mhz: float = 0.0,
    forbidden_offsets: Optional[Sequence[float]] = None,
    forbidden_tol_mhz: float = 0.0,
) -> list[RescueEvent]:
    """Run the residual-rescue B-loop on a finished window and update its
    outcome in place. Returns the per-round :class:`RescueEvent` records.

    Operates on the *post-thaw* outcome: the rescue's notion of "what the
    initial fit missed" includes any contributor's line a thaw promoted to
    a free peak (which appears in the window's free-peak list after a
    successful thaw via :func:`_install_cofit_outcome`). The frozen
    contributor background is held fixed across the rescue chain -- the
    rescue addresses missed lines, not contributor renegotiation.

    The conservative-fit audit trail is preserved (the rescue is a
    separate phase, not a continuation of the conservative loop). The
    consolidated knockouts overwrite the original ones since the peak
    set has changed.
    """
    data_minus_bg = outcome.complex_spectrum - outcome.background
    spur_mask = getattr(outcome, "_spur_mask", None)
    if os.environ.get("FTMW_DEBUG_FRINGE_DIR"):
        ctx = dict(validation._fringe_window_ctx or {})
        ctx["bg_u"] = np.asarray(outcome.offset_grid_mhz, dtype=float)
        ctx["bg"] = np.asarray(outcome.background, dtype=np.complex128)
        validation._fringe_window_ctx = ctx
    # Same per-window sigma_eff skirt budget the conservative fit's gates used
    # (the rescue operates on the identical background-subtracted data).
    kappa_skirt = validation.DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT
    budget_extra = (
        float(kappa_skirt) * np.abs(outcome.background)
        if kappa_skirt is not None
        else None
    )
    if os.environ.get("FTMW_DEBUG_SKIRT"):
        _sig = np.asarray(outcome.rms_noise, dtype=float)
        _bud = 0.0 if budget_extra is None else float(np.median(budget_extra))
        print(
            f"[rescue-budget] w{outcome.window_id}: "
            f"med|data|={float(np.median(np.abs(outcome.complex_spectrum))):.3e} "
            f"med|bg|={float(np.median(np.abs(outcome.background))):.3e} "
            f"med|resid|={float(np.median(np.abs(data_minus_bg))):.3e} "
            f"med_sigma={float(np.median(_sig)):.3e} med_budget={_bud:.3e}",
            flush=True,
        )
    consolidated = rescue_and_consolidate(
        outcome.offset_grid_mhz,
        data_minus_bg,
        outcome.rms_noise,
        outcome.fit,
        tau0_us,
        acquisition_us,
        max_rescue_rounds=max_residual_rescue_rounds,
        conservative_kwargs=conservative_kwargs,
        shape=outcome.fit.fit.shape,
        spur_mask=spur_mask,
        gate_budget_extra=budget_extra,
        gate_background=outcome.background,
        protected_offsets=protected_offsets,
        protected_tol_mhz=protected_tol_mhz,
        forbidden_offsets=forbidden_offsets,
        forbidden_tol_mhz=forbidden_tol_mhz,
        **rescue_kwargs,
    )

    if any(r.accepted for r in consolidated.rounds):
        # Mutate the existing ConservativeFitResult so any reference to it
        # stays current (matches the _install_cofit_outcome pattern).
        outcome.fit.fit = consolidated.fit.fit
        outcome.fit.knockouts = consolidated.fit.knockouts
        # Re-evaluate the free model on the caller's grid -- fit_window's
        # internal arrays may be on a sorted version and we want bin-for-bin
        # alignment with outcome.offset_grid_mhz / outcome.complex_spectrum.
        free_model = model_spectrum(
            outcome.offset_grid_mhz,
            consolidated.fit.fit.peaks,
            consolidated.fit.fit.tau_us,
            acquisition_us,
            shape=consolidated.fit.fit.shape,
        ) + evaluate_baseline(consolidated.fit.fit, outcome.offset_grid_mhz)
        outcome.fit.fit.fitted_spectrum = free_model
        outcome.fit.fit.residual = data_minus_bg - free_model
        outcome.full_fitted_spectrum = free_model + outcome.background
        outcome.full_residual = outcome.complex_spectrum - outcome.full_fitted_spectrum
        low, high = residual_edge_coherence(
            outcome.full_residual, outcome.rms_noise, band_m=residual_edge_m
        )
        outcome.edge_coherence_low = low
        outcome.edge_coherence_high = high

    events: list[RescueEvent] = []
    for diag in consolidated.rounds:
        ev = RescueEvent(
            window_id=win.window_id,
            round_idx=diag.round_idx,
            n_initial_peaks=diag.n_initial_peaks,
            n_candidates=len(diag.rescue.candidates),
            n_rescue_added=diag.n_rescue_added,
            n_pruned_by_knockout=diag.n_pruned_total,
            n_pruned_rescue_origin=diag.n_pruned_rescue_origin,
            n_merged=diag.n_merged,
            chi2_before=diag.chi2_before,
            chi2_after=diag.chi2_after,
            tau_us_before=diag.tau_us_before,
            tau_us_after=diag.tau_us_after,
            accepted=diag.accepted,
            reason=diag.reason,
            candidates=list(diag.rescue.candidates),
        )
        events.append(ev)
        outcome.rescue_events.append(ev)
        if os.environ.get("FTMW_DEBUG_RESCUE"):
            cands = [round(c.frequency_mhz, 3) for c in diag.rescue.candidates]
            jchi = (
                f"{diag.joint_fit.chi_squared:.0f}"
                if diag.joint_fit is not None and diag.joint_fit.success
                else "n/a"
            )
            jtau = (
                f"{diag.joint_fit.tau_us:.2f}"
                if diag.joint_fit is not None and diag.joint_fit.success
                else "n/a"
            )
            print(
                f"[rescue] w{win.window_id} round {ev.round_idx}: "
                f"cands={cands} added={ev.n_rescue_added} "
                f"merged={ev.n_merged} pruned={ev.n_pruned_by_knockout} "
                f"(rescue-origin {ev.n_pruned_rescue_origin}) "
                f"chi2 {ev.chi2_before:.0f}->{ev.chi2_after:.0f} "
                f"joint chi2={jchi} tau={jtau} "
                f"tau {ev.tau_us_before:.2f}->{ev.tau_us_after:.2f} "
                f"accepted={ev.accepted} {ev.reason}",
                flush=True,
            )
    return events


def _smooth_residual_stat(
    offset_grid_mhz: np.ndarray,
    residual: np.ndarray,
    rms_noise: np.ndarray,
    order: int,
) -> float:
    """F-test numerator for an order-``p`` complex polynomial fit of a residual.

    Projects the complex ``residual`` onto a noise-weighted degree-``order``
    polynomial in the scaled offset and returns the chi-squared drop the smooth
    term explains, divided by its ``2(order+1)`` real degrees of freedom. Pure
    noise scores ~1 per dof; a smooth leakage pedestal scores far higher. This is
    the in-band counterpart to the edge-coherence trigger -- it catches a smooth
    pedestal ramping across the whole window, which the two-edge test misses.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    if u.size <= order + 1:
        return 0.0
    sig = np.asarray(rms_noise, dtype=float)
    if sig.ndim == 0:
        sig = np.full(u.size, float(sig))
    span = max(float(u.max() - u.min()), 1e-9) / 2.0
    us = (u - u.mean()) / span
    vander = np.vander(us, order + 1, increasing=True)
    w = 1.0 / np.maximum(sig, 1e-30)
    coef, *_ = np.linalg.lstsq(vander * w[:, None], residual * w, rcond=None)
    smooth = vander @ coef
    chi2_full = float(np.sum((np.abs(residual) / np.maximum(sig, 1e-30)) ** 2))
    chi2_resid = float(
        np.sum((np.abs(residual - smooth) / np.maximum(sig, 1e-30)) ** 2)
    )
    return (chi2_full - chi2_resid) / (2 * (order + 1))


_BASELINE_TAU_INPUT_KEYS = (
    "min_separation_factor",
    "max_decay_factor",
    "amp_max_headroom",
    "phase_penalty_lambda",
    "amp_penalty_lambda",
    "phase_penalty_cutoff_fwhm",
    "tau_penalty_lambda",
    "tau_penalty_n_sigma",
    "weak_window_snr_threshold",
    "fit_tau_min_snr",
    "tau_apodization_us",
    "tau_maj_us",
    "sigma_tau_us",
)


def _baseline_tau_inputs(conservative_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Subset of ``conservative_kwargs`` accepted by
    :func:`~ftmwpipeline.fitting.window_fit.derive_window_fit_constraints`, so the
    baseline's tau-free refit derives the same bounds / penalty as the primary
    per-window fit."""
    return {
        k: conservative_kwargs[k]
        for k in _BASELINE_TAU_INPUT_KEYS
        if k in conservative_kwargs
    }


def _apply_baseline_to_outcome(
    outcome: WindowOutcome,
    *,
    acquisition_us: float,
    residual_edge_m: int,
    baseline_order: int,
    baseline_edge_threshold: float,
    baseline_smooth_threshold: float,
    tau0_us: float,
    conservative_kwargs: dict[str, Any],
) -> bool:
    """Refit a window with a complex baseline when a coherent or smooth residual
    clears its trigger; update the outcome in place.

    A neighboring strong line's mismodeled leakage skirt -- and, on a dense
    ultra-high-SNR spectrum, the summed far-wings of the many lines the discrete
    frozen contributors cannot fully subtract -- leaves a systematic residual
    that inflates the dependent window's chi-squared and biases the lines sitting
    on it. The window's established lines are refit jointly with a low-order
    complex baseline ``B(u) = Σ_{k≤p}(a_k + i b_k)(u/u_s)^k`` (too smooth to
    represent a narrow line, so it can only soak up a broad wing or pedestal)
    when either trigger fires:

    * **edge coherence** -- ``max(edge_low, edge_high)`` on the residual exceeds
      ``baseline_edge_threshold`` (a coherent wing at a window edge); or
    * **smooth residual** -- an order-``p`` polynomial explains the residual at
      more than ``baseline_smooth_threshold`` chi-squared per added dof (a smooth
      in-band leakage pedestal the edge test misses).

    ``tau`` is re-freed and re-anchored at the band majority ``tau0_us`` for the
    joint refit: with the pedestal absorbed by the baseline, ``tau`` relaxes back
    from the collapsed value it took to soak the pedestal to its physical
    per-band value. The joint covariance prices the baseline's degrees of freedom
    into the reported per-line uncertainties.

    Returns ``True`` iff the baseline fired (triggered, converged, and did not
    worsen the data chi-squared); ``False`` leaves the outcome untouched.
    """
    inner = outcome.fit.fit
    if not inner.peaks:
        # The baseline prices its flexibility against the free lines; a window
        # carrying no free peaks (a pure frozen-background slice) has nothing
        # to fit it against, so it cannot fire.
        return False

    data_minus_bg = outcome.complex_spectrum - outcome.background
    u = outcome.offset_grid_mhz
    # The trigger residual must credit a baseline the fit already carries
    # (the early conservative-phase baseline), else the carried pedestal
    # re-reads as residual structure here.
    residual = (
        data_minus_bg
        - model_spectrum(
            u, inner.peaks, inner.tau_us, acquisition_us, shape=inner.shape
        )
        - evaluate_baseline(inner, u)
    )
    s_coh = max(outcome.edge_coherence_low, outcome.edge_coherence_high)
    edge_fire = np.isfinite(s_coh) and s_coh > baseline_edge_threshold
    smooth_stat = _smooth_residual_stat(u, residual, outcome.rms_noise, baseline_order)
    smooth_fire = smooth_stat > baseline_smooth_threshold
    # A fit already carrying a baseline (the early conservative-phase trigger
    # fired) always takes this joint refit: the residual credit above means
    # neither trigger re-fires, but the refit's value on a pedestal window is
    # the tau re-free -- the conservative phase ran tau against the pedestal
    # and the anchor below lets it relax to the band value. The chi-squared
    # acceptance guard below still applies.
    if os.environ.get("FTMW_DEBUG_PHASES"):
        print(
            f"[late-baseline] w{outcome.window_id}: s_coh={s_coh:.2f} "
            f"smooth={smooth_stat:.1f} edge_fire={edge_fire} "
            f"smooth_fire={smooth_fire} carried={inner.baseline_order}",
            flush=True,
        )
    if not (edge_fire or smooth_fire or inner.baseline_order is not None):
        return False

    # Re-free tau (anchored at the band majority) so it relaxes off the collapsed
    # value once the baseline carries the pedestal. The penalty / bound policy
    # mirrors the primary fit's via ``derive_window_fit_constraints``.
    constraints = derive_window_fit_constraints(
        data_minus_bg,
        outcome.rms_noise,
        tau0_us,
        acquisition_us,
        fit_tau=True,
        **_baseline_tau_inputs(conservative_kwargs),
    )
    refit_kwargs = dict(constraints.fit_kwargs_inner)
    refit_kwargs.setdefault("shape", inner.shape)
    spur_mask = getattr(outcome, "_spur_mask", None)
    refit = fit_window(
        u,
        data_minus_bg,
        outcome.rms_noise,
        [ModelPeak(p.amplitude, p.offset_mhz, p.phase) for p in inner.peaks],
        tau0_us,
        acquisition_us,
        spur_mask=spur_mask,
        baseline_order=baseline_order,
        **refit_kwargs,
    )
    if os.environ.get("FTMW_DEBUG_PHASES"):
        print(
            f"[late-baseline] w{outcome.window_id}: refit success={refit.success} "
            f"chi2 {inner.chi_squared:.0f} -> {refit.chi_squared:.0f} "
            f"tau {inner.tau_us:.2f} -> {refit.tau_us:.2f}",
            flush=True,
        )
    if not refit.success or refit.chi_squared > inner.chi_squared + 1e-9:
        return False

    # Install the joint fit. The baseline carries the pedestal and the re-freed
    # ``tau`` has relaxed to its physical value, so the re-fit ``tau`` / errors
    # are installed too. The fitted spectrum carries peaks + baseline; the
    # frozen background is added back for the full model.
    free_plus_baseline = refit.fitted_spectrum
    inner.peaks = refit.peaks
    inner.peak_errors = refit.peak_errors
    inner.covariance = refit.covariance
    inner.chi_squared = refit.chi_squared
    inner.cost = refit.cost
    inner.n_params = refit.n_params
    inner.n_data = refit.n_data
    inner.tau_us = refit.tau_us
    inner.tau_error = refit.tau_error
    inner.tau_was_fit = refit.tau_was_fit
    inner.fitted_spectrum = free_plus_baseline
    inner.residual = data_minus_bg - free_plus_baseline
    inner.baseline_order = refit.baseline_order
    inner.baseline_coeffs = refit.baseline_coeffs
    inner.baseline_offset_scale = refit.baseline_offset_scale

    outcome.full_fitted_spectrum = free_plus_baseline + outcome.background
    outcome.full_residual = outcome.complex_spectrum - outcome.full_fitted_spectrum
    low, high = residual_edge_coherence(
        outcome.full_residual, outcome.rms_noise, band_m=residual_edge_m
    )
    outcome.baseline_applied = True
    outcome.baseline_order = refit.baseline_order
    outcome.baseline_coeffs = refit.baseline_coeffs
    outcome.baseline_offset_scale = refit.baseline_offset_scale
    outcome.baseline_edge_coherence = float(s_coh)
    outcome.edge_coherence_low = low
    outcome.edge_coherence_high = high
    return True
