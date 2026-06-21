"""Post-fit doublet-alternative adjudication for Stage 5 fitting.

For every adjacent pair of fitted lines in a window's final table whose
separation falls within the sub-resolution trigger band, this module refits
the window with the pair collapsed to one peak and records two discriminating
statistics alongside the bookkeeping necessary for downstream consumers
(reports, candidate-revival ledger):

1. **ΔAICc / Δχ²** — the raw chi-squared and AICc difference between the
   production (doublet) fit and the merged (single-peak) alternative.  A
   large AICc advantage for the doublet is necessary but not sufficient
   evidence of a genuine second transition: under a high-SNR line, a
   sub-percent lineshape deficit is hundreds of σ per bin, so a spurious
   partner also wins raw χ².

2. **Orthogonal evidence score** — the merged residual projected onto the
   weak partner's template *after* projecting out the parent's shape-
   derivative subspace ``{h, dh/df, dh/dτ}``.  First-order lineshape / τ
   error lives in that subspace; a genuine second transition retains a
   component orthogonal to it.  The score is the nuisance-projected
   matched-filter delta-chi-squared (:func:`~.validation.line_evidence_escape`
   in raw noise currency).

The pass is **observation-only**: it records an alternative and statistics;
acceptance behavior is unchanged.  Persistence and consumption belong to
the pipeline wiring layer, not this module.

Public interface
----------------
:data:`DEFAULT_DOUBLET_K_RES`
    Default maximum separation in resolution elements.
:data:`DEFAULT_DOUBLET_R_MIN`
    Default minimum amplitude ratio for the trigger.
:class:`DoubletAdjudication`
    Per-pair record (fit-frame coordinates).
:func:`adjudicate_close_pairs`
    Entry point: enumerate qualifying pairs and compute records.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .peak_model import ModelPeak, PeakShape, model_spectrum
from .spur_detection import SpurMaskSpec
from .validation import (
    calculate_aicc,
    line_escape_nuisance_columns,
    line_escape_support_slice,
    line_evidence_escape,
)
from .window_fit import WindowFitResult, fit_window

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DOUBLET_K_RES",
    "DEFAULT_DOUBLET_R_MIN",
    "DoubletAdjudication",
    "adjudicate_close_pairs",
]

# Maximum separation (in resolution elements ``1/T_active`` MHz) between two
# adjacent fitted peaks that triggers the doublet-alternative adjudication.
# 1.5 resolution elements is the regime where raw-χ² cannot arbitrate a
# genuine second transition from a lineshape-floor absorber: the overlap
# between the two hypotheses' lineshapes is substantial and both hypotheses
# fit the data well at different parameter values.
DEFAULT_DOUBLET_K_RES: float = 1.5

# Minimum weak/strong amplitude ratio to trigger adjudication.  Below this
# the partner is at the skirt level and the existing cleanup layer owns it;
# the doublet-alternative pass is for pairs where both members have
# non-negligible amplitude.
DEFAULT_DOUBLET_R_MIN: float = 0.05


@dataclass
class DoubletAdjudication:
    """Per-pair doublet-alternative record (fit-frame coordinates).

    All offsets and parameters are in the same baseband fit frame as the
    production :class:`~.window_fit.WindowFitResult` they were derived from.
    ``pair_index_a`` / ``pair_index_b`` are indices into ``fit.peaks``
    (``a`` = lower offset, ``b`` = higher offset) so the caller can
    cross-reference into the production table.

    Attributes
    ----------
    pair_index_a, pair_index_b : int
        Indices into the production fit's ``peaks`` list.  ``a`` is the
        peak with the lower (or equal) ``offset_mhz``.
    offset_a_mhz, offset_b_mhz : float
        Baseband offsets of the two production peaks (MHz).
    amplitude_a, amplitude_b : float
        Amplitudes of the two production peaks.
    separation_res_elements : float
        ``|offset_b - offset_a| * acquisition_us`` — the separation in
        units of the active-FT Fourier resolution element
        ``1/T_active = 1/acquisition_us`` MHz.
    amp_ratio : float
        ``min(amplitude_a, amplitude_b) / max(amplitude_a, amplitude_b)``
        — weak-to-strong ratio in ``[0, 1]``.
    chi2r_production : float
        Reduced chi-squared of the production fit.
    chi2r_merged : float
        Reduced chi-squared of the merged alternative fit.
        ``nan`` if the merged refit failed.
    delta_chi2_raw : float
        ``chi_squared(merged) - chi_squared(production)``; positive when
        the production doublet fit is better.  ``nan`` on refit failure.
    delta_aicc : float
        ``AICc(merged) - AICc(production)`` evaluated with each fit's own
        ``n_data`` / ``n_params``.  Positive values favor the production
        doublet.  ``nan`` on refit failure or degenerate AICc.
    merged_offset_mhz : float
        Fitted offset of the merged peak in the refit result.
        ``nan`` on refit failure.
    merged_amplitude : float
        Fitted amplitude of the merged peak.  ``nan`` on refit failure.
    merged_phase : float
        Fitted phase of the merged peak.  ``nan`` on refit failure.
    merged_tau_us : float
        Shared tau of the merged refit.  ``nan`` on refit failure.
    merged_success : bool
        Whether the merged refit solver reported convergence.
    orth_evidence_delta_chi2 : float
        Nuisance-projected matched-filter delta-chi-squared of the weak
        partner template on the merged residual.  Large values indicate
        that the merged fit leaves behind structure that the weak
        partner's lineshape alone (not lineshape-error of the parent)
        explains — evidence for a genuine second transition.
        0.0 when the support slice is degenerate; ``nan`` on refit
        failure.
    orth_evidence_n_params : int
        Number of peak parameters in the template (always 3: amplitude,
        offset, phase).
    support_bins : int
        Length of the support slice used for the orthogonal-evidence
        computation (0 when the slice was unavailable).
    """

    pair_index_a: int
    pair_index_b: int
    offset_a_mhz: float
    offset_b_mhz: float
    amplitude_a: float
    amplitude_b: float
    separation_res_elements: float
    amp_ratio: float
    chi2r_production: float
    chi2r_merged: float
    delta_chi2_raw: float
    delta_aicc: float
    merged_offset_mhz: float
    merged_amplitude: float
    merged_phase: float
    merged_tau_us: float
    merged_success: bool
    orth_evidence_delta_chi2: float
    orth_evidence_n_params: int
    support_bins: int


def _nan_adjudication(
    idx_a: int,
    idx_b: int,
    pk_a: ModelPeak,
    pk_b: ModelPeak,
    separation_res: float,
    amp_ratio: float,
    chi2r_production: float,
) -> DoubletAdjudication:
    """Failure-path record: refit did not succeed or stats are undefined."""
    nan = float("nan")
    return DoubletAdjudication(
        pair_index_a=idx_a,
        pair_index_b=idx_b,
        offset_a_mhz=pk_a.offset_mhz,
        offset_b_mhz=pk_b.offset_mhz,
        amplitude_a=pk_a.amplitude,
        amplitude_b=pk_b.amplitude,
        separation_res_elements=separation_res,
        amp_ratio=amp_ratio,
        chi2r_production=chi2r_production,
        chi2r_merged=nan,
        delta_chi2_raw=nan,
        delta_aicc=nan,
        merged_offset_mhz=nan,
        merged_amplitude=nan,
        merged_phase=nan,
        merged_tau_us=nan,
        merged_success=False,
        orth_evidence_delta_chi2=nan,
        orth_evidence_n_params=3,
        support_bins=0,
    )


def _find_merged_peak(
    refit: WindowFitResult, seed_offset: float
) -> Optional[ModelPeak]:
    """Return the fitted peak nearest ``seed_offset`` in the refit result."""
    if not refit.peaks:
        return None
    return min(refit.peaks, key=lambda p: abs(p.offset_mhz - seed_offset))


def adjudicate_close_pairs(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: WindowFitResult,
    *,
    acquisition_us: float,
    shape: PeakShape | str,
    k_res: float = DEFAULT_DOUBLET_K_RES,
    r_min: float = DEFAULT_DOUBLET_R_MIN,
    frozen_background: Optional[np.ndarray] = None,
    spur_mask: Optional[SpurMaskSpec] = None,
    refit_kwargs: Optional[dict[str, Any]] = None,
) -> List[DoubletAdjudication]:
    """Enumerate close pairs in ``fit`` and compute doublet-alternative records.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid for the window (MHz), 1-D, ascending.
    complex_spectrum : np.ndarray
        Complex window data on ``offset_grid_mhz`` **minus** the frozen
        background — the same ``z`` the production free-peak fit was run on.
    rms_noise : np.ndarray
        Per-bin complex noise RMS ``sigma`` (canonical Stage 2 noise).
    fit : WindowFitResult
        The production free-peak fit result for this window.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds, ``> 0``).  The
        resolution element is ``1/acquisition_us`` MHz.
    shape : PeakShape or str
        Line-shape selector, matching the production fit.
    k_res : float, default :data:`DEFAULT_DOUBLET_K_RES`
        Maximum separation in resolution elements to trigger adjudication.
    r_min : float, default :data:`DEFAULT_DOUBLET_R_MIN`
        Minimum weak/strong amplitude ratio to trigger adjudication.
    frozen_background : np.ndarray, optional
        Frozen contributor background on ``offset_grid_mhz`` (i.e. the
        subtracted skirt that was removed from the data before fitting).
        Used as a nuisance column in the orthogonal-evidence computation.
        ``None`` omits background nuisance columns.
    spur_mask : SpurMaskSpec, optional
        Forwarded verbatim to the merged refit.
    refit_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`~.window_fit.fit_window`
        for the merged refit.  These should reproduce the production fit's
        penalty and constraint conditions.  ``fit_tau`` and ``tau0_us``
        are managed internally and cannot be overridden via this dict.

    Returns
    -------
    list of DoubletAdjudication
        One record per qualifying adjacent pair, in the order they appear
        in ``fit.peaks`` (sorted by offset).  Empty when ``fit.n_peaks < 2``
        or no pair qualifies.

    Notes
    -----
    *Trigger*: pairs are evaluated on ``fit.peaks`` **sorted by offset**.
    Adjacent pairs with ``|Δoffset| * acquisition_us <= k_res`` **and**
    ``min(A)/max(A) >= r_min`` qualify.  Chains of 3+ peaks within the
    floor produce one record per adjacent qualifying pair.

    *Merged refit*: the pair is replaced by one seed at the amplitude-
    weighted centroid with amplitude ``max(A_a, A_b)`` and the stronger
    member's phase.  All other peaks and the baseline policy (if the
    production fit carried one) are forwarded unchanged.  The tau policy
    follows ``fit.fit_tau`` seeded at ``fit.tau_us`` — the alternative is
    not handicapped by forcing tau fixed.

    *Orthogonal evidence*: computed on the merged refit's residual using
    the weak partner's template (at production parameters) as the signal
    and the merged-peak (parent) shape derivatives plus optional frozen
    background as nuisance columns — exactly the matched-filter machinery
    of :func:`~.validation.line_evidence_escape`.  A large score means the
    merged residual retains a component the parent's shape-error subspace
    cannot explain, which is positive evidence for a genuine second line.
    """
    shape_resolved = PeakShape.coerce(shape)
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))

    if fit.n_peaks < 2:
        return []

    # Sort peaks by offset; track original indices for the record.
    sorted_items: list[tuple[int, ModelPeak]] = sorted(
        enumerate(fit.peaks), key=lambda t: t[1].offset_mhz
    )

    if acquisition_us <= 0.0:
        return []
    res_element_mhz = 1.0 / acquisition_us
    sep_threshold_mhz = k_res * res_element_mhz

    # Build the refit kwargs, guarding against caller overrides of tau policy.
    base_kwargs: dict[str, Any] = dict(refit_kwargs or {})
    # tau policy: follow the production window's decision (fit.fit_tau),
    # seeded at the production tau.  Pop any caller-supplied overrides so the
    # alternative is not handicapped by an unintended fixed-tau constraint.
    base_kwargs.pop("fit_tau", None)
    base_kwargs.pop("tau0_us", None)
    # shape is an explicit parameter of this function; a caller-supplied
    # refit_kwargs entry (e.g. a forwarded fit_kwargs_inner bag) must not
    # collide with the explicit keyword in the fit_window call below.
    base_kwargs.pop("shape", None)
    # Forward baseline if the production fit had one.
    if fit.baseline_order is not None and "baseline_order" not in base_kwargs:
        base_kwargs["baseline_order"] = fit.baseline_order
        if fit.baseline_offset_scale is not None:
            base_kwargs.setdefault("baseline_offset_scale", fit.baseline_offset_scale)

    results: List[DoubletAdjudication] = []

    for step in range(len(sorted_items) - 1):
        orig_a, pk_a = sorted_items[step]
        orig_b, pk_b = sorted_items[step + 1]

        sep_mhz = abs(pk_b.offset_mhz - pk_a.offset_mhz)
        if sep_mhz > sep_threshold_mhz:
            continue

        a_amp = float(pk_a.amplitude)
        b_amp = float(pk_b.amplitude)
        max_amp = max(a_amp, b_amp)
        if max_amp <= 0.0:
            continue
        amp_ratio = min(a_amp, b_amp) / max_amp
        if amp_ratio < r_min:
            continue

        separation_res = sep_mhz * acquisition_us
        chi2r_prod = float(fit.reduced_chi2)

        # --- Merged refit seed -------------------------------------------
        # Amplitude-weighted centroid offset.
        total_amp = a_amp + b_amp
        seed_offset = (
            (a_amp * pk_a.offset_mhz + b_amp * pk_b.offset_mhz) / total_amp
            if total_amp > 0.0
            else 0.5 * (pk_a.offset_mhz + pk_b.offset_mhz)
        )
        # Seed amplitude at the stronger member; phase from the stronger.
        if a_amp >= b_amp:
            seed_amp = a_amp
            seed_phase = pk_a.phase
        else:
            seed_amp = b_amp
            seed_phase = pk_b.phase

        merged_seed = ModelPeak(
            amplitude=seed_amp,
            offset_mhz=seed_offset,
            phase=seed_phase,
        )

        # All other peaks (those not in the pair) are passed through unchanged.
        other_peaks: list[ModelPeak] = [
            pk for i, pk in enumerate(fit.peaks) if i != orig_a and i != orig_b
        ]
        merged_initial = other_peaks + [merged_seed]

        # ---- Run the merged refit ----------------------------------------
        try:
            refit = fit_window(
                u,
                z,
                sigma,
                merged_initial,
                float(fit.tau_us),
                acquisition_us,
                fit_tau=bool(fit.fit_tau),
                spur_mask=spur_mask,
                shape=shape_resolved,
                **base_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "doublet merged refit raised on pair (%.4f, %.4f) MHz: %r",
                pk_a.offset_mhz,
                pk_b.offset_mhz,
                exc,
            )
            results.append(
                _nan_adjudication(
                    orig_a, orig_b, pk_a, pk_b, separation_res, amp_ratio, chi2r_prod
                )
            )
            continue

        if not refit.success:
            results.append(
                DoubletAdjudication(
                    pair_index_a=orig_a,
                    pair_index_b=orig_b,
                    offset_a_mhz=pk_a.offset_mhz,
                    offset_b_mhz=pk_b.offset_mhz,
                    amplitude_a=pk_a.amplitude,
                    amplitude_b=pk_b.amplitude,
                    separation_res_elements=separation_res,
                    amp_ratio=amp_ratio,
                    chi2r_production=chi2r_prod,
                    chi2r_merged=float("nan"),
                    delta_chi2_raw=float("nan"),
                    delta_aicc=float("nan"),
                    merged_offset_mhz=float("nan"),
                    merged_amplitude=float("nan"),
                    merged_phase=float("nan"),
                    merged_tau_us=float("nan"),
                    merged_success=False,
                    orth_evidence_delta_chi2=float("nan"),
                    orth_evidence_n_params=3,
                    support_bins=0,
                )
            )
            continue

        # --- Fill statistics ----------------------------------------------
        chi2r_merged = float(refit.reduced_chi2)
        delta_chi2_raw = float(refit.chi_squared) - float(fit.chi_squared)

        # AICc: use each fit's own (n_data, n_params) pair.  n_data is 2*M
        # (stacked Re/Im), but may differ between fits if spur_mask removes
        # bins differently — use what the fits report directly.
        aicc_prod = calculate_aicc(
            float(fit.chi_squared),
            fit.n_params,
            float(fit.n_data),
        )
        aicc_merged = calculate_aicc(
            float(refit.chi_squared),
            refit.n_params,
            float(refit.n_data),
        )
        if math.isfinite(aicc_merged) and math.isfinite(aicc_prod):
            delta_aicc = aicc_merged - aicc_prod
        else:
            delta_aicc = float("nan")

        # Identify the merged peak as the one nearest the seed offset.
        merged_pk = _find_merged_peak(refit, seed_offset)
        if merged_pk is None:
            merged_offset = float("nan")
            merged_amplitude = float("nan")
            merged_phase = float("nan")
        else:
            merged_offset = float(merged_pk.offset_mhz)
            merged_amplitude = float(merged_pk.amplitude)
            merged_phase = float(merged_pk.phase)

        # --- Orthogonal evidence on the merged residual -------------------
        # Identify the weak partner (the member with the lower amplitude).
        if a_amp <= b_amp:
            weak_pk = pk_a
        else:
            weak_pk = pk_b

        # Template: the weak partner at its production parameters, evaluated
        # at the production tau (not the refit tau, which reflects the merged
        # model's best tau).
        template = model_spectrum(
            u, [weak_pk], float(fit.tau_us), acquisition_us, shape=shape_resolved
        )

        # Nuisance columns: parent (merged peak) shape derivatives + optional
        # frozen background.  Use the merged-refit tau so the nuisance spans
        # the actual lineshape freedom of the merged model.
        nuisance = line_escape_nuisance_columns(
            u,
            list(refit.peaks),
            float(refit.tau_us),
            acquisition_us,
            shape=shape_resolved,
            background=frozen_background,
        )

        evidence = np.asarray(refit.residual, dtype=np.complex128)
        support_sl = line_escape_support_slice(template)
        support_bins = (
            (support_sl.stop - support_sl.start) if support_sl is not None else 0
        )

        try:
            _fires, orth_delta = line_evidence_escape(
                evidence,
                sigma,
                template,
                nuisance,
                n_params_peak=3,
            )
        except Exception:  # noqa: BLE001
            orth_delta = 0.0
            support_bins = 0

        results.append(
            DoubletAdjudication(
                pair_index_a=orig_a,
                pair_index_b=orig_b,
                offset_a_mhz=pk_a.offset_mhz,
                offset_b_mhz=pk_b.offset_mhz,
                amplitude_a=pk_a.amplitude,
                amplitude_b=pk_b.amplitude,
                separation_res_elements=separation_res,
                amp_ratio=amp_ratio,
                chi2r_production=chi2r_prod,
                chi2r_merged=chi2r_merged,
                delta_chi2_raw=delta_chi2_raw,
                delta_aicc=delta_aicc,
                merged_offset_mhz=merged_offset,
                merged_amplitude=merged_amplitude,
                merged_phase=merged_phase,
                merged_tau_us=float(refit.tau_us),
                merged_success=True,
                orth_evidence_delta_chi2=float(orth_delta),
                orth_evidence_n_params=3,
                support_bins=support_bins,
            )
        )

    return results
