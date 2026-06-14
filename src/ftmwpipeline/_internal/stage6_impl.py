"""
Shared implementation for Stage 6: review, candidate ledger, and user-directed
single-window refit.

Pass 1 — candidate ledger derivation (``review show``/``--candidates``) and
the single-window refit engine (``review edit``).

The ledger is a pure function of the already-persisted Stage 5 audit trail
(``FittingResult.audit_trail``) and rescue events
(``FittingResult.rescue_events``).  It is derived on demand; no re-fitting
and no writes to the Stage 5 group.

The refit engine (``refit_window_impl``) re-fits a single window using the
production NLS primitives, starting from the persisted peaks as seeds.  It
supports add/remove edits with protected/forbidden immunity so cleanup and
rescue cannot undo human decisions.

Wrapped identically by the CLI, Pipeline class, and functional API.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..core.data_structures import (
    AuditStep,
    FittedPeak,
    FittingResult,
    LedgerCandidate,
    RescueCandidateInfo,
    RescueRoundInfo,
    Sideband,
    SpectrumFit,
)
from ..fitting.peak_model import ModelPeak
from ..fitting.peak_model import molecular_frequency as _molecular_frequency
from ..fitting.peak_model import sideband_sign
from ..io.fitting_serialization import load_spectrum_fit_from_hdf5
from .stage0_impl import load_fid_from_pipeline_impl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display-bar defaults
# ---------------------------------------------------------------------------

# Candidates below this residual / rescue SNR bar are hidden by default. The
# bar is a *display* threshold only -- it has no effect on the fit. The ledger
# is a per-window drill-down (what the fit considered and dropped in a window
# under review), so the default is set to keep the per-window count modest on
# the densest fixtures while preserving genuinely-marginal lines well above it.
DEFAULT_DISPLAY_BAR: float = 4.0

# If a candidate's evidence is within this factor of the accept gate it
# passes the bar even when its raw SNR is below DEFAULT_DISPLAY_BAR.
_NEAR_GATE_FACTOR: float = 10.0

# Deduplicate candidates whose molecular frequencies are within this window.
_DEDUP_TOL_MHZ: float = 0.02  # 20 kHz; roughly half an active-FT bin at 13 µs


# ---------------------------------------------------------------------------
# Revivable decision labels from the conservative add-loop
# ---------------------------------------------------------------------------

_REVIVABLE_DECISIONS = frozenset({"reject", "tentative"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sideband_from_value(value: Union[str, Sideband]) -> Sideband:
    if isinstance(value, Sideband):
        return value
    key = str(value).strip().lower()
    if key in ("lower", "lsb"):
        return Sideband.LOWER
    if key in ("upper", "usb"):
        return Sideband.UPPER
    raise ValueError(f"unknown sideband: {value!r}")


def _to_molecular(offset_mhz: float, center_mhz: float, sideband: Sideband) -> float:
    """Convert a single baseband offset to molecular MHz."""
    arr = np.array([offset_mhz])
    result = _molecular_frequency(arr, center_mhz, sideband)
    return float(result[0])


def _window_center(fit: FittingResult) -> Optional[float]:
    """Return the molecular centre of ``fit``'s window, or ``None``."""
    if fit.window is not None and fit.window.freq_range is not None:
        lo, hi = fit.window.freq_range
        return (lo + hi) / 2.0
    return None


# ---------------------------------------------------------------------------
# Candidate-extraction helpers (one per source type)
# ---------------------------------------------------------------------------


def _audit_step_candidates(
    audit_trail: List[AuditStep],
    center_mhz: float,
    sideband: Sideband,
) -> List[Dict]:
    """Extract revivable candidates from the add-loop audit trail.

    Each revivable step (decision in ``{"reject", "tentative"}``) becomes one
    raw candidate dict with keys: offset_mhz, freq_mhz, evidence, kind,
    reason, site, amplitude.
    """
    out: List[Dict] = []
    for step in audit_trail:
        if step.decision not in _REVIVABLE_DECISIONS:
            continue

        offset = step.candidate_offset_mhz
        freq = _to_molecular(offset, center_mhz, sideband)

        # Evidence: prefer AICc-delta (gate stat), fall back to F-test p.
        if not math.isnan(step.aicc_delta):
            # The add-loop rejected this candidate because aicc_delta >= 0
            # (the K+1 model was not strictly preferred over K).  A *marginal*
            # reject has aicc_delta close to 0 -- the gate was a near-miss and
            # the candidate is potentially revivable.  A *decisive* reject has a
            # large positive aicc_delta -- re-fitting with a user hint is
            # unlikely to change the verdict.  Expose this as ``kind="aicc_delta"``
            # with ``evidence = aicc_delta`` so ``_passes_bar`` can apply the
            # near-gate criterion (small = interesting, large = filter out).
            evidence = step.aicc_delta
            kind = "aicc_delta"
        elif not math.isnan(step.p_value):
            # Rejected by separation / blend-split / nan-AICc path; use F-test p.
            evidence = step.p_value
            kind = "f_p"
        else:
            evidence = abs(step.chi2_before - step.chi2_after)
            kind = "delta_chi2"

        out.append(
            {
                "offset_mhz": offset,
                "freq_mhz": freq,
                "evidence": evidence,
                "kind": kind,
                "reason": step.reason or step.decision,
                "site": f"add-loop:{step.decision}",
                "amplitude": None,  # not available from audit trail
            }
        )
    return out


def _rescue_round_candidates(
    rescue_events: List[RescueRoundInfo],
    center_mhz: float,
    sideband: Sideband,
) -> List[Dict]:
    """Extract candidates from all rescue rounds for a window."""
    out: List[Dict] = []
    for rnd in rescue_events:
        for cand in rnd.candidates:
            offset = cand.frequency_mhz  # already baseband offset
            freq = _to_molecular(offset, center_mhz, sideband)
            out.append(
                {
                    "offset_mhz": offset,
                    "freq_mhz": freq,
                    "evidence": cand.snr,
                    "kind": "residual_snr",
                    "reason": "rescue-candidate",
                    "site": f"rescue-round:{rnd.round_idx}",
                    "amplitude": cand.magnitude,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Deduplication and bar filter
# ---------------------------------------------------------------------------


def _dedup_and_merge(raw: List[Dict], tol_mhz: float) -> List[Dict]:
    """Merge raw candidates within ``tol_mhz`` of each other.

    Within a tolerance window, keep the candidate with the highest evidence
    and accumulate reasons / sites from all members.
    """
    if not raw:
        return []

    # Sort by molecular frequency for sequential scan.
    sorted_raw = sorted(raw, key=lambda c: c["freq_mhz"])

    groups: List[List[Dict]] = []
    current: List[Dict] = [sorted_raw[0]]

    for item in sorted_raw[1:]:
        if abs(item["freq_mhz"] - current[-1]["freq_mhz"]) <= tol_mhz:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    merged: List[Dict] = []
    for group in groups:
        # Best evidence: prefer residual_snr kind (most interpretable).  For
        # aicc_delta kind, smaller is better (more marginal = more revivable);
        # for all others, larger is better.  Sort priority: residual_snr first,
        # then aicc_delta ascending, then delta_chi2/f_p descending.
        def _evidence_key(c: Dict) -> tuple:
            k = str(c["kind"])
            ev = float(c["evidence"])
            if k == "residual_snr":
                return (0, -ev)  # highest SNR first
            if k == "aicc_delta":
                return (1, ev)  # smallest delta first (most marginal)
            return (2, -ev)  # largest chi2/fp first

        best = min(group, key=_evidence_key)

        reasons = list(dict.fromkeys(c["reason"] for c in group if c["reason"]))
        sites = list(dict.fromkeys(c["site"] for c in group))
        amplitude = next(
            (c["amplitude"] for c in group if c["amplitude"] is not None), None
        )

        merged.append(
            {
                "offset_mhz": best["offset_mhz"],
                "freq_mhz": best["freq_mhz"],
                "evidence": best["evidence"],
                "kind": best["kind"],
                "reason": reasons,
                "sites": sites,
                "amplitude": amplitude,
            }
        )

    return merged


def _passes_bar(candidate: Dict, bar: float) -> bool:
    """Return True if the candidate clears the display bar."""
    ev: float = float(candidate["evidence"])
    kind: str = str(candidate["kind"])

    if kind == "residual_snr":
        # SNR >= bar passes directly.
        return ev >= bar

    if kind == "aicc_delta":
        # A ``tentative`` decision is the add-loop's *explicit* "held as a
        # marginal near-miss" flag (the patience mechanism) -- revivable by
        # definition, so it surfaces regardless of the AICc delta magnitude.
        # A ``reject`` is surfaced only when it was a near-gate miss: ``ev`` is
        # the raw positive AICc delta (>= 0 for a rejected K+1 model); a
        # *marginal* reject has a small delta, a *decisive* reject a large one.
        # Pass rejects whose delta is within ``_NEAR_GATE_FACTOR`` of the gate
        # value (0) -- the "within an order of magnitude of the bar" criterion
        # from §D of the planning doc.
        sites = candidate.get("sites") or []
        if any("tentative" in str(s) for s in sites):
            return True
        return ev <= _NEAR_GATE_FACTOR

    # For delta_chi2 (chi2-difference fallback): pass when evidence is large.
    if kind == "delta_chi2":
        return ev >= bar

    # f_p: smaller p-value is stronger; pass when p <= 1/bar (heuristic).
    if kind == "f_p":
        return ev <= (1.0 / bar) if bar > 0 else True

    return True


# ---------------------------------------------------------------------------
# Public derivation API
# ---------------------------------------------------------------------------


def derive_candidate_ledger(
    fitting_result: FittingResult,
    *,
    center_mhz: float,
    sideband: Sideband,
    bar: float = DEFAULT_DISPLAY_BAR,
) -> List[LedgerCandidate]:
    """Derive the candidate ledger for one fit window.

    Walks ``fitting_result.audit_trail`` and ``fitting_result.rescue_events``,
    converts baseband offsets to molecular MHz, deduplicates within
    ``_DEDUP_TOL_MHZ``, applies the display ``bar``, and returns a list of
    :class:`~ftmwpipeline.core.data_structures.LedgerCandidate` sorted by
    molecular frequency.

    Parameters
    ----------
    fitting_result :
        The per-window :class:`~ftmwpipeline.core.data_structures.FittingResult`.
    center_mhz :
        Molecular centre of the fit window (midpoint of its ``freq_range``).
    sideband :
        Pipeline sideband (``Sideband.UPPER`` or ``Sideband.LOWER``).
    bar :
        Display SNR / evidence bar.  Candidates below it are dropped.

    Returns
    -------
    list of LedgerCandidate
        Sorted by ``frequency_mhz``.
    """
    window_id: int = (
        fitting_result.window_id if fitting_result.window_id is not None else -1
    )

    raw: List[Dict] = []
    raw.extend(_audit_step_candidates(fitting_result.audit_trail, center_mhz, sideband))
    raw.extend(
        _rescue_round_candidates(fitting_result.rescue_events, center_mhz, sideband)
    )

    merged = _dedup_and_merge(raw, _DEDUP_TOL_MHZ)

    # Drop candidates that coincide with an installed fitted peak.  The rescue
    # round records every *detected* candidate, including those the conservative
    # sub-fit then accepted -- those are now real peaks in the line list and are
    # not revivable.  A candidate within a dedup tolerance of any fitted peak is
    # the same sub-resolution feature, so subtract it.
    fitted_freqs = np.array(
        [p.frequency_mhz for p in fitting_result.fitted_peaks], dtype=float
    )
    if fitted_freqs.size:
        not_installed = [
            c
            for c in merged
            if np.min(np.abs(fitted_freqs - c["freq_mhz"])) > _DEDUP_TOL_MHZ
        ]
    else:
        not_installed = merged

    filtered = [c for c in not_installed if _passes_bar(c, bar)]

    candidates: List[LedgerCandidate] = []
    for c in sorted(filtered, key=lambda x: x["freq_mhz"]):
        candidates.append(
            LedgerCandidate(
                frequency_mhz=c["freq_mhz"],
                seed_offset_mhz=c["offset_mhz"],
                seed_amplitude=c["amplitude"],
                best_evidence=c["evidence"],
                evidence_kind=c["kind"],
                reasons=c["reason"],
                decision_sites=c["sites"],
                window_id=window_id,
            )
        )

    return candidates


def get_candidate_ledger_impl(
    file_path: Union[Path, str],
    window_id: Optional[int] = None,
    *,
    bar: float = DEFAULT_DISPLAY_BAR,
) -> List[LedgerCandidate]:
    """Load Stage 5 fit from ``file_path`` and derive the candidate ledger.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    window_id :
        When given, return candidates for that window only.  ``None`` returns
        candidates across all windows.
    bar :
        Display bar passed to :func:`derive_candidate_ledger`.

    Returns
    -------
    list of LedgerCandidate
        Combined across all (or the selected) window(s), sorted by
        ``frequency_mhz``.

    Raises
    ------
    ValueError
        When Stage 5 has not been run yet (no ``stage5_fitting`` group).
    KeyError
        When ``window_id`` is given but not found in the fit.
    """
    path = str(file_path)

    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])

    fid = load_fid_from_pipeline_impl(path)
    sideband = _sideband_from_value(fid.sideband)

    window_fits = spectrum_fit.window_fits
    if window_id is not None:
        window_fits = [wf for wf in window_fits if wf.window_id == window_id]
        if not window_fits:
            raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")

    all_candidates: List[LedgerCandidate] = []
    for wf in window_fits:
        center = _window_center(wf)
        if center is None:
            logger.warning(
                "window_id=%s has no freq_range; skipping ledger derivation",
                wf.window_id,
            )
            continue
        all_candidates.extend(
            derive_candidate_ledger(
                wf,
                center_mhz=center,
                sideband=sideband,
                bar=bar,
            )
        )

    return sorted(all_candidates, key=lambda c: c.frequency_mhz)


# ---------------------------------------------------------------------------
# Single-window refit result
# ---------------------------------------------------------------------------


@dataclass
class RefitWindowResult:
    """Outcome of a user-directed single-window refit.

    Attributes
    ----------
    window_id : int
        The window that was refitted.
    n_peaks_before : int
        Number of fitted peaks in the window before the refit.
    n_peaks_after : int
        Number of fitted peaks after the refit.
    chi2r_before : float
        Reduced chi-squared before the refit.
    chi2r_after : float
        Reduced chi-squared after the refit.
    fitted_peaks : list of FittedPeak
        The new per-window fitted peaks (already persisted).
    """

    window_id: int
    n_peaks_before: int
    n_peaks_after: int
    chi2r_before: float
    chi2r_after: float
    fitted_peaks: List[FittedPeak] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-window refit engine
# ---------------------------------------------------------------------------

# Tolerance for matching add/remove frequency requests to fitted peaks or
# ledger candidates (in MHz).  Half the dedup tolerance (10 kHz) is tight
# enough to snap unambiguously to one peak while forgiving coarse user input.
_REFIT_SNAP_TOL_MHZ: float = 0.05


def _parse_complex_amplitude(value: object) -> complex:
    """Parse a complex amplitude stored as ``str(complex)`` in JSON.

    ``result_conversion.py`` stores ``frozen.model_peak.amplitude`` (a real
    float) via ``json.dumps(..., default=str)``, which calls ``repr(v)`` on
    non-serialisable values.  For a real float the repr is just the float
    string; for an accidentally-complex value it would be ``"(a+bj)"``.
    Both cases are handled here to cover legacy files.
    """
    if isinstance(value, (int, float)):
        return complex(float(value))
    if isinstance(value, complex):
        return value
    # Try eval on the string repr (safe: only complex/float literals enter here).
    try:
        return complex(float(str(value)))
    except (ValueError, TypeError):
        try:
            return complex(str(value))
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot parse complex amplitude from persisted value {value!r}"
            )


def _reconstruct_frozen_peaks(
    fixed_parameters: Dict[str, Dict],
    center_mhz: float,
    sideband: Sideband,
) -> List:  # list of FrozenPeak-like namedtuples from plan_execution
    """Rebuild :class:`~ftmwpipeline.fitting.plan_execution.FrozenPeak` objects
    from the persisted ``fixed_parameters`` dict on a :class:`FittingResult`.

    The persisted format for each entry (key ``frozen_peak_<N>``) is::

        {
            "peak_index":        int,
            "primary_window_id": int,
            "frequency_mhz":     float,   # molecular frequency
            "amplitude":         float,   # model_peak.amplitude (real)
            "phase":             float,   # model_peak.phase (radians)
            "freeze_eligible":   bool,
        }

    The offset in the dependent window's baseband frame is derived from the
    molecular frequency and the window centre (same convention as
    :func:`~ftmwpipeline.fitting.plan_execution.evaluate_fixed_contributor`).
    """
    from ..fitting.plan_execution import FrozenPeak

    frozen: List[FrozenPeak] = []
    s = sideband_sign(sideband)
    for key, entry in fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        freq_mhz = float(entry["frequency_mhz"])
        amplitude = float(_parse_complex_amplitude(entry["amplitude"]).real)
        phase = float(entry.get("phase", 0.0))
        offset_mhz = float(s * (freq_mhz - center_mhz))
        model_peak = ModelPeak(
            amplitude=amplitude,
            offset_mhz=offset_mhz,
            phase=phase,
        )
        frozen.append(
            FrozenPeak(
                peak_index=int(entry["peak_index"]),
                primary_window_id=int(entry["primary_window_id"]),
                model_peak=model_peak,
                frequency_mhz=freq_mhz,
                freeze_eligible=bool(entry.get("freeze_eligible", True)),
                edge_free=False,
            )
        )
    return frozen


def refit_window_impl(
    file_path: Union[Path, str],
    window_id: int,
    *,
    add: Sequence[float] = (),
    remove: Sequence[float] = (),
    add_seeds: Optional[List[ModelPeak]] = None,
    snap_tol_mhz: float = _REFIT_SNAP_TOL_MHZ,
) -> RefitWindowResult:
    """User-directed single-window refit for Stage 6 review decisions.

    Re-fits one window from the persisted Stage 5 fit using the production
    NLS primitive (``fit_window``).  Starts from the persisted ``fitted_peaks``
    as seed ``ModelPeak`` objects, reconstructs the frozen background from the
    window's own persisted ``fixed_parameters`` and replays the persisted
    leakage-wing baseline, applies the caller's ``add``/``remove`` edits, and
    runs a single joint NLS over the full seeded set.  It is **NLS-only**: it
    deliberately does NOT run the conservative add-one-peak discovery loop or
    the residual-rescue pass -- a refit holds the window's peak *set* (plus or
    minus the edit) and re-converges it, rather than re-discovering peaks (the
    automatic discovery already ran; the persisted peaks are its result).  The
    result replaces ONLY that window's entry in the persisted ``SpectrumFit``;
    all other windows are untouched (no cascade, no thaw, no replan).

    Identity refit (``add=()`` and ``remove=()``) reproduces the persisted fit
    to ~1e-5 MHz on a window whose peaks are a single-window optimum.  A window
    that was subject to a **thaw co-fit** holds peaks at a *two-window* joint
    optimum; a single-window NLS relaxes those toward the one-window optimum, so
    such windows reproduce only to ~kHz (the neighbour's data is intentionally
    not re-included).  This is an inherent property of thaw, not a defect.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file (read-write).
    window_id :
        The ``FitWindow.window_id`` / ``FittingResult.window_id`` to refit.
    add :
        Molecular frequencies (MHz) of peaks to add.  Each is snapped to the
        nearest ledger candidate within ``snap_tol_mhz`` (to reuse the
        recorded seed offset/amplitude) or seeded fresh at the given
        frequency.  User-added peaks carry ``origin="user"``.
    remove :
        Molecular frequencies (MHz) of fitted peaks to remove.  Each is
        matched to the nearest fitted peak within ``snap_tol_mhz`` and dropped
        from the seed set before the NLS.
    add_seeds :
        Optional explicit :class:`~ftmwpipeline.fitting.peak_model.ModelPeak`
        seeds (in the window's baseband-offset frame) for the added peaks.
        When given, ``len(add_seeds)`` must equal ``len(add)`` and each seed
        provides the starting amplitude / offset / phase for the corresponding
        ``add`` frequency (overrides the ledger-candidate or default seed).
    snap_tol_mhz :
        Maximum distance (MHz) for frequency snapping to an existing peak or
        ledger candidate.  Defaults to :data:`_REFIT_SNAP_TOL_MHZ` (50 kHz).

    Returns
    -------
    RefitWindowResult
        Old vs new peak count, χ²ᵣ before/after, and the new fitted peaks.

    Raises
    ------
    ValueError
        When Stage 5 has not been run, the ``window_id`` is not found,
        ``len(add_seeds) != len(add)``, or any ``remove`` frequency does not
        match a fitted peak within ``snap_tol_mhz``.
    """
    from ..core.stage_fit_settings import ShapeSpec, StageFitSettings
    from ..core.stage_fit_settings import resolve as resolve_stage_fit_settings
    from ..fitting.peak_model import PeakShape
    from ..fitting.plan_execution import (
        FrozenPeak,
        WindowOutcome,
        materialize_window,
        residual_edge_coherence,
        subtract_frozen_background,
    )
    from ..fitting.result_conversion import window_outcome_to_fitting_result
    from ..fitting.window_fit import (
        ConservativeFitResult,
        KnockoutResult,
        WindowFitResult,
        derive_window_fit_constraints,
        fit_window,
        knockout_test,
    )
    from ..io.fitting_serialization import (
        load_spectrum_fit_from_hdf5,
        save_spectrum_fit_to_hdf5,
    )
    from ..io.stage_fit_settings_serialization import (
        load_stage_fit_settings_from_h5,
        read_recommended_clock_sources,
        read_stage2b_recommended_shape,
    )
    from .stage2b_g_impl import load_tau_G_calibration_impl, tau_G_calibration_present
    from .stage2b_impl import load_tau_calibration_impl, tau_calibration_present
    from .stage3_impl import load_peaks_impl
    from .stage4_impl import load_windows_impl
    from .stage5_impl import (
        Stage5FitContext,
        _required_bool,
        _required_float,
        _required_int,
        _required_str,
        _resolve_tau_calibration_for_fit,
        build_stage5_fit_context,
    )

    path = str(file_path)

    if add_seeds is not None and len(add_seeds) != len(add):
        raise ValueError(
            f"len(add_seeds)={len(add_seeds)} must equal len(add)={len(add)}"
        )

    # --- Load persisted Stage 5 fit ----------------------------------------
    with h5py.File(path, "r") as h5f:
        if "stage5_fitting" not in h5f:
            raise ValueError("No Stage 5 fit found in this file. Run 'fit run' first.")
        spectrum_fit: SpectrumFit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        shape_str: str = str(h5f["stage5_fitting"].attrs.get("shape", "lorentzian"))

    # Locate the target window's FittingResult.
    wf_list = [wf for wf in spectrum_fit.window_fits if wf.window_id == window_id]
    if not wf_list:
        raise KeyError(f"window_id={window_id} not found in the Stage 5 fit")
    wf: FittingResult = wf_list[0]
    chi2r_before = float(wf.reduced_chi2)
    n_peaks_before = len(wf.fitted_peaks)

    # --- Load Stage 4 WindowPlan and locate the FitWindow ------------------
    plan_result = load_windows_impl(path)
    plan = plan_result["plan"]
    fit_window_map = {w.window_id: w for w in plan.windows}
    if window_id not in fit_window_map:
        raise KeyError(
            f"window_id={window_id} not found in the Stage 4 WindowPlan. "
            "Stage 4 may have been re-run and changed the window geometry."
        )
    fit_win = fit_window_map[window_id]

    # --- Load Stage 3 peaks (for peak_id matching in result conversion) ----
    peaks_loaded = load_peaks_impl(path)
    peak_frequencies_mhz = [float(p.frequency) for p in peaks_loaded["peaks"]]

    # --- Resolve StageFitSettings from persisted layer + recommended -------
    # Use the PERSISTED settings as the authoritative source so the refit
    # operates under the same settings the original fit used.  No explicit
    # overrides: the user's edit is at the peak level (add/remove), not the
    # settings level.
    persisted_settings = load_stage_fit_settings_from_h5(path)
    recommended_shape_str = read_stage2b_recommended_shape(path)
    recommended_clocks = read_recommended_clock_sources(path)
    recommended_settings: Optional[StageFitSettings] = None
    if recommended_shape_str is not None or recommended_clocks is not None:
        from ..core.stage_fit_settings import SpurSubSettings

        recommended_settings = StageFitSettings(
            shape=(
                ShapeSpec.coerce(recommended_shape_str)
                if recommended_shape_str is not None
                else None
            ),
            spur=SpurSubSettings(clocks=recommended_clocks),
        )
    resolved = resolve_stage_fit_settings(
        explicit=StageFitSettings(),  # no explicit overrides
        preset=None,
        persisted=persisted_settings,
        recommended=recommended_settings,
    )
    assert resolved.shape is not None
    shape_enum = resolved.shape.kind

    # --- Stage 2b calibration (shape-routed, same logic as fit_peaks_impl) -
    persisted_cal = None
    if shape_enum is PeakShape.GAUSSIAN:
        if tau_G_calibration_present(path):
            persisted_cal = load_tau_G_calibration_impl(path)["tau_G_calibration"]
    else:
        if tau_calibration_present(path):
            persisted_cal = load_tau_calibration_impl(path)["tau_calibration"]
    tau_maj_override_v = resolved.tau.tau_maj_override_us
    sigma_tau_override_v = resolved.tau.sigma_tau_override_us
    tau_maj_us, sigma_tau_us, _tau_source = _resolve_tau_calibration_for_fit(
        persisted_cal, tau_maj_override_v, sigma_tau_override_v
    )

    # --- Build shared active-FT context (VERBATIM helper) ------------------
    # Replay the persisted Stage 5 gated spur catalogue rather than re-running
    # detection: the catalogue is a Stage 5 product, so the refit must mask the
    # window exactly as the fit did (a detector-code change between the fit and
    # the refit would otherwise silently re-mask the window it is editing).
    fit_ctx: Stage5FitContext = build_stage5_fit_context(
        path,
        resolved,
        persisted_cal,
        shape_enum,
        replay_spur_catalogue=spectrum_fit.parameters,
    )
    active_ft = fit_ctx.active_ft
    rms_for_fit = fit_ctx.rms_for_fit
    sideband: Sideband = fit_ctx.sideband
    acquisition_us = fit_ctx.acquisition_us
    spur_set = fit_ctx.spur_set

    # --- Materialize the window (grid / data / noise / center) -------------
    _, offset_grid, z_slice, sig_slice, center_mhz = materialize_window(
        fit_win,
        active_ft,
        rms_for_fit,
        sideband=sideband,
    )

    # --- Identify thawed lines in this window ---------------------------------
    # A thaw is accepted when a primary-window frozen contributor is co-fit
    # jointly with the dependent window and the residual edge coherence improves.
    # The thawed line then appears in this window's fitted_peaks (as a free peak)
    # but is owned by its primary window.  Its persisted value is the joint-fit
    # optimum; re-fitting it with only ONE window's data would relax it off that
    # joint optimum (worst observed: 28 kHz / 28% drift).  Decision: freeze it.
    #
    # Collect the molecular frequencies of all *accepted* thaw contributors for
    # this (dependent) window.  Ignore NaN frequencies (no-op records).
    thawed_freqs: List[float] = []
    for te in wf.thaw_events:
        if te.accepted and not math.isnan(te.contributor_frequency_mhz):
            thawed_freqs.append(float(te.contributor_frequency_mhz))

    def _is_thawed(freq_mhz: float) -> bool:
        """True when ``freq_mhz`` matches a thawed contributor within snap_tol."""
        return any(abs(freq_mhz - tf) <= snap_tol_mhz for tf in thawed_freqs)

    # --- Reconstruct frozen background from persisted fixed_parameters -----
    frozen_peaks = _reconstruct_frozen_peaks(wf.fixed_parameters, center_mhz, sideband)
    # The frozen background uses the window's persisted tau as the dependent
    # tau (exactly the convention in evaluate_fixed_contributor).
    tau_persisted = float(
        wf.shared_parameters.get("tau_us", {}).get("value", acquisition_us / 3.0)
    )
    # Compute background and data-minus-background.  If thawed peaks are present
    # they will be added to frozen_peaks during the partition step below, after
    # which background and data_minus_bg are recomputed with the full frozen set.
    # We do a preliminary computation here so that win_constraints (which needs
    # data_minus_bg to derive amplitude bounds) has data to work with; it is
    # immediately replaced after the partition step.
    background, data_minus_bg = subtract_frozen_background(
        offset_grid,
        z_slice,
        frozen_peaks,
        tau_persisted,
        acquisition_us,
        shape=shape_enum,
    )

    # --- Per-window spur mask (same derivation as _fit_one_window) ---------
    spur_mask = None
    if spur_set is not None and spur_set:
        lo, hi = fit_win.freq_range
        spur_mask = spur_set.window_mask_spec(lo, hi, center_mhz, sideband)

    # --- Build conservative_kwargs from resolved settings ------------------
    # Mirror the subset of conservative_kwargs that fit_window needs.
    s_sign = sideband_sign(sideband)
    max_decay_v = _required_float(resolved.tau.max_decay_factor, "tau.max_decay_factor")
    n_eff_kind_v = _required_str(
        resolved.conservative.n_eff_kind, "conservative.n_eff_kind"
    )

    # tau0 for this window: use the persisted tau as the starting point so
    # the optimizer begins at the known-good value.  For windows where tau
    # was fixed (fitted=False), the persisted tau IS the tau; for thawed
    # windows it is the converged free tau from the original fit -- in both
    # cases it is the best seed available.
    tau0_for_window = tau_persisted
    # fit_tau legitimately stays None in _HARD_DEFAULTS (the None means "use
    # the per-window decision from the original fit").  Mirror the same None
    # fallback that fit_peaks_impl uses.
    tau_was_fit = bool(wf.shared_parameters.get("tau_us", {}).get("fitted", True))
    fit_tau_for_window: bool = (
        tau_was_fit if resolved.tau.fit_tau is None else bool(resolved.tau.fit_tau)
    )

    # Build the constraint kwargs forwarded to fit_window.  Only the knobs
    # fit_window actually accepts (not the conservative-loop add-gate ones).
    # When tau_maj_us is None (no Stage 2b calibration) the tau penalty
    # reference is absent; zero the lambda so fit_window's validation passes,
    # mirroring derive_window_fit_constraints's effective_tau_penalty_lambda.
    raw_tau_penalty_lambda = _required_float(
        resolved.tau.tau_penalty_lambda, "tau.tau_penalty_lambda"
    )
    effective_tau_penalty_lambda: float = (
        raw_tau_penalty_lambda if tau_maj_us is not None and tau_maj_us > 0.0 else 0.0
    )
    phase_penalty_lambda_v: float = _required_float(
        resolved.penalties.phase_penalty_lambda, "penalties.phase_penalty_lambda"
    )
    phase_penalty_cutoff_fwhm_v: float = _required_float(
        resolved.penalties.phase_penalty_cutoff_fwhm,
        "penalties.phase_penalty_cutoff_fwhm",
    )
    amp_penalty_lambda_v: float = _required_float(
        resolved.penalties.amp_penalty_lambda, "penalties.amp_penalty_lambda"
    )

    # fw_kwargs is built in two passes:
    #   1. Penalty/tau knobs that do not depend on the data are set here.
    #   2. Amplitude / FWHM bounds (from derive_window_fit_constraints) are
    #      filled after the thawed-peak partition step, where frozen_peaks is
    #      extended and data_minus_bg is recomputed against the final frozen set.
    fw_kwargs: Dict[str, object] = {
        "fit_tau": fit_tau_for_window,
        "max_decay_factor": max_decay_v,
        "tau_penalty_lambda": effective_tau_penalty_lambda,
        "tau_penalty_reference": tau_maj_us,
        "tau_penalty_sigma_us": sigma_tau_us,
        "phase_penalty_lambda": phase_penalty_lambda_v,
        "phase_penalty_cutoff_fwhm": phase_penalty_cutoff_fwhm_v,
        "amp_penalty_lambda": amp_penalty_lambda_v,
        "shape": shape_enum,
    }
    if spur_mask is not None:
        fw_kwargs["spur_mask"] = spur_mask

    # Reproduce the persisted leakage-wing baseline. In production the baseline
    # is evidence-triggered; for a refit we replay exactly what the persisted
    # fit recorded -- co-fit the same order (and offset scale) on the same
    # frozen-bg-subtracted data so the joint NLS lands on matching baseline
    # coefficients. Without this the leakage pedestal stays in the residual and
    # the peaks shift to absorb it.
    _qm = wf.quality_metrics or {}
    if _qm.get("baseline_applied", 0.0) and "baseline_order" in _qm:
        fw_kwargs["baseline_order"] = int(_qm["baseline_order"])
        _bscale = _qm.get("baseline_offset_scale")
        if _bscale:
            fw_kwargs["baseline_offset_scale"] = float(_bscale)

    # --- Build seed ModelPeak list from persisted fitted_peaks + edits -----
    s = sideband_sign(sideband)

    # Partition fitted_peaks into free seeds (this window's own peaks) and
    # thawed-line hold-outs.  Thawed lines are owned by their primary window;
    # re-fitting them with one window's data relaxes them off the joint optimum.
    # Strategy: add each thawed peak to frozen_peaks (so it stays in the
    # frozen background the free-peak NLS fits against) and remember the
    # FittedPeak verbatim for re-insertion after the NLS.
    #
    # Note: ``frozen_peaks`` was reconstructed from ``fixed_parameters`` above
    # (the original Stage 5 frozen contributors).  A thawed line was *removed*
    # from ``fixed_parameters`` when it was accepted — it is not there.  We
    # must add it back explicitly.
    thawed_held_peaks: List[FittedPeak] = []  # verbatim re-append after NLS

    seed_peaks_with_origin: List[Tuple[ModelPeak, str]] = []
    for fp in wf.fitted_peaks:
        freq_mhz = float(fp.frequency_mhz)
        offset = float(s * (freq_mhz - center_mhz))
        if _is_thawed(freq_mhz):
            # Hold this peak out of the free NLS.  Add it to the frozen
            # background so the window's own peaks fit against it correctly.
            # Use a sentinel primary_window_id of -1 (we don't need it here;
            # the frozen model only needs amplitude/offset/phase and tau).
            mp = ModelPeak(
                amplitude=float(fp.amplitude),
                offset_mhz=offset,
                phase=float(fp.phase) if fp.phase is not None else 0.0,
            )
            frozen_peaks.append(
                FrozenPeak(
                    peak_index=int(fp.peak_id) if fp.peak_id is not None else -1,
                    primary_window_id=-1,
                    model_peak=mp,
                    frequency_mhz=freq_mhz,
                    freeze_eligible=False,
                    edge_free=False,
                )
            )
            thawed_held_peaks.append(fp)
        else:
            mp = ModelPeak(
                amplitude=float(fp.amplitude),
                offset_mhz=offset,
                phase=float(fp.phase) if fp.phase is not None else 0.0,
            )
            seed_peaks_with_origin.append((mp, fp.origin))

    # Apply "remove" edits: drop seeds closest to remove frequencies.
    # A "remove" on a thawed line drops it entirely (not frozen): the user
    # decided to remove it, so it is neither re-fit nor re-appended.
    forbidden_offsets: List[float] = []
    for rm_freq in remove:
        rm_offset = float(s * (float(rm_freq) - center_mhz))

        # Check if this removal targets a thawed held-out peak.
        thawed_match_idx: Optional[int] = None
        for ti, theld in enumerate(thawed_held_peaks):
            if abs(float(theld.frequency_mhz) - float(rm_freq)) <= snap_tol_mhz:
                thawed_match_idx = ti
                break
        if thawed_match_idx is not None:
            # Drop the thawed peak from the frozen background and the hold-out
            # list.  Find the matching FrozenPeak by frequency and remove it.
            removed_fp = thawed_held_peaks.pop(thawed_match_idx)
            frozen_peaks = [
                fp
                for fp in frozen_peaks
                if abs(fp.frequency_mhz - float(removed_fp.frequency_mhz))
                > snap_tol_mhz
            ]
            forbidden_offsets.append(rm_offset)
            continue

        if not seed_peaks_with_origin:
            raise ValueError(
                f"remove={rm_freq:.4f} MHz: no fitted peaks in window "
                f"{window_id} to remove"
            )
        closest_idx = min(
            range(len(seed_peaks_with_origin)),
            key=lambda i: abs(seed_peaks_with_origin[i][0].offset_mhz - rm_offset),
        )
        closest_dist_val = abs(
            seed_peaks_with_origin[closest_idx][0].offset_mhz - rm_offset
        )
        if closest_dist_val > snap_tol_mhz:
            closest_mol_freq = (
                center_mhz + s * seed_peaks_with_origin[closest_idx][0].offset_mhz
            )
            raise ValueError(
                f"remove={rm_freq:.4f} MHz: no fitted peak within "
                f"{snap_tol_mhz:.3f} MHz (closest is at "
                f"{closest_mol_freq:.4f} MHz, "
                f"distance={closest_dist_val:.4f} MHz)"
            )
        # Record the exact fitted offset as forbidden (rescue must not re-add it).
        removed_offset = seed_peaks_with_origin.pop(closest_idx)[0].offset_mhz
        forbidden_offsets.append(removed_offset)

    # Recompute background and data_minus_bg with the final frozen_peaks set
    # (which now includes any thawed peaks that were not removed).  This
    # replaces the preliminary computation made before the partition step.
    # Even for windows with no thawed lines the recompute is a no-op (frozen_peaks
    # is unchanged), so we always do it to keep the code simple.
    background, data_minus_bg = subtract_frozen_background(
        offset_grid,
        z_slice,
        frozen_peaks,
        tau_persisted,
        acquisition_us,
        shape=shape_enum,
    )

    # Derive amp_floor, fwhm_mhz, and amp_max from the (now-final) data_minus_bg
    # so fit_window's amplitude and phase penalties are properly bounded against
    # the data the NLS will actually see (with thawed lines already subtracted).
    win_constraints = derive_window_fit_constraints(
        data_minus_bg,
        sig_slice,
        tau0_for_window,
        acquisition_us,
        fit_tau=fit_tau_for_window,
        max_decay_factor=max_decay_v,
        phase_penalty_lambda=phase_penalty_lambda_v,
        amp_penalty_lambda=amp_penalty_lambda_v,
        tau_penalty_lambda=effective_tau_penalty_lambda,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        shape=shape_enum,
    )
    fw_kwargs["amp_floor"] = win_constraints.amp_floor
    fw_kwargs["fwhm_mhz"] = win_constraints.fwhm
    fw_kwargs["amp_max"] = win_constraints.amp_max

    # Apply "add" edits: append new ModelPeak seeds.
    protected_offsets: List[float] = []
    if add_seeds is not None:
        explicit_seeds = list(add_seeds)
    else:
        explicit_seeds = []

    # Derive per-window ledger candidates for snap-to-candidate logic.
    center_for_ledger = center_mhz
    wf_sideband = _sideband_from_value(sideband)

    for i, add_freq in enumerate(add):
        add_offset = float(s * (float(add_freq) - center_mhz))
        if explicit_seeds:
            mp = explicit_seeds[i]
        else:
            # Snap to the nearest ledger candidate if within tolerance.
            ledger = derive_candidate_ledger(
                wf,
                center_mhz=center_for_ledger,
                sideband=wf_sideband,
                bar=0.0,  # all candidates; the user has decided to add this peak
            )
            best_cand = None
            best_dist = float("inf")
            for cand in ledger:
                dist = abs(float(cand.frequency_mhz) - float(add_freq))
                if dist < best_dist:
                    best_dist = dist
                    best_cand = cand
            if best_cand is not None and best_dist <= snap_tol_mhz:
                # Reuse the ledger candidate's recorded seed amplitude and offset.
                cand_offset = float(s * (float(best_cand.frequency_mhz) - center_mhz))
                cand_amp = (
                    float(best_cand.seed_amplitude)
                    if best_cand.seed_amplitude is not None
                    else float(np.max(np.abs(data_minus_bg)))
                )
                mp = ModelPeak(
                    amplitude=cand_amp,
                    offset_mhz=cand_offset,
                    phase=0.0,
                )
            else:
                # Fresh seed: amplitude from data at nearest bin.
                nearest_bin = int(np.argmin(np.abs(offset_grid - add_offset)))
                amp_seed = float(
                    2.0
                    * np.abs(data_minus_bg[nearest_bin])
                    / max(tau0_for_window, 1e-6)
                )
                mp = ModelPeak(
                    amplitude=max(amp_seed, 1e-30),
                    offset_mhz=add_offset,
                    phase=float(np.angle(data_minus_bg[nearest_bin])),
                )
        seed_peaks_with_origin.append((mp, "user"))
        protected_offsets.append(mp.offset_mhz)

    # Extract final seed list in offset order.
    final_seeds = [mp for mp, _ in seed_peaks_with_origin]
    origin_flags = [orig for _, orig in seed_peaks_with_origin]

    # --- Single joint NLS over the seeded set (production fit_window) ------
    fit_result: WindowFitResult = fit_window(
        offset_grid,
        data_minus_bg,
        sig_slice,
        initial_peaks=final_seeds,
        tau0_us=tau0_for_window,
        acquisition_us=acquisition_us,
        **fw_kwargs,  # type: ignore[arg-type]
    )

    # Run knockout test to populate KnockoutResult per peak. ``knockout_test``
    # passes ``tau0_us``/``acquisition_us`` (positionally) and ``spur_mask``
    # (explicitly) into its inner ``fit_window`` refits, so ``fit_kwargs_inner``
    # must carry only the *other* fit_window kwargs -- including any of those
    # here would collide ("multiple values for argument") on the multi-peak
    # path where the refit actually runs.
    knockout_inner = {k: v for k, v in fw_kwargs.items() if k != "spur_mask"}
    knockouts = knockout_test(
        offset_grid,
        data_minus_bg,
        sig_slice,
        fit_result,
        acquisition_us,
        fit_kwargs_inner=knockout_inner,
        spur_mask=spur_mask,
        n_eff_kind=n_eff_kind_v,
    )

    # Wrap in ConservativeFitResult (empty audit_trail — this is a joint refit,
    # not a conservative loop; the audit provenance is the user decision log).
    conservative_result = ConservativeFitResult(
        fit=fit_result,
        audit_trail=[],
        knockouts=knockouts,
    )

    # A user refit is NLS-only: it holds the persisted peak set (plus/minus the
    # user's edit) and re-converges it. It deliberately does NOT run residual
    # rescue / discovery -- that pass re-litigates the whole window (it would
    # add brand-new peaks the user did not ask for and break the
    # "changes its own peaks only" contract). The window's discovery already
    # ran during the automatic fit; the persisted peaks are its result.
    # ``protected_offsets`` / ``forbidden_offsets`` are computed above for the
    # edit bookkeeping but no automatic add/prune pass runs here to consult
    # them.
    final_conservative = conservative_result

    # --- Build WindowOutcome (no thaw, no baseline re-fit in user refit) ---
    from ..fitting.peak_model import model_spectrum
    from ..fitting.window_fit import evaluate_baseline

    free_model = model_spectrum(
        offset_grid,
        final_conservative.fit.peaks,
        final_conservative.fit.tau_us,
        acquisition_us,
        shape=shape_enum,
    ) + evaluate_baseline(final_conservative.fit, offset_grid)
    full_fitted = free_model + background
    full_residual = z_slice - full_fitted
    low_coh, high_coh = residual_edge_coherence(
        full_residual,
        sig_slice,
        band_m=_required_int(resolved.thaw.residual_edge_m, "thaw.residual_edge_m"),
    )
    outcome = WindowOutcome(
        window_id=window_id,
        fit=final_conservative,
        fixed_peaks=frozen_peaks,
        offset_grid_mhz=offset_grid,
        complex_spectrum=z_slice,
        rms_noise=sig_slice,
        background=background,
        full_fitted_spectrum=full_fitted,
        full_residual=full_residual,
        edge_coherence_low=low_coh,
        edge_coherence_high=high_coh,
    )
    # Stash center and spur mask as _fit_one_window does (used by converters).
    outcome._center_mhz = center_mhz  # type: ignore[attr-defined]
    outcome._spur_mask = spur_mask  # type: ignore[attr-defined]

    # --- Convert to FittingResult ------------------------------------------
    new_wf: FittingResult = window_outcome_to_fitting_result(
        outcome,
        fit_win,
        sideband=sideband,
        peak_frequencies_mhz=peak_frequencies_mhz,
        acquisition_us=acquisition_us,
    )

    # Stamp user-origin on peaks added by the caller.  The origin_flags list
    # mirrors the seed_peaks_with_origin order; the converged fit may have
    # fewer/reordered peaks (rescue drops/merges), so we match by nearest
    # frequency within snap_tol_mhz.
    user_freq_mhz = [
        float(center_mhz + s * mp.offset_mhz)
        for mp, orig in zip(final_seeds, origin_flags)
        if orig == "user"
    ]
    for fp in new_wf.fitted_peaks:
        for uf in user_freq_mhz:
            if abs(float(fp.frequency_mhz) - uf) <= snap_tol_mhz:
                fp.origin = "user"
                break

    # --- Re-insert thawed lines verbatim ------------------------------------
    # Thawed lines were held out of the NLS and frozen into the background so
    # the window's own peaks converged correctly against them.  Now re-attach
    # them to the output UNCHANGED (same frequency / amplitude / phase / errors
    # / origin as the persisted FittedPeak).  Their model contribution is
    # already accounted for in ``full_fitted`` / ``full_residual`` (they were
    # part of ``background``), so χ²ᵣ in the result is consistent.
    if thawed_held_peaks:
        new_wf.fitted_peaks = sorted(
            new_wf.fitted_peaks + thawed_held_peaks,
            key=lambda fp2: float(fp2.frequency_mhz),
        )
        logger.debug(
            "Stage 6 refit window %d: re-inserted %d thawed peak(s) verbatim",
            window_id,
            len(thawed_held_peaks),
        )

    # --- Replace this window's entry in SpectrumFit + persist --------------
    # Update the global fitted_peaks list: remove old peaks for this window,
    # insert new ones, keep all other windows' peaks unchanged, re-sort.
    new_global_peaks = [
        p for p in spectrum_fit.fitted_peaks if p.window_id != window_id
    ] + list(new_wf.fitted_peaks)
    new_global_peaks.sort(key=lambda p: float(p.frequency_mhz))
    spectrum_fit.fitted_peaks = new_global_peaks

    # Replace the per-window FittingResult entry.
    spectrum_fit.window_fits = [
        new_wf if wf_entry.window_id == window_id else wf_entry
        for wf_entry in spectrum_fit.window_fits
    ]

    # Persist the updated SpectrumFit.
    shape_attr = str(spectrum_fit.parameters.get("shape", "lorentzian"))
    with h5py.File(path, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        grp = h5f.create_group("stage5_fitting")
        save_spectrum_fit_to_hdf5(spectrum_fit, grp)
        grp.attrs["shape"] = shape_attr

    chi2r_after = float(new_wf.reduced_chi2)
    n_peaks_after = len(new_wf.fitted_peaks)
    logger.info(
        "Stage 6 refit window %d: %d → %d peaks, χ²ᵣ %.3g → %.3g",
        window_id,
        n_peaks_before,
        n_peaks_after,
        chi2r_before,
        chi2r_after,
    )
    return RefitWindowResult(
        window_id=window_id,
        n_peaks_before=n_peaks_before,
        n_peaks_after=n_peaks_after,
        chi2r_before=chi2r_before,
        chi2r_after=chi2r_after,
        fitted_peaks=list(new_wf.fitted_peaks),
    )
