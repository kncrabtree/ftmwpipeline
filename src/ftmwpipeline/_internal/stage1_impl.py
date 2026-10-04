"""
Shared implementation for Stage 1: FT Processing and Spectrum Operations.

Stage 1 owns the *persisted* FT processing settings. Before Stage 1 has run,
the settings resolve through ``explicit override > import-time recommended``
(see :mod:`ftmwpipeline.core.settings`). A user-driven Stage 1 invocation
(``persist=True``) writes the resolved :class:`FTSettings` -- the concrete
active region and the frequency ``trim`` -- to
``processing_parameters/ft_processing`` as the experiment's binding settings.

From then on that record is authoritative: :func:`resolve_ft_settings_h5`, the
one resolver every reader goes through (Stages 2-5 via :func:`compute_ft_impl`,
Stage 2b, the shape recommendation, the timebase, the metadata view), reads
``explicit > persisted`` and never consults the recommended layer, so a later
``start run`` or chirp-window stamp to that layer changes nothing Stage 1 is
read as having used. The recommendation is still stored for display; adopting
it is an explicit Stage 1 re-run with the new value, which invalidates every
stage built on the old spectrum. A record from before this rule (no field-set
version, or an older one) keeps the ``explicit > persisted > recommended``
fall-through it was written under.

This module is a thin orchestration layer; it is wrapped identically by the
CLI, the ``Pipeline`` class, and the functional API.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from ..core.data_structures import FID, ComplexFT
from ..core.settings import (
    FT_PROCESSING_FIELD_SET_VERSION,
    FT_PROCESSING_PATH,
    RECOMMENDED_PATH,
    FTSettings,
    resolve,
)
from ..io.provenance import (
    RecordProvenance,
    group_provenance,
    record_provenance,
    stamp_stage_epoch,
    write_field_set_version,
)
from .compaction import compact_file
from .shared_utils import fold_settings_blob
from .stage0_impl import load_fid_from_pipeline_impl

logger = logging.getLogger(__name__)


def _read_settings_layer(file_path: str, group_path: str) -> Optional[FTSettings]:
    """Read one resolution layer (persisted or recommended) as ``FTSettings``.

    Tolerant of older ``ft_processing`` records that stored only a JSON
    ``parameters`` blob: its keys are folded in so those files still resolve
    sensibly.
    """
    with h5py.File(file_path, "r") as h5f:
        return _layer_from_h5(h5f, group_path)


def ft_settings_provenance(file_path: str) -> Optional[RecordProvenance]:
    """The ``ft_processing`` record's field-set version against
    :data:`~ftmwpipeline.core.settings.FT_PROCESSING_FIELD_SET_VERSION`, or
    ``None`` if Stage 1 has persisted no record."""
    return record_provenance(
        file_path, FT_PROCESSING_PATH, FT_PROCESSING_FIELD_SET_VERSION
    )


def _layer_from_h5(h5f: h5py.File, group_path: str) -> Optional[FTSettings]:
    """One resolution layer of an open file, read as :func:`_read_settings_layer`
    reads it."""
    group = h5f.get(group_path)
    if group is None:
        return None
    return FTSettings.from_attrs(fold_settings_blob(dict(group.attrs)))


def ft_record_is_authoritative(h5f: h5py.File) -> bool:
    """True when ``h5f`` carries a Stage 1 record written under the
    authoritative rule (field-set version 2 or later).

    Such a record holds the concrete values Stage 1 ran with, so nothing falls
    through it to the recommended layer. A record with no version, or an older
    one, is pre-provenance and keeps the fall-through it was written under. A
    record from a newer version is treated as authoritative too: the rule only
    tightens, so the newer writer certainly meant it.
    """
    group = h5f.get(FT_PROCESSING_PATH)
    if group is None:
        return False
    provenance = group_provenance(group, FT_PROCESSING_FIELD_SET_VERSION)
    return not provenance.is_pre_provenance


def resolve_ft_settings_h5(
    h5f: h5py.File, explicit: Optional[FTSettings] = None
) -> FTSettings:
    """The Stage 1 settings resolver for an open file -- the only one.

    With an authoritative Stage 1 record (see
    :func:`ft_record_is_authoritative`) the chain is ``explicit > persisted``:
    the recommended layer takes no part, and a field the record holds as unset
    (only ``trim`` can be: "no trim") stays unset. Otherwise -- before Stage 1
    has run, or on a pre-provenance record -- it is the original
    ``explicit > persisted > recommended`` chain. ``units_power`` always ends in
    its hard default.

    The active region may still be unset here (an old record, or no record);
    callers that need the window call
    :meth:`~ftmwpipeline.core.settings.FTSettings.with_effective_window` with
    the FID duration, which every reader in the pipeline does through
    :func:`compute_ft_impl` or :func:`persisted_ft_settings`.
    """
    persisted = _layer_from_h5(h5f, FT_PROCESSING_PATH)
    recommended = (
        None
        if ft_record_is_authoritative(h5f)
        else _layer_from_h5(h5f, RECOMMENDED_PATH)
    )
    return resolve(explicit, persisted, recommended)


def _resolve_settings(file_path: str, explicit: Optional[FTSettings]) -> FTSettings:
    """Resolve the FT settings for ``file_path`` (:func:`resolve_ft_settings_h5`)."""
    with h5py.File(file_path, "r") as h5f:
        return resolve_ft_settings_h5(h5f, explicit)


def persisted_ft_settings(
    file_path: str, stage_name: str, fid_duration_us: float
) -> FTSettings:
    """The settings Stage 1 is read as having used, with a concrete window.

    For a consumer that requires Stage 1 to have persisted its record (Stage 2b,
    the shape recommendation, the timebase): raises
    :class:`~ftmwpipeline.file_manager.StageDependencyError` naming
    ``stage_name`` when there is none. The values come from
    :func:`resolve_ft_settings_h5`, the same resolver :func:`compute_ft_impl`
    uses for Stages 2-5, so every consumer agrees on the window and the trim.
    """
    from ..file_manager import StageDependencyError

    with h5py.File(file_path, "r") as h5f:
        if FT_PROCESSING_PATH not in h5f:
            raise StageDependencyError(
                stage_name,
                ["stage1_complex_ft"],
                Path(file_path),
            )
        resolved = resolve_ft_settings_h5(h5f)
    return resolved.with_effective_window(fid_duration_us)


def _reject_window_past_record(
    end_us: float, duration_us: float, sample_dt_us: float
) -> None:
    """Refuse an active-region end time beyond the end of the recording.

    An ``end_us`` past the record is a mistake, not a request to be honored:
    the active region would claim a duration the FID does not contain, and
    ``1 / T`` -- the active-FT bin spacing every bin-relative tolerance in the
    pipeline resolves against (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8)
    -- would describe a spectrum that does not exist. The sample slice clamps
    itself, so nothing crashes; the tolerances just quietly come out wrong, by
    the ratio of the claimed length to the real one.

    It is refused rather than trimmed because there is already a correct way to
    say "to the end of the record", and it costs the caller nothing: leave
    ``end_us`` unset. Trimming would be the right compromise only if a caller
    were obliged to state an exact end time, and none is.

    A half-sample tolerance is allowed so that naming the record's own duration
    -- the commonest way to write "all of it" explicitly -- is never rejected
    by float representation.
    """
    if end_us > duration_us + 0.5 * sample_dt_us:
        raise ValueError(
            f"end_us={end_us:g} us is past the end of the recording "
            f"({duration_us:g} us). The active region cannot be longer than "
            f"the FID: its length sets the active-FT bin spacing that every "
            f"frequency tolerance is resolved against. Pass an end_us within "
            f"the record, or omit it to use the whole record."
        )


def compute_ft_impl(
    file_path: str,
    settings: Optional[FTSettings] = None,
    validate_only: bool = False,
    persist: bool = False,
) -> Dict[str, Any]:
    """Compute the FT for a ``.ftmw`` file using the resolved settings.

    Parameters
    ----------
    file_path:
        Path to the ``.ftmw`` pipeline file.
    settings:
        Explicit overrides (sparse :class:`FTSettings`; ``None`` fields fall
        through to persisted, then recommended). ``None`` means "no overrides"
        -- the normal path used by downstream stages.
    validate_only:
        If True, return validation info without constructing the ``ComplexFT``.
    persist:
        If True (a user-driven Stage 1 invocation), write the *resolved*
        settings to ``processing_parameters/ft_processing`` and
        mark Stage 1 complete. Internal recomputes pass ``False``.

    Returns
    -------
    dict
        ``complex_ft`` plus metadata. Return keys are kept stable for the
        untouched Stage 2/3 callers (``complex_ft``, ``trim_range``,
        ``processing_params``, ...).
    """
    try:
        fid = load_fid_from_pipeline_impl(file_path)
        logger.info(f"Loaded FID with {len(fid.data):,} points from pipeline file")
    except Exception as e:
        raise RuntimeError(f"Failed to load FID from pipeline file {file_path}: {e}")

    # The window is made concrete here, once: an unset start is 0.0 and an
    # unset end is the FID duration, which select exactly the samples the unset
    # bounds did. Every caller (and the persisted record) then sees the values
    # the FT was actually computed with.
    resolved = _resolve_settings(file_path, settings).with_effective_window(
        float(fid.duration_us)
    )
    trim_range = resolved.trim
    _start_us, end_us = resolved.active_window_us()

    _reject_window_past_record(
        end_us,
        float(fid.duration_us),
        float(fid.spacing) * 1e6,
    )

    logger.info("Resolved FT processing settings:")
    for name, value in resolved.to_preprocess_kwargs().items():
        logger.info(f"  {name}: {value}")
    logger.info(f"  trim: {trim_range}")

    try:
        preprocessed_fid = fid.preprocess(**resolved.to_preprocess_kwargs())
        logger.info(f"Preprocessing complete: {len(preprocessed_fid.data):,} points")
    except Exception as e:
        raise ValueError(f"FID preprocessing failed: {e}")

    try:
        complex_spectrum, freq_array = preprocessed_fid.compute_fft()
        logger.info(
            f"FFT complete: {len(complex_spectrum):,} frequency points "
            f"({freq_array[0]:.1f} - {freq_array[-1]:.1f} MHz)"
        )
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}")

    processing_params_dict = resolved.to_preprocess_kwargs()

    if validate_only:
        result: Dict[str, Any] = {
            "status": "validated",
            "fid_points": len(fid.data),
            "preprocessed_points": len(preprocessed_fid.data),
            "frequency_points": len(complex_spectrum),
            "frequency_range": (freq_array[0], freq_array[-1]),
            "processing_params": processing_params_dict,
        }
        if trim_range is not None:
            import numpy as np

            mask = (freq_array >= trim_range[0]) & (freq_array <= trim_range[1])
            if not np.any(mask):
                raise ValueError(
                    f"No data points in trim range "
                    f"[{trim_range[0]:.1f}, {trim_range[1]:.1f}] MHz"
                )
            result["trimmed_points"] = int(np.sum(mask))
            result["trim_range"] = trim_range
        if persist:
            _persist_ft_settings(
                file_path, resolved, fid_duration_us=float(fid.duration_us)
            )
        return result

    try:
        fid_context = {
            "probe_freq_mhz": fid.probe_freq_mhz,
            "spacing_us": fid.spacing * 1e6,
            "sideband": fid.sideband.value,
            "original_fid_length": len(fid.data),
        }
        complex_ft = ComplexFT.from_spectrum(
            complex_spectrum=complex_spectrum,
            freq_array=freq_array,
            metadata={
                "processing_params": preprocessed_fid.processing_params,
                "fid_context": fid_context,
            },
        )
        logger.info(f"ComplexFT created: {len(complex_ft.complex_spectrum):,} points")
    except Exception as e:
        raise RuntimeError(f"ComplexFT creation failed: {e}")

    if trim_range is not None:
        try:
            complex_ft = complex_ft.trim_to_range(trim_range[0], trim_range[1])
            logger.info(
                f"Trimmed spectrum: {len(complex_ft.complex_spectrum):,} points"
            )
        except Exception as e:
            raise ValueError(f"Frequency trimming failed: {e}")

    result = {
        "status": "success",
        "complex_ft": complex_ft,
        "preprocessed_fid": preprocessed_fid,
        "original_fid": fid,
        "processing_params": processing_params_dict,
        "resolved_settings": resolved,
        "fid_points": len(fid.data),
        "preprocessed_points": len(preprocessed_fid.data),
        "frequency_points": len(complex_ft.complex_spectrum),
        "frequency_range": (complex_ft.freq_array[0], complex_ft.freq_array[-1]),
    }
    if trim_range is not None:
        result["trim_range"] = trim_range

    if persist:
        try:
            _persist_ft_settings(
                file_path, resolved, fid_duration_us=float(fid.duration_us)
            )
            logger.info("FT settings + Stage 1 completion persisted")
        except Exception as e:
            logger.warning(f"Failed to persist FT settings: {e}")

    return result


def visualize_ft_impl(
    file_path: str,
    settings: Optional[FTSettings] = None,
    title: Optional[str] = None,
    show_fid_panels: bool = True,
    interactive: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Render the FID-to-spectrum workflow for the resolved settings.

    Visualization never persists; it is an exploration tool.
    """
    ft_result = compute_ft_impl(
        file_path=file_path, settings=settings, validate_only=False
    )
    complex_ft = ft_result["complex_ft"]
    original_fid = ft_result["original_fid"]
    resolved_settings = ft_result["resolved_settings"]
    trim_range = ft_result.get("trim_range")

    try:
        from ..visualization.spectrum_visualization import plot_complex_ft
    except ImportError:
        raise ImportError(
            "Spectrum visualization not available - visualization module missing"
        )

    if title is None:
        title = f"Pipeline {Path(file_path).stem} - FT Visualization"
        if trim_range is not None:
            title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"

    display_ft, active_time_us, active_data = _build_display_and_active_fid(
        original_fid, resolved_settings, complex_ft, show_fid_panels
    )

    try:
        fig = plot_complex_ft(
            complex_ft=display_ft,
            title=title,
            interactive=interactive,
            fid=original_fid if show_fid_panels else None,
            active_fid_time_us=active_time_us,
            active_fid_data=active_data,
            show_fid_panels=show_fid_panels,
            **plot_kwargs,
        )
        logger.info("FT visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create FT visualization: {e}")


def _build_display_and_active_fid(
    original_fid: FID,
    resolved_settings: FTSettings,
    complex_ft: ComplexFT,
    show_fid_panels: bool,
) -> Tuple[ComplexFT, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build the zero-padded active-band display FT plus the active-region
    FID slice (time axis + DC-removed data) for :func:`visualize_ft_impl`.

    Reuses the same padded-display-FT construction Stage 5's report / ``fit
    show`` use (:func:`stage5_impl._padded_active_display_ft`), driven by
    ``resolved_settings`` -- the settings actually in effect for *this*
    visualization call (persisted, or the caller's explicit overrides) --
    rather than only the persisted Stage 1 record, so ``ft show --start-us
    ... --end-us ...`` still changes what is plotted. It is display-only:
    computes no statistics and feeds nothing downstream.
    """
    from ..fitting.active_ft import active_region_bounds
    from .stage5_impl import UNITS_LABEL_BY_POWER, _padded_active_display_ft

    fid_samples = np.asarray(original_fid.data, dtype=float)
    sample_dt_us = original_fid.spacing * 1e6
    start_us, end_us = resolved_settings.active_window_us()
    _reject_window_past_record(end_us, float(original_fid.duration_us), sample_dt_us)

    freq, spectrum = _padded_active_display_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=original_fid.probe_freq_mhz,
        sideband=original_fid.sideband,
    )

    # Trim to the resolved (possibly trim_range-limited) complex_ft's own
    # band, same tolerance convention as compute_display_ft_impl.
    fmin = float(np.min(complex_ft.freq_array))
    fmax = float(np.max(complex_ft.freq_array))
    tol = float(freq[1] - freq[0]) / 4.0 if freq.size > 1 else 0.0
    band = (freq >= fmin - tol) & (freq <= fmax + tol)
    freq = np.ascontiguousarray(freq[band])
    spectrum = np.ascontiguousarray(spectrum[band])

    units_power = resolved_settings.units_power
    if units_power is None:
        amplitude_scale, units_label = 1.0, ""
    else:
        amplitude_scale = 10.0 ** int(units_power)
        units_label = UNITS_LABEL_BY_POWER.get(int(units_power), f"·10^{units_power} V")

    display_ft = ComplexFT.from_spectrum(
        complex_spectrum=spectrum,
        freq_array=freq,
        metadata={
            "amplitude_scale": amplitude_scale,
            "units_label": units_label,
        },
    )

    if not show_fid_panels:
        return display_ft, None, None

    start_idx, end_idx = active_region_bounds(
        fid_samples.size, sample_dt_us, start_us, end_us
    )
    active_data = fid_samples[start_idx:end_idx].copy()
    if active_data.size:
        active_data -= active_data.mean()
    active_time_us = original_fid.time_array_us()[start_idx:end_idx]

    return display_ft, active_time_us, active_data


def _dependents_of(stage: str, deps: Dict[str, Any]) -> set:
    """Transitive set of stages that depend (directly/indirectly) on ``stage``.

    Excludes ``stage`` itself. Used to invalidate everything built on the FT
    when the persisted FT settings change.
    """
    result: set = set()
    changed = True
    while changed:
        changed = False
        for st, required in deps.items():
            if st in result:
                continue
            if any(r == stage or r in result for r in required):
                result.add(st)
                changed = True
    return result


def _stored_fid_duration_us(file_path: str) -> Optional[float]:
    """The stored FID's duration, or ``None`` for a file that holds no FID."""
    with h5py.File(file_path, "r") as h5f:
        if "stage0_fid_data" not in h5f:
            return None
    return float(load_fid_from_pipeline_impl(file_path).duration_us)


def _effective_attrs(
    settings: FTSettings, fid_duration_us: Optional[float]
) -> Dict[str, Any]:
    """``settings`` as record attrs, with the window made concrete when the FID
    duration is known. A file with no FID (only ever hand-built) keeps an unset
    bound, which the resolver reads as the whole record all the same."""
    if fid_duration_us is not None:
        settings = settings.with_effective_window(fid_duration_us)
    return settings.to_attrs()


def _persist_ft_settings(
    file_path: str,
    resolved: FTSettings,
    *,
    fid_duration_us: Optional[float] = None,
) -> None:
    """Write resolved settings to ``ft_processing`` and mark Stage 1 complete.

    The record is written at
    :data:`~ftmwpipeline.core.settings.FT_PROCESSING_FIELD_SET_VERSION` with a
    concrete active region (``start_us`` 0.0 and ``end_us`` the FID duration
    when ``resolved`` leaves them unset; ``fid_duration_us`` is read from the
    file when the caller does not pass it) and its ``trim`` as resolved, unset
    meaning "no trim". That makes it authoritative (see
    :func:`resolve_ft_settings_h5`).

    No ``ComplexFT`` is stored -- the lightweight ``.ftmw`` model recomputes it
    on demand from the FID + these settings. If the resolved settings *differ*
    from a previously persisted record, every stage built on the FT
    (Stage 2 noise, Stage 3 peaks, ...) is invalidated: its stored result is
    removed, it is dropped from the completed set, and a loud warning is
    logged. An identical re-persist (idempotent Jupyter re-run, or an older
    record's unset window rewritten as the concrete same window) invalidates
    nothing.

    Completing Stage 1 stamps its analysis epoch (``stage1_complex_ft``) before
    writing anything else; a stamp that cannot be written raises before the
    record or the completion is touched (see :mod:`ftmwpipeline.io.provenance`).
    """
    from ..file_manager import PipelineStageTracker

    if fid_duration_us is None:
        fid_duration_us = _stored_fid_duration_us(file_path)
    new_attrs = _effective_attrs(resolved, fid_duration_us)
    with h5py.File(file_path, "a") as h5f:
        # Stamp first, so a stamp that cannot be written leaves the record and
        # the completion untouched. Stage 1 stores no computed artifact (the
        # FT is recomputed on demand), so persisting it again never overwrites
        # old numbers: it is not a re-run in the legacy-warning sense.
        stamp_stage_epoch(h5f, "stage1_complex_ft", rerun=False)
        proc = h5f.require_group("processing_parameters")
        old_attrs = None
        if "ft_processing" in proc:
            # Compared on the effective window, so upgrading an older record
            # whose window was unset (and selected the whole record) to the
            # concrete spelling of that same window is not a change and
            # invalidates nothing.
            old_attrs = _effective_attrs(
                FTSettings.from_attrs(dict(proc["ft_processing"].attrs)),
                fid_duration_us,
            )
            del proc["ft_processing"]
        ft_group = proc.create_group("ft_processing")
        for name, value in new_attrs.items():
            ft_group.attrs[name] = value
        write_field_set_version(ft_group.attrs, FT_PROCESSING_FIELD_SET_VERSION)
        # Human/debug mirror of the persisted record.
        ft_group.attrs["parameters"] = json.dumps(new_attrs, default=str)
        ft_group.attrs["last_updated"] = datetime.now().isoformat()

        stages = h5f.require_group("pipeline_stages")
        completed = json.loads(stages.attrs.get("completed_stages", "[]"))

        if old_attrs is not None and old_attrs != new_attrs:
            deps = PipelineStageTracker.STAGE_DEPENDENCIES
            paths = PipelineStageTracker.STAGE_DATA_PATHS
            invalidated = []
            for st in _dependents_of("stage1_complex_ft", deps):
                data_path = paths.get(st, st)
                if data_path in h5f:
                    del h5f[data_path]
                if st in completed:
                    completed.remove(st)
                    invalidated.append(st)
            if invalidated:
                logger.warning(
                    "FT settings changed (%s -> %s); invalidated "
                    "downstream stage(s) %s -- re-run them on the new "
                    "spectrum.",
                    old_attrs,
                    new_attrs,
                    sorted(invalidated),
                )

        if "stage1_complex_ft" not in completed:
            completed.append("stage1_complex_ft")
        stages.attrs["completed_stages"] = json.dumps(completed)
        stages.attrs["last_updated"] = datetime.now().isoformat()
    logger.info("FT parameters and stage tracking saved to pipeline file")
    # Stage 1 stamps its own completion rather than going through
    # ``_update_stage_completion``, so it reclaims its own dead space too.
    compact_file(file_path)


def save_ft_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """Persist an explicit settings dict (used by --save flows)."""
    settings = FTSettings.from_attrs(parameters)
    resolved = _resolve_settings(file_path, settings)
    _persist_ft_settings(file_path, resolved)
    logger.info("Processing parameters saved successfully")


def compare_ft_parameters_impl(
    file_path: str, current_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Report which of ``current_params`` differ from the resolved defaults."""
    defaults = _resolve_settings(file_path, None).to_attrs()
    custom = {
        k: v
        for k, v in current_params.items()
        if v is not None and defaults.get(k) != v
    }
    return {
        "default_params": defaults,
        "current_params": current_params,
        "custom_params": custom,
        "has_custom_params": len(custom) > 0,
    }
