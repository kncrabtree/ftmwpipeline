"""Acquisition layout for segmented scope records.

A long scope record contains a quiet pre-record, N chirp-FID frames at a
fixed repetition period, and a dead tail.  This module provides the
*operator-supplied* segment map, a pure function that slices the record
into its constituent parts, and interleave-ADC offset cleanup helpers.

Glossary
--------
pre_record_us : float
    Duration of the quiet pre-record (before the first frame), in
    microseconds.
frame_period_us : float
    Repetition period of one frame (chirp + FID + dead time within the
    frame), in microseconds.
n_frames : int
    Number of frames in the record.
frame : int or None
    When ``None``, coherently average all frames into the science FID.
    When an int (0-based), select that single frame as the science FID.
keep_frames : bool
    When ``True``, preserve the full per-frame data array in the returned
    result for downstream per-frame statistics.

Interleave-offset cleanup
-------------------------
Direct-sampling ADCs that interleave M sub-ADCs produce periodic DC
offsets at multiples of fs/M.  ``estimate_interleave_pattern`` computes
per-phase means from the quiet pre-record; ``subtract_interleave_pattern``
tiles and subtracts the pattern from the full record.  Multiple factors are
applied sequentially: each factor's pattern is estimated on the *residual*
pre-record after all previous factors have been subtracted.  The pre-record
is truncated to a multiple of M before averaging so that every phase
contributes the same number of samples.

Phase convention: sample index 0 of the record is phase 0.  The pre-record
occupies record indices 0..pre_samples-1, so its local indices equal the
global phases.  When pre_samples is a multiple of every interleave factor
(as is the case for the reference fixture where pre_samples = 1 600 000),
the frame data inherits the same phase alignment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class AcquisitionLayout:
    """Operator-supplied segment map for a segmented scope record.

    Attributes
    ----------
    pre_record_us : float
        Duration of the quiet pre-record segment, in microseconds.
    frame_period_us : float
        Repetition period of one frame, in microseconds.
    n_frames : int
        Number of frames in the record.
    frame : int or None
        Frame index (0-based) to use as the science FID, or ``None`` to
        coherently average all frames.
    keep_frames : bool
        When ``True``, preserve the per-frame array in the sliced result.
    """

    pre_record_us: float
    frame_period_us: float
    n_frames: int
    frame: Optional[int] = None
    keep_frames: bool = False

    def __post_init__(self) -> None:
        if self.pre_record_us < 0:
            raise ValueError("pre_record_us must be non-negative")
        if self.frame_period_us <= 0:
            raise ValueError("frame_period_us must be positive")
        if self.n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        if self.frame is not None and not (0 <= self.frame < self.n_frames):
            raise ValueError(
                f"frame index {self.frame} out of range for n_frames={self.n_frames}"
            )


@dataclass
class SlicedRecord:
    """Result of :func:`slice_record`.

    Attributes
    ----------
    science_fid : np.ndarray
        The science FID: either the coherent mean of all frames or a single
        selected frame.  Same length as one frame (``frame_samples``
        samples).
    pre_record : np.ndarray
        The quiet pre-record segment (``pre_samples`` samples).
    tail : np.ndarray
        Samples following the last frame (may be empty).
    frames : np.ndarray or None
        Per-frame 2-D array of shape ``(n_frames, frame_samples)``, present
        only when ``AcquisitionLayout.keep_frames`` was ``True``.  ``None``
        otherwise.
    layout : AcquisitionLayout
        The layout that produced this result.
    sample_dt : float
        Sample interval in seconds.
    pre_samples : int
        Number of pre-record samples.
    frame_samples : int
        Number of samples per frame.
    """

    science_fid: np.ndarray
    pre_record: np.ndarray
    tail: np.ndarray
    frames: Optional[np.ndarray]
    layout: AcquisitionLayout
    sample_dt: float
    pre_samples: int
    frame_samples: int


def estimate_interleave_pattern(quiet: np.ndarray, m: int) -> np.ndarray:
    """Estimate per-phase DC offsets for M interleaved ADCs.

    Parameters
    ----------
    quiet : np.ndarray
        1-D quiet pre-record segment (real-valued, any numeric dtype).
        The pre-record is assumed to start at record index 0, so its local
        sample indices are the global ADC phases.
    m : int
        Number of interleaved ADC phases.  Must be >= 1.

    Returns
    -------
    np.ndarray
        Length-``m`` float64 array of per-phase means.  Phase ``k``
        contains the mean of all pre-record samples whose index satisfies
        ``index % m == k``.  The pre-record is truncated to the largest
        multiple of ``m`` before averaging so every phase contributes the
        same number of samples.

    Raises
    ------
    ValueError
        If ``m < 1`` or the (truncated) quiet segment is empty.
    """
    if m < 1:
        raise ValueError(f"Interleave factor m must be >= 1, got {m}")
    quiet_f: np.ndarray = np.asarray(quiet, dtype=np.float64).ravel()
    n_full = (len(quiet_f) // m) * m
    if n_full == 0:
        raise ValueError(
            f"Quiet segment has {len(quiet_f)} samples, which is fewer than "
            f"the interleave factor m={m}."
        )
    q: np.ndarray = quiet_f[:n_full].reshape(-1, m)
    result: np.ndarray = q.mean(axis=0)
    return result


def subtract_interleave_pattern(record: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Subtract a tiled interleave-offset pattern from a record.

    Parameters
    ----------
    record : np.ndarray
        1-D sample record (real-valued, any numeric dtype).
    pattern : np.ndarray
        Length-``m`` per-phase offset array (output of
        :func:`estimate_interleave_pattern`).

    Returns
    -------
    np.ndarray
        Float64 copy of ``record`` with the tiled pattern subtracted.
        The phase of sample ``i`` is ``i % m``.
    """
    rec: np.ndarray = np.asarray(record, dtype=np.float64).ravel()
    m = len(pattern)
    if m == 0:
        return rec
    phases: np.ndarray = np.arange(len(rec)) % m
    out: np.ndarray = rec - pattern[phases]
    return out


def apply_interleave_cleanup(
    record: np.ndarray,
    quiet: np.ndarray,
    factors: List[int],
) -> tuple:
    """Apply sequential interleave-offset cleanup to a full record.

    For each factor ``m`` in ``factors``:

    1. Estimate the per-phase pattern from the *residual* pre-record
       (after all previous factors have been subtracted).
    2. Subtract the tiled pattern from the *full* record.

    Parameters
    ----------
    record : np.ndarray
        Full 1-D scope record in its original units (e.g. raw LSB).
    quiet : np.ndarray
        Pre-record segment (first ``len(quiet)`` samples of ``record``).
        Must equal ``record[:len(quiet)]`` before this call.
    factors : list of int
        Interleave factors to apply, in order.

    Returns
    -------
    cleaned_record : np.ndarray
        Float64 record with all patterns subtracted.
    patterns : dict[int, np.ndarray]
        Mapping from each factor to its estimated pattern.  The pattern
        for factor ``m`` was estimated on the residual pre-record *after*
        all preceding factors were subtracted.
    """
    record = np.asarray(record, dtype=np.float64).ravel()
    residual_quiet = np.asarray(quiet, dtype=np.float64).ravel().copy()
    patterns: Dict[int, np.ndarray] = {}

    for m in factors:
        pattern = estimate_interleave_pattern(residual_quiet, m)
        patterns[m] = pattern
        record = subtract_interleave_pattern(record, pattern)
        # Update the residual quiet for the next factor's estimation
        residual_quiet = subtract_interleave_pattern(residual_quiet, pattern)

    return record, patterns


def slice_record(
    record: np.ndarray,
    layout: AcquisitionLayout,
    sample_dt: float,
) -> SlicedRecord:
    """Slice a raw scope record into its acquisition segments.

    Parameters
    ----------
    record : np.ndarray
        1-D raw voltage record (real-valued, any numeric dtype).
    layout : AcquisitionLayout
        Operator-supplied segment map.
    sample_dt : float
        Sample interval in seconds (reciprocal of the sample rate).

    Returns
    -------
    SlicedRecord
        The segmented record with the science FID, pre-record, tail, and
        optionally the full per-frame array.

    Raises
    ------
    ValueError
        If the segment map does not fit within the record.
    """
    record = np.asarray(record, dtype=np.float64).ravel()
    n_total = len(record)

    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")

    # Convert µs durations to sample counts
    pre_samples = int(round(layout.pre_record_us * 1e-6 / sample_dt))
    frame_samples = int(round(layout.frame_period_us * 1e-6 / sample_dt))

    # Validate that the layout fits
    total_frame_samples = layout.n_frames * frame_samples
    required = pre_samples + total_frame_samples
    if required > n_total:
        raise ValueError(
            f"Segment map requires {required} samples "
            f"(pre={pre_samples}, frames={layout.n_frames}×{frame_samples}) "
            f"but record has only {n_total} samples."
        )

    # Slice segments
    pre_record = record[:pre_samples]
    tail = record[required:]

    # Build per-frame array
    frame_array: np.ndarray = np.empty(
        (layout.n_frames, frame_samples), dtype=np.float64
    )
    for k in range(layout.n_frames):
        start = pre_samples + k * frame_samples
        frame_array[k] = record[start : start + frame_samples]

    # Compute the science FID
    if layout.frame is None:
        science_fid = frame_array.mean(axis=0)
    else:
        science_fid = frame_array[layout.frame].copy()

    return SlicedRecord(
        science_fid=science_fid,
        pre_record=pre_record.copy(),
        tail=tail.copy(),
        frames=frame_array if layout.keep_frames else None,
        layout=layout,
        sample_dt=sample_dt,
        pre_samples=pre_samples,
        frame_samples=frame_samples,
    )
