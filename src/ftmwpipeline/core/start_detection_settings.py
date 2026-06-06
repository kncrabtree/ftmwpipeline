"""Settings for data-driven FID start-time detection.

``StartDetectionSettings`` bundles the knobs for
:func:`ftmwpipeline.preprocessing.start_detection.detect_start_time`. Unlike
the Stage 2b / Stage 5 settings, this is a flat frozen dataclass with concrete
hard defaults (not an ``Optional``-everywhere resolution-chain bundle): start
detection is a pre-Stage-1 recommender invoked explicitly, and its only
persisted output is the recommended ``start_us`` it stamps into the Stage 0
``recommended_processing`` layer -- the knobs themselves are not persisted.

The defaults are tuned on the Blackchirp 2638-family instrument (LO 40960 MHz,
lower sideband). ``guard_margin_us`` in particular is instrument-specific (the
switch-bounce ringdown length past the chirp); expose it for re-tuning on other
instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StartDetectionSettings:
    """Knobs for the Σ|FT|-vs-start_us start-time detector.

    Parameters
    ----------
    sweep_max_us :
        Upper bound of the start-time sweep (capped to the FID duration).
        Must clear the chirp end plus the post-chirp molecular tail used to
        estimate the floor.
    step_us :
        Sweep step. Finer resolves the chirp-end corner more precisely at
        linear cost.
    floor_factor :
        Chirp-end is the first start where Σ|FT| falls below
        ``floor_factor * floor`` (the deep-tail floor). The chirp collapse is
        ~2-3 decades, so any factor in a few-x of the floor lands on the same
        corner.
    floor_tail_us :
        Width of the deep-tail window (at the end of the sweep) used for the
        robust floor estimate.
    guard_margin_us :
        Added to the detected chirp-end to clear the switch-bounce ringdown.
        The primary recommendation is ``chirp_end + guard_margin_us``.
        Instrument-specific.
    min_chirp_drop_ratio :
        Minimum plateau/floor ratio for the chirp collapse to be considered
        present. Below this the start time cannot be inferred from the data
        (no excitation transient in the recorded FID).
    band_min_mhz, band_max_mhz :
        Optional explicit integration band override. When unset, the impl
        resolves the band from the canonical Stage 1 frequency trim, falling
        back to the full positive spectrum.
    """

    sweep_max_us: float = 7.5
    step_us: float = 0.02
    floor_factor: float = 3.0
    floor_tail_us: float = 1.0
    guard_margin_us: float = 0.67
    min_chirp_drop_ratio: float = 10.0
    band_min_mhz: Optional[float] = None
    band_max_mhz: Optional[float] = None


__all__ = ["StartDetectionSettings"]
