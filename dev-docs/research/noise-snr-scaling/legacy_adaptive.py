"""Minimal legacy *level-based* noise estimator -- comparison reference only.

This is the stripped-to-the-essence form of the retired Stage 2 ``adaptive``
estimator (MAD/median subdivision + skewness-trimmed noise mask + moving
window-RMS + Lorentzian-skirt exclusion). The production pipeline has moved
entirely to the high-pass ``scatter`` estimator (see ``prototype.py`` and
``report.md``); this file exists solely so the report's central comparison --
*level-based vs high-pass* -- stays runnable without resurrecting the bulky
production architecture.

The defining property the report contrasts: a level-based estimator tracks the
magnitude *envelope*, so on line-dense / high-leakage bands it rides up with the
leakage pedestal (it "flattens", over-estimating sigma), whereas the high-pass
scatter estimator measures the residual scatter and stays on the true floor. A
plain moving median of ``|X|`` reproduces that envelope-tracking behaviour; the
full subdivision/skewness/skirt machinery only sharpened it at the margins.

Not imported by the package. Run against the report's fixtures for the
level-vs-high-pass figures.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

# Rayleigh(sigma_c): median(|X|) = sigma_c * sqrt(ln 4); sigma_x = sigma_c*sqrt(2)
# => sigma_x = median(|X|) * sqrt(2 / ln 4) = median(|X|) / sqrt(ln 2).
_RAYLEIGH_MEDIAN_TO_SIGMA_X = 1.0 / np.sqrt(np.log(2.0))


def estimate_sigma_adaptive(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    smoothing_window_mhz: float = 300.0,
) -> np.ndarray:
    """Per-bin sigma_x from a moving median of ``|X|`` (level-based envelope).

    A broad running median of the magnitude, converted to the complex-RMS
    sigma_x via the Rayleigh median factor. Envelope-tracking by construction:
    where leakage skirts lift the local magnitude, the median -- and hence the
    reported sigma -- lifts with it.
    """
    mag = np.abs(np.asarray(magnitudes, dtype=float))
    if mag.size < 2:
        return mag * _RAYLEIGH_MEDIAN_TO_SIGMA_X
    df = abs(float(frequencies[1] - frequencies[0]))
    width = max(3, int(round(smoothing_window_mhz / df)))
    if width % 2 == 0:
        width += 1
    median = median_filter(mag, size=width, mode="nearest")
    return median * _RAYLEIGH_MEDIAN_TO_SIGMA_X
