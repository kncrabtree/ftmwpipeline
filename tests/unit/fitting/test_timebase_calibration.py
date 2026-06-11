"""Unit tests for the scope-timebase self-calibration engine.

The synthetic record injects Rb-locked lattice tones at ``k*g*(1+eps_true)``
baseband plus contaminants -- a decaying molecular tone near an in-band
lattice point and an off-nominal kHz-offset tone at the lattice fundamental --
and asserts the estimator recovers ``eps_true`` within ~0.1 ppm, rejects the
off-nominal tone, and reports drift controls when an unlocked clock is
declared.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline.fitting.timebase_calibration import (
    calibrate_timebase_from_fid,
)

G = 320.0  # MHz lattice fundamental (gcd of the locked clocks below)
EPS_TRUE = 2.0e-6
DT_US = 2.0e-5  # 50 GSa/s  (Nyquist = 25 GHz baseband)
N = 200_000


def _synthetic_record(
    *,
    eps: float = EPS_TRUE,
    seed: int = 0,
    off_nominal_offset_mhz: float = -5.0e-3,
    add_molecular: bool = True,
    add_unlocked: bool = True,
) -> np.ndarray:
    """Build a synthetic FID with locked tones, a molecular line, and a spur.

    Three strong CW lattice tones (k=10, 30, 56 => 3.2, 9.6, 17.92 GHz
    baseband) are scaled by ``(1 + eps)``. A decaying ``exp(-t/tau)`` molecular
    tone sits in-band but off the lattice. One off-nominal tone at the
    fundamental ``G`` (k=1) carries a fixed kHz offset that is NOT eps-scaled.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT_US  # microseconds
    x = np.zeros(N, dtype=float)

    # Strong locked lattice tones (high SNR), scaled by (1 + eps). Several
    # tones so the eps-consistent majority outvotes the off-nominal spur in
    # the shared-eps rejection, as it does on real records (dozens of tones).
    for k, amp in ((10, 5.0), (20, 4.5), (30, 4.0), (40, 5.5), (56, 6.0)):
        f = k * G * (1.0 + eps)  # MHz
        x += amp * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    # Decaying "molecular" tone in-band but OFF the lattice (3360 is 10.5*G),
    # so it cannot masquerade as a lattice tone but still tests robustness.
    if add_molecular:
        tau = 1.0  # us
        f_mol = 3360.0
        x += (
            8.0
            * np.exp(-t / tau)
            * np.cos(2 * np.pi * f_mol * t + rng.uniform(0, 2 * np.pi))
        )

    # Off-nominal tone at the fundamental G with a fixed (non-eps) kHz offset.
    f_off = G + off_nominal_offset_mhz
    x += 3.5 * np.cos(2 * np.pi * f_off * t + rng.uniform(0, 2 * np.pi))

    # Unlocked-clock (scope, 6250 MHz) tone: at its exact nominal frequency,
    # i.e. NOT eps-scaled, the drift-control discriminant.
    if add_unlocked:
        x += 4.0 * np.cos(2 * np.pi * 6250.0 * t + rng.uniform(0, 2 * np.pi))

    # White noise.
    x += rng.normal(0.0, 0.05, size=N)
    return x


def test_recovers_eps_within_tolerance():
    x = _synthetic_record()
    result = calibrate_timebase_from_fid(
        x,
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=[5120.0, 5440.0],  # gcd = 320
    )
    assert result.lattice_g_mhz == pytest.approx(320.0)
    assert result.preconditions_passed
    assert result.n_used >= 3
    # Recovered eps within ~0.1 ppm of truth.
    assert abs(result.epsilon - EPS_TRUE) < 0.1e-6, (
        f"eps={result.epsilon * 1e6:+.4f} ppm vs truth " f"{EPS_TRUE * 1e6:+.4f} ppm"
    )


def test_off_nominal_tone_detected_but_rejected():
    x = _synthetic_record()
    result = calibrate_timebase_from_fid(
        x,
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=[5120.0, 5440.0],
    )
    # The fundamental k=1 tone (320 MHz) carries the -5 kHz non-eps offset.
    fund = [t for t in result.tone_reads if t.k == 1 and not t.drift_control]
    assert fund, "fundamental lattice tone was not detected at all"
    tone = fund[0]
    assert tone.snr >= result.snr_min  # detected
    assert not tone.used  # rejected by shared-eps consistency
    # Its measured offset is the planted non-eps kHz offset, not eps*f.
    assert tone.df_mhz < -1.0e-3


def test_drift_controls_reported():
    x = _synthetic_record()
    result = calibrate_timebase_from_fid(
        x,
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=[5120.0, 5440.0],
        unlocked_freqs_mhz=[6250.0],  # scope clock
    )
    controls = [t for t in result.tone_reads if t.drift_control]
    assert controls, "no drift controls reported despite unlocked clock"
    # Controls never enter the eps fit.
    assert all(not t.used for t in controls)
    # The injected 6250 control sits at its exact nominal frequency, so it
    # reads offset ~0 (NOT eps*f) -- the drifting-family discriminant.
    ctrl_6250 = [t for t in controls if t.k == 1]
    assert ctrl_6250, "the 6250 MHz drift control was not detected"
    assert abs(ctrl_6250[0].df_mhz) < 5.0e-3  # not eps*6250 (~12.5 kHz)


def test_precondition_no_locked_clocks():
    x = _synthetic_record(add_molecular=False)
    result = calibrate_timebase_from_fid(
        x,
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=[],
    )
    assert not result.preconditions_passed
    assert result.lattice_g_mhz == 0.0
    assert result.n_used == 0
    assert any("no locked clocks" in n for n in result.preconditions_notes)


def test_precondition_too_few_tones():
    # Only noise + the single off-nominal tone: no clean lattice tones survive
    # the shared-eps consistency fit, so < 3 tones remain.
    rng = np.random.default_rng(3)
    t = np.arange(N) * DT_US
    x = 3.5 * np.cos(2 * np.pi * (G - 5.0e-3) * t) + rng.normal(0, 0.05, N)
    result = calibrate_timebase_from_fid(
        x,
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=[5120.0, 5440.0],
    )
    assert not result.preconditions_passed
    assert result.n_used < 3
