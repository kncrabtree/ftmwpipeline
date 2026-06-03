"""Unit tests for the Stage 3 gap-pass matched-filter helpers.

Covers the shape-aware matched window in :func:`_mf_gap_spectrum` (exp vs
Gaussian) and its white-noise gain, plus the analytic σ propagation that
replaces a third scatter estimate on the gap spectrum.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import scipy.signal as spsig

from ftmwpipeline._internal.stage3_impl import (
    _mf_gap_spectrum,
    _primary_active_spectrum,
    _propagate_active_sigma_to_grid,
)


def _fake_fid(n=4000, dt_us=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        spacing=dt_us * 1e-6,  # seconds (code multiplies by 1e6 -> µs)
        data=rng.normal(0.0, 1.0, n),
        sideband="upper",
        probe_freq_mhz=8000.0,
    )


def _base_pp(start_us=2.0, end_us=70.0):
    return SimpleNamespace(start_us=start_us, end_us=end_us)


def _expected_gain(n_active, dt_us, tau, shape):
    t = np.arange(n_active) * dt_us
    w = np.exp(-((t / tau) ** 2)) if shape == "gaussian" else np.exp(-t / tau)
    return float(np.sqrt(np.sum(w * w) / n_active))


@pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
def test_gain_matches_window_and_is_below_one(shape):
    fid = _fake_fid()
    base_pp = _base_pp()
    dt_us = fid.spacing * 1e6
    n_active = int(round(base_pp.end_us / dt_us)) - int(round(base_pp.start_us / dt_us))
    tau = 3.0
    _, gain = _mf_gap_spectrum(fid, base_pp, None, tau_basis_us=tau, shape=shape)
    assert gain == pytest.approx(_expected_gain(n_active, dt_us, tau, shape), rel=1e-9)
    # A decaying window concentrates energy -> gain < 1 (boxcar limit is 1).
    assert 0.0 < gain < 1.0


def test_shape_changes_spectrum_and_gain():
    fid = _fake_fid()
    base_pp = _base_pp()
    cft_l, gain_l = _mf_gap_spectrum(fid, base_pp, None, tau_basis_us=3.0, shape="lorentzian")
    cft_g, gain_g = _mf_gap_spectrum(fid, base_pp, None, tau_basis_us=3.0, shape="gaussian")
    assert cft_l.freq_array.shape == cft_g.freq_array.shape
    # The two matched windows produce materially different spectra and gains.
    assert gain_l != pytest.approx(gain_g, rel=1e-3)
    assert not np.allclose(cft_l.magnitude_spectrum, cft_g.magnitude_spectrum)


def test_primary_active_spectrum_gain_matches_window():
    fid = _fake_fid()
    base_pp = _base_pp()
    dt_us = fid.spacing * 1e6
    n_active = int(round(base_pp.end_us / dt_us)) - int(round(base_pp.start_us / dt_us))
    cft, gain = _primary_active_spectrum(
        fid, base_pp, None, window_function="blackmanharris"
    )
    w = spsig.get_window("blackmanharris", n_active)
    assert gain == pytest.approx(float(np.sqrt(np.sum(w * w) / n_active)), rel=1e-9)
    assert 0.0 < gain < 0.6  # Blackman-Harris concentrates energy heavily
    assert cft.freq_array.shape == cft.magnitude_spectrum.shape


def test_primary_and_gap_share_active_grid():
    # Same active region + same active zpf -> identical detection grid, so the
    # primary and gap spectra are co-registered on the active FT.
    fid = _fake_fid()
    base_pp = _base_pp()
    prim, _ = _primary_active_spectrum(
        fid, base_pp, None, window_function="blackmanharris", zpf_active=2
    )
    gap, _ = _mf_gap_spectrum(fid, base_pp, None, tau_basis_us=3.0, zpf_active=2)
    assert np.array_equal(prim.freq_array, gap.freq_array)


def test_propagation_boxcar_identity():
    # gain=1 and same grid -> propagation reproduces the input σ exactly.
    freq = np.linspace(26500.0, 40000.0, 500)
    sigma = np.linspace(1.0, 2.0, 500)
    out = _propagate_active_sigma_to_grid(freq, sigma, freq, 1.0)
    assert np.allclose(out, sigma)


def test_propagation_applies_gain():
    freq = np.linspace(26500.0, 40000.0, 500)
    sigma = np.full(500, 3.0)
    out = _propagate_active_sigma_to_grid(freq, sigma, freq, 0.5)
    assert np.allclose(out, 1.5)


def test_propagation_descending_axis_interpolates_correctly():
    # 2638 has a descending frequency axis; the helper must handle it.
    asc_freq = np.linspace(26500.0, 40000.0, 1000)
    asc_sigma = 1.0 + (asc_freq - 26500.0) / 13500.0  # 1.0 -> 2.0 linearly
    desc_freq = asc_freq[::-1]
    desc_sigma = asc_sigma[::-1]
    # Query a finer ascending grid; expect the same linear σ(f) recovered.
    gap = np.linspace(27000.0, 39000.0, 333)
    out_from_desc = _propagate_active_sigma_to_grid(desc_freq, desc_sigma, gap, 1.0)
    expected = 1.0 + (gap - 26500.0) / 13500.0
    assert np.allclose(out_from_desc, expected, atol=1e-3)
