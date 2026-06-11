"""Unit tests for the Stage 5 spur detector and joint gate."""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.spur_detection import (
    GatedSpur,
    SpurMaskSpec,
    SpurSet,
    build_spur_set,
    detect_active_ft_spurs,
    gate_spurs,
)
from ftmwpipeline.fitting.tau_calibration import SpurCluster

# Active-FT bin spacing ~79 kHz on 2638; use it so integer-MHz centers land
# on a grid bin within the detector's integer tolerance.
SPACING = 0.079
BAND = (26500.0, 40000.0)


def _grid(centers_and_widths, *, lo=26500.0, hi=40000.0, noise=1.0):
    """Build a synthetic sorted active-FT with the given features.

    ``centers_and_widths`` is a list of ``(center_mhz, peak_mag, fwhm_mhz)``.
    A ``fwhm_mhz`` of 0 makes a single-bin (spur-like) spike; a positive
    fwhm makes a Gaussian skirt (line-like). The grid is built at the 2638
    active-FT spacing over a band tight around the features so the integer
    sweep stays fast.
    """
    centers = [c for c, _, _ in centers_and_widths]
    glo = min(min(centers) - 5.0, lo)
    ghi = max(max(centers) + 5.0, glo + 10.0)
    freqs = np.arange(glo, ghi, SPACING)
    spec = np.zeros(freqs.size, dtype=np.complex128)
    for center, peak, fwhm in centers_and_widths:
        if fwhm <= 0.0:
            j = int(np.argmin(np.abs(freqs - center)))
            spec[j] += peak
        else:
            sigma = fwhm / 2.355
            spec += peak * np.exp(-0.5 * ((freqs - center) / sigma) ** 2)
    sig_c = np.full(freqs.size, noise, dtype=float)
    return freqs, spec, sig_c


def _band_for(*centers):
    return (min(centers) - 5.0, max(centers) + 5.0)


# ---------------------------------------------------------------------------
# Frequency-domain detector
# ---------------------------------------------------------------------------
def test_narrow_integer_spur_detected():
    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert len(spurs) == 1
    assert spurs[0].integer_mhz == 30720
    assert spurs[0].narrowness_ratio < 0.30


def test_broad_integer_line_spared():
    # A real molecular line sitting exactly at an integer MHz, with a skirt:
    # the narrowness gate must NOT flag it (zero-false-positive guard).
    freqs, spec, sig_c = _grid([(32960.0, 100.0, 0.4)])
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert spurs == []


def test_narrow_noninteger_feature_rejected():
    # Narrow but off an integer MHz: the integer gate rejects it.
    freqs, spec, sig_c = _grid([(33421.1, 100.0, 0.0)])
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert spurs == []


def test_low_snr_integer_spike_rejected():
    freqs, spec, sig_c = _grid([(30720.0, 3.0, 0.0)], noise=1.0)
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND, snr_threshold=5.0)
    assert spurs == []


# ---------------------------------------------------------------------------
# Joint gate
# ---------------------------------------------------------------------------
def test_gate_union_of_narrow_and_saturated():
    narrow = detect_active_ft_spurs(*_grid([(30720.0, 100.0, 0.0)]), band=BAND)
    # A split-bin spur the frequency test missed, caught by the flat catalogue.
    sat = SpurCluster(
        center_freq_mhz=39040.0,
        peak_bin_index=10,
        n_bins=3,
        bin_indices=(9, 10, 11),
        saturated=True,
    )
    gated = gate_spurs(narrow, [sat])
    centers = sorted(g.integer_mhz for g in gated)
    assert centers == [30720, 39040]


def test_gate_rejects_unsaturated_integer_cluster():
    # An integer-MHz cls==1 cluster that is NOT saturated (the erratic
    # beat/blend false positive) must not enter the gated set.
    sat = SpurCluster(
        center_freq_mhz=38744.0,
        peak_bin_index=10,
        n_bins=1,
        bin_indices=(10,),
        saturated=False,
    )
    assert gate_spurs([], [sat]) == []


def test_gate_rejects_noninteger_saturated_cluster():
    sat = SpurCluster(
        center_freq_mhz=33421.1,
        peak_bin_index=10,
        n_bins=1,
        bin_indices=(10,),
        saturated=True,
    )
    assert gate_spurs([], [sat]) == []


def test_gate_merges_agreeing_detections():
    narrow = detect_active_ft_spurs(*_grid([(35840.0, 100.0, 0.0)]), band=BAND)
    sat = SpurCluster(
        center_freq_mhz=35840.0,
        peak_bin_index=10,
        n_bins=3,
        bin_indices=(9, 10, 11),
        saturated=True,
    )
    gated = gate_spurs(narrow, [sat])
    assert len(gated) == 1
    assert gated[0].source == "narrow+saturated"


# ---------------------------------------------------------------------------
# SpurSet / mask geometry
# ---------------------------------------------------------------------------
def test_build_spur_set_and_window_mask():
    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    spur_set = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        mask_half_width_bins=2,
    )
    assert bool(spur_set)
    # Window straddling the spur.
    spec_mask = spur_set.window_mask_spec(
        30715.0,
        30725.0,
        center_mhz=30720.0,
        sideband=Sideband.LOWER,
    )
    assert spec_mask is not None
    # The spur center maps to ~offset 0 in this window (within one bin of
    # the window center, which we set to the spur frequency).
    assert any(abs(o) < SPACING for o in spec_mask.offsets_mhz)
    # +/-2 bins masked: build a uniform offset grid at the active-FT spacing.
    spacing = spur_set.bin_spacing_mhz
    grid = np.arange(-5, 6) * spacing
    mask = spec_mask.bin_mask(grid)
    # center bin +/-2 = 5 masked bins.
    assert int(mask.sum()) == 5


def test_window_mask_none_when_no_spur_in_range():
    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    spur_set = build_spur_set(freqs, spec, sig_c, band=BAND)
    spec_mask = spur_set.window_mask_spec(
        31000.0,
        31010.0,
        center_mhz=31005.0,
        sideband=Sideband.LOWER,
    )
    assert spec_mask is None


def test_candidate_on_spur_tolerance():
    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    spur_set = build_spur_set(freqs, spec, sig_c, band=BAND)
    assert spur_set.candidate_on_spur(30720.02)
    assert not spur_set.candidate_on_spur(30720.5)


def test_empty_spur_set_is_falsey():
    freqs, spec, sig_c = _grid([(32960.0, 100.0, 0.4)])  # broad line, no spur
    spur_set = build_spur_set(freqs, spec, sig_c, band=BAND)
    assert not bool(spur_set)
    assert (
        spur_set.window_mask_spec(
            32955.0,
            32965.0,
            center_mhz=32960.0,
            sideband=Sideband.LOWER,
        )
        is None
    )


def test_use_stft_catalogue_false_drops_flat_only_spurs():
    sat = SpurCluster(
        center_freq_mhz=39040.0,
        peak_bin_index=10,
        n_bins=3,
        bin_indices=(9, 10, 11),
        saturated=True,
    )
    freqs, spec, sig_c = _grid([(32960.0, 100.0, 0.4)])  # no narrow spur
    spur_set = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        saturated_clusters=[sat],
        use_stft_catalogue=False,
    )
    assert not bool(spur_set)


def test_mask_spec_bin_mask_empty_offsets():
    spec = SpurMaskSpec(offsets_mhz=(), half_width_mhz=0.16)
    grid = np.linspace(-1, 1, 21)
    assert spec.bin_mask(grid).sum() == 0


# ---------------------------------------------------------------------------
# Time-domain decay probe + arbitration
# ---------------------------------------------------------------------------
PROBE_FREQ = 40960.0


def _synthetic_fid(tones, *, dt_us=0.002, n=7500, noise=0.001, seed=7):
    """Real FID with the given ``(f_mol_mhz, amplitude, tau_us)`` tones.

    ``tau_us = 0`` makes a non-decaying CW tone. Lower-sideband convention
    (``f_bb = PROBE_FREQ - f_mol``), matching the fixtures.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt_us
    x = rng.normal(0.0, noise, size=n)
    for f_mol, amp, tau in tones:
        f_bb = PROBE_FREQ - f_mol
        env = np.exp(-t / tau) if tau > 0 else 1.0
        x = x + amp * env * np.cos(2 * np.pi * f_bb * t + 0.3)
    return x, dt_us, n * dt_us


def _probe_for(tones, **kwargs):
    from ftmwpipeline.fitting.spur_detection import make_decay_probe

    x, dt_us, dur = _synthetic_fid(tones, **kwargs)
    return make_decay_probe(
        x,
        dt_us,
        start_us=0.0,
        end_us=dur,
        probe_freq_mhz=PROBE_FREQ,
        sideband=Sideband.LOWER,
    )


def test_decay_probe_separates_line_from_cw_tone():
    probe = _probe_for([(30720.0, 1.0, 0.0), (33421.1, 1.0, 4.0)])
    ratio_cw, snr_cw = probe(30720.0)
    ratio_line, snr_line = probe(33421.1)
    assert ratio_cw > 0.8
    assert snr_cw > 10.0
    assert ratio_line < 0.6
    assert snr_line > 10.0


def test_gate_vetoes_narrow_nominee_that_decays():
    # A decaying line that happens to sit at an integer MHz and pass the
    # narrowness test: the probe sees it decay and vetoes the gate.
    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    narrow = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert narrow
    probe = _probe_for([(30720.0, 1.0, 4.0)])
    assert gate_spurs(narrow, [], decay_probe=probe) == []
    # Flat probe keeps it gated.
    probe_flat = _probe_for([(30720.0, 1.0, 0.0)])
    gated = gate_spurs(narrow, [], decay_probe=probe_flat)
    assert [g.integer_mhz for g in gated] == [30720]


def test_gate_accepts_probe_confirmed_flat_cluster_off_integer():
    # A non-integer, NON-saturated cluster: legacy gate drops it, but a
    # probe-confirmed flat tone is gated with source "flat".
    cl = SpurCluster(
        center_freq_mhz=33421.1,
        peak_bin_index=10,
        n_bins=1,
        bin_indices=(10,),
        saturated=False,
    )
    probe = _probe_for([(33421.1, 1.0, 0.0)])
    gated = gate_spurs([], [cl], decay_probe=probe)
    assert len(gated) == 1
    assert gated[0].source == "flat"
    assert abs(gated[0].center_mhz - 33421.1) < 1e-9


def _pair_split_grid(f_int=35840.0):
    """A CW tone between two bins: power split across the pair, sinc-level
    second neighbours (the measured 655 35840 profile)."""
    freqs = np.arange(f_int - 5.0, f_int + 5.0, SPACING)
    spec = np.zeros(freqs.size, dtype=np.complex128)
    k = int(np.argmin(np.abs(freqs - f_int)))
    spec[k] = 100.0
    spec[k - 1] = 65.0
    spec[k - 2] = 25.0
    spec[k + 1] = 27.0
    sig_c = np.full(freqs.size, 1.0)
    return freqs, spec, sig_c, k


def test_pair_split_tone_nominated_as_pair():
    # Split power defeats the single-bin test (neighbour ratio 0.65), but
    # the two-bin pair towers over its second neighbours.
    freqs, spec, sig_c, k = _pair_split_grid()
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert len(spurs) == 1
    assert spurs[0].pair
    assert spurs[0].bin_index == k
    assert spurs[0].narrowness_ratio <= 0.30


def test_pair_nominee_requires_flat_probe_to_gate():
    freqs, spec, sig_c, _ = _pair_split_grid()
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert spurs and spurs[0].pair
    # Without a probe the ambiguous frequency-domain signature is never
    # gated (a blended doublet looks identical).
    assert gate_spurs(spurs, []) == []
    # A decaying probe (a real blend) keeps it out.
    assert gate_spurs(spurs, [], decay_probe=lambda f: (0.2, 50.0)) == []
    # Probe-confirmed flatness gates it with the pair provenance.
    gated = gate_spurs(spurs, [], decay_probe=lambda f: (1.0, 50.0))
    assert [g.source for g in gated] == ["narrow-pair"]
    # Flat but below the probe-SNR floor: still skipped.
    assert gate_spurs(spurs, [], decay_probe=lambda f: (1.0, 2.0)) == []


def test_blended_doublet_not_pair_nominated():
    # Two real lines ~3 bins apart around an integer MHz: the bin flanking
    # the pair carries line wings, so the pair test fails too.
    freqs = np.arange(35835.0, 35845.0, SPACING)
    spec = np.zeros(freqs.size, dtype=np.complex128)
    k = int(np.argmin(np.abs(freqs - 35840.0)))
    spec[k] = 100.0
    spec[k - 1] = 110.0
    spec[k - 2] = 59.0  # the 655 39040 left-smear profile
    spec[k + 1] = 11.0
    sig_c = np.full(freqs.size, 1.0)
    spurs = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    assert spurs == []


def test_gate_rejects_saturated_cluster_that_decays():
    # The bare saturated flag is not trusted when the probe sees decay
    # (measured false positives on strong erratic lines).
    cl = SpurCluster(
        center_freq_mhz=33421.1,
        peak_bin_index=10,
        n_bins=1,
        bin_indices=(10,),
        saturated=True,
    )
    probe = _probe_for([(33421.1, 1.0, 4.0)])
    assert gate_spurs([], [cl], decay_probe=probe) == []


# ---------------------------------------------------------------------------
# fit_window spur-mask integration (n_data reduction + bin exclusion)
# ---------------------------------------------------------------------------
def test_fit_window_spur_mask_reduces_n_data_and_chi2():
    from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum
    from ftmwpipeline.fitting.window_fit import fit_window

    acq = 12.65
    tau = 4.0
    u = np.linspace(-0.5, 0.5, 41)
    peak = ModelPeak(amplitude=1.0, offset_mhz=0.0, phase=0.2)
    clean = model_spectrum(u, [peak], tau, acq)
    # Inject a CW spur as a large single-bin spike away from the line center.
    spur_bin = 33
    z = clean.copy()
    z[spur_bin] += 50.0 + 0.0j
    sigma = np.full(u.size, 0.02)

    init = [ModelPeak(amplitude=0.9, offset_mhz=0.0, phase=0.0)]
    unmasked = fit_window(u, z, sigma, init, tau, acq)
    spec = SpurMaskSpec(
        offsets_mhz=(float(u[spur_bin]),),
        half_width_mhz=float(np.median(np.diff(u))) * 0.5,
    )
    masked = fit_window(u, z, sigma, init, tau, acq, spur_mask=spec)

    # One bin masked -> n_data drops by 2 (Re + Im).
    assert masked.n_data == unmasked.n_data - 2
    # The spur dominates the unmasked chi^2; masking it collapses chi^2.
    assert masked.chi_squared < unmasked.chi_squared / 100.0
    # The fitted/residual arrays stay full-length (model evaluated everywhere).
    assert masked.fitted_spectrum.size == u.size
    assert masked.residual.size == u.size


def test_fit_window_null_model_respects_mask():
    from ftmwpipeline.fitting.window_fit import fit_window

    acq = 12.65
    u = np.linspace(-0.5, 0.5, 21)
    z = np.zeros(u.size, dtype=np.complex128)
    z[10] += 100.0  # lone spur
    sigma = np.full(u.size, 0.02)
    spec = SpurMaskSpec(
        offsets_mhz=(float(u[10]),),
        half_width_mhz=float(np.median(np.diff(u))) * 0.5,
    )
    null_masked = fit_window(u, z, sigma, [], 4.0, acq, spur_mask=spec)
    assert null_masked.n_data == 2 * (u.size - 1)
    assert null_masked.chi_squared < 1e-6
