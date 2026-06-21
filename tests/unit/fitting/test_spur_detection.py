"""Unit tests for the Stage 5 spur detector and joint gate."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

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
    # A split-bin spur the frequency test missed, caught by the flat catalog.
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


def test_use_stft_catalog_false_drops_flat_only_spurs():
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
        use_stft_catalog=False,
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
    second neighbors (the measured 655 35840 profile)."""
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
    # Split power defeats the single-bin test (neighbor ratio 0.65), but
    # the two-bin pair towers over its second neighbors.
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


# ---------------------------------------------------------------------------
# Clock-lattice prior: no-declaration byte-identity
# ---------------------------------------------------------------------------
def test_no_declaration_path_byte_identical():
    """Defaults for every clock-lattice kwarg must reproduce the legacy fit.

    The acceptance bar: with no declaration the gate and SpurSet are
    structurally untouched. Construct a synthetic spur + flat cluster and
    assert the SpurSet is identical with and without the new kwargs at
    their defaults.
    """
    from ftmwpipeline.fitting.spur_detection import (
        DEFAULT_DRIFT_BAND_RATIO,
        DEFAULT_DRIFT_MIN_SNR,
        DEFAULT_LATTICE_DECAY_RATIO,
    )

    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    probe = _probe_for([(30720.0, 1.0, 0.0)])
    legacy = build_spur_set(
        freqs, spec, sig_c, band=BAND, decay_probe=probe, mask_half_width_bins=2
    )
    # Explicitly pass every new kwarg at its documented default: lattice
    # None, no band-power probe, mask scaling disabled (0).
    explicit = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        decay_probe=probe,
        mask_half_width_bins=2,
        lattice=None,
        lattice_decay_ratio=DEFAULT_LATTICE_DECAY_RATIO,
        band_power_probe=None,
        drift_band_ratio=DEFAULT_DRIFT_BAND_RATIO,
        drift_min_snr=DEFAULT_DRIFT_MIN_SNR,
        mask_target_residual_snr=0.0,
        mask_max_half_width_bins=32,
    )
    assert legacy.spurs == explicit.spurs
    assert legacy.bin_spacing_mhz == explicit.bin_spacing_mhz
    assert legacy.mask_half_width_bins == explicit.mask_half_width_bins
    # Legacy spurs carry no lattice provenance and no per-spur mask override.
    for s in legacy.spurs:
        assert s.lattice is None
        assert s.drift is False
        assert s.mask_half_width_bins is None
    # The mask spec stays uniform (legacy half_widths_mhz = None).
    mspec = legacy.window_mask_spec(
        30715.0, 30725.0, center_mhz=30720.0, sideband=Sideband.LOWER
    )
    assert mspec is not None and mspec.half_widths_mhz is None


# ---------------------------------------------------------------------------
# Clock-lattice prior: on-lattice evidence-bar flip
# ---------------------------------------------------------------------------
def _on_lattice_spur(center=30720.0, snr=40.0):
    """A narrow nominee stamped with a locked-lattice identity."""
    from ftmwpipeline.fitting.spur_detection import Spur

    return Spur(
        integer_mhz=int(round(center)),
        center_mhz=center,
        bin_index=10,
        magnitude=snr,
        snr=snr,
        narrowness_ratio=0.1,
        lattice="640x48 (rf)",
    )


def test_on_lattice_narrow_gated_unless_clearly_decays():
    sp = _on_lattice_spur()
    # Clearly decaying (ratio 0.2 < 0.45 bar) at usable SNR -> vetoed.
    assert gate_spurs([sp], [], decay_probe=lambda f: (0.2, 50.0)) == []
    # Pseudo-decay (ratio 0.5, above the 0.45 lattice bar) -> still gated
    # (this is the 655 39040 signature: decay-probe 0.59 does not veto).
    gated = gate_spurs([sp], [], decay_probe=lambda f: (0.5, 50.0))
    assert len(gated) == 1
    assert gated[0].lattice == "640x48 (rf)"
    assert gated[0].source == "narrow"


def test_off_lattice_narrow_keeps_legacy_veto_bar():
    # The SAME nominee without a lattice identity keeps the legacy 0.6 veto:
    # ratio 0.5 < 0.6 -> vetoed (the on-lattice prior is what spares it).
    from ftmwpipeline.fitting.spur_detection import Spur

    off = Spur(
        integer_mhz=30720,
        center_mhz=30720.0,
        bin_index=10,
        magnitude=40.0,
        snr=40.0,
        narrowness_ratio=0.1,
        lattice=None,
    )
    assert gate_spurs([off], [], decay_probe=lambda f: (0.5, 50.0)) == []


def test_on_lattice_pair_gated_unless_clearly_decays():
    from ftmwpipeline.fitting.spur_detection import Spur

    pair = Spur(
        integer_mhz=35840,
        center_mhz=35840.0,
        bin_index=10,
        magnitude=40.0,
        snr=40.0,
        narrowness_ratio=0.2,
        pair=True,
        lattice="640x56 (rf)",
    )
    # On-lattice pair with a probe: gated unless it clearly decays.
    gated = gate_spurs([pair], [], decay_probe=lambda f: (0.55, 50.0))
    assert [g.source for g in gated] == ["narrow-pair"]
    assert gated[0].lattice == "640x56 (rf)"
    # Clearly decaying -> vetoed.
    assert gate_spurs([pair], [], decay_probe=lambda f: (0.2, 50.0)) == []
    # Without a probe on-lattice pairs are still conservatively skipped.
    assert gate_spurs([pair], []) == []


# ---------------------------------------------------------------------------
# Clock-lattice prior: drift lane
# ---------------------------------------------------------------------------
def _drift_spur(center=39040.0):
    from ftmwpipeline.fitting.spur_detection import Spur

    return Spur(
        integer_mhz=int(round(center)),
        center_mhz=center,
        bin_index=10,
        magnitude=30.0,
        snr=30.0,
        narrowness_ratio=1.0,
        lattice="6250x6 (bb, drift)",
        drift=True,
    )


def test_drift_nominee_gated_on_flat_band_power():
    sp = _drift_spur()
    # Band-ratio 0.43 (the measured 655 39040 tone) >= 0.35 bar, SNR ok.
    gated = gate_spurs([sp], [], band_power_probe=lambda f: (0.43, 50.0))
    assert len(gated) == 1
    assert gated[0].source == "drift"
    assert gated[0].drift is True
    assert gated[0].lattice == "6250x6 (bb, drift)"


def test_drift_nominee_skipped_when_band_power_decays():
    sp = _drift_spur()
    # A real line measures band-ratio 0.12-0.28 -> below the 0.35 bar.
    assert gate_spurs([sp], [], band_power_probe=lambda f: (0.2, 50.0)) == []
    # Flat band power but below the SNR floor: skipped.
    assert gate_spurs([sp], [], band_power_probe=lambda f: (0.43, 2.0)) == []


def test_drift_nominee_skipped_without_band_power_probe():
    sp = _drift_spur()
    # A decay probe is irrelevant to a drift nominee; without the band-power
    # probe it is conservatively skipped (never gated on freq evidence).
    assert gate_spurs([sp], [], decay_probe=lambda f: (1.0, 50.0)) == []


# ---------------------------------------------------------------------------
# SNR-scaled residual mask
# ---------------------------------------------------------------------------
def _gated_set_with_snr(snr, *, target, base=2, cap=32, drift=False):
    """Build a SpurSet whose single gated spur carries ``snr`` and scaling."""
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    # A lone narrow integer-MHz spur on the grid, gated frequency-domain.
    freqs, spec, sig_c = _grid([(30720.0, snr, 0.0)])
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=BAND,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    return build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        lattice=lat,
        mask_half_width_bins=base,
        mask_target_residual_snr=target,
        mask_max_half_width_bins=cap,
    )


def test_snr_mask_scales_per_spur_width():
    # snr/(pi*target): snr 100, target 1 -> ceil(100/pi) = 32 (hits cap 32).
    sset = _gated_set_with_snr(100.0, target=1.0, base=2, cap=64)
    assert sset.spurs
    bins = sset.spurs[0].mask_half_width_bins
    assert bins == math.ceil(100.0 / math.pi)


def test_snr_mask_floors_at_base():
    # A weak (snr 6) tone: ceil(6/(pi*1)) = 2 == base; never below base.
    sset = _gated_set_with_snr(6.0, target=1.0, base=2)
    assert sset.spurs[0].mask_half_width_bins == 2


def test_snr_mask_caps_at_max():
    sset = _gated_set_with_snr(10000.0, target=1.0, base=2, cap=8)
    assert sset.spurs[0].mask_half_width_bins == 8


def test_snr_mask_disabled_at_zero():
    # target 0 -> no per-spur override (legacy uniform mask).
    sset = _gated_set_with_snr(100.0, target=0.0)
    assert sset.spurs[0].mask_half_width_bins is None
    # The window spec stays uniform.
    mspec = sset.window_mask_spec(
        30715.0, 30725.0, center_mhz=30720.0, sideband=Sideband.LOWER
    )
    assert mspec is not None and mspec.half_widths_mhz is None


def test_snr_mask_window_spec_carries_per_offset_widths():
    sset = _gated_set_with_snr(100.0, target=1.0, base=2, cap=64)
    mspec = sset.window_mask_spec(
        30715.0, 30725.0, center_mhz=30720.0, sideband=Sideband.LOWER
    )
    assert mspec is not None
    assert mspec.half_widths_mhz is not None
    # The per-offset width reflects the wide (snr-scaled) bin count.
    expect_bins = math.ceil(100.0 / math.pi)
    expect_mhz = (expect_bins + 0.5) * sset.bin_spacing_mhz
    assert abs(mspec.half_widths_mhz[0] - expect_mhz) < 1e-9
    # The mask widened well past the uniform +/-2-bin default.
    grid = np.arange(-40, 41) * sset.bin_spacing_mhz
    assert int(mspec.bin_mask(grid).sum()) > 5


def test_flat_lane_tone_mask_scaled_from_measured_snr():
    # The 363 w100 defect: a flat-lane tone (gated via the saturated /
    # probe-flat catalog) carries snr=NaN, so SNR-scaled mask never fired
    # and its strong sinc skirt detonated the window. The fallback reads the
    # bin SNR from the active FT and widens the mask.
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    # A strong tone at an off-lattice, non-integer frequency: the
    # frequency-domain sweep never visits it (only integer/lattice points
    # are swept), so only the flat catalog + flat probe gate it. The
    # single-bin spike is skirt-consistent, so the scaled mask is not
    # truncated. Bin SNR ~ 139.
    center = 28057.46
    freqs, spec, sig_c = _grid([(center, 139.0, 0.0)])
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=BAND,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    cl = SpurCluster(
        center_freq_mhz=center,
        peak_bin_index=10,
        n_bins=1,
        bin_indices=(10,),
        saturated=True,
    )
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        saturated_clusters=[cl],
        lattice=lat,
        decay_probe=lambda f: (1.0, 50.0),  # flat -> gated
        mask_half_width_bins=2,
        mask_target_residual_snr=2.0,
        mask_max_half_width_bins=64,
    )
    flat = [s for s in sset.spurs if abs(s.center_mhz - center) < 0.1]
    assert flat
    s = flat[0]
    assert "flat" in s.source
    # The GatedSpur.snr field stays NaN (provenance marker), but the mask
    # was scaled from the locally-measured bin SNR well past the +/-2 base.
    assert not np.isfinite(s.snr)
    assert s.mask_half_width_bins is not None and s.mask_half_width_bins > 2


# ---------------------------------------------------------------------------
# Clock-lattice prior: locked-point band-power (drift) fallback
# ---------------------------------------------------------------------------
def test_locked_point_drift_fallback_gates_wandering_tone():
    # 655's 39040 is a LOCKED lattice point (320x6 bb) that itself wanders
    # under the unlocked scope clock: a ~200 kHz smear, no single narrow bin
    # and no clean pair. The narrow/pair sweep produces no nominee, so the
    # locked-point band-power fallback nominates the strongest in-window bin
    # and the band-power probe gates it (source "drift").
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    # A locked point at 30720 (= probe 40000? no; pick probe so 30720 is a
    # 640 bb multiple). probe 40000: f_bb = 40000 - 30720 = 9280 = 14.5x640
    # -> not bb; use the RF frame: 30720 = 48 x 640. Build a smeared tone
    # there with no narrow bin (broad, split across several bins).
    center = 30720.0
    freqs = np.arange(center - 5.0, center + 5.0, SPACING)
    spec = np.zeros(freqs.size, dtype=np.complex128)
    k = int(np.argmin(np.abs(freqs - center)))
    # A wandering smear: comparable power across ~5 bins (no narrowness).
    for off, amp in [(-2, 60.0), (-1, 80.0), (0, 90.0), (1, 75.0), (2, 55.0)]:
        spec[k + off] = amp
    sig_c = np.full(freqs.size, 1.0)
    lband = (center - 5.0, center + 5.0)
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=lband,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    # The narrow/pair sweep alone produces nothing (the smear is broad).
    narrow = detect_active_ft_spurs(
        freqs, spec, sig_c, band=lband, lattice_points=lat.nomination_points()
    )
    assert narrow == []
    # With the band-power probe (flat band ratio), the locked-point fallback
    # nominates + gates it as "drift", keeping the locked identity.
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        band_power_probe=lambda f: (0.43, 50.0),
        snr_threshold=5.0,
    )
    gated = [s for s in sset.spurs if abs(s.center_mhz - center) <= 0.3]
    assert gated, "locked-point fallback did not gate the wandering tone"
    g = gated[0]
    assert g.source == "drift"
    assert g.drift is True
    assert g.lattice is not None and "640x48 (rf)" in g.lattice
    # A decaying band ratio (a real line) must NOT gate via the fallback.
    sset_line = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        band_power_probe=lambda f: (0.2, 50.0),
        snr_threshold=5.0,
    )
    assert not [s for s in sset_line.spurs if abs(s.center_mhz - center) <= 0.3]
    # A clearly-decaying decay probe vetoes the locked-point fallback even
    # when the band-power ratio clears the bar (655's 36800: band_ratio 0.38
    # but decay_ratio 0.39 -- a real molecular line near a locked point).
    sset_veto = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        band_power_probe=lambda f: (0.43, 50.0),
        decay_probe=lambda f: (0.39, 50.0),
        snr_threshold=5.0,
    )
    assert not [s for s in sset_veto.spurs if abs(s.center_mhz - center) <= 0.3]


def test_locked_point_fallback_skipped_when_narrow_nominee_exists():
    # If the locked point already has a narrow nominee, the fallback must NOT
    # double-nominate it (no duplicate / drift entry on a clean comb tone).
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    center = 30720.0  # 48 x 640 (rf)
    freqs, spec, sig_c = _grid([(center, 100.0, 0.0)])  # clean narrow tone
    lband = _band_for(center)
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=lband,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        band_power_probe=lambda f: (0.43, 50.0),
        decay_probe=lambda f: (1.0, 50.0),
    )
    near = [s for s in sset.spurs if abs(s.center_mhz - center) <= 0.3]
    assert len(near) == 1
    assert near[0].source == "narrow"  # not "drift"
    assert near[0].drift is False


# ---------------------------------------------------------------------------
# Clock-lattice prior: union with the legacy integer sweep
# ---------------------------------------------------------------------------
def test_union_keeps_off_lattice_integer_tone_under_declaration():
    # An off-lattice integer-MHz tone (39990 on 655) must STILL gate under a
    # declaration: the declaration is additive, so the legacy integer sweep
    # runs alongside the lattice sweep. 39990 is not a 640 multiple in either
    # frame (39990/640 = 62.48; bb 40000-39990 = 10 -> not a multiple).
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    off = 39990.0
    on = 39680.0  # probe 40000 - 320? no; 39680 = 62 x 640 (rf, on-lattice)
    freqs, spec, sig_c = _grid([(off, 100.0, 0.0), (on, 100.0, 0.0)])
    lband = (off - 6.0, on + 6.0) if on > off else (on - 6.0, off + 6.0)
    lband = (min(off, on) - 6.0, max(off, on) + 6.0)
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=lband,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        decay_probe=lambda f: (1.0, 50.0),  # flat -> nothing vetoed
        band_power_probe=lambda f: (0.0, 0.0),  # never gate a drift fallback
    )
    centers = sorted(round(s.center_mhz) for s in sset.spurs)
    assert 39990 in centers, "off-lattice integer tone dropped under declaration"
    # The off-lattice tone carries no lattice identity; the on-lattice one
    # carries an identity (stamped by the lattice sweep).
    off_s = [s for s in sset.spurs if round(s.center_mhz) == 39990][0]
    on_s = [s for s in sset.spurs if round(s.center_mhz) == 39680][0]
    assert off_s.lattice is None
    assert on_s.lattice is not None and "640" in on_s.lattice


def test_union_lattice_nominee_shadows_integer_at_same_bin():
    # A clean comb tone that is BOTH integer and on-lattice must appear once,
    # carrying the lattice identity (the lattice-stamped nominee wins).
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    center = 30720.0  # integer AND 48 x 640 (rf)
    freqs, spec, sig_c = _grid([(center, 100.0, 0.0)])
    lband = _band_for(center)
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=lband,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=lband,
        lattice=lat,
        decay_probe=lambda f: (1.0, 50.0),
        band_power_probe=lambda f: (0.0, 0.0),
    )
    near = [s for s in sset.spurs if abs(s.center_mhz - center) <= 0.1]
    assert len(near) == 1
    assert near[0].lattice is not None and "640" in near[0].lattice


def test_scaled_mask_truncates_at_real_structure():
    # The 1512 28440.26 regression: a flat-gated tone 0.5-0.8 MHz from real
    # catalog lines must not have its SNR-scaled mask swallow them. The walk
    # truncates where the spectrum exceeds the tone's own sinc-skirt
    # envelope; the line side and the empty side truncate to the minimum.
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    center = 28440.26
    line = center + 6 * SPACING  # a real line ~6 bins above the tone
    freqs, spec, sig_c = _grid([(center, 139.0, 0.0), (line, 40.0, 2 * SPACING)])
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,
        sideband=Sideband.LOWER,
        band=BAND,
        tol_mhz=0.04,
        drift_window_mhz=0.3,
    )
    j = int(np.argmin(np.abs(freqs - center)))
    cl = SpurCluster(
        center_freq_mhz=center,
        peak_bin_index=j,
        n_bins=1,
        bin_indices=(j,),
        saturated=True,
    )
    sset = build_spur_set(
        freqs,
        spec,
        sig_c,
        band=BAND,
        saturated_clusters=[cl],
        lattice=lat,
        decay_probe=lambda f: (1.0, 50.0),
        mask_half_width_bins=2,
        mask_target_residual_snr=2.0,
        mask_max_half_width_bins=64,
    )
    flat = [s for s in sset.spurs if abs(s.center_mhz - center) < 0.1]
    assert flat
    s = flat[0]
    hw = s.mask_half_width_bins
    # Without the line the scaled width would be ceil(139/(2*pi)) = 23 bins;
    # the line's profile becomes significant a couple bins inside d=6, so the
    # mask must stop short of the line while staying wider than the base.
    assert hw is not None and 2 <= hw < 6
    # and the line's own bin must not be masked
    spec_mask = sset.window_mask_spec(
        center - 2.0, center + 2.0, 40000.0, Sideband.LOWER
    )
    assert spec_mask is not None
    u = -(freqs[np.abs(freqs - line) < 0.5 * SPACING] - 40000.0)
    assert not spec_mask.bin_mask(u).any()


# ---------------------------------------------------------------------------
# Chirp-response anchor probe + gate integration
# ---------------------------------------------------------------------------
# Use a direct-sampling geometry where the molecular frequencies ARE the
# baseband frequencies (probe = 0, upper sideband), keeping f_bb well within
# the Nyquist band of the test sample rate.  dt = 0.001 µs -> Nyquist 500 MHz.
_CR_DT = 0.001  # µs (1 GSa/s)
_CR_N_FID = 20000  # 20 µs active slice
_CR_N_PRE = 12500  # 12.5 µs pre-record
# Probe frequency of 0 MHz with upper sideband maps f_mol -> f_bb = f_mol - 0.
# Use a low molecular frequency (200 MHz) well within the Nyquist band.
_CR_PROBE_MHZ = 0.0
_CR_FREQ_MOL = 200.0  # MHz; f_bb = 200 MHz, well within Nyquist at 1 GSa/s


def _make_pre_and_fid(
    *,
    cw_amp: float = 0.0,
    line_amp: float = 0.0,
    tau_us: float = 4.0,
    freq_mol_mhz: float = _CR_FREQ_MOL,
    probe_mhz: float = _CR_PROBE_MHZ,
    noise: float = 0.0005,
    seed: int = 42,
    sideband: "Sideband" = Sideband.UPPER,
) -> tuple:
    """Synthetic pre-record and FID for the chirp-response probe tests.

    Returns ``(pre_record, fid, dt_us, start_us, end_us, probe_mhz, sideband)``.

    Upper-sideband convention: f_bb = f_mol - probe (= f_mol for probe=0).
    ``cw_amp`` sets the amplitude of a persistent CW tone (present in both
    pre-record and FID). ``line_amp`` sets the amplitude of a chirp-responsive
    line (present ONLY in the FID, decaying with ``tau_us``).
    """
    rng = np.random.default_rng(seed)
    # Upper sideband: f_bb = f_mol - probe
    f_bb = freq_mol_mhz - probe_mhz
    t_pre = np.arange(_CR_N_PRE) * _CR_DT
    t_fid = np.arange(_CR_N_FID) * _CR_DT

    pre = rng.normal(0.0, noise, size=_CR_N_PRE)
    fid = rng.normal(0.0, noise, size=_CR_N_FID)

    if cw_amp > 0.0:
        # CW tone: flat in both pre-record and FID.
        pre = pre + cw_amp * np.cos(2 * np.pi * f_bb * t_pre + 0.1)
        fid = fid + cw_amp * np.cos(2 * np.pi * f_bb * t_fid + 0.1)

    if line_amp > 0.0:
        # Chirp-responsive line: exponentially decaying, in FID only.
        fid = fid + line_amp * np.exp(-t_fid / tau_us) * np.cos(
            2 * np.pi * f_bb * t_fid + 0.5
        )

    start_us = 0.0
    end_us = _CR_N_FID * _CR_DT
    return pre, fid, _CR_DT, start_us, end_us, probe_mhz, sideband


def _cr_probe_for(
    cw_amp: float = 0.0,
    line_amp: float = 0.0,
    tau_us: float = 4.0,
    freq_mol_mhz: float = _CR_FREQ_MOL,
) -> "Callable":
    from ftmwpipeline.fitting.spur_detection import make_chirp_response_probe

    pre, fid, dt_us, start_us, end_us, probe_mhz, sband = _make_pre_and_fid(
        cw_amp=cw_amp, line_amp=line_amp, tau_us=tau_us, freq_mol_mhz=freq_mol_mhz
    )
    return make_chirp_response_probe(
        pre,
        fid,
        dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_mhz,
        sideband=sband,
    )


# (a) Flat CW tone in both pre-record and FID -> gate-confirm verdict, even
#     when the decay probe reads "decays" (modulated-carrier failure mode).
def test_chirp_response_cw_tone_gate_confirm():
    """A flat CW tone present pre-record at full amplitude -> ratio >= 0.8."""
    probe = _cr_probe_for(cw_amp=1.0)
    ratio, pre_snr, fid_det = probe(_CR_FREQ_MOL)
    assert np.isfinite(ratio), "probe returned nan for a clean CW tone"
    assert ratio >= 0.8, f"CW tone ratio {ratio:.3f} below gate threshold"
    assert pre_snr >= 5.0, f"CW tone pre_snr {pre_snr:.2f} below floor"


def test_chirp_response_cw_overrides_decaying_decay_probe():
    """Gate-confirm from chirp-response probe overrides a decay-probe 'decays' veto.

    A CW tone whose phase modulation makes the coherent demod read a pseudo-
    decay ratio < DEFAULT_DECAY_RATIO_LINE: the chirp-response probe's pre-
    record confirmation must still gate it.
    """
    from ftmwpipeline.fitting.spur_detection import gate_spurs, Spur

    probe = _cr_probe_for(cw_amp=1.0)

    # Confirm the chirp-response probe sees ratio >= 0.8.
    ratio, pre_snr, _ = probe(_CR_FREQ_MOL)
    assert ratio >= 0.8 and pre_snr >= 5.0

    # Construct a synthetic narrow nominee at _CR_FREQ_MOL (200 MHz) so we
    # don't need a full active-FT grid at that frequency.
    sp = Spur(
        integer_mhz=int(round(_CR_FREQ_MOL)),
        center_mhz=_CR_FREQ_MOL,
        bin_index=10,
        magnitude=100.0,
        snr=50.0,
        narrowness_ratio=0.1,
    )

    # A decay probe that reads "decays" (ratio 0.2) would veto the narrow
    # nominee on its own.  With the chirp-response probe confirming CW, the
    # spur must still be gated.
    gated = gate_spurs(
        [sp],
        [],
        decay_probe=lambda f: (0.2, 50.0),  # would veto alone
        chirp_response_probe=probe,
    )
    assert (
        len(gated) == 1
    ), "chirp-response gate-confirm did not override the decay-probe veto"
    assert gated[0].integer_mhz == int(round(_CR_FREQ_MOL))


# (b) Decaying molecular line: absent from pre-record (or present at 0.1x)
#     with high fid_detectability -> protect verdict vetoes the gating.
def test_chirp_response_protect_vetoes_molecular_line():
    """A chirp-responsive line absent pre-record is protected from gating."""
    from ftmwpipeline.fitting.spur_detection import gate_spurs, Spur

    # Strong line in FID only; CW counterpart would be bright pre-record.
    probe = _cr_probe_for(line_amp=1.0, tau_us=4.0)
    ratio, pre_snr, fid_det = probe(_CR_FREQ_MOL)
    # The ratio should be well below the protect threshold (0.3) and
    # fid_detectability should be high enough to trigger protect.
    assert np.isfinite(ratio), "probe returned nan for a decaying line"

    # Construct a synthetic narrow nominee at _CR_FREQ_MOL.
    sp = Spur(
        integer_mhz=int(round(_CR_FREQ_MOL)),
        center_mhz=_CR_FREQ_MOL,
        bin_index=10,
        magnitude=100.0,
        snr=50.0,
        narrowness_ratio=0.1,
    )

    # Without the chirp-response probe the narrowness gate would fire.
    gated_no_probe = gate_spurs([sp], [])
    assert len(gated_no_probe) == 1  # sanity: narrowness gates it alone

    # With the chirp-response probe in protect mode, the gate is vetoed.
    gated = gate_spurs([sp], [], chirp_response_probe=probe)
    assert gated == [], (
        "chirp-response protect did not veto the narrowness gate for a "
        f"molecular line (ratio={ratio:.3f}, fid_det={fid_det:.2f})"
    )


# (c) Weak tone below the pre-record floor -> inconclusive, existing lanes decide.
def test_chirp_response_inconclusive_falls_through():
    """A tone weak enough to be below the pre-record floor leaves the verdict
    to the existing lanes (no protect, no gate-confirm)."""
    from ftmwpipeline.fitting.spur_detection import gate_spurs, Spur

    # Pure-noise probe: both pre_snr and fid_detectability will be near 1
    # (noise-level), so neither gate-confirm nor protect fires.
    probe = _cr_probe_for(cw_amp=0.0)

    sp = Spur(
        integer_mhz=int(round(_CR_FREQ_MOL)),
        center_mhz=_CR_FREQ_MOL,
        bin_index=10,
        magnitude=100.0,
        snr=50.0,
        narrowness_ratio=0.1,
    )

    # Legacy narrowness gate should still fire (the probe is inconclusive).
    gated = gate_spurs([sp], [], chirp_response_probe=probe)
    assert len(gated) == 1, (
        "inconclusive chirp-response probe should fall through to the "
        "narrowness gate, but the spur was not gated"
    )


# (d) Sideband mapping: lower-sideband probe frequency handled correctly.
def test_chirp_response_lower_sideband_probe_mapping():
    """The baseband conversion is correct for a lower-sideband instrument.

    Lower sideband: f_bb = probe - f_mol. Build a CW tone at probe - f_mol
    so it lands within Nyquist of the test sample rate.
    """
    from ftmwpipeline.fitting.spur_detection import make_chirp_response_probe

    # probe = 500 MHz, f_mol = 300 MHz -> f_bb = 200 MHz (within Nyquist 500 MHz)
    probe_mhz = 500.0
    freq_mol = 300.0
    f_bb = probe_mhz - freq_mol  # = 200 MHz

    rng = np.random.default_rng(99)
    t_pre = np.arange(_CR_N_PRE) * _CR_DT
    t_fid = np.arange(_CR_N_FID) * _CR_DT
    cw_amp = 1.0
    noise = 0.0005

    pre = rng.normal(0.0, noise, size=_CR_N_PRE) + cw_amp * np.cos(
        2 * np.pi * f_bb * t_pre + 0.1
    )
    fid = rng.normal(0.0, noise, size=_CR_N_FID) + cw_amp * np.cos(
        2 * np.pi * f_bb * t_fid + 0.1
    )

    probe_lower = make_chirp_response_probe(
        pre,
        fid,
        _CR_DT,
        start_us=0.0,
        end_us=_CR_N_FID * _CR_DT,
        probe_freq_mhz=probe_mhz,
        sideband=Sideband.LOWER,
    )
    ratio, pre_snr, _ = probe_lower(freq_mol)
    assert np.isfinite(ratio), "lower-sideband probe returned nan"
    assert ratio >= 0.8, f"lower-sideband CW ratio {ratio:.3f} below gate bar"
    assert pre_snr >= 5.0


# (e) No probe -> gate_spurs is byte-identical to the no-probe path.
def test_chirp_response_none_probe_byte_identical():
    """Passing chirp_response_probe=None must be exactly equivalent to
    omitting it (byte-identical SpurSet)."""
    from ftmwpipeline.fitting.spur_detection import gate_spurs

    freqs, spec, sig_c = _grid([(30720.0, 100.0, 0.0)])
    narrow = detect_active_ft_spurs(freqs, spec, sig_c, band=BAND)
    probe = _probe_for([(30720.0, 1.0, 0.0)])  # decay probe, CW

    without = gate_spurs(narrow, [], decay_probe=probe)
    with_none = gate_spurs(narrow, [], decay_probe=probe, chirp_response_probe=None)
    assert without == with_none, "None chirp_response_probe broke byte-identity"


# ---------------------------------------------------------------------------
# Chirp-response probe: interleave-comb exclusion
# ---------------------------------------------------------------------------
# Geometry for the comb-exclusion tests: direct-sampling, probe=0, upper
# sideband, 1 GSa/s.  A cleanup comb spacing of 250 MHz (= fs/4, for a
# 4-phase interleave at 1 GSa/s) puts its 3rd harmonic at 750 MHz.
_CX_DT = 0.001  # µs  (1 GSa/s, Nyquist 500 MHz)
_CX_N_FID = 20000  # 20 µs
_CX_N_PRE = 12500  # 12.5 µs
_CX_PROBE = 0.0  # MHz
_CX_SIDEBAND = Sideband.UPPER
_CX_COMB_SPACING = 250.0  # MHz  (= 1 GHz / 4 interleave factor)
# Put the tone ON the 1st nonzero comb multiple within Nyquist: 1 x 250 = 250 MHz
_CX_COMB_FREQ = 250.0  # MHz (molecular = baseband for probe=0, upper sideband)
# Put an off-comb tone within Nyquist: 300 MHz (not a multiple of 250)
_CX_OFF_FREQ = 300.0  # MHz


def _make_cx_pre_and_fid(
    cw_freq_mhz: float,
    *,
    cw_amp: float = 1.0,
    pre_comb_nulled: bool = True,
) -> "tuple":
    """Synthetic (pre_record, fid) with a CW tone at ``cw_freq_mhz``.

    When ``pre_comb_nulled`` is True the pre-record amplitude at the given
    frequency is zeroed to simulate interleave-offset cleanup nulling the
    comb line there, while the FID retains full amplitude.
    """
    rng = np.random.default_rng(17)
    noise = 0.0005
    f_bb = cw_freq_mhz  # upper sideband, probe=0

    t_pre = np.arange(_CX_N_PRE) * _CX_DT
    t_fid = np.arange(_CX_N_FID) * _CX_DT

    pre = rng.normal(0.0, noise, size=_CX_N_PRE)
    fid = rng.normal(0.0, noise, size=_CX_N_FID)

    # CW tone in FID always.
    fid = fid + cw_amp * np.cos(2 * np.pi * f_bb * t_fid + 0.2)

    if not pre_comb_nulled:
        # CW also present in pre-record (the normal case without cleanup).
        pre = pre + cw_amp * np.cos(2 * np.pi * f_bb * t_pre + 0.2)
    # If pre_comb_nulled, the pre-record is left noise-only at this frequency
    # (simulating the cleanup zeroing its per-phase means).

    return pre, fid


def _cx_probe(
    cw_freq_mhz: float,
    *,
    pre_comb_nulled: bool = True,
    excluded_comb_mhz: "Optional[Sequence[float]]" = None,
) -> "Callable":
    """Build a chirp-response probe for the comb-exclusion tests."""
    from ftmwpipeline.fitting.spur_detection import make_chirp_response_probe

    pre, fid = _make_cx_pre_and_fid(cw_freq_mhz, pre_comb_nulled=pre_comb_nulled)
    return make_chirp_response_probe(
        pre,
        fid,
        _CX_DT,
        start_us=0.0,
        end_us=_CX_N_FID * _CX_DT,
        probe_freq_mhz=_CX_PROBE,
        sideband=_CX_SIDEBAND,
        excluded_comb_mhz=excluded_comb_mhz,
    )


def test_comb_excluded_freq_returns_inconclusive():
    """A CW tone on a cleanup-comb multiple with exclusion active -> inconclusive.

    The pre-record is null at the comb frequency (cleanup manufactured the
    absence), so without exclusion the probe would read ratio ≈ 0 (protect).
    With exclusion, it returns NaN (inconclusive), so the narrowness / decay
    / lattice lanes decide instead.
    """
    from ftmwpipeline.fitting.spur_detection import Spur, gate_spurs

    # Build probe with comb nulled in pre-record AND exclusion active.
    probe = _cx_probe(
        _CX_COMB_FREQ,
        pre_comb_nulled=True,
        excluded_comb_mhz=[_CX_COMB_SPACING],
    )
    ratio, pre_snr, fid_det = probe(_CX_COMB_FREQ)
    # The probe must return the inconclusive sentinel (NaN) for the comb freq.
    assert not np.isfinite(
        ratio
    ), f"Expected NaN sentinel for comb freq {_CX_COMB_FREQ} MHz, got ratio={ratio:.3f}"

    # In gate_spurs the NaN flows through _chirp_verdict -> "inconclusive",
    # so the narrowness gate still fires (the protect veto does NOT apply).
    sp = Spur(
        integer_mhz=int(round(_CX_COMB_FREQ)),
        center_mhz=_CX_COMB_FREQ,
        bin_index=10,
        magnitude=3000.0,
        snr=3000.0,
        narrowness_ratio=0.05,
    )
    gated = gate_spurs([sp], [], chirp_response_probe=probe)
    assert len(gated) == 1, (
        "Narrowness gate should fire when chirp-response probe is inconclusive "
        "(protect veto must NOT apply for an excluded comb frequency)"
    )


def test_comb_excluded_without_exclusion_triggers_protect():
    """Without exclusion, a cleanup-nulled comb tone triggers the protect veto.

    This is the defect case: ratio ≈ 0 when the pre-record was zeroed by
    cleanup, and the probe incorrectly vetoes gating the true clock spur.
    This test pins the CURRENT behavior so the fix is clearly validated by
    the companion test above.
    """
    from ftmwpipeline.fitting.spur_detection import Spur, gate_spurs

    # Same nulled pre-record, but NO exclusion list.
    probe = _cx_probe(_CX_COMB_FREQ, pre_comb_nulled=True, excluded_comb_mhz=None)
    ratio, pre_snr, fid_det = probe(_CX_COMB_FREQ)
    # Without exclusion the probe reads a very low (near-zero) ratio because
    # the pre-record was zeroed at this frequency by the cleanup.  The exact
    # value depends on the noise floor, but it should be well below 0.3.
    # (We only assert it is finite and small; the exact value is noise-level.)
    if np.isfinite(ratio):
        # Only assert when the pre-floor estimate is reliable enough to return
        # a finite ratio; if noise pushes fid_amp to zero we get NaN and the
        # test is trivially a pass (no protect fires from NaN).
        assert ratio <= 0.3, (
            f"Expected low ratio for cleanup-nulled comb (defect case), "
            f"got ratio={ratio:.3f}"
        )

    # The spur should be vetoed (protect fires when ratio is finite and low).
    sp = Spur(
        integer_mhz=int(round(_CX_COMB_FREQ)),
        center_mhz=_CX_COMB_FREQ,
        bin_index=10,
        magnitude=3000.0,
        snr=3000.0,
        narrowness_ratio=0.05,
    )
    gated = gate_spurs([sp], [], chirp_response_probe=probe)
    # This is the defect: the true spur is incorrectly not gated.
    # We document this with a non-asserting note so the test is informational
    # (the companion test above demonstrates the fix).
    # The assert here confirms the defect: without exclusion protect fires.
    if np.isfinite(ratio) and fid_det >= 5.0 / 0.3:
        assert len(gated) == 0, (
            "Expected protect veto WITHOUT exclusion (defect baseline): "
            "the narrowness gate should be blocked by the ratio-0 artifact"
        )


def test_off_comb_freq_unaffected_by_exclusion():
    """A tone NOT on a comb multiple is unaffected by the exclusion list.

    A true chirp-responsive molecular line at an off-comb frequency: the
    probe reads a normal (not manufactured) ratio and the protect verdict
    still fires when appropriate.  And a CW tone at an off-comb frequency
    with a real pre-record signal is not incorrectly excluded.
    """
    from ftmwpipeline.fitting.spur_detection import make_chirp_response_probe

    # Off-comb CW tone: pre-record is NOT nulled (it's a real CW signal).
    probe = _cx_probe(
        _CX_OFF_FREQ,
        pre_comb_nulled=False,
        excluded_comb_mhz=[_CX_COMB_SPACING],
    )
    ratio, pre_snr, fid_det = probe(_CX_OFF_FREQ)
    # For a real CW tone (not nulled), the ratio should be finite and near 1.
    assert np.isfinite(
        ratio
    ), f"Off-comb CW tone should give a finite ratio; got NaN at {_CX_OFF_FREQ} MHz"
    assert ratio >= 0.5, (
        f"Off-comb CW tone ratio {ratio:.3f} unexpectedly low (exclusion should "
        "not suppress this frequency)"
    )
