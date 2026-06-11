"""Unit tests for the Stage 5 spur detector and joint gate."""

from __future__ import annotations

import math

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
    # probe-flat catalogue) carries snr=NaN, so SNR-scaled mask never fired
    # and its strong sinc skirt detonated the window. The fallback reads the
    # bin SNR from the active FT and widens the mask.
    from ftmwpipeline.core.stage_fit_settings import ClockSource
    from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

    # A strong tone at an off-lattice, non-integer frequency: the
    # frequency-domain sweep never visits it (only integer/lattice points
    # are swept), so only the flat catalogue + flat probe gate it. The
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
    spec_mask = sset.window_mask_spec(center - 2.0, center + 2.0, 40000.0, Sideband.LOWER)
    assert spec_mask is not None
    u = -(freqs[np.abs(freqs - line) < 0.5 * SPACING] - 40000.0)
    assert not spec_mask.bin_mask(u).any()
