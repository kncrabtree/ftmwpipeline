"""Unit tests for the instrument clock-lattice spur prior."""

from __future__ import annotations

import math

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.clock_lattice import build_clock_lattice

# Home-instrument declaration: locked {5760, 5120, 16000} -> gcd 640 (each
# is an exact multiple of 640: 9x / 8x / 25x); the unlocked 6250-MHz ADC
# predicts the drifting family. Probe 40960 (lower sideband), active band
# 26500-40000.
PROBE = 40960.0
BAND = (26500.0, 40000.0)
TOL = 0.04
DRIFT_WIN = 0.3

HOME_CLOCKS = (
    ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
    ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
    ClockSource(freq_mhz=16000.0, locked=True, label="awg"),
    ClockSource(freq_mhz=6250.0, locked=False, label="scope-adc"),
)


def _lattice(clocks=HOME_CLOCKS, *, sideband=Sideband.LOWER, band=BAND):
    return build_clock_lattice(
        clocks,
        probe_freq_mhz=PROBE,
        sideband=sideband,
        band=band,
        tol_mhz=TOL,
        drift_window_mhz=DRIFT_WIN,
    )


def test_no_declaration_returns_none():
    assert _lattice(()) is None
    assert _lattice(None) is None


def test_gcd_of_locked_clocks():
    lat = _lattice()
    assert lat is not None
    assert lat.g_mhz == 640  # gcd(5760, 5120, 16000)
    assert lat.unlocked_freqs_mhz == (6250.0,)


def test_locked_membership_both_frames_lower_sideband():
    # Probe (40960 = 64 x 640) is itself on the lattice, so every RF
    # multiple of g is also a baseband multiple. Use a clock declaration
    # whose probe is OFF the lattice to exercise the two frames separately.
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=640.0, locked=True),),
        probe_freq_mhz=40000.0,  # 40000 / 640 = 62.5 -> off the lattice
        sideband=Sideband.LOWER,
        band=BAND,
        tol_mhz=TOL,
        drift_window_mhz=DRIFT_WIN,
    )
    assert lat is not None
    # RF frame: 27520 = 43 x 640 (multiple of g directly); not a bb multiple
    # (f_bb = 40000 - 27520 = 12480 = 19.5 x 640).
    m_rf = lat.match(27520.0)
    assert m_rf is not None and not m_rf.drift
    assert "rf" in m_rf.identity and "640x43" in m_rf.identity
    # Baseband frame: f_bb = probe - f_mol = 640 -> f_mol = 39360; 39360 is
    # NOT an RF multiple of 640 (39360 / 640 = 61.5), so it matches bb only.
    m_bb = lat.match(39360.0)
    assert m_bb is not None and not m_bb.drift
    assert "bb" in m_bb.identity and "640x1" in m_bb.identity


def test_off_lattice_frequency_no_match():
    lat = _lattice()
    assert lat is not None
    # 27520 + 100 kHz: off the 640 grid in both frames by > tol.
    assert lat.match(27520.1) is None


def test_match_excludes_k_zero():
    # A lattice with g = probe so f_bb = 0 at f_mol = probe (k=0); must not
    # match. Use a single locked clock equal to the probe.
    lat = build_clock_lattice(
        (ClockSource(freq_mhz=PROBE, locked=True),),
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        band=BAND,
        tol_mhz=TOL,
        drift_window_mhz=DRIFT_WIN,
    )
    assert lat is not None
    # f_bb = 0 exactly at the probe; but the probe (40960) is out of band and
    # also k=0 in bb frame -> no match on the bb DC term.
    assert lat.match(PROBE) is not None  # 40960 = 8 x 5120 in the RF frame
    # A frequency whose only "match" would be the k=0 baseband term: pick
    # f_mol = probe so f_bb = 0; the RF match (k=8) is the real reason it
    # gates, so test a non-multiple in RF that is bb-DC.
    lat2 = build_clock_lattice(
        (ClockSource(freq_mhz=320.0, locked=True),),
        probe_freq_mhz=33333.0,  # not a multiple of 320
        sideband=Sideband.LOWER,
        band=(33000.0, 34000.0),
        tol_mhz=TOL,
        drift_window_mhz=DRIFT_WIN,
    )
    assert lat2 is not None
    # f_mol = probe -> f_bb = 0 (k=0); 33333 is not a multiple of 320 in RF
    # either, so no match.
    assert lat2.match(33333.0) is None


def test_non_integral_locked_clock_skipped(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        lat = build_clock_lattice(
            (
                ClockSource(freq_mhz=5120.0, locked=True),
                ClockSource(freq_mhz=5760.5, locked=True),  # non-integral
            ),
            probe_freq_mhz=PROBE,
            sideband=Sideband.LOWER,
            band=BAND,
            tol_mhz=TOL,
            drift_window_mhz=DRIFT_WIN,
        )
    assert lat is not None
    # Only 5120 survives -> gcd = 5120 (5760.5 skipped).
    assert lat.g_mhz == 5120
    assert any("non-integral" in r.message for r in caplog.records)


def test_locked_beats_drift_dedupe():
    lat = _lattice()
    assert lat is not None
    pts = lat.nomination_points()
    # No drift point may coincide (within tol) with a locked point.
    locked_freqs = [p.freq_mhz for p in pts if not p.drift]
    for dp in (p for p in pts if p.drift):
        assert not any(abs(dp.freq_mhz - lf) <= TOL for lf in locked_freqs)
    # The drifting 39040 tone (bb 1920 = 3 x 640) is itself a locked point,
    # so it dedupes to locked; the surviving 6250 drift harmonics are
    # genuinely off the 640 grid.
    assert any(p.drift for p in pts)


def test_probe_on_lattice_degeneracy_prefers_bb():
    # Probe 40960 = 64 x 640, so every bb point f_mol = probe - k*640 is
    # ALSO an rf multiple of 640 -- the dedupe must keep the bb identity.
    lat = _lattice()
    assert lat is not None
    pts = lat.nomination_points()
    locked = [p for p in pts if not p.drift]
    # Find the point at 39680 (= probe - 1280 = 2 x 640): must be the bb id.
    near = [p for p in locked if abs(p.freq_mhz - 39680.0) <= TOL]
    assert near and "bb" in near[0].identity
    # Every locked point at a 640-grid frequency exists once (deduped).
    freqs = [round(p.freq_mhz, 3) for p in locked]
    assert len(freqs) == len(set(freqs))


def test_nomination_points_sorted_and_in_band():
    lat = _lattice()
    assert lat is not None
    pts = lat.nomination_points()
    assert pts == sorted(pts, key=lambda p: p.freq_mhz)
    lo, hi = BAND
    assert all(lo <= p.freq_mhz <= hi for p in pts)
    # Locked points use the tol window; drift points the drift window.
    for p in pts:
        assert p.window_mhz == (DRIFT_WIN if p.drift else TOL)


def test_drift_membership_uses_drift_window():
    lat = _lattice()
    assert lat is not None
    # 6250 baseband harmonic: f_bb = 6250 -> f_mol = probe - 6250 = 34710
    # (in band, not a 640 multiple: 34710 / 640 = 54.23...). Off by 0.1 MHz
    # still matches the drift window.
    m = lat.match(34710.0 + 0.1)
    assert m is not None and m.drift
    assert "6250" in m.identity and "bb" in m.identity
    # Off by more than the drift window: no match.
    assert lat.match(34710.0 + 0.5) is None


def test_unlocked_drift_points_are_baseband_only():
    # An unlocked ADC clock injects at digitization, so its drifting family
    # lives in the baseband frame only -- no molecular-frame (RF) point can
    # be predicted (a spurious 6250x5 (rf) = 31250 would land on a real
    # molecular band, measured on 363).
    lat = _lattice()
    assert lat is not None
    pts = lat.nomination_points()
    drift = [p for p in pts if p.drift]
    assert drift  # the bb drift family is present
    assert all("bb" in p.identity for p in drift)
    assert not any("rf" in p.identity for p in drift)
    # 31250 = 5 x 6250 in the RF frame is no longer matched as a drift point
    # (it is also not a 640 multiple, so no locked match either).
    assert lat.match(31250.0) is None


def test_upper_sideband_frame_mapping():
    lat = _lattice(sideband=Sideband.UPPER, band=(41000.0, 54000.0))
    assert lat is not None
    # Upper sideband: f_mol = probe + f_bb. bb 640 -> f_mol = 41600 = 65x640.
    m = lat.match(41600.0)
    assert m is not None and not m.drift
    # 41600 is a 640 multiple in RF too; the bb degeneracy dedupe should
    # still yield a single nomination point.
    pts = [p for p in lat.nomination_points() if abs(p.freq_mhz - 41600.0) <= TOL]
    assert len(pts) == 1


# ---------------------------------------------------------------------------
# annotate_lattice_matches helper
# ---------------------------------------------------------------------------
class TestAnnotateLatticeMatches:
    """Tests for the Stage 5 post-fit annotation helper.

    Drives ``annotate_lattice_matches`` directly (no full pipeline run) to
    verify the annotation logic in isolation.
    """

    def _make_fit(self, freqs):
        """Build a minimal SpectrumFit carrying peaks at the given frequencies."""
        from ftmwpipeline.core.data_structures import (
            FittedPeak,
            FittingResult,
            SpectrumFit,
        )

        peaks = [
            FittedPeak(
                peak_id=i,
                frequency_mhz=float(f),
                amplitude=1.0,
                window_id=0,
            )
            for i, f in enumerate(freqs)
        ]
        fr = FittingResult(success=True, window_id=0)
        fr.fitted_peaks = list(peaks)
        fr.shared_parameters["tau_us"] = {
            "value": 3.0,
            "error": 0.1,
            "peak_ids": [p.peak_id for p in peaks],
        }
        sf = SpectrumFit(window_fits=[fr], fitted_peaks=list(peaks))
        return sf

    def test_no_declaration_is_noop(self):
        """When ``lattice`` is None no peak is annotated."""
        from ftmwpipeline._internal.stage5_impl import annotate_lattice_matches

        sf = self._make_fit([36100.0, 36200.0])
        annotate_lattice_matches(sf, None)
        assert all(p.clock_lattice is None for p in sf.fitted_peaks)

    def test_on_lattice_peak_annotated(self):
        """A peak on a locked-lattice frequency receives the identity string."""
        from ftmwpipeline._internal.stage5_impl import annotate_lattice_matches

        lat = _lattice()
        assert lat is not None
        # 39040 = probe - 1920; 1920 = 3 x 640 -> a locked bb point.
        f_on = 39040.0
        m = lat.match(f_on)
        assert m is not None and not m.drift

        sf = self._make_fit([f_on, 36150.3])  # second is off-lattice
        annotate_lattice_matches(sf, lat)

        by_freq = {p.frequency_mhz: p for p in sf.fitted_peaks}
        assert by_freq[f_on].clock_lattice == m.identity
        assert by_freq[36150.3].clock_lattice is None

    def test_off_lattice_peak_stays_none(self):
        """A peak whose frequency matches no lattice point keeps None."""
        from ftmwpipeline._internal.stage5_impl import annotate_lattice_matches

        lat = _lattice()
        sf = self._make_fit([36150.3])  # off the 640-MHz grid in both frames
        annotate_lattice_matches(sf, lat)
        assert sf.fitted_peaks[0].clock_lattice is None

    def test_drift_peak_annotated_with_drift_suffix(self):
        """A drifting-family match gets the '(bb, drift)' identity."""
        from ftmwpipeline._internal.stage5_impl import annotate_lattice_matches

        lat = _lattice()
        assert lat is not None
        # 6250 ADC harmonic: bb 6250 -> f_mol = probe - 6250 = 34710.
        f_drift = 34710.0
        m = lat.match(f_drift)
        assert m is not None and m.drift

        sf = self._make_fit([f_drift])
        annotate_lattice_matches(sf, lat)
        p = sf.fitted_peaks[0]
        assert p.clock_lattice is not None
        assert "drift" in p.clock_lattice

    def test_per_window_peaks_also_annotated(self):
        """The per-window FittingResult's peak list gets the same annotation."""
        from ftmwpipeline._internal.stage5_impl import annotate_lattice_matches

        lat = _lattice()
        assert lat is not None
        f_on = 39040.0
        sf = self._make_fit([f_on])
        annotate_lattice_matches(sf, lat)
        # Check the per-window list matches the global list.
        wf_peak = sf.window_fits[0].fitted_peaks[0]
        global_peak = sf.fitted_peaks[0]
        assert wf_peak.clock_lattice == global_peak.clock_lattice
        assert wf_peak.clock_lattice is not None


def test_baseband_mhz_matches_f_bb_and_round_trips():
    """``baseband_mhz`` exposes the private ``_f_bb`` and round-trips.

    For every nomination point the public baseband frequency must equal the
    sideband-signed offset from the probe (>= 0), match the private ``_f_bb``,
    and -- for a baseband-frame identity -- land on a multiple of the lattice
    GCD. The probe frequency itself maps to baseband 0.
    """
    lat = _lattice()
    assert lat is not None
    assert lat.baseband_mhz(PROBE) == 0.0
    for p in lat.nomination_points():
        bb = lat.baseband_mhz(p.freq_mhz)
        assert bb >= 0.0
        assert bb == abs(lat.sideband_sign * (p.freq_mhz - PROBE))
        assert bb == lat._f_bb(p.freq_mhz)
        if "(bb)" in p.identity:
            # A baseband-frame point sits on a multiple of the GCD.
            r = bb % lat.g_mhz
            assert min(r, lat.g_mhz - r) < 1e-6
