"""
Unit tests for Stage 6 candidate-ledger derivation.

Tests :func:`ftmwpipeline._internal.stage6_impl.derive_candidate_ledger`
against synthetic :class:`FittingResult` objects with hand-crafted audit
trails and rescue events.  No file I/O is needed.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from ftmwpipeline._internal.stage6_impl import (
    _DEDUP_TOL_MHZ,
    DEFAULT_DISPLAY_BAR,
    _dedup_and_merge,
    _audit_step_candidates,
    _rescue_round_candidates,
    _to_molecular,
    derive_candidate_ledger,
)
from ftmwpipeline.core.data_structures import (
    AuditStep,
    FittedPeak,
    FittingResult,
    LedgerCandidate,
    RescueCandidateInfo,
    RescueRoundInfo,
    Sideband,
    SpectralWindow,
)

# ---------------------------------------------------------------------------
# Helpers: build synthetic FittingResult objects
# ---------------------------------------------------------------------------

_LOWER = Sideband.LOWER
_UPPER = Sideband.UPPER

# Window used in most tests: 36000–36020 MHz → center 36010
_CENTER = 36010.0
_FREQ_LO = 36000.0
_FREQ_HI = 36020.0


def _make_audit_step(
    offset: float,
    decision: str = "reject",
    reason: str = "aicc",
    aicc_delta: float = float("nan"),
    p_value: float = float("nan"),
    chi2_before: float = 100.0,
    chi2_after: float = 90.0,
) -> AuditStep:
    return AuditStep(
        n_peaks_before=0,
        candidate_offset_mhz=offset,
        chi2_before=chi2_before,
        chi2_after=chi2_after,
        f_statistic=0.0,
        p_value=p_value,
        aic_before=200.0,
        aic_after=190.0,
        separation_ok=True,
        decision=decision,
        reason=reason,
        aicc_delta=aicc_delta,
    )


def _make_rescue_round(
    candidates: list,
    round_idx: int = 0,
    window_id: int = 0,
    accepted: bool = False,
) -> RescueRoundInfo:
    return RescueRoundInfo(
        window_id=window_id,
        round_idx=round_idx,
        n_initial_peaks=1,
        n_rescue_added=0,
        n_pruned_total=0,
        n_pruned_rescue_origin=0,
        n_merged=0,
        chi2_before=100.0,
        chi2_after=100.0,
        tau_us_before=5.0,
        tau_us_after=5.0,
        accepted=accepted,
        reason="no improvement",
        candidates=candidates,
    )


def _make_fitting_result(
    audit_trail: list | None = None,
    rescue_events: list | None = None,
    window_id: int = 0,
) -> FittingResult:
    fr = FittingResult(
        success=True,
        window_id=window_id,
    )
    # Attach a minimal SpectralWindow so center can be derived
    dummy_freq = np.linspace(_FREQ_LO, _FREQ_HI, 50)
    dummy_spec = np.zeros(50, dtype=complex)
    fr.window = SpectralWindow(
        parent_ft=None,
        freq_array=dummy_freq,
        complex_spectrum=dummy_spec,
        freq_range=(_FREQ_LO, _FREQ_HI),
    )
    if audit_trail is not None:
        fr.audit_trail = audit_trail
    if rescue_events is not None:
        fr.rescue_events = rescue_events
    return fr


# ---------------------------------------------------------------------------
# Tests: offset → molecular conversion
# ---------------------------------------------------------------------------


class TestMolecularConversion:
    """Verify _to_molecular handles both sideband signs."""

    def test_lower_sideband(self):
        # lower sideband: mol = center - offset (s = -1)
        mol = _to_molecular(2.0, 36010.0, _LOWER)
        assert mol == pytest.approx(36008.0, abs=1e-9)

    def test_upper_sideband(self):
        # upper sideband: mol = center + offset (s = +1)
        mol = _to_molecular(2.0, 36010.0, _UPPER)
        assert mol == pytest.approx(36012.0, abs=1e-9)

    def test_negative_offset_lower(self):
        mol = _to_molecular(-3.0, 36010.0, _LOWER)
        assert mol == pytest.approx(36013.0, abs=1e-9)

    def test_negative_offset_upper(self):
        mol = _to_molecular(-3.0, 36010.0, _UPPER)
        assert mol == pytest.approx(36007.0, abs=1e-9)

    def test_zero_offset(self):
        for sb in (_LOWER, _UPPER):
            mol = _to_molecular(0.0, 36010.0, sb)
            assert mol == pytest.approx(36010.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests: deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_no_candidates(self):
        assert _dedup_and_merge([], tol_mhz=0.02) == []

    def test_single_candidate(self):
        raw = [
            {
                "offset_mhz": 1.0,
                "freq_mhz": 36011.0,
                "evidence": 5.0,
                "kind": "residual_snr",
                "reason": "r",
                "site": "s",
                "amplitude": 1.2,
            }
        ]
        merged = _dedup_and_merge(raw, tol_mhz=0.02)
        assert len(merged) == 1
        assert merged[0]["freq_mhz"] == pytest.approx(36011.0)
        assert merged[0]["evidence"] == pytest.approx(5.0)

    def test_two_far_apart_are_kept(self):
        raw = [
            {
                "offset_mhz": 1.0,
                "freq_mhz": 36011.0,
                "evidence": 5.0,
                "kind": "residual_snr",
                "reason": "r1",
                "site": "add-loop:reject",
                "amplitude": None,
            },
            {
                "offset_mhz": 2.0,
                "freq_mhz": 36012.0,
                "evidence": 6.0,
                "kind": "residual_snr",
                "reason": "r2",
                "site": "rescue-round:0",
                "amplitude": 2.0,
            },
        ]
        merged = _dedup_and_merge(raw, tol_mhz=0.02)
        assert len(merged) == 2

    def test_two_within_tol_merge(self):
        """Two hits within tolerance collapse to one; best evidence wins."""
        raw = [
            {
                "offset_mhz": 1.00,
                "freq_mhz": 36011.000,
                "evidence": 4.0,
                "kind": "residual_snr",
                "reason": "low_aic",
                "site": "add-loop:reject",
                "amplitude": None,
            },
            {
                "offset_mhz": 1.01,
                "freq_mhz": 36011.010,  # 10 kHz apart — within 20 kHz tol
                "evidence": 8.0,
                "kind": "residual_snr",
                "reason": "rescue-candidate",
                "site": "rescue-round:0",
                "amplitude": 3.0,
            },
        ]
        merged = _dedup_and_merge(raw, tol_mhz=0.02)
        assert len(merged) == 1
        # Best evidence is 8.0 (second item)
        assert merged[0]["evidence"] == pytest.approx(8.0)
        # Both reasons accumulated
        assert "low_aic" in merged[0]["reason"]
        assert "rescue-candidate" in merged[0]["reason"]
        # Both sites accumulated
        assert len(merged[0]["sites"]) == 2
        # Amplitude from the rescue candidate
        assert merged[0]["amplitude"] == pytest.approx(3.0)

    def test_three_multi_round_collapse(self):
        """Three rescue rounds for the same frequency deduplicate to one."""
        base = 36005.0
        raw = [
            {
                "offset_mhz": -5.0 + i * 0.005,
                "freq_mhz": base + i * 0.005,
                "evidence": 3.5 + i,
                "kind": "residual_snr",
                "reason": "rescue-candidate",
                "site": f"rescue-round:{i}",
                "amplitude": 1.0 + i,
            }
            for i in range(3)
        ]
        merged = _dedup_and_merge(raw, tol_mhz=0.02)
        assert len(merged) == 1
        assert merged[0]["evidence"] == pytest.approx(5.5)  # 3.5 + 2
        assert len(merged[0]["sites"]) == 3


# ---------------------------------------------------------------------------
# Tests: derive_candidate_ledger
# ---------------------------------------------------------------------------


class TestDeriveCandidateLedger:
    def test_empty_returns_empty(self):
        fr = _make_fitting_result()
        result = derive_candidate_ledger(fr, center_mhz=_CENTER, sideband=_LOWER)
        assert result == []

    def test_accept_steps_excluded(self):
        """Accepted and seeded steps are not revivable."""
        fr = _make_fitting_result(
            audit_trail=[
                _make_audit_step(1.0, decision="accept"),
                _make_audit_step(2.0, decision="seed"),
                _make_audit_step(3.0, decision="promote"),
            ]
        )
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert result == []

    def test_reject_included(self):
        fr = _make_fitting_result(
            audit_trail=[
                _make_audit_step(
                    1.0,
                    decision="reject",
                    reason="aicc",
                    chi2_before=100.0,
                    chi2_after=90.0,
                ),
            ]
        )
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
        cand = result[0]
        # Lower sideband: mol = 36010 - 1 = 36009
        assert cand.frequency_mhz == pytest.approx(36009.0, abs=1e-9)
        assert cand.seed_offset_mhz == pytest.approx(1.0)
        assert cand.window_id == 0
        assert "add-loop:reject" in cand.decision_sites

    def test_tentative_included(self):
        fr = _make_fitting_result(
            audit_trail=[
                _make_audit_step(2.0, decision="tentative"),
            ]
        )
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
        assert "add-loop:tentative" in result[0].decision_sites

    def test_rescue_candidate_included(self):
        cand_info = RescueCandidateInfo(frequency_mhz=-3.0, magnitude=5.0, snr=7.0)
        rnd = _make_rescue_round([cand_info])
        fr = _make_fitting_result(rescue_events=[rnd])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=0.0
        )
        assert len(result) == 1
        cand = result[0]
        # Upper sideband: mol = 36010 + (-3) = 36007
        assert cand.frequency_mhz == pytest.approx(36007.0, abs=1e-9)
        assert cand.evidence_kind == "residual_snr"
        assert cand.seed_amplitude == pytest.approx(5.0)

    def test_bar_filters_low_snr(self):
        """Candidates below the bar are excluded."""
        cand_low = RescueCandidateInfo(frequency_mhz=1.0, magnitude=0.5, snr=1.5)
        cand_high = RescueCandidateInfo(frequency_mhz=2.0, magnitude=4.0, snr=5.0)
        rnd = _make_rescue_round([cand_low, cand_high])
        fr = _make_fitting_result(rescue_events=[rnd])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=3.0
        )
        # Only the high-SNR one passes the bar
        assert len(result) == 1
        # Upper: mol = 36010 + 2 = 36012
        assert result[0].frequency_mhz == pytest.approx(36012.0, abs=1e-9)

    def test_installed_candidate_excluded(self):
        """A rescue candidate that became a fitted peak is not revivable.

        ``RescueRoundInfo.candidates`` records every detected candidate,
        including those the sub-fit then accepted.  Such a candidate now sits
        in the fitted line list and must be subtracted from the ledger; a
        genuinely-rejected candidate at a different frequency must survive.
        """
        installed = RescueCandidateInfo(frequency_mhz=2.0, magnitude=6.0, snr=9.0)
        rejected = RescueCandidateInfo(frequency_mhz=-4.0, magnitude=5.0, snr=8.0)
        rnd = _make_rescue_round([installed, rejected])
        fr = _make_fitting_result(rescue_events=[rnd])
        # Upper sideband: the installed candidate's molecular freq is 36012.
        fr.fitted_peaks = [FittedPeak(peak_id=0, frequency_mhz=36012.0, amplitude=6.0)]
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=3.0
        )
        freqs = [c.frequency_mhz for c in result]
        # The installed one (36012) is gone; the rejected one (36006) survives.
        assert all(abs(f - 36012.0) > _DEDUP_TOL_MHZ for f in freqs)
        assert any(abs(f - 36006.0) <= _DEDUP_TOL_MHZ for f in freqs)

    def test_dedup_across_audit_and_rescue(self):
        """One offset appearing in both audit and rescue collapses to one entry."""
        step = _make_audit_step(
            2.0,
            decision="reject",
            reason="aicc",
            chi2_before=100.0,
            chi2_after=90.0,
        )
        # Rescue sees the same frequency (within tolerance)
        cand_info = RescueCandidateInfo(frequency_mhz=2.005, magnitude=6.0, snr=8.0)
        rnd = _make_rescue_round([cand_info])
        fr = _make_fitting_result(audit_trail=[step], rescue_events=[rnd])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1  # deduped
        # Both sites should appear
        assert any("add-loop" in s for s in result[0].decision_sites)
        assert any("rescue-round" in s for s in result[0].decision_sites)

    def test_sorted_by_freq(self):
        """Candidates are returned sorted by molecular frequency."""
        fr = _make_fitting_result(
            audit_trail=[
                _make_audit_step(
                    5.0, decision="reject", chi2_before=100.0, chi2_after=90.0
                ),
                _make_audit_step(
                    1.0, decision="reject", chi2_before=100.0, chi2_after=90.0
                ),
                _make_audit_step(
                    3.0, decision="reject", chi2_before=100.0, chi2_after=90.0
                ),
            ]
        )
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        freqs = [c.frequency_mhz for c in result]
        assert freqs == sorted(freqs)

    def test_evidence_kind_aicc(self):
        """When aicc_delta is available it is used for evidence."""
        step = _make_audit_step(
            1.0,
            decision="reject",
            aicc_delta=2.5,
        )
        fr = _make_fitting_result(audit_trail=[step])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
        assert result[0].evidence_kind == "aicc_delta"
        assert result[0].best_evidence == pytest.approx(2.5)

    def test_evidence_kind_pvalue_fallback(self):
        """When aicc_delta is nan but p_value is available, use f_p."""
        step = _make_audit_step(
            1.0,
            decision="reject",
            aicc_delta=float("nan"),
            p_value=0.01,
        )
        fr = _make_fitting_result(audit_trail=[step])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
        assert result[0].evidence_kind == "f_p"
        assert result[0].best_evidence == pytest.approx(0.01)

    def test_window_id_attached(self):
        """window_id from the FittingResult is propagated to each candidate."""
        fr = _make_fitting_result(
            audit_trail=[
                _make_audit_step(
                    1.0, decision="reject", chi2_before=100.0, chi2_after=90.0
                )
            ],
            window_id=42,
        )
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=0.0
        )
        assert all(c.window_id == 42 for c in result)

    def test_multi_round_rescue_dedup(self):
        """The same line appearing in rescue rounds 0 and 1 deduplicates."""
        c0 = RescueCandidateInfo(frequency_mhz=2.000, magnitude=3.0, snr=4.0)
        c1 = RescueCandidateInfo(frequency_mhz=2.008, magnitude=5.0, snr=7.0)
        rnd0 = _make_rescue_round([c0], round_idx=0)
        rnd1 = _make_rescue_round([c1], round_idx=1)
        fr = _make_fitting_result(rescue_events=[rnd0, rnd1])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=0.0
        )
        assert len(result) == 1  # deduped
        assert result[0].best_evidence == pytest.approx(7.0)
        assert len(result[0].decision_sites) == 2

    def test_upper_lower_sideband_symmetry(self):
        """Upper and lower sidebands map the same offset to mirror-image frequencies."""
        step = _make_audit_step(
            5.0, decision="reject", chi2_before=100.0, chi2_after=90.0
        )
        fr = _make_fitting_result(audit_trail=[step])
        r_lower = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        r_upper = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_UPPER, bar=0.0
        )
        # lower: 36010 - 5 = 36005; upper: 36010 + 5 = 36015
        assert r_lower[0].frequency_mhz == pytest.approx(36005.0, abs=1e-9)
        assert r_upper[0].frequency_mhz == pytest.approx(36015.0, abs=1e-9)
        assert r_lower[0].seed_offset_mhz == r_upper[0].seed_offset_mhz


# ---------------------------------------------------------------------------
# Tests: LedgerCandidate dataclass
# ---------------------------------------------------------------------------


class TestLedgerCandidateDataclass:
    def test_construction(self):
        lc = LedgerCandidate(
            frequency_mhz=36010.5,
            seed_offset_mhz=0.5,
            seed_amplitude=2.3,
            best_evidence=5.1,
            evidence_kind="residual_snr",
            reasons=["reject"],
            decision_sites=["add-loop:reject"],
            window_id=7,
        )
        assert lc.frequency_mhz == pytest.approx(36010.5)
        assert lc.window_id == 7
        assert lc.evidence_kind == "residual_snr"
        assert lc.seed_amplitude is not None

    def test_none_amplitude_allowed(self):
        lc = LedgerCandidate(
            frequency_mhz=36010.0,
            seed_offset_mhz=0.0,
            seed_amplitude=None,
            best_evidence=3.0,
            evidence_kind="delta_chi2",
            reasons=[],
            decision_sites=[],
            window_id=0,
        )
        assert lc.seed_amplitude is None


# ---------------------------------------------------------------------------
# C1 AICc-reject semantic fix tests (Task 4)
# ---------------------------------------------------------------------------


class TestAICcRejectSemanticFix:
    """Verify the corrected AICc-reject evidence direction.

    A *marginal* reject (aicc_delta near 0) must surface; a *decisive* reject
    (large aicc_delta) must be filtered out. The same near-gate rule applies to
    ``tentative`` decisions -- ``aicc_delta`` is the K+1 model's cost, not
    support, so a never-significant tentative is no more revivable than a
    reject.

    The rule: kind="aicc_delta", evidence=aicc_delta (raw positive delta);
    _passes_bar passes when evidence <= _NEAR_GATE_FACTOR.
    """

    def _ledger_from_step(
        self, aicc_delta: float, bar: float = DEFAULT_DISPLAY_BAR
    ) -> list:
        step = _make_audit_step(
            1.0,
            decision="reject",
            aicc_delta=aicc_delta,
        )
        fr = _make_fitting_result(audit_trail=[step])
        return derive_candidate_ledger(fr, center_mhz=_CENTER, sideband=_LOWER, bar=bar)

    def test_marginal_aicc_reject_surfaces(self):
        """A near-zero aicc_delta (marginal reject) must appear in the ledger."""
        # aicc_delta = 0.5 is well within _NEAR_GATE_FACTOR = 10.0
        result = self._ledger_from_step(aicc_delta=0.5)
        assert (
            len(result) == 1
        ), "Marginal AICc-reject (delta=0.5) must surface in the ledger"
        assert result[0].evidence_kind == "aicc_delta"
        assert result[0].best_evidence == pytest.approx(0.5)

    def test_decisive_aicc_reject_filtered(self):
        """A large aicc_delta (decisive reject) must be filtered out."""
        # aicc_delta = 100.0 >> _NEAR_GATE_FACTOR = 10.0 -> filtered
        result = self._ledger_from_step(aicc_delta=100.0)
        assert (
            len(result) == 0
        ), "Decisive AICc-reject (delta=100) should be filtered from the ledger"

    def test_boundary_at_near_gate_factor(self):
        """A candidate at exactly aicc_delta = _NEAR_GATE_FACTOR still passes."""
        from ftmwpipeline._internal.stage6_impl import _NEAR_GATE_FACTOR

        result = self._ledger_from_step(aicc_delta=float(_NEAR_GATE_FACTOR))
        assert len(result) == 1, (
            "Candidate at the _NEAR_GATE_FACTOR boundary (delta=%.1f) must pass"
            % _NEAR_GATE_FACTOR
        )

    def test_just_above_near_gate_factor_filtered(self):
        """A candidate just above _NEAR_GATE_FACTOR is filtered."""
        from ftmwpipeline._internal.stage6_impl import _NEAR_GATE_FACTOR

        result = self._ledger_from_step(aicc_delta=float(_NEAR_GATE_FACTOR) + 1.0)
        assert len(result) == 0, "Candidate above _NEAR_GATE_FACTOR must be filtered"

    def test_tentative_follows_near_gate_rule_like_reject(self):
        """A ``tentative`` is near-gate filtered on ``aicc_delta``, like ``reject``.

        ``aicc_delta`` is the AICc *cost* of the K+1 model, not support for the
        line. A ``tentative`` ("held pending a jointly-significant batch") that
        never became significant carries a large positive delta and is no more
        revivable than a decisive reject -- so it follows the same near-gate
        rule (pass only when ``aicc_delta <= _NEAR_GATE_FACTOR``). Surfacing
        large-delta tentatives unconditionally was the candidate_bearing
        over-surfacing bug; a near-gate tentative still surfaces.
        """
        # Large delta (23 >> _NEAR_GATE_FACTOR): tentative is filtered, same as
        # a reject at the same delta.
        big = _make_audit_step(1.0, decision="tentative", aicc_delta=23.0)
        fr_big = _make_fitting_result(audit_trail=[big])
        assert (
            len(
                derive_candidate_ledger(
                    fr_big, center_mhz=_CENTER, sideband=_LOWER, bar=DEFAULT_DISPLAY_BAR
                )
            )
            == 0
        ), "A large-delta tentative (23) must be filtered, like a reject"
        assert len(self._ledger_from_step(aicc_delta=23.0)) == 0

        # Near-gate delta (within _NEAR_GATE_FACTOR): tentative surfaces.
        small = _make_audit_step(1.0, decision="tentative", aicc_delta=3.0)
        fr_small = _make_fitting_result(audit_trail=[small])
        near = derive_candidate_ledger(
            fr_small, center_mhz=_CENTER, sideband=_LOWER, bar=DEFAULT_DISPLAY_BAR
        )
        assert len(near) == 1, "A near-gate tentative (3) must surface"
        assert near[0].evidence_kind == "aicc_delta"

    def test_rescue_snr_path_unchanged(self):
        """Rescue-SNR candidates still use the SNR >= bar rule (path unchanged)."""
        from ftmwpipeline.core.data_structures import RescueCandidateInfo

        cand = RescueCandidateInfo(frequency_mhz=1.0, snr=6.0, magnitude=0.5)
        rnd = _make_rescue_round([cand])
        fr = _make_fitting_result(rescue_events=[rnd])
        # bar=4.0, SNR=6.0 -> passes
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=4.0
        )
        assert len(result) == 1
        assert result[0].evidence_kind == "residual_snr"
        # SNR below bar: filtered
        cand_low = RescueCandidateInfo(frequency_mhz=1.0, snr=2.0, magnitude=0.2)
        rnd_low = _make_rescue_round([cand_low])
        fr_low = _make_fitting_result(rescue_events=[rnd_low])
        result_low = derive_candidate_ledger(
            fr_low, center_mhz=_CENTER, sideband=_LOWER, bar=4.0
        )
        assert len(result_low) == 0

    def test_nan_aicc_delta_falls_back_to_pvalue(self):
        """When aicc_delta is nan the f_p path is unchanged."""
        step = _make_audit_step(
            1.0,
            decision="reject",
            aicc_delta=float("nan"),
            p_value=0.01,
        )
        fr = _make_fitting_result(audit_trail=[step])
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
        assert result[0].evidence_kind == "f_p"


# ---------------------------------------------------------------------------
# Tests: F-3 brightness-scaled shape-error filter
# ---------------------------------------------------------------------------

_RES_ELEMENT = 0.08  # MHz; ~1 / 12.65 us acquisition


class TestShapeErrorReach:
    """The brightness-scaled shape-error filter (``SHAPE_ERROR_REACH_KAPPA``).

    A candidate is a lineshape sidelobe -- dropped -- when
    ``sep_res <= kappa * snr / evidence`` for some fitted peak; the reach grows
    with the neighbor line's SNR, so a bright line's far sidelobes are filtered
    while a genuine companion of a faint line survives.
    """

    def test_bright_line_sidelobe_filtered(self):
        """A candidate 5 res from a snr-30000 line is its lineshape sidelobe."""
        # Lower sideband: mol = center - offset. Bright fitted peak at offset 5
        # MHz -> mol 36005; candidate 5 res (0.4 MHz) further out -> mol 36004.6.
        cand = RescueCandidateInfo(frequency_mhz=5.4, magnitude=30.0, snr=30.0)
        fr = _make_fitting_result(rescue_events=[_make_rescue_round([cand])])
        fr.fitted_peaks = [
            FittedPeak(peak_id=0, frequency_mhz=36005.0, amplitude=1.0, snr=30000.0)
        ]
        # Reach = 0.2 * 30000 / 30 = 200 res >> 5 res -> dropped.
        result = derive_candidate_ledger(
            fr,
            center_mhz=_CENTER,
            sideband=_LOWER,
            bar=0.0,
            res_element_mhz=_RES_ELEMENT,
        )
        assert result == []

    def test_faint_line_companion_kept(self):
        """A companion 3 res from a snr-20 line is not inside its tiny shadow."""
        # Faint fitted peak at offset 2 MHz -> mol 36008; candidate 3 res
        # (0.24 MHz) out -> mol 36007.76, evidence 12.
        cand = RescueCandidateInfo(frequency_mhz=2.24, magnitude=12.0, snr=12.0)
        fr = _make_fitting_result(rescue_events=[_make_rescue_round([cand])])
        fr.fitted_peaks = [
            FittedPeak(peak_id=0, frequency_mhz=36008.0, amplitude=1.0, snr=20.0)
        ]
        # Reach = 0.2 * 20 / 12 = 0.33 res < 3 res -> kept.
        result = derive_candidate_ledger(
            fr,
            center_mhz=_CENTER,
            sideband=_LOWER,
            bar=0.0,
            res_element_mhz=_RES_ELEMENT,
        )
        assert len(result) == 1
        assert result[0].frequency_mhz == pytest.approx(36007.76, abs=1e-6)

    def test_filter_disabled_without_res_element(self):
        """Without ``res_element_mhz`` the sidelobe survives (filter off)."""
        cand = RescueCandidateInfo(frequency_mhz=5.4, magnitude=30.0, snr=30.0)
        fr = _make_fitting_result(rescue_events=[_make_rescue_round([cand])])
        fr.fitted_peaks = [
            FittedPeak(peak_id=0, frequency_mhz=36005.0, amplitude=1.0, snr=30000.0)
        ]
        result = derive_candidate_ledger(
            fr, center_mhz=_CENTER, sideband=_LOWER, bar=0.0
        )
        assert len(result) == 1
