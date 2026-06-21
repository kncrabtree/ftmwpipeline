"""
Tests for instrument clock-source extraction and recommended-layer persistence.

Covers:
1. BlackchirpLoader._extract_clock_sources against the real 2638 example data.
2. Round-trip of write_recommended_clock_sources / read_recommended_clock_sources.
3. End-to-end import of the 2638 example: the loader populates metadata, Stage 0
   wires it into the .ftmw, and the Stage 5 resolver picks it up at the
   recommended layer (with override semantics verified).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import h5py
import pandas as pd
import pytest

from ftmwpipeline.core.stage_fit_settings import (
    ClockSource,
    SpurSubSettings,
    StageFitSettings,
    coerce_clock_sources,
)
from ftmwpipeline.core.stage_fit_settings import resolve as resolve_stage_fit_settings
from ftmwpipeline.io.data_loaders.blackchirp import BlackChirpLoader
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_recommended_clock_sources,
    write_recommended_clock_sources,
)

# Path to the real 2638 example data checked into the repo.
_DATA_2638 = Path("examples/blackchirp_data/2638")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_clock(
    sources: Tuple[ClockSource, ...], freq_mhz: float
) -> Optional[ClockSource]:
    """Return the first ClockSource whose freq_mhz is close to ``freq_mhz``."""
    for cs in sources:
        if abs(cs.freq_mhz - freq_mhz) < 0.001:
            return cs
    return None


# ---------------------------------------------------------------------------
# 1. Loader extraction against 2638 example data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATA_2638.exists(),
    reason="examples/blackchirp_data/2638 not present",
)
class TestClockSourceExtraction:
    """BlackChirpLoader._extract_clock_sources against 2638 real files."""

    @pytest.fixture(scope="class")
    def extracted(self) -> list:
        raw = BlackChirpLoader._extract_clock_sources(_DATA_2638)
        assert raw is not None, "Expected clock sources from 2638, got None"
        return raw

    def test_returns_four_entries(self, extracted) -> None:
        # 2638: DownLO fundamental 5120, UpLO fundamental 5760, AWG 16000, digitizer 50000
        assert len(extracted) == 4

    def test_downlo_fundamental(self, extracted) -> None:
        """DownLO 40960 / factor 8 = 5120 MHz, locked."""
        sources = coerce_clock_sources(extracted)
        assert sources is not None
        cs = _find_clock(sources, 5120.0)
        assert cs is not None, "Expected 5120 MHz entry (DownLO fundamental)"
        assert cs.locked is True
        assert "downlo" in cs.label.lower() or "clock.0" in cs.label.lower()

    def test_uplo_fundamental(self, extracted) -> None:
        """UpLO 11520 / factor 2 = 5760 MHz, locked."""
        sources = coerce_clock_sources(extracted)
        assert sources is not None
        cs = _find_clock(sources, 5760.0)
        assert cs is not None, "Expected 5760 MHz entry (UpLO fundamental)"
        assert cs.locked is True

    def test_awg_entry(self, extracted) -> None:
        """ChirpConfig SampleRate 16000 MHz, locked, label contains 'awg'."""
        sources = coerce_clock_sources(extracted)
        assert sources is not None
        cs = _find_clock(sources, 16000.0)
        assert cs is not None, "Expected 16000 MHz AWG entry"
        assert cs.locked is True
        assert "awg" in cs.label.lower()

    def test_digitizer_entry(self, extracted) -> None:
        """FtmwDigitizer.0 SampleRate 5e10 Hz → 50000 MHz, unlocked."""
        sources = coerce_clock_sources(extracted)
        assert sources is not None
        cs = _find_clock(sources, 50000.0)
        assert cs is not None, "Expected 50000 MHz digitizer entry"
        assert cs.locked is False
        assert "digitizer" in cs.label.lower()

    def test_no_duplicate_fundamentals(self, extracted) -> None:
        """Both clocks.csv rows share Clock.0; only one entry per fundamental."""
        sources = coerce_clock_sources(extracted)
        assert sources is not None
        freqs = [round(cs.freq_mhz, 3) for cs in sources]
        assert len(freqs) == len(
            set(freqs)
        ), "Duplicate fundamentals in extracted clocks"


class TestClockOperationNormalization:
    """``_clock_operation`` accepts the string and integer-enum forms."""

    def test_string_forms(self) -> None:
        assert BlackChirpLoader._clock_operation("Multiply") == "Multiply"
        assert BlackChirpLoader._clock_operation("Divide") == "Divide"

    def test_integer_enum_forms(self) -> None:
        # Older Blackchirp metadata writes the operation as an integer enum.
        assert BlackChirpLoader._clock_operation(0) == "Multiply"
        assert BlackChirpLoader._clock_operation(1) == "Divide"
        assert BlackChirpLoader._clock_operation("0") == "Multiply"
        assert BlackChirpLoader._clock_operation("1") == "Divide"

    def test_unknown_is_none(self) -> None:
        assert BlackChirpLoader._clock_operation("garbage") is None
        assert BlackChirpLoader._clock_operation(7) is None


class TestDigitizerKeyDiscovery:
    """``_digitizer_hw_key`` resolves both the bare and suffixed conventions."""

    @staticmethod
    def _exp(keys):
        return SimpleNamespace(hardware=pd.DataFrame({"key": keys}))

    def test_bare_key(self) -> None:
        exp = self._exp(["AWG", "Clock0", "FtmwDigitizer", "PulseGenerator"])
        assert BlackChirpLoader._digitizer_hw_key(exp) == "FtmwDigitizer"

    def test_suffixed_key(self) -> None:
        exp = self._exp(["AWG.0", "Clock.0", "FtmwDigitizer.0"])
        assert BlackChirpLoader._digitizer_hw_key(exp) == "FtmwDigitizer.0"

    def test_no_digitizer_is_none(self) -> None:
        assert BlackChirpLoader._digitizer_hw_key(self._exp(["AWG", "Clock0"])) is None

    def test_no_hardware_is_none(self) -> None:
        assert (
            BlackChirpLoader._digitizer_hw_key(SimpleNamespace(hardware=None)) is None
        )


class TestHeaderRateUnitConversion:
    """``_header_rate_mhz`` converts to MHz via the declared header unit."""

    @staticmethod
    def _exp(value, unit):
        return SimpleNamespace(
            header_value=lambda o, v: value,
            header_unit=lambda o, v: unit,
        )

    def test_hz_to_mhz(self) -> None:
        assert BlackChirpLoader._header_rate_mhz(self._exp(5e10, "Hz"), "D", "S") == 5e4

    def test_mhz_passthrough(self) -> None:
        assert (
            BlackChirpLoader._header_rate_mhz(self._exp(16000, "MHz"), "A", "S")
            == 16000.0
        )

    def test_absent_unit_defaults_mhz(self) -> None:
        assert (
            BlackChirpLoader._header_rate_mhz(self._exp(16000, None), "A", "S")
            == 16000.0
        )

    def test_missing_value_is_none(self) -> None:
        def _raise(o, v):
            raise KeyError("no such header row")

        exp = SimpleNamespace(header_value=_raise, header_unit=lambda o, v: None)
        assert BlackChirpLoader._header_rate_mhz(exp, "A", "S") is None


class _FakeExp:
    """Stand-in for ``BCExperiment`` carrying the older metadata format.

    Numeric ``Operation`` / ``ClockType``, bare hardware keys (``Clock0``,
    ``FtmwDigitizer``), and the digitizer ``SampleRate`` in Hz -- the shape that
    made the hand-rolled parser drop the (unlocked) digitizer.
    """

    def __init__(self) -> None:
        self.clocks = pd.DataFrame(
            {
                "Index": [0, 0],
                "ClockType": [0, 1],
                "FreqMHz": [11520, 40960],
                "Operation": [0, 0],  # numeric enum: 0 == Multiply
                "Factor": [2, 8],
                "HwKey": ["Clock0", "Clock0"],
                "OutputNum": [0, 1],
            }
        )
        self.hardware = pd.DataFrame(
            {"key": ["AWG", "Clock0", "FtmwDigitizer", "PulseGenerator"]}
        )
        self._header = {
            ("ChirpConfig", "SampleRate"): (16000.0, "MHz"),
            ("FtmwDigitizer", "SampleRate"): (5e10, "Hz"),
        }

    def header_value(self, obj_key, value_key):
        if (obj_key, value_key) not in self._header:
            raise KeyError(f"no header row for {obj_key}/{value_key}")
        return self._header[(obj_key, value_key)][0]

    def header_unit(self, obj_key, value_key):
        return self._header.get((obj_key, value_key), (None, None))[1]


class TestOldFormatExtraction:
    """The full extractor on the older metadata format (regression guard).

    Reproduces the failure mode where a free-running digitizer was reported as
    absent / locked: numeric Operation must still divide by Factor, and the
    bare ``FtmwDigitizer`` key must still yield the unlocked digitizer entry.
    """

    @pytest.fixture
    def extracted(self, monkeypatch) -> list:
        import blackchirp

        monkeypatch.setattr(blackchirp, "BCExperiment", lambda _path: _FakeExp())
        raw = BlackChirpLoader._extract_clock_sources(Path("ignored"))
        assert raw is not None
        return raw

    def test_fundamentals_divided_by_factor(self, extracted) -> None:
        freqs = {round(e["freq_mhz"], 1) for e in extracted}
        # 11520/2 = 5760, 40960/8 = 5120 -- the numeric Operation is applied.
        assert 5760.0 in freqs and 5120.0 in freqs
        assert 11520.0 not in freqs and 40960.0 not in freqs

    def test_digitizer_present_and_unlocked(self, extracted) -> None:
        dig = next((e for e in extracted if abs(e["freq_mhz"] - 50000.0) < 0.1), None)
        assert dig is not None, "bare FtmwDigitizer key must still be discovered"
        assert dig["locked"] is False

    def test_has_an_unlocked_source(self, extracted) -> None:
        # The defect was an all-locked declaration -> calibration state rb_locked.
        assert any(not e["locked"] for e in extracted)


class TestClockSourceExtractionMissing:
    """Graceful handling when the experiment cannot be read."""

    def test_returns_none_for_empty_dir(self, tmp_path) -> None:
        """A junk directory (no readable experiment) returns None, not an error."""
        assert BlackChirpLoader._extract_clock_sources(tmp_path) is None

    def test_returns_none_when_experiment_unreadable(self, monkeypatch) -> None:
        import blackchirp

        def _boom(_path):
            raise RuntimeError("cannot parse experiment")

        monkeypatch.setattr(blackchirp, "BCExperiment", _boom)
        assert BlackChirpLoader._extract_clock_sources(Path("x")) is None


# ---------------------------------------------------------------------------
# 2. Round-trip: write / read_recommended_clock_sources
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_stage0_ftmw(tmp_path) -> str:
    """An .ftmw file with a stage0_fid_data group but no clock attr."""
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("stage0_fid_data")
    return str(p)


@pytest.fixture
def bare_ftmw_no_stage0(tmp_path) -> str:
    """An .ftmw file without a stage0_fid_data group."""
    p = tmp_path / "no_stage0.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("other_group")
    return str(p)


class TestRecommendedClocksRoundTrip:
    def test_read_absent_returns_none(self, bare_stage0_ftmw) -> None:
        assert read_recommended_clock_sources(bare_stage0_ftmw) is None

    def test_write_none_reads_none(self, bare_stage0_ftmw) -> None:
        write_recommended_clock_sources(bare_stage0_ftmw, None)
        assert read_recommended_clock_sources(bare_stage0_ftmw) is None

    def test_no_stage0_group_write_noop(self, bare_ftmw_no_stage0) -> None:
        """write is a no-op when stage0_fid_data is absent."""
        clocks = (ClockSource(freq_mhz=5120.0, locked=True, label="test"),)
        write_recommended_clock_sources(bare_ftmw_no_stage0, clocks)
        assert read_recommended_clock_sources(bare_ftmw_no_stage0) is None

    def test_round_trip_single_clock(self, bare_stage0_ftmw) -> None:
        clocks = (ClockSource(freq_mhz=5120.0, locked=True, label="downconv-ref"),)
        write_recommended_clock_sources(bare_stage0_ftmw, clocks)
        result = read_recommended_clock_sources(bare_stage0_ftmw)
        assert result is not None
        assert len(result) == 1
        assert result[0].freq_mhz == 5120.0
        assert result[0].locked is True
        assert result[0].label == "downconv-ref"

    def test_round_trip_mixed_locked(self, bare_stage0_ftmw) -> None:
        clocks = (
            ClockSource(freq_mhz=5760.0, locked=True, label="upconv-lo"),
            ClockSource(freq_mhz=5120.0, locked=True, label="downconv-ref"),
            ClockSource(freq_mhz=16000.0, locked=True, label="awg"),
            ClockSource(freq_mhz=50000.0, locked=False, label="digitizer"),
        )
        write_recommended_clock_sources(bare_stage0_ftmw, clocks)
        result = read_recommended_clock_sources(bare_stage0_ftmw)
        assert result is not None
        assert len(result) == 4
        locked_freqs = {cs.freq_mhz for cs in result if cs.locked}
        unlocked_freqs = {cs.freq_mhz for cs in result if not cs.locked}
        assert locked_freqs == {5760.0, 5120.0, 16000.0}
        assert unlocked_freqs == {50000.0}

    def test_overwrite_replaces_prior(self, bare_stage0_ftmw) -> None:
        """Second write replaces the first (re-import semantics)."""
        first = (ClockSource(freq_mhz=5120.0, locked=True, label="first"),)
        second = (ClockSource(freq_mhz=9999.0, locked=False, label="second"),)
        write_recommended_clock_sources(bare_stage0_ftmw, first)
        write_recommended_clock_sources(bare_stage0_ftmw, second)
        result = read_recommended_clock_sources(bare_stage0_ftmw)
        assert result is not None
        assert len(result) == 1
        assert result[0].freq_mhz == 9999.0

    def test_empty_tuple_reads_back_empty(self, bare_stage0_ftmw) -> None:
        write_recommended_clock_sources(bare_stage0_ftmw, ())
        result = read_recommended_clock_sources(bare_stage0_ftmw)
        assert result is not None
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. End-to-end import + Stage 5 resolver integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATA_2638.exists(),
    reason="examples/blackchirp_data/2638 not present",
)
class TestImportEndToEnd:
    """Import 2638 into a tmp .ftmw and verify the clock recommendation chain."""

    @pytest.fixture(scope="class")
    def imported_ftmw(self, tmp_path_factory) -> str:
        tmp = tmp_path_factory.mktemp("clock_e2e")
        ftmw_path = str(tmp / "exp_2638.ftmw")
        import ftmwpipeline.api as ftmw_api

        ftmw_api.import_data(ftmw_path, source=str(_DATA_2638))
        return ftmw_path

    def test_recommended_clocks_present(self, imported_ftmw) -> None:
        result = read_recommended_clock_sources(imported_ftmw)
        assert result is not None
        assert len(result) == 4

    def test_expected_fundamentals_present(self, imported_ftmw) -> None:
        result = read_recommended_clock_sources(imported_ftmw)
        assert result is not None
        freqs = {round(cs.freq_mhz, 1) for cs in result}
        assert 5120.0 in freqs, "DownLO fundamental 5120 missing"
        assert 5760.0 in freqs, "UpLO fundamental 5760 missing"
        assert 16000.0 in freqs, "AWG 16000 missing"
        assert 50000.0 in freqs, "Digitizer 50000 missing"

    def test_digitizer_is_unlocked(self, imported_ftmw) -> None:
        result = read_recommended_clock_sources(imported_ftmw)
        assert result is not None
        dig = next((cs for cs in result if abs(cs.freq_mhz - 50000.0) < 0.1), None)
        assert dig is not None
        assert dig.locked is False

    def test_resolver_picks_up_recommended_clocks_when_no_explicit(
        self, imported_ftmw
    ) -> None:
        """When no explicit/persisted/preset layer sets clocks, the recommended
        layer supplies the declaration extracted at import."""
        from ftmwpipeline.core.stage_fit_settings import resolve as _resolve
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            read_recommended_clock_sources,
        )

        recommended_clocks = read_recommended_clock_sources(imported_ftmw)
        assert recommended_clocks is not None

        # Simulate the resolver with only the recommended layer active.
        recommended = StageFitSettings(spur=SpurSubSettings(clocks=recommended_clocks))
        resolved = _resolve(
            explicit=None,
            preset=None,
            persisted=None,
            recommended=recommended,
        )
        assert resolved.spur.clocks is not None
        assert len(resolved.spur.clocks) == 4

    def test_explicit_overrides_recommended(self, imported_ftmw) -> None:
        """An explicit clocks value replaces the recommended declaration."""
        from ftmwpipeline.core.stage_fit_settings import resolve as _resolve
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            read_recommended_clock_sources,
        )

        recommended_clocks = read_recommended_clock_sources(imported_ftmw)
        assert recommended_clocks is not None

        override = (ClockSource(freq_mhz=1234.0, locked=True, label="override"),)
        explicit = StageFitSettings(spur=SpurSubSettings(clocks=override))
        recommended = StageFitSettings(spur=SpurSubSettings(clocks=recommended_clocks))
        resolved = _resolve(
            explicit=explicit,
            preset=None,
            persisted=None,
            recommended=recommended,
        )
        assert resolved.spur.clocks is not None
        assert len(resolved.spur.clocks) == 1
        assert resolved.spur.clocks[0].freq_mhz == 1234.0
