"""
D7 regression tests: FT settings (including trim) persisted to ft_processing.

These tests verify that:
1. compute_ft_impl(persist=True) writes a complete canonical record to
   processing_parameters/ft_processing — including trim_min_mhz/trim_max_mhz.
2. A no-arg compute_ft_impl() reproduces the exact same spectrum as the
   original explicit call.
3. The persisted record survives a fresh open of the HDF5 file.
4. All three interfaces (CLI, Pipeline, api) produce identical ComplexFT
   objects and identical persisted records (gated: may fail until the
   public-API refactor lands).

Marked @pytest.mark.integration so they can be skipped on pure-unit runs.
"""

import json
import subprocess
import pytest
import numpy as np
import h5py

from pathlib import Path
from ftmwpipeline._internal.stage0_impl import import_data_impl
from ftmwpipeline._internal.stage1_impl import compute_ft_impl
from ftmwpipeline.core.settings import FTSettings, FT_PROCESSING_PATH
from ftmwpipeline.core.data_structures import ComplexFT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_ftmw(source: str, tmp_path: Path) -> Path:
    """Import experiment 2638 into a fresh .ftmw file and return its path."""
    dest = tmp_path / "exp_2638.ftmw"
    import_data_impl(str(dest), source=source)
    return dest


def _read_ft_attrs(file_path: str) -> dict:
    """Read the raw HDF5 attrs from processing_parameters/ft_processing."""
    with h5py.File(file_path, "r") as h5f:
        grp = h5f[FT_PROCESSING_PATH]
        return dict(grp.attrs)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["ftmwpipeline", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result


# ---------------------------------------------------------------------------
# Core D7 regression: impl-level persistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestD7ImplPersistence:
    """Verify compute_ft_impl persist=True writes and no-arg reproduce works."""

    def test_persist_writes_canonical_record(self, exp_2638_data_path, tmp_path):
        """persist=True must write trim, zpf, expf_us to ft_processing."""
        f = _create_ftmw(exp_2638_data_path, tmp_path)

        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        compute_ft_impl(str(f), settings=settings, persist=True)

        attrs = _read_ft_attrs(str(f))
        assert float(attrs["trim_min_mhz"]) == pytest.approx(26500.0)
        assert float(attrs["trim_max_mhz"]) == pytest.approx(40000.0)
        assert int(attrs["zpf"]) == 2
        assert float(attrs["expf_us"]) == pytest.approx(5.0)

    def test_stage1_marked_complete_after_persist(self, exp_2638_data_path, tmp_path):
        """pipeline_stages.completed_stages must include stage1_complex_ft."""
        f = _create_ftmw(exp_2638_data_path, tmp_path)
        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        compute_ft_impl(str(f), settings=settings, persist=True)

        with h5py.File(str(f), "r") as h5f:
            completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage1_complex_ft" in completed

    def test_noarg_reproduces_same_spectrum(self, exp_2638_data_path, tmp_path):
        """
        No-arg compute_ft_impl() must reproduce identical n_points, freq_array,
        and magnitude_spectrum as the original explicit call (D7: trim persisted).
        """
        f = _create_ftmw(exp_2638_data_path, tmp_path)

        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        first = compute_ft_impl(str(f), settings=settings, persist=True)
        ft1: ComplexFT = first["complex_ft"]

        # No-arg call — resolves through persisted record only
        second = compute_ft_impl(str(f))
        ft2: ComplexFT = second["complex_ft"]

        assert ft1.n_points == ft2.n_points, "No-arg reproduce: n_points mismatch"
        np.testing.assert_allclose(
            ft1.freq_array,
            ft2.freq_array,
            rtol=1e-12,
            atol=0,
            err_msg="No-arg reproduce: freq_array differs",
        )
        np.testing.assert_allclose(
            ft1.magnitude_spectrum,
            ft2.magnitude_spectrum,
            rtol=1e-12,
            atol=0,
            err_msg="No-arg reproduce: magnitude_spectrum differs",
        )

    def test_noarg_freq_range_matches_trim(self, exp_2638_data_path, tmp_path):
        """No-arg reproduced spectrum must lie within the persisted trim range."""
        f = _create_ftmw(exp_2638_data_path, tmp_path)

        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        compute_ft_impl(str(f), settings=settings, persist=True)

        result = compute_ft_impl(str(f))
        ft: ComplexFT = result["complex_ft"]

        assert (
            ft.freq_array.min() >= 26499.0
        ), f"freq min {ft.freq_array.min():.1f} < trim_min 26500"
        assert (
            ft.freq_array.max() <= 40001.0
        ), f"freq max {ft.freq_array.max():.1f} > trim_max 40000"

    def test_persisted_settings_survive_fresh_open(self, exp_2638_data_path, tmp_path):
        """
        Reading ft_processing attrs from a freshly opened h5 file must yield
        the same values as those written at persist time.
        """
        f = _create_ftmw(exp_2638_data_path, tmp_path)
        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        compute_ft_impl(str(f), settings=settings, persist=True)

        # Close implicit by function; open fresh
        attrs_fresh = _read_ft_attrs(str(f))
        assert float(attrs_fresh["trim_min_mhz"]) == pytest.approx(26500.0)
        assert float(attrs_fresh["trim_max_mhz"]) == pytest.approx(40000.0)
        assert int(attrs_fresh["zpf"]) == 2
        assert float(attrs_fresh["expf_us"]) == pytest.approx(5.0)

    def test_from_attrs_restores_persisted_record(self, exp_2638_data_path, tmp_path):
        """FTSettings.from_attrs(attrs) must round-trip the persisted record."""
        f = _create_ftmw(exp_2638_data_path, tmp_path)
        settings = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0))
        compute_ft_impl(str(f), settings=settings, persist=True)

        attrs = _read_ft_attrs(str(f))
        restored = FTSettings.from_attrs(attrs)

        assert restored.zpf == 2
        assert restored.expf_us == pytest.approx(5.0)
        assert restored.trim is not None
        assert restored.trim[0] == pytest.approx(26500.0)
        assert restored.trim[1] == pytest.approx(40000.0)


# ---------------------------------------------------------------------------
# Cross-interface consistency (gated on public-API refactor)
#
# These tests call Pipeline.compute_ft / api.compute_ft / CLI compute-ft with
# the new kwargs (zpf=, expf_us=, trim=).  They will fail with AttributeError
# or TypeError until Task #3 (API surface refactor) lands — that is expected.
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrossInterfaceD7:
    """
    Cross-interface D7 consistency: all three interfaces must produce
    identical ComplexFT and identical persisted ft_processing trim attrs.

    NOTE: These tests may fail with TypeError/AttributeError until the
    public-API refactor (Task #3) lands.  Failures here are expected-pending,
    not regressions in the foundation.
    """

    def _run_cli_import(self, ftmw_file: Path, source: str) -> None:
        r = _run_cli("import-data", str(ftmw_file), "--source", source)
        if r.returncode != 0:
            pytest.fail(f"import-data failed:\n{r.stderr}")

    def _run_cli_compute_ft(
        self,
        ftmw_file: Path,
        zpf: int,
        expf_us: float,
        trim_min: float,
        trim_max: float,
    ) -> None:
        r = _run_cli(
            "compute-ft",
            str(ftmw_file),
            "--zpf",
            str(zpf),
            "--expf_us",
            str(expf_us),
            "--trim",
            f"{trim_min}:{trim_max}",
        )
        if r.returncode != 0:
            pytest.fail(f"compute-ft failed:\n{r.stderr}")

    def test_all_three_interfaces_produce_identical_complex_ft(
        self, exp_2638_data_path, tmp_path
    ):
        """
        Pipeline.compute_ft, api.compute_ft, and CLI compute-ft must all
        produce a ComplexFT with identical n_points, freq range, and
        magnitude_spectrum, and the same persisted trim attrs.
        """
        import ftmwpipeline.api as ftmw
        from ftmwpipeline import Pipeline

        zpf, expf_us = 2, 5.0
        trim = (26500.0, 40000.0)

        # -- Pipeline interface --
        pipe_file = tmp_path / "pipe.ftmw"
        pipe = Pipeline.create(pipe_file, source=exp_2638_data_path)
        ft_pipe = pipe.compute_ft(zpf=zpf, expf_us=expf_us, trim=trim)
        assert isinstance(ft_pipe, ComplexFT)

        pipe_attrs = _read_ft_attrs(str(pipe_file))
        assert float(pipe_attrs["trim_min_mhz"]) == pytest.approx(26500.0)
        assert float(pipe_attrs["trim_max_mhz"]) == pytest.approx(40000.0)

        # -- Functional API --
        api_file = tmp_path / "api.ftmw"
        ftmw.import_data(api_file, source=exp_2638_data_path)
        ft_api = ftmw.compute_ft(api_file, zpf=zpf, expf_us=expf_us, trim=trim)
        assert isinstance(ft_api, ComplexFT)

        api_attrs = _read_ft_attrs(str(api_file))
        assert float(api_attrs["trim_min_mhz"]) == pytest.approx(26500.0)
        assert float(api_attrs["trim_max_mhz"]) == pytest.approx(40000.0)

        # -- CLI --
        cli_file = tmp_path / "cli.ftmw"
        self._run_cli_import(cli_file, exp_2638_data_path)
        self._run_cli_compute_ft(cli_file, zpf, expf_us, trim[0], trim[1])
        # Load back via impl (CLI does not return a Python object)
        cli_result = compute_ft_impl(str(cli_file))
        ft_cli: ComplexFT = cli_result["complex_ft"]

        cli_attrs = _read_ft_attrs(str(cli_file))
        assert float(cli_attrs["trim_min_mhz"]) == pytest.approx(26500.0)
        assert float(cli_attrs["trim_max_mhz"]) == pytest.approx(40000.0)

        # -- Numerical consistency across all three --
        assert (
            ft_pipe.n_points == ft_api.n_points == ft_cli.n_points
        ), "n_points mismatch across interfaces"
        np.testing.assert_allclose(
            ft_pipe.freq_array,
            ft_api.freq_array,
            rtol=1e-12,
            err_msg="Pipeline vs API: freq_array differs",
        )
        np.testing.assert_allclose(
            ft_pipe.freq_array,
            ft_cli.freq_array,
            rtol=1e-12,
            err_msg="Pipeline vs CLI: freq_array differs",
        )
        np.testing.assert_allclose(
            ft_pipe.magnitude_spectrum,
            ft_api.magnitude_spectrum,
            rtol=1e-10,
            err_msg="Pipeline vs API: magnitude_spectrum differs",
        )
        np.testing.assert_allclose(
            ft_pipe.magnitude_spectrum,
            ft_cli.magnitude_spectrum,
            rtol=1e-10,
            err_msg="Pipeline vs CLI: magnitude_spectrum differs",
        )
