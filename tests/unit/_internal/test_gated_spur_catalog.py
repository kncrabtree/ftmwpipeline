"""The gated spur catalog a Stage 5 fit masked with is replayed at full precision.

``SpectrumFit.parameters["spur_centers_mhz"]`` rounds the centers to 4 decimals
for display; ``diagnostics["gated_spurs"]`` carries them, and the per-spur mask
overrides, at full precision. Every reader that rebuilds the fit's mask goes
through :func:`~ftmwpipeline._internal.stage5_impl.gated_spur_catalog`, so the
Stage 6 refits and the window/spectrum models mask exactly the bins the fit did.
"""

from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.model_impl import _fit_spur_set
from ftmwpipeline._internal.stage5_impl import gated_spur_catalog, replay_spur_set
from ftmwpipeline.core.data_structures import SpectrumFit
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    read_fit_diagnostics,
    save_spectrum_fit_to_hdf5,
)

FULL = [28123.45678912, 33000.78912345, 35000.00004321]
ROUNDED = [round(c, 4) for c in FULL]
GRID = np.linspace(28000.0, 36000.0, 80001)  # 0.1 MHz bins


def _parameters(**over):
    p = {
        "spur_masking_enabled": True,
        "spur_centers_mhz": list(ROUNDED),
        "spur_sources": ["narrow", "saturated", "drift"],
        "spur_lattice": [None, "320x6 (bb)", None],
        "spur_drift": [False, False, True],
        "spur_mask_half_width_bins": 2,
        "spur_mask_half_width_bins_per_spur": [None, None, None],
        "acquisition_us": 12.65,
    }
    p.update(over)
    return p


def _gated():
    return [
        {
            "center_mhz": FULL[0],
            "source": "narrow",
            "lattice": None,
            "drift": False,
            "mask_half_width_bins": None,
        },
        {
            "center_mhz": FULL[1],
            "source": "saturated",
            "lattice": "320x6 (bb)",
            "drift": False,
            "mask_half_width_bins": 7,
        },
        {
            "center_mhz": FULL[2],
            "source": "drift",
            "lattice": None,
            "drift": True,
            "mask_half_width_bins": None,
        },
    ]


class TestGatedSpurCatalog:
    def test_full_precision_centers_replace_the_rounded_ones(self):
        cat = gated_spur_catalog(_parameters(), {"gated_spurs": _gated()})
        assert cat["spur_centers_mhz"] == FULL
        assert cat["spur_centers_mhz"] != ROUNDED

    def test_every_per_spur_list_is_overlaid_in_order(self):
        cat = gated_spur_catalog(_parameters(), {"gated_spurs": _gated()})
        assert cat["spur_sources"] == ["narrow", "saturated", "drift"]
        assert cat["spur_lattice"] == [None, "320x6 (bb)", None]
        assert cat["spur_drift"] == [False, False, True]
        # The per-spur mask override lives only in the diagnostics record.
        assert cat["spur_mask_half_width_bins_per_spur"] == [None, 7, None]

    def test_other_parameters_are_kept_and_the_input_is_not_mutated(self):
        params = _parameters()
        before = copy.deepcopy(params)
        cat = gated_spur_catalog(params, {"gated_spurs": _gated()})
        assert cat["spur_mask_half_width_bins"] == 2
        assert cat["acquisition_us"] == 12.65
        assert cat["spur_masking_enabled"] is True
        assert params == before
        assert cat is not params

    @pytest.mark.parametrize(
        "diagnostics", [None, {}, {"gated_spurs": []}, {"gated_spurs": None}]
    )
    def test_no_gated_record_keeps_the_stored_lists(self, diagnostics):
        """A fit that gated nothing, or predates the record, replays as stored."""
        params = _parameters()
        cat = gated_spur_catalog(params, diagnostics)
        assert cat == params
        assert cat is not params

    def test_a_gated_entry_missing_optional_fields_takes_the_defaults(self):
        cat = gated_spur_catalog(_parameters(), {"gated_spurs": [{"center_mhz": 1.5}]})
        assert cat["spur_centers_mhz"] == [1.5]
        assert cat["spur_sources"] == ["narrow"]
        assert cat["spur_lattice"] == [None]
        assert cat["spur_drift"] == [False]
        assert cat["spur_mask_half_width_bins_per_spur"] == [None]

    def test_the_replayed_set_masks_at_full_precision(self):
        cat = gated_spur_catalog(_parameters(), {"gated_spurs": _gated()})
        spur_set = replay_spur_set(cat, GRID)
        assert [s.center_mhz for s in spur_set.spurs] == FULL
        assert [s.mask_half_width_bins for s in spur_set.spurs] == [None, 7, None]
        assert spur_set.bin_spacing_mhz == pytest.approx(0.1)
        # Replaying the rounded lists (the old Stage 6 behaviour) differs.
        rounded = replay_spur_set(_parameters(), GRID)
        assert [s.center_mhz for s in rounded.spurs] == ROUNDED


def _mask_view(spur_set):
    """What drives the mask (``snr`` is NaN on a replayed spur, never compared)."""
    return (
        [
            (s.center_mhz, s.source, s.lattice, s.drift, s.mask_half_width_bins)
            for s in spur_set.spurs
        ],
        spur_set.bin_spacing_mhz,
        spur_set.mask_half_width_bins,
    )


class TestModelSpurSet:
    """``window_model`` / ``spectrum_model`` replay the same catalog."""

    def _fit(self, **over):
        return SpectrumFit(
            window_fits=[],
            fitted_peaks=[],
            parameters=_parameters(**over),
            diagnostics={"gated_spurs": _gated()},
        )

    def test_model_replays_the_full_precision_catalog(self):
        fit = self._fit()
        spur_set = _fit_spur_set(fit, GRID)
        want = replay_spur_set(
            gated_spur_catalog(fit.parameters, fit.diagnostics), GRID
        )
        assert _mask_view(spur_set) == _mask_view(want)
        assert [s.center_mhz for s in spur_set.spurs] == FULL

    def test_model_and_stage6_replay_agree(self):
        """One catalog function feeds both readers, so their sets are equal."""
        fit = self._fit()
        stage6_catalog = gated_spur_catalog(fit.parameters, fit.diagnostics)
        assert _mask_view(_fit_spur_set(fit, GRID)) == _mask_view(
            replay_spur_set(stage6_catalog, GRID)
        )

    def test_masking_disabled_is_no_spur_set(self):
        assert _fit_spur_set(self._fit(spur_masking_enabled=False), GRID) is None

    def test_nothing_gated_is_no_spur_set(self):
        fit = SpectrumFit(
            window_fits=[],
            fitted_peaks=[],
            parameters=_parameters(
                spur_centers_mhz=[],
                spur_sources=[],
                spur_lattice=[],
                spur_drift=[],
                spur_mask_half_width_bins_per_spur=[],
            ),
            diagnostics={},
        )
        assert _fit_spur_set(fit, GRID) is None


class TestReadFitDiagnostics:
    def _write(self, path, diagnostics):
        fit = SpectrumFit(
            window_fits=[],
            fitted_peaks=[],
            parameters=_parameters(),
            diagnostics=diagnostics,
        )
        with h5py.File(path, "w") as h5f:
            save_spectrum_fit_to_hdf5(fit, h5f.create_group("stage5_fitting"))
        return path

    def test_equals_the_full_loader(self, tmp_path):
        diag = {"gated_spurs": _gated(), "n_nonconverged": 2}
        f = self._write(tmp_path / "d.ftmw", diag)
        with h5py.File(f, "r") as h5f:
            cheap = read_fit_diagnostics(h5f["stage5_fitting"])
            full = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"]).diagnostics
        assert cheap == full == diag

    def test_absent_attribute_is_an_empty_dict(self, tmp_path):
        f = self._write(tmp_path / "e.ftmw", {"x": 1})
        with h5py.File(f, "r+") as h5f:
            del h5f["stage5_fitting"].attrs["diagnostics"]
            assert read_fit_diagnostics(h5f["stage5_fitting"]) == {}

    def test_a_read_leaves_the_file_unchanged(self, tmp_path):
        f = self._write(tmp_path / "r.ftmw", {"gated_spurs": _gated()})
        before = f.read_bytes()
        with h5py.File(f, "r") as h5f:
            read_fit_diagnostics(h5f["stage5_fitting"])
        assert f.read_bytes() == before
