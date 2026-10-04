"""Stage 6 refits replay the Stage 5 spur catalog at full precision.

``SpectrumFit.parameters["spur_centers_mhz"]`` is rounded to 4 decimals for
display; the fit itself, and the window and spectrum models, mask at the full
precision ``diagnostics["gated_spurs"]`` carries. The Stage 6 shared fit context
used to replay the rounded centers, so a refit of a window could mask a
different bin than the fit it edits (a spur's mask is a whole-bin half-width, so
only a mask edge within the rounding of a bin moves). It now goes through
``gated_spur_catalog`` like the models do.

The 2638 build and the fit are read-only here: the catalog attributes are
rewritten on a private copy of the post-fit fixture, and the shared context is
built from it with ``replay_spur_set`` spied on.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal import stage5_impl, stage6_impl

pytestmark = [pytest.mark.unit]

FULL = [28123.45678912, 33000.78912345]
ROUNDED = [round(c, 4) for c in FULL]


def _inject_catalog(path, *, gated: bool):
    with h5py.File(path, "r+") as h5f:
        g = h5f["stage5_fitting"]
        params = json.loads(g.attrs["parameters"])
        diag = json.loads(g.attrs["diagnostics"])
        params["spur_masking_enabled"] = True
        params["spur_centers_mhz"] = list(ROUNDED)
        params["spur_sources"] = ["narrow", "saturated"]
        params["spur_lattice"] = [None, None]
        params["spur_drift"] = [False, False]
        params["spur_mask_half_width_bins"] = 2
        params["spur_mask_half_width_bins_per_spur"] = [None, None]
        diag.pop("gated_spurs", None)
        if gated:
            diag["gated_spurs"] = [
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
                    "lattice": None,
                    "drift": False,
                    "mask_half_width_bins": 5,
                },
            ]
        g.attrs["parameters"] = json.dumps(params)
        g.attrs["diagnostics"] = json.dumps(diag)


@pytest.fixture
def replay_spy(monkeypatch):
    calls = []
    real = stage5_impl.replay_spur_set

    def spy(catalog, sorted_freq):
        spur_set = real(catalog, sorted_freq)
        calls.append((dict(catalog), spur_set))
        return spur_set

    monkeypatch.setattr(stage5_impl, "replay_spur_set", spy)
    return calls


def test_shared_context_replays_full_precision_centers(stage5_small_file, replay_spy):
    _inject_catalog(stage5_small_file, gated=True)
    shared = stage6_impl._build_shared_fit_ctx(str(stage5_small_file))

    assert len(replay_spy) == 1
    catalog, _ = replay_spy[0]
    assert catalog["spur_centers_mhz"] == FULL  # not the rounded display copy
    assert catalog["spur_mask_half_width_bins_per_spur"] == [None, 5]
    centers = [s.center_mhz for s in shared.fit_ctx.spur_set.spurs]
    assert centers == FULL and centers != ROUNDED
    assert [s.mask_half_width_bins for s in shared.fit_ctx.spur_set.spurs] == [None, 5]


def test_shared_context_without_a_gated_record_keeps_the_stored_lists(
    stage5_small_file, replay_spy
):
    """A fit that predates ``diagnostics["gated_spurs"]`` replays as stored."""
    _inject_catalog(stage5_small_file, gated=False)
    shared = stage6_impl._build_shared_fit_ctx(str(stage5_small_file))

    assert len(replay_spy) == 1
    assert replay_spy[0][0]["spur_centers_mhz"] == ROUNDED
    centers = [s.center_mhz for s in shared.fit_ctx.spur_set.spurs]
    assert centers == ROUNDED


def test_shared_context_replays_what_the_model_replays(stage5_small_file, replay_spy):
    """The Stage 6 replay and ``model_impl._fit_spur_set`` mask the same bins."""
    from ftmwpipeline._internal.model_impl import _fit_spur_set
    from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

    _inject_catalog(stage5_small_file, gated=True)
    shared = stage6_impl._build_shared_fit_ctx(str(stage5_small_file))
    with h5py.File(stage5_small_file, "r") as h5f:
        fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    grid = np.sort(np.asarray(shared.fit_ctx.active_ft.freq_mhz, dtype=float))
    model_set = _fit_spur_set(fit, grid)
    stage6_set = shared.fit_ctx.spur_set

    def view(spur_set):
        return [
            (s.center_mhz, s.source, s.drift, s.mask_half_width_bins)
            for s in spur_set.spurs
        ]

    assert view(model_set) == view(stage6_set)
    assert model_set.bin_spacing_mhz == stage6_set.bin_spacing_mhz


def test_the_shared_context_read_does_not_write(stage5_small_file, replay_spy):
    _inject_catalog(stage5_small_file, gated=True)
    before = stage5_small_file.read_bytes()
    stage6_impl._build_shared_fit_ctx(str(stage5_small_file))
    assert stage5_small_file.read_bytes() == before
