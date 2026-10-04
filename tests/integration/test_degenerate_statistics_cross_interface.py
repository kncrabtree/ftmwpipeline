"""Degenerate statistics read ``UNDEFINED`` identically on API, Pipeline and CLI.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
migration rule "Degenerate statistics are ``UNDEFINED``"
(``dev-docs/CONTRACT_STRATEGY.md`` §Missing values). The rule lives once in
``_internal/read_impl.py``; the three surfaces must return the same columns.

The fixture is a private copy of the small Stage 5 baseline (real 2638 data)
into which an earlier release's encoding of each degenerate statistic is
injected with h5py, next to a genuine value of the same kind that must keep
reading as present. Everything is written under pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline
from ftmwpipeline.cli.main import main

pytestmark = [pytest.mark.integration]

NR = Absent.NOT_RUN.status
UD = Absent.UNDEFINED.status

COLUMNS = {
    "peaks": ["snr", "snr__status", "internal_snr", "internal_snr__status"],
    "fit_windows": [
        "window_id",
        "edge_coherence_low",
        "edge_coherence_low__status",
        "edge_coherence_high",
        "edge_coherence_high__status",
    ],
    "fit_audit": [
        "window_id",
        "step_index",
        "decision",
        "f_statistic",
        "f_statistic__status",
        "p_value",
        "p_value__status",
    ],
}


def _inject(path):
    """Earlier-release encodings of the degenerate statistics; returns the row
    handles the assertions use."""
    handles = {}
    with h5py.File(path, "r+") as h5f:
        # --- peaks: zero local noise (SNR 0.0), internal-pass SNR 0.0 / nan.
        p = h5f["stage3_peaks"]
        n = p["snr"].shape[0]
        assert n >= 5
        noise, snr = p["noise_std_local"][:], p["snr"][:]
        isnr, ifreq = p["internal_snr"][:], p["internal_frequency"][:]
        noise[0], snr[0] = 0.0, 0.0
        ifreq[1:4] = 100.0 + np.arange(3)
        isnr[1], isnr[2], isnr[3] = 0.0, np.nan, 12.0
        noise[4], snr[4] = 1.0, 0.0  # a clamped SNR on positive noise: measured
        p["noise_std_local"][:] = noise
        p["snr"][:] = snr
        p["internal_snr"][:] = isnr
        p["internal_frequency"][:] = ifreq

        # --- fit_windows / fit_audit on the lowest window id.
        w = h5f["stage5_fitting/windows"]
        ids = w["window_id"][:]
        row = int(np.argmin(ids))
        handles["window_id"] = int(ids[row])
        low = w["edge_coherence_low"][:]
        low[row] = 0.0
        w["edge_coherence_low"][:] = low
        quality = json.loads(w["quality_metrics"][row])
        quality["edge_coherence_low"] = 0.0
        w["quality_metrics"][row] = json.dumps(quality)

        trail = json.loads(w["audit_trail"][row])
        seed = trail[0]
        degenerate = dict(
            seed,
            decision="accept",
            chi2_after=seed["chi2_before"] - 1.0,
            f_statistic=0.0,
            p_value=1.0,
            n_eff=3.0,
            aicc_delta=-1.0,
            separation_ok=True,
        )
        genuine = dict(
            degenerate,
            decision="reject",
            chi2_after=seed["chi2_before"] + 5.0,
            aicc_delta=1.0,
        )
        trail += [degenerate, genuine]
        handles["audit_degenerate"] = len(trail) - 2
        handles["audit_genuine"] = len(trail) - 1
        w["audit_trail"][row] = json.dumps(trail)
    return handles


@pytest.fixture(scope="module")
def injected(baseline_2638_stage5_small, tmp_path_factory):
    path = tmp_path_factory.mktemp("degenerate_xi") / "injected.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    return str(path), _inject(path)


def _via_api(path, table):
    return ftmw.read_table(path, table, COLUMNS[table])


def _via_pipeline(path, table):
    return Pipeline.open(path).read_table(table, COLUMNS[table])


def _via_cli(path, table, out, capsys):
    rc = main(
        [
            "read",
            "read_table",
            path,
            table,
            "--columns",
            ",".join(COLUMNS[table]),
            "--output",
            str(out),
            "--format",
            "json",
        ]
    )
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    env = json.loads(cap.out)
    return {col: np.load(out / env[col], allow_pickle=False) for col in COLUMNS[table]}


@pytest.mark.parametrize("table", sorted(COLUMNS))
def test_the_three_surfaces_agree(injected, table, tmp_path, capsys):
    path, _ = injected
    api = _via_api(path, table)
    pipe = _via_pipeline(path, table)
    cli = _via_cli(path, table, tmp_path / "cli", capsys)
    assert set(api) == set(pipe) == set(cli) == set(COLUMNS[table])
    for col in COLUMNS[table]:
        np.testing.assert_array_equal(api[col], pipe[col], err_msg=col)
        np.testing.assert_array_equal(api[col], cli[col], err_msg=col)


def test_peaks_snr_is_undefined_where_noise_is_zero(injected):
    path, _ = injected
    t = ftmw.read_table(path, "peaks", COLUMNS["peaks"])
    assert t["snr__status"][0] == UD and np.isnan(t["snr"][0])
    assert t["snr__status"][4] == 0 and t["snr"][4] == 0.0  # measured, kept
    assert list(t["internal_snr__status"][1:4]) == [UD, UD, 0]
    assert np.isnan(t["internal_snr"][[1, 2]]).all() and t["internal_snr"][3] == 12.0


def test_fit_windows_old_zero_edge_is_undefined(injected):
    path, handles = injected
    t = ftmw.read_table(path, "fit_windows", COLUMNS["fit_windows"])
    row = int(np.flatnonzero(t["window_id"] == handles["window_id"])[0])
    assert t["edge_coherence_low__status"][row] == UD
    assert np.isnan(t["edge_coherence_low"][row])


def test_fit_audit_degenerate_f_test_is_undefined_and_a_non_improvement_is_not(
    injected,
):
    path, handles = injected
    t = ftmw.read_table(path, "fit_audit", COLUMNS["fit_audit"])
    rows = np.flatnonzero(t["window_id"] == handles["window_id"])
    bad = rows[handles["audit_degenerate"]]
    good = rows[handles["audit_genuine"]]
    for col in ("f_statistic", "p_value"):
        assert t[col + "__status"][bad] == UD and np.isnan(t[col][bad])
        assert t[col + "__status"][good] == 0
    assert (t["f_statistic"][good], t["p_value"][good]) == (0.0, 1.0)


def test_the_reads_leave_the_file_unchanged(injected):
    path, _ = injected
    before = open(path, "rb").read()
    for table in COLUMNS:
        _via_api(path, table)
        _via_pipeline(path, table)
    assert open(path, "rb").read() == before
