"""Uniform ``--json`` on every CLI verb (CONTRACT_STRATEGY, Serialization).

Runs the real CLI in-process on a tiny synthetic native-HDF5 experiment and
checks the bytes a calling program would see: stdout is exactly one JSON
document, stage-running / curation verbs print ``ftmw/run_result@1``, other
verbs print their natural payload, errors go to stderr as ``ftmw/error@1``, and
the human output is unchanged without ``--json``.
"""

from __future__ import annotations

import argparse
import json
import shutil

import h5py
import numpy as np
import pytest

from ftmwpipeline.cli.main import create_parser, main
from ftmwpipeline.contract import MANIFEST

pytestmark = [pytest.mark.integration]

RUN_KEYS = {"schema", "verb", "stage", "invalidated", "summary"}


def _cli(capsys, *argv):
    capsys.readouterr()
    rc = main([str(a) for a in argv])
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _doc(out):
    """Parse stdout strictly: one document, no NaN/Infinity constants."""

    def refuse(const):
        raise AssertionError(f"non-finite constant {const} in --json output")

    return json.loads(out, parse_constant=refuse)


@pytest.fixture(scope="module")
def base_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli_json")
    src = tmp / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(1024) * 0.01))
    out = tmp / "base.ftmw"
    assert main(["data", "import", str(out), str(src), "--format", "ftmw-hdf5"]) == 0
    assert main(["ft", "run", str(out)]) == 0
    assert main(["noise", "run", str(out)]) == 0
    return out, src


@pytest.fixture
def exp(base_file, tmp_path):
    out = tmp_path / "exp.ftmw"
    shutil.copy(base_file[0], out)
    return out


# --- every leaf verb accepts --json ---------------------------------------


def _leaf_verbs(parser, path=()):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            seen = set()
            for name, sub in action.choices.items():
                if id(sub) not in seen:
                    seen.add(id(sub))
                    yield from _leaf_verbs(sub, path + (name,))
            return
    if parser.get_default("func") is not None:
        yield path, parser


def test_every_leaf_verb_accepts_json():
    # Mutation: drop install_json(parser) from create_parser -> verbs lack --json.
    verbs = list(_leaf_verbs(create_parser()))
    assert len(verbs) > 40
    missing = [
        " ".join(p) for p, sp in verbs if "--json" not in sp._option_string_actions
    ]
    assert not missing


# --- stage / curation verbs: run_result@1 ---------------------------------


def test_data_import_run_result(capsys, tmp_path, base_file):
    # Mutation: print the human text under --json -> stdout is not JSON.
    rc, out, _ = _cli(
        capsys,
        "data",
        "import",
        tmp_path / "n.ftmw",
        base_file[1],
        "--format",
        "ftmw-hdf5",
        "--json",
    )
    assert rc == 0
    doc = _doc(out)
    assert set(doc) == RUN_KEYS
    assert doc["schema"] == "ftmw/run_result@1"
    assert doc["verb"] == "data import" and doc["stage"] == "data"
    assert doc["invalidated"] == [] and doc["summary"]["n_points"] == 1024


@pytest.mark.parametrize(
    "argv, verb, stage, summary_key",
    [
        (("ft", "run"), "ft run", "ft", "fid_points"),
        (("noise", "run"), "noise run", "noise", "noise_fraction"),
        (("clocks", "add", "5760:locked:s"), "clocks add", None, "n_clock_sources"),
        (("settings", "set", "stage1.trim", "100,900"), "settings set", None, "path"),
    ],
)
def test_stage_verbs_print_run_result(capsys, exp, argv, verb, stage, summary_key):
    # Mutation: a verb that skips record_run_result prints no envelope / wrong keys.
    rc, out, _ = _cli(capsys, argv[0], argv[1], exp, *argv[2:], "--json")
    assert rc == 0
    doc = _doc(out)
    assert set(doc) == RUN_KEYS
    assert doc["schema"] == "ftmw/run_result@1"
    assert (doc["verb"], doc["stage"]) == (verb, stage)
    assert isinstance(doc["invalidated"], list)
    assert summary_key in doc["summary"]
    assert not any(isinstance(v, (list, tuple)) for v in doc["summary"].values())


@pytest.mark.parametrize(
    "flags, echoed",
    [
        (("--n-iter", "2"), "n_iter: 2"),
        (("--line-k", "6"), "line_k: 6.0"),
        (("--no-region-aware",), "region_aware: False"),
    ],
)
def test_noise_run_knob_flags_echo_and_run(capsys, exp, flags, echoed):
    # Regression: any noise knob flag crashed echoing the explicit settings
    # ('NoiseSettings' object has no attribute 'to_yaml_dict').
    rc, out, err = _cli(capsys, "noise", "run", exp, *flags)
    assert rc == 0, err
    assert echoed in out


def test_invalidated_comes_from_the_result(capsys, exp):
    # Mutation: hard-code invalidated=[] -> settings set misses "noise".
    rc, out, _ = _cli(
        capsys, "settings", "set", exp, "stage1.trim", "100,900", "--json"
    )
    assert rc == 0
    assert _doc(out)["invalidated"] == ["noise"]


def test_settings_unset_and_clocks_clear(capsys, exp):
    for argv in (
        ("clocks", "add", exp, "5760"),
        ("clocks", "clear", exp, "--json"),
        ("settings", "unset", exp, "stage1.trim", "--json"),
    ):
        rc, out, _ = _cli(capsys, *argv)
        assert rc == 0
        if "--json" in argv:
            assert _doc(out)["schema"] == "ftmw/run_result@1"


# --- other verbs: natural payload / paths ---------------------------------


@pytest.mark.parametrize(
    "argv, key",
    [
        (("formats",), "formats"),
        (("scan", "list"), "knobs"),
        (("clocks", "show", "{f}"), "clocks"),
        (("settings", "show", "{f}"), "settings"),
        (("info", "{f}"), "completed_stages"),
        (("version",), "version"),
    ],
)
def test_natural_payload_verbs(capsys, exp, argv, key):
    # Mutation: leave stdout captured without printing the payload -> no JSON.
    rc, out, _ = _cli(capsys, *[str(exp) if a == "{f}" else a for a in argv], "--json")
    assert rc == 0
    assert key in _doc(out)


def test_plot_verb_prints_paths(capsys, exp, tmp_path):
    # Mutation: drop the savefig tracker -> paths empty.
    png = tmp_path / "ft.png"
    rc, out, _ = _cli(capsys, "ft", "show", exp, "--output", png, "--json")
    assert rc == 0
    assert _doc(out) == {"paths": [str(png)]}
    assert png.exists()


def test_format_json_synonym_matches_json(capsys, exp):
    # Mutation: --json stops mapping to format json on info.
    _, a, _ = _cli(capsys, "info", exp, "--json")
    _, b, _ = _cli(capsys, "info", exp, "--format", "json")
    da, db = _doc(a), _doc(b)
    assert da.keys() == db.keys()


# --- errors ---------------------------------------------------------------


def test_error_under_json_is_error_dict_on_stderr(capsys, tmp_path):
    # Mutation: main() ignores --json -> text error, or dict lands on stdout.
    rc, out, err = _cli(capsys, "ft", "run", tmp_path / "missing.ftmw", "--json")
    assert rc != 0
    assert out == ""
    line = next(x for x in err.splitlines() if x.startswith("{"))
    doc = _doc(line)
    assert doc["schema"] == "ftmw/error@1" and doc["code"] == "not_found"


# --- human output unchanged -----------------------------------------------


def test_human_output_unchanged_without_json(capsys, exp):
    # Mutation: wrapper activates without --json -> human text becomes JSON.
    rc, out, _ = _cli(capsys, "clocks", "add", exp, "5760:locked:s")
    assert rc == 0
    assert out == "Clock sources now (1):\n   5760 MHz  locked  [s]\n"
    rc, out, _ = _cli(capsys, "clocks", "show", exp)
    assert out.startswith("Declared clock sources for ")
    rc, out, _ = _cli(capsys, "settings", "set", exp, "stage1.trim", "100,900")
    assert out == (
        "Set stage1.trim = 100, 900 (persisted to .ftmw).\n"
        "Invalidated downstream stage(s): noise -- re-run them to refresh.\n"
    )
    rc, out, _ = _cli(capsys, "version")
    assert out.startswith("ftmwpipeline ")


def test_run_result_schema_declared():
    assert "ftmw/run_result@1" in MANIFEST.schemas


@pytest.mark.slow
def test_json_does_not_change_the_format_of_a_written_file(
    stage5_reviewed_2638, tmp_path, capsys
):
    # Mutation: --json forcing --format=json on a verb whose --format is the
    # format of the file it writes (report run's table became e_lines.json).
    fp = tmp_path / "e.ftmw"
    shutil.copy(stage5_reviewed_2638, fp)
    out_dir = tmp_path / "rep"
    rc, out, _ = _cli(
        capsys, "report", "run", fp, "--output-dir", out_dir, "--level1-only", "--json"
    )
    assert rc == 0
    doc = _doc(out)
    assert doc["schema"] == "ftmw/run_result@1"
    assert str(doc["summary"]["table"]).endswith(".csv")
    assert list(out_dir.glob("*.csv")) and not list(out_dir.glob("*.json"))


# ---- which --format json makes errors JSON ---------------------------------


def _last_error_dict(err):
    lines = [ln for ln in err.splitlines() if ln.strip()]
    return json.loads(lines[-1])


@pytest.mark.parametrize(
    "argv",
    [
        ("info", "{f}", "--format", "json"),
        ("read", "meta", "{f}", "--format", "json"),
        ("read", "table", "{f}", "fit_peaks", "--format", "json"),
        ("report", "table", "{f}", "--format", "json"),
        ("report", "run", "{f}", "--json"),
    ],
    ids=lambda a: " ".join(a[:2]),
)
def test_json_error_under_json_and_printing_format_synonyms(tmp_path, capsys, argv):
    missing = str(tmp_path / "absent.ftmw")
    rc, _, err = _cli(capsys, *(a.format(f=missing) for a in argv))
    assert rc == 1
    payload = _last_error_dict(err)
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "not_found"


@pytest.mark.parametrize(
    "argv",
    [
        # --format names the table file report run writes, not the output mode.
        ("report", "run", "{f}", "--format", "json"),
        # --format names the format of the --output file.
        ("read", "table", "{f}", "fit_peaks", "--format", "json", "--output", "{o}"),
        ("report", "table", "{f}", "--format", "json", "--output", "{o}"),
    ],
    ids=lambda a: " ".join(a[:2]) + (" --output" if "--output" in a else ""),
)
def test_a_file_format_json_does_not_make_errors_json(tmp_path, capsys, argv):
    missing = str(tmp_path / "absent.ftmw")
    out = str(tmp_path / "out.json")
    rc, stdout, err = _cli(capsys, *(a.format(f=missing, o=out) for a in argv))
    assert rc == 1
    assert "ftmw/error@1" not in stdout + err
    assert "Pipeline file not found" in stdout + err
