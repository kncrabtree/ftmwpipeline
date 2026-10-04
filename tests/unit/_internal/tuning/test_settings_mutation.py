"""Unit tests for the settings change-grammar core (issue #28, step 4).

``set_setting`` coercion + persistence + stage invalidation, ``unset_setting``
clearing a field back to the resolver's layers, and ``export_settings``
round-tripping through the stages' ``load_preset`` path -- all against bare
HDF5 ``.ftmw`` files with stage groups stamped directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

import ftmwpipeline._internal.tuning.settings_mutation as mutation
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.tuning import (
    export_settings,
    set_setting,
    unset_setting,
)
from ftmwpipeline._internal.tuning.settings_inspection import (
    SOURCE_DEFAULT,
    SOURCE_FTMW,
    resolve_settings_view,
)
from ftmwpipeline.core import noise_settings as noise_mod
from ftmwpipeline.core import peak_shape as ps_mod
from ftmwpipeline.core import stage_fit_settings as fit_mod
from ftmwpipeline.core.knob_metadata import FieldTyping, make_bounds
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.io.noise_settings_serialization import save_noise_settings_to_h5
from ftmwpipeline.io.stage_fit_settings_serialization import (
    save_stage_fit_settings_to_h5,
)


@pytest.fixture
def bare_ftmw(tmp_path: Path) -> Path:
    p = tmp_path / "bare.ftmw"
    with h5py.File(p, "w"):
        pass
    return p


def _row(file: Path, path: str):
    rows = {r.path: r for r in resolve_settings_view(file, include_advanced=True)}
    return rows[path]


def _stamp_stages(file: Path, completed: list[str]) -> None:
    """Stamp pipeline_stages + a placeholder data group per completed stage."""
    from ftmwpipeline.file_manager import PipelineStageTracker

    paths = PipelineStageTracker.STAGE_DATA_PATHS
    with h5py.File(file, "a") as h5f:
        grp = h5f.require_group("pipeline_stages")
        grp.attrs["completed_stages"] = json.dumps(completed)
        for stage in completed:
            data_path = paths.get(stage, stage)
            if data_path not in h5f:
                h5f.require_group(data_path)


# --- set_setting: coercion + persistence -----------------------------------
def test_set_float_persists_and_reads_back(bare_ftmw: Path) -> None:
    result = set_setting(bare_ftmw, "stage2.window_mhz", "123.5")
    assert result.value == 123.5
    row = _row(bare_ftmw, "stage2.window_mhz")
    assert row.source == SOURCE_FTMW
    assert row.value == 123.5


def test_set_int_and_bool_coercion(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage2.n_iter", "5")
    set_setting(bare_ftmw, "stage2.region_aware", "false")
    assert _row(bare_ftmw, "stage2.n_iter").value == 5
    # HDF5 reads booleans back as numpy bool; compare by value, not identity.
    assert bool(_row(bare_ftmw, "stage2.region_aware").value) is False


def test_set_subblock_field(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage5.tau.max_decay_factor", "4.0")
    assert _row(bare_ftmw, "stage5.tau.max_decay_factor").value == 4.0


def test_set_shape(bare_ftmw: Path) -> None:
    result = set_setting(bare_ftmw, "stage5.shape", "gaussian")
    assert result.value.kind is ps_mod.PeakShape.GAUSSIAN
    assert _row(bare_ftmw, "stage5.shape").value.kind is ps_mod.PeakShape.GAUSSIAN


def test_set_unknown_knob_raises(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        set_setting(bare_ftmw, "stage2.not_a_field", "1")


def test_set_unknown_stage_raises(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="unknown settings stage"):
        set_setting(bare_ftmw, "stage9.foo", "1")


# --- set_setting: Stage 1 rules --------------------------------------------
@pytest.mark.parametrize("field", ["zpf", "expf_us", "window_function"])
def test_set_stage1_retired_apodization_knobs_rejected(
    bare_ftmw: Path, field: str
) -> None:
    """The retired apodization knobs no longer exist on FTSettings, so setting
    them is rejected as an unknown field."""
    with pytest.raises(ValueError, match="unknown setting"):
        set_setting(bare_ftmw, f"stage1.{field}", "2")


def test_set_stage1_start_us_persists(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage1.start_us", "3.25")
    row = _row(bare_ftmw, "stage1.start_us")
    assert row.source == SOURCE_FTMW
    assert row.value == 3.25


# --- set_setting: invalidation ---------------------------------------------
def test_set_invalidates_stage_and_downstream(bare_ftmw: Path) -> None:
    _stamp_stages(
        bare_ftmw,
        [
            "stage0_fid_data",
            "stage1_complex_ft",
            "stage2_noise_result",
            "stage3_peaks",
        ],
    )
    result = set_setting(bare_ftmw, "stage2.window_mhz", "90")
    # Stage 2 itself and its dependent Stage 3 are invalidated, reported as
    # canonical stage names in rerun order.
    assert result.invalidated == ("noise", "peaks")
    with h5py.File(bare_ftmw, "r") as h5f:
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage2_noise_result" not in completed
        assert "stage3_peaks" not in completed
        assert "stage1_complex_ft" in completed  # upstream untouched
        assert "stage2_noise_result" not in h5f  # results dropped
        assert "stage3_peaks" not in h5f


def test_set_on_unrun_stage_invalidates_nothing(bare_ftmw: Path) -> None:
    _stamp_stages(bare_ftmw, ["stage0_fid_data", "stage1_complex_ft"])
    result = set_setting(bare_ftmw, "stage5.tau.max_decay_factor", "4.0")
    assert result.invalidated == ()


# --- set_setting: the value encoding ----------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "100.0,50.0,20.0",  # comma-joined scalars, the CLI's shorthand
        "[100.0, 50.0, 20.0]",  # a JSON array, the natural machine rendering
        " [100.0,50.0,20.0] ",
        [100.0, 50.0, 20.0],  # native list
        (100.0, 50.0, 20.0),  # native tuple
    ],
)
def test_tuple_field_accepts_every_documented_encoding(bare_ftmw: Path, raw) -> None:
    """A JSON array must coerce like the comma form, not corrupt the tuple.

    The old splitter split on ',' alone, so a bracketed rendering persisted
    ``('[100.0', 50.0, '20.0]')`` -- strings inside a Tuple[float, ...] field,
    silently, with the next run fitting against them.
    """
    result = set_setting(bare_ftmw, "stage2b.gaussian.tau_G_seeds", raw)
    assert result.value == (100.0, 50.0, 20.0)
    assert all(isinstance(v, float) for v in result.value)
    assert tuple(_row(bare_ftmw, "stage2b.gaussian.tau_G_seeds").value) == (
        100.0,
        50.0,
        20.0,
    )


def test_fixed_arity_tuple_checks_its_length(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage1.trim", "[26500.0, 40000.0]")
    assert tuple(_row(bare_ftmw, "stage1.trim").value) == (26500.0, 40000.0)
    with pytest.raises(ValueError, match="expected 2 value"):
        set_setting(bare_ftmw, "stage1.trim", "[26500.0, 30000.0, 40000.0]")


def test_tuple_elements_take_the_declared_element_type(bare_ftmw: Path) -> None:
    """A Tuple[str, ...] keeps numeric-looking labels as text."""
    result = set_setting(bare_ftmw, "stage2b.band.band_labels", "K,Ka,18")
    assert result.value == ("K", "Ka", "18")


@pytest.mark.parametrize(
    "knob,raw,message",
    [
        ("stage2b.gaussian.tau_G_seeds", "None", "cannot parse a number"),
        ("stage2b.gaussian.tau_G_seeds", "[1.0, oops]", "cannot parse a number"),
        ("stage2.window_mhz", "wide", "cannot parse a number"),
        ("stage2.n_iter", "3.5", "cannot parse an integer"),
        ("stage2.region_aware", "maybe", "cannot parse boolean"),
    ],
)
def test_unparsable_value_raises_and_persists_nothing(
    bare_ftmw: Path, knob: str, raw: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        set_setting(bare_ftmw, knob, raw)
    assert _row(bare_ftmw, knob).source == SOURCE_DEFAULT


def test_native_values_are_accepted_directly(bare_ftmw: Path) -> None:
    assert set_setting(bare_ftmw, "stage2.window_mhz", 123.5).value == 123.5
    assert set_setting(bare_ftmw, "stage2.region_aware", True).value is True
    # JSON has one number type, so an integral float is how it spells an int.
    assert set_setting(bare_ftmw, "stage2.n_iter", 5.0).value == 5
    assert set_setting(bare_ftmw, "stage5.shape", "gaussian").value.kind == (
        ps_mod.PeakShape.GAUSSIAN
    )


def test_string_none_is_only_ever_text(bare_ftmw: Path) -> None:
    """No string encodes the unset state; "None" is four characters of text.

    Valid on a str field that states no choices, a coercion error on a numeric
    field, and a refused choice on a field that declares them (``"None"`` is not
    one of ``n_eff_kind``'s choices) -- the unset request is the native ``None``
    (equivalently ``unset_setting``).
    """
    assert set_setting(
        bare_ftmw, "stage3.primary_pass.primary_window", "None"
    ).value == ("None")
    with pytest.raises(ValueError):
        set_setting(bare_ftmw, "stage2.window_mhz", "None")
    with pytest.raises(BadSettingError) as excinfo:
        set_setting(bare_ftmw, "stage5.conservative.n_eff_kind", "None")
    assert excinfo.value.value == "None"
    assert set_setting(bare_ftmw, "stage5.conservative.n_eff_kind", None).value is None


def test_non_string_on_a_str_field_is_refused(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="expected a string"):
        set_setting(bare_ftmw, "stage5.conservative.n_eff_kind", 3)


# --- set_setting: declared choices and bounds -------------------------------
_N_EFF = "stage5.conservative.n_eff_kind"
_N_EFF_CHOICES = ("perplexity_log1p_snr", "kish_mag_sq", "kish_mag", "hard_radius")


def _md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("choice", _N_EFF_CHOICES)
def test_every_declared_choice_is_accepted(bare_ftmw: Path, choice: str) -> None:
    assert set_setting(bare_ftmw, _N_EFF, choice).value == choice
    assert _row(bare_ftmw, _N_EFF).value == choice


def test_a_value_outside_the_declared_choices_is_bad_setting(
    bare_ftmw: Path,
) -> None:
    """Mutation: drop the choices check in ``_coerce_or_unset``."""
    before = _md5(bare_ftmw)
    with pytest.raises(BadSettingError) as excinfo:
        set_setting(bare_ftmw, _N_EFF, "bogus")
    err = excinfo.value
    assert isinstance(err, ValueError)  # still what a bad value raised before
    assert err.code == "bad_setting"
    assert err.path == _N_EFF
    assert err.value == "bogus"
    assert err.expected == "one of " + ", ".join(repr(c) for c in _N_EFF_CHOICES)
    assert err.to_dict()["path"] == _N_EFF
    assert _md5(bare_ftmw) == before  # the file is untouched
    assert _row(bare_ftmw, _N_EFF).source == SOURCE_DEFAULT


def test_a_bad_choice_does_not_disturb_a_previously_set_value(
    bare_ftmw: Path,
) -> None:
    set_setting(bare_ftmw, _N_EFF, "kish_mag")
    before = _md5(bare_ftmw)
    with pytest.raises(BadSettingError):
        set_setting(bare_ftmw, _N_EFF, "bogus")
    assert _md5(bare_ftmw) == before
    assert _row(bare_ftmw, _N_EFF).value == "kish_mag"


def test_a_choice_is_case_sensitive(bare_ftmw: Path) -> None:
    with pytest.raises(BadSettingError):
        set_setting(bare_ftmw, _N_EFF, "KISH_MAG")


def test_unset_skips_the_choices_check(bare_ftmw: Path) -> None:
    """Mutation: run the choices check on ``None`` (the unset request)."""
    set_setting(bare_ftmw, _N_EFF, "kish_mag")
    assert set_setting(bare_ftmw, _N_EFF, None).value is None
    assert unset_setting(bare_ftmw, _N_EFF).value is None
    assert _row(bare_ftmw, _N_EFF).source == SOURCE_DEFAULT


def test_a_wrong_type_is_refused_before_the_choices_comparison(
    bare_ftmw: Path,
) -> None:
    """A wrong type is still refused as a coercion error (path and value
    intact), before any choices comparison is attempted."""
    with pytest.raises(BadSettingError) as excinfo:
        set_setting(bare_ftmw, _N_EFF, 3)
    assert excinfo.value.path == _N_EFF and excinfo.value.value == 3


# Synthetic typings: no registry field declares bounds yet, so the bounds rule
# is exercised on ``_check_typing`` directly.
def _refused(value, **bounds) -> bool:
    typing = FieldTyping(bounds=make_bounds(**bounds))
    try:
        mutation._check_typing("x.y", value, value, typing)
    except BadSettingError as err:
        assert err.path == "x.y" and err.value == value
        return True
    return False


@pytest.mark.parametrize(
    "value, refused",
    [(-0.1, True), (0.0, False), (0.5, False), (1.0, False), (1.1, True)],
)
def test_inclusive_bounds_admit_their_ends(value: float, refused: bool) -> None:
    assert _refused(value, min=0.0, max=1.0) is refused


@pytest.mark.parametrize(
    "value, refused",
    [(0.0, True), (1e-12, False), (0.999, False), (1.0, True), (2.0, True)],
)
def test_exclusive_bounds_refuse_their_ends(value: float, refused: bool) -> None:
    """Mutation: treat ``min_inclusive`` / ``max_inclusive`` False as True."""
    assert (
        _refused(value, min=0.0, max=1.0, min_inclusive=False, max_inclusive=False)
        is refused
    )


def test_a_one_sided_bound_leaves_the_other_end_open() -> None:
    assert _refused(-1e9, min=0.0) is True
    assert _refused(1e9, min=0.0) is False
    assert _refused(1e9, max=10.0) is True
    assert _refused(-1e9, max=10.0) is False


def test_nan_is_refused_whenever_bounds_exist() -> None:
    """Mutation: let nan through (every comparison with it is False)."""
    assert _refused(float("nan"), min=0.0, max=1.0) is True
    assert _refused(float("nan"), min=0.0) is True
    assert _refused(float("nan"), max=1.0, max_inclusive=False) is True


def test_bounds_apply_to_each_element_of_a_tuple_or_list() -> None:
    typing = FieldTyping(bounds=make_bounds(min=0.0, max=10.0))
    mutation._check_typing("x.y", (1.0, 9.0), (1.0, 9.0), typing)
    mutation._check_typing("x.y", [0.0, 10.0], [0.0, 10.0], typing)
    for bad in ((1.0, 11.0), [-1.0, 5.0], (float("nan"), 1.0)):
        with pytest.raises(BadSettingError) as excinfo:
            mutation._check_typing("x.y", bad, bad, typing)
        assert excinfo.value.value == bad


def test_the_bounds_error_names_the_interval_and_the_raw_value() -> None:
    typing = FieldTyping(bounds=make_bounds(min=0.0, max=1.0, min_inclusive=False))
    with pytest.raises(BadSettingError) as excinfo:
        mutation._check_typing("x.y", 5.0, "5", typing)
    err = excinfo.value
    assert err.expected == "a value in (0.0, 1.0]"
    assert err.value == "5"  # what the caller passed, not the coerced number


def test_bounds_do_not_apply_to_a_non_numeric_value() -> None:
    typing = FieldTyping(bounds=make_bounds(min=0.0, max=1.0))
    mutation._check_typing("x.y", "text", "text", typing)
    mutation._check_typing("x.y", True, True, typing)  # a bool is not a number


def test_no_declared_typing_checks_nothing() -> None:
    mutation._check_typing("x.y", "anything", "anything", FieldTyping())
    mutation._check_typing("x.y", 1e99, 1e99, FieldTyping())


def test_coerce_or_unset_enforces_the_typing_it_is_given() -> None:
    typing = FieldTyping(bounds=make_bounds(min=0.0))
    assert mutation._coerce_or_unset("x.y", float, False, "2.5", typing) == 2.5
    with pytest.raises(BadSettingError) as excinfo:
        mutation._coerce_or_unset("x.y", float, False, "-2.5", typing)
    assert excinfo.value.value == "-2.5"
    # unset skips the typing; so does no typing at all
    assert mutation._coerce_or_unset("x.y", float, True, None, typing) is None
    assert mutation._coerce_or_unset("x.y", float, False, "-2.5") == -2.5


def test_field_typing_reads_the_declaration_from_the_dataclass() -> None:
    from ftmwpipeline.core import settings as ft_settings

    typing = mutation._field_typing(
        fit_mod.StageFitSettings, "conservative", "n_eff_kind"
    )
    assert typing.choices == _N_EFF_CHOICES
    stated_nothing = mutation._field_typing(ft_settings.FTSettings, None, "start_us")
    assert stated_nothing.choices is None and stated_nothing.bounds is None


# --- unset_setting ----------------------------------------------------------
def test_unset_restores_the_resolver_layers(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage2.window_mhz", "123.5")
    assert _row(bare_ftmw, "stage2.window_mhz").source == SOURCE_FTMW

    result = unset_setting(bare_ftmw, "stage2.window_mhz")
    assert result.value is None
    row = _row(bare_ftmw, "stage2.window_mhz")
    assert row.source == SOURCE_DEFAULT
    assert row.value == row.hard_default


def test_unset_is_set_with_none(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage2b.gaussian.snr_min", "7.5")
    assert set_setting(bare_ftmw, "stage2b.gaussian.snr_min", None).value is None
    assert _row(bare_ftmw, "stage2b.gaussian.snr_min").source == SOURCE_DEFAULT


def test_unset_stage1_field(bare_ftmw: Path) -> None:
    """Stage 1's record is authoritative, so an unset re-resolves the field from
    the layers below it (here: none, and no FID to make the window concrete)
    and records that, rather than leaving a gap for a later layer to fill."""
    set_setting(bare_ftmw, "stage1.start_us", "3.25")
    assert _row(bare_ftmw, "stage1.start_us").value == 3.25
    unset_setting(bare_ftmw, "stage1.start_us")
    row = _row(bare_ftmw, "stage1.start_us")
    assert row.source == SOURCE_FTMW
    assert row.value is None


def test_unset_invalidates_like_a_set(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage2.window_mhz", "90")
    _stamp_stages(
        bare_ftmw,
        ["stage0_fid_data", "stage1_complex_ft", "stage2_noise_result", "stage3_peaks"],
    )
    result = unset_setting(bare_ftmw, "stage2.window_mhz")
    assert result.invalidated == ("noise", "peaks")


def test_unset_unknown_knob_raises(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        unset_setting(bare_ftmw, "stage2.no_such_field")


# --- export_settings -------------------------------------------------------
def test_export_round_trips_via_load_preset(bare_ftmw: Path, tmp_path: Path) -> None:
    with atomic_write(str(bare_ftmw)):
        save_noise_settings_to_h5(
            str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=77.0, line_k=9.0)
        )
    fit = fit_mod.StageFitSettings(shape=fit_mod.ShapeSpec(ps_mod.PeakShape.GAUSSIAN))
    fit.tau.max_decay_factor = 4.0
    with atomic_write(str(bare_ftmw)):
        save_stage_fit_settings_to_h5(str(bare_ftmw), fit)

    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out)
    assert "stage2.window_mhz" in result.paths
    assert "stage5.shape" in result.paths

    noise = noise_mod.load_preset(out)
    assert noise.window_mhz == 77.0
    assert noise.line_k == 9.0
    loaded_fit = fit_mod.load_preset(out)
    assert loaded_fit.shape.kind is ps_mod.PeakShape.GAUSSIAN
    assert loaded_fit.tau.max_decay_factor == 4.0


def test_export_selector_scopes_blocks(bare_ftmw: Path, tmp_path: Path) -> None:
    with atomic_write(str(bare_ftmw)):
        save_noise_settings_to_h5(
            str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=77.0)
        )
    fit = fit_mod.StageFitSettings()
    fit.tau.max_decay_factor = 4.0
    with atomic_write(str(bare_ftmw)):
        save_stage_fit_settings_to_h5(str(bare_ftmw), fit)

    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out, "stage5")
    assert all(p.startswith("stage5.") for p in result.paths)
    noise = noise_mod.load_preset(out)
    assert noise.window_mhz is None  # stage2 block excluded by selector


def test_export_nothing_persisted_writes_empty(bare_ftmw: Path, tmp_path: Path) -> None:
    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out, name="empty")
    assert result.paths == ()
    assert out.exists()
