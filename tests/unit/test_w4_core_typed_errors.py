"""Stage-call parameter refusals in the core stages are typed (Wave 4, core group).

Each refusal is a program-actionable reason (a bad setting value) raised as a
``BadSettingError`` naming the setting, still a ``ValueError`` for callers that
caught the built-in.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.stage5_impl import _resolve_tau_calibration_for_fit
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.io.data_loaders.registry import get_loader
from ftmwpipeline.pipeline import Pipeline
from ftmwpipeline.utils.signal_processing import make_apodization

pytestmark = [pytest.mark.unit]


@pytest.fixture
def pipeline(tmp_path) -> Pipeline:
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(6325) * 0.01))
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return Pipeline.open(out)


@pytest.mark.parametrize(
    "kwargs, path",
    [
        ({"start_us": -1.0}, "ft.start_us"),
        ({"start_us": 5.0, "end_us": 2.0}, "ft.end_us"),
        ({"end_us": 100.0}, "ft.end_us"),
        ({"trim": (1.0e7, 2.0e7)}, "ft.trim"),
    ],
)
def test_compute_ft_refusals_name_the_setting(pipeline, kwargs, path):
    with pytest.raises(BadSettingError) as exc:
        pipeline.compute_ft(**kwargs)
    assert exc.value.path == path
    assert isinstance(exc.value, ValueError)
    assert exc.value.to_dict()["code"] == "bad_setting"


def test_timebase_bad_kappa_and_snr(pipeline):
    pipeline.compute_ft(trim=(40900.0, 41020.0))
    with pytest.raises(BadSettingError) as exc:
        pipeline.calibrate_timebase(kappa_sys=-1.0)
    assert exc.value.path == "kappa_sys"
    with pytest.raises(BadSettingError) as exc:
        pipeline.calibrate_timebase(snr_min=0.0)
    assert exc.value.path == "snr_min"


def test_timebase_without_clock_declaration(pipeline):
    pipeline.compute_ft(trim=(40900.0, 41020.0))
    with pytest.raises(BadSettingError) as exc:
        pipeline.calibrate_timebase()
    assert exc.value.path == "stage5.spur.clocks"


def test_timebase_malformed_clocks_argument(pipeline):
    pipeline.compute_ft(trim=(40900.0, 41020.0))
    with pytest.raises(BadSettingError) as exc:
        pipeline.calibrate_timebase(clocks=[{"nope": 1}])
    assert exc.value.path == "clocks"


def test_unknown_format_is_a_bad_setting():
    with pytest.raises(BadSettingError) as exc:
        get_loader("bogus")
    assert exc.value.path == "format"
    assert exc.value.value == "bogus"


def test_import_with_unknown_format_is_a_bad_setting(tmp_path):
    from ftmwpipeline._internal.stage0_impl import import_data_impl

    src = tmp_path / "src.h5"
    src.write_bytes(b"x")
    with pytest.raises(BadSettingError) as exc:
        import_data_impl(str(tmp_path / "o.ftmw"), str(src), format_name="bogus")
    assert exc.value.path == "format"


def test_apodization_names_the_caller_setting():
    t = np.arange(16, dtype=float)
    with pytest.raises(BadSettingError) as exc:
        make_apodization("nonsense-window", t, setting="stage3.primary_pass.x")
    assert exc.value.path == "stage3.primary_pass.x"
    with pytest.raises(BadSettingError) as exc:
        make_apodization("exp", t)
    assert exc.value.path == "apodize_us"


def test_tau_override_pair_refusals():
    with pytest.raises(BadSettingError) as exc:
        _resolve_tau_calibration_for_fit(None, 3.0, None)
    assert exc.value.path == "tau_maj_override_us"
    with pytest.raises(BadSettingError) as exc:
        _resolve_tau_calibration_for_fit(None, 3.0, -1.0)
    assert exc.value.path == "sigma_tau_override_us"


def test_tau_n_seg_too_large_names_the_setting(pipeline):
    from ftmwpipeline.core.tau_calibration_settings import (
        StftSubSettings,
        TauCalibrationSettings,
    )

    pipeline.compute_ft(trim=(40900.0, 41020.0))
    pipeline.estimate_noise()
    with pytest.raises(BadSettingError) as exc:
        pipeline.calibrate_tau(
            settings=TauCalibrationSettings(stft=StftSubSettings(n_seg=100000))
        )
    assert exc.value.path == "stage2b.stft.n_seg"


def test_bad_primary_window_names_the_setting(pipeline):
    from ftmwpipeline.core.peak_detection_settings import (
        PeakDetectionSettings,
        PrimaryPassSubSettings,
    )

    pipeline.compute_ft(trim=(40900.0, 41020.0))
    pipeline.estimate_noise()
    with pytest.raises(BadSettingError) as exc:
        pipeline.detect_peaks(
            settings=PeakDetectionSettings(
                primary_pass=PrimaryPassSubSettings(primary_window="zzz")
            )
        )
    assert exc.value.path == "stage3.primary_pass.primary_window"
