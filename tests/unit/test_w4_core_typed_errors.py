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


# ---------------------------------------------------------------------------
# Public-interface coverage (api + Pipeline), every converted setting refusal
# ---------------------------------------------------------------------------
import ftmwpipeline.api as ftmw  # noqa: E402
from ftmwpipeline.core.noise_settings import NoiseSettings  # noqa: E402
from ftmwpipeline.core.peak_detection_settings import (  # noqa: E402
    PeakDetectionSettings,
    PromotionSubSettings,
    SavgolSubSettings,
)
from ftmwpipeline.core.start_detection_settings import (  # noqa: E402
    StartDetectionSettings,
)
from ftmwpipeline.core.tau_calibration_settings import (  # noqa: E402
    BandSubSettings,
    GaussianSubSettings,
    StftSubSettings,
    TauCalibrationSettings,
)
from ftmwpipeline.file_manager import StageDependencyError  # noqa: E402

_TRIM = (40900.0, 41020.0)


def _assert_bad_setting(exc, path):
    err = exc.value
    assert err.path == path
    # The built-in it replaced still catches it.
    assert isinstance(err, ValueError)
    d = err.to_dict()
    assert d["code"] == "bad_setting"
    assert d["path"] == path
    assert d["expected"]


@pytest.fixture
def ft_file(pipeline) -> str:
    pipeline.compute_ft(trim=_TRIM)
    return str(pipeline.filepath)


@pytest.fixture
def noise_file(ft_file) -> str:
    ftmw.estimate_noise(ft_file)
    return ft_file


@pytest.mark.parametrize(
    "kwargs, path, value",
    [
        ({"n_iter": 0}, "stage2.n_iter", 0),
        ({"smoothing_percentile": 120.0}, "stage2.smoothing_percentile", 120.0),
    ],
)
def test_api_estimate_noise_bad_knob(ft_file, kwargs, path, value):
    # Mutation: the knob reaches the kernel and comes back as a flattened error.
    with pytest.raises(BadSettingError) as exc:
        ftmw.estimate_noise(ft_file, settings=NoiseSettings(**kwargs))
    _assert_bad_setting(exc, path)
    assert exc.value.value == value


def test_api_calibrate_tau_bad_shape(noise_file):
    # Mutation: _check_shape raises a plain ValueError.
    with pytest.raises(BadSettingError) as exc:
        ftmw.calibrate_tau(noise_file, shape="bogus")
    _assert_bad_setting(exc, "shape")
    assert exc.value.value == "bogus"


@pytest.mark.parametrize(
    "shape, settings, path",
    [
        (
            "lorentzian",
            TauCalibrationSettings(stft=StftSubSettings(sigma_time=-1.0)),
            "stage2b.stft.sigma_time",
        ),
        (
            "lorentzian",
            TauCalibrationSettings(
                band=BandSubSettings(
                    compute_band_majorities=True, band_edges_mhz=(10.0,)
                )
            ),
            "stage2b.band.band_edges_mhz",
        ),
        (
            "lorentzian",
            TauCalibrationSettings(
                band=BandSubSettings(
                    compute_band_majorities=True,
                    band_edges_mhz=(40950.0,),
                    band_labels=("a",),
                )
            ),
            "stage2b.band.band_labels",
        ),
        (
            "gaussian",
            TauCalibrationSettings(gaussian=GaussianSubSettings(tau_G_bound_hi=1e-9)),
            "stage2b.gaussian.tau_G_bound_hi",
        ),
        (
            "gaussian",
            TauCalibrationSettings(
                gaussian=GaussianSubSettings(tau_G_upper_fraction=1.5)
            ),
            "stage2b.gaussian.tau_G_upper_fraction",
        ),
    ],
)
def test_api_calibrate_tau_bad_settings(noise_file, shape, settings, path):
    # Mutation: the kernel's own ValueError reaches the caller untyped.
    with pytest.raises(BadSettingError) as exc:
        ftmw.calibrate_tau(noise_file, shape=shape, settings=settings)
    _assert_bad_setting(exc, path)


def test_api_calibrate_tau_and_recommend_shape_need_trim(pipeline):
    # Mutation: the missing-trim refusal is a plain ValueError.
    pipeline.compute_ft()
    path = str(pipeline.filepath)
    ftmw.estimate_noise(path)
    with pytest.raises(BadSettingError) as exc:
        ftmw.calibrate_tau(path)
    _assert_bad_setting(exc, "ft.trim")
    assert exc.value.value is None
    with pytest.raises(BadSettingError) as exc:
        ftmw.recommend_shape(path)
    _assert_bad_setting(exc, "ft.trim")


def test_api_recommend_shape_bad_n_seg(noise_file):
    # Mutation: the shape-recommendation path skips the n_seg pre-check.
    with pytest.raises(BadSettingError) as exc:
        ftmw.recommend_shape(
            noise_file,
            settings=TauCalibrationSettings(stft=StftSubSettings(n_seg=100000)),
        )
    _assert_bad_setting(exc, "stage2b.stft.n_seg")


@pytest.mark.parametrize(
    "settings, path",
    [
        (
            PeakDetectionSettings(promotion=PromotionSubSettings(min_snr=0.0)),
            "stage3.promotion.min_snr",
        ),
        (
            PeakDetectionSettings(savgol=SavgolSubSettings(sg_order=0)),
            "stage3.savgol.sg_order",
        ),
        (
            PeakDetectionSettings(savgol=SavgolSubSettings(sg_window=10)),
            "stage3.savgol.sg_window",
        ),
    ],
)
def test_api_detect_peaks_bad_settings(noise_file, settings, path):
    # Mutation: Stage 3 hands the knob to the kernel and its ValueError escapes.
    with pytest.raises(BadSettingError) as exc:
        ftmw.detect_peaks(noise_file, settings=settings)
    _assert_bad_setting(exc, path)


def test_api_detect_start_time_fid_too_short(pipeline):
    # Mutation: the short-FID check is a plain ValueError.
    with pytest.raises(BadSettingError) as exc:
        ftmw.detect_start_time(
            str(pipeline.filepath), settings=StartDetectionSettings(step_us=50.0)
        )
    _assert_bad_setting(exc, "stage0.step_us")
    assert exc.value.value == 50.0


def test_api_run_pipeline_requires_trim(tmp_path):
    # Mutation: run_pipeline raises a plain ValueError for trim=None.
    with pytest.raises(BadSettingError) as exc:
        ftmw.run_pipeline(str(tmp_path / "s"), str(tmp_path / "o.ftmw"), trim=None)
    _assert_bad_setting(exc, "trim")


def test_read_noise_result_before_noise_is_stage_not_run(ft_file):
    # Mutation: load_noise_result_impl raises ValueError and the wrapper
    # flattens it to RuntimeError.
    from ftmwpipeline._internal.stage2_impl import load_noise_result_impl

    with pytest.raises(StageDependencyError) as exc:
        load_noise_result_impl(ft_file)
    assert isinstance(exc.value, ValueError)
    d = exc.value.to_dict()
    assert d["code"] == "stage_not_run"
    assert d["missing_dependencies"] == ["noise"]
    assert d["command"] == "noise run"


def test_typed_error_survives_the_stage_wrappers(tmp_path):
    # Mutation: the ``except Exception -> RuntimeError`` wrapper in stage1/stage2
    # runs before ``except PipelineFileError: raise``.
    from ftmwpipeline.file_manager import PipelineFileNotFoundError

    missing = str(tmp_path / "nope.ftmw")
    for call in (ftmw.compute_ft, ftmw.estimate_noise):
        with pytest.raises(PipelineFileNotFoundError):
            call(missing)
