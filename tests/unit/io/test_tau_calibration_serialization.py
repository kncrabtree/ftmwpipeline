"""Round-trip tests for TauCalibrationResult HDF5 serialization."""

from __future__ import annotations

import h5py
import numpy as np

from ftmwpipeline.fitting.tau_calibration import (
    FrequencyThird,
    GMMBimodality,
    SpurCluster,
    TauCalibrationResult,
)
from ftmwpipeline.io.tau_calibration_serialization import (
    GROUP_PATH,
    SCHEMA_VERSION,
    load_tau_calibration_from_hdf5,
    save_tau_calibration_to_hdf5,
)


def _make_sample_result(
    *, n_contrib: int = 50, n_clusters: int = 3
) -> TauCalibrationResult:
    rng = np.random.default_rng(0xC0FFEE)
    taus = rng.normal(6.0, 1.0, size=n_contrib)
    snrs = rng.uniform(10.0, 100.0, size=n_contrib)
    freqs = np.linspace(27000.0, 39000.0, n_contrib)
    bin_indices = np.arange(100, 100 + n_contrib, dtype=np.int64)
    clusters = []
    base = 1000
    for i in range(n_clusters):
        peak = base + 200 * i
        member_bins = tuple(range(peak - 4, peak + 5))
        clusters.append(
            SpurCluster(
                center_freq_mhz=27000.0 + 1000.0 * i,
                peak_bin_index=peak,
                n_bins=len(member_bins),
                bin_indices=member_bins,
            )
        )
    bm = GMMBimodality(
        n=n_contrib,
        mu1=6.0,
        sigma1=1.0,
        mu_a=5.0,
        sigma_a=0.8,
        mu_b=7.5,
        sigma_b=1.2,
        pi_a=0.6,
        aic1=120.0,
        aic2=110.5,
        delta_aic=9.5,
        two_component_preferred=True,
        dominant_weight=0.6,
    )
    thirds = (
        FrequencyThird("low", 27000.0, 31000.0, 17, 7.34),
        FrequencyThird("mid", 31000.0, 35000.0, 16, 6.36),
        FrequencyThird("high", 35000.0, 39000.0, 17, 6.06),
    )
    return TauCalibrationResult(
        tau_maj_us=6.33,
        sigma_tau_us=1.62,
        n_contributors=n_contrib,
        n_spur_bins=sum(c.n_bins for c in clusters),
        spur_clusters=tuple(clusters),
        bimodality=bm,
        pearson_r_log_snr_vs_tau=-0.30,
        pearson_r_freq_vs_tau=-0.32,
        frequency_thirds=thirds,
        contributor_bin_indices=bin_indices,
        contributor_taus_us=taus,
        contributor_snrs=snrs,
        contributor_freqs_mhz=freqs,
        n_seg=10,
        t_sigma=5.0,
        tau_max_us=63.25,
        rss_gate_factor=5.0,
        sample_dt_us=0.020,
        start_us=2.35,
        end_us=15.0,
        probe_freq_mhz=40000.0,
        sideband="lower",
        trim_lo_mhz=26500.0,
        trim_hi_mhz=40000.0,
        sigma_x_full=4.68e-7,
        sigma_frame=1.48e-7,
        snr_weighted=True,
        preconditions_passed=False,
        preconditions_notes=(
            "ok",
            "strongly bimodal",
            "sigma_tau/tau_maj=0.26 >= 0.20",
        ),
    )


def _assert_results_equal(a: TauCalibrationResult, b: TauCalibrationResult) -> None:
    # Scalars
    for field in (
        "tau_maj_us",
        "sigma_tau_us",
        "n_contributors",
        "n_spur_bins",
        "pearson_r_log_snr_vs_tau",
        "pearson_r_freq_vs_tau",
        "n_seg",
        "t_sigma",
        "tau_max_us",
        "rss_gate_factor",
        "sample_dt_us",
        "start_us",
        "end_us",
        "probe_freq_mhz",
        "sideband",
        "trim_lo_mhz",
        "trim_hi_mhz",
        "sigma_x_full",
        "sigma_frame",
        "snr_weighted",
        "preconditions_passed",
    ):
        assert getattr(a, field) == getattr(b, field), field
    assert tuple(a.preconditions_notes) == tuple(b.preconditions_notes)
    # Bimodality (every attribute)
    for field in (
        "n",
        "mu1",
        "sigma1",
        "mu_a",
        "sigma_a",
        "mu_b",
        "sigma_b",
        "pi_a",
        "aic1",
        "aic2",
        "delta_aic",
        "two_component_preferred",
        "dominant_weight",
    ):
        assert getattr(a.bimodality, field) == getattr(b.bimodality, field), field
    # Frequency thirds
    assert len(a.frequency_thirds) == len(b.frequency_thirds)
    for t1, t2 in zip(a.frequency_thirds, b.frequency_thirds):
        assert t1 == t2
    # Contributor arrays
    np.testing.assert_array_equal(a.contributor_bin_indices, b.contributor_bin_indices)
    np.testing.assert_array_equal(a.contributor_taus_us, b.contributor_taus_us)
    np.testing.assert_array_equal(a.contributor_snrs, b.contributor_snrs)
    np.testing.assert_array_equal(a.contributor_freqs_mhz, b.contributor_freqs_mhz)
    # Spur clusters
    assert len(a.spur_clusters) == len(b.spur_clusters)
    for c1, c2 in zip(a.spur_clusters, b.spur_clusters):
        assert c1.center_freq_mhz == c2.center_freq_mhz
        assert c1.peak_bin_index == c2.peak_bin_index
        assert c1.n_bins == c2.n_bins
        assert c1.bin_indices == c2.bin_indices


class TestRoundTrip:
    def test_full_round_trip(self, tmp_path):
        original = _make_sample_result()
        test_file = tmp_path / "tau_calib.h5"
        with h5py.File(test_file, "w") as h5f:
            grp = h5f.create_group(GROUP_PATH)
            save_tau_calibration_to_hdf5(original, grp)
        with h5py.File(test_file, "r") as h5f:
            grp = h5f[GROUP_PATH]
            loaded = load_tau_calibration_from_hdf5(grp)
        _assert_results_equal(original, loaded)

    def test_round_trip_no_spurs_no_thirds(self, tmp_path):
        result = TauCalibrationResult(
            tau_maj_us=5.0,
            sigma_tau_us=0.6,
            n_contributors=5,
            n_spur_bins=0,
            spur_clusters=(),
            bimodality=GMMBimodality(
                n=5,
                mu1=float("nan"),
                sigma1=float("nan"),
                mu_a=float("nan"),
                sigma_a=float("nan"),
                mu_b=float("nan"),
                sigma_b=float("nan"),
                pi_a=float("nan"),
                aic1=float("nan"),
                aic2=float("nan"),
                delta_aic=float("nan"),
                two_component_preferred=False,
                dominant_weight=float("nan"),
            ),
            pearson_r_log_snr_vs_tau=float("nan"),
            pearson_r_freq_vs_tau=float("nan"),
            frequency_thirds=(),
            contributor_bin_indices=np.arange(5, dtype=np.int64),
            contributor_taus_us=np.full(5, 5.0),
            contributor_snrs=np.full(5, 20.0),
            contributor_freqs_mhz=np.linspace(30000.0, 35000.0, 5),
            n_seg=10,
            t_sigma=5.0,
            tau_max_us=63.25,
            rss_gate_factor=5.0,
            sample_dt_us=0.020,
            start_us=0.0,
            end_us=12.65,
            probe_freq_mhz=40000.0,
            sideband="lower",
            trim_lo_mhz=26500.0,
            trim_hi_mhz=40000.0,
            sigma_x_full=1e-6,
            sigma_frame=3e-7,
            snr_weighted=True,
            preconditions_passed=False,
            preconditions_notes=("only 5 contributors", "ok", "ok"),
        )
        test_file = tmp_path / "tau_minimal.h5"
        with h5py.File(test_file, "w") as h5f:
            save_tau_calibration_to_hdf5(result, h5f.create_group(GROUP_PATH))
        with h5py.File(test_file, "r") as h5f:
            loaded = load_tau_calibration_from_hdf5(h5f[GROUP_PATH])
        # NaN-aware comparison for the GMM fields.
        assert np.isnan(loaded.bimodality.mu1)
        assert np.isnan(loaded.bimodality.delta_aic)
        assert loaded.spur_clusters == ()
        assert loaded.frequency_thirds == ()
        np.testing.assert_array_equal(
            loaded.contributor_taus_us,
            result.contributor_taus_us,
        )

    def test_overwrite_clears_old_contents(self, tmp_path):
        test_file = tmp_path / "tau_overwrite.h5"
        with h5py.File(test_file, "w") as h5f:
            grp = h5f.create_group(GROUP_PATH)
            # Write a result with 3 clusters.
            save_tau_calibration_to_hdf5(_make_sample_result(n_clusters=3), grp)
            # Then overwrite with 1 cluster — the new group must have only 1.
            save_tau_calibration_to_hdf5(_make_sample_result(n_clusters=1), grp)
        with h5py.File(test_file, "r") as h5f:
            loaded = load_tau_calibration_from_hdf5(h5f[GROUP_PATH])
        assert len(loaded.spur_clusters) == 1

    def test_schema_version_present(self, tmp_path):
        test_file = tmp_path / "tau_schema.h5"
        with h5py.File(test_file, "w") as h5f:
            grp = h5f.create_group(GROUP_PATH)
            save_tau_calibration_to_hdf5(_make_sample_result(), grp)
            assert h5f[GROUP_PATH]["algorithm_info"].attrs["version"] == SCHEMA_VERSION
