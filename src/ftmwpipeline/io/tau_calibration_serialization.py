"""HDF5 serialization for :class:`TauCalibrationResult`.

Persisted at ``/stage2b_tau_calibration``. The schema deliberately stores
only the artifacts downstream consumers and visualizations need; the full
``(n_seg, n_bins)`` STFT magnitude grid is large (~50 MB on the 2638 fixture
even before compression) and can be recomputed on demand from the FID plus
the persisted calibration knobs (``n_seg``, ``start_us``, ``end_us``,
``sample_dt_us``, ``sigma_x_full``).

HDF5 layout::

    /stage2b_tau_calibration/
        scalars                  group, attrs:
            tau_maj_us, sigma_tau_us, n_seg, t_sigma, tau_max_us,
            rss_gate_factor, sample_dt_us, start_us, end_us,
            probe_freq_mhz, sideband, trim_lo_mhz, trim_hi_mhz,
            sigma_x_full, sigma_frame, snr_weighted,
            preconditions_passed, n_contributors, n_spur_bins
        preconditions_notes      dataset, str, length 3
        bimodality               group, attrs:
            n, mu1, sigma1, mu_a, sigma_a, mu_b, sigma_b, pi_a,
            aic1, aic2, delta_aic, two_component_preferred,
            dominant_weight
        pearson                  group, attrs:
            r_log_snr_vs_tau, r_freq_vs_tau
        contributors             group:
            bin_indices (int64)
            taus_us (float64)
            snrs (float64)
            freqs_mhz (float64)
        spur_clusters            group:
            center_freqs_mhz (float64)
            peak_bin_indices (int64)
            n_bins (int32)
            saturated (bool)         per-cluster flat/CW-tone flag
                                     (spur_by_tau); absent on legacy files
            (cluster bin-index lists are stored as a single flat int64
            dataset ``bin_indices_flat`` with an int32 ``offsets`` dataset
            so each cluster's member bins are recoverable; matches the
            CSR pattern used elsewhere in the pipeline.)
        frequency_thirds         group, one subgroup per third (low/mid/high):
            attrs: label, freq_lo_mhz, freq_hi_mhz, n, median_tau_us
        algorithm_info           group, attrs:
            method, version
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

from ..fitting.tau_calibration import (
    BandMajority,
    FrequencyThird,
    GMMBimodality,
    SpurCluster,
    TauCalibrationResult,
)
from ._hdf5_helpers import (
    REQUIRED,
    ColumnSpec,
    build_columns,
    load_json_attr,
    read_attr_value,
    read_dataset_column,
    reset_group,
    resolve_column_selection,
)

SCHEMA_VERSION = "1.0"
GROUP_PATH = "stage2b_tau_calibration"

#: The Gaussian twin: ``tau run --gaussian`` writes a second, independent
#: calibration here, in the identical layout. Both may coexist on one file.
GAUSSIAN_GROUP_PATH = "stage2b_tau_G_calibration"


__all__ = [
    "save_tau_calibration_to_hdf5",
    "load_tau_calibration_from_hdf5",
    "SCHEMA_VERSION",
    "GROUP_PATH",
    "GAUSSIAN_GROUP_PATH",
    "TAU_BAND_COLUMN_SPECS",
    "TAU_THIRD_COLUMN_SPECS",
    "TAU_CONTRIBUTOR_COLUMN_SPECS",
    "TAU_SPUR_COLUMN_SPECS",
    "read_tau_band_columns",
    "read_tau_third_columns",
    "read_tau_contributor_columns",
    "read_tau_spur_columns",
    "read_tau_scalars",
]


def save_tau_calibration_to_hdf5(
    result: TauCalibrationResult,
    h5_group: h5py.Group,
) -> None:
    """Save a :class:`TauCalibrationResult` into the given HDF5 group.

    The group is wiped and rebuilt; existing subgroups under it are deleted.
    Callers should pass a freshly-created group or accept that all of its
    contents will be replaced.
    """
    # Wipe existing contents.
    reset_group(h5_group, attrs=True)

    # --- scalars -------------------------------------------------------------
    scalars = h5_group.create_group("scalars")
    scalars.attrs["tau_maj_us"] = float(result.tau_maj_us)
    scalars.attrs["sigma_tau_us"] = float(result.sigma_tau_us)
    scalars.attrs["n_seg"] = int(result.n_seg)
    scalars.attrs["t_sigma"] = float(result.t_sigma)
    scalars.attrs["tau_max_us"] = float(result.tau_max_us)
    scalars.attrs["rss_gate_factor"] = float(result.rss_gate_factor)
    scalars.attrs["sample_dt_us"] = float(result.sample_dt_us)
    scalars.attrs["start_us"] = float(result.start_us)
    scalars.attrs["end_us"] = float(result.end_us)
    scalars.attrs["probe_freq_mhz"] = float(result.probe_freq_mhz)
    scalars.attrs["sideband"] = result.sideband
    scalars.attrs["trim_lo_mhz"] = float(result.trim_lo_mhz)
    scalars.attrs["trim_hi_mhz"] = float(result.trim_hi_mhz)
    scalars.attrs["sigma_x_full"] = float(result.sigma_x_full)
    scalars.attrs["sigma_frame"] = float(result.sigma_frame)
    scalars.attrs["snr_weighted"] = bool(result.snr_weighted)
    scalars.attrs["preconditions_passed"] = bool(result.preconditions_passed)
    scalars.attrs["n_contributors"] = int(result.n_contributors)
    scalars.attrs["n_spur_bins"] = int(result.n_spur_bins)

    # --- preconditions notes (variable-length string dataset) --------------
    notes = np.asarray(result.preconditions_notes, dtype=object)
    h5_group.create_dataset(
        "preconditions_notes",
        data=notes,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # --- bimodality ---------------------------------------------------------
    bm = result.bimodality
    bg = h5_group.create_group("bimodality")
    for key, value in {
        "n": int(bm.n),
        "mu1": float(bm.mu1),
        "sigma1": float(bm.sigma1),
        "mu_a": float(bm.mu_a),
        "sigma_a": float(bm.sigma_a),
        "mu_b": float(bm.mu_b),
        "sigma_b": float(bm.sigma_b),
        "pi_a": float(bm.pi_a),
        "aic1": float(bm.aic1),
        "aic2": float(bm.aic2),
        "delta_aic": float(bm.delta_aic),
        "two_component_preferred": bool(bm.two_component_preferred),
        "dominant_weight": float(bm.dominant_weight),
    }.items():
        bg.attrs[key] = value

    # --- pearson ------------------------------------------------------------
    pg = h5_group.create_group("pearson")
    pg.attrs["r_log_snr_vs_tau"] = float(result.pearson_r_log_snr_vs_tau)
    pg.attrs["r_freq_vs_tau"] = float(result.pearson_r_freq_vs_tau)

    # --- contributors -------------------------------------------------------
    cg = h5_group.create_group("contributors")
    cg.create_dataset(
        "bin_indices",
        data=np.asarray(result.contributor_bin_indices, dtype=np.int64),
        compression="gzip",
        compression_opts=6,
    )
    cg.create_dataset(
        "taus_us",
        data=np.asarray(result.contributor_taus_us, dtype=np.float64),
        compression="gzip",
        compression_opts=6,
    )
    cg.create_dataset(
        "snrs",
        data=np.asarray(result.contributor_snrs, dtype=np.float64),
        compression="gzip",
        compression_opts=6,
    )
    cg.create_dataset(
        "freqs_mhz",
        data=np.asarray(result.contributor_freqs_mhz, dtype=np.float64),
        compression="gzip",
        compression_opts=6,
    )

    # --- spur clusters (CSR-style flat layout) -----------------------------
    sg = h5_group.create_group("spur_clusters")
    clusters = list(result.spur_clusters)
    n_clusters = len(clusters)
    if n_clusters > 0:
        centers = np.asarray([c.center_freq_mhz for c in clusters], dtype=np.float64)
        peak_bins = np.asarray([c.peak_bin_index for c in clusters], dtype=np.int64)
        n_bins_arr = np.asarray([c.n_bins for c in clusters], dtype=np.int32)
        saturated_arr = np.asarray([c.saturated for c in clusters], dtype=bool)
        flat = np.concatenate(
            [np.asarray(c.bin_indices, dtype=np.int64) for c in clusters]
        )
        offsets: np.ndarray = np.empty(n_clusters + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(n_bins_arr, out=offsets[1:])
    else:
        centers = np.zeros(0, dtype=np.float64)
        peak_bins = np.zeros(0, dtype=np.int64)
        n_bins_arr = np.zeros(0, dtype=np.int32)
        saturated_arr = np.zeros(0, dtype=bool)
        flat = np.zeros(0, dtype=np.int64)
        offsets = np.zeros(1, dtype=np.int32)
    sg.create_dataset("center_freqs_mhz", data=centers)
    sg.create_dataset("peak_bin_indices", data=peak_bins)
    sg.create_dataset("n_bins", data=n_bins_arr)
    sg.create_dataset("saturated", data=saturated_arr)
    sg.create_dataset("bin_indices_flat", data=flat, compression="gzip")
    sg.create_dataset("offsets", data=offsets)
    sg.attrs["n_clusters"] = int(n_clusters)

    # --- frequency thirds ---------------------------------------------------
    fg = h5_group.create_group("frequency_thirds")
    for third in result.frequency_thirds:
        third_g = fg.create_group(third.label)
        third_g.attrs["label"] = third.label
        third_g.attrs["freq_lo_mhz"] = float(third.freq_lo_mhz)
        third_g.attrs["freq_hi_mhz"] = float(third.freq_hi_mhz)
        third_g.attrs["n"] = int(third.n)
        third_g.attrs["median_tau_us"] = float(third.median_tau_us)

    # --- band majorities (optional, per-band SNR-weighted majority) ---------
    # Empty when calibration was run without ``compute_band_majorities_flag``.
    # When present, Stage 5 may consume these as per-window tau anchors.
    bg = h5_group.create_group("band_majorities")
    bg.attrs["n_bands"] = int(len(result.band_majorities))
    for i, band in enumerate(result.band_majorities):
        band_g = bg.create_group(f"band_{i:02d}")
        band_g.attrs["label"] = band.label
        band_g.attrs["freq_lo_mhz"] = float(band.freq_lo_mhz)
        band_g.attrs["freq_hi_mhz"] = float(band.freq_hi_mhz)
        band_g.attrs["n"] = int(band.n)
        band_g.attrs["tau_maj_us"] = float(band.tau_maj_us)
        band_g.attrs["sigma_tau_us"] = float(band.sigma_tau_us)

    # --- algorithm info -----------------------------------------------------
    ai = h5_group.create_group("algorithm_info")
    ai.attrs["method"] = "sliding_active_window_stft"
    ai.attrs["version"] = SCHEMA_VERSION


def load_tau_calibration_from_hdf5(
    h5_group: h5py.Group,
) -> TauCalibrationResult:
    """Inverse of :func:`save_tau_calibration_to_hdf5`."""
    scalars = h5_group["scalars"]
    s_attrs = dict(scalars.attrs)

    # --- bimodality ---------------------------------------------------------
    bm_attrs = dict(h5_group["bimodality"].attrs)
    bm = GMMBimodality(
        n=int(bm_attrs["n"]),
        mu1=float(bm_attrs["mu1"]),
        sigma1=float(bm_attrs["sigma1"]),
        mu_a=float(bm_attrs["mu_a"]),
        sigma_a=float(bm_attrs["sigma_a"]),
        mu_b=float(bm_attrs["mu_b"]),
        sigma_b=float(bm_attrs["sigma_b"]),
        pi_a=float(bm_attrs["pi_a"]),
        aic1=float(bm_attrs["aic1"]),
        aic2=float(bm_attrs["aic2"]),
        delta_aic=float(bm_attrs["delta_aic"]),
        two_component_preferred=bool(bm_attrs["two_component_preferred"]),
        dominant_weight=float(bm_attrs["dominant_weight"]),
    )

    # --- pearson ------------------------------------------------------------
    p_attrs = dict(h5_group["pearson"].attrs)
    r_log_snr = float(p_attrs["r_log_snr_vs_tau"])
    r_freq = float(p_attrs["r_freq_vs_tau"])

    # --- contributors -------------------------------------------------------
    cg = h5_group["contributors"]
    contributor_bin_indices = np.asarray(cg["bin_indices"][:], dtype=np.int64)
    contributor_taus = np.asarray(cg["taus_us"][:], dtype=np.float64)
    contributor_snrs = np.asarray(cg["snrs"][:], dtype=np.float64)
    contributor_freqs = np.asarray(cg["freqs_mhz"][:], dtype=np.float64)

    # --- spur clusters ------------------------------------------------------
    sg = h5_group["spur_clusters"]
    centers = sg["center_freqs_mhz"][:]
    peak_bins = sg["peak_bin_indices"][:]
    n_bins_arr = sg["n_bins"][:]
    flat = sg["bin_indices_flat"][:]
    offsets = sg["offsets"][:]
    # ``saturated`` is absent on catalogs written before the Stage 5 spur
    # gate; default to all-False so legacy files load (the gate then falls
    # back to its frequency-domain narrowness detector for those bins).
    saturated_arr = (
        sg["saturated"][:]
        if "saturated" in sg
        else np.zeros(int(centers.size), dtype=bool)
    )
    clusters: list[SpurCluster] = []
    for i in range(int(centers.size)):
        lo = int(offsets[i])
        hi = int(offsets[i + 1])
        clusters.append(
            SpurCluster(
                center_freq_mhz=float(centers[i]),
                peak_bin_index=int(peak_bins[i]),
                n_bins=int(n_bins_arr[i]),
                bin_indices=tuple(int(b) for b in flat[lo:hi]),
                saturated=bool(saturated_arr[i]),
            )
        )

    # --- frequency thirds ---------------------------------------------------
    fg = h5_group["frequency_thirds"]
    thirds: list[FrequencyThird] = []
    for label in ("low", "mid", "high"):
        if label not in fg:
            continue
        third_attrs = dict(fg[label].attrs)
        thirds.append(
            FrequencyThird(
                label=(
                    str(third_attrs["label"])
                    if not isinstance(third_attrs["label"], bytes)
                    else third_attrs["label"].decode("utf-8")
                ),
                freq_lo_mhz=float(third_attrs["freq_lo_mhz"]),
                freq_hi_mhz=float(third_attrs["freq_hi_mhz"]),
                n=int(third_attrs["n"]),
                median_tau_us=float(third_attrs["median_tau_us"]),
            )
        )

    # --- band majorities (optional; missing on legacy files / opt-out runs) -
    bands: list[BandMajority] = []
    if "band_majorities" in h5_group:
        bg = h5_group["band_majorities"]
        for key in sorted(bg.keys()):
            band_attrs = dict(bg[key].attrs)
            label_raw = band_attrs["label"]
            bands.append(
                BandMajority(
                    label=(
                        label_raw.decode("utf-8")
                        if isinstance(label_raw, bytes)
                        else str(label_raw)
                    ),
                    freq_lo_mhz=float(band_attrs["freq_lo_mhz"]),
                    freq_hi_mhz=float(band_attrs["freq_hi_mhz"]),
                    n=int(band_attrs["n"]),
                    tau_maj_us=float(band_attrs["tau_maj_us"]),
                    sigma_tau_us=float(band_attrs["sigma_tau_us"]),
                )
            )

    # --- preconditions notes -----------------------------------------------
    notes_raw = h5_group["preconditions_notes"][:]
    notes = tuple(
        (n.decode("utf-8") if isinstance(n, bytes) else str(n)) for n in notes_raw
    )

    sideband_attr = s_attrs["sideband"]
    if isinstance(sideband_attr, bytes):
        sideband_attr = sideband_attr.decode("utf-8")

    return TauCalibrationResult(
        tau_maj_us=float(s_attrs["tau_maj_us"]),
        sigma_tau_us=float(s_attrs["sigma_tau_us"]),
        n_contributors=int(s_attrs["n_contributors"]),
        n_spur_bins=int(s_attrs["n_spur_bins"]),
        spur_clusters=tuple(clusters),
        bimodality=bm,
        pearson_r_log_snr_vs_tau=r_log_snr,
        pearson_r_freq_vs_tau=r_freq,
        frequency_thirds=tuple(thirds),
        band_majorities=tuple(bands),
        contributor_bin_indices=contributor_bin_indices,
        contributor_taus_us=contributor_taus,
        contributor_snrs=contributor_snrs,
        contributor_freqs_mhz=contributor_freqs,
        n_seg=int(s_attrs["n_seg"]),
        t_sigma=float(s_attrs["t_sigma"]),
        tau_max_us=float(s_attrs["tau_max_us"]),
        rss_gate_factor=float(s_attrs["rss_gate_factor"]),
        sample_dt_us=float(s_attrs["sample_dt_us"]),
        start_us=float(s_attrs["start_us"]),
        end_us=float(s_attrs["end_us"]),
        probe_freq_mhz=float(s_attrs["probe_freq_mhz"]),
        sideband=str(sideband_attr),
        trim_lo_mhz=float(s_attrs["trim_lo_mhz"]),
        trim_hi_mhz=float(s_attrs["trim_hi_mhz"]),
        sigma_x_full=float(s_attrs["sigma_x_full"]),
        sigma_frame=float(s_attrs["sigma_frame"]),
        snr_weighted=bool(s_attrs["snr_weighted"]),
        preconditions_passed=bool(s_attrs["preconditions_passed"]),
        preconditions_notes=notes,
    )


# ---------------------------------------------------------------------------
# Read-only bulk column access (no TauCalibrationResult reconstruction)
# ---------------------------------------------------------------------------
#
# Unlike the Stage 4/5 readers, these are *not* here for speed: Stage 2b
# persists one group, not one group per window, so the full loader already
# costs a few milliseconds. They exist so every persisted table is reachable
# through the same read-only surface -- a consumer should not have to know
# which stages happen to have a cheap loader, and a CSV of the per-band decay
# times is worth having whether or not deserializing was ever the bottleneck.
#
# Scope: the calibrated decay times (global and per band) and the raw per-bin
# contributors. The bimodality fit, the Pearson correlations, the frequency
# thirds and the spur clusters are diagnostics for the calibration algorithm
# itself; ``load_tau_calibration`` serves those, and cheaply.

#: Per-band read columns, in canonical order. All are attributes of a
#: ``band_majorities/band_NN`` subgroup. Empty when the calibration ran without
#: band majorities, in which case Stage 5 falls back to the global ``tau_maj_us``.
TAU_BAND_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "label": ("str", REQUIRED),
    "freq_lo_mhz": ("f8", REQUIRED),
    "freq_hi_mhz": ("f8", REQUIRED),
    "n_contributors": ("i8", REQUIRED),
    "tau_maj_us": ("f8", REQUIRED),
    "sigma_tau_us": ("f8", REQUIRED),
}

#: Per-third read columns: the same shape as a band, from the fixed low/mid/high
#: split the bimodality diagnostic uses. A third reports a median rather than an
#: SNR-weighted majority, and carries no spread -- it answers "does tau drift
#: with frequency", not "what tau should this window use".
TAU_THIRD_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "label": ("str", REQUIRED),
    "freq_lo_mhz": ("f8", REQUIRED),
    "freq_hi_mhz": ("f8", REQUIRED),
    "n_contributors": ("i8", REQUIRED),
    "median_tau_us": ("f8", REQUIRED),
}

#: Per-contributor read columns, in canonical order -- one row per STFT bin that
#: survived the gates and voted on the decay time.
TAU_CONTRIBUTOR_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "bin_index": ("i8", REQUIRED),
    "freq_mhz": ("f8", REQUIRED),
    "tau_us": ("f8", REQUIRED),
    "snr": ("f8", REQUIRED),
}

#: Per-spur-cluster read columns: one row per contiguous run of bins the spur
#: gate excluded. The member bin indices of each cluster are stored ragged (a
#: CSR pair) and so are not a column here; the full loader returns them.
TAU_SPUR_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "center_freq_mhz": ("f8", REQUIRED),
    "peak_bin_index": ("i8", REQUIRED),
    "n_bins": ("i8", REQUIRED),
    "saturated": ("bool", False),
}

#: On-disk dataset name for each read column whose stored name is plural.
#: The read surface uses singular column names -- a CSV header names one row's
#: field -- while the file keeps the names its writer chose.
_TAU_CONTRIBUTOR_DATASETS = {
    "bin_index": "bin_indices",
    "freq_mhz": "freqs_mhz",
    "tau_us": "taus_us",
    "snr": "snrs",
}
_TAU_SPUR_DATASETS = {
    "center_freq_mhz": "center_freqs_mhz",
    "peak_bin_index": "peak_bin_indices",
}
#: ``BandMajority.n`` / ``FrequencyThird.n`` are stored as a bare ``n``; a CSV
#: header of ``n`` says nothing, so the read column names what it counts.
_TAU_BAND_ATTRS = {"n_contributors": "n"}

#: Scalars lifted from the ``scalars`` subgroup, in reporting order.
_TAU_SCALAR_ATTRS = (
    "tau_maj_us",
    "sigma_tau_us",
    "n_contributors",
    "n_spur_bins",
    "n_seg",
    "preconditions_passed",
    "snr_weighted",
    "start_us",
    "end_us",
    "sample_dt_us",
    "tau_max_us",
    "t_sigma",
    "rss_gate_factor",
    "sigma_x_full",
    "sigma_frame",
    "trim_lo_mhz",
    "trim_hi_mhz",
    "probe_freq_mhz",
    "sideband",
)

#: Fields of the two-component GMM fit that decides whether the decay-time
#: distribution is bimodal -- the main reason a calibration fails preconditions.
_TAU_BIMODALITY_ATTRS = (
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
)


def _read_subgroup_table(
    h5_group: h5py.Group,
    subgroup: str,
    specs: Dict[str, ColumnSpec],
    columns: Optional[Sequence[str]],
    *,
    table: str,
    attr_names: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Read one row per child of *subgroup*, from each child's attributes.

    Shared by the band-majority and frequency-third tables, which have the same
    shape on disk: a subgroup per frequency slice, each carrying its bounds and
    statistics as attributes. Rows come back ordered by ``freq_lo_mhz``.
    """
    requested = resolve_column_selection(columns, list(specs), table=table)
    if subgroup not in h5_group:
        raise ValueError(f"{h5_group.name} missing required {subgroup!r} subgroup")
    children = h5_group[subgroup]

    # freq_lo_mhz always read: it defines the row order.
    to_read = list(dict.fromkeys(["freq_lo_mhz", *requested]))
    rows: Dict[str, List[Any]] = {c: [] for c in to_read}
    for name in sorted(children.keys()):
        child = children[name]
        where = f"{subgroup}/{name}"
        for col in to_read:
            rows[col].append(
                read_attr_value(
                    child, attr_names.get(col, col), specs[col], where=where
                )
            )

    built = build_columns(rows, specs, to_read)
    order = np.argsort(built["freq_lo_mhz"], kind="stable")
    return {c: built[c][order] for c in requested}


def _read_dataset_table(
    h5_group: h5py.Group,
    subgroup: str,
    specs: Dict[str, ColumnSpec],
    columns: Optional[Sequence[str]],
    *,
    table: str,
    dataset_names: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Read parallel-array columns out of *subgroup*, in on-disk row order."""
    requested = resolve_column_selection(columns, list(specs), table=table)
    if subgroup not in h5_group:
        raise ValueError(f"{h5_group.name} missing required {subgroup!r} subgroup")
    source = h5_group[subgroup]

    n: Optional[int] = None
    out: Dict[str, np.ndarray] = {}
    for col in requested:
        column = read_dataset_column(
            source,
            col,
            specs[col],
            n,
            where=f"{h5_group.name} {subgroup}",
            dataset=dataset_names.get(col, col),
        )
        if n is None:
            n = len(column)
        out[col] = column
    return out


def read_tau_band_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the per-band decay times from a tau-calibration group in bulk.

    Rows are ordered by ascending ``freq_lo_mhz`` (the bands partition the
    analysis band, so that is also their on-disk ``band_NN`` order). The table
    is empty when the calibration ran without band majorities, in which case
    Stage 5 falls back to the global ``tau_maj_us``.

    See :data:`TAU_BAND_COLUMN_SPECS` for the available columns.
    """
    return _read_subgroup_table(
        h5_group,
        "band_majorities",
        TAU_BAND_COLUMN_SPECS,
        columns,
        table="tau_bands",
        attr_names=_TAU_BAND_ATTRS,
    )


def read_tau_third_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the low/mid/high frequency thirds from a tau-calibration group.

    Rows are ordered by ascending ``freq_lo_mhz``. See
    :data:`TAU_THIRD_COLUMN_SPECS` for the available columns.
    """
    return _read_subgroup_table(
        h5_group,
        "frequency_thirds",
        TAU_THIRD_COLUMN_SPECS,
        columns,
        table="tau_thirds",
        attr_names=_TAU_BAND_ATTRS,
    )


def read_tau_contributor_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the per-bin tau contributors from a tau-calibration group in bulk.

    Rows keep their on-disk order (ascending STFT bin index), matching the
    ``contributor_*`` arrays :func:`load_tau_calibration_from_hdf5` returns.

    See :data:`TAU_CONTRIBUTOR_COLUMN_SPECS` for the available columns.
    """
    return _read_dataset_table(
        h5_group,
        "contributors",
        TAU_CONTRIBUTOR_COLUMN_SPECS,
        columns,
        table="tau_contributors",
        dataset_names=_TAU_CONTRIBUTOR_DATASETS,
    )


def read_tau_spur_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the excluded spur clusters from a tau-calibration group in bulk.

    Rows keep their on-disk order, matching
    :attr:`TauCalibrationResult.spur_clusters`. See
    :data:`TAU_SPUR_COLUMN_SPECS` for the available columns.
    """
    return _read_dataset_table(
        h5_group,
        "spur_clusters",
        TAU_SPUR_COLUMN_SPECS,
        columns,
        table="tau_spurs",
        dataset_names=_TAU_SPUR_DATASETS,
    )


def _plain(value: Any) -> Any:
    """An HDF5 attribute as a plain Python scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _copy_attrs(
    group: Optional[h5py.Group],
    names: Sequence[str],
    out: Dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    """Copy the named attributes of *group* into *out*, as plain scalars."""
    if group is None:
        return
    for name in names:
        if name in group.attrs:
            out[f"{prefix}{name}"] = _plain(group.attrs[name])


def read_tau_scalars(h5_group: h5py.Group) -> Dict[str, Any]:
    """Read the cheap calibration-level scalars from a tau-calibration group.

    Group attributes only -- the contributor, band, third and spur arrays are
    not touched. Everything the calibration persists as a scalar is here:

    * the fitted decay time and its spread, the gate knobs that produced them,
      and the analyzed window;
    * ``preconditions_passed`` with its three ``preconditions_notes``, which say
      *why* a calibration is or is not trustworthy;
    * ``bimodality_*``, the two-component GMM fit that is the usual reason a
      precondition fails, and the ``pearson_*`` correlations of decay time
      against SNR and frequency;
    * the line-shape vote: ``shape`` is what the file is set to use,
      ``recommended_shape`` is what the vote favored, and the ``shape_vote_*``
      rates are the vote itself -- the three coexist because a curator may keep
      a shape the vote did not pick.
    """
    out: Dict[str, Any] = {}
    _copy_attrs(h5_group.get("scalars"), _TAU_SCALAR_ATTRS, out)

    bands = h5_group.get("band_majorities")
    if bands is not None:
        out["n_bands"] = int(bands.attrs.get("n_bands", 0))
    spurs = h5_group.get("spur_clusters")
    if spurs is not None:
        out["n_spur_clusters"] = int(spurs.attrs.get("n_clusters", 0))

    notes = h5_group.get("preconditions_notes")
    if notes is not None:
        out["preconditions_notes"] = [
            _plain(note) for note in np.asarray(notes[:]).tolist()
        ]

    _copy_attrs(
        h5_group.get("bimodality"), _TAU_BIMODALITY_ATTRS, out, prefix="bimodality_"
    )
    _copy_attrs(
        h5_group.get("pearson"),
        ("r_log_snr_vs_tau", "r_freq_vs_tau"),
        out,
        prefix="pearson_",
    )
    _copy_attrs(h5_group.get("algorithm_info"), ("method", "version"), out)

    _copy_attrs(h5_group, ("shape", "recommended_shape", "creation_time"), out)
    votes = load_json_attr(h5_group, "shape_vote_rates", {}, label="stage2b")
    if isinstance(votes, dict):
        for kind, rate in votes.items():
            out[f"shape_vote_{kind}"] = float(rate)
    return out
