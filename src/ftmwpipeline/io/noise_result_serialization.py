"""
NoiseResult serialization to HDF5 format.

This module provides serialization and deserialization of NoiseResult objects
to HDF5 format. The boolean noise_mask is stored compactly as signal indices
(its False positions), and the per-bin σ array is stored verbatim for an exact
round-trip.

Functions
---------
save_noise_result_to_hdf5 : Save NoiseResult to HDF5 group
load_noise_result_from_hdf5 : Load NoiseResult from HDF5 group
_extract_signal_indices : Convert boolean noise_mask to signal indices
_reconstruct_noise_mask : Reconstruct boolean noise_mask from signal indices
"""

from typing import Any, Dict, cast

import h5py
import numpy as np

from ..preprocessing.noise_estimation import NoiseResult


def save_noise_result_to_hdf5(
    noise_result: NoiseResult,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    h5_group: h5py.Group,
) -> None:
    """
    Save NoiseResult object to HDF5 group.

    The boolean noise_mask is stored compactly as signal indices (its False
    positions); the per-bin σ array is stored verbatim for an exact round-trip.

    HDF5 Structure
    --------------
    /noise_result/
    ├── signal_indices         [dataset: int32] -- mask False positions
    ├── rms_noise_full         [dataset: float64] -- per-bin σ, verbatim
    ├── bin_info/              [group: estimator diagnostics]
    │   ├── algorithm         [attr: str]
    │   ├── noise_fraction    [attr: float]
    │   └── ...               [estimator knob attrs]
    └── algorithm_info/        [group: method metadata]
        ├── method            [attr: str]
        └── version           [attr: str]

    Parameters
    ----------
    noise_result : NoiseResult
        NoiseResult object to serialize
    frequencies : np.ndarray
        Original frequency array (MHz)
    magnitudes : np.ndarray
        Original magnitude array
    h5_group : h5py.Group
        HDF5 group to write data to

    Raises
    ------
    ValueError
        If arrays have mismatched lengths or invalid noise_result
    RuntimeError
        If HDF5 write operation fails
    """
    try:
        # Validate inputs
        if len(frequencies) != len(magnitudes):
            raise ValueError("frequencies and magnitudes must have same length")

        if len(noise_result.noise_mask) != len(frequencies):
            raise ValueError("noise_mask length must match frequency array length")

        # Store signal indices (major storage optimization: ~375KB → ~150KB)
        signal_indices = _extract_signal_indices(noise_result.noise_mask)
        h5_group.create_dataset(
            "signal_indices",
            data=signal_indices,
            compression="gzip",
            compression_opts=6,
        )

        # Store the σ array verbatim for an exact round-trip. (The scatter
        # estimator's σ is not reproducible from a compact parameterization, so
        # it is stored in full -- a few hundred KB compressed.)
        h5_group.create_dataset(
            "rms_noise_full",
            data=np.asarray(noise_result.rms_noise, dtype=np.float64),
            compression="gzip",
            compression_opts=6,
        )

        # Store bin_info metadata
        if noise_result.bin_info:
            bin_group = h5_group.create_group("bin_info")

            for key, value in noise_result.bin_info.items():
                if isinstance(value, np.ndarray):
                    bin_group.create_dataset(key, data=value)
                else:
                    bin_group.attrs[key] = value

        # Store algorithm metadata
        algo_group = h5_group.create_group("algorithm_info")
        algo_group.attrs["method"] = "verbatim_sigma"
        algo_group.attrs["version"] = "2.0"
        algo_group.attrs["storage_optimization"] = "signal_indices_plus_verbatim_sigma"

    except Exception as e:
        raise RuntimeError(f"Failed to save NoiseResult to HDF5: {e}") from e


def load_noise_result_from_hdf5(
    h5_group: h5py.Group, frequencies: np.ndarray, magnitudes: np.ndarray
) -> NoiseResult:
    """
    Load NoiseResult object from HDF5 group with signal indices and convolution reconstruction.

    Reconstructs the boolean noise_mask from stored signal indices and
    reconstructs the RMS array using stored convolution parameters for
    exact reproduction of the original noise estimation.

    Parameters
    ----------
    h5_group : h5py.Group
        HDF5 group containing serialized NoiseResult data
    frequencies : np.ndarray
        Original frequency array (MHz) - needed for RMS reconstruction
    magnitudes : np.ndarray
        Original magnitude array - needed for RMS reconstruction

    Returns
    -------
    NoiseResult
        Reconstructed NoiseResult object with exact RMS values

    Raises
    ------
    KeyError
        If required data is missing from HDF5 group
    ValueError
        If reconstruction fails or data is invalid
    RuntimeError
        If HDF5 read operation fails
    """
    try:
        # Validate inputs
        if len(frequencies) != len(magnitudes):
            raise ValueError("frequencies and magnitudes must have same length")

        # Reconstruct noise_mask from signal indices
        signal_indices = h5_group["signal_indices"][:]
        noise_mask = _reconstruct_noise_mask(signal_indices, len(frequencies))

        # σ is stored verbatim (see save_noise_result_to_hdf5) -- read it back.
        rms_noise = h5_group["rms_noise_full"][:]
        bin_info = _load_bin_info(h5_group)

        return NoiseResult(
            rms_noise=rms_noise,
            noise_mask=noise_mask,
            bin_info=bin_info,
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load NoiseResult from HDF5: {e}") from e


def _load_bin_info(h5_group: h5py.Group) -> Dict[str, Any]:
    """Reconstruct the ``bin_info`` dict from a serialized NoiseResult group."""
    bin_info: Dict[str, Any] = {}
    if "bin_info" in h5_group:
        bin_group = h5_group["bin_info"]

        # Load datasets
        for key in bin_group.keys():
            bin_info[key] = bin_group[key][:]

        # Load attributes
        for key in bin_group.attrs.keys():
            value = bin_group.attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            bin_info[key] = value
    return bin_info


def _extract_signal_indices(noise_mask: np.ndarray) -> np.ndarray:
    """
    Convert boolean noise_mask to signal indices for storage optimization.

    Since signal points are typically ~10% of the data, storing their indices
    as int32 values uses ~60% less space than the full boolean mask.

    For typical FTMW data:
    - noise_mask: 375k bools × 1 byte = ~375KB
    - signal_indices: 37.5k int32 × 4 bytes = ~150KB (60% reduction)

    Parameters
    ----------
    noise_mask : np.ndarray
        Boolean array where True indicates noise points

    Returns
    -------
    np.ndarray
        Integer indices where noise_mask is False (signal points)
    """
    # Signal points are where noise_mask is False
    signal_indices = cast(np.ndarray, np.where(~noise_mask)[0].astype(np.int32))
    return signal_indices


def _reconstruct_noise_mask(
    signal_indices: np.ndarray, total_length: int
) -> np.ndarray:
    """
    Reconstruct boolean noise_mask from signal indices.

    Parameters
    ----------
    signal_indices : np.ndarray
        Integer indices of signal points (where original noise_mask was False)
    total_length : int
        Length of the original frequency/magnitude arrays

    Returns
    -------
    np.ndarray
        Reconstructed boolean noise_mask
    """
    noise_mask: np.ndarray = np.ones(total_length, dtype=bool)  # Start with all noise
    noise_mask[signal_indices] = False  # Mark signal points
    return noise_mask
