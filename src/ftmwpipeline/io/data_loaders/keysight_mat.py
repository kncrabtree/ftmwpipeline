"""Keysight MATLAB v7.3 scope-record loader.

Reads single-channel scope acquisitions saved as MATLAB v7.3 files
(which are HDF5 files) from Keysight oscilloscopes.  The loader extracts:

* ``Channel_N/Data`` — raw int16 samples (shape ``(1, M)`` or ``(M,)``);
* ``Channel_N/XInc`` / ``Channel_N/XOrg`` — sample clock (seconds);
* ``Channel_N/YInc`` / ``Channel_N/YOrg`` — vertical scaling
  (``volts = raw * YInc + YOrg``);
* ``Frame/Model`` / ``Frame/Serial`` — instrument identity strings
  (uint16-encoded, null-terminated).

The layout parameters (``pre_record_us``, ``frame_period_us``,
``n_frames``) are operator-supplied rather than inferred from the file
because the hardware does not embed segmentation metadata.

The instrument is a direct-sampler (no upconversion), so the baseband
frequency IS the molecular frequency.  The FID is stored with
``probe_freq_mhz = 0`` and ``sideband = upper`` so that
``FID.apply_molecular_frequency(f_bb) = 0 + f_bb = f_bb`` returns the
molecular frequency directly.

Interleave-offset cleanup
-------------------------
When ``interleave_factors`` is given (e.g. ``[16, 512]``), the loader
applies sequential per-phase DC subtraction before slicing/averaging.
For each factor M the per-phase means are estimated on the quiet
pre-record (residual after all preceding factors) and tiled over the full
record.  The estimated patterns are persisted in the ``acquisition_segments``
HDF5 group for audit.  The interleave clock frequencies (``fs / M``) are
injected into ``fid.metadata["clock_sources"]`` as reference-locked entries
so that the Stage 5 spur gate can account for them.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseLoader, LoaderError

if TYPE_CHECKING:
    from ...core.data_structures import FID


def _decode_uint16_string(arr: np.ndarray) -> str:
    """Decode a uint16-encoded, null-terminated string from a MATLAB v7.3 file.

    The array may be shaped ``(N, 1)`` or ``(N,)``; null characters are
    stripped.
    """
    flat = arr.ravel().tolist()
    return "".join(chr(c) for c in flat if c != 0)


class KeysightMatLoader(BaseLoader):
    """Loader for Keysight MATLAB v7.3 scope-record files.

    MATLAB v7.3 files are HDF5 containers.  This loader identifies them
    by the ``.mat`` extension *and* an h5py-openable signature *and* the
    presence of a ``Channel_*/XInc`` dataset — preventing confusion with
    plain HDF5 FTMW files that the generic ``HDF5Loader`` handles.

    Loading Parameters
    ------------------
    pre_record_us : float
        Duration of the quiet pre-record (µs).  Required.
    frame_period_us : float
        Frame repetition period (µs).  Required.
    n_frames : int
        Number of frames in the record.  Required.
    channel : str, optional
        Channel group name (e.g. ``"Channel_3"``).  Defaults to the sole
        channel present; raises if multiple channels are present and no
        selection is made.
    frame : int, optional
        Single-frame index (0-based) to use as the science FID instead of
        the coherent average.  Defaults to ``None`` (coherent average).
    keep_frames : bool, optional
        Persist the per-frame array alongside the FID.  Default ``False``.
    interleave_factors : list of int, optional
        ADC interleave factors to apply sequentially for offset cleanup
        (e.g. ``[16, 512]``).  Each factor M causes the per-phase means to
        be estimated on the quiet pre-record (residual after prior factors)
        and subtracted from the full record before slicing.  Default
        ``None`` (no cleanup).
    """

    format_name = "keysight-mat"
    file_extensions = [".mat"]
    directory_indicators: List[str] = []

    def can_load(self, source_path: Union[str, Path]) -> bool:
        """Return ``True`` if the file is a Keysight MATLAB v7.3 scope record.

        The checks are ordered from cheapest to most expensive:
        1. ``.mat`` extension;
        2. h5py-openable (rules out MATLAB v5 files);
        3. a ``Channel_*/XInc`` dataset present (rules out plain HDF5 FTMW
           files with the ``fid_data`` or ``voltage_data`` group that the
           generic ``HDF5Loader`` handles).
        """
        source_path = Path(source_path)
        if not source_path.is_file():
            return False
        if source_path.suffix.lower() != ".mat":
            return False

        try:
            import h5py

            with h5py.File(source_path, "r") as h5f:
                return self._find_channel_group(h5f) is not None
        except Exception:
            return False

    def validate_source(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": False,
            "metadata": {},
            "options": {},
            "errors": [],
        }
        source_path = Path(source_path)

        if not self.can_load(source_path):
            result["errors"].append("Not a Keysight MATLAB v7.3 scope file")
            return result

        try:
            import h5py

            with h5py.File(source_path, "r") as h5f:
                channel_names = self._list_channel_groups(h5f)
                channel_name = self._find_channel_group(h5f)
                assert channel_name is not None
                ch = h5f[channel_name]
                xinc = float(ch["XInc"][...].ravel()[0])
                n_samples = int(ch["Data"].shape[-1])
                model = ""
                serial = ""
                if "Frame/Model" in h5f:
                    model = _decode_uint16_string(h5f["Frame/Model"][...])
                if "Frame/Serial" in h5f:
                    serial = _decode_uint16_string(h5f["Frame/Serial"][...])

                result["metadata"]["sample_rate_gsa_s"] = round(1.0 / xinc / 1e9, 6)
                result["metadata"]["n_samples"] = n_samples
                result["metadata"]["duration_us"] = n_samples * xinc * 1e6
                result["metadata"]["model"] = model
                result["metadata"]["serial"] = serial
                result["options"]["available_channels"] = channel_names
                result["valid"] = True
        except Exception as exc:
            result["errors"].append(f"Validation failed: {exc}")

        return result

    def load_fid(self, source_path: Union[str, Path], **kwargs: Any) -> "FID":
        """Load the science FID from a segmented scope record.

        Required keyword arguments
        --------------------------
        pre_record_us : float
        frame_period_us : float
        n_frames : int

        Optional keyword arguments
        --------------------------
        channel : str
        frame : int or None
        keep_frames : bool
        interleave_factors : list of int or None

        Returns
        -------
        FID
            Science FID (averaged or single-frame).  The FID carries the
            :class:`~ftmwpipeline.io.acquisition_layout.AcquisitionLayout`
            as ``fid.metadata["acquisition_layout"]`` for serialization.
        """
        from ...core.data_structures import FID, FIDProcessingParameters, Sideband
        from ..acquisition_layout import (
            AcquisitionLayout,
            apply_interleave_cleanup,
            slice_record,
        )

        params = self.validate_parameters(**kwargs)["parameters"]

        pre_record_us: float = params["pre_record_us"]
        frame_period_us: float = params["frame_period_us"]
        n_frames: int = int(params["n_frames"])
        channel_hint: Optional[str] = params.get("channel")
        frame_sel: Optional[int] = params.get("frame")
        keep_frames: bool = bool(params.get("keep_frames", False))
        interleave_factors: Optional[List[int]] = params.get("interleave_factors")

        source_path = Path(source_path)

        try:
            import h5py
        except ImportError as exc:
            raise LoaderError("h5py is required to load Keysight MAT files") from exc

        try:
            with h5py.File(source_path, "r") as h5f:
                # Resolve channel
                channel_names = self._list_channel_groups(h5f)
                if not channel_names:
                    raise LoaderError(f"No Channel_* group found in {source_path.name}")

                if channel_hint is not None:
                    if channel_hint not in h5f:
                        raise LoaderError(
                            f"Channel '{channel_hint}' not found in "
                            f"{source_path.name}. Available: {channel_names}"
                        )
                    channel_name = channel_hint
                elif len(channel_names) == 1:
                    channel_name = channel_names[0]
                else:
                    raise LoaderError(
                        f"Multiple channels present in {source_path.name}: "
                        f"{channel_names}. Specify --channel / channel= "
                        f"to select one."
                    )

                ch = h5f[channel_name]
                xinc = float(ch["XInc"][...].ravel()[0])
                yinc = float(ch["YInc"][...].ravel()[0])
                yorg = float(ch["YOrg"][...].ravel()[0])

                raw_data = np.asarray(ch["Data"][...], dtype=np.float64).ravel()

                # Instrument identity for provenance
                model = ""
                serial = ""
                if "Frame/Model" in h5f:
                    model = _decode_uint16_string(h5f["Frame/Model"][...])
                if "Frame/Serial" in h5f:
                    serial = _decode_uint16_string(h5f["Frame/Serial"][...])

        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to read {source_path.name}: {exc}") from exc

        # Interleave-offset cleanup on the raw LSB record before voltage
        # scaling.  Estimation and subtraction in sample (LSB) units keeps
        # the arithmetic exact and independent of the vertical calibration.
        interleave_patterns: Optional[Dict[int, np.ndarray]] = None
        if interleave_factors:
            pre_samples = int(round(pre_record_us * 1e-6 / xinc))
            quiet_raw = raw_data[:pre_samples]
            try:
                raw_data, interleave_patterns = apply_interleave_cleanup(
                    raw_data, quiet_raw, interleave_factors
                )
            except ValueError as exc:
                raise LoaderError(f"Interleave cleanup failed: {exc}") from exc

        # Scale to volts
        voltage_data = raw_data * yinc + yorg

        # Build the acquisition layout and slice the record
        layout = AcquisitionLayout(
            pre_record_us=pre_record_us,
            frame_period_us=frame_period_us,
            n_frames=n_frames,
            frame=frame_sel,
            keep_frames=keep_frames,
        )

        try:
            sliced = slice_record(voltage_data, layout, sample_dt=xinc)
        except ValueError as exc:
            raise LoaderError(str(exc)) from exc

        source_meta = self._create_source_metadata(
            source_path,
            channel=channel_name,
            pre_record_us=pre_record_us,
            frame_period_us=frame_period_us,
            n_frames=n_frames,
            frame=frame_sel,
            keep_frames=keep_frames,
            interleave_factors=interleave_factors,
        )
        source_meta["instrument_model"] = model
        source_meta["instrument_serial"] = serial
        source_meta["xinc_s"] = xinc
        source_meta["yinc_v"] = yinc
        source_meta["yorg_v"] = yorg
        # Carry the sliced segments so fid_serialization.py can persist them.
        # The segments are the post-cleanup versions: interleave subtraction
        # runs before slice_record, so sliced.pre_record / .tail / .frames
        # already have the offsets removed.
        source_meta["acquisition_layout"] = {
            "pre_record_us": pre_record_us,
            "frame_period_us": frame_period_us,
            "n_frames": n_frames,
            "frame": frame_sel,
            "keep_frames": keep_frames,
        }
        source_meta["_sliced_pre_record"] = sliced.pre_record
        source_meta["_sliced_tail"] = sliced.tail
        if keep_frames:
            source_meta["_sliced_frames"] = sliced.frames
        # Private transport key for pattern persistence in acquisition_segments
        if interleave_patterns is not None:
            source_meta["_interleave_patterns"] = interleave_patterns

        # Inject interleave clock frequencies as recommended clock sources.
        # The sample rate is 1/xinc; each interleave factor M produces a
        # comb at fs/M.  On this instrument class the ADC is driven from
        # the Rb-locked reference, so these clocks are marked locked=True.
        if interleave_factors:
            fs_mhz = 1.0 / xinc / 1e6
            clock_sources = [
                {
                    "freq_mhz": round(fs_mhz / m, 6),
                    "locked": True,
                    "label": f"interleave_m{m}",
                }
                for m in interleave_factors
            ]
            source_meta["clock_sources"] = clock_sources

        # Direct sampling: probe = 0, upper sideband → f_mol = 0 + f_bb = f_bb
        return FID(
            data=sliced.science_fid,
            spacing=xinc,
            probe_freq_mhz=0.0,
            sideband=Sideband.UPPER,
            shots=n_frames if layout.frame is None else 1,
            processing=FIDProcessingParameters(),
            metadata=source_meta,
        )

    def get_required_parameters(self) -> List[str]:
        return ["pre_record_us", "frame_period_us", "n_frames"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "channel": None,
            "frame": None,
            "keep_frames": False,
            "interleave_factors": None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_channel_groups(h5f: Any) -> List[str]:
        """Return all ``Channel_*`` group names that have an ``XInc`` dataset."""
        names = []
        for key in h5f.keys():
            if key.startswith("Channel_"):
                grp = h5f[key]
                if "XInc" in grp:
                    names.append(key)
        return sorted(names)

    @staticmethod
    def _find_channel_group(h5f: Any) -> Optional[str]:
        """Return the first ``Channel_*`` group name with ``XInc``, or ``None``."""
        names = KeysightMatLoader._list_channel_groups(h5f)
        return names[0] if names else None
