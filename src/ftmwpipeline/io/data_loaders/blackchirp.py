"""
Blackchirp data format loader.

This module implements the loader for Blackchirp experimental data format,
handling FID data extraction from Blackchirp directory structures with
proper metadata preservation.
"""

from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from .base import BaseLoader, LoaderError

if TYPE_CHECKING:
    from ...core.data_structures import FID, FIDProcessingParameters, Sideband


def _get_fid_classes() -> (
    Tuple[Type["FID"], Type["FIDProcessingParameters"], Type["Sideband"]]
):
    from ...core.data_structures import FID, FIDProcessingParameters, Sideband

    return FID, FIDProcessingParameters, Sideband


class BlackChirpLoader(BaseLoader):
    """
    Loader for Blackchirp experimental data format.

    Blackchirp stores FTMW experiments in directory structures containing
    FID data, processing parameters, and experimental metadata.
    """

    format_name = "blackchirp"
    file_extensions = []  # Blackchirp uses directories, not files
    directory_indicators = [
        "fid",
        "fidparams.csv",
    ]  # Files that indicate Blackchirp format

    def can_load(self, source_path: Union[str, Path]) -> bool:
        """
        Check if source is a Blackchirp experiment directory.

        Blackchirp experiments have the following structure:
        experiment_dir/
        ├── fid/
        │   ├── fidparams.csv
        │   ├── processing.csv (optional)
        │   ├── 0.csv
        │   ├── 1.csv (if multiple FIDs)
        │   └── ...
        """
        source_path = Path(source_path)

        if not source_path.exists():
            return False

        if not source_path.is_dir():
            return False

        # Check for FID directory
        fid_dir = source_path / "fid"
        if not fid_dir.exists() or not fid_dir.is_dir():
            return False

        # Check for fidparams.csv
        fidparams_file = fid_dir / "fidparams.csv"
        if not fidparams_file.exists():
            return False

        # Check for at least one FID data file
        fid_files = list(fid_dir.glob("*.csv"))
        fid_data_files = [f for f in fid_files if f.name.replace(".csv", "").isdigit()]

        return len(fid_data_files) > 0

    def validate_source(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Validate Blackchirp experiment directory.

        Returns information about available FIDs, processing parameters,
        and experimental metadata.
        """
        result: Dict[str, Any] = {
            "valid": False,
            "metadata": {},
            "options": {},
            "errors": [],
        }

        source_path = Path(source_path)

        try:
            # Check basic structure
            if not self.can_load(source_path):
                result["errors"].append("Not a valid Blackchirp experiment directory")
                return result

            fid_dir = source_path / "fid"

            # Load and validate FID parameters
            fidparams_file = fid_dir / "fidparams.csv"
            try:
                fidparams_df = pd.read_csv(fidparams_file, sep=";")
                result["metadata"]["n_fids"] = len(fidparams_df)
                result["options"]["available_fid_indices"] = list(
                    range(len(fidparams_df))
                )

                # Extract key parameters from first FID
                if len(fidparams_df) > 0:
                    first_fid = fidparams_df.iloc[0]
                    result["metadata"]["probe_freq_mhz"] = float(first_fid["probefreq"])
                    result["metadata"]["spacing_us"] = (
                        float(first_fid["spacing"]) * 1e6
                    )  # Convert to μs
                    result["metadata"]["sideband"] = first_fid["sideband"]
                    result["metadata"]["shots"] = int(first_fid["shots"])

            except Exception as e:
                result["errors"].append(f"Failed to read FID parameters: {e}")
                return result

            # Check FID data files
            missing_files = []
            for i in range(len(fidparams_df)):
                fid_file = fid_dir / f"{i}.csv"
                if not fid_file.exists():
                    missing_files.append(str(fid_file))

            if missing_files:
                result["errors"].append(f"Missing FID data files: {missing_files}")
                return result

            # Load processing parameters if available
            processing_file = fid_dir / "processing.csv"
            if processing_file.exists():
                try:
                    proc_df = pd.read_csv(processing_file, sep=";", index_col="ObjKey")
                    proc_dict = proc_df["Value"].to_dict()
                    result["metadata"]["processing_parameters"] = proc_dict
                except Exception as e:
                    result["errors"].append(
                        f"Warning: Could not read processing parameters: {e}"
                    )

            # Load experiment metadata
            metadata_files = {
                "header.csv": "header",
                "hardware.csv": "hardware",
                "version.csv": "version",
            }

            for filename, key in metadata_files.items():
                metadata_file = source_path / filename
                if metadata_file.exists():
                    try:
                        if key == "version":
                            with open(metadata_file, "r") as f:
                                result["metadata"][key] = f.read().strip()
                        else:
                            df = pd.read_csv(metadata_file, sep=";")
                            if key == "header":
                                result["metadata"][key] = (
                                    df.to_dict("records")[0] if len(df) > 0 else {}
                                )
                            else:
                                result["metadata"][key] = df.to_dict("records")
                    except Exception as e:
                        result["errors"].append(
                            f"Warning: Could not read {filename}: {e}"
                        )

            # Extract clock sources from clocks.csv + header.csv.
            # Failure is non-fatal: missing files or malformed rows yield no
            # clock_sources entry rather than a load failure.
            clock_sources = self._extract_clock_sources(source_path)
            if clock_sources is not None:
                result["metadata"]["clock_sources"] = clock_sources

            # Extract declared chirp-window timing.  Failure is non-fatal.
            chirp_window = self._extract_chirp_window(source_path)
            if chirp_window is not None:
                result["metadata"]["chirp_window"] = chirp_window

            result["valid"] = True
            return result

        except Exception as e:
            result["errors"].append(f"Validation failed: {e}")
            return result

    def load_fid(
        self, source_path: Union[str, Path], fid_index: int = 0, **kwargs: Any
    ) -> "FID":
        """
        Load FID data from Blackchirp experiment.

        Parameters
        ----------
        source_path : str or Path
            Path to Blackchirp experiment directory
        fid_index : int, optional
            Index of FID to load (default: 0)
        **kwargs
            Additional parameters (ignored for Blackchirp)

        Returns
        -------
        FID
            Loaded FID object with Blackchirp metadata

        Raises
        ------
        LoaderError
            If loading fails
        """
        try:
            # Get runtime imports
            FID, FIDProcessingParameters, Sideband = _get_fid_classes()

            source_path = Path(source_path)

            # Validate source first
            validation = self.validate_source(source_path)
            if not validation["valid"]:
                raise LoaderError(f"Invalid Blackchirp source: {validation['errors']}")

            # Delegate the FID payload (base-36 decode, voltage scaling,
            # spacing/probe/shots, frame handling) to the blackchirp module
            # rather than re-parsing the CSVs by hand.
            try:
                from blackchirp import BCFTMW
            except ImportError as e:  # pragma: no cover - dependency guard
                raise LoaderError(
                    "The 'blackchirp' package (>=0.1.0rc2) is required to load "
                    f"Blackchirp experiments: {e}"
                ) from e

            ftmw = BCFTMW(str(source_path), sep=self._read_separator(source_path))

            if fid_index >= ftmw.numfids:
                raise LoaderError(
                    f"FID index {fid_index} not found. "
                    f"Available indices: 0-{ftmw.numfids - 1}"
                )

            bcfid = ftmw.get_fid(fid_index)
            params = bcfid.fidparams
            # bcfid.data is (n_samples, n_frames); the pipeline analyzes the
            # primary (cumulative) frame, matching the historical behavior.
            voltage_data = np.asarray(bcfid.data)[:, 0]
            # Sideband is resolved locally rather than via
            # ``bcfid.is_lower_sideband()``: pandas reads an integer-coded
            # cell back as a numpy scalar (the fidparams row mixes int/float
            # columns, so the value is float64), and the module's enum
            # resolver only accepts ``str``/``int`` (kncrabtree/blackchirp#26).
            # Re-delegate once the module accepts numpy scalars.
            sideband = self._resolve_sideband(params["sideband"])

            # processing.csv is informational only here -- the FT is driven by
            # the resolved FTSettings, not these values -- so we carry the
            # stable scalars for display and deliberately ignore the
            # version-fragile window-function enum.
            processing_params = self._build_processing_parameters(ftmw.proc)

            # Create source metadata
            source_metadata = self._create_source_metadata(
                source_path,
                fid_index=fid_index,
                blackchirp_params=params.to_dict(),
            )
            source_metadata.update(validation["metadata"])

            return FID(
                data=voltage_data,
                spacing=float(params["spacing"]),  # seconds
                probe_freq_mhz=float(params["probefreq"]),  # MHz
                sideband=sideband,
                shots=int(bcfid.shots),
                processing=processing_params,
                metadata=source_metadata,
            )

        except LoaderError:
            raise
        except Exception as e:
            raise LoaderError(f"Failed to load Blackchirp FID: {e}") from e

    @staticmethod
    def _extract_clock_sources(
        source_path: Path,
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract instrument clock declarations from a Blackchirp experiment.

        Delegates metadata parsing to the ``blackchirp`` package
        (:class:`~blackchirp.BCExperiment`), which normalizes the CSV format
        across Blackchirp versions, and applies the instrument semantics on top:
        the synthesiser chain fundamentals, the AWG sample clock, and the
        free-running digitizer clock. Returns a list of
        ``{"freq_mhz": float, "locked": bool, "label": str}`` dicts (the
        serialized form of a :class:`~...ClockSource`) consumed by the Stage 5
        spur gate and the timebase self-calibration, or ``None`` when nothing is
        parseable. The caller stores it in ``result["metadata"]["clock_sources"]``.

        Clock-source rules
        ------------------
        * ``clocks`` rows -> synthesiser chain fundamental = FreqMHz / Factor for
          a Multiply operation, FreqMHz * Factor for Divide, pass-through
          otherwise. ``Operation`` is accepted as the string form
          (``"Multiply"`` / ``"Divide"``) **and** the integer enum (``0`` / ``1``)
          older Blackchirp metadata writes. Fundamentals are deduplicated on the
          rounded MHz value (a dual-output synthesiser driving two chains yields
          two rows at one fundamental). All synthesiser sources are locked
          (referenced to the instrument Rb standard).
        * AWG ``ChirpConfig`` SampleRate -> AWG entry, locked.
        * The ``FtmwDigitizer`` sample rate -> digitizer entry, locked=False (the
          scope oscillator free-runs; the metadata carries no locked flag, so the
          conservative default is unlocked). The hardware key is discovered from
          the ``hardware`` table, so both the suffixed (``FtmwDigitizer.0``) and
          bare (``FtmwDigitizer``) conventions resolve.

        Tolerant: a missing package, unreadable experiment, missing column, or
        bad value yields fewer (or no) entries rather than a load failure.
        """
        try:
            from blackchirp import BCExperiment
        except ImportError:  # pragma: no cover - dependency guard
            return None
        try:
            exp = BCExperiment(str(source_path))
        except Exception:
            return None

        entries: List[Dict[str, Any]] = []

        # --- synthesiser chain fundamentals (clocks table) ---
        clocks = getattr(exp, "clocks", None)
        if (
            clocks is not None
            and not clocks.empty
            and {"FreqMHz", "Operation", "Factor"}.issubset(clocks.columns)
        ):
            seen_fundamentals: set = set()
            for _, row in clocks.iterrows():
                try:
                    freq_mhz = float(row["FreqMHz"])
                    factor = float(row["Factor"])
                    operation = BlackChirpLoader._clock_operation(row["Operation"])
                    if operation == "Multiply":
                        fundamental = freq_mhz / factor
                    elif operation == "Divide":
                        fundamental = freq_mhz * factor
                    else:
                        fundamental = freq_mhz
                    key = round(fundamental, 3)
                    if key in seen_fundamentals:
                        continue
                    seen_fundamentals.add(key)
                    clock_type = str(row.get("ClockType", "")).strip()
                    hw_key = str(row.get("HwKey", "")).strip()
                    out_num = row.get("OutputNum", "")
                    entries.append(
                        {
                            "freq_mhz": fundamental,
                            "locked": True,
                            "label": f"{clock_type} ({hw_key}:{out_num})".strip(),
                        }
                    )
                except (ValueError, TypeError):
                    continue

        # --- AWG sample clock (ChirpConfig / SampleRate, MHz) ---
        awg_mhz = BlackChirpLoader._header_rate_mhz(exp, "ChirpConfig", "SampleRate")
        if awg_mhz is not None:
            entries.append({"freq_mhz": awg_mhz, "locked": True, "label": "awg"})

        # --- digitizer sample clock: the one free-running (unlocked) source ---
        dig_key = BlackChirpLoader._digitizer_hw_key(exp)
        if dig_key is not None:
            dig_mhz = BlackChirpLoader._header_rate_mhz(exp, dig_key, "SampleRate")
            if dig_mhz is not None:
                entries.append(
                    {"freq_mhz": dig_mhz, "locked": False, "label": "digitizer"}
                )

        return entries if entries else None

    @staticmethod
    def _clock_operation(value: Any) -> Optional[str]:
        """Normalize a clocks ``Operation`` to ``"Multiply"`` / ``"Divide"``.

        Accepts the string serialization and the integer enum older Blackchirp
        metadata writes (``0`` = Multiply, ``1`` = Divide); anything else is
        ``None`` (pass-through fundamental).
        """
        s = str(value).strip()
        if s in ("Multiply", "Divide"):
            return s
        try:
            return {0: "Multiply", 1: "Divide"}.get(int(float(s)))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _digitizer_hw_key(exp: Any) -> Optional[str]:
        """The FTMW digitizer's hardware key from the ``hardware`` table.

        Matches any key beginning ``FtmwDigitizer`` so both the suffixed
        (``FtmwDigitizer.0``) and bare (``FtmwDigitizer``) conventions resolve;
        ``None`` when the table is absent or carries no digitizer.
        """
        hardware = getattr(exp, "hardware", None)
        if hardware is None or hardware.empty or "key" not in hardware.columns:
            return None
        for key in hardware["key"]:
            if str(key).startswith("FtmwDigitizer"):
                return str(key)
        return None

    @staticmethod
    def _header_rate_mhz(exp: Any, obj_key: str, value_key: str) -> Optional[float]:
        """A header sample-rate value converted to MHz via its declared unit.

        Uses the unit-aware :meth:`BCExperiment.header_value` /
        :meth:`~BCExperiment.header_unit`; a missing entry, bad number, or
        unreadable header yields ``None``. An absent unit defaults to MHz (the
        AWG convention); Hz / kHz / GHz are converted.
        """
        try:
            raw = exp.header_value(obj_key, value_key)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            value = float(raw)
        except (ValueError, TypeError):
            return None
        try:
            unit = exp.header_unit(obj_key, value_key)
        except Exception:
            unit = None
        u = str(unit).strip().lower() if unit is not None else ""
        scale = {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3, "": 1.0}.get(u)
        return value * scale if scale is not None else value

    @staticmethod
    def _extract_chirp_window(
        source_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """Extract declared chirp-window timing from Blackchirp metadata.

        Reads the chirp duration from ``chirps.csv`` (summing
        ``DurationUs`` for the first chirp waveform index, i.e.
        ``Chirp == 0``) and the pre-chirp hardware delay from
        ``header.csv`` (``ChirpConfig / PreGate`` +
        ``ChirpConfig / PreProtection``).

        Returns a dict with keys ``chirp_end_us`` and optionally
        ``chirp_start_us``, or ``None`` when the data cannot be parsed.
        The dict is the transport form passed through ``fid.metadata``; it
        is coerced to :class:`~ftmwpipeline.core.data_structures.ChirpWindow`
        by Stage 0 on persistence.

        Derivation
        ----------
        * ``chirp_start_us = PreGate + PreProtection`` — the gate delay and
          protection interval before the AWG output begins.  Both fields are
          in µs and are recorded by Blackchirp for every experiment.
        * ``chirp_duration_us = sum(DurationUs for Chirp==0 rows)`` — the
          total waveform length for the first chirp waveform (each row is one
          segment; single-segment experiments have one row).
        * ``chirp_end_us = chirp_start_us + chirp_duration_us``.

        For the checked-in example (exp 2638):
          PreGate=0.5 µs, PreProtection=0.1 µs, DurationUs=1.0 µs →
          chirp_start_us=0.60 µs, chirp_end_us=1.60 µs.
        """
        try:
            exp = __import__("blackchirp").BCExperiment(str(source_path))
        except Exception:
            return None

        try:
            # Pre-chirp delay: hardware gate + protection before AWG fires.
            pre_gate = float(exp.header_value("ChirpConfig", "PreGate"))
            pre_prot = float(exp.header_value("ChirpConfig", "PreProtection"))
            chirp_start_us = pre_gate + pre_prot
        except Exception:
            return None  # cannot determine start; skip rather than guess

        try:
            chirps_df = exp.chirps
            first_waveform = chirps_df[chirps_df["Chirp"] == 0]
            if first_waveform.empty:
                return None
            chirp_duration_us = float(first_waveform["DurationUs"].sum())
        except Exception:
            return None

        chirp_end_us = chirp_start_us + chirp_duration_us
        return {
            "chirp_start_us": chirp_start_us,
            "chirp_end_us": chirp_end_us,
        }

    def get_required_parameters(self) -> List[str]:
        """Blackchirp loader has no required parameters."""
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Get optional parameters for Blackchirp loading."""
        return {"fid_index": 0}  # Which FID to load if multiple are available

    @staticmethod
    def _resolve_sideband(value: Any) -> "Sideband":
        """Resolve a Blackchirp ``sideband`` cell to a :class:`Sideband`.

        Handles every on-disk / parsed form: the canonical Q_ENUM names
        (``"LowerSideband"`` / ``"UpperSideband"``), the underlying integer
        code (``1`` = lower, ``0`` = upper), and the numpy scalar pandas
        produces for the integer code. The blackchirp module owns the
        name<->int mapping for its own ``ft`` path, but its resolver rejects
        numpy scalars, so the loader keeps this thin local resolver.
        """
        _, _, Sideband = _get_fid_classes()

        if isinstance(value, np.number):
            value = value.item()
        if not isinstance(value, str):
            return Sideband.LOWER if int(value) == 1 else Sideband.UPPER

        s = value.strip()
        try:
            return Sideband.LOWER if int(s) == 1 else Sideband.UPPER
        except ValueError:
            pass
        low = s.lower()
        if "lower" in low:
            return Sideband.LOWER
        if "upper" in low:
            return Sideband.UPPER
        raise LoaderError(f"Unrecognised Blackchirp sideband value: {value!r}")

    @staticmethod
    def _read_separator(source_path: Path) -> str:
        """Return the CSV delimiter Blackchirp records on the first line of
        ``version.csv`` (``";"`` in practice). Falls back to ``";"`` when the
        file is absent or its first line is not a bare delimiter.
        """
        version_file = source_path / "version.csv"
        if version_file.exists():
            try:
                first = version_file.read_text().splitlines()[0].strip()
            except (OSError, IndexError):
                first = ""
            # A real delimiter line is a single non-alphanumeric character;
            # anything else (e.g. a stray version string) is not a separator.
            if len(first) == 1 and not first.isalnum():
                return first
        return ";"

    @staticmethod
    def _build_processing_parameters(
        proc: Dict[str, Any],
    ) -> "FIDProcessingParameters":
        """Build informational :class:`FIDProcessingParameters` from the
        Blackchirp ``processing.csv`` mapping.

        These values do not drive the pipeline FT (the resolved ``FTSettings``
        do); they are retained only for display/serialization. The instrument's
        apodization / zero-pad cells are deliberately ignored -- the canonical
        FT is unconditionally unapodized and native-length.
        """
        _, FIDProcessingParameters, _ = _get_fid_classes()
        if not proc:
            return FIDProcessingParameters()

        def _f(key: str, default: float) -> float:
            value = proc.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        end_us = _f("FidEndUs", 0.0)
        # DC removal is unconditional in the canonical FT, so the instrument's
        # FidRemoveDC flag is not carried through.
        return FIDProcessingParameters(
            start_us=_f("FidStartUs", 0.0),
            end_us=end_us if end_us > 0 else None,
            units_power=int(_f("FtUnits", 6.0)),
        )
