"""
Core data structures for FTMW pipeline.

This module contains the fundamental data structures used throughout the pipeline:

Architecture:
- FTMWData: Top-level container for complete FTMW experiment
  ├── FID: Time domain data with processing parameters
  ├── ComplexFT: Frequency domain data (computed from FID)
  └── SpectralWindow[]: Analysis windows (subsets of ComplexFT)
      └── Peak[]: Pre-fitting detected peaks

- FittingResult: Post-fitting results container
  └── FittedPeak[]: Fitted parameters for individual peaks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import scipy.fft as sfft


class PeakClassification(Enum):
    """Peak strength classification based on SNR."""

    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class Sideband(Enum):
    """Sideband configuration for frequency conversion."""

    UPPER = "upper"
    LOWER = "lower"
    LSB = "lower"  # Alias for compatibility
    USB = "upper"  # Alias for compatibility


@dataclass
class FIDProcessingParameters:
    """
    Processing parameters for FID-to-FT conversion.

    The canonical FT is unconditionally unapodized, un-windowed, and
    native-length, so the only parameters are data selection (``start_us`` /
    ``end_us``), DC removal, and the display amplitude scale.
    """

    start_us: Optional[float] = None  # Start time in μs for windowing
    end_us: Optional[float] = None  # End time in μs for windowing
    rdc: bool = True  # Remove DC component (subtract average)
    units_power: int = 6  # Scaling factor (10^units_power, 6 for μV)

    def __post_init__(self) -> None:
        """Validate processing parameters."""
        if self.start_us is not None and self.start_us < 0:
            raise ValueError("Start time must be non-negative")
        if self.end_us is not None and self.end_us < 0:
            raise ValueError("End time must be non-negative")
        if (
            self.start_us is not None
            and self.end_us is not None
            and self.start_us >= self.end_us
        ):
            raise ValueError("Start time must be less than end time")


class PreprocessedFID:
    """
    Preprocessed FID data ready for FFT calculation.

    Contains time-domain FID data that has been preprocessed with windowing,
    zero-padding, filtering, etc. Separates preprocessing from FFT calculation.
    """

    def __init__(
        self,
        data: np.ndarray,
        spacing: float,
        probe_freq_mhz: float,
        sideband: Union[str, Sideband],
        original_length: int,
        processing_params: FIDProcessingParameters,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PreprocessedFID object.

        Parameters
        ----------
        data : np.ndarray
            Preprocessed FID data (windowed, filtered, zero-padded)
        spacing : float
            Original time spacing in seconds
        probe_freq_mhz : float
            Probe/LO frequency in MHz
        sideband : str or Sideband
            Sideband configuration
        original_length : int
            Original FID length before preprocessing (for normalization)
        processing_params : FIDProcessingParameters
            Parameters used for preprocessing
        metadata : dict, optional
            Preprocessing metadata
        """
        self.data = np.asarray(data, dtype=float)
        self.spacing = float(spacing)
        self.probe_freq_mhz = float(probe_freq_mhz)

        if isinstance(sideband, str):
            self.sideband = Sideband(sideband.lower())
        else:
            self.sideband = sideband

        self.original_length = int(original_length)
        self.processing_params = processing_params
        self.metadata = metadata or {}

    @property
    def n_points(self) -> int:
        """Number of preprocessed data points."""
        return len(self.data)

    def apply_molecular_frequency(self, scope_freq_mhz: np.ndarray) -> np.ndarray:
        """
        Convert scope frequency to molecular frequency.

        For upper sideband: molecular = probe + scope
        For lower sideband: molecular = probe - scope
        """
        if self.sideband in (Sideband.LOWER, Sideband.LSB):
            return cast(np.ndarray, self.probe_freq_mhz - scope_freq_mhz)
        else:
            return cast(np.ndarray, self.probe_freq_mhz + scope_freq_mhz)

    def compute_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of preprocessed data.

        Returns
        -------
        tuple
            (complex_spectrum, frequency_array) where complex_spectrum is the
            FFT result and frequency_array is in MHz
        """
        # Compute real FFT (since FID data is real)
        ft_data = sfft.rfft(self.data)

        # Generate frequency axis using rfftfreq
        scope_freqs = sfft.rfftfreq(len(self.data), d=self.spacing) / 1e6  # MHz

        # Convert to molecular frequencies
        mol_freqs = self.apply_molecular_frequency(scope_freqs)

        # Apply normalization (divide by original FID length, not padded length)
        ft_data /= self.original_length

        # Apply scaling
        scale_factor = 10**self.processing_params.units_power
        ft_data *= scale_factor

        # Note: autoscale_MHz feature has been removed - use 'trim' for frequency range selection

        return ft_data, mol_freqs

    def compute_complex_ft(self) -> "ComplexFT":
        """
        Complete FT processing including metadata.

        Returns
        -------
        ComplexFT
            ComplexFT object with frequency domain data
        """
        complex_spectrum, freq_array = self.compute_fft()

        return ComplexFT(
            freq_array=freq_array,
            complex_spectrum=complex_spectrum,
            metadata={"processing_params": self.processing_params},
        )


class FID:
    """
    Free Induction Decay time-domain data container.

    Contains real-valued time-domain voltage data and all parameters
    needed for FT processing. Designed to work with a single averaged FID.
    """

    def __init__(
        self,
        data: np.ndarray,
        spacing: float,
        probe_freq_mhz: float,
        sideband: Union[str, Sideband] = Sideband.UPPER,
        shots: int = 1,
        processing: Optional[FIDProcessingParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FID object.

        Parameters
        ----------
        data : np.ndarray
            Real-valued FID voltage data (1D array)
        spacing : float
            Time spacing between points in seconds
        probe_freq_mhz : float
            Probe/LO frequency in MHz
        sideband : str or Sideband
            Sideband configuration ('upper' or 'lower')
        shots : int
            Number of shots averaged
        processing : FIDProcessingParameters, optional
            FT processing parameters
        metadata : dict, optional
            Additional experimental metadata
        """
        self.data = np.asarray(data, dtype=float).flatten()  # Ensure 1D real array
        self.spacing = float(spacing)  # seconds
        self.probe_freq_mhz = float(probe_freq_mhz)

        if isinstance(sideband, str):
            self.sideband = Sideband(sideband.lower())
        else:
            self.sideband = sideband

        self.shots = int(shots)
        self.processing = processing or FIDProcessingParameters()
        self.metadata = metadata or {}

        # Validate
        if self.spacing <= 0:
            raise ValueError("Spacing must be positive")
        if self.shots <= 0:
            raise ValueError("Shots must be positive")

    @property
    def n_points(self) -> int:
        """Number of time points."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """FID duration in seconds."""
        return self.n_points * self.spacing

    @property
    def duration_us(self) -> float:
        """FID duration in microseconds."""
        return self.duration * 1e6

    def time_array(self) -> np.ndarray:
        """Generate time array in seconds."""
        return cast(np.ndarray, np.arange(self.n_points) * self.spacing)

    def time_array_us(self) -> np.ndarray:
        """Generate time array in microseconds."""
        return cast(np.ndarray, self.time_array() * 1e6)

    def apply_molecular_frequency(self, scope_freq_mhz: np.ndarray) -> np.ndarray:
        """
        Convert scope frequency to molecular frequency.

        For upper sideband: molecular = probe + scope
        For lower sideband: molecular = probe - scope
        """
        if self.sideband in (Sideband.LOWER, Sideband.LSB):
            return cast(np.ndarray, self.probe_freq_mhz - scope_freq_mhz)
        else:
            return cast(np.ndarray, self.probe_freq_mhz + scope_freq_mhz)

    def preprocess(
        self,
        start_us: Optional[float] = None,
        end_us: Optional[float] = None,
        rdc: bool = True,
        units_power: int = 6,
    ) -> PreprocessedFID:
        """
        Apply preprocessing to FID data, return new PreprocessedFID object.

        Stage 1 of FT processing: preprocessing only, no FFT computation. The
        canonical FT is unconditionally unapodized, un-windowed, and
        native-length, so preprocessing is just active-region selection plus
        optional DC removal:

        1. Extract the active region (``start_us`` to ``end_us``); points
           outside it are zeroed.
        2. Remove the DC component of the active region (when ``rdc``).

        Parameters
        ----------
        start_us : float, optional
            Start time in μs for windowing
        end_us : float, optional
            End time in μs for windowing
        rdc : bool, default=True
            Remove DC component (subtract average)
        units_power : int, default=6
            Scaling factor (10^units_power, 6 for μV)

        Returns
        -------
        PreprocessedFID
            PreprocessedFID object ready for FFT calculation
        """
        processing_params = FIDProcessingParameters(
            start_us=start_us,
            end_us=end_us,
            rdc=rdc,
            units_power=units_power,
        )

        # Step 1: Determine windowing boundaries in original FID
        time_us = self.time_array_us()
        start_idx = 0
        end_idx = len(self.data)

        if processing_params.start_us is not None:
            start_idx = int(np.searchsorted(time_us, processing_params.start_us))
        if processing_params.end_us is not None:
            end_idx = int(np.searchsorted(time_us, processing_params.end_us))

        # Start with full original data and zero regions outside bounds
        windowed_data = self.data.copy()
        original_length = len(self.data)  # For proper normalization

        # Zero out regions outside start_us/end_us bounds
        if start_idx > 0:
            windowed_data[:start_idx] = 0.0
        if end_idx < len(windowed_data):
            windowed_data[end_idx:] = 0.0

        # Step 2: Remove DC component from the active region
        if processing_params.rdc and start_idx < end_idx:
            active_data = windowed_data[start_idx:end_idx]
            dc_offset = np.mean(active_data)
            windowed_data[start_idx:end_idx] -= dc_offset

        # The canonical FT runs at native length -- no zero-padding.
        return PreprocessedFID(
            data=windowed_data,
            spacing=self.spacing,
            probe_freq_mhz=self.probe_freq_mhz,
            sideband=self.sideband,
            original_length=original_length,
            processing_params=processing_params,
            metadata={"source_fid_metadata": self.metadata},
        )

    # NOTE: FID.ft() method has been removed to enforce proper three-stage workflow:
    # 1. fid.preprocess(**params) -> PreprocessedFID
    # 2. preprocessed_fid.compute_fft() -> (spectrum, freq_array)
    # 3. ComplexFT.from_spectrum(spectrum, freq_array) -> ComplexFT
    # This separation provides cleaner architecture and better control over processing stages.


class ComplexFT:
    """
    Complex Fourier Transform frequency-domain data container.

    Contains the frequency-domain representation of FTMW data with
    associated experimental parameters. Computed from FID data.
    """

    def __init__(
        self,
        freq_array: np.ndarray,
        complex_spectrum: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ComplexFT object.

        Parameters
        ----------
        freq_array : np.ndarray
            Frequency array in MHz
        complex_spectrum : np.ndarray
            Complex spectrum data
        metadata : dict, optional
            Additional metadata
        """
        self.freq_array = np.asarray(freq_array, dtype=float)
        self.complex_spectrum = np.asarray(complex_spectrum, dtype=complex)

        if len(self.freq_array) != len(self.complex_spectrum):
            raise ValueError("Frequency and spectrum arrays must have same length")

        # Note: FID back-reference removed for cleaner architecture
        self.metadata = metadata or {}

        # Cached properties
        self._magnitude_spectrum: Optional[np.ndarray] = None
        self._freq_step: Optional[float] = None

    # NOTE: from_fid class method removed - use proper three-stage workflow:
    # 1. preprocessed = fid.preprocess(**params)
    # 2. spectrum, freqs = preprocessed.compute_fft()
    # 3. complex_ft = ComplexFT.from_spectrum(spectrum, freqs)

    @classmethod
    def from_spectrum(
        cls,
        complex_spectrum: np.ndarray,
        freq_array: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ComplexFT":
        """
        Create ComplexFT from spectrum data.

        Used for Stage 3 post-processing after FFT computation.

        Parameters
        ----------
        complex_spectrum : np.ndarray
            Complex spectrum data
        freq_array : np.ndarray
            Frequency array in MHz
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        ComplexFT
            ComplexFT object
        """
        return cls(
            freq_array=freq_array, complex_spectrum=complex_spectrum, metadata=metadata
        )

    @property
    def magnitude_spectrum(self) -> np.ndarray:
        """Magnitude spectrum (cached)."""
        if self._magnitude_spectrum is None:
            self._magnitude_spectrum = cast(np.ndarray, np.abs(self.complex_spectrum))
        return self._magnitude_spectrum

    @property
    def real_spectrum(self) -> np.ndarray:
        """Real component of spectrum."""
        return cast(np.ndarray, np.real(self.complex_spectrum))

    @property
    def imag_spectrum(self) -> np.ndarray:
        """Imaginary component of spectrum."""
        return cast(np.ndarray, np.imag(self.complex_spectrum))

    @property
    def freq_step(self) -> float:
        """Frequency step in MHz (cached)."""
        if self._freq_step is None:
            self._freq_step = float(np.mean(np.diff(self.freq_array)))
        return self._freq_step

    @property
    def freq_range(self) -> Tuple[float, float]:
        """Frequency range (min, max) in MHz."""
        return float(np.min(self.freq_array)), float(np.max(self.freq_array))

    @property
    def n_points(self) -> int:
        """Number of frequency points."""
        return len(self.freq_array)

    def extract_window(self, freq_min: float, freq_max: float) -> "SpectralWindow":
        """Extract a frequency window."""
        mask = (self.freq_array >= freq_min) & (self.freq_array <= freq_max)

        if not np.any(mask):
            raise ValueError(
                f"No data points in frequency range [{freq_min}, {freq_max}] MHz"
            )

        return SpectralWindow(
            parent_ft=self,
            freq_array=self.freq_array[mask],
            complex_spectrum=self.complex_spectrum[mask],
            freq_range=(freq_min, freq_max),
        )

    def trim_to_range(self, freq_min: float, freq_max: float) -> "ComplexFT":
        """
        Create a new ComplexFT object trimmed to the specified frequency range.

        This method is useful for removing noise regions and focusing analysis
        on the spectral activity region.

        Parameters
        ----------
        freq_min : float
            Minimum frequency in MHz
        freq_max : float
            Maximum frequency in MHz

        Returns
        -------
        ComplexFT
            New ComplexFT object containing only the specified frequency range
        """
        mask = (self.freq_array >= freq_min) & (self.freq_array <= freq_max)

        if not np.any(mask):
            raise ValueError(
                f"No data points in frequency range [{freq_min}, {freq_max}] MHz"
            )

        # Create new ComplexFT with trimmed data
        return ComplexFT(
            freq_array=self.freq_array[mask],
            complex_spectrum=self.complex_spectrum[mask],
            metadata={**self.metadata, "trimmed_range": (freq_min, freq_max)},
        )


class Peak:
    """
    Pre-fitting detected peak representation.

    Represents peaks detected in the spectrum before fitting, used for
    initial parameter guesses.
    """

    def __init__(
        self,
        frequency: float,
        intensity: float,
        index: Optional[int] = None,
        snr: Optional[float] = None,
        noise_std_local: Optional[float] = None,
        classification: Optional[Union[str, PeakClassification]] = None,
        **properties: Any,
    ):
        """
        Initialize Peak object.

        Parameters
        ----------
        frequency : float
            Peak frequency in MHz
        intensity : float
            Peak intensity (height above baseline)
        index : int, optional
            Index in original frequency array
        snr : float, optional
            Signal-to-noise ratio
        noise_std_local : float, optional
            Local noise standard deviation
        classification : str or PeakClassification, optional
            Peak strength classification
        **properties
            Additional peak properties
        """
        self.frequency = float(frequency)
        self.intensity = float(intensity)
        self.index = index
        self.snr = snr
        self.noise_std_local = noise_std_local

        # Handle classification
        self.classification: Optional[PeakClassification]
        if isinstance(classification, str):
            try:
                self.classification = PeakClassification(classification)
            except ValueError:
                self.classification = None
        else:
            self.classification = classification

        self.properties = properties

    @property
    def is_classified(self) -> bool:
        """Check if peak has been classified."""
        return self.classification is not None

    @property
    def is_strong(self) -> bool:
        """Check if peak is classified as strong."""
        return self.classification == PeakClassification.STRONG

    @property
    def is_medium(self) -> bool:
        """Check if peak is classified as medium."""
        return self.classification == PeakClassification.MEDIUM

    @property
    def is_weak(self) -> bool:
        """Check if peak is classified as weak."""
        return self.classification == PeakClassification.WEAK

    def __repr__(self) -> str:
        classification_str = (
            self.classification.value if self.classification else "unclassified"
        )
        snr_str = f"{self.snr:.1f}" if self.snr is not None else "None"
        return f"Peak(freq={self.frequency:.3f} MHz, intensity={self.intensity:.2e}, SNR={snr_str}, {classification_str})"

    def __lt__(self, other: "Peak") -> bool:
        """Sort peaks by intensity (strongest first)."""
        return self.intensity > other.intensity


@dataclass
class KnockoutInfo:
    """Per-line knockout-test outcome attached to a :class:`FittedPeak`.

    Persistent twin of :class:`ftmwpipeline.fitting.window_fit.KnockoutResult`:
    the algorithm-side dataclass is the in-flight working record, this one is
    the user-facing snapshot stored on the fitted peak.

    Attributes
    ----------
    delta_chi2 : float
        Diagnostic chi-squared increase under the freeze-others convention
        (every other peak held at its K-fit value when this peak is removed).
        Meaningful as the "energy carried by this line" check; no longer the
        gate because frozen-others leaves duplicate twins half-fit and
        produces spurious large increases.
    expected_delta_chi2 : float
        Diagnostic: the line's own noise-weighted energy.
    supported : bool
        Whether the AICc-with-n_eff gate prefers the K-peak fit
        (``aicc_delta >= 0``; REJECT-on-tie). ``supported = False`` flags
        the peak as redundant: removing it and re-fitting the surviving
        (K-1) peaks (with tau locked at the K-fit value) produces a
        strictly better AICc.
    p_value : float
        Diagnostic F-test p-value of the K-peak fit vs the (K-1)-peak
        refit. ``nan`` when the refit failed to converge or for peaks
        loaded from older files written before this column existed.
    n_eff : float
        Effective sample size used by the AICc gate. ``nan`` for older
        files written before this column existed.
    aicc_delta : float
        ``AICc(K-1 refit) - AICc(K)`` at the shared ``n_eff``; the gate
        statistic. Negative values mean the simpler model is preferred
        (peak redundant). ``nan`` when the refit failed to converge or
        for older files.
    """

    delta_chi2: float
    expected_delta_chi2: float
    supported: bool
    p_value: float = float("nan")
    n_eff: float = float("nan")
    aicc_delta: float = float("nan")


@dataclass
class FittedPeak:
    """
    Post-fitting peak results with fitted parameters.

    Represents the results of fitting a single peak, including fitted
    parameters, uncertainties, and quality metrics. The peak originates in one
    Stage 4 fit window; its ``peak_id`` is the Stage 3 promoted-peak index that
    seeded it, ``window_id`` is the originating fit window's id, and
    ``knockout`` carries the per-peak validation result from the Stage 5
    knockout test.
    """

    peak_id: Union[str, int]
    """Stage 3 promoted-peak index of the line (the entry in the persisted peak
    list that seeded this fit). For lines added by the blend-aware seeder
    without their own Stage 3 detection, this is the seeded peak's index --
    multiple :class:`FittedPeak` s may share the same ``peak_id`` in a blend."""
    frequency_mhz: float
    amplitude: float
    decay_rate: Optional[float] = None
    phase: Optional[float] = None

    # Parameter uncertainties
    frequency_error: Optional[float] = None
    amplitude_error: Optional[float] = None
    decay_rate_error: Optional[float] = None
    phase_error: Optional[float] = None

    # Quality metrics
    snr: Optional[float] = None
    chi_squared: Optional[float] = None

    # Stage 5 wiring: originating window id and the per-peak knockout result.
    window_id: Optional[int] = None
    """``FitWindow.window_id`` the line was fit in."""
    knockout: Optional[KnockoutInfo] = None
    """Knockout-test outcome from :func:`ftmwpipeline.fitting.window_fit.knockout_test`."""

    # Additional fitted parameters
    extra_parameters: Dict[str, float] = field(default_factory=dict)
    extra_errors: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        ferr = (
            f"{self.frequency_error:.6f}" if self.frequency_error is not None else "?"
        )
        return (
            f"FittedPeak(id={self.peak_id}, "
            f"freq={self.frequency_mhz:.6f}±{ferr} MHz, "
            f"amp={self.amplitude:.2e})"
        )


class SpectralWindow:
    """
    Analysis window - a subset of ComplexFT data for focused analysis.

    Represents a frequency range extracted from a ComplexFT for
    targeted peak detection and fitting operations.
    """

    def __init__(
        self,
        parent_ft: Optional[ComplexFT],
        freq_array: np.ndarray,
        complex_spectrum: np.ndarray,
        freq_range: Tuple[float, float],
        window_id: Optional[Union[str, int]] = None,
        peaks: Optional[List[Peak]] = None,
    ):
        """
        Initialize SpectralWindow.

        Parameters
        ----------
        parent_ft : ComplexFT, optional
            Parent ComplexFT this window was extracted from. ``None`` when the
            window was materialized from a non-persisted spectrum (e.g. Stage
            5's active-portion FT, which is regenerated on demand and not
            stored on the experiment).
        freq_array : np.ndarray
            Frequency array for this window
        complex_spectrum : np.ndarray
            Complex spectrum data for this window
        freq_range : tuple
            (min_freq, max_freq) in MHz
        window_id : str or int, optional
            Identifier for this window. Stage 5 materializes one window per
            :class:`FitWindow` and carries the integer ``FitWindow.window_id``
            here directly; earlier callers used opaque string ids.
        peaks : list of Peak, optional
            Detected peaks in this window
        """
        self.parent_ft = parent_ft
        self.freq_array = np.asarray(freq_array, dtype=float)
        self.complex_spectrum = np.asarray(complex_spectrum, dtype=complex)
        self.freq_range = freq_range
        self.window_id = window_id
        self.peaks = peaks or []

        if len(self.freq_array) != len(self.complex_spectrum):
            raise ValueError("Frequency and spectrum arrays must have same length")

    @property
    def magnitude_spectrum(self) -> np.ndarray:
        """Magnitude spectrum."""
        return cast(np.ndarray, np.abs(self.complex_spectrum))

    @property
    def real_spectrum(self) -> np.ndarray:
        """Real component."""
        return cast(np.ndarray, np.real(self.complex_spectrum))

    @property
    def imag_spectrum(self) -> np.ndarray:
        """Imaginary component."""
        return cast(np.ndarray, np.imag(self.complex_spectrum))

    @property
    def n_points(self) -> int:
        """Number of frequency points."""
        return len(self.freq_array)

    @property
    def n_peaks(self) -> int:
        """Number of detected peaks."""
        return len(self.peaks)

    @property
    def center_frequency(self) -> float:
        """Center frequency in MHz."""
        return (self.freq_range[0] + self.freq_range[1]) / 2

    @property
    def bandwidth(self) -> float:
        """Bandwidth in MHz."""
        return self.freq_range[1] - self.freq_range[0]

    def add_peak(self, peak: Peak) -> None:
        """Add a detected peak to this window."""
        if not isinstance(peak, Peak):
            raise TypeError("peak must be a Peak object")

        # Validate peak is within window
        if not (self.freq_range[0] <= peak.frequency <= self.freq_range[1]):
            raise ValueError(
                f"Peak frequency {peak.frequency} MHz outside window range {self.freq_range}"
            )

        self.peaks.append(peak)

    def get_peaks_by_classification(
        self, classification: Union[str, PeakClassification]
    ) -> List[Peak]:
        """Get peaks with specified classification."""
        if isinstance(classification, str):
            classification = PeakClassification(classification)
        return [peak for peak in self.peaks if peak.classification == classification]

    def __repr__(self) -> str:
        return (
            f"SpectralWindow(id={self.window_id}, "
            f"range=[{self.freq_range[0]:.1f}, {self.freq_range[1]:.1f}] MHz, "
            f"n_points={self.n_points}, n_peaks={self.n_peaks})"
        )


@dataclass
class AuditStep:
    """One decision in the conservative add-one-peak audit trail.

    Persistent twin of :class:`ftmwpipeline.fitting.window_fit.AddStep`. The
    add-one-peak loop records one of these per seed / re-seed / candidate so
    the conservative loop's accept/reject behaviour can be replayed and
    curated after the fact.

    Attributes
    ----------
    n_peaks_before : int
        Number of accepted lines before this step.
    candidate_offset_mhz : float
        Baseband offset (MHz) of the line tested at this step.
    chi2_before, chi2_after : float
        Noise-weighted chi-squared before / with the candidate.
    f_statistic, p_value : float
        Diagnostic nested-model F-test of the chi-squared improvement.
        Kept as a familiar statistic; the accept gate is
        AICc-with-``n_eff`` (see ``n_eff`` / ``aicc_delta``).
    aic_before, aic_after : float
        Diagnostic AIC at the raw ``n_data``.
    separation_ok : bool
        Whether the candidate cleared the peak-separation constraint.
    decision : str
        ``"seed"``, ``"seed-blend"``, ``"accept"``, ``"promote"``,
        ``"tentative"``, or ``"reject"``.
    reason : str
        Free-text annotation.
    n_eff : float
        Effective sample size shared by the K-vs-(K+1) AICc evaluation;
        computed once from the K+1 (trial) model magnitude. ``nan`` on
        steps that do not run the gate (the K=1 seed and separation-
        rejected candidates) and on hand-edited or legacy audit blobs
        that omit the field.
    aicc_delta : float
        ``AICc(K+1) - AICc(K)`` at the shared ``n_eff``; negative means
        the gate accepted (the K+1 model is preferred). ``nan`` on the
        same steps as ``n_eff``.
    """

    n_peaks_before: int
    candidate_offset_mhz: float
    chi2_before: float
    chi2_after: float
    f_statistic: float
    p_value: float
    aic_before: float
    aic_after: float
    separation_ok: bool
    decision: str
    reason: str = ""
    n_eff: float = float("nan")
    aicc_delta: float = float("nan")


@dataclass
class ThawInfo:
    """One residual-edge-coherence thaw record on a fit window.

    Persistent twin of :class:`ftmwpipeline.fitting.plan_execution.ThawEvent`.
    Stage 5's residual edge-coherence handshake unfreezes a fixed contributor
    and co-fits it with its dependent window; one of these records is appended
    per attempt (accepted or not).

    Attributes
    ----------
    dependent_window_id : int
        Window whose post-fit residual edge triggered the thaw.
    primary_window_id : int
        Window the thawed contributor was originally fit in. ``-1`` when no
        contributor was available on the flagged side (a no-op record).
    contributor_peak_index : int
        Stage 3 peak index of the thawed line. ``-1`` for the no-op record.
    contributor_frequency_mhz : float
        Molecular frequency of the thawed line (MHz); ``nan`` for the no-op.
    edge_side : str
        ``"low"`` or ``"high"`` -- the flagged residual edge.
    edge_coherence_before, edge_coherence_after : float
        Residual ``S_coh`` on the flagged edge before / after the co-fit.
        ``nan`` for ``after`` if the co-fit did not converge.
    accepted : bool
        Whether the co-fit converged and lowered the flagged-edge coherence to
        at or below the threshold.
    reason : str
        Free-text annotation.
    """

    dependent_window_id: int
    primary_window_id: int
    contributor_peak_index: int
    contributor_frequency_mhz: float
    edge_side: str
    edge_coherence_before: float
    edge_coherence_after: float
    accepted: bool
    reason: str = ""


class FittingResult:
    """
    Container for fitting results from analysis of SpectralWindow(s).

    Stores fitted parameters, quality metrics, and diagnostic information.
    Stage 5 also wires the per-iteration ``audit_trail`` (the conservative
    add-one-peak loop's decision log) and the per-window ``thaw_events``
    (the residual-edge-coherence renegotiation outcomes that touched this
    window).
    """

    def __init__(
        self,
        success: bool = False,
        fitted_spectrum: Optional[np.ndarray] = None,
        cost: float = np.inf,
        iterations: int = 0,
        aic: float = np.inf,
        reduced_chi2: float = np.inf,
        window: Optional[SpectralWindow] = None,
        window_id: Optional[int] = None,
        shape: str = "lorentzian",
    ):
        """Initialize FittingResult."""
        self.success = success
        self.fitted_spectrum = fitted_spectrum
        self.cost = cost
        self.iterations = iterations
        self.aic = aic
        self.reduced_chi2 = reduced_chi2
        self.window = window
        self.window_id = window_id
        # Line-shape selector this window was fit with ("lorentzian" or
        # "gaussian"); recorded per-window so a future mixed-shape fit
        # stays representable in the persisted schema even though the
        # current driver applies one shape across every window.
        self.shape = shape

        # Fitted peaks
        self.fitted_peaks: List[FittedPeak] = []

        # Global parameters (shared across peaks): tau and its error live here.
        # Each entry has free-form keys (value, error, peak_ids, ...) so the
        # inner-dict value type widens beyond float.
        self.shared_parameters: Dict[str, Dict[str, Any]] = {}
        # Frozen-contributor summaries used in the fit live here.
        self.fixed_parameters: Dict[str, Dict[str, Any]] = {}

        # Diagnostics
        self.residuals: Optional[np.ndarray] = None
        self.quality_metrics: Dict[str, float] = {}

        # Stage 5 wiring: conservative-loop audit trail and the per-window
        # slice of the plan-level thaw history (chronological).
        self.audit_trail: List[AuditStep] = []
        self.thaw_events: List[ThawInfo] = []
        # Per-window slice of the plan-level rescue history (chronological).
        # ``RescueRoundInfo`` is defined below SpectrumFit; the forward
        # reference here lets it stay in execution order in the module
        # without splitting the class definitions.
        self.rescue_events: List["RescueRoundInfo"] = []

    def add_fitted_peak(self, fitted_peak: FittedPeak) -> None:
        """Add a fitted peak result."""
        self.fitted_peaks.append(fitted_peak)

    def set_shared_parameter(
        self,
        name: str,
        value: float,
        error: Optional[float] = None,
        peak_ids: Optional[List] = None,
    ) -> None:
        """Set a parameter shared across multiple peaks."""
        self.shared_parameters[name] = {
            "value": value,
            "error": error,
            "peak_ids": peak_ids or [p.peak_id for p in self.fitted_peaks],
        }

    def set_fixed_parameter(
        self, name: str, value: float, peak_ids: Optional[List] = None
    ) -> None:
        """Set a parameter held fixed during fitting."""
        self.fixed_parameters[name] = {
            "value": value,
            "peak_ids": peak_ids or [p.peak_id for p in self.fitted_peaks],
        }

    @property
    def n_peaks_fitted(self) -> int:
        """Number of fitted peaks."""
        return len(self.fitted_peaks)

    @property
    def frequencies_mhz(self) -> List[float]:
        """Fitted frequencies for all peaks."""
        return [p.frequency_mhz for p in self.fitted_peaks]

    @property
    def amplitudes(self) -> List[float]:
        """Fitted amplitudes for all peaks."""
        return [p.amplitude for p in self.fitted_peaks]

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"FittingResult(status={status}, n_peaks={self.n_peaks_fitted}, "
            f"cost={self.cost:.2e}, AIC={self.aic:.2f})"
        )


class FTMWData:
    """
    Top-level container for complete FTMW experiment data.

    Contains FID (time domain), ComplexFT (frequency domain), and
    analysis windows with detected peaks. Provides the main interface
    for FTMW data processing workflows.
    """

    def __init__(
        self,
        fid: FID,
        experiment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FTMWData.

        Parameters
        ----------
        fid : FID
            Time-domain FID data
        experiment_id : str, optional
            Experiment identifier
        metadata : dict, optional
            Experiment metadata
        """
        self.fid = fid
        self.experiment_id = experiment_id
        self.metadata = metadata or {}

        # Frequency domain data (computed lazily)
        self._complex_ft: Optional[ComplexFT] = None

        # Analysis windows
        self.spectral_windows: List[SpectralWindow] = []

        # Fitting results
        self.fitting_results: List[FittingResult] = []

    @property
    def complex_ft(self) -> ComplexFT:
        """Get or compute ComplexFT from FID using default parameters."""
        if self._complex_ft is None:
            # Use three-stage workflow with default parameters
            preprocessed = self.fid.preprocess()
            spectrum, freqs = preprocessed.compute_fft()
            self._complex_ft = ComplexFT.from_spectrum(
                spectrum,
                freqs,
                metadata={"processing_params": preprocessed.processing_params},
            )
        return self._complex_ft

    def compute_ft(self, **ft_kwargs: Any) -> ComplexFT:
        """Compute ComplexFT with custom parameters using three-stage workflow."""
        # Use three-stage workflow with custom parameters
        preprocessed = self.fid.preprocess(**ft_kwargs)
        spectrum, freqs = preprocessed.compute_fft()
        self._complex_ft = ComplexFT.from_spectrum(
            spectrum,
            freqs,
            metadata={"processing_params": preprocessed.processing_params},
        )
        return self._complex_ft

    def create_spectral_window(
        self, freq_min: float, freq_max: float, window_id: Optional[str] = None
    ) -> SpectralWindow:
        """Create and store a new spectral window."""
        window = self.complex_ft.extract_window(freq_min, freq_max)
        window.window_id = window_id
        self.spectral_windows.append(window)
        return window

    def add_fitting_result(self, result: FittingResult) -> None:
        """Add a fitting result."""
        self.fitting_results.append(result)

    @property
    def n_windows(self) -> int:
        """Number of spectral windows."""
        return len(self.spectral_windows)

    @property
    def n_fitted_results(self) -> int:
        """Number of fitting results."""
        return len(self.fitting_results)

    @property
    def total_fitted_peaks(self) -> int:
        """Total number of fitted peaks across all results."""
        return sum(result.n_peaks_fitted for result in self.fitting_results)

    def __repr__(self) -> str:
        return (
            f"FTMWData(id={self.experiment_id}, "
            f"fid_duration={self.fid.duration_us:.1f} μs, "
            f"n_windows={self.n_windows}, "
            f"n_fitted_peaks={self.total_fitted_peaks})"
        )


# ---------------------------------------------------------------------------
# Stage 4: window-assignment plan
# ---------------------------------------------------------------------------
#
# Stage 4 turns the promoted Stage 3 peak list into a *fit plan*: an ordered set
# of disjoint analysis windows, each carrying the peaks to fit freely, the
# strong out-of-band lines whose leakage must be carried as a frozen background,
# a fit dependency DAG, and a difficulty class. The plain ``SpectralWindow``
# above is the data-bearing window used downstream; the structures here are the
# *planning* substrate (no spectrum arrays — only references into the Stage 3
# peak list). See ``dev-docs/planning/stage4-window-assignment.md``.


class WindowDifficulty(Enum):
    """Stage 4 difficulty class for a fit window.

    ``EASY`` windows are isolated/independent and can be fit in parallel;
    ``HARD`` windows contain or are materially influenced by a strong line (or
    exceed the width cap, or sit in a coupled cluster) and warrant extra Stage 5
    budget.
    """

    EASY = "easy"
    HARD = "hard"


@dataclass
class FixedContributor:
    """A strong line, fit freely in *its own* window, contributing leakage here.

    A fixed contributor is evaluated in a dependent window's model with its
    parameters frozen at the values found in its ``primary_window_id`` — it
    contributes only its finite-T leakage skirt, it is not re-fit. This is what
    lets a strong line's leakage be represented in every window it reaches
    without fitting it more than once.

    Attributes
    ----------
    peak_index : int
        Index into the persisted Stage 3 peak list of the strong line.
    primary_window_id : int
        ``window_id`` of the fit window that fits this line as a free peak.
    frequency_mhz : float
        Frequency of the line (MHz), carried for diagnostics/serialization.
    freeze_eligible : bool
        Whether the line's SNR clears ``min_freeze_snr`` so its parameters are
        stable enough to freeze without contaminating the dependent window
        (Stage 4 open question O4-2). ``False`` flags a thaw-and-re-fit
        candidate for Stage 5.
    """

    peak_index: int
    primary_window_id: int
    frequency_mhz: float
    freeze_eligible: bool = True


@dataclass
class FitWindow:
    """One analysis window in a Stage 4 :class:`WindowPlan`.

    Fit windows are **disjoint** and cover each spectrum point at most once —
    that is the hard Stage 4 invariant. The contributor sets (``free_peak_*``
    plus ``fixed_contributors``) intentionally *do* overlap across windows.

    Attributes
    ----------
    window_id : int
        Stable identifier, assigned in ascending-frequency order.
    freq_range : tuple of float
        ``(min_mhz, max_mhz)`` span whose points enter this window's residual.
    free_peak_indices : list of int
        Indices into the persisted Stage 3 peak list of the peaks fit freely in
        this window (leakage-artifact detections already pruned out).
    fixed_contributors : list of FixedContributor
        Out-of-band strong lines whose frozen leakage is carried here.
    difficulty : WindowDifficulty
        ``EASY`` or ``HARD``.
    batch : int
        Parallel-execution group: all windows in a batch are mutually
        independent and depend only on earlier batches.
    split_proposal : float, optional
        A complex-edge-clean interior frequency at which Stage 5 *may* split a
        too-wide hard window. ``None`` when no split is proposed.
    needs_joint_treatment : bool
        Set when a hard window is strongly coupled with no clean interior split
        point — Stage 5 must treat it jointly.
    diagnostics : dict
        Free-form diagnostics (predicted vs trimmed extent, edge-statistic
        values, width-cap hit flag, pruned-artifact count, …).
    """

    window_id: int
    freq_range: Tuple[float, float]
    free_peak_indices: List[int] = field(default_factory=list)
    fixed_contributors: List[FixedContributor] = field(default_factory=list)
    difficulty: WindowDifficulty = WindowDifficulty.EASY
    batch: int = 0
    split_proposal: Optional[float] = None
    needs_joint_treatment: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def width_mhz(self) -> float:
        """Width of the window in MHz."""
        return abs(self.freq_range[1] - self.freq_range[0])

    @property
    def n_free_peaks(self) -> int:
        """Number of free peaks in this window."""
        return len(self.free_peak_indices)

    def __repr__(self) -> str:
        return (
            f"FitWindow(id={self.window_id}, "
            f"range=[{self.freq_range[0]:.1f}, {self.freq_range[1]:.1f}] MHz, "
            f"free={self.n_free_peaks}, fixed={len(self.fixed_contributors)}, "
            f"{self.difficulty.value}, batch={self.batch})"
        )


@dataclass
class MergeRequest:
    """Structural re-plan request: merge two adjacent fit windows into one.

    Emitted by Stage 5 when the residual edge-coherence check on a fit
    window's edge flags above threshold and *no fixed contributor* on that
    side is available to thaw — i.e. a real spectral feature crosses the
    window boundary. Routed through
    :func:`~ftmwpipeline.preprocessing.window_planning.replan`, which
    produces a revised :class:`WindowPlan` with a bumped
    :attr:`WindowPlan.plan_revision`; Stage 5 then re-fits the affected
    batches.

    Attributes
    ----------
    window_a_id : int
        ``window_id`` of one of the two windows to merge.
    window_b_id : int
        ``window_id`` of the other window. The two windows must be adjacent
        in the plan (no other window's ``freq_range`` lies between them);
        ``replan`` raises if not. The surviving merged window keeps the
        lower of the two ids.
    reason : str
        Free-text annotation for the renegotiation audit log.
    """

    window_a_id: int
    window_b_id: int
    reason: str = ""


@dataclass
class WindowPlan:
    """The complete Stage 4 fit plan: ordered windows + a fit dependency DAG.

    Attributes
    ----------
    windows : list of FitWindow
        The disjoint fit windows, ordered by ascending frequency.
    dependency_edges : list of tuple of int
        ``(window_id, depends_on_window_id)`` pairs — a window depends on the
        windows that fit its fixed contributors. The graph is acyclic.
    topological_order : list of int
        ``window_id`` values in a valid fit order (every window appears after
        all windows it depends on).
    parameters : dict
        The Stage 4 parameters used (edge M/threshold, width cap, …).
    diagnostics : dict
        Plan-level diagnostics (e.g. coherent regions with no identifiable
        strong-line source — a hint that peak detection missed a line).
    plan_revision : int
        Monotonic counter bumped each time
        :func:`~ftmwpipeline.preprocessing.window_planning.replan` applies a
        structural change. ``0`` is the initial plan from
        :func:`~ftmwpipeline.preprocessing.window_planning.build_window_plan`;
        downstream stages can use this to detect plan churn between Stage 5
        invocations.
    """

    windows: List[FitWindow] = field(default_factory=list)
    dependency_edges: List[Tuple[int, int]] = field(default_factory=list)
    topological_order: List[int] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    plan_revision: int = 0

    @property
    def n_windows(self) -> int:
        """Number of fit windows in the plan."""
        return len(self.windows)

    @property
    def n_batches(self) -> int:
        """Number of parallel-execution batches."""
        if not self.windows:
            return 0
        return max(w.batch for w in self.windows) + 1

    def window(self, window_id: int) -> "FitWindow":
        """Return the window with the given ``window_id`` (raises if absent)."""
        for w in self.windows:
            if w.window_id == window_id:
                return w
        raise KeyError(f"no window with window_id={window_id}")

    def __repr__(self) -> str:
        n_hard = sum(1 for w in self.windows if w.difficulty == WindowDifficulty.HARD)
        return (
            f"WindowPlan(n_windows={self.n_windows}, hard={n_hard}, "
            f"n_batches={self.n_batches}, "
            f"n_dependencies={len(self.dependency_edges)})"
        )


# ---------------------------------------------------------------------------
# Stage 5: persistent fit aggregate
# ---------------------------------------------------------------------------
#
# Stage 5 turns the Stage 4 ``WindowPlan`` into a fitted line list. The
# in-flight working records (``WindowOutcome`` / ``PlanFitOutcome`` in
# :mod:`ftmwpipeline.fitting.plan_execution`) carry the raw least-squares
# arrays needed to drive the orchestrator; the persistent user-facing types
# below carry the spectroscopic parameters, the audit / renegotiation
# history, and the plan-level diagnostics that downstream consumers
# (serialization, visualization, hand-edit) need. The two layers are
# connected by the pure converters in
# :mod:`ftmwpipeline.fitting.result_conversion`.


@dataclass
class ReplanInfo:
    """One structural-replan record in :class:`SpectrumFit`.

    Persistent twin of :class:`ftmwpipeline.fitting.plan_execution.ReplanEvent`.
    Emitted when a residual edge-coherence flag has no fixed contributor to
    thaw and a frequency-adjacent neighbour exists, prompting Stage 4 to
    merge the two windows and bump the plan revision.

    Attributes
    ----------
    triggering_window_id : int
        Window whose flagged edge prompted the merge.
    partner_window_id : int
        Adjacent window the trigger asked to merge with.
    surviving_window_id : int
        ``min(triggering_window_id, partner_window_id)`` -- the id that carries
        the merged window in the revised plan.
    edge_side : str
        ``"low"`` or ``"high"``.
    edge_coherence_before : float
        Residual ``S_coh`` on the flagged edge before the merge.
    revision_before, revision_after : int
        :attr:`WindowPlan.plan_revision` before / after the merge. Equal when
        the request was rejected (no-op).
    accepted : bool
        Whether the merge was applied and the refit completed.
    reason : str
        Free-text annotation.
    """

    triggering_window_id: int
    partner_window_id: int
    surviving_window_id: int
    edge_side: str
    edge_coherence_before: float
    revision_before: int
    revision_after: int
    accepted: bool
    reason: str = ""


@dataclass
class RescueCandidateInfo:
    """One detector candidate from a rescue round.

    Persistent twin of the fields read off
    :class:`ftmwpipeline.fitting.residual_screening.ResidualPeakCandidate`.
    Carries only the bin frequency / magnitude / SNR -- enough to render
    the audit-trail candidates row from the persisted fit without
    re-running the rescue. The detector's ``bin_index`` and
    ``prominence_sigma_c`` are not persisted because they are tied to the
    transient residual-slice indexing of the round in which the candidate
    was nominated.

    Attributes
    ----------
    frequency_mhz : float
        Baseband-offset (MHz) of the detected candidate in the window's
        signed-offset grid -- the same frame the rescue's
        :func:`conservative_fit` consumed it in.
    magnitude : float
        ``|residual|`` at the candidate's bin.
    snr : float
        ``magnitude / sigma_c`` at the bin, with ``sigma_c = sigma / sqrt(2)``.
    """

    frequency_mhz: float
    magnitude: float
    snr: float


@dataclass
class RescueRoundInfo:
    """One residual-rescue round in a :class:`SpectrumFit`.

    Persistent twin of the in-flight
    :class:`ftmwpipeline.fitting.residual_rescue.RescueRoundDiagnostics`
    (algorithm-side) and
    :class:`ftmwpipeline.fitting.plan_execution.RescueEvent` (plan-executor-
    side). One record per round actually executed; an empty list on a
    :class:`FittingResult` means the rescue chain produced no candidates
    on the first round, was disabled for the run, or that the fit predates
    the rescue persistence schema.

    Used at two levels mirroring the :class:`ThawInfo` precedent: per
    window on :attr:`FittingResult.rescue_events`, and chronologically
    across all windows on :attr:`SpectrumFit.rescue_history`. The
    ``window_id`` field disambiguates the latter.

    The intermediate per-round :class:`WindowFitResult` /
    :class:`KnockoutResult` objects are not persisted -- they are
    deterministic functions of the persisted initial fit, the consolidated
    fit, and the candidate lists carried here, and re-running
    :func:`rescue_and_consolidate` from the persisted state reproduces them.
    The final-round per-peak knockout already lives on
    :attr:`FittedPeak.knockout`.

    Attributes
    ----------
    window_id : int
        :class:`FitWindow` id this round belongs to.
    round_idx : int
        Zero-based round counter within the window's chain.
    n_initial_peaks : int
        Number of peaks the round inherited from the previous round.
    n_rescue_added : int
        Number of peaks the rescue's conservative fit accepted (before
        the joint refit / knockout sweep).
    n_pruned_total : int
        Number of peaks the iterative AICc cleanup dropped from the joint
        refit.
    n_pruned_rescue_origin : int
        Of the pruned peaks, how many were rescue-origin (i.e., added in
        this round). The **failsafe diagnostic**: a high count signals
        the joint refit may not have escaped a pathological basin and is
        undoing the rescue's contribution.
    n_merged : int
        Number of close-peak pairs the merge cleanup collapsed before
        the knockout sweep.
    chi2_before, chi2_after : float
        Noise-weighted chi-squared of the previous round's fit and of
        the consolidated (post-pruning) fit, both evaluated against the
        same window data. Equal when ``accepted`` is False.
    tau_us_before, tau_us_after : float
        Shared decay constant before and after the round.
    accepted : bool
        Whether this round's contribution replaced the previous round's
        fit. False when the rescue nominated nothing, the joint refit
        failed, or pruning would have emptied the model.
    reason : str
        Free-text annotation -- which termination case fired, how many
        peaks were pruned or merged, etc.
    candidates : list of RescueCandidateInfo
        Detector candidates the rescue passed to ``conservative_fit``.
    """

    window_id: int
    round_idx: int
    n_initial_peaks: int
    n_rescue_added: int
    n_pruned_total: int
    n_pruned_rescue_origin: int
    n_merged: int
    chi2_before: float
    chi2_after: float
    tau_us_before: float
    tau_us_after: float
    accepted: bool
    reason: str = ""
    candidates: List[RescueCandidateInfo] = field(default_factory=list)


@dataclass
class SpectrumFit:
    """The complete Stage 5 fit of a :class:`WindowPlan`.

    Parallels :class:`WindowPlan`: each :class:`FitWindow` produces one
    :class:`FittingResult`, and the plan-level audit and renegotiation
    histories live alongside the per-window results. The merged global line
    list (``fitted_peaks``) is the curated user-facing output, sorted by
    molecular frequency, with each peak tagged with the
    :attr:`FittedPeak.window_id` of its originating fit window.

    Attributes
    ----------
    window_fits : list of FittingResult
        Per-window fit results, ordered by ascending ``window_id``.
    fitted_peaks : list of FittedPeak
        Merged global line list, sorted by ``frequency_mhz``. Each peak's
        ``window_id`` points back to its originating window.
    thaw_history : list of ThawInfo
        Every thaw attempt across all windows, in execution order. The
        per-window slice is mirrored on each :class:`FittingResult` for
        convenience; this list is the canonical plan-level history.
    replan_history : list of ReplanInfo
        Every structural-replan attempt, in execution order. Empty when no
        merges were proposed.
    rescue_history : list of RescueRoundInfo
        Every residual-rescue round across all windows, in execution
        order. The per-window slice is mirrored on
        :attr:`FittingResult.rescue_events` for convenience; this list
        is the canonical plan-level history. Empty when the rescue was
        disabled or no window produced any candidates.
    final_plan_revision : int
        :attr:`WindowPlan.plan_revision` of the plan the ``window_fits``
        describe. ``0`` when no structural change happened.
    parameters : dict
        The Stage 5 parameters used to produce this fit (``tau0_us``,
        ``residual_edge_threshold``, ``max_thaw_rounds``, etc.). Recorded so
        the persisted fit is self-describing.
    diagnostics : dict
        Plan-level diagnostics (e.g. windows whose fit did not converge,
        bookkeeping summaries).
    """

    window_fits: List[FittingResult] = field(default_factory=list)
    fitted_peaks: List[FittedPeak] = field(default_factory=list)
    thaw_history: List[ThawInfo] = field(default_factory=list)
    replan_history: List[ReplanInfo] = field(default_factory=list)
    rescue_history: List["RescueRoundInfo"] = field(default_factory=list)
    final_plan_revision: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        """Number of fitted windows."""
        return len(self.window_fits)

    @property
    def n_fitted_peaks(self) -> int:
        """Total number of fitted peaks across all windows."""
        return len(self.fitted_peaks)

    def window_fit(self, window_id: int) -> FittingResult:
        """Return the :class:`FittingResult` for ``window_id`` (raises if absent)."""
        for fit in self.window_fits:
            if fit.window_id == window_id:
                return fit
        raise KeyError(f"no fitting result for window_id={window_id}")

    def __repr__(self) -> str:
        n_thaw = len(self.thaw_history)
        n_thaw_accept = sum(1 for e in self.thaw_history if e.accepted)
        n_replan = len(self.replan_history)
        return (
            f"SpectrumFit(n_windows={self.n_windows}, "
            f"n_fitted_peaks={self.n_fitted_peaks}, "
            f"thaw={n_thaw_accept}/{n_thaw}, "
            f"replans={n_replan}, revision={self.final_plan_revision})"
        )
