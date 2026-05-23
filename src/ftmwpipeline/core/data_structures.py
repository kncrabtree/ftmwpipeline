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

import numpy as np
import scipy.fft as sfft
import scipy.signal as spsig
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum


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
    
    Based on BlackChirp processing settings, these control how the time-domain
    FID is converted to frequency domain.
    """
    start_us: Optional[float] = None  # Start time in μs for windowing
    end_us: Optional[float] = None    # End time in μs for windowing
    winf: Optional[str] = None        # Window function name (scipy.signal compatible)
    zpf: int = 1                      # Zero padding factor (powers of 2)
    rdc: bool = True                  # Remove DC component (subtract average)
    expf_us: Optional[float] = None   # Exponential decay filter time constant (μs)
    units_power: int = 6              # Scaling factor (10^units_power, 6 for μV)
    
    def __post_init__(self):
        """Validate processing parameters."""
        if self.zpf < 0:
            raise ValueError("Zero padding factor must be non-negative")
        if self.start_us is not None and self.start_us < 0:
            raise ValueError("Start time must be non-negative")
        if self.end_us is not None and self.end_us < 0:
            raise ValueError("End time must be non-negative")
        if self.start_us is not None and self.end_us is not None and self.start_us >= self.end_us:
            raise ValueError("Start time must be less than end time")
        if self.expf_us is not None and self.expf_us <= 0:
            raise ValueError("Exponential filter time constant must be positive")


class PreprocessedFID:
    """
    Preprocessed FID data ready for FFT calculation.
    
    Contains time-domain FID data that has been preprocessed with windowing,
    zero-padding, filtering, etc. Separates preprocessing from FFT calculation.
    """
    
    def __init__(self, data: np.ndarray, spacing: float, 
                 probe_freq_mhz: float, sideband: Union[str, Sideband],
                 original_length: int, processing_params: FIDProcessingParameters,
                 metadata: Optional[Dict[str, Any]] = None):
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
            return self.probe_freq_mhz - scope_freq_mhz
        else:
            return self.probe_freq_mhz + scope_freq_mhz
    
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
        scale_factor = 10 ** self.processing_params.units_power
        ft_data *= scale_factor
        
        # Note: autoscale_MHz feature has been removed - use 'trim' for frequency range selection
        
        return ft_data, mol_freqs
    
    def compute_complex_ft(self) -> 'ComplexFT':
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
            metadata={'processing_params': self.processing_params}
        )


class FID:
    """
    Free Induction Decay time-domain data container.
    
    Contains real-valued time-domain voltage data and all parameters 
    needed for FT processing. Designed to work with a single averaged FID.
    """
    
    def __init__(self, data: np.ndarray, spacing: float, 
                 probe_freq_mhz: float, sideband: Union[str, Sideband] = Sideband.UPPER,
                 shots: int = 1, processing: Optional[FIDProcessingParameters] = None,
                 metadata: Optional[Dict[str, Any]] = None):
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
        return np.arange(self.n_points) * self.spacing
    
    def time_array_us(self) -> np.ndarray:
        """Generate time array in microseconds."""
        return self.time_array() * 1e6
    
    def apply_molecular_frequency(self, scope_freq_mhz: np.ndarray) -> np.ndarray:
        """
        Convert scope frequency to molecular frequency.
        
        For upper sideband: molecular = probe + scope
        For lower sideband: molecular = probe - scope
        """
        if self.sideband in (Sideband.LOWER, Sideband.LSB):
            return self.probe_freq_mhz - scope_freq_mhz
        else:
            return self.probe_freq_mhz + scope_freq_mhz
    
    def preprocess(self, start_us: Optional[float] = None, end_us: Optional[float] = None, 
                   zpf: int = 1, expf_us: Optional[float] = None, 
                   window_function: Optional[str] = None, rdc: bool = True,
                   units_power: int = 6) -> PreprocessedFID:
        """
        Apply preprocessing to FID data, return new PreprocessedFID object.
        
        Stage 1 of FT processing: preprocessing only, no FFT computation.
        
        CRITICAL: Proper preprocessing sequence:
        1. Extract windowed data (start_us to end_us)
        2. Apply exponential filtering ONLY to windowed data
        3. Apply window function ONLY to windowed/filtered data
        4. Apply zero padding to processed window
        
        Parameters
        ----------
        start_us : float, optional
            Start time in μs for windowing
        end_us : float, optional
            End time in μs for windowing
        zpf : int, default=1
            Zero padding factor (powers of 2)
        expf_us : float, optional
            Exponential decay filter time constant (μs) - applied ONLY to windowed data
        window_function : str, optional
            Window function name (scipy.signal compatible) - applied ONLY to windowed data
        rdc : bool, default=True
            Remove DC component (subtract average)
        units_power : int, default=6
            Scaling factor (10^units_power, 6 for μV)
            
        Returns
        -------
        PreprocessedFID
            PreprocessedFID object ready for FFT calculation
        """
        # Create processing parameters from inputs (autoscale_MHz deprecated and removed)
        processing_params = FIDProcessingParameters(
            start_us=start_us,
            end_us=end_us,
            winf=window_function,
            zpf=zpf,
            rdc=rdc,
            expf_us=expf_us,
            units_power=units_power
        )
        
        # Step 1: Determine windowing boundaries in original FID
        time_us = self.time_array_us()
        start_idx = 0
        end_idx = len(self.data)
        
        if processing_params.start_us is not None:
            start_idx = np.searchsorted(time_us, processing_params.start_us)
        if processing_params.end_us is not None:
            end_idx = np.searchsorted(time_us, processing_params.end_us)
        
        # Start with full original data and zero regions outside bounds
        windowed_data = self.data.copy()
        original_length = len(self.data)  # For proper normalization
        
        # Zero out regions outside start_us/end_us bounds
        if start_idx > 0:
            windowed_data[:start_idx] = 0.0
        if end_idx < len(windowed_data):
            windowed_data[end_idx:] = 0.0
        
        # Step 2: Apply exponential filtering ONLY to active (non-zeroed) region
        if processing_params.expf_us is not None and start_idx < end_idx:
            # Calculate decay relative to active region time
            active_time_us = time_us[start_idx:end_idx]
            relative_time_us = active_time_us - active_time_us[0]
            decay = np.exp(-relative_time_us / processing_params.expf_us)
            windowed_data[start_idx:end_idx] *= decay
        
        # Step 3: Apply window function ONLY to active region
        if processing_params.winf is not None and start_idx < end_idx:
            window = spsig.get_window(processing_params.winf, end_idx - start_idx)
            windowed_data[start_idx:end_idx] *= window
        
        # Step 4: Remove DC component from active region (after windowing)
        if processing_params.rdc and start_idx < end_idx:
            active_data = windowed_data[start_idx:end_idx]
            dc_offset = np.mean(active_data)
            windowed_data[start_idx:end_idx] -= dc_offset
        
        # Step 5: Zero padding to full-length processed data
        final_data = windowed_data
        if processing_params.zpf > 0:
            # Handle edge case of empty data
            if len(final_data) == 0:
                # For empty data, create minimal padded array
                n_padded = 2 ** processing_params.zpf
            else:
                # Pad to next power of 2, then extend by 2^zpf
                n_padded = 2 ** (int(np.log2(len(final_data))) + 1 + processing_params.zpf)
            fid_padded = np.zeros(n_padded, dtype=float)
            fid_padded[:len(final_data)] = final_data
            final_data = fid_padded
        
        return PreprocessedFID(
            data=final_data,
            spacing=self.spacing,
            probe_freq_mhz=self.probe_freq_mhz,
            sideband=self.sideband,
            original_length=original_length,
            processing_params=processing_params,
            metadata={'source_fid_metadata': self.metadata}
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
    
    def __init__(self, freq_array: np.ndarray, complex_spectrum: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None):
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
        self._magnitude_spectrum = None
        self._freq_step = None
    
    # NOTE: from_fid class method removed - use proper three-stage workflow:
    # 1. preprocessed = fid.preprocess(**params)
    # 2. spectrum, freqs = preprocessed.compute_fft()
    # 3. complex_ft = ComplexFT.from_spectrum(spectrum, freqs)
    
    @classmethod
    def from_spectrum(cls, complex_spectrum: np.ndarray, freq_array: np.ndarray,
                     metadata: Optional[Dict[str, Any]] = None) -> 'ComplexFT':
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
        return cls(freq_array=freq_array, complex_spectrum=complex_spectrum,
                  metadata=metadata)
    
    @property
    def magnitude_spectrum(self) -> np.ndarray:
        """Magnitude spectrum (cached)."""
        if self._magnitude_spectrum is None:
            self._magnitude_spectrum = np.abs(self.complex_spectrum)
        return self._magnitude_spectrum
    
    @property
    def real_spectrum(self) -> np.ndarray:
        """Real component of spectrum."""
        return np.real(self.complex_spectrum)
    
    @property
    def imag_spectrum(self) -> np.ndarray:
        """Imaginary component of spectrum."""
        return np.imag(self.complex_spectrum)
    
    @property
    def freq_step(self) -> float:
        """Frequency step in MHz (cached)."""
        if self._freq_step is None:
            self._freq_step = np.mean(np.diff(self.freq_array))
        return self._freq_step
    
    @property
    def freq_range(self) -> Tuple[float, float]:
        """Frequency range (min, max) in MHz."""
        return float(np.min(self.freq_array)), float(np.max(self.freq_array))
    
    @property
    def n_points(self) -> int:
        """Number of frequency points."""
        return len(self.freq_array)
    
    def extract_window(self, freq_min: float, freq_max: float) -> 'SpectralWindow':
        """Extract a frequency window."""
        mask = (self.freq_array >= freq_min) & (self.freq_array <= freq_max)
        
        if not np.any(mask):
            raise ValueError(f"No data points in frequency range [{freq_min}, {freq_max}] MHz")
        
        return SpectralWindow(
            parent_ft=self,
            freq_array=self.freq_array[mask],
            complex_spectrum=self.complex_spectrum[mask],
            freq_range=(freq_min, freq_max)
        )
    
    def trim_to_range(self, freq_min: float, freq_max: float) -> 'ComplexFT':
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
            raise ValueError(f"No data points in frequency range [{freq_min}, {freq_max}] MHz")
        
        # Create new ComplexFT with trimmed data
        return ComplexFT(
            freq_array=self.freq_array[mask],
            complex_spectrum=self.complex_spectrum[mask],
            metadata={**self.metadata, 'trimmed_range': (freq_min, freq_max)}
        )


class Peak:
    """
    Pre-fitting detected peak representation.
    
    Represents peaks detected in the spectrum before fitting, used for
    initial parameter guesses.
    """
    
    def __init__(self, frequency: float, intensity: float,
                 index: Optional[int] = None, snr: Optional[float] = None,
                 noise_std_local: Optional[float] = None,
                 classification: Optional[Union[str, PeakClassification]] = None,
                 **properties):
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
        classification_str = self.classification.value if self.classification else 'unclassified'
        snr_str = f"{self.snr:.1f}" if self.snr is not None else "None"
        return f"Peak(freq={self.frequency:.3f} MHz, intensity={self.intensity:.2e}, SNR={snr_str}, {classification_str})"
    
    def __lt__(self, other):
        """Sort peaks by intensity (strongest first)."""
        return self.intensity > other.intensity


@dataclass
class FittedPeak:
    """
    Post-fitting peak results with fitted parameters.
    
    Represents the results of fitting a single peak, including
    fitted parameters, uncertainties, and quality metrics.
    """
    peak_id: Union[str, int]
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
    
    # Additional fitted parameters
    extra_parameters: Dict[str, float] = field(default_factory=dict)
    extra_errors: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"FittedPeak(id={self.peak_id}, freq={self.frequency_mhz:.6f}±{self.frequency_error:.6f} MHz, amp={self.amplitude:.2e})"


class SpectralWindow:
    """
    Analysis window - a subset of ComplexFT data for focused analysis.
    
    Represents a frequency range extracted from a ComplexFT for
    targeted peak detection and fitting operations.
    """
    
    def __init__(self, parent_ft: ComplexFT, freq_array: np.ndarray, 
                 complex_spectrum: np.ndarray, freq_range: Tuple[float, float],
                 window_id: Optional[str] = None, peaks: Optional[List[Peak]] = None):
        """
        Initialize SpectralWindow.
        
        Parameters
        ----------
        parent_ft : ComplexFT
            Parent ComplexFT object this window was extracted from
        freq_array : np.ndarray
            Frequency array for this window
        complex_spectrum : np.ndarray
            Complex spectrum data for this window
        freq_range : tuple
            (min_freq, max_freq) in MHz
        window_id : str, optional
            Identifier for this window
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
        return np.abs(self.complex_spectrum)
    
    @property
    def real_spectrum(self) -> np.ndarray:
        """Real component."""
        return np.real(self.complex_spectrum)
    
    @property
    def imag_spectrum(self) -> np.ndarray:
        """Imaginary component."""
        return np.imag(self.complex_spectrum)
    
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
            raise ValueError(f"Peak frequency {peak.frequency} MHz outside window range {self.freq_range}")
        
        self.peaks.append(peak)
    
    def get_peaks_by_classification(self, classification: Union[str, PeakClassification]) -> List[Peak]:
        """Get peaks with specified classification."""
        if isinstance(classification, str):
            classification = PeakClassification(classification)
        return [peak for peak in self.peaks if peak.classification == classification]
    
    def __repr__(self) -> str:
        return (f"SpectralWindow(id={self.window_id}, "
                f"range=[{self.freq_range[0]:.1f}, {self.freq_range[1]:.1f}] MHz, "
                f"n_points={self.n_points}, n_peaks={self.n_peaks})")


class FittingResult:
    """
    Container for fitting results from analysis of SpectralWindow(s).
    
    Stores fitted parameters, quality metrics, and diagnostic information.
    """
    
    def __init__(self, success: bool = False, fitted_spectrum: Optional[np.ndarray] = None,
                 cost: float = np.inf, iterations: int = 0, aic: float = np.inf,
                 reduced_chi2: float = np.inf, window: Optional[SpectralWindow] = None):
        """Initialize FittingResult."""
        self.success = success
        self.fitted_spectrum = fitted_spectrum
        self.cost = cost
        self.iterations = iterations
        self.aic = aic
        self.reduced_chi2 = reduced_chi2
        self.window = window
        
        # Fitted peaks
        self.fitted_peaks: List[FittedPeak] = []
        
        # Global parameters (shared across peaks)
        self.shared_parameters: Dict[str, Dict[str, float]] = {}
        self.fixed_parameters: Dict[str, Dict[str, float]] = {}
        
        # Diagnostics
        self.residuals: Optional[np.ndarray] = None
        self.quality_metrics: Dict[str, float] = {}
    
    def add_fitted_peak(self, fitted_peak: FittedPeak) -> None:
        """Add a fitted peak result."""
        self.fitted_peaks.append(fitted_peak)
    
    def set_shared_parameter(self, name: str, value: float, error: Optional[float] = None,
                            peak_ids: Optional[List] = None) -> None:
        """Set a parameter shared across multiple peaks."""
        self.shared_parameters[name] = {
            'value': value,
            'error': error,
            'peak_ids': peak_ids or [p.peak_id for p in self.fitted_peaks]
        }
    
    def set_fixed_parameter(self, name: str, value: float, 
                           peak_ids: Optional[List] = None) -> None:
        """Set a parameter held fixed during fitting."""
        self.fixed_parameters[name] = {
            'value': value,
            'peak_ids': peak_ids or [p.peak_id for p in self.fitted_peaks]
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
        return (f"FittingResult(status={status}, n_peaks={self.n_peaks_fitted}, "
                f"cost={self.cost:.2e}, AIC={self.aic:.2f})")


class FTMWData:
    """
    Top-level container for complete FTMW experiment data.
    
    Contains FID (time domain), ComplexFT (frequency domain), and 
    analysis windows with detected peaks. Provides the main interface
    for FTMW data processing workflows.
    """
    
    def __init__(self, fid: FID, experiment_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
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
            self._complex_ft = ComplexFT.from_spectrum(spectrum, freqs, 
                                                      metadata={'processing_params': preprocessed.processing_params})
        return self._complex_ft
    
    def compute_ft(self, **ft_kwargs) -> ComplexFT:
        """Compute ComplexFT with custom parameters using three-stage workflow."""
        # Use three-stage workflow with custom parameters
        preprocessed = self.fid.preprocess(**ft_kwargs)
        spectrum, freqs = preprocessed.compute_fft()
        self._complex_ft = ComplexFT.from_spectrum(spectrum, freqs, 
                                                  metadata={'processing_params': preprocessed.processing_params})
        return self._complex_ft
    
    def create_spectral_window(self, freq_min: float, freq_max: float, 
                              window_id: Optional[str] = None) -> SpectralWindow:
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
        return (f"FTMWData(id={self.experiment_id}, "
                f"fid_duration={self.fid.duration_us:.1f} μs, "
                f"n_windows={self.n_windows}, "
                f"n_fitted_peaks={self.total_fitted_peaks})")


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
        n_hard = sum(
            1 for w in self.windows if w.difficulty == WindowDifficulty.HARD
        )
        return (
            f"WindowPlan(n_windows={self.n_windows}, hard={n_hard}, "
            f"n_batches={self.n_batches}, "
            f"n_dependencies={len(self.dependency_edges)})"
        )