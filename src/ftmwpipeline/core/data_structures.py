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

Extracted and adapted from bcfitting/newfitting/ codebase.
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
    start_us: Optional[float] = None  # Start time in μs (earlier points set to 0)
    end_us: Optional[float] = None    # End time in μs (later points set to 0)
    winf: Optional[str] = None        # Window function name (scipy.signal compatible)
    zpf: int = 1                      # Zero padding factor (powers of 2)
    rdc: bool = True                  # Remove DC component (subtract average)
    expf_us: Optional[float] = None   # Exponential decay filter time constant (μs)
    autoscale_MHz: Optional[float] = None  # Suppress noise near DC (MHz range)
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
    
    def ft(self, **processing_overrides) -> 'ComplexFT':
        """
        Compute Fourier transform of FID and return ComplexFT object.
        
        Uses processing parameters from self.processing unless overridden.
        
        Parameters
        ----------
        **processing_overrides
            Override any FIDProcessingParameters
            
        Returns
        -------
        ComplexFT
            ComplexFT object containing frequency domain data
        """
        # Get processing parameters (make a copy to avoid modifying original)
        proc = FIDProcessingParameters(
            start_us=self.processing.start_us,
            end_us=self.processing.end_us,
            winf=self.processing.winf,
            zpf=self.processing.zpf,
            rdc=self.processing.rdc,
            expf_us=self.processing.expf_us,
            autoscale_MHz=self.processing.autoscale_MHz,
            units_power=self.processing.units_power
        )
        
        # Apply overrides
        for key, value in processing_overrides.items():
            if hasattr(proc, key):
                setattr(proc, key, value)
        
        # Start with copy of FID data
        fid_data = self.data.copy()
        
        # Apply time windowing
        time_us = self.time_array_us()
        if proc.start_us is not None:
            mask = time_us < proc.start_us
            fid_data[mask] = 0
        if proc.end_us is not None:
            mask = time_us > proc.end_us
            fid_data[mask] = 0
        
        # Remove DC component
        if proc.rdc:
            fid_data -= np.mean(fid_data)
        
        # Apply exponential decay filter
        if proc.expf_us is not None:
            decay = np.exp(-time_us / proc.expf_us)
            fid_data *= decay
        
        # Apply window function
        if proc.winf is not None:
            window = spsig.get_window(proc.winf, len(fid_data))
            fid_data *= window
        
        # Store original FID length for normalization (before zero padding)
        original_fid_length = len(fid_data)
        
        # Zero padding
        if proc.zpf > 0:
            # Pad to next power of 2, then extend by 2^zpf
            n_padded = 2 ** (int(np.log2(len(fid_data))) + 1 + proc.zpf)
            fid_padded = np.zeros(n_padded, dtype=float)
            fid_padded[:len(fid_data)] = fid_data
            fid_data = fid_padded
        
        # Compute real FFT (since FID data is real)
        ft_data = sfft.rfft(fid_data)
        
        # Generate frequency axis using rfftfreq
        scope_freqs = sfft.rfftfreq(len(fid_data), d=self.spacing) / 1e6  # MHz
        
        # Convert to molecular frequencies
        mol_freqs = self.apply_molecular_frequency(scope_freqs)
        
        # Apply normalization (divide by original FID length, not padded length)
        ft_data /= original_fid_length
        
        # Apply scaling
        scale_factor = 10 ** proc.units_power
        ft_data *= scale_factor
        
        # Apply autoscale (suppress near-DC noise)
        if proc.autoscale_MHz is not None:
            dc_mask = np.abs(scope_freqs) < proc.autoscale_MHz
            ft_data[dc_mask] = 0
        
        # Create and return ComplexFT object
        return ComplexFT(
            freq_array=mol_freqs,
            complex_spectrum=ft_data,
            fid=self,
            metadata={'processing_params': proc}
        )


class ComplexFT:
    """
    Complex Fourier Transform frequency-domain data container.
    
    Contains the frequency-domain representation of FTMW data with
    associated experimental parameters. Computed from FID data.
    """
    
    def __init__(self, freq_array: np.ndarray, complex_spectrum: np.ndarray,
                 fid: Optional[FID] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ComplexFT object.
        
        Parameters
        ----------
        freq_array : np.ndarray
            Frequency array in MHz
        complex_spectrum : np.ndarray
            Complex spectrum data
        fid : FID, optional
            Source FID object
        metadata : dict, optional
            Additional metadata
        """
        self.freq_array = np.asarray(freq_array, dtype=float)
        self.complex_spectrum = np.asarray(complex_spectrum, dtype=complex)
        
        if len(self.freq_array) != len(self.complex_spectrum):
            raise ValueError("Frequency and spectrum arrays must have same length")
        
        self.fid = fid
        self.metadata = metadata or {}
        
        # Cached properties
        self._magnitude_spectrum = None
        self._freq_step = None
    
    @classmethod
    def from_fid(cls, fid: FID, **ft_kwargs) -> 'ComplexFT':
        """Create ComplexFT from FID using FT processing."""
        return fid.ft(**ft_kwargs)
    
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


class Peak:
    """
    Pre-fitting detected peak representation.
    
    Based on ClassifiedPeak from bcfitting. Represents peaks detected
    in the spectrum before fitting, used for initial parameter guesses.
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
    
    Based on FitResult from bcfitting. Stores fitted parameters,
    quality metrics, and diagnostic information.
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
        """Get or compute ComplexFT from FID."""
        if self._complex_ft is None:
            self._complex_ft = self.fid.ft()
        return self._complex_ft
    
    def compute_ft(self, **ft_kwargs) -> ComplexFT:
        """Compute ComplexFT with custom parameters."""
        self._complex_ft = self.fid.ft(**ft_kwargs)
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