"""
Noise estimation for FTMW spectroscopy data.

This module provides improved noise estimation algorithms based on statistical analysis
of magnitude spectra, using adaptive binning and RMS metrics for robust noise characterization.
"""

import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as spsm
import scipy.signal as spsig
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class NoiseResult:
    """Result container for noise estimation.
    
    Attributes:
        rms_noise: RMS noise estimate across the spectrum
        noise_mask: Boolean mask indicating which points were used for noise estimation
        bin_info: Dictionary containing binning information for diagnostics
    """
    rms_noise: np.ndarray
    noise_mask: np.ndarray
    bin_info: Dict[str, Union[np.ndarray, int, float]]


def estimate_noise_adaptive(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    skew_target: float = 0.631,
    inc: float = 0.005,
    min_bin_fraction: float = 1/64,  # Minimum bin size as fraction of total data
    smoothing_window: Optional[int] = None,
    adaptive_strategy: str = "variance_based",
    min_noise_fraction: float = 2/3,
) -> NoiseResult:
    """
    Estimate frequency-dependent noise using adaptive binning strategy.
    
    This function improves upon the original estimate_baseline_noise by:
    1. Using RMS instead of standard deviation for noise metric
    2. Implementing adaptive binning based on local signal characteristics
    3. Using overlapping bins to reduce boundary artifacts
    4. Returning detailed information about the noise model
    
    The algorithm identifies noise regions by iteratively removing high-magnitude
    points until the remaining data matches a Rayleigh distribution (skewness ≈ 0.631).
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values (Hz)
    magnitudes : np.ndarray  
        Magnitude spectrum values
    skew_target : float, default=0.631
        Target skewness for Rayleigh distribution
    inc : float, default=0.005
        Fraction of points to remove per iteration during skewness optimization
    min_bin_fraction : float, default=1/64
        Minimum bin size as fraction of total data length (e.g., 1/64 ≈ 1.6% of data)
    smoothing_window : int, optional
        Window size for smoothing. If None, uses adaptive sizing
    adaptive_strategy : str, default="variance_based"
        Strategy for adaptive binning: "variance_based", "fixed", or "frequency_dependent"
    min_noise_fraction : float, default=2/3
        Minimum fraction of points that must be identified as noise
        
    Returns:
    --------
    NoiseResult
        Container with RMS noise estimate, noise mask, and binning diagnostics
        
    Raises:
    -------
    ValueError
        If input arrays have different shapes or invalid parameters
    """
    
    # Input validation
    if frequencies.shape != magnitudes.shape:
        raise ValueError("frequencies and magnitudes must have the same shape")
    
    if frequencies.ndim != 1 or magnitudes.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    
    n_points = len(frequencies)
    min_bin_size = max(int(n_points * min_bin_fraction), 100)  # At least 100 points
    
    # Determine adaptive binning strategy
    if adaptive_strategy == "variance_based":
        bin_edges = _compute_variance_based_bins(
            magnitudes, min_bin_size, skew_target, inc, min_noise_fraction, frequencies
        )
    elif adaptive_strategy == "frequency_dependent":
        bin_edges = _compute_frequency_dependent_bins(
            frequencies, min_bin_size, max_bin_size, overlap_factor
        )
    else:  # fixed strategy (similar to original)
        n_bins = max(2, min(n_points // min_bin_size, n_points // max_bin_size))
        bin_edges = _compute_fixed_bins(n_points, n_bins, overlap_factor)
    
    # Process each bin to identify noise points
    noise_mask = np.zeros(n_points, dtype=bool)
    bin_weights = np.zeros(n_points)
    
    for i in range(len(bin_edges) - 1):
        start_idx = bin_edges[i]
        end_idx = bin_edges[i + 1]
        
        # Extract bin data
        bin_magnitudes = magnitudes[start_idx:end_idx]
        bin_indices = np.arange(start_idx, end_idx)
        
        # Apply skewness-based filtering to identify noise points
        noise_indices, _ = _filter_by_skewness(
            bin_magnitudes, bin_indices, skew_target, inc
        )
        
        # Update global noise mask (no overlap weighting needed)
        weights = np.ones(len(noise_indices))
        
        noise_mask[noise_indices] = True
        bin_weights[noise_indices] = weights
    
    # Compute RMS noise estimate with smoothing
    if smoothing_window is None:
        smoothing_window = max(min_bin_size // 4, 10)
    
    rms_noise = _compute_rms_noise_smoothed(
        frequencies, magnitudes, noise_mask, bin_weights, smoothing_window, bin_edges
    )
    
    # Compile diagnostic information
    bin_info = {
        "bin_edges": np.array(bin_edges),
        "n_bins": len(bin_edges) - 1,
        "adaptive_strategy": adaptive_strategy,
        "noise_fraction": np.sum(noise_mask) / n_points,
        "smoothing_window": smoothing_window,
    }
    
    return NoiseResult(rms_noise, noise_mask, bin_info)


def _compute_variance_based_bins(
    magnitudes: np.ndarray, min_bin_size: int, 
    skew_target: float, inc: float, min_noise_fraction: float,
    frequencies: np.ndarray = None
) -> list:
    """Compute bin edges using recursive binary subdivision based on variance."""
    
    def check_noise_fraction(start_idx: int, end_idx: int) -> float:
        """Check noise fraction for a potential bin."""
        bin_magnitudes = magnitudes[start_idx:end_idx]
        bin_indices = np.arange(start_idx, end_idx)
        noise_indices, _ = _filter_by_skewness(bin_magnitudes, bin_indices, skew_target, inc)
        return len(noise_indices) / len(bin_magnitudes)
    
    def should_subdivide(start_idx: int, end_idx: int) -> bool:
        """Determine if a region should be subdivided based on size, variance, and noise fraction."""
        region_size = end_idx - start_idx
        
        # Don't subdivide if too small
        if region_size < 2 * min_bin_size:
            return False
        
        # Check if variance is low enough to keep as single bin
        region_data = magnitudes[start_idx:end_idx]
        region_var = np.var(region_data)
        
        # Use coefficient of variation as stability metric
        region_mean = np.mean(region_data)
        cv = region_var / (region_mean**2 + 1e-10)  # Avoid division by zero
        
        # If variation is low, don't subdivide regardless of noise fraction
        if cv <= 0.1:
            return False
        
        # Check if subdivision would maintain adequate noise fraction
        mid_idx = (start_idx + end_idx) // 2
        if mid_idx - start_idx < min_bin_size:
            mid_idx = start_idx + min_bin_size
        if end_idx - mid_idx < min_bin_size:
            mid_idx = end_idx - min_bin_size
            
        # Only subdivide if valid mid point exists
        if not (start_idx < mid_idx < end_idx):
            return False
            
        # Check noise fractions for both potential halves
        left_noise_fraction = check_noise_fraction(start_idx, mid_idx)
        right_noise_fraction = check_noise_fraction(mid_idx, end_idx)
        
        # Only subdivide if BOTH halves have sufficient noise
        return left_noise_fraction >= min_noise_fraction and right_noise_fraction >= min_noise_fraction
    
    def recursive_subdivide(start_idx: int, end_idx: int, edges: list):
        """Recursively subdivide region if needed."""
        if not should_subdivide(start_idx, end_idx):
            return
        
        # Find subdivision point
        mid_idx = (start_idx + end_idx) // 2
        
        # Ensure we don't create bins that are too small
        if mid_idx - start_idx < min_bin_size:
            mid_idx = start_idx + min_bin_size
        if end_idx - mid_idx < min_bin_size:
            mid_idx = end_idx - min_bin_size
            
        # Add subdivision point if valid
        if start_idx < mid_idx < end_idx:
            edges.append(mid_idx)
            
            # Recursively subdivide both halves
            recursive_subdivide(start_idx, mid_idx, edges)
            recursive_subdivide(mid_idx, end_idx, edges)
    
    # Start with full range
    n_points = len(magnitudes)
    bin_edges = [0]
    
    # Perform recursive subdivision
    recursive_subdivide(0, n_points, bin_edges)
    
    # Add final edge and sort
    bin_edges.append(n_points)
    bin_edges = sorted(set(bin_edges))  # Remove duplicates and sort
    
    # No overlap needed with noise fraction validation
    return bin_edges


def _compute_frequency_dependent_bins(
    frequencies: np.ndarray, min_bin_size: int, max_bin_size: int, overlap_factor: float
) -> list:
    """Compute bin edges with frequency-dependent sizing."""
    n_points = len(frequencies)
    freq_range = frequencies[-1] - frequencies[0]
    
    # Use smaller bins at higher frequencies where noise characteristics may change more rapidly
    freq_norm = (frequencies - frequencies[0]) / freq_range
    
    # Linear scaling: smaller bins at high frequency
    size_factor = 1.0 - 0.5 * freq_norm  # 50% reduction at highest frequency
    adaptive_bin_sizes = min_bin_size + (max_bin_size - min_bin_size) * size_factor
    
    # Convert to bin edges similar to variance-based approach
    bin_edges = [0]
    current_pos = 0
    
    while current_pos < n_points - min_bin_size:
        local_idx = min(current_pos + min_bin_size // 2, n_points - 1)
        next_bin_size = int(adaptive_bin_sizes[local_idx])
        
        step_size = int(next_bin_size * (1 - overlap_factor))
        current_pos += step_size
        
        if current_pos < n_points:
            bin_edges.append(min(current_pos, n_points))
    
    if bin_edges[-1] < n_points:
        bin_edges.append(n_points)
    
    return bin_edges


def _compute_fixed_bins(n_points: int, n_bins: int, overlap_factor: float) -> list:
    """Compute bin edges for fixed binning strategy (similar to original algorithm)."""
    if overlap_factor == 0:
        # No overlap - simple division
        bin_size = n_points // n_bins
        return [i * bin_size for i in range(n_bins)] + [n_points]
    else:
        # With overlap
        effective_bin_size = n_points // (n_bins * (1 - overlap_factor) + overlap_factor)
        step_size = int(effective_bin_size * (1 - overlap_factor))
        
        bin_edges = [0]
        current_pos = 0
        
        while current_pos < n_points - effective_bin_size:
            current_pos += step_size
            bin_edges.append(min(current_pos, n_points))
        
        if bin_edges[-1] < n_points:
            bin_edges.append(n_points)
        
        return bin_edges


def _filter_by_skewness(
    bin_magnitudes: np.ndarray, 
    bin_indices: np.ndarray, 
    skew_target: float, 
    inc: float
) -> Tuple[np.ndarray, float]:
    """Filter bin data by iteratively removing high values until target skewness is reached."""
    
    cutoff = 0.0
    current_data = bin_magnitudes.copy()
    
    while True:
        # Remove top percentile of data
        filtered_data = spsm.trim(current_data, (0, cutoff), relative=True)
        
        if len(filtered_data) < 3:  # Need minimum points for skewness calculation
            break
            
        # Calculate skewness
        desc = spsm.describe(filtered_data)
        
        if desc.skewness < skew_target:
            # Create mask for points that passed the filter
            if cutoff == 0:
                noise_indices = bin_indices
            else:
                threshold = np.percentile(current_data, 100 * (1 - cutoff))
                mask = current_data <= threshold
                noise_indices = bin_indices[mask]
            
            return noise_indices, desc.skewness
        else:
            cutoff += inc
            
        # Safety check to prevent infinite loop
        if cutoff >= 0.9:
            # If we can't achieve target skewness, use what we have
            threshold = np.percentile(current_data, 10)  # Keep bottom 10%
            mask = current_data <= threshold
            noise_indices = bin_indices[mask] if np.any(mask) else bin_indices[:1]
            return noise_indices, desc.skewness if len(filtered_data) >= 3 else 0.0
    
    # Fallback: return at least some points
    return bin_indices[:max(1, len(bin_indices) // 10)], 0.0


def _compute_rms_noise_smoothed(
    frequencies: np.ndarray,
    magnitudes: np.ndarray, 
    noise_mask: np.ndarray,
    weights: np.ndarray,
    smoothing_window: int,
    bin_edges: list,
) -> np.ndarray:
    """Compute smoothed RMS noise estimate from identified noise points."""
    
    # Extract noise points
    noise_magnitudes = magnitudes[noise_mask]
    noise_frequencies = frequencies[noise_mask]
    noise_weights = weights[noise_mask]
    
    if len(noise_magnitudes) == 0:
        # Fallback: use minimum values if no noise points identified
        return np.full_like(frequencies, np.min(magnitudes))
    
    # Follow original algorithm pattern exactly:
    # 1. Use bl_bin = len(original_data) // 20 for window size
    # 2. Pad the noise-only data 
    # 3. Convolve on noise-only data
    # 4. Remove padding and interpolate
    
    # Use smoothing window based on bin structure: ~2× average bin width
    n_bins = len(bin_edges) - 1
    bl_bin = len(frequencies) // (n_bins // 2)  # 2× average bin width
    
    # Pad noise data (similar to original bl_pre, bl_post approach)
    bl_pre = noise_magnitudes[0 : bl_bin // 2] if len(noise_magnitudes) >= bl_bin // 2 else noise_magnitudes[:1]
    bl_post = noise_magnitudes[-(bl_bin // 2):] if len(noise_magnitudes) >= bl_bin // 2 else noise_magnitudes[-1:]
    noise_padded = np.concatenate([bl_pre, noise_magnitudes, bl_post])
    
    # Compute RMS using same convolution approach as original stdev calculation
    # RMS = sqrt(mean(x^2)) for noise data
    rms_values_padded = np.sqrt(
        spsig.oaconvolve(
            noise_padded ** 2, np.ones(bl_bin) / bl_bin, mode="same"
        )
    )
    
    # Remove padding (same as original)
    rms_values = rms_values_padded[bl_bin // 2 : -(bl_bin // 2)]
    
    # Interpolate back to original frequency grid
    if len(noise_frequencies) == len(frequencies) and np.allclose(noise_frequencies, frequencies):
        # If all points were used, no interpolation needed
        return rms_values
    else:
        # Interpolate to full frequency grid
        if frequencies[0] > frequencies[-1]:
            # Handle descending frequency order
            rms_interpolated = np.interp(frequencies, noise_frequencies[::-1], rms_values[::-1])
        else:
            rms_interpolated = np.interp(frequencies, noise_frequencies, rms_values)
        
        return rms_interpolated


# Legacy function name for backward compatibility
def estimate_baseline_noise(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy wrapper for estimate_noise_adaptive that returns baseline and noise.
    
    Since we've moved away from baseline estimation for time-domain fitting,
    this returns a zero baseline and the RMS noise estimate.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values
    magnitudes : np.ndarray
        Magnitude values
    **kwargs
        Additional arguments passed to estimate_noise_adaptive
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (baseline, noise) where baseline is zeros and noise is RMS estimate
    """
    
    result = estimate_noise_adaptive(frequencies, magnitudes, **kwargs)
    baseline = np.zeros_like(frequencies)  # No baseline estimation needed
    
    return baseline, result.rms_noise
