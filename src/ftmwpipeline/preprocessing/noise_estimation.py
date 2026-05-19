"""
Noise estimation for FTMW spectroscopy data.

This module provides improved noise estimation algorithms based on statistical analysis
of magnitude spectra, using adaptive binning and RMS metrics for robust noise characterization.
"""

import logging

import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as spsm
import scipy.signal as spsig
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    inc: float = 0.01,  # Increased for efficiency
    min_bin_fraction: float = 1/64,  # Minimum bin size as fraction of total data
    smoothing_window_mhz: Optional[float] = None,
    min_noise_fraction: float = 2/3,
    verbose: bool = False,
) -> NoiseResult:
    """
    Estimate frequency-dependent noise using variance-based adaptive binning.
    
    This function improves upon the original estimate_baseline_noise by:
    1. Using RMS instead of standard deviation for noise metric
    2. Implementing recursive binary subdivision based on local variance
    3. Enforcing noise fraction constraints to maintain statistical quality
    4. Using bin-aware smoothing for stable RMS estimates
    5. Returning detailed information about the noise model
    
    The algorithm uses recursive subdivision to create bins, stopping when:
    - Bins would be too small (< min_bin_fraction of data)
    - Local variance is low (stable regions)
    - Noise fraction would be insufficient (< min_noise_fraction)
    
    Within each bin, noise points are identified by iteratively removing 
    high-magnitude points until skewness approaches the target value.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values (MHz)
    magnitudes : np.ndarray  
        Magnitude spectrum values
    skew_target : float, default=0.631
        Target skewness for noise identification (Rayleigh ≈ 0.631)
    inc : float, default=0.01
        Fraction of points to remove per iteration during skewness optimization
    min_bin_fraction : float, default=1/64
        Minimum bin size as fraction of total data length
    smoothing_window_mhz : float, optional
        RMS smoothing window size in MHz. If None, uses 2× average bin width in frequency
    min_noise_fraction : float, default=2/3
        Minimum fraction of points that must be identified as noise per bin
    verbose : bool, default=False
        Whether to print detailed subdivision decisions and statistics
        
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
    
    # Use variance-based adaptive binning strategy - returns bin edges and cached results
    bin_edges, noise_results_cache = _compute_variance_based_bins(
        magnitudes, min_bin_size, skew_target, inc, min_noise_fraction, frequencies, verbose
    )
    
    # Process each bin to identify noise points using cached results from subdivision
    noise_mask = np.zeros(n_points, dtype=bool)
    bin_weights = np.zeros(n_points)
    
    for i in range(len(bin_edges) - 1):
        start_idx = bin_edges[i]
        end_idx = bin_edges[i + 1]
        
        # Use cached noise filtering results from subdivision
        cache_key = (start_idx, end_idx)
        if cache_key in noise_results_cache:
            noise_indices, _ = noise_results_cache[cache_key]
        else:
            # Fallback: compute if not cached (shouldn't happen normally)
            bin_magnitudes = magnitudes[start_idx:end_idx]
            bin_indices = np.arange(start_idx, end_idx)
            noise_indices, stats = _filter_by_skewness_cached(
                bin_magnitudes, bin_indices, skew_target, inc, cache_key
            )
        
        # Update global noise mask
        weights = np.ones(len(noise_indices))
        noise_mask[noise_indices] = True
        bin_weights[noise_indices] = weights
    
    # Compute RMS noise estimate with smoothing
    if smoothing_window_mhz is None:
        # Default: use 2× average bin width in frequency
        n_bins = len(bin_edges) - 1
        freq_range = abs(frequencies[-1] - frequencies[0])
        avg_bin_width_mhz = freq_range / n_bins
        smoothing_window_mhz = 2.0 * avg_bin_width_mhz
    
    # Convert MHz to points
    freq_step = abs(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 1.0
    smoothing_window_points = max(int(smoothing_window_mhz / freq_step), 10)  # At least 10 points
    
    rms_noise = _compute_rms_noise_smoothed(
        frequencies, magnitudes, noise_mask, bin_weights, smoothing_window_points, bin_edges
    )
    
    # Compile diagnostic information
    bin_info = {
        "bin_edges": np.array(bin_edges),
        "n_bins": len(bin_edges) - 1,
        "algorithm": "variance_based_subdivision",
        "noise_fraction": np.sum(noise_mask) / n_points,
        "smoothing_window_mhz": smoothing_window_mhz,
        "smoothing_window_points": smoothing_window_points,
    }
    
    return NoiseResult(rms_noise, noise_mask, bin_info)


def _compute_variance_based_bins(
    magnitudes: np.ndarray, min_bin_size: int, 
    skew_target: float, inc: float, min_noise_fraction: float,
    frequencies: np.ndarray = None, verbose: bool = False
) -> Tuple[list, dict]:
    """Compute bin edges using recursive binary subdivision based on variance.
    
    Returns:
        tuple: (bin_edges, noise_results_cache) where cache contains precomputed
               noise filtering results for all final bins.
    """
    
    # Global cache for statistical computations
    _stats_cache = {}  # (start_idx, end_idx) -> scipy.stats.describe result
    _noise_results_cache = {}  # (start_idx, end_idx) -> (noise_indices, final_stats)
    
    def get_bin_stats(start_idx: int, end_idx: int):
        """Get cached scipy.stats.describe result for a bin region."""
        cache_key = (start_idx, end_idx)
        if cache_key not in _stats_cache:
            bin_data = magnitudes[start_idx:end_idx]
            _stats_cache[cache_key] = spsm.describe(bin_data)
        return _stats_cache[cache_key]
    
    def get_noise_result(start_idx: int, end_idx: int):
        """Get cached noise filtering result for a bin."""
        cache_key = (start_idx, end_idx)
        if cache_key not in _noise_results_cache:
            bin_magnitudes = magnitudes[start_idx:end_idx]
            bin_indices = np.arange(start_idx, end_idx)
            noise_indices, final_stats = _filter_by_skewness_cached(
                bin_magnitudes, bin_indices, skew_target, inc, cache_key
            )
            _noise_results_cache[cache_key] = (noise_indices, final_stats)
        return _noise_results_cache[cache_key]
    
    def should_subdivide(start_idx: int, end_idx: int) -> bool:
        """Determine if a region should be subdivided based on size, variance, and noise fraction."""
        region_size = end_idx - start_idx
        
        if verbose:
            freq_info = ""
            if frequencies is not None:
                start_freq = frequencies[start_idx] 
                end_freq = frequencies[end_idx-1] if end_idx < len(frequencies) else frequencies[-1]
                freq_info = f" freqs=[{start_freq:.0f}:{end_freq:.0f}]"
            logger.debug(f"  Checking subdivision for [{start_idx}:{end_idx}]{freq_info} (size={region_size})")
        
        # Don't subdivide if too small
        if region_size < 2 * min_bin_size:
            if verbose:
                logger.debug(f"    → TOO SMALL: {region_size} < {2 * min_bin_size}")
            return False
        
        # Check if left and right halves have substantially different variances
        mid_idx = (start_idx + end_idx) // 2
        if mid_idx - start_idx < min_bin_size:
            mid_idx = start_idx + min_bin_size
        if end_idx - mid_idx < min_bin_size:
            mid_idx = end_idx - min_bin_size
            
        # Only proceed if valid mid point exists
        if not (start_idx < mid_idx < end_idx):
            if verbose:
                logger.debug(f"    → INVALID MID: start={start_idx}, mid={mid_idx}, end={end_idx}")
            return False
            
        # Use POST-FILTERING statistics for subdivision decisions
        # Get the filtered noise results for each half
        left_noise_indices, left_filtered_stats = get_noise_result(start_idx, mid_idx)
        right_noise_indices, right_filtered_stats = get_noise_result(mid_idx, end_idx)
        
        # Extract filtered data statistics
        left_filtered_mean = left_filtered_stats.mean if left_filtered_stats else 0
        left_filtered_var = left_filtered_stats.variance if left_filtered_stats else 0
        right_filtered_mean = right_filtered_stats.mean if right_filtered_stats else 0
        right_filtered_var = right_filtered_stats.variance if right_filtered_stats else 0
        
        # Calculate percentage differences and statistical significance
        mean_diff_pct = abs(left_filtered_mean - right_filtered_mean) / (0.5 * (left_filtered_mean + right_filtered_mean) + 1e-10) * 100
        var_diff_pct = abs(left_filtered_var - right_filtered_var) / (0.5 * (left_filtered_var + right_filtered_var) + 1e-10) * 100
        
        # Statistical significance tests
        # 1. Z-test for difference in means
        left_se = np.sqrt(left_filtered_var / len(left_noise_indices)) if len(left_noise_indices) > 0 else 1e-10
        right_se = np.sqrt(right_filtered_var / len(right_noise_indices)) if len(right_noise_indices) > 0 else 1e-10
        pooled_se = np.sqrt(left_se**2 + right_se**2)
        z_score = abs(left_filtered_mean - right_filtered_mean) / (pooled_se + 1e-10)
        
        # 2. F-test for difference in variances
        # F = larger_variance / smaller_variance, with df1, df2 = n1-1, n2-1
        f_stat = max(left_filtered_var, right_filtered_var) / (min(left_filtered_var, right_filtered_var) + 1e-10)
        df1 = max(len(left_noise_indices) - 1, 1)
        df2 = max(len(right_noise_indices) - 1, 1)
        
        # Critical F-value for 5-sigma equivalent (p ≈ 5.7e-7, very conservative)
        # For practical purposes, use F > 2.0 as significant difference threshold
        f_critical = 2.0  # Conservative threshold for practical significance
        
        if verbose:
            logger.debug(f"    → FILTERED VARIANCES: left={left_filtered_var:.2e}, right={right_filtered_var:.2e}, diff={var_diff_pct:.1f}%")
            logger.debug(f"    → FILTERED MEANS: left={left_filtered_mean:.2e}, right={right_filtered_mean:.2e}, diff={mean_diff_pct:.1f}%")
            logger.debug(f"    → STATISTICAL TESTS: z={z_score:.1f}, F={f_stat:.1f} (df1={df1}, df2={df2})")
        
        # Decision criteria: require significant difference in EITHER means OR variances
        pct_threshold = 20.0     # Require at least 20% difference
        sigma_threshold = 5.0    # Require at least 5-sigma significance for means
        
        # Test significance for means (percentage OR statistical)
        mean_significant = mean_diff_pct >= pct_threshold or z_score >= sigma_threshold
        
        # Test significance for variances (percentage OR F-test)
        var_significant = var_diff_pct >= pct_threshold or f_stat >= f_critical
        
        if not (mean_significant or var_significant):
            if verbose:
                logger.debug(f"    → SIMILAR REGIONS: mean(diff={mean_diff_pct:.1f}%, z={z_score:.1f}) and var(diff={var_diff_pct:.1f}%, F={f_stat:.1f}) both non-significant")
            return False
        else:
            if verbose:
                mean_reason = f"diff={mean_diff_pct:.1f}%" if mean_diff_pct >= pct_threshold else f"z={z_score:.1f}"
                var_reason = f"diff={var_diff_pct:.1f}%" if var_diff_pct >= pct_threshold else f"F={f_stat:.1f}"
                logger.debug(f"    → SIGNIFICANT DIFFERENCE: mean_sig={mean_significant} ({mean_reason}), var_sig={var_significant} ({var_reason})")
        
        # Calculate noise fractions (we already have the noise indices from above)
        left_noise_fraction = len(left_noise_indices) / (mid_idx - start_idx)
        right_noise_fraction = len(right_noise_indices) / (end_idx - mid_idx)
        
        if verbose:
            logger.debug(f"    → NOISE FRACTIONS: left={left_noise_fraction:.3f}, right={right_noise_fraction:.3f} (need >={min_noise_fraction:.3f})")
        
        # Only subdivide if BOTH halves have sufficient noise
        can_subdivide = left_noise_fraction >= min_noise_fraction and right_noise_fraction >= min_noise_fraction
        if verbose:
            logger.debug(f"    → DECISION: {'SUBDIVIDE' if can_subdivide else 'KEEP AS SINGLE BIN'}")
        return can_subdivide
    
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
    
    if verbose:
        logger.debug(f"Starting subdivision with {n_points} points, min_bin_size={min_bin_size}")
        logger.debug(f"Parameters: CV_threshold=0.1, min_noise_fraction={min_noise_fraction}")
        if frequencies is not None:
            logger.debug(f"Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} MHz (descending: {frequencies[0] > frequencies[-1]})")
    
    # Perform recursive subdivision
    recursive_subdivide(0, n_points, bin_edges)
    
    # Add final edge and sort
    bin_edges.append(n_points)
    bin_edges = sorted(set(bin_edges))  # Remove duplicates and sort
    
    if verbose:
        logger.debug(f"Final subdivision: {len(bin_edges)-1} bins created")
        bin_sizes = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
        logger.debug(f"Bin sizes: min={min(bin_sizes)}, max={max(bin_sizes)}, avg={sum(bin_sizes)/len(bin_sizes):.1f}")
    
    # No overlap needed with noise fraction validation
    return bin_edges, _noise_results_cache




def _filter_by_skewness_cached(
    bin_magnitudes: np.ndarray, 
    bin_indices: np.ndarray, 
    skew_target: float, 
    inc: float,
    cache_key: tuple
) -> Tuple[np.ndarray, object]:
    """Filter bin data by iteratively removing high values until target skewness is reached.
    
    Returns both noise indices and the final scipy.stats.describe result for caching.
    """
    
    cutoff = 0.0
    current_data = bin_magnitudes.copy()
    desc = None
    
    while True:
        # Remove top percentile of data
        filtered_data = spsm.trim(current_data, (0, cutoff), relative=True)
        
        if len(filtered_data) < 3:  # Need minimum points for skewness calculation
            break
            
        # Calculate statistics - this becomes our cached result
        desc = spsm.describe(filtered_data)
        
        if desc.skewness < skew_target:
            # Create mask for points that passed the filter
            if cutoff == 0:
                noise_indices = bin_indices
            else:
                threshold = np.percentile(current_data, 100 * (1 - cutoff))
                mask = current_data <= threshold
                noise_indices = bin_indices[mask]
            
            return noise_indices, desc
        else:
            cutoff += inc
            
        # Safety check to prevent infinite loop
        if cutoff >= 0.9:
            # If we can't achieve target skewness, use what we have
            threshold = np.percentile(current_data, 10)  # Keep bottom 10%
            mask = current_data <= threshold
            noise_indices = bin_indices[mask] if np.any(mask) else bin_indices[:1]
            return noise_indices, desc if desc is not None else spsm.describe(current_data[:1])
    
    # Fallback: return at least some points
    fallback_indices = bin_indices[:max(1, len(bin_indices) // 10)]
    fallback_desc = spsm.describe(bin_magnitudes[fallback_indices - bin_indices[0]])
    return fallback_indices, fallback_desc


def compute_rms_noise_convolution(
    frequencies: np.ndarray,
    magnitudes: np.ndarray, 
    noise_mask: np.ndarray,
    bl_bin: int,
) -> np.ndarray:
    """Core RMS noise computation using convolution.
    
    This is the modularized core algorithm that can be used by both the original
    noise estimation and the deserialization reconstruction to ensure bit-perfect
    reproduction.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Full frequency array (MHz)
    magnitudes : np.ndarray
        Full magnitude array
    noise_mask : np.ndarray
        Boolean mask indicating which points are noise
    bl_bin : int
        Smoothing window size in points
    
    Returns
    -------
    np.ndarray
        RMS noise array interpolated to the full frequency grid
    """
    # Extract noise points
    noise_magnitudes = magnitudes[noise_mask]
    noise_frequencies = frequencies[noise_mask]
    
    if len(noise_magnitudes) == 0:
        # Fallback: use minimum values if no noise points identified
        return np.full_like(frequencies, np.min(magnitudes))
    
    # Ensure bl_bin is valid
    bl_bin = max(bl_bin, 1)
    
    # Pad noise data (exact same logic as original)
    if len(noise_magnitudes) >= bl_bin // 2:
        bl_pre = noise_magnitudes[0 : bl_bin // 2]
        bl_post = noise_magnitudes[-(bl_bin // 2):]
    else:
        bl_pre = noise_magnitudes[:1]
        bl_post = noise_magnitudes[-1:]
    
    noise_padded = np.concatenate([bl_pre, noise_magnitudes, bl_post])
    
    # Compute RMS using exact convolution approach
    rms_values_padded = np.sqrt(
        spsig.oaconvolve(
            noise_padded ** 2, np.ones(bl_bin) / bl_bin, mode="same"
        )
    )
    
    # Remove padding (exact same logic as original)
    rms_values = rms_values_padded[bl_bin // 2 : -(bl_bin // 2)]
    
    # Interpolate back to original frequency grid
    if len(noise_frequencies) == len(frequencies) and np.allclose(noise_frequencies, frequencies):
        # If all points were used, no interpolation needed
        return rms_values
    else:
        # Interpolate to full frequency grid
        if len(noise_frequencies) == 0 or len(rms_values) == 0:
            # Fallback if we have no data after padding removal
            return np.full_like(frequencies, np.min(magnitudes))
            
        # Ensure rms_values and noise_frequencies have same length
        if len(rms_values) != len(noise_frequencies):
            # Adjust lengths to match (this can happen due to padding edge effects)
            min_len = min(len(rms_values), len(noise_frequencies))
            rms_values = rms_values[:min_len]
            noise_frequencies = noise_frequencies[:min_len]
            
        if len(rms_values) == 0:
            return np.full_like(frequencies, np.min(magnitudes))
        
        if frequencies[0] > frequencies[-1]:
            # Handle descending frequency order
            rms_interpolated = np.interp(frequencies, noise_frequencies[::-1], rms_values[::-1])
        else:
            rms_interpolated = np.interp(frequencies, noise_frequencies, rms_values)
        
        return rms_interpolated


def _compute_rms_noise_smoothed(
    frequencies: np.ndarray,
    magnitudes: np.ndarray, 
    noise_mask: np.ndarray,
    weights: np.ndarray,
    smoothing_window: int,
    bin_edges: list,
) -> np.ndarray:
    """Compute smoothed RMS noise estimate from identified noise points.
    
    This function now delegates to the modularized core algorithm to ensure
    bit-perfect reconstruction during deserialization.
    """
    
    # Use the passed-in smoothing_window as bl_bin directly
    # This was the original behavior that I accidentally changed
    bl_bin = smoothing_window
    
    # Use the modularized core algorithm
    return compute_rms_noise_convolution(frequencies, magnitudes, noise_mask, bl_bin)


