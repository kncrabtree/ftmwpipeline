#!/usr/bin/env python3
"""
Simple test script for noise estimation on BlackChirp experiment 2638.

Easy to modify and rerun with different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import ftmwpipeline modules
from ftmwpipeline.io.experimental_formats import load_blackchirp_experiment
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive
from ftmwpipeline.io.result_serialization import save_pipeline_cache
from ftmwpipeline.visualization.noise_visualization import (
    plot_noise_estimation,
    plot_noise_estimation_from_cache
)


def main():
    """Test noise estimation with configurable parameters."""
    
    # =========================
    # ALGORITHM PARAMETERS - MODIFY THESE
    # =========================
    algorithm_params = {
        "min_bin_fraction": 1/32,
        "skew_target": 0.631,     # More aggressive skewness target to remove more signal points
        "inc": 0.01,            # Increased for efficiency (was 0.005)
        "smoothing_window_mhz": 1000.,
        "verbose": False,  # Disable detailed output for clean results
    }
    
    print("Testing Variance-Based Noise Estimation")
    print("=" * 40)
    print(f"Bin fraction: {algorithm_params['min_bin_fraction']}")
    print(f"Skew target: {algorithm_params['skew_target']}")
    print()
    
    # Load and process data
    print("Loading experiment 2638...")
    data_path = Path(__file__).parent.parent.parent / "examples/blackchirp_data/2638"
    
    ftmw_data = load_blackchirp_experiment(str(data_path), fid_index=0)
    complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)
    trimmed_ft = complex_ft.trim_to_range(26500, 40000)  # Activity region in MHz
    
    frequencies = trimmed_ft.freq_array
    magnitudes = trimmed_ft.magnitude_spectrum
    
    print(f"✓ Loaded {len(frequencies):,} frequency points")
    print(f"✓ Range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} MHz")
    print()
    
    # Run noise estimation
    print("Running noise estimation...")
    result = estimate_noise_adaptive(frequencies, magnitudes, **algorithm_params)
    
    # Print results
    print("Results:")
    print(f"  Number of bins: {result.bin_info['n_bins']}")
    print(f"  Noise fraction: {result.bin_info['noise_fraction']:.3f}")
    print(f"  RMS mean: {np.mean(result.rms_noise):.2e}")
    print(f"  RMS std: {np.std(result.rms_noise):.2e}")
    print()
    
    # Cache the results for future visualization
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    print("Caching pipeline results...")
    cache_file = save_pipeline_cache(
        "exp_2638", 
        complex_ft=trimmed_ft, 
        noise_result=result,
        cache_dir=str(cache_dir)
    )
    print(f"✓ Cached to: {cache_file}")
    
    # Save output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Demonstrate both visualization APIs
    print("\n1. Creating plot using direct objects...")
    fig1 = plot_noise_estimation(
        frequencies, magnitudes, result,
        title="Direct API: Noise Estimation Diagnostics",
        figsize=(14, 10)
    )
    output_file1 = output_dir / "noise_direct_api.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"✓ Saved direct API plot: {output_file1}")
    
    print("\n2. Creating plot using cached data...")
    fig2 = plot_noise_estimation_from_cache(
        "exp_2638",
        cache_dir=str(cache_dir),
        title="Cache API: Noise Estimation Diagnostics",
        figsize=(14, 10)
    )
    output_file2 = output_dir / "noise_cache_api.png"
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved cache API plot: {output_file2}")
    
    print(f"\n✅ Dual API demonstration complete!")
    print(f"   Both plots should be identical")
    print(f"   Cache enables visualization without recomputation")
    
    # Show the cache-based plot (as an example)
    plt.show()


if __name__ == "__main__":
    main()