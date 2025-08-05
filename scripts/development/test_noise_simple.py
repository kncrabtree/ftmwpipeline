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
from ftmwpipeline.visualization.noise_diagnostics import plot_noise_estimation


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
    
    # Create visualization
    print("Creating diagnostic plot...")
    fig = plot_noise_estimation(
        frequencies, magnitudes, result,
        title="Noise Estimation: Variance-Based",
        figsize=(14, 10)
    )
    
    # Save to output directory (ignored by git)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "noise_test.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.show()


if __name__ == "__main__":
    main()