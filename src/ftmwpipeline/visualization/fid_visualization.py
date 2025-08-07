"""
FID visualization for data validation and quality assessment.

This module provides plotting functions for FID time-domain data,
enabling visual validation of loaded data and assessment of signal quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any
from pathlib import Path

from ..core.data_structures import FID
from ..io.fid_serialization import load_fid_cache


def plot_fid(fid: FID, 
             show_metadata: bool = True,
             time_units: str = 'us',
             voltage_units: str = 'V',
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (12, 8)) -> Any:
    """
    Create a comprehensive plot of FID time-domain data.
    
    Parameters
    ----------
    fid : FID
        FID object to plot
    show_metadata : bool, optional
        Whether to display metadata in the plot (default: True)
    time_units : str, optional
        Time axis units ('s', 'ms', 'us'), default 'us'
    voltage_units : str, optional
        Voltage axis units, default 'V'
    title : str, optional
        Plot title, auto-generated if None
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        Created figure object
        
    Examples
    --------
    >>> # Basic FID plot
    >>> fig = plot_fid(fid)
    >>> plt.show()
    
    >>> # Plot with custom title and save
    >>> fig = plot_fid(fid, title="Experiment 2638 - FID Data")
    >>> fig.savefig("fid_plot.png", dpi=300, bbox_inches='tight')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for FID visualization")
    
    # Set up time array and units
    if time_units == 's':
        time_data = fid.time_array()
        time_label = "Time (s)"
        time_scale = 1.0
    elif time_units == 'ms':
        time_data = fid.time_array() * 1000
        time_label = "Time (ms)"
        time_scale = 1000.0
    else:  # 'us'
        time_data = fid.time_array_us()
        time_label = "Time (μs)"
        time_scale = 1e6
    
    voltage_data = fid.data
    
    # Create figure with subplots
    if show_metadata:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Main FID plot
    ax1.plot(time_data, voltage_data, 'b-', linewidth=0.8, alpha=0.8)
    ax1.set_xlabel(time_label)
    ax1.set_ylabel(f"Voltage ({voltage_units})")
    ax1.grid(True, alpha=0.3)
    
    # Set title
    if title is None:
        title = f"FID Time-Domain Data ({fid.n_points:,} points, {fid.duration_us:.1f} μs)"
    ax1.set_title(title)
    
    # Add signal statistics
    voltage_rms = np.sqrt(np.mean(voltage_data**2))
    voltage_max = np.max(np.abs(voltage_data))
    
    stats_text = (
        f"RMS: {voltage_rms:.2e} {voltage_units}\n"
        f"Peak: {voltage_max:.2e} {voltage_units}\n"
        f"Points: {fid.n_points:,}\n"
        f"Duration: {fid.duration_us:.1f} μs"
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add metadata display if requested
    if show_metadata and ax2 is not None:
        ax2.axis('off')
        
        # Acquisition parameters
        acq_text = (
            f"Acquisition Parameters:\n"
            f"  Probe Frequency: {fid.probe_freq_mhz:.3f} MHz\n"
            f"  Sideband: {fid.sideband.value.title()}\n"
            f"  Time Spacing: {fid.spacing:.4e} s\n"
            f"  Shots Averaged: {fid.shots:,}"
        )
        
        # Processing parameters
        proc = fid.processing
        proc_text = (
            f"Processing Parameters:\n"
            f"  Zero Padding Factor: {proc.zpf}\n"
            f"  Remove DC: {proc.rdc}\n"
            f"  Exponential Filter: {proc.expf_us or 'None'} μs\n"
            f"  Window Function: {proc.winf or 'None'}"
        )
        
        # Source information
        source_text = ""
        if fid.metadata:
            source_format = fid.metadata.get('source_format', 'Unknown')
            source_path = fid.metadata.get('source_path', 'Unknown')
            if source_path != 'Unknown':
                source_path = Path(source_path).name  # Just show filename
            
            source_text = (
                f"Source Information:\n"
                f"  Format: {source_format}\n"
                f"  File: {source_path}\n"
                f"  Loader: {fid.metadata.get('loader_class', 'Unknown')}"
            )
        
        # Layout metadata text in columns
        ax2.text(0.02, 0.95, acq_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=9, family='monospace')
        ax2.text(0.35, 0.95, proc_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=9, family='monospace')
        if source_text:
            ax2.text(0.68, 0.95, source_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=9, family='monospace')
    
    plt.tight_layout()
    return fig


def plot_fid_from_cache(experiment_id: str,
                       cache_dir: str = "cache",
                       **plot_kwargs) -> Any:
    """
    Plot FID data directly from cache file.
    
    Parameters
    ----------
    experiment_id : str
        Experiment identifier
    cache_dir : str, optional
        Cache directory (default: "cache")
    **plot_kwargs
        Additional arguments passed to plot_fid()
        
    Returns
    -------
    matplotlib.figure.Figure
        Created figure object
        
    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    RuntimeError
        If loading or plotting fails
        
    Examples
    --------
    >>> # Plot FID from cache
    >>> fig = plot_fid_from_cache("exp_2638")
    >>> plt.show()
    
    >>> # Plot with custom options
    >>> fig = plot_fid_from_cache("exp_2638", 
    ...                          title="Cached FID Data",
    ...                          time_units='ms')
    """
    try:
        # Load FID from cache
        fid = load_fid_cache(experiment_id, cache_dir)
        
        # Create plot
        return plot_fid(fid, **plot_kwargs)
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"FID cache not found for experiment '{experiment_id}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to plot FID from cache: {e}") from e


def plot_fid_comparison(fids: list,
                       labels: Optional[list] = None,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (12, 8)) -> Any:
    """
    Create a comparison plot of multiple FID datasets.
    
    Parameters
    ----------
    fids : list of FID
        List of FID objects to compare
    labels : list of str, optional
        Labels for each FID dataset
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        Created figure object
        
    Examples
    --------
    >>> # Compare two FID datasets
    >>> fig = plot_fid_comparison([fid1, fid2], 
    ...                          labels=['Dataset 1', 'Dataset 2'])
    >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for FID visualization")
    
    if not fids:
        raise ValueError("At least one FID object is required")
    
    if labels is None:
        labels = [f"FID {i+1}" for i in range(len(fids))]
    
    if len(labels) != len(fids):
        raise ValueError("Number of labels must match number of FID objects")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fids)))
    
    for i, (fid, label, color) in enumerate(zip(fids, labels, colors)):
        time_data = fid.time_array_us()
        ax.plot(time_data, fid.data, color=color, linewidth=0.8, 
               label=label, alpha=0.8)
    
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title is None:
        title = f"FID Comparison ({len(fids)} datasets)"
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_fid_overview(fid: FID,
                     figsize: Tuple[float, float] = (15, 10)) -> Any:
    """
    Create a comprehensive overview plot with multiple views of FID data.
    
    This function creates a multi-panel plot showing:
    - Full time-domain FID
    - Zoomed view of early time decay
    - Voltage histogram
    - Basic statistics
    
    Parameters
    ----------
    fid : FID
        FID object to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        Created figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for FID visualization")
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    time_data = fid.time_array_us()
    voltage_data = fid.data
    
    # Full FID plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_data, voltage_data, 'b-', linewidth=0.8)
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Complete FID - {fid.n_points:,} points, {fid.duration_us:.1f} μs duration")
    ax1.grid(True, alpha=0.3)
    
    # Early time zoom (first 10% or 2 μs, whichever is smaller)
    zoom_time = min(fid.duration_us * 0.1, 2.0)
    zoom_mask = time_data <= zoom_time
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_data[zoom_mask], voltage_data[zoom_mask], 'r-', linewidth=1.0)
    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title(f"Early Time Detail (0-{zoom_time:.1f} μs)")
    ax2.grid(True, alpha=0.3)
    
    # Voltage histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(voltage_data, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel("Voltage (V)")
    ax3.set_ylabel("Count")
    ax3.set_title("Voltage Distribution")
    ax3.grid(True, alpha=0.3)
    
    # Statistics and metadata
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate statistics
    v_mean = np.mean(voltage_data)
    v_rms = np.sqrt(np.mean(voltage_data**2))
    v_std = np.std(voltage_data)
    v_max = np.max(voltage_data)
    v_min = np.min(voltage_data)
    v_pp = v_max - v_min
    
    stats_text = (
        f"Signal Statistics:\n"
        f"  Mean: {v_mean:.3e} V\n"
        f"  RMS: {v_rms:.3e} V\n"
        f"  Std Dev: {v_std:.3e} V\n"
        f"  Peak-to-Peak: {v_pp:.3e} V\n"
        f"  Min: {v_min:.3e} V\n"
        f"  Max: {v_max:.3e} V"
    )
    
    acq_text = (
        f"Acquisition:\n"
        f"  Probe Freq: {fid.probe_freq_mhz:.3f} MHz\n"
        f"  Sideband: {fid.sideband.value}\n"
        f"  Spacing: {fid.spacing:.4e} s\n"
        f"  Shots: {fid.shots:,}\n"
        f"  Duration: {fid.duration_us:.1f} μs\n"
        f"  Points: {fid.n_points:,}"
    )
    
    processing_text = (
        f"Processing:\n"
        f"  ZPF: {fid.processing.zpf}\n"
        f"  Remove DC: {fid.processing.rdc}\n"
        f"  Exp Filter: {fid.processing.expf_us or 'None'} μs\n"
        f"  Window: {fid.processing.winf or 'None'}\n"
        f"  Start: {fid.processing.start_us or 0:.1f} μs\n"
        f"  End: {fid.processing.end_us or fid.duration_us:.1f} μs"
    )
    
    # Layout text in columns
    ax4.text(0.02, 0.9, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, family='monospace')
    ax4.text(0.35, 0.9, acq_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, family='monospace')
    ax4.text(0.68, 0.9, processing_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, family='monospace')
    
    return fig