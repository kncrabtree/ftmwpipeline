"""
Shared implementation for Stage 1: FT Processing and Spectrum Operations.

This module contains the core implementation functions for FT computation,
spectrum visualization, and parameter management that are shared between 
CLI, Pipeline class, and functional API interfaces.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np

from ..core.data_structures import FID, PreprocessedFID, ComplexFT, FIDProcessingParameters
from ..io.fid_serialization import update_fid_processing_defaults
from ..file_manager import open_pipeline_file
from .stage0_impl import load_fid_from_pipeline_impl


logger = logging.getLogger(__name__)


def compute_ft_impl(
    file_path: str,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    zpf: Optional[int] = None,
    expf_us: Optional[float] = None,
    window_function: Optional[str] = None,
    units_power: Optional[int] = None,
    trim_range: Optional[Tuple[float, float]] = None,
    validate_only: bool = False
) -> Dict[str, Any]:
    """
    Shared implementation for FT computation from .ftmw pipeline files.
    
    This function implements the three-stage FT workflow:
    1. Load FID from pipeline file
    2. Apply preprocessing (windowing, filtering, zero-padding)
    3. Compute FFT and create ComplexFT object
    4. Optionally apply frequency trimming
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    start_us : float, optional
        FID window start time in microseconds
    end_us : float, optional
        FID window end time in microseconds
    zpf : int, optional
        Zero padding factor
    expf_us : float, optional
        Exponential filter time constant in microseconds
    window_function : str, optional
        Window function name (hann, hamming, etc.)
    units_power : int, optional
        Units scaling power of 10
    trim_range : tuple of float, optional
        Frequency range (min_mhz, max_mhz) for trimming
    validate_only : bool, default False
        If True, only validate parameters without returning ComplexFT
        
    Returns
    -------
    dict
        Processing results including ComplexFT object and metadata
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If processing parameters are invalid
    """
    # Load FID from pipeline file
    try:
        # file_path_obj, source_metadata, stage_tracker = open_pipeline_file(file_path)
        fid = load_fid_from_pipeline_impl(file_path)
        logger.info(f"Loaded FID with {len(fid.data):,} points from pipeline file")
    except Exception as e:
        raise RuntimeError(f"Failed to load FID from pipeline file {file_path}: {e}")
    
    # Merge user parameters with cached recommended settings
    cached_defaults = fid.processing
    processing_params = {
        'start_us': start_us if start_us is not None else cached_defaults.start_us,
        'end_us': end_us if end_us is not None else cached_defaults.end_us,
        'zpf': zpf if zpf is not None else (cached_defaults.zpf if cached_defaults.zpf is not None else 1),
        'expf_us': expf_us if expf_us is not None else (cached_defaults.expf_us if cached_defaults.expf_us is not None else 5.0),
        'window_function': window_function if window_function is not None else cached_defaults.winf,
        'units_power': units_power if units_power is not None else (cached_defaults.units_power if cached_defaults.units_power is not None else 6)
    }
    
    # Handle trim_range parameter - load from file if not provided by user
    if trim_range is None:
        # Try to load saved trim parameters from the .ftmw file
        try:
            import h5py
            with h5py.File(file_path, 'r') as h5f:
                rec_proc_path = 'stage0_fid_data/recommended_processing'
                if rec_proc_path in h5f:
                    rec_proc = h5f[rec_proc_path]
                    trim_min = rec_proc.attrs.get('trim_min_mhz')
                    trim_max = rec_proc.attrs.get('trim_max_mhz')
                    
                    # Reconstruct trim_range tuple if both values exist and are not None markers
                    if (trim_min is not None and trim_max is not None and 
                        trim_min != "__None__" and trim_max != "__None__"):
                        trim_range = (float(trim_min), float(trim_max))
                        logger.info(f"Loaded saved trim range: {trim_range[0]:.1f}-{trim_range[1]:.1f} MHz")
        except Exception as e:
            logger.warning(f"Could not load saved trim parameters: {e}")
            # Continue without trim - this is not a fatal error
    
    logger.info("Processing parameters:")
    for param, value in processing_params.items():
        logger.info(f"  {param}: {value}")
    
    # Stage 1: FID Preprocessing
    try:
        preprocessed_fid = fid.preprocess(**processing_params)
        logger.info(f"Preprocessing complete: {len(preprocessed_fid.data):,} points (zero-padded)")
    except Exception as e:
        raise ValueError(f"FID preprocessing failed: {e}")
    
    # Stage 2: FFT Calculation
    try:
        complex_spectrum, freq_array = preprocessed_fid.compute_fft()
        logger.info(f"FFT computation complete: {len(complex_spectrum):,} frequency points")
        logger.info(f"Frequency range: {freq_array[0]:.1f} - {freq_array[-1]:.1f} MHz")
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}")
    
    # If validation only, return early with basic info
    if validate_only:
        result = {
            'status': 'validated',
            'fid_points': len(fid.data),
            'preprocessed_points': len(preprocessed_fid.data),
            'frequency_points': len(complex_spectrum),
            'frequency_range': (freq_array[0], freq_array[-1]),
            'processing_params': processing_params
        }
        
        # Test trimming if requested
        if trim_range:
            mask = (freq_array >= trim_range[0]) & (freq_array <= trim_range[1])
            if not np.any(mask):
                raise ValueError(f"No data points in trim range [{trim_range[0]:.1f}, {trim_range[1]:.1f}] MHz")
            result['trimmed_points'] = np.sum(mask)
            result['trim_range'] = trim_range
        
        return result
    
    # Stage 3: ComplexFT creation
    try:
        # Prepare FID context for serialization (universal experimental parameters)
        fid_context = {
            'probe_freq_mhz': fid.probe_freq_mhz,
            'spacing_us': fid.spacing * 1e6,  # Convert to microseconds
            'sideband': fid.sideband.value,
            'original_fid_length': len(fid.data)
        }
        
        complex_ft = ComplexFT.from_spectrum(
            complex_spectrum=complex_spectrum,
            freq_array=freq_array,
            metadata={
                'processing_params': preprocessed_fid.processing_params,
                'fid_context': fid_context
            }
        )
        logger.info(f"ComplexFT created: {len(complex_ft.complex_spectrum):,} frequency points")
    except Exception as e:
        raise RuntimeError(f"ComplexFT creation failed: {e}")
    
    # Apply frequency trimming if requested
    if trim_range:
        try:
            complex_ft = complex_ft.trim_to_range(trim_range[0], trim_range[1])
            logger.info(f"Trimmed spectrum: {len(complex_ft.complex_spectrum):,} points")
        except Exception as e:
            raise ValueError(f"Frequency trimming failed: {e}")
    
    # Return comprehensive results
    result = {
        'status': 'success',
        'complex_ft': complex_ft,
        'preprocessed_fid': preprocessed_fid,
        'original_fid': fid,
        'processing_params': processing_params,
        'fid_points': len(fid.data),
        'preprocessed_points': len(preprocessed_fid.data),
        'frequency_points': len(complex_ft.complex_spectrum),
        'frequency_range': (complex_ft.freq_array[0], complex_ft.freq_array[-1])
    }
    
    if trim_range:
        result['trim_range'] = trim_range
    
    # Save FT processing parameters and mark Stage 1 as completed
    try:
        _save_ft_parameters_and_stage_completion(file_path, processing_params)
        logger.info("FT parameters and stage completion saved to pipeline file")
    except Exception as e:
        logger.warning(f"Failed to save FT parameters: {e}")
        # Don't fail the computation if storage fails
    
    return result


def visualize_ft_impl(
    file_path: str,
    start_us: Optional[float] = None,
    end_us: Optional[float] = None,
    zpf: Optional[int] = None,
    expf_us: Optional[float] = None,
    window_function: Optional[str] = None,
    units_power: Optional[int] = None,
    trim_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_fid_panels: bool = True,
    backend: str = 'matplotlib',
    interactive: bool = True,
    **plot_kwargs
) -> Any:
    """
    Shared implementation for FT visualization from .ftmw pipeline files.
    
    This function creates enhanced multi-panel plots showing the complete
    FID-to-spectrum processing workflow with optional parameter saving.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    start_us, end_us, zpf, expf_us, window_function, units_power : optional
        FT processing parameters (same as compute_ft_impl)
    trim_range : tuple of float, optional
        Frequency range for trimming
    title : str, optional
        Plot title
    show_fid_panels : bool, default True
        Whether to show FID processing panels
    backend : str, default 'matplotlib'
        Plotting backend
    interactive : bool, default True
        Whether to create interactive plots
    **plot_kwargs
        Additional plotting parameters
        
    Returns
    -------
    matplotlib.Figure or plotly.Figure
        The created figure object
    """
    # Compute FT with current parameters
    ft_result = compute_ft_impl(
        file_path=file_path,
        start_us=start_us,
        end_us=end_us,
        zpf=zpf,
        expf_us=expf_us,
        window_function=window_function,
        units_power=units_power,
        trim_range=trim_range,
        validate_only=False
    )
    
    complex_ft = ft_result['complex_ft']
    preprocessed_fid = ft_result['preprocessed_fid']
    original_fid = ft_result['original_fid']
    
    # Import visualization function
    try:
        from ..visualization.spectrum_visualization import plot_complex_ft
    except ImportError:
        raise ImportError("Spectrum visualization not available - visualization module missing")
    
    # Generate title if not provided
    if title is None:
        pipeline_name = Path(file_path).stem
        title = f"Pipeline {pipeline_name} - FT Visualization"
        if trim_range:
            title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"
    
    # Create enhanced plot
    try:
        fig = plot_complex_ft(
            complex_ft=complex_ft,
            title=title,
            backend=backend,
            interactive=interactive,
            fid=original_fid if show_fid_panels else None,
            preprocessed_fid=preprocessed_fid if show_fid_panels else None,
            show_fid_panels=show_fid_panels,
            **plot_kwargs
        )
        logger.info("FT visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create FT visualization: {e}")


def save_ft_parameters_impl(
    file_path: str,
    parameters: Dict[str, Any]
) -> None:
    """
    Shared implementation for saving FT processing parameters to pipeline file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    parameters : dict
        Processing parameters to save
    """
    try:
        # Import the file manager function
        from ..file_manager import update_processing_parameters
        
        # Update parameters in the .ftmw file
        update_processing_parameters(file_path, parameters)
        logger.info("Processing parameters saved successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to save parameters: {e}")


def compare_ft_parameters_impl(
    file_path: str,
    current_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Shared implementation for comparing current parameters with cached defaults.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    current_params : dict
        Current processing parameters
        
    Returns
    -------
    dict
        Comparison results with custom parameters identified
    """
    try:
        # Load cached FID to get default parameters
        fid = open_pipeline_file(file_path)
        default_params = {
            'start_us': fid.processing.start_us,
            'end_us': fid.processing.end_us,
            'zpf': fid.processing.zpf,
            'expf_us': fid.processing.expf_us,
            'window_function': fid.processing.winf,
            'rdc': fid.processing.rdc,
            'units_power': fid.processing.units_power
        }
        
        # Find parameters that differ from cached defaults
        custom_params = {}
        for param, user_value in current_params.items():
            default_value = default_params.get(param)
            if user_value != default_value and user_value is not None:
                # Special handling for default values that might indicate customization
                if param == 'zpf' and user_value != 1:
                    custom_params[param] = user_value
                elif param == 'expf_us' and user_value != 5.0:
                    custom_params[param] = user_value
                elif param == 'units_power' and user_value != 6:
                    custom_params[param] = user_value
                elif param in ['start_us', 'end_us', 'window_function'] and user_value is not None:
                    custom_params[param] = user_value
        
        return {
            'default_params': default_params,
            'current_params': current_params,
            'custom_params': custom_params,
            'has_custom_params': len(custom_params) > 0
        }
    except Exception as e:
        raise RuntimeError(f"Failed to compare parameters: {e}")


def _save_ft_parameters_and_stage_completion(file_path: str, parameters_used: Dict[str, Any]) -> None:
    """
    Save FT processing parameters and mark Stage 1 as completed.
    
    This function saves the processing parameters for persistence and parameter exploration,
    and updates stage tracking to mark Stage 1 as completed. It does NOT store ComplexFT data,
    maintaining the lightweight .ftmw file design where ComplexFT is computed on-demand.
    """
    import h5py
    import json
    from datetime import datetime
    
    try:
        with h5py.File(file_path, 'a') as h5f:
            # Save FT processing parameters
            if 'processing_parameters' not in h5f:
                h5f.create_group('processing_parameters')
            
            processing_group = h5f['processing_parameters']
            
            # Remove existing FT parameters if present (allow parameter updates)
            if 'ft_processing' in processing_group:
                del processing_group['ft_processing']
            
            # Create FT parameters group and save parameters
            ft_params_group = processing_group.create_group('ft_processing')
            ft_params_group.attrs['parameters'] = json.dumps(parameters_used, default=str)
            ft_params_group.attrs['last_updated'] = datetime.now().isoformat()
            
            # Update pipeline stages to mark Stage 1 as completed
            if 'pipeline_stages' not in h5f:
                h5f.create_group('pipeline_stages')
            
            stages_group = h5f['pipeline_stages']
            
            # Load current completed stages
            completed_stages_json = stages_group.attrs.get('completed_stages', '[]')
            completed_stages = json.loads(completed_stages_json)
            
            # Add stage1_complex_ft if not already present. This must match the
            # canonical key in PipelineStageTracker.STAGE_DEPENDENCIES so the
            # stage tracker and dependency checks recognize Stage 1 completion.
            if 'stage1_complex_ft' not in completed_stages:
                completed_stages.append('stage1_complex_ft')
            
            # Update completed stages and timestamp
            stages_group.attrs['completed_stages'] = json.dumps(completed_stages)
            stages_group.attrs['last_updated'] = datetime.now().isoformat()
        
        logger.info("FT parameters and stage tracking saved to pipeline file")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save FT parameters to pipeline file: {e}")

