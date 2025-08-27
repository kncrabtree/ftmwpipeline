"""
Shared implementation for Stage 2: Noise Estimation.

This module contains the core implementation functions for noise estimation
and visualization that are shared between CLI, Pipeline class, and functional 
API interfaces.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import h5py

from ..preprocessing.noise_estimation import estimate_noise_adaptive, NoiseResult
from ..io.noise_result_serialization import save_noise_result_to_hdf5, load_noise_result_from_hdf5
from ..file_manager import open_pipeline_file


logger = logging.getLogger(__name__)


def compute_noise_estimation_impl(
    file_path: str,
    skew_target: Optional[float] = None,
    min_bin_fraction: Optional[float] = None,
    smoothing_window_mhz: Optional[float] = None,
    min_noise_fraction: Optional[float] = None,
    from_saved_params: bool = False
) -> Dict[str, Any]:
    """
    Shared implementation for noise estimation from .ftmw pipeline files.
    
    This function implements the Stage 2 workflow:
    1. Load ComplexFT from pipeline file (requires Stage 1)
    2. Merge user parameters with any saved noise parameters
    3. Call estimate_noise_adaptive() with merged parameters
    4. Return structured result with NoiseResult and metadata
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    skew_target : float, optional
        Target skewness for noise identification (default: 0.631)
    min_bin_fraction : float, optional
        Minimum bin size as fraction of total data (default: 1/64)
    smoothing_window_mhz : float, optional
        RMS smoothing window size in MHz
    min_noise_fraction : float, optional
        Minimum fraction of points that must be noise per bin (default: 2/3)
    from_saved_params : bool, default False
        If True, use only saved parameters and ignore user parameters
        
    Returns
    -------
    dict
        Processing results with keys:
        - 'noise_result': NoiseResult object
        - 'complex_ft': ComplexFT object used for estimation
        - 'parameters_used': Dict of actual parameters used
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If Stage 1 dependencies are not met or parameters are invalid
    """
    # Compute ComplexFT on-demand using Stage 1 implementation (correct architecture)
    try:
        # Check that Stage 1 parameters are available (Stage 1 dependency)
        with h5py.File(file_path, 'r') as h5f:
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError(
                    "Stage 1 (FT computation) must be completed before noise estimation. "
                    "Run compute_ft() or ft-process command first."
                )
        
        # Import Stage 1 implementation for on-demand ComplexFT computation
        from .stage1_impl import compute_ft_impl
        
        # Compute ComplexFT using saved Stage 1 parameters (lightweight on-demand computation)
        stage1_result = compute_ft_impl(file_path=file_path)
        
        complex_ft = stage1_result['complex_ft']
        logger.info(f"Computed ComplexFT on-demand with {len(complex_ft.freq_array):,} frequency points")
            
    except Exception as e:
        raise RuntimeError(f"Failed to compute ComplexFT from pipeline file {file_path}: {e}")
    
    # Load saved noise parameters if requested or available
    saved_params = {}
    if from_saved_params:
        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'processing_parameters' in h5f and 'noise_estimation' in h5f['processing_parameters']:
                    noise_group = h5f['processing_parameters/noise_estimation']
                    for param_name in ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']:
                        if param_name in noise_group.attrs:
                            value = noise_group.attrs[param_name]
                            # Handle None values stored as strings
                            if isinstance(value, str) and value == "__None__":
                                value = None
                            saved_params[param_name] = value
                    logger.info(f"Loaded saved noise parameters: {saved_params}")
        except Exception as e:
            logger.warning(f"Could not load saved noise parameters: {e}")
    
    # Merge parameters (saved parameters override defaults, user parameters override both)
    default_params = {
        'skew_target': 0.631,
        'min_bin_fraction': 1/64,
        'smoothing_window_mhz': None,  # Will be auto-calculated
        'min_noise_fraction': 2/3
    }
    
    # Priority: user params > saved params > defaults
    processing_params = {}
    for param_name in default_params.keys():
        if from_saved_params:
            # Use saved parameters when explicitly requested
            processing_params[param_name] = saved_params.get(param_name, default_params[param_name])
        else:
            # Use user parameters if provided, otherwise saved, otherwise default
            user_value = locals().get(param_name)  # Get the parameter from function arguments
            if user_value is not None:
                processing_params[param_name] = user_value
            elif param_name in saved_params:
                processing_params[param_name] = saved_params[param_name]
            else:
                processing_params[param_name] = default_params[param_name]
    
    logger.info("Noise estimation parameters:")
    for param, value in processing_params.items():
        logger.info(f"  {param}: {value}")
    
    # Perform noise estimation
    try:
        noise_result = estimate_noise_adaptive(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            skew_target=processing_params['skew_target'],
            min_bin_fraction=processing_params['min_bin_fraction'],
            smoothing_window_mhz=processing_params['smoothing_window_mhz'],
            min_noise_fraction=processing_params['min_noise_fraction'],
            verbose=True  # Enable logging for diagnostics
        )
        logger.info("Noise estimation completed successfully")
        logger.info(f"  Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}")
        logger.info(f"  Number of bins: {noise_result.bin_info.get('n_bins', 'unknown')}")
        logger.info(f"  RMS noise range: {noise_result.rms_noise.min():.2e} - {noise_result.rms_noise.max():.2e}")
    except Exception as e:
        raise ValueError(f"Noise estimation failed: {e}")
    
    # Store NoiseResult and mark stage as completed
    try:
        # Save NoiseResult to pipeline file
        save_noise_result_impl(
            file_path=file_path,
            noise_result=noise_result,
            complex_ft=complex_ft,
            parameters_used=processing_params
        )
        
        # Update stage tracker to mark stage2_noise_result complete
        _update_stage_completion(file_path, 'stage2_noise_result')
        
        logger.info("Stage 2: Noise estimation results saved and marked complete")
        
    except Exception as e:
        logger.error(f"Failed to save noise estimation results: {e}")
        raise RuntimeError(f"Noise estimation succeeded but storage failed: {e}")
    
    # Return comprehensive results
    result = {
        'status': 'success',
        'noise_result': noise_result,
        'complex_ft': complex_ft,
        'parameters_used': processing_params,
        'frequency_points': len(complex_ft.freq_array),
        'frequency_range': (complex_ft.freq_array[0], complex_ft.freq_array[-1]),
        'noise_points': int(noise_result.noise_mask.sum()),
        'total_points': len(complex_ft.freq_array)
    }
    
    return result


def visualize_noise_impl(
    file_path: str,
    y_max_factor: Optional[float] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    show_bin_boundaries: Optional[bool] = None,
    show_noise_points: Optional[bool] = None,
    backend: str = 'matplotlib',
    interactive: bool = True,
    save_params: bool = False,
    **plot_kwargs
) -> Any:
    """
    Shared implementation for noise visualization from .ftmw pipeline files.
    
    This function creates noise estimation diagnostic plots showing spectrum,
    noise points, bin boundaries, and RMS estimates.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    y_max_factor : float, optional
        Y-axis maximum as multiple of median RMS noise (default: 20.0)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (16, 6))
    title : str, optional
        Custom title for the plot
    show_bin_boundaries : bool, optional
        Whether to show adaptive bin boundaries (default: True)
    show_noise_points : bool, optional
        Whether to highlight noise points (default: True)
    backend : str, default 'matplotlib'
        Plotting backend ('matplotlib' or 'plotly')
    interactive : bool, default True
        Whether to create interactive plots
    save_params : bool, default False
        Whether to save custom parameters for future use
    **plot_kwargs
        Additional plotting parameters
        
    Returns
    -------
    matplotlib.Figure or plotly.Figure
        The created figure object
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If Stage 2 dependencies are not met
    """
    # Load NoiseResult and ComplexFT from pipeline file
    try:
        with h5py.File(file_path, 'r') as h5f:
            # Check dependencies
            if 'stage2_noise_result' not in h5f:
                raise ValueError(
                    "Stage 2 (noise estimation) must be completed before visualization. "
                    "Run estimate_noise() or estimate-noise command first."
                )
            
            # Check that Stage 1 parameters exist (needed for on-demand ComplexFT computation)
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError(
                    "Stage 1 (FT computation) required for noise visualization. "
                    "Run compute_ft() or ft-process command first."
                )
        
        # Compute ComplexFT on-demand for visualization (consistent with Stage 2 architecture)
        from .stage1_impl import compute_ft_impl
        stage1_result = compute_ft_impl(file_path=file_path)
        complex_ft = stage1_result['complex_ft']
        
        # Load NoiseResult data
        with h5py.File(file_path, 'r') as h5f:
            
            noise_result = load_noise_result_from_hdf5(
                h5f['stage2_noise_result'],
                complex_ft.freq_array,
                complex_ft.magnitude_spectrum
            )
            logger.info("Loaded NoiseResult and ComplexFT from pipeline file")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load data from pipeline file {file_path}: {e}")
    
    # Import visualization function
    try:
        from ..visualization.noise_visualization import plot_noise_estimation
    except ImportError:
        raise ImportError("Noise visualization not available - visualization module missing")
    
    # Set parameter defaults
    plot_params = {
        'y_max_factor': y_max_factor if y_max_factor is not None else 20.0,
        'figsize': figsize if figsize is not None else (16, 6),
        'show_bin_boundaries': show_bin_boundaries if show_bin_boundaries is not None else True,
        'show_noise_points': show_noise_points if show_noise_points is not None else True,
        'backend': backend,
    }
    
    # Generate title if not provided
    if title is None:
        pipeline_name = Path(file_path).stem
        title = f"Pipeline {pipeline_name} - Noise Estimation"
        freq_range = (complex_ft.freq_array[0], complex_ft.freq_array[-1])
        title += f" ({freq_range[0]:.0f}-{freq_range[1]:.0f} MHz)"
    
    plot_params['title'] = title
    plot_params.update(plot_kwargs)
    
    # Create diagnostic plot
    try:
        fig = plot_noise_estimation(
            frequencies=complex_ft.freq_array,
            magnitudes=complex_ft.magnitude_spectrum,
            noise_result=noise_result,
            **plot_params
        )
        
        # Save parameters if requested
        if save_params:
            try:
                # Read the current noise estimation parameters from the pipeline file
                # These are the parameters that were used to create the current NoiseResult
                with h5py.File(file_path, 'r') as h5f:
                    if 'processing_parameters' in h5f and 'noise_estimation' in h5f['processing_parameters']:
                        noise_group = h5f['processing_parameters/noise_estimation']
                        current_noise_params = {}
                        for param_name in ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']:
                            if param_name in noise_group.attrs:
                                value = noise_group.attrs[param_name]
                                # Handle None values stored as strings
                                if isinstance(value, str) and value == "__None__":
                                    value = None
                                current_noise_params[param_name] = value
                        
                        # Save the current noise estimation parameters (this enables from_saved_params=True)
                        save_noise_parameters_impl(file_path, current_noise_params)
                        logger.info(f"Saved noise estimation parameters for future use: {current_noise_params}")
                    else:
                        logger.warning("No noise estimation parameters found in pipeline file")
                        
                # Also save any custom visualization parameters that differ from defaults
                custom_vis_params = {}
                if y_max_factor is not None and y_max_factor != 20.0:
                    custom_vis_params['y_max_factor'] = y_max_factor
                if figsize is not None and figsize != (16, 6):
                    custom_vis_params['figsize'] = figsize
                if show_bin_boundaries is not None and show_bin_boundaries != True:
                    custom_vis_params['show_bin_boundaries'] = show_bin_boundaries
                if show_noise_points is not None and show_noise_points != True:
                    custom_vis_params['show_noise_points'] = show_noise_points
                
                if custom_vis_params:
                    save_noise_parameters_impl(file_path, {'visualization': custom_vis_params})
                    logger.info(f"Saved {len(custom_vis_params)} visualization parameters")
                    
            except Exception as e:
                logger.warning(f"Failed to save parameters: {e}")
        
        logger.info("Noise estimation visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create noise visualization: {e}")


def save_noise_parameters_impl(file_path: str, parameters: Dict[str, Any]) -> None:
    """
    Save noise estimation parameters to .ftmw file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    parameters : dict
        Noise estimation parameters to save
    """
    try:
        # Create parameters dict for noise estimation (filter to known parameter names)
        noise_params = {}
        param_names = ['skew_target', 'min_bin_fraction', 'smoothing_window_mhz', 'min_noise_fraction']
        
        for param_name in param_names:
            if param_name in parameters:
                value = parameters[param_name]
                # Convert None to string marker for HDF5 storage
                if value is None:
                    value = "__None__"
                noise_params[param_name] = value
        
        # Save noise parameters to processing_parameters/noise_estimation (same pattern as FT)
        if noise_params:
            with h5py.File(file_path, 'a') as h5f:
                # Ensure processing_parameters group exists
                if 'processing_parameters' not in h5f:
                    h5f.create_group('processing_parameters')
                
                processing_group = h5f['processing_parameters']
                
                # Remove existing noise parameters if present (allow parameter updates)
                if 'noise_estimation' in processing_group:
                    del processing_group['noise_estimation']
                
                # Create noise parameters group and save parameters
                noise_params_group = processing_group.create_group('noise_estimation')
                
                # Store parameters individually as attributes (for easy loading with from_saved_params)
                for param_name, value in noise_params.items():
                    noise_params_group.attrs[param_name] = value
                
                # Also store as JSON for completeness
                noise_params_group.attrs['parameters'] = json.dumps(noise_params, default=str)
                noise_params_group.attrs['last_updated'] = datetime.now().isoformat()
                
            logger.info(f"Saved {len(noise_params)} noise estimation parameters to processing_parameters/noise_estimation")
        else:
            logger.warning("No valid noise parameters to save")
            
    except Exception as e:
        raise RuntimeError(f"Failed to save noise parameters: {e}")


def save_noise_result_impl(
    file_path: str, 
    noise_result: NoiseResult, 
    complex_ft,
    parameters_used: Dict[str, Any]
) -> None:
    """
    Save NoiseResult to .ftmw pipeline file in stage2_noise_result group.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    noise_result : NoiseResult
        NoiseResult object to save
    complex_ft : ComplexFT
        ComplexFT object used for noise estimation (for freq/mag arrays)
    parameters_used : dict
        Parameters used for noise estimation
    """
    try:
        with h5py.File(file_path, 'a') as h5f:
            # Remove existing noise result if present
            if 'stage2_noise_result' in h5f:
                del h5f['stage2_noise_result']
            
            # Create stage2_noise_result group
            stage2_group = h5f.create_group('stage2_noise_result')
            
            # Save NoiseResult using existing serialization
            save_noise_result_to_hdf5(
                noise_result=noise_result,
                frequencies=complex_ft.freq_array,
                magnitudes=complex_ft.magnitude_spectrum,
                h5_group=stage2_group
            )
            
            # Add metadata and timestamp
            stage2_group.attrs['creation_time'] = datetime.now().isoformat()
            stage2_group.attrs['stage_name'] = 'stage2_noise_estimation'
            stage2_group.attrs['parameters_used'] = json.dumps(parameters_used, default=str)
            
        logger.info("NoiseResult saved to pipeline file successfully")
        
        # Also save parameters for future use
        save_noise_parameters_impl(file_path, parameters_used)
        
    except Exception as e:
        raise RuntimeError(f"Failed to save NoiseResult to pipeline file: {e}")


def load_noise_result_impl(file_path: str) -> Dict[str, Any]:
    """
    Load NoiseResult from .ftmw pipeline file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
        
    Returns
    -------
    dict
        Dict containing 'noise_result', 'complex_ft', and metadata
    """
    try:
        with h5py.File(file_path, 'r') as h5f:
            # Check dependencies
            if 'stage2_noise_result' not in h5f:
                raise ValueError("No noise estimation results found in pipeline file")
                
            # Check that Stage 1 parameters exist (needed for ComplexFT computation)
            if 'processing_parameters' not in h5f or 'ft_processing' not in h5f['processing_parameters']:
                raise ValueError("Stage 1 parameters missing - cannot compute ComplexFT for NoiseResult loading")
        
        # Compute ComplexFT on-demand (consistent with new architecture)
        from .stage1_impl import compute_ft_impl
        stage1_result = compute_ft_impl(file_path=file_path)
        complex_ft = stage1_result['complex_ft']
        
        # Load NoiseResult using computed ComplexFT
        with h5py.File(file_path, 'r') as h5f:
            
            # Load NoiseResult
            noise_result = load_noise_result_from_hdf5(
                h5f['stage2_noise_result'],
                complex_ft.freq_array,
                complex_ft.magnitude_spectrum
            )
            
            # Load metadata
            stage2_group = h5f['stage2_noise_result']
            creation_time = stage2_group.attrs.get('creation_time', 'unknown')
            parameters_used = {}
            if 'parameters_used' in stage2_group.attrs:
                try:
                    parameters_used = json.loads(stage2_group.attrs['parameters_used'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse saved parameters")
            
        return {
            'noise_result': noise_result,
            'complex_ft': complex_ft,
            'creation_time': creation_time,
            'parameters_used': parameters_used
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to load NoiseResult from pipeline file: {e}")


# Add missing import
from datetime import datetime
import json


def _update_stage_completion(file_path: str, stage_name: str) -> None:
    """
    Update stage completion in the pipeline file.
    
    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    stage_name : str
        Name of the stage to mark as completed
    """
    try:
        with h5py.File(file_path, 'a') as h5f:
            # Load current stage tracker
            from ..file_manager import _load_stage_tracker
            stage_tracker = _load_stage_tracker(file_path, h5f)
            
            # Mark stage as completed
            stage_tracker.mark_completed(stage_name)
            
            # Update pipeline_stages group
            if 'pipeline_stages' not in h5f:
                stages_group = h5f.create_group('pipeline_stages')
            else:
                stages_group = h5f['pipeline_stages']
            
            # Save updated completion status
            stages_group.attrs['completed_stages'] = json.dumps(list(stage_tracker.completed_stages))
            stages_group.attrs['last_updated'] = datetime.now().isoformat()
            
    except Exception as e:
        raise RuntimeError(f"Failed to update stage completion: {e}")