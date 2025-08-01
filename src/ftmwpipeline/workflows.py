"""
Convenience workflow functions for common FTMW processing tasks.

This module provides high-level functions that combine multiple pipeline
steps for common use cases, making the package easier to use for
routine analyses.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from .pipeline import Pipeline
from .core.data_structures import FittingResult


def process_experiment(
    data_path: Union[str, Path],
    config: Optional[Dict] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Process a single FTMW experiment with default settings.
    
    This is a convenience function that creates a Pipeline instance
    and runs the complete analysis workflow.
    
    Parameters
    ----------
    data_path : str or Path
        Path to experimental data file
    config : dict, optional
        Pipeline configuration parameters
    output_dir : str or Path, optional
        Directory for output files
        
    Returns
    -------
    dict
        Processing results and metadata
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing experiment: {data_path}")
    
    # Create and run pipeline
    pipeline = Pipeline(config=config)
    results = pipeline.process_experiment(data_path, output_dir)
    
    return results


def batch_process_experiments(
    data_paths: List[Union[str, Path]],
    config: Optional[Dict] = None,
    output_dir: Optional[Union[str, Path]] = None,
    parallel: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple FTMW experiments in batch.
    
    Parameters
    ----------
    data_paths : list of str or Path
        List of paths to experimental data files
    config : dict, optional
        Pipeline configuration parameters
    output_dir : str or Path, optional
        Base directory for output files
    parallel : bool, default False
        Whether to process experiments in parallel
        
    Returns
    -------
    list of dict
        List of processing results for each experiment
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Batch processing {len(data_paths)} experiments")
    
    results = []
    
    if parallel:
        # Placeholder for parallel processing - will implement later
        logger.warning("Parallel processing not yet implemented, using sequential")
    
    # Sequential processing
    for data_path in data_paths:
        try:
            result = process_experiment(data_path, config, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {data_path}: {e}")
            results.append({
                "data_path": str(data_path),
                "status": "error",
                "error": str(e)
            })
    
    return results


def quick_fit(
    frequencies: List[float],
    intensities: List[float],
    peak_frequencies: Optional[List[float]] = None,
    **kwargs
) -> List[FittingResult]:
    """
    Quick peak fitting for simple cases.
    
    This function provides a simplified interface for fitting peaks
    when you already have frequency and intensity data.
    
    Parameters
    ----------
    frequencies : list of float
        Frequency values
    intensities : list of float  
        Intensity values
    peak_frequencies : list of float, optional
        Known peak frequencies for fitting. If None, peaks will be detected.
    **kwargs
        Additional fitting parameters
        
    Returns
    -------
    list of FittingResult
        Fitting results for detected/specified peaks
    """
    logger = logging.getLogger(__name__)
    logger.info("Running quick fit analysis")
    
    # Placeholder implementation
    logger.warning("Quick fit not yet implemented - returning empty results")
    return []


def validate_installation() -> Dict[str, bool]:
    """
    Validate that the ftmwpipeline installation is working correctly.
    
    Returns
    -------
    dict
        Dictionary indicating which components are working
    """
    validation_results = {
        "core_imports": False,
        "dependencies": False,
        "test_data": False,
        "pipeline_creation": False
    }
    
    try:
        # Test core imports
        from . import core, preprocessing, peak_detection
        validation_results["core_imports"] = True
    except ImportError:
        pass
    
    try:
        # Test dependencies
        import numpy, scipy, matplotlib
        validation_results["dependencies"] = True
    except ImportError:
        pass
    
    try:
        # Test pipeline creation
        pipeline = Pipeline()
        validation_results["pipeline_creation"] = True
    except Exception:
        pass
    
    return validation_results