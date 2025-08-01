"""
Main Pipeline class for orchestrating FTMW spectroscopy data processing.

This module provides the high-level interface for running the complete
FTMW analysis pipeline, from data loading through final fitting results.
"""

from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

from .core.data_structures import SpectralWindow, Peak, FittingResult, FIDParameters
from .config.pipeline_config import PipelineConfig


class Pipeline:
    """
    Main pipeline class for FTMW spectroscopy data processing.
    
    This class orchestrates the complete analysis workflow:
    1. Data loading and preprocessing
    2. Baseline and noise estimation  
    3. Peak detection and classification
    4. Analysis window assignment
    5. Peak fitting with validation
    6. Result serialization and visualization
    """
    
    def __init__(self, config: Optional[Union[Dict, PipelineConfig]] = None):
        """
        Initialize the FTMW pipeline.
        
        Parameters
        ----------
        config : dict or PipelineConfig, optional
            Pipeline configuration parameters. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration management (placeholder)
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
            
        # Pipeline state
        self.fid_parameters: Optional[FIDParameters] = None
        self.spectral_windows: List[SpectralWindow] = []
        self.peaks: List[Peak] = []
        self.fitting_results: List[FittingResult] = []
        
    def process_experiment(
        self, 
        data_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete FTMW experiment.
        
        Parameters
        ----------
        data_path : str or Path
            Path to experimental data file
        output_dir : str or Path, optional
            Directory for output files
            
        Returns
        -------
        dict
            Dictionary containing processing results and metadata
        """
        self.logger.info(f"Starting FTMW pipeline processing for {data_path}")
        
        # Placeholder implementation - will be filled in later phases
        results = {
            "data_path": str(data_path),
            "output_dir": str(output_dir) if output_dir else None,
            "status": "not_implemented", 
            "message": "Pipeline implementation pending - Phase 1 infrastructure only"
        }
        
        return results
    
    def run_step_by_step(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run pipeline with step-by-step control for debugging.
        
        Parameters
        ----------
        data_path : str or Path
            Path to experimental data file
            
        Returns
        -------
        dict
            Dictionary containing step-by-step results
        """
        # Placeholder - will implement in later phases
        return {"status": "not_implemented"}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current pipeline state.
        
        Returns
        -------
        dict
            Summary of pipeline processing results
        """
        return {
            "fid_parameters": self.fid_parameters is not None,
            "num_windows": len(self.spectral_windows),
            "num_peaks": len(self.peaks), 
            "num_fits": len(self.fitting_results),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }