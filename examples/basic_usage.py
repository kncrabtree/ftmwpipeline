#!/usr/bin/env python3
"""
Basic usage example for ftmwpipeline.

This script demonstrates the simplest way to process FTMW spectroscopy data
using the ftmwpipeline package.
"""

import sys
from pathlib import Path

# Add the source directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import ftmwpipeline as fmw


def basic_processing_example():
    """Demonstrate basic FTMW data processing."""
    print("=== FTMW Pipeline Basic Usage Example ===\n")
    
    # Check package information
    print(f"Package version: {fmw.__version__}")
    print(f"Package info: {fmw.PACKAGE_INFO}\n")
    
    # Validate installation
    print("Validating installation...")
    validation = fmw.workflows.validate_installation()
    for component, status in validation.items():
        status_str = "✅ OK" if status else "❌ FAILED"
        print(f"  {component}: {status_str}")
    print()
    
    # Create a pipeline instance
    print("Creating pipeline...")
    pipeline = fmw.Pipeline()
    print(f"Pipeline summary: {pipeline.get_summary()}\n")
    
    # Example data path (placeholder - will work with real data in later phases)
    example_data_path = "data/experiment.h5"
    
    print(f"Processing example data: {example_data_path}")
    try:
        results = pipeline.process_experiment(example_data_path)
        print("Processing results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Processing not yet implemented: {e}")
    
    print("\n=== Example completed ===")


def workflow_functions_example():
    """Demonstrate workflow convenience functions."""
    print("\n=== Workflow Functions Example ===\n")
    
    # Single experiment processing
    print("1. Single experiment processing")
    try:
        results = fmw.process_experiment("data/test.h5")
        print(f"   Results: {results}")
    except Exception as e:
        print(f"   Not yet implemented: {e}")
    
    # Batch processing
    print("\n2. Batch processing")
    data_files = ["data/exp1.h5", "data/exp2.h5", "data/exp3.h5"]
    try:
        results = fmw.batch_process_experiments(data_files)
        print(f"   Processed {len(results)} experiments")
    except Exception as e:
        print(f"   Not yet implemented: {e}")
    
    # Quick fitting
    print("\n3. Quick fitting")
    try:
        frequencies = [8000.0, 8001.0, 8002.0]  # MHz
        intensities = [1.0, 2.0, 1.5]
        results = fmw.quick_fit(frequencies, intensities)
        print(f"   Fit results: {results}")
    except Exception as e:
        print(f"   Not yet implemented: {e}")


if __name__ == "__main__":
    basic_processing_example()
    workflow_functions_example()
    
    print("\nNote: This is Phase 1 infrastructure. Actual data processing")
    print("functionality will be implemented in subsequent phases.")