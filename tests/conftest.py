"""
Pytest configuration and shared fixtures for ftmwpipeline tests.

This module provides common test fixtures and configuration for the
entire test suite.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple, Any

# Test data directory (will be populated in later phases)
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def sample_frequencies():
    """Generate sample frequency array for testing."""
    return np.linspace(8000.0, 12000.0, 1000)  # MHz


@pytest.fixture
def sample_intensities():
    """Generate sample intensity data with synthetic peaks."""
    freqs = np.linspace(8000.0, 12000.0, 1000)

    # Create synthetic spectrum with a few peaks
    intensities = np.random.normal(0, 0.1, len(freqs))  # Noise floor

    # Add synthetic peaks
    peak_centers = [8500.0, 9200.0, 10800.0]  # MHz
    peak_amplitudes = [2.0, 1.5, 3.0]
    peak_widths = [0.5, 0.3, 0.7]  # MHz

    for center, amp, width in zip(peak_centers, peak_amplitudes, peak_widths):
        # Gaussian peaks
        intensities += amp * np.exp(-0.5 * ((freqs - center) / width) ** 2)

    return intensities


@pytest.fixture
def sample_spectral_data(sample_frequencies, sample_intensities):
    """Provide combined frequency and intensity data."""
    return {
        "frequencies": sample_frequencies,
        "intensities": sample_intensities,
        "metadata": {
            "source": "synthetic_test_data",
            "frequency_unit": "MHz",
            "intensity_unit": "arbitrary",
        },
    }


@pytest.fixture
def sample_fid_parameters():
    """Generate sample FID parameters for testing."""
    return {
        "shot_count": 100000,
        "sample_rate": 50.0,  # GSa/s
        "record_length": 100000,
        "center_frequency": 10000.0,  # MHz
        "probe_frequency": 10000.0,  # MHz
        "attenuation": 20.0,  # dB
        "temperature": 298.0,  # K
        "pressure": 1.0,  # atm
    }


@pytest.fixture
def sample_peaks():
    """Generate sample peak data for testing."""
    return [
        {
            "frequency": 8500.0,
            "amplitude": 2.0,
            "width": 0.5,
            "snr": 20.0,
            "classification": "strong",
        },
        {
            "frequency": 9200.0,
            "amplitude": 1.5,
            "width": 0.3,
            "snr": 15.0,
            "classification": "medium",
        },
        {
            "frequency": 10800.0,
            "amplitude": 3.0,
            "width": 0.7,
            "snr": 30.0,
            "classification": "strong",
        },
    ]


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="ftmwpipeline_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Provide sample pipeline configuration."""
    return {
        "preprocessing": {
            "baseline_method": "polynomial",
            "baseline_order": 2,
            "noise_estimation_method": "std",
        },
        "peak_detection": {
            "method": "basic",
            "threshold_factor": 3.0,
            "min_separation": 0.1,  # MHz
        },
        "window_assignment": {
            "method": "greedy",
            "fwhm_factor": 4.0,
            "overlap_threshold": 0.1,
        },
        "fitting": {
            "method": "time_domain",
            "max_iterations": 100,
            "convergence_threshold": 1e-6,
            "use_constraints": True,
        },
    }


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (may be skipped)")


@pytest.fixture(autouse=True)
def reset_numpy_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib

        return True
    except ImportError:
        return False


def _has_plotly() -> bool:
    """Check if plotly is available."""
    try:
        import plotly

        return True
    except ImportError:
        return False


# Skip markers for optional dependencies
pytest_matplotlib = pytest.mark.skipif(
    not _has_matplotlib(), reason="matplotlib not available"
)

pytest_plotly = pytest.mark.skipif(not _has_plotly(), reason="plotly not available")
