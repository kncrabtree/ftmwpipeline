# FTMW Pipeline - Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for ftmwpipeline's dual-interface architecture, ensuring robust validation across CLI, Pipeline class, and functional API workflows while maintaining the scientific accuracy and reliability required for spectroscopy analysis.

## Testing Architecture

### Multi-Interface Testing Approach
The ftmwpipeline provides three user interfaces that must maintain consistent behavior:
1. **CLI Commands**: File-centric command-line operations
2. **Pipeline Class**: Object-oriented Python API with file binding
3. **Functional API**: Stateless Python functions

**Core Testing Principle**: All interfaces must produce identical scientific results given identical inputs and parameters.

## Testing Hierarchy

### 1. Unit Tests (Existing → Updated)
**Status**: Mostly complete, requires `.ftmw` file adaptation

**Scope**: Test individual components and algorithms in isolation
- **Core Algorithm Tests**: FT processing, noise estimation, peak detection logic
- **Data Structure Tests**: FID, ComplexFT, NoiseResult object validation
- **Serialization Tests**: HDF5 round-trip accuracy for all data types
- **Parameter Validation**: Input validation and error handling
- **File Format Tests**: Multi-format data loader validation

**Updates Needed**:
```python
# Current pattern (update needed)
def test_fid_serialization(tmp_path):
    cache_file = save_fid_cache("test_exp", fid, str(tmp_path))
    loaded_fid = load_fid_cache("test_exp", str(tmp_path))
    
# New pattern (.ftmw files)
def test_fid_serialization(tmp_path):
    pipeline_file = tmp_path / "test_experiment.ftmw"
    save_pipeline_stage(pipeline_file, fid, stage="fid")
    loaded_fid = load_pipeline_stage(pipeline_file, stage="fid")
```

### 2. Integration Tests (Complete Redesign)
**Status**: Requires complete rework for dual-interface architecture

**Scope**: Test complete workflows across all interfaces

#### 2.1 CLI Integration Tests
**Target**: Validate complete command-line workflows
```bash
# Test complete Stage 0-1 workflow
ftmwpipeline import-data test_exp.ftmw --source examples/blackchirp_data/2638/
ftmwpipeline compute-ft test_exp.ftmw --zpf 2 --expf_us 5.0
ftmwpipeline visualize-ft test_exp.ftmw --trim 26500:40000 --save
```

**Test Categories**:
- **Basic Workflows**: Stage 0 → 1 → 2 progression
- **Parameter Persistence**: Save/load parameter sets between commands
- **Error Handling**: Missing files, invalid parameters, dependency failures
- **File Management**: Creation, validation, corruption recovery
- **Batch Operations**: Multiple files, scripting scenarios

#### 2.2 Pipeline Class Integration Tests  
**Target**: Validate object-oriented Python API workflows
```python
# Test complete Pipeline class workflow
def test_pipeline_class_workflow(tmp_path, sample_data):
    pipeline_file = tmp_path / "test_experiment.ftmw"
    
    # Creation and initial processing
    pipe = Pipeline.create(pipeline_file, source=sample_data)
    complex_ft = pipe.compute_ft(zpf=2, expf_us=5.0)
    
    # File persistence and reopening
    pipe2 = Pipeline.open(pipeline_file)
    noise_result = pipe2.estimate_noise()
    
    # Verify results
    assert complex_ft.freq_array.shape == pipe2.get_ft().freq_array.shape
```

**Test Categories**:
- **Creation vs Opening**: `Pipeline.create()` vs `Pipeline.open()` semantics
- **Parameter Management**: Default parameter handling and persistence
- **Stage Dependencies**: Automatic dependency validation
- **Error Handling**: File conflicts, missing dependencies, invalid parameters
- **Jupyter Patterns**: Safe re-execution scenarios

#### 2.3 Functional API Integration Tests
**Target**: Validate stateless function-based workflows  
```python
# Test functional API workflow
def test_functional_api_workflow(tmp_path, sample_data):
    pipeline_file = tmp_path / "test_experiment.ftmw"
    
    # Import and process
    ftmw.import_data(pipeline_file, source=sample_data)
    complex_ft = ftmw.compute_ft(pipeline_file, zpf=2, expf_us=5.0)
    noise_result = ftmw.estimate_noise(pipeline_file)
    
    # Verify file persistence
    info = ftmw.get_info(pipeline_file)
    assert info['stages_complete'] == ['import', 'ft', 'noise']
```

**Test Categories**:
- **Stateless Operations**: Each function call is independent
- **File-Based State**: Proper state persistence between function calls
- **Parameter Handling**: Explicit parameter passing without defaults
- **Error Propagation**: Clear error messages for file and parameter issues
- **Batch Processing**: Multiple file operations

### 3. Cross-Interface Consistency Tests
**Status**: New requirement for dual-interface architecture

**Scope**: Verify identical results across all three interfaces

#### 3.1 Result Consistency Tests
```python
def test_cross_interface_consistency(tmp_path, sample_data):
    """Verify CLI, Pipeline class, and functional API produce identical results."""
    
    # Set up identical test parameters
    zpf, expf_us, trim_range = 2, 5.0, (26500, 40000)
    
    # Test via CLI (subprocess)
    cli_file = tmp_path / "cli_test.ftmw"
    run_cli_workflow(cli_file, sample_data, zpf, expf_us, trim_range)
    cli_result = load_ft_result(cli_file)
    
    # Test via Pipeline class
    pipeline_file = tmp_path / "pipeline_test.ftmw" 
    pipe = Pipeline.create(pipeline_file, source=sample_data)
    pipe_result = pipe.compute_ft(zpf=zpf, expf_us=expf_us, trim=trim_range)
    
    # Test via functional API
    func_file = tmp_path / "func_test.ftmw"
    ftmw.import_data(func_file, source=sample_data)
    func_result = ftmw.compute_ft(func_file, zpf=zpf, expf_us=expf_us, trim=trim_range)
    
    # Verify identical results
    np.testing.assert_array_equal(cli_result.complex_spectrum, pipe_result.complex_spectrum)
    np.testing.assert_array_equal(pipe_result.complex_spectrum, func_result.complex_spectrum)
```

#### 3.2 Parameter Handling Consistency
```python
def test_parameter_consistency(tmp_path, sample_data):
    """Verify parameter defaults and validation behave identically."""
    
    # Test default parameter behavior across interfaces
    cli_defaults = get_cli_defaults("compute-ft")
    pipeline_defaults = get_pipeline_defaults(Pipeline, "compute_ft")
    func_defaults = get_function_defaults(ftmw.compute_ft)
    
    assert cli_defaults == pipeline_defaults == func_defaults
    
    # Test parameter validation consistency
    invalid_params = {"zpf": -1, "expf_us": 0, "trim": "invalid"}
    
    for param, value in invalid_params.items():
        with pytest.raises(ValidationError):
            run_cli_with_param(param, value)
        with pytest.raises(ValidationError):
            Pipeline.create("test.ftmw", source=sample_data).compute_ft(**{param: value})
        with pytest.raises(ValidationError):
            ftmw.compute_ft("test.ftmw", **{param: value})
```

### 4. File Management Tests
**Status**: New requirement for `.ftmw` file architecture

**Scope**: Validate pipeline data file operations and edge cases

#### 4.1 File Creation and Opening Tests
```python
def test_pipeline_file_lifecycle(tmp_path, sample_data):
    """Test complete .ftmw file lifecycle."""
    
    pipeline_file = tmp_path / "lifecycle_test.ftmw"
    
    # Test creation
    pipe1 = Pipeline.create(pipeline_file, source=sample_data)
    assert pipeline_file.exists()
    assert pipe1.get_info()['stages_complete'] == ['import']
    
    # Test opening existing file
    pipe2 = Pipeline.open(pipeline_file)
    assert pipe2.get_info() == pipe1.get_info()
    
    # Test modification and persistence
    pipe2.compute_ft(zpf=2)
    assert pipe2.get_info()['stages_complete'] == ['import', 'ft']
    
    # Test reopening with modifications
    pipe3 = Pipeline.open(pipeline_file)
    assert pipe3.get_info()['stages_complete'] == ['import', 'ft']
```

#### 4.2 Safe Import Detection Tests
```python
def test_safe_reimport_behavior(tmp_path, sample_data):
    """Test smart source detection for safe re-imports."""
    
    pipeline_file = tmp_path / "reimport_test.ftmw"
    
    # Initial creation
    pipe1 = Pipeline.create(pipeline_file, source=sample_data)
    original_mtime = pipeline_file.stat().st_mtime
    
    # Re-import with identical source (should load existing)
    pipe2 = Pipeline.create(pipeline_file, source=sample_data)  
    assert pipeline_file.stat().st_mtime == original_mtime  # No file modification
    assert pipe2.get_info() == pipe1.get_info()
    
    # Re-import with different source (should require force)
    different_data = "path/to/different/data"
    with pytest.raises(PipelineExistsError):
        Pipeline.create(pipeline_file, source=different_data)
    
    # Force overwrite
    pipe3 = Pipeline.create(pipeline_file, source=different_data, force=True)
    assert pipeline_file.stat().st_mtime > original_mtime
```

#### 4.3 Error Handling and Recovery Tests
```python
def test_file_error_handling(tmp_path):
    """Test error handling for corrupted or invalid files."""
    
    # Test missing file
    missing_file = tmp_path / "missing.ftmw"
    with pytest.raises(FileNotFoundError):
        Pipeline.open(missing_file)
    
    # Test corrupted file
    corrupted_file = tmp_path / "corrupted.ftmw"
    corrupted_file.write_text("not an HDF5 file")
    with pytest.raises(PipelineDataError):
        Pipeline.open(corrupted_file)
    
    # Test incomplete file (missing required stages)
    incomplete_file = create_incomplete_pipeline_file(tmp_path)
    with pytest.raises(PipelineStageError):
        ftmw.compute_ft(incomplete_file)
```

## Test Data and Fixtures

### Shared Test Data
```python
# Shared fixtures for all test types
@pytest.fixture(scope="session")
def experiment_2638_data():
    """Real experiment 2638 data for integration testing."""
    return Path("examples/blackchirp_data/2638")

@pytest.fixture
def sample_pipeline_file(tmp_path, experiment_2638_data):
    """Pre-created pipeline file with Stage 0 complete."""
    pipeline_file = tmp_path / "sample.ftmw"
    Pipeline.create(pipeline_file, source=experiment_2638_data)
    return pipeline_file

@pytest.fixture
def processed_pipeline_file(tmp_path, experiment_2638_data):
    """Pipeline file with Stages 0-1 complete."""
    pipeline_file = tmp_path / "processed.ftmw"
    pipe = Pipeline.create(pipeline_file, source=experiment_2638_data)
    pipe.compute_ft(zpf=2, expf_us=5.0)
    return pipeline_file
```

### Test File Management
```python
# Utilities for managing test files
def create_test_pipeline_files(tmp_path, count=3):
    """Create multiple test pipeline files for batch testing."""
    files = []
    for i in range(count):
        file_path = tmp_path / f"test_{i}.ftmw"
        pipe = Pipeline.create(file_path, source="test_data")
        files.append(file_path)
    return files

def compare_pipeline_files(file1, file2, stages=None):
    """Compare scientific results between pipeline files."""
    info1 = ftmw.get_info(file1)
    info2 = ftmw.get_info(file2)
    
    if stages is None:
        stages = info1['stages_complete']
    
    for stage in stages:
        data1 = load_stage_data(file1, stage)
        data2 = load_stage_data(file2, stage)
        assert_scientific_equivalence(data1, data2)
```

## Test Execution Strategy

### Continuous Integration
```yaml
# Example GitHub Actions workflow
test_matrix:
  - unit_tests: Fast algorithm and component tests
  - integration_cli: CLI workflow tests  
  - integration_python: Pipeline class and functional API tests
  - consistency: Cross-interface comparison tests
  - performance: Timing and memory benchmarks
```

### Test Organization
```
tests/
├── unit/                          # Existing unit tests (update for .ftmw)
│   ├── core/                      # Data structures and algorithms
│   ├── io/                        # File loading and serialization  
│   └── preprocessing/             # Individual processing functions
├── integration/                   # New integration test suite
│   ├── cli/                       # CLI workflow tests
│   ├── pipeline_class/            # Pipeline class workflow tests  
│   ├── functional_api/            # Functional API workflow tests
│   └── consistency/               # Cross-interface consistency tests
├── fixtures/                      # Shared test data and utilities
│   ├── sample_data/               # Real experiment data
│   ├── pipeline_generators.py    # Test file creation utilities
│   └── comparison_utils.py       # Result comparison functions
└── conftest.py                    # Shared pytest configuration
```

## Implementation Priorities

### Phase 1: Unit Test Updates
1. **File Extension Updates**: Change test artifacts to use `.ftmw` extensions
2. **Serialization Tests**: Update to use new pipeline file format
3. **Fixture Updates**: Ensure test fixtures generate proper pipeline files

### Phase 2: Integration Test Implementation  
1. **CLI Integration Tests**: Complete command-line workflow validation
2. **Pipeline Class Tests**: Object-oriented API workflow validation
3. **Functional API Tests**: Stateless function workflow validation

### Phase 3: Consistency and Advanced Testing
1. **Cross-Interface Consistency**: Verify identical results across interfaces
2. **Error Handling Tests**: Comprehensive error scenario coverage
3. **Performance Tests**: Timing and memory usage validation
4. **Batch Processing Tests**: Multi-file operation validation

## Quality Assurance Goals

### Coverage Targets
- **Unit Test Coverage**: >95% for core algorithms and data structures
- **Integration Coverage**: 100% of user-facing workflows
- **Cross-Interface Coverage**: All major operations tested across all interfaces

### Scientific Accuracy
- **Bit-Perfect Consistency**: Identical results across interfaces for identical inputs
- **Parameter Validation**: Comprehensive validation of all user inputs
- **Error Boundaries**: Clear error messages and proper exception handling

### User Experience
- **Jupyter Safety**: All patterns safe for notebook re-execution
- **File Management**: Robust handling of file creation, opening, and error scenarios
- **Performance**: Integration tests complete within reasonable time bounds

## Migration Strategy

### Existing Test Updates
1. **Identify Tests to Update**: Survey existing tests for cache/experiment ID patterns
2. **Update Test Patterns**: Convert to `.ftmw` file-based patterns
3. **Fixture Migration**: Update test fixtures for new architecture
4. **Backward Compatibility**: Ensure migration doesn't break existing functionality

### New Test Development
1. **CLI Test Framework**: Establish patterns for command-line testing
2. **API Test Patterns**: Create reusable patterns for Python API testing  
3. **Consistency Test Framework**: Build comparison utilities for cross-interface testing
4. **Documentation**: Comprehensive testing documentation and examples

This testing strategy ensures the dual-interface architecture maintains scientific accuracy, user experience quality, and robust error handling across all usage patterns.

---

**Last Updated**: 2025-01-11  
**Status**: Strategy complete, ready for implementation  
**Dependencies**: Requires API_STRATEGY.md and CLI_STRATEGY.md implementation