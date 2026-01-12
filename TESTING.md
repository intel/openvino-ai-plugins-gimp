# Testing Guide for GIMP OpenVINO AI Plugins

This document provides comprehensive information about the testing infrastructure for the GIMP OpenVINO AI Plugins project.

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Linux/macOS
./run_tests.sh

# Windows
run_tests.bat

# Or directly with pytest
pytest
```

### Run Smoke Tests Only

```bash
# Linux/macOS
./run_tests.sh smoke

# Windows
run_tests.bat smoke

# Or directly
pytest -m smoke
```

## Testing Infrastructure Overview

The project uses **pytest** as the primary testing framework with the following structure:

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Shared fixtures and configuration
├── README.md                        # Detailed testing documentation
├── unit/                            # Unit tests directory
├── integration/                     # Integration tests directory
├── fixtures/                        # Test data and fixtures
├── test_socket_communication.py     # Socket server tests (ports 65432, 65433, 65434)
├── test_model_loading.py            # Model loading and OpenVINO tests
└── test_plugin_registration.py     # GIMP plugin registration tests
```

## Test Categories

Tests are organized using pytest markers:

| Marker | Description | Command |
|--------|-------------|---------|
| `smoke` | Quick tests for basic functionality | `pytest -m smoke` |
| `unit` | Tests for individual components | `pytest -m unit` |
| `integration` | Tests for component interaction | `pytest -m integration` |
| `socket` | Socket communication tests | `pytest -m socket` |
| `model` | Model loading tests | `pytest -m model` |
| `plugin` | GIMP plugin interface tests | `pytest -m plugin` |

## Available Test Commands

### Using Test Runner Scripts

**Linux/macOS:**
```bash
./run_tests.sh {all|smoke|unit|integration|socket|model|plugin|coverage|quick}
```

**Windows:**
```cmd
run_tests.bat {all|smoke|unit|integration|socket|model|plugin|coverage|quick}
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_socket_communication.py

# Run specific test function
pytest tests/test_socket_communication.py::test_main_server_basic_connection

# Run tests matching a pattern
pytest -k "socket"

# Run with coverage
pytest --cov=gimpopenvino --cov-report=html

# Run and stop at first failure
pytest -x

# Run last failed tests
pytest --lf
```

## Test Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=gimpopenvino --cov-report=term-missing

# HTML report (view in browser)
pytest --cov=gimpopenvino --cov-report=html
# Open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=gimpopenvino --cov-report=xml
```

### Coverage Configuration

Coverage settings are configured in `pytest.ini`:
- Source: `gimpopenvino` package
- Minimum threshold: 0% (can be increased as test coverage improves)
- Omitted: test files, `__pycache__`, site-packages

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

### Path Fixtures
- `test_data_dir` - Path to test data directory
- `mock_weights_dir` - Temporary weights directory
- `mock_config_dir` - Temporary config directory

### Socket Fixtures
- `mock_socket` - Mock socket object
- `socket_server_factory` - Factory for creating test socket servers
- `mock_socket_connection` - Mock socket connection

### OpenVINO Fixtures
- `mock_openvino_core` - Mock OpenVINO Core
- `mock_stable_diffusion_engine` - Mock SD engine
- `mock_model_path` - Mock model directory with config

### GIMP Fixtures
- `mock_gimp` - Mock GIMP module
- `mock_gimp_ui` - Mock GimpUi module
- `mock_gimp_image` - Mock GIMP image object
- `mock_gimp_layer` - Mock GIMP layer object

### Data Fixtures
- `sample_prompt` - Sample text prompt
- `sample_image_array` - Sample numpy image array
- `mock_sd_config` - Mock Stable Diffusion configuration

## Writing New Tests

### Test File Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Basic Test Template

```python
import pytest

@pytest.mark.smoke  # Add appropriate markers
@pytest.mark.socket
def test_example_functionality(mock_socket):
    """
    Test description explaining what is being tested.
    
    This test verifies that...
    """
    # Arrange
    expected_value = "test"
    
    # Act
    result = some_function(mock_socket)
    
    # Assert
    assert result == expected_value
```

### Using Fixtures

```python
def test_with_multiple_fixtures(mock_openvino_core, mock_weights_dir, tmp_path):
    """Test using multiple fixtures."""
    # All fixtures are automatically injected
    assert mock_weights_dir.exists()
    # Your test logic here
```

## Current Test Coverage

### Socket Communication Tests (test_socket_communication.py)

✅ **Port 65432 (Main SD Server):**
- Basic connection
- Ping command
- Model name query
- Multiple sequential connections

✅ **Port 65433 (Handshake Server):**
- Ready signal reception
- Basic connection handling

✅ **Port 65434 (Model Management Server):**
- Basic connection
- Ping command
- Get all models query

✅ **Error Handling:**
- Connection refused
- Socket timeout
- SO_REUSEADDR option

### Model Loading Tests (test_model_loading.py)

✅ **OpenVINO Core:**
- Core initialization
- Device detection and selection
- Available devices query

✅ **Model Paths:**
- Path resolution for various model types
- Config file handling
- Multiple model configurations

✅ **Model Initialization:**
- Engine mocking
- Device list configuration
- Different model type patterns

✅ **Configuration:**
- Power mode settings
- Device selection from config
- Default fallback handling

✅ **Model Management:**
- ModelManager initialization
- Model details retrieval
- Installation status tracking

✅ **FastSD Models:**
- Config loading
- Model name normalization

### Plugin Registration Tests (test_plugin_registration.py)

✅ **Plugin Structure:**
- Base class structure
- Procedure query
- i18n configuration

✅ **Procedure Registration:**
- ImageProcedure creation
- Documentation setup
- Menu registration
- Attribution configuration

✅ **Parameter Registration:**
- Integer arguments
- Double/float arguments
- String arguments
- Boolean arguments
- Multiple parameters

✅ **Plugin-Specific:**
- Stable Diffusion plugin
- Super Resolution plugin
- Semantic Segmentation plugin
- FastSD plugin

✅ **UI Initialization:**
- GimpUi initialization
- Dialog creation and configuration

✅ **GIMP Interfaces:**
- Image interface
- Layer interface
- Layer creation

## Continuous Integration

An example GitHub Actions workflow is provided in `.github/workflows/tests.yml.example`.

To enable CI:
1. Rename `tests.yml.example` to `tests.yml`
2. Adjust the configuration as needed
3. Commit to your repository

The workflow will:
- Run on push and pull requests
- Test on multiple OS (Ubuntu, Windows, macOS)
- Test on multiple Python versions (3.10, 3.11, 3.12)
- Generate coverage reports
- Upload to Codecov (optional)

## Troubleshooting

### Common Issues

**"Address already in use" errors:**
```bash
# Wait a few seconds for ports to be released
sleep 5 && pytest

# Or check for processes using the ports
lsof -i :65432  # Linux/macOS
netstat -ano | findstr :65432  # Windows
```

**Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/project

# Reinstall dev dependencies
pip install -r requirements-dev.txt
```

**Tests not discovered:**
```bash
# Verify pytest can find tests
pytest --collect-only

# Check test file naming (must start with test_)
# Check function naming (must start with test_)
```

**GIMP/OpenVINO not installed:**
- Tests use mocks - no actual installation required
- If you see import errors, check that mocking is properly configured

## Best Practices

1. **Write tests before fixing bugs** - Helps prevent regressions
2. **Use descriptive test names** - Clearly describe what is being tested
3. **Keep tests focused** - One test should verify one thing
4. **Use fixtures** - Avoid duplication with shared fixtures
5. **Add markers** - Categorize tests for easier filtering
6. **Document complex tests** - Add docstrings explaining the test
7. **Mock external dependencies** - Use mocks for GIMP, OpenVINO, sockets
8. **Clean up resources** - Ensure sockets and files are properly closed

## Next Steps

1. **Increase coverage** - Add more unit tests for individual functions
2. **Add integration tests** - Test full workflows end-to-end
3. **Performance tests** - Add benchmarks for critical paths
4. **Expand fixtures** - Add more reusable test fixtures as needed
5. **CI/CD integration** - Set up automated testing on commits

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

## Support

For questions or issues with the test suite:
1. Check this documentation
2. Review test examples in the test files
3. Check the detailed README in `tests/README.md`
4. Open an issue on the project repository
