# Testing Documentation for GIMP OpenVINO AI Plugins

This directory contains the testing infrastructure for the GIMP OpenVINO AI Plugins project.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Test Organization](#test-organization)
- [Writing New Tests](#writing-new-tests)
- [Test Patterns](#test-patterns)
- [Coverage](#coverage)

## Overview

The test suite uses **pytest** as the testing framework with the following key features:

- **Smoke tests** for basic functionality verification
- **Unit tests** for individual component testing
- **Integration tests** for component interaction testing
- **Mocking patterns** for external dependencies (OpenVINO, GIMP, sockets)
- **Coverage reporting** via pytest-cov

## Setup

### Install Testing Dependencies

Install the development dependencies required for testing:

```bash
pip install -r requirements-dev.txt
```

This will install:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- pytest-asyncio (async test support)
- pytest-mock (enhanced mocking)
- pytest-timeout (test timeout management)

### Verify Installation

```bash
pytest --version
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

Using markers to run specific test types:

```bash
# Run only smoke tests
pytest -m smoke

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run socket-related tests
pytest -m socket

# Run model-related tests
pytest -m model
```

### Run Specific Test Files

```bash
# Run socket communication tests
pytest tests/test_socket_communication.py

# Run model loading tests
pytest tests/test_model_loading.py

# Run plugin registration tests
pytest tests/test_plugin_registration.py
```

### Run Specific Test Functions

```bash
pytest tests/test_socket_communication.py::test_main_server_basic_connection
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
# Terminal coverage report
pytest --cov=gimpopenvino --cov-report=term-missing

# HTML coverage report (opens in htmlcov/index.html)
pytest --cov=gimpopenvino --cov-report=html

# XML coverage report (for CI/CD)
pytest --cov=gimpopenvino --cov-report=xml
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── README.md                # This file
├── unit/                    # Unit tests for individual components
│   └── __init__.py
├── integration/             # Integration tests for component interaction
│   └── __init__.py
├── fixtures/                # Test data and fixture files
│   └── __init__.py
├── test_socket_communication.py  # Socket server smoke tests
├── test_model_loading.py         # Model loading smoke tests
└── test_plugin_registration.py   # Plugin registration smoke tests
```

### Test Categories

- **Smoke Tests** (`@pytest.mark.smoke`): Quick tests to verify basic functionality
- **Unit Tests** (`@pytest.mark.unit`): Test individual functions/classes in isolation
- **Integration Tests** (`@pytest.mark.integration`): Test component interactions
- **Socket Tests** (`@pytest.mark.socket`): Test socket communication
- **Model Tests** (`@pytest.mark.model`): Test model loading and inference
- **Plugin Tests** (`@pytest.mark.plugin`): Test GIMP plugin interfaces

## Writing New Tests

### Basic Test Structure

```python
import pytest

@pytest.mark.smoke
def test_example(mock_socket):
    """Test description."""
    # Arrange
    expected = "test_value"
    
    # Act
    result = some_function(mock_socket)
    
    # Assert
    assert result == expected
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used by including them as function parameters:

```python
def test_with_fixtures(mock_openvino_core, mock_weights_dir):
    """Test using multiple fixtures."""
    # Fixtures are automatically injected
    assert mock_weights_dir.exists()
```

### Available Fixtures

See `conftest.py` for all available fixtures. Key fixtures include:

- **Path Fixtures**: `test_data_dir`, `mock_weights_dir`, `mock_config_dir`
- **Socket Fixtures**: `mock_socket`, `socket_server_factory`, `mock_socket_connection`
- **OpenVINO Fixtures**: `mock_openvino_core`, `mock_stable_diffusion_engine`, `mock_model_path`
- **GIMP Fixtures**: `mock_gimp`, `mock_gimp_ui`, `mock_gimp_image`, `mock_gimp_layer`
- **Data Fixtures**: `sample_prompt`, `sample_image_array`, `mock_sd_config`

## Test Patterns

### Pattern 1: Mocking Socket Communication

```python
@pytest.mark.socket
def test_socket_connection(socket_server_factory):
    """Test socket server connection."""
    # Start a test server
    server = socket_server_factory(port=65432, response=b"Ready")
    
    try:
        # Connect to server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("127.0.0.1", 65432))
            client.sendall(b"ping")
            response = client.recv(1024)
            assert response == b"Ready"
    finally:
        server.shutdown()
```

### Pattern 2: Mocking OpenVINO Models

```python
@pytest.mark.model
def test_model_loading(mock_openvino_core, mock_model_path):
    """Test OpenVINO model loading."""
    from unittest.mock import patch
    
    with patch('openvino.Core', return_value=mock_openvino_core):
        # Your model loading code here
        core = openvino.Core()
        assert "CPU" in core.available_devices
```

### Pattern 3: Mocking GIMP Plugin Interface

```python
@pytest.mark.plugin
def test_plugin_registration(mock_gimp):
    """Test GIMP plugin registration."""
    from unittest.mock import patch
    
    with patch('gi.repository.Gimp', mock_gimp):
        # Your plugin code here
        procedure = mock_gimp.ImageProcedure.new()
        assert procedure is not None
```

### Pattern 4: Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

### Pattern 5: Testing with Temporary Files

```python
def test_config_file(tmp_path):
    """Test configuration file handling."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"key": "value"}')
    
    # Test code that reads config
    assert config_file.exists()
```

## Coverage

### Viewing Coverage Reports

After running tests with coverage:

```bash
pytest --cov=gimpopenvino --cov-report=html
```

Open `htmlcov/index.html` in your browser to see detailed coverage information.

### Coverage Thresholds

The project is configured with a minimum coverage threshold of 0% (`--cov-fail-under=0` in pytest.ini). This allows tests to pass even with low initial coverage as the test suite is being built out.

To set a minimum threshold:

```bash
pytest --cov=gimpopenvino --cov-fail-under=80
```

## Continuous Integration

The test suite is designed to run in CI/CD environments. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=gimpopenvino --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Socket address already in use:**
- Tests create socket servers that may not clean up immediately
- Wait a few seconds and retry
- Check for orphaned processes: `lsof -i :65432`

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements-dev.txt`
- Check PYTHONPATH includes project root

**GIMP/OpenVINO not available:**
- Tests use mocks, so actual GIMP/OpenVINO installation is not required
- If errors persist, ensure mocks are properly configured in conftest.py

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.smoke`, etc.)
3. Include docstrings describing what the test verifies
4. Use fixtures from `conftest.py` where applicable
5. Update this README if adding new test patterns or fixtures
