# Testing Quick Start Guide

## Installation

```bash
# Install testing dependencies
pip install -r requirements-dev.txt
```

## Running Tests

### Quick Commands

```bash
# Run all tests
pytest

# Run smoke tests only (recommended for quick validation)
pytest -m smoke

# Run with coverage report
pytest --cov=gimpopenvino --cov-report=term-missing
```

### Using Test Runner Scripts

**Linux/macOS:**
```bash
chmod +x run_tests.sh
./run_tests.sh smoke       # Quick smoke tests
./run_tests.sh coverage    # Full coverage report
./run_tests.sh all         # All tests
```

**Windows:**
```cmd
run_tests.bat smoke
run_tests.bat coverage
run_tests.bat all
```

## Test Categories

| Command | Description | Test Count |
|---------|-------------|------------|
| `pytest -m smoke` | Quick validation tests | 25+ tests |
| `pytest -m unit` | Unit tests | 10+ tests |
| `pytest -m integration` | Integration tests | 2+ tests |
| `pytest -m socket` | Socket communication | 12+ tests |
| `pytest -m model` | Model loading | 20+ tests |
| `pytest -m plugin` | Plugin registration | 35+ tests |

## What's Tested

✅ **Socket Communication** (Ports 65432, 65433, 65434)
- Main server connections
- Handshake protocols
- Model management server

✅ **Model Loading**
- OpenVINO initialization
- Device detection
- Model path resolution
- Configuration handling

✅ **Plugin Registration**
- GIMP plugin interfaces
- Procedure registration
- Parameter handling
- UI initialization

## Expected Output

When tests pass, you'll see:
```
======================== test session starts ========================
collected 70 items

tests/test_socket_communication.py::test_main_server_basic_connection PASSED
tests/test_model_loading.py::test_openvino_core_initialization PASSED
tests/test_plugin_registration.py::test_plugin_class_structure PASSED
...

======================== 70 passed in 5.23s =========================
```

## Troubleshooting

**"Address already in use" error:**
- Wait a few seconds and retry
- Kill any processes using test ports: `lsof -i :65432` (Linux/macOS)

**Import errors:**
- Ensure you're in the project root directory
- Reinstall: `pip install -r requirements-dev.txt`

**No tests collected:**
- Verify you're in the correct directory
- Run: `pytest --collect-only` to see what pytest finds

## Documentation

📚 **Detailed Guides:**
- `TESTING.md` - Comprehensive testing documentation
- `tests/README.md` - Detailed test patterns and examples
- `tests/TEST_SUMMARY.md` - Infrastructure summary

## Next Steps

1. Run smoke tests to verify setup: `pytest -m smoke`
2. Check coverage: `pytest --cov=gimpopenvino --cov-report=html`
3. Open `htmlcov/index.html` to view coverage details
4. Read `tests/README.md` to learn about writing new tests

---

**Need help?** See the full documentation in `TESTING.md`
