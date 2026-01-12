#!/bin/bash
# Test runner script for GIMP OpenVINO AI Plugins

set -e

echo "=================================================="
echo "GIMP OpenVINO AI Plugins - Test Suite"
echo "=================================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed."
    echo "Please install test dependencies: pip install -r requirements-dev.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
    smoke)
        echo "Running smoke tests..."
        pytest -v -m smoke
        ;;
    unit)
        echo "Running unit tests..."
        pytest -v -m unit
        ;;
    integration)
        echo "Running integration tests..."
        pytest -v -m integration
        ;;
    socket)
        echo "Running socket communication tests..."
        pytest -v -m socket
        ;;
    model)
        echo "Running model loading tests..."
        pytest -v -m model
        ;;
    plugin)
        echo "Running plugin registration tests..."
        pytest -v -m plugin
        ;;
    coverage)
        echo "Running all tests with coverage report..."
        pytest -v --cov=gimpopenvino --cov-report=term-missing --cov-report=html
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    quick)
        echo "Running quick smoke tests only..."
        pytest -v -m smoke --tb=short
        ;;
    all)
        echo "Running all tests..."
        pytest -v
        ;;
    *)
        echo "Usage: $0 {all|smoke|unit|integration|socket|model|plugin|coverage|quick}"
        echo ""
        echo "Test categories:"
        echo "  all         - Run all tests"
        echo "  smoke       - Run smoke tests only"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  socket      - Run socket communication tests"
        echo "  model       - Run model loading tests"
        echo "  plugin      - Run plugin registration tests"
        echo "  coverage    - Run all tests with coverage report"
        echo "  quick       - Run quick smoke tests"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Test run complete!"
echo "=================================================="
