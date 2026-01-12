@echo off
REM Test runner script for GIMP OpenVINO AI Plugins (Windows)

echo ==================================================
echo GIMP OpenVINO AI Plugins - Test Suite
echo ==================================================
echo.

REM Check if pytest is installed
where pytest >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: pytest is not installed.
    echo Please install test dependencies: pip install -r requirements-dev.txt
    exit /b 1
)

REM Parse command line arguments
set TEST_TYPE=%1
if "%TEST_TYPE%"=="" set TEST_TYPE=all

if "%TEST_TYPE%"=="smoke" (
    echo Running smoke tests...
    pytest -v -m smoke
) else if "%TEST_TYPE%"=="unit" (
    echo Running unit tests...
    pytest -v -m unit
) else if "%TEST_TYPE%"=="integration" (
    echo Running integration tests...
    pytest -v -m integration
) else if "%TEST_TYPE%"=="socket" (
    echo Running socket communication tests...
    pytest -v -m socket
) else if "%TEST_TYPE%"=="model" (
    echo Running model loading tests...
    pytest -v -m model
) else if "%TEST_TYPE%"=="plugin" (
    echo Running plugin registration tests...
    pytest -v -m plugin
) else if "%TEST_TYPE%"=="coverage" (
    echo Running all tests with coverage report...
    pytest -v --cov=gimpopenvino --cov-report=term-missing --cov-report=html
    echo.
    echo Coverage report generated in htmlcov\index.html
) else if "%TEST_TYPE%"=="quick" (
    echo Running quick smoke tests only...
    pytest -v -m smoke --tb=short
) else if "%TEST_TYPE%"=="all" (
    echo Running all tests...
    pytest -v
) else (
    echo Usage: %0 {all^|smoke^|unit^|integration^|socket^|model^|plugin^|coverage^|quick}
    echo.
    echo Test categories:
    echo   all         - Run all tests
    echo   smoke       - Run smoke tests only
    echo   unit        - Run unit tests only
    echo   integration - Run integration tests only
    echo   socket      - Run socket communication tests
    echo   model       - Run model loading tests
    echo   plugin      - Run plugin registration tests
    echo   coverage    - Run all tests with coverage report
    echo   quick       - Run quick smoke tests
    exit /b 1
)

echo.
echo ==================================================
echo Test run complete!
echo ==================================================
