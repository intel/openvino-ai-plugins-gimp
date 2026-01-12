"""
Shared pytest fixtures and configuration for GIMP OpenVINO AI Plugins tests.

This module provides common fixtures for:
- Mock socket connections
- Test data paths
- Mock OpenVINO models
- Mock GIMP interfaces
"""

import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def mock_weights_dir(tmp_path):
    """Create a temporary weights directory for testing."""
    weights = tmp_path / "weights"
    weights.mkdir()
    return weights


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary config directory for testing."""
    config = tmp_path / "config"
    config.mkdir()
    return config


# ============================================================================
# Socket Fixtures
# ============================================================================

@pytest.fixture
def mock_socket():
    """Create a mock socket object."""
    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.recv.return_value = b"test_data"
    mock_sock.accept.return_value = (MagicMock(), ("127.0.0.1", 12345))
    return mock_sock


@pytest.fixture
def socket_server_factory():
    """
    Factory fixture for creating test socket servers.
    
    Usage:
        server = socket_server_factory(port=65432, response=b"Ready")
        # ... run tests ...
        server.shutdown()
    """
    def _create_server(port, response=b"OK", host="127.0.0.1"):
        stop_event = threading.Event()
        
        def server_thread():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(1.0)  # Timeout to allow periodic checking of stop_event
                try:
                    s.bind((host, port))
                    s.listen()
                    while not stop_event.is_set():
                        try:
                            conn, addr = s.accept()
                            with conn:
                                data = conn.recv(1024)
                                if data:
                                    conn.sendall(response)
                        except socket.timeout:
                            continue
                        except Exception:
                            break
                except Exception as e:
                    print(f"Server error on port {port}: {e}")
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        time.sleep(0.1)  # Give server time to start
        
        class ServerControl:
            def shutdown(self):
                stop_event.set()
                thread.join(timeout=2.0)
        
        return ServerControl()
    
    return _create_server


@pytest.fixture
def mock_socket_connection():
    """Mock socket connection for client-side testing."""
    mock_conn = MagicMock()
    mock_conn.recv.return_value = b"test_response"
    mock_conn.sendall.return_value = None
    return mock_conn


# ============================================================================
# OpenVINO Fixtures
# ============================================================================

@pytest.fixture
def mock_openvino_core():
    """Create a mock OpenVINO Core object."""
    with patch('openvino.Core') as mock_core:
        core_instance = MagicMock()
        mock_core.return_value = core_instance
        core_instance.available_devices = ["CPU", "GPU"]
        core_instance.read_model.return_value = MagicMock()
        core_instance.compile_model.return_value = MagicMock()
        yield core_instance


@pytest.fixture
def mock_stable_diffusion_engine():
    """Create a mock Stable Diffusion engine."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = MagicMock()  # Mock generated image
    return mock_engine


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model directory with config."""
    model_dir = tmp_path / "stable-diffusion-ov" / "stable-diffusion-1.5"
    model_dir.mkdir(parents=True)
    
    config = {
        "power modes supported": "yes",
        "best performance": ["CPU", "CPU", "CPU", "CPU"],
        "balanced": ["CPU", "CPU", "CPU", "CPU"]
    }
    
    config_file = model_dir / "config.json"
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    return model_dir


# ============================================================================
# GIMP Fixtures
# ============================================================================

@pytest.fixture
def mock_gimp():
    """Mock GIMP module and basic structures."""
    gimp_mock = MagicMock()
    
    # Mock Gimp.PlugIn
    gimp_mock.PlugIn = MagicMock
    
    # Mock Gimp.ImageProcedure
    procedure_mock = MagicMock()
    gimp_mock.ImageProcedure.new.return_value = procedure_mock
    
    # Mock common GIMP enums
    gimp_mock.RunMode.INTERACTIVE = 1
    gimp_mock.RunMode.NONINTERACTIVE = 0
    gimp_mock.PDBProcType.PLUGIN = 1
    
    return gimp_mock


@pytest.fixture
def mock_gimp_ui():
    """Mock GimpUi module for UI-related tests."""
    gimp_ui_mock = MagicMock()
    gimp_ui_mock.init.return_value = None
    gimp_ui_mock.Dialog.return_value = MagicMock()
    return gimp_ui_mock


@pytest.fixture
def mock_gimp_image():
    """Create a mock GIMP image object."""
    image = MagicMock()
    image.get_width.return_value = 512
    image.get_height.return_value = 512
    image.list_layers.return_value = []
    return image


@pytest.fixture
def mock_gimp_layer():
    """Create a mock GIMP layer object."""
    layer = MagicMock()
    layer.get_width.return_value = 512
    layer.get_height.return_value = 512
    return layer


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_prompt():
    """Sample text prompt for image generation."""
    return "a beautiful landscape with mountains and a lake"


@pytest.fixture
def sample_image_array():
    """Create a sample numpy array representing an image."""
    import numpy as np
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def mock_sd_config():
    """Mock Stable Diffusion configuration."""
    return {
        "model_name": "sd_1.5_square",
        "prompt": "test prompt",
        "num_infer_steps": 20,
        "guidance_scale": 7.5,
        "src_width": 512,
        "src_height": 512,
        "seed": None,
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_sockets():
    """Ensure all sockets are cleaned up after each test."""
    yield
    # Cleanup happens after test
    time.sleep(0.1)  # Brief pause to allow socket cleanup
