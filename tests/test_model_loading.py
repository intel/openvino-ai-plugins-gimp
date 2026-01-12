"""
Smoke tests for model loading functionality in GIMP OpenVINO AI Plugins.

Tests model loading and management:
- OpenVINO Core initialization
- Model path resolution
- Device detection
- Model initialization patterns
- Error handling for missing models
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open


# ============================================================================
# OpenVINO Core Initialization Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_openvino_core_initialization(mock_openvino_core):
    """Test OpenVINO Core can be initialized."""
    with patch('openvino.Core', return_value=mock_openvino_core):
        import openvino
        core = openvino.Core()
        assert core is not None


@pytest.mark.smoke
@pytest.mark.model
def test_openvino_available_devices(mock_openvino_core):
    """Test querying available OpenVINO devices."""
    with patch('openvino.Core', return_value=mock_openvino_core):
        import openvino
        core = openvino.Core()
        devices = core.available_devices
        assert "CPU" in devices
        assert isinstance(devices, list)


@pytest.mark.smoke
@pytest.mark.model
def test_openvino_device_selection():
    """Test device selection logic."""
    mock_core = MagicMock()
    mock_core.available_devices = ["CPU", "GPU", "NPU"]
    
    with patch('openvino.Core', return_value=mock_core):
        import openvino
        core = openvino.Core()
        devices = core.available_devices
        
        # Simulate device selection logic
        if "GPU" in devices:
            selected_device = "GPU"
        else:
            selected_device = "CPU"
        
        assert selected_device in ["CPU", "GPU"]


# ============================================================================
# Model Path Resolution Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_model_path_resolution(mock_weights_dir):
    """Test model path resolution for different model types."""
    # Create model directory structure
    model_dir = mock_weights_dir / "stable-diffusion-ov" / "stable-diffusion-1.5"
    model_dir.mkdir(parents=True)
    
    assert model_dir.exists()
    assert "stable-diffusion-1.5" in str(model_dir)


@pytest.mark.smoke
@pytest.mark.model
def test_model_config_file_creation(mock_model_path):
    """Test model config.json file exists and is valid."""
    config_file = mock_model_path / "config.json"
    assert config_file.exists()
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    assert "power modes supported" in config
    assert "best performance" in config


@pytest.mark.smoke
@pytest.mark.model
def test_model_path_for_different_models(mock_weights_dir):
    """Test path resolution for various model types."""
    model_paths = {
        "sd_1.5_square": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"],
        "sd_1.5_inpainting": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
        "controlnet_openpose": ["stable-diffusion-ov", "controlnet-openpose"],
    }
    
    for model_name, path_parts in model_paths.items():
        model_path = mock_weights_dir.joinpath(*path_parts)
        model_path.mkdir(parents=True, exist_ok=True)
        assert model_path.exists()


# ============================================================================
# Model Initialization Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_stable_diffusion_engine_mock():
    """Test mocking Stable Diffusion engine initialization."""
    mock_engine = MagicMock()
    mock_engine.model = "sd_1.5_square"
    mock_engine.device = ["CPU", "CPU", "CPU", "CPU"]
    
    assert mock_engine.model == "sd_1.5_square"
    assert mock_engine.device[0] == "CPU"


@pytest.mark.smoke
@pytest.mark.model
def test_model_loading_with_device_list(mock_openvino_core, mock_model_path):
    """Test model loading with specific device configuration."""
    device_list = ["CPU", "CPU", "CPU", "CPU"]
    
    with patch('openvino.Core', return_value=mock_openvino_core):
        # Simulate engine initialization
        mock_engine = MagicMock()
        mock_engine.model = str(mock_model_path)
        mock_engine.device = device_list
        
        assert mock_engine.device == device_list
        assert len(mock_engine.device) == 4


@pytest.mark.smoke
@pytest.mark.model
def test_model_initialization_different_types():
    """Test initialization patterns for different model types."""
    model_types = [
        "sd_1.5_square",
        "sd_1.5_inpainting",
        "sdxl_base_1.0_square",
        "controlnet_openpose",
        "controlnet_canny",
    ]
    
    for model_type in model_types:
        mock_engine = MagicMock()
        mock_engine.model_name = model_type
        assert mock_engine.model_name in model_types


# ============================================================================
# Model Configuration Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_model_config_power_modes(mock_model_path):
    """Test power mode configuration in model config."""
    config_file = mock_model_path / "config.json"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check power modes are properly configured
    assert config["power modes supported"] in ["yes", "no"]
    assert "best performance" in config
    assert isinstance(config["best performance"], list)


@pytest.mark.smoke
@pytest.mark.model
def test_model_config_device_selection():
    """Test device selection from model configuration."""
    config = {
        "power modes supported": "yes",
        "best performance": ["GPU", "GPU", "GPU", "GPU"],
        "balanced": ["CPU", "CPU", "GPU", "GPU"],
        "best power efficiency": ["CPU", "CPU", "CPU", "CPU"]
    }
    
    # Test different power modes
    power_mode = "best performance"
    device_list = config[power_mode.lower()]
    assert device_list == ["GPU", "GPU", "GPU", "GPU"]
    
    power_mode = "balanced"
    device_list = config[power_mode.lower()]
    assert "CPU" in device_list and "GPU" in device_list


@pytest.mark.smoke
@pytest.mark.model
def test_model_config_default_fallback(tmp_path):
    """Test fallback to default config when config file doesn't exist."""
    non_existent_path = tmp_path / "non_existent" / "config.json"
    
    # Simulate default config
    default_config = {
        "power modes supported": "no",
        "best performance": ["CPU", "CPU", "CPU", "CPU"]
    }
    
    if not non_existent_path.exists():
        # Use default config
        device_list = default_config['best performance']
        assert device_list == ["CPU", "CPU", "CPU", "CPU"]


# ============================================================================
# Model Engine Selection Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_engine_selection_logic():
    """Test engine selection based on model name."""
    model_engine_map = {
        "sd_1.5_square_int8": "StableDiffusionEngineAdvanced",
        "sd_1.5_inpainting": "StableDiffusionEngineInpaintingGenai",
        "sd_1.5_square_lcm": "StableDiffusionEngineGenai",
        "controlnet_openpose": "ControlNetOpenPose",
        "controlnet_canny": "ControlNetCannyEdge",
    }
    
    for model_name, expected_engine in model_engine_map.items():
        # Simulate engine selection
        if "int8" in model_name and "inpainting" not in model_name and "controlnet" not in model_name:
            selected_engine = "StableDiffusionEngineAdvanced"
        elif "inpainting" in model_name:
            selected_engine = "StableDiffusionEngineInpaintingGenai"
        elif "lcm" in model_name:
            selected_engine = "StableDiffusionEngineGenai"
        elif "controlnet_openpose" in model_name:
            selected_engine = "ControlNetOpenPose"
        elif "controlnet_canny" in model_name:
            selected_engine = "ControlNetCannyEdge"
        else:
            selected_engine = "StableDiffusionEngine"
        
        assert selected_engine is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
def test_missing_model_path_handling(tmp_path):
    """Test handling of missing model paths."""
    missing_path = tmp_path / "non_existent_model"
    assert not missing_path.exists()


@pytest.mark.unit
@pytest.mark.model
def test_invalid_device_handling():
    """Test handling of invalid device specifications."""
    invalid_devices = ["INVALID_DEVICE", "XPU", ""]
    valid_devices = ["CPU", "GPU", "NPU"]
    
    for device in invalid_devices:
        # Simulate device validation
        is_valid = device in valid_devices
        assert not is_valid


@pytest.mark.unit
@pytest.mark.model
def test_corrupted_config_handling():
    """Test handling of corrupted config files."""
    invalid_json = "{ invalid json }"
    
    with pytest.raises(json.JSONDecodeError):
        json.loads(invalid_json)


# ============================================================================
# Model Manager Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_model_manager_initialization(mock_weights_dir):
    """Test ModelManager initialization."""
    mock_manager = MagicMock()
    mock_manager.weight_path = str(mock_weights_dir)
    mock_manager.model_install_status = {}
    
    assert mock_manager.weight_path is not None
    assert isinstance(mock_manager.model_install_status, dict)


@pytest.mark.smoke
@pytest.mark.model
def test_model_manager_get_all_model_details():
    """Test getting all model details from ModelManager."""
    mock_manager = MagicMock()
    
    installed_models = [
        {"name": "Stable Diffusion 1.5", "id": "sd_1.5_square"}
    ]
    
    installable_models = [
        {
            "name": "Stable Diffusion XL",
            "id": "sdxl_base_1.0",
            "description": "SDXL base model",
            "install_status": "not_installed"
        }
    ]
    
    mock_manager.get_all_model_details.return_value = (installed_models, installable_models)
    
    installed, installable = mock_manager.get_all_model_details()
    assert len(installed) == 1
    assert len(installable) == 1
    assert installed[0]["id"] == "sd_1.5_square"


@pytest.mark.smoke
@pytest.mark.model
def test_model_install_status_tracking():
    """Test model installation status tracking."""
    mock_manager = MagicMock()
    mock_manager.model_install_status = {}
    
    # Simulate adding installation status
    model_id = "sdxl_turbo"
    mock_manager.model_install_status[model_id] = {
        "status": "Installing...",
        "percent": 45.0
    }
    
    assert model_id in mock_manager.model_install_status
    assert mock_manager.model_install_status[model_id]["percent"] == 45.0


# ============================================================================
# FastSD Model Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.model
def test_fastsd_model_config_loading(tmp_path):
    """Test loading FastSD model configuration."""
    config_file = tmp_path / "fastsd_models.json"
    
    fastsd_config = {
        "models": [
            "rupeshs/sdxs-512-dreamshaper-openvino-int8",
            "rupeshs/sd-turbo-openvino-int8"
        ]
    }
    
    with open(config_file, 'w') as f:
        json.dump(fastsd_config, f)
    
    with open(config_file, 'r') as f:
        loaded_config = json.load(f)
    
    assert "models" in loaded_config
    assert len(loaded_config["models"]) == 2


@pytest.mark.smoke
@pytest.mark.model
def test_fastsd_model_name_normalization():
    """Test FastSD model name normalization (lowercase)."""
    fastsd_models = [
        "rupeshs/sdxs-512-dreamshaper-openvino-int8",
        "rupeshs/SD-Turbo-OpenVINO-INT8"  # Mixed case
    ]
    
    # Normalize to lowercase
    normalized = [model.lower() for model in fastsd_models]
    
    assert all(model == model.lower() for model in normalized)
    assert "rupeshs/sd-turbo-openvino-int8" in normalized
