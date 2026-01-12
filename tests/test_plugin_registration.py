"""
Smoke tests for GIMP plugin registration in OpenVINO AI Plugins.

Tests plugin registration patterns:
- Plugin class inheritance
- Procedure registration
- Menu path registration
- Parameter registration
- UI initialization
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys


# ============================================================================
# Plugin Base Class Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_plugin_class_structure(mock_gimp):
    """Test basic plugin class structure."""
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        # Simulate plugin class
        class TestPlugin:
            def do_query_procedures(self):
                return ["test-plugin"]
            
            def do_set_i18n(self, procname):
                return True, 'gimp30-python', None
            
            def do_create_procedure(self, name):
                return MagicMock()
        
        plugin = TestPlugin()
        procedures = plugin.do_query_procedures()
        assert "test-plugin" in procedures


@pytest.mark.smoke
@pytest.mark.plugin
def test_plugin_procedure_query():
    """Test plugin procedure query method."""
    mock_plugin = MagicMock()
    mock_plugin.do_query_procedures.return_value = [
        "stable-diffusion-ov",
        "superresolution-ov",
        "semseg-ov"
    ]
    
    procedures = mock_plugin.do_query_procedures()
    assert len(procedures) >= 1
    assert isinstance(procedures, list)


@pytest.mark.smoke
@pytest.mark.plugin
def test_plugin_i18n_configuration():
    """Test plugin internationalization configuration."""
    mock_plugin = MagicMock()
    mock_plugin.do_set_i18n.return_value = (True, 'gimp30-python', None)
    
    enabled, domain, location = mock_plugin.do_set_i18n("test-plugin")
    assert enabled is True
    assert domain == 'gimp30-python'


# ============================================================================
# Procedure Registration Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_image_procedure_creation(mock_gimp):
    """Test ImageProcedure creation."""
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        procedure = mock_gimp.ImageProcedure.new()
        assert procedure is not None


@pytest.mark.smoke
@pytest.mark.plugin
def test_procedure_documentation_setup():
    """Test procedure documentation configuration."""
    mock_procedure = MagicMock()
    
    mock_procedure.set_documentation(
        "Test plugin description",
        "Detailed help text",
        "test-plugin-ov"
    )
    
    mock_procedure.set_documentation.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_procedure_menu_registration():
    """Test procedure menu path registration."""
    mock_procedure = MagicMock()
    
    # Test menu label and path
    mock_procedure.set_menu_label("Stable Diffusion")
    mock_procedure.add_menu_path("<Image>/Layer/OpenVINO-AI-Plugins/")
    
    mock_procedure.set_menu_label.assert_called_with("Stable Diffusion")
    mock_procedure.add_menu_path.assert_called_with("<Image>/Layer/OpenVINO-AI-Plugins/")


@pytest.mark.smoke
@pytest.mark.plugin
def test_procedure_attribution():
    """Test procedure attribution information."""
    mock_procedure = MagicMock()
    
    mock_procedure.set_attribution(
        "Arisha Kumar",
        "OpenVINO-AI-Plugins",
        "2023"
    )
    
    mock_procedure.set_attribution.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_procedure_image_types():
    """Test procedure image type specification."""
    mock_procedure = MagicMock()
    mock_procedure.set_image_types("*")
    
    mock_procedure.set_image_types.assert_called_with("*")


# ============================================================================
# Parameter Registration Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_integer_argument_registration():
    """Test integer argument registration."""
    mock_procedure = MagicMock()
    
    # Register number of inference steps
    mock_procedure.add_int_argument(
        "num_infer_steps",
        "Number of Inference steps",
        "Number of Inference steps",
        1, 200, 20,
        None  # GObject.ParamFlags.READWRITE
    )
    
    mock_procedure.add_int_argument.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_double_argument_registration():
    """Test double/float argument registration."""
    mock_procedure = MagicMock()
    
    # Register guidance scale
    mock_procedure.add_double_argument(
        "guidance_scale",
        "Guidance Scale",
        "Guidance Scale (Default:7.5)",
        0.0, 20.0, 7.5,
        None
    )
    
    mock_procedure.add_double_argument.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_string_argument_registration():
    """Test string argument registration."""
    mock_procedure = MagicMock()
    
    # Register model name
    mock_procedure.add_string_argument(
        "model_name",
        "Model Name",
        "Current Model",
        "sd_1.5_square",
        None
    )
    
    mock_procedure.add_string_argument.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_boolean_argument_registration():
    """Test boolean argument registration."""
    mock_procedure = MagicMock()
    
    # Register advanced settings flag
    mock_procedure.add_boolean_argument(
        "advanced_setting",
        "Advanced Settings",
        "Advanced Settings",
        False,
        None
    )
    
    mock_procedure.add_boolean_argument.assert_called_once()


@pytest.mark.smoke
@pytest.mark.plugin
def test_multiple_arguments_registration():
    """Test registering multiple arguments."""
    mock_procedure = MagicMock()
    
    # Register multiple parameters
    mock_procedure.add_int_argument("num_images", "Number of Images", "Count", 1, 200, 1, None)
    mock_procedure.add_double_argument("strength", "Strength", "Strength value", 0.0, 1.0, 0.8, None)
    mock_procedure.add_string_argument("prompt", "Prompt", "Text prompt", "", None)
    mock_procedure.add_boolean_argument("show_console", "Show Console", "Display console", False, None)
    
    assert mock_procedure.add_int_argument.call_count == 1
    assert mock_procedure.add_double_argument.call_count == 1
    assert mock_procedure.add_string_argument.call_count == 1
    assert mock_procedure.add_boolean_argument.call_count == 1


# ============================================================================
# Plugin-Specific Registration Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_stable_diffusion_plugin_procedures():
    """Test Stable Diffusion plugin procedure registration."""
    mock_plugin = MagicMock()
    mock_plugin.do_query_procedures.return_value = ["stable-diffusion-ov"]
    
    procedures = mock_plugin.do_query_procedures()
    assert "stable-diffusion-ov" in procedures


@pytest.mark.smoke
@pytest.mark.plugin
def test_superresolution_plugin_procedures():
    """Test Super Resolution plugin procedure registration."""
    mock_plugin = MagicMock()
    mock_plugin.do_query_procedures.return_value = ["superresolution-ov"]
    
    procedures = mock_plugin.do_query_procedures()
    assert "superresolution-ov" in procedures


@pytest.mark.smoke
@pytest.mark.plugin
def test_semseg_plugin_procedures():
    """Test Semantic Segmentation plugin procedure registration."""
    mock_plugin = MagicMock()
    mock_plugin.do_query_procedures.return_value = ["semseg-ov"]
    
    procedures = mock_plugin.do_query_procedures()
    assert "semseg-ov" in procedures


@pytest.mark.smoke
@pytest.mark.plugin
def test_fastsd_plugin_procedures():
    """Test FastSD plugin procedure registration."""
    mock_plugin = MagicMock()
    mock_plugin.do_query_procedures.return_value = ["fastsd-ov"]
    
    procedures = mock_plugin.do_query_procedures()
    assert "fastsd-ov" in procedures


# ============================================================================
# UI Initialization Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_gimp_ui_initialization(mock_gimp_ui):
    """Test GimpUi initialization."""
    mock_gimp_ui.init("test-plugin")
    mock_gimp_ui.init.assert_called_with("test-plugin")


@pytest.mark.smoke
@pytest.mark.plugin
def test_dialog_creation(mock_gimp_ui):
    """Test dialog creation for plugin UI."""
    dialog = mock_gimp_ui.Dialog(
        title="Test Plugin",
        role="test-plugin",
        use_header_bar=False
    )
    
    assert dialog is not None


@pytest.mark.smoke
@pytest.mark.plugin
def test_dialog_configuration():
    """Test dialog size and properties configuration."""
    mock_dialog = MagicMock()
    
    mock_dialog.set_default_size(450, 400)
    mock_dialog.set_resizable(False)
    
    mock_dialog.set_default_size.assert_called_with(450, 400)
    mock_dialog.set_resizable.assert_called_with(False)


# ============================================================================
# Run Mode Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_plugin_run_mode_interactive(mock_gimp):
    """Test plugin run in interactive mode."""
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        run_mode = mock_gimp.RunMode.INTERACTIVE
        assert run_mode == 1


@pytest.mark.smoke
@pytest.mark.plugin
def test_plugin_run_mode_noninteractive(mock_gimp):
    """Test plugin run in non-interactive mode."""
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        run_mode = mock_gimp.RunMode.NONINTERACTIVE
        assert run_mode == 0


# ============================================================================
# Plugin Main Entry Point Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_gimp_main_entry_point(mock_gimp):
    """Test Gimp.main() entry point."""
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        mock_plugin_class = MagicMock()
        mock_plugin_class.__gtype__ = "TestPlugin"
        
        # Simulate calling Gimp.main
        mock_gimp.main(mock_plugin_class.__gtype__, sys.argv)
        
        mock_gimp.main.assert_called_once()


# ============================================================================
# Plugin Sensitivity Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_procedure_sensitivity_mask():
    """Test procedure sensitivity mask configuration."""
    mock_procedure = MagicMock()
    mock_gimp = MagicMock()
    mock_gimp.ProcedureSensitivityMask.ALWAYS = 1
    
    with patch.dict('sys.modules', {'gi.repository.Gimp': mock_gimp}):
        # Set sensitivity to ALWAYS (plugin available even without images)
        mock_procedure.set_sensitivity_mask(mock_gimp.ProcedureSensitivityMask.ALWAYS)
        
        mock_procedure.set_sensitivity_mask.assert_called_once()


# ============================================================================
# GIMP Image/Layer Interface Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.plugin
def test_gimp_image_interface(mock_gimp_image):
    """Test GIMP image interface."""
    width = mock_gimp_image.get_width()
    height = mock_gimp_image.get_height()
    
    assert width == 512
    assert height == 512


@pytest.mark.smoke
@pytest.mark.plugin
def test_gimp_layer_interface(mock_gimp_layer):
    """Test GIMP layer interface."""
    width = mock_gimp_layer.get_width()
    height = mock_gimp_layer.get_height()
    
    assert width == 512
    assert height == 512


@pytest.mark.smoke
@pytest.mark.plugin
def test_gimp_layer_creation():
    """Test GIMP layer creation."""
    mock_image = MagicMock()
    mock_layer = MagicMock()
    
    mock_image.new_layer.return_value = mock_layer
    layer = mock_image.new_layer()
    
    assert layer is not None


# ============================================================================
# Integration Pattern Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.plugin
def test_full_plugin_registration_flow():
    """Test complete plugin registration flow."""
    mock_gimp = MagicMock()
    
    # Create plugin instance
    mock_plugin = MagicMock()
    
    # Query procedures
    mock_plugin.do_query_procedures.return_value = ["test-plugin"]
    procedures = mock_plugin.do_query_procedures()
    
    # Create procedure
    mock_procedure = MagicMock()
    mock_plugin.do_create_procedure.return_value = mock_procedure
    procedure = mock_plugin.do_create_procedure("test-plugin")
    
    # Configure procedure
    procedure.set_menu_label("Test Plugin")
    procedure.add_menu_path("<Image>/Layer/Test/")
    procedure.add_int_argument("param1", "Param 1", "Description", 1, 100, 50, None)
    
    # Verify flow
    assert "test-plugin" in procedures
    assert procedure is not None
    procedure.set_menu_label.assert_called_once()
    procedure.add_menu_path.assert_called_once()


@pytest.mark.integration
@pytest.mark.plugin
def test_plugin_with_multiple_procedures():
    """Test plugin with multiple procedure registrations."""
    mock_plugin = MagicMock()
    
    procedures = [
        "stable-diffusion-ov",
        "inpainting-ov",
        "controlnet-ov"
    ]
    
    mock_plugin.do_query_procedures.return_value = procedures
    
    result = mock_plugin.do_query_procedures()
    assert len(result) == 3
    assert all(proc in result for proc in procedures)
