# Initializing the MODEL_SETUP flag
$MODEL_SETUP = 0

# Check the first argument ($args[0])
if ($args.Length -gt 0) {
    if ($args[0] -ieq '-i' -or $args[0] -ieq '--install_models') {
        $MODEL_SETUP = 1
    } else {
        Write-Host "Invalid option: $($args[0])"
        Write-Host "Use -i or --install_models to run model setup."
        exit 1
    }
}

# Get the directory of the currently executing script
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Get the current working directory
$current_dir = Get-Location

# Compare the directories
if ($script_dir -ieq $current_dir) {
    # If they are the same, move up one directory
    Set-Location ..
}

Write-Host "**** openvino-ai-plugins-gimp Setup started ****"

# Install virtualenv if not already installed
$virtualenv_install = python -m pip install virtualenv

# Create a virtual environment
python -m virtualenv gimpenv3

# Activate the virtual environment
& gimpenv3\Scripts\Activate.ps1

# Install required packages
pip install wmi
pip install -r "$script_dir\requirements.txt"
pip install "$script_dir\."

# Run Python script to complete the installation
python -c "from gimpopenvino import complete_install; complete_install.setup_python_weights()"

Write-Host "**** openvino-ai-plugins-gimp Setup Ended ****"

# Deactivate the virtual environment
deactivate

# Copy to GIMP plugin directory
$pluginDir = "$env:APPDATA\GIMP\2.99\plug-ins"
Write-Host "Installing plugin in $pluginDir"

$plugins = @('openvino_utils', 'semseg_ov', 'stable_diffusion_ov', 'superresolution_ov')
foreach ($plugin in $plugins) {
    robocopy "gimpenv3\Lib\site-packages\gimpopenvino\plugins\$plugin" "$pluginDir\$plugin" /mir /NFL /NDL /NJH /NJS /nc /ns /np
}

Write-Host "*** openvino-ai-plugins-gimp Installed ***"

# If MODEL_SETUP was set, run the model setup
if ($MODEL_SETUP -eq 1) {
    Write-Host "**** OpenVINO MODEL SETUP STARTED ****"
    & gimpenv3\Scripts\python.exe "$script_dir\model_setup.py"
}

exit 0
