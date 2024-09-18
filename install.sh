#!/bin/bash
MODEL_SETUP=0

# Check the first argument ($1)
if [[ -n "$1" ]]; then
    if [[ "$1" == "-i" || "$1" == "--install_models" ]]; then
        MODEL_SETUP=1
    else
        echo "Invalid option: $1"
        echo "Use -i or --install_models to run model setup."
        exit 1
    fi
fi

# Get the directory of the currently executing script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the current working directory
current_dir="$(pwd)"

# Compare the directories
if [[ "$script_dir" == "$current_dir" ]]; then
    # If they are the same, move up one directory
    cd ..
fi

echo "**** openvino-ai-plugins-gimp Setup started ****"
# Install virtualenv if not already installed
python3 -m pip install virtualenv | grep -v "already satisfied"

# Create a virtual environment
python3 -m virtualenv gimpenv3

# Activate the virtual environment
source gimpenv3/bin/activate

# Upgrade pip and install required packages
pip3 install -r "$script_dir/requirements.txt" | grep -v "already satisfied"
pip3 install "$script_dir/."

# Run Python script to complete the installation
python3 -c "from gimpopenvino import complete_install; complete_install.setup_python_weights()"

echo "**** openvino-ai-plugins-gimp Setup Ended ****"
# Deactivate the virtual environment
deactivate

# Copy to GIMP plugin dir
echo "Installing plugin in $HOME/.config/GIMP/2.99/plug-ins"
for d in openvino_utils semseg_ov stable_diffusion_ov superresolution_ov; do
    rsync -a gimpenv3/lib/python*/site-packages/gimpopenvino/plugins/"$d" "$HOME/.config/GIMP/2.99/plug-ins/$d"
done
echo "*** openvino-ai-plugins-gimp Installed ***"

# If MODEL_SETUP was set, run the model setup
if [[ "$MODEL_SETUP" -eq 1 ]]; then
    echo "**** OpenVINO MODEL SETUP STARTED ****"
    python3 "$script_dir/model_setup.py"
fi

exit 0