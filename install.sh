#!/bin/bash

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
source gimpenv3/bin/activate

# Upgrade pip and install required packages
python3 -m pip install --upgrade pip
pip install -r "$script_dir/requirements.txt" | grep -v "already satisfied"
pip install "$script_dir/"

python3 -c "from gimpopenvino import complete_install; complete_install.setup_python_weights()"
echo "**** openvino-ai-plugins-gimp Setup Ended ****"
deactivate

# Copy to GIMP plugin dir
echo "Installing plugin in $HOME/.config/GIMP/2.99/plug-ins"
for d in openvino_utils semseg_ov stable_diffusion_ov superresolution_ov; do
    rsync -a gimpenv3/Lib/site-packages/gimpopenvino/plugins/$d "$APPDATA/GIMP/2.99/plug-ins/$d"
done

echo "*** openvino-ai-plugins-gimp Installed ***"

# Prompt the user to continue setting up models
echo "**** OpenVINO MODEL SETUP STARTED ****"
gimpenv3/bin/python3 "$script_dir/model_setup.py"



exit 0