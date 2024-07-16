
#!/bin/bash
# Get the directory of the currently executing script
script_dir=$(dirname "$(realpath "$0")")

# Get the current working directory
current_dir=$(pwd)

# Compare the directories
if [ "$script_dir" == "$current_dir" ]; then
    # If they are the same, move up one directory
    cd ..
    echo "Moved up one directory to $(pwd)"
else
    echo "Script is not running from its own directory"
fi

echo "**** openvino-ai-plugins-gimp Setup started ****"

# Install virtualenv if not already installed
python3 -m pip install virtualenv | grep -v "already satisfied"

# Create a virtual environment
python3 -m virtualenv gimpenv3
echo "-----activating python venv------------------------------------------------------------------"
source gimpenv3/bin/activate

# Upgrade pip and install required packages
python3 -m pip install --upgrade pip
pip install -r "$script_dir/plugin-requirements.txt" | grep -v "already satisfied"
pip install "$script_dir/."

echo "*** openvino-ai-plugins-gimp Installed ***"
python3 -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
echo "**** openvino-ai-plugins-gimp Setup Ended ****"
echo "-----deactivating python venv------------------------------------------------------------------"
deactivate
echo "-----------------------------------------------------------------------------------------------"

# Prompt the user to continue setting up models
read -p "Do you want to continue setting up the models for all the plugins now? Enter Y/N: " model_setup
echo "Your choice: $model_setup"

if [[ "$model_setup" =~ ^[Yy]$ ]]; then
    continue="y"
else
    continue="n"
fi

if [ "$continue" == "y" ]; then
    echo "**** OpenVINO MODEL SETUP STARTED ****"
    gimpenv3/bin/python "$script_dir/model_setup.py"
else
    echo "Model setup skipped. Please make sure you have all the required models set up."
fi

echo "**** openvino-ai-plugins-gimp Setup Ended****"

exit 0