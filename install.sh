#!/bin/bash
echo "**** openvino-ai-plugins-gimp Setup started ****"
python3 -m pip install virtualenv
python3 -m virtualenv gimpenv3
source gimpenv3/bin/activate
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pip3 install -r ${THIS_DIR}/plugin-requirements.txt
pip3 install ${THIS_DIR}/.
echo "*** openvino-ai-plugins-gimp Installed ***"
python3 -c "import sys; sys.path.remove(''); import gimpopenvino; gimpopenvino.setup_python_weights()"

continue=0
echo "Do you want to continue setting up the models for all the plugins now? Enter Y/N: "
# Wait for the user to press a key
read -s -n 1 key
 
# Check which key was pressed
case $key in
    y|Y)
    echo "You pressed 'y'. Continuing..."
    continue=1
    ;;
    n|N)
    echo "You pressed 'n'. Exiting..."
    ;;
    *)
    echo "Invalid input. Please press 'y' or 'n'."
    ;;
esac

[[ ${continue} == 1 ]] && python3 ${THIS_DIR}/model_setup.py

deactivate
echo "**** openvino-ai-plugins-gimp Setup Ended ****"
