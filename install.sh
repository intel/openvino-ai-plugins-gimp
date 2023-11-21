#!/bin/bash
echo "**** openvino-ai-plugins-gimp Setup started ****"
python3 -m pip install virtualenv
python3 -m virtualenv gimpenv3
source gimpenv3/bin/activate
pip3 install -r plugin-requirements.txt
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pip3 install ${THIS_DIR}/.
echo "*** openvino-ai-plugins-gimp Installed ***"
python3 -c "import sys; sys.path.remove(''); import gimpopenvino; gimpopenvino.setup_python_weights()"
deactivate
echo **** openvino-ai-plugins-gimp Setup Ended ****
