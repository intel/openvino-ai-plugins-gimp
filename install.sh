#!/bin/bash
echo "**** openvino-ai-plugins-gimp Setup started ****"
python3 -m pip install virtualenv
python3 -m virtualenv gimpenv3
source gimpenv3/bin/activate
pip3 install "transformers>=4.21.1" "diffusers>=0.14.0" "tqdm==4.64.0" "openvino==2022.3.0" "huggingface_hub" "streamlit==1.12.0" "watchdog==2.1.9" "ftfy==6.1.1"
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pip3 install ${THIS_DIR}/.
echo "*** openvino-ai-plugins-gimp Installed ***"
python3 -c "import sys; sys.path.remove(''); import gimpopenvino; gimpopenvino.setup_python_weights()"
deactivate
echo **** openvino-ai-plugins-gimp Setup Ended ****
