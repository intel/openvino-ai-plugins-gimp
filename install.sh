#!/bin/sh
echo "**** openvino-ai-plugins-gimp Setup started ****"
python -m pip install virtualenv
python -m virtualenv gimpenv3
source gimpenv3/bin/activate
pip install transformers==4.23.0 diffusers==0.2.4 tqdm==4.64.0 openvino==2022.3.0 huggingface_hub streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1
pip install openvino-ai-plugins-gimp
echo "*** openvino-ai-plugins-gimp Installed ***"
python -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
deactivate
echo **** openvino-ai-plugins-gimp Setup Ended ****
