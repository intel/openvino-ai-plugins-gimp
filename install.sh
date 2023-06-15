#!/bin/bash
echo "**** openvino-ai-plugins-gimp Setup started ****"
python3 -m pip install virtualenv
python3 -m virtualenv gimpenv3
source gimpenv3/bin/activate
pip3 install "transformers>=4.21.1" "diffusers>=0.14.0" "tqdm==4.64.0" "openvino==2022.3.0" "huggingface_hub" "streamlit==1.12.0" "watchdog==2.1.9" "ftfy==6.1.1" | grep -v "already satisfied"
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pip3 install ${THIS_DIR}/.
echo "*** openvino-ai-plugins-gimp Installed ***"
python3 -c "import sys; sys.path.remove(''); import gimpopenvino; gimpopenvino.setup_python_weights()"
deactivate
echo **** openvino-ai-plugins-gimp Setup Ended ****

echo "-----------------------------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------"


read -p "Do you want to continue setting up the models for all the plugin now? Enter Y/N:  " model_setup
model_setup=${model_setup:=n}

echo "your choice " $model_setup

if [ $model_setup == Y ] || [ $model_setup == y ]
then
	echo **** OpenVINO MODEL SETUP STARTED ****
	source gimpenv3/bin/activate
	python3 ${THIS_DIR}/model_setup.py
	deactivate
	echo "-----------------------------------------------------------------------------------------------"
	echo "**** OPENVINO STABLE DIFFUSION 1.5 MODELS SETUP ****"
	echo "Checking model installation environment"
	
	python3 -m virtualenv model_conv
	source model_conv/bin/activate
	pip3 install --upgrade "pip" "wheel" "setuptools" | grep -v "already satisfied"
	pip3 install -r "${THIS_DIR}/model-requirements.txt" | grep -v "already satisfied"
	
	python3 ${THIS_DIR}/choose_sd_model.py
	echo "**** OPENVINO STABLE DIFFUSION 1.5 MODELS COMPLETE ****"
	deactivate
	
else
	echo "Model setup skipped. Please make sure you have all the required models setup."
	
fi

