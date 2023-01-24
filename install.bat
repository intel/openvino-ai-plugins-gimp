:<<BATCH
    @echo off
    echo **** GIMP-OpenVINO Setup started **** 
    python -m pip install virtualenv
	python -m virtualenv gimpenv3
	gimpenv3\Scripts\python.exe -m pip install  transformers==4.23.0 diffusers==0.2.4 tqdm==4.64.0 openvino==2022.3.0 huggingface_hub streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1
    gimpenv3\Scripts\python.exe -m pip install GIMP-OpenVINO\.
	echo *** GIMP-OpenVINO Installed ***
    gimpenv3\Scripts\python.exe -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
	echo **** GIMP-OpenVINO Setup Ended ****
    exit /b
BATCH


