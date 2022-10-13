:<<BATCH
    @echo off
    echo **** GIMP-OV Setup started **** 
    python -m pip install virtualenv
	python -m virtualenv gimpenv3
	gimpenv3\Scripts\python.exe -m pip install  transformers==4.16.2 diffusers==0.2.4 tqdm==4.64.0 openvino==2022.1.0 huggingface_hub==0.9.0 streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1
    gimpenv3\Scripts\python.exe -m pip install GIMP-OV\.
	echo *** GIMP-OV Installed ***
    gimpenv3\Scripts\python.exe -c "import gimpov; gimpov.setup_python_weights()"
	echo **** GIMP-OV Setup Ended ****
    exit /b
BATCH


