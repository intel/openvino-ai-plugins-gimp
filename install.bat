:<<BATCH
    @echo off
    echo **** openvino-ai-plugins-gimp Setup started **** 
    python -m pip install virtualenv | find /V "already satisfied"
    python -m virtualenv gimpenv3
    gimpenv3\Scripts\python.exe -m pip install safetensors==0.3.2 accelerate transformers==4.31.0 diffusers>=0.23.0 controlnet-aux>=0.0.6 tqdm==4.64.0 openvino==2023.1.0 huggingface_hub streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1 | find /V "already satisfied"
    gimpenv3\Scripts\python.exe -m pip install openvino-ai-plugins-gimp\.
	echo *** openvino-ai-plugins-gimp Installed ***
    gimpenv3\Scripts\python.exe -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
	echo **** openvino-ai-plugins-gimp Setup Ended ****



	echo -----------------------------------------------------------------------------------------------
	echo -----------------------------------------------------------------------------------------------


	set /p model_setup= "Do you want to continue setting up the models for all the plugin now? Enter Y/N:  "
	echo your choice %model_setup%
	if %model_setup%==Y (
		set "continue=y"
	) else if %model_setup%==y (
		set "continue=y"
	) else ( set "continue=n"
		)
		

	
	if %continue%==y (
		echo **** OpenVINO MODEL SETUP STARTED ****
		gimpenv3\Scripts\python.exe openvino-ai-plugins-gimp\model_setup.py

		) else ( echo Model setup skipped. Please make sure you have all the required models setup.
		)
		
		
    exit /b
BATCH


