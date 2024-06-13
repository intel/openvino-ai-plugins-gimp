:<<BATCH
    @echo off
    echo **** openvino-ai-plugins-gimp Setup started **** 
    python -m pip install virtualenv | find /V "already satisfied"

    python -m virtualenv gimpenv3
    echo -----activating python venv------------------------------------------------------------------
    call "gimpenv3\Scripts\activate"
	pip install pywin32
    pip install -r %~dp0\plugin-requirements.txt | find /V "already satisfied"
    pip install %~dp0\.

    echo *** openvino-ai-plugins-gimp Installed ***
    python -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
    echo **** openvino-ai-plugins-gimp Setup Ended ****
    echo -----deactivating python venv------------------------------------------------------------------
    call deactivate
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
	gimpenv3\Scripts\python.exe %~dp0\model_setup.py
    ) else ( echo Model setup skipped. Please make sure you have all the required models setup.
    )
		
    exit /b
BATCH


