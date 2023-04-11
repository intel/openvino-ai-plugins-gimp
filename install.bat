:<<BATCH
    @echo off
    echo **** openvino-ai-plugins-gimp Setup started **** 
    python -m pip install virtualenv
    python -m virtualenv gimpenv3
    gimpenv3\Scripts\python.exe -m pip install  transformers==4.23.0 diffusers==0.2.4 tqdm==4.64.0 openvino==2022.3.0 huggingface_hub streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1 gdown
    gimpenv3\Scripts\python.exe -m pip install .
     echo *** openvino-ai-plugins-gimp Installed ***
    gimpenv3\Scripts\python.exe -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
    echo **** openvino-ai-plugins-gimp Setup Ended ****  
    echo **** copy weights folder to user profile %HOMEPATH%\openvino-ai-plugins-gimp\weights\ ****
    Xcopy /E /I .\weights %HOMEPATH%\openvino-ai-plugins-gimp\weights\
    exit /b
BATCH


