@echo off
REM Get the directory of the currently executing script
set script_dir=%~dp0

REM Remove the trailing backslash
set script_dir=%script_dir:~0,-1%

REM Get the current working directory
set current_dir=%cd%

REM Compare the directories
if /i "%script_dir%"=="%current_dir%" (
    REM If they are the same, move up one directory
    cd ..
) else (
    echo. 
)

echo **** openvino-ai-plugins-gimp Setup started **** 

REM Install virtualenv if not already installed
python -m pip install virtualenv | find /V "already satisfied"

REM Create a virtual environment
python -m virtualenv gimpenv3
call "gimpenv3\Scripts\activate"

REM Upgrade pip and install required packages
python -m pip install --upgrade pip 
pip install -r "%~dp0\requirements.txt" | find /V "already satisfied"
pip install "%~dp0\."

python -c "from gimpopenvino import complete_install; complete_install.setup_python_weights()"
echo **** openvino-ai-plugins-gimp Setup Ended ****
call deactivate
rem cls
echo.   
REM copy to gimp plugin dir
echo Installing plugin in %appdata%\GIMP\2.99\plug-ins
for /d %%d in (openvino_utils semseg_ov stable_diffusion_ov superresolution_ov ) do ( robocopy gimpenv3\Lib\site-packages\gimpopenvino\plugins\%%d %appdata%\GIMP\2.99\plug-ins\%%d /mir /NFL /NDL /NJH /NJS /nc /ns /np )

echo *** openvino-ai-plugins-gimp Installed ***
echo.    
REM Prompt the user to continue setting up models

echo **** OpenVINO MODEL SETUP STARTED ****
gimpenv3\Scripts\python.exe "%~dp0\model_setup.py"

exit /b

