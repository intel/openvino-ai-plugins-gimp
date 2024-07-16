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
pip install pywin32
pip install -r "%~dp0\plugin-requirements.txt" | find /V "already satisfied"
pip install "%~dp0\."

python -c "import gimpopenvino; gimpopenvino.setup_python_weights()"
echo **** openvino-ai-plugins-gimp Setup Ended ****
call deactivate
cls
echo.   
REM copy to gimp plugin dir
echo Installing plugin in %appdata%\GIMP\2.99\plug-ins
xcopy %~dp0\gimpopenvino\plugins\* %appdata%\GIMP\2.99\plug-ins /s /e /q /y
echo *** openvino-ai-plugins-gimp Installed ***
echo.    
REM Prompt the user to continue setting up models
set /p model_setup="Do you want to continue setting up the models for all the plugins now? Enter Y/N:  "
echo your choice: %model_setup%

if /i "%model_setup%"=="Y" (
    set "continue=y"
) else if /i "%model_setup%"=="y" (
    set "continue=y"
) else (
    set "continue=n"
)

if "%continue%"=="y" (
    echo **** OpenVINO MODEL SETUP STARTED ****
    gimpenv3\Scripts\python.exe "%~dp0\model_setup.py"
) else (
    echo Model setup skipped. Please make sure you have all the required models set up.
)

exit /b
