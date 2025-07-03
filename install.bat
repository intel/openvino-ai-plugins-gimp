@echo off
set MODEL_SETUP=0
if "%1" NEQ "" (
    if /I "%1"=="-i" ( 
        set MODEL_SETUP=1 
    ) else (
        if /I "%1"=="--install_models" (
            set MODEL_SETUP=1 
        ) else (
            echo Invalid option: %1
            echo Use -i or --install_models to run model setup.
            exit /b
        )
    )
)

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

REM Create a virtual environment
call python -m venv gimpenv3

call "gimpenv3\Scripts\activate"

REM Install required packages
pip install wmi
pip install -r "%~dp0\requirements.txt" | find /V "already satisfied"
pip install "%~dp0\."

REM post install steps:
python -c "from gimpopenvino import install_utils; install_utils.complete_install(repo_weights_dir=r'%script_dir%\weights')"

echo **** openvino-ai-plugins-gimp Setup Ended ****
call deactivate
rem cls
echo.   
REM copy to gimp plugin dir
echo Installing plugin in "%appdata%\GIMP\3.0\plug-ins"
for /d %%d in (openvino_utils semseg_ov stable_diffusion_ov superresolution_ov ) do ( robocopy "gimpenv3\Lib\site-packages\gimpopenvino\plugins\%%d" "%appdata%\GIMP\3.0\plug-ins\%%d" /mir /NFL /NDL /NJH /NJS /nc /ns /np )

echo *** openvino-ai-plugins-gimp Installed ***
echo.    

if %MODEL_SETUP% EQU 1 (
    echo **** OpenVINO MODEL SETUP STARTED ****
    gimpenv3\Scripts\python.exe "%~dp0\model_setup.py"
)

REM return to the directory where we started.
cd %current_dir%
exit /b
