@echo off
set MODEL_SETUP=0
set DEFAULT_GIMP_VERSION=3.2

REM Parse arguments
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-i" (
    set MODEL_SETUP=1
    shift /1
    goto parse_args
)
if /I "%~1"=="--install_models" (
    set MODEL_SETUP=1
    shift /1
    goto parse_args
)
if /I "%~1"=="--gimp-version" (
    if "%~2"=="" (
        echo --gimp-version needs a value, e.g. --gimp-version 3.2
        exit /b 1
    )
    set "GIMP_VERSION=%~2"
    shift /1
    shift /1
    goto parse_args
)
echo Invalid option: %1
echo Usage: install.bat [-i ^| --install_models] [--gimp-version X.Y]
echo   -i, --install_models   run the interactive model setup after installing
echo   --gimp-version X.Y     GIMP version to install the plugins for (e.g. 3.0 or 3.2).
echo                          Detected automatically if not given.
exit /b 1
:args_done

REM Find the GIMP version (major.minor) the plugins are installed for
if not defined GIMP_VERSION (
    call :detect_gimp_version
    if defined GIMP_VERSION (
        call echo Detected GIMP %%GIMP_VERSION%%
    ) else (
        set "GIMP_VERSION=%DEFAULT_GIMP_VERSION%"
        echo Could not detect the installed GIMP version, assuming GIMP %DEFAULT_GIMP_VERSION%.
        echo Use --gimp-version X.Y to choose a different version.
    )
)
echo %GIMP_VERSION%| findstr /r "^3\.[0-9][0-9]*$" >nul || (
    echo Invalid GIMP version: %GIMP_VERSION% ^(expected major.minor, e.g. 3.2^)
    exit /b 1
)
set "PLUGIN_DIR=%APPDATA%\GIMP\%GIMP_VERSION%\plug-ins"

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
call "python" -m venv gimpenv3

call "gimpenv3\Scripts\activate"

REM Install required packages
pip install wmi
pip install "%~dp0\."

REM post install steps:
python -c "from gimpopenvino import install_utils; install_utils.complete_install(repo_weights_dir=r'%script_dir%\weights')"

echo **** openvino-ai-plugins-gimp Setup Ended ****
call deactivate
rem cls
echo.   
REM copy to gimp plugin dir
echo Installing plugin in "%PLUGIN_DIR%"
for /d %%d in (openvino_utils semseg_ov stable_diffusion_ov superresolution_ov fastsd_ov ) do ( robocopy "gimpenv3\Lib\site-packages\gimpopenvino\plugins\%%d" "%PLUGIN_DIR%\%%d" /mir /NFL /NDL /NJH /NJS /nc /ns /np )

echo *** openvino-ai-plugins-gimp Installed ***
echo.    

if %MODEL_SETUP% EQU 1 (
    echo **** OpenVINO MODEL SETUP STARTED ****
    gimpenv3\Scripts\python.exe "%~dp0\model_setup.py"
)

REM return to the directory where we started.
cd %current_dir%
exit /b

REM Look for gimp-<major.minor>.exe in the default GIMP 3 install folders
REM (all users and current user) and set GIMP_VERSION to the newest one found.
:detect_gimp_version
set "GIMP_VERSION="
for %%p in ("%ProgramFiles%\GIMP 3\bin" "%LOCALAPPDATA%\Programs\GIMP 3\bin") do (
    for %%f in ("%%~p\gimp-3.*.exe") do call :check_gimp_exe "%%~nf"
)
goto :eof

:check_gimp_exe
set "gimp_exe_name=%~1"
set "gimp_exe_ver=%gimp_exe_name:~5%"
echo %gimp_exe_ver%| findstr /r "^3\.[0-9][0-9]*$" >nul || goto :eof
if not defined GIMP_VERSION (
    set "GIMP_VERSION=%gimp_exe_ver%"
    goto :eof
)
REM keep the higher minor version
set /a "new_minor=%gimp_exe_ver:~2%", "cur_minor=%GIMP_VERSION:~2%"
if %new_minor% GTR %cur_minor% set "GIMP_VERSION=%gimp_exe_ver%"
goto :eof
