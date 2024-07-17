@echo off
setlocal

:: Get the script directory
set SCRIPT_DIR=%~dp0

:: Request administrative privileges for setting the PATH
net session >nul 2>nul
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb runAs"
    goto end
)

:: Check if Git is already installed
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo Git is already installed.
) else (
    :: Download Git installer
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe' -OutFile '%TEMP%\GitInstaller.exe'"

    :: Install Git
    echo Installing Git...
    %TEMP%\GitInstaller.exe /SILENT /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"

    :: Delete the installer
    del %TEMP%\GitInstaller.exe
)

:install_venv
echo Installing virtual environment and dependencies...

:: Enable delayed variable expansion
setlocal enabledelayedexpansion

cd /d "%SCRIPT_DIR%"

:: Create a virtual environment
echo Creating virtual environment...
python -m venv "venv"
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    goto end
)

:: Install project dependencies
echo Installing project dependencies...
"venv\Scripts\python" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    goto end
)

echo Virtual environment setup complete.

echo Press any key to exit...
pause >nul
goto :EOF

:end
echo Script ended.
