@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Nexovara Pose - application build

echo ============================================
echo Nexovara Pose - application build
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" goto :novenvironment

call :progress 10 "checking build environment"
.venv\Scripts\python.exe -m pip install --disable-pip-version-check --only-binary=:all: "pyinstaller>=6,<7"
if errorlevel 1 goto :fail

call :progress 35 "building executable"
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name "Nexovara Pose" --collect-all mediapipe --collect-all cv2_enumerate_cameras --add-data "models;models" main.py
if errorlevel 1 goto :fail

call :progress 100 "completed"
echo.
echo Build completed:
echo dist\Nexovara Pose\Nexovara Pose.exe
exit /b 0

:novenvironment
echo Python environment was not found.
echo Run setup.bat first, then run build_exe.bat again.
exit /b 1

:fail
echo.
echo Application build failed.
exit /b 1

:progress
set "PCT=%~1"
set "PLABEL=%~2"
set "BAR="
if "%PCT%"=="10" set "BAR=###---------------------------"
if "%PCT%"=="35" set "BAR=##########--------------------"
if "%PCT%"=="100" set "BAR=##############################"
echo [%BAR%] %PCT%%% %PLABEL%
exit /b 0
