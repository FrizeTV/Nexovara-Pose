@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto :missing
start "Nexovara Pose" .venv\Scripts\pythonw.exe -I main.py
exit /b 0
:missing
echo Run setup.bat first.
exit /b 1
