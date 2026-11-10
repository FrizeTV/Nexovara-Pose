@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Nexovara Pose - OSC monitor
if not exist ".venv\Scripts\python.exe" goto :missing
.venv\Scripts\python.exe -I tools\osc_monitor.py %*
exit /b %errorlevel%
:missing
echo Run setup.bat first.
exit /b 1
