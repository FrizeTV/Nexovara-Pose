@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Nexovara Pose - self check
if not exist ".venv\Scripts\python.exe" goto :missing
.venv\Scripts\python.exe -I tools\selfcheck.py %*
exit /b %errorlevel%
:missing
echo Run setup.bat first.
exit /b 1
