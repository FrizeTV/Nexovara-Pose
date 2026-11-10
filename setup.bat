@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Nexovara Pose - setup

echo ============================================
echo Nexovara Pose - automatic setup
echo ============================================
echo.
set "PYEXE="
set "PYARG="

call :try_launcher 3.14
call :try_launcher 3.13
call :try_launcher 3.12
call :try_launcher 3.11
call :try_launcher 3.10
if defined PYEXE goto :run_setup

where python >nul 2>&1
if errorlevel 1 goto :try_install
python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>&1
if errorlevel 1 goto :try_install
set "PYEXE=python"
set "PYARG="
goto :run_setup

:try_install
where winget >nul 2>&1
if errorlevel 1 goto :nopython
echo Python 3.12 was not found. Installing it with winget...
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :nopython
call :find_local
if not defined PYEXE goto :nopython
goto :run_setup

:run_setup
echo Using: %PYEXE% %PYARG%
"%PYEXE%" %PYARG% -I tools\setup.py %*
if errorlevel 1 goto :setup_failed
echo Setup completed.
exit /b 0

:try_launcher
if defined PYEXE exit /b 0
py -%~1 -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYEXE=py"
set "PYARG=-%~1"
exit /b 0

:find_local
for %%V in (314 313 312 311 310) do call :try_local %%V
exit /b 0

:try_local
if defined PYEXE exit /b 0
if not exist "%LOCALAPPDATA%\Programs\Python\Python%~1\python.exe" exit /b 0
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python%~1\python.exe"
set "PYARG="
exit /b 0

:nopython
echo Python 3.10-3.14 was not found and could not be installed.
echo Install Python from https://www.python.org/downloads/ and run setup.bat again.
exit /b 1

:setup_failed
echo Setup failed. Read the error above.
exit /b 1
