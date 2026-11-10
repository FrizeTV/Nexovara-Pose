@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Nexovara Pose - SteamVR driver build

echo ============================================
echo Nexovara Pose - SteamVR driver build
echo ============================================
echo.

call :progress 0 "checking tools"
set "CMAKE_EXE="
where cmake >nul 2>&1
if not errorlevel 1 set "CMAKE_EXE=cmake"

if defined CMAKE_EXE goto :cmake_ready
if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
if defined CMAKE_EXE goto :cmake_ready
if exist "%LOCALAPPDATA%\Programs\CMake\bin\cmake.exe" set "CMAKE_EXE=%LOCALAPPDATA%\Programs\CMake\bin\cmake.exe"
if defined CMAKE_EXE goto :cmake_ready
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if defined CMAKE_EXE goto :cmake_ready
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if defined CMAKE_EXE goto :cmake_ready

set "CMAKE_ROOT=%CD%\tools\cmake"
if exist "%CMAKE_ROOT%\bin\cmake.exe" set "CMAKE_EXE=%CMAKE_ROOT%\bin\cmake.exe"
if defined CMAKE_EXE goto :cmake_ready

call :progress 10 "downloading portable CMake"
echo CMake was not found. Downloading the portable copy into:
echo %CMAKE_ROOT%
if not exist "%CD%\tools" mkdir "%CD%\tools"
if not exist "%CMAKE_ROOT%" mkdir "%CMAKE_ROOT%"
set "CMAKE_ZIP=%CD%\tools\cmake-portable.zip"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-windows-x86_64.zip' -OutFile '%CMAKE_ZIP%'; Expand-Archive -Path '%CMAKE_ZIP%' -DestinationPath '%CD%\tools\cmake-extract' -Force; $d=Get-ChildItem '%CD%\tools\cmake-extract' -Directory | Select-Object -First 1; Copy-Item ($d.FullName + '\*') '%CMAKE_ROOT%' -Recurse -Force; Remove-Item '%CD%\tools\cmake-extract' -Recurse -Force; Remove-Item '%CMAKE_ZIP%' -Force"
if errorlevel 1 goto :nocmake
if not exist "%CMAKE_ROOT%\bin\cmake.exe" goto :nocmake
set "CMAKE_EXE=%CMAKE_ROOT%\bin\cmake.exe"

:cmake_ready
if not exist driver\CMakeLists.txt goto :nosource

call :progress 25 "configuring CMake"
echo Starting CMake. Visual Studio Build Tools and Windows SDK are required.
echo.

"%CMAKE_EXE%" -S driver -B driver\build -A x64 -DNEXOVARAPOSE_BUILD_TESTS=ON -DNEXOVARAPOSE_DIST="%CD%\driver\dist"
if errorlevel 1 goto :fail

call :progress 55 "compiling driver"
"%CMAKE_EXE%" --build driver\build --config Release --parallel
if errorlevel 1 goto :fail

call :progress 85 "running driver tests"
if exist driver\build\Release\nexovarapose_protocol_test.exe driver\build\Release\nexovarapose_protocol_test.exe
if errorlevel 1 goto :fail

if exist driver\build\Release\nexovarapose_driver_host_test.exe driver\build\Release\nexovarapose_driver_host_test.exe driver\dist\nexovarapose\bin\win64\driver_nexovarapose.dll
if errorlevel 1 goto :fail

call :progress 100 "completed"
echo.
echo Build completed: %CD%\driver\dist\nexovarapose
exit /b 0

:nocmake
echo CMake could not be downloaded into the project folder.
echo Check your internet connection and run this file again.
exit /b 1

:nosource
echo driver\CMakeLists.txt was not found.
echo Run this file from the Nexovara Pose project root.
exit /b 1

:fail
echo.
echo Driver build or test failed.
echo Check that Visual Studio Build Tools has C++ Desktop Development and Windows SDK.
exit /b 1

:progress
set "PCT=%~1"
set "PLABEL=%~2"
set "BAR="
if "%PCT%"=="0" set "BAR=------------------------------"
if "%PCT%"=="10" set "BAR=###---------------------------"
if "%PCT%"=="25" set "BAR=########----------------------"
if "%PCT%"=="55" set "BAR=################----------------"
if "%PCT%"=="85" set "BAR=#########################-----"
if "%PCT%"=="100" set "BAR=##############################"
echo [%BAR%] %PCT%%% %PLABEL%
exit /b 0
