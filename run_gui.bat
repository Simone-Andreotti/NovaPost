@echo off
setlocal
cd /d %~dp0

if not exist venv\Scripts\python.exe (
    echo venv not found. running setup...
    call setup.bat
)

if not exist venv\Scripts\python.exe (
    echo failed to prepare venv.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python web_panel.py

if errorlevel 1 (
    echo gui failed to start.
    pause
    exit /b 1
)
