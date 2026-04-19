@echo off
setlocal
cd /d %~dp0

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist bin mkdir bin

if not exist bin\ffmpeg.exe (
    if exist ffmpeg-build\bin\ffmpeg.exe (
        copy /Y ffmpeg-build\bin\ffmpeg.exe bin\ffmpeg.exe >nul
    )
)

if not exist bin\ffprobe.exe (
    if exist ffmpeg-build\bin\ffprobe.exe (
        copy /Y ffmpeg-build\bin\ffprobe.exe bin\ffprobe.exe >nul
    )
)

if not exist bin\ffmpeg.exe (
    echo Missing bin\ffmpeg.exe. Place ffmpeg.exe in bin.
)

if not exist bin\ffprobe.exe (
    echo Missing bin\ffprobe.exe. Place ffprobe.exe in bin.
)

echo Setup complete.
