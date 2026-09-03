@echo off
title Whisper
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Сначала запустите setup.bat
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0transcribe_app.py"
