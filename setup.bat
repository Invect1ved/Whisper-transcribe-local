@echo off
title Whisper — установка
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python не найден. Установите Python 3.12+ с python.org или через winget:
  echo   winget install Python.Python.3.12
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo FFmpeg не найден. Установите через winget:
  echo   winget install Gyan.FFmpeg
  echo.
)

echo Создаю виртуальное окружение...
python -m venv .venv
if errorlevel 1 (
  echo Не удалось создать .venv
  pause
  exit /b 1
)

echo Устанавливаю зависимости (это может занять несколько минут)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Ошибка установки
  pause
  exit /b 1
)

echo Пробую GPU-версию sherpa-onnx для лекторов...
".venv\Scripts\python.exe" -m pip install --upgrade "sherpa-onnx==1.13.7+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
if errorlevel 1 (
  echo GPU sherpa-onnx не установилась — лекторы будут на CPU.
)

echo.
echo Готово. Запускайте run.bat
pause
