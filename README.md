# Whisper-transcribe local

Бесплатная транскрибация аудио и видео на Windows через [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Работает офлайн после первого скачивания модели.

## Возможности

- GUI на tkinter
- Модели от `tiny` до `large-v3`
- Ускорение на NVIDIA GPU (CUDA)
- Экспорт в `.txt` и `.srt`
- Русский, английский, украинский и автоопределение языка

## Требования

- Windows 10/11
- Python 3.12+
- FFmpeg
- NVIDIA GPU (опционально, для ускорения)

## Установка

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

Клонируйте репозиторий и запустите установку:

```powershell
git clone <url-репозитория>
cd whisper-transcribe
.\setup.bat
```

## Запуск

```powershell
.\run.bat
```

Или вручную:

```powershell
.\.venv\Scripts\pythonw.exe transcribe_app.py
```

## Использование

1. Нажмите «Выбрать…» и укажите аудио или видео файл.
2. Выберите модель (по умолчанию `large-v3`).
3. Нажмите «Транскрибировать».
4. Результат сохранится рядом с исходным файлом: `имя.txt` и `имя.srt`.

При первом запуске выбранная модель скачается в `%USERPROFILE%\.cache\huggingface`.

## Структура проекта

```
whisper-transcribe/
├── transcribe_app.py   # основное приложение
├── requirements.txt    # зависимости Python
├── setup.bat           # установка окружения
├── run.bat             # запуск GUI
└── README.md
```

## Лицензия

MIT
