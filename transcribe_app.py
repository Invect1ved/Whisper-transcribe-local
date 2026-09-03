#!/usr/bin/env python3
"""Локальная транскрибация через faster-whisper (бесплатно, без интернета после скачивания модели)."""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tkinter as tk

AUDIO_TYPES = [
    ("Аудио и видео", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma *.mp4 *.mkv *.webm *.avi *.mov *.opus"),
    ("Все файлы", "*.*"),
]

MODELS = [
    ("tiny — быстро, черновик", "tiny"),
    ("base — быстрее среднего", "base"),
    ("small — баланс скорости", "small"),
    ("medium — точнее", "medium"),
    ("large-v3 — максимальное качество (рекомендуется)", "large-v3"),
    ("distil-large-v3 — почти large, быстрее", "distil-large-v3"),
]

LANGUAGES = [
    ("Автоопределение", None),
    ("Русский", "ru"),
    ("English", "en"),
    ("Українська", "uk"),
]


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_cuda_dlls() -> None:
    """Windows не подхватывает cuBLAS/cuDNN из pip без явного add_dll_directory."""
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return
    dirs: list[str] = []
    for bindir in nvidia.glob("*/bin"):
        if bindir.is_dir():
            dirs.append(str(bindir))
    for bindir in nvidia.glob("*/lib/x64"):
        if bindir.is_dir():
            dirs.append(str(bindir))
    if not dirs:
        return
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    if hasattr(os, "add_dll_directory"):
        for bindir in dirs:
            os.add_dll_directory(bindir)


ensure_cuda_dlls()


def ffmpeg_candidates() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("FFMPEG_BINARY")
    if env:
        paths.append(Path(env))
    paths.append(app_dir() / "ffmpeg.exe")
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.exists():
        paths.extend(winget.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    return paths


def ensure_ffmpeg() -> None:
    from shutil import which

    if which("ffmpeg"):
        return
    for candidate in ffmpeg_candidates():
        if candidate.is_file():
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
            return


def detect_device() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Whisper — транскрибация")
        self.geometry("760x560")
        self.minsize(680, 500)
        self.configure(bg="#1e1e1e")

        self.file_var = tk.StringVar()
        self.model_var = tk.StringVar(value=MODELS[4][0])
        self.lang_var = tk.StringVar(value=LANGUAGES[1][0])
        self.txt_var = tk.BooleanVar(value=True)
        self.srt_var = tk.BooleanVar(value=True)
        self.busy = False
        self.device, self.compute = detect_device()

        self._build_style()
        self._build()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg, fg, accent = "#1e1e1e", "#f3f3f3", "#3d7ea6"
        style.configure(".", background=bg, foreground=fg, fieldbackground="#2b2b2b")
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#b0b0b0", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("TCheckbutton", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground="#2b2b2b", background="#2b2b2b", foreground=fg)
        style.configure("TEntry", fieldbackground="#2b2b2b", foreground=fg)
        style.map("TCombobox", fieldbackground=[("readonly", "#2b2b2b")])
        style.configure("Accent.TButton", background=accent, foreground="white")
        style.map("Accent.TButton", background=[("active", "#4b93bd")])

    def _build(self) -> None:
        pad = ttk.Frame(self, padding=20)
        pad.pack(fill="both", expand=True)

        ttk.Label(pad, text="Бесплатная транскрибация на ПК", style="Title.TLabel").pack(anchor="w")
        device_text = (
            f"Ускорение: NVIDIA GPU ({self.compute})"
            if self.device == "cuda"
            else "Ускорение: процессор (GPU не найден)"
        )
        ttk.Label(pad, text=device_text, style="Hint.TLabel").pack(anchor="w", pady=(4, 16))

        ttk.Label(pad, text="Файл аудио или видео").pack(anchor="w")
        row = ttk.Frame(pad)
        row.pack(fill="x", pady=(4, 12))
        ttk.Entry(row, textvariable=self.file_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать…", command=self.pick_file).pack(side="left", padx=(8, 0))

        opts = ttk.Frame(pad)
        opts.pack(fill="x", pady=(0, 12))

        left = ttk.Frame(opts)
        left.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ttk.Label(left, text="Модель").pack(anchor="w")
        ttk.Combobox(
            left,
            textvariable=self.model_var,
            values=[m[0] for m in MODELS],
            state="readonly",
        ).pack(fill="x", pady=(4, 0))

        right = ttk.Frame(opts)
        right.pack(side="left", fill="x", expand=True)
        ttk.Label(right, text="Язык").pack(anchor="w")
        ttk.Combobox(
            right,
            textvariable=self.lang_var,
            values=[lang[0] for lang in LANGUAGES],
            state="readonly",
        ).pack(fill="x", pady=(4, 0))

        checks = ttk.Frame(pad)
        checks.pack(fill="x", pady=(0, 12))
        ttk.Checkbutton(checks, text="Сохранить .txt", variable=self.txt_var).pack(side="left")
        ttk.Checkbutton(checks, text="Сохранить субтитры .srt", variable=self.srt_var).pack(side="left", padx=(16, 0))

        self.run_btn = ttk.Button(pad, text="Транскрибировать", style="Accent.TButton", command=self.start)
        self.run_btn.pack(fill="x", pady=(4, 12))

        ttk.Label(pad, text="Журнал").pack(anchor="w")
        self.log = tk.Text(
            pad,
            height=14,
            bg="#141414",
            fg="#dcdcdc",
            insertbackground="white",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
        )
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.log_line("Готово. Выберите файл и нажмите «Транскрибировать».")
        self.log_line("Первый запуск модели скачает её один раз (интернет нужен только для этого).")

    def pick_file(self) -> None:
        path = filedialog.askopenfilename(title="Выберите запись", filetypes=AUDIO_TYPES)
        if path:
            self.file_var.set(path)

    def log_line(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.update_idletasks()

    def start(self) -> None:
        if self.busy:
            return
        path = Path(self.file_var.get().strip().strip('"'))
        if not path.is_file():
            messagebox.showerror("Нет файла", "Сначала выберите аудио или видео файл.")
            return
        if not self.txt_var.get() and not self.srt_var.get():
            messagebox.showerror("Формат", "Включите хотя бы один формат: TXT или SRT.")
            return
        self.busy = True
        self.run_btn.configure(state="disabled", text="Идёт транскрибация…")
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path: Path) -> None:
        try:
            ensure_ffmpeg()
            model_id = dict(MODELS)[self.model_var.get()]
            language = dict(LANGUAGES)[self.lang_var.get()]
            self.after(0, self.log_line, f"Файл: {path.name}")
            self.after(0, self.log_line, f"Модель: {model_id}, устройство: {self.device}")
            self.after(0, self.log_line, "Загрузка модели…")

            from faster_whisper import WhisperModel

            model = WhisperModel(model_id, device=self.device, compute_type=self.compute)
            self.after(0, self.log_line, "Модель готова, распознаю речь…")

            segments, info = model.transcribe(
                str(path),
                language=language,
                vad_filter=True,
                beam_size=5,
            )
            detected = info.language or "неизвестно"
            self.after(0, self.log_line, f"Язык: {detected}")

            rows = []
            text_parts = []
            for seg in segments:
                rows.append((seg.start, seg.end, seg.text.strip()))
                text_parts.append(seg.text.strip())
                preview = seg.text.strip()
                if preview:
                    self.after(0, self.log_line, f"[{fmt_ts(seg.start)} → {fmt_ts(seg.end)}] {preview}")

            full_text = "\n".join(p for p in text_parts if p).strip() + "\n"
            saved = []
            if self.txt_var.get():
                txt_path = path.with_suffix(".txt")
                txt_path.write_text(full_text, encoding="utf-8")
                saved.append(txt_path)
            if self.srt_var.get():
                srt_path = path.with_suffix(".srt")
                srt_path.write_text(to_srt(rows), encoding="utf-8")
                saved.append(srt_path)

            names = ", ".join(p.name for p in saved) or "ничего"
            self.after(0, self.log_line, f"Готово. Сохранено рядом с файлом: {names}")
            self.after(
                0,
                lambda: messagebox.showinfo("Готово", f"Транскрипт сохранён:\n" + "\n".join(str(p) for p in saved)),
            )
        except Exception:
            err = traceback.format_exc()
            self.after(0, self.log_line, "Ошибка:\n" + err)
            self.after(0, lambda: messagebox.showerror("Ошибка", "Не получилось транскрибировать. Смотрите журнал."))
        finally:
            self.after(0, self._idle)

    def _idle(self) -> None:
        self.busy = False
        self.run_btn.configure(state="normal", text="Транскрибировать")


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def to_srt(rows: list[tuple[float, float, str]]) -> str:
    blocks = []
    for i, (start, end, text) in enumerate(rows, 1):
        if not text:
            continue
        blocks.append(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n")
    return "\n".join(blocks)


def main() -> None:
    ensure_ffmpeg()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
