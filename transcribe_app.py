#!/usr/bin/env python3
"""Локальная транскрибация через faster-whisper с определением лекторов."""

from __future__ import annotations

import os
import sys


def ensure_stdio() -> None:
    """pythonw.exe оставляет stdout/stderr = None, из-за этого падает tqdm."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


ensure_stdio()

import faulthandler
import json
import threading
import time
import traceback
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

AUDIO_TYPES = [
    ("Аудио и видео", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma *.mp4 *.mkv *.webm *.avi *.mov *.opus"),
    ("Все файлы", "*.*"),
]

MODELS = [
    ("tiny — быстро, черновик", "tiny"),
    ("base — быстрее среднего", "base"),
    ("small — баланс скорости", "small"),
    ("medium — точнее", "medium"),
    ("large-v3 — максимальное качество", "large-v3"),
    ("distil-large-v3 — почти large, быстрее", "distil-large-v3"),
]

LANGUAGES = [
    ("Автоопределение", None),
    ("Русский", "ru"),
    ("English", "en"),
    ("Українська", "uk"),
]

SPEAKER_COUNTS = [
    ("Авто", None),
    ("1 лектор", 1),
    ("2 лектора", 2),
    ("3 лектора", 3),
    ("4 лектора", 4),
    ("5 лекторов", 5),
]

SPEAKER_COLORS = ["#7dd3fc", "#f9a8d4", "#86efac", "#fcd34d", "#c4b5fd", "#fdba74", "#fca5a5", "#67e8f9"]

BG = "#0f1419"
SIDE = "#151b22"
CARD = "#1c242e"
ACCENT = "#3b82f6"
TEXT = "#e8eef5"
MUTED = "#8b9bb0"


@dataclass(frozen=True)
class Job:
    path: Path
    model_id: str
    language: str | None
    num_speakers: int | None
    use_diarize: bool
    save_txt: bool
    save_json: bool
    save_srt: bool


def app_data_dir() -> Path:
    path = Path.home() / "AppData" / "Local" / "Whisper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def enable_crash_log() -> None:
    try:
        handle = open(app_data_dir() / "crash.log", "w", encoding="utf-8")
        faulthandler.enable(handle, all_threads=True)
    except Exception:
        faulthandler.enable()


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return Path(__file__).resolve().parent


def cuda_bin_dirs() -> list[Path]:
    roots = [
        bundle_dir(),
        app_dir(),
        app_dir() / "_internal",
        Path(sys.prefix) / "Lib" / "site-packages",
    ]
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidates = [root]
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            candidates.append(nvidia)
        for base in candidates:
            for pattern in ("*/bin", "nvidia/*/bin", "*/lib/x64"):
                for bindir in base.glob(pattern):
                    key = str(bindir.resolve()) if bindir.exists() else str(bindir)
                    if bindir.is_dir() and key not in seen:
                        seen.add(key)
                        dirs.append(bindir)
        if (root / "cublas64_12.dll").is_file():
            key = str(root.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(root)
    return dirs


def ensure_cuda_dlls() -> None:
    dirs = [str(path) for path in cuda_bin_dirs()]
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
    paths.append(bundle_dir() / "ffmpeg.exe")
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


def fmt_ts(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def to_txt(rows: list[tuple[float, float, str, str]]) -> str:
    blocks = []
    for start, end, speaker, text in rows:
        if not text:
            continue
        who = f"{speaker} · " if speaker else ""
        blocks.append(f"[{fmt_ts(start)} → {fmt_ts(end)}] {who}{text}")
    return "\n".join(blocks) + ("\n" if blocks else "")


def to_srt(rows: list[tuple[float, float, str, str]]) -> str:
    blocks = []
    index = 1
    for start, end, speaker, text in rows:
        if not text:
            continue
        line = f"{speaker}: {text}" if speaker else text
        blocks.append(f"{index}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{line}\n")
        index += 1
    return "\n".join(blocks)


def to_json(rows: list[tuple[float, float, str, str]], source: Path, language: str) -> str:
    payload = {
        "source": source.name,
        "language": language,
        "segments": [
            {
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "speaker": speaker or None,
                "text": text,
            }
            for start, end, speaker, text in rows
            if text
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Whisper Transcribe Local")
        self.geometry("1120x720")
        self.minsize(960, 620)
        self.configure(fg_color=BG)

        self.file_var = ctk.StringVar()
        self.model_var = ctk.StringVar(value=MODELS[4][0])
        self.lang_var = ctk.StringVar(value=LANGUAGES[1][0])
        self.speakers_var = ctk.StringVar(value=SPEAKER_COUNTS[0][0])
        self.txt_var = ctk.BooleanVar(value=True)
        self.json_var = ctk.BooleanVar(value=False)
        self.srt_var = ctk.BooleanVar(value=True)
        self.diarize_var = ctk.BooleanVar(value=True)
        self.busy = False
        self.device, self.compute = detect_device()
        self._last_model_ui = 0.0
        self._last_job_ui = 0.0
        self._pending_lines: list[tuple[str, str, str, int]] = []
        self._last_insert = 0.0

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self._build_footer()
        self._append_log("Выберите запись лекции и нажмите «Транскрибировать».")
        self._append_log("Первый запуск скачает модели один раз, дальше всё работает офлайн.")

    def _build_sidebar(self) -> None:
        side = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color=SIDE)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            side,
            text="Whisper",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 0))
        ctk.CTkLabel(
            side,
            text="Локальная транскрибация лекций",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", padx=22, pady=(2, 16))

        device = "NVIDIA GPU · float16" if self.device == "cuda" else "Процессор · int8"
        badge = ctk.CTkFrame(side, fg_color=CARD, corner_radius=12)
        badge.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        ctk.CTkLabel(badge, text="Ускорение", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=14, pady=(10, 0)
        )
        ctk.CTkLabel(badge, text=device, text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=14, pady=(0, 12)
        )

        ctk.CTkLabel(side, text="Файл", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, sticky="w", padx=22
        )
        file_row = ctk.CTkFrame(side, fg_color="transparent")
        file_row.grid(row=4, column=0, sticky="ew", padx=18, pady=(6, 14))
        file_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(
            file_row,
            textvariable=self.file_var,
            placeholder_text="аудио или видео…",
            fg_color=CARD,
            border_color="#2a3441",
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            file_row,
            text="Выбрать",
            width=88,
            fg_color=ACCENT,
            hover_color="#2563eb",
            command=self.pick_file,
        ).grid(row=0, column=1, padx=(8, 0))

        ctk.CTkLabel(side, text="Модель Whisper", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=5, column=0, sticky="w", padx=22
        )
        self.model_combo = ctk.CTkComboBox(
            side,
            values=[item[0] for item in MODELS],
            variable=self.model_var,
            fg_color=CARD,
            border_color="#2a3441",
            button_color=ACCENT,
            dropdown_fg_color=CARD,
        )
        self.model_combo.grid(row=6, column=0, sticky="ew", padx=18, pady=(6, 12))

        ctk.CTkLabel(side, text="Язык", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=7, column=0, sticky="w", padx=22
        )
        self.lang_combo = ctk.CTkComboBox(
            side,
            values=[item[0] for item in LANGUAGES],
            variable=self.lang_var,
            fg_color=CARD,
            border_color="#2a3441",
            button_color=ACCENT,
            dropdown_fg_color=CARD,
        )
        self.lang_combo.grid(row=8, column=0, sticky="ew", padx=18, pady=(6, 12))

        speakers = ctk.CTkFrame(side, fg_color=CARD, corner_radius=14)
        speakers.grid(row=9, column=0, sticky="ew", padx=18, pady=(0, 12))
        speakers.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(speakers, text="Лекторы", text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 0)
        )
        ctk.CTkSwitch(
            speakers,
            text="Определять, кто говорит",
            variable=self.diarize_var,
            progress_color=ACCENT,
            button_color="#f8fafc",
            command=self._toggle_speakers,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(8, 6))
        self.speaker_combo = ctk.CTkComboBox(
            speakers,
            values=[item[0] for item in SPEAKER_COUNTS],
            variable=self.speakers_var,
            fg_color=SIDE,
            border_color="#2a3441",
            button_color=ACCENT,
            dropdown_fg_color=CARD,
        )
        self.speaker_combo.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))

        checks = ctk.CTkFrame(side, fg_color="transparent")
        checks.grid(row=10, column=0, sticky="ew", padx=18, pady=(0, 12))
        ctk.CTkCheckBox(checks, text="TXT", variable=self.txt_var, fg_color=ACCENT).pack(side="left")
        ctk.CTkCheckBox(checks, text="JSON", variable=self.json_var, fg_color=ACCENT).pack(
            side="left", padx=(14, 0)
        )
        ctk.CTkCheckBox(checks, text="SRT", variable=self.srt_var, fg_color=ACCENT).pack(
            side="left", padx=(14, 0)
        )

        self.run_btn = ctk.CTkButton(
            side,
            text="Транскрибировать",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT,
            hover_color="#2563eb",
            command=self.start,
        )
        self.run_btn.grid(row=11, column=0, sticky="ew", padx=18, pady=(8, 22))

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            main,
            text="Транскрипт",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(22, 4))
        self.summary = ctk.CTkLabel(main, text="Лекторы появятся после распознавания", text_color=MUTED)
        self.summary.grid(row=1, column=0, sticky="w", padx=24, pady=(0, 10))

        self.transcript = ctk.CTkTextbox(
            main,
            fg_color=CARD,
            text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=14),
            wrap="word",
            corner_radius=16,
        )
        self.transcript.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 22))
        for i, color in enumerate(SPEAKER_COLORS):
            self.transcript.tag_config(f"spk{i}", foreground=color)
        self.transcript.tag_config("time", foreground=MUTED)
        self.transcript.tag_config("body", foreground=TEXT)

    def _build_footer(self) -> None:
        self.footer = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0)
        self.footer.grid_columnconfigure(0, weight=1)
        self._model_visible = False
        self._job_visible = False

        self.model_row = ctk.CTkFrame(self.footer, fg_color="transparent")
        self.model_row.grid_columnconfigure(0, weight=1)
        self.model_status = ctk.CTkLabel(
            self.model_row, text="Загрузка модели", text_color=MUTED, font=ctk.CTkFont(size=12)
        )
        self.model_status.grid(row=0, column=0, sticky="w", padx=22, pady=(12, 0))
        self.model_pct = ctk.CTkLabel(self.model_row, text="0%", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.model_pct.grid(row=0, column=1, sticky="e", padx=22, pady=(12, 0))
        self.model_bar = ctk.CTkProgressBar(self.model_row, height=10, progress_color=ACCENT, fg_color="#243041")
        self.model_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=22, pady=(6, 12))
        self.model_bar.set(0)

        self.job_row = ctk.CTkFrame(self.footer, fg_color="transparent")
        self.job_row.grid_columnconfigure(0, weight=1)
        self.job_status = ctk.CTkLabel(
            self.job_row, text="Транскрибация", text_color=MUTED, font=ctk.CTkFont(size=12)
        )
        self.job_status.grid(row=0, column=0, sticky="w", padx=22, pady=(8, 0))
        self.job_pct = ctk.CTkLabel(self.job_row, text="0%", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.job_pct.grid(row=0, column=1, sticky="e", padx=22, pady=(8, 0))
        self.job_bar = ctk.CTkProgressBar(self.job_row, height=10, progress_color="#22c55e", fg_color="#243041")
        self.job_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=22, pady=(6, 14))
        self.job_bar.set(0)

    def _layout_footer(self) -> None:
        if self._model_visible or self._job_visible:
            self.footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        else:
            self.footer.grid_remove()
        if self._model_visible:
            self.model_row.grid(row=0, column=0, sticky="ew")
        else:
            self.model_row.grid_remove()
        if self._job_visible:
            self.job_row.grid(row=1, column=0, sticky="ew")
        else:
            self.job_row.grid_remove()

    def _show_model_progress(self) -> None:
        if not self._model_visible:
            self._model_visible = True
            self._layout_footer()

    def _hide_model_progress(self) -> None:
        if self._model_visible:
            self._model_visible = False
            self._layout_footer()

    def _show_job_progress(self) -> None:
        if not self._job_visible:
            self._job_visible = True
            self._layout_footer()

    def _hide_job_progress(self) -> None:
        if self._job_visible:
            self._job_visible = False
            self._layout_footer()

    def _toggle_speakers(self) -> None:
        state = "normal" if self.diarize_var.get() else "disabled"
        self.speaker_combo.configure(state=state)

    def pick_file(self) -> None:
        path = ctk.filedialog.askopenfilename(title="Выберите запись", filetypes=AUDIO_TYPES)
        if path:
            self.file_var.set(path)

    def _append_log(self, text: str, tag: str = "body") -> None:
        self.transcript.insert("end", text + "\n", tag)
        self.transcript.see("end")

    def _ui(self, fn) -> None:
        self.after(0, fn)

    def _set_model_progress(self, value: float, text: str) -> None:
        value = max(0.0, min(1.0, value))
        now = time.monotonic()
        if value < 0.999 and now - self._last_model_ui < 0.2:
            return
        self._last_model_ui = now

        def apply() -> None:
            self._show_model_progress()
            self.model_bar.set(value)
            self.model_status.configure(text=text)
            self.model_pct.configure(text=f"{int(value * 100)}%")

        self._ui(apply)

    def _set_job_progress(self, value: float, text: str) -> None:
        value = max(0.0, min(1.0, value))
        now = time.monotonic()
        if value < 0.999 and now - self._last_job_ui < 0.2:
            return
        self._last_job_ui = now

        def apply() -> None:
            self._show_job_progress()
            self.job_bar.set(value)
            self.job_status.configure(text=text)
            self.job_pct.configure(text=f"{int(value * 100)}%")

        self._ui(apply)

    def start(self) -> None:
        if self.busy:
            return
        path = Path(self.file_var.get().strip().strip('"'))
        if not path.is_file():
            messagebox.showerror("Нет файла", "Сначала выберите аудио или видео файл.")
            return
        if not self.txt_var.get() and not self.json_var.get() and not self.srt_var.get():
            messagebox.showerror("Формат", "Включите хотя бы один формат: TXT, JSON или SRT.")
            return
        self.busy = True
        self.run_btn.configure(state="disabled", text="Идёт распознавание…")
        self.transcript.delete("1.0", "end")
        self.summary.configure(text="Обработка…")
        self._pending_lines = []
        try:
            model_id = dict(MODELS)[self.model_combo.get()]
            language = dict(LANGUAGES)[self.lang_combo.get()]
            num_speakers = dict(SPEAKER_COUNTS)[self.speaker_combo.get()]
        except KeyError:
            self.busy = False
            self.run_btn.configure(state="normal", text="Транскрибировать")
            messagebox.showerror("Настройки", "Проверьте модель, язык и число лекторов.")
            return
        job = Job(
            path=path,
            model_id=model_id,
            language=language,
            num_speakers=num_speakers,
            use_diarize=bool(self.diarize_var.get()),
            save_txt=bool(self.txt_var.get()),
            save_json=bool(self.json_var.get()),
            save_srt=bool(self.srt_var.get()),
        )
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _download_whisper(self, model_id: str) -> str:
        from tqdm.auto import tqdm
        import huggingface_hub
        from faster_whisper.utils import _MODELS

        repo = _MODELS.get(model_id, model_id)
        app = self

        class HubTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["file"] = StringIO()
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                result = super().update(n)
                total = self.total or 0
                if total:
                    app._set_model_progress(0.08 + 0.72 * min(1.0, self.n / total), "Скачивание модели Whisper…")
                return result

        return huggingface_hub.snapshot_download(
            repo,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ],
            tqdm_class=HubTqdm,
        )

    def _run(self, job: Job) -> None:
        try:
            ensure_ffmpeg()
            path = job.path
            self._ui(lambda: self._append_log(f"Файл: {path.name}", "time"))
            self._set_model_progress(0.08, "Скачивание / проверка Whisper…")
            model_path = self._download_whisper(job.model_id)
            self._set_model_progress(0.82, "Загрузка модели в память…")

            from faster_whisper import WhisperModel

            model = WhisperModel(model_path, device=self.device, compute_type=self.compute)
            self._set_model_progress(1.0, "Модель готова")
            self._ui(self._hide_model_progress)
            self._set_job_progress(0.04, "Распознаю речь…")

            segments, info = model.transcribe(
                str(path),
                language=job.language,
                vad_filter=True,
                beam_size=5,
                word_timestamps=bool(job.use_diarize and job.num_speakers != 1),
            )
            duration = float(getattr(info, "duration", 0) or 0)
            detected = info.language or "неизвестно"
            self._ui(lambda: self._append_log(f"Язык: {detected}", "time"))

            rows: list[tuple[float, float, str, str]] = []
            words: list[tuple[float, float, str]] = []
            for seg in segments:
                text = (seg.text or "").strip()
                rows.append((seg.start, seg.end, "", text))
                for word in getattr(seg, "words", None) or []:
                    token = (getattr(word, "word", "") or "").strip()
                    if token:
                        words.append((float(word.start), float(word.end), token))
                if duration > 0:
                    self._set_job_progress(min(0.74, 0.04 + 0.70 * (seg.end / duration)), "Транскрибация…")
                if text:
                    self._queue_segment(fmt_ts(seg.start) + "  ", "", text, 0)

            self._flush_segments()
            self._set_job_progress(0.76, "Транскрипт готов")

            turns: list[tuple[float, float, str]] = []
            if job.use_diarize and job.num_speakers == 1:
                rows = [(start, end, "Лектор 1", text) for start, end, _, text in rows]
                self._ui(lambda: self.summary.configure(text="Найдены: Лектор 1"))
                self._ui(lambda r=rows: self._render_rows(r, detected))
            elif job.use_diarize:
                from diarize import align_transcript, diarize_isolated

                try:
                    turns = diarize_isolated(
                        path,
                        num_speakers=job.num_speakers,
                        progress=lambda f, t: self._set_job_progress(0.76 + 0.22 * f, t),
                    )
                    names = ", ".join(dict.fromkeys(name for *_, name in turns)) or "не найдены"
                    self._ui(lambda n=names: self.summary.configure(text=f"Найдены: {n}"))
                    rows = align_transcript(rows, turns, words=words)
                    self._ui(lambda r=rows: self._render_rows(r, detected))
                except Exception:
                    self._ui(lambda: self._append_log("Не удалось определить лекторов, текст сохранён без них.", "time"))
                    self._write_error_log(traceback.format_exc())

            self._set_job_progress(1.0, "Готово")
            saved = []
            if job.save_txt:
                txt_path = path.with_suffix(".txt")
                txt_path.write_text(to_txt(rows), encoding="utf-8")
                saved.append(txt_path)
            if job.save_json:
                json_path = path.with_suffix(".json")
                json_path.write_text(to_json(rows, path, detected), encoding="utf-8")
                saved.append(json_path)
            if job.save_srt:
                srt_path = path.with_suffix(".srt")
                srt_path.write_text(to_srt(rows), encoding="utf-8")
                saved.append(srt_path)
            names = ", ".join(p.name for p in saved) or "ничего"
            self._ui(lambda: self._append_log(f"\nСохранено: {names}", "time"))
            self._ui(
                lambda: messagebox.showinfo("Готово", "Транскрипт сохранён:\n" + "\n".join(str(p) for p in saved))
            )
        except Exception:
            err = traceback.format_exc()
            self._write_error_log(err)
            self._ui(lambda: self._append_log("Ошибка:\n" + err, "time"))
            self._ui(lambda: messagebox.showerror("Ошибка", "Не получилось транскрибировать. Смотрите журнал."))
        finally:
            self._ui(self._idle)

    def _queue_segment(self, time_text: str, speaker: str, body: str, color_index: int) -> None:
        self._pending_lines.append((time_text, speaker, body, color_index))
        now = time.monotonic()
        if len(self._pending_lines) < 8 and now - self._last_insert < 0.25:
            return
        batch = self._pending_lines
        self._pending_lines = []
        self._last_insert = now
        self._ui(lambda b=batch: self._insert_batch(b))

    def _flush_segments(self) -> None:
        if not self._pending_lines:
            return
        batch = self._pending_lines
        self._pending_lines = []
        self._ui(lambda b=batch: self._insert_batch(b))

    def _insert_batch(self, batch: list[tuple[str, str, str, int]]) -> None:
        for time_text, speaker, body, color_index in batch:
            self._insert_segment(time_text, speaker, body, color_index)

    def _render_rows(self, rows: list[tuple[float, float, str, str]], detected: str) -> None:
        self.transcript.delete("1.0", "end")
        self._append_log(f"Язык: {detected}", "time")
        for start, _end, speaker, text in rows:
            if not text:
                continue
            spk_index = 0
            if speaker.startswith("Лектор "):
                try:
                    spk_index = max(0, int(speaker.split()[-1]) - 1) % len(SPEAKER_COLORS)
                except ValueError:
                    spk_index = 0
            self._insert_segment(fmt_ts(start) + "  ", f"{speaker}  " if speaker else "", text, spk_index)

    def _write_error_log(self, text: str) -> None:
        try:
            (app_data_dir() / "last_error.log").write_text(text, encoding="utf-8")
        except OSError:
            pass

    def _insert_segment(self, time_text: str, speaker: str, body: str, color_index: int) -> None:
        self.transcript.insert("end", time_text, "time")
        if speaker:
            self.transcript.insert("end", speaker, f"spk{color_index}")
        self.transcript.insert("end", body + "\n", "body")
        self.transcript.see("end")

    def _idle(self) -> None:
        self.busy = False
        self.run_btn.configure(state="normal", text="Транскрибировать")
        self._hide_model_progress()
        self._hide_job_progress()


def main() -> None:
    ensure_stdio()
    enable_crash_log()
    ensure_ffmpeg()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
