# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

site = Path(sys.prefix) / "Lib" / "site-packages"
nvidia = site / "nvidia"

SKIP_DLLS = {
    "nvblas64_12.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cufft64_11.dll",
    "cufftw64_11.dll",
    "nvrtc-builtins64_129.dll",
    "nvrtc64_120_0.alt.dll",
    "nvrtc64_120_0.dll",
}

binaries: list[tuple[str, str]] = []
if nvidia.is_dir():
    for dll in nvidia.rglob("*.dll"):
        if dll.name.lower() in SKIP_DLLS:
            continue
        binaries.append((str(dll), str(dll.parent.relative_to(site))))

ffmpeg = None
for candidate in (
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
):
    matches = sorted(candidate.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")) if candidate.is_dir() else []
    if matches:
        ffmpeg = matches[-1]
        break
if ffmpeg is None:
    from shutil import which

    found = which("ffmpeg")
    if found:
        ffmpeg = Path(found)
if ffmpeg and ffmpeg.is_file():
    binaries.append((str(ffmpeg), "."))

datas: list[tuple[str, str]] = []
hiddenimports: list[str] = [
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "av",
    "tokenizers",
    "huggingface_hub",
    "tqdm",
    "numpy",
    "customtkinter",
    "darkdetect",
    "sherpa_onnx",
    "diarize",
]

for package in (
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "av",
    "tokenizers",
    "huggingface_hub",
    "customtkinter",
    "sherpa_onnx",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("av")
binaries += collect_dynamic_libs("sherpa_onnx")

a = Analysis(
    ["transcribe_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperTranscribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WhisperTranscribe",
)
