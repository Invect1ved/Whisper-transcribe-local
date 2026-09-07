"""Определение лекторов через sherpa-onnx (pyannote segmentation + speaker embeddings)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

ProgressFn = Callable[[float, str], None]

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_MODELS = [
    (
        "wespeaker_en_voxceleb_resnet293_LM.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet293_LM.onnx",
    ),
    (
        "wespeaker_en_voxceleb_CAM++_LM.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_CAM++_LM.onnx",
    ),
    (
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    ),
]


def models_dir() -> Path:
    local = Path.home() / "AppData" / "Local" / "Whisper" / "diarization"
    local.mkdir(parents=True, exist_ok=True)
    return local


def enable_cuda_dlls() -> None:
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return
    dirs = [p for p in nvidia.glob("*/bin") if p.is_dir()]
    if not dirs:
        return
    os.environ["PATH"] = os.pathsep.join([str(p) for p in dirs] + [os.environ.get("PATH", "")])
    if hasattr(os, "add_dll_directory"):
        for bindir in dirs:
            os.add_dll_directory(str(bindir))


enable_cuda_dlls()


def detect_provider() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _download(url: str, dest: Path, progress: ProgressFn, label: str) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        progress(1.0, f"{label}: уже скачано")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-transcribe-local"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                progress(min(0.99, done / total), label)
    tmp.replace(dest)
    progress(1.0, f"{label}: готово")


def _extract_segmentation(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)
    found = list(dest_dir.rglob("model.onnx"))
    if not found:
        raise FileNotFoundError("Не найден model.onnx в архиве сегментации")
    return found[0]


def ensure_models(progress: ProgressFn | None = None) -> tuple[Path, Path]:
    progress = progress or (lambda *_: None)
    root = models_dir()
    archive = root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    seg_guess = root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"

    if seg_guess.is_file():
        seg = seg_guess
        progress(0.45, "Модель сегментации готова")
    else:
        found = [p for p in root.rglob("model.onnx") if "segmentation" in str(p).lower()]
        if found:
            seg = found[0]
            progress(0.45, "Модель сегментации готова")
        else:
            _download(
                SEGMENTATION_URL,
                archive,
                lambda f, _t: progress(0.45 * f, "Скачивание модели лекторов…"),
                "Сегментация",
            )
            seg = _extract_segmentation(archive, root)

    embed: Path | None = None
    last_error: Exception | None = None
    for index, (name, url) in enumerate(EMBEDDING_MODELS):
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size <= 1_000_000:
            try:
                candidate.unlink()
            except OSError:
                pass
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            embed = candidate
            break
        try:
            _download(
                url,
                candidate,
                lambda f, _t, i=index: progress(0.45 + 0.55 * ((i + f) / len(EMBEDDING_MODELS)), "Скачивание модели голоса…"),
                "Эмбеддинги",
            )
            embed = candidate
            break
        except Exception as exc:
            last_error = exc
            continue
    if embed is None:
        raise RuntimeError(f"Не удалось скачать модель голоса: {last_error}")
    progress(1.0, "Модель голоса готова")
    return seg, embed


def _tool(name: str) -> str:
    from shutil import which

    found = which(name)
    if found:
        return found
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.is_dir():
        matches = sorted(winget.glob(f"Gyan.FFmpeg*/ffmpeg-*/bin/{name}.exe"))
        if matches:
            return str(matches[-1])
    raise FileNotFoundError(f"Не найден {name}. Нужен FFmpeg.")


def _run_hidden(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    if os.name == "nt":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(cmd, **kwargs)


def _probe_duration(path: Path) -> float:
    try:
        result = _run_hidden(
            [
                _tool("ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return float(result.stdout.strip().split()[0])
    except Exception:
        return 0.0


def _load_audio(path: Path, sample_rate: int, progress: ProgressFn | None = None) -> np.ndarray:
    progress = progress or (lambda *_: None)
    duration = _probe_duration(path)
    expected = int(duration * sample_rate) if duration > 0 else 0
    cmd = [
        _tool("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
    assert proc.stdout is not None
    chunks: list[np.ndarray] = []
    samples = 0
    last_emit = -1.0
    while True:
        raw = proc.stdout.read(sample_rate * 4 * 2)
        if not raw:
            break
        raw = raw[: len(raw) - (len(raw) % 4)]
        if not raw:
            continue
        piece = np.frombuffer(raw, dtype=np.float32).copy()
        chunks.append(piece)
        samples += len(piece)
        frac = min(0.99, samples / expected) if expected else 0.0
        if frac - last_emit >= 0.02 or expected == 0:
            last_emit = frac
            if expected:
                progress(frac, f"Чтение аудио для лекторов… {samples / sample_rate:.0f} с")
            else:
                progress(0.05, f"Чтение аудио для лекторов… {samples / sample_rate:.0f} с")
    code = proc.wait()
    if code != 0:
        raise RuntimeError("FFmpeg не смог прочитать файл для лекторов")
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks)
    progress(1.0, "Аудио прочитано")
    return audio


def _relabel(turns: list[tuple[float, float, int]]) -> list[tuple[float, float, str]]:
    mapping: dict[int, str] = {}
    labeled: list[tuple[float, float, str]] = []
    for start, end, spk in turns:
        if spk not in mapping:
            mapping[spk] = f"Лектор {len(mapping) + 1}"
        labeled.append((start, end, mapping[spk]))
    return labeled


def _smooth_turns(
    turns: list[tuple[float, float, int]],
    min_dur: float = 0.85,
    max_gap: float = 0.8,
) -> list[tuple[float, float, int]]:
    if not turns:
        return []
    ordered = sorted((float(s), float(e), int(spk)) for s, e, spk in turns if e > s)
    merged: list[list[float | int]] = []
    for start, end, spk in ordered:
        if merged and spk == merged[-1][2] and start - float(merged[-1][1]) <= max_gap:
            merged[-1][1] = max(float(merged[-1][1]), end)
        else:
            merged.append([start, end, spk])

    items = merged
    for _ in range(4):
        out: list[list[float | int]] = []
        changed = False
        for index, item in enumerate(items):
            start, end, spk = float(item[0]), float(item[1]), int(item[2])
            if end - start >= min_dur or (not out and index + 1 >= len(items)):
                if out and spk == out[-1][2] and start - float(out[-1][1]) <= max_gap:
                    out[-1][1] = max(float(out[-1][1]), end)
                else:
                    out.append([start, end, spk])
                continue
            prev_len = float(out[-1][1]) - float(out[-1][0]) if out else -1.0
            nxt = items[index + 1] if index + 1 < len(items) else None
            next_len = (float(nxt[1]) - float(nxt[0])) if nxt is not None else -1.0
            if nxt is not None and next_len >= prev_len:
                nxt[0] = min(float(nxt[0]), start)
                changed = True
            elif out:
                out[-1][1] = max(float(out[-1][1]), end)
                changed = True
            else:
                out.append([start, end, spk])
        items = out
        if not changed:
            break

    cleaned: list[list[float | int]] = []
    for start, end, spk in items:
        if cleaned and spk == cleaned[-1][2] and float(start) - float(cleaned[-1][1]) <= max_gap:
            cleaned[-1][1] = max(float(cleaned[-1][1]), float(end))
        else:
            cleaned.append([float(start), float(end), int(spk)])
    return [(float(s), float(e), int(spk)) for s, e, spk in cleaned]


def _mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return mean
    return mean / norm


def _embed_span(extractor, audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray | None:
    take = min(max(end - start, 0.0), 8.0)
    if take < 0.6:
        return None
    mid = (start + end) / 2.0
    a = max(0.0, mid - take / 2.0)
    b = min(audio.size / sr, a + take)
    i0, i1 = int(a * sr), int(b * sr)
    samples = np.ascontiguousarray(audio[i0:i1], dtype=np.float32)
    if samples.size < int(0.5 * sr):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sr, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    vec = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def _cluster_vectors(vectors: list[np.ndarray], k: int | None, threshold: float) -> list[int]:
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    cents = {i: vectors[i] for i in range(n)}
    while len(active) > 1:
        if k is not None and len(active) <= k:
            break
        best_d = 2.0
        best: tuple[int, int] | None = None
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                dist = 1.0 - float(np.dot(cents[a], cents[b]))
                if dist < best_d:
                    best_d = dist
                    best = (a, b)
        if best is None:
            break
        if k is None and best_d > threshold:
            break
        a, b = best
        clusters[a].extend(clusters[b])
        del clusters[b]
        active.remove(b)
        cents[a] = _mean_embedding([vectors[i] for i in clusters[a]])
    labels = [0] * n
    for cid, members in enumerate(clusters.values()):
        for idx in members:
            labels[idx] = cid
    return labels


def _recluster_speakers(
    audio: np.ndarray,
    sr: int,
    turns: list[tuple[float, float, int]],
    emb_model: Path,
    provider: str,
    num_speakers: int | None,
    progress: ProgressFn,
) -> list[tuple[float, float, int]]:
    import sherpa_onnx

    ids = sorted({spk for *_, spk in turns})
    if len(ids) <= 1:
        return turns
    progress(0.94, "Уточнение лекторов по голосу…")
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(emb_model),
        num_threads=1 if provider == "cuda" else 2,
        provider=provider,
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    vectors: list[np.ndarray] = []
    kept: list[int] = []
    for spk in ids:
        spans = sorted(((e - s, s, e) for s, e, sid in turns if sid == spk), reverse=True)
        collected: list[np.ndarray] = []
        for _, start, end in spans[:6]:
            vec = _embed_span(extractor, audio, sr, start, end)
            if vec is not None:
                collected.append(vec)
            if len(collected) >= 4:
                break
        if collected:
            vectors.append(_mean_embedding(collected))
            kept.append(spk)
    if len(vectors) <= 1:
        return turns
    k = int(num_speakers) if num_speakers and num_speakers > 1 else None
    if k is not None:
        k = min(k, len(vectors))
    labels = _cluster_vectors(vectors, k=k, threshold=0.32)
    mapping = {spk: lab for spk, lab in zip(kept, labels)}
    leftover = max(labels) + 1 if labels else 0
    remapped = []
    for start, end, spk in turns:
        if spk in mapping:
            remapped.append((start, end, mapping[spk]))
        else:
            remapped.append((start, end, leftover))
            leftover += 1
    return remapped


def _stitch_speakers(
    prev: list[tuple[float, float, int]],
    incoming: list[tuple[float, float, int]],
    overlap_start: float,
    next_id: int,
) -> tuple[list[tuple[float, float, int]], int]:
    locals_ids = sorted({spk for _, _, spk in incoming})
    mapping: dict[int, int] = {}
    for local_id in locals_ids:
        best_global = None
        best_ov = 0.4
        for ls, le, _spk in incoming:
            if _spk != local_id:
                continue
            if le <= overlap_start:
                continue
            for ps, pe, gid in prev:
                if pe <= overlap_start:
                    continue
                ov = min(le, pe) - max(ls, ps)
                if ov > best_ov:
                    best_ov = ov
                    best_global = gid
        if best_global is None:
            mapping[local_id] = next_id
            next_id += 1
        else:
            mapping[local_id] = best_global
    stitched = [(s, e, mapping[spk]) for s, e, spk in incoming]
    return stitched, next_id


def diarize(
    path: Path,
    num_speakers: int | None = None,
    progress: ProgressFn | None = None,
    download_progress: ProgressFn | None = None,
) -> list[tuple[float, float, str]]:
    """Вернёт отрезки (start, end, 'Лектор N')."""
    import sherpa_onnx

    progress = progress or (lambda *_: None)
    if num_speakers == 1:
        progress(1.0, "Один лектор")
        return [(0.0, 10**9, "Лектор 1")]

    seg_model, emb_model = ensure_models(download_progress)

    clusters = -1 if not num_speakers or num_speakers < 1 else int(num_speakers)
    pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model))

    def make_engine(provider: str):
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=pyannote,
                num_threads=1 if provider == "cuda" else 2,
                provider=provider,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model),
                num_threads=1 if provider == "cuda" else 2,
                provider=provider,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=clusters,
                threshold=0.45 if clusters < 1 else 0.5,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if hasattr(config, "validate") and not config.validate():
            raise RuntimeError("Конфиг диаризации невалиден — проверьте модели лекторов")
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    provider = detect_provider()
    try:
        engine = make_engine(provider)
    except Exception:
        if provider != "cpu":
            provider = "cpu"
            engine = make_engine("cpu")
        else:
            raise

    device_label = "GPU" if provider == "cuda" else "CPU"
    progress(0.02, f"Чтение аудио для лекторов ({device_label})…")
    audio = _load_audio(path, engine.sample_rate, progress=lambda f, t: progress(0.02 + 0.18 * f, t))
    if audio.size == 0:
        progress(1.0, "Нет аудио")
        return []

    sr = int(engine.sample_rate)
    duration_s = audio.size / sr

    def turns_from_result(result, offset: float = 0.0) -> list[tuple[float, float, int]]:
        if hasattr(result, "sort_by_start_time"):
            result = result.sort_by_start_time()
        return [(float(item.start) + offset, float(item.end) + offset, int(item.speaker)) for item in result]

    def process_full() -> list[tuple[float, float, int]]:
        def cb(processed: int, total: int) -> int:
            frac = processed / total if total else 1.0
            progress(0.22 + 0.76 * frac, f"Определение лекторов на {device_label}…")
            return 0

        progress(0.22, f"Определение лекторов на {device_label}…")
        piece = np.ascontiguousarray(audio, dtype=np.float32)
        return turns_from_result(engine.process(piece, callback=cb))

    def process_chunked(chunk_n: int, hop_n: int) -> list[tuple[float, float, int]]:
        total = int(audio.size)
        starts: list[int] = []
        pos = 0
        while pos < total:
            starts.append(pos)
            if pos + chunk_n >= total:
                break
            pos += hop_n

        all_turns: list[tuple[float, float, int]] = []
        next_id = 0
        n_chunks = max(len(starts), 1)
        for index, start in enumerate(starts):
            end = min(total, start + chunk_n)
            if end - start < sr:
                continue
            piece = np.ascontiguousarray(audio[start:end], dtype=np.float32)
            offset = start / sr
            progress(0.20 + 0.78 * (index / n_chunks), f"Определение лекторов… {index + 1}/{n_chunks}")
            incoming = turns_from_result(engine.process(piece), offset)
            if not incoming:
                continue
            if not all_turns:
                remap: dict[int, int] = {}
                mapped = []
                for s, e, spk in incoming:
                    if spk not in remap:
                        remap[spk] = next_id
                        next_id += 1
                    mapped.append((s, e, remap[spk]))
                all_turns.extend(mapped)
            else:
                stitched, next_id = _stitch_speakers(all_turns, incoming, offset, next_id)
                all_turns.extend(stitched)
        return all_turns

    try:
        if provider == "cuda" or duration_s <= 240:
            turns = process_full()
        else:
            turns = process_chunked(180 * sr, 160 * sr)
    except Exception:
        if provider != "cuda":
            raise
        try:
            turns = process_chunked(600 * sr, 570 * sr)
        except Exception:
            provider = "cpu"
            device_label = "CPU"
            engine = make_engine("cpu")
            turns = process_chunked(180 * sr, 160 * sr)

    turns = _smooth_turns(turns)
    try:
        turns = _recluster_speakers(audio, sr, turns, emb_model, provider, num_speakers, progress)
        turns = _smooth_turns(turns)
    except Exception:
        pass
    progress(1.0, "Лекторы определены")
    return _relabel(turns)


def diarize_isolated(
    path: Path,
    num_speakers: int | None = None,
    progress: ProgressFn | None = None,
) -> list[tuple[float, float, str]]:
    """Запускает диаризацию в отдельном процессе, чтобы падение ONNX не убивало GUI."""
    progress = progress or (lambda *_: None)
    exe = sys.executable
    if Path(exe).name.lower() == "pythonw.exe":
        candidate = Path(exe).with_name("python.exe")
        if candidate.is_file():
            exe = str(candidate)
    cmd = [
        exe,
        str(Path(__file__).resolve()),
        "--json",
        str(path),
        "--speakers",
        str(-1 if not num_speakers or num_speakers < 1 else int(num_speakers)),
    ]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=flags,
        env=env,
    )
    turns: list[tuple[float, float, str]] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "progress" in msg:
            progress(float(msg["progress"]), str(msg.get("text") or "Определение лекторов…"))
        if "turns" in msg:
            turns = [(float(a), float(b), str(c)) for a, b, c in msg["turns"]]
        if "error" in msg:
            raise RuntimeError(str(msg["error"]))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Процесс определения лекторов завершился с кодом {code}")
    return turns


def assign_speaker(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:
    mid = (start + end) / 2.0
    for t0, t1, name in turns:
        if t0 <= mid <= t1:
            return name
    best = ""
    best_ov = 0.0
    for t0, t1, name in turns:
        overlap = min(end, t1) - max(start, t0)
        if overlap > best_ov:
            best_ov = overlap
            best = name
    return best or "Лектор"


def align_transcript(
    rows: list[tuple[float, float, str, str]],
    turns: list[tuple[float, float, str]],
    words: list[tuple[float, float, str]] | None = None,
) -> list[tuple[float, float, str, str]]:
    if not turns:
        return rows
    if not words:
        return [(start, end, assign_speaker(start, end, turns), text) for start, end, _, text in rows]

    grouped: list[tuple[float, float, str, str]] = []
    buf: list[str] = []
    cur_spk = ""
    cur_start = 0.0
    cur_end = 0.0
    for start, end, token in words:
        text = (token or "").strip()
        if not text:
            continue
        spk = assign_speaker(start, end, turns)
        if buf and spk != cur_spk:
            grouped.append((cur_start, cur_end, cur_spk, " ".join(buf)))
            buf = []
        if not buf:
            cur_spk = spk
            cur_start = start
        buf.append(text)
        cur_end = end
    if buf:
        grouped.append((cur_start, cur_end, cur_spk, " ".join(buf)))
    return grouped or [(start, end, assign_speaker(start, end, turns), text) for start, end, _, text in rows]


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="path")
    parser.add_argument("--speakers", type=int, default=-1)
    args = parser.parse_args()
    if not args.path:
        return 2

    def emit(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def prog(frac: float, text: str) -> None:
        emit({"progress": float(frac), "text": text})

    try:
        speakers = None if args.speakers < 1 else args.speakers
        turns = diarize(Path(args.path), num_speakers=speakers, progress=prog, download_progress=prog)
        emit({"turns": turns})
        return 0
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
