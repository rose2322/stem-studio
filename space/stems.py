"""Stem Studio 핵심 처리: 오디오 읽기 → AI 분리 → 320kbps MP3 저장.

AI 모델(Demucs)과 무관한 부분(디코딩/인코딩/파일 이름)은 여기서 처리하고,
실제 분리는 app.py 의 GPU 함수가 맡는다.
"""
import os
import re
import subprocess
import tempfile

import numpy as np

SAMPLE_RATE = 44100          # Demucs htdemucs 학습 샘플레이트
MP3_BITRATE = "320k"         # 최종 목표 음질
MAX_SECONDS = 10 * 60        # 무료 GPU 한도를 지키기 위한 최대 길이(10분)

# Demucs 출력 이름 → 화면/파일 이름
STEM_LABELS = {
    "drums": "DRUMS",
    "bass": "BASS",
    "other": "INST",     # 기타·피아노·신스 등 나머지 악기
    "vocals": "VOCALS",
}
STEM_ORDER = ["drums", "bass", "other", "vocals"]


def safe_basename(path: str) -> str:
    """원곡 파일 이름에서 확장자를 떼고 파일명에 못 쓰는 문자를 정리."""
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name[:80] or "song"


def probe_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def load_audio(path: str) -> np.ndarray:
    """어떤 형식이든 ffmpeg로 44.1kHz 스테레오 float32 (2, N) 배열로 읽는다."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("오디오 파일을 읽을 수 없어요: " + proc.stderr.decode(errors="ignore")[-300:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T
    return np.ascontiguousarray(audio)


def save_mp3(audio: np.ndarray, out_path: str) -> str:
    """(2, N) float 배열을 320kbps CBR MP3로 저장."""
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    pcm = np.ascontiguousarray(audio.T).tobytes()
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2",
         "-ar", str(SAMPLE_RATE), "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, out_path],
        input=pcm, capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("MP3 저장 실패: " + proc.stderr.decode(errors="ignore")[-300:])
    return out_path


def write_stems(stems: dict, song_path: str, out_dir: str | None = None) -> list[str]:
    """{'drums': arr, ...} → ['곡명_DRUMS.mp3', ..., '곡명_VOCALS.mp3'] 순서로 저장."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="stems_")
    os.makedirs(out_dir, exist_ok=True)
    base = safe_basename(song_path)
    paths = []
    for key in STEM_ORDER:
        path = os.path.join(out_dir, f"{base}_{STEM_LABELS[key]}.mp3")
        paths.append(save_mp3(stems[key], path))
    return paths
