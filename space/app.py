"""Stem Studio AI 서버 (Hugging Face Spaces · ZeroGPU · 무료)

원곡 1개 → Demucs(htdemucs) AI 분리 → DRUMS / BASS / INST / VOCALS 4개 MP3(320kbps)
웹앱(index.html)은 이 Space의 /separate API를 호출한다.
"""
import gradio as gr
import numpy as np
import spaces
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from stems import MAX_SECONDS, STEM_ORDER, load_audio, probe_seconds, write_stems

MODEL_NAME = "htdemucs"  # 드럼·베이스·기타악기·보컬 4스템 공식 모델

model = get_model(MODEL_NAME)
model.eval()
model.to("cuda")  # ZeroGPU: GPU 함수 안에서만 실제 GPU가 붙는다


def _gpu_seconds(audio_np, seconds):
    # 곡 길이에 맞춰 GPU 시간을 적게 요청 → 무료 일일 한도를 아낀다
    return int(min(120, 20 + seconds * 0.15))


@spaces.GPU(duration=_gpu_seconds)
def _separate_on_gpu(audio_np: np.ndarray, seconds: float) -> dict:
    wav = torch.from_numpy(audio_np).to("cuda")
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        out = apply_model(model, wav[None], device="cuda", split=True,
                          overlap=0.25, shifts=1, progress=False)[0]
    out = out * std + mean
    out = out.cpu().numpy()
    return {name: out[i] for i, name in enumerate(model.sources)}


def separate(song):
    """원곡 파일 경로 → [DRUMS, BASS, INST, VOCALS] MP3 경로 4개"""
    if not song:
        raise gr.Error("음악 파일을 먼저 올려 주세요.")
    seconds = probe_seconds(song)
    if seconds > MAX_SECONDS:
        raise gr.Error(f"무료 서버에서는 {MAX_SECONDS // 60}분 이하 곡만 처리할 수 있어요.")
    audio = load_audio(song)
    seconds = audio.shape[1] / 44100
    stems = _separate_on_gpu(audio, seconds)
    missing = [k for k in STEM_ORDER if k not in stems]
    if missing:
        raise gr.Error(f"모델 출력에 {missing} 트랙이 없어요.")
    return write_stems(stems, song)


with gr.Blocks(title="Stem Studio AI") as demo:
    gr.Markdown("## ✦ Stem Studio AI 서버\n원곡 하나를 올리면 DRUMS · BASS · INST · VOCALS 4개 MP3(320kbps)로 분리합니다.")
    inp = gr.Audio(type="filepath", label="원곡 (MP3 등)")
    btn = gr.Button("AI 스템 분리", variant="primary")
    outs = [gr.File(label=n) for n in ["DRUMS", "BASS", "INST", "VOCALS"]]
    btn.click(separate, inputs=inp, outputs=outs, api_name="separate")

demo.queue(max_size=8).launch()
