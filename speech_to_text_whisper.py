import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import subprocess
import os

MODEL_SIZE = "base"  # tiny / base / small / medium
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds per utterance

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",          # auto GPU if available
    compute_type="int8"    # fast on CPU
)

audio_queue = queue.Queue()

def transcribe_webm(audio_bytes: bytes) -> str:
    # 1️⃣ Create temp webm file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(audio_bytes)
        src_path = src.name  # save path

    # 2️⃣ Create temp wav path (file NOT open)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
        dst_path = dst.name

    print("from to", src_path, "->", dst_path)
    try:
        # 3️⃣ ffmpeg runs AFTER files are closed
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                dst_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # 4️⃣ Whisper reads wav
        segments, info = model.transcribe(dst_path)
        text = "".join(segment.text for segment in segments)
        return text

    finally:
        # 5️⃣ Cleanup (VERY IMPORTANT)
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

def listen():
    print("🎙 Whisper listening...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        buffer = []

        while True:
            try:
                data = audio_queue.get_nowait()  # non-blocking
            except queue.Empty:
                continue  # no audio yet, go back to flag check
            buffer.append(data)

            length = sum(len(b) for b in buffer) / SAMPLE_RATE
            if length >= CHUNK_DURATION:
                audio = np.concatenate(buffer, axis=0).flatten()
                buffer.clear()

                segments, _ = model.transcribe(
                    audio,
                    language="en",
                    vad_filter=True
                )

                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    yield text