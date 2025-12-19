import os
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # tiny / base / small / medium
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds per utterance

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",          # auto GPU if available
    compute_type="int8"    # fast on CPU
)

audio_queue = queue.Queue()

def transcribe_audio(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"]

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