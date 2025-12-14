import vosk
import pyaudio
import json
import os

MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"
RATE = 16000
CHUNK = 4096

def listen():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Vosk model not found")

    model = vosk.Model(MODEL_PATH)
    rec = vosk.KaldiRecognizer(model, RATE)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    stream.start_stream()
    print("🎙 Voice agent listening...")

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                yield text