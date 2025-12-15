import os
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

# Config
OUTPUT_DIR = "tts_output"
VOICE_SAMPLE = "voices/vera.wav"
LANGUAGE = "en"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load XTTS ONCE at startup
print("🔊 Loading XTTS v2...")
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)
print("✅ XTTS ready")


def speak(text: str, filename: str = "response.wav"):

    if not text.strip():
        return None

    output_path = os.path.join(OUTPUT_DIR, filename)

    # Generate speech
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=VOICE_SAMPLE,
        language=LANGUAGE
    )

    # Play audio
    audio, sr = sf.read(output_path)
    sd.play(audio, sr)
    sd.wait()  # Wait until playback finishes

    return output_path