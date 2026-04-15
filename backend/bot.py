import os, base64
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
from dotenv import load_dotenv
from sarvamai import SarvamAI
from groq import Groq

# =============================
# INIT
# =============================
load_dotenv()

sarvam = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

SAMPLE_RATE = 16000
RECORD_SECONDS = 6

# =============================
# AUDIO
# =============================
def record_audio(filename="input.wav", duration=RECORD_SECONDS):
    print(f"\n🎤 Listening for {duration}s...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, recording, SAMPLE_RATE)

def play_audio(file):
    wave_obj = sa.WaveObject.from_wave_file(file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# =============================
# STT (Sarvam)
# =============================
def speech_to_text(file="input.wav"):
    with open(file, "rb") as f:
        res = sarvam.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="transcribe"
        )
    return res.transcript.strip(), res.language_code

# =============================
# LLM (Groq)
# =============================
def chat_with_llm(text: str) -> str:

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Reply in same language. "
                    "Give direct answer only. "
                    "Max 20 words. No explanation."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.3,
        max_tokens=80
    )

    reply = response.choices[0].message.content.strip()

    # minimal safety (no over-cleaning!)
    if not reply or len(reply.split()) < 2:
        return "Please ask clearly."

    return reply

# =============================
# TTS (Sarvam)
# =============================
def text_to_speech(text: str, lang: str = "en-IN", output="reply.wav"):

    lang_map = {
        "te": "te-IN", "hi": "hi-IN", "ta": "ta-IN",
        "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN",
        "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN",
        "en": "en-IN",
    }

    lang_code = lang_map.get(lang[:2].lower(), "en-IN")

    speaker_map = {
        "te-IN": "arvind",
        "hi-IN": "meera",
        "ta-IN": "meera",
        "kn-IN": "meera",
        "ml-IN": "meera",
        "en-IN": "arya",
    }

    speaker = speaker_map.get(lang_code, "meera")

    res = sarvam.text_to_speech.convert(
        text=text,
        target_language_code=lang_code,
        speaker=speaker,
    )

    audio_bytes = base64.b64decode(res.audios[0])

    with open(output, "wb") as f:
        f.write(audio_bytes)

    return output

# =============================
# MAIN LOOP
# =============================
def run_bot():
    print("━" * 50)
    print("   Voice Assistant (Groq + Sarvam)")
    print("━" * 50)

    while True:
        try:
            record_audio()

            text, lang = speech_to_text()

            if not text:
                print("(no speech detected)")
                continue

            print(f"👤 [{lang}] {text}")

            if text.lower() in {"exit", "quit", "stop"}:
                print("👋 Bye")
                break

            reply = chat_with_llm(text)
            print(f"🤖 {reply}")

            audio_file = text_to_speech(reply, lang)
            play_audio(audio_file)

        except KeyboardInterrupt:
            print("\n👋 Stopped")
            break

        except Exception as e:
            print("❌ Error:", e)

# =============================
if __name__ == "__main__":
    run_bot()