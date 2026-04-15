from dotenv import load_dotenv
import os, time, wave, tempfile, asyncio, base64, re
import webrtcvad
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sarvamai import SarvamAI
from groq import Groq

# =============================
# INIT
# =============================
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# AUDIO CONFIG
# =============================
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SIZE * 2

vad = webrtcvad.Vad(3)
SILENCE_THRESHOLD = 0.8

# =============================
# CONTROL FLAGS
# =============================
last_llm_time = 0
LLM_COOLDOWN = 2

last_api_call = 0
MIN_API_INTERVAL = 1.5

processing = False
MAX_HISTORY = 6

# =============================
# 🔊 NOISE REDUCTION
# =============================
def reduce_noise(audio_bytes):
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(audio_np) == 0:
            return audio_bytes

        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-6)

        threshold = 0.02
        audio_np[np.abs(audio_np) < threshold] = 0

        audio_np = (audio_np * 32767).astype(np.int16)

        return audio_np.tobytes()

    except Exception as e:
        print("Noise reduction error:", e)
        return audio_bytes
    
def is_clean_text(text):
    if not text:
        return False

    text = text.strip()

    # ❌ reject repeated characters
    if len(set(text)) < 5:
        return False

    # ❌ reject too many repeated syllables
    if any(text.count(x) > 10 for x in text.split()):
        return False

    # ❌ reject gibberish (no vowels)
    if not re.search(r'[aeiouఅఆఇఈఉఊఎఏఐఒఓఔ]', text.lower()):
        return False

    # ❌ reject too long nonsense
    if len(text) > 200:
        return False

    return True
# =============================
# 🔊 ENERGY CHECK
# =============================
def has_speech_energy(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio_np) == 0:
        return False
    energy = np.mean(np.abs(audio_np))
    return energy > 1200

# =============================
# RATE LIMIT SAFE CALL
# =============================
def safe_api_call(func, retries=5):
    delay = 1
    for _ in range(retries):
        try:
            throttle()
            return func()
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Rate limited. Retry in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise Exception("Max retries exceeded")

# =============================
# THROTTLE
# =============================
def throttle():
    global last_api_call
    now = time.time()

    if now - last_api_call < MIN_API_INTERVAL:
        time.sleep(MIN_API_INTERVAL - (now - last_api_call))

    last_api_call = time.time()

# =============================
# LANGUAGE
# =============================
def normalize_lang(lang):
    return lang if lang else "en-IN"

# =============================
# VALIDATION
# =============================
def is_valid_speech(text) :
    return text and len(text.strip()) >= 5

# =============================
# 🧠 INTENT DETECTION (MULTI-LANG)
# =============================
def enhance_user_input(text):
    t = text.strip().lower()

    brand_keywords = [
        "brand", "company", "name", "which brand",
        "బ్రాండ్", "పేరు", "ఏ కంపెనీ",
        "कौन सा ब्रांड", "नाम", "कंपनी"
    ]

    if any(k in t for k in brand_keywords):
        return "Give only fertilizer brand names. Crop context: " + text

    return text

# =============================
# 🧠 SIMILARITY CHECK
# =============================
def is_similar(a, b):
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return False
    return len(a_words & b_words) / len(a_words) > 0.7

# =============================
# STT
# =============================
def speech_to_text(audio_bytes):

    def call():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)

            with open(f.name, "rb") as af:
                res = client.speech_to_text.transcribe(
                    file=af,
                    model="saaras:v3",
                    mode="transcribe"
                )
        return res.transcript, res.language_code

    return safe_api_call(call)

# =============================
# VAD
# =============================
def vad_detect(frame):
    if len(frame) != FRAME_BYTES:
        return False
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

# =============================
# TTS
# =============================
def run_tts(text, lang):

    def call():
        res = client.text_to_speech.convert(
            text=text,
            target_language_code=lang,
            speaker="hitesh"
        )
        audio = b"".join([base64.b64decode(a) for a in res.audios])
        return base64.b64encode(audio).decode()

    return safe_api_call(call)

# =============================
# CHAT
# =============================
async def chat_once(history, lang, websocket):

    global last_llm_time

    if time.time() - last_llm_time < LLM_COOLDOWN:
        return

    last_llm_time = time.time()

    try:
        system_prompt = """
You are an expert farming assistant for Indian farmers.

STRICT RULES:
- Answer ONLY the current question
- DO NOT repeat previous answers
- DO NOT give same fertilizer advice unless asked again
- Give short, practical answers
- Max 120 words

SPECIAL:
If user asks for brands:
→ Give only brand names

NEVER:
- Repeat content
- Loop answers
- Give generic advice unnecessarily
"""

        messages = [{"role": "system", "content": system_prompt}]
        messages += history[-4:]

        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=300,
            presence_penalty=1.0,
            frequency_penalty=0.8,
            stream=True
        )

        full_text = ""

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue

            token = delta.content
            full_text += token

            await websocket.send_json({
                "type": "stream",
                "token": token
            })

        clean = full_text.strip()

        # ❌ weak response filter
        if len(clean.split()) < 3:
            print("⚠️ Weak response ignored")
            return

        if not clean.endswith((".", "!", "?")):
            clean += "."

        # 🚫 duplicate / similar response block
        if history and history[-1]["role"] == "assistant":
            if is_similar(clean, history[-1]["content"]):
                print("⚠️ Similar response blocked")
                return

        history.append({"role": "assistant", "content": clean})

        await websocket.send_json({
            "type": "final",
            "text": clean
        })

        audio = await asyncio.to_thread(run_tts, clean, lang)

        await websocket.send_json({
            "type": "audio",
            "audio": audio
        })

    except Exception as e:
        print("❌ STREAM ERROR:", e)

# =============================
# WEBSOCKET
# =============================
@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):

    global processing

    await websocket.accept()
    print("✅ CONNECTED")

    buffer = b""
    history = []
    last_voice_time = time.time()
    speaking = False

    try:
        while True:
            chunk = await websocket.receive_bytes()

            if not chunk:
                continue

            for i in range(0, len(chunk), FRAME_BYTES):
                frame = chunk[i:i+FRAME_BYTES]

                if len(frame) != FRAME_BYTES:
                    continue

                clean_frame = reduce_noise(frame)
                buffer += clean_frame

                if vad_detect(clean_frame):
                    speaking = True
                    last_voice_time = time.time()

            if speaking and (time.time() - last_voice_time > SILENCE_THRESHOLD):

                if processing:
                    continue

                if not has_speech_energy(buffer):
                    buffer = b""
                    speaking = False
                    continue

                text, lang = speech_to_text(buffer)
                lang = normalize_lang(lang)

                if is_valid_speech(text) and is_clean_text(text):
                    processing = True

                    enhanced = enhance_user_input(text)

                    await websocket.send_json({"type": "user", "text": text})

                    history.append({"role": "user", "content": enhanced})

                    if len(history) > MAX_HISTORY:
                        history = history[-MAX_HISTORY:]

                    await chat_once(history, lang, websocket)

                    processing = False

                buffer = b""
                speaking = False

            elif speaking and len(buffer) > SAMPLE_RATE * 12 and not processing:

                if not has_speech_energy(buffer):
                    buffer = b""
                    speaking = False
                    continue

                print("⚠️ FORCE PROCESS SAFE")

                text, lang = speech_to_text(buffer)
                lang = normalize_lang(lang)

                if is_valid_speech(text) and is_clean_text(text):
                    processing = True

                    enhanced = enhance_user_input(text)

                    await websocket.send_json({"type": "user", "text": text})

                    history.append({"role": "user", "content": enhanced})

                    if len(history) > MAX_HISTORY:
                        history = history[-MAX_HISTORY:]

                    await chat_once(history, lang, websocket)

                    processing = False

                buffer = b""
                speaking = False

            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        print("❌ DISCONNECTED")

    except Exception as e:
        print("❌ ERROR:", e)