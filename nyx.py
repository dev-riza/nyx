# -*- coding: utf-8 -*-
import requests
import json
import subprocess
import time
import re
import wave
import audioop
import os
import urllib.request
import urllib.parse
import webrtcvad

GROQ_API_KEY = "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"
CEREBRAS_API_KEY = "csk-pev8yxx8wv3mmcd5crj22ntwmpyx39jy48mpvrhx2rhpn6cc"

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

cerebras_headers = {
    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
    "Content-Type": "application/json"
}

MEMORY_FILE = "/home/riza/ai-assistant/nyx_memory.json"
EN_VOICE = "/home/riza/piper-voices/en_US-lessac-medium.onnx"
RU_VOICE = "/home/riza/piper-voices/ru_RU-ruslan-medium.onnx"

SEARCH_TRIGGERS = [
    "what time", "current time", "weather", "news today", "latest news",
    "temperature", "forecast", "search for", "look up", "what happened",
    "who won", "live score", "current price", "stock price",
    "который час", "погода", "новости", "температура", "найди", "поищи"
]

RUSSIAN_CHARS = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

def is_russian(text):
    russian_count = sum(1 for c in text if c in RUSSIAN_CHARS)
    return russian_count > len(text) * 0.3

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def build_system_prompt(memory):
    base = """You are Nyx, a highly intelligent AI assistant and companion — think Jarvis from Iron Man but with your own identity.
You are confident, composed, and sharp. You have a dry wit and occasionally slip in subtle sarcasm or clever remarks without overdoing it.
You speak casually but intelligently — like a brilliant friend who happens to know everything.
You are loyal and genuinely care about the user, but you don't sugarcoat things.
Keep answers short — 2 to 4 sentences max. No fluff, no filler words.
Never say things like "certainly!", "of course!", "great question!" or "I'd be happy to help!" — that's not you.
Occasionally show personality — a dry comment, a witty observation, a subtle joke — but keep it natural, not forced.
Do NOT ask questions back unless the user specifically asks for your opinion or advice.
You have access to web search results — when provided, use them to give accurate, current answers.
You ONLY speak English and Russian. If the user speaks English, respond in English. If the user speaks Russian, respond in Russian. If the user speaks any other language, politely tell them in English and Russian that you only support these two languages.
CRITICAL: Only reference things you actually know about the user from memory. Never invent or assume facts."""
    if memory:
        facts = "\n".join(f"- {k}: {v}" for k, v in memory.items())
        base += f"\n\nWhat you know about the user (only reference these, nothing else):\n{facts}"
    return base

def extract_memory(conversation, existing_memory):
    prompt = f"""You are a memory manager for a personal AI assistant named Nyx. Your job is to maintain an accurate, clean profile of the user based on conversations.

CURRENT MEMORY:
{json.dumps(existing_memory, indent=2)}

NEW CONVERSATION:
{conversation}

RULES:
1. ONLY add facts that the user explicitly stated themselves. Never infer or assume.
2. If the user CORRECTS or CONTRADICTS something in memory, UPDATE or REMOVE it immediately.
3. If the user says they DISLIKE or HATE something, remove it from positive categories and add it to a "dislikes" field.
4. Keep memory clean and minimal — only truly important personal facts.
5. Remove any field that was based on assumption, not something the user actually said.
6. Valid fields: name, age, gender, occupation, hobbies, dislikes, favourite_food, favourite_music, pets, family, personality, goals, other_facts.

Return ONLY a valid JSON object. No extra text, no markdown, no explanation."""

    try:
        response = requests.post(
            CEREBRAS_URL,
            headers=cerebras_headers,
            json={
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "reasoning_effort": "low"
            }
        )
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"Memory extraction error: {e}")
        return existing_memory

def get_mic_device():
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "USB Audio" in line or "USB" in line:
            match = re.search(r"card (\d+)", line)
            if match:
                return f"plughw:{match.group(1)},0"
    return "plughw:1,0"

def set_volume():
    os.system("amixer -c 0 sset 'Headphone' 90% > /dev/null 2>&1")
    os.system("amixer -c 0 sset 'PCM' 90% > /dev/null 2>&1")
    print("Volume set to 90%.")

def web_search(query):
    print(f"Searching: {query}")
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
        answer = data.get("AbstractText", "") or data.get("Answer", "")
        if not answer:
            topics = data.get("RelatedTopics", [])
            if topics and "Text" in topics[0]:
                answer = topics[0]["Text"]
        return answer if answer else "No results found."
    except Exception as e:
        return f"Search failed: {e}"

def needs_search(text):
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in SEARCH_TRIGGERS)

def transcribe(filepath):
    with open(filepath, "rb") as f:
        response = requests.post(
            GROQ_WHISPER_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": "whisper-large-v3-turbo"}
        )
    result = response.json()
    return result.get("text", "").strip().lower()

def contains_wake_word(text):
    pattern = r'\b(nyx|nix|никс)\b'
    return bool(re.search(pattern, text))

def speak(text):
    global is_speaking
    is_speaking = True
    print("Nyx:", text)
    voice = RU_VOICE if is_russian(text) else EN_VOICE
    try:
        piper = subprocess.Popen(
            ["piper", "--model", voice, "--output_raw", "--sentence-silence", "0.0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", "hw:0,0", "-r", "22050", "-f", "S16_LE", "-c", "1"],
            stdin=piper.stdout,
            stderr=subprocess.DEVNULL
        )
        piper.stdin.write((text + "\n").encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
    except Exception as e:
        print(f"Speak error: {e}")
    wait_time = max(0.5, len(text.split()) * 0.1)
    time.sleep(wait_time)
    is_speaking = False

def ask_ai(user_input):
    messages.append({"role": "user", "content": user_input})

    search_context = ""
    if needs_search(user_input):
        search_result = web_search(user_input)
        if search_result and "failed" not in search_result and "No results" not in search_result:
            search_context = f"\n\nWeb search result for '{user_input}':\n{search_result}\n\nUse this to answer accurately."
            print(f"Search result: {search_result}")

    send_messages = messages.copy()
    if search_context:
        send_messages[-1] = {
            "role": "user",
            "content": user_input + search_context
        }

    try:
        data = {
            "model": "gpt-oss-120b",
            "messages": send_messages,
            "max_tokens": 250,
            "stream": False,
            "reasoning_effort": "low"
        }

        print("Thinking...")
        response = requests.post(CEREBRAS_URL, headers=cerebras_headers, json=data, timeout=15)
        result = response.json()

        if "choices" not in result:
            print(f"API error: {result}")
            speak("Sorry, something went wrong.")
            messages.pop()
            return

        full_reply = result["choices"][0]["message"].get("content", "")

        if full_reply.strip():
            speak(full_reply.strip())

        messages.append({"role": "assistant", "content": full_reply})

    except Exception as e:
        print(f"ask_ai error: {e}")
        speak("Sorry, I ran into an error.")
        messages.pop()

def record_audio_fixed(duration=3, samplerate=16000):
    subprocess.run(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
         "-r", str(samplerate), "-c", "1", "-d", str(duration), "-q", "/tmp/audio.wav"],
        stderr=subprocess.DEVNULL
    )
    return "/tmp/audio.wav"

def record_with_vad(max_duration=15, samplerate=16000):
    vad = webrtcvad.Vad(2)
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000) * 2

    arecord = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
         "-r", str(samplerate), "-c", "1", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    frames = []
    silent_frames = 0
    voiced_frames = 0
    max_silent_frames = int(1200 / frame_duration)
    max_frames = int(max_duration * 1000 / frame_duration)
    started_talking = False

    for _ in range(max_frames):
        frame = arecord.stdout.read(frame_size)
        if len(frame) < frame_size:
            break
        is_speech = vad.is_speech(frame, samplerate)
        frames.append(frame)
        if is_speech:
            voiced_frames += 1
            silent_frames = 0
            started_talking = True
        else:
            if started_talking:
                silent_frames += 1
                if silent_frames > max_silent_frames:
                    break

    arecord.terminate()

    if voiced_frames < 3:
        return None

    with wave.open("/tmp/audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(b"".join(frames))

    return "/tmp/audio.wav"

def is_silent(filepath, threshold=300):
    if not os.path.exists(filepath):
        return True
    with wave.open(filepath, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        if len(frames) == 0:
            return True
        rms = audioop.rms(frames, wf.getsampwidth())
        return rms < threshold

def update_memory_from_session():
    if len(messages) <= 1:
        return
    transcript = ""
    for m in messages[1:]:
        if m["role"] in ["user", "assistant"] and isinstance(m.get("content"), str):
            role = "User" if m["role"] == "user" else "Nyx"
            transcript += f"{role}: {m['content']}\n"
    if not transcript.strip():
        return
    print("Updating memory from this session...")
    updated = extract_memory(transcript, memory)
    save_memory(updated)
    print(f"Memory saved: {updated}")

memory = load_memory()
MIC_DEVICE = get_mic_device()
print(f"Using mic device: {MIC_DEVICE}")
set_volume()

messages = [
    {"role": "system", "content": build_system_prompt(memory)}
]

is_speaking = False

print("Nyx is sleeping... say 'Nyx' to wake her up.")

try:
    while True:
        if is_speaking:
            time.sleep(0.1)
            continue

        record_audio_fixed(duration=3)

        if is_silent("/tmp/audio.wav"):
            continue

        try:
            text = transcribe("/tmp/audio.wav")
        except Exception as e:
            print(f"Transcription error: {e}")
            continue

        if not text:
            continue

        print(f"Heard: '{text}'")

        if not (contains_wake_word(text) and len(text.split()) <= 4):
            continue

        speak("Yes, I am here.")

        consecutive_silent = 0
        while True:
            print("Listening...")
            filepath = record_with_vad(max_duration=15)

            if filepath is None:
                consecutive_silent += 1
                if consecutive_silent >= 2:
                    print("No input detected, going back to sleep.")
                    speak("I am here if you need me.")
                    break
                continue

            consecutive_silent = 0

            try:
                user_input = transcribe(filepath)
            except Exception as e:
                print(f"Transcription error: {e}")
                continue

            if not user_input:
                continue

            print("You:", user_input)

            if any(word in user_input for word in ["stop", "goodbye", "стоп", "пока"]):
                speak("Goodbye.")
                raise KeyboardInterrupt

            ask_ai(user_input)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    update_memory_from_session()
    print("Memory saved.")
