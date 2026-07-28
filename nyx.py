# -*- coding: utf-8 -*-
import requests
import json
import subprocess
import time
import re
import wave
import audioop
import os
import webrtcvad

from config import (
    GROQ_API_KEY, CEREBRAS_URL, GROQ_WHISPER_URL,
    cerebras_headers
)
from tts import speak, is_speaking
from timer import start_timer, check_timers, cancel_timers, is_timer_request, is_timer_check, is_timer_cancel, parse_timer
from memory import load_memory, build_system_prompt, update_memory_from_session
from search import web_search, needs_search
from gmail import read_emails, send_email, get_calendar_events

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
    pattern = r'\b(nyx|nix|nick|nicks|mix|niece|phoenix|next|naked|никс)\b'
    return bool(re.search(pattern, text))

def ask_ai(user_input, messages):
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

def handle_send_email(messages):
    speak("Who should I send it to?")
    filepath = record_with_vad(max_duration=10)
    if not filepath:
        speak("I didn't catch that.")
        return
    to = transcribe(filepath)
    print(f"To: {to}")

    speak("What should the subject be?")
    filepath = record_with_vad(max_duration=10)
    if not filepath:
        speak("I didn't catch that.")
        return
    subject = transcribe(filepath)
    print(f"Subject: {subject}")

    speak("What should I say in the email?")
    filepath = record_with_vad(max_duration=15)
    if not filepath:
        speak("I didn't catch that.")
        return
    body = transcribe(filepath)
    print(f"Body: {body}")

    speak(f"Sending email to {to} with subject {subject}. Is that correct? Say yes or no.")
    filepath = record_with_vad(max_duration=5)
    if filepath:
        confirm = transcribe(filepath)
        if "yes" in confirm or "да" in confirm:
            success = send_email(to, subject, body)
            if success:
                speak("Email sent successfully.")
            else:
                speak("Sorry, I couldn't send the email.")
        else:
            speak("Email cancelled.")

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

memory = load_memory()
MIC_DEVICE = get_mic_device()
print(f"Using mic device: {MIC_DEVICE}")
set_volume()

messages = [
    {"role": "system", "content": build_system_prompt(memory)}
]

print("Nyx is sleeping... say 'Nyx' to wake her up.")

try:
    while True:
        import tts as tts_module
        if tts_module.is_speaking:
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

        if not contains_wake_word(text):
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

            if any(word in user_input for word in ["read my emails", "check my emails", "any emails", "unread emails", "почта", "письма"]):
                result = read_emails()
                speak(result)
            elif any(word in user_input for word in ["my calendar", "my schedule", "upcoming events", "what's on", "календарь", "расписание"]):
                result = get_calendar_events()
                speak(result)
            elif "send email" in user_input or "отправь письмо" in user_input:
                handle_send_email(messages)
            elif is_timer_check(user_input):
                check_timers()
            elif is_timer_cancel(user_input):
                cancel_timers()
            elif is_timer_request(user_input):
                seconds, label = parse_timer(user_input)
                if seconds:
                    start_timer(seconds, label)
                else:
                    speak("I didn't catch the time. Try saying something like set a timer for 5 minutes.")
            else:
                ask_ai(user_input, messages)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    update_memory_from_session(messages, memory)
    print("Memory saved.")