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
    GROQ_API_KEY, GROQ_CHAT_URL, GROQ_WHISPER_URL,
    groq_headers
)
from tts import speak, is_speaking
from timer import start_timer, check_timers, cancel_timers, is_timer_request, is_timer_check, is_timer_cancel, parse_timer
from memory import load_memory, build_system_prompt, update_memory_from_session
from search import web_search, needs_search
from googleapi import read_emails, send_email, get_calendar_events, search_drive
from vision import analyze_scene, capture_frame
from detect import detect_objects, describe_detections, EmotionMonitor
from body_language import PresenceMonitor, analyze_body

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

def clean_email(text):
    text = text.lower()
    text = text.replace(" at ", "@")
    text = text.replace(" dot ", ".")
    text = text.replace("ə", "a")
    text = text.replace("ä", "a")
    text = text.replace("dotcom", ".com")
    text = text.replace("dot com", ".com")
    text = text.replace("yandexcom", "yandex.com")
    text = text.replace("gmailcom", "gmail.com")
    text = text.replace("hotmailcom", "hotmail.com")
    for prefix in ["to ", "send to ", "email to ", "send it to "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace(" ", "")
    return text.strip()

def contains_wake_word(text):
    pattern = r'\b(nyx|nix|nick|nicks|mix|niece|phoenix|next|naked|никс)\b'
    return bool(re.search(pattern, text))

def check_in_on_mood():
    latest = emotion_monitor.latest
    if latest and not emotion_monitor._seems_concerning(latest):
        emotion_monitor.reset_check_in()
        return
    if emotion_monitor.should_check_in():
        speak("Hey, is everything okay? You seem a little off today.")
        emotion_monitor.mark_checked_in()

def process_command(user_input, messages):
    print(f"You: {user_input}")

    if any(word in user_input for word in ["stop", "goodbye", "стоп", "пока"]):
        speak("Goodbye.")
        raise KeyboardInterrupt

    if any(word in user_input for word in ["read my email", "read my emails", "check my email", "check my emails", "any emails", "unread emails", "latest email", "new email", "почта", "письма"]):
        result = read_emails()
        speak(result)
    elif any(word in user_input for word in ["my calendar", "my schedule", "upcoming events", "what's on", "календарь", "расписание"]):
        result = get_calendar_events()
        speak(result)
    elif any(word in user_input for word in ["search drive", "find file", "find in drive", "найди файл"]):
        speak("What should I search for?")
        time.sleep(0.5)
        filepath = record_with_vad(max_duration=10)
        if filepath:
            query = transcribe(filepath)
            result = search_drive(query)
            speak(result)
    elif any(word in user_input for word in ["send email", "send an email", "write an email", "email to", "отправь письмо", "напиши письмо"]):
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
    elif any(word in user_input for word in ["what do you see", "look around", "what's in front", "describe the room", "что видишь", "осмотрись"]):
        speak("Let me take a look...")
        result = analyze_scene()
        speak(result)
    elif any(word in user_input for word in ["who's here", "who is here", "who's in the room", "is anyone here", "кто здесь"]):
        img = capture_frame()
        if img is not None:
            result = analyze_body(img)
            speak(result['summary'])
        else:
            speak("Camera not available.")
    elif any(word in user_input for word in ["my posture", "how am i sitting", "моя осанка"]):
        img = capture_frame()
        if img is not None:
            result = analyze_body(img)
            speak(result['summary'])
        else:
            speak("Camera not available.")
    else:
        ask_ai(user_input, messages)

def ask_ai(user_input, messages):
    messages.append({"role": "user", "content": user_input})

    search_context = ""
    if needs_search(user_input):
        search_result = web_search(user_input)
        if search_result and "failed" not in search_result and "No results" not in search_result:
            search_context = f"\n\nWeb search result for '{user_input}':\n{search_result}\n\nUse this to answer accurately."
            print(f"Search result: {search_result}")

    send_messages = messages.copy()
    extra_context = search_context

    mood = emotion_monitor.latest
    if mood and emotion_monitor._seems_concerning(mood):
        extra_context += (
            f"\n\n[The person may currently seem {mood.lower()} based on a "
            f"casual visual read -- this is just a hint, not certain. If it "
            f"feels natural, you can be a bit gentler or warmer in tone, "
            f"but don't mention that you're reading their mood or comment "
            f"on their appearance directly.]"
        )

    if extra_context:
        send_messages[-1] = {
            "role": "user",
            "content": user_input + extra_context
        }

    try:
        data = {
            "model": "openai/gpt-oss-120b",
            "messages": send_messages,
            "max_tokens": 250,
            "stream": False
        }

        print("Thinking...")
        response = requests.post(GROQ_CHAT_URL, headers=groq_headers, json=data, timeout=15)
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
    speak("Who should I send it to? Please say the email address clearly, saying at for the @ symbol and dot for periods.")
    time.sleep(0.5)
    filepath = record_with_vad(max_duration=15)
    if not filepath:
        speak("I didn't catch that.")
        return
    to_raw = transcribe(filepath)
    to = clean_email(to_raw)
    print(f"To: {to}")

    speak(f"I heard {to_raw}. The email address is {to}. Is that correct? Say yes or no.")
    time.sleep(0.5)
    filepath = record_with_vad(max_duration=6)
    if filepath:
        confirm = transcribe(filepath)
        confirm_clean = confirm.replace(" ", "").lower()
        if not any(word in confirm_clean for word in ["yes", "yeah", "yep", "correct", "sure", "да", "конечно"]):
            speak("Let's try again. Please say the email address slowly and clearly.")
            time.sleep(0.5)
            filepath = record_with_vad(max_duration=15)
            if not filepath:
                speak("I didn't catch that.")
                return
            to_raw = transcribe(filepath)
            to = clean_email(to_raw)
            print(f"To (retry): {to}")

    speak("What should the subject be?")
    time.sleep(0.5)
    filepath = record_with_vad(max_duration=10)
    if not filepath:
        speak("I didn't catch that.")
        return
    subject = transcribe(filepath)
    print(f"Subject: {subject}")

    speak("What should I say in the email?")
    time.sleep(0.5)
    filepath = record_with_vad(max_duration=15)
    if not filepath:
        speak("I didn't catch that.")
        return
    body = transcribe(filepath)
    print(f"Body: {body}")

    speak(f"Ready to send to {to} with subject {subject}. Shall I send it? Say yes or no.")
    time.sleep(0.5)
    filepath = record_with_vad(max_duration=8)
    if not filepath:
        speak("No response, email cancelled.")
        return

    confirm = transcribe(filepath)
    print(f"Confirmation heard: {confirm}")
    confirm_clean = confirm.replace(" ", "").lower()

    if any(word in confirm_clean for word in ["yes", "yeah", "yep", "correct", "sure", "да", "конечно"]):
        try:
            success = send_email(to, subject, body)
            if success:
                speak("Email sent successfully.")
            else:
                speak("Sorry, I couldn't send the email.")
        except Exception as e:
            print(f"Send error: {e}")
            speak("Sorry, something went wrong sending the email.")
    elif any(word in confirm_clean for word in ["no", "nope", "нет"]):
        speak("Email cancelled.")
    else:
        speak("I didn't catch that, email cancelled.")

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

# Startup
memory = load_memory()
MIC_DEVICE = get_mic_device()
print(f"Using mic device: {MIC_DEVICE}")
set_volume()

# Presence monitor - body language based
def on_person_enter():
    speak("Hey, welcome back.")

def on_person_leave():
    pass

presence_monitor = PresenceMonitor(
    on_enter=on_person_enter,
    on_leave=on_person_leave,
    check_interval=5.0
)
presence_monitor.start(capture_frame)
print("Presence monitor started.")

# Emotion monitor
emotion_monitor = EmotionMonitor(poll_interval=60)
emotion_monitor.start()
print("Emotion monitor started.")

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
        check_in_on_mood()

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

        command = re.sub(r'\b(nyx|nix|nick|nicks|mix|niece|phoenix|next|naked|никс)\b', '', text).strip()
        command = re.sub(r'^(hey|ok|okay|hi|hello)\s*', '', command).strip()
        command = re.sub(r'^[,.\s]+', '', command).strip()

        if command:
            process_command(command, messages)
        else:
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

            process_command(user_input, messages)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    presence_monitor.stop()
    emotion_monitor.stop()
    update_memory_from_session(messages, memory)
    print("Memory saved.")