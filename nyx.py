# -*- coding: utf-8 -*-
import requests
import json
import subprocess
import time
import re
import wave
import audioop
import os
import threading
import webrtcvad

from config import (
    GROQ_API_KEY, GROQ_CHAT_URL, GROQ_WHISPER_URL,
    groq_headers
)
import tts
from tts import speak
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
        if "reSpeaker" in line or "XVF3800" in line or "Seeed" in line:
            match = re.search(r"card (\d+)", line)
            if match:
                return f"plughw:{match.group(1)},0"
    return "default"

def wait_until_quiet(buffer=0.3):
    while tts.is_speaking:
        time.sleep(0.1)
    time.sleep(buffer)

def spoke_since(start):
    # True if Nyx talked at any point after `start` (e.g. presence greeting mid-recording)
    return tts.is_speaking or tts.last_speech_end > start

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

WAKE_PATTERN = r'\b(nyx|nix|niks|nyks|nics|nixon|nick|nicks|mix|niece|phoenix|next|naked|никс|никсон)\b'

VISION_PATTERN = (
    r"\b(do|can|could|did) you see\b|\bwhat (do|can) you see\b|\blook(ing)? (at|around)\b"
    r"|\b(take|have) a look\b|\bwhat am i (wearing|holding|doing)\b|\bhow do i look\b"
    r"|\bwhat(\s+is|'s) (this|that|in front)\b|\bdescribe (the room|me|what)\b|\bin front of you\b"
    r"|видишь|посмотри|осмотрись|что на мне"
)

def contains_wake_word(text):
    return bool(re.search(WAKE_PATTERN, text))

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
    elif re.search(VISION_PATTERN, user_input):
        speak("Let me take a look...")
        result = analyze_scene(
            f"The user asked: \"{user_input}\". Answer their question based on this camera image "
            f"in 2-3 short spoken sentences. If they ask about themselves, describe the person in view."
        )
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
    wait_until_quiet(0.5)
    start = time.time()
    subprocess.run(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
         "-r", str(samplerate), "-c", "2", "-d", str(duration), "-q", "/tmp/audio.wav"],
        stderr=subprocess.DEVNULL
    )
    if spoke_since(start):
        return None
    return "/tmp/audio.wav"

def record_with_vad(max_duration=15, samplerate=16000, min_voiced_frames=10, min_rms=400):
    wait_until_quiet(0.3)
    start = time.time()
    vad = webrtcvad.Vad(3)
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000) * 2 * 2  # stereo * 16bit

    arecord = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(samplerate), "-c", "2", "-q"],
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
        if tts.is_speaking:
            break
        frame_mono = audioop.tomono(frame, 2, 0.5, 0.5)
        is_speech = vad.is_speech(frame_mono, samplerate)
        frames.append(frame_mono)
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
    if spoke_since(start) or voiced_frames < min_voiced_frames:
        return None
    # Quiet noise bursts get transcribed by Whisper as "thank you." etc.
    if audioop.rms(b"".join(frames), 2) < min_rms:
        return None

    with wave.open("/tmp/audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(b"".join(frames))

    return "/tmp/audio.wav"

def is_silent(filepath, threshold=500):
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
presence_wake = threading.Event()  # set when the greeting should start a conversation

def on_person_enter():
    wait_until_quiet(0.5)
    speak("Hey, welcome back.")
    presence_wake.set()

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

def conversation_loop(messages):
    consecutive_silent = 0
    while True:
        print("Listening...")
        filepath = record_with_vad(max_duration=15)

        if filepath is None:
            consecutive_silent += 1
            if consecutive_silent >= 2:
                print("No input detected, going back to sleep.")
                speak("I am here if you need me.")
                presence_wake.clear()  # ignore greetings that happened mid-conversation
                return
            continue

        consecutive_silent = 0

        try:
            user_input = transcribe(filepath)
        except Exception as e:
            print(f"Transcription error: {e}")
            continue

        if not user_input:
            continue

        # Skip if transcription is just noise/punctuation
        if len(user_input.strip('.?,! ')) < 3:
            continue

        process_command(user_input, messages)

print("Nyx is sleeping... say 'Nyx' to wake him up.")

try:
    while True:
        if presence_wake.is_set():
            presence_wake.clear()
            conversation_loop(messages)
            continue

        if tts.is_speaking:
            time.sleep(0.1)
            continue

        filepath = record_audio_fixed(duration=3)
        check_in_on_mood()

        if filepath is None or is_silent(filepath):
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

        command = re.sub(WAKE_PATTERN, '', text)
        command = re.sub(r'^[\W_]*(hey|ok|okay|hi|hello|хей|привет)\b', '', command)
        command = command.strip(' ,.!?;:')
        if len(command) < 3:
            command = ""

        if command:
            process_command(command, messages)
        else:
            speak("Yes, I am here.")

        presence_wake.clear()  # already in a conversation
        conversation_loop(messages)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    presence_monitor.stop()
    emotion_monitor.stop()
    update_memory_from_session(messages, memory)
    print("Memory saved.")