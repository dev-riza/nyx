# -*- coding: utf-8 -*-
import threading
import time
import re
from tts import speak, is_russian

active_timers = {}

def words_to_number(text):
    numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "forty five": 45, "half": 30
    }
    for word, num in numbers.items():
        text = text.replace(word, str(num))
    return text

def parse_timer(text):
    text = words_to_number(text.lower())
    match = re.search(r'(\d+)\s*(second|minute|hour|sec|min|hr)s?', text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit in ["minute", "min"]:
            seconds = amount * 60
            label = f"{amount} minute"
        elif unit in ["hour", "hr"]:
            seconds = amount * 3600
            label = f"{amount} hour"
        else:
            seconds = amount
            label = f"{amount} second"
        return seconds, label

    match = re.search(r'(\d+)\s*(секунд|минут|час)', text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "минут":
            seconds = amount * 60
            label = f"{amount} минут"
        elif unit == "час":
            seconds = amount * 3600
            label = f"{amount} час"
        else:
            seconds = amount
            label = f"{amount} секунд"
        return seconds, label

    return None, None

def start_timer(seconds, label):
    timer_id = label
    end_time = time.time() + seconds
    active_timers[timer_id] = end_time

    def timer_thread():
        time.sleep(seconds)
        active_timers.pop(timer_id, None)
        msg = f"{label} timer is done!" if not is_russian(label) else f"Таймер на {label} завершён!"
        print(f"\n⏰ {msg}")
        speak(msg)

    threading.Thread(target=timer_thread, daemon=True).start()
    if is_russian(label):
        speak(f"Таймер на {label} установлен.")
    else:
        speak(f"Timer set for {label}.")

def check_timers():
    if not active_timers:
        speak("No active timers.")
        return
    responses = []
    for label, end_time in active_timers.items():
        remaining = int(end_time - time.time())
        if remaining > 0:
            mins = remaining // 60
            secs = remaining % 60
            if mins > 0:
                responses.append(f"{label} timer has {mins} minutes and {secs} seconds left.")
            else:
                responses.append(f"{label} timer has {secs} seconds left.")
    if responses:
        speak(" ".join(responses))
    else:
        speak("No active timers.")

def cancel_timers():
    if not active_timers:
        speak("No active timers to cancel.")
        return
    active_timers.clear()
    speak("All timers cancelled.")

def is_timer_request(text):
    timer_words = ["timer", "remind me in", "set a timer", "countdown",
                   "таймер", "напомни через", "отсчёт"]
    return any(word in text.lower() for word in timer_words)

def is_timer_check(text):
    check_words = ["how much time", "time left", "timer left", "remaining",
                   "how long", "сколько осталось", "сколько времени"]
    return any(word in text.lower() for word in check_words)

def is_timer_cancel(text):
    cancel_words = ["cancel", "delete", "remove", "clear",
                    "отмени", "удали", "убери"]
    text_lower = text.lower()
    return any(word in text_lower for word in cancel_words) and "timer" in text_lower