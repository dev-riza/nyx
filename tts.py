# -*- coding: utf-8 -*-
import subprocess
import time
from config import EN_VOICE, RU_VOICE, RUSSIAN_CHARS

is_speaking = False

def is_russian(text):
    russian_count = sum(1 for c in text if c in RUSSIAN_CHARS)
    return russian_count > len(text) * 0.1

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