# -*- coding: utf-8 -*-
import os

GROQ_API_KEY = "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"
CEREBRAS_API_KEY = "csk-pev8yxx8wv3mmcd5crj22ntwmpyx39jy48mpvrhx2rhpn6cc"

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

MEMORY_FILE = "/home/riza/ai-assistant/nyx_memory.json"
EN_VOICE = "/home/riza/piper-voices/en_US-lessac-medium.onnx"
RU_VOICE = "/home/riza/piper-voices/ru_RU-ruslan-medium.onnx"

RUSSIAN_CHARS = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

SEARCH_TRIGGERS = [
    "what time", "current time", "weather", "news today", "latest news",
    "temperature", "forecast", "search for", "look up", "what happened",
    "who won", "live score", "current price", "stock price",
    "который час", "погода", "новости", "температура", "найди", "поищи"
]

cerebras_headers = {
    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
    "Content-Type": "application/json"
}