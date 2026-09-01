# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
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

groq_headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}