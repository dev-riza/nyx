# -*- coding: utf-8 -*-
import json
import re
import requests
from config import MEMORY_FILE, CEREBRAS_URL, cerebras_headers

def load_memory():
    import os
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

def update_memory_from_session(messages, memory):
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