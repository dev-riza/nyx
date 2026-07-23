# -*- coding: utf-8 -*-
import urllib.request
import urllib.parse
import json
from config import SEARCH_TRIGGERS

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