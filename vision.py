# -*- coding: utf-8 -*-
import base64, requests, io, time
import numpy as np
from PIL import Image
from config import GROQ_API_KEY
from camera.astra import get_color_frame

VISION_MODEL = 'qwen/qwen3.8-27b'

MAX_BRIGHTNESS_FACTOR = 1.7
GAMMA = 0.85

ACCURACY_NOTE = (
    " Only describe things that are clearly visible in the image. "
    "If something is unclear or you are not sure, say so instead of guessing."
)


def _adjust_brightness(img_arr):
    arr = img_arr.astype(np.float32) / 255.0
    arr = np.power(arr, GAMMA)
    # Brighten only until the brightest areas reach near-white; a fixed boost
    # blows out white walls/doors and the vision model hallucinates shapes there
    bright = np.percentile(arr, 99)
    factor = min(MAX_BRIGHTNESS_FACTOR, 0.97 / bright) if bright > 0 else 1.0
    arr = np.clip(arr * max(factor, 1.0), 0, 1)
    return (arr * 255).astype(np.uint8)


def capture_frame():
    img_arr = get_color_frame()
    if img_arr is None:
        return None
    return _adjust_brightness(img_arr)


def _ask_vision(img_arr, prompt, max_tokens=200, attempts=2):
    """Send an image + prompt to the Groq vision model. Returns the answer
    text, or None if every attempt failed. Groq sometimes stalls or returns
    "over capacity" for a while, so one retry usually gets through."""
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    for attempt in range(attempts):
        if attempt:
            time.sleep(2)
        try:
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
                json={
                    'model': VISION_MODEL,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                            {'type': 'text', 'text': prompt}
                        ]
                    }],
                    'max_tokens': max_tokens,
                    'temperature': 0.2
                },
                timeout=10
            )
            result = response.json()
        except Exception as e:
            print(f"Vision request error (attempt {attempt + 1}): {e}")
            continue

        if 'choices' in result:
            content = result['choices'][0]['message']['content'] or ''
            if '</think>' in content:
                content = content.split('</think>')[-1]
            return content.strip()
        print(f"Vision API error (attempt {attempt + 1}): {result}")
    return None


def analyze_scene(prompt="Describe what you see in this scene briefly in 2-3 sentences."):
    img_arr = capture_frame()
    if img_arr is None:
        return "Camera not available."
    answer = _ask_vision(img_arr, prompt + ACCURACY_NOTE, max_tokens=200)
    return answer or "Sorry, my vision service isn't responding right now. Try again in a moment."


def analyze_emotion():
    """Capture a frame and read apparent emotional state / body language.
    Returns a short descriptive phrase, or None if camera unavailable.
    This is a soft observation from a vision model, not a diagnosis --
    treat it as a hint, never state it as fact to the user."""
    img_arr = capture_frame()
    if img_arr is None:
        return None

    prompt = (
        "Look at the person's face and posture in this image. In one short "
        "phrase (under 12 words), describe their apparent mood or body "
        "language -- e.g. 'relaxed and smiling', 'looks tired', 'seems "
        "focused', 'appears a bit down'. If no person is clearly visible, "
        "say 'no person visible'. Be tentative, not certain -- this is a "
        "guess from a single image, not a diagnosis."
    )
    # Background check -- one attempt is enough, it runs again next poll
    return _ask_vision(img_arr, prompt, max_tokens=300, attempts=1)
