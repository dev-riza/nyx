# -*- coding: utf-8 -*-
import base64, requests, io
import numpy as np
from PIL import Image
from config import GROQ_API_KEY
from camera.astra import get_color_frame

VISION_MODEL = 'qwen/qwen3.6-27b'

BRIGHTNESS_FACTOR = 1.7
GAMMA = 0.85


def _adjust_brightness(img_arr):
    arr = img_arr.astype(np.float32) / 255.0
    arr = np.power(arr, GAMMA)
    arr = arr * BRIGHTNESS_FACTOR
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def capture_frame():
    img_arr = get_color_frame()
    if img_arr is None:
        return None
    return _adjust_brightness(img_arr)


def analyze_scene(prompt="Describe what you see in this scene briefly in 2-3 sentences."):
    img_arr = capture_frame()
    if img_arr is None:
        return "Camera not available."

    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

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
                'max_tokens': 200
            },
            timeout=15
        )
        result = response.json()
        if 'choices' in result:
            content = result['choices'][0]['message']['content']
            if '</think>' in content:
                content = content.split('</think>')[-1].strip()
            return content
        return "Could not analyze scene."
    except Exception as e:
        return f"Vision error: {e}"
