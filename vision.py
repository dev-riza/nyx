# -*- coding: utf-8 -*-
import base64, requests, io, struct
import numpy as np
from PIL import Image
from config import GROQ_API_KEY

VISION_MODEL = 'qwen/qwen3.6-27b'

def capture_frame():
    try:
        import usb.core, usb.util
    except ImportError:
        return None

    def send_cmd(dev, opcode, payload=b''):
        size = len(payload) // 2
        header = struct.pack('<HHHH', 0x4d47, size, opcode, 0)
        dev.ctrl_transfer(0x40, 0, 0, 0, header + payload, timeout=3000)
        return dev.ctrl_transfer(0xC0, 0, 0, 0, 512, timeout=3000).tobytes()

    def set_param(dev, param, value):
        return send_cmd(dev, 3, struct.pack('<HH', param, value))

    dev = usb.core.find(idVendor=0x2bc5, idProduct=0x0401)
    if dev is None:
        return None

    for i in range(3):
        try:
            if dev.is_kernel_driver_active(i): dev.detach_kernel_driver(i)
        except: pass

    dev.set_configuration()
    usb.util.claim_interface(dev, 0)
    send_cmd(dev, 6, struct.pack('<H', 1))
    set_param(dev, 16, 15)
    set_param(dev, 5, 1)

    for _ in range(300):
        try: dev.read(0x82, 3072, timeout=50)
        except: pass

    raw_packets = []
    for _ in range(3000):
        try:
            pkt = bytes(dev.read(0x82, 3072, timeout=200))
            if len(pkt) == 3072: raw_packets.append(pkt)
            if len(raw_packets) >= 1200: break
        except: continue

    frames = []
    current = []
    for pkt in raw_packets:
        magic, ntype = struct.unpack_from('<HH', pkt, 0)
        if magic != 0x4252: continue
        if ntype == 0x8100:
            if len(current) >= 190: frames.append(current)
            current = [pkt]
        elif ntype == 0x8200 and current:
            current.append(pkt)

    usb.util.release_interface(dev, 0)

    if not frames:
        return None

    frame = frames[0]
    pixel_data = bytearray()
    for pkt in frame:
        pixel_data.extend(pkt)

    data = np.frombuffer(bytes(pixel_data[:614400]), dtype=np.uint8).reshape(-1, 4)
    u  = data[:,0].astype(float) - 128
    y0 = data[:,1].astype(float)
    v  = data[:,2].astype(float) - 128
    y1 = data[:,3].astype(float)

    r0 = np.clip(y0 + 1.402*v, 0, 255).astype(np.uint8)
    g0 = np.clip(y0 - 0.344*u - 0.714*v, 0, 255).astype(np.uint8)
    b0 = np.clip(y0 + 1.772*u, 0, 255).astype(np.uint8)
    r1 = np.clip(y1 + 1.402*v, 0, 255).astype(np.uint8)
    g1 = np.clip(y1 - 0.344*u - 0.714*v, 0, 255).astype(np.uint8)
    b1 = np.clip(y1 + 1.772*u, 0, 255).astype(np.uint8)

    rgb = np.zeros((len(data)*2, 3), dtype=np.uint8)
    rgb[0::2] = np.stack([r0,g0,b0], axis=1)
    rgb[1::2] = np.stack([r1,g1,b1], axis=1)
    w = 642
    h = len(rgb) // w
    img_arr = np.roll(rgb[:w*h].reshape(h, w, 3), 400, axis=1)
    return img_arr

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