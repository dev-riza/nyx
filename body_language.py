# -*- coding: utf-8 -*-
import numpy as np
import os
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'movenet.tflite')

# Keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

_interp = None

def _get_interpreter():
    global _interp
    if _interp is None:
        _interp = Interpreter(model_path=MODEL_PATH)
        _interp.allocate_tensors()
    return _interp

def _run_pose(img_arr):
    interp = _get_interpreter()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    
    # Resize to 257x257
    img = Image.fromarray(img_arr).transpose(Image.FLIP_LEFT_RIGHT).resize((257, 257))
    img_np = (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]    
    interp.set_tensor(inp[0]['index'], img_np)
    interp.invoke()
    
    # Output shape: [1, 9, 9, 17] - heatmaps
    # We need to find peak locations
    heatmaps = interp.get_tensor(out[0]['index'])[0]  # (9, 9, 17)
    
    keypoints = []
    for kp_idx in range(17):
        heatmap = heatmaps[:, :, kp_idx]
        flat_idx = np.argmax(heatmap)
        y_idx, x_idx = divmod(flat_idx, 9)
        confidence = heatmap[y_idx, x_idx]
        # Normalize to 0-1
        y = y_idx / 8.0
        x = x_idx / 8.0
        keypoints.append((y, x, float(confidence)))
    
    return keypoints

def analyze_body(img_arr):
    """
    Takes a numpy RGB image array.
    Returns a dict with presence, posture, and gesture info.
    """
    results = {}
    
    try:
        kp = _run_pose(img_arr)
    except Exception as e:
        return {'present': False, 'summary': f'Error: {e}'}
    
    # Check confidence - is anyone there?
    nose_conf = kp[NOSE][2]
    shoulder_conf = (kp[LEFT_SHOULDER][2] + kp[RIGHT_SHOULDER][2]) / 2
    
    if nose_conf < 0.5 and shoulder_conf < 0.5:
        results['present'] = False
        results['summary'] = 'No person detected'
        return results
    
    results['present'] = True
    
    # Head tilt - ear heights
    left_ear_y = kp[LEFT_EAR][0]
    right_ear_y = kp[RIGHT_EAR][0]
    head_tilt = left_ear_y - right_ear_y
    if abs(head_tilt) < 0.05:
        results['head'] = 'level'
    elif head_tilt > 0:
        results['head'] = 'tilted right'
    else:
        results['head'] = 'tilted left'
    
    # Shoulder level
    left_shoulder_y = kp[LEFT_SHOULDER][0]
    right_shoulder_y = kp[RIGHT_SHOULDER][0]
    shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
    results['shoulders'] = 'level' if shoulder_diff < 0.05 else 'uneven'
    
    # Posture lean - nose vs shoulder midpoint x
    nose_x = kp[NOSE][1]
    shoulder_mid_x = (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2
    lean = nose_x - shoulder_mid_x
    if abs(lean) < 0.05:
        results['posture'] = 'centered'
    elif lean > 0:
        results['posture'] = 'leaning left'
    else:
        results['posture'] = 'leaning right'
        
    # Arms raised
    left_wrist_y = kp[LEFT_WRIST][0]
    right_wrist_y = kp[RIGHT_WRIST][0]
    left_raised = left_wrist_y < left_shoulder_y - 0.05
    right_raised = right_wrist_y < right_shoulder_y - 0.05
    
    if left_raised and right_raised:
        results['arms'] = 'both raised'
    elif left_raised:
        results['arms'] = 'left arm raised'
    elif right_raised:
        results['arms'] = 'right arm raised'
    else:
        results['arms'] = 'at sides'
    
    # Slouching - shoulders vs hips
    hip_y = (kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    results['slouching'] = shoulder_y > hip_y * 0.7
    
    # Build summary
    parts = []
    if results['arms'] != 'at sides':
        parts.append(results['arms'])
    if results['posture'] != 'centered':
        parts.append(results['posture'])
    if results['head'] != 'level':
        parts.append('head ' + results['head'])
    if results.get('slouching'):
        parts.append('slouching')
    
    results['summary'] = 'Person present. ' + (', '.join(parts) if parts else 'Relaxed posture.')
    results['keypoints'] = kp
    
    return results


def quick_presence(img_arr):
    """Fast check - is someone in frame?"""
    try:
        kp = _run_pose(img_arr)
        nose_conf = kp[NOSE][2]
        shoulder_conf = (kp[LEFT_SHOULDER][2] + kp[RIGHT_SHOULDER][2]) / 2
        return nose_conf > 0.5 or shoulder_conf > 0.5
    except:
        return False


import threading
import time

class PresenceMonitor:
    def __init__(self, on_enter=None, on_leave=None, check_interval=3.0):
        self.on_enter = on_enter  # callback when person appears
        self.on_leave = on_leave  # callback when person leaves
        self.check_interval = check_interval
        self.present = False
        self.running = False
        self._thread = None

    def start(self, capture_fn):
        """Start monitoring. capture_fn should return a numpy RGB image array."""
        self.capture_fn = capture_fn
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                img = self.capture_fn()
                if img is not None:
                    now_present = quick_presence(img)
                    if now_present and not self.present:
                        self.present = True
                        if self.on_enter:
                            self.on_enter()
                    elif not now_present and self.present:
                        self.present = False
                        if self.on_leave:
                            self.on_leave()
            except Exception as e:
                pass
            time.sleep(self.check_interval)