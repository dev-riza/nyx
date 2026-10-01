# -*- coding: utf-8 -*-
import os
import threading
from ultralytics import YOLO

# YOLOv8n-pose exported to NCNN (regenerate: YOLO('yolov8n-pose.pt').export(format='ncnn', imgsz=320))
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n-pose_ncnn_model')
IMGSZ = 320
PERSON_CONF = 0.5     # minimum confidence that a detection is a person
KEYPOINT_CONF = 0.5   # minimum confidence to trust a single keypoint

# Keypoint indices (COCO order). "Left"/"right" are the person's own sides.
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

_model = None
_model_lock = threading.Lock()  # presence thread and voice commands share the model

def _get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH, task='pose')
    return _model

def detect_people(img_arr):
    """Run pose detection on an RGB image array.

    Returns a list of people, largest (closest) first. Each person is a list
    of 17 (x, y, confidence) keypoints in image pixels."""
    with _model_lock:
        # Ultralytics expects BGR numpy arrays
        result = _get_model()(img_arr[:, :, ::-1], imgsz=IMGSZ, conf=PERSON_CONF, verbose=False)[0]
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return []

    xy = result.keypoints.xy.numpy()
    conf = result.keypoints.conf.numpy() if result.keypoints.conf is not None else None
    areas = result.boxes.xywh.numpy()[:, 2] * result.boxes.xywh.numpy()[:, 3]

    people = []
    for i in areas.argsort()[::-1]:
        people.append([
            (float(x), float(y), float(conf[i][k]) if conf is not None else 1.0)
            for k, (x, y) in enumerate(xy[i])
        ])
    return people


def _ok(kp, *idxs):
    return all(kp[i][2] >= KEYPOINT_CONF for i in idxs)


def analyze_body(img_arr):
    """
    Takes a numpy RGB image array.
    Returns a dict with presence, posture, and gesture info for the main
    (closest) person. Measurements are relative to shoulder width, so they
    work at any distance from the camera.
    """
    results = {}

    try:
        people = detect_people(img_arr)
    except Exception as e:
        return {'present': False, 'people': 0, 'summary': f'Error: {e}'}

    if not people:
        return {'present': False, 'people': 0, 'summary': 'No person detected'}

    results['present'] = True
    results['people'] = len(people)
    kp = people[0]
    results['keypoints'] = kp

    shoulders_ok = _ok(kp, LEFT_SHOULDER, RIGHT_SHOULDER)
    if shoulders_ok:
        scale = abs(kp[LEFT_SHOULDER][0] - kp[RIGHT_SHOULDER][0]) or 1.0
        shoulder_y = (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2
        shoulder_mid_x = (kp[LEFT_SHOULDER][0] + kp[RIGHT_SHOULDER][0]) / 2

        # Shoulder level
        shoulder_diff = abs(kp[LEFT_SHOULDER][1] - kp[RIGHT_SHOULDER][1]) / scale
        results['shoulders'] = 'level' if shoulder_diff < 0.1 else 'uneven'

        # Lean - nose offset from shoulder midpoint. The camera isn't mirrored,
        # so the person's left is on the image's right (larger x).
        if _ok(kp, NOSE):
            lean = (kp[NOSE][0] - shoulder_mid_x) / scale
            if abs(lean) < 0.2:
                results['posture'] = 'centered'
            elif lean > 0:
                results['posture'] = 'leaning left'
            else:
                results['posture'] = 'leaning right'

            # Slouching - head sunk down toward the shoulders. Hips are usually
            # out of frame at a desk, so compare nose height to shoulder width.
            results['slouching'] = (shoulder_y - kp[NOSE][1]) / scale < 0.35

        # Arms raised - wrist above its shoulder
        left_raised = _ok(kp, LEFT_WRIST) and kp[LEFT_WRIST][1] < kp[LEFT_SHOULDER][1]
        right_raised = _ok(kp, RIGHT_WRIST) and kp[RIGHT_WRIST][1] < kp[RIGHT_SHOULDER][1]
        if left_raised and right_raised:
            results['arms'] = 'both raised'
        elif left_raised:
            results['arms'] = 'left arm raised'
        elif right_raised:
            results['arms'] = 'right arm raised'
        else:
            results['arms'] = 'down'

    # Head tilt - ear heights relative to the distance between the ears
    if _ok(kp, LEFT_EAR, RIGHT_EAR):
        ear_dist = abs(kp[LEFT_EAR][0] - kp[RIGHT_EAR][0]) or 1.0
        tilt = (kp[LEFT_EAR][1] - kp[RIGHT_EAR][1]) / ear_dist
        if abs(tilt) < 0.15:
            results['head'] = 'level'
        elif tilt > 0:
            results['head'] = 'tilted left'   # left ear lower
        else:
            results['head'] = 'tilted right'

    # Build summary
    count = len(people)
    summary = 'One person present.' if count == 1 else f'{count} people present.'
    parts = []
    if results.get('arms', 'down') != 'down':
        parts.append(results['arms'])
    if results.get('posture', 'centered') != 'centered':
        parts.append(results['posture'])
    if results.get('head', 'level') != 'level':
        parts.append('head ' + results['head'])
    if results.get('slouching'):
        parts.append('slouching')
    if not shoulders_ok:
        parts.append('shoulders not visible, so posture is unclear')

    results['summary'] = summary + ' ' + (', '.join(parts).capitalize() + '.' if parts else 'Upright, relaxed posture.')
    return results


def quick_presence(img_arr):
    """Fast check - is someone in frame?"""
    try:
        return len(detect_people(img_arr)) > 0
    except Exception:
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
        import time as _time
        import tts as tts_module
        while self.running:
            try:
                img = self.capture_fn()
                if img is not None:
                    now_present = quick_presence(img)
                    if now_present and not self.present:
                        self.present = True
                        if self.on_enter:
                            _time.sleep(1.5)
                            while tts_module.is_speaking:
                                _time.sleep(0.5)
                            self.on_enter()
                    elif not now_present and self.present:
                        self.present = False
                        if self.on_leave:
                            self.on_leave()
            except Exception as e:
                pass
            _time.sleep(self.check_interval)