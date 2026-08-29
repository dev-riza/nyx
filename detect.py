# -*- coding: utf-8 -*-
import threading
import time
from collections import Counter
from ultralytics import YOLO
from camera.astra import get_color_frame, CameraServer

MODEL_PATH = 'yolov8n_ncnn_model'
model = YOLO(MODEL_PATH, task='detect')

POLL_IMGSZ = 320


def _run_yolo(frame, conf_threshold, imgsz=None):
    kwargs = {'conf': conf_threshold, 'verbose': False}
    if imgsz:
        kwargs['imgsz'] = imgsz
    results = model(frame, **kwargs)
    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'label': label,
                'confidence': round(confidence, 2),
                'box': tuple(round(v) for v in xyxy)
            })
    return detections


def detect_objects(conf_threshold=0.4):
    frame = get_color_frame()
    if frame is None:
        return None
    return _run_yolo(frame, conf_threshold)


def describe_detections(detections):
    if detections is None:
        return "Camera not available."
    if not detections:
        return "I don't see anything recognizable."
    counts = Counter(d['label'] for d in detections)
    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(f"a {label}")
        else:
            parts.append(f"{count} {label}s")
    if len(parts) == 1:
        return f"I see {parts[0]}."
    return f"I see {', '.join(parts[:-1])} and {parts[-1]}."


class PresenceMonitor:
    def __init__(self, poll_interval=5, conf_threshold=0.4, watch_labels=None,
                 imgsz=POLL_IMGSZ):
        self.poll_interval = poll_interval
        self.conf_threshold = conf_threshold
        self.watch_labels = watch_labels or {'person'}
        self.imgsz = imgsz

        self._camera = CameraServer()
        self._lock = threading.Lock()
        self._last_detections = None
        self._last_check_time = None
        self._person_present = False
        self._last_frame_time_ms = None

        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._camera.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 2)
        self._camera.stop()

    def _run(self):
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                frame = self._camera.get_frame()
                dets = _run_yolo(frame, self.conf_threshold, imgsz=self.imgsz) if frame is not None else None
            except Exception as e:
                print(f"PresenceMonitor detection error: {e}")
                dets = None
            elapsed_ms = (time.time() - t0) * 1000

            with self._lock:
                self._last_detections = dets
                self._last_check_time = time.time()
                self._last_frame_time_ms = elapsed_ms
                if dets:
                    self._person_present = any(d['label'] in self.watch_labels for d in dets)
                else:
                    self._person_present = False

            remaining = max(0, self.poll_interval - elapsed_ms / 1000)
            slept = 0
            while slept < remaining and not self._stop_event.is_set():
                time.sleep(0.2)
                slept += 0.2

    @property
    def person_present(self):
        with self._lock:
            return self._person_present

    @property
    def last_detections(self):
        with self._lock:
            return self._last_detections

    @property
    def last_check_time(self):
        with self._lock:
            return self._last_check_time

    @property
    def last_frame_time_ms(self):
        with self._lock:
            return self._last_frame_time_ms

    @property
    def seconds_since_last_check(self):
        with self._lock:
            if self._last_check_time is None:
                return None
            return time.time() - self._last_check_time

    def describe_now(self):
        return describe_detections(self.last_detections)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        print("Starting PresenceMonitor (Ctrl+C to stop)...")
        print("First poll includes camera startup (~7s), after that should be fast.")
        monitor = PresenceMonitor(poll_interval=5)
        monitor.start()
        try:
            last_seen_time = None
            while True:
                time.sleep(0.5)
                if monitor.last_check_time != last_seen_time:
                    last_seen_time = monitor.last_check_time
                    frame_ms = monitor.last_frame_time_ms
                    print(f"[capture+inference took {frame_ms:.0f}ms] present={monitor.person_present} -- {monitor.describe_now()}")
        except KeyboardInterrupt:
            print("\nStopping...")
            monitor.stop()
    else:
        print("Running single detection (one-shot, includes camera open overhead)...")
        start = time.time()
        dets = detect_objects()
        elapsed = time.time() - start
        print(f"Took {elapsed:.2f}s")
        print(describe_detections(dets))
        if dets:
            for d in dets:
                print(f"  {d['label']} ({d['confidence']}) at {d['box']}")
