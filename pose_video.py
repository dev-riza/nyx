# -*- coding: utf-8 -*-
"""Record a video from the Orbbec camera with pose keypoints drawn on it.

Usage:  python pose_video.py [--seconds 10] [--output pose_video.mp4] [--min-score 0.3] [--no-pose]

Stop nyx.py first -- only one program can use the camera at a time.

Frames are captured at full camera speed first (camera/capture_frames), then
YOLOv8n-pose runs on each one (~0.15 s/frame on the Pi), so the finished
video plays smoothly at real speed even though processing is slower.
"""
import argparse
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from body_language import detect_people
from camera.astra import OPENNI2_DRIVERS_PATH
from vision import _adjust_brightness

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_FRAMES_BIN = os.path.join(HERE, 'camera', 'capture_frames')
WIDTH, HEIGHT = 640, 480
FRAME_BYTES = WIDTH * HEIGHT * 3
WARMUP_FRAMES = 10  # first frames are dark while auto-exposure settles

# Pairs of keypoint indices to connect (see body_language.py for names)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # face
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]
LEFT_COLOR = (255, 160, 0)    # BGR: blue-ish for the person's left side
RIGHT_COLOR = (0, 140, 255)   # orange for right side
CENTER_COLOR = (0, 230, 0)


def side_color(idx):
    if idx == 0:
        return CENTER_COLOR
    return LEFT_COLOR if idx % 2 == 1 else RIGHT_COLOR


def draw_pose(bgr, keypoints, min_score):
    for a, b in SKELETON:
        xa, ya, sa = keypoints[a]
        xb, yb, sb = keypoints[b]
        if sa >= min_score and sb >= min_score:
            color = side_color(a) if side_color(a) == side_color(b) else CENTER_COLOR
            cv2.line(bgr, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for idx, (x, y, s) in enumerate(keypoints):
        if s >= min_score:
            cv2.circle(bgr, (int(x), int(y)), 5, side_color(idx), -1, cv2.LINE_AA)
            cv2.circle(bgr, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)


def capture(seconds, raw_path):
    """Stream frames from the camera into raw_path. Returns (frame_count, fps)."""
    count = WARMUP_FRAMES + int(seconds * 30)
    env = os.environ.copy()
    env['OPENNI2_DRIVERS_PATH'] = OPENNI2_DRIVERS_PATH
    proc = subprocess.Popen(
        ['sudo', '-E', CAPTURE_FRAMES_BIN, str(count)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    saved = 0
    first_time = last_time = None
    with open(raw_path, 'wb') as f:
        for i in range(count):
            header = proc.stdout.readline()
            if not header:
                break
            data = proc.stdout.read(FRAME_BYTES)
            if len(data) < FRAME_BYTES:
                break
            if i < WARMUP_FRAMES:
                continue
            f.write(data)
            saved += 1
            last_time = time.time()
            if first_time is None:
                first_time = last_time
            if saved % 30 == 0:
                print(f"  recorded {saved // 30}s...")

    proc.wait()
    if saved == 0:
        raise RuntimeError(f"no frames captured: {proc.stderr.read().decode(errors='replace')}")
    elapsed = last_time - first_time
    fps = (saved - 1) / elapsed if elapsed > 0 else 30.0
    return saved, fps


def render(raw_path, frame_count, fps, output, min_score, show_pose=True):
    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}', '-r', f'{fps:.2f}',
         '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', output],
        stdin=subprocess.PIPE
    )
    with open(raw_path, 'rb') as f:
        for i in range(frame_count):
            rgb = np.frombuffer(f.read(FRAME_BYTES), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            rgb = _adjust_brightness(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if show_pose:
                for keypoints in detect_people(rgb):
                    draw_pose(bgr, keypoints, min_score)
            ffmpeg.stdin.write(bgr.tobytes())
            if (i + 1) % 30 == 0 or i + 1 == frame_count:
                print(f"  processed {i + 1}/{frame_count} frames")
    ffmpeg.stdin.close()
    ffmpeg.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', default='pose_video.mp4')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='hide keypoints the model is less confident about (0-1)')
    parser.add_argument('--no-pose', action='store_true', help='record a plain video without pose dots')
    args = parser.parse_args()

    fd, raw_path = tempfile.mkstemp(suffix='.raw', dir=HERE)
    os.close(fd)
    try:
        print(f"Recording {args.seconds:g}s...")
        frame_count, fps = capture(args.seconds, raw_path)
        print(f"Captured {frame_count} frames at {fps:.1f} fps. " + ("Saving..." if args.no_pose else "Drawing pose..."))
        render(raw_path, frame_count, fps, args.output, args.min_score, show_pose=not args.no_pose)
    finally:
        os.remove(raw_path)
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
