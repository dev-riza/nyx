# -*- coding: utf-8 -*-
"""Quick tool: capture a frame, run YOLO, save an annotated image so you can
actually see the bounding boxes instead of just reading text output."""

import cv2
from detect import model
from camera.astra import get_color_frame

def capture_and_annotate(save_path='/tmp/detection_view.png', conf_threshold=0.4):
    frame = get_color_frame()
    if frame is None:
        print("Camera capture failed.")
        return None

    results = model(frame, conf=conf_threshold, verbose=False)

    # results[0].plot() returns a numpy array (BGR) with boxes/labels drawn on it
    annotated = results[0].plot()

    # plot() returns BGR (OpenCV convention) -- convert to RGB before saving with cv2
    # actually cv2.imwrite expects BGR directly, so no conversion needed here
    cv2.imwrite(save_path, annotated)
    print(f"Saved annotated image to {save_path}")
    return save_path

if __name__ == '__main__':
    capture_and_annotate()
