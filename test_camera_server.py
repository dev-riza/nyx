from camera.astra import CameraServer
from PIL import Image
import numpy as np

server = CameraServer()
print("Starting camera server...")
server.start()
print("Started. Grabbing 3 frames...")

for i in range(3):
    frame = server.get_frame()
    if frame is None:
        print(f"Frame {i}: FAILED")
        continue
    print(f"Frame {i}: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    Image.fromarray(frame).save(f"/tmp/server_frame_{i}.png")

server.stop()
print("Done, saved to /tmp/server_frame_0.png, _1.png, _2.png")
