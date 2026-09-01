# camera/astra.py
# Orbbec Astra camera driver
# VID:PID = 2bc5:0401
#
# COLOR: fixed via OpenNI2. The camera's firmware defaults to sending
# JPEG-compressed color data, and the PS1080 driver's JPEG decoder
# corrupts frames on this unit -- that caused tilted/doubled images from
# the old raw-pyusb approach. The fix forces the firmware into
# uncompressed mode before streaming (XN_STREAM_PROPERTY_INPUT_FORMAT = 5).
#
# CONCURRENCY: this camera's USB connection is fragile -- overlapping
# opens (e.g. PresenceMonitor and EmotionMonitor both polling around the
# same time) can wedge the device into a state that needs a physical
# unplug/replug to recover. get_color_frame() uses a lock so only one
# caller can access the camera at a time; everyone else waits their turn
# instead of racing for the USB device.
#
# CameraServer (persistent capture) still has a known unresolved bug --
# frames after the first one in a session return garbage/noise data under
# sustained polling. Not currently used by anything; kept here for future
# debugging. Use get_color_frame() for all real capture needs.
#
# DEPTH: still using the original raw-pyusb protocol. Known, unresolved
# corruption issue -- separate root cause from the color bug.

import os
import subprocess
import struct
import threading
import time
import usb.core
import usb.util
import numpy as np

MAGIC_HOST = 0x4d47
MAGIC_FW = 0x4252

CAMERA_DIR = os.path.dirname(__file__)
CAPTURE_COLOR_BIN = os.path.join(CAMERA_DIR, 'capture_color')
CAMERA_STREAM_BIN = os.path.join(CAMERA_DIR, 'camera_stream')
OPENNI2_DRIVERS_PATH = '/usr/lib/aarch64-linux-gnu/OpenNI2/Drivers'

# Serializes all access to get_color_frame() -- prevents concurrent
# background monitors (PresenceMonitor, EmotionMonitor) and on-demand
# voice commands from opening the camera at the same time, which can
# wedge the device's USB state.
_camera_lock = threading.Lock()


def send_cmd(dev, opcode, payload=b''):
    size = len(payload) // 2
    header = struct.pack('<HHHH', MAGIC_HOST, size, opcode, 0)
    dev.ctrl_transfer(0x40, 0, 0, 0, header + payload, timeout=3000)
    return dev.ctrl_transfer(0xC0, 0, 0, 0, 512, timeout=3000).tobytes()


def set_param(dev, param, value):
    return send_cmd(dev, 3, struct.pack('<HH', param, value))


def open_device():
    dev = usb.core.find(idVendor=0x2bc5, idProduct=0x0401)
    if dev is None:
        raise Exception('Orbbec Astra not found')
    for i in range(3):
        try:
            if dev.is_kernel_driver_active(i):
                dev.detach_kernel_driver(i)
        except Exception:
            pass
    dev.set_configuration()
    usb.util.claim_interface(dev, 0)
    send_cmd(dev, 6, struct.pack('<H', 1))
    return dev


def close_device(dev):
    usb.util.release_interface(dev, 0)


def get_color_frame(dev=None):
    """Capture one 640x480 RGB888 color frame by opening a fresh camera
    connection. dev is accepted for backwards compatibility but unused.
    Returns a (480, 640, 3) uint8 numpy array, or None on failure.

    Thread-safe: only one caller can access the camera at a time.
    Concurrent callers will block and wait their turn rather than racing
    for the USB device (which can wedge it).
    """
    if not os.path.exists(CAPTURE_COLOR_BIN):
        print(f"capture_color binary not found at {CAPTURE_COLOR_BIN} -- build it first")
        return None

    with _camera_lock:
        try:
            env = os.environ.copy()
            env['OPENNI2_DRIVERS_PATH'] = OPENNI2_DRIVERS_PATH
            result = subprocess.run(
                ['sudo', '-E', CAPTURE_COLOR_BIN],
                env=env,
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                print("capture_color failed:", result.stderr.decode(errors='replace'))
                return None

            return _parse_frame_bytes(result.stdout)
        except Exception as e:
            print(f"get_color_frame error: {e}")
            return None


def _parse_frame_bytes(stdout):
    newline_idx = stdout.index(b'\n')
    header = stdout[:newline_idx].decode().strip()
    width, height = map(int, header.split())
    if width == 0 or height == 0:
        return None
    raw = stdout[newline_idx + 1:]
    expected = width * height * 3
    if len(raw) != expected:
        print(f"unexpected frame size: got {len(raw)}, expected {expected}")
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


class CameraServer:
    """Persistent camera server -- KNOWN BUG, not currently used. Frames
    after the first one in a session return garbage/noise data under
    sustained polling. Kept here for future debugging; do not wire this
    into anything until the underlying issue is found. Use
    get_color_frame() instead."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def start(self, timeout=15):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(CAMERA_STREAM_BIN):
            raise FileNotFoundError(f"camera_stream binary not found at {CAMERA_STREAM_BIN}")
        env = os.environ.copy()
        env['OPENNI2_DRIVERS_PATH'] = OPENNI2_DRIVERS_PATH
        self._proc = subprocess.Popen(
            ['sudo', '-E', CAMERA_STREAM_BIN],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._proc.stderr.readline()
            if not line:
                break
            if b'READY' in line:
                return
        if self._proc.poll() is not None:
            stderr_output = self._proc.stderr.read().decode(errors='replace')
            raise RuntimeError(f"camera_stream failed to start: {stderr_output}")
        raise TimeoutError("camera_stream did not report READY in time")

    def get_frame(self, timeout=5):
        if self._proc is None or self._proc.poll() is not None:
            return None
        with self._lock:
            try:
                self._proc.stdin.write(b"GET\n")
                self._proc.stdin.flush()
                header_line = self._proc.stdout.readline()
                if not header_line:
                    return None
                width, height = map(int, header_line.decode().strip().split())
                if width == 0 or height == 0:
                    return None
                expected = width * height * 3
                raw = self._proc.stdout.read(expected)
                if len(raw) != expected:
                    return None
                return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            except Exception as e:
                print(f"CameraServer.get_frame error: {e}")
                return None

    def stop(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(b"QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None

    @property
    def is_running(self):
        return self._proc is not None and self._proc.poll() is None


def unpack11to16(data):
    output = []
    i = 0
    while i + 11 <= len(data):
        p = data[i:i+11]
        a0 = ((p[0]&0xFF)<<3)|((p[1]>>5)&0x07)
        a1 = ((p[1]&0x1F)<<6)|((p[2]>>2)&0x3F)
        a2 = ((p[2]&0x03)<<9)|((p[3]&0xFF)<<1)|((p[4]>>7)&0x01)
        a3 = ((p[4]&0x7F)<<4)|((p[5]>>4)&0x0F)
        a4 = ((p[5]&0x0F)<<7)|((p[6]>>1)&0x7F)
        a5 = ((p[6]&0x01)<<10)|((p[7]&0xFF)<<2)|((p[8]>>6)&0x03)
        a6 = ((p[8]&0x3F)<<5)|((p[9]>>3)&0x1F)
        a7 = ((p[9]&0x07)<<8)|(p[10]&0xFF)
        output.extend([a0,a1,a2,a3,a4,a5,a6,a7])
        i += 11
    return np.array(output, dtype=np.uint16)


def get_depth_frame(dev):
    set_param(dev, 6, 2)
    pixel_data = bytearray()
    frame_started = False
    while len(pixel_data) < 106200:
        try:
            pkt = bytes(dev.read(0x81, 3072, timeout=2000))
            magic, ntype, _, _ = struct.unpack_from('<HHHH', pkt, 0)
            if magic != MAGIC_FW:
                continue
            if ntype == 0x7100:
                frame_started = True
                pixel_data = bytearray()
            elif ntype == 0x7200 and frame_started:
                pixel_data.extend(pkt[8:8+3064])
        except Exception:
            break
    depth = unpack11to16(bytes(pixel_data))[:76800].reshape(240, 320)
    return depth