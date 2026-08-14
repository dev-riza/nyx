# Orbbec Astra camera driver - reverse engineered
# Protocol: PS1080 OpenNI2 vendor USB protocol
# VID:PID = 2bc5:0401

import usb.core, usb.util, struct, numpy as np

MAGIC_HOST = 0x4d47
MAGIC_FW = 0x4252

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
        except: pass
    dev.set_configuration()
    usb.util.claim_interface(dev, 0)
    send_cmd(dev, 6, struct.pack('<H', 1))  # SetMode PS
    return dev

def close_device(dev):
    usb.util.release_interface(dev, 0)

def uyvy_to_rgb(data, width=640, height=476):
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
    u  = arr[:,0].astype(float) - 128
    y0 = arr[:,1].astype(float)
    v  = arr[:,2].astype(float) - 128
    y1 = arr[:,3].astype(float)
    def conv(y):
        r = np.clip(y + 1.402*v, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.344*u - 0.714*v, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.772*u, 0, 255).astype(np.uint8)
        return np.stack([r,g,b], axis=1)
    rgb = np.zeros((len(arr)*2, 3), dtype=np.uint8)
    rgb[0::2] = conv(y0)
    rgb[1::2] = conv(y1)
    return rgb[:width*height].reshape(height, width, 3)

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

def get_color_frame(dev):
    set_param(dev, 5, 1)  # PARAM_GENERAL_STREAM0_MODE = COLOR
    frame_packets = {}
    frame_started = False
    for _ in range(1000):
        try:
            pkt = bytes(dev.read(0x82, 3072, timeout=1000))
            magic, ntype, packet_id, _ = struct.unpack_from('<HHHH', pkt, 0)
            if magic != MAGIC_FW: continue
            if ntype == 0x8100:
                if frame_started and len(frame_packets) > 40:
                    break
                frame_started = True
                frame_packets = {}
            elif ntype == 0x8200 and frame_started:
                frame_packets[packet_id] = bytes(pkt[8:8+3064])
        except: continue
    pixel_data = bytearray()
    for k in sorted(frame_packets.keys()):
        pixel_data.extend(frame_packets[k])
    return uyvy_to_rgb(bytes(pixel_data))

def get_depth_frame(dev):
    set_param(dev, 6, 2)  # PARAM_GENERAL_STREAM1_MODE = DEPTH
    pixel_data = bytearray()
    frame_started = False
    while len(pixel_data) < 106200:
        try:
            pkt = bytes(dev.read(0x81, 3072, timeout=2000))
            magic, ntype, _, _ = struct.unpack_from('<HHHH', pkt, 0)
            if magic != MAGIC_FW: continue
            if ntype == 0x7100:
                frame_started = True
                pixel_data = bytearray()
            elif ntype == 0x7200 and frame_started:
                pixel_data.extend(pkt[8:8+3064])
        except: break
    depth = unpack11to16(bytes(pixel_data))[:76800].reshape(240, 320)
    return depth
