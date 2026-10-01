// capture_frames.cpp
// Like capture_color, but keeps the stream open and captures N consecutive
// color frames (default 100) in one session -- for recording video.
// Usage: capture_frames [count]
//
// Writes each frame to stdout in the same format as capture_color:
// "<width> <height>\n" header line, then width*height*3 raw RGB888 bytes.
// Frames are read back-to-back as they arrive (unlike camera_stream's
// on-demand GET polling, which returned garbage after the first frame).

#include <OpenNI.h>
#include <cstdio>
#include <cstdlib>

#define XN_STREAM_PROPERTY_INPUT_FORMAT 0x10800001
#define XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422 5

int main(int argc, char** argv) {
    int width = 640, height = 480;
    int count = argc > 1 ? atoi(argv[1]) : 100;

    if (openni::OpenNI::initialize() != openni::STATUS_OK) {
        fprintf(stderr, "initialize failed: %s\n", openni::OpenNI::getExtendedError());
        return 1;
    }

    openni::Device device;
    if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK) {
        fprintf(stderr, "device open failed: %s\n", openni::OpenNI::getExtendedError());
        return 1;
    }

    openni::VideoStream color;
    if (color.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK) {
        fprintf(stderr, "stream create failed: %s\n", openni::OpenNI::getExtendedError());
        return 1;
    }

    const openni::SensorInfo& info = color.getSensorInfo();
    const openni::Array<openni::VideoMode>& modes = info.getSupportedVideoModes();
    bool modeFound = false;
    for (int i = 0; i < modes.getSize(); i++) {
        const openni::VideoMode& m = modes[i];
        if (m.getResolutionX() == width && m.getResolutionY() == height &&
            m.getPixelFormat() == openni::PIXEL_FORMAT_RGB888) {
            color.setVideoMode(m);
            modeFound = true;
            break;
        }
    }
    if (!modeFound) {
        fprintf(stderr, "requested %dx%d RGB888 mode not supported\n", width, height);
        return 1;
    }

    // Force uncompressed wire format before starting the stream (see capture_color.cpp)
    int inputFormat = XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422;
    color.setProperty(XN_STREAM_PROPERTY_INPUT_FORMAT, &inputFormat, sizeof(inputFormat));

    if (color.start() != openni::STATUS_OK) {
        fprintf(stderr, "stream start failed: %s\n", openni::OpenNI::getExtendedError());
        return 1;
    }

    int captured = 0, failures = 0;
    openni::VideoFrameRef frame;
    while (captured < count && failures < 30) {
        openni::VideoStream* pStream = &color;
        int changed;
        if (openni::OpenNI::waitForAnyStream(&pStream, 1, &changed, 1000) != openni::STATUS_OK ||
            color.readFrame(&frame) != openni::STATUS_OK ||
            !frame.isValid() || frame.getDataSize() != width * height * 3) {
            failures++;
            continue;
        }
        failures = 0;
        printf("%d %d\n", frame.getWidth(), frame.getHeight());
        fwrite(frame.getData(), 1, frame.getDataSize(), stdout);
        fflush(stdout);
        captured++;
    }

    color.stop();
    color.destroy();
    device.close();
    openni::OpenNI::shutdown();

    if (captured < count) {
        fprintf(stderr, "only captured %d of %d frames\n", captured, count);
        return captured > 0 ? 0 : 1;
    }
    return 0;
}
