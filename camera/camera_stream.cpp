#include <OpenNI.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define XN_STREAM_PROPERTY_INPUT_FORMAT 0x10800001
#define XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422 5

int main(int argc, char** argv) {
    int width = 640, height = 480;

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

    int inputFormat = XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422;
    color.setProperty(XN_STREAM_PROPERTY_INPUT_FORMAT, &inputFormat, sizeof(inputFormat));

    if (color.start() != openni::STATUS_OK) {
        fprintf(stderr, "stream start failed: %s\n", openni::OpenNI::getExtendedError());
        return 1;
    }

    fprintf(stderr, "READY\n");
    fflush(stderr);

    char line[64];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (strncmp(line, "QUIT", 4) == 0) break;

        openni::VideoFrameRef frame;
        bool gotFrame = false;
        for (int attempt = 0; attempt < 10 && !gotFrame; attempt++) {
            openni::VideoStream* pStream = &color;
            int changed;
            if (openni::OpenNI::waitForAnyStream(&pStream, 1, &changed, 1000) != openni::STATUS_OK) {
                continue;
            }
            if (color.readFrame(&frame) != openni::STATUS_OK) continue;
            if (frame.isValid() && frame.getDataSize() == width * height * 3) {
                gotFrame = true;
            }
        }

        if (!gotFrame) {
            printf("0 0\n");
            fflush(stdout);
            continue;
        }

        printf("%d %d\n", frame.getWidth(), frame.getHeight());
        fflush(stdout);
        fwrite(frame.getData(), 1, frame.getDataSize(), stdout);
        fflush(stdout);
    }

    color.stop();
    color.destroy();
    device.close();
    openni::OpenNI::shutdown();
    return 0;
}
