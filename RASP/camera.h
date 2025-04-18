#ifndef CAMERA_H
#define CAMERA_H

#include <string>

class Camera {
public:
    Camera();
    bool detect_object();
    std::string capture_image();
};

#endif
