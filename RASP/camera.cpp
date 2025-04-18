#include "camera.h"
#include <opencv2/opencv.hpp>
#include <iostream>

Camera::Camera() {}

std::string Camera::capture_image() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir la cámara" << std::endl;
        return "";
    }
    cv::Mat frame;
    cap >> frame;
    std::string filename = "captura.jpg";
    cv::imwrite(filename, frame);
    return filename;
}
