#include "camera.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Constructor de la clase Camera (actualmente no realiza ninguna acción)
Camera::Camera() {}

// Captura una imagen desde la cámara y la guarda como archivo JPEG
std::string Camera::capture_image() {
    // Abre el dispositivo de captura de vídeo (cámara por defecto)
    cv::VideoCapture cap(0);
    
    // Verifica si la cámara se abrió correctamente
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir la cámara" << std::endl;
        return "";  // Devuelve una cadena vacía en caso de fallo
    }

    cv::Mat frame;     // Objeto Mat para almacenar el fotograma capturado
    cap >> frame;      // Captura una imagen (fotograma) desde la cámara

    // Nombre del archivo donde se guardará la imagen capturada
    std::string filename = "captura.jpg";

    // Guarda la imagen en disco en formato JPEG
    cv::imwrite(filename, frame);

    // Devuelve el nombre del archivo generado
    return filename;
}
