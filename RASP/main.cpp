// main.cpp
#include "comms.h"
#include "camera.h"
#include "classifier.h"
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <map>
#include <jsoncpp/json/json.h>

// Definición de topics MQTT y tiempo máximo de inactividad
#define MQTT_TOPIC "sensor/datos"
#define MQTT_RESPONSE_TOPIC "sensor/comandos"
#define TIMEOUT_SECONDS 1200  // 20 minutos

// Instancias de las clases principales
Comms comms("192.168.1.87", 1883, MQTT_TOPIC, MQTT_RESPONSE_TOPIC);
Camera camera;
Classifier classifier("model.ts");

// Variables para la sesión
std::string current_user;                  // Usuario actual de la sesión
std::vector<std::string> session_results;  // Recompensas obtenidas durante la sesión
time_t last_action_time = time(nullptr);   // Timestamp de la última acción
bool session_active = true;                // Bandera de control de sesión

// Función que calcula la recompensa basada en la clase detectada y si es metálico
float calcular_recompensa(int clase, bool es_metal) {
    // clase 0 -> botella plástico
    // clase 1 -> botella vidrio
    // clase 2 -> lata
    // clase 3 -> marca de valor (valor especial)
    if (clase == 4) return 3;
    if (clase == 0 && !es_metal) return 2.1;
    if (clase == 0 && es_metal) return 1;
    if (clase == 1 && !es_metal) return 2.2;
    if (clase == 1 && es_metal) return 1;
    if (clase == 2 && es_metal) return 2.3;
    if (clase == 2 && !es_metal) return 0;
    return 0; // Caso por defecto
}

// Función que finaliza la sesión activa y publica los resultados vía MQTT
void finalizar_sesion() {
    comms.send_message("REST_MODE");  // Señal para cambiar a modo reposo

    Json::Value resultado_final;
    resultado_final["usuario"] = current_user;
    resultado_final["objetos"] = Json::arrayValue;

    for (const auto& r : session_results) {
        resultado_final["objetos"].append(r);
    }

    Json::StreamWriterBuilder writer;
    std::string mensaje_final = Json::writeString(writer, resultado_final);

    comms.send_message(mensaje_final);  // Envía resumen por MQTT
    std::cout << "\n[SISTEMA] Sesión finalizada. Datos enviados.\n";
    session_active = false;
}

// Callback que se activa al recibir mensajes MQTT desde el ESP32
void message_callback(const std::string& payload) {
    last_action_time = time(nullptr);  // Resetea el temporizador de inactividad

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::istringstream ss(payload);
    std::string errs;

    // Intenta parsear el mensaje como JSON
    if (!Json::parseFromStream(builder, ss, &root, &errs)) {
        std::cerr << "[ERROR] Fallo al parsear JSON: " << errs << std::endl;
        return;
    }

    // Extrae campos relevantes del mensaje JSON
    bool objeto_control = root.get("objeto_control", false).asBool();
    bool corte = root.get("entrada_cortada", false).asBool();
    bool es_metal = root.get("metal", false).asBool();

    if (!session_active) return;

    if (objeto_control) {
        if (corte) {
            std::cout << "[AVISO] Corte detectado. Objeto descartado.\n";
            session_results.push_back("0");  // Penaliza el corte
            return;
        }

        // Captura imagen del objeto y lo clasifica
        std::string image_path = camera.capture_image();
        int clase = classifier.predict(image_path);

        // Calcula y muestra la recompensa
        float recompensa = calcular_recompensa(clase, es_metal);
        std::cout << "[INFO] Objeto: " << clase << " | Metal: " << (es_metal ? "Sí" : "No") 
                  << " => Recompensa: " << recompensa << "\n";

        // Almacena la recompensa en el historial de sesión
        session_results.push_back(std::to_string(recompensa));
    }
}

int main() {
    // Solicita el nombre del usuario al iniciar sesión
    std::cout << "Introduce tu nombre de usuario: ";
    std::cin >> current_user;

    // Configura el sistema de comunicación y conecta al broker MQTT
    comms.set_callback(message_callback);
    comms.connect();

    // Envía señal para activar el sistema
    comms.send_message("OP_MODE");

    // Bucle principal del programa
    while (session_active) {
        comms.loop();               // Gestiona eventos MQTT
        usleep(500000);            // Espera 500 ms

        // Verifica si ha pasado el tiempo de inactividad
        time_t now = time(nullptr);
        if (difftime(now, last_action_time) > TIMEOUT_SECONDS) {
            std::cout << "[AVISO] Tiempo de inactividad superado. Cerrando sesión automáticamente
