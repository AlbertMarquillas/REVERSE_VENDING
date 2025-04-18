// main.cpp
#include "comms.h"
#include "camera.h"
#include "classifier.h"
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <map>
#include <jsoncpp/json/json.h>

#define MQTT_TOPIC "sensor/datos"
#define MQTT_RESPONSE_TOPIC "sensor/comandos"
#define TIMEOUT_SECONDS 1200  // 20 minutos

Comms comms("192.168.1.87", 1883, MQTT_TOPIC, MQTT_RESPONSE_TOPIC);
Camera camera;
Classifier classifier("model.ts");

std::string current_user;
std::vector<std::string> session_results;
time_t last_action_time = time(nullptr);
bool session_active = true;

float calcular_recompensa(int clase, bool es_metal) {
    // clase 0 -> botella_plastico
    // clase 1 -> botella_vidrio
    // clase 2 -> lata
    // classe 3 -> marca de valor
    if (clase == 4) return 3;
    if (clase == 0 && !es_metal) return 2.1;
    if (clase == 0 && es_metal) return 1;
    if (clase == 1 && !es_metal) return 2.2;
    if (clase == 1 && es_metal) return 1;
    if (clase == 2 && es_metal) return 2.3;
    if (clase == 2 && !es_metal) return 0;
    return 0;
}

void finalizar_sesion() {
    Json::Value resultado_final;
    resultado_final["usuario"] = current_user;
    resultado_final["objetos"] = Json::arrayValue;

    for (const auto& r : session_results) {
        resultado_final["objetos"].append(r);
    }

    Json::StreamWriterBuilder writer;
    std::string mensaje_final = Json::writeString(writer, resultado_final);

    comms.send_message(mensaje_final);
    std::cout << "\n[SISTEMA] Sesión finalizada. Datos enviados.\n";
    session_active = false;
}

void message_callback(const std::string& payload) {
    last_action_time = time(nullptr);

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::istringstream ss(payload);
    std::string errs;

    if (!Json::parseFromStream(builder, ss, &root, &errs)) {
        std::cerr << "[ERROR] Fallo al parsear JSON: " << errs << std::endl;
        return;
    }

    bool objeto_control = root.get("objeto_control", false).asBool();
    bool corte = root.get("sensor_corte", false).asBool();
    bool es_metal = root.get("sensor_inductivo", false).asBool();

    if (!session_active) return;

    if (objeto_control) {
        if (corte) {
            std::cout << "[AVISO] Corte detectado. Objeto descartado.\n";
            session_results.push_back("0");
            return;
        }

        std::string image_path = camera.capture_image();
        int clase = classifier.predict(image_path);

        float recompensa = calcular_recompensa(clase, es_metal);
        std::cout << "[INFO] Objeto: " << clase << " | Metal: " << (es_metal ? "Sí" : "No") 
                  << " => Recompensa: " << recompensa << "\n";

        session_results.push_back(std::to_string(recompensa));
    }
}

int main() {
    std::cout << "Introduce tu nombre de usuario: ";
    std::cin >> current_user;

    comms.set_callback(message_callback);
    comms.connect();

    comms.send_message("OP_MODE");

    while (session_active) {
        comms.loop();
        usleep(500000); // 500 ms

        time_t now = time(nullptr);
        if (difftime(now, last_action_time) > TIMEOUT_SECONDS) {
            std::cout << "[AVISO] Tiempo de inactividad superado. Cerrando sesión automáticamente.\n";
            finalizar_sesion();
        }

        // Verifica si el usuario quiere finalizar manualmente
        if (std::cin.rdbuf()->in_avail()) {
            std::string comando;
            std::cin >> comando;
            if (comando == "finalizar") {
                finalizar_sesion();
            }
        }
    }

    return 0;
}
