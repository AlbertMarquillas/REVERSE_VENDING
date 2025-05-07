#include "comms.h"
#include <iostream>

// Inicialización del puntero a función de callback del usuario
void (*Comms::user_callback)(const std::string&) = nullptr;

// Constructor de la clase Comms
Comms::Comms(const std::string& broker, int port, const std::string& topic, const std::string& response_topic)
    : broker(broker), port(port), topic(topic), response_topic(response_topic) {
    
    // Inicializa la librería de Mosquitto (debe llamarse una vez por proceso)
    mosquitto_lib_init();

    // Crea una nueva instancia del cliente Mosquitto
    mosq = mosquitto_new(nullptr, true, this);  // `true` indica que se usa una sesión limpia

    // Establece la función de callback que se llamará cuando llegue un mensaje
    mosquitto_message_callback_set(mosq, message_callback);
}

// Destructor de la clase Comms
Comms::~Comms() {
    // Libera los recursos asociados al cliente
    mosquitto_destroy(mosq);

    // Limpia la librería de Mosquitto (debe llamarse al finalizar el uso)
    mosquitto_lib_cleanup();
}

// Conecta al broker MQTT y se suscribe al topic indicado
void Comms::connect() {
    // Establece usuario y contraseña antes de conectar
    const std::string mqtt_user = "usuario_mqtt";
    const std::string mqtt_password = "clave_mqtt";
    mosquitto_username_pw_set(mosq, mqtt_user.c_str(), mqtt_password.c_str());
    
    // Intenta establecer la conexión con el broker
    mosquitto_connect(mosq, broker.c_str(), port, 60);  // 60 segundos de keep-alive

    // Se suscribe al topic para recibir mensajes
    mosquitto_subscribe(mosq, nullptr, topic.c_str(), 0);

    // Inicia un bucle de recepción en un hilo aparte (no bloqueante)
    mosquitto_loop_start(mosq);
}

// Ejecuta un ciclo del loop MQTT (solo necesario si no se usa `loop_start`)
void Comms::loop() {
    mosquitto_loop(mosq, -1, 1);  // Loop con espera bloqueante
}

// Publica un mensaje en el topic de respuesta
void Comms::send_message(const std::string& message) {
    mosquitto_publish(
        mosq, nullptr,
        response_topic.c_str(),
        message.size(),
        message.c_str(),
        0,      // QoS 0: entrega "al menos una vez"
        false   // No conservar el mensaje (retain = false)
    );
}

// Asigna una función de callback que se ejecutará cuando llegue un mensaje
void Comms::set_callback(void (*callback)(const std::string&)) {
    user_callback = callback;
}

// Callback estático llamado por la librería Mosquitto al recibir un mensaje
void Comms::message_callback(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg) {
    // Si se ha registrado un callback de usuario, lo llama con el contenido del mensaje
    if (user_callback) {
        user_callback(std::string((char*)msg->payload, msg->payloadlen));
    }
}
