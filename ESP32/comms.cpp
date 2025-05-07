#include "comms.h"
#include "sensors.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// Umbral de distancia para detectar una interrupción o corte (en cm)
#define DIST_CORTE 35

// Umbral mínimo de peso para considerar un objeto válido (en gramos * 1000)
#define PESO_MIN 1050

// Credenciales de red Wi-Fi
const char* ssid = "WIFI_SSID";
const char* password = "PASSWORD";

// Configuración del servidor MQTT
const char* mqtt_server = "192.168.1.87";
const int mqtt_port = 1883;
const char* mqtt_topic = "sensor/datos";

// Credenciales del broker MQTT
const char* mqtt_user = "usuario_mqtt";
const char* mqtt_password = "clave_mqtt";

// Cliente WiFi y cliente MQTT
WiFiClient espClient;
PubSubClient client(espClient);

// Variable global para almacenar el mensaje recibido por MQTT
String received_message = "";

// Establece la conexión Wi-Fi
void Comms::setup_wifi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");  // Indica que se sigue intentando conectar
    }
    Serial.println("\nConectado a WiFi");
}

// Configura el cliente MQTT con servidor y callback
void Comms::setup_mqtt() {
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback_mqtt);
}

// Verifica si el cliente MQTT está conectado
bool Comms::mqtt_connected() {
    return client.connected();
}

// Intenta reconectar al servidor MQTT en caso de desconexión
void Comms::reconnect_mqtt() {
    while (!client.connected()) {
        Serial.print("Conectando a MQTT...");
        if (client.connect("ESP32Client", mqtt_user, mqtt_password))) {
            Serial.println("Conectado");
            client.subscribe("sensor/comandos");  // Se suscribe al topic donde se reciben comandos
        } else {
            Serial.print("Fallo, rc=");
            Serial.print(client.state());  // Muestra el código de error
            Serial.println(" Intentando de nuevo en 5 segundos");
            delay(5000);  // Espera antes de reintentar
        }
    }
}

// Mantiene el cliente MQTT activo (procesa mensajes entrantes y mantiene la conexión)
void Comms::mqtt_loop() {
    client.loop();
}

// Envía los datos recogidos por los sensores en formato JSON a través de MQTT
void Comms::send_sensor_data(Sensors& sensors) {
    StaticJsonDocument<256> json;

    // Verifica si se ha interrumpido la entrada (objeto presente)
    json["entrada_cortada"] = sensors.read_ultrasonic() < DIST_CORTE;

    // Lee el valor del sensor de peso
    float peso = sensors.read_load_cell();
    json["peso"] = peso;

    // Determina si el peso corresponde a un objeto válido
    json["objeto_control"] = peso > PESO_MIN;

    // Lee si el objeto es metálico (sensor inductivo)
    json["metal"] = sensors.read_inductive_sensor();

    // Evalúa el nivel de iluminación (para procesamiento por cámara)
    json["iluminacion"] = sensors.read_light_sensor();

    // Serializa el JSON y lo publica en el topic MQTT
    char buffer[256];
    serializeJson(json, buffer);
    client.publish(mqtt_topic, buffer);
}

// Verifica si hay un mensaje pendiente de ser procesado
bool Comms::check_mqtt_message() {
    return received_message.length() > 0;
}

// Obtiene el mensaje MQTT recibido y limpia la variable
String Comms::get_mqtt_message() {
    String message = received_message;
    received_message = "";  // Limpia después de la lectura
    return message;
}

// Callback ejecutado cuando se recibe un mensaje MQTT
void Comms::callback_mqtt(char* topic, byte* payload, unsigned int length) {
    payload[length] = '\0';  // Asegura que el mensaje sea una cadena terminada en null

    // Procesa solo los mensajes que llegan al topic "sensor/comandos"
    if (String(topic) == "sensor/comandos") {
        received_message = String((char*)payload);
        Serial.println("Mensaje MQTT recibido en sensor/comandos: " + received_message);
    }
}
