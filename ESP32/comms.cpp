#include "comms.h"
#include "sensors.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#define DIST_CORTE 35 // Distancia para considerar corte de 35cm o menos
#define PESO_MIN 50 // Peso minimo de 50g

const char* ssid = "WIFI_SSID";
const char* password = "PASSWORD";
const char* mqtt_server = "192.168.1.87";
const int mqtt_port = 1883;
const char* mqtt_topic = "sensor/datos";

WiFiClient espClient;
PubSubClient client(espClient);
String received_message = "";

void Comms::setup_wifi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConectado a WiFi");
}

void Comms::setup_mqtt() {
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback_mqtt);
}

bool Comms::mqtt_connected() {
    return client.connected();
}

void Comms::reconnect_mqtt() {
    while (!client.connected()) {
        Serial.print("Conectando a MQTT...");
        if (client.connect("ESP32Client")) {
            Serial.println("Conectado");
            client.subscribe("sensor/comandos");  // Suscribirse solo a este topic
        } else {
            Serial.print("Fallo, rc=");
            Serial.print(client.state());
            Serial.println(" Intentando de nuevo en 5 segundos");
            delay(5000);
        }
    }
}

void Comms::mqtt_loop() {
    client.loop();
}

void Comms::send_sensor_data(Sensors& sensors) {
    StaticJsonDocument<256> json;
    json["entrada_cortada"] = sensors.read_ultrasonic()<DIST_CORTE;
    json["peso"] = sensors.read_load_cell();
    json["objeto_control"] = sensors.read_load_cell()>PESO_MIN;
    json["metal"] = sensors.read_inductive_sensor();
    json["iluminacion"] = sensors.read_light_sensor();
    
    char buffer[256];
    serializeJson(json, buffer);
    client.publish(mqtt_topic, buffer);
}

bool Comms::check_mqtt_message() {
    return received_message.length() > 0;
}

String Comms::get_mqtt_message() {
    String message = received_message;
    received_message = "";  // Limpiar el mensaje después de leerlo
    return message;
}

void Comms::callback_mqtt(char* topic, byte* payload, unsigned int length) {
    payload[length] = '\0';  // Asegurar el final de la cadena

    if (String(topic) == "sensor/comandos") {  // Filtrar solo los mensajes del topic correcto
        received_message = String((char*)payload);
        Serial.println("Mensaje MQTT recibido en sensor/comandos: " + received_message);
    }
}
