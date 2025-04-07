#include <Arduino.h>
#include "comms.h"
#include "sensors.h"
#include "lora_com.h"

#define SLEEP_TIME 6e6 // 6 segundos en microsegundos

Comms comms;
Sensors sensors;
LoRaModule lora;

void setup() { // Setea los diferentes modulos
    Serial.begin(115200);
    comms.setup_wifi();
    sensors.setup_sensors();
    comms.setup_mqtt();
    lora.setup_lora();
}

void loop() { // Bucle principal para iterar
    if (!comms.mqtt_connected()) {
        comms.reconnect_mqtt();
    }
    comms.mqtt_loop();
    comms.send_sensor_data(sensors); // Envia la informacion de los sensores
    delay(5000);  // Espera para verificar si hay mensajes MQTT

    if (comms.check_mqtt_message()) {
        String message = comms.get_mqtt_message();
        lora.send_data(message);  // Enviar el mensaje por LoRa
    }

    Serial.println("Entrando en modo Deep Sleep...");
    esp_sleep_enable_timer_wakeup(SLEEP_TIME); // Modo deep sleep para consumir menos enegría
    esp_deep_sleep_start();
}