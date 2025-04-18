#include <Arduino.h>
#include "comms.h"
#include "sensors.h"
#include "lora_com.h"

#define SLEEP_TIME 10e6 // 10 segundos en microsegundos
#define SLEEP_OP_MODE 1e6 // 1 segundo en microsegundos

Comms comms;
Sensors sensors;
LoRaModule lora;

bool op_mode = false;

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
        if(message["info"]=="OP_MODE") // Comprovacion mensaje
		op_mode = true;
	elif(message=="REST_MODE")
		op_mode = false;
	elif(message=="CALIBRATE")
            sensors.calibrate();
        else
            lora.send_data(message);  // Enviar el mensaje por LoRa
    }
    esp_sleep_enable_timer_wakeup(op_mode?SLEEP_OP_MODE:SLEEP_TIME); // Modo deep sleep
    esp_deep_sleep_start();
}
