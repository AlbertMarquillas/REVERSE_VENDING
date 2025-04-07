#include "lora_com.h"
#include <LoRa.h>

#define LORA_CS    5
#define LORA_RST   14
#define LORA_IRQ   2

void LoRaModule::setup_lora() {
    LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);
    if (!LoRa.begin(868E6)) {
        Serial.println("Fallo al iniciar LoRa");
        while (1);
    }
    Serial.println("LoRa inicializado");
}

void LoRaModule::send_data(String message) {
    LoRa.beginPacket();
    LoRa.print(message);
    LoRa.endPacket();
    Serial.println("Mensaje enviado por LoRa: " + message);
}