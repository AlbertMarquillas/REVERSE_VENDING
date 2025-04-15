#include "lora_com.h"
#include <LoRa.h>

// Definición de pines utilizados por el módulo LoRa
#define LORA_CS    5    // Pin Chip Select (SPI)
#define LORA_RST   14   // Pin Reset del módulo LoRa
#define LORA_IRQ   2    // Pin de interrupción (IRQ) del LoRa

// Inicializa el módulo LoRa con la configuración de pines y la frecuencia
void LoRaModule::setup_lora() {
    // Configura los pines utilizados por el módulo LoRa
    LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);
    
    // Inicia la comunicación LoRa a 868 MHz (frecuencia típica en Europa)
    if (!LoRa.begin(868E6)) {
        Serial.println("Fallo al iniciar LoRa");
        while (1);  // Detiene el programa si no se pudo inicializar LoRa
    }
    Serial.println("LoRa inicializado");
}

// Envía un mensaje de texto a través de LoRa
void LoRaModule::send_data(String message) {
    LoRa.beginPacket();     // Inicia un nuevo paquete de datos
    LoRa.print(message);    // Escribe el mensaje en el paquete
    LoRa.endPacket();       // Finaliza y envía el paquete
    Serial.println("Mensaje enviado por LoRa: " + message);
}
