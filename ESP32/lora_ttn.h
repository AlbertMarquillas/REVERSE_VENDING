#ifndef LORA_COM_H
#define LORA_COM_H

#include <lmic.h>
#include <hal/hal.h>
#include <SPI.h>

class LoRaModule {
public:
    void setup_lora();
    void send_data(const String& message); // Envia mensaje
    bool is_joined(); // Para verificar si ya está unido a la red TTN
};

#endif
