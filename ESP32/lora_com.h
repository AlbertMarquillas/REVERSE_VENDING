#ifndef LORAMODULE_H
#define LORAMODULE_H

#include <LoRa.h>
#include <Arduino.h>

class LoRaModule {
  LoRaClass LoRa;
public:
  void setup_lora();
  void send_data(String message);
};

#endif