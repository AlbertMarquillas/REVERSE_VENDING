#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

class Sensors {
public:
    Sensors() {};
    void setup_sensors();
    float read_ultrasonic();
    float read_load_cell();
    bool read_inductive_sensor();
    int read_light_sensor();
};

#endif
