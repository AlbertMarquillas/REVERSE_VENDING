#ifndef COMMS_H
#define COMMS_H

#include <Arduino.h>
#include "sensors.h"

class Comms {
public:
    void setup_wifi();
    void setup_mqtt();
    bool mqtt_connected();
    void reconnect_mqtt();
    void mqtt_loop();
    void send_sensor_data(Sensors& sensors);
    bool check_mqtt_message();
    String get_mqtt_message();
    static void callback_mqtt(char* topic, byte* payload, unsigned int length);
};

#endif