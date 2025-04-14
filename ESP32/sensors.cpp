#include "sensors.h"
#include <Arduino.h>
#include "HX711.h"

#define TRIG_PIN 16
#define ECHO_PIN 17
#define INDUCTIVE_PIN 32
#define LIGHT_PIN 33

#define DT  26
#define SCK 25

HX711 balanza;
float factor_calibracion = -7050.0; 

void Sensors::setup_sensors() {
    // Inicializaciones de pines para los diferentes sensores
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(INDUCTIVE_PIN, INPUT);
    pinMode(LIGHT_PIN, INPUT);

    balanza.begin(DT, SCK);
    balanza.set_scale(factor_calibracion);
    balanza.tare();
}

float Sensors::read_ultrasonic() {
    // Devuelve la distancia detectada por el sensor de ultrasonidos
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH);
    return duration/59; // Escalado a cm
}

float Sensors::read_load_cell() {
    // Devuelve el peso detectado por la celda de carga
    return balanza.get_units(10); // Devuelve media de 10 lecturas
}

bool Sensors::read_inductive_sensor() {
    // Devuelve la lectura del sensor inductivo
    return digitalRead(INDUCTIVE_PIN);
}

int Sensors::read_light_sensor() {
    // Devuelve el valor de luz ambiente
    return (analogRead(LIGHT_PIN) / 4095.0) * 100.0; // Devuelve porcentaje
}
