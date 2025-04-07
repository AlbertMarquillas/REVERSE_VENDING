#include "sensors.h"
#include <Arduino.h>
#include "DHT.h"

#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define TRIG_PIN 5
#define ECHO_PIN 18
#define LOAD_PIN 34
#define FORCE_PIN 35
#define INDUCTIVE_PIN 32
#define LIGHT_PIN 33
#define CUT_SENSOR_PIN 25

void Sensors::setup_sensors() {
    // Inicializaciones de pines para los diferentes sensores
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(INDUCTIVE_PIN, INPUT);
    pinMode(LIGHT_PIN, INPUT);
    pinMode(CUT_SENSOR_PIN, INPUT);
    dht.begin();
}

float Sensors::read_temperature() {
    // Devuelve la temperatura del DHT11
    return dht.readTemperature();
}

float Sensors::read_humidity() {
    // Devuelve la humedad del DHT11
    return dht.readHumidity();
}

float Sensors::read_ultrasonic() {
    // Devuelve la distancia detectada por el sensor de ultrasonidos
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH);
    return duration * 0.034 / 2;
}

float Sensors::read_load_cell() {
    // Devuelve el peso detectado por la celda de carga
    return analogRead(LOAD_PIN) * (5.0 / 4095.0);
}

bool Sensors::read_inductive_sensor() {
    // Devuelve la lectura del sensor inductivo
    return digitalRead(INDUCTIVE_PIN);
}

int Sensors::read_light_sensor() {
    // Devuelve el valor de luz ambiente
    return analogRead(LIGHT_PIN);
}

bool Sensors::read_cut_sensor() {
    // Devuelve la lectura del sensor de corte
    return digitalRead(CUT_SENSOR_PIN);
}