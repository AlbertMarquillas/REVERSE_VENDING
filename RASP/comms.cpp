#include "comms.h"
#include <iostream>

void (*Comms::user_callback)(const std::string&) = nullptr;

Comms::Comms(const std::string& broker, int port, const std::string& topic, const std::string& response_topic)
    : broker(broker), port(port), topic(topic), response_topic(response_topic) {
    mosquitto_lib_init();
    mosq = mosquitto_new(nullptr, true, this);
    mosquitto_message_callback_set(mosq, message_callback);
}

Comms::~Comms() {
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
}

void Comms::connect() {
    mosquitto_connect(mosq, broker.c_str(), port, 60);
    mosquitto_subscribe(mosq, nullptr, topic.c_str(), 0);
    mosquitto_loop_start(mosq);
}

void Comms::loop() {
    mosquitto_loop(mosq, -1, 1);
}

void Comms::send_message(const std::string& message) {
    mosquitto_publish(mosq, nullptr, response_topic.c_str(), message.size(), message.c_str(), 0, false);
}

void Comms::set_callback(void (*callback)(const std::string&)) {
    user_callback = callback;
}

void Comms::message_callback(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg) {
    if (user_callback) {
        user_callback(std::string((char*)msg->payload, msg->payloadlen));
    }
}
