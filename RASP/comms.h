#ifndef COMMS_H
#define COMMS_H

#include <string>
#include <mosquitto.h>

class Comms {
public:
    Comms(const std::string& broker, int port, const std::string& topic, const std::string& response_topic);
    ~Comms();
    void connect();
    void loop();
    void send_message(const std::string& message);
    void set_callback(void (*callback)(const std::string&));
private:
    std::string broker;
    int port;
    std::string topic;
    std::string response_topic;
    struct mosquitto* mosq;
    static void message_callback(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg);
    static void (*user_callback)(const std::string&);
};

#endif
