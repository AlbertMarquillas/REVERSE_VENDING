#include "lora_ttn.h"

// Añadir claves de TTN
static const u1_t PROGMEM APPEUI[8] = { 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 };
static const u1_t PROGMEM DEVEUI[8] = { 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02 };
static const u1_t PROGMEM APPKEY[16] = { 
  0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 
  0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF 
};

// Funciones necesarias por LMIC para acceder a los identificadores
void os_getArtEui (u1_t* buf) { memcpy_P(buf, APPEUI, 8); }
void os_getDevEui (u1_t* buf) { memcpy_P(buf, DEVEUI, 8); }
void os_getDevKey (u1_t* buf) { memcpy_P(buf, APPKEY, 16); }

// Pines adaptados a tu ESP32
const lmic_pinmap lmic_pins = {
    .nss = 5,                      // LORA_CS
    .rxtx = LMIC_UNUSED_PIN,
    .rst = 14,                     // LORA_RST
    .dio = {2, LMIC_UNUSED_PIN, LMIC_UNUSED_PIN}, // LORA_IRQ
};

static osjob_t sendjob;

// Variable de estado para saber si ya está unido a TTN
static bool joined = false;

void onEvent(ev_t ev) {
    switch(ev) {
        case EV_JOINED:
            Serial.println("Se ha unido a TTN!");
            joined = true;
            break;
        case EV_JOIN_FAILED:
            Serial.println("Error intentando conectar a TTN");
            break;
        case EV_TXCOMPLETE:
            Serial.println("Mensaje enviado a TTN");
            break;
        default:
            Serial.print("Evento LMIC: ");
            Serial.println(ev);
            break;
    }
}

void LoRaModule::setup_lora() {
    Serial.println("Iniciando LoRaWAN (TTN)");

    os_init();               // Inicializa LMIC
    LMIC_reset();            // Resetea estado interno

    // Desactiva transmisiones automáticas y link-checks
    LMIC_setLinkCheckMode(0);

    // Establece región (868 MHz para Europa)
    LMIC_setDrTxpow(DR_SF7, 14);

    // Comienza unión OTAA
    LMIC_startJoining();
}

void LoRaModule::send_data(const String& message) {
    if (!joined) {
        Serial.println("No unido a TTN, no se puede enviar.");
        return;
    }

    if (LMIC.opmode & OP_TXRXPEND) {
        Serial.println("Envio en progreso, espera...");
        return;
    }

    // Convierte mensaje a bytes
    int len = message.length();
    uint8_t payload[len];
    message.getBytes(payload, len + 1);

    // Envía los datos al puerto 1 sin confirmación (puedes usar otros puertos o confirmar si quieres)
    LMIC_setTxData2(1, payload, len, 0);
    Serial.println("Enviando mensaje con exito a TTN: " + message);
}

bool LoRaModule::is_joined() {
    return joined;
}
