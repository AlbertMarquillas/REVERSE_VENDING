import paho.mqtt.client as mqtt
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

MQTT_BROKER = "192.168.1.87"
MQTT_TOPIC = "sensor/datos"
MQTT_RESPONSE_TOPIC = "sensor/comandos"

# Cargar modelo de PyTorch
model = torch.jit.load("model.ts")
model.eval()

def detect_object():
    return True  # Aquí iría la lógica real de detección

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error al capturar imagen")
        return None
    filename = "captura.jpg"
    cv2.imwrite(filename, frame)
    return filename

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    predicted_class = output.argmax(1).item()
    return "correcto" if predicted_class == 1 else "incorrecto"

def on_message(client, userdata, msg):
    print(f"Mensaje recibido: {msg.payload.decode()}")
    if detect_object():
        image_path = capture_image()
        if image_path:
            classification = classify_image(image_path)
            print(f"Clasificación: {classification}")
            if classification == "correcto":
                client.publish(MQTT_RESPONSE_TOPIC, "parar")

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, 1883, 60)
client.subscribe(MQTT_TOPIC)

client.loop_start()
while True:
    time.sleep(0.5)
