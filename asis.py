from flask import Flask, render_template, jsonify, request
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser
import datetime
import os
import pywhatkit
import threading
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import io
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Bloqueo para evitar solapamientos en la función hablar
speak_lock = threading.Lock()

# Cargar el modelo preentrenado ResNet50
model = YOLO("yolov8m.pt")

def hablar(texto):
    """Convierte texto en voz y lo reproduce."""
    with speak_lock:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("voice", engine.getProperty("voices")[0].id)
        engine.say(texto)
        engine.runAndWait()
        engine.stop()

def escuchar():
    """Escucha y reconoce la voz del usuario."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    try:
        comando = recognizer.recognize_google(audio, language="es-ES")
        print(f"Usuario: {comando}")
        return comando.lower()
    except sr.UnknownValueError:
        print("No entendí el comando.")
        return None
    except sr.RequestError:
        print("Error en el reconocimiento de voz.")
        return None

@app.route("/")
def index():
    """Renderiza la página principal."""
    return render_template("index.html")

@app.route("/iniciar", methods=["POST"])
def iniciar():
    """Inicia el asistente con un saludo y activa la escucha."""
    hablar("Hola, soy Stich22, tu asistente. ¿En qué puedo ayudarte?")
    return jsonify({"mensaje": "Asistente iniciado. Presiona 'Escuchar' para hablar."})

@app.route("/escuchar", methods=["POST"])
def escuchar_route():
    """Inicia la escucha del usuario y devuelve el comando."""
    comando = escuchar()
    if comando:
        return jsonify({"comando": comando})
    else:
        return jsonify({"error": "No entendí el comando."})

@app.route("/procesar", methods=["POST"])
def procesar():
    """Procesa el comando del usuario y responde."""
    comando = escuchar()
    if not comando:
        return jsonify({"mensaje": "No entendí, intenta de nuevo."})
    
    respuesta = ""
    
    if "wikipedia" in comando:
        hablar("Buscando en Wikipedia...")
        comando = comando.replace("buscar en wikipedia", "")
        resultado = wikipedia.summary(comando, sentences=1)
        hablar(resultado)
        respuesta = resultado
    elif "abre google" in comando:
        hablar("Abriendo Google")
        webbrowser.open("https://www.google.com")
        respuesta = "Google abierto."
    elif "senati" in comando:
        hablar("Abriendo la página de SENATI")
        webbrowser.open("https://www.senati.edu.pe")
        respuesta = "Página de SENATI abierta."
    elif "hora" in comando:
        hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
        hablar(f"La hora actual es {hora_actual}")
        respuesta = f"La hora actual es {hora_actual}."
    elif "youtube" in comando:
        cancion = comando.replace("reproduce en youtube", "").strip()
        hablar(f"Reproduciendo {cancion} en YouTube")
        pywhatkit.playonyt(cancion)
        respuesta = f"Reproduciendo {cancion} en YouTube."
    elif "detener" in comando or "salir" in comando:
        hablar("Cerrando el asistente. Hasta luego.")
        os._exit(0)
    else:
        hablar("No reconozco ese comando.")
        respuesta = "No reconozco ese comando."
    
    return jsonify({"mensaje": respuesta})

@app.route("/procesar_imagen", methods=["POST"])
def procesar_imagen():
    """Procesa una imagen y devuelve la clasificación con YOLOv8."""
    try:
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No se recibió imagen"})

        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image is None or image.size == (0, 0):
            return jsonify({"error": "La imagen es inválida o está vacía."})

        # Convertir a formato compatible con OpenCV
        image_cv = np.array(image.convert("RGB"))  # Convertir a RGB
        results = model(image_cv)  # Pasar la imagen a YOLOv8

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]  # Obtener el nombre del objeto detectado
                detections.append(class_name)

        if detections:
            result_text = "He detectado: " + ", ".join(detections)
            hablar(result_text)
            return jsonify({"mensaje": result_text, "detecciones": detections})
        else:
            return jsonify({"mensaje": "No se detectó ningún objeto en la imagen."})

    except Exception as e:
        print(f"Error procesando la imagen: {str(e)}")
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"})


if __name__ == "__main__":
    hablar("Iniciando el asistente.")
    app.run(debug=True, threaded=True)






