import requests
import base64
import os
import datetime
import webbrowser
import wikipedia
import pywhatkit
from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

# Configuración de la API de Azure
AZURE_ENDPOINT = "https://trainner-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/75354010-a2ff-4a74-82d9-87a62ccaf7ae/classify/iterations/Iteration1/image"
AZURE_KEY = "2b596BVpBh6IuLayCrmft9GYfnJyuKvLw3dAwQ3as09bcutvqQ1TJQQJ99BBACYeBjFXJ3w3AAAIACOG0Urk"

def hablar(texto):
    """Convierte texto a voz."""
    try:
        engine = pyttsx3.init()
        engine.say(texto)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"Error en hablar(): {str(e)}")
        return False

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
    except Exception as e:
        print(f"Error inesperado en escuchar(): {str(e)}")
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
    data = request.get_json()
    comando = data.get("comando", "")
    
    if not comando:
        return jsonify({"mensaje": "No entendí, intenta de nuevo."})

    respuesta = ""

    if "wikipedia" in comando:
        hablar("Buscando en Wikipedia...")
        comando = comando.replace("buscar en wikipedia", "").strip()
        try:
            resultado = wikipedia.summary(comando, sentences=1)
            hablar(resultado)
            respuesta = resultado
        except wikipedia.exceptions.PageError:
            hablar("No encontré información en Wikipedia sobre eso.")
            respuesta = "No encontré información en Wikipedia sobre eso."
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
        respuesta = "Cerrando el asistente. Hasta luego."
    else:
        hablar("No reconozco ese comando.")
        respuesta = "No reconozco ese comando."

    return jsonify({"mensaje": respuesta})

@app.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    try:
        data = request.get_json()
        image_data = data.get("image")  # Recibe la imagen en base64

        if not image_data:
            return jsonify({"error": "No se recibió una imagen"}), 400

        # Decodificar la imagen base64 a binario
        image_binary = base64.b64decode(image_data.split(",")[1])

        # Enviar la imagen a la API de Azure
        headers = {
            "Prediction-Key": AZURE_KEY,
            "Content-Type": "application/octet-stream"
        }

        response = requests.post(AZURE_ENDPOINT, headers=headers, data=image_binary)
        result = response.json()

        # Extraer predicciones si están disponibles
        predictions = result.get("predictions", [])
        detecciones = [f"{p['tagName']}: {p['probability']:.2%}" for p in predictions]

        return jsonify({"mensaje": "Imagen procesada", "detecciones": detecciones})

    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 500

if __name__ == "__main__":
    # Asegurar que existe el directorio de templates
    os.makedirs("templates", exist_ok=True)
    
    # Ejecutar la aplicación
    app.run(debug=True)