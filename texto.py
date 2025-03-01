import pywhatkit as kit
import cv2

import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import random

import pywhatkit




wikipedia.set_lang("es")  
engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")

engine.setProperty("voice",voices[0].id)

def Decir(audio):
	engine.say(audio)
	engine.runAndWait()

def Saludar():
	hour = int(datetime.datetime.now().hour)
	if(hour>=0 and hour<12):
		Decir("Buenos dias!")
	elif(hour>=12 and hour<18):
		Decir("Buenas tardes!")
	else:
		Decir("Buenas noches!")

	Decir("Mi nombre es stich22, por favor deme el texto para escribir?")

def Consulta():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("Escuchando...")
		r.pause_threshould = 1
		audio = r.listen(source)

	try:
		print("Analizando...")
		query = r.recognize_google(audio, language="es-ES")
		print(f"El usuario dijo: {query}\n")
		
        
		kit.text_to_handwriting(f"{query}\n", save_to="archi.png")
		img = cv2.imread("archi.png")
		cv2.namedWindow('image',WINDOW_NORMAL)
		cv2.resizeWindow('image', 900,900)
		cv2.imshow("el titulo", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	except Exception as ex:
		print("Repita por favor...")
		return "None"

	return query



def main():
	Saludar()
	while True:
		query = Consulta().lower()



main()		


