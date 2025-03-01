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

	Decir("Mi nombre es STICH22, soy el asistente del profesor Wilbert, por favor digame, en que le puedo ayudar?")

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
	except Exception as ex:
		print("Repita por favor...")
		return "None"

	return query

def main():
	Saludar()
	entra = True
	while entra:
		query = Consulta().lower()

		if("wikipedia" in query):
			Decir("Buscando en  wikipedia...")
			query = query.replace("buscar en wikipedia","")
			results = wikipedia.summary(query, sentences=1)
			Decir("De acuerdo a wikipedia")
			print(results)
			Decir(results)
		elif("google" in query):
			query = query.replace("buscar en google","")
			webbrowser.open(query)
		elif("música" in query):
			music_dir = "d:\\python\\Music"
			songs = os.listdir(music_dir)
			os.startfile(os.path.join(music_dir, songs[random.randint(0,3)]))
		elif("dame el tiempo" in query):
			strTime = datetime.datetime.now().strftime("%H:%M:%S")
			strFecha = datetime.datetime.now().strftime('%d-%m-%Y')
			print(strFecha)
			print(strTime)
			Decir(f"la fecha actual es {strFecha}")
			Decir(f"y la hora es {strTime}")
		elif("reproduce" in query):
			song = query.replace("reproduce en youtube","")
			Decir("reproduciendo: " + song)
			pywhatkit.playonyt(song)
		elif("el curso" in query):
			strDatos="el curso es estructurda de"
			with open("d:/python/datoscurso.txt","r") as archivo:
				for linea in archivo:
					print(linea)
					Decir(linea)
		elif("detener" in query):
			entra=False

main()