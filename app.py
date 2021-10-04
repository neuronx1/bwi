import streamlit as st
import time
import ai


st.header("BWI")
st.subheader("Die Wetterstation für jeder Mann")
st.write("Dieses Projekt wurde von Simon und Florian erstellt")
menu = ["Startseite"]

start_input = st.text_input("Gib deinen Ort ein")
startButton = st.button("Vorhersage")
if startButton:
   st.write('Wettervorhersage wird berechnet')
   temp,nied=ai.main(start_input)
   if temp!=False:
      st.write('Die Durchschnittstemperatur in {} wird {} °Celcius sein'.format(start_input,temp))
      st.write('Die Niederschlagsmenge in {} wird {} mm sein'.format(start_input,nied))
   else:
      st.write('Der Städtename wurde nicht erkannt')
