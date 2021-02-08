import streamlit as st
from multiapp import MultiApp
from apps import theory, practise, example

app = MultiApp()

# Add all your application here
app.add_app("Теория", theory.app)
app.add_app("Пример", example.app)
app.add_app("Практика", practise.app)

# The main app
app.run()
