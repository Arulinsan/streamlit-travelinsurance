#import library

import streamlit as st
from web_function import load_data

from Tabs import home, predict, visualise

Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualisation" : visualise
}

# Membuat sidebar

st.sidebar.title("Navigasi")

#membuat Radio option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

#load dataset
df, x, y = load_data()

if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()