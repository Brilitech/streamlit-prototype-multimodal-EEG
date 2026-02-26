import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Identifikasi Fokus Belajar Mahasiswa")
uploaded_file = st.file_uploader("Upload file EEG & BPM (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview Data", data.head())

    # Misal ada visualisasi
    st.line_chart(data['bpm'])  # atau EEG
