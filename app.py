
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi IP Semester 6",
    page_icon="🎓",
    layout="centered"
)

# Memuat model
model = joblib.load("model_prediksi_ip_semester6.pkl")

# Judul aplikasi
st.title("🎓 Prediksi IP Semester 6")
st.write(
    "Aplikasi ini menggunakan metode Artificial Neural Network "
    "(MLP) untuk memprediksi IP mahasiswa semester 6 berdasarkan "
    "nilai semester 1 sampai 5."
)

st.divider()

st.subheader("Masukkan IP Semester")

# Input nilai
semester_1 = st.number_input("IP Semester 1", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
semester_2 = st.number_input("IP Semester 2", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
semester_3 = st.number_input("IP Semester 3", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
semester_4 = st.number_input("IP Semester 4", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
semester_5 = st.number_input("IP Semester 5", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

# Tombol prediksi
if st.button("Prediksi IP Semester 6", type="primary"):

    data_input = pd.DataFrame([{
        "semester_1": semester_1,
        "semester_2": semester_2,
        "semester_3": semester_3,
        "semester_4": semester_4,
        "semester_5": semester_5
    }])

    hasil_prediksi = model.predict(data_input)[0]

    # Membatasi tampilan prediksi pada rentang IP 0-4
    hasil_prediksi = np.clip(hasil_prediksi, 0, 4)

    st.success(f"Prediksi IP Semester 6: **{hasil_prediksi:.2f}**")

    st.caption(
        "Hasil ini merupakan estimasi dari model AI, "
        "bukan nilai akademik resmi."
    )
