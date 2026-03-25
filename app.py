import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi YOLO Sekolah", layout="centered")
st.title("🔍 Aplikasi Deteksi Objek Real-Time")
st.write("Gunakan kamera laptop untuk mendeteksi objek secara otomatis.")

# 1. Load Model YOLO (Versi Nano agar ringan di Cloud)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 2. Input Kamera
img_file_buffer = st.camera_input("Klik 'Take Photo' untuk deteksi")

if img_file_buffer is not None:
    # Proses Gambar
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Prediksi
    results = model.predict(cv2_img, conf=0.5)

    # Tampilkan Hasil
    for r in results:
        res_plotted = r.plot()
        st.image(res_plotted, caption='Hasil Analisis YOLO', channels="BGR", use_container_width=True)
        
        # Tampilkan daftar benda
        labels = [model.names[int(cls)] for cls in r.boxes.cls]
        if labels:
            st.success(f"Objek terdeteksi: {', '.join(set(labels))}")
        else:
            st.warning("Tidak ada objek yang dikenali.")
