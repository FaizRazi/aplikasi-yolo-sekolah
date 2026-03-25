import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# ==============================
# 1. Load Model YOLO
# ==============================
# Using st.cache_resource to load the model only once for Streamlit performance
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# ==============================
# 3. Jalankan Deteksi (Modified for Streamlit)
# ==============================
def run_detection_streamlit():
    st.title("Object Detection with YOLOv8 and Streamlit")

    # Introduce Streamlit camera input widget
    captured_image = st.camera_input("Ambil foto untuk deteksi objek")

    if captured_image is not None:
        try:
            st.text("Melakukan deteksi...")

            # Convert Streamlit's UploadedFile (bytes-like object) to PIL Image
            pil_image = Image.open(captured_image)

            # Perform object detection
            results = model(pil_image)

            # Display results
            for r in results:
                # r.plot() returns an image with detections drawn, typically BGR format
                im_array = r.plot()
                # Convert BGR (OpenCV format) to RGB for PIL/Streamlit display
                im = Image.fromarray(im_array[..., ::-1])
                st.image(im, caption="Hasil Deteksi", use_column_width=True)

            st.success("Deteksi Selesai.")

        except Exception as err:
            st.error(f"Terjadi kesalahan saat deteksi: {err}")

# ==============================
# 4. Eksekusi Streamlit App
# ==============================
if __name__ == "__main__":
    run_detection_streamlit()
