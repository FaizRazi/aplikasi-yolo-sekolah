import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("🔥 YOLOv8 Realtime Webcam (Streamlit)")

# Load model (cache biar cepat)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# ==============================
# VIDEO TRANSFORMER (REALTIME)
# ==============================
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Ambil frame dari webcam
        img = frame.to_ndarray(format="bgr24")

        # YOLO detection
        results = model(img, conf=0.3)

        # Gambar bounding box
        annotated = results[0].plot()

        return annotated


# ==============================
# STREAM WEBCAM
# ==============================
webrtc_streamer(
    key="yolo",
    video_transformer_factory=YOLOVideoTransformer
)
