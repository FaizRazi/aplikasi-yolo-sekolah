import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("🔥 YOLOv8 Realtime Webcam (Streamlit)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")

        results = model(img, conf=0.3)

        annotated = results[0].plot()
        annotated = annotated[:, :, ::-1]  # convert ke RGB

        return annotated


webrtc_streamer(
    key="yolo",
    video_transformer_factory=YOLOVideoTransformer
)
