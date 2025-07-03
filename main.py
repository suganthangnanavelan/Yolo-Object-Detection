import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import time
from ultralytics import YOLO

from src.detect_image import detect_image
from src.detect_video import detect_video_frame
from src.detect_realtime import detect_realtime_frame

st.set_page_config(page_title="YOLOv8 Detector", layout="wide", page_icon="🔍")
MODEL_PATH = "models/yolov8n.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 22px;
        padding: 0.5rem 1.5rem;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 0.5em 1.2em;
        border-radius: 6px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🔍 YOLOv8 Object Detection Dashboard</h1><hr>", unsafe_allow_html=True)

st.markdown("### 🎯 Confidence Threshold")
conf_col = st.columns([2, 4, 2])[1]
with conf_col:
    conf = st.slider("Select confidence", 0.1, 1.0, 0.4, 0.05)

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Image"

tabs = st.tabs(["📷 Image", "🎥 Video", "💻 Webcam"])
current_tab = "Image"

if tabs[0]:
    current_tab = "Image"
elif tabs[1]:
    current_tab = "Video"
elif tabs[2]:
    current_tab = "Webcam"

if current_tab != st.session_state.active_tab:
    st.session_state.webcam_active = False
    st.session_state.active_tab = current_tab

# ---------------------- IMAGE TAB ----------------------
with tabs[0]:
    st.subheader("Upload an image")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_upload")

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        result_img = detect_image(image, conf=conf)
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.image(result_img, caption="Detection Result", width=400)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- VIDEO TAB ----------------------
with tabs[1]:
    st.subheader("Upload a video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="video_upload")

    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

        st.info("🔍 Detecting...")
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_img = detect_video_frame(model, frame, conf)
            stframe.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
            stframe.image(result_img, channels="BGR", width=400)
            stframe.markdown("</div>", unsafe_allow_html=True)
        cap.release()

# ---------------------- WEBCAM TAB ----------------------
with tabs[2]:
    st.subheader("Webcam Detection")
    st.warning("⚠️ Works only in local environment (not on Streamlit Cloud)")

    toggle_btn = st.button("▶️ Start Webcam" if not st.session_state.webcam_active else "⏹ Stop Webcam")

    if toggle_btn:
        st.session_state.webcam_active = not st.session_state.webcam_active

    webcam_placeholder = st.empty()

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not accessible.")
            st.session_state.webcam_active = False
        else:
            st.info("Streaming webcam. Press stop to end.")
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Couldn't read frame.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = model(frame_rgb, conf=conf)
                annotated = result[0].plot()

                webcam_placeholder.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
                webcam_placeholder.image(annotated, channels="BGR", width=400)
                webcam_placeholder.markdown("</div>", unsafe_allow_html=True)
                time.sleep(0.03)

            cap.release()
            webcam_placeholder.empty()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2025 YOLOv8 Detector • Built with ❤️ using Streamlit + Ultralytics</div>",
    unsafe_allow_html=True
)
