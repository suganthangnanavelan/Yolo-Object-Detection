import os
import time
import tempfile
import base64

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="Object Detector", layout="wide", page_icon="./assests/icon.png")

st.markdown("""
<style>
    .stDeployButton {display: none;}
    .stAlert {display: none;}
    .stSpinner {display: none;}
    .stToast {display: none;}
    .stSuccess {display: none;}
    .stInfo {display: none;}
    .stWarning {display: none;}
    .stError {display: none;}
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #43cea2 0%, #f9f586 50%, #185a9d 100%);
        color: #102542;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .confidence-section {
        background: linear-gradient(135deg, #e3ffe7, #d9e7ff);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    .confidence-section h4 {
        margin: 0 0 1rem 0;
        font-size: 20px;
        color: #102542;
    }
    .stButton>button {
        background: linear-gradient(45deg, #43cea2, #185a9d);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(67, 206, 162, 0.3);
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(67, 206, 162, 0.4);
        background: linear-gradient(45deg, #185a9d, #43cea2);
    }
    .stButton>button:disabled {
        background: #cccccc;
        color: #666666;
        transform: none;
        box-shadow: none;
        opacity: 0.6;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px 12px 0 0;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        margin-right: 8px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #43cea2, #185a9d);
        color: white;
        border-color: #43cea2;
    }
    .stSlider > div > div > div > div {
        font-size: 16px;
        font-weight: 600;
    }
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
    }
    h4 {
        font-size: 24px;
        font-weight: 700;
        color: #333;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 26px;
        font-weight: 700;
        color: #333;
        margin-bottom: 1.5rem;
    }
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px dashed #43cea2;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div > div > div:hover {
        border-color: #185a9d;
        background: linear-gradient(135deg, #e9ecef, #dee2e6);
    }
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stProgress [data-testid="stProgress"] {
        background: linear-gradient(90deg, #43cea2, #185a9d);
        height: 8px;
        border-radius: 4px;
    }
    .footer {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        text-align: center;
        color: #666;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        margin: 1rem 0;
    }
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e9ecef;
        border-top: 4px solid #43cea2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = YOLO("models/yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

st.markdown('<div class="main-header"><h1>You Look Only Once</h1></div>', unsafe_allow_html=True)

model = load_model()
if model is None:
    st.stop()

st.markdown("""
    <style>
        .conf-box-wrapper {
            display: flex;
            justify-content: left;
            margin-bottom: 1rem;
        }
        .conf-box {
            background: linear-gradient(135deg, #43cea2 0%, #f9f586 50%, #185a9d 100%);
            padding: 0.6rem 1.6rem;
            border-radius: 8px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            color: #102542;
            text-align: center;
            font-size: 18px;
            font-weight: 650;
            font-family: 'Segoe UI', sans-serif;
            display: inline-block;
        }
    </style>
    <div class="conf-box-wrapper">
        <div class="conf-box">Confidence Threshold</div>
    </div>
""", unsafe_allow_html=True)

conf = st.slider(
    label="Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Minimum confidence for detections",
    key="conf_slider",
    label_visibility="collapsed"
)

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "video_processing" not in st.session_state:
    st.session_state.video_processing = False
if "processed_video_name" not in st.session_state:
    st.session_state.processed_video_name = None
if "processed_image_name" not in st.session_state:
    st.session_state.processed_image_name = None

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Live Webcam"])

with tab1:
    if st.session_state.video_processing:
        st.session_state.video_processing = False

    st.markdown("#### Upload an Image")
    uploaded_image = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        already_processed = st.session_state.processed_image_name == uploaded_image.name
        if already_processed:
            st.info("Image already processed. Click 'Process Image' to process again.")

        if st.button("Process Image", disabled=not uploaded_image):
            image = Image.open(uploaded_image).convert("RGB")
            try:
                results = model(np.array(image), conf=conf)
                annotated_image = Image.fromarray(results[0].plot())
                st.image(annotated_image, use_container_width=True)
                num_detections = len(results[0].boxes)
                if num_detections > 0:
                    st.markdown(f"**Found {num_detections} object{'s' if num_detections != 1 else ''}**")
                else:
                    st.markdown("**No objects detected - try lowering the confidence threshold**")
                st.session_state.processed_image_name = uploaded_image.name
            except Exception as e:
                st.markdown(f"**Detection error: {e}**")

with tab2:
    st.markdown("#### Upload a Video")
    uploaded_video = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        already_processed = st.session_state.processed_video_name == uploaded_video.name
        if already_processed:
            st.info("Video already processed. Click 'Process Video' to process again.")

        if st.button("Process Video", disabled=not uploaded_video):
            if st.session_state.webcam_running:
                st.session_state.webcam_running = False
                st.rerun()
            st.session_state.video_processing = True

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name

            st.markdown("**Processing video...**")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.markdown("**Could not open video file**")
                os.unlink(video_path)
                st.session_state.video_processing = False
                st.stop()

            stframe = st.empty()
            progress_bar = st.progress(0)

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 3 == 0:
                    results = model(frame, conf=conf)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            os.unlink(video_path)
            progress_bar.empty()
            st.markdown("**Video processing complete!**")
            st.session_state.video_processing = False
            st.session_state.processed_video_name = uploaded_video.name

with tab3:
    if st.session_state.video_processing:
        st.session_state.video_processing = False

    st.markdown("#### Real-time Webcam Detection")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Webcam", disabled=st.session_state.video_processing):
            st.session_state.webcam_running = True

    with col2:
        if st.button("Stop Webcam", disabled=st.session_state.video_processing):
            st.session_state.webcam_running = False

    if st.session_state.video_processing:
        st.info("Webcam disabled while processing video. Please wait for video processing to complete.")

    webcam_placeholder = st.empty()

    if st.session_state.webcam_running and not st.session_state.video_processing:
        with webcam_placeholder.container():
            st.markdown("""
            <div class="loading-spinner">
                <div class="spinner"></div>
                <div style="margin-left: 1rem;">
                    <h4>Starting webcam...</h4>
                    <p>Please allow camera access when prompted</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam. Please check your camera connection.")
            st.session_state.webcam_running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            time.sleep(2)

            try:
                frame_count = 0
                while st.session_state.webcam_running and not st.session_state.video_processing:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    if frame_count % 2 == 0:
                        results = model(frame, conf=conf, verbose=False)
                        annotated_frame = results[0].plot()
                        _, buffer = cv2.imencode(".jpg", annotated_frame)
                        img_b64 = base64.b64encode(buffer).decode("utf-8")

                        webcam_placeholder.markdown(
                            f"""
                            <div style="text-align: center;">
                                <img 
                                    src="data:image/jpeg;base64,{img_b64}" 
                                    style="
                                        height: 650px; 
                                        width: auto; 
                                        border-radius: 12px; 
                                        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
                                    "
                                >
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    frame_count += 1
                    time.sleep(0.05)

            except Exception as e:
                st.error(f"Webcam error: {e}")
            finally:
                cap.release()

    elif not st.session_state.webcam_running:
        webcam_placeholder.empty()

st.markdown("""
<style>
.footer-heading {
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    margin-bottom: 0;
    background: linear-gradient(135deg, #43cea2 0%, #f9f586 50%, #185a9d 100%);
    color: #102542;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.footer-heading h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    color: #102542;
}
</style>
<div class="footer-heading">
    <h1>Built with Lemon Juice</h1>
</div>
""", unsafe_allow_html=True)