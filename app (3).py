import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os
import tempfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="Aerial Damage AI", page_icon="🌪️", layout="wide")

# --- 1. CACHE THE MODELS (Massive Speed Boost) ---
@st.cache_resource
def load_models():
    standard_model = YOLO('best.pt')
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.30,
        device="cpu" # Safe for free cloud deployment
    )
    return standard_model, sahi_model

standard_model, sahi_model = load_models()

# --- 2. BUILD THE UI ---
st.title("🌪️ Aerial Disaster Response AI")
st.markdown("Upload satellite/drone imagery. The AI uses **SAHI (Slicing Aided Hyper Inference)** to detect damaged vs. safe buildings.")

# Create Tabs
tab1, tab2 = st.tabs(["📸 Image Detection (High Accuracy SAHI)", "🎥 Video Detection (Real-Time YOLO)"])

# --- TAB 1: IMAGE PROCESSING ---
with tab1:
    uploaded_image = st.file_uploader("Upload Drone Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # 🛠️ THE UI UPGRADE: Create two side-by-side columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        # Put the button in the middle
        st.markdown("---")
        if st.button("Analyze Damage", type="primary", use_container_width=True):
            with st.spinner("SAHI is slicing and analyzing the image..."):
                img_array = np.array(image)
                
                # Run SAHI
                result = get_sliced_prediction(
                    img_array,
                    sahi_model,
                    slice_height=320,
                    slice_width=320,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                
                # SAHI Bug Fix: Save to temp folder, read with OpenCV
                os.makedirs("temp_preds", exist_ok=True)
                result.export_visuals(export_dir="temp_preds", file_name="temp_result", text_size=0.5, rect_th=2)
                
                annotated_img = cv2.imread("temp_preds/temp_result.png")
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.markdown("### Damage Assessment")
                    st.image(annotated_img, use_container_width=True)
                    st.success("Analysis Complete!")

# --- TAB 2: VIDEO PROCESSING ---
with tab2:
    uploaded_video = st.file_uploader("Upload Drone Video (MP4)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        if st.button("Process Video", type="primary"):
            with st.spinner("YOLOv8 is processing the video frame by frame..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                results = standard_model.predict(
                    source=tfile.name, conf=0.35, iou=0.45, agnostic_nms=True, line_width=2, save=True
                )
                
                save_dir = results[0].save_dir
                video_name = os.path.basename(tfile.name)
                output_path = os.path.join(save_dir, video_name.rsplit('.', 1)[0] + '.avi')
                
                st.success("Video Processing Complete!")
                st.video(output_path)
