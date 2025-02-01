import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import time

# Load models
human_model = YOLO("yolo11n.pt")
violence_model_path = "modelv3L-84.35-noatt.keras"
violence_model = load_model(violence_model_path, safe_mode=False)

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["Non Kekerasan", "Kekerasan"]

def detect_human(frame):
    """Deteksi manusia dalam frame dan gambar bounding box."""
    results = human_model(frame, classes=[0])
    annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return len(results[0].boxes) > 0, annotated_frame  # True jika ada manusia, dan frame dengan bounding box

def predict_video(video_file_path):
    """Memprediksi apakah video mengandung kekerasan atau tidak."""
    video_reader = cv2.VideoCapture(video_file_path)
    frames_list = []
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    video_reader.release()
    
    if len(frames_list) < SEQUENCE_LENGTH:
        return "Error: Video terlalu pendek untuk diproses"
    
    predicted_probs = violence_model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_probs)
    return CLASSES_LIST[predicted_label], predicted_probs[predicted_label]  # Return label dan probabilitas

# Streamlit UI
st.title("Deteksi Kekerasan dalam Video")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
if uploaded_video:
    st.video(uploaded_video)
    if st.button("Start Detection"):
        temp_video_path = f"temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        video_reader = cv2.VideoCapture(temp_video_path)
        frame_count = 0
        log_messages = []
        model_run_frames = []  # Menyimpan frame di mana model kekerasan dijalankan
        placeholder = st.empty()
        sidebar_log = st.sidebar.empty()
        
        while True:
            success, frame = video_reader.read()
            if not success:
                break
            frame_count += 1
            human_detected, annotated_frame = detect_human(frame)
            
            if human_detected:
                log_messages.append(f"Frame {frame_count}: Manusia terdeteksi, menjalankan model deteksi kekerasan...")
                model_run_frames.append(frame_count)  # Catat frame di mana model dijalankan
                print(f"[INFO] Frame {frame_count}: Model kekerasan dijalankan pada {time.time()}")
            else:
                log_messages.append(f"Frame {frame_count}: Tidak ada manusia")
                print(f"[INFO] Frame {frame_count}: Model kekerasan tidak dijalankan")
            
            # Update tampilan video & log secara real-time
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")
            sidebar_log.write("\n".join(log_messages[-10:]))
            
            time.sleep(0.05)  # Delay untuk tampilan live
        
        video_reader.release()
        
        # Setelah semua frame diproses, jalankan model deteksi kekerasan pada seluruh video
        with st.spinner("Memproses..."):
            final_result, confidence = predict_video(temp_video_path)
        
        # Tampilkan hasil akhir dengan informasi tambahan
        st.success(f"Hasil Akhir: {final_result}")
        st.write(f"Model modelv3L-84.35-noatt.keras berjalan pada frame:\n{model_run_frames}")
        st.write(f"Akurasi Prediksi: {confidence * 100:.2f}%")
