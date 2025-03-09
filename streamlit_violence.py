import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import time

# Load YOLO model
human_model = YOLO("D:\College_Stuff\TA\models\yolo11n.pt")

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["Non Kekerasan", "Kekerasan"]

# Model paths dictionary
MODEL_PATHS = {
    "Model v3L": "D:\College_Stuff\TA\models\MobileNetv3L-1702L.keras",
    "Model v3S": "D:\College_Stuff\TA\models\modelv3s-81.39-noatt.keras",
    "Model v2": "D:\College_Stuff\TA\models\modelv2-89.30-0402.keras"
}

def load_violence_model(model_name):
    """Load the selected violence detection model."""
    model_path = MODEL_PATHS[model_name]
    return load_model(model_path, safe_mode=False)

def detect_human(frame):
    """Deteksi manusia dalam frame dan gambar bounding box."""
    results = human_model(frame, classes=[0])
    annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return len(results[0].boxes) > 0, annotated_frame

def process_frame_for_violence(frame):
    """Memproses frame untuk input ke model deteksi kekerasan."""
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255
    return normalized_frame

def predict_violence(frames, violence_model):
    """Memprediksi kekerasan dari sequence frames dengan pengukuran waktu yang akurat."""
    if len(frames) < SEQUENCE_LENGTH:
        last_frame = frames[-1]
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(last_frame)
    elif len(frames) > SEQUENCE_LENGTH:
        frames = frames[-SEQUENCE_LENGTH:]
    
    frames_array = np.array(frames)
    batch_input = np.expand_dims(frames_array, axis=0)
    
    # Warm up prediction (untuk menghindari overhead first-time prediction)
    _ = violence_model.predict(batch_input)
    
    # Pengukuran waktu sebenarnya
    times = []
    num_runs = 1  # Lakukan beberapa kali untuk mendapatkan rata-rata yang stabil
    
    for _ in range(num_runs):
        start_time = time.perf_counter()  # Gunakan perf_counter untuk akurasi lebih tinggi
        predicted_probs = violence_model.predict(batch_input, verbose=0)[0]
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_prediction_time = sum(times) / len(times)
    predicted_label = np.argmax(predicted_probs)
    
    return CLASSES_LIST[predicted_label], predicted_probs[predicted_label], avg_prediction_time

def process_video(video_path, use_yolo, violence_model):
    """Proses video dengan pengukuran waktu yang terpisah dan mempertahankan bounding box."""
    video_reader = cv2.VideoCapture(video_path)
    all_frames = []
    yolo_frames = []
    frame_count = 0
    total_frames = 0
    total_filtered_frames = 0

    while True:
        success, frame = video_reader.read()
        if not success:
            break
        
        total_frames += 1  # Hitung semua frame dalam video
        processed_frame = process_frame_for_violence(frame)
        display_frame = frame.copy()
        
        if use_yolo == "Dengan YOLO":
            human_detected, annotated_frame = detect_human(frame)
            if human_detected:
                yolo_frames.append(processed_frame)
                total_filtered_frames += 1  # Hitung frame yang lolos filter YOLO
            display_frame = annotated_frame
        else:
            all_frames.append(processed_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:  # Update progress setiap 10 frame
            progress = int((frame_count / int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))) * 100)
            yield "progress", progress, display_frame
    
    video_reader.release()
    
    # Siapkan frames untuk prediksi
    frames_for_prediction = yolo_frames if use_yolo == "Dengan YOLO" else all_frames[-SEQUENCE_LENGTH:]
    
    # Pastikan kita memiliki tepat SEQUENCE_LENGTH frames
    if len(frames_for_prediction) < SEQUENCE_LENGTH:
        last_frame = frames_for_prediction[-1] if frames_for_prediction else all_frames[-1]
        while len(frames_for_prediction) < SEQUENCE_LENGTH:
            frames_for_prediction.append(last_frame)
    elif len(frames_for_prediction) > SEQUENCE_LENGTH:
        frames_for_prediction = frames_for_prediction[-SEQUENCE_LENGTH:]
    
    frames_array = np.array(frames_for_prediction)
    batch_input = np.expand_dims(frames_array, axis=0)
    
    # Warm up prediction
    _ = violence_model.predict(batch_input, verbose=0)
    
    # Pengukuran waktu yang lebih akurat
    start_time = time.perf_counter()
    predicted_probs = violence_model.predict(batch_input, verbose=0)[0]
    end_time = time.perf_counter()
    
    pred_time = end_time - start_time
    predicted_label = np.argmax(predicted_probs)
    
    yield "result", CLASSES_LIST[predicted_label], predicted_probs[predicted_label], pred_time, len(frames_for_prediction), total_frames, total_filtered_frames

# Streamlit UI
st.title("Deteksi Kekerasan dalam Video")

selected_model = st.selectbox(
    "Pilih Model Deteksi Kekerasan",
    list(MODEL_PATHS.keys())
)

use_yolo = st.radio(
    "Metode Deteksi",
    ["Dengan YOLO", "Tanpa YOLO"],
    index=0
)

violence_model = load_violence_model(selected_model)

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
if uploaded_video:
    st.video(uploaded_video)
    
    if st.button("Start Detection"):
        temp_video_path = f"temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        progress_bar = st.progress(0)
        frame_display = st.empty()
        
        # Process video
        for output in process_video(temp_video_path, use_yolo, violence_model):
            if output[0] == "progress":
                _, progress, frame = output
                progress_bar.progress(progress)
                frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            elif output[0] == "result":
                _, result, confidence, pred_time, num_frames, total_frames, total_filtered_frames = output
                
                st.success(f"Prediksi: {result}")
                st.write(f"Confidence score: {confidence * 100:.2f}%")
                st.write(f"Jumlah frame yang diproses: {num_frames}")
                #st.write(f"Waktu rata-rata runtime: {None} ms") #pred_time*1000:.2f
                
                # Tambahan informasi detail
                st.write("Detail Pengukuran:")
                st.write(f"- Model yang digunakan: {selected_model}")
                st.write(f"- Metode deteksi: {use_yolo}")
                st.write(f"- Total frame dalam video: {total_frames}")
                if use_yolo == "Dengan YOLO":
                    st.write(f"- Total frame setelah filtering YOLO: {total_filtered_frames}")
                    st.write(f"- Frame yang digunakan untuk prediksi: {num_frames} frames")
                else:
                    st.write(f"- Frame yang digunakan untuk prediksi: {num_frames} frames")
                st.write(f"- Sequence length: {SEQUENCE_LENGTH} frames")
