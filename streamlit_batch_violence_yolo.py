import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
import tempfile

from ultralytics import YOLO
# Load YOLOv11 model
@st.cache_resource
def load_yolo_model():
    return YOLO("D:\\College_Stuff\\TA\\models\\yolo11n.pt")

# Load violence detection model
@st.cache_resource
def load_violence_model(model_name):
    model_path = MODEL_PATHS[model_name]
    return load_model(model_path, safe_mode=False)

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["Non Kekerasan", "Kekerasan"]

MODEL_PATHS = {
    "Model v3L": "D:\\College_Stuff\\TA\\models\\MobileNetv3L-1702L.keras",
    "Model v3S": "D:\\College_Stuff\\TA\\models\\modelv3s-81.39-noatt.keras",
    "Model v2": "D:\\College_Stuff\\TA\\models\\modelv2-89.30-0402.keras"
}

def detect_humans(frame, yolo_model):
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 = 'person' dalam YOLO
                return True
    return False

def predict_video(video_file_path, model, yolo_model, sequence_length=SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_list = []
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            break
        
        if detect_humans(frame, yolo_model):
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)
    
    video_reader.release()
    if len(frames_list) < sequence_length:
        sequence_length = len(frames_list) 
    
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    confidence = predicted_labels_probabilities[predicted_label]
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    return predicted_class_name, confidence

def main():
    st.sidebar.title("Model Settings")
    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
    
    with st.spinner(f"Loading {selected_model}..."):
        model = load_violence_model(selected_model)
        st.sidebar.success(f"{selected_model} loaded successfully!")
    
    with st.spinner("Loading YOLO model..."):
        yolo_model = load_yolo_model()
        st.sidebar.success("YOLO model loaded successfully!")
    
    uploaded_files = st.file_uploader("Upload videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)
    
    if uploaded_files:
        temp_files = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(file.read())
                temp_files.append({"path": tmp_file.name, "name": file.name})
        
        st.subheader("Batch Label Selection and Prediction")
        batch_true_label = st.radio("Select True Label for ALL Videos:", options=["Kekerasan", "Non Kekerasan"], key="batch_label")
        predict_button = st.button("Predict All Videos")
        
        if predict_button:
            results = []
            correct_count = 0
            total_count = len(temp_files)
            progress_bar = st.progress(0)
            
            for i, file_info in enumerate(temp_files):
                progress_bar.progress(int((i + 1) / total_count * 100))
                predicted_label, confidence = predict_video(file_info["path"], model, yolo_model)
                if predicted_label is None:
                    continue
                is_correct = (predicted_label == batch_true_label)
                if is_correct:
                    correct_count += 1
                results.append({
                    "Video Name": file_info["name"],
                    "True Label": batch_true_label,
                    "Predicted Label": predicted_label,
                    "Confidence": f"{confidence:.2%}",
                    "Status": "Correct" if is_correct else "Incorrect"
                })
            
            progress_bar.empty()
            st.session_state.prediction_results = {"results": results, "correct_count": correct_count, "total_count": total_count}
            st.session_state.has_predicted = True
        
        if st.session_state.has_predicted and st.session_state.prediction_results:
            results = st.session_state.prediction_results["results"]
            correct_count = st.session_state.prediction_results["correct_count"]
            total_count = st.session_state.prediction_results["total_count"]
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            accuracy = correct_count / total_count if total_count > 0 else 0
            col1.metric("Total Videos", f"{total_count}")
            col2.metric("Correct Predictions", f"{correct_count}")
            col3.metric("Accuracy", f"{accuracy:.2%}")
            df = pd.DataFrame(results)
            # Function to apply conditional styling per row
            def highlight_status(row):
                color = "rgba(0, 255, 0, 0.3)" if row["Status"] == "Correct" else "rgba(255, 0, 0, 0.3)"
                return [f"background-color: {color}"] * len(row)

            # Apply styling
            styled_df = df.style.apply(highlight_status, axis=1)

            # Show the styled DataFrame
            st.dataframe(styled_df, use_container_width=True)
        
        for file_info in temp_files:
            try:
                os.unlink(file_info["path"])
            except:
                pass

if __name__ == "__main__":
    main()
