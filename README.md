<h1>VIDEO-BASED VIOLENCE PREDICTION WITH COMBINATION OF YOLOV11 AND MOBILENETV3</h1>
<b>This project is part of the final requirement for graduation in Informatics Engineering.</b><br><br>

<h2>📌 Project Overview</h2>This project aims to develop an AI model capable of detecting acts of violence in public surveillance videos efficiently and lightweightly. To achieve this, we leverage a combination of YOLOv11 for human detection and MobileNetV3 + Bi-LSTM for violence classification. The goal is to maintain a balance between accuracy and computational efficiency, making it suitable for real-time applications.

<h2>📜 Methodology</h2>
<ol>
  <li>Human Detection with YOLOv11</li>
  <ul>
    <li>Filters out frames that do not contain detected humans.</li>
    <li>Filtered frame are proceeds to pre-processing before handed by MobileNetv3 for violence prediction</li>
  </ul>
  <li>Feature Extraction with MobileNetV3</li>
  <ul>
    <li>Processes the extracted frames to obtain deep visual features.</li>
  </ul>
  <li>Temporal Analysis with Bi-LSTM</li>
  <ul>
    <li>Utilizes sequential frame information to classify violent vs. non-violent behavior.</li>
    <li>Helps capture motion patterns over time.</li>
  </ul>
</ol>

<h2>🛠️ Requirements</h2>
Make sure to install the necessary dependencies before running the project:

<code>pip install -r requirements.txt</code>

Recommended Frameworks:
<ul>
    <li><b>YOLOv11</b> - For real-time human detection</li>
    <li><b>MobileNetV3</b> - A lightweight convolutional neural network for feature extraction</li>
    <li><b>Bi-LSTM</b> - A bidirectional LSTM to capture temporal dependencies in video sequences</li>
    <li><b>TensorFlow/Keras</b> - Deep learning framework for training the model</li>
    <li><b>OpenCV</b> - For video frame processing</li>
    <li><b>Streamlit</b> - For building an interactive UI</li>
</ul>

<h2>⚡ Features</h2>
<ul>
    <li>Lightweight and optimized for real-time inference</li>
    <li>Uses human filtering via YOLO to reduce unnecessary processing</li>
    <li>Can analyze short video clips for violence detection</li>
    <li>Provides a confidence score for each prediction</li>
</ul>

<h2>📌 Notes</h2>
<ul>
    <li>Ensure you are using the latest versions of TensorFlow and PyTorch to avoid compatibility issues with Bi-LSTM and MobileNetV3.</li>
    <li>The model is optimized for short video clips and may not work well on very long sequences, that's why YOLO is used for filtering the non-human-frames.</li>
    <li>the dataset used in this project is primary and secondary datasets. the primary datasets are not included in this repo because of privacy. you can take your own videos and clip it 1-5 seconds in a relevant part of frames. ;)</li>
    <li>use the batch if you want to do preditcion with several files.</li>
</u>
<br>
<img src="imagess/Screenshot 2025-08-29 105429.png" width="800" />
