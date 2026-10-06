## Project Overview

This project is a **Hand Gesture Recognition system** built using Python, OpenCV, MediaPipe, and Machine Learning.

The system detects a hand through the webcam, extracts **21 hand landmarks**, and uses a trained Machine Learning model to recognize different hand gestures in real time.

### How It Works

Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Landmark Normalization
   ↓
Machine Learning Model
   ↓
Gesture Prediction
```

### Supported Gestures

The current model recognizes the following gestures:

* ✊ Fist
* 👌 OK
* ✋ Open Palm
* ✌️ Peace
* 🤘 Rock
* 👍 Thumbs Up

### Technologies Used

* **Python** – Main programming language
* **OpenCV** – Webcam access, image processing, and visualization
* **MediaPipe** – Hand landmark detection
* **NumPy** – Numerical calculations and landmark processing
* **Pandas** – Dataset and CSV handling
* **Scikit-learn** – Machine Learning models
* **Joblib** – Saving and loading the trained model

### Machine Learning

The project uses hand landmark coordinates as features.

Each hand contains **21 landmarks**, and every landmark has:

* X coordinate
* Y coordinate
* Z coordinate

This gives:

**21 × 3 = 63 features**

The landmarks are normalized before being given to the Machine Learning model. This helps the model recognize gestures even when the hand changes position or distance from the camera.

The current Random Forest model achieved approximately **91% accuracy on the test dataset**.

### Project Goal

The goal of this project is to understand the complete Machine Learning workflow:

**Data Collection → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Real-Time Prediction**

This project is also being developed as a learning project to improve my practical understanding of Python, Computer Vision, and Machine Learning.
