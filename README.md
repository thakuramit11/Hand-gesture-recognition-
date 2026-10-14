# Hand Gesture Recognition Using Machine Learning

A real-time hand gesture recognition system built with Python, OpenCV, MediaPipe, and Scikit-learn. The project detects hand landmarks through a webcam and uses a trained machine learning model to classify six hand gestures.

## Project Overview

This project recognizes hand gestures in real time using 21 hand landmarks extracted by MediaPipe's Hand Landmarker.

The landmark coordinates are normalized and passed to a trained machine learning pipeline, which predicts the corresponding gesture.

## Supported Gestures

| Gesture | Label     |
| ------- | --------- |
| ✊       | Fist      |
| 👌      | OK        |
| ✋       | Open Palm |
| ✌️      | Peace     |
| 🤘      | Rock      |
| 👍      | Thumbs Up |

## How It Works

```text
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
Trained ML Pipeline
   ↓
Gesture Prediction
   ↓
Real-Time Display
```

## Technologies Used

* **Python** — Main programming language
* **OpenCV** — Webcam access and image visualization
* **MediaPipe** — Hand landmark detection
* **NumPy** — Numerical calculations
* **Pandas** — Dataset loading and CSV handling
* **Scikit-learn** — Model training and evaluation
* **Joblib** — Saving and loading the trained model

## Machine Learning Approach

Each detected hand contains 21 landmarks. Every landmark provides X, Y, and Z coordinates, giving a total of 63 numerical features.

The project normalizes the landmark coordinates before passing them to the model. This reduces sensitivity to the hand's position and scale.

The training script evaluates multiple algorithms:

* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Multi-Layer Perceptron (MLP) Neural Network

The best-performing model is selected based on held-out test accuracy and saved as `gesture_model.pkl`.

### Latest Model Results

| Metric                |             Result |
| --------------------- | -----------------: |
| Selected model        | MLP Neural Network |
| Test accuracy         |                96% |
| Open Palm precision   |                90% |
| Open Palm recall      |                98% |
| Number of gestures    |                  6 |
| Total dataset samples |                978 |

*These results are from the current held-out test split. Performance on new hands, camera setups, and lighting conditions may differ.*

## Project Structure

```text
Hand-gesture-recognition-/
├── Fist_data.csv
├── OK_data.csv
├── Open palm_data.csv
├── Peace_data.csv
├── Rock_data.csv
├── thumbs up_data.csv
├── gesture_model.pkl
├── hand_landmarker.task
├── gesture_recognition_ml.py
├── train_gesture_model.py
├── collect_gesture_data..py
├── test_gesture_model.py
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/thakuramit11/Hand-gesture-recognition-.git
cd Hand-gesture-recognition-
```

### 2. Install dependencies

Make sure Python is installed, then run:

```bash
python -m pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

### 3. Check the required files

Ensure `gesture_model.pkl` and `hand_landmarker.task` are available in the project directory.

### 4. Run real-time recognition

```bash
python gesture_recognition_ml.py
```

Allow webcam access if prompted.

## Training and Testing

To retrain the model using the available CSV datasets:

```bash
python train_gesture_model.py
```

To evaluate the saved model using the diagnostic script:

```bash
python test_gesture_model.py
```

**Note:** The diagnostic script evaluates the dataset using the same fixed train/test split used by the training script. It is useful for checking model performance, but it is not an independent real-world evaluation.

## Data Collection

Run the data collection script:

```bash
python "collect_gesture_data..py"
```

Follow the prompts to select a gesture. Press **S** to save a sample and **Q** to finish collection. The script appends newly collected samples to the corresponding CSV file.

## Controls

* **Q** — Quit real-time recognition or finish data collection
* **S** — Save a sample during data collection

## Future Improvements

* Test with independent datasets and different users.
* Improve robustness under varied lighting and camera angles.
* Reduce confusion between visually similar gestures.
* Improve prediction stability during transitions.
* Explore recognition of additional gestures.

## Learning Outcomes

This project provides practical experience with:

* Webcam-based computer vision
* Hand landmark extraction and feature normalization
* CSV-based dataset collection and preprocessing
* Training and evaluating multiple machine learning algorithms
* Saving and loading trained models
* Real-time model inference
* Git and GitHub project management
