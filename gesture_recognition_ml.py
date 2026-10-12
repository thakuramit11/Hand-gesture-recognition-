
import cv2
import mediapipe as mp
import joblib
import numpy as np

# --------------------------------------------------
# 1. LOAD TRAINED MACHINE LEARNING MODEL
# --------------------------------------------------

clf = joblib.load("gesture_model.pkl")

print("Model loaded successfully.")
print(f"Model type: {clf.__class__.__name__}")


# --------------------------------------------------
# 2. NORMALIZE HAND LANDMARKS
# --------------------------------------------------

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks, dtype=float).reshape(21, 3)

    # Make the wrist the reference point
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    # Scale landmarks to reduce the effect of hand size
    distances = np.linalg.norm(landmarks, axis=1)
    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# --------------------------------------------------
# 3. PREDICT GESTURE AND CONFIDENCE
# --------------------------------------------------

def predict_gesture(landmarks, threshold=0.65):
    normalized_landmarks = normalize_landmarks(landmarks)

    probabilities = clf.predict_proba([normalized_landmarks])[0]

    best_index = np.argmax(probabilities)
    confidence = float(probabilities[best_index])
    predicted_gesture = str(clf.classes_[best_index])

    # Reject predictions with low confidence
    return predicted_gesture, confidence


# --------------------------------------------------
# 4. CONFIGURE MEDIAPIPE HAND LANDMARKER
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


# --------------------------------------------------
# 5. OPEN WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

# --------------------------------------------------
# 6. GESTURE SMOOTHING SETTINGS
# --------------------------------------------------

gesture_history = []

current_gesture = "Show your hand"
current_confidence = 0.0

HISTORY_LENGTH = 5
REQUIRED_COUNT = 4

# Hand landmark connections for drawing
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# 7. REAL-TIME GESTURE RECOGNITION
# --------------------------------------------------

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read camera frame.")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            # ------------------------------------------
            # 8. CHECK WHETHER A HAND IS DETECTED
            # ------------------------------------------

            if result.hand_landmarks:

                hand_landmarks = result.hand_landmarks[0]
                h, w, _ = frame.shape

                # Draw landmark points
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections between landmarks
                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)

                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Convert 21 landmarks into 63 values
                landmarks = []

                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Predict gesture and confidence
                detected_gesture, confidence = predict_gesture(
                    landmarks
                )

                # --------------------------------------
                # 9. SMOOTH PREDICTIONS
                # --------------------------------------

                if detected_gesture == "Unknown":

                    gesture_history.clear()

                    current_gesture = "Unknown"
                    current_confidence = confidence

                else:

                    # Store gesture and its confidence together
                    gesture_history.append(
                        (detected_gesture, confidence)
                    )

                    # Keep only the latest five predictions
                    if len(gesture_history) > HISTORY_LENGTH:
                        gesture_history.pop(0)

                    # Extract gesture names from history
                    gesture_names = [
                        item[0] for item in gesture_history
                    ]

                    # Require at least four matching predictions
                    if gesture_names.count(
                        detected_gesture
                    ) >= REQUIRED_COUNT:

                        current_gesture = detected_gesture

                        # Average confidence for the stable gesture
                        matching_confidences = [
                            score
                            for name, score in gesture_history
                            if name == current_gesture
                        ]

                        current_confidence = (
                            sum(matching_confidences)
                            / len(matching_confidences)
                        )

            else:
                # Reset when no hand is detected
                gesture_history.clear()

                current_gesture = "No hand detected"
                current_confidence = 0.0

            # ------------------------------------------
            # 10. DISPLAY RESULTS
            # ------------------------------------------

            cv2.putText(
                frame,
                f"Gesture: {current_gesture}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.0%}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("ML Gesture Recognition", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed.")