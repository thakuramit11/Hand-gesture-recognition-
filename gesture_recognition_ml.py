import cv2
import mediapipe as mp
import joblib

# Load model
clf = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture_history = []
current_gesture = ""

def predict_gesture(landmarks):
    return clf.predict([landmarks])[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    detected_gesture = ""

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        detected_gesture = predict_gesture(landmarks)

    # Frame history filter
    if detected_gesture:
        gesture_history.append(detected_gesture)
    if len(gesture_history) > 5:
        gesture_history.pop(0)

    if gesture_history.count(detected_gesture) > 3:
        current_gesture = detected_gesture

    cv2.putText(frame, current_gesture, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),3)
    cv2.imshow("ML Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
