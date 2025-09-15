import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []
gesture_name = input("Enter gesture name: ")

print("Press 's' to save frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Flatten landmarks into a list
        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        cv2.putText(frame, gesture_name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and results.multi_hand_landmarks:
        landmarks.append(gesture_name)  # Add label at end
        data.append(landmarks)
        print("Saved frame")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(f"{gesture_name}_data.csv", index=False, header=False)
print(f"Saved {len(data)} frames to {gesture_name}_data.csv")

