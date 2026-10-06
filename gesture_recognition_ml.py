import cv2
import mediapipe as mp
import joblib
import numpy as np


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

clf = joblib.load("gesture_model.pkl")

print("Model loaded successfully.")
print(f"Model type: {clf.__class__.__name__}")


# ==================================================
# NORMALIZE HAND LANDMARKS
# ==================================================

def normalize_landmarks(row):
    """
    Convert 63 raw MediaPipe landmark values into
    position- and size-independent features.

    21 landmarks × 3 coordinates = 63 features
    """

    landmarks = np.array(
        row,
        dtype=float
    ).reshape(21, 3)

    # ------------------------------------------------
    # Use wrist (landmark 0) as the origin
    # ------------------------------------------------

    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist


    # ------------------------------------------------
    # Normalize hand size
    # ------------------------------------------------

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale


    # ------------------------------------------------
    # Return 63 normalized features
    # ------------------------------------------------

    return landmarks.flatten()


# ==================================================
# PREDICT GESTURE
# ==================================================

def predict_gesture(landmarks):

    normalized_landmarks = normalize_landmarks(
        landmarks
    )

    prediction = clf.predict(
        [normalized_landmarks]
    )

    return prediction[0]


# ==================================================
# MEDIAPIPE HAND LANDMARKER
# ==================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# --------------------------------------------------
# Hand Landmarker model
# --------------------------------------------------

MODEL_PATH = "hand_landmarker.task"


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.7,

    min_hand_presence_confidence=0.7,

    min_tracking_confidence=0.7
)


# ==================================================
# OPEN CAMERA
# ==================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Error: Could not open camera.")

    exit()


# ==================================================
# GESTURE SMOOTHING
# ==================================================

gesture_history = []

current_gesture = ""

HISTORY_LENGTH = 5

REQUIRED_COUNT = 4


# ==================================================
# HAND CONNECTIONS
# ==================================================

connections = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


# ==================================================
# CREATE HAND LANDMARKER
# ==================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ------------------------------------------------
        # Read camera frame
        # ------------------------------------------------

        ret, frame = cap.read()


        if not ret:

            print(
                "Error: Could not read camera frame."
            )

            break


        # ------------------------------------------------
        # Mirror camera
        # ------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ------------------------------------------------
        # Convert BGR → RGB
        # ------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ------------------------------------------------
        # Create MediaPipe image
        # ------------------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )


        # ------------------------------------------------
        # Detect hand
        # ------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        detected_gesture = ""


        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:

            hand_landmarks = (
                result.hand_landmarks[0]
            )


            h, w, _ = frame.shape


            # ------------------------------------------------
            # Draw landmark points
            # ------------------------------------------------

            for lm in hand_landmarks:

                x = int(
                    lm.x * w
                )

                y = int(
                    lm.y * h
                )


                cv2.circle(

                    frame,

                    (x, y),

                    5,

                    (0, 255, 0),

                    -1
                )


            # ------------------------------------------------
            # Draw landmark connections
            # ------------------------------------------------

            for start, end in connections:

                x1 = int(
                    hand_landmarks[start].x * w
                )

                y1 = int(
                    hand_landmarks[start].y * h
                )


                x2 = int(
                    hand_landmarks[end].x * w
                )

                y2 = int(
                    hand_landmarks[end].y * h
                )


                cv2.line(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )


            # ------------------------------------------------
            # Extract 63 raw landmark values
            # ------------------------------------------------

            landmarks = []


            for lm in hand_landmarks:

                landmarks.extend([

                    lm.x,

                    lm.y,

                    lm.z

                ])


            # ------------------------------------------------
            # Predict gesture
            # ------------------------------------------------

            detected_gesture = (
                predict_gesture(
                    landmarks
                )
            )


        # =================================================
        # GESTURE HISTORY / SMOOTHING
        # =================================================

        if detected_gesture:

            gesture_history.append(
                detected_gesture
            )


            # Keep only last 5 predictions
            if len(gesture_history) > HISTORY_LENGTH:

                gesture_history.pop(0)


            # ----------------------------------------------
            # Check majority prediction
            # ----------------------------------------------

            if (
                gesture_history.count(
                    detected_gesture
                )
                >= REQUIRED_COUNT
            ):

                current_gesture = (
                    detected_gesture
                )


        # =================================================
        # DISPLAY CURRENT GESTURE
        # =================================================

        cv2.putText(

            frame,

            current_gesture,

            (50, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.5,

            (0, 0, 255),

            3
        )


        # ------------------------------------------------
        # Display instructions
        # ------------------------------------------------

        cv2.putText(

            frame,

            "Press Q to quit",

            (50, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        # ------------------------------------------------
        # Display camera
        # ------------------------------------------------

        cv2.imshow(

            "ML Gesture Recognition",

            frame
        )


        # ------------------------------------------------
        # Quit
        # ------------------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


# ==================================================
# CLEANUP
# ==================================================

cap.release()

cv2.destroyAllWindows()

print("Program closed.")
