import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

RECOMMENDED_TARGET = 120
MODEL_PATH = "hand_landmarker.task"


# ============================================================
# NORMALIZE LANDMARKS
# Same normalization used by the training and recognition code
# ============================================================

def normalize_landmarks(landmarks):

    # Convert 63 values into 21 x 3
    landmarks = np.array(
        landmarks,
        dtype=float
    ).reshape(21, 3)

    # Use wrist (landmark 0) as the origin
    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist

    # Normalize hand size
    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale

    # Return 63 values
    return landmarks.flatten()


# ============================================================
# GET GESTURE NAME
# ============================================================

gesture_name = input(
    "\nEnter gesture name "
    "(Fist / OK / Open palm / Peace / Rock / thumbs up): "
).strip()


if not gesture_name:

    print("Error: Gesture name cannot be empty.")

    exit()


# ============================================================
# FILE NAME
# ============================================================

filename = f"{gesture_name}_data.csv"


# ============================================================
# LOAD EXISTING DATA
# ============================================================

existing_data = []


if os.path.exists(filename):

    try:

        old_df = pd.read_csv(
            filename,
            header=None
        )

        # Check number of columns
        if old_df.shape[1] != 64:

            print(
                f"\nError: {filename} has "
                f"{old_df.shape[1]} columns."
            )

            print(
                "Expected 64 columns "
                "(63 landmarks + 1 label)."
            )

            exit()


        existing_data = old_df.values.tolist()


        print(
            f"\nExisting samples found: "
            f"{len(existing_data)}"
        )


    except Exception as e:

        print(
            f"\nCould not read existing file:"
        )

        print(e)

        exit()


else:

    print("\nNo existing data found.")

    print("Starting from 0 samples.")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print(
        f"\nError: {MODEL_PATH} not found."
    )

    print(
        "Make sure hand_landmarker.task "
        "is in the project folder."
    )

    exit()


# ============================================================
# MEDIAPIPE TASKS API
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.7,

    min_hand_presence_confidence=0.7,

    min_tracking_confidence=0.7
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "\nError: Could not open camera."
    )

    exit()


# ============================================================
# NEW DATA FOR THIS SESSION
# ============================================================

new_data = []


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# INFORMATION
# ============================================================

print("\n==========================================")
print(f"Gesture            : {gesture_name}")
print(
    f"Already collected : "
    f"{len(existing_data)}"
)
print(
    f"Recommended target: "
    f"{RECOMMENDED_TARGET}"
)
print("==========================================")

print("\nInstructions:")
print("1. Show ONLY the selected gesture.")
print("2. Press S to save one sample.")
print("3. Move your hand slightly between samples.")
print("4. You can collect MORE than 120 samples.")
print("5. Press Q whenever you want to finish.")

print("\nCamera starting...")


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()


        if not ret:

            print(
                "Could not read camera frame."
            )

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # CONVERT BGR TO RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # CREATE MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        hand_detected = False

        landmarks = None


        # ====================================================
        # IF HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand_detected = True

            hand_landmarks = (
                result.hand_landmarks[0]
            )


            h, w, _ = frame.shape


            # ------------------------------------------------
            # DRAW LANDMARK POINTS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # DRAW HAND CONNECTIONS
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
            # EXTRACT 21 LANDMARKS
            # ------------------------------------------------

            landmarks = []


            for lm in hand_landmarks:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])


        # ====================================================
        # CALCULATE TOTAL
        # ====================================================

        total_collected = (
            len(existing_data)
            + len(new_data)
        )


        # ====================================================
        # DISPLAY GESTURE
        # ====================================================

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY SAMPLE COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Collected: {total_collected}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY TARGET
        # ====================================================

        cv2.putText(
            frame,
            f"Recommended: {RECOMMENDED_TARGET}+",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # HAND STATUS
        # ====================================================

        if hand_detected:

            cv2.putText(
                frame,
                "HAND DETECTED",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            frame,
            "S = Save Sample",
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Q = Finish",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Gesture Data Collection",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # SAVE SAMPLE WITH S
        # ====================================================

        if key == ord("s"):

            if landmarks is not None:

                # Normalize landmarks
                normalized = (
                    normalize_landmarks(
                        landmarks
                    )
                )


                # Create row
                row = list(
                    normalized
                )


                # Add gesture label
                row.append(
                    gesture_name
                )


                # Add to new data
                new_data.append(
                    row
                )


                total_collected = (
                    len(existing_data)
                    + len(new_data)
                )


                print(
                    f"Saved sample | "
                    f"Collected: "
                    f"{total_collected}"
                )


            else:

                print(
                    "No hand detected. "
                    "Sample NOT saved."
                )


        # ====================================================
        # QUIT WITH Q
        # ====================================================

        elif key == ord("q"):

            print(
                "\nFinishing collection..."
            )

            break


# ============================================================
# CLOSE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE DATA
# ============================================================

if len(new_data) == 0:

    print(
        "\nNo new samples were collected."
    )

    exit()


# Combine old + new
all_data = (
    existing_data
    + new_data
)


# Create DataFrame
df = pd.DataFrame(
    all_data
)


# Save CSV
df.to_csv(
    filename,
    index=False,
    header=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("DATA COLLECTION UPDATED")
print("==========================================")

print(
    f"Gesture          : {gesture_name}"
)

print(
    f"Previous samples : "
    f"{len(existing_data)}"
)

print(
    f"New samples      : "
    f"{len(new_data)}"
)

print(
    f"Total samples    : "
    f"{len(all_data)}"
)

print(
    f"File             : "
    f"{filename}"
)

print("\n63 normalized landmarks + 1 label")

print("\nSaved successfully! ✅")