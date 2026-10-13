
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


# ==========================================
# 1. NORMALIZE HAND LANDMARKS
# ==========================================

def normalize_landmarks(row):
    landmarks = np.array(row, dtype=float).reshape(21, 3)

    # Set wrist landmark as the origin
    landmarks = landmarks - landmarks[0].copy()

    # Normalize hand size
    scale = np.max(np.linalg.norm(landmarks, axis=1))

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# ==========================================
# 2. LOAD GESTURE DATA
# ==========================================

files = [
    "Fist_data.csv",
    "OK_data.csv",
    "Open palm_data.csv",
    "Peace_data.csv",
    "Rock_data.csv",
    "thumbs up_data.csv"
]

print("Loading gesture data...")

dataframes = []

for file in files:
    try:
        df = pd.read_csv(file, header=None)

        if df.shape[1] != 64:
            print(f"Skipping {file}: expected 64 columns.")
            continue

        dataframes.append(df)
        print(f"Loaded {file}: {len(df)} samples")

    except FileNotFoundError:
        print(f"WARNING: File not found: {file}")


if not dataframes:
    raise SystemExit("No valid gesture CSV files were loaded.")

data = pd.concat(dataframes, ignore_index=True)

# Separate landmarks and labels
raw_X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(f"\nTotal samples: {len(data)}")

print("\nSamples per gesture:")
print(pd.Series(y).value_counts().to_string())


# ==========================================
# 3. NORMALIZE LANDMARKS
# ==========================================

print("\nNormalizing landmarks...")

X = np.array([
    normalize_landmarks(row)
    for row in raw_X
])


# ==========================================
# 4. CREATE THE SAME TEST SPLIT
# ==========================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. LOAD THE SAVED MODEL
# ==========================================

print("\nLoading gesture_model.pkl...")

model = joblib.load("gesture_model.pkl")

print(f"Model loaded: {type(model).__name__}")

# Check whether the model supports probabilities
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
else:
    predictions = model.predict(X_test)


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

labels = sorted(set(y_test) | set(predictions))

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\n========== CONFUSION MATRIX ==========")
print("Label order:", labels)
print(matrix)

print("\nRows = Actual gestures")
print("Columns = Predicted gestures")


# ==========================================
# 8. CHECK OPEN PALM PREDICTIONS
# ==========================================

open_palm_label = "Open palm"

mask = y_test == open_palm_label

print("\n========== OPEN PALM ANALYSIS ==========")

if np.any(mask):
    actual_open_palm_predictions = predictions[mask]

    print("Actual Open palm test samples:", int(mask.sum()))
    print("\nPredictions for actual Open palm samples:")
    print(
        pd.Series(actual_open_palm_predictions)
        .value_counts()
        .to_string()
    )

    correct = np.sum(
        actual_open_palm_predictions == open_palm_label
    )

    print(
        f"\nCorrect Open palm predictions: "
        f"{correct}/{int(mask.sum())}"
    )
else:
    print("No Open palm samples found in the test split.")

print("\nDiagnostic test completed.")


