import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob
import os


# ==================================================
# NORMALIZE HAND LANDMARKS
# ==================================================

def normalize_landmarks(row):
    """
    Convert 63 raw MediaPipe values into
    position- and size-independent features.

    21 landmarks × (x, y, z) = 63 features
    """

    landmarks = np.array(row, dtype=float).reshape(21, 3)

    # ----------------------------------------------
    # Use wrist (landmark 0) as origin
    # ----------------------------------------------

    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist


    # ----------------------------------------------
    # Normalize hand size
    # ----------------------------------------------

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale


    # ----------------------------------------------
    # Flatten back to 63 values
    # ----------------------------------------------

    return landmarks.flatten()


# ==================================================
# LOAD CSV FILES
# ==================================================

all_files = glob.glob("*_data.csv")

all_files = [
    file
    for file in all_files
    if os.path.getsize(file) > 0
]


if not all_files:

    print("No *_data.csv files found.")
    exit()


print("CSV files found:")

for file in all_files:
    print(f"  - {file}")


data_list = []


for file in all_files:

    try:

        df = pd.read_csv(
            file,
            header=None
        )

        # 63 features + 1 label
        if df.shape[1] != 64:

            print(
                f"Skipping {file}: "
                f"expected 64 columns, "
                f"found {df.shape[1]}"
            )

            continue

        data_list.append(df)

    except Exception as e:

        print(
            f"Could not read {file}: {e}"
        )


if not data_list:

    print("No valid gesture CSV files found.")
    exit()


# ==================================================
# COMBINE DATA
# ==================================================

data = pd.concat(
    data_list,
    ignore_index=True
)

print("\nTotal samples:", len(data))
print("Total columns:", data.shape[1])


# ==================================================
# FEATURES & LABELS
# ==================================================

raw_X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# ==================================================
# NORMALIZE FEATURES
# ==================================================

print("\nNormalizing hand landmarks...")

X = np.array([
    normalize_landmarks(row)
    for row in raw_X
])


# ==================================================
# SHOW CLASS DISTRIBUTION
# ==================================================

print("\nGestures found:")

for gesture, count in pd.Series(y).value_counts().items():

    print(
        f"  {gesture}: {count} samples"
    )


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==================================================
# MODELS
# ==================================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    ),

    "MLP (Neural Net)": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    )
}


# ==================================================
# TRAIN MODELS
# ==================================================

best_model = None
best_acc = 0
best_name = ""


print("\n========== MODEL RESULTS ==========\n")


for name, model in models.items():

    print(f"Training {name}...")

    try:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        acc = accuracy_score(
            y_test,
            y_pred
        )

        print(
            f"{name} Accuracy: {acc:.2f}"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        if acc > best_acc:

            best_acc = acc
            best_model = model
            best_name = name

    except Exception as e:

        print(
            f"{name} failed: {e}"
        )


# ==================================================
# SAVE BEST MODEL
# ==================================================

if best_model is None:

    print(
        "\nNo model could be trained."
    )

    exit()


joblib.dump(
    best_model,
    "gesture_model.pkl"
)


print("\n===================================")
print(
    f"Best Model: {best_name}"
)
print(
    f"Accuracy: {best_acc:.2f}"
)
print(
    "Model saved as gesture_model.pkl"
)
print("===================================")
