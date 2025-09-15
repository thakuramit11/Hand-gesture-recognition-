import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import glob

# Load all gesture CSVs
all_files = glob.glob("*_data.csv")
data_list = []

for file in all_files:
    df = pd.read_csv(file, header=None)
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)

# Features & labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try multiple models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

# Save best model
joblib.dump(best_model, "gesture_model.pkl")
print(f"\n✅ Best Model: {best_model.__class__.__name__} (Accuracy: {best_acc:.2f})")
print("Model saved as gesture_model.pkl")
