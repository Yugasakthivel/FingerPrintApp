import os
import cv2
import joblib
import json
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

DATA_DIR = "data/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
hog_params = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def load_data():
    X, y = [], []
    for user in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, user)
        for img_file in os.listdir(user_dir):
            path = os.path.join(user_dir, img_file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            features = hog(img, **hog_params)
            X.append(features)
            y.append(user)
    return np.array(X), np.array(y)

print("[INFO] Loading dataset...")
X, y = load_data()
print(f"[INFO] Samples: {len(y)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
print("[INFO] Training SVM...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y_encoded)

# Evaluate
y_pred = clf.predict(X)
acc = accuracy_score(y_encoded, y_pred)
print(f"[RESULT] Training Accuracy: {acc:.2f}")

# Save model + encoder + hog_params
joblib.dump(clf, os.path.join(MODEL_DIR, "fingerprint_svm.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
with open(os.path.join(MODEL_DIR, "hog_params.json"), "w") as f:
    json.dump(hog_params, f)
print("[SAVED] Model and encoder saved.")
