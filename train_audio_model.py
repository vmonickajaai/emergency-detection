# scripts/train_audio_model.py
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from extract_features import extract_features

DATA_DIR = os.path.join("audio_dataset")  # adjust if needed
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

# allowed extensions
EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        if fname.lower().endswith(EXTS):
            path = os.path.join(class_path, fname)
            try:
                feat = extract_features(path)
                X.append(feat)
                y.append(class_name)
            except Exception as e:
                print("Error reading", path, "->", e)

X = np.array(X)
y = np.array(y)

# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# train SVM (probability=True for confidences)
model = SVC(kernel="rbf", probability=True, class_weight="balanced")
model.fit(X_train, y_train)

# evaluate
acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc:.3f}")

# save artifacts
joblib.dump({"model": model, "scaler": scaler, "label_encoder": le}, os.path.join(MODEL_DIR, "audio_model.pkl"))
print("Saved: models/audio_model.pkl")
