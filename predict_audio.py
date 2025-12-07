# scripts/predict_audio.py
import sys
import joblib
from extract_features import extract_features
import numpy as np

model_data = joblib.load("models/audio_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
le = model_data["label_encoder"]

file_path = sys.argv[1]  # python predict_audio.py audio_dataset/scream/s1.wav

feat = extract_features(file_path).reshape(1, -1)
feat_scaled = scaler.transform(feat)
probs = model.predict_proba(feat_scaled)[0]
idx = probs.argmax()
label = le.inverse_transform([idx])[0]
conf = probs[idx]

print(f"Prediction: {label}  confidence: {conf:.3f}")
