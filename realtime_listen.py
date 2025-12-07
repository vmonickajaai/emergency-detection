# scripts/realtime_listen.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib
import tempfile
import time
from extract_features import extract_features

model_data = joblib.load("models/audio_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
le = model_data["label_encoder"]

SR = 22050
DURATION = 1.0  # seconds per clip
THRESH = 0.6    # only alert if conf > THRESH

def callback(indata, frames, time_info, status):
    pass  # we use blocking record for simplicity

print("Listening... Press Ctrl+C to stop.")
try:
    while True:
        rec = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        # write temp file (librosa loads from path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, rec, SR)
        feat = extract_features(tmp.name).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        probs = model.predict_proba(feat_scaled)[0]
        idx = probs.argmax()
        conf = probs[idx]
        label = le.inverse_transform([idx])[0]
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if conf >= THRESH and label != "noise":
            print(f"[{ts}] ALERT: {label} ({conf:.2f})")
        else:
            print(f"[{ts}] no alert ({label} {conf:.2f})")
except KeyboardInterrupt:
    print("Stopped.")
