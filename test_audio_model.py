# test_audio_model.py
import sys
from pathlib import Path
import numpy as np
import librosa
import joblib

MODEL_PATH = Path("models/audio_model.pkl")
SAMPLE_RATE = 16000
N_MFCC = 40  # we expect 40 -> 40*3 = 120 features

def extract_features_for_model(file_path: str, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Produce 120 features: mfcc_mean, mfcc_std, delta_mean (shape (1,120))."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    try:
        y, _ = librosa.effects.trim(y)
    except Exception:
        pass
    if y.shape[0] < int(0.2 * sr):
        pad = np.zeros(int(0.2 * sr) - y.shape[0], dtype=y.dtype)
        y = np.concatenate([y, pad])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, frames)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std, delta_mean], axis=0).astype(np.float32)
    return feat.reshape(1, -1)

def load_model_components(path: Path):
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")
        label_encoder = data.get("label_encoder")
    else:
        model = data
        scaler = None
        label_encoder = None
    return model, scaler, label_encoder

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_audio_model.py path/to/file.wav")
        return
    wav = sys.argv[1]
    model, scaler, le = load_model_components(MODEL_PATH)
    print("Loaded model:", type(model))
    print("Scaler:", type(scaler))
    print("Label encoder:", type(le))
    feat = extract_features_for_model(wav)
    print("Feature shape:", feat.shape)
    if scaler is not None:
        feat = scaler.transform(feat)
    # predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        idx = int(probs.argmax())
        conf = float(probs.max())
    else:
        pred = model.predict(feat)
        # if model returns index
        if isinstance(pred[0], (int, np.integer)):
            idx = int(pred[0]); conf = 1.0
        else:
            # maybe returns label string directly
            label_str = pred[0]
            idx = None; conf = 1.0
    if idx is not None:
        if le is not None:
            label = le.inverse_transform([idx])[0]
        else:
            label = f"class_{idx}"
    else:
        label = label_str
    print(f"\nPREDICTION for {wav}: {label} (conf={conf:.2f})")

if __name__ == "__main__":
    main()
