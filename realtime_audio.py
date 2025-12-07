# realtime_audio.py
"""
Real-time audio detection with defensive checks and Phase-8 alert hook:
- RMS + peak silence gating
- top-3 probability debug output
- temporal smoothing (k-of-m) and per-class cooldown
- uses model/scaler/label_encoder from models/audio_model.pkl
- calls decision_engine.handle_audio (preferred) or falls back to process_event
- when decision.should_alert -> saves WAV snippet and POSTs JSON to BACKEND_ALERT_URL (async)
"""

from pathlib import Path
import time
from datetime import datetime
import collections
import threading
import json

import numpy as np
import joblib
import librosa
import sounddevice as sd

# optional save helper libs
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    # fallback to scipy wavfile if soundfile not installed
    try:
        from scipy.io import wavfile
        _HAS_SF = False
    except Exception:
        _HAS_SF = None

# Decision engine import (user project)
# Expected APIs:
#   engine.handle_audio(label, conf, extra) -> DecisionResult-like
#   OR engine.process_event(event) -> DecisionResult-like
# DecisionResult-like must have at least: should_alert (bool)
from decision_engine import engine

import requests

# ----------------- CONFIG ----------------
MODEL_PATH = Path("models/audio_model.pkl")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0        # 1s chunks
N_MFCC = 40                # MFCC count -> feature vector 3*N_MFCC = 120
CONF_THRESHOLD = 0.6       # required prob to be considered positive

EMERGENCY_LABELS = {"scream", "gunshot", "explosion", "help", "glass"}

MIN_PEAK = 0.01            # absolute peak amplitude threshold (0-1); below = skip
MIN_RMS = 0.005            # RMS threshold (0-1); below = skip

SMOOTH_M = 3               # examine last M predictions
SMOOTH_K = 2               # require at least K of M to be emergency
COOLDOWN_SECONDS = 3.0     # per-class cooldown

PRINT_TOPK = 3             # debug top-k printed

# Backend alert endpoint (same as your video code)
BACKEND_ALERT_URL = "http://localhost:5000/alert"  # set to '' to disable network
CAPTURE_DIR = Path("logs/captured_frames")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------

def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_120_features(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return (1,120) float32 feature vector: [mfcc_mean, mfcc_std, delta_mean]"""
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    try:
        y, _ = librosa.effects.trim(y)
    except Exception:
        pass
    # ensure minimum length for stable MFCC
    min_len = int(0.2 * sr)
    if y.shape[0] < min_len:
        pad = np.zeros(min_len - y.shape[0], dtype=y.dtype)
        y = np.concatenate([y, pad])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std, delta_mean], axis=0).astype(np.float32)
    return feat.reshape(1, -1)

def load_components(path: Path):
    """Load joblib. Accept dict or estimator."""
    data = joblib.load(path)
    model = None
    scaler = None
    label_encoder = None
    if isinstance(data, dict):
        model = data.get("model") or data.get("clf") or data.get("estimator") or data.get("classifier")
        scaler = data.get("scaler") or data.get("preprocessor")
        label_encoder = data.get("label_encoder") or data.get("le") or data.get("labels")
        if model is None:
            # heuristically pick an object with predict(...)
            for v in data.values():
                if hasattr(v, "predict"):
                    model = v
                    break
    else:
        model = data
    return model, scaler, label_encoder

def topk_from_probs(probs, labels, k=3):
    idxs = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in idxs]

def send_alert_async(payload: dict, audio_path: str = None):
    """Send JSON payload (and optional file) to BACKEND_ALERT_URL asynchronously."""
    if not BACKEND_ALERT_URL:
        return

    def _worker(p, audio_p):
        try:
            files = None
            if audio_p and Path(audio_p).exists():
                with open(audio_p, "rb") as f:
                    files = {"audio": f}
                    # send form-data (audio + json fields)
                    resp = requests.post(BACKEND_ALERT_URL, data=p, files=files, timeout=6)
            else:
                resp = requests.post(BACKEND_ALERT_URL, json=p, timeout=6)
            # optional debug:
            # print("[send_alert_async] status", resp.status_code, resp.text)
        except Exception as e:
            print("[send_alert_async] failed:", e)

    threading.Thread(target=_worker, args=(payload, audio_path), daemon=True).start()

def save_wav_from_array(arr: np.ndarray, sr: int, path: Path):
    """Save float32 array (shape (n,) or (n,1)) to WAV file. Uses soundfile if available, otherwise scipy."""
    try:
        if _HAS_SF is True:
            sf.write(str(path), arr, sr)
            return True
        elif _HAS_SF is False:
            # scipy expects int PCM; scale float32 to int16
            scaled = np.int16(np.clip(arr * 32767, -32768, 32767))
            wavfile.write(str(path), sr, scaled)
            return True
        else:
            # no writer available
            print("[save_wav] No soundfile or scipy available to write WAV.")
            return False
    except Exception as e:
        print("[save_wav] failed:", e)
        return False

def _call_engine_for_audio(label: str, conf: float, extra: dict):
    """
    Call decision engine: prefer handle_audio, fallback to process_event.
    Return whatever the engine returns; it must have .should_alert attribute.
    """
    # prefer new dedicated API
    if hasattr(engine, "handle_audio"):
        try:
            return engine.handle_audio(label=label, conf=float(conf), extra=extra)
        except Exception as e:
            print("[_call_engine_for_audio] engine.handle_audio failed:", e)
    # fallback to generic process_event
    if hasattr(engine, "process_event"):
        try:
            event = {
                "source_type": "audio",
                "label": label,
                "confidence": float(conf),
                "timestamp": datetime.now(),
                "extra": extra,
            }
            return engine.process_event(event)
        except Exception as e:
            print("[_call_engine_for_audio] engine.process_event failed:", e)
    # last-resort simple decision object
    class Simple:
        def __init__(self, should_alert, reason):
            self.should_alert = should_alert
            self.severity = "normal" if should_alert else None
            self.reason = reason
            self.label = label
            self.confidence = conf
    should = (label in EMERGENCY_LABELS) and (conf >= CONF_THRESHOLD)
    return Simple(should, f"fallback audio rule: {label} (conf={conf:.2f})")

def main():
    print("[INFO] Loading model components...")
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return
    model, scaler, le = load_components(MODEL_PATH)
    if model is None:
        print("[ERROR] Could not find model inside", MODEL_PATH)
        return

    labels = None
    if le is not None:
        try:
            labels = list(le.classes_)
        except Exception:
            labels = None

    supports_proba = hasattr(model, "predict_proba")
    print("[INFO] model type:", type(model))
    print("[INFO] n_features_in_:", getattr(model, "n_features_in_", None))
    print("[INFO] labels:", labels)
    print("[INFO] supports_proba:", supports_proba)
    print(f"[INFO] Recording {CHUNK_SECONDS}s chunks at {SAMPLE_RATE}Hz. Ctrl+C to stop.\n")

    frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    recent_preds = collections.deque(maxlen=SMOOTH_M)
    last_alert_time = {}

    try:
        while True:
            rec = sd.rec(frames=frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = rec.flatten()
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[DEBUG] peak:{peak:.4f} rms:{rms:.5f}", end="  ")

            # silence gating
            if peak < MIN_PEAK or rms < MIN_RMS:
                print("[AUDIO] below gating thresholds -> skipping prediction")
                recent_preds.append(("__silence__", 0.0))
                time.sleep(0.01)
                continue

            feat = extract_120_features(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            if scaler is not None:
                try:
                    feat_scaled = scaler.transform(feat)
                except Exception as e:
                    print("[WARN] scaler.transform failed, using raw features:", e)
                    feat_scaled = feat
            else:
                feat_scaled = feat

            label_str = None
            try:
                if supports_proba:
                    probs = model.predict_proba(feat_scaled)[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs.max())
                else:
                    pred = model.predict(feat_scaled)
                    if isinstance(pred[0], (int, np.integer)):
                        idx = int(pred[0])
                        conf = 1.0
                        probs = None
                    else:
                        idx = None
                        conf = 1.0
                        probs = None
                        label_str = pred[0]
            except Exception as e:
                print("\n[ERROR] model.predict failed:", e)
                raise

            if idx is not None:
                if le is not None:
                    try:
                        label = le.inverse_transform([idx])[0]
                    except Exception:
                        label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
                else:
                    label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
            else:
                label = label_str if label_str is not None else "class_unknown"

            # debug top-k
            if supports_proba and labels:
                topk = topk_from_probs(probs, labels, k=PRINT_TOPK)
                print(f"[PRED] {nowstr()} -> {label} (conf={conf:.2f})  top{PRINT_TOPK}:{topk}")
            else:
                print(f"[PRED] {nowstr()} -> {label} (conf={conf:.2f})")

            recent_preds.append((label, conf))

            # smoothing vote
            emergency_votes = sum(1 for (l, c) in recent_preds if (l in EMERGENCY_LABELS and c >= CONF_THRESHOLD))
            if emergency_votes >= SMOOTH_K:
                # pick best label among recent emergency candidates
                candidates = {}
                for l, c in recent_preds:
                    if l in EMERGENCY_LABELS:
                        candidates.setdefault(l, []).append(c)
                if candidates:
                    best_label = max(candidates.items(), key=lambda kv: (np.mean(kv[1]), len(kv[1])))[0]
                    best_conf = float(np.mean(candidates[best_label]))
                else:
                    best_label = label
                    best_conf = conf

                last_t = last_alert_time.get(best_label, 0)
                elapsed = time.time() - last_t
                if elapsed >= COOLDOWN_SECONDS:
                    extra = {"sample_rate": SAMPLE_RATE, "chunk_seconds": CHUNK_SECONDS}
                    decision = _call_engine_for_audio(best_label, best_conf, extra)

                    should_alert = getattr(decision, "should_alert", False)
                    severity = getattr(decision, "severity", None)
                    reason = getattr(decision, "reason", "")
                    print(f"[DECISION][AUDIO] alert={should_alert} severity={severity} reason={reason}")

                    if should_alert:
                        # Save the current chunk as evidence
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_name = f"{ts}_{best_label}_{int(best_conf*100)}_audio.wav"
                        wav_path = CAPTURE_DIR / wav_name
                        saved = save_wav_from_array(audio, SAMPLE_RATE, wav_path)
                        if saved:
                            print(f"[save_audio_snippet] Saved audio -> {wav_path}")
                        else:
                            wav_path = None
                        # Build payload and POST to backend (async)
                        payload = {
                            "emergency": best_label,
                            "confidence": float(best_conf),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "audio",
                            "frame_idx": 0,
                            "file_name": wav_name if saved else None,
                        }
                        send_alert_async(payload, audio_path=str(wav_path) if wav_path else None)
                        last_alert_time[best_label] = time.time()
                    else:
                        print(f"[INFO] decision engine suppressed alert for {best_label}")
                else:
                    print(f"[INFO] suppressed {best_label} due cooldown ({elapsed:.1f}s elapsed)")

            # small sleep to be cooperative
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (CTRL+C). Exiting.")
    except Exception as e:
        print("[ERROR] unexpected:", e)
        raise

if __name__ == "__main__":
    main()
