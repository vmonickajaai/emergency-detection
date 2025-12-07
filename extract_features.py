# scripts/extract_features.py
import librosa
import numpy as np

def extract_features(file_path, sr=22050, n_mfcc=40):
    # load audio (librosa supports .wav, .mp3, .ogg)
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    # normalize amplitude -> helps with mixed file formats
    audio = librosa.util.normalize(audio)
    # optional trimming of silence
    audio, _ = librosa.effects.trim(audio, top_db=30)
    # extract MFCCs and take mean over time (simple & effective)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # delta and delta-delta (optional, often helps)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # combine and take mean across frames -> fixed-length vector
    features = np.concatenate((np.mean(mfcc, axis=1),
                               np.mean(delta, axis=1),
                               np.mean(delta2, axis=1)))
    return features
