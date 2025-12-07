# inspect_audio_model.py
import joblib
from pathlib import Path

MODEL_PATH = Path("models/audio_model.pkl")

def main():
    print("Loading:", MODEL_PATH)
    data = joblib.load(MODEL_PATH)

    print("\nTYPE OF LOADED OBJECT:", type(data))

    if isinstance(data, dict):
        print("DICT KEYS:", list(data.keys()))

        model = None
        labels = None

        # try to find model
        for k in ("model", "clf", "estimator", "pipeline", "classifier"):
            if k in data:
                model = data[k]
                break
        # try to find labels
        for k in ("labels", "label_map", "classes", "class_names", "labels_"):
            if k in data:
                labels = data[k]
                break

        print("\nMODEL TYPE:", type(model))
        print("LABELS RAW:", labels)

        if hasattr(model, "n_features_in_"):
            print("model.n_features_in_ =", model.n_features_in_)

    else:
        model = data
        print("MODEL TYPE:", type(model))
        if hasattr(model, "n_features_in_"):
            print("model.n_features_in_ =", model.n_features_in_)

if __name__ == "__main__":
    main()
