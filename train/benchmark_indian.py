# GenderLens v1.0 - github.com/your-username/gender-lens
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
OUT_PATH = ROOT / "train" / "benchmark_indian_results.json"

SAMPLES = [
    ("Subhashree", "Female"), ("Lipsa", "Female"), ("Smruti", "Female"), ("Jyotirmayee", "Female"),
    ("Rashmita", "Female"), ("Sujata", "Female"), ("Lopamudra", "Female"), ("Pallabi", "Female"),
    ("Ipsita", "Female"), ("Nibedita", "Female"), ("Hemalata", "Female"), ("Chitralekha", "Female"),
    ("Ananya", "Female"), ("Pooja", "Female"), ("Shreya", "Female"), ("Aishwarya", "Female"),
    ("Lakshmi", "Female"), ("Meenakshi", "Female"), ("Kavitha", "Female"), ("Padmavathi", "Female"),
    ("Arpita", "Female"), ("Indrani", "Female"), ("Tanushree", "Female"), ("Vaishnavi", "Female"),
    ("Biswajit", "Male"), ("Satyajit", "Male"), ("Debasis", "Male"), ("Niladri", "Male"),
    ("Trilochan", "Male"), ("Pranab", "Male"), ("Subrat", "Male"), ("Madhusudan", "Male"),
    ("Jagannath", "Male"), ("Laxmidhar", "Male"), ("Rahul", "Male"), ("Vikram", "Male"),
    ("Rakesh", "Male"), ("Sanjay", "Male"), ("Murugan", "Male"), ("Senthil", "Male"),
    ("Karthik", "Male"), ("Venkatesh", "Male"), ("Srinivas", "Male"), ("Krishna", "Male"),
    ("Abhijit", "Male"), ("Kaushik", "Male"), ("Partha", "Male"), ("Tanmoy", "Male"),
    ("Pratyush", "Male"), ("Sourav", "Male"), ("Ritwik", "Male"), ("Harshad", "Male"),
    ("Tanvi", "Female"), ("Aaradhya", "Female"), ("Shraddha", "Female"), ("Ishita", "Female"),
    ("Aparna", "Female"), ("Sakshi", "Female"), ("Nupur", "Female"), ("Bhavani", "Female")
]


def normalize_name(name: str) -> str:
    return str(name or "").strip().lower().split()[0][:20]


def encode_name(name: str) -> np.ndarray:
    clean = normalize_name(name)
    tensor = np.zeros((20, 28), dtype=np.float32)
    for i, ch in enumerate(clean[:20]):
        code = ord(ch) - 96
        if 1 <= code <= 26:
            tensor[i, code] = 1.0
        elif ch == " ":
            tensor[i, 27] = 1.0
    return tensor


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, 28)),
        tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d"),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu", name="conv1d_1"),
        tf.keras.layers.GlobalMaxPooling1D(name="global_max_pooling1d"),
        tf.keras.layers.Dense(32, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1")
    ])
    return model


def load_weights_from_bin(model: tf.keras.Model, bin_path: Path):
    model(np.zeros((1, 20, 28), dtype=np.float32), training=False)
    current = model.get_weights()
    raw = np.frombuffer(bin_path.read_bytes(), dtype="<f4")

    offset = 0
    loaded = []
    for arr in current:
        size = int(np.prod(arr.shape))
        chunk = raw[offset:offset + size]
        if chunk.size != size:
            raise ValueError("weights.bin size mismatch")
        loaded.append(chunk.reshape(arr.shape).astype(np.float32))
        offset += size

    model.set_weights(loaded)


def predict_name(model: tf.keras.Model, name: str):
    x = np.expand_dims(encode_name(name), axis=0)
    prob = float(model(x, training=False).numpy().reshape(-1)[0])
    gender = "Male" if prob > 0.5 else "Female"
    confidence = max(0, min(100, round(abs(prob - 0.5) * 200)))
    return gender, confidence, prob


def main():
    model = build_model()
    load_weights_from_bin(model, MODEL_DIR / "weights.bin")

    rows = []
    correct = 0
    for name, expected in SAMPLES:
        pred, conf, prob = predict_name(model, name)
        ok = pred == expected
        correct += int(ok)
        rows.append({
            "name": name,
            "expected": expected,
            "predicted": pred,
            "confidence": conf,
            "prob_male": round(prob, 4),
            "correct": ok,
        })

    summary = {
        "total": len(SAMPLES),
        "correct": correct,
        "accuracy": round(correct / len(SAMPLES), 4),
        "avg_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 2),
    }

    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
