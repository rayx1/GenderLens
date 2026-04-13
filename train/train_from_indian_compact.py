# GenderLens v1.0 - github.com/your-username/gender-lens
import argparse
import json
import random
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data" / "name_gender_dict.json"
MODEL_DIR = ROOT / "model"
META_PATH = MODEL_DIR / "metadata.json"

INDIAN_SUFFIXES = (
    "jit", "jeet", "nath", "esh", "endra", "kumar", "swamy", "murthy", "priya", "lata", "lekha", "mita",
    "deep", "preet", "anshu", "isha", "vathi", "wati", "amma", "appa", "reddy", "rao", "das", "dev", "raj"
)
INDIAN_PREFIXES = (
    "shri", "sri", "subh", "biswa", "deb", "jaga", "madhu", "krish", "venka", "nara", "anan", "priya",
    "laksh", "meen", "sar", "tan", "raj", "man", "san", "har", "vij", "adi", "om", "sud"
)


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


def is_indian_like(n: str) -> bool:
    return any(n.startswith(p) for p in INDIAN_PREFIXES) or any(n.endswith(s) for s in INDIAN_SUFFIXES)


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
    model.compile(optimizer=tf.keras.optimizers.Adam(9e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Compact weighted training from enriched dictionary")
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--indian-weight", type=float, default=2.2)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    data = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    names, labels, weights = [], [], []
    indian_like_count = 0

    for name, info in data.items():
        gender = info.get("gender")
        if gender not in {"Male", "Female"}:
            continue
        n = normalize_name(name)
        if len(n) < 2:
            continue

        y = 1 if gender == "Male" else 0
        conf = float(info.get("confidence", 0.85))
        w = 1.0 + max(0.0, conf - 0.8)
        if is_indian_like(n):
            w *= args.indian_weight
            indian_like_count += 1

        names.append(n)
        labels.append(y)
        weights.append(w)

        if len(n) > 3:
            names.append(n[:-1]); labels.append(y); weights.append(w * 0.9)
        if len(n) > 4:
            names.append(n[:-2]); labels.append(y); weights.append(w * 0.8)

    X = np.stack([encode_name(n) for n in names]).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    sw = np.array(weights, dtype=np.float32)

    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        X, y, sw, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = build_model()
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )

    _, test_acc = model.evaluate(X_test, y_test, sample_weight=w_test, verbose=0)

    with (MODEL_DIR / "weights.bin").open("wb") as f:
        for arr in model.get_weights():
            flat = np.asarray(arr, dtype=np.float32).ravel(order="C")
            f.write(struct.pack("<" + "f" * flat.size, *flat.tolist()))

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta["trainingDataset"] = "Enriched dictionary + weighted compact training from Indian datasets"
    meta["trainedAt"] = datetime.utcnow().isoformat() + "Z"
    meta["testAccuracy"] = round(float(test_acc), 4)
    meta["note"] = f"Compact weighted training. indian_weight={args.indian_weight}."
    meta["indianWeight"] = args.indian_weight
    meta["indianLikeSamples"] = indian_like_count
    meta["augmentedSamples"] = int(len(X))
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("trained_test_accuracy", round(float(test_acc), 4))
    print("samples", len(X))
    print("indian_like", indian_like_count)


if __name__ == "__main__":
    main()
