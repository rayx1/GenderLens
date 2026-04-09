# GenderLens v1.0 - github.com/your-username/gender-lens
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
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    data = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    names = []
    labels = []
    for name, info in data.items():
        gender = info.get("gender")
        if gender not in {"Male", "Female"}:
            continue
        n = normalize_name(name)
        if len(n) < 2:
            continue
        names.append(n)
        labels.append(1 if gender == "Male" else 0)

    aug_names, aug_labels = [], []
    for n, y in zip(names, labels):
        if len(n) > 3:
            aug_names.append(n[:-1]); aug_labels.append(y)
        if len(n) > 4:
            aug_names.append(n[:-2]); aug_labels.append(y)
        if len(n) < 18:
            aug_names.append(n + ("a" if y == 0 else "n")); aug_labels.append(y)

    names.extend(aug_names)
    labels.extend(aug_labels)

    X = np.stack([encode_name(n) for n in names]).astype(np.float32)
    y = np.array(labels, dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    weights = model.get_weights()
    with (MODEL_DIR / "weights.bin").open("wb") as f:
        for arr in weights:
            flat = np.asarray(arr, dtype=np.float32).ravel(order="C")
            f.write(struct.pack("<" + "f" * flat.size, *flat.tolist()))

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta["trainingDataset"] = "Built from bundled name_gender_dict.json (Male/Female entries only, augmented)"
    meta["trainedAt"] = datetime.utcnow().isoformat() + "Z"
    meta["testAccuracy"] = round(float(test_acc), 4)
    meta["note"] = "Weights trained from bundled dictionary; replace with larger curated dataset for best real-world results."
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("trained_test_accuracy", round(float(test_acc), 4))
    print("samples", len(X))


if __name__ == "__main__":
    main()
