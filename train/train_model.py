# GenderLens v1.0 - github.com/your-username/gender-lens
import argparse
import json
import os
import re
import struct
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def normalize_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return ""
    return re.split(r"[\s,.-]+", text)[0][:20]


def normalize_gender(value) -> int:
    s = str(value).strip().lower()
    if s in {"m", "male", "1", "true"}:
        return 1
    if s in {"f", "female", "0", "false"}:
        return 0
    raise ValueError(f"Unsupported gender value: {value}")


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
        tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def generate_random_weights(output_folder: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    output = os.path.join(output_folder, "weights.bin")
    rng = np.random.default_rng(42)
    shapes = [3 * 28 * 64, 64, 3 * 64 * 32, 32, 32 * 32, 32, 32 * 1, 1]
    with open(output, "wb") as f:
        for size in shapes:
            arr = rng.uniform(-0.05, 0.05, size).astype(np.float32)
            f.write(struct.pack("<" + "f" * size, *arr.tolist()))
    return output


def main():
    parser = argparse.ArgumentParser(description="Train GenderLens Conv1D model and export to TF.js")
    parser.add_argument("--data", default="names_dataset.csv", help="Training CSV with columns: name, gender")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--output", default=os.path.join("..", "model"), help="Output folder")
    parser.add_argument("--random-init", action="store_true", help="Only generate random weights.bin")
    args = parser.parse_args()

    if args.random_init:
        path = generate_random_weights(args.output)
        print(f"Random weights generated at: {path}")
        print("Ensure model.json exists in output folder.")
        return

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training CSV not found: {args.data}")

    df = pd.read_csv(args.data)
    if "name" not in df.columns or "gender" not in df.columns:
        raise ValueError("CSV must include columns: name, gender")

    names, labels = [], []
    for _, row in df.iterrows():
        name = normalize_name(row["name"])
        if len(name) < 2 or len(name) > 25:
            continue
        try:
            y = normalize_gender(row["gender"])
        except ValueError:
            continue
        names.append(name)
        labels.append(y)

    if len(names) < 100:
        raise ValueError("Need at least 100 valid rows after preprocessing")

    X = np.stack([encode_name(n) for n in names])
    y = np.array(labels, dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = build_model()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    import tensorflowjs as tfjs
    os.makedirs(args.output, exist_ok=True)
    tfjs.converters.save_keras_model(model, args.output)

    with open(os.path.join(args.output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "modelName": "GenderLens-Conv1D",
            "version": "1.0.0",
            "architecture": "Conv1D character-level classifier",
            "inputShape": [20, 28],
            "outputClasses": ["Female", "Male"],
            "trainedAt": datetime.utcnow().isoformat() + "Z",
            "testAccuracy": float(test_acc)
        }, f, indent=2)

    print("Model exported to:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
