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

INDIAN_CORE_NAMES = {
    "subhashree", "lipsa", "smruti", "sasmita", "jyotirmayee", "rashmita", "sujata", "lopamudra", "sarada",
    "itishree", "debasmita", "pallabi", "ipsita", "pratima", "nibedita", "sanghamitra", "chitralekha", "sucharita",
    "hemalata", "lilavati", "malati", "kuntala", "binodini", "biswajit", "satyajit", "debasis", "sipun", "niladri",
    "trilochan", "pranab", "subrat", "biswaranjan", "bichitrananda", "hrudananda", "sudhansu", "amarendra", "rabindra",
    "jatindra", "patitapaban", "brundaban", "rahul", "amit", "vikram", "sunil", "anil", "ravi", "ashok", "vinod",
    "rajesh", "mukesh", "manoj", "sanjay", "ajay", "vijay", "vivek", "gaurav", "ananya", "divya", "pooja", "neha",
    "shreya", "aishwarya", "kavita", "rekha", "preeti", "meera", "murugan", "selvam", "senthil", "karthik", "manikandan",
    "venkatesh", "narayana", "srinivas", "raju", "krishna", "padmavathi", "annapurna", "bhavani", "kalyani", "arpita",
    "barnali", "chandrima", "debjani", "indrani", "kaushik", "partha", "tanmoy", "vaishnavi", "lakshmi", "meenakshi",
    "saraswathi", "vijayalakshmi", "thenmozhi", "mangayarkarasi", "jagannath", "madhusudan", "laxmidhar", "kailash"
}

INDIAN_SUFFIXES = (
    "jit", "jeet", "nath", "esh", "endra", "kumar", "swamy", "murthy", "priya", "lata", "lekha", "mita",
    "deep", "preet", "anshu", "isha", "vathi", "amma", "appa", "reddy", "rao", "das", "deb", "bhai", "raj"
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


def is_indian_name(name: str) -> bool:
    n = normalize_name(name)
    if n in INDIAN_CORE_NAMES:
        return True
    if any(n.startswith(p) for p in INDIAN_PREFIXES):
        return True
    if any(n.endswith(s) for s in INDIAN_SUFFIXES):
        return True
    return False


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


def make_augmented_samples(data_dict, indian_boost: int):
    names, labels = [], []
    indian_count = 0

    for name, info in data_dict.items():
        gender = info.get("gender")
        if gender not in {"Male", "Female"}:
            continue

        n = normalize_name(name)
        if len(n) < 2:
            continue

        label = 1 if gender == "Male" else 0
        conf = float(info.get("confidence", 0.85))

        repeat = 1
        if is_indian_name(n):
            repeat += indian_boost
            indian_count += 1
        if conf >= 0.95:
            repeat += 1

        for _ in range(repeat):
            names.append(n)
            labels.append(label)

            if len(n) > 3:
                names.append(n[:-1]); labels.append(label)
            if len(n) > 4:
                names.append(n[:-2]); labels.append(label)
            if len(n) < 18:
                names.append(n + ("a" if label == 0 else "n")); labels.append(label)

            if is_indian_name(n):
                if len(n) < 19:
                    names.append(n + ("i" if label == 0 else "h")); labels.append(label)
                if len(n) > 5:
                    names.append(n[:-1] + ("a" if label == 0 else "k")); labels.append(label)

    return names, labels, indian_count


def main():
    parser = argparse.ArgumentParser(description="Train GenderLens with Indian-name boosted sampling")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--indian-boost", type=int, default=4, help="extra repeat factor for Indian-like names")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    data = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    names, labels, indian_count = make_augmented_samples(data, indian_boost=max(0, args.indian_boost))

    X = np.stack([encode_name(n) for n in names]).astype(np.float32)
    y = np.array(labels, dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    with (MODEL_DIR / "weights.bin").open("wb") as f:
        for arr in model.get_weights():
            flat = np.asarray(arr, dtype=np.float32).ravel(order="C")
            f.write(struct.pack("<" + "f" * flat.size, *flat.tolist()))

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta["trainingDataset"] = "Bundled dictionary with Indian-boosted sampling and augmentation"
    meta["trainedAt"] = datetime.utcnow().isoformat() + "Z"
    meta["testAccuracy"] = round(float(test_acc), 4)
    meta["note"] = f"Indian-name boost enabled (boost={args.indian_boost}). For best real-world coverage, train with larger external Indian datasets."
    meta["indianBoost"] = args.indian_boost
    meta["indianCoreNamesMatched"] = indian_count
    meta["augmentedSamples"] = len(X)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("trained_test_accuracy", round(float(test_acc), 4))
    print("samples", len(X))
    print("indian_core_matched", indian_count)


if __name__ == "__main__":
    main()
