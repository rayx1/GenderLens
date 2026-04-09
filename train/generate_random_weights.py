# GenderLens v1.0 - github.com/your-username/gender-lens
import argparse
import os
import struct
import numpy as np


def generate_random_weights(output_file: str):
    shapes = [3 * 28 * 64, 64, 3 * 64 * 32, 32, 32 * 32, 32, 32 * 1, 1]
    rng = np.random.default_rng(42)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        for size in shapes:
            arr = rng.uniform(-0.05, 0.05, size).astype(np.float32)
            f.write(struct.pack("<" + "f" * size, *arr.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random weights.bin for GenderLens")
    parser.add_argument("--output", default=os.path.join("..", "model", "weights.bin"), help="Output path for weights.bin")
    args = parser.parse_args()
    generate_random_weights(args.output)
    print(f"Generated random weights at: {os.path.abspath(args.output)}")
