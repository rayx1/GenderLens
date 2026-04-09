<!-- GenderLens v1.0 - github.com/your-username/gender-lens -->
# README_TRAINING

## 1. Recommended free datasets
- Kaggle: "gender-by-name" dataset (147K names, multiple nationalities)
- Kaggle: "indian-names-gender" dataset (60K Indian names)
- Download both, merge, and deduplicate.

## 2. How to run training
```bash
pip install -r requirements.txt
python train_model.py --data merged_names.csv --epochs 50
```

## 3. Expected output
Accuracy is typically in the ~85% to ~92% range depending on dataset quality.

## 4. Copy trained model into extension
```bash
cp model.json ../model/model.json
cp weights.bin ../model/weights.bin
```

If tensorflowjs writes shard files (for example `group1-shard1of1.bin`) copy those shard files too.

## 5. Reload extension in Chrome
- Open `chrome://extensions`
- Click reload on GenderLens after replacing model files
