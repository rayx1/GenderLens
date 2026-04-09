<!-- GenderLens v1.0 - github.com/your-username/gender-lens -->
# GenderLens Chrome Extension

## What it does
GenderLens is an offline Chrome extension that reads CSV/XLS/XLSX files, predicts gender for each name, appends `Predicted Gender` and `Confidence %`, and exports the full result as CSV.

## Distribution-ready status
This folder is already bundled with:
- `lib/xlsx.full.min.js` (SheetJS)
- `lib/tf.min.js` (TensorFlow.js)
- `model/model.json` + `model/weights.bin`
- `data/name_gender_dict.json`

No runtime internet calls are required after installation.

## Load in Chrome
- Go to `chrome://extensions`
- Enable Developer Mode (top right toggle)
- Click "Load unpacked"
- Select the `gender-lens/` folder

## How prediction works
1. Dictionary lookup against `data/name_gender_dict.json`
2. Local TF.js Conv1D model inference
3. JS heuristic suffix/prefix fallback rules

If model loading fails, extension still works with dictionary + rules.

## Accuracy expectations
- Random weights + dictionary: about 70% (strong on dictionary names)
- Trained weights + dictionary + rules: about 85% to 92%

## Training your own model
See `train/README_TRAINING.md`.

## Adding more Indian names
Add entries in `data/name_gender_dict.json` in lowercase first-name format:
```json
"priya": { "gender": "Female", "confidence": 0.97 }
```

## Extension size breakdown
- SheetJS: ~1MB
- TF.js: ~1.4MB
- Model weights: ~50KB (current random init)
- Dictionary: ~115KB
- Total: ~3MB (well within Chrome limits)

## Known limitations
- Rare names outside dictionary depend on model quality.
- Word-like names (Joy, Grace, Faith) can be misclassified.
- Without training, model weights are random so model-only performance is weak.
