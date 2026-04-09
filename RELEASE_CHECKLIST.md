<!-- GenderLens v1.0 - github.com/your-username/gender-lens -->
# Release Checklist

## Package generated
- `dist/gender-lens-v1.0.0.zip`

## Pre-publish checks
- Open `chrome://extensions`
- Load unpacked from this `gender-lens` folder
- Verify popup opens and shows model status
- Upload a sample `.csv` and run predictions
- Confirm CSV download works

## Chrome Web Store notes
- If publishing to Web Store, replace placeholder icons with branded final icons.
- Keep all runtime assets bundled in the package (`lib/`, `model/`, `data/`).
- Consider retraining model and replacing `model/weights.bin` for production quality.
