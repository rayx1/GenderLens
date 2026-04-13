# GenderLens v1.0 - github.com/your-username/gender-lens
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data" / "name_gender_dict.json"
OUT_REPORT = ROOT / "train" / "indian_dataset_merge_report.json"

MALE_CSV = Path(r"C:/Users/HP/OneDrive - MITS SCHOOL OF BIOTECHNOLOGY/Downloads/Indian-Male-Names.csv")
FEMALE_CSV = Path(r"C:/Users/HP/OneDrive - MITS SCHOOL OF BIOTECHNOLOGY/Downloads/Indian-Female-Names.csv")
UNLABELED_CSV = Path(r"C:/Users/HP/OneDrive - MITS SCHOOL OF BIOTECHNOLOGY/Downloads/Indian_Names.csv")


def normalize_name(v: str) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    first = re.split(r"[\s,._-]+", s)[0]
    first = re.sub(r"[^a-z]", "", first)
    return first[:20]


def load_names(path: Path, col: str):
    df = pd.read_csv(path)
    return [normalize_name(x) for x in df[col].tolist()]


def main():
    d = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    male_raw = load_names(MALE_CSV, "name")
    female_raw = load_names(FEMALE_CSV, "name")
    unlabeled_raw = load_names(UNLABELED_CSV, "Name")

    male = {n for n in male_raw if len(n) >= 2}
    female = {n for n in female_raw if len(n) >= 2}
    unlabeled = {n for n in unlabeled_raw if len(n) >= 2}

    overlap = male & female
    male_only = male - overlap
    female_only = female - overlap

    added_m, added_f, set_unknown = 0, 0, 0

    for n in sorted(male_only):
        prev = d.get(n)
        d[n] = {"gender": "Male", "confidence": 0.94}
        if not prev:
            added_m += 1

    for n in sorted(female_only):
        prev = d.get(n)
        d[n] = {"gender": "Female", "confidence": 0.94}
        if not prev:
            added_f += 1

    for n in sorted(overlap):
        d[n] = {"gender": "Unknown", "confidence": 0.58}
        set_unknown += 1

    # incorporate unlabeled names as low-confidence unknown only if absent
    added_unlabeled_unknown = 0
    for n in sorted(unlabeled):
        if n not in d:
            d[n] = {"gender": "Unknown", "confidence": 0.52}
            added_unlabeled_unknown += 1

    ordered = {k: d[k] for k in sorted(d)}
    DICT_PATH.write_text(json.dumps(ordered, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    report = {
        "male_unique": len(male),
        "female_unique": len(female),
        "overlap": len(overlap),
        "male_only": len(male_only),
        "female_only": len(female_only),
        "added_new_male": added_m,
        "added_new_female": added_f,
        "set_unknown_from_overlap": set_unknown,
        "unlabeled_unique": len(unlabeled),
        "added_unlabeled_unknown": added_unlabeled_unknown,
        "final_dictionary_size": len(ordered),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
