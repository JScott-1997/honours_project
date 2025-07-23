import os
import re
import zipfile
from io import StringIO

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
RAW_ZIP_DIR   = "data/raw"
LABEL_DIR     = "data/labels"
OUTPUT_CSV    = "data/video/visual_features_with_labels.csv"
CONF_THRESH   = 0.90          # keep frames with high tracking confidence
STATS = ["mean", "std", "min", "p25", "p50", "p75", "max"]  # feature functionals

# -----------------------------
# 1. Load PHQ-8 labels
# -----------------------------
def load_labels(label_dir: str) -> dict:
    train = pd.read_csv(os.path.join(label_dir, "train_split_Depression_AVEC2017.csv"))
    dev   = pd.read_csv(os.path.join(label_dir, "dev_split_Depression_AVEC2017.csv"))
    df    = pd.concat([train, dev], ignore_index=True)
    df["Participant_ID"] = df["Participant_ID"].apply(lambda pid: f"P{int(pid):03d}")
    return dict(zip(df["Participant_ID"], df["PHQ8_Score"]))

# -----------------------------
# 2. Extract & aggregate 3-D landmarks
# -----------------------------
def process_zip(zip_path: str, label_lookup: dict) -> dict | None:
    """
    Return a dict of aggregated 3-D landmark statistics for one participant,
    or None if any prerequisite (file, label, valid frames) is missing.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_file = next((f for f in zf.namelist()
                         if f.endswith("_CLNF_features3D.txt")), None)
        if txt_file is None:
            return None

        participant_id = f"P{txt_file.split('_')[0]}"
        if participant_id not in label_lookup:
            return None

        raw_text = zf.read(txt_file).decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(raw_text), sep=r"\s+|,", engine="python")

        # --- identify meta columns flexibly
        cols_lc = {c.lower(): c for c in df.columns}
        success_col     = cols_lc.get("detection_success") or cols_lc.get("success")
        confidence_col  = cols_lc.get("confidence")        or cols_lc.get("conf")

        if success_col in df.columns:
            df = df[df[success_col] == 1]
        if confidence_col in df.columns:
            df = df[df[confidence_col] > CONF_THRESH]
        if df.empty:
            return None

        # --- keep only uppercase X / Y / Z landmark columns
        landmark_cols = [c for c in df.columns
                         if re.match(r"^[XYZ][0-9]+$", str(c))]
        if len(landmark_cols) != 204:
            # skip corrupted or non-standard files
            print(f"⚠️  {participant_id}: found {len(landmark_cols)} 3-D cols (need 204) – skipping.")
            return None

        # --- compute functionals
        arr = df[landmark_cols].to_numpy(dtype=np.float32)

        stats_dict = {}
        stats_dict.update({f"{col}_mean": v for col, v in zip(landmark_cols, arr.mean(axis=0))})
        stats_dict.update({f"{col}_std":  v for col, v in zip(landmark_cols, arr.std (axis=0))})
        stats_dict.update({f"{col}_min":  v for col, v in zip(landmark_cols, arr.min (axis=0))})
        stats_dict.update({f"{col}_p25":  v for col, v in zip(landmark_cols, np.percentile(arr, 25, axis=0))})
        stats_dict.update({f"{col}_p50":  v for col, v in zip(landmark_cols, np.percentile(arr, 50, axis=0))})
        stats_dict.update({f"{col}_p75":  v for col, v in zip(landmark_cols, np.percentile(arr, 75, axis=0))})
        stats_dict.update({f"{col}_max":  v for col, v in zip(landmark_cols, arr.max (axis=0))})

        # participant-level metadata
        stats_dict["participant_id"] = participant_id
        stats_dict["phq8_score"]     = label_lookup[participant_id]
        return stats_dict

# -----------------------------
# 3. Run over all zips
# -----------------------------
def extract_all(zip_dir: str, labels: dict) -> pd.DataFrame:
    records = []
    for fname in tqdm(os.listdir(zip_dir), desc="📦 Processing video zips"):
        if fname.lower().endswith(".zip"):
            rec = process_zip(os.path.join(zip_dir, fname), labels)
            if rec:
                records.append(rec)
    return pd.DataFrame(records)

# -----------------------------
# 4. Main
# -----------------------------
def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("📁 Loading labels …")
    labels = load_labels(LABEL_DIR)

    print("🧠 Extracting 3-D landmark functionals …")
    df = extract_all(RAW_ZIP_DIR, labels)

    if df.empty:
        print("❌ No visual features extracted.")
        return

    print(f"✅ Extracted {df.shape[0]} participants × {df.shape[1]-2} features")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
