import os
import zipfile
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_ZIP_DIR = "data/raw"
LABEL_DIR = "data/labels"
OUTPUT_PATH = "data/audio/audio_with_labels.csv"
AUDIO_DIR = "data/audio/wav"

# Step 1: Load PHQ-8 labels
def load_labels(label_dir):
    train_df = pd.read_csv(os.path.join(label_dir, "train_split_Depression_AVEC2017.csv"))
    dev_df = pd.read_csv(os.path.join(label_dir, "dev_split_Depression_AVEC2017.csv"))
    combined = pd.concat([train_df, dev_df], ignore_index=True)
    combined["Participant_ID"] = combined["Participant_ID"].apply(lambda x: f"P{int(x):03d}")
    return dict(zip(combined["Participant_ID"], combined["PHQ8_Score"]))

# Step 2: Extract audio paths from zipped archives
def extract_audio_entries(zip_dir, label_lookup, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    entries = []

    for filename in tqdm(os.listdir(zip_dir), desc="🔊 Extracting audio"):
        if not filename.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, filename)
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                wav_file = next((f for f in archive.namelist() if f.endswith(".wav")), None)
                if not wav_file:
                    continue

                participant_id = f"P{wav_file.split('_')[0]}"
                if participant_id not in label_lookup:
                    continue

                out_path = os.path.join(audio_dir, os.path.basename(wav_file))
                if not os.path.exists(out_path):
                    with archive.open(wav_file) as f_in, open(out_path, "wb") as f_out:
                        f_out.write(f_in.read())

                entries.append({
                    "participant_id": participant_id,
                    "audio_path": out_path.replace("\\", "/"),
                    "phq8_score": label_lookup[participant_id]
                })

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    return pd.DataFrame(entries)

# Step 3: Save to CSV
def save_csv(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")

# Main
if __name__ == "__main__":
    labels = load_labels(LABEL_DIR)
    audio_df = extract_audio_entries(RAW_ZIP_DIR, labels, AUDIO_DIR)

    if not audio_df.empty:
        save_csv(audio_df, OUTPUT_PATH)
    else:
        print("⚠️ No audio data matched.")
