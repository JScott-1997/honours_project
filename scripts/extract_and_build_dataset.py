import os
import zipfile
import pandas as pd
from tqdm import tqdm
from io import StringIO

# ---------------------------
# Configuration
# ---------------------------
RAW_ZIP_DIR = "data/raw"
LABEL_DIR = "data/labels"
OUTPUT_PATH = "data/transcripts/transcripts_with_labels.csv"
EXPECTED_COLUMNS = ["start_time", "stop_time", "speaker", "value"]

# ---------------------------
# Step 1: Load Labels
# ---------------------------
def load_labels(label_dir):
    print("📁 Loading label files...")
    train_df = pd.read_csv(os.path.join(label_dir, "train_split_Depression_AVEC2017.csv"))
    dev_df = pd.read_csv(os.path.join(label_dir, "dev_split_Depression_AVEC2017.csv"))
    combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    combined_df["Participant_ID"] = combined_df["Participant_ID"].apply(lambda pid: f"P{int(pid):03d}")
    label_lookup = dict(zip(combined_df["Participant_ID"], combined_df["PHQ8_Score"]))
    print(f"🔢 Loaded {len(label_lookup)} participant labels.")
    print(f"🗞 Example participant IDs: {list(label_lookup.keys())[:5]}")
    return label_lookup

# ---------------------------
# Step 2: Extract Transcripts
# ---------------------------
def extract_transcripts(zip_dir, label_lookup):
    print("🧠 Extracting transcripts from zip files...")
    extracted_data = []
    zip_filenames = [name for name in os.listdir(zip_dir) if name.endswith(".zip")]

    for zip_filename in tqdm(zip_filenames, desc="📦 Processing transcripts"):
        zip_path = os.path.join(zip_dir, zip_filename)

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                transcript_file = next((f for f in archive.namelist() if f.endswith("_TRANSCRIPT.csv")), None)
                if not transcript_file:
                    continue

                participant_num = transcript_file.split("_")[0]
                participant_id = f"P{participant_num}"

                if participant_id not in label_lookup:
                    continue

                raw_text = archive.read(transcript_file).decode("latin-1")

                transcript_df = pd.read_csv(StringIO(raw_text), delimiter="\t")

                if not all(column in transcript_df.columns for column in EXPECTED_COLUMNS):
                    print(f"⚠️ Skipping {participant_id}: unexpected columns {list(transcript_df.columns)}")
                    continue

                participant_lines = transcript_df[transcript_df["speaker"].str.lower() == "participant"]["value"]
                full_text = " ".join(str(text).strip() for text in participant_lines if pd.notnull(text))

                extracted_data.append({
                    "participant_id": participant_id,
                    "transcript_text": full_text,
                    "phq8_score": label_lookup[participant_id]
                })

        except Exception as error:
            print(f"❌ Failed to process {zip_filename}: {error}")

    if not extracted_data:
        print("❌ No valid transcripts extracted.")

    return pd.DataFrame(extracted_data)

# ---------------------------
# Step 3: Save Output
# ---------------------------
def save_to_csv(dataframe, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"✅ Saved processed dataset to: {output_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    label_lookup = load_labels(LABEL_DIR)
    result_df = extract_transcripts(RAW_ZIP_DIR, label_lookup)

    if not result_df.empty:
        save_to_csv(result_df, OUTPUT_PATH)
    else:
        print("⚠️ No data to save.")

if __name__ == "__main__":
    main()