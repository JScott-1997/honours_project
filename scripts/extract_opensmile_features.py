import os
import subprocess
import pandas as pd
from tqdm import tqdm

# ---------------------
# Configuration
# ---------------------
AUDIO_DIR = "data/audio/wav"
FEATURE_CSV_DIR = "data/audio_features"
OUTPUT_CSV = os.path.join(FEATURE_CSV_DIR, "audio_features_all.csv")

SMIL_PATH = r"C:\Users\jackb\Documents\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
CONFIG_PATH = r"C:\Users\jackb\Documents\opensmile-3.0.2-windows-x86_64\config\gemaps\v01a\GeMAPSv01a.conf"

# ---------------------
# Ensure output directory exists
# ---------------------
os.makedirs(FEATURE_CSV_DIR, exist_ok=True)

# ---------------------
# Extract openSMILE features
# ---------------------
print("🔊 Extracting features from WAV files...")
all_records = []

for filename in tqdm(os.listdir(AUDIO_DIR)):
    if not filename.endswith(".wav"):
        continue

    input_path = os.path.join(AUDIO_DIR, filename)
    participant_id = os.path.splitext(filename)[0].split("_")[0] 
    output_path = os.path.join(FEATURE_CSV_DIR, f"{participant_id}_features.csv")


    print(f"▶️ Processing: {input_path} → {output_path}")

    try:
        subprocess.run(
            [
                SMIL_PATH,
                "-C", CONFIG_PATH,
                "-I", input_path,
                "-csvoutput", output_path,
                "-appendcsv", "0"
            ],
            check=True,
            capture_output=True,
            text=True
        )

        # Read and clean CSV output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            df = pd.read_csv(output_path, sep=";")

            # Drop non-feature columns if present
            df = df.drop(columns=[col for col in df.columns if col.lower() in ["name", "file", "class"]], errors="ignore")
            df["participant_id"] = participant_id
            all_records.append(df)

            os.remove(output_path)

        else:
            print(f"⚠️ No output for {participant_id}")

    except subprocess.CalledProcessError as e:
        print(f"❌ openSMILE failed on {filename}")
        print(e.stderr)

# ---------------------
# Save all combined features
# ---------------------
if all_records:
    final_df = pd.concat(all_records, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ All features saved to: {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted.")
