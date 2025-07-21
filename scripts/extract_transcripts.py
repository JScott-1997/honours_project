# scripts/extract_transcripts.py

import os
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/transcripts")

def extract_transcript_from_csv(df: pd.DataFrame, participant_id: str) -> str:
    """
    Given a raw transcript dataframe, return a cleaned participant-only text.
    """
    # Try to find the utterance column (might be 'value', 'utterance', etc.)
    text_col = next((c for c in df.columns if c.lower() in {"utterance", "value", "text"}), None)
    speaker_col = next((c for c in df.columns if c.lower() in {"speaker", "participant"}), None)

    if text_col is None or speaker_col is None:
        raise ValueError(f"Missing expected columns in transcript for {participant_id}")

    # Keep only participant responses, remove NaNs, strip whitespace
    participant_lines = df[df[speaker_col].str.lower() == "participant"][text_col]
    participant_lines = participant_lines.dropna().astype(str).str.strip()

    return " ".join(participant_lines.tolist())

def process_zip_file(zip_path: Path):
    """
    Extract the *_TRANSCRIPT.csv file from a zip and write cleaned participant transcript.
    """
    participant_id = zip_path.stem.split("_")[0]  # e.g., P001

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = next((f for f in zf.namelist() if f.endswith("_TRANSCRIPT.csv")), None)
        if csv_name is None:
            print(f"⚠️  No transcript CSV found in {zip_path.name}")
            return

        with zf.open(csv_name) as f:
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                f.seek(0)
                df = pd.read_csv(f, encoding="latin-1")

        try:
            transcript = extract_transcript_from_csv(df, participant_id)
        except Exception as e:
            print(f"❌ Error processing {participant_id}: {e}")
            return

        # Save the transcript as a plain .txt file
        out_path = OUT_DIR / f"{participant_id}_transcript.txt"
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(transcript)

        print(f"✅ Saved {out_path.name}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(RAW_DIR.glob("*_P.zip"))
    if not zip_files:
        print(f"❌ No *_P.zip files found in {RAW_DIR}")
        return

    for zip_path in tqdm(zip_files, desc="📦 Extracting transcripts"):
        process_zip_file(zip_path)

    print("🎉 All transcripts extracted.")

if __name__ == "__main__":
    main()