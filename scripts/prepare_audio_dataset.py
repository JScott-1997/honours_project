import os
import pandas as pd

# --------------------------
# File paths
# --------------------------
FEATURES_CSV = "data/audio_features/audio_features_all.csv"
DEV_LABELS = "data/labels/dev_split_Depression_AVEC2017.csv"
TEST_LABELS = "data/labels/test_split_Depression_AVEC2017.csv"
OUTPUT_CSV = "data/audio_features/audio_dataset_final.csv"

# --------------------------
# Load and clean audio features
# --------------------------
print("📥 Loading raw audio features...")
features_df = pd.read_csv(FEATURES_CSV)

if "participant_id" not in features_df.columns:
    raise ValueError("❌ 'participant_id' column missing from features file.")

# Clean participant ID: e.g., '302_AUDIO' → 302
features_df["participant_id"] = (
    features_df["participant_id"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(int)
)

# --------------------------
# Load PHQ-8 scores from train and dev
# --------------------------
print("📥 Loading train and dev PHQ-8 labels...")
train_df = pd.read_csv(os.path.join("data/labels", "train_split_Depression_AVEC2017.csv"))
dev_df = pd.read_csv(os.path.join("data/labels", "dev_split_Depression_AVEC2017.csv"))

label_df = pd.concat([train_df, dev_df], ignore_index=True)
label_df["participant_id"] = label_df["Participant_ID"].astype(int)
label_df = label_df[["participant_id", "PHQ8_Score"]]


# --------------------------
# Merge and save
# --------------------------
print("🔗 Merging features with labels...")
merged_df = pd.merge(features_df, label_df, on="participant_id", how="inner")

print(f"📊 Merged dataset shape: {merged_df.shape}")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Final dataset saved to: {OUTPUT_CSV}")
