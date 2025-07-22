import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Config
# ---------------------------
TEXT_CSV = "data/transcripts/transcripts_with_labels.csv"
AUDIO_CSV = "data/audio_features/audio_dataset_final.csv"
OUTPUT_DIR = "data/feature_analysis"

# ---------------------------
# Load
# ---------------------------
print("📥 Loading transcript and audio CSVs...")
text_df = pd.read_csv(TEXT_CSV)
audio_df = pd.read_csv(AUDIO_CSV)

# ---------------------------
# Prepare
# ---------------------------
# Ensure consistent participant_id format
text_df["participant_id"] = text_df["participant_id"].astype(int)
audio_df["participant_id"] = audio_df["participant_id"].astype(int)

merged = pd.merge(text_df, audio_df, on="participant_id", how="inner")


# Merge
print("🔗 Merging text and audio on participant_id...")
merged_df = pd.merge(audio_df, text_df[["participant_id", "phq8_score"]], on="participant_id", how="inner")

# Drop obvious non-feature columns
non_feature_cols = ["participant_id", "phq8_score", "audio_path", "transcript_text"]
feature_cols = [col for col in merged_df.columns if col not in non_feature_cols]

# ---------------------------
# Basic Stats
# ---------------------------
print("📊 Feature statistics:")
summary_stats = merged_df[feature_cols].describe().transpose()
print(summary_stats)

# Save summary
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_stats.to_csv(os.path.join(OUTPUT_DIR, "feature_summary.csv"))

# ---------------------------
# Correlation Heatmap
# ---------------------------
print("📈 Generating correlation heatmap...")
correlation = merged_df[feature_cols + ["phq8_score"]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation[["phq8_score"]].drop("phq8_score"), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature–PHQ-8 Correlations")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlations.png"))
plt.close()

print("✅ Feature analysis complete.")