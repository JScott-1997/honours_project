import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class DAICTextDataset(Dataset):
    def __init__(self, dataframe, max_len=128):
        self.texts = dataframe["transcript"]
        self.labels = dataframe["phq8_score"]
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts.iloc[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels.iloc[idx], dtype=torch.float)
        }