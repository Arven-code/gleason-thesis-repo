import os
import json
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW


class ProsodyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["transcript"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        prosody = torch.tensor([
            row["f0_mean"], row["f0_median"], row["f0_std"], row["f0_iqr"],
            row["f0_p10"], row["f0_p90"], row["f0_range"], row["f0_cv"],
            row["f0_slope"], row["dF0_iqr"], row["voicing_rate"], row["frames"]
        ], dtype=torch.float)

        label = int(row["label"] - 1)
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "prosody": prosody,
            "label": torch.tensor(label)
        }


class ProsodyBERT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.prosody_proj = nn.Linear(12, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, prosody):
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        pros = self.prosody_proj(prosody)
        combined = cls + pros
        logits = self.classifier(combined)
        return logits


def load_all_csvs(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), files


def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    base = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model = ProsodyBERT(base)

    df, files = load_all_csvs("DATA_PATH_HERE")

    with open("artifacts/loaded_files.json", "w") as f:
        json.dump(files, f, indent=2)

    df = df.sample(frac=1, random_state=42)
    split = int(0.8 * len(df))
    train_df = df[:split]
    val_df = df[split:]

    train_ds = ProsodyDataset(train_df, tokenizer)
    val_ds = ProsodyDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    counts = train_df["label"].value_counts()
    weights = torch.tensor([
        1.0 / counts[1.0],
        1.0 / counts[2.0]
    ])

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(8):
        model.train()
        for batch in train_loader:
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["prosody"]
            )
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), "out_mlm_cls_prosody/model.pt")


if __name__ == "__main__":
    train()
