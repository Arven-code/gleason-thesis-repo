# train_prosody_mlm_cls.py
# Main training script for the BERT + prosody classifier.
# It also saves the evaluation figures and metrics.

import os, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GleasonDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["transcript"])
        label = int(row["label"] == 2.0)

        prosody_cols = [
            "f0_mean","f0_median","f0_std","f0_iqr","f0_p10","f0_p90",
            "f0_range","f0_cv","f0_slope","dF0_iqr","voicing_rate","frames"
        ]

        prosody = row[prosody_cols].fillna(0).values.astype(np.float32)

        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "prosody": torch.tensor(prosody),
            "label": torch.tensor(label)
        }

class ProsodyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(12, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, prosody):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]

        p = self.proj(prosody)
        x = cls + p

        logits = self.classifier(x)
        return logits

def load_data(paths):
    dfs = []
    for p in paths:
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)

def train_model(train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_ds = GleasonDataset(train_df, tokenizer)
    val_ds = GleasonDataset(val_df, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8)

    model = ProsodyBERT().to(DEVICE)

    y = (train_df["label"] == 2.0).astype(int)
    w_pos = len(y) / (2 * y.sum())
    w_neg = len(y) / (2 * (len(y) - y.sum()))

    weights = torch.tensor([w_neg, w_pos]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(8):
        model.train()
        for batch in train_dl:
            opt.zero_grad()

            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["prosody"].to(DEVICE)
            )

            loss = loss_fn(logits, batch["label"].to(DEVICE))
            loss.backward()
            opt.step()

    return model, val_dl

def evaluate(model, val_dl):
    model.eval()
    preds, probs, labels = [], [], []

    with torch.no_grad():
        for batch in val_dl:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["prosody"].to(DEVICE)
            )

            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (p > 0.5).astype(int)

            probs.extend(p)
            preds.extend(pred)
            labels.extend(batch["label"].numpy())

    cm = confusion_matrix(labels, preds)

    roc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    os.makedirs("artifacts", exist_ok=True)

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(labels, probs)
    plt.savefig("artifacts/roc_curve.png")
    plt.close()

    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(labels, probs)
    plt.savefig("artifacts/pr_curve.png")
    plt.close()

    with open("artifacts/val_metrics.json", "w") as f:
        json.dump({"ROC_AUC": float(roc), "AP": float(ap)}, f)

if __name__ == "__main__":
    paths = json.load(open("artifacts/loaded_files.json"))

    df = load_data(paths)
    df = df.sample(frac=1, random_state=42)

    split = int(len(df) * 0.8)
    train_df = df[:split]
    val_df = df[split:]

    model, val_dl = train_model(train_df, val_df)
    evaluate(model, val_dl)
