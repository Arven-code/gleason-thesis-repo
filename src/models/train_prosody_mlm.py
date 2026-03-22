import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DATA_DIRS = [
    Path(r"/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Dinner/echo_datasets_v2"),
    Path(r"/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/FINAL"),
]
GLOB_PATTERNS = [
    "*_training_echo_STRICT.csv",
    "*_training_echo_v2.csv",
    "combined_training_echo_STRICT.csv",
    "combined_training_echo_v2.csv",
]

MODEL_NAME = "bert-base-uncased"
PROS_TOKEN = "[PROS]"
MAX_LEN = 256
MLM_PROB = 0.15

PROSODY_COLS = [
    "f0_mean","f0_std","f0_min","f0_max",
    "f0_median","f0_iqr","f0_slope",
    "voicing_rate","frames"
]
ADD_HAS_PROSODY_BIT = True
PROS_HIDDEN = 128

OUT_DIR = "out_mlm_pros"
ARTIFACTS_DIR = "artifacts"
SEED = 42


def find_csvs():
    files = []
    for data_dir in DATA_DIRS:
        for pat in GLOB_PATTERNS:
            files.extend(sorted(data_dir.glob(pat)))
    uniq, seen = [], set()
    for p in files:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


def load_and_concat(csv_paths):
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = p.name
            df["__source_dir"] = str(p.parent)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No training CSVs found under: {', '.join(map(str, DATA_DIRS))}")
    return pd.concat(dfs, ignore_index=True)


def build_dataset(df: pd.DataFrame) -> Dataset:
    if "text" in df.columns:
        df["text"] = df["text"].astype(str)
    elif "child_text" in df.columns:
        df["text"] = df["child_text"].astype(str)
    else:
        raise ValueError("No 'text' or 'child_text' column found in CSVs.")

    if "child_label" in df.columns:
        df["label_acc"] = (df["child_label"] == 2.0).astype(int)
    elif "label_acc" in df.columns:
        df["label_acc"] = df["label_acc"].astype(int)
    else:
        df["label_acc"] = 0

    for c in PROSODY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    has_pros = (~df["f0_mean"].isna()).astype(int)
    df[PROSODY_COLS] = df[PROSODY_COLS].fillna(0.0)

    pros_cols = PROSODY_COLS.copy()
    if ADD_HAS_PROSODY_BIT:
        df["has_prosody"] = has_pros
        pros_cols += ["has_prosody"]

    df["pros_vec"] = df[pros_cols].apply(lambda r: np.array(r.values, dtype=np.float32), axis=1)

    keep = ["text","label_acc","pros_vec","__source_file","__source_dir"]
    if "child_row_idx" in df.columns:
        keep.append("child_row_idx")
    if "child_id" in df.columns:
        keep.append("child_id")
    return Dataset.from_pandas(df[keep])


class ProsodyProjector(nn.Module):
    def __init__(self, hidden_size: int, pros_dim: int, pros_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pros_dim, pros_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(pros_hidden, hidden_size),
        )

    def forward(self, f):
        return self.mlp(f)


class ProsodyCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, shuffle_prosody=False):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.shuffle = shuffle_prosody

    def __call__(self, examples):
        base_examples = [{k: e[k] for k in ("input_ids","attention_mask")} for e in examples]
        batch = self.base(base_examples)

        if any("label_acc" in e for e in examples):
            batch["label_acc"] = torch.tensor([e.get("label_acc", 0) for e in examples], dtype=torch.long)

        pros_lists, any_pros = [], False
        for e in examples:
            if "pros_vec" in e and e["pros_vec"] is not None:
                pros_lists.append(list(np.asarray(e["pros_vec"], dtype=np.float32)))
                any_pros = True
            else:
                pros_lists.append(None)

        if any_pros:
            first = next((p for p in pros_lists if p is not None), None)
            dim = len(first) if first is not None else 1
            filled = [p if p is not None else [0.0] * dim for p in pros_lists]
            if self.shuffle and len(filled) > 1:
                idx = np.arange(len(filled))
                np.random.shuffle(idx)
                filled = [filled[i] for i in idx]
            batch["pros_vec"] = torch.tensor(filled, dtype=torch.float32)

        return batch


class ProsodyTrainer(Trainer):
    def __init__(self, *args, use_prosody=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prosody = use_prosody

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(device)

        if self.use_prosody and "pros_vec" in inputs:
            pros_vec = inputs["pros_vec"].to(device)
            word_emb = model.get_input_embeddings()(input_ids)
            pros_emb = model.pros_projector(pros_vec)
            word_emb[:, 0, :] = pros_emb
            outputs = model(inputs_embeds=word_emb, attention_mask=attention_mask, labels=labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def make_tok_map(use_prosody, tokenizer, pros_token_id, prosody_cols, add_has_prosody_bit, max_len):
    def tok_map(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len)

        if use_prosody:
            new_ids, new_att = [], []
            for ids, att in zip(enc["input_ids"], enc["attention_mask"]):
                new_ids.append([pros_token_id] + ids)
                new_att.append([1] + att)
            enc["input_ids"] = new_ids
            enc["attention_mask"] = new_att

        enc["label_acc"] = list(batch.get("label_acc", [0] * len(enc["input_ids"])))

        pros_dim = len(prosody_cols) + (1 if add_has_prosody_bit else 0)
        if "pros_vec" in batch:
            enc["pros_vec"] = [list(np.asarray(v, dtype=np.float32)) for v in batch["pros_vec"]]
        else:
            zero = [0.0] * pros_dim
            enc["pros_vec"] = [zero for _ in enc["input_ids"]]

        return enc
    return tok_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-prosody", action="store_true", help="Disable prosody features")
    parser.add_argument("--shuffle-prosody", action="store_true", help="Shuffle prosody vectors")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    csvs = find_csvs()
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under: {', '.join(map(str, DATA_DIRS))}")

    print("[INFO] Found CSVs:")
    per_dir = {}
    for p in csvs:
        print(" -", p)
        per_dir.setdefault(str(p.parent), 0)
        per_dir[str(p.parent)] += 1
    print("[INFO] Per-folder counts:")
    for d, n in per_dir.items():
        print(f"   {d} -> {n} file(s)")

    df_all = load_and_concat(csvs)
    ds = build_dataset(df_all)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    use_prosody = not args.no_prosody
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    if use_prosody:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [PROS_TOKEN]})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        pros_token_id = tokenizer.convert_tokens_to_ids(PROS_TOKEN)
    else:
        pros_token_id = None

    tok_map = make_tok_map(
        use_prosody=use_prosody,
        tokenizer=tokenizer,
        pros_token_id=pros_token_id,
        prosody_cols=PROSODY_COLS,
        add_has_prosody_bit=ADD_HAS_PROSODY_BIT,
        max_len=MAX_LEN,
    )

    ds = ds.map(tok_map, batched=True, remove_columns=ds.column_names)

    if use_prosody:
        pros_dim = len(PROSODY_COLS) + (1 if ADD_HAS_PROSODY_BIT else 0)
        model.pros_projector = ProsodyProjector(
            hidden_size=model.config.hidden_size,
            pros_dim=pros_dim,
            pros_hidden=PROS_HIDDEN
        )

    collator = ProsodyCollator(tokenizer, mlm_probability=MLM_PROB, shuffle_prosody=args.shuffle_prosody)

    run_tag = "textonly" if not use_prosody else ("shufflePROS" if args.shuffle_prosody else "prosody")
    out_dir = f"{OUT_DIR}_{run_tag}"

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        report_to=["none"],
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = ProsodyTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        use_prosody=use_prosody,
    )

    trainer.train()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    export_cols = ["text","label_acc"] + PROSODY_COLS + (["has_prosody"] if "has_prosody" in df_all.columns else [])
    for c in export_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
    df_all.to_csv(Path(ARTIFACTS_DIR) / f"train_rows_for_glmm_{run_tag}.csv", index=False)

    with open(Path(ARTIFACTS_DIR) / f"training_config_{run_tag}.json", "w") as f:
        json.dump({
            "data_dirs": [str(d) for d in DATA_DIRS],
            "loaded_files": [p.name for p in csvs],
            "model_name": MODEL_NAME,
            "prosody_cols": PROSODY_COLS,
            "add_has_prosody_bit": ADD_HAS_PROSODY_BIT,
            "max_len": MAX_LEN,
            "mlm_prob": MLM_PROB,
            "batch_size": args.batch,
            "lr": args.lr,
            "epochs": args.epochs,
            "use_prosody": use_prosody,
            "shuffle_prosody": args.shuffle_prosody,
        }, f, indent=2)

    print("\n[OK] Training complete.")
    print(f"- Checkpoints: {out_dir}")
    print(f"- GLMM input:  {ARTIFACTS_DIR}/train_rows_for_glmm_{run_tag}.csv")
    print(f"- Config:      {ARTIFACTS_DIR}/training_config_{run_tag}.json")


if __name__ == "__main__":
    main()
