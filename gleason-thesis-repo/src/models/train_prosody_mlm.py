import json
import argparse
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

PROS_TOKEN = "[PROS]"
MLM_PROB = 0.15
SEED = 42

PROSODY_COLS = [
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "f0_median",
    "f0_iqr",
    "f0_slope",
    "voicing_rate",
    "frames",
]

ADD_HAS_PROSODY_BIT = True


def find_csvs(data_dirs, patterns):
    files = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"[WARN] Missing folder: {data_dir}")
            continue
        for pat in patterns:
            files.extend(sorted(data_dir.glob(pat)))

    unique_files = []
    seen = set()
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(p)

    return unique_files


def detect_text_column(df: pd.DataFrame) -> str:
    for col in ["text", "child_text", "transcript", "Transcript"]:
        if col in df.columns:
            return col
    raise ValueError(f"No usable text column found. Columns: {df.columns.tolist()}")


def detect_label_column(df: pd.DataFrame):
    for col in ["child_label", "label_acc", "label"]:
        if col in df.columns:
            return col
    return None


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
        raise FileNotFoundError("No readable CSV files found.")

    return pd.concat(dfs, ignore_index=True)


def build_dataset(df: pd.DataFrame) -> Dataset:
    text_col = detect_text_column(df)

    df["text"] = df[text_col].map(lambda x: "" if pd.isna(x) else str(x))
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""].copy()
    df = df[df["text"].str.lower() != "nan"].copy()

    label_col = detect_label_column(df)
    if label_col == "child_label":
        raw = pd.to_numeric(df["child_label"], errors="coerce")
        df["label_acc"] = (raw == 2.0).astype(int)
    elif label_col == "label_acc":
        df["label_acc"] = pd.to_numeric(df["label_acc"], errors="coerce").fillna(0).astype(int)
    elif label_col == "label":
        raw = pd.to_numeric(df["label"], errors="coerce")
        df["label_acc"] = (raw == 2.0).astype(int)
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
        pros_cols.append("has_prosody")

    df["pros_vec"] = df[pros_cols].apply(
        lambda r: np.array(r.values, dtype=np.float32),
        axis=1,
    )

    keep = ["text", "label_acc", "pros_vec", "__source_file", "__source_dir"]

    if "child_id" in df.columns:
        df["child_id"] = df["child_id"].map(lambda x: "" if pd.isna(x) else str(x))
        keep.append("child_id")

    out = df[keep].copy()
    out["pros_vec"] = out["pros_vec"].apply(lambda x: list(np.asarray(x, dtype=np.float32)))

    return Dataset.from_pandas(out, preserve_index=False)


class ProsodyProjector(nn.Module):
    def __init__(self, hidden_size: int, pros_dim: int, pros_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pros_dim, pros_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(pros_hidden, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class ProsodyCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, shuffle_prosody=False):
        self.base = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
        self.shuffle = shuffle_prosody

    def __call__(self, examples):
        base_examples = [
            {k: e[k] for k in ("input_ids", "attention_mask")}
            for e in examples
        ]
        batch = self.base(base_examples)

        if any("label_acc" in e for e in examples):
            batch["label_acc"] = torch.tensor(
                [e.get("label_acc", 0) for e in examples],
                dtype=torch.long,
            )

        pros_lists = []
        any_pros = False

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
            outputs = model(
                inputs_embeds=word_emb,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def make_tok_map(use_prosody, tokenizer, pros_token_id, max_len):
    def tok_map(batch):
        texts = ["" if x is None else str(x) for x in batch["text"]]
        enc = tokenizer(texts, truncation=True, max_length=max_len)

        if use_prosody:
            new_ids = []
            new_att = []
            for ids, att in zip(enc["input_ids"], enc["attention_mask"]):
                new_ids.append([pros_token_id] + ids)
                new_att.append([1] + att)
            enc["input_ids"] = new_ids
            enc["attention_mask"] = new_att

        enc["label_acc"] = list(batch.get("label_acc", [0] * len(enc["input_ids"])))

        pros_dim = len(PROSODY_COLS) + (1 if ADD_HAS_PROSODY_BIT else 0)
        if "pros_vec" in batch:
            enc["pros_vec"] = [
                list(np.asarray(v, dtype=np.float32))
                for v in batch["pros_vec"]
            ]
        else:
            zero = [0.0] * pros_dim
            enc["pros_vec"] = [zero for _ in enc["input_ids"]]

        return enc

    return tok_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", required=True, help="Repeat for each input folder.")
    parser.add_argument("--pattern", action="append", default=None, help="Repeat for each glob pattern.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--pros-hidden", type=int, default=128)
    parser.add_argument("--out-dir", default="out_mlm_pros")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--max-steps", type=int, default=-1, help="Use -1 for no cap.")
    parser.add_argument("--no-prosody", action="store_true")
    parser.add_argument("--shuffle-prosody", action="store_true")
    args = parser.parse_args()

    patterns = args.pattern if args.pattern is not None else [
        "*_training_echo_v2.csv",
        "*_with_speaker_training_echo_v2.csv",
        "*_reconstructed_from_audio.csv",
    ]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    csvs = find_csvs(args.data_dir, patterns)
    if not csvs:
        raise FileNotFoundError("No CSV files found with the provided folders/patterns.")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    use_prosody = not args.no_prosody

    if use_prosody:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [PROS_TOKEN]}
        )
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        pros_token_id = tokenizer.convert_tokens_to_ids(PROS_TOKEN)
    else:
        pros_token_id = None

    tok_map = make_tok_map(
        use_prosody=use_prosody,
        tokenizer=tokenizer,
        pros_token_id=pros_token_id,
        max_len=args.max_len,
    )

    ds = ds.map(tok_map, batched=True, remove_columns=ds.column_names)

    if use_prosody:
        pros_dim = len(PROSODY_COLS) + (1 if ADD_HAS_PROSODY_BIT else 0)
        model.pros_projector = ProsodyProjector(
            hidden_size=model.config.hidden_size,
            pros_dim=pros_dim,
            pros_hidden=args.pros_hidden,
        )

    collator = ProsodyCollator(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROB,
        shuffle_prosody=args.shuffle_prosody,
    )

    run_tag = "textonly" if not use_prosody else ("shufflePROS" if args.shuffle_prosody else "prosody")
    out_dir = f"{args.out_dir}_{run_tag}"

    training_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        report_to=["none"],
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    if args.max_steps is not None and args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = ProsodyTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        use_prosody=use_prosody,
    )

    trainer.train()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    export_cols = ["text", "label_acc", "__source_file", "__source_dir"] + PROSODY_COLS
    if "has_prosody" in df_all.columns:
        export_cols.append("has_prosody")

    for c in export_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    df_all.to_csv(
        artifacts_dir / f"train_rows_for_glmm_{run_tag}.csv",
        index=False,
    )

    with open(artifacts_dir / f"training_config_{run_tag}.json", "w") as f:
        json.dump(
            {
                "data_dirs": args.data_dir,
                "patterns": patterns,
                "loaded_files": [str(p) for p in csvs],
                "model_name": args.model_name,
                "prosody_cols": PROSODY_COLS,
                "add_has_prosody_bit": ADD_HAS_PROSODY_BIT,
                "max_len": args.max_len,
                "mlm_prob": MLM_PROB,
                "batch_size": args.batch,
                "lr": args.lr,
                "epochs": args.epochs,
                "use_prosody": use_prosody,
                "shuffle_prosody": args.shuffle_prosody,
                "out_dir": out_dir,
                "artifacts_dir": str(artifacts_dir),
                "max_steps": args.max_steps,
            },
            f,
            indent=2,
        )

    print("\n[OK] Training complete.")
    print(f"- Saved model:  {out_dir}")
    print(f"- GLMM input:   {artifacts_dir / f'train_rows_for_glmm_{run_tag}.csv'}")
    print(f"- Config:       {artifacts_dir / f'training_config_{run_tag}.json'}")


if __name__ == "__main__":
    main()
    print("\nTraining complete.")
    print(f"- Checkpoints: {out_dir}")
    print(f"- GLMM input:  {ARTIFACTS_DIR}/train_rows_for_glmm_{run_tag}.csv")
    print(f"- Config:      {ARTIFACTS_DIR}/training_config_{run_tag}.json")

if __name__ == "__main__":
    main()
