import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


def find_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def detect_text_col(df):
    for c in ["text", "claim", "sentence", "content"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a text column in {list(df.columns)}")


def detect_label_col(df):
    for c in ["label", "y", "target", "llm_green_suggested", "green", "is_green", "is_green_silver"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a label column in {list(df.columns)}")


def make_sft_text(claim: str, label: int) -> str:
    return (
        "Task: Classify the patent claim.\n"
        "Return only one digit:\n"
        "0 = non-green\n"
        "1 = green\n\n"
        f"Claim:\n{claim}\n\n"
        "Answer:\n"
        f"{int(label)}"
    )


def read_gold_csv(path: str, sep: str | None = None) -> pd.DataFrame:
    # Your HITL file is very likely semicolon-separated; we try ; first then ,.
    if sep is not None:
        return pd.read_csv(path, sep=sep, engine="python")

    try:
        return pd.read_csv(path, sep=";", engine="python")
    except Exception:
        return pd.read_csv(path, sep=",", engine="python")


def prepare_jsonl(train_parquet, gold_csv, out_dir, oversample, gold_sep=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_parquet)
    gold_df = read_gold_csv(gold_csv, sep=gold_sep)

    train_text = detect_text_col(train_df)
    train_label = detect_label_col(train_df)
    gold_text = detect_text_col(gold_df)
    gold_label = detect_label_col(gold_df)

    train_df = train_df.rename(columns={train_text: "claim", train_label: "label"})
    gold_df = gold_df.rename(columns={gold_text: "claim", gold_label: "label"})

    train_df = train_df[["claim", "label"]].dropna()
    gold_df = gold_df[["claim", "label"]].dropna()

    train_df["label"] = train_df["label"].astype(int)
    gold_df["label"] = gold_df["label"].astype(int)

    if oversample > 1:
        gold_os = pd.concat([gold_df] * oversample, ignore_index=True)
    else:
        gold_os = gold_df

    combined = pd.concat([train_df, gold_os], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    combined_out = pd.DataFrame(
        {"text": [make_sft_text(c, y) for c, y in zip(combined["claim"], combined["label"])]}
    )
    out_path = out_dir / "train.jsonl"
    combined_out.to_json(out_path, orient="records", lines=True, force_ascii=False)

    sanity = {
        "train_parquet": train_parquet,
        "gold_csv": gold_csv,
        "gold_sep_used": gold_sep if gold_sep is not None else "auto(; then ,)",
        "train_rows_original": int(len(train_df)),
        "gold_rows_original": int(len(gold_df)),
        "gold_oversample": int(oversample),
        "combined_rows": int(len(combined)),
        "label_counts_combined": combined["label"].value_counts().to_dict(),
        "train_text_col": train_text,
        "train_label_col": train_label,
        "gold_text_col": gold_text,
        "gold_label_col": gold_label,
    }
    with open(out_dir / "sanity.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    return str(out_path), str(out_dir / "sanity.json")


def train_qlora(train_jsonl, out_dir, model_name, epochs, lr, batch_size, grad_accum, max_seq_len):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files=train_jsonl, split="train")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=20,
        save_steps=200,
        save_strategy="steps",
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=lora_config,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", default="train_silver_70_no_gold.parquet")
    ap.add_argument("--gold_csv", default=None)
    ap.add_argument("--gold_sep", default=None, help="Set explicitly if needed: ';' or ','")
    ap.add_argument("--out_data_dir", default="data_qlora")
    ap.add_argument("--out_model_dir", default="outputs/qlora_qwen25_3b")
    ap.add_argument("--oversample", type=int, default=10)

    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=512)
    args = ap.parse_args()

    gold_csv = args.gold_csv or find_existing(
        [
            "hitl_green_100_gold 1.csv",
            "hitl_green_100.csv",
            "outputs/hitl_green_100.csv",
            "outputs/hitl_green_100_gold 1.csv",
        ]
    )
    if gold_csv is None:
        raise FileNotFoundError("Could not find gold CSV. Use --gold_csv path/to/file.csv")

    print("Preparing training jsonl...", flush=True)
    train_jsonl, sanity_path = prepare_jsonl(
        train_parquet=args.train_parquet,
        gold_csv=gold_csv,
        out_dir=args.out_data_dir,
        oversample=args.oversample,
        gold_sep=args.gold_sep,
    )
    print(f"Train JSONL: {train_jsonl}", flush=True)
    print(f"Sanity file: {sanity_path}", flush=True)

    print("Starting QLoRA fine-tuning...", flush=True)
    Path(args.out_model_dir).mkdir(parents=True, exist_ok=True)
    train_qlora(
        train_jsonl=train_jsonl,
        out_dir=args.out_model_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
    )

    print(f"Done. Saved adapters to: {args.out_model_dir}", flush=True)


if __name__ == "__main__":
    main()