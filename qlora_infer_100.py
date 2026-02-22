import re
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def pick_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def read_claims(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p, sep=";", engine="python")
        except Exception:
            df = pd.read_csv(p, sep=",", engine="python")
        return df
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def detect_text_col(df: pd.DataFrame) -> str:
    for c in ["text", "claim", "sentence", "content"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a text column in {list(df.columns)}")


def detect_id_col(df: pd.DataFrame) -> str | None:
    for c in ["doc_id", "id", "patent_id"]:
        if c in df.columns:
            return c
    return None


def make_prompt(claim: str) -> str:
    # Keep it consistent with training format, but ask for a short rationale too.
    return (
        "Task: Classify the patent claim.\n"
        "Return ONLY in this exact format:\n"
        "Label: <0 or 1>\n"
        "Rationale: <one short sentence>\n\n"
        "0 = non-green\n"
        "1 = green\n\n"
        f"Claim:\n{claim}\n"
    )


def parse_output(txt: str):
    # Try to extract "Label: 0/1"
    m = re.search(r"Label\s*:\s*([01])", txt)
    label = int(m.group(1)) if m else None

    # Extract rationale line if present
    r = re.search(r"Rationale\s*:\s*(.+)", txt)
    rationale = r.group(1).strip() if r else ""

    # Fallback: first 0/1 anywhere
    if label is None:
        m2 = re.search(r"\b([01])\b", txt)
        label = int(m2.group(1)) if m2 else None

    return label, rationale


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default=None, help="CSV/Parquet containing 100 high-risk claims")
    ap.add_argument("--adapter_dir", default="outputs/qlora_qwen25_3b")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out_csv", default="outputs/qlora_predictions_100.csv")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    input_file = args.input_file or pick_existing([
        "outputs/hitl_green_100.csv",
        "hitl_green_100_gold 1.csv",
        "hitl_green_100.csv",
        "outputs/pool_unlabeled.parquet",
    ])
    if input_file is None:
        raise FileNotFoundError("Could not find input_file. Pass --input_file explicitly.")

    df = read_claims(input_file)
    text_col = detect_text_col(df)
    id_col = detect_id_col(df)

    # If input has more than 100 rows, take first 100 (or you can filter before)
    if len(df) > 100:
        df = df.head(100).copy()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    out_rows = []
    for i, row in df.iterrows():
        claim = str(row[text_col])
        prompt = make_prompt(claim)

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
        )

        decoded = tok.decode(gen[0], skip_special_tokens=True)
        # Keep only the generated tail after the prompt (best-effort)
        tail = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        label, rationale = parse_output(tail)

        out = {
            "row_idx": int(i),
            "doc_id": row[id_col] if id_col else None,
            "text": claim,
            "qlora_label": label,
            "qlora_rationale": rationale,
            "raw_output": tail,
        }
        out_rows.append(out)

        if (len(out_rows) % 10) == 0:
            print(f"Done {len(out_rows)}/{len(df)}", flush=True)

    out_df = pd.DataFrame(out_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()