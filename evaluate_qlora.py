import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def main():
    pred_path = "outputs/qlora_predictions_100.csv"
    gold_path = "hitl_green_100_gold 1.csv"

    pred_label_col = "qlora_label"
    gold_label_col = "llm_green_suggested"  # your gold label col (from your printed columns)

    print("Loading prediction file...", flush=True)
    pred_df = pd.read_csv(pred_path)

    print("Loading gold file (semicolon-separated)...", flush=True)
    gold_df = pd.read_csv(gold_path, sep=";", engine="python")

    print("Pred rows:", len(pred_df), flush=True)
    print("Gold rows:", len(gold_df), flush=True)
    print("Pred columns:", pred_df.columns.tolist(), flush=True)
    print("Gold columns:", gold_df.columns.tolist(), flush=True)

    if pred_label_col not in pred_df.columns:
        raise ValueError(f"Missing '{pred_label_col}' in predictions file.")
    if gold_label_col not in gold_df.columns:
        raise ValueError(f"Missing '{gold_label_col}' in gold file.")

    # --- Alignment ---
    used_alignment = None
    if "doc_id" in pred_df.columns and "doc_id" in gold_df.columns:
        pred_df["doc_id"] = pred_df["doc_id"].astype(str).str.strip()
        gold_df["doc_id"] = gold_df["doc_id"].astype(str).str.strip()
        merged = pred_df.merge(gold_df[["doc_id", gold_label_col]], on="doc_id", how="inner")

        print("Matched rows after doc_id merge:", len(merged), flush=True)
        if len(merged) > 0:
            df = merged
            used_alignment = "doc_id merge"
            gold_used_col = gold_label_col
        else:
            print("No doc_id matches. Using row-order alignment.", flush=True)
            df = pred_df.copy()
            df["gold_label"] = gold_df[gold_label_col].values[: len(df)]
            used_alignment = "row-order"
            gold_used_col = "gold_label"
    else:
        print("doc_id not present in both files. Using row-order alignment.", flush=True)
        df = pred_df.copy()
        df["gold_label"] = gold_df[gold_label_col].values[: len(df)]
        used_alignment = "row-order"
        gold_used_col = "gold_label"

    # --- Clean + coerce labels safely ---
    df[pred_label_col] = pd.to_numeric(df[pred_label_col], errors="coerce")
    df[gold_used_col] = pd.to_numeric(df[gold_used_col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[pred_label_col, gold_used_col]).copy()
    after = len(df)
    dropped = before - after

    print("\nAlignment used:", used_alignment, flush=True)
    print("Samples before cleaning:", before, flush=True)
    print("Dropped (missing/invalid labels):", dropped, flush=True)
    print("Samples used for metrics:", after, flush=True)

    if after == 0:
        raise ValueError(
            "No usable rows after cleaning. This means qlora_label is missing for all rows. "
            "Check outputs/qlora_predictions_100.csv raw_output / parsing."
        )

    y_true = df[gold_used_col].astype(int)
    y_pred = df[pred_label_col].astype(int)

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n==== EVALUATION RESULTS ====", flush=True)
    print("Accuracy :", accuracy, flush=True)
    print("Precision:", precision, flush=True)
    print("Recall   :", recall, flush=True)
    print("F1 Score :", f1, flush=True)

    print("\nDetailed Report:\n", flush=True)
    print(classification_report(y_true, y_pred, zero_division=0), flush=True)


if __name__ == "__main__":
    main()