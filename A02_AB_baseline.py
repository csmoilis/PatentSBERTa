import pandas as pd
import numpy as np
import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    # Setup output directory
    output_path = "./outputs/"
    os.makedirs(output_path, exist_ok=True)
    
    # --- Load Data and Model ---
    print("Loading dataset...")
    ds = load_dataset("AI-Growth-Lab/patents_claims_1.5m_traim_test")
    patents = ds['train'].to_pandas()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa', device=device)

    # --- Feature Engineering ---
    fixed_cols = ["id", "text", "date"]
    y02_cols = [col for col in patents.columns if col.startswith("Y02")]
    df_selected = patents[fixed_cols + y02_cols].copy()
    df_selected['is_green_silver'] = df_selected[y02_cols].any(axis=1).astype(int)

    # Balanced Sampling
    df_green = df_selected[df_selected["is_green_silver"] == 1]
    df_not_green = df_selected[df_selected["is_green_silver"] == 0]
    
    green_sample = df_green.sample(n=min(25000, len(df_green)), random_state=42)
    not_green_sample = df_not_green.sample(n=min(25000, len(df_not_green)), random_state=42)
    
    df_final = pd.concat([green_sample, not_green_sample], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Embedding Generation ---
    print("Generating embeddings...")
    sentences = df_final['text'].tolist()
    X_embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    df_final["embedding"] = list(X_embeddings)

    # --- Train/Eval/Pool Splits ---
    df_train_eval, df_pool_unlabeled = train_test_split(
        df_final, test_size=0.20, random_state=42, stratify=df_final["is_green_silver"]
    )
    df_train_silver, df_eval_silver = train_test_split(
        df_train_eval, test_size=0.20, random_state=42, stratify=df_train_eval["is_green_silver"]
    )

    # --- Logistic Regression Baseline ---
    X_train = np.vstack(df_train_silver["embedding"].values)
    y_train = df_train_silver["is_green_silver"]
    X_eval = np.vstack(df_eval_silver["embedding"].values)
    y_eval = df_eval_silver["is_green_silver"]

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)

    # Save Performance Report
    report = classification_report(y_eval, y_pred, target_names=['Not Green', 'Green'])
    with open(os.path.join(output_path, "performance_report.txt"), "w") as f:
        f.write(report)

    # --- Part B: HITL & Uncertainty ---
    print("Calculating uncertainty for Pool...")
    X_pool = np.vstack(df_pool_unlabeled["embedding"].values)
    probs_pool = clf.predict_proba(X_pool)
    
    # Uncertainty metric: 1 - 2 * |prob - 0.5|
    df_pool_unlabeled = df_pool_unlabeled.copy()
    df_pool_unlabeled['uncertainty'] = 1 - 2 * np.abs(probs_pool[:, 1] - 0.5)
    
    hitl_green_100 = df_pool_unlabeled.sort_values(by='uncertainty', ascending=False).head(100)
    hitl_green_100[['id', 'text', 'is_green_silver', 'uncertainty']].to_csv(os.path.join(output_path, 'hitl_green_100.csv'), index=False)

    # --- Save Final Parquet Files ---
    print("Saving parquet files...")
    df_train_silver.to_parquet(os.path.join(output_path, "train_silver.parquet"), index=False)
    df_eval_silver.to_parquet(os.path.join(output_path, "eval_silver.parquet"), index=False)
    df_pool_unlabeled.to_parquet(os.path.join(output_path, "pool_unlabeled.parquet"), index=False)
    
    print(f"Job finished. Files saved in {output_path}")

if __name__ == "__main__":
    main()
    print("Process finished successfully.")
    sys.exit(0)  # Explicitly exit