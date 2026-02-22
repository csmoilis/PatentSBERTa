import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from sklearn.metrics import classification_report
import torch
import evaluate
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden_path", default="HIDL_100_gold.xlsx")
    ap.add_argument("--test", default="yes")
    args = ap.parse_args()
    
    # 1. Configuration & Device Setup
    model_ckpt = "AI-Growth-Lab/PatentSBERTa"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    INPUT_PATH = "./outputs/"
    train_path = os.path.join(INPUT_PATH, "train_silver.parquet")
    
    # 2. Load Data (Update these paths to your AI-LAB directory)
    # Using placeholders based on your notebook structure
    golden_path = args.golden_path
    
    HITL_golden = pd.read_excel(golden_path) 

    df_eval = pd.read_parquet("./outputs/eval_silver.parquet")
    
    df_train = pd.read_parquet(train_path)
    if(args.test == "yes"):
        df_train = df_train.head(1000)
    
    
    df_train = df_train[["text", "is_green_silver"]]

    HITL_golden = HITL_golden[["text", "is_green_human"]]

    HITL_golden = HITL_golden.rename(columns={"is_green_human": "is_green_silver"})
    
    new_train_dataset = pd.concat([df_train, HITL_golden])
    
    # 3. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=256, truncation=True)
    
    # Map and REMOVE the original 'text' column immediately to avoid formatting issues
    train_ds = Dataset.from_pandas(new_train_dataset).map(tokenize_function, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_pandas(df_eval).map(tokenize_function, batched=True, remove_columns=["text"])
    gold_ds = Dataset.from_pandas(HITL_golden).map(tokenize_function, batched=True, remove_columns=["text"])

    # Rename labels to the standard expected by Hugging Face models
    train_ds = train_ds.rename_column("is_green_silver", "labels")
    eval_ds = eval_ds.rename_column("is_green_silver", "labels")
    gold_ds = gold_ds.rename_column("is_green_silver", "labels")

    # Explicitly set the format and only keep columns the model uses
    # This ensures the DataLoader yields a dictionary of Tensors
    model_inputs = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=model_inputs)
    eval_ds.set_format(type="torch", columns=model_inputs)
    gold_ds.set_format(type="torch", columns=model_inputs)

        

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(eval_ds, batch_size=4)
    gold_dataloader = DataLoader(gold_ds, batch_size=4)
    
    
    
    # 4. Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # 6. Training Loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # 7. Evaluation
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
       

    metric = evaluate.load("accuracy")
    model.eval()

    # 1. Initialize lists to store all predictions and true labels
    all_preds = []
    all_labels = []

    for batch in eval_dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # 2. Collect predictions and labels
        # We move them to CPU and convert to list to avoid GPU memory build-up
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
        
        # Keep your existing accuracy metric if you need it
        metric.add_batch(predictions=predictions, references=batch["labels"])

    # 3. Calculate and Save Performance Report
    # Use the lists we populated in the loop
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=['Not Green', 'Green'],
        digits=4 # Optional: added for more precision in your master's project
    )

    # Define your output path (ensure it exists)
    output_path = "./outputs/"
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "performance_report_sberta.txt"), "w") as f:
        f.write("Classification Report for Green Patent Labeling\n")
        f.write("==============================================\n")
        f.write(report)

    # --- 7b. Evaluation on Gold Human-in-the-loop Set ---
    print("Running evaluation on HIDL Gold dataset...")
    model.eval()
    gold_preds = []
    gold_labels = []

    for batch in gold_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        gold_preds.extend(predictions.cpu().numpy())
        gold_labels.extend(batch["labels"].cpu().numpy())

    # 3. Generate Report for Gold Set
    gold_report = classification_report(
        gold_labels, 
        gold_preds, 
        target_names=['Not Green', 'Green'],
        digits=4
    )

    # Save specifically for Gold
    with open(os.path.join(output_path, "performance_report_GOLD_HIDL.txt"), "w") as f:
        f.write("Classification Report: HUMAN-IN-THE-LOOP GOLD SET (100 samples)\n")
        f.write("==============================================================\n")
        f.write(gold_report)
    
    print("Gold HIDL performance report saved.")
    
    
    # Final Accuracy Check
    final_acc = metric.compute()
    print(f"Final Accuracy: {final_acc['accuracy']:.4f}")
    print("Performance report saved successfully.")
    # 8. Save Model
    model.save_pretrained("./fine_tuned_patent_model")
    tokenizer.save_pretrained("./fine_tuned_patent_model")

if __name__ == "__main__":
    main()