import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import time
import argparse

def main():
    start_time = time.time()
    train_path = os.path.join("outputs", "train_silver.parquet")
    
    #reading data

    df_eval = pd.read_parquet("./outputs/eval_silver.parquet")
    df_train = pd.read_parquet(train_path)
    df_train = df_train[["text", "is_green_silver"]]

    new_train_dataset = df_train.copy()
    # 1. Configuration
    model_name = "unsloth/llama-3-8b-bnb-4bit" # Optimized 4-bit Llama-3
    #model_name = "unsloth/mistral-7b-v0.3-bnb-4bit" # Alternative
    max_seq_length = 2048 
    dtype = None # Auto-detect
    load_in_4bit = True 

    # 2. Load Model & Tokenizer with Unsloth Speedups
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    from unsloth import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # or "mistral" if using Mistral
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # Standard Unsloth mapping
    )

    # 3. Add QLoRA Adapters (Targeting ALL modules for better domain adaptation)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank (from Notebook 2 & 3)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0, # Optimized for Unsloth
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    def format_patent_data(examples):
        instructions = []

        for text, label in zip(examples["text"], examples["is_green_silver"]):

            if label == 1:
                answer = (
                    "Green: 1\n"
                    "Argument: This patent relates to environmentally sustainable technology "
                    "such as renewable energy, emission reduction, or energy efficiency improvements."
                )
            else:
                answer = (
                    "Green: 0\n"
                    "Argument: This patent does not address environmental sustainability, "
                    "renewable energy, or climate mitigation technologies."
                )

            msg = [
                {"role": "system", "content": "You are an expert in green technology classification."},
                {"role": "user", "content": f"Analyze the following patent and determine if it qualifies as green technology. Provide a binary decision and a short argument.\n\n{text}"},
                {"role": "assistant", "content": answer},
            ]

            instructions.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        return {"formatted_text": instructions}

    dataset = Dataset.from_pandas(new_train_dataset)
    dataset = dataset.map(format_patent_data, batched=True)

    # 5. Training Setup (SFT - Supervised Fine-Tuning)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="formatted_text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        train_on_responses_only=True,  # it will provide a response
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=300,                # more steps the model will see more data
            learning_rate=1e-4,           # for more stability
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_patent_qlora",
        ),
    )

    trainer_stats = trainer.train()

    model.save_pretrained("patent_qlora_adapter")
    tokenizer.save_pretrained("patent_qlora_adapter")
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
if __name__ == "__main__":
    main()