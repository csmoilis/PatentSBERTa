from unsloth import FastLanguageModel, get_chat_template
from unsloth import get_chat_template
from peft import PeftModel
import os
import torch
import pandas as pd
import re
import gc


def main():
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="patent_qlora_adapter",   # ← LOAD YOUR FINETUNED MODEL DIRECTLY
    max_seq_length=2048,
    load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    print(type(model))
    print("print of the model:")
    print(model)
    
    # TEST
    test_patent = """
    A photovoltaic solar panel system with battery storage 
    that improves renewable energy efficiency and reduces carbon emissions.
    """
    messages = [
    {"role": "system", "content": "You are an expert in Y02 green technology classification."},
    {"role": "user", "content": f"Analyze the following patent and determine if it qualifies as green technology. Provide a binary decision and a short argument.\n\n{test_patent}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,   # 🔥 VERY IMPORTANT
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=120,
            temperature=0.1,
            do_sample=False,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nFULL OUTPUT:\n")
    print(decoded)
    
    assistant_answer = decoded.split("assistant")[-1].strip()

    print("\nASSISTANT ANSWER:\n")
    print(assistant_answer)

    
    # -------------------------
    # 0. Load Data
    # -------------------------
    file_path = "outputs/hitl_wrong_pred_top100.csv"
    df = pd.read_csv(file_path)

    df = df.head(2).copy()

    # -----------------------------
    # 1. Load Base Model
    # -----------------------------
    

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )
    model.print_trainable_parameters()
    # -----------------------------
    # 2. Load Your Trained Adapter
    # -----------------------------
    model = PeftModel.from_pretrained(model, "patent_qlora_adapter")
    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # -----------------------------
    # Classification Function
    # -----------------------------
    def classify(text):
        messages = [
            {"role": "system", "content": "You are an expert in patent linguistics and Y02 green technology classification."},
            {"role": "user", "content": f"Analyze the following patent text and determine if it belongs to a Y02 green category:\n\n{text}"},
            {"role": "assistant", "content": ""},  # CRITICAL FIX
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new assistant text
        answer = full_text[len(prompt):].strip().lower()

        print("MODEL ANSWER:", answer)

        if "Green: 1" in answer:
            return 1
        else:
            return 0

    # -----------------------------
    # 6. Run Classification
    # -----------------------------
    predictions = []

    for i, row in df.iterrows():
        print(f"Processing {i+1}/{len(df)}")
        pred = classify(row["text"])
        predictions.append(pred)

    df["pred_Qlora"] = predictions

    # -----------------------------
    # 7. Save Results
    # -----------------------------
    df.to_csv("outputs/qlora_temp_predictions.csv", index=False)

    print("Done. Saved to outputs/qlora_temp_predictions.csv")


if __name__ == "__main__":
    main()
