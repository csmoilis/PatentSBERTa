import numpy as np
import pandas as pd
import os
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Configuration ---
INPUT_PATH = "./outputs/"
POOL_FILE = os.path.join(INPUT_PATH, "hitl_green_100.csv")
OUTPUT_FILE = os.path.join(INPUT_PATH, "hitl_vllm_results.csv")

# MODEL: Mistral-7B-v0.3 (Fits perfectly in 22GB VRAM)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" 

SYSTEM_PROMPT = """You are a specialist in Patent Law and Green Technology. 
Your task is to classify patent claims.

CRITERIA for Green Technology (llm_green_suggested = 1):
- Renewable energy (solar, wind, hydro, geothermal).
- Energy efficiency (low-power electronics, thermal insulation).
- Emissions reduction (carbon capture, catalytic converters).
- Circular economy (recycling, waste-to-energy).

CRITERIA for Non-Green (llm_green_suggested = 0):
- General mechanical components (gears, bolts) without specific efficiency gains.
- Purely medical or biological inventions unrelated to the environment.
- General software/network protocols.

RULES:
- Output ONLY valid JSON.
- Fields: llm_green_suggested (0 or 1), llm_confidence ("low", "medium", "high"), llm_rationale (1-3 sentences).
"""

def clean_and_parse_json(text):
    """Handles potential model chatter and extracts JSON."""
    try:
        # Mistral-7B-v0.3 usually doesn't use <think> tags, but we keep this for safety
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # Regex to find the JSON object {}
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON block found")
    except Exception as e:
        return {
            "llm_green_suggested": None, 
            "llm_confidence": "error", 
            "llm_rationale": f"Parsing Error: {str(e)} | Raw: {text[:50]}"
        }

def main():
    # 1. Load Data
    print(f"Loading data from {POOL_FILE}...")
    hitl_df = pd.read_csv(POOL_FILE)
    
    # 2. Initialize vLLM
    print(f"Initializing vLLM with {MODEL_NAME}...")
    # 0.8 utilization provides a great balance of speed and stability
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.8, max_model_len=4096)
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Keeping it deterministic
        max_tokens=512
    )

    # 3. Format Prompts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted_prompts = []
    
    for _, row in hitl_df.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim to evaluate:\n{row['text']}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    # 4. Execute Batch Inference
    print(f"Running inference for {len(formatted_prompts)} samples...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. Parse and Combine
    llm_results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        res = clean_and_parse_json(generated_text)
        llm_results.append(res)

    # 6. Final Dataframe Preparation
    res_df = pd.DataFrame(llm_results)
    final_hitl = pd.concat([hitl_df.reset_index(drop=True), res_df], axis=1)
    
    # Adding empty columns for your manual Human step
    final_hitl['is_green_human'] = np.nan
    final_hitl['notes'] = ""

    # 7. Save
    final_hitl.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS!")
    print(f"File saved to: {OUTPUT_FILE}")
    print("You can now open this CSV and fill in the 'is_green_human' column.")

if __name__ == "__main__":
    main()