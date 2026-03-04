# PatentSBERTa Finetunning

All the scripts have a specific code. A means "assignment" and the number means to which assignment the code belongs.


## 100 Claims results

•	Mistral-7B-Instruct-v0.3: Agreed with the gold label in 89 out of 100 patents analyzed.  

•	Meta-Llama-3.1-8B-Instruct-bnb-4bit (Fine-tuned): Agreed with the gold label in 42 out of 100 patents analyzed.  

•	Agentic Judge System: Agreed with the gold label in 38 out of 100 patents analyzed.


## QLoRA Fine-Tuning: Results
   
The Qlora matched the human label 99% of the time, whereas the simple LLM from Assignment 2 matched 94%.  

Excel with 100 patents are in: https://huggingface.co/datasets/csmoilis/model_df_patentSBERTa/tree/main


## Classification Report for Green Patent Labeling

This document outlines the performance metrics for the Green Patent Labeling project, comparing the **GOLD LLM** and **GOLD Qlora** datasets.

### Baseline Performance Metrics

| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Green** | 0.77 | 0.79 | 0.78 | 4000 |
| **Green** | 0.78 | 0.77 | 0.78 | 4000 |
| **Accuracy** | | | **0.78** | **8000** |


---

### GOLD LLM Dataset: Model Performance Metrics

| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Green** | 0.8026 | 0.8113 | 0.8069 | 4000 |
| **Green** | 0.8092 | 0.8005 | 0.8048 | 4000 |
| **Accuracy** | | | **0.8059** | **8000** |


---

### GOLD Qlora Dataset: Model Performance Metrics


| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Green** | 0.8021 | 0.8155 | 0.8087 | 4000 |
| **Green** | 0.8124 | 0.7987 | 0.8055 | 4000 |
| **Accuracy** | | | **0.8071** | **8000** |


---

## Instructions

in your terminal: git clone {link}

link: go to github <>Code -> copy HTTPS

git pull to update the project

## How to use Git to save your work

type on the terminal:

git pull

With git pull you update the work from others. Then you save your progress typing on the terminal:  

git add .  
git commit -m "you put a message here"  
git push  

## Project setup 

curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

uv venv

source .venv/bin/activate

uv pip install .

## Executing:

sbatch submit_process.sh
squeue --me
