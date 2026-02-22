# PatentSBERTa Finetunning

## QLoRA Fine-Tuning: Results
   
The Qlora matched the human label 99% of the time, whereas the simple LLM from Assignment 2 matched 94%.  

Excel with 100 patents are in: https://huggingface.co/datasets/csmoilis/model_df_patentSBERTa/tree/main


## Classification Report for Green Patent Labeling

This document outlines the performance metrics for the Green Patent Labeling project, comparing the **GOLD LLM** and **GOLD Qlora** datasets.

---

### GOLD LLM Dataset: Model Performance Metrics

| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Green** | 0.8026 | 0.8113 | 0.8069 | 4000 |
| **Green** | 0.8092 | 0.8005 | 0.8048 | 4000 |
| **Accuracy** | | | **0.8059** | **8000** |
| **Macro Avg** | 0.8059 | 0.8059 | 0.8059 | 8000 |
| **Weighted Avg** | 0.8059 | 0.8059 | 0.8059 | 8000 |

---

### GOLD Qlora Dataset: Model Performance Metrics


| Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Green** | 0.8021 | 0.8155 | 0.8087 | 4000 |
| **Green** | 0.8124 | 0.7987 | 0.8055 | 4000 |
| **Accuracy** | | | **0.8071** | **8000** |
| **Macro Avg** | 0.8072 | 0.8071 | 0.8071 | 8000 |
| **Weighted Avg** | 0.8072 | 0.8071 | 0.8071 | 8000 |

---

> **Note:** For patent data sourcing and further validation, refer to [Google Patents](https://patents.google.com).
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