#!/bin/bash
#SBATCH --job-name=patent-processing
#SBATCH --output=process_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # Increased memory for large pandas dataframes
#SBATCH --gres=gpu:1               # Request 1 GPU for embeddings
#SBATCH --time=02:00:00            # Estimated 2 hours for 50k embeddings

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source .venv/bin/activate 
echo "Virtual environment activated."

# Run script
echo "Starting PatentSBERTa processing..."
#python A02_AB_baseline.py
#python A02_C_LLM_judge.py
#python temp.py
#python A02_D_Patent_SBERTa.py 
python temp.py  \
    --golden_path "./outputs/HIDL_Qlora_100_gold.xlsx" \
    --test "no" \