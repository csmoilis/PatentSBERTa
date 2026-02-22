#!/bin/bash
#SBATCH --job-name=qlora
#SBATCH --output=qlora_%j.out
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"

source .venv/bin/activate
python3 -u qlora.py \
  --train_parquet "train_silver_70_no_gold.parquet" \
  --gold_csv "hitl_green_100_gold 1.csv" \
  --gold_sep ";"