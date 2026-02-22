#!/bin/bash
#SBATCH --job-name=qlora_inf
#SBATCH --output=qlora_inf_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"

source .venv/bin/activate

python3 -u qlora_infer_100.py \
  --input_file "outputs/hitl_green_100.csv" \
  --adapter_dir "outputs/qlora_qwen25_3b" \
  --out_csv "outputs/qlora_predictions_100.csv"