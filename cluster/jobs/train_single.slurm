#!/bin/bash
# SLURM single job — une seule expérience (fallback / test)
# Usage : CONFIG=configs/exp01_perfect_foresight.yaml sbatch cluster/jobs/train_single.slurm
# Ou via : ./cluster/submit.sh <exp_name>
#
#SBATCH -J train
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail

CONFIG=${CONFIG:-configs/exp01_perfect_foresight.yaml}
echo "[train_single] config=${CONFIG}"

module purge
module load orsu
module load gnu12
module load anaconda3/2024.06
conda activate microgrid-rl

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

python experiments/run_experiment.py --config "${CONFIG}"
