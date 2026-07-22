#!/bin/bash
# SLURM single job — une seule expérience (fallback / test)
# Usage : CONFIG=configs/holdout/exp31_c1_hiver_haute.yaml sbatch cluster/jobs/train_single.slurm
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

# Forcer UTF-8 (voir train_array.slurm) : nœuds en locale C/POSIX → Python encode stdout en
# ASCII strict et plante sur les messages FR accentués / tirets cadratins de run_experiment.py.
export PYTHONUTF8=1
export LC_ALL=C.UTF-8 LANG=C.UTF-8

CONFIG=${CONFIG:-configs/holdout/exp31_c1_hiver_haute.yaml}
echo "[train_single] config=${CONFIG}"

module purge
module load orsu
module load gnu12
module load anaconda3/2024.06
conda activate microgrid-rl

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

python experiments/run_experiment.py --config "${CONFIG}"
