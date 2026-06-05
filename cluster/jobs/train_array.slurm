#!/bin/bash
# SLURM job array — Axe 2 sweep parallèle
# Soumis via : cluster/sweep/launch_sweep.sh <sweep_name>
# Ne pas soumettre directement (MANIFEST doit être défini).
#
#SBATCH -J sweep
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
# --array=0-N%K est injecté par launch_sweep.sh au moment du sbatch (K = throttle).
# Dimensionnement : SAC mono-env/MLP ne scale pas au-delà de ~2-4 threads ; 4 c + 8 Go suffisent
# (buffer ~1.8 Go). time=8h couvre 1M steps (~3.4h) avec marge ; un nœud Intel = 52 c / 192 Go.

set -euo pipefail

if [[ -z "${MANIFEST:-}" ]]; then
    echo "[train_array] MANIFEST non défini. Utilisez launch_sweep.sh." >&2
    exit 1
fi

# Résoudre le config correspondant à cet index d'array
CONFIG=$(python3 -c "
import json, sys
rows = json.load(open('${MANIFEST}'))
idx = int('${SLURM_ARRAY_TASK_ID}')
if idx >= len(rows):
    sys.exit(f'index {idx} hors manifest ({len(rows)} entrées)')
print(rows[idx]['config'])
")

echo "[train_array] task=${SLURM_ARRAY_TASK_ID}  config=${CONFIG}"

# Environnement
module purge
module load orsu
module load gnu12
module load anaconda3/2024.06
conda activate microgrid-rl

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

# Aligner les threads BLAS/OpenMP sur les cœurs alloués : un SAC/MLP mono-env ne tire
# rien d'un sur-threading et l'oversubscription dégrade le débit quand plusieurs tâches
# d'array partagent un nœud. (À comparer 1 vs 4 via un micro-bench si besoin.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python experiments/run_experiment.py --config "${CONFIG}"
