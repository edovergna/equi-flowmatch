#!/usr/bin/env bash
#SBATCH --job-name=equi-flowmatch
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
python_executable="${EQUI_PYTHON:-$SLURM_SUBMIT_DIR/.snellius-env/bin/python}"
if [[ ! -x "$python_executable" ]]; then
    echo "Missing Python executable: $python_executable" >&2
    echo "Run scripts/setup_snellius.sh or set EQUI_PYTHON before submitting." >&2
    exit 1
fi

if [[ "$#" -eq 0 ]]; then
    echo "Pass a training script and its arguments to sbatch." >&2
    exit 2
fi

nvidia-smi
srun "$python_executable" "$@"
