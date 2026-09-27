#!/usr/bin/env bash
#SBATCH --job-name=equi-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This setup must run as a Slurm job." >&2
    echo "Submit it with: sbatch scripts/setup_snellius.sbatch" >&2
    exit 2
fi

repository_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$repository_dir"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is unavailable in the batch job environment." >&2
    echo "Load your Snellius conda module before calling sbatch, then resubmit." >&2
    exit 1
fi

environment_dir="${EQUI_ENV_DIR:-$repository_dir/.snellius-env}"

echo "Running setup in Slurm job $SLURM_JOB_ID on $(hostname)"
echo "Environment location: $environment_dir"

if [[ -x "$environment_dir/bin/python" ]]; then
    conda env update --prefix "$environment_dir" --file environment-snellius.yml --prune
else
    conda env create --prefix "$environment_dir" --file environment-snellius.yml
fi
mkdir -p .deps
if [[ ! -d .deps/anode ]]; then
    git clone https://github.com/amirgholami/anode.git .deps/anode
fi
git -C .deps/anode checkout 870444bcca907a1b99c2d8ecb7f1ab3da745566d

site_packages="$($environment_dir/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$repository_dir/.deps/anode" > "$site_packages/anode.pth"

"$environment_dir/bin/python" -m pip check
"$environment_dir/bin/python" -c \
  'import torch, bgflow, bgmol, ot; print("torch", torch.__version__, "CUDA build", torch.version.cuda)'
echo "Environment created at $environment_dir"
