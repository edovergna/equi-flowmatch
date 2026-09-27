#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
environment_dir="${EQUI_ENV_DIR:-$PWD/.snellius-env}"

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
printf '%s\n' "$PWD/.deps/anode" > "$site_packages/anode.pth"

"$environment_dir/bin/python" -m pip check
"$environment_dir/bin/python" -c \
  'import torch, bgflow, bgmol, ot; print("torch", torch.__version__, "CUDA build", torch.version.cuda)'
echo "Environment created at $environment_dir"
