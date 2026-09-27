# Command-line experiments

These CUDA-only scripts replace the interactive training portions of the
notebooks. Run every command from the repository root.

| Dataset | Training command | Published defaults |
| --- | --- | --- |
| DW4 | `python experiments/dw4/train.py` | 50 epochs, batch 256 |
| LJ13 | `python experiments/lj13/train.py` | 1000 epochs, batch 256 |
| LJ55 | `python experiments/lj55/train.py` | 600 epochs, batch 256 |
| Alanine dipeptide | `python experiments/alanine_dipeptide/train_ot.py` | 1000 epochs, batch 256 |
| Weighted alanine | `python experiments/alanine_dipeptide/train_weighted_ot.py` | 1000 epochs, batch 256 |

Each script saves a resumable checkpoint under `runs/` by default. Common
options are available through `--help`, including `--epochs`, `--batch-size`,
`--learning-rate`, `--output`, `--resume`, and `--max-batches`.

For example, check DW4 with two batches and then run the full schedule:

```bash
python experiments/dw4/train.py --epochs 1 --max-batches 2
python experiments/dw4/train.py
```

Resume a stopped job by setting both the checkpoint to read and the total epoch
at which training should stop:

```bash
python experiments/dw4/train.py \
  --resume runs/dw4/checkpoint.pt \
  --epochs 50
```

The `notebook*.py` files are linear exports of every code cell in the original
notebooks. They include training, sampling, evaluation, and plotting. They
retain the published constants and commented checkpoint selection, with only
cluster-safe paths and compatibility fixes added. Use the focused `train*.py`
programs for batch jobs.

The weighted notebook constructs weighted samples but trains from the original
array. `train_weighted_ot.py` uses the weighted array, matching the experiment's
stated intent.

## Evaluation

Evaluate the published equivariant DW4 checkpoint against the disjoint MCMC
holdout split with 50,000 samples from each distribution:

```console
sbatch --time=02:00:00 --job-name=dw4-evaluation \
  experiments/snellius_gpu.sbatch experiments/dw4/evaluate.py
```

The evaluator writes to `runs/dw4/evaluation_published/` by default. Its output
includes energy and pair-distance PDFs/ECDFs, an invariant configuration-space
comparison, canonicalized configuration examples, per-distance-rank errors,
the generated and holdout arrays, and `metrics.json`. Metrics include raw and
importance-reweighted Wasserstein-1 and Jensen-Shannon distances, KS statistics,
energy and distance moments/quantiles, effective sample size, invariant sliced
Wasserstein distance, and held-out negative log likelihood. Use `--help` to
change the checkpoint, sample counts, integration tolerances, or output path.

Run the matching LJ13 evaluation with:

```console
sbatch --time=02:00:00 --job-name=lj13-evaluation \
  experiments/snellius_gpu.sbatch experiments/lj13/evaluate.py \
  --batch-size 1000
```

The finalized DW4 and LJ13 metric tables and the shared comparison protocol are
recorded in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

## Snellius

Create the pinned CUDA conda environment once through Slurm:

```bash
sbatch scripts/setup_snellius.sbatch
```

If `conda` is provided through a Snellius module in your account, load that
module before submitting so that it is available in the batch environment. The
setup job requests four CPU cores, 16 GiB of memory, and two hours; it writes to
Slurm's standard `slurm-<job-id>.out` log. Wait for the setup job to finish
successfully before submitting training. The setup also pins the bgflow, bgmol,
and ANODE revisions used by this repository. Direct shell execution is rejected
to keep the installation off the login node. Verify CUDA from a GPU allocation
with:

```bash
sbatch experiments/snellius_gpu.sbatch -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

Submit a one-hour A100 smoke test:

```bash
sbatch experiments/snellius_gpu.sbatch \
  experiments/dw4/train.py --epochs 1 --max-batches 2
```

For production, edit the `#SBATCH --time` line and remove `--max-batches`.
Snellius currently gives one A100 GPU allocation 18 CPU cores and 120 GiB host
RAM. The job file requests exactly that quarter-node allocation. Set
`EQUI_PYTHON=/path/to/python` before `sbatch` if the environment is elsewhere.

Transfer the repository's `data/` directory to Snellius before submitting.
LJ55 needs the combined `data/all_data_LJ55-1000.npy`; its two `part` files are
not needed once the combined file exists. Do not copy the macOS `.conda/`
directory to Snellius.
