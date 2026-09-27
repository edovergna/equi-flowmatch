# Equivariant flow matching - supplementary material

## Command-line experiments

The original notebooks have been restored to their CUDA-only behavior. For
clusters without notebook support, CUDA training entry points and full linear
Python exports are organized by dataset under `experiments/`:

- `experiments/dw4/`
- `experiments/lj13/`
- `experiments/lj55/`
- `experiments/alanine_dipeptide/`

See [experiments/README.md](experiments/README.md) for commands, checkpointing,
environment setup, and a Snellius SLURM example.

The downloaded datasets and checkpoints are in `data/` and `models/`. The two
published LJ55 pieces have also been combined into the filename expected by the
notebook and command-line program: `data/all_data_LJ55-1000.npy`.

## Datasets and model checkpoints
Datasets as well as model checkpoints are available [here](https://osf.io/srqg7/?view_only=28deeba0845546fb96d1b2f355db0da5).
To run the code the datasets must be pasted in the data folder.

## Code
There are several jupyter notebooks to reproduce the main experiments of the paper.
Equivalent Python programs for GPU batch systems are provided in `experiments/`.
Moreover, the code will be integrated in the [bgflow](https://github.com/noegroup/bgflow) library.

Requirements to run the code can be found in requirements.txt. 
Note that [bgflow](https://github.com/noegroup/bgflow) and [bgmol](https://github.com/noegroup/bgmol/tree/main) are also required.
Moreover, the [ANODE](https://github.com/amirgholami/anode) package must be installed manually,
but is only required to perform integration with the fix step integrator.  
