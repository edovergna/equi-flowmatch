# Equivariant flow-matching evaluation

These tables report the published equivariant flow-matching checkpoints using
the same evaluation protocol. They are intended as the reference results for
comparison with EquiDrift.

## Protocol

- 50,000 generated samples and 50,000 held-out test samples
- evaluation seed 0
- batch size 1,000 on one NVIDIA A100 40 GB GPU
- adaptive Dopri5 integration with `atol=1e-5` and `rtol=1e-5`
- a sample is valid when its energy is finite and no greater than the 99.9th
  percentile of the corresponding holdout energy distribution
- energy and pair-distance metrics use valid generated and valid holdout samples
- invariant sliced Wasserstein uses sorted pair distances, holdout-only feature
  standardization, 64 fixed random projections, and projection seed 2
- time per valid sample is total sampling wall time divided by the number of
  valid generated samples
- NFE is the number of vector-field evaluations per adaptive batched trajectory
  solve; the table gives mean ± population standard deviation and the range

Lower is better for all metrics except valid samples.

## DW4 — equivariant OT flow matching

Checkpoint: `models/DW4_eq_OT_flow_matching`  
SHA-256: `f1c5ebe21e4d465b4015fd07a8d6032656dde805e8f9ab49fb61f764a7f12369`

| Metric | Result |
|---|---:|
| Valid samples | **98.908%** |
| Energy Wasserstein-1 | **0.134652** |
| Absolute mean-energy error | **0.133324** |
| Pair-distance Wasserstein-1 | **0.005544** |
| Invariant sliced Wasserstein | **0.014987** |
| Time per valid sample | **2.870 ms** |
| Sampling NFE | **75.20 ± 2.68** |
| Sampling NFE range | **68–80** |

Detailed outputs: `runs/dw4/evaluation_published/`

## LJ13 — equivariant OT flow matching

Checkpoint: `models/LJ13_eq_OT_flow_matching`  
SHA-256: `27ec776d8039f13d50a4a7ffc23c8d05e7d571d22958856b9f7117bac39f32e5`

| Metric | Result |
|---|---:|
| Valid samples | **99.790%** |
| Energy Wasserstein-1 | **1.589985** |
| Absolute mean-energy error | **1.589973** |
| Pair-distance Wasserstein-1 | **0.002906** |
| Invariant sliced Wasserstein | **0.072370** |
| Time per valid sample | **23.902 ms** |
| Sampling NFE | **138.68 ± 7.80** |
| Sampling NFE range | **122–158** |

Detailed outputs: `runs/lj13/evaluation_published/`

## Interpretation notes

Metrics should only be compared between models evaluated on the same physical
system; their scales differ between DW4 and LJ13. Timing comparisons require the
same hardware, batch size, solver tolerances, and density-tracking behavior.
Valid-only metrics must always be reported together with the valid-sample rate.
