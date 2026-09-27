#!/usr/bin/env python3
"""Evaluate the published equivariant LJ13 flow-matching checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from bgflow import BlackBoxDynamics, BoltzmannGenerator, DiffEqFlow
from bgflow import MeanFreeNormalDistribution
from bgflow.utils import remove_mean

from eq_ot_flow.estimator import BruteForceEstimatorFast
from eq_ot_flow.LJ import LennardJonesPotential
from eq_ot_flow.models import EGNN_dynamics
from experiments.common import ROOT, require_cuda, seed_everything


DIM = 39
N_PARTICLES = 13
N_DIMENSIONS = 3
N_PAIR_DISTANCES = N_PARTICLES * (N_PARTICLES - 1) // 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate the published equivariant LJ13 checkpoint."
    )
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/LJ13_eq_OT_flow_matching",
    )
    result.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/all_data_LJ13-1000.npy",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/lj13/evaluation_published",
    )
    result.add_argument("--samples", type=int, default=50_000)
    result.add_argument("--test-samples", type=int, default=50_000)
    result.add_argument("--batch-size", type=int, default=200)
    result.add_argument("--sliced-projections", type=int, default=64)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--atol", type=float, default=1e-5)
    result.add_argument("--rtol", type=float, default=1e-5)
    return result


def build_model(
    device: torch.device,
) -> tuple[BoltzmannGenerator, LennardJonesPotential]:
    target = LennardJonesPotential(
        DIM,
        N_PARTICLES,
        eps=1.0,
        rm=1.0,
        oscillator_scale=1.0,
        two_event_dims=False,
    )
    prior = MeanFreeNormalDistribution(
        DIM, N_PARTICLES, two_event_dims=False
    ).to(device)
    dynamics = EGNN_dynamics(
        n_particles=N_PARTICLES,
        device=str(device),
        n_dimension=N_DIMENSIONS,
        hidden_nf=32,
        act_fn=torch.nn.SiLU(),
        n_layers=3,
        recurrent=True,
        tanh=True,
        attention=True,
        condition_time=True,
        mode="egnn_dynamics",
        agg="sum",
    )
    flow = DiffEqFlow(
        dynamics=BlackBoxDynamics(
            dynamics_function=dynamics,
            divergence_estimator=BruteForceEstimatorFast(),
        )
    )
    return BoltzmannGenerator(prior, flow, target).to(device), target


def load_holdout(path: Path, count: int) -> torch.Tensor:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] != DIM:
        raise ValueError(f"Expected LJ13 data shaped (n, {DIM}), got {data.shape}")
    if count > len(data) - 500_000:
        raise ValueError("Requested more samples than the published holdout split.")

    # Reproduce the RandomState-based permutation in the published notebook.
    split_rng = np.random.RandomState(0)
    indices = split_rng.choice(np.arange(len(data)), len(data), replace=False)
    selected = indices[500_000 : 500_000 + count]
    holdout = torch.from_numpy(np.asarray(data[selected]).copy())
    return remove_mean(holdout.float(), N_PARTICLES, N_DIMENSIONS)


def pair_distances(samples: torch.Tensor) -> torch.Tensor:
    coordinates = samples.reshape(-1, N_PARTICLES, N_DIMENSIONS)
    row, column = torch.triu_indices(
        N_PARTICLES, N_PARTICLES, offset=1, device=coordinates.device
    )
    return torch.linalg.vector_norm(
        coordinates[:, row] - coordinates[:, column], dim=-1
    )


def sliced_wasserstein(
    reference: np.ndarray,
    generated: np.ndarray,
    projections: int,
    seed: int,
) -> float:
    # Holdout-only scaling makes scores directly comparable between models.
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale < 1e-12] = 1.0
    reference = (reference - mean) / scale
    generated = (generated - mean) / scale
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, reference.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    reference_projected = reference @ directions.T
    generated_projected = generated @ directions.T
    return float(
        np.mean(
            [
                wasserstein_distance(
                    reference_projected[:, index],
                    generated_projected[:, index],
                )
                for index in range(projections)
            ]
        )
    )


def plot_distribution(
    reference: np.ndarray,
    generated: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    combined = np.concatenate([reference, generated])
    low, high = np.quantile(combined, [0.001, 0.999])
    bins = np.linspace(low, high, 121)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(
        reference,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label="MCMC holdout",
    )
    axes[0].hist(
        generated,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label="Generated",
    )
    axes[0].set(xlabel=xlabel, ylabel="Density", title=title)
    axes[0].legend()
    for values, label in (
        (reference, "MCMC holdout"),
        (generated, "Generated"),
    ):
        ordered = np.sort(values)
        axes[1].plot(
            ordered,
            np.linspace(0, 1, len(ordered)),
            label=label,
        )
    axes[1].set(
        xlabel=xlabel,
        ylabel="Empirical CDF",
        title=f"{title} ECDF",
        xlim=(low, high),
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    if args.samples <= 0 or args.test_samples <= 0 or args.batch_size <= 0:
        raise ValueError("Sample counts and batch size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = require_cuda()

    model, target = build_model(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    model.flow._integrator_atol = args.atol
    model.flow._integrator_rtol = args.rtol
    model.flow._use_checkpoints = False
    model.flow._kwargs = {}

    holdout = load_holdout(args.data, args.test_samples)
    with torch.no_grad():
        holdout_energies = target.energy(holdout).numpy().reshape(-1)
    holdout_distances = pair_distances(holdout).numpy()

    nfe_counter = {"count": 0}

    def count_vector_field_call(_module, _inputs, _output) -> None:
        nfe_counter["count"] += 1

    vector_field = model.flow._dynamics._dynamics._dynamics_function
    nfe_handle = vector_field.register_forward_hook(count_vector_field_call)
    generated_batches = []
    energy_batches = []
    distance_batches = []
    sampling_nfes = []
    completed = 0
    started = time.perf_counter()
    while completed < args.samples:
        batch_size = min(args.batch_size, args.samples - completed)
        nfe_before = nfe_counter["count"]
        with torch.no_grad():
            generated = model.sample(batch_size)
            sampling_nfes.append(nfe_counter["count"] - nfe_before)
            energies = target.energy(generated)
            distances = pair_distances(generated)
        generated_batches.append(generated.cpu())
        energy_batches.append(energies.cpu())
        distance_batches.append(distances.cpu())
        completed += batch_size
        print(
            f"Generated {completed}/{args.samples}; "
            f"sampling NFE {sampling_nfes[-1]}",
            flush=True,
        )
    generation_seconds = time.perf_counter() - started
    nfe_handle.remove()

    generated = torch.cat(generated_batches).numpy()
    generated_energies = torch.cat(energy_batches).numpy().reshape(-1)
    generated_distances = torch.cat(distance_batches).numpy()

    validity_threshold = float(np.quantile(holdout_energies, 0.999))
    holdout_valid = np.isfinite(holdout_energies) & (
        holdout_energies <= validity_threshold
    )
    generated_valid = np.isfinite(generated_energies) & (
        generated_energies <= validity_threshold
    )
    if not generated_valid.any():
        raise RuntimeError("No generated samples pass the validity criterion.")

    reference_energy = holdout_energies[holdout_valid]
    model_energy = generated_energies[generated_valid]
    reference_distances = holdout_distances[holdout_valid]
    model_distances = generated_distances[generated_valid]
    reference_features = np.sort(reference_distances, axis=1)
    model_features = np.sort(model_distances, axis=1)

    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "generated_samples": args.samples,
        "holdout_samples": args.test_samples,
        "validity": {
            "definition": "finite energy <= holdout energy q99.9",
            "energy_threshold": validity_threshold,
            "generated_count": int(generated_valid.sum()),
            "generated_fraction": float(generated_valid.mean()),
            "holdout_count": int(holdout_valid.sum()),
            "holdout_fraction": float(holdout_valid.mean()),
        },
        "valid_only": {
            "energy_wasserstein_1": float(
                wasserstein_distance(reference_energy, model_energy)
            ),
            "mean_energy_generated": float(model_energy.mean()),
            "mean_energy_holdout": float(reference_energy.mean()),
            "mean_energy_absolute_error": float(
                abs(model_energy.mean() - reference_energy.mean())
            ),
            "pair_distance_wasserstein_1": float(
                wasserstein_distance(
                    reference_distances.reshape(-1),
                    model_distances.reshape(-1),
                )
            ),
            "invariant_sliced_wasserstein": sliced_wasserstein(
                reference_features,
                model_features,
                args.sliced_projections,
                args.seed + 2,
            ),
        },
        "performance": {
            "generation_seconds": generation_seconds,
            "time_per_attempted_sample_seconds": (
                generation_seconds / args.samples
            ),
            "time_per_valid_sample_seconds": (
                generation_seconds / int(generated_valid.sum())
            ),
            "sampling_nfe_mean": float(np.mean(sampling_nfes)),
            "sampling_nfe_std": float(np.std(sampling_nfes)),
            "sampling_nfe_min": int(np.min(sampling_nfes)),
            "sampling_nfe_max": int(np.max(sampling_nfes)),
            "sampling_nfe_per_batch": sampling_nfes,
            "nominal_batch_size": args.batch_size,
        },
        "integrator_atol": args.atol,
        "integrator_rtol": args.rtol,
    }

    plot_distribution(
        reference_energy,
        model_energy,
        "Potential energy",
        "Valid-only LJ13 energy distribution",
        args.output_dir / "energy_distribution_valid.png",
    )
    plot_distribution(
        reference_distances.reshape(-1),
        model_distances.reshape(-1),
        "Pair distance",
        "Valid-only LJ13 pair-distance distribution",
        args.output_dir / "distance_distribution_valid.png",
    )
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        generated_samples=generated,
        holdout_samples=holdout.numpy(),
        generated_energies=generated_energies,
        holdout_energies=holdout_energies,
        generated_pair_distances=generated_distances,
        holdout_pair_distances=holdout_distances,
        generated_valid=generated_valid,
        holdout_valid=holdout_valid,
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Evaluation written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
