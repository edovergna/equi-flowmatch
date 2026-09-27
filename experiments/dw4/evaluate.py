#!/usr/bin/env python3
"""Thorough, reproducible evaluation of a trained DW4 flow model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
import torch
from bgflow import (
    BlackBoxDynamics,
    BoltzmannGenerator,
    DiffEqFlow,
    MeanFreeNormalDistribution,
    MultiDoubleWellPotential,
)
from bgflow.utils import remove_mean

from eq_ot_flow.estimator import BruteForceEstimatorFast
from eq_ot_flow.models import EGNN_dynamics
from experiments.common import ROOT, require_cuda, seed_everything


DIM = 8
N_PARTICLES = 4
N_DIMENSIONS = 2
N_PAIR_DISTANCES = N_PARTICLES * (N_PARTICLES - 1) // 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate a published or locally trained DW4 checkpoint."
    )
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/DW4_eq_OT_flow_matching",
    )
    result.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/dw4-dataidx.npy",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/dw4/evaluation_published",
    )
    result.add_argument("--samples", type=int, default=50_000)
    result.add_argument("--test-samples", type=int, default=50_000)
    result.add_argument("--batch-size", type=int, default=1_000)
    result.add_argument("--nll-samples", type=int, default=5_000)
    result.add_argument("--nll-batch-size", type=int, default=500)
    result.add_argument("--sliced-projections", type=int, default=64)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--atol", type=float, default=1e-5)
    result.add_argument("--rtol", type=float, default=1e-5)
    return result


def build_model(device: torch.device) -> tuple[BoltzmannGenerator, MultiDoubleWellPotential]:
    target = MultiDoubleWellPotential(
        DIM,
        N_PARTICLES,
        0.9,
        -4,
        0,
        4,
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
    model = BoltzmannGenerator(prior, flow, target).to(device)
    return model, target


def load_holdout(path: Path, count: int, seed: int) -> torch.Tensor:
    packed = np.load(path, allow_pickle=True)
    coordinates = packed[0]
    if not isinstance(coordinates, torch.Tensor):
        coordinates = torch.as_tensor(coordinates)
    coordinates = remove_mean(
        coordinates.float(), N_PARTICLES, N_DIMENSIONS
    )
    indices = packed[1]
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    holdout_indices = indices[-500_000:]
    if count > len(holdout_indices):
        raise ValueError(
            f"Requested {count} test samples, but only {len(holdout_indices)} "
            "are in the published holdout split."
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(holdout_indices, size=count, replace=False)
    return coordinates[torch.as_tensor(selected, dtype=torch.long)]


def pair_distances(samples: torch.Tensor) -> torch.Tensor:
    coordinates = samples.reshape(-1, N_PARTICLES, N_DIMENSIONS)
    row, column = torch.triu_indices(
        N_PARTICLES, N_PARTICLES, offset=1, device=coordinates.device
    )
    return torch.linalg.vector_norm(
        coordinates[:, row] - coordinates[:, column], dim=-1
    )


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Generated log weights contain non-finite values.")
    shifted = values - values.max()
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Generated importance weights cannot be normalized.")
    return weights / total


def weighted_quantile(
    values: np.ndarray, quantiles: tuple[float, ...], weights: np.ndarray
) -> list[float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return [
        float(np.interp(quantile, cumulative, sorted_values))
        for quantile in quantiles
    ]


def describe(values: np.ndarray, weights: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = (0.01, 0.05, 0.5, 0.95, 0.99)
    if weights is None:
        mean = float(values.mean())
        variance = float(values.var())
        selected = [float(value) for value in np.quantile(values, quantiles)]
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights = weights / weights.sum()
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - mean) ** 2))
        selected = weighted_quantile(values, quantiles, weights)
    return {
        "count": int(len(values)),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": selected[0],
        "q05": selected[1],
        "median": selected[2],
        "q95": selected[3],
        "q99": selected[4],
    }


def jensen_shannon(
    reference: np.ndarray,
    generated: np.ndarray,
    generated_weights: np.ndarray | None = None,
    bins: int = 200,
) -> float:
    combined = np.concatenate([reference, generated])
    low, high = np.quantile(combined, [0.001, 0.999])
    edges = np.linspace(low, high, bins + 1)
    p, _ = np.histogram(reference, bins=edges)
    q, _ = np.histogram(generated, bins=edges, weights=generated_weights)
    p = p.astype(np.float64) + 1e-12
    q = q.astype(np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def sliced_wasserstein(
    reference: np.ndarray,
    generated: np.ndarray,
    projections: int,
    seed: int,
    generated_weights: np.ndarray | None = None,
) -> float:
    combined = np.concatenate([reference, generated], axis=0)
    mean = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    reference = (reference - mean) / scale
    generated = (generated - mean) / scale
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, reference.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = [
        wasserstein_distance(
            reference @ direction,
            generated @ direction,
            v_weights=generated_weights,
        )
        for direction in directions
    ]
    return float(np.mean(distances))


def distribution_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    generated_weights: np.ndarray,
) -> dict:
    generated_weights = np.asarray(generated_weights, dtype=np.float64)
    generated_weights = generated_weights / generated_weights.sum()
    return {
        "reference": describe(reference),
        "generated": describe(generated),
        "generated_reweighted": describe(generated, generated_weights),
        "wasserstein_1": float(wasserstein_distance(reference, generated)),
        "wasserstein_1_reweighted": float(
            wasserstein_distance(
                reference, generated, v_weights=generated_weights
            )
        ),
        "ks_statistic": float(
            ks_2samp(reference, generated, method="auto").statistic
        ),
        "jensen_shannon": jensen_shannon(reference, generated),
        "jensen_shannon_reweighted": jensen_shannon(
            reference, generated, generated_weights
        ),
        "mean_absolute_error": float(
            abs(generated.mean() - reference.mean())
        ),
        "mean_absolute_error_reweighted": float(
            abs(np.sum(generated_weights * generated) - reference.mean())
        ),
    }


def plot_scalar_distribution(
    reference: np.ndarray,
    generated: np.ndarray,
    generated_weights: np.ndarray,
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
    axes[0].hist(
        generated,
        bins=bins,
        weights=generated_weights,
        density=True,
        histtype="step",
        linewidth=2.2,
        label="Generated, reweighted",
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
            np.linspace(0, 1, len(ordered), endpoint=True),
            label=label,
        )
    order = np.argsort(generated)
    axes[1].plot(
        generated[order],
        np.cumsum(generated_weights[order]),
        label="Generated, reweighted",
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


def plot_invariant_distribution(
    reference_features: np.ndarray,
    generated_features: np.ndarray,
    path: Path,
    seed: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(
        np.concatenate([reference_features, generated_features], axis=0)
    )
    reference_2d = embedded[: len(reference_features)]
    generated_2d = embedded[len(reference_features) :]
    low = np.quantile(embedded, 0.002, axis=0)
    high = np.quantile(embedded, 0.998, axis=0)
    plot_range = [[low[0], high[0]], [low[1], high[1]]]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True)
    axes[0].hist2d(
        reference_2d[:, 0],
        reference_2d[:, 1],
        bins=90,
        range=plot_range,
        norm=LogNorm(),
        cmap="Blues",
    )
    axes[0].set_title("MCMC holdout")
    axes[1].hist2d(
        generated_2d[:, 0],
        generated_2d[:, 1],
        bins=90,
        range=plot_range,
        norm=LogNorm(),
        cmap="Oranges",
    )
    axes[1].set_title("Generated")

    rng = np.random.default_rng(seed)
    reference_selection = rng.choice(
        len(reference_2d), min(4_000, len(reference_2d)), replace=False
    )
    generated_selection = rng.choice(
        len(generated_2d), min(4_000, len(generated_2d)), replace=False
    )
    axes[2].scatter(
        reference_2d[reference_selection, 0],
        reference_2d[reference_selection, 1],
        s=5,
        alpha=0.2,
        label="MCMC holdout",
    )
    axes[2].scatter(
        generated_2d[generated_selection, 0],
        generated_2d[generated_selection, 1],
        s=5,
        alpha=0.2,
        label="Generated",
    )
    axes[2].set_title("Overlay")
    axes[2].legend(markerscale=3)
    for axis in axes:
        axis.set_xlabel("Invariant-distance PC1")
    axes[0].set_ylabel("Invariant-distance PC2")
    figure.suptitle(
        "DW4 configuration distribution via sorted pairwise distances\n"
        f"PC variance explained: {pca.explained_variance_ratio_.sum():.1%}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return pca, reference_2d, generated_2d


def canonicalize(samples: np.ndarray) -> np.ndarray:
    coordinates = samples.reshape(-1, N_PARTICLES, N_DIMENSIONS).copy()
    coordinates -= coordinates.mean(axis=1, keepdims=True)
    result = np.empty_like(coordinates)
    for index, configuration in enumerate(coordinates):
        covariance = configuration.T @ configuration
        _, eigenvectors = np.linalg.eigh(covariance)
        rotated = configuration @ eigenvectors[:, ::-1]
        farthest = np.argmax(np.linalg.norm(rotated, axis=1))
        if rotated[farthest, 0] < 0:
            rotated[:, 0] *= -1
        if rotated[np.argmax(np.abs(rotated[:, 1])), 1] < 0:
            rotated[:, 1] *= -1
        result[index] = rotated
    return result


def plot_gallery(
    reference: np.ndarray,
    generated: np.ndarray,
    path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    reference = canonicalize(
        reference[rng.choice(len(reference), 8, replace=False)]
    )
    generated = canonicalize(
        generated[rng.choice(len(generated), 8, replace=False)]
    )
    configurations = np.concatenate([reference, generated], axis=0)
    limit = max(2.5, float(np.quantile(np.abs(configurations), 0.995)) * 1.1)
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for index, (axis, configuration) in enumerate(
        zip(axes.flat, configurations)
    ):
        axis.scatter(
            configuration[:, 0],
            configuration[:, 1],
            s=90,
            c=np.arange(N_PARTICLES),
            cmap="viridis",
            edgecolor="black",
        )
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
        if index == 0:
            axis.set_title("MCMC holdout (top)")
        if index == 8:
            axis.set_title("Generated (bottom)")
    figure.suptitle("Canonicalized DW4 configuration examples")
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
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model.flow._integrator_atol = args.atol
    model.flow._integrator_rtol = args.rtol
    model.flow._use_checkpoints = False
    model.flow._kwargs = {}

    holdout = load_holdout(args.data, args.test_samples, args.seed + 1)
    with torch.no_grad():
        reference_energies = (
            target.energy(holdout).cpu().numpy().reshape(-1)
        )
    reference_distances = pair_distances(holdout).numpy()
    reference_features = np.sort(reference_distances, axis=1)

    generated_batches = []
    latent_batches = []
    energy_batches = []
    distance_batches = []
    log_weight_batches = []
    sampling_nfes = []
    nfe_counter = {"count": 0}

    def count_vector_field_call(_module, _inputs, _output) -> None:
        nfe_counter["count"] += 1

    vector_field = model.flow._dynamics._dynamics._dynamics_function
    nfe_handle = vector_field.register_forward_hook(count_vector_field_call)
    started = time.perf_counter()
    completed = 0
    while completed < args.samples:
        batch_size = min(args.batch_size, args.samples - completed)
        nfe_before = nfe_counter["count"]
        with torch.no_grad():
            generated, latent, dlogp = model.sample(
                batch_size, with_latent=True, with_dlogp=True
            )
            sampling_nfes.append(nfe_counter["count"] - nfe_before)
            log_weights = model.log_weights_given_latent(
                generated, latent, dlogp, normalize=False
            )
            energies = target.energy(generated)
            distances = pair_distances(generated)
        generated_batches.append(generated.cpu())
        latent_batches.append(latent.cpu())
        energy_batches.append(energies.cpu())
        distance_batches.append(distances.cpu())
        log_weight_batches.append(log_weights.cpu())
        completed += batch_size
        print(
            f"Generated {completed}/{args.samples}; "
            f"sampling NFE {sampling_nfes[-1]}",
            flush=True,
        )
    nfe_handle.remove()
    generation_seconds = time.perf_counter() - started

    generated = torch.cat(generated_batches).numpy()
    latent = torch.cat(latent_batches).numpy()
    generated_energies = torch.cat(energy_batches).numpy().reshape(-1)
    generated_distances = torch.cat(distance_batches).numpy()
    log_weights = torch.cat(log_weight_batches).numpy().reshape(-1)
    weights = normalized_weights(log_weights)
    generated_features = np.sort(generated_distances, axis=1)

    nll_count = min(args.nll_samples, len(holdout))
    nll_values = []
    if nll_count:
        for start in range(0, nll_count, args.nll_batch_size):
            batch = holdout[start : start + args.nll_batch_size].to(device)
            with torch.no_grad():
                nll_values.append(model.energy(batch).cpu().reshape(-1))
            print(
                f"Evaluated holdout NLL {min(start + len(batch), nll_count)}/"
                f"{nll_count}",
                flush=True,
            )
    mean_holdout_nll = (
        float(torch.cat(nll_values).mean()) if nll_values else None
    )

    distance_weights = np.repeat(weights, N_PAIR_DISTANCES)
    energy_metrics = distribution_metrics(
        reference_energies, generated_energies, weights
    )
    distance_metrics = distribution_metrics(
        reference_distances.reshape(-1),
        generated_distances.reshape(-1),
        distance_weights,
    )
    rank_wasserstein = [
        float(
            wasserstein_distance(
                reference_features[:, rank],
                generated_features[:, rank],
            )
        )
        for rank in range(N_PAIR_DISTANCES)
    ]
    rank_wasserstein_reweighted = [
        float(
            wasserstein_distance(
                reference_features[:, rank],
                generated_features[:, rank],
                v_weights=weights,
            )
        )
        for rank in range(N_PAIR_DISTANCES)
    ]
    swd = sliced_wasserstein(
        reference_features,
        generated_features,
        args.sliced_projections,
        args.seed + 2,
    )
    swd_reweighted = sliced_wasserstein(
        reference_features,
        generated_features,
        args.sliced_projections,
        args.seed + 2,
        weights,
    )

    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    effective_sample_size = float(1.0 / np.sum(weights**2))
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "generated_samples": args.samples,
        "holdout_samples": args.test_samples,
        "generation_seconds": generation_seconds,
        "samples_per_second": args.samples / generation_seconds,
        "time_per_sample_seconds": generation_seconds / args.samples,
        "sampling_nfe": {
            "mean": float(np.mean(sampling_nfes)),
            "std": float(np.std(sampling_nfes)),
            "min": int(np.min(sampling_nfes)),
            "max": int(np.max(sampling_nfes)),
            "per_batch": sampling_nfes,
            "nominal_batch_size": args.batch_size,
        },
        "integrator_atol": args.atol,
        "integrator_rtol": args.rtol,
        "importance_sampling": {
            "effective_sample_size": effective_sample_size,
            "effective_sample_size_fraction": effective_sample_size / args.samples,
            "log_weight": describe(log_weights),
        },
        "energy": energy_metrics,
        "pair_distance": distance_metrics,
        "configuration": {
            "sorted_distance_sliced_wasserstein": swd,
            "sorted_distance_sliced_wasserstein_reweighted": swd_reweighted,
            "distance_rank_wasserstein": rank_wasserstein,
            "distance_rank_wasserstein_reweighted": rank_wasserstein_reweighted,
        },
        "holdout_nll": {
            "samples": nll_count,
            "mean": mean_holdout_nll,
        },
    }

    plot_scalar_distribution(
        reference_energies,
        generated_energies,
        weights,
        "Potential energy",
        "Energy distribution",
        args.output_dir / "energy_distribution.png",
    )
    plot_scalar_distribution(
        reference_distances.reshape(-1),
        generated_distances.reshape(-1),
        distance_weights,
        "Pair distance",
        "Pair-distance distribution",
        args.output_dir / "distance_distribution.png",
    )
    pca, reference_2d, generated_2d = plot_invariant_distribution(
        reference_features,
        generated_features,
        args.output_dir / "configuration_distribution.png",
        args.seed + 3,
    )
    metrics["configuration"]["pca_explained_variance_ratio"] = [
        float(value) for value in pca.explained_variance_ratio_
    ]
    plot_gallery(
        holdout.numpy(),
        generated,
        args.output_dir / "configuration_gallery.png",
        args.seed + 4,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(1, N_PAIR_DISTANCES + 1)
    axis.plot(
        positions,
        rank_wasserstein,
        marker="o",
        label="Generated",
    )
    axis.plot(
        positions,
        rank_wasserstein_reweighted,
        marker="o",
        label="Generated, reweighted",
    )
    axis.set(
        xlabel="Sorted pair-distance rank",
        ylabel="Wasserstein-1 distance",
        title="Per-rank invariant distance error",
        xticks=positions,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "distance_rank_wasserstein.png", dpi=220
    )
    plt.close(figure)

    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        generated_samples=generated,
        latent_samples=latent,
        holdout_samples=holdout.numpy(),
        generated_energies=generated_energies,
        holdout_energies=reference_energies,
        generated_pair_distances=generated_distances,
        holdout_pair_distances=reference_distances,
        log_weights=log_weights,
        normalized_weights=weights,
        invariant_pca_holdout=reference_2d,
        invariant_pca_generated=generated_2d,
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Evaluation written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
