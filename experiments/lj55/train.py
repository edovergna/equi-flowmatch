#!/usr/bin/env python3
"""Train equivariant optimal-transport flow matching on LJ55."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import ot as pot
from scipy.optimize import linear_sum_assignment
import torch

from bgflow import BlackBoxDynamics, BoltzmannGenerator, DiffEqFlow
from bgflow import MeanFreeNormalDistribution
from bgflow.utils import IndexBatchIterator, remove_mean
from eq_ot_flow.estimator import BruteForceEstimatorFast
from eq_ot_flow.LJ import LennardJonesPotential
from eq_ot_flow.models import EGNN_dynamics
from eq_ot_flow.utilities import superpose_points, superpose_points_batch
from experiments.common import base_parser, report, require_cuda
from experiments.common import resume_if_requested, save_checkpoint, seed_everything


def main() -> None:
    parser = base_parser("Train LJ55 equivariant OT flow matching", 600, 5e-4)
    parser.set_defaults(output=ROOT / "runs/lj55/checkpoint.pt")
    args = parser.parse_args()
    require_cuda()
    seed_everything(args.seed)

    dim, n_particles, n_dimensions = 165, 55, 3
    target = LennardJonesPotential(
        dim, n_particles, eps=1.0, rm=1, oscillator_scale=1, two_event_dims=False
    )
    loaded = np.load(ROOT / "data/all_data_LJ55-1000.npy", allow_pickle=True)
    all_data = torch.from_numpy(remove_mean(loaded, n_particles, n_dimensions))
    indices = np.random.permutation(len(all_data))
    data = all_data[indices[:100000]]

    prior = MeanFreeNormalDistribution(
        dim, n_particles, two_event_dims=False
    ).cuda()
    prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
    dynamics = EGNN_dynamics(
        n_particles=n_particles,
        device="cuda",
        n_dimension=n_dimensions,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=7,
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
    model = BoltzmannGenerator(prior, flow, target).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = resume_if_requested(model, optimizer, args.resume)

    for epoch in range(start_epoch, args.epochs):
        batches = IndexBatchIterator(len(data), args.batch_size)
        for iteration, batch_indices in enumerate(batches):
            if args.max_batches is not None and iteration >= args.max_batches:
                break
            optimizer.zero_grad()
            x1 = data[batch_indices]
            batch_size = x1.shape[0]
            t = torch.rand(batch_size, 1, device="cuda")
            x0 = prior_cpu.sample(batch_size)

            cost = torch.zeros(batch_size, batch_size)
            for source_index in range(batch_size):
                source = x0[source_index].reshape(n_particles, n_dimensions)
                reordered = []
                for target_index in range(batch_size):
                    destination = x1[target_index].reshape(n_particles, n_dimensions)
                    distances = torch.cdist(source, destination)
                    _, columns = linear_sum_assignment(distances.square())
                    reordered.append(destination[columns].unsqueeze(0))
                aligned = superpose_points_batch(torch.cat(reordered), reference=source)
                cost[source_index] = torch.cdist(
                    source.reshape(1, -1), aligned.reshape(batch_size, -1)
                )[0]

            cost = cost.square()
            cost /= cost.max()
            plan = pot.emd(
                pot.unif(batch_size), pot.unif(batch_size), cost.numpy()
            )
            choices = np.random.choice(
                plan.size, p=plan.ravel() / plan.sum(), size=batch_size
            )
            source_indices, target_indices = np.divmod(choices, plan.shape[1])
            x0, x1 = x0[source_indices], x1[target_indices]
            for index in range(batch_size):
                source = x0[index].reshape(n_particles, n_dimensions)
                destination = x1[index].reshape(n_particles, n_dimensions)
                distances = torch.cdist(source, destination)
                _, columns = linear_sum_assignment(distances.square())
                x1[index] = superpose_points(destination[columns], source).flatten()

            x0, x1 = x0.cuda(), x1.cuda()
            interpolation = t * x1 + (1 - t) * x0
            x = interpolation + args.sigma * prior.sample(batch_size)
            loss = torch.mean((dynamics(t, x) - (x1 - x0)).square())
            loss.backward()
            optimizer.step()
            if iteration % args.report_every == 0:
                report(epoch, iteration, len(batches), loss)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args.output, model, optimizer, epoch, args)


if __name__ == "__main__":
    main()
