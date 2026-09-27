"""Shared alanine-dipeptide CUDA training implementation."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np
import ot as pot
import six
import sys
import torch
from scipy.stats import vonmises

# bgmol still imports the copy of six that old MDTraj versions bundled.
sys.modules.setdefault("mdtraj.utils.six", six)

from bgflow import BlackBoxDynamics, BoltzmannGenerator, DiffEqFlow
from bgflow import MeanFreeNormalDistribution
from bgflow.utils import IndexBatchIterator, remove_mean
from bgmol.datasets import AImplicitUnconstrained
from eq_ot_flow.estimator import BruteForceEstimatorFast
from eq_ot_flow.models import EGNN_dynamics_AD2_cat
from experiments.common import ROOT, report, require_cuda
from experiments.common import resume_if_requested, save_checkpoint, seed_everything


def train(args, weighted: bool) -> None:
    require_cuda()
    seed_everything(args.seed)
    n_particles, n_dimensions, dim = 22, 3, 66

    # bgmol normally downloads this small topology file to the node's temp dir.
    # Seed that cache from the repository so compute nodes need no internet access.
    shutil.copyfile(ROOT / "data/A.pdb", Path(tempfile.gettempdir()) / "A.pdb")
    dataset = AImplicitUnconstrained(root=str(ROOT / "data"), read=False)
    target = dataset.get_energy_model(n_workers=1)
    topology = dataset.system.mdtraj_topology

    atom_lookup = {"H": 0, "C": 1, "N": 2, "O": 3}
    atom_types = np.array([atom_lookup[atom.name[0]] for atom in topology.atoms])
    atom_types[[4, 6, 8, 14, 16]] = np.arange(4, 9)
    h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))

    data = torch.from_numpy(np.load(ROOT / "data/AD2_relaxed.npy")).float() / 10
    data = remove_mean(data, n_particles, n_dimensions).reshape(-1, dim) * 10
    if weighted:
        trajectory = md.Trajectory(
            data.numpy().reshape(-1, n_particles, n_dimensions), topology
        )
        phi = md.compute_phi(trajectory)[1].ravel()
        weights = 150 * vonmises.pdf(phi - 1.0, 10) + 1
        indices = np.random.choice(
            np.arange(len(data)), len(data), p=weights / weights.sum(), replace=True
        )
        data = data[indices]

    prior = MeanFreeNormalDistribution(
        dim, n_particles, two_event_dims=False
    ).cuda()
    dynamics = EGNN_dynamics_AD2_cat(
        n_particles=n_particles,
        device="cuda",
        n_dimension=n_dimensions,
        h_initial=h_initial,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=5,
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
            x1 = data[batch_indices].cuda()
            batch_size = x1.shape[0]
            t = torch.rand(batch_size, 1, device="cuda")
            x0 = prior.sample(batch_size)
            cost = torch.cdist(x0, x1).square()
            cost /= cost.max()
            plan = pot.emd(
                pot.unif(batch_size), pot.unif(batch_size), cost.detach().cpu().numpy()
            )
            choices = np.random.choice(
                plan.size, p=plan.ravel() / plan.sum(), size=batch_size
            )
            source_indices, target_indices = np.divmod(choices, plan.shape[1])
            x0, x1 = x0[source_indices], x1[target_indices]
            interpolation = x0 * (1 - t) + x1 * t
            x = interpolation + args.sigma * prior.sample(batch_size)
            loss = torch.mean((dynamics(t, x) - (x1 - x0)).square())
            loss.backward()
            optimizer.step()
            if iteration % args.report_every == 0:
                report(epoch, iteration, len(batches), loss)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args.output, model, optimizer, epoch, args)
