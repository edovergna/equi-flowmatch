import torch
from bgflow.utils import as_numpy
import matplotlib.pyplot as plt
def superpose_points(points1, points2):


    # Compute optimal rotation and translation
    M = torch.matmul(points1.t(), points2)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.t())

    # Apply rotation and translation to superpose points1 onto points2
    superposed_points1 = torch.matmul(points1, R) 
    
    return superposed_points1

def superpose_points_batch(points, reference):


    # Compute optimal rotation and translation
    M = torch.matmul(points.transpose(-2, -1), reference)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.transpose(-2, -1))
    # Apply rotation and translation to superpose points1 onto points2
    superposed_points = torch.matmul(points, R) 
    
    return superposed_points


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def plot_flowpath_trajectory(traj, n_dimensions=2):
    plt.figure(figsize=(9,9))
    latent_sample = as_numpy(traj[0].reshape(-1, n_dimensions))
    target_sample = as_numpy(traj[-1].reshape(-1, n_dimensions))
    plt.scatter(*latent_sample.T, alpha=0.95, label="latent",s=100)
    traj = as_numpy(traj)
    plt.scatter(traj[:, :,0].flatten(), traj[:, :,1].flatten(), color="black", s=10, label="path")
    plt.scatter(*target_sample.T, alpha=0.95, label="target",s=100)
    plt.legend()
    plt.title("Flow path", fontsize=45);
    plt.xticks(fontsize=45) 
    plt.yticks(fontsize=45);
    plt.legend(fontsize=25);
    
def plot_flowpath_trajectory_3d(traj, n_dimensions=3):
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(projection='3d')
    latent_sample = as_numpy(traj[0].reshape(-1, n_dimensions))
    target_sample = as_numpy(traj[-1].reshape(-1, n_dimensions))
    ax.scatter(*latent_sample.T, alpha=0.95, label="latent",s=100)

    traj = as_numpy(traj)
    ax.scatter(traj[:, :,0].flatten(), traj[:, :,1].flatten(), traj[:, :,2].flatten(), color="black", s=10, label="path")
    ax.scatter(*target_sample.T, alpha=0.95, label="target",s=100)
    ax.set_xlim((-1.5,1.5))
    ax.set_ylim((-1.5,1.5))
    ax.set_zlim((-1.5,1.5))
