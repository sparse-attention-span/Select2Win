import sys

import torch
import torch.nn as nn

from einops import rearrange

sys.path.append("./")
from models import ErwinTransformer
from models.erwin import NSAMSA_triton, BallMSA, NSAMSA

from benchmark.bench_visual_field import compute_specific_grads


def generate_point_cloud(
    n_groups=5, samples_per_group=4, std_dev=0.1, seed=None, angle: float = 0.0
):
    """
    Generates a 2D point cloud consisting of samples around the roots of z^n_groups = 1.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Compute the n-th roots of unity
    roots = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(torch.tensor(2 * torch.pi * k / n_groups)),
                    torch.sin(torch.tensor(2 * torch.pi * k / n_groups)),
                ]
            )
            for k in range(n_groups)
        ]
    )

    # Sample points around roots
    points = []

    for root in roots:
        mean = root.expand(samples_per_group, 2)
        std = torch.full_like(mean, std_dev)
        samples = torch.normal(mean=mean, std=std)
        points.append(samples)

    # Rotate to have nicer balls when doing the ball tree construction
    rotation_matrix = get_rotation_matrix(angle)
    points = torch.vstack(points)
    points = points @ rotation_matrix.T

    return points.contiguous(), points.clone().contiguous()


def get_rotation_matrix(angle: float) -> torch.Tensor:
    theta = torch.deg2rad(torch.tensor(angle))
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )
    return rotation_matrix


def add_model_visual_field(model, x, node_positions, i, ax):
    jacobian = compute_specific_grads(model, x, i)
    jacobian = jacobian.detach()
    thresholded = torch.any(
        torch.abs(jacobian) > 0.00005, dim=(-2, -1), keepdim=True
    ).squeeze()

    affected_nodes = node_positions[thresholded].cpu()

    # affected NB
    ax.scatter(node_positions[:, 0].cpu(), node_positions[:, 1].cpu())
    ax.scatter(
        affected_nodes[:, 0],
        affected_nodes[:, 1],
        color="orange",
    )
    ax.scatter(
        node_positions[i, 0].cpu(),
        node_positions[i, 1].cpu(),
        marker="x",
        s=100,
        color="black",
        label="Target node",
    )


if __name__ == "__main__":
    from balltree import build_balltree
    import matplotlib.pyplot as plt

    # seed
    import random
    import numpy as np

    # SEED = 2818923
    SEED = 303489
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPS = 1e-20
    feature_dim = 2
    pos_dim = 2
    n_balls = 8
    std_dev_samples = 0.1
    sample_angle = 15
    num_samples = 16
    ball_size = num_samples
    num_points = num_samples * n_balls
    num_heads = 1
    topk = 2
    use_diff_topk = False
    thetas = [0]
    run_unit_tests = True

    assert (num_points > 0) and (
        (num_points & (num_points - 1)) == 0
    ), "Num points must be power of 2"
    assert 1 <= len(thetas) <= 2

    i = 0

    batch_idx = torch.repeat_interleave(torch.arange(1), num_points)
    node_positions, node_features = generate_point_cloud(
        n_groups=n_balls,
        samples_per_group=num_samples,
        std_dev=std_dev_samples,
        angle=sample_angle,
    )
    tree_idx, tree_mask = build_balltree(node_positions, batch_idx)
    assert tree_mask.all()
    node_positions = node_positions[tree_idx].to(device)
    node_features = node_features[tree_idx].to(device)
    node_features.requires_grad_(True)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    # dim: int,
    #     num_heads: int, # space dim
    #     ball_size: int,
    #     dimensionality: int = 3, #feature dim
    #     topk: int = 2,
    #     use_diff_topk: bool = True,
    #     selection_ball_size: int = 16,
    #     use_flex: bool = False,
    #     custom_num_heads: int | None = None,
    #     head_dim_factor: int = 1,
    #     masks: bool = True,
    #     size=16384
    config = {
        "dim": feature_dim,
        "num_heads": num_heads,
        "dimensionality": pos_dim,  # for visualization
        "ball_size": num_samples,
        "selection_ball_size": 16,
        "topk": 3,
        "masks": True,
    }

    configBall = {
        "dim": feature_dim,
        "num_heads": num_heads,
        "dimensionality": pos_dim,  # for visualization
        "ball_size": num_samples,
    }

    # Erwin
    # model = BallMSA(**configBall).to(device)
    # add_model_visual_field(
    #     lambda x: model(x, node_positions, batch_idx),
    #     node_features,
    #     node_positions,
    #     i,
    #     axes[0],
    # )

    # Erwin with NSA
    # model = NSAMSA_triton(**config).to(device)
    # add_model_visual_field(
    #     lambda x: model(x, node_positions, 1),
    #     node_features,
    #     node_positions,
    #     i,
    #     axes[0],
    # )

    model = NSAMSA(**config).to(device)
    add_model_visual_field(
        lambda x: model(x, node_positions, 1),
        node_features,
        node_positions,
        i,
        axes[1],
    )

    axes[0].set_title("Receptive field of triton")
    axes[1].set_title("Receptive field of pytorch")
    axes[0].legend()
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("field.png")
