import sys

import torch
import torch.nn as nn

from einops import rearrange

sys.path.append("./")
from models import ErwinTransformer
from models.erwin import (
    NSAMSA,
    BallMSA,
    NativelySparseBallAttention,
    BasicLayer,
    SpErwinTransformer,
)

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
        torch.abs(jacobian) > 0, dim=(-2, -1), keepdim=True
    ).squeeze()

    affected_nodes = node_positions[thresholded]

    # affected NB
    ax.scatter(node_positions[:, 0], node_positions[:, 1])
    ax.scatter(
        affected_nodes[:, 0],
        affected_nodes[:, 1],
        color="orange",
    )
    ax.scatter(
        node_positions[i, 0],
        node_positions[i, 1],
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

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    node_positions = node_positions[tree_idx]
    node_features = node_features[tree_idx]
    node_features.requires_grad_(True)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    config = {
        "c_in": feature_dim,
        "c_hidden": feature_dim,
        "ball_sizes": [ball_size],
        "enc_num_heads": [
            1,
        ],
        "enc_depths": [
            2,
        ],
        "dec_num_heads": [],
        "dec_depths": [],
        "strides": [],  # no coarsening
        "mp_steps": 0,  # no MPNN
        "decode": True,  # no decoder
        "dimensionality": pos_dim,  # for visualization
        "rotate": 45,
    }

    # Erwin
    model = ErwinTransformer(**config)
    add_model_visual_field(
        lambda x: model(x, node_positions, batch_idx),
        node_features,
        node_positions,
        i,
        axes[0],
    )

    # Erwin with NSA
    model = SpErwinTransformer(**config)
    add_model_visual_field(
        lambda x: model(x, node_positions, batch_idx),
        node_features,
        node_positions,
        i,
        axes[1],
    )

    axes[0].set_title("Receptive field of Erwin")
    axes[1].set_title("Receptive field of Erwin-NSA")
    axes[0].legend()
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("field.pdf")
