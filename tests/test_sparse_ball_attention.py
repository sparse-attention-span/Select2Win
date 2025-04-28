import sys

import torch
import torch.nn as nn

from einops import rearrange

sys.path.append("./")
from models import ErwinTransformer
from models.erwin import NSAMSA, BallMSA

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

    points = []
    point_features = []

    for root in roots:
        mean = root.expand(samples_per_group, 2)
        std = torch.full_like(mean, std_dev)
        samples = torch.normal(mean=mean, std=std)
        points.append(samples)
        features = mean
        point_features.append(features)

    points = torch.vstack(points)
    point_features = torch.vstack(point_features)

    rotation_matrix = get_rotation_matrix(angle)
    points = points @ rotation_matrix.T
    point_features = point_features @ rotation_matrix.T

    return points.contiguous(), point_features.contiguous()


def get_rotation_matrix(angle: float) -> torch.Tensor:
    theta = torch.deg2rad(torch.tensor(angle))
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )
    return rotation_matrix


def test_data_roots(debug_data):
    x = debug_data["x_with_emb"]
    ball_size = debug_data["ball_size"]
    topk_idx = debug_data["topk_idx"]
    topk_idx = rearrange(topk_idx, "1 1 (n m) topk -> n m topk", n=ball_size)

    # check if selected indices agree with topk and OG values
    k_after_gather = debug_data["k_after_gather"]
    k_after_gather = rearrange(k_after_gather, "1 1 a b c d -> a b c d")

    for idx, topk in enumerate(rearrange(topk_idx, "n m topk -> (n m) topk")):
        for grp_idx, topk_grp_idx in enumerate(topk):
            ball_start = topk_grp_idx * ball_size
            ball_end = ball_start + ball_size
            assert torch.all(
                x[ball_start:ball_end] == k_after_gather[idx, grp_idx]
            ), "selection not working correctly"

    # check if points in ball select the same balls, since they have the same features
    # if added pos embs, then this is not always true
    # also doesnt account for arbitrary ties
    # assert torch.all(
    #     (topk_idx.min(1).values - topk_idx.max(1).values) == 0
    # ), "Not all balls attend to the same feature (is stdev sufficiently small?)"


if __name__ == "__main__":
    from balltree import build_balltree
    import matplotlib.pyplot as plt

    # if 32 * 8, with topk=4 and sort then not ok
    # if 8 * 8, with even topk, then not ok (also with above fixed)

    EPS = 1e-20
    feature_dim = 2
    pos_dim = 2
    ball_size = 4
    n_balls = 8
    num_points = ball_size * n_balls
    std_dev_samples = 0.1
    sample_angle = 15
    num_heads = 1
    topk = 2
    use_diff_topk = False
    thetas = [0]
    assert (num_points > 0) and (
        (num_points & (num_points - 1)) == 0
    ), "Num points must be power of 2"
    assert 1 <= len(thetas) <= 2

    i = 0

    batch_idx = torch.repeat_interleave(torch.arange(1), num_points)
    node_positions, node_features = generate_point_cloud(
        n_groups=n_balls,
        samples_per_group=ball_size,
        std_dev=std_dev_samples,
        angle=sample_angle,
    )
    tree_idx, tree_mask = build_balltree(node_positions, batch_idx)
    assert tree_mask.all()
    node_positions = node_positions[tree_idx]
    node_features = node_features[tree_idx]
    node_features.requires_grad_(True)

    print(f"#poinst: {num_points}")
    print(f"tree idx: {tree_idx.shape}; tree mask: {tree_mask.shape}")
    print(f"Should have {ball_size * topk} (possibly +1) activated grads")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 12),
        sharex=True,
        sharey=True,
    )

    for theta, ax in zip(thetas, axes):
        print(f"Computing theta={theta}")
        model = NSAMSA(
            dim=feature_dim,
            num_heads=num_heads,
            ball_size=ball_size,
            dimensionality=pos_dim,
            topk=topk,
            use_diff_topk=use_diff_topk,
        )
        # model = BallMSA(
        #     dim=feature_dim,
        #     num_heads=num_heads,
        #     ball_size=ball_size,
        #     dimensionality=pos_dim,
        # )

        def model_wrapper(x):
            out = model(x, node_positions, debug=True)
            return out

        # compute FOV of point_1
        jacobian, (_, debug_data) = compute_specific_grads(
            model_wrapper, node_features, i, return_out=True
        )
        jacobian = jacobian.detach()
        thresholded = torch.any(
            torch.abs(jacobian) > 0, dim=(-2, -1), keepdim=True
        ).squeeze()

        affected_nodes = node_positions[thresholded]

        # visualize OG balltree
        groups = rearrange(node_positions, "(n m) E -> n m E", n=n_balls)
        colors = plt.cm.get_cmap("tab20", n_balls)

        for group_idx in range(n_balls):
            points = groups[group_idx]
            ax[0].scatter(
                points[:, 0], points[:, 1], color=colors(group_idx), s=50, marker="o"
            )

        # affected NB
        ax[1].scatter(node_positions[:, 0], node_positions[:, 1])
        ax[1].scatter(affected_nodes[:, 0], affected_nodes[:, 1], color="orange")

        # grad norm - normalize values to see differences
        grad_norms = torch.linalg.matrix_norm(jacobian, dim=(-2, -1))
        nonzero_grad_idx = grad_norms > 0
        # grad_norms = torch.log(grad_norms[nonzero_grad_idx])
        grad_norms = grad_norms[nonzero_grad_idx]
        grad_norms = grad_norms.cpu().numpy()
        non_zero_grad_nodes = node_positions[nonzero_grad_idx]

        print(f"{nonzero_grad_idx.sum()} points with non-zero grad")
        ax[2].scatter(
            non_zero_grad_nodes[:, 0],
            non_zero_grad_nodes[:, 1],
            c=grad_norms,
            cmap="viridis",
        )

        for subax in ax:
            subax.scatter(
                node_positions[i, 0],
                node_positions[i, 1],
                marker="x",
                s=100,
                color="black",
            )

        ax[0].set_title(f"Ball tree theta={theta}")
        ax[1].set_title("Receptive field")
        ax[2].set_title("Log-gradient norm")

    # plot points after pos embedding
    if feature_dim == 2:
        node_features_with_emb = node_features.detach() + model.compute_rel_pos(
            node_positions
        )
        axes[1, 1].scatter(node_features_with_emb[:, 0], node_features_with_emb[:, 1])
        axes[1, 1].set_title("Features with pos embeddings")

        axes[1, 0].scatter(node_features.detach()[:, 0], node_features.detach()[:, 1])
        axes[1, 0].set_title("Features")

    plt.savefig("field.png")

    # check if everything OK
    test_data_roots(debug_data)
