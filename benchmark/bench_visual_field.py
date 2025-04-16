import sys

import torch
import torch.nn as nn

sys.path.append("./")
from models import ErwinTransformer


def compute_specific_grads(
        model: nn.Module,
        x: torch.Tensor,
        i: int,
        j: int | None = None
    ) -> torch.Tensor:
    """
    Computes Jacobian d(out_j)/dx_i of output of model with respect to input.

    Arguments:
        model: nn.Module model mapping a (n, d) tensor to a (n, d') tensor
        x: tensor of shape (n, d) with grads enabled
        i: i-th datapoint for which the gradients will be computed
        j: j-th output point as target fn. If set to None, computes for all outputs.

    Returns:
        If j is specified:
            (d, d') tensor containig the Jacobian d(out_j)/dx_i
        Otherwise:
            (n, d, d') tensor containing Jacobians d(out_j)/dx_i for each output point j
    """
    out = model(x)
    n, d = x.shape
    device = x.device
    n_prime, d_prime = out.shape

    if (j_is_set := j is not None):
        jacobian_shape = (d_prime, d)
    else:
        jacobian_shape = (n_prime, d_prime, d)

    jacobian = torch.zeros(jacobian_shape, device=device)
    output_iter = [j] if j_is_set else range(n_prime)

    for j_prime in output_iter:
        for k in range(d_prime):
            # compute jacobian by computing each row d(out[j', k]/dx_i)
            # this is to prevent using jacobian function of PyTorch, which
            # computes unnecessarily many gradients
            if x.grad is not None:
                x.grad.zero_()

            out[j_prime, k].backward(retain_graph=True)
            grad_i = x.grad[i].detach().clone()

            if j_is_set:
                jacobian[k] = grad_i
            else:
                jacobian[j_prime, k, :] = grad_i

    return jacobian


def measure_interaction(model: nn.Module,
        x: torch.Tensor,
        i: int,
        j: int
        ) -> torch.Tensor:
    jacobian = compute_specific_grads(model, x, i, j)
    return jacobian


def measure_interaction_full(model: nn.Module,
        x: torch.Tensor,
        i: int,
        *args,
        threshold: float = 0.0
        ) -> float:
    """ Computes sum_j |d(out_j)/dx_i| > 1 """

    def model_wrapper(x):
        return model(x, *args)

    jacobian = compute_specific_grads(model_wrapper, x, i)
    thresholded = torch.any(torch.abs(jacobian) > threshold, dim=(-2, -1))
    interaction_count = thresholded.sum().item()
    return interaction_count


if __name__ == "__main__":
    from balltree import build_balltree, build_balltree_with_rotations
    import matplotlib.pyplot as plt
    import math

    EPS = 1e-20

    c_in = 16
    dimensionality = 2
    ball_sizes = [128]
    strides = []

    bs = 1
    num_points = 512
    i = 0

    node_features = torch.randn(num_points * bs, c_in, requires_grad=True)
    node_positions = torch.rand(num_points * bs, dimensionality)
    batch_idx = torch.repeat_interleave(torch.arange(bs), num_points)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=True, sharey=True)
    thetas = [0, 45.0]

    for theta, ax in zip(thetas, axes):
        print(f"Computing theta={theta}")
        config = {
            "c_in": c_in,
            "c_hidden": 16,
            "ball_sizes": ball_sizes,
            "enc_num_heads": [1,],
            "enc_depths": [2,],
            "dec_num_heads": [],
            "dec_depths": [],
            "strides": strides, # no coarsening
            "mp_steps": 0, # no MPNN
            "decode": True, # no decoder
            "dimensionality": dimensionality, # for visualization
            "rotate": theta,
        }

        model = ErwinTransformer(**config)

        def model_wrapper(x):
            return model(x, node_positions, batch_idx)

        # build tree
        rad = math.radians(theta)
        s, c = math.sin(rad), math.cos(rad)
        rot_mat = torch.tensor([[c, -s], [s, c]])
        tree_idx, tree_mask = build_balltree(node_positions @ rot_mat.T, batch_idx)
        grouped_points = node_positions[tree_idx]

        # compute FOV of point_1
        jacobian = compute_specific_grads(model_wrapper, node_features, i).detach()
        thresholded = torch.any(torch.abs(jacobian) > 0, dim=(-2, -1), keepdim=True).squeeze()
        affected_nodes = node_positions[thresholded]

        # visualize OG balltree
        groups = grouped_points.reshape(-1, config["ball_sizes"][0], config["dimensionality"]) # (num balls, ball size, dim)
        num_balls, ball_size, _ = groups.shape
        colors = plt.cm.get_cmap('tab20', num_balls)

        for group_idx in range(num_balls):
            points = groups[group_idx]
            ax[0].scatter(points[:, 0], points[:, 1], color=colors(group_idx), s=50, marker="o")

        # affected NB
        ax[1].scatter(node_positions[:, 0], node_positions[:, 1])
        ax[1].scatter(affected_nodes[:, 0], affected_nodes[:, 1], color="orange")

        # grad norm - normalize values to see differences
        grad_norms = torch.linalg.matrix_norm(jacobian, dim=(-2, -1))
        nonzero_grad_idx = grad_norms > 0
        grad_norms = torch.log10(grad_norms[nonzero_grad_idx] + EPS).cpu().numpy()
        non_zero_grad_nodes = node_positions[nonzero_grad_idx]

        print(nonzero_grad_idx.shape, grad_norms.shape, non_zero_grad_nodes.shape)
        ax[2].scatter(non_zero_grad_nodes[:, 0], non_zero_grad_nodes[:, 1], c=grad_norms, cmap='viridis')

        for subax in ax:
            subax.scatter(node_positions[i, 0], node_positions[i, 1], marker="x", s=100, color="black")

        ax[0].set_title(f"Ball tree theta={theta}")
        ax[1].set_title("Receptive field")
        ax[2].set_title("Re-scaled log-gradient norm")

    plt.savefig("field.png")

