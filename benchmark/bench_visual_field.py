from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from models import ErwinTransformer


def compute_specific_grads(
        model: nn.Module,
        x: torch.Tensor,
        i: int,
        j: int | None = None
    ) -> torch.Tensor:
    """
    Arguments:
        model: nn.Module model mapping a (n, d) tensor to a (n, d') tensor
        x: tensor of shape (n, d) with grads enabled
        i: i-th datapoint for which the gradients will be computed
        j: j-th output point as target fn. If none, computes for all outputs.

    Returns:
        If j is specified:
            (d, d') tensor containig the Jacobian d(out_j)/dx_i
        Else:
            (n, d, d') tensor containing Jacobians for each output point
    """
    out = model(x)
    n, d = x.shape
    d_prime = out.shape[-1]

    j_is_set = j is not None

    if j_is_set:
        jacobian_shape = (d, d_prime)
    else:
        jacobian_shape = (n, d, d_prime)

    jacobian = torch.zeros(jacobian_shape, device=x.device)
    output_it = range(j, j + 1) if j is not None else range(n)

    for j_prime in output_it:
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


def compute_interaction(model: ErwinTransformer,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        i: int,
        threshold: float = 0.0
        ):
    def erwin_wrapper(x):
        return model(x, pos, batch_idx)

    jacobian = compute_specific_grads(erwin_wrapper, x, i)
    thresholded = torch.linalg.matrix_norm(jacobian) > threshold
    interaction_count = thresholded.sum()

    return interaction_count


if __name__ == "__main__":
    simple_test()

    config = {
        "c_in": 16,
        "c_hidden": 16,
        "ball_sizes": [128],
        "enc_num_heads": [1,],
        "enc_depths": [2,],
        "dec_num_heads": [],
        "dec_depths": [],
        "strides": [], # no coarsening
        "mp_steps": 0, # no MPNN
        "decode": True, # no decoder
        "dimensionality": 2, # for visualization
        "rotate": 0,
    }

    model = ErwinTransformer(**config)

    bs = 1
    num_points = 1024

    node_features = torch.randn(num_points * bs, config["c_in"], requires_grad=True)
    node_positions = torch.rand(num_points * bs, config["dimensionality"])
    batch_idx = torch.repeat_interleave(torch.arange(bs), num_points)
