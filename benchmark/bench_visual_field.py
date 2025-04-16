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
        *args,
        threshold: float = 0.0
        ) -> int:
    """ Computes \sum_j |d(out_j)/dx_i| > 1 """

    def model_wrapper(x):
        return model(x, *args)

    jacobian = compute_specific_grads(model_wrapper, x, i)
    thresholded = torch.any(torch.abs(jacobian) > threshold, dim=(-2, -1))
    interaction_count = thresholded.sum().item()
    return interaction_count


if __name__ == "__main__":
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
