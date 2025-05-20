import torch
import torch.nn as nn


def compute_specific_grads(
    model: nn.Module, x: torch.Tensor, target_idx: int
) -> torch.Tensor:
    """
    Computes Jacobian d(out_target)/dx_i of output of model with respect to input.

    Arguments:
        model: nn.Module model mapping a (n, d) tensor to a (n, d) tensor
        x: tensor of shape (n, d)
        target: output of d(out_target)/dx_i

    Returns:
            (n, d, d) tensor containing Jacobians d(out_target)/dx_i for each input point i
    """
    assert x.requires_grad

    out = model(x)
    n, d = x.shape
    jacobian = torch.zeros((n, d, d), device=x.device)

    for k in range(d):
        if x.grad is not None:
            x.grad.zero_()

        out[target_idx, k].backward(retain_graph=True)

        for j in range(n):
            jacobian[j, k, :] = x.grad[j].detach()

    return jacobian
