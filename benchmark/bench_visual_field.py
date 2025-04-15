import torch
import torch.nn as nn

from models import ErwinTransformer


def measure_interaction(
        model: ErwinTransformer,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        i: int,
        j: int
    ) -> torch.Tensor:
    """
    Arguments:
        model: Erwin transformer instance
        x: tensor of shape (n * bs) x c_in
    """
    d = x.shape[-1]
    out = model(x, pos, batch_idx)
    jacobian = torch.zeros((d, d), device=x.device)

    for k in range(d):
        if x.grad is not None:
            x.grad.zero_()
        out[j, k].backward(retain_graph=True)
        grad_i = x.grad[i].detach().clone()
        jacobian[k] = grad_i

    return jacobian


def total_interaction(model: ErwinTransformer,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        i: int):
    n, d = x.shape
    out = model(x, pos, batch_idx)
    interaction_count = 0

    for j in range(n):
        jacobian = torch.zeros((d, d), device=x.device)

        for k in range(d):
            if x.grad is not None:
                x.grad.zero_()
            out[j, k].backward(retain_graph=True)
            grad_i = x.grad[i].detach().clone()
            jacobian[k] = grad_i

        if torch.linalg.matrix_norm(jacobian) > 0:
            interaction_count += 1

    return interaction_count


def simple_test():
    """ Check Jacobian of identity map """
    def model(x, *args):
        return x

    n, d = 4, 5
    samples  = torch.randn(n, d, requires_grad=True)
    tol = 1e-20
    I = torch.eye(d, dtype=samples.dtype, device=samples.device)

    for i in range(n):
        for j in range(n):
            jacob = measure_interaction(model, samples, None, None, i, j)

            if i != j:
                assert torch.all(torch.abs(jacob) < tol)
            else:
                assert torch.all(torch.abs(jacob - I) < tol)

        # only dy_i/x_i should be 1, so should be 1
        assert total_interaction(model, samples, None, None, i) == 1



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
