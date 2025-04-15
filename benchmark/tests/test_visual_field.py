import torch
import torch.nn as nn

from ..bench_visual_field import measure_interaction

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