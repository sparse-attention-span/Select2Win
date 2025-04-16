import torch
import torch.nn as nn

from bench_visual_field import compute_specific_grads, measure_interaction

SEED = 42


def test_interaction_identity():
    """ Check Jacobian of identity map """
    def model(x, *args):
        return x

    n, d = 4, 5
    torch.manual_seed(SEED)
    samples  = torch.randn(n, d, requires_grad=True)
    tol = 1e-20
    I = torch.eye(d, dtype=samples.dtype, device=samples.device)


    for i in range(n):
        for j in range(n):
            # test specific jacobs
            jacobian = compute_specific_grads(model, samples, i, j)

            assert list(jacobian.shape) == [d, d]

            if i != j:
                assert torch.all(torch.abs(jacobian) < tol)
            else:
                assert torch.all(torch.abs(jacobian - I) < tol)

        # only dy_i/x_i should be 1, so should be 1
        assert measure_interaction(model, samples, i=i) == 1

    print("Passed test for identity")


def test_interaction_matmul():
    """ Check Jacobian of linear map """
    n, d, d_prime = 4, 5, 6

    torch.manual_seed(SEED)
    W = torch.randn(d, d_prime)
    samples  = torch.randn(n, d, requires_grad=True)

    def model(x, *args):
        return x @ W

    tol = 1e-20

    for i in range(n):
        for j in range(n):
            # test specific jacobs
            jacobian = compute_specific_grads(model, samples, i, j)

            assert list(jacobian.shape) == [d_prime, d]

            if i != j:
                assert torch.all(torch.abs(jacobian) < tol)
            else:
                assert torch.all(torch.abs(jacobian - W.T) < tol)

        # no interaction between input vars
        assert measure_interaction(model, samples, i=i) == 1

    print("Passed test for matmul")


def test_interaction_scalar():
    """ Check when output is a scalar """
    def model(x, *args):
        return x.sum().reshape(1, 1)

    n, d, n_prime, d_prime = 4, 5, 1, 1
    torch.manual_seed(SEED)
    samples  = torch.randn(n, d, requires_grad=True)
    tol = 1e-20
    target = torch.ones(n_prime, d_prime, d)


    for i in range(n):
        jacobian = compute_specific_grads(model, samples, i)
        assert list(jacobian.shape) == [1, d_prime, d]
        assert torch.all(torch.abs(jacobian - target) < tol)

        # only dy_i/x_i should be 1, so should be 1
        assert measure_interaction(model, samples, i=i) == 1

    print("Passed test for scalar")


def test_interaction_multi_interaction():
    """ Check when output is a scalar """
    def model(x, *args):
        return x.prod(dim=0, keepdim=True) * x.prod(dim=0, keepdim=True).T

    n, d, = 4, 5
    torch.manual_seed(SEED)
    samples  = torch.arange(n * d, dtype=torch.float32).reshape(n, d)
    samples.requires_grad = True

    for i in range(n):
        assert measure_interaction(model, samples, i=i) > 1

    print("Passed test for multi_interaction")



if __name__ == "__main__":
    test_interaction_identity()
    test_interaction_matmul()
    test_interaction_scalar()
    test_interaction_multi_interaction()
