import re
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat
from typing import Tuple


# Create synthetic molecule-like 2D points
def create_molecule_like_points(num_atoms=20, num_points_per_atom=400, seed=42):
    np.random.seed(seed)

    # Generate random atom centers
    atom_centers = np.random.rand(num_atoms, 3) * 10  # Scale for better visualization

    # Create points around each atom (simulating electron density)
    all_points = []
    for center in atom_centers:
        # Generate points with Gaussian distribution around atom centers
        points = np.random.normal(loc=center, scale=0.4, size=(num_points_per_atom, 3))
        all_points.append(points)

    # Combine all points
    molecule_points = np.vstack(all_points)

    # # Create bonds between some atoms (lines of points)
    # for i in range(num_atoms - 1):
    #     if np.random.random() < 0.6:  # 60% chance of bond between consecutive atoms
    #         start, end = atom_centers[i], atom_centers[i + 1]
    #         # Points along the bond
    #         t = np.linspace(0, 1, 50).reshape(-1, 1)
    #         bond_points = start * (1 - t) + end * t
    #         all_points.append(bond_points)

    # molecule_points = np.vstack(all_points)

    # Convert to 3D by setting z=0
    # molecule_3d = np.column_stack((molecule_points, np.zeros(len(molecule_points))))

    return torch.tensor(molecule_points, dtype=torch.float)


# Generate molecule-like points and replace the loaded points
# Generate 20 batches of these points, with 4 heads each.
points = torch.stack(
    [
        torch.stack(
            [
                create_molecule_like_points(
                    num_atoms=15, num_points_per_atom=500, seed=s * t
                )
                for s in range(20)
            ]
        )
        for t in range(4)
    ]
)


def select_balls(
    q: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    queries = rearrange(q, "b H n m E -> b H (n m) E")
    keys = reduce(k, "b H n m E -> b H E n", "mean")
    similarity = queries @ keys
    topk_values, topk_indices = torch.topk(similarity, 2, dim=-1)
    return topk_indices

def forward(x: torch.Tensor):
    layer = nn.Linear(3, 15)
    layer.to("cuda")
    x = layer(x)
    q, k, v = rearrange(
        x,
        "H b (n m) (K E) -> K b H n m E",
        b=20,
        H=4,
        m=500,
        K=3,
    )
    # tensor are of shape b h (n m) topk
    num_points = q.shape[-2] * q.shape[-1]
    topk_indices = select_balls(q, k)
    num_balls = topk_indices.shape[-1]

    # print(topk_indices[0, 0, 0])

    # gates = straight_through(topk_values, 1.0) if self.use_diff_topk else None

    out = torch.zeros_like(v)
    out = rearrange(out, "b H n m E -> b H (n m) E")

    print(f"K:{k.shape}")
    print(f"V:{v.shape}")
    print(f"topk:{topk_indices.shape}")
    print(f"Q:{q.shape}")
    print(f"out:{out.shape}")
    for t in range(num_points):
        for b in num_balls:
            ball = k[]

points = points.to("cuda")
forward(points)
