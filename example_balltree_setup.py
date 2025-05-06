import numpy as np
import torch

from datasets import ShapenetCarDataset
from balltree import build_balltree


# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# points = torch.load("shapenet_car/mlcfd_data/processed/param1/1dc58be25e1b6e5675cad724c63e222e/mesh_points.th", weights_only=True)

# # center and scale for display
# # points = points - points.mean(0)
# # points = points / points.norm(dim=1).max()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.8)
# ax.view_init(elev=30, azim=120)
# ax.set_axis_off()
# plt.savefig("test.png")

# exit()


# seed = 42
# data_path = "./shapenet_car/mlcfd_data/processed"
# knn = 8


# train_dataset = ShapenetCarDataset(
#     data_path=data_path,
#     split="train",
#     knn=8,
# )
# Create synthetic molecule-like 2D points
def create_molecule_like_points(num_atoms=20, num_points_per_atom=400, seed=42):
    np.random.seed(seed)

    # Generate random atom centers
    atom_centers = np.random.rand(num_atoms, 2) * 10  # Scale for better visualization

    # Create points around each atom (simulating electron density)
    all_points = []
    for center in atom_centers:
        # Generate points with Gaussian distribution around atom centers
        points = np.random.normal(loc=center, scale=0.4, size=(num_points_per_atom, 2))
        all_points.append(points)

    # Combine all points
    molecule_points = np.vstack(all_points)

    # Create bonds between some atoms (lines of points)
    for i in range(num_atoms-1):
        if np.random.random() < 0.6:  # 60% chance of bond between consecutive atoms
            start, end = atom_centers[i], atom_centers[i+1]
            # Points along the bond
            t = np.linspace(0, 1, 50).reshape(-1, 1)
            bond_points = start * (1-t) + end * t
            all_points.append(bond_points)

    molecule_points = np.vstack(all_points)

    # Convert to 3D by setting z=0
    molecule_3d = np.column_stack((molecule_points, np.zeros(len(molecule_points))))

    return torch.tensor(molecule_3d, dtype=torch.float)

# Generate molecule-like points and replace the loaded points
points = create_molecule_like_points(num_atoms=15, num_points_per_atom=500)

# print(points)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig, axes = plt.subplots(nrows=1, ncols=6)


batch_idx = torch.repeat_interleave(torch.arange(1), points.shape[0])
tree_idx, tree_mask = build_balltree(points, batch_idx) # build tree
grouped_points = points[tree_idx] # sort points into the tree

level_to_node_size = lambda level: 2**(level)

fig = plt.figure(figsize=(18, 12))
fig.suptitle("3D Point Cloud by Level", fontsize=20)

nrows, ncols = 3, 4
n_levels = 12

assert n_levels <= nrows * ncols

for level in range(n_levels):
    groups = grouped_points.reshape(-1, level_to_node_size(level), 3) # (num balls, ball size, dim)
    num_balls, ball_size, _ = groups.shape


    ax = fig.add_subplot(nrows, ncols, level + 1, projection='3d')
    colors = plt.cm.viridis

    for i in range(num_balls):
        points = groups[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors(i), s=1)

    ax.set_title(f'Level {level} (Ball Size: {ball_size})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("test.png")