import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange
import numpy as np


ASPECT_RATIO = (1, 2, 1)
ROTATE_ANGLE = 180
BALL_SIZE = 32
MIN_ALPHA = 0.01
POINT_SIZE=10


def add_args():
    parser = argparse.ArgumentParser(description="Visualize attention maps from saved data")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the saved PyTorch file containing attention map data")
    parser.add_argument("--output", type=str, default="attn_maps.png",
                        help="Output image filename (default: attn_maps.png)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show attention maps averaged over all points, overrides target_idx")
    parser.add_argument("--target-idx", type=int, nargs="+", default=[0],
                        help="Target indices to visualize if not showing all")
    parser.add_argument("--single-color", action="store_true", help="Uses 1 color.")
    parser.add_argument("--hide-ax", action="store_true", help="Disable axes in figures")

    args = parser.parse_args()
    return args

def add_point_cloud(ax, data, mask, cmap, single_color: int, aspect_ratio=(1, 1, 1), weights=None, hide_ax=False):
    n_points = len(data)
    assert n_points == len(mask)

    if weights is not None:
        assert len(weights) == len(data)
    else:
        weights = [None] * n_points

    # plot per group to get colors
    for j, (group_samples, group_weights, group_mask) in enumerate(zip(data, weights, mask)):
        if group_weights is not None:
            group_weights = group_weights[group_mask]
        group_samples = group_samples[group_mask]

        ax.scatter(group_samples[:, 0], group_samples[:, 1], group_samples[:, 2], alpha=group_weights,
                    color=cmap((j % cmap.N / cmap.N) * single_color), s=POINT_SIZE)

    ax.set_box_aspect(aspect=aspect_ratio)

    if hide_ax:
        ax.set_axis_off()

def rotate_around_x(points, degrees):
    theta = np.radians(degrees)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return points @ rot.T

def main(args):
    path = args.input_path
    data = torch.load(path, map_location="cpu", weights_only=True)
    tree_mask = data["tree_mask"]
    tree_idx = data["tree_idx"]
    attn_maps = data["attn_maps"]
    samples = data["sample"]["node_positions"]
    samples = samples[tree_idx]

    # shapenet and matplotlib coordinate system differ
    samples = samples - samples.mean(axis=0, keepdims=True)
    samples = rotate_around_x(samples, ROTATE_ANGLE)

    single_color = int(args.single_color)
    single_color = 1 - single_color

    show_all = args.show_all  # overrides settings below
    target_idx = args.target_idx
    show_target = not show_all and len(target_idx) == 1

    fig = plt.figure(figsize=(50, 10))
    nrows = len(attn_maps) + 1
    ncols = max(mp["value"].shape[1] for mp in attn_maps.values())
    cmap = plt.get_cmap("tab20")

    # plot ball tree
    ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
    samples_ = rearrange(samples, "(n m) d -> n m d", m=BALL_SIZE)  # todo: adapt for each layer
    tree_mask_ = rearrange(tree_mask, "(n m)-> n m", m=BALL_SIZE)  # todo: adapt for each layer
    add_point_cloud(ax, samples_, tree_mask_, cmap, single_color, aspect_ratio=ASPECT_RATIO, hide_ax=args.hide_ax)
    ax.set_title("Ball Partitioning")

    if show_target:
        target_point = samples[target_idx].squeeze()

    # plot attention maps
    for i, (layer_name, attn_map_data) in enumerate(attn_maps.items()):
        attn_map = attn_map_data["value"][0]  # assume batch size 1
        ball_size = attn_map_data["ball size"]

        start_ax_idx = (i + 1) * ncols + 1

        for h, head_attn_map in enumerate(attn_map):
            if not show_all:
                head_attn_map = head_attn_map[target_idx]  # [ntargets, mn]
            head_attn_map = head_attn_map.mean(dim=0)
            head_attn_map = head_attn_map - head_attn_map.min()
            head_attn_map = head_attn_map / head_attn_map.max()
            head_attn_map = head_attn_map * (1 - MIN_ALPHA) + MIN_ALPHA

            ax = fig.add_subplot(nrows, ncols, start_ax_idx + h, projection='3d')
            grouped_head_attn_map = rearrange(head_attn_map, "(n m) -> n m", m=ball_size)
            add_point_cloud(ax, samples_, tree_mask_, cmap, single_color, weights=grouped_head_attn_map, aspect_ratio=ASPECT_RATIO, hide_ax=args.hide_ax)

            if show_target:
                ax.scatter(
                    target_point[0],
                    target_point[1],
                    target_point[2],
                    color='black',
                    s=100,
                    marker='X',
                    linewidths=1.5,
                    label='Target Point'
                )
            ax.set_title(f"{layer_name} head {h}")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    args = add_args()
    main(args)
