import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange
import numpy as np

from attn_maps import add_args, add_point_cloud, rotate_around_x, BALL_SIZE, ASPECT_RATIO, ROTATE_ANGLE, MIN_ALPHA
plt.rcParams.update({'font.size': 14})  #

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

    fig = plt.figure(figsize=(20, 20))
    nrows = 1
    ncols = 4
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

        start_ax_idx = 2 + i

        h = 0
        head_attn_map = attn_map[h]
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
        ax.set_title(f"{layer_name.split('.')[1].capitalize()}")

    # plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Saved figure to {args.output}")



if __name__ == "__main__":
    args = add_args()
    main(args)
