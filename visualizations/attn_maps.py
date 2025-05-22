import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange


def main(args):
    path = args.input_path
    data = torch.load(path, map_location="cpu", weights_only=True)
    tree_mask = data["tree_mask"]
    tree_idx = data["tree_idx"]
    attn_maps = data["attn_maps"]
    samples = data["sample"]["node_positions"]
    samples = samples[tree_idx]

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
    samples_ = rearrange(samples, "(n m) d -> n m d", m=32)  # todo: adapt for each layer
    tree_mask_ = rearrange(tree_mask, "(n m)-> n m", m=32)  # todo: adapt for each layer

    for i, (group, tree_mask__) in enumerate(zip(samples_, tree_mask_)):
        group_samples = group[tree_mask__]
        ax.scatter(group_samples[:, 0], group_samples[:, 1], group_samples[:, 2], color=cmap(i % 20 / 20))

    if args.hide_ax:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # plot attention maps
    if show_target:
        target_point = samples[target_idx].squeeze()

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
            head_attn_map = head_attn_map * 0.99 + 0.01

            ax = fig.add_subplot(nrows, ncols, start_ax_idx + h, projection='3d')
            grouped_head_attn_map = rearrange(head_attn_map, "(n m) -> n m", m=ball_size)

            # plot per group to get colors
            for j, (group_samples, group_head_attn_map, group_mask) in enumerate(zip(samples_, grouped_head_attn_map, tree_mask_)):
                group_head_attn_map = group_head_attn_map[group_mask]
                group_samples = group_samples[group_mask]

                ax.scatter(group_samples[:, 0], group_samples[:, 1], group_samples[:, 2], alpha=group_head_attn_map,
                           color=cmap((j % 20 / 20) * single_color))

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

            if args.hide_ax:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
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
    main(args)
