import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange

path = "../experiments/attn_maps/attn_map_erwin_k8_medium_0_best.pt"
data = torch.load(path, map_location="cpu", weights_only=True)
tree_mask = data["tree_mask"]
tree_idx = data["tree_idx"]
attn_maps = data["attn_maps"]
samples = data["sample"]["node_positions"]
samples= samples[tree_idx]

inverse_perm = torch.argsort(tree_idx[tree_mask])

target_idx = 0


fig = plt.figure(figsize=(50, 10))
nrows = len(attn_maps)
ncols = max(mp["value"].shape[1] for mp in attn_maps.values())


# plot ball tree
ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
samples_ = rearrange(samples, "(n m) d -> n m d", m=32) # todo: adapt for each layer
tree_mask_ = rearrange(tree_mask, "(n m)-> n m", m=32) # todo: adapt for each layer

for group, tree_mask__ in zip(samples_, tree_mask_):
    group_samples = group[tree_mask__]
    ax.scatter(group_samples[:, 0], group_samples[:, 1], group_samples[:, 2])

# plot attention maps
target_point = samples[target_idx]

for i, (layer_name, attn_map_data) in enumerate(attn_maps.items()):
    attn_map = attn_map_data["value"][0] # assume batch size 1
    ball_size = attn_map_data["ball size"]

    for h, head_attn_map in enumerate(attn_map):
        head_attn_map = head_attn_map[target_idx]
        head_attn_map = head_attn_map - head_attn_map.min()
        head_attn_map = head_attn_map / head_attn_map.max()
        head_attn_map = head_attn_map * 0.99 + 0.01

        ax = fig.add_subplot(nrows, ncols, i + 2 + h, projection='3d')
        grouped_head_attn_map = rearrange(head_attn_map, "(n m) -> n m", m=ball_size)

        # plot per group to get colors
        for group_samples, group_head_attn_map, group_mask in zip(samples_, grouped_head_attn_map, tree_mask_):
            group_head_attn_map = group_head_attn_map[group_mask]
            group_samples = group_samples[group_mask]

            ax.scatter(group_samples[:, 0], group_samples[:, 1], group_samples[:, 2], alpha=group_head_attn_map)
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

        break

    break


plt.savefig("attn_maps.png")