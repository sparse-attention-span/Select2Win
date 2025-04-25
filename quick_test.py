import torch

from models.erwin import NativelySparseBallAttention


topk_idx = torch.tensor(
    [[0, 1], [1, 3], [2, 1], [3, 2], [0, 1], [1, 3], [2, 1], [3, 2]]
)
topk_idx = topk_idx.unsqueeze(1)

attn_module = NativelySparseBallAttention(
    dim=1,
    num_heads=1,
    ball_size=2,
    dimensionality=1,
    topk=2,
)

mod_fn = attn_module.create_selection_block_mask(topk_idx, debug=True)

grid = []

for i in range(topk_idx.shape[0]):
    row = []
    for j in range(topk_idx.shape[0]):
        row.append(int(mod_fn(0, 0, i, j)))
    grid.append(row)


mod_fn = attn_module.create_selection_block_mask(topk_idx, debug=True)
print()


######################################################################################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dim = 8
n_heads = 2
dimensionality = 3
topk = 4
ball_size = 128
n_balls = 64
n_points = ball_size * n_balls

attn_module = NativelySparseBallAttention(
    dim=dim,
    num_heads=n_heads,
    ball_size=ball_size,
    dimensionality=dimensionality,
    topk=topk,
).to(device)


samples = torch.randn(n_points, dim).to(device)
pos = torch.randn(n_points, dimensionality).to(device)

res = attn_module(samples, pos)
print(res.shape)
