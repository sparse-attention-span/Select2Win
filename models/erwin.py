from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster

import einx
from einops import einsum, rearrange, reduce, repeat

from .native_sparse_attention import (
    SparseAttention,
    create_sliding_mask,
    create_fine_mask,
)  # local file
from typing import Literal, List
from dataclasses import dataclass
from native_sparse_attention_pytorch.compress_networks import GroupedMLP  # lib

from balltree import build_balltree_with_rotations

# Enable debug prints
DBGPRINTS = False

flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass


def print_memory_usage():
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024**2
    print(mem_used_MB)


def printd(*args, **kwargs):
    if DBGPRINTS:
        print(*args, **kwargs)


def straight_through(t, target):
    return t + (target - t).detach()


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """
    Averages all values from src into the receivers at the indices specified by idx.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).

    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    result = result / count.unsqueeze(1).clamp(min=1)
    return result


class SwiGLU(nn.Module):
    """W_3 SiLU(W_1 x) ⊗ W_2 x"""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update

    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.message_fns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * dim + dimensionality, dim),
                    nn.GELU(),
                    nn.LayerNorm(dim),
                )
                for _ in range(mp_steps)
            ]
        )

        self.update_fns = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
                for _ in range(mp_steps)
            ]
        )

    def layer(
        self,
        message_fn: nn.Module,
        update_fn: nn.Module,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        row, col = edge_index
        messages = message_fn(torch.cat([h[row], h[col], edge_attr], dim=-1))
        message = scatter_mean(messages, col, h.size(0))
        update = update_fn(torch.cat([h, message], dim=-1))
        return h + update

    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        return pos[edge_index[0]] - pos[edge_index[1]]

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_attr = pos[edge_index[0]] - pos[edge_index[1]]
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


class ErwinEmbedding(nn.Module):
    """Linear projection -> MPNN."""

    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.mp_steps = mp_steps
        self.embed_fn = nn.Linear(in_dim, dim)
        self.mpnn = MPNN(dim, mp_steps, dimensionality)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        x = self.embed_fn(x)
        return self.mpnn(x, pos, edge_index) if self.mp_steps > 0 else x


@dataclass
class Node:
    """Dataclass to store the hierarchical node information."""

    x: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None


class BallPooling(nn.Module):
    """
    Pooling of leaf nodes in a ball (eq. 12):
        1. select balls of size 'stride'.
        2. concatenate leaf nodes inside each ball along with their relative positions to the ball center.
        3. apply linear projection and batch normalization.
        4. the output is the center of each ball endowed with the pooled features.
    """

    def __init__(self, dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        input_dim = stride * dim + stride * dimensionality
        self.proj = nn.Linear(input_dim, stride * dim)
        self.norm = nn.BatchNorm1d(stride * dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1:  # no pooling
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        with torch.no_grad():
            batch_idx = node.batch_idx[:: self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)
            pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = rearrange(pos - centers[:, None], "n s d -> n (s d)")

        x = torch.cat(
            [rearrange(node.x, "(n s) c -> n (s c)", s=self.stride), rel_pos], dim=1
        )
        x = self.norm(self.proj(x))

        batch_idx = node.batch_idx[:: self.stride].contiguous()
        return Node(x=x, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    """
    Ball unpooling (refinement; eq. 13):
        1. compute relative positions of children (from before pooling) to the center of the ball.
        2. concatenate the pooled features with the relative positions.
        3. apply linear projection and self-connection followed by batch normalization.
        4. the output is a refined tree with the same number of nodes as before pooling.
    """

    def __init__(self, dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        input_dim = stride * dim + stride * dimensionality
        self.proj = nn.Linear(input_dim, stride * dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            rel_pos = (
                rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
                - node.pos[:, None]
            )
            rel_pos = rearrange(rel_pos, "n m d -> n (m d)")

        x = torch.cat([node.x, rel_pos], dim=-1)
        node.children.x = self.norm(
            node.children.x
            + rearrange(self.proj(x), "n (m d) -> (n m) d", m=self.stride)
        )

        return node.children


class BallMSA(nn.Module):
    """Ball Multi-Head Self-Attention (BMSA) module (eq. 8)."""

    def __init__(
        self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """Distance-based attention bias (eq. 10)."""
        pos = rearrange(pos, "(n m) d -> n m d", m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """Relative position of leafs wrt the center of the ball (eq. 9)."""
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        q, k, v = rearrange(
            self.qkv(x),
            "(n m) (H E K) -> K n H m E",
            H=self.num_heads,
            m=self.ball_size,
            K=3,
        )
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.create_attention_mask(pos)
        )
        x = rearrange(x, "n H m E -> (n m) (H E)", H=self.num_heads, m=self.ball_size)
        return self.proj(x)


class LucidRains(nn.Module):
    """NSA wrapper disregarding ball structure"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        per_ball: bool,
        use_flex_attn: bool,
        use_triton_impl: bool,
        use_gqa: bool,
        dimensionality: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.per_ball = per_ball
        self.use_flex_attn = use_flex_attn
        self.use_triton_impl = use_triton_impl
        if use_flex_attn:
            assert flex_attention is not None

        if not per_ball:
            SLIDING_WINDOW_SIZE = ball_size
            COMPRESS_BLOCK_SIZE = ball_size
            COMPRESS_BLOCK_SLIDING_STRIDE = ball_size // 2

            FINE_BLOCK_SIZE = ball_size
            NUM_FINE_SELECTED = 2
        else:
            SLIDING_WINDOW_SIZE = ball_size // 8
            COMPRESS_BLOCK_SIZE = ball_size // 8
            COMPRESS_BLOCK_SLIDING_STRIDE = ball_size // 16

            FINE_BLOCK_SIZE = ball_size // 8
            NUM_FINE_SELECTED = 1

        kv_heads = num_heads // 4 if use_gqa else num_heads

        dim_head = dim // num_heads * 2

        assert dim % num_heads == 0
        assert not (use_triton_impl and use_flex_attn)

        self.sparse_attn = SparseAttention(
            dim=dim,
            dim_head=dim_head,
            heads=num_heads,
            kv_heads=kv_heads,
            sliding_window_size=SLIDING_WINDOW_SIZE,
            compress_block_size=COMPRESS_BLOCK_SIZE,
            compress_block_sliding_stride=COMPRESS_BLOCK_SLIDING_STRIDE,
            compress_mlp=GroupedMLP(
                dim_head=dim_head,
                compress_window_size=COMPRESS_BLOCK_SIZE,
                heads=kv_heads,
            ),
            selection_block_size=FINE_BLOCK_SIZE,
            num_selected_blocks=NUM_FINE_SELECTED,
            use_diff_topk=False,
            use_triton_kernel=self.use_triton_impl,
            query_heads_share_selected_kv=True,
        )

        self.sliding_window_size = SLIDING_WINDOW_SIZE
        self.selection_block_size = FINE_BLOCK_SIZE
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.B = None

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """Relative position of leafs wrt the center of the ball (eq. 9)."""
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        if self.per_ball:
            x = rearrange(
                x, "(n m) E -> n m E", m=self.ball_size
            )  # Batch balls instead of computing global attn
        else:
            x = x.unsqueeze(0)
        if not self.B:
            self.B = x.shape[1]
            printd("seq_len is", self.B)
        elif x.shape[1] != self.B:
            printd(f"points changed ({self.B} -> {x.shape[1]})")
        disable_triton = not self.use_triton_impl

        seq_len = x.shape[1]

        if self.use_flex_attn and x.shape[1] == self.B:
            sliding_window_flex_mask = create_sliding_mask(
                seq_len, self.sliding_window_size
            )
            fine_selection_flex_mask = create_fine_mask(
                seq_len, self.selection_block_size
            )
            x = self.sparse_attn(
                x,
                sliding_window_flex_mask=sliding_window_flex_mask,
                fine_selection_flex_mask=fine_selection_flex_mask,
            )
        else:
            x = self.sparse_attn(x, disable_triton_kernel=disable_triton)

        if self.per_ball:
            x = rearrange(x, "n m E -> (n m) E", m=self.ball_size)
        else:
            x = x.squeeze(0)

        return x


class NSAMSA(nn.Module):
    """Ball Multi-Head Self-Attention (BMSA) module (eq. 8)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int = 3,
        topk: int = 2,
        use_diff_topk: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.topk = topk
        self.scale = dim**-0.5
        self.use_diff_topk = use_diff_topk

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)

        from einops.layers.torch import Rearrange

        # self.qkv = nn.Identity()
        # self.proj = nn.Identity()
        # self.pe_proj = nn.Identity()

        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """Distance-based attention bias (eq. 10)."""
        pos = rearrange(pos, "(n m) d -> n m d", m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """Relative position of leafs wrt the center of the ball (eq. 9)."""
        pos = rearrange(pos, "(n m) E -> n m E", m=self.ball_size)
        # num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        # pos = pos.view(num_balls, self.ball_size, dim)
        pos = pos - pos.mean(dim=1, keepdim=True)
        return rearrange(pos, "n m E -> (n m) E")
        # return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    @torch.no_grad()
    def select_balls(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = rearrange(q, "b H n m E -> b H (n m) E")
        keys = reduce(k, "b H n m E -> b H E n", "mean")
        similarity = queries @ keys
        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)
        return topk_indices

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        qkv = self.qkv(x)
        q, k, v = repeat(
            qkv,
            "(b n m) (H E K) -> K b H n m E",
            b=num_batches,
            H=self.num_heads,
            m=self.ball_size,
            K=3,
        )
        # tensor are of shape b h (n m) topk
        num_points = q.shape[-2] * q.shape[-1]
        topk_indices = self.select_balls(q, k)

        # print(topk_indices[0, 0, 0])

        # gates = straight_through(topk_values, 1.0) if self.use_diff_topk else None
        
        k = rearrange(k, "b n H m E -> b H n m E")
        v = rearrange(v, "b n H m E -> b H n m E")
        
        # # out = torch.zeros_like(v)
        # out = rearrange(out, "b H n m E -> b H nm E")
        
        # for t in range(num_points):
        #     idx = topk_indices[:, :, t, :].reshape(-1)
        #     k_t = k[:, :, idx, :]
        #     v_t = v[:, :, idx, :]

        k = repeat(k, "b H n m E -> b H nm n m E", nm=num_points)
        v = repeat(v, "b H n m E -> b H nm n m E", nm=num_points)

        topk_indices = repeat(
            topk_indices,
            "b H nm topk -> b H nm topk m E",
            m=self.ball_size,
            E=v.shape[-1],
        )

        desired_values = torch.gather(v, dim=2, index=topk_indices)
        desired_keys = torch.gather(k, dim=2, index=topk_indices)

        # Rearrange topk
        desired_keys = rearrange(desired_keys, "... n m E -> ... (n m) E")
        desired_values = rearrange(desired_values, "... n m E -> ... (n m) E")
        
        q = rearrange(q, "b H n m E -> b H (n m) E")
        
        attn = torch.softmax(einsum(q, desired_keys, "b H nm E, b H nm km E -> b H nm km") / (q.shape[-1] ** 0.5), dim=-1)
        out = einsum(attn, desired_values, "b H nm km, b H nm km E -> b H nm E")
        out = rearrange(out, "b H nm E -> (b nm) (H E)")
        # Rearrange k into (n m)
        # TODO NEEDS POSITION BIAS MASK
        return self.proj(out)


class ErwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        msa_type: str,
        dimensionality: int = 3,
        attn_kwargs: dict = {},
    ):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        MSABase = {
            "BallMSA": BallMSA,
            "NSAMSA": NSAMSA,
            "LucidRains": LucidRains,
        }[msa_type]

        self.BMSA = MSABase(
            dim, num_heads, ball_size, dimensionality=dimensionality, **attn_kwargs
        )
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
        x = x + self.BMSA(self.norm1(x), pos, num_batches)
        return x + self.swiglu(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        direction: Literal[
            "down", "up", None
        ],  # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        rotate: bool,
        msa_type: str,
        dimensionality: int = 3,
        attn_kwargs: dict = {},
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    dim,
                    num_heads,
                    ball_size,
                    mlp_ratio,
                    msa_type,
                    dimensionality,
                    attn_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == "down" and stride is not None:
            self.pool = BallPooling(dim, stride, dimensionality)
        elif direction == "up" and stride is not None:
            self.unpool = BallUnpooling(dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        printd("Erwin transformer blocks:")
        node = self.unpool(node)

        if (
            len(self.rotate) > 1 and self.rotate[1]
        ):  # if rotation is enabled, it will be used in the second block
            assert (
                node.tree_idx_rot is not None
            ), "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(
                node.tree_idx_rot
            )  # map from rotated to original

        num_batches = node.batch_idx.max() + 1
        assert num_batches > 0

        for i, (rotate, blk) in enumerate(zip(self.rotate, self.blocks)):
            printd(f"{i} ", end="")
            if rotate:
                node.x = blk(node.x[node.tree_idx_rot], node.pos[node.tree_idx_rot], num_batches)[
                    tree_idx_rot_inv
                ]
            else:
                node.x = blk(node.x, node.pos, num_batches)
        return self.pool(node)


class FullTransformer(nn.Module):
    """
    Runs full attention between all points.

    Args:

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ErwinTransformer(nn.Module):
    """
    Erwin Transformer.

    Args:
        c_in (int): number of input channels.
        c_hidden (int): number of hidden channels. With every layer, the number of channels is multiplied by stride.
        ball_size (List): list of ball sizes for each encoder layer (reverse for decoder).
        enc_num_heads (List): list of number of heads for each encoder layer.
        enc_depths (List): list of number of ErwinTransformerBlock layers for each encoder layer.
        dec_num_heads (List): list of number of heads for each decoder layer.
        dec_depths (List): list of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List): list of strides for each encoder layer (reverse for decoder).
        rotate (int): angle of rotation for cross-ball interactions; if 0, no rotation.
        decode (bool): whether to decode or not. If not, returns latent representation at the coarsest level.
        mlp_ratio (int): ratio of SWIGLU's hidden dim to a layer's hidden dim.
        dimensionality (int): dimensionality of the input data.
        mp_steps (int): number of message passing steps in the MPNN Embedding.

    Notes:
        - lengths of ball_size, enc_num_heads, enc_depths must be the same N (as it includes encoder and bottleneck).
        - lengths of strides, dec_num_heads, dec_depths must be N - 1.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        ball_sizes: List,
        enc_num_heads: List,
        enc_depths: List,
        dec_num_heads: List,
        dec_depths: List,
        strides: List,
        rotate: int,
        msa_type: str,
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
        mp_steps: int = 3,
        attn_kwargs: dict = {},
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_sizes) - 1

        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        self.embed = ErwinEmbedding(c_in, c_hidden, mp_steps, dimensionality)

        num_layers = len(enc_depths) - 1  # last one is a bottleneck
        num_hidden = [c_hidden] + [
            c_hidden * math.prod(strides[:i]) for i in range(1, num_layers + 1)
        ]

        print(num_layers)

        print(f"using {msa_type}")

        if msa_type == "LucidRains":
            print(attn_kwargs)

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                BasicLayer(
                    direction="down",
                    depth=enc_depths[i],
                    stride=strides[i],
                    dim=num_hidden[i],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    rotate=rotate > 0,
                    mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality,
                    msa_type=msa_type,
                    attn_kwargs=attn_kwargs,
                )
            )

        self.bottleneck = BasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            dim=num_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            rotate=rotate > 0,
            mlp_ratio=mlp_ratio,
            dimensionality=dimensionality,
            msa_type=msa_type,
            attn_kwargs=attn_kwargs,
        )

        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction="up",
                        depth=dec_depths[i],
                        stride=strides[i],
                        dim=num_hidden[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],
                        rotate=rotate > 0,
                        mlp_ratio=mlp_ratio,
                        dimensionality=dimensionality,
                        msa_type=msa_type,
                        attn_kwargs=attn_kwargs,
                    )
                )

        self.in_dim = c_in
        self.out_dim = c_hidden if decode else num_hidden[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        node_features: torch.Tensor,
        node_positions: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: torch.Tensor = None,
        tree_idx: torch.Tensor = None,
        tree_mask: torch.Tensor = None,
        radius: float = None,
        **kwargs,
    ):
        with torch.no_grad():
            # if not given, build the ball tree and radius graph
            if tree_idx is None and tree_mask is None:
                tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(
                    node_positions,
                    batch_idx,
                    self.strides,
                    self.ball_sizes,
                    self.rotate,
                )
            if edge_index is None and self.embed.mp_steps:
                assert (
                    radius is not None
                ), "radius (float) must be provided if edge_index is not given to build radius graph"
                edge_index = torch_cluster.radius_graph(
                    node_positions, radius, batch=batch_idx, loop=True
                )

        x = self.embed(node_features, node_positions, edge_index)

        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None,  # will be populated in the encoder
        )

        for i, layer in enumerate(self.encoder):
            printd(f"\n    encoder {i}")
            node.tree_idx_rot = tree_idx_rot.pop(0)
            node = layer(node)

        node.tree_idx_rot = tree_idx_rot.pop(0)
        printd(f"\n    bottleneck")
        node = self.bottleneck(node)

        if self.decode:
            for i, layer in enumerate(self.decoder):
                printd(f"\n    decoder {i}")
                node = layer(node)
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]

        return node.x, node.batch_idx
