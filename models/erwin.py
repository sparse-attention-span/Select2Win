from __future__ import annotations

import re
import select
from typing import Tuple
import inspect

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster

import einx
from einops import einsum, rearrange, reduce, repeat

from .native_sparse_attention import (
    SparseAttention,
    SparseAttentionMinimal,
    create_sliding_mask,
    create_fine_mask,
)  # local file
from typing import Literal, List
from dataclasses import dataclass
from native_sparse_attention_pytorch.compress_networks import GroupedMLP  # lib
from .triton_native_sparse_attention import native_sparse_attend

from balltree import build_balltree_with_rotations

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# Enable debug prints
DBGPRINTS = False

flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def parse_index_list(s, depth):
    import ast
    try:
        result = ast.literal_eval(s)
        result = [r if r>=0 else depth + 1 + r for r in result]
        if (
            isinstance(result, list) and
            all(isinstance(x, int) for x in result) and
            result == sorted([i for i in result if i >= 0])
        ):
            return result
    except (ValueError, SyntaxError):
        pass
    raise ValueError("nsa-loc must be a string representing a list of indices in order from first to last.")


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

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
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

class MiniBallAttn(nn.Module):
    """Mini Ball Attention module. """
    def __init__(
        self, ball_size: int,
    ):
        super().__init__()
        self.ball_size = ball_size
        # self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    # @torch.no_grad()
    # def create_attention_mask(self, pos: torch.Tensor):
    #     """Distance-based attention bias (eq. 10)."""
    #     pos = rearrange(pos, "(n m) d -> n m d", m=self.ball_size)
    #     return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    #attn_mask=self.create_attention_mask(pos)

    def forward(self, q, k, v):
        B = q.shape[0]
        q, k, v = tuple(rearrange(t, "B H (n m) ... -> (B n) H m ...", m=self.ball_size) for t in (q, k, v))
        x = F.scaled_dot_product_attention(
            q, k, v
        )
        x = rearrange(x, "(B n) H m ... -> B H (n m) ...", m=self.ball_size, B=B)

        return x

class LucidRains(nn.Module):
    """NSA wrapper disregarding ball structure"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int,
        per_ball: bool = False,
        use_flex_attn: bool = False,
        use_triton_impl: bool = True,
        use_miniballattn: bool = True,
        topk: int = 2,
        kv_head_factor: int = 4,
        dim_head_factor: int = 1,
        compress_stride_fraction: int = 1,
        compress_mlp_expand_factor: float = 1.0
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
            assert not use_triton_impl

        if not per_ball:
            SLIDING_WINDOW_SIZE = ball_size
            COMPRESS_BLOCK_SIZE = ball_size
            COMPRESS_BLOCK_SLIDING_STRIDE = ball_size//compress_stride_fraction
            FINE_BLOCK_SIZE = ball_size
        else:
            print("WARNING: per_ball uses hardcoded values!")
            SLIDING_WINDOW_SIZE = ball_size//8
            COMPRESS_BLOCK_SIZE = ball_size//8
            COMPRESS_BLOCK_SLIDING_STRIDE = ball_size//(8*compress_stride_fraction)
            FINE_BLOCK_SIZE = ball_size//8

        if use_miniballattn:
            sliding_window_attn = MiniBallAttn(ball_size)
        else:
            SLIDING_WINDOW_SIZE = None # fixes bug where NSA defaults to sliding window
            sliding_window_attn = None

        print(f"Ball size: {ball_size}")

        # basic dim checks
        assert dim % num_heads == 0
        assert num_heads % kv_head_factor == 0
        dim_head = dim // num_heads * dim_head_factor
        kv_heads = num_heads // kv_head_factor

        self.sparse_attn = SparseAttention(
            dim = dim,
            dim_head = dim_head,
            heads = num_heads,
            kv_heads = kv_heads,
            sliding_window_size = SLIDING_WINDOW_SIZE,
            compress_block_size = COMPRESS_BLOCK_SIZE,
            compress_block_sliding_stride = COMPRESS_BLOCK_SLIDING_STRIDE,
            # compress_mlp = GroupedMLP(
            #     dim_head = dim_head,
            #     compress_window_size = COMPRESS_BLOCK_SIZE,
            #     heads = kv_heads,
            # ),
            compress_mlp_expand_factor=compress_mlp_expand_factor,
            selection_block_size = FINE_BLOCK_SIZE,
            num_selected_blocks = topk,
            use_diff_topk = False,
            use_triton_kernel = self.use_triton_impl,
            query_heads_share_selected_kv = True,
            sliding_window_attn = sliding_window_attn
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
        # with torch.cuda.amp.autocast():
        assert not torch.isnan(x).any(), "NaN in inputs!"
        assert not torch.isinf(x).any(), "Inf in inputs!"
        assert not torch.isnan(pos).any(), "NaN in pos!"
        assert not torch.isinf(pos).any(), "Inf in pos!"
        # print(f"x has shape: {x.shape}")
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        if self.per_ball:
            x = rearrange(x, "(n m) E -> n m E", m=self.ball_size)
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

class LucidRainsMinimal(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int,
        selection_ball_size: int,
        per_ball: bool = False,
        use_flex_attn: bool = False,
        use_triton_impl: bool = True,
        use_miniballattn: bool = True,
        topk: int = 2,
        kv_head_factor: int = 4,
        dim_head_factor: int = 1,
        compress_stride_fraction: int = 1,
        compress_mlp_expand_factor: float = 1.0,
        custom_num_heads: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        if custom_num_heads is not None:
            num_heads = custom_num_heads
        self.num_heads = num_heads
        self.ball_size = selection_ball_size
        self.per_ball = per_ball
        self.use_flex_attn = use_flex_attn
        self.use_triton_impl = use_triton_impl


        if use_flex_attn:
            assert flex_attention is not None
            assert not use_triton_impl

        if not per_ball:
            SLIDING_WINDOW_SIZE = self.ball_size
            COMPRESS_BLOCK_SIZE = self.ball_size
            COMPRESS_BLOCK_SLIDING_STRIDE = self.ball_size//compress_stride_fraction
            FINE_BLOCK_SIZE = self.ball_size
        else:
            print("WARNING: per_ball uses hardcoded values!")
            SLIDING_WINDOW_SIZE = ball_size//8
            COMPRESS_BLOCK_SIZE = ball_size//8
            COMPRESS_BLOCK_SLIDING_STRIDE = ball_size//(8*compress_stride_fraction)
            FINE_BLOCK_SIZE = ball_size//8

        if use_miniballattn:
            sliding_window_attn = MiniBallAttn(ball_size)
        else:
            SLIDING_WINDOW_SIZE = None # fixes bug where NSA defaults to sliding window
            sliding_window_attn = None

        print(f"Ball size: {ball_size}, selection ball size: {self.ball_size}")

        # basic dim checks
        assert dim % num_heads == 0
        assert num_heads % kv_head_factor == 0
        dim_head = dim // num_heads * dim_head_factor
        kv_heads = num_heads // kv_head_factor

        self.sparse_attn = SparseAttentionMinimal(
            dim = dim,
            dim_head = dim_head,
            heads = num_heads,
            kv_heads = kv_heads,
            sliding_window_size = SLIDING_WINDOW_SIZE,
            compress_block_size = COMPRESS_BLOCK_SIZE,
            compress_block_sliding_stride = COMPRESS_BLOCK_SLIDING_STRIDE,
            # compress_mlp = GroupedMLP(
            #     dim_head = dim_head,
            #     compress_window_size = COMPRESS_BLOCK_SIZE,
            #     heads = kv_heads,
            # ),
            compress_mlp_expand_factor=compress_mlp_expand_factor,
            selection_block_size = FINE_BLOCK_SIZE,
            num_selected_blocks = topk,
            use_diff_topk = False,
            use_triton_kernel = self.use_triton_impl,
            query_heads_share_selected_kv = True,
            sliding_window_attn = sliding_window_attn
        )

        self.sliding_window_size = SLIDING_WINDOW_SIZE
        self.selection_block_size = FINE_BLOCK_SIZE
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.B = None

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """ Relative position of leafs wrt the center of the ball (eq. 9). """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
        # with torch.cuda.amp.autocast():
        assert not torch.isnan(x).any(), "NaN in inputs!"
        assert not torch.isinf(x).any(), "Inf in inputs!"
        assert not torch.isnan(pos).any(), "NaN in pos!"
        assert not torch.isinf(pos).any(), "Inf in pos!"
        # print(f"x has shape: {x.shape}")
        x = x + self.pe_proj(self.compute_rel_pos(pos))

        if self.per_ball:
            x = rearrange(x, "(n m) E -> n m E", m=self.ball_size)
        else:
            x = rearrange(x, "(bs num_pts) E -> bs num_pts E", bs=num_batches)
        # if not self.B:
        #     self.B = x.shape[1]
        #     printd("seq_len is", self.B)
        # elif x.shape[1] != self.B:
        #     printd(f"points changed ({self.B} -> {x.shape[1]})")
        disable_triton = not self.use_triton_impl

        seq_len = x.shape[1]

        if self.use_flex_attn and x.shape[1] == self.B:
            sliding_window_flex_mask = create_sliding_mask(seq_len, self.sliding_window_size)
            fine_selection_flex_mask = create_fine_mask(seq_len, self.selection_block_size)
            x = self.sparse_attn(
                x,
                sliding_window_flex_mask=sliding_window_flex_mask,
                fine_selection_flex_mask=fine_selection_flex_mask
            )
        else:
            x = self.sparse_attn(x, disable_triton_kernel=disable_triton)

        if self.per_ball:
            x = rearrange(x, "n m E -> (n m) E", m=self.ball_size)
        else:
            # x = x.squeeze(0)
            x = rearrange(x, "bs num_pts E -> (bs num_pts) E")

        return x

class LucidRainsMinimal_buggy(nn.Module):
    """ NSA wrapper disregarding ball structure """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int,
        per_ball: bool = False,
        use_flex_attn: bool = False,
        use_triton_impl: bool = True,
        use_miniballattn: bool = True,
        topk: int = 2,
        kv_head_factor: int = 4,
        dim_head_factor: int = 2,
        compress_stride_fraction: int = 1,
        compress_mlp_expand_factor: float = 1.0
    ):
        # keeping above args for backward compatibility
        super().__init__()


        self.num_heads = num_heads
        self.ball_size = ball_size

        print("Init minimal lucidrains")

        if use_miniballattn:
            local_attn = MiniBallAttn(ball_size)
        else:
            local_attn = None

        # basic dim checks
        assert dim % num_heads == 0
        assert num_heads % kv_head_factor == 0
        dim_head = dim // num_heads
        kv_heads = num_heads // kv_head_factor

        print(f"head dim: {dim_head}")

        self.sparse_attn = SparseAttentionMinimal(
            dim = dim,
            dim_head = dim_head,
            q_heads = num_heads,
            kv_heads = kv_heads,
            block_size=ball_size,
            topk=topk,
            local_attn = local_attn
        )

        self.pe_proj = nn.Linear(dimensionality, dim)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """ Relative position of leafs wrt the center of the ball (eq. 9). """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # with torch.cuda.amp.autocast():
        assert not torch.isnan(x).any(), "NaN in inputs!"
        assert not torch.isinf(x).any(), "Inf in inputs!"
        assert not torch.isnan(pos).any(), "NaN in pos!"
        assert not torch.isinf(pos).any(), "Inf in pos!"
        # print(f"x has shape: {x.shape}")
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        x = x.unsqueeze(0)
        x = self.sparse_attn(x)
        x = x.squeeze(0)
        return x

def sparse_attention_pytorch(q, k, v, selection_size, topk_indices):
    # tensor are of shape b h (n m) topk
    num_points = q.shape[-3] * q.shape[-2]
    k = repeat(k, "b H n m E -> b H nm n m E", nm=num_points)
    v = repeat(v, "b H n m E -> b H nm n m E", nm=num_points)

    topk_indices = repeat(
        topk_indices,
        "b H nm topk -> b H nm topk m E",
        m=selection_size,
        E=v.shape[-1],
    )

    desired_values = torch.gather(v, dim=2, index=topk_indices)
    desired_keys = torch.gather(k, dim=2, index=topk_indices)

    # Rearrange topk
    desired_keys = rearrange(desired_keys, "... n m E -> ... (n m) E")
    desired_values = rearrange(desired_values, "... n m E -> ... (n m) E")

    q = repeat(q, "b H n m E -> b H (n m) km E", km=1)

    out = F.scaled_dot_product_attention(
        q, desired_keys, desired_values, attn_mask=None)

    out = rearrange(out, "b H nm km E -> (b nm) (km H E)")

class FullAttention(nn.Module):
    """Full attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size,
        dimensionality: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)

    @torch.no_grad()
    def compute_rel_to_cloud(self, pos: torch.Tensor, num_batches: int):
        """Relative position of leafs wrt the center of the pointcloud (eq. 9)."""
        pos = rearrange(pos, "(b n) E -> b n E", b=num_batches)
        return pos - pos.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
        x = rearrange(x, "(b n) E -> b n E", b=num_batches)
        x = x + self.pe_proj(self.compute_rel_to_cloud(pos, num_batches))
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv,
            "b n (H E K) -> K b H n E",
            H=self.num_heads,
            K=3,
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n e -> (b n) (h e)")
        return self.proj(out)

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
        selection_ball_size: int = 16,
        implementation: str = "triton",
        custom_num_heads: int | None = None,
        head_dim_factor: int = 1,
        masks: bool = True,
        size=16384
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mask_block_size = ball_size
        self.selection_ball_size = selection_ball_size
        self.topk = topk
        self.scale = dim**-0.5
        self.use_diff_topk = use_diff_topk
        head_dim = dim // num_heads * head_dim_factor
        self.impl = implementation

        self.use_masks = masks
        if masks:
            diagonal_matrices = [torch.ones((self.mask_block_size, self.mask_block_size//self.selection_ball_size), dtype=torch.bool)] * self.mask_block_size
            self.attn_mask = nn.Buffer(torch.block_diag(*diagonal_matrices))

        self.qkv = nn.Linear(dim, 3 * dim * head_dim_factor)
        self.proj = nn.Linear(dim * head_dim_factor, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)

        self.selection_proj = nn.Linear(selection_ball_size * head_dim, head_dim)
        # from einops.layers.torch import Rearrange

        # self.qkv = nn.Identity()
        # self.proj = nn.Identity()
        # self.pe_proj = nn.Identity()

        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """Distance-based attention bias (eq. 10)."""
        pos = rearrange(pos, "(n m) d -> n m d", m=self.selection_ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """Relative position of leafs wrt the center of the ball (eq. 9)."""
        pos = rearrange(pos, "(n m) E -> n m E", m=self.selection_ball_size)
        # num_balls, dim = pos.shape[0] // self.selection_ball_size, pos.shape[1]
        # pos = pos.view(num_balls, self.selection_ball_size, dim)
        pos = pos - pos.mean(dim=1, keepdim=True)
        return rearrange(pos, "n m E -> (n m) E")
        # return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    @torch.no_grad()
    def compute_rel_to_cloud(self, pos: torch.Tensor, num_batches: int):
        """Relative position of leafs wrt the center of the pointcloud (eq. 9)."""
        pos = rearrange(pos, "(b n) E -> b n E", b=num_batches)
        return pos - pos.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def select_balls_mean(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = rearrange(q, "b H n m E -> b H (n m) E")
        keys = reduce(k, "b H n m E -> b H E n", "mean")
        similarity = queries @ keys
        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)
        return topk_indices

    def create_mask(self, nm, n):
        return self.attn_mask[:nm, :n]


    def select_balls_mlp(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        queries = rearrange(q, "b H n m E -> b H (n m) E")
        keys = self.selection_proj(
            rearrange(k, "b H n m E -> b H n (m E)")
        ).squeeze(-1)
        keys = rearrange(keys, "b H n E -> b H E n")
        similarity = queries @ keys # b H (n m) E @ b H E n -> b H (n m) n
        if self.use_masks:
            n = keys.shape[-1]
            nm = queries.shape[-2]
            mask = self.create_mask(nm, n)
            similarity = similarity.masked_fill(mask, float('-inf'))
            print(f"sim: {similarity}")
        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)
        print(f"topk indices: {topk_indices}")
        return topk_values, topk_indices

    def forward(self, x: torch.Tensor, pos: torch.Tensor, num_batches: int):
        # # Enable asserts to debug NaNs
        # assert not torch.isnan(x).any(), "NaN in inputs!"
        # assert not torch.isinf(x).any(), "Inf in inputs!"
        # assert not torch.isnan(pos).any(), "NaN in pos!"
        # assert not torch.isinf(pos).any(), "Inf in pos!"

        x = rearrange(x, "(b n) E -> b n E", b=num_batches)
        x = x + self.pe_proj(self.compute_rel_to_cloud(pos, num_batches))
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (n m) (H E K) -> K b H n m E",
            b=num_batches,
            H=self.num_heads,
            m=self.selection_ball_size,
            K=3,
        )

        topk_values, topk_indices = self.select_balls_mlp(q, k)
        topk_indices = topk_indices.contiguous()

        if self.impl == "pytorch":
            out = sparse_attention_pytorch(q, k, v, self.selection_ball_size, topk_indices)

        elif self.impl == "triton":
            q, k, v = tuple(rearrange(t, "b H n m E -> b H (n m) E").contiguous() for t in (q, k, v))

            # topk_values = F.pad(topk_values, (1, 0), value = -1e3)
            # topk_values = topk_values.softmax(dim = -1)
            # topk_values = topk_values[..., 1:]


            # fmask = (topk_values > 1e-10).contiguous()
            fmask = torch.ones_like(topk_values).bool()
            assert torch.all(fmask)
            out = native_sparse_attend(
                q, k, v,
                self.selection_ball_size,
                topk_indices,
                fmask,
                include_block_causal = False
            )
            out = rearrange(out, "b H nm E -> (b nm) (H E)")

        elif self.impl == "flex":
            q, k, v = tuple(rearrange(t, "b H n m E -> b H (n m) E").contiguous() for t in (q, k, v))
            fine_selection_flex_mask = create_fine_mask(q.shape[2], self.selection_ball_size)
            fine_block_mask = fine_selection_flex_mask(topk_indices)

            out = flex_attention(q, k, v, block_mask = fine_block_mask)
            out = rearrange(out, "b H nm E -> (b nm) (H E)")
        else:
            assert False # self.impl should be one of the aforementioned

        return self.proj(out)



class ErwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        msa_type: str = "BallMSA",
        dimensionality: int = 3,
        attn_kwargs: dict = {},
    ):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        # self.dropout = nn.Dropout(p=0.1)

        MSABase = {
            "BallMSA": BallMSA,
            "NSAMSA": NSAMSA,
            "LucidRains": LucidRainsMinimal,
            "FullAttention": FullAttention,
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
        msa_type: str = "BallMSA",
        dimensionality: int = 3,
        nsa_type: str | None = None,
        nsa_loc: str = "last",
        attn_kwargs: dict = {},
    ):
        super().__init__()

        if nsa_loc in ("last", "end"):
            nsa_loc = [depth]
        elif nsa_loc in ("middle", "center"):
            nsa_loc = [depth//2]
        elif nsa_loc in ("first", "begin", "start"):
            nsa_loc = [0]
        else:
            nsa_loc = parse_index_list(nsa_loc, depth)
            nsa_loc = [loc + i for i, loc in enumerate(nsa_loc)]

        # quick assertion to make sure we aren't sharing attn_kwargs
        if nsa_type is not None:
            assert msa_type == "BallMSA"

        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    dim,
                    num_heads,
                    ball_size,
                    mlp_ratio,
                    msa_type,
                    dimensionality,
                    attn_kwargs if nsa_type is None else {},
                )
                for _ in range(depth)
            ]
        )

        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        if nsa_type is not None:
            assert len(nsa_loc) > 0
            for loc in nsa_loc:
                self.blocks.insert(
                    loc,
                    ErwinTransformerBlock(
                        dim,
                        num_heads,
                        ball_size,
                        mlp_ratio,
                        nsa_type,
                        dimensionality,
                        attn_kwargs,
                    )
                )
                self.rotate.insert(loc, 0)

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
            len(self.rotate) > 1 and any(self.rotate)
        ):
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



class BasicLayerNSA(nn.Module):
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
        msa_type: str = "BallMSA",
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
                    dimensionality,
                    "BallMSA",
                    attn_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.rotate_block = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    dim,
                    num_heads,
                    ball_size,
                    mlp_ratio,
                    dimensionality,
                    msa_type,
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

        for i, (rotate, blk) in enumerate(zip(self.rotate, self.blocks)):
            printd(f"{i} ", end='')
            if rotate:
                node.x = self.rotate_block[i](node.x, node.pos)
            else:
                node.x = blk(node.x, node.pos)

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
        msa_type: str = "BallMSA",
        nsa_type: str | None = None,
        nsa_loc: str = "last",
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

        print(f"msa_type = {msa_type}")
        if nsa_type is not None:
            print(f"nsa_type = {nsa_type}")
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

        if nsa_type == "LucidRains" or msa_type == "LucidRains":
            for kw, v in get_default_args(LucidRains.__init__).items():
                print(f"{kw}:", attn_kwargs[kw] if kw in attn_kwargs.keys() else v)


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
                    nsa_type=nsa_type,
                    nsa_loc=nsa_loc,
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
            nsa_type=nsa_type,
            nsa_loc=nsa_loc,
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
                        nsa_type=nsa_type,
                        nsa_loc=nsa_loc,
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
