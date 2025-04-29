from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster

import einx
from einops import einsum, rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from typing import Literal, List
from dataclasses import dataclass

from balltree import build_balltree_with_rotations
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    create_block_mask,
)


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
    return result / count.unsqueeze(1).clamp(min=1)


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
        edge_attr = self.compute_edge_attr(pos, edge_index)
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


class NativelySparseBallAttention(nn.Module):
    """Ball attention based on NSA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int = 3,
        topk: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.topk = topk

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))
        self.flex_attn_reshape = Rearrange("n H m E -> 1 H (n m) E")

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

    @torch.no_grad()
    def create_selection_block_mask(self, idx: torch.Tensor) -> BlockMask:
        """
        Creates block mask for sparse FlexAttention

        Arguments:
            idx: Tensor of shape (n m) H topk containing topk ball indices

        Returns:
            Block mask for corresponding points for FlexAttention
        """
        num_points = idx.shape[0]
        num_balls = num_points // self.ball_size
        device = idx.device

        # create helper matrix for indices
        # shape: (n m) H n
        one_hot_selected_block_indices = torch.zeros(
            (*idx.shape[:-1], num_balls), dtype=torch.bool, device=device
        )
        one_hot_selected_block_indices.scatter_(-1, idx, True)

        def nsa_mask_mod(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            """Creates sparse attention mask NSA style"""
            kv_ball_idx = kv_idx // self.ball_size
            is_selected = one_hot_selected_block_indices[q_idx, h, kv_ball_idx]
            return is_selected

        # def nsa_mask_mod(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        #     """Ball attention"""
        #     q_ball_idx = q_idx // self.ball_size
        #     kv_ball_idx = kv_idx // self.ball_size
        #     same_ball = q_ball_idx == kv_ball_idx
        #     return same_ball

        block_mask = create_block_mask(
            nsa_mask_mod,
            B=1,
            H=self.num_heads,
            Q_LEN=num_points,
            KV_LEN=num_points,
            BLOCK_SIZE=self.ball_size,
            device=device,
            _compile=True,
        )

        return block_mask

    @torch.no_grad()
    def get_topk_idx(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Get topk indices of keys with highest similarity

        Arguments:
            q: Tensor with shape n H m E
            k: Tensor with same shape as q

        Returns:
            Tensor of shape: (n m) H topk
        """
        queries = rearrange(q, "n H m E -> H (n m) E")
        keys_center = reduce(k, "n H m E -> H E n", "mean")
        similarity = queries @ keys_center  # H (n m) n
        _, topk_idx = torch.topk(similarity, self.topk, dim=-1)  # H (n m) topk
        topk_idx = rearrange(topk_idx, "H nm topk -> nm H topk")
        return topk_idx

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of NSAMSA

        Arguments:
            x: tensor of shape (n m) d, where n = #balls, m = ball size, d is dim
        """
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        q, k, v = rearrange(
            self.qkv(x),
            "(n m) (H E K) -> K n H m E",
            H=self.num_heads,
            m=self.ball_size,
            K=3,
        )

        # create selection mask
        topk_idx = self.get_topk_idx(q, k)
        attn_block_mask = self.create_selection_block_mask(topk_idx)

        # compute attention
        q = self.flex_attn_reshape(q).contiguous()
        k = self.flex_attn_reshape(q).contiguous()
        v = self.flex_attn_reshape(q).contiguous()
        attn = flex_attention(q, k, v, block_mask=attn_block_mask)

        # projection onto correct dimension
        attn = rearrange(attn, "1 H nm E -> nm (H E)")
        out = self.proj(attn)

        if debug:
            return out, {}

        return out


class tempIdfn(nn.Module):
    """This class just mimicks QKV proj layer where the transformation is the id map"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = repeat(x, "nm E -> nm E K", K=3)
        out = rearrange(out, "nm E K -> nm (E K)")
        return out


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
        if torch.cuda.is_available():
            flex_attention = torch.compile(flex_attention)

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.topk = topk
        self.scale = dim**-0.5
        self.use_diff_topk = use_diff_topk

        # self.qkv = nn.Linear(dim, 3 * dim)
        # self.proj = nn.Linear(dim, dim)
        # self.pe_proj = nn.Linear(dimensionality, dim)
        self.qkv = tempIdfn()
        self.proj = nn.Identity()
        self.pe_proj = nn.Identity()

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
        return (pos - pos.mean(dim=1, keepdim=True)).contiguous().view(-1, dim)

    # @torch.no_grad()
    def select_balls(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = rearrange(q, "H n m E -> H (n m) E")
        keys = reduce(k, "H n m E -> H E n", "mean")
        similarity = torch.softmax(queries @ keys * self.scale, dim=-1)
        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)
        return topk_values, topk_indices

    def forward(self, x: torch.Tensor, pos: torch.Tensor, debug: bool = False):
        if debug:
            debug_data = {"ball_size": self.ball_size}

        # add positional encoding
        pos_emb = self.pe_proj(self.compute_rel_pos(pos))
        x = x + pos_emb

        # get qkv matrices
        q, k, v = repeat(
            self.qkv(x),
            "(n m) (H E K) -> K H n m E",
            H=self.num_heads,
            m=self.ball_size,
            K=3,
        )

        # get topk balls
        num_points = q.shape[1] * q.shape[2]
        topk_values, topk_indices = self.select_balls(q, k)

        if debug:
            debug_data["x_with_emb"] = x.detach().clone()
            debug_data["topk_idx"] = topk_indices.detach().clone()

        # gather all points in topk balls
        topk_indices = repeat(
            topk_indices,
            "H nm topk -> H nm topk m E",
            m=self.ball_size,
            E=v.shape[-1],
        )
        k = repeat(k, "H n m E -> H nm n m E", nm=num_points)
        v = repeat(v, "H n m E -> H nm n m E", nm=num_points)
        k = k.gather(2, topk_indices)
        v = v.gather(2, topk_indices)

        if debug:
            debug_data["k_after_gather"] = k.detach().clone()
            debug_data["v_after_gather"] = v.detach().clone()

        if self.use_diff_topk:
            gates = straight_through(topk_values, 1.0)
            k = einx.multiply("H nm topk, H nm topk j E -> H nm topk j E", gates, k)

        # TODO NEEDS POSITION BIAS MASK
        # compute attention
        q = rearrange(q, "H n m E -> (n m) H 1 E")
        k = rearrange(k, "H nm w j E -> nm H (w j) E")
        v = rearrange(v, "H nm w j E -> nm H (w j) E")
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False).squeeze()

        out = self.proj(out)

        if debug:
            return out, debug_data

        return out


class ErwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        dimensionality: int = 3,
    ):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = NSAMSA(
            dim, num_heads, ball_size, dimensionality, use_diff_topk=True
        )
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
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
        dimensionality: int = 3,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    dim, num_heads, ball_size, mlp_ratio, dimensionality
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

        for rotate, blk in zip(self.rotate, self.blocks):
            if rotate:
                node.x = blk(node.x[node.tree_idx_rot], node.pos[node.tree_idx_rot])[
                    tree_idx_rot_inv
                ]
            else:
                node.x = blk(node.x, node.pos)
        return self.pool(node)


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
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
        mp_steps: int = 3,
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

        for layer in self.encoder:
            node.tree_idx_rot = tree_idx_rot.pop(0)
            node = layer(node)

        node.tree_idx_rot = tree_idx_rot.pop(0)
        node = self.bottleneck(node)

        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]

        return node.x, node.batch_idx
