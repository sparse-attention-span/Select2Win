from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster

import einx
from einops import einsum, rearrange, reduce, repeat

from typing import Literal, List
from dataclasses import dataclass
from native_sparse_attention_pytorch.compress_networks import GroupedMLP  # lib

from balltree import build_balltree_with_rotations

from erwin import (
    LucidRains,
    SwiGLU,
    ErwinEmbedding,
    MSATYPE,
    USE_FLEX_ATTN,
    USE_TRITON_IMPL,
    USE_GQA,
    PER_BALL,
    LUCIDRAINS_DEFAULTS,
    DBGPRINTS,
)


class SpErwinLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        dimensionality: int = 3,
        attn_kwargs: dict = {},
    ):
        super().__init__()

        self.norm_1 = nn.RMSNorm(dim)
        self.attn = LucidRains(dim, num_heads, ball_size, dimensionality, **attn_kwargs)
        self.norm_2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), SwiGLU(dim, dim * mlp_ratio))
        self.norm_3 = nn.RMSNorm(dim)

    def forward(self, node: Node) -> Node:
        x = self.norm_1(node.x)
        x = self.attn(x, node.pos)
        x = self.norm_2(node.x)
        x = self.mlp(x)
        x = self.norm_3(x)
        node.x = x

        return node


class SpErwinTransformer(nn.Module):
    """
    SpErwin Transformer.

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
        attn_kwargs: dict = {},
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_sizes) - 1

        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        self.embed = ErwinEmbedding(c_in, c_hidden, mp_steps, dimensionality)

        num_layers = len(enc_depths) - 1  # last one is a bottleneck
        num_hidden = [c_hidden] + [
            c_hidden * math.prod(strides[:i]) for i in range(1, num_layers + 1)
        ]

        print(num_layers)

        for kw, v in LUCIDRAINS_DEFAULTS.items():
            print(f"{kw}:", attn_kwargs[kw] if kw in attn_kwargs.keys() else v)

        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(
                SpErwinLayer(
                    dim=num_hidden[i],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality,
                    attn_kwargs=attn_kwargs,
                )
            )

        self.head = nn.Linear(999, 999)  # TODO: change

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

        for i, layer in enumerate(self.blocks):
            printd(f"\n    encoder {i}")
            node.tree_idx_rot = tree_idx_rot.pop(0)
            node = layer(node)

        node.x = self.head(node.x)

        return node.x, node.batch_idx
