import torch
import torch.nn as nn


class Positional_Encoder(nn.Module):
    def __init__(self, num_features, pos_length=8):
        super().__init__()
        self.pos_start = 0
        self.pos_length = pos_length
        self.num_features = num_features

    def forward(self, pos):
        original_shape = pos.shape
        pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(self.pos_start, self.pos_start + self.pos_length, device=pos.device).float()
        freq = 2**index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding[..., : self.num_features]


class ShapenetCarModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = nn.Linear(3, main_model.in_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 1),
        )

    def forward(self, node_positions, **kwargs):
        node_features = self.embedding_model(node_positions)
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    def step(self, batch, prefix="train"):
        pred = self(**batch).squeeze(-1)
        loss = ((pred - batch["target"]) ** 2).mean()
        return {f"{prefix}/loss": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
