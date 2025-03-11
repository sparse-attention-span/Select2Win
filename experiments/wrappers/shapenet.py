import torch
import torch.nn as nn


class ShapenetCarModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 1),
        )

        self.y_std_denorm = 48.5743

    def forward(self, node_positions, **kwargs):
        return self.pred_head(self.main_model(node_positions, node_positions, **kwargs))

    @torch.no_grad()
    def denorm_mse(self, pred, target):
        pred = pred * self.y_std_denorm
        target = target * self.y_std_denorm
        return ((pred - target) ** 2).mean()

    def step(self, batch, prefix="train"):
        pred = self(**batch).squeeze(-1)
        loss = ((pred - batch["target"]) ** 2).mean()
        mse = self.denorm_mse(pred, batch["target"])
        return {f"{prefix}/loss": loss, f"{prefix}/mse": mse}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
