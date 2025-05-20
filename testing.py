import wandb
import torch
import time
import os
from tqdm import tqdm
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
    ExecutionTraceObserver
)
from contextlib import ExitStack
import pandas as pd

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from contextlib import ExitStack
import pandas as pd

def setup_wandb_logging(model, config, project_name="erwin-more-data"):
    wandb.init(
        project=project_name,
        config=config,
        name=config["model"] + "_" + config["experiment"],
    )
    wandb.watch(model)
    wandb.config.update(
        {
            "num_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        },
        allow_val_change=True,
    )


def save_checkpoint(model, optimizer, scheduler, config, val_loss, global_step):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "val_loss": val_loss,
        "global_step": global_step,
        "config": config,
    }

    save_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    if config["model"] in ["erwin", "pointtransformer"]:
        checkpoint_path = os.path.join(
            save_dir,
            f"{config['model']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt",
        )
    else:
        checkpoint_path = os.path.join(
            save_dir,
            f"{config['model']}_{config['experiment']}_{config['seed']}_best.pt",
        )
    torch.save(checkpoint, checkpoint_path)

    if config.get("use_wandb", False):
        wandb.log({"checkpoint/best_val_loss": val_loss}, step=global_step)



def load_checkpoint(model, optimizer, scheduler, config):
    save_dir = config.get("checkpoint_dir", "checkpoints")
    if config["model"] in ["erwin", "pointtransformer"]:
        checkpoint_path = os.path.join(
            save_dir,
            f"{config['model']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt",
        )
    else:
        checkpoint_path = os.path.join(
            save_dir,
            f"{config['model']}_{config['experiment']}_{config['seed']}_best.pt",
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["val_loss"], checkpoint["global_step"]


def train_step(model, batch, optimizer, scheduler):
    optimizer.zero_grad()
    stat_dict = model.training_step(batch)
    stat_dict["train/loss"].backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    stat_dict["train/lr"] = optimizer.param_groups[0]["lr"]
    return stat_dict


def validate(model, val_loader, config):
    model.eval()
    val_stats = {}
    num_batches = 0

    use_tqdm = not config.get("use_wandb", False)
    iterator = tqdm(val_loader, desc="Validation") if use_tqdm else val_loader

    for batch in iterator:
        batch = {k: v.cuda() for k, v in batch.items()}
        stat_dict = model.validation_step(batch)

        for k, v in stat_dict.items():
            if k not in val_stats:
                val_stats[k] = 0
            val_stats[k] += v.cpu().detach()

        if use_tqdm:
            iterator.set_postfix({"Loss": f"{stat_dict['val/loss'].item():.4f}"})

        num_batches += 1

    avg_stats = {f"avg/{k}": v / num_batches for k, v in val_stats.items()}
    return avg_stats


def fit(
    config,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader=None,
    timing_window_start=100,
    timing_window_size=500,
):
    if config.get("use_wandb", False):
        setup_wandb_logging(model, config)

    use_tqdm = not config.get("use_wandb", False)
    running_train_stats = {}
    num_train_batches = 0
    global_step = 0
    best_val_loss = float("inf")
    max_steps = config["num_epochs"]

    if test_loader is not None and config.get("test", False):
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(
            f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}"
        )

        test_stats = validate(model, test_loader, config)
        if config.get("use_wandb", False):
            wandb.log(
                {
                    **{
                        f"test/{k.replace('val/', '')}": v
                        for k, v in test_stats.items()
                    },
                    "global_step": global_step,
                },
                step=global_step,
            )
        loss_keys = [k for k in test_stats.keys() if "loss" in k]
        for k in loss_keys:
            print(f"Test {k}: {test_stats[k]:.4f}")
    return model


def convert_units(df):
    # time: µs → ms
    time_cols = [c for c in df.columns if "time_total" in c or "time_avg" in c]
    for c in time_cols:
        df[c] = df[c] / 1_000.0
    # memory: bytes → MB
    mem_cols = [c for c in df.columns if "memory_usage" in c]
    for c in mem_cols:
        df[c] = df[c] / (1024**2)

    df = df.rename(
        columns={c: c.replace("time", "time_ms") for c in time_cols}
        | {c: c.replace("memory_usage", "memory_MB") for c in mem_cols}
    )

    return df
