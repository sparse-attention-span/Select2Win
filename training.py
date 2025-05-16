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

    while global_step < max_steps:
        iterator = (
            tqdm(train_loader, desc=f"Training (step {global_step + 1}/{max_steps})")
            if use_tqdm
            else train_loader
        )

        # Enter the profiling context only if "profile" is set in the config.
        with ExitStack() as stack:
            if config.get("profile"):
                prof = stack.enter_context(
                    profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        profile_memory=True,
                        with_flops=True,
                        record_shapes=True,
                        on_trace_ready=tensorboard_trace_handler("log_dir"),
                        with_modules=True,
                    )
                )
                stack.enter_context(record_function("model_inference"))

            for batch in iterator:
                if global_step >= max_steps:
                    break

                model.train()
                batch = {k: v.cuda() for k, v in batch.items()}

                # # measure runtime statistics
                # if global_step == timing_window_start:
                #     timing_start = time.perf_counter()

                # if global_step == timing_window_start + timing_window_size:
                #     timing_end = time.perf_counter()
                #     total_time = timing_end - timing_start
                #     steps_per_second = timing_window_size / total_time
                #     if config.get("use_wandb", False):
                #         wandb.log(
                #             {"stats/steps_per_second": steps_per_second},
                #             step=global_step,
                #         )
                #     else:
                #         print(f"Steps per second: {steps_per_second:.2f}")

                stat_dict = train_step(model, batch, optimizer, scheduler)

                for k, v in stat_dict.items():
                    if "lr" not in k:
                        if k not in running_train_stats:
                            running_train_stats[k] = 0
                        running_train_stats[k] += v.cpu().detach()
                num_train_batches += 1

                if use_tqdm:
                    loss_keys = [k for k in stat_dict.keys() if "loss" in k]
                    iterator.set_postfix(
                        {
                            "step": f"{global_step + 1}/{max_steps}",
                            **{k: f"{stat_dict[k].item():.4f}" for k in loss_keys},
                        }
                    )
                else:
                    wandb.log(
                        {
                            f"{k}": v.item()
                            for k, v in stat_dict.items()
                            if "lr" not in k
                        },
                        step=global_step,
                    )

                # Validation and checkpointing
                if (global_step + 1) % config["val_every_iter"] == 0:
                    train_stats = {
                        f"avg/{k}": v / num_train_batches
                        for k, v in running_train_stats.items()
                    }

                    running_train_stats = {}
                    num_train_batches = 0

                    val_stats = validate(model, val_loader, config)
                    current_val_loss = val_stats["avg/val/loss"]

                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            config,
                            best_val_loss,
                            global_step,
                        )
                        if not config.get("use_wandb", False):
                            print(
                                f"New best validation loss: {best_val_loss:.4f}, saved checkpoint"
                            )

                    if config.get("use_wandb", False):
                        wandb.log(
                            {**train_stats, **val_stats, "global_step": global_step},
                            step=global_step,
                        )
                    else:
                        loss_keys = [k for k in val_stats.keys() if "loss" in k]
                        for k in loss_keys:
                            print(f"Validation {k}: {val_stats[k]:.4f}")

                global_step += 1

        if config.get("profile"):
            # events = prof.key_averages()
            # df = pd.DataFrame(map(vars, events))
            # df = df.sort_values(by='device_memory_usage', ascending=False)

            # print("Available columns:", list(df.columns))
            # keep = [
            #     "key",
            #     "cpu_time_total",
            #     "device_time_total",
            #     "cpu_memory_usage",
            #     "device_memory_usage"
            #     #"flops" doesn't work. need to use fvcore.
            # ]
            # df = df[keep]
            # df = convert_units(df)
            # print(df)
            print("Calculating runtime stats.....")
            print(
                prof.key_averages().table(
                    sort_by="s            else train_loaderelf_cuda_memory_usage", row_limit=10
                )
            )

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
        else:
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
