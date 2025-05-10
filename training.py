import wandb
import torch
import time
import os
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from contextlib import ExitStack
import pandas as pd

def setup_wandb_logging(model, config, project_name="ballformer"):
    wandb.init(project=project_name, config=config, name=config["model"] + '_' + config['msa_type'] + '_' + config["experiment"])
    wandb.watch(model)
    wandb.config.update({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, allow_val_change=True)


def save_checkpoint(model, optimizer, scheduler, config, val_loss, global_step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'val_loss': val_loss,
        'global_step': global_step,
        'config': config
    }
    
    save_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    if config['model'] in ['erwin', 'pointtransformer']:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['msa_type']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt")
    else:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['msa_type']}_{config['experiment']}_{config['seed']}_best.pt")
    torch.save(checkpoint, checkpoint_path)
    
    if config.get("use_wandb", False):
        wandb.log({"checkpoint/best_val_loss": val_loss}, step=global_step)


def load_checkpoint(model, optimizer, scheduler, config):
    save_dir = config.get('checkpoint_dir', 'checkpoints')
    if config['model'] in ['erwin', 'pointtransformer']:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['msa_type']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt")
    else:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['msa_type']}_{config['experiment']}_{config['seed']}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['val_loss'], checkpoint['global_step']


def train_step(model, batch, optimizer, scheduler):
    optimizer.zero_grad()
    stat_dict = model.training_step(batch)
    listt = []
    # for param in model.parameters():
    #     listt.append(param.isnan().any())
    # print(f"are there any NaNs? {any(listt)}")
    stat_dict["train/loss"].backward()
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    stat_dict['train/lr'] = optimizer.param_groups[0]['lr']
    return stat_dict


def validate(model, val_loader, config):
    model.eval()
    val_stats = {}
    num_batches = 0
    
    use_tqdm = not config.get("use_wandb", False)
    print(f"total number of batch items in val loader: {len(val_loader)}")
    # iterator = tqdm(val_loader, desc="Validation") if use_tqdm else val_loader

    print("About to create val iterator", flush=True)
    it = iter(val_loader)
    print("Iterator created, now fetching first batch…", flush=True)
    _ = next(it)   # blocks here until workers yield
    print("Got first validation batch!", flush=True)

    with torch.no_grad():
        for batch_num, batch in enumerate(it):
            num_batches += 1
            print(f"validation batch: {batch_num}", flush=True)
            batch_num += 1
            batch = {k: v.cuda() for k, v in batch.items()}
            stat_dict = model.validation_step(batch)
            
            for k, v in stat_dict.items():
                if k not in val_stats:
                    val_stats[k] = 0
                val_stats[k] += v.cpu().detach()
            
            # if use_tqdm:
            #     iterator.set_postfix({"Loss": f"{stat_dict['val/loss'].item():.4f}"})
    print("finished validation, calculating stats", flush=True)
    avg_stats = {f"avg/{k}": v / num_batches for k, v in val_stats.items()}
    return avg_stats

# fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, num_epochs, args.val_every_iter)

def fit(config, model, optimizer, scheduler, train_loader, val_loader, test_loader=None, timing_window_start=100, timing_window_size=500):
    if config.get("use_wandb", False):
        setup_wandb_logging(model, config)
    
    use_tqdm = not config.get("use_wandb", False)
    running_train_stats = {}
    best_val_loss = float('inf')
    max_steps = config["num_epochs"]

    print(f"use tqdm: {use_tqdm}")
    print(f"max epochs: {max_steps}")
    print(f"testing: {config.get('test', False)}")
    print(f"test loader: {test_loader}")
    num_train_batches = len(train_loader)
    print(f"Batches per epoch: {num_train_batches}")
    
    torch.autograd.set_detect_anomaly(True)
    logging_step = 0
    for epoch in range(config["num_epochs"]):
        print(f"epoch: {epoch}")
        # Enter the profiling context only if "profile" is set in the config.
        with ExitStack() as stack:
            if config.get("profile"): 
                prof = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,with_flops=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("log_dir")  ))
                stack.enter_context(record_function("model_inference"))
            
            # train 1 epoch
            for i, batch in enumerate(train_loader):
                    
                model.train()
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # measure runtime statistics
                if i == timing_window_start:
                    timing_start = time.perf_counter()
                
                if i == timing_window_start + timing_window_size:
                    timing_end = time.perf_counter()
                    total_time = timing_end - timing_start
                    steps_per_second = timing_window_size / total_time
                    if config.get("use_wandb", False):
                        wandb.log({"stats/steps_per_second": steps_per_second}, step=logging_step)
                    else:
                        print(f"Steps per second: {steps_per_second:.2f}")
                logging_step += 1
                
                stat_dict = train_step(model, batch, optimizer, scheduler)
                # print("trained")
                for k, v in stat_dict.items():
                    if "lr" not in k:
                        if k not in running_train_stats:
                            running_train_stats[k] = 0
                        running_train_stats[k] += v.cpu().detach()
                if not use_tqdm:
                    # print("logging")
                    wandb.log({f"{k}": v.item() for k, v in stat_dict.items() if "lr" not in k}, step=logging_step)
                
            # Validation and checkpointing
            print(f"Validation!")
            train_stats = {f"avg/{k}": v / num_train_batches for k, v in running_train_stats.items()}
            
            running_train_stats = {}
            print(f"running validation")
            val_stats = validate(model, val_loader, config)
            current_val_loss = val_stats['avg/val/loss']
            
            if current_val_loss < best_val_loss:
                print("saving new best model")
                best_val_loss = current_val_loss
                save_checkpoint(model, optimizer, scheduler, config, best_val_loss, logging_step)
                if not config.get("use_wandb", False):
                    print(f"New best validation loss: {best_val_loss:.4f}, saved checkpoint")
                print("done saving new best model")
            if not use_tqdm:
                print("logging to wandb")
                wandb.log({**train_stats, **val_stats, 'global_step': epoch}, step=logging_step)
            else:
                loss_keys = [k for k in val_stats.keys() if "loss" in k]
                for k in loss_keys: 
                    print(f"Validation {k}: {val_stats[k]:.4f}")
                
        if config.get("profile"):
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Test after all epochs
    print("on to testing")
    if test_loader is not None and config.get('test', False):
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}")
        
        test_stats = validate(model, test_loader, config)
        if not use_tqdm:
            wandb.log({
                **{f"test/{k.replace('val/', '')}": v for k, v in test_stats.items()},
                'global_step': config["num_epochs"]
            }, step=logging_step)
        else:
            loss_keys = [k for k in test_stats.keys() if "loss" in k]
            for k in loss_keys:
                print(f"Test {k}: {test_stats[k]:.4f}")
    print("out of fit loop")
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
    columns={c: c.replace("time", "time_ms") for c in time_cols} |
            {c: c.replace("memory_usage", "memory_MB") for c in mem_cols})
    
    return df
