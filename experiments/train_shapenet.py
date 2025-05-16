import sys

sys.path.append("../../")
sys.path.append("../")

import argparse
import torch

torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit
from erwin.models.erwin import ErwinTransformer
from erwin.experiments.datasets import ShapenetCarDataset
from erwin.experiments.wrappers import ShapenetCarModel
import time
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="erwin",
        choices=("mpnn", "pointtransformer", "pointnetpp", "erwin"),
    )
    parser.add_argument("--data-path", type=str, default="../shapenet_car/preprocessed")
    parser.add_argument("--size", type=str, default="small", 
                        choices=('small', 'medium', 'large'))
    parser.add_argument("--num-epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-wandb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every-iter", type=int, default=100,
                        help="Validation frequency")
    parser.add_argument("--experiment", type=str, default="shapenet",
                        help="Experiment name in wandb")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use minimal profile configuration for testing",
    )
    parser.add_argument("--msa-type", type=str, default="BallMSA",
                        choices=["BallMSA", "NSAMSA", "LucidRains"])

    parser.add_argument("--lucidrains-per-ball", action='store_true')
    parser.add_argument("--lucidrains-gqa", action='store_true')
    parser.add_argument("--lucidrains-triton-kernel", action='store_true')
    parser.add_argument("--lucidrains-flex-attn", action='store_true')

    parser.add_argument("--nsamsa-use-diff-topk", action='store_true')

    return parser.parse_args()

def get_attn_kwargs(args):
    if args.msa_type == "LucidRains":
        kwargs =  {
            "per_ball": args.lucidrains_per_ball,
            "use_flex_attn": args.lucidrains_flex_attn,
            "use_triton_impl": args.lucidrains_triton_kernel,
            "use_gqa": args.lucidrains_gqa,
        }
    elif args.msa_type == "NSAMSA":
        kwargs = {
            "use_diff_topk": args.nsamsa_use_diff_topk
        }
    else:
        kwargs = {}

    return kwargs

erwin_configs = {
    "profile": {
        "c_in": 64,
        "c_hidden": 64,
        "ball_sizes": [256,],
        "enc_num_heads": [8,],
        "enc_depths": [1,],
        "dec_num_heads": [],
        "dec_depths": [],
        "strides": [],
        "rotate": 0,
        "mp_steps": 3,
        "msa_type": ""
    },
    "small": {
        "c_in": 64,
        "c_hidden": 64,
        "ball_sizes": [4, 4],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
        "msa_type": ""
    },
    "medium": {
        "c_in": 64,
        "c_hidden": 128,
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
        "msa_type": ""
    },
    "large": {
        "c_in": 64,
        "c_hidden": 256,
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
        "msa_type": ""
    },
}

model_cls = {
    "erwin": ErwinTransformer,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="train",
        knn=args.knn,
    )

    valid_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    test_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    model_config["msa_type"] = args.msa_type

    main_model = model_cls[args.model](**model_config, attn_kwargs=get_attn_kwargs(args))
    model = ShapenetCarModel(main_model).cuda()
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=5e-5)

    config = vars(args)
    config.update(model_config)
    num_epochs = args.num_epochs

    if args.profile:
        gc.collect()
    # Run the training
    fit(
        config,
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs,
        args.val_every_iter,
    )
