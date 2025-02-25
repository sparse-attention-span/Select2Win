import sys
sys.path.append("../../")

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit
from erwin.models import ErwinTransformer
from erwin.datasets import ShapenetCarDataset
from erwin.experiments import ShapenetCarModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin", 
                        choices=('mpnn', 'pointtransformer', 'pointnetpp', 'erwin'))
    parser.add_argument("--data-path", type=str, default="/home/mzhdano/shapenet_car/preprocessed")
    parser.add_argument("--size", type=str, default="small", 
                        choices=('small', 'medium', 'large'))
    parser.add_argument("--num-epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-wandb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every-iter", type=int, default=100, 
                        help="Validation frequency")
    parser.add_argument("--experiment", type=str, default="shapenet", 
                        help="Experiment name in wandb")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn", type=int, default=16)
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 8,
        "c_hidden": 32,
        "ball_size": [256, 256],
        "enc_num_heads": [8, 16],
        "enc_depths": [6, 2],
        "dec_num_heads": [8],
        "dec_depths": [2],
        "strides": [2],
    },
    "medium": {
        "c_in": 8,
        "c_hidden": 64,
        "ball_size": [256, 256],
        "enc_num_heads": [8, 16],
        "enc_depths": [6, 2],
        "dec_num_heads": [8],
        "dec_depths": [2],
        "strides": [2],
    },
    "large": {
        "c_in": 8,
        "c_hidden": 96,
        "ball_size": [256, 256],
        "enc_num_heads": [8, 16],
        "enc_depths": [6, 2],
        "dec_num_heads": [8],
        "dec_depths": [2],
        "strides": [2],
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
    
    main_model = model_cls[args.model](**model_config)
    model = ShapenetCarModel(main_model).cuda()
    # model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 110, 160)