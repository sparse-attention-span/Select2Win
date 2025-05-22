import sys

sys.path.append("../../")
sys.path.append("../")


import yaml
import torch
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")


from experiments.datasets import ShapenetCarDataset
from experiments.wrappers import ShapenetCarModel
from models.erwin import AccessibleNSAMSA

from train_shapenet import parse_args, erwin_configs, model_cls


def get_attn_maps(model, sample):
    _ = model.validation_step(sample)

    attn_maps = {
        name: {"value": module.attn_cache, "ball size": module.selection_ball_size, "topk": module.topk}
        for name, module in model.named_modules()
        if isinstance(module, AccessibleNSAMSA)
    }

    return attn_maps


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    valid_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        num_workers=args.batch_size,
        persistent_workers=True,
    )

    if args.nsa_type == "":
        args.nsa_type = None

    if args.model == "erwin":
        if args.config:
            with open(f"{args.config}", "r") as f:
                model_config = dict(yaml.safe_load(f))
                print(model_config)
        else:
            attn_kwargs = {}
            attn_kwargs |= {"implementation": "pytorch"} if args.no_triton else {}
            attn_kwargs |= {"topk": args.topk} if args.topk is not None else {}
            model_config = erwin_configs[args.size] | {
                "msa_type": args.msa_type,
                "nsa_type": args.nsa_type,
                "nsa_loc": args.nsa_loc,
                "attn_kwargs": attn_kwargs,
            }
        # if args.profile:
        #     model_config = erwin_configs["profile"]
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    main_model = model_cls[args.model](**model_config)
    model = ShapenetCarModel(main_model)

    # load weights (hardcoded for now)
    checkpoint_name = "erwin_k8_medium_0_best"
    checkpoint = torch.load(f"checkpoints/{checkpoint_name}.pt", map_location="cuda:0", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()} # torch.compile shenanigans
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()


    sample = next(iter(valid_loader))
    sample = {k: v.cuda() for k, v in sample.items()}
    attn_maps = get_attn_maps(model, sample)
    print("attn keys")
    print(attn_maps.keys())
    results = {"attn_maps": attn_maps, "sample": sample, "tree_mask": model.main_model.tree_mask, "tree_idx": model.main_model.tree_idx}
    torch.save(results, f"attn_maps/attn_map_{checkpoint_name}.pt")
