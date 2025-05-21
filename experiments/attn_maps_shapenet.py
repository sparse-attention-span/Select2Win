import sys

sys.path.append("../../")
sys.path.append("../")


import yaml
import torch

torch.set_float32_matmul_precision("high")


from erwin.experiments.datasets import ShapenetCarDataset
from erwin.experiments.wrappers import ShapenetCarModel
from erwin.models.erwin import AccessibleNSAMSA

from train_shapenet import parse_args, erwin_configs, model_cls


def get_attn_maps(model, sample):
    _ = model(sample)

    attn_maps = {
        name: module.attn_cache
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

    sample = valid_dataset[0]
    print(f"Sample shape: {sample.shape}")
    main_model = model_cls[args.model](**model_config)
    model = ShapenetCarModel(main_model)

    # load weights (hardcoded for now)
    checkpoint = torch.load("checkpoints/.pt", map_location="cuda:0", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.cuda()
    attn_maps = get_attn_maps(model, sample)
    print(attn_maps.keys())
