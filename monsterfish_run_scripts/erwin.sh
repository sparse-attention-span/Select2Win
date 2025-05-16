cd experiments

uv run train_shapenet.py \
    --num-epochs 35000 \
    --data-path "$HOME/erwin/shapenet_car/preprocessed" \
    --use-wandb 1 \
    --batch-size 2 \
    --msa-type BallMSA